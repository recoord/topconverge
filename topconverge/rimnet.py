"""rimnet - the rim's four corners found by iterating on the top view.

The rim is four points in the picture: the corners of the half pitch the camera looks at, in the order
  1 frontline / centerline (the near end of the halfway line), 2 frontline / goalline, 3 goalline / sideline, 4 sideline / centerline.
Instead of finding them in the 4K picture, the picture is rendered from above through the homography of the CURRENT estimate
(the four points at the corners of a 100 x 100 square, the frontline at the bottom) over the plane window -M..100+M, and a
network (FPN, ResNet-34 encoder, 4 input channels: RGB + a mask of where camera pixels land) gives four heat maps whose
soft-argmax is where the true corners appear in that view. Mapped back through the homography and the lens, that is the next
estimate; repeated, it converges, and at the fixed point the rim is the square itself.

All homographies live in UNDISTORTED normalised camera coordinates (the lens model is OpenCV's 14 coefficients), where the
straight painted lines of the pitch are straight. A right lens' rim turns the other way, so its view comes out mirrored and
point 1 is always at the bottom right of the square for the network.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SIDE = 100.0
RMAX = 5.0                     # undistorted radius beyond which the lens model is too flat to trust
DST = np.float32([[SIDE, 0], [0, 0], [0, SIDE], [SIDE, SIDE]])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TOL_PX, TOL_UNITS = 0.5, 0.1   # converged: every corner moved under half a pixel, or under 0.1 plane unit


# ---------------------------------------------------------------- lens and geometry
def lens_arrays(lens):
    """{'fx', 'fy', 'cx', 'cy', 'dist': [14]} -> (K 3x3, D 14, flat 18-vector for the renderer)"""
    K = np.array([[lens['fx'], 0, lens['cx']], [0, lens['fy'], lens['cy']], [0, 0, 1]], np.float64)
    D = np.array((list(lens['dist']) + [0.0] * 14)[:14], np.float64)
    return K, D, np.concatenate([[lens['fx'], lens['fy'], lens['cx'], lens['cy']], D])


def undistort(px, K, D):
    return cv2.undistortPointsIter(np.asarray(px, np.float64).reshape(-1, 1, 2), K, D, None, None,
                                   (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-9)).reshape(-1, 2)


def distort(n, K, D):
    n = np.asarray(n, np.float64).reshape(-1, 2)
    pts = np.concatenate([n, np.ones((len(n), 1))], 1)
    return cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)


def homography(n):
    """undistorted corners (4 x 2) -> (H: undistorted -> plane, Hinv: plane -> undistorted, signed so the square's middle is in
    front of the camera)"""
    H = cv2.getPerspectiveTransform(np.float32(n), DST).astype(np.float64)
    Hi = np.linalg.inv(H)
    if (Hi @ np.array([SIDE / 2, SIDE / 2, 1.0]))[2] < 0:
        Hi = -Hi
    return H, Hi


def apply_h(M, p):
    p = np.asarray(p, np.float64).reshape(-1, 2)
    q = np.concatenate([p, np.ones((len(p), 1))], 1) @ M.T
    return q[:, :2] / q[:, 2:3], q[:, 2]


def turn_sign(n):
    """+1 / -1 for a convex quadrilateral turning one way or the other, 0 when it is not convex"""
    s = 0
    for k in range(4):
        a, b, c = n[k], n[(k + 1) % 4], n[(k + 2) % 4]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        t = 1 if cr > 1e-12 else -1 if cr < -1e-12 else 0
        if not t or (s and t != s):
            return 0
        s = t
    return s


def valid_state(n, s0):
    return not np.any(np.hypot(n[:, 0], n[:, 1]) > RMAX) and turn_sign(n) == s0


def start_points(view, w=3840.0, h=2160.0):
    """where an annotation starts: the middle of each side of the picture (1 bottom, 2 left, 3 top, 4 right), a right lens mirrored"""
    c = np.array([[w / 2, h], [0, h / 2], [w / 2, 0], [w, h / 2]], np.float64)
    if view == 'R':
        c[:, 0] = w - c[:, 0]
    return c


# ---------------------------------------------------------------- the model
class RimNet:
    """the network, its view window and the step. weights_dir holds config.json and the weights it names."""

    def __init__(self, weights_dir, device='auto'):
        import segmentation_models_pytorch as smp
        weights_dir = Path(weights_dir)
        self.cfg = json.loads((weights_dir / 'config.json').read_text())
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.W0, self.W1 = -float(self.cfg['margin']), SIDE + float(self.cfg['margin'])
        self.res = int(self.cfg['res'])
        self.tex_w, self.tex_h = self.cfg['tex']
        self.alpha = float(self.cfg.get('alpha', 1.0))
        net = smp.FPN(encoder_name=self.cfg['encoder'], encoder_weights=None, in_channels=self.cfg['in_channels'], classes=self.cfg['classes'])
        sd = torch.load(weights_dir / self.cfg['weights'], map_location='cpu')
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        self.net = net.to(self.device).eval()
        u = torch.linspace(self.W0, self.W1, self.res, device=self.device)
        v = torch.linspace(self.W1, self.W0, self.res, device=self.device)     # top row = far side
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        self.grid = (uu, vv)

    def texture(self, path):
        """the frame as the network reads it: half resolution, RGB 0..1, (3, H, W) on the device"""
        im = cv2.imread(str(path))
        if im is None:
            raise OSError(f'cannot read {path}')
        self.full_w, self.full_h = im.shape[1], im.shape[0]
        im = cv2.resize(im[:, :, ::-1], (self.tex_w, self.tex_h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).to(self.device)
        return t.float() / 255.0

    def render(self, tex, Hinv, lens18):
        """the top view: (1, 4, RES, RES) normalised RGB (0 where no camera pixel lands) and the mask"""
        uu, vv = self.grid
        GH, GW = uu.shape
        Hi = torch.tensor(Hinv, device=self.device, dtype=torch.float32)
        lens = torch.tensor(lens18, device=self.device, dtype=torch.float32)
        P = torch.stack([uu, vv, torch.ones_like(uu)], -1).view(-1, 3)
        q = P @ Hi.T
        w = q[:, 2]
        ok = w > 1e-9
        x, y = q[:, 0] / w.clamp(min=1e-9), q[:, 1] / w.clamp(min=1e-9)
        r2 = x * x + y * y
        ok &= r2 < RMAX * RMAX
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = [lens[4 + i] for i in range(12)]
        rad = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2 * r2
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r2 * r2
        px, py = xd * lens[0] + lens[2], yd * lens[1] + lens[3]
        Wf, Hf = self.tex_w * 2.0, self.tex_h * 2.0                                # full-resolution image size
        ok &= (px >= 0) & (py >= 0) & (px <= Wf) & (py <= Hf)
        g = torch.stack([px / Wf * 2 - 1, py / Hf * 2 - 1], -1).view(1, GH, GW, 2)
        g = torch.where(ok.view(1, GH, GW, 1), g, torch.full_like(g, -2.0))
        rgb = F.grid_sample(tex[None], g, mode='bilinear', padding_mode='zeros', align_corners=False)
        m = ok.view(1, 1, GH, GW).float()
        rgb = (rgb - MEAN.to(rgb)) / STD.to(rgb) * m
        return torch.cat([rgb, m], 1)

    @torch.no_grad()
    def predict(self, x):
        """network input -> where the four true corners appear in the view (4 x 2 plane units)"""
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
            G = self.net(x)
        uu, vv = self.grid
        p = torch.softmax(G.float().view(1, 4, -1), -1)
        pred = torch.stack([(p * uu.reshape(1, 1, -1)).sum(-1), (p * vv.reshape(1, 1, -1)).sum(-1)], -1)
        return pred[0].cpu().numpy().astype(np.float64)

    def step(self, tex, lens, cur_px, alpha=None):
        """one step from the current rim (4 x 2 image px) -> (next rim, how far each corner moved in px, largest move in plane units,
        the network input, the predicted corners in plane units)"""
        alpha = self.alpha if alpha is None else alpha
        K, D, lens18 = lens_arrays(lens)
        cur_px = np.asarray(cur_px, np.float64)
        n_cur = undistort(cur_px, K, D)
        s0 = turn_sign(n_cur)
        H, Hi = homography(n_cur)
        x = self.render(tex, Hi, lens18)
        pred = self.predict(x)
        # map back; if the full step would leave the valid region (behind the camera, beyond the lens' range, folded), shorten it
        for f in (1.0, 0.5, 0.25, 0.125, 0.0625):
            f = f * alpha
            c = DST + f * (pred - DST)
            n, w = apply_h(Hi, c)
            if np.all(w > 0) and valid_state(n, s0):
                nxt = distort(n, K, D)
                return nxt, np.hypot(*(nxt - cur_px).T), float(f * np.hypot(*(pred - DST).T).max()), x, pred
        return cur_px.copy(), np.zeros(4), 0.0, x, pred

    def iterate(self, tex, lens, start, n=60, alpha=None):
        """steps from start until converged or n steps -> (rim, steps taken, converged)"""
        cur = np.asarray(start, np.float64)
        for i in range(n):
            cur, mv, mu, _, _ = self.step(tex, lens, cur, alpha)
            if mv.max() < TOL_PX or mu < TOL_UNITS:
                return cur, i + 1, True
        return cur, n, False

    def to_view_px(self, p):
        """plane units -> pixels of the rendered view (x right, y down)"""
        p = np.asarray(p, np.float64)
        s = (self.res - 1) / (self.W1 - self.W0)
        return np.stack([(p[..., 0] - self.W0) * s, (self.W1 - p[..., 1]) * s], -1)

    def view_image(self, x, pred=None):
        """the network input as a picture (BGR uint8): RGB where camera pixels land, a checkerboard where none do, the current
        square (green) and, if given, the predicted corners (numbered, magenta)"""
        rgb = (x[0, :3] * STD.to(x)[0] + MEAN.to(x)[0]).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        m = x[0, 3].cpu().numpy() > 0.5
        img = np.ascontiguousarray((rgb[:, :, ::-1] * 255).astype(np.uint8))
        yy, xx = np.mgrid[:self.res, :self.res]
        chk = np.where(((xx // 32) + (yy // 32)) % 2 == 0, 90, 60).astype(np.uint8)
        img[~m] = np.stack([chk] * 3, -1)[~m]
        sq = self.to_view_px(DST).round().astype(np.int32)
        cv2.polylines(img, [sq.reshape(-1, 1, 2)], True, (80, 220, 80), 2, cv2.LINE_AA)
        if pred is not None:
            for k, (a, b) in enumerate(self.to_view_px(pred)):
                c = (int(round(a)), int(round(b)))
                cv2.circle(img, c, 9, (255, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, str(k + 1), (c[0] + 11, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
        return img
