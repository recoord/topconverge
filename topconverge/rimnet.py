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
import math
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
    src = np.asarray(px, np.float64).reshape(-1, 1, 2)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-9)
    if hasattr(cv2, 'undistortPointsIter'):                     # the plain undistortPoints of these builds takes no criteria: 5 iterations, off by up to 16 px in the corners
        return cv2.undistortPointsIter(src, K, D, None, None, crit).reshape(-1, 2)
    return cv2.undistortPoints(src, K, D, None, None, None, criteria=crit).reshape(-1, 2)


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


# ---------------------------------------------------------------- rendering (batched)
def _dist_coeffs(lens):
    d = lens[:, 4:]
    return [d[:, i] for i in range(12)]


def render_view(tex, Hinv, lens, grid, full_wh):
    """the top view of a batch. tex (B, 3, h, w) 0..1, Hinv (B, 3, 3) plane -> undistorted, lens (B, 18), full_wh the full-resolution
    picture size -> (x (B, 4, R, R): normalised RGB, 0 where no camera pixel lands, and the mask; px, py (B, R, R): where each view
    pixel lies in the full-resolution picture; ok (B, R, R): it lands on the picture)"""
    uu, vv = grid
    B = tex.shape[0]
    GH, GW = uu.shape
    P = torch.stack([uu, vv, torch.ones_like(uu)], -1).view(1, -1, 3).expand(B, -1, -1)
    q = P @ Hinv.transpose(1, 2)
    w = q[..., 2]
    ok = w > 1e-9
    x, y = q[..., 0] / w.clamp(min=1e-9), q[..., 1] / w.clamp(min=1e-9)
    r2 = x * x + y * y
    ok &= r2 < RMAX * RMAX
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = [c[:, None] for c in _dist_coeffs(lens)]
    rad = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2 * r2
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r2 * r2
    px = xd * lens[:, 0:1] + lens[:, 2:3]
    py = yd * lens[:, 1:2] + lens[:, 3:4]
    Wf, Hf = float(full_wh[0]), float(full_wh[1])
    ok &= (px >= 0) & (py >= 0) & (px <= Wf) & (py <= Hf)
    g = torch.stack([px / Wf * 2 - 1, py / Hf * 2 - 1], -1).view(B, GH, GW, 2)
    g = torch.where(ok.view(B, GH, GW, 1), g, torch.full_like(g, -2.0))
    rgb = F.grid_sample(tex, g, mode='bilinear', padding_mode='zeros', align_corners=False)
    m = ok.view(B, 1, GH, GW).to(rgb.dtype)
    rgb = (rgb - MEAN.to(rgb)) / STD.to(rgb) * m
    return torch.cat([rgb, m], 1), px.view(B, GH, GW), py.view(B, GH, GW), ok.view(B, GH, GW)


# ---------------------------------------------------------------- corner crops
# A crop model sees, beside the top view, a size x size crop of the FULL-resolution picture round each current corner's image position.
# Each crop gets its own heat map (a second FPN; corner k reads output channel k of crop k). The answer stays one distribution over the
# plane: every view pixel and every crop pixel is a sample point at its plane position, weighted by the plane area it covers (log area
# added to its logit), and inside the part of the picture a crop covers the crop's pixels replace the view's - their logit is the view's
# heat map sampled there PLUS the crop's own. So a crop refines the view's answer where it sees the corner; it cannot claim a corner the
# view puts somewhere else without outweighing it. With the crop head's last layer zero, the model is exactly the view-only model.
def crop_origin(cur_px, size):
    """current corners (4 x 2 image px) -> top-left pixel of each crop (4 x 2 int); crop pixel j covers [x0 + j, x0 + j + 1)"""
    return np.floor(np.asarray(cur_px, np.float64) - size / 2.0 + 0.5).astype(np.int64)


def cut_crop(img, x0, y0, size):
    """size x size x 3 of an H x W x 3 uint8 picture from (x0, y0), zero outside the picture"""
    H, W = img.shape[:2]
    out = np.zeros((size, size, 3), np.uint8)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, W), min(y0 + size, H)
    if xa < xb and ya < yb:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return out


def undistort_t(px, py, lens, iters=40):
    """OpenCV's iterative undistortion in torch (rational, tangential, thin prism). px, py (B, ...) image px, lens (B, 18) ->
    (x, y undistorted normalised, ok: the iteration converged to within 0.5 px and stayed inside RMAX)"""
    sh = (-1,) + (1,) * (px.dim() - 1)
    fx, fy, cx, cy = [lens[:, i].view(sh) for i in range(4)]
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = [c.view(sh) for c in _dist_coeffs(lens)]
    xd, yd = (px - cx) / fx, (py - cy) / fy
    x, y = xd.clone(), yd.clone()
    for _ in range(iters):
        r2 = x * x + y * y
        icd = (1 + ((k6 * r2 + k5) * r2 + k4) * r2) / (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2 * r2
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r2 * r2
        x, y = ((xd - dx) * icd).clamp(-50, 50), ((yd - dy) * icd).clamp(-50, 50)
    r2 = x * x + y * y
    rad = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
    ex = (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2 * r2 - xd) * fx
    ey = (y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r2 * r2 - yd) * fy
    ok = (ex * ex + ey * ey < 0.25) & (r2 < RMAX * RMAX)
    return x, y, ok


def crop_geometry(origins, size, Hinv, lens, full_wh, W0, W1):
    """origins (B, 4, 2) crop top-left px -> (u, v (B, 4, S, S) plane position of every crop pixel, valid (B, 4, S, S), log of the
    plane area each covers (B, 4, S, S), inimg (B, 4, S, S): the pixel is on the picture)"""
    dev = Hinv.device
    j = torch.arange(size, device=dev, dtype=torch.float32) + 0.5
    px = origins[..., 0, None, None].float() + j.view(1, 1, 1, -1)
    py = origins[..., 1, None, None].float() + j.view(1, 1, -1, 1)
    px, py = px.expand(-1, -1, size, size), py.expand(-1, -1, size, size)
    Wf, Hf = float(full_wh[0]), float(full_wh[1])
    inimg = (px >= 0) & (py >= 0) & (px <= Wf) & (py <= Hf)
    x, y, ok = undistort_t(px, py, lens)
    H = torch.linalg.inv(Hinv.double()).float()                                 # undistorted -> plane, with Hinv's sign: in front <=> c > 0
    h = [H[:, a, b].view(-1, 1, 1, 1) for a in range(3) for b in range(3)]
    a_ = h[0] * x + h[1] * y + h[2]
    b_ = h[3] * x + h[4] * y + h[5]
    c_ = h[6] * x + h[7] * y + h[8]
    ok &= c_ > 1e-9
    u, v = a_ / c_.clamp(min=1e-9), b_ / c_.clamp(min=1e-9)
    ok &= inimg & (u >= W0) & (u <= W1) & (v >= W0) & (v <= W1)
    # plane area of each pixel from forward differences; a pixel whose neighbours are not valid is left out
    def fd(t, dim):
        d = t.diff(dim=dim)
        return torch.cat([d, d.narrow(dim, d.shape[dim] - 1, 1)], dim)
    def fdm(m, dim):
        mm = m[..., :, 1:] & m[..., :, :-1] if dim == -1 else m[..., 1:, :] & m[..., :-1, :]
        return torch.cat([mm, mm.narrow(dim, mm.shape[dim] - 1, 1)], dim)
    du_j, dv_j, du_i, dv_i = fd(u, -1), fd(v, -1), fd(u, -2), fd(v, -2)
    valid = ok & fdm(ok, -1) & fdm(ok, -2)
    area = (du_j * dv_i - dv_j * du_i).abs()
    return u, v, valid, torch.log(area.clamp(min=1e-12)), inimg


def crop_input(crops, inimg, mirror):
    """crops (B, 4, S, S, 3) 0..1 on the device, inimg (B, 4, S, S), mirror (B,) bool: the right lens' crops flipped left-right like its
    view -> (B * 4, 4, S, S) normalised RGB + mask"""
    B, K, S = crops.shape[:3]
    x = crops.permute(0, 1, 4, 2, 3).reshape(B * K, 3, S, S)
    m = inimg.reshape(B * K, 1, S, S).to(x.dtype)
    x = torch.cat([(x - MEAN.to(x)) / STD.to(x) * m, m], 1)
    fl = mirror.repeat_interleave(K)
    return torch.where(fl.view(-1, 1, 1, 1), x.flip(-1), x), fl


NEG = -1e4                                       # logit of a sample point that is left out (finite, so 0 * log p stays 0)


def union_heat(top_logits, crop_logits, flipped, grid, W0, W1, top_px, top_py, top_ok, origins, size, cu, cv, cvalid, clogarea):
    """one distribution per corner over the view's pixels and its crop's pixels -> (logits (B, 4, N), u, v (B, 4, N), log area (B, 4, N))"""
    uu, vv = grid
    B, K, R, _ = top_logits.shape
    S = size
    la_top = 2.0 * math.log((W1 - W0) / (R - 1))
    ox, oy = origins[..., 0].float().view(B, K, 1, 1), origins[..., 1].float().view(B, K, 1, 1)
    claimed = top_ok[:, None] & (top_px[:, None] >= ox) & (top_px[:, None] < ox + S) & (top_py[:, None] >= oy) & (top_py[:, None] < oy + S)
    tl = top_logits.float()
    top_l = (tl + la_top).masked_fill(claimed, NEG).view(B, K, -1)
    top_la = torch.full_like(tl, la_top).masked_fill(claimed, NEG).view(B, K, -1)
    g = torch.stack([(cu - W0) / (W1 - W0) * 2 - 1, (W1 - cv) / (W1 - W0) * 2 - 1], -1).view(B * K, S, S, 2)
    samp = F.grid_sample(tl.reshape(B * K, 1, R, R), g, mode='bilinear', padding_mode='border', align_corners=True).view(B, K, S, S)
    cl = crop_logits.float()                                                    # (B * 4, 4, S, S)
    cl = torch.where(flipped.view(-1, 1, 1, 1), cl.flip(-1), cl)
    diag = cl.view(B, K, K, S, S)[:, torch.arange(K), torch.arange(K)]          # crop k, channel k
    crop_l = (samp + diag + clogarea).masked_fill(~cvalid, NEG).view(B, K, -1)
    crop_la = clogarea.masked_fill(~cvalid, NEG).view(B, K, -1)
    logits = torch.cat([top_l, crop_l], -1)
    pu = torch.cat([uu.reshape(1, 1, -1).expand(B, K, -1), cu.reshape(B, K, -1)], -1)
    pv = torch.cat([vv.reshape(1, 1, -1).expand(B, K, -1), cv.reshape(B, K, -1)], -1)
    return logits, pu, pv, torch.cat([top_la, crop_la], -1)


class CropRimNet(torch.nn.Module):
    """the view's FPN (top) and the crops' FPN (crop), saved as one state dict"""

    def __init__(self, encoder, crop_encoder, imagenet=False):
        super().__init__()
        import segmentation_models_pytorch as smp
        enc_w = 'imagenet' if imagenet else None
        self.top = smp.FPN(encoder_name=encoder, encoder_weights=enc_w, in_channels=4, classes=4)
        self.crop = smp.FPN(encoder_name=crop_encoder, encoder_weights=enc_w, in_channels=4, classes=4)

    def zero_crop_head(self):
        conv = self.crop.segmentation_head[0]
        torch.nn.init.zeros_(conv.weight)
        torch.nn.init.zeros_(conv.bias)


def crops_forward(net, x, crops, inimg, mirror, origins, cgeom, render_info, grid, W0, W1, size):
    """the crop model's forward pass -> union_heat's (logits, u, v, log area)"""
    top_px, top_py, top_ok = render_info
    cu, cv, cvalid, clogarea = cgeom
    xc, fl = crop_input(crops, inimg, mirror)
    with torch.autocast('cuda', dtype=torch.float16, enabled=x.is_cuda):
        tl = net.top(x)
        cl = net.crop(xc.to(x.dtype))
    return union_heat(tl, cl, fl, grid, W0, W1, top_px, top_py, top_ok, origins, size, cu, cv, cvalid, clogarea)


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
        self.crop = int(self.cfg.get('crop', 0))                                # 0: the view only; else the side of the corner crops, px
        if self.crop:
            net = CropRimNet(self.cfg['encoder'], self.cfg.get('crop_encoder', self.cfg['encoder']))
        else:
            net = smp.FPN(encoder_name=self.cfg['encoder'], encoder_weights=None, in_channels=self.cfg['in_channels'], classes=self.cfg['classes'])
        sd = torch.load(weights_dir / self.cfg['weights'], map_location='cpu')
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        self.net = net.to(self.device).eval()
        u = torch.linspace(self.W0, self.W1, self.res, device=self.device)
        v = torch.linspace(self.W1, self.W0, self.res, device=self.device)     # top row = far side
        vv, uu = torch.meshgrid(v, u, indexing='ij')
        self.grid = (uu, vv)

    def texture(self, path):
        """the frame as the network reads it: half resolution, RGB 0..1, (3, H, W) on the device; a crop model also keeps the
        full-resolution picture (uint8 RGB, on the CPU) and gets (half, full)"""
        im = cv2.imread(str(path))
        if im is None:
            raise OSError(f'cannot read {path}')
        self.full_w, self.full_h = im.shape[1], im.shape[0]
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        half = cv2.resize(rgb, (self.tex_w, self.tex_h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(half)).permute(2, 0, 1).to(self.device).float() / 255.0
        return (t, rgb) if self.crop else t

    def render(self, tex, Hinv, lens18, info=False):
        """the top view: (1, 4, RES, RES) normalised RGB (0 where no camera pixel lands) and the mask (with info: also render_view's
        px, py, ok)"""
        Hi = torch.tensor(Hinv, device=self.device, dtype=torch.float32)[None]
        lens = torch.tensor(lens18, device=self.device, dtype=torch.float32)[None]
        x, px, py, ok = render_view(tex[None], Hi, lens, self.grid, (self.tex_w * 2.0, self.tex_h * 2.0))
        return (x, (px, py, ok)) if info else x

    @torch.no_grad()
    def predict(self, x):
        """network input -> where the four true corners appear in the view (4 x 2 plane units)"""
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.device.type == 'cuda'):
            G = self.net(x)
        uu, vv = self.grid
        p = torch.softmax(G.float().view(1, 4, -1), -1)
        pred = torch.stack([(p * uu.reshape(1, 1, -1)).sum(-1), (p * vv.reshape(1, 1, -1)).sum(-1)], -1)
        return pred[0].cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def predict_crops(self, x, info, full, cur_px, Hinv, lens18, view):
        """the crop model: view + four full-resolution crops round the current corners -> 4 x 2 plane units"""
        S = self.crop
        org = crop_origin(cur_px, S)
        crops = np.stack([cut_crop(full, int(a), int(b), S) for a, b in org])
        dev = self.device
        crops = torch.from_numpy(crops).to(dev).float()[None] / 255.0
        org_t = torch.from_numpy(org).to(dev)[None]
        Hi = torch.tensor(Hinv, device=dev, dtype=torch.float32)[None]
        lens = torch.tensor(lens18, device=dev, dtype=torch.float32)[None]
        cu, cv, cvalid, cla, inimg = crop_geometry(org_t, S, Hi, lens, (self.tex_w * 2.0, self.tex_h * 2.0), self.W0, self.W1)
        mirror = torch.tensor([view == 'R'], device=dev)
        logits, pu, pv, _ = crops_forward(self.net, x, crops, inimg, mirror, org_t, (cu, cv, cvalid, cla), info, self.grid, self.W0, self.W1, S)
        p = torch.softmax(logits, -1)
        pred = torch.stack([(p * pu).sum(-1), (p * pv).sum(-1)], -1)
        R2 = self.res * self.res
        self.last_crops = {'crops': (crops[0] * 255).round().byte().cpu().numpy(), 'origins': org, 'inimg': inimg[0].cpu().numpy(),
                           'share': p[0, :, R2:].sum(-1).cpu().numpy(), 'mirror': view == 'R', 'cur_px': np.asarray(cur_px, np.float64)}
        return pred[0].cpu().numpy().astype(np.float64)

    def step(self, tex, lens, cur_px, alpha=None, view=None):
        """one step from the current rim (4 x 2 image px) -> (next rim, how far each corner moved in px, largest move in plane units,
        the network input, the predicted corners in plane units)"""
        alpha = self.alpha if alpha is None else alpha
        K, D, lens18 = lens_arrays(lens)
        cur_px = np.asarray(cur_px, np.float64)
        n_cur = undistort(cur_px, K, D)
        s0 = turn_sign(n_cur)
        H, Hi = homography(n_cur)
        if self.crop:
            tex, full = tex
            x, info = self.render(tex, Hi, lens18, info=True)
            if view is None:                                                    # the turn of the rim tells the lens: L turns one way, R the other
                view = 'L' if s0 == turn_sign(undistort(start_points('L', self.tex_w * 2.0, self.tex_h * 2.0), K, D)) else 'R'
            pred = self.predict_crops(x, info, full, cur_px, Hi, lens18, view)
            n_pred, w_pred = apply_h(Hi, pred)
            self.last_crops['pred_px'] = np.where(w_pred[:, None] > 0, distort(n_pred, K, D), np.nan)
        else:
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

    def iterate(self, tex, lens, start, n=60, alpha=None, view=None):
        """steps from start until converged or n steps -> (rim, steps taken, converged)"""
        cur = np.asarray(start, np.float64)
        for i in range(n):
            cur, mv, mu, _, _ = self.step(tex, lens, cur, alpha, view)
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
        if self.crop and getattr(self, 'last_crops', None) is not None:
            img = np.concatenate([img, self.crops_image(self.res)], 1)
        return img

    def crops_image(self, side):
        """the four crops of the last step as the network saw them (a right lens' mirrored), in a 2 x 2 grid of side x side px (BGR):
        checkerboard off the picture, the current corner (green cross, the crop's middle), the predicted corner (magenta circle) and the
        share of the corner's answer that lies on the crop's pixels"""
        L = self.last_crops
        S, h = self.crop, side // 2
        out = np.zeros((side, side, 3), np.uint8)
        for k in range(4):
            c = L['crops'][k][:, :, ::-1].copy()
            yy, xx = np.mgrid[:S, :S]
            chk = np.where(((xx // 32) + (yy // 32)) % 2 == 0, 90, 60).astype(np.uint8)
            off = ~L['inimg'][k]
            c[off] = np.stack([chk] * 3, -1)[off]
            def at(pt):
                q = np.asarray(pt, np.float64) - L['origins'][k]
                if L['mirror']:
                    q[0] = S - q[0]
                return q
            if L['mirror']:
                c = np.ascontiguousarray(c[:, ::-1])
            c = cv2.resize(c, (h, h), interpolation=cv2.INTER_AREA) if h != S else c
            f = h / S
            q = at(L['cur_px'][k]) * f
            cv2.drawMarker(c, (int(round(q[0])), int(round(q[1]))), (80, 220, 80), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
            pp = L.get('pred_px')
            if pp is not None and np.all(np.isfinite(pp[k])):
                q = at(pp[k]) * f
                if -20 < q[0] < h + 20 and -20 < q[1] < h + 20:
                    cv2.circle(c, (int(round(q[0])), int(round(q[1]))), 9, (255, 0, 255), 2, cv2.LINE_AA)
            txt = f'{k + 1}  crop {L["share"][k]:.0%}'
            cv2.rectangle(c, (0, 0), (8 + 13 * len(txt), 28), (0, 0, 0), -1)
            cv2.putText(c, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(c, (0, 0), (h - 1, h - 1), (120, 120, 120), 1)
            r, col = divmod(k, 2)
            out[r * h:(r + 1) * h, col * h:(col + 1) * h] = c
        return out
