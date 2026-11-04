"""train - train the rim network on the dataset.

    python -m topconverge.train --out runs/mine [--steps 16000] [--batch 3] [--splits train,after] [--quality 1,2,3]
    python -m topconverge.train --out runs/mine --init weights        # continue from existing weights

A sample is a starred rim with the current estimate disturbed from the truth: in the plane (each corner moved by a random amount, 0.3 ..
--sig-max units) or on the picture (somewhere between the start and the truth, plus --start-noise px per corner). The picture is rendered
through the estimate's homography, and the network learns to put the true corners where they appear in that view: smooth L1 of the
heat maps' soft-argmax in plane units, plus the cross-entropy of each heat map against a Gaussian at the true corner, which keeps the
soft-argmax from being dragged by probability spread over the view (outside the picture especially).

Frames of the heldout split never train; they are scored every --eval-every steps by Auto from the start (eval-<step>.json). --out then
holds config.json and model.pt, the same layout as weights/, so `server.py --weights` and `topconverge.evaluate --weights` read it.
The defaults are the recipe of the shipped weights (weights/); the earlier weights/iter4-sides saw only the train split (--splits train).
"""
import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from topconverge.rimnet import (DST, MEAN, RMAX, STD, RimNet, apply_h, distort, homography, lens_arrays, start_points, turn_sign,
                                undistort, valid_state)

HERE = Path(__file__).resolve().parent.parent
TEX_W, TEX_H = 1920, 1080                # frames are kept at half resolution, as the network reads them


class Sample:
    def __init__(self, lab, jpg, name):
        self.name, self.view, self.jpg = name, lab['view'], jpg
        self.lab = lab
        self.gt = np.asarray(lab['rim'], np.float64)
        self.K, self.D, self.lens = lens_arrays(lab['lens'])
        self.n_gt = undistort(self.gt, self.K, self.D)
        self.tex = None

    def load(self, pin):
        """half resolution, uint8, on the CPU: the training set does not have to fit on the card beside the network"""
        im = cv2.imread(str(self.jpg))
        if im is None:
            raise OSError(f'cannot read {self.jpg}')
        im = cv2.resize(im[:, :, ::-1], (TEX_W, TEX_H), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)
        self.tex = t.contiguous().pin_memory() if pin else t


def load_samples(data, splits, qualities):
    out = []
    for p in sorted(Path(data, 'labels').glob('*.json')):
        lab = json.loads(p.read_text())
        if lab.get('rim') is None or lab.get('quality') not in qualities or lab.get('split') not in splits:
            continue
        out.append(Sample(lab, Path(data) / lab['image'], p.stem))
    return out


def sample_state(s, rng, start_frac, sig_max, start_noise):
    """a current estimate (image px, undistorted) for sample s: disturbed in the plane, or between the start and the truth"""
    s0 = turn_sign(s.n_gt)
    _, Hi = homography(s.n_gt)
    for _ in range(50):
        if rng.random() < start_frac:
            t = rng.random() ** 0.7
            cur = t * start_points(s.view) + (1 - t) * s.gt + rng.normal(0, start_noise, (4, 2))
        else:
            sig = math.exp(rng.uniform(math.log(0.3), math.log(sig_max)))
            c = DST + rng.normal(0, sig, (4, 2))
            if rng.random() < 0.3:
                c = c + rng.normal(0, sig, (1, 2))                              # a shift of the whole outline
            n, w = apply_h(Hi, c)
            if np.any(w <= 0):
                continue
            cur = distort(n, s.K, s.D)
        n_cur = undistort(cur, s.K, s.D)
        if valid_state(n_cur, s0):
            return n_cur
    return s.n_gt.copy()


def render_batch(tex, Hinv, lens, grid):
    """RimNet.render for a batch. tex (B, 3, TEX_H, TEX_W) 0..1, Hinv (B, 3, 3), lens (B, 18) -> (B, 4, RES, RES)"""
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
    d = lens[:, 4:]
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = [d[:, i:i + 1] for i in range(12)]
    rad = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + s1 * r2 + s2 * r2 * r2
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + s3 * r2 + s4 * r2 * r2
    px = xd * lens[:, 0:1] + lens[:, 2:3]
    py = yd * lens[:, 1:2] + lens[:, 3:4]
    Wf, Hf = TEX_W * 2.0, TEX_H * 2.0                                           # full-resolution image size
    ok &= (px >= 0) & (py >= 0) & (px <= Wf) & (py <= Hf)
    g = torch.stack([px / Wf * 2 - 1, py / Hf * 2 - 1], -1).view(B, GH, GW, 2)
    g = torch.where(ok.view(B, GH, GW, 1), g, torch.full_like(g, -2.0))
    rgb = F.grid_sample(tex, g, mode='bilinear', padding_mode='zeros', align_corners=False)
    m = ok.view(B, 1, GH, GW).to(rgb.dtype)
    rgb = (rgb - MEAN.to(rgb)) / STD.to(rgb) * m
    return torch.cat([rgb, m], 1)


def make_batch(samples, rng, device, a, W0, W1):
    """-> (frames, plane -> undistorted, lens, where the true corners appear in plane units)"""
    texs, His, lenses, targets = [], [], [], []
    for i in rng.integers(0, len(samples), a.batch):
        s = samples[i]
        H, Hi = homography(sample_state(s, rng, a.start_frac, a.sig_max, a.start_noise))
        tgt, _ = apply_h(H, s.n_gt)
        targets.append(np.clip(tgt, W0 + 1, W1 - 1))
        His.append(Hi); lenses.append(s.lens)
        t = s.tex.to(device, non_blocking=True).float() / 255.0
        texs.append((t * (0.75 + 0.5 * rng.random()) + (rng.random() - 0.5) * 0.15).clamp(0, 1))        # colour jitter
    return (torch.stack(texs), *[torch.tensor(np.array(v), device=device, dtype=torch.float32) for v in (His, lenses, targets)])


def gpu_temp(index):
    try:
        return int(subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], timeout=20).split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def save(net, out, a, step):
    tmp = out / 'model.pt.tmp'
    torch.save(net.state_dict(), tmp)
    os.replace(tmp, out / 'model.pt')
    (out / 'config.json').write_text(json.dumps({'weights': 'model.pt', 'arch': 'smp.FPN', 'encoder': a.encoder, 'in_channels': 4, 'classes': 4,
                                                 'margin': a.margin, 'res': a.res, 'tex': [TEX_W, TEX_H], 'alpha': a.alpha, 'start': 'sides',
                                                 'train_step': step, 'splits': a.splits}, indent=1))


def evaluate(out, samples, device, n, log):
    """Auto from the start, through what was just saved, so a run is scored the way server.py and topconverge.evaluate will use it"""
    model = RimNet(out, device)
    rows = []
    for s in samples:
        tex = model.texture(s.jpg)
        rim, steps, ok = model.iterate(tex, s.lab['lens'], start_points(s.view, s.lab['width'], s.lab['height']), n)
        rows.append({'id': s.name, 'err_px': np.hypot(*(rim - s.gt).T).round(2).tolist(), 'steps': steps, 'settled': ok})
    E = np.array([r['err_px'] for r in rows])
    log(f'[eval] {len(rows)} frames, Auto from the start, median corner error px ' + ' '.join(f'{k + 1}: {np.median(E[:, k]):.1f}' for k in range(4)) +
        f'; corners 2-4 within 10 px {np.mean(E[:, 1:].max(1) < 10):.0%}; settled {np.mean([r["settled"] for r in rows]):.0%}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='folder for config.json, model.pt, history.jsonl, eval-<step>.json')
    ap.add_argument('--data', default=str(HERE / 'data'))
    ap.add_argument('--splits', default='train,after', help='splits to train on; heldout is refused')
    ap.add_argument('--quality', default='1,2,3', help='states to train on (the stars)')
    ap.add_argument('--steps', type=int, default=16000)
    ap.add_argument('--batch', type=int, default=3, help='3 fits a 6 GB card')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--init', default=None, help='start from the weights in this folder (config.json + weights) instead of ImageNet')
    ap.add_argument('--margin', type=float, default=12.5, help='plane units of view round the 100 x 100 square')
    ap.add_argument('--res', type=int, default=1024, help='render size of the view (px per side)')
    ap.add_argument('--alpha', type=float, default=0.5, help='the step factor stored for Auto')
    ap.add_argument('--hm-weight', type=float, default=1.0)
    ap.add_argument('--hm-sigma', type=float, default=4.0, help='of the heat-map target, render px')
    ap.add_argument('--start-frac', type=float, default=0.3, help='share of samples between the start and the truth, the rest disturbed in the plane')
    ap.add_argument('--sig-max', type=float, default=25.0, help='largest disturbance in the plane, units')
    ap.add_argument('--start-noise', type=float, default=40.0, help='px of scatter per corner on the samples between the start and the truth')
    ap.add_argument('--eval-every', type=int, default=4000)
    ap.add_argument('--eval-steps', type=int, default=60, help='most steps of Auto when scoring')
    ap.add_argument('--eval-frames', type=int, default=0, help='score only this many held-out frames (0: all)')
    ap.add_argument('--device', default='auto', help='auto (cuda if available), cpu, cuda, cuda:1, ...')
    ap.add_argument('--pause-temp', type=int, default=80, help='pause training at this GPU temperature, C (0: never; needs nvidia-smi)')
    ap.add_argument('--resume-temp', type=int, default=72)
    ap.add_argument('--temp-gpu', type=int, default=0, help='nvidia-smi index of the GPU to watch')
    a = ap.parse_args()

    import segmentation_models_pytorch as smp
    splits = set(a.splits.split(','))
    if 'heldout' in splits:
        raise SystemExit('the heldout split is what a run is scored on; it does not train')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    cuda = device.type == 'cuda'
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed); random.seed(a.seed)
    log = lambda m: print(m, flush=True)     # noqa: E731

    qs = {int(q) if q.isdigit() else q for q in a.quality.split(',')}
    tr = load_samples(a.data, splits, qs)
    ho = load_samples(a.data, {'heldout'}, {1, 2, 3})
    if a.eval_frames:
        ho = ho[:a.eval_frames]
    if not tr:
        raise SystemExit('no frames match')
    log(f'[data] {len(tr)} training frames ({a.splits}, states {sorted(map(str, qs))}), {len(ho)} held out')
    for s in tr:
        s.load(cuda)

    W0, W1 = -a.margin, 100.0 + a.margin
    u = torch.linspace(W0, W1, a.res, device=device)
    v = torch.linspace(W1, W0, a.res, device=device)                            # top row = far side
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    net = smp.FPN(encoder_name=a.encoder, encoder_weights=None if a.init else 'imagenet', in_channels=4, classes=4)
    if a.init:
        init = json.loads(Path(a.init, 'config.json').read_text())
        sd = torch.load(Path(a.init, init['weights']), map_location='cpu')
        net.load_state_dict({k: p.float() if p.is_floating_point() else p for k, p in sd.items()})
    net = net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'recipe.json').write_text(json.dumps(vars(a), indent=2))
    sg = a.hm_sigma * (W1 - W0) / (a.res - 1)                                   # the heat-map target's width in plane units
    t0, t_chk, ema = time.time(), 0.0, None
    with open(out / 'history.jsonl', 'w') as hist:
        for it in range(1, a.steps + 1):
            if cuda and a.pause_temp and time.time() - t_chk > 3:
                t_chk = time.time()
                t = gpu_temp(a.temp_gpu)
                if t is not None and t >= a.pause_temp:
                    log(f'[gpu] {t} C: paused until {a.resume_temp} C')
                    while (gpu_temp(a.temp_gpu) or 0) > a.resume_temp:
                        time.sleep(10)
            tex, Hi, lens, y = make_batch(tr, rng, device, a, W0, W1)
            x = render_batch(tex, Hi, lens, (uu, vv))
            del tex
            with torch.autocast('cuda', dtype=torch.float16, enabled=cuda):
                logits = net(x)
            lg = logits.float().view(logits.shape[0], 4, -1)
            p = torch.softmax(lg, -1)
            pred = torch.stack([(p * uu.reshape(1, 1, -1)).sum(-1), (p * vv.reshape(1, 1, -1)).sum(-1)], -1)
            d2 = (uu.reshape(1, 1, -1) - y[..., 0:1]) ** 2 + (vv.reshape(1, 1, -1) - y[..., 1:2]) ** 2
            tgt = torch.softmax(-d2 / (2 * sg * sg), -1)
            ce = -(tgt * torch.log_softmax(lg, -1)).sum(-1).mean()
            loss = F.smooth_l1_loss(pred, y, beta=1.0) + a.hm_weight * ce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if it % 50 == 0 or it == 1:
                err = (pred - y).norm(dim=-1).mean().item()
                hist.write(json.dumps({'step': it, 'loss': round(ema, 4), 'err_units': round(err, 3), 'min': round((time.time() - t0) / 60, 1)}) + '\n'); hist.flush()
                log(f'[train] step {it} loss {ema:.4f} corner error {err:.2f} units ({(time.time() - t0) / it:.2f} s/step)')
            if it % a.eval_every == 0 or it == a.steps:
                save(net, out, a, it)
                if ho:
                    (out / f'eval-{it}.json').write_text(json.dumps(evaluate(out, ho, device, a.eval_steps, log)))
                    if cuda:
                        torch.cuda.empty_cache()
    log(f'[done] {out / "model.pt"}')


if __name__ == '__main__':
    main()
