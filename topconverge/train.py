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

--crop S trains the crop model (rimnet.CropRimNet): beside the view, an S x S crop of the full-resolution picture round each current
corner goes through a second FPN, and each corner's answer is one distribution over the view's and its crop's pixels (rimnet.union_heat).
With --init pointing at a view-only model, its weights start the view half and the crop head starts at zero, so step 0 is that model.
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

from topconverge.rimnet import (DST, CropRimNet, RimNet, apply_h, crop_geometry, crop_origin, crops_forward, cut_crop, distort,
                                homography, lens_arrays, render_view, start_points, turn_sign, undistort, valid_state)

HERE = Path(__file__).resolve().parent.parent
TEX_W, TEX_H = 1920, 1080                # frames are kept at half resolution, as the network reads them


class Sample:
    def __init__(self, lab, jpg, name):
        self.name, self.view, self.jpg = name, lab['view'], jpg
        self.lab = lab
        self.gt = np.asarray(lab['rim'], np.float64)
        self.K, self.D, self.lens = lens_arrays(lab['lens'])
        self.n_gt = undistort(self.gt, self.K, self.D)
        self.tex = self.full = None

    def load(self, pin, full=False):
        """half resolution, uint8, on the CPU: the training set does not have to fit on the card beside the network (with full, the
        full-resolution picture is kept as well, H x W x 3 RGB, for the crops)"""
        im = cv2.imread(str(self.jpg))
        if im is None:
            raise OSError(f'cannot read {self.jpg}')
        rgb = np.ascontiguousarray(im[:, :, ::-1])
        if full:
            self.full = rgb
        im = cv2.resize(rgb, (TEX_W, TEX_H), interpolation=cv2.INTER_AREA)
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


FULL_WH = (TEX_W * 2.0, TEX_H * 2.0)                                            # full-resolution image size


def render_batch(tex, Hinv, lens, grid, info=False):
    """RimNet.render for a batch. tex (B, 3, TEX_H, TEX_W) 0..1, Hinv (B, 3, 3), lens (B, 18) -> (B, 4, RES, RES) (with info: also
    render_view's px, py, ok)"""
    x, px, py, ok = render_view(tex, Hinv, lens, grid, FULL_WH)
    return (x, (px, py, ok)) if info else x


def make_batch(samples, rng, device, a, W0, W1):
    """-> (frames, plane -> undistorted, lens, where the true corners appear in plane units; with --crop also the crops (B, 4, S, S, 3)
    0..1, their origins (B, 4, 2) and which samples are right lenses)"""
    texs, His, lenses, targets, crops, origins, mirror = [], [], [], [], [], [], []
    for i in rng.integers(0, len(samples), a.batch):
        s = samples[i]
        n_cur = sample_state(s, rng, a.start_frac, a.sig_max, a.start_noise)
        H, Hi = homography(n_cur)
        tgt, _ = apply_h(H, s.n_gt)
        targets.append(np.clip(tgt, W0 + 1, W1 - 1))
        His.append(Hi); lenses.append(s.lens)
        gain, off = 0.75 + 0.5 * rng.random(), (rng.random() - 0.5) * 0.15                            # colour jitter
        t = s.tex.to(device, non_blocking=True).float() / 255.0
        texs.append((t * gain + off).clamp(0, 1))
        if a.crop:
            org = crop_origin(distort(n_cur, s.K, s.D), a.crop)
            c = np.stack([cut_crop(s.full, int(x0), int(y0), a.crop) for x0, y0 in org])
            c = torch.from_numpy(c).to(device, non_blocking=True).float() / 255.0
            crops.append((c * gain + off).clamp(0, 1)); origins.append(org); mirror.append(s.view == 'R')
    f32 = [torch.tensor(np.array(v), device=device, dtype=torch.float32) for v in (His, lenses, targets)]
    if not a.crop:
        return (torch.stack(texs), *f32)
    return (torch.stack(texs), *f32, torch.stack(crops), torch.tensor(np.array(origins), device=device), torch.tensor(mirror, device=device))


def gpu_temp(index):
    try:
        return int(subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], timeout=20).split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def save(net, out, a, step):
    tmp = out / 'model.pt.tmp'
    torch.save(net.state_dict(), tmp)
    os.replace(tmp, out / 'model.pt')
    cfg = {'weights': 'model.pt', 'arch': 'smp.FPN', 'encoder': a.encoder, 'in_channels': 4, 'classes': 4,
           'margin': a.margin, 'res': a.res, 'tex': [TEX_W, TEX_H], 'alpha': a.alpha, 'start': 'sides',
           'train_step': step, 'splits': a.splits}
    if a.crop:
        cfg.update(arch='CropRimNet', crop=a.crop, crop_encoder=a.crop_encoder)
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))


def evaluate(out, samples, device, n, log):
    """Auto from the start, through what was just saved, so a run is scored the way server.py and topconverge.evaluate will use it"""
    model = RimNet(out, device)
    rows = []
    for s in samples:
        tex = model.texture(s.jpg)
        rim, steps, ok = model.iterate(tex, s.lab['lens'], start_points(s.view, s.lab['width'], s.lab['height']), n, view=s.view)
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
    ap.add_argument('--crop', type=int, default=0, help='side of the full-resolution crops round each corner, px (0: the view only)')
    ap.add_argument('--crop-encoder', default='resnet34')
    ap.add_argument('--crop-lr-mult', type=float, default=1.0, help='learning rate of the crop branch over --lr (it starts at zero and learns slowly)')
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
        s.load(cuda, full=bool(a.crop))

    W0, W1 = -a.margin, 100.0 + a.margin
    u = torch.linspace(W0, W1, a.res, device=device)
    v = torch.linspace(W1, W0, a.res, device=device)                            # top row = far side
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    if a.crop:
        net = CropRimNet(a.encoder, a.crop_encoder, imagenet=True)
        net.zero_crop_head()
    else:
        net = smp.FPN(encoder_name=a.encoder, encoder_weights=None if a.init else 'imagenet', in_channels=4, classes=4)
    if a.init:
        init = json.loads(Path(a.init, 'config.json').read_text())
        sd = {k: p.float() if p.is_floating_point() else p for k, p in torch.load(Path(a.init, init['weights']), map_location='cpu').items()}
        if a.crop and init.get('crop'):
            net.load_state_dict(sd)
        elif a.crop:
            net.top.load_state_dict(sd)                                         # a view-only model starts the view half
        else:
            net.load_state_dict(sd)
    net = net.to(device).train()
    if a.crop:
        groups = [{'params': net.top.parameters(), 'lr': a.lr}, {'params': net.crop.parameters(), 'lr': a.lr * a.crop_lr_mult}]
    else:
        groups = [{'params': net.parameters(), 'lr': a.lr}]
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=a.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'recipe.json').write_text(json.dumps(vars(a), indent=2))
    sg = a.hm_sigma * (W1 - W0) / (a.res - 1)                                   # the heat-map target's width in plane units
    t0, t_chk, ema = time.time(), 0.0, None
    with open(out / 'history.jsonl', 'w') as hist:
        for it in range(1, a.steps + 1):
            if cuda and a.pause_temp and torch.version.hip is None and time.time() - t_chk > 3:     # the temperature guard reads nvidia-smi
                t_chk = time.time()
                t = gpu_temp(a.temp_gpu)
                if t is not None and t >= a.pause_temp:
                    log(f'[gpu] {t} C: paused until {a.resume_temp} C')
                    while (gpu_temp(a.temp_gpu) or 0) > a.resume_temp:
                        time.sleep(10)
            batch = make_batch(tr, rng, device, a, W0, W1)
            tex, Hi, lens, y = batch[:4]
            if a.crop:
                crops, org, mirror = batch[4:]
                x, info = render_batch(tex, Hi, lens, (uu, vv), info=True)
                del tex
                cu, cv, cvalid, cla, inimg = crop_geometry(org, a.crop, Hi, lens, FULL_WH, W0, W1)
                lg, pu, pv, la = crops_forward(net, x, crops, inimg, mirror, org, (cu, cv, cvalid, cla), info, (uu, vv), W0, W1, a.crop)
            else:
                x = render_batch(tex, Hi, lens, (uu, vv))
                del tex
                with torch.autocast('cuda', dtype=torch.float16, enabled=cuda):
                    logits = net(x)
                lg = logits.float().view(logits.shape[0], 4, -1)
                pu, pv, la = uu.reshape(1, 1, -1), vv.reshape(1, 1, -1), 0.0
            p = torch.softmax(lg, -1)
            pred = torch.stack([(p * pu).sum(-1), (p * pv).sum(-1)], -1)
            d2 = (pu - y[..., 0:1]) ** 2 + (pv - y[..., 1:2]) ** 2
            tgt = torch.softmax(-d2 / (2 * sg * sg) + la, -1)                   # a Gaussian density over the plane, as mass per sample point
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
