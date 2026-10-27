"""evaluate - run Auto from the start on annotated frames and compare with the human rim.

    python -m topconverge.evaluate [--split heldout] [--quality 1,2,3] [--steps 60] [--device auto] [--json out.json]

For every frame of the split with a rim in one of the states, the rim is iterated from the start (the middle of each side of the
picture) until it settles or --steps, and the distance of each corner to the annotated one is reported in image px (medians over
frames, per corner) together with the share of frames whose four corners all lie within 10 / 25 px, and whose corners 2-4 do
(corner 1 usually lies far outside the picture, where a pixel is a long way on the ground and both the label and the network are
loose). The heldout split was never
seen in training (recordings held out by crc32 of the recording id), so it is the honest number; 'train' frames were seen.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from topconverge.rimnet import RimNet, start_points

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(HERE / 'data'))
    ap.add_argument('--weights', default=str(HERE / 'weights'))
    ap.add_argument('--split', default='heldout', help='heldout, train, after or all')
    ap.add_argument('--quality', default='1,2,3', help='states to include (comma list of 1, 2, 3, needs_work, unusable)')
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--alpha', type=float, default=None, help='step factor (default: the model config)')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--json', default=None, help='write the per-frame rows here')
    a = ap.parse_args()
    qs = {int(q) if q.isdigit() else q for q in a.quality.split(',')}
    net = RimNet(a.weights, a.device)
    rows = []
    for p in sorted(Path(a.data, 'labels').glob('*.json')):
        lab = json.loads(p.read_text())
        if lab.get('rim') is None or lab.get('quality') not in qs or (a.split != 'all' and lab.get('split') != a.split):
            continue
        t0 = time.time()
        tex = net.texture(Path(a.data) / lab['image'])
        rim, n, ok = net.iterate(tex, lab['lens'], start_points(lab['view'], lab['width'], lab['height']), a.steps, a.alpha)
        err = np.hypot(*(rim - np.asarray(lab['rim'])).T)
        rows.append({'id': p.stem, 'err_px': err.round(2).tolist(), 'steps': n, 'settled': ok, 's': round(time.time() - t0, 2)})
        print(f'{p.stem}: {n:2d} steps{" settled" if ok else ""}, corner error px {np.round(err, 1).tolist()}', flush=True)
    if not rows:
        raise SystemExit('no frames match')
    E = np.array([r['err_px'] for r in rows])
    print(f'\n{len(rows)} frames ({a.split}, states {sorted(map(str, qs))}): median corner error px '
          f'1: {np.median(E[:, 0]):.1f}  2: {np.median(E[:, 1]):.1f}  3: {np.median(E[:, 2]):.1f}  4: {np.median(E[:, 3]):.1f}; '
          f'all four within 10 px {np.mean(E.max(1) < 10):.0%}, within 25 px {np.mean(E.max(1) < 25):.0%}; '
          f'corners 2-4 within 5 px {np.mean(E[:, 1:].max(1) < 5):.0%}, 10 px {np.mean(E[:, 1:].max(1) < 10):.0%}; '
          f'settled {np.mean([r["settled"] for r in rows]):.0%}, median {np.median([r["steps"] for r in rows]):.0f} steps')
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
