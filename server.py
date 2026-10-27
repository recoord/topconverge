#!/usr/bin/env python3
"""topconverge server - browse, annotate and auto-place rims on the frames of the dataset.

    python server.py [--data data] [--weights weights] [--port 8000] [--host 127.0.0.1] [--device auto|cpu|cuda]

Open http://127.0.0.1:8000/ : the list of frames (filter by state / split), a click opens the annotator. Saving writes the
frame's label file in data/labels/ (rim + quality + an "edited" time), so changes show up in git.
"""
import argparse
import io
import json
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from topconverge.rimnet import TOL_PX, TOL_UNITS, RimNet, start_points

HERE = Path(__file__).resolve().parent
QUALITIES = ('needs_work', 'unusable', 1, 2, 3)


def create_app(data, weights, device):
    data = Path(data).resolve()
    labels, images, thumbs = data / 'labels', data / 'images', data / '.thumbs'
    app = Flask(__name__, static_folder=str(HERE / 'static'), static_url_path='/static')
    model = {}
    lock = threading.Lock()
    texcache = OrderedDict()

    def net():
        if 'net' not in model:
            t0 = time.time()
            model['net'] = RimNet(weights, device)
            print(f'[topconverge] model loaded on {model["net"].device} in {time.time() - t0:.1f} s', flush=True)
        return model['net']

    def label_path(img_id):
        p = labels / f'{img_id}.json'
        if '/' in img_id or '..' in img_id or not p.exists():
            abort(404)
        return p

    def read_label(img_id):
        return json.loads(label_path(img_id).read_text())

    def texture(img_id):
        tex = texcache.get(img_id)
        if tex is None:
            tex = net().texture(images / f'{img_id}.jpg')
            texcache[img_id] = tex
            while len(texcache) > 6:
                texcache.popitem(last=False)
        texcache.move_to_end(img_id)
        return tex

    def rim_of(body, lab):
        rim = body.get('rim')
        if rim is None:
            return start_points(lab['view'], lab['width'], lab['height'])
        rim = np.asarray(rim, np.float64)
        if rim.shape != (4, 2) or not np.all(np.isfinite(rim)):
            abort(400, 'rim must be four [x, y] points')
        return rim

    @app.get('/')
    def index():
        return send_from_directory(HERE / 'static', 'index.html')

    @app.get('/edit/<img_id>')
    def edit(img_id):
        label_path(img_id)
        return send_from_directory(HERE / 'static', 'edit.html')

    @app.get('/images/<img_id>.jpg')
    def image(img_id):
        label_path(img_id)
        return send_from_directory(images, f'{img_id}.jpg', max_age=3600)

    @app.get('/thumbs/<img_id>.jpg')
    def thumb(img_id):
        label_path(img_id)
        p = thumbs / f'{img_id}.jpg'
        if not p.exists():
            thumbs.mkdir(exist_ok=True)
            im = cv2.imread(str(images / f'{img_id}.jpg'), cv2.IMREAD_REDUCED_COLOR_4)
            im = cv2.resize(im, (480, int(480 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return send_file(p, max_age=3600)

    @app.get('/api/items')
    def items():
        out = []
        for p in sorted(labels.glob('*.json')):
            lab = json.loads(p.read_text())
            out.append({'id': p.stem, 'recording': lab.get('recording'), 'view': lab.get('view'), 'quality': lab.get('quality'),
                        'split': lab.get('split'), 'source': lab.get('source'), 'has_rim': lab.get('rim') is not None,
                        'rim': lab.get('rim'), 'width': lab.get('width'), 'height': lab.get('height'), 'edited': lab.get('edited')})
        return jsonify(out)

    @app.get('/api/label/<img_id>')
    def get_label(img_id):
        lab = read_label(img_id)
        lab['start'] = start_points(lab['view'], lab['width'], lab['height']).tolist()
        return jsonify(lab)

    @app.post('/api/label/<img_id>')
    def save_label(img_id):
        p = label_path(img_id)
        lab = json.loads(p.read_text())
        body = request.get_json(force=True)
        q = body.get('quality')
        if q not in QUALITIES and q is not None:
            abort(400, f'quality must be one of {QUALITIES}')
        rim = body.get('rim')
        if rim is not None:
            rim = rim_of(body, lab).round(3).tolist()
        lab.update(rim=rim, quality=q, edited=datetime.now(timezone.utc).isoformat(timespec='seconds'))
        if 'auto_used' in body:
            lab['auto_used'] = bool(lab.get('auto_used')) or bool(body['auto_used'])
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(lab, indent=1) + '\n')
        tmp.replace(p)
        return jsonify(lab)

    @app.post('/api/step/<img_id>')
    def step(img_id):
        """one network step from the posted rim (or the start) -> the next rim"""
        lab = read_label(img_id)
        body = request.get_json(force=True)
        cur = rim_of(body, lab)
        with lock:
            t0 = time.time()
            nxt, mv, mu, _, _ = net().step(texture(img_id), lab['lens'], cur, body.get('alpha'))
        return jsonify({'rim': nxt.round(3).tolist(), 'moved_px': mv.round(2).tolist(), 'moved_units': round(mu, 3),
                        'converged': bool(mv.max() < TOL_PX or mu < TOL_UNITS), 'ms': round((time.time() - t0) * 1000)})

    @app.post('/api/view/<img_id>')
    def view(img_id):
        """the exact network input for the posted rim as a PNG, with the current square and the corners it predicts"""
        lab = read_label(img_id)
        body = request.get_json(force=True)
        cur = rim_of(body, lab)
        with lock:
            _, _, _, x, pred = net().step(texture(img_id), lab['lens'], cur)
            img = net().view_image(x, pred)
        ok, buf = cv2.imencode('.png', img)
        return send_file(io.BytesIO(buf.tobytes()), mimetype='image/png')

    @app.get('/api/status')
    def status():
        return jsonify({'loaded': 'net' in model, 'device': str(model['net'].device) if 'net' in model else None,
                        'weights': json.loads((Path(weights) / 'config.json').read_text())})

    return app, net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(HERE / 'data'))
    ap.add_argument('--weights', default=str(HERE / 'weights'))
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--device', default='auto', help='auto (cuda if available), cpu, cuda, cuda:1, ...')
    a = ap.parse_args()
    app, net = create_app(a.data, a.weights, a.device)
    net()                                          # load the model before the first request
    print(f'[topconverge] serving {a.data} on http://{a.host}:{a.port}/', flush=True)
    try:
        from waitress import serve
        serve(app, host=a.host, port=a.port, threads=8)
    except ImportError:
        app.run(host=a.host, port=a.port, threaded=True)


if __name__ == '__main__':
    main()
