/* edit.js - the rim annotator: four points on the picture, the rim's sides drawn as the straight lines of the world they are
 * (straight in undistorted space, curved through the lens), Step / Auto from the network, state, save, undo / redo. */
import { lensUndist, lensDist } from '/static/lens.js';
import { FILTERS } from '/static/filters.js';

const $ = id => document.getElementById(id);
const ID = decodeURIComponent(location.pathname.split('/').pop());
const QS = new URLSearchParams(location.search);
const SEG = ['frontline', 'goalline', 'sideline', 'centerline'];      // side k runs from point k+1 to point k+2
const AUTO_MAX = 60;

const cv = $('cv'), g = cv.getContext('2d'), work = $('work');
const img = new Image();
let lab = null, rim = null, quality = null, saved = null;           // saved: {rim, quality} as stored
let view = { s: 1, x: 0, y: 0 };                                     // screen = image * s + (x, y), css px
let sel = -1, drag = null, autoRun = false, autoUsed = false;
const undo = [], redo = [];
const pointers = new Map();
let netOn = localStorage.getItem('tc-net') === '1';
let helpOn = localStorage.getItem('tc-help') !== '0';

// ------------------------------------------------------------- state
const snap = () => JSON.stringify(rim);
function pushUndo() { undo.push(snap()); redo.length = 0; }
function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
function dirty() { return !same(rim, saved.rim) || quality !== saved.quality; }
function changed() { $('save').disabled = !dirty(); draw(); scheduleNet(); }

function setQuality(q) {
    quality = q;
    document.querySelectorAll('[data-q]').forEach(b => b.classList.toggle('on', String(b.dataset.q) === String(q)));
    changed();
}

// ------------------------------------------------------------- drawing
function toScreen(p) { return [p[0] * view.s + view.x, p[1] * view.s + view.y]; }
function toImage(x, y) { return [(x - view.x) / view.s, (y - view.y) / view.s]; }

function sidePoints(a, b, n = 64) {
    /* the straight line of the world between two image points: straight in undistorted space, drawn back through the lens */
    const L = lab.lens, ua = lensUndist(L, a[0], a[1]), ub = lensUndist(L, b[0], b[1]), out = [];
    for (let i = 0; i <= n; i++) {
        const t = i / n;
        out.push(lensDist(L, ua[0] + (ub[0] - ua[0]) * t, ua[1] + (ub[1] - ua[1]) * t));
    }
    return out;
}

function draw() {
    const dpr = devicePixelRatio || 1, W = work.clientWidth, H = work.clientHeight;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!lab || !img.complete) return;
    g.imageSmoothingEnabled = view.s < 1.5;                           // zoomed in: the camera's pixels as squares
    g.drawImage(img, view.x, view.y, img.naturalWidth * view.s, img.naturalHeight * view.s);
    if (!rim) return;
    g.lineWidth = 2;
    for (let k = 0; k < 4; k++) {
        const pts = sidePoints(rim[k], rim[(k + 1) % 4]).map(toScreen);
        g.strokeStyle = 'rgba(255,0,255,.8)';
        g.setLineDash([10, 8]);
        g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.stroke();
        g.setLineDash([]);
        const m = pts[pts.length >> 1];
        g.font = '12px system-ui'; g.fillStyle = 'rgba(0,0,0,.6)';
        const w = g.measureText(SEG[k]).width;
        g.fillRect(m[0] - w / 2 - 4, m[1] - 9, w + 8, 18);
        g.fillStyle = '#fbcfe8'; g.textAlign = 'center'; g.textBaseline = 'middle';
        g.fillText(SEG[k], m[0], m[1]);
    }
    rim.forEach((p, i) => {
        const [x, y] = toScreen(p);
        g.beginPath(); g.arc(x, y, i === sel ? 9 : 7, 0, Math.PI * 2);
        g.fillStyle = i === sel ? '#f59e0b' : 'rgba(255,0,255,.9)'; g.fill();
        g.strokeStyle = '#000'; g.lineWidth = 1.5; g.stroke();
        g.fillStyle = '#fff'; g.font = 'bold 13px system-ui'; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
        g.strokeStyle = '#000'; g.lineWidth = 3; g.strokeText(String(i + 1), x + 10, y - 9); g.fillText(String(i + 1), x + 10, y - 9);
    });
}

function fit() {
    const W = work.clientWidth, H = work.clientHeight, iw = img.naturalWidth || lab.width, ih = img.naturalHeight || lab.height;
    const s = Math.min(W / iw, H / ih) * 0.92;
    view = { s, x: (W - iw * s) / 2, y: (H - ih * s) / 2 };
    draw();
}

function zoomAt(x, y, f) {
    const s = Math.min(64, Math.max(0.02, view.s * f)), k = s / view.s;
    view = { s, x: x - (x - view.x) * k, y: y - (y - view.y) * k };
    draw();
}

// ------------------------------------------------------------- the network view
let netTimer = 0, netSeq = 0;
function scheduleNet() {
    if (!netOn || !rim) return;
    clearTimeout(netTimer);
    netTimer = setTimeout(async () => {
        const seq = ++netSeq;
        const r = await fetch(`/api/view/${ID}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rim }) });
        if (!r.ok || seq !== netSeq) return;
        const url = URL.createObjectURL(await r.blob());
        const old = $('netimg').src;
        $('netimg').src = url;
        if (old.startsWith('blob:')) URL.revokeObjectURL(old);
    }, autoRun ? 0 : 200);
}
function showNet(on) {
    netOn = on; localStorage.setItem('tc-net', on ? '1' : '0');
    $('netview').style.display = on ? 'block' : 'none';
    $('netbtn').classList.toggle('on', on);
    scheduleNet();
}
function showHelp(on) {
    helpOn = on; localStorage.setItem('tc-help', on ? '1' : '0');
    $('help').style.display = on ? 'block' : 'none';
    $('helpbtn').classList.toggle('on', on);
}

// ------------------------------------------------------------- Step / Auto
async function netStep() {
    const r = await fetch(`/api/step/${ID}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rim }) });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
}
async function step() {
    if (autoRun) return;
    pushUndo();
    try {
        const s = await netStep();
        rim = s.rim; autoUsed = true;
        $('status').textContent = `step: moved ${Math.max(...s.moved_px).toFixed(1)} px${s.converged ? ', settled' : ''} (${s.ms} ms)`;
    } catch (e) { undo.pop(); $('status').textContent = `step failed: ${e.message}`; }
    changed();
}
async function auto() {
    if (autoRun) { autoRun = false; return; }
    autoRun = true; $('auto').classList.add('on');
    pushUndo();
    const before = undo[undo.length - 1], t0 = performance.now();
    let n = 0, done = false;
    try {
        while (autoRun && n < AUTO_MAX) {
            const s = await netStep();
            if (!autoRun) break;
            rim = s.rim; n++; autoUsed = true;
            $('status').textContent = `auto: step ${n}, moved ${Math.max(...s.moved_px).toFixed(1)} px`;
            changed();
            if (s.converged) { done = true; break; }
        }
        $('status').textContent = `auto: ${done ? 'settled' : autoRun ? 'did not settle' : 'stopped'} after ${n} steps (${((performance.now() - t0) / 1000).toFixed(1)} s)`;
    } catch (e) { $('status').textContent = `auto failed: ${e.message}`; }
    if (snap() === before) undo.pop();
    autoRun = false; $('auto').classList.remove('on');
    changed();
}

// ------------------------------------------------------------- pointer
function hit(x, y) {
    let best = -1, bd = 16;
    (rim || []).forEach((p, i) => { const q = toScreen(p), d = Math.hypot(q[0] - x, q[1] - y); if (d < bd) { bd = d; best = i; } });
    return best;
}
function local(e) { const r = work.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

work.addEventListener('pointerdown', e => {
    autoRun = false;
    work.setPointerCapture(e.pointerId);
    const [x, y] = local(e);
    pointers.set(e.pointerId, [x, y]);
    if (pointers.size === 2) { drag = { pinch: true, ...pinchState() }; return; }
    const i = hit(x, y);
    if (i >= 0) {
        sel = i; pushUndo();
        const p = toScreen(rim[i]);
        drag = { point: i, dx: p[0] - x, dy: p[1] - y, moved: false };
    } else drag = { pan: true, x, y, vx: view.x, vy: view.y };
    draw();
});
function pinchState() {
    const [a, b] = [...pointers.values()];
    return { d: Math.hypot(a[0] - b[0], a[1] - b[1]), cx: (a[0] + b[0]) / 2, cy: (a[1] + b[1]) / 2, s: view.s, vx: view.x, vy: view.y };
}
work.addEventListener('pointermove', e => {
    if (!pointers.has(e.pointerId)) return;
    const [x, y] = local(e);
    pointers.set(e.pointerId, [x, y]);
    if (!drag) return;
    if (drag.pinch && pointers.size === 2) {
        const p = pinchState(), s = Math.min(64, Math.max(0.02, drag.s * p.d / drag.d)), k = s / drag.s;
        view = { s, x: p.cx - (drag.cx - drag.vx) * k, y: p.cy - (drag.cy - drag.vy) * k };
        draw();
    } else if (drag.point !== undefined) {
        rim[drag.point] = toImage(x + drag.dx, y + drag.dy);
        drag.moved = true;
        draw();
    } else if (drag.pan) {
        view.x = drag.vx + x - drag.x; view.y = drag.vy + y - drag.y;
        draw();
    }
});
function up(e) {
    pointers.delete(e.pointerId);
    if (drag && drag.point !== undefined) {
        if (!drag.moved) undo.pop();
        changed();
    }
    drag = pointers.size === 1 && drag && drag.pinch ? (() => { const [x, y] = [...pointers.values()][0]; return { pan: true, x, y, vx: view.x, vy: view.y }; })() : null;
}
work.addEventListener('pointerup', up);
work.addEventListener('pointercancel', up);
work.addEventListener('wheel', e => {
    e.preventDefault();
    const [x, y] = local(e), d = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;
    zoomAt(x, y, Math.exp(-d * 0.0015));
}, { passive: false });
new ResizeObserver(() => draw()).observe(work);

// ------------------------------------------------------------- buttons and keys
async function save() {
    if (!dirty()) return true;
    $('save').disabled = true;
    const r = await fetch(`/api/label/${ID}`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                 body: JSON.stringify({ rim: !lab.rim && same(rim, lab.start) ? null : rim, quality, auto_used: autoUsed }) });
    if (!r.ok) { $('status').textContent = `save failed: ${await r.text()}`; $('save').disabled = false; return false; }
    lab = { ...await r.json(), start: lab.start };
    saved = { rim: JSON.parse(JSON.stringify(rim)), quality };
    $('status').textContent = `saved ${lab.edited}`;
    changed();
    return true;
}
function doUndo() { if (!undo.length) return; redo.push(snap()); rim = JSON.parse(undo.pop()); changed(); }
function doRedo() { if (!redo.length) return; undo.push(snap()); rim = JSON.parse(redo.pop()); changed(); }
function reset() { autoRun = false; pushUndo(); rim = JSON.parse(JSON.stringify(lab.start)); changed(); }

let order = [];
function go(d) {
    const i = order.indexOf(ID);
    if (i < 0 || !order.length) return;
    if (dirty() && !confirm('This frame has unsaved changes. Leave without saving?')) return;
    saved = null;                                                     // answered: no second question from beforeunload
    location.href = `/edit/${order[(i + d + order.length) % order.length]}?${QS}`;
}

$('step').onclick = step;
$('auto').onclick = auto;
$('reset').onclick = reset;
$('undo').onclick = doUndo;
$('redo').onclick = doRedo;
$('fit').onclick = fit;
$('save').onclick = save;
$('prev').onclick = () => go(-1);
$('next').onclick = () => go(1);
$('netbtn').onclick = () => showNet(!netOn);
$('helpbtn').onclick = () => showHelp(!helpOn);
document.querySelectorAll('[data-q]').forEach(b => b.onclick = () => setQuality(/^\d$/.test(b.dataset.q) ? +b.dataset.q : b.dataset.q));

addEventListener('keydown', e => {
    if (e.target.closest('input, select, textarea')) return;
    const k = e.key.toLowerCase(), mod = e.ctrlKey || e.metaKey;
    if (mod && k === 'z') { e.preventDefault(); e.shiftKey ? doRedo() : doUndo(); }
    else if (mod && k === 'y') { e.preventDefault(); doRedo(); }
    else if (mod && k === 's') { e.preventDefault(); save(); }
    else if (mod || e.altKey) return;
    else if (k === 'a') e.shiftKey ? step() : auto();
    else if (k === 's') save();
    else if (k === 'r') reset();
    else if (k === 'f') fit();
    else if (k === 'v') showNet(!netOn);
    else if (k === 'h') showHelp(!helpOn);
    else if (k === '0') setQuality('needs_work');
    else if (k === 'x') setQuality('unusable');
    else if ('123'.includes(k) && k) setQuality(+k);
    else if (e.key === 'ArrowLeft') go(-1);
    else if (e.key === 'ArrowRight') go(1);
    else if (e.key === 'Escape') { autoRun = false; sel = -1; draw(); }
});
addEventListener('beforeunload', e => { if (saved && dirty()) { e.preventDefault(); e.returnValue = ''; } });

// ------------------------------------------------------------- load
lab = await (await fetch(`/api/label/${ID}`)).json();
rim = lab.rim ? lab.rim.map(p => [...p]) : JSON.parse(JSON.stringify(lab.start));
quality = lab.quality ?? null;
saved = { rim: lab.rim ? lab.rim.map(p => [...p]) : null, quality };
if (!lab.rim) saved.rim = JSON.parse(JSON.stringify(rim));        // the start alone is not a change: move a point or pick a state
document.title = `${ID} - topconverge`;
$('name').textContent = ID;
$('meta').textContent = `${lab.view} lens · ${lab.split}${lab.auto_used ? ' · auto used' : ''}${lab.edited ? ' · edited ' + lab.edited : ''}`;
$('back').href = `/`;
setQuality(quality);
showNet(netOn);
showHelp(helpOn);
img.onload = () => fit();
img.src = `/images/${ID}.jpg`;
fetch('/api/items').then(r => r.json()).then(items => {
    const f = FILTERS[QS.get('f')] || FILTERS.all, v = QS.get('v');
    order = items.filter(it => f(it) && (!v || it.view === v)).map(it => it.id);
    if (!order.includes(ID)) order = items.map(it => it.id);
    $('pos').textContent = `${order.indexOf(ID) + 1} / ${order.length}`;
});
