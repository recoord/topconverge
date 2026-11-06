/* netview.js - the network view drawn in the browser, so it follows a dragged point at once.
 *
 * The top view is rimnet.RimNet.render redone in a WebGL2 fragment shader: every view pixel is a point (u, v) of the plane window
 * W0..W1 (top row = far side), taken through the inverse homography of the current rim (its corners undistorted and sent to the
 * 100 x 100 square with point 1 at the bottom right - a right lens comes out mirrored, as for the network) and the lens distortion to
 * the picture; a checkerboard where no camera pixel lands. A crop model's four crops (S x S full-resolution px round each current
 * corner, mirrored for a right lens) are cut from the picture with a 2D canvas. The predicted corners and each crop's share of the
 * answer come from the server (/api/pred) and are drawn only while they belong to the rim on screen.
 * The top view samples a half-resolution copy of the frame bilinearly, as the network does (the browser's downscale stands in for
 * OpenCV's INTER_AREA, so the pixels differ slightly); the server's PNG (the option off) is the exact input. */
import { lensUndist } from '/static/lens.js';

const SIDE = 100, DST = [[SIDE, 0], [0, 0], [0, SIDE], [SIDE, SIDE]];

function perspective(src, dst) {
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
        const [x, y] = src[i], [u, v] = dst[i];
        A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
        A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    for (let c = 0; c < 8; c++) {
        let p = c;
        for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
        if (Math.abs(A[p][c]) < 1e-15) return null;
        [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
        for (let r = 0; r < 8; r++) {
            if (r === c) continue;
            const f = A[r][c] / A[c][c];
            for (let k = c; k < 8; k++) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    const h = b.map((v, i) => v / A[i][i]);
    return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}
function inv3(m) {
    const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-300) return null;
    return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
            [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
            [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
}
/* rim (4 x [x, y] image px) -> plane -> undistorted (rimnet.homography: signed so the square's middle is in front) */
function rimHinv(rim, lens) {
    const und = rim.map(p => lensUndist(lens, p[0], p[1]));
    const h = perspective(und, DST);
    if (!h) return null;
    let hi = inv3(h);
    if (!hi) return null;
    if (hi[2][0] * 50 + hi[2][1] * 50 + hi[2][2] < 0) hi = hi.map(r => r.map(v => -v));
    return hi;
}

const VS = `#version 300 es
    in vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }`;
const FS = `#version 300 es
    precision highp float;
    uniform float u_n; uniform vec2 u_win; uniform mat3 u_hinv; uniform vec4 u_k; uniform float u_d[12]; uniform vec2 u_img;
    uniform float u_cell; uniform sampler2D u_tex;
    out vec4 color;
    void main() {
      vec2 V = vec2(gl_FragCoord.x, u_n - gl_FragCoord.y);               /* canvas px, y down */
      float t = (u_win.y - u_win.x) / u_n;
      vec2 uv = vec2(u_win.x + V.x * t, u_win.y - V.y * t);              /* plane units, top row = far side */
      float chk = mod(floor(V.x / u_cell) + floor(V.y / u_cell), 2.0) < 0.5 ? 90.0 / 255.0 : 60.0 / 255.0;
      color = vec4(vec3(chk), 1.0);
      vec3 q = u_hinv * vec3(uv, 1.0);
      if (q.z <= 1e-9) return;
      vec2 n = q.xy / q.z;
      float r2 = dot(n, n);
      if (r2 >= 25.0) return;
      float k1 = u_d[0], k2 = u_d[1], p1 = u_d[2], p2 = u_d[3], k3 = u_d[4], k4 = u_d[5], k5 = u_d[6], k6 = u_d[7];
      float s1 = u_d[8], s2 = u_d[9], s3 = u_d[10], s4 = u_d[11];
      float rad = (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1.0 + ((k6 * r2 + k5) * r2 + k4) * r2);
      float xd = n.x * rad + 2.0 * p1 * n.x * n.y + p2 * (r2 + 2.0 * n.x * n.x) + s1 * r2 + s2 * r2 * r2;
      float yd = n.y * rad + p1 * (r2 + 2.0 * n.y * n.y) + 2.0 * p2 * n.x * n.y + s3 * r2 + s4 * r2 * r2;
      vec2 px = vec2(xd * u_k.x + u_k.z, yd * u_k.y + u_k.w);
      if (px.x < 0.0 || px.y < 0.0 || px.x > u_img.x || px.y > u_img.y) return;
      color = vec4(texture(u_tex, px / u_img).rgb, 1.0);
    }`;

export class NetView {
    /* box: the #netview element; it gets a WebGL canvas (top view) and a 2D canvas over everything (crops, marks) */
    constructor(box) {
        this.box = box;
        this.glc = document.createElement('canvas'); this.glc.className = 'netgl';
        this.c2 = document.createElement('canvas'); this.c2.className = 'net2d';
        box.prepend(this.c2); box.prepend(this.glc);
        this.gl = null; this.tex = null; this.texSrc = null; this.cfg = { margin: 12.5, res: 1024, crop: 0 };
    }
    setConfig(cfg) { this.cfg = { margin: cfg.margin ?? 12.5, res: cfg.res ?? 1024, crop: cfg.crop || 0 }; }
    show(on) { this.glc.style.display = this.c2.style.display = on ? 'block' : 'none'; }
    init() {
        if (this.gl) return true;
        const gl = this.glc.getContext('webgl2', { antialias: false });
        if (!gl) return false;
        const sh = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s)); return s; };
        const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { console.error(gl.getProgramInfoLog(prog)); return false; }
        const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
        this.U = {};
        ['u_n', 'u_win', 'u_hinv', 'u_k', 'u_d', 'u_img', 'u_cell', 'u_tex'].forEach(n => { this.U[n] = gl.getUniformLocation(prog, n); });
        this.gl = gl; this.prog = prog;
        return true;
    }
    texture(img) {
        const gl = this.gl;
        if (this.tex && this.texSrc === img.currentSrc) return;
        if (!this.tex) this.tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        // as the network reads it: a half-resolution copy (area-averaged), sampled bilinearly without mipmaps
        const half = document.createElement('canvas');
        half.width = Math.round(img.naturalWidth / 2); half.height = Math.round(img.naturalHeight / 2);
        const hg = half.getContext('2d'); hg.imageSmoothingEnabled = true; hg.imageSmoothingQuality = 'high';
        hg.drawImage(img, 0, 0, half.width, half.height);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, half);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this.texSrc = img.currentSrc;
    }
    /* img: the full-resolution frame, lens, rim (4 x [x, y]), mirror (right lens), pred: the server's answer for this rim or null */
    draw(img, lens, rim, mirror, pred) {
        if (!img.complete || !img.naturalWidth || !rim || !this.init()) return false;
        const S = this.cfg.crop, dpr = devicePixelRatio || 1;
        this.box.classList.toggle('wide', !!S);
        const Hc = this.box.clientHeight, Wc = this.box.clientWidth;
        const n = Math.max(64, Math.round(Hc * dpr));                   // top view: a square of the box's height
        const W0 = -this.cfg.margin, W1 = SIDE + this.cfg.margin;
        const hi = rimHinv(rim, lens);
        // ---- top view (WebGL)
        const gl = this.gl, U = this.U;
        if (this.glc.width !== n || this.glc.height !== n) { this.glc.width = this.glc.height = n; }
        this.glc.style.width = this.glc.style.height = Hc + 'px';
        this.texture(img);
        gl.viewport(0, 0, n, n);
        gl.useProgram(this.prog);
        gl.uniform1f(U.u_n, n); gl.uniform2f(U.u_win, W0, W1);
        const H = hi || [[0, 0, 0], [0, 0, 0], [0, 0, -1]];
        gl.uniformMatrix3fv(U.u_hinv, false, new Float32Array([H[0][0], H[1][0], H[2][0], H[0][1], H[1][1], H[2][1], H[0][2], H[1][2], H[2][2]]));
        gl.uniform4f(U.u_k, lens.fx, lens.fy, lens.cx, lens.cy);
        const d = (lens.dist || []).slice(0, 12); while (d.length < 12) d.push(0); gl.uniform1fv(U.u_d, new Float32Array(d));
        gl.uniform2f(U.u_img, img.naturalWidth, img.naturalHeight);
        gl.uniform1f(U.u_cell, n * 32 / this.cfg.res);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex); gl.uniform1i(U.u_tex, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        // ---- crops and marks (2D)
        const c2 = this.c2;
        if (c2.width !== Math.round(Wc * dpr) || c2.height !== Math.round(Hc * dpr)) { c2.width = Math.round(Wc * dpr); c2.height = Math.round(Hc * dpr); }
        c2.style.width = Wc + 'px'; c2.style.height = Hc + 'px';
        const g = c2.getContext('2d');
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        g.clearRect(0, 0, Wc, Hc);
        const toView = p => [(p[0] - W0) / (W1 - W0) * Hc, (W1 - p[1]) / (W1 - W0) * Hc];
        g.lineWidth = 2; g.strokeStyle = 'rgb(80,220,80)';
        g.beginPath(); DST.map(toView).forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.closePath(); g.stroke();
        if (pred && pred.pred) this.marks(g, pred.pred.map(toView));
        if (S) this.crops(g, img, rim, mirror, pred, Hc, S);
        return true;
    }
    marks(g, pts, labels = true) {
        g.lineWidth = 2; g.strokeStyle = 'rgb(255,0,255)'; g.fillStyle = 'rgb(255,0,255)'; g.font = 'bold 13px system-ui';
        pts.forEach((p, k) => {
            if (!p) return;
            g.beginPath(); g.arc(p[0], p[1], 6, 0, Math.PI * 2); g.stroke();
            if (labels) g.fillText(String(k + 1), p[0] + 8, p[1] - 6);
        });
    }
    crops(g, img, rim, mirror, pred, Hc, S) {
        const h = Hc / 2, f = h / S, IW = img.naturalWidth, IH = img.naturalHeight;
        for (let k = 0; k < 4; k++) {
            const x0 = Math.floor(rim[k][0] - S / 2 + 0.5), y0 = Math.floor(rim[k][1] - S / 2 + 0.5);
            const cx = Hc + (k % 2) * h, cy = Math.floor(k / 2) * h;
            g.save();
            g.beginPath(); g.rect(cx, cy, h, h); g.clip();
            const cell = 32 * f;                                            // checkerboard, as the server draws it
            for (let i = 0; i * cell < h; i++) for (let j = 0; j * cell < h; j++) {
                g.fillStyle = (i + j) % 2 ? 'rgb(60,60,60)' : 'rgb(90,90,90)';
                g.fillRect(cx + j * cell, cy + i * cell, cell + 0.5, cell + 0.5);
            }
            // crop px (x0 + a, y0 + b) -> cell (cx + a f, cy + b f), mirrored: cx + (S - a) f
            if (mirror) { g.translate(cx + h, cy); g.scale(-1, 1); } else g.translate(cx, cy);
            const sx = Math.max(0, x0), sy = Math.max(0, y0), ex = Math.min(IW, x0 + S), ey = Math.min(IH, y0 + S);
            if (sx < ex && sy < ey) {
                g.imageSmoothingEnabled = f < 1;
                g.drawImage(img, sx, sy, ex - sx, ey - sy, (sx - x0) * f, (sy - y0) * f, (ex - sx) * f, (ey - sy) * f);
            }
            g.restore();
            const at = p => { const a = (p[0] - x0) * f, b = (p[1] - y0) * f; return [cx + (mirror ? h - a : a), cy + b]; };
            const c = at(rim[k]);
            g.strokeStyle = 'rgb(80,220,80)'; g.lineWidth = 2;
            g.beginPath(); g.moveTo(c[0] - 10, c[1]); g.lineTo(c[0] + 10, c[1]); g.moveTo(c[0], c[1] - 10); g.lineTo(c[0], c[1] + 10); g.stroke();
            if (pred && pred.pred_px && pred.pred_px[k]) {
                const q = at(pred.pred_px[k]);
                if (q[0] > cx - 10 && q[0] < cx + h + 10 && q[1] > cy - 10 && q[1] < cy + h + 10) { g.save(); g.beginPath(); g.rect(cx, cy, h, h); g.clip(); this.marks(g, [q], false); g.restore(); }
            }
            const txt = `${k + 1}  crop ${pred && pred.share ? Math.round(pred.share[k] * 100) + ' %' : '…'}`;
            g.font = '12px system-ui';
            g.fillStyle = 'rgba(0,0,0,.7)'; g.fillRect(cx, cy, g.measureText(txt).width + 10, 18);
            g.fillStyle = '#fff'; g.fillText(txt, cx + 5, cy + 13);
            g.strokeStyle = 'rgb(120,120,120)'; g.lineWidth = 1; g.strokeRect(cx + 0.5, cy + 0.5, h - 1, h - 1);
        }
    }
}

