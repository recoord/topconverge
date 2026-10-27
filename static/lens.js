/* lens.js - the camera lens in the browser: OpenCV's 14-coefficient model, undistortion by the same iterative
 * scheme the server uses (60 iterations, as cv2.undistortPointsIter) and forward distortion. A lens is
 * {fx, fy, cx, cy, dist: [k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy]} as the labels carry it. */

export const IDENTITY = { fx: 1, fy: 1, cx: 0, cy: 0, dist: [] };

export function lensCoef(l) {
    const d = l.dist || [], g = i => d[i] || 0;
    return { k1: g(0), k2: g(1), p1: g(2), p2: g(3), k3: g(4), k4: g(5), k5: g(6), k6: g(7), s1: g(8), s2: g(9), s3: g(10), s4: g(11) };
}

/* image pixel -> normalised undistorted coordinates */
export function lensUndist(l, x, y) {
    const c = lensCoef(l), ux = (x - l.cx) / l.fx, uy = (y - l.cy) / l.fy;
    if (!l.dist || !l.dist.length) return [ux, uy];
    let px = ux, py = uy;
    for (let i = 0; i < 60; i++) {
        const r2 = px * px + py * py;
        const ic = (1 + ((c.k6 * r2 + c.k5) * r2 + c.k4) * r2) / (1 + ((c.k3 * r2 + c.k2) * r2 + c.k1) * r2);
        const dx = 2 * c.p1 * px * py + c.p2 * (r2 + 2 * px * px) + c.s1 * r2 + c.s2 * r2 * r2;
        const dy = c.p1 * (r2 + 2 * py * py) + 2 * c.p2 * px * py + c.s3 * r2 + c.s4 * r2 * r2;
        px = (ux - dx) * ic; py = (uy - dy) * ic;
    }
    return [px, py];
}

/* normalised undistorted coordinates -> image pixel */
export function lensDist(l, x, y) {
    if (!l.dist || !l.dist.length) return [x * l.fx + l.cx, y * l.fy + l.cy];
    const c = lensCoef(l), r2 = x * x + y * y;
    const rad = (1 + ((c.k3 * r2 + c.k2) * r2 + c.k1) * r2) / (1 + ((c.k6 * r2 + c.k5) * r2 + c.k4) * r2);
    const xd = x * rad + 2 * c.p1 * x * y + c.p2 * (r2 + 2 * x * x) + c.s1 * r2 + c.s2 * r2 * r2;
    const yd = y * rad + c.p1 * (r2 + 2 * y * y) + 2 * c.p2 * x * y + c.s3 * r2 + c.s4 * r2 * r2;
    return [xd * l.fx + l.cx, yd * l.fy + l.cy];
}
