"""Procedural teeth, gums and tongue, placed from the fitted skeleton.

Teeth: 14 per arch (no third molars), each a tapered superellipsoid crown
oriented on a smooth dental arch (mesiodistal along the arch, labial along
its outward normal), with generic crown proportions per tooth type. The
lower arch is a little narrower and sits behind the upper incisors (overjet)
and above their edges (overbite). Gums are swept U-shaped bands along each
arch. The tongue is a lofted, domed ellipse section along the tongue joints,
narrowing and thinning to a rounded tip. Dimensions are generic textbook-
scale choices (millimetres, times the subject's scale), not measured data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .common import normalize, resample_polyline


@dataclass
class PartMesh:
    name: str
    positions: np.ndarray
    tris: np.ndarray
    joints: np.ndarray      # (V, 4) joint names' indices (into the skeleton list)
    weights: np.ndarray     # (V, 4)
    material: str
    uv: np.ndarray | None = None


# per quadrant from the midline: (width, labio-lingual thickness, crown height, kind)
UPPER = [(8.6, 7.0, 10.5, "incisor"), (6.6, 6.2, 9.0, "incisor"), (7.6, 8.0, 10.0, "canine"),
         (7.0, 9.0, 8.2, "premolar"), (6.5, 9.0, 7.6, "premolar"), (10.0, 11.0, 7.4, "molar"),
         (9.0, 10.5, 7.0, "molar")]
LOWER = [(5.3, 5.8, 9.0, "incisor"), (5.9, 6.0, 9.2, "incisor"), (6.9, 7.5, 10.5, "canine"),
         (7.0, 7.8, 8.2, "premolar"), (7.1, 8.0, 7.8, "premolar"), (11.0, 10.3, 7.4, "molar"),
         (10.5, 10.0, 7.0, "molar")]
# half arch (mm): x lateral, z backwards from the incisal midpoint (smoothed polyline)
ARCH = np.array([(0, 0), (6, -1.2), (11.5, -4.2), (16.5, -9.0), (20, -15), (22.6, -22), (25, -30), (27.2, -39),
                 (28.8, -48)], np.float64)


def _arch(scale_x: float, scale_z: float, n: int = 400):
    half = ARCH * [scale_x, scale_z]
    full = np.concatenate([half[::-1] * [-1, 1], half[1:]])
    pts = resample_polyline(full, n)
    # arc length from the midline, signed (right negative)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    s -= np.interp(0.0, pts[:, 0], s)
    return pts, s


def _at(pts, s, t):
    """Point and unit tangent on the arch at signed arc length t (mm)."""
    x = np.interp(t, s, pts[:, 0])
    z = np.interp(t, s, pts[:, 1])
    dt = 0.5
    x2, z2 = np.interp(t + dt, s, pts[:, 0]), np.interp(t + dt, s, pts[:, 1])
    tan = normalize(np.array([x2 - x, z2 - z]))
    return np.array([x, z]), tan


def _crown(w, th, h, kind, lat=10, lon=18, root=4.0):
    """Tooth in its frame (mm): x mesiodistal, y from the incisal edge (0)
    towards the gum (+h) and root (+h+root), z labial."""
    v = np.linspace(0, 1, lat)
    u = np.linspace(0, 2 * math.pi, lon, endpoint=False)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    p = 3.2
    cu, su = np.cos(uu), np.sin(uu)
    rx = np.sign(cu) * np.abs(cu) ** (2 / p) * w / 2
    rz = np.sign(su) * np.abs(su) ** (2 / p) * th / 2
    y = vv * (h + root)
    # taper: crowns bulge a little above the gum line; incisors thin to an edge
    edge = np.clip(1 - vv * (h + root) / max(h * 0.45, 1e-6), 0, 1)
    thick = {"incisor": 1 - 0.75 * edge ** 1.5, "canine": 1 - 0.55 * edge ** 1.5,
             "premolar": 1 - 0.2 * edge, "molar": 1 - 0.12 * edge}[kind]
    wide = 1 - 0.18 * np.clip((vv * (h + root) - h) / root, 0, 1) - (0.35 * edge ** 2 if kind == "canine" else 0)
    x = rx * wide
    z = rz * thick
    if kind == "canine":                       # a cusp tip
        y = y - 1.2 * np.exp(-(x / (0.3 * w)) ** 2) * edge
    pos = np.stack([x, y, z], -1).reshape(-1, 3)
    tris = []
    for i in range(lat - 1):
        for j in range(lon):
            a, b = i * lon + j, i * lon + (j + 1) % lon
            c, d = a + lon, b + lon
            tris += [(a, c, b), (b, c, d)]
    # close the incisal end with a fan to its centre
    cen = len(pos)
    pos = np.vstack([pos, [0, -0.3 if kind != "canine" else -1.2, 0]])
    for j in range(lon):
        tris.append((cen, j, (j + 1) % lon))
    return pos, np.asarray(tris, np.int64)


def teeth(skel: dict, upper: bool, scale: float, jidx: dict) -> tuple[PartMesh, PartMesh]:
    s = scale
    table = UPPER if upper else LOWER
    joint = "teeth_upper" if upper else "teeth_lower"
    bind = np.asarray(next(j["bind"] for j in skel["joints"] if j["name"] == joint))
    origin = bind[:3, 3]
    sx, sz = (1.0, 1.0) if upper else (0.9, 0.92)
    pts, sarc = _arch(sx, sz)
    pos_all, tri_all = [], []
    gum_frames = []
    base = 0
    for side in (-1, 1):
        t = 0.0
        for w, th, h, kind in table:
            c_arc = side * (t + w / 2)
            t += w
            p2, tan = _at(pts, sarc, c_arc)
            out = np.array([tan[1], -tan[0]])             # labial: away from the arch's inside
            if out @ np.array([p2[0], p2[1] + 20]) < 0:
                out = -out
            local, tri = _crown(w, th, h, kind)
            ex = np.array([tan[0], 0, tan[1]])
            ez = np.array([out[0], 0, out[1]])
            ey = np.array([0, 1.0 if upper else -1.0, 0])
            # molars sit a little higher (upper) / lower (lower): the curve of Spee
            rise = 0.03 * max(0.0, abs(c_arc) - 15.0)
            p3 = np.array([p2[0], rise * (1 if upper else -1), p2[1]])
            world = p3 + local[:, :1] * ex + local[:, 1:2] * ey + local[:, 2:3] * ez
            if np.linalg.det(np.stack([ex, ey, ez], 1)) < 0:     # a mirrored frame: keep faces outward
                tri = tri[:, [0, 2, 1]]
            pos_all.append(world)
            tri_all.append(tri + base)
            base += len(world)
            gum_frames.append((c_arc, h))
    pos = np.concatenate(pos_all) * 0.001 * s + origin
    tris = np.concatenate(tri_all)
    n = len(pos)
    jj = np.zeros((n, 4), np.int64)
    jj[:, 0] = jidx[joint]
    ww = np.zeros((n, 4))
    ww[:, 0] = 1
    tooth = PartMesh(f"teeth_{'upper' if upper else 'lower'}", pos, tris, jj, ww, "teeth")
    gum = _gum(pts, sarc, table, upper, s, origin, jidx[joint])
    return tooth, gum


def _gum(pts, sarc, table, upper, s, origin, joint_index):
    half = sum(w for w, *_ in table)
    ts = np.linspace(-half, half, 72)
    prof = []                                       # U section (mm): (normal offset, height)
    for a in np.linspace(0, math.pi, 9):
        prof.append((5.4 * math.cos(a), 7.2 + 3.0 * math.sin(a)))
    prof = [(6.0, 5.0)] + prof + [(-6.0, 5.0)]
    rows = []
    for t in ts:
        p2, tan = _at(pts, sarc, t)
        out = np.array([tan[1], -tan[0]])
        if out @ np.array([p2[0], p2[1] + 20]) < 0:
            out = -out
        th = np.interp(abs(t), np.cumsum([0] + [w for w, *_ in table]), [x[1] for x in table] + [table[-1][1]])
        k = th / 9.0
        ring = []
        for o, hgt in prof:
            y = hgt if upper else -hgt
            ring.append([p2[0] + out[0] * o * k, y, p2[1] + out[1] * o * k])
        rows.append(ring)
    rows = np.asarray(rows)                          # (T, P, 3)
    T_, P_ = rows.shape[:2]
    pos = rows.reshape(-1, 3) * 0.001 * s + origin
    tris = []
    for i in range(T_ - 1):
        for j in range(P_ - 1):
            a, b = i * P_ + j, i * P_ + j + 1
            c, d = a + P_, b + P_
            tris += [(a, b, c), (b, d, c)] if upper else [(a, c, b), (b, c, d)]
    n = len(pos)
    jj = np.zeros((n, 4), np.int64)
    jj[:, 0] = joint_index
    ww = np.zeros((n, 4))
    ww[:, 0] = 1
    return PartMesh(f"gums_{'upper' if upper else 'lower'}", pos, np.asarray(tris, np.int64), jj, ww, "gums")


def tongue(skel: dict, scale: float, jidx: dict, nu: int = 28, nv: int = 20) -> PartMesh:
    names = ["tongue_01", "tongue_02", "tongue_03", "tongue_04"]
    P = np.array([np.asarray(next(j["bind"] for j in skel["joints"] if j["name"] == n))[:3, 3] for n in names])
    mm = 0.001 * scale
    # the body runs a little past the last joint (the tip) and behind the first (the root)
    root = P[0] - normalize(P[1] - P[0]) * 8 * mm
    tip = P[-1] + normalize(P[-1] - P[-2]) * 5 * mm
    curve = resample_polyline(np.vstack([root, P, tip]), nu)
    u = np.linspace(0, 1, nu)
    half_w = np.interp(u, [0, .3, .65, .88, 1], [19, 21, 19, 13, 3]) * mm
    thick = np.interp(u, [0, .3, .7, .9, 1], [14, 13, 9, 5.5, 1.5]) * mm
    tang = normalize(np.gradient(curve, axis=0))
    side = np.array([1.0, 0, 0])
    upv = normalize(np.cross(tang, side))
    upv[upv[:, 1] < 0] *= -1
    v = np.linspace(0, 2 * math.pi, nv, endpoint=False)
    rows = []
    for i in range(nu):
        c, s = np.cos(v), np.sin(v)
        y = np.where(s > 0, 0.55 * s, 0.45 * s) * thick[i] + 0.1 * thick[i] * (1 - c ** 2) * (s > 0)
        x = c * half_w[i]
        rows.append(curve[i] + x[:, None] * side + y[:, None] * upv[i])
    rows = np.asarray(rows)
    pos = rows.reshape(-1, 3)
    tris = []
    for i in range(nu - 1):
        for j in range(nv):
            a, b = i * nv + j, i * nv + (j + 1) % nv
            c_, d = a + nv, b + nv
            tris += [(a, b, c_), (b, d, c_)]
    for end, i in (("root", 0), ("tip", nu - 1)):
        cen = len(pos)
        pos = np.vstack([pos, rows[i].mean(0)])
        for j in range(nv):
            a, b = i * nv + j, i * nv + (j + 1) % nv
            tris.append((cen, b, a) if end == "root" else (cen, a, b))
    tris = np.asarray(tris, np.int64)
    # orientation: outward (from the centre line)
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    cen_line = pos[tris].mean(1) - curve[np.clip((tris[:, 0] // nv), 0, nu - 1)]
    if (fn * cen_line).sum() < 0:
        tris = tris[:, [0, 2, 1]]
    # weights along the chain: piecewise linear in arc position
    t_v = np.concatenate([np.repeat(u, nv), [0.0, 1.0]])
    jp = np.array([0.18, 0.42, 0.66, 0.9])            # joints' positions along u
    n = len(pos)
    jj = np.zeros((n, 4), np.int64)
    ww = np.zeros((n, 4))
    for k in range(n):
        t = t_v[k]
        if t <= jp[0]:
            jj[k, :2] = [jidx["jaw"], jidx[names[0]]]
            f = t / jp[0]
            ww[k, :2] = [1 - f, f]
            continue
        i = min(int(np.searchsorted(jp, t)) - 1, 2)
        f = np.clip((t - jp[i]) / (jp[i + 1] - jp[i]), 0, 1)
        jj[k, :2] = [jidx[names[i]], jidx[names[i + 1]]]
        ww[k, :2] = [1 - f, f]
    return PartMesh("tongue", pos, tris, jj, ww, "tongue")
