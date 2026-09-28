"""The template head: one fixed topology that every generated head is fitted to.

Original procedural construction (no scanned or licensed template):
- Every skin vertex has a *direction* from a centre C inside the skull
  (C = CENTER in the head frame). Directions are drawn in a chart: the
  azimuthal equidistant projection about +Z (the face), in "chart mm"
  (angle times R0 = 100 mm).
- Around the eyes, the mouth and the neck cut, vertices lie on edge loops:
  ring 0 is the feature contour (lid margin, lip seam, neck cut) and ring k
  its outward offset by RING_MM[k]. Rings are rebuilt from each subject's
  own contours with the same function (`rings_chart`), so the loops follow
  the subject's lids and lips exactly.
- Elsewhere a blue-noise point set (denser on the face) is thinned from a
  Fibonacci sphere and triangulated by the spherical Delaunay triangulation
  (the convex hull of the directions); triangles spanning a hole are removed.
- Inside the holes, strips continue the sheet: the lid margin and its lining
  (two rings per eye) and the lips' inner surface and the mouth cavity (a bag
  closed by a fan).
Canonical feature curves below are synthetic, rounded choices for an adult
head, used only to lay out the topology; each subject's own features replace
them during fitting.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .common import normalize, smoothstep

VERSION = 1
R0 = 100.0                      # chart mm per radian
CENTER = np.array([0.0, 0.0, -0.075])   # metres, head frame
EYE_N = 48
MOUTH_HALF = 32                 # samples per lip between the corners
MOUTH_N = 2 * MOUTH_HALF        # ring size (corners at 0 and MOUTH_HALF)
NECK_N = 48
EYE_RING_MM = (0.0, 1.1, 2.4, 4.0, 6.0, 8.6, 11.8, 15.5)
MOUTH_RING_MM = (0.0, 1.0, 2.1, 3.4, 5.0, 7.2, 10.0, 13.5, 18.0)
NECK_RING_MM = (0.0, 5.0)
EYE_INNER = 2                   # lid margin, lining
MOUTH_INNER = 6                 # wet lip, lip/gum fold, 4 cavity rings (then a cap)
FACE_SPACING = 1.5              # chart mm (the thinned points end up ~1.4x apart)
MOUTH_LENS_MM = 4.5             # half-height of the layout's open mouth
BACK_SPACING = 4.5
KIND = {"free": 0, "eye": 1, "mouth": 2, "neck": 3, "eye_inner": 4, "mouth_inner": 5, "cap": 6}


# ---- chart -----------------------------------------------------------------------

def to_chart(d: np.ndarray) -> np.ndarray:
    """Unit directions (N, 3) -> chart mm (N, 2) (azimuthal equidistant about +Z)."""
    d = normalize(d)
    a = np.arccos(np.clip(d[:, 2], -1, 1))
    r = np.hypot(d[:, 0], d[:, 1])
    s = np.where(r > 1e-12, a / np.maximum(r, 1e-12), 1.0) * R0
    return np.stack([d[:, 0] * s, d[:, 1] * s], 1)


def from_chart(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float64)
    a = np.hypot(c[:, 0], c[:, 1]) / R0
    k = np.where(a > 1e-12, np.sin(a) / np.maximum(a, 1e-12), 1.0) / R0
    return np.stack([c[:, 0] * k, c[:, 1] * k, np.cos(a)], 1)


def chart_of_points(p: np.ndarray, center=CENTER) -> np.ndarray:
    return to_chart(np.asarray(p, np.float64) - center)


def offset_polygon(poly: np.ndarray, d: float, outward_sign: float = 1.0) -> np.ndarray:
    """Offset a closed 2D polygon along its vertex normals (mitred, capped).
    Works for zero-area slits too: the halves separate, the tips extend."""
    if d == 0:
        return poly.copy()
    nxt, prv = np.roll(poly, -1, 0), np.roll(poly, 1, 0)
    e1 = normalize(poly - prv)
    e2 = normalize(nxt - poly)
    n1 = np.stack([e1[:, 1], -e1[:, 0]], 1)
    n2 = np.stack([e2[:, 1], -e2[:, 0]], 1)
    n = n1 + n2
    tip = np.linalg.norm(n, axis=1) < 1e-6          # a slit's corner: the edges fold back
    n[tip] = e1[tip] * outward_sign        # a tip extends along its incoming edge, whatever the winding
    n = normalize(n)
    cos = np.clip((n * n1).sum(1), 0.35, 1.0)
    cos[tip] = 1.0
    # the polygon is counter-clockwise: (e.y, -e.x) points outward
    return poly + outward_sign * n * (d / cos)[:, None]


def offset_loop(poly: np.ndarray, d: float, sign: float) -> np.ndarray:
    """An evenly sampled offset ring: offset, round the tips a little (more
    for far rings), then resample each half (between samples 0 and N/2, the
    corners) by arc length so the loops keep their vertex correspondence."""
    q = offset_polygon(poly, d, sign)
    n = len(q)
    h = n // 2
    for _ in range(int(d / 1.2)):              # the corners (0, N/2) hold: the ring must pass beyond them
        s = 0.5 * q + 0.25 * (np.roll(q, 1, 0) + np.roll(q, -1, 0))
        s[[0, h]] = q[[0, h]]
        q = s
    a = _resample(q[:h + 1], h + 1)
    b = _resample(np.vstack([q[h:], q[:1]]), n - h + 1)
    return np.vstack([a, b[1:-1]])


def _resample(p: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 1e-12:
        return np.repeat(p[:1], n, 0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, c]) for c in range(p.shape[1])], 1)


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None], poly[:, 1][None]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    cond = (y1 > y) != (y2 > y)
    xin = x1 + (y - y1) * (x2 - x1) / np.where(y2 - y1 == 0, 1e-12, y2 - y1)
    return (cond & (x < xin)).sum(1) % 2 == 1


# ---- canonical features (synthetic) ---------------------------------------------------

def _almond(cx, cy, cz, half_w, up, down, medial_sign, n=EYE_N):
    """A lid-margin loop in 3D (metres): medial corner, upper lid, lateral, lower."""
    t = np.linspace(0, math.pi, n // 2, endpoint=False)
    u = np.cos(t)                                 # +1 medial .. -1 lateral
    x_up = cx + medial_sign * half_w * u
    y_up = cy + up * np.sin(t) ** 0.8 + 0.0008 * u
    x_lo = cx - medial_sign * half_w * u
    y_lo = cy - down * np.sin(t) ** 0.9 - 0.0008 * u
    x = np.concatenate([x_up, x_lo])
    y = np.concatenate([y_up, y_lo])
    z = cz + 0.006 * (1 - ((x - cx) / half_w) ** 2) - 0.002 * medial_sign * (x - cx) / half_w * 0
    return np.stack([x, y, z], 1)


def canonical_features() -> dict:
    eyes = {}
    for side, sx in (("right", -1.0), ("left", 1.0)):
        # medial is towards the nose: -sx
        eyes[side] = _almond(sx * 0.0345, 0.0, 0.006, 0.0125, 0.0055, 0.005, -sx)
    j = np.linspace(-1, 1, MOUTH_HALF + 1)
    seam_x = 0.031 * j
    seam = np.stack([seam_x, -0.0745 + 0.0015 * j ** 2, 0.048 - 0.015 * j ** 2], 1)
    # an open lens in the chart for the layout (the rings need room); subjects close it
    sc = chart_of_points(seam)
    sc[:, 1] = sc[:, 1].mean()           # a straight layout seam (the lips bulge forward)
    lift = np.stack([0 * j, MOUTH_LENS_MM * np.sin((j + 1) * math.pi / 2) ** 0.7], 1)
    upper, lower = sc + lift, sc - lift
    th = np.linspace(-math.pi, math.pi, NECK_N, endpoint=False) + math.pi / NECK_N
    neck = np.stack([0.058 * np.sin(th), np.full(NECK_N, -0.153), -0.035 + 0.062 * np.cos(th)], 1)
    pts = {"nose_tip": [0, -0.037, 0.073], "subnasale": [0, -0.058, 0.051], "sellion": [0, 0.0, 0.029],
           "glabella": [0, 0.011, 0.028], "pogonion": [0, -0.107, 0.057], "menton": [0, -0.121, 0.052],
           "crown": [0, 0.123, -0.07], "ear_right": [-0.099, -0.002, -0.078], "ear_left": [0.099, -0.002, -0.078]}
    return {"eyes": eyes, "mouth_chart_upper": upper, "mouth_chart_lower": lower, "seam": seam, "neck": neck,
            "points": {k: np.asarray(v, np.float64) for k, v in pts.items()}}


def mouth_loop(upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Ring 0 of the mouth from two (MOUTH_HALF+1)-point lip lines sharing
    their corners (right -> left): right corner, upper lip, left corner,
    lower lip back to the right."""
    return np.concatenate([upper, lower[-2:0:-1]])


# ---- rings (canonical and subject alike) ---------------------------------------------

def _ccw(poly):
    return poly if signed_area(poly) > 0 else None


def rings_chart(eye_loops: dict, mouth_chart: np.ndarray, neck_loop: np.ndarray, center=CENTER) -> dict:
    """Chart positions of every ring vertex, from ring-0 loops: eyes and neck
    in 3D (head frame), the mouth already in the chart. Returns
    {name: (K, N, 2)}; outward offsets follow *_RING_MM."""
    out = {}
    for side, loop in eye_loops.items():
        c = chart_of_points(loop, center)
        sign = 1.0 if signed_area(c) > 0 else -1.0
        out[f"eye_{side}"] = np.stack([offset_loop(c, d, sign) for d in EYE_RING_MM])
    c = np.asarray(mouth_chart, np.float64)
    area = signed_area(c)
    if abs(area) < 1e-6:
        # a closed mouth (both lips on the seam): the ring runs right -> left
        # along the upper lip, i.e. towards +x along the top: clockwise
        sign = -1.0
    else:
        sign = 1.0 if area > 0 else -1.0
    out["mouth"] = np.stack([offset_loop(c, d, sign) for d in MOUTH_RING_MM])
    c = chart_of_points(neck_loop, center)
    sign = 1.0 if signed_area(c) > 0 else -1.0
    # the neck's hole is the region inside ring 0: rings grow away from it
    out["neck"] = np.stack([offset_loop(c, d, sign) for d in NECK_RING_MM])
    return out


# ---- template --------------------------------------------------------------------------

@dataclass
class Template:
    chart: np.ndarray           # (V, 2) canonical chart mm (inner rings: inset positions)
    kind: np.ndarray            # (V,) KIND
    group: np.ndarray           # (V,) 0 right eye, 1 left eye, 2 mouth, 3 neck, -1 none
    ring: np.ndarray            # (V,) ring index (negative inside), 0 for free
    sample: np.ndarray          # (V,) index along the ring
    tris: np.ndarray            # (T, 3)
    tri_mat: np.ndarray         # (T,) 0 skin, 1 mouth
    uv: np.ndarray              # (U, 2)
    tri_uv: np.ndarray          # (T, 3) indices into uv
    rings: dict = field(default_factory=dict)   # name -> (K, N) vertex ids (inner rings first)
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.kind)

    def ring_ids(self, name: str, k: int) -> np.ndarray:
        """Vertex ids of ring k (k < 0: inner) of a feature."""
        inner = {"eye_right": EYE_INNER, "eye_left": EYE_INNER, "mouth": MOUTH_INNER, "neck": 0}[name]
        return self.rings[name][k + inner]

    def save(self, path: Path):
        arrays = {k: getattr(self, k) for k in ("chart", "kind", "group", "ring", "sample", "tris", "tri_mat",
                                                 "uv", "tri_uv")}
        for name, ids in self.rings.items():
            arrays[f"rings_{name}"] = ids
        np.savez_compressed(path, info=json.dumps(self.info), **arrays)

    @classmethod
    def load(cls, path: Path) -> "Template":
        z = np.load(path)
        rings = {k[len("rings_"):]: z[k] for k in z.files if k.startswith("rings_")}
        return cls(*(z[k] for k in ("chart", "kind", "group", "ring", "sample", "tris", "tri_mat", "uv", "tri_uv")),
                   rings=rings, info=json.loads(str(z["info"])))


def _spacing(chart: np.ndarray) -> np.ndarray:
    """Target vertex spacing (chart mm): fine on the face, coarse behind."""
    # face ellipse: from above the brows to under the chin, ear to ear-ish
    cx, cy, ax, ay = 0.0, -30.0, 72.0, 92.0
    q = np.hypot((chart[:, 0] - cx) / ax, (chart[:, 1] - cy) / ay)
    return FACE_SPACING + (BACK_SPACING - FACE_SPACING) * smoothstep(1.0, 1.5, q)


def _fibonacci(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def _thin(cands: np.ndarray, spacing: np.ndarray, fixed: np.ndarray, fixed_spacing: np.ndarray,
          rng) -> np.ndarray:
    """Greedy blue-noise thinning on the unit sphere: keep a candidate if no
    kept point (or fixed point) is closer than the local spacing."""
    from scipy.spatial import cKDTree
    order = rng.permutation(len(cands))
    cell = spacing.max() / R0
    grid: dict = {}

    def key(p):
        return tuple(np.floor(p / cell).astype(int))

    kept = []
    tree_f = cKDTree(fixed) if len(fixed) else None
    for i in order:
        p = cands[i]
        s = spacing[i] / R0
        if tree_f is not None:
            d, j = tree_f.query(p)
            if d < 0.75 * max(s, fixed_spacing[j] / R0):
                continue
        k = key(p)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q, sq in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if np.dot(p - q, p - q) < (0.5 * (s + sq)) ** 2 * 0.81:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault(k, []).append((p, s))
            kept.append(i)
    return np.asarray(kept, np.int64)


def _hull_tris(dirs: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull
    t = ConvexHull(dirs).simplices.astype(np.int64)
    n = np.cross(dirs[t[:, 1]] - dirs[t[:, 0]], dirs[t[:, 2]] - dirs[t[:, 0]])
    flip = (n * dirs[t].mean(1)).sum(1) < 0
    t[flip] = t[flip][:, [0, 2, 1]]
    return t


def build(seed: int = 7) -> Template:
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    cf = canonical_features()
    loop = mouth_loop(cf["mouth_chart_upper"], cf["mouth_chart_lower"])
    rc = rings_chart(cf["eyes"], loop, cf["neck"])
    names = ["eye_right", "eye_left", "mouth", "neck"]
    ring_chart, ring_meta = [], []
    for g, name in enumerate(names):
        kinds = {"eye_right": 1, "eye_left": 1, "mouth": 2, "neck": 3}[name]
        for k, ring in enumerate(rc[name]):
            ring_chart.append(ring)
            ring_meta.append(np.stack([np.full(len(ring), kinds), np.full(len(ring), g), np.full(len(ring), k),
                                       np.arange(len(ring))], 1))
    ring_chart = np.concatenate(ring_chart)
    ring_meta = np.concatenate(ring_meta)
    ring_dirs = from_chart(ring_chart)
    # candidates, minus the ring patches (inside each feature's outermost ring)
    cands = _fibonacci(60000)
    cc = to_chart(cands)
    sp = _spacing(cc)
    inside = np.zeros(len(cands), bool)
    for name in names:
        outer = rc[name][-1]
        if name == "neck":
            inside |= point_in_polygon(cc, rc[name][0])
            continue
        inside |= point_in_polygon(cc, outer)
    cands, sp = cands[~inside], sp[~inside]
    ring_sp = _spacing(ring_chart)
    keep = _thin(cands, sp, ring_dirs, ring_sp, rng)
    free = cands[keep]
    dirs = np.concatenate([ring_dirs, free])
    nr = len(ring_dirs)
    meta = np.concatenate([ring_meta, np.stack([np.zeros(len(free)), -np.ones(len(free)), np.zeros(len(free)),
                                                np.zeros(len(free))], 1).astype(ring_meta.dtype)])

    def hole_tris(t):
        """Triangles spanning a hole: all corners on its ring 0, or (large
        holes such as the neck's cap) the chart centroid inside ring 0."""
        bad = np.zeros(len(t), bool)
        cen = to_chart(normalize(dirs[t].mean(1)))
        for g, name in enumerate(names):
            ring0 = (meta[:, 1] == g) & (meta[:, 2] == 0)
            bad |= ring0[t].all(1)
            bad |= ring0[t].any(1) & point_in_polygon(cen, rc[name][0])
        return bad

    # relax the free points (a few Laplacian steps on the sphere), then triangulate
    for _ in range(4):
        t = _hull_tris(dirs)
        t = t[~hole_tris(t)]
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        acc = np.zeros_like(dirs)
        cnt = np.zeros(len(dirs))
        np.add.at(acc, e[:, 0], dirs[e[:, 1]])
        np.add.at(cnt, e[:, 0], 1)
        avg = normalize(acc / np.maximum(cnt, 1)[:, None])
        mv = np.arange(len(dirs)) >= nr
        dirs[mv] = normalize(dirs[mv] + 0.5 * (avg[mv] - dirs[mv]))
    tris = _hull_tris(dirs)
    tris = tris[~hole_tris(tris)]
    # drop free points that ended up inside a hole (none expected) and unused ones
    used = np.zeros(len(dirs), bool)
    used[tris] = True
    chart = to_chart(dirs)
    kind = meta[:, 0].astype(np.int64)
    group = meta[:, 1].astype(np.int64)
    ring = meta[:, 2].astype(np.int64)
    sample = meta[:, 3].astype(np.int64)
    if not used.all():
        remap = np.cumsum(used) - 1
        tris = remap[tris]
        chart, kind, group, ring, sample = (a[used] for a in (chart, kind, group, ring, sample))
    V = len(chart)
    tri_mat = np.zeros(len(tris), np.int64)
    # ring vertex ids
    rings = {}
    for g, name in enumerate(names):
        K = rc[name].shape[0]
        N = rc[name].shape[1]
        ids = np.full((K, N), -1, np.int64)
        m = np.flatnonzero((group == g) & (kind != 0))
        ids[ring[m], sample[m]] = m
        rings[name] = ids
    # inner strips
    chart_l, kind_l, group_l, ring_l, sample_l = [chart], [kind], [group], [ring], [sample]
    new_tris, new_mat = [tris], [tri_mat]
    nxt = V

    def add_ring(g, kind_v, k, pts_chart):
        nonlocal nxt
        n = len(pts_chart)
        ids = np.arange(nxt, nxt + n)
        nxt += n
        chart_l.append(pts_chart)
        kind_l.append(np.full(n, kind_v))
        group_l.append(np.full(n, g))
        ring_l.append(np.full(n, k))
        sample_l.append(np.arange(n))
        return ids

    def strip(a, b, mat):
        n = len(a)
        j = np.arange(n)
        j1 = (j + 1) % n
        # a is the outer ring, b the next one inwards. The strips fold behind
        # the skin (around the lid margin / into the mouth), so the sheet's
        # outside faces the eyeball / the oral cavity: clockwise in the chart.
        t = np.concatenate([np.stack([a[j], a[j1], b[j]], 1), np.stack([b[j], a[j1], b[j1]], 1)])
        new_tris.append(t)
        new_mat.append(np.full(len(t), mat))

    for g, name in enumerate(names[:3]):
        ring0 = rings[name][0]
        c0 = chart[ring0]
        sign = 1.0 if signed_area(c0) > 0 else -1.0
        inner = EYE_INNER if name.startswith("eye") else MOUTH_INNER
        prev = ring0
        ids_inner = []
        cen = c0.mean(0)
        for k in range(1, inner + 1):
            # inset positions for the layout/UVs only (ring -k)
            f = 1.0 - k / (inner + 1.5)
            pts = cen + (c0 - cen) * f
            ids = add_ring(g, KIND["eye_inner"] if inner == EYE_INNER else KIND["mouth_inner"], -k, pts)
            if sign < 0:
                strip(prev[::-1], ids[::-1], 0 if inner == EYE_INNER else 1)
            else:
                strip(prev, ids, 0 if inner == EYE_INNER else 1)
            prev = ids
            ids_inner.append(ids)
        if name == "mouth":
            cap = add_ring(g, KIND["cap"], -(inner + 1), cen[None])
            n = len(prev)
            j = np.arange(n)
            t = np.stack([prev[j], prev[(j + 1) % n], np.full(n, cap[0])], 1)
            if sign < 0:
                t = t[:, [0, 2, 1]]
            new_tris.append(t)
            new_mat.append(np.ones(n, np.int64))
        rings[name] = np.concatenate([np.stack(ids_inner[::-1]), rings[name]])
    chart = np.concatenate(chart_l)
    kind = np.concatenate(kind_l).astype(np.int64)
    group = np.concatenate(group_l).astype(np.int64)
    ring = np.concatenate(ring_l).astype(np.int64)
    sample = np.concatenate(sample_l).astype(np.int64)
    tris = np.concatenate(new_tris)
    tri_mat = np.concatenate(new_mat)
    uv, tri_uv = _uv_layout(chart, kind, group, ring, sample, tris, tri_mat)
    t = Template(chart, kind, group, ring, sample, tris, tri_mat, uv, tri_uv, rings,
                 info={"version": VERSION, "seed": seed, "vertices": int(len(kind)), "triangles": int(len(tris)),
                       "skin_triangles": int((tri_mat == 0).sum()), "mouth_triangles": int((tri_mat == 1).sum())})
    return t


FRONT_MAX_DEG = 100.0
UV_FRONT = (0.37, 0.5, 0.36)     # centre u, v, radius
UV_BACK = (0.86, 0.27, 0.135)


def _uv_layout(chart, kind, group, ring, sample, tris, tri_mat):
    """Face-vertex UVs: skin in two azimuthal charts (front about +Z, back
    about -Z, seam where the angle from +Z passes FRONT_MAX_DEG); the mouth
    interior in its own unit square (u around the ring, v with depth)."""
    dirs = from_chart(chart)
    a = np.degrees(np.arccos(np.clip(dirs[:, 2], -1, 1)))
    front_r = FRONT_MAX_DEG / 180 * math.pi * R0
    back_r = (180 - FRONT_MAX_DEG) / 180 * math.pi * R0
    uv_f = np.array(UV_FRONT[:2]) + chart / front_r * UV_FRONT[2]
    # back chart: azimuthal about -Z, mirrored in x so it is not flipped
    d_b = dirs * np.array([-1.0, 1.0, -1.0])
    cb = to_chart(d_b)
    uv_b = np.array(UV_BACK[:2]) + cb / back_r * UV_BACK[2]
    V = len(chart)
    tri_front = a[tris].mean(1) < FRONT_MAX_DEG
    inner_skin = np.isin(kind, [KIND["eye_inner"]])
    tri_front |= inner_skin[tris].any(1)
    # mouth UVs: u = sample / N along the ring, v = depth
    mouth_v = np.clip(-ring / (MOUTH_INNER + 1), 0, 1)
    uv_m = np.stack([sample / MOUTH_N, 0.05 + 0.9 * mouth_v], 1)
    cap = kind == KIND["cap"]
    uv_m[cap] = [0.5, 0.95]
    uv = np.concatenate([uv_f, uv_b, uv_m])
    tri_uv = np.where(tri_front[:, None], tris, tris + V)
    m = tri_mat == 1
    tri_uv[m] = tris[m] + 2 * V
    # the mouth ring wraps: the last quad column spans u = (N-1)/N .. 1
    tm = tri_uv[m] - 2 * V
    s = sample[tm]
    wrap = (s.max(1) - s.min(1)) > MOUTH_N // 2
    extra = []
    for ti in np.flatnonzero(wrap):
        row = tri_uv[m][ti].copy()
        for c in range(3):
            v = row[c] - 2 * V
            if sample[v] == 0 and kind[v] != KIND["cap"]:
                extra.append([1.0, uv_m[v, 1]])
                row[c] = len(uv) + len(extra) - 1
        idx = np.flatnonzero(m)[ti]
        tri_uv[idx] = row
    if extra:
        uv = np.concatenate([uv, np.asarray(extra)])
    # compact
    used, inv = np.unique(tri_uv.reshape(-1), return_inverse=True)
    return uv[used], inv.reshape(-1, 3)


_CACHE: dict = {}


def get(cache_dir: Path | None = None) -> Template:
    key = f"v{VERSION}"
    if key in _CACHE:
        return _CACHE[key]
    path = None
    if cache_dir is not None:
        src = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        path = Path(cache_dir) / f"template-{key}-{src}.npz"
        if path.exists():
            t = Template.load(path)
            _CACHE[key] = t
            return t
    t = build()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        t.save(path)
    _CACHE[key] = t
    return t
