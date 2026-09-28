"""Levels of detail of the head: coarser templates driven by LOD0's rig.

LOD k is a template built with a coarser Layout (template.py): ring sample
counts that divide LOD0's, a subset of LOD0's rings, larger free-point
spacing. Every LOD vertex is a fixed combination of LOD0 vertices:
- ring, inner-strip and cap vertices map exactly: (group, ring, sample) ->
  (group, LOD0 ring with the same offset, sample * N0 / Nk);
- free vertices: barycentrics of the LOD0 skin triangle containing their
  canonical chart position.
Any per-vertex rig data (positions, skin weights, blendshapes incl. the ML
correctives, contact sets) transfers through that (Vk x V0) matrix, so all
LODs animate from the same controls and rig.json. Ring vertices keep LOD0's
chart positions, so the UV atlas (a function of the chart) is shared.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import template as T

LAYOUTS = {
    1: T.Layout(eye_n=24, mouth_half=16, neck_n=24, eye_rings=(0, 1, 3, 5, 7), mouth_rings=(0, 1, 2, 4, 6, 8),
                neck_rings=(0, 1), face_spacing=2.8, back_spacing=7.5, mouth_lens_mm=6.5),
    2: T.Layout(eye_n=12, mouth_half=8, neck_n=12, eye_rings=(0, 2, 5, 7), mouth_rings=(0, 2, 5, 8),
                neck_rings=(0,), face_spacing=5.2, back_spacing=12.0, mouth_lens_mm=8.0, neck_shrink=0.97),
}
NAMES = ("eye_right", "eye_left", "mouth", "neck")


def get(level: int, cache_dir=None) -> T.Template:
    lay = LAYOUTS[level]
    path = None
    if cache_dir is not None:
        src = hashlib.sha256(Path(T.__file__).read_bytes() + Path(__file__).read_bytes()).hexdigest()[:12]
        path = Path(cache_dir) / f"template-lod{level}-{src}.npz"
        if path.exists():
            return T.Template.load(path)
    t = T.build(layout=lay)
    t.info["lod"] = level
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        t.save(path)
    return t


def _ring_lists(lay: T.Layout) -> dict:
    return {"eye_right": lay.eye_rings, "eye_left": lay.eye_rings, "mouth": lay.mouth_rings, "neck": lay.neck_rings}


def mapping(t0: T.Template, tk: T.Template, level: int) -> tuple[np.ndarray, np.ndarray]:
    """(idx (Vk, 3), w (Vk, 3)): LOD vertex = sum_j w_j * LOD0 vertex idx_j."""
    from scipy.spatial import cKDTree
    lay, lay0 = LAYOUTS[level], T.Layout()
    rings = _ring_lists(lay)
    n0 = {"eye_right": lay0.eye_n, "eye_left": lay0.eye_n, "mouth": lay0.mouth_n, "neck": lay0.neck_n}
    nk = {"eye_right": lay.eye_n, "eye_left": lay.eye_n, "mouth": lay.mouth_n, "neck": lay.neck_n}
    Vk = tk.n
    idx = np.zeros((Vk, 3), np.int64)
    w = np.zeros((Vk, 3))
    exact = np.zeros(Vk, bool)
    cap0 = np.flatnonzero(t0.kind == T.KIND["cap"])
    for i in range(Vk):
        kind, g = tk.kind[i], tk.group[i]
        if kind == T.KIND["cap"]:
            idx[i, 0], w[i, 0], exact[i] = cap0[0], 1.0, True
        elif kind != T.KIND["free"]:
            name = NAMES[g]
            r = tk.ring[i]
            r0 = rings[name][r] if r >= 0 else r
            s0 = tk.sample[i] * (n0[name] // nk[name])
            idx[i, 0], w[i, 0], exact[i] = t0.ring_ids(name, r0)[s0], 1.0, True
    # free vertices: the LOD0 skin triangle containing their canonical chart point
    skin = t0.tris[t0.tri_mat == 0]
    tri_c = t0.chart[skin]
    tree = cKDTree(tri_c.mean(1))
    free = np.flatnonzero(~exact)
    _, cand = tree.query(tk.chart[free], k=12)
    for n, i in enumerate(free):
        p = tk.chart[i]
        best, best_b, best_err = None, None, np.inf
        for c in cand[n]:
            a, b, cc = tri_c[c]
            m = np.array([[b[0] - a[0], cc[0] - a[0]], [b[1] - a[1], cc[1] - a[1]]])
            try:
                u, v = np.linalg.solve(m, p - a)
            except np.linalg.LinAlgError:
                continue
            bc = np.array([1 - u - v, u, v])
            err = -min(bc.min(), 0.0)
            if err < best_err:
                best, best_b, best_err = c, bc, err
                if err == 0:
                    break
        bc = np.clip(best_b, 0, None)
        idx[i] = skin[best]
        w[i] = bc / bc.sum()
    return idx, w


def apply(idx: np.ndarray, w: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Per-vertex data (V0, ...) -> (Vk, ...)."""
    d = np.asarray(data)
    return (w.reshape(len(idx), 3, *([1] * (d.ndim - 1))) * d[idx]).sum(1)


def snap_uvs(t0: T.Template, tk: T.Template, idx: np.ndarray, w: np.ndarray) -> T.Template:
    """Give exactly mapped vertices LOD0's chart positions and redo the UVs,
    so the LOD shares LOD0's texture atlas."""
    exact = w[:, 0] == 1.0
    tk.chart = tk.chart.copy()
    tk.chart[exact] = t0.chart[idx[exact, 0]]
    lay = LAYOUTS[tk.info.get("lod", 1)]
    tk.uv, tk.tri_uv = T._uv_layout(tk.chart, tk.kind, tk.group, tk.ring, tk.sample, tk.tris, tk.tri_mat, lay.mouth_n)
    return tk


def contacts_viz(viz0: dict, idx: np.ndarray, w: np.ndarray, tris_k: np.ndarray) -> dict:
    """LOD0's contact model restricted to the LOD's exactly mapped vertices."""
    from . import contacts
    exact = w[:, 0] == 1.0
    to_k = {int(j): i for i, j in zip(np.flatnonzero(exact), idx[exact, 0])}
    out = {k: v for k, v in viz0.items() if k not in ("eye", "lip_ids", "pairs_upper", "pairs_lower", "verts",
                                                       "nbr_ptr", "nbr_idx")}
    out["eye"] = [dict(e, ids=[to_k[v] for v in e["ids"] if v in to_k]) for e in viz0["eye"]]
    out["lip_ids"] = [to_k[v] for v in viz0["lip_ids"] if v in to_k]
    pu, pl = [], []
    for u, l in zip(viz0["pairs_upper"], viz0["pairs_lower"]):
        if u in to_k and l in to_k:
            pu.append(to_k[u])
            pl.append(to_k[l])
    out["pairs_upper"], out["pairs_lower"] = pu, pl
    out.update(contacts.graph(out, tris_k))
    return out
