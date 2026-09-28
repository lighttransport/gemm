"""Post-skinning contact projection: exact contacts after the deformer.

The ML correctives remove most contacts smoothly; this pass makes them exact
on the final, skinned surface. Per frame, `ITERATIONS` rounds of:
1. eye: lid vertices closer to their (moving) eyeball centre than their
   threshold are pushed out radially onto it;
2. lips: each upper/lower lip pair closer than its floor along the mean of
   the head's and the jaw's up axes is separated symmetrically;
3. spheres: lip/vestibule vertices inside a tooth or tongue sphere's
   threshold are pushed out radially, sphere by sphere (last: a lip through
   a tooth is the visible failure when constraints conflict).
The displacement is then smoothed over the contact vertices' mesh graph
(SMOOTH_STEPS Jacobi steps) so neighbours follow instead of leaving a jagged
edge, and the projection runs once more, so the result stays exact.
Thresholds are rest-relative, as in the training contacts (mldeformer): a
vertex never ends closer than at rest or than the margin, whichever is smaller.
Every vertex belongs to at most one eye, one lip set and one pair, so each
phase is independent per vertex: the C, CUDA, Vulkan and JS versions run the
same arithmetic in the same order.

Tensors (package prefix "contact."; metres, bind space):
    eye_ids (Ne,) eye_joint (Ne,) eye_center (Ne, 3) eye_thr (Ne,)
    lip_ids (L,) sph_center (T, 3) sph_joint (T, 2) sph_weight (T, 2) sph_thr (L, T)
    pair_u (P,) pair_l (P,) pair_floor (P,) up_joints (2,)   [head, jaw]
    verts (Nc,) nbr_ptr (Nc+1,) nbr_idx (E,)   contact vertices and their graph (CSR)
"""
from __future__ import annotations

import numpy as np

ITERATIONS = 4
SPHERE_PASSES = 8
SMOOTH_STEPS = 2


def graph(viz: dict, tris: np.ndarray) -> dict:
    """The contact vertices (sorted) and their mesh neighbours among them (CSR),
    for smoothing the projection's displacement. Added to viz["contacts"]."""
    ids = set(viz["lip_ids"]) | set(viz["pairs_upper"]) | set(viz["pairs_lower"])
    for e in viz["eye"]:
        ids |= set(e["ids"])
    verts = np.array(sorted(ids))
    pos = {v: i for i, v in enumerate(verts)}
    nb = [set() for _ in verts]
    for a, b in np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]):
        if a in pos and b in pos:
            nb[pos[a]].add(pos[b])
            nb[pos[b]].add(pos[a])
    ptr = np.concatenate([[0], np.cumsum([len(n) for n in nb])])
    idx = np.concatenate([sorted(n) for n in nb]) if len(verts) else np.zeros(0)
    return {"verts": verts.tolist(), "nbr_ptr": ptr.astype(int).tolist(), "nbr_idx": np.asarray(idx, int).tolist()}


def tensors(viz: dict, rest: np.ndarray) -> dict:
    """From viz.json's contact export and the welded rest positions."""
    mm = 1e-3
    eye_ids, eye_joint, eye_center, eye_thr = [], [], [], []
    for e in viz["eye"]:
        ids = np.asarray(e["ids"])
        c = np.asarray(e["center"])
        d = np.linalg.norm(rest[ids] - c, axis=1)
        eye_ids.append(ids)
        eye_joint.append(np.full(len(ids), e["joint"]))
        eye_center.append(np.tile(c, (len(ids), 1)))
        eye_thr.append(np.minimum((e["radius_mm"] + viz["margins_mm"]["eye"]) * mm, d))
    lip = np.asarray(viz["lip_ids"])
    C, J2, W2, R = [], [], [], []
    for S in viz["spheres"].values():
        C.append(np.asarray(S["centers"]))
        J2.append(np.asarray(S["joints"]))
        W2.append(np.asarray(S["weights"]))
        R.append(np.asarray(S["radius_mm"]))
    C, J2, W2, R = (np.concatenate(a) for a in (C, J2, W2, R))
    d = np.linalg.norm(rest[lip][:, None] - C[None], axis=-1)
    thr = np.minimum((R[None] + viz["margins_mm"]["sphere"]) * mm, d)
    up, lo = np.asarray(viz["pairs_upper"]), np.asarray(viz["pairs_lower"])
    floor = np.minimum(rest[up, 1] - rest[lo, 1], 0.0)
    return {"contact.eye_ids": np.concatenate(eye_ids).astype(np.int32),
            "contact.eye_joint": np.concatenate(eye_joint).astype(np.int32),
            "contact.eye_center": np.concatenate(eye_center).astype(np.float32),
            "contact.eye_thr": np.concatenate(eye_thr).astype(np.float32),
            "contact.lip_ids": lip.astype(np.int32), "contact.sph_center": C.astype(np.float32),
            "contact.sph_joint": J2.astype(np.int32), "contact.sph_weight": W2.astype(np.float32),
            "contact.sph_thr": thr.astype(np.float32),
            "contact.pair_u": up.astype(np.int32), "contact.pair_l": lo.astype(np.int32),
            "contact.pair_floor": floor.astype(np.float32),
            "contact.up_joints": np.asarray([viz["head"], viz["jaw"]], np.int32),
            "contact.verts": np.asarray(viz["verts"], np.int32),
            "contact.nbr_ptr": np.asarray(viz["nbr_ptr"], np.int32),
            "contact.nbr_idx": np.asarray(viz["nbr_idx"], np.int32)}


def project(pos: np.ndarray, skin: np.ndarray, t: dict, iterations: int = ITERATIONS,
            smooth: int = SMOOTH_STEPS) -> np.ndarray:
    """pos (V, 3) posed, skin (J, 4, 4): the projected positions (a copy).
    Project; smooth the displacement of the contact vertices over their
    graph (Jacobi: d <- (d + mean of the neighbours' d) / 2) so neighbours
    follow; project again (exact)."""
    x = np.array(pos, np.float64)
    S = np.asarray(skin, np.float64)
    g = _geometry(S, t)
    _solve(x, t, g, iterations)
    if smooth and "contact.verts" in t:
        cv, ptr, nbr = t["contact.verts"], t["contact.nbr_ptr"], t["contact.nbr_idx"]
        x0 = np.asarray(pos, np.float64)[cv]
        cnt = np.diff(ptr)
        rows = np.repeat(np.arange(len(cv)), cnt)
        for _ in range(smooth):
            d = x[cv] - x0
            acc = np.zeros_like(d)
            np.add.at(acc, rows, d[nbr])
            mean = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], d)
            x[cv] = x0 + 0.5 * (d + mean)
        _solve(x, t, g, iterations)
    return x


def _geometry(S, t):
    ei, ej = t["contact.eye_ids"], t["contact.eye_joint"]
    ec = np.einsum("nij,nj->ni", S[ej][:, :3, :3], t["contact.eye_center"]) + S[ej][:, :3, 3]
    Ms = (t["contact.sph_weight"][:, :, None, None] * S[t["contact.sph_joint"]]).sum(1)      # (T, 4, 4)
    sc = np.einsum("tij,tj->ti", Ms[:, :3, :3], t["contact.sph_center"]) + Ms[:, :3, 3]
    hj, jj = t["contact.up_joints"]
    up = S[hj, :3, 1] + S[jj, :3, 1]
    return ec, sc, up / np.linalg.norm(up)


def _solve(x, t, g, iterations):
    ec, sc, up = g
    ei = t["contact.eye_ids"]
    lip = t["contact.lip_ids"]
    pu, pl, fl = t["contact.pair_u"], t["contact.pair_l"], t["contact.pair_floor"]
    thr_s = t["contact.sph_thr"]
    for _ in range(iterations):                    # stop early: an iteration that moves nothing is final
        any_move = False
        v = x[ei] - ec
        d = np.linalg.norm(v, axis=1)
        m = d < t["contact.eye_thr"]
        any_move |= bool(m.any())
        x[ei[m]] = ec[m] + v[m] * (t["contact.eye_thr"][m] / np.maximum(d[m], 1e-12))[:, None]
        sep = (x[pu] - x[pl]) @ up
        dl = np.maximum(fl - sep, 0.0) * 0.5
        any_move |= bool((dl > 0).any())
        x[pu] += dl[:, None] * up
        x[pl] -= dl[:, None] * up
        q = x[lip]
        active = np.ones(len(lip), bool)          # per vertex: passes until one pushes nothing
        for _ in range(SPHERE_PASSES):            # (overlapping spheres push into each other)
            moved = np.zeros(len(lip), bool)
            for k in range(len(sc)):              # sphere by sphere, in order
                v = q - sc[k]
                d = np.linalg.norm(v, axis=1)
                m = active & (d < thr_s[:, k])
                q[m] = sc[k] + v[m] * (thr_s[m, k] / np.maximum(d[m], 1e-12))[:, None]
                moved |= m
            any_move |= bool(moved.any())
            active &= moved
            if not active.any():
                break
        x[lip] = q
        if not any_move:
            break


def depths(pos: np.ndarray, skin: np.ndarray, t: dict) -> dict:
    """Penetration counts (> 0.05 mm) per class, as the viewer and training count them."""
    x = np.asarray(pos, np.float64)
    S = np.asarray(skin, np.float64)
    ei, ej = t["contact.eye_ids"], t["contact.eye_joint"]
    ec = np.einsum("nij,nj->ni", S[ej][:, :3, :3], t["contact.eye_center"]) + S[ej][:, :3, 3]
    eye = int(((t["contact.eye_thr"] - np.linalg.norm(x[ei] - ec, axis=1)) > 5e-5).sum())
    Ms = (t["contact.sph_weight"][:, :, None, None] * S[t["contact.sph_joint"]]).sum(1)
    sc = np.einsum("tij,tj->ti", Ms[:, :3, :3], t["contact.sph_center"]) + Ms[:, :3, 3]
    d = np.linalg.norm(x[t["contact.lip_ids"]][:, None] - sc[None], axis=-1)
    sph = int(((t["contact.sph_thr"] - d).max(1) > 5e-5).sum())
    hj, jj = t["contact.up_joints"]
    up = S[hj, :3, 1] + S[jj, :3, 1]
    up /= np.linalg.norm(up)
    sep = (x[t["contact.pair_u"]] - x[t["contact.pair_l"]]) @ up
    lips = int(((t["contact.pair_floor"] - sep) > 5e-5).sum())
    return {"eye": eye, "spheres": sph, "lips": lips}
