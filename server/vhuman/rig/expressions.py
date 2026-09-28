"""Procedural expression shapes on a fitted template (head frame, metres).

Each shape is an original analytic displacement field over the facial
coordinates of fields.py, scaled by the subject's size:
- lids rotate about the fitted eyeball centre (blink, wide, squint, gaze
  follow), so the margins slide over the eye instead of cutting into it;
  a blink closes each upper-lid sample onto its lower-lid partner;
- lip shapes act on the lip rings (the seam splits them exactly) with
  falloffs over the orbicularis loops;
- corners, cheeks, brows and the nose use smooth neighbourhoods of the
  detected landmarks;
- mouthClose is solved against the jaw's skinning so that jawOpen +
  mouthClose meet the lips halfway.
A final pass smooths each field lightly over the surface (PyTorch sparse
Laplacian when available) so no shape tears across the ring patches.
"""
from __future__ import annotations

import math

import numpy as np

from . import template as T
from .common import edges, normalize, rotation, smoothstep
from .fields import Fields
from .rigdef import euler_matrix

UP = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])


def _eye(F: Fields, side: str) -> dict:
    return next(e for e in F.feat.eyes if e["side"] == side)


def _lid_angles(F: Fields, side: str) -> np.ndarray:
    """Per lid sample j (upper half), the signed angle about the eye's x axis
    that brings the upper margin onto the lower margin."""
    e = _eye(F, side)
    c, ax = e["center"], e["rotation"][:, 0]
    tm = F.tmpl
    ring0 = tm.ring_ids(f"eye_{side}", 0)
    N = T.EYE_N
    ang = np.zeros(N)
    for j in range(1, N // 2):
        a = F.pos[ring0[j]] - c
        b = F.pos[ring0[N - j]] - c
        a -= (a @ ax) * ax
        b -= (b @ ax) * ax
        s = np.dot(np.cross(a, b), ax)
        ang[j] = math.atan2(s, np.dot(a, b))
    return ang


# lid ring falloffs: ring index -> weight (inner rings follow the margin)
UPPER_FALL = {-2: 1.0, -1: 1.0, 0: 1.0, 1: 0.92, 2: 0.78, 3: 0.58, 4: 0.36, 5: 0.18, 6: 0.07, 7: 0.0}
LOWER_FALL = {-2: 1.0, -1: 1.0, 0: 1.0, 1: 0.85, 2: 0.62, 3: 0.4, 4: 0.22, 5: 0.1, 6: 0.03, 7: 0.0}


def _rotate_lid(F: Fields, side: str, upper: bool, angle_of_sample, fall=None, axis_col=0) -> np.ndarray:
    """Delta rotating one lid (ring vertices of one half) about the eye centre.
    angle_of_sample: (EYE_N,) angles per sample (only the half's samples used)."""
    e = _eye(F, side)
    c, ax = e["center"], e["rotation"][:, axis_col]
    g = 0 if side == "right" else 1
    fall = fall or (UPPER_FALL if upper else LOWER_FALL)
    d = np.zeros_like(F.pos)
    N = T.EYE_N
    m = (F.eye_side == g) & (F.eye_upper == upper)
    corner = (F.eye_sample == 0) | (F.eye_sample == N // 2)
    m &= ~corner
    idx = np.flatnonzero(m)
    k = F.eye_k[idx]
    w = np.array([fall.get(int(kk), 0.0) for kk in k])
    j = F.eye_sample[idx]
    ang = np.asarray(angle_of_sample)[np.where(upper, j, (N - j) % N)] * w
    p = F.pos[idx] - c
    R = rotation(ax, ang)
    d[idx] = np.einsum("nij,nj->ni", R, p) - p
    return d


def build(F: Fields, skel: dict, skin_w: tuple) -> dict:
    """{name: (V, 3) delta} for every expression and corrective shape."""
    mm = F.mm
    pos = F.pos
    tm = F.tmpl
    S = {}
    left_w, right_w = F.side(True), F.side(False)
    # ---- eyes ------------------------------------------------------------------
    for side, sname in (("left", "Left"), ("right", "Right")):
        ang = _lid_angles(F, side)
        S[f"eyeBlink{sname}"] = _rotate_lid(F, side, True, 0.8 * ang) + _rotate_lid(F, side, False, -0.2 * ang)
        up = np.full(T.EYE_N, math.radians(-9.0))
        lo = np.full(T.EYE_N, math.radians(2.5))
        S[f"eyeWide{sname}"] = _rotate_lid(F, side, True, up) + _rotate_lid(F, side, False, lo)
        sq = _rotate_lid(F, side, False, -0.32 * ang) + _rotate_lid(F, side, True, 0.1 * ang)
        sq += F.gauss(F.cheek[side], 14)[:, None] * UP * 1.0 * mm
        S[f"eyeSquint{sname}"] = sq
        # gaze follow: the lids ride along with the eye (joints turn the eyeball)
        S[f"eyeLookDown{sname}"] = (_rotate_lid(F, side, True, np.full(T.EYE_N, math.radians(0.55 * 26)))
                                   + _rotate_lid(F, side, False, np.full(T.EYE_N, math.radians(0.35 * 26))))
        S[f"eyeLookUp{sname}"] = (_rotate_lid(F, side, True, np.full(T.EYE_N, math.radians(-0.6 * 22)))
                                 + _rotate_lid(F, side, False, np.full(T.EYE_N, math.radians(-0.3 * 22))))
        s_in = -1.0 if side == "left" else 1.0         # eye-local y rotation sign for looking in
        for nm, deg_ in (("In", 28 * s_in), ("Out", -32 * s_in)):
            a = np.full(T.EYE_N, math.radians(0.22 * deg_))
            S[f"eyeLook{nm}{sname}"] = (_rotate_lid(F, side, True, a, axis_col=1)
                                        + _rotate_lid(F, side, False, a, axis_col=1))
        S[f"corr_blink_lookDown_{sname[0]}"] = -_rotate_lid(F, side, True,
                                                            np.full(T.EYE_N, math.radians(0.55 * 26)))
    # ---- brows -------------------------------------------------------------------
    for side, sname in (("left", "Left"), ("right", "Right")):
        br = F.brow[side]
        d = np.min(np.linalg.norm(pos[:, None] - br[None], axis=-1), 1)
        near = np.exp(-(d / (13 * mm)) ** 2)
        # medialness along the brow: 1 at the inner end
        j = np.argmin(np.linalg.norm(pos[:, None] - br[None], axis=-1), 1)
        med = 1 - j / (len(br) - 1)
        above_eye = (F.eye_k >= 3) | (F.eye_k == 99)
        towards_mid = -np.sign(pos[:, 0])[:, None] * np.array([1.0, 0, 0])
        dd = (near * (0.7 + 0.3 * med))[:, None] * (-UP * 4.0 * mm + FWD * 0.6 * mm) \
            + (near * med)[:, None] * towards_mid * 2.5 * mm
        dd *= above_eye[:, None]
        S[f"browDown{sname}"] = dd
        outer = br[-1]
        S[f"browOuterUp{sname}"] = (F.gauss(outer, 15) * above_eye)[:, None] * UP * 4.5 * mm
    inner = 0.5 * (F.brow["left"][0] + F.brow["right"][0])
    bi = F.gauss(F.brow["left"][0], 15) + F.gauss(F.brow["right"][0], 15)
    fh = F.gauss(inner + UP * 25 * mm, 30)
    above = ((F.eye_k >= 3) | (F.eye_k == 99))
    S["browInnerUp"] = (np.clip(bi, 0, 1) * 5.0 * mm + fh * 2.0 * mm)[:, None] * UP * above[:, None]
    # ---- mouth ---------------------------------------------------------------
    lip = F.lip_k
    upper = F.upper
    inner_ring = lip < 0
    k_fall = np.where(lip <= 0, 1.0, np.clip(1 - lip / 7.0, 0, 1))
    k_fall = np.where(lip == 99, 0.0, k_fall)
    k_fall = np.where(lip == -1, 0.9, np.where(lip == -2, 0.5, np.where(lip <= -3, 0.0, k_fall)))
    lat = smoothstep(1.35, 0.85, np.abs(F.u))
    near_mouth = np.exp(-((F.u / 1.25) ** 2 + (F.h / (20 * mm)) ** 2))
    region = np.maximum(k_fall, near_mouth * (lip == 99))
    vermilion = np.where((lip >= -1) & (lip <= 3), 1.0, np.where(lip == 4, 0.4, 0.0))
    for side, sname in (("left", "Left"), ("right", "Right")):
        sx = 1.0 if side == "left" else -1.0
        cw = F.side(side == "left")
        corner = F.corner[side]
        out = np.array([sx, 0.0, 0.0])
        c14 = F.gauss(corner, 14)
        c16 = F.gauss(corner, 16)
        S[f"mouthSmile{sname}"] = (c16[:, None] * (UP * 7.0 + out * 4.5 - FWD * 3.5) * mm
                                   + F.gauss(F.cheek[side], 18)[:, None] * (UP * 3.5 + FWD * 1.8) * mm
                                   + (F.gauss(F.cheek[side], 12) * ((F.eye_k >= 1) & (F.eye_k <= 5)
                                                                    & ~F.eye_upper))[:, None] * UP * 0.8 * mm)
        S[f"mouthFrown{sname}"] = F.gauss(corner, 13)[:, None] * (-UP * 4.5 + out * 0.8 + FWD * 0.5) * mm
        S[f"mouthDimple{sname}"] = F.gauss(corner, 10)[:, None] * (-FWD * 3.0 + out * 1.5) * mm
        thin = (vermilion * cw * lat)[:, None] * np.where(upper, -1.0, 1.0)[:, None] * UP * 0.8 * mm
        S[f"mouthStretch{sname}"] = F.gauss(corner, 15)[:, None] * (out * 5.0 - UP * 2.0 - FWD * 1.0) * mm + thin
        press = (vermilion * cw * lat)[:, None] * (np.where(upper, -1.0, 1.0)[:, None] * UP * 1.0 - FWD * 0.7) * mm
        S[f"mouthPress{sname}"] = press
        half_lat = np.exp(-((F.u - sx * 0.45) / 0.55) ** 2)
        low = (~upper) & (lip != 99) | ((lip == 99) & (F.h < 0))
        S[f"mouthLowerDown{sname}"] = (half_lat * region * low * (lip > -3))[:, None] * (-UP * 4.0 + FWD * 0.5) * mm
        hi = upper & (lip != 99) | ((lip == 99) & (F.h > 0))
        S[f"mouthUpperUp{sname}"] = (half_lat * region * hi * (lip > -3))[:, None] * (UP * 4.0 + FWD * 0.5) * mm
        S[f"noseSneer{sname}"] = (F.gauss(F.ala[side], 8)[:, None] * (UP * 2.5 - FWD * 0.5) * mm
                                  + F.gauss(0.5 * (F.ala[side] + _eye(F, side)["center"]), 12)[:, None] * UP * 1.8 * mm
                                  + (np.exp(-((F.u - sx * 0.4) / 0.35) ** 2) * hi * region * (lip > -3))[:, None]
                                  * UP * 1.2 * mm)
        S[f"cheekSquint{sname}"] = (F.gauss(F.cheek[side], 15)[:, None] * UP * 2.5 * mm
                                    + _rotate_lid(F, side, False, np.full(T.EYE_N, math.radians(-4.0))))
        S[f"mouth{sname}"] = (np.exp(-((F.u / 1.6) ** 2 + (F.h / (24 * mm)) ** 2)) * (lip > -3))[:, None] \
            * out * 6.0 * mm
        S[f"corr_jawOpen_smile_{sname[0]}"] = c14[:, None] * (UP * 2.0 + out * 1.0) * mm
    lips = (region * lat * (lip > -3))[:, None]
    open_ = np.cos(np.clip(F.u, -1, 1) * math.pi / 2) ** 0.8
    to_mid = np.stack([-(pos[:, 0] - F.mouth_mid[0]), np.zeros(len(pos)), np.zeros(len(pos))], 1)
    S["mouthFunnel"] = (lips * FWD * 5.0 * mm * np.clip(1 - np.maximum(lip, 0) / 6.0, 0, 1)[:, None]
                        + (vermilion * open_ * lat)[:, None] * np.where(upper, 2.5, -3.0)[:, None] * UP * mm
                        + (region * (lip > -3))[:, None] * to_mid * 0.25)
    S["mouthPucker"] = (lips * FWD * 6.0 * mm * np.clip(1 - np.maximum(lip, 0) / 7.0, 0, 1)[:, None]
                        + (region * (lip > -3))[:, None] * to_mid * 0.4)
    roll = (vermilion * lat)[:, None]
    S["mouthRollLower"] = roll * (~upper)[:, None] * (-FWD * 3.0 + UP * 1.5) * mm
    S["mouthRollUpper"] = roll * upper[:, None] * (-FWD * 3.0 - UP * 1.5) * mm
    chin = F.gauss(F.feat.points["pogonion"], 15)
    lower_lip = (region * (~upper) * (lip > -3))[:, None]
    S["mouthShrugLower"] = lower_lip * (UP * 2.5 + FWD * 1.5) * mm + chin[:, None] * (UP * 1.5 + FWD * 1.0) * mm
    S["mouthShrugUpper"] = (region * upper * (lip > -3))[:, None] * (UP * 1.5 + FWD * 1.0) * mm
    # cheeks puff between the corners and the jaw's angle
    nrm = normalize(_normals(F))
    puff = np.zeros(len(pos))
    for side in ("left", "right"):
        sx = 1.0 if side == "left" else -1.0
        cp = F.corner[side] + np.array([sx * 16, 4, -14]) * mm
        puff += F.gauss(cp, 20)
    S["cheekPuff"] = (np.clip(puff, 0, 1) * (tm.kind != T.KIND["mouth_inner"]))[:, None] * nrm * 5.0 * mm \
        + lips * FWD * 1.0 * mm
    # ---- jaw ------------------------------------------------------------------------
    S["jawOpen"] = (np.exp(-((F.u / 2.2) ** 2 + ((F.h - 6 * mm) / (14 * mm)) ** 2)) * upper * (lip > -3))[:, None] \
        * (-UP * 0.8) * mm
    S["mouthClose"] = _mouth_close(F, skel, skin_w, region * (lip > -3) * lat)
    for k in list(S):
        S[k] = np.asarray(S[k], np.float64)
        S[k][~np.isfinite(S[k])] = 0.0
    return smooth_shapes(F, S)


def _normals(F: Fields):
    from .common import vertex_normals
    return vertex_normals(F.pos, F.tmpl.tris[F.tmpl.tri_mat == 0])


def _mouth_close(F: Fields, skel: dict, skin_w: tuple, mask: np.ndarray) -> np.ndarray:
    """With jawOpen at 1, move the lips to meet halfway: solve per vertex
    the pre-skinning delta whose skinned result reaches the midpoint."""
    from .rigdef import joint_matrix_entries
    jw = skin_w[1]
    jn = skin_w[0]
    names = [j["name"] for j in skel["joints"]]
    jaw_i = names.index("jaw")
    f = (jw * (jn == jaw_i)).sum(1)
    ent = {(e[1], e[2]): e[3] for e in joint_matrix_entries(skel["scale"]) if e[0] == "jawOpen"}
    Jb = np.asarray(next(j["bind"] for j in skel["joints"] if j["name"] == "jaw"))
    R = euler_matrix(ent.get(("jaw", "rx"), 0), ent.get(("jaw", "ry"), 0), ent.get(("jaw", "rz"), 0))
    t = np.array([ent.get(("jaw", "tx"), 0), ent.get(("jaw", "ty"), 0), ent.get(("jaw", "tz"), 0)])
    c = Jb[:3, 3]
    p = F.pos
    rp = (p - c) @ R.T + c + t
    lbs = (1 - f)[:, None] * p + f[:, None] * rp
    target = 0.5 * (p + rp)
    want = mask[:, None] * (target - lbs)
    A = (1 - f)[:, None, None] * np.eye(3) + f[:, None, None] * R
    return np.linalg.solve(A, want[..., None])[..., 0]


def smooth_shapes(F: Fields, S: dict, iters: int = 4) -> dict:
    """Light Laplacian smoothing of every delta field over the skin (lid and
    lip ring-0 vertices held: their motion is exact)."""
    tm = F.tmpl
    E = edges(tm.tris)
    n = tm.n
    hold = ((F.eye_k <= 0) & (F.eye_side >= 0)) | (F.lip_k <= 0)
    try:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        i = torch.tensor(np.concatenate([E[:, 0], E[:, 1]]), device=dev)
        j = torch.tensor(np.concatenate([E[:, 1], E[:, 0]]), device=dev)
        deg = torch.bincount(i, minlength=n).clamp(min=1).double()
        vals = torch.ones(len(i), dtype=torch.float64, device=dev)
        A = torch.sparse_coo_tensor(torch.stack([i, j]), vals, (n, n)).coalesce()
        names = list(S)
        X = torch.tensor(np.concatenate([S[k] for k in names], 1), device=dev)
        X0 = X.clone()
        H = torch.tensor(hold, device=dev)[:, None]
        for _ in range(iters):
            avg = torch.sparse.mm(A, X) / deg[:, None]
            X = torch.where(H, X0, X + 0.5 * (avg - X))
        Xn = X.cpu().numpy()
        return {k: Xn[:, 3 * c:3 * c + 3] for c, k in enumerate(names)}
    except ImportError:
        deg = np.bincount(E.reshape(-1), minlength=n).astype(np.float64)
        out = {}
        for k, d in S.items():
            x = d.copy()
            for _ in range(iters):
                acc = np.zeros_like(x)
                np.add.at(acc, E[:, 0], x[E[:, 1]])
                np.add.at(acc, E[:, 1], x[E[:, 0]])
                x = np.where(hold[:, None], d, x + 0.5 * (acc / np.maximum(deg, 1)[:, None] - x))
            out[k] = x
        return out


def sparse(S: dict, tol: float = 2e-6) -> dict:
    out = {}
    for k, d in S.items():
        idx = np.flatnonzero(np.linalg.norm(d, axis=1) > tol)
        out[k] = (idx, d[idx])
    return out
