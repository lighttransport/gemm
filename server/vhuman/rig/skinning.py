"""Skin weights of the fitted template on root / neck / head / jaw.

- Vertical: the head gives way to the neck below a boundary that runs from
  the nape up to under the chin, and the neck to the root near the cut.
- Jaw: the lips split exactly at the seam (the lower lip, the floor of the
  mouth bag -> jaw); on the cheeks the boundary runs from each mouth corner
  back to the jaw hinge, with a transition that widens away from the lips;
  nothing behind the hinge follows the jaw. The field is then diffused over
  the surface (lips held) so the transition is smooth.
"""
from __future__ import annotations

import numpy as np

from . import template as T
from .common import edges, smoothstep
from .fields import Fields


def _diffuse(values, E, n, fixed, iters=30, lam=0.5):
    v = values.copy()
    deg = np.bincount(E.reshape(-1), minlength=n).astype(np.float64)
    for _ in range(iters):
        acc = np.zeros(n)
        np.add.at(acc, E[:, 0], v[E[:, 1]])
        np.add.at(acc, E[:, 1], v[E[:, 0]])
        avg = acc / np.maximum(deg, 1)
        nv = v + lam * (avg - v)
        v = np.where(fixed, values, nv)
    return v


def weights(tmpl: T.Template, pos: np.ndarray, F: Fields, skel: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    jb = {j["name"]: np.asarray(j["bind"])[:3, 3] for j in skel["joints"]}
    mm = F.mm
    feat = F.feat
    y, z, x = pos[:, 1], pos[:, 2], pos[:, 0]
    n = tmpl.n
    E = edges(tmpl.tris)
    # jaw fraction
    J = jb["jaw"]
    xJ = 0.5 * (abs(feat.points["ear_left"][0]) + abs(feat.points["ear_right"][0])) - 12 * mm
    xc = F.half_width
    a = np.clip((np.abs(x - F.mouth_mid[0]) - xc) / max(xJ - xc, 1e-6), 0, 1)
    corner_y = 0.5 * (F.corner["left"][1] + F.corner["right"][1])
    y_bd = np.where(np.abs(x - F.mouth_mid[0]) <= xc, y - F.h, corner_y + (J[1] - corner_y) * a ** 0.8)
    tw = (3 + 17 * a) * mm
    f = smoothstep(tw, -tw, y - y_bd)
    f *= smoothstep(J[2] - 5 * mm, J[2] + 18 * mm, z)
    # mouth rings: exact halves, softened at the corners
    mouth = tmpl.group == 2
    H = T.MOUTH_HALF
    s = tmpl.sample
    th = np.where(s <= H, np.pi * s / H, np.pi + np.pi * (s - H) / H)
    corner_soft = smoothstep(0.0, 0.45, np.abs(np.sin(th)))
    f_ring = 0.5 + 0.5 * np.where(F.upper, -1.0, 1.0) * corner_soft
    ring_k = tmpl.ring
    near_lips = mouth & (ring_k <= 3)
    f = np.where(near_lips, f_ring, f)
    far_ring = mouth & (ring_k > 3)
    blend = np.clip((ring_k - 3) / 5.0, 0, 1)
    f = np.where(far_ring, (1 - blend) * f_ring + blend * f, f)
    # the mouth bag: the floor goes with the jaw
    inner = mouth & (ring_k < 0)
    cav = inner & (ring_k <= -3)
    ang_y = (pos[:, 1] - (F.mouth_mid[1] - 2 * mm)) / (8 * mm)
    f = np.where(cav, smoothstep(0.6, -0.6, ang_y), f)
    cap = tmpl.kind == T.KIND["cap"]
    f[cap] = 0.5
    fixed = mouth & (ring_k <= 1)
    f = _diffuse(f, E, n, fixed | cav | cap, iters=25)
    f = np.clip(f, 0, 1)
    # vertical: head -> neck -> root
    head = jb["head"]
    menton = feat.points["menton"]
    zb = np.array([head[2] - 60 * mm, head[2], menton[2]])
    yb = np.array([head[1] - 12 * mm, head[1] - 22 * mm, menton[1] - 14 * mm])
    y_b = np.interp(z, zb, yb)
    neck = smoothstep(y_b + 8 * mm, y_b - 28 * mm, y)
    neck = np.where(tmpl.kind == T.KIND["mouth_inner"], 0.0, neck)
    neck = np.where(cap, 0.0, neck)
    root = smoothstep(feat.neck["y"] + 30 * mm, feat.neck["y"] + 2 * mm, y)
    neck = _diffuse(neck, E, n, (tmpl.kind == T.KIND["mouth_inner"]) | cap, iters=15)
    w_head = 1 - neck
    w_neck = neck * (1 - root)
    w_root = neck * root
    # the jaw takes its fraction of the head share, and some of the submental neck
    front = smoothstep(menton[2] - 45 * mm, menton[2] - 10 * mm, z)
    w_jaw = f * w_head + 0.5 * f * front * w_neck
    w_neck = w_neck - 0.5 * f * front * w_neck
    w_head = w_head - f * w_head
    names = [j["name"] for j in skel["joints"]]
    ji = {nm: i for i, nm in enumerate(names)}
    W = np.stack([w_root, w_neck, w_head, w_jaw], 1)
    W = np.clip(W, 0, None)
    W /= W.sum(1, keepdims=True)
    Jn = np.tile(np.array([ji["root"], ji["neck"], ji["head"], ji["jaw"]]), (n, 1))
    info = {"jaw_vertices": int((W[:, 3] > 0.5).sum()), "neck_vertices": int((W[:, 1] > 0.5).sum())}
    return Jn, W, info
