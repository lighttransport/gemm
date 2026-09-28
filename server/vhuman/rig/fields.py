"""Per-vertex facial coordinates on a fitted template, shared by skinning and
the expression shapes: mouth-relative position (lateral u, height above the
lip seam), lip/eye ring membership, landmark neighbourhoods."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import template as T
from .common import normalize, smoothstep
from .features import Features


@dataclass
class Fields:
    pos: np.ndarray
    tmpl: T.Template
    feat: Features
    mm: float                      # metres per synthetic millimetre (scale / 1000)
    u: np.ndarray                  # mouth lateral: -1 right corner, +1 left corner
    h: np.ndarray                  # height above the lip seam at the vertex's x (m)
    mouth_mid: np.ndarray
    half_width: float
    upper: np.ndarray              # bool: above the seam (lip halves by ring membership)
    lip_k: np.ndarray              # mouth ring index for mouth-group vertices, else 99
    eye_k: np.ndarray              # eye ring index (per eye group), else 99
    eye_side: np.ndarray           # 0 right, 1 left, -1 none
    eye_upper: np.ndarray          # bool: upper lid half of an eye ring
    eye_sample: np.ndarray
    corner: dict                   # "right"/"left": corner point
    cheek: dict                    # cheek apex points
    ala: dict                      # nose ala points
    brow: dict                     # brow polylines (medial -> lateral)

    def gauss(self, p, r_mm):
        return np.exp(-(np.linalg.norm(self.pos - p, axis=1) / (r_mm * self.mm)) ** 2)

    def side(self, left: bool, soft_mm: float = 6.0):
        """Weight of the subject's left (+x) or right half, soft across the midline."""
        x = self.pos[:, 0] / (soft_mm * self.mm)
        return smoothstep(-1, 1, x) if left else smoothstep(1, -1, x)


def surface_point(pos, skin_mask, x, y, reach=0.003):
    m = skin_mask & (np.abs(pos[:, 0] - x) < reach) & (np.abs(pos[:, 1] - y) < reach)
    if not m.any():
        return np.array([x, y, pos[skin_mask, 2].max()])
    i = np.flatnonzero(m)[np.argmax(pos[m, 2])]
    return pos[i].copy()


def compute(tmpl: T.Template, pos: np.ndarray, feat: Features, scale: float) -> Fields:
    mm = 0.001 * scale
    seam = feat.seam
    mid = seam[len(seam) // 2]
    cR, cL = seam[0], seam[-1]
    half = 0.5 * float(np.linalg.norm(cL - cR))
    u = (pos[:, 0] - mid[0]) / half
    order = np.argsort(seam[:, 0])
    ys = np.interp(pos[:, 0], seam[order, 0], seam[order, 1])
    h = pos[:, 1] - ys
    upper = h > 0
    lip_k = np.full(tmpl.n, 99)
    mouth = tmpl.group == 2
    lip_k[mouth] = tmpl.ring[mouth]
    H = T.MOUTH_HALF
    ms = tmpl.sample[mouth]
    upper_ring = (ms > 0) & (ms < H)
    lower_ring = ms > H
    idx = np.flatnonzero(mouth)
    upper[idx[upper_ring]] = True
    upper[idx[lower_ring]] = False
    cap = tmpl.kind == T.KIND["cap"]
    upper[cap] = False
    eye_k = np.full(tmpl.n, 99)
    eye_side = np.full(tmpl.n, -1)
    eye_upper = np.zeros(tmpl.n, bool)
    eye_sample = np.full(tmpl.n, -1)
    for g in (0, 1):
        m = tmpl.group == g
        eye_k[m] = tmpl.ring[m]
        eye_side[m] = g
        eye_sample[m] = tmpl.sample[m]
        eye_upper[m] = tmpl.sample[m] < T.EYE_N // 2
    skin = tmpl.kind != T.KIND["mouth_inner"]
    eyes = {e["side"]: e for e in feat.eyes}
    cheek, ala, brow = {}, {}, {}
    nose_y = feat.points["subnasale"][1]
    for side, sx in (("right", -1.0), ("left", 1.0)):
        e = eyes[side]
        cheek[side] = surface_point(pos, skin, e["center"][0] * 1.08, e["center"][1] - 27 * mm)
        ala[side] = surface_point(pos, skin, sx * 15.5 * mm, nose_y + 5 * mm)
        brow[side] = feat.brows[0 if side == "right" else 1]
    return Fields(pos, tmpl, feat, mm, u, h, mid, half, upper, lip_k, eye_k, eye_side, eye_upper, eye_sample,
                  {"right": cR, "left": cL}, cheek, ala, brow)
