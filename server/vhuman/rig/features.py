"""Facial features of a fitted head: portrait mouth/brows, 3D landmarks.

The portrait is frontal and neutral with the mouth closed (the Qwen
template asks for that), so simple image measurements suffice:
- mouth: the lip seam is the darkest horizontal valley of the lower face;
  a dynamic-programming path follows it across the columns, weighted by
  lip colour (redder than the cheeks); the corners are where the valley
  fades. The vermilion borders are where the lip colour ends above/below.
- brows: the darkest horizontal band above each eye.
All of it is cast through Pixal3D's camera onto the head. The midline
profile of the head then gives the nose tip, subnasale, chin and the other
sagittal landmarks; the ears are the widest points at eye height.
Coordinates are in the head frame H (common.py), metres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..eye import optics
from ..head import landmarks as L
from ..head import lids
from ..head.camera import PixalCamera, ray_mesh
from .common import Subject, normalize, resample_polyline

EYE_RING = 48          # samples of each lid-margin contour (medial corner first, upper lid, lateral, lower lid)
SEAM_SAMPLES = 33      # samples of the lip seam, subject's right corner to left corner


class FeatureError(ValueError):
    pass


@dataclass
class Mouth:
    corners: np.ndarray          # (2, 2) pixels: subject's right (image left), left
    seam: np.ndarray             # (SEAM_SAMPLES, 2) pixels, right corner -> left corner
    upper: np.ndarray            # (SEAM_SAMPLES, 2) upper vermilion border
    lower: np.ndarray            # (SEAM_SAMPLES, 2) lower vermilion border
    score: float = 0.0
    fallback: bool = False

    def as_dict(self):
        return {"corners": self.corners.round(1).tolist(), "score": round(self.score, 3),
                "fallback": self.fallback, "width_px": round(float(np.ptp(self.corners[:, 0])), 1)}


def _box(img, x0, y0, x1, y1):
    h, w = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    return img[y0:y1, x0:x1], x0, y0


def find_mouth(portrait, eyes) -> Mouth:
    rgb, alpha = L._load(portrait)
    er, el = eyes[0], eyes[1]                       # subject's right (image left), left
    ipd = math.hypot(el.cx - er.cx, el.cy - er.cy)
    mx, my = 0.5 * (er.cx + el.cx), 0.5 * (er.cy + el.cy)
    lum_full = L._gauss(rgb @ optics.LUMA, 1.2)
    s = rgb.sum(-1) + 1e-3
    red_full = L._gauss(rgb[..., 0] / s - 0.5 * (rgb[..., 1] + rgb[..., 2]) / s, 2.0)
    # the cheeks' colour: below each eye
    cheeks = [red_full[int(e.cy + 0.55 * ipd), int(e.cx)] for e in eyes]
    skin_red = float(np.median(cheeks))
    x0, x1 = mx - 0.62 * ipd, mx + 0.62 * ipd
    y0, y1 = my + 0.72 * ipd, my + 1.62 * ipd
    lum, bx, by = _box(lum_full, x0, y0, x1, y1)
    red, _, _ = _box(red_full, x0, y0, x1, y1)
    a, _, _ = _box(alpha, x0, y0, x1, y1)
    if lum.size == 0:
        raise FeatureError("mouth search box is outside the portrait")
    lip = np.clip((red - skin_red) / 0.02, 0, 1) * (a > 0.9)
    d = max(2, int(round(0.018 * ipd)))
    valley = np.zeros_like(lum)
    valley[d:-d] = 0.5 * (lum[:-2 * d] + lum[2 * d:]) - lum[d:-d]
    valley = np.clip(valley, 0, None)
    lip_near = L._gauss(lip, 0.02 * ipd)
    score = valley * (0.25 + lip_near)
    # dynamic programming: one row per column, |dy| <= 1
    h, w = score.shape
    acc = score[:, 0].copy()
    back = np.zeros((h, w), np.int64)
    for c in range(1, w):
        prev = np.stack([np.roll(acc, 1), acc, np.roll(acc, -1)])
        prev[0, 0] = prev[2, -1] = -np.inf
        k = np.argmax(prev, 0)
        back[:, c] = np.arange(h) + (k - 1)
        acc = prev[k, np.arange(h)] + score[:, c]
    path = np.zeros(w, np.int64)
    path[-1] = int(np.argmax(acc))
    for c in range(w - 1, 0, -1):
        path[c - 1] = back[path[c], c]
    # the valley must stay on lip colour: the smile folds beyond the corners are dark too
    on_lip = lip_near[path, np.arange(w)] > 0.12
    strength = L._gauss(np.tile(score[path, np.arange(w)] * on_lip, (3, 1)), 0.01 * ipd)[1]
    centre = w // 2
    peak = float(np.median(strength[centre - w // 8:centre + w // 8]))
    fallback = peak < 1e-3
    if not fallback:
        on = strength > 0.3 * peak
        left = centre
        while left > 0 and on[left - 1]:
            left -= 1
        right = centre
        while right < w - 1 and on[right + 1]:
            right += 1
        width = (right - left) / ipd
        fallback = not 0.45 < width < 1.15
    if fallback:                     # proportions: a mouth ~0.78 IPD wide, 1.1 IPD below the eyes
        left, right = int(centre - 0.39 * ipd), int(centre + 0.39 * ipd)
        path = np.full(w, int(round(my + 1.1 * ipd - by)))
    xs = np.linspace(left, right, SEAM_SAMPLES)
    ys = np.interp(xs, np.arange(w), L._gauss(np.tile(path.astype(np.float64), (3, 1)), 1.5)[1])
    seam = np.stack([xs + bx, ys + by], 1)
    # vermilion borders: the lip colour's extent above and below the seam
    up, lo = [], []
    for x, y in zip(xs, ys):
        xi = int(round(x))
        col = lip[:, min(max(xi, 0), w - 1)]
        yi = int(round(y))
        t = 0.5 * max(float(col[max(yi - d, 0):yi + d + 1].max()), 1e-3)
        a_ = yi
        while a_ > 0 and col[a_ - 1] > t and yi - a_ < 0.16 * ipd:
            a_ -= 1
        b_ = yi
        while b_ < h - 1 and col[b_ + 1] > t and b_ - yi < 0.2 * ipd:
            b_ += 1
        up.append(a_)
        lo.append(b_)
    taper = np.sin(np.linspace(0, math.pi, SEAM_SAMPLES)) ** 0.6
    up_h = np.clip(ys - np.array(up), 0.03 * ipd, 0.16 * ipd)
    lo_h = np.clip(np.array(lo) - ys, 0.04 * ipd, 0.2 * ipd)
    up_h = np.median(up_h[SEAM_SAMPLES // 4: -SEAM_SAMPLES // 4]) * taper if not fallback else 0.07 * ipd * taper
    lo_h = np.median(lo_h[SEAM_SAMPLES // 4: -SEAM_SAMPLES // 4]) * taper if not fallback else 0.09 * ipd * taper
    upper = np.stack([xs + bx, ys + by - up_h], 1)
    lower = np.stack([xs + bx, ys + by + lo_h], 1)
    corners = np.array([seam[0], seam[-1]])
    return Mouth(corners, seam, upper, lower, float(peak), fallback)


def find_brows(portrait, eyes) -> list[np.ndarray]:
    """Per eye, a 9-point brow centre line in pixels (medial -> lateral)."""
    rgb, alpha = L._load(portrait)
    lum = L._gauss(rgb @ optics.LUMA, 2.0)
    ipd = math.hypot(eyes[1].cx - eyes[0].cx, eyes[1].cy - eyes[0].cy)
    out = []
    for e in eyes:
        s_med = 1.0 if e.side == "right" else -1.0      # medial is +x for the right eye
        xs = e.cx + s_med * np.linspace(0.18, -0.34, 9) * ipd
        ys = []
        for x in xs:
            y0, y1 = int(e.cy - 0.52 * ipd), int(e.cy - 0.14 * ipd)
            col = lum[y0:y1, int(np.clip(x, 0, lum.shape[1] - 1))]
            ys.append(y0 + int(np.argmin(col)) if len(col) else e.cy - 0.3 * ipd)
        ys = np.convolve(np.pad(np.asarray(ys, float), 2, mode="edge"), np.ones(5) / 5, "valid")
        out.append(np.stack([xs, ys], 1))
    return out


# ---- 3D ------------------------------------------------------------------------------

@dataclass
class Features:
    eyes: list                   # per eye: {side, center, radius, rotation, contour (EYE_RING, 3), corners}
    seam: np.ndarray             # (SEAM_SAMPLES, 3) lip seam, right corner -> left corner
    upper_lip: np.ndarray        # (SEAM_SAMPLES, 3) upper vermilion border
    lower_lip: np.ndarray        # (SEAM_SAMPLES, 3)
    brows: list                  # 2 x (9, 3)
    points: dict                 # named sagittal/lateral landmarks (3,)
    neck: dict                   # {y, ring (N, 3)}: the neck cut
    info: dict = field(default_factory=dict)

    def as_dict(self):
        return {"eyes": [{"side": e["side"], "center": e["center"].round(5).tolist(),
                          "contour": e["contour"].round(5).tolist()} for e in self.eyes],
                "seam": self.seam.round(5).tolist(), "upper_lip": self.upper_lip.round(5).tolist(),
                "lower_lip": self.lower_lip.round(5).tolist(),
                "brows": [b.round(5).tolist() for b in self.brows],
                "points": {k: np.asarray(v).round(5).tolist() for k, v in self.points.items()},
                "neck_y": round(float(self.neck["y"]), 5), "info": self.info}


class Caster:
    """Portrait pixels -> hits on the subject surface (H frame), restricted to
    the triangles that project near the queried pixels."""

    def __init__(self, subj: Subject, cam: PixalCamera):
        self.subj, self.cam = subj, cam
        self.pix = cam.project(subj.frame.to_glb(subj.positions))
        self.o = subj.frame.to_h(cam.origin)

    def cast(self, px: np.ndarray, pad: float = 12.0) -> np.ndarray:
        px = np.asarray(px, np.float64).reshape(-1, 2)
        lo, hi = px.min(0) - pad, px.max(0) + pad
        T = self.subj.triangles
        tp = self.pix[T]
        sel = ((tp.max(1) >= lo) & (tp.min(1) <= hi)).all(1)
        # the front surface only: facing the camera
        P = self.subj.positions
        T = T[sel]
        v0, e1, e2 = P[T[:, 0]], P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]]
        _, d = self.cam.rays(px[:, 0], px[:, 1])
        d = self.subj.frame.dir_to_h(d)
        out = np.full((len(px), 3), np.nan)
        for i in range(len(px)):
            t, _ = ray_mesh(self.o, d[i], v0, e1, e2)
            if math.isfinite(t):
                out[i] = self.o + t * d[i]
        return out

    def ray(self, px):
        px = np.asarray(px, np.float64).reshape(-1, 2)
        _, d = self.cam.rays(px[:, 0], px[:, 1])
        return self.o, self.subj.frame.dir_to_h(d)


def _fill_nan(p: np.ndarray) -> np.ndarray:
    ok = np.isfinite(p).all(1)
    if ok.all():
        return p
    if ok.sum() < 2:
        raise FeatureError("too few surface hits")
    idx = np.arange(len(p))
    return np.stack([np.interp(idx, idx[ok], p[ok, c]) for c in range(3)], 1)


def eye_contour_px(eye, n: int = EYE_RING) -> np.ndarray:
    """The smoothed fissure contour: n pixels from the medial corner over the
    upper lid to the lateral corner and back along the lower lid."""
    r_of = lids.eye_contour(eye)
    if r_of is None:
        raise FeatureError(f"no fissure contour for the {eye.side} eye")
    s_med = 1.0 if eye.side == "right" else -1.0     # medial is image +x for the subject's right eye
    th_m = 0.0 if s_med > 0 else math.pi
    # image y grows downwards, so the upper lid lies at -pi/2. Upper lid:
    # medial -> top -> lateral; lower lid: lateral -> bottom -> medial.
    th_up = th_m - s_med * np.linspace(0, math.pi, n // 2, endpoint=False)
    th = np.concatenate([th_up, th_up + math.pi])
    th = np.arctan2(np.sin(th), np.cos(th))
    r = r_of(th)
    return np.stack([eye.cx + r * np.cos(th), eye.cy + r * np.sin(th)], 1)


def _ray_sphere_far_or_closest(o, d, c, r):
    """Near hit of rays on a sphere; where a ray misses, the point of the ray
    closest to the sphere (the canthi reach beyond the eyeball's silhouette)."""
    oc = o - c
    b = d @ oc if oc.ndim == 1 else (d * oc).sum(-1)
    disc = b * b - ((oc ** 2).sum(-1) - r * r)
    t = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), -b)
    return o + t[:, None] * d


def lid_margin(eye, pose: dict, caster: Caster, margin_m: float = 0.0004) -> np.ndarray:
    """The lid margin in 3D: the portrait's fissure contour cast onto the
    eyeball, lifted by `margin_m` (the lid rests on the eye)."""
    px = eye_contour_px(eye)
    o, d = caster.ray(px)
    return _ray_sphere_far_or_closest(o, d, pose["center"], pose["radius"] + margin_m)


def profile(P: np.ndarray, half_width: float = 0.003, step: float = 0.0008):
    """The front-most midline profile z(y): (ys, zs) over the head."""
    band = P[np.abs(P[:, 0]) < half_width]
    y0, y1 = band[:, 1].min(), band[:, 1].max()
    bins = np.arange(y0, y1 + step, step)
    k = np.clip(((band[:, 1] - y0) / step).astype(int), 0, len(bins) - 1)
    z = np.full(len(bins), -np.inf)
    np.maximum.at(z, k, band[:, 2])
    # a thin bin can miss the front surface's vertices: front-most of 5 bins
    z = np.max(np.stack([np.roll(z, i) for i in range(-2, 3)]), 0)
    ok = np.isfinite(z)
    zs = np.interp(bins, bins[ok], z[ok])
    zs = np.convolve(np.pad(zs, 2, mode="edge"), np.ones(5) / 5, "valid")
    return bins, zs


def _argext(ys, zs, lo, hi, fn):
    m = (ys >= lo) & (ys <= hi)
    if not m.any():
        raise FeatureError(f"profile has no samples in [{lo:.3f}, {hi:.3f}]")
    i = np.flatnonzero(m)[fn(zs[m])]
    return np.array([0.0, ys[i], zs[i]])


def neck_cut(P: np.ndarray, tris: np.ndarray, n: int = 48, lift: float = 0.004) -> dict:
    """The open neck boundary: its lowest height, and a ring just above it
    (vertices near that plane, by angle around their centroid)."""
    from .common import boundary_edges
    be = boundary_edges(tris)
    bv = np.unique(be)
    low = bv[P[bv, 1] < np.percentile(P[:, 1], 2)] if len(bv) else np.array([], int)
    y = float(np.max(P[low, 1])) if len(low) else float(P[:, 1].min())
    y += lift
    band = P[np.abs(P[:, 1] - y) < 0.002]
    c = band.mean(0)
    ang = np.arctan2(band[:, 0] - c[0], band[:, 2] - c[2])
    bins = np.linspace(-math.pi, math.pi, n, endpoint=False)
    k = ((ang + math.pi) / (2 * math.pi) * n).astype(int) % n
    rad = np.hypot(band[:, 0] - c[0], band[:, 2] - c[2])
    r = np.zeros(n)
    np.maximum.at(r, k, rad)
    idx = np.arange(n)
    ok = r > 0
    r = np.interp(idx, idx[ok], r[ok], period=n)
    th = bins + math.pi / n
    ring = np.stack([c[0] + r * np.sin(th), np.full(n, y), c[2] + r * np.cos(th)], 1)
    return {"y": y, "center": np.array([c[0], y, c[2]]), "ring": ring}


def extract(subj: Subject, fov_deg: float | None = None) -> Features:
    cam_d = subj.fit["camera"]
    fov = math.radians(fov_deg if fov_deg is not None else cam_d["fov_deg"])
    cam = PixalCamera.from_portrait(subj.portrait, fov)
    eyes2d = L.find_eyes(subj.portrait)
    caster = Caster(subj, cam)
    eyes = []
    for e2, pose in zip(eyes2d, subj.eyes):
        contour = lid_margin(e2, pose, caster)
        eyes.append(dict(pose, contour=contour, corners=contour[[0, EYE_RING // 2]]))
    mouth = find_mouth(subj.portrait, eyes2d)
    seam = _fill_nan(caster.cast(mouth.seam))
    upper = _fill_nan(caster.cast(mouth.upper))
    lower = _fill_nan(caster.cast(mouth.lower))
    brows = [_fill_nan(caster.cast(b)) for b in find_brows(subj.portrait, eyes2d)]
    P = subj.positions
    ys, zs = profile(P)
    y_m = float(np.median(seam[:, 1]))
    pts = {}
    pts["nose_tip"] = _argext(ys, zs, y_m + 0.012, -0.012, np.argmax)
    pts["subnasale"] = _argext(ys, zs, y_m + 0.006, pts["nose_tip"][1] - 0.004, np.argmin)
    # sellion: deepest below the chord from the nose tip to the forehead
    y_t, z_t = pts["nose_tip"][1], pts["nose_tip"][2]
    z_f = float(np.interp(0.045, ys, zs))
    chord = z_t + (ys - y_t) * (z_f - z_t) / (0.045 - y_t)
    pts["sellion"] = _argext(ys, zs - chord, -0.012, 0.03, np.argmin)
    pts["sellion"][2] = float(np.interp(pts["sellion"][1], ys, zs))
    pts["glabella"] = _argext(ys, zs, pts["sellion"][1] + 0.004, 0.05, np.argmax)
    pts["upper_lip"] = _argext(ys, zs, y_m, y_m + 0.016, np.argmax)
    pts["lower_lip"] = _argext(ys, zs, y_m - 0.018, y_m, np.argmax)
    pts["pogonion"] = _argext(ys, zs, y_m - 0.065, y_m - 0.02, np.argmax)
    pts["sulcus"] = _argext(ys, zs, pts["pogonion"][1], pts["lower_lip"][1], np.argmin)
    # menton: where the chin's profile turns back (its slope passes -1 below the pogonion)
    below = (ys < pts["pogonion"][1]) & (ys > pts["pogonion"][1] - 0.04)
    slope = np.gradient(zs, ys)
    cand = np.flatnonzero(below & (slope > 1.0))
    i_m = cand[-1] if len(cand) else np.flatnonzero(below)[0]
    pts["menton"] = np.array([0.0, ys[i_m], zs[i_m]])
    pts["crown"] = P[np.argmax(P[:, 1])].copy()
    pts["mouth_right"], pts["mouth_left"] = seam[0].copy(), seam[-1].copy()
    # ears: the widest points between the eyes' and the mouth's heights, behind the eyes
    for side, sgn in (("right", -1.0), ("left", 1.0)):
        m = (P[:, 1] < 0.01) & (P[:, 1] > y_m) & (P[:, 2] < -0.02)
        q = P[m]
        pts[f"ear_{side}"] = q[np.argmax(sgn * q[:, 0])].copy()
    neck = neck_cut(P, subj.triangles)
    info = {"mouth": mouth.as_dict(), "fov_deg": math.degrees(fov),
            "ipd_m": float(np.linalg.norm(eyes[1]["center"] - eyes[0]["center"])),
            "mouth_width_m": float(np.linalg.norm(seam[-1] - seam[0]))}
    return Features(eyes, seam, upper, lower, brows, pts, neck, info)
