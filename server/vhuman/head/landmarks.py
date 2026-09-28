"""Eyes in a frontal portrait: iris circles, eye openings and iris colour.

There is no face-landmark model in this repository, and none is needed for
a frontal, neutral portrait (what our Qwen template asks for):
1. dark-blob detection (difference of Gaussians over a few scales) in the
   upper half of the head;
2. pairing: two blobs at the same height, of similar size, one each side
   of the face's axis, with white sclera beside them;
3. each iris refined with eye/extract.py's detector on a crop (limbus by an
   integro-differential search, eyelid-aware);
4. the eye opening (palpebral fissure): unsaturated, bright (sclera) or
   iris pixels connected to the iris, grown under a box constraint, then
   closed;
5. the fissure: the opening misses sclera in shadow (often all of one
   side), so the palpebral fissure is modelled as an almond of anatomical
   width, with the lid heights measured at the iris, unioned with the
   opening;
6. iris colour: the dominant colours of the visible iris ring (pupil,
   catchlights and lids excluded), with a chart U/V suggestion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from ..eye import chart, extract, optics


@dataclass
class Eye:
    side: str                   # "right" = the subject's right eye (image left)
    cx: float                   # iris centre, portrait pixels
    cy: float
    r: float                    # iris (limbus) radius, pixels
    pupil_r: float
    opening: np.ndarray = field(repr=False, default=None)   # bool mask (portrait size)
    fissure: np.ndarray = field(repr=False, default=None)   # bool mask: the modelled palpebral fissure
    corners: tuple = ()         # ((x, y) medial, (x, y) lateral)
    color: dict = field(default_factory=dict)
    score: float = 0.0

    def as_dict(self) -> dict:
        return {"side": self.side, "center": [round(self.cx, 2), round(self.cy, 2)], "radius": round(self.r, 2),
                "pupil_radius": round(self.pupil_r, 2), "corners": [[round(v, 1) for v in c] for c in self.corners],
                "opening_px": int(self.opening.sum()) if self.opening is not None else 0,
                "fissure_px": int(self.fissure.sum()) if self.fissure is not None else 0,
                "color": self.color, "score": round(self.score, 3)}


class LandmarkError(ValueError):
    pass


def _gauss(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian via FFT (float)."""
    h, w = img.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    g = np.exp(-2 * math.pi ** 2 * sigma ** 2 * (ky ** 2 + kx ** 2))
    return np.fft.irfft2(np.fft.rfft2(img) * g, s=img.shape).astype(np.float32)


def _load(portrait) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(portrait) if not isinstance(portrait, Image.Image) else portrait
    rgba = np.asarray(img.convert("RGBA"), np.float32) / 255.0
    return rgba[..., :3], rgba[..., 3]


def _head_box(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the head: the alpha bounding box, cut where the
    silhouette widens into the shoulders (row width jumps above the head's)."""
    rows = np.flatnonzero((alpha > 0.5).any(1))
    if len(rows) == 0:
        raise LandmarkError("the portrait has no foreground")
    y0, y1 = rows[0], rows[-1]
    widths = (alpha > 0.5).sum(1)
    top = widths[y0:y0 + max(4, (y1 - y0) // 3)]
    head_w = np.percentile(top, 90)
    below = np.flatnonzero(widths[y0:] > 1.35 * head_w)
    y_shoulder = y0 + (below[0] if len(below) else (y1 - y0))
    cols = np.flatnonzero((alpha[y0:y_shoulder] > 0.5).any(0))
    return int(cols[0]), int(y0), int(cols[-1]), int(y_shoulder)


def _blobs(lum: np.ndarray, mask: np.ndarray, radii) -> list[tuple]:
    """Dark blobs: (x, y, r, strength), from a difference of Gaussians at
    each radius (centre darker than its surround)."""
    out = []
    for r in radii:
        s = r / math.sqrt(2)
        dog = _gauss(lum, 1.6 * s) - _gauss(lum, s)          # positive where the centre is darker
        dog = np.where(mask, dog, 0.0)
        # local maxima: 3x3 neighbourhood on a subsampled grid
        mx = dog.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    mx = np.maximum(mx, np.roll(np.roll(dog, dy, 0), dx, 1))
        peaks = np.argwhere((dog >= mx) & (dog > 0.02))
        for y, x in peaks:
            out.append((float(x), float(y), float(r), float(dog[y, x])))
    out.sort(key=lambda b: -b[3])
    return out[:200]


def _sclera_beside(rgb: np.ndarray, x: float, y: float, r: float) -> float:
    """How white and unsaturated the pixels left and right of a blob are."""
    h, w = rgb.shape[:2]
    vals = []
    for sx in (-1, 1):
        xs = np.clip((x + sx * np.linspace(1.2, 1.9, 6) * r).astype(int), 0, w - 1)
        ys = np.clip(np.full(6, y).astype(int), 0, h - 1)
        px = rgb[ys, xs]
        lum = px @ optics.LUMA
        sat = (px.max(-1) - px.min(-1)) / np.maximum(px.max(-1), 1e-3)
        vals.append(float(np.mean(lum * (1.0 - np.clip(sat * 2, 0, 1)))))
    return min(vals)


LATERAL = extract._angles(((-40.0, 40.0), (140.0, 220.0)), 16)


def _lateral_step(lum: np.ndarray, x: float, y: float, radii: np.ndarray):
    """Iris-to-sclera step on the lateral arcs around (x, y): the best
    (step, radius) over `radii` (the lids hide the top and bottom)."""
    prof = extract._circle_profile(lum, [x], [y], radii, LATERAL)[0]
    d = np.diff(prof)
    k = int(np.argmax(d))
    return float(d[k]), float(0.5 * (radii[k] + radii[k + 1])), float(prof[0])


def _conv(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Correlation of img with a small centred kernel via FFT (same size)."""
    h, w = img.shape
    kh, kw = kernel.shape
    pad = np.zeros((h, w), np.float32)
    pad[:kh, :kw] = kernel
    pad = np.roll(np.roll(pad, -(kh // 2), 0), -(kw // 2), 1)
    return np.fft.irfft2(np.fft.rfft2(img) * np.conj(np.fft.rfft2(pad)), s=img.shape).astype(np.float32)


def _kernels(r: float):
    """A disk (the dark iris) and two lateral annulus sectors (the sclera
    either side), normalised to unit sum."""
    n = int(math.ceil(2.2 * r)) * 2 + 1
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n] - c
    d = np.hypot(xx, yy)
    ang = np.degrees(np.arctan2(-yy, xx))
    disk = (d < 0.75 * r).astype(np.float32)
    ring = (d > 1.15 * r) & (d < 2.1 * r)
    left = (ring & (np.abs(np.abs(ang) - 180) < 32)).astype(np.float32)
    right = (ring & (np.abs(ang) < 32)).astype(np.float32)
    return disk / disk.sum(), left / left.sum(), right / right.sum()


def find_eyes(portrait) -> list[Eye]:
    """Both eyes of a frontal portrait, subject's right (image left) first.
    A matched filter: a dark disk (the iris) with white, unsaturated
    sclera on both sides, over a few radii; then left/right pairing."""
    rgb, alpha = _load(portrait)
    h, w = alpha.shape
    rows = np.flatnonzero((alpha > 0.5).any(1))
    cols = np.flatnonzero((alpha > 0.5).any(0))
    if len(rows) == 0:
        raise LandmarkError("the portrait has no foreground")
    ay0, ay1, ax0, ax1 = rows[0], rows[-1], cols[0], cols[-1]
    lum = _gauss(rgb @ optics.LUMA, 0.7)
    sat = (rgb.max(-1) - rgb.min(-1)) / np.maximum(rgb.max(-1), 1e-3)
    face = alpha > 0.9
    skin = float(np.median(lum[face]))
    dark = np.clip((skin - lum) / max(skin, 1e-3), 0, 1) * face
    sclera = (np.clip((lum - 0.85 * skin) / 0.2, 0, 1) * np.clip((0.32 - sat) / 0.16, 0, 1) * face).astype(np.float32)
    band = np.zeros_like(face)
    band[int(ay0 + 0.2 * (ay1 - ay0)):int(ay0 + 0.62 * (ay1 - ay0)), :] = True
    best_score = np.zeros((h, w), np.float32)
    best_r = np.zeros((h, w), np.float32)
    for f in (0.013, 0.017, 0.021, 0.026):
        r = f * w
        disk, left, right = _kernels(r)
        score = _conv(dark, disk) * np.minimum(_conv(sclera, left), _conv(sclera, right))
        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_r = np.where(better, r, best_r)
    best_score *= band
    # peaks (non-maximum suppression in a window of the iris size)
    cands = []
    sc = best_score.copy()
    for _ in range(12):
        y, x = np.unravel_index(np.argmax(sc), sc.shape)
        if sc[y, x] <= 1e-4:
            break
        r = float(best_r[y, x])
        cands.append((float(x), float(y), r, float(sc[y, x])))
        sc[max(0, int(y - 2 * r)):int(y + 2 * r) + 1, max(0, int(x - 2.5 * r)):int(x + 2.5 * r) + 1] = 0
    axis = 0.5 * (ax0 + ax1)
    best, best_pair = None, -1.0
    for i, a in enumerate(cands):
        for b_ in cands[i + 1:]:
            left_, right_ = (a, b_) if a[0] < b_[0] else (b_, a)
            dx = right_[0] - left_[0]
            if not 0.08 * w < dx < 0.4 * w or abs(right_[1] - left_[1]) > 0.04 * w:
                continue
            if max(left_[2], right_[2]) / min(left_[2], right_[2]) > 1.6:
                continue
            asym = abs(0.5 * (left_[0] + right_[0]) - axis) / w
            if asym > 0.06:
                continue
            score = math.sqrt(left_[3] * right_[3]) * (1.0 - 8 * asym)
            if score > best_pair:
                best, best_pair = (left_, right_), score
    if best is None:
        raise LandmarkError("no pair of eyes found (is the portrait frontal, with open eyes?)")
    eyes = [_refine(rgb, alpha, lum, side, x, y, r, best_pair) for side, (x, y, r, _) in zip(("right", "left"), best)]
    # Both irises are the same size: when the fits disagree, one locked onto
    # an inner ring (a collarette, a limbal ring); refit both at the larger.
    ra, rb = eyes[0].r, eyes[1].r
    if abs(ra - rb) > 0.15 * max(ra, rb):
        r0 = max(ra, rb)
        eyes = [_refine(rgb, alpha, lum, e.side, e.cx, e.cy, r0, best_pair, radius_range=(0.92, 1.08))
                for e in eyes]
    return eyes


def _refine(rgb, alpha, lum, side, x, y, r, score, radius_range=(0.75, 1.3)) -> Eye:
    """Pupil first (the darkest compact blob near the candidate), then the
    iris edge around it: a lateral integro-differential search, per-angle
    edges and a robust circle fit (the pupil and iris are near-concentric)."""
    h, w = lum.shape
    half = int(math.ceil(2.2 * r))
    x0, y0 = max(0, int(x) - half), max(0, int(y) - half)
    x1, y1 = min(w, int(x) + half + 1), min(h, int(y) + half + 1)
    crop = lum[y0:y1, x0:x1]
    s_p = max(1.0, 0.32 * r / math.sqrt(2))
    dog = _gauss(crop, 1.6 * s_p) - _gauss(crop, s_p)          # dark centre, brighter ring
    yy, xx = np.mgrid[y0:y1, x0:x1]
    near = np.hypot(xx - x, yy - y) < 0.9 * r
    k_ = np.argmax(np.where(near, dog, -np.inf))
    py, px = np.unravel_index(k_, dog.shape)
    px, py = float(px + x0), float(py + y0)
    k = max(1.0, 0.15 * r)
    centers = np.stack(np.meshgrid(np.arange(px - k, px + k + 0.5, 0.5), np.arange(py - k, py + k + 0.5, 0.5)),
                       -1).reshape(-1, 2)
    c, ir, _ = extract._ido_search(lum, centers, np.linspace(radius_range[0] * r, radius_range[1] * r, 24),
                                   LATERAL, +1.0, smooth=1)
    angles = np.linspace(0, 2 * math.pi, 90, endpoint=False)
    circle = extract.Circle(float(c[0]), float(c[1]), float(ir))
    (ex, ey), strength = extract._edge_points(lum, circle, angles, 0.75, 1.3, +1.0, n=32, midpoint=True)
    lateral = np.abs(np.cos(angles)) > math.cos(math.radians(45))
    wts = np.clip(strength / max(np.percentile(strength[lateral], 75), 1e-6), 0, 1) * lateral
    if (wts > 0.3).sum() >= 8:
        fitted, _ = extract.fit_circle(ex, ey, wts)
        lo, hi = radius_range[0] * r, radius_range[1] * r
        if (abs(fitted.r - ir) < 0.25 * ir and math.hypot(fitted.cx - px, fitted.cy - py) < 0.2 * ir
                and lo <= fitted.r <= hi):
            circle = fitted
    # Pupil radius: dark-to-bright outwards, all round, at the pupil centre.
    pc = np.stack(np.meshgrid(np.arange(px - 1.5, px + 2.0, 0.5), np.arange(py - 1.5, py + 2.0, 0.5)),
                  -1).reshape(-1, 2)
    _, pr, _ = extract._ido_search(lum, pc, np.linspace(0.12 * circle.r, 0.7 * circle.r, 24),
                                   np.linspace(0, 2 * math.pi, 32, endpoint=False), +1.0, smooth=1)
    eye = Eye(side, circle.cx, circle.cy, circle.r, float(pr), score=float(score))
    eye.opening, eye.corners = _opening(rgb, alpha, eye)
    eye.fissure = fissure(eye)
    eye.color = _iris_color(rgb, eye)
    return eye


def _opening(rgb, alpha, eye: Eye):
    """The visible eye (sclera + iris) connected to the iris, within the
    palpebral fissure's proportions; returns the mask and the two corners.

    Sclera is judged against the local skin (a ring 2.2-3.5 iris radii
    out): clearly less saturated and not much darker. The fissure is about
    30 mm wide and a little under the iris (11.7 mm) tall, so growth is
    confined to +-2.8 iris radii across and +-1.05 up and down."""
    h, w = alpha.shape
    lum = rgb @ optics.LUMA
    sat = (rgb.max(-1) - rgb.min(-1)) / np.maximum(rgb.max(-1), 1e-3)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - eye.cx, yy - eye.cy)
    iris = d < eye.r * 0.95
    ring = (d > 2.2 * eye.r) & (d < 3.5 * eye.r) & (alpha > 0.9)
    skin_lum = float(np.median(lum[ring]))
    skin_sat = float(np.median(sat[ring]))
    sclera = (lum > 0.72 * skin_lum) & (sat < min(0.3, 0.62 * skin_sat))
    box = (np.abs(xx - eye.cx) < 2.8 * eye.r) & (np.abs(yy - eye.cy) < 1.05 * eye.r)
    allowed = (sclera | iris) & box
    region = iris & allowed
    for _ in range(int(4 * eye.r)):
        grown = region | np.roll(region, 1, 0) | np.roll(region, -1, 0) | np.roll(region, 1, 1) | np.roll(region, -1, 1)
        grown &= allowed
        if (grown == region).all():
            break
        region = grown
    region |= iris & box
    # close small gaps (lashes, catchlights)
    img = Image.fromarray((region * 255).astype(np.uint8))
    img = img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    region = (np.asarray(img) > 127) & box
    cols = np.flatnonzero(region.any(0))
    if len(cols) == 0:
        return region, ()
    xl, xr = cols[0], cols[-1]
    yl = float(np.mean(np.flatnonzero(region[:, xl])))
    yr = float(np.mean(np.flatnonzero(region[:, xr])))
    left, right = (float(xl), yl), (float(xr), yr)
    # medial = towards the nose (the face axis lies between the two eyes)
    medial, lateral = (right, left) if eye.side == "right" else (left, right)
    return region, (medial, lateral)


# The palpebral fissure, in iris radii (adult: ~28-30 mm long for an 11.7 mm
# iris; the lateral canthus sits a little above the medial one).
FISSURE_HALF_WIDTH_R = 2.35
CANTHUS_DROP_R = {"medial": 0.12, "lateral": -0.06}   # corner heights below the iris centre


def fissure(eye: Eye) -> np.ndarray:
    """The modelled fissure mask: an almond through the lid heights the
    opening shows at the iris (above and below), FISSURE_HALF_WIDTH_R wide
    each side, unioned with the opening itself."""
    op = eye.opening
    h, w = op.shape
    xs = np.arange(max(0, int(eye.cx - 0.5 * eye.r)), min(w, int(eye.cx + 0.5 * eye.r) + 1))
    rows = [np.flatnonzero(op[:, x]) for x in xs]
    rows = [r for r in rows if len(r)]
    top = eye.cy - (np.median([r[0] for r in rows]) if rows else eye.cy - 0.8 * eye.r)
    bottom = (np.median([r[-1] for r in rows]) if rows else eye.cy + 0.6 * eye.r) - eye.cy
    top = float(np.clip(top, 0.35 * eye.r, 1.1 * eye.r))
    bottom = float(np.clip(bottom, 0.25 * eye.r, 1.05 * eye.r))
    a = FISSURE_HALF_WIDTH_R * eye.r
    # medial is towards the image centre: +x for the subject's right eye (image left)
    s_med = 1.0 if eye.side == "right" else -1.0
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx - eye.cx) / a
    t = np.clip(1.0 - u * u, 0.0, None)
    med = CANTHUS_DROP_R["medial"] * eye.r
    lat = CANTHUS_DROP_R["lateral"] * eye.r
    base = eye.cy + np.where(u * s_med > 0, med, lat) * np.abs(u)
    # corners stay pointed (exponents < 1 keep the almond full near the iris)
    upper = base - top * t ** 0.7
    lower = base + bottom * t ** 0.85
    almond = (np.abs(u) < 1.0) & (yy > upper) & (yy < lower)
    return almond | op


def _iris_color(rgb, eye: Eye) -> dict:
    """Dominant colours of the visible iris ring, and chart U/V for them."""
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - eye.cx, yy - eye.cy)
    ring = (d > max(eye.pupil_r * 1.25, eye.r * 0.3)) & (d < eye.r * 0.9)
    if eye.opening is not None:
        ring &= eye.opening
    px = optics.srgb_to_linear(rgb[ring])
    if len(px) < 12:
        return {}
    lum = px @ optics.LUMA
    keep = lum < np.percentile(lum, 92)            # drop catchlights
    px = px[keep]
    med = np.median(px, axis=0)
    u = np.linspace(0, 1, 101)
    uu, vv = np.meshgrid(u, u)
    grid = chart.chart_color(uu, vv)
    err = ((np.log(grid + 0.01) - np.log(med + 0.01)) ** 2).sum(-1)
    j, i = np.unravel_index(np.argmin(err), err.shape)
    return {"median_linear": [round(float(v), 4) for v in med], "pixels": int(len(px)),
            "suggested": {"primary_color_u": round(float(u[i]), 3), "primary_color_v": round(float(u[j]), 3)}}
