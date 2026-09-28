"""Iris detection and normalisation from an image (numpy only; no scipy/opencv).

Turns a close-up of an eye (a Qwen-Image iris "plate", or a photo) into our
iris texture layout, so its photographic detail can replace the procedural
structure:

1. limbus: a coarse-to-fine integro-differential search (Daugman) over
   centres and radii, on the lateral sectors (the eyelids hide the top and
   bottom), then per-angle edges and an IRLS (Tukey) circle fit, plus an
   ellipse check (an oblique gaze makes the iris elliptical: rejected);
2. pupil: the same inside the limbus, dark-to-bright outwards;
3. specular highlights (bright, unsaturated blobs) and eyelid sectors (no
   limbus edge, skin-coloured) become an invalid mask;
4. rubber-sheet unwrap between the two boundaries into a polar strip,
   in-fill of the invalid samples (periodic normalised convolution, then
   angular patch transfer across big eyelid gaps), and removal of the
   illumination gradient around the circle;
5. structure masks (like the procedural ones) and the photo colour, resampled
   to the square layout at the reference pupil ratio; a quality report
   (fit residual, concentricity, ellipticity, occlusion, highlights,
   contrast, kaleidoscope symmetry) decides whether a plate is kept.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import chart, iris, noise, optics
from . import params as P

N_THETA, N_RHO = 2048, 256
OUTER_MARGIN = 0.97
LATERAL = ((-50.0, 50.0), (130.0, 230.0))      # degrees; eyelids hide the top and bottom


@dataclass
class Circle:
    cx: float
    cy: float
    r: float

    def as_list(self, scale: float = 1.0) -> list:
        return [round(self.cx * scale, 3), round(self.cy * scale, 3), round(self.r * scale, 3)]


@dataclass
class Plate:
    masks: np.ndarray          # (res, res, 4) float32 in the iris texture layout
    photo: np.ndarray          # (res, res, 3) sRGB 0-1, same layout
    limbus: Circle
    pupil: Circle
    quality: dict
    colors: dict
    strip: np.ndarray = field(repr=False, default=None)      # (Nrho, Ntheta, 3) linear, filled
    valid: np.ndarray = field(repr=False, default=None)      # (Nrho, Ntheta) bool before filling


class ExtractError(ValueError):
    pass


# ---- helpers -------------------------------------------------------------------

def _load(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    else:
        img = Image.open(image)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            rgba = np.asarray(img, np.float32) / 255.0
            a = rgba[..., 3:4]
            arr = rgba[..., :3] * a + 0.5 * (1 - a)       # transparent -> mid grey
        else:
            arr = np.asarray(img.convert("RGB"), np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, -1)
    return arr[..., :3]


def _box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur (edge-clamped) via cumulative sums."""
    if r <= 0:
        return img
    out = img
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
        n = out.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def _sample(img: np.ndarray, x, y) -> np.ndarray:
    """Bilinear sample of (H, W) or (H, W, C) at pixel coordinates."""
    if img.ndim == 3:
        return np.moveaxis(noise.bilinear(np.moveaxis(img, -1, 0), x, y), 0, -1)
    return noise.bilinear(img, x, y)


def _angles(sectors, n: int) -> np.ndarray:
    out = []
    for lo, hi in sectors:
        out.append(np.linspace(math.radians(lo), math.radians(hi), n, endpoint=False))
    return np.concatenate(out)


def _circle_profile(lum, cx, cy, radii, angles):
    """Mean luminance on circles: (len(cx), len(radii))."""
    cx = np.asarray(cx, np.float32)[:, None, None]
    cy = np.asarray(cy, np.float32)[:, None, None]
    r = np.asarray(radii, np.float32)[None, :, None]
    x = cx + r * np.cos(angles)[None, None, :]
    y = cy - r * np.sin(angles)[None, None, :]
    return _sample(lum, x, y).mean(-1)


def _ido_search(lum, centers, radii, angles, sign: float, smooth: int = 2, sat=None):
    """Integro-differential operator: the centre/radius with the largest
    (signed) blurred radial derivative of the circular mean. With `sat`,
    a circle only counts where what lies just outside it is unsaturated
    (sclera), so an inner colour ring (the collarette) cannot win."""
    prof = _circle_profile(lum, centers[:, 0], centers[:, 1], radii, angles)
    d = np.diff(prof, axis=1) * sign
    if smooth:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        d = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, d)
    if sat is not None:
        mid = 0.5 * (radii[1:] + radii[:-1])
        outside = _circle_profile(sat, centers[:, 0], centers[:, 1], mid * 1.12, angles)
        d = d * np.clip((0.4 - outside) / 0.15, 0.0, 1.0)
    best = np.unravel_index(np.argmax(d), d.shape)
    return centers[best[0]], 0.5 * (radii[best[1]] + radii[best[1] + 1]), float(d[best])


def fit_circle(x, y, w=None, iterations: int = 25) -> tuple[Circle, float]:
    """Algebraic (Kasa) circle fit with Tukey-biweight IRLS; returns the
    circle and the weighted RMS radial residual."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    w = np.ones_like(x) if w is None else np.asarray(w, np.float64).clip(0, None)
    wt = w.copy()
    circle = Circle(float(x.mean()), float(y.mean()), float(np.hypot(x - x.mean(), y - y.mean()).mean()))
    for _ in range(iterations):
        a = np.stack([2 * x, 2 * y, np.ones_like(x)], -1) * np.sqrt(wt)[:, None]
        b = (x * x + y * y) * np.sqrt(wt)
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy = sol[0], sol[1]
        r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
        moved = math.hypot(cx - circle.cx, cy - circle.cy) + abs(r - circle.r)
        circle = Circle(float(cx), float(cy), float(r))
        if moved < 1e-6 * r:
            break
        res = np.hypot(x - cx, y - cy) - r
        scale = 4.685 * max(1.4826 * np.median(np.abs(res)), 0.25)
        u = np.clip(res / scale, -1, 1)
        wt = w * (1 - u * u) ** 2
    res = np.hypot(x - circle.cx, y - circle.cy) - circle.r
    rms = float(np.sqrt(np.sum(wt * res * res) / max(wt.sum(), 1e-9)))
    return circle, rms


def fit_ellipse_ratio(x, y, w) -> float:
    """Minor/major axis ratio of a weighted algebraic conic fit (1 = circle)."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    mx, my = x.mean(), y.mean()
    s = max(np.std(x), np.std(y), 1e-9)
    u, v = (x - mx) / s, (y - my) / s
    d = np.stack([u * u, u * v, v * v, u, v, np.ones_like(u)], -1) * np.sqrt(np.clip(w, 0, None))[:, None]
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    a, b, c = vt[-1][:3]
    m = np.array([[a, b / 2], [b / 2, c]])
    ev = np.linalg.eigvalsh(m)
    if ev[0] * ev[1] <= 0:
        return 0.0
    ev = np.abs(ev)
    return float(math.sqrt(ev.min() / ev.max()))


# ---- detection -----------------------------------------------------------------

def _dark_blob(lum: np.ndarray) -> Circle:
    """The pupil as the darkest compact blob near the centre: threshold the
    darkest pixels, then iterate centroid / radius-from-area inside a
    shrinking window (no connected components needed)."""
    h, w = lum.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    central = np.hypot((xx - w / 2) / w, (yy - h / 2) / h) < 0.35
    low = np.percentile(lum[central], 1.5)
    thr = low + 0.15 * (np.median(lum[central]) - low)
    dark = (lum < thr) & central
    cx, cy = float(xx[dark].mean()), float(yy[dark].mean())
    r = math.sqrt(dark.sum() / math.pi)
    for _ in range(6):
        near = dark & (np.hypot(xx - cx, yy - cy) < 1.6 * r + 2)
        if near.sum() < 10:
            break
        cx, cy = float(xx[near].mean()), float(yy[near].mean())
        r = math.sqrt(near.sum() / math.pi)
    return Circle(cx, cy, r)


def _edge_points(lum, circle: Circle, angles, lo: float, hi: float, sign: float, n: int = 48,
                 midpoint: bool = False):
    """Per-angle edge between lo*r and hi*r: the strongest signed step, or
    with `midpoint` the radius where the profile crosses half-way between
    the inner level (just inside the step) and the outer plateau. The
    limbus is a long ramp (dark ring, then the sclera); its steepest point
    sits outside the iris edge and its start well inside."""
    radii = np.linspace(lo * circle.r, hi * circle.r, n)
    x = circle.cx + radii[None, :] * np.cos(angles)[:, None]
    y = circle.cy - radii[None, :] * np.sin(angles)[:, None]
    prof = _sample(lum, x, y)
    d = np.diff(prof, axis=1) * sign
    k = np.argmax(d, axis=1)
    rows = np.arange(len(angles))
    strength = d[rows, k]
    rr = 0.5 * (radii[k] + radii[k + 1])
    if midpoint:
        cols = np.arange(n)[None, :]
        inner = np.where(cols <= k[:, None], prof, np.inf).min(1) if sign > 0 else \
            np.where(cols <= k[:, None], prof, -np.inf).max(1)
        outer = np.median(prof[:, -max(3, n // 8):], axis=1)
        level = 0.5 * (inner + outer)
        above = (prof - level[:, None]) * sign > 0
        # first crossing at or before the steepest step, walking out from the inner level
        start = np.where(cols <= k[:, None], np.where(sign > 0, prof, -prof), np.inf).argmin(1)
        cross = np.where(above & (cols >= start[:, None]), cols, n - 1).min(1)
        c0 = np.maximum(cross - 1, 0)
        p0, p1 = prof[rows, c0], prof[rows, cross]
        f = np.clip((level - p0) / np.where(np.abs(p1 - p0) < 1e-6, 1e-6, p1 - p0), 0, 1)
        rr = radii[c0] + f * (radii[cross] - radii[c0])
    return (circle.cx + rr * np.cos(angles), circle.cy - rr * np.sin(angles)), strength


def _harmonic_ratio(angles, radii, weights) -> float | None:
    """Axis ratio from the second harmonic of r(theta) over the valid
    angles; None when they do not span enough directions to tell."""
    w = np.asarray(weights, np.float64)
    if (w > 0.3).mean() < 0.5:
        return None
    a = np.stack([np.ones_like(angles), np.cos(2 * angles), np.sin(2 * angles)], -1) * np.sqrt(w)[:, None]
    sol, *_ = np.linalg.lstsq(a, radii * np.sqrt(w), rcond=None)
    amp = math.hypot(sol[1], sol[2]) / max(sol[0], 1e-9)
    return float((1 - amp) / (1 + amp))


def detect(img: np.ndarray, work_size: int = 512) -> dict:
    """Pupil first (the darkest compact blob), then the limbus around it;
    circles in image pixels, with fit diagnostics."""
    h, w = img.shape[:2]
    scale = work_size / max(h, w)
    small = np.asarray(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
        (max(8, round(w * scale)), max(8, round(h * scale))), Image.BILINEAR), np.float32) / 255.0
    lum = _box_blur(small @ optics.LUMA, 1)
    mx, mn = small.max(-1), small.min(-1)
    sat = _box_blur((mx - mn) / np.maximum(mx, 1e-3), 1)
    sh, sw = lum.shape
    side = min(sh, sw)
    angles = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    # Pupil: dark blob, then an integro-differential refinement and a fit.
    blob = _dark_blob(lum)
    k = max(2.0, blob.r * 0.3)
    centers = np.stack(np.meshgrid(np.arange(blob.cx - k, blob.cx + k + 0.5, max(0.5, k / 6)),
                                   np.arange(blob.cy - k, blob.cy + k + 0.5, max(0.5, k / 6))), -1).reshape(-1, 2)
    radii = np.linspace(blob.r * 0.7, blob.r * 1.4, 32)
    pc, pr, _ = _ido_search(lum, centers, radii, np.linspace(0, 2 * math.pi, 48, endpoint=False), +1.0, smooth=1)
    (qx, qy), qs = _edge_points(lum, Circle(float(pc[0]), float(pc[1]), float(pr)), angles, 0.7, 1.35, +1.0,
                                n=40, midpoint=True)
    pupil, prms = fit_circle(qx, qy, np.clip(qs / max(np.percentile(qs, 75), 1e-6), 0, 1))
    # Limbus: centred near the pupil, clearly larger, on the lateral sectors.
    lateral = _angles(LATERAL, 32)
    off = max(3.0, side * 0.04)
    centers = np.stack(np.meshgrid(np.arange(pupil.cx - off, pupil.cx + off + 0.5, max(1.0, off / 5)),
                                   np.arange(pupil.cy - off, pupil.cy + off + 0.5, max(1.0, off / 5))),
                       -1).reshape(-1, 2)
    # A pupil is 12-72% of the iris radius (params.pupil_ratio's clamp).
    radii = np.linspace(max(pupil.r * 1.35, side * 0.08), min(pupil.r / 0.12, side * 0.5), 64)
    c, r, _ = _ido_search(lum, centers, radii, lateral, +1.0, sat=sat)
    step = max(1.0, off / 5)
    centers = np.stack(np.meshgrid(np.arange(c[0] - step, c[0] + step + 0.5, 0.5),
                                   np.arange(c[1] - step, c[1] + step + 0.5, 0.5)), -1).reshape(-1, 2)
    c, r, _ = _ido_search(lum, centers, np.linspace(r * 0.93, r * 1.07, 28), _angles(LATERAL, 48), +1.0, smooth=1,
                          sat=sat)
    limbus0 = Circle(float(c[0]), float(c[1]), float(r))
    (ex, ey), strength = _edge_points(lum, limbus0, angles, 0.8, 1.25, +1.0, n=64, midpoint=True)
    sw_ = np.clip(strength / max(np.percentile(strength, 75), 1e-6), 0, 1)
    # Eyelid margins also make bright-outward edges; the sclera beyond a true
    # limbus is unsaturated, skin is not. Fit the lateral sectors first, then
    # keep the angles that agree with that circle and have sclera outside.
    out_x = limbus0.cx + 1.12 * limbus0.r * np.cos(angles)
    out_y = limbus0.cy - 1.12 * limbus0.r * np.sin(angles)
    sclera_out = _sample(sat, out_x, out_y) < 0.3
    lat = (np.abs(np.cos(angles)) > math.cos(math.radians(50))) & sclera_out
    first, _ = fit_circle(ex[lat], ey[lat], sw_[lat]) if lat.sum() >= 12 else fit_circle(ex, ey, sw_)
    res = np.abs(np.hypot(ex - first.cx, ey - first.cy) - first.r) / first.r
    inlier = sclera_out & (res < 0.04) & (sw_ > 0.3)
    limbus, rms = fit_circle(ex[inlier], ey[inlier], sw_[inlier]) if inlier.sum() >= 12 else (first, 1.0)
    edge_r = np.hypot(ex - limbus.cx, ey - limbus.cy)
    sw_ = np.where(inlier, sw_, 0.0)
    inv = 1.0 / scale
    return {"limbus": Circle(limbus.cx * inv, limbus.cy * inv, limbus.r * inv),
            "pupil": Circle(pupil.cx * inv, pupil.cy * inv, pupil.r * inv),
            "limbus_rms": rms / limbus.r, "pupil_rms": prms / max(pupil.r, 1e-6),
            "ellipse_ratio": _harmonic_ratio(angles, edge_r, sw_),
            "edge_ok": sw_ > 0.3, "angles": angles, "scale": scale}


# ---- unwrap and fill -------------------------------------------------------------

def unwrap(img: np.ndarray, limbus: Circle, pupil: Circle, n_theta: int = N_THETA, n_rho: int = N_RHO):
    """Rubber-sheet strip (n_rho, n_theta, 3) between the pupil and limbus
    circles; theta counter-clockwise from +x with y up (image y down)."""
    theta = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    rho = np.linspace(0, 1, n_rho)[:, None]
    xp = pupil.cx + pupil.r * np.cos(theta)
    yp = pupil.cy - pupil.r * np.sin(theta)
    xl = limbus.cx + limbus.r * np.cos(theta)
    yl = limbus.cy - limbus.r * np.sin(theta)
    x = (1 - rho) * xp[None, :] + rho * xl[None, :]
    y = (1 - rho) * yp[None, :] + rho * yl[None, :]
    return _sample(img, x, y), theta


def highlight_mask(strip_lin: np.ndarray) -> np.ndarray:
    """Specular highlights: bright, unsaturated outliers (then dilated)."""
    lum = strip_lin @ optics.LUMA
    mx = strip_lin.max(-1)
    mn = strip_lin.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-4)
    local = _box_blur(lum, 12)
    med = np.median(lum)
    mad = np.median(np.abs(lum - med)) + 1e-4
    hot = (lum > med + 5 * mad) & (lum > local * 1.5) & (sat < 0.35)
    hot |= (lum > 0.85) & (sat < 0.2)
    for _ in range(3):
        hot = hot | np.roll(hot, 1, 0) | np.roll(hot, -1, 0) | np.roll(hot, 1, 1) | np.roll(hot, -1, 1)
    return hot


def eyelid_columns(det: dict, n_theta: int) -> np.ndarray:
    """Angles where the limbus edge is missing (covered by a lid), per strip
    column; only accepted in the top/bottom halves."""
    ok = det["edge_ok"].astype(np.float32)
    ok = np.convolve(np.concatenate([ok[-3:], ok, ok[:3]]), np.ones(7) / 7, "same")[3:-3] > 0.5
    ang = det["angles"]
    lid = (~ok) & (np.abs(np.sin(ang)) > 0.35)
    # Lid margins cover the iris edge a little beyond where its edge vanishes.
    grow = 6                                   # 12 degrees at 2-degree steps
    lid = np.convolve(np.concatenate([lid[-grow:], lid, lid[:grow]]).astype(np.float32),
                      np.ones(2 * grow + 1), "same")[grow:-grow] > 0
    return np.interp(np.linspace(0, 2 * math.pi, n_theta, endpoint=False), ang, lid.astype(np.float32),
                     period=2 * math.pi) > 0.5


def _blur_periodic(img: np.ndarray, sigma_rho: float, sigma_theta: float) -> np.ndarray:
    """Gaussian blur: periodic along theta (axis 1), mirrored along rho.
    Wide kernels run on a 4x decimated copy and are upsampled back."""
    nr, nt = img.shape[:2]
    if sigma_theta >= 12 and sigma_rho >= 2 and nr % 4 == 0 and nt % 4 == 0:
        small = img.reshape(nr // 4, 4, nt // 4, 4, *img.shape[2:]).mean((1, 3))
        blurred = _blur_periodic(small, sigma_rho / 4, sigma_theta / 4)
        yy = (np.arange(nr, dtype=np.float32)[:, None] + 0.5) / 4 - 0.5
        xx = (np.arange(nt, dtype=np.float32)[None, :] + 0.5) / 4 - 0.5
        yy, xx = np.broadcast_to(yy, (nr, nt)), np.broadcast_to(xx, (nr, nt))
        if blurred.ndim == 3:
            return np.moveaxis(noise.bilinear(np.moveaxis(blurred, -1, 0), xx, yy, wrap_x=True), 0, -1)
        return noise.bilinear(blurred, xx, yy, wrap_x=True)
    ext = np.concatenate([img[::-1], img, img[::-1]], 0).astype(np.float32)
    ky = np.fft.fftfreq(ext.shape[0])[:, None]
    kx = np.fft.rfftfreq(nt)[None, :]
    g = np.exp(-2 * math.pi ** 2 * ((ky * sigma_rho) ** 2 + (kx * sigma_theta) ** 2)).astype(np.float32)
    if img.ndim == 3:
        out = np.fft.irfft2(np.fft.rfft2(ext, axes=(0, 1)) * g[..., None], s=ext.shape[:2], axes=(0, 1))
    else:
        out = np.fft.irfft2(np.fft.rfft2(ext) * g, s=ext.shape)
    return out[nr:2 * nr].astype(np.float32)


def lid_samples(strip_lin: np.ndarray, lid_cols: np.ndarray) -> np.ndarray:
    """Within the eyelid angles, the samples that are not iris: colour far
    from the iris ring's median at that radius, grown outwards (a lid
    covers the iris from its edge inwards)."""
    logc = np.log(np.clip(strip_lin, 1e-3, None))
    clean = ~lid_cols
    ref = np.median(logc[:, clean], axis=1) if clean.any() else np.median(logc, axis=1)
    dist = np.linalg.norm(logc - ref[:, None, :], axis=-1)
    spread = np.median(dist[:, clean], axis=1) if clean.any() else np.median(dist, axis=1)
    odd = dist > np.maximum(3.0 * spread, 0.35)[:, None]
    odd &= lid_cols[None, :]
    # The lid is the run of odd samples touching the iris edge (rows go
    # pupil -> limbus); smooth along the radius so a crypt does not break it.
    k = np.ones(7) / 7
    sm = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, odd.astype(np.float32)) > 0.5
    return np.cumprod(sm[::-1], axis=0)[::-1].astype(bool)


def fill(strip: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid samples: normalised convolution at growing scales, then
    for wide angular gaps, texture from the nearest valid angles (mirrored)
    blended on top so the fill is not a smear."""
    out = strip.copy()
    v = valid.astype(np.float32)
    filled = valid.copy()
    for sigma in (3, 8, 20, 60, 180):
        num = _blur_periodic(strip * v[..., None], sigma * 0.5, sigma)
        den = _blur_periodic(v, sigma * 0.5, sigma)
        take = (~filled) & (den > 1e-3)
        out[take] = (num[take] / den[take][:, None])
        filled |= take
        if filled.all():
            break
    # Angular patch transfer for columns mostly invalid (eyelids).
    col_bad = (~valid).mean(0) > 0.5
    if col_bad.any() and not col_bad.all():
        nt = strip.shape[1]
        bad = np.flatnonzero(col_bad)
        # Mirror each bad column across the nearest good boundary.
        good = np.flatnonzero(~col_bad)
        d = (bad[:, None] - good[None, :] + nt // 2) % nt - nt // 2
        nearest = good[np.argmin(np.abs(d), axis=1)]
        dist = d[np.arange(len(bad)), np.argmin(np.abs(d), axis=1)]
        src = (nearest - dist) % nt
        src_ok = ~col_bad[src]
        detail = strip[:, src] - _blur_periodic(strip, 4, 24)[:, src]
        blend = np.clip(1.0 - np.abs(dist) / max(nt * 0.25, 1), 0.3, 1.0)[None, :, None]
        out[:, bad] = np.where(src_ok[None, :, None], out[:, bad] + detail * blend, out[:, bad])
    return out


def normalise_lighting(strip_lin: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Divide out the illumination gradient around the circle (keep the
    radial colour zones): per-row luminance blurred along theta."""
    lum = strip_lin @ optics.LUMA
    around = _blur_periodic(lum, 6, 200)
    row = around.mean(1, keepdims=True)
    gain = np.clip(row / np.maximum(around, 1e-4), 0.5, 2.0)
    return strip_lin * gain[..., None]


def structure_layers(strip_lin: np.ndarray) -> np.ndarray:
    """Mask layers (4, Nrho, Ntheta): R
    detail (local structure), G shadow (1 lit, low in dark detail and the
    pupillary ruff), B secondary region (a soft falloff inside the strongest
    radial colour change, the collarette), A pupil (0 in the strip)."""
    nr, nt = strip_lin.shape[:2]
    lum = strip_lin @ optics.LUMA
    local = _blur_periodic(lum, 6, 24)
    detail = lum / np.maximum(local, 1e-4) - 1.0
    d = np.clip(0.5 + 1.6 * detail, 0, 1)
    dark = np.clip(-2.5 * detail, 0, 1)
    col = _blur_periodic(strip_lin, 4, 60)
    rows = slice(int(nr * 0.12), int(nr * 0.6))
    grad = np.linalg.norm(np.diff(col[rows], axis=0), axis=-1)
    k = np.argmax(_blur_periodic(grad, 2, 40), axis=0) + rows.start
    rc = _blur_periodic(k[None, :].astype(np.float32), 0.01, 30)[0] / (nr - 1)
    rho = np.linspace(0, 1, nr)[:, None]
    secondary = np.clip(1.0 - rho / (1.7 * np.maximum(rc[None, :], 0.05)), 0.0, 1.0) ** 0.7
    ruff = 1.0 - noise.smoothstep(0.02, 0.05, rho) * np.ones((1, nt))
    shadow = 1.0 - np.maximum(dark, 0.8 * ruff)
    return np.stack([d, shadow, secondary, np.zeros_like(d)]).astype(np.float32)


def kaleidoscope_score(strip_lin: np.ndarray) -> float:
    """Peak angular autocorrelation of the structure away from zero lag:
    generated irises sometimes repeat a pattern around the circle."""
    lum = strip_lin[int(strip_lin.shape[0] * 0.2):int(strip_lin.shape[0] * 0.9)] @ optics.LUMA
    x = lum - _blur_periodic(lum, 3, 30)
    x = x - x.mean(1, keepdims=True)
    f = np.fft.rfft(x, axis=1)
    ac = np.fft.irfft(f * np.conj(f), n=x.shape[1], axis=1).sum(0)
    ac /= max(ac[0], 1e-12)
    nt = x.shape[1]
    lo = nt // 60
    return float(ac[lo:nt // 2].max())


def dominant_colors(strip_lin: np.ndarray, valid: np.ndarray) -> dict:
    """2-means in linear RGB of the valid iris samples, split by radius:
    the inner cluster is the secondary colour, the outer the primary."""
    nr = strip_lin.shape[0]
    rows = np.arange(nr)[:, None] * np.ones((1, strip_lin.shape[1]))
    sel = valid & (rows > nr * 0.06) & (rows < nr * 0.92)
    px = strip_lin[sel]
    rr = rows[sel] / nr
    if len(px) < 50:
        return {}
    centers = np.stack([px[rr < 0.35].mean(0) if (rr < 0.35).any() else px.mean(0),
                        px[rr >= 0.35].mean(0) if (rr >= 0.35).any() else px.mean(0)])
    for _ in range(8):
        lab = np.argmin(((px[:, None, :] - centers[None]) ** 2).sum(-1), 1)
        for k in range(2):
            if (lab == k).any():
                centers[k] = px[lab == k].mean(0)
    inner = int(np.argmin([rr[lab == k].mean() if (lab == k).any() else 1 for k in range(2)]))
    sec, prim = centers[inner], centers[1 - inner]

    def chart_uv(c):
        u = np.linspace(0, 1, 41)
        uu, vv = np.meshgrid(u, u)
        grid = chart.chart_color(uu, vv)
        err = ((np.log(grid + 0.01) - np.log(np.asarray(c) + 0.01)) ** 2).sum(-1)
        j, i = np.unravel_index(np.argmin(err), err.shape)
        return round(float(u[i]), 3), round(float(u[j]), 3)

    pu, pv = chart_uv(prim)
    su, sv = chart_uv(sec)
    return {"primary_linear": [round(float(v), 4) for v in prim], "secondary_linear": [round(float(v), 4) for v in sec],
            "suggested": {"primary_color_u": pu, "primary_color_v": pv, "secondary_color_u": su,
                          "secondary_color_v": sv}}


# ---- the whole extraction ----------------------------------------------------------

THRESHOLDS = {"limbus_rms": 0.03, "concentricity": 0.12, "ellipse_ratio": 0.88, "occluded": 0.45,
              "highlights": 0.12, "pupil_contrast": 0.08, "kaleidoscope": 0.6, "limbus_fraction": (0.12, 0.49),
              "pupil_ratio": (0.12, 0.75)}


def extract(image, res: int = 1024) -> Plate:
    """Detect, unwrap, fill and resample an iris; raises ExtractError when
    no plausible iris is found. Check `plate.quality['ok']` for the gate."""
    started = time.perf_counter()
    img = _load(image)
    det = detect(img)
    limbus, pupil = det["limbus"], det["pupil"]
    if not (0.05 * limbus.r < pupil.r < 0.9 * limbus.r):
        raise ExtractError("no pupil found inside the iris")
    lin = optics.srgb_to_linear(img).astype(np.float32)
    # Stop 3% inside the limbus: its outer half is already the bright
    # sclera, and the shader draws its own limbal ring and corneal limbus.
    strip, _ = unwrap(lin, Circle(limbus.cx, limbus.cy, limbus.r * OUTER_MARGIN), pupil)
    hot = highlight_mask(strip)
    lid = lid_samples(strip, eyelid_columns(det, N_THETA))
    valid = ~(hot | lid)
    filled = fill(strip, valid)
    filled = normalise_lighting(filled, valid)
    layers = structure_layers(filled)
    # Resample the strip (pupil margin .. limbus) into the square layout.
    masks = iris.polar_to_square(layers, res, iris.PUPIL_MASKS, iris.OUTSIDE)
    srgb = optics.linear_to_srgb(np.clip(filled, 0, 1)).astype(np.float32)
    photo = iris.polar_to_square(np.moveaxis(srgb, -1, 0), res, (0.02, 0.02, 0.02),
                                 tuple(float(v) for v in np.median(srgb[-8:], axis=(0, 1))))
    lum_img = lin @ optics.LUMA
    yy, xx = np.mgrid[0:lin.shape[0], 0:lin.shape[1]]
    d = np.hypot(xx - pupil.cx, yy - pupil.cy)
    pupil_l = float(np.median(lum_img[d < pupil.r * 0.7])) if (d < pupil.r * 0.7).any() else 1.0
    iris_l = float(np.median(strip[N_RHO // 5:, :][valid[N_RHO // 5:, :]] @ optics.LUMA)) if valid.any() else 0.0
    h, w = img.shape[:2]
    quality = {
        "limbus_rms": round(det["limbus_rms"], 4),
        "concentricity": round(math.hypot(pupil.cx - limbus.cx, pupil.cy - limbus.cy) / limbus.r, 4),
        "ellipse_ratio": None if det["ellipse_ratio"] is None else round(det["ellipse_ratio"], 4),
        "occluded": round(float(lid.mean()), 4),
        "highlights": round(float(hot.mean()), 4),
        "pupil_contrast": round((iris_l - pupil_l) / max(iris_l, 1e-4), 4),
        "kaleidoscope": round(kaleidoscope_score(filled), 4),
        "limbus_fraction": round(limbus.r / min(h, w), 4),
        "pupil_ratio": round(pupil.r / limbus.r, 4),
        "effective_resolution": int(round(2 * limbus.r)),
    }
    fails = []
    t = THRESHOLDS
    if quality["limbus_rms"] > t["limbus_rms"]:
        fails.append("limbus fit")
    if quality["concentricity"] > t["concentricity"]:
        fails.append("off-centre pupil")
    if quality["ellipse_ratio"] is not None and quality["ellipse_ratio"] < t["ellipse_ratio"]:
        fails.append("elliptical iris (oblique gaze)")
    if quality["occluded"] > t["occluded"]:
        fails.append("eyelids cover the iris")
    if quality["highlights"] > t["highlights"]:
        fails.append("specular highlights")
    if quality["pupil_contrast"] < t["pupil_contrast"]:
        fails.append("no dark pupil")
    if quality["kaleidoscope"] > t["kaleidoscope"]:
        fails.append("repeating (kaleidoscope) pattern")
    if not t["limbus_fraction"][0] <= quality["limbus_fraction"] <= t["limbus_fraction"][1]:
        fails.append("iris too small or cropped")
    if not t["pupil_ratio"][0] <= quality["pupil_ratio"] <= t["pupil_ratio"][1]:
        fails.append("implausible pupil size")
    quality["ok"] = not fails
    quality["failures"] = fails
    quality["seconds"] = round(time.perf_counter() - started, 3)
    masks = np.moveaxis(masks, 0, -1).astype(np.float32)
    photo = np.clip(np.moveaxis(photo, 0, -1), 0, 1).astype(np.float32)
    return Plate(masks, photo, limbus, pupil, quality, dominant_colors(filled, valid), filled, valid)


# ---- synthetic test images -------------------------------------------------------

def synthetic_eye(p: dict, size: int, seed: int, *, occlude: bool = True, highlight: bool = True,
                  ellipse: float = 1.0, blur: float = 0.8, noise_std: float = 0.01, radius_frac: float | None = None,
                  centred: bool = False, iris_tex=None):
    """A test image with a known iris: the procedural iris pasted at a random
    centre/scale onto a sclera, with catchlights and eyelids. Returns the
    image (sRGB 0-1) and the true limbus and pupil circles."""
    g = np.random.default_rng(seed)
    tex = iris_tex or iris.build(p, 512)
    white = lambda r_uv, ph: np.broadcast_to(np.array([0.7, 0.66, 0.62]), np.shape(r_uv) + (3,))  # noqa: E731
    iris_rgb = optics.linear_to_srgb(iris.bake_color(tex, p, 512, white) * 1.8)
    r = (radius_frac or g.uniform(0.26, 0.40)) * size
    cx = size / 2 + (0.0 if centred else g.uniform(-0.08, 0.08) * size)
    cy = size / 2 + (0.0 if centred else g.uniform(-0.08, 0.08) * size)
    rot = 0.0 if centred else g.uniform(0, math.pi)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = xx - cx, -(yy - cy)
    c, s = math.cos(rot), math.sin(rot)
    ex, ey = c * dx + s * dy, (-s * dx + c * dy) / ellipse
    ddx, ddy = c * ex - s * ey, s * ex + c * ey
    t = np.hypot(ddx, ddy) / r                               # iris units (limbus = 1)
    iris_r = optics.iris_radius(p) / optics.profile_from_params(p).limbus_radius   # texture t=1 is the iris edge
    tt = t / iris_r
    u = 0.5 + optics.IRIS_TEX_SCALE * tt * ddx / np.maximum(np.hypot(ddx, ddy), 1e-6)
    v = 0.5 - optics.IRIS_TEX_SCALE * tt * ddy / np.maximum(np.hypot(ddx, ddy), 1e-6)
    col = _sample(iris_rgb, u * 512 - 0.5, v * 512 - 0.5)
    sclera_c = np.array([0.86, 0.82, 0.79]) * (1 - 0.1 * np.clip(t - 1, 0, 1))[..., None]
    w = np.clip((t - 1.0) / 0.02 + 0.5, 0, 1)[..., None]
    img = col * (1 - w) + sclera_c * w
    if highlight:
        for _ in range(g.integers(1, 3)):
            hx = cx + g.uniform(-0.5, 0.5) * r
            hy = cy + g.uniform(-0.5, 0.5) * r
            hr = g.uniform(0.04, 0.09) * r
            m = np.clip(1.2 - np.hypot(xx - hx, yy - hy) / hr, 0, 1)[..., None]
            img = img * (1 - m) + m * 0.98
    if occlude:
        skin = np.array([0.78, 0.58, 0.48])
        # Lid margins are arcs: the upper one highest over the centre, covering
        # the top 25% of the iris there; the lower one the bottom 15%.
        top = cy - r * (1 - 2 * 0.25) + 0.12 * ((xx - cx) / r) ** 2 * r
        bot = cy + r * (1 - 2 * 0.15) - 0.10 * ((xx - cx) / r) ** 2 * r
        lid = np.clip((top - yy) / 2 + 0.5, 0, 1) + np.clip((yy - bot) / 2 + 0.5, 0, 1)
        img = img * (1 - np.clip(lid, 0, 1)[..., None]) + skin * np.clip(lid, 0, 1)[..., None]
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if blur:
        from PIL import ImageFilter
        out = out.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(out, np.float32) / 255.0 + g.normal(0, noise_std, (size, size, 3)).astype(np.float32)
    pupil_r = P.pupil_ratio(p) * iris_r * r
    # The visible iris edge: where the corneal-limbus composite's iris weight
    # crosses 0.5 at the selected iris radius.
    ts = np.linspace(0.6, 1.4, 1601)
    w = iris.cornea_mask(ts * p["cornea"]["size"], p)
    t50 = float(ts[np.argmax(w < 0.5)]) if (w < 0.5).any() else 1.0
    return np.clip(arr, 0, 1), Circle(cx, cy, r * min(t50, 1.0) * iris_r), Circle(cx, cy, pupil_r)
