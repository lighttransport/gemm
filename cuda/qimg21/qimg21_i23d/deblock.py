"""Remove the Qwen-Image 2.1 VAE decoder's latent-cell pattern from a decoded image.

The decoder imprints a faint pattern that repeats every 16 pixels (one latent cell): alternating rows and a
4/8-pixel rhythm inside each cell, from its 2x upsampling stages. It is not a port artifact: the diffusers
decoder produces it from the same latents, and an encode/decode round trip of a pattern-free image adds it
(cell-folded std 0.03 -> 0.11 levels). It is ~0.2 levels on flat background but ~1.4 levels on textured
skin, where it reads as a faint crosshatch. Its shape differs between images and drifts across one image
(quadrant patterns correlate 0.7-0.9), and its amplitude follows the local texture, so it is estimated locally:

  hp   = x - blur(x)                              high-pass (sigma 8)
  u    = hp / rms(hp)                             texture-normalised
  P    = cell-folded mean of u per 256-px window  (cell-aligned, stride 128, Hann-blended)
  g    = local regression of u on P               clamped to [0, 2 x image-wide gain]
  x'   = x - clip(g * rms * P, +-max_levels)

On a 1024^2 portrait the cell-folded pattern drops 0.71 -> 0.12 levels, at the 15/17-px noise floor in every
256-px region (worst ratio 1.19), while fine detail keeps 98% of its energy (1.0 level rms change). Flip-
averaged decoding was rejected: the decoder is not flip-equivariant (14 levels rms apart), so it blurs.
Windows of 128 px start removing real texture (the 15/17-px controls drop too).
"""
from __future__ import annotations

import numpy as np

PERIOD = 16


def _fold(u, period):
    h, w = u.shape[0] // period * period, u.shape[1] // period * period
    f = u[:h, :w].reshape(h // period, period, w // period, period).mean((0, 2))
    return f - f.mean()


def _starts(n, window, stride, period):
    last = (n - window) // period * period
    return sorted(set(list(range(0, last + 1, stride)) + [last]))


def deblock(image: np.ndarray, period: int = PERIOD, window: int = 256, stride: int = 128, gain_sigma: float = 24.0,
            max_levels: float = 3.0, scale: float = 255.0) -> np.ndarray:
    """image: [C, H, W] float with `scale` = full range (255 for levels, 2 for [-1, 1]). Returns a float64 copy
    with the periodic component removed; max_levels is in 8-bit levels. window/stride: multiples of period."""
    from scipy.ndimage import gaussian_filter

    if window % period or stride % period:
        raise ValueError("window and stride must be multiples of the period")
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 3:
        raise ValueError("deblock expects [C, H, W]")
    _, h, w = img.shape
    if h < 2 * period or w < 2 * period:
        return img.copy()
    limit = max_levels * scale / 255.0
    out = np.empty_like(img)
    for c, x in enumerate(img):
        hp = x - gaussian_filter(x, 8)
        rms = np.sqrt(gaussian_filter(hp * hp, 16)) + 1e-6 * scale
        u = hp / rms
        whole = np.tile(_fold(u, period), (h // period + 1, w // period + 1))[:h, :w]
        if h >= window and w >= window:
            num = np.zeros((h, w)); den = np.zeros((h, w))
            hann = np.outer(np.hanning(window), np.hanning(window)) + 1e-3
            reps = window // period
            for y in _starts(h, window, stride, period):
                for x0 in _starts(w, window, stride, period):
                    local = np.tile(_fold(u[y:y + window, x0:x0 + window], period), (reps, reps))
                    num[y:y + window, x0:x0 + window] += hann * local
                    den[y:y + window, x0:x0 + window] += hann
            tiled = np.where(den > 0, num / np.maximum(den, 1e-12), whole)   # uncovered edge strips: whole image
        else:
            tiled = whole
        norm = (tiled * tiled).sum()
        if norm <= 0:
            out[c] = x
            continue
        overall = max((u * tiled).sum() / norm, 0.0)
        gain = gaussian_filter(u * tiled, gain_sigma) / (gaussian_filter(tiled * tiled, gain_sigma) + 1e-12)
        gain = np.clip(gain, 0.0, 2.0 * overall)
        out[c] = x - np.clip(gain * rms * tiled, -limit, limit)
    return out


def cell_pattern_std(gray: np.ndarray, period: int = PERIOD) -> float:
    """Std of the cell-folded high-pass (the pattern's strength, in the input's units). Compare with an
    incommensurate period (15 or 17) for the noise floor."""
    from scipy.ndimage import gaussian_filter

    g = np.asarray(gray, dtype=np.float64)
    hp = g - gaussian_filter(g, 8)
    h, w = g.shape[0] // period * period, g.shape[1] // period * period
    return float(hp[:h, :w].reshape(h // period, period, w // period, period).mean((0, 2)).std())
