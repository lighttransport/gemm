"""Remove the Qwen-Image 2.1 VAE decoder's latent-cell pattern from a decoded image.

The decoder imprints a faint pattern that repeats every 16 pixels (one latent cell): alternating rows and a
4/8-pixel rhythm inside each cell, from its 2x upsampling stages. It is not a port artifact: the diffusers
decoder produces the same pattern from the same latents, and an encode/decode round trip of a pattern-free
image adds it (cell-folded std 0.03 -> 0.11 levels). It is ~0.2 levels on flat background but ~1.4 levels
on textured skin, where it reads as slight blockiness. Its shape differs between images and its amplitude
follows the local texture, so it is estimated per image:

  hp   = x - blur(x)                         high-pass (sigma 8)
  P    = cell-folded mean of hp / rms(hp)    the image's 16x16 unit pattern
  g    = local regression of hp/rms on P     clamped to [0, 2 x image-wide gain]
  x'   = x - clip(g * rms * tile(P), +-max_levels)

On a 1024^2 portrait it halves the cell-folded pattern (skin 1.43 -> 0.90 levels) and changes the image by
0.7 levels rms (PSNR 51 dB); 15/17-pixel controls are untouched.
"""
from __future__ import annotations

import numpy as np

PERIOD = 16


def deblock(image: np.ndarray, period: int = PERIOD, gain_sigma: float = 24.0, max_levels: float = 3.0,
            scale: float = 255.0) -> np.ndarray:
    """image: [C, H, W] float with `scale` = full range (255 for levels, 2 for [-1, 1]). Returns a float64 copy
    with the per-image periodic component removed; max_levels is in 8-bit levels."""
    from scipy.ndimage import gaussian_filter

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 3:
        raise ValueError("deblock expects [C, H, W]")
    _, h, w = img.shape
    if h < 2 * period or w < 2 * period:
        return img.copy()
    hc, wc = h // period * period, w // period * period
    limit = max_levels * scale / 255.0
    out = np.empty_like(img)
    for c, x in enumerate(img):
        hp = x - gaussian_filter(x, 8)
        rms = np.sqrt(gaussian_filter(hp * hp, 16)) + 1e-6 * scale
        u = hp / rms
        pattern = u[:hc, :wc].reshape(hc // period, period, wc // period, period).mean((0, 2))
        pattern -= pattern.mean()
        tiled = np.tile(pattern, (h // period + 1, w // period + 1))[:h, :w]
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
