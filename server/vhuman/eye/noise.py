"""Procedural noise in numpy: FFT spectral synthesis, warping and sampling.

Spectral synthesis shapes white noise in the frequency domain, so the result
is periodic along both axes (the iris strip's angle wraps without a seam)
and anisotropy is one multiply: radial iris fibres are noise that is long
along the radius and short along the angle. One rfft2 of a 4096x384 strip
takes tens of milliseconds.
"""
from __future__ import annotations

import numpy as np


def rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0x7FFFFFFF, *salt])


def spectral(shape, seed: int, *, beta: float = 2.0, aniso=(1.0, 1.0), kmin: float = 1.0,
             kmax: float | None = None, salt: int = 0) -> np.ndarray:
    """Zero-mean, unit-variance noise of shape (H, W) with power ~ k^-beta.

    aniso = (ax, ay) scales the frequency axes before the power law: a large
    ay suppresses variation along rows (axis 0), stretching features along
    axis 0. kmin/kmax (cycles per period, after scaling) band-limit it."""
    h, w = shape
    g = rng(seed, 0x51E7, salt)
    white = g.standard_normal((h, w)).astype(np.float32)
    spec = np.fft.rfft2(white)
    ky = np.fft.fftfreq(h)[:, None] * h * aniso[1]
    kx = np.fft.rfftfreq(w)[None, :] * w * aniso[0]
    k = np.sqrt(kx * kx + ky * ky)
    amp = np.where(k >= kmin, np.power(np.maximum(k, 1e-6), -beta / 2.0), 0.0)
    if kmax is not None:
        amp *= np.exp(-(k / kmax) ** 2)
    out = np.fft.irfft2(spec * amp, s=(h, w)).astype(np.float32)
    out -= out.mean()
    std = out.std()
    return out / std if std > 0 else out


def spectral_smooth(shape, seed: int, *, factor: int = 4, **kw) -> np.ndarray:
    """spectral() for smooth (steep-spectrum) fields: synthesised at 1/factor
    resolution and upsampled bilinearly (still periodic along axis 1)."""
    h, w = shape
    small = spectral((max(8, h // factor), max(8, w // factor)), seed, **kw)
    sh, sw = small.shape
    y = (np.arange(h, dtype=np.float32)[:, None] + 0.5) * sh / h - 0.5
    x = (np.arange(w, dtype=np.float32)[None, :] + 0.5) * sw / w - 0.5
    return bilinear(small, np.broadcast_to(x, (h, w)), np.broadcast_to(y, (h, w)), wrap_x=True)


def periodic_1d(n: int, seed: int, *, beta: float = 2.0, kmin: float = 1.0, kmax: float | None = None,
                salt: int = 0) -> np.ndarray:
    """Unit-variance periodic noise of length n."""
    g = rng(seed, 0x1D, salt)
    spec = np.fft.rfft(g.standard_normal(n))
    k = np.arange(len(spec), dtype=np.float64)
    amp = np.where(k >= kmin, np.power(np.maximum(k, 1e-6), -beta / 2.0), 0.0)
    if kmax is not None:
        amp *= np.exp(-(k / kmax) ** 2)
    out = np.fft.irfft(spec * amp, n)
    out -= out.mean()
    return (out / (out.std() or 1.0)).astype(np.float32)


def bilinear(img: np.ndarray, x, y, *, wrap_x: bool = False, wrap_y: bool = False) -> np.ndarray:
    """Sample img (H, W) or (C, H, W) at float pixel coordinates (pixel
    centres at integers). Edges clamp unless wrapped."""
    stacked = img.ndim == 3
    h, w = img.shape[-2:]
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = (x - x0)
    fy = (y - y0)
    x0 = x0.astype(np.int64)
    y0 = y0.astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    if wrap_x:
        x0 %= w; x1 %= w
    else:
        x0 = np.clip(x0, 0, w - 1); x1 = np.clip(x1, 0, w - 1)
    if wrap_y:
        y0 %= h; y1 %= h
    else:
        y0 = np.clip(y0, 0, h - 1); y1 = np.clip(y1, 0, h - 1)
    flat = img.reshape((img.shape[0], -1) if stacked else (-1,))
    i00, i01, i10, i11 = y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1
    w00 = (1 - fx) * (1 - fy); w01 = fx * (1 - fy); w10 = (1 - fx) * fy; w11 = fx * fy
    if stacked:
        return (flat[:, i00] * w00 + flat[:, i01] * w01 + flat[:, i10] * w10 + flat[:, i11] * w11)
    return flat[i00] * w00 + flat[i01] * w01 + flat[i10] * w10 + flat[i11] * w11


def warp(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, *, wrap_x: bool = True) -> np.ndarray:
    """Domain warp: out[y, x] = img[y + dy, x + dx]."""
    h, w = img.shape[-2:]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return bilinear(img, xx + dx, yy + dy, wrap_x=wrap_x)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def normal_map(height: np.ndarray, strength: float, *, wrap_x: bool = False) -> np.ndarray:
    """Tangent-space normal (OpenGL convention, +Y up in the image: rows go
    down, so dh/dv is negated) from a height field in [0, 1]; returns
    float (H, W, 3) in [-1, 1]."""
    if wrap_x:
        gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5
    else:
        gx = np.gradient(height, axis=1)
    gy = np.gradient(height, axis=0)
    n = np.stack([-gx * strength, gy * strength, np.ones_like(height)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)
