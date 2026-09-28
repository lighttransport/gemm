"""Original iris palette parameterized by hue-family U and brightness V.

The anchors are artist-selected colours. CPU and browser quantize lookup
coordinates to the same 256-pixel chart for deterministic colour matching."""
from __future__ import annotations

import numpy as np

# (U, linear RGB at V = 0.5) anchors along the hue axis.
ANCHORS = (
    (0.00, (0.130, 0.060, 0.032)),   # reddish dark brown
    (0.12, (0.190, 0.100, 0.042)),   # brown
    (0.22, (0.260, 0.150, 0.055)),   # amber / light brown
    (0.33, (0.200, 0.170, 0.075)),   # hazel / olive
    (0.43, (0.130, 0.160, 0.085)),   # green
    (0.52, (0.150, 0.170, 0.150)),   # grey-green
    (0.60, (0.175, 0.180, 0.185)),   # grey
    (0.70, (0.130, 0.165, 0.210)),   # blue-grey
    (0.82, (0.085, 0.150, 0.270)),   # blue
    (1.00, (0.140, 0.230, 0.360)),   # light blue
)
_U = np.array([a for a, _ in ANCHORS])
_RGB = np.array([c for _, c in ANCHORS])
LUMA = np.array([0.2126, 0.7152, 0.0722])


def chart_color(u, v):
    """Linear RGB for chart coordinates (arrays broadcast)."""
    u = np.clip(np.asarray(u, np.float64), 0.0, 1.0)
    v = np.clip(np.asarray(v, np.float64), 0.0, 1.0)
    base = np.stack([np.interp(u, _U, _RGB[:, k]) for k in range(3)], -1)
    value = np.exp2(3.2 * (v - 0.5))[..., None]              # 0 -> x0.33, 0.5 -> x1, 1 -> x3.0
    sat = (1.15 - 0.3 * v)[..., None]                         # lighter is a little greyer
    lum = (base @ LUMA)[..., None]
    return np.clip((lum + (base - lum) * sat) * value, 0.0, 1.0)


def chart_image(size: int = 256) -> np.ndarray:
    """The chart as sRGB uint8 (V down, as a texture's pixel rows):
    pixel (round((W-1)U), round((H-1)V)) is chart_color(U, V)."""
    from .optics import linear_to_srgb
    u = np.linspace(0.0, 1.0, size)
    uu, vv = np.meshgrid(u, u)
    return (linear_to_srgb(chart_color(uu, vv)) * 255.0 + 0.5).astype(np.uint8)


def chart_lookup(image: np.ndarray, u: float, v: float) -> np.ndarray:
    """Nearest pixel of the generated palette."""
    h, w = image.shape[:2]
    return image[int(round((h - 1) * v)), int(round((w - 1) * u))]


def pick(u, v, image: np.ndarray | None = None) -> np.ndarray:
    """The material's chart sample at (U, V), clamped, as linear RGB: the
    nearest pixel of the 256^2 chart (what the shader's texelFetch does)."""
    q = 255.0
    uq = np.round(np.clip(np.asarray(u, np.float64), 0, 1) * q) / q
    vq = np.round(np.clip(np.asarray(v, np.float64), 0, 1) * q) / q
    return chart_color(uq, vq)
