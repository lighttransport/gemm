"""Texture padding for Pixal3D atlases.

Pixal3D bakes thousands of small UV charts into its atlas and inpaints the
background dark. Bilinear filtering at a chart's border then mixes in those
dark texels and draws thin cracks along every seam. Here the chart
coverage is rasterized from the triangles (dense barycentric samples),
and the chart colours are grown outwards a few texels (an average of
covered neighbours per step), so border lookups stay inside the chart's
colour.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def coverage(uvs: np.ndarray, tris: np.ndarray, size: tuple[int, int], density: float = 3.0,
             seed: int = 0) -> np.ndarray:
    """Texels covered by the UV triangles: vertices, edge midpoints and
    random barycentric samples (about `density` per covered texel)."""
    h, w = size
    g = np.random.default_rng(seed)
    a, b, c = (uvs[tris[:, k]] * np.array([w, h]) for k in range(3))
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    mask = np.zeros(h * w, bool)

    def mark(pts):
        x = np.clip(pts[:, 0].astype(np.int64), 0, w - 1)
        y = np.clip(pts[:, 1].astype(np.int64), 0, h - 1)
        mask[y * w + x] = True

    for pts in (a, b, c, (a + b) / 2, (b + c) / 2, (c + a) / 2, (a + b + c) / 3):
        mark(pts)
    n = np.minimum(np.ceil(area * density).astype(np.int64), 4000)
    rep = np.repeat(np.arange(len(tris)), n)
    if len(rep):
        u, v = g.random(len(rep)), g.random(len(rep))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        mark(a[rep] + u[:, None] * (b[rep] - a[rep]) + v[:, None] * (c[rep] - a[rep]))
    return mask.reshape(h, w)


def pad(image: np.ndarray, covered: np.ndarray, steps: int = 8) -> np.ndarray:
    """Grow covered texels' colours into their uncovered neighbours."""
    img = image.astype(np.float32).copy()
    cov = covered.copy()
    for _ in range(steps):
        acc = np.zeros_like(img)
        cnt = np.zeros(cov.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            sc = np.roll(np.roll(cov, dy, 0), dx, 1)
            si = np.roll(np.roll(img, dy, 0), dx, 1)
            acc += si * sc[..., None]
            cnt += sc
        grow = (~cov) & (cnt > 0)
        if not grow.any():
            break
        img[grow] = acc[grow] / cnt[grow][:, None]
        cov |= grow
    return np.clip(img + 0.5, 0, 255).astype(image.dtype)


def erode(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    for _ in range(steps):
        mask = mask & np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    return mask


def tint(image: np.ndarray, uvs: np.ndarray, tris: np.ndarray, rgb, strength: float = 0.85,
         grow: int = 2) -> tuple[np.ndarray, int]:
    """Blend the texels of `tris` (grown by `grow` texels, soft-edged)
    towards the sRGB colour `rgb`, keeping a little of their luminance
    detail. Returns the image and the number of texels touched."""
    h, w = image.shape[:2]
    m = coverage(uvs, tris, (h, w)).astype(np.float32)
    for _ in range(grow):
        nb = np.maximum.reduce([np.roll(m, 1, 0), np.roll(m, -1, 0), np.roll(m, 1, 1), np.roll(m, -1, 1)])
        m = np.maximum(m, 0.6 * nb)
    out = image.astype(np.float32).copy()
    rgb3 = out[..., :3]
    lum = rgb3.mean(-1, keepdims=True)
    detail = lum / max(float(lum[m > 0.99].mean()) if (m > 0.99).any() else 1.0, 1e-3)
    target = np.asarray(rgb, np.float32)[None, None, :] * np.clip(0.8 + 0.2 * detail, 0.6, 1.2)
    a = (m * strength)[..., None]
    out[..., :3] = rgb3 * (1 - a) + target * a
    return np.clip(out + 0.5, 0, 255).astype(image.dtype), int((m > 0).sum())


def pad_png(png: bytes, uvs: np.ndarray, tris: np.ndarray, steps: int = 10, border: int = 1,
            tints: list | None = None) -> tuple[bytes, dict]:
    """Pad a PNG atlas (glTF UVs: v down) and re-encode it. The charts'
    outermost `border` texels are replaced too: Pixal3D leaves them half
    baked (dark). `tints` is an optional list of (triangles, sRGB colour)
    to recolour first (texture.tint)."""
    img = Image.open(io.BytesIO(png))
    mode = img.mode
    arr = np.asarray(img)
    tinted = 0
    for t_tris, rgb in tints or ():
        if len(t_tris) and arr.ndim == 3:
            arr, n = tint(arr, uvs, t_tris, rgb)
            tinted += n
    cov = erode(coverage(uvs, tris, arr.shape[:2]), border)
    out = pad(arr if arr.ndim == 3 else arr[..., None], cov, steps)
    if arr.ndim == 2:
        out = out[..., 0]
    buf = io.BytesIO()
    Image.fromarray(out, mode).save(buf, format="PNG", compress_level=3)
    return buf.getvalue(), {"covered": round(float(cov.mean()), 4), "steps": steps, "tinted_texels": tinted}
