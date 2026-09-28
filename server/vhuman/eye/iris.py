"""Original procedural iris structure and independently designed albedo.

Spectral noise produces radial fibres, crypts, furrows and spots. Masks hold
detail, occlusion, region and pupil channels; the iris fills a centred circle
of texture radius .5. Colour is a two-colour mixture with neutral occlusion,
a user-controlled limbal ring and a soft pupil opening."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import chart, noise, optics
from . import params as P

# Per-pattern style: the nine Pattern choices are nine structure styles.
PATTERNS = {
    "pattern_1": dict(name="fine stroma", fine=1.0, coarse=0.5, wave=0.5, crypt=0.6, furrow=0.6, spot=0.6,
                    lobes=16, coll_amp=0.030, ruff=1.0),
    "pattern_2": dict(name="crypt rich", fine=0.8, coarse=0.7, wave=0.8, crypt=1.6, furrow=0.5, spot=0.6,
                    lobes=12, coll_amp=0.045, ruff=1.0),
    "pattern_3": dict(name="furrowed", fine=0.7, coarse=0.8, wave=0.7, crypt=0.6, furrow=1.8, spot=0.8,
                    lobes=14, coll_amp=0.035, ruff=1.2),
    "pattern_4": dict(name="velvet", fine=0.6, coarse=0.35, wave=0.4, crypt=0.35, furrow=0.9, spot=0.5,
                    lobes=18, coll_amp=0.020, ruff=1.3),
    "pattern_5": dict(name="stellate", fine=1.2, coarse=0.9, wave=0.3, crypt=0.8, furrow=0.4, spot=0.4,
                    lobes=10, coll_amp=0.060, ruff=0.9),
    "pattern_6": dict(name="freckled", fine=0.8, coarse=0.6, wave=0.6, crypt=0.7, furrow=0.8, spot=2.2,
                    lobes=15, coll_amp=0.035, ruff=1.1),
    "pattern_7": dict(name="wide collarette", fine=0.9, coarse=0.7, wave=0.6, crypt=1.0, furrow=0.7, spot=0.6,
                    lobes=13, coll_amp=0.050, ruff=1.0),
    "pattern_8": dict(name="coarse fibres", fine=0.6, coarse=1.4, wave=1.0, crypt=0.9, furrow=0.8, spot=0.5,
                    lobes=11, coll_amp=0.045, ruff=1.0),
    "pattern_9": dict(name="mixed", fine=1.0, coarse=1.0, wave=0.8, crypt=1.1, furrow=1.1, spot=1.0,
                    lobes=14, coll_amp=0.040, ruff=1.1),
}

ANGLE_COLOR = np.array([0.035, 0.028, 0.026])      # the iris root / chamber angle beyond the limbus
OUTSIDE = (0.5, 1.0, 0.0, 0.0)                      # masks beyond the iris edge (t > 1)
PUPIL_MASKS = (0.5, 0.0, 1.0, 1.0)                  # masks inside the texture pupil (t < P_REF)

REC709 = np.array([0.2126, 0.7152, 0.0722])


@dataclass
class IrisTextures:
    res: int
    masks: np.ndarray       # (res, res, 4) float32, 0-1
    normal: np.ndarray      # (res, res, 3) float32, -1..1 (OpenGL convention)
    strip: np.ndarray | None  # (5, Nrho, Ntheta) float32: R, G, B, A, height (procedural only)
    seconds: float = 0.0
    photo: np.ndarray | None = None   # (res, res, 3) sRGB 0-1: an iris plate's own colours


def strip_shape(res: int) -> tuple[int, int]:
    """(Nrho, Ntheta): about one strip sample per output texel on the rim."""
    # Strip density follows the texture up to 1024^2 output; beyond that the
    # structure is upsampled (fibre detail past ~0.6 texel per sample is
    # not worth 2.5x the synthesis time).
    r_px = min(res * optics.IRIS_TEX_SCALE, 640.0)
    ntheta = int(math.ceil(2 * math.pi * r_px / 256.0)) * 256
    nrho = int(math.ceil(r_px * (1.0 - P.P_REF) * 1.1 / 32.0)) * 32
    return nrho, ntheta


def _blur_theta(img: np.ndarray, sigma: float) -> np.ndarray:
    """Periodic Gaussian blur along the angle (axis 1) via FFT."""
    nt = img.shape[1]
    k = np.fft.rfftfreq(nt) * nt
    kernel = np.exp(-2.0 * (math.pi * k * sigma / nt) ** 2)
    return np.fft.irfft(np.fft.rfft(img, axis=1) * kernel[None, :], n=nt, axis=1).astype(np.float32)


def _strands(nr: int, nt: int, g: np.random.Generator, n: int, start, end, width, bright, wave_amp,
             wave_freq, drift: np.ndarray) -> np.ndarray:
    """Draw n wavy radial strands into an (nr, nt) image.

    Each strand is a path x(rho) (columns) between start and end (rho),
    tapered at both ends. Strands are splatted linearly (two bincounts per
    width class) and each class is then blurred along the angle to its
    width, so the cost does not grow with the strand width. `drift` is a
    shared low-frequency sideways displacement (nr, nt) that makes
    neighbouring fibres wave together."""
    out = np.zeros((nr, nt), np.float32)
    if n == 0:
        return out
    rows = np.arange(nr)
    r = (rows / (nr - 1))[None, :]
    x0 = g.uniform(0, nt, n)
    phase = g.uniform(0, 2 * math.pi, n)
    xs = (x0[:, None] + wave_amp[:, None] * np.sin(2 * math.pi * wave_freq[:, None] * r + phase[:, None])
          + drift[rows[None, :], x0.astype(np.int64)[:, None] % nt])
    taper = (noise.smoothstep(start[:, None], start[:, None] + 0.05, r)
             * noise.smoothstep(end[:, None], end[:, None] - 0.08, r)) * bright[:, None]
    edges = np.quantile(width, [0.0, 1 / 3, 2 / 3, 1.0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (width >= lo) & (width <= hi)
        if not sel.any():
            continue
        active = taper[sel] > 0.0
        xa = xs[sel][active]
        wa = taper[sel][active]
        ra = np.broadcast_to(rows[None, :], (int(sel.sum()), nr))[active]
        base = np.floor(xa).astype(np.int64)
        frac = xa - base
        acc = np.bincount(ra * nt + base % nt, weights=wa * (1 - frac), minlength=nr * nt)
        acc += np.bincount(ra * nt + (base + 1) % nt, weights=wa * frac, minlength=nr * nt)
        sigma = 0.5 * float(width[sel].mean())
        # Blurring spreads each strand's unit weight: rescale so a strand
        # keeps a peak near its brightness whatever its width.
        out += _blur_theta(acc.reshape(nr, nt).astype(np.float32), sigma) * (sigma * math.sqrt(2 * math.pi))
    return out


def build_strip(p: dict, res: int) -> np.ndarray:
    """The structure layers in the polar strip (5, Nrho, Ntheta)."""
    p = P.validate(p)
    s = p["structure"]
    style = PATTERNS[p["iris"]["pattern"]]
    seed = s["seed"] * 7919 + P.IRIS_PATTERNS.index(p["iris"]["pattern"])
    g = noise.rng(seed, 0xA11)
    nr, nt = strip_shape(res)
    rho = np.linspace(0.0, 1.0, nr, dtype=np.float32)[:, None]
    t_row = (1.0 - P.P_REF) / (nr - 1)          # iris-radius units per row; shapes below use these units

    # Collarette: a zigzag ring; its radius sets the two zones.
    c0 = 0.16 + 0.24 * s["collarette"]
    lobes = style["lobes"]
    wig = noise.periodic_1d(nt, seed, beta=2.0, kmin=lobes * 0.6, kmax=lobes * 1.8, salt=1)
    wig += 0.35 * noise.periodic_1d(nt, seed, beta=1.0, kmin=lobes * 2, kmax=lobes * 5, salt=2)
    rc = c0 + style["coll_amp"] * wig[None, :]
    zone = 1.0 - noise.smoothstep(rc - 0.09, rc + 0.07, rho)
    ridge = np.exp(-((rho - rc) / 0.02) ** 2)

    low = noise.spectral_smooth((nr, nt), seed, beta=3.2, kmin=1, salt=6)
    grain = noise.spectral((nr, nt), seed, beta=0.6, aniso=(1.0, 4.0), kmin=60, salt=3)
    cavity = np.zeros((nr, nt), np.float32)
    height = 0.5 + 0.16 * ridge
    crypt_body = np.zeros((nr, nt), np.float32)
    pigment = np.zeros((nr, nt), np.float32)

    # Fuchs' crypts: irregular lens-shaped openings, mostly just outside the
    # collarette; fibres drawn afterwards cross them.
    n_crypts = int(round(s["crypts"] * style["crypt"] * 46))
    for _ in range(n_crypts):
        near = g.random() < 0.65
        cy = float(np.clip((c0 + g.uniform(0.02, 0.16)) if near else g.uniform(0.40, 0.85), 0.05, 0.93))
        cx = g.uniform(0, nt)
        length = g.uniform(0.05, 0.14) * (1.15 if near else 0.8)        # iris-radius units
        width = g.uniform(0.016, 0.05)
        t_c = P.P_REF + cy * (1 - P.P_REF)
        col_t = 2 * math.pi * t_c / nt
        half_cols = int(width / col_t * 1.6) + 2
        half_rows = int(length / t_row * 0.5 * 1.6) + 2
        row_c = cy * (nr - 1)
        r0, r1 = max(0, int(row_c) - half_rows), min(nr, int(row_c) + half_rows + 1)
        cidx = np.arange(int(cx) - half_cols, int(cx) + half_cols + 1)
        rr = (np.arange(r0, r1)[:, None] - row_c) * t_row / (length * 0.5)
        cc = (cidx[None, :] - cx) * col_t / (width * 0.5)
        ang = np.arctan2(rr, cc + 1e-6)
        edge = (0.16 * np.sin(3 * ang + g.uniform(0, 6.3)) + 0.10 * np.sin(5 * ang + g.uniform(0, 6.3))
                + 0.12 * grain[r0:r1][:, cidx % nt])
        f = 1.0 - rr * rr - np.abs(cc) ** 1.8 + edge
        body = noise.smoothstep(-0.1, 0.7, f) * g.uniform(0.4, 0.9)
        rim = np.exp(-((f + 0.04) / 0.10) ** 2) * (f < 0.2)
        cols = cidx % nt
        crypt_body[r0:r1, cols] = np.maximum(crypt_body[r0:r1, cols], body)
        height[r0:r1, cols] += 0.10 * rim
        cavity[r0:r1, cols] = np.maximum(cavity[r0:r1, cols], 0.25 * rim)

    # Radial trabeculae: wavy strands. Ciliary ones start near the
    # collarette and run to the root; pupillary ones are finer and straighter.
    drift = noise.spectral_smooth((nr, nt), seed, beta=3.6, kmin=2, salt=5) * (style["wave"] * nt / 700.0)
    n_cil = int(nt * 0.30 * (0.4 + 0.6 * style["coarse"]))
    x_start = rc[0, g.integers(0, nt, n_cil)]
    cil = _strands(nr, nt, g, n_cil,
                   start=np.clip(x_start + g.uniform(-0.04, 0.10, n_cil), 0.05, 0.9),
                   end=g.uniform(0.62, 1.05, n_cil),
                   width=nt * g.uniform(0.0007, 0.0024, n_cil) * (0.7 + 0.5 * style["coarse"]),
                   bright=g.uniform(0.25, 1.0, n_cil),
                   wave_amp=nt * g.uniform(0.0002, 0.0014, n_cil) * style["wave"],
                   wave_freq=g.uniform(0.5, 2.0, n_cil), drift=drift)
    n_pup = int(nt * 0.28 * (0.4 + 0.6 * style["fine"]))
    pup = _strands(nr, nt, g, n_pup,
                   start=g.uniform(0.0, 0.06, n_pup), end=np.clip(c0 + g.uniform(-0.05, 0.06, n_pup), 0.08, 0.6),
                   width=nt * g.uniform(0.0004, 0.0012, n_pup), bright=g.uniform(0.3, 1.0, n_pup),
                   wave_amp=nt * g.uniform(0.0002, 0.0012, n_pup), wave_freq=g.uniform(0.5, 2.0, n_pup),
                   drift=drift * 0.3)
    contrast = 0.4 + 1.2 * s["fibers"]
    fib = 1.0 - np.exp(-(cil * 0.55 + pup * 0.5))
    fib = np.clip(fib + 0.08 * grain, 0.0, 1.0)

    struct = (0.5 + contrast * 0.42 * (fib - 0.45) + 0.14 * ridge + 0.10 * low
              - 0.32 * crypt_body * (1.0 - 0.5 * fib))
    cavity = np.maximum(cavity, crypt_body * (1.0 - 0.55 * fib))
    cavity = np.maximum(cavity, 0.30 * contrast * np.clip(0.35 - fib, 0.0, 1.0) * (1.0 - zone))
    height = height + 0.22 * fib - 0.42 * crypt_body * (1.0 - 0.5 * fib)

    # Contraction furrows: interrupted concentric arcs in the outer ciliary zone.
    n_furrows = int(round(s["furrows"] * style["furrow"] * 9))
    for k in range(n_furrows):
        rf = g.uniform(0.55, 0.95)
        w = g.uniform(0.010, 0.022)
        wob = 0.012 * noise.periodic_1d(nt, seed, beta=2.5, kmin=2, kmax=20, salt=100 + k)
        extent = noise.smoothstep(0.1, 0.6, noise.periodic_1d(nt, seed, beta=2.0, kmin=1, kmax=6,
                                                              salt=200 + k) + g.uniform(-0.3, 0.5))
        row_c = rf * (nr - 1)
        half = int((w * 3 + 0.02) * (nr - 1)) + 2
        r0, r1 = max(0, int(row_c) - half), min(nr, int(row_c) + half + 1)
        band = np.exp(-((rho[r0:r1] - rf - wob[None, :]) / w) ** 2) * extent[None, :]
        cavity[r0:r1] = np.maximum(cavity[r0:r1], 0.35 * band)
        height[r0:r1] -= 0.2 * band
        struct[r0:r1] -= 0.08 * band

    # Pupillary ruff: the crinkled dark frill at the pupil margin.
    crinkle = noise.periodic_1d(nt, seed, beta=0.8, kmin=60, kmax=nt / 6, salt=7)
    rr_edge = (0.028 + 0.010 * crinkle * style["ruff"])[None, :] * style["ruff"]
    ruff = 1.0 - noise.smoothstep(rr_edge - 0.008, rr_edge + 0.008, rho)
    pigment = np.maximum(pigment, ruff)
    height += 0.12 * ruff * (0.5 + 0.5 * np.clip(crinkle[None, :], -1, 1))

    # Pigment spots (freckles), in the ciliary zone.
    n_spots = int(round(s["freckles"] * style["spot"] * 26))
    for _ in range(n_spots):
        cy = g.uniform(min(c0 + 0.08, 0.9), 0.96)
        cx = g.uniform(0, nt)
        radius = g.uniform(0.010, 0.038)
        t_c = P.P_REF + cy * (1 - P.P_REF)
        col_t = 2 * math.pi * t_c / nt
        half_cols = int(radius / col_t * 2) + 2
        half_rows = int(radius / t_row * 2) + 2
        row_c = cy * (nr - 1)
        r0, r1 = max(0, int(row_c) - half_rows), min(nr, int(row_c) + half_rows + 1)
        cidx = np.arange(int(cx) - half_cols, int(cx) + half_cols + 1)
        d = np.hypot((np.arange(r0, r1)[:, None] - row_c) * t_row, (cidx[None, :] - cx) * col_t) / radius
        blob = 1.0 - noise.smoothstep(0.55, 1.0, d + 0.18 * grain[r0:r1][:, cidx % nt])
        pigment[r0:r1, cidx % nt] = np.maximum(pigment[r0:r1, cidx % nt], blob * g.uniform(0.5, 0.9))

    # The stroma thins towards the root: soften structure near the edge.
    edge_fade = noise.smoothstep(1.0, 0.88, rho)
    struct = 0.5 + (struct - 0.5) * (0.6 + 0.4 * edge_fade)
    detail = np.clip(0.5 + 1.8 * (struct - 0.5), 0, 1)
    shadow = 1.0 - np.clip(0.85 * cavity + 0.95 * pigment, 0, 1)
    # Secondary region: a soft radial falloff that follows the collarette's
    # zigzag and leaks out along bright fibres, so Colour Blend Softness
    # shapes a gradient rather than a hard edge.
    secondary = np.clip(1.0 - rho / (1.7 * rc), 0.0, 1.0) ** 0.7
    secondary = np.clip(secondary + 0.18 * (detail - 0.5) * (rho < 1.9 * rc), 0.0, 1.0)
    return np.stack([detail, shadow, secondary, np.zeros_like(detail),
                     np.clip(height, 0, 1)]).astype(np.float32)


@lru_cache(maxsize=8)
def _gather_coords(res: int, nr: int, nt: int):
    """Strip coordinates of every square texel (cached per size), plus the
    bilinear taps: four flat indices and weights into an (nr, nt) layer."""
    c = (np.arange(res, dtype=np.float32) + 0.5) / res - 0.5
    dx = c[None, :] / optics.IRIS_TEX_SCALE
    dy = -c[:, None] / optics.IRIS_TEX_SCALE
    t = np.sqrt(dx * dx + dy * dy)
    phi = np.mod(np.arctan2(dy, dx), 2 * math.pi)
    x = (phi / (2 * math.pi) * nt).astype(np.float32)
    y = (np.clip((t - P.P_REF) / (1.0 - P.P_REF), 0.0, 1.0) * (nr - 1)).astype(np.float32)
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    x0 = x0.astype(np.int64) % nt
    x1 = (x0 + 1) % nt
    y0 = np.clip(y0.astype(np.int64), 0, nr - 1)
    y1 = np.clip(y0 + 1, 0, nr - 1)
    taps = ((y0 * nt + x0, (1 - fx) * (1 - fy)), (y0 * nt + x1, fx * (1 - fy)),
            (y1 * nt + x0, (1 - fx) * fy), (y1 * nt + x1, fx * fy))
    return t, taps


def polar_to_square(layers: np.ndarray, res: int, pupil_values, outside_values) -> np.ndarray:
    """Resample polar layers (C, Nrho, Ntheta) -- rho 0 at the pupil margin
    (texture pupil ratio P_REF), 1 at the iris edge -- to the square iris
    layout (C, res, res); pupil and beyond-the-edge texels take the given
    per-channel values."""
    nr, nt = layers.shape[1:]
    t, taps = _gather_coords(res, nr, nt)
    flat = layers.reshape(layers.shape[0], -1)
    out = sum(np.take(flat, idx, axis=1) * w for idx, w in taps).astype(np.float32)
    inside = noise.smoothstep(P.P_REF - 0.004, P.P_REF + 0.004, t)
    outside = noise.smoothstep(1.0, 1.03, t)
    for ch, (pupil_v, out_v) in enumerate(zip(pupil_values, outside_values)):
        out[ch] = pupil_v + (out[ch] - pupil_v) * inside
        out[ch] = out[ch] + (out_v - out[ch]) * outside
    return out


def strip_to_square(strip: np.ndarray, res: int) -> tuple[np.ndarray, np.ndarray]:
    """One stacked gather: (res, res, 4) masks and the (res, res) height."""
    layers = polar_to_square(strip, res, PUPIL_MASKS + (0.5,), OUTSIDE + (0.5,))
    return np.moveaxis(layers[:4], 0, -1).astype(np.float32), layers[4].astype(np.float32)


_CACHE: "OrderedDict[str, IrisTextures]" = OrderedDict()


def build(p: dict, res: int = 1024) -> IrisTextures:
    """Structure textures for the non-live parameters (LRU cached)."""
    import time
    key = P.structure_key(p, res)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    started = time.perf_counter()
    strip = build_strip(p, res)
    masks, height = strip_to_square(strip, res)
    normal = noise.normal_map(height, strength=res / 64.0)
    tex = IrisTextures(res, masks, normal.astype(np.float32), strip, time.perf_counter() - started)
    _CACHE[key] = tex
    while len(_CACHE) > 16:
        _CACHE.popitem(last=False)
    return tex


# ---- sampling and colour (mirrored by the web shader) ----------------------------

def sample_masks(masks: np.ndarray, t, phi) -> np.ndarray:
    """Bilinear sample of a square iris-layout texture (res, res, C) at iris
    radius t, angle phi."""
    res = masks.shape[0]
    t = np.asarray(t, np.float32)
    u = 0.5 + optics.IRIS_TEX_SCALE * t * np.cos(phi)
    v = 0.5 - optics.IRIS_TEX_SCALE * t * np.sin(phi)
    layers = noise.bilinear(np.moveaxis(masks, -1, 0), u * res - 0.5, v * res - 0.5)
    return np.moveaxis(layers, 0, -1)


def desaturate(c, f):
    """Rec.709 luminance interpolation; negative amounts increase saturation."""
    c = np.asarray(c, np.float64)
    lum = (c @ REC709)[..., None]
    f = np.asarray(f, np.float64)
    if f.ndim:
        f = f[..., None]
    return c + (lum - c) * f


def custom_iris(m: np.ndarray, p: dict, photo: np.ndarray | None = None) -> np.ndarray:
    """Two user colours mixed by a procedural mask and neutral detail shading."""
    i = p["iris"]
    detail, shadow, region = (m[..., k].astype(np.float64) for k in range(3))
    primary = chart.pick(i["primary_color_u"], i["primary_color_v"])
    secondary = chart.pick(i["secondary_color_u"], i["secondary_color_v"])
    driver = region if i["blend_method"] == "Radial" else detail
    half = i["color_blend_softness"] * .5
    blend = noise.smoothstep(i["color_blend"] - half, i["color_blend"] + half, driver)[..., None]
    c = primary * (1 - blend) + secondary * blend
    c = c * (.65 + .7 * detail[..., None]) * (1 - .6 * i["shadow_details"] * (1 - shadow[..., None]))
    if photo is not None:
        c = c * (1 - i["photo_mix"]) + photo * i["photo_mix"]
    return desaturate(c, 1 - i["global_saturation"])


def iris_base(tex: "IrisTextures", t, phi, p: dict) -> np.ndarray:
    """Independent iris albedo: colour, outer ring, then a soft pupil opening."""
    i = p["iris"]
    phi = np.asarray(phi, np.float64) - 2 * math.pi * i["rotation"]
    radius = np.minimum(optics.pupil_scale(t, P.pupil_scale(p)), 1.)
    masks = sample_masks(tex.masks, radius, phi)
    photo = None if tex.photo is None else optics.srgb_to_linear(sample_masks(tex.photo, radius, phi))
    color = custom_iris(masks, p, photo) * np.array(i["global_tint"])
    ring = noise.smoothstep(i["limbal_ring_size"] - i["limbal_ring_softness"],
                            i["limbal_ring_size"] + i["limbal_ring_softness"], radius)[..., None]
    color *= 1 + (np.array(i["limbal_ring_color"]) - 1) * ring
    edge = .005 + .04 * p["pupil"]["feather"]
    opening = noise.smoothstep(P.P_REF - edge, P.P_REF + edge, radius)
    return color * opening[..., None]


def cornea_mask(r_uv, p: dict):
    """Smooth transition across the user-selected iris radius."""
    size, half = p["cornea"]["size"], .5 * p["cornea"]["limbus_softness"]
    return 1 - noise.smoothstep(size - half, size + half, r_uv)


def iris_plane_color(tex: "IrisTextures", t, phi, p: dict, sclera_fn) -> np.ndarray:
    """Blend iris and sclera albedo across the limbus."""
    r = np.asarray(t, np.float64) * p["cornea"]["size"]
    weight = cornea_mask(r, p)[..., None]
    return iris_base(tex, t, phi, p) * weight + sclera_fn(r, phi) * (1 - weight)


def bake_color(tex: IrisTextures, p: dict, res: int | None = None, sclera_fn=None) -> np.ndarray:
    """The iris texture coloured at the chosen dilation (linear RGB), for
    the portable GLB's opaque iris mesh (no caustics compensation)."""
    p = P.validate(p)
    res = res or tex.res
    c = (np.arange(res, dtype=np.float64) + 0.5) / res - 0.5
    dx = c[None, :] / optics.IRIS_TEX_SCALE
    dy = -c[:, None] / optics.IRIS_TEX_SCALE
    t = np.sqrt(dx * dx + dy * dy)
    phi = np.arctan2(dy, dx)
    if sclera_fn is None:
        sclera_fn = lambda r_uv, ph: np.broadcast_to(np.array([0.62, 0.58, 0.55]), np.shape(r_uv) + (3,))  # noqa: E731
    col = iris_plane_color(tex, t, phi, p, sclera_fn)
    # Beyond the limbus the iris plane meets the chamber angle (dark).
    limbus_t = optics.CORNEA_SIZE_REF / p["cornea"]["size"]
    angle = noise.smoothstep(limbus_t, limbus_t + 0.03, t)[..., None]
    return (col + (ANGLE_COLOR - col) * angle).astype(np.float32)
