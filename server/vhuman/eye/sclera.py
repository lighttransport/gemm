"""Original branching-vessel textures and synthetic sclera albedo.

Vessels are random walks drawn with PIL. Four channels encode large vessels,
fine vessels, deep haze and tint variation. An optical-density approximation
colours the vessels independently of any external material graph."""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import noise, optics
from . import params as P

SCLERA_ALBEDO = np.array([0.74, 0.70, 0.66])      # linear; conjunctiva over a white sclera
START_R = (0.36, 0.56)
PROGRESS_OUTER = 0.34          # (vessel growth bookkeeping)
EXPOSED_HALF_ANGLE = math.radians(42)


@dataclass
class ScleraTextures:
    res: int
    masks: np.ndarray      # (res, res, 4) float32
    normal: np.ndarray     # (res, res, 3) float32
    seconds: float = 0.0


def _grow(g, res2: int, r_stop: float, count: int, width_px: tuple, step: float, branch_p: float,
          exposed_bias: float, start_r=START_R, reach=(0.55, 1.05)):
    """Branching random walks inward. Returns branches as
    (points (n, 2) in UV-centred coordinates, widths (n-1,), opacity,
    progress (n,)); progress is 0 beyond PROGRESS_OUTER and 1 at the limbus."""
    branches = []
    span = PROGRESS_OUTER - r_stop
    for _ in range(count):
        if g.random() < exposed_bias:
            phi = (0.0 if g.random() < 0.5 else math.pi) + g.normal(0, EXPOSED_HALF_ANGLE * 0.6)
        else:
            phi = g.uniform(0, 2 * math.pi)
        r0 = g.uniform(*start_r)
        stack = [(r0 * math.cos(phi), r0 * math.sin(phi), phi + math.pi + g.normal(0, 0.35),
                  g.uniform(*width_px), g.uniform(0.55, 1.0), 0.0, g.uniform(*reach))]
        while stack:
            x, y, heading, width, opacity, progress, limit = stack.pop()
            pts, widths, prog = [(x, y)], [], [progress]
            for _ in range(600):
                r = math.hypot(x, y)
                if r < r_stop or progress > limit or width < 0.25:
                    break
                inward = math.atan2(-y, -x)
                d = (inward - heading + math.pi) % (2 * math.pi) - math.pi
                heading += 0.07 * d + g.normal(0, 0.16)
                x, y = x + step * math.cos(heading), y + step * math.sin(heading)
                progress = min(1.0, max(0.0, (PROGRESS_OUTER - math.hypot(x, y)) / span))
                pts.append((x, y)); widths.append(width); prog.append(progress)
                width *= 0.988
                if g.random() < branch_p:
                    side = 1 if g.random() < 0.5 else -1
                    stack.append((x, y, heading + side * g.uniform(0.35, 1.0), width * g.uniform(0.55, 0.8),
                                  opacity * 0.85, progress, limit * g.uniform(0.7, 1.0)))
            if len(pts) > 1:
                branches.append((np.array(pts), np.array(widths), opacity, np.array(prog)))
    return branches


CHUNK = 6       # segments per polyline call: widths change slowly along a vessel


def _chunks(branches, res2: int) -> list:
    """Vessel branches as short polylines: (opacity, progress, width px, points)."""
    out = []
    for pts, widths, opacity, prog in branches:
        xy = np.stack([(0.5 + pts[:, 0]) * res2, (0.5 - pts[:, 1]) * res2], -1).tolist()
        starts = np.arange(0, len(widths), CHUNK)
        counts = np.minimum(starts + CHUNK, len(widths)) - starts
        w = np.add.reduceat(widths, starts) / counts
        pr = (np.add.reduceat(prog[:-1], starts) + prog[np.minimum(starts + counts, len(prog) - 1)]) / (counts + 1)
        for a, n, wa, pa in zip(starts.tolist(), counts.tolist(), w.tolist(), pr.tolist()):
            out.append((opacity * min(1.0, wa), pa, max(1, int(round(wa))), xy[a:a + n + 1]))
    return out


def _draw(chunks, res2: int, *, value: str, blur: float = 0.0) -> Image.Image:
    """Draw polylines; `value` is opacity (scaled down for sub-pixel
    widths), progress, or cover (1)."""
    img = Image.new("F", (res2, res2), 0.0)
    draw = ImageDraw.Draw(img)
    if value == "progress":
        # Lower progress last, so overlaps keep the vessel point nearest the fornix.
        chunks = sorted(chunks, key=lambda c: -c[1])
    k = {"opacity": 0, "progress": 1}.get(value)
    for c in chunks:
        draw.line(c[3], fill=float(c[k]) if k is not None else 1.0, width=c[2], joint="curve" if c[2] > 2 else None)
    if blur:
        # GaussianBlur needs an 8-bit image; the haze is soft, so 8 bits suffice.
        arr = np.asarray(img, np.float32)
        img8 = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))
        arr = np.asarray(img8.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255.0
        img = Image.fromarray(arr, "F")
    return img


def _down(img: Image.Image, res: int) -> np.ndarray:
    return np.asarray(img.resize((res, res), Image.BOX), np.float32)


_CACHE: "OrderedDict[str, ScleraTextures]" = OrderedDict()


def build(p: dict, res: int = 1024) -> ScleraTextures:
    key = P.structure_key(p, res)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    started = time.perf_counter()
    p = P.validate(p)
    seed = p["structure"]["seed"] * 104729 + (0 if p["optics"]["side"] == "left" else 1)
    g = noise.rng(seed, 0x5C1)
    res2 = res * 2
    scale = res2 / 2048.0
    r_stop = optics.CORNEA_SIZE_REF + 0.004
    step = 3.0 / res2 + 0.0015
    veins = _grow(g, res2, r_stop, 34, (1.6 * scale * 2, 4.2 * scale * 2), step, 0.045, 0.75)
    capillaries = _grow(g, res2, r_stop, 110, (0.5 * scale * 2, 1.4 * scale * 2), step, 0.03, 0.8,
                        start_r=(0.26, 0.46), reach=(0.3, 0.9))
    deep = _grow(g, res2, r_stop + 0.02, 12, (6 * scale * 2, 11 * scale * 2), step * 1.5, 0.02, 0.7,
                 reach=(0.4, 0.8))
    opacity = _down(_draw(_chunks(veins + capillaries, res2), res2, value="opacity"), res)
    natural = _down(_draw(_chunks(capillaries, res2), res2, value="opacity"), res)
    haze = _down(_draw(_chunks(deep, res2), res2, value="opacity", blur=10 * scale * 2), res) * 0.8

    c = (np.arange(res, dtype=np.float32) + 0.5) / res - 0.5
    r_uv = np.hypot(c[None, :], c[:, None])
    periphery = noise.smoothstep(0.24, 0.40, r_uv)
    low = noise.spectral((res // 4, res // 4), seed, beta=3.0, kmin=1, salt=11)
    low = np.asarray(Image.fromarray(low).resize((res, res), Image.BICUBIC), np.float32)
    haze = np.clip(haze + 0.25 * periphery * np.clip(low * 0.5 + 0.5, 0, 1), 0, 1)
    tint = np.clip(0.5 + 0.18 * low, 0, 1)
    masks = np.stack([np.clip(opacity, 0, 1), np.clip(natural, 0, 1), haze, tint], -1).astype(np.float32)
    grain = noise.spectral((res // 2, res // 2), seed, beta=2.6, kmin=6, salt=12)
    grain = np.asarray(Image.fromarray(grain).resize((res, res), Image.BICUBIC), np.float32)
    # Vessels lie in the conjunctiva: only a faint relief over a smooth, wet surface.
    v = masks[..., 0]
    soft = (v + np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1)) / 5.0
    height = 0.12 * soft + 0.004 * grain
    normal = noise.normal_map(height, strength=res / 1024.0)
    tex = ScleraTextures(res, masks, normal.astype(np.float32), time.perf_counter() - started)
    _CACHE[key] = tex
    while len(_CACHE) > 16:
        _CACHE.popitem(last=False)
    return tex


def sample(masks: np.ndarray, uv) -> np.ndarray:
    res = masks.shape[0]
    uv = np.asarray(uv, np.float64)
    layers = noise.bilinear(np.moveaxis(masks, -1, 0), uv[..., 0] * res - 0.5, uv[..., 1] * res - 0.5)
    return np.moveaxis(layers, 0, -1)


def rotate_uv(uv, rotation: float):
    """Sclera Rotation (0-1 of a turn) about the apex."""
    a = 2 * math.pi * rotation
    d = np.asarray(uv, np.float64) - 0.5
    c, s = math.cos(a), math.sin(a)
    return np.stack([0.5 + c * d[..., 0] + s * d[..., 1], 0.5 - s * d[..., 0] + c * d[..., 1]], -1)


def colorize(m: np.ndarray, r_uv, p: dict) -> np.ndarray:
    """Synthetic vessel optical density over user-tinted sclera albedo."""
    s, c = p["sclera"], p["cornea"]
    r = np.asarray(r_uv, np.float64)
    R, G, B, A = (m[..., k:k + 1].astype(np.float64) for k in range(4))
    coverage = noise.smoothstep(s["vascularity_coverage"], s["vascularity_coverage"] + .08, r)[..., None]
    density = .35 * G + .25 * B + s["vascularity_intensity"] * R * coverage
    base = SCLERA_ALBEDO * np.array(P.sclera_tint(p)) * (.92 + .16 * A)
    base *= np.exp(-density * np.array([.15, 1.8, 2.]))
    base *= 1 + (np.array(s["transmission_color"]) - 1) * transmission_amount(r, p)[..., None]
    base *= 1 + (np.array(c["limbus_color"]) - 1) * iris_cornea_mask(r, p)[..., None]
    return np.maximum(base, 0.)


def iris_cornea_mask(r_uv, p: dict):
    from .iris import cornea_mask
    return cornea_mask(r_uv, p)


def transmission_amount(r_uv, p: dict):
    """The transmission band beyond the cornea: circMask(r, Cornea Size, 0,
    TransmissionSpread) -- also where light wraps through the thin sclera."""
    return optics.circ_mask(r_uv, p["cornea"]["size"], 0.0, p["sclera"]["transmission_spread"])


def sampler(tex: "ScleraTextures", p: dict):
    """fn(r_uv, phi) -> sclera colour at eye-UV polar coordinates (with
    Sclera Rotation), for the iris-plane composite."""
    def fn(r_uv, phi):
        r_uv = np.asarray(r_uv, np.float64)
        uv = np.stack([0.5 + r_uv * np.cos(phi), 0.5 - r_uv * np.sin(phi)], -1)
        m = sample(tex.masks, rotate_uv(uv, p["sclera"]["rotation"]))
        return colorize(m, r_uv, p)
    return fn
