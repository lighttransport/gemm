"""Original portrait-preserving skin textures with procedural artist controls."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..eye import optics
from ..eye.geometry import compute_tangents
from . import texture, relight

REGIONS = ("scalp", "forehead", "nose", "under_eye", "cheeks", "lips", "chin", "ears")
FILES = ("skin_basecolor.png", "skin_normal.png", "skin_orm.png", "skin_mask.png", "skin.json")


def validate(value=None):
    p = {"enabled": True, "tone_u": .5, "tone_v": .5, "tone_strength": 0.,
         "roughness": 1., "pore_strength": .45, "delight_strength": 0., "seed": 11,
         "freckles": {"density": 0., "strength": .45, "saturation": .5, "tone_shift": 0., "mask": "cheeks_nose"},
         "regions": {r: {"redness": 0., "saturation": 0., "lightness": 0.} for r in REGIONS}}
    if value is None:
        return p
    if not isinstance(value, dict):
        raise ValueError("skin must be an object")
    if set(value) - set(p):
        raise ValueError("unknown skin parameters: " + ", ".join(sorted(set(value) - set(p))))
    for key, val in value.items():
        if key == "regions":
            if not isinstance(val, dict) or set(val) - set(REGIONS):
                raise ValueError("invalid skin regions")
            for region, controls in val.items():
                if not isinstance(controls, dict) or set(controls) - set(p[key][region]):
                    raise ValueError("invalid skin region controls")
                p[key][region].update(controls)
        elif key == "freckles":
            if not isinstance(val, dict) or set(val) - set(p[key]):
                raise ValueError("invalid freckles controls")
            p[key].update(val)
        else:
            p[key] = val
    if not isinstance(p["enabled"], bool):
        raise ValueError("skin.enabled must be boolean")
    def bounded(x, lo, hi, name):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or not lo <= x <= hi:
            raise ValueError(f"skin.{name} must be in [{lo}, {hi}]")
    for k in ("tone_u", "tone_v", "tone_strength", "pore_strength", "delight_strength"):
        bounded(p[k], 0, 1, k)
    bounded(p["roughness"], .2, 2., "roughness")
    bounded(p["seed"], 0, 2**31 - 1, "seed")
    if not isinstance(p["seed"], int):
        raise ValueError("skin.seed must be an integer")
    if p["freckles"]["mask"] not in ("cheeks_nose", "cheeks", "full_face"):
        raise ValueError("skin.freckles.mask must be cheeks_nose, cheeks or full_face")
    for k, v in p["freckles"].items():
        if k == "mask":
            continue
        bounded(v, -1 if k == "tone_shift" else 0, 1, "freckles." + k)
    for r, controls in p["regions"].items():
        for k, v in controls.items():
            bounded(v, -1, 1, "regions." + r + "." + k)
    return p


def tone_color(u, v):
    """Original two-dimensional chart: cool/warm undertone × pigmentation.

    Values are linear reflectance, not display RGB. The default bake keeps
    the portrait's colour; the chart only acts when tone_strength > 0.
    """
    light = np.array([.64, .42, .31])
    dark = np.array([.075, .035, .018])
    rgb = light * (1 - v) + dark * v
    return np.clip(rgb * np.array([1 + .14 * (u - .5), 1., 1 - .3 * (u - .5)]), 0, 1)


def atlas_samples(uvs, triangles, shape):
    """Yield exact texel-centre barycentrics in bounded triangle batches.

    glTF UV v points down. Duplicate UV seams are preserved. Degenerate
    UV triangles are ignored; no interpolation across chart boundaries.
    """
    h, w = shape
    uv = np.asarray(uvs) * [w, h] - .5
    for first in range(0, len(triangles), 2048):
        tri = triangles[first:first + 2048]
        pp = uv[tri]
        lo = np.maximum(np.ceil(pp.min(1)).astype(int), 0)
        hi = np.minimum(np.floor(pp.max(1)).astype(int), [w - 1, h - 1])
        span = np.maximum(hi - lo + 1, 0)
        counts = span.prod(1)
        # A pathological large chart is handled separately, row by row.
        for start in range(0, len(tri), 1 if counts.sum() > 2_000_000 else len(tri)):
            end = start + (1 if counts.sum() > 2_000_000 else len(tri))
            cc = counts[start:end]
            total = int(cc.sum())
            if not total:
                continue
            # Bound the expansion even for a full-atlas triangle.
            ends = np.cumsum(cc)
            for offset in range(0, total, 1_000_000):
                seq = np.arange(offset, min(total, offset + 1_000_000))
                idx = np.searchsorted(ends, seq, side="right")
                local = seq - np.r_[0, ends[:-1]][idx]
                ii = idx + start
                x = lo[ii, 0] + local % span[ii, 0]
                y = lo[ii, 1] + local // span[ii, 0]
                a, b, c = pp[ii, 0], pp[ii, 1], pp[ii, 2]
                den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
                safe = np.where(abs(den) > 1e-12, den, 1.)
                u = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / safe
                v = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / safe
                weights = np.stack([u, v, 1 - u - v], 1)
                inside = (weights >= -1e-7).all(1) & (abs(den) > 1e-12)
                yield y[inside], x[inside], tri[ii[inside]], weights[inside]


def _hash(cell, seed):
    # uint arithmetic is deterministic on every supported NumPy platform.
    c = np.asarray(cell, np.int64).astype(np.uint32)
    n = c[:, 0] * np.uint32(73856093) ^ c[:, 1] * np.uint32(19349663) ^ c[:, 2] * np.uint32(83492791)
    n ^= np.uint32(seed)
    n ^= n >> np.uint32(16)
    n *= np.uint32(0x7feb352d)
    n ^= n >> np.uint32(15)
    return n.astype(np.float64) / 4294967296.


def cellular(points, spacing, seed, density=1.):
    """Smooth compact spots and spatial derivatives, continuous at cell seams."""
    q = points / spacing
    base = np.floor(q).astype(np.int64)
    value = np.zeros(len(q))
    grad = np.zeros_like(q)
    # Centres stay in the middle 80% of each cell. A radius .55 kernel can
    # only touch its own and immediately neighbouring cells.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = base + [dx, dy, dz]
                active = _hash(cell, seed + 31) < density
                if not active.any():
                    continue
                ids = np.flatnonzero(active)
                jitter = np.stack([_hash(cell[ids], seed + k) for k in (101, 307, 701)], 1)
                delta = q[ids] - (cell[ids] + .1 + .8 * jitter)
                d2 = (delta * delta).sum(1) / .3025
                s = np.maximum(1 - d2, 0)
                value[ids] += s ** 3
                grad[ids] += -6 * delta / (.3025 * spacing) * (s * s)[:, None]
    return value, grad


def portrait_regions(portrait, eyes):
    """Measure a lip ellipse from portrait chroma in an eye-aligned mouth ROI.

    Lips tend to have more red/blue relative to green than nearby skin.
    Compare against both cheeks to avoid a fixed skin-tone threshold. This
    is a bounded frontal-portrait heuristic, with an explicit fallback when
    the chromatic support is weak; it is not a semantic face parser.
    """
    im = np.asarray(Image.open(portrait).convert("RGBA"), float) / 255
    centers = np.array(sorted([[e.cx, e.cy] for e in eyes]))
    origin = centers.mean(0)
    direction = centers[-1] - centers[0]
    ipd = max(float(np.linalg.norm(direction)), 1.)
    right = direction / ipd
    down = np.array([-right[1], right[0]])
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    delta = np.stack([xx, yy], -1) - origin
    x, y = delta @ right / ipd, delta @ down / ipd
    rgb = np.maximum(im[..., :3], .02)
    chroma = np.log(rgb[..., 0]) + np.log(rgb[..., 2]) - 2 * np.log(rgb[..., 1])
    cheek = (np.abs(np.abs(x) - .58) < .15) & (np.abs(y - .52) < .15) & (im[..., 3] > .9)
    fallback = {"lips": [0., 1.05, .4, .16], "method": "anatomical fallback", "confidence": 0.}
    if cheek.sum() < 16:
        return fallback
    reference = float(np.median(chroma[cheek]))
    roi = (np.abs(x) < .55) & (y > .78) & (y < 1.50) & (im[..., 3] > .9)
    # Dark nostrils/beard and clipped highlights are poor colour evidence.
    valid = roi & (im[..., :3].min(-1) > .06) & (im[..., :3].max(-1) < .98)
    score = np.clip(chroma - reference - .04, 0, .8) * valid
    score *= np.exp(-2 * (x / .5) ** 2)
    if not np.any(score):
        return fallback
    # A small blur merges lip texture into a stable chromatic region.
    from .landmarks import _gauss
    smooth = np.maximum(_gauss(score, max(ipd * .018, .8)), 0) * roi
    peak = float(smooth.max())
    if peak < .045:
        return fallback
    peak_y, peak_x = np.unravel_index(np.argmax(smooth), smooth.shape)
    local = (np.abs(y - y[peak_y, peak_x]) < .18) & (smooth > peak * .22)
    weights = smooth * local
    total = weights.sum()
    if total < ipd * ipd * .0005:
        return fallback
    cx, cy = float((weights * x).sum() / total), float((weights * y).sum() / total)
    # The chromatic core is narrower than the lip: dark corners and the
    # upper vermilion have weaker colour evidence. Cover that envelope,
    # rather than interpreting the high-score core as the complete mouth.
    sx = float(np.sqrt((weights * (x - cx) ** 2).sum() / total) * 4.4)
    sy = float(np.sqrt((weights * (y - cy) ** 2).sum() / total) * 3.7)
    if abs(cx) > .15 or not .85 < cy < 1.42 or not .22 < sx < .85 or not .06 < sy < .36:
        return fallback
    return {"lips": [cx, cy, max(sx, .28), max(sy, .10)],
            "method": "portrait lip chroma", "confidence": min(peak / .20, 1.)}


def region_masks(pix, eyes, features=None):
    """Soft facial regions in portrait space, scaled by detected IPD.

    These are approximate anatomical masks, not a face parser. Colour
    gating below protects hair, brows and clothing from skin controls.
    """
    centres = np.array(sorted([[e.cx, e.cy] for e in eyes]))
    centre = centres.mean(0)
    direction = centres[-1] - centres[0]
    ipd = max(float(np.linalg.norm(direction)), 1.)
    right = direction / ipd
    down = np.array([-right[1], right[0]])
    # Follow portrait roll so bilateral accents stay attached to the face.
    delta = np.asarray(pix) - centre
    x, y = delta @ right / ipd, delta @ down / ipd
    def blob(cx, cy, sx, sy):
        return np.exp(-2 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    lip = features["lips"] if features is not None else (0, 1.05, .4, .16)
    return {"scalp": blob(0, -1.3, 1.1, .5), "forehead": blob(0, -.65, .8, .48),
            "nose": blob(0, .45, .23, .45),
            "under_eye": np.maximum(blob(-.5, .22, .35, .18), blob(.5, .22, .35, .18)),
            "cheeks": np.maximum(blob(-.65, .55, .45, .42), blob(.65, .55, .45, .42)),
            "lips": blob(*lip), "chin": blob(0, 1.4, .48, .28),
            "ears": np.maximum(blob(-1.05, .3, .25, .7), blob(1.05, .3, .25, .7))}


def detail_mask(pix, eyes, regions):
    """Keep pores/freckles off lips and eye openings, with soft boundaries.

    Colour adjustments retain their independent lip-region control. This
    mask only gates surface detail, where skin-coloured lips otherwise
    receive the same pores as cheeks. It follows the eye-aligned regions.
    """
    mask = 1.0 - np.clip(regions["lips"] * 3.0, 0, 1)
    centres = np.array(sorted([[e.cx, e.cy] for e in eyes]))
    direction = centres[-1] - centres[0]
    ipd = max(float(np.linalg.norm(direction)), 1.)
    right = direction / ipd
    down = np.array([-right[1], right[0]])
    for eye in eyes:
        delta = np.asarray(pix) - [eye.cx, eye.cy]
        # A narrow eye-opening ellipse leaves the surrounding lid skin.
        rx = max(float(getattr(eye, "r", .10 * ipd)) * 1.65, 1.)
        ry = max(float(getattr(eye, "r", .10 * ipd)) * .70, 1.)
        q = np.sqrt((delta @ right / rx) ** 2 + (delta @ down / ry) ** 2)
        t = np.clip((q - .9) / .25, 0, 1)
        mask *= t * t * (3 - 2 * t)
    return mask


def portrait_color(portrait, eyes):
    im = np.asarray(Image.open(portrait).convert("RGBA"), float) / 255
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    masks = region_masks(np.stack([xx.ravel(), yy.ravel()], 1), eyes)
    cheek = masks["cheeks"].reshape(xx.shape) > .5
    valid = cheek & (im[..., 3] > .9)
    rgb = im[..., :3][valid]
    return optics.srgb_to_linear(np.median(rgb, 0)) if len(rgb) else tone_color(.5, .5)


def portrait_brow_exclusion(portrait, eyes):
    """Conservative dark-hair evidence inside eye-aligned eyebrow regions.

    Compare each brow with its local bright skin, avoiding a fixed skin-tone
    threshold. This does not identify pale brows or perform hair segmentation.
    A uniform or weak-contrast region contributes no exclusion.
    """
    im = np.asarray(Image.open(portrait).convert("RGBA"), float) / 255
    centers = np.array(sorted([[e.cx, e.cy] for e in eyes]))
    direction = centers[-1] - centers[0]
    ipd = max(float(np.linalg.norm(direction)), 1.)
    right = direction / ipd
    down = np.array([-right[1], right[0]])
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    delta = np.stack([xx, yy], -1) - centers.mean(0)
    x, y = delta @ right / ipd, delta @ down / ipd
    lum = optics.srgb_to_linear(im[..., :3]) @ optics.LUMA
    exclusion = np.zeros(lum.shape)
    for side in (-.5, .5):
        roi = (np.abs(x-side) < .36) & (y > -.5) & (y < -.16) & (im[..., 3] > .9)
        if roi.sum() < 16:
            continue
        reference = float(np.percentile(lum[roi], 75))
        if reference < .003:
            continue
        score = np.clip((.4 - lum / reference) / .22, 0, 1) * roi
        exclusion = np.maximum(exclusion, score)
    from .landmarks import _gauss
    return np.clip(_gauss(exclusion, max(ipd * .003, .5)), 0, 1)


def sample_portrait_mask(mask, pixels):
    """Bilinear portrait evidence; points outside the image have no evidence."""
    p = np.asarray(pixels)
    h, w = mask.shape
    valid = (p[:, 0] >= 0) & (p[:, 0] <= w-1) & (p[:, 1] >= 0) & (p[:, 1] <= h-1)
    q = np.clip(p, [0, 0], [w-1, h-1])
    a = np.floor(q).astype(int)
    b = np.minimum(a+1, [w-1, h-1])
    u, v = (q-a).T
    value = ((1-u)*(1-v)*mask[a[:, 1], a[:, 0]] +
             u*(1-v)*mask[a[:, 1], b[:, 0]] +
             (1-u)*v*mask[b[:, 1], a[:, 0]] + u*v*mask[b[:, 1], b[:, 0]])
    return value * valid


def skin_mask(rgb, reference):
    """Soft log-colour gate retains shading while rejecting hair/cloth/white."""
    lr = np.log(np.maximum(rgb, .003))
    ref = np.log(np.maximum(reference, .003))
    chroma = lr - lr.mean(1, keepdims=True)
    rc = ref - ref.mean()
    distance = np.linalg.norm(chroma - rc, axis=1)
    lum = rgb @ np.array([.2126, .7152, .0722])
    ref_lum = float(reference @ np.array([.2126, .7152, .0722]))
    brightness = np.clip((lum / max(ref_lum, .005) - .12) / .35, 0, 1)
    return np.clip((.62 - distance) / .32, 0, 1) * brightness


def estimate_lighting(mesh, triangles, base, reference, eyes, features, cam):
    """Area-weighted visible cheek/forehead samples, avoiding lip/brow colours."""
    tri = triangles[::max(1, len(triangles) // 50000)]
    vertices = mesh["positions"][tri]
    pos = vertices.mean(1)
    n = optics.normalize(mesh["normals"][tri].mean(1))
    uv = mesh["uvs"][tri].mean(1)
    h, w = base.shape[:2]
    xy = np.clip(np.rint(uv * [w, h] - .5).astype(int), [0, 0], [w-1, h-1])
    rgb = optics.srgb_to_linear(base[xy[:, 1], xy[:, 0]].astype(float) / 255)
    regions = region_masks(cam.project(pos), eyes, features)
    support = np.maximum.reduce([regions["cheeks"], .8*regions["forehead"], .4*regions["chin"]])
    facing = np.clip((n * optics.normalize(cam.origin - pos)).sum(1), 0, 1)
    area = np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0], vertices[:,2]-vertices[:,0]), axis=1)
    confident_skin = np.clip((skin_mask(rgb, reference) - .6) / .4, 0, 1)
    weight = area * facing**2 * support * confident_skin * (support > .15)
    return relight.estimate(n, rgb, weight)


def bake(mesh, triangles, base_png, orm_png, portrait, eyes, poses, cam, params, out,
         roughness_factor=1., metallic_factor=1.):
    """Bake maps onto source charts, with tangent normals matching the GLB."""
    started = time.perf_counter()
    p = validate(params)
    base = np.asarray(Image.open(io.BytesIO(base_png)).convert("RGB")).copy()
    source_base = base.copy()
    h, w = base.shape[:2]
    orm = np.asarray(Image.open(io.BytesIO(orm_png)).convert("RGB").resize((w, h))).copy() if orm_png else np.full_like(base, [255, 255, 255])
    orm[..., 1] = np.round(orm[..., 1] * roughness_factor).astype(np.uint8)
    orm[..., 2] = np.round(orm[..., 2] * metallic_factor).astype(np.uint8)
    source_orm = orm.copy()
    normal = np.empty_like(base); normal[:] = [128, 128, 255]
    mask_img = np.zeros((h, w), np.uint8)
    covered = np.zeros((h, w), bool)
    tangents = compute_tangents(mesh["positions"], mesh["normals"], mesh["uvs"], triangles)
    reference = portrait_color(portrait, eyes)
    features = portrait_regions(portrait, eyes)
    brow_exclusion = portrait_brow_exclusion(portrait, eyes)
    lighting = (estimate_lighting(mesh, triangles, base, reference, eyes, features, cam)
                if p["delight_strength"] else {"status": "disabled"})
    k = float(np.mean([pose.units_per_m for pose in poses]))
    for y, x, tri, weights in atlas_samples(mesh["uvs"], triangles, (h, w)):
        if not len(y):
            continue
        def interp(attr):
            return (attr[tri] * weights[..., None]).sum(1)
        pos = interp(mesh["positions"])
        n = optics.normalize(interp(mesh["normals"]))
        tan = interp(tangents)
        t = optics.normalize(tan[:, :3] - (tan[:, :3] * n).sum(1)[:, None] * n)
        bit = np.cross(n, t) * np.where(tan[:, 3:] < 0, -1., 1.)
        rgb = optics.srgb_to_linear(source_base[y, x].astype(float) / 255)
        mask = skin_mask(rgb, reference)
        # Back-facing projected regions do not paint facial accents onto
        # the back of the skull; colour-selected skin still gets pores.
        facing = np.clip((n * optics.normalize(cam.origin - pos)).sum(1), 0, 1)
        pix = cam.project(pos)
        mask *= 1 - facing * sample_portrait_mask(brow_exclusion, pix)
        regions = region_masks(pix, eyes, features)
        # Back-facing skin does not inherit projected mouth/eye exclusions.
        detail_support = detail_mask(pix, eyes, regions)
        surface_detail = 1 - facing * (1 - detail_support)
        detail = mask * surface_detail
        regions = {r: v * facing for r, v in regions.items()}
        pore, grad = cellular(pos / k, .00085, p["seed"])
        perturbed = optics.normalize(n + grad * (.000022 * p["pore_strength"] * detail[:, None]))
        local = np.stack([(perturbed * t).sum(1), (perturbed * bit).sum(1), (perturbed * n).sum(1)], 1)
        normal[y, x] = np.round(np.clip(local * .5 + .5, 0, 1) * 255).astype(np.uint8)
        face_support = np.clip(sum(v for r, v in regions.items() if r not in ("scalp", "ears")) * 2, 0, 1)
        illumination_gain = relight.gain(n, lighting, p["delight_strength"]) ** face_support
        changed = rgb * illumination_gain[:, None]
        tone = tone_color(p["tone_u"], p["tone_v"])
        changed *= np.exp(np.clip(np.log((tone + .005) / (reference + .005)), -2, 2) * p["tone_strength"])
        for region, controls in p["regions"].items():
            amount = regions[region][:, None]
            changed *= np.exp(amount * controls["redness"] * np.array([.25, -.13, -.1]))
            gray = changed @ np.array([.2126, .7152, .0722])
            changed = gray[:, None] + (changed - gray[:, None]) * (1 + amount * controls["saturation"])
            changed *= np.exp(amount * controls["lightness"] * .55)
        f = p["freckles"]
        if f["density"] > 0 and f["strength"] > 0:
            spots, _ = cellular(pos / k, .0032, p["seed"] + 1009, f["density"])
            distribution = np.clip(regions["cheeks"] + .65 * regions["nose"] + .25 * regions["forehead"], 0, 1)
            if f["mask"] == "cheeks":
                distribution = regions["cheeks"]
            elif f["mask"] == "full_face":
                distribution = np.clip(sum(v for r, v in regions.items() if r not in ("scalp", "lips", "ears")), 0, 1)
            tint = np.array([.48, .26, .12])
            tint = tint.mean() + (tint - tint.mean()) * (2 * f["saturation"])
            tint = np.clip(tint + .15 * f["tone_shift"], .05, .9)
            distribution *= detail_support
            changed *= 1 - (spots * distribution * f["strength"])[:, None] * (1 - tint)
        result = rgb + (changed - rgb) * mask[:, None]
        base[y, x] = np.round(np.clip(optics.linear_to_srgb(np.maximum(result, 0)), 0, 1) * 255).astype(np.uint8)
        rough = source_orm[y, x, 1].astype(float) / 255
        target = np.clip((.56 + .12 * pore * surface_detail - .08 * regions["nose"]) * p["roughness"], .08, 1)
        orm[y, x, 1] = np.round((rough + (target - rough) * mask) * 255).astype(np.uint8)
        orm[y, x, 2] = np.round(source_orm[y, x, 2] * (1 - mask)).astype(np.uint8)
        mask_img[y, x] = np.round(mask * 255).astype(np.uint8)
        covered[y, x] = True
    out = Path(out)
    maps = {"skin_basecolor.png": base, "skin_normal.png": normal, "skin_orm.png": orm}
    for name, data in maps.items():
        Image.fromarray(texture.pad(data, covered, steps=8)).save(out / name)
    Image.fromarray(mask_img).save(out / "skin_mask.png")
    info = {"source": "portrait + procedural object-space detail", "params": p,
            "reference_linear": reference.tolist(), "resolution": [w, h],
            "facial_regions": features,
            "illumination": lighting,
            "brow_exclusion_pixels": int((brow_exclusion > .5).sum()),
            "covered_texels": int(covered.sum()), "skin_texels": int((mask_img > 127).sum()),
            "seconds": round(time.perf_counter() - started, 3), "files": list(FILES),
            "limitations": "Approximate colour/region masks. Optional illumination removal estimates broad directional light; cast shadows, highlights and albedo/lighting ambiguity remain."}
    (out / "skin.json").write_text(json.dumps(info, indent=1))
    return info, tangents
