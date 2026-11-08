"""Engine-neutral texture sets, portable glTF eyes and live shader exports."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import chart, geometry, iris, noise, optics, sclera
from . import params as P
from .glb import GLBBuilder

LIVE_TEXTURES = ("iris_masks", "iris_normal", "sclera_masks", "sclera_normal", "iris_color_chart")
IPD = 0.063


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def normal_u8(n: np.ndarray, *, directx: bool = False) -> np.ndarray:
    n = n.copy()
    if directx:
        n[..., 1] = -n[..., 1]
    return to_u8(n * 0.5 + 0.5)


def srgb_u8(linear: np.ndarray) -> np.ndarray:
    return to_u8(optics.linear_to_srgb(linear))


def uv_grid(res: int):
    """Texel-centre UVs (u right, v down) of a square texture."""
    c = (np.arange(res, dtype=np.float64) + 0.5) / res
    return np.stack(np.meshgrid(c, c), -1)


def structures(p: dict, res: int, iris_tex=None):
    """Iris and sclera structure, built concurrently (numpy FFTs and PIL
    drawing release the GIL); `iris_tex` overrides the procedural iris
    (an iris plate)."""
    if iris_tex is not None:
        return iris_tex, sclera.build(p, res)
    with ThreadPoolExecutor(2) as pool:
        it = pool.submit(iris.build, p, res)
        st = pool.submit(sclera.build, p, res)
        return it.result(), st.result()


def live_textures(p: dict, res: int = 1024) -> tuple[dict, dict]:
    """The web shader's textures (uint8 arrays) and timings."""
    started = time.perf_counter()
    it, st = structures(p, res)
    out = {"iris_masks": to_u8(it.masks), "iris_normal": normal_u8(it.normal),
           "sclera_masks": to_u8(st.masks), "sclera_normal": normal_u8(st.normal),
           "iris_color_chart": chart.chart_image(256)}
    return out, {"iris_s": round(it.seconds, 4), "sclera_s": round(st.seconds, 4),
                 "total_s": round(time.perf_counter() - started, 4)}


def sclera_color_uv(p: dict, st, uv: np.ndarray) -> np.ndarray:
    """Sclera colour at eyeball UVs (rotation applied)."""
    ruv = sclera.rotate_uv(uv, p["sclera"]["rotation"])
    m = sclera.sample(st.masks, ruv)
    r_uv = np.hypot(uv[..., 0] - 0.5, uv[..., 1] - 0.5)
    return sclera.colorize(m, r_uv, p)


def eye_basecolor(p: dict, it, st, res: int) -> np.ndarray:
    """Flat (unrefracted) eyeball colour in the angular UV layout. For texture and
    as a preview."""
    uv = uv_grid(res)
    r_uv = np.hypot(uv[..., 0] - 0.5, uv[..., 1] - 0.5)
    phi = np.arctan2(0.5 - uv[..., 1], uv[..., 0] - 0.5)
    col = iris.iris_plane_color(it, r_uv / p["cornea"]["size"], phi, p, sclera.sampler(st, p))
    return col.astype(np.float32)


def eye_tissue_factor(p: dict, res: int) -> np.ndarray:
    """Linear RGB multiplier for the exposed, mottled peripheral sclera.

    Stored at half scale so values above one survive in an ordinary linear PNG.
    The iris and inner limbus remain unchanged; intensity controls the additional
    red peripheral tint, while a seeded field supplies low-amplitude tissue variation.
    """
    p = P.validate(p)
    uv = uv_grid(res)
    r = np.hypot(uv[..., 0] - .5, uv[..., 1] - .5)
    tissue = noise.smoothstep(.22, .33, r).astype(np.float64)
    vascularity = p['sclera']['vascularity_intensity']
    field = noise.spectral_smooth((res, res), p['structure']['seed'], beta=2.8, kmin=4, kmax=24, salt=87)
    mottling = 1. + .11 * tissue * np.clip(field, -2., 2.)
    factors = np.stack([1. - .20 * tissue,
                        1. - (.44 + .16 * vascularity) * tissue,
                        1. - (.48 + .16 * vascularity) * tissue], -1)
    return np.clip(factors * mottling[..., None] * .5, 0., 1.).astype(np.float32)


def eye_sclera_microfactor(p: dict, res: int, strength: float) -> np.ndarray:
    """Seeded fine sclera variation as a half-scale linear RGB multiplier."""
    p = P.validate(p)
    if not 0. <= strength <= .3:
        raise ValueError('sclera microtexture strength must be in [0, .3]')
    uv = uv_grid(res); r = np.hypot(uv[..., 0] - .5, uv[..., 1] - .5)
    tissue = noise.smoothstep(p['cornea']['size'] - .01, p['cornea']['size'] + .035, r)
    field = noise.spectral_smooth((res, res), p['structure']['seed'], beta=2.5, kmin=8, kmax=64, salt=145)
    factor = np.clip((1. + strength * tissue * np.clip(field, -2., 2.)) * .5, 0., 1.)
    return np.repeat(factor[..., None], 3, axis=-1).astype(np.float32)


def eye_sclera_surface_microfactor(p: dict, res: int, strength: float, *, fine: bool = False) -> np.ndarray:
    """Isotropic, seeded variation on the spherical surface, encoded at half scale.

    The angular atlas stretches UV-space noise near the limbus. Sampling a
    band-limited 3D field at spherical directions avoids that radial stretching.
    """
    p = P.validate(p)
    if not 0. <= strength <= .3:
        raise ValueError('sclera surface microtexture strength must be in [0, .3]')
    uv = uv_grid(res); x = uv[..., 0] - .5; y = .5 - uv[..., 1]
    r = np.hypot(x, y); angle = np.minimum(optics.alpha_from_uv(r, optics.profile_from_params(p)), np.pi)
    radial = np.sin(angle) / np.maximum(r, 1e-12)
    sx = (x * radial).astype(np.float32); sy = (y * radial).astype(np.float32); sz = np.cos(angle).astype(np.float32)
    g = noise.rng(p['structure']['seed'], 0x5C3)
    directions = g.normal(size=(96 if fine else 48, 3)); directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    frequencies = g.uniform(70., 180., len(directions)) if fine else g.uniform(35., 85., len(directions))
    phases = g.uniform(0., 2 * np.pi, len(directions))
    field = np.zeros((res, res), np.float32)
    for d, frequency, phase in zip(directions, frequencies, phases):
        field += np.cos(frequency * (d[0] * sx + d[1] * sy + d[2] * sz) + phase).astype(np.float32)
    band = (r > .22) & (r < .4)
    field = (field - field[band].mean()) / max(float(field[band].std()), 1e-6)
    tissue = noise.smoothstep(p['cornea']['size'] - .01, p['cornea']['size'] + .035, r)
    factor = np.clip((1. + strength * tissue * np.clip(field, -2., 2.)) * .5, 0., 1.)
    return np.repeat(factor[..., None], 3, axis=-1).astype(np.float32)


def reduce_iris_angular_contrast(color: np.ndarray, p: dict, strength: float) -> np.ndarray:
    """Blend iris detail toward its radial mean, retaining its radial colour profile.

    This is an opt-in calibration for procedural structural patterns that have
    more angular contrast than a reference iris. The sclera is left unchanged.
    """
    p = P.validate(p)
    color = np.asarray(color, np.float32)
    if color.ndim != 3 or color.shape[0] != color.shape[1] or color.shape[2] != 3:
        raise ValueError('expected a square RGB eye texture')
    if not 0. <= strength <= 1.:
        raise ValueError('iris angular contrast strength must be in [0, 1]')
    res = color.shape[0]
    uv = uv_grid(res)
    r = np.hypot(uv[..., 0] - .5, uv[..., 1] - .5)
    bins = np.floor(r * res * 2).astype(np.int32)
    counts = np.bincount(bins.ravel())
    means = np.stack([np.bincount(bins.ravel(), weights=color[..., k].ravel()) / np.maximum(counts, 1)
                      for k in range(3)], -1)
    iris = 1. - noise.smoothstep(p['cornea']['size'] - .02, p['cornea']['size'], r)
    weight = strength * iris[..., None]
    return (color * (1. - weight) + means[bins] * weight).astype(np.float32)


def calibrate_iris_color(color: np.ndarray, p: dict, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the baked inner iris median to a measured linear RGB target.

    Gains are bounded to avoid amplifying noise or near-black channels. The
    calibration fades across the limbus and leaves the sclera unchanged.
    """
    p = P.validate(p)
    color = np.asarray(color, np.float32)
    target = np.asarray(target, np.float64)
    if color.ndim != 3 or color.shape[0] != color.shape[1] or color.shape[2] != 3:
        raise ValueError('expected a square RGB eye texture')
    if target.shape != (3,) or not np.isfinite(target).all() or (target < 0.).any():
        raise ValueError('iris calibration target must be finite nonnegative RGB')
    uv = uv_grid(color.shape[0]); r = np.hypot(uv[..., 0] - .5, uv[..., 1] - .5)
    size = p['cornea']['size']
    region = (r >= .45 * size) & (r < .65 * size)
    source = np.median(color[region], axis=0)
    # Treat very dark channels as uninformative; division by them amplifies quantisation.
    gains = np.where((source >= .002) & (target >= .002),
                     np.clip(target / np.maximum(source, 1e-6), .5, 3.), 1.)
    weight = (1. - noise.smoothstep(size - .5 * p['cornea']['limbus_softness'],
                                    size + .5 * p['cornea']['limbus_softness'], r))[..., None]
    corrected = color * (1. + weight * (gains - 1.))
    return corrected.astype(np.float32), gains.astype(np.float32)


def calibrate_iris_outer(color: np.ndarray, p: dict, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match an outer-iris median after inner calibration, fading at the limbus."""
    p = P.validate(p); color = np.asarray(color, np.float32); target = np.asarray(target, np.float64)
    if color.ndim != 3 or color.shape[0] != color.shape[1] or color.shape[2] != 3:
        raise ValueError('expected a square RGB eye texture')
    if target.shape != (3,) or not np.isfinite(target).all() or (target < 0.).any():
        raise ValueError('outer iris calibration target must be finite nonnegative RGB')
    uv = uv_grid(color.shape[0]); r = np.hypot(uv[..., 0] - .5, uv[..., 1] - .5) / p['cornea']['size']
    source = np.median(color[(r >= .65) & (r < .85)], axis=0)
    gains = np.where((source >= .002) & (target >= .002),
                     np.clip(target / np.maximum(source, 1e-6), .5, 3.), 1.)
    weight = (noise.smoothstep(.55, .70, r) * (1. - noise.smoothstep(.92, 1.08, r)))[..., None]
    return (color * (1. + weight * (gains - 1.))).astype(np.float32), gains.astype(np.float32)


def eye_basecolor_layered_calibrated(p: dict, it, st, res: int, inner_target, outer_target, *, smoothing: float = .6,
                                     radial_targets=None, fiber_chroma: float = 0.):
    """Calibrate iris colors before mixing them with sclera at the limbus.

    The two targets are linear RGB statistics, not reference pixels. Separating
    the contributions keeps the correction from brightening the sclera itself.
    """
    p = P.validate(p)
    inner_target = np.asarray(inner_target, np.float64); outer_target = np.asarray(outer_target, np.float64)
    if any(t.shape != (3,) or not np.isfinite(t).all() or (t < 0.).any() for t in (inner_target, outer_target)):
        raise ValueError('iris calibration targets must be finite nonnegative RGB')
    uv = uv_grid(res); x = uv[..., 0] - .5; y = .5 - uv[..., 1]
    radius = np.hypot(x, y); t = radius / p['cornea']['size']; phi = np.arctan2(y, x)
    iris_color = iris.iris_base(it, t, phi, p).astype(np.float64)
    sclera_color = sclera.sampler(st, p)(radius, phi).astype(np.float64)
    weight = iris.cornea_mask(radius, p)[..., None].astype(np.float64)
    composite = iris_color * weight + sclera_color * (1. - weight)
    # Use the same angular smoothing as the existing diagnostic, on the iris
    # layer before compositing; the peripheral sclera remains untouched.
    iris_color = reduce_iris_angular_contrast(iris_color.astype(np.float32), p, smoothing).astype(np.float64)
    inner = (t >= .45) & (t < .65)
    src_inner = np.median(composite[inner], axis=0)
    inner_gain = np.where((src_inner >= .002) & (inner_target >= .002),
                          np.clip(inner_target / np.maximum(src_inner, 1e-6), .5, 3.), 1.)
    iris_color *= inner_gain
    outer = (t >= .65) & (t < .85)
    source_iris = np.median((iris_color * weight)[outer], axis=0)
    source_sclera = np.median((sclera_color * (1. - weight))[outer], axis=0)
    required = np.maximum(outer_target - source_sclera, 0.)
    outer_gain = np.where((source_iris >= .002) & (outer_target >= .002),
                          np.clip(required / np.maximum(source_iris, 1e-6), .5, 2.), 1.)
    radial = noise.smoothstep(.55, .75, t)[..., None]
    iris_color *= 1. + radial * (outer_gain - 1.)
    if radial_targets is not None:
        targets = np.asarray(radial_targets, np.float64)
        if targets.shape != (3, 3) or not np.isfinite(targets).all() or (targets < 0.).any():
            raise ValueError('radial iris targets must contain three finite nonnegative RGB values')
        centers = (.475, .625, .775)
        gains = []
        for (lo, hi), target in zip(((.40, .55), (.55, .70), (.70, .85)), targets):
            band = (t >= lo) & (t < hi)
            source_iris = np.median((iris_color * weight)[band], axis=0)
            source_sclera = np.median((sclera_color * (1. - weight))[band], axis=0)
            needed = np.maximum(target - source_sclera, 0.)
            gains.append(np.where((source_iris >= .002) & (target >= .002),
                                  np.clip(needed / np.maximum(source_iris, 1e-6), .5, 2.), 1.))
        gains = np.asarray(gains)
        profile = np.stack([np.interp(t, centers, gains[:, k]) for k in range(3)], axis=-1)
        iris_color *= profile
    if not 0. <= fiber_chroma <= 1.:
        raise ValueError('fiber chroma strength must be between 0 and 1')
    if fiber_chroma:
        # Color the existing procedural fibers using only the difference between
        # measured inner and outer chromaticities; retain their own random layout.
        masks = iris.sample_masks(it.masks, np.minimum(optics.pupil_scale(t, P.pupil_scale(p)), 1.),
                                  phi - 2. * np.pi * p['iris']['rotation'])
        detail = masks[..., 0].astype(np.float64)
        bins = np.clip((t * 128).astype(np.int32), 0, 255)
        counts = np.bincount(bins.ravel(), minlength=256)
        means = np.bincount(bins.ravel(), weights=detail.ravel(), minlength=256) / np.maximum(counts, 1)
        residual = detail - means[bins]
        region = (t >= .4) & (t < .85)
        residual /= max(float(residual[region].std()), .05)
        residual = np.clip(residual, -2., 2.)
        inner_chroma = inner_target / max(float(inner_target @ iris.REC709), .01)
        outer_chroma = outer_target / max(float(outer_target @ iris.REC709), .01)
        direction = np.clip(inner_chroma - outer_chroma, -1.5, 1.5)
        fade = noise.smoothstep(.35, .48, t) * (1. - noise.smoothstep(.78, .95, t))
        iris_color *= np.clip(1. + fiber_chroma * fade[..., None] * residual[..., None] * direction, .4, 1.8)
    result = iris_color * weight + sclera_color * (1. - weight)
    return result.astype(np.float32), inner_gain.astype(np.float32), outer_gain.astype(np.float32)


def shell_textures(p: dict, st, res: int) -> dict:
    """GLB shell: sclera base colour (white over the cornea, where the
    transmission mask is 1), roughness and the normal map."""
    uv = uv_grid(res)
    r_uv = np.hypot(uv[..., 0] - 0.5, uv[..., 1] - 0.5)
    cornea = 1.0 - np.clip((r_uv - (optics.CORNEA_SIZE_REF - 0.004)) / 0.008, 0.0, 1.0)
    scl = sclera_color_uv(p, st, uv)
    base = scl * (1 - cornea[..., None]) + cornea[..., None]
    o = p["optics"]
    rough = o["cornea_roughness"] * cornea + o["sclera_roughness"] * (1 - cornea)
    orm = np.stack([np.ones_like(rough), rough, np.zeros_like(rough)], -1)
    n = sclera.sample(np.concatenate([st.normal, np.zeros(st.normal.shape[:2] + (1,), np.float32)], -1),
                      sclera.rotate_uv(uv, p["sclera"]["rotation"]))[..., :3]
    n = n * (1 - cornea[..., None]) + np.array([0.0, 0.0, 1.0]) * cornea[..., None]
    n = optics.normalize(n)
    trans = np.repeat(cornea[..., None], 3, -1)
    return {"base": srgb_u8(base), "orm": to_u8(orm), "normal": normal_u8(n), "transmission": to_u8(trans)}


def sclera_vessel_texture(st, res: int) -> np.ndarray:
    """Uncoloured vessel layers anchored to the exposed horizontal eye axis.

    The authored sclera rotation still controls its baked albedo. This independent overlay keeps its
    horizontal vessel bias visible on both sides of the iris even when the authored atlas is rotated.
    """
    uv = uv_grid(res)
    return to_u8(sclera.sample(st.masks, uv)[..., :3])


def add_eye_meshes(b: GLBBuilder, p: dict, res: int = 2048, *, profile=None, iris_tex=None,
                   name: str = "eye", volume: bool = True, runtime: bool = False) -> tuple[int, int]:
    """Add one eye's materials and meshes (metres, +Z gaze) to a builder:
    a transparent-cornea shell over an opaque iris. Returns the mesh indices.
    volume=False leaves the cornea thin-walled: real-time viewers (three.js)
    then skip their screen-space refraction offset, which in a head shows
    displaced copies of the eyelids through the cornea. runtime=True uses the
    low-polygon shell and iris (about 1.3k triangles, geometry.runtime_*)
    that the analytic mesh eye material shades; the dense meshes stay for
    DCC export."""
    p = P.validate(p)
    profile = profile or optics.profile_from_params(p)
    it, st = structures(p, res, iris_tex)
    sh = shell_textures(p, st, res)
    iris_rgb = iris.bake_color(it, p, res, sclera.sampler(st, p))
    iris_orm = np.stack([0.4 + 0.6 * it.masks[..., 1], np.full(it.masks.shape[:2], 0.6),
                         np.zeros(it.masks.shape[:2])], -1)          # occlusion from the shadow mask
    shell_mat = b.material({
        "name": f"{name}_shell",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": b.texture(sh["base"], f"{name}_shell_basecolor")},
                                 "metallicRoughnessTexture": {"index": b.texture(sh["orm"], f"{name}_shell_orm")},
                                 "metallicFactor": 1.0, "roughnessFactor": 1.0},
        "normalTexture": {"index": b.texture(sh["normal"], f"{name}_shell_normal")},
        "extensions": {
            "KHR_materials_transmission": {"transmissionFactor": 1.0,
                                           "transmissionTexture": {"index": b.texture(sh["transmission"],
                                                                                      f"{name}_shell_transmission")}},
            "KHR_materials_ior": {"ior": p["optics"]["ior"]},
            **({"KHR_materials_volume": {"thicknessFactor": p["optics"]["chamber_depth"] * 0.7}} if volume else {}),
        }})
    iris_base = b.texture(srgb_u8(iris_rgb), f"{name}_iris_basecolor")
    iris_orm_tex = b.texture(to_u8(iris_orm), f"{name}_iris_orm")
    iris_mat = b.material({
        "name": f"{name}_iris",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": iris_base},
                                 "metallicRoughnessTexture": {"index": iris_orm_tex},
                                 "metallicFactor": 1.0, "roughnessFactor": 1.0},
        "normalTexture": {"index": b.texture(normal_u8(it.normal), f"{name}_iris_normal"), "scale": 0.6},
        "occlusionTexture": {"index": iris_orm_tex}})
    if runtime:
        shell_mesh = b.mesh(geometry.runtime_shell(profile), shell_mat)
        iris_mesh = b.mesh(geometry.runtime_iris(p, profile), iris_mat)
    else:
        shell_mesh = b.mesh(geometry.shell(profile), shell_mat)
        iris_mesh = b.mesh(geometry.iris_disk(p, profile), iris_mat)
    return shell_mesh, iris_mesh


def export_glb(p: dict, path, res: int = 2048, *, pair: bool = False, profile=None,
               iris_tex=None) -> dict:
    """The portable eye: a transparent-cornea shell over an opaque iris."""
    started = time.perf_counter()
    p = P.validate(p)
    b = GLBBuilder()
    shell_mesh, iris_mesh = add_eye_meshes(b, p, res, profile=profile, iris_tex=iris_tex)
    info = {"params": p, "ipd": IPD if pair else None}
    if pair:
        for side, x in (("left", IPD / 2), ("right", -IPD / 2)):
            kids = [b.node(f"eye_{side}_shell", mesh=shell_mesh, root=False),
                    b.node(f"eye_{side}_iris", mesh=iris_mesh, root=False)]
            b.node(f"eye_{side}", translation=(x, 0, 0), children=kids)
    else:
        kids = [b.node("eye_shell", mesh=shell_mesh, root=False), b.node("eye_iris", mesh=iris_mesh, root=False)]
        b.node("eye", children=kids, extras={"units": "metres", "gaze": "+Z"})
    size = b.write(path)
    info.update(bytes=size, seconds=round(time.perf_counter() - started, 3), res=res)
    return info


def export_textures(p: dict, out_dir, res: int = 2048, iris_tex=None) -> dict:
    """Engine-neutral textures and the public procedural parameter JSON."""
    from PIL import Image
    p = P.validate(p)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    it, st = structures(p, res, iris_tex)
    uv = uv_grid(res)
    files = {
        "iris_masks.png": to_u8(it.masks),
        "iris_normal.png": normal_u8(it.normal),
        "sclera_basecolor.png": srgb_u8(sclera_color_uv(p, st, uv)),
        "eye_basecolor.png": srgb_u8(eye_basecolor(p, it, st, res)),
        "iris_color_chart.png": chart.chart_image(256),
    }
    for name, arr in files.items():
        Image.fromarray(arr).save(out / name, compress_level=9)
    (out / "params.json").write_text(json.dumps(p, indent=1))
    (out / "measurements.json").write_text(json.dumps({
        "units": "metres", "source": "user parameters with synthetic defaults",
        "profile": {k: p["optics"][k] for k in ("sclera_radius", "limbus_radius", "cornea_radius", "limbus_blend")},
        "uv_mapping": "angular-two-segment-v1", "normal_convention": "+Y / OpenGL"
    }, indent=1))
    return {"dir": str(out), "files": sorted(list(files) + ["params.json", "measurements.json"])}



def write_live(p: dict, out_dir, res: int = 1024) -> dict:
    """The live shader set as PNGs plus uniforms.json."""
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    textures, timings = live_textures(p, res)
    started = time.perf_counter()
    for name, arr in textures.items():
        Image.fromarray(arr).save(out / f"{name}.png", compress_level=3)
    (out / "uniforms.json").write_text(json.dumps(optics.uniforms(p)))
    timings["png_s"] = round(time.perf_counter() - started, 4)
    return {"dir": str(out), "files": [f"{n}.png" for n in textures] + ["uniforms.json"], "timings": timings}
