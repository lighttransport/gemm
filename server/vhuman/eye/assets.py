"""Engine-neutral texture sets, portable glTF eyes and live shader exports."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import chart, geometry, iris, optics, sclera
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


def add_eye_meshes(b: GLBBuilder, p: dict, res: int = 2048, *, profile=None, iris_tex=None,
                   name: str = "eye", volume: bool = True) -> tuple[int, int]:
    """Add one eye's materials and meshes (metres, +Z gaze) to a builder:
    a transparent-cornea shell over an opaque iris. Returns the mesh indices.
    volume=False leaves the cornea thin-walled: real-time viewers (three.js)
    then skip their screen-space refraction offset, which in a head shows
    displaced copies of the eyelids through the cornea."""
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
