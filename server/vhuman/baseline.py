"""The Pixal3D baseline: what image-to-3D makes of an eyeball.

Two inputs, both single view:
- "analytic": our CPU render of the analytic eye (known geometry, so the
  reconstruction can be measured against the truth);
- "qwen": a Qwen-Image 2.1 text-to-object eyeball.

Pixal3D (the repository's native CUDA runner, `preview` quality) turns the
image into a textured GLB. The metrics: how spherical it is (sphere-fit
RMS / radius), whether it has a cornea bulge and how tall, the symmetric
Chamfer distance to the analytic shell once the bulge is aligned to +Z
(the shell is axially symmetric, so that fixes the pose up to scale), and
how many texels its texture spends on the iris. Pixal3D's materials have
no transmission, so the cornea cannot be transparent.
"""
from __future__ import annotations

import json
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from . import gpu
from .eye import geometry, optics, render
from .eye import params as P
from .eye.glb import GLB, GLBBuilder

ROOT = Path(__file__).resolve().parents[2]
QIMG21 = ROOT / "cuda" / "qimg21"
FOV_DEG = 20.0
QWEN_EYEBALL = ("A single realistic human eyeball, a glossy white sphere with fine red veins and a {color} iris "
                "with a black pupil facing the viewer, the cornea a clear dome over the iris, isolated, product "
                "photograph, studio lighting.")


def _import_qimg21():
    if str(QIMG21) not in sys.path:
        sys.path.insert(0, str(QIMG21))


def availability(mock=False) -> dict:
    if mock:
        return {"available": True, "runner": "mock"}
    try:
        _import_qimg21()
        from qimg21_i23d import reconstruct
        ok, missing = reconstruct.Pixal3DNative("cuda").available()
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}
    return {"available": ok, "runner": "native", **({} if ok else {"reason": "missing " + ", ".join(missing)})}


# ---- inputs -------------------------------------------------------------------------

def analytic_input(params: dict, out: Path, size: int = 1024, fill: float = 0.85) -> dict:
    """Our render, perspective at FOV 20 degrees, transparent background,
    the eye filling `fill` of the frame (Pixal3D's native preprocessing
    crops to the alpha anyway)."""
    prof = optics.ANATOMICAL
    extent = prof.sclera_radius / fill
    cam = render.Camera(fov_deg=FOV_DEG, distance=extent / math.tan(math.radians(FOV_DEG) / 2) + prof.apex_z)
    img = render.render(P.validate(params), size, cam, spp=4, transparent=True)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA").save(out)
    return {"camera": cam.__dict__, "size": size}


def qwen_input(backend, out: Path, color: str, seed: int, steps: int = 20) -> dict:
    _import_qimg21()
    from qimg21_i23d import ops
    prompt = QWEN_EYEBALL.format(color=color)
    info = ops.generate_object(prompt, out, backend, width=1024, height=1024, size=(1024, 1024), fill=0.85,
                               transparent=True, steps=steps, seed=seed)
    return {"prompt": prompt, "seconds": info.get("seconds"), "extraction": info.get("extraction")}


# ---- reconstruction -----------------------------------------------------------------

def fake_reconstruction(out: Path, seed: int = 0) -> dict:
    """--mock: the analytic shell, jittered and in Pixal3D's frame (unit
    size, axes (-x, y, -z)), standing in for a Pixal3D GLB."""
    mesh = geometry.shell(rings=48, segments=64)
    g = np.random.default_rng(seed)
    pos = mesh.positions / (2 * optics.ANATOMICAL.sclera_radius)
    pos = pos + g.normal(0, 0.004, pos.shape)
    mesh.positions = (pos * np.array([-1.0, 1.0, -1.0])).astype(np.float32)
    b = GLBBuilder("mock pixal3d")
    tex = b.texture(np.full((64, 64, 3), 200, np.uint8), "base")
    mat = b.material({"name": "pixal3d", "pbrMetallicRoughness": {"baseColorTexture": {"index": tex}}})
    b.node("mesh", mesh=b.mesh(mesh, mat))
    b.write(out)
    return {"runner": "mock", "seconds": 0.0}


def run_pixal3d(rgba: Path, out: Path, work: Path, quality: str = "preview", cancel=None) -> dict:
    _import_qimg21()
    from qimg21_i23d import reconstruct
    settings = reconstruct.ReconSettings.preset(quality)
    settings.cancel = cancel
    runner = reconstruct.Pixal3DNative("cuda", settings=settings)
    return runner.single(rgba, out, work, fov_rad=math.radians(FOV_DEG))


# ---- metrics ------------------------------------------------------------------------

def glb_triangles(path: Path):
    """All primitives of a GLB: positions (N, 3), triangles (M, 3), UVs and
    the base-colour texture size (for texel counts)."""
    g = GLB.load(path)
    pos, tri, uvs, tex_px = [], [], [], None
    base = 0
    for mesh in g.doc["meshes"]:
        for prim in mesh["primitives"]:
            p = g.accessor(prim["attributes"]["POSITION"]).astype(np.float64)
            t = g.accessor(prim["indices"]).reshape(-1, 3).astype(np.int64) + base
            uv = g.accessor(prim["attributes"]["TEXCOORD_0"]) if "TEXCOORD_0" in prim["attributes"] \
                else np.zeros((len(p), 2))
            pos.append(p); tri.append(t); uvs.append(uv)
            base += len(p)
            mat = g.doc["materials"][prim.get("material", 0)] if g.doc.get("materials") else {}
            texinfo = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
            if texinfo is not None and tex_px is None:
                img = g.image(g.doc["textures"][texinfo["index"]]["source"])
                tex_px = (img.shape[1], img.shape[0])
    return np.concatenate(pos), np.concatenate(tri), np.concatenate(uvs), tex_px


def sphere_fit(points: np.ndarray):
    a = np.c_[2 * points, np.ones(len(points))]
    b = (points ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(sol[3] + c @ c)
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(points - c, axis=1) - r) ** 2)))


def sample_surface(pos, tri, n, seed=0):
    g = np.random.default_rng(seed)
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pick = g.choice(len(tri), n, p=area / area.sum())
    u, v = g.random(n), g.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[pick] + u[:, None] * (b[pick] - a[pick]) + v[:, None] * (c[pick] - a[pick])


def nearest(src, dst, chunk=1024):
    out = np.empty(len(src))
    for i in range(0, len(src), chunk):
        d = ((src[i:i + chunk, None, :] - dst[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d.min(1))
    return out


def rotation_to_z(v: np.ndarray) -> np.ndarray:
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, z)
    s, c = np.linalg.norm(axis), float(v @ z)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * kx @ kx


def measure(glb_path: Path, params: dict | None = None, samples: int = 6000,
            front=(0.0, 0.0, -1.0)) -> dict:
    """Compare a reconstructed eyeball with the analytic eye (millimetres,
    after scaling the reconstruction's sphere fit to the anatomical sclera).

    `front` is the direction the input camera saw in the GLB's frame:
    Pixal3D writes (-x, y, -z), so the face the camera looked at is -Z. The
    radial profile along it shows whether a cornea dome (positive) or a pit
    (negative) was reconstructed."""
    prof = optics.ANATOMICAL
    pos, tri, uvs, tex_px = glb_triangles(glb_path)
    c, r, rms = sphere_fit(pos)
    scale = prof.sclera_radius / r
    q = (pos - c) * scale
    q = q @ rotation_to_z(np.asarray(front, np.float64)).T
    dist = np.linalg.norm(q, axis=1)
    ang = np.degrees(np.arccos(np.clip(q[:, 2] / np.maximum(dist, 1e-12), -1, 1)))
    bins = np.arange(0, 60, 5)
    profile = [round(float((dist[(ang >= lo) & (ang < lo + 5)].mean() - prof.sclera_radius) * 1000), 3)
               if ((ang >= lo) & (ang < lo + 5)).any() else None for lo in bins]
    analytic_profile = [round(float((optics.surface_radius(math.radians(lo + 2.5)) - prof.sclera_radius) * 1000), 3)
                        for lo in bins]
    apex_mm = float((dist[ang < 7.5].mean() - prof.sclera_radius) * 1000) if (ang < 7.5).any() else None
    analytic = geometry.shell(prof, rings=96, segments=128)
    a_pts = sample_surface(analytic.positions.astype(np.float64), analytic.indices.astype(np.int64), samples, 1)
    r_pts = sample_surface(q, tri, samples, 2)
    d_ra, d_ar = nearest(r_pts, a_pts), nearest(a_pts, r_pts)
    chamfer = math.sqrt(0.5 * (np.mean(d_ra ** 2) + np.mean(d_ar ** 2))) * 1000
    # Texels the reconstruction spends on the iris (triangles inside the limbus cone).
    cen = q[tri].mean(1)
    cang = np.arccos(np.clip(cen[:, 2] / np.maximum(np.linalg.norm(cen, axis=1), 1e-12), -1, 1))
    in_iris = cang < prof.alpha_limbus
    texels = None
    if tex_px is not None:
        a, b_, c_ = uvs[tri[:, 0]], uvs[tri[:, 1]], uvs[tri[:, 2]]
        uv_area = 0.5 * np.abs((b_[:, 0] - a[:, 0]) * (c_[:, 1] - a[:, 1]) - (c_[:, 0] - a[:, 0]) * (b_[:, 1] - a[:, 1]))
        texels = float(uv_area[in_iris].sum() * tex_px[0] * tex_px[1])
    return {
        "vertices": int(len(pos)), "triangles": int(len(tri)),
        "sphere_rms_over_radius": round(rms / r, 5),
        "sphere_rms_mm": round(rms * scale * 1000, 3),
        "apex_height_mm": None if apex_mm is None else round(apex_mm, 3),
        "analytic_apex_height_mm": round((prof.apex_z - prof.sclera_radius) * 1000, 3),
        "front_profile_mm": profile, "analytic_front_profile_mm": analytic_profile,
        "chamfer_rms_mm": round(chamfer, 4),
        "p95_distance_mm": round(float(np.percentile(np.concatenate([d_ra, d_ar]), 95) * 1000), 4),
        "iris_texels": None if texels is None else int(texels),
        "iris_texel_diameter": None if texels is None else round(2 * math.sqrt(texels / math.pi), 1),
        "texture": tex_px, "transparent_cornea": False,
    }


def baseline_job(service, request: dict, progress, cancel, mock: bool = False, python=None) -> dict:
    """{source: analytic|qwen, params, color, seed, quality}."""
    source = request.get("source", "analytic")
    if source not in ("analytic", "qwen"):
        raise ValueError("source must be analytic or qwen")
    quality = request.get("quality", "preview")
    params = P.validate(request.get("params") or {})
    seed = int(request.get("seed", 7))
    folder = service.work / "baselines" / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    started = time.perf_counter()
    record = {"id": folder.name, "source": source, "quality": quality, "params": params, "created": time.time()}
    rgba = folder / "input.png"
    lock = (service.work / "mock-gpu.lock") if mock else gpu.LOCK_PATH
    if source == "analytic":
        progress(0.02, "rendering the analytic eye")
        record["input"] = analytic_input(params, rgba)
    else:
        from . import qwen
        progress(0.02, "waiting for the GPU (Qwen)")
        with gpu.device_session(gpu.QWEN_MIN_FREE_MIB, cancel, lock_path=lock, check_memory=not mock):
            backend = qwen.make_backend(python, mock)
            try:
                progress(0.05, "Qwen-Image: text -> eyeball")
                if mock:
                    analytic_input(params, rgba)
                    record["input"] = {"mock": True}
                else:
                    record["input"] = qwen_input(backend, rgba, qwen.color_words(request.get("color", "hazel")), seed)
            finally:
                backend.close()
    progress(0.15, "waiting for the GPU (Pixal3D)")
    glb = folder / "pixal3d.glb"
    with gpu.device_session(gpu.PIXAL3D_MIN_FREE_MIB, cancel, lock_path=lock, check_memory=not mock):
        progress(0.2, "Pixal3D single view")
        record["pixal3d"] = fake_reconstruction(glb) if mock else run_pixal3d(rgba, glb, folder / "work", quality,
                                                                                cancel)
    progress(0.9, "measuring")
    record["metrics"] = measure(glb, params)
    record["seconds"] = round(time.perf_counter() - started, 2)
    (folder / "baseline.json").write_text(json.dumps(record, indent=1, default=str))
    return {"id": folder.name, "metrics": record["metrics"], "seconds": record["seconds"],
            "glb_url": f"/v1/baselines/{folder.name}/pixal3d.glb", "input_url": f"/v1/baselines/{folder.name}/input.png"}


if __name__ == "__main__":
    # python3 -m server.vhuman.baseline tmp/vhuman/baselines/<id>   (re-measure and update baseline.json)
    folder = Path(sys.argv[1])
    record = json.loads((folder / "baseline.json").read_text())
    record["metrics"] = measure(folder / "pixal3d.glb")
    (folder / "baseline.json").write_text(json.dumps(record, indent=1, default=str))
    print(json.dumps(record["metrics"], indent=1))
