"""Text -> photoreal head with analytic eyes: Qwen portrait, Pixal3D, eye fit.

1. Qwen-Image 2.1 draws a frontal, neutral head-and-neck portrait with open
   eyes on a transparent background (the template asks for exactly what the
   fit needs).
2. landmarks.py finds both eyes; a portrait without a clear pair is rejected
   before any Pixal3D time is spent.
3. Pixal3D (native CUDA) reconstructs the head from the portrait, single
   view, at a known FOV.
4. fit.py casts the eyes onto the head, scales and poses two analytic
   eyeballs, opens the lids and writes one GLB.

Everything lands in tmp/vhuman/heads/<id>/ (portrait.png, landmarks.png,
pixal3d.glb, head_eyes.glb, fit.json, head.json).
"""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .. import gpu
from ..eye import extract
from ..eye import params as P
from ..eye.geometry import Mesh
from ..eye.glb import GLBBuilder
from . import fit, iris_match, landmarks, skin

TEMPLATE = ("Photorealistic studio portrait photograph of {subject}: head and neck only, facing the camera "
            "straight-on, symmetric frontal view, neutral relaxed expression, mouth closed, both eyes wide open "
            "looking directly into the camera, even soft diffuse lighting, sharp focus, natural skin texture, "
            "no glasses, no jewellery.")
FILES = ("portrait.png", "portrait_raw.png", "landmarks.png", "pixal3d.glb", "head_eyes.glb", "fit.json",
         "head.json") + skin.FILES


def portrait_prompt(subject: str) -> str:
    subject = " ".join(str(subject).split())[:400]
    if not subject:
        raise ValueError("describe the person (subject)")
    return TEMPLATE.format(subject=subject)


# ---- mocks (no GPU): a synthetic face and an ellipsoid head -------------------

def mock_portrait(out: Path, seed: int = 0, size: int = 1024) -> Path:
    """A frontal 'face': a skin ellipse with two synthetic eyes pasted where
    a head-and-neck portrait has them, on a transparent background."""
    g = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy, ax, ay = size * 0.5, size * 0.45, size * 0.26, size * 0.36
    head = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 < 1.0
    rgba = np.zeros((size, size, 4), np.float32)
    skin = np.array([0.80, 0.62, 0.52]) * (1 + 0.03 * g.standard_normal(3))
    rgba[head, :3] = skin
    rgba[head, 3] = 1.0
    names = list(P.PRESETS)
    p = P.preset(names[seed % len(names)])
    patch = int(size * 0.12)
    eye_img, _, _ = extract.synthetic_eye(p, patch, seed, occlude=True, highlight=True, radius_frac=0.25,
                                          centred=True, noise_std=0.0)
    for ex in (cx - size * 0.085, cx + size * 0.085):
        x0, y0 = int(ex - patch / 2), int(size * 0.40 - patch / 2)
        rgba[y0:y0 + patch, x0:x0 + patch, :3] = eye_img
    img = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    img.save(out)
    return out


def mock_head_glb(out: Path, portrait: Path, fov_deg: float) -> Path:
    """An ellipsoid head in Pixal3D's GLB frame that projects onto the mock
    portrait's face, with a flat texture."""
    from .camera import PixalCamera
    cam = PixalCamera.from_portrait(portrait, math.radians(fov_deg))
    rgba = np.asarray(Image.open(portrait).convert("RGBA"))
    ys, xs = np.nonzero(rgba[..., 3] > 127)
    cx, cy = xs.mean(), ys.mean()
    rx, ry = (xs.max() - xs.min()) / 2, (ys.max() - ys.min()) / 2
    o, d = cam.rays(np.array([cx, cx + rx, cx]), np.array([cy, cy, cy + ry]))
    depth = cam.distance
    pts = o + d * (depth / d[:, 2:3])            # the plane z = 0
    centre = pts[0]
    ax, ay = abs(pts[1, 0] - centre[0]), abs(pts[2, 1] - centre[1])
    az = 0.8 * ax
    n_lat, n_lon = 256, 384          # triangles well below the eye opening size
    lat = np.linspace(0, math.pi, n_lat)
    lon = np.linspace(0, 2 * math.pi, n_lon, endpoint=False)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    unit = np.stack([np.sin(la) * np.cos(lo), np.cos(la), np.sin(la) * np.sin(lo)], -1).reshape(-1, 3)
    pos = centre + unit * np.array([ax, ay, az])
    nrm = unit / np.array([ax, ay, az]) ** 2
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    uv = np.stack([lo.reshape(-1) / (2 * math.pi), la.reshape(-1) / math.pi], -1)
    r, c = np.meshgrid(np.arange(n_lat - 1), np.arange(n_lon), indexing="ij")
    a = r * n_lon + c
    b = r * n_lon + (c + 1) % n_lon
    tri = np.concatenate([np.stack([a, b, a + n_lon], -1).reshape(-1, 3),        # outward winding
                          np.stack([b, b + n_lon, a + n_lon], -1).reshape(-1, 3)]).astype(np.uint32)
    bld = GLBBuilder("mock pixal3d head")
    tex = bld.texture(np.full((64, 64, 3), [200, 160, 140], np.uint8), "base")
    mr = bld.texture(np.full((64, 64, 3), [0, 200, 0], np.uint8), "mr")
    mat = bld.material({"name": "pixal3d", "pbrMetallicRoughness": {"baseColorTexture": {"index": tex},
                                                                     "metallicRoughnessTexture": {"index": mr}}})
    bld.node("mesh", mesh=bld.mesh(Mesh("head", pos.astype(np.float32), nrm.astype(np.float32),
                                        uv.astype(np.float32), tri), mat))
    bld.write(out)
    return out


# ---- the job --------------------------------------------------------------------

def head_job(service, request: dict, progress, cancel, python=None, mock: bool = False) -> dict:
    """{subject, seed, quality (preview|standard), fov, steps, portrait_attempts}."""
    from .. import baseline, qwen
    subject = request.get("subject", "a 30-year-old person with short dark hair")
    prompt = portrait_prompt(subject)
    seed = int(request.get("seed", 11))
    quality = request.get("quality", "standard")
    if quality not in ("preview", "standard", "high"):
        raise ValueError("quality must be preview, standard or high")
    fov = float(request.get("fov", 20.0))
    if not 8.0 <= fov <= 60.0:
        raise ValueError("fov must be in [8, 60] degrees")
    iris_source = request.get("iris_source", "portrait")
    if iris_source not in ("portrait", "library"):
        raise ValueError("iris_source must be portrait or library")
    preset = request.get("qwen_preset", "fast12")
    if preset not in ("fast12", "low8"):
        raise ValueError("qwen_preset must be fast12 or low8")
    skin_params = skin.validate(request.get("skin"))
    steps = int(request.get("steps", 24))
    attempts = int(request.get("portrait_attempts", 3))
    folder = service.work / "heads" / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    started = time.perf_counter()
    record = {"id": folder.name, "subject": subject, "prompt": prompt, "seed": seed, "quality": quality,
              "fov": fov, "created": time.time(), "license": qwen.LICENSE}
    lock = (service.work / "mock-gpu.lock") if mock else gpu.LOCK_PATH
    portrait = folder / "portrait.png"
    # 1-2. portrait until both eyes are found (the gate costs a second; Pixal3D minutes)
    eyes, tried = None, []
    plates = service.list_plates()
    iris_info = {"requested": iris_source, "source": "library"}
    with gpu.device_session(8192 if preset == "low8" else gpu.QWEN_MIN_FREE_MIB, cancel, lock_path=lock, check_memory=not mock):
        backend = qwen.make_backend(python, mock, preset=preset) if (not mock or iris_source == "portrait") else None
        try:
            for k in range(max(1, attempts)):
                s = seed + k
                progress(0.02 + 0.03 * k, f"portrait (seed {s})")
                if mock:
                    mock_portrait(portrait, s)
                else:
                    qwen._import_qimg21()
                    from qimg21_i23d import ops
                    ops.generate_object(prompt, portrait, backend, width=1024, height=1024, size=(1024, 1024),
                                        fill=0.85, transparent=True, steps=steps, seed=s)
                try:
                    eyes = landmarks.find_eyes(portrait)
                    record["seed_used"] = s
                    break
                except landmarks.LandmarkError as exc:
                    tried.append({"seed": s, "error": str(exc)})
            if eyes is not None and iris_source == "portrait":
                plates, iris_info = iris_match.generate(service, eyes, backend, head_id=folder.name,
                    seed=record["seed_used"], cancel=cancel,
                    progress=lambda f, m: progress(.08 + .03 * f, "iris: " + m))
        finally:
            if backend is not None:
                backend.close()
    record["portrait_attempts"] = tried
    if eyes is None:
        raise ValueError(f"no usable portrait in {attempts} attempts: {tried[-1]['error']}")
    # 3. Pixal3D
    progress(0.12, "waiting for the GPU (Pixal3D)")
    glb = folder / "pixal3d.glb"
    with gpu.device_session(gpu.PIXAL3D_MIN_FREE_MIB, cancel, lock_path=lock, check_memory=not mock):
        progress(0.15, f"Pixal3D head ({quality})")
        if mock:
            mock_head_glb(glb, portrait, fov)
            record["pixal3d"] = {"runner": "mock"}
        else:
            r = baseline.run_pixal3d(portrait, glb, folder / "work", quality, cancel)
            record["pixal3d"] = {k: r[k] for k in ("seconds", "mesh", "stats") if k in r}
    # 4. fit the eyes
    progress(0.85, "fitting the eyes")
    fitted = fit.fit_head(portrait, glb, folder, fov_deg=fov, plates=plates,
                          plate_loader=service.plate_iris, res=1024, iris_info=iris_info, skin_params=skin_params)
    record["fit"] = {k: fitted[k] for k in ("fit", "plate", "eyes", "seconds")}
    record["iris"] = fitted["iris"]
    record["skin"] = fitted["skin"]
    record["export"] = fitted["export"]
    record["seconds"] = round(time.perf_counter() - started, 2)
    (folder / "head.json").write_text(json.dumps(record, indent=1, default=float))
    return {"id": folder.name, "seconds": record["seconds"], "glb_url": f"/v1/heads/{folder.name}/head_eyes.glb",
            "portrait_url": f"/v1/heads/{folder.name}/portrait.png"}


def skin_job(service, request, progress, cancel):
    """New skin variant of an existing head, preserving its selected iris.

    CPU-only: refit the original reconstruction so successive colour edits
    never accumulate. The source head and its files remain untouched.
    """
    import shutil
    params = skin.validate(request.get("skin"))
    source_id = request.get("head_id")
    source_meta = service.head_file(source_id, "head.json")
    source = source_meta.parent
    original = json.loads(source_meta.read_text())
    selected = (original.get("fit") or {}).get("plate")
    plates = [p for p in service.list_plates() if p["id"] == selected] if selected else []
    if selected and not plates:
        raise ValueError("the source head's iris plate is no longer available")
    progress(.02, "preparing skin variant")
    if cancel.is_set():
        raise gpu.Cancelled("cancelled")
    started = time.perf_counter()
    folder = service.work / "heads" / uuid.uuid4().hex[:12]
    folder.mkdir(parents=True)
    for name in ("portrait.png", "pixal3d.glb"):
        shutil.copyfile(source / name, folder / name)
    progress(.1, "fitting and baking skin textures")
    fitted = fit.fit_head(folder / "portrait.png", folder / "pixal3d.glb", folder,
        fov_deg=original.get("fov", 20.), plates=plates, plate_loader=service.plate_iris,
        iris_info=original.get("iris"), skin_params=params)
    progress(.98, "saving skin variant")
    if cancel.is_set():
        raise gpu.Cancelled("cancelled")
    record = dict(original, id=folder.name, source_head=source_id, created=time.time(),
                  skin=fitted["skin"], iris=fitted["iris"], export=fitted["export"],
                  fit={k: fitted[k] for k in ("fit", "plate", "eyes", "seconds")},
                  seconds=round(time.perf_counter() - started, 2))
    (folder / "head.json").write_text(json.dumps(record, indent=1, default=float))
    return {"id": folder.name, "source_head": source_id, "seconds": record["seconds"],
            "glb_url": f"/v1/heads/{folder.name}/head_eyes.glb"}
