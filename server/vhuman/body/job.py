"""Qwen full-body image -> SAM 3D Body -> Pixal3D -> combined avatar job."""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import baseline, gpu, qwen
from ..service import ROOT

MODEL_DIR = Path("/mnt/disk1/models/sam3d-body")
SAM3_MODEL = Path("/mnt/disk1/models/sam3/sam3.model.safetensors")
CLIP_BPE = Path("/mnt/disk1/models/clip-bpe")
DEFAULT_RIG_PYTHON = ROOT / "tmp/vhuman-rig-venv/bin/python"
FILES = ("body_image.png", "body_image_raw.png", "body_mhr.glb", "body_mhr.glb.json",
         "pixal3d_full.glb", "body_basecolor.png", "avatar.glb", "avatar.usda", "avatar_usd.zip",
         "avatar.json", "body_report.json", "garments.json", "deformer.lrm",
         "wm_smile.png", "wm_brow_up.png", "wm_brow_down.png", "wm_mouth.png")
PROMPT = ("Full-length photorealistic studio photograph of the same person's facial identity as the reference "
          "portrait. Use the portrait for face identity only, not for its crop or clothing. Dress the person in "
          "{outfit}; every named garment is worn and clearly visible. Head to toe in frame, standing upright "
          "in a relaxed symmetrical A-pose, arms angled outward and "
          "separated from the torso, straight legs slightly apart, both hands and shoes fully visible. "
          "Front view, level camera, plain transparent background, even diffuse lighting, sharp details. "
          "One person only, no props, no crop, no text.")
BODY_FAST12_MIN_FREE_MIB = 15000


def _qwen_preset(requested: str, status: dict | None) -> str:
    if requested not in ("auto", "fast12", "low8"):
        raise ValueError("qwen_preset must be auto, fast12 or low8")
    if requested != "auto":
        return requested
    return "fast12" if status and status["free_mib"] >= BODY_FAST12_MIN_FREE_MIB else "low8"


def availability(model_dir=MODEL_DIR, rig_python=None, mock=False) -> dict:
    model_dir = Path(model_dir)
    required = [model_dir / "safetensors/sam3d_body_mhr_jit.safetensors",
                model_dir / "safetensors/sam3d_body_dinov3.safetensors",
                model_dir / "dinov3/assets/mhr_model.pt",
                model_dir / "safetensors/sam3d_body_mhr_head.safetensors",
                Path(rig_python) if rig_python else DEFAULT_RIG_PYTHON]
    missing = [str(p) for p in required if not p.is_file()]
    has_gpu = gpu.gpu_status() is not None
    return {"available": not missing and (mock or has_gpu), "model_dir": str(model_dir),
            "missing": missing, "gpu": has_gpu, "mock": mock,
            "reason": "Qwen-Image 2.1 and Pixal3D require a CUDA or ROCm GPU" if not mock and not has_gpu else None}


def _run(cmd: list[str], cancel, *, cwd=ROOT, timeout=2400) -> str:
    from ..runtime import python_command
    proc = subprocess.Popen(python_command(cmd), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    lines = []
    stop = threading.Event()
    def watch():
        deadline = time.monotonic() + timeout
        while not stop.wait(.25):
            if proc.poll() is not None:
                return
            if cancel.is_set():
                proc.terminate()
                return
            if time.monotonic() >= deadline:
                proc.kill()
                return
    threading.Thread(target=watch, daemon=True).start()
    try:
        for line in proc.stdout:
            lines.append(line.rstrip())
            lines = lines[-40:]
        code = proc.wait()
    finally:
        stop.set()
    if cancel.is_set():
        raise gpu.Cancelled("cancelled")
    if code:
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({code}): " + " | ".join(lines[-8:]))
    return "\n".join(lines)


def _binary(kind: str, use_cuda: bool, cancel) -> Path:
    selected = gpu.backend() if use_cuda else "cpu"
    directory = ROOT / {"cuda": "cuda", "rocm": "rdna4", "cpu": "cpu"}[selected] / kind
    name = {"cuda": "test_cuda_", "rocm": "test_hip_", "cpu": "test_"}[selected] + kind
    path = directory / name
    if not path.is_file():
        cmd = ["make", "-C", str(directory), name]
        if selected == "cpu":
            cmd.append("ARCH=native")
        _run(cmd, cancel, timeout=600)
    return path


def _mock_image(out: Path, seed: int):
    """Body silhouette for the GPU-free end-to-end export check."""
    im = Image.new("RGBA", (768, 1280), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    skin = (185 + seed % 15, 143, 119, 255)
    cloth = (60, 104, 144, 255)
    d.ellipse((330, 75, 438, 205), fill=skin)
    d.polygon([(351, 193), (417, 193), (447, 465), (321, 465)], fill=cloth)
    d.polygon([(333, 216), (288, 249), (150, 409), (170, 431), (342, 323)], fill=cloth)
    d.polygon([(435, 216), (480, 249), (618, 409), (598, 431), (426, 323)], fill=cloth)
    d.ellipse((140, 401, 176, 450), fill=skin)
    d.ellipse((592, 401, 628, 450), fill=skin)
    d.polygon([(329, 463), (380, 463), (368, 979), (324, 979)], fill=(50, 60, 70, 255))
    d.polygon([(388, 463), (439, 463), (444, 979), (400, 979)], fill=(50, 60, 70, 255))
    d.rectangle((306, 970, 368, 1022), fill=(35, 35, 38, 255))
    d.rectangle((400, 970, 462, 1022), fill=(35, 35, 38, 255))
    im.save(out)


def _image_gate(path: Path) -> tuple[int, int, int, int]:
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    alpha = rgba[..., 3] > 127
    ys, xs = np.nonzero(alpha)
    if len(xs) < rgba.shape[0] * rgba.shape[1] * .025:
        raise ValueError("body foreground is too small")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    h, w = rgba.shape[:2]
    if x1 - x0 < .48 * w or y1 - y0 < .67 * h:
        raise ValueError("full body, separated arms or feet are not visible")
    if min(x0, y0, w - x1, h - y1) < 5:
        raise ValueError("the generated body is cropped by the image edge")
    return x0, y0, x1, y1


def _generate_image(head: Path, out: Path, outfit: str, seed: int, attempts: int,
                    steps: int, python, preset: str, mock: bool, progress, cancel) -> dict:
    image = out / "body_image.png"
    if mock:
        _mock_image(image, seed)
        return {"prompt": PROMPT.format(outfit=outfit), "seed": seed, "bbox": _image_gate(image),
                "backend": "mock", "preset": preset}
    qwen._import_qimg21()
    from qimg21_i23d import backends, ops, imageops
    prompt = PROMPT.format(outfit=outfit)
    errors = []
    with gpu.device_session(8192 if preset == "low8" else BODY_FAST12_MIN_FREE_MIB, cancel):
        backend = qwen.make_backend(python, False, preset=preset)
        try:
            for attempt in range(attempts):
                s = seed + attempt
                raw = out / "body_image_raw.png"
                progress(.025 + .025 * attempt, f"Qwen full-body image (seed {s})")
                try:
                    backend.generate(backends.GenRequest(prompt=prompt, out=raw,
                        width=768, height=1280, steps=steps, seed=s,
                        references=(head / "portrait.png",)))
                    rgba = imageops.load_rgba(raw)
                    method = "alpha" if (rgba[..., 3] < 250).any() else "rmbg"
                    # Preserve the tall canvas. normalize_object(size=...) uses
                    # its shorter dimension as a square-object target and
                    # would shrink a full-length person to about 60% height.
                    ops.preprocess_object(raw, image, backend, method=method,
                                          size=None, fill=None, center=False)
                    bbox = _image_gate(image)
                    return {"prompt": prompt, "seed": s, "bbox": bbox,
                            "backend": backend.name, "preset": preset,
                            "extraction": method, "attempts": errors}
                except (ValueError, RuntimeError) as exc:
                    errors.append({"seed": s, "error": str(exc)})
        finally:
            backend.close()
    raise ValueError("no usable full-body image: " + str(errors[-1]))


def _sam_body(out: Path, bbox, model_dir: Path, mock: bool, cancel) -> dict:
    if mock:
        meta = {"bbox": list(bbox), "image": {"width": 768, "height": 1280},
                "focal_px": 1000.0, "cam_t": [0.0, 0.0, 3.0],
                "model_params": [0.0] * 204, "shape": [0.0] * 45,
                "mhr_params": [0.0] * 519, "keypoints_3d": [], "keypoints_2d": []}
        (out / "body_mhr.glb.json").write_text(json.dumps(meta))
        return {"backend": "mock", "bbox": bbox}
    use_cuda = gpu.gpu_status() is not None
    binary = _binary("sam3d_body", use_cuda, cancel)
    cmd = [str(binary), "--safetensors-dir", str(model_dir / "safetensors"),
           "--mhr-assets", str(model_dir / "safetensors"), "--image", str(out / "body_image.png"),
           "--bbox", *[str(n) for n in bbox], "--device", str(gpu.device_index()), "--backbone", "dinov3", "-o", str(out / "body_mhr.glb")]
    lock = gpu.LOCK_PATH if use_cuda else None
    if lock:
        with gpu.device_session(2048, cancel):
            _run(cmd, cancel)
    else:
        _run(cmd, cancel)
    meta = json.loads((out / "body_mhr.glb.json").read_text())
    if len(meta.get("model_params", [])) != 204:
        raise ValueError("SAM 3D Body runner lacks decoded MHR parameters; rebuild it")
    return {"backend": gpu.backend() if use_cuda else "cpu", "bbox": bbox}


def _pixal(out: Path, image: Path, name: str, quality: str, mock: bool, cancel) -> dict:
    if mock:
        return {"runner": "mock", "status": "skipped"}
    target = out / name
    with gpu.device_session(gpu.PIXAL3D_MIN_FREE_MIB, cancel):
        if name.startswith("garment_"):
            qwen._import_qimg21()
            from qimg21_i23d import reconstruct
            settings = reconstruct.ReconSettings.preset("preview", triangle_target=80_000)
            settings.cancel = cancel
            runner = baseline.pixal_runner(settings)
            return runner.single(image, target, out / (name + ".work"), fov_rad=math.radians(20))
        return baseline.run_pixal3d(image, target, out / (name + ".work"), quality, cancel)


def _garments(out: Path, names: list[str], mock: bool, cancel, progress,
              sam3_model: Path = SAM3_MODEL, clip_bpe: Path = CLIP_BPE) -> list[dict]:
    if not names:
        (out / "garments.json").write_text("[]")
        return []
    source = np.asarray(Image.open(out / "body_image.png").convert("RGBA"))
    use_cuda = gpu.gpu_status() is not None and not mock
    sam3_model = Path(sam3_model)
    clip_bpe = Path(clip_bpe)
    entries = []
    for i, name in enumerate(names):
        stem = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")[:32] or "item"
        slug = f"{i + 1:02d}_{stem}"
        record = {"name": slug, "prompt": name}
        try:
            if mock:
                raise ValueError("segmentation skipped in mock mode")
            required = (sam3_model, clip_bpe / "vocab.json", clip_bpe / "merges.txt")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ValueError("optional SAM 3 garment assets missing: " + ", ".join(missing))
            slot_start = .51 + .24 * i / len(names)
            progress(slot_start, f"segmenting {name}")
            mask_npy = out / f"garment_{slug}_masks.npy"
            cmd = [str(_binary("sam3", use_cuda, cancel)), str(sam3_model), str(out / "body_image.png"),
                   "--device", str(gpu.device_index()), "--phrase", name, "-o", str(mask_npy),
                   "--vocab", str(clip_bpe / "vocab.json"), "--merges", str(clip_bpe / "merges.txt")]
            if use_cuda:
                with gpu.device_session(2048, cancel):
                    _run(cmd, cancel)
            else:
                _run(cmd, cancel)
            if not mask_npy.is_file():
                raise ValueError("SAM 3 found no garment instance")
            masks = np.load(mask_npy)
            if masks.ndim != 3 or masks.shape[1:] != source.shape[:2]:
                raise ValueError("SAM 3 mask dimensions differ from body image")
            alpha = source[..., 3] > 127
            score = np.asarray([(m.astype(bool) & alpha).sum() for m in masks])
            mask = masks[int(score.argmax())].astype(bool) & alpha
            if int(mask.sum()) < source.shape[0] * source.shape[1] * .004:
                raise ValueError("garment mask is too small")
            png = f"garment_{slug}_mask.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(out / png)
            crop = source.copy()
            crop[..., 3] = np.where(mask, source[..., 3], 0)
            crop_path = out / f"garment_{slug}_input.png"
            Image.fromarray(crop).save(crop_path)
            glb = f"garment_{slug}.glb"
            progress(slot_start + .08 / len(names), f"Pixal3D {name}")
            _pixal(out, crop_path, glb, "preview", False, cancel)
            record.update(status="reconstructed", mask=png, glb=glb, pixels=int(mask.sum()))
        except gpu.Cancelled:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            record.update(status="fallback_body", reason=str(exc))
        entries.append(record)
    (out / "garments.json").write_text(json.dumps(entries, indent=1))
    return entries


def body_job(service, request: dict, progress, cancel, *, python=None, rig_python=None,
             model_dir=MODEL_DIR, sam3_model=SAM3_MODEL, clip_bpe=CLIP_BPE, mock=False) -> dict:
    head_id = request.get("head_id")
    head = service.head_file(head_id, "head.json").parent
    for name in ("portrait.png", "rig/rig.glb", "rig/rig.json", "rig/features.json"):
        if "/" in name:
            service.rig_file(head_id, name.split("/", 1)[1])
        else:
            service.head_file(head_id, name)
    model_dir = Path(model_dir)
    sam3_model = Path(sam3_model)
    clip_bpe = Path(clip_bpe)
    rig_python = Path(rig_python) if rig_python else DEFAULT_RIG_PYTHON
    avail = availability(model_dir, rig_python, mock)
    if not avail["available"]:
        reason = ("body model or rig interpreter missing: " + ", ".join(avail["missing"])) \
                 if avail["missing"] else avail["reason"]
        raise ValueError(reason)
    quality = request.get("quality", "standard")
    if quality not in ("preview", "standard", "high"):
        raise ValueError("quality must be preview, standard or high")
    outfit = " ".join(str(request.get("outfit", "plain fitted shirt, trousers and shoes")).split())[:300]
    if not outfit:
        raise ValueError("outfit must be nonempty")
    seed = int(request.get("seed", 11))
    steps = int(request.get("steps", 24))
    attempts = max(1, min(3, int(request.get("attempts", 3))))
    qwen_preset = _qwen_preset(request.get("qwen_preset", "auto"), gpu.gpu_status())
    names = request.get("garments", ["shirt", "pants", "shoes"])
    if not isinstance(names, list) or len(names) > 5 or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("garments must be a list of at most five phrases")
    out = head / (".body-" + uuid.uuid4().hex[:12])
    out.mkdir()
    try:
        started = time.perf_counter()
        generated = _generate_image(head, out, outfit, seed, attempts, steps, python,
                                    qwen_preset, mock, progress, cancel)
        progress(.12, "SAM 3D Body mesh and pose")
        sam = _sam_body(out, generated["bbox"], model_dir, mock, cancel)
        progress(.37, "Pixal3D whole-body appearance")
        pixal = None
        try:
            pixal = _pixal(out, out / "body_image.png", "pixal3d_full.glb", quality, mock, cancel)
        except gpu.Cancelled:
            raise
        except (RuntimeError, OSError, ValueError) as exc:
            pixal = {"status": "fallback_photo", "reason": str(exc)}
        garment_report = _garments(out, names, mock, cancel, progress, sam3_model, clip_bpe)
        progress(.77, "assembling MHR body and facial rig")
        res = {"preview": 1024, "standard": 2048, "high": 4096}[quality]
        cmd = [str(rig_python), "-m", "server.vhuman.body.assemble", str(head),
               "--out", str(out), "--model", str(model_dir / "dinov3/assets/mhr_model.pt"),
               "--head-assets", str(model_dir / "safetensors/sam3d_body_mhr_head.safetensors"),
               "--res", str(res)]
        with gpu.device_session(1024, cancel) if gpu.backend() != "cpu" else nullcontext():
            _run(cmd, cancel, timeout=1200)
        report = json.loads((out / "body_report.json").read_text())
        report["generation"] = generated
        report["sam3d_body"] = sam
        report["pixal3d_run"] = {k: pixal.get(k) for k in ("runner", "seconds", "status", "reason") if k in pixal}
        report["garment_attempts"] = garment_report
        report["provenance"] = {"head_id": head_id, "sam3d_body_model_dir": str(model_dir),
                                "mhr_model": str(model_dir / "dinov3/assets/mhr_model.pt"),
                                "sam3_model": str(sam3_model) if names else None,
                                "clip_bpe": str(clip_bpe) if names else None}
        report["total_seconds"] = round(time.perf_counter() - started, 2)
        (out / "body_report.json").write_text(json.dumps(report, indent=1, default=float))
        published = head / "body"
        previous = head / (".body-previous-" + uuid.uuid4().hex[:8])
        if published.exists():
            published.rename(previous)
        try:
            out.rename(published)
        except Exception:
            if previous.exists():
                previous.rename(published)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        progress(.99, "avatar ready")
        return {"id": head_id, "seconds": report["total_seconds"],
                "glb_url": f"/v1/heads/{head_id}/body/avatar.glb",
                "usd_url": f"/v1/heads/{head_id}/body/avatar_usd.zip",
                "report_url": f"/v1/heads/{head_id}/body/body_report.json"}
    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
