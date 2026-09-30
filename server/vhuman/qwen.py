"""Qwen-Image 2.1 iris plates: photographic iris detail, generated on demand.

A plate is one generated macro photograph of an iris, normalised by
eye/extract.py into our iris texture layout (structure masks plus the
photo's own colours) and kept in a library under tmp/vhuman/plates/<id>/.
The GPU is used only here, in jobs: a job batches its images on one
resident model and closes it at the end, so nothing stays in device memory.

Two modes:
- "text": text -> image from a prompt template and an iris colour;
- "refine": SDEdit of our procedural iris (the current parameters) at a
  given strength, which keeps its layout and colours and adds photographic
  detail.

Generated images are covered by the Qwen Research License (non-commercial):
plates stay in tmp/ and carry "license": "qwen-research".
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import gpu
from .eye import extract, params as P

ROOT = Path(__file__).resolve().parents[2]
QIMG21 = ROOT / "cuda" / "qimg21"
LICENSE = "qwen-research"

STYLES = {
    "macro": (
        "Extreme close-up macro photograph of a single human iris, {color}. The iris is a perfect circle seen "
        "perfectly straight-on, centred in the frame and filling about two thirds of the image width, with a round "
        "black pupil exactly in its centre and white sclera visible all around it. Fine radial stroma fibres, "
        "crypts, a zigzag collarette and a dark limbal ring, in sharp focus across the whole iris. Cross-polarised "
        "ring-light illumination with no reflections, no catchlights and no glare. No eyelids, no eyelashes, no "
        "skin, no text."),
    "clinical": (
        "Clinical iris photograph for iris recognition: a {color} human iris, frontal, perfectly circular and "
        "centred, the pupil in the middle, the whole iris and a margin of white sclera in the frame, evenly lit "
        "with diffuse near-infrared-free white light, no specular highlights, very high detail of the iris "
        "texture, fibres and crypts. No eyelids or lashes covering the iris. Photorealistic."),
    "ocularist": (
        "Hyper-realistic photograph of a {color} human iris as painted by an ocularist for a prosthetic eye: a "
        "perfectly round iris viewed head-on on a white sclera, a round black pupil in the centre, intricate "
        "radial fibres, crypts and a collarette, matte finish with no reflections, even lighting, sharp focus. "
        "No eyelids, no text."),
    "minimal": "A photo of a {color} human iris, straight-on, centred, round black pupil, no reflections.",
}
REFINE_PROMPT = (
    "Photorealistic extreme macro photograph of this human iris seen straight-on: keep exactly the same colours, "
    "the same pupil size and the same pattern layout, and add natural photographic detail: fine radial stroma "
    "fibres, crypts, a textured collarette and a soft limbal ring. Cross-polarised lighting, no reflections, no "
    "eyelids.")
COLORS = {
    "light_blue": "light blue", "blue": "blue", "blue_gray": "grey-blue", "gray": "grey", "green": "green",
    "gray_green": "grey-green", "hazel": "hazel, green with a golden-brown ring around the pupil",
    "amber": "amber, golden-copper", "light_brown": "light brown", "brown": "brown", "dark_brown": "dark brown",
    "central_heterochromia": "blue with an amber-brown centre around the pupil (central heterochromia)",
}


def color_words(color: str) -> str:
    """A preset name or free text (short, safe for a prompt)."""
    if color in COLORS:
        return COLORS[color]
    text = " ".join(str(color).split())[:80]
    if not text:
        raise ValueError("an iris colour is needed")
    return text


def plate_prompt(color: str, style: str = "macro") -> str:
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}")
    return STYLES[style].format(color=color_words(color))


class SyntheticIrisBackend:
    """--mock stand-in for Qwen: returns a synthetic eye photograph (our
    procedural iris, placed off-centre with a catchlight), so the plate
    pipeline (extraction, gate, library, web page) runs without a GPU."""

    name = "mock"

    def generate(self, request):
        from qimg21_i23d.backends import GenResult
        started = time.perf_counter()
        names = list(P.PRESETS)
        preset = names[request.seed % len(names)]
        img, _, _ = extract.synthetic_eye(P.preset(preset), request.width, request.seed, occlude=False)
        Image.fromarray((img * 255).astype(np.uint8)).save(request.out)
        return GenResult(Path(request.out), time.perf_counter() - started, self.name, {"preset": preset})

    def close(self):
        pass


def _import_qimg21():
    if str(QIMG21) not in sys.path:
        sys.path.insert(0, str(QIMG21))


def availability(python=None, mock=False) -> dict:
    if mock:
        return {"available": True, "backend": "mock"}
    try:
        _import_qimg21()
        from qimg21_i23d.native import NativeBackend
    except Exception as exc:  # noqa: BLE001  (report, never fail /health)
        return {"available": False, "reason": f"qimg21_i23d: {exc}"}
    if not native_available():
        return {"available": False, "reason": "the native Qwen-Image 2.1 runner or its weights are missing"}
    if not python or not Path(python).exists():
        return {"available": False, "reason": "no --qwen-python interpreter (it needs torch)"}
    return {"available": True, "backend": "native-rocm" if gpu.backend() == "rocm" else "native"}


def native_available():
    if gpu.backend() == "cpu":
        return False
    directory = ROOT / ("rdna4" if gpu.backend() == "rocm" else "cuda") / "qimg21"
    prefix = "test_hip_" if gpu.backend() == "rocm" else "test_cuda_"
    return all((directory / (prefix + "qimg21_" + name)).is_file()
               for name in ("fast", "text", "vision", "vae", "vae_encode")) and gpu.model_path("qimg-21").is_dir()


def make_backend(python=None, mock=False, preset="fast12"):
    _import_qimg21()
    if mock:
        return SyntheticIrisBackend()
    from qimg21_i23d.native import NativeBackend
    if preset not in ("fast12", "low8"):
        raise ValueError("qwen_preset must be fast12 or low8")
    return NativeBackend(model=gpu.model_path("qimg-21"), python=python, preset=preset,
                         resident=preset != "low8", backend=gpu.backend(), device=gpu.device_index(),
                         quant_package=gpu.model_path("qimg-21-fast/int8-smooth-a0.6"))


def _thumb(photo: np.ndarray, size: int = 160) -> Image.Image:
    img = Image.fromarray((np.clip(photo, 0, 1) * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([1, 1, size - 2, size - 2], fill=255)
    out = Image.new("RGBA", (size, size))
    out.paste(img, (0, 0), mask)
    return out


def save_plate(root: Path, plate: extract.Plate, source: Path, record: dict) -> dict:
    """Write a plate folder: source image, masks, photo, thumbnail, plate.json."""
    plate_id = uuid.uuid4().hex[:16]
    folder = root / plate_id
    folder.mkdir(parents=True)
    Image.open(source).convert("RGB").save(folder / "source.png")
    Image.fromarray((np.clip(plate.masks, 0, 1) * 255 + 0.5).astype(np.uint8)).save(folder / "iris_masks.png")
    Image.fromarray((np.clip(plate.photo, 0, 1) * 255 + 0.5).astype(np.uint8)).save(folder / "iris_photo.png")
    _thumb(plate.photo).save(folder / "thumb.png")
    record = dict(record, id=plate_id, created=time.time(), license=LICENSE, quality=plate.quality,
                  colors=plate.colors, limbus=plate.limbus.as_list(), pupil=plate.pupil.as_list())
    (folder / "plate.json").write_text(json.dumps(record, indent=1, default=float))
    return record


def replate(library: Path) -> dict:
    """Re-extract every plate from its stored source image (after the iris
    layout or mask semantics change); keeps ids, prompts and seeds."""
    from .eye import params as P
    done, failed = [], []
    for meta in sorted(Path(library).glob("*/plate.json")):
        folder = meta.parent
        rec = json.loads(meta.read_text())
        try:
            plate = extract.extract(folder / "source.png")
        except extract.ExtractError as exc:
            failed.append({"id": rec["id"], "error": str(exc)})
            continue
        Image.fromarray((np.clip(plate.masks, 0, 1) * 255 + 0.5).astype(np.uint8)).save(folder / "iris_masks.png")
        Image.fromarray((np.clip(plate.photo, 0, 1) * 255 + 0.5).astype(np.uint8)).save(folder / "iris_photo.png")
        _thumb(plate.photo).save(folder / "thumb.png")
        rec.update(quality=plate.quality, colors=plate.colors, limbus=plate.limbus.as_list(),
                   pupil=plate.pupil.as_list(), algo_version=P.ALGO_VERSION)
        meta.write_text(json.dumps(rec, indent=1, default=float))
        done.append(rec["id"])
    return {"replated": len(done), "failed": failed}


def refine_init(params: dict, size: int, path: Path) -> Path:
    """Our procedural iris, centred on a white sclera, as the SDEdit start."""
    p = P.validate(params)
    img, _, _ = extract.synthetic_eye(p, size, 0, occlude=False, highlight=False, blur=0.0, noise_std=0.0,
                                      radius_frac=0.34, centred=True)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path


def generate_plates(backend, library: Path, rejected: Path, *, count: int = 4, seed: int = 1,
                    color: str = "hazel", style: str = "macro", mode: str = "text", params: dict | None = None,
                    strength: float = 0.5, size: int = 1024, steps: int = 20, progress=None) -> dict:
    """Generate `count` images, extract them and keep the ones that pass the
    gate in the library (the rest go to `rejected` for the study)."""
    _import_qimg21()
    from qimg21_i23d.backends import GenRequest
    work = library.parent / "plate-work"
    work.mkdir(parents=True, exist_ok=True)
    kept, dropped = [], []
    prompt = REFINE_PROMPT if mode == "refine" else plate_prompt(color, style)
    init = refine_init(params or {}, size, work / f"init-{uuid.uuid4().hex[:8]}.png") if mode == "refine" else None
    for i in range(count):
        s = seed + i
        if progress:
            progress(i / count, f"image {i + 1}/{count} (seed {s})")
        out = work / f"qwen-{uuid.uuid4().hex[:8]}.png"
        started = time.perf_counter()
        req = GenRequest(prompt=prompt, out=out, width=size, height=size, steps=steps, seed=s,
                         init_image=init, strength=strength if mode == "refine" else 1.0)
        result = backend.generate(req)
        gen_s = time.perf_counter() - started
        record = {"prompt": prompt, "style": style if mode == "text" else "refine", "mode": mode,
                  "color": color, "seed": s, "steps": steps, "size": size, "backend": result.backend,
                  "generate_seconds": round(gen_s, 2), "strength": strength if mode == "refine" else None}
        try:
            plate = extract.extract(out)
        except extract.ExtractError as exc:
            rec = dict(record, error=str(exc))
            dropped.append(rec)
            (rejected / f"{out.stem}.json").write_text(json.dumps(rec, indent=1))
            out.replace(rejected / out.name)
            continue
        if plate.quality["ok"]:
            kept.append(save_plate(library, plate, out, record))
            out.unlink(missing_ok=True)
        else:
            dropped.append(save_plate(rejected, plate, out, record))
            out.unlink(missing_ok=True)
    if init is not None:
        init.unlink(missing_ok=True)
    return {"kept": kept, "rejected": dropped, "prompt": prompt}


def plates_job(service, request: dict, progress, cancel, python=None, mock=False) -> dict:
    """A GPU job: {count, seed, color, style, mode, params, strength, steps}."""
    count = int(request.get("count", 4))
    if not 1 <= count <= 16:
        raise ValueError("count must be in [1, 16]")
    seed = int(request.get("seed", 1))
    steps = int(request.get("steps", 20))
    if not 4 <= steps <= 50:
        raise ValueError("steps must be in [4, 50]")
    mode = request.get("mode", "text")
    if mode not in ("text", "refine"):
        raise ValueError("mode must be text or refine")
    strength = float(request.get("strength", 0.5))
    if not 0.2 <= strength <= 0.9:
        raise ValueError("strength must be in [0.2, 0.9]")
    color = request.get("color", "hazel")
    style = request.get("style", "macro")
    plate_prompt(color, style)             # validates colour and style before any GPU work
    rejected = service.work / "plates-rejected"
    rejected.mkdir(parents=True, exist_ok=True)
    progress(0.0, "waiting for the GPU")
    session = gpu.device_session(gpu.QWEN_MIN_FREE_MIB, cancel, check_memory=not mock,
                                 lock_path=(service.work / "mock-gpu.lock") if mock else gpu.LOCK_PATH)
    with session:
        backend = make_backend(python, mock)
        try:
            def step(fraction, message):
                if cancel.is_set():
                    raise gpu.Cancelled("cancelled")
                progress(fraction, message)
            result = generate_plates(backend, service.plates, rejected, count=count, seed=seed, color=color,
                                     style=style, mode=mode, params=request.get("params"), strength=strength,
                                     steps=steps, progress=step)
        finally:
            backend.close()
    kept = result["kept"]
    return {"kept": [k["id"] for k in kept], "rejected": len(result["rejected"]),
            "failures": [r.get("quality", {}).get("failures") or r.get("error") for r in result["rejected"]],
            "prompt": result["prompt"]}
