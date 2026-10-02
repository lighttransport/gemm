"""Portrait expression video jobs. Generated clips are previews, not rig training data."""
from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
import threading
import uuid
from . import gpu
from .video_backend import select as select_backend
from .service import ServiceError
from cuda.hunyuan_video15 import native_generate as native

EXPRESSIONS = {
    "smile": "A natural closed-mouth smile gradually appears, holds, then relaxes.",
    "laugh": "The person laughs softly, cheeks lifting and eyes narrowing naturally, then relaxes.",
    "surprise": "The eyebrows rise and eyes widen in mild surprise, then return to neutral.",
    "sad": "The expression becomes gently sad, inner eyebrows lifting, then relaxes.",
    "angry": "The brows gently furrow and lips tighten in restrained anger, then relaxes.",
    "blink": "Brief natural eyelid closures while maintaining a relaxed face.",
}
# This measured prompt gives more complete, shorter closures in the fast12
# portrait test. The model still does not reliably follow the requested count.
BRIEF_BLINK_PROMPT = (
    "A steady close-up shot. The person keeps their face still and eyes open, "
    "makes one quick natural blink, immediately opens both eyes again, and stays relaxed. "
    "The lips remain gently closed."
)
FILES = {"clip.mp4", "poster.png", "manifest.json", "metrics.json"}

def validate(request):
    if not isinstance(request, dict):
        raise ServiceError("video request must be an object")
    hid = request.get("head_id")
    if not isinstance(hid, str) or not re.fullmatch(r"[A-Za-z0-9]{1,32}", hid):
        raise ServiceError("invalid head_id")
    expression = request.get("expression", "smile")
    if not isinstance(expression, str) or expression not in EXPRESSIONS:
        raise ServiceError("unknown expression")
    preset = request.get("preset", "quality")
    if not isinstance(preset, str) or preset not in ("quality", "fast12"):
        raise ServiceError("preset must be quality or fast12")
    frames, seed = request.get("frames", 81), request.get("seed", 42)
    if type(frames) is not int or frames not in (81, 121):
        raise ServiceError("frames must be 81 or 121")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ServiceError("seed must be a non-negative signed 64-bit integer")
    prompt = request.get("prompt", "")
    if not isinstance(prompt, str) or len(prompt.encode()) > 4096:
        raise ServiceError("prompt must be a string of at most 4096 bytes")
    prompt = prompt.strip() or (BRIEF_BLINK_PROMPT if expression == "blink" else
        "Photorealistic close-up video of the person in the reference portrait. " + EXPRESSIONS[expression] +
        " Preserve facial identity, hairstyle, skin texture and lighting. Fixed camera, subtle natural head motion.")
    return {"head_id": hid, "expression": expression, "preset": preset,
            "frames": frames, "seed": seed, "prompt": prompt}

def directory(service, hid):
    return service.head_file(hid, "portrait.png").parent / "videos"

def video_file(service, hid, run, name):
    if not re.fullmatch(r"[a-f0-9]{32}", run) or name not in FILES:
        raise ServiceError("no such video file")
    base = directory(service, hid).resolve()
    path = (base / run / name).resolve()
    if not path.is_relative_to(base) or not path.is_file() or not (base / run / "manifest.json").is_file():
        raise ServiceError("no such video file")
    return path

def list_videos(service, hid):
    base = directory(service, hid)
    result = []
    if base.exists():
        for run in sorted(base.iterdir(), key=lambda p: p.name, reverse=True):
            if not re.fullmatch(r"[a-f0-9]{32}", run.name):
                continue
            try:
                manifest = json.loads(video_file(service, hid, run.name, "manifest.json").read_text())
                video_file(service, hid, run.name, "clip.mp4")
                result.append({"id": run.name, "head_id": hid, "request": manifest.get("request", {}),
                    "backend": manifest.get("backend"), "url": f"/v1/heads/{hid}/videos/{run.name}/clip.mp4",
                    "poster": f"/v1/heads/{hid}/videos/{run.name}/poster.png"})
            except (ServiceError, ValueError, OSError):
                continue
    return result

def availability(model=None, runner=None, *, mock=False, allow_experimental=False, backend='repo'):
    selected = native if mock else select_backend(backend)
    presets = ["quality", "fast12"] if mock else []
    if not mock and model:
        for preset in ("quality", "fast12"):
            try:
                selected.load_manifest(model, "i2v", preset)
                presets.append(preset)
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return {"available": mock or bool(presets and Path(runner or selected.RUNNER).is_file() and allow_experimental),
        "experimental": not mock, "mock": mock, "presets": presets,
        "expressions": list(EXPRESSIONS), "frames": list(getattr(selected, "frames", (81, 121))), "fps": 24,
        "backend": "mock" if mock else backend,
        "parity": "unverified", "memory_fit": "unverified"}

def video_job(service, request, progress, cancel: threading.Event, *, model=None,
              runner=None, mock=False, allow_experimental=False, backend='repo'):
    selected = native if mock else select_backend(backend)
    req = validate(request)
    if req['frames'] not in getattr(selected, 'frames', (81, 121)):
        raise ServiceError('selected video backend does not support this frame count')
    portrait = service.head_file(req["head_id"], "portrait.png")
    base = directory(service, req["head_id"])
    base.mkdir(parents=True, exist_ok=True)
    run = uuid.uuid4().hex
    out = base / run
    progress(0.02, "preparing expression video")
    try:
        if mock:
            out.mkdir()
            prepared, pixels = native.prepare_portrait(portrait, out, 480, 848)
            prepared.replace(out / "poster.png")
            pixels.unlink()
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise ServiceError("ffmpeg is required even for mock video")
            native.run_process([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-loop", "1", "-framerate", "24", "-i", out / "poster.png", "-frames:v", req["frames"],
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out / "clip.mp4"], cancel=cancel)
            manifest = {"schema": "hunyuan_video15.video.v1", "backend": "mock",
                "description": "static portrait fixture; no expression generation", "request": req}
            (out / "metrics.json").write_text(json.dumps({"mock": True}) + "\n")
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        else:
            if not allow_experimental:
                raise ServiceError("enable --video-experimental; full pipeline parity is unverified")
            if gpu.backend() != "cuda":
                raise ServiceError("HunyuanVideo-1.5 currently requires CUDA")
            if not model:
                raise ServiceError("configure --video-model with a prepared HunyuanVideo-1.5 model directory")
            progress(0.03, "waiting for the CUDA device")
            with gpu.device_session(14336, cancel=cancel):
                manifest = selected.generate(model=model, runner=runner or selected.RUNNER,
                    image=portrait, prompt=req["prompt"], out=out, preset=req["preset"],
                    frames=req["frames"], seed=req["seed"], device=gpu.device_index(),
                    allow_experimental=allow_experimental, cancel=cancel,
                    progress=lambda s, n: progress(0.05 + 0.85 * s / max(n, 1),
                        "rendering video frames" if n > 0 and s >= n else f"video step {s}/{n}"))
            manifest["request"] = req
            partial = out / "manifest.json.partial"
            partial.write_text(json.dumps(manifest, indent=2) + "\n")
            partial.replace(out / "manifest.json")
        if cancel.is_set():
            raise gpu.Cancelled("cancelled")
        progress(1.0, "video ready")
        return {"head_id": req["head_id"], "id": run, "backend": "mock" if mock else backend,
            "url": f"/v1/heads/{req['head_id']}/videos/{run}/clip.mp4",
            "poster": f"/v1/heads/{req['head_id']}/videos/{run}/poster.png"}
    except selected.Cancelled:
        shutil.rmtree(out, ignore_errors=True)
        raise gpu.Cancelled("cancelled") from None
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
