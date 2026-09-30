"""Optional face-video observations and PyTorch fitting of the vhuman rig.

MediaPipe supplies observations only. The fitted controls are found by our
TorchRig projection and temporal optimizer; browser playback has no tracker.
"""
from __future__ import annotations

from contextlib import nullcontext
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import uuid
from pathlib import Path

import numpy as np

from ..service import ServiceError
from . import rigdef, safetensors as st
from .. import runtime, gpu

VIDEO_TYPES = {"video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov"}
MAX_UPLOAD = 64 << 20
MAX_SECONDS = 30
FPS = 30
MODEL = Path(__file__).resolve().parents[3] / "tmp/vhuman-rig/models/face_landmarker.task"
ANCHORS = (("nose_tip", 4), ("upper_lip", 13), ("lower_lip", 14),
           ("menton", 152), ("mouth_right", 61), ("mouth_left", 291),
           ("eye_right", (33, 133, 159, 145)), ("eye_left", (263, 362, 386, 374)))


def upload(service, stream, length: int, content_type: str) -> dict:
    ctype = content_type.split(";", 1)[0].strip().lower()
    if ctype not in VIDEO_TYPES:
        raise ServiceError("face upload must be MP4, WebM or MOV")
    if not 0 < length <= MAX_UPLOAD:
        raise ServiceError("face video must be between 1 byte and 64 MiB")
    uid = uuid.uuid4().hex[:16]
    folder = service.work / "rig_uploads" / uid
    folder.mkdir(parents=True)
    source = folder / ("source" + VIDEO_TYPES[ctype])
    try:
        digest = hashlib.sha256()
        with source.open("wb") as out:
            remaining = length
            while remaining:
                block = stream.read(min(1 << 20, remaining))
                if not block:
                    raise ServiceError("face upload ended before Content-Length")
                out.write(block)
                digest.update(block)
                remaining -= len(block)
        with source.open("rb") as inp:
            sig = inp.read(32)
        if not (sig[4:8] == b"ftyp" or sig.startswith(b"\x1a\x45\xdf\xa3")):
            raise ServiceError("face video container signature is invalid")
        meta = {"id": uid, "mime": ctype, "bytes": length, "sha256": digest.hexdigest(), "file": source.name}
        (folder / "upload.json").write_text(json.dumps(meta))
        return {k: meta[k] for k in ("id", "bytes")}
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def source_file(service, upload_id: str) -> tuple[Path, dict]:
    if not isinstance(upload_id, str) or len(upload_id) != 16 or any(c not in "0123456789abcdef" for c in upload_id):
        raise ServiceError("invalid face upload id")
    folder = service.work / "rig_uploads" / upload_id
    meta_path = folder / "upload.json"
    if not meta_path.is_file():
        raise ServiceError("no such face upload")
    meta = json.loads(meta_path.read_text())
    source = folder / meta["file"]
    if not source.is_file():
        raise ServiceError("no such face upload")
    return source, meta


def _extract(source: Path, stage: Path) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is needed for face video")
    info = subprocess.run([ffmpeg, "-hide_banner", "-i", str(source)],
                          capture_output=True, text=True, timeout=20)
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info.stderr)
    if not match:
        raise ValueError("face video has no known duration")
    duration = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
    if not math.isfinite(duration) or not .1 <= duration <= MAX_SECONDS:
        raise ValueError("face video duration must be between 0.1 and 30 seconds")
    frames = stage / "frames"
    frames.mkdir()
    subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "2",
                    "-i", str(source), "-vf", f"fps={FPS},scale=768:768:force_original_aspect_ratio=decrease",
                    "-frames:v", str(FPS * MAX_SECONDS), "-q:v", "3", "-an", str(frames / "%05d.jpg")],
                   check=True, timeout=180)
    paths = sorted(frames.glob("*.jpg"))
    if not 3 <= len(paths) <= FPS * MAX_SECONDS:
        raise ValueError("face video has too few or too many decoded frames")
    audio = stage / "audio.wav"
    result = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "2",
                             "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                             str(audio)], capture_output=True, timeout=180)
    if result.returncode or not audio.is_file() or audio.stat().st_size <= 44:
        audio.unlink(missing_ok=True)
    return paths


def _observe(paths: list[Path], model: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # MediaPipe's package initializer imports its unused audio task. On some
    # headless hosts that stalls while PortAudio probes devices. Load only the
    # vision task in this dedicated fitting subprocess.
    import sys
    import types

    cache = Path(__file__).resolve().parents[3] / "tmp/vhuman-rig/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    sys.modules.setdefault("mediapipe.tasks.python.audio", types.ModuleType("mediapipe.tasks.python.audio"))
    import mediapipe as mp

    if not model.is_file():
        raise ValueError(f"face landmarker model is missing: {model}; run setup_face_video.sh")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    names = list(rigdef.CONTROLS)
    controls = np.zeros((len(paths), len(names)), np.float32)
    landmarks = np.zeros((len(paths), len(ANCHORS), 2), np.float32)
    valid = np.zeros(len(paths), bool)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as tracker:
        for i, path in enumerate(paths):
            image = mp.Image.create_from_file(str(path))
            result = tracker.detect_for_video(image, round(i * 1000 / FPS))
            if len(result.face_landmarks) > 1:
                raise ValueError("face video must contain exactly one visible face")
            if not result.face_landmarks:
                continue
            valid[i] = True
            points = result.face_landmarks[0]
            for j, (_, index) in enumerate(ANCHORS):
                chosen = [points[k] for k in (index if isinstance(index, tuple) else (index,))]
                landmarks[i, j] = [np.mean([p.x for p in chosen]), np.mean([p.y for p in chosen])]
            for category in result.face_blendshapes[0]:
                if category.category_name in rigdef.CONTROLS:
                    controls[i, names.index(category.category_name)] = category.score
    if valid.mean() < .8:
        raise ValueError("face is missing in more than 20% of video frames")
    observed = np.flatnonzero(valid)
    for series in (controls, landmarks):
        for j in np.ndindex(series.shape[1:-1]) if series.ndim > 2 else [()]:
            for k in range(series.shape[-1]):
                column = series[(slice(None), *j, k)]
                series[(slice(None), *j, k)] = np.interp(np.arange(len(series)), observed, column[observed])
    return controls, landmarks, valid


def _anchor_rig(rig_dir: Path):
    from .torchrig import TorchRig

    definition = json.loads((rig_dir / "rig.json").read_text())
    package, meta = st.load(rig_dir / "rig_deformer.safetensors")
    rest = package["rest"].astype(np.float32)
    features = json.loads((rig_dir / "features.json").read_text())
    eye = {"eye_" + e["side"]: e["center"] for e in features["eyes"]}
    locations = [features["points"].get(name, eye.get(name)) for name, _ in ANCHORS]
    ids = np.asarray([np.argmin(np.linalg.norm(rest - np.asarray(p), axis=1)) for p in locations], np.int32)
    morph_names = json.loads(meta["morphs"])
    base_names = {b["name"] for b in definition["blendshapes"]}
    shapes = {name: package["morph"][i, ids] for i, name in enumerate(morph_names) if name in base_names}
    rig = TorchRig(definition, rest[ids], shapes, package["skin.joints"][ids],
                   package["skin.weights"][ids], device=runtime.torch_device(__import__("torch")))
    return rig, definition, rest[ids]


def fit_observations(rig_dir: Path, direct: np.ndarray, landmarks: np.ndarray,
                     valid: np.ndarray, steps: int = 120) -> tuple[np.ndarray, dict]:
    """Fit visible 2D landmark motion while keeping MediaPipe controls as priors."""
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = runtime.torch_device(torch)
    rig, definition, anchor_rest = _anchor_rig(rig_dir)
    if direct.shape != (len(landmarks), len(definition["controls"])) or landmarks.shape != (len(direct), len(ANCHORS), 2):
        raise ValueError("invalid video observation shapes")
    lo = np.array([c["min"] for c in definition["controls"]], np.float32)
    hi = np.array([c["max"] for c in definition["controls"]], np.float32)
    direct = np.clip(direct, lo, hi)
    eye_distance = anchor_rest[7, 0] - anchor_rest[6, 0]
    observed_distance = np.median(landmarks[valid, 7, 0] - landmarks[valid, 6, 0])
    if eye_distance <= .02 or observed_distance <= .02:
        raise ValueError("face eye spacing is invalid for projection fit")
    scale = float(observed_distance / eye_distance)
    center = landmarks[:, 0] - np.stack((scale * np.full(len(direct), anchor_rest[0, 0]),
                                         -scale * np.full(len(direct), anchor_rest[0, 1])), axis=1)
    prior = torch.from_numpy(direct).to(device)
    observed = torch.from_numpy(landmarks).to(device)
    center_t = torch.from_numpy(center.astype(np.float32)).to(device)
    confidence = torch.from_numpy(valid.astype(np.float32)).to(device)[:, None, None]
    with torch.no_grad():
        baseline = rig(prior)["pos"][..., :2]
        baseline = torch.stack((baseline[..., 0], -baseline[..., 1]), -1) * scale + center_t[:, None]
        reference = int(np.flatnonzero(valid)[np.argmin(direct[valid, rig.controls.index("jawOpen")])])
        offset = observed[reference] - baseline[reference]
    controls = torch.nn.Parameter(prior.clone())
    optimizer = torch.optim.Adam([controls], lr=.025)
    anchor_weight = torch.tensor([.5, 1.5, 1.5, .5, 1.5, 1.5, .3, .3], device=device)[None, :, None]
    for _ in range(steps):
        optimizer.zero_grad()
        posed = rig(controls)["pos"][..., :2]
        projected = torch.stack((posed[..., 0], -posed[..., 1]), -1) * scale + center_t[:, None] + offset
        reprojection = torch.nn.functional.smooth_l1_loss(projected, observed, beta=.01, reduction="none")
        loss = (reprojection * confidence * anchor_weight).mean()
        loss = loss + .06 * ((controls - prior) ** 2).mean()
        loss = loss + .02 * ((controls[1:] - controls[:-1]) ** 2).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            controls.clamp_(torch.from_numpy(lo).to(device), torch.from_numpy(hi).to(device))
    with torch.no_grad():
        result = rig(controls)["pos"][..., :2]
        result = torch.stack((result[..., 0], -result[..., 1]), -1) * scale + center_t[:, None] + offset
        error_before = float(torch.linalg.vector_norm((baseline + offset - observed)[valid], dim=-1).mean())
        error_after = float(torch.linalg.vector_norm((result - observed)[valid], dim=-1).mean())
    return controls.detach().cpu().numpy(), {"landmark_error_before_normalized": error_before,
                                        "landmark_error_after_normalized": error_after,
                                        "valid_frames": int(valid.sum()), "frames": len(valid),
                                        "projection_scale": scale}


def fit_video(rig_dir: Path, source: Path, out_dir: Path, model: Path = MODEL) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _extract(source, out_dir)
    direct, observed, valid = _observe(frames, model)
    fitted, report = fit_observations(rig_dir, direct, observed, valid)
    names = list(rigdef.CONTROLS)
    animation = {"format": "vhuman.performance.v1", "fps": FPS, "duration": len(frames) / FPS,
                 "controls": names, "frames": [{"t": i / FPS, "v": {name: round(float(v), 5)
                    for name, v in zip(names, row) if abs(v) > 1e-4}} for i, row in enumerate(fitted)],
                 "source": "face_video", "speech_strength": 1., "emotion_strength": 0.,
                 "secondary_strength": 0.}
    (out_dir / "animation.json").write_text(json.dumps(animation, separators=(",", ":")))
    report.update({"format": "vhuman.face_video_fit.v1", "fps": FPS, "duration": animation["duration"],
                   "tracker": "MediaPipe Face Landmarker", "optimizer": "vhuman TorchRig"})
    (out_dir / "fit_report.json").write_text(json.dumps(report, indent=2))
    shutil.rmtree(out_dir / "frames")
    return report


def fit_job(service, request: dict, progress, cancel, *, python=None, model: Path = MODEL) -> dict:
    from .job import DEFAULT_PYTHON

    head = request.get("head_id")
    rig_path = service.rig_file(head, "rig.json")
    source, upload_meta = source_file(service, request.get("upload_id"))
    py = Path(python) if python else DEFAULT_PYTHON
    if not py.is_file():
        raise ValueError(f"rig interpreter is missing: {py}")
    if not Path(model).is_file():
        raise ValueError(f"face landmarker model is missing: {model}; run setup_face_video.sh")
    take_id = uuid.uuid4().hex[:12]
    root = rig_path.parent / "takes"
    root.mkdir(parents=True, exist_ok=True)
    stage = root / (".partial-" + take_id)
    stage.mkdir()
    cmd = [str(py), "-m", "server.vhuman.rig.video_fit", "decode", str(rig_path.parent),
           str(source), str(stage), "--model", str(model)]
    progress(.02, "observing facial video")
    with gpu.device_session(1536, cancel) if gpu.backend() != "cpu" else nullcontext():
        proc = subprocess.Popen(runtime.python_command(cmd), cwd=Path(__file__).resolve().parents[3], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        try:
            while proc.poll() is None:
                if cancel.is_set():
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.communicate(timeout=5)
                    raise RuntimeError("face video fitting cancelled")
                import time
                time.sleep(.1)
            output = proc.communicate()[0]
            if proc.returncode:
                raise RuntimeError("face video fitting failed: " + output[-1500:])
            report = json.loads((stage / "fit_report.json").read_text())
            manifest = {"id": take_id, "head_id": head, "format": "vhuman.performance.v1",
                        "source": "face_video", "source_upload_id": upload_meta["id"],
                        "source_sha256": upload_meta["sha256"], "duration": report["duration"],
                        "fps": FPS, "frames": report["frames"],
                        "rig_sha256": hashlib.sha256(rig_path.read_bytes()).hexdigest()[:16]}
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
            stage.rename(root / take_id)
            progress(.99, "face video take ready")
            return service.take_summary(head, take_id)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("decode",))
    ap.add_argument("rig_dir", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL)
    args = ap.parse_args(argv)
    print(json.dumps(fit_video(args.rig_dir, args.source, args.out, args.model)))


if __name__ == "__main__":
    main()
