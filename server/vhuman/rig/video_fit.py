"""Optional face-video observations and PyTorch fitting of the vhuman rig.

Native repository inference of the pinned MediaPipe models supplies observations. The fitted controls are found by our
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
from ..face_assets import asset_path
MODEL = asset_path("mediapipe")
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


def _observe(paths: list[Path], model: Path, *, dense=False, threads=4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ..native_landmarks import FaceLandmarker
    from PIL import Image

    if not model.is_file():
        raise ValueError(f"face landmarker model is missing: {model}; run setup_face_video.sh")
    names = list(rigdef.CONTROLS)
    controls = np.zeros((len(paths), len(names)), np.float32)
    landmarks = np.zeros((len(paths), len(ANCHORS), 2), np.float32)
    valid = np.zeros(len(paths), bool)
    full = np.zeros((len(paths),478,3),np.float32)
    # Per-image inference is deterministic and independent of frame sampling.
    # Temporal regularization remains in fit_observations, below.
    with FaceLandmarker(model,threads=threads) as tracker:
        for i, path in enumerate(paths):
            with Image.open(path) as image:
                result = tracker.detect(np.asarray(image.convert('RGB')))
            if len(result) > 1:
                raise ValueError("face video must contain exactly one visible face")
            if not result:
                continue
            valid[i] = True
            points = result[0]['landmarks']
            full[i] = points
            for j, (_, index) in enumerate(ANCHORS):
                ids = list(index) if isinstance(index, tuple) else [index]
                landmarks[i, j] = points[ids, :2].mean(0)
            for name, score in result[0]['blendshapes'].items():
                if name in names:
                    controls[i, names.index(name)] = score
    if valid.mean() < .8:
        raise ValueError("face is missing in more than 20% of video frames")
    observed = np.flatnonzero(valid)
    for series in (controls, landmarks):
        for j in np.ndindex(series.shape[1:-1]) if series.ndim > 2 else [()]:
            for k in range(series.shape[-1]):
                column = series[(slice(None), *j, k)]
                series[(slice(None), *j, k)] = np.interp(np.arange(len(series)), observed, column[observed])
    return (controls, landmarks, valid, full) if dense else (controls, landmarks, valid)


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
                                        "backend": gpu.backend(), "device": device,
                                        "landmark_error_after_normalized": error_after,
                                        "valid_frames": int(valid.sum()), "frames": len(valid),
                                        "projection_scale": scale}


def _rig_receipt(rig_dir: Path) -> dict:
    """Identify fitting inputs, including bind sidecars and older source layouts."""
    names = ('rig.json', 'rig_deformer.safetensors', 'features.json', 'rig_report.json',
             'gnm_bind.json', 'gnm_bind.npz')
    if not (rig_dir/'gnm_bind.json').is_file():
        names += ('../geometry.npz', '../manifest.json', '../observations.json',
                  '../head.json', '../fit.json', '../head_eyes.glb', '../portrait.png')
    receipt = {}
    for name in names:
        path = rig_dir/name
        if name in ('rig.json', 'rig_deformer.safetensors') or path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(block)
            receipt[name] = digest.hexdigest()
        else:
            receipt[name] = None
    return receipt


def _verify_rig_receipt(rig_dir: Path, expected: dict):
    try:
        unchanged = _rig_receipt(rig_dir) == expected
    except OSError as exc:
        raise RuntimeError('rig changed during face video fitting; retry with the current rig') from exc
    if not unchanged:
        raise RuntimeError('rig changed during face video fitting; retry with the current rig')


def fit_video(rig_dir: Path, source: Path, out_dir: Path, model: Path = MODEL, *, iterations=400, threads=16, cancel=None) -> dict:
    rig_source = _rig_receipt(rig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _extract(source, out_dir)
    definition=json.loads((rig_dir/'rig.json').read_text())
    if definition.get('face_model') == 'gnm_v3':
        from ..reconstruction.gnm_motion import rig_bind,views_from_landmarks,fit_views,retarget,overlay
        from ..reconstruction.reference import Camera
        from ..face_parsing import FaceParser
        from PIL import Image
        bind=rig_bind(rig_dir)
        direct,observed,valid,dense=_observe(frames,model,dense=True,threads=threads)
        with Image.open(frames[0]) as image: size=image.size
        camera=Camera.from_dict(bind['metadata']['camera'])
        old_size=bind['metadata']['size'];ratio=min(size[0]/old_size[0],size[1]/old_size[1])
        camera=camera.scaled(ratio)
        camera.cx+=(size[0]-old_size[0]*ratio)/2;camera.cy+=(size[1]-old_size[1]*ratio)/2
        parser=FaceParser(threads=threads);labels=[]
        for path in frames:
            if cancel is not None and cancel.is_set(): raise gpu.Cancelled('cancelled')
            with Image.open(path) as image: labels.append(parser.predict(np.asarray(image.convert('RGB')))[0])
        views=views_from_landmarks(dense,valid,camera,size,labels=np.asarray(labels),synthetic=False)
        result=fit_views(bind,views,np.arange(len(frames))/FPS,out=out_dir,iterations=iterations,threads=threads,cancel=cancel)
        fitted,retarget_report=retarget(rig_dir,bind,result,direct,threads=threads,cancel=cancel)
        if not result['report']['quality_gate']:
            fitted=direct.copy()
            retarget_report.update(rms_mm=retarget_report['baseline_rms_mm'],p95_mm=None,
                                   fallback_frames=len(frames),fallback_reason='GNM refinement rejected')
        result['track'].update(clip=str(source.resolve()),clip_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source='real face video')
        (out_dir/'motion.json').write_text(json.dumps(result['track'],indent=2))
        overlay(out_dir,frames,result,fps=FPS,cancel=cancel)
        report=dict(result['track'],valid_frames=int(valid.sum()),retarget=retarget_report,optimizer='full GNM CPU Adam/L-BFGS')
    else:
        direct, observed, valid = _observe(frames, model)
        fitted, report = fit_observations(rig_dir, direct, observed, valid)
    names = [c["name"] for c in definition["controls"]]
    animation = {"format": "vhuman.performance.v1", "fps": FPS, "duration": len(frames) / FPS,
                 "controls": names, "frames": [{"t": i / FPS, "v": {name: round(float(v), 5)
                    for name, v in zip(names, row) if abs(v) > 1e-4}} for i, row in enumerate(fitted)],
                 "source": "face_video", "speech_strength": 1., "emotion_strength": 0.,
                 "secondary_strength": 0.}
    (out_dir / "animation.json").write_text(json.dumps(animation, separators=(",", ":")))
    report.update({"format": "vhuman.face_video_fit.v1", "fps": FPS, "duration": animation["duration"],
                   "tracker": "MediaPipe Face Landmarker", "optimizer": report.get("optimizer","vhuman TorchRig")})
    _verify_rig_receipt(rig_dir, rig_source)
    report['rig_source_sha256'] = rig_source
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
    iterations=request.get('iterations',400);threads=request.get('threads',16)
    if type(iterations) is not int or not 10<=iterations<=2000: raise ValueError('iterations must be 10..2000')
    if type(threads) is not int or not 1<=threads<=256: raise ValueError('threads must be 1..256')
    rig_source = _rig_receipt(rig_path.parent)
    take_id = uuid.uuid4().hex[:12]
    root = rig_path.parent / "takes"
    root.mkdir(parents=True, exist_ok=True)
    stage = root / (".partial-" + take_id)
    stage.mkdir()
    cmd = [str(py), "-m", "server.vhuman.rig.video_fit", "decode", str(rig_path.parent),
           str(source), str(stage), "--model", str(model),"--iterations",str(iterations),"--threads",str(threads)]
    progress(.02, "observing facial video")
    with gpu.device_session(1536, cancel) if gpu.backend() != "cpu" else nullcontext():
        from ..process import run
        tail=[]
        def on_line(line):
            tail.append(line)
            del tail[:-20]
        try:
            rc=run(runtime.python_command(cmd),cancel,on_line,
                   cwd=Path(__file__).resolve().parents[3],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            if rc: raise RuntimeError('face video fitting failed: '+' | '.join(tail[-8:]))
            report = json.loads((stage / "fit_report.json").read_text())
            if report.get('rig_source_sha256') != rig_source:
                raise RuntimeError('face video worker used a different rig; retry with the current rig')
            _verify_rig_receipt(rig_path.parent, rig_source)
            manifest = {"id": take_id, "head_id": head, "format": "vhuman.performance.v1",
                        "source": "face_video", "source_upload_id": upload_meta["id"],
                        "source_sha256": upload_meta["sha256"], "duration": report["duration"],
                        "fps": FPS, "frames": report["frames"],
                        "rig_sha256": rig_source['rig.json'][:16],
                        "quality_gate": report.get("quality_gate",True),"solver":report.get("solver","legacy"),
                        "retarget":report.get("retarget"),
                        "rig_package_sha256":rig_source['rig_deformer.safetensors'],
                        "rig_source_sha256": rig_source}
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
            stage.rename(root / take_id)
            progress(.99, "face video take ready")
            return service.take_summary(head, take_id)
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("decode",))
    ap.add_argument("rig_dir", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--iterations",type=int,default=400);ap.add_argument("--threads",type=int,default=16)
    args = ap.parse_args(argv)
    print(json.dumps(fit_video(args.rig_dir, args.source, args.out, args.model, iterations=args.iterations,threads=args.threads)))


if __name__ == "__main__":
    main()
