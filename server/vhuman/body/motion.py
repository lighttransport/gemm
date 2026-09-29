"""Fit an uploaded image or short video to an existing MHR avatar.

SAM 3D Body estimates each sampled frame independently. The MHR decoder then
evaluates those poses with the avatar's fixed identity, and only local joint
rotations are retargeted. Root translation stays at the avatar bind position:
monocular camera translation is not a reliable world-space motion track.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .. import gpu
from ..eye.glb import FLOAT, GLB, GLBBuilder
from ..service import ROOT, ServiceError
from . import job as body_job

IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
VIDEO_TYPES = {"video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov"}
MAX_UPLOAD = 64 << 20
MAX_SECONDS = 8
MAX_FRAMES = 32
FPS = 4
MOTION_FILES = {"motion.json", "motion.glb", "manifest.json"}


def upload(service, stream, length: int, content_type: str) -> dict:
    ctype = content_type.split(";", 1)[0].strip().lower()
    ext = (IMAGE_TYPES | VIDEO_TYPES).get(ctype)
    if ext is None:
        raise ServiceError("upload must be PNG, JPEG, WebP, MP4, WebM or MOV")
    if not 0 < length <= MAX_UPLOAD:
        raise ServiceError(f"upload must be between 1 byte and {MAX_UPLOAD >> 20} MiB")
    uid = uuid.uuid4().hex[:16]
    folder = service.work / "body_uploads" / uid
    folder.mkdir(parents=True)
    source = folder / ("source" + ext)
    try:
        digest = hashlib.sha256()
        with source.open("wb") as dst:
            remaining = length
            while remaining:
                block = stream.read(min(1 << 20, remaining))
                if not block:
                    raise ServiceError("upload ended before Content-Length")
                dst.write(block)
                digest.update(block)
                remaining -= len(block)
        kind = "image" if ctype in IMAGE_TYPES else "video"
        if kind == "image":
            try:
                with Image.open(source) as im:
                    w, h = im.size
                    if max(w, h) > 8192 or w * h > 32_000_000:
                        raise ServiceError("image dimensions are too large")
                    im.verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise ServiceError("image is not decodable") from exc
        else:
            with source.open("rb") as stream:
                sig = stream.read(32)
            if not (sig[4:8] == b"ftyp" or sig.startswith(b"\x1a\x45\xdf\xa3")):
                raise ServiceError("video container signature is invalid")
        meta = {"id": uid, "kind": kind, "mime": ctype, "bytes": length,
                "sha256": digest.hexdigest(), "file": source.name}
        (folder / "upload.json").write_text(json.dumps(meta))
        return {k: meta[k] for k in ("id", "kind", "bytes")}
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def source_file(service, upload_id: str) -> tuple[Path, dict]:
    if not isinstance(upload_id, str) or len(upload_id) != 16 or any(c not in "0123456789abcdef" for c in upload_id):
        raise ServiceError("invalid upload id")
    folder = service.work / "body_uploads" / upload_id
    meta_path = folder / "upload.json"
    if not meta_path.is_file():
        raise ServiceError("no such upload")
    meta = json.loads(meta_path.read_text())
    source = folder / meta["file"]
    if not source.is_file():
        raise ServiceError("no such upload")
    return source, meta


def _frames(source: Path, kind: str, out: Path, cancel) -> list[Path]:
    frames = out / "frames"
    frames.mkdir()
    if kind == "image":
        with Image.open(source) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            path = frames / "00000.png"
            im.save(path)
        return [path]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is needed to sample uploaded video")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "2",
           "-i", str(source), "-t", str(MAX_SECONDS), "-vf",
           f"fps={FPS},scale=1280:1280:force_original_aspect_ratio=decrease",
           "-frames:v", str(MAX_FRAMES), "-an", str(frames / "%05d.png")]
    body_job._run(cmd, cancel, timeout=180)
    paths = sorted(frames.glob("*.png"))
    if not paths:
        raise ValueError("video contains no decodable frames")
    return paths


def _bbox(image: Path) -> tuple[int, int, int, int]:
    with Image.open(image) as im:
        rgba = np.asarray(im.convert("RGBA"))
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3] > 40
    if alpha.mean() < .98 and alpha.mean() > .03:
        ys, xs = np.nonzero(alpha)
        return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    # HOG is a lightweight optional hint for an opaque single-person photo.
    # If it misses, SAM receives the full image, which works for centered shots.
    try:
        import cv2
        rgb = cv2.imread(str(image))
        if rgb is not None:
            scale = min(1.0, 640 / max(w, h))
            small = cv2.resize(rgb, None, fx=scale, fy=scale) if scale < 1 else rgb
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            boxes, scores = hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.07)
            if len(boxes):
                i = int(np.argmax([float(s) * bw * bh for (_, _, bw, bh), s in zip(boxes, scores)]))
                x, y, bw, bh = boxes[i]
                pad = .08 * max(bw, bh)
                return (max(0, int((x - pad) / scale)), max(0, int((y - pad) / scale)),
                        min(w, int((x + bw + pad) / scale)), min(h, int((y + bh + pad) / scale)))
    except (ImportError, OSError):
        pass
    return 0, 0, w, h


def _sam_frame(image: Path, out: Path, bbox: tuple[int, ...], model_dir: Path, cancel, mock: bool) -> Path:
    sidecar = out.with_suffix(out.suffix + ".json")
    if mock:
        meta = {"model_params": [0.] * 204, "shape": [0.] * 45, "bbox": bbox}
        sidecar.write_text(json.dumps(meta))
        return sidecar
    use_cuda = gpu.gpu_status() is not None
    binary = body_job._binary("sam3d_body", use_cuda, cancel)
    cmd = [str(binary), "--safetensors-dir", str(model_dir / "safetensors"),
           "--mhr-assets", str(model_dir / "safetensors"), "--image", str(image),
           "--bbox", *map(str, bbox), "--backbone", "dinov3", "-o", str(out)]
    if use_cuda:
        with gpu.device_session(2048, cancel):
            body_job._run(cmd, cancel)
    else:
        body_job._run(cmd, cancel)
    return sidecar


def _decode(model_path: Path, identity: np.ndarray, sidecars: list[Path]) -> tuple[list[str], np.ndarray]:
    import torch
    model = torch.jit.load(str(model_path), map_location="cpu")
    poses = []
    for path in sidecars:
        p = np.asarray(json.loads(path.read_text())["model_params"], np.float32)
        if p.shape != (204,) or not np.isfinite(p).all():
            raise ValueError(f"invalid MHR pose in {path.name}")
        poses.append(p)
    params = np.stack(poses)
    names = ["mhr_" + n for n in model.get_joint_names()]
    parents = model.character_torch.skeleton.joint_parents.cpu().numpy().astype(int)
    tracks = []
    with torch.no_grad():
        for start in range(0, len(params), 4):
            batch = params[start:start + 4]
            _, state = model(torch.from_numpy(np.repeat(identity[None], len(batch), axis=0)),
                             torch.from_numpy(batch), torch.zeros((len(batch), 72)))
            global_q = state.numpy()[:, :, 3:7].astype(np.float64)
            global_q /= np.maximum(np.linalg.norm(global_q, axis=2, keepdims=True), 1e-9)
            local = global_q.copy()
            from scipy.spatial.transform import Rotation
            for j, parent in enumerate(parents):
                if parent >= 0:
                    local[:, j] = (Rotation.from_quat(global_q[:, parent]).inv() *
                                   Rotation.from_quat(global_q[:, j])).as_quat()
            tracks.append(local)
    return names, np.concatenate(tracks)


def _continuous(rotations: np.ndarray) -> np.ndarray:
    out = rotations.copy()
    for i in range(1, len(out)):
        sign = np.sum(out[i - 1] * out[i], axis=-1) < 0
        out[i, sign] *= -1
        # Mild temporal smoothing reduces independent-frame SAM jitter.
        if len(out) > 2:
            out[i] = .7 * out[i] + .3 * out[i - 1]
            out[i] /= np.linalg.norm(out[i], axis=-1, keepdims=True)
    return out


def _export_glb(avatar: Path, out: Path, names: list[str], rotations: np.ndarray, times: np.ndarray) -> None:
    source = GLB.load(avatar)
    b = GLBBuilder("vhuman uploaded body motion")
    b.doc = source.doc
    b.bin = bytearray(source.bin)
    b.extensions = set(b.doc.get("extensionsUsed", []))
    for key in ("images", "textures", "samplers"):
        b.doc.setdefault(key, [])
    nodes = {n.get("name"): i for i, n in enumerate(b.doc["nodes"])}
    for j, name in enumerate(names):
        if name not in nodes:
            raise ValueError(f"avatar is missing MHR joint {name}")
        b.doc["nodes"][nodes[name]]["rotation"] = rotations[0, j].tolist()
    if len(times) > 1:
        t_view = b._view(times.astype("<f4").tobytes())
        t_acc = len(b.doc["accessors"])
        b.doc["accessors"].append({"bufferView": t_view, "componentType": FLOAT, "count": len(times),
                                   "type": "SCALAR", "min": [float(times[0])], "max": [float(times[-1])]})
        animation = {"name": "uploaded_body_motion", "samplers": [], "channels": []}
        for j, name in enumerate(names):
            view = b._view(rotations[:, j].astype("<f4").tobytes())
            acc = len(b.doc["accessors"])
            b.doc["accessors"].append({"bufferView": view, "componentType": FLOAT,
                                       "count": len(times), "type": "VEC4"})
            sampler = len(animation["samplers"])
            animation["samplers"].append({"input": t_acc, "output": acc, "interpolation": "LINEAR"})
            animation["channels"].append({"sampler": sampler,
                                          "target": {"node": nodes[name], "path": "rotation"}})
        b.doc.setdefault("animations", []).append(animation)
    b.write(out)


def fit(service, request: dict, progress, cancel, *, model_dir=body_job.MODEL_DIR,
        rig_python=body_job.DEFAULT_RIG_PYTHON, mock=False) -> dict:
    hid = request.get("head_id")
    avatar = service.body_file(hid, "avatar.glb")
    avatar_json = service.body_file(hid, "avatar.json")
    body_meta = service.body_file(hid, "body_mhr.glb.json")
    upload_id = request.get("upload_id")
    source, upload_meta = source_file(service, upload_id)
    identity = np.asarray(json.loads(body_meta.read_text())["shape"], np.float32)
    if identity.shape != (45,):
        raise ValueError("avatar lacks its MHR identity parameters")
    model_dir = Path(model_dir)
    rig_python = Path(rig_python)
    if not rig_python.is_file() or not (model_dir / "dinov3/assets/mhr_model.pt").is_file():
        raise ValueError("MHR model or rig Python is missing")
    take_id = uuid.uuid4().hex[:12]
    root = avatar.parent / "motions"
    stage = root / ("." + take_id)
    stage.mkdir(parents=True)
    try:
        progress(.02, "sampling uploaded media")
        frames = _frames(source, upload_meta["kind"], stage, cancel)
        sidecars = []
        for i, frame in enumerate(frames):
            if cancel.is_set():
                raise gpu.Cancelled("cancelled")
            progress(.08 + .72 * i / len(frames), f"SAM 3D Body pose {i+1}/{len(frames)}")
            sidecars.append(_sam_frame(frame, stage / f"pose_{i:05d}.glb", _bbox(frame), model_dir,
                                       cancel, mock))
        progress(.82, "retargeting MHR joints to avatar")
        output = stage / "motion.json"
        cmd = [str(rig_python), "-m", "server.vhuman.body.motion", "decode",
               "--avatar-json", str(avatar_json), "--identity", str(body_meta),
               "--model", str(model_dir / "dinov3/assets/mhr_model.pt"),
               "--source-glb", str(avatar), "--output", str(output),
               "--kind", upload_meta["kind"], "--fps", str(FPS),
               *map(str, sidecars)]
        body_job._run(cmd, cancel, timeout=300)
        motion = json.loads(output.read_text())
        manifest = {"id": take_id, "head_id": hid, "kind": upload_meta["kind"],
                    "source_upload_id": upload_id, "source_sha256": upload_meta["sha256"],
                    "frames": len(frames), "fps": FPS if len(frames) > 1 else 0,
                    "duration": motion["duration"],
                    "avatar_sha256": hashlib.sha256(avatar_json.read_bytes()).hexdigest()[:16]}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=1))
        for path in stage.iterdir():
            if path.name not in MOTION_FILES:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        final = root / take_id
        stage.rename(final)
        progress(.99, "body motion ready")
        base = f"/v1/heads/{hid}/body/motions/{take_id}/"
        return {**manifest, "motion_url": base + "motion.json", "glb_url": base + "motion.glb"}
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def decode_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("decode",))
    ap.add_argument("--avatar-json", type=Path, required=True)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--source-glb", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--kind", choices=("image", "video"), required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("sidecars", nargs="+")
    a = ap.parse_args(argv)
    definition = json.loads(a.avatar_json.read_text())
    identity = np.asarray(json.loads(a.identity.read_text())["shape"], np.float32)
    names, rotations = _decode(a.model, identity, list(map(Path, a.sidecars)))
    avatar_names = {j["name"] for j in definition["joints"]}
    if not set(names) <= avatar_names:
        raise ValueError("avatar MHR skeleton differs from decoder")
    rotations = _continuous(rotations)
    times = np.arange(len(rotations), dtype=np.float32) / a.fps
    result = {"version": 1, "kind": a.kind, "fps": a.fps if len(times) > 1 else 0,
              "duration": float(times[-1]), "joints": names,
              "frames": [{"t": float(t), "rotations": r.tolist()} for t, r in zip(times, rotations)],
              "root_translation": "avatar_bind", "space": "MHR local xyzw"}
    a.output.write_text(json.dumps(result, separators=(",", ":")))
    _export_glb(a.source_glb, a.output.with_suffix(".glb"), names, rotations, times)
    print(json.dumps({"frames": len(times), "joints": len(names), "duration": result["duration"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(decode_main())
