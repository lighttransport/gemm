#!/usr/bin/env python3
"""Small Python web API and demo for the native Pixal3D runner.

The server deliberately keeps inference in the project native executable.  This
avoids importing a second copy of the large model into the web process and
allows the same endpoint to serve CPU, CUDA, and ROCm hosts.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import copy
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path("/mnt/disk2/models/Pixal3D")
DEFAULT_DINOV3 = Path("/mnt/disk2/models/dinov3-vitl16/model.safetensors")
DEFAULT_NAF = ROOT / "ref/pixal3d/weights/naf_release.safetensors"
DEFAULT_RMBG = Path("/mnt/disk2/models/RMBG-2.0")
DEFAULT_MOGE = Path("/mnt/disk2/models/moge-2-vitl/model.pt")
MAX_BODY_BYTES = 256 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_GLB_BYTES = 256 * 1024 * 1024


class JobCancelled(Exception):
    pass


class QueueFull(Exception):
    pass


def error_payload(code: str, message: str) -> dict:
    return {"ok": False, "error": message, "error_code": code}


def reference_request(request: dict, native_result: dict) -> dict:
    """Carry resolved automatic camera values into the comparison run."""
    resolved = copy.deepcopy(request)
    preparation = native_result.get("preparation")
    if preparation and "fov" in preparation:
        resolved["fov"] = preparation["fov"]
        resolved["distance"] = preparation["distance"]
        resolved["auto_camera"] = False
    return resolved


class UploadStore:
    """Own raw request images until a queued job consumes them."""
    def __init__(self, root: Path, retained: int = 64, ttl: float = 3600):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.retained = retained
        self.ttl = ttl
        self.files: dict[str, Path] = {}
        self.lock = threading.Lock()
        # Upload IDs are process-local capabilities. Files left by a previous
        # process can never be claimed and should not consume the new quota.
        for path in self.root.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)

    def _expire_locked(self, now: float | None = None) -> None:
        cutoff = (time.time() if now is None else now) - self.ttl
        for upload_id, path in list(self.files.items()):
            try:
                expired = path.stat().st_mtime < cutoff
            except FileNotFoundError:
                expired = True
            if expired:
                path.unlink(missing_ok=True)
                self.files.pop(upload_id, None)

    def put(self, data: bytes) -> str:
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"upload exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit")
        upload_id = uuid.uuid4().hex
        path = self.root / upload_id
        with self.lock:
            self._expire_locked()
            if len(self.files) >= self.retained:
                raise QueueFull("upload store is full")
            path.write_bytes(data)
            self.files[upload_id] = path
        return upload_id

    def claim(self, request: dict) -> tuple[dict, list[Path]]:
        resolved = copy.deepcopy(request)
        refs: list[tuple[dict, str, str]] = []
        for source, destination in (("image_upload", "image_b64"),
                                    ("mask_upload", "mask_b64")):
            if source in resolved:
                refs.append((resolved, source, destination))
        if isinstance(resolved.get("views"), list):
            for view in resolved["views"]:
                if not isinstance(view, dict):
                    continue
                for source, destination in (("image_upload", "image_b64"),
                                            ("mask_upload", "mask_b64")):
                    if source in view:
                        refs.append((view, source, destination))
        ids = [container[source] for container, source, _ in refs]
        if len(ids) != len(set(ids)):
            raise ValueError("an upload ID may only be used once")
        with self.lock:
            self._expire_locked()
            if any(not isinstance(item, str) or item not in self.files for item in ids):
                raise ValueError("unknown or expired upload ID")
            paths = [self.files.pop(item) for item in ids]
        for (container, source, destination), path in zip(refs, paths):
            container[destination] = path
            del container[source]
        return resolved, paths

    def delete(self, upload_id: str) -> bool:
        with self.lock:
            self._expire_locked()
            path = self.files.pop(upload_id, None)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        return True


def run_command(command: list[str], timeout: float, cancel: threading.Event | None = None,
                progress=None) -> subprocess.CompletedProcess:
    if cancel is None and progress is None:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def drain(stream, lines, report=False):
        for line in iter(stream.readline, ""):
            lines.append(line)
            if report and progress is not None:
                progress(line.rstrip())

    readers = [threading.Thread(target=drain, args=(process.stdout, stdout_lines), daemon=True),
               threading.Thread(target=drain, args=(process.stderr, stderr_lines, True), daemon=True)]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    cancelled = False
    timed_out = False
    while process.poll() is None:
        if cancel is not None and cancel.is_set():
            cancelled = True
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            break
        if time.monotonic() >= deadline:
            timed_out = True
            process.kill()
            process.wait()
            break
        time.sleep(0.1)
    for reader in readers:
        reader.join(timeout=5)
    process.stdout.close()
    process.stderr.close()
    if cancelled:
        raise JobCancelled("job cancelled")
    if timed_out:
        raise subprocess.TimeoutExpired(command, timeout,
                                        output="".join(stdout_lines), stderr="".join(stderr_lines))
    return subprocess.CompletedProcess(command, process.returncode,
                                       "".join(stdout_lines), "".join(stderr_lines))


def native_progress(line: str) -> tuple[str, int] | None:
    """Translate stable native stderr messages into monotonic UI milestones."""
    match = re.search(r"Pixal3D (structure|shape512|shape1024|texture): step (\d+)/(\d+)", line)
    if match:
        stage, step, total = match.group(1), int(match.group(2)), int(match.group(3))
        start, span = {"structure": (3, 17), "shape512": (23, 17),
                       "shape1024": (45, 17), "texture": (65, 17)}[stage]
        return f"{stage} diffusion {step}/{total}", start + span * step // total
    match = re.search(r"Pixal3D (structure|shape512|shape1024|texture): conditioning", line)
    if match:
        stage = match.group(1)
        return f"{stage} conditioning", {"structure": 2, "shape512": 22,
                                          "shape1024": 44, "texture": 64}[stage]
    if "Pixal3D FDG:" in line:
        return "mesh extraction", 84
    if "Pixal3D simplify:" in line:
        return "mesh simplification", 89
    if "Pixal3D bake:" in line:
        return "PBR texture baking", 94
    if "Pixal3D inpaint:" in line:
        return "texture inpainting", 97
    return None


def decode_b64(value: object, name: str, limit: int) -> bytes:
    if isinstance(value, Path):
        if not value.is_file() or value.stat().st_size <= 0 or value.stat().st_size > limit:
            raise ValueError(f"{name} exceeds the {limit // (1024 * 1024)} MiB limit")
        return value.read_bytes()
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty base64 string")
    if value.startswith("data:"):
        value = value.split(",", 1)[-1]
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{name} is not valid base64") from exc
    if not data or len(data) > limit:
        raise ValueError(f"{name} exceeds the {limit // (1024 * 1024)} MiB limit")
    return data


def finite_number(value: object, name: str, lo: float, hi: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(out) or out < lo or out > hi:
        raise ValueError(f"{name} must be finite and in [{lo}, {hi}]")
    return out


def bounded_integer(value: object, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    out = int(number)
    if out < lo or out > hi:
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    return out


def boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def camera_matrix(value: object, name: str) -> list[list[float]]:
    if not (isinstance(value, list) and len(value) == 4 and
            all(isinstance(row, list) and len(row) == 4 for row in value)):
        raise ValueError(f"{name} must be a 4x4 array")
    return [[finite_number(cell, f"{name}[{r}][{c}]", -1e6, 1e6)
             for c, cell in enumerate(row)] for r, row in enumerate(value)]


def model_ready(model_dir: Path, dino: Path, naf: Path) -> bool:
    # pipeline.json names the remaining checkpoints; checking it plus the two
    # explicit files catches the common setup errors without loading weights.
    if not (model_dir / "pipeline.json").is_file() or not dino.is_file() or not naf.is_file():
        return False
    try:
        cfg = json.loads((model_dir / "pipeline.json").read_text())
    except (OSError, ValueError):
        return False
    for item in cfg.get("checkpoints", []):
        if isinstance(item, str) and not (model_dir / item).is_file():
            return False
    return True


def rmbg_ready(path: Path) -> bool:
    return ((path / "config.json").is_file() and
            (any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))))


def reference_environment_ready(backend: str) -> tuple[bool, list[str]]:
    environment = (ROOT / "ref/pixal3d/.venv-reference-cuda310" if backend == "cuda"
                   else ROOT / f"ref/pixal3d/.venv-{backend}")
    missing = []
    if not (environment / "bin/python").is_file():
        missing.append("python")
    if backend == "cuda" and not (environment / ".pixal3d-reference-ready").is_file():
        missing.append("validated_environment")
    site_packages = list((environment / "lib").glob("python*/site-packages"))
    if not site_packages or not any((site / "o_voxel").is_dir() or
                                    any(site.glob("o_voxel*.so"))
                                    for site in site_packages):
        missing.append("o_voxel")
    if backend == "cuda" and not any(
            (site / "cumesh").is_dir() or any(site.glob("cumesh*.so"))
            for site in site_packages):
            missing.append("cumesh")
    return not missing, missing


def valid_output(path: Path, limit: int = MAX_GLB_BYTES) -> bool:
    return path.is_file() and 0 < path.stat().st_size <= limit


def publish_artifact(partial: Path, final: Path, limit: int = MAX_GLB_BYTES) -> Path:
    """Validate and atomically publish a runner output in retained job storage."""
    if not valid_output(partial, limit):
        raise RuntimeError(f"runner did not produce a valid {final.suffix.lstrip('.').upper()}")
    with partial.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(partial, final)
    descriptor = os.open(final.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return final


def retained_artifact_dir(request: dict) -> Path | None:
    value = request.get("_artifact_dir")
    return value if isinstance(value, Path) else None


def retain_reference_input(request: dict, name: str, source: Path) -> Path | None:
    """Keep prepared RGBA alive between native and queued reference stages."""
    directory = retained_artifact_dir(request)
    if directory is None:
        return None
    inputs = directory / ".reference-inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    target = inputs / name
    temporary = inputs / f".{name}.{uuid.uuid4().hex}.partial"
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)
    return target


def glb_mesh_summary(path: Path) -> dict:
    """Read comparison-scale mesh facts without decoding textures."""
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", raw)
    if magic != 0x46546c67 or version != 2 or total != len(raw):
        raise ValueError("invalid GLB header")
    json_size, json_kind = struct.unpack_from("<II", raw, 12)
    if json_kind != 0x4e4f534a:
        raise ValueError("missing GLB JSON chunk")
    scene = json.loads(raw[20:20 + json_size])
    primitive = scene["meshes"][0]["primitives"][0]
    position = scene["accessors"][primitive["attributes"]["POSITION"]]
    indices = scene["accessors"][primitive["indices"]]
    bounds = [position.get("min"), position.get("max")]
    return {"bytes": len(raw), "vertices": position["count"],
            "triangles": indices["count"] // 3, "bounds": bounds}


def mesh_comparison(native: dict, reference: dict) -> dict:
    result = {"native": native, "reference": reference}
    if native.get("bounds") and reference.get("bounds"):
        result["bounds_max_abs_delta"] = max(
            abs(float(a) - float(b))
            for side_a, side_b in zip(native["bounds"], reference["bounds"])
            for a, b in zip(side_a, side_b))
    for name in ("vertices", "triangles"):
        denominator = max(1, int(reference[name]))
        result[f"{name}_relative_delta"] = (int(native[name]) - int(reference[name])) / denominator
    return result


def attach_comparison(pixal, result: dict, cancel: threading.Event | None = None) -> None:
    """Attach cheap structural and optional bounded surface diagnostics."""
    native_mesh = result.get("mesh_summary", {})
    reference = result.get("reference", {})
    reference_mesh = reference.get("mesh_summary", {})
    if not (native_mesh.get("vertices") and reference_mesh.get("vertices")):
        return
    comparison = mesh_comparison(native_mesh, reference_mesh)
    if hasattr(pixal, "surface_comparison"):
        try:
            comparison["surface"] = pixal.surface_comparison(result, reference, cancel)
        except JobCancelled:
            raise
        except Exception as exc:
            comparison["surface"] = {"available": False, "error": str(exc)}
    result["comparison"] = comparison


class PixalServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.binary = Path(args.binary).resolve()
        self.model_dir = Path(args.model_dir).resolve()
        self.dinov3 = Path(args.dinov3).resolve()
        self.naf = Path(args.naf).resolve()
        self.rembg = Path(getattr(args, "rembg", DEFAULT_RMBG)).resolve()
        self.moge = Path(getattr(args, "moge", DEFAULT_MOGE)).resolve()
        self.work_dir = Path(args.work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.locks = {backend: threading.Lock() for backend in ("cpu", "cuda", "rocm")}
        self.reference_script = ROOT / "ref/pixal3d/run_reference_sv.py"
        self.reference_mv_script = ROOT / "ref/pixal3d/run_reference_mv.py"
        self.reference_mv_upstream = ROOT / "ref/pixal3d/upstream/inference_mv.py"
        self.python_launcher = ROOT / "ref/pixal3d/run.sh"
        self.reference_launcher = ROOT / "ref/pixal3d/run_reference_cuda310.sh"
        self.prepare_script = ROOT / "ref/pixal3d/prepare_input.py"
        self.compare_script = ROOT / "ref/pixal3d/compare_outputs.py"

    def health(self) -> dict:
        lib = {
            "cuda": ROOT / "cuda/pixal3d/libpixal3d_cuda.so",
            "rocm": ROOT / "rdna4/pixal3d/libpixal3d_rocm.so",
        }
        out = {}
        for backend in ("cpu", "cuda", "rocm"):
            out[backend] = {
                "available": self.binary.is_file() and (backend == "cpu" or lib[backend].is_file()),
                "binary": str(self.binary),
                "gpu_library": str(lib[backend]) if backend != "cpu" else None,
                "models_ready": model_ready(self.model_dir, self.dinov3, self.naf),
                "multiview_ready": (self.model_dir / "pipeline_mv.json").is_file(),
            }
        reference = {}
        for backend in ("cuda", "rocm"):
            environment = (ROOT / "ref/pixal3d/.venv-reference-cuda310/bin/python"
                           if backend == "cuda"
                           else ROOT / f"ref/pixal3d/.venv-{backend}/bin/python")
            environment_ready, missing = reference_environment_ready(backend)
            reference[backend] = {
                "available": (environment_ready and self.reference_script.is_file() and
                              self.reference_mv_script.is_file() and
                              self.reference_mv_upstream.is_file() and model_ready(
                                  self.model_dir, self.dinov3, self.naf)),
                "environment": str(environment),
                "single_view_source": self.reference_script.is_file(),
                "multiview_source": self.reference_mv_script.is_file(),
                "multiview_upstream_source": self.reference_mv_upstream.is_file(),
                "missing_dependencies": missing,
            }
        return {"ok": True, "service": "pixal3d", "default_backend": self.args.backend,
                "default_gpu_execution": self.args.gpu_execution,
                "default_gpu_kernels": self.args.gpu_kernels,
                "default_gpu_flow_precision": self.args.gpu_flow_precision,
                "preparation": {"mask_ready": rmbg_ready(self.rembg),
                                "camera_ready": self.moge.is_file()},
                "reference": reference,
                "limits": {"body_bytes": MAX_BODY_BYTES, "image_bytes": MAX_IMAGE_BYTES,
                           "glb_bytes": MAX_GLB_BYTES, "views": 16}, "backends": out}

    def infer(self, request: dict, cancel: threading.Event | None = None, progress=None) -> dict:
        backend = request.get("backend", self.args.backend)
        if backend not in ("cpu", "cuda", "rocm"):
            raise ValueError("backend must be cpu, cuda, or rocm")
        multiview = request.get("views") is not None
        views = request.get("views") if multiview else None
        if multiview and (not isinstance(views, list) or not 1 <= len(views) <= 16):
            raise ValueError("views must contain 1 to 16 posed images")
        auto_mask = boolean(request.get("auto_mask", False), "auto_mask")
        auto_camera = boolean(request.get("auto_camera", False), "auto_camera")
        if multiview and auto_camera:
            raise ValueError("auto_camera is only available for single-view inference")
        if auto_camera and not self.moge.is_file():
            raise ValueError(f"automatic camera estimation is unavailable; missing MoGe model: {self.moge}")
        image = None if multiview else decode_b64(request.get("image_b64"), "image_b64", MAX_IMAGE_BYTES)
        mask = None if multiview else (decode_b64(request["mask_b64"], "mask_b64", MAX_IMAGE_BYTES) if request.get("mask_b64") else None)
        ext = str(request.get("image_ext", ".png")).lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        fov = finite_number(request.get("fov", 0.857556), "fov", 0.05, 3.14)
        distance = finite_number(request.get("distance", 0.0), "distance", 0.0, 1000.0)
        mesh_scale = finite_number(request.get("mesh_scale", 1.0), "mesh_scale", 1e-5, 1000.0)
        seed = bounded_integer(request.get("seed", 42), "seed", 0, 2**32 - 1)
        threads = bounded_integer(request.get("threads", self.args.threads), "threads", 0, 1024)
        texture_size = bounded_integer(request.get("texture_size", 4096), "texture_size", 1024, 4096)
        if texture_size not in (1024, 2048, 4096):
            raise ValueError("texture_size must be 1024, 2048, or 4096")
        triangle_target = bounded_integer(request.get("triangle_target", 1000000),
                                          "triangle_target", 10000, 5000000)
        include_ply = request.get("include_ply", False)
        if not isinstance(include_ply, bool):
            raise ValueError("include_ply must be a boolean")
        with self.locks[backend], tempfile.TemporaryDirectory(prefix="request-", dir=self.work_dir) as td:
            run_dir = Path(td)
            artifact_dir = retained_artifact_dir(request)
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True)
            output_path = ((artifact_dir / ".native.glb.partial") if artifact_dir
                           else run_dir / "output.glb")
            ply_path = ((artifact_dir / ".native.ply.partial") if artifact_dir
                        else run_dir / "output.ply")
            preparation = None
            mask_path = None
            if multiview:
                frames = []
                view_preparation = []
                for index, item in enumerate(views):
                    if not isinstance(item, dict):
                        raise ValueError("each view must be an object")
                    data = decode_b64(item.get("image_b64"), f"views[{index}].image_b64", MAX_IMAGE_BYTES)
                    name = f"view{index:02d}.png"
                    view_path = run_dir / name
                    view_path.write_bytes(data)
                    view_mask = (decode_b64(item["mask_b64"],
                                            f"views[{index}].mask_b64", MAX_IMAGE_BYTES)
                                 if item.get("mask_b64") else None)
                    view_mask_path = None
                    if view_mask is not None:
                        view_mask_path = run_dir / f"view{index:02d}-mask.png"
                        view_mask_path.write_bytes(view_mask)
                    matrix = camera_matrix(item.get("transform_matrix"), f"views[{index}].transform_matrix")
                    frame = {"file_path": name, "transform_matrix": matrix}
                    if item.get("fov") is not None:
                        frame["camera_angle_x"] = finite_number(item["fov"], f"views[{index}].fov", 0.05, 3.14)
                    if auto_mask or view_mask_path is not None:
                        prepared = run_dir / f"view{index:02d}-prepared.png"
                        metadata = run_dir / f"view{index:02d}-prepared.json"
                        prep = [str(self.python_launcher), backend, str(self.prepare_script),
                                "--input", str(view_path), "--output", str(prepared),
                                "--metadata", str(metadata),
                                "--fov", str(frame.get("camera_angle_x", fov)),
                                "--mesh-scale", str(mesh_scale), "--device",
                                "cpu" if backend == "cpu" else "cuda"]
                        if view_mask_path is not None:
                            prep += ["--mask", str(view_mask_path)]
                        elif auto_mask:
                            prep += ["--rembg-model", str(self.rembg)]
                        proc = run_command(prep, self.args.reference_timeout, cancel)
                        if proc.returncode:
                            raise RuntimeError((proc.stderr or proc.stdout).strip()[-4000:])
                        frame["file_path"] = prepared.name
                        item_preparation = json.loads(metadata.read_text())
                        item_preparation["view"] = index
                        view_preparation.append(item_preparation)
                        retained = retain_reference_input(
                            request, f"view{index:02d}.png", prepared)
                        if retained is not None:
                            item["_reference_image_path"] = retained
                        elif request.get("reference"):
                            item["_reference_image_b64"] = base64.b64encode(
                                prepared.read_bytes()).decode("ascii")
                    frames.append(frame)
                if view_preparation:
                    preparation = {"views": view_preparation}
                (run_dir / "transforms.json").write_text(json.dumps({"camera_angle_x": fov, "mesh_scale": mesh_scale, "frames": frames}))
                cmd = [str(self.binary), "--backend", backend, "--views-dir", str(run_dir), "--output", str(output_path),
                       "--seed", str(seed), "--model-dir", str(self.model_dir), "--dinov3", str(self.dinov3), "--naf", str(self.naf)]
            else:
                image_path = run_dir / ("input" + ext)
                image_path.write_bytes(image)
                if mask is not None:
                    mask_path = run_dir / "mask.png"
                    mask_path.write_bytes(mask)
                if auto_mask or auto_camera or (mask_path is not None and request.get("reference")):
                    prepared = run_dir / "prepared.png"
                    metadata = run_dir / "prepared.json"
                    prep = [str(self.python_launcher), backend, str(self.prepare_script),
                            "--input", str(image_path), "--output", str(prepared),
                            "--metadata", str(metadata), "--mesh-scale", str(mesh_scale),
                            "--device", "cpu" if backend == "cpu" else "cuda"]
                    if mask_path:
                        prep += ["--mask", str(mask_path)]
                    if auto_mask:
                        prep += ["--rembg-model", str(self.rembg)]
                    if auto_camera:
                        prep += ["--moge-model", str(self.moge)]
                    else:
                        prep += ["--fov", str(fov)]
                    proc = run_command(prep, self.args.reference_timeout, cancel)
                    if proc.returncode:
                        raise RuntimeError((proc.stderr or proc.stdout).strip()[-4000:])
                    preparation = json.loads(metadata.read_text())
                    image_path = prepared
                    mask_path = None
                    fov, distance = preparation["fov"], preparation["distance"]
                    if request.get("reference"):
                        retained = retain_reference_input(request, "single.png", prepared)
                        if retained is not None:
                            request["_reference_image_path"] = retained
                        else:
                            request["_reference_image_b64"] = base64.b64encode(
                                prepared.read_bytes()).decode("ascii")
                        request["_reference_image_ext"] = ".png"
                cmd = [str(self.binary), "--backend", backend, "--input", str(image_path), "--output", str(output_path),
                       "--fov", str(fov), "--distance", str(distance), "--mesh-scale", str(mesh_scale), "--seed", str(seed),
                       "--model-dir", str(self.model_dir), "--dinov3", str(self.dinov3), "--naf", str(self.naf)]
            execution = request.get("gpu_execution", self.args.gpu_execution)
            kernels = request.get("gpu_kernels", self.args.gpu_kernels)
            flow_precision = request.get("gpu_flow_precision", self.args.gpu_flow_precision)
            if execution not in ("legacy", "resident") or kernels not in ("auto", "blas", "mma") or flow_precision not in ("bf16", "fp32", "mixed"):
                raise ValueError("Invalid GPU execution or kernel selection")
            if backend == "cpu":
                execution = "legacy"
            profile = run_dir / "profile.json"
            cmd += ["--gpu-execution", execution, "--gpu-kernels", kernels,
                    "--gpu-flow-precision", flow_precision, "--profile-json", str(profile),
                    "--texture-size", str(texture_size), "--triangle-target", str(triangle_target)]
            if include_ply:
                cmd += ["--ply-output", str(ply_path)]
            if threads:
                cmd += ["--threads", str(threads)]
            if request.get("device") is not None:
                cmd += ["--device", str(bounded_integer(request["device"], "device", 0, 255))]
            if request.get("vram_budget_mib") is not None:
                cmd += ["--vram-budget-mib", str(bounded_integer(request["vram_budget_mib"],
                                                                  "vram_budget_mib", 513, 14336))]
            if mask_path:
                cmd += ["--mask", str(mask_path)]
            started = time.monotonic()
            try:
                proc = run_command(cmd, self.args.timeout, cancel, progress)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"inference exceeded {self.args.timeout:g}s") from exc
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "native runner failed").strip()[-4000:]
                raise RuntimeError(detail)
            if artifact_dir is not None:
                output_path = publish_artifact(output_path, artifact_dir / "native.glb")
                if include_ply:
                    ply_path = publish_artifact(ply_path, artifact_dir / "native.ply")
            else:
                if not valid_output(output_path):
                    raise RuntimeError("native runner did not produce a valid GLB")
                if include_ply and not valid_output(ply_path):
                    raise RuntimeError("native runner did not produce a valid PLY")
            stats = {}
            for line in reversed(proc.stdout.splitlines()):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict):
                        stats = candidate
                        break
                except ValueError:
                    continue
            result = {"ok": True, "backend": backend,
                      "elapsed_ms": round((time.monotonic() - started) * 1000),
                      "stats": stats,
                      "profile": json.loads(profile.read_text()) if profile.is_file() else {}}
            if artifact_dir is not None:
                result["_artifact_files"] = {"native.glb": output_path}
            else:
                result["glb_b64"] = base64.b64encode(output_path.read_bytes()).decode("ascii")
            try:
                result["mesh_summary"] = glb_mesh_summary(output_path)
            except (KeyError, IndexError, OSError, TypeError, ValueError, struct.error):
                result["mesh_summary"] = {"available": False}
            if include_ply:
                if artifact_dir is not None:
                    result["_artifact_files"]["native.ply"] = ply_path
                else:
                    result["ply_b64"] = base64.b64encode(ply_path.read_bytes()).decode("ascii")
            if preparation is not None:
                result["preparation"] = preparation
            return result

    def reference(self, request: dict, cancel: threading.Event | None = None) -> dict:
        """Run the pinned upstream PyTorch pipeline for visual verification."""
        backend = request.get("backend", self.args.backend)
        if backend not in ("cuda", "rocm"):
            raise ValueError("PyTorch reference comparison requires CUDA or ROCm")
        multiview = request.get("views") is not None
        views = request.get("views") if multiview else None
        if multiview and (not isinstance(views, list) or not 1 <= len(views) <= 16):
            raise ValueError("views must contain 1 to 16 posed images")
        prepared_image = request.get("_reference_image_path") or request.get("_reference_image_b64")
        if not multiview and request.get("mask_b64") and not prepared_image:
            raise ValueError("explicit-mask reference comparison requires prepared RGBA input")
        image = None if multiview else decode_b64(
            prepared_image or request.get("image_b64"),
            "prepared reference image" if prepared_image else "image_b64", MAX_IMAGE_BYTES)
        ext = str(request.get(
            "_reference_image_ext" if prepared_image else "image_ext", ".png")).lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        fov = finite_number(request.get("fov", 0.857556), "fov", 0.05, 3.14)
        seed = bounded_integer(request.get("seed", 42), "seed", 0, 2**32 - 1)
        with self.locks[backend], tempfile.TemporaryDirectory(prefix="reference-", dir=self.work_dir) as td:
            run_dir = Path(td)
            artifact_dir = retained_artifact_dir(request)
            output_path = ((artifact_dir / ".reference.glb.partial") if artifact_dir
                           else run_dir / "reference.glb")
            launcher = self.reference_launcher if backend == "cuda" else self.python_launcher
            if multiview:
                frames = []
                for index, item in enumerate(views):
                    if not isinstance(item, dict):
                        raise ValueError("each view must be an object")
                    prepared_view = (item.get("_reference_image_path") or
                                     item.get("_reference_image_b64"))
                    data = decode_b64(prepared_view or item.get("image_b64"),
                                      f"views[{index}].prepared_image" if prepared_view else
                                      f"views[{index}].image_b64", MAX_IMAGE_BYTES)
                    name = f"view{index:02d}.png"
                    (run_dir / name).write_bytes(data)
                    frame = {"file_path": name,
                             "transform_matrix": camera_matrix(item.get("transform_matrix"),
                                                                 f"views[{index}].transform_matrix")}
                    if item.get("fov") is not None:
                        frame["camera_angle_x"] = finite_number(item["fov"], f"views[{index}].fov", 0.05, 3.14)
                    frames.append(frame)
                mesh_scale = finite_number(request.get("mesh_scale", 1.0), "mesh_scale", 1e-5, 1000.0)
                (run_dir / "transforms.json").write_text(json.dumps(
                    {"camera_angle_x": fov, "mesh_scale": mesh_scale, "frames": frames}))
                cmd = [str(launcher), backend, str(self.reference_mv_script),
                       "--views_dir", str(run_dir), "--output", str(output_path),
                       "--seed", str(seed), "--model_path", str(self.model_dir),
                       "--low_vram", "--resolution", "1024"]
            else:
                image_path = run_dir / ("input" + ext)
                image_path.write_bytes(image)
                cmd = [str(launcher), backend, str(self.reference_script), "--image", str(image_path),
                       "--output", str(output_path), "--seed", str(seed), "--fov", str(fov),
                       "--model_path", str(self.model_dir), "--low_vram", "--resolution", "1024"]
            started = time.monotonic()
            try:
                proc = run_command(cmd, self.args.reference_timeout, cancel)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"PyTorch reference exceeded {self.args.reference_timeout:g}s") from exc
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "PyTorch reference failed").strip()[-4000:]
                raise RuntimeError(detail)
            if artifact_dir is not None:
                output_path = publish_artifact(output_path, artifact_dir / "reference.glb")
            elif not valid_output(output_path):
                raise RuntimeError("PyTorch reference did not produce a valid GLB")
            result = {"backend": backend,
                      "elapsed_ms": round((time.monotonic() - started) * 1000),
                      "log_tail": (proc.stdout or "").strip()[-2000:]}
            if artifact_dir is not None:
                result["_artifact_files"] = {"reference.glb": output_path}
            else:
                result["glb_b64"] = base64.b64encode(output_path.read_bytes()).decode("ascii")
            try:
                result["mesh_summary"] = glb_mesh_summary(output_path)
            except (KeyError, IndexError, OSError, TypeError, ValueError, struct.error):
                result["mesh_summary"] = {"available": False}
            return result

    def surface_comparison(self, native: dict, reference: dict,
                           cancel: threading.Event | None = None) -> dict:
        """Run bounded geometry metrics in an isolated short-lived process."""
        with tempfile.TemporaryDirectory(prefix="comparison-", dir=self.work_dir) as td:
            directory = Path(td)
            native_path = directory / "native.glb"
            reference_path = directory / "reference.glb"
            native_files = native.get("_artifact_files", {})
            reference_files = reference.get("_artifact_files", {})
            if native_files.get("native.glb"):
                native_path = Path(native_files["native.glb"])
            else:
                native_path.write_bytes(base64.b64decode(native["glb_b64"], validate=True))
            if reference_files.get("reference.glb"):
                reference_path = Path(reference_files["reference.glb"])
            else:
                reference_path.write_bytes(base64.b64decode(reference["glb_b64"], validate=True))
            command = [str(self.python_launcher), "cpu", str(self.compare_script),
                       str(native_path), str(reference_path), "--samples", "50000"]
            proc = run_command(command, min(self.args.reference_timeout, 600), cancel)
            if proc.returncode:
                raise RuntimeError((proc.stderr or proc.stdout).strip()[-2000:])
            measured = json.loads(proc.stdout)
            return {"available": True, "samples": measured["samples"],
                    "seed": measured["seed"], **measured["geometry"]}


class JobQueue:
    """Bounded queue with atomic manifests for retained job results."""
    MANIFEST = "job.json"
    MANIFEST_VERSION = 2
    TERMINAL = ("complete", "failed", "cancelled")

    def __init__(self, pixal: PixalServer, retained: int = 4,
                 uploads: UploadStore | None = None, ttl: float = 86400):
        self.pixal = pixal
        self.retained = retained
        self.uploads = uploads
        self.ttl = ttl
        self.jobs: dict[str, dict] = {}
        self.pending: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        work_dir = getattr(pixal, "work_dir", None)
        self.result_root = Path(work_dir) / "results" if work_dir is not None else None
        if self.result_root is not None:
            self.result_root.mkdir(parents=True, exist_ok=True)
            self._recover()
        threading.Thread(target=self._worker, daemon=True, name="pixal3d-jobs").start()

    @staticmethod
    def _public_job(job: dict) -> dict:
        return {key: value for key, value in job.items()
                if key != "request" and not key.startswith("_")}

    def _persist_locked(self, job: dict) -> None:
        if self.result_root is None:
            return
        directory = self.result_root / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        job["_artifact_dir"] = directory
        payload = {"version": self.MANIFEST_VERSION, **self._public_job(job)}
        temporary = directory / f".{self.MANIFEST}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x") as handle:
                json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, directory / self.MANIFEST)
            descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _artifact_paths(directory: Path, result: dict) -> dict[str, Path]:
        stored = {}
        expected = [(result, "glb", "native.glb", True),
                    (result, "ply", "native.ply", False)]
        if isinstance(result.get("reference"), dict):
            expected.append((result["reference"], "glb", "reference.glb", True))
        for container, public, name, required in expected:
            artifacts = container.get("artifacts")
            if not isinstance(artifacts, dict) or public not in artifacts:
                if required:
                    raise ValueError(f"recovered result has no {name}")
                continue
            path = directory / name
            if not path.is_file():
                raise ValueError(f"missing recovered artifact {name}")
            stored[name] = path
            artifacts[public] = f"/v1/jobs/{directory.name}/artifacts/{name}"
        return stored

    def _recover(self) -> None:
        now = time.time()
        recovered = []
        for directory in self.result_root.iterdir():
            if (not directory.is_dir() or
                    re.fullmatch(r"[0-9a-f]{32}", directory.name) is None):
                if directory.is_dir():
                    shutil.rmtree(directory, ignore_errors=True)
                else:
                    directory.unlink(missing_ok=True)
                continue
            try:
                payload = json.loads((directory / self.MANIFEST).read_text())
                if (payload.get("version") not in (1, self.MANIFEST_VERSION) or
                        payload.get("id") != directory.name or
                        payload.get("state") not in (*self.TERMINAL, "queued", "running")):
                    raise ValueError("invalid job manifest")
                created = float(payload["created_at"])
                updated = float(payload["updated_at"])
                if not math.isfinite(created) or not math.isfinite(updated):
                    raise ValueError("invalid job timestamps")
                if payload["state"] in self.TERMINAL and updated < now - self.ttl:
                    shutil.rmtree(directory, ignore_errors=True)
                    continue
                job = {key: value for key, value in payload.items() if key != "version"}
                job.update(created_at=created, updated_at=updated)
                job.update(_cancel=threading.Event(), _uploads=[], _artifact_dir=directory,
                           _artifacts={})
                if job["state"] in ("queued", "running"):
                    progress = int(job.get("progress", 0))
                    shutil.rmtree(directory)
                    directory.mkdir()
                    job.pop("result", None)
                    job.update(state="failed", phase="failed",
                               progress=max(0, min(progress, 99)),
                               updated_at=now, completed_at=now, error_code="server_restarted",
                               error="server restarted before job completed",
                               _artifact_dir=directory, _artifacts={})
                elif job["state"] == "complete":
                    if not isinstance(job.get("result"), dict):
                        raise ValueError("complete job has no result")
                    job["_artifacts"] = self._artifact_paths(directory, job["result"])
                recovered.append(job)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                shutil.rmtree(directory, ignore_errors=True)
        recovered.sort(key=lambda job: job["updated_at"], reverse=True)
        for job in recovered[:self.retained]:
            self.jobs[job["id"]] = job
            if job["state"] == "failed" and job.get("error_code") == "server_restarted":
                self._persist_locked(job)
        for job in recovered[self.retained:]:
            shutil.rmtree(job["_artifact_dir"], ignore_errors=True)

    def _remove_artifacts_locked(self, job: dict) -> None:
        directory = job.get("_artifact_dir")
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)

    def _store_artifacts(self, job_id: str, result: dict) -> None:
        if self.result_root is None:
            return
        directory = self.result_root / job_id
        directory.mkdir(parents=True, exist_ok=True)
        stored = {}

        for container in (result, result.get("reference")):
            if not isinstance(container, dict):
                continue
            files = container.pop("_artifact_files", {})
            if not isinstance(files, dict):
                raise ValueError("invalid retained artifact map")
            for name, source in files.items():
                if name not in ("native.glb", "native.ply", "reference.glb"):
                    raise ValueError(f"invalid retained artifact name: {name}")
                path = Path(source)
                try:
                    path.relative_to(directory)
                except ValueError as exc:
                    raise ValueError("retained artifact escaped job storage") from exc
                if not valid_output(path):
                    raise ValueError(f"missing retained artifact {name}")
                stored[name] = path
                public = "ply" if name.endswith(".ply") else "glb"
                container.setdefault("artifacts", {})[public] = (
                    f"/v1/jobs/{job_id}/artifacts/{name}")

        def save(container: dict, field: str, name: str, public: str) -> None:
            encoded = container.pop(field, None)
            if encoded is None:
                return
            path = directory / name
            path.write_bytes(base64.b64decode(encoded, validate=True))
            stored[name] = path
            container.setdefault("artifacts", {})[public] = (
                f"/v1/jobs/{job_id}/artifacts/{name}")

        save(result, "glb_b64", "native.glb", "glb")
        save(result, "ply_b64", "native.ply", "ply")
        if isinstance(result.get("reference"), dict):
            save(result["reference"], "glb_b64", "reference.glb", "glb")
        with self.lock:
            self.jobs[job_id]["_artifact_dir"] = directory
            self.jobs[job_id]["_artifacts"] = stored

    def _expire_locked(self, now: float | None = None) -> None:
        cutoff = (time.time() if now is None else now) - self.ttl
        for job_id, job in list(self.jobs.items()):
            if (job["state"] not in ("queued", "running") and
                    job.get("updated_at", job["created_at"]) < cutoff):
                self._remove_artifacts_locked(job)
                self.jobs.pop(job_id, None)

    def submit(self, request: dict) -> dict:
        upload_paths: list[Path] = []
        if self.uploads:
            request, upload_paths = self.uploads.claim(request)
        job_id = uuid.uuid4().hex
        now = time.time()
        with self.lock:
            self._expire_locked(now)
            active = sum(j["state"] in ("queued", "running") for j in self.jobs.values())
            if active >= self.retained:
                for path in upload_paths:
                    path.unlink(missing_ok=True)
                raise QueueFull("job queue is full")
            terminal = sorted((j for j in self.jobs.values()
                               if j["state"] not in ("queued", "running")),
                              key=lambda j: j["updated_at"])
            while len(terminal) >= self.retained:
                expired = terminal.pop(0)
                self._remove_artifacts_locked(expired)
                self.jobs.pop(expired["id"], None)
            if self.result_root is not None:
                request["_artifact_dir"] = self.result_root / job_id
            self.jobs[job_id] = {"id": job_id, "state": "queued", "phase": "queued", "progress": 0,
                                 "created_at": now, "updated_at": now, "request": request,
                                 "_cancel": threading.Event(), "_uploads": upload_paths}
            self._persist_locked(self.jobs[job_id])
        self.pending.put(job_id)
        return self.status(job_id)

    def status(self, job_id: str, include_result: bool = False) -> dict:
        with self.lock:
            self._expire_locked()
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            out = {key: value for key, value in self._public_job(job).items()
                   if key != "result"}
            out["queue_position"] = self._queue_position(job_id) if job["state"] == "queued" else None
            if include_result and job["state"] == "complete":
                out["result"] = job["result"]
            return out

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            self._expire_locked()
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            if job["state"] == "queued":
                job.update(state="cancelled", phase="cancelled", updated_at=time.time())
                self._persist_locked(job)
            elif job["state"] == "running":
                job["cancel_requested"] = True
                job["_cancel"].set()
                job["updated_at"] = time.time()
                self._persist_locked(job)
            return {key: value for key, value in self._public_job(job).items()
                    if key != "result"}

    def delete(self, job_id: str) -> dict:
        with self.lock:
            self._expire_locked()
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            if job["state"] in ("queued", "running"):
                if job["state"] == "queued":
                    job.update(state="cancelled", phase="cancelled", updated_at=time.time())
                else:
                    job["cancel_requested"] = True
                    job["_cancel"].set()
                    job["updated_at"] = time.time()
                self._persist_locked(job)
                return {"id": job_id, "state": job["state"], "deleted": False}
            self.jobs.pop(job_id)
            self._remove_artifacts_locked(job)
            return {"id": job_id, "state": "deleted", "deleted": True}

    def artifact(self, job_id: str, name: str) -> Path:
        if name not in ("native.glb", "native.ply", "reference.glb"):
            raise KeyError(name)
        with self.lock:
            self._expire_locked()
            job = self.jobs.get(job_id)
            path = None if job is None else job.get("_artifacts", {}).get(name)
            if path is None or not path.is_file():
                raise KeyError(name)
            return path

    def _queue_position(self, job_id: str) -> int:
        queued = sorted((j for j in self.jobs.values() if j["state"] == "queued"),
                        key=lambda j: j["created_at"])
        return next((i + 1 for i, job in enumerate(queued) if job["id"] == job_id), 0)

    def _update(self, job_id: str, **values):
        with self.lock:
            job = self.jobs[job_id]
            job.update(values, updated_at=time.time())
            if "state" in values:
                self._persist_locked(job)

    def _worker(self):
        while True:
            job_id = self.pending.get()
            request = None
            try:
                with self.lock:
                    job = self.jobs.get(job_id)
                    if job is None:
                        continue
                    if job["state"] == "cancelled":
                        continue
                    request = job["request"]
                self._update(job_id, state="running", phase="starting native inference",
                             progress=1, started_at=time.time())
                cancel = job["_cancel"]
                def report(line):
                    update = native_progress(line)
                    if update:
                        phase, percent = update
                        with self.lock:
                            previous = self.jobs[job_id].get("progress", 0)
                        self._update(job_id, phase=phase, progress=max(previous, percent))
                result = self.pixal.infer(request, cancel, report)
                with self.lock:
                    cancelled = self.jobs[job_id].get("cancel_requested", False)
                if cancelled:
                    self._update(job_id, state="cancelled", phase="cancelled")
                    continue
                if request.get("reference"):
                    self._update(job_id, phase="PyTorch reference", progress=98)
                    result["reference"] = self.pixal.reference(reference_request(request, result), cancel)
                    self._update(job_id, phase="surface comparison", progress=99)
                    attach_comparison(self.pixal, result, cancel)
                self._store_artifacts(job_id, result)
                self._update(job_id, state="complete", phase="complete", progress=100, result=result,
                             completed_at=time.time())
            except JobCancelled:
                self._update(job_id, state="cancelled", phase="cancelled",
                             completed_at=time.time())
            except Exception as exc:
                code = ("invalid_request" if isinstance(exc, ValueError) else
                        "timeout" if isinstance(exc, TimeoutError) else "execution_failed")
                self._update(job_id, state="failed", phase="failed", error=str(exc), error_code=code,
                             completed_at=time.time())
            finally:
                with self.lock:
                    job = self.jobs.get(job_id)
                    paths = [] if job is None else job.get("_uploads", [])
                for path in paths:
                    path.unlink(missing_ok=True)
                artifact_dir = retained_artifact_dir(request) if request is not None else None
                if artifact_dir is not None:
                    shutil.rmtree(artifact_dir / ".reference-inputs", ignore_errors=True)
                    for partial in artifact_dir.glob(".*.partial"):
                        partial.unlink(missing_ok=True)
                self.pending.task_done()


class Handler(BaseHTTPRequestHandler):
    server_version = "Pixal3DWeb/1.0"
    def json_response(self, status: int, payload: dict):
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)
    def file_response(self, path: Path):
        content_type = "model/gltf-binary" if path.suffix == ".glb" else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile, length=1024 * 1024)
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self.json_response(200, self.server.pixal.health())
            return
        if path in ("/", "/index.html"):
            data = (ROOT / "web/pixal3d.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        if path.startswith("/v1/jobs/"):
            try:
                job_id, tail = (path[len("/v1/jobs/"):].split("/", 1) + [""])[:2]
                if tail.startswith("artifacts/"):
                    self.file_response(self.server.jobs.artifact(
                        job_id, tail[len("artifacts/"):]))
                    return
                self.json_response(200, self.server.jobs.status(job_id, tail == "result"))
            except KeyError:
                self.json_response(404, error_payload("not_found", "job not found"))
            return
        self.json_response(404, error_payload("not_found", "not found"))
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/uploads":
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length <= 0 or length > MAX_IMAGE_BYTES:
                    raise ValueError("invalid upload size")
                upload_id = self.server.uploads.put(self.rfile.read(length))
                self.json_response(201, {"ok": True, "upload_id": upload_id, "bytes": length})
            except QueueFull as exc: self.json_response(429, error_payload("queue_full", str(exc)))
            except ValueError as exc: self.json_response(400, error_payload("invalid_request", str(exc)))
            return
        if path not in ("/v1/infer", "/v1/jobs"):
            self.json_response(404, error_payload("not_found", "not found")); return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0 or length > MAX_BODY_BYTES: raise ValueError("request body too large")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            if path == "/v1/jobs":
                self.json_response(202, self.server.jobs.submit(request)); return
            result = self.server.pixal.infer(request)
            if request.get("reference"):
                result["reference"] = self.server.pixal.reference(reference_request(request, result))
                attach_comparison(self.server.pixal, result)
            self.json_response(200, result)
        except QueueFull as exc: self.json_response(429, error_payload("queue_full", str(exc)))
        except TimeoutError as exc: self.json_response(504, error_payload("timeout", str(exc)))
        except (ValueError, json.JSONDecodeError) as exc: self.json_response(400, error_payload("invalid_request", str(exc)))
        except Exception as exc: self.json_response(500, error_payload("internal_error", str(exc)))
    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/v1/uploads/"):
            upload_id = path[len("/v1/uploads/"):]
            if self.server.uploads.delete(upload_id):
                self.json_response(200, {"ok": True, "upload_id": upload_id, "deleted": True})
            else:
                self.json_response(404, error_payload("not_found", "upload not found"))
            return
        if not path.startswith("/v1/jobs/"):
            self.json_response(404, error_payload("not_found", "not found")); return
        try:
            self.json_response(200, self.server.jobs.delete(path[len("/v1/jobs/"):]))
        except KeyError:
            self.json_response(404, error_payload("not_found", "job not found"))
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {fmt % args}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Pixal3D Python web demo server")
    p.add_argument("--bind", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765)
    p.add_argument("--gpu-execution", choices=("legacy", "resident"), default="resident")
    p.add_argument("--gpu-kernels", choices=("auto", "blas", "mma"), default="auto")
    p.add_argument("--gpu-flow-precision", choices=("bf16", "fp32", "mixed"), default="mixed")
    p.add_argument("--backend", choices=("cpu", "cuda", "rocm"), default="cuda")
    p.add_argument("--binary", default=str(ROOT / "cpu/pixal3d/pixal3d")); p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--dinov3", default=str(DEFAULT_DINOV3)); p.add_argument("--naf", default=str(DEFAULT_NAF))
    p.add_argument("--rembg", default=str(DEFAULT_RMBG)); p.add_argument("--moge", default=str(DEFAULT_MOGE))
    p.add_argument("--work-dir", default=str(ROOT / "tmp/pixal3d/web-runs")); p.add_argument("--threads", type=int, default=0); p.add_argument("--timeout", type=float, default=7200); p.add_argument("--reference-timeout", type=float, default=10800)
    p.add_argument("--retained-jobs", type=int, default=4)
    p.add_argument("--retained-uploads", type=int, default=64)
    p.add_argument("--job-ttl", type=float, default=86400)
    p.add_argument("--upload-ttl", type=float, default=3600)
    args = p.parse_args(); srv = ThreadingHTTPServer((args.bind, args.port), Handler); srv.pixal = PixalServer(args)
    srv.uploads = UploadStore(srv.pixal.work_dir / "uploads",
                              bounded_integer(args.retained_uploads, "retained_uploads", 1, 256),
                              finite_number(args.upload_ttl, "upload_ttl", 1, 604800))
    srv.jobs = JobQueue(srv.pixal, bounded_integer(args.retained_jobs, "retained_jobs", 1, 32),
                        srv.uploads, finite_number(args.job_ttl, "job_ttl", 1, 2592000))
    print(f"Pixal3D demo: http://{args.bind}:{args.port}/ (backend={args.backend})", flush=True); srv.serve_forever()


if __name__ == "__main__": main()
