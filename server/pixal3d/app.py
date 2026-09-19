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
from pathlib import Path
import queue
import re
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
    def __init__(self, root: Path, retained: int = 64):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.retained = retained
        self.files: dict[str, Path] = {}
        self.lock = threading.Lock()

    def put(self, data: bytes) -> str:
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError(f"upload exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit")
        upload_id = uuid.uuid4().hex
        path = self.root / upload_id
        with self.lock:
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
                if isinstance(view, dict) and "image_upload" in view:
                    refs.append((view, "image_upload", "image_b64"))
        ids = [container[source] for container, source, _ in refs]
        if len(ids) != len(set(ids)):
            raise ValueError("an upload ID may only be used once")
        with self.lock:
            if any(not isinstance(item, str) or item not in self.files for item in ids):
                raise ValueError("unknown or expired upload ID")
            paths = [self.files.pop(item) for item in ids]
        for (container, source, destination), path in zip(refs, paths):
            container[destination] = path
            del container[source]
        return resolved, paths


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


def valid_output(path: Path, limit: int = MAX_GLB_BYTES) -> bool:
    return path.is_file() and 0 < path.stat().st_size <= limit


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
        self.reference_script = ROOT / "ref/pixal3d/upstream/inference.py"
        self.reference_mv_script = ROOT / "ref/pixal3d/upstream/inference_mv.py"
        self.reference_launcher = ROOT / "ref/pixal3d/run.sh"
        self.prepare_script = ROOT / "ref/pixal3d/prepare_input.py"

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
        return {"ok": True, "service": "pixal3d", "default_backend": self.args.backend,
                "default_gpu_execution": self.args.gpu_execution,
                "default_gpu_kernels": self.args.gpu_kernels,
                "default_gpu_flow_precision": self.args.gpu_flow_precision,
                "preparation": {"mask_ready": rmbg_ready(self.rembg),
                                "camera_ready": self.moge.is_file()},
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
            output_path = run_dir / "output.glb"
            ply_path = run_dir / "output.ply"
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
                    matrix = camera_matrix(item.get("transform_matrix"), f"views[{index}].transform_matrix")
                    frame = {"file_path": name, "transform_matrix": matrix}
                    if item.get("fov") is not None:
                        frame["camera_angle_x"] = finite_number(item["fov"], f"views[{index}].fov", 0.05, 3.14)
                    if auto_mask:
                        prepared = run_dir / f"view{index:02d}-prepared.png"
                        metadata = run_dir / f"view{index:02d}-prepared.json"
                        prep = [str(self.reference_launcher), backend, str(self.prepare_script),
                                "--input", str(view_path), "--output", str(prepared),
                                "--metadata", str(metadata), "--rembg-model", str(self.rembg),
                                "--fov", str(frame.get("camera_angle_x", fov)),
                                "--mesh-scale", str(mesh_scale), "--device",
                                "cpu" if backend == "cpu" else "cuda"]
                        proc = run_command(prep, self.args.reference_timeout, cancel)
                        if proc.returncode:
                            raise RuntimeError((proc.stderr or proc.stdout).strip()[-4000:])
                        frame["file_path"] = prepared.name
                        item_preparation = json.loads(metadata.read_text())
                        item_preparation["view"] = index
                        view_preparation.append(item_preparation)
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
                if auto_mask or auto_camera:
                    prepared = run_dir / "prepared.png"
                    metadata = run_dir / "prepared.json"
                    prep = [str(self.reference_launcher), backend, str(self.prepare_script),
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
                      "glb_b64": base64.b64encode(output_path.read_bytes()).decode("ascii"),
                      "stats": stats,
                      "profile": json.loads(profile.read_text()) if profile.is_file() else {}}
            if include_ply:
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
        if not multiview and request.get("mask_b64"):
            raise ValueError("PyTorch reference comparison currently requires an image without a separate mask")
        image = None if multiview else decode_b64(request.get("image_b64"), "image_b64", MAX_IMAGE_BYTES)
        ext = str(request.get("image_ext", ".png")).lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        fov = finite_number(request.get("fov", 0.857556), "fov", 0.05, 3.14)
        seed = bounded_integer(request.get("seed", 42), "seed", 0, 2**32 - 1)
        with self.locks[backend], tempfile.TemporaryDirectory(prefix="reference-", dir=self.work_dir) as td:
            run_dir = Path(td)
            output_path = run_dir / "reference.glb"
            if multiview:
                frames = []
                for index, item in enumerate(views):
                    if not isinstance(item, dict):
                        raise ValueError("each view must be an object")
                    data = decode_b64(item.get("image_b64"), f"views[{index}].image_b64", MAX_IMAGE_BYTES)
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
                cmd = [str(self.reference_launcher), backend, str(self.reference_mv_script),
                       "--views_dir", str(run_dir), "--output", str(output_path),
                       "--seed", str(seed), "--model_path", str(self.model_dir),
                       "--low_vram", "--resolution", "1024"]
            else:
                image_path = run_dir / ("input" + ext)
                image_path.write_bytes(image)
                cmd = [str(self.reference_launcher), backend, str(self.reference_script), "--image", str(image_path),
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
            if not valid_output(output_path):
                raise RuntimeError("PyTorch reference did not produce a valid GLB")
            return {"backend": backend, "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "glb_b64": base64.b64encode(output_path.read_bytes()).decode("ascii"),
                    "log_tail": (proc.stdout or "").strip()[-2000:]}


class JobQueue:
    """Bounded in-memory queue for long-running demo requests."""
    def __init__(self, pixal: PixalServer, retained: int = 4,
                 uploads: UploadStore | None = None):
        self.pixal = pixal
        self.retained = retained
        self.uploads = uploads
        self.jobs: dict[str, dict] = {}
        self.pending: queue.Queue[str] = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="pixal3d-jobs").start()

    def submit(self, request: dict) -> dict:
        upload_paths: list[Path] = []
        if self.uploads:
            request, upload_paths = self.uploads.claim(request)
        job_id = uuid.uuid4().hex
        now = time.time()
        with self.lock:
            active = sum(j["state"] in ("queued", "running") for j in self.jobs.values())
            if active >= self.retained:
                for path in upload_paths:
                    path.unlink(missing_ok=True)
                raise QueueFull("job queue is full")
            terminal = sorted((j for j in self.jobs.values()
                               if j["state"] not in ("queued", "running")),
                              key=lambda j: j["updated_at"])
            while len(terminal) >= self.retained:
                self.jobs.pop(terminal.pop(0)["id"], None)
            self.jobs[job_id] = {"id": job_id, "state": "queued", "phase": "queued", "progress": 0,
                                 "created_at": now, "updated_at": now, "request": request,
                                 "_cancel": threading.Event(), "_uploads": upload_paths}
        self.pending.put(job_id)
        return self.status(job_id)

    def status(self, job_id: str, include_result: bool = False) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            out = {key: value for key, value in job.items()
                   if key not in ("request", "result") and not key.startswith("_")}
            out["queue_position"] = self._queue_position(job_id) if job["state"] == "queued" else None
            if include_result and job["state"] == "complete":
                out["result"] = job["result"]
            return out

    def cancel(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            if job["state"] == "queued":
                job.update(state="cancelled", phase="cancelled", updated_at=time.time())
            elif job["state"] == "running":
                job["cancel_requested"] = True
                job["_cancel"].set()
                job["updated_at"] = time.time()
            return {key: value for key, value in job.items()
                    if key not in ("request", "result") and not key.startswith("_")}

    def _queue_position(self, job_id: str) -> int:
        queued = sorted((j for j in self.jobs.values() if j["state"] == "queued"),
                        key=lambda j: j["created_at"])
        return next((i + 1 for i, job in enumerate(queued) if job["id"] == job_id), 0)

    def _update(self, job_id: str, **values):
        with self.lock:
            self.jobs[job_id].update(values, updated_at=time.time())

    def _worker(self):
        while True:
            job_id = self.pending.get()
            try:
                with self.lock:
                    job = self.jobs[job_id]
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
                    paths = self.jobs[job_id].get("_uploads", [])
                for path in paths:
                    path.unlink(missing_ok=True)
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
            self.json_response(200, result)
        except QueueFull as exc: self.json_response(429, error_payload("queue_full", str(exc)))
        except TimeoutError as exc: self.json_response(504, error_payload("timeout", str(exc)))
        except (ValueError, json.JSONDecodeError) as exc: self.json_response(400, error_payload("invalid_request", str(exc)))
        except Exception as exc: self.json_response(500, error_payload("internal_error", str(exc)))
    def do_DELETE(self):
        path = urlparse(self.path).path
        if not path.startswith("/v1/jobs/"):
            self.json_response(404, error_payload("not_found", "not found")); return
        try:
            self.json_response(200, self.server.jobs.cancel(path[len("/v1/jobs/"):]))
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
    args = p.parse_args(); srv = ThreadingHTTPServer((args.bind, args.port), Handler); srv.pixal = PixalServer(args)
    srv.uploads = UploadStore(srv.pixal.work_dir / "uploads",
                              bounded_integer(args.retained_uploads, "retained_uploads", 1, 256))
    srv.jobs = JobQueue(srv.pixal, bounded_integer(args.retained_jobs, "retained_jobs", 1, 32), srv.uploads)
    print(f"Pixal3D demo: http://{args.bind}:{args.port}/ (backend={args.backend})", flush=True); srv.serve_forever()


if __name__ == "__main__": main()
