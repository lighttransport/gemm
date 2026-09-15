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
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path("/mnt/disk2/models/Pixal3D")
DEFAULT_DINOV3 = Path("/mnt/disk2/models/dinov3-vitl16/model.safetensors")
DEFAULT_NAF = ROOT / "ref/pixal3d/weights/naf_release.safetensors"
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_GLB_BYTES = 256 * 1024 * 1024


def decode_b64(value: object, name: str, limit: int) -> bytes:
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


class PixalServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.binary = Path(args.binary).resolve()
        self.model_dir = Path(args.model_dir).resolve()
        self.dinov3 = Path(args.dinov3).resolve()
        self.naf = Path(args.naf).resolve()
        self.work_dir = Path(args.work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.locks = {backend: threading.Lock() for backend in ("cpu", "cuda", "rocm")}
        self.reference_script = ROOT / "ref/pixal3d/upstream/inference.py"
        self.reference_launcher = ROOT / "ref/pixal3d/run.sh"

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
            }
        return {"ok": True, "service": "pixal3d", "default_backend": self.args.backend,
                "default_gpu_execution": self.args.gpu_execution,
                "default_gpu_kernels": self.args.gpu_kernels, "backends": out}

    def infer(self, request: dict) -> dict:
        backend = request.get("backend", self.args.backend)
        if backend not in ("cpu", "cuda", "rocm"):
            raise ValueError("backend must be cpu, cuda, or rocm")
        image = decode_b64(request.get("image_b64"), "image_b64", MAX_IMAGE_BYTES)
        mask = decode_b64(request["mask_b64"], "mask_b64", MAX_IMAGE_BYTES) if request.get("mask_b64") else None
        ext = str(request.get("image_ext", ".png")).lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        fov = finite_number(request.get("fov", 0.857556), "fov", 0.05, 3.14)
        distance = finite_number(request.get("distance", 0.0), "distance", 0.0, 1000.0)
        mesh_scale = finite_number(request.get("mesh_scale", 1.0), "mesh_scale", 1e-5, 1000.0)
        seed = int(request.get("seed", 42))
        threads = int(request.get("threads", self.args.threads))
        if threads < 0 or threads > 1024:
            raise ValueError("threads must be between 0 and 1024")
        with self.locks[backend], tempfile.TemporaryDirectory(prefix="request-", dir=self.work_dir) as td:
            run_dir = Path(td)
            image_path = run_dir / ("input" + ext)
            output_path = run_dir / "output.glb"
            image_path.write_bytes(image)
            mask_path = None
            if mask is not None:
                mask_path = run_dir / "mask.png"
                mask_path.write_bytes(mask)
            cmd = [str(self.binary), "--backend", backend, "--input", str(image_path), "--output", str(output_path),
                   "--fov", str(fov), "--distance", str(distance), "--mesh-scale", str(mesh_scale), "--seed", str(seed),
                   "--model-dir", str(self.model_dir), "--dinov3", str(self.dinov3), "--naf", str(self.naf)]
            execution = request.get("gpu_execution", self.args.gpu_execution)
            kernels = request.get("gpu_kernels", self.args.gpu_kernels)
            if execution not in ("legacy", "resident") or kernels not in ("auto", "blas", "mma"):
                raise ValueError("Invalid GPU execution or kernel selection")
            if backend == "cpu":
                execution = "legacy"
            profile = run_dir / "profile.json"
            cmd += ["--gpu-execution", execution, "--gpu-kernels", kernels, "--profile-json", str(profile)]
            if threads:
                cmd += ["--threads", str(threads)]
            if request.get("device") is not None:
                cmd += ["--device", str(request["device"])]
            if request.get("vram_budget_mib") is not None:
                cmd += ["--vram-budget-mib", str(int(request["vram_budget_mib"]))]
            if mask_path:
                cmd += ["--mask", str(mask_path)]
            started = time.monotonic()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.args.timeout)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"inference exceeded {self.args.timeout:g}s") from exc
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "native runner failed").strip()[-4000:]
                raise RuntimeError(detail)
            if not output_path.is_file() or output_path.stat().st_size > MAX_GLB_BYTES:
                raise RuntimeError("native runner did not produce a valid GLB")
            stats = {}
            for line in reversed(proc.stdout.splitlines()):
                try:
                    candidate = json.loads(line)
                    if isinstance(candidate, dict):
                        stats = candidate
                        break
                except ValueError:
                    continue
            return {"ok": True, "backend": backend, "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "glb_b64": base64.b64encode(output_path.read_bytes()).decode("ascii"), "stats": stats,
                    "profile": json.loads(profile.read_text()) if profile.is_file() else {}}

    def reference(self, request: dict) -> dict:
        """Run the pinned upstream PyTorch pipeline for visual verification."""
        backend = request.get("backend", self.args.backend)
        if backend not in ("cuda", "rocm"):
            raise ValueError("PyTorch reference comparison requires CUDA or ROCm")
        if request.get("mask_b64"):
            raise ValueError("PyTorch reference comparison currently requires an image without a separate mask")
        image = decode_b64(request.get("image_b64"), "image_b64", MAX_IMAGE_BYTES)
        ext = str(request.get("image_ext", ".png")).lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            ext = ".png"
        fov = finite_number(request.get("fov", 0.857556), "fov", 0.05, 3.14)
        seed = int(request.get("seed", 42))
        with self.locks[backend], tempfile.TemporaryDirectory(prefix="reference-", dir=self.work_dir) as td:
            run_dir = Path(td)
            image_path = run_dir / ("input" + ext)
            output_path = run_dir / "reference.glb"
            image_path.write_bytes(image)
            cmd = [str(self.reference_launcher), backend, str(self.reference_script), "--image", str(image_path),
                   "--output", str(output_path), "--seed", str(seed), "--fov", str(fov),
                   "--model_path", str(self.model_dir), "--low_vram", "--resolution", "1024"]
            started = time.monotonic()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.args.reference_timeout)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"PyTorch reference exceeded {self.args.reference_timeout:g}s") from exc
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "PyTorch reference failed").strip()[-4000:]
                raise RuntimeError(detail)
            if not output_path.is_file() or output_path.stat().st_size > MAX_GLB_BYTES:
                raise RuntimeError("PyTorch reference did not produce a valid GLB")
            return {"backend": backend, "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "glb_b64": base64.b64encode(output_path.read_bytes()).decode("ascii"),
                    "log_tail": (proc.stdout or "").strip()[-2000:]}


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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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
        self.json_response(404, {"ok": False, "error": "not found"})
    def do_POST(self):
        if urlparse(self.path).path != "/v1/infer":
            self.json_response(404, {"ok": False, "error": "not found"}); return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 0 or length > MAX_BODY_BYTES: raise ValueError("request body too large")
            request = json.loads(self.rfile.read(length))
            result = self.server.pixal.infer(request)
            if request.get("reference"):
                result["reference"] = self.server.pixal.reference(request)
            self.json_response(200, result)
        except TimeoutError as exc: self.json_response(504, {"ok": False, "error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc: self.json_response(400, {"ok": False, "error": str(exc)})
        except Exception as exc: self.json_response(500, {"ok": False, "error": str(exc)})
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {fmt % args}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Pixal3D Python web demo server")
    p.add_argument("--bind", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765)
    p.add_argument("--gpu-execution", choices=("legacy", "resident"), default="legacy")
    p.add_argument("--gpu-kernels", choices=("auto", "blas", "mma"), default="auto")
    p.add_argument("--backend", choices=("cpu", "cuda", "rocm"), default="cuda")
    p.add_argument("--binary", default=str(ROOT / "cpu/pixal3d/pixal3d")); p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--dinov3", default=str(DEFAULT_DINOV3)); p.add_argument("--naf", default=str(DEFAULT_NAF))
    p.add_argument("--work-dir", default=str(ROOT / "tmp/pixal3d/web-runs")); p.add_argument("--threads", type=int, default=0); p.add_argument("--timeout", type=float, default=7200); p.add_argument("--reference-timeout", type=float, default=10800)
    args = p.parse_args(); srv = ThreadingHTTPServer((args.bind, args.port), Handler); srv.pixal = PixalServer(args)
    print(f"Pixal3D demo: http://{args.bind}:{args.port}/ (backend={args.backend})", flush=True); srv.serve_forever()


if __name__ == "__main__": main()
