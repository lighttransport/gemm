#!/usr/bin/env python3
"""Standalone Qwen-Image 2.1 web demo server.

The server intentionally shells out to the checked-in native and reference
drivers.  This keeps the web process small and prevents two model copies from
being resident in the 12--16 GB CUDA device at once.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
DEFAULT_MODEL = Path("/mnt/nvme01/models/qimg-21")
DEFAULT_QUANT = ROOT / "tmp/qimg21-int8-package"
DEFAULT_PYTHON = ROOT / "tmp/qimg21-ref-venv/bin/python"
MAX_BODY = 32 * 1024


class Demo:
    def __init__(self, model: Path, quant: Path, python: Path, work: Path,
                 native: Path, host: str, port: int,
                 *, native_rocm: Path | None = None,
                 python_rocm: Path | None = None):
        self.model = model.resolve()
        self.quant = quant.resolve()
        # Preserve a venv/uv launcher symlink; Path.resolve() would collapse it
        # to the host interpreter and lose the environment's Torch packages.
        self.python = python.absolute()
        self.python_rocm = (python_rocm or python).absolute()
        self.work = work.resolve()
        self.native = native.resolve()
        self.native_rocm = (native_rocm or native).resolve()
        self.host, self.port = host, port
        self.lock = threading.Lock()

    def native_components(self, backend: str) -> dict[str, Path]:
        """Return the complete native subprocess set for one backend."""
        root = ROOT / ("cuda/qimg21" if backend == "cuda" else "rdna4/qimg21")
        return {
            "transformer": self.native if backend == "cuda" else self.native_rocm,
            "text": root / ("test_cuda_qimg21_text" if backend == "cuda" else "test_hip_qimg21_text"),
            "vision": root / ("test_cuda_qimg21_vision" if backend == "cuda" else "test_hip_qimg21_vision"),
            "vae": root / ("test_cuda_qimg21_vae" if backend == "cuda" else "test_hip_qimg21_vae"),
            "vae_encode": root / ("test_cuda_qimg21_vae_encode" if backend == "cuda" else "test_hip_qimg21_vae_encode"),
        }

    def _validate(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        prompt = request.get("prompt", "").strip()
        if not prompt or len(prompt) > 4000:
            raise ValueError("prompt must contain 1--4000 characters")
        # A request that names a backend but omits mode is a native request.
        # Keep the legacy mode=cuda/rocm shorthand for older clients.
        mode = request.get("mode", "native" if "backend" in request else "cuda")
        backend = request.get("backend", "cuda")
        # Preserve the original API where mode=cuda meant native CUDA.  New
        # callers should use backend=cuda|rocm and mode=native|reference|compare.
        if mode in {"cuda", "rocm"}:
            backend = mode
            mode = "native"
        if backend not in {"cuda", "rocm"}:
            raise ValueError("backend must be cuda or rocm")
        if mode not in {"native", "reference", "compare"}:
            raise ValueError("mode must be native, reference, or compare")
        width = int(request.get("width", 256)); height = int(request.get("height", 256))
        steps = int(request.get("steps", 2)); seed = int(request.get("seed", 42))
        if width < 256 or height < 256 or width > 1024 or height > 1024 or width % 32 or height % 32:
            raise ValueError("width and height must be 256..1024 and divisible by 32")
        if steps < 1 or steps > 40:
            raise ValueError("steps must be between 1 and 40")
        negative = request.get("negative_prompt", "")
        if not isinstance(negative, str) or len(negative) > 4000:
            raise ValueError("negative_prompt must be at most 4000 characters")
        quantized = bool(request.get("quantized", False))
        return {"prompt": prompt, "negative_prompt": negative.strip(),
                "mode": mode, "backend": backend,
                "width": width, "height": height, "steps": steps, "seed": seed,
                "quantized": quantized}

    def _run(self, command: list[str], cwd: Path, log: Path,
             env: dict[str, str] | None = None) -> None:
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=3600, check=False, env=env)
        if result.returncode:
            tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"inference exited with {result.returncode}: {tail}")

    def _native(self, cfg: dict, out: Path) -> Path:
        backend = cfg["backend"]
        python = self.python if backend == "cuda" else self.python_rocm
        if backend == "rocm" and not python.is_file():
            # The native ROCm path only needs NumPy and Pillow. A ROCm Torch
            # environment is required for reference mode, but not generation.
            python = Path(sys.executable)
        native = self.native if backend == "cuda" else self.native_rocm
        image = out / f"{backend}.png"
        work = out / f"{backend}-work"
        attention = "cutlass-efficient" if backend == "cuda" else "wmma"
        native_vae = (backend == "cuda" or
                      (ROOT / "rdna4/qimg21/test_hip_qimg21_vae").is_file())
        command = [str(python), "cuda/qimg21/native_generate.py", "--backend", backend,
                   "--model", str(self.model),
                   "--prompt", cfg["prompt"], "--height", str(cfg["height"]),
                   "--width", str(cfg["width"]), "--steps", str(cfg["steps"]),
                   "--seed", str(cfg["seed"]), "--dtype", "bf16",
                   "--native-bin", str(native),
                   "--native-attention", attention, "--native-normalization", "vector4",
                   "--native-rope", "host-table-exact", "--work-dir", str(work), "--out", str(image)]
        if native_vae:
            command.insert(command.index("--native-bin"), "--native-vae")
        if cfg["negative_prompt"]:
            command += ["--negative-prompt", cfg["negative_prompt"], "--true-cfg-scale", "4.0"]
        if cfg["quantized"]:
            if not self.quant.is_dir():
                raise RuntimeError(f"quantized package is unavailable: {self.quant}")
            command += ["--quantized-transformer", str(self.quant)]
            if backend == "cuda":
                command += ["--int8-tensor-core", "--int8-bf16-tail-blocks", "16"]
        # uv-managed reference environments can expose the host interpreter as
        # sys.executable from a child process; carry the selected interpreter
        # explicitly to the fixture helper so it retains Torch/CUDA imports.
        env = os.environ.copy()
        env["QIMG21_PYTHON"] = str(python)
        self._run(command, ROOT, out / f"{backend}.log", env=env)
        return image

    def _reference(self, cfg: dict, out: Path) -> Path:
        backend = cfg["backend"]
        python = self.python if backend == "cuda" else self.python_rocm
        image = out / "reference" / f"{backend}.png"
        command = [str(python), "cuda/qimg21/reference.py", "--model", str(self.model),
                   "--prompt", cfg["prompt"], "--height", str(cfg["height"]),
                   "--width", str(cfg["width"]), "--steps", str(cfg["steps"]),
                   "--seed", str(cfg["seed"]), "--dtype", "bf16", "--sdpa-backend", "efficient",
                   "--dump-dir", str(image.parent)]
        if cfg["negative_prompt"]:
            command += ["--negative-prompt", cfg["negative_prompt"], "--true-cfg-scale", "4.0"]
        self._run(command, ROOT, out / f"reference-{backend}.log")
        return image

    @staticmethod
    def _data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def generate(self, request: dict, progress=None) -> dict:
        cfg = self._validate(request)
        job = self.work / uuid.uuid4().hex
        job.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        results: dict = {"request": cfg, "job": job.name}
        with self.lock:
            if cfg["mode"] in {"native", "compare"}:
                backend = cfg["backend"]
                if progress: progress(f"Qwen {backend.upper()} native", 12)
                path = self._native(cfg, job)
                if progress: progress(f"Qwen {backend.upper()} native complete", 72 if cfg["mode"] == "compare" else 96)
                results[backend] = {"image": self._data_url(path)}
            if cfg["mode"] in {"reference", "compare"}:
                if progress: progress(f"Qwen {cfg['backend'].upper()} PyTorch reference", 78)
                path = self._reference(cfg, job)
                if progress: progress("Qwen PyTorch reference complete", 96)
                results["reference"] = {"image": self._data_url(path)}
        results["elapsed_ms"] = round((time.monotonic() - started) * 1000)
        return results


class Handler(BaseHTTPRequestHandler):
    server_version = "qwen-image21-demo/1.0"

    def _json(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:
        demo: Demo = self.server.demo  # type: ignore[attr-defined]
        path = urlparse(self.path).path
        if path == "/api/health":
            components = {backend: {name: item.is_file()
                                    for name, item in demo.native_components(backend).items()}
                          for backend in ("cuda", "rocm")}
            self._json(200, {"ok": True, "model": str(demo.model),
                             "quantized_available": demo.quant.is_dir(),
                             "native": {"cuda": demo.native.is_file(),
                                        "rocm": demo.native_rocm.is_file()},
                             "native_components": components,
                             "reference": {"cuda": demo.python.is_file(),
                                           "rocm": demo.python_rocm.is_file()}})
            return
        if path in {"/", "/index.html"}:
            data = (WEB / "qwen_image21.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/generate":
            self._json(404, {"ok": False, "error": "not found"}); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY: raise ValueError("request body is too large")
            request = json.loads(self.rfile.read(length))
            result = self.server.demo.generate(request)  # type: ignore[attr-defined]
            self._json(200, {"ok": True, **result})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
        except Exception as exc:  # inference errors are shown in the UI
            self._json(500, {"ok": False, "error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[qwen-image21] {self.address_string()} {fmt % args}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--quant-package", type=Path, default=DEFAULT_QUANT)
    ap.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    ap.add_argument("--native", type=Path, default=ROOT / "cuda/qimg21/test_cuda_qimg21_native")
    ap.add_argument("--native-rocm", type=Path,
                    default=ROOT / "rdna4/qimg21/test_hip_qimg21_native")
    ap.add_argument("--python-rocm", type=Path,
                    default=ROOT / "tmp/qimg21-rocm-venv/bin/python")
    ap.add_argument("--work-dir", type=Path, default=ROOT / "tmp/qimg21-web-jobs")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    if not args.model.is_dir(): ap.error(f"model directory not found: {args.model}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    demo = Demo(args.model, args.quant_package, args.python, args.work_dir, args.native,
                args.host, args.port, native_rocm=args.native_rocm,
                python_rocm=args.python_rocm)
    server = ThreadingHTTPServer((args.host, args.port), Handler); server.demo = demo  # type: ignore[attr-defined]
    print(f"Qwen Image 2.1 demo: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
