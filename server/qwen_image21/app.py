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
                 native: Path, host: str, port: int):
        self.model = model.resolve()
        self.quant = quant.resolve()
        # Preserve a venv/uv launcher symlink; Path.resolve() would collapse it
        # to the host interpreter and lose the environment's Torch packages.
        self.python = python.absolute()
        self.work = work.resolve()
        self.native = native.resolve()
        self.host, self.port = host, port
        self.lock = threading.Lock()

    def _validate(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        prompt = request.get("prompt", "").strip()
        if not prompt or len(prompt) > 4000:
            raise ValueError("prompt must contain 1--4000 characters")
        mode = request.get("mode", "cuda")
        if mode not in {"cuda", "reference", "compare"}:
            raise ValueError("mode must be cuda, reference, or compare")
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
        return {"prompt": prompt, "negative_prompt": negative.strip(), "mode": mode,
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

    def _cuda(self, cfg: dict, out: Path) -> Path:
        image = out / "cuda.png"
        work = out / "cuda-work"
        command = [str(self.python), "cuda/qimg21/native_generate.py", "--model", str(self.model),
                   "--prompt", cfg["prompt"], "--height", str(cfg["height"]),
                   "--width", str(cfg["width"]), "--steps", str(cfg["steps"]),
                   "--seed", str(cfg["seed"]), "--dtype", "bf16", "--native-vae",
                   "--native-attention", "cutlass-efficient", "--native-normalization", "vector4",
                   "--native-rope", "host-table-exact", "--work-dir", str(work), "--out", str(image)]
        if cfg["negative_prompt"]:
            command += ["--negative-prompt", cfg["negative_prompt"], "--true-cfg-scale", "4.0"]
        if cfg["quantized"]:
            if not self.quant.is_dir():
                raise RuntimeError(f"quantized package is unavailable: {self.quant}")
            command += ["--quantized-transformer", str(self.quant), "--int8-tensor-core",
                        "--int8-bf16-tail-blocks", "16"]
        # uv-managed reference environments can expose the host interpreter as
        # sys.executable from a child process; carry the selected interpreter
        # explicitly to the fixture helper so it retains Torch/CUDA imports.
        env = os.environ.copy()
        env["QIMG21_PYTHON"] = str(self.python)
        self._run(command, ROOT, out / "cuda.log", env=env)
        return image

    def _reference(self, cfg: dict, out: Path) -> Path:
        image = out / "reference" / "reference.png"
        command = [str(self.python), "cuda/qimg21/reference.py", "--model", str(self.model),
                   "--prompt", cfg["prompt"], "--height", str(cfg["height"]),
                   "--width", str(cfg["width"]), "--steps", str(cfg["steps"]),
                   "--seed", str(cfg["seed"]), "--dtype", "bf16", "--sdpa-backend", "efficient",
                   "--dump-dir", str(image.parent)]
        if cfg["negative_prompt"]:
            command += ["--negative-prompt", cfg["negative_prompt"], "--true-cfg-scale", "4.0"]
        self._run(command, ROOT, out / "reference.log")
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
            if cfg["mode"] in {"cuda", "compare"}:
                if progress: progress("Qwen CUDA native", 12)
                path = self._cuda(cfg, job)
                if progress: progress("Qwen CUDA native complete", 72 if cfg["mode"] == "compare" else 96)
                results["cuda"] = {"image": self._data_url(path)}
            if cfg["mode"] in {"reference", "compare"}:
                if progress: progress("Qwen PyTorch reference", 78)
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
            self._json(200, {"ok": True, "model": str(demo.model), "quantized_available": demo.quant.is_dir()})
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
    ap.add_argument("--work-dir", type=Path, default=ROOT / "tmp/qimg21-web-jobs")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()
    if not args.model.is_dir(): ap.error(f"model directory not found: {args.model}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    demo = Demo(args.model, args.quant_package, args.python, args.work_dir, args.native, args.host, args.port)
    server = ThreadingHTTPServer((args.host, args.port), Handler); server.demo = demo  # type: ignore[attr-defined]
    print(f"Qwen Image 2.1 demo: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
