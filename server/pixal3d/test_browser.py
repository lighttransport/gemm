#!/usr/bin/env python3
"""Headless-Chrome smoke test for the real Pixal3D demo document."""
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.pixal3d import app


class FakePixal:
    args = argparse.Namespace(backend="cuda", gpu_execution="resident",
                              gpu_kernels="auto", gpu_flow_precision="mixed")

    def health(self):
        ready = {"available": True, "models_ready": True, "multiview_ready": True}
        return {"ok": True, "service": "pixal3d", "default_backend": "cuda",
                "default_gpu_execution": "resident", "default_gpu_kernels": "auto",
                "default_gpu_flow_precision": "mixed",
                "backends": {name: ready for name in ("cpu", "cuda", "rocm")}}


def main():
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise SystemExit("Chrome/Chromium is required")
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    server.pixal = FakePixal()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    scratch = app.ROOT / "tmp/pixal3d/browser-test"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="chrome-", dir=scratch) as profile:
            url = f"http://127.0.0.1:{server.server_port}/"
            run = subprocess.run(
                [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                 f"--user-data-dir={profile}", "--virtual-time-budget=3000", "--dump-dom", url],
                capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr[-2000:]
        document = run.stdout
        for marker in ("Pixal3D Studio", 'id="view-folder"', 'id="camera-map"',
                       'id="texture-size"', 'id="triangle-target"', 'id="cancel"',
                       "'/v1/uploads'", "CUDA: ready · multiview"):
            assert marker in document, marker
        print("Pixal3D browser UI smoke test: PASS")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
