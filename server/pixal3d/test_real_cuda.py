#!/usr/bin/env python3
"""Opt-in real queued HTTP/CUDA test using a pinned public Pixal3D image."""
import argparse
import ctypes as C
import hashlib
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

from PIL import Image
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.pixal3d import app


IMAGE_URL = ("https://raw.githubusercontent.com/TencentARC/Pixal3D/"
             "f7cf38429b0bd264f1995f0f8743a88b1c728b94/assets/images/1_img.png")
IMAGE_SHA256 = "fdd82d60b7ec11e6d5699df29693d8ab538f9dab4b04e3f2abaa59ccd7b4709a"


def request_json(url, method="GET", body=None):
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, method=method,
                      headers={"Content-Type": "application/json"} if data else {})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def upload(url, payload):
    request = Request(url + "/v1/uploads", data=payload, method="POST",
                      headers={"Content-Type": "application/octet-stream"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read())["upload_id"]


class QuietHandler(app.Handler):
    """Keep one-second polling from obscuring inference diagnostics."""

    def log_message(self, _format, *_args):
        pass


class Monitor:
    class Memory(C.Structure):
        _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong),
                    ("used", C.c_ulonglong)]

    def __init__(self):
        self.process = psutil.Process()
        self.peak_rss = 0
        self.peak_device = 0
        self.stop = threading.Event()
        try:
            self.nvml = C.CDLL("libnvidia-ml.so.1")
            assert self.nvml.nvmlInit_v2() == 0
            self.handle = C.c_void_p()
            assert self.nvml.nvmlDeviceGetHandleByIndex_v2(0, C.byref(self.handle)) == 0
        except (OSError, AssertionError):
            self.nvml = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(.25):
            try:
                rss = self.process.memory_info().rss
                rss += sum(child.memory_info().rss for child in self.process.children(recursive=True))
                self.peak_rss = max(self.peak_rss, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if self.nvml:
                memory = self.Memory()
                if self.nvml.nvmlDeviceGetMemoryInfo(self.handle, C.byref(memory)) == 0:
                    self.peak_device = max(self.peak_device, memory.used)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        if self.nvml:
            self.nvml.nvmlShutdown()


def wait_job(url, job_id, timeout, phases):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = request_json(f"{url}/v1/jobs/{job_id}")
        phases.add((status["state"], status.get("phase"), status.get("progress")))
        if status["state"] in ("complete", "failed", "cancelled"):
            return status
        time.sleep(1)
    raise TimeoutError(f"job {job_id} did not finish in {timeout:g}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true",
                        help="Also run the pinned PyTorch comparison")
    parser.add_argument("--skip-cancel", action="store_true")
    parser.add_argument("--explicit-mask", action="store_true",
                        help="Upload the pinned image alpha as an explicit mask")
    parser.add_argument("--timeout", type=float, default=2400)
    parser.add_argument("--vram-budget-mib", type=int, default=12288)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "tmp/pixal3d/real-web-cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    public = urlopen(IMAGE_URL, timeout=60).read()
    digest = hashlib.sha256(public).hexdigest()
    if digest != IMAGE_SHA256:
        raise RuntimeError(f"public image SHA-256 changed: {digest}")
    source = Image.open(io.BytesIO(public)).convert("RGBA")
    opaque = io.BytesIO()
    source.convert("RGB").save(opaque, "PNG")
    image = opaque.getvalue()
    mask_buffer = io.BytesIO()
    source.getchannel("A").save(mask_buffer, "PNG")
    mask = mask_buffer.getvalue()

    options = argparse.Namespace(
        binary=str(ROOT / "cpu/pixal3d/pixal3d"),
        model_dir=str(app.DEFAULT_MODEL_DIR), dinov3=str(app.DEFAULT_DINOV3),
        naf=str(app.DEFAULT_NAF), rembg=str(app.DEFAULT_RMBG), moge=str(app.DEFAULT_MOGE),
        work_dir=str(args.output_dir / "server"), backend="cuda",
        gpu_execution="resident", gpu_kernels="auto", gpu_flow_precision="mixed",
        threads=0, timeout=args.timeout, reference_timeout=args.timeout)
    pixal = app.PixalServer(options)
    server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
    server.pixal = pixal
    server.uploads = app.UploadStore(pixal.work_dir / "uploads", retained=8)
    server.jobs = app.JobQueue(pixal, retained=4, uploads=server.uploads)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    phases = set()
    started = time.monotonic()
    try:
        health = request_json(url + "/health")
        if not (health["backends"]["cuda"]["available"] and
                health["backends"]["cuda"]["models_ready"] and
                health["preparation"]["mask_ready"]):
            raise RuntimeError(f"CUDA/RMBG health is not ready: {health}")

        body = {"backend": "cuda", "image_ext": ".png",
                "auto_mask": not args.explicit_mask,
                "fov": .857556, "seed": 1, "gpu_execution": "resident",
                "gpu_kernels": "auto", "gpu_flow_precision": "mixed",
                "vram_budget_mib": args.vram_budget_mib,
                "texture_size": 1024, "triangle_target": 100000,
                "reference": args.reference}

        if not args.skip_cancel:
            cancelled_body = dict(body, image_upload=upload(url, image), reference=False)
            if args.explicit_mask:
                cancelled_body["mask_upload"] = upload(url, mask)
            cancelled = request_json(url + "/v1/jobs", "POST", cancelled_body)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                status = request_json(f"{url}/v1/jobs/{cancelled['id']}")
                if status["state"] == "running":
                    break
                time.sleep(.1)
            request_json(f"{url}/v1/jobs/{cancelled['id']}", "DELETE")
            status = wait_job(url, cancelled["id"], 120, phases)
            if status["state"] != "cancelled":
                raise RuntimeError(f"real cancellation failed: {status}")

        body["image_upload"] = upload(url, image)
        if args.explicit_mask:
            body["mask_upload"] = upload(url, mask)
        submitted = request_json(url + "/v1/jobs", "POST", body)
        with Monitor() as monitor:
            status = wait_job(url, submitted["id"], args.timeout, phases)
        if status["state"] != "complete":
            raise RuntimeError(f"real CUDA job failed: {status}")
        result = request_json(f"{url}/v1/jobs/{submitted['id']}/result")["result"]
        expected_mask_source = "mask" if args.explicit_mask else "rmbg-2.0"
        if result.get("preparation", {}).get("mask_source") != expected_mask_source:
            raise RuntimeError(
                f"unexpected mask preparation: {result.get('preparation')}")
        native_url = url + result["artifacts"]["glb"]
        native_path = args.output_dir / "native.glb"
        native_path.write_bytes(urlopen(native_url, timeout=60).read())
        subprocess.run([sys.executable, str(ROOT / "ref/pixal3d/validate_glb.py"),
                        str(native_path)], check=True)
        reference_path = None
        if args.reference:
            reference_path = args.output_dir / "reference.glb"
            reference_path.write_bytes(urlopen(
                url + result["reference"]["artifacts"]["glb"], timeout=60).read())
            subprocess.run([sys.executable, str(ROOT / "ref/pixal3d/validate_glb.py"),
                            str(reference_path), "--allow-material-red"], check=True)
            surface = result.get("comparison", {}).get("surface", {})
            if not surface.get("available"):
                raise RuntimeError(f"surface comparison unavailable: {surface}")

        record = {"image_url": IMAGE_URL, "image_sha256": digest,
                  "input_mode": "explicit-mask" if args.explicit_mask else "automatic-rmbg",
                  "job_id": submitted["id"], "seconds": time.monotonic() - started,
                  "peak_host_rss": monitor.peak_rss,
                  "peak_device_bytes": monitor.peak_device,
                  "phases": sorted(phases, key=str), "result": result,
                  "native_glb": str(native_path),
                  "reference_glb": str(reference_path) if reference_path else None}
        (args.output_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        print("Pixal3D real queued HTTP/CUDA test: PASS")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
