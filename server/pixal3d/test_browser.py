#!/usr/bin/env python3
"""Functional headless-Chrome test for the Pixal3D web demo."""
import argparse
import base64
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from server.pixal3d import app


class FakePixal:
    args = argparse.Namespace(backend="cuda", gpu_execution="resident",
                              gpu_kernels="auto", gpu_flow_precision="mixed")

    def __init__(self):
        self.requests = []

    def health(self):
        ready = {"available": True, "models_ready": True, "multiview_ready": True}
        reference = {"available": True}
        return {"ok": True, "service": "pixal3d", "default_backend": "cuda",
                "default_gpu_execution": "resident", "default_gpu_kernels": "auto",
                "default_gpu_flow_precision": "mixed",
                "preparation": {"mask_ready": True, "camera_ready": True},
                "reference": {"cuda": reference, "rocm": reference},
                "backends": {name: ready for name in ("cpu", "cuda", "rocm")}}

    def infer(self, request, cancel=None, progress=None):
        self.requests.append(request)
        if request.get("seed") == 999:
            while not cancel.wait(0.01):
                pass
            raise app.JobCancelled("job cancelled")
        if progress:
            progress("Pixal3D texture: step 12/12")
        return {"ok": True, "backend": request["backend"], "elapsed_ms": 1,
                "glb_b64": base64.b64encode(b"glTF-browser-test").decode(),
                "ply_b64": base64.b64encode(b"ply-browser-test").decode(),
                "stats": {"views": len(request.get("views", [None]))}, "profile": {},
                "mesh_summary": {"vertices": 100, "triangles": 80,
                                 "bounds": [[-.5, -.5, -.5], [.5, .5, .5]]}}

    def reference(self, request, cancel=None):
        return {"backend": request["backend"], "elapsed_ms": 1,
                "glb_b64": base64.b64encode(b"glTF-reference-test").decode(),
                "mesh_summary": {"vertices": 99, "triangles": 79,
                                 "bounds": [[-.5, -.5, -.5], [.5, .5, .5]]}}

    def surface_comparison(self, native, reference, cancel=None):
        return {"available": True, "samples": 50000, "seed": 17,
                "symmetric_chamfer_rms": .01,
                "native_to_reference": {"p95": .02, "normal_abs_cosine_mean": .98},
                "reference_to_native": {"p95": .03, "normal_abs_cosine_mean": .97}}


class Cdp:
    """Minimal WebSocket client for the small CDP surface used by this test."""
    def __init__(self, url):
        host, rest = url.removeprefix("ws://").split("/", 1)
        hostname, port = host.split(":")
        self.sock = socket.create_connection((hostname, int(port)), timeout=10)
        key = base64.b64encode(b"pixal3d-browser-test").decode()
        request = (f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(request.encode())
        response = b""
        while b"\r\n\r\n" not in response:
            response += self.sock.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(response.decode(errors="replace"))
        self.next_id = 1

    def _read_exact(self, length):
        data = b""
        while len(data) < length:
            data += self.sock.recv(length - len(data))
        return data

    def _read(self):
        first, second = self._read_exact(2)
        length = second & 0x7f
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        data = self._read_exact(length)
        if (first & 0x0f) == 8:
            raise EOFError("Chrome closed the DevTools socket")
        return json.loads(data)

    def call(self, method, params=None):
        call_id = self.next_id
        self.next_id += 1
        payload = json.dumps({"id": call_id, "method": method,
                              "params": params or {}}).encode()
        mask = b"P3DT"
        encoded = bytes(value ^ mask[i % 4] for i, value in enumerate(payload))
        if len(payload) < 126:
            header = bytearray([0x81, 0x80 | len(payload)])
        else:
            header = bytearray([0x81, 0x80 | 126]) + struct.pack("!H", len(payload))
        self.sock.sendall(header + mask + encoded)
        while True:
            message = self._read()
            if message.get("id") == call_id:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression,
                           "awaitPromise": True, "returnByValue": True})
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def set_files(self, selector, paths):
        document = self.call("DOM.getDocument")
        node = self.call("DOM.querySelector", {"nodeId": document["root"]["nodeId"],
                                                "selector": selector})
        self.call("DOM.setFileInputFiles", {"nodeId": node["nodeId"], "files": paths})

    def close(self):
        self.sock.close()


def wait_for(cdp, expression, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = cdp.evaluate(expression)
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"browser condition timed out: {expression}")


def main():
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        raise SystemExit("Chrome/Chromium is required")
    pixal = FakePixal()
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    server.pixal = pixal
    scratch = app.ROOT / "tmp/pixal3d/browser-test"
    scratch.mkdir(parents=True, exist_ok=True)
    pixal.work_dir = scratch
    server.uploads = app.UploadStore(scratch / "uploads", retained=16)
    server.jobs = app.JobQueue(pixal, retained=4, uploads=server.uploads)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="chrome-", dir=scratch) as profile:
            process = subprocess.Popen(
                [chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                 "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            cdp = None
            try:
                active = Path(profile) / "DevToolsActivePort"
                deadline = time.monotonic() + 10
                while not active.is_file() and time.monotonic() < deadline:
                    time.sleep(0.05)
                port = int(active.read_text().splitlines()[0])
                target = json.loads(urlopen(Request(
                    f"http://127.0.0.1:{port}/json/new",
                    method="PUT"), timeout=5).read())
                cdp = Cdp(target["webSocketDebuggerUrl"])
                cdp.call("Runtime.enable")
                cdp.call("Page.navigate", {"url": f"http://127.0.0.1:{server.server_port}/"})
                wait_for(cdp, "document.readyState === 'complete'")
                wait_for(cdp, "document.getElementById('health').textContent.includes('PyTorch reference')")

                image = scratch / "input.png"
                image.write_bytes(b"browser-image")
                cdp.set_files("#image", [str(image)])
                cdp.evaluate("document.getElementById('include-ply').checked=true; document.getElementById('reference').checked=true; document.getElementById('form').requestSubmit()")
                wait_for(cdp, "document.getElementById('status').textContent === 'Complete'")
                assert cdp.evaluate("!document.getElementById('download').hidden && !document.getElementById('ply-download').hidden && !document.getElementById('reference-download').hidden")
                assert cdp.evaluate("document.getElementById('stats').textContent.includes('Chamfer RMS: 0.010000')")
                assert cdp.evaluate("Promise.all(['download','ply-download','reference-download'].map(id=>fetch(document.getElementById(id).href).then(r=>r.ok&&r.arrayBuffer()).then(b=>b.byteLength))).then(v=>v.every(n=>n>0))")
                assert len(pixal.requests) == 1 and "views" not in pixal.requests[0]

                transforms = scratch / "transforms.json"
                transforms.write_text(json.dumps({"camera_angle_x": 0.8, "frames": [
                    {"file_path": "view0.png", "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]},
                    {"file_path": "view1.png", "transform_matrix": [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}]}))
                view0, view1 = scratch / "view0.png", scratch / "view1.png"
                view0.write_bytes(b"view-zero")
                view1.write_bytes(b"view-one")
                cdp.evaluate("document.querySelector('[data-mode=multi]').click()")
                cdp.set_files("#transforms", [str(transforms)])
                cdp.set_files("#view-images", [str(view0), str(view1)])
                wait_for(cdp, "document.getElementById('view-summary').textContent.includes('2 ordered views')")
                cdp.evaluate("document.getElementById('form').requestSubmit()")
                wait_for(cdp, "document.getElementById('status').textContent === 'Complete'")
                assert len(pixal.requests[-1]["views"]) == 2

                cdp.evaluate("document.getElementById('seed').value='999'; document.getElementById('form').requestSubmit()")
                wait_for(cdp, "!document.getElementById('cancel').hidden")
                cdp.evaluate("document.getElementById('cancel').click()")
                wait_for(cdp, "document.getElementById('status').textContent.includes('cancelled')")
                print("Pixal3D functional browser test: PASS")
            finally:
                if cdp is not None:
                    try:
                        cdp.call("Browser.close")
                    except (EOFError, OSError, RuntimeError):
                        pass
                    cdp.close()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
