#!/usr/bin/env python3
"""SAM 3D Body 3-pane demo server.

Single endpoint /v1/infer runs both backends in parallel and returns a
combined response so the web UI can show:
  - input image (client-side from upload)
  - ours mesh   (C runner subprocess)
  - pytorch mesh (in-process via pytorch_runner)

Request:
    {"image_b64": "...", "bbox": [x0,y0,x1,y1] | null,
     "auto_bbox": true,  "auto_thresh": 0.5,
     "backends": ["ours","pytorch"]}     # optional; default both

Response:
    {"ours":     {"obj_b64","json","timing_ms"} | null,
     "pytorch":  {"obj_b64","json","timing_ms"} | null,
     "errors":   {"ours"?: str, "pytorch"?: str},
     "bbox_used": [..]   # echoed from ours backend if auto_bbox was used
     "timings_ms": {"total": ...}}
"""
import argparse
import base64
import concurrent.futures as cf
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _cors(h):
    h.send_header("access-control-allow-origin", "*")
    h.send_header("access-control-allow-methods", "POST, GET, OPTIONS")
    h.send_header("access-control-allow-headers", "content-type")


def _send_json(h, status, payload):
    body = json.dumps(payload).encode("utf-8")
    h.send_response(status)
    h.send_header("content-type", "application/json")
    h.send_header("content-length", str(len(body)))
    _cors(h)
    h.end_headers()
    h.wfile.write(body)


def _send_file(h, path, ctype):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        h.send_error(404); return
    h.send_response(200)
    h.send_header("content-type", ctype)
    h.send_header("content-length", str(len(data)))
    _cors(h); h.end_headers()
    h.wfile.write(data)


_BBOX_PAREN_RE = re.compile(
    r"bbox=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")


def _parse_bbox_from_log(stderr):
    for line in stderr.splitlines():
        if "auto-bbox" not in line:
            continue
        m = _BBOX_PAREN_RE.search(line)
        if m:
            return [float(m.group(i)) for i in (1, 2, 3, 4)]
        d = {}
        for t in line.split():
            if "=" in t:
                k, v = t.split("=", 1)
                try: d[k] = float(v)
                except ValueError: pass
        if {"x0", "y0", "x1", "y1"}.issubset(d):
            return [d["x0"], d["y0"], d["x1"], d["y1"]]
    return None


# --------------------------------------------------------------------------
# Ours backend: cpu/sam3d_body/test_sam3d_body subprocess
# --------------------------------------------------------------------------

class OursBackend:
    def __init__(self, bin_path, sft_dir, mhr_dir, rt_detr_model):
        self.bin = os.path.abspath(bin_path)
        self.sft_dir = os.path.abspath(sft_dir)
        self.mhr_dir = os.path.abspath(mhr_dir)
        self.rt_detr_model = (os.path.abspath(rt_detr_model)
                              if rt_detr_model else None)
        self.lock = threading.Lock()

    def available(self):
        return (os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)
                and os.path.isdir(self.sft_dir)
                and os.path.isdir(self.mhr_dir))

    def infer(self, img_bytes, image_ext, bbox, auto_thresh):
        with self.lock, tempfile.TemporaryDirectory() as td:
            img = os.path.join(td, f"in.{image_ext.lstrip('.') or 'jpg'}")
            obj = os.path.join(td, "out.obj")
            with open(img, "wb") as f:
                f.write(img_bytes)
            cmd = [self.bin,
                   "--safetensors-dir", self.sft_dir,
                   "--mhr-assets",      self.mhr_dir,
                   "--image", img, "-o", obj, "-v"]
            if bbox is not None:
                cmd += ["--bbox", *(f"{v:g}" for v in bbox)]
            else:
                cmd += ["--auto-bbox"]
                if self.rt_detr_model:
                    cmd += ["--rt-detr-model", self.rt_detr_model]
                if auto_thresh is not None:
                    cmd += ["--auto-thresh", f"{float(auto_thresh):g}"]

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(obj):
                raise RuntimeError(
                    f"ours runner failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            with open(obj, "rb") as f: obj_bytes = f.read()
            side_path = obj + ".json"
            side = None
            if os.path.isfile(side_path):
                with open(side_path) as f: side = json.load(f)
            bbox_used = bbox if bbox is not None else _parse_bbox_from_log(proc.stderr)
            if side and "bbox" in side and bbox_used is None:
                bbox_used = side["bbox"]
        return {
            "obj_b64":   base64.b64encode(obj_bytes).decode("ascii"),
            "json":      side or {},
            "timing_ms": dt_ms,
            "bbox_used": bbox_used,
        }


# --------------------------------------------------------------------------
# Pytorch backend: in-process via server/sam3d/pytorch_runner.py
# --------------------------------------------------------------------------

class PytorchBackend:
    def __init__(self):
        self._loaded = False

    def load(self, local_ckpt_dir, hf_repo_id, device):
        import pytorch_runner
        pytorch_runner.load(local_ckpt_dir=local_ckpt_dir,
                            hf_repo_id=hf_repo_id, device=device)
        self._loaded = True

    def available(self):
        return self._loaded

    def infer(self, img_bytes, bbox):
        from PIL import Image
        import numpy as np
        import pytorch_runner
        img_rgb = np.asarray(
            Image.open(io.BytesIO(img_bytes)).convert("RGB"), dtype=np.uint8)
        if bbox is None:
            raise RuntimeError(
                "pytorch backend requires a bbox (no detector bundled). "
                "Enable auto-bbox so the ours runner returns a bbox, or "
                "draw one on the preview.")
        t0 = time.time()
        r = pytorch_runner.infer(img_rgb, bbox_xyxy=bbox)
        dt_ms = int((time.time() - t0) * 1000)
        return {
            "obj_b64":   base64.b64encode(r["obj_text"].encode("utf-8")).decode("ascii"),
            "json":      r["json"],
            "timing_ms": dt_ms,
            "bbox_used": bbox,
        }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def make_handler(ours, torch_be, web_root, demo_html):
    POOL = cf.ThreadPoolExecutor(max_workers=4)

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[sam3d/app] " + (fmt % a) + "\n")

        def do_OPTIONS(self):
            self.send_response(204); _cors(self); self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/sam3d_demo", "/sam3d_demo.html"):
                _send_file(self, demo_html, "text/html; charset=utf-8")
                return
            if path == "/health":
                _send_json(self, 200, {
                    "ok": True,
                    "backends": {
                        "ours":    bool(ours and ours.available()),
                        "pytorch": bool(torch_be and torch_be.available()),
                    },
                })
                return
            if path.startswith("/web/"):
                rel = path[len("/web/"):]
                full = os.path.realpath(os.path.join(web_root, rel))
                base = os.path.realpath(web_root)
                if not full.startswith(base + os.sep):
                    self.send_error(403); return
                ext = os.path.splitext(full)[1].lower()
                ctype = {".html": "text/html; charset=utf-8",
                         ".js":   "application/javascript",
                         ".css":  "text/css",
                         ".png":  "image/png",
                         ".jpg":  "image/jpeg",
                         ".jpeg": "image/jpeg",
                         }.get(ext, "application/octet-stream")
                _send_file(self, full, ctype); return
            self.send_error(404)

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/v1/infer":
                self.send_error(404); return
            try:
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n) if n else b""
                req = json.loads(raw.decode("utf-8"))
            except Exception as e:
                _send_json(self, 400, {"error": f"bad json: {e}"}); return

            t_all = time.time()
            try:
                img_b64 = req.get("image_b64")
                if not img_b64:
                    _send_json(self, 400, {"error": "image_b64 required"}); return
                try:
                    img_bytes = base64.b64decode(img_b64)
                except Exception as e:
                    _send_json(self, 400, {"error": f"bad image_b64: {e}"}); return
                bbox = req.get("bbox")
                if bbox is not None:
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        _send_json(self, 400,
                                    {"error": "bbox must be [x0,y0,x1,y1]"}); return
                    bbox = [float(x) for x in bbox]
                image_ext  = req.get("image_ext") or "jpg"
                auto_thresh = req.get("auto_thresh", 0.5)
                want = set(req.get("backends") or ["ours", "pytorch"])

                fut_ours = None
                if "ours" in want and ours and ours.available():
                    fut_ours = POOL.submit(ours.infer, img_bytes, image_ext,
                                           bbox, auto_thresh)
                # We may need bbox from ours for pytorch backend; if no bbox
                # was provided and ours is running, wait for it first.
                fut_torch = None
                if "pytorch" in want and torch_be and torch_be.available():
                    if bbox is None and fut_ours is not None:
                        # serial: ours must complete first to provide bbox
                        pass
                    else:
                        fut_torch = POOL.submit(torch_be.infer, img_bytes, bbox)

                resp = {"ours": None, "pytorch": None,
                        "errors": {}, "bbox_used": bbox}

                if fut_ours is not None:
                    try:
                        resp["ours"] = fut_ours.result()
                        if resp["bbox_used"] is None:
                            resp["bbox_used"] = resp["ours"].get("bbox_used")
                    except Exception as e:
                        resp["errors"]["ours"] = str(e)
                elif "ours" in want:
                    resp["errors"]["ours"] = "ours backend unavailable"

                # Now the pytorch backend, possibly using bbox from ours.
                if "pytorch" in want and fut_torch is None:
                    if torch_be and torch_be.available():
                        try:
                            pyb = resp["bbox_used"]
                            resp["pytorch"] = torch_be.infer(img_bytes, pyb)
                        except Exception as e:
                            resp["errors"]["pytorch"] = str(e)
                    else:
                        resp["errors"]["pytorch"] = "pytorch backend unavailable"
                elif fut_torch is not None:
                    try:
                        resp["pytorch"] = fut_torch.result()
                    except Exception as e:
                        resp["errors"]["pytorch"] = str(e)

                resp["timings_ms"] = {"total": int((time.time() - t_all) * 1000)}
                _send_json(self, 200, resp)
            except Exception as e:
                traceback.print_exc()
                _send_json(self, 500, {"error": str(e)})

    return H


def main():
    default_sft = "/mnt/disk01/models/sam3d-body/safetensors"
    default_rt_detr = "/mnt/disk01/models/rt_detr_s/model.safetensors"
    default_pytorch_ckpt = "/mnt/disk01/models/sam3d-body/dinov3"

    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--c-runner",
                    default=os.path.join(REPO_ROOT, "cpu", "sam3d_body",
                                          "test_sam3d_body"))
    ap.add_argument("--safetensors-dir", default=default_sft)
    ap.add_argument("--mhr-assets", default=None)
    ap.add_argument("--rt-detr-model", default=default_rt_detr)
    ap.add_argument("--pytorch-ckpt", default=default_pytorch_ckpt,
                    help="local sam-3d-body snapshot dir (overrides --hf-repo-id)")
    ap.add_argument("--hf-repo-id", default="facebook/sam-3d-body-dinov3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pytorch", action="store_true")
    ap.add_argument("--web-root",
                    default=os.path.join(REPO_ROOT, "web"))
    ap.add_argument("--demo-html",
                    default=os.path.join(REPO_ROOT, "web", "sam3d_demo.html"))
    args = ap.parse_args()

    sft = os.path.abspath(args.safetensors_dir)
    mhr = os.path.abspath(args.mhr_assets or args.safetensors_dir)
    web_root = os.path.abspath(args.web_root)

    ours = OursBackend(args.c_runner, sft, mhr, args.rt_detr_model)
    if not ours.available():
        print(f"[sam3d/app] WARN: ours backend unavailable "
              f"(bin={args.c_runner}, sft={sft}, mhr={mhr})", file=sys.stderr)
        ours = None

    torch_be = None
    if not args.no_pytorch:
        sys.path.insert(0, HERE)  # for `import pytorch_runner`
        torch_be = PytorchBackend()
        try:
            ckpt = args.pytorch_ckpt if os.path.isdir(args.pytorch_ckpt) else None
            torch_be.load(local_ckpt_dir=ckpt,
                          hf_repo_id=args.hf_repo_id, device=args.device)
        except Exception as e:
            print(f"[sam3d/app] WARN: pytorch backend load failed: {e}",
                  file=sys.stderr)
            traceback.print_exc()
            torch_be = None

    if not ours and not torch_be:
        sys.exit("[sam3d/app] no backends available; aborting")

    Handler = make_handler(ours, torch_be, web_root, args.demo_html)
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"[sam3d/app] listening on http://{args.bind}:{args.port}  "
          f"ours={'on' if ours else 'off'}  "
          f"pytorch={'on' if torch_be else 'off'}",
          flush=True)
    print(f"[sam3d/app] sft={sft} mhr={mhr} demo={args.demo_html}",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
