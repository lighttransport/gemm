#!/usr/bin/env python3
"""SAM 3D Body compare server.

One HTTP service that dispatches /v1/infer across backends for
single-image full-body 3D mesh recovery:

  - "ours-cpu":  shells out to cpu/sam3d_body/test_sam3d_body
  - "ours-cuda": shells out to cuda/sam3d_body/test_cuda_sam3d_body

A pytorch-cuda backend that calls into ref/sam3d-body/.venv lands in A3.2.

Request shape:

    {
      "inputs": {
        "image_base64": "...",           # required (PNG/JPG)
        "bbox": [x0, y0, x1, y1],        # optional; else auto-detect
      },
      "params": {
        "backend": "ours-cpu|ours-cuda",
        "auto_bbox": true,               # default true if bbox missing
        "auto_thresh": 0.5,
        "focal": 0.0                     # optional
      }
    }

Response:

    {
      "ok": true,
      "backend": "ours-cuda",
      "outputs": [{
        "type": "mesh",
        "mime": "model/obj",
        "n_verts": 18439,
        "n_faces": 36874,
        "data_base64": "<OBJ text base64>",
        "bbox_used": [x0, y0, x1, y1]
      }],
      "timings_ms": {"infer": 8770, "total": 8800}
    }
"""
import argparse
import base64
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


def _cors(h):
    h.send_header("access-control-allow-origin", "*")
    h.send_header("access-control-allow-methods", "POST, GET, OPTIONS")
    h.send_header("access-control-allow-headers", "content-type")


def _json(h, status, payload):
    body = json.dumps(payload).encode("utf-8")
    h.send_response(status)
    h.send_header("content-type", "application/json")
    h.send_header("content-length", str(len(body)))
    _cors(h)
    h.end_headers()
    h.wfile.write(body)


def _save_image(b64, path):
    raw = base64.b64decode(b64)
    if not raw:
        raise ValueError("empty image_base64")
    with open(path, "wb") as f:
        f.write(raw)


def _read_obj_counts(path):
    nv = nf = 0
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"v "):
                nv += 1
            elif line.startswith(b"f "):
                nf += 1
    return nv, nf


_BBOX_PAREN_RE = re.compile(
    r"bbox=\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")


def _parse_bbox_log(stderr):
    """Parse the runner's auto-bbox log line.
    CPU  emits: 'auto-bbox: score=0.980 bbox=(0.1,5.9,768.1,1019.4)'
    CUDA emits: 'auto-bbox score=0.9797 x0=0.1 y0=5.9 x1=768.1 y1=1019.4'
    """
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


# ---- Backends ------------------------------------------------------------

class CRunnerBackend:
    """Common shell-out for CPU and CUDA runners.

    Both binaries share argv:
        <bin> <safetensors-dir> <image>
              --mhr-assets DIR
              [--bbox x0 y0 x1 y1 | --auto-bbox [--rt-detr-model PATH] [--auto-thresh F]]
              [--focal F] [-o body.obj] [-v]
    """

    def __init__(self, name, bin_path, sft_dir, mhr_dir, rt_detr_model):
        self.name = name
        self.bin = os.path.abspath(bin_path)
        self.sft_dir = os.path.abspath(sft_dir)
        self.mhr_dir = os.path.abspath(mhr_dir)
        self.rt_detr_model = (os.path.abspath(rt_detr_model)
                              if rt_detr_model else None)
        self.lock = threading.Lock()

    def available(self):
        return (os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)
                and os.path.isdir(self.sft_dir) and os.path.isdir(self.mhr_dir))

    def infer(self, image_b64, params, bbox):
        with self.lock, tempfile.TemporaryDirectory() as td:
            ext = (params.get("image_ext") or "jpg").lstrip(".")
            img = os.path.join(td, f"in.{ext}")
            obj = os.path.join(td, "out.obj")
            _save_image(image_b64, img)

            cmd = [self.bin, self.sft_dir, img,
                   "--mhr-assets", self.mhr_dir, "-o", obj, "-v"]
            if bbox is not None:
                cmd += ["--bbox", f"{bbox[0]:g}", f"{bbox[1]:g}",
                                  f"{bbox[2]:g}", f"{bbox[3]:g}"]
            else:
                cmd += ["--auto-bbox"]
                if self.rt_detr_model:
                    cmd += ["--rt-detr-model", self.rt_detr_model]
                if params.get("auto_thresh") is not None:
                    cmd += ["--auto-thresh", f"{float(params['auto_thresh']):g}"]
            if params.get("focal"):
                cmd += ["--focal", f"{float(params['focal']):g}"]

            backbone = (params.get("backbone") or "dinov3").lower()
            if backbone not in ("dinov3", "vith"):
                raise RuntimeError(
                    f"unknown backbone {backbone!r} (expected dinov3|vith)")
            if backbone != "dinov3":
                cmd += ["--backbone", backbone]

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(obj):
                raise RuntimeError(
                    f"{self.name} failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")

            nv, nf = _read_obj_counts(obj)
            with open(obj, "rb") as f:
                obj_bytes = f.read()
            bbox_used = bbox if bbox is not None else _parse_bbox_log(proc.stderr)
        return obj_bytes, nv, nf, dt_ms, bbox_used


class PytorchShellBackend:
    """ref/sam3d-body/run_pytorch_pipeline.py via the dedicated venv.

    Bbox is required (no auto-detect): the upstream estimator falls back
    to detectron2 person detection when bboxes=None, which we don't
    bundle. The web client sends a bbox alongside the image — either the
    user-drawn one OR the bbox previously returned by ours-cuda /
    ours-cpu (whichever ran first).
    """

    def __init__(self, script_path, python_exe, ckpt_dirs, device,
                 hf_repo_ids):
        """ckpt_dirs / hf_repo_ids: dicts keyed by backbone variant
        ('dinov3', 'vith'). When the request specifies a backbone, the
        matching ckpt-dir / hf-repo-id is selected; missing entries fall
        back to the dinov3 default."""
        self.script = os.path.abspath(script_path)
        self.python = python_exe
        self.ckpt_dirs = {k: (os.path.abspath(v) if v else None)
                          for k, v in (ckpt_dirs or {}).items()}
        self.device = device
        self.hf_repo_ids = dict(hf_repo_ids or {})
        self.lock = threading.Lock()

    def available(self):
        return (os.path.isfile(self.script)
                and self.python and os.path.isfile(self.python))

    def infer(self, image_b64, params, bbox):
        if bbox is None:
            raise RuntimeError(
                "pytorch backend requires a bbox (no detectron2 in venv). "
                "Run ours-cuda or ours-cpu first to get an auto-bbox, then "
                "submit again with that bbox.")
        with self.lock, tempfile.TemporaryDirectory() as td:
            ext = (params.get("image_ext") or "jpg").lstrip(".")
            img = os.path.join(td, f"in.{ext}")
            obj = os.path.join(td, "out.obj")
            _save_image(image_b64, img)

            backbone = (params.get("backbone") or "dinov3").lower()
            if backbone not in ("dinov3", "vith"):
                raise RuntimeError(
                    f"unknown backbone {backbone!r} (expected dinov3|vith)")
            ckpt_dir = self.ckpt_dirs.get(backbone) or self.ckpt_dirs.get("dinov3")
            hf_repo  = self.hf_repo_ids.get(backbone) or self.hf_repo_ids.get("dinov3")

            cmd = [self.python, self.script,
                   "--image", img, "--out", obj,
                   "--bbox", f"{bbox[0]:g}", f"{bbox[1]:g}",
                             f"{bbox[2]:g}", f"{bbox[3]:g}",
                   "--device", self.device]
            if ckpt_dir:
                cmd += ["--local-ckpt-dir", ckpt_dir]
            elif hf_repo:
                cmd += ["--hf-repo-id", hf_repo]

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(obj):
                raise RuntimeError(
                    f"pytorch failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")

            nv, nf = _read_obj_counts(obj)
            with open(obj, "rb") as f:
                obj_bytes = f.read()
        return obj_bytes, nv, nf, dt_ms, bbox


# ---- HTTP dispatch -------------------------------------------------------

_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp",
    ".gif": "image/gif",  ".txt": "text/plain; charset=utf-8",
    ".html": "text/html; charset=utf-8",
}


def _list_samples(samples_dir):
    if not os.path.isdir(samples_dir):
        return []
    out = []
    for name in sorted(os.listdir(samples_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        path = os.path.join(samples_dir, name)
        try:
            sz = os.path.getsize(path)
        except OSError:
            continue
        out.append({"name": name,
                    "url": "/public/sam3d_body/" + name,
                    "size": sz})
    return out


def make_handler(backends, web_root):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[sam3d_body] " + (fmt % a) + "\n")

        def do_OPTIONS(self):
            self.send_response(204); _cors(self); self.end_headers()

        def _serve_file(self, path, ctype):
            try:
                with open(path, "rb") as f: data = f.read()
            except OSError:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            _cors(self); self.end_headers()
            self.wfile.write(data)

        def _serve_static(self, sub):
            """Serve files from <web_root>/public/<sub>. Refuses any path
            that escapes the public dir (defends against ../ traversal)."""
            base = os.path.realpath(os.path.join(web_root, "public"))
            full = os.path.realpath(os.path.join(base, sub))
            if not full.startswith(base + os.sep) and full != base:
                self.send_error(403); return
            if not os.path.isfile(full):
                self.send_error(404); return
            ctype = _MIME.get(os.path.splitext(full)[1].lower(),
                              "application/octet-stream")
            self._serve_file(full, ctype)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/sam3d_body", "/sam3d_body.html"):
                self._serve_file(os.path.join(web_root, "sam3d_body.html"),
                                 "text/html; charset=utf-8")
                return
            if path.startswith("/public/"):
                self._serve_static(path[len("/public/"):])
                return
            if path == "/health":
                _json(self, 200, {"ok": True,
                                  "backends": sorted(backends.keys())})
                return
            if path in ("/models", "/v1/models"):
                _json(self, 200, {"ok": True, "models": [
                    {"id": "sam3d_body", "tasks": ["image-to-mesh"],
                     "backends": sorted(backends.keys())}
                ]})
                return
            if path == "/v1/samples":
                samples_dir = os.path.join(web_root, "public", "sam3d_body")
                _json(self, 200, {"ok": True,
                                  "samples": _list_samples(samples_dir)})
                return
            self.send_error(404)

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/v1/infer":
                self.send_error(404); return
            try:
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n) if n else b""
                req = json.loads(raw.decode("utf-8"))
            except Exception as e:
                _json(self, 400, {"ok": False, "error": f"bad json: {e}"}); return

            t_all = time.time()
            try:
                inputs = req.get("inputs") or {}
                params = req.get("params") or {}
                img_b64 = inputs.get("image_base64")
                if not img_b64:
                    _json(self, 400, {"ok": False,
                                      "error": "inputs.image_base64 required"}); return
                be_name = str(params.get("backend") or "").strip()
                if be_name not in backends:
                    _json(self, 400, {"ok": False,
                                      "error": f"unknown backend '{be_name}'; "
                                               f"have {sorted(backends.keys())}"}); return

                bbox_in = inputs.get("bbox")
                bbox = None
                if bbox_in is not None:
                    if (not isinstance(bbox_in, list)) or len(bbox_in) != 4:
                        _json(self, 400, {"ok": False,
                                          "error": "bbox must be [x0,y0,x1,y1]"}); return
                    bbox = [float(x) for x in bbox_in]

                be = backends[be_name]
                obj_bytes, nv, nf, dt_infer, bbox_used = be.infer(
                    img_b64, params, bbox)

                dt_ms = int((time.time() - t_all) * 1000)
                _json(self, 200, {
                    "ok": True,
                    "model": "sam3d_body",
                    "task": "image-to-mesh",
                    "backend": be_name,
                    "outputs": [{
                        "type": "mesh",
                        "mime": "model/obj",
                        "n_verts": int(nv),
                        "n_faces": int(nf),
                        "data_base64": base64.b64encode(obj_bytes).decode("ascii"),
                        "bbox_used": bbox_used,
                    }],
                    "timings_ms": {"infer": int(dt_infer), "total": dt_ms},
                })
            except Exception as e:
                traceback.print_exc()
                _json(self, 500, {"ok": False, "error": str(e)})

    return H


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    default_sft = "/mnt/disk01/models/sam3d-body/safetensors"
    default_rt_detr = "/mnt/disk01/models/rt_detr_s/model.safetensors"

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--cpu-bin",
                    default=os.path.join(repo_root, "cpu", "sam3d_body",
                                         "test_sam3d_body"))
    ap.add_argument("--cuda-bin",
                    default=os.path.join(repo_root, "cuda", "sam3d_body",
                                         "test_cuda_sam3d_body"))
    ap.add_argument("--safetensors-dir", default=default_sft,
                    help="sam3d_body safetensors dir (also used as --mhr-assets)")
    ap.add_argument("--mhr-assets", default=None,
                    help="MHR assets dir; defaults to --safetensors-dir")
    ap.add_argument("--rt-detr-model", default=default_rt_detr,
                    help="RT-DETR-S safetensors used by --auto-bbox")
    ap.add_argument("--web-root", default=os.path.join(repo_root, "web"))
    ap.add_argument("--disable", default="",
                    help="comma-sep list of backends to skip: "
                         "ours-cpu,ours-cuda,pytorch")
    ap.add_argument("--ref-script",
                    default=os.path.join(here, "run_pytorch_pipeline.py"))
    ap.add_argument("--ref-python",
                    default=os.path.join(here, ".venv", "bin", "python"),
                    help="python interpreter for pytorch shell-out "
                         "(default: ref/sam3d-body/.venv/bin/python)")
    ap.add_argument("--ref-ckpt-dir",
                    default="/mnt/disk01/models/sam3d-body/dinov3",
                    help="local sam-3d-body snapshot dir for the pytorch "
                         "backend (dinov3 variant; overrides HF download)")
    ap.add_argument("--ref-ckpt-dir-vith",
                    default="/mnt/disk01/models/sam3d-body/vith",
                    help="local sam-3d-body snapshot dir for the pytorch "
                         "backend (vit-h variant)")
    ap.add_argument("--ref-device", default="cuda")
    ap.add_argument("--ref-hf-repo-id", default="facebook/sam-3d-body-dinov3")
    ap.add_argument("--ref-hf-repo-id-vith",
                    default="facebook/sam-3d-body-vith")
    args = ap.parse_args()

    sft = os.path.abspath(args.safetensors_dir)
    mhr = os.path.abspath(args.mhr_assets or args.safetensors_dir)

    disabled = {x.strip() for x in args.disable.split(",") if x.strip()}
    backends = {}
    if "ours-cpu" not in disabled:
        be = CRunnerBackend("ours-cpu", args.cpu_bin, sft, mhr,
                            args.rt_detr_model)
        if be.available():
            backends["ours-cpu"] = be
        else:
            print(f"[sam3d_body] ours-cpu disabled "
                  f"(bin={args.cpu_bin} sft={sft} mhr={mhr})", file=sys.stderr)
    if "ours-cuda" not in disabled:
        be = CRunnerBackend("ours-cuda", args.cuda_bin, sft, mhr,
                            args.rt_detr_model)
        if be.available():
            backends["ours-cuda"] = be
        else:
            print(f"[sam3d_body] ours-cuda disabled "
                  f"(bin={args.cuda_bin})", file=sys.stderr)
    if "pytorch" not in disabled:
        ckpt_dirs = {"dinov3": args.ref_ckpt_dir,
                     "vith":   args.ref_ckpt_dir_vith}
        hf_ids    = {"dinov3": args.ref_hf_repo_id,
                     "vith":   args.ref_hf_repo_id_vith}
        be = PytorchShellBackend(args.ref_script, args.ref_python,
                                 ckpt_dirs, args.ref_device,
                                 hf_ids)
        if be.available():
            backends["pytorch-cuda"] = be
        else:
            print(f"[sam3d_body] pytorch-cuda disabled "
                  f"(script={args.ref_script} python={args.ref_python})",
                  file=sys.stderr)

    if not backends:
        sys.exit("no backends enabled")

    web_root = os.path.abspath(args.web_root)
    Handler = make_handler(backends, web_root)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[sam3d_body] listening on http://{args.host}:{args.port}  "
          f"backends={sorted(backends.keys())}  web_root={web_root}",
          flush=True)
    print(f"[sam3d_body] sft={sft}  mhr={mhr}  "
          f"rt_detr={args.rt_detr_model}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
