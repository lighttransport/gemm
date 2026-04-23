#!/usr/bin/env python3
"""Qwen-Image side-by-side compare server.

One HTTP service that dispatches /v1/infer across backends for a
text-to-image prompt:

  - "ours-cpu":     shells out to cpu/qwen_image/test_qwen_image --generate <dit> <vae> <enc>
  - "ours-cuda":    (advertised only if cuda/qwen_image/test_cuda_qwen_image exists)
  - "pytorch-cuda": shells out to ref/qwen_image/gen_diffusers_reference.py

Variants register {dit, vae, text_encoder} triplets by convention.
Each variant directory is expected to contain::

    <dir>/diffusion_models/*.safetensors  OR  <dir>/diffusion-models/*.gguf
    <dir>/vae/*.safetensors
    <dir>/text_encoders/*.safetensors      OR  <dir>/text-encoder/*.gguf

Request shape mirrors the C server (server/server.c):

    {
      "inputs": {"text": "<prompt>", "negative_text": "..."},
      "params": {
        "backend": "ours-cpu|ours-cuda|pytorch-cuda",
        "variant": "<name>",
        "width": 256, "height": 256,
        "steps": 20, "seed": 42
      }
    }

Response: PNG base64 + timings.
"""
import argparse
import base64
import glob
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


def _cors_headers(handler):
    handler.send_header("access-control-allow-origin", "*")
    handler.send_header("access-control-allow-methods", "POST, GET, OPTIONS")
    handler.send_header("access-control-allow-headers", "content-type")


def _json(handler, status, payload):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(body)))
    _cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(body)


# ---- Variant triplet discovery -------------------------------------------

def _first_match(root, *globs):
    for g in globs:
        hits = sorted(glob.glob(os.path.join(root, g)))
        if hits:
            return hits[0]
    return None


def detect_triplet(model_dir):
    """Return (dit, vae, text_encoder) or raise. Accept both hyphen/underscore
    sub-dir names."""
    d = os.path.abspath(model_dir)
    dit = _first_match(d,
                       "diffusion_models/*.safetensors",
                       "diffusion_models/*.gguf",
                       "diffusion-models/*.safetensors",
                       "diffusion-models/*.gguf")
    vae = _first_match(d, "vae/*.safetensors", "vae/*.gguf")
    enc = _first_match(d,
                       "text_encoders/*.safetensors",
                       "text_encoders/*.gguf",
                       "text-encoder/*.gguf",
                       "text-encoder/*.safetensors",
                       "text_encoder/*.safetensors",
                       "text_encoder/*.gguf")
    # Skip multimodal projector file if a real encoder is present alongside.
    if enc and os.path.basename(enc).lower().startswith("mmproj"):
        enc2 = _first_match(d,
                            "text-encoder/Qwen*",
                            "text_encoders/Qwen*",
                            "text_encoder/Qwen*")
        if enc2:
            enc = enc2
    if not (dit and vae and enc):
        raise FileNotFoundError(
            f"qwen-image variant '{d}' is missing one of "
            f"{{dit, vae, text_encoder}}: dit={dit} vae={vae} enc={enc}")
    return dit, vae, enc


def detect_kind(dit_path):
    """'gguf' or 'safetensors' — fp8 vs Q4_0 roughly maps to the filename."""
    return "gguf" if dit_path.endswith(".gguf") else "safetensors"


class Variant:
    __slots__ = ("name", "model_dir", "dit", "vae", "enc", "kind")
    def __init__(self, name, model_dir):
        self.name = name
        self.model_dir = os.path.abspath(model_dir)
        self.dit, self.vae, self.enc = detect_triplet(self.model_dir)
        self.kind = detect_kind(self.dit)


def parse_variants_arg(s):
    out = []
    if not s: return out
    for piece in s.split(","):
        piece = piece.strip()
        if not piece: continue
        if ":" not in piece:
            raise ValueError(f"bad --variants entry '{piece}'; want name:dir")
        name, d = piece.split(":", 1)
        out.append(Variant(name.strip(), d.strip()))
    return out


# ---- Image IO helpers ----------------------------------------------------

def _read_image_file(path):
    """Read PNG/JPG/PPM file into (PIL.Image, raw_bytes). The C runner writes
    PPM; the python ref writes PNG. Convert PPM → PNG in-memory."""
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        raise RuntimeError(f"empty output image: {path}")
    im = Image.open(io.BytesIO(data)).convert("RGB")
    if data.startswith(b"P6") or data.startswith(b"P3"):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data = buf.getvalue()
    return im, data


# ---- Shell-out backends --------------------------------------------------

class OursCpuBackend:
    """cpu/qwen_image/test_qwen_image --generate <dit> <vae> <enc> --prompt ... --out <path>"""
    def __init__(self, bin_path):
        self.bin = os.path.abspath(bin_path)
        self.lock = threading.Lock()
        if not os.path.isfile(self.bin):
            print(f"[ours-cpu] warning: binary not found at {self.bin}",
                  file=sys.stderr)

    def infer(self, variant, req):
        prompt = str(req.get("text") or "")
        width  = int(req.get("width")  or 256)
        height = int(req.get("height") or 256)
        steps  = int(req.get("steps")  or 20)
        seed   = int(req.get("seed")   or 42)
        with self.lock, tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.ppm")
            cmd = [self.bin, "--generate", variant.dit, variant.vae, variant.enc,
                   "--prompt", prompt,
                   "--width", str(width), "--height", str(height),
                   "--steps", str(steps), "--seed", str(seed),
                   "--out", out_path]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(out_path):
                raise RuntimeError(
                    f"ours-cpu({variant.name}) failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            _, png = _read_image_file(out_path)
        return png, (width, height), dt_ms


class OursCudaBackend:
    """cuda/qwen_image/test_cuda_qwen_image — optional, advertised only when built."""
    def __init__(self, bin_path):
        self.bin = os.path.abspath(bin_path)
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.bin)

    def infer(self, variant, req):
        if not self.available():
            raise RuntimeError(f"ours-cuda binary missing: {self.bin}")
        prompt = str(req.get("text") or "")
        width  = int(req.get("width")  or 256)
        height = int(req.get("height") or 256)
        steps  = int(req.get("steps")  or 20)
        seed   = int(req.get("seed")   or 42)
        with self.lock, tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.png")
            cmd = [self.bin, "--generate", variant.dit, variant.vae, variant.enc,
                   "--prompt", prompt,
                   "--width", str(width), "--height", str(height),
                   "--steps", str(steps), "--seed", str(seed),
                   "--out", out_path]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(out_path):
                raise RuntimeError(
                    f"ours-cuda({variant.name}) failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            _, png = _read_image_file(out_path)
        return png, (width, height), dt_ms


class PytorchShellBackend:
    """Shell-out to ref/qwen_image/gen_diffusers_reference.py.

    Loads the diffusers pipeline per call (slow); avoids keeping 6–8 GB
    resident between requests. Variants differ only by --fp8-dit.
    """
    def __init__(self, script_path, python_exe):
        self.script = os.path.abspath(script_path)
        self.python = python_exe
        self.lock = threading.Lock()

    def infer(self, variant, req):
        prompt = str(req.get("text") or "")
        width  = int(req.get("width")  or 256)
        height = int(req.get("height") or 256)
        steps  = int(req.get("steps")  or 20)
        seed   = int(req.get("seed")   or 42)
        # gen_diffusers_reference.py uses --size (square); take min for now.
        size = min(width, height)
        fp8 = variant.dit if variant.dit.endswith(".safetensors") else ""
        with self.lock, tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.png")
            cmd = [self.python, self.script,
                   "--prompt", prompt,
                   "--size", str(size),
                   "--steps", str(steps),
                   "--seed", str(seed),
                   "-o", out_path,
                   "--fp8-dit", fp8]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(out_path):
                raise RuntimeError(
                    f"pytorch-cuda({variant.name}) failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            _, png = _read_image_file(out_path)
        return png, (size, size), dt_ms


# ---- HTTP dispatch -------------------------------------------------------

def make_handler(backends, variants, default_variant, web_root):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[qwen] " + (fmt % a) + "\n")

        def do_OPTIONS(self):
            self.send_response(204)
            _cors_headers(self)
            self.end_headers()

        def _serve_file(self, path, ctype):
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                self.send_error(404); return
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            _cors_headers(self)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/qwen_image_compare", "/qwen_image_compare.html"):
                self._serve_file(
                    os.path.join(web_root, "qwen_image_compare.html"),
                    "text/html; charset=utf-8")
                return
            if path == "/health":
                _json(self, 200, {"ok": True,
                                  "backends": sorted(backends.keys()),
                                  "variants": [{"name": v.name, "kind": v.kind}
                                               for v in variants.values()]})
                return
            if path in ("/models", "/v1/models"):
                _json(self, 200, {"ok": True, "models": [
                    {"id": "qwen-image", "tasks": ["text-to-image"],
                     "backends": sorted(backends.keys()),
                     "variants": [{"name": v.name, "kind": v.kind}
                                  for v in variants.values()]}
                ]})
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
                _json(self, 400, {"ok": False, "error": f"bad json: {e}"})
                return

            t_all = time.time()
            try:
                inputs = req.get("inputs") or {}
                params = req.get("params") or {}
                text = str(inputs.get("text") or "").strip()
                be_name = str(params.get("backend") or "").strip()
                var_name = str(params.get("variant") or default_variant).strip()
                if not text:
                    _json(self, 400, {"ok": False,
                                      "error": "inputs.text required"})
                    return
                if be_name not in backends:
                    _json(self, 400, {"ok": False,
                                      "error": f"unknown backend '{be_name}'; "
                                               f"have {sorted(backends.keys())}"})
                    return
                if var_name not in variants:
                    _json(self, 400, {"ok": False,
                                      "error": f"unknown variant '{var_name}'; "
                                               f"have {sorted(variants.keys())}"})
                    return

                req_pack = {
                    "text": text,
                    "width":  params.get("width"),
                    "height": params.get("height"),
                    "steps":  params.get("steps"),
                    "seed":   params.get("seed"),
                }
                be = backends[be_name]
                variant = variants[var_name]
                png, (W, H), dt_infer = be.infer(variant, req_pack)

                dt_ms = int((time.time() - t_all) * 1000)
                _json(self, 200, {
                    "ok": True,
                    "model": "qwen-image",
                    "task": "text-to-image",
                    "backend": be_name,
                    "variant": var_name,
                    "variant_kind": variant.kind,
                    "outputs": [{
                        "type": "image",
                        "mime": "image/png",
                        "width": int(W),
                        "height": int(H),
                        "data_base64": base64.b64encode(png).decode("ascii"),
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--cpu-bin",
                    default=os.path.join(repo_root, "cpu", "qwen_image",
                                         "test_qwen_image"))
    ap.add_argument("--cuda-bin",
                    default=os.path.join(repo_root, "cuda", "qwen_image",
                                         "test_cuda_qwen_image"))
    ap.add_argument("--ref-script",
                    default=os.path.join(here, "gen_diffusers_reference.py"))
    ap.add_argument("--ref-python", default=sys.executable,
                    help="python interpreter used for the pytorch reference shell-out")
    ap.add_argument("--variants", default=None,
                    help="multi-variant spec: name:dir,name:dir,...")
    ap.add_argument("--default-variant", default=None)
    ap.add_argument("--web-root", default=os.path.join(repo_root, "web"))
    ap.add_argument("--disable", default="",
                    help="comma-sep list of backends to skip: ours-cpu,ours-cuda,pytorch")
    args = ap.parse_args()

    variants_list = parse_variants_arg(args.variants) if args.variants else []
    if not variants_list:
        sys.exit("error: supply --variants name:dir,...")
    variants = {v.name: v for v in variants_list}
    default_variant = args.default_variant or variants_list[0].name
    if default_variant not in variants:
        sys.exit(f"error: --default-variant {default_variant!r} not in variants")

    disabled = {x.strip() for x in args.disable.split(",") if x.strip()}
    backends = {}
    if "ours-cpu" not in disabled:
        backends["ours-cpu"] = OursCpuBackend(args.cpu_bin)
    if "ours-cuda" not in disabled:
        be = OursCudaBackend(args.cuda_bin)
        if be.available():
            backends["ours-cuda"] = be
        else:
            print(f"[qwen] ours-cuda: binary not present at {args.cuda_bin}, "
                  f"backend will not be advertised", file=sys.stderr)
    if "pytorch" not in disabled:
        backends["pytorch-cuda"] = PytorchShellBackend(
            args.ref_script, args.ref_python)

    if not backends:
        sys.exit("no backends enabled")

    web_root = os.path.abspath(args.web_root)
    Handler = make_handler(backends, variants, default_variant, web_root)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[qwen] listening on http://{args.host}:{args.port}  "
          f"backends={sorted(backends.keys())}  web_root={web_root}",
          flush=True)
    print(f"[qwen] variants: " +
          ", ".join(f"{v.name}({v.kind})" for v in variants.values()),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
