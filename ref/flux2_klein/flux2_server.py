#!/usr/bin/env python3
"""Flux.2-klein side-by-side compare server.

One HTTP service that dispatches /v1/infer across backends for a
text-to-image prompt:

  - "ours-cpu":     shells out to cpu/flux2/test_flux2 --generate ...
  - "ours-cuda":    shells out to cuda/flux2/test_cuda_flux2 --generate --gpu-enc ...
  - "ours-hip":     (advertised only if rdna4/flux2/test_hip_flux2 exists)
  - "pytorch-cuda": shells out to ref/flux2_klein/gen_reference.py

Variants select the DiT checkpoint inside the shared klein2-4b dir:

    distilled → diffusion_models/flux-2-klein-4b-fp8.safetensors      (4 steps)
    base      → diffusion_models/flux-2-klein-base-4b-fp8.safetensors (20 steps)

Request shape mirrors the C server:

    {
      "inputs": {"text": "<prompt>"},
      "params": {
        "backend": "ours-cpu|ours-cuda|ours-hip|pytorch-cuda",
        "variant": "distilled|base",
        "width": 256, "height": 256,
        "steps": 4, "seed": 42
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


# ---- Variant discovery ---------------------------------------------------

def _first_match(root, *globs):
    for g in globs:
        hits = sorted(glob.glob(os.path.join(root, g)))
        if hits:
            return hits[0]
    return None


def _all_matches(root, *globs):
    out = []
    for g in globs:
        out.extend(sorted(glob.glob(os.path.join(root, g))))
    return out


def _pick_dit(model_dir, kind):
    """kind = 'base' → pick *-base-*fp8.safetensors; else non-base."""
    cands = _all_matches(model_dir,
                         "diffusion_models/*.safetensors",
                         "diffusion-models/*.safetensors")
    base_cands = [c for c in cands if "base" in os.path.basename(c).lower()]
    dist_cands = [c for c in cands if "base" not in os.path.basename(c).lower()]
    pool = base_cands if kind == "base" else dist_cands
    if not pool:
        raise FileNotFoundError(
            f"flux2 variant '{kind}' has no matching DiT in "
            f"{model_dir}/diffusion_models/ (cands={cands})")
    return pool[0]


def _detect_enc(model_dir):
    """flux2 uses a directory of sharded text_encoder weights, not a single file."""
    for sub in ("text_encoder", "text-encoder", "text_encoders"):
        p = os.path.join(model_dir, sub)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"no text_encoder dir under {model_dir}")


def _detect_vae(model_dir):
    v = _first_match(model_dir, "vae/*.safetensors")
    if not v:
        raise FileNotFoundError(f"no vae/*.safetensors under {model_dir}")
    return v


class Variant:
    __slots__ = ("name", "model_dir", "dit", "vae", "enc", "kind",
                 "default_steps")
    def __init__(self, name, model_dir, kind_hint=None):
        self.name = name
        self.model_dir = os.path.abspath(model_dir)
        # Variant kind: explicit hint wins; otherwise derive from name.
        kind = (kind_hint or name).lower()
        self.kind = "base" if "base" in kind else "distilled"
        self.dit = _pick_dit(self.model_dir, self.kind)
        self.vae = _detect_vae(self.model_dir)
        self.enc = _detect_enc(self.model_dir)
        self.default_steps = 20 if self.kind == "base" else 4


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

def _base_dist_flag(variant):
    return "--base" if variant.kind == "base" else "--distilled"


def _req(req, key, default):
    v = req.get(key)
    return default if v is None else v


class OursCpuBackend:
    """cpu/flux2/test_flux2 --generate ... --out <ppm>."""
    def __init__(self, bin_path, tok_path):
        self.bin = os.path.abspath(bin_path)
        self.tok = tok_path
        self.lock = threading.Lock()
        if not os.path.isfile(self.bin):
            print(f"[ours-cpu] warning: binary not found at {self.bin}",
                  file=sys.stderr)

    def infer(self, variant, req):
        prompt = str(req.get("text") or "")
        width  = int(_req(req, "width",  256))
        height = int(_req(req, "height", 256))
        steps  = int(_req(req, "steps",  variant.default_steps))
        seed   = int(_req(req, "seed",   42))
        with self.lock, tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.ppm")
            cmd = [self.bin, "--generate", _base_dist_flag(variant),
                   "--dit", variant.dit,
                   "--vae", variant.vae,
                   "--enc", variant.enc,
                   "--tok", self.tok,
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
    """cuda/flux2/test_cuda_flux2 --generate --gpu-enc ... --out <ppm>."""
    def __init__(self, bin_path, tok_path):
        self.bin = os.path.abspath(bin_path)
        self.tok = tok_path
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.bin)

    def infer(self, variant, req):
        if not self.available():
            raise RuntimeError(f"ours-cuda binary missing: {self.bin}")
        prompt = str(req.get("text") or "")
        width  = int(_req(req, "width",  256))
        height = int(_req(req, "height", 256))
        steps  = int(_req(req, "steps",  variant.default_steps))
        seed   = int(_req(req, "seed",   42))
        with self.lock, tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.ppm")
            cmd = [self.bin, "--generate", "--gpu-enc", _base_dist_flag(variant),
                   "--dit", variant.dit,
                   "--vae", variant.vae,
                   "--enc", variant.enc,
                   "--tok", self.tok,
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


class OursHipBackend:
    """rdna4/flux2/test_hip_flux2 --generate ... --out <ppm>."""
    def __init__(self, bin_path, tok_path):
        self.bin = os.path.abspath(bin_path)
        self.tok = tok_path
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.bin)

    def infer(self, variant, req):
        if not self.available():
            raise RuntimeError(f"ours-hip binary missing: {self.bin}")
        prompt = str(req.get("text") or "")
        width  = int(_req(req, "width",  256))
        height = int(_req(req, "height", 256))
        steps  = int(_req(req, "steps",  variant.default_steps))
        seed   = int(_req(req, "seed",   42))
        # HIP binary takes --size (square only)
        size = min(width, height)
        with self.lock, tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.ppm")
            cmd = [self.bin, "--generate",
                   "--dit", variant.dit,
                   "--vae", variant.vae,
                   "--enc", variant.enc,
                   "--tok", self.tok,
                   "--prompt", prompt,
                   "--size", str(size),
                   "--steps", str(steps), "--seed", str(seed),
                   "--out", out_path]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(out_path):
                raise RuntimeError(
                    f"ours-hip({variant.name}) failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            _, png = _read_image_file(out_path)
        return png, (size, size), dt_ms


class PytorchShellBackend:
    """Shell-out to ref/flux2_klein/gen_reference.py."""
    def __init__(self, script_path, python_exe):
        self.script = os.path.abspath(script_path)
        self.python = python_exe
        self.lock = threading.Lock()

    def infer(self, variant, req):
        prompt = str(req.get("text") or "")
        width  = int(_req(req, "width",  256))
        height = int(_req(req, "height", 256))
        steps  = int(_req(req, "steps",  variant.default_steps))
        seed   = int(_req(req, "seed",   42))
        size = min(width, height)
        with self.lock, tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.png")
            cmd = [self.python, self.script,
                   "--prompt", prompt,
                   "--size", str(size),
                   "--steps", str(steps),
                   "--seed", str(seed),
                   "-o", out_path]
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
            sys.stderr.write("[flux2] " + (fmt % a) + "\n")

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
            if path in ("/", "/flux2_compare", "/flux2_compare.html"):
                self._serve_file(
                    os.path.join(web_root, "flux2_compare.html"),
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
                    {"id": "flux2-klein", "tasks": ["text-to-image"],
                     "backends": sorted(backends.keys()),
                     "variants": [{"name": v.name, "kind": v.kind,
                                   "default_steps": v.default_steps}
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
                    "model": "flux2-klein",
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
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--cpu-bin",
                    default=os.path.join(repo_root, "cpu", "flux2",
                                         "test_flux2"))
    ap.add_argument("--cuda-bin",
                    default=os.path.join(repo_root, "cuda", "flux2",
                                         "test_cuda_flux2"))
    ap.add_argument("--hip-bin",
                    default=os.path.join(repo_root, "rdna4", "flux2",
                                         "test_hip_flux2"))
    ap.add_argument("--ref-script",
                    default=os.path.join(here, "gen_reference.py"))
    ap.add_argument("--ref-python", default=sys.executable,
                    help="python interpreter used for the pytorch reference shell-out")
    ap.add_argument("--tok", required=True,
                    help="path to Qwen3-VL GGUF used for the tokenizer vocab")
    ap.add_argument("--variants", default=None,
                    help="multi-variant spec: name:dir,name:dir,...")
    ap.add_argument("--default-variant", default=None)
    ap.add_argument("--web-root", default=os.path.join(repo_root, "web"))
    ap.add_argument("--disable", default="",
                    help="comma-sep list of backends to skip: ours-cpu,ours-cuda,ours-hip,pytorch")
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
        backends["ours-cpu"] = OursCpuBackend(args.cpu_bin, args.tok)
    if "ours-cuda" not in disabled:
        be = OursCudaBackend(args.cuda_bin, args.tok)
        if be.available():
            backends["ours-cuda"] = be
        else:
            print(f"[flux2] ours-cuda: binary not present at {args.cuda_bin}, "
                  f"backend will not be advertised", file=sys.stderr)
    if "ours-hip" not in disabled:
        be = OursHipBackend(args.hip_bin, args.tok)
        if be.available():
            backends["ours-hip"] = be
        else:
            print(f"[flux2] ours-hip: binary not present at {args.hip_bin}, "
                  f"backend will not be advertised", file=sys.stderr)
    if "pytorch" not in disabled:
        backends["pytorch-cuda"] = PytorchShellBackend(
            args.ref_script, args.ref_python)

    if not backends:
        sys.exit("no backends enabled")

    web_root = os.path.abspath(args.web_root)
    Handler = make_handler(backends, variants, default_variant, web_root)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[flux2] listening on http://{args.host}:{args.port}  "
          f"backends={sorted(backends.keys())}  web_root={web_root}",
          flush=True)
    print(f"[flux2] variants: " +
          ", ".join(f"{v.name}({v.kind},{v.default_steps}steps)"
                    for v in variants.values()),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
