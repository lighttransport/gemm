#!/usr/bin/env python3
"""Hunyuan3D-2.1 side-by-side compare server.

One HTTP service that dispatches /v1/infer across backends for
image-conditioned 3D shape generation:

  - "ours-cpu":     shells out to cpu/hy3d/test_hy3d
  - "ours-cuda":    shells out to cuda/hy3d/test_cuda_hy3d
  - "pytorch-cuda": shells out to ref/hy3d/run_full_pipeline.py

Input: RGB/RGBA PNG (base64). C runners take PPM, so the server converts
PNG -> PPM in a temp dir. Pytorch ref takes PNG directly and writes GLB.

Output: GLB (base64). OBJ produced by C runners is converted with trimesh.

Request shape mirrors the C server:

    {
      "inputs": {"image_base64": "..."},        # required
      "params": {
        "backend": "ours-cpu|ours-cuda|pytorch-cuda",
        "variant": "<name>",
        "steps": 30, "guidance": 5.0,
        "grid": 256, "seed": 42
      }
    }
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


# ---- Variant discovery ---------------------------------------------------

def _first(root, *globs):
    for g in globs:
        hits = sorted(glob.glob(os.path.join(root, g)))
        if hits: return hits[0]
    return None


class Variant:
    """A hy3d variant = a directory with {conditioner,model,vae}.safetensors."""
    __slots__ = ("name", "model_dir", "cond", "model", "vae",
                 "default_steps", "default_guidance", "default_grid")

    def __init__(self, name, model_dir):
        self.name = name
        self.model_dir = os.path.abspath(model_dir)
        self.cond  = _first(self.model_dir, "conditioner.safetensors",
                                            "hunyuan3d-dit-*/conditioner.safetensors")
        self.model = _first(self.model_dir, "model.safetensors",
                                            "hunyuan3d-dit-*/model.safetensors")
        self.vae   = _first(self.model_dir, "vae.safetensors",
                                            "hunyuan3d-vae-*/vae.safetensors")
        if not (self.cond and self.model and self.vae):
            raise FileNotFoundError(
                f"variant '{name}' missing weights under {self.model_dir}: "
                f"cond={self.cond}, model={self.model}, vae={self.vae}")
        self.default_steps = 30
        self.default_guidance = 5.0
        self.default_grid = 256


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


# ---- Mesh IO -------------------------------------------------------------

def _decode_input_image(b64):
    raw = base64.b64decode(b64)
    im = Image.open(io.BytesIO(raw))
    return im


def _save_ppm(im, path):
    """C runners read PPM P6 RGB."""
    im.convert("RGB").save(path, format="PPM")


def _save_png(im, path):
    im.save(path, format="PNG")


def _mesh_to_glb(path):
    """Load a mesh file (GLB/OBJ/PLY) and return GLB bytes.
    If already GLB, return as-is."""
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        return data, "model/gltf-binary"
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue(), "model/gltf-binary"


def _count_mesh(path):
    try:
        import trimesh
        m = trimesh.load(path, force="mesh")
        return int(len(m.vertices)), int(len(m.faces))
    except Exception:
        return 0, 0


# ---- Backends ------------------------------------------------------------

class OursCpuBackend:
    """cpu/hy3d/test_hy3d cond.st model.st vae.st -i img.ppm -o out.obj"""
    def __init__(self, bin_path):
        self.bin = os.path.abspath(bin_path)
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)

    def infer(self, variant, image, params):
        with self.lock, tempfile.TemporaryDirectory() as td:
            ppm = os.path.join(td, "in.ppm")
            obj = os.path.join(td, "out.obj")
            _save_ppm(image, ppm)
            cmd = [self.bin, variant.cond, variant.model, variant.vae,
                   "-i", ppm, "-o", obj]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(obj):
                raise RuntimeError(
                    f"ours-cpu failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            nv, nf = _count_mesh(obj)
            glb, mime = _mesh_to_glb(obj)
        return glb, mime, nv, nf, dt_ms


class OursCudaBackend:
    """cuda/hy3d/test_cuda_hy3d cond.st model.st vae.st -i img.ppm -o out.obj
       -s steps -g guidance --grid res --seed N"""
    def __init__(self, bin_path):
        self.bin = os.path.abspath(bin_path)
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)

    def infer(self, variant, image, params):
        steps    = int(params.get("steps")    or variant.default_steps)
        guidance = float(params.get("guidance") or variant.default_guidance)
        grid     = int(params.get("grid")     or variant.default_grid)
        seed     = int(params.get("seed")     or 42)
        with self.lock, tempfile.TemporaryDirectory() as td:
            ppm = os.path.join(td, "in.ppm")
            obj = os.path.join(td, "out.obj")
            _save_ppm(image, ppm)
            cmd = [self.bin, variant.cond, variant.model, variant.vae,
                   "-i", ppm, "-o", obj,
                   "-s", str(steps), "-g", f"{guidance:g}",
                   "--grid", str(grid), "--seed", str(seed)]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(obj):
                raise RuntimeError(
                    f"ours-cuda failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            nv, nf = _count_mesh(obj)
            glb, mime = _mesh_to_glb(obj)
        return glb, mime, nv, nf, dt_ms


class PytorchShellBackend:
    """ref/hy3d/run_full_pipeline.py --image <png> --out <glb>
       --steps N --guidance G --octree R --seed S"""
    def __init__(self, script_path, python_exe):
        self.script = os.path.abspath(script_path)
        self.python = python_exe
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.script)

    def infer(self, variant, image, params):
        steps    = int(params.get("steps")    or variant.default_steps)
        guidance = float(params.get("guidance") or variant.default_guidance)
        grid     = int(params.get("grid")     or variant.default_grid)
        seed     = int(params.get("seed")     or 42)
        with self.lock, tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "in.png")
            glb = os.path.join(td, "out.glb")
            _save_png(image, png)
            env = os.environ.copy()
            env.setdefault("HY3DGEN_MODELS", os.path.dirname(variant.model_dir))
            env.setdefault("HY3D_MODEL_NAME", os.path.basename(variant.model_dir))
            cmd = [self.python, self.script,
                   "--image", png, "--out", glb,
                   "--steps", str(steps), "--guidance", f"{guidance:g}",
                   "--octree", str(grid), "--seed", str(seed)]
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(glb):
                raise RuntimeError(
                    f"pytorch-cuda failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            nv, nf = _count_mesh(glb)
            glb_data, mime = _mesh_to_glb(glb)
        return glb_data, mime, nv, nf, dt_ms


# ---- HTTP dispatch -------------------------------------------------------

def make_handler(backends, variants, default_variant, web_root):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[hy3d] " + (fmt % a) + "\n")

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

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/hy3d", "/hy3d.html"):
                self._serve_file(os.path.join(web_root, "hy3d.html"),
                                 "text/html; charset=utf-8")
                return
            if path == "/health":
                _json(self, 200, {"ok": True,
                                  "backends": sorted(backends.keys()),
                                  "variants": [{"name": v.name}
                                               for v in variants.values()]})
                return
            if path in ("/models", "/v1/models"):
                _json(self, 200, {"ok": True, "models": [
                    {"id": "hy3d", "tasks": ["image-to-3d"],
                     "backends": sorted(backends.keys()),
                     "variants": [{"name": v.name,
                                   "default_steps": v.default_steps,
                                   "default_guidance": v.default_guidance,
                                   "default_grid": v.default_grid}
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
                var_name = str(params.get("variant") or default_variant).strip()
                if be_name not in backends:
                    _json(self, 400, {"ok": False,
                                      "error": f"unknown backend '{be_name}'; "
                                               f"have {sorted(backends.keys())}"}); return
                if var_name not in variants:
                    _json(self, 400, {"ok": False,
                                      "error": f"unknown variant '{var_name}'; "
                                               f"have {sorted(variants.keys())}"}); return

                image = _decode_input_image(img_b64)
                be = backends[be_name]
                variant = variants[var_name]
                glb, mime, nv, nf, dt_infer = be.infer(variant, image, params)

                dt_ms = int((time.time() - t_all) * 1000)
                _json(self, 200, {
                    "ok": True,
                    "model": "hy3d",
                    "task": "image-to-3d",
                    "backend": be_name,
                    "variant": var_name,
                    "outputs": [{
                        "type": "mesh",
                        "mime": mime,
                        "n_verts": int(nv),
                        "n_faces": int(nf),
                        "data_base64": base64.b64encode(glb).decode("ascii"),
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
    ap.add_argument("--port", type=int, default=8087)
    ap.add_argument("--cpu-bin",
                    default=os.path.join(repo_root, "cpu", "hy3d", "test_hy3d"))
    ap.add_argument("--cuda-bin",
                    default=os.path.join(repo_root, "cuda", "hy3d", "test_cuda_hy3d"))
    ap.add_argument("--ref-script",
                    default=os.path.join(here, "run_full_pipeline.py"))
    ap.add_argument("--ref-python", default=sys.executable,
                    help="python interpreter for pytorch shell-out")
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
        be = OursCpuBackend(args.cpu_bin)
        if be.available():
            backends["ours-cpu"] = be
        else:
            print(f"[hy3d] ours-cpu binary not present at {args.cpu_bin}",
                  file=sys.stderr)
    if "ours-cuda" not in disabled:
        be = OursCudaBackend(args.cuda_bin)
        if be.available():
            backends["ours-cuda"] = be
        else:
            print(f"[hy3d] ours-cuda binary not present at {args.cuda_bin}",
                  file=sys.stderr)
    if "pytorch" not in disabled:
        be = PytorchShellBackend(args.ref_script, args.ref_python)
        if be.available():
            backends["pytorch-cuda"] = be
        else:
            print(f"[hy3d] pytorch ref script missing at {args.ref_script}",
                  file=sys.stderr)

    if not backends:
        sys.exit("no backends enabled")

    web_root = os.path.abspath(args.web_root)
    Handler = make_handler(backends, variants, default_variant, web_root)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[hy3d] listening on http://{args.host}:{args.port}  "
          f"backends={sorted(backends.keys())}  web_root={web_root}",
          flush=True)
    print(f"[hy3d] variants: " +
          ", ".join(f"{v.name}" for v in variants.values()), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
