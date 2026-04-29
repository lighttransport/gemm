#!/usr/bin/env python3
"""TRELLIS.2 side-by-side compare server (Stage 1 structure mesh).

One HTTP service that dispatches /v1/infer across backends for
image-conditioned 3D structure generation:

  - "ours-cpu":     shells out to cpu/trellis2/test_trellis2 full + mesh
                    (disabled by default — far too slow)
  - "ours-cuda":    shells out to cuda/trellis2/test_cuda_trellis2 with --image
  - "pytorch-cuda": shells out to ref/trellis2/gen_stage1_ref.py, then runs
                    marching cubes on the resulting occupancy.npy in-process.

Input: RGB(A) PNG base64 (rembg runs in the pytorch path; C runners skip it).
Output: GLB (base64). OBJ from C runners is converted via trimesh.
"""
import argparse
import base64
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


# ---- Variant spec --------------------------------------------------------

class Variant:
    """A trellis2 variant references explicit weight triplets."""
    __slots__ = ("name", "stage1", "decoder", "dinov3",
                 "default_steps", "default_cfg", "default_grid")

    def __init__(self, name, stage1, decoder, dinov3):
        self.name = name
        self.stage1  = os.path.abspath(stage1)
        self.decoder = os.path.abspath(decoder)
        self.dinov3  = os.path.abspath(dinov3)
        for p, lbl in [(self.stage1, "stage1"),
                       (self.decoder, "decoder"),
                       (self.dinov3, "dinov3")]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"variant '{name}' {lbl}: {p} not a file")
        self.default_steps = 12
        self.default_cfg = 7.5
        self.default_grid = 64


def parse_variants_arg(s):
    """Spec: name=stage1:decoder:dinov3,name=stage1:decoder:dinov3

    Example:
      default=/ckpts/ss_flow.st:/ckpts/ss_dec.st:/models/dinov3-vitl16/model.safetensors
    """
    out = []
    if not s: return out
    for piece in s.split(","):
        piece = piece.strip()
        if not piece: continue
        if "=" not in piece:
            raise ValueError(f"bad --variants entry '{piece}'; want name=stage1:decoder:dinov3")
        name, paths = piece.split("=", 1)
        parts = paths.split(":")
        if len(parts) != 3:
            raise ValueError(f"variant '{name}': expected stage1:decoder:dinov3, got {paths}")
        out.append(Variant(name.strip(), *parts))
    return out


# ---- Mesh helpers --------------------------------------------------------

def _decode_image(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _mesh_to_glb(path):
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return buf.getvalue(), "model/gltf-binary", int(len(mesh.vertices)), int(len(mesh.faces))


def _occupancy_to_obj(occ_npy_path, out_obj_path, threshold=0.0):
    """Run marching cubes on an occupancy .npy and write an OBJ."""
    import numpy as np
    from skimage.measure import marching_cubes
    occ = np.load(occ_npy_path).astype(np.float32)
    if occ.ndim == 4: occ = occ[0]
    verts, faces, _, _ = marching_cubes(occ, level=threshold)
    # Normalize coords to [-0.5, 0.5]
    n = float(max(occ.shape))
    verts = verts / n - 0.5
    with open(out_obj_path, "w") as f:
        f.write(f"# TRELLIS.2 pytorch ref: {len(verts)} verts, {len(faces)} tris\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


# ---- Backends ------------------------------------------------------------

class OursCpuBackend:
    """cpu/trellis2/test_trellis2 full ... -o occ.npy;
       cpu/trellis2/test_trellis2 mesh occ.npy -o out.obj"""
    def __init__(self, bin_path):
        self.bin = os.path.abspath(bin_path)
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)

    def infer(self, variant, image, params):
        seed = int(params.get("seed") or 42)
        threshold = float(params.get("threshold") or 0.0)
        with self.lock, tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "in.png")
            occ = os.path.join(td, "occ.npy")
            obj = os.path.join(td, "out.obj")
            image.save(png, format="PNG")
            t0 = time.time()
            proc = subprocess.run(
                [self.bin, "full",
                 variant.dinov3, variant.stage1, variant.decoder, png,
                 "-o", occ, "-s", str(seed)],
                capture_output=True, text=True)
            if proc.returncode != 0 or not os.path.isfile(occ):
                raise RuntimeError(
                    f"ours-cpu (full) failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            proc = subprocess.run(
                [self.bin, "mesh", occ, "-o", obj, "-t", f"{threshold:g}"],
                capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(obj):
                raise RuntimeError(
                    f"ours-cpu (mesh) failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            glb, mime, nv, nf = _mesh_to_glb(obj)
        return glb, mime, nv, nf, dt_ms


class OursCudaBackend:
    """cuda/trellis2/test_cuda_trellis2 <stage1> <decoder> <dummy.npy>
         --image img.png --dinov3 dinov3.st -o out.obj
         -n steps -g cfg -s seed --grid N
    """
    def __init__(self, bin_path):
        self.bin = os.path.abspath(bin_path)
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.bin) and os.access(self.bin, os.X_OK)

    def infer(self, variant, image, params):
        steps = int(params.get("steps") or variant.default_steps)
        cfg   = float(params.get("cfg") or params.get("guidance") or variant.default_cfg)
        grid  = int(params.get("grid") or variant.default_grid)
        seed  = int(params.get("seed") or 42)
        with self.lock, tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "in.png")
            obj = os.path.join(td, "out.obj")
            dummy_features = os.path.join(td, "features.npy")
            image.save(png, format="PNG")
            # Write a tiny dummy .npy so argv[3] is readable (the binary
            # ignores its contents when --image is passed).
            import numpy as np
            np.save(dummy_features, np.zeros((1, 1), dtype=np.float32))
            t0 = time.time()
            proc = subprocess.run(
                [self.bin, variant.stage1, variant.decoder, dummy_features,
                 "--image", png, "--dinov3", variant.dinov3,
                 "-o", obj,
                 "-n", str(steps), "-g", f"{cfg:g}",
                 "-s", str(seed), "--grid", str(grid)],
                capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(obj):
                raise RuntimeError(
                    f"ours-cuda failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            glb, mime, nv, nf = _mesh_to_glb(obj)
        return glb, mime, nv, nf, dt_ms


class PytorchShellBackend:
    """ref/trellis2/gen_stage1_ref.py --image ... --dinov3 ... --stage1 ...
        --decoder ... --seed N --steps N --output-dir <dir>
       Then in-process marching cubes on ref_occupancy.npy -> .obj."""
    def __init__(self, script_path, python_exe):
        self.script = os.path.abspath(script_path)
        self.python = python_exe
        self.lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.script)

    def infer(self, variant, image, params):
        steps = int(params.get("steps") or variant.default_steps)
        cfg   = float(params.get("cfg") or params.get("guidance") or variant.default_cfg)
        seed  = int(params.get("seed") or 42)
        threshold = float(params.get("threshold") or 0.0)
        with self.lock, tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "in.png")
            out_dir = os.path.join(td, "out")
            os.makedirs(out_dir, exist_ok=True)
            image.save(png, format="PNG")
            t0 = time.time()
            proc = subprocess.run(
                [self.python, self.script,
                 "--image", png,
                 "--dinov3", variant.dinov3,
                 "--stage1", variant.stage1,
                 "--decoder", variant.decoder,
                 "--seed", str(seed),
                 "--steps", str(steps),
                 "--cfg", f"{cfg:g}",
                 "--output-dir", out_dir],
                capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"pytorch-cuda failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            occ_path = os.path.join(out_dir, "ref_occupancy.npy")
            if not os.path.isfile(occ_path):
                raise RuntimeError(
                    f"pytorch-cuda: occupancy missing at {occ_path}\n"
                    f"stderr tail:\n{proc.stderr[-1000:]}")
            obj_path = os.path.join(td, "out.obj")
            _occupancy_to_obj(occ_path, obj_path, threshold=threshold)
            dt_ms = int((time.time() - t0) * 1000)
            glb, mime, nv, nf = _mesh_to_glb(obj_path)
        return glb, mime, nv, nf, dt_ms


# ---- HTTP dispatch -------------------------------------------------------

def make_handler(backends, variants, default_variant, web_root):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[trellis2] " + (fmt % a) + "\n")

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
            if path in ("/", "/trellis2", "/trellis2.html"):
                self._serve_file(os.path.join(web_root, "trellis2.html"),
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
                    {"id": "trellis2", "tasks": ["image-to-3d"],
                     "backends": sorted(backends.keys()),
                     "variants": [{"name": v.name,
                                   "default_steps": v.default_steps,
                                   "default_cfg": v.default_cfg,
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

                image = _decode_image(img_b64)
                be = backends[be_name]
                variant = variants[var_name]
                glb, mime, nv, nf, dt_infer = be.infer(variant, image, params)

                dt_ms = int((time.time() - t_all) * 1000)
                _json(self, 200, {
                    "ok": True,
                    "model": "trellis2",
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
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--cpu-bin",
                    default=os.path.join(repo_root, "cpu", "trellis2", "test_trellis2"))
    ap.add_argument("--cuda-bin",
                    default=os.path.join(repo_root, "cuda", "trellis2", "test_cuda_trellis2"))
    ap.add_argument("--ref-script",
                    default=os.path.join(here, "gen_stage1_ref.py"))
    ap.add_argument("--ref-python", default=sys.executable)
    ap.add_argument("--variants", default=None,
                    help="spec: name=stage1:decoder:dinov3,name=stage1:decoder:dinov3")
    ap.add_argument("--default-variant", default=None)
    ap.add_argument("--web-root", default=os.path.join(repo_root, "web"))
    ap.add_argument("--disable", default="ours-cpu",
                    help="comma-sep list of backends to skip. Default disables ours-cpu "
                         "(far too slow for interactive compare). Use --disable '' "
                         "or DISABLE='' env to override.")
    args = ap.parse_args()

    variants_list = parse_variants_arg(args.variants) if args.variants else []
    if not variants_list:
        sys.exit("error: supply --variants name=stage1:decoder:dinov3,...")
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
            print(f"[trellis2] ours-cpu binary not present at {args.cpu_bin}",
                  file=sys.stderr)
    if "ours-cuda" not in disabled:
        be = OursCudaBackend(args.cuda_bin)
        if be.available():
            backends["ours-cuda"] = be
        else:
            print(f"[trellis2] ours-cuda binary not present at {args.cuda_bin}",
                  file=sys.stderr)
    if "pytorch" not in disabled:
        be = PytorchShellBackend(args.ref_script, args.ref_python)
        if be.available():
            backends["pytorch-cuda"] = be
        else:
            print(f"[trellis2] pytorch ref script missing at {args.ref_script}",
                  file=sys.stderr)

    if not backends:
        sys.exit("no backends enabled")

    web_root = os.path.abspath(args.web_root)
    Handler = make_handler(backends, variants, default_variant, web_root)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[trellis2] listening on http://{args.host}:{args.port}  "
          f"backends={sorted(backends.keys())}  web_root={web_root}",
          flush=True)
    print(f"[trellis2] variants: " +
          ", ".join(f"{v.name}" for v in variants.values()), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
