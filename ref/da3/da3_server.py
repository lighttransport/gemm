#!/usr/bin/env python3
"""DA3 side-by-side compare server.

One HTTP service that dispatches /v1/infer to three backends based on
`params.backend`:

  - "ours-cpu":     shells out to cpu/da3/test_da3  with --npy
  - "ours-cuda":    shells out to cuda/da3/test_cuda_da3 with --npy
  - "pytorch-cuda": in-process via one of the three DA3 reference modules,
                    auto-selected from the config:
                        dualdpt  → run_reference.DepthAnything3   (small/base/large)
                        giant    → run_reference_giant.DepthAnything3Giant  (giant)
                        nested   → run_reference_nested.NestedDA3  (nested 1.0 / 1.1)

Multiple model variants can be registered via --variants; the request picks
one with `params.variant`. Each variant is a {name, kind, model_dir, ckpt}.
Torch models are loaded lazily per variant and cached.

Depth output is colormapped (turbo) to an RGB PNG and returned base64
so the browser can draw it directly. Raw stats (min/max/mean) are also
returned so the UI can report numerics.

Usage examples:

    # one variant (legacy form)
    python da3_server.py --model-dir /mnt/disk01/models/da3-small

    # register multiple variants
    python da3_server.py \\
        --variants small:/mnt/disk01/models/da3-small,base:/mnt/disk01/models/da3-base,\\
large:/mnt/disk01/models/da3-large,giant:/mnt/disk01/models/da3-giant,\\
nested-1.0:/mnt/disk01/models/da3nested-giant-large,\\
nested-1.1:/mnt/disk01/models/da3nested-giant-large-1.1
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

import numpy as np
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


# ---- turbo colormap (Google) ---------------------------------------------
# Compact LUT; applied to normalized depth ∈ [0,1]. Index 0=dark blue, 255=red.
_TURBO = np.array([
 [48,18,59],[50,21,67],[51,24,74],[52,27,81],[53,30,88],[54,33,95],[55,36,102],
 [56,39,109],[57,42,115],[58,45,121],[59,47,128],[60,50,134],[61,53,139],
 [62,56,145],[63,59,151],[63,62,156],[64,64,162],[65,67,167],[65,70,172],
 [66,73,177],[66,75,181],[67,78,186],[68,81,191],[68,84,195],[68,86,199],
 [69,89,203],[69,92,207],[69,94,211],[70,97,214],[70,100,218],[70,102,221],
 [70,105,224],[70,107,227],[71,110,230],[71,113,233],[71,115,235],[71,118,238],
 [71,120,240],[71,123,242],[70,125,244],[70,128,246],[70,130,248],[70,133,250],
 [70,135,251],[69,138,252],[69,140,253],[68,143,254],[67,145,254],[66,148,255],
 [65,150,255],[64,153,255],[62,155,254],[61,158,254],[59,160,253],[58,163,252],
 [56,165,251],[55,168,250],[53,171,248],[51,173,247],[49,175,245],[47,178,244],
 [46,180,242],[44,183,240],[42,185,238],[40,188,235],[39,190,233],[37,192,231],
 [35,195,228],[34,197,226],[32,199,223],[31,201,221],[30,203,218],[28,205,216],
 [27,207,214],[26,209,211],[26,211,209],[25,213,207],[24,215,205],[24,217,202],
 [23,218,200],[23,220,198],[23,222,196],[23,223,193],[23,224,191],[24,226,188],
 [25,227,185],[25,228,182],[26,229,180],[27,230,177],[29,231,174],[30,232,170],
 [32,233,167],[34,234,164],[36,234,160],[38,235,156],[41,235,153],[43,236,149],
 [46,236,145],[50,236,141],[53,236,137],[57,237,133],[60,237,128],[64,237,124],
 [68,237,120],[72,237,116],[77,237,111],[81,236,107],[85,236,102],[89,236,98],
 [94,236,94],[98,235,89],[102,235,85],[107,234,81],[111,234,77],[115,233,73],
 [120,233,69],[124,232,65],[128,231,61],[133,231,58],[137,230,54],[141,229,51],
 [145,228,48],[150,227,45],[154,225,42],[158,224,40],[162,223,37],[166,222,35],
 [170,220,32],[174,219,30],[178,217,28],[181,216,27],[185,214,25],[188,212,23],
 [192,210,22],[195,209,20],[198,207,19],[201,205,17],[204,203,16],[207,201,15],
 [210,199,14],[213,197,13],[216,195,12],[218,193,11],[221,191,10],[223,188,10],
 [226,186,9],[228,184,8],[230,181,8],[232,179,7],[234,176,6],[236,174,6],
 [238,171,5],[239,169,5],[241,166,4],[242,164,4],[244,161,3],[245,158,3],
 [246,156,3],[247,153,2],[248,150,2],[249,148,2],[250,145,1],[250,142,1],
 [251,140,1],[251,137,1],[252,134,0],[252,132,0],[252,129,0],[253,126,0],
 [253,124,0],[253,121,0],[253,118,0],[253,116,0],[253,113,0],[253,111,0],
 [253,108,0],[253,105,0],[253,103,0],[252,100,0],[252,97,0],[251,94,0],
 [251,91,0],[250,88,0],[249,85,0],[249,82,0],[248,79,0],[247,76,0],[246,73,0],
 [245,70,0],[244,67,0],[243,64,0],[241,61,0],[240,58,0],[238,55,0],[237,52,0],
 [235,50,0],[233,47,0],[232,44,0],[230,42,0],[228,39,0],[226,36,0],[224,34,0],
 [222,32,0],[219,29,0],[217,27,0],[215,25,0],[212,23,0],[210,21,0],[207,19,0],
 [205,17,0],[202,15,0],[199,14,0],[197,12,0],[194,10,0],[191,9,0],[188,7,0],
 [185,6,0],[182,5,0],[179,4,0],[176,3,0],[173,2,0],[170,1,0],[167,1,0],
 [164,0,0],[161,0,0],[158,0,0],[155,0,0],[152,0,0],[149,0,0],[145,0,0],
 [142,0,0],[139,0,0],[136,0,0],[133,0,0],[130,0,0],[127,0,0],[124,0,0],
 [121,0,0],[118,0,0],[115,0,0],[112,0,0],[109,0,0],[106,0,0],[103,0,0],
 [100,0,0],[97,0,0],[94,0,0],[91,0,0],[88,0,0],[85,0,0],[82,0,0],[79,0,0],
 [76,0,0],[73,0,0],[70,0,0],[67,0,0],[64,0,0],[61,0,0],[58,0,0],[55,0,0],
 [52,0,0],[50,0,0],
], dtype=np.uint8)
if _TURBO.shape[0] < 256:
    pad = np.tile(_TURBO[-1:], (256 - _TURBO.shape[0], 1))
    _TURBO = np.vstack([_TURBO, pad])
_TURBO = _TURBO[:256]


def depth_to_png_b64(depth, dmin=None, dmax=None):
    """Normalize depth to [0,1] and colormap to turbo RGB PNG."""
    d = np.asarray(depth, dtype=np.float32)
    if dmin is None: dmin = float(d.min())
    if dmax is None: dmax = float(d.max())
    rng = dmax - dmin if dmax > dmin else 1.0
    norm = np.clip((d - dmin) / rng, 0.0, 1.0)
    idx = (norm * 255.0 + 0.5).astype(np.uint8)
    rgb = _TURBO[idx]
    im = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---- Variant discovery ---------------------------------------------------

def detect_kind(model_dir):
    """Read config.json from model_dir and return one of
    'dualdpt' | 'giant' | 'nested'."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    cfg = raw.get("config", raw)
    # Nested has top-level anyview + metric.
    if "anyview" in cfg and "metric" in cfg:
        return "nested"
    # The small run_reference.DepthAnything3 class only supports vits/vitb/vitl;
    # vitg must go through run_reference_giant regardless of head path.
    net = cfg.get("net", {})
    if net.get("name") == "vitg":
        return "giant"
    # Giant-style DPT head (with gs_head, aux_dpt) — detectable when the config
    # top-level has gs_head (indicates DA3-GIANT's multi-output model).
    if "gs_head" in cfg:
        return "giant"
    # Default: DualDPT (small, base, large).
    return "dualdpt"


class Variant:
    __slots__ = ("name", "kind", "model_dir", "ckpt")
    def __init__(self, name, model_dir, ckpt=None, kind=None):
        self.name = name
        self.model_dir = os.path.abspath(model_dir)
        self.ckpt = os.path.abspath(ckpt) if ckpt else \
                    os.path.join(self.model_dir, "model.safetensors")
        self.kind = kind or detect_kind(self.model_dir)


def parse_variants_arg(s):
    """Parse 'name:dir[,name:dir,...]' into a list of Variant."""
    out = []
    if not s: return out
    for piece in s.split(","):
        piece = piece.strip()
        if not piece: continue
        if ":" not in piece:
            raise ValueError(f"bad --variants entry '{piece}'; want name:dir")
        name, d = piece.split(":", 1)
        name = name.strip()
        d = d.strip()
        out.append(Variant(name, d))
    return out


# ---- Shell-out backends --------------------------------------------------

class ShellBackend:
    """Wraps one of the C test binaries. Writes input to tmp, runs, reads npy."""
    def __init__(self, name, bin_path, extra_args=None):
        self.name = name
        self.bin = os.path.abspath(bin_path)
        self.extra = list(extra_args or [])
        self.lock = threading.Lock()
        if not os.path.isfile(self.bin):
            print(f"[{name}] warning: binary not found at {self.bin}",
                  file=sys.stderr)

    def infer(self, image_bytes, variant):
        with self.lock, tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "input.jpg")
            npy_path = os.path.join(td, "depth.npy")
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            cmd = [self.bin, variant.ckpt, "-i", img_path, "--npy", npy_path] \
                  + self.extra
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(npy_path):
                raise RuntimeError(
                    f"{self.name}({variant.name}) failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}")
            depth = np.load(npy_path)
            return depth, dt_ms


# ---- In-process pytorch backend -----------------------------------------

class TorchRefBackend:
    """Lazy per-variant torch model cache; dispatches by variant.kind.

    If unload_after=True, the model is dropped + CUDA cache emptied after
    each infer. Useful on tight-VRAM GPUs where coexistence with the
    shell-out CUDA runner causes OOM.
    """
    def __init__(self, device, unload_after=False):
        self.device = device
        self.unload_after = unload_after
        self.lock = threading.Lock()
        self._cache = {}   # name -> (kind, model, dtype)
        self._imports = None

    def _do_imports(self):
        if self._imports is not None: return self._imports
        import torch  # noqa
        import torch.nn.functional as F  # noqa
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import run_reference as rr
        import run_reference_giant as rrg
        import run_reference_nested as rrn
        self._imports = (torch, F, rr, rrg, rrn)
        return self._imports

    def _build(self, variant):
        torch, F, rr, rrg, rrn = self._do_imports()
        with open(os.path.join(variant.model_dir, "config.json")) as f:
            full_cfg = json.load(f)
        cfg = full_cfg.get("config", full_cfg)
        kind = variant.kind

        print(f"[pytorch] building variant='{variant.name}' kind={kind} "
              f"dir={variant.model_dir}", flush=True)

        if kind == "dualdpt":
            model = rr.DepthAnything3(cfg)
            rr.load_weights(model, variant.ckpt)
            if hasattr(model.backbone, "camera_token"):
                model.backbone.camera_token = None
            dtype = torch.float32
        elif kind == "giant":
            oc1_shapes = rrg.scan_aux_oc1_shapes(variant.ckpt)
            model = rrg.DepthAnything3Giant(cfg, oc1_shapes=oc1_shapes)
            rrg.load_giant_weights(model, variant.ckpt)
            # bfloat16 on GPU to fit (giant ~5GB in fp32)
            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        elif kind == "nested":
            oc1_shapes = rrg.scan_aux_oc1_shapes(variant.ckpt)
            model = rrn.NestedDA3(cfg, oc1_shapes=oc1_shapes)
            rrn.load_nested_weights(model, variant.ckpt)
            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        else:
            raise ValueError(f"unknown kind {kind}")

        model = model.to(device=self.device, dtype=dtype).eval()
        print(f"[pytorch] ready: {variant.name} on {self.device} ({dtype})",
              flush=True)
        return (kind, model, dtype)

    def infer(self, image_bytes, variant):
        with self.lock:
            if variant.name not in self._cache:
                self._cache[variant.name] = self._build(variant)
        kind, model, dtype = self._cache[variant.name]
        torch, F, rr, _, _ = self._imports

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        H, W = arr.shape[:2]
        inp = rr.preprocess(arr).to(device=self.device, dtype=dtype)

        t0 = time.time()
        with torch.no_grad():
            if kind == "dualdpt":
                out = model(inp)                # (1, 2, h, w)
                logit_depth = out[0, 0].float()
            elif kind == "giant":
                dpt_out, *_ = model(inp, inp.clone())
                logit_depth = dpt_out[0, 0].float()
            elif kind == "nested":
                dpt_out, *_ = model(inp, inp.clone())
                logit_depth = dpt_out[0, 0].float()
            depth_small = torch.exp(logit_depth)
            depth_up = F.interpolate(
                depth_small.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=True,
            ).squeeze().cpu().numpy().astype(np.float32)
        dt_ms = int((time.time() - t0) * 1000)

        if self.unload_after:
            with self.lock:
                self._cache.pop(variant.name, None)
            del model
            if self.device != "cpu":
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass
            import gc
            gc.collect()
            print(f"[pytorch] unloaded {variant.name} (unload_after)",
                  flush=True)

        return depth_up, dt_ms


# ---- HTTP dispatch ------------------------------------------------------

def make_handler(backends, variants, default_variant, web_root):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[da3] " + (fmt % a) + "\n")

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
            if path in ("/", "/da3_compare", "/da3_compare.html"):
                self._serve_file(os.path.join(web_root, "da3_compare.html"),
                                 "text/html; charset=utf-8")
                return
            if path.startswith("/public/"):
                fp = os.path.normpath(os.path.join(web_root, path.lstrip("/")))
                if not fp.startswith(os.path.abspath(web_root)):
                    self.send_error(403); return
                ctype = "application/octet-stream"
                if fp.endswith(".jpg") or fp.endswith(".jpeg"): ctype = "image/jpeg"
                elif fp.endswith(".png"): ctype = "image/png"
                elif fp.endswith(".webp"): ctype = "image/webp"
                elif fp.endswith(".txt"): ctype = "text/plain; charset=utf-8"
                self._serve_file(fp, ctype); return
            if path == "/health":
                _json(self, 200, {"ok": True,
                                  "backends": sorted(backends.keys()),
                                  "variants": [{"name": v.name, "kind": v.kind}
                                               for v in variants.values()]})
                return
            if path in ("/models", "/v1/models"):
                _json(self, 200, {"ok": True, "models": [
                    {"id": "da3", "tasks": ["depth"],
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
                b64 = inputs.get("image_base64") or ""
                be_name = str(params.get("backend") or "").strip()
                var_name = str(params.get("variant") or default_variant).strip()
                if not b64:
                    _json(self, 400, {"ok": False,
                                      "error": "inputs.image_base64 required"})
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
                try:
                    image_bytes = base64.b64decode(b64)
                    Image.open(io.BytesIO(image_bytes)).size  # validate
                except Exception as e:
                    _json(self, 400, {"ok": False, "error": f"bad image: {e}"})
                    return

                be = backends[be_name]
                variant = variants[var_name]
                depth, dt_infer = be.infer(image_bytes, variant)
                dmin = float(depth.min())
                dmax = float(depth.max())
                dmean = float(depth.mean())
                png_b64 = depth_to_png_b64(depth, dmin, dmax)

                dt_ms = int((time.time() - t_all) * 1000)
                _json(self, 200, {
                    "ok": True,
                    "model": "da3",
                    "task": "depth",
                    "backend": be_name,
                    "variant": var_name,
                    "variant_kind": variant.kind,
                    "outputs": [{
                        "type": "depth",
                        "mime": "image/png",
                        "width": int(depth.shape[-1]),
                        "height": int(depth.shape[-2]),
                        "data_base64": png_b64,
                        "stats": {"min": dmin, "max": dmax, "mean": dmean},
                    }],
                    "timings_ms": {"infer": int(dt_infer), "total": dt_ms},
                })
            except Exception as e:
                traceback.print_exc()
                _json(self, 500, {"ok": False, "error": str(e)})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8083)
    ap.add_argument("--cpu-bin",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "cpu", "da3", "test_da3"))
    ap.add_argument("--cuda-bin",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "cuda", "da3", "test_cuda_da3"))
    ap.add_argument("--model-dir", default=None,
                    help="legacy: single-variant dir (registers as 'default')")
    ap.add_argument("--ckpt", default=None,
                    help="legacy: safetensors path for --model-dir")
    ap.add_argument("--variants", default=None,
                    help="multi-variant spec: name:dir,name:dir,...")
    ap.add_argument("--default-variant", default=None,
                    help="variant name to use when request omits params.variant")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--web-root",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "web"))
    ap.add_argument("--disable", default="",
                    help="comma-sep list of backends to skip: ours-cpu,ours-cuda,pytorch")
    ap.add_argument("--torch-unload-after",
                    action="store_true",
                    default=bool(os.environ.get("TORCH_UNLOAD_AFTER")),
                    help="Drop pytorch model + empty CUDA cache after every "
                         "infer. Avoids OOM when sharing VRAM with ours-cuda.")
    args = ap.parse_args()

    # Build variant registry.
    variants_list = []
    if args.variants:
        variants_list = parse_variants_arg(args.variants)
    if args.model_dir:
        v = Variant("default", args.model_dir, ckpt=args.ckpt)
        variants_list.append(v)
    if not variants_list:
        sys.exit("error: supply --model-dir or --variants")
    variants = {v.name: v for v in variants_list}
    default_variant = args.default_variant or variants_list[0].name
    if default_variant not in variants:
        sys.exit(f"error: --default-variant '{default_variant}' not in {list(variants)}")

    print("[da3] variants:", flush=True)
    for v in variants_list:
        print(f"         - {v.name}  kind={v.kind}  dir={v.model_dir}", flush=True)
    print(f"[da3] default variant: {default_variant}", flush=True)

    disabled = {x.strip() for x in args.disable.split(",") if x.strip()}
    backends = {}
    if "ours-cpu" not in disabled:
        backends["ours-cpu"] = ShellBackend(
            "ours-cpu", args.cpu_bin,
            extra_args=["-t", str(max(1, os.cpu_count() or 4))])
    if "ours-cuda" not in disabled:
        backends["ours-cuda"] = ShellBackend("ours-cuda", args.cuda_bin)
    if "pytorch" not in disabled:
        backends["pytorch-cuda"] = TorchRefBackend(
            args.device, unload_after=args.torch_unload_after)

    if not backends:
        sys.exit("no backends enabled")

    web_root = os.path.abspath(args.web_root)
    Handler = make_handler(backends, variants, default_variant, web_root)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[da3] listening on http://{args.host}:{args.port}  "
          f"backends={sorted(backends.keys())}  web_root={web_root}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


if __name__ == "__main__":
    main()
