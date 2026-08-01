#!/usr/bin/env python3
"""LLM / VLM side-by-side compare server.

One HTTP service that dispatches /v1/chat across backends:

  - "ours-cpu-llm":   cpu/llm/test_transformer  (Qwen3.5 text-only, GGUF)
  - "ours-cpu-vlm":   cpu/vlm/test_gemma4_vision (Gemma4 VLM, GGUF)
  - "ours-cuda-vlm":  cuda/vlm/test_cuda_gemma4_vlm (Gemma4 VLM on GPU)
  - "pytorch-cuda":   ref/llm/llm_ref.py (HF transformers reference)

Variant spec (comma-separated entries):
    name=<kind>:<gguf>:<mmproj_or_empty>:<hf_path_or_empty>
  kind     : "llm" or "vlm"
  gguf     : path to GGUF for the C runners
  mmproj   : vision projector GGUF (VLM only; empty for LLM variants)
  hf_path  : HF model dir or hub id for pytorch-cuda (empty if unavailable)

  qwen35-4b=llm:/mnt/.../Qwen3.5-4B-UD-Q8_K_XL.gguf::Qwen/Qwen3.5-4B
  gemma4-4b=vlm:/mnt/.../gemma-4-E4B-it-UD-Q8_K_XL.gguf:/mnt/.../mmproj-F16.gguf:google/gemma-4-4b-it

Endpoints:
  GET  /v1/models       - list variants + kind + advertised backends
  POST /v1/chat         - run one backend + variant, return generated text
  POST /v1/similarity   - compute agreement metrics between two texts
                          (token-level via HF tokenizer + char-level Levenshtein)
  OPTIONS *             - CORS preflight

Request /v1/chat:
    {
      "inputs":  {"text": "<user>", "image_base64": "<optional>"},
      "params":  {"backend": "...", "variant": "...",
                  "max_tokens": 128, "system": "...", "seed": 0}
    }

Response /v1/chat:
    {
      "ok": true,
      "model": "llm",
      "backend": "...",
      "variant": "...",
      "outputs": [{"type": "text", "text": "...", "tokens": [int...]}],
      "timings_ms": {"total": 1234, "gen": 1200}
    }

Request /v1/similarity:
    {
      "variant": "<name>",
      "ref":     "<reference text>",
      "hyp":     "<hypothesis text>"
    }

Response:
    {
      "ok": true,
      "method": "token-agreement|char-levenshtein",
      "tokenizer": "<path or ''>",
      "metrics": {
        "len_ref": N, "len_hyp": M,
        "common_prefix_tokens": K,
        "token_match_rate": 0..1,
        "first_divergence_pos": K,
        "char_len_ref": A, "char_len_hyp": B,
        "char_edit_distance": E,
        "char_similarity": 1 - E/max(A,B),
      }
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


# ---------------- HTTP helpers ----------------

def _cors(h):
    h.send_header("access-control-allow-origin", "*")
    h.send_header("access-control-allow-methods", "POST, GET, OPTIONS")
    h.send_header("access-control-allow-headers", "content-type")


def _json(h, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    h.send_response(status)
    h.send_header("content-type", "application/json; charset=utf-8")
    h.send_header("content-length", str(len(body)))
    _cors(h)
    h.end_headers()
    h.wfile.write(body)


# ---------------- Variant discovery ----------------

def _detect_family(gguf_path):
    """Best-effort architecture sniff from the GGUF metadata header."""
    if not gguf_path or not os.path.isfile(gguf_path):
        return "unknown"
    try:
        with open(gguf_path, "rb") as f:
            head = f.read(1 << 16)
    except OSError:
        return "unknown"
    if b"gemma4" in head: return "gemma4"
    if b"qwen35" in head or b"qwen3vl" in head or b"qwen3" in head:
        return "qwen"
    return "unknown"


class Variant:
    __slots__ = ("name", "kind", "family", "gguf", "mmproj", "hf_path")

    def __init__(self, name, kind, gguf, mmproj, hf_path, family=None):
        self.name    = name
        self.kind    = kind
        self.gguf    = gguf or ""
        self.mmproj  = mmproj or ""
        self.hf_path = hf_path or ""
        if kind not in ("llm", "vlm"):
            raise ValueError(f"variant '{name}': kind must be llm|vlm, got '{kind}'")
        if kind == "vlm" and not self.mmproj:
            raise ValueError(f"variant '{name}': mmproj is required for vlm")
        if self.gguf and not os.path.isfile(self.gguf):
            raise FileNotFoundError(f"variant '{name}': gguf missing: {self.gguf}")
        if self.mmproj and not os.path.isfile(self.mmproj):
            raise FileNotFoundError(f"variant '{name}': mmproj missing: {self.mmproj}")
        self.family = family or _detect_family(self.gguf)

    def to_json(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "family": self.family,
            "has_gguf":   bool(self.gguf),
            "has_mmproj": bool(self.mmproj),
            "has_hf":     bool(self.hf_path),
        }


def parse_variants_arg(s):
    out = []
    if not s: return out
    for piece in s.split(","):
        piece = piece.strip()
        if not piece: continue
        if "=" not in piece:
            raise ValueError(f"bad --variants entry '{piece}'; want name=kind:gguf:mmproj:hf[:family=...]")
        name, spec = piece.split("=", 1)
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(f"bad spec '{spec}'; want kind:gguf[:mmproj[:hf[:family=...]]]")
        family = None
        if parts and parts[-1].startswith("family="):
            family = parts[-1][len("family="):].strip() or None
            parts = parts[:-1]
        kind = parts[0].strip()
        gguf = parts[1].strip() if len(parts) > 1 else ""
        mmproj = parts[2].strip() if len(parts) > 2 else ""
        hf_path = ":".join(parts[3:]).strip() if len(parts) > 3 else ""
        out.append(Variant(name.strip(), kind, gguf, mmproj, hf_path, family=family))
    return out


# ---------------- Output cleaning ----------------

# test_transformer prints the prompt first, then generated tokens, then \n.
# We ask the C runner to take the *chat-formatted* prompt so the response
# starts right after the user's last turn marker — the cleanup below strips
# the prompt echo if the binary prepends it verbatim.

def _strip_prompt_echo(text, prompt):
    if prompt and text.startswith(prompt):
        return text[len(prompt):]
    return text


def _tail(s, n=2000):
    if s is None: return ""
    s = s if isinstance(s, str) else s.decode("utf-8", "replace")
    return s[-n:]


# ---------------- Backends ----------------

class OursCpuLlmBackend:
    """cpu/llm/test_transformer <gguf> <prompt> <max_gen> <n_threads>

    Deterministic greedy decode. Output: prompt echoed to stdout, then the
    generation, then '\\n'. We strip the prompt echo.
    """
    def __init__(self, bin_path, n_threads):
        self.bin = os.path.abspath(bin_path)
        self.n_threads = n_threads
        self.lock = threading.Lock()

    def available(self): return os.path.isfile(self.bin)
    def supports(self, variant): return variant.kind == "llm" and bool(variant.gguf)

    def infer(self, variant, prompt, image_png_bytes, params):
        max_toks = int(params.get("max_tokens") or 128)
        cmd = [self.bin, variant.gguf, prompt, str(max_toks), str(self.n_threads)]
        t0 = time.time()
        with self.lock:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        dt_ms = int((time.time() - t0) * 1000)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ours-cpu-llm failed (rc={proc.returncode}):\n"
                f"stderr tail:\n{_tail(proc.stderr)}")
        out = proc.stdout or ""
        out = _strip_prompt_echo(out, prompt)
        out = out.rstrip("\n")
        return {"text": out, "tokens": [], "dt_ms": dt_ms}


class OursGemma4VlmBackend:
    """cpu/vlm/test_gemma4_vision / cuda/vlm/test_cuda_gemma4_vlm:
       <model> <mmproj> [image] [prompt] [max_gen] [--budget N]
    """
    def __init__(self, bin_path, label, budget):
        self.bin = os.path.abspath(bin_path)
        self.label = label
        self.budget = budget
        self.lock = threading.Lock()

    def available(self): return os.path.isfile(self.bin)
    def supports(self, variant):
        return (variant.kind == "vlm"
                and variant.family == "gemma4"
                and bool(variant.gguf) and bool(variant.mmproj))

    def infer(self, variant, prompt, image_png_bytes, params):
        max_toks = int(params.get("max_tokens") or 128)
        with self.lock, tempfile.TemporaryDirectory() as td:
            img_path = None
            if image_png_bytes:
                img_path = os.path.join(td, "in.png")
                with open(img_path, "wb") as f:
                    f.write(image_png_bytes)
            cmd = [self.bin, variant.gguf, variant.mmproj]
            if img_path:
                cmd.append(img_path)
            cmd.extend([prompt, str(max_toks)])
            if self.budget is not None:
                cmd.extend(["--budget", str(self.budget)])
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            dt_ms = int((time.time() - t0) * 1000)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.label} failed (rc={proc.returncode}):\n"
                f"stderr tail:\n{_tail(proc.stderr)}")
        # Gemma4 vlm prints the visible channel to stdout (reasoning silenced).
        out = (proc.stdout or "").strip("\n")
        return {"text": out, "tokens": [], "dt_ms": dt_ms}


class OursQwenVlmBackend:
    """cpu/vlm/test_vision (Qwen3-VL / Qwen3.6-VL):
       <model> <mmproj> [image] [prompt] [max_gen]

    No reasoning-budget flag. Chat template is resolved internally from the
    main GGUF (chatml + vision). Visible output goes to stdout.
    """
    def __init__(self, bin_path, label):
        self.bin = os.path.abspath(bin_path)
        self.label = label
        self.lock = threading.Lock()

    def available(self): return os.path.isfile(self.bin)
    def supports(self, variant):
        return (variant.kind == "vlm"
                and variant.family == "qwen"
                and bool(variant.gguf) and bool(variant.mmproj))

    def infer(self, variant, prompt, image_png_bytes, params):
        max_toks = int(params.get("max_tokens") or 128)
        with self.lock, tempfile.TemporaryDirectory() as td:
            img_path = None
            if image_png_bytes:
                img_path = os.path.join(td, "in.png")
                with open(img_path, "wb") as f:
                    f.write(image_png_bytes)
            cmd = [self.bin, variant.gguf, variant.mmproj]
            if img_path:
                cmd.append(img_path)
            cmd.extend([prompt, str(max_toks)])
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            dt_ms = int((time.time() - t0) * 1000)
        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.label} failed (rc={proc.returncode}):\n"
                f"stderr tail:\n{_tail(proc.stderr)}")
        out = (proc.stdout or "").strip("\n")
        return {"text": out, "tokens": [], "dt_ms": dt_ms}


class PytorchRefBackend:
    """ref/llm/llm_ref.py --kind {llm|vlm} --model <hf> --prompt <text>
         [--image <png>] --max-tokens N --out out.json --device cuda
    """
    def __init__(self, script_path, python_exe):
        self.script = os.path.abspath(script_path)
        self.python = python_exe
        self.lock = threading.Lock()

    def available(self): return os.path.isfile(self.script)
    def supports(self, variant): return bool(variant.hf_path)

    def infer(self, variant, prompt, image_png_bytes, params):
        max_toks = int(params.get("max_tokens") or 128)
        system = params.get("system") or ""
        seed = int(params.get("seed") or 0)
        with self.lock, tempfile.TemporaryDirectory() as td:
            out_json = os.path.join(td, "out.json")
            cmd = [self.python, self.script,
                   "--kind", variant.kind,
                   "--model", variant.hf_path,
                   "--prompt", prompt,
                   "--max-tokens", str(max_toks),
                   "--out", out_json]
            if system:
                cmd.extend(["--system", system])
            if seed:
                cmd.extend(["--seed", str(seed)])
            if variant.kind == "vlm":
                if not image_png_bytes:
                    raise RuntimeError("pytorch-cuda: image required for vlm variant")
                img_path = os.path.join(td, "in.png")
                with open(img_path, "wb") as f:
                    f.write(image_png_bytes)
                cmd.extend(["--image", img_path])
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            dt_ms = int((time.time() - t0) * 1000)
            if proc.returncode != 0 or not os.path.isfile(out_json):
                raise RuntimeError(
                    f"pytorch-cuda failed (rc={proc.returncode}):\n"
                    f"stderr tail:\n{_tail(proc.stderr)}")
            with open(out_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
        return {
            "text": payload.get("text", ""),
            "tokens": payload.get("tokens", []),
            "dt_ms": dt_ms,
        }


# ---------------- Similarity ----------------

_TOKENIZERS = {}  # hf_path -> tokenizer
_TOK_LOCK = threading.Lock()


def _load_tokenizer(hf_path):
    if not hf_path: return None
    with _TOK_LOCK:
        if hf_path in _TOKENIZERS:
            return _TOKENIZERS[hf_path]
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[llm] tokenizer load failed for {hf_path}: {e}\n")
            tok = None
        _TOKENIZERS[hf_path] = tok
        return tok


def _levenshtein(a, b):
    if a == b: return 0
    if not a:  return len(b)
    if not b:  return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compute_similarity(variant, ref_text, hyp_text):
    """Returns (method, tokenizer_name, metrics_dict)."""
    tok = _load_tokenizer(variant.hf_path) if variant else None
    ref_ids = tok(ref_text, add_special_tokens=False).input_ids if tok else None
    hyp_ids = tok(hyp_text, add_special_tokens=False).input_ids if tok else None

    char_ed = _levenshtein(ref_text or "", hyp_text or "")
    char_max = max(len(ref_text or ""), len(hyp_text or ""), 1)

    metrics = {
        "char_len_ref": len(ref_text or ""),
        "char_len_hyp": len(hyp_text or ""),
        "char_edit_distance": char_ed,
        "char_similarity": 1.0 - char_ed / char_max,
    }
    method = "char-levenshtein"
    tok_name = ""

    if ref_ids is not None and hyp_ids is not None:
        k = 0
        n = min(len(ref_ids), len(hyp_ids))
        while k < n and ref_ids[k] == hyp_ids[k]:
            k += 1
        metrics.update({
            "len_ref": len(ref_ids),
            "len_hyp": len(hyp_ids),
            "common_prefix_tokens": k,
            "first_divergence_pos": k if (k < n or len(ref_ids) != len(hyp_ids)) else len(ref_ids),
            "token_match_rate": (k / max(len(ref_ids), len(hyp_ids), 1)),
        })
        method = "token-agreement"
        tok_name = variant.hf_path if variant else ""

    return method, tok_name, metrics


# ---------------- HTTP dispatch ----------------

def make_handler(backends, variants, default_variant, web_root):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[llm] " + (fmt % a) + "\n")

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
            if path in ("/", "/llm", "/llm_chat", "/llm.html"):
                self._serve_file(os.path.join(web_root, "llm_chat.html"),
                                 "text/html; charset=utf-8")
                return
            if path in ("/image_chat", "/image-chat", "/image-explain",
                        "/image_chat.html"):
                self._serve_file(os.path.join(web_root, "image_chat.html"),
                                 "text/html; charset=utf-8")
                return
            if path == "/health":
                _json(self, 200, {"ok": True,
                                  "backends": sorted(backends.keys()),
                                  "variants": [v.to_json() for v in variants.values()]})
                return
            if path in ("/models", "/v1/models"):
                _json(self, 200, {"ok": True, "models": [{
                    "id": "llm",
                    "tasks": ["text-generation", "image-text-to-text"],
                    "backends": sorted(backends.keys()),
                    "default_variant": default_variant,
                    "variants": [v.to_json() for v in variants.values()],
                }]})
                return
            if path.startswith("/public/") or path.startswith("/assets/"):
                safe = os.path.normpath(path.lstrip("/"))
                if ".." in safe.split("/"):
                    self.send_error(400); return
                fp = os.path.join(web_root, safe)
                if not os.path.isfile(fp):
                    self.send_error(404); return
                ctype = "image/jpeg" if fp.endswith((".jpg", ".jpeg")) else \
                        "image/png" if fp.endswith(".png") else \
                        "application/octet-stream"
                self._serve_file(fp, ctype); return
            self.send_error(404)

        def _read_json(self):
            n = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(n) if n else b""
            return json.loads(raw.decode("utf-8"))

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/v1/chat":
                    self._handle_chat()
                elif path == "/v1/similarity":
                    self._handle_similarity()
                else:
                    self.send_error(404)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write("[llm] unhandled error: " + traceback.format_exc() + "\n")
                _json(self, 500, {"ok": False, "error": str(e)})

        def _handle_chat(self):
            try:
                req = self._read_json()
            except Exception as e:  # noqa: BLE001
                _json(self, 400, {"ok": False, "error": f"bad json: {e}"}); return
            t_all = time.time()
            inputs = req.get("inputs") or {}
            params = req.get("params") or {}
            prompt = str(inputs.get("text") or "").strip()
            if not prompt:
                _json(self, 400, {"ok": False, "error": "inputs.text required"}); return
            img_b64 = inputs.get("image_base64") or ""
            img_bytes = base64.b64decode(img_b64) if img_b64 else b""

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
            be = backends[be_name]
            variant = variants[var_name]
            if not be.supports(variant):
                _json(self, 400, {"ok": False,
                                  "error": f"backend '{be_name}' does not support variant "
                                           f"'{var_name}' (kind={variant.kind})"}); return
            try:
                r = be.infer(variant, prompt, img_bytes, params)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write("[llm] infer error: " + traceback.format_exc() + "\n")
                _json(self, 500, {"ok": False, "error": str(e)}); return

            dt_ms = int((time.time() - t_all) * 1000)
            _json(self, 200, {
                "ok": True,
                "model": "llm",
                "backend": be_name,
                "variant": var_name,
                "variant_kind": variant.kind,
                "outputs": [{
                    "type": "text",
                    "text": r.get("text", ""),
                    "tokens": r.get("tokens", []),
                }],
                "timings_ms": {"total": dt_ms, "gen": r.get("dt_ms", dt_ms)},
            })

        def _handle_similarity(self):
            try:
                req = self._read_json()
            except Exception as e:  # noqa: BLE001
                _json(self, 400, {"ok": False, "error": f"bad json: {e}"}); return
            var_name = str(req.get("variant") or default_variant).strip()
            ref_text = req.get("ref") or ""
            hyp_text = req.get("hyp") or ""
            variant = variants.get(var_name)
            method, tok_name, metrics = compute_similarity(variant, ref_text, hyp_text)
            _json(self, 200, {
                "ok": True,
                "method": method,
                "tokenizer": tok_name,
                "variant": var_name,
                "metrics": metrics,
            })

    return H


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--cpu-llm-bin", required=True)
    ap.add_argument("--cpu-vlm-bin", required=True,
                    help="gemma4-family VLM binary (CPU)")
    ap.add_argument("--cuda-vlm-bin", required=True,
                    help="gemma4-family VLM binary (CUDA)")
    ap.add_argument("--cpu-qwen-vlm-bin", default="cpu/vlm/test_vision",
                    help="qwen-family VLM binary (CPU)")
    ap.add_argument("--cuda-qwen-vlm-bin", default="",
                    help="qwen-family VLM binary (CUDA), if available")
    ap.add_argument("--ref-script", required=True)
    ap.add_argument("--ref-python", default=sys.executable)
    ap.add_argument("--web-root", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--default-variant", default=None)
    ap.add_argument("--n-threads", type=int,
                    default=int(os.environ.get("OURS_CPU_THREADS", "8")))
    ap.add_argument("--budget", type=int, default=32,
                    help="gemma4 reasoning-token budget")
    ap.add_argument("--disable", default="",
                    help="comma-sep backend names to skip")
    args = ap.parse_args()

    disabled = {s.strip() for s in args.disable.split(",") if s.strip()}

    variant_list = parse_variants_arg(args.variants)
    if not variant_list:
        raise SystemExit("no variants")
    variants = {v.name: v for v in variant_list}
    default_variant = args.default_variant or variant_list[0].name
    if default_variant not in variants:
        raise SystemExit(f"default variant '{default_variant}' not in {list(variants)}")

    all_bes = {
        "ours-cpu-llm":       OursCpuLlmBackend(args.cpu_llm_bin, args.n_threads),
        "ours-cpu-vlm":       OursGemma4VlmBackend(args.cpu_vlm_bin, "ours-cpu-vlm", args.budget),
        "ours-cuda-vlm":      OursGemma4VlmBackend(args.cuda_vlm_bin, "ours-cuda-vlm", args.budget),
        "ours-cpu-qwen-vlm":  OursQwenVlmBackend(args.cpu_qwen_vlm_bin, "ours-cpu-qwen-vlm"),
        "ours-cuda-qwen-vlm": OursQwenVlmBackend(args.cuda_qwen_vlm_bin, "ours-cuda-qwen-vlm")
                               if args.cuda_qwen_vlm_bin else None,
        "pytorch-cuda":       PytorchRefBackend(args.ref_script, args.ref_python),
    }
    all_bes = {k: v for k, v in all_bes.items() if v is not None}
    backends = {}
    for name, be in all_bes.items():
        if name in disabled:
            sys.stderr.write(f"[llm] backend '{name}' disabled by --disable\n"); continue
        if not be.available():
            path = getattr(be, "bin", None) or getattr(be, "script", "<?>")
            sys.stderr.write(f"[llm] backend '{name}' not present at {path}\n"); continue
        backends[name] = be

    handler = make_handler(backends, variants, default_variant, args.web_root)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    sys.stderr.write(f"[llm] listening on http://{args.host}:{args.port}  "
                     f"backends={sorted(backends.keys())}  web_root={args.web_root}\n")
    sys.stderr.write(f"[llm] variants: {', '.join(v.name for v in variant_list)}\n")
    sys.stderr.write(f"[llm] default_variant={default_variant}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[llm] shutting down\n")


if __name__ == "__main__":
    main()
