#!/usr/bin/env python3
"""llama-server-like HTTP API for the DeepSeek-V4-Flash 11-node EP runner (DS4F_SERVE mode).

Runs on the controller node; drives the persistent `ds4f_ep_runner` (loaded once, looping on
requests) over shared-FS files:  <BASE>.req / .reqseq  (prompt in) and  <BASE>.resp / .respseq
(generated ids out). The runner's 11 ranks all read the same request -> lockstep, no broadcast.

Endpoints:
  POST /v1/completions   {"prompt": str, "max_tokens": int}   (OpenAI text-completion shape)
  POST /completion       {"prompt": str, "n_predict": int}     (llama.cpp shape)
  GET  /health

Env: PORT (8080), TOK (~/models/ds4f/tokenizer.json), DS4F_SERVE_BASE, DS4F_SERVE_TIMEOUT (1200s).
Start via run_ds4f_serve_11n.sh (which launches the runner first, then this)."""
import http.server, json, os, socketserver, subprocess, sys, tempfile, threading, time


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.environ.get("TOK", os.path.expanduser("~/models/ds4f/tokenizer.json"))
TOKCLI = os.path.join(HERE, "tools", "ds4f_tokenizer.py")
BASE = os.environ.get("DS4F_SERVE_BASE", "/tmp/ds4f_serve")
REQ, RESP, REQSEQ, RESPSEQ = BASE + ".req", BASE + ".resp", BASE + ".reqseq", BASE + ".respseq"
PORT = int(os.environ.get("PORT", "8080"))
TIMEOUT = float(os.environ.get("DS4F_SERVE_TIMEOUT", "1200"))
_lock = threading.Lock()        # the runner is single-stream: serialize requests
_seq = 0


def _tok(args):
    return subprocess.run([sys.executable, TOKCLI] + args, capture_output=True, text=True)


def encode(text):
    pf = of = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(text); pf = f.name
        of = pf + ".ids"
        _tok(["encode", "--tokenizer", TOK, "--prompt-file", pf, "--out", of])
        with open(of) as g:
            return [int(x) for x in g.read().split()]
    finally:
        for p in (pf, of):
            if p:
                try: os.unlink(p)
                except OSError: pass


def decode(ids):
    idf = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as f:
            f.write(" ".join(map(str, ids))); idf = f.name
        return _tok(["decode", "--tokenizer", TOK, "--ids-file", idf]).stdout
    finally:
        if idf:
            try: os.unlink(idf)
            except OSError: pass


def infer(prompt, max_tokens):
    global _seq
    with _lock:
        ids = encode(prompt)
        if not ids:
            return [], [], ""
        with open(REQ, "w") as f:
            f.write(str(max_tokens) + "\n" + " ".join(map(str, ids)) + "\n")
        _seq += 1
        with open(REQSEQ, "w") as f:
            f.write(str(_seq) + "\n")                # write req then bump seq -> runner reads a complete file
        t0 = time.time()
        while True:
            try:
                with open(RESPSEQ) as f:
                    rs = int(f.read().strip() or 0)
            except (OSError, ValueError):
                rs = 0
            if rs >= _seq:
                break
            if time.time() - t0 > TIMEOUT:
                raise TimeoutError("runner timeout")
            time.sleep(0.01)
        with open(RESP) as f:
            gen = [int(x) for x in f.read().split()]
        return ids, gen, decode(gen)


class H(http.server.BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._json(200, {"status": "ok", "model": "ds4f"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": "bad json"})
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(map(str, prompt))
        max_tokens = int(body.get("max_tokens", body.get("n_predict", 128)))
        t0 = time.time()
        try:
            ids, gen, text = infer(prompt, max_tokens)
        except TimeoutError:
            return self._json(504, {"error": "runner timeout"})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        stop = bool(gen and gen[-1] == 1)   # DS4F_EOS_ID == 1
        self._json(200, {
            "id": "cmpl-ds4f", "object": "text_completion", "model": "ds4f",
            "choices": [{"text": text, "index": 0, "finish_reason": "stop" if stop else "length"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen),
                      "total_tokens": len(ids) + len(gen)},
            "timings": {"wall_s": round(time.time() - t0, 3)},
        })

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    try:
        with open(RESPSEQ, "w") as f:
            f.write("0\n")                    # clear any stale response marker
    except OSError:
        pass
    print(f"[ds4f-serve] listening on http://0.0.0.0:{PORT}", flush=True)
    print(f"[ds4f-serve]   POST /v1/completions  {{\"prompt\": \"...\", \"max_tokens\": 128}}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
