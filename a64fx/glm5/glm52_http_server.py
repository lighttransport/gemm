#!/usr/bin/env python3
"""Small serialized HTTP gateway for the 12-node GLM-5.2 Q2 runner."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from glm5_tokenizer import Tok, TOKJSON  # noqa: E402


class Service:
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tok(args.tokenizer)
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = 0

    def complete(self, body):
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        max_tokens = body.get("max_tokens", self.args.max_tokens)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= self.args.max_tokens:
            raise ValueError("max_tokens must be an integer in [1, %d]" % self.args.max_tokens)
        ids = self.tokenizer.chat(prompt, think=bool(body.get("think", False)))
        if len(ids) + max_tokens > self.args.max_context:
            raise ValueError("prompt plus max_tokens exceeds max_context")

        with self.lock:
            self.requests += 1
            request_id = "glm52-%d-%06d" % (int(time.time()), self.requests)
            with tempfile.TemporaryDirectory(prefix=request_id + "-", dir=self.args.work_dir) as td:
                prompt_file = Path(td) / "prompt.ids"
                output_file = Path(td) / "generated.ids"
                prompt_file.write_text(" ".join(map(str, ids)) + "\n")
                ctx = len(ids) + max_tokens + 128
                cmd = [
                    self.args.runner, "generate", "--no-stage",
                    "--no-enforce", "--stable-outputs", "--ctx", str(ctx),
                    "--prompt-ids", str(prompt_file), "--gen-out", str(output_file),
                    "--max-new", str(max_tokens), "--min-new", "0",
                ]
                if len(ids) >= self.args.int4_threshold:
                    cmd += ["--kv-tier-bf16=0"]
                started = time.time()
                proc = subprocess.run(cmd, cwd=HERE, universal_newlines=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, timeout=self.args.timeout)
                if proc.returncode or not output_file.is_file():
                    tail = (proc.stderr or proc.stdout)[-2000:]
                    raise RuntimeError("runner failed (exit %d): %s" % (proc.returncode, tail))
                generated = [int(x) for x in output_file.read_text().split()]
                text = self.tokenizer.decode(generated)
                return {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(started),
                    "model": "glm-5.2-q2-a64fx-ep12",
                    "choices": [{"text": text, "index": 0,
                                 "finish_reason": "stop" if len(generated) < max_tokens else "length"}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated),
                              "total_tokens": len(ids) + len(generated)},
                    "elapsed_seconds": round(time.time() - started, 3),
                }


class Handler(BaseHTTPRequestHandler):
    server_version = "glm52-http/1"

    def send_json(self, status, value):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            s = self.server.service
            self.send_json(200, {"status": "ok", "busy": s.lock.locked(),
                                 "requests": s.requests, "uptime_seconds": int(time.time()-s.started)})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/completions", "/generate"):
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.server.service.args.max_body:
                raise ValueError("invalid request body size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
            self.send_json(200, self.server.service.complete(body))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "runner timed out"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--tokenizer", default=TOKJSON)
    p.add_argument("--runner", default=str(HERE / "run_glm52_q2_12n.sh"))
    p.add_argument("--work-dir", default=str(HERE / "logs"))
    p.add_argument("--max-context", type=int, default=262144)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--int4-threshold", type=int, default=23000)
    p.add_argument("--max-body", type=int, default=8 << 20)
    p.add_argument("--timeout", type=int, default=21600)
    args = p.parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.service = Service(args)
    print("GLM52 HTTP listening on http://%s:%d" % (args.host, args.port), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
