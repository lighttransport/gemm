#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim for the persistent HIP runner.

The GPU process speaks a deliberately tiny base64 line protocol.  This layer
owns JSON, HTTP, OpenAI response shapes, message canonicalization, and request
limits, while the child keeps the model and KV/SSM state resident.
"""
import argparse
import base64
import json
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") in ("text", None))
    return ""


def chat_prompt(messages):
    # Stable ChatML-like framing gives the backend an exact token prefix to
    # reuse when an agent resends its prior conversation plus one new turn.
    out = []
    for m in messages:
        role = m.get("role", "user")
        text = content_text(m.get("content", ""))
        out.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    out.append("<|im_start|>assistant\n")
    return "".join(out)


def fit_context(messages, context_tokens, output_tokens):
    """Keep system/developer instructions and the newest turns.

    The tokenizer lives in the GPU child, so this uses a conservative 4-byte
    estimate and lets the child report the exact usage.  It prevents an agent
    from silently pushing the system prompt out of the context window.
    """
    budget = max(128, context_tokens - output_tokens) * 4
    if len(chat_prompt(messages)) <= budget:
        return messages
    pinned = [m for m in messages if m.get("role") in ("system", "developer")]
    recent = [m for m in messages if m.get("role") not in ("system", "developer")]
    kept = list(pinned)
    for message in reversed(recent):
        trial = kept + [message]
        if len(chat_prompt(trial)) > budget and len(kept) > len(pinned):
            break
        kept.append(message)
    return sorted(kept, key=lambda m: messages.index(m))


class Backend:
    def __init__(self, args):
        cmd = [args.runner, args.model, "--stdio-server", "--gpu-only-bench", "-s", str(args.context)]
        if args.moe_cache_mb:
            cmd += ["--moe-cache-mb", str(args.moe_cache_mb)]
        if args.coding:
            cmd += ["--coding"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.model = args.model.rsplit("/", 1)[-1]

    def generate(self, prompt, max_tokens, temperature, top_p, top_k, presence):
        payload = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        line = f"REQ {max_tokens} {temperature} {top_p} {top_k} {presence} {payload}\n"
        with self.lock:
            if self.proc.poll() is not None:
                raise RuntimeError("runner exited")
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
            result = self.proc.stdout.readline().strip()
        if not result.startswith("OK "):
            raise RuntimeError(result)
        fields = result.split(" ", 4)
        if len(fields) != 5:
            raise RuntimeError("malformed runner response")
        cached, prompt_tokens, completion_tokens, finish, encoded = fields
        text = base64.b64decode(encoded).decode("utf-8", "replace")
        return text, int(cached), int(prompt_tokens), int(completion_tokens), finish


class Handler(BaseHTTPRequestHandler):
    backend = None
    model = "local"
    max_tokens = 256
    context = 4096

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] " + (fmt % args) + "\n")

    def send_json(self, status, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/v1/models":
            now = int(time.time())
            self.send_json(200, {"object": "list", "data": [{"id": self.model, "object": "model", "created": now, "owned_by": "local"}]})
        else:
            self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions", "/v1/responses"):
            self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n))
            if self.path == "/v1/completions":
                prompt = req.get("prompt", "")
                if isinstance(prompt, list): prompt = "".join(map(str, prompt))
                messages = [{"role": "user", "content": prompt}]
            elif self.path == "/v1/responses":
                messages = []
                if req.get("instructions"): messages.append({"role": "system", "content": req["instructions"]})
                inp = req.get("input", "")
                if isinstance(inp, str): messages.append({"role": "user", "content": inp})
                else: messages.extend(inp)
            else:
                messages = req.get("messages", [])
            limit = min(int(req.get("max_tokens", req.get("max_output_tokens", self.max_tokens))), self.max_tokens)
            messages = fit_context(messages, self.context, limit)
            prompt = chat_prompt(messages)
            temp = float(req.get("temperature", 0.2))
            top_p = float(req.get("top_p", 0.95))
            top_k = int(req.get("top_k", 20))
            presence = float(req.get("presence_penalty", 0.0))
            text, cached, ptok, ctok, finish = self.backend.generate(prompt, limit, temp, top_p, top_k, presence)
            ident = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            usage = {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok, "cached_tokens": cached}
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                if self.path == "/v1/responses":
                    response_id = "resp-" + uuid.uuid4().hex
                    created_obj = {"type": "response.created", "response": {"id": response_id, "object": "response", "status": "in_progress", "model": self.model}}
                    delta_obj = {"type": "response.output_text.delta", "item_id": response_id + "-item", "output_index": 0, "content_index": 0, "delta": text}
                    done_obj = {"type": "response.completed", "response": {"id": response_id, "object": "response", "status": "completed", "model": self.model, "usage": usage}}
                    for event, obj in (("response.created", created_obj), ("response.output_text.delta", delta_obj), ("response.completed", done_obj)):
                        self.wfile.write(("event: " + event + "\ndata: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                else:
                    if text:
                        obj = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                    obj = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                    self.wfile.write(("data: " + json.dumps(obj) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
                return
            if self.path == "/v1/responses":
                self.send_json(200, {"id": "resp-" + uuid.uuid4().hex, "object": "response", "model": self.model, "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}], "status": "completed", "usage": usage})
            elif self.path == "/v1/completions":
                self.send_json(200, {"id": ident, "object": "text_completion", "created": created, "model": self.model, "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage})
            else:
                self.send_json(200, {"id": ident, "object": "chat.completion", "created": created, "model": self.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}], "usage": usage})
        except Exception as exc:
            self.send_json(500, {"error": {"message": str(exc), "type": "server_error"}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--runner", default="./test_hip_llm")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--max-output", type=int, default=256)
    ap.add_argument("--moe-cache-mb", type=int, default=0)
    ap.add_argument("--coding", action="store_true")
    args = ap.parse_args()
    Handler.backend = Backend(args)
    Handler.model = Handler.backend.model
    Handler.max_tokens = args.max_output
    Handler.context = args.context
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        Handler.backend.proc.terminate()


if __name__ == "__main__":
    main()
