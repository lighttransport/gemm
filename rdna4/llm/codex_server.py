#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim for the persistent HIP runner.

The GPU process speaks a deliberately tiny base64 line protocol.  This layer
owns JSON, HTTP, OpenAI response shapes, message canonicalization, and request
limits, while the child keeps the model and KV/SSM state resident.
"""
import argparse
import base64
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Responses API messages use input_text/output_text, while Chat
        # Completions uses text.  Dropping input_text silently turns a real
        # user request into an empty message and makes the model answer the
        # surrounding Codex system prompt instead.
        return "".join(x.get("text", "") for x in content
                       if isinstance(x, dict) and
                       x.get("type") in ("text", "input_text", "output_text", None))
    return ""


def chat_prompt(messages):
    # Stable ChatML-like framing gives the backend an exact token prefix to
    # reuse when an agent resends its prior conversation plus one new turn.
    out = []
    for m in messages:
        role = m.get("role", "user")
        text = content_text(m.get("content", ""))
        out.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    # This checkpoint's non-thinking form closes the optional reasoning block
    # before answer tokens.  Leaving it open exposes the model's work plan as
    # assistant output even for a simple user message.
    out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(out)


def chat_prefix(messages):
    """Return the stable leading system/developer frames for prefix KV reuse."""
    out = []
    for m in messages:
        if m.get("role") not in ("system", "developer"):
            break
        role = m.get("role", "system")
        out.append(f"<|im_start|>{role}\n{content_text(m.get('content', ''))}<|im_end|>\n")
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
        self.coding = args.coding
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, text=True, bufsize=1)
        self.lock = threading.Lock()
        self.model = args.model.rsplit("/", 1)[-1]

    def cancel(self):
        """Request cooperative cancellation in the resident runner."""
        if self.proc.poll() is None:
            try:
                os.kill(self.proc.pid, signal.SIGUSR1)
            except ProcessLookupError:
                pass

    def generate(self, prompt, max_tokens, temperature, top_p, top_k, presence, prefix=""):
        prefix_payload = base64.b64encode(prefix.encode("utf-8")).decode("ascii") if prefix else "-"
        payload = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        line = f"REQ {max_tokens} {temperature} {top_p} {top_k} {presence} {prefix_payload} {payload}\n"
        with self.lock:
            if self.proc.poll() is not None:
                raise RuntimeError("runner exited")
            self.proc.stdin.write(line)
            self.proc.stdin.flush()
            # Preserve the final empty field: an immediate EOS is a valid
            # completion and the runner's OK line intentionally ends with an
            # empty base64 payload in that case.
            result = self.proc.stdout.readline().rstrip("\r\n")
        if not result.startswith("OK "):
            raise RuntimeError(result)
        fields = result.split(" ", 5)
        if len(fields) == 5 and fields[4] in ("stop", "length", "cancelled"):
            fields.append("")
        if len(fields) != 6:
            raise RuntimeError("malformed runner response")
        if fields[0] != "OK":
            raise RuntimeError("malformed runner response")
        cached, prompt_tokens, completion_tokens, finish, encoded = fields[1:]
        text = base64.b64decode(encoded).decode("utf-8", "replace")
        return text, int(cached), int(prompt_tokens), int(completion_tokens), finish


class Handler(BaseHTTPRequestHandler):
    backend = None
    model = "local"
    max_tokens = 256
    context = 4096
    coding = False

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] " + (fmt % args) + "\n")

    def send_json(self, status, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/health", "/v1/health"):
            self.send_json(200, {"status": "ok"})
        elif path in ("/v1/models", "/models"):
            now = int(time.time())
            self.send_json(200, {"object": "list", "data": [{"id": self.model, "object": "model", "created": now, "owned_by": "local"}]})
        else:
            self.log_message("404 GET %s", self.path)
            self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def _watch_disconnect(self, stop, cancelled):
        """Cancel inference when the client closes its request socket."""
        while not stop.wait(0.05):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    continue
                data = self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                if not data:
                    self.backend.cancel()
                    cancelled.set()
                    return
            except (BlockingIOError, InterruptedError):
                continue
            except (OSError, ValueError):
                self.backend.cancel()
                cancelled.set()
                return

    def do_POST(self):
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path.startswith("/v1/"):
            api_path = path
        elif path in ("/chat/completions", "/completions", "/responses"):
            api_path = "/v1" + path
        else:
            api_path = path
        if api_path == "/v1/cancel":
            self.backend.cancel()
            self.send_json(202, {"status": "cancellation_requested"})
            return
        if api_path not in ("/v1/chat/completions", "/v1/completions", "/v1/responses"):
            self.log_message("404 POST %s", self.path)
            self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n))
            if api_path == "/v1/completions":
                prompt = req.get("prompt", "")
                if isinstance(prompt, list): prompt = "".join(map(str, prompt))
                messages = [{"role": "user", "content": prompt}]
            elif api_path == "/v1/responses":
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
            prefix = chat_prefix(messages)
            # Match test_hip_llm's validated Qwen3.8 coding profile unless a
            # client explicitly supplies sampling controls.  Previously
            # --coding was passed only to the child binary, where it has no
            # effect on protocol requests; API calls therefore used a very
            # low-temperature generic profile that produces meta-commentary.
            # The local Qwen checkpoint is substantially more reliable for
            # Codex's terse control prompts with low-entropy sampling.  The
            # old coding defaults (T=0.7, presence=1.5) could turn a simple
            # confirmation into repeated fragments even though the request
            # and transport completed successfully.
            temp = float(req.get("temperature", 0.05 if self.coding else 0.2))
            top_p = float(req.get("top_p", 1.0 if self.coding else 0.95))
            top_k = int(req.get("top_k", 1 if self.coding else 20))
            presence = float(req.get("presence_penalty", 0.0))
            stop_watcher = threading.Event()
            cancelled = threading.Event()
            watcher = threading.Thread(target=self._watch_disconnect,
                                       args=(stop_watcher, cancelled), daemon=True)
            watcher.start()
            try:
                text, cached, ptok, ctok, finish = self.backend.generate(prompt, limit, temp, top_p, top_k, presence, prefix)
            finally:
                stop_watcher.set()
                watcher.join(timeout=0.2)
            if cancelled.is_set() or finish == "cancelled":
                self.log_message("request cancelled: %s", self.path)
                return
            ident = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            usage = {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok, "cached_tokens": cached}
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                if api_path == "/v1/responses":
                    response_id = "resp-" + uuid.uuid4().hex
                    item_id = response_id + "-item"
                    part = {"type": "output_text", "text": text, "annotations": []}
                    item = {"type": "message", "id": item_id, "role": "assistant", "status": "completed", "content": [part]}
                    response_base = {"id": response_id, "object": "response", "created_at": created,
                                     "status": "in_progress", "model": self.model, "output": []}
                    created_obj = {"type": "response.created", "response": response_base}
                    in_progress_obj = {"type": "response.in_progress", "response": response_base}
                    added_obj = {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []}}
                    part_added_obj = {"type": "response.content_part.added", "item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}
                    delta_obj = {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": text}
                    text_done_obj = {"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "content_index": 0, "text": text}
                    part_done_obj = {"type": "response.content_part.done", "item_id": item_id, "output_index": 0, "content_index": 0, "part": part}
                    item_done_obj = {"type": "response.output_item.done", "output_index": 0, "item": item}
                    response_done = {**response_base, "status": "completed", "output": [item], "usage": {"input_tokens": ptok, "output_tokens": ctok, "total_tokens": ptok + ctok, "input_tokens_details": {"cached_tokens": cached}}}
                    done_obj = {"type": "response.completed", "response": response_done}
                    events = (("response.created", created_obj), ("response.in_progress", in_progress_obj),
                              ("response.output_item.added", added_obj), ("response.content_part.added", part_added_obj),
                              ("response.output_text.delta", delta_obj), ("response.output_text.done", text_done_obj),
                              ("response.content_part.done", part_done_obj), ("response.output_item.done", item_done_obj),
                              ("response.completed", done_obj))
                    for event, obj in events:
                        self.wfile.write(("event: " + event + "\ndata: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                else:
                    if text:
                        obj = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                    obj = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                    self.wfile.write(("data: " + json.dumps(obj) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
                return
            if api_path == "/v1/responses":
                response_id = "resp-" + uuid.uuid4().hex
                item = {"type": "message", "id": response_id + "-item", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}
                self.send_json(200, {"id": response_id, "object": "response", "created_at": created,
                                     "model": self.model, "output": [item], "output_text": text,
                                     "status": "completed", "usage": {"input_tokens": ptok,
                                     "output_tokens": ctok, "total_tokens": ptok + ctok,
                                     "input_tokens_details": {"cached_tokens": cached}}})
            elif self.path == "/v1/completions":
                self.send_json(200, {"id": ident, "object": "text_completion", "created": created, "model": self.model, "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage})
            else:
                self.send_json(200, {"id": ident, "object": "chat.completion", "created": created, "model": self.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}], "usage": usage})
        except (BrokenPipeError, ConnectionResetError):
            # Clients commonly cancel a request after their own timeout. The
            # backend may finish its serialized inference, but there is no
            # socket left to report an error on.
            return
        except Exception as exc:
            self.log_message("500 POST %s: %s", self.path, exc)
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
    Handler.coding = args.coding
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        Handler.backend.proc.terminate()


if __name__ == "__main__":
    main()
