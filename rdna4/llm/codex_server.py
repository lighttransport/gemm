#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim for the persistent HIP runner.

The GPU process speaks a deliberately tiny base64 line protocol.  This layer
owns JSON, HTTP, OpenAI response shapes, message canonicalization, and request
limits, while the child keeps the model and KV/SSM state resident.
"""
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
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

from qwen_tools import call_events, parse_calls, tool_instructions, tool_registry


WEB_DIR = Path(__file__).with_name("web")


def _handle_sigterm(signum, frame):
    """Turn service-manager termination into the normal cleanup path."""
    del signum, frame
    raise KeyboardInterrupt


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


def responses_input_messages(value):
    """Normalize Responses input message objects and direct content items."""
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        return [{"role": value.get("role", "user"),
                 "content": value.get("content", value.get("text", ""))}]
    if not isinstance(value, list):
        return []
    messages = []
    direct = []
    for item in value:
        if isinstance(item, dict) and item.get("role"):
            messages.append(item)
        elif isinstance(item, dict) and item.get("type") == "function_call":
            # Responses represents the assistant's tool invocation as an
            # output item rather than a role-bearing message. Keep it when a
            # client sends the full prior turn back for the tool-result turn;
            # otherwise the tool output has no causal assistant context.
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, ValueError):
                parsed = {"input": str(arguments)}
            if not isinstance(parsed, dict):
                parsed = {"input": json.dumps(parsed, ensure_ascii=False)}
            call = ["<tool_call>", f"<function={name}>"]
            for key, value in parsed.items():
                rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                call.extend((f"<parameter={key}>", rendered, "</parameter>"))
            call.extend(("</function>", "</tool_call>"))
            messages.append({"role": "assistant", "content": "\n".join(call)})
        elif isinstance(item, dict) and item.get("type") == "function_call_output":
            # Responses sends tool results as input items rather than chat
            # messages. Preserve them as a tool turn; dropping them makes a
            # follow-up generation repeat the same call without its result.
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            call_id = item.get("call_id", "")
            label = f" call_id={call_id}" if call_id else ""
            messages.append({"role": "tool",
                             "content": f"<tool_response{label}>\n{output}\n</tool_response>"})
        elif isinstance(item, dict) and "content" in item:
            # Be liberal with message-shaped Responses items that omit role;
            # treating them as user content is safer than silently dropping
            # the actual request and answering only the system prompt.
            messages.append({"role": "user", "content": item["content"]})
        elif isinstance(item, dict) and item.get("type") in ("input_text", "output_text", "text"):
            direct.append(item)
    if direct:
        messages.append({"role": "user", "content": direct})
    return messages


def chat_prompt(messages):
    # Match the checkpoint's non-thinking ChatML template, including the
    # reasoning frame on historical assistant turns. Omitting it changes the
    # token prefix and discards reusable conversation KV on every follow-up.
    out = [chat_prefix(messages)]
    leading = True
    for m in messages:
        role = m.get("role", "user")
        if leading and role in ("system", "developer"):
            continue
        leading = False
        text = content_text(m.get("content", "")).strip()
        if role == "assistant":
            text = "<think>\n\n</think>\n\n" + text
        out.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    # Qwen3.8 Flash Next thinks by default.  Match llama.cpp's explicit
    # non-thinking mode by placing an empty reasoning block before the final
    # answer; otherwise <think> content leaks into the Responses text.
    out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(out)


def chat_prefix(messages):
    """Return the stable leading system/developer frames for prefix KV reuse."""
    out = []
    for m in messages:
        if m.get("role") not in ("system", "developer"):
            break
        text = content_text(m.get("content", "")).strip()
        if text:
            out.append(text)
    # Qwen3.8's GGUF template merges consecutive system/developer messages.
    return "<|im_start|>system\n" + "\n".join(out) + "<|im_end|>\n" if out else ""


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
        self.cancel_lock = threading.Lock()
        self.active_cancel = None
        self.ready = False
        self.model = args.model.rsplit("/", 1)[-1]
        try:
            ready_timeout = float(os.environ.get("QWEN38_READY_TIMEOUT", "300"))
            if not math.isfinite(ready_timeout) or ready_timeout <= 0:
                raise ValueError
            self._wait_ready(ready_timeout)
        except Exception:
            # A failed startup must not leave a model-sized child process
            # behind, especially when readiness times out or the port setup
            # fails immediately afterwards.
            self.close()
            raise

    def health(self):
        """Return readiness state without sending a request to the runner."""
        exit_status = self.proc.poll()
        return {
            "status": "ready" if self.ready and exit_status is None else "unavailable",
            "runner_alive": exit_status is None,
            "runner_exit_status": exit_status,
        }

    def close(self):
        """Stop and reap the resident runner during server shutdown."""
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def _wait_ready(self, timeout=None):
        """Wait until the resident runner has loaded the model."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while not self.ready:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("runner did not become ready before timeout")
                try:
                    fd = self.proc.stdout.fileno()
                    if not select.select([fd], [], [], remaining)[0]:
                        raise RuntimeError("runner did not become ready before timeout")
                except (AttributeError, io.UnsupportedOperation, ValueError):
                    # In-memory streams used by the protocol tests do not
                    # expose a selectable file descriptor.
                    pass
            raw = self.proc.stdout.readline()
            if not raw:
                status = self.proc.poll()
                if status is not None:
                    wait = getattr(self.proc, "wait", None)
                    if wait is not None:
                        wait()
                raise RuntimeError(
                    "runner exited before READY" +
                    (f" (status {status})" if status is not None else ""))
            if raw.rstrip("\r\n") == "READY":
                self.ready = True
            else:
                sys.stderr.write("[runner diagnostic] " + raw)

    def cancel(self, cancellation=None):
        """Request cooperative cancellation in the resident runner."""
        with self.cancel_lock:
            if self.active_cancel is None:
                return
            if cancellation is not None and cancellation is not self.active_cancel:
                return
            self.active_cancel.set()
            if self.proc.poll() is None:
                try:
                    os.kill(self.proc.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass

    def generate(self, prompt, max_tokens, temperature, top_p, top_k, presence, min_p,
                 prefix="", cancellation=None, on_token=None):
        cancellation = cancellation if cancellation is not None else threading.Event()
        prefix_payload = base64.b64encode(prefix.encode("utf-8")).decode("ascii") if prefix else "-"
        payload = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        line = f"REQ {max_tokens} {temperature} {top_p} {top_k} {presence} {min_p} {prefix_payload} {payload}\n"
        with self.lock:
            if self.proc.poll() is not None:
                raise RuntimeError("runner exited")
            # The constructor normally consumes READY; retain this check for
            # tests and callers that construct Backend without __init__.
            self._wait_ready()
            with self.cancel_lock:
                if cancellation.is_set():
                    return "", 0, 0, 0, "cancelled"
                self.active_cancel = cancellation
            # Preserve the final empty field: an immediate EOS is a valid
            # completion and the runner's OK line intentionally ends with an
            # empty base64 payload in that case.
            # HIP libraries may print diagnostics on stdout. Consume those
            # within the transaction: returning early leaves its OK queued
            # and makes the next HTTP request receive the previous answer.
            try:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
                while True:
                    raw = self.proc.stdout.readline()
                    if not raw:
                        raise RuntimeError("runner closed its response pipe")
                    result = raw.rstrip("\r\n")
                    if result.startswith("TOK "):
                        if on_token is not None:
                            try:
                                on_token(base64.b64decode(result[4:]).decode("utf-8", "replace"))
                            except (ValueError, UnicodeError):
                                sys.stderr.write("[runner diagnostic] malformed token frame\n")
                        continue
                    if result.startswith(("OK ", "ERR ")):
                        break
                    sys.stderr.write("[runner diagnostic] " + result + "\n")
            finally:
                # Clear ownership before another request can take self.lock.
                with self.cancel_lock:
                    self.active_cancel = None
        if not result.startswith("OK "):
            raise RuntimeError(result)
        fields = result.split(" ", 7)
        if len(fields) == 5 and fields[4] in ("stop", "length", "cancelled"):
            fields.append("")
        if len(fields) == 6:
            fields.extend(("0", "0"))
        if len(fields) != 8:
            raise RuntimeError("malformed runner response")
        if fields[0] != "OK":
            raise RuntimeError("malformed runner response")
        cached, prompt_tokens, completion_tokens, finish, encoded, prefill_ms, decode_ms = fields[1:]
        self.last_metrics = {"prompt_ms": float(prefill_ms), "generation_ms": float(decode_ms)}
        self.last_metrics["pp_tok_s"] = (1000.0 * (int(prompt_tokens) - int(cached)) / float(prefill_ms)
                                          if float(prefill_ms) > 0 else 0.0)
        self.last_metrics["tg_tok_s"] = (1000.0 * int(completion_tokens) / float(decode_ms)
                                          if float(decode_ms) > 0 else 0.0)
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

    def send_file(self, path, content_type):
        try:
            raw = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return True
        except FileNotFoundError:
            self.send_json(404, {"error": {"message": "UI asset not found", "type": "server_error"}})
            return False
        except (BrokenPipeError, ConnectionResetError):
            return False

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path in ("/", "/ui"):
            self.send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        elif path in ("/health", "/v1/health"):
            health = self.backend.health()
            self.send_json(200 if health["status"] == "ready" else 503, health)
        elif path == "/v1/ui/capabilities":
            self.send_json(200, {
                "model": self.model,
                "coding": self.coding,
                "research": True,
                "vision": False,
                "vision_note": "This Qwen endpoint is text-only; image upload is preview-only until a vision backend is connected.",
            })
        elif path in ("/v1/models", "/models"):
            now = int(time.time())
            self.send_json(200, {"object": "list", "data": [{"id": self.model, "object": "model", "created": now, "owned_by": "local"}]})
        else:
            self.log_message("404 GET %s", self.path)
            self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def _watch_disconnect(self, stop, cancelled):
        """Cancel inference when the client closes its request socket."""
        while not stop.wait(0.05):
            if cancelled.is_set():
                # Repeat until the transaction completes: the first signal
                # can arrive just before the runner reads its REQ line.
                self.backend.cancel(cancelled)
                continue
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if not readable:
                    continue
                data = self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                if not data:
                    cancelled.set()
                    self.backend.cancel(cancelled)
            except (BlockingIOError, InterruptedError):
                continue
            except (OSError, ValueError):
                cancelled.set()
                self.backend.cancel(cancelled)

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
            if n <= 0:
                self.send_json(400, {"error": {"message": "request body is required", "type": "invalid_request_error"}})
                return
            try:
                req = json.loads(self.rfile.read(n))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.send_json(400, {"error": {"message": f"invalid JSON: {exc}", "type": "invalid_request_error"}})
                return
            if not isinstance(req, dict):
                self.send_json(400, {"error": {"message": "request body must be a JSON object", "type": "invalid_request_error"}})
                return
            if api_path == "/v1/completions":
                prompt = req.get("prompt", "")
                if isinstance(prompt, list): prompt = "".join(map(str, prompt))
                messages = [{"role": "user", "content": prompt}]
            elif api_path == "/v1/responses":
                messages = []
                if req.get("instructions"): messages.append({"role": "system", "content": req["instructions"]})
                inp = req.get("input", "")
                messages.extend(responses_input_messages(inp))
            else:
                messages = req.get("messages", [])
            registry = tool_registry(req.get("tools", []))
            if registry:
                messages.insert(0, {"role": "system", "content": tool_instructions(registry)})
            try:
                requested_limit = int(req.get("max_tokens", req.get("max_output_tokens", self.max_tokens)))
            except (TypeError, ValueError):
                self.send_json(400, {"error": {"message": "max_tokens must be an integer", "type": "invalid_request_error"}})
                return
            if requested_limit < 0:
                self.send_json(400, {"error": {"message": "max_tokens must be non-negative", "type": "invalid_request_error"}})
                return
            limit = min(requested_limit, self.max_tokens)
            messages = fit_context(messages, self.context, limit)
            prompt = chat_prompt(messages)
            prefix = chat_prefix(messages)
            # Explicit API sampling controls override the requested coding
            # profile (T=1, top_p=.95, top_k=40, min_p=.01, no penalties).
            # The child receives these values through the request protocol;
            # its standalone benchmark sampling defaults do not apply here.
            default_temp, default_top_p, default_top_k, default_presence = (
                (1.0, 0.95, 40, 0.0) if self.coding else (0.2, 0.95, 20, 0.0))
            try:
                temp = float(req.get("temperature", default_temp))
                top_p = float(req.get("top_p", default_top_p))
                top_k = int(req.get("top_k", default_top_k))
                presence = float(req.get("presence_penalty", default_presence))
                min_p = float(req.get("min_p", 0.01 if self.coding else 0.0))
            except (TypeError, ValueError):
                self.send_json(400, {"error": {"message": "sampling parameters must be numeric", "type": "invalid_request_error"}})
                return
            if (not math.isfinite(temp) or temp < 0 or
                    not math.isfinite(top_p) or not 0 <= top_p <= 1 or
                    top_k < 1 or not math.isfinite(presence) or
                    not math.isfinite(min_p) or not 0 <= min_p <= 1):
                self.send_json(400, {"error": {"message": "invalid sampling parameters", "type": "invalid_request_error"}})
                return
            stop_watcher = threading.Event()
            cancelled = threading.Event()
            stream_keepalive_stop = threading.Event()
            stream_keepalive = None
            stream_write_lock = threading.Lock()
            stream_response_id = "resp-" + uuid.uuid4().hex if req.get("stream") else None
            stream_created = int(time.time())
            if req.get("stream"):
                # Send headers before inference and periodically emit SSE
                # comments.  Qwen3.8's long prompt prefill can otherwise
                # leave Codex's streaming HTTP request silent for minutes.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                stream_base = {"id": stream_response_id, "object": "response",
                               "created_at": stream_created, "status": "in_progress",
                               "model": self.model, "output": []}
                for sequence_number, event in enumerate(("response.created", "response.in_progress")):
                    payload = {"type": event, "response": stream_base,
                               "sequence_number": sequence_number}
                    self.wfile.write(("event: " + event + "\ndata: " +
                                     json.dumps(payload, ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()

                def keepalive():
                    while not stream_keepalive_stop.wait(5.0):
                        try:
                            with stream_write_lock:
                                self.wfile.write(b": keep-alive\n\n")
                                self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            cancelled.set()
                            self.backend.cancel(cancelled)
                            return

                stream_keepalive = threading.Thread(target=keepalive, daemon=True)
                stream_keepalive.start()
            def stream_token(token):
                if not req.get("stream") or api_path != "/v1/chat/completions":
                    return
                obj = {"id": stream_response_id, "object": "chat.completion.chunk",
                       "created": stream_created, "model": self.model,
                       "choices": [{"index": 0, "delta": {"content": token},
                                    "finish_reason": None}]}
                try:
                    with stream_write_lock:
                        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    cancelled.set()
                    self.backend.cancel(cancelled)
            watcher = threading.Thread(target=self._watch_disconnect,
                                       args=(stop_watcher, cancelled), daemon=True)
            watcher.start()
            try:
                text, cached, ptok, ctok, finish = self.backend.generate(
                    prompt, limit, temp, top_p, top_k, presence, min_p, prefix, cancelled,
                    stream_token)
            finally:
                stop_watcher.set()
                watcher.join(timeout=0.2)
                stream_keepalive_stop.set()
                if stream_keepalive is not None:
                    stream_keepalive.join(timeout=0.2)
            if cancelled.is_set() or finish == "cancelled":
                self.log_message("request cancelled: %s", self.path)
                return
            tool_text, calls = parse_calls(text, registry)
            if calls:
                text = tool_text
            ident = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            usage = {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok, "cached_tokens": cached}
            performance = getattr(self.backend, "last_metrics", {})
            if req.get("stream"):
                # The stream headers and keepalive comments were sent before
                # inference; now append the buffered response event sequence.
                # Explicitly terminate the SSE response so clients that use
                # EOF as the stream boundary (including Codex) finish cleanly.
                self.close_connection = True
                if api_path == "/v1/responses":
                    response_id = stream_response_id
                    if calls:
                        response_base = {"id": response_id, "object": "response",
                                         "created_at": created, "status": "in_progress",
                                         "model": self.model, "output": []}
                        events = list(call_events(response_id, calls))[2:]
                        response_done = {**response_base, "status": "completed",
                                         "output": calls,
                                         "usage": {"input_tokens": ptok,
                                                   "output_tokens": ctok,
                                                   "total_tokens": ptok + ctok,
                                                   "input_tokens_details": {"cached_tokens": cached}}}
                        events.append({"type": "response.completed", "response": response_done})
                        for sequence_number, obj in enumerate(events, 2):
                            event = obj["type"]
                            obj = {**obj, "sequence_number": sequence_number}
                            self.wfile.write(("event: " + event + "\ndata: " +
                                              json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                        self.wfile.flush()
                        return
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
                    events = (("response.output_item.added", added_obj), ("response.content_part.added", part_added_obj),
                              ("response.output_text.delta", delta_obj), ("response.output_text.done", text_done_obj),
                              ("response.content_part.done", part_done_obj), ("response.output_item.done", item_done_obj),
                              ("response.completed", done_obj))
                    for sequence_number, (event, obj) in enumerate(events, 2):
                        # Responses stream consumers use this to order and
                        # validate events.  In particular, Codex silently
                        # discards otherwise well-formed events without it.
                        obj = {**obj, "sequence_number": sequence_number}
                        self.wfile.write(("event: " + event + "\ndata: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                else:
                    if calls:
                        delta = {"role": "assistant", "tool_calls": [
                            {"index": i, "id": item["call_id"], "type": item["type"],
                             "function": {"name": item["name"],
                                          "arguments": item.get("arguments", "")}}
                            for i, item in enumerate(calls)]}
                        obj = {"id": ident, "object": "chat.completion.chunk",
                               "created": created, "model": self.model,
                               "choices": [{"index": 0, "delta": delta,
                                             "finish_reason": None}]}
                        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) +
                                          "\n\n").encode())
                    elif text and api_path != "/v1/chat/completions":
                        obj = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                    obj = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "performance": performance}
                    self.wfile.write(("data: " + json.dumps(obj) + "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
                return
            if api_path == "/v1/responses":
                response_id = "resp-" + uuid.uuid4().hex
                if calls:
                    self.send_json(200, {"id": response_id, "object": "response",
                                         "created_at": created, "model": self.model,
                                         "output": calls, "status": "completed",
                                         "usage": {"input_tokens": ptok,
                                                   "output_tokens": ctok,
                                                   "total_tokens": ptok + ctok,
                                                   "input_tokens_details": {"cached_tokens": cached}}})
                    return
                item = {"type": "message", "id": response_id + "-item", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}
                self.send_json(200, {"id": response_id, "object": "response", "created_at": created,
                                     "model": self.model, "output": [item], "output_text": text,
                                     "status": "completed", "usage": {"input_tokens": ptok,
                                     "output_tokens": ctok, "total_tokens": ptok + ctok,
                                     "input_tokens_details": {"cached_tokens": cached}}})
            elif api_path == "/v1/completions":
                self.send_json(200, {"id": ident, "object": "text_completion", "created": created, "model": self.model, "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage})
            else:
                message = {"role": "assistant", "content": text}
                if calls:
                    message["content"] = None
                    message["tool_calls"] = [{"id": item["call_id"], "type": item["type"],
                                               "function": {"name": item["name"],
                                                            "arguments": item.get("arguments", "")}}
                                              for item in calls]
                self.send_json(200, {"id": ident, "object": "chat.completion", "created": created, "model": self.model, "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else finish}], "usage": usage, "performance": performance})
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
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except Exception:
        # The runner is already resident at this point.  Do not leak a model
        # process when the requested port is busy or the bind is invalid.
        Handler.backend.close()
        raise
    signal.signal(signal.SIGTERM, _handle_sigterm)
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        Handler.backend.close()


if __name__ == "__main__":
    main()
