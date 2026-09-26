#!/usr/bin/env python3
"""OpenAI-compatible HTTP shim for the persistent HIP runner.

The GPU process speaks a deliberately tiny base64 line protocol.  This layer
owns JSON, HTTP, OpenAI response shapes, message canonicalization, and request
limits, while the child keeps the model and KV/SSM state resident.
"""
import argparse
import base64
import hashlib
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

from qwen_chat import (RawTurnCache, StreamSplitter, chat_input_messages, content_text,
                       encode_raw,
                       generation_suffix, prefix_boundaries, render_messages,
                       responses_input_messages, split_generation, system_frame)
from qwen_tools import call_events, parse_calls, tool_registry
import anthropic_api
from live_stream import MessagesLive, ResponsesLive, new_ref
from prefix_store import PrefixStore


WEB_DIR = Path(__file__).with_name("web")
MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_RUNNER_LINE_BYTES = 4 * 1024 * 1024


class FairRequestGate:
    """FIFO admission for the single mutable GPU execution context."""

    def __init__(self):
        self.condition = threading.Condition()
        self.next_ticket = 0
        self.serving = 0
        self.abandoned = set()
        self.closed = False

    def _advance(self):
        while self.serving in self.abandoned:
            self.abandoned.remove(self.serving)
            self.serving += 1

    def acquire(self, cancellation):
        with self.condition:
            if self.closed:
                raise RuntimeError("backend is closed")
            ticket = self.next_ticket
            self.next_ticket += 1
            while ticket != self.serving:
                if cancellation.is_set() or self.closed:
                    self.abandoned.add(ticket)
                    self._advance()
                    self.condition.notify_all()
                    return None
                self.condition.wait(0.05)
            if cancellation.is_set() or self.closed:
                self.serving += 1
                self._advance()
                self.condition.notify_all()
                return None
            return ticket

    def release(self, ticket):
        if ticket is None:
            return
        with self.condition:
            if ticket != self.serving:
                raise RuntimeError("request gate released out of order")
            self.serving += 1
            self._advance()
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def queued(self):
        with self.condition:
            return max(0, self.next_ticket - self.serving - 1 - len(self.abandoned))


def runner_command(args):
    """Build the resident runner command without changing benchmark defaults."""
    server_profile = bool(getattr(args, "qwen35_server_profile", False))
    dflash = getattr(args, "qwen35_dflash2", None)
    qwen35_mtp = getattr(args, "qwen35_mtp", None)
    if server_profile and dflash:
        raise ValueError("--qwen35-server-profile and --qwen35-dflash2 are mutually exclusive")
    qwen4_mtp = getattr(args, "qwen4_mtp", None)
    if dflash and qwen4_mtp:
        raise ValueError("--qwen35-dflash2 cannot be combined with --qwen4-mtp")
    if dflash and qwen35_mtp:
        raise ValueError("--qwen35-dflash2 cannot be combined with --qwen35-mtp")
    if qwen35_mtp and qwen4_mtp:
        raise ValueError("--qwen35-mtp cannot be combined with --qwen4-mtp")
    cmd = [args.runner, args.model, "--stdio-server", "--gpu-only-bench", "-s", str(args.context)]
    cmd += ["--context-cache-entries",
            str(getattr(args, "context_cache_entries", 4)),
            "--context-cache-max-mib",
            str(getattr(args, "context_cache_max_mib", 2048))]
    snapshot_limit = getattr(args, "qwen35_snapshot_max_tokens", 0)
    if snapshot_limit:
        cmd += ["--qwen35-snapshot-max-tokens", str(snapshot_limit)]
    if getattr(args, "moe_cache_mb", 0):
        cmd += ["--moe-cache-mb", str(args.moe_cache_mb)]
    trust_mtp = getattr(args, "qwen4_mtp_trust_draft", False)
    effective_coding = getattr(args, "coding", False) and not trust_mtp
    if effective_coding:
        cmd += ["--coding"]
    if getattr(args, "qwen4_coding_profile", False) and not trust_mtp:
        cmd += ["--qwen4-coding-profile"]
    if getattr(args, "qwen4_exact", False):
        cmd += ["--qwen4-exact"]
    if qwen4_mtp:
        cmd += ["--qwen4-mtp", qwen4_mtp,
                "--qwen4-mtp-draft", str(getattr(args, "qwen4_mtp_draft", 1)),
                "--qwen4-mtp-cache-mb", str(getattr(args, "qwen4_mtp_cache_mb", 128)),
                "--qwen4-mtp-verify", getattr(args, "qwen4_mtp_verify", "scalar")]
    if qwen35_mtp:
        qwen35_mtp_draft = int(getattr(args, "qwen35_mtp_draft", 3))
        if not 1 <= qwen35_mtp_draft <= 15:
            raise ValueError("--qwen35-mtp-draft must be 1..15 for resident serving")
        if not server_profile:
            cmd += ["--kv-cache", "q8q8", "--qwen35-prefill-bf16",
                    "--qwen35-decode-graph", "--qwen35-native-q8-prefill",
                    "--qwen35-native-mmvq"]
        cmd += ["--qwen35-mtp", qwen35_mtp,
                "--qwen35-mtp-draft", str(qwen35_mtp_draft),
                "--qwen35-mtp-window"]
    if server_profile:
        # Keep the resident Qwen3.8 HTTP/stdio route on the validated exact
        # Q8/Q8 profile. This only changes server command construction; the
        # benchmark binary and its defaults remain untouched.
        cmd += ["--kv-cache", "q8q8", "--qwen35-prefill-bf16",
                "--qwen35-decode-graph", "--qwen35-native-q8-prefill",
                "--qwen35-native-mmvq", "--sampling-profile", "llama"]
    if dflash:
        cmd += ["--kv-cache", "q8q8", "--qwen35-prefill-bf16",
                "--qwen35-decode-graph", "--qwen35-native-q8-prefill",
                "--qwen35-native-mmvq",
                "--qwen35-dflash2", dflash,
                "--qwen35-dflash2-draft", str(getattr(args, "qwen35_dflash2_draft", 7))]
    return cmd


def _handle_sigterm(signum, frame):
    """Turn service-manager termination into the normal cleanup path."""
    del signum, frame
    raise KeyboardInterrupt


def chat_prompt(messages, registry=None, thinking=False, effort=None):
    """Render messages with the checkpoint's chat template (see qwen_chat)."""
    return render_messages(messages, registry, thinking, effort)


def chat_prefix(messages, registry=None, thinking=False, effort=None):
    """Return the stable leading system frame for prefix KV reuse."""
    return system_frame(messages, registry, thinking, effort)


def fit_context(messages, context_tokens, output_tokens, render=None):
    """Keep system/developer instructions and the newest complete user turns.

    The tokenizer lives in the GPU child, so this uses a conservative 4-byte
    estimate and lets the child report the exact usage.  It prevents an agent
    from silently pushing the system prompt out of the context window. User,
    assistant, and tool messages are trimmed as turn groups so a tool result is
    not retained without the request and call that produced it.
    """
    render = render or chat_prompt
    budget = max(128, context_tokens - output_tokens) * 4
    if len(render(messages)) <= budget:
        return messages
    indexed = list(enumerate(messages))
    pinned = [(i, message) for i, message in indexed
              if message.get("role") in ("system", "developer")]
    recent = [(i, message) for i, message in indexed
              if message.get("role") not in ("system", "developer")]
    groups = []
    for item in recent:
        if item[1].get("role") == "user" and groups and groups[-1]:
            groups.append([])
        elif not groups:
            groups.append([])
        groups[-1].append(item)
    kept = list(pinned)
    for group in reversed(groups):
        trial_items = sorted(kept + group, key=lambda item: item[0])
        trial = [message for _, message in trial_items]
        if len(render(trial)) > budget and len(kept) > len(pinned):
            break
        kept.extend(group)
    return [message for _, message in sorted(kept, key=lambda item: item[0])]


class Backend:
    def __init__(self, args):
        trust_mtp = getattr(args, "qwen4_mtp_trust_draft", False)
        cmd = runner_command(args)
        # The child intentionally disables MTP in coding mode.  Trusted MTP
        # is an explicit greedy/approximate request, so let it select the
        # sidecar path even if the surrounding server profile is coding.
        effective_coding = args.coding and not trust_mtp
        runner_env = os.environ.copy()
        # Keep direct codex_server launches consistent with the ROCm launcher:
        # explicit approximate decode uses the six-token exact-refresh cadence
        # that passed the 256K coding control.  An explicit caller override
        # remains authoritative.
        if runner_env.get("LLM_QWEN4_APPROX_DECODE", "0") != "0":
            runner_env.setdefault("LLM_QWEN4_DEVICE_REFRESH_INTERVAL", "6")
        if args.qwen4_exact and args.qwen4_mtp:
            # Exact Qwen3.8 MTP uses staged cold misses by default. Direct BAR
            # mapping remains an explicit diagnostic override; Q6_K/Q8_0
            # mapped kernels are not parity-safe yet.
            runner_env.setdefault("LLM_MOE_REGISTER_HOST", "1")
            runner_env.setdefault("LLM_QWEN4_MAPPED_MISSES", "1")
            runner_env.setdefault("LLM_QWEN4_DIRECT_MISSES", "0")
            runner_env.setdefault("LLM_MOE_CPU_DECODE_MISSES", "0")
            runner_env.setdefault("LLM_BMAX", "1")
            if not args.moe_cache_mb:
                cmd += ["--moe-cache-mb", "9728"]
        if getattr(args, "qwen4_mtp_adaptive", False):
            runner_env["LLM_QWEN4_MTP_ADAPTIVE"] = "1"
        self.coding = effective_coding
        if trust_mtp:
            if not args.qwen4_mtp or args.qwen4_exact:
                raise ValueError("--qwen4-mtp-trust-draft requires approximate --qwen4-mtp")
            runner_env["LLM_QWEN4_MTP_APPROX"] = "1"
            runner_env["LLM_QWEN4_MTP_TRUST_DRAFT"] = "1"
        # Put the runner in its own process group.  A timeout or service
        # manager may terminate this Python parent without running `finally`;
        # group teardown then prevents a model-sized HIP child from retaining
        # VRAM and poisoning the next benchmark.
        popen_kwargs = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=None, text=True, bufsize=1, env=runner_env)
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        self.proc = subprocess.Popen(cmd, **popen_kwargs)
        self.lock = threading.Lock()
        self.request_gate = FairRequestGate()
        self.cancel_lock = threading.Lock()
        self.active_cancel = None
        self.active_request_id = None
        self.request_cancellations = {}
        self.metrics_local = threading.local()
        self.ready = False
        self.model = (getattr(args, "served_model_name", None) or
                      args.model.rsplit("/", 1)[-1])
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
        gate = getattr(self, "request_gate", None)
        return {
            "status": "ready" if self.ready and exit_status is None else "unavailable",
            "runner_alive": exit_status is None,
            "runner_exit_status": exit_status,
            "active_request": getattr(self, "active_request_id", None),
            "queued_requests": gate.queued() if gate is not None else 0,
        }

    def close(self):
        """Stop and reap the resident runner during server shutdown."""
        gate = getattr(self, "request_gate", None)
        if gate is not None:
            gate.close()
        if self.proc.poll() is None:
            if os.name == "posix" and hasattr(self.proc, "pid"):
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix" and hasattr(self.proc, "pid"):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
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
                return False
            if cancellation is not None and cancellation is not self.active_cancel:
                return False
            self.active_cancel.set()
            if self.proc.poll() is None:
                try:
                    os.kill(self.proc.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass
            return True

    def register_request(self, request_id, cancellation):
        """Reserve a request ID before HTTP streaming headers are committed."""
        with self.cancel_lock:
            if request_id in self.request_cancellations:
                return False
            self.request_cancellations[request_id] = cancellation
            return True

    def unregister_request(self, request_id, cancellation):
        """Release a request ID only when the caller still owns it."""
        with self.cancel_lock:
            if self.request_cancellations.get(request_id) is cancellation:
                self.request_cancellations.pop(request_id, None)

    def cancel_request(self, request_id):
        """Cancel one queued or active request without touching its peers."""
        with self.cancel_lock:
            cancellation = getattr(self, "request_cancellations", {}).get(request_id)
            if cancellation is None:
                return False
            cancellation.set()
            active = cancellation is self.active_cancel
            if active and self.proc.poll() is None:
                try:
                    os.kill(self.proc.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass
            return True

    @property
    def last_metrics(self):
        local = getattr(self, "metrics_local", None)
        return getattr(local, "value", {}) if local is not None else {}

    @last_metrics.setter
    def last_metrics(self, value):
        if not hasattr(self, "metrics_local"):
            self.metrics_local = threading.local()
        self.metrics_local.value = value

    def generate(self, prompt, max_tokens, temperature, top_p, top_k, presence, repetition, min_p,
                 prefix="", cancellation=None, on_token=None, seed=None,
                 frequency=0.0, penalty_last_n=64, cache_key="shared",
                 request_id=None, request_registered=False):
        request_start = time.monotonic()
        cancellation = cancellation if cancellation is not None else threading.Event()
        # One stable prefix, or several (each gets a shared runner snapshot).
        prefixes = [p for p in ([prefix] if isinstance(prefix, str) else prefix or []) if p]
        prefix_payload = ",".join(base64.b64encode(p.encode("utf-8")).decode("ascii")
                                  for p in prefixes) or "-"
        payload = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        cache_identity = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()
        seed_field = "-" if seed is None else str(seed)
        line = (f"REQ3 {cache_identity} {seed_field} {max_tokens} {temperature} "
                f"{top_p} {top_k} {presence} {repetition} {min_p} {frequency} "
                f"{penalty_last_n} {prefix_payload} {payload}\n")
        if len(line) >= MAX_RUNNER_LINE_BYTES:
            raise ValueError("encoded prompt exceeds the runner protocol limit")
        if not hasattr(self, "request_gate"):
            self.request_gate = FairRequestGate()
        if not hasattr(self, "request_cancellations"):
            self.request_cancellations = {}
        request_id = request_id or "req-" + uuid.uuid4().hex
        if request_registered:
            with self.cancel_lock:
                if self.request_cancellations.get(request_id) is not cancellation:
                    raise RuntimeError("request_id reservation was lost")
        elif not self.register_request(request_id, cancellation):
            raise ValueError("duplicate active request_id")
        ticket = None
        try:
            ticket = self.request_gate.acquire(cancellation)
            if ticket is None:
                return "", 0, 0, 0, "cancelled"
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
                    self.active_request_id = request_id
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
                        self.active_request_id = None
        finally:
            self.request_gate.release(ticket)
            if not request_registered:
                self.unregister_request(request_id, cancellation)
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
        # Measure the complete runner request, including protocol and
        # synchronization overhead.  This is the number users experience for
        # a streamed 256K request; pp/tg are useful component rates only.
        e2e_ms = (time.monotonic() - request_start) * 1000.0
        self.last_metrics["e2e_ms"] = e2e_ms
        self.last_metrics["e2e_tok_s"] = (
            1000.0 * (int(prompt_tokens) - int(cached) + int(completion_tokens)) / e2e_ms
            if e2e_ms > 0.0 else 0.0)
        text = base64.b64decode(encoded).decode("utf-8", "replace")
        return text, int(cached), int(prompt_tokens), int(completion_tokens), finish


def responses_output(response_id, reasoning, raw_turn, text, calls):
    """Responses output items for one assistant turn.

    The leading reasoning item carries the raw generated turn as opaque
    encrypted_content.  Clients that keep reasoning items (Codex requests
    reasoning.encrypted_content with store=false) send it back, so the next
    prompt reproduces this turn byte for byte and the runner extends its live
    KV/recurrent state instead of re-prefilling or restoring a snapshot.
    """
    items = [{"type": "reasoning", "id": "rs_" + uuid.uuid4().hex,
              "summary": ([{"type": "summary_text", "text": reasoning}]
                          if reasoning else []),
              "encrypted_content": encode_raw(raw_turn)}]
    if text or not calls:
        items.append({"type": "message", "id": response_id + "-item",
                      "role": "assistant", "status": "completed",
                      "content": [{"type": "output_text", "text": text,
                                   "annotations": []}]})
    items.extend(calls)
    return items


class Handler(BaseHTTPRequestHandler):
    backend = None
    model = "local"
    max_tokens = 256
    context = 4096
    coding = False
    thinking = "auto"
    raw_turns = RawTurnCache()
    prefix_store = None

    def count_tokens(self):
        """Estimate /v1/messages/count_tokens from the rendered prompt."""
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_REQUEST_BYTES:
                raise ValueError("invalid request body size")
            req = json.loads(self.rfile.read(n))
            if not isinstance(req, dict):
                raise ValueError("request body must be a JSON object")
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.send_json(400, {"type": "error", "error": {
                "type": "invalid_request_error", "message": str(exc)}})
            return
        thinking, effort = self.thinking_mode(req, "/v1/messages")
        registry = tool_registry(anthropic_api.tool_definitions(req.get("tools")))
        prompt = chat_prompt(anthropic_api.request_messages(req), registry, thinking, effort)
        # About 3.5 bytes per token for this tokenizer on code and English.
        self.send_json(200, {"input_tokens": max(1, round(len(prompt.encode("utf-8")) / 3.5))})

    def thinking_mode(self, req, api_path):
        """Return (thinking, effort) for a request.

        auto: Responses requests think when they ask for a reasoning effort
        (Codex always sends one); Chat Completions think only when asked via
        reasoning_effort or chat_template_kwargs.enable_thinking.
        """
        if api_path == "/v1/messages":
            requested, effort = anthropic_api.thinking_request(req)
            if self.thinking == "off":
                return False, effort
            if self.thinking == "on":
                return True, effort or "medium"
            return requested, effort
        effort = None
        reasoning = req.get("reasoning")
        if isinstance(reasoning, dict) and isinstance(reasoning.get("effort"), str):
            effort = reasoning["effort"]
        elif isinstance(req.get("reasoning_effort"), str):
            effort = req["reasoning_effort"]
        kwargs = req.get("chat_template_kwargs")
        explicit = kwargs.get("enable_thinking") if isinstance(kwargs, dict) else None
        if self.thinking == "off":
            return False, effort
        if self.thinking == "on":
            return explicit is not False and effort not in ("none",), effort
        if isinstance(explicit, bool):
            return explicit, effort
        return effort is not None and effort not in ("none", "minimal"), effort

    def log_message(self, fmt, *args):
        sys.stderr.write("[api] " + (fmt % args) + "\n")

    def send_json(self, status, obj):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            request_id = getattr(self, "_request_id", None)
            if request_id:
                self.send_header("X-Request-ID", request_id)
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

    def do_HEAD(self):
        # Connectivity probes (Claude Code sends HEAD /api/hello).
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self._request_id = None
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
            # llama.cpp router shape (status/meta) so router clients such as
            # pi's llama.cpp extension accept this single resident model.
            self.send_json(200, {"object": "list", "data": [{
                "id": self.model, "object": "model", "created": now, "owned_by": "local",
                "status": {"value": "loaded"}, "source": "resident",
                "meta": {"n_ctx": self.context, "n_ctx_train": self.context},
                "architecture": {"input_modalities": ["text"]}}]})
        elif path in ("/props", "/v1/props"):
            template = Path(__file__).with_name("qwen38_chat_template.jinja")
            self.send_json(200, {
                "models_autoload": False,
                "chat_template": template.read_text() if template.exists() else "",
                "default_generation_settings": {"n_ctx": self.context},
                "model_path": self.model})
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
        self._request_id = None
        response_started = False
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path.startswith("/v1/"):
            api_path = path
        elif path in ("/chat/completions", "/completions", "/responses", "/messages",
                      "/messages/count_tokens"):
            api_path = "/v1" + path
        else:
            api_path = path
        if api_path == "/v1/cancel":
            request_id = None
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n < 0 or n > 4096:
                    raise ValueError("cancellation body is too large")
                if n > 0:
                    body = json.loads(self.rfile.read(n))
                    if not isinstance(body, dict):
                        raise ValueError("cancellation body must be a JSON object")
                    request_id = body.get("request_id")
                    if request_id is not None and (
                            not isinstance(request_id, str) or not request_id or
                            len(request_id) > 128):
                        raise ValueError(
                            "request_id must be a non-empty string of at most 128 characters")
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.send_json(400, {"error": {"message": str(exc),
                                                "type": "invalid_request_error"}})
                return
            if request_id is None:
                found = self.backend.cancel()
            else:
                self._request_id = request_id
                found = self.backend.cancel_request(request_id)
            self.send_json(202 if found else 404, {
                "status": "cancellation_requested" if found else "request_not_found",
                "request_id": request_id,
            })
            return
        if api_path == "/v1/messages/count_tokens":
            self.count_tokens()
            return
        if path in ("/models/load", "/models/unload"):
            # One resident model: loading is a no-op and it stays loaded.
            n = int(self.headers.get("Content-Length", "0") or 0)
            if 0 < n <= 4096:
                self.rfile.read(n)
            self.send_json(200, {"success": True})
            return
        if api_path not in ("/v1/chat/completions", "/v1/completions", "/v1/responses",
                            "/v1/messages"):
            self.log_message("404 POST %s", self.path)
            self.send_json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        try:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except (TypeError, ValueError):
                self.send_json(400, {"error": {"message": "invalid Content-Length",
                                                "type": "invalid_request_error"}})
                return
            if n <= 0:
                self.send_json(400, {"error": {"message": "request body is required", "type": "invalid_request_error"}})
                return
            if n > MAX_REQUEST_BYTES:
                self.close_connection = True
                self.send_json(413, {"error": {"message": "request body is too large",
                                                "type": "invalid_request_error"}})
                return
            try:
                req = json.loads(self.rfile.read(n))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.send_json(400, {"error": {"message": f"invalid JSON: {exc}", "type": "invalid_request_error"}})
                return
            if not isinstance(req, dict):
                self.send_json(400, {"error": {"message": "request body must be a JSON object", "type": "invalid_request_error"}})
                return
            if "request_id" in req:
                request_id = req["request_id"]
            elif self.headers.get("X-Request-ID") is not None:
                request_id = self.headers.get("X-Request-ID")
            else:
                request_id = "req-" + uuid.uuid4().hex
            if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
                self.send_json(400, {"error": {"message": "request_id must be a non-empty string of at most 128 characters",
                                                "type": "invalid_request_error"}})
                return
            self._request_id = request_id
            metadata = req.get("metadata") if isinstance(req.get("metadata"), dict) else {}
            if "prompt_cache_key" in req:
                cache_key = req["prompt_cache_key"]
            elif "conversation_id" in metadata:
                cache_key = metadata["conversation_id"]
            elif "session_id" in metadata:
                cache_key = metadata["session_id"]
            elif self.headers.get("X-Prompt-Cache-Key") is not None:
                cache_key = self.headers.get("X-Prompt-Cache-Key")
            elif self.headers.get("X-Claude-Code-Session-Id") is not None:
                cache_key = self.headers.get("X-Claude-Code-Session-Id")
            elif isinstance(metadata.get("user_id"), str) and metadata["user_id"]:
                # Claude Code embeds its session id in metadata.user_id.
                cache_key = metadata["user_id"][:512]
            else:
                cache_key = "shared"
            if not isinstance(cache_key, str) or not cache_key or len(cache_key) > 512:
                self.send_json(400, {"error": {"message": "prompt_cache_key must be a non-empty string of at most 512 characters",
                                                "type": "invalid_request_error"}})
                return
            if api_path == "/v1/completions":
                prompt = req.get("prompt", "")
                if isinstance(prompt, list): prompt = "".join(map(str, prompt))
                messages = [{"role": "user", "content": prompt}]
            elif api_path == "/v1/messages":
                messages = anthropic_api.request_messages(req)
            elif api_path == "/v1/responses":
                messages = []
                if req.get("instructions"): messages.append({"role": "system", "content": req["instructions"]})
                inp = req.get("input", "")
                messages.extend(responses_input_messages(inp))
            else:
                messages = req.get("messages", [])
                if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
                    messages = chat_input_messages(messages)
            if (not isinstance(messages, list) or
                    any(not isinstance(message, dict) for message in messages)):
                self.send_json(400, {"error": {
                    "message": "messages must be an array of objects",
                    "type": "invalid_request_error"}})
                return
            registry = tool_registry(
                anthropic_api.tool_definitions(req.get("tools"))
                if api_path == "/v1/messages" else req.get("tools", []))
            thinking, effort = self.thinking_mode(req, api_path)
            try:
                requested_limit = int(req.get("max_tokens", req.get("max_output_tokens", self.max_tokens)))
            except (TypeError, ValueError):
                self.send_json(400, {"error": {"message": "max_tokens must be an integer", "type": "invalid_request_error"}})
                return
            if requested_limit < 0:
                self.send_json(400, {"error": {"message": "max_tokens must be non-negative", "type": "invalid_request_error"}})
                return
            limit = min(requested_limit, self.max_tokens)
            self.raw_turns.resolve(messages)
            messages = fit_context(
                messages, self.context, limit,
                lambda m: chat_prompt(m, registry, thinking, effort))
            prompt = chat_prompt(messages, registry, thinking, effort)
            prefix = prefix_boundaries(messages, registry, thinking, effort)
            trace_dir = os.environ.get("QWEN38_TRACE_DIR")
            if trace_dir:
                # Diagnostic only: raw request plus the exact rendered prompt,
                # for diffing successive agent turns' prefixes.
                stamp = f"{time.time():.6f}"
                with open(os.path.join(trace_dir, f"{stamp}.json"), "w", encoding="utf-8") as f:
                    json.dump({"path": api_path, "cache_key": cache_key, "request": req,
                               "prompt": prompt, "prefix": prefix}, f, ensure_ascii=False)
            # The C child reads one complete request into a fixed 4 MiB line.
            # Check the actual UTF-8/base64 expansion before streaming headers;
            # an oversized partial line would otherwise desynchronize every
            # later request on the resident runner.
            prompt_bytes = len(prompt.encode("utf-8"))
            prefix_bytes = sum(4 * ((len(p.encode("utf-8")) + 2) // 3) + 1 for p in prefix)
            encoded_bytes = (4 * ((prompt_bytes + 2) // 3) + max(prefix_bytes, 1) +
                             2048)
            if encoded_bytes >= MAX_RUNNER_LINE_BYTES:
                self.send_json(400, {"error": {
                    "message": "encoded prompt exceeds the runner protocol limit",
                    "type": "invalid_request_error"}})
                return
            # Explicit API sampling controls override the requested coding
            # profile.  Keep the server default aligned with the local
            # non-thinking coding evaluation profile; it is deliberately
            # narrower than the general chat defaults so approximate routed
            # MoE decode remains both useful and coherent.
            # The child receives these values through the request protocol;
            # its standalone benchmark sampling defaults do not apply here.
            # Qwen's published profiles: thinking (coding) 0.6/0.95/20, and
            # non-thinking 0.7/0.8/20 with presence 1.5.  Agent (Responses)
            # traffic without explicit controls takes them; a near-greedy
            # default makes tool-using agents repeat one failing call.
            if thinking:
                defaults = (0.6, 0.95, 20, 0.0)
            elif self.coding or api_path in ("/v1/responses", "/v1/messages"):
                defaults = (0.7, 0.80, 20, 1.5)
            else:
                defaults = (0.2, 0.95, 20, 0.0)
            default_temp, default_top_p, default_top_k, default_presence = defaults
            if api_path == "/v1/messages" and os.environ.get("QWEN38_HONOR_CLIENT_SAMPLING", "0") == "0":
                # Anthropic clients send temperature 1 (required there with
                # extended thinking); Qwen's profiles are the useful values.
                req = {k: v for k, v in req.items()
                       if k not in ("temperature", "top_p", "top_k")}
            try:
                temp = float(req.get("temperature", default_temp))
                top_p = float(req.get("top_p", default_top_p))
                top_k = int(req.get("top_k", default_top_k))
                presence = float(req.get("presence_penalty", default_presence))
                repetition = float(req.get("repetition_penalty", 1.0))
                min_p = float(req.get("min_p", 0.0))
                seed = int(req["seed"]) if "seed" in req else None
                frequency = float(req.get("frequency_penalty", 0.0))
                penalty_last_n = int(req.get("penalty_last_n", 64))
                if seed is None and (frequency != 0 or penalty_last_n != 64):
                    seed = 42
            except (TypeError, ValueError):
                self.send_json(400, {"error": {"message": "sampling parameters must be numeric", "type": "invalid_request_error"}})
                return
            if (not math.isfinite(temp) or temp < 0 or
                    not math.isfinite(top_p) or not 0 <= top_p <= 1 or
                    (top_k < 1 and seed is None) or not math.isfinite(presence) or
                    not math.isfinite(repetition) or repetition <= 0 or
                    not math.isfinite(min_p) or not 0 <= min_p <= 1 or
                    not math.isfinite(frequency) or penalty_last_n < 0 or
                    (seed is not None and (seed < 0 or seed >= 2**32 - 1 or self.coding))):
                self.send_json(400, {"error": {"message": "invalid sampling parameters", "type": "invalid_request_error"}})
                return
            stop_watcher = threading.Event()
            cancelled = threading.Event()
            stream_keepalive_stop = threading.Event()
            stream_keepalive = None
            stream_write_lock = threading.Lock()
            stream_response_id = "resp-" + uuid.uuid4().hex if req.get("stream") else None
            stream_created = int(time.time())
            if not self.backend.register_request(request_id, cancelled):
                self.send_json(409, {"error": {
                    "message": "request_id is already active",
                    "type": "invalid_request_error"}})
                return
            watcher = None
            live = None
            live_ref = None
            sequence = [2]

            def sse(event, payload):
                with stream_write_lock:
                    self.wfile.write(("event: " + event + "\ndata: " +
                                      json.dumps(payload, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()

            def sse_responses(obj):
                obj = {**obj, "sequence_number": sequence[0]}
                sequence[0] += 1
                sse(obj["type"], obj)

            try:
                if req.get("stream"):
                    # Send headers only after reserving the ID. A duplicate
                    # request must receive a normal 409 response rather than
                    # an HTTP 200 followed by malformed SSE error bytes.
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("X-Request-ID", request_id)
                    self.end_headers()
                    response_started = True
                    stream_base = {"id": stream_response_id, "object": "response",
                                   "created_at": stream_created, "status": "in_progress",
                                   "model": self.model, "output": []}
                    if api_path == "/v1/messages":
                        stream_response_id = "msg_" + uuid.uuid4().hex[:24]
                        event, payload = anthropic_api.stream_start(
                            stream_response_id, self.model, 0, 0)
                        self.wfile.write(("event: " + event + "\ndata: " +
                                          json.dumps(payload) + "\n\n").encode())
                    else:
                        for sequence_number, event in enumerate(("response.created", "response.in_progress")):
                            payload = {"type": event, "response": stream_base,
                                       "sequence_number": sequence_number}
                            self.wfile.write(("event: " + event + "\ndata: " +
                                             json.dumps(payload, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()
                    # Reasoning and answer text stream while the model runs.
                    if api_path == "/v1/messages":
                        live_ref = new_ref()
                        live = MessagesLive(sse, live_ref)
                    elif api_path == "/v1/responses":
                        live_ref = new_ref()
                        live = ResponsesLive(sse_responses, stream_response_id, live_ref)
                    keepalive_bytes = (b'event: ping\ndata: {"type": "ping"}\n\n'
                                       if api_path == "/v1/messages" else b": keep-alive\n\n")

                    def keepalive():
                        while not stream_keepalive_stop.wait(5.0):
                            try:
                                with stream_write_lock:
                                    self.wfile.write(keepalive_bytes)
                                    self.wfile.flush()
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                cancelled.set()
                                self.backend.cancel(cancelled)
                                return

                    stream_keepalive = threading.Thread(target=keepalive, daemon=True)
                    stream_keepalive.start()

                splitter = StreamSplitter(thinking)

                def stream_token(token):
                    if not req.get("stream"):
                        return
                    if live is not None:
                        parts = splitter.feed(token)
                        if parts:
                            try:
                                live.feed(parts)
                            except (BrokenPipeError, ConnectionResetError, OSError):
                                cancelled.set()
                                self.backend.cancel(cancelled)
                        return
                    if api_path != "/v1/chat/completions":
                        return
                    parts = splitter.feed(token)
                    if not parts:
                        return
                    try:
                        with stream_write_lock:
                            for kind, piece in parts:
                                field = "reasoning_content" if kind == "reasoning" else "content"
                                obj = {"id": stream_response_id, "object": "chat.completion.chunk",
                                       "created": stream_created, "model": self.model,
                                       "choices": [{"index": 0, "delta": {field: piece},
                                                    "finish_reason": None}]}
                                self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) +
                                                  "\n\n").encode())
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        cancelled.set()
                        self.backend.cancel(cancelled)

                watcher = threading.Thread(target=self._watch_disconnect,
                                           args=(stop_watcher, cancelled), daemon=True)
                watcher.start()
                text, cached, ptok, ctok, finish = self.backend.generate(
                    prompt, limit, temp, top_p, top_k, presence, repetition, min_p, prefix, cancelled,
                    stream_token, seed=seed, frequency=frequency,
                    penalty_last_n=penalty_last_n, cache_key=cache_key,
                    request_id=request_id, request_registered=True)
            finally:
                stop_watcher.set()
                if watcher is not None:
                    watcher.join(timeout=0.2)
                stream_keepalive_stop.set()
                if stream_keepalive is not None:
                    stream_keepalive.join(timeout=0.2)
                self.backend.unregister_request(request_id, cancelled)
            if cancelled.is_set() or finish == "cancelled":
                self.log_message("request cancelled: %s", self.path)
                self.close_connection = True
                return
            if self.prefix_store is not None:
                self.prefix_store.record(prefix)
            # The exact bytes of this assistant turn after "<|im_start|>assistant\n".
            raw_turn = generation_suffix(thinking) + text
            reasoning_text, answer = split_generation(text, thinking)
            tool_text, calls = parse_calls(answer, registry)
            text = tool_text if calls else answer
            if not calls and "<tool_call>" in answer:
                # The client receives this as plain text; make it visible.
                sys.stderr.write("[tool-call] unparsed: " +
                                 answer[answer.find("<tool_call>"):][:400].replace("\n", "\\n") + "\n")
            if api_path == "/v1/messages":
                content = anthropic_api.response_content(reasoning_text, raw_turn, text, calls)
                self.raw_turns.remember(raw_turn, text, calls,
                                        [b["id"] for b in content if b["type"] == "tool_use"])
                reason = anthropic_api.stop_reason(calls, finish)
                use = anthropic_api.usage(ptok, cached, ctok)
                if req.get("stream"):
                    self.close_connection = True
                    self.raw_turns.remember_ref(live_ref, raw_turn)
                    live.finish(text, [b for b in content if b["type"] == "tool_use"],
                                reason, use)
                else:
                    self.send_json(200, anthropic_api.message_object(
                        "msg_" + uuid.uuid4().hex[:24], self.model, content, reason, use))
                return
            if api_path != "/v1/messages":
                self.raw_turns.remember(raw_turn, text, calls,
                                        [c.get("call_id") for c in calls])
            ident = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            usage = {"prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok,
                     "cached_tokens": cached, "prompt_tokens_details": {"cached_tokens": cached}}
            performance = getattr(self.backend, "last_metrics", {})
            if req.get("stream"):
                # The stream headers and keepalive comments were sent before
                # inference; now append the buffered response event sequence.
                # Explicitly terminate the SSE response so clients that use
                # EOF as the stream boundary (including Codex) finish cleanly.
                self.close_connection = True
                if api_path == "/v1/responses":
                    response_id = stream_response_id
                    self.raw_turns.remember_ref(live_ref, raw_turn)
                    output = live.finish(text, calls)
                    response_done = {"id": response_id, "object": "response",
                                     "created_at": created, "status": "completed",
                                     "model": self.model, "output": output,
                                     "usage": {"input_tokens": ptok, "output_tokens": ctok,
                                               "total_tokens": ptok + ctok,
                                               "input_tokens_details": {"cached_tokens": cached}}}
                    # Codex discards Responses events without sequence_number.
                    sse_responses({"type": "response.completed", "response": response_done})
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
                    elif text.startswith(splitter.content) and len(text) > len(splitter.content):
                        # Held-back answer text that turned out not to be a call.
                        obj = {"id": ident, "object": "chat.completion.chunk", "created": created,
                               "model": self.model, "choices": [{"index": 0, "delta": {
                                   "content": text[len(splitter.content):]}, "finish_reason": None}]}
                        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
                    obj = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": self.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else finish}], "performance": performance}
                    self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                    options = req.get("stream_options")
                    if isinstance(options, dict) and options.get("include_usage"):
                        obj = {"id": ident, "object": "chat.completion.chunk", "created": created,
                               "model": self.model, "choices": [],
                               "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                                         "total_tokens": ptok + ctok,
                                         "prompt_tokens_details": {"cached_tokens": cached}}}
                        self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if api_path == "/v1/responses":
                response_id = "resp-" + uuid.uuid4().hex
                output = responses_output(response_id, reasoning_text, raw_turn, text, calls)
                self.send_json(200, {"id": response_id, "object": "response",
                                     "created_at": created, "model": self.model,
                                     "output": output, "output_text": "" if calls else text,
                                     "status": "completed",
                                     "usage": {"input_tokens": ptok, "output_tokens": ctok,
                                               "total_tokens": ptok + ctok,
                                               "input_tokens_details": {"cached_tokens": cached}}})
            elif api_path == "/v1/completions":
                self.send_json(200, {"id": ident, "object": "text_completion", "created": created, "model": self.model, "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage})
            else:
                message = {"role": "assistant", "content": text}
                if reasoning_text:
                    message["reasoning_content"] = reasoning_text
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
            if response_started:
                # HTTP status and SSE headers are already committed. Appending
                # a JSON error response would corrupt the event stream.
                self.close_connection = True
                return
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
    ap.add_argument("--prefix-store", default="",
                    help="JSON file remembering agents' system prefixes across restarts "
                         "(contains system prompts; written 0600); empty disables")
    ap.add_argument("--prefix-warmup", type=int, default=4,
                    help="stored prefixes to re-prefill at startup (0 disables)")
    ap.add_argument("--served-model-name", default=None,
                    help="model id reported by /v1/models and responses "
                         "(default: the GGUF file name)")
    ap.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                    help="Qwen3.8 reasoning mode: auto follows the request's reasoning "
                         "effort (Responses) or enable_thinking (Chat Completions)")
    ap.add_argument("--qwen4-coding-profile", action="store_true")
    ap.add_argument("--qwen4-mtp", help="NextN sidecar; accelerate greedy requests only")
    ap.add_argument("--qwen4-exact", action="store_true", help="exact Qwen4 routing/QSA baseline")
    ap.add_argument("--qwen4-mtp-draft", type=int, choices=range(1, 33), default=1)
    ap.add_argument("--qwen4-mtp-cache-mb", type=int, default=128)
    ap.add_argument("--qwen4-mtp-verify", choices=("scalar", "window"), default="scalar")
    ap.add_argument("--qwen4-mtp-trust-draft", action="store_true",
                    help="approximate sidecar-only MTP; skips target verification")
    ap.add_argument("--qwen4-mtp-adaptive", action="store_true",
                    help="exact MTP fallback after low draft acceptance")
    ap.add_argument("--qwen35-server-profile", action="store_true",
                    help="use the validated exact Qwen3.8 Q8/Q8 HTTP/stdio profile")
    ap.add_argument("--qwen35-mtp", metavar="SIDECAR",
                    help="exact dense Qwen3.8 NextN sidecar for greedy serving")
    ap.add_argument("--qwen35-mtp-draft", type=int, choices=range(1, 16), default=3)
    ap.add_argument("--qwen35-mtp-window", action="store_true",
                    help="compatibility flag; resident Dense NextN always uses exact windows")
    ap.add_argument("--qwen35-dflash2", metavar="SIDECAR",
                    help="exact Qwen3.8 DFlash2 sidecar for HTTP/stdio serving")
    ap.add_argument("--qwen35-dflash2-draft", type=int, choices=range(1, 8), default=7)
    ap.add_argument("--qwen35-snapshot-max-tokens", type=int, default=0,
                    help="bound host-side Qwen3.8 Q8 prompt snapshots (0=16K default)")
    ap.add_argument("--context-cache-entries", type=int, default=4,
                    help="maximum portable conversation snapshots (0 disables)")
    ap.add_argument("--context-cache-max-mib", type=int, default=2048,
                    help="host-memory budget for portable conversation snapshots")
    args = ap.parse_args()
    if args.context <= 0:
        ap.error("--context must be positive")
    if args.max_output < 0:
        ap.error("--max-output must be non-negative")
    if args.moe_cache_mb < 0:
        ap.error("--moe-cache-mb must be non-negative")
    if args.qwen4_mtp_cache_mb < 0:
        ap.error("--qwen4-mtp-cache-mb must be non-negative")
    if args.qwen35_snapshot_max_tokens < 0:
        ap.error("--qwen35-snapshot-max-tokens must be non-negative")
    if not 0 <= args.context_cache_entries <= 64:
        ap.error("--context-cache-entries must be 0..64")
    if not 0 <= args.context_cache_max_mib <= 65536:
        ap.error("--context-cache-max-mib must be 0..65536")
    Handler.backend = Backend(args)
    Handler.model = Handler.backend.model
    Handler.max_tokens = args.max_output
    Handler.context = args.context
    Handler.coding = args.coding
    Handler.thinking = args.thinking
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except Exception:
        # The runner is already resident at this point.  Do not leak a model
        # process when the requested port is busy or the bind is invalid.
        Handler.backend.close()
        raise
    signal.signal(signal.SIGTERM, _handle_sigterm)
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1", flush=True)
    if args.prefix_store:
        Handler.prefix_store = PrefixStore(args.prefix_store)
        stored = Handler.prefix_store.warmup_list()[:max(0, args.prefix_warmup)]
        if stored:
            def warmup():
                # Queued like ordinary requests: an agent request that
                # arrives meanwhile waits only for the item in progress.
                for boundaries in stored:
                    start = time.monotonic()
                    try:
                        _, cached, tokens, _, _ = Handler.backend.generate(
                            boundaries[-1], 0, 0.0, 0.95, 20, 0.0, 1.0, 0.0,
                            prefix=boundaries, cache_key="prefix-warmup")
                        sys.stderr.write(f"[warmup] prefix {tokens} tokens "
                                         f"({tokens - cached} prefilled) in "
                                         f"{time.monotonic() - start:.1f}s\n")
                    except Exception as exc:  # warm-up is best effort
                        sys.stderr.write(f"[warmup] failed: {exc}\n")
            threading.Thread(target=warmup, daemon=True).start()
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        server.server_close()
        Handler.backend.close()


if __name__ == "__main__":
    main()
