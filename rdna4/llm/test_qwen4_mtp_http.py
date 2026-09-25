"""Opt-in real-GPU HTTP regression; uses an ephemeral loopback port.

Run with --model TARGET --sidecar NEXTN (never part of GPU-free unit tests).
"""
import argparse
import json
import signal
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from codex_server import Backend, Handler


_active_backend = None


def _cleanup_on_signal(signum, _frame):
    """Release the HIP child when an external timeout stops this test."""
    global _active_backend
    if _active_backend is not None:
        _active_backend.close()
        _active_backend = None
    raise SystemExit(128 + signum)


def exercise(args, mode):
    global _active_backend
    options = SimpleNamespace(
        runner=args.runner, model=args.model, context=512,
        moe_cache_mb=args.cache_mb, coding=False, qwen4_coding_profile=False,
        qwen4_exact=True, qwen4_mtp=args.sidecar if mode != "baseline" else None,
        qwen4_mtp_draft=4, qwen4_mtp_cache_mb=128, qwen4_mtp_verify=mode,
    )
    backend = Backend(options)
    _active_backend = backend
    class TestHandler(Handler):
        pass
    TestHandler.backend = backend
    TestHandler.model = backend.model
    TestHandler.max_tokens = 64
    TestHandler.context = 512
    TestHandler.coding = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    request = {"messages": [{"role": "user", "content": "Write a C function returning the larger of two integers."}],
               "temperature": 0, "max_tokens": 16}

    def post(data):
        req = urllib.request.Request(url, json.dumps(data).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    try:
        first = post(request)
        text = first["choices"][0]["message"]["content"]
        assert first["usage"]["completion_tokens"] == 16, first
        repeated = post(request)
        assert repeated["choices"][0]["message"]["content"] == text
        assert repeated["usage"]["cached_tokens"] > 0, repeated
        # A zero-output request leaves a fully cached prompt: the subsequent
        # request must read its existing logits, not return an empty answer.
        post(dict(request, max_tokens=0))
        assert post(request)["choices"][0]["message"]["content"] == text
        assert post(dict(request, max_tokens=1))["usage"]["completion_tokens"] == 1
        assert post(dict(request, temperature=0.7, max_tokens=3))["usage"]["completion_tokens"] == 3
        req = urllib.request.Request(url, json.dumps(dict(request, stream=True)).encode(),
                                     {"Content-Type": "application/json"})
        pieces = []
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("choices"):
                    pieces.append(chunk["choices"][0].get("delta", {}).get("content", ""))
        assert "".join(pieces) == text
        # Cancel after a verified streamed token, drain the transaction, then
        # require the same deterministic answer after the runner reset.
        event = threading.Event()
        def cancel_after_token(_piece):
            backend.cancel(event)
        cancelled = backend.generate("Count from one to one hundred.", 64, 0, 1, 20, 0, 1, 0,
                                     cancellation=event, on_token=cancel_after_token)
        assert event.is_set() and cancelled[-1] == "cancelled", cancelled
        assert post(request)["choices"][0]["message"]["content"] == text
        print(f"HTTP {mode}: repeat/cache/output-limit/sampling/SSE/cancel PASS", flush=True)
        return text
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        backend.close()
        if _active_backend is backend:
            _active_backend = None


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _cleanup_on_signal)
    signal.signal(signal.SIGINT, _cleanup_on_signal)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--runner", default="./rdna4/llm/test_hip_llm")
    parser.add_argument("--cache-mb", type=int, default=2048)
    args = parser.parse_args()
    baseline = exercise(args, "baseline")
    for mode in ("scalar", "window"):
        assert exercise(args, mode) == baseline, mode
    print("HTTP baseline/scalar/window greedy parity PASS")
