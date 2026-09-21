"""Opt-in GPU quality gate for the resident DFlash2 HTTP window path.

Run with ``--model TARGET --sidecar DFLASH``.  The test is intentionally
separate from the CPU-only protocol suite because it loads both GGUF files.
"""
import argparse
import concurrent.futures
import http.client
import json
import signal
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from codex_server import Backend, Handler, chat_prefix, chat_prompt


HTTP_TIMEOUT = 180
_active_backend = None


def require(condition, message):
    """Keep this opt-in gate effective even when invoked with python -O."""
    if not condition:
        raise AssertionError(message)


def post(port, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return json.load(response)


def concurrent_quality_cases(port):
    cases = (
        ("Answer 5+5 with just 10.", "10"),
        ("Answer 7+7 with just 14.", "14"),
    )
    bodies = [({"messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 8}, expected)
              for prompt, expected in cases]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: post(port, item[0]), bodies))
    for result, (_, expected) in zip(results, bodies):
        text = result["choices"][0]["message"]["content"]
        require(result.get("usage", {}).get("completion_tokens", 0) > 0, result)
        require(expected in text, text)


def cancel_stream(port, prompt="List ten facts about C++."):
    """Close an active stream after its first token and verify cleanup later."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 64, "stream": True,
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=HTTP_TIMEOUT)
    try:
        connection.request("POST", "/v1/chat/completions",
                           body=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        require(response.status == 200, response.status)
        saw_token = False
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                saw_token = True
                break
        require(saw_token, "stream ended before the first generated token")
        # Closing after a real token leaves a DFlash window transaction active
        # and exercises cooperative rollback in the resident stdio child.
    finally:
        connection.close()


def direct_stdio_cases(backend):
    """Exercise the JSONL child independently of HTTP request formatting."""
    messages = [{"role": "user", "content": "What is 8+5? Answer with just 13."}]
    prompt = chat_prompt(messages)
    prefix = chat_prefix(messages)
    greedy = backend.generate(prompt, 8, 0, 0.95, 20, 0, 1, 0,
                              prefix=prefix, seed=42)
    repeated = backend.generate(prompt, 8, 0, 0.95, 20, 0, 1, 0,
                                prefix=prefix, seed=42)
    require(greedy[0] == repeated[0] and "13" in greedy[0], greedy)
    require(greedy[3] > 0 and repeated[1] > 0, repeated)

    sampled = backend.generate(prompt, 8, 0.7, 0.95, 20, 0, 1, 0,
                               prefix=prefix, seed=42)
    sampled_repeat = backend.generate(prompt, 8, 0.7, 0.95, 20, 0, 1, 0,
                                      prefix=prefix, seed=42)
    require(sampled[0] == sampled_repeat[0] and sampled[3] > 0, sampled)
    require(sampled_repeat[1] > 0, sampled_repeat)

    cancellation = threading.Event()

    def cancel_after_token(piece):
        if piece:
            backend.cancel(cancellation)

    cancelled = backend.generate(
        chat_prompt([{"role": "user", "content": "List twenty C++ language features."}]),
        64, 0, 0.95, 20, 0, 1, 0, cancellation=cancellation,
        on_token=cancel_after_token, seed=42)
    require(cancellation.is_set() and cancelled[-1] == "cancelled", cancelled)
    recovered = backend.generate(prompt, 8, 0, 0.95, 20, 0, 1, 0,
                                 prefix=prefix, seed=42)
    require(recovered[0] == greedy[0], recovered)

    cases = (
        ("Answer 6+6 with just 12.", "12"),
        ("Answer 9+9 with just 18.", "18"),
    )

    def run_case(item):
        text, expected = item
        result = backend.generate(
            chat_prompt([{"role": "user", "content": text}]),
            8, 0, 0.95, 20, 0, 1, 0, seed=42)
        return result, expected

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_case, cases))
    for result, expected in results:
        require(result[3] > 0 and expected in result[0], result)
    print("DFlash2 stdio window/cache/sampling/cancel/concurrency: PASS")


def cleanup_on_signal(signum, _frame):
    global _active_backend
    if _active_backend is not None:
        _active_backend.close()
        _active_backend = None
    raise SystemExit(128 + signum)


def main():
    global _active_backend
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--runner", default="./rdna4/llm/test_hip_llm")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--snapshot-max-tokens", type=int, default=0)
    parser.add_argument("--long-prompt-tokens", type=int, default=0,
                        help="also exercise a deterministic longer cached prompt")
    args = parser.parse_args()
    options = SimpleNamespace(
        runner=args.runner, model=args.model, context=args.context,
        moe_cache_mb=0, coding=False, qwen4_coding_profile=False,
        qwen4_exact=False, qwen4_mtp=None, qwen35_mtp=None,
        qwen35_server_profile=False, qwen35_dflash2=args.sidecar,
        qwen35_dflash2_draft=7,
        qwen35_snapshot_max_tokens=args.snapshot_max_tokens,
    )
    backend = Backend(options)
    _active_backend = backend
    direct_stdio_cases(backend)

    class TestHandler(Handler):
        pass

    TestHandler.backend = backend
    TestHandler.model = backend.model
    TestHandler.max_tokens = 64
    TestHandler.context = args.context
    TestHandler.coding = False
    server = ThreadingHTTPServer(("127.0.0.1", args.port), TestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
            health = json.load(response)
        require(health.get("status") == "ready", health)

        cases = (
            ("Write one short sentence about C++.", "C++"),
            ("What is 2+2? Answer with just the number 4.", "4"),
        )
        for prompt_text, expected in cases:
            prompt = [{"role": "user", "content": prompt_text}]
            greedy = {"messages": prompt, "temperature": 0, "max_tokens": 8}
            first = post(args.port, greedy)
            second = post(args.port, greedy)
            third = post(args.port, greedy)
            first_text = first["choices"][0]["message"]["content"]
            require(first.get("usage", {}).get("completion_tokens", 0) > 0, first)
            require(first_text == second["choices"][0]["message"]["content"],
                    "greedy request was not repeatable")
            require(first_text == third["choices"][0]["message"]["content"],
                    "third greedy request was not repeatable")
            require(second.get("usage", {}).get("cached_tokens", 0) > 0,
                    "repeated prompt did not report cache reuse")
            require(third.get("usage", {}).get("cached_tokens", 0) > 0,
                    "third prompt did not report cache reuse")
            require(expected in first_text, first_text)

        long_text = ""
        if args.long_prompt_tokens:
            require(args.long_prompt_tokens > 0, args.long_prompt_tokens)
            require(args.long_prompt_tokens + 32 < args.context,
                    "long prompt must fit the selected context")
            unit = ("C++ uses deterministic compilation, explicit ownership, and "
                    "well-defined arithmetic. ")
            long_text = (unit * ((args.long_prompt_tokens * 4) // len(unit) + 1))[:
                args.long_prompt_tokens * 4]
            long_body = {"messages": [{"role": "user", "content": long_text}],
                         "temperature": 0, "max_tokens": 8}
            long_a = post(args.port, long_body)
            long_b = post(args.port, long_body)
            require(long_a["choices"][0]["message"]["content"] ==
                    long_b["choices"][0]["message"]["content"],
                    "long cached prompt was not repeatable")
            require(long_b.get("usage", {}).get("cached_tokens", 0) > 0,
                    "long prompt did not report cache reuse")

            long_sampled = dict(long_body, temperature=0.7, top_p=0.95,
                                top_k=20, seed=42)
            sampled_a = post(args.port, long_sampled)
            sampled_b = post(args.port, long_sampled)
            require(sampled_a["choices"][0]["message"]["content"] ==
                    sampled_b["choices"][0]["message"]["content"],
                    "long seeded-sampled prompt was not repeatable")
            require(sampled_a.get("usage", {}).get("completion_tokens", 0) > 0,
                    sampled_a)
            require(sampled_b.get("usage", {}).get("cached_tokens", 0) > 0,
                    "long sampled prompt did not reuse the cached transaction")

        cancel_stream(args.port, long_text or "List ten facts about C++.")
        time.sleep(1)
        recovery = post(args.port, {
            "messages": [{"role": "user", "content": "Answer 3+3 with 6."}],
            "temperature": 0, "max_tokens": 8,
        })
        require(recovery.get("usage", {}).get("completion_tokens", 0) > 0,
                recovery)
        require("6" in recovery["choices"][0]["message"]["content"], recovery)

        concurrent_quality_cases(args.port)

        prompt = [{"role": "user", "content": cases[0][0]}]
        sampled = {
            "messages": prompt, "temperature": 0.7, "top_p": 0.95,
            "top_k": 20, "seed": 42, "max_tokens": 8,
        }
        sampled_a = post(args.port, sampled)
        sampled_b = post(args.port, sampled)
        sampled_text = sampled_a["choices"][0]["message"]["content"]
        require(sampled_a.get("usage", {}).get("completion_tokens", 0) > 0,
                sampled_a)
        require(sampled_text == sampled_b["choices"][0]["message"]["content"],
                "seeded sampled request was not repeatable")
        require("C++" in sampled_text, sampled_text)
        print("DFlash2 HTTP greedy/sampled repeatability and quality: PASS")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        backend.close()
        if _active_backend is backend:
            _active_backend = None


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, cleanup_on_signal)
    signal.signal(signal.SIGINT, cleanup_on_signal)
    main()
