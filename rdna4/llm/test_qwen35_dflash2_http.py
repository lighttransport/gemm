"""Opt-in GPU quality gate for resident Qwen3.8 speculative HTTP paths.

Run with ``--model TARGET`` and either ``--sidecar DFLASH`` or ``--mtp NEXTN``.
The test is separate from the CPU-only protocol suite because it loads both
GGUF files.
"""
import argparse
import concurrent.futures
import http.client
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
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


def post_error(port, body, status, path="/v1/chat/completions"):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=HTTP_TIMEOUT)
    except urllib.error.HTTPError as error:
        require(error.code == status, error.code)
        return json.load(error)
    raise AssertionError(f"request unexpectedly succeeded; expected HTTP {status}")


def concurrent_quality_cases(port):
    cases = (
        ("Answer 5+5 with just 10.", "10"),
        ("Answer 7+7 with just 14.", "14"),
    )
    bodies = [({"messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 8,
                "prompt_cache_key": f"concurrent-{expected}"}, expected)
              for prompt, expected in cases]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: post(port, item[0]), bodies))
    for result, (_, expected) in zip(results, bodies):
        text = result["choices"][0]["message"]["content"]
        require(result.get("usage", {}).get("completion_tokens", 0) > 0, result)
        require(expected in text, text)


def cancel_stream(port, prompt="List ten facts about C++.", explicit=False):
    """Close an active stream after its first token and verify cleanup later."""
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 64, "stream": True,
        "prompt_cache_key": "cancel-stream",
        "request_id": "cancel-explicit" if explicit else "cancel-disconnect",
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
        require(response.getheader("X-Request-ID") == body["request_id"],
                dict(response.getheaders()))
        if explicit:
            cancel_request = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/cancel",
                data=json.dumps({"request_id": body["request_id"]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(cancel_request, timeout=HTTP_TIMEOUT) as cancelled:
                payload = json.load(cancelled)
            require(payload.get("status") == "cancellation_requested", payload)
            while response.readline():
                pass
        # Closing after a real token can leave a speculative target window
        # active and exercises cooperative rollback in the resident child.
    finally:
        connection.close()


def direct_stdio_cases(backend, label="DFlash2"):
    """Exercise the JSONL child independently of HTTP request formatting."""
    messages = [{"role": "user", "content": "What is 8+5? Answer with just 13."}]
    prompt = chat_prompt(messages)
    prefix = chat_prefix(messages)
    greedy = backend.generate(prompt, 8, 0, 0.95, 20, 0, 1, 0,
                              prefix=prefix, seed=42, cache_key="direct-a")
    repeated = backend.generate(prompt, 8, 0, 0.95, 20, 0, 1, 0,
                                prefix=prefix, seed=42, cache_key="direct-a")
    require(greedy[0] == repeated[0] and "13" in greedy[0], greedy)
    require(greedy[3] > 0 and repeated[1] > 0, repeated)

    sampled = backend.generate(prompt, 8, 0.7, 0.95, 20, 0, 1, 0,
                               prefix=prefix, seed=42, cache_key="direct-a")
    sampled_repeat = backend.generate(prompt, 8, 0.7, 0.95, 20, 0, 1, 0,
                                      prefix=prefix, seed=42, cache_key="direct-a")
    require(sampled[0] == sampled_repeat[0] and sampled[3] > 0, sampled)
    require(sampled_repeat[1] > 0, sampled_repeat)

    cancellation = threading.Event()

    def cancel_after_token(piece):
        if piece:
            backend.cancel(cancellation)

    cancelled = backend.generate(
        chat_prompt([{"role": "user", "content": "List twenty C++ language features."}]),
        64, 0, 0.95, 20, 0, 1, 0, cancellation=cancellation,
        on_token=cancel_after_token, seed=42, cache_key="cancelled")
    require(cancellation.is_set() and cancelled[-1] == "cancelled", cancelled)
    recovered = backend.generate(prompt, 8, 0, 0.95, 20, 0, 1, 0,
                                 prefix=prefix, seed=42, cache_key="direct-a")
    require(recovered[0] == greedy[0] and recovered[1] > 0, recovered)

    other_prompt = chat_prompt([
        {"role": "user", "content": "What is 4+4? Answer with just 8."}])
    other = backend.generate(other_prompt, 8, 0, 0.95, 20, 0, 1, 0,
                             seed=42, cache_key="direct-b")
    restored = backend.generate(prompt, 8, 0, 0.95, 20, 0, 1, 0,
                                prefix=prefix, seed=42, cache_key="direct-a")
    require("8" in other[0], other)
    require(restored[0] == greedy[0] and restored[1] > 0,
            "A/B/A context restore failed: " + repr(restored))

    cases = (
        ("Answer 6+6 with just 12.", "12"),
        ("Answer 9+9 with just 18.", "18"),
    )

    def run_case(item):
        text, expected = item
        result = backend.generate(
            chat_prompt([{"role": "user", "content": text}]),
            8, 0, 0.95, 20, 0, 1, 0, seed=42,
            cache_key=f"direct-concurrent-{expected}")
        return result, expected

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_case, cases))
    for result, expected in results:
        require(result[3] > 0 and expected in result[0], result)
    print(f"{label} stdio window/cache/sampling/cancel/concurrency: PASS")


def parity_cases(backend, namespace):
    """Return exact bytes for fixed greedy, sampled, coding, and retrieval requests."""
    cases = (
        ("Explain RAII in one short sentence.", 24, 0.0, None),
        ("Write one short sentence about deterministic C++ builds.",
         24, 0.7, None),
        ("Return only valid C++17 source code for a complete program that "
         "prints exactly 42 followed by a newline.", 64, 0.0, None),
        ("The passphrase is ZEPHYR-7319. Remember it and answer this question "
         "using only the passphrase: what is the passphrase?",
         24, 0.0, "ZEPHYR-7319"),
    )
    outputs = []
    for index, (text, max_tokens, temperature, expected) in enumerate(cases):
        messages = [{"role": "user", "content": text}]
        result = backend.generate(
            chat_prompt(messages), max_tokens, temperature, 0.95, 20,
            0, 1, 0, prefix=chat_prefix(messages), seed=42,
            cache_key=f"{namespace}-{index}")
        require(result[3] > 0 and result[-1] in ("stop", "length"), result)
        if expected is not None:
            require(result[0].strip() == expected, result)
        outputs.append(result[0])
    return outputs


def extract_cpp(text):
    if "```" not in text:
        return text.strip()
    chunks = text.split("```")
    for i in range(1, len(chunks), 2):
        candidate = chunks[i]
        if candidate.lstrip().startswith(("cpp\n", "c++\n", "C++\n")):
            candidate = candidate.split("\n", 1)[1]
        if "#include" in candidate or "int main" in candidate:
            return candidate.strip()
    return text.strip()


def compile_cpp_answer(text):
    source = extract_cpp(text)
    scratch_root = Path(__file__).with_name("tmp")
    scratch_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dflash-agent-", dir=scratch_root) as directory:
        path = Path(directory)
        source_path = path / "answer.cpp"
        binary_path = path / "answer"
        source_path.write_text(source)
        build = subprocess.run(
            ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
             str(source_path), "-o", str(binary_path)],
            capture_output=True, text=True, timeout=30)
        require(build.returncode == 0,
                f"generated C++ did not compile:\n{build.stderr}\n{source}")
        run = subprocess.run([str(binary_path)], capture_output=True,
                             text=True, timeout=5)
        require(run.returncode == 0 and run.stdout.strip() == "42",
                f"generated C++ returned {run.returncode}: {run.stdout!r} {run.stderr!r}")


def coding_agent_cases(port, label="DFlash2"):
    first_messages = [{
        "role": "user",
        "content": ("Return only valid C++17 source code, without Markdown. "
                    "Write a complete program that prints exactly 42 followed by a newline."),
    }]
    first_body = {"messages": first_messages, "temperature": 0,
                  "max_tokens": 64, "prompt_cache_key": "coding-session"}
    first = post(port, first_body)
    first_text = first["choices"][0]["message"]["content"]
    compile_cpp_answer(first_text)
    distractor = post(port, {
        "messages": [{"role": "user", "content": "Answer 1+1 with just 2."}],
        "temperature": 0, "max_tokens": 8,
        "prompt_cache_key": "coding-distractor",
    })
    require("2" in distractor["choices"][0]["message"]["content"], distractor)
    second_messages = first_messages + [
        {"role": "assistant", "content": first_text},
        {"role": "user", "content": (
            "Keep the observable output identical. Add a constexpr function named "
            "answer that returns 42, and have main print answer(). Return only source code.")},
    ]
    second = post(port, {"messages": second_messages, "temperature": 0,
                         "max_tokens": 96, "prompt_cache_key": "coding-session"})
    second_text = second["choices"][0]["message"]["content"]
    require(second.get("usage", {}).get("cached_tokens", 0) > 0, second)
    require("answer" in second_text, second_text)
    compile_cpp_answer(second_text)
    print(f"{label} multi-turn C++ compile/run quality: PASS")


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
    sidecars = parser.add_mutually_exclusive_group(required=True)
    sidecars.add_argument("--sidecar", help="DFlash2 GGUF")
    sidecars.add_argument("--mtp", help="dense NextN GGUF")
    parser.add_argument("--runner", default="./rdna4/llm/test_hip_llm")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--snapshot-max-tokens", type=int, default=0)
    parser.add_argument("--long-prompt-tokens", type=int, default=0,
                        help="target token count for a deterministic longer cached prompt")
    args = parser.parse_args()
    label = "Dense NextN" if args.mtp else "DFlash2"
    options = SimpleNamespace(
        runner=args.runner, model=args.model, context=args.context,
        moe_cache_mb=0, coding=False, qwen4_coding_profile=False,
        qwen4_exact=False, qwen4_mtp=None, qwen35_mtp=args.mtp,
        qwen35_mtp_draft=3, qwen35_mtp_window=bool(args.mtp),
        qwen35_server_profile=False, qwen35_dflash2=args.sidecar,
        qwen35_dflash2_draft=7,
        qwen35_snapshot_max_tokens=args.snapshot_max_tokens,
        context_cache_entries=2, context_cache_max_mib=2048,
    )
    ordinary_outputs = None
    if args.mtp:
        ordinary_options = SimpleNamespace(**vars(options))
        ordinary_options.qwen35_mtp = None
        ordinary_options.qwen35_server_profile = True
        ordinary = Backend(ordinary_options)
        try:
            ordinary_outputs = parity_cases(ordinary, "ordinary-parity")
        finally:
            ordinary.close()
    backend = Backend(options)
    _active_backend = backend
    if ordinary_outputs is not None:
        mtp_outputs = parity_cases(backend, "mtp-parity")
        require(mtp_outputs == ordinary_outputs,
                "Dense NextN changed ordinary target response bytes")
        print("Dense NextN ordinary-target byte parity: PASS")
    direct_stdio_cases(backend, label)

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
        idle_cancel = post_error(args.port, {}, 404, "/v1/cancel")
        require(idle_cancel.get("status") == "request_not_found", idle_cancel)
        bad_cache = post_error(args.port, {
            "messages": [{"role": "user", "content": "ignored"}],
            "prompt_cache_key": ["not", "a", "string"],
        }, 400)
        require(bad_cache.get("error", {}).get("type") == "invalid_request_error",
                bad_cache)
        bad_request_id = post_error(args.port, {
            "messages": [{"role": "user", "content": "ignored"}],
            "request_id": ["not", "a", "string"],
        }, 400)
        require(bad_request_id.get("error", {}).get("type") == "invalid_request_error",
                bad_request_id)
        bad_messages = post_error(args.port, {"messages": {"role": "user"}}, 400)
        require(bad_messages.get("error", {}).get("type") == "invalid_request_error",
                bad_messages)

        cases = (
            ("Write one short sentence about C++.", "C++"),
            ("What is 2+2? Answer with just the number 4.", "4"),
        )
        for case_index, (prompt_text, expected) in enumerate(cases):
            prompt = [{"role": "user", "content": prompt_text}]
            greedy = {"messages": prompt, "temperature": 0, "max_tokens": 8,
                      "prompt_cache_key": f"http-repeat-{case_index}"}
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
            # This tokenizer averages close to six source characters per token
            # for the repeated sentence. Keep the requested value meaningful
            # and verify the actual API count below rather than relying only
            # on a character heuristic.
            char_budget = args.long_prompt_tokens * 6
            long_text = (unit * (char_budget // len(unit) + 1))[:char_budget]
            long_body = {"messages": [{"role": "user", "content": long_text}],
                         "temperature": 0, "max_tokens": 8,
                         "prompt_cache_key": "long-session"}
            long_a = post(args.port, long_body)
            actual_long_tokens = long_a.get("usage", {}).get("prompt_tokens", 0)
            require(actual_long_tokens >= int(args.long_prompt_tokens * 0.85),
                    f"long prompt token target missed: requested "
                    f"{args.long_prompt_tokens}, got {actual_long_tokens}")
            long_b = post(args.port, long_body)
            require(long_a["choices"][0]["message"]["content"] ==
                    long_b["choices"][0]["message"]["content"],
                    "long cached prompt was not repeatable")
            require(long_b.get("usage", {}).get("cached_tokens", 0) > 0,
                    "long prompt did not report cache reuse")

            # A same-context repeat can still pass when the live GPU state is
            # correct but the host snapshot is incomplete.  Force another
            # conversation through the one-runner backend, then require the
            # long context to restore byte-for-byte equivalent output.
            distractor = post(args.port, {
                "messages": [{"role": "user",
                              "content": "Answer only: context switch complete."}],
                "temperature": 0, "max_tokens": 8,
                "prompt_cache_key": "long-distractor",
            })
            require(distractor.get("usage", {}).get("completion_tokens", 0) > 0,
                    distractor)
            long_c = post(args.port, long_body)
            require(long_a["choices"][0]["message"]["content"] ==
                    long_c["choices"][0]["message"]["content"],
                    "long context changed after an interleaved conversation")
            require(long_c.get("usage", {}).get("cached_tokens", 0) > 0,
                    "long context was not restored after interleaving")

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
        cancel_stream(args.port, "List twenty C++ standard library types.",
                      explicit=True)
        time.sleep(1)
        recovery = post(args.port, {
            "messages": [{"role": "user", "content": "Answer 3+3 with 6."}],
            "temperature": 0, "max_tokens": 8,
            "prompt_cache_key": "recovery",
        })
        require(recovery.get("usage", {}).get("completion_tokens", 0) > 0,
                recovery)
        require("6" in recovery["choices"][0]["message"]["content"], recovery)

        concurrent_quality_cases(args.port)

        # Capacity two: after A/B/C, A's committed snapshot must be evicted.
        eviction_results = []
        for key, value in (("evict-a", "11"), ("evict-b", "12"),
                           ("evict-c", "13"), ("evict-a", "11")):
            eviction_results.append(post(args.port, {
                "messages": [{"role": "user", "content":
                              f"Answer with just the number {value}."}],
                "temperature": 0, "max_tokens": 8,
                "prompt_cache_key": key,
            }))
        require(eviction_results[-1].get("usage", {}).get("cached_tokens", -1) == 0,
                "LRU did not evict the oldest context")

        coding_agent_cases(args.port, label)

        prompt = [{"role": "user", "content": cases[0][0]}]
        sampled = {
            "messages": prompt, "temperature": 0.7, "top_p": 0.95,
            "top_k": 20, "seed": 42, "max_tokens": 8,
            "prompt_cache_key": "sampled-cpp",
        }
        sampled_a = post(args.port, sampled)
        sampled_b = post(args.port, sampled)
        sampled_text = sampled_a["choices"][0]["message"]["content"]
        require(sampled_a.get("usage", {}).get("completion_tokens", 0) > 0,
                sampled_a)
        require(sampled_text == sampled_b["choices"][0]["message"]["content"],
                "seeded sampled request was not repeatable")
        require("C++" in sampled_text, sampled_text)
        print(f"{label} HTTP greedy/sampled repeatability and quality: PASS")
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
