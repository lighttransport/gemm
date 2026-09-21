"""Opt-in GPU quality gate for the resident DFlash2 HTTP window path.

Run with ``--model TARGET --sidecar DFLASH``.  The test is intentionally
separate from the CPU-only protocol suite because it loads both GGUF files.
"""
import argparse
import concurrent.futures
import http.client
import json
import os
import subprocess
import time
import urllib.request


HTTP_TIMEOUT = 180


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


def cancel_stream(port):
    """Close an active stream after its first token and verify cleanup later."""
    body = {
        "messages": [{"role": "user", "content": "List ten facts about C++."}],
        "temperature": 0, "max_tokens": 64, "stream": True,
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=HTTP_TIMEOUT)
    try:
        connection.request("POST", "/v1/chat/completions",
                           body=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        require(response.status == 200, response.status)
        while response.readline():
            # Closing the connection exercises the server's disconnect watcher.
            break
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--runner", default="./rdna4/llm/test_hip_llm")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--long-prompt-tokens", type=int, default=0,
                        help="also exercise a deterministic longer cached prompt")
    args = parser.parse_args()
    command = [
        "python3", "rdna4/llm/codex_server.py", args.model,
        "--runner", args.runner, "--context", str(args.context), "--port", str(args.port),
        "--max-output", "8", "--qwen35-dflash2", args.sidecar,
        "--qwen35-dflash2-draft", "7",
    ]
    os.makedirs("tmp/qwen38/dflash-http-quality", exist_ok=True)
    server_log_path = os.path.join(
        "tmp/qwen38/dflash-http-quality", f"server-{args.port}.log")
    server_log = open(server_log_path, "w", encoding="utf-8")
    # Keep diagnostics in a repository-local file so long GPU runs cannot
    # deadlock on an undrained subprocess pipe, while failures remain useful.
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                stderr=server_log, text=True)
    try:
        for _ in range(180):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                    health = json.load(response)
                break
            except Exception:
                if process.poll() is not None:
                    server_log.flush()
                    with open(server_log_path, encoding="utf-8") as diagnostics:
                        tail = diagnostics.read()[-4000:]
                    raise RuntimeError(
                        f"DFlash2 server exited before readiness (status {process.returncode})\n{tail}")
                time.sleep(1)
        else:
            raise RuntimeError("DFlash2 server did not become ready")
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
            first_text = first["choices"][0]["message"]["content"]
            require(first.get("usage", {}).get("completion_tokens", 0) > 0, first)
            require(first_text == second["choices"][0]["message"]["content"],
                    "greedy request was not repeatable")
            require(second.get("usage", {}).get("cached_tokens", 0) > 0,
                    "repeated prompt did not report cache reuse")
            require(expected in first_text, first_text)

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

        cancel_stream(args.port)
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
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        server_log.close()


if __name__ == "__main__":
    main()
