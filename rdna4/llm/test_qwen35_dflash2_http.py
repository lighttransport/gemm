"""Opt-in GPU quality gate for the resident DFlash2 HTTP window path.

Run with ``--model TARGET --sidecar DFLASH``.  The test is intentionally
separate from the CPU-only protocol suite because it loads both GGUF files.
"""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--runner", default="./rdna4/llm/test_hip_llm")
    parser.add_argument("--port", type=int, default=18090)
    args = parser.parse_args()
    command = [
        "python3", "rdna4/llm/codex_server.py", args.model,
        "--runner", args.runner, "--context", "512", "--port", str(args.port),
        "--max-output", "8", "--qwen35-dflash2", args.sidecar,
        "--qwen35-dflash2-draft", "7",
    ]
    # The runner inherits this stream from codex_server.  Discard it here so
    # long GPU runs cannot deadlock on an undrained diagnostic pipe.
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    try:
        for _ in range(180):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                health = json.load(response)
                break
            except Exception:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"DFlash2 server exited before readiness (status {process.returncode})")
                time.sleep(1)
        else:
            raise RuntimeError("DFlash2 server did not become ready")
        require(health.get("status") == "ready", health)

        prompt = [{"role": "user", "content": "Write one short sentence about C++."}]
        greedy = {"messages": prompt, "temperature": 0, "max_tokens": 8}
        first = post(args.port, greedy)
        second = post(args.port, greedy)
        first_text = first["choices"][0]["message"]["content"]
        require(first.get("usage", {}).get("completion_tokens", 0) > 0, first)
        require(first_text == second["choices"][0]["message"]["content"],
                "greedy request was not repeatable")
        require("C++" in first_text, first_text)

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


if __name__ == "__main__":
    main()
