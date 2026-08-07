#!/usr/bin/env python3
"""Measure DS4F HTTP TTFT and generation throughput.

Use max_tokens=1 for a prefill/TTFT sample and a larger generation for decode.
The server's final usage object supplies model token counts; the first SSE
event timestamp includes prompt prefill plus the first decode step.
"""

import argparse
import json
import statistics
import time
import urllib.request


def request(url, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    start = time.perf_counter()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as response:
        if body.get("stream"):
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    event = json.loads(payload.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                if first is None and (event.get("type", "").endswith("delta") or
                                       event.get("choices")):
                    first = time.perf_counter()
                candidate = event.get("usage")
                if isinstance(candidate, dict):
                    usage = candidate
                response_obj = event.get("response")
                if isinstance(response_obj, dict) and isinstance(response_obj.get("usage"), dict):
                    usage = response_obj["usage"]
        else:
            payload = json.loads(response.read().decode("utf-8"))
            usage = payload.get("usage") or {}
            first = time.perf_counter()
    end = time.perf_counter()
    prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    first_s = (first or end) - start
    wall_s = end - start
    decode_s = max(wall_s - first_s, 1e-9)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "ttft_s": first_s, "wall_s": wall_s,
            "prefill_tok_s": prompt_tokens / max(first_s, 1e-9),
            "decode_tok_s": completion_tokens / decode_s}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:21274/v1/responses")
    parser.add_argument("--model", default="ds4f")
    parser.add_argument("--prompt", default="Implement a bounded binary search in C.")
    parser.add_argument("--prompt-file")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()
    if args.prompt_file:
        with open(args.prompt_file) as f:
            args.prompt = f.read()
    body = {"model": args.model, "input": args.prompt,
            "max_output_tokens": args.max_tokens,
            "temperature": 0, "stream": not args.no_stream}
    for _ in range(args.warmup):
        request(args.url, body)
    samples = [request(args.url, body) for _ in range(args.repeat)]
    for index, sample in enumerate(samples, 1):
        print(json.dumps({"sample": index, **sample}, sort_keys=True))
    for key in ("ttft_s", "prefill_tok_s", "decode_tok_s"):
        values = [sample[key] for sample in samples]
        print(json.dumps({"summary": key, "median": statistics.median(values),
                          "min": min(values), "max": max(values)}, sort_keys=True))


if __name__ == "__main__":
    main()
