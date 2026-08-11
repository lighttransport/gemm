#!/usr/bin/env python3
"""One-shot full-EP DS4F serving benchmark with a tokenizer-real prompt."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
from ds4f_serve_runner import Sampling, Serve, load_lib  # noqa: E402
from ds4f_tokenizer import DS4FTokenizer  # noqa: E402


DEFAULT_TEXT = (
    "Explain how a heterogeneous CPU and GPU inference engine should schedule "
    "attention, shared experts, and routed experts while preserving numerical "
    "quality. Include concrete tradeoffs, failure modes, and measurements. "
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-dir", default=os.environ.get("DS4F_STAGE_DIR", "/tmp/ds4f_single"))
    ap.add_argument("--tokenizer", default="/mnt/disk1/models/ds4f-0731/tokenizer.json")
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--warm-decode", type=int, default=32)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cmgs", type=int, default=4)
    ap.add_argument("--hip-device", type=int, default=0)
    ap.add_argument("--hip-mxfp4-wmma", type=int, choices=(0, 1, 2), default=1)
    ap.add_argument("--hip-routed-ffn", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-expert-stream", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-expert-pinned-staging", type=int, choices=(0, 1), default=0)
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    tokenizer = DS4FTokenizer(args.tokenizer)
    unit = tokenizer.encode(DEFAULT_TEXT, add_bos=True)
    if not unit:
        raise SystemExit("tokenizer produced an empty prompt")
    prompt = (unit * ((args.prompt_tokens + len(unit) - 1) // len(unit)))[:args.prompt_tokens]

    lib_path = os.environ.get("DS4F_SERVE_LIB", os.path.join(HERE, "../../libds4f_serve.so"))
    sess = Serve(load_lib(lib_path), args.stage_dir, not args.cpu_only,
                 args.hip_device, args.threads, args.cmgs,
                 args.prompt_tokens + args.warm_decode + args.decode_tokens + 8,
                 # Keep the one-shot benchmark on the same tuned path as the
                 # production runner.  This tuple mirrors the runner defaults.
                 (1, 1, 1, 1, 1, 1, 1, args.hip_routed_ffn, 2, 1, 1, 2,
                  args.hip_mxfp4_wmma, args.hip_expert_stream, 128,
                  args.hip_expert_pinned_staging))
    greedy = Sampling(0.0, 1.0, 1, 0.0, 1.0, 1)
    try:
        t0 = time.perf_counter()
        if sess.prefill(prompt, 0) != 0:
            raise RuntimeError("prefill failed")
        prefill_s = time.perf_counter() - t0
        pos = len(prompt)
        for _ in range(args.warm_decode):
            token = sess.sample(greedy)
            if token < 0 or sess.decode(token, pos) < 0:
                raise RuntimeError("warm decode failed")
            pos += 1
        t0 = time.perf_counter()
        completed = 0
        for _ in range(args.decode_tokens):
            token = sess.sample(greedy)
            if token < 0 or sess.decode(token, pos) < 0:
                raise RuntimeError("timed decode failed")
            pos += 1
            completed += 1
        decode_s = time.perf_counter() - t0
        print("prefill tokens=%d seconds=%.6f tok/s=%.3f" %
              (len(prompt), prefill_s, len(prompt) / prefill_s))
        print("decode warm=%d tokens=%d seconds=%.6f tok/s=%.3f" %
              (args.warm_decode, completed, decode_s, completed / decode_s))
    finally:
        sess.close()


if __name__ == "__main__":
    main()
