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
    ap.add_argument("--speculative-tokens", type=int, default=0,
                    help="enable greedy DSpark blocks (checkpoint maximum: 5)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cmgs", type=int, default=4)
    ap.add_argument("--hip-device", type=int, default=0)
    ap.add_argument("--ep-rank", type=int, default=0)
    ap.add_argument("--ep-size", type=int, default=1,
                    help="expert-parallel shard count (requires matching staged manifest)")
    ap.add_argument("--hip-mxfp4-wmma", type=int, choices=(0, 1, 2), default=1)
    ap.add_argument("--hip-routed-ffn", type=int, choices=(0, 1), default=0)
    ap.add_argument("--hip-decode-routed-ffn", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-tb2-batch", type=int, choices=(0, 1), default=0)
    ap.add_argument("--hip-expert-stream", type=int, choices=(0, 1), default=1)
    ap.add_argument("--hip-expert-pinned-staging", type=int, choices=(0, 1), default=0)
    ap.add_argument("--hip-expert-cache-mb", type=int, default=0,
                    help="cache prompt-hot routed experts on GPU before decode (0 disables)")
    ap.add_argument("--hip-expert-cache-reserve-mb", type=int, default=1536)
    ap.add_argument("--adaptive-cache-period", type=int, default=0,
                    help="refresh decode-window expert residency every N exact tokens")
    ap.add_argument("--adaptive-cache-mb", type=int, default=-1)
    ap.add_argument("--adaptive-cache-reserve-mb", type=int, default=1536)
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
                  args.hip_expert_pinned_staging, args.hip_tb2_batch,
                  args.hip_decode_routed_ffn), args.speculative_tokens,
                 args.ep_rank, args.ep_size)
    greedy = Sampling(0.0, 1.0, 1, 0.0, 1.0, 1)
    try:
        if (args.hip_expert_cache_mb or args.adaptive_cache_period) and \
                sess.enable_route_telemetry(True) != 0:
            raise RuntimeError("route telemetry is unavailable")
        t0 = time.perf_counter()
        if sess.prefill(prompt, 0) != 0:
            raise RuntimeError("prefill failed")
        prefill_s = time.perf_counter() - t0
        cache_result = None
        if args.hip_expert_cache_mb:
            cache_result = sess.cache_hot_experts(
                args.hip_expert_cache_mb, args.hip_expert_cache_reserve_mb, 1)
            if cache_result < 0:
                raise RuntimeError("hot expert cache initialization failed")
        if args.adaptive_cache_period:
            if sess.set_adaptive_cache(args.adaptive_cache_period,
                                       args.adaptive_cache_mb,
                                       args.adaptive_cache_reserve_mb) != 0:
                raise RuntimeError("adaptive expert cache is unavailable")
        pos = len(prompt)
        warmed = 0
        while warmed < args.warm_decode:
            token = sess.sample(greedy)
            if token < 0:
                raise RuntimeError("warm decode failed")
            if args.speculative_tokens:
                toks, _ = sess.speculate(token, pos, args.warm_decode - warmed)
                warmed += len(toks); pos = sess.pos()
            else:
                if sess.decode(token, pos) < 0: raise RuntimeError("warm decode failed")
                warmed += 1; pos += 1
        t0 = time.perf_counter()
        completed = 0; blocks = 0; generated = []
        while completed < args.decode_tokens:
            token = sess.sample(greedy)
            if token < 0:
                raise RuntimeError("timed decode failed")
            if args.speculative_tokens:
                toks, _ = sess.speculate(token, pos, args.decode_tokens - completed)
                generated.extend(toks); completed += len(toks); blocks += 1
                pos = sess.pos()
            else:
                if sess.decode(token, pos) < 0: raise RuntimeError("timed decode failed")
                generated.append(token); pos += 1; completed += 1
        decode_s = time.perf_counter() - t0
        print("prefill tokens=%d seconds=%.6f tok/s=%.3f" %
              (len(prompt), prefill_s, len(prompt) / prefill_s))
        if cache_result is not None:
            print("hot expert cache result=%d" % cache_result)
        print("decode warm=%d tokens=%d seconds=%.6f tok/s=%.3f" %
              (args.warm_decode, completed, decode_s, completed / decode_s))
        if args.speculative_tokens:
            print("dspark blocks=%d committed/block=%.3f" %
                  (blocks, completed / max(blocks, 1)))
        print("generation=%r" % tokenizer.decode(generated))
    finally:
        sess.close()


if __name__ == "__main__":
    main()
