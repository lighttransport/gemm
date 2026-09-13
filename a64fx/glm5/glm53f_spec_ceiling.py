#!/usr/bin/env python3
"""Optimistic GLM-5.3F speculative-decode ceiling from measured A64FX costs."""
import argparse


def accepted(alpha, drafts):
    return sum(alpha ** i for i in range(drafts + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-ms", type=float, default=14.53)
    ap.add_argument("--shared-ms", type=float, default=4.234)
    ap.add_argument("--wire-ms", type=float, default=6.116)
    ap.add_argument("--mtp-ms", type=float, default=0.524)
    ap.add_argument("--expert-speedup", type=float, default=1.45)
    ap.add_argument("--max-drafts", type=int, default=8)
    args = ap.parse_args()
    routed = args.expert_ms - args.shared_ms
    print("GLM53F_SPEC_CEILING optimistic=YES missing_graph_cost=0")
    print("measured routed_ms=%.3f shared_ms=%.3f wire_ms=%.3f mtp_partial_ms=%.3f speedup=%.3f" %
          (routed, args.shared_ms, args.wire_ms, args.mtp_ms,
           args.expert_speedup))
    print("drafts verify_tokens lower_bound_ms alpha70 alpha80 alpha90 alpha100")
    for drafts in range(1, args.max_drafts + 1):
        tokens = drafts + 1
        # Optimistic assumptions: routed work scales with verified tokens but
        # every shared-expert weight is read only once; one 84-call collective
        # sequence serves the entire verification batch; all expert work gets
        # the measured MXFP4 speedup. MTP cost includes only its measured
        # expert+collective path. Missing attention/eh_proj/lm_head cost is zero.
        ms = ((tokens * routed + args.shared_ms) / args.expert_speedup +
              args.wire_ms + drafts * args.mtp_ms)
        rates = [1000.0 * accepted(a, drafts) / ms
                 for a in (0.7, 0.8, 0.9, 1.0)]
        print("%d %d %.3f %.2f %.2f %.2f %.2f" %
              (drafts, tokens, ms, *rates))


if __name__ == "__main__":
    main()
