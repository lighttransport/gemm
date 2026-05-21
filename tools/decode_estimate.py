#!/usr/bin/env python3
"""
decode_estimate.py — A64FX/Fugaku decode throughput estimator for GLM-5.1
(GlmMoeDsa: 744B/40B MoE, MLA + DeepSeek Sparse Attention).

Estimates single-stream and batched decode tok/s at long context from a simple
roofline: per-step time = max(memory-bound, comm-bound) with overlap, plus a
fit-in-HBM check. The cost constants come from the a64fx/utofu-tests benchmark
suite and the in-model decode profile (see a64fx/llm/decode.md):

  * realized matvec/GEMV bandwidth ~650 GB/s/node (NOT the 900 GB/s stream peak)
  * uTofu per-hop = 1.23 us fixed + bytes / 6.36 GB/s
  * tree all-reduce step count = log2(floor_pow2(G)) (+2 if G not a power of 2)
  * comm is context-INDEPENDENT; weights amortize over batch, KV does not

Defaults model GLM-5.1; override any constant on the command line.

Examples
    ./decode_estimate.py --bits 8 --nodes 48 --ctx 1000000
    ./decode_estimate.py --bits 4 --nodes 24,48,60 --ctx 1000000
    ./decode_estimate.py --bits 8 --nodes 48 --ctx 1000000 --batch 32
    ./decode_estimate.py --sweep            # the canonical table

All numbers are OPTIMISTIC roofline ceilings. Real single-stream is typically
2-3x lower (DSA top-k sort, softmax, routing, kernel launch are not modeled);
use --derate to apply an efficiency factor.
"""

import argparse
import math

# ----------------------------------------------------------------------------
# GLM-5.1 (GlmMoeDsa) architecture — HF GlmMoeDsaConfig defaults
# ----------------------------------------------------------------------------
TOTAL_PARAMS   = 744e9
ACTIVE_PARAMS  = 40e9
N_LAYERS       = 78
HIDDEN         = 6144
# MLA: compressed KV latent stored per token per layer
KV_LORA_RANK   = 512
QK_ROPE_DIM    = 64
MLA_LATENT     = KV_LORA_RANK + QK_ROPE_DIM        # 576 values / token / layer
# DSA: lightning indexer + sparse top-k selection
INDEX_TOPK     = 2048
INDEX_HEAD_DIM = 128
# fraction of layers that run a full indexer scan over all context
# (indexer_types: first full, then every-k-th full, rest "shared" -> ~half)
INDEX_LAYER_FRAC = 0.5

# footprints (GiB on disk/RAM) for unsloth dynamic quants; 16-bit = bf16 weights
FOOTPRINT_GB = {2: 220.0, 4: 420.0, 8: 805.0, 16: 1650.0}

# ----------------------------------------------------------------------------
# A64FX / Fugaku platform constants (measured)
# ----------------------------------------------------------------------------
HBM_GIB_PER_NODE = 32.0          # HBM2 per node
ACT_RESERVE_GIB  = 2.0           # leave headroom for activations / runtime
BW_PER_NODE      = 650e9         # realized decode matvec BW (B/s); stream peak ~900e9
HOP_FIXED_S      = 1.23e-6       # uTofu fixed per-hop latency
HOP_BW           = 6.36e9        # uTofu per-link payload BW (B/s)
COLLECTIVES_PER_LAYER = 2        # attn TP all-reduce + MoE dispatch/combine.
                                 # MEASURED ok: a64fx/utofu-tests/moe_dispatch_bench
                                 # shows real multi-TNI dispatch+combine = 0.88-0.93x
                                 # of 2x tree-all-reduce (this proxy is ~right, slightly
                                 # conservative) -- but ONLY with multi-TNI all-to-all;
                                 # naive single-TNI is ~2.8x and would blow the budget.

GIB = 1024**3


def tree_steps(g):
    """All-reduce step count over a group of g nodes (Rabenseifner/recursive
    doubling), calibrated to the measured 12-node = 5 steps."""
    if g <= 1:
        return 0
    floor_p2 = 1 << (g.bit_length() - 1)
    steps = int(math.log2(floor_p2))
    if g != floor_p2:
        steps += 2                 # non-power-of-2 fold + broadcast
    return steps


def pick_tp(nodes):
    """Heuristic TP-group size: keep it small (8) to bound per-collective
    latency, growing only when the node count is small."""
    if nodes <= 8:
        return nodes
    if nodes % 8 == 0:
        return 8
    if nodes % 12 == 0:
        return 12
    return 8 if nodes >= 8 else nodes


def estimate(bits, nodes, ctx, batch, kv_bytes, idx_bytes, act_bytes,
             tp=None, derate=1.0, bw_per_node=BW_PER_NODE):
    tp = tp or pick_tp(nodes)

    # ---- per-token weight read (amortizes over batch: GEMV -> GEMM) ----
    bytes_per_param = bits / 8.0
    weight_read = ACTIVE_PARAMS * bytes_per_param            # B / step (any batch)

    # ---- per-token attention read at this context (MLA + DSA), per sequence ----
    n_idx_layers = max(1, round(N_LAYERS * INDEX_LAYER_FRAC))
    indexer_scan = INDEX_HEAD_DIM * idx_bytes * n_idx_layers * ctx
    main_attn    = MLA_LATENT * kv_bytes * min(INDEX_TOPK, ctx) * N_LAYERS
    attn_read    = indexer_scan + main_attn                  # B / token / sequence

    # ---- memory-bound step time ----
    agg_bw = bw_per_node * nodes
    mem_traffic_step = weight_read + attn_read * batch       # weights shared, KV per-seq
    t_mem = mem_traffic_step / agg_bw

    # ---- comm-bound step time ----
    payload = HIDDEN * act_bytes * batch                     # all-reduce payload / step
    per_hop = HOP_FIXED_S + payload / HOP_BW
    per_collective = tree_steps(tp) * per_hop
    t_comm = N_LAYERS * COLLECTIVES_PER_LAYER * per_collective

    # ---- combine (overlap -> max; also report no-overlap sum) ----
    t_overlap = max(t_mem, t_comm) / derate
    t_serial  = (t_mem + t_comm) / derate

    # ---- memory fit ----
    kv_total_b  = MLA_LATENT * kv_bytes * N_LAYERS * ctx * batch
    idx_total_b = INDEX_HEAD_DIM * idx_bytes * n_idx_layers * ctx * batch
    footprint_b = FOOTPRINT_GB.get(bits, TOTAL_PARAMS * bytes_per_param / GIB * 1.12) * GIB \
        if bits in FOOTPRINT_GB else TOTAL_PARAMS * bytes_per_param * 1.12
    per_node_b  = (footprint_b + kv_total_b + idx_total_b) / nodes
    per_node_gib = per_node_b / GIB
    fits = per_node_gib <= (HBM_GIB_PER_NODE - ACT_RESERVE_GIB)

    return {
        "tp": tp,
        "agg_bw_tbs": agg_bw / 1e12,
        "weight_gb": weight_read / 1e9,
        "attn_gb": attn_read / 1e9,
        "t_mem_ms": t_mem * 1e3,
        "t_comm_ms": t_comm * 1e3,
        "t_overlap_ms": t_overlap * 1e3,
        "t_serial_ms": t_serial * 1e3,
        "agg_toks_overlap": batch / t_overlap,
        "stream_toks_overlap": 1.0 / t_overlap,
        "agg_toks_serial": batch / t_serial,
        "bound": "comm" if t_comm > t_mem else "mem",
        "per_node_gib": per_node_gib,
        "fits": fits,
    }


def fmt_row(label, r):
    fit = "OK " if r["fits"] else "OOM"
    return (f"{label:<22} TP={r['tp']:<3} {r['agg_bw_tbs']:6.1f}TB/s  "
            f"mem {r['t_mem_ms']:6.2f}ms  comm {r['t_comm_ms']:6.2f}ms  "
            f"-> {r['t_overlap_ms']:6.2f}ms ({r['bound']:>4})  "
            f"stream {r['stream_toks_overlap']:6.0f} t/s  agg {r['agg_toks_overlap']:7.0f} t/s  "
            f"{r['per_node_gib']:5.1f}GiB/node {fit}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", default="8", help="effective weight bits (2,4,8,16); comma list ok")
    p.add_argument("--nodes", default="48", help="node count; comma list ok")
    p.add_argument("--ctx", type=int, default=1_000_000, help="context length (tokens)")
    p.add_argument("--batch", type=int, default=1, help="concurrent sequences (decode batch)")
    p.add_argument("--tp", type=int, default=0, help="force TP group size (0 = heuristic)")
    p.add_argument("--kv-bytes", type=float, default=2.0, help="MLA latent dtype bytes (bf16=2)")
    p.add_argument("--idx-bytes", type=float, default=1.0, help="DSA indexer key bytes (fp8=1)")
    p.add_argument("--act-bytes", type=float, default=2.0, help="activation/all-reduce dtype bytes")
    p.add_argument("--derate", type=float, default=1.0, help="efficiency derate (e.g. 0.4 = 2.5x slower)")
    p.add_argument("--bw", type=float, default=BW_PER_NODE/1e9, help="realized BW GB/s/node")
    p.add_argument("--sweep", action="store_true", help="print the canonical bits x nodes table")
    args = p.parse_args()

    bw = args.bw * 1e9

    if args.sweep:
        print(f"GLM-5.1 decode roofline @ ctx={args.ctx:,} batch={args.batch} "
              f"BW={args.bw:.0f}GB/s/node derate={args.derate}")
        print("-" * 118)
        for bits in (2, 4, 8, 16):
            for nodes in (12, 24, 48, 60, 96, 192):
                r = estimate(bits, nodes, args.ctx, args.batch,
                             args.kv_bytes, args.idx_bytes, args.act_bytes,
                             tp=(args.tp or None), derate=args.derate, bw_per_node=bw)
                print(fmt_row(f"{bits}-bit / {nodes} nodes", r))
            print()
        return

    bits_list  = [int(x) for x in str(args.bits).split(",")]
    nodes_list = [int(x) for x in str(args.nodes).split(",")]
    print(f"GLM-5.1 decode roofline @ ctx={args.ctx:,} batch={args.batch} "
          f"BW={args.bw:.0f}GB/s/node derate={args.derate}")
    print("-" * 118)
    for bits in bits_list:
        for nodes in nodes_list:
            r = estimate(bits, nodes, args.ctx, args.batch,
                         args.kv_bytes, args.idx_bytes, args.act_bytes,
                         tp=(args.tp or None), derate=args.derate, bw_per_node=bw)
            print(fmt_row(f"{bits}-bit / {nodes} nodes", r))


if __name__ == "__main__":
    main()
