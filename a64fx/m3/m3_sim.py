#!/usr/bin/env python3
"""
m3_sim.py — minimal-node sizing for MiniMax-M3 (428B MoE) on A64FX / Fugaku.

Ports `m3_arena_size` (common/m3_impl.h) — the exact per-rank HBM arena with EP+TP sharding — and
adds the mxfp8 weight factor + the EP-imbalance term that actually sets the floor: the busiest EP
rank owns `ceil(N_EXP / N)` experts. TP is mandatory (M3_TP=1) so dense/embed/head/attn shard ~1/N;
experts shard by EP (e % N). Node HBM 32 GB, usable ~27.

Calibration: mxfp8 = 1.03125 B/weight (1 + e8m0 1/32) vs bf16 2 B → 0.516; matches the measured
manifest split exactly (fp8 experts 426 GB / bf16 826 GB = 0.516). embed+head stay bf16 in the fp8
model (manifest embed+head = 4.9 GB = bf16 size). m3.md anchor: ~18-23 GB/rank @96n bf16 TP.

Run:    python3 m3_sim.py
Import: from m3_sim import min_nodes, arena_gb   # min_nodes(fmt, ctx, M) -> (N, arena_gb)
"""
import math

# ---- MiniMax-M3 config (common/m3.h) ----
H = 6144; N_HEADS = 64; N_KV = 4; HEAD_DIM = 128
QD  = N_HEADS * HEAD_DIM      # 8192  q_dim
KVD = N_KV    * HEAD_DIM      # 512   kv_dim (GQA)
VOCAB = 200064
N_LAYERS = 60; N_DENSE = 3; N_MOE = N_LAYERS - N_DENSE      # 57
N_EXP = 128; MOE_INTER = 3072; DENSE_INTER = 12288
IDX_DIM = 128; IDX_Q_DIM = 4 * IDX_DIM                      # 512 (MSA index proj)
ROTARY_DIM = 64
USABLE_GB = 27.0                      # 32 GB HBM - reserve/activations/scratch

WB = {'bf16': 2.0, 'fp8': 1.03125}    # bytes/weight (fp8 = 1 + e8m0 scale 1/32)
EMB_B = 2.0                           # embed/head stay bf16 even in the fp8 model
# Operating overhead ABOVE the weight+KV arena: staged-blob page cache + THP + pool. These are
# RSS-invisible (m3.md: "validate via MemFree, not RSS" — bit DS4P repeatedly). Calibrated so
# arena_gb(96,'bf16',8192) lands in the measured ~18-23 GB/rank band. Conservative on purpose:
# underestimating the arena risks an OOM, and an OOM SIGKILL degrades PMIx and costs the whole alloc.
OVERHEAD_GB = 4.5

def _ceil(a, b): return (a + b - 1) // b

def arena_gb(N, fmt='fp8', ctx=256, M=1, kvb=2, tp=True, cp=False):
    """Busiest-rank HBM arena (GB). Busiest rank owns ceil(N_EXP/N) experts and a TP head/dim shard."""
    wb = WB[fmt]
    no     = _ceil(N_EXP, N)                                 # busiest-rank experts (EP imbalance)
    qrows  = _ceil(N_HEADS, N) * HEAD_DIM if tp else QD      # TP head-shard (busiest)
    shrows = _ceil(MOE_INTER, N)   if tp else MOE_INTER
    ffrows = _ceil(DENSE_INTER, N) if tp else DENSE_INTER
    hrows  = _ceil(VOCAB, N) if tp else VOCAB
    erows  = _ceil(VOCAB, N) if tp else VOCAB
    kvslots = _ceil(ctx, N) if cp else ctx                   # CP shards the KV across ranks

    # attention: wq/wk/wv/wo (fp8) + q/k/out norms (bf16) + K/V cache (M streams)
    attn = (qrows*H + 2*KVD*H + H*qrows) * wb + (2*H + 2*HEAD_DIM) * 2
    attn += 2 * kvslots * KVD * kvb * M
    # MSA indexer: idx_q/idx_k proj (bf16) + norms + idx-key cache (1 MQA head)
    msa = (IDX_Q_DIM*H + IDX_DIM*H) * 2 + 2*IDX_DIM*2 + kvslots * IDX_DIM * kvb * M
    # per MoE layer: attn + msa + router gate(F32) + gate bias + shared expert + owned experts
    per_moe = (attn + msa + N_EXP*H*4 + N_EXP*4
               + (2*shrows*H + H*shrows) * wb
               + no * 3 * MOE_INTER * H * wb)
    # per dense layer: attn + dense FFN
    per_dense = attn + (2*ffrows*H + H*ffrows) * wb
    total = N_DENSE*per_dense + N_MOE*per_moe
    total += erows*H*EMB_B + hrows*H*EMB_B + H*2               # embed + head + out_norm
    total += ctx * (ROTARY_DIM//2) * 4 * 2                     # rope cos/sin
    return total / 1e9 + OVERHEAD_GB                           # + operating overhead (page cache/THP)

def min_nodes(fmt='fp8', ctx=256, M=1, kvb=2, usable_gb=USABLE_GB, tp=True, cp=False, nmax=256):
    """Fewest EP nodes N with busiest-rank arena <= usable_gb. Returns (N, arena_gb) or None."""
    for N in range(1, nmax + 1):
        g = arena_gb(N, fmt, ctx, M, kvb, tp, cp)
        if g <= usable_gb:
            return N, g
    return None

if __name__ == "__main__":
    K = 1024
    print("MiniMax-M3 minimal-node sizing (TP mandatory; EP imbalance = busiest rank owns ceil(128/N))")
    print("=" * 78)
    print(f"total weights: bf16 ~{arena_gb(1,'bf16',256):.0f} GB | fp8 ~{arena_gb(1,'fp8',256):.0f} GB "
          f"(1-rank arena, no TP benefit)")
    print(f"calibration: arena_gb(96,'bf16',8192) = {arena_gb(96,'bf16',8192):.1f} GB/rank "
          f"(m3.md anchor ~18-23)")
    print(f"             arena_gb(24,'fp8',256)   = {arena_gb(24,'fp8',256):.1f} GB/rank")
    print()
    print(f"MIN NODES (usable {USABLE_GB} GB/node, M3_TP=1):")
    print(f"  {'fmt':<5} {'ctx':>6} {'M':>3} {'KV':>5} {'min N':>6} {'arena@minN':>11} {'busiest exp':>12}")
    for fmt in ('fp8', 'bf16'):
        for ctx, M, kvb in [(256,1,2),(8*K,1,2),(32*K,1,2),(128*K,1,2),(8*K,8,2)]:
            r = min_nodes(fmt, ctx, M, kvb)
            if r is None:
                print(f"  {fmt:<5} {ctx//K if ctx>=K else ctx}{'k' if ctx>=K else '':>1} {M:>3} "
                      f"{'bf16':>5} {'>256':>6}"); continue
            N, g = r
            ctxs = f"{ctx//K}k" if ctx >= K else str(ctx)
            print(f"  {fmt:<5} {ctxs:>6} {M:>3} {('int4' if kvb<2 else 'bf16'):>5} {N:>6} "
                  f"{g:>9.1f}G {_ceil(N_EXP,N):>10} exp")
    print()
    print("recommended (arena numbers are operating estimates = weight arena + ~4.5 GB page-cache/THP):")
    print(f"  fp8  -> floor {min_nodes('fp8',256)[0]}n | MINIMAL 24n short-ctx gen "
          f"({arena_gb(24,'fp8',256):.1f} GB, TIGHT - validate MemFree) | 32n comfortable/batched/validated")
    print(f"  bf16 -> floor {min_nodes('bf16',256)[0]}n | run 48n (4x4x3:torus, busiest {_ceil(N_EXP,48)} exp, "
          f"{arena_gb(48,'bf16',256):.1f} GB) - smallest validated bf16")
    print("  NOTE: 24n fp8 fits only short ctx (KV grows the floor: 8k->26n, 32k->32n). Keep M3_MAXPOS small,")
    print("        or M3_CP=1 (shard KV) + M3_INT4_KV=1 (halve KV) to hold ctx without adding nodes.")
    print("  WARNING: an OOM SIGKILL degrades PMIx and costs the whole alloc - do not run tight w/o MemFree.")
