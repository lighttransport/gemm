#!/usr/bin/env python3
"""
ds4f_sim.py — minimal-node sizing for DeepSeek-V4-Flash (ds4f) on A64FX / Fugaku.

Memory model, CALIBRATED to the measured 22.18 GB/node arena at 11 EP nodes (ds4f.md); this
model's manifest scan gives 22.71 GB/node @11n, within ~2%:

    per_node(N, ctx, M, kvb) = REPLICATED_GB              # dense 7.96 + embed/head 1.06 = 9.02
                             + EXPERTS_GB / N              # fp4 (MXFP4) experts, EP-sharded (e % N)
                             + KV_total(ctx, M, kvb) / N   # MLA latent KV, context-parallel (DS4F_CP)

    KV_total = N_LAYERS * ctx * KVC * kvb * M / 1e9        # KVC = kv_lora 512 + qk_rope 64 = 576
    min_nodes = fewest N with per_node <= USABLE_GB        # 28 usable - ~1 for mHC 4x-stream scratch

Pattern guidance (baked into the report):
  decode-optimized  -> run AT the node floor (fewest ranks = most efficient; comm ~flat in N).
  prefill-optimized -> run ABOVE the floor (extra ranks buy attention/expert-GEMM parallelism).
  prefill+decode    -> node count PINNED by the persistent full-ctx KV; use DS4F_CP (ctx>=512k) and
                       DS4F_INT8_KV (kvb=1) to cut ~1/3 of the nodes.

Run:  python3 ds4f_sim.py         # prints the sizing table
Import: from ds4f_sim import min_nodes, per_node   # min_nodes(ctx, M, kvb) -> (N, weights_gb, kv_gb)
"""
import math

# ---- DeepSeek-V4-Flash weight split (from the ~/models/ds4f safetensors manifest scan) ----
REPLICATED_GB = 9.02      # dense (attn/shared/router/indexer/mHC) 7.96 + embed+head 1.06, per node
EXPERTS_GB    = 150.59    # 256 routed experts, fp4 (MXFP4), EP-sharded across N nodes
N_EXPERTS     = 256
N_LAYERS      = 43
KVC           = 576       # MLA latent KV per token per layer: kv_lora 512 + qk_rope 64
USABLE_GB     = 27.0      # 32 HBM - 2 reserve - 2 activations - ~1 mHC(hc_mult=4) residual scratch
# TP-sharded dense (all knobs built + gated). Two tiers matter for minimal-node runs:
#   Q8-fast  : DS4F_TP_HEAD+TP_EMBED only (shard lm_head+embed = 1.06 GB; speed-NEUTRAL, compatible
#              with the fast Q8_DENSE decode). The 7.96 GB attn/shared/router stays replicated.
#   FP8-full : + TP_ATTN/SHARED/OPROJ/WOB (needs sliceable FP8 dense, NOT Q8 -> ~2x slower decode);
#              drops replicated dense to ~0.7 GB/node (measured: 11n full-TP arena 14.36 = 13.69
#              experts + 0.67 dense-shard). Modeled as SHARDABLE_GB/N + RESIDUAL_GB.
EMBED_HEAD_GB  = 1.06     # lm_head + embed (the TP_HEAD/TP_EMBED shard, speed-neutral)
SHARDABLE_GB   = 7.4      # attn/shared/o-proj dense that FP8+full-TP shards (0.67*11)
RESIDUAL_GB    = 1.5      # norms/router-gate/mHC scratch that stays replicated even under full TP

# ---- DeepSeek-V4 BASE (~/models/ds4fbase): identical graph, FP8-e4m3 experts instead of MXFP4 ----
# 24 MiB/expert vs Flash's 12.75 -> 1.88x the expert memory; dense/embed/head are unchanged.
# Base therefore CANNOT run dense-replicated at any interactive node count and REQUIRES fp8full TP.
# MEASURED @12 EP (run_ds4fbase_12n.sh, real weights): arena 25.34 GB on the fullest (22-expert)
# rank, 24.15 GB on a 21-expert rank -- i.e. 23.80 GB experts + ~1.5 GB TP'd dense. The model below
# reproduces that to ~1%.
EXPERTS_GB_BASE = 276.9   # 43 layers * 256 experts * 24 MiB (FP8 e4m3 + 128x128 E8M0 scale)

import math as _math
def fullest_experts_gb(N, experts_gb=None):
    """Ragged EP shard e%%N: the fullest node holds ceil(N_EXPERTS/N) experts and bounds memory."""
    eg = EXPERTS_GB if experts_gb is None else experts_gb
    return eg * _math.ceil(N_EXPERTS / N) / N_EXPERTS

def weights_gb_per_node(N, tp="plain", experts_gb=None):
    """Per-node weight memory (fullest node). tp in {plain, q8fast, fp8full}.
    Pass experts_gb=EXPERTS_GB_BASE for the base (fp8-expert) model."""
    exp = fullest_experts_gb(N, experts_gb)
    if tp == "q8fast":    # TP_HEAD/EMBED only; 7.96 GB attn/shared stays replicated
        return (REPLICATED_GB - EMBED_HEAD_GB) + EMBED_HEAD_GB / N + exp
    if tp == "fp8full":   # full TP: shardable/N + small replicated residual
        return RESIDUAL_GB + SHARDABLE_GB / N + exp
    return REPLICATED_GB + exp

def weight_floor_tiers(usable_gb=USABLE_GB, nmax=64):
    """Fewest EP nodes for weights alone (KV=0) under each TP tier."""
    out = {}
    for tp in ("plain", "q8fast", "fp8full"):
        out[tp] = next((N for N in range(1, nmax + 1)
                        if weights_gb_per_node(N, tp) <= usable_gb), None)
    return out

def kv_gb_total(ctx, M, kvb=2):
    """Persistent MLA latent KV for M concurrent streams at context length ctx (kvb=2 bf16, 1 int8)."""
    return N_LAYERS * ctx * KVC * kvb * M / 1e9

def kv_gb_per_node(N, ctx, M, kvb=2, cp=True):
    tot = kv_gb_total(ctx, M, kvb)
    return tot / N if cp else tot

def per_node(N, ctx, M, kvb=2, cp=True):
    return weights_gb_per_node(N) + kv_gb_per_node(N, ctx, M, kvb, cp)

def min_nodes(ctx, M=1, kvb=2, usable_gb=USABLE_GB, cp=True, nmax=1024):
    """Fewest EP nodes N such that per_node(N) <= usable_gb. Returns (N, weights_gb, kv_gb) or None."""
    for N in range(1, nmax + 1):
        w = weights_gb_per_node(N)
        kv = kv_gb_per_node(N, ctx, M, kvb, cp)
        if w + kv <= usable_gb:
            return N, w, kv
    return None

def weight_floor(usable_gb=USABLE_GB):
    """Fewest nodes for the weights alone (KV=0)."""
    return math.ceil(EXPERTS_GB / (usable_gb - REPLICATED_GB))

def max_ctx(N, M=1, kvb=2, cp=False, usable_gb=USABLE_GB):
    """Longest context (prompt+gen) that fits on N nodes. cp=False = KV replicated (ds4f attn is
    replicated), which is the case for the interactive agentic config at 11n."""
    budget = usable_gb - weights_gb_per_node(N)
    if budget <= 0:
        return 0
    per_tok = N_LAYERS * KVC * kvb * M / 1e9          # GB per context token (all layers, M streams)
    denom = per_tok / N if cp else per_tok
    return int(budget / denom)

if __name__ == "__main__":
    K = 1024
    print("DeepSeek-V4-Flash (ds4f) minimal-node sizing")
    print("=" * 60)
    print(f"weights: {REPLICATED_GB:.2f} GB replicated + {EXPERTS_GB:.2f} GB experts (EP) "
          f"= {REPLICATED_GB + EXPERTS_GB:.1f} GB total")
    tiers = weight_floor_tiers()
    print(f"weight floor (KV=0, ragged e%N): usable = {USABLE_GB} GB/node")
    print(f"  plain (dense replicated)     >= {tiers['plain']} nodes")
    print(f"  q8fast (TP_HEAD/EMBED, fast)  >= {tiers['q8fast']} nodes  <- pjsub --preset decode; 8 is edge")
    print(f"  fp8full (full TP, ~2x slower) >= {tiers['fp8full']} nodes  <- absolute minimum (6 does NOT fit)")
    print(f"  per-node W(N,tp):  " + "  ".join(
        f"N{N}:{weights_gb_per_node(N,'q8fast'):.1f}/{weights_gb_per_node(N,'fp8full'):.1f}"
        for N in (6, 7, 8, 9)) + "  (q8fast/fp8full GB)")
    print(f"calibration: per_node(11, 4k, M=8) = {per_node(11, 4*K, 8):.2f} GB "
          f"(measured arena @11n = 22.18)")
    print()
    print("MIN NODES (weights EP-sharded + MLA KV context-parallel):")
    print(f"  {'pattern':<10} {'ctx':>6} {'M':>3} {'KV-bytes':>9} {'KV/node':>9} {'min N':>6}")
    rows = [
        ("decode",  4*K,      8, 2), ("decode", 4*K,      8, 1),
        ("serve",   128*K,    1, 2), ("serve",  512*K,    1, 2),
        ("serve",   1024*K,   1, 2), ("serve",  1024*K,   1, 1),
        ("serve",   512*K,    8, 2), ("serve",  512*K,    8, 1),
        ("serve",   1024*K,   8, 2), ("serve",  1024*K,   8, 1),
    ]
    for pat, ctx, M, kvb in rows:
        r = min_nodes(ctx, M, kvb)
        if r is None:
            print(f"  {pat:<10} {ctx//K:>5}k {M:>3} {('int8' if kvb==1 else 'bf16'):>9} "
                  f"{'--':>9} {'>1024':>6}")
            continue
        N, w, kv = r
        print(f"  {pat:<10} {ctx//K:>5}k {M:>3} {('int8' if kvb==1 else 'bf16'):>9} "
              f"{kv:>8.1f}G {N:>6}")
    print()
    print("DeepSeek-V4 BASE (ds4fbase; FP8-e4m3 experts, 1.88x Flash's expert memory):")
    print(f"  weights: {EXPERTS_GB_BASE:.1f} GB experts (EP) + the same ~9.0 GB dense/embed/head")
    print(f"  {'N':>3}  {'plain':>8}  {'fp8full':>8}   (GB/node, fullest rank; usable ~{USABLE_GB})")
    for N in (10, 11, 12, 13, 16):
        pl = weights_gb_per_node(N, "plain",   EXPERTS_GB_BASE)
        tp = weights_gb_per_node(N, "fp8full", EXPERTS_GB_BASE)
        mark = "  <- 12n interactive: measured arena 25.34 GB (fullest rank)" if N == 12 else ""
        print(f"  {N:>3}  {pl:>7.1f}{'!' if pl > USABLE_GB else ' '}  {tp:>7.1f}{'!' if tp > USABLE_GB else ' '} {mark}")
    print("  '!' = over the usable budget. plain NEVER fits at interactive node counts ->")
    print("  the fp8full TP stack (TP_ATTN/OPROJ/WOB/SHARED/HEAD/EMBED) is MANDATORY for base.")
    print()
    print("guidance:")
    print("  decode-opt  -> run AT the floor (~11n); NUMA (DS4F_NUMA=1) ~1.40x -> target ~14 tok/s")
    print("  prefill-opt -> run ABOVE the floor (extra ranks = attention/expert-GEMM parallelism)")
    print("  serve       -> pinned by persistent KV; DS4F_CP for ctx>=512k, DS4F_INT8_KV cuts ~1/3 nodes")
    print()
    print("AGENTIC CODING (12-node interactive alloc = 11 EP; --preset decode, NUMA on):")
    print(f"  context ceiling @11 EP (prompt+gen, KV replicated). The MLA-KV model below OMITS the")
    print(f"  Tier-B2 indexer/compressed-key caches + warm-fill scratch, so the EMPIRICAL ceiling is")
    print(f"  much lower (ops log: OOM at ctx~32k -> safe ~16k). An OOM kills the whole alloc (gotcha #6).")
    print(f"    bf16 KV (KVBITS=16): model {max_ctx(11, kvb=2)//1024:>4}k  | EMPIRICAL safe ~16k (sweet spot ~10k)")
    print(f"    int8 KV (KVBITS=8) : model {max_ctx(11, kvb=1)//1024:>4}k  | + DS4F_INT8_CMP to extend (unproven)")
    print(f"    DS4F_CP=1 (sharded): model {max_ctx(11, kvb=2, cp=True)//1024:>4}k | validated long-ctx path -> 512k-1M")
    print(f"  launcher: run_ds4f_agentic_11n.sh (PROMPT_FILE=task.txt [KVBITS=8]); long ctx: run_ds4f_longctx_11n.sh")
