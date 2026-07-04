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
N_LAYERS      = 43
KVC           = 576       # MLA latent KV per token per layer: kv_lora 512 + qk_rope 64
USABLE_GB     = 27.0      # 32 HBM - 2 reserve - 2 activations - ~1 mHC(hc_mult=4) residual scratch

def weights_gb_per_node(N):
    return REPLICATED_GB + EXPERTS_GB / N

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

if __name__ == "__main__":
    K = 1024
    print("DeepSeek-V4-Flash (ds4f) minimal-node sizing")
    print("=" * 60)
    print(f"weights: {REPLICATED_GB:.2f} GB replicated + {EXPERTS_GB:.2f} GB experts (EP) "
          f"= {REPLICATED_GB + EXPERTS_GB:.1f} GB total")
    print(f"weight floor (KV=0): >= {weight_floor()} nodes   usable = {USABLE_GB} GB/node")
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
    print("guidance:")
    print("  decode-opt  -> run AT the floor (~11n); NUMA (DS4F_NUMA=1) ~1.40x -> target ~14 tok/s")
    print("  prefill-opt -> run ABOVE the floor (extra ranks = attention/expert-GEMM parallelism)")
    print("  serve       -> pinned by persistent KV; DS4F_CP for ctx>=512k, DS4F_INT8_KV cuts ~1/3 nodes")
