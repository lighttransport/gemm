#!/usr/bin/env python3
"""
m3_decode_sim.py — MiniMax-M3 batched-decode THROUGHPUT model (companion to m3_sim.py, which does
node/memory SIZING). Calibrated to the measured mstream sweep (probes P1/P1b/P2, 48n synth bf16,
TP on, 12 threads, TP_AR_BF16=1). Answers: for a given deployment (nodes, ctx, KV format), what
batch size M fits, and what aggregate decode tok/s does it deliver?

MEASURED (48n, full-60L synth bf16, bf16-AR) — aggregate tok/s vs M:
    M     1     8     16     32     48      64
    tok/s 3.13  13.19 14.29  16.70  17.75   17.35   <- PEAK at M=48 (=node count); M=64 regresses
Key facts encoded:
  - Aggregate throughput PEAKS at M~=48 then declines (per-stream overhead beats comm amortization;
    single-node compute ceiling ~17.6, probe P2). So M>48 is never worth it.
  - bf16-AR (TP_AR_BF16=1) is a free +2-3%, lockstep-preserving (baked into the calibration).
  - tok/s is ~flat in node count (m3.md) -> this curve is treated as N-independent; nodes/ctx/KV only
    gate the *max M that fits* (per-stream KV x M), via m3_sim's arena.
  - REAL-WEIGHT CORRECTION (probe P3, 48n real bf16): the curve is SYNTHETIC and OPTIMISTIC at high M.
    Measured real-weight M=48 = 15.25 tok/s (vs synth 17.75, -14%) — real-router expert-load imbalance
    raises comm 27%->37%. Low M (~<16) ~= synth (comm small). Use decode_tok_s_real() / REAL_FACTOR for a
    deployment estimate; the synthetic decode_tok_s() is the upper bound. (bf16-AR confirmed lossless on
    real weights: coherent "Paris" gen.)
  - int4-KV IS now wired into batched decode (Lever 1, m3_impl.h: m3_forward_batch_decode packs int4 K/V
    into ms->k_q4/v_q4 + m3_q4_dot/axpy). Validated (job 49441928): engages (out0 differs from bf16), NaN=0,
    tok/s unchanged (unpack is free) → ~3.9x smaller per-stream KV lets high-M fit at long ctx. **Real-weight
    QUALITY gate still pending** (does int4-KV keep coherent gen?) — treat int4 rows below as memory-enabled
    but quality-unverified for production.

Run:    python3 m3_decode_sim.py
Import: from m3_decode_sim import decode_tok_s, best_decode
"""
import os, sys, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_sim import arena_gb, OVERHEAD_GB, _ceil        # memory model (weights + KV, EP+TP)

# Memory-fit uses the RAW weight+KV arena (m3_sim's arena minus its ~4.5 GB page-cache OVERHEAD).
# Rationale: the raw arena matches the MEASURED arena_used (P1b: 21.46 GB weights @48n, and M=48 KV at
# maxpos=512 fit 32 GB; real 96n run = 13.23 GB) — m3_sim's overhead was calibrated to a projected
# 18-23 band the real runs never hit, so it over-caps batching. We fit raw arena <= FIT_USABLE_GB and
# leave the rest of the 32 GB node as physical headroom for page cache + scratch.
FIT_USABLE_GB = 28.0

THROUGHPUT_PEAK_M = 48                                  # measured peak; M>48 regresses -> never exceed
# measured aggregate tok/s anchors (48n, bf16-AR). M=1 from m3.md single-stream; [8..64] = P1/P1b.
_ANCHOR = [(1, 3.13), (8, 13.19), (16, 14.29), (32, 16.70), (48, 17.75), (64, 17.35)]
_MX = [a[0] for a in _ANCHOR]; _MY = [a[1] for a in _ANCHOR]

def decode_tok_s(M):
    """Aggregate decode tok/s at batch M (piecewise-linear over the measured anchors; calibrated @48n
    bf16-AR, ~flat in N). M>64 held at the M=64 value (regressing region — don't run there)."""
    if M <= _MX[0]:  return _MY[0] * M / _MX[0]         # M<1 shouldn't happen; linear to origin
    if M >= _MX[-1]: return _MY[-1]                      # >=64: hold (regresses, not recommended)
    i = bisect.bisect_right(_MX, M) - 1
    x0, y0, x1, y1 = _MX[i], _MY[i], _MX[i+1], _MY[i+1]
    return y0 + (y1 - y0) * (M - x0) / (x1 - x0)

def ms_per_step(M):
    """Per-decode-step latency (ms) for a batch of M streams = 1000*M/agg_tok_s."""
    t = decode_tok_s(M);  return 1000.0 * M / t if t > 0 else float('inf')

# Real-weight correction (P3, 48n real bf16): measured M=48 = 15.25 vs synth 17.75 (real-router comm
# imbalance). ~1.0 up to M=16 (comm small), 0.859 at M=48, linear between.
REAL_M48 = 15.25
def real_factor(M):
    f48 = REAL_M48 / decode_tok_s(48)
    if M <= 16: return 1.0
    if M >= 48: return f48
    return 1.0 + (f48 - 1.0) * (M - 16) / 32.0
def decode_tok_s_real(M):
    """Deployment (real-weight) throughput estimate = synth curve x the P3 correction (synth is upper bound)."""
    return decode_tok_s(M) * real_factor(M)

def raw_arena_gb(N, fmt, maxpos, M, kvb):
    """Weight + M-stream KV arena, WITHOUT m3_sim's page-cache overhead (matches measured arena_used)."""
    return arena_gb(N, fmt, maxpos, M, kvb) - OVERHEAD_GB

def max_M_fits(N, fmt='bf16', maxpos=512, kvb=2, usable_gb=FIT_USABLE_GB, mcap=256):
    """Largest batch M whose RAW per-node arena (weights + M-stream KV) fits usable_gb."""
    best = 0
    for M in range(1, mcap + 1):
        if raw_arena_gb(N, fmt, maxpos, M, kvb) <= usable_gb: best = M
        else: break
    return best

def best_decode(N, fmt='bf16', maxpos=512, kvb=2, usable_gb=FIT_USABLE_GB):
    """Best deployable (M, agg tok/s) at N nodes: throughput peaks at M<=48, capped by what fits.
    Returns (M_star, tok_s, max_M_fits, memory_capped?)."""
    mmax = max_M_fits(N, fmt, maxpos, kvb, usable_gb)
    m_star = min(THROUGHPUT_PEAK_M, mmax)
    capped = mmax < THROUGHPUT_PEAK_M
    return m_star, decode_tok_s(m_star), mmax, capped

if __name__ == "__main__":
    K = 1024
    print("MiniMax-M3 batched-decode throughput model (calibrated: 48n synth bf16, TP_AR_BF16=1)")
    print("=" * 82)
    print("measured aggregate tok/s vs M (bf16-AR, synth):  M=8 13.19 | 16 14.29 | 32 16.70 | 48 17.75 | 64 17.35")
    print(f"PEAK at M={THROUGHPUT_PEAK_M} (=node count); M=64 regresses. Best config: M3_MSTREAM=48 TP_AR_BF16=1.")
    print(f"REAL-WEIGHT (P3): M=48 = {REAL_M48} tok/s (synth 17.75 is ~14% optimistic; real-router comm imbalance).\n")
    print("curve:  M     8      16      24      32      48      64")
    print("  tok/s " + "  ".join(f"{decode_tok_s(m):5.1f}" for m in (8,16,24,32,48,64)))
    print("  ms/st " + "  ".join(f"{ms_per_step(m):5.0f}" for m in (8,16,24,32,48,64)))
    print()
    print("DEPLOYABLE best (M capped by per-stream KV x M fitting the arena):")
    print(f"  {'nodes':>5} {'fmt':>5} {'maxpos':>7} {'KV':>5} {'maxM_fit':>9} {'M*':>4} {'synth t/s':>10} {'real~t/s':>9} {'note':>13}")
    # int4-KV IS wired into batched decode now (Lever 1) -> long-ctx rows compare bf16 vs int4 (quality TBD).
    rows = [
        (24,'fp8', 512, 2), (24,'fp8', 4096, 2), (24,'fp8', 4096, 0.5),
        (48,'bf16', 512, 2), (48,'bf16', 4096, 2), (48,'bf16', 4096, 0.5),
        (96,'bf16', 4096, 2), (96,'bf16', 4096, 0.5),
    ]
    for N, fmt, mp, kvb in rows:
        m_star, tk, mmax, capped = best_decode(N, fmt, mp, kvb)
        kvl = 'int4' if kvb < 1 else 'bf16'
        note = 'KV-capped' if capped else 'peak M=48'
        print(f"  {N:>5} {fmt:>5} {mp:>7} {kvl:>5} {mmax:>9} {m_star:>4} {tk:>9.1f} {decode_tok_s_real(m_star):>8.1f}  {note:>13}")
    print()
    print("reads: 'M*' = batch to run (min of the M=48 peak and what fits). 'real~t/s' = synth x P3 correction.")
    print("'KV-capped' = can't fit M=48 streams' KV at this ctx -> raise nodes, drop maxpos, or int4-KV.")
    print("int4-KV IS wired into batched decode (Lever 1, validated); its long-ctx rows need a real-weight")
    print("quality gate before production (int4 changes attention output; synth showed engage+NaN=0+free).")
