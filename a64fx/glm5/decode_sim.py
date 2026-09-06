#!/usr/bin/env python3
"""
GLM-5.2 A64FX decode-throughput estimator + theoretical upper bound  (no job submission).

v2: architecture-derived. Computes the active bytes/token from the real GLM-5.2 dims, the
BANDWIDTH-BOUND theoretical ceiling (the hard upper bound on decode speed), and a decomposed,
calibrated comm-bound prediction. The gap between the two = the all-reduce overhead we actually
pay. Use to plan optimization and to know how much headroom exists.

CALIBRATION ANCHOR (measured int8): 96 nodes, single-stream (M=1) decode = 0.25 tok/s.

Two ceilings reported:
  * THEORETICAL UPPER BOUND = bandwidth limit: must read the active weights (+KV) at least once
    per forward; tok/s <= M / (active_bytes_per_node / per_node_HBM_BW). Comm-free idealization.
  * CALIBRATED PREDICTION  = upper bound + the (calibrated) all-reduce latency, which dominates.
"""
import math

# ---------- GLM-5.2 architecture ----------
H, N_LAYERS, N_MOE, N_DENSE = 6144, 78, 75, 3
N_HEADS, QK_NOPE, QK_ROPE, V_HEAD = 64, 192, 64, 256
QK_HEAD = QK_NOPE + QK_ROPE                 # 256
Q_LORA, KV_LORA = 2048, 512
KVC = KV_LORA + QK_ROPE                     # 576 (latent KV cache dim/pos/layer)
MOE_INTER, DENSE_INTER = 2048, 12288
N_EXP, N_ACT, VOCAB = 256, 8, 154880

# ---------- platform ----------
# MEASURED 2026-07-03 (fapp + streaming kernel bench, node c33-7214c, 48t): NUMA-local (numactl
# --interleave=all + OMP_PROC_BIND/PLACES, now the landed default) node read-BW ceiling = 739 GB/s;
# the w8a16 decode matvec ACHIEVES ~336 GB/s effective (M=1, 40960x6144). ⚠ default CMG0 prepage
# (pre-NUMA-fix) was capped at ~100 GB/s (single-CMG cross-CMG limit) — the old undiagnosed 48t
# regression. See CALIBRATION.md "MEASURED A64FX kernel perf".
BW_NODE = 336e9            # per-node EFFECTIVE w8a16 matvec HBM BW, NUMA-local (was 300; CMG0-default ~100e9; load ceiling 739e9)

# ---------- MEASURED compute ceilings (fapp region-profile, 48t, L2-resident microbench = COMPUTE
# ceiling; real decode is BW-bound above, real prefill is compute-bound so these apply). Gop/s
# (=2*macs). fapp FP-peak% counts FP ops only -> the SDOT kernels show low FP% (integer svdot pipe)
# despite 2-4x the throughput; use these wall-clock Gop/s. Imported by prefill_sim.py. ----------
DECODE_MV_GOPS = {'w8a16': 449.0, 'int16': 431.0}          # M=1 matvec (int16 decode is a net e2e LOSS: BW-bound)
GEMM_GOPS = {  # prefill GEMM, single-node 48t Gop/s by precision x M (group-128, measured kernel bench)
    'w8a16': {8:487, 16:692, 32:904, 64:1124},             # int8 w8a16 bf16-tile FMA (26% of f32 FMA peak)
    'int16': {8:1481, 16:1989, 32:2160, 64:2276},          # int16 svdot_s64, near-lossless (rms 1.5e-5), ~2x
    'int8':  {8:1869, 16:2600, 32:3060, 64:3175},           # int8 svdot_s32 register-blocked, lossy (rms 4e-3), ~3x
    'bf16':  {8:487, 16:692, 32:904, 64:1124},              # bf16-widen->f32 (fp16 native NOT worth it: 1.23x/0.51x)
}
AR_M1_96 = 0.25            # STALE legacy anchor (kept for the historical decomposition); see REAL_DECODE below.

# ---------- REAL decode anchors (MEASURED, jobs 49419683/684, int8 full model, short ctx, 2026-07-03) ----------
# Full 78L int8, 8 slots, maxpos 2048, per-slot (bd=0) vs batched (bd=1, M<=4). tok/s/slot + comm fraction:
REAL_DECODE = {  # N: (bd0_tok_s_per_slot, bd1_tok_s_per_slot, comm_frac_bd0)
    32: (1.72, 2.94, 0.370),
    48: (1.66, 2.72, 0.405),   # attn 26.5 ms/tok == 32n: throughput flat 32->48n
    96: (1.79, 2.90, 0.318),   # attn jumps to 48 ms/tok (distinct worse regime)
}
# Headlines: (1) real single-stream int8 decode ~1.75 tok/s/slot -- the old 0.25 anchor was ~7x pessimistic.
# (2) 32n ~= 96n throughput -> 32n is ~3x more NODE-EFFICIENT for short-ctx int8 serving. (3) comm ~30-37%
# (~flat in N; NOT lower at fewer ranks). (4) batched ~1.6-1.7x. (5) compute ~65%, attn-dominated & NON-scaling.
REAL_DECODE_COMPUTE_MS = {  # bd=0 ms/tok by stage: (32n, 96n) -- attn is the top target (grows with N)
    'attn':(26.4,48.2), 'qkv':(20.2,26.8), 'shared':(20.0,20.9), 'router':(14.2,15.1),
    'o_proj':(9.7,10.7), 'experts':(9.1,3.8), 'dense':(2.4,1.1), 'head':(0.7,0.2),
}

# ---------- weight params (then bytes by precision) ----------
def attn_params():
    return (Q_LORA*H                      # q_a
          + N_HEADS*QK_HEAD*Q_LORA        # q_b
          + KVC*H                         # kv_a
          + N_HEADS*(QK_NOPE+V_HEAD)*KV_LORA  # kv_b
          + H*N_HEADS*V_HEAD)             # o_proj
ATTN_P   = attn_params()*N_LAYERS         # all layers
ROUTER_P = N_EXP*H*N_MOE                  # bf16, replicated
SHARED_P = 3*MOE_INTER*H*N_MOE            # int8, TP-sharded
EXPERT_P = 3*MOE_INTER*H                  # one routed expert, int8
DENSE_P  = 3*DENSE_INTER*H*N_DENSE        # int8, TP-sharded
HEAD_P   = VOCAB*H                        # bf16, vocab-sharded

def distinct_experts(M):                  # E[#distinct of N_EXP hit when M tokens pick N_ACT each]
    return N_EXP*(1-(1-N_ACT/N_EXP)**M)

def active_bytes_per_node(N, M, ctx, prec='int8', shard_attn=True, attn_ways=None):
    """Bytes read by ONE node in ONE forward serving M streams at context length `ctx`.
    attn_ways: how many ways attention is sharded (None -> N if shard_attn else 1;
    k<N models a TP SUBGROUP of k ranks holding 1/k of the heads each)."""
    wb = 1 if prec=='int8' else 2
    if attn_ways is None: attn_ways = N if shard_attn else 1
    attn = ATTN_P*wb/attn_ways                            # sharded heads -> /ways; replicated -> full
    router = ROUTER_P*2                                   # bf16, replicated
    shared = SHARED_P*wb/N                                # TP-sharded
    dense  = DENSE_P*wb/N
    head   = HEAD_P*2/N                                   # bf16, vocab-sharded
    experts = distinct_experts(M)/N * EXPERT_P*wb * N_MOE # distinct active experts this node owns
    kv = M * N_LAYERS * ctx * KVC * 2                     # per-stream latent KV read (bf16), CP off
    return attn + router + shared + dense + head + experts + kv

def ub_tok_s(N, M, ctx, prec='int8', shard_attn=True):    # THEORETICAL bandwidth ceiling (comm-free)
    return M / (active_bytes_per_node(N,M,ctx,prec,shard_attn)/BW_NODE)

# ---------- uTofu all-reduce cost model (MEASURED on Fugaku, ARPROBE ladder 2026-07-03) ----------
# Real tp_allreduce anatomy, robust=1, M=1, [1,H] fp32 payload (GLM5_AR_PROBE, GLM5_PREFILL_GROUPS=1
# so the AR spans all N ranks). Recursive-doubling, ceil(log2 N) rounds, each a uTofu Put + robust
# completion. Model: ar = UTOFU_INIT + rounds*UTOFU_ROUND + bytes*UTOFU_BW.
#   N:    8      16     32     96          <- measured us_per_ar (robust=1):
#   AR:   72.4   86.2   139.6  141.1  us   -> flat ~24-26 us/round, init ~0. robust 0/1/2 within ~30%.
# >>> This kills the fictional 26 ms/AR robustness tax (tight-loop AR is ~0.14 ms), BUT is NOT the
#     in-decode cost -- see the EFFECTIVE_AR_FACTOR correction below and MEASURED CALIBRATION report.
UTOFU_INIT   = 4.0e-6           # fixed per-AR setup (s); measured init ~0 -> small floor
UTOFU_ROUND  = 26.0e-6         # MEASURED tight-loop per-round cost (s) ~26 us (was fictional 3.4 ms)
UTOFU_BW     = 1.0/8e9         # s/byte once in flight (~8 GB/s effective per-link AR bandwidth)
# >>> CORRECTION (real-weight cbatch, job 49419532, bf16 12L 12n, 2026-07-03): the ARPROBE tight loop
#     UNDER-measures the IN-DECODE all-reduce. Real decode: 6.22 tok/s/slot, comm 41.5%, 3224 AR calls
#     over 5.1 s => ~0.66 ms EFFECTIVE per-AR at 12n, ~8x the tight-loop probe (~0.08 ms @12n). The gap
#     is straggler-synchronization (ranks arrive at the AR at different times after uneven expert
#     compute) + cache-cold/MRQ-under-load interleave -- NOT the robust completion path. So decode is
#     ~40% comm at 12n (rising with N), NOT the "0.5% / 99% non-comm" the tight-loop probe alone implied.
EFFECTIVE_AR_FACTOR = 8.0      # MEASURED in-decode per-AR / tight-loop probe per-AR (straggler+interleave)
def n_allreduce(shard_attn): return N_MOE + (N_LAYERS if shard_attn else 0)
SHARD_ATTN = True              # eval config: attention heads sharded -> 2 AR/MoE layer
_bw96_1 = active_bytes_per_node(96,1,128)/BW_NODE
_ar96_probe = UTOFU_INIT + math.ceil(math.log2(96))*UTOFU_ROUND    # tight-loop per-AR @96n (~0.19 ms)
_ar96 = _ar96_probe*EFFECTIVE_AR_FACTOR                            # EFFECTIVE in-decode per-AR @96n (~1.5 ms)
_comm96_1 = n_allreduce(SHARD_ATTN)*_ar96                          # comm s at (96,M=1) from the effective AR
_resid96_1 = 1.0/AR_M1_96 - _bw96_1 - _comm96_1                    # residual: compute/HBM/overhead/contention
#     UNEXPLAINED by comm -> decode is NOT comm-bound. That residual is the real target; a real-weight
#     decode PROFILE job (comm% breakdown) must attribute it (compute vs HBM vs per-token overhead vs contention).

# ---------- qlair wire-floor calibration (MEASURED, contention-free) ----------
# tp_ar_8rank.elf under `qlair --native --ranks {2,4,8}` x TP_AR_ROBUST {0,1,2}, CNT=6144 fp32.
# qlair models the Tofu put (1400ns + bytes/6.3GB/s) + the real SVE reduction, but has NO
# multi-node topology, incast, OS jitter, or MRQ-drain-under-contention. So it gives the WIRE
# FLOOR and the *shape*, not the cluster latency. Fit (robust=1, production baseline):
QLAIR_WIRE_INIT_MS  = 0.105          # per-AR fixed wire cost (setup + first round overheads)
QLAIR_WIRE_ROUND_MS = 0.0677         # per-round wire cost; latency linear in ceil(log2 N) -> CONFIRMS the model
QLAIR_ROBUST_TAX    = 1.15           # robust=1 / robust=0 (measured 1.13-1.17): the correctness tax
QLAIR_LEAN_RECOVER  = 0.89           # robust=2 / robust=1 (measured 0.87-0.91): lean AR recovers most of the tax, bit-exact
def qlair_wire_ar_ms(N):             # qlair --native per-AR estimate at N ranks
    return QLAIR_WIRE_INIT_MS + math.ceil(math.log2(N))*QLAIR_WIRE_ROUND_MS
def qlair_vs_real(n_ref=96):         # qlair wire / MEASURED tight-loop per-AR probe
    return qlair_wire_ar_ms(n_ref) / (_ar96_probe*1e3)
# POST-JOB RECONCILIATION (ARPROBE ladder measured the REAL AR):
#  * qlair --native OVER-estimated the wire ~2.6x (0.58 ms est vs 0.14 ms real @96n) -- pthread-barrier
#    sim overhead, NOT a cluster term. The earlier "45x cluster multiplier" was an artifact of the
#    fictional 26 ms/AR; there is NO 45x tax. Use the MEASURED UTOFU_ROUND above, not the qlair floor.
#  * qlair still got the SHAPE right (linear in ceil(log2 N)) and robust=2 bit-exactness -- those hold.
#  * robust 0/1/2 real per-AR are within ~30% (no 3.4 ms robustness tax) -> the lean-AR lever is
#    near-worthless for M=1; comm is ~0.5% of the token. Batching's win (measured 2.6x @8n synthetic)
#    comes from amortizing per-FORWARD compute/overhead, not AR count.

def recalibrate(ar_ms=None, round_ms=None, init_ms=None, n_ref=96):
    """Re-anchor the comm constants from a MEASUREMENT (qlair-8rank or a job):
    either the per-round cost directly (round_ms) or a full per-AR latency (ar_ms at n_ref ranks).
    Call before using pred_tok_s/m1_tput to project with the measured collective."""
    global UTOFU_ROUND, UTOFU_INIT
    if init_ms  is not None: UTOFU_INIT  = init_ms*1e-3
    if round_ms is not None: UTOFU_ROUND = round_ms*1e-3
    elif ar_ms  is not None: UTOFU_ROUND = (ar_ms*1e-3 - UTOFU_INIT)/math.ceil(math.log2(n_ref))

def ar_latency(N, msg_bytes, round_ms=None, ar_bf16=False):
    r = UTOFU_ROUND if round_ms is None else round_ms*1e-3
    if ar_bf16: msg_bytes //= 2                                    # TP_AR_BF16 payload
    return UTOFU_INIT + math.ceil(math.log2(N))*r + msg_bytes*UTOFU_BW

def comm_time(N, M, shard_attn=SHARD_ATTN, round_ms=None, attn_group=None, ar_bf16=False):
    """Per forward (one AR/layer serves M tokens). MoE ARs span all N ranks; attention o-proj
    ARs span attn_group ranks (TP subgroup lever; default N = full-group head-shard)."""
    t = N_MOE*ar_latency(N, M*H*4, round_ms, ar_bf16)              # MoE routed-sum (dense/head ARs folded into calibration)
    if shard_attn:
        t += N_LAYERS*ar_latency(attn_group or N, M*H*4, round_ms, ar_bf16)
    return t

def pred_tok_s(N, M, ctx=128, prec='int8', mtp=0.0, shard_attn=SHARD_ATTN,
               round_ms=None, attn_group=None, ar_bf16=False):
    ways = (attn_group if (shard_attn and attn_group) else None)
    t = comm_time(N,M,shard_attn,round_ms,attn_group,ar_bf16) \
      + active_bytes_per_node(N,M,ctx,prec,shard_attn,attn_ways=ways)/BW_NODE
    agg = M/t
    if mtp>0: agg *= (1+mtp)/1.18
    return agg

def m1_tput(N, shard_attn, round_ms, ctx=128, prec='int8', attn_group=None):
    """Single-stream (M=1) tok/s with an OVERRIDABLE per-AR-round cost (round_ms) so we can model a
    leaner decode-AR completion path. round_ms = UTOFU_ROUND default reproduces the measured 0.25."""
    return pred_tok_s(N, 1, ctx, prec, 0.0, shard_attn, round_ms, attn_group)

# ---------- memory feasibility (32 GB HBM2/node) ----------
NODE_GB, RESERVE_GB = 32.0, 2.0
def mem_per_node_gb(N, prec='int8', attn_ways=None, M=0, max_pos=2048):
    """Resident GB on one node: EP-sharded experts + TP-sharded shared/dense/head + replicated
    router/embed + attention (sharded attn_ways-way; 1 = fully replicated) + M-stream KV."""
    wb = 1 if prec=='int8' else 2
    if attn_ways is None: attn_ways = N
    w = (ATTN_P*wb/attn_ways + ROUTER_P*2 + SHARED_P*wb/N + DENSE_P*wb/N
         + (HEAD_P+VOCAB*H)*2/N                       # lm_head + embed, vocab-sharded bf16
         + N_EXP*EXPERT_P*wb*N_MOE/N)                 # ALL experts stored, EP-sharded
    kv = M*N_LAYERS*max_pos*KVC*2                     # per-stream latent KV (bf16)
    return (w+kv)/1e9
def fits(gb): return "OK" if gb <= NODE_GB-RESERVE_GB else "DOES NOT FIT"

# ---------- long-context node sizing (weights sharded + KV context-parallel) ----------
# At 512k/1M ctx the latent KV DOMINATES: one stream's full-ctx KV is 78L*ctx*576*2B (47GB@512k,
# 94GB@1M) -> must context-parallel shard (GLM5_CP=1: each node holds ctx/N positions), else it
# doesn't fit a single node. Weights are secondary (int8 floor ~27n, bf16 ~54n). "int8" = the shipped
# int8-store / int16-compute path (GLM5_GEMM_SDOT=2). kv_bytes: bf16 KV=2, int4 KV=0.5 (GLM5_INT4_KV).
USABLE_GB = NODE_GB - RESERVE_GB - 2.0            # 32 - 2 reserve - ~2 activations/scratch = 28
def weights_gb_per_node(N, prec='int8'):
    """Replicated router(bf16) + sharded /N: attn(TP) + shared/dense(TP) + experts(EP) + head/embed
    (bf16 vocab-sharded). Weight bytes: int8=1, bf16=2."""
    wb = 1 if prec=='int8' else 2
    return (ROUTER_P*2 + (ATTN_P*wb + SHARED_P*wb + DENSE_P*wb + N_EXP*EXPERT_P*N_MOE*wb
            + (HEAD_P+VOCAB*H)*2)/N)/1e9
def kv_gb_per_node(N, ctx, M, kv_bytes=2, cp=True):
    """Latent KV/node: M streams x 78L x (ctx/N if CP else ctx) positions x 576 x kv_bytes."""
    pos = math.ceil(ctx/N) if cp else ctx
    return M*N_LAYERS*pos*KVC*kv_bytes/1e9
def min_nodes(prec='int8', ctx=524288, M=8, kv_bytes=2, usable_gb=USABLE_GB, cp=True, nmax=8192):
    """Fewest nodes where weights + CP-KV fit `usable_gb`. Returns (N, weights_gb, kv_gb) or None.
    This is the MEMORY-fit minimum: decode-optimized wants exactly this (fewer ranks = more efficient,
    measured 32n~=96n); prefill-optimized wants MORE (compute parallelism); prefill+decode (serving)
    is pinned here by the persistent full-ctx KV."""
    for N in range(1, nmax+1):
        w = weights_gb_per_node(N, prec); kv = kv_gb_per_node(N, ctx, M, kv_bytes, cp)
        if w+kv <= usable_gb: return N, w, kv
    return None

# ---------- report ----------
def hdr(s): print("\n"+s+"\n"+"-"*len(s))

if __name__ == "__main__":                       # report only when run directly
    print(__doc__)
    
    hdr("Active bytes per TOKEN (int8 model, M=1, ctx=128) — what must move")
    GB=1e9
    for nm,val in [("attention(all 78L)",ATTN_P),("routed experts(8/L active)",distinct_experts(1)/1*EXPERT_P*N_MOE),
                   ("shared expert",SHARED_P),("dense FFN",DENSE_P),("router(bf16)",ROUTER_P*2),("lm_head(bf16)",HEAD_P*2)]:
        print(f"  {nm:<28} {val/GB:6.2f} GB (full model, pre-shard)")
    print(f"  per-node @96n (sharded)      {active_bytes_per_node(96,1,128)/GB:6.3f} GB/forward")
    
    hdr("THEORETICAL UPPER BOUND — bandwidth ceiling (comm-free), int8, ctx=128")
    print("   N\\M " + "".join(f"{m:>9}" for m in [1,8,16,32,64]))
    for N in [24,48,96,192,384]:
        print(f"  {N:>4} " + "".join(f"{ub_tok_s(N,m,128):>9.0f}" for m in [1,8,16,32,64]))
    print(f"  M=1 96n, attention REPLICATED (no TP head-shard): {ub_tok_s(96,1,128,shard_attn=False):.0f} tok/s")
    print("  (ceiling rises with M until expert reads saturate; falls at long ctx as KV read grows)")
    
    hdr("uTofu all-reduce cost (per call, calibrated)")
    print(f"  per-AR @96n (msg={1*H*4//1024} KB, M=1): {ar_latency(96,1*H*4)*1e3:.1f} ms = "
          f"{UTOFU_INIT*1e3:.1f} init + {math.ceil(math.log2(96))}rounds*{UTOFU_ROUND*1e3:.1f} + bw")
    print(f"  AR/token: {n_allreduce(True)} (attn sharded: 2/MoE layer) vs {n_allreduce(False)} (attn replicated: 1/MoE layer)")
    print(f"  hardware floor would be ~0.01-0.02 ms/AR -> the {UTOFU_ROUND*1e3:.0f} ms/round is the robust-completion tax.")
    
    hdr("CUT 2 ALL-REDUCES/LAYER -> 1  (replicate attention; the only valid 1-AR path)")
    print("   M    2 AR/layer   1 AR/layer   speedup   +MTP(1AR)")
    for M in [1,8,16,32]:
        a=pred_tok_s(96,M,shard_attn=True); b=pred_tok_s(96,M,shard_attn=False)
        print(f"  {M:>3}   {a:>9.2f}    {b:>9.2f}    {b/a:>5.2f}x    {pred_tok_s(96,M,mtp=0.4,shard_attn=False):>7.2f}")
    print("  (decode is comm-bound, so replicating attention's +bandwidth is cheap; prefill keeps sharding.)")
    
    hdr("*** MEASURED CALIBRATION (Fugaku 2026-07-03) — two anchors, reconciled ***")
    print("  (1) ARPROBE tight-loop tp_allreduce (robust=1, M=1, all N ranks):")
    print("      N=8:72us N=16:86us N=32:140us N=96:141us  (flat ~24us/round; robust 0/1/2 within 30%).")
    print("      => the fictional 26 ms/AR 'robustness tax' is DEAD; the bare collective is ~0.14 ms.")
    print("  (2) REAL-WEIGHT cbatch decode (job 49419532, bf16 12L 12n, 4 slots):")
    print("      6.22 tok/s/slot (24.9 agg), comm 41.5%, 3224 AR calls/5.1s => ~0.66 ms EFFECTIVE per-AR @12n.")
    print(f"      => in-decode AR is ~{EFFECTIVE_AR_FACTOR:.0f}x the tight loop (straggler-sync + interleave), so")
    print("         decode IS ~40% comm at 12n (rising with N) -- NOT the 0.5% the tight loop alone implied.")
    print(f"  RECONCILED comm @96n (M=1): {n_allreduce(SHARD_ATTN)} AR x {_ar96*1e3:.2f} ms eff = {_comm96_1*1e3:.0f} ms/token"
          f" -> ~{1/_comm96_1:.0f} tok/s comm-ceiling.")
    print( "  => CORRECTED lever read: comm is a REAL ~40%+ term, but it is straggler/interleave-bound, NOT the")
    print( "     robust completion path -> lean-AR (robust=2) still ~worthless; the levers are BATCHING (amortize")
    print( "     AR over M: measured 2.6x @8n synth), FEWER RANKS (smaller AR + less straggler spread), and load")
    print( "     balance. The other ~60% is compute/HBM. Next: real-weight A/B (bd=0 vs bd=1) + 96n int8 profile.")
    print(f"  qlair --native over-estimated the tight-loop wire ~{qlair_vs_real():.1f}x (sim overhead); it does NOT see")
    print( "     straggler-sync either -> real-weight jobs remain the only source for the effective in-decode AR.")

    hdr("*** REAL INT8 DECODE PROFILE (jobs 49419683/684, full 78L, short ctx) — the 0.25 anchor is DEAD ***")
    print("   N     bd0 tok/s/slot   bd1 (batched)   batched win   comm%")
    for N in sorted(REAL_DECODE):
        b0,b1,cf = REAL_DECODE[N]
        print(f"  {N:>3}      {b0:>6.2f}          {b1:>6.2f}        {b1/b0:>5.2f}x       {cf*100:>4.1f}%")
    print(f"  => real single-stream int8 decode ~1.75 tok/s/slot (~{1.75/AR_M1_96:.0f}x the stale 0.25 anchor).")
    print( "  => 32n ~= 96n throughput -> 32n is ~3x more NODE-EFFICIENT for short-ctx int8 serving (run at 32n).")
    print( "  => comm ~30-37% (flat/slightly-lower at more ranks): 'fewer ranks = less comm' is FALSE; 32n wins on")
    print( "     EFFICIENCY, not speed. batched 1.6-1.7x (bit-identical only on slot0 -> batched path needs a fix).")
    print( "  compute ~65% of the token (bd=0 ms/tok, 32n->96n):")
    for k,(a,b) in REAL_DECODE_COMPUTE_MS.items():
        tag = "  <- top target, does NOT scale with N" if k=='attn' else ("  (scales well w/ EP)" if k=='experts' else "")
        print(f"     {k:<8} {a:>5.1f} -> {b:>5.1f} ms{tag}")

    hdr("Calibrated PREDICTION — batching + MTP (96 nodes, int8, ctx=128)")
    print("   M    agg tok/s   +MTP(0.4)   % of UB")
    for M in [1,8,16,32,64]:
        print(f"  {M:>3}   {pred_tok_s(96,M):>8.2f}   {pred_tok_s(96,M,mtp=0.4):>8.2f}   {100*pred_tok_s(96,M)/ub_tok_s(96,M,128):>6.1f}%")
    
    hdr("Long-context UB (96n, int8) — KV read grows the floor")
    for L in [128,2048,8192,32768]:
        print(f"  ctx={L:>6}: UB M=1 {ub_tok_s(96,1,L):>6.0f}  M=16 {ub_tok_s(96,16,L):>6.0f} tok/s")
    
    hdr("M=1 lever-stack, int8 -> 1 tok/s (single-stream, NO MTP, NO batching)")
    R0 = UTOFU_ROUND*1e3                                              # calibrated per-round cost (~3.4 ms)
    print(f"  target: 1.00 tok/s (4x over 0.25). 1 AR/layer budget = ~13 ms/AR; current = {_ar96*1e3:.0f} ms/AR.")
    print(f"  {'baseline 96n, 2 AR/layer':<46} {m1_tput(96,True ,R0):>5.2f} tok/s")
    print(f"  {'+ 2->1 AR (replicate attention) @96n':<46} {m1_tput(96,False,R0):>5.2f}   (safe; ~2x)")
    print(f"  {'+ smallest int8-fit node count @32n':<46} {m1_tput(32,False,R0):>5.2f}   (safe; smaller AR group)")
    print(f"  {'+ leaner decode-AR completion 1.7 ms/round':<46} {m1_tput(32,False,1.7):>5.2f}   (RISKY: relaxes robust-completion)")
    print(f"  {'+ leaner completion 1.0 ms/round':<46} {m1_tput(32,False,1.0):>5.2f}")
    print(f"  note: round-cost is the trailer-seq/civac/MRQ robustness tax ({R0:.1f} ms vs ~0.02 ms HW floor);")
    print(f"        trimming it is the only lever needing a job to validate (correctness under the races it fixed).")
    
    hdr("M=1 lever-stack, bf16 -> 2 tok/s target (bf16 needs >=96n; replicated attn DOES NOT FIT in bf16)")
    print(f"  bf16 replicated attention = {ATTN_P*2/1e9:.1f} GB/node -> attention must stay SHARDED (or subgroup).")
    print(f"  {'baseline 96n, 2 AR/layer, robust':<52} {m1_tput(96,True ,R0 ,prec='bf16'):>5.2f} tok/s")
    print(f"  {'+ lean AR 1.0 ms/round':<52} {m1_tput(96,True ,1.0,prec='bf16'):>5.2f}")
    print(f"  {'+ lean AR 0.5 ms/round':<52} {m1_tput(96,True ,0.5,prec='bf16'):>5.2f}")
    print(f"  {'+ lean AR 0.3 ms/round':<52} {m1_tput(96,True ,0.3,prec='bf16'):>5.2f}")
    print(f"  {'+ attn TP SUBGROUP k=8 (o-proj AR over 8), 0.5ms':<52} {m1_tput(96,True ,0.5,prec='bf16',attn_group=8):>5.2f}"
          f"   (attn mem {ATTN_P*2/8/1e9:.1f} GB/node: {fits(mem_per_node_gb(96,'bf16',attn_ways=8,M=1))})")
    print(f"  {'+ attn TP SUBGROUP k=8, 0.3ms':<52} {m1_tput(96,True ,0.3,prec='bf16',attn_group=8):>5.2f}")
    print(f"  => bf16 2 tok/s needs the lean AR at <=0.5 ms/round PLUS the attention subgroup (or <=0.3 ms alone).")
    
    hdr("Batched aggregate vs lean-AR round cost (sharded attn, ctx=128) — path to 20 tok/s")
    print("  prec  N    round_ms " + "".join(f"{m:>8}" for m in [8,16,32,64]))
    for prec in ['int8','bf16']:
        for N,rms in [(96,None),(96,1.0),(96,0.5),(32,None),(32,1.0),(32,0.5)]:
            if prec=='bf16' and N<48: continue                        # bf16 does not fit below ~48n
            lbl = f"{UTOFU_ROUND*1e3:.1f}(cal)" if rms is None else f"{rms:.1f}"
            print(f"  {prec:<5}{N:>4}  {lbl:>9} " + "".join(f"{pred_tok_s(N,m,prec=prec,round_ms=rms):>8.1f}" for m in [8,16,32,64]))
    print(f"  +MTP multiplies by ~{(1+0.4)/1.18:.2f} (accept 0.4).  AR msg at M=64 = {64*H*4/1024:.0f} KB -> use TP_AR_BF16.")
    
    hdr("Memory feasibility per node (32 GB HBM2, 2 GB reserve)")
    print("  config                                          GB/node   fit")
    for desc,N,prec,ways,M,mp in [("int8  96n sharded attn, M=32 kv2048", 96,'int8',None,32,2048),
                                   ("int8  96n REPLICATED attn, M=1",      96,'int8',1,   1,2048),
                                   ("int8  32n sharded attn, M=64 kv1024", 32,'int8',None,64,1024),
                                   ("int8  32n REPLICATED attn, M=1",      32,'int8',1,   1,2048),
                                   ("bf16  96n sharded attn, M=32 kv2048", 96,'bf16',None,32,2048),
                                   ("bf16  96n REPLICATED attn, M=1",      96,'bf16',1,   1,2048),
                                   ("bf16  96n attn subgroup k=8, M=1",    96,'bf16',8,   1,2048),
                                   ("bf16  96n attn subgroup k=8, M=32",   96,'bf16',8,  32,2048)]:
        gb = mem_per_node_gb(N,prec,attn_ways=ways,M=M,max_pos=mp)
        print(f"  {desc:<46} {gb:>7.1f}   {fits(gb)}")
    print(f"  (ms->kc per-stream KV: M=32 x 78L x 2048pos x {KVC} x 2B = {32*N_LAYERS*2048*KVC*2/1e9:.1f} GB)")

    hdr(f"Long-context MIN NODES (weights sharded + KV context-parallel, <={USABLE_GB:.0f} GB/node, M=8)")
    print(f"  total weights: int8 {(ROUTER_P*2+ATTN_P+SHARED_P+DENSE_P+N_EXP*EXPERT_P*N_MOE+(HEAD_P+VOCAB*H)*2)/1e9:.0f} GB "
          f"| bf16 {(ROUTER_P*2+(ATTN_P+SHARED_P+DENSE_P+N_EXP*EXPERT_P*N_MOE)*2+(HEAD_P+VOCAB*H)*2)/1e9:.0f} GB "
          f"(floors: int8 >={min_nodes('int8',1,0)[0]}n, bf16 >={min_nodes('bf16',1,0)[0]}n for weights alone)")
    print(f"  KV per stream (78L x ctx x {KVC} x 2B): 512k={N_LAYERS*524288*KVC*2/1e9:.0f} GB  1M={N_LAYERS*1048576*KVC*2/1e9:.0f} GB -> CP-shard mandatory")
    print("   prec   ctx   bf16-KV   int4-KV   (weights+KV GB/node at the int4 min)")
    for prec in ['int8','bf16']:
        for ctx in [524288,1048576]:
            b=min_nodes(prec,ctx,8,2); q=min_nodes(prec,ctx,8,0.5)
            print(f"  {prec:<5}{ctx//1024:>5}k   {b[0]:>5}n    {q[0]:>5}n     ({q[1]:.1f}+{q[2]:.1f})")
    print("  int8=int8-store/int16-compute (GEMM_SDOT=2); int4-KV=GLM5_INT4_KV. batching M=1/4/8 (int8 512k bf16KV): "
          f"{min_nodes('int8',524288,1,2)[0]}/{min_nodes('int8',524288,4,2)[0]}/{min_nodes('int8',524288,8,2)[0]}n.")
    print("  => decode-opt: run AT this min (fewer ranks = more efficient). prefill-opt: run ABOVE (compute")
    print("     parallelism, query-SP). prefill+decode serving: pinned here by the persistent full-ctx KV.")
    print("  query: min_nodes(prec='int8'|'bf16', ctx, M, kv_bytes=2|0.5) -> (N, weights_gb, kv_gb).")

    hdr("Takeaways")
    print(f"""  * THEORETICAL UPPER BOUND (bandwidth) at 96n int8: ~{ub_tok_s(96,1,128):.0f} tok/s (M=1, sharded attn),
        rising to ~{ub_tok_s(96,32,128):.0f} with M=32. Replicated attention would cap it at ~{ub_tok_s(96,1,128,shard_attn=False):.0f}.
      * We run at ~{100*AR_M1_96/ub_tok_s(96,1,128):.1f}% of that — the loss is ~{n_allreduce(SHARD_ATTN)} all-reduces/token at ~{_ar96*1e3:.0f} ms each
        (a ~50-100x-too-slow collective from the robust-completion path), NOT bandwidth/compute.
      * Batching M amortizes the comm: predicted M=16 ~{pred_tok_s(96,16):.1f}, M=32 ~{pred_tok_s(96,32):.1f} tok/s; +MTP ~1.2x.
      * Headroom to the bandwidth ceiling is ~{ub_tok_s(96,16,128)/pred_tok_s(96,16):.0f}x even after batching -> cutting AR latency/count
        (2->1 AR/layer, lighter completion, smaller groups) is the second big lever after batching.""")
    