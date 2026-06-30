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
BW_NODE = 300e9            # per-node EFFECTIVE matvec HBM bandwidth. qlair/A64FX HBM2 peak=1024 GB/s/node (4 stacks); measured decode-BW bench ~300 matvec / ~770 load. (was 150)
AR_M1_96 = 0.25            # measured M=1 96n tok/s -> calibrates the comm term

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

def active_bytes_per_node(N, M, ctx, prec='int8', shard_attn=True):
    """Bytes read by ONE node in ONE forward serving M streams at context length `ctx`."""
    wb = 1 if prec=='int8' else 2
    attn = ATTN_P*wb/(N if shard_attn else 1)            # sharded heads -> /N; replicated -> full
    router = ROUTER_P*2                                   # bf16, replicated
    shared = SHARED_P*wb/N                                # TP-sharded
    dense  = DENSE_P*wb/N
    head   = HEAD_P*2/N                                   # bf16, vocab-sharded
    experts = distinct_experts(M)/N * EXPERT_P*wb * N_MOE # distinct active experts this node owns
    kv = M * N_LAYERS * ctx * KVC * 2                     # per-stream latent KV read (bf16), CP off
    return attn + router + shared + dense + head + experts + kv

def ub_tok_s(N, M, ctx, prec='int8', shard_attn=True):    # THEORETICAL bandwidth ceiling (comm-free)
    return M / (active_bytes_per_node(N,M,ctx,prec,shard_attn)/BW_NODE)

# ---------- uTofu all-reduce cost model (explicit) ----------
# Each EP all-reduce is recursive-doubling over N ranks: ceil(log2 N) rounds, each a uTofu Put +
# a completion wait. The A64FX/uTofu Put itself is ~1-2 us, but the robust-completion path used here
# (trailer-seq fence + civac cache-coherence flush + MRQ draining, added for correctness) dominates
# with a large per-round fixed cost. Model: ar = UTOFU_INIT + rounds*(UTOFU_ROUND) + bytes*UTOFU_BW.
# UTOFU_ROUND is CALIBRATED to reproduce the measured per-AR latency at 96 ranks; the split into
# init/round/bandwidth is illustrative (and shows what a leaner completion path would buy).
UTOFU_INIT  = 2.0e-3            # fixed per-AR setup/teardown (s) -- registration, fence issue
UTOFU_BW    = 1.0/8e9          # s/byte once in flight (~8 GB/s effective per-link AR bandwidth)
def n_allreduce(shard_attn): return N_MOE + (N_LAYERS if shard_attn else 0)
SHARD_ATTN = True              # eval config: attention heads sharded -> 2 AR/MoE layer
_bw96_1 = active_bytes_per_node(96,1,128)/BW_NODE
_comm96_1 = 1.0/AR_M1_96 - _bw96_1                                 # total comm s at (96, M=1)
_ar96 = _comm96_1/n_allreduce(SHARD_ATTN)                         # measured per-AR latency at 96n (~26 ms)
UTOFU_ROUND = (_ar96 - UTOFU_INIT)/math.ceil(math.log2(96))       # calibrated per-round cost (the robustness tax)
def ar_latency(N, msg_bytes):
    return UTOFU_INIT + math.ceil(math.log2(N))*UTOFU_ROUND + msg_bytes*UTOFU_BW
def comm_time(N, M, shard_attn=SHARD_ATTN):                        # per forward (one AR/layer serves M tokens)
    return n_allreduce(shard_attn)*ar_latency(N, M*H*4)            # AR message = [M,hidden] f32

def pred_tok_s(N, M, ctx=128, prec='int8', mtp=0.0, shard_attn=SHARD_ATTN):
    t = comm_time(N,M,shard_attn) + active_bytes_per_node(N,M,ctx,prec,shard_attn)/BW_NODE
    agg = M/t
    if mtp>0: agg *= (1+mtp)/1.18
    return agg

def m1_tput(N, shard_attn, round_ms, ctx=128):
    """Single-stream (M=1) tok/s with an OVERRIDABLE per-AR-round cost (round_ms) so we can model a
    leaner decode-AR completion path. round_ms = UTOFU_ROUND default reproduces the measured 0.25."""
    nar = n_allreduce(shard_attn)
    ar  = UTOFU_INIT + math.ceil(math.log2(N))*round_ms*1e-3 + (1*H*4)*UTOFU_BW
    return 1.0/(nar*ar + active_bytes_per_node(N,1,ctx,'int8',shard_attn)/BW_NODE)

# ---------- report ----------
def hdr(s): print("\n"+s+"\n"+"-"*len(s))
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

hdr("Calibration + the gap (96 nodes, int8, M=1)")
print(f"  bandwidth-bound time   : {_bw96_1*1e3:7.2f} ms/token  -> UB {ub_tok_s(96,1,128):.0f} tok/s")
print(f"  measured / comm-bound  : {1/AR_M1_96*1e3:7.0f} ms/token  -> {AR_M1_96} tok/s")
print(f"  => all-reduce overhead : {_comm96_1*1e3:7.0f} ms ({100*_comm96_1/(1/AR_M1_96):.0f}% of the token); "
      f"{n_allreduce(SHARD_ATTN)} AR/token @ {_ar96*1e3:.0f} ms each")
print(f"  decode runs at {100*AR_M1_96/ub_tok_s(96,1,128):.1f}% of the bandwidth ceiling -> ~{ub_tok_s(96,1,128)/AR_M1_96:.0f}x headroom, all comm.")

hdr("Calibrated PREDICTION — batching + MTP (96 nodes, int8, ctx=128)")
print("   M    agg tok/s   +MTP(0.4)   % of UB")
for M in [1,8,16,32,64]:
    print(f"  {M:>3}   {pred_tok_s(96,M):>8.2f}   {pred_tok_s(96,M,mtp=0.4):>8.2f}   {100*pred_tok_s(96,M)/ub_tok_s(96,M,128):>6.1f}%")

hdr("Long-context UB (96n, int8) — KV read grows the floor")
for L in [128,2048,8192,32768]:
    print(f"  ctx={L:>6}: UB M=1 {ub_tok_s(96,1,L):>6.0f}  M=16 {ub_tok_s(96,16,L):>6.0f} tok/s")

hdr("M=1 lever-stack -> 1 tok/s (single-stream, NO MTP, NO batching)")
R0 = UTOFU_ROUND*1e3                                              # calibrated per-round cost (~3.4 ms)
print(f"  target: 1.00 tok/s (4x over 0.25). 1 AR/layer budget = ~13 ms/AR; current = {_ar96*1e3:.0f} ms/AR.")
print(f"  {'baseline 96n, 2 AR/layer':<46} {m1_tput(96,True ,R0):>5.2f} tok/s")
print(f"  {'+ 2->1 AR (replicate attention) @96n':<46} {m1_tput(96,False,R0):>5.2f}   (safe; ~2x)")
print(f"  {'+ smallest int8-fit node count @32n':<46} {m1_tput(32,False,R0):>5.2f}   (safe; smaller AR group)")
print(f"  {'+ leaner decode-AR completion 1.7 ms/round':<46} {m1_tput(32,False,1.7):>5.2f}   (RISKY: relaxes robust-completion)")
print(f"  {'+ leaner completion 1.0 ms/round':<46} {m1_tput(32,False,1.0):>5.2f}")
print(f"  note: round-cost is the trailer-seq/civac/MRQ robustness tax ({R0:.1f} ms vs ~0.02 ms HW floor);")
print(f"        trimming it is the only lever needing a job to validate (correctness under the races it fixed).")

hdr("Takeaways")
print(f"""  * THEORETICAL UPPER BOUND (bandwidth) at 96n int8: ~{ub_tok_s(96,1,128):.0f} tok/s (M=1, sharded attn),
    rising to ~{ub_tok_s(96,32,128):.0f} with M=32. Replicated attention would cap it at ~{ub_tok_s(96,1,128,shard_attn=False):.0f}.
  * We run at ~{100*AR_M1_96/ub_tok_s(96,1,128):.1f}% of that — the loss is ~{n_allreduce(SHARD_ATTN)} all-reduces/token at ~{_ar96*1e3:.0f} ms each
    (a ~50-100x-too-slow collective from the robust-completion path), NOT bandwidth/compute.
  * Batching M amortizes the comm: predicted M=16 ~{pred_tok_s(96,16):.1f}, M=32 ~{pred_tok_s(96,32):.1f} tok/s; +MTP ~1.2x.
  * Headroom to the bandwidth ceiling is ~{ub_tok_s(96,16,128)/pred_tok_s(96,16):.0f}x even after batching -> cutting AR latency/count
    (2->1 AR/layer, lighter completion, smaller groups) is the second big lever after batching.""")
