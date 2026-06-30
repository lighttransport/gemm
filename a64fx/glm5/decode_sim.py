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
BW_NODE = 150e9            # effective per-node HBM matvec bandwidth (A64FX HBM2 ~256 GB/s peak; ~60% eff)
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

# ---------- comm model (calibrated) ----------
# decode pays n_ar all-reduces/forward (MoE combine + attention o-proj if heads sharded), amortized
# over M. AR latency ~ fixed floor + recursive-doubling rounds (log2 N). Calibrate the floor to the
# measured M=1 point: comm(96,1) = 1/0.25 - bw_time(96,1).
def n_allreduce(shard_attn): return N_MOE + (N_LAYERS if shard_attn else 0)
def _ar_shape(N): return 0.30 + 0.70*math.log2(N)/math.log2(96)     # fixed floor + log2 growth, =1 at 96
SHARD_ATTN = True
_bw96_1 = active_bytes_per_node(96,1,128)/BW_NODE
_comm96_1 = 1.0/AR_M1_96 - _bw96_1                                  # total comm seconds at (96, M=1)
AR_LAT96 = _comm96_1/n_allreduce(SHARD_ATTN)                        # per-all-reduce latency at 96n
def comm_time(N, M):                                               # per forward (amortizes over M via 1 call)
    return n_allreduce(SHARD_ATTN)*AR_LAT96*_ar_shape(N)           # (msg-size term negligible at decode sizes)

def pred_tok_s(N, M, ctx=128, prec='int8', mtp=0.0):
    t = comm_time(N,M) + active_bytes_per_node(N,M,ctx,prec)/BW_NODE
    agg = M/t
    if mtp>0: agg *= (1+mtp)/1.18
    return agg

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

hdr("Calibration + the gap (96 nodes, int8, M=1)")
print(f"  bandwidth-bound time   : {_bw96_1*1e3:7.2f} ms/token  -> UB {ub_tok_s(96,1,128):.0f} tok/s")
print(f"  measured / comm-bound  : {1/AR_M1_96*1e3:7.0f} ms/token  -> {AR_M1_96} tok/s")
print(f"  => all-reduce overhead : {_comm96_1*1e3:7.0f} ms ({100*_comm96_1/(1/AR_M1_96):.0f}% of the token); "
      f"{n_allreduce(SHARD_ATTN)} AR/token @ {AR_LAT96*1e3:.0f} ms each")
print(f"  decode runs at {100*AR_M1_96/ub_tok_s(96,1,128):.1f}% of the bandwidth ceiling -> ~{ub_tok_s(96,1,128)/AR_M1_96:.0f}x headroom, all comm.")

hdr("Calibrated PREDICTION — batching + MTP (96 nodes, int8, ctx=128)")
print("   M    agg tok/s   +MTP(0.4)   % of UB")
for M in [1,8,16,32,64]:
    print(f"  {M:>3}   {pred_tok_s(96,M):>8.2f}   {pred_tok_s(96,M,mtp=0.4):>8.2f}   {100*pred_tok_s(96,M)/ub_tok_s(96,M,128):>6.1f}%")

hdr("Long-context UB (96n, int8) — KV read grows the floor")
for L in [128,2048,8192,32768]:
    print(f"  ctx={L:>6}: UB M=1 {ub_tok_s(96,1,L):>6.0f}  M=16 {ub_tok_s(96,16,L):>6.0f} tok/s")

hdr("Takeaways")
print(f"""  * THEORETICAL UPPER BOUND (bandwidth) at 96n int8: ~{ub_tok_s(96,1,128):.0f} tok/s (M=1, sharded attn),
    rising to ~{ub_tok_s(96,32,128):.0f} with M=32. Replicated attention would cap it at ~{ub_tok_s(96,1,128,shard_attn=False):.0f}.
  * We run at ~{100*AR_M1_96/ub_tok_s(96,1,128):.1f}% of that — the loss is ~{n_allreduce(SHARD_ATTN)} all-reduces/token at ~{AR_LAT96*1e3:.0f} ms each
    (a ~50-100x-too-slow collective from the robust-completion path), NOT bandwidth/compute.
  * Batching M amortizes the comm: predicted M=16 ~{pred_tok_s(96,16):.1f}, M=32 ~{pred_tok_s(96,32):.1f} tok/s; +MTP ~1.2x.
  * Headroom to the bandwidth ceiling is ~{ub_tok_s(96,16,128)/pred_tok_s(96,16):.0f}x even after batching -> cutting AR latency/count
    (2->1 AR/layer, lighter completion, smaller groups) is the second big lever after batching.""")
