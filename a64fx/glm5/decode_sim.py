#!/usr/bin/env python3
"""
GLM-5.2 A64FX decode-throughput estimator  (no job submission required).

WHY: single-stream (M=1) decode measured ~0.25 tok/s on int8/96n, ~15x below the
~3-5 tok/s one would expect. This model explains where the time goes and predicts
the speedup from the available levers (batching M, node count N, speculative MTP)
so we can plan source-level optimization without spending job hours.

CALIBRATION ANCHORS (measured this session, int8 w8a16):
  * 96 nodes, single-stream (M=1) decode .......... 0.25 tok/s  (latency-bound)
  * 96 nodes, bulk prefill (M=256 batch) .......... 21   tok/s  (best batched throughput)

MODEL (two-term, per forward serving M concurrent sequences):
  forward_time(N,M) = A(N)            # fixed per-forward cost: the per-layer all-reduce
                                      #   latency x #MoE-layers + dispatch/sync. Amortized over M.
                    + B * M           # marginal per-token cost (weight bandwidth + activation).
  aggregate_tok_s = M / forward_time         # throughput across the M streams
  per_seq_tok_s   = 1 / forward_time         # latency of a single stream
As M grows the fixed comm cost A(N) amortizes away and throughput -> 1/B (the prefill ceiling).
The decode wall is A(N): ~78 collective all-reduces per token, fully exposed at M=1.
"""
import math

# ----- GLM-5.2 architecture -----
H, N_LAYERS, N_MOE = 6144, 78, 75
Q_LORA, KV_LORA, N_HEADS, QK_HEAD, V_HEAD = 2048, 512, 64, 192, 256
MOE_INTER, N_EXP, N_ACT, VOCAB = 2048, 256, 8, 154880

# ----- platform / precision -----
BW_NODE   = 150e9            # effective per-node matvec HBM bandwidth (A64FX HBM2 ~256 GB/s peak)
PREFILL_CEILING = 21.0       # measured bulk-prefill tok/s = conservative batched ceiling -> sets B
DECODE_M1_96N   = 0.25       # measured single-stream decode tok/s -> calibrates A(96)

def attn_bytes(prec):        # all-layer attention weights read every token (dense, not sparse)
    b = 1 if prec=='int8' else 2
    per = (Q_LORA*H + N_HEADS*QK_HEAD*Q_LORA + (KV_LORA+QK_HEAD//3)*H
           + N_HEADS*(V_HEAD+128)*KV_LORA + H*N_HEADS*V_HEAD)
    return N_LAYERS*per*b
def expert_bytes_per(prec):  # one routed/shared expert (gate+up+down)
    return 3*MOE_INTER*H*(1 if prec=='int8' else 2)
def distinct_experts(M):     # E[#distinct experts hit] when M tokens each pick N_ACT of N_EXP
    return N_EXP*(1-(1-N_ACT/N_EXP)**M)

# ----- two-term cost model, calibrated to the anchors -----
B = 1.0/PREFILL_CEILING                       # s/token marginal cost (batched ceiling)
A96 = 1.0/DECODE_M1_96N - B                   # s/forward fixed comm cost at 96 nodes (~3.95 s)
def A(N):
    # per-forward fixed cost is ~ #MoE all-reduces x AR latency; AR latency ~ recursive-doubling
    # rounds (log2 N) with a fixed floor. Anchored at A(96); scales with the collective size.
    f = lambda n: 0.40 + 0.60*math.log2(n)/math.log2(96)   # floor 0.4, log2 growth
    return A96 * f(N)/f(96)

def forward_time(N, M, prec='int8'):
    return A(N) + B*M*( (1 if prec=='int8' else 2.0) )      # bf16 ~2x the marginal bandwidth term

def decode_tok_s(N, M, prec='int8', mtp_accept=0.0):
    t = forward_time(N, M, prec)
    agg = M/t
    if mtp_accept > 0:
        agg *= (1.0+mtp_accept)/1.18           # MTP: ~(1+alpha) tokens/forward, ~18% draft+verify overhead
    return agg

# ----- reports -----
def hdr(s): print("\n"+s+"\n"+"-"*len(s))

print(__doc__)
hdr("Calibration check")
print(f"  int8 96n M=1  : {decode_tok_s(96,1):.3f} tok/s  (target 0.25)")
print(f"  B (marginal)  : {B*1e3:.1f} ms/token  -> batched ceiling {1/B:.0f} tok/s")
print(f"  A(96) (comm)  : {A(96):.2f} s/forward  = {A(96)/N_MOE*1e3:.0f} ms per MoE all-reduce x {N_MOE} layers")

hdr("Where the M=1 decode token goes (96 nodes, int8)")
t=forward_time(96,1); comm=A(96); bw=B
print(f"  comm/dispatch (78 collectives) : {comm:.2f} s  ({100*comm/t:.0f}%)  <-- the wall")
print(f"  weight bandwidth + compute     : {bw:.3f} s  ({100*bw/t:.0f}%)")
print(f"  => decode is collective-LATENCY bound, not bandwidth/compute bound.")

hdr("Aggregate tok/s vs batch M and nodes N (int8, no MTP)")
Ms=[1,2,4,8,16,32,64]
print("   N\\M " + "".join(f"{m:>8}" for m in Ms))
for N in [12,24,48,96,192,384]:
    print(f"  {N:>4} " + "".join(f"{decode_tok_s(N,m):>8.2f}" for m in Ms))

hdr("Path to 3-5 tok/s (96 nodes, int8)")
for M in [1,8,16,32]:
    base=decode_tok_s(96,M); mtp=decode_tok_s(96,M,mtp_accept=0.4)
    print(f"  M={M:>2}: {base:>5.2f} tok/s (agg) | per-seq {1/forward_time(96,M):.2f} | +MTP(a=0.4) {mtp:>5.2f}")
print("  fewer nodes for decode (24n) :")
for M in [8,16,32]:
    print(f"    24n M={M:>2}: {decode_tok_s(24,M):>5.2f} tok/s | +MTP {decode_tok_s(24,M,mtp_accept=0.4):>5.2f}")

hdr("Takeaways")
print("""  1. M=1 decode is ~99% collective latency (78 all-reduces/token, fully exposed).
  2. Batched/continuous decode (M>1) amortizes those all-reduces ~M-fold:
       M=16 ~3.4 tok/s, M=32 ~5-6 tok/s (aggregate) at 96 nodes.
  3. MTP speculative (accept ~0.4) adds ~1.2-1.3x on top, multiplicative with batching.
  4. Fewer nodes for decode (smaller all-reduce group) helps the fixed term ~1.2-1.5x.
  5. The batched-decode path already exists (run_cbatch / multi-stream); single-stream
     generation just never used it. That is the highest-leverage source-level change.""")
