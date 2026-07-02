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
    
    hdr("Takeaways")
    print(f"""  * THEORETICAL UPPER BOUND (bandwidth) at 96n int8: ~{ub_tok_s(96,1,128):.0f} tok/s (M=1, sharded attn),
        rising to ~{ub_tok_s(96,32,128):.0f} with M=32. Replicated attention would cap it at ~{ub_tok_s(96,1,128,shard_attn=False):.0f}.
      * We run at ~{100*AR_M1_96/ub_tok_s(96,1,128):.1f}% of that — the loss is ~{n_allreduce(SHARD_ATTN)} all-reduces/token at ~{_ar96*1e3:.0f} ms each
        (a ~50-100x-too-slow collective from the robust-completion path), NOT bandwidth/compute.
      * Batching M amortizes the comm: predicted M=16 ~{pred_tok_s(96,16):.1f}, M=32 ~{pred_tok_s(96,32):.1f} tok/s; +MTP ~1.2x.
      * Headroom to the bandwidth ceiling is ~{ub_tok_s(96,16,128)/pred_tok_s(96,16):.0f}x even after batching -> cutting AR latency/count
        (2->1 AR/layer, lighter completion, smaller groups) is the second big lever after batching.""")
    