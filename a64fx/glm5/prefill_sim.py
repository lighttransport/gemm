#!/usr/bin/env python3
"""
GLM-5.2 A64FX PREFILL-throughput model (no job submission) — the decision gate for the
prefill comm rework, mirroring decode_sim.py's role for decode.

ANCHOR (measured int8, 96 nodes, pchunk S=2048, th48): 21.43 tok/s -> t_chunk = 95.6 s.
Phase profile (glm5.2-summary.md): shared 50.0% (= MoE-combine AR wait), qkv 24.8%,
o_proj 11.0%, attn 8.7%, router 2.2%; measured total AR (g_ar_secs) fraction 20.1%.

DECOMPOSITION the model fits from those anchors:
  t_chunk = compute(phases) + AR wire (fitted eff rate over 153 ARs of [S,H]) + SYNC,
  where SYNC := shared-phase wait minus the MoE-AR wire share — the straggler barrier
  (head-shard leaves 32/96 ranks idle in attention; expert fan-in varies; OS jitter).
  At the anchor: wire ~19.2 s, SYNC ~37.9 s (40% of wall!) — the barrier, not the wire,
  is the single biggest term. Any lever that only shrinks wire bytes misses it.

LEVERS modeled (per-MoE-layer comm per rank, bf16 wire b=2):
  (a) status quo      : o_proj AR + MoE AR, each ~2*S*H*b wire       + full SYNC
  (A2) overlap2       : (a) but MoE-AR hidden behind next-layer QKV compute
  (c) token-home a2a  : o_proj AR + dispatch/combine 16*S*H*b/N + allgather S*H*b + partial SYNC
  (d) full query-SP   : a2a 16*S*H*b/N + latent-KV allgather S*KVC*b + idx allgather S*128*b,
                        NO o_proj AR, compute spread N-way over the replicated phases,
                        SYNC -> residual jitter only.
Skew derate on a2a terms: uniform 1.0 / zipf 2.18 (moe_dispatch_bench worst-case finding).
Small-GEMM derate for (d): eff(rows) = rows/(rows+ROWS_HALF) applied to the /N phases —
S/N rows per rank is the main risk (S=2048/N=96 -> 21 rows).

All fitted constants are exposed and replaceable by the PREFILL_PROBE job via
recalibrate_prefill(); a2a/allgather rates default to the moe_dispatch/qlair-floor numbers
times a ROBUST_TAX pessimism factor (the uTofu robust-completion overhead ARPROBE showed is
~50-100x the wire floor for ARs; a2a puts pay a per-put version of it).
"""
import math
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_sim as ds                      # arch constants (H, KVC, N_MOE, ...), silent import

H, KVC, N_MOE, N_DENSE, N_LAYERS = ds.H, ds.KVC, ds.N_MOE, ds.N_DENSE, ds.N_LAYERS
IDXD = 128                                   # MSA index-key dim per position

# ---------------- anchor + fitted decomposition ----------------
AN_N, AN_S, AN_TOKS = 96, 2048, 21.43
T_CHUNK = AN_S/AN_TOKS                                        # 95.56 s
PHASE = dict(shared=0.500, qkv=0.248, oproj=0.110, attn=0.087, router=0.022)
PHASE['rest'] = 1.0 - sum(PHASE.values())                     # experts+dense+head+other ~3.3%
COMM_FRAC = 0.201                                             # measured g_ar_secs / wall
N_AR = N_MOE + N_LAYERS                                       # 75 MoE-combine + 78 o_proj ARs

_wire   = COMM_FRAC*T_CHUNK                                   # 19.2 s total AR wire+tax
AR_S    = _wire/N_AR                                          # per-AR seconds at (96, S=2048)
OPROJ_AR= N_LAYERS*AR_S                                       # o_proj share ~9.8 s
MOE_AR  = N_MOE*AR_S                                          # MoE share  ~9.4 s
SH_GEMM = 0.5                                                 # shared-expert GEMM (TP-sharded, tiny)
SYNC    = PHASE['shared']*T_CHUNK - MOE_AR - SH_GEMM          # straggler barrier ~37.9 s
QKV_C   = PHASE['qkv']*T_CHUNK                                # replicated compute terms
ATTN_C  = PHASE['attn']*T_CHUNK
OPROJ_C = PHASE['oproj']*T_CHUNK - OPROJ_AR                   # o_proj GEMM ~0.7 s
ROUTER_C= PHASE['router']*T_CHUNK
REST_C  = PHASE['rest']*T_CHUNK

# ---------------- tunable transport/efficiency constants ----------------
A2A_GBS    = 86e9        # multi-TNI a2a aggregate (moe_dispatch finding #10, uniform)
AG_GBS     = 40e9        # allgather effective per-node injection (finding #12 / 6x6.8 GB/s spec)
ROBUST_TAX = 5.0         # pessimism: per-put robust-completion overhead vs wire floor (probe-measured later)
ROWS_HALF  = 16          # GEMM efficiency: eff(rows)=rows/(rows+ROWS_HALF) (KERNBENCH-replaceable)
SYNC_RESID = 0.15        # (d): residual jitter as a fraction of remaining compute
SKEW = dict(uniform=1.0, zipf=2.18)

def recalibrate_prefill(a2a_gbs=None, ag_gbs=None, robust_tax=None, rows_half=None,
                        ar_s=None, sync_s=None):
    """Override the fitted constants from PREFILL_PROBE / KERNBENCH measurements."""
    global A2A_GBS, AG_GBS, ROBUST_TAX, ROWS_HALF, AR_S, SYNC
    if a2a_gbs    is not None: A2A_GBS   = a2a_gbs
    if ag_gbs     is not None: AG_GBS    = ag_gbs
    if robust_tax is not None: ROBUST_TAX= robust_tax
    if rows_half  is not None: ROWS_HALF = rows_half
    if ar_s       is not None: AR_S      = ar_s
    if sync_s     is not None: SYNC      = sync_s

def eff(rows):                      # low-M GEMM efficiency (time multiplier = 1/eff)
    return rows/(rows+ROWS_HALF)

def ar_time(N, S, n_ar):
    """AR wire+tax, scaled from the fitted per-AR anchor: bytes prop to S, rounds prop to log2 N."""
    return n_ar*AR_S*(S/AN_S)*(math.ceil(math.log2(N))/math.ceil(math.log2(AN_N)))

def a2a_time(N, S, skew='uniform'):
    """MoE dispatch+combine under token-home routing: 16*S*H*b/N bytes/rank/layer."""
    bytes_l = 16*S*H*2/N
    return N_MOE*bytes_l/A2A_GBS*SKEW[skew]*ROBUST_TAX

def allgather_time(S, dim, layers):
    return layers*S*dim*2/AG_GBS*ROBUST_TAX

def scale(S, N):                     # compute phases scale ~linearly in S (attn ~S^2 but anchored at one S; keep linear for S near anchor, add quadratic for attn)
    return S/AN_S

# ---------------- lever models: return (tok/s, breakdown dict) ----------------
def lever_a(N, S):
    f = scale(S, N); fa = (S/AN_S)**2          # attention score loop ~S^2
    comp = QKV_C*f + ATTN_C*fa + OPROJ_C*f + ROUTER_C*f + REST_C*f + SH_GEMM*f
    comm = ar_time(N, S, N_AR)
    sync = SYNC*f
    t = comp+comm+sync
    return S/t, dict(comp=comp, comm=comm, sync=sync, t=t)

def lever_A2(N, S):
    """(a) with the MoE-combine AR overlapped behind next-layer QKV (hide up to min(moe_ar, qkv))."""
    toks, br = lever_a(N, S)
    moe_ar = ar_time(N, S, N_MOE); qkv = QKV_C*scale(S,N)
    hidden = min(moe_ar, qkv)
    t = br['t'] - hidden
    return S/t, dict(**br, hidden=hidden, t2=t)

def lever_c(N, S, skew='uniform'):
    f = scale(S, N); fa = (S/AN_S)**2
    comp = QKV_C*f + ATTN_C*fa + OPROJ_C*f + ROUTER_C*f + REST_C*f + SH_GEMM*f
    comm = ar_time(N, S, N_LAYERS) + a2a_time(N, S, skew) + allgather_time(S, H, N_MOE)
    sync = SYNC*f*0.7                          # o_proj barrier remains; MoE barrier mostly gone
    t = comp+comm+sync
    return S/t, dict(comp=comp, comm=comm, sync=sync, t=t)

def lever_d(N, S, skew='uniform'):
    """Full query-SP: replicated phases spread N-way (with low-rows derate), no o_proj AR,
    MoE via a2a, per-layer latent-KV + index-key allgathers, sync -> residual."""
    rows = S/N
    e = eff(rows)
    f = scale(S, N); fa = (S/AN_S)**2
    qkv    = QKV_C*f/N/e
    router = ROUTER_C*f/N/e
    attn   = ATTN_C*fa*(64/96)                 # all heads x S/N queries, balanced (vs 1 head x S busy-rank)
    oproj  = OPROJ_C*f*(64/96)/e
    comp = qkv+router+attn+oproj + REST_C*f + SH_GEMM*f     # experts/shared: same total ops
    comm = a2a_time(N, S, skew) + allgather_time(S, KVC, N_LAYERS) + allgather_time(S, IDXD, N_MOE)
    sync = SYNC_RESID*comp
    t = comp+comm+sync
    return S/t, dict(comp=comp, comm=comm, sync=sync, t=t, rows=rows, gemm_eff=e)

# ---------------- report ----------------
def hdr(s): print("\n"+s+"\n"+"-"*len(s))

if __name__ == "__main__":
    print(__doc__)

    hdr(f"Anchor decomposition (int8, N={AN_N}, S={AN_S}, measured {AN_TOKS} tok/s)")
    toks, br = lever_a(AN_N, AN_S)
    print(f"  model reproduces: {toks:.2f} tok/s (t={br['t']:.1f}s)  [target {AN_TOKS}]")
    print(f"  compute {br['comp']:.1f}s | AR wire {br['comm']:.1f}s ({N_AR} ARs x {AR_S*1e3:.0f} ms) | SYNC barrier {br['sync']:.1f}s")
    print(f"  => the straggler/sync barrier ({br['sync']/br['t']*100:.0f}% of wall) exceeds the wire ({br['comm']/br['t']*100:.0f}%).")

    hdr("Lever table — predicted prefill tok/s (int8-era anchor, uniform routing)")
    print("   N     S    (a)base  (A2)ovl   (c)a2a   (d)SP    SP rows/rank  gemm_eff")
    for N in [24, 48, 96, 192]:
        for S in [2048, 4096, 8192]:
            a,_ = lever_a(N,S); A,_ = lever_A2(N,S); c,_ = lever_c(N,S); d,bd = lever_d(N,S)
            print(f"  {N:>4} {S:>5}   {a:>6.1f}   {A:>6.1f}   {c:>6.1f}   {d:>6.1f}      {bd['rows']:>5.0f}       {bd['gemm_eff']*100:>4.0f}%")

    hdr("Query-SP (d) breakdown at N=96")
    for S in [2048, 4096, 8192]:
        d, b = lever_d(96, S)
        print(f"  S={S:>5}: {d:6.1f} tok/s  comp {b['comp']:.2f}s (qkv-spread) comm {b['comm']:.2f}s sync {b['sync']:.2f}s")

    hdr("Sensitivity of (d) at N=96, S=4096")
    base,_ = lever_d(96, 4096)
    for name, setter, vals in [
        ("ROBUST_TAX (a2a/allgather overhead vs wire)", 'robust_tax', [1, 5, 20, 50]),
        ("ROWS_HALF (GEMM 50%-eff point, rows)",        'rows_half',  [8, 16, 32, 64]),
    ]:
        row = []
        for v in vals:
            recalibrate_prefill(**{setter: v})
            row.append(f"{v}:{lever_d(96,4096)[0]:.1f}")
        recalibrate_prefill(robust_tax=5.0, rows_half=16)
        print(f"  {name:<46} " + "  ".join(row))
    saved = SYNC_RESID
    row = []
    for v in [0.0, 0.15, 0.3, 0.6]:
        globals()['SYNC_RESID'] = v
        row.append(f"{v}:{lever_d(96,4096)[0]:.1f}")
    globals()['SYNC_RESID'] = saved
    print(f"  {'SYNC_RESID (residual jitter fraction)':<46} " + "  ".join(row))

    hdr("Zipf-skew check on (d) (a2a term derated 2.18x)")
    for S in [2048, 4096]:
        u,_ = lever_d(96,S,'uniform'); z,_ = lever_d(96,S,'zipf')
        print(f"  S={S}: uniform {u:.1f} vs zipf {z:.1f} tok/s")

    hdr("Verdict")
    d96, b96 = lever_d(96, 2048); a96,_ = lever_a(96, 2048)
    print(f"""  * (d) query-SP at N=96, S=2048: predicted ~{d96:.0f} tok/s vs baseline {a96:.1f} = {d96/a96:.1f}x.
    The win survives ROBUST_TAX up to ~50x and Zipf skew (comm terms are tiny under SP);
    it is dominated by removing the SYNC barrier + spreading the replicated 45% of compute.
  * pchunk: S=2048 is OPTIMAL — the attention S^2 term outweighs the small-GEMM derate
    (57% eff at 21 rows costs less than 4x more attention work). Do NOT raise pchunk for (d).
  * BIGGEST model risk: how much of the fitted 37.9s SYNC term is truly removable
    (vs mis-attributed wire/compute) — the FIRST number PREFILL_PROBE must split.
  * (A2) overlap2 is worth banking regardless: hides most of the MoE-AR wire behind QKV.
  * (c) is confirmed WEAK (keeps o_proj AR + [S,H] allgather) — fallback only.
  * All transport constants here are fitted/spec — the PREFILL_PROBE job replaces them
    via recalibrate_prefill() before any final go/no-go at scale.""")
