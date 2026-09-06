#!/bin/bash
# DeepSeek-V4 BASE (~/models/ds4fbase) real-weight EP harness on ALL 12 A64FX nodes.
#
# Sibling of run_ds4f_11n.sh. The base model is the SAME graph as Flash — config.json differs
# in exactly one field, expert_dtype fp8 (vs fp4) — so this reuses ds4f_ep_runner verbatim and
# only flips DS4F_MODEL=ds4fbase (which sets cfg.expert_qt = DS4F_FP8; see common/ds4f.h).
#
# ---- WHY THE TP STACK IS MANDATORY HERE ----------------------------------------------------
# FP8 experts cost 24 MiB each vs MXFP4's 12.75, so at EP=12 the experts alone are 22.17 GiB/node
# (Flash at EP=11: 12.85). Per-node budget (GiB, measured model, node MemAvailable = 29.0):
#
#              experts  dense  tb2w  emb/head  TOTAL
#   no TP        22.17   5.50  0.87    1.97    30.52   <-- DOES NOT FIT
#   +TP          22.17   1.26  0.87    0.16    24.47   <-- fits, ~4.5 GiB headroom
#
# So DS4F_TP_ATTN/OPROJ/WOB/SHARED/HEAD/EMBED default ON in this script. Turning any of them off
# is what will OOM you. (All were built+validated for ds4p; see common/ds4f_impl.h:104-190.)
# For the same reason DS4F_FP8_BF16 stays 0 — the bf16 dense predequant costs ~+6 GB we do not
# have, which also means DS4F_PREFILL_BATCH must stay 0 (batched prefill requires FP8_BF16=1).
#
# ---- THE SHARED NODE ------------------------------------------------------------------------
# Interactive is capped at 12 nodes, so ALL 12 are EP ranks and one shares the node with the
# login/claude session. The vcoordfile puts that node LAST => it is EP rank 11, and since
# 256 = 12*21 + 4 ranks 0-3 own 22 experts while ranks 4-11 own 21 — so the shared node gets the
# SMALL shard (~23.5 GiB instead of 24.5). Keep the MemAvailable guard armed regardless.
#
# ---- FAST DECODE: baked Q8 dense + TP_ATTN=0  (14.06 tok/s vs 11.34, +24%) ------------------
# The default below (FP8-on-demand dense + full TP) is the SAFE/reference path. The FAST path is:
#
#   ./build/ds4f_bake  with DS4F_MODEL=ds4fbase            # once, ~2 min -> ~/models/ds4fbase-fast
#   DS4F_DENSE=q8pv DS4F_STAGE_DIR=/local/ds4fbase_q8 ./run_ds4fbase_stage_12n.sh
#   DS4F_DENSE=q8pv DS4F_STAGE_DIR=/local/ds4fbase_q8 DS4F_TP_ATTN=0 ./run_ds4fbase_12n.sh
#
# Measured 12n, same allocation (decode tok/s / comm / o_proj ms):
#   FP8 dense, full TP  (reference)  11.34   40.2%   19.34
#   Q8  dense, full TP                9.02   59.2%   40.08   <- TP_ATTN's s_attn reduce dominates
#   Q8  dense, TP_ATTN=0             14.06   32.1%    4.60   <- FAST
#
# WHY TP_ATTN=0 only wins with Q8: o_proj's timer includes TP_ATTN's 128 KB/layer s_attn
# all-reduce (wo_a needs the FULL s_attn). Under FP8 the extra wq_b compute from un-sharding
# costs MORE than the reduce saves (11.34 -> 8.16, a LOSS). Under Q8 the dense matvec is ~1.8x
# faster, so the trade flips hard. Both need the Q8 cshard (TP_WOB) + the block-diag activation
# pre-quantize; without those Q8 is a net loss. See a64fx/ds4f.md.
# NOTE: DS4F_DENSE must MATCH how the blob was staged, or the loader aborts (by design).
#
# Run INSIDE the existing 12-node allocation (NO pjsub), AFTER run_ds4fbase_stage_12n.sh:
#   ./run_ds4fbase_12n.sh
#   DS4F_LAYERS=4 DS4F_MAXGEN=4 ./run_ds4fbase_12n.sh    # fast smoke / memory probe
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"
UTOFU_DIR="$LLM_DIR/../utofu-tests"

NP=${NP:-12}
LAST=${LAST:-0,0,0}                       # login/claude node -> placed last -> EP rank 11
VCOORD=${VCOORD:-vcoord_ds4fbase.txt}     # SAME file the stager used (rank<->node binding)

# ---- vcoordfile (identical construction to run_ds4fbase_stage_12n.sh) ----
if [ "${REUSE_VCOORD:-0}" != "1" ] || [ ! -s "$VCOORD" ]; then
    SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
    SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
    SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
    : > "$VCOORD"
    for ((x=0; x<SX; x++)); do for ((y=0; y<SY; y++)); do for ((z=0; z<SZ; z++)); do
        [ "$x,$y,$z" = "$LAST" ] && continue
        echo "($x,$y,$z)" >> "$VCOORD"
    done; done; done
    echo "($LAST)" >> "$VCOORD"
    head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
fi
echo "[run_ds4fbase] placing $NP ranks via $VCOORD (($LAST) last => EP rank $((NP-1)), 21 experts):"
cat -n "$VCOORD" | sed 's/^/    /'

# ---- threads / NUMA ----
# *** LEAVE ONE CORE FREE. 47, NOT 48. This is worth +40% and is not optional. ***
# The node's cgroup gives us cores 12-59 (48 compute cores; the assistant cores 0-1 are NOT in
# our cpuset, so nothing of ours can be moved off). With 48 OMP threads pinned 1:1 to 48 cores,
# ANY other process on the node (the co-located claude session, an MPI progress thread, an OS
# daemon) forces one OMP thread to timeshare -- and because the pool barrier waits for ALL 48,
# that single descheduled thread stalls the rank, which stalls all 12 ranks (comm == wait for
# the slowest). Measured 12n, DS4F_DENSE=q8pv + TP_ATTN=0 + CMP_LOCAL + HC_SVE:
#
#   threads  decode              compute spread   comm
#   48       12.00 / 12.51       20.3 / 15.0 ms   44.6   <- rank11 (claude node) 53 ms vs 33 min
#   47       17.06 / 17.05        5.3 /  5.3      21.6   <- +40%, and REPRODUCIBLE to 0.01
#   46       16.91 / 16.89        5.0 /  5.0      22.2
#
# renice does NOT fix it (the OMP threads spin-wait, so a niced competitor still preempts them):
# 48t + nice 19 still gave spread 20.4 ms. Only leaving a core free works.
# This also explains the "run-to-run variance" I chased earlier -- it was our own session
# preempting an OMP thread, not the fabric.
# The residual 5.3 ms spread is the MoE routing imbalance (top-6-of-256 over 12 ranks => E[max]
# ~2 experts vs mean 0.5). That one is structural; only batched decode amortizes it.
export LLM_THREADS=${LLM_THREADS:-47}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
export DS4F_NUMA=${DS4F_NUMA:-1}
export DS4F_CMGS=${DS4F_CMGS:-4}

# ---- model: BASE (fp8 experts) ----
export DS4F_MODEL=${DS4F_MODEL:-ds4fbase}
export DS4F_REAL=${DS4F_REAL:-1}
export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4fbase}
export DS4F_EXACT=${DS4F_EXACT:-1}          # real DeepSeek math (RoPE/YaRN, MQA window+sink, ...)
export DS4F_TIERB2=${DS4F_TIERB2:-1}        # stateful compressor/indexer decode path (implies EXACT)
export DS4F_MHC=${DS4F_MHC:-1}              # exact 4-stream manifold hyper-connections
export DS4F_HC_PAR=${DS4F_HC_PAR:-1}        # bit-exact mHC parallelization
export DS4F_HC_RMSPAR=${DS4F_HC_RMSPAR:-1}  # bit-exact mHC RMS fold

# ---- MANDATORY dense tensor-parallel stack (see the memory table above) ----
export DS4F_TP_ATTN=${DS4F_TP_ATTN:-1}      # shard n_heads (wq_b + attn); o-proj partial via ar_cb
export DS4F_TP_OPROJ=${DS4F_TP_OPROJ:-1}    # row-shard wo_a by o_inter
export DS4F_TP_WOB=${DS4F_TP_WOB:-1}        # col-shard wo_b (FP8; pairs with TP_OPROJ)
export DS4F_TP_SHARED=${DS4F_TP_SHARED:-1}  # col-shard shared-expert up/gate
export DS4F_TP_HEAD=${DS4F_TP_HEAD:-1}      # vocab-shard lm_head
export DS4F_TP_EMBED=${DS4F_TP_EMBED:-1}    # vocab-shard embedding (bit-exact: sum via ar_cb)

# ---- dense stays FP8: the bf16 predequant (+6 GB) does not fit at EP=12 with FP8 experts ----
export DS4F_FP8_BF16=${DS4F_FP8_BF16:-0}
export DS4F_PREFILL_BATCH=${DS4F_PREFILL_BATCH:-0}   # needs FP8_BF16=1 -> must stay 0 here

# ---- PREFILL: batch it through the verify path. ON by default as of 2026-07-13 --------------
# Was off because ds4f_forward_verify was BROKEN (the TP_WOB column-shard bug, f9daca59) and this
# produced fast garbage. Post-fix it is gated: coherent completion AND the prefill argmax is
# IDENTICAL to the token-at-a-time control (361) at every K below.
#
#   K:        1(off)   8      16     32     64     128
#   tok/s:    17.88   24.75  26.47  27.56  27.96  28.31     (+58% at K=128)
#   comm:     20.3%   16.4%  16.0%  15.2%  15.0%  14.9%
#   ar_calls: 6090    844    500    328    242    156
#
# NOTE the mechanism is NOT what the code comment used to claim ("comm / K"): ar_calls falls 39x
# but comm only 20.3% -> 14.9%. Splitting ms/tok, compute drops 44.5 -> 30.0 (-14.5 ms) while comm
# drops 11.3 -> 5.3 (-6.0 ms) -- so ~70% of the win is the dense M=K GEMM replacing K matvecs, not
# the all-reduce elision. It saturates past K~16 because attn/Tier-B2/mHC stay per-position
# (looped, causal) and are the irreducible floor. K=32 takes 54 of the 58 points; the last 4 cost 4x K.
export DS4F_PREFILL_GEMM=${DS4F_PREFILL_GEMM:-1}
export DS4F_PREFILL_K=${DS4F_PREFILL_K:-32}

# ---- DECODE COMPUTE levers. Measure COMPUTE = ms/tok x (1-comm%), NOT tok/s ----------------
# tok/s on this fabric swings 10.2 - 13.9 for the IDENTICAL config (comm 23-60 ms, external
# contention). Compute is rock-stable (+/-0.1 ms) and is the only part we control:
#     Q8 + TP_ATTN=0                       compute 48.3 ms   (tb2prep 12.4, mhc_pre 10.2)
#   + DS4F_CMP_LOCAL=1                     compute 44.3 ms   (tb2prep 12.4 -> 8.3, BIT-EXACT)
#   + DS4F_HC_SVE=1                        compute 36.6 ms   (mhc_pre 10.2 -> 2.4)   = -24%
#
# !! THOSE NUMBERS ARE AT ctx=8 -- this script's DEFAULT synthetic prefill -- and that is NOT a
# representative context. Decode tok/s is a function of context, and NOT monotonically:
#     ctx     8: 22.50 tok/s   (tb2scan  2.73 ms)
#     ctx    64: 18.07 tok/s   (tb2scan 12.80 ms)   <- WORST: serial scalar indexer scan
#     ctx   256: 22.94 tok/s   (tb2scan  0.24 ms)   <- pooled SVE scan: 4x work, 1/53 the time
#     ctx  1024: 22.41 tok/s
# The dip was a real BUG (fixed 2026-07-13): ds4f_index_score fell back to a scalar SINGLE-THREADED
# loop for T < 64 compressed tokens (~ctx < 256) -- i.e. for every short prompt a chat/serving
# workload actually has. See DS4F_IDX_SCAN_MIN in common/ds4f_impl.h. ALWAYS state the context a
# decode number was measured at; a bare "22.45 tok/s" is not a fact about the model.
#
# CMP_LOCAL is BIT-EXACT (reader-local page placement only) -> ON by default below.
# HC_SVE is a REASSOCIATION-class lever (SVE half-row hcmix): NOT bit-exact, ids diverge from
# the baseline trajectory. Validated coherent on base (12n gen: valid quicksort, NaN=0, 12/12
# lockstep) -- but that is one eyeballed completion, not an accuracy gate, so it stays OPT-IN.
# Turn it on with DS4F_HC_SVE=1 if you accept the reassoc class; it is the single biggest
# remaining compute win (mhc_pre -77%).
export DS4F_CMP_LOCAL=${DS4F_CMP_LOCAL:-1}
export DS4F_HC_SVE=${DS4F_HC_SVE:-0}

# ---- bit-exact perf levers (all validated token-identical on ds4f) ----
export DS4F_OPROJ_FUSE=${DS4F_OPROJ_FUSE:-1}
export DS4F_QNR_PAR=${DS4F_QNR_PAR:-1}
export DS4F_TB2ROPE_PAR=${DS4F_TB2ROPE_PAR:-1}
export DS4F_ATTN_SVE=${DS4F_ATTN_SVE:-1}
export DS4F_FLAGBAR=${DS4F_FLAGBAR:-1}

# ---- LONG CONTEXT: leave DS4F_INT8_KV OFF. It COSTS context, it does not save it. ----
# INT8_KV allocates ly->kv_q = max_pos * kv_lora for EVERY layer (ds4f_impl.h ~2040), which DEFEATS
# Tier-B2's KV windowing (the default kv_cache path uses ly->kv_slots -> 128 slots on the 41 sparse
# layers). So it converts a flat O(1) KV into 43*512 = 22 KB/token of ARENA growth. Measured 12n:
#   INT8_KV=1 -> 192k and 256k OOM (SIGKILL).
#   INT8_KV=0 -> arena FLAT at 25.34 GB at ANY ctx; only the compressed caches grow (~1.7 KB/tok);
#                1M ctx fits (caches 1.75 GB).
# For long ctx use the compressed-cache levers instead: DS4F_INT4_CMP=1 DS4F_IDX_INT4=1.
# The ceiling is then PERFORMANCE, not memory: decode 8.47 tok/s @32k -> 1.53 @128k -> 0.25 @1M
# (comm 56% -> 91%, the O(T) indexer scan). Usable range is <=32k; 128k is marginal.
# (DS4F_IDX_REUSE=4 does NOT rescue it: 0.92 vs 1.02 tok/s @256k -- refuted, see a64fx/ds4f.md.)
export DS4F_INT8_KV=${DS4F_INT8_KV:-0}
export DS4F_INT4_CMP=${DS4F_INT4_CMP:-0}
export DS4F_IDX_INT4=${DS4F_IDX_INT4:-0}

# ---- workload ----
export DS4F_PREFILL=${DS4F_PREFILL:-8}
export DS4F_MAXGEN=${DS4F_MAXGEN:-16}
export DS4F_MAXPOS=${DS4F_MAXPOS:-4096}
export DS4F_LAYERS=${DS4F_LAYERS:-0}
export DS4F_PROF=${DS4F_PROF:-1}
export TF_HW_BARRIER=${TF_HW_BARRIER:-1}

# ---- OOM guard: clean _exit(42) before the node dies (protects the co-located session) ----
export DS4F_WARM_RSS_TRACE=${DS4F_WARM_RSS_TRACE:-0}
export DS4F_WARM_MEMAVAIL_STOP_GB=${DS4F_WARM_MEMAVAIL_STOP_GB:-1.5}

echo "=== DS4F-BASE EP harness on $NP node(s) (REAL weights <- $DS4F_STAGE_DIR) ==="
echo "threads=$LLM_THREADS prefill=$DS4F_PREFILL maxgen=$DS4F_MAXGEN max_pos=$DS4F_MAXPOS layers=${DS4F_LAYERS:-43}"
# Print what is ACTUALLY configured. This line used to be a hardcoded
#   echo "experts=FP8(e4m3)  dense=FP8  TP=attn/oproj/wob/shared/head/embed"
# which reported FP8 + full TP no matter what you passed -- so a q8pv/TP_ATTN=0 run looked like a
# stock FP8 run in every log we archived. A banner that cannot be wrong is worth more than a pretty one.
tp=""
for k in ATTN OPROJ WOB SHARED HEAD EMBED; do
    v="DS4F_TP_$k"; [ "${!v}" = "1" ] && tp="$tp$(echo $k | tr 'A-Z' 'a-z')/"
done
echo "experts=${DS4F_EXPERTS:-fp8(e4m3)}  dense=${DS4F_DENSE:-fp8}  TP=${tp:-none}  (bake=${DS4F_BAKE_DIR:-\$HOME/models/<model>-fast})"

make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
make -C "$LLM_DIR" ds4f_ep_runner CC=fcc OPENMP=1 >/dev/null
BIN="$LLM_DIR/build/ds4f_ep_runner"

rm -f ds4f_ep_perf_rank*.txt ds4f_ep_load_rank*.txt ds4f_ep_stderr_rank*.txt ds4f_ep_rank00.txt

if [ "${SKIP_TOPO:-0}" != "1" ]; then
    mpiexec -np "$NP" -vcoordfile "$VCOORD" "$UTOFU_DIR/tofu_topo_helper"
fi
echo "--- tofu topo ($(wc -l < tofu_topo.txt) rows) ---"; cat tofu_topo.txt

echo "--- launching ds4f_ep_runner (NP=$NP) ---"
mpiexec -np "$NP" -vcoordfile "$VCOORD" "$BIN"

echo "=== per-rank load (alloc + first-touch + RSS) ==="; cat ds4f_ep_load_rank*.txt 2>/dev/null
echo "=== per-rank perf (compute / all-reduce comm / GB-s) ==="; cat ds4f_ep_perf_rank*.txt 2>/dev/null
echo "=== rank0 summary ==="; cat ds4f_ep_rank00.txt 2>/dev/null
