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
export LLM_THREADS=${LLM_THREADS:-48}
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

echo "=== DS4F-BASE EP harness on $NP node(s) (REAL fp8-expert weights <- $DS4F_STAGE_DIR) ==="
echo "threads=$LLM_THREADS prefill=$DS4F_PREFILL maxgen=$DS4F_MAXGEN max_pos=$DS4F_MAXPOS layers=${DS4F_LAYERS:-43}"
echo "experts=FP8(e4m3)  dense=FP8  TP=attn/oproj/wob/shared/head/embed"

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
