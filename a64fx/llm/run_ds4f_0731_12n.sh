#!/bin/bash
# DeepSeek-V4-Flash-0731 real-weight runner on all 12 nodes.
#
# This is deliberately separate from the historical 11-node launcher: the
# 0731 model has 48 source shards and this path makes the EP size, topology,
# status directory, and stage directory explicit and reproducible.
set -euo pipefail

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"
UTOFU_DIR="$LLM_DIR/../utofu-tests"
cd "$LLM_DIR"

NP=${NP:-12}
LAST=${LAST:-0,0,0}
VCOORD=${VCOORD:-vcoord_ds4f_0731_12n.txt}
RESULT_DIR=${RESULT_DIR:-$LLM_DIR/runs/ds4f-0731-${PJM_JOBID:-manual-$$}}
STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f-0731-${PJM_JOBID:-manual}}
DS4F_PROFILE=${DS4F_PROFILE:-exact}

if [ "${PJM_MPI_PROC:-$NP}" -ne "$NP" ]; then
    echo "expected a ${NP}-process interactive allocation (PJM_MPI_PROC=${PJM_MPI_PROC:-unset})" >&2
    exit 2
fi
[ -f "$VCOORD" ] || {
    SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
    SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
    SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
    : > "$VCOORD"
    for ((x=0; x<SX; x++)); do
        for ((y=0; y<SY; y++)); do
            for ((z=0; z<SZ; z++)); do
                [ "$x,$y,$z" = "$LAST" ] || echo "($x,$y,$z)" >> "$VCOORD"
            done
        done
    done
    echo "($LAST)" >> "$VCOORD"
    head -n "$NP" "$VCOORD" > "$VCOORD.tmp"
    mv "$VCOORD.tmp" "$VCOORD"
}

mkdir -p "$RESULT_DIR"
VCOORD=$(realpath "$VCOORD")
RESULT_DIR=$(realpath "$RESULT_DIR")

if [ "$DS4F_PROFILE" = fast ]; then
    DS4F_TP_SHARED=${DS4F_TP_SHARED:-1}
    DS4F_TP_SHARED_FULL=${DS4F_TP_SHARED_FULL:-1}
    DS4F_COMM_MODE=${DS4F_COMM_MODE:-2d}
    TP_AR_BF16=${TP_AR_BF16:-1}
    TP_AR_ROBUST=${TP_AR_ROBUST:-2}
else
    DS4F_TP_SHARED=${DS4F_TP_SHARED:-0}
    DS4F_TP_SHARED_FULL=${DS4F_TP_SHARED_FULL:-0}
    DS4F_COMM_MODE=${DS4F_COMM_MODE:-flat}
    TP_AR_BF16=${TP_AR_BF16:-0}
    TP_AR_ROBUST=${TP_AR_ROBUST:-1}
fi

# K3-compatible runtime defaults: one free core for progress/interactive noise,
# robust bootstrap, bounded TCQ polling, and durable status files.  These are
# all overrideable for A/B runs.
export LLM_THREADS=${LLM_THREADS:-47}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}
export OMP_DYNAMIC=false
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
export DS4F_NUMA=${DS4F_NUMA:-1}
export DS4F_CMGS=${DS4F_CMGS:-4}
export DS4F_COMM_ROBUST=${DS4F_COMM_ROBUST:-1}
export DS4F_COMM_POLL_SPINS=${DS4F_COMM_POLL_SPINS:-1}
export TP_AR_ROBUST
export TP_AR_POLL_SPINS=${TP_AR_POLL_SPINS:-8}
export TP_AR_BF16
export TP_AR_A2A=${TP_AR_A2A:-0}
export TP_AR_A2A_MAX=${TP_AR_A2A_MAX:-4096}
export DS4F_COMM_MODE
export DS4F_COMM_2D_A=${DS4F_COMM_2D_A:-4}
export DS4F_REQUIRE_NODES="$NP"
export DS4F_RUN_TAG=${DS4F_RUN_TAG:-$(basename "$RESULT_DIR")}
export DS4F_STATUS_DIR="$RESULT_DIR"

# 0731 model/runtime contract.
export DS4F_MODEL=""
export DS4F_MODEL_DIR=${DS4F_MODEL_DIR:-$HOME/models/ds4f-0731}
export DS4F_NSHARDS=48
export DS4F_STAGE_DIR="$STAGE_DIR"
export DS4F_EP_SIZE="$NP"
export DS4F_REAL=${DS4F_REAL:-1}
export DS4F_EXACT=${DS4F_EXACT:-1}
export DS4F_TIERB2=${DS4F_TIERB2:-1}
export DS4F_MHC=${DS4F_MHC:-1}
export DS4F_HC_PAR=${DS4F_HC_PAR:-1}
export DS4F_HC_RMSPAR=${DS4F_HC_RMSPAR:-1}
export DS4F_CMP_LOCAL=${DS4F_CMP_LOCAL:-1}
export DS4F_HC_SVE=${DS4F_HC_SVE:-0}
export DS4F_MV_FUSE=${DS4F_MV_FUSE:-1}
export DS4F_OPROJ_FUSE=${DS4F_OPROJ_FUSE:-1}
export DS4F_QNR_PAR=${DS4F_QNR_PAR:-1}
export DS4F_TB2ROPE_PAR=${DS4F_TB2ROPE_PAR:-1}
export DS4F_ATTN_SVE=${DS4F_ATTN_SVE:-1}
export DS4F_FLAGBAR=${DS4F_FLAGBAR:-1}

# Target preset: predequantize the dense FP8 tensors to pair-interleaved BF16,
# then repack the dominant dense projections to W8A8 SDOT.  This is the
# measured fastest real-weight decode representation and also enables the
# batched Q8 GEMM path used by chunked mHC/Tier-B2 prefill.
export DS4F_FP8_BF16=${DS4F_FP8_BF16:-1}
export DS4F_BF16_PV=${DS4F_BF16_PV:-1}
export DS4F_Q8_DENSE=${DS4F_Q8_DENSE:-1}
export DS4F_GEMM_TILE_K=${DS4F_GEMM_TILE_K:-4096}
export DS4F_MXFP4_GEMM_TILE=${DS4F_MXFP4_GEMM_TILE:-8}
export DS4F_TP_HEAD=${DS4F_TP_HEAD:-1}
export DS4F_TP_SHARED
export DS4F_TP_SHARED_FULL
export DS4F_PREFILL=${DS4F_PREFILL:-64}
export DS4F_MAXGEN=${DS4F_MAXGEN:-32}
export DS4F_MAXPOS=${DS4F_MAXPOS:-256}
export DS4F_PREFILL_BATCH=${DS4F_PREFILL_BATCH:-0}
export DS4F_PREFILL_VERIFY=${DS4F_PREFILL_VERIFY:-0}

# Fast decode preset: the CSA index cache is quantized once at write time and
# scanned with SVE SDOT. This removes the f32 index scan from the hot
# tb2_prepare path (the measured 0731 bottleneck at short/medium context).
# It is intentionally overrideable: DS4F_IDX_INT8=0 restores exact f32
# selection, while DS4F_IDX_INT4=1 selects the smaller, slightly lossier
# nibble cache. The int8 path has the better selected-set agreement and is
# the default for this performance-oriented 0731 runner.
export DS4F_IDX_INT8=${DS4F_IDX_INT8:-1}
export DS4F_IDX_INT4=${DS4F_IDX_INT4:-0}
export DS4F_CTX_WARM=${DS4F_CTX_WARM:-0}
export DS4F_LAYERS=${DS4F_LAYERS:-0}
export DS4F_PROF=${DS4F_PROF:-1}
export TF_HW_BARRIER=${TF_HW_BARRIER:-1}

echo "DS4F_PROFILE=$DS4F_PROFILE DS4F_COMM_MODE=$DS4F_COMM_MODE DS4F_COMM_2D_A=$DS4F_COMM_2D_A TP_AR_BF16=$TP_AR_BF16 TP_AR_A2A=$TP_AR_A2A DS4F_TP_SHARED=$DS4F_TP_SHARED DS4F_TP_SHARED_FULL=$DS4F_TP_SHARED_FULL" >&2

make -C "$LLM_DIR" ds4f_ep_runner CC=fcc OPENMP=1 >/dev/null
make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
BIN="$LLM_DIR/build/ds4f_ep_runner"

rm -f "$RESULT_DIR"/ds4f_ep_perf_rank*.txt "$RESULT_DIR"/ds4f_ep_load_rank*.txt \
      "$RESULT_DIR"/ds4f_ep_stderr_rank*.txt "$RESULT_DIR"/ds4f_ep_rank00.txt \
      "$RESULT_DIR"/ds4f_status_rank*.txt

if [ "${SKIP_TOPO:-0}" != "1" ]; then
    rm -f "$RESULT_DIR/tofu_topo.txt"
fi

(
    cd "$RESULT_DIR"
    topo_ok=0
    if [ "${SKIP_TOPO:-0}" = "1" ]; then
        [ -s tofu_topo.txt ] && [ "$(grep -vc '^#' tofu_topo.txt)" -eq "$NP" ] && topo_ok=1
    else
        for attempt in 1 2 3; do
            rm -f tofu_topo.txt
            if mpiexec -np "$NP" -vcoordfile "$VCOORD" "$UTOFU_DIR/tofu_topo_helper" \
                    && [ -s tofu_topo.txt ] \
                    && [ "$(grep -vc '^#' tofu_topo.txt)" -eq "$NP" ]; then
                topo_ok=1
                break
            fi
            echo "topology launch attempt $attempt did not produce ${NP} rows; waiting for coordinate release" >&2
            sleep 10
        done
    fi
    [ "$topo_ok" -eq 1 ] || { echo "unable to establish ${NP}-rank topology" >&2; exit 5; }
    echo "DS4F_0731_TOPO_PASS nodes=$NP result=$RESULT_DIR"
    mpiexec -np "$NP" -vcoordfile "$VCOORD" "$BIN" \
        --threads "$LLM_THREADS" --cmgs "$DS4F_CMGS" \
        --prefill "$DS4F_PREFILL" --decode "$DS4F_MAXGEN" \
        --max-pos "$DS4F_MAXPOS" --layers "$DS4F_LAYERS" \
        --ctx-warm "$DS4F_CTX_WARM" --prefill-batch "$DS4F_PREFILL_BATCH" \
        --prefill-verify "$DS4F_PREFILL_VERIFY" \
        --comm-poll-spins "$DS4F_COMM_POLL_SPINS" --comm-robust "$DS4F_COMM_ROBUST"
)

echo "=== DS4F-0731 12-node result: $RESULT_DIR ==="
cat "$RESULT_DIR"/ds4f_ep_load_rank*.txt 2>/dev/null || true
cat "$RESULT_DIR"/ds4f_ep_perf_rank*.txt 2>/dev/null || true
cat "$RESULT_DIR/ds4f_ep_rank00.txt" 2>/dev/null || true
done_count=$(grep -h '^state=done' "$RESULT_DIR"/ds4f_status_rank*.txt 2>/dev/null | wc -l)
echo "DS4F_0731_STATUS_DONE=$done_count/$NP"
[ "$done_count" -eq "$NP" ]
