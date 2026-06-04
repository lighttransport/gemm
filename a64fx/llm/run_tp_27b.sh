#!/bin/bash
# Tensor-parallel decode of Qwen3.6-27B BF16 (2-file split GGUF, ~54GB) across
# N A64FX nodes. TP-ONLY: there is NO single-node reference pass — the full 27B
# (~54GB) does NOT fit one 32GB node, which is the entire reason for the TP
# rewrite. Correctness is already byte-validated on smaller hybrids (tp_sim_test
# + 0.8B multinode); here the questions are (1) does it FIT per node and
# (2) what tok/s vs the 6 tok/s PP baseline.
#
# Run inside the existing N-node allocation (NO pjsub):
#   NP=4 ./run_tp_27b.sh            # TP=4: 6Q/1KV (shard) /4352ff,         dt_rank 48->12
#   NP=6 ./run_tp_27b.sh            # TP=6: 4Q/4KV (REPLICATED) /2912-2848ff, dt_rank 48->8
#   NP=8 ./run_tp_27b.sh            # TP=8: 3Q/1KV (shard) /2176ff,         dt_rank 48->6
#   PROF=1 NP=4 ./run_tp_27b.sh     # use tp_runner_prof (-DTF_POOL_PROFILE decode breakdown)
#   NP=11 ./run_tp_27b.sh           # TP=11 (unbalanced row shard, first node)
#   NP=12 ./run_tp_27b.sh           # TP=12 (balanced row/col shard) /2048ff
# 27B: n_heads=24 n_kv=4 n_ff=17408 dt_rank=48.
#  - Q heads divide at 4/6/8; SSM dt at 4/6/8; vocab shard handles any remainder.
#  - KV heads (4) divide at 4 but NOT 6 -> TP=6 REPLICATES all 4 KV heads (full KV cache,
#    Q-heads sharded). n_ff (=2^10*17) has no factor of 3 -> TP=6 uses a balanced
#    mult-of-16 FFN split (ranks 0-4: 2912, rank 5: 2848). Both are byte-no-ops at 4/8.
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"
UTOFU_DIR="$LLM_DIR/../utofu-tests"

# Optional rank-placement file. When set (e.g. VCOORD=vcoord_no0.txt to exclude
# the relative-(0,0,0) node so a co-located interactive session is not OOM-killed),
# every mpiexec below is pinned to exactly those nodes, and the topo step is NOT
# allowed to re-place ranks onto the excluded node. Empty => legacy behavior.
MPI_PLACE=()
if [ -n "${VCOORD:-}" ]; then
    [ -f "$VCOORD" ] || { echo "VCOORD file not found: $VCOORD" >&2; exit 1; }
    MPI_PLACE=(-vcoordfile "$VCOORD")
    echo "[run_tp_27b] pinning all mpiexec to -vcoordfile $VCOORD"
fi
# Skip the topo-regeneration step (keep an externally-prepared tofu_topo.txt, e.g.
# tofu_topo.txt.no0 already copied into place). SKIP_TOPO=1 to use as-is.
SKIP_TOPO=${SKIP_TOPO:-0}

NP=${NP:-${PJM_NODE:-4}}
QWEN27B_NODES=${QWEN27B_NODES:-$NP}
QWEN27B_SOURCE=${QWEN27B_SOURCE:-$HOME/models/qwen36/27b}
QWEN27B_PACKAGE=${QWEN27B_PACKAGE:-$HOME/models/qwen36/27b/${QWEN27B_NODES}nodes}
MODEL=${MODEL:-$QWEN27B_PACKAGE/Qwen3.6-27B-BF16-00001-of-00002.gguf}

if [ "${QWEN27B_PREPARE:-1}" != "0" ]; then
    chmod +x "$LLM_DIR/prepare_qwen36_27b_12nodes.sh"
    "$LLM_DIR/prepare_qwen36_27b_12nodes.sh" "$QWEN27B_SOURCE" "$QWEN27B_PACKAGE"
    if [ "$MODEL" = "$QWEN27B_SOURCE/$(basename "$MODEL")" ]; then
        MODEL="$QWEN27B_PACKAGE/$(basename "$MODEL")"
    fi
fi

export GGUF_LAZY_MMAP=1
export LLM_THREADS=${LLM_THREADS:-48}
export OMP_NUM_THREADS=${LLM_THREADS}
export TP_PROMPT=${TP_PROMPT:-"Hello, who are you?"}
export TP_SYNTH_TOKENS=${TP_SYNTH_TOKENS:-0}
export TP_MAXGEN=${TP_MAXGEN:-32}
export TP_MAXSEQ=${TP_MAXSEQ:-0}
export QWEN27B_LOCAL_DIR=${QWEN27B_LOCAL_DIR:-/local/qwen36/27b}
export TP_PREFILL_ONLY=${TP_PREFILL_ONLY:-0}
export TP_PREFILL_GEMM=${TP_PREFILL_GEMM:-1}
export TP_PREFILL_ATTN_TILE=${TP_PREFILL_ATTN_TILE:-1}
export TP_PREFILL_ATTN_TILE_Q=${TP_PREFILL_ATTN_TILE_Q:-32}
export TP_PREFILL_ATTN_TILE_K=${TP_PREFILL_ATTN_TILE_K:-256}
export TP_PREFILL_ATTN_TILE_ALGO=${TP_PREFILL_ATTN_TILE_ALGO:-0}
export TP_PREFILL_ATTN_TILE_STACK_BYTES=${TP_PREFILL_ATTN_TILE_STACK_BYTES:-2097152}
export TP_PREFILL_NULL_GEMM=${TP_PREFILL_NULL_GEMM:-0}
export TP_PREFILL_SSM_GEMM=${TP_PREFILL_SSM_GEMM:-1}
export TP_PREFILL_SSM_BLOCK=${TP_PREFILL_SSM_BLOCK:-1}
export TP_PREFILL_SSM_NULL_FINISH=${TP_PREFILL_SSM_NULL_FINISH:-0}
export TP_PREFILL_FFN_ALLREDUCE=${TP_PREFILL_FFN_ALLREDUCE:-1}
export TP_AR_BF16=${TP_AR_BF16:-1}
export TP_CACHE_SHARED=${TP_CACHE_SHARED:-auto}
export TP_CACHE_DIR=${TP_CACHE_DIR:-"${QWEN27B_LOCAL_DIR}/tp_cache/${QWEN27B_NODES}"}
export TP_CACHE_TAG=${TP_CACHE_TAG:-"tp_np${NP}"}
export TP_CACHE_LOAD=${TP_CACHE_LOAD:-0}
export TP_CACHE_SAVE=${TP_CACHE_SAVE:-0}
export TP_DRY=${TP_DRY:-0}
export TP_DRY_PREFILL=${TP_DRY_PREFILL:-$TP_DRY}
export TP_DRY_DECODE=${TP_DRY_DECODE:-0}
export TP_DRY_WORK_REPS=${TP_DRY_WORK_REPS:-0}
export TP_DRY_AR_STEPS=${TP_DRY_AR_STEPS:-0}
export TP_DRY_TOKEN_STEP=${TP_DRY_TOKEN_STEP:-1}
# long-context perf sweep knobs (per-token decode cost vs ctx -> tp_curve_rank00.txt)
[ -n "$TP_MAXSEQ" ]     && export TP_MAXSEQ
[ -n "$TP_IGNORE_EOS" ] && export TP_IGNORE_EOS
# Forward debug/validation knobs to the ranks (mpiexec forwards EXPORTED env only;
# without these, env-prefixed TF_ATTN_PP / TP_DUMP_TOKENS never reach the runner).
export TF_ATTN_PP=${TF_ATTN_PP:-}
export TP_DUMP_TOKENS=${TP_DUMP_TOKENS:-0}
# Per-rank lm_head isolation dump: post-final-norm hidden + local logit shard
# (tp_lmhead_rank<NN>.txt). TP_DUMP_LMHEAD=N dumps the first N decode steps.
export TP_DUMP_LMHEAD=${TP_DUMP_LMHEAD:-0}
# Per-layer NaN localization: scan m->x / mixer-out each layer, write first
# NaN-introducing layer/op to tp_nantrace_rank<NN>.txt. Requires the per-op
# forward path (TP_DECODE_PERSIST=0) since the persistent worker isn't traced.
export TP_NAN_TRACE=${TP_NAN_TRACE:-0}
export TP_NAN_TRACE_STEPS=${TP_NAN_TRACE_STEPS:-2}
export TP_DECODE_PERSIST=${TP_DECODE_PERSIST:-1}

if [ "${SKIP_STAGE:-0}" != "1" ]; then
    chmod +x "$LLM_DIR/stage_gguf_shards.sh"
    mpiexec -np "$NP" "${MPI_PLACE[@]}" "$LLM_DIR/stage_gguf_shards.sh" "$MODEL" "$QWEN27B_LOCAL_DIR"
fi
MODEL_LOCAL="$QWEN27B_LOCAL_DIR/$(basename "$MODEL")"
# The /local existence check only makes sense when the launcher node is itself a
# run node. With VCOORD set we deliberately exclude this (launcher) node from the
# run, so its /local is empty by design — staging happened on the run nodes only.
# Each rank verifies its own /local at load time, so skip the launcher-side check.
if [ -z "${VCOORD:-}" ] && [ ! -f "$MODEL_LOCAL" ]; then
    echo "model local missing after stage: $MODEL_LOCAL" >&2
    exit 1
fi

echo "=== TP-only 27B decode on $NP node(s) ==="
echo "model=$MODEL_LOCAL  source=$MODEL  threads=$LLM_THREADS  maxgen=$TP_MAXGEN"
echo "knobs prefill_only=$TP_PREFILL_ONLY prefill_gemm=$TP_PREFILL_GEMM null_gemm=$TP_PREFILL_NULL_GEMM attn_tile=$TP_PREFILL_ATTN_TILE q=$TP_PREFILL_ATTN_TILE_Q k=$TP_PREFILL_ATTN_TILE_K algo=$TP_PREFILL_ATTN_TILE_ALGO stack=$TP_PREFILL_ATTN_TILE_STACK_BYTES ssm_gemm=$TP_PREFILL_SSM_GEMM ssm_block=$TP_PREFILL_SSM_BLOCK ssm_null=$TP_PREFILL_SSM_NULL_FINISH ffn_allreduce=$TP_PREFILL_FFN_ALLREDUCE ar_bf16=$TP_AR_BF16 cache_shared=$TP_CACHE_SHARED"

make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
if [ -n "$PROF" ]; then
    make -C "$LLM_DIR" tp_runner_prof CC=fcc OPENMP=1 >/dev/null
    BIN="$LLM_DIR/build/tp_runner_prof"
else
    make -C "$LLM_DIR" tp_runner CC=fcc OPENMP=1 >/dev/null
    BIN="$LLM_DIR/build/tp_runner"
fi

rm -f tp_run_*.txt tp_load_rank*.txt tp_perf_rank*.txt tp_stderr_rank*.txt tp_lmhead_rank*.txt tp_nantrace_rank*.txt
if [ "$SKIP_TOPO" != "1" ]; then
    mpiexec -np "$NP" "${MPI_PLACE[@]}" "$UTOFU_DIR/tofu_topo_helper"
else
    echo "[run_tp_27b] SKIP_TOPO=1: using existing tofu_topo.txt"
fi
echo "--- tofu topo ---"; cat tofu_topo.txt
echo "--- launching $(basename "$BIN") (NP=$NP) ---"
mpiexec -np "$NP" "${MPI_PLACE[@]}" "$BIN" "$MODEL_LOCAL"
echo "=== per-rank load/shard dims ==="; cat tp_load_rank*.txt 2>/dev/null
echo "=== per-rank decode perf (compute/comm/GB-s) ==="; cat tp_perf_rank*.txt 2>/dev/null
echo "=== rank0 decode output ==="; cat $(ls -t tp_run_*.txt | head -1) 2>/dev/null
echo "=== rank0 tp_slice + pool profile (captured stderr) ==="
grep -E "tp_slice: rank|tf_pool:|numa: phase 1 done" tp_stderr_rank00.txt 2>/dev/null
if [ "${TP_DUMP_LMHEAD:-0}" != "0" ]; then
    echo "=== per-rank lm_head dump (hidden + local logit shard, step 0) ==="
    for f in $(ls tp_lmhead_rank*.txt 2>/dev/null | sort); do
        echo "--- $f ---"; sed -n '1,4p' "$f"
    done
fi
if [ "${TP_NAN_TRACE:-0}" != "0" ]; then
    echo "=== per-rank NaN trace: FIRST line where nan>0 (step 0 full forward) ==="
    for f in $(ls tp_nantrace_rank*.txt 2>/dev/null | sort); do
        first=$(grep -m1 -E "nan=[1-9]" "$f" || true)
        echo "--- $f --- ${first:-<no NaN in trace>}"
    done
fi
