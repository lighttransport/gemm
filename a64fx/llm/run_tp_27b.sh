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
export TP_MAXGEN=${TP_MAXGEN:-32}
export QWEN27B_LOCAL_DIR=${QWEN27B_LOCAL_DIR:-/local/qwen36/27b}
# long-context perf sweep knobs (per-token decode cost vs ctx -> tp_curve_rank00.txt)
[ -n "$TP_MAXSEQ" ]     && export TP_MAXSEQ
[ -n "$TP_IGNORE_EOS" ] && export TP_IGNORE_EOS

if [ "${SKIP_STAGE:-0}" != "1" ]; then
    chmod +x "$LLM_DIR/stage_gguf_shards.sh"
    mpiexec -np "$NP" "$LLM_DIR/stage_gguf_shards.sh" "$MODEL" "$QWEN27B_LOCAL_DIR"
fi
MODEL_LOCAL="$QWEN27B_LOCAL_DIR/$(basename "$MODEL")"
if [ ! -f "$MODEL_LOCAL" ]; then
    echo "model local missing after stage: $MODEL_LOCAL" >&2
    exit 1
fi

echo "=== TP-only 27B decode on $NP node(s) ==="
echo "model=$MODEL_LOCAL  source=$MODEL  threads=$LLM_THREADS  maxgen=$TP_MAXGEN"

make -C "$UTOFU_DIR" tofu_topo_helper >/dev/null
if [ -n "$PROF" ]; then
    make -C "$LLM_DIR" tp_runner_prof CC=fcc OPENMP=1 >/dev/null
    BIN="$LLM_DIR/build/tp_runner_prof"
else
    make -C "$LLM_DIR" tp_runner CC=fcc OPENMP=1 >/dev/null
    BIN="$LLM_DIR/build/tp_runner"
fi

rm -f tp_run_*.txt tp_load_rank*.txt tp_perf_rank*.txt tp_stderr_rank*.txt
mpiexec -np "$NP" "$UTOFU_DIR/tofu_topo_helper"
echo "--- tofu topo ---"; cat tofu_topo.txt
echo "--- launching $(basename "$BIN") (NP=$NP) ---"
mpiexec -np "$NP" "$BIN" "$MODEL_LOCAL"
echo "=== per-rank load/shard dims ==="; cat tp_load_rank*.txt 2>/dev/null
echo "=== per-rank decode perf (compute/comm/GB-s) ==="; cat tp_perf_rank*.txt 2>/dev/null
echo "=== rank0 decode output ==="; cat $(ls -t tp_run_*.txt | head -1) 2>/dev/null
echo "=== rank0 tp_slice + pool profile (captured stderr) ==="
grep -E "tp_slice: rank|tf_pool:|numa: phase 1 done" tp_stderr_rank00.txt 2>/dev/null
