#!/bin/bash
# Gemma4 31B Q4_0 single-node A64FX smoke run.
#
# Defaults are chosen for 32GB HBM nodes: resident hugepage-backed weights,
# A64FX SVE Q4_0 kernels, and Gemma4 KV stored as F16. Override
# MODEL/PROMPT/MAXSEQ/MAXGEN from the environment for longer tests.
set -euo pipefail

cd "$(dirname "$0")"

MODEL=${MODEL:-$HOME/models/gemma4/31b-qat/gemma-4-31B_q4_0-it.gguf}
PROMPT=${PROMPT:-"The capital of France is"}
MAXSEQ=${MAXSEQ:-65536}
MAXGEN=${MAXGEN:-16}
LLM_THREADS=${LLM_THREADS:-48}
OUT=${OUT:-gemma4_31b_q4.out}
TIMEOUT=${TIMEOUT:-900}
NUMA_BIND=${NUMA_BIND:-1}

RUN_PREFIX=()
if [[ "$NUMA_BIND" != "0" ]] && command -v numactl >/dev/null 2>&1; then
    RUN_PREFIX=(numactl -C12-59 -m4-7)
fi

make -C "$PWD" CC="${CC:-fcc}" OPENMP=1

: > "$OUT"
echo "=== Gemma4 31B Q4_0 single-node $(date +%H:%M:%S) ===" | tee -a "$OUT"
echo "model=$MODEL" | tee -a "$OUT"
echo "max_seq=$MAXSEQ max_gen=$MAXGEN llm_threads=$LLM_THREADS" | tee -a "$OUT"
echo "numa_bind=$NUMA_BIND numa_distribute=${NUMA_DISTRIBUTE:-1} numa_budget_gb=${NUMA_CMG_BUDGET_GB:-7}" | tee -a "$OUT"

set +e
timeout "$TIMEOUT" env \
    XOS_MMM_L_HPAGE_TYPE="${XOS_MMM_L_HPAGE_TYPE:-hugetlbfs}" \
    XOS_MMM_L_PAGING_POLICY="${XOS_MMM_L_PAGING_POLICY:-demand:demand:demand}" \
    XOS_MMM_L_ARENA_FREE="${XOS_MMM_L_ARENA_FREE:-2}" \
    NO_MMAP="${NO_MMAP:-1}" \
    NUMA_DISTRIBUTE="${NUMA_DISTRIBUTE:-1}" \
    NUMA_N_CMGS="${NUMA_N_CMGS:-4}" \
    NUMA_CMG_BUDGET_GB="${NUMA_CMG_BUDGET_GB:-7}" \
    NUMA_ALIGNMENT="${NUMA_ALIGNMENT:-2097152}" \
    OMP_NUM_THREADS="$LLM_THREADS" \
    TF_NO_PANEL="${TF_NO_PANEL:-0}" \
    TF_NO_BF16_PV="${TF_NO_BF16_PV:-1}" \
    TF_KV_DTYPE="${TF_KV_DTYPE:-f16}" \
    TF_PREFILL_GEMM="${TF_PREFILL_GEMM:-1}" \
    TF_DUMP_LOGITS="${TF_DUMP_LOGITS:-1}" \
    /usr/bin/time -v "${RUN_PREFIX[@]}" ./build/llm_runner "$MODEL" \
        --prompt "$PROMPT" --max-seq "$MAXSEQ" --max-gen "$MAXGEN" \
        --llm-threads "$LLM_THREADS" \
    2>&1 | tee -a "$OUT"
rc=${PIPESTATUS[0]}
set -e

echo "RC=$rc" | tee -a "$OUT"
echo "GEMMA4_31B_Q4_DONE $(date +%H:%M:%S)" | tee -a "$OUT"
exit "$rc"
