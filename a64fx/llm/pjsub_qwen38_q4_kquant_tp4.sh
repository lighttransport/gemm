#!/bin/bash
# Four-node staging, exact-token, memory, and throughput gate for Q38KQC1.
# Submit from a Fugaku login node:
#   pjsub --no-check-directory a64fx/llm/pjsub_qwen38_q4_kquant_tp4.sh
#PJM -g hp250467
#PJM -L "rscgrp=small,node=4,elapse=03:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=4"
#PJM --llio localtmp-size=48Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/glm53f
LLM=$REPO/a64fx/llm
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf}
TAG=${PJM_JOBID:-manual_$$}
RESULT=$REPO/tmp/qwen38_q4_kquant_tp4_$TAG
STAGE=/local/u14346/codex-research/qwen38-q4-tp4
KQUANT_STAGE=/local/u14346/codex-research/qwen38-q4-tp4-kquant
COMPILER_TMP=/local/u14346/codex-research/compiler-tmp

export PATH=/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin
export PJM_MPI_PROC=4 TP_SIZE=4 TP_STAGE_DIR=$STAGE
export LLM_THREADS=48 OMP_NUM_THREADS=48 OMP_PROC_BIND=spread OMP_PLACES=cores
export OMP_DYNAMIC=false OMP_WAIT_POLICY=active KMP_BLOCKTIME=1
export TP_RAW_PROMPT=1 TP_PROMPT='Explain why deterministic reductions matter.'
export TP_AR_DETERMINISTIC=1 TP_PERF_WARMUP=0 TP_IGNORE_EOS=1
export TP_MAXSEQ=512 TP_DUMP_TOKENS=1
# Q5R/IQ4R retain the established compact-A8 activation quantizer.  Compare
# against that exact compact-weight baseline, not the separate FP32-x kernel.
export TF_KQUANT_A8=1
# Q5R remains diagnostic-only because the full-cache run failed the 256-token
# greedy gate.  The default cache path materializes and attaches exact IQ4R.
export TP_KQUANT_Q5=0
mkdir -p "$COMPILER_TMP"
export TMPDIR=$COMPILER_TMP

mkdir -p "$RESULT"
cd "$LLM"

record_memory() {
    local label=$1
    export Q38_RESULT=$RESULT Q38_MEMORY_LABEL=$label
    mpiexec -np 4 sh -c '
        rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-unknown}}}
        file="$Q38_RESULT/memory_${Q38_MEMORY_LABEL}_rank${rank}.txt"
        printf "host=%s rank=%s " "$(hostname)" "$rank" > "$file"
        awk '\''/^MemAvailable:/{print $1 " " $2 " " $3}'\'' /proc/meminfo >> "$file"'
}

save_run() {
    local name=$1
    mkdir -p "$RESULT/$name"
    cp -f tp_run_*.txt tp_perf_rank*.txt tp_stderr_rank*.txt \
        tp_tokens_rank00.txt "$RESULT/$name/" 2>/dev/null || true
    sha256sum tp_tokens_rank00.txt | tee "$RESULT/$name/token.sha256"
}

make qwen38_tp_stage qwen38_kquant_stage qwen38_kquant_check tp_runner \
    CC=fcc OPENMP=1

record_memory initial
echo "=== compact stage: $(date) ==="
bash ./run_qwen38_q4_tp4.sh stage
record_memory compact

echo "=== kquant stage: $(date) ==="
TP_KQUANT_STAGE_DIR=$KQUANT_STAGE bash ./run_qwen38_q4_tp4.sh kquant-stage
record_memory kquant

echo "=== strict sidecar check: $(date) ==="
export Q38_RESULT=$RESULT Q38_KQUANT_STAGE=$KQUANT_STAGE
mpiexec -np 4 sh -c '
    rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-unknown}}}
    ./build/qwen38_kquant_check "$TP_STAGE_DIR" "$Q38_KQUANT_STAGE" \
        > "$Q38_RESULT/kquant-check-rank${rank}.txt" 2>&1'
record_memory checked

for tokens in 128 256; do
    echo "=== compact decode $tokens: $(date) ==="
    TP_MAXGEN=$tokens bash ./run_qwen38_q4_tp4.sh bench
    save_run compact_$tokens

    echo "=== IQ4R cached decode $tokens: $(date) ==="
    TP_MAXGEN=$tokens TP_KQUANT_STAGE_DIR=$KQUANT_STAGE \
        bash ./run_qwen38_q4_tp4.sh bench
    save_run cached_iq4_$tokens

    cmp -s "$RESULT/compact_$tokens/tp_tokens_rank00.txt" \
           "$RESULT/cached_iq4_$tokens/tp_tokens_rank00.txt"
done

record_memory final
sha256sum "$RESULT"/compact_*/tp_tokens_rank00.txt \
          "$RESULT"/cached_iq4_*/tp_tokens_rank00.txt \
    | tee "$RESULT/token-hashes.txt"
grep -hE 'decode\(|kquant decode cache attached|kquant_stage:' \
    "$RESULT"/cached_iq4_*/tp_run_*.txt \
    "$RESULT"/cached_iq4_*/tp_stderr_rank*.txt \
    | tee "$RESULT/summary.txt"
echo "SENTINEL qwen38_q4_kquant_tp4=OK result=$RESULT $(date)"
