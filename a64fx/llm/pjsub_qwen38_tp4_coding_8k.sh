#!/bin/bash
# Four-node Qwen3.8-27B BF16 TP4 long-context coding quality and throughput gate.
# Submit from a Fugaku login node:
#   pjsub --no-check-directory a64fx/llm/pjsub_qwen38_tp4_coding_8k.sh
#PJM -g hp250467
#PJM -L "rscgrp=small,node=4,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=4"
#PJM --llio localtmp-size=48Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4f
LLM=$REPO/a64fx/llm
MODEL=${MODEL:-/home/u14346/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf}
TAG=${PJM_JOBID:-manual_$$}
RESULT=$REPO/tmp/qwen38_tp4_coding_8k_$TAG
STAGE=/local/u14346/qwen38-bf16-tp4-coding-$TAG
CACHE=$STAGE/cache
PROMPT=$LLM/qwen38_coding_quality_prompt.txt

export PATH=/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin
export PJM_MPI_PROC=4 LLM_THREADS=48 OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_DYNAMIC=false
export TP_STAGE_DIR=$STAGE TP_MAXSEQ=24640 TP_RAW_PROMPT=1
# The supplied CedarDB brief is a realistic code-generation request.  Its
# deterministic repetition supplies a stable approximately-8k-token context;
# record the actual tokenizer count and fail rather than silently benchmarking
# a short context if the tokenizer/model changes.
export TP_PROMPT_FILE=$PROMPT TP_PROMPT_REPEAT=32 TP_PROMPT_TOKEN_LIMIT=8192
export TP_IGNORE_EOS=1 TP_AR_DETERMINISTIC=1 TP_AR_A2A=0
export TP_CACHE_DIR=$CACHE TP_CACHE_TAG=coding8k TP_CACHE_SHARED=0
export TP_AR_BATCH=512 TF_PREFILL_KEEP_POOL_OFF=1
mkdir -p "$RESULT"
cd "$LLM"

save() {
    local name=$1
    mkdir -p "$RESULT/$name"
    cp -f tp_run_*.txt tp_perf_rank*.txt tp_stderr_rank*.txt tp_tokens_rank00.txt \
        "$RESULT/$name/" 2>/dev/null || true
}

make tp_runner qwen38_tp_stage CC=fcc OPENMP=1

echo "=== stage: $(date) ==="
bash ./run_qwen38_bf16_tp4.sh stage

# Isolated prefill optimization measurement.  P-odd changes the resident
# matrix layout, so it is deliberately a prefill-only process and never the
# quality/decode process below.
echo "=== prefill p-odd: $(date) ==="
TP_PREFILL_PREPACK_PODD=1 TF_PODD=1 TP_PREFILL_ONLY=1 TP_PREFILL_GEMM=1 \
    TP_SYNTH_TOKENS=8192 TP_MAXGEN=0 bash ./run_qwen38_bf16_tp4.sh prefill
save prefill_podd_8192

# Build an exact row-major cache for the quality gate.  This path also reports
# real long-context prefill speed, unlike the isolated packed-layout probe.
echo "=== prefill exact/cache: $(date) ==="
TP_PREFILL_REAL_PROMPT=1 TP_PREFILL_ONLY=1 TP_PREFILL_GEMM=1 TP_CACHE_SAVE=1 TP_MAXGEN=0 \
    bash ./run_qwen38_bf16_tp4.sh prefill
save prefill_exact_8k
grep -h 'P=8[0-9][0-9][0-9].*prefill' tp_run_*.txt >/dev/null || {
    echo "FATAL: coding prompt did not tokenize to approximately 8k tokens" >&2; exit 1;
}

# Exactness check: K=0 and K=5 start from the same 8k cache.  Their first 512
# greedy IDs must be identical before the long MTP run is accepted.
echo "=== K0 parity control: $(date) ==="
TP_PREFILL_GEMM=0 TP_CACHE_LOAD=1 TP_CACHE_SAVE=0 TP_MAXGEN=512 \
    TP_PERF_WARMUP=0 TP_DUMP_TOKENS=1 bash ./run_qwen38_bf16_tp4.sh bench
save k0_512
mv tp_tokens_rank00.txt "$RESULT/k0_512.tokens"

echo "=== K5 coding generation: $(date) ==="
TP_PREFILL_GEMM=0 TP_CACHE_LOAD=1 TP_CACHE_SAVE=0 TP_SPEC_K=5 TP_MTP_BATCH=1 \
    TP_MAXGEN=16384 TP_PERF_WARMUP=0 TP_DUMP_TOKENS=1 \
    TP_GENERATED_TEXT_FILE="$RESULT/cedardb.cpp" \
    bash ./run_qwen38_bf16_tp4.sh mtp-sustained
save k5_16384
mv tp_tokens_rank00.txt "$RESULT/k5_16384.tokens"
cmp -s <(sed -n '1,512p' "$RESULT/k0_512.tokens") \
       <(sed -n '1,512p' "$RESULT/k5_16384.tokens")

# Quality is independently observable: the requested output is one C++20
# file, with an explicit final sentinel.  Compilation is a strict minimum;
# model-provided tests may then be run by the caller with its chosen directory.
grep -q '// CEDARDB_COMPLETE' "$RESULT/cedardb.cpp"
g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic "$RESULT/cedardb.cpp" -o "$RESULT/cedardb_test"
sha256sum "$RESULT"/k0_512.tokens "$RESULT"/k5_16384.tokens "$RESULT/cedardb.cpp" \
    | tee "$RESULT/sha256.txt"
grep -hE 'prefill\(|prefill-only:|decode\(|MTP greedy match|MTP accepted' \
    "$RESULT"/k5_16384/tp_run_*.txt "$RESULT"/prefill_exact_8k/tp_run_*.txt \
    | tee "$RESULT/summary.txt"
echo "SENTINEL qwen38_tp4_coding_8k=done result=$RESULT $(date)"
