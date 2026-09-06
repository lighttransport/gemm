#!/bin/bash
# Clean four-node Qwen3.8-27B BF16 decode/MTP gate.  The runner talks to uTofu
# directly; MPI is used only to launch one native process per node.
# Submit from a Fugaku login node:
#   pjsub --no-check-directory a64fx/llm/pjsub_qwen38_bf16_mtp4.sh
#PJM -g hp250467
#PJM -L "rscgrp=small,node=4,elapse=01:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=4"
#PJM --llio localtmp-size=48Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/ds4f
LLM=$REPO/a64fx/llm
JOB_TAG=${PJM_JOBID:-manual_$$}
OUT=$REPO/tmp/qwen38_bf16_mtp4_$JOB_TAG
REP_STAGE=/local/u14346/qwen38-bf16-tp4-$JOB_TAG
SHARD_STAGE=/local/u14346/qwen38-bf16-tp4-nextnshard-$JOB_TAG

export PATH=/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/bin
export PJM_MPI_PROC=4 LLM_THREADS=48 OMP_NUM_THREADS=48
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_DYNAMIC=false
mkdir -p "$OUT"
cd "$LLM"

make tp_runner qwen38_tp_stage CC=fcc OPENMP=1

echo "=== stage replicated NextN: $(date) ==="
TP_NEXTN_SHARD=0 TP_STAGE_DIR=$REP_STAGE \
    bash ./run_qwen38_bf16_tp4.sh stage
echo "=== stage TP4-sharded NextN: $(date) ==="
TP_NEXTN_SHARD=1 TP_STAGE_DIR=$SHARD_STAGE \
    bash ./run_qwen38_bf16_tp4.sh stage-mtp

save_run() {
    tag=$1
    mkdir -p "$OUT/$tag"
    cp -f tp_run_*.txt tp_perf_rank*.txt tp_stderr_rank*.txt \
        tp_tokens_rank00.txt "$OUT/$tag/" 2>/dev/null || true
    sha256sum tp_tokens_rank00.txt | tee "$OUT/$tag/token.sha256"
    grep -hE 'MTP (round|async|greedy|horizons|accepted)|decode\(' \
        tp_run_*.txt | tee "$OUT/$tag/summary.txt" || true
}

echo "=== plain BF16 control: $(date) ==="
TP_NEXTN_SHARD=0 TP_STAGE_DIR=$REP_STAGE TP_MAXGEN=64 TP_PERF_WARMUP=32 \
TP_AR_DETERMINISTIC=1 TP_AR_A2A=0 \
    bash ./run_qwen38_bf16_tp4.sh bench
save_run plain_bf16

echo "=== replicated NextN K5: $(date) ==="
TP_NEXTN_SHARD=0 TP_STAGE_DIR=$REP_STAGE TP_SPEC_K=5 TP_MAXGEN=256 \
TP_AR_DETERMINISTIC=1 TP_AR_A2A=0 \
    bash ./run_qwen38_bf16_tp4.sh mtp-sustained
save_run mtp_k5_replicated

echo "=== sharded NextN K4: $(date) ==="
TP_NEXTN_SHARD=1 TP_STAGE_DIR=$SHARD_STAGE TP_SPEC_K=4 TP_MAXGEN=256 \
TP_AR_DETERMINISTIC=1 TP_AR_A2A=0 \
    bash ./run_qwen38_bf16_tp4.sh mtp-sustained
save_run mtp_k4_sharded

echo "=== asynchronous sharded NextN K4: $(date) ==="
TP_NEXTN_SHARD=1 TP_STAGE_DIR=$SHARD_STAGE TP_SPEC_K=4 TP_MAXGEN=256 \
TP_MTP_ASYNC=1 TP_AR_DETERMINISTIC=1 TP_AR_A2A=0 \
    bash ./run_qwen38_bf16_tp4.sh mtp-sustained
save_run mtp_k4_async

echo "SENTINEL qwen38_bf16_mtp4=done out=$OUT $(date)"
