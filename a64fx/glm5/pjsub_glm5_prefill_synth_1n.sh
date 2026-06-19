#!/bin/bash
# Fast synthetic-token prefill smoke for GLM-5.2 chunked MLA prefill.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=1,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
LLM="$REPO/a64fx/llm"
UTOFU="$REPO/a64fx/utofu-tests"
GLM5="$REPO/a64fx/glm5"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-1}
WORK="$GLM5/prefill_run_${PJM_JOBID:-manual}_synth1n"
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2

export GLM5_REAL=${GLM5_REAL:-0}
export GLM5_LAYERS=${GLM5_LAYERS:-4}
export GLM5_EXPERTS=${GLM5_EXPERTS:-8}
export GLM5_MAXPOS=${GLM5_MAXPOS:-256}
export GLM5_PREFILL_SYNTH=${GLM5_PREFILL_SYNTH:-64}
export GLM5_PCHUNK=${GLM5_PCHUNK:-16}
export GLM5_PREFILL_ONLY=1
export GLM5_MSA=${GLM5_MSA:-0}
export LLM_THREADS=${LLM_THREADS:-12}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$LLM_THREADS}

echo "=== GLM5.2 synthetic prefill 1n: layers=$GLM5_LAYERS experts=$GLM5_EXPERTS synth=$GLM5_PREFILL_SYNTH pchunk=$GLM5_PCHUNK job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK"
date

make -C "$UTOFU" tofu_topo_helper >/dev/null || exit 3
make -C "$LLM" glm5_ep_runner CC=fcc OPENMP=1 >/dev/null || exit 3

mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" || exit 3
mpiexec -np "$NP" "$LLM/build/glm5_ep_runner" || { echo "FATAL: prefill smoke"; exit 5; }

echo "--- rank0 log ---"
cat glm5_ep_rank00.txt 2>/dev/null
echo "=== synthetic prefill done $(date) ==="
