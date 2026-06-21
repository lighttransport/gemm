#!/bin/bash
# One-node kernel-only FP8 f32-scale matvec validation/benchmark.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=1,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
GLM5="$REPO/a64fx/glm5"
COMMON="$REPO/common"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12}

cd "$GLM5" || exit 2
echo "=== GLM5.2 FP8 kernel-only test job=${PJM_JOBID:-?} threads=$OMP_NUM_THREADS ==="
date
test -x "$GLM5/glm5_fp8_kernel_test" || { echo "FATAL: missing prebuilt $GLM5/glm5_fp8_kernel_test"; exit 3; }
ROWS=${ROWS:-8192} COLS=${COLS:-6144} REPS=${REPS:-5} "$GLM5/glm5_fp8_kernel_test" || exit 4
echo "=== kernel-only done $(date) ==="
