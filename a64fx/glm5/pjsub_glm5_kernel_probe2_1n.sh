#!/bin/bash
# K0b — native int8-GEMV probe (1 node, ~5 min). qlair is UNRELIABLE for load/pipeline-bound kernels
# (K0 lesson: it mis-ranked the axpy 1.03x vs native 1.93x). The int8 w8a16 GEMV is byte-load +
# dequant-heavy, so its baseline-vs-u2 ranking MUST be confirmed natively. Reports real A64FX ns +
# the qlair-vs-native gap for this kernel class.

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM -j
set -u
REPO=/home/u14346/work/gemm/glm5-1
GLM5="$REPO/a64fx/glm5"; KERN="$GLM5/qlair/kernels"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
WORK="$GLM5/kernel_probe2_run_${PJM_JOBID:-manual}"
mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
export OMP_NUM_THREADS=1
echo "=== GLM5.2 int8-GEMV native probe (1 node) job=${PJM_JOBID:-?} ==="; date; echo "host: $(hostname)"

# native build (real SVE codegen). -O3; the kernel uses explicit intrinsics.
fcc -Nclang -O3 -march=armv8.2-a+sve -fno-math-errno -I "$REPO/common" \
    -o int8_gemv_bench "$KERN/int8_gemv_bench.c" -lm || { echo "FATAL: build"; exit 3; }

echo "--- int8_gemv_bench NATIVE (baseline 8row vs u2 2x-c-unroll; real A64FX ns) ---"
./int8_gemv_bench || echo "WARN: rc=$?"
echo "  qlair reference (uncalibrated): baseline 1027us, u2 1052us (x0.98 = NO win in qlair)."
echo "  If native also shows no win -> int8 GEMV is convert-throughput-bound (near-optimal for w8a16),"
echo "  and the compute lever is the attention dot/axpy only. If native DIFFERS -> trust native (K0 axpy lesson)."
echo "SENTINEL glm5_kernel_probe2_1n=done"; date
