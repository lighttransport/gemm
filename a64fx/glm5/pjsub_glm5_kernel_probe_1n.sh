#!/bin/bash
# K0 — qlair calibration probe (1 node, ~5 min). Runs the attention-decode kernel microbench
# (qlair/kernels/attn_decode_bench.c) NATIVELY on one A64FX node with fcc, so its real ns can be
# compared against the qlair cycle-mode ns to set --friction (raise qlair sim accuracy to <=10%).
# Also re-runs bdecode_kern_bench for the decode GEMV shapes. NO staging, NO MPI, ~1-2 nh.

#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
GLM5="$REPO/a64fx/glm5"; KERN="$GLM5/qlair/kernels"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
WORK="$GLM5/kernel_probe_run_${PJM_JOBID:-manual}"
mkdir -p "$WORK" || exit 2; cd "$WORK" || exit 2
export OMP_NUM_THREADS=1

echo "=== GLM5.2 kernel calibration probe (1 node) job=${PJM_JOBID:-?} ==="; date
echo "host: $(hostname)  cntfrq: check bench output (A64FX arch counter)"

# Native A64FX build (real SVE codegen; -O3, no -fno-tree-vectorize needed off-qlair).
fcc -Nclang -O3 -march=armv8.2-a+sve -fno-math-errno -o attn_decode_bench \
    "$KERN/attn_decode_bench.c" -lm || { echo "FATAL: build bench"; exit 3; }
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE -fopenmp \
    -I "$REPO/common" -o bdecode_kern_bench "$GLM5/bdecode_kern_bench.c" -lm || echo "WARN: kernbench build"

echo "--- attn_decode_bench NATIVE (real A64FX ns; compare vs qlair -n 8G cycle ns) ---"
./attn_decode_bench || echo "WARN: bench rc=$?"
echo "  qlair cycle-mode reference (2026-07-03, uncalibrated): full-loop dot1 742us dot4 550us (1.35x);"
echo "  pure dot dot1 312us dot4 157us (1.98x) dot8 135us (2.31x); axpy4 1.03x (no win)."
echo "  friction := tune so qlair ns ~= native ns above (QLAIR_VERIFICATION methodology, target <=10%)."

echo "--- bdecode_kern_bench NATIVE (decode GEMV shapes -> BW_NODE / decode_sim) ---"
[ -x ./bdecode_kern_bench ] && ./bdecode_kern_bench || echo "WARN: kernbench skipped"

echo "SENTINEL glm5_kernel_probe_1n=done"; date
