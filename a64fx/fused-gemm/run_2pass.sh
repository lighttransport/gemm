#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
#PJM --no-check-directory

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/fused-gemm

echo "=== 2-Pass Softmax + P@V Benchmark ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Pin to single core for stable timing
export OMP_NUM_THREADS=1
export FLIB_CNTL_BARRIER_ERR=FALSE

./bench_2pass
