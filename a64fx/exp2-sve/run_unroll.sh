#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:05:00"
#PJM --no-check-directory

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/exp2-sve

# Build
bash build_unroll.sh

echo ""
echo "=== Running benchmark ==="
./bench_unroll

# Also test with larger sizes to stress memory bandwidth
echo ""
echo "=== Larger sizes ==="
./bench_unroll 16777216  # 16M elements = 64 MB
