#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:05:00"
#PJM --no-check-directory
#PJM -j

set -x

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/exp2-sve

# Build
make

echo "=== SVE FEXPA exp2 Kernel Benchmark ==="
echo ""

# Run benchmark with different sizes
echo "--- 1M elements ---"
./bench_exp2 1048576 100

echo ""
echo "--- 4M elements ---"
./bench_exp2 4194304 50

echo ""
echo "--- 16M elements ---"
./bench_exp2 16777216 20

echo "Done!"
