#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
#PJM -g hp250467
#PJM -o dgemm_sector.out
#PJM -e dgemm_sector.err

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/sector-cache

echo "=== Building DGEMM Sector Cache Benchmark ==="

# Copy reference kernel
cp ../ref/dgemm.kernel.s .

# Build
fcc -Nclang -Ofast -march=armv8.2-a+sve -o bench_dgemm_sector \
    bench_dgemm_sector.c dgemm.kernel.s dgemm_no_sector.s -lm

if [ ! -f bench_dgemm_sector ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "=== Running DGEMM Sector Cache Comparison ==="
echo ""

# Test with different sizes
echo "--- Small size (fits in L1) ---"
./bench_dgemm_sector 64 80 64

echo ""
echo "--- Medium size (A tile fits, B streams) ---"
./bench_dgemm_sector 128 160 256

echo ""
echo "--- Large size (stress test) ---"
./bench_dgemm_sector 256 160 512

echo ""
echo "=== Done ==="
