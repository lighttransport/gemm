#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:05:00"
#PJM --no-check-directory

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/exp2-sve

echo "=== Building ==="
make clean
make bench_flash_tiled 2>&1
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "=== FlashAttention-style LD1RW Benchmark ==="
echo "Date: $(date)"
echo ""

# Run with different Nc values
for Nc in 32 64 128; do
    echo "--- Nc=$Nc ---"
    ./bench_flash_tiled $Nc
    echo ""
done
