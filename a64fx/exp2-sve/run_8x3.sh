#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:05:00"
#PJM --no-check-directory

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/exp2-sve

# Build
./build_8x3.sh

echo "=========================================="
echo "Two-pass FlashAttention: exp2 + 8x3 GEMM"
echo "=========================================="
echo ""

# Different Nc (K) values with D=48 (1 tile)
for Nc in 64 128 256 512; do
    echo "=========================================="
    echo "Nc=$Nc, D=48"
    echo "=========================================="
    ./bench_twopass_8x3 $Nc 48 2000
    echo ""
done

echo "=========================================="
echo "Nc=256, D=96 (2 tiles)"
echo "=========================================="
./bench_twopass_8x3 256 96 2000
echo ""

echo "=========================================="
echo "Nc=512, D=96 (typical FlashAttention)"
echo "=========================================="
./bench_twopass_8x3 512 96 2000
