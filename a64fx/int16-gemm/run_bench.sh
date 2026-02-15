#!/bin/bash

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int16-gemm

echo "=== INT16 SDOT GEMM Benchmark ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo ""

# Run C intrinsic version
echo ">>> Running C Intrinsic Version <<<"
./bench_int16_sdot 1000000
echo ""

# Run ASM kernel version
echo ">>> Running ASM Kernel Version <<<"
./bench_int16_sdot_asm 1000000
echo ""

echo "=== Done ==="
