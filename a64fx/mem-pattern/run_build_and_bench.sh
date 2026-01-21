#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -L "elapse=00:10:00"
#PJM -L "freq=2000,eco_state=0"
#PJM -g hp250467
#PJM -j

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/mem-pattern

echo "==============================================="
echo "Memory Access Pattern Analysis for Fused Attn"
echo "==============================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Build
echo "=== Building ==="
make clean
make COMPILER=fcc

if [ ! -f bench_mem_pattern ]; then
    echo "Build failed!"
    exit 1
fi
echo "Build successful!"
echo ""

# Run benchmark
echo "=== Running Benchmark ==="
./bench_mem_pattern -i 1000 -w 100

echo ""
echo "==============================================="
echo "Done at $(date)"
