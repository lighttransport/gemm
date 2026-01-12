#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:30:00"
#PJM -g hp250467
#PJM --no-check-directory
#PJM -j
#PJM -S

# Fused FlashAttention V2 Benchmark Runner
# Runs on Fugaku A64FX compute node

set -x

cd $PJM_O_WORKDIR

echo "========================================"
echo "Fused FlashAttention V2 Benchmark"
echo "========================================"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Build if binary doesn't exist
if [ ! -f bench_flash_fused_v2 ]; then
    echo "Building bench_flash_fused_v2..."
    bash build_flash_v2.sh
    echo ""
fi

# Run correctness check (L=64)
echo "=== Correctness Check (L=64) ==="
./bench_flash_fused_v2 64 128 1

# Run performance benchmarks
for L in 1024 2048 4096; do
    echo ""
    echo "=== Benchmark L=$L, head_dim=128 ==="
    ./bench_flash_fused_v2 $L 128 5
done

echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"
