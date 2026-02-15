#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "node=1"
#PJM -L "elapse=00:10:00"
#PJM -L "freq=2000,eco_state=0"
#PJM -g hp250467
#PJM -j

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/mem-pattern

echo "=============================================="
echo "L1 vs L2 Cache Performance Test"
echo "=============================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Run L2 benchmark
echo "=== Running L2 Pattern Benchmark ==="
./bench_l2_pattern 1000 100

echo ""
echo "=== Running Original L1 Benchmark ==="
./bench_mem_pattern -i 1000

echo ""
echo "=============================================="
echo "L2 Performance Analysis"
echo "=============================================="
echo ""
echo "A64FX Memory Hierarchy:"
echo "  L1D: 64KB, 11 cycles, 128 byte/cycle load"
echo "  L2:  8MB/CMG, 27-37 cycles, ~42.6 byte/cycle/core"
echo ""
echo "Key observations:"
echo "  - L1_small (N=64):  ~34KB working set -> L1 resident"
echo "  - L1_edge  (N=128): ~70KB working set -> L1 spill to L2"
echo "  - L2_small (N=256): ~140KB -> L2 resident"
echo "  - L2_medium(N=512): ~280KB -> L2 resident"
echo "  - L2_large (N=1024):~560KB -> L2 resident"
echo ""
echo "Expected behavior:"
echo "  - L1 configs: high efficiency (approaching 128 B/cy)"
echo "  - L2 configs: limited to ~42.6 B/cy"
echo "  - Transition visible at L1_edge configuration"
echo ""
echo "=============================================="
echo "Test complete at $(date)"
echo "=============================================="
