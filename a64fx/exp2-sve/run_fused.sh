#!/bin/bash
#
# Run fused GEMM+exp2 benchmark on A64FX
#

echo "=== Fused GEMM + exp2 Benchmark ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Run with different K values
for K in 64 128 256; do
    echo "--- K=$K ---"
    ./bench_fused $K 1000
    echo ""
done
