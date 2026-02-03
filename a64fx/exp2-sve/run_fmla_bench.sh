#!/bin/bash
cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/exp2-sve

echo "=== Fused exp2 + FMLA GEMM Benchmark ==="
echo "Testing different Nc values (sequence length)"
echo ""

# Test with typical attention tile sizes
for Nc in 32 64 128 256; do
    echo "========================================"
    echo "Nc = $Nc"
    echo "========================================"
    ./bench_fmla $Nc 1000
    echo ""
done

echo "=== Simple bench test ==="
./bench_fmla_simple
