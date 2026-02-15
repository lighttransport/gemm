#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/fp8-quant

echo "Building comparison benchmark..."
make clean
make bench_compare

echo ""
echo "Running comparison benchmark..."
./bench_compare
