#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building LayerNorm/RMSNorm test..."
make -f Makefile.layernorm clean
make -f Makefile.layernorm

echo ""
echo "Running tests..."
./test_layernorm
