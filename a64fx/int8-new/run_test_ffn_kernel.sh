#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building FFN GEMM kernel test..."
fcc -Nclang -O3 -c kernel_ffn_6row_gemm.S -o kernel_ffn_6row_gemm.o
fcc -Nclang -O3 -march=armv8.2-a+sve -o test_ffn_kernel test_ffn_kernel.c kernel_ffn_6row_gemm.o

echo ""
./test_ffn_kernel
