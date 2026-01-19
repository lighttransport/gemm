#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building D=256 GEMM kernel test..."
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d256.S -o kernel_ffn_6row_gemm_d256.o
fcc -Nclang -O3 -march=armv8.2-a+sve -o test_d256_kernel test_d256_kernel.c kernel_ffn_6row_gemm_d256.o

echo ""
./test_d256_kernel
