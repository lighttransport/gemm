#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building INT8 kernel efficiency benchmark..."

fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d256.S -o kernel_ffn_6row_gemm_d256.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm.S -o kernel_ffn_6row_gemm_d512.o
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_kernel_efficiency bench_kernel_efficiency.c \
    kernel_ffn_6row_gemm_d256.o kernel_ffn_6row_gemm_d512.o

echo ""
./bench_kernel_efficiency
