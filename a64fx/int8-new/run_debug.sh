#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building debug benchmark..."
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d256.S -o kernel_d256_base.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d256_prefetch.S -o kernel_d256_pref.o
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_debug bench_debug.c \
    kernel_d256_base.o kernel_d256_pref.o

echo ""
./bench_debug
