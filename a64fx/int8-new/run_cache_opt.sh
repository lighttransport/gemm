#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building cache optimization benchmarks..."

# Compile kernels
fcc -Nclang -O3 -c kernel_ffn_6row_gemm.S -o kernel_d512_baseline.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d512_prefetch.S -o kernel_d512_prefetch.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d512_ktile.S -o kernel_d512_ktile.o

# Compile benchmark
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_cache_opt bench_cache_opt.c \
    kernel_d512_baseline.o kernel_d512_prefetch.o kernel_d512_ktile.o

echo ""
echo "Running cache optimization benchmark..."
echo ""
./bench_cache_opt
