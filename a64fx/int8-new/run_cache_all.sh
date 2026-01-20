#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building comprehensive cache optimization benchmarks..."

# Compile D=256 kernels
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d256.S -o kernel_d256_baseline.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d256_prefetch.S -o kernel_d256_prefetch.o

# Compile D=512 kernels
fcc -Nclang -O3 -c kernel_ffn_6row_gemm.S -o kernel_d512_baseline.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d512_prefetch.S -o kernel_d512_prefetch.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d512_ktile.S -o kernel_d512_ktile.o

# Compile benchmark
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_cache_all bench_cache_all.c \
    kernel_d256_baseline.o kernel_d256_prefetch.o \
    kernel_d512_baseline.o kernel_d512_prefetch.o kernel_d512_ktile.o

echo ""
echo "Running comprehensive cache optimization benchmark..."
echo ""
./bench_cache_all
