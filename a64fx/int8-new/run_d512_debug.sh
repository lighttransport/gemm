#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building D=512 debug benchmark..."
fcc -Nclang -O3 -c kernel_ffn_6row_gemm.S -o kernel_d512_base.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d512_prefetch.S -o kernel_d512_pref.o
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d512_ktile.S -o kernel_d512_ktile.o
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_d512_debug bench_d512_debug.c \
    kernel_d512_base.o kernel_d512_pref.o kernel_d512_ktile.o

echo ""
./bench_d512_debug
