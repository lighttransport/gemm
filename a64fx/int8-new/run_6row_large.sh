#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building 5-row D=512 kernel..."
fcc -Nclang -O3 -c kernel_fused_d512_5row.S -o kernel_fused_d512_5row.o

echo "Building 6-row D=512 kernel..."
fcc -Nclang -O3 -c kernel_fused_d512_6row.S -o kernel_fused_d512_6row.o

echo "Building benchmark..."
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_6row_large bench_6row_large.c \
    kernel_fused_d512_5row.o kernel_fused_d512_6row.o

echo ""
./bench_6row_large
