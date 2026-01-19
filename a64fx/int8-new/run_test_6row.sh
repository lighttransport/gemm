#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building 5-row D=512 kernel..."
fcc -Nclang -O3 -c kernel_fused_d512_5row.S -o kernel_fused_d512_5row.o

echo "Building 6-row D=512 kernel..."
fcc -Nclang -O3 -c kernel_fused_d512_6row.S -o kernel_fused_d512_6row.o

echo "Building test..."
fcc -Nclang -O3 -march=armv8.2-a+sve -o test_6row test_6row.c kernel_fused_d512_5row.o kernel_fused_d512_6row.o

echo ""
echo "Running test..."
./test_6row
