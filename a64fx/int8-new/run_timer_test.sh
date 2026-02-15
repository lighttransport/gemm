#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building timer test..."
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d256.S -o kernel_d256_test.o
fcc -Nclang -O3 -march=armv8.2-a+sve -o test_timer test_timer.c kernel_d256_test.o

echo ""
echo "Running timer test..."
./test_timer
