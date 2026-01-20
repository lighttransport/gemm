#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building SVE test..."
fcc -Nclang -O3 -march=armv8.2-a+sve -o test_sve_basic test_sve_basic.c

echo "Running SVE test..."
./test_sve_basic
