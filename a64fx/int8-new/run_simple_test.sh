#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building simple test..."
fcc -Nclang -O3 -march=armv8.2-a+sve -c layernorm_int8.c
fcc -Nclang -O3 -march=armv8.2-a+sve -o test_simple_ln test_simple_ln.c layernorm_int8.o -lm

echo "Running simple test..."
./test_simple_ln
