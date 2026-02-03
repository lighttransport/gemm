#!/bin/bash
set -e

CC=clang-21
CFLAGS="-O3 -march=armv8.2-a+sve -Wall"

echo "=== Building two-pass exp2 + 8x3 GEMM benchmark ==="

# Compile exp2 kernels
$CC $CFLAGS -c exp2_fast.S -o exp2_fast.o
$CC $CFLAGS -c exp2_colmajor.S -o exp2_colmajor.o

# Compile 8x3 GEMM kernel from int8-cmg
$CC $CFLAGS -c ../int8-cmg/micro_kernel_fp32_8x3_unroll4.S -o micro_kernel_fp32_8x3_unroll4.o

# Compile benchmark
$CC $CFLAGS bench_twopass_8x3.c exp2_fast.o exp2_colmajor.o micro_kernel_fp32_8x3_unroll4.o -o bench_twopass_8x3 -lm

echo "Build complete: bench_twopass_8x3"
