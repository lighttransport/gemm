#!/bin/bash
set -e

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building Qwen3-style SwiGLU FFN benchmark..."

echo "Compiling D=256 kernel..."
fcc -Nclang -O3 -c kernel_ffn_6row_gemm_d256.S -o kernel_ffn_6row_gemm_d256.o

echo "Compiling D=512 kernel..."
fcc -Nclang -O3 -c kernel_ffn_6row_gemm.S -o kernel_ffn_6row_gemm_d512.o

echo "Compiling activations..."
fcc -Nclang -O3 -march=armv8.2-a+sve -c activation.c -o activation.o

echo "Compiling Qwen3 FFN..."
fcc -Nclang -O3 -march=armv8.2-a+sve -c ffn_qwen3.c -o ffn_qwen3.o

echo "Compiling benchmark..."
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_qwen3_ffn bench_qwen3_ffn.c \
    ffn_qwen3.o kernel_ffn_6row_gemm_d256.o kernel_ffn_6row_gemm_d512.o activation.o

echo ""
./bench_qwen3_ffn
