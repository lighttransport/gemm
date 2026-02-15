#!/bin/bash

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/fp8-conv

# Compile the FP16 path benchmark
echo "Compiling bench_fp16_path..."
fcc -Ofast -Kfast -Nclang -o bench_fp16_path \
    bench_fp16_path.c \
    fp8_kernel_asm.S \
    /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-cmg/micro_kernel_fp16fp32_8x3.S

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Running benchmark..."
./bench_fp16_path
