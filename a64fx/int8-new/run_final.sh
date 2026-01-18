#!/bin/bash
cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

echo "Building FINAL optimized attention..."

fcc -Nclang -O3 -march=armv8.2-a+sve -c -o fused_attention_final.o fused_attention_final.c
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_final bench_final.c \
    fused_attention_opt.o fused_attention_final.o softmax_sve.o exp2_int.o \
    kernel_qkt_6x4_2x.o kernel_pv_int8_opt.o -lm

if [ $? -eq 0 ]; then
    echo ""
    ./bench_final
fi
