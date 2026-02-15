#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_fused_optimized_final bench_fused_optimized_final.c -lm
./bench_fused_optimized_final
