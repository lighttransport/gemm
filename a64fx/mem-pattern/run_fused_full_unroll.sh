#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_fused_full_unroll bench_fused_full_unroll.c -lm
./bench_fused_full_unroll
