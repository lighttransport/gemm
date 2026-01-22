#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_fused_interleaved bench_fused_interleaved.c -lm
./bench_fused_interleaved
