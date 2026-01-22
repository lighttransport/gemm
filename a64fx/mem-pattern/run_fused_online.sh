#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_fused_online_softmax bench_fused_online_softmax.c -lm
./bench_fused_online_softmax
