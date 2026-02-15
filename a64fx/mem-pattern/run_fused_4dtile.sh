#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_fused_4dtile bench_fused_4dtile.c -lm
./bench_fused_4dtile
