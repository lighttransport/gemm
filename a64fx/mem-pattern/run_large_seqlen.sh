#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_large_seqlen bench_large_seqlen.c -lm
./bench_large_seqlen
