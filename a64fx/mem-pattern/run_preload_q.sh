#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_preload_q bench_preload_q.c -lm
./bench_preload_q
