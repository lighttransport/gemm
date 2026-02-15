#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_ultimate_opt bench_ultimate_opt.c -lm
./bench_ultimate_opt
