#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_larger_tiles bench_larger_tiles.c -lm
./bench_larger_tiles
