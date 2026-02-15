#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_k_unroll_pipeline bench_k_unroll_pipeline.c -lm
./bench_k_unroll_pipeline
