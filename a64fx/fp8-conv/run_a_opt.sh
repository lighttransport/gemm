#!/bin/bash
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:05:00"
#PJM -g hp250467
#PJM --no-check-directory
#PJM -j
#PJM -o run_a_opt.out

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/fp8-conv

# Compile with clang for SVE intrinsics
fcc -Ofast -march=armv8.2-a+sve -fno-strict-aliasing -o bench_a_opt bench_a_opt.c

echo "=== Running A Conversion Optimization Benchmark ==="
./bench_a_opt
