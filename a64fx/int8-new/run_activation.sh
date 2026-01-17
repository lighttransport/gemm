#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,rscgrp=small,node=1,elapse=00:10:00"
#PJM -j
#PJM -S

echo "Date: $(date)"
echo "Host: $(hostname)"

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new

# Build if needed
make bench_activation

# Run the benchmark
./bench_activation
