#!/bin/bash
#PJM -L "node=1"
#PJM -L "rscgrp=small"
#PJM -L "elapse=00:05:00"
#PJM -L "freq=2000,eco_state=0"
#PJM -g hp250467
#PJM -j

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/mem-pattern

echo "Building memory access pattern analysis..."
echo "Date: $(date)"
echo ""

make clean
make COMPILER=fcc

echo ""
if [ -f bench_mem_pattern ]; then
    echo "Build successful!"
    ls -la bench_mem_pattern
else
    echo "Build failed!"
    exit 1
fi
