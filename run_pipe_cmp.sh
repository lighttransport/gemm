#!/bin/bash
#PJM -L "rscunit=rscunit_ft01"
#PJM -L "rscgrp=small"
#PJM -L "node=1"
#PJM -L "elapse=00:05:00"
#PJM --mpi "proc=1"
#PJM -j
#PJM -S

echo "=== Pipelined Kernel Comparison ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/int8-new
echo "Testing baseline, pipeV2, and deep (4x unroll) kernels..."
./bench_pipe_cmp
