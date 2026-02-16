#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "node=1"
#PJM --mpi "max-proc-per-node=1"
#PJM -L "elapse=00:10:00"
#PJM -x PJM_LLIO_GFSCACHE=/vol0006:/vol0005
#PJM -g hp250467
#PJM -s

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=12
export DEBUG_TIMING=1

./bench_fp16_gemm 8192 512
