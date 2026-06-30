#!/bin/bash
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:05:00"
#PJM -L "freq=2000,eco_state=0"
#PJM --mpi "proc=1"
#PJM -j
cd ~/work/gemm/glm5-1/a64fx/glm5
OMP_NUM_THREADS=1 ./int8ct
