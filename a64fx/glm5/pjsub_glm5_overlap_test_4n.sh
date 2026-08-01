#!/bin/bash
# Batch wrapper: run the overlap reproducer on 4 synthetic nodes. (Interactive is preferred — see
# overap-test.md — but this lets you fire-and-forget.) Edit node=/proc=/NP together for 2..12.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=4,elapse=00:20:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=4"
#PJM -j
set -u
export NP=${PJM_MPI_PROC:-4} CORES=48 REPO=/home/u14346/work/gemm/glm5-1
bash "$REPO/a64fx/glm5/glm5_overlap_test.sh"
