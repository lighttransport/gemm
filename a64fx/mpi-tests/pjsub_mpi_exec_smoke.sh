#!/bin/bash
# Minimal MPI runtime smoke for non-contiguous node allocations.
# Submit with:
#   pjsub -g hp250467 -L "rscgrp=small-s2,node=4,elapse=01:01:00" \
#     --mpi "proc=4" --no-check-directory a64fx/mpi-tests/pjsub_mpi_exec_smoke.sh

#PJM -j
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
set -u

NP=${PJM_MPI_PROC:-1}
echo "=== MPI exec smoke: NP=$NP job=${PJM_JOBID:-?} ==="
date
which mpiexec || true
mpiexec -np "$NP" /bin/hostname || { echo "FATAL: mpiexec hostname"; exit 5; }
echo "=== MPI exec smoke done $(date) ==="
