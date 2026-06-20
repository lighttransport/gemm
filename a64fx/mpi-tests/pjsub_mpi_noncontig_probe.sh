#!/bin/bash
# MPI-only non-contiguous allocation probe. Submit with explicit resources, e.g.
#   pjsub -g hp250467 -L "rscgrp=small-s2,node=12,elapse=00:20:00" \
#     --mpi "proc=12" --no-check-directory a64fx/mpi-tests/pjsub_mpi_noncontig_probe.sh

#PJM -j
set -u

REPO=/home/u14346/work/gemm/glm5-1
MPI_DIR="$REPO/a64fx/mpi-tests"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-${MPI_PROBE_NP:-1}}
WORK="$MPI_DIR/run_${PJM_JOBID:-manual}_${NP}r"
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2

echo "=== MPI noncontig probe: NP=$NP job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK"
date

make -C "$MPI_DIR" mpi_noncontig_probe MPICC=mpiclang >/dev/null || exit 3

MPI_PROBE_ITERS=${MPI_PROBE_ITERS:-20} \
MPI_PROBE_BYTES=${MPI_PROBE_BYTES:-1048576} \
MPI_PROBE_ALLTOALL_EACH=${MPI_PROBE_ALLTOALL_EACH:-4096} \
  mpiexec -np "$NP" "$MPI_DIR/mpi_noncontig_probe" || { echo "FATAL: mpi probe"; exit 5; }

echo "=== MPI noncontig probe done $(date) ==="
