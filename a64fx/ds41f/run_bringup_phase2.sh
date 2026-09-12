#!/bin/sh
# Do not edit this script while it is running: dash/bash can retain offsets
# into an open script while waiting for a foreground MPI program.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores
export DS41F_STAGE_ROOT="/local/$USER/ds41f-$PJM_JOBID"
export DS41F_META_FILE="$1/engram_meta.bin"
mkdir -p "$1/phase2"
cd "$1/phase2"
mpiexec -np 12 sh "$here/run_rank_tests.sh"
mpiexec -np 12 "$here/ds41f_utofu_test" --staged-experts "$DS41F_STAGE_ROOT"
mpiexec -np 12 sh "$here/run_dense_rank.sh"
echo "PHASE2 PASS tests and dense ownership staged"
