#!/bin/sh
# Execute inside the existing allocation. Wait at most one hour for all
# twelve stagers, then launch one rank/node. Does not cancel or restart staging.
set -eu
test "$#" = 2 || { echo "usage: $0 staging_log_dir results_dir" >&2; exit 2; }
staging=$1
results=$2
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$results"
attempt=0
while test "$attempt" -lt 360; do
    completed=0
    rank=0
    while test "$rank" -lt 12; do
        if grep -q "^STAGE PASS rank=$rank " "$staging/stage.rank$rank.log"; then
            completed=$((completed+1))
        fi
        rank=$((rank+1))
    done
    if test "$completed" -eq 12; then break; fi
    attempt=$((attempt+1))
    sleep 10
done
test "$completed" -eq 12 || { echo "staging incomplete after one hour" >&2; exit 1; }
# Allow the original MPI launcher to reap the final Python process.
sleep 5
cd "$results"
export OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores
export DS41F_STAGE_ROOT="/local/$USER/ds41f-$PJM_JOBID"
export DS41F_META_FILE="$staging/../engram_meta.bin"
mpiexec -np 12 sh "$here/run_rank_tests.sh"
exec mpiexec -np 12 "$here/ds41f_utofu_test" --staged-experts "$DS41F_STAGE_ROOT"
