#!/bin/bash
#PJM -g hp250467
#PJM -L "freq=2000,eco_state=0,retention_state=0,rscgrp=small-s2,node=2x4x2:torus,elapse=00:10:00"
#PJM --mpi "proc=16"
#PJM -j
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
set -eu

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

REPO=${REPO:-"$HOME/work/gemm/ds4p"}
UTOFU_DIR="$REPO/a64fx/utofu-tests"
NP=${NP:-${PJM_MPI_PROC:-16}}

cd "$UTOFU_DIR"

rm -f tofu_topo.txt mpi_hostid_rank_*.txt

echo "=== pjsub mpi hostid test ==="
echo "job=${PJM_JOBID:-unset}"
echo "subjob=${PJM_SUBJOBID:-unset}"
echo "shell_host=$(hostname)"
echo "pwd=$PWD"
echo "np=$NP"
env | sort | grep -E '^(PJM|PMI|PMIX|PLEXEC)_' || true

echo "=== build tofu_topo_helper ==="
make tofu_topo_helper

echo "=== mpiexec topology discovery ==="
mpiexec -np "$NP" ./tofu_topo_helper

echo "=== tofu_topo.txt ==="
cat tofu_topo.txt

echo "=== mpiexec rank/host discovery ==="
mpiexec -np "$NP" /bin/sh -c '
rank=${PMIX_RANK:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-unknown}}}
host=$(hostname)
line="MPI_HOSTID rank=${rank} host=${host} PMIX_RANK=${PMIX_RANK:-unset} PMI_RANK=${PMI_RANK:-unset} PJM_SUBJOBID=${PJM_SUBJOBID:-unset}"
echo "$line"
printf "%s\n" "$line" > "mpi_hostid_rank_${rank}.txt"
'

echo "=== per-rank host files ==="
for f in mpi_hostid_rank_*.txt; do
    [ -f "$f" ] || continue
    cat "$f"
done

echo "SENTINEL mpi_hostid_16n=OK"
