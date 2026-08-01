#!/bin/bash
# uTofu init probe for non-contiguous allocations. Submit examples:
#   pjsub -g hp250467 --no-check-directory -L "rscgrp=small-s2,node=12,elapse=01:01:00" \
#     --mpi "proc=12" a64fx/utofu-tests/pjsub_utofu_init_probe.sh

#PJM -j
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
set -u

REPO=/home/u14346/work/gemm/glm5-1
UTOFU="$REPO/a64fx/utofu-tests"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-${UTOFU_PROBE_NP:-1}}
WORK="$UTOFU/run_utofu_${PJM_JOBID:-manual}_${NP}r"
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2

echo "=== uTofu init probe: NP=$NP job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK"
date

make -C "$UTOFU" utofu_init_probe CC=fcc >/dev/null || exit 3
make -C "$UTOFU" tofu_put_demo CC=fcc >/dev/null || exit 3
make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3

export UTOFU_PROBE_DIR="$WORK"
echo "--- pure uTofu init ($(date)) ---"
mpiexec -np "$NP" "$UTOFU/utofu_init_probe"
pure_rc=$?
echo "pure_utofu_rc=$pure_rc logs=$(ls "$WORK"/utofu_probe_rank*.log 2>/dev/null | wc -l)"

echo "--- MPI + uTofu topo helper ($(date)) ---"
rm -f tofu_topo.txt
mpiexec -np "$NP" "$UTOFU/tofu_topo_helper"
topo_rc=$?
topo_rows=$(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0)
echo "topo_helper_rc=$topo_rc topo_rows=$topo_rows"
[ -f tofu_topo.txt ] && sed -n '1,40p' tofu_topo.txt

put_rc=99
if [ "$topo_rc" -eq 0 ]; then
    echo "--- pure uTofu ring put ($(date)) ---"
    rm -f demo_log_*.txt
    mpiexec -np "$NP" "$UTOFU/tofu_put_demo"
    put_rc=$?
    echo "put_demo_rc=$put_rc logs=$(ls "$WORK"/demo_log_*.txt 2>/dev/null | wc -l)"
    grep -h "received from peer rank .* OK\\|FATAL\\|VERIFY FAILED\\|SELF-CHECK FAILED" demo_log_*.txt 2>/dev/null | sed -n '1,80p'
fi

echo "=== uTofu init probe done $(date) pure_rc=$pure_rc topo_rc=$topo_rc put_rc=$put_rc ==="
[ "$pure_rc" -eq 0 ] && [ "$topo_rc" -eq 0 ] && [ "$put_rc" -eq 0 ]
