#!/bin/bash
# uTofu point-to-point latency/bandwidth on non-contiguous allocations.
# Submit examples:
#   pjsub -g hp250467 --no-check-directory -L "rscgrp=small-s2,node=48,elapse=01:01:00" \
#     --mpi "proc=48" a64fx/utofu-tests/pjsub_p2p_noncontig.sh

#PJM -j
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
set -u

REPO=/home/u14346/work/gemm/glm5-1
UTOFU="$REPO/a64fx/utofu-tests"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"

NP=${PJM_MPI_PROC:-${P2P_NP:-1}}
WORK="$UTOFU/run_p2p_${PJM_JOBID:-manual}_${NP}r"
mkdir -p "$WORK" || exit 2
cd "$WORK" || exit 2

echo "=== uTofu p2p noncontig: NP=$NP job=${PJM_JOBID:-?} ==="
echo "workdir=$WORK"
date

make -C "$UTOFU" tofu_topo_helper MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || exit 3
make -C "$UTOFU" p2p_latbw_bench CC=fcc >/dev/null || exit 3

rm -f tofu_topo.txt p2p_log_*.txt p2p_rank0.log
mpiexec -np "$NP" "$UTOFU/tofu_topo_helper" || { echo "FATAL: topo helper"; exit 4; }
topo_rows=$(grep -vc '^#' tofu_topo.txt 2>/dev/null || echo 0)
echo "topo_rows=$topo_rows"
[ "$topo_rows" -eq "$NP" ] || { echo "FATAL: topo rows mismatch"; exit 4; }
sed -n '1,40p' tofu_topo.txt

export P2P_ITERS=${P2P_ITERS:-1000}
export P2P_WARMUP=${P2P_WARMUP:-100}
export P2P_BW_ITERS=${P2P_BW_ITERS:-120}
export P2P_BW_WARMUP=${P2P_BW_WARMUP:-12}
export P2P_MAXBYTES=${P2P_MAXBYTES:-4194304}
export P2P_WINDOW=${P2P_WINDOW:-8}
export P2P_LAT_BYTES=${P2P_LAT_BYTES:-8}

echo "bench knobs: iters=$P2P_ITERS warmup=$P2P_WARMUP bw_iters=$P2P_BW_ITERS maxbytes=$P2P_MAXBYTES window=$P2P_WINDOW"
mpiexec -np "$NP" "$UTOFU/p2p_latbw_bench" || { echo "FATAL: p2p bench"; exit 5; }

rank0_log=$(grep -l "=== uTofu point-to-point latency / bandwidth ===" p2p_log_*.txt 2>/dev/null | head -1)
if [ -n "$rank0_log" ]; then
    cp "$rank0_log" p2p_rank0.log
    echo "rank0_log=$rank0_log"
    grep -E "nodes=|lat:|peer  dist|^[[:space:]]*[0-9]+[[:space:]]+[0-9]+|-- bandwidth|^[[:space:]]*[0-9]+KiB|^[[:space:]]*[0-9]+MiB" p2p_rank0.log | sed -n '1,220p'
else
    echo "FATAL: rank0 p2p log not found"
    exit 6
fi

echo "=== uTofu p2p noncontig done $(date) ==="
