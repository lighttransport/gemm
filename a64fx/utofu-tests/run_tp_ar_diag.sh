#!/bin/bash
# Launch tp_ar_diag_bench on an NP-node subset of the current allocation, EXCLUDING
# the relative-(0,0,0) node (the login/claude node — never place a rank there).
# Regenerates tofu_topo.txt for exactly the placed nodes, then runs the bench.
#
#   ./run_tp_ar_diag.sh            # NP=4, robust default (TP_AR_ROBUST=1)
#   NP=2 ./run_tp_ar_diag.sh
#   TP_AR_ROBUST=0 ./run_tp_ar_diag.sh   # robust-off A/B
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"

NP=${NP:-4}
EXCLUDE=${EXCLUDE:-0,0,0}
VCOORD=${VCOORD:-vcoord_diag.txt}

SX=${PJM_MPI_SHAPE_X:-${PJM_NODE_X:-2}}
SY=${PJM_MPI_SHAPE_Y:-${PJM_NODE_Y:-3}}
SZ=${PJM_MPI_SHAPE_Z:-${PJM_NODE_Z:-2}}
: > "$VCOORD"; n=0
for ((x=0; x<SX; x++)); do for ((y=0; y<SY; y++)); do for ((z=0; z<SZ; z++)); do
    [ "$x,$y,$z" = "$EXCLUDE" ] && continue
    echo "($x,$y,$z)" >> "$VCOORD"; n=$((n+1))
done; done; done
[ "$n" -lt "$NP" ] && { echo "shape ${SX}x${SY}x${SZ} minus ($EXCLUDE) = $n < NP=$NP" >&2; exit 1; }
head -n "$NP" "$VCOORD" > "$VCOORD.tmp" && mv "$VCOORD.tmp" "$VCOORD"
echo "[diag] placing $NP ranks (excl $EXCLUDE):"; sed 's/^/    /' "$VCOORD"

# regenerate topo for exactly these nodes, then run the bench (same placement)
mpiexec -vcoordfile "$VCOORD" -np "$NP" ./tofu_topo_helper
# forward the tp_allreduce.h env knobs the bench/communicator read
export TP_AR_ROBUST=${TP_AR_ROBUST:-1}
export TP_AR_BF16=${TP_AR_BF16:-0}
export DIAG_COUNT DIAG_ITERS DIAG_WARMUP DIAG_EVICT DIAG_SKEWRK
mpiexec -vcoordfile "$VCOORD" -np "$NP" ./tp_ar_diag_bench
echo "[diag] done; rank0 log:"; ls -t tp_ar_diag_*.txt | head -1
