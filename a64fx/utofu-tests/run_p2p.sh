#!/bin/bash
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#
# uTofu point-to-point latency / bandwidth characterization (no MPI in the bench).
# Measures, all driven from rank 0:
#   - latency vs torus distance for Put (ping-pong), Get (one-sided), armw8, over
#     all peers on the 12-node 2x3x2 in-unit torus, and
#   - Put/Get bandwidth vs message size to one peer (default the farthest), using
#     a pipelined async window of W outstanding ops.
#
# Submit on an N-node allocation (2 <= N <= 12), e.g. 12 nodes (2x3x2 in-unit):
#   pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=12,elapse=00:10:00" \
#         --mpi "shape=12,proc=12" run_p2p.sh
# Or 2 nodes: node=2 ... --mpi "shape=2,proc=2". Interactive: rscgrp=int --interact.
#
# Step 1 (MPI helper) discovers each node's Tofu coordinates -> tofu_topo.txt.
# Step 2 (pure uTofu bench) is launched by mpiexec only for node placement; the
# binary makes zero MPI calls. NP defaults to the allocation's node count
# ($PJM_NODE); override by exporting NP. stdout is swallowed by mpiexec, so read
# rank 0's p2p_log_<coords>.txt for the result tables.
#
# Tunables (env): P2P_MAXBYTES P2P_WINDOW P2P_LAT_BYTES P2P_ITERS P2P_WARMUP
#                 P2P_BW_ITERS P2P_BW_WARMUP P2P_BW_PEER  (see p2p_latbw_bench.c).
set -e

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

NP=${NP:-${PJM_NODE:-2}}
if [ "$NP" -gt 12 ]; then
    echo "warning: NP=$NP exceeds the validated ceiling of 12 nodes" >&2
fi
echo "running on $NP node(s)"

echo "=== step 1: discover topology (MPI helper) ==="
mpiexec -np "$NP" ./tofu_topo_helper
cat tofu_topo.txt

echo "=== step 2: point-to-point latency/bandwidth (no MPI in the binary) ==="
mpiexec -np "$NP" ./p2p_latbw_bench

echo "=== done (see p2p_log_*.txt for rank 0's tables) ==="
