#!/bin/bash
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#
# MPI-free uTofu multi-node Put demo. Ring topology (peer = rank+1 mod N),
# validated for 2..12 nodes; the code itself is N-agnostic (MAX_NODES=256).
#
# Submit on an N-node allocation (2 <= N <= 12), e.g. 12 nodes:
#   pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=12,elapse=00:10:00" \
#         --mpi "shape=12,proc=12" run_demo.sh
# Or 2 nodes: node=2 ... --mpi "shape=2,proc=2". Interactive: rscgrp=int --interact.
#
# Step 1 (MPI helper) discovers each node's Tofu coordinates -> tofu_topo.txt.
# Step 2 (pure uTofu app) is launched by mpiexec only for node placement; the
# binary itself makes zero MPI calls. NP defaults to the allocation's node
# count ($PJM_NODE); override by exporting NP.
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

echo "=== step 2: uTofu Put exchange (no MPI in the binary) ==="
mpiexec -np "$NP" ./tofu_put_demo

echo "=== done ==="
