#!/bin/bash
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#
# MPI-free uTofu multi-node Put demo.
#
# Submit on a 2-node allocation:
#   pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=2,elapse=00:10:00" \
#         --mpi "shape=2,proc=2" run_demo.sh
# Interactive variant uses rscgrp=int with --interact.
#
# Step 1 (MPI helper) discovers each node's Tofu coordinates -> tofu_topo.txt.
# Step 2 (pure uTofu app) is launched by mpiexec only for node placement; the
# binary itself makes zero MPI calls. Set NP to the node count.
set -e

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

NP=${NP:-2}

echo "=== step 1: discover topology (MPI helper) ==="
mpiexec -np "$NP" ./tofu_topo_helper
cat tofu_topo.txt

echo "=== step 2: uTofu Put exchange (no MPI in the binary) ==="
mpiexec -np "$NP" ./tofu_put_demo

echo "=== done ==="
