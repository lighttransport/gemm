#!/bin/bash
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#
# Ring-attention DECODE cost estimate over uTofu (no GEMM, no MPI in the bench).
# Estimates two things per decode step:
#   1. per-node KV-shard memory-read time (S/N positions per node), and
#   2. the uTofu ring-reduce communication cost (N-1 hops of the partials).
#
# Submit on an N-node allocation (2 <= N <= 12), e.g. 12 nodes:
#   pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=12,elapse=00:10:00" \
#         --mpi "shape=12,proc=12" run_bench.sh
# Or 2 nodes: node=2 ... --mpi "shape=2,proc=2". Interactive: rscgrp=int --interact.
#
# Step 1 (MPI helper) discovers each node's Tofu coordinates -> tofu_topo.txt.
# Step 2 (pure uTofu bench) is launched by mpiexec only for node placement; the
# binary makes zero MPI calls. NP defaults to the allocation's node count
# ($PJM_NODE); override by exporting NP.
#
# Memory BW is NUMA-sensitive on A64FX (4 CMGs, ~200 GB/s each, ~800 GB/s/node).
# The bench manages its own placement internally -- one buffer per CMG, mbind'd
# to that CMG's NUMA node, threads pinned, read with an 8-accumulator SVE kernel
# -- so it reaches full node BW (~900 GB/s here) WITHOUT any numactl wrapper (do
# not wrap it in numactl, which would impose a conflicting default policy). Just
# give it all 48 cores. Tune the model shape via RA_* env vars (RA_SEQ, RA_QH,
# RA_KVH, RA_HD, RA_KVB, RA_BWMB). To bypass on-node measurement and use a
# known/profiled read BW for the estimate, set RA_BW_GBPS=645.
set -e

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"

NP=${NP:-${PJM_NODE:-2}}
if [ "$NP" -gt 12 ]; then
    echo "warning: NP=$NP exceeds the validated ceiling of 12 nodes" >&2
fi
echo "running on $NP node(s)"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}

echo "=== step 1: discover topology (MPI helper) ==="
mpiexec -np "$NP" ./tofu_topo_helper
cat tofu_topo.txt

echo "=== step 2: ring-attention decode cost estimate (no MPI in the binary) ==="
mpiexec -np "$NP" ./ring_attn_bench

echo "=== done ==="
