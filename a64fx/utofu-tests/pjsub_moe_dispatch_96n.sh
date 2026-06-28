#!/bin/bash
# Phase 0 go/no-go: measure the MoE expert-dispatch ALL-TO-ALL vs the recursive-doubling TREE
# (the current dense-allreduce baseline) on 96 A64FX nodes, GLM-5.2 MoE shape (E=256,K=8,HID=6144).
# If an all-to-all variant beats TREE on a prefill-sized batch, the all-to-all MoE effort is worth it.
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=96,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM -j
set -u
UTOFU=/home/u14346/work/gemm/glm5-1/a64fx/utofu-tests
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
NP=${PJM_MPI_PROC:-96}
cd "$UTOFU" || exit 2
echo "=== MoE dispatch all-to-all bench: NP=$NP job=${PJM_JOBID:-?} ==="
date
make tofu_topo_helper moe_dispatch_bench CC=fccpx MPICC=/opt/FJSVxtclanga/tcsds-1.2.43/bin/mpifccpx >/dev/null || { echo FATAL build; exit 3; }
topo_ok=0; for t in $(seq 1 40); do rm -f tofu_topo.txt   # persist through intermittent uTofu-init windows
  mpiexec -np "$NP" ./tofu_topo_helper 2>/dev/null && [ "$(grep -vc '^#' tofu_topo.txt 2>/dev/null||echo 0)" -ge "$NP" ] && { topo_ok=1; echo "topo got window on try $t"; break; }; sleep 12; done
[ "$topo_ok" = 1 ] || { echo "FATAL: topo"; exit 3; }; echo "topo OK ($(grep -vc '^#' tofu_topo.txt) nodes)"

# GLM-5.2 MoE shape; sweep batch from decode(1) to prefill chunk(2048).
export MOE_E=256 MOE_K=8 MOE_HID=6144 MOE_ABYTES=2 MOE_NTNI=6 MOE_LAYERS=75 MOE_ITERS=500 MOE_WARMUP=50
for B in 1 256 1024 2048; do
  echo "--- MOE_BATCH=$B ($(date)) ---"
  MOE_BATCH=$B mpiexec -np "$NP" ./moe_dispatch_bench 2>&1 | grep -iE "BATCH|NAIVE|MULTI|TREE|DIM|BRUCK|us|GB/s|per-layer|whole|best|win" | head -30
done
echo "=== done $(date) ==="
