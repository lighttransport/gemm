#!/bin/bash
#PJM -g hp250467
#PJM -L "rscgrp=small,node=32,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=32"
#PJM --llio localtmp-size=1Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
LOG="$K3/logs/pool-32n-${PJM_JOBID}"
mkdir -p "$LOG"

"$K3/run_k3_ep.sh" --mode dummy --nodes 32 --layer 1 --layers 1 --tokens 64 \
  --threads 48 --ar-groups auto --result-dir "$LOG/dummy"
"$K3/run_kda_probe_mpi.sh" --nodes 32 --layer 0 --head 0 2>&1 | tee "$LOG/kda.log"
"$K3/run_moe_probe_mpi.sh" --nodes 32 --layer 1 --experts-per-rank 4 --threads 48 \
  --result-dir "$LOG/moe" 2>&1 | tee "$LOG/moe.log"
"$K3/run_expert_tp_probe_mpi.sh" --nodes 32 --layer 1 --experts 16 --threads 48 \
  --result-dir "$LOG/expert_tp" 2>&1 | tee "$LOG/expert_tp.log"
"$K3/run_k3_ep.sh" --mode dummy --nodes 32 --layer 3 --layers 1 --tokens 256 \
  --kda-threads 8 --mla-cache-bf16 --ar-groups auto --result-dir "$LOG/attention"
