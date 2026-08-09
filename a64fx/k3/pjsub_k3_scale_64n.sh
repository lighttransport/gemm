#!/bin/bash
# Scalar 64-node K3 module/regression sweep.  TP64 is the largest divisor
# available at this size; TP96 coverage is exercised by the 24/32/48-node jobs.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=64,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=64"
#PJM --llio localtmp-size=1Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
LOG="$K3/logs/pool-64n-${PJM_JOBID}"
mkdir -p "$LOG"

"$K3/run_k3_ep.sh" --mode dummy --nodes 64 --layer 1 --layers 1 --tokens 64 \
  --threads 48 --comm-deterministic 1 --comm-robust 2 --ar-groups auto \
  --result-dir "$LOG/dummy"
"$K3/run_kda_probe_mpi.sh" --nodes 64 --layer 0 --head 0 2>&1 | tee "$LOG/kda.log"
"$K3/run_kda_probe_mpi.sh" --nodes 64 --layer 0 --head 1 2>&1 | tee "$LOG/kda-head1.log"
"$K3/run_moe_probe_mpi.sh" --nodes 64 --layer 1 --experts-per-rank 4 --threads 48 \
  --result-dir "$LOG/moe" 2>&1 | tee "$LOG/moe.log"
"$K3/run_expert_tp_probe_mpi.sh" --nodes 64 --layer 1 --experts 16 --threads 48 \
  --logical-tp 64 --logical-waves 1 --result-dir "$LOG/expert_tp" \
  2>&1 | tee "$LOG/expert_tp.log"
"$K3/run_expert_tp_probe_mpi.sh" --nodes 64 --layer 1 --experts 16 --threads 48 \
  --prefill --logical-tp 64 --logical-waves 1 --result-dir "$LOG/expert_tp_prefill" \
  2>&1 | tee "$LOG/expert_tp_prefill.log"
"$K3/run_k3_ep.sh" --mode dummy --nodes 64 --layer 3 --layers 1 --tokens 256 \
  --kda-threads 8 --comm-deterministic 1 --comm-robust 2 --mla-cache-bf16 \
  --ar-groups auto --result-dir "$LOG/attention"
