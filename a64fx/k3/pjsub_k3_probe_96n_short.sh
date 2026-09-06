#!/bin/bash
# Short TP96 probe for dummy + small real-weight decode.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=1Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
LOG="$K3/logs/pjsub_probe_96n-${PJM_JOBID}"
mkdir -p "$LOG"

"$K3/run_k3_ep.sh" --mode dummy --nodes 96 --layer 1 --layers 1 --tokens 16 \
  --threads 48 --ar-groups auto --result-dir "$LOG/dummy"

"$K3/run_k3_ep.sh" --mode real --nodes 96 --layer 1 --experts 0-15 --layers 1 --tokens 64 \
  --threads 47 --fused-threads 47 --ar-groups auto --prefetch-mib 16 --prefetch-threads 32 \
  --heartbeat-tokens 64 --model-dir "$HOME/models/kimi-k3" \
  --stage-dir "/local/$USER/k3-probe-96n-${PJM_JOBID}" --result-dir "$LOG/decode"
