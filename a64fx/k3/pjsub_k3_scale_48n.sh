#!/bin/bash
#PJM -g hp250467
#PJM -L "rscgrp=small,node=48,elapse=00:30:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=48"
#PJM --llio localtmp-size=1Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
LOG="$K3/logs/pool-48n-${PJM_JOBID}"
mkdir -p "$LOG"

"$K3/run_k3_ep.sh" --mode dummy --nodes 48 --layer 1 --layers 1 --tokens 64 \
  --threads 48 --comm-deterministic 1 --comm-robust 2 --ar-groups auto --result-dir "$LOG/dummy"
"$K3/run_kda_probe_mpi.sh" --nodes 48 --layer 0 --head 0 2>&1 | tee "$LOG/kda.log"
"$K3/run_kda_probe_mpi.sh" --nodes 48 --layer 0 --head 1 2>&1 | tee "$LOG/kda-head1.log"
"$K3/run_moe_probe_mpi.sh" --nodes 48 --layer 1 --experts-per-rank 4 --threads 48 \
  --result-dir "$LOG/moe" 2>&1 | tee "$LOG/moe.log"
"$K3/run_expert_tp_probe_mpi.sh" --nodes 48 --layer 1 --experts 16 --threads 48 \
  --logical-tp 96 --logical-waves 2 --result-dir "$LOG/expert_tp" 2>&1 | tee "$LOG/expert_tp.log"
"$K3/run_expert_tp_probe_mpi.sh" --nodes 48 --layer 1 --experts 16 --threads 48 \
  --prefill --logical-tp 96 --logical-waves 2 --result-dir "$LOG/expert_tp_prefill" \
  2>&1 | tee "$LOG/expert_tp_prefill.log"
"$K3/run_k3_ep.sh" --mode dummy --nodes 48 --layer 3 --layers 1 --tokens 256 \
  --kda-threads 8 --comm-deterministic 1 --comm-robust 2 --mla-cache-bf16 \
  --ar-groups auto --result-dir "$LOG/attention"
"$K3/run_k3_ep.sh" --mode dummy --nodes 48 --layer 0 --layers 4 --tokens 64 \
  --threads 48 --kda-threads 8 --comm-deterministic 1 --comm-robust 2 \
  --mla-cache-bf16 --ar-groups auto --result-dir "$LOG/window-kda-mla"
"$K3/run_k3_ep.sh" --mode dummy --nodes 48 --layer 4 --layers 8 --tokens 64 \
  --threads 48 --kda-threads 8 --comm-deterministic 1 --comm-robust 2 \
  --mla-cache-bf16 --ar-groups auto --result-dir "$LOG/window-middle"
"$K3/run_k3_ep.sh" --mode dummy --nodes 48 --layer 12 --layers 12 --tokens 32 \
  --threads 48 --kda-threads 8 --comm-deterministic 1 --comm-robust 2 \
  --mla-cache-bf16 --ar-groups auto --result-dir "$LOG/window-long"
"$K3/run_k3_ep.sh" --mode dummy --nodes 48 --layer 88 --layers 5 --tokens 32 \
  --threads 48 --kda-threads 8 --comm-deterministic 1 --comm-robust 2 \
  --mla-cache-bf16 --ar-groups auto --result-dir "$LOG/window-tail"
"$K3/run_k3_ep.sh" --mode real --nodes 48 --tp-nodes 48 --layer 1 --layers 1 \
  --experts 0-15 --tokens 32 --threads 47 --kda-threads 8 --fused-threads 47 \
  --comm-deterministic 1 --comm-robust 2 --ar-groups auto --profile \
  --model-dir "$HOME/models/kimi-k3" \
  --stage-dir "/local/$USER/k3-ep-real1-48n-${PJM_JOBID}" \
  --result-dir "$LOG/real-ep1"
"$K3/run_k3_ep.sh" --mode real --nodes 48 --tp-nodes 48 --layer 92 --layers 1 \
  --experts 0-15 --tokens 32 --threads 47 --kda-threads 8 --fused-threads 47 \
  --comm-deterministic 1 --comm-robust 2 --ar-groups auto --profile \
  --model-dir "$HOME/models/kimi-k3" \
  --stage-dir "/local/$USER/k3-ep-real92-48n-${PJM_JOBID}" \
  --result-dir "$LOG/real-ep92"
