#!/bin/bash
# Sequentially gate partial real I/O on a successful dummy graph/collective run.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=00:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=1Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -eu

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
RUN="$REPO/a64fx/k3/run_k3_ep.sh"

"$RUN" --mode dummy --nodes 96 --layer 2 --layers 3 --tokens 2 --threads 48 \
    --result-dir "$REPO/a64fx/k3/logs/dummy-${PJM_JOBID}"

echo "dummy gate passed on 96 nodes; starting bounded real-weight stage"
"$RUN" --mode real --nodes 96 --layer 1 --experts 0-15 \
    --layers 1 --tokens 2 --threads 48 --model-dir "$HOME/models/kimi-k3" \
    --result-dir "$REPO/a64fx/k3/logs/real-${PJM_JOBID}"
