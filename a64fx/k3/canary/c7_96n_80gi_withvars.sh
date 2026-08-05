#!/bin/bash
# Gate-check canary: the 96n script's directive block verbatim, 1-minute body.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=00:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -x K3_BARRIER_ITERS
#PJM -x K3_THREADS
#PJM -x K3_MODEL_DIR
#PJM -j
set -eu
echo "K3_CANARY c7 host=$(hostname) localtmp=${PJM_LLIO_LOCALTMP_SIZE:-unset} nodes=${PJM_NODE:-?}"
df -h /local || true
