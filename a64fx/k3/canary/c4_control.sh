#!/bin/bash
# Gate-check canary: control, no --llio directive at all.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM -j
set -eu
echo "K3_CANARY c4 host=$(hostname) localtmp=${PJM_LLIO_LOCALTMP_SIZE:-unset}"
df -h /local || true
