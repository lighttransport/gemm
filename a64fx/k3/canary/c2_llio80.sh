#!/bin/bash
# Gate-check canary: the 96n directive set with the known-good localtmp-size=80Gi.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -eu
echo "K3_CANARY c2 host=$(hostname) localtmp=${PJM_LLIO_LOCALTMP_SIZE:-unset}"
df -h /local || true
