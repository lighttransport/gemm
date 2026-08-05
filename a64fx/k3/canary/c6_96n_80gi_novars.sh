#!/bin/bash
# Gate-check canary: 96 nodes at the known-good 80Gi, no bare -x directives.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=00:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -eu
echo "K3_CANARY c6 host=$(hostname) localtmp=${PJM_LLIO_LOCALTMP_SIZE:-unset} nodes=${PJM_NODE:-?}"
df -h /local || true
