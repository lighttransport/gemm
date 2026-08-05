#!/bin/bash
# Verify that `pjsub -x NAME=value` on the command line reaches the job, so the
# script can drop the bare `#PJM -x NAME` directives without silently falling
# back to its own defaults.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -eu
echo "K3_CANARY c9 host=$(hostname)"
for v in K3_THREADS K3_PREFILL_TOKENS K3_MOE_SHARD_LAYOUT K3_MODEL_DIR K3_FULL_STAGE_DIR; do
    eval "val=\${$v:-<UNSET>}"
    echo "  $v=$val"
done
echo "  localtmp=${PJM_LLIO_LOCALTMP_SIZE:-unset} gfscache=${PJM_LLIO_GFSCACHE:-unset}"
