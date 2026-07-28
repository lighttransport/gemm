#!/bin/bash
# Six-hour best-node Laguna S-2.1 FP8 llmgr allocation.
#
# The selected shape is 12 A64FX nodes in the small resource group, one EP
# rank per node, 2 GHz, eco-state and retention disabled.  The common llmgr
# supervisor/tunnel implementation is reused after these directives.
#
#   pjsub --no-check-directory a64fx/llmgr/pjsub_llmgr_laguna_fp8_12n.sh
#
# Once the tunnel is up, start the runner with:
#   curl -sS -X POST localhost:21374/runner/start \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"laguna","variant":"fp8","mode":"serve",'\
#         '"np":12,"port":8080,"maxpos":32768,"stage":true}'
#
# Do not add health probes to the runner itself while it is loading; llmgr's
# passive readiness check waits for all rank log markers first.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=12,elapse=06:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j

set -euo pipefail
REPO=${REPO:-/home/u14346/work/gemm/glm5-1}
export FRONTEND_PORT="${FRONTEND_PORT:-21374}"
export SERVER_PORT="${SERVER_PORT:-21274}"
exec "$REPO/a64fx/llmgr/pjsub_llmgr_12n.sh"
