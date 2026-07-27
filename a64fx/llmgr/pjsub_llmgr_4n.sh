#!/bin/bash
# Four-node llmgr batch allocation for Gemma4 12B smoke testing.
#
# Submit from a Fugaku frontend with:
#   pjsub --no-check-directory a64fx/llmgr/pjsub_llmgr_4n.sh
#
# The existing 12-node supervisor is reused below; its PJM comments are inert
# when it is invoked as a shell script.  The four-node resource group uses a
# non-shaped allocation, so all four ranks are used.  PP splits the model and
# leaves only a few GB on the llmgr head node.

#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=4,elapse=06:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=4"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j

set -euo pipefail

REPO=${REPO:-/home/u14346/work/gemm/glm5-1}
export REPO

# Prefer the frontend on which this job was submitted, then use the known
# Fugaku login frontends as fallbacks if the private frontend address is not
# reachable from the allocated compute head node.
export FRONTEND_HOST=${FRONTEND_HOST:-fn01sv04}
export FRONTEND_SSH_TARGET=${FRONTEND_SSH_TARGET:-10.4.128.24}
export FRONTEND_SSH_TARGETS=${FRONTEND_SSH_TARGETS:-10.4.128.24,login1.fugaku.r-ccs.riken.jp,login2.fugaku.r-ccs.riken.jp}
export FRONTEND_PORT=${FRONTEND_PORT:-21374}
export KEEPALIVE_SECONDS=${KEEPALIVE_SECONDS:-21000}
export NP=${NP:-4}
export EXCLUDE=${EXCLUDE:-none}

exec "$REPO/a64fx/llmgr/pjsub_llmgr_12n.sh"
