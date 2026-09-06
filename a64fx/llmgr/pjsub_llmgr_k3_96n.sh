#!/bin/bash
# One-hour 96-node K3 llmgr allocation with supervised HTTP reverse tunnel.
# After the SENTINEL appears on the login node:
#   python3 a64fx/llmgr/llmgr_cli.py --url http://127.0.0.1:21375 models
#   python3 a64fx/llmgr/llmgr_cli.py --url http://127.0.0.1:21375 build --model k3
#   python3 a64fx/llmgr/llmgr_cli.py --url http://127.0.0.1:21375 stage \
#       --model k3 --np 96 --layer 1 --experts 0-15
#   python3 a64fx/llmgr/llmgr_cli.py --url http://127.0.0.1:21375 start \
#       --model k3 --mode generate --np 96 --layer 1 --tokens 256
# Live source edits/builds use `llmgr_cli.py sh`; stop the active MPI child,
# rebuild, and restart it against the retained /local stage to apply a fix.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=01:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=1Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j

set -euo pipefail
REPO=${REPO:-/vol0006/mdt0/data/hp250467/work/gemm/k3}
export REPO
export FRONTEND_PORT="${FRONTEND_PORT:-21375}"
export SERVER_PORT="${SERVER_PORT:-21275}"
exec "$REPO/a64fx/llmgr/pjsub_llmgr_12n.sh"
