#!/bin/bash
# Memory-safe 12-node throughput attachment.  Run inside a PJM 12-node
# allocation; rank-local staging is streamed by run_k3_ep.sh.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
JOB_TAG=${PJM_JOBID:-manual-$$}
if [[ "${PJM_MPI_PROC:-}" != 12 ]]; then
    echo "$0: requires an active 12-node allocation (PJM_MPI_PROC=12)" >&2
    exit 2
fi

exec "$SCRIPT_DIR/run_k3_ep.sh" \
    --mode hybrid \
    --nodes 12 --tp-nodes 12 \
    --layer 1 --layers 93 --tokens 64 \
    --threads 47 --kda-threads 8 --fused-threads 47 \
    --heartbeat-tokens 64 --min-available-mib 2048 \
    --ar-groups 2 --comm-robust 2 --comm-deterministic 0 \
    --prefetch-mib 16 --prefetch-threads 32 --profile \
    --stage-dir "/local/$USER/k3-hybrid-12n-$JOB_TAG" \
    --result-dir "$SCRIPT_DIR/logs/hybrid-12n-$JOB_TAG" \
    "$@"
