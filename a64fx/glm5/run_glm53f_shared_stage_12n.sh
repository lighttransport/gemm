#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
np=${PJM_MPI_PROC:-12}
if [ "$np" -ne 12 ]; then
    echo "expected 12-node allocation, PJM_MPI_PROC=$np" >&2
    exit 2
fi

stage=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-target-shared-${PJM_JOBID}}
status=${GLM53F_SHARED_STATUS_DIR:-$PWD/stage_status_${PJM_JOBID}_shared}
mkdir -p "$stage" "$status"
export GLM53F_RANKS=12
export GLM53F_EXPERT_PARTS=12
export GLM53F_STAGE_FIRST_LAYER=3
export GLM53F_STAGE_LAYERS=45
export GLM53F_STAGE_SHARED_ONLY=1
export GLM53F_STAGE_DIR=$stage
export GLM53F_STATUS_DIR=$status

echo "GLM53F shared stage job=$PJM_JOBID stage=$stage status=$status"
date
mpiexec -np 12 ./glm53f_decode_stage "$HOME/models/glm53f"
test "$(grep -l ' OK$' "$status"/*.status | wc -l)" -eq 12
echo "SENTINEL glm53f_shared_stage_12n=OK"
date
