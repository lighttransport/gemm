#!/bin/bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$repo/a64fx/glm5"
np=${PJM_MPI_PROC:-12}
if [ "$np" -ne 12 ]; then
    echo "expected 12-node allocation, PJM_MPI_PROC=$np" >&2
    exit 2
fi

stage=${GLM53F_STAGE_DIR:-/local/glm53f-decode-${PJM_JOBID}}
status=${GLM53F_STATUS_DIR:-$PWD/stage_status_${PJM_JOBID}_full}
mkdir -p "$stage" "$status"

export GLM53F_RANKS=12
export GLM53F_EXPERT_PARTS=4
export GLM53F_STAGE_FIRST_LAYER=3
export GLM53F_STAGE_LAYERS=45
export GLM53F_STAGE_DIR=$stage
export GLM53F_STATUS_DIR=$status

echo "GLM53F full expert stage job=$PJM_JOBID stage=$stage status=$status"
date
mpiexec -np 12 ./glm53f_decode_stage "$HOME/models/glm53f"
cat "$status"/*.status
test "$(grep -l ' OK$' "$status"/*.status | wc -l)" -eq 12
echo "SENTINEL glm53f_decode_stage_full_12n=OK"
date
