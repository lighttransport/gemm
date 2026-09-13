#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
job=${PJM_JOBID:?PJM_JOBID is required}
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
routed=${GLM53F_MTP_STAGE_DIR:-/local/glm53f-mtp-routed-$job}
shared=${GLM53F_MTP_SHARED_STAGE_DIR:-/local/glm53f-mtp-shared-$job}
status_root=${GLM53F_MTP_STATUS_DIR:-$PWD/stage_status_${job}_mtp}
logdir=${GLM53F_MTP_STAGE_LOG_DIR:-../../tmp/mtp-stage-$job}
mkdir -p "$routed" "$shared" "$status_root/routed" "$status_root/shared" "$logdir"

export GLM53F_RANKS=12
export GLM53F_EXPERT_PARTS=12
export GLM53F_STAGE_FIRST_LAYER=45
export GLM53F_STAGE_LAYERS=46

GLM53F_STAGE_DIR="$routed" GLM53F_STATUS_DIR="$status_root/routed" \
    mpiexec -np 12 -of-proc "$logdir/routed" ./glm53f_decode_stage "$model"
test "$(grep -l ' OK$' "$status_root"/routed/rank*.status | wc -l)" -eq 12

GLM53F_STAGE_SHARED_ONLY=1 GLM53F_STAGE_DIR="$shared" \
GLM53F_STATUS_DIR="$status_root/shared" \
    mpiexec -np 12 -of-proc "$logdir/shared" ./glm53f_decode_stage "$model"
test "$(grep -l ' OK$' "$status_root"/shared/rank*.status | wc -l)" -eq 12

echo "SENTINEL glm53f_mtp_stage_12n=OK routed=$routed shared=$shared"
