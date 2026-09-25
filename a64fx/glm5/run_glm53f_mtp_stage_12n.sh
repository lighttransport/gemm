#!/bin/bash
# Stage only safetensors layer 45; target weights are staged independently.
set -euo pipefail
fail() { echo "error: $*" >&2; exit 2; }
source "$(dirname "$0")/scripts/glm53f_env.sh"
source "$glm53f_dir/scripts/glm53f_launch.sh"
require_allocation
# Layer 45 comes from the checkpoint, not the target's layers 0--44 image.
unset GLM53F_REPACK_DIR GLM53F_REPACK_REQUIRE GLM53F_REPACK_TRACE_ONLY
routed=${GLM53F_MTP_STAGE_DIR:-/local/glm53f-mtp-routed-$job}
shared=${GLM53F_MTP_SHARED_STAGE_DIR:-/local/glm53f-mtp-shared-$job}
logdir=$(realpath -m "${GLM53F_MTP_STAGE_LOG_DIR:-$logdir}")
status_root=${GLM53F_MTP_STATUS_DIR:-$logdir/mtp-status-$run_tag}
mkdir -p "$status_root/routed" "$status_root/shared" "$logdir"
export GLM53F_RANKS=12 GLM53F_EXPERT_PARTS=12
export GLM53F_STAGE_FIRST_LAYER=45 GLM53F_STAGE_LAYERS=46
for kind in routed shared; do
    destination=$routed
    shared_only=0
    if [ "$kind" = shared ]; then destination=$shared; shared_only=1; fi
    GLM53F_STAGE_DIR="$destination" GLM53F_STATUS_DIR="$status_root/$kind" \
    GLM53F_STAGE_SHARED_ONLY=$shared_only \
        mpi_run "mtp-$kind-stage" "$GLM53F_BIN_DIR/glm53f_decode_stage" "$model"
    for rank in {0..11}; do
        printf -v status "%s/%s/rank%02d.status" "$status_root" "$kind" "$rank"
        grep -q ' OK$' "$status" || fail "incomplete MTP stage: $status"
    done
done
echo "SENTINEL glm53f_mtp_stage_12n=OK routed=$routed shared=$shared"
