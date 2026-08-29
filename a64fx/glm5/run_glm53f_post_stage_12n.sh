#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
repo=$(cd ../.. && pwd)
job=${PJM_JOBID:?PJM_JOBID is required}
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
routed=${GLM53F_STAGE_DIR:-/local/glm53f-target-p12-$job}
shared=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-target-shared-$job}
logdir=${GLM53F_POST_LOG_DIR:-$repo/tmp/post-stage-$job}
mkdir -p "$logdir"

routed_status="$logdir/routed-status"
mkdir -p "$routed_status"
if [ "$(find "$routed_status" -name 'rank*.status' -exec grep -l ' OK$' {} + 2>/dev/null | wc -l)" -ne 12 ]; then
    echo "staging routed experts job=$job"
    GLM53F_RANKS=12 \
    GLM53F_EXPERT_PARTS=12 \
    GLM53F_STAGE_FIRST_LAYER=3 \
    GLM53F_STAGE_LAYERS=45 \
    GLM53F_STAGE_DIR="$routed" \
    GLM53F_STATUS_DIR="$routed_status" \
        mpiexec -np 12 -of-proc "$logdir/routed" \
        ./glm53f_decode_stage "$model"
fi
test "$(grep -l ' OK$' "$routed_status"/rank*.status | wc -l)" -eq 12

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-47}
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpiexec -np 12 -of-proc "$logdir/kda" \
    ./glm53f_kda_layer_12n "$model" 44
mpiexec -np 12 -of-proc "$logdir/kda-callback" \
    ./glm53f_kda_callback_check "$model" 44
mpiexec -np 12 -of-proc "$logdir/dense" \
    ./glm53f_dense_ffn_12n "$model" 0
mpiexec -np 12 -of-proc "$logdir/sparse" \
    ./glm53f_sparse_layer_12n "$model" 2053 43

if [ "$(find "$logdir/shared-status" -name 'rank*.status' -exec grep -l ' OK$' {} + 2>/dev/null | wc -l)" -ne 12 ]; then
    GLM53F_SHARED_STAGE_DIR="$shared" \
    GLM53F_SHARED_STATUS_DIR="$logdir/shared-status" \
        ./run_glm53f_shared_stage_12n.sh
fi

mpiexec -np 12 -of-proc "$logdir/target-layer" \
    ./glm53f_target_layer_check_12n "$model" "$routed" "$shared" 44

GLM53F_STAGE_DIR="$routed" \
GLM53F_SHARED_STAGE_DIR="$shared" \
GLM53F_MODEL_DIR="$model" \
GLM53F_FIRST_LAYER=3 \
GLM53F_LAYER_COUNT=42 \
GLM53F_ATTENTION_COMBINE=1 \
GLM53F_DECODE_TOKENS=${GLM53F_DECODE_TOKENS:-20} \
MPIEXEC_OF_PROC="$logdir/expert" \
    ./run_glm53f_expert_decode_12n.sh

echo "SENTINEL glm53f_post_stage_12n=OK"
