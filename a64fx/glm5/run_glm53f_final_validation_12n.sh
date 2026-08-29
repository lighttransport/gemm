#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
repo=$(cd ../.. && pwd)
job=${PJM_JOBID:?PJM_JOBID is required}
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
routed=${GLM53F_STAGE_DIR:-/local/glm53f-target-p12-$job}
shared=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-target-shared-$job}
logdir=${GLM53F_FINAL_LOG_DIR:-$repo/tmp/final-validation-$job}
mkdir -p "$logdir"

while pgrep -f '^/bin/bash ./run_glm53f_post_stage_12n.sh$' >/dev/null; do
    sleep 20
done

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-47}
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close
export OMP_PLACES=cores

mpiexec -np 12 -of-proc "$logdir/head" \
    ./glm53f_target_head_12n "$model"
mpiexec -np 12 -of-proc "$logdir/sparse-callback" \
    ./glm53f_sparse_callback_check "$model" 43
mpiexec -np 12 -of-proc "$logdir/full-target" \
    ./glm53f_target_decode_12n "$model" "$routed" "$shared" 1 1

echo "SENTINEL glm53f_final_validation_12n=OK"
