#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
repo=$(cd ../.. && pwd)
job=${PJM_JOBID:?PJM_JOBID is required}
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
target_routed=${GLM53F_STAGE_DIR:-/local/glm53f-target-routed-$job}
target_shared=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-target-shared-$job}
mtp_routed=${GLM53F_MTP_STAGE_DIR:-/local/glm53f-mtp-routed-$job}
mtp_shared=${GLM53F_MTP_SHARED_STAGE_DIR:-/local/glm53f-mtp-shared-$job}
logdir=${GLM53F_SPEC_LOG_DIR:-$repo/tmp/spec-validation-$job}
mkdir -p "$logdir"

while pgrep -f '^/bin/bash ./run_glm53f_final_validation_12n.sh$' >/dev/null; do
    sleep 20
done

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cycles=${GLM53F_SPEC_CYCLES:-16}
drafts=${GLM53F_SPEC_DRAFTS:-1}
warmup=${GLM53F_SPEC_WARMUP:-128}
if [ "$cycles" -eq 16 ] && [ "$drafts" -eq 1 ] && [ "$warmup" -eq 128 ]; then
    export GLM53F_SPEC_MIN_ALPHA=${GLM53F_SPEC_MIN_ALPHA:-0.60}
    export GLM53F_SPEC_EXPECT_ACCEPTED=${GLM53F_SPEC_EXPECT_ACCEPTED:-10}
    export GLM53F_SPEC_EXPECT_FINAL=${GLM53F_SPEC_EXPECT_FINAL:-40591}
fi

mpiexec -np 12 -of-proc "$logdir/spec" \
    ./glm53f_spec_decode_12n "$model" "$target_routed" "$target_shared" \
    "$mtp_routed" "$mtp_shared" 1 "$cycles" "$drafts" "$warmup"

echo "SENTINEL glm53f_spec_validation_12n=OK"
