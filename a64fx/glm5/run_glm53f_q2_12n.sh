#!/bin/bash
# Build, stage and run the GLM-5.3-Flash mixed-IQ routed-expert path directly
# inside an allocated 12-node Fugaku job.  All large rank-local files live on
# each compute node's /local filesystem.
set -euo pipefail

cd "$(dirname "$0")"
repo_glm5=$(pwd -P)
cd "$repo_glm5"
test "${PJM_MPI_PROC:-0}" -eq 12 || {
    echo "error: a 12-rank interactive allocation is required" >&2
    exit 2
}

job=${PJM_JOBID:?PJM_JOBID is required}
q2_root=${GLM53F_Q2_ROOT:-$HOME/models/glm53f-gguf}
q2=${GLM53F_Q2_MODEL:-$q2_root/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf}
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
q2_stage=${GLM53F_STAGE_DIR:-/local/glm53f-q2-routed-$job}
shared_stage=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-q2-shared-$job}
core_stage=${GLM53F_REPACK_STAGE_DIR:-/local/glm53f-q2-core-$job}
shared_source=${GLM53F_SHARED_SOURCE:-$model/a64fx_ep12_v1/shared}
core_source=${GLM53F_CORE_SOURCE:-$model/a64fx_ep12_v2_core}
logdir=${GLM53F_Q2_LOG_DIR:-../../tmp/glm53f-q2-$job}
mkdir -p "$logdir"
logdir=$(cd "$logdir" && pwd -P)
run_tag=${GLM53F_Q2_RUN_TAG:-$$}

# The inherited interactive shell can point OPAL_PREFIX at OSS-CN even though
# mpifcc is the Fujitsu wrapper.  Keep compiler and launcher in one MPI tree.
export OPAL_PREFIX=${GLM53F_MPI_HOME:-/opt/FJSVxtclanga/tcsds-1.2.43}
export MPI_HOME=$OPAL_PREFIX
mpiexec_bin=$OPAL_PREFIX/bin/mpiexec
export GLM53F_MPICC=${GLM53F_MPICC:-mpifcc}
export GLM53F_BUILD_DIR=${GLM53F_BUILD_DIR:-/local/glm53f-q2-build-$job}
export GLM53F_FAST_MATH=${GLM53F_FAST_MATH:-1}
export GLM53F_NO_MATH_ERRNO=${GLM53F_NO_MATH_ERRNO:-1}
export TMPDIR=${TMPDIR:-$GLM53F_BUILD_DIR}
mkdir -p "$TMPDIR"

sh ./build_glm53f_integrated_12n.sh

echo "staging Q2 routed experts to $q2_stage"
"$mpiexec_bin" -n 12 -of-proc "$logdir/q2-stage-$run_tag" \
    ./glm53f_q2_stage "$q2" "$q2_stage"
test "$(grep -l 'SENTINEL glm53f_q2_stage=' "$logdir"/q2-stage-$run_tag.*.* | wc -l)" -eq 12

echo "staging shared expert and compact core to node-local storage"
"$mpiexec_bin" -n 12 -of-proc "$logdir/shared-stage-$run_tag" sh -c '
    r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
    exec "$3" "$1" "$2" "$r" model
' sh "$shared_source" "$shared_stage" "$repo_glm5/glm53f_core_stage"
test "$(grep -l 'SENTINEL glm53f_rank_stage=' "$logdir"/shared-stage-$run_tag.*.* | wc -l)" -eq 12
"$mpiexec_bin" -n 12 -of-proc "$logdir/core-stage-$run_tag" sh -c '
    r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
    exec "$3" "$1" "$2" "$r" core
' sh "$core_source" "$core_stage" "$repo_glm5/glm53f_core_stage"
test "$(grep -l 'SENTINEL glm53f_rank_stage=' "$logdir"/core-stage-$run_tag.*.* | wc -l)" -eq 12

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-47}
export OMP_DYNAMIC=false OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close OMP_PLACES=cores FLIB_BARRIER=HARD
export GLM53F_REPACK_DIR=$core_stage
export GLM53F_REPACK_REQUIRE=${GLM53F_REPACK_REQUIRE:-0}

"$mpiexec_bin" -n 12 ../utofu-tests/tofu_topo_helper
test "$(grep -vc '^#' tofu_topo.txt)" -eq 12
export GLM53F_UTOFU=1 TOFU_TOPO_PATH=$PWD/tofu_topo.txt

steps=${GLM53F_TARGET_STEPS:-128}
echo "running GLM-5.3-Flash Q2-routed decode steps=$steps"
"$mpiexec_bin" -n 12 -of-proc "$logdir/decode-$run_tag" \
    ./glm53f_target_decode_12n "$model" "$q2_stage" "$shared_stage" 1 "$steps"
cat "$logdir"/decode-$run_tag.*.0
result=$(grep 'GLM53F_TARGET_DECODE_12N' "$logdir"/decode-$run_tag.*.0 | tail -1)
rate=$(printf '%s\n' "$result" | sed -n 's/.* tok_s=\([0-9.]*\).*/\1/p')
awk -v rate="$rate" -v target="${GLM53F_MIN_TOK_S:-20}" \
    'BEGIN { exit !(rate + 0 >= target + 0) }'
echo "SENTINEL glm53f_q2_12n=OK logdir=$logdir"
