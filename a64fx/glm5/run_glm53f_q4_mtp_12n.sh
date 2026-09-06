#!/bin/bash
# Direct execution inside an existing 12-node A64FX interactive allocation.
set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 PROMPT_IDS OUTPUT_IDS [cycles=128] [drafts=1]" >&2
    exit 2
fi
job=${PJM_JOBID:?Run inside the 12-node interactive job}
repo_glm5=$(cd "$(dirname "$0")" && pwd)
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
routed=${GLM53F_Q4_STAGE_DIR:-/local/glm53f-q4-routed-$job}
shared=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-q4-shared-$job}
mtp_routed=${GLM53F_MTP_STAGE_DIR:-/local/glm53f-mtp-routed-$job}
mtp_shared=${GLM53F_MTP_SHARED_STAGE_DIR:-/local/glm53f-mtp-shared-$job}
logdir=${GLM53F_SPEC_LOG_DIR:-$repo_glm5/../../tmp/glm53f-q4-mtp-$job}
mkdir -p "$logdir"
export OPAL_PREFIX=${GLM53F_MPI_HOME:-/opt/FJSVxtclanga/tcsds-1.2.43}
export MPI_HOME=$OPAL_PREFIX
export PATH=/opt/local/mpiexec:$MPI_HOME/bin:$PATH
export TMPDIR=/local
export GLM53F_MPICC=${GLM53F_MPICC:-mpifcc}
export GLM53F_FAST_MATH=1 GLM53F_NO_MATH_ERRNO=1
if [ "${GLM53F_BUILD:-1}" = 1 ]; then
    bash "$repo_glm5/build_glm53f_integrated_12n.sh"
fi
# Weight staging is deliberately separate and restart-specific. Check every
# rank before allocating model memory; never silently reuse another job's paths.
mpiexec -np 12 sh -c '
    r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
    for directory do
        manifest=$(printf "%s/rank%02d.manifest" "$directory" "$r")
        test -s "$manifest" || { echo "missing stage: $manifest" >&2; exit 2; }
    done
' sh "$routed" "$shared" "$mtp_routed" "$mtp_shared"
export GLM53F_REPACK_DIR=${GLM53F_REPACK_DIR:-/local/glm53f-q4-core-$job}
export GLM53F_REPACK_REQUIRE=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-47}
export OMP_DYNAMIC=false OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close OMP_PLACES=cores FLIB_BARRIER=HARD
export GLM53F_UTOFU=1 GLM53F_PROFILE=1
export GLM53F_KDA_BATCH_TEAM=${GLM53F_KDA_BATCH_TEAM:-1}
export GLM53F_Q4_BATCH_SHARED=${GLM53F_Q4_BATCH_SHARED:-1}
export GLM53F_SPARSE_BATCH_OP=${GLM53F_SPARSE_BATCH_OP:-0}
(cd "$repo_glm5" && mpiexec -np 12 ../utofu-tests/tofu_topo_helper)
export TOFU_TOPO_PATH=$repo_glm5/tofu_topo.txt
export GLM53F_SPEC_PROMPT_IDS=$1 GLM53F_SPEC_OUTPUT_IDS=$2
mpiexec -np 12 -of-proc "$logdir/decode" \
    "${GLM53F_SPEC_BINARY:-$repo_glm5/glm53f_spec_decode_12n}" \
    "$model" "$routed" "$shared" "$mtp_routed" "$mtp_shared" \
    1 "${3:-128}" "${4:-1}" 0
grep -E 'GLM53F_SPEC_(PHASE|DECODE|REFERENCE|VARIANT)' "$logdir"/decode.*.0
