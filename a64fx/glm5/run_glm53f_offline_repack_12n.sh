#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
test "${PJM_MPI_PROC:-0}" -eq 12
model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
root=${GLM53F_REPACK_ROOT:-$model/a64fx_ep12_v1}
trace=${GLM53F_CORE_TRACE_DIR:?set GLM53F_CORE_TRACE_DIR to the completed rank traces}
mkdir -p "$root/routed" "$root/shared" "$root/core" \
         "$root/status/routed" "$root/status/shared"

if [ "$(find "$root/status/routed" -name 'rank??.status' -exec grep -l ' OK$' {} + 2>/dev/null | wc -l)" -ne 12 ]; then
    GLM53F_RANKS=12 GLM53F_EXPERT_PARTS=12 GLM53F_STAGE_FIRST_LAYER=3 \
    GLM53F_STAGE_LAYERS=45 GLM53F_STAGE_DIR="$root/routed" \
    GLM53F_STATUS_DIR="$root/status/routed" \
        mpiexec -np 12 ./glm53f_decode_stage "$model"
fi

if [ "$(find "$root/status/shared" -name 'rank??.status' -exec grep -l ' OK$' {} + 2>/dev/null | wc -l)" -ne 12 ]; then
    GLM53F_RANKS=12 GLM53F_EXPERT_PARTS=12 GLM53F_STAGE_FIRST_LAYER=3 \
    GLM53F_STAGE_LAYERS=45 GLM53F_STAGE_SHARED_ONLY=1 \
    GLM53F_STAGE_DIR="$root/shared" GLM53F_STATUS_DIR="$root/status/shared" \
        mpiexec -np 12 ./glm53f_decode_stage "$model"
fi

mpiexec -np 12 sh -c '
  r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
  exec ./glm53f_repack_trace "$1" "$2/rank$(printf %02d "$r").trace" "$3" "$r"
' sh "$model" "$trace" "$root/core"

test "$(find "$root/routed" -name 'rank??.blob' | wc -l)" -eq 12
test "$(find "$root/shared" -name 'rank??.blob' | wc -l)" -eq 12
test "$(find "$root/core" -name 'rank??.core.blob' | wc -l)" -eq 12
touch "$root/COMPLETE"
echo "SENTINEL glm53f_offline_repack_12n=OK root=$root"
