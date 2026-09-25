#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
model=${1:?first Q8_0 GGUF shard is required}
root=${GLM53F_Q8_IMAGE_ROOT:-$HOME/models/glm53f-q8-rank12-v1}
ranks=${PJM_MPI_PROC:-12}
stage_bin=${GLM53F_Q8_STAGE_BIN:-./glm53f_q8_stage}
test "$ranks" -eq 12
test -x "$stage_bin"
mkdir -p "$root"

mpiexec -np "$ranks" -stdout-proc "$root/stage" sh -c '
  r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
  exec "$4" "$1" "$2" "$r" "$3"
' sh "$model" "$root" "$ranks" "$stage_bin"

test "$(find "$root" -maxdepth 1 -name 'rank??.blob' | wc -l)" -eq "$ranks"
test "$(find "$root" -maxdepth 1 -name 'rank??.manifest' | wc -l)" -eq "$ranks"
printf 'GLM53F_Q8_IMAGE_ROOT_V1 ranks=%d\n' "$ranks" > "$root/.COMPLETE.tmp"
mv "$root/.COMPLETE.tmp" "$root/COMPLETE"
printf 'SENTINEL glm53f_q8_stage_12n=OK root=%s ranks=%d\n' "$root" "$ranks"
