#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
root=${1:?shared Q8 image root is required}
job=${PJM_JOBID:-manual}
local_root=${GLM53F_Q8_LOCAL_ROOT:-/local/glm53f-q8-$job}
ranks=${PJM_MPI_PROC:-12}
copy_bin=${GLM53F_Q8_COPY_BIN:-./glm53f_core_stage}
resident_bin=${GLM53F_Q8_RESIDENT_BIN:-./test_glm53f_q8_resident}
test "$ranks" -eq 12
test -x "$copy_bin"
test -f "$root/COMPLETE"

mpiexec -np "$ranks" -stdout-proc "$root/local-stage" -stderr-proc "$root/local-stage-err" sh -c '
  r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
  exec "$4" "$1" "$2" "$r" model
' sh "$root" "$local_root" "$ranks" "$copy_bin"

if [ "${GLM53F_Q8_RESIDENT_TEST:-0}" = 1 ]; then
    test -x "$resident_bin"
    mpiexec -np "$ranks" -stdout-proc "$root/resident" -stderr-proc "$root/resident-err" sh -c '
      r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
      exec "$3" "$1" "$r"
    ' sh "$local_root" "$ranks" "$resident_bin"
fi

printf 'SENTINEL glm53f_q8_local_stage_12n=OK root=%s local=%s ranks=%d resident_test=%s\n' \
    "$root" "$local_root" "$ranks" "${GLM53F_Q8_RESIDENT_TEST:-0}"
