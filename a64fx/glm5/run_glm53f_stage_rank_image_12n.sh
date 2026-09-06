#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
test "${PJM_MPI_PROC:-0}" -eq 12
job=${PJM_JOBID:?}
root=${GLM53F_REPACK_ROOT:-$HOME/models/glm53f/a64fx_ep12_v1}
routed=${GLM53F_STAGE_DIR:-/local/glm53f-target-routed-$job}
shared=${GLM53F_SHARED_STAGE_DIR:-/local/glm53f-target-shared-$job}
core=${GLM53F_REPACK_STAGE_DIR:-/local/glm53f-target-core-$job}
test -f "$root/COMPLETE"

mpiexec -np 12 sh -c '
  r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
  exec ./glm53f_core_stage "$1" "$2" "$r" "$3"
' sh "$root/routed" "$routed" model
mpiexec -np 12 sh -c '
  r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
  exec ./glm53f_core_stage "$1" "$2" "$r" "$3"
' sh "$root/shared" "$shared" model
mpiexec -np 12 sh -c '
  r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
  exec ./glm53f_core_stage "$1" "$2" "$r" "$3"
' sh "$root/core" "$core" core

echo "SENTINEL glm53f_stage_rank_image_12n=OK root=$root"
