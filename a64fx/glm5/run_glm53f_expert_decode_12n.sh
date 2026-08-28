#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
np=${PJM_MPI_PROC:-12}
if [ "$np" -ne 12 ]; then
    echo "expected 12-node allocation, PJM_MPI_PROC=$np" >&2
    exit 2
fi

stage=${GLM53F_STAGE_DIR:-/local/glm53f-decode-${PJM_JOBID}}
tokens=${GLM53F_DECODE_TOKENS:-20}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
ofp=()
if [ -n "${MPIEXEC_OF_PROC:-}" ]; then
    ofp=(-of-proc "$MPIEXEC_OF_PROC")
fi

test -x ./glm53f_expert_decode_12n
echo "GLM53F expert decode job=$PJM_JOBID stage=$stage tokens=$tokens threads=$OMP_NUM_THREADS"
date
mpiexec -np 12 "${ofp[@]}" ./glm53f_expert_decode_12n "$stage" "$tokens"
echo "SENTINEL glm53f_expert_decode_12n=OK"
date
