#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
test "${PJM_MPI_PROC:-12}" -eq 12

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-47}
export OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-active}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

model=${GLM53F_MODEL_DIR:-$HOME/models/glm53f}
layer=${GLM53F_KDA_LAYER:-44}
exec mpiexec -np 12 ./glm53f_kda_layer_12n "$model" "$layer"
