#!/bin/bash
# Prepared Q2_K-XL mixed-quant staging job. Submit explicitly with pjsub.
#PJM -L "node=48"
#PJM --mpi "proc=48"
#PJM -L "elapse=01:00:00"
#PJM -j
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
MODEL=${K3_MODEL_DIR:-$HOME/models/k3/q2}
OUT=${K3_STAGE_DIR:-/local/$USER/k3-q2-48n-${PJM_JOBID:-manual}}
LAYER=${K3_LAYER_INDEX:-1}
mkdir -p "$OUT"
mpiexec -np 48 sh -c 'r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PJM_MPI_RANK:?no rank}}}; exec python3 "$1/k3_gguf_stage.py" --model-dir "$2" --output-dir "$3" --rank "$r" --nodes 48 --layer-index "$4" --format q2 --force' sh "$S" "$MODEL" "$OUT" "$LAYER"
echo "K3_Q2_STAGE_READY nodes=48 layer=$LAYER output=$OUT"
