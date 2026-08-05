#!/bin/bash
# Prepared IQ1-S/M mixed-quant staging job. Submit explicitly with pjsub.
#PJM -L "node=32"
#PJM --mpi "proc=32"
#PJM -L "elapse=01:00:00"
#PJM -j
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
MODEL=${K3_MODEL_DIR:-$HOME/models/k3/iq1}
OUT=${K3_STAGE_DIR:-/local/$USER/k3-iq1-32n-${PJM_JOBID:-manual}}
LAYER=${K3_LAYER_INDEX:-1}
mkdir -p "$OUT"
mpiexec -np 32 sh -c 'r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PJM_MPI_RANK:?no rank}}}; exec python3 "$1/k3_gguf_stage.py" --model-dir "$2" --output-dir "$3" --rank "$r" --nodes 32 --layer-index "$4" --format iq1 --force' sh "$S" "$MODEL" "$OUT" "$LAYER"
echo "K3_IQ1_STAGE_READY nodes=32 layer=$LAYER output=$OUT"
