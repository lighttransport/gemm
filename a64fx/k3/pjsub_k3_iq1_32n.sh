#!/bin/bash
# Prepared IQ1-S/M mixed-quant staging job. Submit explicitly with pjsub.
#PJM -g hp250467
#PJM -L "node=32"
#PJM --mpi "proc=32"
#PJM -L "elapse=01:00:00"
#PJM -j
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
MODEL=${K3_MODEL_DIR:-$HOME/models/k3/iq1}
OUT=${K3_STAGE_DIR:-/local/$USER/k3-iq1-32n-${PJM_JOBID:-manual}}
LAYER=${K3_LAYER_INDEX:-1}
REPS=${K3_BENCH_REPS:-2}
mkdir -p "$OUT"
mpiexec -np 32 sh -c 'r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PJM_MPI_RANK:?no rank}}}; exec python3 "$1/k3_gguf_stage.py" --model-dir "$2" --output-dir "$3" --rank "$r" --nodes 32 --layer-index "$4" --format iq1 --force' sh "$S" "$MODEL" "$OUT" "$LAYER"
mpiexec -np 32 -of-proc "$OUT/bench" sh -c '
    r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PJM_MPI_RANK:?no rank}}}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48} K3_QUANT_KERNEL=${K3_QUANT_KERNEL:-sve-q8}
    exec "$1/k3_gguf_layer_bench" "$2/rank$(printf "%03d" "$r").manifest" \
        "$2/rank$(printf "%03d" "$r").blob" "$3"
' sh "$S" "$OUT" "$REPS"
"$S/run_k3_ep.sh" --mode dummy --nodes 32 --tp-nodes 32 \
    --layers 1 --layer 3 --tokens 1 --cache-tokens 16384 \
    --threads 47 --kda-threads 8 --fused-threads 47 \
    --mla-cache-int8 --heartbeat-tokens 1 --min-available-mib 2048 \
    --ar-groups 2 --comm-robust 2 --result-dir "$OUT/context-preflight"
echo "K3_IQ1_STAGE_READY nodes=32 layer=$LAYER output=$OUT"
echo "K3_IQ1_DEQUANT_CHECK nodes=32 layer=$LAYER kernel=${K3_QUANT_KERNEL:-sve-q8} logs=$OUT/bench.*"
echo "K3_IQ1_CONTEXT_CHECK nodes=32 cache_tokens=16384 cache=int8 result=$OUT/context-preflight"
