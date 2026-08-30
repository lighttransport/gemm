#!/bin/sh
# Sweep the resident-HBM/uTofu owner knobs.  Run inside the allocated job.
set -eu

bin=${Q38FN_UTOFU_PIPELINE_BIN:-./q38fn/ngram_utofu_pipeline_probe}
model=${1:?usage: $0 MODEL_DIR [TOPOLOGY] [RANKS] [ITERATIONS] [WORKERS] [WINDOW]}
topology=${2:-tofu_topo.txt}
ranks=${3:-4}
iterations=${4:-1000}
workers=${5:-8}
window=${6:-4}

if [ ! -x "$bin" ]; then
    echo "missing executable: $bin" >&2
    exit 1
fi

for pin in 1 0; do
    for distance in 0 2 4 8; do
        echo "Q38FN_SWEEP pin_service=$pin prefetch_rows=$distance"
        Q38FN_UTOFU_PIN_SERVICE=$pin \
        Q38FN_HBM_PREFETCH_ROWS=$distance \
        mpiexec -np "$ranks" "$bin" "$model" "$topology" "$ranks" \
            "$iterations" "$workers" "$window" resident
    done
done
