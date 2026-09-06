#!/bin/bash
# Rank-local K3 checkpoint staging.  full96 stages the complete rank image;
# layer12 stages one selected layer for instant 12-node debug.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=${1:-${HOME}/models/kimi-k3}
OUTPUT_DIR=${2:?usage: $0 MODEL_DIR OUTPUT_DIR NODES RANK}
NODES=${3:?usage: $0 MODEL_DIR OUTPUT_DIR NODES RANK}
RANK=${4:?usage: $0 MODEL_DIR OUTPUT_DIR NODES RANK}
CHUNK_MIB=${CHUNK_MIB:-8}
MODE=${5:-full96}
LAYER_INDEX=${6:-}

if [ -n "$LAYER_INDEX" ]; then
    exec python3 "$SCRIPT_DIR/k3_full_stage.py" \
        --model-dir "$MODEL_DIR" --output-dir "$OUTPUT_DIR" \
        --nodes "$NODES" --rank "$RANK" --mode "$MODE" \
        --layer-index "$LAYER_INDEX" --chunk-mib "$CHUNK_MIB"
fi
exec python3 "$SCRIPT_DIR/k3_full_stage.py" \
    --model-dir "$MODEL_DIR" --output-dir "$OUTPUT_DIR" \
    --nodes "$NODES" --rank "$RANK" --mode "$MODE" --chunk-mib "$CHUNK_MIB"
