#!/bin/bash
# Rank-local K3 checkpoint staging.  full96 stages the complete rank image;
# layer12 stages one selected layer for instant 12-node debug; with
# K3_EXPERT_TP=1 it uses the same fused expert-TP layout as full96.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=${1:-${HOME}/models/kimi-k3}
OUTPUT_DIR=${2:?usage: $0 MODEL_DIR OUTPUT_DIR NODES RANK}
NODES=${3:?usage: $0 MODEL_DIR OUTPUT_DIR NODES RANK}
RANK=${4:?usage: $0 MODEL_DIR OUTPUT_DIR NODES RANK}
CHUNK_MIB=${CHUNK_MIB:-8}
MODE=${5:-full96}
LAYER_INDEX=${6:-}
EXPERT_TP=${7:-${K3_EXPERT_TP:-0}}
MOE_SHARD_LAYOUT=${8:-${K3_MOE_SHARD_LAYOUT:-replicated}}
CHUNK_MIB=${9:-${CHUNK_MIB:-8}}
PYTHON=${K3_PYTHON:-$SCRIPT_DIR/.venv-$(uname -m)/bin/python}

EXPERT_ARGS=()
if [ "$EXPERT_TP" = 1 ]; then EXPERT_ARGS+=(--expert-tp); fi
EXPERT_ARGS+=(--moe-shard-layout "$MOE_SHARD_LAYOUT" --chunk-mib "$CHUNK_MIB")

if [ -n "$LAYER_INDEX" ]; then
    exec "$PYTHON" "$SCRIPT_DIR/k3_full_stage.py" \
        --model-dir "$MODEL_DIR" --output-dir "$OUTPUT_DIR" \
        --nodes "$NODES" --rank "$RANK" --mode "$MODE" \
        --layer-index "$LAYER_INDEX" --chunk-mib "$CHUNK_MIB" "${EXPERT_ARGS[@]}"
fi
exec "$PYTHON" "$SCRIPT_DIR/k3_full_stage.py" \
    --model-dir "$MODEL_DIR" --output-dir "$OUTPUT_DIR" \
    --nodes "$NODES" --rank "$RANK" --mode "$MODE" --chunk-mib "$CHUNK_MIB" \
    "${EXPERT_ARGS[@]}"
