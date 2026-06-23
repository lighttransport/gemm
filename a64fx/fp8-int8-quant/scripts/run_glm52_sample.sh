#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$HOME/models/glm52-fp8}"
TENSOR="${2:-model.layers.0.self_attn.q_a_proj.weight}"

make -C "$DIR"
"$DIR/quant_analyze" \
  --model "$MODEL" \
  --tensor "$TENSOR" \
  --scheme all \
  --rows 512 \
  --cols 4096 \
  --block 128 \
  --svd-rank 4
