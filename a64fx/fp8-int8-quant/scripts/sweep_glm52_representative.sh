#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-$HOME/models/glm52-fp8}"
OUT="${2:-$DIR/glm52_representative_sweep.csv}"
ROWS="${ROWS:-256}"
COLS="${COLS:-2048}"
BLOCK="${BLOCK:-128}"
SVD_RANK="${SVD_RANK:-4}"

TENSORS=(
  model.layers.0.self_attn.q_a_proj.weight
  model.layers.0.self_attn.q_b_proj.weight
  model.layers.0.self_attn.kv_a_proj_with_mqa.weight
  model.layers.0.self_attn.kv_b_proj.weight
  model.layers.0.self_attn.o_proj.weight
  model.layers.0.mlp.gate_proj.weight
  model.layers.0.mlp.up_proj.weight
  model.layers.0.mlp.down_proj.weight
  model.layers.3.mlp.experts.0.gate_proj.weight
  model.layers.3.mlp.experts.0.up_proj.weight
  model.layers.3.mlp.experts.0.down_proj.weight
)

make -C "$DIR"
rm -f "$OUT"

for tensor in "${TENSORS[@]}"; do
  echo "== $tensor"
  "$DIR/quant_analyze" \
    --model "$MODEL" \
    --tensor "$tensor" \
    --scheme all \
    --rows "$ROWS" \
    --cols "$COLS" \
    --block "$BLOCK" \
    --svd-rank "$SVD_RANK" \
    --csv "$OUT"
done

python3 - "$OUT" <<'PY'
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="")))
by_tensor = defaultdict(list)
for r in rows:
    by_tensor[r["tensor"]].append(r)

print("\nsummary_by_rel_l2")
print("tensor,best_scheme,best_rel_l2,row_rel_l2,block_rel_l2,block_mse_rel_l2,block_p99_rel_l2,smooth_rel_l2,awq_rel_l2,svd_rel_l2,i16_rel_l2,i16_row_rel_l2,i16_block_rel_l2,i16_awq_rel_l2")
for tensor in sorted(by_tensor):
    vals = by_tensor[tensor]
    best = min(vals, key=lambda r: float(r["rel_l2"]))
    m = {r["scheme"]: r for r in vals}
    def rel(name):
        return m.get(name, {}).get("rel_l2", "")
    print(",".join([
        tensor,
        best["scheme"],
        best["rel_l2"],
        rel("row"),
        rel("block"),
        rel("block_mse"),
        rel("block_p99"),
        rel("smooth"),
        rel("awq"),
        rel("svd"),
        rel("i16"),
        rel("i16_row"),
        rel("i16_block"),
        rel("i16_awq"),
    ]))
PY
