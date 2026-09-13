#!/bin/sh
# Materialize one dense-TP/shared-TP rank after the base and dense stages.
# The original root must be the rank-local base image so Engram and owner-only
# tensors remain node-local symlinks while row shards come from safetensors.
set -eu
rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}}
test -n "$rank"
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
model=${DS41F_MODEL_DIR:-/home/u14346/models/ds41f}
base=${DS41F_BASE_ROOT:?set DS41F_BASE_ROOT to the rank-local base stage root}
destination=${DS41F_TP_ROOT:?set DS41F_TP_ROOT to the output stage root}
dense_tp=${DS41F_DENSE_TP:-4}
shared_tp=${DS41F_SHARED_TP:-12}
set -- --model "$model" --original "$base/rank$rank" --destination "$destination/rank$rank" \
    --rank "$rank" --tp "$dense_tp" --shared-tp "$shared_tp"
if test "${DS41F_ATTENTION_TP12:-0}" = 1; then
    set -- "$@" --attention-tp12
fi
exec python3 "$here/stage_tp.py" "$@"
