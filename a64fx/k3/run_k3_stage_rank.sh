#!/bin/sh
set -eu

if [ "$#" -ne 8 ]; then
    echo "usage: $0 SCRIPT_DIR WORLD_NODES TP_NODES MODEL_DIR STAGE_DIR LAYER EXPERTS CHUNK_MIB" >&2
    exit 2
fi
script_dir=$1
world_nodes=$2
tp_nodes=$3
model_dir=$4
stage_dir=$5
layer=$6
experts=$7
chunk_mib=$8
rank=${PMIX_RANK:?PMIX_RANK is not set}
tp_rank=$((rank % tp_nodes))
if [ -e "$stage_dir" ]; then
    echo "$0: refusing to overwrite pre-existing rank-local stage: $stage_dir" >&2
    exit 2
fi
stage_tmp="${stage_dir}.tmp.rank${rank}.$$"
if [ -e "$stage_tmp" ]; then
    echo "$0: temporary stage path already exists: $stage_tmp" >&2
    exit 2
fi

PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
    --model-dir "$model_dir" --output-dir "$stage_tmp" \
    --layer "$layer" --experts "$experts" --nodes "$world_nodes" --rank "$rank" \
    --expert-tp --tp-size "$tp_nodes" --tp-rank "$tp_rank" \
    --chunk-mib "$chunk_mib"

marker="$stage_tmp/stage-rank$(printf '%03d' "$rank").status"
marker_tmp="$marker.tmp.$$"
trap 'rm -f "$marker_tmp"' EXIT HUP INT TERM
printf 'rank=%s nodes=%s tp_rank=%s tp_nodes=%s layer=%s experts=%s\n' \
    "$rank" "$world_nodes" "$tp_rank" "$tp_nodes" "$layer" "$experts" > "$marker_tmp"
mv "$marker_tmp" "$marker"
trap - EXIT HUP INT TERM
sync -f "$marker"
mv -T "$stage_tmp" "$stage_dir"
