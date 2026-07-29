#!/bin/sh
set -eu

if [ "$#" -ne 7 ]; then
    echo "usage: $0 SCRIPT_DIR NODES MODEL_DIR STAGE_DIR LAYER EXPERTS CHUNK_MIB" >&2
    exit 2
fi
script_dir=$1
nodes=$2
model_dir=$3
stage_dir=$4
layer=$5
experts=$6
chunk_mib=$7
rank=${PMIX_RANK:?PMIX_RANK is not set}
if [ -e "$stage_dir" ]; then
    echo "$0: refusing to overwrite pre-existing rank-local stage: $stage_dir" >&2
    exit 2
fi

PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
    --model-dir "$model_dir" --output-dir "$stage_dir" \
    --layer "$layer" --experts "$experts" --nodes "$nodes" --rank "$rank" \
    --expert-tp --tp-size "$nodes" --tp-rank "$rank" \
    --chunk-mib "$chunk_mib"

marker="$stage_dir/stage-rank$(printf '%03d' "$rank").status"
marker_tmp="$marker.tmp.$$"
trap 'rm -f "$marker_tmp"' EXIT HUP INT TERM
printf 'rank=%s nodes=%s layer=%s experts=%s\n' "$rank" "$nodes" "$layer" "$experts" > "$marker_tmp"
sync -f "$marker_tmp"
mv "$marker_tmp" "$marker"
trap - EXIT HUP INT TERM
