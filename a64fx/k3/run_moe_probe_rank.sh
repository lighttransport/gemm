#!/bin/sh
set -eu

if [ "$#" -ne 8 ]; then
    echo "usage: $0 SCRIPT_DIR NODES JOB_TAG MODEL_DIR LAYER EXPERTS THREADS TILE_THRESHOLD" >&2
    exit 2
fi
script_dir=$1
nodes=$2
job_tag=$3
model_dir=$4
layer=$5
probe_experts=$6
threads=$7
tile_threshold=$8
rank=${PMIX_RANK:?PMIX_RANK is not set}
stage_dir="/local/k3-moe-mpi-probe-$job_tag"
mkdir -p "$stage_dir"
cleanup() {
    if [ "${K3_KEEP_PROBE:-0}" != 1 ]; then rm -rf -- "$stage_dir"; fi
}
trap cleanup EXIT HUP INT TERM

set --
j=0
while [ "$j" -lt "$probe_experts" ]; do
    expert=$((rank + j * nodes))
    PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
        --model-dir "$model_dir" --output-dir "$stage_dir" --layer "$layer" \
        --expert "$expert" --rank "$rank" --nodes "$nodes"
    stem="layer$(printf '%02d' "$layer")_expert$(printf '%03d' "$expert")"
    set -- "$@" "$stage_dir/$stem.blob" "$stage_dir/$stem.manifest"
    j=$((j + 1))
done
"$script_dir/k3_moe_probe" --threads "$threads" \
    --tile-threshold "$tile_threshold" "$@"
