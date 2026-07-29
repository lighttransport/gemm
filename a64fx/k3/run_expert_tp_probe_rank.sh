#!/bin/sh
set -eu

if [ "$#" -ne 7 ]; then
    echo "usage: $0 SCRIPT_DIR NODES JOB_TAG MODEL_DIR LAYER EXPERTS THREADS" >&2
    exit 2
fi
script_dir=$1
nodes=$2
job_tag=$3
model_dir=$4
layer=$5
experts=$6
threads=$7
rank=${PMIX_RANK:?PMIX_RANK is not set}
stage_dir="/local/k3-etp-probe-$job_tag"
mkdir -p "$stage_dir"
cleanup() {
    if [ "${K3_KEEP_PROBE:-0}" != 1 ]; then rm -rf -- "$stage_dir"; fi
}
trap cleanup EXIT HUP INT TERM

set --
expert=0
while [ "$expert" -lt "$experts" ]; do
    expert_dir="$stage_dir/expert$expert"
    PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
        --model-dir "$model_dir" --output-dir "$expert_dir" \
        --layer "$layer" --expert "$expert" \
        --expert-tp --tp-size "$nodes" --tp-rank "$rank" --force
    stem="layer$(printf '%02d' "$layer")_expert$(printf '%03d' "$expert")"
    set -- "$@" "$expert_dir/$stem.blob" "$expert_dir/$stem.manifest"
    expert=$((expert + 1))
done
"$script_dir/k3_moe_probe" --tp-selected --threads "$threads" "$@"
