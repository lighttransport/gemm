#!/bin/sh
set -eu

script_dir=$1
nodes=$2
job_tag=$3
rank=${PMIX_RANK:?PMIX_RANK is not set}
stage_dir="/local/k3-etp-probe-$job_tag"
mkdir -p "$stage_dir"
cleanup() {
    if [ "${K3_KEEP_PROBE:-0}" != 1 ]; then rm -rf -- "$stage_dir"; fi
}
trap cleanup EXIT HUP INT TERM

set --
expert=0
while [ "$expert" -lt "${K3_TP_EXPERTS:-16}" ]; do
    expert_dir="$stage_dir/expert$expert"
    PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
        --model-dir "${K3_MODEL_DIR:-$HOME/models/kimi-k3}" \
        --output-dir "$expert_dir" --layer "${K3_MOE_LAYER:-1}" --expert "$expert" \
        --expert-tp --tp-size "$nodes" --tp-rank "$rank" --force
    stem="layer$(printf '%02d' "${K3_MOE_LAYER:-1}")_expert$(printf '%03d' "$expert")"
    set -- "$@" "$expert_dir/$stem.blob" "$expert_dir/$stem.manifest"
    expert=$((expert + 1))
done
K3_TP_SELECTED=1 "$script_dir/k3_moe_probe" "$@"
