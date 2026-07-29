#!/bin/sh
set -eu

script_dir=$1
nodes=$2
job_tag=$3
rank=${PMIX_RANK:?PMIX_RANK is not set}
stage_dir="/local/k3-moe-mpi-probe-$job_tag"
mkdir -p "$stage_dir"
cleanup() {
    if [ "${K3_KEEP_PROBE:-0}" != 1 ]; then rm -rf -- "$stage_dir"; fi
}
trap cleanup EXIT HUP INT TERM

set --
probe_experts=${K3_MOE_PROBE_EXPERTS:-4}
j=0
while [ "$j" -lt "$probe_experts" ]; do
    expert=$((rank + j * nodes))
    PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
        --model-dir "${K3_MODEL_DIR:-$HOME/models/kimi-k3}" \
        --output-dir "$stage_dir" --layer "${K3_MOE_LAYER:-1}" \
        --expert "$expert" --rank "$rank" --nodes "$nodes"
    stem="layer$(printf '%02d' "${K3_MOE_LAYER:-1}")_expert$(printf '%03d' "$expert")"
    set -- "$@" "$stage_dir/$stem.blob" "$stage_dir/$stem.manifest"
    j=$((j + 1))
done
"$script_dir/k3_moe_probe" "$@"
