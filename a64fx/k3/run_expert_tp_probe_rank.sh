#!/bin/sh
set -eu

if [ "$#" -ne 10 ]; then
    echo "usage: $0 SCRIPT_DIR NODES JOB_TAG MODEL_DIR LAYER EXPERTS THREADS PREFILL LOGICAL_TP LOGICAL_WAVES" >&2
    exit 2
fi
script_dir=$1
nodes=$2
job_tag=$3
model_dir=$4
layer=$5
experts=$6
threads=$7
prefill=$8
logical_tp=$9
logical_waves=${10}
rank=${PMIX_RANK:?PMIX_RANK is not set}
stage_root="/local/k3-etp-probe-$job_tag"
mkdir -p "$stage_root"
cleanup() {
    if [ "${K3_KEEP_PROBE:-0}" != 1 ]; then rm -rf -- "$stage_root"; fi
}
trap cleanup EXIT HUP INT TERM

probe_flag=--tp-selected
if [ "$prefill" -eq 1 ]; then probe_flag=--prefill; fi
wave=0
while [ "$wave" -lt "$logical_waves" ]; do
    logical_rank=$((rank + wave * nodes))
    stage_dir="$stage_root/wave$(printf '%02d' "$wave")"
    echo "LOGICAL_SAMPLE_BEGIN physical_rank=$rank wave=$wave logical_rank=$logical_rank logical_tp=$logical_tp"
    PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
        --model-dir "$model_dir" --output-dir "$stage_dir" \
        --layer "$layer" --experts "0-$((experts-1))" \
        --expert-tp --tp-size "$logical_tp" --tp-rank "$logical_rank" --force
    set --
    expert=0
    while [ "$expert" -lt "$experts" ]; do
        expert_dir="$stage_dir/expert$(printf '%03d' "$expert")"
        stem="layer$(printf '%02d' "$layer")_expert$(printf '%03d' "$expert")"
        set -- "$@" "$expert_dir/$stem.blob" "$expert_dir/$stem.manifest"
        expert=$((expert + 1))
    done
    "$script_dir/k3_moe_probe" "$probe_flag" --threads "$threads" "$@"
    echo "LOGICAL_SAMPLE_END physical_rank=$rank wave=$wave logical_rank=$logical_rank logical_tp=$logical_tp"
    wave=$((wave + 1))
done
