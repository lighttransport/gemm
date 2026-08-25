#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
model_dir=$HOME/models/kimi-k3
layer=0
head=0
stage_root=/tmp
trace=
usage() {
    echo "usage: $0 [--model-dir DIR] [--layer N] [--head N] [--stage-root DIR] [--trace FILE]" >&2
}
need_value() {
    if [ "$#" -lt 2 ]; then
        echo "$0: missing value for $1" >&2
        usage
        exit 2
    fi
}
while [ "$#" -gt 0 ]; do
    case $1 in
        --model-dir) need_value "$@"; model_dir=$2; shift 2 ;;
        --layer) need_value "$@"; layer=$2; shift 2 ;;
        --head) need_value "$@"; head=$2; shift 2 ;;
        --stage-root) need_value "$@"; stage_root=$2; shift 2 ;;
        --trace) need_value "$@"; trace=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "$0: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done
for value in "$layer" "$head"; do
    case "$value" in ''|*[!0-9]*) echo "$0: layer and head must be non-negative integers" >&2; exit 2 ;; esac
done
if [ ! -d "$stage_root" ]; then
    echo "$0: stage root is not a directory: $stage_root" >&2
    exit 2
fi
stage_dir=$(mktemp -d "$stage_root/k3-kda-stage.XXXXXX")
cleanup() { rm -rf -- "$stage_dir"; }
trap cleanup EXIT HUP INT TERM

make -C "$script_dir" k3_kda_probe
PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_kda_stage.py" \
    --model-dir "$model_dir" --output-dir "$stage_dir" --layer "$layer" \
    --head "$head"

stem="layer$(printf '%02d' "$layer")_head$(printf '%02d' "$head")"
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
if [ -n "$trace" ]; then
    OMP_PROC_BIND=close OMP_PLACES=cores "$script_dir/k3_kda_probe" \
        "$stage_dir/$stem.blob" "$stage_dir/$stem.manifest" "$trace"
else
    OMP_PROC_BIND=close OMP_PLACES=cores "$script_dir/k3_kda_probe" \
        "$stage_dir/$stem.blob" "$stage_dir/$stem.manifest"
fi
