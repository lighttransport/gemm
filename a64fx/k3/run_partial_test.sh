#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
model_dir=$HOME/models/kimi-k3
stage_root=/tmp
layer=1
expert=0
rank=0
nodes=96
usage() {
    echo "usage: $0 [--model-dir DIR] [--stage-root DIR] [--layer N]" >&2
    echo "          [--expert N] [--rank N] [--nodes N]" >&2
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
        --stage-root) need_value "$@"; stage_root=$2; shift 2 ;;
        --layer) need_value "$@"; layer=$2; shift 2 ;;
        --expert) need_value "$@"; expert=$2; shift 2 ;;
        --rank) need_value "$@"; rank=$2; shift 2 ;;
        --nodes) need_value "$@"; nodes=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "$0: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done
for value in "$layer" "$expert" "$rank" "$nodes"; do
    case "$value" in ''|*[!0-9]*) echo "$0: numeric options must be integers" >&2; exit 2 ;; esac
done
if [ "$nodes" -eq 0 ]; then echo "$0: nodes must be greater than zero" >&2; exit 2; fi
if [ "$rank" -ge "$nodes" ]; then
    echo "$0: rank $rank is outside node range [0, $((nodes - 1))]" >&2
    exit 2
fi
if [ ! -d "$stage_root" ]; then
    echo "$0: stage root is not a directory: $stage_root" >&2
    exit 2
fi
stage_dir=$(mktemp -d "$stage_root/k3-stage.XXXXXX")
cleanup() { rm -rf -- "$stage_dir"; }
trap cleanup EXIT HUP INT TERM

make -C "$script_dir" all
PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
    --model-dir "$model_dir" --output-dir "$stage_dir" --layer "$layer" \
    --expert "$expert" --rank "$rank" --nodes "$nodes"

blob="$stage_dir/layer$(printf '%02d' "$layer")_expert$(printf '%03d' "$expert").blob"
manifest="$stage_dir/layer$(printf '%02d' "$layer")_expert$(printf '%03d' "$expert").manifest"
"$script_dir/k3_real_mxfp4_test" "$blob" "$manifest" 0
"$script_dir/k3_real_mxfp4_test" "$blob" "$manifest" 1024

echo "K3 bounded partial real-weight test: PASS"
