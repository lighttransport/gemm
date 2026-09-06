#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
nodes=${PJM_NODE:-1}
job_tag=${PJM_JOBID:-$$}
stage_dir=$HOME/.cache/k3-kda-probe-$job_tag
model_dir=$HOME/models/kimi-k3
layer=0
head=0
usage() {
    echo "usage: $0 [--nodes N] [--stage-dir DIR] [--model-dir DIR] [--layer N] [--head N]" >&2
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
        --nodes) need_value "$@"; nodes=$2; shift 2 ;;
        --stage-dir) need_value "$@"; stage_dir=$2; shift 2 ;;
        --model-dir) need_value "$@"; model_dir=$2; shift 2 ;;
        --layer) need_value "$@"; layer=$2; shift 2 ;;
        --head) need_value "$@"; head=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "$0: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done
for value in "$nodes" "$layer" "$head"; do
    case "$value" in ''|*[!0-9]*) echo "$0: nodes, layer, and head must be integers" >&2; exit 2 ;; esac
done
if [ "$nodes" -eq 0 ]; then echo "$0: nodes must be greater than zero" >&2; exit 2; fi
result_prefix="$stage_dir/result"
created=0
cleanup() {
    if [ "$created" = 1 ] && [ "${K3_KEEP_PROBE:-0}" != 1 ]; then rm -rf -- "$stage_dir"; fi
}
trap cleanup EXIT HUP INT TERM

if [ -e "$stage_dir" ]; then
    echo "$0: stage directory already exists: $stage_dir" >&2
    exit 2
fi
mkdir -p "$stage_dir"
created=1
make -C "$script_dir" k3_kda_probe
PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_kda_stage.py" \
    --model-dir "$model_dir" --output-dir "$stage_dir" --layer "$layer" \
    --head "$head"

stem="layer$(printf '%02d' "$layer")_head$(printf '%02d' "$head")"
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
mpiexec -n "$nodes" -of-proc "$result_prefix" \
    -x OMP_PROC_BIND=close -x OMP_PLACES=cores \
    -x XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
    "$script_dir/k3_kda_probe" "$stage_dir/$stem.blob" "$stage_dir/$stem.manifest"

for result in "$result_prefix".*; do
    echo "===== $result ====="
    cat "$result"
done
