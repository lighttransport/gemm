#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
stage_root=${K3_STAGE_TMPDIR:-/tmp}
stage_dir=$(mktemp -d "$stage_root/k3-kda-stage.XXXXXX")
cleanup() { rm -rf -- "$stage_dir"; }
trap cleanup EXIT HUP INT TERM

make -C "$script_dir" k3_kda_probe
PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_kda_stage.py" \
    --model-dir "${K3_MODEL_DIR:-$HOME/models/kimi-k3}" \
    --output-dir "$stage_dir" --layer "${K3_KDA_LAYER:-0}" \
    --head "${K3_KDA_HEAD:-0}"

stem="layer$(printf '%02d' "${K3_KDA_LAYER:-0}")_head$(printf '%02d' "${K3_KDA_HEAD:-0}")"
OMP_PROC_BIND=close OMP_PLACES=cores "$script_dir/k3_kda_probe" \
    "$stage_dir/$stem.blob" "$stage_dir/$stem.manifest"
