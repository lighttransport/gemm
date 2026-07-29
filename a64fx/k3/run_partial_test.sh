#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
stage_root=${K3_STAGE_TMPDIR:-/tmp}
stage_dir=$(mktemp -d "$stage_root/k3-stage.XXXXXX")
cleanup() { rm -rf -- "$stage_dir"; }
trap cleanup EXIT HUP INT TERM

make -C "$script_dir" all
PYTHONDONTWRITEBYTECODE=1 python3 "$script_dir/k3_stage.py" \
    --model-dir "${K3_MODEL_DIR:-$HOME/models/kimi-k3}" \
    --output-dir "$stage_dir" --layer "${K3_TEST_LAYER:-1}" \
    --expert "${K3_TEST_EXPERT:-0}" --rank "${K3_TEST_RANK:-0}" \
    --nodes "${K3_TEST_NODES:-96}"

blob="$stage_dir/layer$(printf '%02d' "${K3_TEST_LAYER:-1}")_expert$(printf '%03d' "${K3_TEST_EXPERT:-0}").blob"
manifest="$stage_dir/layer$(printf '%02d' "${K3_TEST_LAYER:-1}")_expert$(printf '%03d' "${K3_TEST_EXPERT:-0}").manifest"
"$script_dir/k3_real_mxfp4_test" "$blob" "$manifest" 0
"$script_dir/k3_real_mxfp4_test" "$blob" "$manifest" 1024

echo "K3 bounded partial real-weight test: PASS"
