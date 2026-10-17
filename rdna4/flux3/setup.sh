#!/bin/sh
# Optional CPU text encoder/reference environment; native runner needs only gcc/ROCm.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
mkdir -p tmp
export TMPDIR="$root/tmp" UV_CACHE_DIR="$root/tmp/uv-cache"
venv="$root/tmp/flux3-text-venv"
[ -x "$venv/bin/python" ] || uv venv --python 3.12 "$venv"
uv pip install --python "$venv/bin/python" --index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0'
uv pip install --python "$venv/bin/python" 'transformers==5.8.1' 'safetensors==0.8.0' accelerate numpy pillow
make -C rdna4/flux3
printf '\nRun with %s/bin/python rdna4/flux3/generate.py --help\n' "$venv"
