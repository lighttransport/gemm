#!/bin/sh
# Build/cache settings only; runtime tuning is supplied as explicit CLI arguments.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$root"
work="$root/tmp/vhuman-realtime"
mkdir -p "$work/cache/tmp" "$work/cache/torch_rgb"
export TMPDIR="$work/cache/tmp"
export HF_HOME="$work/cache/hf"
export TORCH_EXTENSIONS_DIR="$work/cache/torch_rgb"
export TORCH_CUDA_ARCH_LIST=12.0
export BUILD_3DGS=1 BUILD_3DGUT=0 BUILD_2DGS=0 BUILD_ADAM=0 BUILD_RELOC=0 BUILD_LOSSES=0 NUM_CHANNELS=3 MAX_JOBS=4
export PYTHONPATH="$work/deps:$work/upstream/gsplat${PYTHONPATH:+:$PYTHONPATH}"
exec "${VHUMAN_PYTHON:-$root/tmp/vhuman-rig-venv/bin/python}" -m server.vhuman.realtime "$@"
