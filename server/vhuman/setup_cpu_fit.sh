#!/bin/sh
# Run from any directory. All environments/caches/build files stay in the repo.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
mkdir -p tmp/vhuman-cpu-build tmp/uv-cpufit
export TMPDIR="$root/tmp/vhuman-cpu-build"
export UV_CACHE_DIR="$root/tmp/uv-cpufit"
export PYTHONDONTWRITEBYTECODE=1
if ! command -v uv >/dev/null 2>&1; then
    echo 'Install uv first; CPU fitting setup requires uv and Python 3.11/3.12.' >&2
    exit 1
fi
if [ ! -x tmp/vhuman-cpu-venv/bin/python ]; then
    uv venv --python "${1:-python3.12}" tmp/vhuman-cpu-venv
fi
tmp/vhuman-cpu-venv/bin/python -B - <<'PY'
import sys
if sys.version_info[:2] not in ((3,11),(3,12)):
    raise SystemExit('CPU fitting environment requires Python 3.11 or 3.12.')
PY
# https://pytorch.org/get-started/previous-versions/#v291
uv pip install --python tmp/vhuman-cpu-venv/bin/python 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python tmp/vhuman-cpu-venv/bin/python -r server/vhuman/requirements-cpu-fit.lock
make -C cpu/vhuman libvhuman_landmarks.so libvhuman_training.so
tmp/vhuman-cpu-venv/bin/python -B - <<'PY'
import torch
assert torch.version.cuda is None and torch.version.hip is None, 'CPU Torch wheel required'
from server.vhuman import gpu
with gpu.execution('cpu'):
    from server.vhuman.face_assets import asset_path
    for name in ('gnm','mediapipe','parsing'):
        if not asset_path(name).is_file():
            raise SystemExit(f'{name} asset missing: run python -m server.vhuman.face_assets --help')
print('CPU fitting environment ready; no GPU discovery performed.')
PY
uv pip freeze --python tmp/vhuman-cpu-venv/bin/python > tmp/vhuman-cpu-build/installed.txt
