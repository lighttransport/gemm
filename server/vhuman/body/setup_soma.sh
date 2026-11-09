#!/bin/sh
# Create the interpreter used by --body-model soma (py-soma-x + CPU torch + Warp).
# usage: body/setup_soma.sh [VENV_DIR]   then: export VHUMAN_SOMA_PYTHON=VENV_DIR/bin/python
set -eu
VENV="${1:-tmp/vhuman-soma-venv}"
uv venv --python 3.12 "$VENV"
# Use the CUDA/ROCm torch wheel instead of the CPU index when GPU decoding is wanted
# (then also export VHUMAN_SOMA_DEVICE=cuda).
uv pip install --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$VENV/bin/python" "py-soma-x==0.3.6" huggingface_hub
# Fetch the SOMA-X assets (~0.9 GB, Hugging Face nvidia/SOMA-X) into the HF cache and smoke-test.
"$VENV/bin/python" -I -c "
import torch
from soma import SOMALayer
b = SOMALayer(identity_model_type='mhr', device='cpu')
o = b(torch.zeros(1, 77, 3), torch.zeros(1, 45), torch.zeros(1, 68))
assert o.vertices.shape == (1, 18056, 3), o.vertices.shape
print('SOMA-X OK:', tuple(o.vertices.shape))"
echo "export VHUMAN_SOMA_PYTHON=$(cd "$VENV" && pwd)/bin/python"
