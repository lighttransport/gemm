#!/bin/sh
# Isolated ROCm interpreter; all caches and temporary files stay in this tree.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
export TMPDIR="$root/tmp/vhuman-rocm-tmp"
export UV_CACHE_DIR="$root/tmp/uv-cache"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
export LD_LIBRARY_PATH="/opt/rocm/core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python="$root/tmp/vhuman-rocm-venv/bin/python"
if [ ! -x "$python" ]; then uv venv --python 3.12 "$root/tmp/vhuman-rocm-venv"; fi
# Reuse the RDNA4 wheels pinned by ref/pixal3d; keep the system interpreter intact.
uv pip install --python "$python" \
 'torch @ https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torch-2.11.0+rocm7.2.2.lw.git4e323059-cp312-cp312-linux_x86_64.whl' \
 'torchvision @ https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torchvision-0.26.0+rocm7.2.2.git336d36e8-cp312-cp312-linux_x86_64.whl' \
 'triton @ https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/triton-3.6.0+rocm7.2.2.git4ed88892-cp312-cp312-linux_x86_64.whl' \
 numpy pillow scipy opencv-python-headless safetensors timm einops easydict trimesh xatlas psutil \
 accelerate qwen-vl-utils click matplotlib kornia \
 'utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183' \
 'pipeline @ git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6' \
 'diffusers @ git+https://github.com/huggingface/diffusers@80c7ed262aeffbeb43ef13ae04baeb9b84515a69' \
 'transformers @ git+https://github.com/huggingface/transformers@c587bc884db2c2e31fc2b8102314656b17aa07b1'
"$python" -B -c 'import torch; assert torch.version.hip, "expected a ROCm PyTorch build"; print(torch.__version__, torch.version.hip)'

cd "$root"
"$python" -B cuda/qimg21/export_text_rope.py --out rdna4/qimg21/qwen21_text_rope.npy
"$python" -B cuda/qimg21/export_vision_rope.py --out rdna4/qimg21/qwen21_vision_rope.npy
"$python" -B cuda/qimg21/export_rope_frequencies.py --out rdna4/qimg21/qwen21_rope_freqs.npy

# Pin the standalone gfx12 SageAttention implementation and AMD WMMA headers.
if [ ! -d tmp/sageattention-gfx12/.git ]; then
    git clone --no-checkout https://github.com/jammm/SageAttention.git tmp/sageattention-gfx12
    git -C tmp/sageattention-gfx12 checkout --detach 30c94f510a0e0c54532ea3f940a69fac3b9fcd3e
fi
if [ ! -d tmp/rocwmma/.git ]; then
    git clone --no-checkout https://github.com/ROCm/rocWMMA.git tmp/rocwmma
    git -C tmp/rocwmma checkout --detach 48b7db12a9ade97f0b7ab2ff9321ba0cbb4e5b77
fi
[ "$(git -C tmp/rocwmma rev-parse HEAD)" = 48b7db12a9ade97f0b7ab2ff9321ba0cbb4e5b77 ]
"$python" -B rdna4/qimg21/extract_sage.py tmp/qimg21/sage_device.inc
"$python" -B -m server.vhuman.fetch_rocm_assets

if [ ! -d ref/pixal3d/moge-upstream/.git ]; then
    git clone --no-checkout https://github.com/microsoft/MoGe.git ref/pixal3d/moge-upstream
    git -C ref/pixal3d/moge-upstream checkout --detach b942f00bdc2a2a23ebb474fbe034d487e6dcceec
fi
[ "$(git -C ref/pixal3d/moge-upstream rev-parse HEAD)" = b942f00bdc2a2a23ebb474fbe034d487e6dcceec ]
