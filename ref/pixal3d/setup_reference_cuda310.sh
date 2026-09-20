#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export TMPDIR="$project_dir/../../tmp/pixal3d"
export UV_CACHE_DIR="$project_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$project_dir/.cache/python"
environment="$project_dir/.venv-reference-cuda310"
mkdir -p "$TMPDIR"
uv venv --python 3.10 "$environment"
rm -f "$environment/.pixal3d-reference-ready"
uv pip install --python "$environment/bin/python" \
  --index-strategy unsafe-best-match \
  -r "$project_dir/requirements-reference-cuda310.txt"
clone_pinned() {
  url=$1
  directory=$2
  revision=$3
  if [ ! -d "$directory/.git" ]; then
    git clone --recursive "$url" "$directory"
  fi
  git -C "$directory" fetch origin "$revision"
  git -C "$directory" checkout --detach "$revision"
  git -C "$directory" submodule update --init --recursive
}
clone_pinned https://github.com/JeffreyXiang/CuMesh.git \
  "$project_dir/cumesh-upstream" 12289e1062f0603f2f0d0771b02e1395d247f26f
clone_pinned https://github.com/JeffreyXiang/FlexGEMM.git \
  "$project_dir/flexgemm-upstream" 6dd94a859c26ee8246888502eada3dd8ad85532e
clone_pinned https://github.com/quantaji/o-voxel-gpu.git \
  "$project_dir/o-voxel-upstream" e477c69457c650dc680c02c2a1fb2ca58d3429ad
clone_pinned https://github.com/NVlabs/nvdiffrast.git \
  "$project_dir/nvdiffrast-upstream" 253ac4fcea7de5f396371124af597e6cc957bfae
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export MAX_JOBS=${MAX_JOBS:-4}
site_packages="$environment/lib/python3.10/site-packages"
export CPATH="$site_packages/nvidia/curand/include:$site_packages/nvidia/cusparse/include:$site_packages/nvidia/cublas/include:$site_packages/nvidia/cusolver/include${CPATH:+:$CPATH}"
for source in cumesh-upstream flexgemm-upstream o-voxel-upstream nvdiffrast-upstream; do
  uv pip install --python "$environment/bin/python" --no-build-isolation \
    --no-deps --reinstall "$project_dir/$source"
done
"$environment/bin/python" -c \
  'import torch, o_voxel, cumesh, flex_gemm, nvdiffrast.torch, natten; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability() >= (12, 0); print(torch.__version__, torch.cuda.get_device_name())'
touch "$environment/.pixal3d-reference-ready"
