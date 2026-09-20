#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend=${1:?Usage: run_reference_cuda310.sh cuda SCRIPT [ARGS...]}
shift
if [ "$backend" != cuda ]; then
    echo "Pinned upstream reference environment supports CUDA only: $backend" >&2
    exit 2
fi
environment="$project_dir/.venv-reference-cuda310"
if [ ! -x "$environment/bin/python" ]; then
    echo "Missing reference environment; run ref/pixal3d/setup_reference_cuda310.sh" >&2
    exit 2
fi
export TMPDIR="$project_dir/../../tmp/pixal3d"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_EXTENSIONS_DIR="$project_dir/.cache/extensions-reference-cuda310"
export TORCHINDUCTOR_CACHE_DIR="$project_dir/.cache/inductor-reference-cuda310"
export TRITON_CACHE_DIR="$project_dir/.cache/triton-reference-cuda310"
export ATTN_BACKEND=${ATTN_BACKEND:-sdpa}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec "$environment/bin/python" "$@"
