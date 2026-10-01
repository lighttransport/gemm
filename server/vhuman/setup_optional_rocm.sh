#!/bin/sh
# Install the pinned optional depth, video, emotion and facial teacher modules.
# Usage: setup_optional_rocm.sh [--with-lightgeom]
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
with_teacher=0
for arg in "$@"; do
    case "$arg" in
        --with-lightgeom) with_teacher=1 ;;
        *) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
python="$root/tmp/vhuman-rocm-venv/bin/python"
[ -x "$python" ] || { echo 'Run server/vhuman/setup_rocm.sh first' >&2; exit 1; }
export TMPDIR="$root/tmp/vhuman-runtime"
export UV_CACHE_DIR="$root/tmp/uv-cache"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/tmp/vhuman-cache}"
export LD_LIBRARY_PATH="/opt/rocm/core/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME"
VHUMAN_PYTHON="$python" sh server/vhuman/rig/setup_face_video.sh --install-deps
"$python" -B -m server.vhuman.reconstruction.setup_depth
"$python" -B -m server.vhuman.fetch_emotion_assets
export ROCM_PATH=/opt/rocm/core
cmake -S rdna4/sensevoice -B tmp/vhuman-emotion/build-rocm -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++" -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build tmp/vhuman-emotion/build-rocm --target llama-funasr-sensevoice --parallel 4
mkdir -p tmp/vhuman-emotion/runtime
ln -sf ../build-rocm/bin/llama-funasr-sensevoice tmp/vhuman-emotion/runtime/llama-funasr-sensevoice
if [ "$with_teacher" -eq 1 ]; then
    sh server/vhuman/rig/setup_face_sources.sh --teacher-only
fi
"$python" -B -m server.vhuman.check_rocm_assets
