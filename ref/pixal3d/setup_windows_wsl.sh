#!/usr/bin/env bash
# Build the complete native host and sm_86 CUDA plugin under WSL2 without sudo.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
WORK="$ROOT/tmp/pixal3d"
CUDA_ROOT="$ROOT/.cuda-wsl/13.3"
DOWNLOADS="$WORK/cuda-wsl/downloads"
UV_CACHE_DIR="$ROOT/tmp/uv-cache-wsl"
VENV="$ROOT/.venv-pixal3d-wsl"
JOBS=${JOBS:-8}

grep -qi microsoft /proc/version || { echo "Run this script inside WSL2" >&2; exit 1; }
command -v curl >/dev/null
command -v cmake >/dev/null
command -v g++ >/dev/null
command -v make >/dev/null
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
mkdir -p "$WORK" "$DOWNLOADS" "$UV_CACHE_DIR"
rm -rf "$CUDA_ROOT"
mkdir -p "$CUDA_ROOT"

# OpenCV plus its TBB runtime are staged by the shared no-sudo helper.
sh "$ROOT/ref/pixal3d/setup_native.sh"

stage_debs() {
    local package_dir=$1 destination=$2
    shift 2
    mkdir -p "$package_dir" "$destination"
    (cd "$package_dir" && apt-get download "$@")
    local package
    for package in "$package_dir"/*.deb; do
        dpkg-deb -x "$package" "$destination"
    done
}

stage_debs "$WORK/boost-packages" "$WORK/boost" \
    libboost1.83-dev libboost-container1.83-dev libboost-json1.83-dev \
    libboost-json1.83.0 libboost-system1.83-dev
stage_debs "$WORK/openblas-packages" "$WORK/openblas" \
    libopenblas-dev libopenblas-pthread-dev libopenblas0-pthread

UV_ARCHIVE="$WORK/uv-0.9.21.tar.gz"
UV_SHA256=0a1ab27383c28ef1c041f85cbbc609d8e3752dfb4b238d2ad97b208a52232baf
if [ ! -f "$UV_ARCHIVE" ]; then
    curl -L --fail --retry 3 -o "$UV_ARCHIVE" \
        https://github.com/astral-sh/uv/releases/download/0.9.21/uv-x86_64-unknown-linux-gnu.tar.gz
fi
printf '%s  %s\n' "$UV_SHA256" "$UV_ARCHIVE" | sha256sum -c -
rm -rf "$WORK/uv-linux"
mkdir -p "$WORK/uv-linux" "$WORK/bin"
tar -xzf "$UV_ARCHIVE" -C "$WORK/uv-linux"
UV=$(find "$WORK/uv-linux" -type f -name uv -print -quit)
test -n "$UV"
cp "$UV" "$WORK/bin/uv"
chmod +x "$WORK/bin/uv"
if [ ! -x "$VENV/bin/python" ]; then
    UV_CACHE_DIR="$UV_CACHE_DIR" "$WORK/bin/uv" venv --python /usr/bin/python3 "$VENV"
fi
UV_CACHE_DIR="$UV_CACHE_DIR" "$WORK/bin/uv" pip install --python "$VENV/bin/python" \
    'ninja==1.13.0' 'numpy==2.2.6' 'pillow==11.3.0' 'psutil==7.0.0' 'scipy==1.16.2'

BASE=https://developer.download.nvidia.com/compute/cuda/redist
while read -r component relative_path sha256; do
    archive="$DOWNLOADS/$(basename "$relative_path")"
    extract="$DOWNLOADS/extract-$component"
    if [ ! -f "$archive" ]; then
        curl -L --fail --retry 3 -o "$archive" "$BASE/$relative_path"
    fi
    printf '%s  %s\n' "$sha256" "$archive" | sha256sum -c -
    rm -rf "$extract"
    mkdir -p "$extract"
    tar -xJf "$archive" -C "$extract"
    payload=$(find "$extract" -mindepth 1 -maxdepth 1 -type d -print -quit)
    test -n "$payload"
    cp -a "$payload/." "$CUDA_ROOT/"
    rm -rf "$extract"
done <<'PACKAGES'
cuda_nvcc cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.3.33-archive.tar.xz 93b098bda4a562ebf3541523ce82adc43f106a81dcf28bcbf8f0d8e093d1c66f
cuda_cudart cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.3.29-archive.tar.xz 1e59c4888267d27ba1a9bd0f3669a6439db1334a96e754cd9013c7c73e18dc9d
cuda_crt cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.3.33-archive.tar.xz 4755d36d24c6ef7697a2d3e1dbb23c4562c9c0d97d48390d4cbd8ab32dec5b5f
cccl cccl/linux-x86_64/cccl-linux-x86_64-13.3.3.3.1-archive.tar.xz 67746da12f16229ac4ebde78ce7895e42b069d1d3e2ae2d2d25f90bc43679d68
libnvvm libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.3.33-archive.tar.xz fc9c1fd5844e44c0e5eeb051378c1b13cf0e3bb3fe4966d5103c38885424f802
libcublas libcublas/linux-x86_64/libcublas-linux-x86_64-13.5.1.27-archive.tar.xz 35a898360520d6101ffcaf36c0d04496b54d4fc2afb82f7fce44218c54513808
PACKAGES

# The redistributable cudart archive contains the shared runtime. nvcc's
# compiler probe additionally needs the static runtime and device runtime from
# NVIDIA's matching Ubuntu development package.
CUDART_DEV=cuda-cudart-dev-13-3_13.3.29-1_amd64.deb
CUDART_DEV_SHA256=600e5cf3685d0afae85970ba02451358068b7b56c954999b9149900ec5d940d9
CUDART_DEV_ARCHIVE="$DOWNLOADS/$CUDART_DEV"
if [ ! -f "$CUDART_DEV_ARCHIVE" ]; then
    curl -L --fail --retry 3 -o "$CUDART_DEV_ARCHIVE" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/$CUDART_DEV"
fi
printf '%s  %s\n' "$CUDART_DEV_SHA256" "$CUDART_DEV_ARCHIVE" | sha256sum -c -
rm -rf "$DOWNLOADS/cudart-dev"
mkdir -p "$DOWNLOADS/cudart-dev"
dpkg-deb -x "$CUDART_DEV_ARCHIVE" "$DOWNLOADS/cudart-dev"
ln -s lib "$CUDA_ROOT/lib64"
cp "$DOWNLOADS/cudart-dev/usr/local/cuda-13.3/targets/x86_64-linux/lib/libcudadevrt.a" \
    "$DOWNLOADS/cudart-dev/usr/local/cuda-13.3/targets/x86_64-linux/lib/libcudart_static.a" \
    "$CUDA_ROOT/lib64/"
rm -rf "$DOWNLOADS/cudart-dev"
test -f "$CUDA_ROOT/include/crt/host_config.h"
test -f "$CUDA_ROOT/lib64/libcudadevrt.a"
test -f "$CUDA_ROOT/lib64/libcudart_static.a"

"$CUDA_ROOT/bin/nvcc" --version
cmake -S "$ROOT/cuda/pixal3d" -B "$ROOT/cuda/pixal3d/build-wsl" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$VENV/bin/ninja" \
    -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DPython3_EXECUTABLE="$VENV/bin/python" \
    -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "$ROOT/cuda/pixal3d/build-wsl" --parallel "$JOBS"
cp "$ROOT/cuda/pixal3d/build-wsl/libpixal3d_cuda.so" "$ROOT/cuda/pixal3d/libpixal3d_cuda.so"

# shellcheck source=windows_wsl_env.sh
source "$ROOT/ref/pixal3d/windows_wsl_env.sh"
make -C "$ROOT/cpu/pixal3d" -j"$JOBS"
make -C "$ROOT/cpu/pixal3d" test
echo "Pixal3D WSL2 build ready: $ROOT/cpu/pixal3d/pixal3d"
