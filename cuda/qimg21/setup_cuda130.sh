#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${PYTORCH_CUDA_HOME:-$ROOT/tmp/cuda130}"
CACHE="${QIMG21_CUDA_CACHE:-$ROOT/tmp/cuda130-download}"
BASE=https://developer.download.nvidia.com/compute/cuda/redist
OFFLINE=0

usage() {
    echo "usage: $0 [--offline] [--dest DIR] [--cache-dir DIR]" >&2
}

while (($#)); do
    case "$1" in
        --offline) OFFLINE=1; shift ;;
        --dest) [[ $# -ge 2 ]] || { usage; exit 2; }; DEST="$2"; shift 2 ;;
        --cache-dir) [[ $# -ge 2 ]] || { usage; exit 2; }; CACHE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

if [[ -x "$DEST/bin/nvcc" ]] &&
   "$DEST/bin/nvcc" --version | grep -F 'release 13.0, V13.0.88' >/dev/null; then
    echo "pinned CUDA compiler already installed at $DEST"
    exit 0
fi
if [[ -e "$DEST" ]]; then
    echo "setup_cuda130: destination exists but is not CUDA 13.0.88: $DEST" >&2
    exit 1
fi

components=(
    "cuda_cccl cuda_cccl-linux-x86_64-13.0.85-archive.tar.xz ed845eae8c1767706b6ee91e40c608a03f6f633551a849b63f7346d32d73ee60"
    "cuda_crt cuda_crt-linux-x86_64-13.0.88-archive.tar.xz 5a3279a049ffc1cdb951c44cb95206acfdde9e9ae5e87825fc18d7e4a6878bb0"
    "cuda_cudart cuda_cudart-linux-x86_64-13.0.88-archive.tar.xz bc44226f069402ab327bcf9e754660621aa6bc61fb7fc6afe03b794dfaaab658"
    "cuda_nvcc cuda_nvcc-linux-x86_64-13.0.88-archive.tar.xz 48e35be3cfbf4b4fbc16828eaec8a7048ee789403049dc409f7b643d6259cf7a"
    "libnvvm libnvvm-linux-x86_64-13.0.88-archive.tar.xz 17ef1665b63670887eeba7d908da5669fa8c66bb73b5b4c1367f49929c086353"
)

mkdir -p "$CACHE" "$ROOT/tmp"
stage="$(mktemp -d "$ROOT/tmp/qimg21-cuda130.XXXXXX")"
trap 'rm -rf "$stage"' EXIT

for entry in "${components[@]}"; do
    read -r component archive checksum <<<"$entry"
    cached="$CACHE/${component}.tar.xz"
    if [[ ! -f "$cached" ]]; then
        if ((OFFLINE)); then
            echo "setup_cuda130: offline cache miss: $cached" >&2
            exit 1
        fi
        partial="$cached.part"
        rm -f "$partial"
        curl -fL "$BASE/$component/linux-x86_64/$archive" -o "$partial"
        mv "$partial" "$cached"
    fi
    echo "$checksum  $cached" | sha256sum --check --status || {
        echo "setup_cuda130: checksum mismatch: $cached" >&2
        exit 1
    }
    unpack="$stage/$component"
    mkdir -p "$unpack"
    tar -xJf "$cached" -C "$unpack" --strip-components=1
done

new="$stage/toolkit"
mkdir -p "$new"
for entry in "${components[@]}"; do
    read -r component _ <<<"$entry"
    cp -a "$stage/$component/." "$new/"
done

mv "$new" "$DEST"
"$DEST/bin/nvcc" --version | grep -F 'release 13.0, V13.0.88' >/dev/null
echo "installed pinned CUDA compiler at $DEST"
