#!/bin/sh
# Optional face-video observer and independent PyTorch compression reference.
# All downloads are confined to ignored repository tmp/.
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
models="$root/tmp/vhuman-rig/models"
reference="$root/tmp/vhuman-rig/compskin"
model="$models/face_landmarker.task"
url=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
hash=64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
reference_url=https://github.com/facebookresearch/compskin.git
reference_rev=22e6f5fa19e533d84916bb1abd9dc88c4ac3054e
with_reference=0
install_deps=0

for arg in "$@"; do
    case "$arg" in
        --with-reference) with_reference=1 ;;
        --install-deps) install_deps=1 ;;
        *) printf 'unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

mkdir -p "$models"
if [ ! -f "$model" ] || [ "$(sha256sum "$model" | cut -d ' ' -f 1)" != "$hash" ]; then
    staged="$model.download.$$"
    trap 'rm -f "$staged"' EXIT
    curl --fail --location --retry 3 --output "$staged" "$url"
    [ "$(sha256sum "$staged" | cut -d ' ' -f 1)" = "$hash" ] || {
        printf 'MediaPipe model checksum mismatch\n' >&2; exit 1;
    }
    mv "$staged" "$model"
    trap - EXIT
fi
printf 'MediaPipe Face Landmarker %s\n' "$model"

if [ "$with_reference" -eq 1 ]; then
    if [ ! -d "$reference/.git" ]; then
        staged="$reference.clone.$$"
        trap 'rm -rf "$staged"' EXIT
        git clone --quiet "$reference_url" "$staged"
        git -C "$staged" checkout --quiet --detach "$reference_rev"
        mv "$staged" "$reference"
        trap - EXIT
    fi
    [ "$(git -C "$reference" rev-parse HEAD)" = "$reference_rev" ] || {
        printf 'Compressed Skinning checkout has a different commit\n' >&2; exit 1;
    }
    printf 'Compressed Skinning %s\n' "$reference_rev"
fi

if [ "$install_deps" -eq 1 ]; then
    python=${VHUMAN_PYTHON:-$root/tmp/vhuman-rig-venv/bin/python}
    export TMPDIR="$root/tmp/vhuman-runtime"
    export UV_CACHE_DIR="$root/tmp/uv-cache"
    mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
    [ -x "$python" ] || { printf 'create the vhuman rig venv first\n' >&2; exit 1; }
    uv pip install --python "$python" 'mediapipe==0.10.32' 'remotezip==0.12.6'
    if [ "$with_reference" -eq 1 ]; then
        uv pip install --python "$python" 'libigl==2.6.3'
    fi
fi
