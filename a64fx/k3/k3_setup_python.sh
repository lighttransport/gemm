#!/bin/bash
# Create/install the architecture-local K3 Python environment with uv.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ARCH=$(uname -m)
VENV="$SCRIPT_DIR/.venv-$ARCH"
REQ="$SCRIPT_DIR/requirements-python.txt"
UV_VERSION=${K3_UV_VERSION:-0.9.7}

usable_uv() {
    local candidate=$1
    [[ -x "$candidate" ]] || return 1
    "$candidate" --version >/dev/null 2>&1
}

UV=${K3_UV:-}
if [[ -z "$UV" ]]; then
    UV=$(command -v uv || true)
fi

if ! usable_uv "$UV"; then
    case "$ARCH" in
        aarch64|arm64) UV_ASSET=uv-aarch64-unknown-linux-gnu ;;
        x86_64|amd64) UV_ASSET=uv-x86_64-unknown-linux-gnu ;;
        *) echo "k3_setup_python: unsupported architecture: $ARCH" >&2; exit 2 ;;
    esac
    UV_CACHE=${K3_UV_CACHE:-$HOME/.cache/k3/uv/$UV_VERSION/$ARCH}
    UV="$UV_CACHE/uv"
    if ! usable_uv "$UV"; then
        mkdir -p "$UV_CACHE"
        tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/k3-uv.XXXXXX")
        archive="$tmp_dir/uv.tar.gz"
        url="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_ASSET.tar.gz"
        cleanup() { rm -rf "$tmp_dir"; }
        trap cleanup EXIT
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "$url" -o "$archive"
        elif command -v wget >/dev/null 2>&1; then
            wget -q "$url" -O "$archive"
        else
            echo "k3_setup_python: curl or wget is required to bootstrap uv" >&2
            exit 2
        fi
        tar -xzf "$archive" -C "$tmp_dir"
        cp "$tmp_dir/$UV_ASSET/uv" "$UV"
        chmod 755 "$UV"
    fi
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    "$UV" venv --python 3.11 "$VENV"
fi
"$UV" pip install --python "$VENV/bin/python" -r "$REQ"
printf 'K3_PYTHON_READY arch=%s python=%s uv=%s\n' "$ARCH" "$VENV/bin/python" "$UV"
