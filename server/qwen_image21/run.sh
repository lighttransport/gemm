#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
exec "${PYTHON:-$root/tmp/qimg21-ref-venv/bin/python}" "$root/server/qwen_image21/app.py" "$@"
