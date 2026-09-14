#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
backend=cuda
if [ "${1:-}" = "--backend" ]; then backend=${2:?missing backend}; shift 2; fi
exec "$root/ref/pixal3d/run.sh" "$backend" "$root/server/pixal3d/app.py" "$@"
