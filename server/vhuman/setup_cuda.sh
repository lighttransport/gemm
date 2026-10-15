#!/bin/sh
# CPU-only installation; GPU validation is a separate, explicit command.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
exec python3 -B -m server.vhuman.dev_setup "$@"
