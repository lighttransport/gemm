#!/usr/bin/env bash
# Run the native generation fixture under WSL2 with repository-local libraries.
set -euo pipefail
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
# shellcheck source=windows_wsl_env.sh
source "$ROOT/ref/pixal3d/windows_wsl_env.sh"
cd "$ROOT"
exec "$ROOT/.venv-pixal3d-wsl/bin/python" "$ROOT/ref/pixal3d/run_fixture.py" "$@"
