#!/bin/bash
# Best-practice Laguna S-2.1 FP8 entry point.
#
# The best measured production shape is one EP rank per A64FX node in the
# 12-node small resource group at 2 GHz, with eco/retention disabled.  This
# wrapper keeps that policy in one name while preserving the generic launcher's
# flags and llmgr's MPIEXEC_OF_PROC integration.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-generate}"
shift || true
exec "$HERE/run_laguna_s21_12n.sh" "$MODE" --fp8 "$@"
