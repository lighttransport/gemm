#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Heldout corpus: $PWD/artifacts/corpus"
echo "Report: $PWD/artifacts/quality-local.json"
exec .venv/bin/python -m glm_reap.cli quality-check \
    --device cuda --corpus "$PWD/artifacts/corpus" \
    --report "$PWD/artifacts/quality-local.json" "$@"
