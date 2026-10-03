#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/vendor/llama.cpp/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-.venv/bin/python}" -m unittest discover -s tests -v
