#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/glm53f.json}"
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
work="$(.venv/bin/python -c 'import sys; from glm_reap.common import load_config; c=load_config(sys.argv[1]); print(c.get("work_dir",c["output"]))' "$config")"
mkdir -p "$work"
.venv/bin/python -u -c 'import sys; from glm_reap.run import run; run(sys.argv[1])' "$config" 2>&1 | tee -a "$work/run.log"
