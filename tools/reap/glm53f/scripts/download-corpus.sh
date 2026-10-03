#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
echo "Corpus: $PWD/artifacts/corpus"
echo "Download cache: $HF_HOME"
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c 'import glm_reap, datasets, transformers, PIL' 2>/dev/null; then
    echo 'Python environment is incomplete. Run: bash scripts/setup-python.sh' >&2
    exit 1
fi
set -o pipefail
.venv/bin/python -u -m glm_reap.cli download-corpus \
    --destination "$PWD/artifacts/corpus" "$@" 2>&1 | tee -a "$PWD/artifacts/corpus-download.log"
