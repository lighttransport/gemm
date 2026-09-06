#!/bin/bash
# Generate and compile/run a C++17 completion using the real 0731 weights.
set -euo pipefail

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LLM_DIR"
RESULT_DIR=${RESULT_DIR:-$LLM_DIR/runs/ds4f-0731-cpp-${PJM_JOBID:-manual-$$}}
PROMPT_FILE=${PROMPT_FILE:-$LLM_DIR/ds4f_cpp_quality_prompt.txt}
RESULT_DIR=$(realpath -m "$RESULT_DIR")
MAX_NEW=${MAX_NEW:-512}

RESULT_DIR="$RESULT_DIR" PROMPT_FILE="$PROMPT_FILE" MAX_NEW="$MAX_NEW" \
    "$LLM_DIR/run_ds4f_0731_gen_12n.sh"
python3 "$LLM_DIR/validate_ds4f_cpp.py" \
    "$RESULT_DIR/completion.cpp.txt" --out-dir "$RESULT_DIR" \
    --prefix-file "$LLM_DIR/ds4f_cpp_quality_prefix.txt" | tee "$RESULT_DIR/cpp_quality.txt"
echo "DS4F_0731_CPP_QUALITY_PASS result=$RESULT_DIR"
