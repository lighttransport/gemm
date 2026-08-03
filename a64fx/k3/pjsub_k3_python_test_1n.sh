#!/bin/bash
# One-node batch smoke test for the shared K3 Python environment.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=1,elapse=00:10:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=1"
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
MODEL_DIR=${K3_MODEL_DIR:-$HOME/models/kimi-k3}
ARCH=$(uname -m)
PYTHON="$K3/.venv-$ARCH/bin/python"
TEST_DIR="$K3/logs/python-test-1n-${PJM_JOBID:-manual-$$}"

mkdir -p "$TEST_DIR"
"$K3/k3_setup_python.sh" | tee "$TEST_DIR/setup.txt"
[[ -x "$PYTHON" ]] || { echo "missing K3 Python: $PYTHON" >&2; exit 2; }
printf 'python=%s\n' "$PYTHON" >"$TEST_DIR/environment.txt"
"$PYTHON" --version | tee -a "$TEST_DIR/environment.txt"
"$PYTHON" -c 'import regex, tiktoken; print("K3_PYTHON_IMPORTS PASS tiktoken=%s regex=%s" % (tiktoken.__version__, regex.__version__))' \
    | tee "$TEST_DIR/imports.txt"

cat >"$TEST_DIR/prompt.txt" <<'EOF'
Implement a bounded concurrent queue in C++20 and explain its memory ordering.
EOF
"$K3/k3_python.sh" "$K3/make_k3_prompt_ids.py" \
    --vocab "$MODEL_DIR/tiktoken.model" --text "$TEST_DIR/prompt.txt" \
    --output "$TEST_DIR/prompt.ids" --tokens 32 --bos --repeat-to \
    | tee "$TEST_DIR/tokenizer.txt"

"$PYTHON" "$K3/k3_sim.py" --help >/dev/null
printf 'K3_PYTHON_TEST_1N status=PASS results=%s\n' "$TEST_DIR"
