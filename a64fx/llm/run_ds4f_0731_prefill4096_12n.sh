#!/bin/bash
# Fixed boost-eco KPI: whole-prompt 4096-token real prefill + 32-token cache check.
set -euo pipefail

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$LLM_DIR/../.." && pwd)"
cd "$LLM_DIR"

[ "${PJM_MPI_PROC:-0}" = 12 ] || { echo "requires 12 MPI ranks" >&2; exit 2; }
[ "${PJM_PK_freq:-0}" = 2200 ] || { echo "requires freq=2200" >&2; exit 2; }
[ "${PJM_PK_eco_state:-x}" = 2 ] || { echo "requires eco_state=2" >&2; exit 2; }

RESULT_DIR=${RESULT_DIR:-$LLM_DIR/runs/boosteco-prefill4096-${PJM_JOBID:-manual}}
STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f-0731-${PJM_JOBID:-manual}}
TOK=${TOK:-$HOME/models/ds4f-0731/tokenizer.json}
mkdir -p "$RESULT_DIR"
RESULT_DIR=$(realpath "$RESULT_DIR")

IDS="$RESULT_DIR/prompt_4096_ids.txt"
PF_OUT="$RESULT_DIR/prefill_argmax_ids.txt"
GEN_OUT="$RESULT_DIR/gen_ids.txt"
python3 make_ds4f_prefill_corpus.py --tokenizer "$TOK" \
    --source "$ROOT_DIR/a64fx/ds4f.md" --out "$IDS" --tokens 4096

export DS4F_PROMPT_IDS="$IDS"
export DS4F_PREFILL_OUT="$PF_OUT"
export DS4F_GEN_OUT="$GEN_OUT"
export DS4F_MAX_NEW=${DS4F_MAX_NEW:-32}
export DS4F_PREFILL_VERIFY=${DS4F_PREFILL_VERIFY:-128}
export DS4F_MAXPOS=${DS4F_MAXPOS:-4224}
export DS4F_STAGE_DIR="$STAGE_DIR"
export DS4F_PROFILE=${DS4F_PROFILE:-exact}
export DS4F_TP_SHARED=${DS4F_TP_SHARED:-0}
export DS4F_TP_SHARED_FULL=${DS4F_TP_SHARED_FULL:-0}
export DS4F_COMM_MODE=${DS4F_COMM_MODE:-flat}
export TP_AR_BF16=${TP_AR_BF16:-0}
export TP_AR_A2A=${TP_AR_A2A:-0}
export TP_AR_ROBUST=${TP_AR_ROBUST:-1}

RESULT_DIR="$RESULT_DIR" ./run_ds4f_0731_12n.sh | tee "$RESULT_DIR/runner.stdout.txt"
[ "$(wc -w < "$PF_OUT")" -eq 4096 ]
[ "$(wc -w < "$GEN_OUT")" -eq "$DS4F_MAX_NEW" ]
sha256sum "$IDS" "$PF_OUT" "$GEN_OUT" | tee "$RESULT_DIR/token_sha256.txt"
echo "DS4F_0731_PREFILL4096_PASS result=$RESULT_DIR"
