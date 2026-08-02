#!/bin/bash
# End-to-end 0731 generation wrapper.  It encodes a prompt with the matching
# tokenizer, runs greedy real-weight generation on 12 nodes, and detokenizes
# the rank-0 token stream into the result directory.
set +e

LLM_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$LLM_DIR" || exit 2

TOK=${TOK:-$HOME/models/ds4f-0731/tokenizer.json}
PROMPT_FILE=${PROMPT_FILE:-$LLM_DIR/ds4f_cpp_quality_prompt.txt}
MAX_NEW=${MAX_NEW:-192}
RESULT_DIR=${RESULT_DIR:-$LLM_DIR/runs/ds4f-0731-gen-${PJM_JOBID:-manual-$$}}
STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f-0731-${PJM_JOBID:-manual}}
MODEL_DIR=${DS4F_MODEL_DIR:-$HOME/models/ds4f-0731}
PROMPT_IDS="$RESULT_DIR/prompt_ids.txt"
GEN_OUT="$RESULT_DIR/gen_ids.txt"
COMPLETION="$RESULT_DIR/completion.cpp.txt"
SENTINEL="$RESULT_DIR/generation.sentinel.txt"

[ -s "$TOK" ] || { echo "missing tokenizer: $TOK" >&2; exit 3; }
[ -s "$PROMPT_FILE" ] || { echo "missing prompt: $PROMPT_FILE" >&2; exit 3; }
mkdir -p "$RESULT_DIR"
RESULT_DIR=$(realpath "$RESULT_DIR")
# The runner changes directory into RESULT_DIR before launching ranks, so all
# files passed through DS4F_* must be absolute. Keeping these paths derived
# after realpath also makes RESULT_DIR=relative/... safe.
PROMPT_IDS="$RESULT_DIR/prompt_ids.txt"
GEN_OUT="$RESULT_DIR/gen_ids.txt"
COMPLETION="$RESULT_DIR/completion.cpp.txt"
SENTINEL="$RESULT_DIR/generation.sentinel.txt"

python3 tools/ds4f_tokenizer.py encode --tokenizer "$TOK" \
    --prompt-file "$PROMPT_FILE" --no-bos --out "$PROMPT_IDS" || exit 4
NPROMPT=$(wc -w < "$PROMPT_IDS")
[ "$NPROMPT" -gt 0 ] || { echo "prompt encoded to zero tokens" >&2; exit 4; }

export DS4F_PROMPT_IDS="$PROMPT_IDS"
export DS4F_GEN_OUT="$GEN_OUT"
export DS4F_MAX_NEW="$MAX_NEW"
export DS4F_PREFILL="$NPROMPT"
export DS4F_MAXGEN="$MAX_NEW"
export DS4F_MAXPOS=$((NPROMPT + MAX_NEW + 64))
# Keep the real generation path at the largest validated verify tile. The
# dense GEMMs and one [K,C] EP reduce amortize much better at 128; callers can
# lower this for memory-constrained or exact A/B runs.
export DS4F_PREFILL_VERIFY=${DS4F_PREFILL_VERIFY:-128}
export SKIP_TOPO=${SKIP_TOPO:-0}
export DS4F_STAGE_DIR="$STAGE_DIR"
export DS4F_RUN_TAG=$(basename "$RESULT_DIR")

echo "=== DS4F-0731 generation ==="
echo "prompt=$PROMPT_FILE tokens=$NPROMPT max_new=$MAX_NEW"
echo "weights=$MODEL_DIR stage=$STAGE_DIR result=$RESULT_DIR"
t0=$(date +%s)
RESULT_DIR="$RESULT_DIR" ./run_ds4f_0731_12n.sh > "$RESULT_DIR/runner.stdout.txt" 2>&1
rc=$?
t1=$(date +%s)

if [ -s "$GEN_OUT" ]; then
    python3 tools/ds4f_tokenizer.py decode --tokenizer "$TOK" --ids-file "$GEN_OUT" > "$COMPLETION"
else
    : > "$COMPLETION"
fi

{
    echo "DS4F_0731_GEN rc=$rc wall=$((t1-t0))s prompt_tokens=$NPROMPT max_new=$MAX_NEW"
    grep -hE '^prefill:|^decode:|^last argmax|NaNs|^gen:' \
        "$RESULT_DIR/ds4f_ep_rank00.txt" "$RESULT_DIR"/ds4f_ep_perf_rank*.txt 2>/dev/null | head -30
    pf=$(grep -h '^prefill:' "$RESULT_DIR"/ds4f_ep_perf_rank*.txt 2>/dev/null | grep -oE 'argmax=[0-9]+$' | cut -d= -f2 | sort -u)
    dc=$(grep -h '^last argmax=' "$RESULT_DIR"/ds4f_ep_perf_rank*.txt 2>/dev/null | grep -oE '^last argmax=[0-9]+' | cut -d= -f2 | sort -u)
    echo "prefill_argmax_distinct=$(printf '%s\n' "$pf" | grep -c .) values=[$(echo "$pf" | tr '\n' ' ')]"
    echo "decode_argmax_distinct=$(printf '%s\n' "$dc" | grep -c .) values=[$(echo "$dc" | tr '\n' ' ')]"
    echo "generated_tokens=$(wc -w < "$GEN_OUT" 2>/dev/null || echo 0)"
    echo "completion=$COMPLETION"
} | tee "$SENTINEL"

exit "$rc"
