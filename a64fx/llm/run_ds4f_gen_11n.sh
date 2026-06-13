#!/bin/bash
# DS4F end-to-end GENERATION-quality test on REAL weights (11 A64FX EP nodes).
#
# Pipeline:  prompt text --(pure-python BPE)--> prompt_ids.txt
#            --(ds4f_ep_runner gen-mode, greedy argmax feedback)--> gen_ids.txt
#            --(detokenize)--> completion text.
#
# Gen-mode is driven entirely by env the runner reads (mpiexec forwards EXPORTED
# env only, so we export them here and let run_ds4f_11n.sh do the launch):
#     DS4F_PROMPT_IDS=<file>   one whitespace-separated prompt token-id stream
#     DS4F_GEN_OUT=<file>      rank0 writes the generated id stream here
#     DS4F_MAX_NEW=<n>         greedy decode budget (stops early on eos=1)
# Real-weight decode path = the production sparse path: REAL + bf16-pv + Tier-B2
# (DS4F_TIERB2 implies EXACT: q-norm/RoPE/YaRN, window+sink, compressor/indexer).
# All ranks compute identical argmax (dense+head replicated) => identical token
# feedback => cross-rank lockstep with no broadcast.
#
# Usage (inside the live 12-node alloc, from a64fx/llm; runs the 11 non-(0,0,0) EP nodes):
#     ./run_ds4f_gen_11n.sh                                  # built-in fibonacci prompt
#     PROMPT_FILE=my_prompt.txt ./run_ds4f_gen_11n.sh        # your own coding prompt
#     MAX_NEW=200 ./run_ds4f_gen_11n.sh                      # longer completion
#
# Prereq: weights staged to /local/ds4f (run_ds4f_stage_11n.sh) — same node set.
set +e
cd "$(dirname "$0")" || exit 2

TOK=${TOK:-$HOME/models/ds4f/tokenizer.json}
MAX_NEW=${MAX_NEW:-128}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT_IDS=${PROMPT_IDS:-prompt_ids.txt}
GEN_OUT=${GEN_OUT:-gen_ids.txt}
LOG=${DS4F_GEN_LOG:-/tmp/ds4f_gen_run.log}
SENT=${DS4F_GEN_SENTINEL:-/tmp/ds4f_gen_sentinel.txt}

# ---- 1. the coding prompt (default: a classic code-completion stub) ----
if [ -z "$PROMPT_FILE" ]; then
    PROMPT_FILE=$(mktemp /tmp/ds4f_prompt.XXXXXX.txt)
    cat > "$PROMPT_FILE" <<'EOF'
def quicksort(arr):
    """Sort a list of numbers in ascending order using the quicksort algorithm."""
EOF
    echo "[gen] using built-in code-completion prompt:"; sed 's/^/    /' "$PROMPT_FILE"
fi

# ---- 2. encode (BOS prepended) ----
python3 tools/ds4f_tokenizer.py encode --tokenizer "$TOK" \
        --prompt-file "$PROMPT_FILE" --out "$PROMPT_IDS" || { echo "[gen] encode FAILED" >&2; exit 3; }
NPROMPT=$(wc -w < "$PROMPT_IDS")
echo "[gen] prompt encoded -> $PROMPT_IDS  ($NPROMPT tokens, max_new=$MAX_NEW)"

# ---- 3. size max_pos for prompt+gen, point the runner at the id files ----
export DS4F_PROMPT_IDS="$PWD/$PROMPT_IDS"
export DS4F_GEN_OUT="$PWD/$GEN_OUT"
export DS4F_MAX_NEW="$MAX_NEW"
export DS4F_MAXPOS=$(( NPROMPT + MAX_NEW + 64 ))
rm -f "$GEN_OUT"

# Real production decode path. Greedy is implicit in gen-mode (argmax feedback).
t0=$(date +%s)
DS4F_REAL=${DS4F_REAL:-1} \
DS4F_FP8_BF16=${DS4F_FP8_BF16:-1} \
DS4F_TIERB2=${DS4F_TIERB2:-1} \
DS4F_MHC=${DS4F_MHC:-1} \
DS4F_HC_PAR=${DS4F_HC_PAR:-1} \
DS4F_HC_RMSPAR=${DS4F_HC_RMSPAR:-1} \
DS4F_Q8_DENSE=${DS4F_Q8_DENSE:-1} \
DS4F_CTX_WARM=0 DS4F_PREFILL_BATCH=0 \
DS4F_PREFILL=$NPROMPT DS4F_MAXGEN=$MAX_NEW \
  ./run_ds4f_11n.sh > "$LOG" 2>&1
rc=$?
t1=$(date +%s)

# ---- 4. detokenize + report ----
{
  echo "===== DS4F GEN SENTINEL ====="
  echo "GEN_RC rc=$rc wall=$((t1-t0))s prompt_toks=$NPROMPT max_new=$MAX_NEW"
  echo "--- rank0 summary (prefill/decode tok/s) ---"
  grep -hE 'prefill:|decode:|gen:|NaNs' ds4f_ep_rank00.txt 2>/dev/null | head -8
  argv=$(grep -hoE 'argmax[ =]+[0-9]+' ds4f_ep_perf_rank*.txt 2>/dev/null | grep -oE '[0-9]+$' | sort -u | tr '\n' ' ')
  echo "argmax_distinct_across_ranks=$(echo $argv | wc -w)  values=[$argv]  (1==lockstep)"
  if [ -s "$GEN_OUT" ]; then
    NGEN=$(wc -w < "$GEN_OUT")
    echo "--- generated $NGEN ids -> detokenized completion ---"
    echo "<<<PROMPT>>>"
    cat "$PROMPT_FILE"
    echo "<<<COMPLETION>>>"
    python3 tools/ds4f_tokenizer.py decode --tokenizer "$TOK" --ids-file "$GEN_OUT"
    echo "<<<END>>>"
  else
    echo "!! no $GEN_OUT produced (rc=$rc). Last crash context:"
    grep -m6 -iE 'segmentation|sigsegv|abort|MISSING tensor|Killed|out of memory|No such file|cannot open' "$LOG"
  fi
  echo "GEN_END $(date +%H:%M:%S)"
} | tee "$SENT"

exit $rc
