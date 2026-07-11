#!/bin/bash
# DS4F-BASE end-to-end GENERATION test on REAL fp8-expert weights (12 A64FX EP nodes).
#
# Sibling of run_ds4f_gen_11n.sh:  prompt text --(BPE)--> ids --(greedy argmax)--> ids --> text.
#
# Two deliberate differences from the ds4f wrapper:
#
#  1. It does NOT set DS4F_FP8_BF16 / DS4F_Q8_DENSE. Both of those first PROMOTE the dense
#     tensors to bf16 (+~6 GB), which does not fit alongside 22.17 GiB of FP8 experts at EP=12.
#     Dense stays FP8 and the TP stack carries the fit — run_ds4fbase_12n.sh owns those defaults.
#
#  2. Base is a BASE model: tokenizer_config has no chat template and add_bos_token=false, and
#     there is no generation_config.json (eos = 1, from config.json). So prompts here are raw
#     COMPLETION stubs, not chat turns. Judge the output as a continuation, not an answer.
#
# The tokenizer is byte-identical to Flash's (md5 3f75dbea81fe67dd8c07843bdf9ce36e for both), so
# tools/ds4f_tokenizer.py is reused unchanged; TOK just points at the base dir for tidiness.
#
# Usage (inside the live 12-node alloc, from a64fx/llm; weights staged by run_ds4fbase_stage_12n.sh):
#     ./run_ds4fbase_gen_12n.sh
#     PROMPT_FILE=my_prompt.txt MAX_NEW=200 ./run_ds4fbase_gen_12n.sh
set +e
cd "$(dirname "$0")" || exit 2

TOK=${TOK:-$HOME/models/ds4fbase/tokenizer.json}
MAX_NEW=${MAX_NEW:-128}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT_IDS=${PROMPT_IDS:-prompt_ids_base.txt}
GEN_OUT=${GEN_OUT:-gen_ids_base.txt}
LOG=${DS4F_GEN_LOG:-/tmp/ds4fbase_gen_run.log}
SENT=${DS4F_GEN_SENTINEL:-/tmp/ds4fbase_gen_sentinel.txt}

# ---- 1. a completion-style prompt (base model: no chat template) ----
if [ -z "$PROMPT_FILE" ]; then
    PROMPT_FILE=$(mktemp /tmp/ds4fbase_prompt.XXXXXX.txt)
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

# Greedy is implicit in gen-mode (argmax feedback). All memory/TP defaults come from
# run_ds4fbase_12n.sh — do NOT add FP8_BF16/Q8_DENSE here (see the header).
t0=$(date +%s)
DS4F_CTX_WARM=0 DS4F_PREFILL_BATCH=0 \
DS4F_PREFILL=$NPROMPT DS4F_MAXGEN=$MAX_NEW \
  ./run_ds4fbase_12n.sh > "$LOG" 2>&1
rc=$?
t1=$(date +%s)

# ---- 4. detokenize + report ----
{
  echo "===== DS4F-BASE GEN SENTINEL ====="
  echo "GEN_RC rc=$rc wall=$((t1-t0))s prompt_toks=$NPROMPT max_new=$MAX_NEW"
  echo "--- rank0 summary (prefill/decode tok/s) ---"
  grep -hE 'prefill:|decode:|gen:|NaNs' ds4f_ep_rank00.txt 2>/dev/null | head -8
  echo "--- per-rank RSS (expect ~24.5 GiB ranks 0-3 / ~23.5 GiB ranks 4-11) ---"
  grep -hoE 'RSS=[0-9.]+ GB' ds4f_ep_load_rank*.txt 2>/dev/null | sort | uniq -c
  # Lockstep = all ranks agree WITHIN a phase. Compare the phases SEPARATELY: prefill and decode
  # legitimately produce different argmaxes, so pooling them reports 2 distinct values on a
  # perfectly healthy run (the ds4f wrapper does exactly that -- don't copy the bug back).
  # (count LINES of unique values, not words -- "last argmax=N" is two words.)
  pf=$(grep -h '^prefill:'    ds4f_ep_perf_rank*.txt 2>/dev/null | grep -oE 'argmax=[0-9]+$'      | cut -d= -f2 | sort -u)
  dc=$(grep -h '^last argmax' ds4f_ep_perf_rank*.txt 2>/dev/null | grep -oE '^last argmax=[0-9]+' | cut -d= -f2 | sort -u)
  echo "lockstep prefill: $(printf '%s\n' "$pf" | grep -c .) distinct across ranks [$(echo $pf)]  (1 == ok)"
  echo "lockstep decode : $(printf '%s\n' "$dc" | grep -c .) distinct across ranks [$(echo $dc)]  (1 == ok)"
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
    grep -m6 -iE 'segmentation|sigsegv|abort|MISSING tensor|dtype|nbytes|Killed|out of memory|No such file|cannot open' "$LOG"
  fi
  echo "GEN_END $(date +%H:%M:%S)"
} | tee "$SENT"

exit $rc
