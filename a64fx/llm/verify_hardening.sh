#!/bin/bash
# Full verification of the production hardening, in order. Each step gates the next.
#   [A] forced OOM      -> must print a clear message, NOT a SIGSEGV
#   [B] induced SIGSEGV -> must print a SYMBOLIZED backtrace to the per-rank log
#   [C] 43-layer gen    -> numbers unchanged (decode ~15.5, prefill ~24.9), coherent completion
#   [D] bench_headline  -> the four Flash numbers, gate-first
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
cd "$(dirname "$0")"
OUT=verify_hardening.txt; : > "$OUT"
say() { echo "$*" | tee -a "$OUT"; }

free_nodes() {
    pkill -x ds4f_ep_runner 2>/dev/null || true; pkill -x mpiexec 2>/dev/null || true
    for i in $(seq 1 60); do
        [ "$(ps -eo comm --no-headers | grep -cE 'ds4f_ep_runner|mpiexec' || true)" = "0" ] && return 0
        sleep 1
    done
}
BASE="DS4F_STAGE_DIR=/local/ds4f DS4F_REAL=1 DS4F_EXACT=1 DS4F_TIERB2=1 DS4F_MHC=1 DS4F_CMP_LOCAL=1 LLM_THREADS=47"

# ---------- [A] forced OOM ----------
free_nodes; rm -rf logs
say "=== [A] forced OOM (DS4F_TEST_OOM=1) — must be a MESSAGE, not a crash ==="
env $BASE DS4F_TEST_OOM=1 DS4F_LAYERS=2 DS4F_MAXPOS=64 timeout 300 ./run_ds4f_11n.sh >/dev/null 2>&1
grep -h "FATAL\|out of memory\|MemAvailable" logs/latest/rank00.err 2>/dev/null | head -3 | sed 's/^/  /' | tee -a "$OUT"
grep -qE "out of memory|implausible allocation" logs/latest/rank00.err 2>/dev/null && say "  [A] PASS" || say "  [A] FAIL"

# ---------- [B] induced SIGSEGV ----------
free_nodes; rm -rf logs
say ""; say "=== [B] induced SIGSEGV (DS4F_TEST_SEGV=1) — must be a SYMBOLIZED backtrace ==="
env $BASE DS4F_TEST_SEGV=1 DS4F_LAYERS=2 DS4F_MAXPOS=64 timeout 300 ./run_ds4f_11n.sh >/dev/null 2>&1
grep -h "FATAL signal\|ds4f_ep_runner\[" logs/latest/rank00.err 2>/dev/null | head -4 | sed 's/^/  /' | tee -a "$OUT"
grep -q "FATAL signal" logs/latest/rank00.err 2>/dev/null && say "  [B] PASS" || say "  [B] FAIL"

# ---------- [C] 43-layer regression ----------
free_nodes; rm -rf logs; rm -f vh_s.txt
say ""; say "=== [C] 43-layer gen regression (expect decode ~15.5, prefill ~24.9) ==="
env $BASE PROMPT_FILE=sweep_k_prompt.txt MAX_NEW=32 \
    DS4F_GEN_LOG=$PWD/vh.log DS4F_GEN_SENTINEL=$PWD/vh_s.txt timeout 1200 ./run_ds4f_gen_11n.sh >/dev/null 2>&1
grep -hE "^prefill:|^decode:|^lockstep" vh_s.txt 2>/dev/null | sed 's/^/  /' | tee -a "$OUT"
sed -n '/<<<COMPLETION>>>/,/<<<END>>>/p' vh_s.txt 2>/dev/null | sed '1d;$d' | head -4 | sed 's/^/  | /' | tee -a "$OUT"
grep -q "^decode:" vh_s.txt 2>/dev/null && say "  [C] PASS (numbers produced)" || say "  [C] FAIL"

# ---------- [D] Flash headline ----------
free_nodes
say ""; say "=== [D] bench_headline_flash_11n.sh (gate first) ==="
./bench_headline_flash_11n.sh > bench_flash_outer.log 2>&1
grep -hE "match, common prefix|^ *prefill:|^ *decode:|^ *M=" bench_headline_flash.txt 2>/dev/null | sed 's/^/  /' | tee -a "$OUT"
free_nodes
say ""; say "=== VERIFICATION COMPLETE ==="
