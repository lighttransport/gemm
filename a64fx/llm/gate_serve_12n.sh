#!/bin/bash
# GATE: does the BATCHED SERVE path return a COHERENT completion for a real prompt?
#
# This is the last path that was broken by the forward_verify x TP_WOB bug (fixed 2026-07-12,
# ds4f_impl.h) and the only one never re-tested end-to-end -- VERIFY_GATE / PREFILL_GEMM /
# DB_BENCH were. The serve loop shares ds4f_forward_verify, so it SHOULD be fixed; this proves it
# instead of inferring it.
#
# Uses the DYNAMIC loop on purpose: the static loop's reqseq counter file ("0\n" -> "1\n", same
# size) is never revalidated by the compute node's FS cache, so it does not see requests. The
# dynbatch loop probes <base>.q.<id> EXISTENCE, which does invalidate.
#
#   ./gate_serve_12n.sh          # 2 concurrent seqs, real prompts, prints decoded text
set -e
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:${PATH}"
LLM_DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$LLM_DIR"

TOK=${TOK:-$HOME/models/ds4fbase/tokenizer.json}
# Serve queue MUST live on the shared FS: rank 0 is on another node, /local is node-private.
Q=${Q:-$HOME/ds4f_serve_gate}
rm -rf "$Q"; mkdir -p "$Q"
BASE="$Q/s"
export DS4F_SERVE=1 DS4F_SERVE_BATCH=${SB:-2} DS4F_SERVE_DYNAMIC=1
export DS4F_SERVE_REQ="$BASE.req" DS4F_SERVE_RESP="$BASE.resp"
export DS4F_SERVE_REQSEQ="$BASE.reqseq" DS4F_SERVE_RESPSEQ="$BASE.respseq"
export DS4F_MAXPOS=${DS4F_MAXPOS:-1024}
MAX_NEW=${MAX_NEW:-48}

# ---- two real prompts -> ids ----
p1="The capital city of France is Paris, and the capital city of Japan is"
p2="def fibonacci(n):\n    if n < 2:\n        return n\n    return"
enc() { printf '%b' "$1" > "$Q/p.txt"; python3 tools/ds4f_tokenizer.py encode \
        --tokenizer "$TOK" --prompt-file "$Q/p.txt" --out "$Q/ids.txt" >/dev/null; cat "$Q/ids.txt"; }
IDS1=$(enc "$p1"); IDS2=$(enc "$p2")

# ---- bring the serve loop up in the background (it never returns) ----
DS4F_MAXGEN=$MAX_NEW ./run_ds4fbase_12n.sh > gate_serve.log 2>&1 &
SRV=$!
echo "[gate] serve loop pid $SRV -- waiting for it to reach the queue..."
for i in $(seq 1 200); do [ -f "$BASE.qhead" ] && break; sleep 5; done
[ -f "$BASE.qhead" ] || { echo "[gate] serve never came up"; tail -20 gate_serve.log; exit 1; }
sleep 10   # let it settle past load

# ---- submit both requests (atomic: write .t then rename, like the frontend does) ----
sub() { printf '%s\n%s\n' "$2" "$3" > "$BASE.q.$1.t"; mv "$BASE.q.$1.t" "$BASE.q.$1"; }
sub 0 "$MAX_NEW" "$IDS1"; sub 1 "$MAX_NEW" "$IDS2"
echo "[gate] submitted 2 requests, polling for responses..."

for i in $(seq 1 240); do
    [ -f "$BASE.r.0" ] && [ -f "$BASE.r.1" ] && break; sleep 5
done
# The serve loop NEVER returns, so it must be torn down explicitly -- and killing the wrapper is
# NOT enough: mpiexec/plexec/ds4f_ep_runner survive it and keep holding the nodes, so the NEXT job
# dies with "PLE 0054 plexec: number of processes exceed the limit on virtual coordinate (0,0,0)".
# Match on the exact process NAME: `pkill -f ds4f_ep_runner` also matches this script's own command
# line and kills the shell running it.
kill $SRV 2>/dev/null || true
pkill -x ds4f_ep_runner 2>/dev/null || true; sleep 2
pkill -x mpiexec 2>/dev/null || true; pkill -x plexec 2>/dev/null || true; sleep 2

fail=0
for n in 0 1; do
    if [ ! -f "$BASE.r.$n" ]; then echo "[gate] req $n: NO RESPONSE -> FAIL"; fail=1; continue; fi
    echo "$( [ $n = 0 ] && echo "$p1" || printf '%b' "$p2" )" > "$Q/pr.$n"
    tr ' ' '\n' < "$BASE.r.$n" | grep -v '^$' | tr '\n' ' ' > "$Q/g.$n"
    txt=$(python3 tools/ds4f_tokenizer.py decode --tokenizer "$TOK" --ids-file "$Q/g.$n" 2>/dev/null)
    echo "--- req $n ---"; echo "  PROMPT: $(cat "$Q/pr.$n")"; echo "  GEN   : $txt"
done
echo "[gate] inspect the completions above: coherent => the serve path is FIXED."
exit $fail
