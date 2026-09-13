#!/bin/sh
# Compare a real free-running greedy continuation from CPU exact and HIP serve.
#
# This is deliberately not a teacher-forced logit test: it catches graph or
# state configuration mistakes that only become visible after one decoded token.
# It needs a fully staged single-node model image and a HIP-enabled
# libds4f_serve.so.  The default prompt IDs are the tokenizer encoding of:
#   Write a short Python function that adds two numbers.
#
# Usage: sh a64fx/llm/test_ds4f_decode_parity.sh [stage_dir]
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
STAGE=${1:-/tmp/ds4f_single}
LIB=${DS4F_SERVE_LIB:-"$HERE/../../libds4f_serve.so"}
THREADS=${LLM_THREADS:-16}
MAX_NEW=${MAX_NEW:-20}
READY_TIMEOUT=${READY_TIMEOUT:-900}
BASE=${TMPDIR:-/tmp}/ds4f_decode_parity.$$
PROMPT='0 21750 260 3107 15255 2019 396 16803 1234 3737 16'
RUNNER=

cleanup() {
    if [ -n "$RUNNER" ]; then
        kill "$RUNNER" 2>/dev/null || true
        wait "$RUNNER" 2>/dev/null || true
    fi
    rm -f "$BASE".*
}
trap cleanup EXIT HUP INT TERM

clear_protocol() {
    rm -f "$BASE.req" "$BASE.resp" "$BASE.reqseq" "$BASE.respseq" \
        "$BASE.error" "$BASE.ready" "$BASE.tok" "$BASE".slot.*
}

run_mode() {
    mode=$1
    tag=$2
    RUNNER=
    clear_protocol
    env DS4F_SERVE_BASE="$BASE" DS4F_STAGE_DIR="$STAGE" \
        DS4F_SERVE_LIB="$LIB" DS4F_SERVE_USE_HIP="$mode" \
        DS4F_MAXPOS=4096 LLM_THREADS="$THREADS" DS4F_CMGS=1 \
        python3 "$HERE/ds4f_serve_runner.py" >"$BASE.$tag.log" 2>&1 &
    RUNNER=$!
    i=0
    while [ ! -f "$BASE.ready" ]; do
        if ! kill -0 "$RUNNER" 2>/dev/null; then
            cat "$BASE.$tag.log" >&2 || true
            return 1
        fi
        i=$((i + 1))
        if [ "$i" -ge "$READY_TIMEOUT" ]; then
            echo "$tag runner did not become ready in ${READY_TIMEOUT}s" >&2
            return 1
        fi
        sleep 1
    done
    printf '%s 0 1 0 0 1 1 0 0\n%s\n' "$MAX_NEW" "$PROMPT" >"$BASE.req.tmp"
    mv "$BASE.req.tmp" "$BASE.req"
    printf '1\n' >"$BASE.reqseq.tmp"
    mv "$BASE.reqseq.tmp" "$BASE.reqseq"
    i=0
    while [ ! -f "$BASE.respseq" ] || [ "$(tr -d '[:space:]' < "$BASE.respseq")" != 1 ]; do
        if ! kill -0 "$RUNNER" 2>/dev/null; then
            cat "$BASE.$tag.log" >&2 || true
            return 1
        fi
        i=$((i + 1))
        if [ "$i" -ge "$READY_TIMEOUT" ]; then
            echo "$tag request did not complete in ${READY_TIMEOUT}s" >&2
            return 1
        fi
        sleep 1
    done
    cp "$BASE.resp" "$BASE.$tag.ids"
    kill "$RUNNER" 2>/dev/null || true
    wait "$RUNNER" 2>/dev/null || true
    RUNNER=
}

run_mode 0 cpu
run_mode 1 hip

if ! cmp -s "$BASE.cpu.ids" "$BASE.hip.ids"; then
    echo 'FAIL: CPU exact and HIP greedy continuations differ' >&2
    diff -u "$BASE.cpu.ids" "$BASE.hip.ids" >&2 || true
    exit 1
fi
count=$(wc -w <"$BASE.cpu.ids")
echo "PASS: CPU exact == HIP greedy continuation (${count}/${MAX_NEW} tokens)"
