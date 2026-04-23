#!/bin/sh
# Launch the three servers that back web/sam3_compare.html:
#   - C diffusion-server         on :8080   (sam3 + sam3.1, cpu/cuda)
#   - sam3   pytorch ref server  on :8082
#   - sam3.1 pytorch ref server  on :8081
#
# Then open http://localhost:8080/sam3_compare in a browser.
#
# Env:
#   MODELS     (required) directory holding sam3/ and sam3.1/ checkpoints
#   PYTHON     (optional) python interpreter (default: python3)
#   HOST       (optional) bind host (default: 0.0.0.0)
#   DEVICE     (optional) torch device for ref servers (default: cuda)
#   LOG_DIR    (optional) where per-server logs land (default: ./logs)
#   SKIP_REF_SAM3 / SKIP_REF_SAM31  set to 1 to skip that ref server
#
# Ctrl-C stops all three.

set -eu

: "${MODELS:?set MODELS=/path/to/models (must contain sam3/ and sam3.1/)}"
PYTHON="${PYTHON:-python3}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-./logs}"
# Ref servers bind to 127.0.0.1 by default (they only need to be reachable
# from the C server, which proxies them to the browser). Override REF_HOST
# if the ref servers run on a different machine than the C server.
REF_HOST="${REF_HOST:-127.0.0.1}"
REF_SAM3_PORT="${REF_SAM3_PORT:-8082}"
REF_SAM31_PORT="${REF_SAM31_PORT:-8081}"
REF_SAM3_URL="${REF_SAM3_URL:-http://$REF_HOST:$REF_SAM3_PORT}"
REF_SAM31_URL="${REF_SAM31_URL:-http://$REF_HOST:$REF_SAM31_PORT}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_BIN="$SCRIPT_DIR/build/diffusion-server"

SAM3_CKPT="$MODELS/sam3/sam3.model.safetensors"
SAM3_VOCAB="$MODELS/sam3/vocab.json"
SAM3_MERGES="$MODELS/sam3/merges.txt"
SAM31_CKPT_C="$MODELS/sam3.1/sam3.1.model.safetensors"
SAM31_CKPT_PT="$MODELS/sam3.1/sam3.1_multiplex.pt"

die() { echo "error: $*" >&2; exit 1; }

[ -x "$SERVER_BIN" ] || die "$SERVER_BIN not built — run: cmake --build $SCRIPT_DIR/build -j"
[ -f "$SAM3_CKPT" ]  || die "missing $SAM3_CKPT"
[ -f "$SAM3_VOCAB" ] || die "missing $SAM3_VOCAB"
[ -f "$SAM3_MERGES" ]|| die "missing $SAM3_MERGES"
[ -f "$SAM31_CKPT_C" ] || die "missing $SAM31_CKPT_C (convert with cuda/sam3.1/convert_pt_to_safetensors.py)"

mkdir -p "$LOG_DIR"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"

PIDS=""
cleanup() {
    echo
    echo "[run-sam3-compare] stopping children: $PIDS"
    # shellcheck disable=SC2086
    [ -n "$PIDS" ] && kill $PIDS 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[run-sam3-compare] C server  -> http://$HOST:8080   (log: $LOG_DIR/c-server.log)"
"$SERVER_BIN" --host "$HOST" --port 8080 --web-root "$REPO_ROOT/web" \
    --sam3-ckpt      "$SAM3_CKPT" \
    --sam3-ckpt-v31  "$SAM31_CKPT_C" \
    --sam3-vocab     "$SAM3_VOCAB" \
    --sam3-merges    "$SAM3_MERGES" \
    --sam3-ref-url   "$REF_SAM3_URL" \
    --sam3-1-ref-url "$REF_SAM31_URL" \
    >"$LOG_DIR/c-server.log" 2>&1 &
PIDS="$PIDS $!"

if [ "${SKIP_REF_SAM3:-0}" != "1" ]; then
    echo "[run-sam3-compare] sam3 ref   -> $REF_SAM3_URL   (log: $LOG_DIR/ref-sam3.log)"
    ( cd "$REPO_ROOT/ref/sam3" && \
      "$PYTHON" sam3_ref_server.py \
          --host "$REF_HOST" --port "$REF_SAM3_PORT" \
          --device "$DEVICE" \
          --ckpt "$SAM3_CKPT" \
          >"$LOG_DIR/ref-sam3.log" 2>&1 ) &
    PIDS="$PIDS $!"
fi

if [ "${SKIP_REF_SAM31:-0}" != "1" ]; then
    if [ ! -f "$SAM31_CKPT_PT" ]; then
        echo "[run-sam3-compare] warning: $SAM31_CKPT_PT missing, skipping sam3.1 ref" >&2
    else
        echo "[run-sam3-compare] sam3.1 ref -> $REF_SAM31_URL   (log: $LOG_DIR/ref-sam3-1.log)"
        ( cd "$REPO_ROOT/ref/sam3.1" && \
          "$PYTHON" sam3_1_ref_server.py \
              --host "$REF_HOST" --port "$REF_SAM31_PORT" \
              --device "$DEVICE" \
              --ckpt "$SAM31_CKPT_PT" \
              >"$LOG_DIR/ref-sam3-1.log" 2>&1 ) &
        PIDS="$PIDS $!"
    fi
fi

echo
echo "[run-sam3-compare] all launched. Open:"
echo "    http://localhost:8080/sam3_compare"
echo "Ctrl-C to stop. Tail logs with:  tail -F $LOG_DIR/*.log"
echo

wait
