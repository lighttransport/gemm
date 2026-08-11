#!/bin/sh
# Single-node DS4F serving wrapper for llmgr and the RX 9070 XT HIP path.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BASE=${DS4F_SERVE_BASE:-$HERE/.ds4f_serve}
SOCK=${DS4F_SERVE_SOCKET:-$BASE.sock}
STAGE=${DS4F_STAGE_DIR:?set DS4F_STAGE_DIR to the staged full-weight manifest}
PORT=${PORT:-8080}
TOK=${TOK:-${DS4F_TOKENIZER:-$HOME/models/ds4f/tokenizer.json}}
RUNNER_LOG=${DS4F_RUNNER_LOG:-$HERE/ds4f_runner.log}
FRONT_LOG=${DS4F_FRONTEND_LOG:-$HERE/ds4f_frontend.log}
AGENT_CACHE_MAX=8192
RUNNER_TIMEOUT=3600
_cache_next=0
_timeout_next=0
for _arg in "$@"; do
  if [ "$_cache_next" = 1 ]; then AGENT_CACHE_MAX=$_arg; _cache_next=0; continue; fi
  if [ "$_timeout_next" = 1 ]; then RUNNER_TIMEOUT=$_arg; _timeout_next=0; continue; fi
  case "$_arg" in
    --agent-cache-max-tokens) _cache_next=1 ;;
    --agent-cache-max-tokens=*) AGENT_CACHE_MAX=${_arg#*=} ;;
    --runner-timeout-sec) _timeout_next=1 ;;
    --runner-timeout-sec=*) RUNNER_TIMEOUT=${_arg#*=} ;;
  esac
done

rm -f "$BASE".req "$BASE".resp "$BASE".reqseq "$BASE".respseq \
      "$BASE".tok "$BASE".conv.* "$BASE".slot.*
rm -f "$SOCK"
mkdir -p "$(dirname "$BASE")"
mkdir -p "$(dirname "$RUNNER_LOG")" "$(dirname "$FRONT_LOG")"

export DS4F_SERVE_BASE="$BASE" DS4F_STAGE_DIR="$STAGE"
export DS4F_SERVE_LIB=${DS4F_SERVE_LIB:-$HERE/../../libds4f_serve.so}
export DS4F_SERVE_USE_HIP=${DS4F_SERVE_USE_HIP:-1}
export DS4F_HIP_DEVICE=${DS4F_HIP_DEVICE:-0}
export DS4F_MAXPOS=${DS4F_MAXPOS:-16384}
export DS4F_SERVE_PREFIX_CACHE=${DS4F_SERVE_PREFIX_CACHE:-1}
export DS4F_SERVE_SLOTS=${DS4F_SERVE_SLOTS:-1}
export DS4F_SERVE_AGENT_CACHE_DIR=${DS4F_SERVE_AGENT_CACHE_DIR:-$BASE.agent-cache}

python3 "$HERE/ds4f_serve_runner.py" --unix-socket "$SOCK" "$@" >"$RUNNER_LOG" 2>&1 &
RUNNER_PID=$!
FRONT_PID=""
cleanup() {
  [ -z "$FRONT_PID" ] || kill "$FRONT_PID" 2>/dev/null || true
  kill "$RUNNER_PID" 2>/dev/null || true
  # Native full-model forwards defer Python's SIGTERM handler. Bound shutdown
  # so a restart cannot leave an old runner competing for CPU and memory.
  for _ in $(seq 1 50); do
    kill -0 "$RUNNER_PID" 2>/dev/null || break
    sleep 0.1
  done
  kill -KILL "$RUNNER_PID" 2>/dev/null || true
  wait "$RUNNER_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

for _ in $(seq 1 360); do
  grep -qE 'serving on|cooperative socket=' "$RUNNER_LOG" 2>/dev/null && break
  kill -0 "$RUNNER_PID" 2>/dev/null || {
    tail -40 "$RUNNER_LOG" >&2 || true
    exit 1
  }
  sleep 1
done
grep -qE 'serving on|cooperative socket=' "$RUNNER_LOG" || {
  echo "DS4F runner did not become ready; see $RUNNER_LOG" >&2
  exit 1
}

PORT="$PORT" TOK="$TOK" DS4F_SERVE_BASE="$BASE" \
  DS4F_SERVE_AGENT_CACHE_DIR="$DS4F_SERVE_AGENT_CACHE_DIR" \
  python3 "$HERE/ds4f_serve.py" --runner-socket "$SOCK" --port "$PORT" \
    --tokenizer "$TOK" --agent-cache-max-tokens "$AGENT_CACHE_MAX" \
    --runner-timeout-sec "$RUNNER_TIMEOUT" \
    >"$FRONT_LOG" 2>&1 &
FRONT_PID=$!
wait "$FRONT_PID"
