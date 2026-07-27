#!/bin/bash
# Start llmgr on the head node of an EXISTING interactive allocation.
#
#   ./run_llmgr.sh                               # foreground
#   ./run_llmgr.sh --daemon                      # background, logs to logs/llmgr.<job>.log
#
# Auth is off by default: the port is loopback-only and Fugaku's fabric is not
# reachable from the internet. Set LLMGR_TOKEN to require a bearer token.
#
# For a batch job that also opens the reverse tunnel back to a frontend, use
# pjsub_llmgr_12n.sh instead.
set -euo pipefail
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
HERE="$(cd "$(dirname "$0")" && pwd)"

LLMGR_TOKEN=${LLMGR_TOKEN:-}
LLMGR_HOST=${LLMGR_HOST:-127.0.0.1}
LLMGR_PORT=${LLMGR_PORT:-21274}

DAEMON=0
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --daemon) DAEMON=1; shift;;
    *) ARGS+=("$1"); shift;;
  esac
done

mkdir -p "$HERE/logs"
LOG="$HERE/logs/llmgr.${PJM_JOBID:-nojob}.log"

CMD=(python3 -u "$HERE/llmgr_server.py"
     --host "$LLMGR_HOST" --port "$LLMGR_PORT")
[ -n "$LLMGR_TOKEN" ] && CMD+=(--token "$LLMGR_TOKEN")
CMD+=("${ARGS[@]+"${ARGS[@]}"}")

# Only send the header when a token is configured; an empty bearer would be
# rejected by a token-protected server and is noise against an open one.
CURL_AUTH=()
[ -n "$LLMGR_TOKEN" ] && CURL_AUTH=(-H "Authorization: Bearer $LLMGR_TOKEN")

if [ "$DAEMON" = 1 ]; then
  setsid "${CMD[@]}" >"$LOG" 2>&1 &
  pid=$!
  echo "llmgr pid=$pid log=$LOG"
  for _ in $(seq 1 40); do
    if curl -fsS "${CURL_AUTH[@]+"${CURL_AUTH[@]}"}" \
        "http://127.0.0.1:$LLMGR_PORT/health" >/dev/null 2>&1; then
      echo "llmgr healthy on http://$LLMGR_HOST:$LLMGR_PORT"
      exit 0
    fi
    kill -0 "$pid" 2>/dev/null || { echo "llmgr died; see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }
    sleep 0.5
  done
  echo "llmgr did not become healthy; see $LOG" >&2
  exit 1
fi

exec "${CMD[@]}"
