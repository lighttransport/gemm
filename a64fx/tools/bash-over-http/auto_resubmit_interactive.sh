#!/bin/bash
set -uo pipefail

# Keep an interactive bash-over-HTTP bridge available across job expiry.
#
# Runs run_bash_http_interactive.sh in a loop. When an allocation ends (time
# limit, node failure, or a failed wait for resources), a new one is
# requested after RESUBMIT_DELAY seconds. After each new allocation passes the
# local health check, the optional READY_HOOK file is sent to the new
# compute node through the bash-over-HTTP client (for example to stage model
# images into the fresh /local). All launcher environment variables
# (NODES, ELAPSE, WAIT_TIME, REMOTE_REPO, PORT_OFFSET, A64FX_MODE, ...) pass
# through unchanged.
#
#   LOCAL_URL        client URL of the local tunnel end (default
#                    http://127.0.0.1:42386 or the configured local
#                    port, plus PORT_OFFSET)
#   READY_HOOK       local file with Bash commands to run on each new node
#   READY_TIMEOUT    seconds allowed for the hook (default 3600)
#   MAX_JOBS         stop after this many allocations (default 4)
#   UNTIL_EPOCH      do not submit after this Unix time (default: none)
#   RESUBMIT_DELAY   seconds between an allocation's end and resubmission
#                    (default 60; failed waits back off up to 600)
#   LOG_DIR          per-job launcher logs (default tmp/bash-http-auto)
#
# Stop with Ctrl-C, or by creating $LOG_DIR/STOP (checked between jobs).
# Stopping this wrapper does not cancel an allocation already running; use
# pjdel for that.

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../../.." && pwd)
cd "$REPO_ROOT"
eval "$(python3 "$HERE/config.py" --project-dir "$REPO_ROOT" --shell)"

PORT_OFFSET=${PORT_OFFSET:-0}
LOCAL_URL=${LOCAL_URL:-http://127.0.0.1:$((${BASH_HTTP_LOCAL_PORT:-42386} + PORT_OFFSET))}
READY_HOOK=${READY_HOOK:-}
READY_TIMEOUT=${READY_TIMEOUT:-3600}
MAX_JOBS=${MAX_JOBS:-4}
UNTIL_EPOCH=${UNTIL_EPOCH:-0}
RESUBMIT_DELAY=${RESUBMIT_DELAY:-60}
LOG_DIR=${LOG_DIR:-tmp/bash-http-auto}
mkdir -p "$LOG_DIR"

log() { echo "[auto-resubmit $(date '+%F %T')] $*" | tee -a "$LOG_DIR/auto.log"; }

run_hook() {
    [[ -n "$READY_HOOK" ]] || return 0
    log "running ready hook $READY_HOOK"
    BH_URL="$LOCAL_URL" BH_TIMEOUT="$READY_TIMEOUT" python3 - "$READY_HOOK" \
        >>"$LOG_DIR/hook.log" 2>&1 <<'EOF'
import os, sys
sys.path.insert(0, "a64fx/tools/bash-over-http")
import bash_http_client as bh
bh.BASE_URL = os.environ["BH_URL"]
with bh.Shell() as shell:
    r = shell.run(open(sys.argv[1]).read(), timeout=int(os.environ["BH_TIMEOUT"]), normalize=True)
    print(r.stdout)
    print("HOOK_EXIT", r.code)
    sys.exit(r.code or 0)
EOF
    log "ready hook exit $?"
}

backoff=$RESUBMIT_DELAY
for ((job = 1; job <= MAX_JOBS; job++)); do
    if [[ -e "$LOG_DIR/STOP" ]]; then log "STOP file present; exiting"; break; fi
    if (( UNTIL_EPOCH > 0 && $(date +%s) >= UNTIL_EPOCH )); then log "past UNTIL_EPOCH; exiting"; break; fi

    jlog="$LOG_DIR/job-$(date +%Y%m%d-%H%M%S).log"
    log "submitting allocation $job/$MAX_JOBS (log $jlog)"
    "$HERE/run_bash_http_interactive.sh" >"$jlog" 2>&1 &
    lpid=$!

    ready=0
    while kill -0 "$lpid" 2>/dev/null; do
        if curl -sf -m 5 "$LOCAL_URL/health" >/dev/null 2>&1; then ready=1; break; fi
        sleep 10
    done
    if (( ready )); then
        jobid=$(grep -oE 'Job [0-9]+ submitted' "$jlog" | head -1 | awk '{print $2}')
        log "allocation ready (job ${jobid:-unknown})"
        echo "${jobid:-}" >"$LOG_DIR/current_jobid"
        run_hook
        backoff=$RESUBMIT_DELAY
    fi
    wait "$lpid"
    rc=$?
    log "launcher exited with $rc: $(grep -E 'PJM|PLE' "$jlog" | tail -1)"
    rm -f "$LOG_DIR/current_jobid"

    if (( ready )); then
        sleep "$RESUBMIT_DELAY"
    else
        # No allocation within WAIT_TIME, or submission error: back off.
        log "no allocation; retrying in ${backoff}s"
        sleep "$backoff"
        backoff=$(( backoff * 2 > 600 ? 600 : backoff * 2 ))
    fi
done
log "done"
