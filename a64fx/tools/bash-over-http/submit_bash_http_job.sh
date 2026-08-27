#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../../.." && pwd)
eval "$(python3 "$HERE/config.py" --project-dir "$REPO_ROOT" --shell)"

# Submit the compute-side bash-over-HTTP bridge through a pinned Fugaku login
# node. Run open_local_tunnel.sh first so REMOTE and the forwarded ports are
# recorded in the shared local state file.

LOGIN_NODE=${LOGIN_NODE:-1}
case "$LOGIN_NODE" in
    [1-8]) ;;
    *) echo "LOGIN_NODE must be 1..8 (got '$LOGIN_NODE')" >&2; exit 2 ;;
esac

if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    CONTROL_ROOT=$XDG_RUNTIME_DIR
elif [[ -d /local ]]; then
    CONTROL_ROOT=/local
else
    CONTROL_ROOT=tmp
fi
CONTROL_DIR=${CONTROL_DIR:-${BASH_HTTP_CONTROL_DIR:-$CONTROL_ROOT/clair-bash-http-${USER}}}
STATE_FILE=${STATE_FILE:-$CONTROL_DIR/state.env}

# Explicit environment values win over state recorded by open_local_tunnel.sh.
REMOTE_ENV=${REMOTE:-}
FRONTEND_SSH_TARGET_ENV=${FRONTEND_SSH_TARGET:-}
FRONTEND_PORT_ENV=${FRONTEND_PORT:-}
if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
fi

REMOTE=${REMOTE_ENV:-${REMOTE:-${BASH_HTTP_REMOTE:-fugaku1}}}
FRONTEND_SSH_TARGET=${FRONTEND_SSH_TARGET_ENV:-${FRONTEND_SSH_TARGET:-${BASH_HTTP_FRONTEND_SSH_TARGET:-login${LOGIN_NODE}.fugaku.r-ccs.riken.jp}}}
FRONTEND_HOST=${FRONTEND_HOST:-${BASH_HTTP_FRONTEND_HOST:-$FRONTEND_SSH_TARGET}}
FRONTEND_PORT=${FRONTEND_PORT_ENV:-${BASH_HTTP_FRONTEND_PORT:-${BASH_HTTP_REMOTE_PORT:-32386}}}
SERVER_HOST=${SERVER_HOST:-${BASH_HTTP_SERVER_HOST:-127.0.0.1}}
SERVER_PORT=${SERVER_PORT:-${BASH_HTTP_SERVER_PORT:-21264}}

PROJECT_ID=${PROJECT_ID:-${BASH_HTTP_PROJECT_ID:-hp250467}}
RSCGRP=${RSCGRP:-${BASH_HTTP_RSCGRP:-small}}
NODES=${NODES:-${BASH_HTTP_NODES:-1}}
ELAPSE=${ELAPSE:-${BASH_HTTP_ELAPSE:-08:00:00}}
A64FX_MODE=${A64FX_MODE:-${BASH_HTTP_A64FX_MODE:-normal}}
GFSCACHE=${GFSCACHE:-${BASH_HTTP_GFSCACHE:-/vol0004}}
LOCALTMP_SIZE=${LOCALTMP_SIZE:-${BASH_HTTP_LOCALTMP_SIZE:-87Gi}}
REMOTE_REPO=${REMOTE_REPO:-${BASH_HTTP_REMOTE_REPO:-\$HOME/work/gemm/glm53f}}
JOB_SCRIPT=${JOB_SCRIPT:-a64fx/tools/bash-over-http/pjsub_bash_http.sh}
MAX_RETRY=${MAX_RETRY:-10}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-15}
HEALTH_EVERY=${HEALTH_EVERY:-8}
KEEPALIVE_SECONDS=${KEEPALIVE_SECONDS:-0}

case "$A64FX_MODE" in
    normal) PJM_FREQ=2000; PJM_ECO_STATE=0 ;;
    boost-eco) PJM_FREQ=2200; PJM_ECO_STATE=2 ;;
    *) echo "A64FX_MODE must be normal or boost-eco" >&2; exit 2 ;;
esac

case "$NODES" in
    ''|*[!0-9]*) echo "NODES must be an integer from 1 through 12" >&2; exit 2 ;;
esac
if (( NODES < 1 || NODES > 12 )); then
    echo "NODES must be an integer from 1 through 12" >&2
    exit 2
fi
case "$FRONTEND_PORT:$SERVER_PORT" in
    *[!0-9:]*) echo "FRONTEND_PORT and SERVER_PORT must be integers" >&2; exit 2 ;;
esac
case "$REMOTE$FRONTEND_SSH_TARGET$FRONTEND_HOST" in
    *[!A-Za-z0-9._@-]*) echo "SSH host values contain unsupported characters" >&2; exit 2 ;;
esac
case "$ELAPSE" in
    [0-9][0-9]:[0-9][0-9]:[0-9][0-9]) ;;
    *) echo "ELAPSE must use HH:MM:SS" >&2; exit 2 ;;
esac

umask 077
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"

SUBMIT_OUTPUT=$(ssh -o BatchMode=yes -o IdentitiesOnly=yes "$REMOTE" \
    "cd $REMOTE_REPO && pjsub -g '$PROJECT_ID' --no-check-directory -L 'freq=$PJM_FREQ,eco_state=$PJM_ECO_STATE,rscgrp=$RSCGRP,node=$NODES,elapse=$ELAPSE' -x PJM_LLIO_GFSCACHE='$GFSCACHE' --llio localtmp-size='$LOCALTMP_SIZE' -x A64FX_MODE='$A64FX_MODE' -x FRONTEND_HOST='$FRONTEND_HOST' -x FRONTEND_SSH_TARGET='$FRONTEND_SSH_TARGET' -x FRONTEND_SSH_TARGETS='$FRONTEND_SSH_TARGET' -x FRONTEND_PORT='$FRONTEND_PORT' -x SERVER_HOST='$SERVER_HOST' -x SERVER_PORT='$SERVER_PORT' -x WORKDIR=\"\$PWD\" -x MAX_RETRY='$MAX_RETRY' -x MONITOR_INTERVAL='$MONITOR_INTERVAL' -x HEALTH_EVERY='$HEALTH_EVERY' -x KEEPALIVE_SECONDS='$KEEPALIVE_SECONDS' '$JOB_SCRIPT'")
printf '%s\n' "$SUBMIT_OUTPUT"
printf '%s\n' "$SUBMIT_OUTPUT" >"$CONTROL_DIR/last-submit.out"

JOB_ID=$(printf '%s\n' "$SUBMIT_OUTPUT" | sed -n 's/.*Job \([0-9][0-9]*\) submitted.*/\1/p' | tail -1)
if [[ -n "$JOB_ID" ]]; then
    printf '%s\n' "$JOB_ID" >"$CONTROL_DIR/jobid"
    chmod 600 "$CONTROL_DIR/jobid"
    echo "JOB_ID=$JOB_ID"
fi
