#!/bin/bash
set -euo pipefail

# Fugaku frontend login node to pin (1..8) -> loginN.fugaku.r-ccs.riken.jp.
# Set LOGIN_NODE (or REMOTE) before launch to choose the frontend. The reverse
# tunnel must land on the SAME node open_bash_http_local_tunnel.sh forwarded, so
# this prefers the REMOTE recorded in state.env. Use a hostname, NOT an IP
# (compute nodes can't ssh the internal frontend IP) and NOT a comma list
# (pjsub -x treats commas as variable separators and rejects the job at submit).
LOGIN_NODE=${LOGIN_NODE:-1}
case "$LOGIN_NODE" in [1-8]) ;; *) echo "LOGIN_NODE must be 1..8 (got '$LOGIN_NODE')" >&2; exit 2 ;; esac
REMOTE_DEFAULT="login${LOGIN_NODE}.fugaku.r-ccs.riken.jp"

REMOTE_ENV=${REMOTE:-}            # explicit REMOTE= override wins over state.env
REMOTE_REPO=${REMOTE_REPO:-\$HOME/work/gemm/ds4p}
JOB_SCRIPT=${JOB_SCRIPT:-tools/pjsub_bash_http_1n.sh}
STATE_FILE=${STATE_FILE:-/tmp/ds4p-bash-http/state.env}
FRONTEND_HOST=${FRONTEND_HOST:-}
FRONTEND_SSH_TARGET=${FRONTEND_SSH_TARGET:-}
FRONTEND_SSH_TARGETS=${FRONTEND_SSH_TARGETS:-}
FRONTEND_PORT=${FRONTEND_PORT:-21364}
TOKEN=${TOKEN:-${DS4P_BASH_HTTP_TOKEN:?set DS4P_BASH_HTTP_TOKEN (bash-over-http bearer token) before submitting}}

# Resource / lifetime. small rscgrp allows elapse up to 72:00:00; the supervised
# reverse tunnel (see pjsub_bash_http_1n.sh) keeps the bridge alive for the whole
# window, reconnecting on drops up to MAX_RETRY consecutive failures.
RSCGRP=${RSCGRP:-small}
ELAPSE=${ELAPSE:-06:00:00}              # HH:MM:SS, max 72:00:00 for rscgrp=small
MAX_RETRY=${MAX_RETRY:-10}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-15}
KEEPALIVE_SECONDS=${KEEPALIVE_SECONDS:-0}   # 0 = run until elapse; >0 = self-exit after N s

if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
fi

# Precedence: explicit REMOTE= env > state.env REMOTE (the forwarded node) > loginN default.
# All FRONTEND_* default to the single pinned hostname (no IP, no comma list).
REMOTE=${REMOTE_ENV:-${REMOTE:-$REMOTE_DEFAULT}}
FRONTEND_HOST=${FRONTEND_HOST:-$REMOTE}
FRONTEND_SSH_TARGET=${FRONTEND_SSH_TARGET:-$REMOTE}
FRONTEND_SSH_TARGETS=${FRONTEND_SSH_TARGETS:-$REMOTE}

ssh -o IdentitiesOnly=yes "$REMOTE" "cd $REMOTE_REPO && pjsub --no-check-directory -L \"rscgrp=$RSCGRP,node=1,elapse=$ELAPSE\" -x FRONTEND_HOST=$FRONTEND_HOST -x FRONTEND_SSH_TARGET=$FRONTEND_SSH_TARGET -x FRONTEND_SSH_TARGETS=$FRONTEND_SSH_TARGETS -x FRONTEND_PORT=$FRONTEND_PORT -x MAX_RETRY=$MAX_RETRY -x MONITOR_INTERVAL=$MONITOR_INTERVAL -x KEEPALIVE_SECONDS=$KEEPALIVE_SECONDS -x TOKEN=$TOKEN '$JOB_SCRIPT'"
