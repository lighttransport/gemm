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

ssh -o IdentitiesOnly=yes "$REMOTE" "cd $REMOTE_REPO && pjsub --no-check-directory -x FRONTEND_HOST=$FRONTEND_HOST -x FRONTEND_SSH_TARGET=$FRONTEND_SSH_TARGET -x FRONTEND_SSH_TARGETS=$FRONTEND_SSH_TARGETS -x FRONTEND_PORT=$FRONTEND_PORT -x TOKEN=$TOKEN '$JOB_SCRIPT'"
