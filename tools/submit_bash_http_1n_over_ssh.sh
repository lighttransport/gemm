#!/bin/bash
set -euo pipefail

REMOTE=${REMOTE:-login1.fugaku.r-ccs.riken.jp}
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

FRONTEND_HOST=${FRONTEND_HOST:-${REMOTE:-login1.fugaku.r-ccs.riken.jp}}
FRONTEND_SSH_TARGET=${FRONTEND_SSH_TARGET:-${REMOTE:-login1.fugaku.r-ccs.riken.jp}}
FRONTEND_SSH_TARGETS=${FRONTEND_SSH_TARGETS:-${REMOTE:-login1.fugaku.r-ccs.riken.jp},${REMOTE_IPV4:-10.4.128.23}}

ssh -o IdentitiesOnly=yes "$REMOTE" "cd $REMOTE_REPO && pjsub --no-check-directory -x FRONTEND_HOST=$FRONTEND_HOST -x FRONTEND_SSH_TARGET=$FRONTEND_SSH_TARGET -x FRONTEND_SSH_TARGETS='$FRONTEND_SSH_TARGETS' -x FRONTEND_PORT=$FRONTEND_PORT -x TOKEN=$TOKEN '$JOB_SCRIPT'"
