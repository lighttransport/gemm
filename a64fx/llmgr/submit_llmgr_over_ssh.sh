#!/bin/bash
# Submit from the workstation whose checkout is synchronized to Fugaku by
# Mutagen. Discovery and pjsub share one SSH session so a load-balanced alias
# cannot select a different frontend between the two operations.
set -euo pipefail

REMOTE=${REMOTE:-fugaku}
REMOTE_REPO=${REMOTE_REPO:-/vol0006/mdt0/data/hp250467/work/gemm/glm5-1}
JOB_SCRIPT=${JOB_SCRIPT:-a64fx/llmgr/pjsub_llmgr_laguna_fp8_12n.sh}
FRONTEND_PORT=${FRONTEND_PORT:-21374}

case "$FRONTEND_PORT" in
    ''|*[!0-9]*) echo "FRONTEND_PORT must be numeric" >&2; exit 2 ;;
esac

ssh "$REMOTE" bash -s -- "$REMOTE_REPO" "$JOB_SCRIPT" "$FRONTEND_PORT" <<'REMOTE_SH'
set -euo pipefail
repo=$1
job_script=$2
frontend_port=$3
cd "$repo"
[[ -r "$job_script" ]] || { echo "job script not found: $repo/$job_script" >&2; exit 2; }

login_host=$(hostname)
case "$login_host" in
    fn01sv0[1-8]) login_num=${login_host#fn01sv0} ;;
    fn01sv[1-8])  login_num=${login_host#fn01sv} ;;
    *) echo "cannot map Fugaku frontend hostname: $login_host" >&2; exit 2 ;;
esac
frontend="login${login_num}.fugaku.r-ccs.riken.jp"

echo "SYNCED_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "LOGIN_HOST=$login_host"
echo "FRONTEND=$frontend"
echo "FRONTEND_PORT=$frontend_port"
pjsub --no-check-directory \
    -x "LOGIN_NODE=$login_num" \
    -x "FRONTEND_HOST=$frontend" \
    -x "FRONTEND_SSH_TARGET=$frontend" \
    -x "FRONTEND_SSH_TARGETS=$frontend" \
    -x "FRONTEND_PORT=$frontend_port" \
    "$job_script"
REMOTE_SH
