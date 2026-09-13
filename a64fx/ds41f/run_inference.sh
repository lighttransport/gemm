#!/bin/sh
# Execute inside an existing 12-node allocation. Does not submit another job.
set -eu
if test "$#" -lt 3; then
    echo "usage: $0 STAGE_ROOT PROMPT_IDS NEW_SHARED_RESULTS_DIR [runner options]" >&2
    exit 2
fi
stage=$1
prompt=$(readlink -f -- "$2")
results=$3
shift 3
test -r "$prompt" || { echo "Unreadable prompt: $prompt" >&2; exit 2; }
case "$stage" in /local/*) ;; *) echo "Stage root must be under /local" >&2; exit 2 ;; esac
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
test -x "$here/ds41f_run" || { echo "Cross-build ds41f_run first" >&2; exit 2; }
# Fail if the directory exists: preserve prior logs and the executable in use.
mkdir -m 700 -- "$results"
cp -- "$here/ds41f_run" "$results/ds41f_run"
cd -- "$results"
sha256sum ds41f_run > binary.sha256
exec env XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
    OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
    mpiexec -np 12 ./ds41f_run --stage-root "$stage" --prompt-ids "$prompt" "$@"
