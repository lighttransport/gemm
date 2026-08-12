#!/bin/bash
set -eu

REMOTE=${REMOTE:-fugaku}
REMOTE_REPO=${REMOTE_REPO:-'$HOME/work/gemm/ds4p'}
JOB_SCRIPT=${JOB_SCRIPT:-a64fx/utofu-tests/pjsub_mpi_hostid_16n.sh}

ssh "$REMOTE" "cd $REMOTE_REPO && pjsub --no-check-directory '$JOB_SCRIPT'"
