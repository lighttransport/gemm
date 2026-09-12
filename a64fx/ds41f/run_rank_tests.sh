#!/bin/sh
# Run in a shared, fresh results directory inside a 12-node allocation.
set -eu
rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}}
test -n "$rank"
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec > "kernel.rank${rank}.log" 2>&1
hostname
"$here/ds41f_sve_test"
"$here/test_ops"
"$here/test_engram"
if test -n "${DS41F_STAGE_ROOT:-}"; then
    "$here/test_staged_expert" "$DS41F_STAGE_ROOT/rank$rank" "$rank"
    if test -n "${DS41F_META_FILE:-}"; then
        cp "$DS41F_META_FILE" "$DS41F_STAGE_ROOT/rank$rank/engram_meta.bin"
        "$here/test_engram" "$DS41F_STAGE_ROOT/rank$rank" "$rank"
    fi
fi
