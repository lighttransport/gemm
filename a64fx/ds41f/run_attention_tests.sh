#!/bin/sh
# Invoke once with mpiexec -np 12, from a fresh shared results directory.
set -eu
test "$#" -eq 1 || { echo "usage: $0 STAGE_ROOT" >&2; exit 2; }
rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}}
test -n "$rank"
here=${DS41F_BIN_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)}
exec >"attention.rank${rank}.log" 2>&1
case "$rank" in
    0) layers=0 ;;
    2) layers='2 14' ;;
    8) layers='8 20' ;;
    *) layers='' ;;
esac
for layer in $layers; do
    "$here/test_attention" "$1/rank$rank" "attention.layer${layer}.bin" "$layer" 6
done
case "$rank" in
    0) "$here/test_index_candidates" "$1/rank$rank" 24 ;;
    8) "$here/test_index_candidates" "$1/rank$rank" 20 ;;
esac
echo "ATTENTION_RANK PASS rank=$rank"
