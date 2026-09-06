#!/bin/sh
set -eu

if [ "$#" -ne 7 ]; then
    echo "usage: $0 SCRIPT_DIR STAGE_DIR WORLD_NODES TP_NODES LAYER EXPERTS THREADS" >&2
    exit 2
fi
script_dir=$1
stage_dir=$2
world_nodes=$3
tp_nodes=$4
layer=$5
experts=$6
threads=$7
rank=${PMIX_RANK:?PMIX_RANK is not set}
tp_rank=$((rank % tp_nodes))
expert_list="0-$((experts - 1))"
marker="$stage_dir/stage-rank$(printf '%03d' "$rank").status"
expected="rank=$rank nodes=$world_nodes tp_rank=$tp_rank tp_nodes=$tp_nodes layer=$layer experts=$expert_list"
if [ ! -f "$marker" ] || [ "$(cat "$marker")" != "$expected" ]; then
    echo "$0: rank $rank stage marker mismatch: $marker" >&2
    exit 3
fi

set --
expert=0
while [ "$expert" -lt "$experts" ]; do
    expert_dir="$stage_dir/expert$(printf '%03d' "$expert")"
    stem="layer$(printf '%02d' "$layer")_expert$(printf '%03d' "$expert")"
    set -- "$@" "$expert_dir/$stem.blob" "$expert_dir/$stem.manifest"
    expert=$((expert + 1))
done
exec "$script_dir/k3_moe_probe" --prefill --threads "$threads" "$@"
