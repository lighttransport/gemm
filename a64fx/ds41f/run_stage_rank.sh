#!/bin/sh
set -eu
rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}}
test -n "$rank"
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec > "stage.rank${rank}.log" 2>&1
hostname
stage_root=${DS41F_STAGE_ROOT:-/local/$USER/ds41f-$PJM_JOBID}
exec python3 "$here/stage_backbone.py" --model /home/u14346/models/ds41f \
    --destination "$stage_root/rank$rank" --rank "$rank"
