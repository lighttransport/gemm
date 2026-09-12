#!/bin/sh
set -eu
rank=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}}
test -n "$rank"
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec > "stage.rank${rank}.log" 2>&1
hostname
exec python3 "$here/stage_backbone.py" --model /home/u14346/models/ds41f \
    --destination "/local/$USER/ds41f-$PJM_JOBID/rank$rank" --rank "$rank"
