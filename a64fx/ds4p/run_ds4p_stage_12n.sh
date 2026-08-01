#!/bin/bash
# DeepSeek-V4-Pro layer-truncated stage on the 12-node interactive alloc.
#
# The full 61-layer DS4P blob is ~101 GB/rank at 11 ranks (> the 87 GiB /local),
# so interactive testing stages only the first $LAYERS layers (default 10:
# ~24 GB/rank, ~20 min at the shared-FS read rate). embed/head/out-norm are
# always staged; mtp.* stays gated on DS4F_STAGE_MTP.
#
#   ./run_ds4p_stage_12n.sh              # layers 0..9
#   LAYERS=4 ./run_ds4p_stage_12n.sh     # faster iteration loop
#
# Thin wrapper: all the vcoordfile / status-file machinery lives in
# ../llm/run_ds4f_stage_11n.sh (same 11-node EP set, relative-(0,0,0) excluded).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"

export DS4F_MODEL=ds4p
export DS4F_MODEL_DIR=${DS4F_MODEL_DIR:-$HOME/models/ds4p}
export DS4F_NSHARDS=${DS4F_NSHARDS:-64}
export DS4F_STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4p}
export DS4F_STAGE_LAYERS=${LAYERS:-10}
export DS4F_STAGE_COMPACT=${DS4F_STAGE_COMPACT:-1}

echo "[ds4p stage] model=$DS4F_MODEL_DIR layers=$DS4F_STAGE_LAYERS -> $DS4F_STAGE_DIR"
exec "$DIR/../llm/run_ds4f_stage_11n.sh"
