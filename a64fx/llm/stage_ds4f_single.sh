#!/bin/sh
# Stage a full DS4F model for the single-node HIP serving path.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_DIR=${DS4F_MODEL_DIR:?set DS4F_MODEL_DIR to the full DS4F model directory}
STAGE_DIR=${DS4F_STAGE_DIR:?set DS4F_STAGE_DIR for the staged manifest}
CC=${CC:-gcc}

mkdir -p "$HERE/build"
"$CC" -O2 -std=c11 -D_GNU_SOURCE -I"$HERE/../../common" \
  -o "$HERE/build/ds4f_stage" "$HERE/ds4f_stage.c"

NSHARDS=${DS4F_NSHARDS:-}
if [ -z "$NSHARDS" ]; then
  first=$(find "$MODEL_DIR" -maxdepth 1 -type f \
          -name 'model-00001-of-*.safetensors' -print -quit)
  [ -z "$first" ] || NSHARDS=$(basename "$first" | \
    sed -n 's/.*-of-\([0-9][0-9]*\)\.safetensors/\1/p' | \
    sed 's/^0*//; s/^$/0/')
fi
DS4F_EP_RANK=0 DS4F_EP_SIZE=1 DS4F_STAGE_NOCOPY=${DS4F_STAGE_NOCOPY:-1} \
DS4F_MODEL_DIR="$MODEL_DIR" DS4F_STAGE_DIR="$STAGE_DIR" \
DS4F_MODEL=${DS4F_MODEL:-ds4f} DS4F_NSHARDS=${NSHARDS:-46} \
  "$HERE/build/ds4f_stage"
