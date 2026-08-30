#!/bin/sh
set -eu
rank=${PMIX_RANK:?PMIX_RANK is required}
src=$1/rank-$(printf '%02d' "$rank")
dst=$2/rank-$(printf '%02d' "$rank")
mkdir -p "$dst"
cp --reflink=always "$src/tp12-v5.blob" "$dst/tp12-v5.blob"
cp "$src/tp12-v5.manifest" "$dst/tp12-v5.manifest"
