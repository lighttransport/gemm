#!/bin/sh
# Build the existing shared CPU preview renderer into the reference cache.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$project_dir/../.." && pwd)
export TMPDIR="$root/tmp/pixal3d"
mkdir -p "$TMPDIR" "$project_dir/.cache"
cat > "$TMPDIR/stbiw.cc" <<'CPP'
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
CPP
${CXX:-c++} -std=c++17 -O3 -Wall -Wextra -Wno-missing-field-initializers -fopenmp \
    -I"$root/common" "$root/common/preview_render_main.cc" "$root/common/lightrt.cc" \
    "$root/common/tinyexr_impl.cc" "$TMPDIR/stbiw.cc" \
    -o "$project_dir/.cache/preview_render.new" -lpthread
mv "$project_dir/.cache/preview_render.new" "$project_dir/.cache/preview_render"
