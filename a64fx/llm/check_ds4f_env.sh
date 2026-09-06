#!/bin/sh
# Validate the inputs needed for a single-node full-weight DS4F HIP run.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_DIR=${DS4F_MODEL_DIR:-$HOME/models/ds4f}
STAGE_DIR=${DS4F_STAGE_DIR:-/local/ds4f}
TOKENIZER=${DS4F_TOKENIZER:-${TOK:-$HOME/models/ds4f/tokenizer.json}}
NSHARDS=${DS4F_NSHARDS:-}
LIB=${DS4F_SERVE_LIB:-$HERE/../../libds4f_serve.so}
errors=0

fail() { echo "ERROR: $*" >&2; errors=$((errors + 1)); }
warn() { echo "WARN: $*" >&2; }

if [ -z "$NSHARDS" ] && [ -d "$MODEL_DIR" ]; then
    first=$(find "$MODEL_DIR" -maxdepth 1 -type f \
            -name 'model-00001-of-*.safetensors' -print -quit)
    if [ -n "$first" ]; then
        NSHARDS=$(basename "$first" | sed -n \
            's/.*-of-\([0-9][0-9]*\)\.safetensors/\1/p' | \
            sed 's/^0*//; s/^$/0/')
    fi
fi
NSHARDS=${NSHARDS:-46}

[ -d "$MODEL_DIR" ] || fail "model directory does not exist: $MODEL_DIR"
[ -r "$TOKENIZER" ] || fail "tokenizer is not readable: $TOKENIZER"
[ -r "$LIB" ] || fail "serving library is not readable: $LIB (run build_ds4f_serve.sh)"
command -v rocm-smi >/dev/null 2>&1 || fail "rocm-smi is not available"

total=0
missing=0
if [ -d "$MODEL_DIR" ]; then
    n=1
    while [ "$n" -le "$NSHARDS" ]; do
        shard=$(printf '%s/model-%05d-of-%05d.safetensors' "$MODEL_DIR" "$n" "$NSHARDS")
        if [ ! -r "$shard" ]; then
            missing=$((missing + 1))
        else
            size=$(stat -c '%s' "$shard" 2>/dev/null || stat -f '%z' "$shard")
            total=$((total + size))
        fi
        n=$((n + 1))
    done
fi
[ "$missing" -eq 0 ] || fail "$missing of $NSHARDS safetensors shards are missing under $MODEL_DIR"

if [ "$total" -gt 0 ]; then
    gib=$(awk -v b="$total" 'BEGIN { printf "%.1f", b/1073741824 }')
    echo "model:      $MODEL_DIR ($gib GiB, $NSHARDS shards)"
    [ "$total" -ge 100000000000 ] || warn "model is smaller than the expected full DS4F checkpoint"
fi
echo "stage:      $STAGE_DIR (NOCOPY manifest; source shards remain in place)"
echo "tokenizer:  $TOKENIZER"
echo "library:    $LIB"
rocm-smi --showproductname --showmeminfo vram 2>/dev/null |
    awk '/VRAM Total Memory|GFX Version|Card Model/ { print "gpu:        " $0 }' ||
    warn "ROCm is present but GPU details could not be queried"

if [ "$errors" -ne 0 ]; then
    echo "preflight: FAILED ($errors error(s))" >&2
    exit 1
fi
echo "preflight: OK"
