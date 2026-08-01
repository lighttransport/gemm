#!/bin/bash
# Validate that a GLM-5.2 Hugging Face safetensors checkout is complete enough
# before submitting/staging a Fugaku job.
set -u

MODEL_DIR=${1:-${GLM5_MODEL_DIR:-$HOME/models/glm5.2}}
NSHARDS=${GLM5_NSHARDS:-282}
NEED_TOKENIZER=0
if [ "${2:-}" = "--tokenizer" ]; then
    NEED_TOKENIZER=1
fi

fail() {
    echo "FATAL: $*" >&2
    exit 2
}

[ -d "$MODEL_DIR" ] || fail "model dir not found: $MODEL_DIR"
[ -f "$MODEL_DIR/config.json" ] || fail "missing config.json in $MODEL_DIR"
[ -f "$MODEL_DIR/model.safetensors.index.json" ] || fail "missing model.safetensors.index.json in $MODEL_DIR"

missing=0
for i in $(seq 1 "$NSHARDS"); do
    shard=$(printf "%s/model-%05d-of-%05d.safetensors" "$MODEL_DIR" "$i" "$NSHARDS")
    if [ ! -s "$shard" ]; then
        missing=$((missing + 1))
        if [ "$missing" -le 10 ]; then
            echo "missing shard: $shard" >&2
        fi
    fi
done

if [ "$missing" -ne 0 ]; then
    have=$((NSHARDS - missing))
    fail "GLM-5.2 download incomplete: $have/$NSHARDS safetensor shards present in $MODEL_DIR"
fi

if [ "$NEED_TOKENIZER" -ne 0 ] && [ ! -s "$MODEL_DIR/tokenizer.json" ]; then
    fail "missing tokenizer.json in $MODEL_DIR"
fi

echo "GLM-5.2 model preflight OK: $MODEL_DIR"
