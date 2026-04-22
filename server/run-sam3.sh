#!/bin/sh
# Launch diffusion-server with SAM 3 CPU + CUDA backends.
# Set MODELS to the directory containing sam3/{sam3.model.safetensors,vocab.json,merges.txt}.
: "${MODELS:?set MODELS=/path/to/models}"
./build/diffusion-server --host 0.0.0.0 --web-root ../web \
    --sam3-ckpt   "$MODELS/sam3/sam3.model.safetensors" \
    --sam3-vocab  "$MODELS/sam3/vocab.json" \
    --sam3-merges "$MODELS/sam3/merges.txt"
