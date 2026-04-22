#!/bin/sh
# Launch diffusion-server with SAM 3.1 CUDA backend.
# Set MODELS to the directory containing sam3.1/{sam3.1.model.safetensors,vocab.json,merges.txt}.
: "${MODELS:?set MODELS=/path/to/models}"
./build/diffusion-server --host 0.0.0.0 --port 8082 --web-root ../web \
    --sam3-ckpt-v31 "$MODELS/sam3.1/sam3.1.model.safetensors" \
    --sam3-vocab    "$MODELS/sam3.1/vocab.json" \
    --sam3-merges   "$MODELS/sam3.1/merges.txt"
