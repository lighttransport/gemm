#!/bin/bash
set -euo pipefail
export LLAMA_SRC="$HOME/work/llama.cpp"
export LLAMA_BUILD="/local/glm53f-stream-a64fx-${PJM_JOBID:-manual}"
export TMPDIR="$LLAMA_BUILD/tmp"
export JOBS=8
# Keep the per-layer F32 payloads and logs on the shared project filesystem.
# Set GLM53F_STREAM_OUT=/local/... only for a deliberately node-local run.
export OUT="${GLM53F_STREAM_OUT:-$PWD/tmp/glm53f-stream-validation-${PJM_JOBID:-manual}}"
export GLM53F_STREAM_TOKEN="${GLM53F_STREAM_TOKEN:-1}"
mkdir -p "$OUT"
bash a64fx/glm5/build_llama_a64fx_ninja.sh >/dev/null
FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/src" -I"$LLAMA_SRC/include" \
  -I"$LLAMA_SRC/ggml/include" -I"$LLAMA_SRC/ggml/src" \
  a64fx/glm5/glm53f_stream_model_probe.cpp \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libllama.so" -lggml -lggml-cpu -lggml-base \
  -o "$OUT/glm53f_stream_model_probe"
"$OUT/glm53f_stream_model_probe" \
  "$HOME/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf" 11
FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/src" -I"$LLAMA_SRC/include" \
  -I"$LLAMA_SRC/ggml/include" -I"$LLAMA_SRC/ggml/src" \
  a64fx/glm5/glm53f_stream_graph_probe.cpp \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libllama.so" -lggml -lggml-cpu -lggml-base \
  -o "$OUT/glm53f_stream_graph_probe"
mkdir -p "$OUT/graph"
manifest="$OUT/graph/manifest.jsonl"
: > "$manifest"
GRAPH_FIRST=${GLM53F_GRAPH_FIRST:-0}
GRAPH_LAST=${GLM53F_GRAPH_LAST:-44}
if [ "$GRAPH_FIRST" -le "$GRAPH_LAST" ]; then
for layer in $(seq "$GRAPH_FIRST" "$GRAPH_LAST"); do
  layer_name=$(printf '%02d' "$layer")
  log="$OUT/graph/layer-${layer_name}.log"
  if [ "$layer" -eq 0 ]; then
    if ! "$OUT/glm53f_stream_graph_probe" \
      "$HOME/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf" \
      "$OUT/graph" "$layer" >"$log" 2>&1; then
      tail -80 "$log"
      exit 1
    fi
  else
    prev=$((layer - 1))
    prev_name=$(printf '%02d' "$prev")
    if ! "$OUT/glm53f_stream_graph_probe" \
      "$HOME/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf" \
      "$OUT/graph" "$layer" "$OUT/graph/layer-${prev_name}.f32" >"$log" 2>&1; then
      tail -80 "$log"
      exit 1
    fi
  fi
  tail -1 "$log"
  printf '{"job":"%s","stage":"layer-%s","dtype":"f32","elements":16384,"artifact":"%s","log":"%s"}\n' \
    "${PJM_JOBID:-manual}" "$layer_name" "$OUT/graph/layer-${layer_name}.f32" "$log" >> "$manifest"
done
fi
echo "GLM53F_STREAM_GRAPH PASS layers=45 hidden=16384"
final_log="$OUT/graph/final.log"
final_input="$OUT/graph/layer-44.f32"
if [ -r "$final_input" ]; then
  final_args=("$OUT/graph" 45 "$final_input")
else
  final_args=("$OUT/graph" 45)
fi
if ! "$OUT/glm53f_stream_graph_probe" \
  "$HOME/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf" \
  "${final_args[@]}" >"$final_log" 2>&1; then
  tail -80 "$final_log"
  exit 1
fi
grep 'GLM53F_STREAM_CUSTOM_FINAL' "$final_log"
tail -1 "$final_log"
printf '{"job":"%s","stage":"final","dtype":"f32","elements":154880,"artifact":"%s","log":"%s"}\n' \
  "${PJM_JOBID:-manual}" "$OUT/graph/layer-45.f32" "$final_log" >> "$manifest"
echo "GLM53F_STREAM_PAYLOAD_JOB PASS job=${PJM_JOBID:-manual}"
