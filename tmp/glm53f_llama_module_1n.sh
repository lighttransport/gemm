#!/bin/bash
set -euo pipefail
export LLAMA_SRC="$HOME/work/llama.cpp"
export LLAMA_BUILD="/local/glm53f-llama-a64fx-${PJM_JOBID:-manual}"
export JOBS=8
export TMPDIR="$LLAMA_BUILD/tmp"
export REPO="${REPO:-$PWD}"
export OUT="${OUT:-/local/glm53f-validation-${PJM_JOBID:-manual}}"
mkdir -p "$OUT"
export GLM53F_STAGE_OUT="${GLM53F_STAGE_OUT:-$OUT/llama-stage}"
export GLM53F_CUSTOM_STAGE_OUT="${GLM53F_CUSTOM_STAGE_OUT:-$OUT/custom-stage}"
export GLM53F_STAGE_PROMPT="${GLM53F_STAGE_PROMPT:-token-id-suite}"
echo "LLAMA_MODULE host=$(hostname) arch=$(uname -m) mpi=${PJM_MPI_PROC:-unknown}"
bash a64fx/glm5/build_llama_a64fx_ninja.sh
bash a64fx/glm5/run_ggml_a64fx_smoke.sh

FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/src" -I"$LLAMA_SRC/include" \
  -I"$LLAMA_SRC/ggml/include" -I"$LLAMA_SRC/ggml/src" \
  "$REPO/a64fx/glm5/glm53f_llama_loader_probe.cpp" \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libllama.so" \
  -lggml -lggml-cpu -lggml-base \
  -o "$OUT/glm53f_llama_loader_probe"
"$OUT/glm53f_llama_loader_probe" \
  "$HOME/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf"

FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/src" -I"$LLAMA_SRC/include" \
  -I"$LLAMA_SRC/ggml/include" -I"$LLAMA_SRC/ggml/src" \
  "$REPO/a64fx/glm5/glm53f_stream_model_probe.cpp" \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libllama.so" \
  -lggml -lggml-cpu -lggml-base \
  -o "$OUT/glm53f_stream_model_probe"
"$OUT/glm53f_stream_model_probe" \
  "$HOME/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf" 11

FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/src" -I"$LLAMA_SRC/include" \
  -I"$LLAMA_SRC/ggml/include" -I"$LLAMA_SRC/ggml/src" \
  "$REPO/a64fx/glm5/glm53f_llama_router_compare.cpp" \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libllama.so" -lggml -lggml-cpu -lggml-base \
  -o "$OUT/glm53f_llama_router_compare"
"$OUT/glm53f_llama_router_compare" \
  "$HOME/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf"
python3 a64fx/glm5/glm53f_validation.py artifact-compare \
  "$GLM53F_STAGE_OUT/manifest.jsonl" "$GLM53F_CUSTOM_STAGE_OUT/manifest.jsonl" \
  --threshold 0

FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/src" -I"$LLAMA_SRC/include" \
  -I"$LLAMA_SRC/ggml/include" -I"$LLAMA_SRC/ggml/src" \
  "$REPO/a64fx/glm5/glm53f_llama_dense_stage.cpp" \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libllama.so" -lggml -lggml-cpu -lggml-base \
  -o "$OUT/glm53f_llama_dense_stage"
export GLM53F_DENSE_LLAMA_OUT="$OUT/dense-llama"
export GLM53F_DENSE_CUSTOM_OUT="$OUT/dense-custom"
"$OUT/glm53f_llama_dense_stage" \
  "$HOME/models/glm53f-gguf/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf"
python3 a64fx/glm5/glm53f_validation.py artifact-compare \
  "$GLM53F_DENSE_LLAMA_OUT/manifest.jsonl" "$GLM53F_DENSE_CUSTOM_OUT/manifest.jsonl" \
  --threshold 1e-3

FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
  "$REPO/a64fx/glm5/glm53f_llama_indexer_stage.cpp" \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libggml.so" "$LLAMA_BUILD/bin/libggml-cpu.so" \
  "$LLAMA_BUILD/bin/libggml-base.so" -o "$OUT/glm53f_llama_indexer_stage"
export GLM53F_INDEXER_LLAMA_OUT="$OUT/indexer-llama"
export GLM53F_INDEXER_CUSTOM_OUT="$OUT/indexer-custom"
"$OUT/glm53f_llama_indexer_stage"
python3 a64fx/glm5/glm53f_validation.py artifact-compare \
  "$GLM53F_INDEXER_LLAMA_OUT/manifest.jsonl" "$GLM53F_INDEXER_CUSTOM_OUT/manifest.jsonl" \
  --threshold 1e-5

FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
  "$REPO/a64fx/glm5/glm53f_llama_tail_stage.cpp" \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libggml.so" "$LLAMA_BUILD/bin/libggml-cpu.so" \
  "$LLAMA_BUILD/bin/libggml-base.so" -o "$OUT/glm53f_llama_tail_stage"
export GLM53F_TAIL_LLAMA_OUT="$OUT/tail-llama"
export GLM53F_TAIL_CUSTOM_OUT="$OUT/tail-custom"
tail_variants=${GLM53F_TAIL_VARIANTS:-0,1,2}
IFS=',' read -r -a tail_variant_list <<< "$tail_variants"
for tail_variant in "${tail_variant_list[@]}"; do
  export GLM53F_TAIL_VARIANT="$tail_variant"
  export GLM53F_TAIL_LLAMA_OUT="$OUT/tail-llama-v$tail_variant"
  export GLM53F_TAIL_CUSTOM_OUT="$OUT/tail-custom-v$tail_variant"
  "$OUT/glm53f_llama_tail_stage"
  python3 a64fx/glm5/glm53f_validation.py artifact-compare \
    "$GLM53F_TAIL_LLAMA_OUT/manifest.jsonl" "$GLM53F_TAIL_CUSTOM_OUT/manifest.jsonl" \
    --threshold 3e-3
done

FCC -Nclang -O3 -std=c++17 -fopenmp \
  -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
  "$REPO/a64fx/glm5/glm53f_llama_kda_stage.cpp" \
  -L"$LLAMA_BUILD/bin" -Wl,-rpath,"$LLAMA_BUILD/bin" \
  "$LLAMA_BUILD/bin/libggml.so" "$LLAMA_BUILD/bin/libggml-cpu.so" \
  "$LLAMA_BUILD/bin/libggml-base.so" -o "$OUT/glm53f_llama_kda_stage"
export GLM53F_KDA_LLAMA_OUT="$OUT/kda-llama"
export GLM53F_KDA_CUSTOM_OUT="$OUT/kda-custom"
"$OUT/glm53f_llama_kda_stage"
python3 a64fx/glm5/glm53f_validation.py artifact-compare \
  "$GLM53F_KDA_LLAMA_OUT/manifest.jsonl" "$GLM53F_KDA_CUSTOM_OUT/manifest.jsonl" \
  --threshold 2e-5
echo "LLAMA_MODULE PASS cross_lane_artifacts=embedding,router,ffn_layer0,dsa_indexer,kda_layer3,mla,moe,mhc,norm,logits,greedy_token build=$LLAMA_BUILD"
