#!/usr/bin/env bash
# Existing allocation: evaluate only a suffix at a controlled context depth.
set -euo pipefail
cd "$(dirname "$0")/../../.."
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
run=${1:?usage: run_depth.sh NEW_RUN_DIR [DEPTH=32768] [SUFFIX=1920] [REPEATS=6] [FILL=tokens] [CHUNK=480] [NODES=12]}
export DEPTH=${2:-32768} SUFFIX=${3:-1920} REPEATS=${4:-6} FILL=${5:-tokens} CHUNK=${6:-480}
export Q38P_KV_I6=${Q38P_KV_I6:-0}
export Q38P_I6_PARALLEL=${Q38P_I6_PARALLEL:-1}
nodes=${7:-12}
(( DEPTH > 0 && SUFFIX > 0 && REPEATS > 0 && CHUNK >= 5 && nodes >= 2 && nodes <= 12 ))
[[ $FILL == tokens || $FILL == synthetic-kv ]]
mkdir "$run"
export DEPTH_RUN="$run"
export Q38P_MODEL=${Q38P_MODEL:-$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf}
export Q38P_IMAGE=${Q38P_IMAGE:-/local/q38/fp4.image}
export Q38P_BIN=${Q38P_BIN:-tmp/q38p/q38p_pp}
mpiexec -n "$nodes" bash -c '
    set --
    if [[ $Q38P_KV_I6 == 1 ]]; then set -- --prefill-kv-i6 --prefill-i6-parallel "$Q38P_I6_PARALLEL"; fi
    Q38D_TP=1 Q38P_PP=1 Q38P_PP_PROMPTS="$REPEATS" Q38P_PP_DECODE=0 Q38P_CHUNK="$CHUNK" \
      /usr/bin/time -f "phase_resource: wall_s=%e max_rss_kib=%M" \
      "$Q38P_BIN" "$Q38P_MODEL" --image "$Q38P_IMAGE" --fmt fp4 --act a16 \
      --bench-depth "$DEPTH" --bench-fill "$FILL" --bench-seed 1 \
      --prompt-tokens "$SUFFIX" --gen 1 "$@" > "$DEPTH_RUN/rank${PMIX_RANK}.log" 2>&1
'
grep -E 'q38p_depth|q38p_pp: nodes=|q38p_prefill:' "$run/rank0.log"
