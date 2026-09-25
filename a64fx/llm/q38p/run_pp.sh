#!/usr/bin/env bash
set -euo pipefail

# Run inside an allocation after the FP4 image has been staged on every node.
cd "$(dirname "$0")/../../.."
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
nodes=${1:?usage: run_pp.sh NODES [PROMPTS=8] [TOKENS=1024] [CHUNK=160] [DECODE=0]}
prompts=${2:-8}
tokens=${3:-1024}
chunk=${4:-160}
decode=${5:-0}
if (( nodes < 2 || nodes > 12 || prompts < 1 || tokens < 1 || chunk < 5 )); then
    echo "invalid pipeline arguments" >&2
    exit 2
fi

mkdir -p tmp/q38p/pp_runs
export Q38P_PP=1 Q38P_PP_PROMPTS="$prompts" Q38P_PP_DECODE="$decode" Q38P_CHUNK="$chunk"
export Q38P_MODEL=${Q38P_MODEL:-$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf}
export Q38P_IMAGE=${Q38P_IMAGE:-/local/q38/fp4.image}
export Q38P_RUN_TAG=${Q38P_RUN_TAG:-pp${nodes}_t${tokens}_p${prompts}_c${chunk}_d${decode}}
export Q38P_TOKENS="$tokens"
export Q38P_GEN=$(( decode ? 256 : 1 ))
mpiexec -n "$nodes" sh -c '
    mkdir -p /local/q38/bin
    cp tmp/q38p/q38p_pp /local/q38/bin/q38p_pp
    Q38P_PP="$Q38P_PP" Q38P_PP_PROMPTS="$Q38P_PP_PROMPTS" \
      Q38P_PP_DECODE="$Q38P_PP_DECODE" Q38P_CHUNK="$Q38P_CHUNK" \
      /local/q38/bin/q38p_pp "$Q38P_MODEL" --fmt fp4 --image "$Q38P_IMAGE" \
      --act a16 --prompt-tokens "$Q38P_TOKENS" --gen "$Q38P_GEN" \
      > "tmp/q38p/pp_runs/${Q38P_RUN_TAG}.rank${PMIX_RANK}.log" 2>&1
'
grep 'q38p_pp: nodes=' "tmp/q38p/pp_runs/${Q38P_RUN_TAG}.rank0.log"
if (( decode )); then
    if [[ -z ${Q38P_REF:-} && "$tokens" != 1024 ]]; then
        echo "set Q38P_REF for a non-1024-token decode comparison" >&2
        exit 2
    fi
    python3 tmp/q38-fast/compare_tokens.py "${Q38P_REF:-/local/q38/ref-f32.log}" \
        "tmp/q38p/pp_runs/${Q38P_RUN_TAG}.rank0.log"
fi
