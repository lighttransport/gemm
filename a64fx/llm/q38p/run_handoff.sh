#!/usr/bin/env bash
# One existing allocation, PP12 prefill -> snapshot -> TP1/2/4 decode.
# Load/repack and process startup are outside the engine's phase timers.
set -euo pipefail
cd "$(dirname "$0")/../../.."
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
if (( $# < 1 )); then
    echo "usage: $0 NEW_RUN_DIR [PROMPT_TOKENS=1024] [GEN=256] [CHUNK=160] [PROMPT_FILE]" >&2
    exit 2
fi
run=$1
pn=${2:-1024}
gn=${3:-256}
chunk=${4:-160}
prompt_file=${5:-}
mkdir "$run"
prefill_pn=${HANDOFF_PREFIX_TOKENS:-$pn}
export HANDOFF_RUN="$run" HANDOFF_PN="$prefill_pn" HANDOFF_GN="$gn" HANDOFF_CHUNK="$chunk" HANDOFF_PROMPT="$prompt_file"
export HANDOFF_KV_I6=${HANDOFF_KV_I6:-0}
export HANDOFF_PREFILL_KV_I6=${HANDOFF_PREFILL_KV_I6:-1}
if [[ $HANDOFF_KV_I6 == 1 && $HANDOFF_PREFILL_KV_I6 == 1 ]]; then
    export Q38P_ATTN_CACHE=0 Q38P_ATTN_QTILE=${HANDOFF_PREFILL_QTILE:-1} Q38P_ATTN_PV_INT16=0
fi
export HANDOFF_MODEL=${HANDOFF_MODEL:-$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf}
export HANDOFF_DECODER=${HANDOFF_DECODER:-a64fx/llm/build/q38d}
export HANDOFF_PREFILL=${HANDOFF_PREFILL:-tmp/q38p/q38p_pp}
export HANDOFF_IMAGE=${HANDOFF_IMAGE:-/local/q38/fp4.image}
export TOFU_TOPO_PATH=${TOFU_TOPO_PATH:-$PWD/tofu_topo.txt}
mpiexec -n 12 bash -c '
    args=(); [[ -z $HANDOFF_PROMPT ]] || args+=(--prompt-file "$HANDOFF_PROMPT")
    if [[ $HANDOFF_KV_I6 == 1 ]]; then
        if [[ $HANDOFF_PREFILL_KV_I6 == 1 ]]; then args+=(--prefill-kv-i6)
        else args+=(--state-out-i6); fi
    fi
    Q38D_TP=1 Q38P_PP=1 Q38P_PP_PROMPTS=1 Q38P_PP_DECODE=0 Q38P_CHUNK="$HANDOFF_CHUNK" \
      /usr/bin/time -f "phase_resource: wall_s=%e max_rss_kib=%M" "$HANDOFF_PREFILL" "$HANDOFF_MODEL" --image "$HANDOFF_IMAGE" --fmt fp4 --act a16 \
      --prompt-tokens "$HANDOFF_PN" --gen 1 "${args[@]}" --state-out "$HANDOFF_RUN/state" \
      > "$HANDOFF_RUN/prefill.rank${PMIX_RANK}.log" 2>&1
'
grep -E 'q38p_pp: nodes=|q38p_prefill:|q38d_state:' "$run/prefill.rank0.log"
bash a64fx/llm/q38p/run_decode_state.sh "$run" "$run/state" "$pn" "$gn" "$prompt_file"
