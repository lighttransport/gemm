#!/usr/bin/env bash
# Export one PP12 INT6 state per distinct prompt for later TP4 grouped decode.
set -euo pipefail
cd "$(dirname "$0")/../../.."
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
manifest=${1:?usage: export_context_states_12n.sh PROMPT_PATH_LIST NEW_RUN_DIR [PROMPT_TOKENS=128] [CHUNK=160]}
run=${2:?new run directory required}
pn=${3:-128}
chunk=${4:-160}
(( pn > 0 && chunk > 0 ))
mapfile -t prompts < "$manifest"
(( ${#prompts[@]} > 0 ))
mkdir "$run"
export Q38P_MODEL=${Q38P_MODEL:-$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf}
export Q38P_IMAGE=${Q38P_IMAGE:-/local/q38/fp4.image}
export Q38P_BIN=${Q38P_BIN:-tmp/q38p/q38p_pp}
export Q38P_EXPORT_FAST=${Q38P_EXPORT_FAST:-0}
if [[ $Q38P_EXPORT_FAST == 0 ]]; then
    export Q38P_ATTN_CACHE=0 Q38P_ATTN_QTILE=1 Q38P_ATTN_PV_INT16=0
fi
export Q38P_EXPORT_PN="$pn" Q38P_EXPORT_CHUNK="$chunk"
for b in "${!prompts[@]}"; do
    [[ -n ${prompts[b]} && -f ${prompts[b]} ]] || { echo "missing prompt: ${prompts[b]}" >&2; exit 2; }
    export Q38P_EXPORT_PROMPT="${prompts[b]}" Q38P_EXPORT_STATE="$run/state$b" Q38P_EXPORT_RUN="$run" Q38P_EXPORT_INDEX="$b"
    mpiexec -n 12 bash -c '
        args=();
        if [[ $Q38P_EXPORT_FAST == 1 ]]; then args+=(--state-out-i6)
        else args+=(--prefill-kv-i6); fi
        Q38D_TP=1 Q38P_PP=1 Q38P_PP_PROMPTS=1 Q38P_PP_DECODE=0 Q38P_CHUNK="$Q38P_EXPORT_CHUNK" \
          /usr/bin/time -f "phase_resource: wall_s=%e max_rss_kib=%M" \
          "$Q38P_BIN" "$Q38P_MODEL" --image "$Q38P_IMAGE" --fmt fp4 --act a16 \
          --prompt-file "$Q38P_EXPORT_PROMPT" --prompt-tokens "$Q38P_EXPORT_PN" --gen 1 \
          "${args[@]}" --state-out "$Q38P_EXPORT_STATE" \
          > "$Q38P_EXPORT_RUN/prefill${Q38P_EXPORT_INDEX}.rank${PMIX_RANK}.log" 2>&1
    '
    [[ -f $run/state$b/meta64.bin ]] || { echo "incomplete state $b" >&2; exit 1; }
    grep -E 'q38p_pp: nodes=|q38d_state: producer_nodes=' "$run/prefill$b.rank0.log"
done
