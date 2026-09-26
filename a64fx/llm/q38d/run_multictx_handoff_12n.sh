#!/usr/bin/env bash
# Evaluate distinct prompts on PP12, then decode their snapshots in three TP4 groups.
set -euo pipefail
cd "$(dirname "$0")/../../.."

run=${1:?usage: run_multictx_handoff_12n.sh NEW_RUN_DIR PROMPT_LIST [PROMPT_TOKENS=32768] [GEN=256] [CHUNK=160]}
list=${2:?PROMPT_LIST is required}
pn=${3:-32768}
gn=${4:-256}
chunk=${5:-160}
(( pn > 0 && gn > 0 && chunk >= 5 ))
mkdir "$run"
run=$(realpath "$run")
list=$(realpath "$list")
mapfile -t prompts < "$list"
count=${#prompts[@]}
(( count >= 6 && count <= 32 )) || { echo 'PROMPT_LIST must contain 6..32 paths' >&2; exit 2; }
for i in "${!prompts[@]}"; do
    [[ -f ${prompts[i]} ]] || { echo "missing prompt: ${prompts[i]}" >&2; exit 2; }
    prompts[i]=$(realpath "${prompts[i]}")
done

module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
export TMPDIR=/local/q38/tmp
export TOFU_TOPO_PATH=${TOFU_TOPO_PATH:-$PWD/tofu_topo.txt}
export Q38_MULTI_MODEL=${Q38_MULTI_MODEL:-$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf}
export Q38_MULTI_IMAGE=${Q38_MULTI_IMAGE:-/local/q38/fp4.image}
export Q38_MULTI_PREFILL=${Q38_MULTI_PREFILL:-tmp/q38p/q38p_pp}
export Q38_MULTI_KV_I6=${Q38_MULTI_KV_I6:-0}
[[ $Q38_MULTI_KV_I6 == 0 || $Q38_MULTI_KV_I6 == 1 ]] || { echo 'Q38_MULTI_KV_I6 must be 0 or 1' >&2; exit 2; }
if [[ $Q38_MULTI_KV_I6 == 1 ]]; then
    export Q38P_ATTN_CACHE=0 Q38P_ATTN_PV_INT16=0
    export Q38P_ATTN_QTILE=${Q38_MULTI_QTILE:-1}
fi
export Q38_MULTI_PN=$pn Q38_MULTI_CHUNK=$chunk

for i in "${!prompts[@]}"; do
    export Q38_MULTI_PROMPT=${prompts[i]} Q38_MULTI_STATE="$run/state$i"
    export Q38_MULTI_LOG="$run/prefill${i}"
    mpiexec -n 12 bash -c '
        args=()
        if [[ $Q38_MULTI_KV_I6 == 1 ]]; then args+=(--prefill-kv-i6); fi
        Q38D_TP=1 Q38P_PP=1 Q38P_PP_PROMPTS=1 Q38P_PP_DECODE=0 Q38P_CHUNK="$Q38_MULTI_CHUNK" \
          /usr/bin/time -f "phase_resource: wall_s=%e max_rss_kib=%M" \
          "$Q38_MULTI_PREFILL" "$Q38_MULTI_MODEL" --image "$Q38_MULTI_IMAGE" \
          --fmt fp4 --act a16 --prompt-file "$Q38_MULTI_PROMPT" \
          --prompt-tokens "$Q38_MULTI_PN" --gen 1 "${args[@]}" --state-out "$Q38_MULTI_STATE" \
          > "${Q38_MULTI_LOG}.rank${PMIX_RANK}.log" 2>&1
    '
    grep -E 'q38p_prefill:|q38d_state:' "${Q38_MULTI_LOG}.rank0.log"
done

prompt_prefix="$run/prompts"
state_prefix="$run/states"
group_counts=()
lo=0
for group in 0 1 2; do
    : > "${prompt_prefix}.group${group}.txt"
    : > "${state_prefix}.group${group}.txt"
    group_count=$(( count / 3 + (group < count % 3) ))
    group_counts+=("$group_count")
    hi=$(( lo + group_count ))
    for (( i=lo; i<hi; i++ )); do
        printf '%s\n' "${prompts[i]}" >> "${prompt_prefix}.group${group}.txt"
        printf '%s\n' "$run/state$i" >> "${state_prefix}.group${group}.txt"
    done
    lo=$hi
done
printf -v group_counts_csv '%s,%s,%s' "${group_counts[@]}"
Q38_GROUP_PRUNE=1 Q38_GROUP_KV_I6="$Q38_MULTI_KV_I6" \
Q38_GROUP_CONTEXTS_BY_GROUP="$group_counts_csv" \
Q38_GROUP_CONTEXT_PROMPT_PREFIX="$prompt_prefix" \
Q38_GROUP_CONTEXT_STATE_PREFIX="$state_prefix" \
    bash a64fx/llm/q38d/run_tp4_groups_12n.sh "$run/decode" "$pn" "$gn" 3
