#!/bin/bash
# Prepared IQ1-S/M mixed-quant staging job. Submit explicitly with pjsub.
#PJM -g hp250467
#PJM -L "node=32"
#PJM --mpi "proc=32"
#PJM -L "elapse=01:00:00"
#PJM -j
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd)
MODEL=${K3_MODEL_DIR:-$HOME/models/k3/iq1}
OUT=${K3_STAGE_DIR:-/local/$USER/k3-iq1-32n-${PJM_JOBID:-manual}}
LAYER=${K3_LAYER_INDEX:-1}
REPS=${K3_BENCH_REPS:-2}
mkdir -p "$OUT"
mpiexec -np 32 sh -c 'r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PJM_MPI_RANK:?no rank}}}; exec python3 "$1/k3_gguf_stage.py" --model-dir "$2" --output-dir "$3" --rank "$r" --nodes 32 --layer-index "$4" --format iq1 --force' sh "$S" "$MODEL" "$OUT" "$LAYER"
mpiexec -np 32 -of-proc "$OUT/bench" sh -c '
    r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PJM_MPI_RANK:?no rank}}}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48} K3_QUANT_KERNEL=${K3_QUANT_KERNEL:-sve-q8}
    exec "$1/k3_gguf_layer_bench" "$2/rank$(printf "%03d" "$r").manifest" \
        "$2/rank$(printf "%03d" "$r").blob" "$3"
' sh "$S" "$OUT" "$REPS"
"$S/run_k3_ep.sh" --mode dummy --nodes 32 --tp-nodes 32 \
    --layers 1 --layer 3 --tokens 1 --cache-tokens 16384 \
    --threads 47 --kda-threads 8 --fused-threads 47 \
    --mla-cache-int8 --heartbeat-tokens 1 --min-available-mib 2048 \
    --ar-groups 2 --comm-robust 2 --result-dir "$OUT/context-preflight"

# Real GGUF text smoke test.  The K3-native safetensor runner and the IQ1
# GGUF package have different MLA layouts, so use the A64FX llama.cpp Kimi
# graph as the generation oracle while keeping the IQ1 dequant kernels above.
RPC_BIN=${K3_RPC_BIN:-$HOME/work/llama.cpp/build-a64fx-rpc-clang/bin/ggml-rpc-server}
TEXT_BIN=${K3_IQ1_TEXT_BIN:-$S/k3_iq1_text_runner}
TEXT_LIB=${K3_TEXT_LIB:-$HOME/work/llama.cpp/build-a64fx-rpc-clang/bin}
TEXT_OMP=${K3_TEXT_OMP:-/vol0004/apps/r/OSS_CN/.image/llvm/llvmorg-21.1.0-sep-2/own_clangfx/lib64}
MODEL_FILE=${K3_MODEL_FILE:-$MODEL/Kimi-K3-UD-IQ1_M-00001-of-00015.gguf}
if [[ ! -x "$RPC_BIN" || ! -x "$TEXT_BIN" || ! -f "$MODEL_FILE" ]]; then
    echo "K3_IQ1_TEXT_ERROR missing RPC/text/model binary" >&2
    exit 4
fi
mapfile -t RPC_HOSTS < <(mpiexec -np 32 hostname)
if (( ${#RPC_HOSTS[@]} != 32 )); then
    echo "K3_IQ1_TEXT_ERROR expected 32 hostnames, got ${#RPC_HOSTS[@]}" >&2
    exit 4
fi
RPC_LIST=""
for ((i = 0; i < 32; ++i)); do
    endpoint="${RPC_HOSTS[$i]}:$((52000 + i))"
    [[ -z "$RPC_LIST" ]] || RPC_LIST+=,
    RPC_LIST+="$endpoint"
done
export LD_LIBRARY_PATH="$TEXT_LIB:$TEXT_OMP:${LD_LIBRARY_PATH:-}"
mpiexec -np 32 sh -c '
    r=${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${PJM_MPI_RANK:?no rank}}}
    exec "$1" -H 0.0.0.0 -p "$((52000 + r))" -d CPU -t 47
' sh "$RPC_BIN" >"$OUT/rpc-server.log" 2>&1 &
RPC_PID=$!
trap 'kill "$RPC_PID" 2>/dev/null || true' EXIT
sleep "${K3_RPC_START_WAIT:-15}"
PROMPT='Answer both tasks briefly. QA: What does a C++ constructor do? C++: write a minimal C++17 function that adds two integers, with a one-line explanation.'
mpiexec -np 1 "$TEXT_BIN" "$MODEL_FILE" "$RPC_LIST" \
    "$OUT/iq1-short-output.txt" "$PROMPT" "${K3_TEXT_TOKENS:-96}"
kill "$RPC_PID" 2>/dev/null || true
wait "$RPC_PID" 2>/dev/null || true
trap - EXIT
echo "K3_IQ1_STAGE_READY nodes=32 layer=$LAYER output=$OUT"
echo "K3_IQ1_DEQUANT_CHECK nodes=32 layer=$LAYER kernel=${K3_QUANT_KERNEL:-sve-q8} logs=$OUT/bench.*"
echo "K3_IQ1_CONTEXT_CHECK nodes=32 cache_tokens=16384 cache=int8 result=$OUT/context-preflight"
echo "K3_IQ1_TEXT_CHECK nodes=32 output=$OUT/iq1-short-output.txt"
