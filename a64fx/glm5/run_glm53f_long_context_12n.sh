#!/bin/bash
set -euo pipefail

# Reproducible 8K-to-16K scalar-greedy controls. Run with bash on a 12-node
# normal-frequency A64FX allocation after build_glm53f_integrated_12n.sh.
if [ "$#" -ne 5 ]; then
    echo "usage: $0 MODEL ROUTED_STAGE SHARED_STAGE PROMPT_IDS LOG_DIR" >&2
    exit 2
fi
model=$1
routed=$2
shared=$3
prompt=$4
logdir=$5
max_new=${GLM53F_MAX_NEW:-8192}
mkdir -p "$logdir"

# Packed prefill collectives require the allocation-specific uTofu topology.
# Generate it inside the log directory so this runner remains self-contained.
export GLM53F_UTOFU=1
if [ -z "${TOFU_TOPO_PATH:-}" ]; then
    (cd "$logdir" && mpiexec -np 12 ../utofu-tests/tofu_topo_helper)
    export TOFU_TOPO_PATH="$logdir/tofu_topo.txt"
fi

np=${PJM_MPI_PROC:-12}
if [ "$np" -ne 12 ]; then
    echo "expected 12 MPI ranks/nodes, got $np" >&2
    exit 2
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-48}
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=active
export OMP_PROC_BIND=close
export OMP_PLACES=cores

run_case() {
    local name=$1
    shift
    echo "GLM53F_LONG_CONTEXT case=$name"
    mpiexec -np 12 -of-proc "$logdir/$name" \
        ./glm53f_target_decode_12n "$model" "$routed" "$shared" \
        --generate "$prompt" "$logdir/$name.ids" "$max_new" \
        --decode-window 512 --ignore-eos --prefill-mode fast --prefill-chunk 512 \
        --prefill-features 27 --prefill-slab 16 --prefill-collective tree-packed \
        "$@" 2>&1 | tee "$logdir/$name.stdout"
}

# Replicated 16K-capacity control: FP32 latent cache, INT8 routed/shared and KDA.
run_case replicated-16k \
    --capacity 16384 --cache-format fp32 --weight-format int8 --int8-kda

# Deployment-shaped CP control: touched 512K capacity and BF16 latent cache.
run_case cp-512k-hot16k \
    --capacity 524288 --touch-cache --cache-format bf16 --weight-format int8 \
    --int8-kda --cp-hot-prefix 16384

echo "SENTINEL glm53f_long_context_12n=OK"
