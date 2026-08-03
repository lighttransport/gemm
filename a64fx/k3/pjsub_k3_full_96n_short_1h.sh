#!/bin/bash
# One-hour TP96 full-weight short-context end-to-end run.
# Measures real full-model prefill and generation after rank-local staging.
#PJM -g hp250467
#PJM -L "rscgrp=small-s2,node=96,elapse=01:01:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
# Propagate launch-time overrides from the login1 environment into the job.
# Without these directives pjsub silently used the shell defaults below.
#PJM -x K3_BARRIER_ITERS
#PJM -x K3_THREADS
#PJM -x K3_PREFILL_TOKENS
#PJM -x K3_NEW_TOKENS
#PJM -x K3_PREFILL_CHUNK
#PJM -x K3_COMM_DETERMINISTIC
#PJM -x K3_COMM_BF16
#PJM -x K3_COMM_ROBUST
#PJM -x K3_COMM_POLL_SPINS
#PJM -x K3_COMM_A2A
#PJM -x K3_COMM_A2A_MAX
#PJM -x K3_PREFETCH_MIB
#PJM -x K3_PROFILE
#PJM -x K3_MODEL_DIR
#PJM -x K3_FULL_STAGE_DIR
#PJM -x K3_AR_GROUPS
#PJM -x K3_MOE_SHARD_LAYOUT
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
UTOFU="$REPO/a64fx/utofu-tests"
MODEL_DIR=${K3_MODEL_DIR:-$HOME/models/kimi-k3}
NODES=96
THREADS=${K3_THREADS:-47}
PREFILL_TOKENS=${K3_PREFILL_TOKENS:-256}
NEW_TOKENS=${K3_NEW_TOKENS:-256}
PREFILL_CHUNK=${K3_PREFILL_CHUNK:-64}
BARRIER_ITERS=${K3_BARRIER_ITERS:-128}
COMM_DETERMINISTIC=${K3_COMM_DETERMINISTIC:-1}
COMM_BF16=${K3_COMM_BF16:-0}
COMM_ROBUST=${K3_COMM_ROBUST:-2}
COMM_POLL_SPINS=${K3_COMM_POLL_SPINS:-4}
COMM_A2A=${K3_COMM_A2A:-0}
COMM_A2A_MAX=${K3_COMM_A2A_MAX:-8192}
PREFETCH_MIB=${K3_PREFETCH_MIB:-16}
PROFILE=${K3_PROFILE:-0}
AR_GROUPS=${K3_AR_GROUPS:-16}
MOE_SHARD_LAYOUT=${K3_MOE_SHARD_LAYOUT:-replicated}
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$K3/logs/full-96n-short-1h-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-full-short-1h-$JOB_TAG"
if [ -n "${K3_FULL_STAGE_DIR:-}" ]; then STAGE_DIR=$K3_FULL_STAGE_DIR; fi
TIMING="$ROOT/stage_timing.tsv"

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export K3_PYTHON="$K3/.venv-$(uname -m)/bin/python" K3_EXPERT_TP=1 K3_MOE_SHARD_LAYOUT="$MOE_SHARD_LAYOUT"

[[ ! -e "$ROOT" ]] || { echo "$0: result root exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
printf 'stage\tstart_epoch\tend_epoch\telapsed_s\trc\n' >"$TIMING"
printf 'K3_FULL_CONFIG nodes=%s threads=%s barrier_iters=%s comm_deterministic=%s comm_bf16=%s comm_robust=%s comm_poll_spins=%s comm_a2a=%s comm_a2a_max=%s prefetch_mib=%s profile=%s ' \
    "$NODES" "$THREADS" "$BARRIER_ITERS" "$COMM_DETERMINISTIC" "$COMM_BF16" \
    "$COMM_ROBUST" "$COMM_POLL_SPINS" "$COMM_A2A" "$COMM_A2A_MAX" "$PREFETCH_MIB" "$PROFILE" | tee "$ROOT/config.txt"
printf 'prefill_tokens=%s new_tokens=%s prefill_chunk=%s\n' \
    "$PREFILL_TOKENS" "$NEW_TOKENS" "$PREFILL_CHUNK" | tee -a "$ROOT/config.txt"
printf 'stage_dir=%s ar_groups=%s moe_shard_layout=%s\n' "$STAGE_DIR" "$AR_GROUPS" "$MOE_SHARD_LAYOUT" | tee -a "$ROOT/config.txt"
"$K3/k3_setup_python.sh"

stage_begin() {
    STAGE_NAME=$1
    STAGE_START=$(date +%s)
    printf 'K3_FULL_SHORT_STAGE_BEGIN stage=%s epoch=%s\n' "$STAGE_NAME" "$STAGE_START"
}
stage_end() {
    STAGE_RC=$1
    STAGE_END=$(date +%s)
    printf '%s\t%s\t%s\t%s\t%s\n' "$STAGE_NAME" "$STAGE_START" "$STAGE_END" \
        "$((STAGE_END - STAGE_START))" "$STAGE_RC" >>"$TIMING"
    printf 'K3_FULL_SHORT_STAGE_END stage=%s elapsed_s=%s rc=%s\n' \
        "$STAGE_NAME" "$((STAGE_END - STAGE_START))" "$STAGE_RC"
}

stage_begin build
make -C "$K3" full-runner >/dev/null
make -C "$UTOFU" tofu_topo_helper >/dev/null
stage_end 0

stage_begin topology
mpiexec -np "$NODES" "$UTOFU/tofu_topo_helper" >"$ROOT/topology.log"
mv tofu_topo.txt "$ROOT/tofu_topo.txt"
[[ $(grep -vc '^#' "$ROOT/tofu_topo.txt") -eq "$NODES" ]]
stage_end 0

stage_begin barrier_preflight
BARRIER_PREFIX="$ROOT/barrier.rank"
mpiexec -np "$NODES" -of-proc "$BARRIER_PREFIX" "$K3/k3_full_runner" \
    --mode barrier --nodes "$NODES" --topo "$ROOT/tofu_topo.txt" \
    --barrier-iters "$BARRIER_ITERS"
grep -h 'K3FULL_BARRIER' "$BARRIER_PREFIX".* >"$ROOT/barrier.log"
stage_end 0

cat >"$ROOT/prompt.txt" <<'EOF'
You are an expert C++20 systems programmer. Implement a bounded concurrent
queue with precise memory-ordering comments, shutdown handling, and tests.
Return compilable code first, then explain the design and failure cases.
EOF
"$K3/k3_python.sh" "$K3/make_k3_prompt_ids.py" --vocab "$MODEL_DIR/tiktoken.model" \
    --text "$ROOT/prompt.txt" --output "$ROOT/prompt.ids" \
    --tokens "$PREFILL_TOKENS" --bos --repeat-to

stage_begin full_weight_staging
if [ -n "${K3_FULL_STAGE_DIR:-}" ]; then
    [[ -s "$STAGE_DIR/rank000.manifest" && -s "$STAGE_DIR/rank095.manifest" ]] || {
        echo "K3_FULL_STAGE_DIR is missing prepared rank manifests: $STAGE_DIR" >&2; exit 4;
    }
    echo "K3_FULL_REUSE_STAGE dir=$STAGE_DIR"
else
    mpiexec -np "$NODES" -of-proc "$ROOT/stage.rank" sh -c \
        "exec '$K3/run_k3_full_stage_rank.sh' '$MODEL_DIR' '$STAGE_DIR' '$NODES' \"\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PMI_RANK:?no MPI rank}}}\""
fi
stage_end 0

stage_begin full_short_generation
PROFILE_ARGS=()
if [ "$PROFILE" -ne 0 ]; then PROFILE_ARGS=(--profile "$ROOT/profile.txt"); fi
mpiexec -np "$NODES" -of-proc "$ROOT/run.rank" \
    "$K3/k3_full_runner" --mode full96 --stage-dir "$STAGE_DIR" \
    --topo "$ROOT/tofu_topo.txt" --prompt-ids "$ROOT/prompt.ids" \
    --output "$ROOT/output.txt" --prefill-tokens "$PREFILL_TOKENS" \
    --new-tokens "$NEW_TOKENS" --prefill-chunk "$PREFILL_CHUNK" \
    --max-seq "$((PREFILL_TOKENS + NEW_TOKENS))" --threads "$THREADS" \
    --comm-deterministic "$COMM_DETERMINISTIC" --comm-bf16 "$COMM_BF16" \
    --comm-robust "$COMM_ROBUST" --comm-poll-spins "$COMM_POLL_SPINS" \
    --comm-a2a "$COMM_A2A" --comm-a2a-max "$COMM_A2A_MAX" \
    --prefetch-mib "$PREFETCH_MIB" \
    --ar-groups "$AR_GROUPS" "${PROFILE_ARGS[@]}"
stage_end 0

stage_begin validation
"$K3/k3_python.sh" "$K3/validate_k3_full_output.py" "$ROOT/output.txt" \
    --nodes "$NODES" --mode full96 | tee "$ROOT/validation.txt"
"$K3/k3_python.sh" "$K3/decode_k3_output.py" "$ROOT/output.txt" \
    --vocab "$MODEL_DIR/tiktoken.model" \
    --text-output "$ROOT/generated.txt" | tee "$ROOT/decode.txt"
stage_end 0

printf 'K3_FULL_SHORT_96 status=PASS prefill_tokens=%s new_tokens=%s output=%s timing=%s\n' \
    "$PREFILL_TOKENS" "$NEW_TOKENS" "$ROOT/output.txt" "$TIMING"
