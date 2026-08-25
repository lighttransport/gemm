#!/bin/bash
# TP96 full-weight short-context end-to-end run.
# Measures real full-model prefill and generation after rank-local staging.
#
# Two hours is sufficient for the whole-expert baseline: prior jobs spent
# 2885--3103 s of their reservation on
# rank-local weight staging alone and was killed mid-validation, and 49922938
# spent 3103 s on the same stage.  One hour cannot fit staging plus generation
# plus validation.
#PJM -g hp250467
# Use the non-torus scalar placement accepted by the K3 96-node probes.
#PJM -L "rscgrp=small,node=96,elapse=02:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=87Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#
# There are deliberately no bare `#PJM -x NAME` directives here.  That form is
# what killed jobs 49973647, 49959412 and 49973486 with REASON=GATE CHECK
# before the script body ever ran: pjsub accepts the directive at submit time,
# then the gate check fails at scale.  Proven by canary, 96 nodes, 1 minute:
#   50001241  87Gi + bare -x     -> ERR / GATE CHECK
#   50001247  80Gi, no bare -x   -> ran
# `-x NAME=value` is the supported form.  Pass launch-time overrides on the
# pjsub command line instead:
#   pjsub --no-check-directory <script> --threads 47 --profile
# Runtime tuning is passed as script options; environment variables are
# reserved for debug instrumentation only.
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
UTOFU="$REPO/a64fx/utofu-tests"
MODEL_DIR="$HOME/models/kimi-k3"
NODES=96
THREADS=47
PREFILL_TOKENS=256
NEW_TOKENS=256
PREFILL_CHUNK=1024
BARRIER_ITERS=128
COMM_DETERMINISTIC=1
COMM_BF16=0
COMM_ROBUST=2
COMM_POLL_SPINS=4
COMM_A2A=0
COMM_A2A_MAX=8192
PREFETCH_MIB=0
PROFILE=0
AR_GROUPS=16
# Full-checkpoint staging defaults to one complete expert per owner.  The
# intermediate-channel TP layout is useful for bounded probes, but expands
# the full plan to ~497k tiny records and makes 96-rank staging metadata-bound.
# Opt into it only after a dedicated large-scale staging run proves the I/O
# path; the whole-expert layout is the reliable end-to-end 96-node baseline.
EXPERT_TP=0
MOE_SHARD_LAYOUT=replicated
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$K3/logs/full-96n-short-3h-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-full-short-3h-$JOB_TAG"
TIMING="$ROOT/stage_timing.tsv"

usage() {
    cat >&2 <<'EOF'
usage: pjsub_k3_full_96n_short_1h.sh [options]
  --model-dir DIR --threads N --prefill-tokens N --new-tokens N
  --prefill-chunk N --barrier-iters N --ar-groups N
  --comm-deterministic 0|1 --comm-bf16 0|1 --comm-robust N
  --comm-poll-spins N --comm-a2a 0|1 --comm-a2a-max N
  --prefetch-mib N --stage-chunk-mib N --stage-limit-seconds N --stage-dir DIR
  --profile --expert-tp --moe-shard-layout replicated|row-aligned
EOF
}
need_arg() { (($# >= 2)) || { echo "$0: $1 requires an argument" >&2; usage; exit 2; }; }
CHUNK_MIB=32
STAGE_LIMIT_SECONDS=${K3_STAGE_LIMIT_SECONDS:-5400}
while (($#)); do
    case "$1" in
        --model-dir) need_arg "$@"; MODEL_DIR=$2; shift 2;;
        --threads) need_arg "$@"; THREADS=$2; shift 2;;
        --prefill-tokens) need_arg "$@"; PREFILL_TOKENS=$2; shift 2;;
        --new-tokens) need_arg "$@"; NEW_TOKENS=$2; shift 2;;
        --prefill-chunk) need_arg "$@"; PREFILL_CHUNK=$2; shift 2;;
        --barrier-iters) need_arg "$@"; BARRIER_ITERS=$2; shift 2;;
        --ar-groups) need_arg "$@"; AR_GROUPS=$2; shift 2;;
        --comm-deterministic) need_arg "$@"; COMM_DETERMINISTIC=$2; shift 2;;
        --comm-bf16) need_arg "$@"; COMM_BF16=$2; shift 2;;
        --comm-robust) need_arg "$@"; COMM_ROBUST=$2; shift 2;;
        --comm-poll-spins) need_arg "$@"; COMM_POLL_SPINS=$2; shift 2;;
        --comm-a2a) need_arg "$@"; COMM_A2A=$2; shift 2;;
        --comm-a2a-max) need_arg "$@"; COMM_A2A_MAX=$2; shift 2;;
        --prefetch-mib) need_arg "$@"; PREFETCH_MIB=$2; shift 2;;
        --stage-chunk-mib) need_arg "$@"; CHUNK_MIB=$2; shift 2;;
        --stage-limit-seconds) need_arg "$@"; STAGE_LIMIT_SECONDS=$2; shift 2;;
        --stage-dir) need_arg "$@"; STAGE_DIR=$2; shift 2;;
        --profile) PROFILE=1; shift;;
        --expert-tp) EXPERT_TP=1; shift;;
        --moe-shard-layout) need_arg "$@"; MOE_SHARD_LAYOUT=$2; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown option $1" >&2; usage; exit 2;;
    esac
done
case "$MOE_SHARD_LAYOUT" in
    replicated|row-aligned) ;;
    *) echo "$0: invalid --moe-shard-layout '$MOE_SHARD_LAYOUT'" >&2; exit 2;;
esac
case "$STAGE_LIMIT_SECONDS" in
    ''|*[!0-9]*) echo "$0: invalid stage limit '$STAGE_LIMIT_SECONDS'" >&2; exit 2;;
esac
if ((STAGE_LIMIT_SECONDS < 1)); then
    echo "$0: stage limit must be positive" >&2
    exit 2
fi

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
# Keep idle workers spinning between parallel regions.  perf on a KDA layer put
# __kmp_fork_barrier at 37% of runtime with __sched_yield at 5%, i.e. threads
# were sleeping and paying a wakeup per region.  Measured 2.146 -> 2.072
# ms/layer, and it also removes most of the run-to-run variance.
export OMP_WAIT_POLICY=active KMP_BLOCKTIME=infinite
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export K3_PYTHON="$K3/.venv-$(uname -m)/bin/python"
# At 96 nodes, dense BF16 TP shards can be confined to one CMG. Replication
# avoids the resulting cross-CMG read bottleneck; disable explicitly if memory
# headroom is insufficient (about 4.14 GiB/rank for the full 93-layer image).
export K3_CMG_REPLICATE=${K3_CMG_REPLICATE:-1}
export K3_MLA_FLASH8=${K3_MLA_FLASH8:-1} K3_MLA_QK_MODE=${K3_MLA_QK_MODE:-auto}
export K3_COMM_ASYNC_LATENT=${K3_COMM_ASYNC_LATENT:-1}
export K3_PREFILL_PIPELINE=${K3_PREFILL_PIPELINE:-on}

[[ ! -e "$ROOT" ]] || { echo "$0: result root exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
printf 'stage\tstart_epoch\tend_epoch\telapsed_s\trc\n' >"$TIMING"
printf 'K3_FULL_CONFIG nodes=%s threads=%s barrier_iters=%s comm_deterministic=%s comm_bf16=%s comm_robust=%s comm_poll_spins=%s comm_a2a=%s comm_a2a_max=%s prefetch_mib=%s profile=%s ' \
    "$NODES" "$THREADS" "$BARRIER_ITERS" "$COMM_DETERMINISTIC" "$COMM_BF16" \
    "$COMM_ROBUST" "$COMM_POLL_SPINS" "$COMM_A2A" "$COMM_A2A_MAX" "$PREFETCH_MIB" "$PROFILE" | tee "$ROOT/config.txt"
printf 'prefill_tokens=%s new_tokens=%s prefill_chunk=%s\n' \
    "$PREFILL_TOKENS" "$NEW_TOKENS" "$PREFILL_CHUNK" | tee -a "$ROOT/config.txt"
printf 'stage_dir=%s ar_groups=%s moe_shard_layout=%s\n' "$STAGE_DIR" "$AR_GROUPS" "$MOE_SHARD_LAYOUT" | tee -a "$ROOT/config.txt"
printf 'cmg_replicate=%s\n' "$K3_CMG_REPLICATE" | tee -a "$ROOT/config.txt"
printf 'stage_limit_seconds=%s\n' "$STAGE_LIMIT_SECONDS" | tee -a "$ROOT/config.txt"
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
# 8 MiB chunks moved ~16 GB/rank in 2885 s (~5.6 MB/s) in job 49931198, far
# under what LLIO can do.  Larger chunks are the cheapest thing to try; the
# stage_timing row is what tells us whether it helped.
timeout --signal=TERM --kill-after=60 "$STAGE_LIMIT_SECONDS" \
    mpiexec -np "$NODES" -of-proc "$ROOT/stage.rank" sh -c \
        "exec '$K3/run_k3_full_stage_rank.sh' '$MODEL_DIR' '$STAGE_DIR' '$NODES' \"\${PMIX_RANK:-\${OMPI_COMM_WORLD_RANK:-\${PJM_MPI_RANK:?no MPI rank}}}\" full96 '' '$EXPERT_TP' '$MOE_SHARD_LAYOUT' '$CHUNK_MIB'"
stage_end 0

stage_begin full_short_generation
PROFILE_ARGS=()
if [ "$PROFILE" -ne 0 ]; then PROFILE_ARGS=(--profile "$ROOT/profile.txt"); fi
mpiexec -np "$NODES" -of-proc "$ROOT/run.rank" \
    "$K3/k3_full_runner" --mode full96 --stage-dir "$STAGE_DIR" \
    --topo "$ROOT/tofu_topo.txt" --prompt-ids "$ROOT/prompt.ids" \
    --output "$ROOT/output.txt" --prefill-tokens "$PREFILL_TOKENS" \
        --new-tokens "$NEW_TOKENS" --prefill-chunk "$PREFILL_CHUNK" --prefill-path batched \
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
