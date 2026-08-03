#!/bin/bash
# Run inside a PJM allocation. Operational settings are CLI-only; environment
# is reserved for PJM rank discovery and the OpenMP/XOS runtime.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
UTOFU="$REPO/a64fx/utofu-tests"
MODE=dummy
NODES=${PJM_MPI_PROC:-96}
TP_NODES=0
LAYERS=1
TOKENS=2
THREADS=48
KDA_THREADS=8
FUSED_THREADS=0
LAYER=1
CACHE_LOAD=
CACHE_SAVE=
EXPERTS=0-15
CHUNK_MIB=8
MODEL_DIR="$HOME/models/kimi-k3"
JOB_TAG=${PJM_JOBID:-manual-$$}
STAGE_DIR="/local/$USER/k3-runner-$JOB_TAG"
RESULT_DIR="$SCRIPT_DIR/logs/run-$JOB_TAG"
PROFILE=0
REUSE_STAGE=0
STAGE_ONLY=0
NO_FUSED_TEAM=0
MLA_CACHE_BF16=1
HEARTBEAT_TOKENS=1024
MIN_AVAILABLE_MIB=2048
AR_GROUPS=auto
COMM_ROBUST=2
COMM_ACK=0
COMM_DETERMINISTIC=0
COMM_POLL_SPINS=4
PREFETCH_MIB=0
PREFETCH_THREADS=0

usage() {
    cat >&2 <<EOF
usage: $0 [--mode dummy|real|hybrid] [--nodes N] [--tp-nodes N] [--layers N] [--tokens N]
          [--threads N] [--kda-threads N] [--fused-threads N] [--layer N] [--experts LIST] [--chunk-mib N]
          [--model-dir DIR] [--stage-dir DIR] [--result-dir DIR]
          [--cache-load PATH] [--cache-save PATH]
          [--profile] [--reuse-stage] [--stage-only] [--no-fused-team]
          [--mla-cache-bf16|--mla-cache-fp32]
          [--heartbeat-tokens N]
          [--min-available-mib N] (coordinated runtime guard, default 2048)
          [--ar-groups auto|N] (auto uses six-rank rows; 0=flat)
          [--comm-robust 1|2] (2=amortized polling, default)
          [--comm-ack 0|1] (ACK/retransmit reliability, default off)
          [--comm-deterministic 0|1] (fixed-root bit-consistent reduction)
          [--comm-poll-spins N] (power-of-two robust-2 cadence, default 4)
          [--prefetch-mib N] (overlap a routed-up weight window with MoE reduce)
          [--prefetch-threads N] (0=auto, default)
EOF
}
need_value() { if (( $# < 2 )); then echo "$0: missing value for $1" >&2; usage; exit 2; fi; }
while (( $# )); do
    case "$1" in
        --mode) need_value "$@"; MODE=$2; shift 2;;
        --nodes) need_value "$@"; NODES=$2; shift 2;;
        --tp-nodes) need_value "$@"; TP_NODES=$2; shift 2;;
        --layers) need_value "$@"; LAYERS=$2; shift 2;;
        --tokens) need_value "$@"; TOKENS=$2; shift 2;;
        --threads) need_value "$@"; THREADS=$2; shift 2;;
        --kda-threads) need_value "$@"; KDA_THREADS=$2; shift 2;;
        --fused-threads) need_value "$@"; FUSED_THREADS=$2; shift 2;;
        --layer) need_value "$@"; LAYER=$2; shift 2;;
        --cache-load) need_value "$@"; CACHE_LOAD=$2; shift 2;;
        --cache-save) need_value "$@"; CACHE_SAVE=$2; shift 2;;
        --experts) need_value "$@"; EXPERTS=$2; shift 2;;
        --chunk-mib) need_value "$@"; CHUNK_MIB=$2; shift 2;;
        --model-dir) need_value "$@"; MODEL_DIR=$2; shift 2;;
        --stage-dir) need_value "$@"; STAGE_DIR=$2; shift 2;;
        --result-dir) need_value "$@"; RESULT_DIR=$2; shift 2;;
        --profile) PROFILE=1; shift;;
        --reuse-stage) REUSE_STAGE=1; shift;;
        --stage-only) STAGE_ONLY=1; shift;;
        --no-fused-team) NO_FUSED_TEAM=1; shift;;
        --mla-cache-bf16) MLA_CACHE_BF16=1; shift;;
        --mla-cache-fp32) MLA_CACHE_BF16=0; shift;;
        --heartbeat-tokens) need_value "$@"; HEARTBEAT_TOKENS=$2; shift 2;;
        --min-available-mib) need_value "$@"; MIN_AVAILABLE_MIB=$2; shift 2;;
        --ar-groups) need_value "$@"; AR_GROUPS=$2; shift 2;;
        --comm-robust) need_value "$@"; COMM_ROBUST=$2; shift 2;;
        --comm-ack) need_value "$@"; COMM_ACK=$2; shift 2;;
        --comm-deterministic) need_value "$@"; COMM_DETERMINISTIC=$2; shift 2;;
        --comm-poll-spins) need_value "$@"; COMM_POLL_SPINS=$2; shift 2;;
        --prefetch-mib) need_value "$@"; PREFETCH_MIB=$2; shift 2;;
        --prefetch-threads) need_value "$@"; PREFETCH_THREADS=$2; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "$0: unknown argument: $1" >&2; usage; exit 2;;
    esac
done
case "$MODE" in dummy|real|hybrid) ;; *) echo "$0: --mode must be dummy, real, or hybrid" >&2; exit 2;; esac
if (( STAGE_ONLY )) && [[ "$MODE" != real && "$MODE" != hybrid ]]; then
    echo "$0: --stage-only requires --mode real or hybrid" >&2
    exit 2
fi
for value in "$NODES" "$TP_NODES" "$LAYERS" "$TOKENS" "$THREADS" "$KDA_THREADS" "$FUSED_THREADS" "$LAYER" "$CHUNK_MIB" "$HEARTBEAT_TOKENS" "$MIN_AVAILABLE_MIB" "$COMM_ROBUST" "$COMM_ACK" "$COMM_DETERMINISTIC" "$COMM_POLL_SPINS" "$PREFETCH_MIB" "$PREFETCH_THREADS"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "$0: numeric options must be integers" >&2; exit 2; }
done
(( FUSED_THREADS == 0 )) && FUSED_THREADS=$THREADS
(( TP_NODES == 0 )) && { (( NODES < 96 )) && TP_NODES=$NODES || TP_NODES=96; }
(( NODES > 0 && LAYERS > 0 && TOKENS > 0 && THREADS > 0 && THREADS <= 48 && KDA_THREADS > 0 && KDA_THREADS <= THREADS && FUSED_THREADS > 0 && FUSED_THREADS <= THREADS && CHUNK_MIB > 0 )) || {
    echo "$0: invalid numeric option range" >&2; exit 2; }
(( NODES <= 512 && TP_NODES > 0 && TP_NODES <= 96 && NODES % TP_NODES == 0 )) || {
    echo "$0: require nodes in [1,512], tp-nodes in [1,96], and nodes divisible by tp-nodes" >&2; exit 2; }
(( PREFETCH_MIB <= 32 )) || { echo "$0: --prefetch-mib must be in [0,32]" >&2; exit 2; }
[[ "$AR_GROUPS" == auto || "$AR_GROUPS" =~ ^[0-9]+$ ]] || { echo "$0: --ar-groups must be auto or an integer" >&2; exit 2; }
if [[ "$AR_GROUPS" != auto ]]; then (( AR_GROUPS == 0 || (AR_GROUPS > 1 && TP_NODES % AR_GROUPS == 0) )) || {
    echo "$0: --ar-groups must be 0 or a divisor in [2,--tp-nodes]" >&2; exit 2; }
fi
(( COMM_ROBUST == 1 || COMM_ROBUST == 2 )) || { echo "$0: --comm-robust must be 1 or 2" >&2; exit 2; }
(( COMM_ACK == 0 || COMM_ACK == 1 )) || { echo "$0: --comm-ack must be 0 or 1" >&2; exit 2; }
(( COMM_DETERMINISTIC == 0 || COMM_DETERMINISTIC == 1 )) || { echo "$0: --comm-deterministic must be 0 or 1" >&2; exit 2; }
(( COMM_POLL_SPINS > 0 && COMM_POLL_SPINS <= 1024 && (COMM_POLL_SPINS & (COMM_POLL_SPINS - 1)) == 0 )) || { echo "$0: --comm-poll-spins must be a power of two in [1,1024]" >&2; exit 2; }
(( PREFETCH_THREADS <= THREADS )) || { echo "$0: --prefetch-threads cannot exceed --threads" >&2; exit 2; }
if (( PREFETCH_MIB > 0 && THREADS > 47 )); then
    echo "$0: --prefetch-mib requires --threads <=47" >&2
    exit 2
fi
if (( NODES > 1 )); then
    if [[ "$CACHE_SAVE" == /tmp/* ]]; then
        echo "$0: warning: --cache-save path is under /tmp ($CACHE_SAVE). On multi-rank jobs, /tmp appears to be node-local in this environment and may only retain rank-local cache shards." >&2
    fi
    if [[ "$CACHE_LOAD" == /tmp/* ]]; then
        echo "$0: warning: --cache-load path is under /tmp ($CACHE_LOAD). On multi-rank jobs, /tmp appears to be node-local in this environment and may only contain rank-local cache shards." >&2
    fi
    if [[ "$CACHE_SAVE" == /local/* ]]; then
        echo "$0: warning: --cache-save path is under /local ($CACHE_SAVE). On multi-rank jobs, /local is node-local and cache shards will not be reliably shared across ranks." >&2
    fi
    if [[ "$CACHE_LOAD" == /local/* ]]; then
        echo "$0: warning: --cache-load path is under /local ($CACHE_LOAD). On multi-rank jobs, /local is node-local and cache shards will not be reliably shared across ranks." >&2
    fi
fi
if [[ -n "${PJM_MPI_PROC:-}" && "$NODES" -ne "$PJM_MPI_PROC" ]]; then
    echo "$0: --nodes $NODES differs from allocation process count $PJM_MPI_PROC" >&2
    exit 2
fi
if [[ -e "$RESULT_DIR" ]]; then echo "$0: result directory already exists: $RESULT_DIR" >&2; exit 2; fi
mkdir -p "$RESULT_DIR"
RESULT_DIR=$(cd "$RESULT_DIR" && pwd)
RUN_OUTPUT_PREFIX=${MPIEXEC_OF_PROC:-$RESULT_DIR/rank}
STAGE_OUTPUT_PREFIX=${MPIEXEC_OF_PROC:+$MPIEXEC_OF_PROC.stage}
STAGE_OUTPUT_PREFIX=${STAGE_OUTPUT_PREFIX:-$RESULT_DIR/stage}

# Keep durable wall-clock records for slow shared-storage staging and scheduler
# sizing.  The EXIT trap also records the active stage when set -e aborts it.
TIMING_FILE="$RESULT_DIR/k3_stage_timing.tsv"
TIMING_TOTAL_START=$(date +%s)
TIMING_ACTIVE=
TIMING_STAGE_START=0
printf 'stage\tstart_epoch\tend_epoch\telapsed_s\trc\n' >"$TIMING_FILE"
timing_begin() {
    TIMING_ACTIVE=$1
    TIMING_STAGE_START=$(date +%s)
    printf 'K3_STAGE_BEGIN stage=%s epoch=%s utc=%s\n' "$TIMING_ACTIVE" \
        "$TIMING_STAGE_START" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
timing_end() {
    timing_rc=$1
    timing_end_epoch=$(date +%s)
    timing_elapsed=$((timing_end_epoch - TIMING_STAGE_START))
    printf '%s\t%s\t%s\t%s\t%s\n' "$TIMING_ACTIVE" "$TIMING_STAGE_START" \
        "$timing_end_epoch" "$timing_elapsed" "$timing_rc" >>"$TIMING_FILE"
    printf 'K3_STAGE_END stage=%s elapsed_s=%s rc=%s\n' \
        "$TIMING_ACTIVE" "$timing_elapsed" "$timing_rc"
    TIMING_ACTIVE=
}
timing_on_exit() {
    timing_rc=$?
    timing_end_epoch=$(date +%s)
    if [[ -n "$TIMING_ACTIVE" ]]; then
        timing_elapsed=$((timing_end_epoch - TIMING_STAGE_START))
        printf '%s\t%s\t%s\t%s\t%s\n' "$TIMING_ACTIVE" "$TIMING_STAGE_START" \
            "$timing_end_epoch" "$timing_elapsed" "$timing_rc" >>"$TIMING_FILE"
        printf 'K3_STAGE_END stage=%s elapsed_s=%s rc=%s\n' \
            "$TIMING_ACTIVE" "$timing_elapsed" "$timing_rc"
    fi
    printf 'total\t%s\t%s\t%s\t%s\n' "$TIMING_TOTAL_START" "$timing_end_epoch" \
        "$((timing_end_epoch - TIMING_TOTAL_START))" "$timing_rc" >>"$TIMING_FILE"
    printf 'K3_STAGE_TOTAL elapsed_s=%s rc=%s timing=%s\n' \
        "$((timing_end_epoch - TIMING_TOTAL_START))" "$timing_rc" "$TIMING_FILE"
}
trap timing_on_exit EXIT

export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
if (( PREFETCH_MIB > 0 )) && [[ -z "${OMP_PLACES:-}" ]]; then
    OMP_PLACES='{12}:47:1'
fi
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND="${OMP_PROC_BIND:-close}" OMP_PLACES="${OMP_PLACES:-cores}"
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
timing_begin build
make -C "$UTOFU" tofu_topo_helper >/dev/null
make -C "$SCRIPT_DIR" runner >/dev/null
timing_end 0

cd "$RESULT_DIR"
timing_begin topology
topology_ok=0
for attempt in 1 2 3 4 5; do
    rm -f tofu_topo.txt
    if mpiexec -np "$NODES" "$UTOFU/tofu_topo_helper" &&
       [[ $(grep -vc '^#' tofu_topo.txt 2>/dev/null || true) -eq "$NODES" ]]; then
        topology_ok=1
        break
    fi
    echo "topology discovery attempt $attempt/5 failed" >&2
    sleep 2
done
(( topology_ok == 1 )) || { echo "$0: topology discovery failed" >&2; exit 3; }
timing_end 0

if [[ "$MODE" == real || "$MODE" == hybrid ]]; then
    if [[ ! -d "$MODEL_DIR" ]]; then echo "$0: model directory is missing: $MODEL_DIR" >&2; exit 4; fi
    if (( REUSE_STAGE )); then
        timing_begin stage_validation
        mpiexec -np "$NODES" /bin/sh -c '
            rank=${PMIX_RANK:?}; marker="$1/stage-rank$(printf "%03d" "$rank").status"
            tp_rank=$((rank % $3))
            expected="rank=$rank nodes=$2 tp_rank=$tp_rank tp_nodes=$3 layer=$4 experts=$5"
            test -f "$marker" && test "$(cat "$marker")" = "$expected"
        ' sh "$STAGE_DIR" "$NODES" "$TP_NODES" "$LAYER" "$EXPERTS" || {
            echo "$0: --reuse-stage validation failed: $STAGE_DIR" >&2; exit 4; }
        echo "reusing rank-local stage: $STAGE_DIR"
        timing_end 0
    else
        timing_begin weight_staging
        mpiexec -np "$NODES" -of-proc "$STAGE_OUTPUT_PREFIX" \
            "$SCRIPT_DIR/run_k3_stage_rank.sh" "$SCRIPT_DIR" "$NODES" "$TP_NODES" \
            "$MODEL_DIR" "$STAGE_DIR" "$LAYER" "$EXPERTS" "$CHUNK_MIB"
        staged=$(find "$(dirname "$STAGE_OUTPUT_PREFIX")" -maxdepth 1 \
            -name "$(basename "$STAGE_OUTPUT_PREFIX").*" -type f | wc -l)
        echo "stage launch output files: $staged/$NODES"
        timing_end 0
    fi
fi

if (( STAGE_ONLY )); then
    timing_begin stage_result_validation
    mpiexec -np "$NODES" /bin/sh -c '
        rank=${PMIX_RANK:?}; marker="$1/stage-rank$(printf "%03d" "$rank").status"
        tp_rank=$((rank % $3))
        expected="rank=$rank nodes=$2 tp_rank=$tp_rank tp_nodes=$3 layer=$4 experts=$5"
        test -f "$marker" && test "$(cat "$marker")" = "$expected"
    ' sh "$STAGE_DIR" "$NODES" "$TP_NODES" "$LAYER" "$EXPERTS"
    echo "K3_STAGE_ONLY status=PASS nodes=$NODES tp_nodes=$TP_NODES contexts=$((NODES / TP_NODES)) layer=$LAYER experts=$EXPERTS stage_dir=$STAGE_DIR"
    timing_end 0
    exit 0
fi

set +e
RUNNER_EXTRA=()
(( PROFILE )) && RUNNER_EXTRA+=(--profile)
(( NO_FUSED_TEAM )) && RUNNER_EXTRA+=(--no-fused-team)
if (( MLA_CACHE_BF16 )); then
    RUNNER_EXTRA+=(--mla-cache-bf16)
else
    RUNNER_EXTRA+=(--mla-cache-fp32)
fi
(( ${#CACHE_LOAD} > 0 )) && RUNNER_EXTRA+=(--cache-load "$CACHE_LOAD")
(( ${#CACHE_SAVE} > 0 )) && RUNNER_EXTRA+=(--cache-save "$CACHE_SAVE")
timing_begin decode_runner
mpiexec -np "$NODES" -of-proc "$RUN_OUTPUT_PREFIX" \
    "$SCRIPT_DIR/k3_ep_runner" --mode "$MODE" --nodes "$NODES" \
    --tp-nodes "$TP_NODES" \
    --layers "$LAYERS" --tokens "$TOKENS" --threads "$THREADS" --kda-threads "$KDA_THREADS" --fused-threads "$FUSED_THREADS" --layer "$LAYER" --heartbeat-tokens "$HEARTBEAT_TOKENS" --min-available-mib "$MIN_AVAILABLE_MIB" --ar-groups "$AR_GROUPS" --comm-robust "$COMM_ROBUST" --comm-ack "$COMM_ACK" --comm-deterministic "$COMM_DETERMINISTIC" --comm-poll-spins "$COMM_POLL_SPINS" --prefetch-mib "$PREFETCH_MIB" --prefetch-threads "$PREFETCH_THREADS" \
    --stage-dir "$STAGE_DIR" --status-dir "$RESULT_DIR" --topo "$RESULT_DIR/tofu_topo.txt" \
    "${RUNNER_EXTRA[@]}"
runner_rc=$?
set -e
timing_end "$runner_rc"

timing_begin result_validation
passes=$(grep -l ' state=pass ' "$RESULT_DIR"/k3_rank*.status 2>/dev/null | wc -l || true)
grep -hE 'K3_RUN|K3_PROGRESS|K3_RESULT|K3_HEALTH|K3_PROFILE|FATAL|timeout|failed' "$RUN_OUTPUT_PREFIX".* 2>/dev/null || true
echo "K3 distributed result: rc=$runner_rc pass_markers=$passes/$NODES results=$RESULT_DIR"
validation_rc=0
(( runner_rc == 0 && passes == NODES )) || validation_rc=5
timing_end "$validation_rc"

# Rank-local storage is job-scoped and is wiped by the scheduler. Deliberately
# leave it untouched here: automatic recursive cleanup of a caller-supplied
# --stage-dir is unsafe, and retaining it helps diagnose a failed run.
(( validation_rc == 0 )) || exit "$validation_rc"
