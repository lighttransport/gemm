#!/bin/bash
# One-hour TP72 non-contiguous partial-real decode/prefill calibration job.
# A scalar node request deliberately avoids a torus-shape placement constraint.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=72,elapse=01:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=72"
#PJM --llio localtmp-size=1Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
RUN="$K3/run_k3_ep.sh"
NODES=72
LAYER=1
THREADS=47
JOB_TAG=${PJM_JOBID:?PJM_JOBID is not set}
ROOT="$K3/logs/probe-72n-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-probe-72n-$JOB_TAG"
TIMING="$ROOT/job_stage_timing.tsv"
SUMMARY="$ROOT/summary.txt"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"

if [[ -e "$ROOT" ]]; then
    echo "$0: result root already exists: $ROOT" >&2
    exit 2
fi
mkdir -p "$ROOT"
printf 'stage\tstart_epoch\tend_epoch\telapsed_s\trc\n' >"$TIMING"
job_start=$(date +%s)
active_stage=
stage_start=0
stage_begin() {
    active_stage=$1
    stage_start=$(date +%s)
    printf 'K3_JOB_STAGE_BEGIN stage=%s epoch=%s utc=%s\n' "$active_stage" \
        "$stage_start" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
stage_end() {
    stage_rc=$1
    stage_stop=$(date +%s)
    stage_elapsed=$((stage_stop - stage_start))
    printf '%s\t%s\t%s\t%s\t%s\n' "$active_stage" "$stage_start" \
        "$stage_stop" "$stage_elapsed" "$stage_rc" >>"$TIMING"
    printf 'K3_JOB_STAGE_END stage=%s elapsed_s=%s rc=%s\n' \
        "$active_stage" "$stage_elapsed" "$stage_rc"
    active_stage=
}
on_exit() {
    job_rc=$?
    job_stop=$(date +%s)
    if [[ -n "$active_stage" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$active_stage" "$stage_start" "$job_stop" \
            "$((job_stop - stage_start))" "$job_rc" >>"$TIMING"
    fi
    printf 'total\t%s\t%s\t%s\t%s\n' "$job_start" "$job_stop" \
        "$((job_stop - job_start))" "$job_rc" >>"$TIMING"
    printf 'K3_JOB_TOTAL elapsed_s=%s rc=%s results=%s\n' \
        "$((job_stop - job_start))" "$job_rc" "$ROOT"
}
trap on_exit EXIT

cat >"$ROOT/workload.txt" <<'EOF'
probe=TP72 non-contiguous partial-real decode and prefill calibration
dummy_tokens=8
decode_tokens=256
layer=1
experts=0-15
placement=scalar node=72 request; no torus shape constraint
tp_layout=ranks 0-23 own 64 expert channels and two attention heads; ranks 24-71 own 32 channels and one head
scope=The runner uses real routed-expert slices with deterministic synthetic activations; it is not an end-to-end K3 generation run.
memory=The modeled full K3 stack is about 28.9 GiB on the fullest TP72 rank and exceeds the strict 27 GiB target.
prefill=Real M=64,256,1024 expert-prefill calibration covers both 64- and 32-channel TP72 ranks.
EOF

stage_begin build
make -C "$K3" runner >/dev/null
stage_end 0

# Gate topology, uTofu communication, numerics, and teardown before checkpoint I/O.
stage_begin dummy_gate
"$RUN" --mode dummy --nodes "$NODES" --layer "$LAYER" --layers 1 --tokens 8 \
    --threads 48 --heartbeat-tokens 4 --profile --ar-groups auto \
    --result-dir "$ROOT/dummy"
stage_end 0

# Stage the bounded ragged TP72 slices once and measure a sustained partial-real run.
stage_begin real_weight_decode
"$RUN" --mode real --nodes "$NODES" --layer "$LAYER" --experts 0-15 \
    --layers 1 --tokens 256 --threads "$THREADS" --fused-threads "$THREADS" \
    --heartbeat-tokens 64 --profile --ar-groups auto --prefetch-mib 16 \
    --prefetch-threads 32 --model-dir "$HOME/models/kimi-k3" \
    --stage-dir "$STAGE_DIR" --result-dir "$ROOT/decode"
stage_end 0

stage_begin real_weight_prefill
mkdir -p "$ROOT/prefill"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
mpiexec -np "$NODES" -of-proc "$ROOT/prefill/rank" \
    "$K3/run_k3_prefill_staged_rank.sh" "$K3" "$STAGE_DIR" "$NODES" \
    "$NODES" "$LAYER" 16 "$THREADS"
prefill_passes=$(grep -l 'K3 expert-TP probe: PASS' "$ROOT"/prefill/rank.* 2>/dev/null | wc -l || true)
(( prefill_passes == NODES )) || {
    echo "K3 TP72 prefill validation failed: $prefill_passes/$NODES ranks passed" >&2
    exit 8
}
grep -h 'PROBE expert-tp-prefill' "$ROOT"/prefill/rank.* >"$ROOT/prefill_samples.txt"
stage_end 0

stage_begin summarize
{
    cat "$ROOT/workload.txt"
    printf '\n[dummy]\n'
    grep -hE 'K3_RUN|K3_PROGRESS|K3_RESULT|K3_HEALTH|K3_PROFILE' \
        "$ROOT"/dummy/rank.* || true
    printf '\n[real_weight_decode]\n'
    grep -hE 'K3_RUN|K3_PROGRESS|K3_RESULT|K3_HEALTH|K3_PROFILE' \
        "$ROOT"/decode/rank.* || true
    printf '\n[real_weight_prefill]\n'
    cat "$ROOT/prefill_samples.txt"
} >"$SUMMARY"

dummy_passes=$(grep -l ' state=pass ' "$ROOT"/dummy/k3_rank*.status 2>/dev/null | wc -l || true)
decode_passes=$(grep -l ' state=pass ' "$ROOT"/decode/k3_rank*.status 2>/dev/null | wc -l || true)
(( dummy_passes == NODES )) || {
    echo "K3 TP72 dummy validation failed: $dummy_passes/$NODES ranks passed" >&2
    exit 6
}
(( decode_passes == NODES )) || {
    echo "K3 TP72 decode validation failed: $decode_passes/$NODES ranks passed" >&2
    exit 7
}
grep -hqE 'K3_RUN mode=real nodes=72 .*allreduce=hierarchical ar_groups=12 ' \
    "$ROOT"/decode/rank.* || {
    echo "K3 TP72 decode did not report the expected 12x6 hierarchy" >&2
    exit 9
}
grep -hqE 'K3_RESULT status=PASS reason=complete tokens_completed=256 ' \
    "$ROOT"/decode/rank.* || {
    echo "K3 TP72 decode did not report 256 completed tokens" >&2
    exit 10
}
grep -hq 'K3_PROFILE rank_max_ms_per_layer' "$ROOT"/decode/rank.* || {
    echo "K3 TP72 decode profile summary is missing" >&2
    exit 11
}
cat "$SUMMARY"
stage_end 0

printf 'K3_PROBE_72 status=PASS dummy=%s decode=%s prefill=%s summary=%s\n' \
    "$ROOT/dummy" "$ROOT/decode" "$ROOT/prefill_samples.txt" "$SUMMARY"
