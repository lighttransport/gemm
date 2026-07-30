#!/bin/bash
# One-hour TP96 real-weight decode/prefill calibration job.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=96,elapse=01:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=96"
#PJM --llio localtmp-size=1Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
RUN="$K3/run_k3_ep.sh"
NODES=96
LAYER=1
EXPERTS=16
THREADS=47
JOB_TAG=${PJM_JOBID:?PJM_JOBID is not set}
ROOT="$K3/logs/profile-96n-$JOB_TAG"
STAGE_DIR="/local/$USER/k3-profile-$JOB_TAG"
TIMING="$ROOT/job_stage_timing.tsv"
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

cat >"$ROOT/workloads.txt" <<'EOF'
decode_profile=cpp-codegen-short
decode_prompt_shape=1024 tokens
decode_generated_tokens=256
decode_prompt=Implement a lock-free bounded MPMC queue in C++20, explain memory ordering, and include tests.
prefill_profile=cpp-code-analysis
prefill_prompt_shape=8192 tokens
prefill_chunks=64,256,1024
prefill_prompt=Analyze a multi-file C++ service for ownership, concurrency, exception-safety, and performance defects; propose a patch plan.
note=The partial runner uses deterministic synthetic activations; prompt text labels the intended workload shape and is not tokenized by this runner.
EOF

stage_begin build
make -C "$K3" runner k3_moe_probe >/dev/null
stage_end 0

# Cheap distributed transport/numerical gate before shared-storage staging.
stage_begin dummy_gate
"$RUN" --mode dummy --nodes "$NODES" --layer 1 --layers 1 --tokens 8 \
    --threads 48 --heartbeat-tokens 4 --profile --ar-groups auto \
    --result-dir "$ROOT/dummy"
stage_end 0

# This phase performs the only checkpoint scan/copy. The staged TP96 slices are
# retained on rank-local storage and reused by the prefill calibration below.
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
    "$LAYER" "$EXPERTS" "$THREADS"
prefill_passes=$(grep -l 'K3 expert-TP probe: PASS' "$ROOT"/prefill/rank.* 2>/dev/null | wc -l)
if (( prefill_passes != NODES )); then
    echo "K3 TP96 prefill probe failed: $prefill_passes/$NODES ranks passed" >&2
    exit 6
fi
grep -h 'PROBE expert-tp-prefill' "$ROOT"/prefill/rank.* >"$ROOT/prefill_samples.txt"
stage_end 0

# Use the slowest rank mean for each chunk: full-model throughput is bounded by
# the critical TP rank, not by an average of independently timed ranks.
stage_begin whole_network_estimate
expert_times=$(awk '
    $1=="PROBE" && $2=="expert-tp-prefill" {
        m=""; value="";
        for(i=3;i<=NF;i++){
            if($i~/^M=/){split($i,a,"=");m=a[2]}
            if($i~/^mean_ms=/){split($i,a,"=");value=a[2]}
        }
        if(m!=""&&value!=""&&value>max[m])max[m]=value
    }
    END {if(!(64 in max)||!(256 in max)||!(1024 in max))exit 2;
         printf "%.6f,%.6f,%.6f",max[64],max[256],max[1024]}
' "$ROOT/prefill_samples.txt")
decode_expert_ms=$(awk '
    /K3_PROFILE rank_max_ms_per_layer/ {
        for(i=1;i<=NF;i++)if($i~/^expert=/){split($i,a,"=");v=a[2]}
    }
    END {if(v=="")exit 2; print v}
' "$ROOT"/decode/rank.*)
printf 'critical_expert_prefill_ms=%s\ncritical_decode_expert_ms=%s\n' \
    "$expert_times" "$decode_expert_ms" >"$ROOT/measured_calibration.txt"
PYTHONDONTWRITEBYTECODE=1 python3 "$K3/k3_sim.py" --nodes 96 \
    --contexts 1024,4096 --batches 1 --prompts 8192 --chunks 64,256,1024 \
    --expert-tp --fused-moe-ar --hierarchical-ar --q8w16-dense \
    --expert-tp-layer-ms "$decode_expert_ms" --expert-prefill-ms "$expert_times" \
    --json "$ROOT/network_estimate.json" | tee "$ROOT/network_estimate.txt"
stage_end 0

printf 'K3_PROFILE_96 status=PASS decode_results=%s prefill_results=%s estimate=%s\n' \
    "$ROOT/decode" "$ROOT/prefill_samples.txt" "$ROOT/network_estimate.txt"
