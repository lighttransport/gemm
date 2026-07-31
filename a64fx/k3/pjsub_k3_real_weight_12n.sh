#!/bin/bash
# Bounded real-weight K3 runner smoke test.
# Each rank stages 16 actual MXFP4 experts, tensor-parallel sliced across 12
# ranks, into its node-local /local filesystem.  The runner then loads all six
# tensors for every selected expert and executes the distributed path.
# Layer 1 exercises the KDA+MoE branch; layer 3 exercises MLA+MoE.
#PJM -g hp250467
#PJM -L "rscgrp=small,node=12,elapse=01:00:00"
#PJM -L "freq=2000,eco_state=0,retention_state=0"
#PJM --mpi "proc=12"
#PJM --llio localtmp-size=4Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM -j
set -euo pipefail

REPO=/vol0006/mdt0/data/hp250467/work/gemm/k3
K3="$REPO/a64fx/k3"
MODEL_DIR=${K3_MODEL_DIR:-$HOME/models/kimi-k3}
NODES=12
TP_NODES=12
THREADS=${K3_THREADS:-48}
JOB_TAG=${PJM_JOBID:-manual-$$}
ROOT="$K3/logs/real-weight-12n-$JOB_TAG"
STAGE_ROOT="/local/$USER/k3-real-weight-12n-$JOB_TAG"
export PATH="/opt/local/mpiexec:/opt/FJSVxtclanga/tcsds-1.2.43/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES=cores
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

[[ ! -e "$ROOT" ]] || { echo "$0: result root exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"

make -C "$K3" runner >/dev/null

run_layer() {
    local layer=$1
    local label=$2
    local stage="$STAGE_ROOT/layer$(printf '%02d' "$layer")"
    local result="$ROOT/$label"
    local launch_log="$ROOT/$label.launch.log"
    echo "K3_REAL_WEIGHT_BEGIN layer=$layer label=$label stage=$stage result=$result"
    "$K3/run_k3_ep.sh" \
        --mode real --nodes "$NODES" --tp-nodes "$TP_NODES" \
        --layer "$layer" --layers 1 --tokens 128 \
        --threads "$THREADS" --kda-threads 8 --fused-threads "$THREADS" \
        --experts 0-15 --model-dir "$MODEL_DIR" --stage-dir "$stage" \
        --result-dir "$result" --profile --ar-groups auto \
        --heartbeat-tokens 32 --min-available-mib 2048 \
        2>&1 | tee "$launch_log"
    mv "$launch_log" "$result/run.log"
    grep -q "K3 distributed result: rc=0 pass_markers=$NODES/$NODES" "$result/run.log"
    grep -q "K3_RESULT status=PASS" "$result"/rank.* 2>/dev/null
    echo "K3_REAL_WEIGHT_END layer=$layer label=$label status=PASS"
}

run_layer 1 kda_layer01
run_layer 3 mla_layer03

printf 'K3_REAL_WEIGHT_12 status=PASS layers=1,3 experts=0-15 nodes=%d stage_root=%s results=%s\n' \
    "$NODES" "$STAGE_ROOT" "$ROOT"
