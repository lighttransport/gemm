#!/bin/bash
# Run test_cuda_praxis on a real checkpoint with one routed-expert format variant and print the accuracy row (action_metrics.py) and, with
# LAT=n, the warm per-call latency (plain / skip-last-video / paired+skip) and the peak device memory.
#   cuda/praxis/run_expert_variant.sh <label> <model dir> <ref dir (torch fp32 oracle)> [VAR=value ...]
# e.g. LAT=3 cuda/praxis/run_expert_variant.sh P $PRAXIS_ROOT/praxis-1-droid tmp/praxis_ref/real_fp32 PRAXIS_NVFP4=1 \
#        PRAXIS_EXPERT_FMT=nvfp4a16 PRAXIS_FP8_BANKS=dn,enc3.gu,enc10.gu,enc13.gu PRAXIS_SLOT_MB=1022
# Run from the repo root. Outputs go to tmp/praxis_out/variant_<label> (tmp is a symlink to a large disk); PRAXIS_DUMP_LITE keeps them small
# (routing dumps only: router logits + top-k, usable by router_flips.py).
set -u
label=$1; model=$2; ref=$3; shift 3
out=tmp/praxis_out/variant_$label
log=tmp/praxis_out/variant_$label.log
mkdir -p tmp/praxis_out
env PRAXIS_DUMP_LITE=1 "$@" cuda/praxis/test_cuda_praxis --model "$model" --ref "$ref" --out "$out" --stream "${STREAM:-2}" --tol 0.5 \
    --repeat "${LAT:-0}" -v > "$log" 2>&1
grep -E "banks ready|run \(|^latency|^\[stream\]|^\[after-latency\]" "$log" | cut -c1-420
~/.venv/bin/python -I cuda/praxis/action_metrics.py --ours "$out" --ref "$ref" --label "$label"
