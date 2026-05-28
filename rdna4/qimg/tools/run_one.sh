#!/usr/bin/env bash
# Serial single-config FP8xFP8 quality probe. One process at a time (OOM-safe).
# Usage: run_one.sh <tag> [extra test_hip_qimg args...]
set -uo pipefail
TAG="$1"; shift
QIMG=/mnt/disk1/work/gemm/main/rdna4/qimg
REF=/mnt/disk1/work/gemm/diffusion/ref/qwen_image
DIT=/mnt/disk1/models/qwen-image/diffusion_models/qwen_image_fp8_e4m3fn.safetensors
LOG=/tmp/qimg_sweep_${TAG}.log
"$QIMG/test_hip_qimg" --generate --dit "$DIT" \
  --height 256 --width 256 --steps 20 \
  --init-bin "$REF/init_latent_256.bin" \
  --txt-bin "$REF/apple_text_256.bin" \
  --sigmas-bin "$REF/sigmas_256.bin" \
  --ref-final "$REF/final_latent_packed_256.bin" \
  --path-stats --fp8-quality-target-db 50 \
  -o /tmp/qimg_${TAG}.ppm "$@" >/dev/null 2>"$LOG"
EC=$?
PSNR=$(grep -oE "psnr_peak=[0-9.]+ dB" "$LOG" | head -1)
COS=$(grep -oE "Final packed latent: n=[0-9]+ cos=[0-9.]+" "$LOG" | grep -oE "cos=[0-9.]+")
SUMM=$(grep "GEMM path summary" "$LOG" | sed 's/.*summary://')
TRAF=$(grep "GEMM traffic summary" "$LOG" | grep -oE "fp8xfp8_wmma=[0-9.]+GB/[0-9.]+TF")
GATE=$(grep -oE "FP8 quality gate (PASS|FAIL)" "$LOG" | head -1)
DEN=$(grep -oE "Denoising done in [0-9.]+s" "$LOG")
printf "%-28s exit=%s %s %s  fp8xfp8:[%s]  %s  | %s\n" "$TAG" "$EC" "$GATE" "$PSNR" "${TRAF:-none}" "$DEN" "$SUMM"
