#!/usr/bin/env bash
# run_paint_pipeline.sh - Top-level Hunyuan3D-2.1 paint pipeline orchestrator.
#
# Phase 4.12 incremental: chains the validated stage binaries via per-stage
# .npy/.png artifacts under WORK. Stages that still depend on pyref dumps for
# their conditioning are noted; once each stage runner is folded into the
# combined runner header, those external inputs go away.
#
# Wired flow:
#   1. UNet (15-step UniPC, dual-stream) -> packed multiview latent .npy
#         test_paint_unet --stage out_loop --save-final-latent <latent.npy>
#   2. VAE decode -> per-view RGB .npy
#         test_paint_vae decode <vae.safetensors> <latent.npy> <recon.npy>
#   3. back_project + bake (GPU) + vertex inpaint -> textured OBJ + PNG
#         test_paint_back_project_e2e --inpaint --gpu-bake --out <out_dir>
#
# Current external pyref inputs:
#   * view_maps render (geometry conditioning)
#   * DINOv2 image conditioning
#   * UniPC initial-noise / scheduler state
#
# Usage: run_paint_pipeline.sh <work_dir>

set -euo pipefail

WORK="${1:-/tmp/hy3d_paint_pipeline}"
MODELS="/mnt/disk01/models/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1"
REF="/tmp/hy3d_paint_unet_ref"
DUMP="/tmp/hy3d_paint_back_project"

mkdir -p "$WORK"

UNET_W="$MODELS/unet/paint_unet_stock.safetensors"
VAE_W="$MODELS/vae/vae.safetensors"

LATENT="$WORK/final_latent.npy"
RECON="$WORK/decoded_views.npy"
OUT="$WORK/textured"

echo "==> Stage 1/3: UNet UniPC out_loop"
./test_paint_unet --stage out_loop --save-final-latent "$LATENT" "$UNET_W" "$REF/"

echo "==> Stage 2/3: VAE decode"
./test_paint_vae decode "$VAE_W" "$LATENT" "$RECON"

echo "==> Stage 3/3: back-project + bake + inpaint"
mkdir -p "$OUT"
./test_paint_back_project_e2e --inpaint --gpu-bake --out "$OUT" "$DUMP/"

echo "==> Pipeline complete. Output: $OUT/"
ls -la "$OUT/"
