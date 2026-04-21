# SAM 3 — CUDA runner (NVRTC)

Text-phrase-prompted instance segmentation for NVIDIA CUDA using runtime
kernel compilation (no `nvcc` dependency). This CUDA port mirrors the staged
pipeline and verification flow from `cpu/sam3` and `rdna4/sam3`.

## Status

End-to-end pipeline complete — IoU 1.0000 vs PyTorch reference on the cat test image.

| Stage                              | Status      | max abs  | mean abs |
|------------------------------------|-------------|----------|----------|
| preprocess + patch_embed + pos add | Phase 1 ok  | 2.5e-3   | 1.2e-4   |
| ViT block 0 (windowed, RoPE)       | Phase 2 ok  | 1.7e-2   | 8.2e-4   |
| ViT block 31 (32 blocks)           | Phase 2 ok* | 2.3e+1   | 5.2e-3   |
| FPN neck level 0 (288×288)         | Phase 3 ok  | 7.7e-3   | 5.8e-5   |
| FPN neck level 1 (144×144)         | Phase 3 ok* | 4.3e-1   | 4.4e-3   |
| FPN neck level 2 (72×72)           | Phase 3 ok  | 3.1e-1   | 2.7e-3   |
| FPN neck level 3 (36×36)           | Phase 3 ok  | 4.2e-2   | 2.2e-3   |
| CLIP text encoder (24 layers)      | Phase 4 ok  | 1.4e-2   | 5.7e-4   |
| DETR encoder (6 layers, hd=32)     | Phase 5 ok* | 1.0e+0   | 5.7e-3   |
| DETR decoder pred_boxes (200,4)    | Phase 6 ok  | 2.3e-1   | 5.0e-3   |
| DETR decoder presence (last)       | Phase 6 ok  | 1.5e-1   | —        |
| dot_product_scoring (6,200)        | Phase 7a ok | 1.9e+0   | 4.9e-2   |
| Mask decoder pred_masks (200,288²) | Phase 7b ok | 8.0e+1   | 3.5e-1   |
| Mask decoder semantic_seg (288²)   | Phase 7b ok | 3.1e-1   | 6.9e-3   |
| Post-process (E2E IoU vs ref)      | Phase 8 ok  | IoU=1.0000 | score 0.0136 vs 0.0136 |

\* Mean-abs healthy (6.6e-3 vs ref mean 6.6e-3); ~150 outlier positions
  diverge — same BF16 WMMA pattern we see in `rdna4/sam2` encoder.

Phase 1 = resize+ImageNet-normalize → Conv2d k=14 s=14 (3→1024) →
tile 24² pos_embed 3×3 → 72² tokens. F16 patch_proj weights; input is
the ref-dumped `input_pixel_values.npy` for bit-close verify (bypasses
stb-resize vs PIL-BILINEAR mismatch).

## Build

```bash
cd cuda/sam3
make              # builds test_cuda_sam3 + all verifiers
```

Builds: `test_cuda_sam3` and 9 stage verifiers — `verify_patch_embed`,
`verify_vit`, `verify_fpn`, `verify_text`, `verify_detr`, `verify_detr_dec`,
`verify_dot_score`, `verify_mask`, `verify_final`.

Requires CUDA/NVRTC libraries at runtime (`libcuda.so`, `libnvrtc.so`).
Kernels compile JIT on first run.

### GEMM dispatch (tensor cores + older-GPU fallback)

`r_gemm` in `cuda_sam3_runner.c` picks one of two kernels from
`cuda/cuda_kernels_common.h`:

- `gemm_f16_f32` — MMA `m16n8k16.f32.f16.f16.f32` tensor-core kernel.
  Requires sm_80+ (Ampere/Ada/Hopper/Blackwell). Used for large projections
  with `n_in % 16 == 0` and `n_out >= 8` (ViT QKV/proj/MLP, FPN, text, DETR
  enc/dec self-/cross-attn, mask decoder attention).
- `gemm_tiled_f16_f32` — shared-memory tiled GEMM, 16×16 tiles, no tensor
  cores. Used on sm<80, whenever `SAM3_GEMM=tiled` is set, or for small/odd
  shapes (box delta head `n_out=4`, presence head `n_out=1`, RPH/RPB MLPs).

Compute capability is detected at `cuda_sam3_create` time via
`cuDeviceGetAttribute`; `ctx->use_mma` is `1` only when `sm >= 80` and the
env var does not force tiled. The runner logs
`sam3: sm_<N> -> gemm=<mma|tiled>` at startup.

Both paths are verified to produce IoU=1.0000 vs the PyTorch reference on
sm_120 (`./verify_final` and `SAM3_GEMM=tiled ./verify_final`). The MMA
path is ~10× looser in max_abs (F16 multiply + F32 accumulate vs the tiled
kernel's F32 multiply + F32 accumulate) but the drift does not change the
final segmentation mask.

The MMA kernel has a Blackwell-specific `a1`/`a2` fragment swap under
`#if __CUDA_ARCH__ >= 1200`. Its correctness on sm_120 is guarded by
`verify_mma_gemm` — a self-contained microtest (see `cuda/sam3/Makefile`).

### CPU reference

A matching CPU port lives at [`../../cpu/sam3/`](../../cpu/sam3/). It is
built separately (`cd cpu/sam3 && make`) and emits the same per-stage
verifiers. Use it when no GPU is available or to triangulate numerics.

The MMA kernel has a Blackwell-specific `a1`/`a2` fragment swap under
`#if __CUDA_ARCH__ >= 1200`. Its correctness on sm_120 is guarded by
`verify_mma_gemm` — a self-contained microtest that compares MMA output
against a CPU reference and the tiled kernel across sam3 dims (1024, 3072,
4736). Expect `cpu vs mma` max_abs ≈ 1e-4 / mean_abs ≈ 1e-5 — F16 multiply
precision, ~10× looser than the tiled path (which up-converts to F32 before
multiplying). Run with `./verify_mma_gemm` (no args, no model needed).

## Verify

Generate ref dumps first via `ref/sam3/gen_image_ref.py`, then:

```bash
# defaults: CKPT=/path/to/sam3.model.safetensors  REFDIR=/tmp/sam3_ref_cat
make verify-patch-embed   # vit_embed: max_abs≈2e-3 mean_abs≈1e-4
make verify-vit-block0    # block 0
make verify-vit-block31   # block 31 (last)
make verify-vit-final     # full 32-block stack
make verify-fpn           # 4-level neck
make verify-text          # CLIP text encoder
make verify-detr          # DETR encoder (6 layers)
make verify-detr-dec      # DETR decoder (pred_boxes + presence)
```

`verify_dot_score`, `verify_mask`, `verify_final` binaries are built by
`make` and runnable directly against `--ckpt` / `--refdir` flags.
