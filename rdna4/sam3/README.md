# SAM 3 — HIP/RDNA4 runner (Phase 1)

Text-phrase-prompted instance segmentation on AMD RDNA4 (gfx1201, RX 9070 XT).
Same architecture as `cpu/sam3/`; reuses BF16 WMMA GEMM, LayerNorm and
FlashAttention kernels from `rdna4/hip_kernels_common.h`.

## Status

| Stage                              | Status      | max abs  | mean abs |
|------------------------------------|-------------|----------|----------|
| preprocess + patch_embed + pos add | Phase 1 ok  | 2.5e-3   | 1.2e-4   |
| ViT block 0 (windowed, RoPE)       | Phase 2 ok  | 1.7e-2   | 8.2e-4   |
| ViT block 31 (32 blocks)           | Phase 2 ok* | 4.5e+1   | 6.6e-3   |
| FPN neck level 0 (288×288)         | Phase 3 ok  | 1.9e-2   | 6.3e-5   |
| FPN neck level 1 (144×144)         | Phase 3 ok* | 1.1e+0   | 5.4e-3   |
| FPN neck level 2 (72×72)           | Phase 3 ok  | 7.0e-1   | 3.2e-3   |
| FPN neck level 3 (36×36)           | Phase 3 ok  | 1.4e-1   | 2.9e-3   |
| CLIP text encoder (24 layers)      | Phase 4 ok  | 1.4e-2   | 5.3e-4   |
| DETR encoder (6 layers, hd=32)     | Phase 5 ok* | 3.4e+0   | 6.1e-3   |
| DETR decoder pred_boxes (200,4)    | Phase 6 ok  | 5.2e-2   | 1.4e-3   |
| DETR decoder presence (last)       | Phase 6 ok  | 2.3e-2   | —        |
| dot_product_scoring (6,200)        | Phase 7a ok | 1.9e-1   | 1.1e-2   |
| Mask decoder pred_masks (200,288²) | Phase 7b ok | 7.0e+0   | 6.9e-2   |
| Mask decoder semantic_seg (288²)   | Phase 7b ok | 2.7e-1   | 4.9e-3   |
| Post-process (E2E IoU vs ref)      | Phase 8 ok  | IoU=1.0000 | score 0.9728 vs 0.9730 |

\* Mean-abs healthy (6.6e-3 vs ref mean 6.6e-3); ~150 outlier positions
  diverge — same BF16 WMMA pattern we see in `rdna4/sam2` encoder.

Phase 1 = resize+ImageNet-normalize → Conv2d k=14 s=14 (3→1024) →
tile 24² pos_embed 3×3 → 72² tokens. F16 patch_proj weights; input is
the ref-dumped `input_pixel_values.npy` for bit-close verify (bypasses
stb-resize vs PIL-BILINEAR mismatch).

## Build

```bash
cd rdna4/sam3
make              # test_hip_sam3, verify_patch_embed
```

Requires HIP/HIPRTC libraries at runtime (`libamdhip64.so`, `libhiprtc.so`).
Kernels compile JIT on first run.

## Verify

```bash
make verify-patch-embed \
    CKPT=/mnt/disk1/models/sam3/sam3.model.safetensors \
    REFDIR=/tmp/sam3_ref_cat
```

Expected: `vit_embed: max_abs≈2e-3 mean_abs≈1e-4` (F16 patch_proj drift).

Generate ref dumps first via `ref/sam3/gen_image_ref.py`.
