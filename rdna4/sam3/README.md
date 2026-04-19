# SAM 3 — HIP/RDNA4 runner (Phase 1)

Text-phrase-prompted instance segmentation on AMD RDNA4 (gfx1201, RX 9070 XT).
Same architecture as `cpu/sam3/`; reuses BF16 WMMA GEMM, LayerNorm and
FlashAttention kernels from `rdna4/hip_kernels_common.h`.

## Status

| Stage                              | Status      | max abs  | mean abs |
|------------------------------------|-------------|----------|----------|
| preprocess + patch_embed + pos add | Phase 1 ok  | 2.5e-3   | 1.2e-4   |
| pre-block LayerNorm                | loaded only | —        | —        |
| ViT blocks (32) + FPN              | pending     | —        | —        |
| CLIP text + DETR enc/dec           | pending     | —        | —        |
| Mask decoder + post-process        | pending     | —        | —        |

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
