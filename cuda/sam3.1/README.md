# CUDA SAM 3.1 runner

Status: **image text-prompt pipeline ported and verified** against Meta's
SAM3.1 PyTorch reference on `fujisan.jpg` / `"mountain"`.

Current reference result:

| phrase | mode | final IoU | top score | top box xyxy |
|--------|------|-----------|-----------|--------------|
| mountain | default fp16/MMA | 0.9996 | 0.8406 | [4.31, 184.27, 1691.96, 509.25] |
| mountain | `SAM3_PRECISION=fp32` | 0.9999 | 0.8410 | [4.29, 184.28, 1691.96, 509.21] |
| snow | default fp16/MMA | 0.9997 | 0.6359 | [596.34, 184.59, 1110.56, 353.12] |
| sky | default fp16/MMA | 1.0000 | 0.9801 | [1.52, 1.49, 1688.91, 452.78] |

Mountain reference: score `0.8409`, box `[4.29, 184.27, 1691.95, 509.19]`.

## Build

```sh
make -C cuda/sam3.1 all
```

Produces `test_cuda_sam3_1` plus the `verify_*` binaries.

## Verification

Generate reference dumps:

```sh
python3 ref/sam3.1/gen_image_ref.py \
  --ckpt /mnt/disk01/models/sam3.1/sam3.1_multiplex.pt \
  --image /home/syoyo/work/gemm/main/fujisan.jpg \
  --phrase mountain \
  --refdir /tmp/sam3.1_ref \
  --device cuda
```

Run verifiers:

```sh
./cuda/sam3.1/verify_final \
  --ckpt /mnt/disk01/models/sam3.1/sam3.1.model.safetensors \
  --refdir /tmp/sam3.1_ref --score 0.3

./cuda/sam3.1/verify_mask \
  --ckpt /mnt/disk01/models/sam3.1/sam3.1.model.safetensors \
  --refdir /tmp/sam3.1_ref
```

`verify_mask` compares raw `segmentation_head_out` logits when present in
the refdir. Current default fp16/MMA raw-mask checks:

| phrase | pred_masks mean_abs | semantic_seg mean_abs |
|--------|---------------------|-----------------------|
| mountain | 8.88e-02 | 6.42e-03 |
| snow | 2.79e-01 | 5.21e-03 |
| sky | 4.70e-02 | 4.99e-03 |

## Notes

- `precision=fp32` / `SAM3_PRECISION=fp32` uploads linear weights as F32
  and routes linear GEMMs through the tiled F32 path. Default remains
  fp16 weights with MMA.
- The DETR prompt path includes Meta's text-only geometry CLS token.
- Decoder layer norm mapping follows Meta's module order:
  `norm2` for self-attention, `catext_norm` for text cross-attention,
  `norm1` for image cross-attention, and `norm3` for MLP.
- Tracker/video (`tracker.model.*`) is still out of scope for this image
  runner.
