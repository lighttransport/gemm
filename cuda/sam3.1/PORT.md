# SAM 3 → SAM 3.1 Porting Notes

This directory is a **clone** of `cuda/sam3/` with identifiers renamed
(`cuda_sam3_` → `cuda_sam3_1_`) and the safetensors prefix changed from
`detector_model.` to `detector.`. It compiles and reaches the loader, but
every leaf key path under the prefix still refers to sam3's naming scheme
— loads fail with `missing …` messages. This document enumerates the work
required to make it run end-to-end.

## Smoke test
```
./verify_patch_embed --ckpt /mnt/nvme02/models/sam3.1/sam3.1.model.safetensors --refdir /tmp/x
# → sam3.1: missing vision_encoder.backbone.embeddings.patch_embeddings.projection.weight
# ... (each missing key identifies a rename task)
```

## 1. ViT backbone (mechanical rename + one kernel change)

| sam3 (current code)                                                    | sam3.1                                                     |
|------------------------------------------------------------------------|------------------------------------------------------------|
| `vision_encoder.backbone.embeddings.patch_embeddings.projection.{w,b}` | `backbone.vision_backbone.trunk.patch_embed.proj.{w,b}`    |
| `vision_encoder.backbone.embeddings.position_embeddings`               | *(RoPE — no learned pos embed; see §4)*                    |
| `vision_encoder.backbone.layers.N.attention.q_proj.{w,b}`              | `backbone.vision_backbone.trunk.blocks.N.attn.qkv.{w,b}` ← **fused** |
| `vision_encoder.backbone.layers.N.attention.k_proj.{w,b}`              | *(fused into qkv)*                                         |
| `vision_encoder.backbone.layers.N.attention.v_proj.{w,b}`              | *(fused into qkv)*                                         |
| `vision_encoder.backbone.layers.N.attention.out_proj.{w,b}`            | `backbone.vision_backbone.trunk.blocks.N.attn.proj.{w,b}`  |
| `vision_encoder.backbone.layers.N.layer_norm1.{w,b}`                   | `backbone.vision_backbone.trunk.blocks.N.norm1.{w,b}`      |
| `vision_encoder.backbone.layers.N.mlp.fc{1,2}.{w,b}`                   | `backbone.vision_backbone.trunk.blocks.N.mlp.fc{1,2}.{w,b}`|
| `vision_encoder.backbone.layer_norm.{w,b}`                             | `backbone.vision_backbone.trunk.norm.{w,b}`                |
| *(relative position bias tables)*                                      | `backbone.vision_backbone.trunk.blocks.N.attn.freqs_cis` (complex64) |

**Actions in `cuda_sam3_1_runner.c`:**
- `fuse_qkv()` / `fuse_qkv_bias()`: replace the 3-tensor fuse with a direct
  upload of `attn.qkv.{weight,bias}` (shape already `(3*D, D)` / `(3*D,)`).
- `load_block()`: rename all leaf paths per the table above.
- Constants match: `S3_DIM=1024`, `S3_MLP=4736`, `S3_PATCH=14`, 32 blocks ✓.

## 2. ViT attention: RoPE replaces relative position bias (kernel change)

sam3 uses axial relative-position-bias tables baked into attention.
sam3.1 uses 2D RoPE with **pre-computed complex-valued `freqs_cis`**
of shape `(grid*grid, head_dim/2)`. The attention kernels in
`cuda_sam3_1_kernels.h` that currently add RPB must be replaced with a
RoPE rotation applied to Q and K before the softmax. This is the largest
single change.

## 3. Text encoder: OpenAI-CLIP layout (not HF-CLIP)

sam3 used the HF-CLIP convention; sam3.1 uses the reference OpenAI CLIP:

| sam3 (HF-CLIP)                                              | sam3.1 (OpenAI-CLIP)                                                  |
|-------------------------------------------------------------|-----------------------------------------------------------------------|
| `text_encoder.text_model.embeddings.token_embedding.weight` | `backbone.language_backbone.encoder.token_embedding.weight`           |
| `text_encoder.text_model.embeddings.position_embedding.weight` | `backbone.language_backbone.encoder.positional_embedding` (raw tensor) |
| `…encoder.layers.N.self_attn.{q,k,v}_proj.{w,b}` (separate) | `…transformer.resblocks.N.attn.in_proj_{weight,bias}` (fused 3D)      |
| `…encoder.layers.N.self_attn.out_proj.{w,b}`                | `…transformer.resblocks.N.attn.out_proj.{w,b}`                        |
| `…encoder.layers.N.layer_norm{1,2}.{w,b}`                   | `…transformer.resblocks.N.ln_{1,2}.{w,b}`                             |
| `…encoder.layers.N.mlp.fc{1,2}.{w,b}`                       | `…transformer.resblocks.N.mlp.{c_fc,c_proj}.{w,b}`                    |
| `text_encoder.text_model.final_layer_norm.{w,b}`            | `backbone.language_backbone.encoder.ln_final.{w,b}`                   |
| *(none)*                                                    | `backbone.language_backbone.encoder.text_projection` (1024 → 512)     |

OpenAI-CLIP is **pre-norm** (norm → attn → +res → norm → mlp → +res) while
the sam3 path assumes post-norm-style ordering in the forward sequence.
Audit `cuda_sam3_1_run_text_encoder()` accordingly.

The `text_projection` produces a 512-D embedding — sam3 currently uses the
1024-D ln_final output directly for DETR cross-attention. Check whether
sam3.1 expects the projected 512-D vector for downstream dot-product
scoring (likely yes).

`fuse_qkv_text()` should be replaced with a direct upload of
`attn.in_proj_{weight,bias}`.

BPE vocab size and CLIP tokenization are the same; reuse
`cpu/sam3/sam3_clip_bpe.{c,h}` unchanged.

## 4. DETR / segmentation head / dot-product scoring

sam3.1 keys under `detector.transformer.*`, `detector.segmentation_head.*`,
`detector.dot_prod_scoring.*`, `detector.geometry_encoder.*`. Diff the
sam3 box-head / RPB embed / mask-decoder key list against the sam3.1
list in `sam3.1_keys.txt` and re-wire. A number of sam3 heads
(`box_rpb_embed_x/y`, `box_head.layer{1,2,3}`) are **absent** from sam3.1
— the detection box head appears to have been consolidated. This is not
a pure rename and will need architecture inspection against the official
sam3.1 model definition.

## 5. Tracker (entirely new — not in sam3)

`tracker.model.*` — 457 tensors implementing the video object tracker
(self-attn + cross-attn + image-cross-attn blocks with D=256, MLP=2048,
plus `maskmem_tpos_enc`, memory-bank embeddings, `interactive_mask_downsample`,
etc.). This has no sam3 analog. Build out as new kernels + runner
entrypoint. Skip for image-only inference.

## Suggested order

1. Fix the ViT loader (rename paths + qkv fusion). Runs end-to-end but
   produces wrong logits until §2 lands.
2. Replace RPB with RoPE in the ViT attention kernel. Verify against
   a PyTorch reference dump.
3. Port the text encoder (loader rename + pre-norm fix + text_projection).
4. Port DETR / segmentation / scoring heads.
5. (Optional) tracker module.
