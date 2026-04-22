# SAM 3 → SAM 3.1 Porting Notes

This directory is a **clone** of `cuda/sam3/` with identifiers renamed
(`cuda_sam3_` → `cuda_sam3_1_`) and the safetensors prefix changed from
`detector_model.` to `detector.`.

**Status (2026-04-23):** full pipeline end-to-end verified vs PyTorch
reference on fujisan.jpg / "mountain":

| stage               | metric                               |
|---------------------|--------------------------------------|
| final detection count | 1 (matches ref)                    |
| top score           | 0.877 (ref 0.841)                    |
| top box (xyxy px)   | [6.4, 185.6, 1690, 511.1] vs ref [4.3, 184.3, 1692, 509.2] |
| mask IoU            | **0.9801**                           |

Per-stage drift vs PyTorch:
- detr_dec_out last-layer (200,256): max_abs=4.20, mean_abs=0.31.
- dot_score per layer: max_abs≈1.2–4.0, mean_abs≈0.23–0.90.
- Drift is consistent with F16 weight-quantization noise compounding
  through ViT → FPN → DETR enc → DETR dec without any identified logic bugs.

Earlier ViT-only drift table:

| stage   | max_abs | mean_abs | ref max | relative |
|---------|---------|----------|---------|----------|
| patch_embed | 2.8e-3 | 1.2e-4 | — | F16 noise |
| block0  | 2.0e-2 | 9.5e-4 | 19.4 | 0.1% |
| block15 | 2.0    | 2.7e-3 | 151.8 | 1.3% |
| block31 | 5.1    | 5.3e-3 | 223.1 | 2.3% |

**Drift source pinned (2026-04-23):** end-to-end drift (DETR enc 2.83,
DETR dec 4.20, mask IoU 0.98 vs ref 1.0) is F16 weight-quantization in
the ViT + downstream MMA GEMM path, compounding linearly. Isolation:

- A numpy F32 replica of `convs.2.conv_{1x1,3x3}` applied to ref
  `trunk_out.npy` matches ref `convs2.npy` to `max_abs=1.0e-3` — the
  FPN conv ops themselves are correct.
- Our ViT block31 output has `max_abs=5.1` drift on 1024 channels;
  the FPN level-2 1x1 conv (Ci=1024) sums these and lifts drift to
  `max_abs=0.48` regardless of the FPN conv precision.
- Confirmed by switching FPN `conv_1x1`/`conv_3x3` weights to F32
  (new `conv2d_{1x1,3x3_pad1}_f32` kernels at
  `cuda_sam3_1_kernels.h`): fpn2 drift unchanged at 0.48. Change kept
  as ~4 MB hardening since it's correct regardless of ViT path.

**Deeper root cause (2026-04-23):** `verify_mma_gemm` shows MMA error is
`~1e-4` per GEMM vs `~3e-7` for the shared-memory tiled path (which
dequantizes F16→F32 before F32 accumulation). MMA's F16×F16→F32 native
accumulate loses precision on each GEMM; across 32 ViT blocks × 4 GEMMs
this compounds into the block31 max_abs=5.1 drift.

**Precision-for-speed knob:** the existing `SAM3_GEMM=tiled` env var now
usefully routes every GEMM (ViT + text + DETR + mask decoder) through
the tiled kernel. Measured on fujisan.jpg/"mountain":

| metric          | SAM3_GEMM=mma (default) | SAM3_GEMM=tiled |
|-----------------|-------------------------|-----------------|
| fpn1 max_abs    | 0.351                   | 0.290           |
| fpn2 max_abs    | 0.478                   | 0.306           |
| detr_enc max_abs| 2.82                    | 2.82            |
| final IoU       | 0.9801                  | 0.9802          |
| wall-clock      | 4.5 s                   | 6.9 s (+52%)    |

IoU is already saturated — DETR encoder drift is dominated by 6 layers
of F16 weight rounding even after feeding cleaner inputs, and the
remaining drift does not move detection/box/mask outputs. The tiled
path is kept as the precision-critical escape hatch; MMA remains
default for latency-sensitive use. Eliminating the F16-weight floor
would require uploading ViT/DETR weights as F32 (~1.5 GB extra) and
a new F32 GEMM kernel — deferred.

Earlier ViT-only drift table:


get here: (a) `trunk.pos_embed` load (skipping the cls_token row) was
missing — sam3.1 uses RoPE AND keeps a learned abs pos_embed applied
before ln_pre. (b) RoPE cos/sin tables were built after the
`SAM31_VISION_ONLY` escape hatch, so vision-only runs dereferenced
NULL tables and faulted inside `rope_apply_qk_f32`.

Text encoder loader retargeted to OpenAI-CLIP keys
(`backbone.language_backbone.encoder.transformer.resblocks.N.{ln_1,ln_2,
attn.in_proj_{weight,bias},attn.out_proj,mlp.{c_fc,c_proj}}` + `ln_final` +
`token_embedding` + `positional_embedding` + `text_projection`). Fused
`attn.in_proj_weight` is loaded directly (no per-head fuse step). Existing
forward path is already pre-norm, which matches OpenAI-CLIP.

**Verified 2026-04-23**: `verify_text` against `text_ln_final.npy`
(ln_final output, pre-`text_projection`). Valid tokens (t ≤ EOS):
`max_abs=1.57e-02`, `mean_abs=6.25e-04` — F16 quant noise across 24
layers. Padding-position stats are higher (`max_abs=5.87e-02`) because
reference masks them out but our forward still computes; this is
harmless since DETR/scoring also masks via `text_mask`.

**Wired 2026-04-23**: DETR-path text projection now uses
`backbone.language_backbone.resizer.{weight,bias}` (1024 → 256 with bias),
replacing the incorrect `text_projection` (1024 → 512) that was previously
loaded. `text_projection` is for similarity/alignment outputs and is NOT
on the DETR path.

DETR encoder forward (verified 2026-04-23):
`verify_detr` vs `detr_enc_memory.npy` (encoder output, 5184×1×256):
`max_abs=2.83, mean_abs=1.44e-01`. Per-layer progression (override path,
ref FPN2 + ref text_ln_final as inputs): L1=0.71, L2=0.76, L3=0.91,
L6=2.82 — consistent with F16 weight-quantization noise compounding
across 6 layers. Full pipeline matches override (2.83), so the encoder
is numerically clean; residual is upstream drift absorbed linearly.

Bug fixed during isolation: `cuda_sam3_1_debug_override_detr_inputs`
bypassed `set_input_ids`, leaving `d_text_mask_i32` uninitialized. With
an invalid key mask the cross-attn attended to the 29 pad tokens,
inflating L1 error from 0.71 → 3.24. `verify_detr --use-ref-inputs`
now calls `set_input_ids` with the ref `input_input_ids.npy` before
invoking the override, so the pad mask is valid.

Verified along the way:
- Encoder layer config matches: `pre_norm=True`, `pos_enc_at_attn=True`,
  `pos_enc_at_cross_attn_{queries,keys}=False`, act=relu, 6 layers,
  `add_pooled_text_to_img_feat=False`.
- Sinusoidal 2-D pos embed matches ref `detr_enc_pos_embed.npy` to
  float precision.
- `text_mask` polarity (1=valid, 0=pad) matches `mha_hd32_f32` kernel.
- Text-path projection via `language_backbone.resizer` (1024→256+bias),
  not `text_projection` (1024→512, similarity-only).

Residual to-dos (not blocking DETR encoder):
- FPN level 2 drifts `max_abs=0.48, mean_abs=2.8e-03` vs `convs2.npy` —
  probably F16 conv accumulator; investigate if it bleeds into downstream
  F16 stages, else accept as noise.

DETR encoder + decoder + `dot_prod_scoring` + `segmentation_head` loaders
retargeted to sam3.1 key layout. Full `cuda_sam3_1_create` now completes
without missing-key errors. Key-level rename map:

- `detr_encoder.layers.N.{layer_norm{1..3}, self_attn.{q,k,v,o}_proj, cross_attn.{q,k,v,o}_proj, mlp.{fc1,fc2}}`
  → `transformer.encoder.layers.N.{norm{1..3}, self_attn.{in_proj,out_proj}, cross_attn_image.{in_proj,out_proj}, linear{1,2}}`
- `detr_decoder.{query_embed, reference_points, presence_token, output_layer_norm, presence_layer_norm, box_head.layer{1..3}, presence_head.layer{1..3}, ref_point_head.layer{1,2}, box_rpb_embed_{x,y}.layer{1,2}, layers.N.*}`
  → `transformer.decoder.{query_embed, reference_points, presence_token, norm, presence_token_out_norm, bbox_embed.layers.{0..2}, presence_token_head.layers.{0..2}, ref_point_head.layers.{0,1}, boxRPB_embed_{x,y}.layers.{0,1}, layers.N.*}`
- `detr_decoder.layers.N.{self_attn, text_cross_attn, vision_cross_attn}.{q,k,v,o}_proj`
  → `transformer.decoder.layers.N.{self_attn, ca_text, cross_attn}.{in_proj_weight, in_proj_bias, out_proj.*}` (fused, split at load via new `split_fused_qkv()`)
- `detr_decoder.layers.N.{self_attn, text_cross_attn, vision_cross_attn, mlp}_layer_norm`
  → `transformer.decoder.layers.N.{norm1, catext_norm, norm2, norm3}`
- `dot_product_scoring.{text_mlp.layer{1,2}, text_mlp_out_norm, text_proj, query_proj}`
  → `dot_prod_scoring.{prompt_mlp.layers.{0,1}, prompt_mlp.out_norm, prompt_proj, hs_proj}`
- `mask_decoder.{prompt_cross_attn.{q,k,v,o}_proj, prompt_cross_attn_norm, pixel_decoder.*, mask_embedder.layers, instance_projection, semantic_projection}`
  → `segmentation_head.{cross_attend_prompt.{in_proj,out_proj}, cross_attn_norm, pixel_decoder.*, mask_predictor.mask_embed.layers, instance_seg_head, semantic_seg_head}`

**Loader-only progress** — forward kernels are NOT retargeted. Known
latent mismatches that will surface during verification:

- `text_projection` is now `(1024, 512)` with no bias; sam3's DETR
  cross-attention pool at line 1560 still gemms into `S3_DETR_DIM=256`.
- `segmentation_head.pixel_decoder` has 3 `conv_layers`/`norms` in
  sam3.1 but the sam3 forward path only uses indices 0/1.
- The `interactive_convs.*` / `propagation_convs.*` vision-backbone
  conv stacks are present in the checkpoint but not loaded yet — they
  feed prompt-driven forward paths which aren't exercised by image-only
  text-prompt inference.
- Tracker (`tracker.model.*`) is entirely out of scope.

PyTorch reference dumps live at `/tmp/sam3.1_ref/` produced by
`ref/sam3.1/gen_image_ref.py` (uses Meta's `build_sam3_image_model` on
`/mnt/disk01/models/sam3.1/sam3.1_multiplex.pt` with an fp32
`addmm_act` monkey-patch — Meta's fused kernel hard-codes bf16).

## Smoke test
```
SAM31_VISION_ONLY=1 ./verify_patch_embed \
    --ckpt /mnt/disk01/models/sam3.1/sam3.1.model.safetensors \
    --refdir /tmp/sam3.1_ref
# → patch_embed: max_abs=2.807e-03 mean_abs=1.209e-04
```

`SAM31_VISION_ONLY=1` tells the runner to stop after the vision backbone
so vision-only verify binaries can run before §3/§4 land.

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

**Actions in `cuda_sam3_1_runner.c`:** *(done 2026-04-22)*
- `fuse_qkv()` / `fuse_qkv_bias()` helpers **removed** — `load_block()`
  now uploads `attn.qkv.{weight,bias}` directly (shape `(3*D, D)` / `(3*D,)`,
  trusted to be `[Q;K;V]` concat per the Meta convention).
- `load_block()`: prefix is `backbone.vision_backbone.trunk.blocks.%d.`;
  leaf names `norm1/norm2`, `attn.qkv`, `attn.proj`, `mlp.fc{1,2}`.
- Entry-level (`cuda_sam3_1_init`): `backbone.vision_backbone.trunk.patch_embed.proj.weight`
  (no bias) + `trunk.ln_pre.{weight,bias}`. `w_pos` is now `NULL` (learned
  pos embed was replaced by RoPE in sam3.1 — though a `trunk.pos_embed` of
  shape (1, 577, 1024) *does* exist in the checkpoint; whether it is still
  used alongside RoPE is a Phase C architectural question).
- Constants match: `S3_DIM=1024`, `S3_MLP=4736`, `S3_PATCH=14`, 32 blocks ✓.

**§1b Still to do (FPN/neck rename):**
sam3's `vision_encoder.neck.fpn_layers.N.{scale_layers, proj1, proj2}`
does not exist in sam3.1. Replace with the three-stage `convs.0..2` stack
(`dconv_2x2_{0,1}` + `conv_1x1` + `conv_3x3`) and mirror-load
`interactive_convs` / `propagation_convs` if the forward path needs them
(likely does — they appear in the detector forward graph).

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
