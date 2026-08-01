# SAM 3 → SAM 3.1 Porting Notes

This directory is a **clone** of `cuda/sam3/` with identifiers renamed
(`cuda_sam3_` → `cuda_sam3_1_`) and the safetensors prefix changed from
`detector_model.` to `detector.`.

**Status (2026-05-14):** image text-prompt pipeline is ported and
verified vs Meta's PyTorch reference on fujisan.jpg / "mountain".
SAM3.1 ViT MLP uses exact
erf-GELU to match PyTorch's default `gelu`, `precision=fp32` now
uploads linear weights as F32 and routes GEMMs through
`gemm_tiled_f32_f32`, the DETR prompt path now includes the
image-conditioned geometry CLS token used by Meta's text-only grounding
path, and decoder layer-norm loading now matches Meta's order
(`norm2` for self-attn, `norm1` for image cross-attn). Fresh
fujisan.jpg / "mountain" refs:

| metric          | default fp16/MMA | precision=fp32 |
|-----------------|------------------|----------------|
| fpn2 max_abs    | 0.328            | 0.140          |
| fpn2 mean_abs   | 1.85e-03         | 1.14e-03       |
| detr_enc max_abs | 0.394           | 0.428          |
| detr_enc mean_abs | 4.92e-03       | 2.55e-03       |
| detr_dec mean_abs | 9.24e-03       | 3.92e-03       |
| dot_score L5 mean_abs | 9.35e-03   | 4.19e-03       |
| final top score | 0.8406           | 0.8410         |
| final top box   | [4.31, 184.27, 1691.96, 509.25] | [4.29, 184.28, 1691.96, 509.21] |
| final mask IoU  | 0.9996           | 0.9999         |

Additional default fp16/MMA regressions:

| phrase | final IoU | top score | top box xyxy |
|--------|-----------|-----------|--------------|
| snow | 0.9997 | 0.6359 | [596.34, 184.59, 1110.56, 353.12] |
| sky | 1.0000 | 0.9801 | [1.52, 1.49, 1688.91, 452.78] |

Raw `segmentation_head_out` dumps are now consumed by `verify_mask`.
Current default fp16/MMA mean_abs: mountain `pred_masks=8.88e-02`,
`semantic_seg=6.42e-03`; snow `pred_masks=2.79e-01`,
`semantic_seg=5.21e-03`; sky `pred_masks=4.70e-02`,
`semantic_seg=4.99e-03`.

The previous FPN2 residual is fixed below the old 0.5 max_abs budget.
The larger DETR residual was not attention math: the CUDA runner was
cross-attending only to the 32 text tokens, while Meta's encoder prompt
is 33 tokens (`text[32] + geometry_cls[1]`). With ref FPN2/text inputs
and the geometry CLS path enabled, DETR encoder drift drops to
`max_abs=7.1e-03, mean_abs=4.1e-04`.

Current drift vs PyTorch:
- Full-image DETR encoder is now below 0.5 max_abs in both default and
  `precision=fp32` modes.
- Default final mask IoU is now above the 0.999 target. The fp32 path
  reaches 0.9999 on this reference.

Historical ViT-only drift table:

| stage   | max_abs | mean_abs | ref max | relative |
|---------|---------|----------|---------|----------|
| patch_embed | 2.8e-3 | 1.2e-4 | — | F16 noise |
| block0  | 2.0e-2 | 9.5e-4 | 19.4 | 0.1% |
| block15 | 2.0    | 2.7e-3 | 151.8 | 1.3% |
| block31 | 5.1    | 5.3e-3 | 223.1 | 2.3% |

**Historical drift note (2026-04-23, superseded):** an earlier run had
DETR enc 2.83 / DETR dec 4.20 / mask IoU 0.98 vs ref. Part of that was
F16 weight-quantization in the ViT + downstream MMA GEMM path, but the
larger DETR/decoder gaps were later traced to missing geometry-CLS prompt
construction and swapped decoder norm loads. Useful isolation data from
that pass:

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

**Historical precision-for-speed knob (2026-04-23):** the existing
`SAM3_GEMM=tiled` env var routed every GEMM (ViT + text + DETR + mask
decoder) through the F16-weight tiled kernel. The newer
`precision=fp32` path is summarized in the 2026-05-13 update above.
Old fujisan.jpg/"mountain" measurements:

| metric          | SAM3_GEMM=mma (default) | SAM3_GEMM=tiled |
|-----------------|-------------------------|-----------------|
| fpn1 max_abs    | 0.351                   | 0.290           |
| fpn2 max_abs    | 0.478                   | 0.306           |
| detr_enc max_abs| 2.82                    | 2.82            |
| final IoU       | 0.9801                  | 0.9802          |
| wall-clock      | 4.5 s                   | 6.9 s (+52%)    |

Those old DETR numbers predate the geometry-CLS prompt fix above. The
tiled/F32-linear path is kept as the precision-critical escape hatch;
MMA remains default for latency-sensitive use.

Historical fixes that landed during this port: (a) `trunk.pos_embed`
load (skipping the cls_token row) was
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

DETR encoder forward current status:
`verify_detr` vs `detr_enc_memory.npy` (encoder output, 5184×1×256)
is below 0.5 max_abs in the full image path. With ref FPN2/text inputs,
`max_abs=7.1e-03, mean_abs=4.1e-04`, confirming that the layer math and
prompt construction now match the reference closely.

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
  → `transformer.decoder.layers.N.{norm2, catext_norm, norm1, norm3}`
- `dot_product_scoring.{text_mlp.layer{1,2}, text_mlp_out_norm, text_proj, query_proj}`
  → `dot_prod_scoring.{prompt_mlp.layers.{0,1}, prompt_mlp.out_norm, prompt_proj, hs_proj}`
- `mask_decoder.{prompt_cross_attn.{q,k,v,o}_proj, prompt_cross_attn_norm, pixel_decoder.*, mask_embedder.layers, instance_projection, semantic_projection}`
  → `segmentation_head.{cross_attend_prompt.{in_proj,out_proj}, cross_attn_norm, pixel_decoder.*, mask_predictor.mask_embed.layers, instance_seg_head, semantic_seg_head}`

Remaining out-of-scope areas:
- The `interactive_convs.*` / `propagation_convs.*` vision-backbone
  stacks are present in the checkpoint but not loaded; they feed
  prompt-driven/video paths which are not exercised by image-only
  text-prompt inference.
- Tracker (`tracker.model.*`) is out of scope for this image runner.

PyTorch reference dumps live at `/tmp/sam3.1_ref/` produced by
`ref/sam3.1/gen_image_ref.py` (uses Meta's `build_sam3_image_model` on
`<path-to-models>/sam3.1/sam3.1_multiplex.pt` with an fp32
`addmm_act` monkey-patch — Meta's fused kernel hard-codes bf16).

## Smoke test
```
SAM31_VISION_ONLY=1 ./verify_patch_embed \
    --ckpt <path-to-models>/sam3.1/sam3.1.model.safetensors \
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

**§1b FPN/neck rename:** done for the image text-prompt path. The runner
uses the SAM3.1 `convs.0..2` stack and keeps the FPN projection convs in
F32 for accuracy. `interactive_convs` / `propagation_convs` remain
outside the image-only path.

## 2. ViT attention: RoPE replaces relative position bias

sam3 uses axial relative-position-bias tables baked into attention.
sam3.1 uses 2D RoPE with **pre-computed complex-valued `freqs_cis`**
of shape `(grid*grid, head_dim/2)`. The CUDA path now builds the needed
RoPE tables at context creation and applies them to Q/K before attention.

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

OpenAI-CLIP is **pre-norm** (norm → attn → +res → norm → mlp → +res).
`cuda_sam3_1_run_text()` follows that ordering and verifies against
`text_ln_final.npy`.

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
