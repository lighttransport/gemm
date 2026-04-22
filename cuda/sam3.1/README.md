# CUDA SAM 3.1 runner (scaffold, cloned from cuda/sam3)

Status: **ViT backbone loader ported**; FPN, text encoder, and DETR/head
loaders still target sam3 key paths. Builds and now reaches the FPN stage
before the first missing-key error. See `PORT.md` for remaining mappings +
kernel changes (RoPE, OpenAI-CLIP, segmentation head).

## Build

```
make -j
```

Produces `test_cuda_sam3_1` plus `verify_*` binaries (same set as `cuda/sam3/`).

## What was done

- Files cloned from `cuda/sam3/` with identifiers renamed:
  `cuda_sam3_runner.{c,h}` → `cuda_sam3_1_runner.{c,h}`, `test_cuda_sam3` →
  `test_cuda_sam3_1`, all internal `cuda_sam3_*` symbols similarly renamed.
- Safetensors top-level prefix changed from `detector_model.` to `detector.`.
- `Makefile` updated (source names, default `CKPT` path, reuses
  `cpu/sam3/sam3_clip_bpe.{c,h}` which is architecture-independent).
- Weight converter: `convert_pt_to_safetensors.py` (flat `.pt` → safetensors;
  pass `--pt <multiplex.pt> --out <model.safetensors>`).
- **ViT backbone loader ported** (2026-04-22): entry-level `patch_embed` +
  `ln_pre` and all 32 blocks (`backbone.vision_backbone.trunk.blocks.N.*`,
  `attn.qkv`/`attn.proj`, `norm1/norm2`, `mlp.fc1/fc2`) load without errors.
  Old `fuse_qkv()` / `fuse_qkv_bias()` helpers removed — sam3.1 stores the
  QKV weight pre-fused as `(3*D, D)`.

## What's left (summary — see `PORT.md` for detail)

1. **FPN / neck**: *main conv stack loader ported* (2026-04-22) —
   `backbone.vision_backbone.convs.{0..2}` now loads cleanly with
   S3_FPN_LEV=3. The parallel `interactive_convs.*` / `propagation_convs.*`
   stacks are **not** yet loaded (deferred to forward-pass Phase C);
   their keys do exist in the checkpoint.
2. **ViT attention kernel**: replace relative-position-bias with 2D RoPE
   (`attn.freqs_cis` per block, complex64 (576, 32)). Loader-side stash
   of `freqs_cis` still TODO — weight is on disk but not uploaded yet.
3. **Text encoder**: retarget to OpenAI-CLIP layout (`transformer.resblocks.*`,
   `attn.in_proj_*`, `ln_{1,2}`, `mlp.{c_fc,c_proj}`, `text_projection`);
   verify pre-norm ordering.
4. **DETR / mask / scoring heads**: re-wire against `detector.transformer.*`,
   `detector.segmentation_head.*`, `detector.dot_prod_scoring.*`,
   `detector.geometry_encoder.*`. Box-head layout differs from sam3.
5. **Tracker** (`tracker.model.*`): 457 new tensors, no sam3 analog —
   implement separately if video tracking is needed.

## Files

- `cuda_sam3_1_runner.{c,h}`, `cuda_sam3_1_kernels.h` — ported runner/kernels
- `test_cuda_sam3_1.c`, `verify_*.c` — entrypoints/tests (names unchanged)
- `Makefile` — build targets
- `convert_pt_to_safetensors.py`, `inspect_ckpt.py`, `sam3.1_keys.txt`
- `.venv/` — uv venv with torch/safetensors (gitignored)
- `PORT.md` — detailed porting guide
