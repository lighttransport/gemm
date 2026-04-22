# CUDA SAM 3.1 runner (scaffold, cloned from cuda/sam3)

Status: **scaffold only**. Builds, reaches the loader, then fails on key-path
differences. See `PORT.md` for the full mapping + required kernel changes.

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
- Weight converter: `convert_pt_to_safetensors.py` (flat `.pt` → safetensors,
  already produced `/mnt/nvme02/models/sam3.1/sam3.1.model.safetensors`).

## What's left (summary — see `PORT.md` for detail)

1. **ViT loader rename** + switch `fuse_qkv` to load pre-fused `attn.qkv`
   (sam3.1 stores QKV fused; sam3 stored them separately).
2. **ViT attention kernel**: replace relative-position-bias with 2D RoPE
   (`attn.freqs_cis` is provided per block, complex64).
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
