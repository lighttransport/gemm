# TRELLIS.2 Assets

This file summarizes the model files, reference inputs, and auxiliary assets
used by the TRELLIS.2 code in this repo.

## Stage 1: Required Assets

These three files are the only fully pinned TRELLIS.2 model assets in the repo
today. They are referenced by the CPU, CUDA, and Vulkan Stage 1 runners.

| Component | Hugging Face source | Expected local path | Size | Format | Used by |
|-----------|----------------------|---------------------|------|--------|---------|
| DINOv3 ViT-L/16 image encoder | `timm/vit_large_patch16_dinov3.lvd1689m` | `/mnt/disk01/models/dinov3-vitl16/model.safetensors` | 1.2 GB | F32 safetensors | CPU, CUDA, Vulkan |
| Stage 1 DiT | `microsoft/TRELLIS.2-4B` | `/mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` | 2.5 GB | BF16 safetensors | CPU, CUDA, Vulkan |
| Stage 1 structure decoder | `microsoft/TRELLIS-image-large` | `/mnt/disk01/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors` | 141 MB | F16/F32 mixed safetensors | CPU, CUDA, Vulkan |

Download commands:

```bash
uvx --from huggingface_hub hf download \
  timm/vit_large_patch16_dinov3.lvd1689m model.safetensors \
  --local-dir /mnt/disk01/models/dinov3-vitl16

uvx --from huggingface_hub hf download \
  microsoft/TRELLIS.2-4B ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
  --local-dir /mnt/disk01/models/trellis2-4b

uvx --from huggingface_hub hf download \
  microsoft/TRELLIS-image-large ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
  --local-dir /mnt/disk01/models/trellis-image-large
```

## Stage 2: Shape Assets

These assets are used by the newer Stage 2 code paths in `cuda/trellis2/` and
the CPU reference scripts. The filenames below are present in local repo notes,
but unlike Stage 1 they are not yet documented with download commands elsewhere
in the repo.

| Component | Expected local path | Size | Format | Notes |
|-----------|---------------------|------|--------|-------|
| Stage 2 shape flow DiT | `/mnt/disk01/models/trellis2-4b/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors` | 2.5 GB | BF16 safetensors | Sparse latent flow model |
| Stage 2 shape decoder | `/mnt/disk01/models/trellis2-4b/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors` | 905 MB | F16 safetensors | SC-VAE decoder, outputs 7 channels |

## Stage 3: Texture Assets

Stage 3 support exists in the CUDA path and in CPU reference scripts, but the
repo does not currently pin exact checkpoint filenames in docs the way it does
for Stage 1.

What the code expects:

- A Stage 3 texture flow safetensors checkpoint passed as `--stage3`
- A texture decoder safetensors checkpoint passed as `--tex-dec`
- The texture decoder uses the same sparse decoder codepath as the shape
  decoder, but produces 6 output channels for PBR field generation

Known supporting config:

- `pipeline.json` with `shape_slat_normalization` and `tex_slat_normalization`
  values is required by:
  - `cpu/trellis2/run_stage2_ref.py`
  - `cpu/trellis2/run_stage3_tex_ref.py`
- Those scripts currently read it from `/tmp/t2cfg/pipeline.json`

## Reference/Test Data In Repo

Local reference artifacts currently checked into `cuda/trellis2/` include:

| File | Shape / meaning |
|------|------------------|
| `teapot_verified_latent.npy` | Verified Stage 1 latent, shape `[8, 16, 16, 16]` |
| `teapot_verified_occ.npy` | Verified occupancy output, shape `[1, 64, 64, 64]` |
| `chair_e2e_occ.npy` | End-to-end occupancy output for a chair sample |
| `teapot_e2e_occ.npy` | End-to-end occupancy output for a teapot sample |
| `stage1_check.npy`, `stage1_check2.npy`, `stage1_recheck.npy` | Intermediate validation dumps |

There are also multiple experimental `.npy` snapshots under `cuda/trellis2/`
used during CUDA verification and tuning.

## Runtime Inputs By Path

### CPU Stage 1

`cpu/trellis2/test_trellis2` can run in these modes:

- `encode`: image + DINOv3 weights
- `stage1`: Stage 1 DiT weights + `features.npy`
- `decode`: decoder weights + `latent.npy`
- `full`: image + all three Stage 1 weights
- `mesh`: `occupancy.npy` only

### CUDA Stage 1/2/3

`cuda/trellis2/test_cuda_trellis2` accepts:

- Stage 1: `<stage1.st> <decoder.st> <features.npy>`
- Optional DINOv3: `--image <path> --dinov3 <path>`
- Stage 2: `--stage2 <path> --shape-dec <path>`
- Stage 3: `--stage3 <path> --tex-dec <path>`

### Vulkan Stage 1

`vulkan/build_t2/test_vulkan_trellis2` currently supports Stage 1 only:

- `<stage1.st> <decoder.st> <features.npy>`
- Optional encode path: `--encode <dinov3.st> <image.npy>`
- No Stage 2 / Stage 3 Vulkan asset path is wired yet

## Vulkan Shader Assets

Vulkan TRELLIS.2 also depends on shader sources under:

- `vulkan/shaders/trellis2/*.comp`

These are compiled into `.spv` files during the CMake build and copied into the
build tree under:

- `vulkan/build_t2/shaders/trellis2/`

## Sampling / Normalization Constants

Pinned defaults in the current codebase:

- Stage 1 Euler steps: `12`
- Stage 1 CFG scale: `7.5`
- Stage 1 timestep rescale: `5.0`
- Guidance interval: `[0.6, 1.0]`
- `sigma_min`: `1e-5`

Pinned normalization arrays for Stage 2 / Stage 3 sparse latents are embedded in:

- `cuda/trellis2/test_cuda_trellis2.c`

and mirrored in `pipeline.json` for the CPU reference scripts.
