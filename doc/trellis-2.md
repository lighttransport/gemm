# TRELLIS.2 Pipeline — Architecture & Implementation Status

TRELLIS.2 (Microsoft, 4B total params) generates textured 3D meshes from single images
via a 3-stage flow-matching diffusion pipeline. This document covers the full pipeline
architecture and our implementation/verification status.

**Repository**: [github.com/microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2)
**Model**: [huggingface.co/microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B)

## Full Pipeline Overview

```
Image (RGB)
  │
  ▼
┌──────────────────────────┐
│  DINOv3 ViT-L/16         │  Image encoder (303M params)
│  512×512 → [1029, 1024]  │  Features: 1 CLS + 4 register + 1024 patch tokens
└──────────┬───────────────┘
           │ conditioning features
           ▼
┌──────────────────────────┐
│  Stage 1: Structure DiT   │  SparseStructureFlowModel (1.29B params)
│  noise [8,16³] → [8,16³] │  30 DiT blocks, 12 Euler steps, CFG=7.5
└──────────┬───────────────┘
           │ latent [8, 16, 16, 16]
           ▼
┌──────────────────────────┐
│  Stage 1: Decoder         │  SparseStructureDecoder (37M params)
│  [8, 16³] → [1, 64³]     │  Conv3D + ResBlocks + pixel_shuffle
└──────────┬───────────────┘
           │ occupancy logits [64, 64, 64]
           ▼
┌──────────────────────────┐
│  Sparse Coords            │  Threshold → extract occupied voxels
│  → [N, 4] (batch,z,y,x)  │  Typically 5-15% of 64³ = 13K-40K voxels
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Stage 2: Shape DiT       │  SLatFlowModel (1.3B params)
│  [N, 32] latent space     │  Sparse DiT, 12 steps, produces shape latent
└──────────┬───────────────┘  (CUDA DiT implemented)
           │
           ▼
┌──────────────────────────┐
│  Stage 2: Shape Decoder   │  SC-VAE decoder
│  shape latent → mesh      │  Vertices + faces + attributes
└──────────┬───────────────┘  (CUDA SC-VAE shape decoder implemented)
           │
           ▼
┌──────────────────────────┐
│  Stage 3: Texture DiT     │  Texture flow model (1.3B params)
│  shape + image → texture  │  12 steps, produces texture latent
└──────────┬───────────────┘  (CUDA DiT implemented)
           │
           ▼
┌──────────────────────────┐
│  Stage 3: Texture Decoder │  SC-VAE texture decoder
│  texture latent → PBR     │  Albedo, roughness, metallic maps
└──────────┬───────────────┘  (CUDA SC-VAE texture decoder implemented)
           │
           ▼
┌──────────────────────────┐
│  Mesh Export              │  Marching cubes (structure) or
│  .obj / .glb / .ply      │  direct mesh extraction (shape)
└──────────────────────────┘
```

## Component Architectures

### DINOv3 ViT-L/16

| Parameter | Value |
|-----------|-------|
| Blocks | 24 |
| embed_dim | 1024 |
| Heads | 16 (head_dim=64) |
| FFN | 4096 |
| Patch size | 16 |
| Image size | 512×512 |
| Register tokens | 4 |
| Position encoding | RoPE only (no learned pos embed) |
| Output | [1029, 1024] (1 CLS + 4 register + 1024 patches) |
| Weights | 303M params, F32 (timm) |
| Final norm | **Unparameterized LayerNorm** (not learned norm) for TRELLIS.2 |

### Stage 1 DiT (SparseStructureFlowModel)

| Parameter | Value |
|-----------|-------|
| Blocks | 30 × ModulatedTransformerCrossBlock |
| model_channels | 1536 |
| Heads | 12 (head_dim=128) |
| FFN | 8192 |
| in/out_channels | 8 |
| Resolution | 16³ (4096 tokens) |
| Conditioning | DINOv3 features [1029, 1024] via cross-attention |
| Position encoding | 3D RoPE (21 freqs/axis, complex-pair convention) |
| Modulation | Shared `adaLN_modulation` + per-block `modulation` bias |
| QK normalization | Per-head RMSNorm, gamma [n_heads, head_dim] |
| Final norm | Unparameterized LayerNorm before out_layer |
| Weights | 1.29B params, BF16 (2.5 GB) |

**Sampling parameters** (from `pipeline.json`):

| Parameter | Value |
|-----------|-------|
| Steps | 12 (Euler) |
| CFG scale | 7.5 |
| CFG rescale | 0.7 (std-ratio matching) |
| Guidance interval | [0.6, 1.0] (CFG off for last 2 steps) |
| Rescale_t | 5.0 |
| sigma_min | 1e-5 |
| Timestep scaling | t × 1000 passed to model |

**Key implementation details**:
- QKV is a fused `[3*dim, dim]` weight, split via `torch.chunk(3)` (NOT per-head interleaved)
- Cross-attention KV is `[2*dim, cond_dim]`, also split via `chunk(2)`
- Timestep embedding: sinusoidal(256) → Linear(256,1536) → SiLU → Linear(1536,1536)
- Shared modulation: SiLU(t_emb) → Linear(1536, 9216) → chunk(6) → shift/scale/gate for SA + MLP
- Per-block modulation: `modulation` parameter [9216] added to shared output
- Self-attention gate applied AFTER output projection: `h += gate * proj(attn(...))`
- Cross-attention: NO gate, direct residual `h += proj(cross_attn(...))`

### Stage 1 Decoder (SparseStructureDecoder)

| Parameter | Value |
|-----------|-------|
| Normalization | **GroupNorm(32)** (32 groups; with C=32 → 1 channel/group = per-channel spatial norm) |
| Groups | 32 (always, regardless of channel count) |

```
input_layer:  Conv3d(8 → 512, k=3, pad=1)        [512, 16³]
middle_block: 2× ResBlock3d(512)                    [512, 16³]
blocks.0-1:   2× ResBlock3d(512)                    [512, 16³]
blocks.2:     Conv3d(512 → 1024, k=3, pad=1)       [1024, 16³]
              pixel_shuffle_3d(factor=2)             → [128, 32³]
blocks.3-4:   2× ResBlock3d(128)                    [128, 32³]
blocks.5:     Conv3d(128 → 256, k=3, pad=1)        [256, 32³]
              pixel_shuffle_3d(factor=2)             → [32, 64³]
blocks.6-7:   2× ResBlock3d(32)                     [32, 64³]
out_layer:    GroupNorm(32) → SiLU → Conv3d(32→1, k=3)
Output:       [1, 64, 64, 64] occupancy logits
```

ResBlock3d: `GN(32) → SiLU → Conv3d → GN(32) → SiLU → Conv3d + skip`

### Stages 2-3

- **Stage 2 Shape**: Sparse DiT (1.3B), operates on sparse voxel coords from Stage 1.
  Two resolution variants: 512 and 1024 (cascade mode). Uses FlexGEMM for sparse conv.
- **Stage 3 Texture**: Sparse DiT (1.3B), conditioned on shape + image.
  Produces texture latent decoded by SC-VAE to PBR material maps.
- **Decoders**: SC-VAE (Sparse Convolutional VAE) for both shape and texture. CUDA
  shape and texture decode paths are implemented; texture uses dense 8x C2S
  because `tex_dec_next_dc_f16c32_fp16` has `pred_subdiv=false`.

## Implementation Status Matrix

### Stage 1: Structure Generation

| Component | CUDA | CPU | PyTorch Ref | Verified |
|-----------|------|-----|-------------|----------|
| DINOv3 ViT-L/16 encoder | — | `common/dinov3.h` | `run_official_e2e.py` | Features differ (timm vs transformers weights) |
| DINOv3 timm→transformers convert | — | — | `run_official_e2e.py` | Matches official features |
| DiT forward (single step) | `cuda_trellis2_runner.c` | `trellis2_dit.h` | `dump_block0_detail.py` | **corr=1.000, rel_L2=3e-6** |
| DiT 12-step no-CFG | `test_cuda_trellis2.c` | `trellis2_stage1.h` | `dump_per_block.py` | **corr=1.000, rel_L2=1.5e-5** |
| DiT 12-step CFG (no rescale) | `test_cuda_trellis2.c` | — | verified | **corr=1.000, rel_L2=1.5e-5** |
| DiT 12-step CFG + rescale | `test_cuda_trellis2.c` | `trellis2_stage1.h` | `run_official_e2e.py` | **corr=1.000, rel_L2=2.4e-5** |
| Structure decoder | `cuda_trellis2_runner.c` | `trellis2_ss_decoder.h` | `run_official_e2e.py` | **corr=1.000, rel_L2=9.3e-5** |
| Marching cubes → .obj | `test_cuda_trellis2.c` (MC lib) | `test_trellis2.c` (inline) | skimage | Both work |
| Full pipeline (features→mesh) | `test_cuda_trellis2` | `test_trellis2` | `run_official_e2e.py` | **Exact match (10.2% = 10.2%)** |

### Stage 1: Vulkan (AMD RX 9070 XT, GFX1201 / RDNA4)

| Component | Status | Notes |
|-----------|--------|-------|
| DINOv3 ViT-L/16 encoder | ✓ CORRECT | max diff 0.011 vs official (corr > 0.9999) |
| Stage 1 DiT (single step) | ✓ CORRECT | matches PyTorch exactly with same noise |
| Stage 1 DiT + decoder (same noise) | ✓ CORRECT | 94% IoU with PyTorch reference |
| Full pipeline (random noise) | ✓ WORKS | 7.8% occupancy with seed=42 |
| Stage 2 shape flow DiT | ✓ IMPLEMENTED | Generic DiT, sparse RoPE, 12-step Euler + CFG |
| Stage 2 shape decoder (SC-VAE) | ✓ IMPLEMENTED | CUDA sparse ConvNeXt + C2S; GPU subdivision/hash/index default |
| Stage 3 texture flow DiT | ✓ IMPLEMENTED | No CFG, [noise\|shape_norm] concat input |
| Stage 3 texture decoder | ✓ IMPLEMENTED | CUDA dense C2S SC-VAE; vertex-color OBJ export by default |
| Cross-attn KV cache | ✓ IMPLEMENTED | Precomputed for all 30 blocks, saves ~2160 dispatches/stage |

### Stage 1: HIP/ROCm (AMD RX 9070 XT, GFX1201 / RDNA4)

| Component | Status | Correlation | Notes |
|-----------|--------|-------------|-------|
| DINOv3 ViT-L/16 encoder | CPU only | — | Uses `common/dinov3.h` (210s @ 16 threads) |
| Stage 1 DiT (single step) | ✓ CORRECT | **0.99999754** | BF16 WMMA GEMM + BF16 WMMA flash-attn (gfx12 matrix cores) |
| Stage 1 DiT (single step, FP8) | ✓ CORRECT | **0.999358** vs BF16 | BF16-act × FP8-wt WMMA; offline-quantized E4M3 weights, 1.27 GB on disk |
| Stage 1 Decoder (zero input) | ✓ CORRECT | **1.00000000** | GroupNorm kernel, 866x speedup vs CPU |
| Full pipeline (12-step CFG + decode) | ⚠ RUNS | — | Pipeline completes; occupancy too high (~70% vs expected ~2-10%). DiT+decoder individually verified, CFG sampling loop under investigation |
| DiT per-step time (BF16 WMMA, T2_WMMA=1) | — | — | **1.66 s/step** (2026-05 measured, gfx1201) |
| DiT per-step time (BF16-act × FP8-wt WMMA) | — | — | **1.46 s/step** (~12% over BF16; weight bw not the bottleneck) |
| DiT per-step time (scalar F32 fallback) | — | — | 143.7 s/step (T2_WMMA=0) — 83× WMMA speedup |
| Decoder time | — | — | 0.4s |
| DINOv3 features (C vs PyTorch) | ✓ MATCH | **0.99999326** | max_diff=0.116, C encoder features essentially correct |
| PyTorch reference (ROCm 7.2) | ✓ WORKS | — | `gen_stage1_ref.py` runs on ROCm GPU (~110s for 12 steps + decode) |

**Bugs fixed during HIP port** (in order of discovery):

| Bug | Symptom | Root Cause | Fix |
|-----|---------|-----------|-----|
| HIP kernel launch ABI | SIGSEGV on GPU forward | Struct-buffer `extra` param crashes on ROCm gfx1201 | Use pointer-array `kernelParams` style |
| DiT tensor names | SIGSEGV (NULL weights) | Wrong safetensors tensor names for block weights | Match `common/trellis2_dit.h` names |
| adaLN_modulation weight dtype | All-zero output | `mod_w` uploaded as F16, kernel reads as F32 | Upload as F32 (`st_upload_f32`) |
| Timestep scaling | Wrong modulation | CPU uses `t * 1000`, GPU used `t` raw | Scale by 1000 before sinusoidal embed |
| RoPE frequencies | Wrong attention | `10000^(-2i/head_dim)` vs correct `10000^(-i/n_freqs)` | Match CPU formula |
| Epsilon value | Minor numerical diff | GPU used eps=1e-5, CPU uses eps=1e-6 | Set all eps to 1e-6 |
| GELU variant | Minor numerical diff | GPU used exact erf GELU, CPU uses tanh approx | Use tanh GELU |
| Double bias add | Bias added 2× | `gemm(bias=x)` then `broadcast_bias(x)` in CA KV cache | Pass NULL bias to gemm |
| **d_t_out buffer aliasing** | **Block 1+ diverges (corr 0.75)** | **FC2 output stored in d_mod, overwritten by per-block modulation** | **Use d_ada_out as FC2 output buffer** |
| **Decoder normalization** | **Decoder output all zeros / wrong sign** | **Used per-spatial ChannelLN instead of GroupNorm(32)** | **Replace kernel with GroupNorm** |
| **ResBlock skip aliasing** | **Decoder output corrupted** | **conv_out = d_dec[1] same as buf (skip), destroying residual** | **3-buffer resblock: buf + tmp1 + tmp2** |

### Stages 2-3

| Component | CUDA | CPU | PyTorch Ref | Status |
|-----------|------|-----|-------------|--------|
| Sparse tensor infrastructure | — | `common/sparse3d.h` | — | CPU ops verified |
| Stage 2 shape flow DiT | `cuda_trellis2_runner.c` | `trellis2_dit.h` | `run_stage2_ref.py` | **CUDA verified (corr=1.0); E2E smoke-tested** |
| Stage 2 shape decoder (SC-VAE) | `cuda_trellis2_runner.c` | `trellis2_shape_decoder.h` | `gen_scvae_decoder_ref.py` | **CUDA verified vs PyTorch; exact coords on capped smoke** |
| Stage 3 texture flow DiT | `cuda_trellis2_runner.c` | `trellis2_dit.h` | `run_stage3_tex_ref.py` | CUDA path implemented; needs fresh ref dump |
| Stage 3 texture decoder (SC-VAE) | `cuda_trellis2_runner.c` | `trellis2_shape_decoder.h` | `gen_scvae_decoder_ref.py` | **CUDA verified vs PyTorch; dense C2S smoke** |
| FDG mesh extraction | — | `trellis2_fdg_mesh.h` | — | CPU verified, .obj export |
| Mesh extraction (marching cubes) | `test_cuda_trellis2.c` | `test_trellis2.c` | — | Both work |

## Numerical Verification Results

All results below use **matched initial noise** (same tensor loaded from .npy).

### CUDA vs PyTorch (RTX 5060 Ti, sm_120)

| Test | Correlation | Relative L2 | Notes |
|------|-------------|-------------|-------|
| DiT single step | 1.00000000 | 3.0e-6 | F32 compute, BF16→F32 weights |
| DiT 12-step no-CFG | 1.00000000 | 1.5e-5 | All 30 blocks × 12 steps exact |
| DiT 12-step CFG no-rescale | 1.00000000 | 1.5e-5 | CFG=7.5, guidance interval |
| DiT 12-step CFG + rescale | 1.00000000 | 2.4e-5 | rescale=0.7, std-ratio matching |
| Decoder (zero input) | 0.99999952 | ~1e-4 | All-negative output (correct) |
| Decoder (real latent) | 0.99999998 | 9.3e-5 | 26750 occupied = exact match |
| **Full pipeline** | **1.00000000** | **~1e-4** | **10.2% occupancy = 10.2%** |
| Stage 2 DiT step (N=100) | 1.00000000 | — | F32 weights |
| Stage 2 DiT step (N=1000) | 1.00000000 | — | F32 weights |
| Stage 2 DiT step (N=100, F16 MMA) | 0.99999990 | ~3e-4 | F16 weights, MMA tensor cores |
| Stage 2 DiT step (N=5000, F16 MMA) | 0.99999992 | ~3e-4 | F16 MMA on sm_120 Blackwell |
| **Stage 1→2 full pipeline** | — | — | **Stage 1: 45s, Stage 2: 58s (N=28K)** |

### Per-block DiT comparison (block 0)

| Intermediate | Match |
|---|---|
| Input embedding | Exact (6 decimal places) |
| adaLN output | Exact |
| Q/K after RMSNorm | Exact |
| Q/K after RoPE | Exact |
| Self-attention output | Exact |
| Cross-attention output | Exact |
| MLP output | Exact |
| Block 0 final hidden (6.3M elements) | corr=1.000, max_diff=0.002 |

## Performance

| Stage | CPU (Zen4, 16T) | CUDA (RTX 5060 Ti) | HIP (RX 9070 XT) | PyTorch (RX 9070 XT) |
|-------|----------------|---------------------|-------------------|----------------------|
| DINOv3 encode | ~3.5 min | — (CPU features) | — (CPU features) | ~10 sec |
| Stage 1 DiT (12 steps, CFG) | ~9 hours | **~50 sec** estimated from 2026-05 sm_120 run | **~42 sec** (BF16 WMMA) | ~110 sec |
| Decoder | ~6.3 min | ~0.5 sec | ~0.4 sec | ~0.2 sec |
| **Total (DiT + decode only)** | **~9.5 hours** | **~50 sec** estimated | **~42 sec** | **~2 min** |

CUDA Stage 1/2/3 DiT uses F16 MMA GEMM for BF16/F16 weights and an MMA attention
path for head_dim=128. The CUDA cross-attention KV cache is keyed by model,
condition hash, and CFG slot: slot 0 for the active image condition and slot 1
for the all-zero negative condition. This avoids rebuilding and evicting the KV
cache on every conditioned/unconditioned CFG pass; a 2-step Stage 1 run now builds
two caches total (`slot=0`, `slot=1`) and reuses both on step 2.

### CUDA SC-VAE Decoders

Implementation files:

- `cuda/trellis2/cuda_trellis2_runner.c`: shape/texture decoder load/run paths.
- `cuda/trellis2/cuda_trellis2_kernels.h`: sparse gather-map, C2S gather, residual repeat kernels.
- `cuda/trellis2/verify_shape_decoder.c`: CPU-vs-CUDA SC-VAE decoder verifier
- `ref/trellis2/gen_scvae_decoder_ref.py`: PyTorch SC-VAE decoder reference
  generator with a tiny pure-PyTorch sparse-conv backend for parity checks.

`verify_shape_decoder` also accepts external PyTorch references:
`--ref-feats <npy> --ref-coords <npy>`. Add `--skip-cpu` when only the CUDA-vs-
PyTorch comparison is needed.

Default precision is F32 for the SC-VAE shape decoder because F16/MMA can change
subdivision decisions near the `to_subdiv > 0` threshold. `T2_SHAPE_DEC_F16=1`
enables the faster F16 path; `T2_SHAPE_DEC_F32=1` forces F32.

Current capped smoke verification, RTX 5060 Ti, input `N=128` from the Fujisan
Stage 2 one-step dump:

```bash
env XDG_RUNTIME_DIR=/run/user/1000 CUDA_RUNNER_PRECISE_MATH=1 \
  ./cuda/trellis2/verify_shape_decoder \
  /mnt/disk01/models/trellis2-4b/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors \
  /tmp/trellis2_fujisan_s2_128_denorm.npy \
  /tmp/trellis2_fujisan_s2_128_coords.npy
```

Result:

| Mode | CPU output | CUDA output | Coord mismatch | Corr | Rel L2 | Max abs | CUDA SC-VAE time |
|------|------------|-------------|----------------|------|--------|---------|------------------|
| F32 default + precise math | N=3279,C=7 | N=3279,C=7 | 0 | 1.00000000 | 9.43e-7 | 2.75e-4 | ~175 ms |
| `T2_SCVAE_CUBLAS=1 T2_SCVAE_PACKED_CONV=1` | — | N=3279,C=7 | 0 | 1.00000000 | 5.28e-7 | 8.39e-5 | ~100 ms |
| `T2_SCVAE_CUBLAS=1 T2_SCVAE_PACKED_CONV=1 T2_SCVAE_CPUAVX_LN=1 T2_SCVAE_OUTPUT_GROUP=25 T2_SCVAE_OUTPUT_GROUP_FMA=1` | — | N=3279,C=7 | 0 | 1.00000000 | 5.07e-7 | 6.48e-5 | ~120 ms |
| previous row + `T2_SCVAE_FINAL_LN_EPS=0.000009` | — | N=3279,C=7 | 0 | 1.00000000 | 5.02e-7 | 6.10e-5 | ~120 ms |

Raw PyTorch fp32 CUDA reference for the same shape input is persisted in
`ref/trellis2/dumps/shape_scvae_fujisan128/`. It bypasses the optional
`cumesh` FDG mesh wrapper and instantiates the base SC-VAE decoder directly.
Result vs CUDA: `N=3279,C=7`, coordinate mismatch `0`, correlation
`1.00000000`, best full-output rel L2 `5.07e-7`, best full-output max abs
`6.48e-5`.

The F32 path keeps the CPU reference subdivision topology exactly on this smoke
case. The F16 subnormal upload fix removed the old `~5e-3` raw-output drift, but
the raw shape decoder is still above the `<1e-5` full-output max-abs target.
The exact final projection is now below target when started from the saved
PyTorch pre-output tensor:

```bash
env XDG_RUNTIME_DIR=/run/user/1000 T2SD_START_PRE_OUTPUT=1 \
  T2_SCVAE_OUTPUT_GROUP=25 T2_SCVAE_OUTPUT_GROUP_FMA=1 \
  ./cuda/trellis2/verify_shape_decoder \
  /mnt/disk01/models/trellis2-4b/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors \
  ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_pre_output_feats.npy \
  ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_pre_output_coords.npy \
  --start-stage 4 --start-block 0 \
  --ref-feats ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_output_feats.npy \
  --ref-coords ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_output_coords.npy \
  --skip-cpu
```

This gives max abs `7.63e-6`; non-FMA pair32 is `1.14e-5`, scalar/cuBLAS-like
orders are `~2.29e-5`. A closer PyTorch match is the cuBLASLt bias epilogue:

```bash
env XDG_RUNTIME_DIR=/run/user/1000 \
  CUBLASEW_LT_WORKSPACE_BYTES=1048576 CUBLASEW_LT_MIN_ALIGNMENT_BYTES=4 \
  T2_SCVAE_CUBLAS=1 T2_SCVAE_CUBLASLT_BIAS_GEMM=1 T2SD_START_PRE_OUTPUT=1 \
  ./cuda/trellis2/verify_shape_decoder \
  /mnt/disk01/models/trellis2-4b/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors \
  ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_pre_output_feats.npy \
  ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_pre_output_coords.npy \
  --start-stage 4 --start-block 0 \
  --ref-feats ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_output_feats.npy \
  --ref-coords ref/trellis2/dumps/shape_scvae_fujisan128/pytorch_fp32_cuda_output_coords.npy \
  --skip-cpu
```

That path is bit-exact (`max abs 0`) from saved PyTorch pre-output, including
with `T2_SCVAE_CUBLAS=1`. Full-output drift is therefore dominated by upstream
feature drift projected by `out_w`, not by the final projection GEMM.

Current localization on the shape smoke:

- Best pre-output max abs is `6.68e-6` with gather-GEMM sparse conv plus
  `T2_SCVAE_CPUAVX_LN=1`, but that error direction projects to `8.01e-5`
  full-output max abs.
- Best full-output max abs is `6.48e-5` with packed sparse conv,
  `T2_SCVAE_CPUAVX_LN=1`, and FMA group25 output. The fixed-pre-output
  output-layer sweep found several equivalent projection orders (`group25_fma`,
  `group27_fma`, `group50`, `group60`, `group62`) below the prior group32/FMA
  `6.87e-5` max; larger groups return to `6.87e-5` or worse.
  Reverse group/channel traversal probes (`T2_SCVAE_OUTPUT_GROUP_MODE=1/2/3`)
  and bias-as-initial-accumulator grouped output
  (`T2_SCVAE_OUTPUT_GROUP_BIASINIT=1`) tie the best PyTorch-reference max
  on the group25/FMA path but do not reduce it further. Pairwise tree
  reduction across group partials (`T2_SCVAE_OUTPUT_GROUP_TREE=1`) also ties
  at group25/group27 and worsens for sampled groups such as group4, group50,
  and group60.
- Starting from exact PyTorch `stage3` features,
  `T2_SCVAE_FINAL_WELFORD_LN=1` reproduces PyTorch CUDA's vectorized no-affine
  final LayerNorm bit-exactly; with cuBLASLt bias output this path is also
  bit-exact through final output. Enabling Welford for every intermediate
  no-affine LN (`T2_SCVAE_WELFORD_LN=1`) is not the best full-smoke mode because
  its remaining error direction projects worse than the CPUAVX LN path.
- cuBLASLt parity work:
  - `cublasew` now uses the CUDA-header-compatible
    `cublasLtMatmulHeuristicResult_t` stride. The old 256-byte opaque record
    made `CUBLASEW_LT_ALGO_INDEX>0` read garbage heuristic entries.
  - `CUBLASEW_LT_WORKSPACE_BYTES=N` and
    `CUBLASEW_LT_MIN_ALIGNMENT_BYTES=N` control the shared cuBLASLt preference.
    PyTorch CUDA fp32 `linear()` on the stage2 block7 `1024 -> 256` MLP uses
    cuBLASLt algo 20 with `1 MiB` workspace and split-K=2.
  - With
    `CUBLASEW_LT_WORKSPACE_BYTES=1048576 CUBLASEW_LT_MIN_ALIGNMENT_BYTES=4
    T2_SCVAE_CUBLASLT_BIAS_GEMM=1`, isolated `stage2_b7` from exact PyTorch
    post-LN input drops from `2.10e-5` to `3.81e-6`.
  - Enabling that path globally gives the best pre-output max seen so far
    (`6.63e-6`) but still produces `7.63e-5` full-output max, because the
    remaining feature-direction error is amplified by the exact output layer.
- Remaining blocker for `<1e-5` full-output:
  - Starting from exact PyTorch `stage2` features, stage3 + final output is
    `4.58e-5` max. Stage3 ConvNeXt block output is only `4.29e-6`, but stage3
    C2S `conv2` reaches a one-ULP `7.63e-6` post-conv drift that is later
    amplified.
  - Full-run stage2 output is `2.86e-5` max vs PyTorch. The largest stage2 C2S
    final drift comes through the repeated skip branch (`post_updown_x`), not
    the stage2 C2S `conv2` result alone.
  - Full-run localization with the current best flags shows `stage2_block7`
    jumps from `1.14e-5` after `stage2_block6` to `2.48e-5`; inside that block
    the MLP output is the visible amplifier. Forcing cuBLASLt only on
    `stage2_block7` MLP2 reduces the block/stage2 max (`1.05e-5`/`1.53e-5`)
    but worsens the final logits (`8.77e-5`), so it remains a probe only.
  - Full-run stage3 C2S `conv2` is the last large amplifier: `post_conv2`
    reaches `3.24e-5` with the packed sparse-conv path. Gather-GEMM,
    no-bias cuBLASLt, direct sparse conv, precise math, and CPU gather-map
    variants did not beat packed sparse conv on this smoke.
  - Per-site sparse-conv probes (`T2SD_SPARSE_*`) confirm the same result when
    only stage3 C2S `conv2` is changed: gather, cuBLASLt, and direct modes tie
    or worsen the old `6.87e-5` full-output max. Stage3 ConvNeXt sparse-conv
    per-block probes also tie or worsen the current best.
  - Per-stage C2S no-affine Welford probes did not help the final output.
    Stage2 reduces stage3 C2S `post_conv2` locally (`3.05e-5`) but projects to
    a worse `1.11e-4` final max; stage3 ties the local C2S max and worsens the
    output to `7.63e-5`.
  - `T2_SCVAE_CUBLASLT_GEMM=1` adds a no-bias cuBLASLt route for sparse-conv
    packed-row GEMMs. It is useful for algorithm probes but did not reduce the
    stage3 C2S `conv2` max on the Fujisan `N=128` smoke.
  - `T2_VERIFY_PROJECT_OUT=1` in `verify_shape_decoder` loads
    `output_layer.weight` on the host and, for 64-channel intermediate
    comparisons, reports the max final-logit delta implied by
    `(cuda_feats - ref_feats) @ output_w.T`. With the current best full-run
    flags and `T2SD_STOP_PRE_OUTPUT=1`, raw pre-output max is `7.87e-6` but the
    projected output-delta max is `6.63e-5` at `row=2835 col=5`, matching the
    observed final-output failure direction.
  - Projection-screened probes on the Fujisan `N=128` smoke did not beat the
    current best pre-output direction: `T2SD_CUBLASLT_MLP=2:7:2` raised the
    projected max to `7.91e-5`, `T2SD_WELFORD_AFFINE_LN=2:7` to `7.23e-5`,
    `T2_SCVAE_FINAL_WELFORD_LN=1` to `7.33e-5`, and final serial LN modes
    tested at `7.79e-5`/`8.40e-5`.
  - The projection diagnostic now also prints the largest channel
    contributions for the worst projected row/output. Current best full-run
    pre-output has top terms `c17=+1.49e-5`, `c34=-1.47e-5`,
    `c19=+1.16e-5`, `c18=+1.13e-5`, `c40=+8.58e-6`, and
    `c59=+8.22e-6`, so the failure is still a broad feature-direction drift.
  - Exact-start projected localization: starting from saved PyTorch stage2
    output (`--start-stage 3`) gives projected pre-output max `3.90e-5`;
    starting from saved PyTorch stage1 output (`--start-stage 2`) gives
    `5.49e-5`; full-run gives `6.63e-5`. Stage3 remains a real contributor,
    but the full-run worst direction is seeded upstream.
  - Stage3 C2S variants from exact stage2 did not improve projected max:
    gather-GEMM for stage3 C2S conv2 tied baseline at `3.90e-5`, C2S norm2
    Welford was `4.02e-5`, and C2S norm1 Welford was `4.23e-5`.
  - cuBLASLt MLP2 stage3 probes from exact stage2 lowered local projected max
    for blocks 0/1 (`3.28e-5`/`3.37e-5`), but the same block0 change worsened
    full-run projected max to `8.10e-5` and full final output to `8.01e-5`.
    Full-run stage3 MLP2 block 1/2/3 projected maxes were also worse
    (`8.47e-5`, `8.39e-5`, `8.04e-5`).
  - Full-run stage2 MLP2 cuBLASLt probes for blocks 0..7 also worsened the
    projected pre-output direction; sampled maxes ranged from `7.48e-5` to
    `1.05e-4`. Stage2 block6/7 and stage3 block0/1 MLP0 cuBLASLt probes were
    numerically unchanged from baseline.
  - Tracing the persistent worst rows through the saved PyTorch coordinate
    hierarchy shows several share the same stage0 C2S parent row `110`, but
    ancestor-row raw errors remain diffuse (`~2e-5` to `4e-5` L2 per row) and
    do not expose one bad channel or one obviously broken child slot.
  - Additional full-output projection probes: group21/group23 FMA worsen to
    `6.87e-5`; group29/group31 FMA tie the current best `6.48e-5`.
    Existing double-accumulation GEMM paths are worse (`9.16e-5` with bias
    GEMMs, `8.77e-5` with all GEMMs), and pair32 output is `6.87e-5`.
  - Automated Python `subprocess` sweeps hit `cuInit failed (100)` on this
    machine even when direct verifier invocations work, so continue using direct
    shell verifier runs for CUDA sweeps.
  - Direct sparse-conv double-accumulation probes with the projected metric did
    not help: stage3 C2S conv2 direct mode2 raised projected max to `7.35e-5`,
    stage2 C2S conv2 direct mode2 to `9.58e-5`, and stage3 C2S conv1 direct
    mode2 to `7.67e-5`.
  - Stage-wide affine Welford LayerNorm probes all worsened the projected
    pre-output direction: stage0 `7.94e-5`, stage1 `1.02e-4`, stage2
    `8.78e-5`, and stage3 `8.47e-5`.
  - Multi-site cuBLASLt MLP2 combination probes were enabled to test whether
    individually bad directions cancel. Sampled combinations still worsened:
    `2:7:2,3:0:2` -> `7.64e-5`, `2:1:2,3:0:2` -> `8.53e-5`,
    `3:0:2,3:1:2` -> `7.91e-5`, and `2:1:2,2:7:2,3:0:2` -> `8.55e-5`.
  - `T2SD_STOP_AFTER_C2S_OP=stage:7` returns `to_subdiv` logits as `[N,8]`
    before subdivision thresholding. All four stage logits are coordinate-exact and
    well away from topology flips on the worst rows. Minimum positive margins:
    stage0 `0.428`, stage1 `0.126`, stage2 `0.00242`, stage3 `0.00253`;
    largest negative logit in stage3 is `-0.000290`. The persistent worst rows
    have emitted-child logits far from zero, so subdivision topology is not the
    hidden failure mode.
  - Final no-affine LayerNorm epsilon is the only knob so far that reduces the
    full-output max. `T2_SCVAE_FINAL_LN_EPS=0.000009` with group25/FMA lowers
    final max from `6.48e-5` to `6.10e-5` and projected pre-output max from
    `6.63e-5` to `6.28e-5`. This is an output-max cancellation/parity tradeoff:
    PyTorch's `F.layer_norm` default is still `1e-5`. Tested `8.9e-6` and
    `9.1e-6` tie `6.10e-5`; `8.5e-6`, `9.25e-6`, and output groups 27/31 or
    cuBLASLt bias output are worse.
  - Additional full-output probes with the `9e-6` final-LN eps did not improve
    the max: CPU-built sparse gather maps, cuBLAS pedantic mode, cuBLAS for all
    ConvNeXt MLP GEMMs, and `sqrtf` LayerNorm all tied `6.10e-5`. Output
    group28/FMA also ties the max; `eps=9.2e-6` with group28 gives slightly
    lower rel L2 (`5.01e-7`) but still maxes at `6.10e-5`. Group24/26 and
    group28 with `eps=8.8e-6`/`9.6e-6` are worse (`6.48e-5` to `6.87e-5`).
    Serial/double final-LN mode3 with `eps=9.2e-6` is also worse (`6.48e-5`).
  - Exact-start downstream checks with the current best flags sharpen the
    remaining target. Starting from saved PyTorch `stage3_block3` and running
    stage3 C2S + final output gives `3.05e-5` max. Starting from saved PyTorch
    `stage2_block7` and running stage2 C2S + stage3 + final output gives
    `7.63e-5` with group25/`eps=9e-6`; group28 improves that exact-start path
    to `6.87e-5`, and group28 with `eps=9.2e-6` improves it to `6.48e-5`, but
    none of these improve the full-run max. Stage2 C2S alone from exact
    `stage2_block7` is only `5.72e-6` max, and exact-stage2 through final LN
    gives pre-output max `4.53e-6`; projected through `output_layer.weight`,
    that pre-output drift is `5.93e-5` at `row=3201 col=4`. This confirms the
    current bottleneck is a tiny C=64 drift direction, not a large local
    feature error.
  - Full-run group28/`eps=9.2e-6` output-order variants: mode1 ties the current
    `6.10e-5` max with rel L2 `5.02e-7`, while modes 2 and 3 are worse
    (`6.87e-5`).

Additional debug knobs for this path:

- `T2SD_STOP_AFTER_MLP_OP=stage:block:op`: stop inside the ConvNeXt MLP
  (`op=0` after `mlp.0`, `op=1` after SiLU, `op=2` after `mlp.2`).
- `T2SD_GROUP_MLP=stage:block:op:group`: force the grouped scalar-order GEMM
  for one MLP op; `block=-1` and `op=-1` are wildcards.
- `T2SD_CUBLASLT_MLP=stage:block:op`: force cuBLASLt bias GEMM for selected
  ConvNeXt MLP ops; wildcards are the same as above. Multiple sites can be
  separated with commas or semicolons.
- `T2_SCVAE_WELFORD_AFFINE_LN=1`: use a PyTorch CUDA vectorized-Welford-order
  affine LayerNorm for all affine SC-VAE LayerNorm sites.
- `T2SD_WELFORD_AFFINE_LN=stage:block`: use the affine Welford LayerNorm for
  one ConvNeXt block norm; `block=-1` is a stage wildcard.
- `T2SD_WELFORD_AFFINE_C2S=stage`: use the affine Welford LayerNorm for one
  C2S `norm1`; `stage=-1` applies to all C2S stages.
- `T2SD_WELFORD_NOAFFINE_C2S=stage`: use the no-affine Welford LayerNorm for
  one C2S `norm2`; `stage=-1` applies to all C2S stages.
- `T2SD_SPARSE_PACKED=stage:block:op:value`: override packed sparse conv for
  one site (`op=0` ConvNeXt conv, `op=1` C2S conv1, `op=2` C2S conv2; C2S
  uses `block=-1`).
- `T2SD_SPARSE_LT=stage:block:op`: force cuBLASLt no-bias GEMM for one sparse
  conv site; same site numbering and wildcards as `T2SD_SPARSE_PACKED`.
- `T2SD_SPARSE_DIRECT=stage:block:op:mode`: force direct sparse conv for one
  sparse conv site (`mode=1/2/3` matches `T2_SCVAE_DIRECT_CONV`).
- `T2SD_SPARSE_CUBLAS=stage:block:op`: force cuBLAS for one gather-GEMM sparse
  conv site.
- `T2_SCVAE_OUTPUT_GROUP_MODE=mode`: output-layer grouped GEMM traversal probe;
  `mode` bit 0 reverses group order and bit 1 reverses channel order inside
  each group.
- `T2_SCVAE_OUTPUT_GROUP_BIASINIT=1`: start the output-layer grouped GEMM
  accumulator from bias instead of adding bias after the dot product.
- `T2_SCVAE_OUTPUT_GROUP_TREE=1`: reduce output-layer group partials with a
  pairwise tree instead of sequential accumulation.
- `T2_SCVAE_FINAL_WELFORD_LN=1`: use a PyTorch CUDA vectorized-Welford-order
  no-affine LayerNorm only for the final `F.layer_norm` before `output_layer`.
- `T2_SCVAE_WELFORD_LN=1`: use the same Welford no-affine LayerNorm for all
  no-affine SC-VAE LayerNorm sites; useful for localization, not the current
  best full-output mode.
- `T2_VERIFY_PROJECT_OUT=1`: verifier-only diagnostic for 64-channel
  intermediate comparisons. It reports the max output-layer-weighted feature
  error without changing the CUDA decoder path.
- `T2_WRITE_SHAPE_OBJ=1` or `--write-shape-obj`: write the optional
  `<output>_shape.obj` debug sidecar. By default the CUDA harness writes only
  the requested final OBJ path to avoid duplicating million-vertex mesh output.

Small end-to-end smoke:

```bash
env XDG_RUNTIME_DIR=/run/user/1000 T2_SCVAE_CUBLAS=1 T2_SCVAE_PACKED_CONV=1 \
  ./cuda/trellis2/test_cuda_trellis2 \
  /mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
  /mnt/disk01/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
  unused.npy --image fujisan.jpg --dinov3 /mnt/disk01/models/dinov3-vitl16/model.safetensors \
  -n 1 --grid 32 \
  --stage2 /mnt/disk01/models/trellis2-4b/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
  --shape-dec /mnt/disk01/models/trellis2-4b/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors \
  --sparse-threshold -1000 --max-sparse 16 --s2-steps 1 \
  --s2-npy /tmp/trellis2_e2e_check_shape_s2_n16.npy \
  --occ /tmp/trellis2_e2e_check_shape_occ_n16.npy \
  -o /tmp/trellis2_e2e_check_shape_n16.obj
```

Measured in the Stage 3 smoke below: Stage 1 one step produced all-negative
occupancy, so the smoke uses top-N sparse selection. Stage 2 DiT one step
`586.8 ms`; CUDA SC-VAE shape decoder emitted `N=918,C=7`; shape OBJ written
to the requested output path. Use `--write-shape-obj` to also emit the
`<output>_shape.obj` debug sidecar.

Fix notes:

- Sparse gather-map lookup now uses the same Fibonacci-hash slot as
  `sp3d_hash_build`: `(key * 0x9E3779B97F4A7C15) & (capacity - 1)`. The old CUDA
  lookup used a different modulo hash and broke sparse-conv neighbor maps.
- ConvNeXt MLP activation is SiLU to match `SparseConvNeXtBlock3d`.
- Dense C2S child-slot order now matches upstream PyTorch
  `SparseChannel2Spatial`: for coords `(b,z,y,x)`, `subidx` bit 0 maps to `z`,
  bit 1 maps to `y`, and bit 2 maps to `x`. The old C/CUDA order matched the
  local C decoder but produced `7680` coord mismatches vs PyTorch on the
  `N=1 -> 4096` texture smoke.
- CUDA LayerNorm now has the two-pass no-affine variance path plus opt-in
  PyTorch vectorized-Welford-order paths for no-affine and affine SC-VAE
  LayerNorm. The final SC-VAE no-affine LayerNorm is bit-exact with PyTorch
  CUDA under `T2_SCVAE_FINAL_WELFORD_LN=1`.
- C2S norm, sparse conv, gather, conv2, residual repeat, subdivision-list
  synthesis, and sparse hash/index construction are CUDA-default paths.
  CPU fallback toggles remain for A/B and debugging.

Texture decoder verification, RTX 5060 Ti, synthetic `N=1` input. The PyTorch
reference tensors are persisted in `ref/trellis2/dumps/tex_scvae_tiny/`.

Generate PyTorch fp32 CUDA reference:

```bash
cpu/trellis2/.venv/bin/python ref/trellis2/gen_scvae_decoder_ref.py \
  --decoder /mnt/disk01/models/trellis2-4b/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors \
  --outdir ref/trellis2/dumps/tex_scvae_tiny \
  --prefix pytorch_fp32_cuda \
  --device cuda --dtype fp32 --dump-stages
```

Compare CUDA decoder to C and PyTorch references:

```bash
env XDG_RUNTIME_DIR=/run/user/1000 ./cuda/trellis2/verify_shape_decoder \
  /mnt/disk01/models/trellis2-4b/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors \
  ref/trellis2/dumps/tex_scvae_tiny/pytorch_fp32_cuda_input_slat.npy \
  ref/trellis2/dumps/tex_scvae_tiny/pytorch_fp32_cuda_input_coords.npy \
  --ref-feats ref/trellis2/dumps/tex_scvae_tiny/pytorch_fp32_cuda_output_feats.npy \
  --ref-coords ref/trellis2/dumps/tex_scvae_tiny/pytorch_fp32_cuda_output_coords.npy
```

Result: C, PyTorch, and CUDA all emit `N=4096,C=6` with coordinate mismatch `0`.

| Reference | Corr | Rel L2 | Max abs |
|-----------|------|--------|---------|
| C fp32 vs CUDA fp32 | 1.00000000 | 2.91e-7 | 1.19e-6 |
| PyTorch fp32 CUDA vs CUDA fp32 | 1.00000000 | 4.33e-7 | 2.18e-6 |

The texture decoder has no `to_subdiv` head, so the CUDA path uses dense C2S:
`1 -> 8 -> 64 -> 512 -> 4096`.

Accuracy/speed knobs tested against the persisted fp32 CUDA PyTorch reference:

| CUDA mode | Corr | Rel L2 | Max abs | Notes |
|-----------|------|--------|---------|-------|
| default F32 tiled GEMM + precise math | 1.00000000 | 4.332e-7 | 2.176e-6 | target `<1e-5` reached |
| `T2_SCVAE_SERIAL_LN=4` | 1.00000000 | 5.166e-7 | 2.056e-6 | serial `1/sqrtf` block LayerNorm, slightly lower max |
| `T2_SCVAE_CUBLAS=1` | 1.00000000 | 2.106e-7 | 9.537e-7 | best speed path and best numeric match on this smoke |
| `T2_SCVAE_CUBLAS_BIAS_ONLY=1` | 1.00000000 | ~4e-7 | ~2e-6 | diagnostic dense-only cuBLAS path |
| `T2_SCVAE_CUBLAS_MLP=1` | 1.00000000 | ~4e-7 | ~2e-6 | diagnostic MLP-only cuBLAS path |
| `T2_SCVAE_CUBLAS_SPARSE=1` | 1.00000000 | ~4e-7 | ~2e-6 | diagnostic sparse-conv-only cuBLAS path |

`T2_SCVAE_CUBLAS_PEDANTIC=1` produced the same result as normal cuBLAS on this
smoke. The `<1e-5` max-abs target is reached; best verified max is
`9.5367432e-7` with `T2_SCVAE_CUBLAS=1`.

Debug stop hooks for the texture SC-VAE parity path:

- `verify_shape_decoder --start-stage S --start-block B`: start CUDA from a
  saved intermediate sparse tensor instead of running `from_latent`.
- `T2SD_START_AT_OP=stage:block:op`: start a ConvNeXt block from a saved op
  input (`op=1` post sparse conv, `op=2` post LayerNorm).
- `T2SD_STOP_AFTER_OP=stage:block:op`: ConvNeXt op stop, where `op=0`
  post sparse conv, `op=1` post affine LayerNorm, and `op=2` post MLP.
- `T2SD_STOP_AFTER_C2S_OP=stage:op`: C2S op stop, where `op=0` pre-conv1,
  `1` post-conv1, `2` post-updown `h`, `3` post-updown residual `x`,
  `4` pre-conv2, `5` post-conv2, `6` final residual output, and `7`
  `to_subdiv` logits before thresholding.
- `T2SD_STOP_AT_START=1`: return the uploaded start tensor before any op.
- `T2_VERIFY_DUMP_FEATS=/path.npy T2_VERIFY_DUMP_COORDS=/path.npy` dumps the
  CUDA tensor emitted by the verifier stop point.

Stage dumps localize the remaining texture error to sparse ConvNeXt/C2S
accumulation order. All stage comparisons have coordinate mismatch `0`.

| Stop point | Shape | Rel L2 | Max abs |
|------------|-------|--------|---------|
| `from_latent` | `N=1,C=1024` | 1.06e-7 | 1.19e-7 |
| `stage3` / C2S final | `N=4096,C=64` | 3.50e-7 | 2.29e-4 |
| `pre_output` | `N=4096,C=64` | 2.47e-7 | 1.43e-6 |
| `output` | `N=4096,C=6` | 4.332e-7 | 2.176e-6 |

Root cause for the previous `~2e-5` output drift was the CUDA runner's C-side
F16->F32 upload for sparse-conv weights: F16 subnormal values were flushed to
signed zero in `t2_upload_conv_transposed`. Several stage3 ConvNeXt dot products
depend on those subnormals under heavy cancellation. Preserving subnormals
reduces the stage3 block3 local post-LayerNorm max error from `1.47e-4` to
`1.43e-6`, and the full output max from `1.97e-5` to `2.18e-6`.

The shipped PyTorch config sets `use_fp16=true`. Against that official-dtype
reference, CUDA fp32 is still coordinate-exact but the numeric error is looser
(`rel_L2=6.10e-4`, max abs `0.00253`); the opt-in CUDA F16 path is similar
(`rel_L2=6.94e-4`, max abs `0.00282`). The fp32 reference is the current tight
numeric parity target.

Stage 3 end-to-end smoke:

```bash
env XDG_RUNTIME_DIR=/run/user/1000 T2_SCVAE_CUBLAS=1 T2_SCVAE_PACKED_CONV=1 \
  ./cuda/trellis2/test_cuda_trellis2 \
  /mnt/disk01/models/trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors \
  /mnt/disk01/models/trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
  unused.npy --image fujisan.jpg --dinov3 /mnt/disk01/models/dinov3-vitl16/model.safetensors \
  -n 1 --grid 32 \
  --stage2 /mnt/disk01/models/trellis2-4b/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors \
  --shape-dec /mnt/disk01/models/trellis2-4b/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors \
  --stage3 /mnt/disk01/models/trellis2-4b/ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors \
  --tex-dec /mnt/disk01/models/trellis2-4b/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors \
  --sparse-threshold -1000 --max-sparse 16 --s2-steps 1 --s3-steps 1 \
  --s2-npy /tmp/trellis2_e2e_check_s2_n16.npy \
  --occ /tmp/trellis2_e2e_check_occ_n16.npy \
  -o /tmp/trellis2_e2e_check_n16.obj
```

Measured output on the rebuilt CUDA binary: Stage 1 one step `4.56 s`, Stage 2
one step `586.8 ms`, Stage 3 one step `292.3 ms`; shape SC-VAE emitted
`N=918,C=7`, texture SC-VAE emitted `N=65536,C=6`, and vertex-colored OBJ was
written to `/tmp/trellis2_e2e_check_n16.obj`. The harness unloads shape SC-VAE
weights before loading texture SC-VAE weights to reduce peak VRAM. UV texture
atlas export is still available with `T2_PBR_TEXTURE_MAP=1`, but the default is
vertex colors because the chart packer is still experimental.

HIP DiT now uses BF16 WMMA GEMM + BF16 WMMA flash-attn (commit 4d59a1e).
1.73 s/step × 12 steps × 2 (CFG fwds) ≈ 42 s. Falls back to scalar F32 GEMM
on non-gfx12 archs (or with `T2_WMMA=0`), 143.7 s/step there. Next-tier
optimization: BF16-act × FP8-weight WMMA (qimg/Flux.2 pattern, ~2× more).

### tex_dec (Stage 3 texture decoder) cold/warm — RX 9070 XT, knight r512 dump

Bench harness: `rdna4/trellis2/bench_tex_dec_coldwarm.py` (PyTorch ROCm 7.2.2 ref;
`T2_TEX_REPS=N` env on `test_hip_tex_dec` for HIP). Cold = rep 0 of a fresh process,
warm = median of rep 1+ (steady state). Both ship-AOT/cache-warmed: PyTorch's
`~/.flex_gemm/autotune_cache.json` + `~/.triton/cache/` populated; HIP's Triton AOT
hsacos baked at build time and `<cache_dir>/triton_prep_*.bin` pre-populated.

| Path | Cold rep 0 | Warm steady |
|---|---|---|
| **PyTorch ROCm 7.2.2 (Triton spconv via FlexGEMM)** | **~2000 ms** (1.9–3.7 s) | **125 ms** |
| **HIP (`test_hip_tex_dec`, T2_TEX_REPS=2)** | **329 ms** | **144 ms** |

- **Cold**: HIP wins by **6–11×**. PyTorch's 2 sec cold is Triton kernel JIT-from-cached-AST
  + first-call hipBLASLt plan compile, even with both caches warm. We ship Triton AOT
  hsacos and disk-persist spconv prep tensors (sorted/vk/vkseg) so neither cost lands.
- **Warm**: PyTorch wins by **~19 ms** (~13%). Their autotuned wide-tile Triton spconv at
  large N (178k, 822k voxels) outperforms our hipBLASLt MT64x64x32 fallback at stage 3;
  per-call host overhead is also slightly lower.
- Implication: for one-shot mesh texturing the HIP path is sub-second to first decoded
  voxel; for batched decode (>1 mesh per process) PyTorch warm 125 ms beats HIP 144 ms.

Bench setup (one-time, see `rdna4/trellis2/pyproject.toml` for full recipe):

```bash
cd /mnt/disk1/work/gemm/trellis2/rdna4/trellis2
uv venv --python 3.12 .venv
uv pip install -e ".[torch-rocm722]"
BUILD_TARGET=rocm GPU_ARCHS=gfx1201 uv pip install --no-build-isolation -e \
    /mnt/disk1/work/gemm/trellis2/cpu/trellis2/trellis2_repo/o-voxel
BUILD_TARGET=rocm GPU_ARCHS=gfx1201 uv pip install --no-build-isolation -e \
    /mnt/disk1/work/gemm/trellis2/rdna4/trellis2/deps/FlexGEMM
.venv/bin/python bench_tex_dec_coldwarm.py --reps 4
```

AMD/RDNA4 drift debug mode:

- HIPRTC defaults to `-ffast-math` for speed. Set `HIP_RUNNER_PRECISE_MATH=1`
  to compile runtime kernels without fast-math when comparing against PyTorch.
- tex_dec SiLU now uses `expf` instead of the approximate `__expf` intrinsic, so
  the precise-math mode has a meaningful effect on activation drift.
- The native tex_dec Makefile depends on `kernels/hip_tex_dec_kernels.h` and
  adds `-Ikernels`; `verify-tex-dec` builds the CPU-correct PBR oracle from
  `tools/build_pbr_cpu_oracle.py`.
- `NO_HIPBLASLT=1` builds the native runner with stub hipBLASLt entry points.
  Use it with `T2_TEX_BLASLT=0` for scalar/precise drift bisection on machines
  that do not have hipBLASLt development headers.

Tightest AMD comparison mode:

```bash
env HIP_RUNNER_PRECISE_MATH=1 \
    T2_LIN_NAIVE=1 T2_TEX_BLASLT=0 T2_TEX_TRITON=0 \
    T2_TEX_KLIN_TRITON=0 T2_TEX_WMMA_SPCONV=0 \
  make -C rdna4/trellis2 verify-tex-dec \
    NO_HIPBLASLT=1 \
    TEX_DEC_WEIGHTS=/mnt/disk01/models/trellis2-4b/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors \
    REF_DIR=/tmp/tex_knight_r512_v2
```

The machine running this needs ROCm development headers/libraries visible under
`ROCM_PATH` for the default hipBLASLt build. With `NO_HIPBLASLT=1`, the scalar
drift path still needs the HIP runtime and HIPRTC libraries available at runtime.

## Weight Files

| Component | Path | Size | Format |
|-----------|------|------|--------|
| DINOv3 ViT-L/16 | `dinov3-vitl16/model.safetensors` | 1.2 GB | F32 (timm) |
| Stage 1 DiT | `trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` | 2.5 GB | BF16 |
| Stage 1 Decoder | `trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors` | 141 MB | F16/F32 mixed |

Download:
```bash
uvx --from huggingface_hub hf download timm/vit_large_patch16_dinov3.lvd1689m model.safetensors \
  --local-dir /mnt/disk1/models/dinov3-vitl16
uvx --from huggingface_hub hf download microsoft/TRELLIS.2-4B \
  ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors --local-dir /mnt/disk1/models/trellis2-4b
uvx --from huggingface_hub hf download microsoft/TRELLIS-image-large \
  ckpts/ss_dec_conv3d_16l8_fp16.safetensors --local-dir /mnt/disk1/models/trellis-image-large
```

## Source Files

### Vulkan implementation (`vulkan/trellis2/`)

| File | Lines | Description |
|------|-------|-------------|
| `vulkan_trellis2_runner.h` | ~175 | Public API (Stages 1-3, DINOv3, decoders) |
| `vulkan_trellis2_runner.cc` | ~2500 | DINOv3 + generic DiT + Stages 1-3 sampling + GPU shape decoder + KV cache |
| `test_vulkan_trellis2.cc` | ~400 | Test harness: full/dit-only/decode-only/encode/stage2/stage3/shape-dec/tex-dec modes |

**Shaders** (`vulkan/shaders/trellis2/` — DiT + decoder):

`adaln_f32`, `channel_layernorm_3d_f32`, `conv3d_k3_f32`, `dinov3_prepend_tokens_f32`,
`gated_add_f32`, `layernorm_noaffine_f32`, `modulation_f32`, `pixel_shuffle_3d_f32`,
`rope_2d_dinov3_f32`, `rope_3d_f32`, `self_attn_tiled_f32`, `silu_inplace_f32`,
`sparse_conv3d_f32` (hash-table neighbor lookup, uint64 keys),
`split_qkv_chunk_f32`, `split_kv_chunk_f32`, `timestep_embed_cossin_f32`

**Shared shaders** (`vulkan/shaders/` — used by all Vulkan models):

`gelu_f32`, `gemm_f32_f32` (tiled 64×16), `layernorm_f32`, `layerscale_add_f32`,
`patch_embed_f32`, `rms_norm_f32` (per-head QK norm), `cross_attn_f32`

### CUDA implementation (`cuda/trellis2/`)

| File | Lines | Description |
|------|-------|-------------|
| `cuda_trellis2_runner.h` | 65 | Public API |
| `cuda_trellis2_runner.c` | ~2500 | DINOv3, Stage 1/2/3 DiT forward, decoder forward, weight loading |
| `cuda_trellis2_kernels.h` | ~280 | CUDA kernels: adaLN, gated_add, modulation, rope_3d, rms_norm_perhead, conv3d, channel_layernorm_3d, pixel_shuffle_3d, timestep_embed_cossin |
| `cuda_trellis2_ops.h` | ~300 | Kernel launch wrappers |
| `test_cuda_trellis2.c` | ~860 | Test harness: full/stage1/stage2/stage3/decode modes, .npy I/O, mesh export |
| `verify_decoder.c` | ~90 | Standalone decoder verification |
| `verify_stage2.c` | ~90 | Stage 2 DiT single-step verifier |
| `verify_stage3.c` | ~90 | Stage 3 DiT single-step verifier |
| `Makefile` | ~30 | gcc build for runner + verifiers (no nvcc, NVRTC runtime compilation) |

### HIP/ROCm implementation (`rdna4/trellis2/`)

| File | Lines | Description |
|------|-------|-------------|
| `hip_trellis2_runner.h` | ~65 | Public C API |
| `hip_trellis2_runner.c` | ~1200 | DiT forward, decoder forward, weight loading, kernel launches |
| `hip_trellis2_kernels.h` | ~500 | HIPRTC kernel strings: adaLN, GroupNorm, conv3d, RoPE, flash-attn, etc. |
| `test_hip_trellis2.c` | ~450 | Test harness: full/dit-only/decode-only/dump-blocks modes |
| `verify_dit.c` | ~150 | DiT single-step CPU vs GPU comparison |
| `verify_decoder.c` | ~180 | Decoder CPU vs GPU comparison |
| `Makefile` | ~100 | gcc build (no hipcc, HIPRTC runtime compilation via `rocew`) |

### CPU implementation (`common/`)

| File | Lines | Description |
|------|-------|-------------|
| `cpu_compute.h` | ~640 | Shared primitives: attention, cross-attention, LayerNorm, GELU, RoPE |
| `dinov3.h` | ~900 | DINOv3 ViT-L/16 encoder |
| `trellis2_dit.h` | ~1000 | DiT blocks, F32 GEMM, adaLN, QK RMSNorm, 3D RoPE |
| `trellis2_stage1.h` | ~250 | Euler flow sampling, CFG, guidance interval |
| `trellis2_ss_decoder.h` | ~620 | Conv3D, GroupNorm, pixel_shuffle_3d, ResBlock3d |

### PyTorch reference (`ref/trellis2/`)

| File | Description |
|------|-------------|
| `gen_stage1_ref.py` | Full Stage 1 pipeline: DINOv3 → DiT 12-step CFG → decoder → .npy outputs |
| `gen_scvae_decoder_ref.py` | Shape/texture SC-VAE PyTorch refs with pure-PyTorch sparse-conv backend |
| `dump_dit_intermediates.py` | Per-block hidden state dump for layer-by-layer comparison |
| `make_comparison.py` | Compare .npy pairs (corr, max_diff, relative L2) |

Requires: `torch>=2.3` with ROCm/CUDA, `transformers`, `safetensors`, `numpy`, `Pillow`.
Existing venv: `/mnt/disk1/work/gemm/diffusion/ref/flux2_klein/.venv` (ROCm 7.2, torch 2.11).
TRELLIS.2 model code: `cpu/trellis2/trellis2_repo/` (cloned from github.com/microsoft/TRELLIS.2).

### Test harness & verification (`cpu/trellis2/`)

| File | Description |
|------|-------------|
| `test_trellis2.c` | CPU test harness (full/stage1/decode/encode/mesh modes) |
| `run_official_e2e.py` | Official TRELLIS.2 pipeline with timm→transformers DINOv3 |
| `run_pytorch_ref.py` | PyTorch reference sampling |
| `dump_per_block.py` | Per-block hidden state dump for comparison |
| `dump_block0_detail.py` | Detailed block 0 intermediate dump |
| `verify_decoder.py` | Decoder output comparison |
| `verify_trellis2.py` | Weight inspection, .npy comparison, occupancy visualization |

## Bugs Found & Fixed

| Bug | Symptom | Root Cause | Fix |
|-----|---------|-----------|-----|
| Timestep embed cos/sin order | Block 0 output wrong | HY3D kernel outputs [sin,cos], TRELLIS.2 expects [cos,sin] | New `timestep_embed_cossin_f32` kernel |
| RoPE pair convention | Per-token values wrong | Split-half pairs (x[j], x[j+N]) vs complex-pairs (x[2k], x[2k+1]) | Rewrite RoPE to use consecutive pairs |
| Input NCDHW transpose | Input embedding wrong | Missing channel-first to token-major transpose | Add transpose in `cuda_trellis2_run_dit` |
| QKV split convention | Q/K/V data garbled | Used per-head interleaved split, should be standard chunk | New `split_qkv_chunk_f32` kernel |
| QKV bias missing | QKV projection off by bias | `sa_qkv_b` not loaded or passed to GEMM | Load and pass bias |
| Final LayerNorm | Output scale wrong | Missing no-affine LN before `out_layer` | Add `layernorm_noaffine_f32` |
| **t_emb buffer overflow** | **Block 1+ explodes 23x** | **d_temb allocated 256 floats, MLP writes 1536** | **Allocate D floats** |
| CFG rescale x0_pos | Wrong std ratio | Used v_uncond for x0_pos, should use v_cond | Save original v_cond before combining |
| CFG rescale std | Minor numerical diff | RMS instead of proper torch.std (mean-subtracted, Bessel) | Use double-precision proper std |
| **Decoder norm type (CUDA)** | **Decoder output flipped** | **Used per-spatial ChannelLN, actual is GroupNorm(32)** | **GroupNorm kernel (groups=32)** |
| DINOv3 final norm | Features range differs | TRELLIS.2 uses unparameterized LN, not learned norm | Apply plain LN in encoder |
| **Stage 2 F16 GEMM** | **All GEMM output garbage** | **`use_f32_gemm` not set when loading Stage 2 standalone** | **Set flag in `cuda_trellis2_load_stage2`** |
| **Stage 2 scratch overflow** | **Cross-attn produces 5e26** | **`scratch[1]` sized for self-attn N only, not cross-attn ctx_len*2D** | **Include `ca_kv_gemm_sz` in max** |
| **CUDA CFG KV cache thrash** | **Cross-attn KV rebuilt for every conditioned/unconditioned pass** | **Single cache slot was invalidated at public API entry and could not hold both CFG conditions** | **Two condition slots keyed by model id + condition hash; zero condition uses slot 1** |
| **Vulkan: DiT QK RMSNorm gamma** | **11/12 heads use wrong gamma → 0% occupancy** | **`rms_norm_f32.comp` used `w[i]` (head 0 gamma) instead of `w[h*head_dim+i]`** | **Fix gamma index (commit 4306224)** |
| **Vulkan: DINOv3 patch embed dispatch** | **Tokens 256-1023 all zeros → features diff 14.75** | **`opPatchEmbed` dispatched 1024 workgroups instead of 4096 (shader covers 256 elements/group)** | **`(n_patches*dim+255)/256` workgroups (commit 4306224)** |
| **Vulkan: GELU approximation** | **DINOv3 features max diff 0.061 vs official** | **`gelu_f32.comp` used tanh approximation; DINOv3 specifies exact erf GELU** | **Replace with A&S erf polynomial, max intrinsic error ~1.5e-7 (commit 70c3b70)** |
| **Stage-1 DiT fp16 range** | **Full latent cosine 0.727; single-step 0.976; one output element sign-flipped (ref 2.11 vs −0.06)** | **PyTorch runs SS-DiT in bf16 (range == f32); our default fp16 MMA (max ~65504) clips a hot intermediate in the dense 4096-token grid** | **Default Stage-1 DiT to F32 GEMM (`t2_dit_use_f16(r,0)`); single-step 0.9998, latent 0.989. `T2_DIT_F16=1` opts back into fp16. Sparse Stage 2/3 verified fp16-faithful (0.9999==F32), stay fp16** |
| **SS coords missing max-pool** | **e2e fed Stage 2 ~21037 voxels @res64 → shape decoder cascaded 21037→…→2.56M rows → OOM/illegal-address, no OBJ** | **`test_cuda_trellis2` thresholded the 64³ logits directly; PyTorch does `decoded>0` then `max_pool3d(2)`→32³ before `argwhere`** | **Max-pool 64³→32³ (block-max OR) then emit res-32 coords; 3515 vs ref 3548 (98.8% overlap). Also fixes Stage-2/3 RoPE (res-64 coords doubled every rope angle)** |
| **Stage-1 e2e noise layout (verification artifact)** | **Stage-1 latent cosine ≈ −0.012 (garbage)** | **A prior session fed `--noise` as token-major `[4096,8]`, but `run_dit` consumes channel-major `[8,4096]` (it transposes internally, mirroring `x.view(B,C,-1).permute(0,2,1)`) → double-transpose scramble. The "`--noise` is token-major" note was false** | **Feed the raw channel-major `02_ss_noise.npy` directly; compare saved latent directly to `03_ss_latent[0]` (no readback "correction"). No code change** |
| **Shape/tex decoder default output GEMM = all zero** | **e2e with no `T2_SCVAE_*` flags → shape decoder `[N,7]` output ALL zero → FDG `0/N` intersected edges → `0 triangles`, no OBJ** | **The fall-through cuBLAS/plain output-GEMM path leaves the small `[N,out_ch]` output zero; only the grouped F32 kernel writes it. The resume baseline always set `T2_SCVAE_OUTPUT_GROUP=25`, masking this** | **Default `group` to 25 in the output projection (was 0). e2e now emits a mesh with no flags (287k tris); full resume flags give 2.9M tris ≈ PyTorch 3.14M** |
| **FDG vertex-offset transform** | **Mesh vertex positions slightly off** | **Harness applied plain `sigmoid` to `feats[0:3]`; `fdg_vae.py` uses `(1+2·margin)·sigmoid − margin` (margin=0.5)** | **Match PyTorch: `2·sigmoid − 0.5` in `test_cuda_trellis2.c` post-process** |
| **Texture decoder dense ×8 subdivision** (FIXED 2026-05-29) | **Full textured e2e: tex decoder cascaded to 16.97M voxels → OOM (`free=0`) → vertex colors all `(0,0,0)` black** | **The tex decoder checkpoint has NO `to_subdiv` head (shape decoder has 8); C fell back to dense ×8 subdivision. PyTorch drives the tex decoder from the SHAPE's res-512 structure → 1.468M voxels (= shape mesh)** | **FIXED: shape decode RECORDS its per-C2S-stage pruned subdivision into a runner-resident `subdiv_plan[8]`; tex decode REPLAYS it (= PyTorch `guide_subs`). Both share the same res-32 coords/order → 3515→16473→73245→320002→1.378M, identical to shape. See "Full TEXTURED e2e" below** |
| **Texture decode + shape decode OOM with DiTs resident** (FIXED 2026-05-29) | **Even with correct subdivision, the SHAPE decode OOM'd at its finest level (`scratch[8]` 338 MB, `free=0`) — all 3 DiTs (Stage 1 F32 = 5.3 GB) + decoders stayed resident** | **C kept every stage on the GPU; PyTorch frees them via `pipeline.low_vram`. `--max-gpu-layers` streamed DiT blocks but was never the lever** | **`cuda_trellis2_unload_dit_stages(r)` (frees Stage 1/2/3 weights + KV cache, idempotent) called by the harness after all 3 latents are host-side, before decoding → frees ~10.5 GB (3100→13024 MB free)** |
| **PBR vertex colors x↔z swap + thin-shell miss** (FIXED 2026-05-29) | **Only ~11% of mesh verts got color (rest black) even after the subdivision fix** | **`trellis2_pbr.h` stored the field as (col3,col2,col1) but the FDG mesh + sampler resolve lookups to (col1,col2,col3) → x↔z swap (hit only on the x==z diagonal). Plus FDG dual-vertex offsets ∈ [−0.5,1.5] floor() off the thin shell** | **Store the field (col1,col2,col3) → trilinear hit 99.7%; nearest-populated-voxel snap fallback (`t2pbr_nearest`, radius 4, `T2_PBR_NO_SNAP=1` disables) covers the 0.3% → 100% covered / 99.7% non-black** |

### Stage-1 e2e parity (2026-05-29)

The "garbage Stage-1 output" tracked in `cuda/trellis2/resume.md` had three independent
causes, now all resolved on the RTX 5060 Ti (sm_120):

1. **Noise layout (verification-side, not a code bug).** Stage 1 is channel-major
   `[C=8,N=4096]` end-to-end: `cuda_trellis2_run_dit` reads `x_t[ch*N+pos]` and transposes
   to token-major internally (mirroring PyTorch `sparse_structure_flow.py:228`); the Euler
   loop and the SS decoder are all channel-major. The prior session's token-major conversion
   and readback "correction" were the scramble. Fix: feed `02_ss_noise.npy` raw.

2. **Coords extraction (real bug, caused the OOM crash).** See table. `max_pool3d` 64³→32³
   before `argwhere` is required, both to avoid the shape-decoder blow-up and because
   Stage-2/3 RoPE uses absolute coord values (res must match PyTorch's 32).

3. **Stage-1 precision (real quality bug).** Default the dense Stage-1 DiT to F32; the
   sparse Stage-2/3 DiTs stay fp16 (measured bit-faithful). Env: `T2_DIT_F16=1` forces fp16
   everywhere, `T2_DIT_F32=1` forces F32 everywhere.

**New verifier:** `verify_stage1.c` (`make -C cuda/trellis2 verify_stage1`) does a single
SS-DiT forward at t=0.5 vs `02b_ss_dit_step_velocity.npy` (pass `02_ss_noise.npy` +
`01_dinov3_cond_512.npy` raw; it calls `run_dit` with `t_raw=0.0005` so the embedder sees
0.5). `verify_stage2`/`verify_stage3` now also default to `t_raw=0.0005` (the 06b/10b dumps
are direct `t=0.5` calls, not sampler steps; optional `argv[6]` overrides). Measured
single-step correlations vs PyTorch: Stage 1 (F32) 0.9998, Stage 2 0.9999, Stage 3 0.9999.
Full Stage-1 latent (F32, 12 steps): cosine 0.989, 17133 positive voxels @64³ (ref 17303),
3515 coords @res32 (ref 3548, 98.8% set overlap).

**End-to-end (image → Stage1 F32 → coords → Stage2 fp16 → shape decoder → FDG):** now
completes (no OOM) and writes a real mesh. With the resume's validated SC-VAE flags
(`T2_SCVAE_CUBLAS=1 T2_SCVAE_PACKED_CONV=1 T2_SCVAE_CPUAVX_LN=1 T2_SCVAE_FINAL_LN_EPS=0.000009
T2_SCVAE_OUTPUT_GROUP=25 T2_SCVAE_OUTPUT_GROUP_FMA=1`): 1.39M verts / 2.90M tris
(PyTorch ref 1.47M / 3.14M). With NO flags (post-fix default group=25): a coarser-but-valid
1.39M verts / 287k tris. The other SC-VAE flags (PACKED_CONV / CUBLAS / CPUAVX_LN) sharpen the
decoder features (more detected surface crossings) — they remain opt-in tuning, as in the
shape-decoder parity work. The cuBLAS/plain output-GEMM fall-through path that produced the
all-zero `[N,out_ch]` output is still buggy (root cause unconfirmed); the group=25 default
sidesteps it — worth fixing separately.

**Full TEXTURED e2e (`--stage3 --tex-dec`) — WORKS (2026-05-29).** The full
image→Stage1→Stage2→shape-dec→Stage3→tex-dec→PBR pipeline now produces a colored OBJ
(1,377,823 verts / 2,826,154 tris, 99.7% non-black). Three fixes were needed:

1. **Texture-decoder subdivision replay (the real bug).** The tex decoder checkpoint has no
   `to_subdiv` head, so the C runner had no per-stage subdivision logits and fell back to dense
   ×8 → 16.97M voxels. Fix: the SHAPE decode now records its per-C2S-stage *pruned* subdivision
   (parent idx, child slot, child coords) into a runner-resident `t2_subdiv_stage subdiv_plan[8]`;
   the TEXTURE decode replays it instead of subdividing densely — exactly PyTorch's
   `decode_tex_slat(tex_slat, guide_subs=subs)`. Both decoders are driven from the same res-32
   sparse coords in the same order, so the recorded parent indices stay valid. Result:
   3515→16473→73245→320002→1.378M, byte-identical to the shape cascade. (A standalone tex decode
   with no prior shape decode falls back to dense, with a logged warning.)

2. **DiT-stage VRAM offload.** Even with correct subdivision, the SHAPE decode itself OOM'd at
   its finest level (`scratch[8]` 338 MB with `free=0`) because all three DiTs stayed resident —
   Stage 1 is **F32 = 5.3 GB** alone, and Stage 2/3 add ~2.6 GB each. New
   `cuda_trellis2_unload_dit_stages(r)` frees Stage 1/2/3 weights + the cross-attn KV cache
   (idempotent) and the harness calls it after all three latents are host-side, before decoding
   → frees ~10.5 GB (3100 → 13024 MB free). This is the C analogue of PyTorch's
   `pipeline.low_vram` CPU offload; `--max-gpu-layers` (DiT block streaming) was never the lever.

3. **PBR vertex-color sampling** (`common/trellis2_pbr.h`). The field was stored in
   (col3,col2,col1) order while the FDG mesh builder + vertex sampler resolve lookups to
   (col1,col2,col3) — an x↔z swap that only hit on the x==z diagonal (~11% of verts colored).
   Storing the field (col1,col2,col3) raised the trilinear hit rate to 99.7%; a
   nearest-populated-voxel snap fallback (`t2pbr_nearest`, Chebyshev radius 4) covers the 0.3%
   of FDG dual vertices whose offset (∈[−0.5,1.5]) floors off the thin surface shell → 100%
   covered, 99.7% non-black. (`T2_PBR_NO_SNAP=1` disables the fallback.)

**Performance (RTX 5060 Ti, 12-step CFG, ~3515 sparse voxels):** Stage-1 DiT now defaults to
**TF32 tensor cores** (`ops.use_tf32_gemm` → `cublasew_gemm_f32_tf32_rowmajor_nt`,
`CUBLAS_COMPUTE_32F_FAST_TF32`). The F32 DiT GEMMs previously fell through to a plain tiled
`gemm_f32` CUDA-core kernel; TF32 keeps the full f32 exponent range (no fp16 clipping) at a
fraction of the cost, and its 10-bit mantissa is *more* precise than PyTorch's bf16 — single-step
cosine 0.99980 vs plain-f32 0.99980 (Δ 7e-7, quality-neutral). The math mode is set per cuBLAS
call, so the decoder's exact-f32 path is untouched. Env opt-out `T2_DIT_NO_TF32=1`.

| Stage | plain F32 | TF32 (default) |
|-------|-----------|----------------|
| Stage 1 (dense 4096-token DiT, 12-step CFG) | 148.6 s | **37.5 s (3.96×)** |
| Stage 2 (sparse shape DiT, F16 MMA) | 38.6 s | 38.4 s |
| Stage 3 (sparse tex DiT, F16 MMA) | 23.2 s | 23.0 s |
| Shape decode | 13.6 s | 13.2 s |
| Texture decode | 13.0 s | 13.5 s |
| **GPU total** | **~237 s** | **~125 s (1.9×)** |

Remaining: Stage-2/3 FULL-sampler parity (only single-step verified); lower the ~12.7 GB DiT
*loading* peak (all stages load upfront) via lazy per-stage load.

### DiT profiling → Stage-2/3 cuBLAS-TF32 + modulation fix — DiT 100→74 s (2026-05-30)

`nsys` kernel profiling of the full Stage-2 sampler (`verify_stage2_full`) pinned the DiT-forward
cost: **`gemm_f16_f32` 46.7%, `attn_mma_hd128_f32` 34.8%, `modulation_f32` 10.1%**, rope 4.4%, the
rest ~4%. Two of these were inefficiencies, not inherent cost:

1. **`modulation_f32` was launched as a SINGLE block** (`grid=1`, 256 threads) for a
   `[9216,1536]·[1536]` adaLN mod matvec — using 1 of ~50 SMs, with uncoalesced stride-1536 reads
   → 6.4 ms/call. Rewrote it **warp-per-row** (one warp per output row, coalesced dot + warp-shuffle
   reduction, grid = `ceil(out_dim/8)` blocks). ~10% off **every** stage (Stage 1 38.4→34.2 s),
   numerically equivalent (Stage-2 cosine 0.985372→0.985397, a benign reduction-order delta).

2. **Stage 2/3 used the hand-written F16-MMA `gemm_f16_f32`** — profiling showed it was the #1 cost
   *and* slower per-voxel than Stage 1's cuBLAS TF32 despite fewer tokens. Switched the Stage-2/3
   default from F16-MMA to **F32 + cuBLAS TF32** (matches Stage 1; `T2_DIT_F16=1` restores the old
   MMA path). cuBLAS TF32 = **1.36×** on the Stage-2 sampler (1924→1417 ms/forward), equivalent
   accuracy (cosine 0.985372→0.985343). VRAM (F32 5.3 GB vs F16 2.5 GB) is fine under the default
   lazy per-stage load (one DiT resident). Combined with (1): **Stage-2 1924→1243 ms/forward (1.55×)**.

End-to-end DiT (RTX 5060 Ti, 12-step CFG, ~3519 sparse voxels):

| Stage | before | after | speedup |
|-------|--------|-------|---------|
| Stage 1 (dense 4096) | 38.4 s | **34.2 s** | 1.12× (modulation only — already cuBLAS) |
| Stage 2 (sparse shape) | 38.6 s | **24.9 s** | **1.55×** |
| Stage 3 (sparse tex) | 23.2 s | **14.9 s** | **1.56×** |
| **DiT total** | **100.2 s** | **74.0 s** | **1.35× (−26 s)** |

Mesh output unchanged in quality (1.47 M verts, 3.22 M tris, 99.7% trilinear / 100% covered); the
final voxel count shifts <1% (1.462 M→1.472 M) from the F16-vs-TF32 Stage-2 numerical difference,
both equally ~0.985 vs PyTorch. **Next levers** (untouched): `attn_mma_hd128_f32` (34.8%, the
materialized O(N²) self-attn) and the two decoders (~28 s combined, dominated by the `c2s 128→64`
upsample to 1.47 M voxels at ~5 s and the 329 K-voxel stage-3 ConvNeXt at ~3.3 s).
`verify_stage2_full` now prints `>>> Sampler loop: … ms/forward`.

### Attention K/V shared-memory staging — DiT 74→48 s, byte-identical (2026-05-30)

The `attn_mma_hd128_f32` kernel (35% of the DiT forward) is already a hand-written FlashAttention
(online softmax, `mma.sync.m16n8k16` tensor cores, O(N) memory) — but each block runs **4 warps over
different query rows and the SAME KV**, and every warp re-read K and V straight from global memory
each 16-token tile. That's 4× redundant K/V traffic, and profiling-by-arithmetic (the kernel runs at
~5 TFLOPS, far below compute peak) said it was memory-bound. Fix: **stage each 16-token K/V tile into
shared memory once per block** (as f32, so every downstream `cvt`/MMA/softmax op is byte-identical),
then all 4 warps read from shared. The coalesced staging load replaces the scattered per-warp global
reads. One required change: the per-warp `if (qb>=q_len) return;` early-out had to go (a returned warp
would hang the others at the new `__syncthreads`) — OOB query rows are already guarded in the Q load
and the output write, so those warps just do harmless wasted compute.

**Result: 1.54× on the whole DiT, output byte-identical** (Stage-2 sampler cosine 0.985411 — exactly
the pre-staging value). The win is uniform because all three stages share the kernel, and is *largest*
for dense Stage 1 (N=4096, the heaviest attention):

| Stage | original | +mod+cuBLAS-TF32 (`1b8253c`) | **+attn K/V staging** |
|-------|----------|------------------------------|------------------------|
| Stage 1 (dense 4096) | 38.4 s | 34.2 s | **21.9 s** |
| Stage 2 (sparse shape) | 38.6 s | 24.9 s | **16.3 s** |
| Stage 3 (sparse tex) | 23.2 s | 14.9 s | **9.8 s** |
| **DiT total** | **100.2 s** | 74.0 s | **48.0 s** |

**Cumulative DiT (after attention staging): 100.2 → 48.0 s = 2.09×** (modulation warp-per-row +
Stage-2/3 cuBLAS-TF32 + attention K/V staging). Per-forward Stage-2: 1924 → 806 ms (2.39×).

### RoPE reparallelization — DiT 48→44 s (2026-05-30)

`rope_3d_f32` ran `for (h = threadIdx.x; h < n_heads; h += blockDim.x)`, so only **`n_heads`=12 of
the 256 threads** per block were active (the rest idle) and each rotated a whole head's 128 dims
serially — at ~1.38 ms for a [N,1536] read+write (~0.1 ms memory floor) it was ~13× off, the same
low-occupancy bug class as `modulation_f32`. Reparallelized to **one thread per (head, axis, freq)
complex-pair** (all 256 threads active, coalesced). Per-element-independent, so no race; the math is
the same, though restructuring lets the compiler contract `re*c - im*s` into FMAs differently → a
benign ~3e-5 cosine shift (Stage-2 0.985411 → 0.985379, still at the ~0.985 floor vs PyTorch — the
same class as the TF32/bf16 reassociations we already accept).

| Stage | +attn-staging | **+rope-opt** |
|-------|---------------|----------------|
| Stage 1 | 21.9 s | **20.0 s** |
| Stage 2 | 16.3 s | **14.8 s** (806 → 731 ms/forward) |
| Stage 3 |  9.8 s | **8.9 s** |
| **DiT total** | 48.0 s | **43.7 s** |

**Cumulative DiT this session: 100.2 → 43.7 s = 2.29×** (modulation warp-per-row + Stage-2/3
cuBLAS-TF32 + attention K/V staging + RoPE reparallelization); per-forward Stage-2 1924 → 731 ms
(2.63×). Mesh stays valid (1.40 M verts, 3.05 M tris, 99.7% trilinear / 100% covered); the count
drifts 1.47 M → 1.40 M as the accumulated benign numerical differences shift near-threshold
subdivision decisions — equivalent quality, both ~0.985 vs PyTorch. Remaining lever: the two decoders
(~28 s, sparse-conv-bound — gather/pack/GEMM/scatter on up to 1.4 M voxels; no single hot kernel,
a deeper effort) are now the largest cost.

### Decoder packed-conv pack caching (2026-05-30) — decoder 27.4 → 12.6 s = 2.18×

Profiling the decoder (`-t cuda` API trace) showed the time was not in any GPU kernel but in
**thousands of synchronous host memory ops** — the per-block `t2_sparse_conv_pack_build`. The
"packed" sparse-conv path (`T2_SCVAE_PACKED_CONV=1`, default) builds 27 per-offset src/dst index
arrays by doing `N*27` CPU-side hash lookups and up to 54 HtoD uploads. That pack is a **pure
function of (coords, hash)** and is *identical* across every ConvNeXt block at a resolution level,
yet it was being rebuilt from scratch for each block.

Fix: cache the pack in the runner keyed on `(coords ptr, N)`. A new level (the c2s `conv2` that
produces the child voxels) rebuilds; every ConvNeXt block and the c2s `conv1` at a level reuse the
cache. Invalidated at decode start and freed at decode end / `cuda_trellis2_free`. The per-stage
ConvNeXt collapse (shape decoder):

| ConvNeXt stage | blocks | before | after | speedup |
|---|---|---|---|---|
| stage 1 (res 256) | 16 | 1762 ms | 485 ms | 3.6× |
| stage 2 (res 128) | 8  | 2016 ms | 338 ms | 6.0× |
| stage 3 (res 64)  | 4  | 3251 ms | 279 ms | **11.6×** |

**Decoder total 27.4 → 12.6 s = 2.18×** (shape 13.9→6.4 s, tex 13.5→6.2 s). Output is
**byte-identical** (1,403,042 verts, 3,048,684 tris, 99.7% trilinear, 100% covered) — pure caching,
zero numerical change. **Combined GPU pipeline this session: DiT+decoder 127.4 → 56.3 s = 2.26×.**
The remaining decoder cost after this change was the c2s `conv2` pack build on the 1.47 M-child
level (~3.75 s shape + 3.67 s tex); see the next section for the GPU-side builder that removes it.

### Decoder GPU pack build from gather_map (2026-05-30) — c2s conv2 4.0 → 0.6 s

The level cache cannot help the c2s `conv2` pack because it is the first sparse conv after each
subdivision and therefore sees a new coordinate level. The old builder still did `N*27` CPU hash
lookups and uploaded up to 54 index arrays. New kernel `sparse_pack_from_gather_map_f32` builds the
same packed `(src_idx,dst_idx,M)` lists directly on GPU from the already-built `[N,27]` gather map.
`T2_SCVAE_CPU_PACK_BUILD=1` keeps the old CPU builder available for A/B.

A/B on the T.png verification dumps (`08_shape_slat_denorm_feats` + `05_ss_coords`) is
**byte-identical** versus the CPU-pack path: feature `max_abs=0`, `rel_L2=0`, coord mismatches `0`.
Focused shape-decoder C2S timings, CPU-pack → GPU-pack:

| C2S level | CPU pack | GPU pack | speedup |
|---|---:|---:|---:|
| 1024 → 512 | 202.7 ms | 102.9 ms | 2.0× |
| 512 → 256  | 337.8 ms | 126.7 ms | 2.7× |
| 256 → 128  | 963.6 ms | 222.5 ms | 4.3× |
| 128 → 64   | 3993.0 ms | 600.9 ms | 6.6× |

Full textured e2e with default GPU-pack completed successfully: shape output `N=1,403,042`, OBJ
`1,403,042` verts / `3,048,684` tris, texture decoder replayed all four shape subdivisions, and PBR
coverage was `99.7%` trilinear / `100%` covered. In that full run, finest-level c2s timings were
shape `558.7 ms` and texture `509.2 ms`.

### Decoder GPU subdivision + sparse hash/index (2026-05-30) — focused shape decoder 11.5 → 6.5 s

The next host bottleneck was C2S subdivision and sparse-index setup. The shape decoder now uses a
stable two-pass GPU subdivision path: `c2s_count_subdiv_f32` counts kept child slots, the CPU computes
the prefix offsets, and `c2s_write_subdiv_stable_f32` writes `idx`, `subidx`, and child coords in the
same parent/child order as the old CPU path. That stability matters: an earlier atomic compaction
changed row order and broke byte-level parity. The sparse hash table is also built on GPU via
`sparse_hash_insert_coords_f32`, then the existing gather-map kernel runs from that device hash.

Defaults and A/B toggles:

- Packed sparse conv is default-on. Use `T2_SCVAE_NO_PACKED_CONV=1` or `T2_SCVAE_PACKED_CONV=0` to
  opt out.
- `T2_SCVAE_CPU_SUBDIV=1` restores host subdivision synthesis.
- `T2_SCVAE_CPU_HASH_BUILD=1` or `T2_SCVAE_CPU_GATHER_MAP=1` restores host hash/gather-map setup.
- `T2_SCVAE_CPU_PACK_BUILD=1` restores the legacy host pack builder and now also keeps the CPU hash
  available for that builder.
- `T2_TIMING=1` prints load, sparse-index, pack-build, and subdivision timings.
- DiT loaders skip unused GPU-to-CPU block copies in the default full-GPU mode. Set
  `T2_DIT_KEEP_CPU_BLOCKS=1` when debugging or using block streaming.

Focused A/B on the T.png verification dumps (`08_shape_slat_denorm_feats` + `05_ss_coords`) is
**byte-identical** versus the CPU-subdivision/hash/pack fallback: feature `max_abs=0`, `rel_L2=0`,
coord mismatches `0`. Cached shape-decoder wall time on RTX 5060 Ti:

| Path | wall time | finest sparse index | finest C2S |
|---|---:|---:|---:|
| CPU subdiv/hash/pack fallback | 11.54 s | 190.6 ms | 4149.8 ms |
| GPU default | **6.52 s** | **38.3 ms** | **392.8 ms** |

Full textured e2e with the new defaults completed in `real 87.50`: Stage 1/2/3 sampler
`19.9/14.8/8.9 s`, shape output `N=1,403,042`, OBJ `1,403,042` verts / `3,048,684` tris, and PBR
coverage `99.7%` trilinear / `100%` covered.

### GPU BF16-to-F32 DiT weight upload (2026-05-30) — full e2e 87.5 → 64.0 s

The F32 DiT path still loaded BF16 checkpoints by converting each tensor to F32 on the CPU, then
uploading the expanded 4-byte weights. The loader now uploads raw BF16 and expands to the exact same
F32 values on GPU with `t2_cast_bf16_to_f32`, cutting host-to-device traffic in half for Stage 1/2/3
DiTs. `T2_CPU_BF16_UPLOAD=1` restores the old CPU conversion path for A/B.

Measured RTX 5060 Ti load times:

| DiT stage | CPU BF16→F32 upload | GPU BF16→F32 upload |
|---|---:|---:|
| Stage 1 | 8.42 s | **1.03 s** |
| Stage 2 | 8.39 s | **0.89 s** |
| Stage 3 | 8.91 s | **0.94 s** |

Verification is byte-identical where the old dumps are available: Stage 1 latent, Stage 2 raw slat,
texture coords, and texture features all compare `max_abs=0`. Focused verifier metrics are unchanged
(`verify_stage1` cosine `0.99980245`, Stage 2 full sampler cosine `0.985379`, Stage 3 full sampler
cosine `0.999980`). Full textured e2e with cached kernels is now `real 64.04`, with the same
`1,403,042` verts / `3,048,684` tris and PBR `99.7%` trilinear / `100%` covered.

### GPU F16 SC-VAE upload + sparse-conv transpose (2026-05-30) — full e2e 64.0 → 57.1 s

The shape and texture SC-VAE checkpoints are F16-heavy. Their previous F32 load path converted dense
weights on CPU and also transposed sparse-conv weights from `[out,27,in]` to `[27,out,in]` on CPU.
The loader now uploads raw 16-bit tensors and expands on GPU with `t2_cast_f16_to_f32`; sparse-conv
weights use `t2_conv3d_transpose_f16_to_f32` or `t2_conv3d_transpose_bf16_to_f32` to combine upload,
conversion, and transpose. CPU fallbacks remain available with `T2_CPU_F16_UPLOAD=1`,
`T2_CPU_BF16_UPLOAD=1`, and `T2_CPU_SCVAE_CONV_UPLOAD=1`.

Measured load times:

| Decoder | CPU conversion/transpose | GPU upload/transpose |
|---|---:|---:|
| Shape SC-VAE | 3.72 s | **0.31 s** |
| Texture SC-VAE | 3.73 s | **0.31 s** |

Focused shape-decoder output is **byte-identical** versus the CPU conversion/transpose fallback:
feature `max_abs=0`, `rel_L2=0`, coord mismatches `0`. Cached focused decoder wall time is now
`3.10 s` (was `6.52 s` after GPU subdivision/index). Full textured e2e is now `real 57.13`, and the
full-run dumps remain byte-identical to the prior GPU-DiT-load run for Stage 1, Stage 2, texture
coords, and texture features. Mesh/PBR output is unchanged: `1,403,042` verts / `3,048,684` tris,
PBR `99.7%` trilinear / `100%` covered.

The same F16 upload helper is also used by the dense Stage 1 occupancy decoder. Its load time is small
but now visible under `T2_TIMING`: CPU conversion `0.28 s`, GPU expansion `0.05 s`. `verify_decoder`
metrics are unchanged, and the final full e2e is `real 56.85` with byte-identical dumps versus the
previous SC-VAE-load run.

### Output-side tail trimming (2026-05-30) — full e2e 57.0 → 55.4 s

Coarse `T2_TIMING` brackets now cover the post-sampler pipeline: Stage-1 structure decode, sparse
coord extraction, shape/texture SC-VAE decode totals, FDG mesh extraction, PBR field build, texture
dump, OBJ writes, and program total. The first full profile showed the hidden tail was mostly output:
`fdg_write_shape_obj 1.63 s`, `pbr_sample_vertices 0.39 s`, and `pbr_write_colored_obj 2.36 s`.

The CUDA harness now skips the redundant `<output>_shape.obj` sidecar by default. The requested final
OBJ is still written in all paths; set `T2_WRITE_SHAPE_OBJ=1` or pass `--write-shape-obj` to restore
the debug sidecar. This avoids a duplicate 110 MB OBJ and cuts the textured e2e to `real 55.47`.

The default vertex-colored OBJ path also streams PBR sampling directly into the writer, avoiding the
`n_verts * sizeof(t2_pbr_attr)` color array (`~32 MiB` at `1,403,042` verts). Wall time is flat
(`real 55.40`) because float formatting dominates the remaining final OBJ write, but the final OBJ
and all `.npy` dumps are byte-identical to the pre-stream no-sidecar run.

### Sparse DiT setup caching (2026-05-31) — repeated setup removed, output unchanged

The Stage 2/3 sparse DiT wrappers rebuilt sparse 3D RoPE tables and uploaded the conditioning tensor
for every sampler forward, even though coords and conditioning are constant within each stage. The
runner now caches sparse RoPE tables per model id keyed by `(coords hash, N, n_freqs)`, and the wrappers
skip the conditioning HtoD upload once the per-block cross-attention KV cache is already hot for the
same `(model_id, cond_hash, n_blocks)`.

This removes repeated CPU `sin/cos` work, RoPE HtoD uploads, and the hot-step conditioning upload.
The full textured e2e remains dominated by DiT math and OBJ formatting, so wall time is essentially
flat but slightly lower: `real 55.40/55.48 -> 55.35` (`T2_TIMING program_total 55252.514 ms`). Final
OBJ and saved dumps (`stage1`, Stage 2, `tex_coords`, `tex_feats`) are byte-identical to the previous
run.

### PyTorch-reference comparison of the full textured e2e (2026-05-29)

Dumped the CUDA intermediates (`--npy` Stage-1 latent, `--s2-npy` shape slat, new `--tex-npy`
tex voxels) and compared against the PyTorch ground truth in
`/mnt/disk01/models/trellis2-4b/verify-dumps/`:

| Quantity | vs PyTorch ref | Result |
|----------|----------------|--------|
| Stage-1 latent (`03_ss_latent`) | dense [8,16,16,16] | cosine **0.990** (TF32 default); **0.997** with `T2_DIT_BF16=1` (T.png) — ≈ PyTorch's own ~0.9975 backend floor; see bf16-block note + multi-prompt study below |
| Tex-voxel PBR feats (`15_tex_voxels`) | on 491,811 overlapping voxels, after `*0.5+0.5` | cosine **0.976** (raw 0.518 → confirms the `*0.5+0.5` scaling) |
| — per channel | R/G/B 0.82/0.81/0.81, metal 0.985, rough 0.974, alpha 0.996 | material maps near-exact |
| Tex-voxel count | 1,377,823 vs 1,468,404 | 93.8% |
| Final mesh verts | vs `13_mesh_vertices` | 93.8% |

**Strong numerical parity confirms the texture decode, TF32 Stage-1, and PBR scaling are all
correct.** Two characterized residual gaps, both pre-existing / inherent (not regressions):

1. **Voxel-coord overlap is only 35.7%** (491,811 of 1.38M) despite matching per-axis ranges
   (identity coord permutation is best; no transpose/flip improves it). This is genuine
   subdivision *divergence*: the Stage-1 latent differs from PyTorch's by relL2 0.14 (we run
   TF32, PyTorch runs bf16 — TF32 is *more* precise but not bit-identical), and that seeds four
   levels of thresholded C2S keep/drop decisions which compound. The PBR feats still agree
   (0.976) on the voxels that do coincide, so the texture decoder itself is correct.

2. **The final mesh is x↔z mirrored vs PyTorch.** The voxel coords are in PyTorch's frame
   (`(c1,c2,c3)` = grid `(d,h,w)`), but the reference `o_voxel.flexible_dual_grid_to_mesh` maps
   the first grid dim `c1 → world x`, whereas the C `t2_fdg_to_mesh` deliberately treats coords
   as `(z,y,x)` and maps `c1 → world z` (a conscious, internally-consistent convention). bbox:
   CUDA x∈[−0.46,0.47]/z∈[−0.5,0.5] vs ref x∈[−0.5,0.5]/z∈[−0.45,0.47] (Y matches exactly). The
   C textured mesh is valid and self-consistent (PBR coverage 99.7%); it just doesn't bit-match
   PyTorch's vertex array. To align exactly: in `t2_fdg_to_mesh` set `verts[0]←coords[0]`,
   `verts[2]←coords[2]` (un-reverse) **and** flip the emitted triangle winding (a reflection
   inverts it), **and — only as part of this same coupled change** — flip the PBR field storage
   to `(c3,c2,c1)` so the sampler stays aligned. **Never flip the PBR storage alone**: with the
   current `(z,y,x)` mesh builder the correct storage is `(col1,col2,col3)` (see the convention
   note just below); `(c3,c2,c1)` on its own drops coverage 99.7% → 10.8%.
   Deferred — it needs visual/normal validation (the centroid-normal test is ~50% on this
   non-convex mesh, so handedness can't be confirmed headless).

**PBR coordinate convention — why the field stores `(col1,col2,col3)`, and the PyTorch-validation
trap (verified 2026-05-30).** The field storage order hinges on one fact: **C and the whole
codebase use `(b,z,y,x)` sparse coords** (col1=z, col2=y, col3=x) — see `cuda_trellis2_runner.c`
("child coords (b,z,y,x)"), `trellis2_shape_decoder.h` (`z=coords[i*4+1]`, `x=coords[i*4+3]`), and
every sam3d SubMConv. **PyTorch instead uses `(b,x,y,z)` (col1=x)** — the two are *x↔z swapped*.
In `15_tex_voxels.coords.npy` (PyTorch) col1 is the full-range axis and equals `mesh.vertices[0]`
(world x); in the C `tex_vox.coords` col1 is *also* full-range but it is world **z** (the C mesh
has world-z full-range / world-x medium — the mirror of PyTorch's mesh).

Since the FDG mesh builder + sampler resolve a vertex to `hash(iz, iy, ix)` with `iz`←world-z and
`ix`←world-x, the field must store **`(col1,col2,col3)`** (axis0/z←col1, axis2/x←col3). That is the
C-internally-consistent order → **99.7% coverage**; the swapped `(col3,col2,col1)` → **10.8%**
(hits only the x==z diagonal).

- **TRAP: do NOT validate the PBR mapping against `15_tex_voxels.coords.npy`.** It is in PyTorch's
  `(b,x,y,z)` frame, so matching "col1 == mesh.vertices[0]" against it yields the *opposite*
  (wrong-for-C) order. This is exactly what misled commit `c040789` "fix X/Z axis swap" →
  `(col3,col2,col1)`; at that time the C tex decoder was *also* broken (all coords collapsed to
  `(0,0,0)`), so real C validation was impossible. `origin/trellis2` still carried that
  `(col3,col2,col1)`; the 2026-05-30 merge **kept ours** (`(col1,col2,col3)`).
- **Decisive validation = real C artifacts together.** Dump the C mesh (`-o cuda.obj`) and C tex
  voxels (`--tex-npy`) from one e2e run, then build the field from `tex_vox.coords` and sample at
  the OBJ vertices: `(col1,col2,col3)` → 99.7%, `(col3,col2,col1)` → 10.8%. The on-by-default
  `t2_pbr_sample_vertices` diagnostic prints this live (`T2 PBR: … covered N (X%)`; `T2_PBR_QUIET=1`
  silences, `T2_PBR_NO_SNAP=1` disables the nearest-voxel fallback).

### Stage-2/3 FULL-sampler parity + a Stage-2 `guidance_rescale` bug (2026-05-30)

The Stage-2 (shape SLat) DiT single step was already verified (`verify_stage2` vs
`06b_slat_dit_step_velocity`: **corr 0.99995**). New `verify_stage2_full` closes the loop on
the **full 12-step sampler**: it feeds PyTorch's *exact* inputs — initial noise
`06_shape_slat_noise_feats`, coords `06b_slat_dit_step_coords`, positive cond
`06b_slat_dit_step_cond`, zero neg-cond — through the same `FlowEulerGuidanceInterval` loop the
e2e harness runs, then compares the raw output to `07_shape_slat_raw_feats`. This isolates the
sampler from upstream Stage-1/coord divergence (PyTorch's `neg_cond` is confirmed *zero* —
`dump_ground_truth.py` line 271 — so the harness's zero uncond is correct).

**Bug found + fixed: Stage-2 `guidance_rescale` was 0.7, should be 0.5.** `model_root/pipeline.json`
`shape_slat_sampler` specifies `guidance_rescale=0.5` (0.7 is *Stage 1's* `sparse_structure_sampler`
value); `test_cuda_trellis2.c` had hardcoded `s2_cfg_rescale=0.7f`. This is the same bug the RDNA4
side already fixed (commit `71d27ae` "fix SLAT guidance_rescale 0.7→0.5"), but that fix only touched
`rdna4/*`, so the CUDA harness still carried it through the 2026-05-30 merge. Effect on Stage-2
full-sampler parity vs `07`:

| `cfg_rescale` | cosine | relL2 |
|---|---|---|
| **0.5** (pipeline.json, fixed) | **0.985** | 0.171 |
| 0.7 (old harness) | 0.946 | 0.325 |

The residual 0.015 (0.985, not ~1.0 like the single step) is the **same fp16-vs-PyTorch-bf16 per-step
compounding** characterized for Stage 1: the single forward is corr 0.99995, but 12 recursive Euler
steps integrate the per-step f16/TF32-vs-bf16 difference. A Stage-2 "bf16-block" mode (analogous to
`T2_DIT_BF16` for Stage 1) was then implemented as `T2_SLAT_BF16` (see the dated note below) — it
improves but does **not** close the gap, confirming a cross-implementation floor. All six sampler params (steps=12,
rescale_t=3.0, strength=7.5, rescale=0.5, interval=[0.6,1.0], σ_min=1e-5) now match pipeline.json.
Stage 3 (`tex_slat_sampler`) has `guidance_strength=1.0` → CFG fully disabled, so its
`guidance_interval=[0.6,0.9]`/`guidance_rescale=0.0` are moot and the harness is already correct there.

**Stage-3 full sampler is essentially exact — and it explains Stage-2's residual.** `verify_stage3_full`
feeds PyTorch's exact Stage-3 inputs (noise `09_tex_slat_noise_feats`, re-normalized shape concat_cond
`10_tex_concat_cond_feats`, coords `10b_tex_dit_step_coords`, image cond) through the 12-step loop —
each step's DiT input is the `[N,64]` concat of the current state with concat_cond — and compares to
`11_tex_slat_raw_feats`: **cosine 0.999980, relL2 6.4e-3.** Stage 3 runs with `guidance_strength=1.0`
(CFG fully disabled), so it integrates only the *raw* per-step f16-vs-bf16 difference → near-perfect.
Stage 2's larger 0.015 residual is therefore explained: its `guidance_strength=7.5` **amplifies** that
same per-step difference ~7.5× before it compounds over 12 Euler steps. So neither sampler has a logic
error — Stage-2's gap is CFG-amplified precision (the `T2_SLAT_BF16` mode below tests this directly), and the
no-CFG Stage-3 path is verified bit-close. Summary: single-step S2 0.99995 / S3 ~1.0; full-sampler
**S2 0.985, S3 0.99998.**

### Stage-2 `T2_SLAT_BF16` bf16-block mode — measured the floor: 0.985 → 0.986, not 0.999 (2026-05-30)

Implemented the predicted bf16-block port for the sparse SLAT flows as `T2_SLAT_BF16` (default OFF;
in `run_sparse_dit_forward`, the exact mechanism as Stage 1's `T2_DIT_BF16`: `use_bf16_gemm=1` +
`bf16_round=1`, x_emb/out stay f32; requires F32-loaded weights, i.e. run with `T2_DIT_F32=1` so the
per-GEMM F32→bf16 cast recovers the original bf16 exactly). Goal was Stage-2 full-sampler cosine 0.999.
**Result: it does not get there. The gap is a genuine cross-implementation floor, not a fixable bug.**

`verify_stage2_full` (cosine vs `07_shape_slat_raw_feats`):

| Stage-2 precision mode | full-sampler cosine |
|---|---|
| F16 weights + f16-MMA (default) | 0.985372 |
| F32 weights + TF32 (`T2_DIT_F32=1`) | 0.985343 |
| F32 weights + **bf16-block** (`T2_DIT_F32=1 T2_SLAT_BF16=1`) | **0.986218** |

Three independent facts pin the residual to an *irreducible* per-step cross-implementation difference,
amplified by CFG — **not** a sampler-logic or precision bug we can chase to 0.999:

1. **The sampler math is provably correct.** Verified line-by-line against the TRELLIS.2 PyTorch source
   (`flow_euler.py`, `classifier_free_guidance_mixin.py`, `guidance_interval_mixin.py`): the t-schedule
   `t·rt/(1+(rt−1)·t)`, the interval check on the *rescaled* t, the CFG combine
   `strength·v_cond+(1−strength)·v_uncond`, and the CFG-rescale std-matching all match. The one subtle
   point — PyTorch's CFG-rescale std is `x_0.std(dim=list(range(1,ndim)))`, and `SparseTensor` has
   `ndim=2`/`shape=[B,32]`, so `reduce(dim=[1])` means `feats.mean(dim=1)` *then* `segment_reduce` over
   voxels per batch = a per-sample (for B=1, **global**) std, exactly what our harness computes (modulo a
   negligible Bessel `/(n−1)` vs `/n`). No bug.
2. **It is not block precision.** F16 (0.985372) ≈ F32+TF32 (0.985343); matching PyTorch's bf16 round
   points (bf16-block) only nudges it to 0.986218 (+0.0008). If the gap were weight/GEMM precision, F32
   would beat F16 and bf16-block would jump like Stage 1 did (0.989→0.997). It does neither — bf16-block
   helps *less* here than for Stage 1, which means most of the Stage-2 gap is something bf16-block does
   not touch (the materialized-f16-MMA-vs-flash-attn per-step difference, identical in F16 and F32).
3. **The magnitude is exactly what CFG=7.5 predicts.** Single-step forward is cosine 0.99995 → per-step
   relL2 ≈ 0.010. Full-sampler relL2 ≈ 0.173 (cosine 0.985). That is 17.3× growth — squarely within the
   `guidance_strength=7.5` × √(9 guided steps) = 22.5 upper bound. Stage 2 amplifies the faithful-but-
   not-bit-exact per-step velocity ~7.5× in `(v_cond−v_uncond)` and compounds it over the 9 guided Euler
   steps. Stage 3 (no CFG, strength=1.0) integrates the *raw* per-step diff → 0.99998, the unamplified floor.

**Conclusion: 0.999 for Stage 2 is not reachable without bit-exactly replicating PyTorch's bf16+flash-attn
forward** (the same irreducible floor as Stage-1's ~0.9975, just larger because CFG amplifies it). The
best achievable here is `T2_SLAT_BF16` = **0.986**, which is the principled reference-matching mode and is
kept as default-OFF scaffolding (the e2e default stays F16 Stage-2/3 for speed; this is correctness-parity
tooling, not a quality lever). The decisive remaining confirmation — re-dumping `07` in this environment to
measure the reference's own backend reproducibility floor — is blocked by the sparse-flow ext deps the
PyTorch stub does not cover (`flex_gemm`/sparse-attn), so it is left as future work.

### Lazy per-stage DiT load — GPU peak 12.7 GB → 5.3 GB (2026-05-30)

The harness previously loaded all three DiTs **+** the shape decoder upfront, so the peak hit
~12.7 GB (only `3100/15850 MB free`) *before Stage 1 even ran* — dangerously tight on the 16 GB card
and a hard wall for larger models. But the stages run strictly sequentially (S1 coords → S2 slat →
S3 tex slat → shape decode → tex decode) and all inter-stage data is host-side, so only one DiT is
ever needed at a time. New per-stage unloads (`cuda_trellis2_unload_stage1/2/3`, factored out of
`unload_dit_stages`) let the harness **load-run-free** each stage in turn:

```
load Stage1 → run → unload_stage1 → load Stage2 → run → unload_stage2
→ load Stage3 → run → unload_dit_stages → load shape_dec → decode
→ unload_shape_dec → load tex_dec → decode    (tex_dec was already lazy)
```

Measured peak `free` per phase: S1 10530, S2 11260, S3 11260, post-unload 14868 MB → **peak usage
~5.3 GB (was ~12.7 GB), a 58% cut.** Safe because the cross-attn KV cache is keyed by
`(model_id, cond_hash)` (`runner.c:1656`), so each freshly-loaded stage recomputes its own KV instead
of reusing the freed stage's; and the free helpers (`CU_FREE`, `dit_model_free_gpu`) zero their
pointers, so the per-stage unloads + the bulk `unload_dit_stages` remain double-free safe.
**Verified computation-neutral:** the e2e Stage-1 latent is byte-identical to the pre-change run
(`max|diff|=0`), and the full image→colored-OBJ pipeline completes through every load/free transition
with no OOM. (Same run also shows the Stage-2 `guidance_rescale` fix lifting the e2e tex-voxel count
to **99.6%** of PyTorch's 1.468M, up from 93.8%.)

**Full bf16-block port — Stage-1 latent 0.9895 → 0.99739 (2026-05-29).** The gap is NOT a bug. The
FlowEuler sampler and every config value match `model_root/pipeline.json` exactly
(`guidance_interval=[0.6,1.0]` on the rescaled t, `guidance_strength=7.5`, `guidance_rescale=0.7`,
`rescale_t=5.0`, `steps=12`, `sigma_min=1e-5`; CFG combine, CFG-rescale std-match, Euler step and x0
formula all verified line-by-line vs `flow_euler.py` + the guidance mixins). The single-step DiT
forward is already cosine **0.9998** vs `02b_ss_dit_step_velocity`; the 12-step latent drifts by
integrating that per-step f32-vs-bf16 difference. PyTorch runs the 30 DiT blocks in **bf16**
(`sparse_structure_flow.py`: `convert_to(dtype)` on the blocks, `manual_cast(h)` in/out;
input_layer/t_embedder/out_layer stay f32). `T2_DIT_BF16=1` now replicates that trajectory:

- **`bf16_round`** (new in-place `t2_round_f32_bf16` kernel + `ops.bf16_round` + `t2_op_round_bf16`)
  rounds EVERY block-op OUTPUT to bf16 precision — 25 `RB()` calls in `run_dit_forward_generic`
  covering adaLN/LN/QKV/RMSNorm/RoPE/attn/out-proj/GELU/residuals.
- **`use_bf16_gemm`** makes the block matmuls true bf16 (W,X→bf16, `cublasew_gemm_bf16_bf16_f32` =
  `CUBLAS_COMPUTE_32F` accumulate). x_emb/out are forced f32 (bf16 suppressed around those two
  GEMMs); t_emb is rounded after the f32 timestep-MLP.

| Stage-1 latent vs `03_ss_latent` (12-step) | cosine | relL2 |
|---|---|---|
| TF32 (production default) | 0.98954 | 0.143 |
| bf16-GEMM only (matmul inputs, OLD negative) | 0.98848 | — |
| **full bf16-block (`T2_DIT_BF16=1`)** | **0.99739** | **0.072** |

Speed is ~parity with TF32 (~37 s Stage 1). **Negative sub-findings while chasing 0.999** (do not
re-attempt): (1) rounding the non-matmul WEIGHTS (norm γ / biases / mod_w) to bf16 is BIT-IDENTICAL —
the per-op output rounding already absorbs sub-bf16 weight differences; (2) rounding the attention
PROBS to bf16 made it WORSE (0.9974→0.9961) — PyTorch SDPA keeps the attention internals in f32, so
our f16-MMA probs (`attn_mma_hd128_f32`, 10-bit) are the closer match; (3) the bf16 GEMM already
accumulates in f32; (4) GELU is the tanh approximation on both sides (`gelu_f32` ==
`nn.GELU(approximate="tanh")`). The residual 0.0026 is **irreducible** — and the multi-prompt study
below proves *why*: it is the scale of PyTorch's *own* run-to-run/backend non-reproducibility of the
12-step bf16 latent. TF32 stays the **production default** (more precise, and — see below — more
*consistent* across images); bf16 is the PyTorch-matching **verification** mode.

> **Caveat (see "Multi-prompt study" below): the 0.99739 win is T.png-specific.** T.png is the only
> image with a canonical reference, and it is a favorable case. Across a 4-image sample, bf16 vs TF32
> is a wash, and both sit at the level of PyTorch's own backend reproducibility (~0.9975).

### Multi-prompt study + PyTorch reference reproducibility (2026-05-30)

Generated fresh PyTorch Stage-1 references for **3 new example images + T.png** and ran our pipeline
(TF32 and bf16) on each image's *identical* noise+cond. (PyTorch here needs `trimesh`+`easydict` and a
`meta_path` `MagicMock` stub for the uninstalled mesh/render CUDA exts `flex_gemm`/`cumesh`/
`nvdiffrast`/`o_voxel` — Stage-1's dense SS-flow + DINOv3 + `flow_euler` don't use them; a `--stage1-only`
flag was added to `dump_ground_truth.py`.)

**The PyTorch reference is backend-dependent.** Re-dumping **T.png with the same seed/noise/cond** as
the canonical 2026-05-26 dumps gives `02_ss_noise` **identical**, `01_cond` Δ=2e-5, but `03_ss_latent`
**cosine = 0.99752** vs canonical. Two fresh runs in *this* env are **bit-identical (1.0)**, so the
0.9975 gap is a systematic dense-attention **backend/version difference** (canonical likely used
flash_attn; this env falls back to sdpa), not run noise. **PyTorch's own bf16 12-step SS-flow latent is
therefore only reproducible to ~0.9975 across backends — so "0.999 vs the reference" is unreachable by
*any* implementation.**

| image | reference | ours TF32 | ours bf16 |
|---|---|---|---|
| T.png | canonical (older backend) | 0.98954 | **0.99738** |
| T.png | sdpa (fresh) | 0.99344 | **0.99764** |
| *PyTorch canon-vs-sdpa floor* | *(same image/noise/cond)* | | *0.99752* |
| img1 | sdpa (fresh) | **0.99489** | 0.98268 |
| img2 | sdpa (fresh) | 0.99513 | 0.99585 |
| img3 | sdpa (fresh) | 0.99301 | 0.99256 |
| **mean (4 imgs, sdpa refs)** | | **0.99412** | 0.99218 |

On **T.png** our bf16 lands *inside* the PyTorch cluster (0.9974/0.9976 ≈ the 0.9975 PyTorch-vs-PyTorch
floor) while TF32 sits *outside* it — bf16 wins. But across the **3 new images** (refs deterministic,
so these are real, not noise) it is a **wash**: bf16 swings 0.983–0.996 while TF32 holds 0.993–0.995,
and mean TF32 (0.9941) slightly edges bf16 (0.9922). Our CUDA attention is f16-MMA (neither flash_attn
nor sdpa), adding image-dependent divergence on top. **Takeaway: bf16 faithfully reproduces a
*particular* PyTorch bf16 run within PyTorch's own ~0.0025 backend ambiguity, but it is not a universal
accuracy win — TF32 is the more consistent and higher-precision default.**

## Next Steps

### CUDA
1. **Remaining decoder compute hot spots**: Profile the post-load decoder path
   again; remaining time is mostly ConvNeXt sparse conv/GEMM/scatter at large N.
2. **PBR atlas export**: Fix/verify the UV chart packer; vertex-colored OBJ is the
   default texture output for now, `T2_PBR_TEXTURE_MAP=1` opts into atlas maps.
3. **Fresh Stage 2/3 DiT reference dumps**: Persist current PyTorch flow-model
   tensors for CUDA verifier coverage beyond old ad-hoc `/tmp` dumps. SC-VAE
   shape/texture refs now live under `ref/trellis2/dumps/`.
4. **Full GPU pipeline (CUDA)**: Image → DINOv3 → Stage 1 → Stage 2 → Stage 3 → mesh.

### HIP/ROCm (RX 9070 XT)
1. **Fix E2E sampling**: 12-step CFG + rescale produces ~70% occupancy vs PyTorch's ~2%.
   Single-step DiT (corr=0.99999754) and decoder (corr=1.0) individually verified.
   Per-step latent comparison with PyTorch ref needed to isolate the CFG/sampling bug.
   PyTorch reference outputs saved at `/tmp/chair_ref/ref_latent_step{0-11}.npy`.
2.5. **FP8 weights** (DONE, 2026-05): BF16-act × FP8-wt WMMA path lands gated by FP8 dtype detection in safetensors. Offline quantizer at `cpu/trellis2/quantize_dit_fp8.py` (E4M3, per-tensor scale, excludes `adaLN_modulation` and `t_embedder`). 12% per-step speedup, 50% weight memory savings, corr=0.999358 vs BF16. Bottleneck has shifted to flash-attn / non-GEMM ops. Optional next step: FP8 flash-attn (rdna4/fp8/bench_fp8_fa.c, 18.7 TF/s).
2. ~~**WMMA GEMM**: Replace F16 tiled GEMM with RDNA4 WMMA cooperative matrix GEMM
   for ~10x speedup (144s/step → ~15s/step target).~~ **DONE** (commit 4d59a1e):
   83× over scalar fallback; 1.73 s/step. Next: **BF16-act × FP8-weight WMMA**
   (qimg/Flux.2 pattern) — halves weight memory, ~2× over BF16 expected.
3. **DINOv3 on GPU**: Port DINOv3 encoder to HIP (currently CPU-only, 210s).

### Vulkan
1. **Test Stages 2–3 on GPU**: Validate Stage 2/3 DiT + decoder against CUDA/Python reference.
   Needs Stage 2/3 weight files downloaded to `/mnt/disk1/models/`.
2. **F16 GEMM**: Use `matmul_coopmat_f16` for cooperative matrix acceleration on RDNA4
   (biggest remaining perf win — all stages bottlenecked by F32 GEMM).
3. **GPU C2S subdivision**: Move coordinate expansion + hash build to GPU to eliminate
   CPU roundtrip in Vulkan shape/texture decoder paths.
4. **FDG mesh extraction**: Port `trellis2_fdg_mesh.h` for final textured mesh output.
