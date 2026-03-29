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
└──────────┬───────────────┘  (NOT YET IMPLEMENTED)
           │
           ▼
┌──────────────────────────┐
│  Stage 2: Shape Decoder   │  SC-VAE decoder
│  shape latent → mesh      │  Vertices + faces + attributes
└──────────┬───────────────┘  (NOT YET IMPLEMENTED)
           │
           ▼
┌──────────────────────────┐
│  Stage 3: Texture DiT     │  Texture flow model (1.3B params)
│  shape + image → texture  │  12 steps, produces texture latent
└──────────┬───────────────┘  (NOT YET IMPLEMENTED)
           │
           ▼
┌──────────────────────────┐
│  Stage 3: Texture Decoder │  SC-VAE texture decoder
│  texture latent → PBR     │  Albedo, roughness, metallic maps
└──────────┬───────────────┘  (NOT YET IMPLEMENTED)
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
| Normalization | **ChannelLayerNorm** (LayerNorm across C per spatial position) |
| Groups | N/A (not GroupNorm) |

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
out_layer:    ChannelLayerNorm(32) → SiLU → Conv3d(32→1, k=3)
Output:       [1, 64, 64, 64] occupancy logits
```

ResBlock3d: `ChannelLN → SiLU → Conv3d → ChannelLN → SiLU → Conv3d + skip`

### Stages 2-3 (Not Yet Implemented)

- **Stage 2 Shape**: Sparse DiT (1.3B), operates on sparse voxel coords from Stage 1.
  Two resolution variants: 512 and 1024 (cascade mode). Uses FlexGEMM for sparse conv.
- **Stage 3 Texture**: Sparse DiT (1.3B), conditioned on shape + image.
  Produces texture latent decoded by SC-VAE to PBR material maps.
- **Decoders**: SC-VAE (Sparse Convolutional VAE) for both shape and texture.

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

### Stages 2-3

| Component | CUDA | CPU | PyTorch Ref | Status |
|-----------|------|-----|-------------|--------|
| Sparse tensor infrastructure | — | `common/sparse3d.h` | — | CPU ops verified |
| Stage 2 shape flow DiT | `cuda_trellis2_runner.c` | `trellis2_dit.h` | `run_stage2_ref.py` | **CUDA verified (corr=1.0)** |
| Stage 2 shape decoder (SC-VAE) | — | `trellis2_shape_decoder.h` | — | CPU verified |
| Stage 3 texture flow DiT | — | `trellis2_dit.h` | `run_stage3_tex_ref.py` | CPU 1-step verified |
| Stage 3 texture decoder (SC-VAE) | — | `trellis2_shape_decoder.h` | — | CPU verified (small N) |
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

| Stage | CPU (Zen2, 4T) | CUDA (RTX 5060 Ti) | PyTorch (RTX 5060 Ti) |
|-------|----------------|---------------------|----------------------|
| DINOv3 encode | ~3.5 min | — (CPU features) | ~8 sec |
| Stage 1 DiT (12 steps, CFG) | ~9 hours | **~8 min** | ~34 sec |
| Decoder | ~22 min | ~0.5 sec | ~0.2 sec |
| **Total** | **~9.5 hours** | **~8 min** | **~42 sec** |

CUDA DiT is ~65x faster than CPU. PyTorch is ~14x faster than our CUDA due to
cuBLAS tensor-core F16 GEMM vs our F32 tiled GEMM.

## Weight Files

| Component | Path | Size | Format |
|-----------|------|------|--------|
| DINOv3 ViT-L/16 | `dinov3-vitl16/model.safetensors` | 1.2 GB | F32 (timm) |
| Stage 1 DiT | `trellis2-4b/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors` | 2.5 GB | BF16 |
| Stage 1 Decoder | `trellis-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors` | 141 MB | F16/F32 mixed |

Download:
```bash
uvx --from huggingface_hub hf download timm/vit_large_patch16_dinov3.lvd1689m model.safetensors \
  --local-dir /mnt/disk01/models/dinov3-vitl16
uvx --from huggingface_hub hf download microsoft/TRELLIS.2-4B \
  ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors --local-dir /mnt/disk01/models/trellis2-4b
uvx --from huggingface_hub hf download microsoft/TRELLIS-image-large \
  ckpts/ss_dec_conv3d_16l8_fp16.safetensors --local-dir /mnt/disk01/models/trellis-image-large
```

## Source Files

### CUDA implementation (`cuda/trellis2/`)

| File | Lines | Description |
|------|-------|-------------|
| `cuda_trellis2_runner.h` | 65 | Public API |
| `cuda_trellis2_runner.c` | ~900 | DiT forward, decoder forward, weight loading |
| `cuda_trellis2_kernels.h` | ~280 | CUDA kernels: adaLN, gated_add, modulation, rope_3d, rms_norm_perhead, conv3d, channel_layernorm_3d, pixel_shuffle_3d, timestep_embed_cossin |
| `cuda_trellis2_ops.h` | ~300 | Kernel launch wrappers |
| `test_cuda_trellis2.c` | ~320 | Test harness: full/stage1/decode modes, .npy I/O, mesh export |
| `verify_decoder.c` | ~65 | Standalone decoder verification |
| `Makefile` | 20 | gcc build (no nvcc, NVRTC runtime compilation) |

### CPU implementation (`common/`)

| File | Lines | Description |
|------|-------|-------------|
| `cpu_compute.h` | ~640 | Shared primitives: attention, cross-attention, LayerNorm, GELU, RoPE |
| `dinov3.h` | ~900 | DINOv3 ViT-L/16 encoder |
| `trellis2_dit.h` | ~1000 | DiT blocks, F32 GEMM, adaLN, QK RMSNorm, 3D RoPE |
| `trellis2_stage1.h` | ~250 | Euler flow sampling, CFG, guidance interval |
| `trellis2_ss_decoder.h` | ~620 | Conv3D, ChannelLayerNorm, pixel_shuffle_3d, ResBlock3d |

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
| **Decoder norm type** | **Decoder output flipped** | **Used GroupNorm, official uses ChannelLayerNorm** | **New channel_layernorm_3d_f32 kernel** |
| DINOv3 final norm | Features range differs | TRELLIS.2 uses unparameterized LN, not learned norm | Apply plain LN in encoder |
| **Stage 2 F16 GEMM** | **All GEMM output garbage** | **`use_f32_gemm` not set when loading Stage 2 standalone** | **Set flag in `cuda_trellis2_load_stage2`** |
| **Stage 2 scratch overflow** | **Cross-attn produces 5e26** | **`scratch[1]` sized for self-attn N only, not cross-attn ctx_len*2D** | **Include `ca_kv_gemm_sz` in max** |

## Next Steps

1. **Stage 2 full sampling**: Integrate Stage 2 DiT into end-to-end flow sampling with CFG
   on GPU (DiT forward verified, need sampling loop + shape decoder).
2. **Stage 3 Texture CUDA**: Port Stage 3 texture flow DiT to GPU (same architecture).
3. **Shape decoder CUDA**: Sparse conv SC-VAE decoder on GPU.
4. **Texture decoder CUDA**: Sparse conv SC-VAE texture decoder on GPU.
5. **F16 GEMM**: Switch from F32 tiled GEMM to F16 tensor-core GEMM for ~10x speedup.
6. **Full GPU pipeline**: Image → DINOv3 → Stage 1 → Stage 2 → Stage 3 → mesh, all on GPU.
