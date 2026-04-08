# Qwen-Image: Module Analysis for CPU & CUDA Implementation

## Architecture Overview

Qwen-Image is a ~20B parameter text-to-image model with 3 main components + scheduler:

```
Text Prompt ──→ [Text Encoder: Qwen2.5-VL-7B] ──→ text_emb (B, T, 3584)
                                                        │
                                                        ↓
Noise z_T ───→ [MMDiT: 60 layers, 24 heads] ←── timestep t
                      │                              ↑
                      ↓                              │
               [Euler Solver × N steps] ─────────────┘
                      │
                      ↓
              z_0 (B, 16, H/8, W/8)
                      │
                      ↓
              [VAE Decoder] ──→ RGB image (H, W, 3)
```

### A. Text Encoder — Qwen2.5-VL (frozen, ~7B total)

| Parameter | Value |
|-----------|-------|
| hidden_size | 3584 |
| num_layers | 28 |
| num_attention_heads | 28 |
| num_kv_heads | 4 (GQA) |
| head_dim | 128 |
| intermediate_size | 18944 |
| vocab_size | 152064 |
| activation | SiLU |
| rope_theta | 1,000,000 |
| max_position_embeddings | 128,000 |

Standard Qwen2.5 LLM with GQA. Includes ViT vision tower (for optional reference image input). Outputs `(B, text_seq_len, 3584)` embeddings + `(B, 768)` pooled projection.

### B. MMDiT — QwenImageTransformer2DModel (~20B)

| Parameter | Value |
|-----------|-------|
| num_layers | 60 |
| num_attention_heads | 24 |
| attention_head_dim | 128 |
| inner_dim (hidden) | 3072 (24 × 128) |
| joint_attention_dim | 3584 |
| in_channels | 64 (latent input after patchify) |
| out_channels | 16 |
| patch_size | 2 |
| pooled_projection_dim | 768 |
| axes_dims_rope | [16, 56, 56] |

**Per-block structure (Dual-Stream Joint Attention):**
```
Input: img_hidden (B, img_seq, 3072), txt_hidden (B, txt_seq, 3584)

1. adaLN modulation from timestep → 6 params each for img and txt
   img: (shift1, scale1, gate1, shift2, scale2, gate2)
   txt: (shift1, scale1, gate1, shift2, scale2, gate2)

2. Modulated RMSNorm on img stream → Q_img, K_img, V_img
   Modulated RMSNorm on txt stream → Q_txt, K_txt, V_txt
   QK normalization (per-head RMSNorm on Q and K)

3. RoPE applied to Q_img, K_img (2D: height, width)
   RoPE applied to Q_txt, K_txt (1D: position)

4. Concatenate: Q = [Q_txt; Q_img], K = [K_txt; K_img], V = [V_txt; V_img]
   Joint attention over full sequence
   Split output back to img_attn, txt_attn

5. Gated residual: img += gate1 * img_attn
                    txt += gate1 * txt_attn

6. Modulated RMSNorm → FFN (SwiGLU for img, GELU for txt)
   Gated residual: img += gate2 * img_ffn
                    txt += gate2 * txt_ffn
```

**Timestep conditioning (adaLN-Zero):**
```
t_emb = sinusoidal_embed(timestep, 256)
t_emb = SiLU(Linear(256, 3072)) → Linear(3072, 3072)
per_block: mod_params = Linear(3072, 6*3072)  # shift, scale, gate × 2
```

**Patchify/Unpatchify:**
```
Patchify:  (B, C, H, W) → (B, H/2 * W/2, C*4)  where C=16, so in_channels=64
Unpatchify: (B, seq, out_channels) → (B, 16, H, W)
```

### C. VAE — AutoencoderKLQwenImage

| Parameter | Value |
|-----------|-------|
| z_dim (latent channels) | 16 |
| base_dim | 96 |
| dim_mult | [1, 2, 4, 4] |
| channels per stage | [96, 192, 384, 384] |
| num_res_blocks | 2 per stage |
| latents_mean | [0.0] |
| latents_std | [0.18215] |

**Encoder:** Conv2d(3→96) → 4 DownBlocks(ResBlock×2 + Downsample) → MidBlock(ResBlock + Attn + ResBlock) → GroupNorm → Conv2d(384→32) → split → μ, logσ²

**Decoder:** Conv2d(16→384) → MidBlock → 4 UpBlocks(ResBlock×2 + Upsample) → GroupNorm → Conv2d(96→3)

**ResBlock:** GroupNorm → SiLU → Conv2d → GroupNorm → SiLU → Conv2d + skip

### D. Scheduler — FlowMatchEulerDiscreteScheduler

| Parameter | Value |
|-----------|-------|
| num_train_timesteps | 1000 |
| time_shift_type | exponential |
| use_dynamic_shifting | true |
| base_shift | 0.5 |
| max_shift | 0.9 |
| shift_terminal | 0.02 |
| base_image_seq_len | 256 |
| max_image_seq_len | 8192 |

**Dynamic shift formula:**
```
mu = (log(seq_len) - log(base_seq_len)) / (log(max_seq_len) - log(base_seq_len))
shift = base_shift + (max_shift - base_shift) * mu
sigma_adjusted = exp(shift) * sigma / (1 + (exp(shift) - 1) * sigma)
```

---

## Reusable Modules (Already Available)

### Text Encoder (Qwen2.5-VL)

| Component | Existing File | Notes |
|-----------|--------------|-------|
| Qwen LLM transformer | `common/transformer.h` | Qwen2/3/3.5 supported. Qwen2.5-VL LLM part is same arch as Qwen2. GQA, RoPE, SiLU all present |
| BPE tokenizer | `common/bpe_tokenizer.h` | Qwen tokenization with UTF-8/GPT-2. vocab_size=152064 matches |
| Vision encoder (ViT) | `common/vision_encoder.h` | Qwen3-VL CLIP with DeepStack. Needs adaptation for Qwen2.5-VL variant |
| GGUF loader | `common/gguf_loader.h` | v2/v3, mmap, all quant types. Text encoder uses standard Qwen GGUF |
| GGML dequantization | `common/ggml_dequant.h` | 24+ types: Q4_K, Q8_0, F16, BF16, etc. |
| CUDA LLM runner | `cuda/llm/cuda_llm_runner.h` | Qwen2/3/3.5 GPU inference. GQA, RoPE, MoE, deepstack |
| CUDA vision encoder | `cuda/vlm/cuda_vision_encoder.h` | F16/F32 modes, dynamic resolution |
| Image I/O | `common/image_utils.h` | JPEG/PNG/BMP/HDR/EXR load/resize/normalize |

### MMDiT (Partial Reuse)

| Component | Existing File | Notes |
|-----------|--------------|-------|
| adaLN modulation pattern | `common/trellis2_dit.h` | TRELLIS.2 uses same scheme: SiLU→Linear→6×dim shift/scale/gate per block |
| RMSNorm (per-head QK) | `cuda/hy3d/cuda_hy3d_kernels.h` | `rms_norm_f32`, `qk_layernorm_f32` kernels |
| Euler solver step | `cuda/hy3d/cuda_hy3d_kernels.h` | `euler_step_f32`: x_new = x - dt * v |
| CFG combine | `cuda/hy3d/cuda_hy3d_kernels.h` | `cfg_combine_f32`: out = uncond + scale * (cond - uncond) |
| Timestep embedding | `cuda/hy3d/cuda_hy3d_kernels.h` | `timestep_embed_f32`: sinusoidal embedding |
| Flash Attention | `cuda/cuda_kernels_common.h` | `attn_prefill_f32` with online softmax, Tensor Core MMA |
| GEMM (F16→F32) | `cuda/cuda_kernels_common.h` | `gemm_f16_f32` with m16n8k16 MMA |
| GEMM (FP8→F32) | `cuda/cuda_kernels_common.h` | `gemm_fp8_f32` with m16n8k32 MMA |
| GELU, SiLU, sigmoid | `cuda/cuda_kernels_common.h` | All present |
| LayerNorm | `cuda/cuda_kernels_common.h` | `layernorm_f32` |
| CPU attention | `common/cpu_compute.h` | `cpu_attention()`, `cpu_cross_attention()` with online softmax |
| CPU LayerNorm/RMSNorm | `common/cpu_compute.h` | `cpu_layernorm()`, plus RMSNorm in transformer.h |
| CPU GELU/SiLU | `common/cpu_compute.h` | Both present |
| CPU GEMM F16 | `common/cpu_compute.h` | `cpu_gemm_f16()` threaded |
| SafeTensors loader | `common/safetensors.h` | For VAE weights |
| CUDA op wrapper pattern | `cuda/hy3d/cuda_hy3d_ops.h` | `op_xxx()` inline wrappers for kernel launch |

### Scheduler (Partial Reuse)

| Component | Existing File | Notes |
|-----------|--------------|-------|
| Euler step | `cuda/hy3d/cuda_hy3d_kernels.h` | Same formula |
| CFG | `cuda/hy3d/cuda_hy3d_kernels.h` | Same formula |

---

## Missing Modules (Need Implementation)

### Priority 1: MMDiT Core (`common/qwen_image_dit.h` + `cuda/qimg/cuda_qimg_dit.h`)

#### 1.1 QwenImage Dual-Stream DiT Block (CPU + CUDA)
- **What**: 60-layer transformer with dual-stream (text + image) joint attention
- **Why new**: Existing DiT (trellis2_dit.h) uses single-stream self-attn + cross-attn. Qwen-Image uses dual-stream where both text and image have their own norms/FFNs but share one joint attention
- **Details**:
  - Per-block: 2× RMSNorm + modulation (img stream), 2× RMSNorm + modulation (txt stream)
  - Fused QKV projection for each stream → concatenate → joint attention → split
  - Image FFN: SwiGLU (gate × up → down)
  - Text FFN: GELU (up → down)
  - Gated residual connections (sigmoid gate from adaLN)
  - QK normalization (per-head RMSNorm on Q and K before attention)
- **Reuse**: adaLN pattern from `trellis2_dit.h`, attention from `cpu_compute.h` / `cuda_kernels_common.h`

#### 1.2 2D/1D RoPE for MMDiT
- **What**: Rotary position embeddings: 2D (height, width) for image tokens, 1D for text tokens
- **Why new**: Existing RoPE is 1D (transformer.h) or 3D (trellis2_dit.h for xyz). Need 2D spatial for image patches + 1D for text, applied to concatenated Q/K before joint attention
- **Config**: axes_dims_rope = [16, 56, 56] → 16 dims for temporal, 56 for height, 56 for width
- **Reuse**: `cpu_rope_2d()` exists but may need adaptation for the concat pattern

#### 1.3 Patchify / Unpatchify
- **What**: Convert latent `(B, 16, H, W)` → `(B, H/2*W/2, 64)` and back
- **Why new**: 2×2 spatial patches folded into channel dimension. Simple reshape/transpose ops but need correct memory layout
- **Complexity**: Low — ~20 lines each for CPU, ~1 CUDA kernel each

#### 1.4 Final Linear Projection
- **What**: Project from inner_dim (3072) → out_channels×patch_size² (16×4=64), then unpatchify
- **Why new**: Standard linear, but needs adaLN modulation before projection
- **Reuse**: GEMM kernels

#### 1.5 GGUF Weight Loading for MMDiT
- **What**: Map city96/Qwen-Image-gguf tensor names to our block structure
- **Why new**: Need tensor name mapping for 60 dual-stream blocks. Names follow `blk.{0-59}.{component}` pattern
- **Details**: GGUF extended name limit (128 chars) for diffusion models. Components: `img_attn_qkv`, `img_attn_proj`, `img_norm1`, `img_mlp_*`, `txt_attn_qkv`, `txt_attn_proj`, `txt_norm1`, `txt_mlp_*`, `img_mod`, `txt_mod`, etc.
- **Reuse**: `gguf_loader.h` for file parsing, `ggml_dequant.h` for weight dequantization

### Priority 2: VAE (`common/qwen_image_vae.h` + `cuda/qimg/cuda_qimg_vae.h`)

#### 2.1 VAE Decoder (latent → RGB)
- **What**: Conv-based decoder: Conv2d(16→384) → MidBlock → 4× UpBlock → Conv2d(96→3)
- **Why new**: No 2D convolutional VAE exists in codebase. Hunyuan3D VAE is for 3D (SDF grids)
- **Details**:
  - ResBlock: GroupNorm(32) → SiLU → Conv3×3 → GroupNorm(32) → SiLU → Conv3×3 + skip
  - UpBlock: ResBlock×2 + nearest-neighbor upsample + Conv3×3
  - MidBlock: ResBlock → Self-Attention → ResBlock
  - Channel progression: 384 → 384 → 192 → 96 → 3
  - Latent denormalization: z = z / 0.18215

#### 2.2 VAE Encoder (RGB → latent) — optional for image editing
- **What**: Conv-based encoder: Conv2d(3→96) → 4× DownBlock → MidBlock → Conv2d(384→32)
- **Why new**: Same reason as decoder. Needed for image-to-image editing tasks
- **Details**:
  - DownBlock: ResBlock×2 + Conv3×3 stride-2 downsample
  - Output: μ, logσ² → sample z = μ + σ·ε

#### 2.3 2D Convolution Layers (CPU + CUDA)
- **What**: Conv2d with kernel_size=3, stride=1, padding=1 (most common) + stride-2 for downsample
- **Why new**: Existing Conv2d is only for patch embedding (specific kernel sizes). Need general Conv2d for VAE
- **Reuse**: Can extend `patch_embed_conv2d` kernel pattern
- **Complexity**: Medium — need efficient tiled implementation for 384-channel convolutions

#### 2.4 GroupNorm
- **What**: Group normalization with 32 groups (standard for VAEs)
- **Why new**: Existing code has LayerNorm and RMSNorm but no GroupNorm
- **Complexity**: Low-Medium — similar to LayerNorm but over channel groups

#### 2.5 Nearest-Neighbor Upsample
- **What**: 2× spatial upsample by pixel duplication
- **Why new**: Existing bilinear upsample is different. VAE uses NN upsample + Conv
- **Complexity**: Low — simple index duplication

#### 2.6 SafeTensors Weight Loading for VAE
- **What**: Load VAE weights from safetensors format
- **Why new**: Need tensor name mapping specific to QwenImage VAE
- **Reuse**: `safetensors.h` for file parsing

### Priority 3: Scheduler (`common/qwen_image_scheduler.h`)

#### 3.1 FlowMatchEulerDiscreteScheduler
- **What**: Timestep schedule generation with dynamic shifting based on image resolution
- **Why new**: Existing code has Euler step but no schedule generation or dynamic shifting logic
- **Details**:
  - Generate sigma schedule: linear spacing in [1, 0] or [1, shift_terminal]
  - Dynamic shift: mu = interp(log(seq_len), log(256), log(8192)); shift = lerp(0.5, 0.9, mu)
  - Adjust sigmas: sigma_adj = exp(shift) * sigma / (1 + (exp(shift)-1) * sigma)
  - Compute dt = sigma[i+1] - sigma[i] for each step
- **Complexity**: Low — ~50 lines of C, pure math
- **Reuse**: Euler step kernel for the actual denoising update

### Priority 4: Text Encoder Adaptation

#### 4.1 Qwen2.5-VL Adaptation
- **What**: Adapt existing Qwen3-VL code for Qwen2.5-VL variant
- **Why partially new**: Architecture is very similar but may have differences in vision tower config, projection layer, spatial merge strategy
- **Details**:
  - hidden_size=3584 (vs Qwen3-VL which may differ)
  - 28 layers, 28/4 heads (GQA ratio 7:1)
  - intermediate_size=18944
  - Need to output both sequence embeddings (3584-dim) and pooled embedding (768-dim)
- **Reuse**: `transformer.h` (Qwen2 arch), `vision_encoder.h` (ViT), `cuda_llm_runner.h`

#### 4.2 Text Pooling + Projection
- **What**: Extract pooled representation (768-dim) from text encoder output for timestep conditioning
- **Why new**: Existing code outputs full sequence, needs pooled vector for adaLN
- **Complexity**: Low — mean/last-token pooling + Linear(3584→768)

### Priority 5: Pipeline Orchestration

#### 5.1 Full Pipeline (`common/qwen_image_pipeline.h` + `cuda/qimg/cuda_qimg_runner.h`)
- **What**: End-to-end text-to-image: tokenize → encode → denoise loop → decode
- **Details**:
  1. Tokenize text prompt (+ optional negative prompt for CFG)
  2. Encode with Qwen2.5-VL → (B, T, 3584) + (B, 768) pooled
  3. Initialize noise z_T ~ N(0,1) of shape (B, 16, H/8, W/8)
  4. Patchify: (B, 16, H/8, W/8) → (B, H/16 * W/16, 64)
  5. For each timestep t in schedule:
     - MMDiT forward: predict velocity v
     - (Optional) CFG: v = v_uncond + scale * (v_cond - v_uncond)
     - Euler step: z = z - dt * v
  6. Unpatchify: (B, seq, 16) → (B, 16, H/8, W/8)
  7. VAE decode: (B, 16, H/8, W/8) → (B, 3, H, W)
  8. Save image

---

## Weight Files

### GGUF (MMDiT) — city96/Qwen-Image-gguf

| Quant | Size | Quality |
|-------|------|---------|
| Q4_K_S | 12.1 GB | Good balance |
| Q4_K_M | 13.1 GB | Better quality |
| Q8_0 | 21.8 GB | Near-lossless |
| BF16 | 40.9 GB | Full precision |

Tensor naming: `blk.{0-59}.{component}.weight` with 128-char name limit.
Dynamic precision: Q2_K/Q3_K keep first/last layers at higher precision.

### SafeTensors (VAE)
- Separate download from Qwen/Qwen-Image repo
- ~127M params (encoder 54M + decoder 73M)

### Text Encoder
- Standard Qwen2.5-VL-7B GGUF (available from many sources)
- Or safetensors from Qwen/Qwen2.5-VL-7B-Instruct

---

## Implementation Effort Estimate

| Module | New Code (approx) | Difficulty |
|--------|-------------------|------------|
| MMDiT dual-stream block (CPU) | ~800 lines | High — largest new component |
| MMDiT dual-stream block (CUDA) | ~600 lines | High — new kernels for dual-stream |
| MMDiT weight loading (GGUF) | ~200 lines | Medium — tensor name mapping |
| VAE decoder (CPU) | ~400 lines | Medium — conv stack |
| VAE decoder (CUDA) | ~400 lines | Medium — Conv2d kernels |
| VAE encoder (CPU) | ~300 lines | Medium — mirror of decoder |
| VAE encoder (CUDA) | ~300 lines | Medium — mirror of decoder |
| GroupNorm (CPU + CUDA) | ~100 lines | Low |
| Conv2d general (CPU + CUDA) | ~200 lines | Medium |
| NN upsample (CPU + CUDA) | ~50 lines | Low |
| Patchify/Unpatchify | ~60 lines | Low |
| 2D RoPE for MMDiT | ~80 lines | Low-Medium |
| FlowMatch scheduler | ~60 lines | Low |
| Text encoder adaptation | ~150 lines | Low-Medium |
| Pipeline orchestration | ~300 lines | Medium |
| **Total** | **~4000 lines** | |

---

## Implementation Status

### Completed (CPU) — 34/34 layer-by-layer PASS (corr=1.000000)
- `common/qwen_image_scheduler.h` — FlowMatch scheduler with dynamic shifting
- `common/qwen_image_dit.h` — 60-layer dual-stream MMDiT, GGUF Q4_0 loading, dump hooks
- `common/qwen_image_vae.h` — 3D causal VAE decoder, BF16 safetensors loading
- `common/qwen_image_text_encoder.h` — Qwen2.5-VL-7B text encoder wrapper (transformer.h + bpe_tokenizer.h)
- `common/transformer.h` — Added qwen2vl arch detection + IQ4_XS dequant support
- `cpu/qwen_image/test_qwen_image.c` — Test harness: --test-sched/vae/dit/enc/generate modes
- `ref/qwen_image/` — PyTorch reference (uv venv): run_vae_reference.py, run_dit_reference.py, run_dit_block_reference.py, compare.py
- **Full pipeline verified**: text enc (3 tokens) → 2 DiT steps → VAE decode → 64x64 PPM

### Key Implementation Details

**VAE**: 3D causal VAE with 5D conv kernels. For image-only mode (T=1), 3D→2D
conversion by summing kernel across temporal dim. Blocks 3, 7, 11 are resample-only
(no residual block). Channel flow: 16→384→384→192→384→192→96→3.

**MMDiT GGUF Tensor Names** (actual from inspection):
- `transformer_blocks.N.attn.to_q/k/v.weight` — image Q/K/V [3072, 3072] Q4_0
- `transformer_blocks.N.attn.add_q/k/v_proj.weight` — text Q/K/V [3072, 3072] Q4_0
- `transformer_blocks.N.attn.norm_q/k.weight` — per-head QK RMSNorm [128] F32
- `transformer_blocks.N.img_mod.1.weight` — image modulation [3072, 18432] Q4_0
- `transformer_blocks.N.img_mlp.net.0.proj.weight` — GELU MLP up [3072, 12288]
- `transformer_blocks.N.img_mlp.net.2.weight` — GELU MLP down [12288, 3072]
- Global: `img_in`, `txt_in`, `txt_norm`, `time_text_embed.timestep_embedder.*`,
  `norm_out.linear`, `proj_out`

**MMDiT FFN**: NOT GEGLU despite `proj` naming — simple Linear(3072,12288)→GELU→Linear(12288,3072)

### CPU Performance (single-threaded, unoptimized)
- Scheduler: instant
- Text encoder (3 tokens, 28 layers): ~3s
- VAE decode (8×8 → 64×64): ~21s
- DiT single step (16 img + 3 txt tokens, 60 blocks): ~100s
- Full pipeline (64×64, 2 steps): ~200s total

### Layer-by-Layer Verification (34/34 PASS)
- VAE: 22 stages (input → post_quant → conv1 → middle → 15 upsample → head)
- DiT: 12 checkpoints (inputs, t_emb, projections, modulation, block 0 full: adaLN, QK-norm, attention+RoPE, MLP)

### TODO (CUDA)
```
cuda/qimg/
  cuda_qimg_runner.h       # CUDA pipeline runner
  cuda_qimg_dit.h          # CUDA MMDiT forward pass
  cuda_qimg_vae.h          # CUDA VAE encoder/decoder
  cuda_qimg_kernels.h      # New CUDA kernels (dual-stream attn, GroupNorm, Conv2d, patchify)
  cuda_qimg_ops.h          # CUDA op wrappers
```

## Weight Files

### Weights on disk: /mnt/disk01/models/qwen-image/
- `diffusion-models/qwen-image-Q4_0.gguf` (12GB) — MMDiT, 1933 tensors
- `text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf` (4.8GB) — LLM
- `text-encoder/mmproj-F16.gguf` (1.3GB) — CLIP vision encoder
- `vae/qwen_image_vae.safetensors` (254MB) — VAE, 194 tensors, BF16
