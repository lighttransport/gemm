# Qwen-Image Text-to-Image Pipeline

End-to-end implementation of the [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) text-to-image model using single-header C libraries with CUDA GPU acceleration.

## Pipeline Overview

```
Text Prompt
    │
    ▼
┌─────────────────┐
│  Text Encoder    │  Qwen2.5-VL-7B (28 layers, GQA)
│  [CUDA or CPU]   │  GGUF Q4_K weights + injected biases
│                  │  → hidden states [N_txt, 3584]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MMDiT           │  60-layer dual-stream DiT
│  [CUDA]          │  FP8 E4M3 weights, 128×128 tiled GEMM
│                  │  × N_steps (Euler flow matching)
│                  │  latent [16, H/8, W/8]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VAE Decoder     │  3D causal VAE (BF16 weights)
│  [CUDA + CPU]    │  → RGB [3, H, W]
└────────┬────────┘
         │
         ▼
    Output Image
```

## Model Specifications

### Text Encoder — Qwen2.5-VL-7B

| Parameter | Value |
|---|---|
| Architecture | `qwen2vl` (GQA transformer) |
| Hidden dim | 3584 |
| Layers | 28 |
| Attention heads | 28 (4 KV heads, 7:1 GQA) |
| Head dim | 128 |
| FFN intermediate | 18944 |
| Vocab size | 152064 |
| RoPE theta | 1,000,000 |
| Norm epsilon | 1e-6 |
| Activation | SiLU |

Output: per-token hidden states `[N_tokens, 3584]`, no pooling.

### MMDiT — 60-Layer Dual-Stream Diffusion Transformer

| Parameter | Value |
|---|---|
| Blocks | 60 |
| Hidden dim | 3072 |
| Attention heads | 24 |
| Head dim | 128 |
| MLP intermediate | 12288 |
| Text input dim | 3584 |
| Image input dim | 64 (after patchify) |
| Latent channels | 16 |
| Patch size | 2×2 |
| RoPE theta | 10,000 |
| RoPE axes dims | [16, 56, 56] (temporal, height, width) |
| Norm epsilon | 1e-6 |

**Per-block flow:**
1. Timestep modulation (adaLN-Zero): `SiLU(t_emb) → Linear(3072→18432)` → split into 6 × [shift, scale, gate]
2. Parallel streams: `adaLN(img)`, `adaLN(txt)`
3. Q/K/V projection (3072→3072 each) + per-head QK RMSNorm
4. RoPE: 3-axis for both image and text tokens (matching ComfyUI convention)
5. Joint attention: concatenate `[txt, img]` for Q/K/V, attend, split back
6. Gated residual: `x += gate × proj(attn_out)`
7. FFN with gated residual: `adaLN → GELU(Linear) → Linear → gated_add`

**Final output:** `LayerNorm + adaLN → Linear(3072→64) → unpatchify → velocity [16, H/8, W/8]`

### VAE — 3D Causal Decoder (Wan2.1)

| Parameter | Value |
|---|---|
| Latent channels | 16 |
| Base channels | 96 |
| Channel multipliers | [1, 2, 4, 4] → [96, 192, 384, 384] |
| Normalization | **RMS norm** (NOT GroupNorm) |
| Activation | SiLU |
| Weight dtype | BF16 (254 MB) |
| Spatial padding | **Zeros** (spatial_padding_mode="zeros") |
| 3D→2D conversion | **Last temporal slice only** (causal_zero for T=1) |
| Upsample method | Nearest-neighbor 2× + Conv2d 3×3 |

The Wan VAE uses `RMS_norm`: `F.normalize(x, dim=channels) × sqrt(C) × gamma` — L2 normalization per spatial position along the channel dimension. This is fundamentally different from GroupNorm.

For CausalConv3d with T=1 input, ComfyUI's `autopad="causal_zero"` truncates the weight to the last temporal slice (`weight[:,:,-1:,:,:]`), not sum of all slices.

Decoder path: `post_quant_conv(16→16) → conv1(16→384) → mid_block → 15 upsample blocks → norm → conv_out(96→3)`

Blocks 3, 7, 11 are resample-only (spatial upsample, no residual block).

### Scheduler — FlowMatch Euler (ComfyUI-compatible)

| Parameter | Value |
|---|---|
| Scheduler type | "simple" (ComfyUI) |
| Shift function | `time_snr_shift(alpha, t) = alpha * t / (1 + (alpha-1) * t)` |
| Shift alpha | 3.1 (NOT `exp(3.1)`) |
| Sigma table | 1000 pre-computed shifted sigmas |
| Step selection | Evenly-spaced indices from reversed table |
| Solver | Euler: `x += dt × velocity` |
| Timestep scale | `timestep = sigma × 1000` (model's Timesteps has scale=1000) |

The "simple" scheduler pre-computes 1000 shifted sigmas, then picks `n_steps` evenly-spaced samples. This differs from computing the shift at exactly `n_steps` points because the shift function is nonlinear.

### Classifier-Free Guidance (CFG)

- CFG scale: 2.5 (ComfyUI default)
- Formula: `v = v_uncond + cfg_scale × (v_cond - v_uncond)`
- Cond and uncond processed with their **original token counts** (no padding — ComfyUI processes them separately)

## Weight Files

All weights stored under `/mnt/disk01/models/qwen-image-st/` (ComfyUI FP8 format):

| Component | File | Format | Size |
|---|---|---|---|
| DiT | `diffusion_models/qwen_image_fp8_e4m3fn.safetensors` | FP8 E4M3 | ~19 GB (1933 tensors) |
| Text Encoder | `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | Scaled FP8 | ~8.7 GB |
| VAE | `vae/qwen_image_vae.safetensors` | BF16 | 254 MB (194 tensors) |

**Alternative GGUF weights** (at `/mnt/disk01/models/qwen-image/`):

| Component | File | Format | Size |
|---|---|---|---|
| DiT | `diffusion-models/qwen-image-Q4_0.gguf` | GGUF Q4_0 | ~12 GB |
| Text Encoder | `text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf` | GGUF Q4_K | ~4.8 GB |

### FP8 E4M3 Format

8-bit float: `[S:1][E:4][M:3]`, exponent bias=7, range ±448.

Dequantization: 256-entry LUT (`fp8_byte → float32`). NaN value (0x7F) mapped to 0.0. LUT verified identical to PyTorch's `torch.float8_e4m3fn` conversion.

Scaled FP8 (text encoder): `actual_weight = fp8_to_f32(byte) × scale_per_tensor`.

## Source Files

### Core Headers (`common/`)

| File | Description |
|---|---|
| `qwen_image_dit.h` | 60-layer MMDiT: patchify, joint attention, adaLN, RoPE, SIMD dot product |
| `qwen_image_vae.h` | 3D causal VAE decoder: RMS norm, zero-padded conv2d, resblocks, upsample |
| `qwen_image_scheduler.h` | FlowMatch scheduler: dynamic shift + ComfyUI-compatible mode |
| `qwen_image_text_encoder.h` | Text encoder: GPU (CUDA LLM runner) or CPU (transformer.h), GGUF + FP8 safetensors |
| `safetensors.h` | SafeTensors file parser (mmap) |
| `gguf_loader.h` | GGUF v3 file loader |
| `transformer.h` | Transformer inference engine (Qwen2.5-VL architecture) |

All are single-header libraries (define `*_IMPLEMENTATION` before include).

### CPU Implementation (`cpu/qwen_image/`)

| File | Description |
|---|---|
| `test_qwen_image.c` | CPU test harness: scheduler, VAE, DiT, text encoder, full pipeline |
| `Makefile` | Build targets for each test mode |

### CUDA Implementation (`cuda/qimg/`)

| File | Description |
|---|---|
| `cuda_qimg_runner.h` | GPU runner: NVRTC kernel compilation, weight management, DiT/VAE forward |
| `test_cuda_qimg.c` | CUDA test harness: init, load, dit step, full generation |
| `test_fp8_gemm.c` | FP8 GEMM correctness test (all sizes PASS, corr=1.0) |

### CUDA LLM Runner (`cuda/llm/`)

| File | Description |
|---|---|
| `cuda_llm_runner.{h,c}` | GPU LLM inference: Qwen2/2.5-VL/3/3.5, GGUF weights, Q/K/V biases |
| `test_cuda_llm.c` | LLM test: CPU vs GPU hidden state comparison |

### PyTorch Reference (`ref/qwen_image/`)

| Script | Description |
|---|---|
| `generate_reference.py` | **Consolidated** reference data generator (text + DiT + VAE) |
| `verify_pipeline.py` | **End-to-end** verification against ComfyUI reference |
| `compare.py` | Single-file .npy comparison tool (corr, max diff, PASS/FAIL) |
| `generate_comfyui.py` | ComfyUI ground truth generator (PyTorch noise) |
| `generate_comfyui_ournoise.py` | ComfyUI ground truth with our PRNG (cos+sin pair caching) |
| `trace_dit_blocks.py` | Block-by-block DiT comparison (saves all 60 block outputs as .npy) |
| `trace_perstep.py` | Per-step latent comparison with ComfyUI sampling loop |
| `trace_block0.py` | Block 0 intermediate trace |
| `encode_text_comfyui.py` | Extract text encoder hidden states from ComfyUI |

## Build & Run

### CUDA (recommended)

```bash
cd cuda/qimg
make test_cuda_qimg

# Full pipeline (256×256, 10 steps)
./test_cuda_qimg --generate \
    --height 256 --width 256 --steps 10 \
    --prompt "a red apple on a white table"

# Options:
#   --no-fp8     Force F32 GEMM (slow but maximum precision)
#   --f16        Force F16 tiled GEMM (correct but slower than FP8 LUT)
#   --no-cfg     Disable classifier-free guidance (single pass, 2× faster)
#   --cfg-scale  CFG scale (default: 2.5)
#   --seed <n>   Random seed (default: 42)
#   --test-vae   VAE-only decode from .npy latent
#   --test-dit   Single DiT step test
#
# Verbose levels (--verbose <n>):
#   0 = silent (errors only)
#   1 = progress (default: step N/M, block loading, timing)
#   2 = stats (per-layer min/max/mean/NaN)
#   3 = dump (save .npy/.bin intermediates for verification)
```

### CPU

```bash
cd cpu/qwen_image
make test_qwen_image

# Full pipeline (128×128, 20 steps)
./test_qwen_image --generate \
    /mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
    /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors \
    /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
    --height 128 --width 128 --steps 20 --prompt "a red apple on a white table"
```

### Verification

```bash
# Step 1: Generate ComfyUI reference data (requires ComfyUI + GPU)
cd ref/qwen_image
python generate_reference.py --all \
    --comfyui-dir /mnt/disk01/ComfyUI \
    --model-dir /mnt/disk01/models/qwen-image-st

# Step 2: Generate our output with debug dumps
cd cuda/qimg
./test_cuda_qimg --generate --height 256 --width 256 --steps 10 --verbose 3

# Step 3: Run verification
cd ref/qwen_image
python verify_pipeline.py --cuda-dir ../../cuda/qimg/ --ref-dir output/
```

## Implementation Status

### Verification Results

| Component | vs ComfyUI | Notes |
|---|---|---|
| **VAE (CUDA)** | SSIM=1.0000 | Pixel-perfect (max diff=2/255) |
| **VAE (CPU)** | SSIM=1.0000 | Pixel-perfect (same 3 fixes as CUDA) |
| **DiT (per-block)** | corr=1.000000 | All 60 blocks match (FP8 LUT = ComfyUI FP8) |
| **DiT (per-block, 6-tok neg)** | corr=1.000000 | Separate token counts work correctly |
| **Text encoder** | corr=1.000000 | FP8 scaled safetensors with Q/K/V biases + template stripping |
| **Scheduler sigmas** | exact match | Pre-shifted table + simple sampling |
| **Full pipeline (10 steps)** | SSIM≈0.73 | FP8 precision compounds over 10 steps |

### Performance (CUDA, RTX 5060 Ti 16GB)

**End-to-end pipeline (256×256, 10 steps, CFG):**

| Stage | Time | Notes |
|---|---|---|
| Text encoder (GPU) | 12.8s | GGUF Q4_K via CUDA LLM runner (CPU fallback: ~710s) |
| DiT load | 1.7s | 41/60 blocks preloaded to VRAM |
| DiT denoising (10 steps) | 36.7s | 3.67s/step, 128×128 tiled GEMM |
| VAE decode | 1.2s | GPU conv2d + CPU middle attention |
| **Total** | **52.4s** | **14.3× faster than initial 748s** |

**DiT per-step performance:**

| Resolution | Latent | Tokens | Time/step (CFG) | Time/step (no-CFG) |
|---|---|---|---|---|
| 256×256 | 32×32 | 256 | 3.67s | ~1.8s |
| 512×512 | 64×64 | 1024 | ~12s | ~6s |

**GPU memory strategy:** Text encoder (4.1 GB) runs first, then releases VRAM. DiT weights (13.3 GB) loaded after. Both share the primary CUDA context via `cuDevicePrimaryCtxRetain`.

### Critical Bugs Fixed

| Bug | Symptom | Fix |
|---|---|---|
| **VAE: GroupNorm instead of RMS norm** | SSIM=0.5, wrong colors/brightness | Use `F.normalize(x,dim=C) × sqrt(C) × gamma` |
| **VAE: Conv3d temporal sum** | Accumulated error from conv1 onward | Use last temporal slice only (`d = kd - 1`) |
| **VAE: Replicate spatial padding** | Boundary artifacts | Use zero padding (`spatial_padding_mode="zeros"`) |
| **Scheduler: exp(shift) instead of shift** | 7× wrong dt, latent too smooth | Use `alpha = shift` directly in SNR formula |
| **Scheduler: direct sigma sampling** | Different sigma schedule from ComfyUI | Pre-compute 1000-entry table, pick evenly |
| **F16 MMA GEMM fragment bug** | Noisy output on Blackwell (sm_120) | Switch to `gemm_tiled_f16_f32` (correct) |
| Missing `cuStreamSynchronize` | Latent drift at >64×64 | Add sync before `cuMemcpyDtoH` |
| FP8 MMA quantizes both inputs | 160× scale error | Use `gemm_fp8w_f32` (FP8 weights, F32 inputs) |
| norm_out scale/shift swap | corr=0.39 → 0.81 | `scale = chunk[0], shift = chunk[1]` |
| Text RoPE: 1D vs 3-axis | corr=0.81 → 0.989 | 3-axis with txt_start offset |
| PRNG: cos-only vs cos+sin | Mismatched noise sequences | Box-Muller pair caching (both cos and sin) |
| **Text enc: missing Q/K/V biases** | Garbage hidden states, random images | Add `attn_q_bias/k_bias/v_bias` to transformer.h |
| **Text enc: no template stripping** | 46 tokens instead of 12 | Strip chat template prefix (match ComfyUI encode_token_weights) |
| **Text enc: wrong cached .npy** | All prompts generate apple | Skip cached .npy when `--prompt` specified |
| **CUDA LLM: IQ4_XS not handled** | NaN from GPU encoder (CUDA error 700) | Add IQ4_XS/IQ4_NL to K-quant upload path |
| **CUDA LLM: Q4_K embed as F16** | NaN embedding → NaN all layers | Dequant Q3_K/Q4_K/Q5_K/Q6_K/IQ4 embeds to F16 at load |

### Optimization History

| Change | Pipeline Time | Speedup |
|---|---|---|
| Baseline (16×16 GEMM, CPU text encoder) | 748s | 1× |
| 128×128 tiled GEMM with register blocking | 714s | 1.05× |
| Multi-threaded CPU text encoder (16 threads) | 710s | 1.05× |
| GGUF Q4_K encoder + injected biases | 710s | 1.05× |
| **GPU text encoder via CUDA LLM runner** | **52s** | **14.3×** |

## TODO / FIXME

### Performance

- [x] **GEMM optimization**: 128×128 tiled GEMM with 8×8 register blocking + fused BF16 truncation. 6.67s → 3.67s/step (1.82×).
- [x] **GPU text encoder**: CUDA LLM runner with GGUF Q4_K weights + injected Q/K/V biases. 710s → 12.8s (56×).
- [ ] **Attention kernel**: Single-block-per-head flash attention works but doesn't scale well to 1024+ tokens. Implement multi-block flash attention or use cuDNN.
- [ ] **VAE on GPU**: Conv2d kernel is naive (one thread per output element). Large convolutions (384 channels × 256×256) would benefit from im2col + GEMM or Winograd.
- [ ] **Block streaming**: Only 41/60 blocks preloaded in 16GB VRAM. On-demand loading adds ~0.5s/block. Pipeline: overlap block loading with computation.
- [ ] **AdaLN + GEMM fusion**: 4 standalone adaln kernels per block → fuse into GEMM input load. ~5-10% DiT speedup.
- [ ] **CFG batching**: Process cond/uncond in single DiT pass (batch_size=2). ~30% DiT speedup.

### Quality

- [x] **CFG with separate n_txt**: Uncond pass uses actual token count (6 tokens) matching ComfyUI.
- [ ] **FP8 precision gap**: Single-step corr=1.0, but 10-step SSIM≈0.73 due to FP8 error compounding. Could be improved by:
  - Mixed precision: F32 for sensitive operations (modulation, attention softmax)
  - Kahan summation in GEMM accumulation
  - Stochastic rounding

### Features

- [ ] **Attention masking**: No `encoder_hidden_states_mask` support. Needed for proper padding-aware attention.
- [ ] **Video generation (T>1)**: VAE temporal convolutions (`time_conv`) not implemented. Only single-frame (T=1) supported.
- [ ] **ControlNet / Reference latents**: `timestep_zero_index` path in `_apply_gate` not implemented.
- [ ] **Image-to-image**: No img2img / inpainting support.
- [x] **Quantized text encoder on GPU**: CUDA LLM runner with GGUF Q4_K weights (12.8s vs 710s CPU).
- [ ] **Multiple resolutions in one session**: DiT weight preloading is resolution-independent, but activation buffers are allocated per-call.

### Code Quality

- [ ] **VAE weight cleanup**: `qimg_vae_free()` doesn't free all weight buffers (memory leak on reload).
- [x] **Debug output gated by verbose levels**: 0=silent, 1=progress, 2=stats, 3=dump .npy.
- [ ] **Error handling**: Some CUDA allocation failures are silently ignored (especially in on-demand block loading).
