# Qwen-Image Text-to-Image Pipeline

End-to-end implementation of the [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) text-to-image model using single-header C libraries with CUDA GPU acceleration.

## Pipeline Overview

```
Text Prompt
    │
    ▼
┌─────────────────┐
│  Text Encoder    │  Qwen2.5-VL-7B (28 layers, GQA)
│  [CPU]           │  → hidden states [N_txt, 3584]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MMDiT           │  60-layer dual-stream DiT
│  [CPU or CUDA]   │  × N_steps (Euler flow matching)
│                  │  latent [16, H/8, W/8]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VAE Decoder     │  3D causal VAE (BF16 weights)
│  [CPU]           │  → RGB [3, H, W]
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
4. RoPE: 2D spatial for image tokens, 1D positional for text tokens
5. Joint attention: concatenate `[txt, img]` for Q/K/V, attend, split back
6. Gated residual: `x += gate × proj(attn_out)`
7. FFN with gated residual: `adaLN → GELU(Linear) → Linear → gated_add`

**Final output:** `adaLN → Linear(3072→64) → unpatchify → velocity [16, H/8, W/8]`

### VAE — 3D Causal Decoder

| Parameter | Value |
|---|---|
| Latent channels | 16 |
| Base channels | 96 |
| Channel multipliers | [1, 2, 4, 4] → [96, 192, 384, 384] |
| GroupNorm groups | 32 |
| Activation | SiLU |
| Weight dtype | BF16 (254 MB) |
| 3D→2D conversion | Sum temporal kernel dimension (for image mode, T=1) |
| Upsample method | Nearest-neighbor 2× + Conv2d 3×3 |

Decoder path: `post_quant_conv(16→16) → conv1(16→384) → mid_block → 15 upsample blocks → norm → conv_out(96→3)`

Blocks 3, 7, 11 are resample-only (spatial upsample, no residual block).

### Scheduler — FlowMatch Euler Discrete

| Parameter | Value |
|---|---|
| Training timesteps | 1000 |
| Shift type | Exponential (dynamic) |
| Base shift | 0.5 |
| Max shift | 0.9 |
| Shift terminal | 0.02 |
| Base image seq len | 256 |
| Max image seq len | 8192 |
| Solver | Euler: `x += dt × velocity` |

Dynamic shift: `mu = clamp((log(n_img) - log(256)) / (log(8192) - log(256)), 0, 1)`, `shift = 0.5 + 0.4 × mu`.

Timestep scaling: `timestep = sigma × 1000` (model expects [0, 1000] range).

### Classifier-Free Guidance (CFG)

- CFG scale: 2.5 (ComfyUI default)
- Formula: `v = v_uncond + cfg_scale × (v_cond - v_uncond)`
- CFGNorm: rescale combined velocity magnitude to match unconditional magnitude

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

Dequantization: 256-entry LUT (`fp8_byte → float32`). NaN value (0x7F) mapped to 0.0.

Scaled FP8 (text encoder): `actual_weight = fp8_to_f32(byte) × scale_per_tensor`.

## Source Files

### Core Headers (`common/`)

| File | Description |
|---|---|
| `qwen_image_dit.h` | 60-layer MMDiT: patchify, joint attention, adaLN, RoPE, SIMD dot product |
| `qwen_image_vae.h` | 3D causal VAE decoder: conv2d, groupnorm, resblocks, upsample |
| `qwen_image_scheduler.h` | FlowMatch scheduler: dynamic shift + ComfyUI-compatible mode |
| `qwen_image_text_encoder.h` | Text encoder wrapper: GGUF + scaled-FP8 safetensors loaders |
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

### PyTorch Reference (`ref/qwen_image/`)

| Script | Description |
|---|---|
| `run_full_pipeline.py` | End-to-end PyTorch reference with block-by-block GPU processing |
| `run_dit_reference.py` | Single DiT block reference with intermediate dumps |
| `run_dit_block_reference.py` | Comprehensive block-level verification |
| `run_vae_reference.py` | VAE decoder layer-by-layer with 22-stage dumps |
| `compare.py` | Numerical comparison tool (max error, correlation) |
| `generate_comfyui.py` | ComfyUI ground truth generator (512×512, 1024×1024) |
| `encode_text_comfyui.py` | Extract text encoder hidden states from ComfyUI |
| `generate_ground_truth.py` | Ground truth image generation |

## Build & Run

### CPU

```bash
cd cpu/qwen_image

# Build
cc -O2 -mavx2 -mfma -I../../common -o test_qwen_image test_qwen_image.c -lm -lpthread

# Full pipeline (safetensors weights)
./test_qwen_image --generate \
    /mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
    /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors \
    /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
    --height 128 --width 128 --steps 20 --prompt "a red apple on a white table"

# Individual tests
./test_qwen_image --test-sched
./test_qwen_image --test-vae /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors
./test_qwen_image --test-dit /mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors
./test_qwen_image --test-enc /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
```

### CUDA

```bash
cd cuda/qimg

# Build (requires cuew.c for CUDA driver API loading)
cc -O2 -mavx2 -mfma -I../../common -I.. -o test_cuda_qimg test_cuda_qimg.c ../cuew.c -lm -ldl -lpthread

# Full pipeline (uses FP8 native GEMM on sm_89+)
./test_cuda_qimg --generate \
    --height 256 --width 256 --steps 20 \
    --prompt "a red apple on a white table"

# Default weight paths (hardcoded, overridable):
#   --dit /mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors
#   --vae /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors
#   --enc /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf

# Options
#   --no-fp8     Force F32 GEMM (disable FP8 weight dequant optimization)
#   --no-cfg     Disable classifier-free guidance (single pass, faster)
#   --seed <n>   Random seed (default: 42)
```

**Text conditioning priority:**
1. ComfyUI pre-encoded `.npy` files (if found at `ref/qwen_image/output/comfyui_text_hidden.npy`)
2. FP8 scaled safetensors text encoder
3. GGUF text encoder fallback

### PyTorch Reference

```bash
cd ref/qwen_image
uv venv && uv pip install torch safetensors numpy pillow

# Generate ComfyUI ground truth
uv run python encode_text_comfyui.py --prompt "a red apple on a white table" --output output/comfyui_text_hidden.npy
uv run python generate_comfyui.py --prompt "a red apple on a white table" --output output/ground_truth_512.png

# Layer-by-layer VAE reference
uv run python run_vae_reference.py --vae-path /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors --output-dir output/

# Compare .npy outputs
uv run python compare.py output/vae_03_mid_res0.npy ../../cpu/qwen_image/vae_03_mid_res0.npy
```

## Implementation Status

### Feature Matrix

| Feature | CPU | CUDA | PyTorch Ref |
|---|:---:|:---:|:---:|
| Text encoder (GGUF) | done | done (fallback) | N/A |
| Text encoder (FP8 safetensors) | done | done | ComfyUI |
| MMDiT forward (60 blocks) | done | done | done |
| FP8 weight loading (safetensors) | done | done (native FP8 + LUT) | N/A |
| GGUF weight loading (Q4_0) | done | — | done |
| AVX2/FMA SIMD (CPU DiT) | done | N/A | N/A |
| FP8 native GEMM (sm_89+) | N/A | done | N/A |
| FP8→F32 LUT dequant GEMM | N/A | done | N/A |
| Joint attention (online softmax) | done | done | done |
| Per-head QK RMSNorm | done | done | done |
| 2D RoPE (image) + 1D RoPE (text) | done | done | done |
| adaLN-Zero modulation | done | done | done |
| Patchify / Unpatchify | done | done (CPU) | done |
| FlowMatch scheduler (dynamic shift) | done | done | done |
| ComfyUI-compatible scheduler | done | done | done |
| CFG + CFGNorm | — | done | done |
| VAE decoder | done | **broken** | done |
| Block preloading (GPU VRAM) | N/A | done (41/60 blocks) | N/A |
| Layer-by-layer verification | 34/34 PASS | partial | baseline |

### Verified Resolutions (CUDA)

| Resolution | Latent | Img Tokens | Status | Time/step |
|---|---|---|---|---|
| 64×64 | 8×8 | 16 | working | ~1.5s |
| 128×128 | 16×16 | 64 | working | ~1.7s |
| 256×256 | 32×32 | 256 | working | ~9.4s |
| 512×512 | 64×64 | 1024 | working (slow) | ~62s |

GPU: NVIDIA RTX 5060 Ti (sm_120, 16 GB VRAM). Times include CFG (2 DiT passes per step).

### Known Issues

1. **CUDA VAE decoder** produces flat output (R=0.251, G=0.263, B=0.086, std=0.000). CPU VAE is used as fallback. Root cause not yet identified.

2. **512×512+ is slow** due to O(n²) attention kernel (single block per head, sequential query processing). Needs tiled/flash attention for production use.

3. **`ulimit -v` breaks CUDA allocations.** Host virtual memory limits affect CUDA driver API's unified virtual addressing. Do not use `ulimit -v` with the CUDA binary.

### Critical Bugs Fixed

| Bug | Symptom | Fix | Commit |
|---|---|---|---|
| Missing `cuStreamSynchronize` | Latent drift at >64×64 (min→-187, max stuck at 3.17) | Add sync before `cuMemcpyDtoH` | `3c4b4ac` |
| FP8 MMA quantizes both inputs | 160× scale error in modulation (GPU vs CPU) | New `gemm_fp8w_f32` kernel: FP8 weights via LUT, F32 inputs | `a32452a` |
| Timestep = sigma (not sigma×1000) | Denoising doesn't converge | `timestep = sigma * 1000` | `133847e` |
| Sinusoidal embedding order | Wrong timestep encoding | Swap to `[cos, sin]` (flip_sin_to_cos=True) | `569fce1` |
| FP8 NaN (0x7F) | NaN propagation through blocks | Map 0x7F → 0.0f in LUT | `70b05d0` |
| Text encoder segfault | `thread_tmp` not allocated | Add `calloc` for scratch buffer | `dbda87d` |
| Attention warp reduction | 128 threads, only 32 reduced | Shared memory tree reduction | `c3b13e7` |

### CPU Performance (FP8 safetensors, AVX2)

| Component | 64×64 | 128×128 |
|---|---|---|
| DiT (20 steps, no CFG) | ~35s/step | ~35s/step |
| VAE decode | ~10s | ~42s |
| Text encoder | ~5s | ~5s |

### CUDA GPU Memory Usage

- Global DiT weights: ~50 MB (F32 biases + norms)
- Per-block weights: ~324 MB (FP8 raw)
- Preloaded blocks: 41/60 (13.3 GB)
- Workspace reservation: 2 GB (activations + scratch)
- Total VRAM: ~15.3 GB used of 16 GB

## Reproduction

### Generate a 256×256 image (CUDA, ~3 min)

```bash
cd cuda/qimg
cc -O2 -mavx2 -mfma -I../../common -I.. -o test_cuda_qimg test_cuda_qimg.c ../cuew.c -lm -ldl -lpthread
./test_cuda_qimg --generate --height 256 --width 256 --steps 20 --prompt "a red apple on a white table"
# Output: cuda_qimg_output.ppm (256×256 PPM image)
```

### Generate a 128×128 image (CPU only, ~15 min)

```bash
cd cpu/qwen_image
cc -O2 -mavx2 -mfma -I../../common -o test_qwen_image test_qwen_image.c -lm -lpthread
./test_qwen_image --generate \
    /mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
    /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors \
    /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
    --height 128 --width 128 --steps 20 --prompt "a red apple on a white table"
# Output: qwen_image_output.ppm
```

### Verify against PyTorch reference (VAE layer-by-layer)

```bash
# 1. Generate PyTorch reference dumps
cd ref/qwen_image
uv run python run_vae_reference.py \
    --vae-path /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors \
    --output-dir output/ --latent-h 8 --latent-w 8 --seed 42

# 2. Run CPU VAE with same input
cd ../../cpu/qwen_image
./test_qwen_image --test-vae /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors \
    --height 64 --width 64

# 3. Compare each layer
cd ../../ref/qwen_image
uv run python compare.py output/vae_03_mid_res0.npy ../../cpu/qwen_image/vae_03_mid_res0.npy
# Expected: correlation = 1.000000, max error < 1e-4
```

### Generate ComfyUI ground truth for comparison

```bash
cd ref/qwen_image

# Encode text (requires ComfyUI installed at /mnt/disk01/ComfyUI)
uv run python encode_text_comfyui.py \
    --prompt "a red apple on a white table" \
    --output output/comfyui_text_hidden.npy

# Generate image
uv run python generate_comfyui.py \
    --prompt "a red apple on a white table" \
    --height 512 --width 512 --steps 30 \
    --output output/ground_truth_comfyui_512.png
```

Ground truth images available at `ref/qwen_image/output/`:
- `ground_truth_comfyui_512.png` (512×512)
- `ground_truth_comfyui_1024.png` (1024×1024)
- `ground_truth_comfyui_512_latent.npy` (pre-VAE latent for comparison)
