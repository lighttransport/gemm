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

The ComfyUI single-file FP8 weights live in the [`Comfy-Org/Qwen-Image_ComfyUI`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI) HuggingFace repo:

- DiT (original): <https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors>
- Text encoder: <https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors>
- VAE: <https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors>

The **Qwen-Image-2512** refresh (Dec 2025) has its own single-file FP8 release on the [Unsloth tutorial page](https://unsloth.ai/docs/jp/moderu/tutorials/qwen-image-2512):

- DiT (2512 FP8): <https://huggingface.co/unsloth/Qwen-Image-2512-FP8/resolve/main/qwen-image-2512-fp8.safetensors> (~20 GB)
- Also available as 4-bit BnB: <https://huggingface.co/unsloth/Qwen-Image-2512-unsloth-bnb-4bit>
- And as GGUF: <https://huggingface.co/unsloth/Qwen-Image-2512-GGUF>

The pytorch-rocm reference under `ref/qwen_image/` defaults to loading the local ComfyUI FP8 DiT via `diffusers.QwenImageTransformer2DModel.from_single_file(...)` so it produces an apples-to-apples comparison against the HIP runner. Pass `--fp8-dit <path>` to point at a different single-file checkpoint (e.g. the 2512 release), or `--fp8-dit ""` to fall back to the BF16 multi-shard transformer from the official `Qwen/Qwen-Image` HF repo.

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
| `test_cuda_qimg.c` | CUDA test harness: init, load, dit step, full generation, `--test-kernels` |
| `test_fp8_gemm.c` | FP8 GEMM correctness test (all sizes PASS, corr=1.0) |
| `gen_compare.py` | End-to-end ComfyUI-vs-ours composite generator (left ours \| diff heatmap \| right comfy) |
| `defsched.c` | Standalone test that runs the deferred scheduler path |

### Shared GPU kernel sources (`cuda/`)

| File | Description |
|---|---|
| `cuda_fp8_mma_kernels.h` | NVRTC source string with all FP8/BF16 MMA kernels: `gemm_fp8_scaled_f32` (m16n8k32 e4m3), `gemm_fp8_scaled_f32_pipe` (cp.async + ldmatrix), `gemm_fp8_pipe_perrow_f32` (per-row X scale), `gemm_bf16_pipe_f32` (BF16 MMA with FP8→BF16 inline dequant), `flash_attn_fp8`, `flash_attn_bf16`, `reduce_max_abs_per_row_f32`, `cast_f32_to_bf16`. Shared with `cuda/flux2/`. |
| `cuda_kernels_common.h` | NVRTC source for non-MMA kernels (LUT GEMMs, layernorm, scheduler ops, VAE conv, etc.) |
| `cuda_runner_common.h` | Host-side CUDA driver helpers: NVRTC compilation, error macros, weight upload |

### CUDA LLM Runner (`cuda/llm/`)

| File | Description |
|---|---|
| `cuda_llm_runner.{h,c}` | GPU LLM inference: Qwen2/2.5-VL/3/3.5, GGUF weights, Q/K/V biases |
| `test_cuda_llm.c` | LLM test: CPU vs GPU hidden state comparison |

### HIP/ROCm Implementation (`rdna4/qimg/`)

Targets RDNA4 (gfx1200/gfx1201). Kernels compiled at runtime via HIPRTC (no `hipcc` required).

| File | Description |
|---|---|
| `hip_qimg_runner.{c,h}` | GPU runner: HIPRTC kernel compilation, FP8 LUT GEMM, optional BF16 WMMA matrix-core path, per-block streaming with up to ~40 preloaded blocks on a 16 GB card, GPU VAE decode |
| `hip_qimg_kernels.h` | HIP kernel strings: `gemm_fp8w_f32` (16×16 scalar LUT), `gemm_opt_fp8` (128×128 tiled LUT with fused BF16 trunc), `gemm_fp8w_bf16a_wmma_t` (BF16×FP8 matrix-core WMMA, 256 threads / 128×128 CTA) |
| `test_hip_qimg.c` | Test harness: `--test-init / --test-dit / --test-vae / --test-enc / --generate`. Supports pinned-input options (`--init-bin`, `--txt-bin`, `--sigmas-bin`, `--dump-final`, `--skip-unstd`) plus PNG output via `stb_image_write.h`. |
| `verify_qimg_dit.c` | CPU↔HIP single-step DiT comparison (max_diff / mean_diff / corr) |
| `Makefile` | Builds `test_hip_qimg` and `verify_qimg_dit` |

**FP8 LUT GEMM** (default ON when the FP8 LUT compiles): raw FP8 E4M3 bytes are uploaded to VRAM (~4× smaller than F32) and dequantized via a 256-entry constant-memory LUT. Qwen-Image FP8 weights have **no per-tensor scale** (unlike Flux.2 Klein), so the kernels do not take a `w_scale` argument.

**BF16×FP8 WMMA** (`QIMG_FP8_WMMA=1`): same 128×128 CTA tile layout as the Flux.2 Klein WMMA path — 256 threads = 8 waves arranged 2×4, each wave computes a 64×32 sub-tile via 4×2 = 8 `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` instructions reusing 8 KB of shared memory per K=16 slab. Activations F32→BF16 truncated at SMEM load; FP8 weights LUT-dequanted then BF16-truncated; F32 accumulator. Verified on RX 9070 XT via `verify_qimg_dit --lat 8`: corr ≈ 0.9765 vs CPU reference (matches the LUT path's 0.9777 floor — the BF16-trunc-on-output noise dominates either way).

| Mode | 256×256 / 20-step DiT | 512×512 / 20-step DiT | VAE 256 | VAE 512 |
|---|---|---|---|---|
| HIP 128×128 tiled FP8 LUT (default) | **36.1 s (1.80 s/step)** | **91.2 s (4.56 s/step)** | 1.5 s | 20.5 s |
| HIP 128×128 BF16×FP8 WMMA (`QIMG_FP8_WMMA=1`) | **29.4 s (1.47 s/step)** | **81.8 s (4.09 s/step)** | 1.5 s | 20.2 s |

WMMA gives ~18 % step-time speedup at 256×256 and ~10 % at 512×512. End-to-end gain is modest because per-block PCIe streaming and (especially at 512×512) the VAE decode dominate; ~40/60 DiT blocks are preloaded into VRAM and the remaining 20 are streamed each step.

**Apples-to-apples vs diffusers reference** (256×256 / 20 steps / cfg=1, both runners loading the same `qwen_image_fp8_e4m3fn.safetensors` and using `init_latent_256.bin / apple_text_256.bin / sigmas_256.bin` from `dump_diffusers_pipeline.py`):

| | diffusers (FP8, sequential cpu offload) | HIP BF16 WMMA |
|---|---|---|
| Wall time | 102 s | **34 s** |
| Final latent cosine vs ref | 1.0 (self) | **0.999962** |
| Final latent max \|diff\| | 0 | 0.0273 |
| PNG PSNR vs ref | ∞ | **49.48 dB** |
| PNG mean \|diff\| | 0 | 0.54 / 255 |

The HIP runner is **3× faster** than the pytorch-rocm diffusers reference (sequential cpu offload is forced because the BF16 transformer doesn't fit in 16 GB VRAM resident — even with the FP8 transformer loaded via `from_single_file`, the text encoder + VAE + activations push past 16 GB during inference). The BF16 WMMA quantization noise on activations introduces only ~0.5 LSB of pixel error, so the apple is visually indistinguishable from the diffusers reference at PSNR ≈ 49 dB.

### PyTorch Reference (`ref/qwen_image/`)

### PyTorch Reference (`ref/qwen_image/`)

| Script | Description |
|---|---|
| `gen_diffusers_reference.py` | **diffusers-based** reference (`QwenImagePipeline`, pytorch-rocm 7.2, BF16 + cpu offload). Generates `apple_ref_<size>.png`. |
| `dump_diffusers_pipeline.py` | Same pipeline plus dumps `apple_text*.bin`, `init_latent*.bin`, `final_latent_packed*.bin`, `sigmas*.bin`, `vae_meta*.txt` for layer-by-layer comparison against the HIP runner via `--init-bin`/`--txt-bin`/`--sigmas-bin`. |
| `make_comparison.py` | Build a 2×2 `apple_benchmark.png` (diffusers vs HIP at 256 / 512). |
| `generate_reference.py` | (legacy) ComfyUI reference data generator |
| `verify_pipeline.py` | End-to-end verification against ComfyUI reference |
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
| **DiT img_in FP8 linear** | corr=0.999547 | MMA kernel vs ComfyUI reference (FP8 noise floor) |
| **Full pipeline latent** | corr=0.999107 | 256×256/10-step vs cf_ournoise2 reference, PSNR 51 dB |
| **Scheduler sigmas** | exact match | Pre-shifted table + simple sampling |

### Performance (CUDA, RTX 5060 Ti 16GB)

**Denoising-only per-step (seed=42, cfg=2.5, simple scheduler, warm cache,
ComfyUI-encoded text hidden states for a fair correctness comparison):**

| Mode | 256×256 (10 steps) | 512×512 (4 steps) | mean diff vs ComfyUI | Gap vs ComfyUI |
|---|---:|---:|---:|---:|
| ComfyUI (cuDNN+cuBLAS) | 1.28 s/step | 1.61 s/step | 0.00 | 1.00× |
| ours — LUT scalar (gold) | 3.49 s/step | 7.90 s/step | 1.00 | 2.73×–4.91× |
| ours — LUT + BF16 attn | 3.43 s/step | 6.93 s/step | 0.79 | 2.68×–4.30× |
| ours — FP8 MMA + scalar attn | 2.04 s/step | — | 1.81 | 1.59× |
| ours — FP8 MMA + FP8 attn (old default) | 1.98 s/step | 3.02 s/step | 5.01 | 1.55×–1.88× (blurry) |
| ours — BF16 MMA + BF16 attn | 3.66 s/step | 9.14 s/step | 0.85 | 2.86×–5.68× |
| ours — PER-ROW FP8 MT2 + BF16 attn | 1.98 s/step | 2.96 s/step | 1.88 | 1.55×–1.84× |
| ours — PER-ROW FP8 MT4 + vectorized BF16 attn | 2.00 s/step | 2.74 s/step | 1.88 | 1.56×–1.70× |
| ours — + cp.async ldmatrix.trans BF16 attn | 2.00 s/step | 2.67 s/step | 1.88 | 1.56×–1.66× |
| **ours — + CFG-batched img MLP (new default)** | **1.95 s/step** | **2.65 s/step** | **1.88** | **1.52×–1.65×** |

The new default reaches **2.98× faster than gold at 512×512 (7.90 → 2.65
s/step)** while staying visually indistinguishable from ComfyUI/gold (mean
pixel diff 1.88 / 255 ≈ 0.7%). At 256×256 it hits **1.52× ComfyUI** (1.95 vs
1.28 s/step). At 512×512 the gap is 1.65×.

Stable 3-run averages (256×256/10 steps, 512×512/5 steps) across this
session: cp.async + ldmatrix.trans saves ~70 ms/step at 512 (2.74 → 2.67).
CFG-batched image MLP (fc1 + GELU + fc2) saves another ~50 ms at 256 and
~20 ms at 512 — at 256 the small n_img (1024) means doubling to 2048 rows
lands in a more efficient MMA tile shape; at 512 n_img=4096 is already
saturated so the gain is mostly W-traffic dedup.

**Recommended invocation (new default):**

```bash
QIMG_FP8_PIPE_PERROW=1 QIMG_BF16_ATTN=1 ./test_cuda_qimg --generate \
    --height 512 --width 512 --steps 4 --seed 42 \
    --prompt "a red apple on a white table"
```

**Env-var matrix (all opt-in until validated, paths fall back automatically):**

| Env var | Effect |
|---|---|
| `QIMG_FP8_MMA=1` | Per-tensor FP8 MMA pipe — fast but ~+0.8 mean blur |
| `QIMG_FP8_PIPE=1` | Cp.async pipelined per-tensor FP8 MMA (under `QIMG_FP8_MMA=1`) |
| `QIMG_FP8_PIPE_PERROW=1` | **Per-row FP8 MMA pipe** — same speed as per-tensor, ~half the blur |
| `QIMG_FP8_ATTN=1` | FP8 flash attention — fast but visible blur (mean +4.4) |
| `QIMG_BF16_ATTN=1` | **BF16 flash attention** — matches ComfyUI BF16 reference |
| `QIMG_BF16_MMA=1` | BF16 MMA GEMM (FP8 weights → BF16 in-kernel) — gold quality, slower |

Correctness: corr=0.999107 vs `cf_ournoise2_10step_latent.npy`, every channel corr ≥ 0.998.

**FP8 stack wins from this optimization pass** (512×512, 4-step):
- LUT baseline:                            21.85 s/step
- +FP8 MMA GEMM (fix P7'):                 11.55 s/step  (1.89×)
- +scratch-slot block loader:              10.22 s/step  (2.14×)
- +FP8 flash attention:                     5.05 s/step  (4.33×)
- +CFG batching, cp.async pipe FP8 MMA:     3.02 s/step  (7.23×)
- +per-row FP8 MMA + BF16 attention:        2.96 s/step  (7.38×)
- +MTILE=4 GEMM + vectorized BF16 attn:     2.74 s/step  (7.97×)
- +cp.async + ldmatrix.trans BF16 attn:     2.67 s/step  (8.18×)
- **+CFG-batched img MLP:                   2.65 s/step  (8.24×)**

(Apple gen at 512×512: 2.69 s/step is **2.94× faster** than the LUT scalar
gold path 7.90 s/step, **1.67× slower** than ComfyUI 1.61 s/step.)

**Dominant remaining cost**: PCIe block loading. Only 11/60 blocks fit in 16 GB VRAM
at 512×512 so 48 blocks are loaded on-demand per forward (~85 ms each through
pageable HtoD). Per-block profile at 512×512:

```
img_q/k/v         2.7 ms    txt_qkv        1.5 ms
attn (FP8)        1.2 ms    rmsnorm_ph     0.1 ms
img_attn_out      0.9 ms    rope           0.2 ms
img_fc1           5.6 ms    img_fc2        4.5 ms
img_adaLN         0.1 ms    txt_mlp_all    2.9 ms
─────────────────────────
img block compute ~20 ms    block load (on-demand) ~85 ms
```

**GPU memory strategy:** Text encoder (4.1 GB) runs first, then releases VRAM. DiT
weights (13.3 GB in FP8) loaded after. Per-runner pre-allocated scratch block slot
(324 MB) reused for all on-demand blocks — eliminates the 30 cuMemAlloc/cuMemFree
per-block driver calls. Both text encoder and DiT share the primary CUDA context
via `cuDevicePrimaryCtxRetain`.

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
| Baseline (16×16 GEMM, CPU text encoder) | 748 s | 1× |
| 128×128 tiled GEMM with register blocking | 714 s | 1.05× |
| Multi-threaded CPU text encoder (16 threads) | 710 s | 1.05× |
| GGUF Q4_K encoder + injected biases | 710 s | 1.05× |
| **GPU text encoder via CUDA LLM runner** | **52 s** | **14.3×** |
| FP8 MMA GEMM default (P7' per-tensor input scaling) | 37 s (10-step 256²) | 1.4× |
| Scratch-slot block loader + hoisted alloc | 38 s | ~1.0× |
| FP8 flash attention (P6 device-ptr scales) | 37 s | 20× total vs 748 s |
| CFG batching + bf16 fusion (single-block forward) | 19.8 s | 1.87× |
| FP8 MMA cp.async + ldmatrix pipe | 19.8 s | bit-identical, +0% |
| **BF16 flash attention** (matches ComfyUI BF16 ref) | 19.8 s | mean diff 5.01 → 2.09 |
| **BF16 MMA GEMM** (FP8 → BF16 in-kernel) | 36.6 s @ 256² | gold quality, opt-in |
| **Per-row FP8 MMA pipe** (one X scale per row) | 19.8 s | mean diff 1.81 → 1.65 |
| Per-row FP8 + BF16 attn | 19.8 s | mean 1.88, ~1.55× ComfyUI |
| **+MTILE=4 GEMM + vectorized BF16 attn load (new default)** | **19.4 s @ 256² / 10.4 s @ 512² (4-step)** | **1.52×–1.62× ComfyUI** |

### Recently fixed FP8 bugs (cuda/qimg, 2026-04)

| Bug | Symptom | Fix |
|---|---|---|
| **FP8 MMA 3× velocity inflation** | End-to-end corr=0.43, latent magnitudes 3× too big | Add per-tensor input scaling to `gemm_fp8_scaled_f32`: reduce_max_abs + divide-at-load / multiply-at-writeback. GELU output from mlp_fc1 peaks at ~650, saturated the `cvt.rn.satfinite.e4m3` at ±448. (commit `a99b9a3`) |
| **Pipeline non-determinism** | v_cond_std varied 0.39..3.28 across runs with same seed | `cuMemcpyDtoD` on default stream raced with t_emb ops on r->stream. Use `cuMemcpyDtoDAsync(…, r->stream)`. (commit `97f426b`) |
| **Wrong latent compare space** | corr 0.03 vs ComfyUI reference | ComfyUI `sample()` output is post-Wan21 `process_out` (VAE-native space). We saved pre-Wan21 (DiT-normalized). Added `cuda_latent_vae.npy` dump after the `latent * wan21_std + wan21_mean` step. (commit `97f426b`) |
| **Per-block cuMemAlloc overhead** | ~85 ms/block load, 1440 driver calls/step | Pre-allocated `scratch_block` slot in runner, on-demand `qimg_load_block_into_slot` uses async HtoD into the shared buffers. (commit `4e22c88`) |
| **flash_attn_fp8 host DtoH stall** | 1-2 ms per attention call | Pass device pointer `&r->d_qkv_scales` to kernel, drop the per-call `cuMemcpyDtoH(scales, 3*f32)`. (commit `95e5536`) |
| **Double Wan21 denormalization** | Apple visibly washed-out / blurry, mean diff 47 vs comfy | `--generate` applied the `latent * std + mean` affine twice before VAE: once via inline block, then again via `qimg_dit_unnormalize_latent()`. Removed the redundant call. (commit `3217cfd`) |
| **FP8 attention per-tensor scale crush** | Visible blur, mean diff 5.01 vs comfy | Per-tensor `max(|Q|)/448` quantization shared one scale across all 24 heads — outlier head crushes precision in the rest. Added `flash_attn_bf16` kernel using `mma.sync.aligned.m16n8k16.bf16.bf16.f32` with no quantization, matching ComfyUI's BF16 reference. (commit `02e830c`) |
| **FP8 MMA X-side outlier blur** | Mean diff 1.81 vs comfy (FP8 MMA default) | Per-tensor `max(\|X\|)/448` outlier compressed precision in non-outlier rows. Added `gemm_fp8_pipe_perrow_f32` + `reduce_max_abs_per_row_f32` so each output row gets its own X scale. mean drops 1.81 → 1.65, max diff 126 → 61, same speed. (commit `56e823c`) |

## TODO / FIXME

### Performance

- [x] **GEMM optimization**: 128×128 tiled GEMM with 8×8 register blocking + fused BF16 truncation. 6.67s → 3.67s/step (1.82×).
- [x] **GPU text encoder**: CUDA LLM runner with GGUF Q4_K weights + injected Q/K/V biases. 710s → 12.8s (56×).
- [x] **FP8 MMA GEMM** (sm_89+, per-tensor input scaling). `gemm_fp8_scaled_f32` via `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. Opt-in via `QIMG_FP8_MMA=1`.
- [x] **FP8 MMA cp.async pipelined GEMM** (`gemm_fp8_scaled_f32_pipe`). 2-stage W double-buffer + `ldmatrix.sync.aligned.m8n8.x4` A frags + pre-quantized smX. ~5-9% over the per-tensor MMA. Opt-in via `QIMG_FP8_PIPE=1`.
- [x] **Per-row FP8 MMA pipe** (`gemm_fp8_pipe_perrow_f32` + `reduce_max_abs_per_row_f32`). One X scale per output row instead of per-tensor → mean diff 1.81 → 1.65, max diff 126 → 61, same speed. Opt-in via `QIMG_FP8_PIPE_PERROW=1`.
- [x] **MTILE=4 per-row FP8 pipe** (`gemm_fp8_pipe_perrow_mt4_f32`). Doubles M coverage per CTA (32→64 rows) — halves row-CTA count at 512×512. Uses `"+f"` accumulator operands so the compiler keeps d0..d3 live through the doubled inner loop (the previous MTILE=4 attempts crashed on `"=f"`/`"f"` aliasing). Auto-dispatched under `QIMG_FP8_PIPE_PERROW=1` when `n_tok % 64 == 0`. **Recommended GEMM path.** 2.92 → 2.61 s/step at 512×512.
- [x] **Vectorized BF16 attention cooperative load**. Replaces the per-thread single-BF16 K/V load loop with a 4-BF16-per-thread `uint2`-sized read. Halves per-thread iter count (32→8). Tried `BKV=64` (slower due to 5→2 CTAs/SM occupancy drop) and full cp.async pipeline (blocked by V transpose — `cp.async` cannot do strided/transposed copies). Vectorized load on `BKV=32` keeps occupancy at 5 CTAs/SM and shaves ~50 ms.
- [x] **BF16 flash attention** (`flash_attn_bf16` via `mma.sync.aligned.m16n8k16.bf16.bf16.f32`). Matches ComfyUI's BF16 reference precision exactly — drops mean diff from 5.01 (FP8 attn) to 0.79–1.88 depending on GEMM path. Opt-in via `QIMG_BF16_ATTN=1`. **Recommended attention path.**
- [x] **BF16 MMA GEMM** (`gemm_bf16_pipe_f32`). Reads FP8 weights, decodes FP8 → BF16 inline at MMA-load time via `d_fp8_to_bf16_lut[256]` constant memory. Best correctness (mean 0.85) but ~2× slower than FP8 MMA on Blackwell consumer (BF16 m16n8k16 throughput limit). Opt-in via `QIMG_BF16_MMA=1`.
- [x] **FP8 flash attention** (FA2-style, head_dim=128, 4 warps/CTA, device-pointer scales). Opt-in via `QIMG_FP8_ATTN=1`. 7.5× vs F32 attention at 512×512 but introduces visible blur — superseded by `QIMG_BF16_ATTN=1`.
- [x] **Block scratch slot** + hoisted modulation allocs. Eliminates 30 cuMemAlloc/cuMemFree per on-demand block (~1440 driver calls/step → 30).
- [x] **CFG batching**: cond/uncond run through each block under a single block-weight load (`cuda_qimg_dit_step_cfg`). Halves per-block PCIe traffic.
- [ ] **Block streaming / prefetch**: Still ~85 ms/block PCIe load (48/60 on-demand at 512×512). `cuMemHostRegister` on the mmap'd safetensors was rejected (operation not supported); a pinned-staging ring was net-slower because the driver already pipelines pageable HtoDAsync via its own copy engine. Real fix needs `cp.async.bulk` (sm_90+) which Blackwell consumer doesn't expose.
- [x] **MTILE=4 GEMM**: ~10% gain at 512×512, see entry above.
- [x] **cp.async + ldmatrix.trans BF16 flash attention**. Switched `smV` from
  col-major `smVT` to row-major so both K and V can be loaded via contiguous
  `cp.async.ca.shared.global` into smem. The PV MMA's B-operand is now loaded
  with one `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` per N-tile — the
  4 result regs map directly to the m16n8k16 B layout we need for both PV
  K-chunks without any per-lane packing. Additionally, `P` is held in per-lane
  `u32` registers (`P_pack[kc][4]`) instead of round-tripping through `smP`
  — each lane already owns its own `P_data`, and the sm_120 a1↔a2 swap is
  applied in-register. smem drops 20 KB → 16 KB (5 → 6 CTAs/SM occupancy).
  Single-buffer `cp.async` (no prefetch) chosen over 2-stage double-buffer
  (32 KB, 3 CTAs/SM) because the occupancy hit outweighed the pipeline gain
  at 256×256. Saves ~70 ms/step at 512×512 (2.74 → 2.67).
- [x] **CFG-batched image MLP**. `cuda_qimg_dit_step_cfg` now allocates one
  contiguous `[2*n_img, dim]` MLP input buffer (`d_img_mlp_in`) and one
  `[2*n_img, mlp_h]` intermediate (`d_img_mlp_h`). Both the cond and uncond
  `forward_block` calls skip their img MLP; instead they write the img
  adaLN2 output into the first / second half of `d_img_mlp_in` via a new
  `img_mlp_in_external` argument. A single batched `fc1 + gelu + fc2` then
  runs on 2*n_img rows, halving W traffic for the two largest GEMMs in the
  block (mlp_h=14336 × dim=3072 each direction). The post-MLP `gated_add` is
  dispatched per-pass by slicing `d_img_mlp_out` into halves. To make this
  fit VRAM, `d_scratch1` / `d_scratch2` / `d_scratch3` are shrunk to their
  actual minimum sizes (`n_img*dim`, `n_txt*dim`, `n_txt*mlp_h` respectively)
  — the img MLP intermediate no longer lives in `d_scratch3`. Saves ~50 ms
  at 256×256 and ~20 ms at 512×512. The 256×256 win is larger because n_img
  = 1024 was below the MMA kernel's preferred M tile count; doubling to 2048
  fills the CTAs more evenly.
- [ ] **Attention kernel for n_tok > 1536**: `flash_attn_bf16` handles our sizes, but won't scale beyond 2K tokens without tiling.
- [x] **VAE middle attention on GPU**. Replaced the CPU DtoH/HtoD round-trip
  and O(N²D) Python-like loop with two F32 transpose kernels (`[c, spatial]
  ↔ [spatial, c]`) plus a warp-per-query online-softmax kernel
  (`vae_attn_sc_f32`) that streams K and V tokens from row-major smem.
  512×512 VAE decode drops from **21.6 s → 1.7 s** (12.7×); the middle
  attention phase alone drops **19.7 s → 0.04 s**. 256×256 VAE drops
  **1.5 s → 0.5 s**. Correctness unchanged (mean pixel diff 1.877 / 255).
- [x] **1328×1328 (Qwen-Image max res)** w/ memory-pressure aware eviction.
  The empirical 256 MB safety margin in `qimg_evict_preloaded_until_free`
  is tuned to keep post-alloc free VRAM above a threshold where the CUDA
  driver starts doing heavy per-launch bookkeeping: at 1328 with only
  50 MB free after alloc, every block was running **6.7× slower on the
  first step** (95 s/step one-time penalty → 14 s/step steady). Kernels
  are identical but driver overhead on memory-tight launches is not.
  Bumping the margin to 256 MB restores steady-state speed on the first
  step. Measured at 1328x1328 (10-step gen): **13.96 s/step DiT, 3.3 s
  VAE**. 1024 and 512 paths unchanged.
- [x] **1024×1024 generation** with dynamic preload eviction + tiled im2col.
  `cuda_qimg_dit_step_cfg` now computes the activation working set up-front
  (`d_img`, QKV, CFG-batched MLP buffers, scratch) and calls a new helper
  `qimg_evict_preloaded_until_free` to drop preloaded DiT blocks from the
  tail until free VRAM covers the workload plus a 64 MB safety margin.
  Freed blocks automatically fall back to the existing on-demand
  `scratch_block` / `scratch_block_b` loader. VAE `vae_op_conv2d_mma` also
  tiles im2col along the token axis so the unfold buffer is capped at
  ~256 MB regardless of spatial dims — the 1024×1024 96→96 conv would
  otherwise need 3.5 GB of im2col buffer alone.
  Measured at 1024×1024:
  - 16 GB default (45/60 preloaded): **6.6 s/step** DiT, **1.6 s** VAE.
  - 8 GB simulated (`QIMG_MAX_PRELOAD=16`): 8.5 s/step DiT, 1.6 s VAE — the
    ~1.9 s/step regression is pure disk I/O for the 44 streamed blocks.
  - Pure on-demand (`QIMG_MAX_PRELOAD=0`): 9.9 s/step DiT.
  512×512 and 256×256 are unaffected.
- [x] **Tensor-core VAE conv2d via im2col + FP8 MMA**. `vae_op_conv2d`
  now routes 3×3 convs with `ci*kh*kw % 32 == 0` through:
  (1) on-device `vae_f32_to_fp8_padded` kernel that quantizes the F32
  weight to e4m3 and pads `[co, n_in]` up to `[pad_co, n_in_pad]` with
  `pad_co % 256 == 0` and `n_in_pad % 32 == 0`;
  (2) `vae_im2col_f32` unfolds the F32 input `[ci, H, W]` into
  `[H*W, n_in_pad]` row-major with zero-padded trailing K cols;
  (3) the existing `gemm_fp8_pipe_perrow_mt4` FP8 MMA tensor-core GEMM
  produces `[H*W, pad_co]` in F32;
  (4) `vae_crop_transpose_add_bias_f32` crops the padded cols, transposes
  to `[co, H*W]` CHW, and fuses the bias add.
  512×512 VAE decode drops **1.7 s → 0.5 s** (3.4×); 256×256 drops
  **0.5 s → 0.2 s** (2.5×). Per-phase conv speedups are 3-6× depending on
  resolution. Mean pixel diff shifts from 1.877 → 2.099 (~0.09% increase)
  from the FP8 weight quantization — visually indistinguishable. Layers
  with `n_in % 32 != 0` (head conv 96→3, and conv1 16→384 where `ci*9=144`)
  fall back to the naive kernel.
- [ ] **AdaLN + GEMM fusion**: 4 standalone adaln kernels per block → fuse into GEMM input load. ~5-10% DiT speedup.

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
