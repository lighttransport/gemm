# Flux.2 Klein 4B Text-to-Image Pipeline

End-to-end implementation of [Flux.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) in single-header C libraries with CUDA GPU acceleration. Supports both the distilled (4-step) and base (20-step) variants.

## Pipeline Overview

```
Text Prompt
    │
    ▼
┌─────────────────┐
│  Text Encoder    │  Klein/Qwen3 text encoder
│  [CUDA or CPU]   │  2-shard safetensors weights
│                  │  → hidden states [N_txt, 7680]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DiT             │  5 double-stream + 20 single-stream blocks
│  [CUDA]          │  FP8 E4M3 weights, F32 GEMM
│                  │  × N_steps (Euler flow matching)
│                  │  latent [32, H/8, W/8]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VAE Decoder     │  2D standard VAE, GroupNorm32 (SDXL-family style)
│  [CUDA]          │  → RGB [3, H, W]
└────────┬────────┘
         │
         ▼
    Output Image
```

## Model Specifications

### Text Encoder — Klein/Qwen3 Text Encoder

| Parameter | Value |
|---|---|
| Architecture | `qwen3` (GQA transformer) |
| Hidden dim | 2560 |
| Output projected dim | 7680 (3 layer snapshots × 2560) |
| Layers | 36 |
| Attention heads | 32 (8 KV heads, 4:1 GQA) |
| Head dim | 128 |
| FFN intermediate | 9728 |
| RoPE theta | 1,000,000 |
| Activation | SiLU (SwiGLU) |

Output: per-token hidden states `[N_real_tokens, 7680]`, built by concatenating hidden snapshots after layers 8, 17, and 26. Only real tokens are returned, so `N_txt` is typically 10–20 for short prompts.

**Weights:** CPU and GPU paths both use the Klein `text_encoder/` safetensors shards. The GGUF file is used only for tokenizer/BPE vocab.

**Chat template:** The implementation uses a text-only prompt path, not the full VL chat template.

### DiT — Flux.2 Klein (5+20 Block Architecture)

Confirmed architecture from `flux-2-klein-4b-fp8.safetensors` weight inspection:

| Parameter | Value |
|---|---|
| Hidden dim | 3072 |
| Attention heads | 24 |
| Head dim | 128 |
| MLP intermediate (n_ff) | 9216 (= 3 × hidden_dim) |
| Double-stream blocks | 5 |
| Single-stream blocks | 20 |
| Text input dim | 7680 (txt_dim projected to hidden_dim) |
| Image input dim | 128 (patch_in_channels = 32 latent ch × 2×2 patch) |
| Latent channels | 32 |
| Patch size | 2×2 (VAE spatial, transparent to DiT: DiT sees H/8 × W/8 tokens) |
| Weight dtype | FP8 E4M3 with per-tensor `weight_scale` |
| MLP activation | SwiGLU |

**Key structural differences from Qwen-Image MMDiT:**

| Feature | Qwen-Image | Flux.2 Klein |
|---|---|---|
| Modulation | Per-block adaLN-Zero | **Global shared** (one mod weight for all double blocks, one for all single) |
| Blocks | 60 double-stream | 5 double + 20 single-stream |
| MLP | GELU | SwiGLU |
| Single-stream | No | Yes — concatenated img+txt with fused linear1/linear2 |
| Double-stream guidance | adaLN-Zero per block (6 params) | Global mod indexed by block (6 params each) |

**Double-stream block flow:**
1. Global modulation: `mod_dbl_img[bi]` → 6×[shift, scale, gate] for img; `mod_dbl_txt[bi]` → same for txt
2. adaLN on img and txt separately
3. Q/K/V projection (3072→3072) + per-head QK RMSNorm
4. 4-axis RoPE: axes [32,32,32,32], θ=2000 — axis 1=row, axis 2=col for img; axis 3=seq for txt; axis 0=identity
5. Joint attention: concat [txt, img] → attend → split
6. Gated residual for both streams
7. SwiGLU FFN (`gate_up` + `mlp_dn`) with gated residual

**Single-stream block flow:**
1. Concatenate txt+img → joint tokens `[N_txt + N_img, H]`
2. Global single-stream modulation → 3×[shift, scale, gate]
3. LayerNorm (no affine)
4. `linear1`: `[3H + 2*n_ff, H]` → Q, K, V (H each) + gate, up (n_ff each)
5. Per-head QK RMSNorm + 4-axis RoPE (same axes, txt and img positions interleaved)
6. Attention over all joint tokens
7. Parallel SwiGLU MLP: `silu(gate) * up`
8. `linear2`: `[H, H + n_ff]` column split → `l2_attn [H,H]`, `l2_mlp [H, n_ff]`; output = `l2_attn(attn_out) + l2_mlp(mlp_out)`
9. Gated residual; discard txt portion, keep only img tokens

**Output:**
- `norm_out`: LayerNorm + adaLN (shift+scale from `Linear(H → 2H)`)
- `proj_out`: Linear(H → 128) → unpatchify → velocity `[32, H/8, W/8]`

### VAE — 2D Standard Decoder (SDXL-family)

| Parameter | Value |
|---|---|
| Latent channels | 32 |
| post_quant_conv | Conv2d(32→32, 1×1) |
| conv_in | Conv2d(32→512, 3×3) |
| mid_block channels | 512 |
| up_block channels | [512, 512, 256, 128, 128] (from inner to outer) |
| Final norm | GroupNorm(32, 128) |
| conv_out | Conv2d(128→3, 3×3) |
| Normalization | **GroupNorm32** (NOT RMS norm — different from Qwen-Image) |
| Activation | SiLU |
| Weight dtype | F32 (in fp8 safetensors; decoded during load) |

The VAE uses 2D convolutions throughout (no temporal dimension). Spatial upsampling via nearest-neighbor 2× + Conv2d 3×3 (same as SDXL).

### Scheduler — FlowMatch Euler (shared with Qwen-Image)

| Parameter | Value |
|---|---|
| Scheduler | `qwen_image_scheduler.h` (reused as-is) |
| Shift mode | `use_dynamic_shifting=False` |
| Base shift | 0.5 (or 1.0 for base model) |
| Distilled steps | 4 steps recommended |
| Base steps | 20–50 steps |
| CFG | Not needed for distilled (guidance=3.5 baked in) |

Reuses `qimg_sched_set_timesteps_comfyui()` from `qwen_image_scheduler.h` with `shift=1.0`.

## Weight Files

All weights stored under `/mnt/disk01/models/klein2-4b/`:

| Component | File | Format | Size |
|---|---|---|---|
| DiT (distilled) | `diffusion_models/flux-2-klein-4b-fp8.safetensors` | FP8 E4M3 | ~2.9 GB |
| DiT (base) | `diffusion_models/flux-2-klein-base-4b-fp8.safetensors` | FP8 E4M3 | ~2.9 GB |
| VAE | `vae/flux2-vae.safetensors` | F32 | ~335 MB |
| Text Encoder | `text_encoder/` | 2-shard safetensors (BF16) | ~4.8 GB |

**Tokenizer:** `/mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf` — GGUF file used for BPE tokenizer only (weights not used).

## Source Files

### Core Headers (`common/`)

| File | Description |
|---|---|
| `flux2_klein_dit.h` | DiT forward: patchify, double+single stream blocks, 4-axis RoPE, FP8 dequant |
| `flux2_klein_vae.h` | VAE decoder: GroupNorm32, Conv2d, ResBlock, upsampling |
| `flux2_klein_text_encoder.h` | Text encoder: safetensors via `transformer.h` (CPU) or `cuda_llm_runner.h` (GPU) |
| `qwen_image_scheduler.h` | FlowMatch scheduler (reused from Qwen-Image) |
| `safetensors.h` | SafeTensors file parser (mmap) |
| `gguf_loader.h` | GGUF v3 file loader |
| `transformer.h` | CPU transformer inference (Qwen3 arch supported) |

All are single-header libraries — define `*_IMPLEMENTATION` before include.

### CPU Implementation (`cpu/flux2/`)

| File | Description |
|---|---|
| `test_flux2.c` | CPU test harness: scheduler, VAE, DiT, text encoder, full pipeline |
| `Makefile` | Build targets for each test mode |

### CUDA Implementation (`cuda/flux2/`)

| File | Description |
|---|---|
| `cuda_flux2_runner.h` | GPU runner: NVRTC kernels, F32 GEMM, FA2 flash attention, 4-axis RoPE, DiT forward, VAE decode |
| `test_cuda_flux2.c` | CUDA test harness: init, load, DiT step, text encoder, VAE, full generation |
| `Makefile` | Build targets |

### HIP/ROCm Implementation (`rdna4/flux2/`)

Targets RDNA4 (gfx1200/gfx1201). Kernels compiled at runtime via HIPRTC (no `hipcc` required).

| File | Description |
|---|---|
| `hip_flux2_runner.c/.h` | GPU runner: HIPRTC kernels, FP8 LUT GEMM + F32 fallback, 4-axis RoPE, DiT forward, GPU VAE decode |
| `hip_flux2_kernels.h` | HIP kernel strings: `gemm_f32_f32`, `gemm_fp8_scaled_f32`, flash attention, RoPE, VAE conv, etc. |
| `test_hip_flux2.c` | Test harness: `--test-dit`, `--generate` (scheduler + VAE + PPM) with CPU text encoder or `--txt-bin` |
| `verify_dit.c` | CPU↔GPU single-step DiT numerical verification (max_diff vs CPU reference) |
| `verify_vae.c` | CPU↔GPU VAE decode verification |
| `Makefile` | Build targets: `make test_hip_flux2 verify_dit verify_vae`, plus `test-gen`/`test-verify-dit` shortcuts |

**FP8 LUT GEMM** (default ON, set `FLUX2_FP8_GEMM=0` to force F32): raw FP8 E4M3 bytes are uploaded to VRAM (~4× smaller than F32), dequantized per-element on load via a 256-entry constant-memory LUT. Per-tensor `weight_scale` is passed to the kernel as a scalar argument. Verified on RTX 9070 XT (gfx1201) — `verify_dit --lat 8` reports `max_diff≈3e-4 corr=1.0` vs CPU reference in both FP8 and F32 modes.

**End-to-end `--generate`**: full denoising pipeline on GPU. Text encoding runs on CPU via `flux2_klein_text_encoder.h` when `--enc`/`--tok` are supplied, or a precomputed `[n_tok, 7680]` F32 tensor can be supplied via `--txt-bin`. If neither is given, the pipeline falls back to a seeded random text hidden tensor (shakeout only — output is not a real image). Text tokens are front-padded to 512 per ComfyUI convention; scheduler uses `qimg_sched_set_timesteps_comfyui` with `shift=expf(2.02)`, `multiplier=1.0`.

## Build & Run

### Prerequisites

```bash
# Weights must be present at:
ls /mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors
ls /mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors
ls /mnt/disk01/models/klein2-4b/text_encoder/
ls /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf
```

### CPU Build & Run

```bash
cd cpu/flux2
make test_flux2

# Full pipeline — distilled model (4 steps, 256×256, 32 threads)
./test_flux2 --generate \
    --dit /mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors \
    --vae /mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors \
    --enc /mnt/disk01/models/klein2-4b/text_encoder \
    --tok /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf \
    --height 256 --width 256 --steps 4 --seed 42 --threads 32 \
    --prompt "a red apple on a white table"

# Full pipeline — base model (20 steps)
./test_flux2 --generate --base \
    --dit /mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-base-4b-fp8.safetensors \
    --vae /mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors \
    --enc /mnt/disk01/models/klein2-4b/text_encoder \
    --tok /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf \
    --height 256 --width 256 --steps 20 --seed 42 --threads 32 \
    --prompt "a red apple on a white table"

# Component tests
make test-sched       # Verify sigma schedule
make test-vae         # VAE decode (64×64 random latent)
make test-dit         # Single DiT step (64×64)
make test-enc         # Text encoder hidden states

# Using Makefile shortcuts (uses paths from Makefile vars)
make test-gen-dist    # Distilled pipeline, 256×256, 4 steps
make test-gen-base    # Base pipeline, 256×256, 20 steps
```

**CPU performance (256×256, 4 steps, 32 threads, ~15 text tokens):**

| Stage | Time |
|---|---|
| Text encoder | ~5s (CPU, transformer.h, GGUF Q8_0) |
| DiT per step | ~114s (OpenMP GEMM + MHA) |
| DiT total (4 steps) | ~457s |
| VAE decode | ~1s (OpenMP conv2d) |
| **Total** | **~463s** |

### CUDA Build & Run

```bash
cd cuda/flux2
make test_cuda_flux2

# Full pipeline — distilled model (4 steps, 256×256)
./test_cuda_flux2 --generate --gpu-enc \
    --enc /mnt/disk01/models/klein2-4b/text_encoder \
    --tok /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf \
    --height 256 --width 256 --steps 4 --seed 42 \
    --prompt "a red apple on a white table"

# Base model (20 steps)
./test_cuda_flux2 --generate --base \
    --enc /mnt/disk01/models/klein2-4b/text_encoder \
    --tok /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf \
    --height 256 --width 256 --steps 20 --seed 42 \
    --prompt "a red apple on a white table"

# Component tests
./test_cuda_flux2 --test-init          # CUDA init + kernel compilation
./test_cuda_flux2 --test-load          # Weight loading
./test_cuda_flux2 --test-dit --height 64 --width 64    # Single DiT step
./test_cuda_flux2 --test-vae --height 64 --width 64    # VAE decode
./test_cuda_flux2 --test-text-enc --prompt "a red apple on a white table"

# Using Makefile shortcuts
make test-init
make test-load
make test-dit
make test-vae
make test-gen-dist    # Distilled pipeline
make test-gen-base    # Base pipeline
```

**CUDA performance (RTX 5060 Ti 16GB sm_120):**

| Stage | Time |
|---|---|
| GPU text encode cold start | ~9.6–9.8s |
| GPU text encode cached in-process | ~0.3s |
| Shared DiT+VAE load | ~16.4–16.6s |
| 64×64, 1-step generation after setup | ~0.3s |
| VAE decode (64×64 output) | ~0.1s |

### CUDA perf snapshot (2026-04-19, `/mnt/disk01/models/klein2-4b`)

Measured on RTX 5060 Ti 16GB with real Klein2-4B weights under `/mnt/disk01/models/klein2-4b`.

Commands used:

```bash
cd cuda/flux2

# Distilled (4-step), 4 repeated runs with shared setup
./test_cuda_flux2 --generate --distilled --gpu-enc \
  --dit /mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors \
  --vae /mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors \
  --enc /mnt/disk01/models/klein2-4b/text_encoder \
  --tok /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf \
  --height 256 --width 256 --steps 4 --seed 42 --repeat 4 \
  --prompt "a red apple on a white table"

# Base (20-step), 4 repeated runs with shared setup
./test_cuda_flux2 --generate --base --gpu-enc \
  --dit /mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-base-4b-fp8.safetensors \
  --vae /mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors \
  --enc /mnt/disk01/models/klein2-4b/text_encoder \
  --tok /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf \
  --height 256 --width 256 --steps 20 --seed 42 --repeat 4 \
  --prompt "a red apple on a white table"
```

Shared setup (once per command):

| Mode | Shared text encode (s) | Shared init+load (s) |
|---|---:|---:|
| Distilled 4-step | 18.3 | 51.5 |
| Base 20-step | 10.9 | 18.9 |

Per-run generation timings (setup excluded):

| Mode | Run 1 total (s) | Run 2 total (s) | Run 3 total (s) | Run 4 total (s) | Median total (s) | Denoising (s) | Denoising (s/step) | VAE (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Distilled 4-step @256² | 11.8 | 11.7 | 11.7 | 11.7 | 11.7 | 11.5 | 2.88 | 0.2 |
| Base 20-step @256² | 57.8 | 57.9 | 57.9 | 57.9 | 57.9 | 57.6 | 2.88 | 0.2 |

Logs:
- `cuda/flux2/bench_20260419_flux2_klein_perf/distilled_256_r4.log`
- `cuda/flux2/bench_20260419_flux2_klein_perf/base_256_r4.log`

**DiT step time vs ComfyUI/PyTorch reference (RTX 5060 Ti, n_txt=8, `--no-cpu-ref`):**

| Image size | ours-FP8 (LT) | ours-FP8 (v7) | ours-F16 (v7) | ours-BF16 (v7) | ComfyUI bf16 (PyTorch 2.11+cu128 fallback) |
|---|---|---|---|---|---|
| 256×256 | **0.021 s** | 0.095 s | 0.048 s | 0.048 s | 0.101 s |
| 512×512 | **0.040 s** | 0.054 s | 0.076 s | 0.076 s | 0.172 s |
| 1024×1024 | **0.116 s** | 0.142 s | 0.367 s | 0.368 s | 0.636 s |

Prior MMA-baseline numbers (kept for reference): F16 0.114 / 0.227 / 0.924 s; BF16 0.199 / 0.329 / 1.243 s at 256² / 512² / 1024².

- ours-FP8 (LT) = default fast path (`FLUX2_BF16_ATTN=1 FLUX2_FP8_GEMM=1 FLUX2_FP8_V7=1` with cuBLAS-LT FP8 e4m3 default ON). Dispatch in `op_gemm_scaled_ex` quantizes X→FP8 (per-tensor scale = max/448), uploads `w_scale` to a 4-byte device scratch, runs `cublasLtMatmul` with BF16 output, then `bf16_to_f32_add_bias_f32` expand+bias. Falls through to v7 if LT init fails, runtime-errors, or shape constraints aren't met. **4.8× / 4.3× / 5.5×** vs ComfyUI bf16 at 256/512/1024 — the 256² jump (0.095 → 0.021 s) is large because v7's `n_tok ≥ 256` gate makes it fall back to MMA-tiled at small image sizes; cuBLAS-LT has no such gate.
- ours-FP8 (v7) = LT path opted out via `FLUX2_CUBLASLT_FP8=0` (cp.async + 4×4 panel-swizzle + ldmatrix). Was the prior fast path before LT integration.
- ours-F16 = `FLUX2_F16_GEMM=1`, ours-BF16 = `FLUX2_BF16_GEMM=1`. Both now dispatch through the **v7 GEMM** ported from `cuda/gemm/cuda_gemm_ptx_kernels.h` (cp.async double-buffer + 4×4 CTA panel swizzle + ldmatrix.x4 + XOR-swizzled SMEM, 64×128×16 tile, 256 threads, 24 KiB dynamic SMEM). The dispatcher quantizes F32 activations to BF16/F16 once per GEMM call (`quant_bf16` / `quant_f16` kernels via `cvt.rn.f16.f32` PTX), runs `gemm_{bf16,f16}_v7`, then `add_bias_inplace_f32`. Source weights are FP8 safetensors dequantized to BF16/F16 with per-tensor `weight_scale` baked in (so the v7 kernel sees pre-scaled W and skips descale). Both modes now beat ComfyUI bf16 at every size.
- 512² ours-FP8 < 256² because FP8 v7's 4×4 CTA panel swizzle requires `n_tok ≥ 256` to fire; at 256² (n_img=64) it falls back to MMA-tiled. The new BF16/F16 v7 path always pads grid to 4× so panel swizzle fires for all sizes (kernel's bounds checks safely drop OOB tiles).
- BF16 ≈ F16 wall time: the kernel cost is dominated by SMEM/MMA throughput, which is identical between the two dtypes; the only difference is the activation quant pass, which is bandwidth-bound and ~1% of total.

> **Note:** The heavy VAE path is on CUDA. The current implementation still bridges the single VAE mid-block attention through CPU, but conv/groupnorm/resblock/upsample work is on GPU.

> **Memory note:** `--generate --gpu-enc` now pre-encodes text once, frees the GPU text encoder, and then loads DiT/VAE. This avoids VRAM exhaustion from keeping both the CUDA LLM runner and Flux2 weights resident at the same time on 16 GB cards.

## Implementation Status

### Verification Results

| Component | Status | Notes |
|---|---|---|
| **Scheduler** | Verified | sigma schedule matches CPU reference |
| **Text encoder (CPU)** | Verified | hidden states match transformer.h reference |
| **Text encoder (GPU)** | Working | CUDA LLM runner, Klein safetensors weights, layer snapshots 8/17/26 |
| **VAE decoder** | Verified | GPU vs CPU `max_diff≈1e-5`; vs diffusers `corr=0.999999` `max_diff≈1e-3` on 64×64; pixel-identical at 256×256 (max pixel diff=1, PSNR=51dB) |
| **Full `--generate --gpu-enc` path** | Working | Prompt is encoded once, encoder is freed, then DiT/VAE load |
| **DiT CPU** | Visual PASS | 64×64 correct red apple (4 steps) |
| **DiT CUDA** | Visual PASS | 256×256 clear red apple (4 steps), max_diff vs CPU < 1e-5 |
| **DiT HIP (RX 9070 XT, gfx1201)** | Numerical PASS | `verify_dit --lat 8`: max_diff 3e-4, corr=1.0 vs CPU (both FP8-LUT and F32) |
| **HIP FP8 LUT GEMM** | Verified | Weight upload ~2.4 s (vs 4.0 s F32), no correctness regression |
| **HIP FP8 128x128 tiled LUT** | Verified | 0.64 s/step at 256×256 (vs 1.87 s/step scalar), corr unchanged |
| **HIP BF16xFP8 WMMA (gfx12 matrix cores)** | Verified | 0.51 s/step at 256×256; 128x128 tile, 8 waves/CTA; corr=0.9991, PSNR 38 dB vs ref |
| **HIP VAE decode** | Numerical PASS | Post `resnets.2` fix: vs diffusers F32 ref cos=0.9999977, PNG PSNR 52.9 dB (near bit-exact) |
| **HIP scheduler** | Verified | Ported FlowMatchEuler exponential dynamic shift; sigmas match diffusers exactly |
| **Full pipeline (CPU)** | Working | 256×256 "a red apple" correct |
| **Full pipeline (CUDA)** | Working | 256×256 "a red apple" correct |
| **Full pipeline (HIP)** | Working | 256×256 end-to-end runs clean (DiT 1.86 s/step, VAE 0.32 s); visual parity pending real text-encoder weights on the test box |

Current CUDA scope: the full DiT path now runs on GPU across both the 5 double-stream blocks and the 20 single-stream blocks, and the VAE decode is fully GPU-accelerated including the mid-block attention. For A/B checks, the old double-stream CPU fallback can still be forced with `FLUX2_CPU_DBL_ATTN=1`. The old CPU-bridge VAE attention path is still available in the source (`flux2_vae_mid_attn_bridge`).

The recent `CUDA_ERROR_ILLEGAL_ADDRESS (700)` seen in `--generate --gpu-enc` on RTX 5060 Ti was not a DiT math bug. It was VRAM exhaustion during DiT/VAE load while the GPU text encoder session was still resident. Allocation checks in `cuda_flux2_runner.h` now fail explicitly, and the generation path frees the GPU text encoder before loading DiT/VAE.

### Critical Bugs Fixed

| Bug | Symptom | Fix |
|---|---|---|
| **RoPE axis mapping** | Wrong output, max_diff ≈ 0.41 vs CPU | Axis 1=row (dims 32-63), axis 2=col (dims 64-95), not axis 2=row, 3=col |
| **linear2 column split byte offset** | All-black output | Row-by-row column extraction during upload (stride is H+n_ff, not H) |
| **Redundant fused lin1 GEMM** | NaN/Inf at 256×256 | Buffer overflow: removed redundant fused call, keep only split Q/K/V/gate/up GEMMs |
| **MMA GEMM on sm_120** | Near-zero velocity | Switch to `gemm_f32_f32` (pure F32 tiled), skips `wmma` |
| **F16 GEMM weight offset** | All-zero with F16 GEMM | QKV/linear1 weight pointer arithmetic used `*4` (F32 bytes) instead of `*WE` (2 bytes for F16) |
| **VAE BN denormalization** | Over-saturated image with speckle artifacts | BN stats (`bn.running_mean/var`) are training artifacts — should NOT be applied during inference. Removed the `latent * sqrt(var) + mean` pre-processing |
| **VAE missing 3rd resnet** | Low correlation (0.74) vs diffusers VAE | Diffusers decoder uses `layers_per_block+1=3` resnets per up_block; our code only had 2. Added `up_res[4][3]` and loading/running of `resnets.2` |
| **HIP VAE missing 3rd resnet** | ~1.19× over-saturated HIP image (PSNR 25.7 dB) | Same bug re-surfaced on HIP side — loader had `up_res[4][3]` but `hip_flux2_vae_decode` only looped over `up_res[bi][0..1]`. Added `resnets.2` call in HIP up_block loop; PSNR jumped to 52.9 dB |
| **Base model FP8/BF16 mixed dtype** | Grid pattern output with base model | Base model `double_blocks.0.*.proj.weight` is BF16 (not FP8) but loader always called `flux2_mat_fp8`. Added `flux2_mat_auto` that detects dtype and dispatches to the correct loader |
| **Base model scheduler** | sigma=1000 (double 1000× application) | Base scheduler used `qimg_sched_set_timesteps` which multiplied by `num_train_timesteps=1000`, but DiT also does `timestep*1000`. Switched to `qimg_sched_set_timesteps_comfyui` with `multiplier=1.0` |
| **Wrong sigma shift formula** | Spotty pixel artifacts, wrong sigma values | Used AuraFlow shift `alpha*t/(1+(alpha-1)*t)` with alpha=2.02. ComfyUI uses Flux shift `exp(mu)*t/(...)` with mu=2.02. Fix: pass `expf(2.02)` ≈ 7.539 as alpha to our scheduler |
| **Text not padded to 512** | Velocity corr=0.92 vs ComfyUI; pixel-level quantization-like artifacts | ComfyUI's `Flux2.extra_conds` front-pads text to 512 tokens with zeros (`comfy/model_base.py:960`). We were using ~15 real tokens. Fix: front-pad text to 512 tokens before passing to DiT. After fix: corr=0.997, std_ratio=0.997 |
| **Double-stream attention path** | Needed verification on sm_120 after earlier divergence during bring-up | Re-verified on RTX 5060 Ti: GPU flash attention now matches CPU (`max_diff≈6e-6` on `--test-dit` 64×64); CPU fallback kept behind `FLUX2_CPU_DBL_ATTN=1` |
| **512-token text padding** | 35× GEMM overhead | Remove padding; return only real tokens (~15 tokens typical) |
| **Chat template** | Wrong hidden states | Strip system/chat prefix; use clean user message tokens only |
| **GPU generate VRAM conflict** | `CUDA_ERROR_ILLEGAL_ADDRESS (700)` or later bad pointers in `--generate --gpu-enc` | Make CUDA allocations fail explicitly and free GPU text encoder before DiT/VAE load |

### Key Architecture Gotchas

1. **Global modulation** — unlike Qwen-Image where each block has its own modulation weights, Flux.2 Klein has one shared modulation weight for all double blocks and one for all single blocks. Each block indexes its own slice of the expanded modulation output.

2. **linear2 column split** — the single-stream `linear2` weight is `[H, H+n_ff]`. The GPU splits this into `l2_attn [H, H]` and `l2_mlp [H, n_ff]` **at upload time** by extracting columns row-by-row (cannot byte-split since each row spans H+n_ff elements).

3. **4-axis RoPE** — axes_dim=[32,32,32,32], θ=2000. Image tokens use axes 1 (row position) and 2 (col position). Text tokens use axis 3 (sequence position). Axis 0 is identity for all tokens.

4. **FP8 with per-tensor scale** — weights are stored as `uint8_t` FP8 E4M3, with a companion `weight_scale` scalar per tensor. Dequant: `f32 = fp8_to_f32(byte) × scale`.

5. **txt_dim=7680** — text hidden states are 7680-dimensional (Qwen3-4B output). This projects down to hidden_dim=3072 via `context_embedder`.

## TODO / Limitations

### Performance

- [x] **All-GPU VAE attention**: VAE mid-block single-head attention now runs on GPU via a dedicated `vae_attn_f32` flash-attention kernel (FA2 style, BKV=8, EPT=16). GPU vs CPU max_diff≈3e-6 at 64×64, ≈4e-4 at 512×512.
- [x] **F16 GEMM (v7)**: `FLUX2_F16_GEMM=1` dispatches to `gemm_f16_v7` (cp.async + ldmatrix + 4×4 CTA panel swizzle, ported from `cuda/gemm/cuda_gemm_ptx_kernels.h`). Activation is quantized F32→F16 once per GEMM via the `quant_f16` kernel (`cvt.rn.f16.f32` PTX); bias added in-place by `add_bias_inplace_f32`. Source weights are FP8 safetensors dequantized to F16 with per-tensor `weight_scale` baked in. **0.367 s/step at 1024²** (was 0.924 with the old MMA-baseline kernel; 1.7× faster than ComfyUI bf16). GPU vs CPU FP32 ref unchanged (`max_diff≈6e-4` at 256²).
- [x] **BF16 GEMM (v7)**: `FLUX2_BF16_GEMM=1` dispatches to `gemm_bf16_v7` (same v7 infra). **0.368 s/step at 1024²** (was 1.243; 1.7× faster than ComfyUI bf16). GPU output min/max unchanged vs the prior MMA baseline (correctness preserved bit-for-bit on the test inputs).
- [ ] **Attention kernel scaling**: Single-block-per-head FA2 flash attention works but may not scale to large resolutions. No multi-block flash attention.
- [ ] **GPU text encoder startup cost**: cold start is still dominated by CUDA LLM init + weight upload. PTX cache and in-process reuse help, but `--generate` currently pre-encodes once and frees the runner rather than keeping text and DiT resident together.

### Quality

- [ ] **ComfyUI numerical verification**: Per-block .npy dump infrastructure added (`--dump-block N` in CPU test, dump callback in `flux2_klein_dit.h`). Comparison script at `ref/flux2_klein/compare_blocks.py`. ComfyUI trace script (`trace_blocks.py`) is a skeleton — needs completion with actual model sub-module calls once ComfyUI model structure is inspected.
- [x] **Base model**: Base model (20-step) now works with CFG=3.5. Fixed mixed BF16/FP8 dtype loading and scheduler timestep convention. Visual output confirmed.

### Features

- [x] **512×512**: VAE decode verified at 512×512 (GPU vs CPU max_diff≈4e-4), DiT step runs at 512×512 (1.02s for 256 img tokens on RTX 5060 Ti). Larger resolutions untested.
- [ ] **Single-session GPU pipeline**: text encoding and DiT both run on GPU, but generation currently frees the text encoder before DiT/VAE load to stay within VRAM budget on 16 GB cards.
