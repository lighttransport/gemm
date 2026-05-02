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

All weights stored under `/path/to/qwen-image-st/` (ComfyUI FP8 format):

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

**Alternative GGUF weights** (at `/path/to/qwen-image/`):

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
| `hip_qimg_kernels.h` | HIP kernel strings: `gemm_fp8w_f32` (16×16 scalar LUT), `gemm_opt_fp8` (128×128 tiled LUT with fused BF16 trunc), `gemm_fp8w_bf16a_wmma_t` (BF16×FP8 matrix-core WMMA, 256 threads / 128×128 CTA), `gemm_fp8_fp8w_perrow_pgr2` + `qimg_quantize_act_perrow_fp8` (FP8×FP8 matrix-core WMMA, 128 threads / 128×128 CTA, 3-deep LDS pipeline, per-row activation scale), `flash_attn_sa_wmma_f32` (BF16 WMMA flash-attention, head_dim=128, Br_cta=64) |
| `test_hip_qimg.c` | Test harness: `--test-init / --test-dit / --test-vae / --test-enc / --generate`. Supports pinned-input options (`--init-bin`, `--txt-bin`, `--sigmas-bin`, `--dump-final`, `--skip-unstd`) plus PNG output via `stb_image_write.h`. |
| `verify_qimg_dit.c` | CPU↔HIP single-step DiT comparison (max_diff / mean_diff / corr) |
| `Makefile` | Builds `test_hip_qimg` and `verify_qimg_dit` |

**FP8 LUT GEMM** (default ON when the FP8 LUT compiles): raw FP8 E4M3 bytes are uploaded to VRAM (~4× smaller than F32) and dequantized via a 256-entry constant-memory LUT. Qwen-Image FP8 weights have **no per-tensor scale** (unlike Flux.2 Klein), so the kernels do not take a `w_scale` argument.

**BF16×FP8 WMMA** (`QIMG_FP8_WMMA=1`): same 128×128 CTA tile layout as the Flux.2 Klein WMMA path — 256 threads = 8 waves arranged 2×4, each wave computes a 64×32 sub-tile via 4×2 = 8 `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` instructions reusing 8 KB of shared memory per K=16 slab. Activations F32→BF16 truncated at SMEM load; FP8 weights LUT-dequanted then BF16-truncated; F32 accumulator. Verified on RX 9070 XT via `verify_qimg_dit --lat 8`: corr ≈ 0.9765 vs CPU reference (matches the LUT path's 0.9777 floor — the BF16-trunc-on-output noise dominates either way).

| Mode | 256×256 / 20-step DiT | 512×512 / 20-step DiT | VAE 256 | VAE 512 |
|---|---|---|---|---|
| HIP 128×128 tiled FP8 LUT (default) | **36.1 s (1.80 s/step)** | **91.2 s (4.56 s/step)** | 1.5 s | 20.5 s |
| HIP 128×128 BF16×FP8 WMMA (`QIMG_FP8_WMMA=1`) | **29.4 s (1.47 s/step)** | **81.8 s (4.09 s/step)** | 1.5 s | 20.2 s |
| + 512 MB activation reserve (45/60 preload) | **24.0 s (1.20 s/step)** | **74.5 s (3.73 s/step)** | 1.3 s | 11.2 s |
| HIP 128×128 **FP8×FP8 WMMA** (`QIMG_FP8_FP8_WMMA=1`) | **22.0 s (1.10 s/step)** | **52.2 s (2.61 s/step)** | 1.5 s | 19.4 s |
| + **BF16 WMMA flash-attention** (`QIMG_BF16_ATTN=1`, default ON) | **20.3 s (1.02 s/step)** | **29.2 s (1.46 s/step)** | 1.5 s | 19.6 s |
| + **BF16 WMMA VAE conv2d + GPU spatial attention** (`QIMG_VAE_WMMA=1`, default ON) | **20.3 s (1.01 s/step)** | **29.1 s (1.45 s/step)** | **0.5 s** | **1.3 s** |
| + **Q-blocked VAE attention (8 queries/CTA, BKV=16)** | — | **29.1 s (1.45 s/step)** | **0.5 s** | **0.9 s** |

**Default-on (no opt-in flags) vs FP8×FP8 opt-in**, current measurements:

| Mode | 256² / 20-step DiT | 512² / 20-step DiT | 1024² / 20-step DiT |
|---|---|---|---|
| HIP BF16×FP8 WMMA + BF16 attn + VAE WMMA (default) | 24.1 s (1.20 s/step) | 55.7 s (2.79 s/step) | 214.5 s (10.72 s/step) |
| + FP8×FP8 (`QIMG_FP8_FP8_WMMA=1`, opt-in, **quality drop**) | 20.3 s (1.02 s/step) | 29.3 s (1.46 s/step) | **89.0 s (4.45 s/step)** |
| Speedup | 1.18× | 1.91× | **2.41×** |

**1024×1024 / 20-step legend** (after Q-block VAE attention fix):

| Stage | Wall | Notes |
|---|---:|---|
| Text encoder (Qwen2.5-VL Q4, GPU) | 10.5 s | one-shot, 46 tokens |
| DiT load + 45/60 block preload | 2.6 s | 1.1 GB free after preload |
| Denoise (60 blocks × 20 steps, n_img=4096+128) | 261 s (13.07 s/step) | FP8×FP8 GEMM + BF16 WMMA attn |
| VAE conv2d (BF16 WMMA) | ~2.2 s | downsample/up chain on GPU |
| VAE middle self-attention (Q-block) | 3.6 s | spatial=16384, c=384 (was 25.5 s pre Q-block) |
| **Total** | **~278 s** | output `apple_1024.ppm` |

At 1024² the VAE middle self-attention initially regressed to 25.5 s on GPU because the original `vae_self_attn_f32` was sized for spatial=4096 — one CTA per query, no Q-blocking, every block re-streamed the full K/V from HBM (~16× redundant load at 16k tokens). Added `vae_self_attn_qb_f32` with QB=8 queries per CTA and BKV bumped 8→16, sharing one 48 KB LDS K/V tile across 8 waves. Drops 1024² VAE attn 25.5 → 3.6 s (7.1×); 512² also benefits (~250 → 57 ms). Auto-dispatched when c%32==0; original kernel kept as fallback.

**FP8×FP8 WMMA** (`QIMG_FP8_FP8_WMMA=1`, **opt-in only**): port of `gemm_fp8_pipe32_pgr2` from `rdna4/fp8/bench_fp8_gemm.c` (~218 TF/s on mm0) into the qimg HIPRTC kernel string. 128 threads = 4 waves, 128×128 CTA tile, MIWT 4×4, K_step=32, 3-deep LDS pipeline (PGR=2). Activations are F32→FP8 quantized per output row (`qimg_quantize_act_perrow_fp8`: scale `s_m = max(|X[m,:]|)/448`), GEMM uses `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`, and writeback fuses the row scale + bias. Mirrors the CUDA sibling's `gemm_fp8_pipe_perrow_f32` pattern. Eligible when `n_tok % 128 == 0`, `n_out % 128 == 0`, and `n_in % 32 == 0`; falls back to BF16×FP8 WMMA otherwise. `QIMG_FP8_WMMA_BF16=1` forces the BF16 path even when FP8×FP8 would qualify.

Speedup grows with spatial: BF16×FP8 → FP8×FP8 = **1.18× at 256² (1.20 → 1.02 s/step)**, **1.91× at 512² (2.79 → 1.46 s/step)**, **2.41× at 1024² (10.72 → 4.45 s/step)**. The 1024² ratio approaches the WMMA hardware peak ratio (~2.0×); residuals come from non-GEMM ops (FA, layernorm, modulation) that don't scale with the FP8 win.

Quality regression (256²/20 steps vs diffusers `final_latent_packed_256.bin`): latent cos drops 0.999971 → 0.956359, latent PSNR 51.70 → 21.32 dB; PNG PSNR vs `apple_ref_256.png` drops 46.63 → 24.60 dB (mean pixel diff 0.82 → 10.65 / 255). The per-row activation scale `max|x|/448` doesn't preserve enough dynamic range for Qwen-Image's MMDiT path — outliers in attention and FFN inputs get crushed to e4m3 mantissa noise. Default kept OFF; ship the env var for users who accept the quality drop in exchange for ~2× denoise at ≥512².

**BF16 WMMA flash-attention** (`QIMG_BF16_ATTN=1`, default ON when kernel loaded): replaces the F32 scalar `flash_attn_f32` with `flash_attn_sa_wmma_f32` ported from `rdna4/trellis2/hip_trellis2_kernels.h`. FA2 online softmax with QK^T and PV computed via 16×16×16 BF16 WMMA tiles (one `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` per intrinsic). 128 threads = 4 waves, each wave owns 16 query rows, Br_cta=64, Bc=16. Q/K/V are F32→BF16 cast at LDS staging, F32 accumulators, F32 output. Constraints: head_dim=128 (Qwen-Image), partial last KV tile masked with -inf so any N is supported (no padding required). Dispatching this kernel collapsed the scalar-attention bottleneck that dominated post-FP8×FP8 step time: at 512², 2.61 → 1.46 s/step = **1.79× speedup on denoise** (the residual ~1.15 s/step that wasn't GEMM was almost entirely the F32 scalar FA). Combined with FP8×FP8 WMMA, end-to-end at 512×512 is **2.93× faster than the BF16×FP8-only baseline** (4.28 → 1.46 s/step).

**VAE BF16 WMMA conv2d + GPU spatial attention** (`QIMG_VAE_WMMA=1`, default ON): three new kernels in `hip_qimg_kernels.h`:
- `vae_conv2d_3x3_wmma_f32` — implicit-GEMM 3×3 conv via 16×16×16 BF16 WMMA tiles. Treats conv as GEMM with M=Co, N=H·W, K=Ci·9. Each CTA = 1 wave producing a 16×16 output tile; lanes own 8 K elements per WMMA. Replicate or zero pad. Constraint: Co%16==0, Ci%16==0, H·W%16==0 (every Qwen-Image VAE conv satisfies this).
- `vae_conv2d_1x1_wmma_f32` — same layout for 1×1 (K = Ci instead of Ci·9). Used by `post_quant_conv` and the middle-block attention QKV / proj convs.
- `vae_self_attn_f32` — GPU FA2 spatial self-attention for the middle-block attention (single head, head_dim=384). One CTA per query row, 1 wave, lane owns ept=12 channels. KV tiled BKV=8 rows into LDS each iteration. Replaces the previous CPU-side O(spatial²·c) loop + 2× D2H/H2D copies — the dominant bottleneck of the entire pipeline.

The VAE attention port is the headline win: at 512² spatial=4096, the CPU loop ran for **17.9 s** (96% of VAE wall time). The GPU FA2 kernel does the same compute in **0.44 s** — a 41× speedup on that op alone. End-to-end VAE decode dropped 19.4 s → 1.3 s at 512² (**15× speedup**), 1.5 s → 0.5 s at 256² (**3× speedup**).

**Activation reserve tuning**: the preload planner used to reserve 2 GB of VRAM for per-step activations, but actual peak usage at 512×512 is ~300 MB (Q/K/V 72 MB, scratch3 48 MB, scratch1/2 18 MB each, d_q/d_k/d_v/d_attn 264 MB total, plus the img/txt buffers). Lowering the reserve to 512 MB frees up room for 5 more preloaded blocks (45/60 instead of 40/60) on a 16 GB card, eliminating ~5 × ~85 ms = ~425 ms/step of PCIe streaming. This is bit-for-bit equivalent to the old path (no math changes — only which blocks live in VRAM vs stream on-demand). For higher resolutions where activation scratch grows past 512 MB, override with `QIMG_WORKSPACE_MB=<n>` (e.g. `QIMG_WORKSPACE_MB=2048` at 1024×1024+).

**Note on the prefetch / copy-stream approach**: the natural next step — running block streaming on a separate `hipStream` concurrently with compute on the default stream (pinned host staging + `hipMemcpyAsync` + slot ring buffer) — was implemented and tested but **regressed ~0.7–1.0 s/step at 512×512** on gfx1201. Tracing showed that (1) sync `hipMemcpy` on the default stream already achieves good implicit overlap with queued compute (the "85 ms/block" really breaks down as ~60 ms compute catchup + ~25 ms DMA), and (2) explicit DMA on a copy stream contends with compute for HBM bandwidth. The current WMMA GEMM is memory-bandwidth-bound, so running DMA in parallel slows compute by more than the DMA saves. The simpler residency win above beats the prefetch approach on this GPU.

**Apples-to-apples vs diffusers reference** (256×256 / 20 steps / cfg=1, both runners loading the same `qwen_image_fp8_e4m3fn.safetensors` and using `init_latent_256.bin / apple_text_256.bin / sigmas_256.bin` from `dump_diffusers_pipeline.py`). Default-on HIP today is **BF16×FP8 WMMA + BF16 WMMA attn + VAE WMMA**; FP8×FP8 is opt-in and trades quality for speed:

| | diffusers (FP8, seq. cpu offload) | HIP BF16×FP8 WMMA + BF16 attn + VAE WMMA (default) | HIP FP8×FP8 + BF16 attn + VAE WMMA (`QIMG_FP8_FP8_WMMA=1`) |
|---|---|---|---|
| Wall time @ 256² | 102 s | **~24 s** (denoise 24.1 s + VAE 0.5 s) | **~21 s** (denoise 20.3 s + VAE 0.5 s) |
| Wall time @ 512² | (unmeasured; ≈400+ s on rocm-7.2 BF16 offload) | **~57 s** (denoise 55.7 s + VAE 1.3 s) | **~30 s** (denoise 29.3 s + VAE 1.3 s) |
| Wall time @ 1024² | n/a | ~218 s (denoise 214.5 s + VAE ~3 s) | **~94 s** (denoise 89.0 s + VAE ~3 s) |
| Final latent cosine vs ref @ 256² | 1.0 (self) | 0.999971 | 0.956359 |
| Final latent PSNR vs ref @ 256² | ∞ | 51.70 dB | 21.32 dB |
| PNG PSNR vs ref @ 256² | ∞ | 46.63 dB | 24.60 dB |

The HIP runner is now **~4× faster** than the pytorch-rocm diffusers reference at 256² (102 s → 24 s) at full BF16×FP8 quality, and **~5× faster** (102 s → 21 s) when opting into FP8×FP8 with the documented quality trade-off. The BF16-WMMA-only baseline was already 3× faster; BF16 WMMA flash-attention (replacing F32 scalar FA) and BF16 WMMA VAE conv2d + GPU spatial-attention port compress the rest. FP8×FP8 stacks on top for users who can absorb the quality drop. PyTorch sequential cpu offload is forced because the BF16 transformer doesn't fit in 16 GB VRAM resident — even with the FP8 transformer loaded via `from_single_file`, the text encoder + VAE + activations push past 16 GB during inference. A live 512² PyTorch comparison wasn't run (no diffusers in the available rocm env, and the offload pattern at 512² historically scales ~4× = 6+ minutes / 20 steps), but the relative gap widens since the BF16-attention/VAE wins matter more at higher spatial counts (4096 vs 1024 tokens for FA, 16× more pixels for VAE).

**Mixed-dtype checkpoint support** (`unsloth/Qwen-Image-2512-FP8`): the December 2025 refresh stores 1093/1933 tensors as BF16 instead of FP8 — biases, norm scales, and 5 of the global GEMM weights (`img_in.weight`, `txt_in.weight`, `time_text_embed.timestep_embedder.linear_1/2.weight`, `norm_out.linear.weight`, `proj_out.weight`). `qimg_upload_weight_auto` detects the dtype per global weight and tags the device pointer; `op_wgemm_bf16_auto` then dispatches the F32 GEMM path (`gemm_f32_f32` + downstream BF16 trunc) for BF16-stored globals while keeping FP8 GEMM for the FP8 ones. Per-block weights are still 100% FP8 in the 2512 file, so they stay on the existing FP8 LUT / WMMA tracks unchanged.

| | diffusers 2512 (FP8 from_single_file) | HIP 2512 |
|---|---|---|
| Wall time at 256×256/20 steps | 186 s | **30 s** |
| Final latent cosine vs ref | 1.0 (self) | **0.9936** |
| PNG PSNR vs ref | ∞ | **30.0 dB** |

PSNR is lower than the original ComfyUI FP8 file (30 dB vs 49 dB) because the 6 BF16-stored global GEMMs run through F32 + downstream BF16 trunc instead of the FP8 LUT/WMMA fused-trunc path. Note: the earlier 30 dB number was measured against a diffusers *reference* generated at **cfg=1** (single forward) with the same 2512 checkpoint, which turned out to be a weak reference — see the next subsection.

**Current 2512 state (2026-04-15) — KNOWN BROKEN at cfg=1.**

Free-running 2512 generations at `true_cfg_scale=1` produce semantically-wrong outputs in **both** diffusers and HIP for most prompts. Side-by-side at 256×256 / 20 steps / seed=42 / cfg=1 (`ref/qwen_image/compare_2512/compare_grid.png`):

| prompt | diffusers 2512 (cfg=1) | HIP 2512 (cfg=1) |
|---|---|---|
| `"a photo of a cat sitting on grass"`   | fox-like creature (not a cat) | woman in a field (not a cat) |
| `"a modern glass office building at sunset"` | abstract coloured grid  | abstract glassy grid |
| `"a snowy mountain landscape with pine trees"` | snowy mountains ✓  | clear mountains ✓ |

Only the mountain prompt reliably grounds. Both runners drift on object/subject prompts, which strongly hints the **text conditioning is the problem**, not the DiT or VAE: mountain-landscape embeddings sit in a denser part of the Qwen2.5-VL latent space and survive even weak guidance, while object prompts need CFG to break out of the "average scene" mode. Hypotheses to rule out (in order):

1. **cfg=1 was inappropriate.** Qwen-Image-2512 was trained with CFG; the released examples (e.g. unsloth tutorial) use `true_cfg_scale=4.0`. The whole cfg=1 regression test needs to be redone at cfg=4. *(IN PROGRESS — HIP runner is single-forward only, so HIP apples-to-apples at cfg=4 requires adding a CFG combine step.)*
2. **2512 text encoder preprocessing differs from the original.** The 2512 refresh may require a different chat template / system prompt / attention-mask layout before the Qwen2.5-VL text hidden states are fed into `txt_in`. `dump_diffusers_pipeline.py` currently calls `pipe.encode_prompt(prompt, …, max_sequence_length=512)` which doesn't thread a prompt template — need to compare the raw `prompt_embeds` tensor from the 2512 pipeline against the original FP8 pipeline for the same prompt to see if they agree up to a shape.
3. **2512's sigma / timestep schedule differs.** The unsloth page advertises "2-step" variants. We've been running the 20-step schedule unchanged from the original checkpoint — if 2512 ships a different `FlowMatchEulerDiscreteScheduler` config (different `shift` / `num_train_timesteps`), running with the old schedule would mis-time every step.
4. **BF16 global GEMMs drift the modulation projections.** The 6 BF16-stored globals include `time_text_embed.timestep_embedder.linear_{1,2}.weight`, which produces the per-block `img_mod / txt_mod` vectors. If our F32 fallback GEMM accumulates differently than diffusers' BF16 matmul, every block's adaLN is off by a few LSBs — which matters most at early timesteps where the signal is smallest.

**TODOs** (in priority order):

- [ ] Re-run `gen_diffusers_reference.py` with `--cfg 4.0` for cat / building / mountain / detailed-retriever prompts; confirm diffusers 2512 at cfg=4 actually produces good images. If yes, (1) is confirmed and the issue is "HIP is single-forward only at the moment".
- [ ] Add `--cfg <scale>` + `--negative <prompt>` to `test_hip_qimg --generate`: runs two DiT forwards per step (positive + negative) and combines via `vel = uncond + cfg * (cond - uncond)`. This roughly doubles DiT wall time but enables apples-to-apples vs diffusers 2512.
- [ ] Dump `prompt_embeds` from both diffusers pipelines (original FP8 vs 2512) for the same prompt+seed and diff them. If they already agree, the text encoder path is fine and only CFG matters; if they disagree, hypothesis (2) is live and we need to match preprocessing.
- [ ] Check `pipe.scheduler.config` on the 2512 pipeline vs original and diff any field that affects sigma generation.
- [ ] Regenerate the `apple_ref_2512.png` reference at cfg=4 and re-measure HIP PSNR against it (once HIP supports CFG).

**What is NOT broken:** the DiT numerics themselves (`verify_qimg_dit` still passes at max_diff < 1e-3 vs the CPU reference on random inputs), the VAE (mountain output is sharp, and the original ComfyUI FP8 file at cfg=1 still reaches 49 dB PSNR), and the per-block FP8 vs BF16 dispatch for mixed-dtype checkpoints (it loads, runs, and decodes without NaNs). Only the free-running 2512 pipeline at cfg=1 is producing weak outputs.

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
    /path/to/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
    /path/to/qwen-image-st/vae/qwen_image_vae.safetensors \
    /path/to/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
    --height 128 --width 128 --steps 20 --prompt "a red apple on a white table"
```

### Verification

```bash
# Step 1: Generate ComfyUI reference data (requires ComfyUI + GPU)
cd ref/qwen_image
python generate_reference.py --all \
    --comfyui-dir /path/to/ComfyUI \
    --model-dir /path/to/qwen-image-st

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

## CUDA Benchmark Snapshot (2026-04-19, RTX 5060 Ti 16GB)

Machine:
- GPU: NVIDIA GeForce RTX 5060 Ti (16 GB)
- Driver: 595.58.03
- CUDA runtime: 13.2 (`nvidia-smi`)
- Date: 2026-04-19 (JST)

Protocol:
- Prompt: `"a red apple on a white table"`
- Seed: `42`
- Steps: `20`
- Kernel mode: `QIMG_FP8_PIPE_PERROW=1 QIMG_BF16_ATTN=1`
- Per size: 1 warm-up run (excluded), then 3 measured runs; table shows median of measured runs.
- Model paths:
  - DiT: `/mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors`
  - VAE: `/mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors`
  - Text encoder GGUF: `/mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf`

Command template:

```bash
QIMG_FP8_PIPE_PERROW=1 QIMG_BF16_ATTN=1 ./test_cuda_qimg --generate \
  --height <H> --width <W> --steps 20 --seed 42 \
  --prompt "a red apple on a white table" \
  --dit /mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors \
  --vae /mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors \
  --enc /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf
```

Median timings (3 measured runs):

| Resolution | Text encoding (s) | Denoising total (s) | Denoising (s/step) | VAE decode (s) | End-to-end total (s) |
|---|---:|---:|---:|---:|---:|
| 256×256 | 14.1 | 37.6 | 1.88 | 0.3 | 51.9 |
| 512×512 | 14.0 | 52.3 | 2.62 | 0.4 | 66.7 |
| 1024×1024 | 13.9 | 130.3 | 6.52 | 1.6 | 145.8 |

Raw run details (same setup; warm-up excluded from medians):

| Resolution | Run | Text encoding (s) | Denoising total (s) | Denoising (s/step) | VAE decode (s) | End-to-end total (s) |
|---|---|---:|---:|---:|---:|---:|
| 256×256 | warmup | 14.0 | 39.1 | 1.95 | 0.3 | 53.4 |
| 256×256 | run1 | 14.1 | 37.6 | 1.88 | 0.3 | 52.0 |
| 256×256 | run2 | 14.1 | 37.6 | 1.88 | 0.2 | 51.9 |
| 256×256 | run3 | 13.9 | 37.6 | 1.88 | 0.3 | 51.8 |
| 512×512 | warmup | 14.0 | 52.4 | 2.62 | 0.4 | 66.8 |
| 512×512 | run1 | 13.8 | 52.3 | 2.62 | 0.4 | 66.5 |
| 512×512 | run2 | 14.1 | 52.4 | 2.62 | 0.4 | 66.9 |
| 512×512 | run3 | 14.0 | 52.3 | 2.61 | 0.4 | 66.7 |
| 1024×1024 | warmup | 14.2 | 130.0 | 6.50 | 1.6 | 145.8 |
| 1024×1024 | run1 | 13.9 | 130.3 | 6.51 | 1.6 | 145.8 |
| 1024×1024 | run2 | 13.9 | 130.3 | 6.52 | 1.6 | 145.8 |
| 1024×1024 | run3 | 13.9 | 130.4 | 6.52 | 1.6 | 145.9 |

Artifacts (`cuda/qimg/bench_20260419_cuda_perf_gpu/`):
- Logs:
  - `perf_256_run{1,2,3}.log`, `perf_512_run{1,2,3}.log`, `perf_1024_run{1,2,3}.log`
  - warm-up logs: `perf_*_warmup.log`
- Representative generated apple images:
  - 256: `cuda_qimg_output_256_run2.ppm`
  - 512: `cuda_qimg_output_512_run2.ppm`
  - 1024: `cuda_qimg_output_1024_run2.ppm`

Note:
- The text-encoder bias safetensors fallback path in this binary points to `/mnt/nvme02/...` and is missing on this machine, so runs print:
  `WARNING: no biases injected (image quality may suffer)`.
  This does not materially affect the throughput figures above.

### CUDA text encoder check (`/mnt/disk01/models/qwen-image`)

Validated CUDA text-encoder execution with the GGUF from:
`/mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf`

Command:

```bash
cd /home/syoyo/work/gemm/diffusion/cuda/llm
./test_cuda_llm /mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf \
  -t "a red apple on a white table" -n 4 -s 256
```

Result:
- Status: **PASS**
- Tokens processed: `4`
- CPU total: `14681.3 ms` (`3670.3 ms/token`)
- GPU total: `368.1 ms` (`92.0 ms/token`)
- Speedup: `39.9x`
