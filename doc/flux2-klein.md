# Flux.2 Klein 4B Text-to-Image Pipeline

End-to-end implementation of [Flux.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) in single-header C libraries with CUDA GPU acceleration. Supports both the distilled (4-step) and base (20-step) variants.

## Pipeline Overview

```
Text Prompt
    │
    ▼
┌─────────────────┐
│  Text Encoder    │  Qwen3-4B (28 layers, GQA)
│  [CUDA or CPU]   │  GGUF Q8_0 weights
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

### Text Encoder — Qwen3-4B

| Parameter | Value |
|---|---|
| Architecture | `qwen3` (GQA transformer) |
| Hidden dim | 2048 |
| Output projected dim | 7680 (output of last layer norm) |
| Layers | 28 |
| Attention heads | 16 (8 KV heads, 2:1 GQA) |
| Head dim | 128 |
| FFN intermediate | 11008 |
| RoPE theta | 1,000,000 |
| Activation | SiLU (SwiGLU) |

Output: per-token hidden states `[N_real_tokens, 7680]` — no padding, no pooling. Only real tokens (not pad tokens) are returned, so N_txt is typically 10–20 for short prompts.

**Chat template:** The model uses a system prompt prefix. Our implementation uses the raw Qwen3 embedding without the VL chat template, relying on a clean text-only system prompt stripped to just the user message tokens.

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
| `cuda_flux2_runner.h` | GPU runner: NVRTC kernels, F32 GEMM, FA2 flash attention, 4-axis RoPE, DiT forward |
| `test_cuda_flux2.c` | CUDA test harness: init, load, DiT step, full generation |
| `Makefile` | Build targets |

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

**CUDA performance (256×256, 4 steps, RTX 5060 Ti 16GB sm_120):**

| Stage | Time |
|---|---|
| Text encoder (GPU) | ~3s (CUDA LLM runner, GGUF Q8_0) |
| DiT load | ~0.5s |
| DiT per step | ~9.5s (F32 GEMM, FA2 flash attention) |
| DiT total (4 steps) | ~38s |
| VAE decode (GPU) | ~0.14s at 8×8 latent test size (64×64 RGB) |
| **Total** | DiT-dominated; old CPU VAE bottleneck removed |

> **Note:** The heavy VAE path is now on CUDA. The current implementation still bridges the single mid-block attention through CPU, but conv/groupnorm/resblock/upsample work is on GPU.

## Implementation Status

### Verification Results

| Component | Status | Notes |
|---|---|---|
| **Scheduler** | Verified | sigma schedule matches CPU reference |
| **Text encoder (CPU)** | Verified | hidden states match transformer.h reference |
| **Text encoder (GPU)** | Working | CUDA LLM runner, GGUF Q8_0 via NVRTC |
| **VAE decoder** | Verified | GPU vs CPU `max_diff≈3e-6` on 64×64 decode; visual output correct |
| **DiT CPU** | Visual PASS | 64×64 correct red apple (4 steps) |
| **DiT CUDA** | Visual PASS | 256×256 clear red apple (4 steps), max_diff vs CPU < 1e-5 |
| **Full pipeline (CPU)** | Working | 256×256 "a red apple" correct |
| **Full pipeline (CUDA)** | Working | 256×256 "a red apple" correct |

Current CUDA scope: the full DiT path now runs on GPU across both the 5 double-stream blocks and the 20 single-stream blocks, and the VAE decode is GPU-accelerated as well. For A/B checks, the old double-stream CPU fallback can still be forced with `FLUX2_CPU_DBL_ATTN=1`. The only remaining CPU bridge in the VAE path is the single mid-block attention calculation.

### Critical Bugs Fixed

| Bug | Symptom | Fix |
|---|---|---|
| **RoPE axis mapping** | Wrong output, max_diff ≈ 0.41 vs CPU | Axis 1=row (dims 32-63), axis 2=col (dims 64-95), not axis 2=row, 3=col |
| **linear2 column split byte offset** | All-black output | Row-by-row column extraction during upload (stride is H+n_ff, not H) |
| **Redundant fused lin1 GEMM** | NaN/Inf at 256×256 | Buffer overflow: removed redundant fused call, keep only split Q/K/V/gate/up GEMMs |
| **MMA GEMM on sm_120** | Near-zero velocity | Switch to `gemm_f32_f32` (pure F32 tiled), skips `wmma` |
| **Double-stream attention path** | Needed verification on sm_120 after earlier divergence during bring-up | Re-verified on RTX 5060 Ti: GPU flash attention now matches CPU (`max_diff≈6e-6` on `--test-dit` 64×64); CPU fallback kept behind `FLUX2_CPU_DBL_ATTN=1` |
| **512-token text padding** | 35× GEMM overhead | Remove padding; return only real tokens (~15 tokens typical) |
| **Chat template** | Wrong hidden states | Strip system/chat prefix; use clean user message tokens only |

### Key Architecture Gotchas

1. **Global modulation** — unlike Qwen-Image where each block has its own modulation weights, Flux.2 Klein has one shared modulation weight for all double blocks and one for all single blocks. Each block indexes its own slice of the expanded modulation output.

2. **linear2 column split** — the single-stream `linear2` weight is `[H, H+n_ff]`. The GPU splits this into `l2_attn [H, H]` and `l2_mlp [H, n_ff]` **at upload time** by extracting columns row-by-row (cannot byte-split since each row spans H+n_ff elements).

3. **4-axis RoPE** — axes_dim=[32,32,32,32], θ=2000. Image tokens use axes 1 (row position) and 2 (col position). Text tokens use axis 3 (sequence position). Axis 0 is identity for all tokens.

4. **FP8 with per-tensor scale** — weights are stored as `uint8_t` FP8 E4M3, with a companion `weight_scale` scalar per tensor. Dequant: `f32 = fp8_to_f32(byte) × scale`.

5. **txt_dim=7680** — text hidden states are 7680-dimensional (Qwen3-4B output). This projects down to hidden_dim=3072 via `context_embedder`.

## TODO / Limitations

### Performance

- [ ] **All-GPU VAE attention**: the VAE mid-block attention still bridges through CPU for now. The heavy conv/groupnorm/upsample path is on GPU already.
- [ ] **F16/FP8 GEMM**: Current GPU GEMM is F32 (`gemm_f32_f32`). The `gemm_tiled_f16_f32` kernel produces ~27% error on sm_120 (Blackwell); `gemm_f16_f32` (MMA-based) outputs near-zero. Investigation needed — likely an NVRTC wmma intrinsic issue on sm_120.
- [ ] **Attention kernel scaling**: Single-block-per-head FA2 flash attention works but may not scale to large resolutions. No multi-block flash attention.
- [ ] **GPU text encoder startup cost**: The GPU text encoder now runs on the correct Klein safetensors weights, but it still compiles kernels and uploads the encoder in a separate CUDA session each run.

### Quality

- [ ] **No ComfyUI numerical verification**: No per-block .npy comparison against ComfyUI reference yet. Visual match confirmed but no correlation metrics.
- [ ] **Base model CFG**: Base model path not verified against reference — distilled path is primary.

### Features

- [ ] **512×512 and larger**: Only 256×256 verified. Latent sizes scale as H/8 × W/8 tokens.
- [ ] **GPU-only pipeline**: DiT and the text encoder run on GPU, but the encoder still uses a separate CUDA session and the VAE mid attention still bridges through CPU.
