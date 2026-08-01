# Qwen-Image INT8 — background "wobble" and the SmoothQuant fix

## Symptom
Plain W8A8 int8 generations have a low-frequency "wobbly"/painterly texture in flat
regions (skies, studio backdrops) that fp8 and the PyTorch/ComfyUI ground truth do not.

## Isolation (same seed, same pipeline, only GEMM precision differs)
- Our **fp8 == ComfyUI PyTorch fp8** (byte-identical composition) → our pipeline is correct
  and the smooth background is the ground truth.
- **int8 vs fp8** differ only in the GEMM precision → the extra texture is the int8 path.
- It is **not** the GEMM (our `gemm_int8_s32` is bit-exact vs cuBLAS, 0/3.1M mismatches) and
  **not** the weights (per-channel int8 ≈ 0.4% error, higher fidelity than fp8).

Low-frequency wobble metric (block-avg, remove planar trend, residual std; lower = smoother):

| region        | plain int8 | **SmoothQuant α=0.5** | fp8 (target) |
|---------------|-----------:|----------------------:|-------------:|
| mountain sky  |     10.35  |              **1.31** |        1.33  |
| owl backdrop  |     18.28  |             **14.71** |       10.09  |

## Root cause: per-token activation quant + activation outliers
W8A8 quantizes activations dynamically **per token** (scale = max(|x_token|)/127). Transformer
activations have huge per-input-channel outliers — measured here (block 0, amax vs median):

- `attn.to_q`: **19×**, `img_mlp` fc1: **147×**, `img_mod` (adaLN): **990×**

A 990× outlier forces a per-token scale so coarse the bulk channels collapse below one int8
level → low-frequency error that compounds over 60 layers × 20 steps → the wobble. fp8 (e4m3)
avoids it because its floating-point range keeps both the outlier and the small values precise.
This is the textbook reason fp8 beats naive int8 W8A8 without calibration.

## Fix: SmoothQuant (α = 0.5)
Migrate per-input-channel activation outliers into the weights, using the identity
`Y = X·Wᵀ = (X/s)·(W·s)ᵀ` with `s[j] = act_absmax[j]^α / weight_absmax[j]^(1-α)`.
Weights then quantize cleanly per-output-channel; activations become smooth so per-token int8
works again. **The GEMM math is unchanged** — so it stays bit-exact CUDA↔HIP and the int32
GEMM still equals cuBLAS.

Result: SmoothQuant α=0.5 brings the mountain sky to **1.31 ≈ fp8's 1.33** (~8× wobble
reduction); backgrounds go from blotchy to clean gradients. **No perf cost** (1.77 s/step,
the divide folds into the existing quant kernel; GEMM untouched).

### α sweet spot (composition-free per-layer GEMM-output relL2 vs f32, lower = better)

| layer (act outlier)        | plain  | **α=0.5** | α=0.75 | α=0.9  |
|----------------------------|-------:|----------:|-------:|-------:|
| `attn.to_q` (19×)          | 0.0152 |    0.0154 | 0.0146 | 0.0237 |
| `img_mlp` fc1 (147×)       | 0.0237 | **0.0157**| 0.0179 | 0.0195 |
| `img_mod` adaLN (990×)     | 0.0199 | **0.0049**| 0.0101 | 0.0139 |
| blk10 `img_mlp` fc1        | 0.0167 | **0.0137**| 0.0147 | 0.0153 |

**α=0.5 wins.** α=0.75/0.9 over-migrate outliers into the weights (the per-output weight scale
gets dominated by a few blown-up columns → weight error overtakes the activation savings).
Empirically α=0.75 also produced a visibly broken `steampunk_owl`. Use **α=0.5**.

## Implementation
- `ref/qwen_image/calib_smoothquant.py` — diffusers forward-hook calibration: per-input-channel
  activation absmax for all 846 Linear layers (fp8 DiT, sequential CPU offload, 2 prompts × 8 steps).
- `ref/qwen_image/quantize_int8_smooth.py` — SmoothQuant requant of the bf16 base → int8 with a
  `<name>.smooth_scale` [in] F32 companion (W pre-scaled by s). `--alpha 0.5`.
  Output: `qwen_image_int8_smooth.safetensors` (20.5 GB; 846 int8 + 846 weight_scale + 846
  smooth_scale + 1087 bf16).
- `cuda/qimg/cuda_qimg_runner.h` — fat buffer extended to `[wscale n_out][smooth n_in][wq]`;
  `quant_act_perrow_int8` takes an optional `smooth` pointer (divides X by s before per-token
  quant); auto-detected via the `.smooth_scale` tensors (`use_int8_smooth`). The plain int8 and
  fp8 paths are byte-identical when off (self-test still PASS, `gemm_int8_s32 == cuBLAS`).

## Sample outputs (cuda/qimg/, untracked)
- `int8_samples/` — plain int8 (wobbly) + fp8/ComfyUI comparison
- `fp8_reference/` — our fp8 + ComfyUI fp8 (clean reference)
- `int8_smooth_samples/` — **SmoothQuant α=0.5 (the keeper)**
- `int8_smooth_a075/` — α=0.75 (worse; rejected)
