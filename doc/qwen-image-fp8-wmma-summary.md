# Qwen-Image FP8xFP8 WMMA Summary

Date: 2026-05-03

## Scope

This note summarizes the native FP8 activation x FP8 weight investigation for Qwen-Image on RDNA4. Full PyTorch/diffusers model construction was not usable in the sandboxed environment because `QwenImageTransformer2DModel.from_single_file()` was host-OOM-killed around 60 GB RSS before the DiT step could run, so verification used qimg pinned runs and low-memory safetensors probes.

## Reference Behavior

ComfyUI's CUDA-native raw FP8 path is FP8 activation x FP8 weight:

- activation scale is scalar `1.0`
- activations are clamped to `[-448, 448]`
- activations and weights are wrapped as quantized FP8 tensors
- `aten.linear` is dispatched to `scaled_mm` / `_scaled_mm`
- absent `comfy_quant` input scale metadata still means scale `1.0`

qimg's original FP8xFP8 path was not a strict Comfy mirror: it used per-row activation scaling, default `max(abs(x)) / 512`.

## Low-Memory PyTorch Probes

`rdna4/qimg/run_pytorch_fp8xfp8_selective_probes.sh` avoids full diffusers loading. It opens the FP8 safetensors lazily, loads one tensor at a time, runs PyTorch native `_scaled_mm(FP8, FP8)`, and compares against F32 matmul.

Expanded block-0 single-GEMM results:

| Probe | Comfy scale=1 | qimg per-row `/512` |
|---|---:|---:|
| `img_in` | 52.88 dB | 47.23 dB |
| `txt_in` | 70.02 dB | 68.03 dB |
| `temb_fc1` | 62.22 dB | 50.91 dB |
| `temb_fc2` | 67.17 dB | 50.21 dB |
| `blk0_img_mod` | 52.80 dB | 42.63 dB |
| `blk0_txt_mod` | 50.81 dB | 39.39 dB |
| `blk0_img_q/k/v` | 66.56 / 70.02 / 59.36 dB | 62.18 / 66.58 / 52.88 dB |
| `blk0_txt_q/k/v` | 63.27 / 63.52 / 60.02 dB | 52.82 / 53.46 / 50.44 dB |
| `blk0_img_mlp_fc1/fc2` | 55.35 / 69.43 dB | 49.61 / 68.34 dB |
| `blk0_txt_mlp_fc1/fc2` | 51.11 / 64.81 dB | 44.46 / 64.96 dB |

This shows scalar scale=1 is better for early isolated GEMMs, especially modulation and text MLP FC1.

## qimg Full-Denoise Results

qimg now supports:

- `--fp8-act-scale-mode perrow` via `QIMG_FP8_ACT_SCALE_DIV`, default `/512`
- `--fp8-act-scale-mode comfy`, scalar scale=1 and clamp/cast
- `--fp8-act-scale-mode clamp`, scale=1 when `row_max <= 448`, otherwise `row_max/448`

Pinned 256x256 / 20-step results against `final_latent_packed_256.bin`:

| Mode | Denoise | Traffic | Latent cosine | Latent PSNR |
|---|---:|---:|---:|---:|
| BF16xFP8 WMMA | 24.1 s | 1723 GB / 73.1 TF | 0.999971 | 51.70 dB |
| FP8xFP8 per-row `/512` | 20.3 s | 984 GB / 73.1 TF | 0.979988 | 24.59 dB |
| FP8xFP8 comfy scale=1 | 22.8 s | 984 GB / 73.1 TF | 0.301305 | 4.20 dB |
| FP8xFP8 clamp | 20.3 s | 984 GB / 73.1 TF | 0.986565 | 25.93 dB |

The strict Comfy mirror does not survive full denoise because later activations saturate. The clamp mode avoids catastrophic saturation and is the best native-FP8 no-fallback mode from this pass, but it remains far below the 50 dB target.

Generated inspection artifact:

- `rdna4/qimg/apple_fp8fp8_clamp_256.png`

## Memory And Bandwidth

BF16xFP8 is the only measured path here that clears 50 dB. It keeps checkpoint weights resident as FP8 bytes and converts FP8 to BF16 inside the GPU WMMA kernel while staging LDS. There is no global BF16-expanded weight buffer and no host-side FP8 to BF16 conversion on the hot path.

At 256x256 / 20 steps:

- BF16xFP8: 1723 GB GEMM traffic, 24.1 s denoise, 51.70 dB
- FP8xFP8: 984 GB GEMM traffic, 20.3 s denoise, 24.59-25.93 dB depending on scale mode
- extra persistent FP8 activation scratch: 3.0 MB

## Conclusion

Native FP8xFP8 without dispatch gating or BF16 fallback does not reach the 50 dB quality target on this checkpoint. Scalar scale=1 explains why early PyTorch/Comfy-style single GEMMs can look good, but full denoise still accumulates activation-FP8 error and saturation. The practical high-quality path remains BF16xFP8 WMMA; any native-FP8 speed path needs selective suppression/fallback or a materially different quantization strategy.
