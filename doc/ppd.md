# PPD Reference Verification And Performance

Pixel-Perfect Depth (PPD) is a diffusion depth pipeline:

1. DA2/DINOv2 ViT-L semantic encoder.
2. 24-block DiT denoiser.
3. Four Euler denoising steps.
4. Final latent-to-depth conversion.

This report tracks the CUDA implementation in `cuda/ppd/` against a PyTorch reference in `ref/ppd/`.

## Reproduction

Weights used for the measurements below:

| Component | Path |
|---|---|
| PPD DiT | `/home/syoyo/work/gemm/vlm-ptx/models/ppd/ppd.pth` |
| DA2 ViT-L semantic encoder | `/home/syoyo/work/gemm/vlm-ptx/models/ppd/depth_anything_v2_vitl.pth` |

Primary test image:

| Image | Path | Resolution |
|---|---|---:|
| Street | `/home/syoyo/work/gemm/vlm-ptx/cuda/ppd/street.ppm` | 640x480 |

Build and run CUDA:

```sh
make -C cuda/ppd
./test_cuda_ppd \
  /home/syoyo/work/gemm/vlm-ptx/models/ppd/ppd.pth \
  /home/syoyo/work/gemm/vlm-ptx/models/ppd/depth_anything_v2_vitl.pth \
  -i /home/syoyo/work/gemm/vlm-ptx/cuda/ppd/street.ppm \
  -o /tmp/ppd_cuda_mma_street.pgm \
  --npy /tmp/ppd_cuda_mma_street.npy \
  -v 1 --repeat 3
```

Run PyTorch reference:

```sh
python3 ref/ppd/run_reference.py \
  -i /home/syoyo/work/gemm/vlm-ptx/cuda/ppd/street.ppm \
  --npy /tmp/ppd_ref_street.npy \
  --dtype fp16
```

Compare outputs:

```sh
python3 ref/ppd/compare.py \
  /tmp/ppd_ref_street.npy \
  /tmp/ppd_cuda_mma_street.npy \
  --mae-thresh 0.01 \
  --pearson-thresh 0.999
```

## Performance Status

Measured on NVIDIA GeForce RTX 5060 Ti, driver 595.71.05, street.ppm 640x480.

| Mode | Total | DA2 semantic | DiT diffusion | Ratio vs PyTorch | Status |
|---|---:|---:|---:|---:|---|
| PyTorch reference FP16 | 1282.2 ms | 470.1 ms | 615.9 ms | 1.00x | PASS reference |
| CUDA default FP16 MMA GEMM | 1315.8 ms | 79.5 ms | 1227.4 ms | 1.03x slower | PASS |
| CUDA tiled GEMM fallback, `PPD_MMA_GEMM=0` | 5571.9 ms | 447.8 ms | 5114.3 ms | 4.35x slower | PASS |

The CUDA default now meets the DA3-style target: about 1.03x slower than the PyTorch FP16 reference, while keeping output drift low.

### CUDA Phase Profile

Verbose CUDA profile, default FP16 MMA GEMM, `-v 2 --repeat 1`:

| Phase | Time |
|---|---:|
| DA2 semantic encoder | 78.9 ms |
| DiT low-res blocks, step 0, 12 blocks, nt=1200 | 40.8 ms |
| DiT high-res blocks, step 0, 12 blocks, nt=4800 | 256.8 ms |
| DiT diffusion, 4 Euler steps | 1224.5 ms |
| End-to-end inference | 1313.0 ms |

Attention remains a major high-res cost:

| Attention sample | Time | 12-block estimate |
|---|---:|---:|
| Low-res L0 attention | 0.99 ms | 11.8 ms |
| High-res L12 attention | 12.65 ms | 151.8 ms |

## Verification

Reference: PyTorch FP16, fixed seed 42.

| Candidate | Pearson | MAE | RMSE | Max AE | Assessment |
|---|---:|---:|---:|---:|---|
| CUDA default FP16 MMA GEMM vs PyTorch | 0.999991 | 0.000816 | 0.001168 | 0.065451 | PASS |
| CUDA tiled fallback vs PyTorch | 0.999992 | 0.000789 | 0.001137 | 0.065493 | PASS |
| CUDA default FP16 MMA GEMM vs tiled fallback | 1.000000 | 0.000136 | 0.000238 | 0.051923 | PASS |

The fixed PyTorch reference uses a libc `rand()` Box-Muller noise generator to match the C/CUDA initial latent sequence.

## Precision Defaults

Default CUDA inference prioritizes lower drift:

- `PPD_MMA_GEMM=1` is the default. Aligned GEMMs use FP16 tensor cores with FP32 accumulation.
- Unaligned GEMMs fall back to the tiled FP16/F32 kernel. The DA2 patch embedding has `K=588`, so it stays on the fallback path.
- `PPD_MMA_GEMM=0` forces the previous tiled path for regression checks.
- FP8 kernels are present in the runner source but remain disabled by default because diffusion trajectories amplify quantization errors.

## TODO

1. Add cumulative CUDA event timing for GEMM, attention, layernorm/adaLN, RoPE, and elementwise work across all DiT steps. Current verbose output samples only block 0 and block 12 attention.
2. Optimize high-res attention. At 4800 tokens, L12 attention is about 12.65 ms and dominates the remaining CUDA gap.
3. Evaluate lower-drift fast attention before changing the default attention path. Any MMA/FP8 attention mode must be gated until it passes PyTorch drift checks on street and tiny images.
4. Revisit FP8 only as an opt-in speed mode with full output comparison. Candidate work: per-row or per-channel weight scales, activation scaling, and selective FP16 fallback for sensitive DiT layers.
5. Reduce Python reference overhead if it becomes part of CI. The current script is intended for local profiling and correctness checks, not fast automated testing.
