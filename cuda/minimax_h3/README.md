# MiniMax H3 INT8 on CUDA

CUDA build of the native MiniMax H3 **Ref2VA pruned INT8 ConvRot** text-to-video
runner. It compiles the shared H3 graph in [`rdna4/minimax_h3`](../../rdna4/minimax_h3)
(`runtime.hpp`, `h3.cpp`, `runner.cpp`) on top of the native Hunyuan CUDA runtime
(`cuda/hunyuan_video15_native/gpu.cpp`). It has no PyTorch dependency. The target
GPU is the RTX 5060 Ti 16 GB (sm_120).

```sh
make -C cuda/minimax_h3 -j16 test
python3 cuda/minimax_h3/generate.py \
  --prompt 'A red ball rolling on a wooden table, cinematic lighting.' \
  --out tmp/video-cuda/h3-video --allow-experimental
```

The default model is `/mnt/nvme01/models/h3/weights`. Build products are in
`tmp/video-cuda/h3-build/` (`h3_cuda`, `libh3_cuda.so`, `component_probe`).
`generate.py` here is the shared wrapper with `--backend cuda` selected.

CUDA-specific execution (`#ifndef HV15N_ROCM` in the shared sources):

- **INT8 projections**: cuBLAS IMMA (INT8×INT8→INT32) writes into the output
  buffer. `h3_dequant` then applies `acc * (x_scale * w_scale)` plus BF16
  rounding. This is the same epilogue as the RDNA4 WMMA kernel, and INT32 sums
  are exact.
- **Dense BF16 / FP32 / ConvRot rotation**: cuBLAS GEMM with BF16 inputs and FP32
  output, and pedantic FP32 respectively (the `*_hipblas` options map to cuBLAS).
- **Attention**: private FlashAttention-2 (`cuda/fa2`). DiT and the refiner use BF16
  with head dim 128; the VAE decoder uses FP16 with head dim 64. Qwen causal
  attention keeps the shared FP32 kernel. The AOTriton bridge is ROCm-only.
- **Weight staging**: tensors larger than 64 MiB upload through bounded 64 MiB
  pinned chunks.

Memory behaviour and the 14,336 MiB budget are unchanged from the RDNA4 runner.
Weights stream per block from mmap, and a 64 GB host is required.

## Measurements (RTX 5060 Ti 16 GB)

- **Full resolution, one update** (1344×768, 124 frames, `--steps 2`): **202 s**
  wall, **11,552 MiB** peak process VRAM (memory fit passes). Stage times against
  the independent PyTorch reference on the same GPU:

  | stage | native CUDA | PyTorch reference |
  |---|---|---|
  | Qwen3-VL text encoder | 7.4 s | 40 s |
  | DiT, per Euler update (50 blocks) | 94 s | ~128 s |
  | VAE decode (124 frames) | 98 s | 249 s |

  DiT attention runs at the same speed as PyTorch's (32.7 vs 32.4 TFLOPS) and is
  ~65% of DiT time. DiT INT8 weights for block i+1 are prefetched by a worker
  thread on a private stream while block i computes (saved ~1.6 s per update). VAE gains come from three changes: the 4.6 GiB decoder stays
  resident, tiles are pipelined (the next tile is enqueued before host stitching;
  pixels come back through async pinned copies), and host FP16 blending uses
  F16C. INT8 projections use 4096-row FFN chunks and a fused bias+round epilogue.
  The optimizations are numerically neutral: a 39-update diagnostic reproduces all
  87 captured arrays bit-for-bit.
- **64×64, five frames, 39 updates**: 152 s and 4.6 GB VRAM (the RX 9070 XT took
  313 s).
- **Components**: checkpoint projections match the CPU INT32 reference (cosine
  ≥0.99999999). Every Qwen layer-0 stage is bit-exact except FP32 causal
  attention (99.95% of values exact). Final Qwen hidden state: relative L2 0.0016.
- **End-to-end parity is not bit-exact**, so the 39-update diagnostic fails its
  thresholds (final video latent relative L2 0.044, audio 0.076). PyTorch-CUDA's
  FlashAttention and cuBLAS FP32 reduction orders differ from the native kernels.
  For comparison, re-running the independent reference with only its SDPA
  backend switched (FLASH→EFFICIENT) drifts by the same amount (0.036 video /
  0.079 audio, frames 0.024). The 39-update chain amplifies any non-bit-exact
  difference, and CUDA sits at that same noise floor.
