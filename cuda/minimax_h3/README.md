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
  | Qwen3-VL text encoder | 6.0 s | 40 s |
  | DiT, per Euler update (50 blocks) | 82 s | ~128 s |
  | VAE decode (124 frames) | 68 s | 249 s |

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

## Optional cuDNN attention (opt-in)

DiT and refiner BF16 head-128 attention can use cuDNN 9's fused SDPA through a
separate bridge library. The default build and runtime do not need cuDNN.

```sh
make -C cuda/minimax_h3 cudnn-deps                      # fetch header-only cudnn-frontend
make -C cuda/minimax_h3 cudnn CUDNN_INCLUDE=/path/to/cudnn9/include
tmp/video-cuda/h3-build/h3_cuda ... --cudnn-attention auto \
    --cudnn-library /path/to/libcudnn.so.9   # optional; or pass a libh3_cudnn.so path
```

- **Finding the bridge:** `auto` loads `libh3_cudnn.so` from beside the runner or
  library; an explicit bridge path may be passed instead.
- **Finding cuDNN:** `--cudnn-library PATH` (C API `cudnn_library`) selects
  `libcudnn.so.9`. Without it, the default loader search path and fixed system
  locations are tried. No environment variables select or locate libraries.
- **Failure handling:** if cuDNN is unavailable, `auto` prints a warning and falls
  back to FlashAttention-2. An explicit bridge path is an error.
- **Other entry points:** the C API is
  `h3_set_cudnn_attention(ctx, mode, cudnn_library, ...)`, and `generate.py`
  accepts `--cudnn-attention` / `--cudnn-library`. Metrics record
  `cudnn_attention_calls` and the loaded library.

With cuDNN 9.19 at full resolution, a DiT update drops from 93.8 s to **90.2 s**.
Outputs stay within the same non-bit-exact noise band against PyTorch as
FlashAttention-2 (64×64 39-update video/audio latent relative L2 0.041 / 0.054
against 0.044 / 0.076).

## Fused kernels (CUDA default)

- **DiT:** each projection input goes through one `h3x_row_quant` pass that does
  RMSNorm+modulate (or SwiGLU of the INT32 fc1 sums, or the packed attention
  heads), the factorized ConvRot-256, and INT8 quantization. The INT32 GEMM sums
  are dequantized by their consumers: `h3x_qkv_pack` (q/k/v split + head RMSNorm +
  RoPE + packing) and `h3x_dequant_gate` (residual gate, per FFN chunk).
- **VAE:** RMSNorm, SwiGLU and attention unpacking write FP16 directly into the
  GEMMs. Bias and rounding are folded into the qkv packer, SwiGLU and the scale-add
  residual.
- **Numerics:** every fused kernel reproduces the unfused expressions, and a
  39-update run is bit-identical to the unfused graph (debug switch
  `H3_DEBUG_UNFUSED=1`; all 87 captures).
- **Default ConvRot on CUDA:** the default is now the factorized rotation
  (`--convrot-hipblas 0`). It differs from dense cuBLAS only in FP32 summation
  order and lands slightly closer to PyTorch (64×64 39-update video/audio latent
  relative L2 0.028 / 0.049, against 0.044 / 0.076 dense). `--convrot-hipblas 1`
  restores the dense, unfused DiT path.
- **Effect:** elementwise GPU time fell from ~21 s to ~5 s per DiT update and from
  ~33 s to ~8 s in the VAE. Wall time gained less (DiT 93.8 → 89 s, VAE ~96 → 89 s):
  attention (~70 s) dominates DiT, and VAE tile enqueue is now partly host-bound
  (~12 s idle).
- **VAE host overlap:** tile inputs upload through pinned staging without a stream
  sync, and each chunk's frames are converted and emitted on a worker thread while
  the GPU decodes the next chunk. VAE 89 → 82 s, with identical frames.
- **Qwen layer prefetch:** layer i+1's INT8 matrices stream on the prefetch worker
  while layer i runs. The staging copy out of the mmap is split across 4 threads
  (page faults made a single-threaded copy the bottleneck) and uses 4 pinned 64 MiB
  slots. Qwen 8.3 → 6.0 s with an identical hidden state; the PCIe 3.0 floor here
  is ~3.6 s for 26 GB.
- **VAE GEMM epilogue:** the four decoder projections run through cuBLASLt with the
  FP16 bias, FP32 accumulation and FP16 output fused into the epilogue (as torch's
  half `F.linear`). Consumers read the finished FP16 values. GEMM throughput rose
  from ~29.8 to ~33.4 TFLOPS (~95% of the measured peak). Full-resolution tiles are
  bit-identical to the repository GEMM path; 64×64 tiles differ by ~3e-4 relative
  L2. The 39-frame VAE component check passes against PyTorch (max relative L2
  0.0014). The debug switch `H3_DEBUG_VAE_REPO_GEMM=1` selects the repository GEMM. VAE 80 → 72 s.
- **DiT row kernel:** each warp rotates whole 256-element groups in registers via
  shuffles, in the same operation order, and the row stays in registers until
  quantization. There are no block barriers per group and no 28 KB shared buffer.
  `h3x_row_quant` went from 2.97 to 1.88 s per update (~70% of memory bandwidth),
  bit-identical. DiT 88 → 84 s.
- **GPU idle removal:**
  - The DiT/Qwen prefetch stages every tensor of the next layer (norms, scales,
    biases, adaLN), and the GPU converts them, so the compute stream never queues
    host-to-device copies behind prefetch traffic. DiT idle fell from ~7 s to ~1.4 s
    per update.
  - The VAE runs one tile pipeline across all temporal chunks with uninitialized
    double-buffered canvases.
  - The 4.6 GB decoder is bulk-staged into the weight cache by the parallel prefetch
    worker, while dummy cuBLASLt GEMMs load the projection kernels.
  - VAE idle fell from ~6.5 s to ~4 s at full resolution, and to 0.75 s inside the
    39-frame probe. Outputs are bit-identical.

Runtime behavior is selected only by runner arguments and the C API. Environment
variables are reserved for debug/parity diagnostics (`H3_DEBUG_UNFUSED`,
`H3_DEBUG_VAE_REPO_GEMM`). The cuDNN bridge sets cudnn-frontend's private
`CUDNN_FRONTEND_CUDART_LIB_NAME` to `libcudart.so.13` (that library's only cudart
selector, never overriding an existing value).
