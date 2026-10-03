# Native HunyuanVideo 1.5 on ROCm

This backend builds the existing C++ Hunyuan model graphs against HIP. It supports
the same quality T2V, quality I2V and fast12 I2V profiles, model manifest,
prepared-image inputs, callbacks and diagnostic captures as the CUDA runner.
FP16 WMMA supplies GEMM and unmasked 128-dimensional attention. Other attention
shapes, causal VAE convolution and FP32 islands use repository HIP kernels.

```sh
make -C rdna4/hunyuan_video15_native -j4
make -C rdna4/hunyuan_video15_native test
make -C rdna4/hunyuan_video15_native test-gpu
python3 rdna4/hunyuan_video15_native/generate.py \
  --model /mnt/disk01/models/hv15 --task t2v --preset quality \
  --prompt 'A person smiles naturally.' \
  --out tmp/video-rocm/hunyuan-video --allow-experimental
```

For I2V, add `--task i2v --image IMAGE`; fast12 uses `--preset fast12`.
The wrapper prepares and packages the same 480×848, 81-frame, 24 fps profile.
The existing wrapper also accepts `--backend rocm` explicitly:

```sh
python3 cuda/hunyuan_video15_native/generate.py --backend rocm \
  --model /mnt/disk01/models/hv15 --task t2v --preset quality \
  --prompt 'A person smiles naturally.' \
  --out tmp/video-rocm/hunyuan-video --allow-experimental
```

Outputs are `tmp/video-rocm/hv15-build/hv15n_rocm` and `libhv15n_rocm.so`.
The C API remains [hv15_native.h](../../cuda/hunyuan_video15_native/hv15_native.h).
ROCm selects `--gemm repo --gemm-fallback error` by default. Optional
`--gemm hipblas` or `--gemm-fallback hipblas` explicitly loads hipBLAS.
CUDA retains its existing defaults. The runner requires gfx1200/gfx1201, HIPRTC,
and HIP headers at `/opt/rocm/include`; GPU access requires `/dev/kfd` and the
AMD render node. No hipcc or GPU libraries are linked at build time.

The default 14,336 MiB budget reserves 3,072 MiB for runtime overhead. Pinned
transfer staging is limited to two 64 MiB buffers. Process VRAM sampling uses
AMD DRM/KFD counters, and the wrapper shares a device lock with H3.
Tuning is supplied through runner arguments.

The RX 9070 XT GPU suite passes GEMM tails, FP32 precision, masked/grouped-query
attention, WMMA attention, causal convolution, fused normalization/residuals,
and stream-ordered buffer reuse. The original CUDA host tests and 15 Python
orchestration tests also pass. Full independent video parity is still required;
outputs remain marked `hv15n_rocm_experimental` and `parity: unverified`.
The reference verifier in `ref/hunyuan_video15_native/verify.py` accepts both
native backend identities and retains the existing pinned-reference checks.

Actual-weight smoke probes completed all 26 Qwen layers, the 12 ByT5 blocks,
the 27 SigLIP attention blocks, the VAE encoder and decoder,
and all 54 DiT blocks for each of `quality_t2v`, `quality_i2v` and `fast12_i2v`.
Independent FP32 encoder comparisons pass on shared inputs: relative L2 is
3.09e-6 for Qwen, 5.41e-7 for ByT5 and 2.92e-5 for SigLIP. The SigLIP reference
initializes its nonpersistent position IDs explicitly after loading meta tensors.
The DiT probes used a 1×2×2 latent geometry and produced finite outputs with
2,877–2,886 MiB managed VRAM. These probes do not certify full-video parity.

A full fast12 I2V native run completed all 12 updates and 81 frames within the
memory budget: 11,615 MiB sampled process VRAM and 11,230 MiB managed allocation.
Independent conditioning comparisons passed, including the VAE encoder
(relative L2 0.000501). The first velocity and first ten latent updates passed,
but the final two updates failed; final relative L2 was 0.0319. This does not
establish full-video acceptance. The ROCm DiT now preserves upstream FP16
rounding boundaries through linear outputs and activations. Autocast LayerNorm
and its modulation multiply/shift remain FP32; only `1 + scale` rounds to
FP16. Gate products round to FP16, while residual addition follows the residual
dtype, including FP32 text promoted by the vision projection. GPU regressions
cover these boundaries and restoration of encoder/scheduler precision scopes.
DiT convolutions also round their core result before the separate FP16 bias
addition. An actual-weight pointwise convolution matched all 245,760 PyTorch
outputs for that sequence; adversarial GPU regressions cover both pointwise
and im2col paths. Autocast LayerNorm uses vectorized Welford reduction with
ROCm's reciprocal and FMA ordering, avoiding cancellation for shifted inputs.
Independent tests match PyTorch exactly at widths 1152 and 2048; width 3072
has maximum FP32 error 1.19e-7. Timestep embeddings retain the upstream CPU
FP32 frequency basis and compute arbitrary-time phases and trigonometry on
GPU. Across 81 quality/fast12/random timestep cases, all 20,736 FP16 values
match the installed PyTorch ROCm reference exactly.
A full-resolution first-update diagnostic with this correction and AOTriton
passed (velocity relative L2 0.00237); that result does not establish complete
schedule acceptance. The pinned upstream uses compiled FlexAttention for its
long joint attention. The ROCm DiT now uses its 128-query/64-key WMMA schedule,
transposed QK score layout, base-2 softmax and reference reduction order.
Independent full-token-length tests matched all 4,368,512 FP16 output values
exactly on the 34,129-token fixture. The 128-channel Q/K RMSNorm path also
matches all 524,288 outputs of a random Half corpus, including reciprocal
square-root refinement and FP32 rounding before Half conversion. Fused Q/K
preparation is checked against that standalone path. Full-video acceptance
with these corrections is still pending.

A fresh full-resolution, 12-update denoising diagnostic with shared independent
conditioning passed all 15 comparisons. Final latent relative L2 was **0.01150**
and cosine was **0.99993394**, with **8,174.8 MiB** managed allocation. This
validates the corrected complete denoising schedule; fresh native conditioning
and all 81 decoded frames still require the complete pipeline comparison.

The FP16 GEMM path uses a 128×128 LDS tile for sufficiently large shapes and
retains the smaller kernel for tails and small matrices. GPU tests compare the
tiled output against both FP64 sums and the smaller kernel. To measure both
kernels on representative projection shapes:

```sh
make -C rdna4/hunyuan_video15_native bench-f16
```

An optional standalone AOTriton adapter supplies long FP16 DiT attention while
GEMMs retain WMMA. Build it with an AOTriton 0.11.2 SDK, then select its path
explicitly with `--aotriton-bridge`. The normal build has no SDK dependency.
The wrapper records the library hash and metrics count provider calls.
The C API adds `hv15n_set_aotriton_bridge()` for configuring an idle context;
the existing configuration struct and CUDA ABI are unchanged.

```sh
make -C rdna4/hunyuan_video15_native aotriton AOTRITON_ROOT=/path/to/sdk
make -C rdna4/hunyuan_video15_native test-aotriton \
  AOTRITON_BRIDGE=tmp/video-rocm/hv15-build/libvideo_aotriton.so
```

FP16 benchmark comparisons matched every output bit on three large shapes.
For 256×21504×5376, the small kernel took 43.9 ms and the tiled kernel 8.81 ms
(6.72 TFLOPS). A full workgroup barrier protects its two LDS operand stages;
long-K tests cover the synchronization defect found by the larger benchmark.
