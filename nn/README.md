# GN: native policy/value network

MIT, independently authored generic C API (`gn.h`), with no shogi/rules or ML
framework dependency. CPU FP32, custom CUDA sm120 BF16 MMA, and HIP gfx1201 BF16
WMMA implement forward, backward, accumulation/clipping and AdamW. GPU source
compiles through NVRTC/HIPRTC; compilation is not hardware correctness validation.
No vendor BLAS or neural runtime is used.

From GEMM root:

```sh
make -C nn check
python3 -B nn/reference.py              # optional numpy + torch
python3 -B nn/compile_gpu.py            # compiler-only GPU checks
nn/build/test_gpu cuda-fp32             # physical hardware required
nn/build/test_gpu cuda
nn/build/test_gpu hip-fp32
nn/build/test_gpu hip
```

GPU tests return 77 for unavailable, not PASS. An optional checkpoint path and
`wide` or `full` argument test 9×9 C32 or the default C256/20-block network,
including checkpoint reload. `stress` retains a highly correlated sinusoidal
full-network case that fails the tight gradient gate even in the FP32 path.
Reduction order and ReLU boundaries are suspected, but the discrepancy is not
fully diagnosed. The test is retained, not silently skipped. The main tests
gate global gradient relative L2 at .001, inference
relative L2 at .01 (BF16) or .0001 (FP32), and check AdamW elementwise using
identical already-compared gradients. These are not a strength certification. Benchmark with
`gn_tool bench MODEL BACKEND BATCH ITERATIONS` after validating that backend.

Defaults: 9×9×80 NHWC, 139 action planes, C256, 20 blocks (every fifth Transformer,
otherwise two 3×3 convolutions), attention head width 32, 2C SwiGLU, policy linear
and 32→flatten→256→3 WDL head. Smaller generic configs support tests. Conv weights
are flattened OHWI; linear weights `[out,in]`. BN epsilon 1e-5, momentum .1,
unbiased running variance; LN epsilon 1e-5. No dropout. BN statistics belong to
each **microbatch**, not the effective accumulated batch.

Inference returns dense raw policy logits and WDL probabilities. Training
targets use -1 for illegal actions, normalized probabilities for legal actions.
`gn_backward` sums gradients; `gn_update` averages by accumulated sample count,
clips, then updates all learned parameters. `gn_zero_grad` discards pending
gradients. Error text is thread-local; externally serialize access to a model.
Training failure may change BN statistics or partial gradients: discard/reload
the failed in-memory transaction rather than blindly retrying its update.

CPU projections have runtime AVX2/FMA dispatch and up to eight persistent workers;
other POSIX architectures use scalar code. CPU backward is serial/vectorized.
GPU weights, gradients and moments stay resident; host transfers are inference
results, metrics and checkpoints. Public operations synchronize. BF16 operand
conversion is round-to-nearest-even. Inference uses one BF16 product. Training
uses three-component operand decomposition and six products (component indices
i+j≤2), with a separate correction accumulator, to approach FP32 accuracy while
using MMA/WMMA. Three-product compensation was insufficient for the deep model.
Matrix accumulators/master state stay FP32; sensitive reductions use doubles.
GPU attention/reduction kernels still need profiling/tuning.

Safetensors stores model tensors, `adam.m.*`, `adam.v.*`, `__config` U64[12]
(gn_config field order including memory cap), and `__state` U64[2] (step, RNG).
Checkpoints require an optimizer boundary, validate shapes/dtypes/finite values,
and use fsync+rename. The consuming application owns JSON provenance sidecars.
CPU tests require exact resumed inference and the next optimizer update.

GNR1 (`gn_replay.h`) is little-endian: magic, u32 side/channels/actions, channel
storage bytes (0 packed bits, 1 broadcast F32, 2/3 coordinates), then records:
u64 game/generation, u32 ply/legal-count/WDL, features, u32 action/visit pairs.
Reader rejects malformed/truncated data, duplicate actions, nonfinite scalars
and zero total visits. `gn_tool train` keeps recent million non-validation offsets
and excludes `game % 20 == 0`; `validate` uses only those held-out games.

The compiled closure uses MIT GEMM safetensors helpers and Apache-2.0 CUEW/ROCEW
loaders, preserving notices. PyTorch/numpy are optional oracle dependencies.
No GPL engine code or third-party shogi model was used to author this module.
