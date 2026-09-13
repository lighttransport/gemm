# GN: native policy/value network

MIT, independently authored generic C API (`gn.h`), with no shogi/rules or ML
framework dependency. CPU FP32, custom CUDA sm120 BF16 MMA, and HIP gfx1201 BF16
WMMA implement forward, backward, accumulation/clipping and AdamW. GPU source
compiles through NVRTC/HIPRTC; compilation is not hardware correctness validation.
Default backends use no vendor BLAS or neural runtime. An explicit optional
`hip-blaslt` build/backend uses AMD hipBLASLt for selected matrices.

The default `cuda` path now uses tiled asynchronous operand staging and parallel
training reductions. See [sm120 results and integer experiments](SM120.md) for
the measured speedup, FLOP accounting and the **unmet 95% peak target**.
`cuda-legacy` supports A/B checks; `cuda-int8` and `cuda-int16` are explicit,
unqualified quantized-operand experiments, not integer-only training.

The `hip` path now uses LDS-tiled WMMA and parallel training operations;
`hip-blaslt` is an opt-in hybrid and `hip-legacy` retains the original kernels.
See [RDNA4 results, builds and qualification](RDNA4.md): **8.12×/8.47×** native/hybrid
batch-16 speedups were measured, but neither reaches 95% peak, and a new full
batch-16 gradient discrepancy remains unresolved (including legacy/FP32 paths).
The [precision follow-up](RDNA4.md#precision-follow-up) adds fused convolution
packing, coalesced attention, and experimental `hip-bf16`, `hip-bf16-acc`,
`hip-bf16-acc128`, `hip-int8`, `hip-int8-i64`, and `hip-int16` training paths.
The last uses four INT8 WMMA products with INT32 partials and an exact INT64
combine, not native INT16 WMMA. All keep FP32 master/optimizer/non-matrix state.
These experiments are not enabled by the engine or long-campaign preflight.

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
including checkpoint reload. A final optional batch argument defaults to 2;
`test_gpu hip-blaslt FILE full 16` exercises the hybrid's vendor training shapes.
`stress` retains a highly correlated sinusoidal
full-network case that fails the tight gradient gate even in the FP32 path.
Reduction order and ReLU boundaries are suspected, but the discrepancy is not
fully diagnosed. The test is retained, not silently skipped. The main tests
gate global gradient relative L2 at .001, inference
relative L2 at .01 (BF16) or .0001 (FP32), and check AdamW elementwise using
identical already-compared gradients. These are not a strength certification. Benchmark with
`gn_tool bench MODEL BACKEND BATCH ITERATIONS` after validating that backend.
The optional trailing `BF16_PEAK_TFLOPS INT8_PEAK_TOPS` overrides the benchmark's
explicit RX 9070 XT dense reference (195 / 389). JSON separates useful GEMM work,
precision-compensation products, FP32 attention, and whole-step peak percentages.
`make -C nn rdna4-precision` builds the standalone accumulator diagnostic;
`sh nn/test_rdna4_precision.sh` exercises it. `test_gpu BACKEND FILE wide 2 report`
continues through numerical mismatches to report gradients/update/reload, but
still returns failure if the unchanged numerical gates fail. Its checkpoints
are diagnostic artifacts, not qualified trained models.

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
CUDA/HIP attention and reductions are parallelized; further tuning remains.
GPU graph reuse does not clear unused host node gradients.

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
