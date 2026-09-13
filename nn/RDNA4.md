# RDNA4 training optimization and qualification

Measured 2026-09-13 on RX 9070 XT, gfx1201, ROCm Core 10.0 layout,
HIP runtime 7.15 and hipBLASLt 1.4.1 (`hipblasLtGetVersion=100401`).
**The 95% and 75% peak targets are not achieved. The current deterministic
fast-plan path passes full batch-16 and batch-64 gradient qualification, but
does not yet reach the requested 2,000 examples/s.**
The latest iteration is recorded under [Precision follow-up](#precision-follow-up);
the preceding measurements below describe the initial ROCm optimization.

## Implementation

- `hip`: original MIT 32×32×32, four-wave LDS-tiled BF16 WMMA training
  kernel, vectorized packed operand loads and separate high/correction
  accumulators. Inference uses a 64×64 tile and one BF16 product.
- Shared parallel normalization, loss and attention kernels now work on HIP
  as well as CUDA. Attention backward separates scores, Q/K/V and relative-bias
  reductions, removing the old contended atomic scatter and large thread stack.
- HIP bias backward reduces eight channels per CTA. Gradient norm uses bounded
  grid-stride reductions, reducing global-atomic contention.
- `hip-legacy`: retained original kernels for numerical/performance A/B tests.
- `hip-blaslt`: explicit, optional hybrid. hipBLASLt handles packed inference
  GEMMs and training matrices with K≥512 and M×N≥262144. Short-K/small training
  matrices use native WMMA. Both still use the legacy direct kernel if any
  matrix dimension is below 32. No silent CPU fallback.

AMD's public BLASLt interface is **hipBLASLt**, not a separate public
rocBLASLt API. The independently authored adapter caches per-model matrix
descriptors and the first supported result of eight heuristic candidates.
It does not autotune by timing candidates. A shared 64 MiB workspace and two
FP32 correction buffers are charged to the existing device tensor budget.
Library-internal allocations are not included in that budget. Plans are not
serialized across library versions. See the [AMD API reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html).

The six-product precision contract, FP32 parameters/gradients/AdamW state,
checkpoint format, architecture and optimizer are unchanged. No GPU power,
clock or system-driver settings were changed. No GPL source or shogi weights
were introduced. The default build remains free of vendor BLAS and SDK headers;
the explicit Lt build links AMD's external library and its C++ runtime.

## End-to-end measurements

Default C256/20-block model, 22,764,238 learned parameters, synthetic input.
`gn_tool bench` warms up inference ten times and training twice, then includes
forward/backward/clipping/AdamW in training wall time. It excludes initialization,
RTC compilation, initial uploads/heuristic selection and checkpoint I/O. The
baseline used ten measured iterations; final batch-16 runs used thirty, larger
batches twenty. GPU phases were serialized; results are representative runs,
not a statistically controlled sustained-power study.

| Backend | Microbatch | Inference ms/batch | Training examples/s | Useful training matrix TFLOP/s |
|---|---:|---:|---:|---:|
| Original HIP baseline | 16 | 38.7566 | 45.7173 | 0.493824 |
| Tiled `hip` | 16 | 9.51128 | 371.092 | 4.00842 |
| Hybrid `hip-blaslt` | 16 | 8.29118 | 387.378 | 4.18434 |
| Hybrid `hip-blaslt` | 32 | 14.9073 | 422.294 | 4.56148 |
| Hybrid `hip-blaslt` | 64 | 27.7413 | 464.597 | 5.01843 |

At unchanged microbatch 16 the native/hybrid improvements are **8.12×/8.47×**.
Host tensor bytes are 877,219,808 / 1,390,076,640 / 2,415,790,304 for batches
16/32/64, respectively, not total host/device/driver RSS. Larger microbatches
alter BN statistics, so the campaign's microbatch remains 16.

Two real-replay updates, effective batch 256 / microbatch 16, also completed:
native 287.801 then 396.546 examples/s; hybrid 161.996 then 415.632. The hybrid's
first step includes cold algorithm selection. First policy/value losses were
6.98195222 / 4.70390844 (native) and 6.98195094 / 4.70390476 (hybrid). Checkpoints
were saved separately; this short run is not convergence or playing-strength
evidence. Replay came from tinyshogi's own self-play, not a reference engine.

The initial rocprofv3 trace covered warmups plus inference and training. Kernel
time was dominated by old attention backward (2370 ms across 48 calls, about
49 ms/call), GEMM (1170 ms), normalization forward/backward (429/366 ms), loss
(140 ms), bias backward (137 ms) and gradient norm (73 ms). Removing the serial
and atomic-heavy operations mattered more than GEMM tiling alone. An intermediate
tiled trace then exposed bias backward and gradient norm as remaining hot spots;
the final revision parallelizes those too. HIP API synchronization durations
overlap device work and must not be added to kernel durations.

## Peak accounting and matrix measurements

AMD specifies 195 TFLOP/s dense FP16 matrix performance for RX 9070 XT, versus
389 with structured sparsity. AMD's RDNA4 table gives equal FP16/BF16 rates.
Thus the nominal dense BF16 reference here is 195 TFLOP/s and its 95% target is
185.25 TFLOP/s. Actual sustained clock/power behavior can differ from the boost
specification. [RX 9070 XT specifications](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html),
[AMD RDNA4 matrix rates](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/).

Representative training GEMM timings (100 iterations, HIP events, prepared
operands), measured before the final dispatch selection:

| M×N×K / operation | Native 32×32 kernel ms | Native six-product TFLOP/s | hipBLASLt six-product ms | hipBLASLt six-product TFLOP/s |
|---|---:|---:|---:|---:|
| 1296×256×2304 / convolution forward | 0.171915 | 53.3576 | 0.130270 | 70.4149 |
| 256×2304×1296 / weight gradient | 0.172663 | 53.1262 | 0.168852 | 54.3253 |
| 1296×2304×256 / input gradient | 0.136364 | 67.2681 | 0.197130 | 46.5326 |

Those are **six products / kernel time**, excluding padding, not useful network
FLOPs. Packing plus kernel times for the same native cases were
0.222654 / 0.222902 / 0.168603 ms; vendor cases 0.164343 / 0.189841 / 0.211222 ms.
This motivated native short-K dispatch. 32×64, 64×32, 64×64, 64×128 and K64
variants are retained in the benchmark; K64 and larger tiles were not a win for
these shapes. They are not production defaults.

A separate 4096³ diagnostic reached 110.224 TFLOP/s for six vendor products
plus combination, or 127.836 TFLOP/s for a single BF16 vendor GEMM. Neither
reached 185.25. Register-resident WMMA issue probes measured roughly
184–189 TFLOP/s; these perform **no actual GEMM data movement or training** and
are never counted as reaching the training target.

`bench_rdna4` reports four distinct quantities:

1. Useful `2MNK / (pack + kernel time)`.
2. Matrix-product work `2MNK × products / kernel time`.
3. Native executed WMMA work including padded tile instructions. Vendor padding
   is unknown and is deliberately not fabricated.
4. Register-only issue throughput, explicitly labeled as not GEMM/training.

The JSON kernel 95% flag uses quantity 2, **not padding or the issue probe**.
The peak denominator is an optional command argument, defaulting to 195 for this
card. Exit status gates correctness, not peak attainment. GPU tests/benchmarks
need device access; a sandbox-unavailable device is not a successful test.

The full model has 10,801,677,312 useful matrix FLOPs/training example. Six-product
compensation consumes roughly six hardware GEMMs for one useful GEMM. Even a
perfect compensated GEMM therefore cannot deliver 95% of the single-product
peak as *useful* throughput: its matrix-only bound is about 195/6 = 32.5 TFLOP/s,
before packing, padding, reductions, optimizer and launches. Reaching the target
requires distinguishing hardware issue utilization from useful model throughput;
we do not silently change precision or count redundant work as useful MFU.

## Correctness and outstanding qualification

Passed: CPU analytic gradients/overfit/checkpoint resume; independent PyTorch
C4/C32 oracle; SDK-free and optional-Lt Make builds; CMake/CTest 14/14 in both
configurations; CUDA C32 regression; HIP C32 FP32 and BF16; HIP and hybrid full
C256 **batch-2** gradients/AdamW/checkpoint reload. Native full batch-2 global
gradient relative L2 is 0.0000153225, below the unchanged 0.001 gate. Hybrid
batch-2 training dispatches native matrices, so this does not qualify its
large-batch vendor branch.

`test_rdna4.sh` passed all four transpose combinations with 35×67×49 tails,
the three training shapes above, repeated accumulation and finite-output checks.
The compensated sampled GEMM errors versus CPU double dot products were below
2e-5; the 4096³ case also passed. This samples 256 output dots and checks every
output for finiteness/accumulation, not every output against a CPU GEMM.
No AMD race-detector qualification is claimed.

**New batch-16 full-model test fails the existing 0.001 gradient gate:**

| Path | Global gradient relative L2 |
|---|---:|
| Original `hip-legacy` | 0.00289522934 |
| Optimized `hip` | 0.00289523382 |
| Hybrid `hip-blaslt` | 0.00383229931 |
| `hip-fp32` diagnostic | 0.006368942 |

Losses agree closely, but that is not sufficient. The native discrepancy is
already present in the retained original path; FP32 also fails. Reduction order
and ReLU boundary sensitivity are hypotheses, not a proven diagnosis. The test
is retained and the tolerance is not loosened. Larger-batch throughput above is
therefore **unqualified**. The parent campaign now runs the actual full ROCm
microbatch preflight and will reject a full campaign on this fixture until the
discrepancy is resolved. Small smoke tests remain available. The earlier
correlated-input stress discrepancy in SM120.md is also still open.

## Reproduction

From GEMM root, SDK-free native runtime:

```sh
make -C nn check
python3 -B nn/compile_gpu.py cuda hip
nn/build/test_gpu hip nn/build/hip-wide.safetensors wide
nn/build/test_gpu hip nn/build/hip-full.safetensors full
make -C nn rdna4-bench
sh nn/test_rdna4.sh nn/build/bench_rdna4
```

Optional Lt build (public ROCm development headers and libraries required):

```sh
make -C nn BUILD=build-lt HIPBLASLT=1 ROCM_PATH=/opt/rocm/core all rdna4-bench
nn/build-lt/gn_tool bench MODEL.safetensors hip-blaslt 16 30
nn/build-lt/test_gpu hip-blaslt nn/build-lt/full-b16.safetensors full 16
sh nn/test_rdna4.sh nn/build-lt/bench_rdna4
nn/build-lt/bench_rdna4 4096 4096 4096 0 1 5 195
```

The batch-16 command is an expected unresolved failure, not an optional test to
skip before deployment. `HIPBLASLT_INCLUDE` and `HIPBLASLT_LIB` override the SDK
paths. This host lacked Lt development headers: three MIT public headers from
AMD's `ROCm/rocm-libraries` **rocm-7.2.0** tag were downloaded to the parent's
ignored `build/dl/hipblaslt-headers/hipblaslt`, with build-local generated export
and version headers. No system SDK files were installed or modified. Header
`hipblaslt.h` SHA256: `86cd3ecf7c3f9c88ecc1184338b9009eed5a9f6dcf0668d7be7dadef5d3cf5dc`.
The runtime reports its actual version separately. Prefer a matching complete
SDK for deployment; this older-header/newer-runtime pairing is the one tested.

## Precision follow-up

### Changes and arithmetic contracts

The normal `hip` and `hip-blaslt` paths now fuse forward im2col directly into
BF16 packing. This removes the FP32 column write/read and one launch while
preserving all three packed components bit-for-bit. Default 81-token/32-wide
attention stages coalesced K/V reads in LDS, retaining the original FP32 dot,
softmax and output reduction order. Other head sizes use the previous kernel.

Explicit experimental training backends (also usable for diagnostic inference):

| Backend | Matrix operands | Architectural accumulator |
|---|---|---|
| `hip-bf16` | One BF16 product | FP32 |
| `hip-bf16-acc` | One BF16 product | Native BF16 C/D for the whole dot |
| `hip-bf16-acc128` | One BF16 product | Native BF16 per K=128, then FP32 partial reduction |
| `hip-int8` | Row-scaled signed INT8 | INT32, with automatic INT64 widening for overflow-length K |
| `hip-int8-i64` | Row-scaled signed INT8 | INT32 partials reduced in INT64 |
| `hip-int16` | Row-scaled INT16, four INT8 products | Four INT32 partials, exact INT64 recombination |

AMD's [RDNA4 ISA, WMMA instruction table](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture)
specifies native BF16 C/D and INT8-to-INT32 operations. It does not list an INT16
WMMA or INT64 WMMA accumulator. The generated gfx1201 assembly was inspected:
the experiments issue `v_wmma_bf16_16x16x16_bf16` and
`v_wmma_i32_16x16x16_iu8`, respectively. INT64 widening is software integer
arithmetic; it is not attributed to an invented native instruction.

INT16 uses `q = 256*hi + lo`, with signed high and unsigned low bytes. Accumulators
are widened **before** weighting or adding cross terms. Chunks of at most 16384
products keep even `255*255*K` below INT32_MAX. Signed INT8 uses an uninterrupted
INT32 dot only when Kpad32 ≤131040; longer dots automatically use INT64 partial
reduction. Integer dot products never reduce through FP32: only final conversion,
row scales and output storage are FP32. Quantization is round-to-nearest-even,
with a symmetric training range ±127 or ±32767. Exact tests additionally cover
the full integer endpoints -128 and -32768. Transposed operands use a coalesced,
bit-preserving FP32 scratch transpose before row scaling; the scratch allocation
is reused and counted in the device budget. A measured 64×32 INT16 tile replaces
the first 32×32 tile; the benchmark retains alternatives for reproduction.

All modes retain FP32 master weights, stored activations/gradients, AdamW state,
scales/dequantization, and non-matrix operations. **None is integer-only or
BF16-only training.** The default six-product precision contract is unchanged;
experimental arithmetic is never selected automatically by the normal backend.

### Measured normal-path improvement

Same full model and microbatch 16, three alternating before/after repetitions,
50 measured steps each. Previous commit `4d982b9dd9bd4cae84f3c447e711db81cc92151f`'s GPU host source and embedded
kernels were rebuilt with the current reporting-only benchmark front end.
Median results (no concurrent GPU tests; clocks/power settings unchanged):

| Backend | Before examples/s | After examples/s | Gain | Before inference ms | After inference ms |
|---|---:|---:|---:|---:|---:|
| `hip` | 369.981 | 383.233 | 3.58% | 9.56240 | 8.18069 |
| `hip-blaslt` | 385.503 | 400.245 | 3.82% | 8.30386 | 6.93760 |

The after training ranges were 382.860–384.450 and 399.512–401.999 examples/s.
These are bounded measurements on this host, not sustained-performance promises.
The normal hybrid median is 4.32332 useful matrix TFLOP/s including attention.
Its convolution/linear work is 4.29105 useful TFLOP/s; the six compensation
products total 25.7463 TFLOP/s over whole-step wall time: **13.2032% of 195**.
Useful GEMM arithmetic alone is **2.20054%** of that dense reference. Compensation,
padding, packing, optimizer and other scalar work must not be conflated.

### Measured experimental whole-step throughput

Full model, microbatch 16, 50 measured steps per mode after warmup. These are
single bounded runs, with no CPU reference job or other GPU benchmark running
concurrently. All use the same architecture/optimizer; reduced precision changes
the mathematical trajectory. **Faster unqualified arithmetic is not a trained
model quality or convergence result.**

| Backend | Examples/s | Useful GEMM Tera-op/s | Matrix-product Tera-op/s | Products / matching dense peak |
|---|---:|---:|---:|---:|
| `hip-bf16` | 598.068 | 6.41193 floating | 6.41193 floating | 3.28817% |
| `hip-bf16-acc` | 592.145 | 6.34842 floating | 6.34842 floating | 3.25560% |
| `hip-bf16-acc128` | 584.832 | 6.27001 floating | 6.27001 floating | 3.21539% |
| `hip-int8` | 521.243 | 5.58827 integer | 5.58827 INT8 | 1.43657% |
| `hip-int8-i64` | 518.739 | 5.56143 integer | 5.56143 INT8 | 1.42967% |
| `hip-int16` | 442.432 | 4.74334 INT16-equivalent | 18.9734 INT8 | 4.87747% |

The denominator is **entire forward/backward/update wall time**, not just matrix
kernel duration. FP32 attention is reported separately (0.0357–0.0482 TFLOP/s here).
INT64 scalar widening and packing are included in elapsed time, not credited as
INT8 WMMA operations. Host tensor accounting is 877,219,808 bytes for each mode;
this is not total RSS or driver memory. The first INT16 implementation measured
387.918 examples/s before tile/transpose/attention improvements, versus 442.432
after; those two experimental runs used 30/50 iterations and are not the paired
normal-path comparison above.

### Matrix-only rates versus peak

Final gfx1201 diagnostic, 100 iterations for the three training shapes. Integer
rows below are INT8 instruction-product TIOP/s, including four products for
INT16; BF16 rows are TFLOP/s. Percentages use 389 / 195 dense respectively.

| M,N,K | Mode | Kernel ms | Pack+kernel ms | Product Tera-op/s | Product peak % | Padded issue peak % |
|---|---|---:|---:|---:|---:|---:|
| 1296,256,2304 | BF16 accumulator | 0.0568460 | 0.0815090 | 26.8941 | 13.7919 | 14.3027 |
| 1296,256,2304 | INT8/INT32 | 0.0304462 | 0.0804854 | 50.2140 | 12.9085 | 13.3866 |
| 1296,256,2304 | INT16/INT64, 64×32 | 0.0725887 | 0.125465 | 84.2458 | 21.6570 | 22.4591 |
| 256,2304,1296 | INT8/INT32 | 0.0231978 | 0.105531 | 65.9037 | 16.9418 | 17.1510 |
| 256,2304,1296 | INT16/INT64, 64×32 | 0.0583140 | 0.141745 | 104.868 | 26.9584 | 27.2913 |
| 1296,2304,256 | INT8/INT32 | 0.0356931 | 0.0659414 | 42.8325 | 11.0109 | 11.4187 |
| 1296,2304,256 | INT16/INT64, 64×32 | 0.0708859 | 0.0991201 | 86.2695 | 22.1773 | 22.9986 |

The first INT16 32×32 convolution kernel took 0.106732 ms; the selected 64×32
tile takes 0.0725887 ms (1.47× kernel speedup). For weight gradients, initial
pack+kernel was 0.257673 ms versus 0.141745 ms after tiling/coalesced transpose.
Enforced device-function inlining avoids an observed out-of-line specialization
regression; alternative tile measurements are retained rather than inferred.

For 4096³ (30 iterations), BF16 accumulation is 44.7612 TFLOP/s / 22.9545%; INT8
is 71.3804 TIOP/s / 18.3497%; INT16's four INT8 products are 90.5634 TIOP/s /
23.2811%. These shapes have no padding. The optional Lt comparison (10 iterations)
is 125.649 TFLOP/s / **64.4353%** for one BF16 product, and 105.559 TFLOP/s /
54.1328% for six-product compensation including the combine kernel. Vendor
internal padding is unknown and is not fabricated. None reaches 95%; the old
register-only issue ceiling is not GEMM or end-to-end training performance.

Ignored parent build artifacts contain the raw JSONL: `rdna4-precision-final-matrix`,
`rdna4-precision-final-large`, `rdna4-lt-final-large`, and
`rdna4-precision-final-training` under `build/dl/`. Before/after normal runs use
`rdna4-{before,after}-{hip,hip-blaslt}-{1,2,3}.json`. Raw files are local experiment
artifacts, not installed runtime dependencies.

### Precision and correctness results

The mandatory numerical gates were not relaxed. A new `report` test argument
continues diagnostics after an approximation mismatch but **still exits 1**;
normal campaign preflight does not use it or expose these experimental backends.
Report-mode checkpoint files are unqualified diagnostic artifacts.

| Mode | C32 batch-2 inference relative L2 | C32 gradient relative L2 |
|---|---:|---:|
| `hip-bf16` | 0.00852518 | 0.0746677 (fail) |
| `hip-bf16-acc` | 0.211776 | 0.394561 (fail) |
| `hip-bf16-acc128` | 0.116112 | 0.331739 (fail) |
| `hip-int8` / `hip-int8-i64` | 0.0272495 | 0.159798 (fail) |
| `hip-int16` | 0.000113825 | 0.000190985 (pass) |

INT16's small-network pass does **not** extend to the full C256/20-block model:
batch-2 inference relative L2 is 0.0000888861 but gradient relative L2 is
**0.0612931 (fail)**. Normal full-model batch-2 HIP remains a pass with gradient
relative L2 0.0000153225, identical to the previous path. The existing full-model
batch-16 discrepancy remains unresolved; the final native rerun is still
0.00289523382 against the unchanged 0.001 gradient gate. No experimental full-training, long
convergence or shogi playing-strength qualification is claimed.

On convolution GEMM M=1296,N=256,K=2304, sampled output relative L2 against CPU
double dots is 0.00222831 (BF16 operands/FP32 accumulator), **0.506040** (native
BF16 accumulator), **0.0526036** (BF16 partials K128), 0.00570492 (INT8), and
0.0000228791 (INT16). Pure BF16 accumulation is unsuitable for the current
precision contract; less FP32 does not automatically mean better throughput or
acceptable gradients. On 4096³, BF16 accumulator error rises to 0.722769.

The exact-accumulator suite covers all integer outputs for small/medium K,
sampled outputs at K=131073, both signs/endpoints, mixed signs, K=16385 crossing
the partial boundary, and signed INT8 overflow-length widening. BF16 tests use
exactly representable products through K=256 and verify both accumulator modes,
tails and residual-add behavior. Fused columns match separate expansion/packing
bit-for-bit across C=3/8/32, 3×3/5×5 kernels, one/three BF16 components. Matrix
sweeps check all four transposes, dimension tails, quantization against the CPU,
exact integer sums before dequantization, finite outputs and add semantics.

CPU gradient/overfit/checkpoint tests, C4/C32 PyTorch oracles, CTest 14/14,
CUDA C32 regression, and NVRTC/HIPRTC compilation pass. No new AMD GPU race-tool
qualification is claimed.

### Reproduce the follow-up

```sh
make -C nn check rdna4-precision
sh nn/test_rdna4_precision.sh nn/build/bench_rdna4_precision
nn/build/bench_rdna4_precision 4096 4096 4096 0 1 30 195 389
nn/build/gn_tool bench MODEL.safetensors hip-int16 16 50 195 389
nn/build/test_gpu hip-int16 nn/build/int16-wide.safetensors wide 2 report
nn/build/test_gpu hip-int16 nn/build/int16-full.safetensors full 2 report # expected failure
```

`gn_tool bench` separates useful GEMM work, product work and FP32 attention,
using whole-step wall time. Integer rates are TIOP/s, not mislabeled TFLOP/s.
The precision benchmark additionally separates matrix-only and pack-plus-matrix
event timing, exact padded WMMA issue counts and product/issued percentages.
Its correctness exit status is separate from approximation/training qualification.
Peaks default explicitly to RX 9070 XT dense BF16 195 TFLOP/s and INT8 389 TOP/s,
and can be overridden by arguments. INT16's four INT8 products are compared to
the INT8 instruction peak; this is **not a native INT16 peak claim**. Sparse
rates are not used. See [AMD's RX 9070 XT specifications](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html).

Next work: diagnose full-model gradient divergence layer by layer, improve
weight-pack reuse, pack backward im2col directly, time/cache Lt algorithms, and
pipeline larger LDS tiles. **95% peak is still not achieved.**

## BF16/FP32 follow-up

The current focus is BF16 operands with **FP32 matrix accumulation**, FP32
master weights/gradients/Adam state, and existing FP32/FP64 nonmatrix math.
The revised targets are 75% nominal dense matrix peak or 1,000 full-model
training examples/s. The latter is reached **as timing only** by the
single-product experiment; no new full-training qualification is claimed.
BF16 accumulation and integer experiments remain available, but are not the
default or promoted paths.

### Implemented changes

- Pack backward convolution operands directly from NHWC into transposed BF16
  tiles, eliminating the FP32 im2col write/read. Both forward/backward packing
  specialize default C256/3x3 and C80/5x5 indexing; other dimensions retain a
  generic path. Packing is bit-identical to separate expansion/conversion.
- Coalesce BN forward/backward over eight neighboring NHWC channels per CTA.
  Statistics and reduction sums remain double; normalization/state remain FP32.
  LayerNorm and CUDA dispatch are unchanged.
- Add `hip-bf16-blaslt` for single-product BF16/FP32 training; add
  `hip-bf16x3` and `hip-bf16x3-blaslt` for two-component, three-product
  compensation (`Ahi*Bhi + Alo*Bhi + Ahi*Blo`, separate FP32 high/correction
  accumulators). These use compensation in inference as well as training.
- Add `hip-bf16-mixed` and `hip-bf16-mixed-blaslt`: six products forward,
  including inference, three products backward. This protects forward/ReLU
  decisions at lower cost than six products for all three matrix operations.
  All new arithmetic variants are explicit, experimental backends; none is
  enabled by campaign preflight or the engine.
- Add per-node value/gradient snapshots and ReLU sign-difference counts to
  `test_gpu ... report`. These transfers occur only in the diagnostic test,
  never normal training or performance timing. Numerical gates are unchanged.
- Add a benchmark-only, 32-candidate hipBLASLt event-timing search. A separate
  workspace destination preserves the caller's C during beta=1 trials; the
  shape cache reuses the same algorithm for zero/nonzero beta, avoiding
  additional reduction-order drift in add checks. Model execution deliberately uses
  deterministic first-supported heuristics: timed selection varied between
  model creation/reload and failed exact checkpoint inference. No timings or
  backend choices are silently serialized into model weights. The adapter
  explicitly resolves HIP event entry points to avoid collisions with ROCEW's
  legacy exported function-pointer names. Library calls follow the
  [public hipBLASLt API](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html).

The Lt workspace is caller-budgeted: 64 MiB library workspace plus an aligned
FP32 output-sized scratch region for the benchmark tuning contract. Plans own
no extra application tensor allocations. The SDK-free build stays SDK-free.

### Measured full-model throughput

RX 9070 XT gfx1201, the same ROCm/HIP/Lt versions documented above, unchanged
device settings. C256/20 blocks, 22,764,238 learned parameters, synthetic
resident benchmark input. Timings include forward, backward, gradient clipping,
and AdamW, but exclude RTC compilation, initialization, initial uploads,
checkpoint I/O and replay decoding. These are bounded steady-state tests,
on a shared host without locked clocks, not a sustained dataset-training or
playing-strength result.

Three alternating before/after runs of 100 measured training steps at batch 16:

| Standard six-product hybrid | Before | After |
|---|---:|---:|
| Median examples/s | 398.975 | **439.431** |
| Range examples/s | 398.406–400.248 | 439.083–441.174 |
| Inference ms (median across runs) | 6.99530 | 5.74940 |

The normal path gains **10.14%** training throughput without changing its
six-product training arithmetic. Whole-step useful GEMM rate is 4.71117
TFLOP/s (2.41598% of dense peak), compensation-product rate 28.2670 TFLOP/s
(14.4959%), and separate FP32 attention 0.0354276 TFLOP/s.

Matched `rocprofv3` runs of `hip-bf16 16 20` locate the gains outside GEMM.
Across 52 forward passes (warmups, inference and training), forward column
packing drops 86.194→42.371 ms and total BN/LN forward 58.649→41.387 ms.
Across 22 backward passes, BN/LN backward drops 40.565→27.006 ms. Backward
FP32 columns (35.886 ms) disappear, replaced by 16.576 ms direct BF16 packing,
also removing 726 ordinary packing calls. GEMM kernel time is approximately
unchanged (200.285→201.764 ms). These are aggregate profiler kernel times,
not extra unprofiled speed measurements or sums including overlapping HIP API time.

For the new variants, three runs of 100 measured steps each at **batch 64**:

| Backend | Median examples/s | Range | Useful matrix TFLOP/s including attention | BF16 product TFLOP/s | Product % dense peak |
|---|---:|---:|---:|---:|---:|
| `hip-bf16-blaslt` | **1133.28** | 1132.31–1133.97 | 12.2413 | 12.1499 | **6.23073%** |
| `hip-bf16x3-blaslt` | 782.170 | 781.347–783.306 | 8.44874 | 25.1570 | 12.9011% |
| `hip-bf16-mixed-blaslt` | 703.266 | 702.055–703.351 | 7.59645 | 30.1590 | 15.4662% |

After the final shape-cache correction, 100-step spot checks were 1139.33
examples/s for single-product and 703.869 for mixed; exact C32 mixed reload
also passed. The table retains the three-run medians, not these faster spots.

Host tensor allocation is 2,415,790,304 bytes at batch 64, versus 877,219,808
at batch 16. Host/device tensor budgets remain separately capped at 6 GiB;
these numbers are not total process/driver memory. Batch changes also change
BN statistics: this sweep does **not** change campaign defaults.

Single bounded batch-16 runs measured 772.626 / 584.703 / 529.010 examples/s
for one-product / three-product / mixed Lt respectively. Batch 32 single-
product reached 976.067 examples/s. Increasing batch amortizes overhead;
it does not imply the arithmetic or accuracy is unchanged.

Rates use 2 operations per FMA and no padded-operation inflation. Useful
conv/linear work is 10,721,055,744 FLOPs/example; FP32 attention contributes
80,621,568. Product multipliers are 1, 3, and **4 averaged over the full
forward/backward step** for the three experiments. A higher compensated
product percentage is not more useful model work. The denominator remains
the 195 TFLOP/s **dense** 16-bit matrix reference, not the sparse 389 rate;
75% is 146.25 TFLOP/s. See [AMD product specifications](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html)
and the [BF16/FP16 RDNA4 throughput table](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/).

The separate tuned one-product Lt diagnostic at 4096 cubed measured
104.204 TFLOP/s (53.4377%, 1.31895 ms matrix only), or 87.0666 TFLOP/s including
packing. At 8192 cubed it was slower: 82.0221 TFLOP/s (42.0626%, 13.4051 ms),
74.9772 including packing. Sampled FP32-reference relative errors were
0.0021058 / 0.00200313. Tuning did not beat the previous 125.649 TFLOP/s
4096-cubed result; neither the current nor prior result reaches 75%.
Register-only issue diagnostics are still **not GEMM or training throughput**.

### Error and qualification

All errors below are dimensionless global gradient relative L2 against the
independent CPU FP32 implementation. The unchanged acceptance gate is 0.001.
Accuracy cases below are **batch 16 or 2, not the batch-64 timing run**.

| Full C256/20 backend | Batch | Gradient relative L2 | Result |
|---|---:|---:|---|
| Native standard six products | 16 | 0.0028952338 | FAIL, unchanged from before |
| `hip-bf16-blaslt` | 16 | 0.38367618 | FAIL |
| `hip-bf16x3-blaslt` | 16 | 0.013713891 | FAIL |
| `hip-bf16-mixed-blaslt` | 16 | **0.0038323371** | FAIL |
| `hip-bf16-mixed-blaslt` | 2 | **0.000024398431** | PASS, including update/reload |

Keeping six-product forward passes cuts batch-16 error about **100x** versus
one-product training and **3.58x** versus all-three-product training. It is
close to the earlier six-product hybrid error 0.00383229931, but still misses
the gate. C32 batch-2 three-product WMMA passes with gradient error
0.00003532031 (one-product previously 0.0746677). This is a precision/speed
tradeoff, not an accuracy-approved replacement for the default.

Node diagnostics provide evidence for nonsmooth amplification: on the native
six-product full batch-16 case, ReLU node 144 has two CPU/GPU sign differences.
Its output-gradient relative error is 0.000011932171, while its input (node
143) gradient error is 0.00010006454. Node 140 has another two differences;
additional single differences occur at 10, 53, 71, 93, 104 and 133. This
locates error amplification at ReLU boundaries; it does not prove all
remaining differences harmless or justify relaxing the gate. No activation
function, tolerance, or campaign acceptance rule was changed.

CPU finite-difference/overfit/exact resume, C4/C32 PyTorch oracles, both RTC
compilers, CTest 14/14 in both builds, six offline DL tooling tests, CUDA C32
hardware regression, fused forward/backward packing and integer/BF16 exact
accumulator checks were exercised. Packing tests now cover side 3/9,
C=3/8/32/80/256, kernels 3/5, all one/two/three-component layouts, and tails.
No full batch-64 accuracy, AMD race-tool, long convergence or strength
qualification is claimed. The 75% target remains unmet, and 1,000 examples/s
is a **rate-only**, not a qualified-training, success.

### Reproduce

```sh
# Opt-in build and runtime setup are described earlier in this report.
nn/build-lt/gn_tool bench MODEL.safetensors hip-bf16-blaslt 64 100 195 389
nn/build-lt/gn_tool bench MODEL.safetensors hip-bf16-mixed-blaslt 64 100 195 389
nn/build-lt/test_gpu hip-bf16-mixed-blaslt nn/build/mixed-b2.safetensors full 2 report
nn/build-lt/test_gpu hip-bf16-mixed-blaslt nn/build/mixed-b16.safetensors full 16 report # fails
nn/build-lt/bench_rdna4 4096 4096 4096 0 1 100 195 hipblaslt_bf16
sh nn/test_rdna4.sh nn/build-lt/bench_rdna4
sh nn/test_rdna4_precision.sh nn/build/bench_rdna4_precision
```

The final benchmark mode argument is optional and exact-name checked. JSON
now separates `75pct_product_rate_met`, `1000_examples_per_second_rate_met`,
and `qualified_target_met` (false here). Raw paired/final observations in the
embedding tinyshogi checkout are `build/dl/bf16fp32-{before,after}-pair-*.json`,
`bf16fp32-final-*.json`, and the corresponding `bf16fp32-*.log` diagnostics.

## Smooth-activation qualification and FP16 comparison

Configuration version 2 replaces configurable ReLUs with SiLU and records
that graph-semantic choice in the checkpoint. Version 1 remains loadable and
unchanged. On the full C256/20 version-2 model, batch 16, the three-product
`hip-bf16x3-blaslt` backend passed the unchanged global parameter-gradient
relative-L2 gate: **0.00041490896 <= 0.001**. Output relative L2 was
0.000019643069 and update/reload also passed.

At batch 64, three 100-step runs measured **807.838 examples/s median**
(806.399--809.994), 8.72600 useful matrix TFLOP/s and 25.9826 BF16 product
TFLOP/s at the median. The latter is **13.3244%** of the RX 9070 XT
195-TFLOP/s nominal dense reference. Neither the 1,000 examples/s nor
75%-of-peak target is met by the qualified path.

An explicit one-product `hip-fp16-blaslt` experiment uses FP16 operands and
FP32 matrix accumulation, master weights, gradients and optimizer state. It
measured **1,148.55 examples/s** at batch 64 (12.4063 useful TFLOP/s,
**6.31471%** of nominal peak), but it is not qualified: the full version-2
batch-16 global gradient error was **0.0241998443**. Its finer mantissa and
faster rate therefore do not make it a safe substitute for compensated BF16.

Two routing experiments were rejected. Sending the large K=256 convolution
backward matrices to three rocBLASLt calls reduced whole-step throughput to
760.583 examples/s despite a favorable isolated GEMM result. Restoring the
one-query-per-CTA attention forward measured 805.210 examples/s, slightly
below the grouped-forward 810.623 observation, while retaining the same
0.00041490896 gradient result. These are short observations, not sustained-
performance guarantees.

## Qualified 1,000 examples/s hybrid

The final `hip-bf16x3-fp16back-blaslt` allocation uses three-product BF16/FP32
for forward, single-product FP16/FP32 for convolution backward and parameter
gradients, and three-product BF16/FP32 for the less numerous linear dX paths.
Fusing residual-add/SiLU and SiLU/gate-multiply pairs removes intermediate
pointwise launches and gradient-memory passes without changing graph values.

The full version-2 C256/20 model passes the unchanged independent CPU gate at
the same batch used for performance:

| Batch | Output relative L2 | Global gradient relative L2 | Gate/result |
|---:|---:|---:|---|
| 16 | 0.000019643069 | 0.00083100988 | <=0.001, PASS |
| 64 | 0.000019664309 | **0.00082061858** | <=0.001, PASS |

Both runs include AdamW comparison and bit-exact same-backend checkpoint
reload. The batch-64 alignment diagnostic leaves 0.00082060973 residual error
after optimal scalar alignment, showing that the pass is not due to a hidden
global rescale.

Three final 100-step batch-64 runs measured **1,025.39 examples/s median**
(1,023.25--1,026.55), meeting the requested 1,000 examples/s target in every
run. Median useful matrix throughput was 11.0759 TFLOP/s. Accounting for the
actual per-node product allocation—not a uniform multiplier—gives 19.2103
16-bit product TFLOP/s, **9.85142%** of the nominal 195-TFLOP/s dense peak.
Thus the qualified rate target passes; the alternative 75%-of-peak target does
not. Benchmark JSON leaves `training_qualification` unresolved because a
generic timing command cannot infer that an arbitrary checkpoint passed a
separate CPU-oracle run; qualification is established by the paired test above.

## 2,000 examples/s follow-up baseline

Batch scaling does not provide the requested second doubling: batch 128
measured 1,037.07 examples/s versus the batch-64 qualified median of 1,025.39,
while host tensors grew from 2.416 to 4.467 GB. The path is compute/memory
limited rather than launch-amortization limited.

A `rocprofv3` trace of the qualified hybrid attributes 15.99% of aggregate
device time to the leading rocBLASLt kernel family, 11.83% to attention
forward, 10.31% to BF16 im2col packing, 5.16%/3.93% to BN forward/backward,
3.91% to normalization, 3.69% to FP16 backward im2col, and 3.25% to attention
score backward. Percentages cover benchmark warmups, inference, and training;
they identify targets but are not per-step additive timing claims.

A native gfx1201 FP16 WMMA implementation was tested and rejected. Routing all
FP16 matrices natively measured 945.210 examples/s; routing only short/small
shapes measured 981.941 examples/s. Both regress against rocBLASLt, so the
experimental kernel and dispatcher were removed. Reaching 2,000 examples/s on
the unchanged model now requires eliminating major data-motion passes—most
plausibly direct convolution fused with operand preparation and normalization—
rather than substituting the vendor GEMM path or increasing batch size.

### Normalization/activation fusion

The version-2 residual tower now fuses convolution bias, batch normalization,
and SiLU forward, plus SiLU derivative and batch-normalization backward. The
biased convolution and normalized intermediate values remain materialized, so
backward and diagnostic graph semantics are unchanged; the standalone bias and
activation launches and intermediate activation-gradient pass are removed.

The exact timed batch-64 configuration still passes the independent CPU oracle:
output relative L2 0.000019664309 and global parameter-gradient relative L2
**0.00082243384** (gate 0.001), including AdamW and exact checkpoint reload.
Three 100-step runs measured **1,040.70 examples/s median**
(1,037.93--1,041.43), up 1.49% from the prior 1,025.39 median. Median useful
matrix throughput is 11.2414 TFLOP/s; actual product throughput is 19.4972
TFLOP/s, **9.99857%** of the nominal dense peak. Packed-activation caching,
rocBLASLt combine/bias fusion, and a materialized fused BN gradient were also
tested but were neutral or slower and are not retained.

The backward fusion now also accumulates each preceding convolution's bias
gradient while producing the BN input gradient. This removes the separate
channel-reduction launch and its full gradient-tensor read without changing the
stored FP32 gradient. The batch-64 CPU-oracle result remains output relative L2
0.000019664309 and global gradient relative L2 **0.00082243384**. Three 100-step
runs measured **1,044.73 examples/s median** (1,044.00--1,046.56), with 11.2848
useful TFLOP/s and 19.5725 product TFLOP/s, or **10.0372%** of the 195-TFLOP/s
nominal peak. A direct C256 3x3 WMMA convolution passed the batch-16 gradient
gate but regressed to 759.910 examples/s; directly feeding rocBLASLt's two
compensation outputs into BN was accurate but neutral at 1,039.64 examples/s.
Neither experiment is retained.

### Backward replay and multi-tensor optimizer

The qualified hybrid records its stable backward DAG as a HIP graph after one
warm step. Target and label uploads remain outside the graph, while replayed
kernels read the current device tensors and updated parameter buffers. Graphs
are invalidated when the batch changes. The report-mode GPU oracle executes two
backward passes so capture and replay, rather than only graph construction, are
covered by the numerical check.

Gradient-norm and AdamW processing previously issued 200 launches apiece per
step. Persistent device descriptor tables now combine each phase into one
multi-tensor kernel. The Adam kernel retains the original per-element FP32
arithmetic; only independent tensor/block scheduling and norm reduction order
change. Batch-64 replay passes with output relative L2 0.000019664309 and global
gradient relative L2 **0.0008224335**. Three 100-step runs measure **1,076.80
examples/s median** (1,076.29--1,077.49), 11.6312 useful TFLOP/s and 20.1734
product TFLOP/s, or **10.3453%** of the nominal 195-TFLOP/s peak.

A 16-query attention CTA was accurate but neutral at 1,076.83 examples/s median
and is not retained. Dropping one of the three compensated BF16 forward
products was rejected: output relative L2 rose to 0.004446 and gradient relative
L2 to 0.135437.

### Fused WMMA attention

The hybrid's 81-token, 32-wide attention now computes QK, probability-times-V,
the backward score product, and the Q/K/V gradient triplet with gfx1201 WMMA.
Each matrix product uses the same high-high, low-high, and high-low BF16
decomposition as the qualified convolution path. Relative-position bias,
softmax, and the softmax Jacobian retain serial FP32/double evaluation. Other
HIP and CPU/CUDA backends retain their original attention kernels.

The batch-64 replay oracle passes with output relative L2 **0.000019973022** and
global gradient relative L2 **0.00082430711**. Three 100-step runs measure
**1,150.29 examples/s median** (1,146.69--1,150.81), with 12.4250 useful matrix
TFLOP/s and 21.8284 logical product TFLOP/s, **11.1941%** of the nominal
195-TFLOP/s dense peak. Product accounting includes the nine compensated
attention matrix products and excludes the 81-to-96 padding. A profile measured
the forward attention kernel at 72.31 ms aggregate versus 129.48 ms for the
previous scalar kernel over the same 128 dispatches.

Packing two 16-query tiles into each 384-thread CTA further halves shared K/V
loading and reduces the launch grid from six to three query groups per head.
The arithmetic and oracle result are unchanged. Three batch-64 runs measure
**1,170.75 examples/s median** (1,167.60--1,172.12), 12.6461 useful TFLOP/s and
22.2167 logical product TFLOP/s, **11.3932%** of nominal peak. Median inference
latency is 20.6297 ms across the same runs.

### Vectorized convolution packing

The C256 forward BF16 im2col packer now converts four adjacent channels per
thread and stores aligned four-element high/residual vectors. The FP16 dW
packer likewise loads four adjacent channels into its 32x32 transpose tile and
emits four transposed values per thread. C256 and all spatial dimensions are
compile-time constants; other shapes retain the scalar generic kernels. These
are layout-only transformations and do not change any floating-point operation.

Batch-64 qualification remains output relative L2 **0.000019973022** and global
gradient relative L2 **0.00082430711**. Three 100-step runs measure **1,240.43
examples/s median** (1,238.24--1,240.56), 13.3987 useful matrix TFLOP/s and
23.5389 logical product TFLOP/s, **12.0712%** of nominal peak. Median inference
latency is 18.8272 ms.

The matching C256 uncolumns kernel accumulates four adjacent channels per
thread with the original nine-tap FP32 order. Generic non-transposed BF16 and
FP16 matrix packing also bypasses LDS: each HIP thread directly converts and
stores one aligned four-element vector, while transposed and non-multiple-of-
four inputs retain safe handling. The batch-64 oracle remains bit-for-purpose
unchanged at 0.00082430711 gradient relative L2. Three 100-step runs measure
**1,263.98 examples/s median** (1,262.98--1,266.40), 13.6532 useful TFLOP/s and
23.9860 logical product TFLOP/s, **12.3005%** of nominal peak. Median inference
latency is 18.6126 ms.

### Deterministic hipBLASLt fast plans

The `hip-bf16x3-fp16back-blaslt-fast` backend selects measured heuristic
ordinals for the batch-64 C256/20 matrix shapes on RX 9070 XT. Unlike the
benchmark-only `-tuned` search, the mapping is deterministic across model
reloads. Unmatched shapes use the first supported heuristic, and SDK versions
other than hipBLASLt 1.4.1 (`100401`) disable the mapping rather than assuming
that heuristic ordering is stable.

Independent CPU-oracle runs pass at batch 16 (output relative L2
0.000019969416, gradient 0.00081994291) and batch 64 (output 0.000019948652,
gradient 0.00082694669). Both include AdamW comparison and bit-exact backend
checkpoint reload. Three fresh 100-step batch-64 runs measure **1,280.16
examples/s median** (1,277.41--1,280.71), a 1.28% improvement over the prior
1,263.98/s median. Median useful throughput is 13.8279 TFLOP/s and logical
product throughput is 24.2930 TFLOP/s, **12.4579%** of the nominal 195-TFLOP/s
peak. Median inference latency is 18.5466 ms. The 2,000 examples/s and 75%
peak targets remain open.

Three final batch-64 runs of 100 measured steps sustain **1,028.09 examples/s
median** (1,026.46--1,029.80). Median useful matrix work is 11.1050 TFLOP/s.
The precision allocation executes an estimated 18.3703 trillion 16-bit matrix
product operations/s, **9.42065%** of the 195-TFLOP/s nominal dense reference.
Thus the requested qualified 1,000 examples/s alternative is met; the separate
75%-of-peak target is not. Product accounting uses the 5/3 average multiplier
for three forward products and the mixed backward allocation and excludes
padding. Benchmark JSON deliberately leaves qualification `unresolved`
because a timing-only command cannot prove arbitrary checkpoint accuracy; the
paired `test_gpu` result above is the qualification evidence.
