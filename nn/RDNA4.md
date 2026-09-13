# RDNA4 training optimization and qualification

Measured 2026-09-13 on RX 9070 XT, gfx1201, ROCm Core 10.0 layout,
HIP runtime 7.15 and hipBLASLt 1.4.1 (`hipblasLtGetVersion=100401`).
**The 95% peak target is not achieved. Full batch-16 gradient qualification
also remains unresolved; these throughput results are not training acceptance.**

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

## Integer path and next work

INT8 operand experiments remain a separate qualification task. AMD lists a
higher INT8 matrix rate, but that is not evidence of correct integer training.
Do not port the failed full-model CUDA INT8/INT16-style paths into production.
An INT16-style split into INT8 products would still need scaling/overflow bounds,
gradient-error tests and convergence evidence, with FP32 master state initially.
This change does **not** add a qualified HIP integer training backend.

Priority follow-ups: locate the batch-16 gradient divergence layer by layer;
then fuse im2col with packing, improve weight-pack reuse and attention memory
access, evaluate cached timed Lt algorithm selection, and investigate pipelined
LDS tiles. Preserve numerical gates and report useful versus executed FLOPs
throughout. No long campaign or FukauraOu strength test was run here.
