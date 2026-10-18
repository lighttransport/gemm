# FP4 GEMM comparison on RTX 5060 Ti

Measured 2026-10-18 on SM120, 16 GB, driver 615.71.09, CUDA/NVRTC 13.2,
and cuBLAS 13.4.1. Clocks were not locked; the recorded initial SM clock was
2077 MHz. Throughput is dense `2MNK/time`, counting multiply and add separately.

FP4 here means E2M1 elements with positive UE4M3 scales per K16 block. It is
the `e2m1` backend in the common runner; `--backend fp4` is an accepted alias.
This comparison uses the NVFP4-style block-scaled instruction, rather than the
MXFP4 K32/UE8M0 variant.

## cuBLAS NVFP4 at 8192³: corrected support and absolute performance

Follow-up measured across 2026-10-18/19. **cuBLASLt 13.4.1 supports NVFP4 on
this RTX 5060 Ti.** The original support probe used incorrect transpose
descriptor IDs (`TRANSA=0`, `TRANSB=1`); the correct IDs are `3` and `4`.
The previous unsupported conclusion was invalid. Every descriptor call is
now checked, and context teardown runs before unloading cuBLASLt so its
module callbacks remain valid during cleanup.

```sh
make -C cuda/fp4 cublas-test
make -C cuda/fp4 cublas-sanitize
make -C cuda/fp4 cublas-bench
```

The last command searches up to 32 cuBLASLt heuristic candidates with a
256 MiB workspace allowance, selects the fastest verified candidate using
the median of three short pilot batches, then measures five rounds of 100
launches after five warmups. Six candidates were returned for each 8192³
output type; candidate index 3 won the recorded run, with zero workspace
required by the selected algorithm. These are dense packed-GEMM timings,
excluding allocation, CPU packing, scale swizzling, transfers, reference
calculation, and algorithm search. Output types are measured sequentially.

| Output type, FP32 accumulation | Median ms | Min–max ms | Dense TOp/s |
|---|---:|---:|---:|
| FP32 | 4.642792 | 4.640137–4.780639 | **236.82** |
| BF16 | 4.211420 | 4.210552–4.353297 | **261.08** |
| FP16 | 4.213640 | 4.212209–4.394419 | **260.94** |

Inputs are varying signed E2M1 codes and varying positive UE4M3 K16 scales,
including both row tiles and K tiles of the documented 128x4 scale layout.
Both matrix shapes are row-major `[8192,8192]`; the output is `A * B^T`.
Sixty-four sampled FP64 CPU dot products passed before algorithm selection
and after timing. FP32-output arithmetic error was zero. BF16 and FP16
relative L2 errors were 0.001632 and 0.000190, respectively, reflecting
output rounding. Each candidate starts from an output filled with NaNs.

All outputs also passed at asymmetric 128x256x128 and 256x128x192 shapes
for all three output types. The latter crosses two row tiles and three K
tiles in the scale layout. Compute Sanitizer reported zero memory errors.
The all-ones support smoke test independently reached 261.2 TOp/s with
BF16 output. The `FP4 + VEC32_UE8M0` API configuration returns invalid-value
status 7; the documented cuBLAS FP4 interface uses `VEC16_UE4M3`.

Artifacts are `build/cublas_nvfp4_8k.csv` and its `.metadata.txt` sidecar,
which records the invocation, library version, opaque selected algorithm,
and GPU clocks before and after each output type. The recorded SM clocks
were 2062–2070 MHz, at a configured 150 W power limit. An Nsight Compute
capture of the FP8 comparison independently measured 2.06 GHz.

**400–500 TOp/s was not observed at these clocks.** The contemporaneous
validated instruction probe measured FP4 at **297.07 TOp/s** and E0M3 at
**297.10 TOp/s**. The BF16-output cuBLAS GEMM reaches about 88% of that
FP4 probe rate. A separate legacy MMA probe measured FP4 around **304**
and FP8 around **76 TOp/s**, reproducing the 4x ratio. That ratio does not
apply to the faster cuBLAS FP8 implementation, which reached roughly
142 TOp/s in the [FP8 sweep](../fp8/RESULTS.md).

Nsight Compute SASS for that cuBLAS FP8 kernel contains
`QMMA.SF.16832.F32.E4M3.E4M3.E8`, a different scaled instruction from the
unscaled FP8 instruction used in the legacy probe. The FP8 capture and
decoded SASS are in `build/cublas_fp8_ncu.ncu-rep` and
`build/cublas_fp8_sass.txt`. Profiled timings are excluded from the table.
The separate FP4 capture (`build/cublas_nvfp4_ncu.ncu-rep`) profiles the
first heuristic candidate, a CUTLASS SM120 block-scaled kernel. Its decoded
SASS (`build/cublas_nvfp4_sass.txt`) contains native
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` instructions.
The current FP4/E0M3 custom GEMMs therefore have considerable headroom:
the earlier 8K custom FP4 result was 54.64 TOp/s with FP32 output, versus
236.82 TOp/s for cuBLAS FP4 with the same output precision. This cuBLAS API
does not expose the experimental E0M3 format.

The scale packing follows NVIDIA's
[cuBLAS 128x4 scale layout](https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout),
and the type/setup follows the
[cuBLASLt FP4 requirements](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul).

```sh
# Reproduce the current validated instruction-throughput reference:
TMPDIR="$PWD/tmp/fp4" ./cuda/int4/build/bench_int4 --mode probe \
  --iters 20 --rounds 5 --csv cuda/fp4/build/current_probe.csv

# Original issue probe, including the unscaled FP8 MMA comparison:
make -C cuda/fp4 mma_fp4_probe
TMPDIR="$PWD/tmp/fp4" ./cuda/fp4/mma_fp4_probe --iters 20000
```

## Attack on 300 TOp/s at current clocks

Measured 2026-10-19 without changing GPU clocks or the configured 150 W power
limit. Recorded clocks before/after the final runs were 2070 MHz. These are
dense NVFP4 GEMMs at **8192x8192x8192**, with varying signed E2M1 codes,
varying UE4M3 K16 scales, FP32 accumulation, and BF16 output. Packing,
allocation, algorithm search, transfers, and verification are outside timing.

| Implementation | Median ms | Dense TOp/s |
|---|---:|---:|
| Paired cuBLASLt control | 4.219760 | **260.56** |
| Selected CUTLASS 256x128x128, release build | 3.916203 | **280.76** |
| Experimental warp handoff, release build | 3.906614 | 281.45 |
| Experimental unroll-4 and source-free epilogue | 3.905762 | 281.51 |

The selected path is **7.8% faster** than the paired cuBLAS control. The
experimental differences are below 0.3%, so the default retains unmodified
CUTLASS headers and the original CTA synchronization. **300 TOp/s has not
been reached.** The nominal dense ceiling at 2.07 GHz is
`36 * 4096 * 2.07 / 1000 = 305.23 TOp/s`; the selected GEMM reaches **92.0%**.
300 TOp/s would require 3.665039 ms, versus the observed 3.916203 ms.
The contemporaneous independently validated instruction probe reached
296.79 TOp/s, so the GEMM reaches 94.6% of that probe's measured rate.

The final run checked 64 FP64 CPU dot products and then compared **all
67,108,864 BF16 outputs** against cuBLAS on the identical packed inputs:
**zero bit mismatches**. BF16 relative L2 against sampled unrounded FP64
references was 0.001632, reflecting output rounding. Each candidate starts
from NaN-filled output; the final validation uses a fresh launch after timing.
Five warmups precede five rounds of 100 CUDA-event-timed launches.

The optional CUDA/CUTLASS build uses CUDA 13.2, `sm_120a`, `-O3 -DNDEBUG`,
and external CUTLASS 4.6.1, commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3`. The selected tile has three
mainloop stages and uses the CLC scheduler, raster AlongN (`--raster 2`),
and maximum swizzle 4. The external CUTLASS checkout was left untouched. The runner reuses
the cuBLAS benchmark's packing, references, and timing through a small dynamic
kernel adapter in `cutlass_fp4.cu`.

```sh
# Optional external checkout; the default is repo tmp/flash-attention-src/csrc/cutlass.
make -C cuda/fp4 cutlass-test
make -C cuda/fp4 cutlass-sanitize
make -C cuda/fp4 cutlass-bench

# Full 14-kernel tile/stage/cooperative/ping-pong comparison:
make -C cuda/fp4 cutlass-sweep

# Broader cuBLAS IDs and validated tile/stage/swizzle settings:
make -C cuda/fp4 cublas-tune
```

For another checkout, append `CUTLASS_DIR=/path/to/cutlass` to the make
commands. The standalone cuBLAS tool retains its SDK-free build; only the
optional CUTLASS adapters require nvcc and CUTLASS headers.

Tuning observations:

- Cache traversal matters at 8K: unswizzled direct CUTLASS measured roughly
  90-106 TOp/s; the cache-aware traversal reached 271-272 with diagnostics
  enabled, and 280-281 after disabling internal diagnostics with `-DNDEBUG`.
- Of the 14 release-mode variants, 256x128x128 was fastest. Larger register
  footprints and the 128x256 ping-pong variant produced spills and regressions.
- Explicit cuBLAS search exposed IDs 66/70/71 and 11 valid configurations
  after checking common documented tiles/stages/swizzles. It stayed near
  260-261 TOp/s. Static scheduling did not beat CLC in the diagnostic-build
  comparison (267 versus 271 TOp/s).
- A K64-tile experiment failed the full CPU reference and was excluded.
  The adapter explicitly requires at least two K64 fragments per mainloop
  tile. None of its timings is included in performance claims.

Optional experimental libraries are `build/libcutlass_fp4_warp_release.so`,
`build/libcutlass_fp4_unroll4.so`, and `build/libcutlass_fp4_static.so`.
`prepare_cutlass_fp4.py` generates local header overrides under `build/`,
checks the expected source patterns, and prints the original source hash.
These experiments are never used by `cutlass-bench`.

Recorded artifacts are `build/cutlass_final_8k.csv`,
`build/cublas_control_8k.csv`, `build/tuning_probe.csv`,
`build/cutlass_release_sweep_8k.log`, the traversal/experimental CSVs,
and their `.metadata.txt` sidecars. `cutlass-bench` writes
`build/cutlass_nvfp4_8k.csv` for the reproducible final build.
Profiled timing is excluded from all performance tables.

The final Makefile build reproduced **280.76 TOp/s** (3.916177 ms), again with
zero bit mismatches against cuBLAS. Full CPU references passed at
128x256x128 and 384x128x320, covering both M/N tails and a partial K tile.
Compute Sanitizer reported **zero memory errors and zero race hazards** for
the selected kernel. Logs are `build/cutlass_final_checks.log` and
`build/cublas_tune_regression.log`.
The optional warp-handoff and unroll-4 variants also passed the full
384x128x320 CPU reference under both memcheck and racecheck, with zero
errors or hazards (`build/cutlass_{warp_release,unroll4}_{memcheck,racecheck}.log`).

An Nsight Compute capture of the release kernel measured **2.06 GHz**,
**92.29% tensor-pipeline utilization**, and **53.01% DRAM utilization**.
The remaining gap is therefore not explained by saturating DRAM bandwidth;
further improvement needs work on feeding/scheduling the tensor pipeline and
overlapping its surrounding work. The capture is
`build/cutlass_release_ncu.ncu-rep`, with exported metrics in
`build/cutlass_release_metrics.txt`.

## Reproduction and reused code

```sh
make -C cuda/fp4 bench

# A shorter, FP4-only run, still including the original three kernels:
make -C cuda/fp4 bench \
  FP4_BENCH_ARGS='--shape square_4k --backend fp4 --mode all --variant 8 --legacy-fp4 1'

# Original standalone validators:
make -C cuda/fp4 w4a4
make -C cuda/fp4 fp4_w4a4_opt
```

The benchmark target reuses the `cuda/int4` runner and its build requirements.
All three original kernel source strings from `fp4_w4a4_opt.c` were moved
byte-for-byte into `fp4_gemm_kernels.h`, which both the original standalone
program and the new benchmark adapter include. They compile through NVRTC once,
outside timing. The packed-input layout and K16 scale layout are shared.

| CSV backend | Kernel |
|---|---|
| `fp4_ref` | Original one-warp 16×8×64 reference |
| `fp4_sync` | Original BM=64, BN=128, BK=64 shared-memory kernel |
| `fp4_async` | Original BM=64, BN=128, BK=64 double-buffered `cp.async` kernel |
| `e2m1` | New FP4 kernel, BM=64, BN=128, BK=128, asynchronous staging |
| `e0m3` | Same new kernel and instruction schedule with E0M3 format bits |

Variant 8 fixes the new FP4/E0M3 tiling, avoiding independent tuning choices.
The three original FP4 kernels and the new FP4 kernel consume exactly the same
packed codes and scales, produced from deterministic uniform FP32 `[-2,2]`.
E0M3 independently quantizes the same source inputs into its own codebook.

Each result uses five warmups, 100 timed launches per round, five rounds, and
rotating job order. CUDA events measure device execution. Compilation,
allocation, transfers, reference calculation, and validation are excluded.
Quantization is included only in the explicitly named pipeline modes.

## 4096³ results

| Kernel or pipeline | Median ms | Dense TOp/s |
|---|---:|---:|
| Original FP4 reference | 16.406940 | 8.38 |
| Original FP4 shared memory | 2.492938 | 55.13 |
| Original FP4 `cp.async` | 2.305320 | 59.62 |
| New FP4 packed GEMM | 1.594553 | 86.19 |
| New FP4 activation quantization + GEMM | 2.141103 | 64.19 |
| New FP4 quantization of both operands + GEMM | 2.634858 | 52.16 |
| E0M3 packed GEMM, identical new tiling | 1.594697 | 86.19 |
| BF16 cuBLAS, FP32 output | 3.674839 | 37.40 |

The new FP4 kernel is **1.45×** the original asynchronous kernel on this shape.
FP4 and E0M3 packed GEMM throughput is effectively equal at fixed tiling.
The [FP8 follow-up](../fp8/RESULTS.md) measures cuBLAS E4M3 at 135.01 TOp/s
on this shape, providing a stronger library baseline than BF16.
E0M3 changes the codebook while preserving the instruction's operation count.
BF16 is performance context using
decoded E0M3 inputs; FP4 correctness uses its own decoded-input reference.

The pipeline measurements use the common runner's FP32 quantizer. They do not
include the existing inference runtime's bias, per-channel scale, or low-rank
epilogues. FP4 input quantization relative L2 at this shape is 0.16481, compared
with 0.09919 for E0M3, on 64 output samples against the original FP32 inputs.
Those are synthetic-input accuracy measurements, separate from arithmetic
correctness against decoded packed values.

## Full fixed-tiling sweep

All entries are packed-GEMM dense TOp/s. The new FP4/E0M3 columns use fixed
variant 8 throughout; the original kernels retain their original tiling.

| Shape | M,N,K | Original FP4 ref | Original FP4 sync | Original FP4 async | New FP4 | E0M3 | BF16 |
|---|---|---:|---:|---:|---:|---:|---:|
| square_1k | 1024,1024,1024 | 11.99 | 41.77 | 48.23 | 60.57 | 60.80 | 30.61 |
| square_2k | 2048,2048,2048 | 8.32 | 51.12 | 55.18 | 77.18 | 77.19 | 36.33 |
| square_4k | 4096,4096,4096 | 8.38 | 55.13 | 59.62 | 86.19 | 86.19 | 37.40 |
| square_8k | 8192,8192,8192 | 6.92 | 31.71 | 24.89 | 37.78 | 37.69 | 37.78 |
| mm0 | 1024,4608,4608 | 13.77 | 57.71 | 62.76 | 88.16 | 88.19 | 37.10 |
| mm2 | 1024,1152,4608 | 13.39 | 53.71 | 60.12 | 84.75 | 84.77 | 36.76 |
| qkv | 512,4096,4096 | 8.34 | 48.95 | 52.95 | 75.89 | 75.88 | 35.48 |
| ffn_up | 512,11008,4096 | 8.38 | 53.24 | 57.36 | 82.39 | 82.37 | 36.92 |
| ffn_down | 512,4096,11008 | 13.55 | 52.17 | 57.34 | 82.29 | 82.30 | 35.89 |
| small_m1 | 1,4096,4096 | skip | 0.57 | 0.67 | 1.05 | 1.06 | 0.21 |
| small_m16 | 16,4096,4096 | 7.11 | 8.91 | 10.58 | 16.75 | 16.78 | 10.34 |
| small_m64 | 64,4096,4096 | 7.83 | 32.71 | 39.59 | 62.34 | 62.34 | 25.41 |

Fixed asynchronous variant 8 is not the best choice for every shape. The
[earlier adaptive sweep](../int4/RESULTS.md) selected synchronous BK=128 staging
for the 8K square and measured FP4 at 54.64 TOp/s. Use `--variant auto` for
per-shape throughput tuning, or a fixed variant for an identical-tiling
FP4/E0M3 comparison. Clocks and kernel selection should remain explicit when
comparing results across runs.

## FP4 quantization pipelines

These columns use the common GPU quantizer and the fixed new FP4 kernel.
Activation mode quantizes A with B prepacked; full mode quantizes both operands.
All throughput is dense TOp/s including the named device work.

| Shape | Packed GEMM | Activation + GEMM | Both quantizers + GEMM |
|---|---:|---:|---:|
| square_1k | 60.57 | 34.01 | 23.63 |
| square_2k | 77.18 | 48.97 | 34.76 |
| square_4k | 86.19 | 64.19 | 52.16 |
| square_8k | 37.78 | 34.76 | 32.76 |
| mm0 | 88.16 | 65.09 | 33.27 |
| mm2 | 84.75 | 43.65 | 24.69 |
| qkv | 75.89 | 61.23 | 20.92 |
| ffn_up | 82.39 | 68.59 | 22.31 |
| ffn_down | 82.29 | 57.68 | 21.36 |
| small_m1 | 1.05 | 1.01 | 0.06 |
| small_m16 | 16.75 | 15.40 | 0.97 |
| small_m64 | 62.34 | 51.11 | 3.84 |

## Validation and limitations

The original fragment/scale validator and standalone GEMM validator passed.
The original optimized and asynchronous kernels matched the original reference
bit-for-bit on the aligned smoke test. The common adapter also passed the
CPU FP64 reference checks with logical K=65 and padded K=128. Compute Sanitizer
reported zero memory errors and zero race hazards for all four FP4 kernels.
All 119 comparison rows passed the 64-output CPU FP64 reference checks, with
zero arithmetic relative L2 error on these synthetic packed inputs.

Original dimension requirements are preserved: N must be even, and the
reference also requires M%16=N%8=0. Unsupported cases print explicit skips;
the new FP4 kernel continues to handle odd M/N and logical K tails. These are
GEMM benchmarks, without a claim of hardware peak or model-level quality.

The full comparison writes `build/fp4_compare.csv`, its metadata sidecar, and
local logs. Compiler/disassembler details for the new CUBINs are recorded in
`../int4/build/toolchain.txt`. Temporary compiler files use repository-local
`tmp/fp4/`.
