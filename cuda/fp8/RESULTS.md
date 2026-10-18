# RTX 5060 Ti FP8 results

Measured 2026-10-18 on NVIDIA GeForce RTX 5060 Ti, SM120, 16 GB, driver
615.71.09, cuBLAS 13.4.1, NVRTC 13.2. Clocks were not locked; the initial
SM clock was 2070 MHz and the reported maximum 3105 MHz. A check during
the 8K sweep showed 2070 MHz and 46°C.

cuBLAS E4M3 reaches **135.01 TOp/s at 4096³** and **141.84 TOp/s at 8192³**.
The reused optimized custom E4M3 kernel reaches **66.23** and **68.30 TOp/s**,
respectively. Performance depends on shape: the current FP4/E0M3 kernels
remain faster than these FP8 paths on several rectangular FFN/projection
shapes. These results compare implementations, not hardware precision ceilings.

The subsequent [cuBLAS NVFP4 benchmark](../fp4/RESULTS.md) corrects the old
FP4 support probe and measures **236.82 TOp/s with FP32 output** and
**261.08 TOp/s with BF16 output** at 8192³. FP4 is substantially faster than
FP8 when comparing these library implementations. The 4x legacy-MMA probe
ratio does not apply directly to cuBLAS's scaled FP8 instruction path.

## Commands and validation

From the repository root:

```sh
make -C cuda/fp8 all cublas_fp8_gemm test-unit
make -C cuda/fp8 bench > cuda/fp8/build/benchmark.log 2>&1
make -C cuda/fp8 test > cuda/fp8/build/test.log 2>&1
make -C cuda/fp8 sanitize > cuda/fp8/build/sanitize.log 2>&1
TMPDIR="$PWD/tmp/fp8" ./cuda/fp8/build/bench_fp8 \
  --shape small_m1 --kernel cublas --csv cuda/fp8/build/fp8_m1.csv
```

The full sweep used five warmups, 100 launches per timed round, five rounds,
and rotated kernel order. All reported performance is packed GEMM, with
FP32 accumulation/output and dense `2MNK/time`. Compilation, allocation,
CPU quantization, transpose preparation, uploads, and validation are excluded.
Median/min/max times and precision errors are saved in
`build/fp8_compare.csv`, with command, GPU, versions, and clocks in
`build/fp8_compare.csv.metadata.txt`. The additional single-row cuBLAS
measurement is in `build/fp8_m1.csv` with its own metadata.

Validation passed:

- 31/31 CPU conversion tests, including E5M2 RNE, finite encoding round trips,
  subnormal boundaries, signed zero, and infinities.
- All 32,768 outputs of a 128x256x128 GEMM for six original tensor-core
  kernels, two reused optimized kernels, cuBLAS FP8, and cuBLAS BF16.
- All outputs for optimized kernels at M=3, N=136, K=64, covering tails and
  the padded v7 panel-swizzle grid.
- The standalone cuBLAS BF16-output test, original 128 tile, and odd-sized
  scalar E4M3/E5M2 fallback tests.
- Compute Sanitizer memory checks for all ten comparison paths, optimized
  M/N tails, and the scalar fallback: zero errors. Race checks on all ten
  aligned paths and both optimized tail paths: zero errors and zero warnings.
- All **105 sweep rows plus the single-row cuBLAS follow-up** passed the
  64-output FP64 decoded-input reference check. Maximum sweep arithmetic
  relative L2 was **1.73e-7**, below the `1e-4` threshold. Missing outputs
  are detected by prefilling the result with NaNs before validation.

The repaired original kernels use the correct m16n8k32 fragment mapping;
the 128 tile now writes its full output tile. Earlier loose correctness
checks could pass incorrect results, so pre-fix performance should not be
used as a valid baseline. The old cuBLAS runner's descriptor IDs and opaque
algorithm structure were also incorrect; FP8 works with the installed library.

## 4096³ comparison with FP4 and E0M3

| Implementation | Median ms | Dense TOp/s |
|---|---:|---:|
| Original FP8 E4M3, tile 32 | 10.804230 | 12.72 |
| Original FP8 E4M3, tile 64 | 9.641214 | 14.26 |
| Corrected original FP8 E4M3, tile 128 | 5.069994 | 27.11 |
| Corrected original FP8 E5M2, tile 128 | 5.070439 | 27.11 |
| Reused E4M3 v5, double-buffered `cp.async` | 2.088853 | 65.80 |
| Reused E4M3 v7, plus panel swizzle | 2.075171 | 66.23 |
| cuBLAS E4M3, FP32 output | **1.018022** | **135.01** |
| BF16 cuBLAS, decoded E4M3 inputs | 3.678488 | 37.36 |
| New FP4 E2M1, previous fixed-variant run | 1.594553 | 86.19 |
| E0M3, previous identical-tiling run | 1.594697 | 86.19 |

FP4/E0M3 numbers are from the [FP4 comparison](../fp4/RESULTS.md), on the
same GPU/driver/library with the same timing methodology. cuBLAS FP8 is
**1.57x** the current FP4/E0M3 throughput here and **3.61x** the current BF16
throughput. The custom FP8 v7 kernel is **2.44x** the corrected original
128 tile. No cuBLAS FP4 path was measured in this comparison.

FP8 uses unit per-tensor scales on deterministic uniform `[-2,2]` inputs;
FP4/E0M3 use K16 UE4M3 scales. The random generator's mapping differs between
the FP8 and INT4 runners, so precision samples are not paired across them.
Within this FP8 run, E4M3 output quantization relative L2 is **0.03992** and
E5M2 **0.07703** against the original FP32 inputs. This is separate from
arithmetic correctness against the decoded inputs. BF16 consumes decoded
E4M3 inputs and therefore shares the E4M3 quantization error.

An independent Nsight Systems trace identified the cuBLAS kernel as:

```text
nvjet_sm120_qqsss_mma_128x128x64_6_64x64x64_tmaAB_alignCD4_bz_TNNN
```

The trace is in `build/cublas_fp8_profile.nsys-rep`, with its kernel summary
in `build/profile_kernels.csv`. Profiling timings were not used in the sweep.

## Full shape sweep

All entries are packed-GEMM dense TOp/s. “Original best” takes the fastest
measured corrected tile from 32/64/128 for that shape; individual tile times
remain in the CSV. The large shapes all chose tile 128. There is no automatic
kernel tuning in the FP8 runner.

| Shape | M,N,K | Original best E4M3 | Original best E5M2 | E4M3 v5 | E4M3 v7 | cuBLAS E4M3 | BF16 |
|---|---|---:|---:|---:|---:|---:|---:|
| square_1k | 1024,1024,1024 | 21.17 | 21.16 | 47.66 | 47.87 | 73.06 | 30.74 |
| square_2k | 2048,2048,2048 | 23.89 | 23.89 | 59.94 | 60.23 | 110.96 | 36.27 |
| square_4k | 4096,4096,4096 | 27.11 | 27.11 | 65.80 | 66.23 | 135.01 | 37.36 |
| square_8k | 8192,8192,8192 | 26.11 | 26.11 | 31.52 | 68.30 | 141.84 | 37.76 |
| mm0 | 1024,4608,4608 | 25.59 | 25.55 | 64.61 | 65.03 | 68.96 | 37.11 |
| mm2 | 1024,1152,4608 | 25.87 | 25.88 | 60.90 | 43.13 | 68.88 | 36.77 |
| qkv | 512,4096,4096 | 23.90 | 23.90 | 57.37 | 57.76 | 66.32 | 35.46 |
| ffn_up | 512,11008,4096 | 23.43 | 23.44 | 23.55 | 62.94 | 58.73 | 36.89 |
| ffn_down | 512,4096,11008 | 23.24 | 23.22 | 47.36 | 59.02 | 65.93 | 35.89 |
| small_m1 | 1,4096,4096 | skip | skip | 0.70 | 0.29 | 1.38* | 0.21 |
| small_m16 | 16,4096,4096 | skip | skip | 10.98 | 4.64 | 22.68 | 10.64 |
| small_m64 | 64,4096,4096 | 11.34 | 11.26 | 42.42 | 18.17 | 56.67 | 19.42 |

*Single-row cuBLAS was measured in the separate follow-up CSV. The current
runner attempts the library call for that shape and supports it on this system.
Original PTX shapes without a supported aligned tile print `SKIP`.

Panel swizzling helps the 8K square substantially but hurts several small
or narrow shapes. cuBLAS FP8 is the strongest measured square-GEMM baseline;
the existing custom FP4/E0M3 kernels still win on shapes such as `mm0`
(about 88 TOp/s) and `ffn_up` (about 82 TOp/s). Keep both shape and kernel
selection explicit when using these results.
