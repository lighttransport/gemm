# RDNA4 Ozaki-II FP64 GEMM

This directory implements real FP64 GEMM on RX 9070 XT (`gfx1201`) using
INT8 or E4M3 FP8 wave matrix multiply-accumulate (WMMA) instructions. It uses
Ozaki Scheme II: scale the FP64 inputs, form exact low-precision modular
products, reconstruct with the Chinese remainder theorem, and rescale to
FP64. The numerical pipeline and modulus plans are adapted from the vendored
[RIKEN-RCCS/GEMMul8](vendor/GEMMul8/README.md), commit
`5f3fae8f075eff1f96fc65cd56432fdac4632a16`, distributed under its
[MIT license](vendor/GEMMul8/LICENSE). Only real FP64 GEMM instantiations are
built here. The original source remains available under `vendor/GEMMul8`.

## Build and run

Requirements are a ROCm/HIP toolchain with `gfx1201` support, hipBLASLt,
rocBLAS, MPFR (benchmark reference), Python 3, and gnuplot (plots). The
reported run used ROCm 10.1 / HIP 7.16 and AMD clang 24. The build discovers
the versioned ROCm root via `hipconfig`. Temporary compiler files go under
repo-local `tmp/`.

```sh
make -C rdna4/gemmmul8 -j8
make -C rdna4/gemmmul8 test
make -C rdna4/gemmmul8 bench
python3 rdna4/gemmmul8/benchmark_campaign.py
python3 rdna4/gemmmul8/plot_results.py
```

GPU access is needed for tests and benchmarks; a sandbox may require access to
`/dev/kfd` and the GPU render node. `make tune` measures 32 variants twice on
six shape families. `make profile` builds a separate instrumented reference
under `tmp/gemmmul8/profile`; its `profile_ozaki M N K int8|fp8 MODULI`
reports GEMMul8's internal phase times. Run `./rdna4/gemmmul8/profile_campaign.sh`
to regenerate [phase_timings.csv](phase_timings.csv). This instrumented run
synchronizes and is separate from the headline timings; it excludes the public
API's outer normalization and epilogue.

`bench_ozaki` accepts `--m`, `--n`, `--k`, `--backend int8|fp8`, `--moduli`
(2–20), `--scaling fast|accurate`, `--variant`, `--seed`, `--iters`,
`--warmups`, `--min-sample-ms`, `--workspace-mib`, `--device`, and `--distribution`.
Distributions include `uniform`, `wide`, `cancellation`, `zero`,
`subnormal-input`, `subnormal-output`, and `huge`. The benchmark prints one
CSV row and returns nonzero when its sampled precision criterion fails.
`--quick` on `benchmark_campaign.py` runs a shorter harness check;
`--resume` continues a campaign with the same seed, variants, and timing
settings. Full runs write [results.csv](results.csv); the two-trial tile
measurements are in [tuning.csv](tuning.csv). The separate
[raw_wmma.csv](raw_wmma.csv) measures the selected device-resident modular
product kernels only; it excludes all scaling, CRT, and FP64 epilogue work.

## API

Include [ozaki2.hpp](ozaki2.hpp) and link `libozaki2.so`. `create` returns a
context; `workspace_size` queries caller-owned device workspace; `dgemm`
enqueues work on the supplied HIP stream; `destroy` releases the context.
A context is not safe for concurrent host-thread use. Matrices are device
pointers in row-major layout, with independent N/T operations and leading
strides in elements. Inputs and scalar coefficients must be finite. The API
supports zero dimensions, K=0, alpha=0, and beta=0 without reading C when
beta=0. Matrix buffers must not overlap. It does not silently call native
FP64 GEMM.

```cpp
#include "ozaki2.hpp"
// A: M x K, B: K x N, C: M x N, all on the selected HIP device.
ozaki2::context *ctx = nullptr;
ozaki2::create(&ctx);
ozaki2::options opts;
opts.arithmetic = ozaki2::backend::int8; // or fp8
opts.moduli = 15;                        // FP8 starting point: 11
opts.max_workspace_bytes = size_t(8) << 30;
ozaki2::gemm p{M, N, K, ozaki2::operation::normal,
                ozaki2::operation::normal, 1.0, A, K, B, N, 0.0, C, N};
size_t bytes = 0;
auto status = ozaki2::workspace_size(ctx, p, opts, &bytes);
// Check status, allocate `bytes` on the GPU, then:
status = ozaki2::dgemm(ctx, p, opts, workspace, bytes, stream);
// Check status and synchronize stream before consuming the result.
// Free workspace after stream completion, then ozaki2::destroy(ctx).
```

A nonzero `max_workspace_bytes` caps all temporary storage, including the
normalization buffers. The planner selects M/N output panels when a full
matrix does not fit; every panel retains the full K dimension to avoid summing
rounded FP64 partial products. It returns `insufficient_workspace` when even
the smallest panel cannot fit. A zero cap requests the full-size workspace.
The benchmark sets its cap to the smaller of 8 GiB and 75% of available VRAM.

The API first normalizes each input by an exact power of two, chosen from its
largest magnitude, so subnormal inputs do not disappear during norm
calculation. It rescales the reconstructed output and applies alpha/beta on
the GPU. The default `moduli=0` selects 15 for INT8 or 11 for FP8. Explicit counts
from 2 through 20 are available. The number of moduli controls precision; no fixed count guarantees
FP64 accuracy for every dynamic range or cancellation pattern. FP8 uses the
reference's E4M3 residue plans, not ordinary lossy FP8 quantization.

## Kernels and checks

The INT8 path uses signed `v_wmma_i32_16x16x16_iu8` with INT32 accumulators;
FP8 uses `v_wmma_f32_16x16x16_fp8_fp8` with FP32 accumulators. Exact
accumulation is bounded to K=65,536 for INT8 and K=15,872 for FP8; upstream
modular reduction carries results between K chunks. The HIP integration uses
these custom WMMA kernels for both individual and strided-batched products.

The automatic tile choice for both backends is variant 27 (128×128×32,
eight waves, two LDS buffers). It maximizes the two-trial geometric mean with
equal weight for square, tall, wide, long-K, short-K, and irregular tuning
families. Variant `-1` selects the conservative 64×64 kernel; variants 0–31
select the full 64/128 M/N, 32/64 K, 4/8 wave, 1/2 LDS-buffer sweep. In the
two-buffer variants, the next K tile is prefetched into registers while WMMA
consumes the current LDS tile, then staged to the alternate buffer.

Disassembly of the selected code objects shows both requested WMMA
instructions. Both selected variants use 192 VGPRs and 16 KiB LDS; the
compiler reports two VGPR spills and 12 bytes of private storage per thread.
A rocprofv3 trace of the baseline dispatch identifies a rocBLAS
double-precision kernel; the matching ROCm code object contains `v_fma_f64`
instructions. The standalone library build excludes upstream BLAS
interception hooks.

`make test` checks both backends and scaling modes, all transpose combinations,
nontrivial alpha/beta, padded strides, non-default streams, capped edge panels,
zero coefficients, subnormal and large exponents, exact modular products for
all 32 variants, extreme residues, and both K-chunk boundaries. It tests every INT8 modulus encoder and three
FP8 residue moduli (weighted and congruence plans) against exact modular identities, plus the 11/15/20
modulus CRT trees against GMP/MPFR. It also checks small complete GEMMs,
including transposes and alpha/beta, against MPFR. The benchmark checks every output for at
most 4096 elements and otherwise checks 256 deterministic positions; ULP
statistics and normalized errors are labeled sampled in the CSV.

For each checked element the normalized error denominator is
`|alpha| * sum_k |A[i,k] * B[k,j]| + |beta*C_initial[i,j]|`. Exact-zero
denominators are checked separately. The ordinary-input acceptance threshold
is `max(4 * native_max_error, 32 * 2^-53)`. Wide-range stress cases can exceed
this threshold at 15 INT8 / 11 FP8 moduli; the sweep records those failures
instead of treating them as valid speedups. The `20`-modulus setting improves
those tested cases, at additional cost.
