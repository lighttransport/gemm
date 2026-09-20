# Native A64FX results

Measured 2026-09-19 with Fujitsu `fcc` 4.12.2 on CPUs 12--23 of one 2.0 GHz
A64FX CMG.  All end-to-end numbers below use `--sync atomic`; they are controls,
not headline hardware-barrier results.

## W8A16 / W8A32: 2026-09-20

The signed INT8 paths meet the **220--230 GB/s** target with both FP16 and
FP32 activations/accumulation. **Neither FP8 format is accepted at that target.**
All six implementations are numerically validated; performance acceptance
must remain separate from correctness.

| weight format | activation / accumulation | launch 1 median | launch 2 median | launch 3 median | >=220 GB/s |
|:--------------|:--------------------------|----------------:|----------------:|----------------:|:-----------|
| signed INT8 | FP16 / FP16 | 229.04 | 230.30 | 229.17 | PASS |
| signed INT8 | FP32 / FP32 | 229.37 | 229.37 | 228.67 | PASS |
| E4M3FN | FP16 / FP16 | 143.20 | 143.19 | 143.18 | FAIL |
| E4M3FN | FP32 / FP32 | 92.25 | 92.24 | 92.24 | FAIL |
| E5M2 | FP16 / FP16 | 116.89 | 116.88 | 116.87 | FAIL |
| E5M2 | FP32 / FP32 | 200.52 | 200.46 | 200.44 | FAIL |

Rates count stored one-byte weights, not widened traffic. Three fresh
launches per combination use 240 MiB, CPUs 12--23, ten iterations, and five
timed trials. Every CPU reports 2.0 GHz; every allocation reports 2 MiB pages
on NUMA node 4, and FPCR is zero. Paired-read medians span
227.63--229.77 GB/s; all eighteen runs qualify. No placement failure explains
the FP8 misses. Raw logs, source/binary hashes, and CPU settings:
`tmp/dequant/w8-acceptance.gFdxIs/`.

The benchmark samples all finite byte codes, including FP8 subnormals, using
a deterministic PRNG; NaN/infinity codes are replaced only in the timed
weight distribution. Activations are `(k % 7 - 3) / 32`. Special weight codes
are checked exhaustively in correctness tests. The optional
`--normal-weights` diagnostic additionally excludes subnormal/zero weights;
it is explicitly labeled and cannot qualify for the all-finite gate.
With the original native pipelined E5M2 implementation that diagnostic measured
203.33 GB/s FP16 and 201.01 GB/s FP32, compared with 16.13/201.09 on all-finite
inputs in the adjacent development sweep. This isolates a large
subnormal-sensitive FP16 cost; it is consistent with hardware arithmetic
assists, but no assist PMU counter was collected. Neither diagnostic is a
220 GB/s result. FPCR is never changed and no values are flushed or clipped.
The selected guarded-rescaling kernel now raises all-finite E5M2/FP16 to
116.87--116.89 GB/s without changing the FMA result. Earlier native acceptance
logs are retained at `tmp/dequant/w8-acceptance.mwKtHH/`.

### Kernel contract and selected schedules

`w8.h` defines K=128 and one shared activation vector. Weights are ordinary
INT8/E4M3FN/E5M2 bytes in `[K][N]` order, where N=256 for FP16 and N=128 for
FP32. Each kernel retains eight output vectors and sequential ascending-K
FMA order. A canonical `[N][K]` model tensor must be transposed into this
explicit tile layout; no type conversion is performed during that packing.
Output has the same width as activation. Scales, tails, and model integration
are outside this probe.

INT8 uses signed widening loads and exact `SCVTF`. E5M2 shifts the byte into
IEEE FP16 bits, with `FCVT` for the FP32 path. INT8, E5M2/FP32, and the
native E5M2/FP16 fallback use a two-K pipeline:
prepare K=0, retire two steps in each of 63 iterations, then retire K=126
and drain K=127 without another input load. No accumulator is reassociated.

The selected E5M2/FP16 path avoids feeding subnormal weights into FMA by
using `w*4` and `a/4` for those lanes. The three nonzero subnormal magnitudes
become normal FP16 values through integer table corrections. Each finite
nonzero activation must have FP16 exponent >=3; dividing by four then stays
normal and is exactly representable. The real product is unchanged, and a
single half FMA performs the same rounding as before. Normal-weight lanes
keep the original activation. Zero, infinity, and NaN behavior is retained.
A whole-tile activation check and a nondefault-FPCR check select the unchanged
native fallback when the preconditions fail. Scalar/native comparisons cover
both sides of the activation guard (0x0bff and 0x0c00), signed zero, infinities,
and tiny activations. This improves the default distribution substantially,
but unsafe tiny activations still encounter the native subnormal cost.

E4M3FN/FP16 uses signed widening and an integer bit transform with a wrapped
`(code+1)&127` correction-table index. One table corrects all seven nonzero
subnormals, zero, and NaNs without a separate sign-selection branch.
E4M3FN/FP32 builds the exact BF16 high/low bytes in two independent 64-byte
streams, then interleaves zero low halfwords to obtain FP32. All E4M3FN
finite values are exactly representable in BF16; this step adds no rounding.
These conversions remain the performance bottleneck, rather than HBM.

The verifier isolates all 256 codes, including E5M2 infinities, and compares
sequential scalar half-FMADD / FP32 `fmaf` against each SVE implementation.
Random finite streams, fractional values, subnormals, cancellation, half
accumulation overflow, and a wider FP32 activation range are also covered.
Finite results must match bits; NaNs must match classification (payloads
are not promised). The existing exhaustive W4/INT16 checks still pass.

### Development measurements and rejected schedules

All figures are five-trial medians with qualified paired reads:

| schedule | E4M3 FP16 | E4M3 FP32 | E5M2 FP32 |
|:---------|----------:|----------:|----------:|
| initial separate sign/subnormal/NaN decoder | 118.61 | 59.26 | 165.09 |
| grouped E5M2 shifts before conversion | -- | -- | 173.46 |
| wrapped E4M3 correction; cross-K E5M2 pipeline | **143.62** | 72.04 | **201.15** |
| serial byte tables and BF16 assembly | 121.03 | 84.51 | unchanged |
| two independent byte-table streams | 126.99 | **92.09** | unchanged |

The selected assembly combines the faster wrapped halfword FP16 decoder
with the parallel byte-wise FP32 decoder. Disassembly shows no inner-loop
spills. The halfword E4M3 schedule needs 64 vector arithmetic instructions
per 256 input bytes; byte-wise FP32 needs 76 (including permutations and
FMAs), well above the earlier measured 48-instruction full-bandwidth budget.
Those counts explain why simple scheduling alone is unlikely to close the
remaining gap, without establishing a universal lower bound on conversion.
The E5M2 FP32 pipeline has 48 vector arithmetic instructions per 256 bytes,
but includes half-to-single conversions rather than the previous pure-FMA
roofline; its measured ~201 GB/s must not be replaced with that roofline.

Guarded E5M2 FP16 rescaling first measured 116.69 GB/s with paired reads
227.86/228.29 GB/s; the final three launches above confirm that gain.
Logs: `tmp/dequant/w8-rescale-*`; the pre-rescaling assembly is retained at
`tmp/dequant/w8-before-rescale.S`.

Development logs/snapshots are under `tmp/dequant/w8-v1-*`, `w8-v2-normal-*`,
`w8-v3b-*`, `w8-v4b-*`, and `w8-v5-*`. The `w8-v3-*` runs accidentally used
the previous binary after an assembly error and are **not candidate results**;
the corrected build/test/run chain uses fail-fast shell execution. Rejected
assembly snapshots are `tmp/dequant/w8-v{1,2,3b,4b,5}.S`.

Reproduce:

```sh
mkdir -p tmp/dequant
TMPDIR="$PWD/tmp/dequant" make -C a64fx/dequant-pipe test CC=fcc
bash a64fx/dequant-pipe/run_w8_acceptance.sh
# Expected current result / exit status 1:
# acceptance=FAIL qualified=18/18 measured=18 failed_targets=12 threshold_GB/s=220
```

The acceptance script retains every launch, checks all-finite distribution
and both read controls, and fails if any of the eighteen FMA medians is below
220 GB/s. Remaining work is a lower-cost exact E4M3 conversion, further E5M2
FP32 conversion scheduling, and reducing the guarded E5M2
FP16 rescaling overhead while retaining its exact fallback. Do not promote FP8 or hide those cases to pass the gate.

## FP16 above 200 GB/s: 2026-09-20

The new `--path fp16 --kernel f16pipe` passes the raised **>200 GB/s**
packed-weight target for both formats in all three fresh launches. The
original `opt`, `opt2`, and `super` kernels remain available unchanged.

| format | arithmetic / selector | launch 1 median | launch 2 median | launch 3 median |
|:-------|:----------------------|----------------:|----------------:|----------------:|
| INT4 | sequential FP16, `fp16/f16pipe` | 204.07 | 204.13 | 204.11 |
| FP4 | sequential FP16, `fp16/f16pipe` | 204.02 | 204.07 | 204.03 |
| INT4 | full-range SDOT, `int16x8-full/opt` | 219.64 | 218.03 | 218.93 |
| FP4 | full-range SDOT, `int16x8-full/opt` | 214.27 | 214.13 | 215.37 |
| INT4 | native INT16 SDOT, `int16/opt` | 201.66 | 201.73 | 201.82 |
| FP4 | native INT16 SDOT, `int16/opt` | 173.13 | 173.24 | 173.33 |

Each launch uses 240 MiB packed weights, twelve cores (CPUs 12--23), ten
iterations and five timed trials. All eighteen runs qualify: paired reads
before/after on the same allocation range from 227.96 to 229.77 GB/s.
Every mapping reports 2048 KiB pages on NUMA node 4; all twelve CPUs report
2,000,000 kHz in `host.txt`. No slow-placement run occurred in this series.
Fresh `opt2` controls measured 171.75 GB/s INT4 and 171.96 GB/s FP4, so the
pipeline improves this baseline by approximately 18.8% and 18.6%.

The prologue prepares K=0. Each of 63 loop iterations retires two K steps;
a final transition retires K=126 and a load-free drain retires K=127.
Next-step packed loads precede extraction; current-step FMA pairs alternate
with next-step TBL pairs. Each activation register is reloaded after its two
consumers, before the next K step. The eight output chains retain ascending-K
half-precision FMA order. There are no partial sums, activation quantization,
clipping, changed layouts, or wider accumulators. E2M1 still uses the doubled
lattice; model scales remain omitted.

Disassembly confirms 48 SVE arithmetic instructions per 256 packed bytes
(16 extraction, 16 TBL, 16 FMLA), 16 weight/activation loads, ten pointer
increments, and loop control. There are no loop spills or indexed FMAs;
only the ABI's d8--d15 saves/restores use the stack. This is the same arithmetic
count as `opt2`: separating lookup producers from consumers improves the
schedule. At 204 GB/s the effective budget is about 30.1 cycles per 256 bytes
per core, versus about 35.8 at 172 GB/s. The remaining gap to paired reads
is consistent with mixed instruction/dependency overhead; these measurements
do not isolate a particular execution port as the limit.

Controlled development candidates (five-trial medians, same launch settings):

| schedule | INT4 GB/s | FP4 GB/s | paired-read medians GB/s |
|:---------|----------:|---------:|:------------------------|
| cross-K, all next-step lookups at end | 198.21 | 198.28 | 228.44--229.63 |
| above, activation offsets with updates every two K | 186.76 | 186.77 | 227.44--229.58 |
| alternating current FMA / next TBL pairs, selected | 204.03 | 203.94 | 228.26--229.35 |

The offset schedule reduces pointer instructions but regresses throughput;
retain the per-step updates. Full 64-byte loads/layout changes were unnecessary
after the existing-layout pipeline passed the target. Development logs are
`tmp/dequant/pipe-v{1,2,3}-{int4,fp4}.log`; rejected assembly snapshots are
`tmp/dequant/kernels-pipe-v{1,2}.S`. Fresh baseline controls are
`tmp/dequant/baseline-opt2-{int4,fp4}.log`.

Reproduce correctness and the complete acceptance procedure:

```sh
mkdir -p tmp/dequant
TMPDIR="$PWD/tmp/dequant" make -C a64fx/dequant-pipe test CC=fcc
bash a64fx/dequant-pipe/run_w4a16_acceptance.sh
# acceptance=PASS qualified=18/18 SDOT=6 FP16=6 failed_targets=0
```

The test compares the new pipeline and both old table kernels bit-for-bit
with scalar half FMADD and the original kernel across fractional inputs,
signed zero, subnormals, cancellation, and overflow. The exhaustive INT16
and dequant regression checks pass. Compiler output is warning-free.
Acceptance logs: `tmp/dequant/w4a16-acceptance.s5EThb/`.
The script now requires **>200 GB/s for both FP16 and full-range SDOT**;
the older FP16 >150 GB/s acceptance below is historical.

## W4A16 acceptance: 2026-09-20

Both requested targets pass on CPUs 12--23 of one A64FX CMG. Every number
below is packed-weight GB/s, with metadata reads and correction included in
elapsed time. Three fresh launches each used 240 MiB of packed weights,
ten iterations, five timed trials, and read controls before and after the
kernel on the same allocation. Observed paired reads were 227.74--229.71
GB/s; all mappings reported 2048 KiB pages and NUMA node 4.

| format | arithmetic / selector | launch 1 median | launch 2 median | launch 3 median |
|:-------|:----------------------|----------------:|----------------:|----------------:|
| INT4 | full-range SDOT, `int16x8-full/opt` | 219.44 | 219.94 | 220.09 |
| FP4 | full-range SDOT, `int16x8-full/opt` | 214.23 | 215.45 | 214.70 |
| INT4 | sequential FP16 FMA, `fp16/opt2` | 171.84 | 172.04 | 172.00 |
| FP4 | sequential FP16 FMA, `fp16/opt2` | 172.18 | 172.15 | 172.08 |
| INT4 | native INT16 SDOT, `int16/opt` | 201.47 | 201.58 | 201.46 |
| FP4 | native INT16 SDOT, `int16/opt` | 173.21 | 173.36 | 173.26 |

On the same allocations, the original `super` kernels measured
171.07--171.10 GB/s for native INT4/INT16, 154.87--155.03 for FP4/INT16,
96.20--96.21 for INT4/FP16, and 84.49--84.56 for FP4/FP16.
The one-K direct FP16 table schedule measured about 169.6 GB/s;
the two-K schedule above is selected.

The full-range route uses signed digits satisfying
`a = lo + 256*hi + 128` for every INT16 input. Each 8192-byte supertile has
a 256-byte INT16 weight-sum trailer (3.125% metadata). INT32 output includes
`128*sum(weights)`. Packing 256 activation values takes approximately
392 ns and is reported separately; static weight sums are prepared outside
timing. No activation clipping is used.

Correctness covers all 65,536 INT16 values, random weights, all constant
nibble codes with alternating activation extremes, native INT16 results,
and bit-exact FP16 comparison with scalar half FMA and the original kernel.
Fractional values, signed zero, subnormals, cancellation, and overflow are
included. The normal exhaustive module tests also pass.

Commands:

```sh
TMPDIR="$PWD/tmp/dequant" make -C a64fx/dequant-pipe test CC=fcc
bash a64fx/dequant-pipe/run_w4a16_acceptance.sh
# acceptance=PASS qualified=18/18 SDOT=6 FP16=6 failed_targets=0
```

Raw logs for this run are in
`tmp/dequant/w4a16-acceptance.ByB7iv/`. The script recreates the complete
procedure, including `taskset -c 12` before XOS is loaded. The earlier
119.92 GB/s radix-256 result was placement-confounded: a pinned rerun of the
unchanged bounded kernel reached 219.99 GB/s (INT4) and 202.00 GB/s (FP4).
It was not evidence that INT8 SDOT saturated at 120 GB/s.

These remain unscaled M=1 compute probes, not end-to-end model rates.
FP16 preserves its original half-precision accumulation semantics.

## Earlier outcomes

- Isolated 256-byte-cache-line HBM reads scale from **188.59 GB/s** with four
  cores to **229.96 GB/s median / 230.00 GB/s best** with eight cores.  This is
  95.8% of the requested approximate 240 GB/s target (89.8% of the nominal
  256 GB/s CMG read interface).
- An explicit L1 prefetch eight lines ahead reduced the eight-core rate to
  **180.69 GB/s**, so the no-prefetch stream is selected.

- Correctness: exhaustive dequant and all INT8/INT16/FP16 kernels pass.  Small
  end-to-end scalar checks pass exactly for integer paths and within the stated
  FP16 tolerance.
- M=6 L2-load+SDOT steady state: **1.8394 SDOT/cycle, 91.97%** of the
  two-instruction/cycle architectural peak.  The requested 96% was not reached.
- Full 192 MiB packed-HBM INT4 W8A8 stream, M=1, 64 KiB handoff: median
  **64.28 GB/s**, best **64.29 GB/s** compressed input.  The requested
  240 GB/s was not reached.
- The chunk sweep selected **64 KiB** as the best tested atomic-control
  handoff.  Smaller chunks expose synchronization cost; 256 KiB loses
  producer/consumer overlap.
- Hardware-barrier headline result: **not measured**.  The `xos_hwb` kernel
  module is loaded and registered as character major 241, but this interactive
  allocation has no `/dev/xos_hwb` device node.  The benchmark now hard-fails
  that mode before allocation and does not relabel an atomic result.
- A new direct-HBM fused INT4 kernel eliminates the expanded L2 ring and passes
  scalar correctness for both arithmetic widths.  After scheduling and layout
  tuning, 12 cores sustain **138.35 GB/s for INT8 SDOT** and **67.55 GB/s for
  INT16 SDOT**.  These are 2.16x and 1.93x the staged-pipeline controls,
  respectively, but neither
  preserves the 229.96 GB/s read-only bandwidth.  The remaining limit is the
  in-register nibble expansion plus SDOT instruction schedule.
- One allocation of the four-block K-major supertile with a 16 KiB inter-core
  gap reached **228.84 GB/s median / 228.94 GB/s best** with 12 cores. The paired
  read-only measurement was 228.31 GB/s median, so this configuration meets
  the no-bandwidth-degradation target within measurement noise. Controlled
  sweeps show this cannot be attributed to the 16 KiB gap alone.

## No-degradation INT8 result

### Arithmetic budget at full HBM bandwidth

A fixed-instruction roofline probe uses the same four SVE loads per 256-byte
line and adds independent SVE SDOT or FMLA instructions. Each point measures a
read-only baseline on the same XOS 2 MiB hugepage arena. With twelve cores and
a roughly 229.5 GB/s paired baseline:

| instructions / 256 B line | SDOT GB/s | percent of paired read |
|---------------------------:|----------:|-----------------------:|
| 48 | 229.51 | 99.96% |
| 52 | 224.25 | 97.68% |
| 56 | 216.44 | 94.32% |
| 60 | 201.71 | 87.88% |
| 64 | 190.39 | 83.00% |

FMLA has the same valid 52/60/64 curve within measurement noise; separate
48-instruction FMLA measurements also retained full bandwidth. Thus the safe
measured budget is **48 SVE arithmetic instructions per 256 packed bytes**.
At 229.5 GB/s and 2.0 GHz, each core receives one line every 26.8 cycles, so
this is about 1.79 arithmetic instructions/cycle. The two-pipe theoretical
limit is about 53.5 instructions/line, but 52 instructions already reduce HBM
bandwidth by 2.3%; production kernels also need dequantization, address, and
control instructions within the same deadline.

Transient runs where the paired read itself fell to about 120 GB/s were
discarded for this threshold. `--paired-baseline` exposes that state directly
and should always be used for compute-budget measurements.

Command:

```sh
a64fx/dequant-pipe/bench_fused_sdot --kernel super --cores 12 \
  --mib 240 --skew-kib 16 --iterations 10 --trials 5
```

The kernel consumes a four-block, K-major 16 KiB supertile as a single
sequential stream. This run inserted a 16 KiB gap between the twelve core-local
shards. After one warm-up, the five trials
were 228.59, 228.84, 228.90, 228.62, and 228.94 GB/s.

| configuration | packed GB/s |
|:--------------|------------:|
| 8-core HBM read-only baseline, median | 228.31 |
| 12-core fused supertile, 16 KiB skew, median | **228.84** |
| 12-core fused supertile, 16 KiB skew, best | **228.94** |
| 12-core fused supertile, zero skew | 120.05 |
| 12-core fused supertile, 32 KiB skew | 121.21 |
| 10-core fused supertile, zero skew | 206.84 |

These rows came from separate allocations and are not a controlled skew
sweep. The A64FX manual explicitly documents a hashed physical L2 index and
`PA[8]` L2-bank interleave, but does not disclose the HBM channel/bank mapping;
see [A64FX memory addressing](../doc/a64fx_memory_addressing.md).  It is not
ordinary alignment or false sharing: every core already owns a disjoint,
aligned region.  With zero skew the twelve 20 MiB regions have
identical relative offsets and advance in lockstep, apparently aliasing an
unfavorable mapping. The 16 KiB gap changes virtual offsets while preserving
supertile alignment. However, `bench_hbm_color` later observed both
all-fast (~229 GB/s) and all-slow (~120 GB/s) allocations while varying the gap
within each allocation. Physical page placement is therefore the stronger
variable, and 16 KiB is not a decoded channel color. The exact responsible
level was not isolated: it may involve L2 indexing, MIB/MAC
scheduling, hardware prefetch, an undocumented HBM mapping, or several of
them.
Single-core supertile throughput is 22.35 GB/s versus 11.13 GB/s for the
four-stream layout.  PMU sampling over equal work reduced cycles from 1.235B
to 0.788B and L2 refills from 32.1M to 19.1M.

Two rejected alternatives remain selectable for reproducibility: `--kernel
lut` reaches 97.49 GB/s on 12 cores, and `--kernel pipe` reaches 109.04 GB/s.
Neither replaces the default shift kernel or the selected `super` kernel.

## Direct fused INT4 + SDOT

Measured on the same 2.0 GHz CMG with a 240 MiB packed stream.  The INT8 kernel
fuses four adjacent K=128 blocks and the INT16 kernel fuses two, giving each 16
independent accumulator vectors.  Expanded weights are never written to memory.

| arithmetic | cores | median packed GB/s | best packed GB/s | versus staged |
|:-----------|------:|--------------------:|-----------------:|--------------:|
| INT8 SDOT, consecutive nibbles | 12 | **138.35** | 138.39 | 2.16x |
| INT16 SDOT, split nibbles | 12 | **67.55** | 67.56 | 1.93x |

The INT8 result is 60.2% of the 229.96 GB/s read-only baseline; INT16 is 29.4%.
Thus direct fusion fixes the ring-traffic bottleneck, but does not yet meet the
no-bandwidth-degradation target.  Scaling measurements (77.98 GB/s at eight
cores and 93.43 GB/s at ten for INT8) also show that this schedule remains
arithmetic-pipe/latency limited before saturating HBM.

For INT8, pairing both packed-vector expansions within each block improved the
12-core median from 136.85 to 138.35 GB/s.  A split-nibble INT8 layout removed
the ZIP instructions but regressed to about 109 GB/s and was rejected.  The
same layout benefits INT16 because it also eliminates permutations before four
`sunpk` operations; its stable 10-iteration result improved from 62.04 to 67.55
GB/s.  Pairing both INT16 halves more aggressively regressed and was rejected.

## Fused W4A16: INT4/FP4 with INT16 SDOT or FP16 FMA

The fused benchmark now includes the missing W4A16 paths. The INT16 route uses
a two-block K-major supertile and keeps 16 independent INT64 accumulator
vectors. The FP16 route uses a four-block N-lane supertile: each K scalar's 32
packed bytes expand into two FP16 output vectors and feed FMA immediately.
Neither route writes expanded weights.

The scalar verifier passes for all four combinations: signed INT4 and E2M1
FP4, each with INT16 SDOT and FP16 FMA. E2M1 is decoded as its exact doubled
integer lattice, with the x0.5 factor reserved for the scale epilogue. The
FP16 correctness input uses exactly representable values so it checks layout,
nibble mapping, conversion, and FMA without conflating expected FP16 rounding.

Controlled 12-core results use the same 240 MiB packed stream and Fujitsu XOS
2 MiB hugepage configuration as the placement study. The paired INT4 W4A8
supertile rerun reached 230.09 GB/s median, confirming the allocation was in
the fast HBM state:

| packed format | A arithmetic | median packed GB/s | best | percent of 230.09 |
|:--------------|:-------------|-------------------:|-----:|------------------:|
| INT4 | INT8 SDOT (W4A8 control) | 230.09 | 230.41 | 100.0% |
| INT4 | INT16 SDOT, original schedule | **147.00** | 147.02 | 63.9% |
| E2M1 FP4 | INT16 SDOT | **132.44** | 132.49 | 57.6% |
| INT4 | FP16 FMA | **96.28** | 96.30 | 41.8% |
| E2M1 FP4 | FP16 FMA | **84.59** | 84.64 | 36.8% |

The INT4 INT16 supertile improves substantially over the original four-stream
kernel: 147.00 GB/s controlled versus the earlier 67.55 GB/s result. On an
ordinary heap allocation the same new kernel plateaued near 120.9 GB/s; a
1/4/6/8/10/12-core sweep scaled 13.24, 49.55, 73.83, 98.32, 117.94, and
119.29 GB/s, identifying the familiar slow-placement ceiling. XOS huge pages
raised the result to 147 GB/s, but not to 230 GB/s.

A subsequent two-block schedule issues all four cache-line loads before
unpacking and alternates block-0/block-1 `sunpk` and SDOT work. Expanding the
unpack window from four to six independent results raised the controlled
INT4 result to **171.03 GB/s median / 171.04 best**, a 16.3% gain over 147.00
GB/s. That schedule was the starting point for the accepted kernels above.

An alternative radix-256 kernel represents activation values in
`[-32768, 32639]` exactly as two signed INT8 digits and computes
`dot(lo) + 256*dot(hi)`.
Its initial INT4 result was **119.92 GB/s median / 120.06 best**, but this
run lacked a same-allocation read control and startup affinity. The rejection
was invalid; see the corrected measurements and full-range successor above.

For the original widened kernels, relative to W4A8, W4A16 doubles
the dot-product count per packed line and adds four signed-byte-to-halfword
unpacks per 64-byte load. FP4 adds table lookup; FP16 adds integer widening,
conversion, and half-precision FMA. These paths are therefore issue/dependency
limited even when the identical allocation permits the W4A8 control to
saturate HBM. The `a64fx/swfp4fp8` kernels informed the E2M1 lookup and fused
register dataflow, but their exact FP8 paths decode to FP32 and their fused
MXFP4 SDOT path is W4A8 with dynamically quantized activations; neither is a
drop-in W4A16 kernel.

## Isolated HBM read prerequisite

Command:

```sh
a64fx/dequant-pipe/bench_hbm_read --sweep-cores --mode baseline \
  --mib 240 --iterations 20 --trials 3 --core-base 12
```

The allocation was first-touched while pinned to CPU 12, placing it in NUMA
node 4, and all readers used CPUs 12--19 in that same CMG.  Each core reads a
disjoint range much larger than L2.  Each assembly loop iteration contains
four full 64-byte SVE loads and advances by exactly one 256-byte cache line.

| cores | median GB/s | best GB/s | median GB/s/core |
|------:|------------:|----------:|-----------------:|
| 4 | 188.59 | 188.71 | 47.15 |
| 5 | 209.54 | 209.56 | 41.91 |
| 6 | 220.55 | 220.59 | 36.76 |
| 7 | 226.85 | 226.94 | 32.41 |
| 8 | **229.96** | **230.00** | 28.75 |

All trials ran at 2.0 GHz.  The eight-core result is close to the approximate
240 GB/s goal; adding cores beyond six gives diminishing returns as the CMG
memory interface saturates.

## Producer-only null dequantization

Adding only L2 ring stores already prevents preservation of the read-only HBM
curve.  The 1x load/store control reaches 187.00 GB/s packed at eight cores.
The realistic W8A8-sized 2x output reaches 110.34 GB/s packed while writing
220.68 GB/s of expanded data into the L2 ring (64 KiB chunks, median of three).

| cores | HBM read-only GB/s | null copy 1x GB/s | null dequant 2x GB/s |
|------:|-------------------:|--------------------:|----------------------:|
| 4 | 188.59 | 109.54 | 55.69 |
| 5 | 209.54 | 135.11 | 69.40 |
| 6 | 220.55 | 159.45 | 83.19 |
| 7 | 226.85 | 176.85 | 96.80 |
| 8 | 229.96 | 187.00 | **110.34** |

The 2x result is consistent with the expanded L2-store path saturating around
220 GB/s: preserving 230 GB/s of packed input would require approximately
460 GB/s of L2 writes before the GEMM core reads the ring.  Therefore a
separate-core 2x dequantizer cannot preserve the raw HBM packed-byte rate with
this handoff contract.

Ring footprint matters.  At eight cores, the selected three-trial median was 110.34
GB/s for 64 KiB chunks (256 KiB double ring/core), 85.08 GB/s for 128 KiB
chunks (512 KiB/core), and 42.05 GB/s for 256 KiB chunks (1 MiB/core).  The
64 KiB configuration is selected for the next paired experiment.

## Fused dequant-to-GEMM cycle budget

The producer result changes the design constraint: expanded weights must not be
materialized in an L2 handoff ring.  A fused consumer must load packed weights
directly from HBM, expand nibbles in registers, and immediately consume them
with SDOT.  Only accumulators/results should be stored.

At 230 GB/s and 2.0 GHz, the CMG receives 115 packed bytes/cycle, or one
256-byte packed cache line every 2.226 CMG cycles.  If cache lines are evenly
distributed over `C` fused consumers, the deadline for one core is:

```text
cycles_per_256B_line = 256 * C * 2.0 GHz / 230 GB/s = 2.226 * C
```

For M=1 W8A8, a 256-byte packed line expands to 512 INT8 weights and requires
eight SVE SDOT instructions.  The measured 1.8394 SDOT/cycle kernel rate makes
that 4.35 cycles, leaving the following budget for fused nibble expansion,
activation movement, addressing, scaling, and loop control:

| fused cores | packed GB/s/core | cycles/256 B line | M=1 SDOT cycles | remaining cycles |
|------------:|-----------------:|------------------:|------------------:|-----------------:|
| 6 | 38.33 | 13.36 | 4.35 | 9.01 |
| 7 | 32.86 | 15.58 | 4.35 | 11.23 |
| 8 | 28.75 | 17.81 | 4.35 | 13.46 |
| 9 | 25.56 | 20.03 | 4.35 | 15.68 |
| 10 | 23.00 | 22.26 | 4.35 | 17.91 |
| 11 | 20.91 | 24.49 | 4.35 | 20.14 |
| 12 | 19.17 | 26.71 | 4.35 | 22.36 |

An optimized signed-INT4 expansion can use approximately five vector integer
instructions per 64 packed bytes (`lsl`, `asr`, `asr`, `zip1`, `zip2`), or 20
instructions per cache line.  Because these compete with SDOT on the two SVE
arithmetic pipes, a first-order 91.97%-of-peak estimate is:

```text
INT4 fused cycles/line ~= (20 dequant + 8*M SDOT) / 1.8394
M=1: 15.22 cycles       M=2: 19.57 cycles
M=4: 28.27 cycles       M=6: 36.97 cycles
```

The corresponding arithmetic-pipe utilization against each core-count
deadline is:

| fused cores | INT4 M=1, 15.22 cycles | FP4 M=1, 19.57 cycles |
|------------:|------------------------:|----------------------:|
| 6 | 114.0% (impossible) | 146.5% (impossible) |
| 7 | 97.7% | 125.6% (impossible) |
| 8 | 85.5% | 109.9% (impossible) |
| 9 | 76.0% | 97.7% |
| 10 | 68.4% | 87.9% |
| 11 | 62.2% | 79.9% |
| 12 | 57.0% | 73.3% |

This estimate excludes activation packing, scale application, address/control
instructions, and latency stalls.  Consequently seven cores are only a
no-overhead M=1 boundary; **eight fused cores are the minimum plausible INT4
M=1 configuration**, and nine or ten provide useful implementation headroom.
M=2 needs roughly ten cores after overhead.  M>=4 cannot sustain 230 GB/s in
one CMG under this instruction model even with all 12 compute cores.

FP4 requires table lookup or equivalent mapping in addition to nibble
separation.  Using a provisional seven vector instructions per 64 packed bytes
gives `(28 + 8*M) / 1.8394 = 19.57` cycles/line for M=1.  That makes nine cores
another no-overhead boundary; **ten to twelve fused cores are the practical
FP4 M=1 range**.  These are scheduling budgets to validate with the fused
assembly kernel, not achieved performance claims.

## M and chunk sweep

Command:

```sh
a64fx/dequant-pipe/bench_dequant_pipe --sync atomic --format int4 \
  --path w8a8 --sweep-m --sweep-chunks --n 12288 --k 4096 \
  --iterations 3 --trials 1
```

Compressed GB/s; each cell is one measured trial:

| M | 4 KiB | 16 KiB | 32 KiB | 64 KiB | 256 KiB |
|---:|------:|-------:|-------:|-------:|--------:|
| 1 | 24.57 | 37.63 | 55.73 | **63.87** | 35.27 |
| 2 | 24.24 | 37.72 | 54.03 | **59.93** | 35.29 |
| 4 | 21.32 | 35.49 | 49.00 | **57.25** | 35.82 |
| 6 | 17.67 | 30.83 | 42.91 | **47.33** | 34.91 |

At 4 KiB, measured time in the atomic barrier was roughly 30--32%.  At 64 KiB
it fell to 8--20%, depending on M.  The reported barrier time includes useful
producer/consumer imbalance wait, not just the primitive's intrinsic latency.

## Format and W8A16 controls

Configuration: M=1, N=12288, K=4096, 64 KiB chunks, three streams per trial,
two trials.  Values are the better measured compressed rate.

| packed format | arithmetic | GB/s | expanded GB/s |
|:--------------|:-----------|-----:|--------------:|
| INT4 | W8A8 INT8 SDOT | 63.87* | 127.73* |
| E2M1 FP4 | W8A8 INT8 SDOT | 58.40 | 116.80 |
| INT4 | W8A16 INT16 SDOT | 35.09 | 140.35 |
| E2M1 FP4 | W8A16 INT16 SDOT | 32.29 | 129.16 |
| INT4 | W8A16 FP16 FMLA (`auto`) | 22.90 | 91.62 |
| E2M1 FP4 | W8A16 FP16 FMLA (`auto`) | 16.76 | 67.03 |

`*` The INT4 W8A8 row comes from the one-trial sweep; the other rows are the
best of two trials.  W8A16 moves four expanded bytes for each packed byte,
versus two for W8A8.

## Interpretation

The experiment disproves the two requested performance assumptions for this
implementation.  The tuned M=6 SDOT loop sustains above 90%, but not 96%, when
its operands stream from L2.  End-to-end packed bandwidth is much lower than
the 240 GB/s HBM interface target because dequant expansion, L2 handoff, the
small-M GEMM dependency structure, block scaling, and pair synchronization are
all included.  In particular, M=1 has only four accumulators per 64-column
tile, so it cannot use the same 24-way latency hiding as the M=6 peak kernel.

The next architectural experiment should fuse several adjacent N tiles in the
M=1/M=2 consumers, increasing independent accumulators without increasing M,
and reorder packed blocks K-major within each pair so those tiles arrive
contiguously.  That is a different data-layout contract and is intentionally
not hidden inside these baseline numbers.
