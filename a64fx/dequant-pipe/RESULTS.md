# Native A64FX results

Measured 2026-09-19 with Fujitsu `fcc` 4.12.2 on CPUs 12--23 of one 2.0 GHz
A64FX CMG.  All end-to-end numbers below use `--sync atomic`; they are controls,
not headline hardware-barrier results.

## Outcome

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
| INT4 | INT16 SDOT | **147.00** | 147.02 | 63.9% |
| E2M1 FP4 | INT16 SDOT | **132.44** | 132.49 | 57.6% |
| INT4 | FP16 FMA | **96.28** | 96.30 | 41.8% |
| E2M1 FP4 | FP16 FMA | **84.59** | 84.64 | 36.8% |

The INT4 INT16 supertile improves substantially over the original four-stream
kernel: 147.00 GB/s controlled versus the earlier 67.55 GB/s result. On an
ordinary heap allocation the same new kernel plateaued near 120.9 GB/s; a
1/4/6/8/10/12-core sweep scaled 13.24, 49.55, 73.83, 98.32, 117.94, and
119.29 GB/s, identifying the familiar slow-placement ceiling. XOS huge pages
raised the result to 147 GB/s, but not to 230 GB/s.

The remaining controlled gap is arithmetic. Relative to W4A8, W4A16 doubles
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
