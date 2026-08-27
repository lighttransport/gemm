# DeepSeek-V4-Flash Layer-0 Results

Measured on one A64FX node with 48 cores, Fujitsu Compiler 4.12.2,
`OMP_PROC_BIND=close`, and `OMP_PLACES=cores`. The 3.287 GiB layer-0 subset was
loaded into anonymous HBM; `MemAvailable` remained approximately 26.3 GiB.
Offline staging and quantization are excluded from GEMM timing.

## Scalar EX-only FP4 arithmetic

Before GEMM integration, a standalone assembly microbenchmark measured
LUT-free scalar AArch64 arithmetic. It contains no NEON, SVE, floating-point,
or `TBL` instructions. E2M1 magnitude uses fixed shifts and a `MADD` to map
`0..7` exactly to the doubled integers `0,1,2,3,4,6,8,12`; sign is applied by
integer mask arithmetic. The kernels write int16 results so the measurement
includes the intended L1 exchange store.

With a 32 KiB packed working set and 4096 repetitions on one core:

| Kernel | Rate | Time per FP4 value |
|---|---:|---:|
| FP4 dequantize to signed integer | 0.212 Gvalue/s | 4.72 ns |
| FP4 x FP4 signed integer product | 0.106 Gproduct/s | 9.43 ns |

Counting one multiply and one eventual accumulation as two GEMM operations,
the product kernel represents only about 0.21 effective GFLOP/s. Consequently,
per-nibble scalar arithmetic is not suitable as the producer for the SVE GEMM;
it would starve the FL consumer despite using a separate execution pipe.

## Optimized single-core N32 kernel

The output-vectorized SVE kernel was measured on one pinned core with synthetic
`N=2048, K=4096` matrices, representative of LLM projection dimensions. These
are median timed GEMMs; FP4 repacking and scale preparation are excluded.

| Format | M | FP32 promotion K=256 | Pure FP16 accumulation |
|---|---:|---:|---:|
| MXFP4 | 1 | 7.3 GFLOP/s | 8.9 GFLOP/s |
| MXFP4 | 6 | 19.6 GFLOP/s | 33.9 GFLOP/s |
| NVFP4 1D | 1 | 7.3 GFLOP/s | 9.0 GFLOP/s |
| NVFP4 1D | 6 | 18.4 GFLOP/s | 32.9 GFLOP/s |
| NVFP4 2D | 1 | 7.3 GFLOP/s | 9.1 GFLOP/s |
| NVFP4 2D | 6 | 18.6 GFLOP/s | 33.3 GFLOP/s |

M=12 and M=24 sustain the same per-operation rates because dispatch repeats
the six-row microkernel. A 12-row intrinsic variant was rejected: register
spills reduced pure-FP16 throughput to about 22--23 GFLOP/s.

### Pre-dequantized L2 panel

An alternative path reads each packed FP4 `N32 x K` panel once, expands it to
a K-major FP16 workspace with SVE, and then reuses that panel for all M rows.
For K=4096 the panel is 256 KiB and therefore resides in L2. Reported timing
includes dequantization, the FP16 panel store, its reload, and GEMM.

| Format | M | K=256 promotion | Pure FP16 |
|---|---:|---:|---:|
| MXFP4 | 6 | 19.5 | 31.5 |
| MXFP4 | 24 | 22.9 | 41.5 |
| MXFP4 | 48 | 23.6 | 43.7 |
| MXFP4 | 128 | 23.8 | 44.5 |
| NVFP4 1D | 24 | 22.0 | 40.9 |
| NVFP4 1D | 128 | 22.9 | 43.7 |
| NVFP4 2D | 24 | 22.5 | 41.1 |
| NVFP4 2D | 128 | 23.2 | 44.1 |

Values are GFLOP/s/core at N=2048, K=4096. M=1 falls to 6.4--7.5 GFLOP/s
because the expanded panel is consumed only once. The crossover is around
M=12. At M=128 the L2 path is 33--35% faster for pure FP16 and 24--28% faster
with K=256 promotion. It saves HBM2 traffic by fetching packed weights once per
N32 panel, at the cost of a local 4x-expanded FP16 write/read stream.

### Full-layer streaming and L1 panel comparison

To exceed the 8 MiB CMG L2, `bench_fp4_stream` uses N=32768, K=4096: 64 MiB
packed MXFP4 plus 8 MiB prepared FP16 scales. CPU 12 and memory node 4 were
explicitly bound with `numactl`. The K=256 L1 panel occupies 16 KiB; the full-K
L2 panel occupies 256 KiB.

| Kernel | M=1 | M=6 | M=24 | M=128 |
|---|---:|---:|---:|---:|
| Direct inline dequant | 8.26 | 36.69 | 36.36 | 35.70 |
| Full-K L2 panel | 8.36 | 38.60 | 50.33 | 53.49 |
| K=256 L1 panel | 8.42 | 38.64 | 48.67 | 46.85 |

Values are GFLOP/s/core with K=256 FP32 promotion. At M=1 the panel paths read
only 2.35--2.37 GB/s of FP4 plus prepared scales. The same 72 MiB buffers reach
9.84 GB/s source bandwidth with the platform `memcpy` (19.68 GB/s counting
both read and write). Therefore even the full-layer test remains dequantization
limited, not HBM2-bandwidth limited. The L1 panel does not help: it ties the
full panel at M<=6 and becomes 12% slower at M=128. Retaining the full-K panel
in L2 gives the best reuse and traversal order.

### Persistent FP16 sidecar for M>1

`fp4_matrix_prepare_bf16` (legacy API name) decodes adjacent N32 tiles into a
K-major `[N/64][K][64]` FP16 sidecar. The timed GEMM packs A into `[K][12]`
microtiles and uses a hand-scheduled 12x64 kernel with 24 resident FP16
accumulators. On a pinned 12-core CMG with N=8192 and K=4096:

| M | K=256 FP32 promotion | Pure FP16 accumulation |
|---:|---:|---:|
| 1 | 186 GFLOP/s | 185 GFLOP/s |
| 6 | 1,092 GFLOP/s | 1,125 GFLOP/s |
| 8 | **1,481 GFLOP/s** | **1,444 GFLOP/s** |
| 12 | **2,122--2,167 GFLOP/s** | **2,187 GFLOP/s** |
| 24 | **2,075--2,097 GFLOP/s** | **2,202 GFLOP/s** |
| 128 | **2,013--2,054 GFLOP/s** | **2,167 GFLOP/s** |

These timings include activation packing and padded-tail handling, but exclude
the persistent sidecar preparation. M=8 computes a zero-padded 12-row tile and
copies the eight live output rows, yet still exceeds the 800 GFLOP/s target by
82%. Complete M12 blocks reach 71% of the 3.072 TFLOP/s hardware peak and 84%
of the previously measured 2.612 TFLOP/s streaming-FP16 software ceiling.

The K=256 follow-up keeps all 12x64 FP32 partial sums in a contiguous 3 KiB
scratch tile until the final store. Previously, row-major partial sums at
N=8192 were 32 KiB apart and repeatedly mapped 12 rows onto the same L1 sets
for all 16 promotion blocks. Removing that conflict raises M=12 from 1.82 to
as much as 2.17 TFLOP/s. Combining activation packing and compute in one
OpenMP region removes another fork/join. A two-panel activation-reuse schedule
and a four-K assembly unroll were measured and rejected: the former displaced
the 32 KiB weight block from L1 and fell to 1.94 TFLOP/s, while the latter was
neutral. The retained K=256 M12 path is within roughly 1.5--3.6% of 2.2
TFLOP/s depending on run-to-run HBM variation; pure FP16 reaches 2.20 TFLOP/s.

The sidecar occupies 64 MiB for this matrix (4x the packed FP4 weight storage).
Unit tests pass for MXFP4, NVFP4 1D, and NVFP4 2D with the same numerical error
as the existing panel path.

### Fused 12-core CMG baseline

The fused OpenMP kernel assigns disjoint N32 tiles to all 12 cores. Every core
owns a private 256 KiB decoded panel and immediately computes all M rows after
decoding, with no producer queue or cross-core output sharing.

| M | Time | CMG GFLOP/s | FP4/scale source rate |
|---:|---:|---:|---:|
| 1 | 2.77 ms | 97 | 27.3 GB/s |
| 6 | 3.56 ms | 452 | 21.2 GB/s |
| 24 | 10.98 ms | 586 | 6.87 GB/s |
| 128 | 54.90 ms | 626 | 1.38 GB/s |

This is an 11.6--11.8x speedup over one core. A 12-core parallel `memcpy` of
the same buffers reaches 66.6 GB/s source bandwidth, or 133.1 GB/s counting
read and write. M=1 FP4 generation therefore uses about 41% of this measured
source-read rate and remains instruction/dequant limited. At larger M, FP16
FMA and FP32 promotion dominate while source traffic is amortized.

### Pipe-aware direct matvec optimization

The A64FX instruction database shows `ZIP1` and `TBL` are FLA-only with
latencies 6 and 6, while vector FMLA/FMUL and vector-vector bit operations can
use either FL pipe. Immediate `AND` is also FLA-only. The optimized decoder
therefore hoists one vector mask, uses vector-vector `AND`, and schedules four
independent tiles. The hand-written MXFP4 loop also pipelines two K steps: the
second four-vector decode separates the first `TBL` group from its dependent
`FMUL`. M=1 broadcasts each activation once and uses four one-uop vector FMLAs
instead of four two-uop indexed FMLAs.

Full 72 MiB MXFP4 streaming results on one 12-core CMG, K=256 promotion:

| M | Selected path | CMG GFLOP/s | Source GB/s |
|---:|---|---:|---:|
| 1 | Eight-tile shared activation-table direct | **400.0** | **112.5** |
| 6 | Direct six-row | 511 | 24.0 |
| 24 | Full-K L2 panel | 591 | 6.92 |
| 128 | Full-K L2 panel | 621 | 1.37 |

The M1 path first improved to 300.5 GFLOP/s by scaling the 16-entry E2M1 table
by each activation once and using decoded products directly with row scales.
The current layout interleaves eight N32 tiles block-major and hoists all K
activation tables into one shared 256 KiB workspace. This removes redundant
per-core table FMULs, exposes eight output chains, and reaches 400 GFLOP/s
while retaining four-bit density. K32 scale-at-boundary spilling (383.5
GFLOP/s) and scalar-offset loads (383.7 GFLOP/s) were measured and rejected.

Pure FP16 M=1 reaches 293 GFLOP/s and 82.3 GB/s. A hand-written eight-tile
variant was rejected at 95 GFLOP/s; widening concentrates still more
`ZIP1`/`TBL` work on FLA and requires callee-save vector state. (The earlier
compiler-generated eight-tile attempt reached 131 GFLOP/s.) At an assumed
230 GB/s CMG HBM rate, the MXFP4 plus prepared-scale bandwidth roof is about
818 GFLOP/s. The current M=1 kernel reaches 34% of that roof; FLA-only nibble
interleave/table lookup remains the principal execution limit.

#### Removing ZIP1 and TBL

The BF16 `p_odd` load works because every BF16 already occupies a separate
16-bit memory position. Two FP4 values share one byte, so placing them directly
in alternating halfword lanes requires either a permutation or padding to one
byte per nibble. An opt-in 1-byte/nibble sidecar measured that trade-off:

| Representation and decode | GFLOP/s | Source GB/s |
|---|---:|---:|
| Packed FP4, `ZIP1` + `TBL` | 280 | 78.6 |
| One-byte nibble, compiler kernel | 226 | 120.3 |
| One-byte nibble, hand assembly | 368 | 195.3 |
| One-byte nibble, K-interleaved hand assembly | **403** | **214.3** |
| One-byte nibble, no `ZIP1`/`TBL`, arithmetic bits | 167 | 88.9 |

All results use M=1, K=256 promotion, and one 12-core CMG. The optimized byte
layout interleaves four N32 tiles at each K step, converting four distant tile
streams into one sequential stream. Its Kx2 assembly loop covers `TBL` latency
with eight independent decoded vectors. At 403 GFLOP/s it is 45% faster than
packed FP4 and reaches 214 GB/s, about 93% of the useful ceiling implied by a
230 GB/s CMG HBM rate. It is the selected speed path when the doubled code
footprint is acceptable; packed FP4 remains the capacity-efficient path.
Arithmetic E2M1 construction replaces `TBL` with about ten mask/shift/predicate
operations and remains slower.

#### Four-bit bitplane arithmetic

A same-density layout stores the four code bits as four 32-bit planes for each
N32/K step. Even outputs occupy the low half and odd outputs the high half, so
`LD1RW` plus lane-dependent shifts reconstruct normal output order without a
permutation. Three exact mappings were measured on the M=1, K=256, 12-core
stream:

| Mapping | CMG GFLOP/s | Source GB/s |
|---|---:|---:|
| Magic `MUL 0x3800` FP16-bit construction | 69.4 | 19.5 |
| Shift/substitute FP16-bit construction | 66.7 | 18.8 |
| Signed fixed-point plus `SCVTF` | 54.8 | 15.4 |

The winning loop has no `ZIP1` or `TBL`, but each vector instead needs four
plane broadcasts, four variable shifts, four masks, and roughly eleven
bit-construction operations. Some compiler-generated loads also become GPR
loads plus vector `MOV`. This overwhelms the saved FLA work, so the bitplane
path remains opt-in; the packed and byte-expanded table paths are both faster.

The result is below the 256 GFLOP/s/core dense-FP16 peak because A64FX has no
FP4 arithmetic. Every 32-output FMA step also requires a 16-byte packed load,
byte-to-halfword expansion, two nibble operations, interleave, table lookup,
and FP16 scale multiply. At M=1 that dequantization is amortized over only one
FMA and dominates. M=6 reuses each decoded vector six times, but the loop still
issues six dependent FP16 accumulator chains plus scalar activation loads.
FP32 promotion additionally converts 192 FP16 lanes per output tile and
reads/writes FP32 partial sums every 256 K values. Thus 128 GFLOP/s is not an
appropriate expectation unless the implementation either reuses each decoded
weight across a substantially larger register tile or moves dequantization
outside the timed kernel (which would no longer measure FP4 GEMM).

### Scalar AArch64 dequantization experiment

A hand-written double-buffered variant uses only scalar base-ISA GPR
instructions for dequantization. It loads two packed 64-bit words, extracts 16
bytes, performs 16 indexed accesses to a 256-entry pair LUT, and stores one
64-byte FP16 vector into alternating L1 cache lines. SVE consumes the other
line for scaling and six-row FP16 FMA. Object-code inspection confirms that
the producer contains no NEON instructions.

| Format | FP32 promotion K=256 | Pure FP16 accumulation |
|---|---:|---:|
| MXFP4 | 10.8 GFLOP/s | 13.7--13.8 GFLOP/s |
| NVFP4 1D | 10.4 GFLOP/s | 13.3--13.4 GFLOP/s |
| NVFP4 2D | 10.5 GFLOP/s | 13.4 GFLOP/s |

This is slower than direct SVE dequantization. Although EX and FL execution
can overlap, each K step requires 16 `AND`, 16 LUT `LDR`, 16 `STR`, the byte
shift chains, and a 64-byte SVE reload. The scalar producer cannot feed the SVE
consumer quickly enough; separate pipes do not offset the expansion count.

## Attention projections

The table reports the median across WKV, WQ_A, WQ_B, WO_A, and WO_B. Each entry
is `GFLOP/s / relative-L2 accumulation error`.

| M | Format | K=32 promotion | K=256 promotion | Pure FP16 |
|---:|---|---:|---:|---:|
| 1 | MXFP4 | 11.26 / 1.33e-4 | 22.15 / 4.83e-4 | 28.69 / 1.61e-3 |
| 1 | NVFP4 1D | 10.30 / 4.21e-4 | 19.86 / 5.85e-4 | 24.91 / 1.59e-3 |
| 1 | NVFP4 2D | 10.16 / 4.14e-4 | 19.28 / 5.40e-4 | 25.70 / 1.59e-3 |
| 32 | MXFP4 | 14.27 / 1.38e-4 | 48.32 / 4.34e-4 | 105.13 / 1.49e-3 |
| 32 | NVFP4 1D | 13.42 / 4.37e-4 | 40.40 / 5.75e-4 | 79.97 / 1.58e-3 |
| 32 | NVFP4 2D | 12.87 / 4.23e-4 | 41.46 / 5.04e-4 | 84.25 / 1.54e-3 |
| 128 | MXFP4 | 14.41 / 1.36e-4 | 49.21 / 4.29e-4 | 111.49 / 1.61e-3 |
| 128 | NVFP4 1D | 13.60 / 4.45e-4 | 41.14 / 5.23e-4 | 85.25 / 1.55e-3 |
| 128 | NVFP4 2D | 13.36 / 4.06e-4 | 42.90 / 5.74e-4 | 84.73 / 1.57e-3 |
| 512 | MXFP4 | 14.07 / 1.29e-4 | 46.58 / 3.92e-4 | 106.20 / 1.86e-3 |
| 512 | NVFP4 1D | 13.64 / 4.13e-4 | 42.05 / 6.18e-4 | 84.48 / 1.68e-3 |
| 512 | NVFP4 2D | 13.40 / 4.02e-4 | 43.61 / 5.10e-4 | 94.12 / 1.63e-3 |

Single-core attention medians at `M=1, K=32` were 0.24 GFLOP/s for MXFP4 and
0.22 GFLOP/s for both NVFP4 layouts. Pure-FP16 accumulation reached 0.80 and
0.59 GFLOP/s respectively.

## Routed FFN

Expert-zero W1/W2/W3 medians at `M=128`:

| Format | K=32 promotion | K=256 promotion | Pure FP16 |
|---|---:|---:|---:|
| MXFP4 | 14.65 / 1.63e-4 | 52.31 / 4.36e-4 | 110.39 / 1.50e-3 |
| NVFP4 1D | 13.94 / 4.16e-4 | 46.32 / 5.17e-4 | 94.78 / 1.51e-3 |
| NVFP4 2D | 13.96 / 3.43e-4 | 45.75 / 5.80e-4 | 92.72 / 1.52e-3 |

The cache-streaming `M=1, K=32` scan touched W1/W2/W3 for all remaining 255
experts (765 projections):

| Format | Total GEMM time | Effective GFLOP/s |
|---|---:|---:|
| MXFP4 | 1298.1 ms | 9.89 |
| NVFP4 1D | 1463.8 ms | 8.77 |
| NVFP4 2D | 1445.8 ms | 8.88 |

Native expert MXFP4 repacking is bit-exact (`0.0` relative error). Requantizing
expert zero from its native MXFP4 values produced relative weight errors of
6.38e-2--6.66e-2 for NVFP4 1D and 7.93e-2--9.03e-2 for NVFP4 2D across
W1/W2/W3. These format-conversion errors are separate from the accumulation
errors in the performance tables.

## FP4 to INT8 SDOT

The exact E2M1-times-two re-encoding and K4 activation quantizer sustain
409.1 GFLOP/s (217.3 GB/s) on the 128 MiB expanded stream. Scalar INT32 and
assembly results agree within 4.7e-8 relative L2. Across layer-0 expert-zero
W1/W2/W3, worst relative L2 is 3.94e-3 and minimum cosine similarity is
0.999992. Larger activation groups trade accuracy for a modest peak of
419.8 GFLOP/s at K32.

An EX-only packed producer was rejected before fusion. A layout-friendly
nibble split reaches 40.6 GB/s packed input across 12 cores, a downstream roof
of 162 GFLOP/s. Exact SWAR E2M1 mapping reaches only 6.06 GB/s and 24.2
GFLOP/s. Both are far below the roughly 220 GB/s packed rate required for an
800 GFLOP/s CMG kernel.

## Packed activation-pair LUT

An exact activation-dependent 256-entry INT16 LUT removes FP4 decode from the
hot loop while retaining the original four-bit weight stream. Eight
independent SVE gather chains process 128 rows, and FP32 conversion/rescaling
occurs only at the requested activation-group boundary.

| Activation K group | Kernel GFLOP/s | Packed source GB/s |
|---:|---:|---:|
| 32 | 178.9 | 50.3 |
| 8 | 177.2 | 49.8 |
| 4 | 178.7 | 50.3 |

Configuration: MXFP4 N=32768, K=4096, 12 cores pinned to one CMG. Synthetic
unit tests agree with the corresponding expanded SDOT output within 5e-8
relative L2. The invariant speed across K32/K8/K4 rules out scale promotion
as the limiter. Indexed `LD1SH` is the measured ceiling, so this route is kept
as a packed-density control rather than replacing the 409.1 GFLOP/s K4 SDOT
kernel.

### Packed K4 SDOT

A second exact-density SDOT layout packs `32 rows x K4` into each 64-byte SVE
load. Two byte table decodes plus `ZIP1/ZIP2` feed two SDOT vectors, with all
four N128 chunks in flight. FP32 activation scaling and accumulation are
scheduled across eight independent vectors. Synthetic MXFP4/NVFP4 results
match expanded K4 SDOT within 1e-7 relative L2.

Despite the lower static operation count, byte `TBL` plus byte permutation is
only 83.5 GFLOP/s on one CMG. Its 64 MiB codes plus 16 MiB FP32 row-scale
sidecar move at roughly 26 GB/s. This rejects packed byte-table SDOT as the
route to 800 GFLOP/s; expanded sequential SDOT remains about 410 GFLOP/s and
the packed FP16 t8 kernel remains about 400 GFLOP/s.

The A64FX instruction data also explains why the nominal 818 GFLOP/s HBM
roof is not attainable by this exact packed decoder. For every K value, its
eight output vectors require eight vector `ZIP1` and eight `TBL` operations.
Both are single-uop, FLA-only instructions, so they consume at least 16 FLA
issue slots for 512 useful FLOPs. At 2.2 GHz and 12 cores this is an 845
GFLOP/s zero-overhead bound, before issuing the eight FMLAs, loads, branches,
or handling six-cycle permutation/table latency. Thus 800 GFLOP/s would
require over 94% of that ideal bound while doing all remaining work for free.
K-promotion sweeps (K64 through K4096) span only 378--403 GFLOP/s, and explicit
1/2 KiB look-ahead prefetch remains 397--400 GFLOP/s. These controls rule out
FP32 spill frequency and ordinary cache-miss latency as the missing factor.

### fapp bottleneck measurement

`bench_fp4_fapp` places an `fp4_m1_t8` fapp region around 400 warmed iterations
of only the fused M1 call. With N=32768, K=4096, K256 promotion, threads pinned
to cores 12--23, raw PMU counters summed over the 12 active threads report:

| event | cycles or utilization |
|---|---:|
| `STALL_BACKEND` (0x0024) | 29.00% |
| `FL_COMP_WAIT` (0x018a) | **25.11%** |
| `LD_COMP_WAIT` (0x0184) | 4.44% |
| `LD_COMP_WAIT_L1_MISS` (0x0182) | 1.05% |
| `LD_COMP_WAIT_L2_MISS` (0x0180) | 2.10% |
| `LD_COMP_WAIT_PFP_BUSY` (0x0186) | 0.02% |
| FLA valid (0x01a4) | **64.80%** |
| FLB valid (0x01a5) | **56.87%** |
| `STALL_FRONTEND` (0x0023) | 0.31% |

The single-core control makes the diagnosis clearer: FLA is valid 76.07%, FLB
66.80%, `FL_COMP_WAIT` rises to 29.53%, while `LD_COMP_WAIT` falls to 1.72%.
Therefore the actual bottleneck is the floating-point/permutation dependency
network, not HBM2, frontend delivery, or the load/prefetch ports. In particular,
each decoded vector has the serial `ZIP1` (six cycles) -> `TBL` (six cycles) ->
`FMLA` chain; `ZIP1` and `TBL` are both restricted to FLA. Eight independent
output chains hide only part of those completion latencies. This also explains
why prefetching and longer FP16 accumulation blocks do not materially improve
the kernel.

### ZIP/TBL removal follow-up

The fapp result motivated three same-density exact controls:

| kernel | configuration | GFLOP/s | result |
|---|---:|---:|---|
| baseline t8 | N32768 K4096 K256 | 397--403 | selected |
| row-half, no `ZIP1` | N32768 K4096 K256 | 378--382 | rejected |
| row-half Kx2 pipeline | N32768 K4096 K256 | 363--365 | rejected |
| t12 wider chains | N36864 K4096 K256 | 400--402 | rejected; t8 is 416.6 at the same N |
| affine E2M1, no `TBL` | N32768 K4096 K256 | 135.1 | rejected |

The row-half layout stores row `j` in the low nibble and row `16+j` in the
high nibble. Two predicated loads of the same 16 bytes place those codes in
the lower and upper vector halves, and `ORR` replaces `ZIP1`. It remains exact,
but fapp reports FLA valid 62.88%, FLB valid 60.50%, `FL_COMP_WAIT` 25.55%, and
`LD_COMP_WAIT` 4.13%. Thus the extra load is inexpensive, but replacing ZIP
does not break the remaining `ORR -> TBL -> FMLA` completion chain.

The affine layout stores sign in bit zero and remaps E2M1 magnitude to
`{1,0,2,3,4,5,6,7}`. It constructs the FP16 magnitude as
`(r << 9) + 0x3800`, predicates `r==1` to zero, inserts the sign, and multiplies
by the activation. This exactly removes `TBL`, but the added FL operations are
three times slower than the table kernel. A predicate-plane representation is
also statically rejected: four plane loads plus eight PRX `PUNPK` operations
are followed by at least four serial predicated code-building operations per
output vector, giving more FL work than the baseline before lookup/FMA.

Finally, a best-case approximate symmetric INT4 fallback was evaluated with a
separate per-row K32 scale on staged DeepSeek W1/W2/W3. Its worst relative L2
error is 10.56%, far beyond the agreed 0.5% gate, so it is not implemented as
a performance kernel.

A second packed experiment replaced each 256-entry gather with two sequential
16-entry `TBL`s, one for each activation in a packed K pair. It is exact and
removes indexed loads, but requires 16 `TBL`s per K pair to cover 128 rows.
Even with two independent row chunks in flight it reaches only 84.5 GFLOP/s
(23.8 GB/s packed). The doubled FLA table-operation count is worse than both
the gather control and the original row-paired `ZIP1`/`TBL` kernel, so this is
retained only as a negative control.

## Conclusion

Promoting every 256 K values is the useful balanced mode on these workloads:
it improves throughput by roughly 3-4x over promotion every 32 values while
keeping median accumulation error below about 6.2e-4. Pure FP16 accumulation
is another 1.8-2.4x faster but raises relative error to roughly 1.5e-3-1.9e-3.
MXFP4 is consistently faster than either NVFP4 layout because its K32 scaling
requires fewer scale decodes.
