# A64FX paired dequant-to-GEMM experiment

This is a native A64FX experiment for bandwidth-limited INT4/FP4 decode.  Six
producer cores dequantize packed weights into double-buffered handoff rings;
six consumer cores read those rings through the shared CMG L2 and run SVE
GEMM kernels.  The default core placement is one complete CMG:

- producers: CPUs 12--17
- consumers: CPUs 18--23
- one pairwise synchronization blade per producer/consumer pair
- one synchronization per multi-block chunk, rather than per 4 KiB block

The packed-byte rate is the primary metric.  It counts only compressed HBM
weight input, so the requested 240 GB/s means 240 GB/s of INT4/FP4 data before
expansion.  FP32 block-128 scales are reported separately as metadata traffic.

## Formats and arithmetic paths

- `--format int4`: signed two's-complement nibbles, `[-8, 7]`.
- `--format fp4`: E2M1 values represented by the exact x2 integer lattice
  `{0,1,2,3,4,6,8,12}` and its signed half; the x0.5 factor is folded into the
  block scale.
- `--path w8a8`: dequantize weights to INT8 and use INT8 SVE `sdot`.
- `--path i16`: widen weights and activations to INT16 and use the A64FX
  INT16-to-INT64 form of `sdot`.
- `--path auto`: convert both sides to FP16 and use FP16 `fmla` only when the
  complete run passes conversion, product-underflow, and block-accumulator
  overflow checks.  Otherwise the run falls back to INT16 SDOT.
- `--path fp16`: require the FP16 path and reject an unsafe scale set.

The M=6 INT8 kernel interleaves activations by K quartet, keeps 24 independent
accumulators, overlaps the next four vector loads with current SDOT work, and
uses a 64-byte-aligned, two-K-group loop.  Its 72-instruction loop puts the
conditional branch in the fourth A64FX decode slot.  `--peak` measures that
load+SDOT loop using a 2 MiB weight stream resident in the CMG L2.

The implementation choices follow Fujitsu's public
[A64FX Microarchitecture Manual](https://github.com/fujitsu/A64FX/blob/master/doc/A64FX_Microarchitecture_Manual_en_1.8.1.pdf)
and [A64FX HPC Extension specification](https://www.stonybrook.edu/commcms/ookami/support/_docs/A64FX_Specification_HPC_Extension_v1_EN.pdf):
two SVE arithmetic pipes, two load ports/one store port, 256-byte cache lines,
and the vendor hardware-barrier facility.

## Build and correctness

Run on an A64FX compute node with Fujitsu `fcc` and `libhwb`:

```sh
make -C a64fx/dequant-pipe
make -C a64fx/dequant-pipe test
```

The self-test exhaustively decodes all nibble codes and checks every M=1, 2,
4, and 6 INT8, INT16, and FP16 kernel against scalar references for both
formats.  A small end-to-end scalar check is:

```sh
a64fx/dequant-pipe/bench_dequant_pipe \
  --sync atomic --format fp4 --path auto --m 6 \
  --n 384 --k 256 --iterations 1 --trials 1 --verify
```

## Benchmarking

Before running the paired pipeline, verify the HBM read prerequisite in one
CMG.  The reader pins the allocator to CPU 12 (NUMA node 4), partitions one
240 MiB allocation into disjoint core-local streams, and executes exactly four
64-byte SVE loads per 256-byte A64FX cache line.  There are no stores, barriers,
or arithmetic operations in the timed assembly loop:

```sh
a64fx/dequant-pipe/bench_hbm_read --sweep-cores --mode baseline \
  --mib 240 --iterations 20 --trials 3 --core-base 12
```

`--mode prefetch` adds one `pldl1keep` eight cache lines ahead as a diagnostic;
it is not selected because it reduced bandwidth on the measured node.

The producer-only controls add an alternating L2 ring without a consumer:

```sh
# 1x load/store control
a64fx/dequant-pipe/bench_hbm_read --sweep-cores --mode null-copy \
  --mib 240 --chunk-kib 128 --iterations 10 --trials 3

# W8A8-sized INT4/FP4 -> INT8 null expansion
a64fx/dequant-pipe/bench_hbm_read --sweep-cores --mode null-dequant \
  --mib 240 --chunk-kib 64 --iterations 10 --trials 3
```

For `null-dequant`, each 64 KiB packed chunk produces a 128 KiB slot.  Two
alternating slots occupy 256 KiB/core.  A 128 KiB chunk occupies the requested
512 KiB/core one-way equivalent, but was slower once seven or eight producers
were active.

The producer-only experiment shows that a 2x expanded L2 handoff is capped at
about 110 GB/s of packed input.  The next kernel must therefore fuse nibble
expansion with SDOT and avoid storing expanded weights.  See the cycle-budget
table in [RESULTS.md](RESULTS.md): at 230 GB/s, INT4 M=1 needs at least eight
fused consumers in practice, while FP4 M=1 should use ten to twelve.

The direct INT4 experiment implements that fused contract for both INT8 and
INT16 SDOT.  It reads tile-major packed weights from HBM, expands nibbles only
in SVE registers, and stores only final dot-product accumulators.  Four K=128
blocks are interleaved for INT8 and two for INT16 to expose 16 independent
accumulator vectors:

```sh
a64fx/dequant-pipe/bench_fused_sdot --verify --path int8 \
  --cores 12 --mib 240 --iterations 10 --trials 3
a64fx/dequant-pipe/bench_fused_sdot --path int16 \
  --cores 12 --mib 240 --iterations 10 --trials 3
```

This is a compute-kernel bandwidth test: block scales are deliberately absent,
so its packed-byte rate is an upper bound for a scaled production GEMM.  INT8
uses the original consecutive-nibble stream.  INT16 uses a split-nibble layout:
the low and high nibbles of each byte hold complete 16-column SDOT vectors,
removing two byte permutations before widening.  A model converter must emit
that layout when selecting the INT16 kernel.

The W4A16 extension covers both signed INT4 and E2M1 FP4 with two arithmetic
routes:

- `--path int16 --kernel super` expands into INT16 registers and uses the
  INT16-to-INT64 SVE `sdot`. Its two K=128 blocks are interleaved by K quartet,
  producing one sequential packed stream and 16 independent accumulator
  vectors. FP4 uses the exact doubled-integer E2M1 table and folds the factor
  of one half into the eventual block scale.
- `--path fp16 --kernel super` uses a separate N-lane layout. For each K
  scalar, 32 packed bytes hold 64 output columns; four independent blocks are
  adjacent. Each load expands directly into two 32-lane FP16 vectors and is
  consumed by FP16 FMA without an expanded-weight store. This experimental
  kernel accumulates in FP16 and therefore needs a production overflow/error
  gate, just like the staged FP16 route.

Both are unscaled compute-kernel bandwidth probes. They demonstrate direct
dequantization and arithmetic, but a production block-quantized GEMV must add
scale application and a wider accumulation policy where model accuracy
requires it.

```sh
a64fx/dequant-pipe/bench_fused_sdot --verify \
  --format int4 --path int16 --kernel super \
  --cores 12 --mib 240 --iterations 10 --trials 5
a64fx/dequant-pipe/bench_fused_sdot \
  --format fp4 --path int16 --kernel super \
  --cores 12 --mib 240 --iterations 10 --trials 5
a64fx/dequant-pipe/bench_fused_sdot \
  --format int4 --path fp16 --kernel super \
  --cores 12 --mib 240 --iterations 10 --trials 5
```

Use the XOS 2 MiB allocation environment shown below for the controlled
single-CMG headline. Ordinary heap allocations can land in the previously
documented approximately 120 GB/s placement state.

The no-degradation INT8 observation uses a four-block K-major supertile. One
observed fast allocation used a 16 KiB gap between core-local shards:

```sh
a64fx/dequant-pipe/bench_fused_sdot --verify --kernel super \
  --cores 12 --mib 240 --skew-kib 16 --iterations 10 --trials 5
```

Within each 16 KiB supertile, the 128 packed bytes for one K quartet from each
of four K=128 blocks are adjacent.  Thus each core presents one sequential HBM
stream rather than four streams separated by 4 KiB. Production sharding must
preserve the supertile ordering. The gap is an experimental placement knob,
not a production mapping rule.
The benchmark performs one unreported warm-up before its timed trials.

### Address color and the discovery probe

“Address color” means the subset or XOR combination of physical-address bits
that selects a cache set/bank or a downstream memory resource.  The A64FX
manual publishes the L2 mapping: `PA[8]` selects one of two L2 banks, while the
11-bit PIPT set index XORs `PA[18:8]` with several high physical-address
fields.  It does not publish the HBM channel/bank mapping.  Two streams can be
far apart and share no cache lines, yet still contend for the same colored
resource.  This is separate from ordinary 64-byte vector alignment, the
256-byte A64FX cache-line size, and the logical INT4 supertile format.  See
[A64FX memory addressing](../doc/a64fx_memory_addressing.md) for the formula
and manual references.

The 240 MiB benchmark divides its payload evenly across twelve cores, so the
unskewed starts are 20 MiB apart. The first separate-process measurements gave
about 120 GB/s with zero skew and 228.84 GB/s with 16 KiB skew. A controlled
sweep subsequently reused one allocation for every point and found all-fast
or all-slow runs independent of the tested gap. The original comparison was
therefore confounded by physical allocation or page placement; it does not
identify 16 KiB as an HBM selector.

Use `bench_hbm_color` to sweep relative stream skew, common base offset, and
individual address bits within the same arena. It supports
`--page-mode thp|base` and reports Linux page backing from `smaps`:

```sh
a64fx/dequant-pipe/bench_hbm_color --cores 12 --mib 240 \
  --min-skew-kib 0 --max-skew-kib 64 --step-bytes 256
a64fx/dequant-pipe/bench_hbm_color --bit-sweep --max-skew-kib 1024
cd a64fx/dequant-pipe
EPOCHS=20 OUTPUT=hbm-color-epochs.csv ./run_hbm_color_epochs.sh
```

The probe reports per-core start PFNs when the kernel permits pagemap access.
On the measured node Linux masks those PFNs, so the epoch runner can discover
repeatable fast allocations but cannot yet name physical HBM selector bits.
For controlled 2 MiB XOS pages, select the heap-backed allocation path:

```sh
LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1 \
XOS_MMM_L_HPAGE_TYPE=hugetlbfs XOS_MMM_L_HUGETLB_SZ=2M \
XOS_MMM_L_HUGE_MALLOC=1 XOS_MMM_L_FORCE_MMAP_THRESHOLD=1 \
XOS_MMM_L_HUGETLB_FALLBACK=0 \
  a64fx/dequant-pipe/bench_hbm_color --page-mode xos
```

The measured 240 MiB, 12-core sweep sustained 227.76--229.71 GB/s across
0--64 KiB skews; `smaps` confirmed 2048 KiB kernel/MMU pages.

To measure how much arithmetic fits under HBM latency/bandwidth, add fixed
SDOT or FMLA work per 256-byte line and pair it with a read measurement on the
same arena:

```sh
# Supported counts: 4, 8, 12, 16, 24, 32, 40, 48, 52, 56, 60, 64.
./bench_hbm_color --page-mode xos --op sdot --ops-per-line 48 \
  --paired-baseline --max-skew-kib 0
```

On the measured CMG, 48 SVE SDOT or FMLA instructions retained approximately
229.5 GB/s. At 52 instructions bandwidth fell to about 224.2 GB/s, making 48
instructions per 256 packed bytes the measured no-degradation budget.

For production staging, allocate all twelve regions from one CMG-local arena
and first-touch it on that CMG. Re-run the placement sweep if payload size,
page policy, allocator, CMG, or node configuration changes.

Headline mode requires the real Fugaku hardware-barrier device and never
silently falls back:

```sh
a64fx/dequant-pipe/bench_dequant_pipe \
  --sync hwbar --format int4 --path w8a8 --m 1 \
  --n 49152 --k 8192 --chunk-kib 64 --iterations 10 --trials 3 --peak
```

The benchmark allocates six independent barrier descriptors and assigns one
window to each thread in a pair.  A `dmb ish` brackets every handoff.  If
`/dev/xos_hwb` or a blade is unavailable, headline mode exits with an explicit
error.  `--sync atomic` is the portable control and is always labeled
`atomic-control` in output; its numbers are not hardware-barrier results.

Useful sweeps:

```sh
# M and handoff-amortization sweep
a64fx/dequant-pipe/bench_dequant_pipe --sync atomic --format int4 \
  --path w8a8 --sweep-m --sweep-chunks --n 12288 --k 4096 \
  --iterations 3 --trials 1

# Force the FP16 safety gate to demonstrate INT16 fallback
a64fx/dequant-pipe/bench_dequant_pipe --sync atomic --format fp4 \
  --path auto --unsafe-scale --m 1 --n 384 --k 256 \
  --iterations 1 --trials 1 --verify
```

See [RESULTS.md](RESULTS.md) for the measurements from the implementation run.
