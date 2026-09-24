# Qwen3.8 FP4 execution layouts: 200+ GB/s on A64FX

The half-predecoded 6-bit payload reaches **219 GB/s on one CMG** and
**852–859 GB/s across four CMGs** on native A64FX. It achieves 95–99% of a
matched scan's bandwidth, depending on shape. Projection time improves by
6–9% over the compact integer-scale kernel; the larger byte count accounts
for much of the reported bandwidth increase. These are synthetic projection
measurements, not a real-model decode/token-rate result.

## Representations and accuracy

`qwen38_nvfp4_expanded_a8.c` adds two execution layouts. Each tile represents
eight output rows and 64 input columns (512 weights):

| Layout | Payload | Scale metadata | Tile bytes | Total bits/weight |
| --- | ---: | ---: | ---: | ---: |
| Existing integer-scale FP4 | 256 B | 96 B | 352 | 5.5 |
| Signed 5-bit | 320 B | 96 B | 416 | 6.5 |
| Half-predecoded 6-bit | 384 B | 96 B | 480 | 7.5 |

The 5-/6-bit names describe the **payload**, not payload plus metadata.
Metadata remains four sets of duplicated INT8 scale multipliers and eight
FP32 base scales. The 5-bit payload stores decoded signed FP4 values in
low-nibble and sign-predicate planes; predicated subtraction replaces table
lookups. The 6-bit payload expands two of the four subblocks to signed INT8
and retains FP4 codes for the other two. It removes half the FP4 lookups.

Both repacks preserve exactly the existing integer-scale representation;
they add no further weight quantization. That existing representation rounds
source scale/base ratios and rejects ratios outside INT8 range. Its accuracy
on arbitrary real-model scales remains unvalidated. Randomized integer-ratio
scales, zero scales, all FP4 codes/signs, and INT8 activation extremes match
an independent source-weight reference within `1e-5 * (1 + abs(reference))`
and the compact kernel bitwise.
Negative scales are rejected. All-row QEMU and native checks also pass.

## Native results

Interactive Fugaku job `51891531`, node `a27-0006c`, `freq=2000,eco_state=0`.
Fujitsu `fcc -Nclang`, O3 kernels, O1 benchmark setup; 1,000 measured passes,
two warmups, three repeats. Times and logical source-byte bandwidth below
use full earliest-start/latest-finish makespan. Repack and correctness checks
are outside the timed region.

One CMG, 6144 x 5120, medians:

| Layout | Projection time | Source GB/s | Matched scan GB/s |
| --- | ---: | ---: | ---: |
| Compact integer scale | 146.954 us | 147.167 | 226.426 |
| Signed 5-bit | 140.502 us | 181.912 | 226.176 |
| Half-predecoded 6-bit | 134.666 us | 218.995 | 226.762 |

For 17408 x 5120 on one CMG, the 6-bit layout reaches 218.709 GB/s and
382.053 us versus 410.286 us for compact integer scale. It increases tile
storage by 36.4% relative to integer scale (25% versus the original 384-byte
FP4/FP32-scale layout). The 5-bit layout increases storage by 18.2% relative
to integer scale but does not reach 200 GB/s.

Four CMGs / 48 workers, medians:

| Shape | Compact GB/s | 6-bit GB/s | 6-bit scan GB/s | Compact / 6-bit time |
| --- | ---: | ---: | ---: | ---: |
| 17408 x 5120 | 584.177 | 852.319 | 877.242 | 104.893 / 98.037 us |
| 15360 x 5120 | 588.439 | 858.609 | 871.028 | 91.882 / 85.869 us |

Each CMG's weight segment exceeds its 8 MB L2. All 24 four-CMG runs returned
`correct=1`, including every output row, and verified the segments on HBM
nodes 4, 5, 6, and 7 respectively.

## HBM placement correction

Pin the initializing thread **before allocation**, not only before the
kernel. In this job a wrapped launch initialized on CPU 35 and all sampled
weight pages were on node 5, despite a successful `mbind(..., node 4, flags=0)`.
Workers pinned to CPUs 12–23 then scanned at about 120 GB/s. A direct launch
initialized on CPU 14, placed pages on node 4, and scanned at 225 GB/s.
`mbind` without migration does not relocate already populated pages.

The benchmark now pins initialization to CPU `12 + 12*cmg` before allocation
and retains `mbind` and local first touch. Set `Q38_QLAIR_DIAG_PLACEMENT=1` to
query a page every 2 MiB after initialization and fail on unexpected node
placement. Wrapped launches now reproduce the local scan rate. Earlier logs
without physical-page checks cannot reliably separate kernel changes from
placement changes. `Q38_QLAIR_NO_MBIND=1` disables the initialization NUMA path
for QEMU correctness runs; it is not a native performance configuration.

## FAPP confirmation

The corrected run collected PA1/2/6/10 and PA17. Link markers with
`-lfjprofcore`; each pthread uses its own region name. Median per-worker
counts over 1,000 passes of 6144 x 5120:

| Event | Compact | 6-bit |
| --- | ---: | ---: |
| Effective instructions, PA1 | 513.75 M | 412.70 M |
| SIMD instructions, PA1 | 430.54 M | 321.28 M |
| SIMD/FPU commit wait / cycles, PA6 | 43.05% | 34.12% |
| L2-miss commit wait / cycles, PA6 | 0.37% | 12.60% |
| SVE register LDR / STR events, PA10 | 35.90 M / 15.36 M | 11.49 M / 5.08 M |

The 6-bit layout reduces instruction pressure and becomes sensitive to weight
supply. PA17 `BUS_READ_TOTAL_MEM * 256 / elapsed` gives **220.57 GB/s** for
6-bit compute and **226.26 GB/s** for its scan, versus 149.95 GB/s for compact
compute. These are medians of overlapping windows of a **CMG-wide** counter;
the twelve worker counts are not summed. This conversion is also used by
`a64fx/preport/preport.js`.

Logs, native assembly, and raw PA CSVs are in
`tmp/q38-expanded-20260924/hw-51891531-local/`. The earlier `hw-51891531/`
directory retains the deliberately investigated remote-placement results.

## Simulator status

The local `~/work/clair/a64fx/build-inference/qlair` passes the new kernels'
full-size numerical checks, but does **not** establish the 200 GB/s gate:

| Clang-generated kernel, 12 workers | Simulated HBM GB/s |
| --- | ---: |
| Compact integer scale | 105.95 |
| Signed 5-bit | 116.23 |
| Half-predecoded 6-bit | 134.63 |
| 6-bit byte scan | 227.58 |

The 5-bit leaf function's single-register pre/post-index stack save/restore
caused repeated calls in qlair's main-thread execution. A standard
`-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer` build completed correctly.
QEMU and hardware pass both builds. Importing the actual Fujitsu compute
assembly (with Clang setup/repack code) also passes qlair correctness but
predicts only 30.42 GB/s for 6-bit and 43.05 GB/s for compact. Fujitsu's full
setup/repack assembly fails earlier in the simulator. These differences need
an emulator/timing-model investigation; native results must not be presented
as simulated results or used to silently retune the model.

## Reproduce

Inside a direct interactive A64FX allocation, build from the repository:

```sh
mkdir -p /local/q38-layout-build
export TMPDIR=/local/q38-layout-build
make -C a64fx/llm CC=fcc BUILD=/local/q38-layout-build \
  qwen38_nvfp4_layout_bench qwen38_nvfp4_expanded_test
/local/q38-layout-build/test_qwen38_nvfp4_expanded_a8
Q38_QLAIR_DIAG_PLACEMENT=1 Q38_QLAIR_CHECK_ALL=1 \
  /local/q38-layout-build/bench_qwen38_nvfp4_qlair packed6_1 6144 5120 12 1000 2
Q38_QLAIR_DIAG_PLACEMENT=1 Q38_QLAIR_CHECK_ALL=1 \
  /local/q38-layout-build/bench_qwen38_nvfp4_qlair packed6_1 17408 5120 48 1000 2
make -C a64fx/llm CC=fcc BUILD=/local/q38-layout-build Q38_FAPP=1 \
  qwen38_nvfp4_layout_bench
fapp -C -d /local/q38-layout-pa17 -Hevent=pa17 \
  /local/q38-layout-build/bench_qwen38_nvfp4_qlair_fapp packed6_1 6144 5120 12 1000 2
```

Use `packed8_iscale1`, `packed5_1`, and their `_stream` controls for comparison.
For real-model integration, read staged `/local` weights in bounded chunks
and repack directly into NUMA-local final storage. Do not hold both full
source and expanded copies in 32 GB HBM. Real-scale accuracy, resident-memory
headroom, and complete decode throughput still need validation; **40 tok/s is
not demonstrated** by these kernel measurements.
