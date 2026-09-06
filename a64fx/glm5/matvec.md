# A64FX GLM-5 w8a16 matvec tuning

## Status (2026-07-18)

The 800 GB/s target is reached by the honest one-stream HBM diagnostic: **807.4 GB/s
best, 802.1 GB/s mean** over a 3.996 GiB working set. The bit-exact production
`int16-SDOT` kernel reaches **388.6 GB/s best, 388.0 GB/s mean**. Its current limiter is
the SDOT/scale instruction path, not an eight-stream memory ceiling.

The best bit-exact candidate changes software prefetch from every 64-byte quant group to
once per 256-byte A64FX cache line, 512 bytes ahead. It reaches **427.1 GB/s best,
426.5 GB/s mean**, or **1.099x** the production kernel. This is a useful result but does
not clear the agreed promotion gate of **1.30x honest kernel throughput plus 5% median
decode**. Therefore `common/glm5_int8.h` is intentionally unchanged and no end-to-end
decode claim is made.

The reproducible benchmark and all explored exact candidates are in
`a64fx/glm5/test_matvec_int16sdot.c`.

The strict-exact follow-up is also complete. Four-phase scheduling, split-half dots, an
8-row/cache-line-interleaved layout, and a hand-written packed SVE kernel all remained
below the row-major prefetch candidate. The compressed follow-up produced a custom Q6A8
group-128 kernel at **287.5 physical GB/s / 368.0 logical GB/s best**. This is a useful
compressed prototype but does not reach 600 logical GB/s and changes the numerical
contract (2.35% synthetic output relative L2 versus W8A16), so it is not promoted.

## 1. Kernel and numerical contract

`glm5_matvec_int16sdot_8row` is in `common/glm5_int8.h`. It computes eight output rows:

```
y[r] = sum_c (w[r,c] - 128) * xq[c] * scale[r,group(c)]
```

- `w` is row-major offset-binary int8, stored as one byte per element.
- `xq` is a shared int16 activation vector.
- The default quantization group is 64 columns with one f32 scale per row/group.
- `xgsum` holds the int64 group sums of `xq`. Folding `-128*xgsum` lets the inner loop
  use `svld1ub_u16` followed by `svdot_s64` without a vector subtract.
- Per-group integer dots are converted to f64 and scaled in the existing order. Exact
  candidates must produce the same eight f32 output bit patterns.

The main benchmark shape is `[2048,6144]`, group size 64. It is representative of the
GLM-5 dense decode projections and matches the eight-row partition used by
`glm5_i16_worker`.

## 2. Critical benchmark correction

The old handoff reported a roughly 190 GB/s full-kernel result and a roughly 215 GB/s
"eight-stream structural ceiling." Those numbers were caused by a first-touch ownership
mismatch in the benchmark, not by the kernel structure.

The old initializer ran one static OpenMP loop over all concatenated repeated tensors.
A thread consequently first-touched large cross-tensor ranges, while the timed loop
partitioned the eight-row blocks of *every tensor* over all threads. Most timed reads were
therefore remote from the CMG that owned their pages. The artifact is directly reproduced
with `GLOBAL_TOUCH=1`:

| 1.992 GiB, 47 threads | best GB/s | mean GB/s |
|---|---:|---:|
| one contiguous stream, mismatched global touch | 203.5 | 202.8 |
| production kernel, mismatched global touch | 176.4 | 175.8 |

The corrected default first-touches each repeated tensor with the same per-tensor
eight-row-block partition used by the timed worker. Production conversion also handles
each tensor separately and distributes its rows across the OpenMP team, so the corrected
placement reflects its per-tensor CMG distribution far better than the concatenated loop.

Three rules remain mandatory:

1. Store and sink all eight outputs. Sinking only a subset lets the compiler delete work.
2. Use at least 1 GiB of fresh anonymous weights. A single 12.6 MB matrix fits in aggregate
   L2 and is not an HBM benchmark.
3. First-touch each tensor with the same ownership pattern used to read it. NUMA placement
   is part of the measurement.

## 3. Corrected ceilings and baseline

Command settings were Fujitsu clang mode, `OMP_NUM_THREADS=47`, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, five repetitions, a fresh 3.996 GiB anonymous weight allocation, and
reader-matched first-touch. GB/s counts weight bytes, matching the decode optimization
objective; the full kernel additionally reads f32 scales amounting to 1/16 of weight size.

| mode | what it measures | best GB/s | mean GB/s |
|---|---|---:|---:|
| `stream1` | all weight bytes as one contiguous stream, eight load accumulators | **807.4** | **802.1** |
| `raw8` | eight row streams, loads/adds only | 667.4 | 665.0 |
| `sdot8` | eight row streams plus int16 SDOT, no scale epilogue | 485.2 | 483.8 |
| `baseline` | production SDOT + f64 scale/bias path | 388.6 | 388.0 |
| `pf512i256` | best exact prefetch candidate | 427.1 | 426.5 |

Interpretation:

- Eight independent row streams retain about 83% of the one-stream HBM diagnostic. The
  old 215 GB/s structural ceiling is refuted.
- Adding SDOT reduces throughput to about 73% of the raw-eight-stream ceiling.
- The scale/bias path reduces it to about 80% of the SDOT-only ceiling.
- The 800+ GB/s physical diagnostic target is real, but the current exact instruction mix
  cannot approach it. Even deleting the entire scale epilogue would leave the measured
  483.8 GB/s SDOT ceiling below `1.30 * 388.0 = 504.4 GB/s`.

## 4. Exact candidate results

All candidates passed 1,824 bitwise comparisons against
`glm5_matvec_int16sdot_8row`: three column sizes, two aligned group offsets, activation
trials including large signed values, and both row-major and packed-layout candidates.
The candidates fall back to the production function outside their specialized
group-64/aligned case.

### Software prefetch

The production loop emits eight `PLDL1KEEP` instructions per 64-byte group, 1024 bytes
ahead. A64FX L1/L2 cache lines are 256 bytes, so four successive groups prefetch addresses
within the same line. Retuning distance and issue interval found:

| candidate | working set | representative result |
|---|---:|---:|
| no software prefetch | 1.992 GiB | 339.5 GB/s best |
| 256 B ahead, every group | 1.992 GiB | 394.2 GB/s best |
| 512 B ahead, every group | 1.992 GiB | 418.2 GB/s best |
| 1024 B ahead, every group | 1.992 GiB | 385.7 GB/s best |
| 2048 B ahead, every group | 1.992 GiB | 325.0 GB/s best |
| 4096 B ahead, every group | 1.992 GiB | 304.5 GB/s best |
| 512 B ahead, every 128 B | 3.996 GiB | 417.2 GB/s mean |
| **512 B ahead, every 256 B** | **3.996 GiB** | **426.5 GB/s mean** |
| 512 B ahead, every 512 B | 3.996 GiB | 311.7 GB/s mean |
| 768 B ahead, every 256 B | 3.996 GiB | 421.7 GB/s mean |

The cache-line-aware candidate is a repeatable 9.9% kernel improvement, but the agreed
1.30x gate deliberately rejects this marginal production change.

### Load scheduling and wider loads

- Explicit batches of 2/4/8 weight vectors reached only about 361--367 GB/s. Fujitsu clang
  generated a vector spill for the most aggressive schedule.
- Loading 64 weight bytes and widening with `uunpklo/uunpkhi` halved the number of load
  instructions but added 16 unpack operations per loop body and reached about 335 GB/s.
- Removing the general divide/control path did not help; the no-prefetch exact clone was
  slower than production.

Fresh assembly inspection showed:

| function | static SDOT | static PRFM | SVE spill | other |
|---|---:|---:|---:|---|
| production baseline | 8 | 8 | no | one `sdiv` |
| no-prefetch exact clone | 8 | 0 | no | no `sdiv` |
| cache-line prefetch clone | 8 | 8 (executed every fourth group) | no | no `sdiv` |
| 64-byte-load clone | 16 | 0 | no | 16 `uunpk*` |
| eight-load batch | 8 | 0 | yes | accumulator spill/reload |

### Strict blocking and interleaved layout

The final exact phase tested alternatives that preserve every f64 group fold and final
f32 bit pattern:

| candidate | layout / scheduling | representative mean GB/s |
|---|---|---:|
| `phase4` | four row streams at a time | 375.9 |
| `panel2` | two independent half-group dots | 394.1 |
| `packedraw` | 8 rows interleaved per 32-column/256-byte panel, loads only | 805.6 |
| `packed` | packed intrinsic exact kernel | about 241--253 |
| `packedwide` | packed layout with wider scheduling | 282.8 |
| `packedasm` | hand-written, spill-free packed SVE | 269.6 |

`packedraw` proves the interleaved layout can feed HBM at the one-stream rate, but the
exact packed compute kernels lose badly. The best packed assembly result was 272.5 GB/s;
removing compiler spills did not recover the decode/scale cost. The row-major
`pf512i256` result therefore remains the strict-exact winner.

## 5. Q6 compressed experiment

`test_matvec_q6.c` and `matvec_q6_a8.S` implement a purpose-built Q6 format. For the
default group size 128, each block stores 64 bytes of low nibbles, 32 bytes of packed high
two-bit fields, and one f32 scale: **100 bytes / 128 weights = 0.78125 B/weight**. Codes
are offset binary 0..63 for signed values -32..31.

The optimized path quantizes the shared activation to int8 per weight group, expands one
64-weight Q6 chunk to a full SVE byte vector, subtracts the Q6 bias in the integer domain,
and uses one int8 SDOT per chunk. The hand-written kernel keeps eight row accumulators in
registers, matches the intrinsic A8 implementation bit-for-bit in the quality suite, and
prefetches every row stream 376 bytes ahead.

The quality test requantizes synthetic groupwise W8 weights to Q6 and compares matvec
outputs to the existing W8A16 kernel over 256 rows and eight activation trials. It is a
kernel-level signal, not model perplexity or task validation.

| Q6 path | bytes/weight | output rel. L2 | best physical GB/s | best logical GB/s |
|---|---:|---:|---:|---:|
| group-64, A16 | 0.81250 | 2.174% | 163.0 | 200.6 |
| group-64, A8 assembly, no prefetch | 0.81250 | 2.196% | 172.6 | 212.4 |
| group-64, A8 assembly, tuned prefetch | 0.81250 | 2.196% | 265.9 | 327.3 |
| **group-128, A8 assembly, tuned prefetch** | **0.78125** | **2.352%** | **287.5** | **368.0** |
| group-256, A8 assembly | 0.76563 | 2.528% | 265.4 | 346.6 |

Physical bandwidth counts the actual packed allocation. Logical bandwidth counts one
byte per original W8 weight, so it measures effective W8-equivalent weight throughput.
A raw group-64 packed stream reaches 784.7 physical / 965.8 logical GB/s, but unpack,
SDOT, and scale folding reduce the useful group-128 kernel to 368.0 logical GB/s. Padding
blocks to 56 or 64 bytes left logical throughput near 327--328 GB/s for group 64, which
confirms an instruction-throughput ceiling rather than a packed-byte HBM ceiling.

Group 128 is the best measured compromise: halving scale folds versus group 64 adds 12%
logical throughput at a modest synthetic error increase. Group 256 lengthens the
dependent dot chain and regresses. Even the best result is only 61% of a 600 GB/s logical
target, so this Q6 representation should remain experimental pending real-model quality
validation and a materially cheaper unpack strategy.

## 6. PMU status and bottleneck conclusion

The harness has optional `fapp_start`/`fapp_stop` markers (`GLM5_MATVEC_FAPP`) and the
intended raw event sets cover cycles, L1/L2/memory waits, prefetch-port waits, demand and
hardware-prefetch refills, outstanding requests, and local memory-bus reads.

Hardware counters were unavailable in the interactive allocation. Normal collection
failed with:

```
Internal error(krm_init failed : -5)
PAPI_thread_init error. function error code = -11
```

`method=fast` also failed (`PAPI_set_domain ... -16`) and terminated inside FAPP. PMU
collection must therefore be repeated in a counter-enabled compute/batch allocation; no
counter values are claimed here.

The counter-independent ceilings still classify the bottleneck reliably enough for the
promotion decision: raw eight-stream loads reach 665 GB/s, while SDOT-only reaches
484 GB/s and the full kernel 388 GB/s. The dominant remaining opportunity is reducing or
restructuring the integer-dot and exact scale work, not increasing HBM stream bandwidth.

## 7. Reproduction

Build and check:

```sh
cd a64fx/glm5
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
  -Wall -Wextra -Wpedantic -I../../common \
  test_matvec_int16sdot.c matvec_int16sdot_packed.S -lm -o test_matvec_int16sdot
MODE=check OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./test_matvec_int16sdot
```

Run a 4 GiB mode:

```sh
MODE=baseline WEIGHT_BYTES=4294967296 REPS=5 \
  OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./test_matvec_int16sdot
```

Useful `MODE` values include `stream1`, `raw8`, `sdot8`, `baseline`, `pf512i256`,
`fast1`, `wide64`, and `load2`/`load4`/`load8`. Set `GLOBAL_TOUCH=1` only to reproduce
the obsolete placement artifact.

Build, check, and run the default Q6 group-128 assembly path:

```sh
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
  -Wall -Wextra -Wpedantic -I../../common \
  test_matvec_q6.c matvec_q6_a8.S -lm -o test_matvec_q6
MODE=check ./test_matvec_q6
MODE=q6a8asm Q6_BYTES=4294967296 REPS=7 \
  OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./test_matvec_q6
```

Compile with `-DQ6_GROUP_SIZE=64` or `-DQ6_GROUP_SIZE=256` to reproduce the rejected
group-size variants. Q6 modes include `raw`, `q6a16`, `q6a8`, and `q6a8asm`.

For a counter-enabled allocation, build with `-DGLM5_MATVEC_FAPP`, set `REPS=1`, and
collect one mode per FAPP run. Raw event names and encodings are documented in
`a64fx/doc/a64fx_pmu_events.csv`; profiler procedure is in
`a64fx/doc/fapp_pmu_profiling.md`.

## 8. Promotion gate and next work

Production remains unchanged because no candidate reached both gates:

- at least **1.30x** on this honest large-working-set kernel benchmark; and
- at least **5% median end-to-end decode** improvement.

The strict-exact blocking/assembly route is exhausted by the candidates above. PMU data
from a counter-enabled job could still identify a different instruction-level direction,
but exact W8A16 remains capped well below 600 GB/s by measured SDOT and scale work. The
current Q6 unpack path also remains below 600 logical GB/s and has not passed a real-model
quality gate. Do not revive the old 190/215 GB/s premise or compare candidates using
concatenated global first-touch.
