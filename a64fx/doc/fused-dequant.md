# Fused dequantization and GEMV/GEMM on A64FX

LLM decode usually applies a large weight matrix to one token at a time. That
is a GEMV, or a very small GEMM when tokens are batched or verified together.
On A64FX the compressed weights, not the multiply count, are normally the
scarce resource: every decode step streams most of the model from HBM2. A good
low-bit kernel therefore has a simple rule:

> Load each packed weight once, expand it in SVE registers, consume it
> immediately, and write only accumulators or final output.

This chapter develops that rule from standalone dequantization through a fused
INT4-to-SDOT kernel. The examples are real kernels and measurements from this
repository, not a claim that one layout fits every quantization format.

## The machine-level constraint

An A64FX core has 512-bit SVE vectors: one vector holds 64 `int8`, 32 `int16`,
or 16 `float32` elements. Twelve compute cores share an 8 MiB L2 and one HBM2
stack in each Core Memory Group (CMG). Cache lines are 256 bytes. At 2.0 GHz,
the documented local memory-to-L2 ceiling is 256 GB/s per CMG; the measured
sequential-read plateau in this work was 229.96 GB/s.

The useful decode metric is **packed bytes per second**. Reporting expanded
bytes makes a dequantizer look faster without reducing model-read time. For
example, 230 GB/s of packed INT4 represents 460 GB/s after expansion to INT8,
before the matrix kernel reads those expanded bytes again.

Decode also has little natural instruction-level parallelism. With one token,
an output row has one dependency chain. The kernel must create independent
chains across output columns or adjacent quantization blocks; increasing the
nominal GEMM `M` is not an option when the application supplied only one row.

## Quantized arithmetic

For block-quantized weights and activations, a typical output is

```text
y[n] = sum_b (dw[b,n] * da[b])
             * dot(qw[b,n,:], qa[b,:])
```

where `qw` and `qa` are small integers and `dw` and `da` are floating-point
scales. The integer dot product should remain in `int32` for a 32- or
128-element block, then be converted and scaled into an FP32 accumulator. Do
not round every individual weight to floating point if one conversion per
block is sufficient.

The experimental INT4 format here stores two signed two's-complement values
in each byte, each in `[-8, 7]`. The FP4 experiment uses E2M1 values represented
by the exact doubled integer lattice `{0,1,2,3,4,6,8,12}` plus sign; the factor
of one half is folded into the block scale. This permits the same SDOT-based
consumer after a table lookup.

## A fast standalone dequantizer

Standalone conversion is appropriate when the expanded representation is
reused many times, when converting a model once into a decode sidecar, or when
it is a correctness reference. It is usually the wrong inner loop for
single-token decode.

The SVE INT4-to-INT8 loop in `a64fx/dequant-pipe/kernels.S` processes 64 packed
bytes at a time:

```asm
ld1b    {z0.b}, p0/z, [src]     // 128 nibbles
mov     z1.d, z0.d
and     z1.b, z1.b, #0x0f       // low nibble
lsr     z2.b, z0.b, #4          // high nibble
lsl     z1.b, z1.b, #4
asr     z1.b, z1.b, #4          // sign-extend low
lsl     z2.b, z2.b, #4
asr     z2.b, z2.b, #4          // sign-extend high
zip1    z3.b, z1.b, z2.b
zip2    z4.b, z1.b, z2.b
st1b    {z3.b}, p0, [dst]
st1b    {z4.b}, p0, [dst, #1, mul vl]
```

The important details are:

- Expand a full 64-byte vector, not scalar nibbles.
- Use shifts for sign extension. The low nibble is moved into the sign
  position and shifted back arithmetically.
- Interleave low and high results with `zip1`/`zip2` so their order matches the
  consumer's K layout.
- Advance the source by 64 bytes and the INT8 destination by 128 bytes.
- Keep the loop sequential and aligned. A64FX's stream prefetcher performed
  better here than an explicit `pldl1keep` eight cache lines ahead.

FP4 uses the same nibble split followed by SVE `tbl` against a 16-entry lookup
vector. INT16 output adds `sunpklo`/`sunpkhi` and writes four vectors for each
input vector. FP16 output widens, converts with `scvtf`, multiplies by the
block scale, and stores the result. Those wider destinations are progressively
more expensive: packed INT4 to INT8 creates 2× output traffic, while INT16 or
FP16 creates 4×.

The repository self-test exhaustively checks all 16 nibble codes and validates
the INT8, INT16, and FP16 paths against scalar references. That exhaustive
code-point test is worth keeping even when the production kernel is fused.

## Why a producer/consumer pipeline loses

A tempting design assigns some cores to dequantization and other cores to
GEMV. The producer writes alternating expanded buffers into the shared L2;
the consumer reads them and executes SDOT. Chunking amortizes synchronization,
and A64FX hardware barriers can make the handoff cheap, but neither removes
the expanded stores and loads.

The measured producer-only control makes the limit clear. In one CMG, an
eight-core packed read reached 229.96 GB/s. Adding a 1× L2 copy reduced it to
187.00 GB/s. Expanding INT4 to INT8 into a double ring reached only 110.34
GB/s of packed input while writing 220.68 GB/s of expanded data. The complete
staged M=1 W8A8 path reached 64.28 GB/s.

This is a traffic problem, not primarily a barrier problem. At the target
packed rate, the standalone producer would have to write roughly 460 GB/s to
L2, after which the consumer must read the same expanded stream. Fusion is the
way to remove both transfers.

## The register-fused INT4 GEMV

The direct kernel loads packed bytes from HBM, expands both nibbles in SVE
registers, immediately executes `sdot`, and never materializes expanded
weights. The inner operation is conceptually:

```text
packed = load_64_bytes(weight)
lo = sign_extend(packed & 0x0f)
hi = arithmetic_shift_right(packed, 4)
(w0, w1) = zip(lo, hi)
a = broadcast_4_activation_bytes()
acc0 += sdot(w0, a)
acc1 += sdot(w1, a)
```

The implementation loads two 64-byte vectors per block step, uses
`lsl`/`asr` to obtain signed low nibbles, `asr` for signed high nibbles, and
four ZIP results to feed four SDOT accumulators. The activation load is an
`ld1rw`: four adjacent INT8 activations are replicated across the SVE vector.
Each `sdot z.s, w.b, a.b` accumulates groups of four byte products into sixteen
32-bit lanes.

One block alone does not expose enough independent work for M=1. The kernel
therefore fuses four adjacent K=128 blocks and holds **16 independent SVE
accumulators**. This hides SDOT and expansion latency without inventing extra
tokens. The best layout is a four-block K-major supertile: for each K quartet,
the 128 packed bytes from each of four blocks are adjacent. The loop sees one
sequential 512-byte stream instead of four streams separated by 4 KiB.

Layout is part of the kernel ABI. A converter or model packer must emit the
same nibble order, K-major block order, scale order, padding, and tail rules
that the assembly expects. Converting inside the hot loop to repair an
inconvenient file layout usually gives back the bandwidth gained by INT4.

## From the benchmark kernel to a scaled LLM kernel

The direct fused bandwidth probe intentionally omits block scales. It measures
the packed-load, nibble-expansion, and SDOT schedule, so it is an upper bound
for a production quantized GEMV. A complete kernel needs an epilogue at each
quantization block:

1. Keep integer partial sums in `int32` accumulators.
2. Convert them to FP32 with `scvtf`.
3. Multiply by weight and activation scales.
4. Accumulate into persistent FP32 output registers.
5. Store output only after the K loop, or at a deliberately chosen K-panel
   boundary when register pressure requires it.

`kernel_q8v2_3x4.S` shows this pattern for a production-shaped Q4_0-derived
GEMM. Its tile is three token rows by 64 output columns. For each K=32 block it
uses 12 INT32 partial vectors, applies per-column weight and per-row activation
scales, and retains 12 FP32 accumulators across blocks. The `_arow` variant
factors a per-row activation scale out of the K loop, saving one scale
operation per block when that quantization contract is acceptable.

For true on-the-fly fusion, insert the nibble expansion before those SDOTs and
budget registers for both expansion temporaries and accumulators. If register
pressure becomes excessive, reduce the token tile before reducing independent
output-column chains. For decode, latency hiding across N is more valuable
than a large M tile.

## GEMV versus small GEMM

The appropriate tile changes with token count:

- **M=1 decode:** fuse adjacent N/K blocks to create 16 or more independent
  accumulator vectors. Optimize packed-byte bandwidth and latency.
- **M=2 or M=3 verification:** reuse each expanded weight vector across token
  rows. Activation traffic grows slowly, while each expansion feeds multiple
  SDOTs.
- **Larger prefill GEMM:** ordinary M×N register blocking supplies enough
  independent accumulators; per-block conversion and scale-out can become the
  ceiling rather than HBM.

The measured L2-resident M=6 load-plus-SDOT loop sustained 1.8394 SDOT per
cycle, 91.97% of the two-per-cycle architectural peak. That rate does not mean
M=1 can do the same: the M=6 kernel has 24 independent accumulators, whereas a
naive M=1 tile has only four.

At 230 GB/s and 2.0 GHz, a 256-byte packed line arrives at one of `C` equally
loaded cores every `2.226*C` cycles. INT4 expansion costs about 20 SVE integer
instructions per line and M=1 needs eight SDOTs. Using the measured issue rate:

```text
cycles/line ~= (20 + 8*M) / 1.8394
M=1: 15.22    M=2: 19.57    M=4: 28.27    M=6: 36.97
```

Eight fused cores are therefore the minimum plausible full-bandwidth INT4
M=1 configuration in a CMG; nine or ten leave practical headroom. FP4 table
mapping raises the expansion budget, making ten to twelve cores more
realistic. M≥4 cannot retain 230 GB/s under this instruction model even with
all twelve cores, but it performs more useful arithmetic per loaded weight.

## Measured outcomes

All figures below are packed input rates on one 2.0 GHz CMG unless noted.

| Path | Median | Interpretation |
|---|---:|---|
| Sequential HBM read, 8 cores | 229.96 GB/s | Packed-read ceiling |
| Standalone 2× expansion stores, 8 cores | 110.34 GB/s | L2 store ceiling |
| Staged INT4 W8A8, M=1 | 64.28 GB/s | Expansion + ring + consumer |
| Direct fused INT4→INT8 SDOT, 12 cores | 138.35 GB/s | 2.16× staged |
| Direct fused INT4→INT16 SDOT, 12 cores | 67.55 GB/s | 1.93× staged |
| Four-block supertile, favorable allocation | 228.84 GB/s | Unscaled probe; matched paired read |

A later W4A16 extension replaced the INT16 kernel's separated block streams
with a two-block K-major supertile and added E2M1 FP4 plus FP16-FMA routes. On
controlled XOS 2 MiB pages, with an INT4 W4A8 control at 230.09 GB/s, the new
INT4/INT16 path reached 147.00 GB/s and FP4/INT16 reached 132.44 GB/s. Direct
FP16-FMA paths reached 96.28 GB/s for INT4 and 84.59 GB/s for FP4. These are
unscaled upper bounds; the FP16 probe accumulates in FP16 and needs an
overflow/accuracy gate before production use. The results also show that
W4A16 does not saturate the 230 GB/s packed-read interface: doubling the dot
work plus widening/conversion moves the bottleneck back to instruction issue
and dependency latency.

The last result needs careful interpretation. Separate allocations showed both
roughly 120 and 229 GB/s states. A controlled sweep later found that changing
a 16 KiB inter-core gap did not select the state; physical allocation and page
placement were stronger variables. With Fujitsu XOS 2 MiB pages, the controlled
12-core sweep sustained 227.76--229.71 GB/s across the tested skews. Treat
16 KiB as an experimental observation, not a decoded HBM channel formula.

Likewise, 228.84 GB/s is not end-to-end LLM throughput: that probe omits scale
loads, integer-to-float conversion, tails, dispatch, and layer integration.
The robust conclusion is that the K-major supertile and a good allocation can
let expansion plus SDOT fit under the memory deadline.

## Placement and threading

Partition work so every core owns disjoint output rows or tiles and reads a
sequential packed stream. Bind the allocation and first-touch to its target
CMG; remote-CMG traffic crosses the coherent ring instead of using the local
L2-to-HBM path. Preserve 256-byte line alignment and the supertile alignment.

A64FX L2 is physically indexed and hashed, with `PA[8]` selecting one of two
banks. The HBM channel/bank mapping is not public. Consequently, benchmark
paired read and compute kernels on the same arena, repeat across allocation
epochs, and prefer controlled huge-page placement where available. Do not
label a virtual-address stride as an HBM color without physical-address or PMU
evidence.

For a whole node, repeat the ownership scheme independently in all four CMGs.
Avoid a shared dynamic queue that lets one CMG consume another CMG's weight
pages. Persistent workers and stable row ownership also avoid redispatch and
first-touch surprises on every layer.

## A practical implementation checklist

1. Define the quantization equations, scale granularity, nibble sign rule, and
   exact tail behavior before writing assembly.
2. Write a scalar reference and exhaustively test all packed code points.
3. Benchmark raw packed reads with the exact thread placement and arena used
   by the compute kernel.
4. Build the standalone SVE dequantizer; use it as a converter and reference.
5. Measure producer-only expanded stores. If they consume the L2 budget, do
   not spend time polishing the handoff barrier.
6. Fuse expansion and dot product in registers. Create M=1 independence by
   spanning adjacent output/K blocks.
7. Repack into one sequential K-major stream; make that layout a versioned
   contract rather than an undocumented assumption.
8. Add block scale-out, FP32 accumulation, tails, and the unchanged fallback.
9. Validate exact integer partials and tolerance or token-level behavior after
   floating-point scale-out.
10. Report packed GB/s, end-to-end token rate, memory residency, and paired
    read bandwidth. A microkernel number alone is not a deployment result.

## Reproducing the repository experiments

On a native A64FX node with Fujitsu `fcc` and `libhwb`:

```sh
make -C a64fx/dequant-pipe
make -C a64fx/dequant-pipe test

a64fx/dequant-pipe/bench_fused_sdot --verify --path int8 \
  --cores 12 --mib 240 --iterations 10 --trials 3

a64fx/dequant-pipe/bench_fused_sdot --verify --kernel super \
  --cores 12 --mib 240 --skew-kib 16 --iterations 10 --trials 5
```

The source and full measurements are in
[`a64fx/dequant-pipe/kernels.S`](../dequant-pipe/kernels.S),
[`a64fx/dequant-pipe/README.md`](../dequant-pipe/README.md), and
[`a64fx/dequant-pipe/RESULTS.md`](../dequant-pipe/RESULTS.md). The scaled
Q8v2 GEMM example is
[`a64fx/gemma4-kernels/kernel_q8v2_3x4.S`](../gemma4-kernels/kernel_q8v2_3x4.S),
and the placement details are in
[`a64fx/doc/a64fx_memory_addressing.md`](a64fx_memory_addressing.md).
