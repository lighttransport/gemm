# A64FX FP4 GEMM M>1 Optimization Plan

## Objective

Raise the one-CMG (12-core) MXFP4/NVFP4 GEMM rate above 800 effective
GFLOP/s for `M >= 8`, while retaining FP16 products and FP16 accumulation
with optional FP32 promotion every `promotion_k` values.  Effective GEMM work
is always counted as `2*M*N*K`, independent of the stored weight format.

The first implementation target is the persistent decoded-weight path.  It
still represents an FP4 model and its decoded sidecar is prepared outside the
timed repeated GEMM, but its 4x weight expansion must be reported explicitly.
The packed four-bit path remains the capacity path and must not regress.

## Roofline

### Compute peak

A64FX executes two 512-bit FP16 FMA instructions per cycle.  Each instruction
updates 32 FP16 lanes and counts as 64 operations:

```
64 op/instruction * 2 instruction/cycle * 2.0 GHz = 256 GFLOP/s/core
256 GFLOP/s/core * 12 cores = 3.072 TFLOP/s/CMG
```

The existing hand-scheduled 12x64 dense FP16 kernel establishes a more useful
software ceiling.  It sustains 206.7 GFLOP/s/core with FP32 output and 217.7
GFLOP/s/core with FP16 output while streaming tiles.  Therefore the measured
12-core ceilings are approximately 2.480 TFLOP/s and 2.612 TFLOP/s,
respectively.  The gap from 3.072 TFLOP/s is primarily vector-register
read-port pressure and the FP16-to-FP32 epilogue.

### Weight-bandwidth roofs

Packed MXFP4 plus prepared FP16 scales uses approximately
`0.5 + 2/32 = 0.5625` byte per weight.  If a packed weight is fetched once and
reused by all M rows, its arithmetic intensity and 230 GB/s HBM roof are:

```
I_packed = 2*M / 0.5625 = 3.5556*M FLOP/byte
P_packed(230 GB/s) = 817.8*M GFLOP/s
```

This reproduces the approximately 818 GFLOP/s M=1 packed-stream roof.  For
M=8 the nominal packed roof is 6.54 TFLOP/s, above the FP16 compute peak, so a
fused packed M>=8 kernel is instruction/decode or FMA bound rather than HBM
bound.

The persistent decoded sidecar uses 2 bytes per weight, with scale already
applied.  A microkernel reusing each vector across MR rows has:

```
I_sidecar = 2*MR / 2 = MR FLOP/byte
```

For the proposed MR=12 kernel, the measured 214 GB/s and expected 230 GB/s
stream rates give 2.568 and 2.760 TFLOP/s.  The practical roof is therefore
`min(2.480 TFLOP/s FP32-output compute, 2.568 TFLOP/s bandwidth)`, before
promotion-block output traffic.  Even after accounting for K=256 partial-C
loads/stores, the roof remains well above 2 TFLOP/s.  The 800 GFLOP/s target
is about 26% of hardware peak and about 32% of the proven FP32-output ceiling.

## Current bottleneck

The current compiler-generated kernel is MR=8, NR=32.  Per K step it issues
one 64-byte weight-vector load and eight vector FMAs.  Eight independent
accumulators cannot cover the roughly nine-cycle FP16 FMA latency while also
feeding both FL pipes; the compiler also pays predication, scalar-broadcast,
loop, and FP32 conversion overhead.  It reaches about 717 GFLOP/s with pure
FP16 accumulation and 643 GFLOP/s with K=256 FP32 promotion on a 12-core CMG.

A tested MR=12, NR=32 intrinsic variant spills registers because its
accumulators and compiler temporaries compete for the 32 SVE registers.  The
solution is not a larger C intrinsic: it is a hand-scheduled MR=12, NR=64
assembly kernel whose register allocation has already been validated by the
dense FP16 work.

## Kernel design

### 1. N64 decoded-weight layout

Add an opt-in FP16 sidecar arranged as:

```
[N/64][K][64 fp16 weights]
```

Two adjacent N32 groups become two contiguous SVE vectors at every K.  This
preserves total sidecar size, makes every K step a 128-byte sequential load,
and matches the proven 12x2 SVE kernel.  Preparation remains parallel and
outside timed repeated GEMM.  Keep the existing packed N32 representation for
M=1 and non-sidecar paths.

### 2. K-major activation packing

For each 12-row activation tile, pack row-major `A[M,K]` into:

```
[K][12 fp16 activations]
```

The K=4096 buffer is 96 KiB per M12 tile and is reused by all N64 output
tiles.  Packing traffic is O(MK), versus O(MNK) GEMM work, and must be included
in end-to-end timing.  Allocate one shared packed-A buffer before the OpenMP N
loop so workers do not duplicate packing.

### 3. MR12 x NR64 assembly microkernel

Use a private, FP4-module copy of the proven A64FX 12x2 software-pipelined
kernel, with symbols specific to this module.  Do not introduce a link-time
dependency on another benchmark directory.

Register allocation:

```
z0-z23   24 FP16 accumulators (12 rows x two N32 vectors)
z24-z25  two FP16 weight vectors
z26-z31  six replicated activation values
p0       all FP16 lanes
p1       all FP32 lanes for the epilogue
```

Per K step:

```
24 FMLA + 12 LD1RH activation loads + 2 weight-vector loads
FMA floor:    24 / 2 = 12 cycles
load floor:   14 / 2 = 7 cycles
decode floor: 38 / 4 = 9.5 cycles
```

The K loop is therefore FMA-bound and has enough independent accumulators to
cover latency.  Use the existing two-K software pipeline first; only test a
four-K unroll after the baseline is correct.

Provide two epilogues:

- INIT: convert 24 FP16 accumulators and store FP32 C.
- ACCUM: convert, load FP32 C, add, and store for later promotion blocks.

`promotion_k=0` invokes INIT once for the full K range.  Nonzero promotion
invokes INIT for the first block and ACCUM for subsequent blocks, preserving
the existing accuracy contract.

### 4. Dispatch and tails

Parallelize over disjoint N64 tiles with static OpenMP scheduling.  Within a
tile, execute all complete M12 blocks consecutively so the 512 KiB K=4096
weight panel can be reused from the local CMG cache for the next M block.

- M>=12 full blocks: new MR12 x NR64 assembly.
- Remaining 1..11 rows: existing validated N32 kernel initially.
- M=8: add a dedicated MR8 x NR64 assembly kernel after MR12 establishes the
  ceiling; 16 explicit accumulators avoid compiler spills and should exceed
  the current MR8 x NR32 result.
- N32 tail: existing kernel.  Current public dimensions are multiples of 32;
  N64 is an optimized fast-path constraint, not a new API requirement.

### 5. Packed FP4 follow-up

After the sidecar kernel establishes the FMA ceiling, evaluate a fused packed
MR12 x NR32 assembly kernel.  It must retain packed four-bit HBM density and
reuse one decoded vector for 12 rows.  FLA-only `ZIP1`/`TBL` decode must be
software-pipelined ahead of the FMA group.  This phase is accepted only if it
beats the sidecar path after including sidecar preparation/amortization for
the intended workload, or materially reduces memory without dropping below
800 GFLOP/s.

## Measurement sequence

1. Record the current MR8 baseline at N=8192, K=4096 for M=8, 12, 24, 128,
   both `promotion_k=0` and 256, pinned to cores 12-23 and HBM node 4.
2. Unit-test N64 layout and packed-A transformation independently.
3. Compare the assembly microkernel against FP32 accumulation of the same
   FP16-rounded decoded weights.
4. Benchmark kernel-only and packing-inclusive rates, reporting sidecar bytes.
5. Use FAPP/PMU counters to check FL issue/completion stalls, L1/L2/HBM traffic,
   and vector-register spill loads/stores.
6. Run representative attention and FFN shapes, not only square synthetic
   matrices.

## Acceptance gates

- Correctness: all existing MXFP4, NVFP4-1D, and NVFP4-2D tests pass; new N64
  path has no worse relative-L2 error than the current sidecar path.
- Performance: at least 800 GFLOP/s/CMG for M=8, 24, and 128 at N=8192,
  K=4096 with K=256 FP32 promotion.
- Stretch target: at least 1.5 TFLOP/s for complete M12 blocks, progressing
  toward the 2.48 TFLOP/s measured dense-kernel ceiling.
- No M=1 regression: packed FP4 remains the selected matvec path.
- Memory accounting: benchmark output distinguishes packed model bytes,
  decoded sidecar bytes, packed-A workspace, and timed preparation cost.

## First implementation result

The N64 sidecar and MR12 x NR64 assembly phases were implemented after this
plan was committed.  With N=8192, K=4096, 12 pinned cores, and HBM node 4, the
packing-inclusive K=256 results are 1.459 TFLOP/s at M=8, 1.808 TFLOP/s at
M=12, 1.794 TFLOP/s at M=24, and 1.712 TFLOP/s at M=128.  Pure FP16 reaches
2.20 TFLOP/s at M=24.  The primary 800 GFLOP/s gate and the 1.5 TFLOP/s
stretch gate are therefore satisfied.  All three FP4 format tests retain the
same relative-L2 error as the previous decoded-panel path.

A promotion-path follow-up stores FP32 partial sums in a contiguous 3 KiB
tile rather than revisiting row-major C at a 32 KiB row stride. Together with
one shared OpenMP packing/compute region, this raises K=256 M=12 to
2.12--2.17 TFLOP/s and M=24 to 2.08--2.10 TFLOP/s. The remaining M12 gap to
2.2 TFLOP/s is 1.5--3.6%, while the pure-FP16 path already reaches 2.20
TFLOP/s.

The next iteration adds an MR8x64 assembly tail and A64FX cache tags: packed A
uses strong-reuse tag `0x9`, and the decoded-weight stream bypasses L1 with
tag `0xb`. K=256 N=8192 M=128 then sustains 2.209--2.236 TFLOP/s across
independent runs. With a balanced 144-tile N=9216 projection and M=192, K=1024
promotion sustains 2.404--2.420 TFLOP/s and pure FP16 sustains
2.507--2.516 TFLOP/s. The stable 2.2 and 2.4 targets are met, with the latter
requiring either K=1024 promotion or pure FP16 accumulation; K=256 reaches
2.358--2.370 TFLOP/s on the balanced shape.
