# Quantization formats on A64FX SVE1

## Recommendation

Native MXFP4 with K=32 E8M0 scaling is the best general-purpose A64FX SVE1
weight format.  It combines 4.25 bits/weight with a decoder that maps naturally
to `svtbl_f32` and power-of-two exponent scaling.  Use QPN8 for tensors where
FP8 accuracy is worth approximately twice the model size.

A practical mixed model should use MXFP4 for most FFN and expert weights, and
QPN8 for accuracy-sensitive projections, embeddings, and the language-model
head.  NVFP4 remains useful as a no-requantization compatibility path.  The
current block-FP8 implementation is not recommended as the primary compute
format.

## Format comparison

| Format | Effective bits/weight | SVE1 decode | Representative M=1 results | Role |
|---|---:|---|---:|---|
| MXFP4 G32 | 4.25 | Nibble split, two `svtbl_f32` operations, E8M0 exponent scale | 28.1 GB/s expert; 30.0 GB/s lm-head | Default |
| NVFP4 G16 | 4.50 | Same E2M1 table plus exact E4M3 scale lookup every 16 K | 27.8 GB/s expert; 33.7 GB/s lm-head | Compatibility |
| QPN8 tile32 | Approximately 8.00 | Byte widen and exact 256-entry E4M3 gather | 44.8 GB/s expert; 58.5 GB/s lm-head | Accuracy-sensitive tensors |
| FP8 block128 | Approximately 8.00 | E4M3 gather plus K-dependent E8M0 block scale | 20.4 GB/s expert; 24.4 GB/s lm-head | Interoperability |

Bandwidth is based on compressed codes and scale metadata read once per call.
It should not be interpreted as weights processed per second without accounting
for format density.  At 30 GB/s, MXFP4 processes roughly 56 billion weights/s
because it stores about 0.531 byte/weight.  This is close to QPN8 at 58.5 GB/s
and one byte/weight, despite QPN8's larger bandwidth figure.

## Instruction fit

A64FX implements 512-bit SVE1, but it has no native FP8 conversion or FP8
matrix instruction.  Exact kernels must decode into FP32 vectors before using
SVE FMA, or quantize activations before using integer `sdot`.

### MXFP4 G32

The sixteen E2M1 values fit in one 16-lane FP32 vector.  A packed byte is
widened, split with AND/shift operations, and both nibble vectors are decoded
with `svtbl_f32`.  E8M0 represents a power of two and can be formed directly as
an FP32 exponent using integer shift and reinterpret instructions.

The K=32 scale contributes 8/32 = 0.25 bit per weight.  It also halves the
number of scale operations relative to NVFP4.  These properties make MXFP4 the
best balance of capacity, decode cost, and measured latency.

### NVFP4 G16

NVFP4 uses the same efficient E2M1 table decoder, but an exact E4M3 scale is
needed every 16 K values.  E4M3 does not fit in the 16-entry SVE table and needs
either a 256-entry gather or a multi-instruction arithmetic decoder.  The
scales add 8/16 = 0.5 bit per weight.

NVFP4 can still win individual shapes, and the packer consumes the V100 QPN2
layout without changing quantized values.  It is therefore the preferred
compatibility format when requantizing a checkpoint is undesirable.

### QPN8 tile32

QPN8 avoids nibble extraction.  Each iteration widens 16 E4M3 bytes, gathers
their exact FP32 representations, folds the tile scale, and issues FP32 FMA.
The simpler instruction sequence compensates for reading twice as many weight
bytes: its latency is often close to FP4 while retaining FP8 accuracy.

The lossy FTZ arithmetic decoder reached 18.68 GB/s versus 18.29 GB/s for exact
QPN8 on the QKV shape.  This small gain does not justify flushing subnormals and
mapping NaN encodings to finite values in the default route.

### FP8 block128

Block FP8 has negligible scale storage overhead, but its E8M0 scale changes
along K.  The extra scale handling makes the present panel kernel substantially
slower than QPN8 without reducing the one-byte weight codes.  It should remain
an import/interoperability format unless a grouped scale-hoisting kernel closes
the measured gap.

## SDOT experiment

E2M1 weights can be represented exactly as doubled signed integers.  The
experimental kernel quantizes activations to int8 per K=16 or K=32 scale group
and executes SVE `sdot`; only the activation side is lossy.

The current path measured about 0.6 GB/s at 48 cores and 0.315% relative L2
error.  Dynamic activation quantization and construction of group-local int8
vectors cost much more than `sdot` saves.  It must not be selected by automatic
dispatch.  A competitive successor would require prequantized activation
blocks and a persistent SDOT-oriented compressed layout.

## Optimization priorities

The NUMA-correct read control reaches approximately 759 GB/s, while exact
kernels use only 1-7% of that ceiling.  Work should target decode and issue
cost, not additional HBM tuning.  The most promising next steps are:

1. Replace the MXFP4 E8M0 scale lookup with direct exponent construction.
2. Hoist block-FP8 scale conversion outside the per-K inner body.
3. Decode bounded exact FP32 tiles into L1 and reuse them across larger M.
4. Retain spill-free M=8 register blocks; FCC's M=16 form regressed severely.

All numbers above come from the native runs recorded in `RESULTS.md`.
