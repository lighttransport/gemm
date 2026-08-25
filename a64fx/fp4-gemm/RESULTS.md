# DeepSeek-V4-Flash Layer-0 Results

Measured on one A64FX node with 48 cores, Fujitsu Compiler 4.12.2,
`OMP_PROC_BIND=close`, and `OMP_PLACES=cores`. The 3.287 GiB layer-0 subset was
loaded into anonymous HBM; `MemAvailable` remained approximately 26.3 GiB.
Offline staging and quantization are excluded from GEMM timing.

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

## Conclusion

Promoting every 256 K values is the useful balanced mode on these workloads:
it improves throughput by roughly 3-4x over promotion every 32 values while
keeping median accumulation error below about 6.2e-4. Pure FP16 accumulation
is another 1.8-2.4x faster but raises relative error to roughly 1.5e-3-1.9e-3.
MXFP4 is consistently faster than either NVFP4 layout because its K32 scaling
requires fewer scale decodes.
