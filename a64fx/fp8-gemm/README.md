# A64FX FP8 E4M3FN FP32 GEMV

This module implements M=1 weight-streaming FP8 GEMV with FP32 activations,
FP32 FMA, and FP32 accumulators. Weights use an A64FX-oriented
`[N/128][K][4][32]` layout so every K step is one sequential 128-byte stream.
DeepSeek-style E8M0 scales are represented as one FP32 value per 128x128 block.

## Build and run

```sh
make -C a64fx/fp8-gemm test
OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores \
  numactl --physcpubind=12-23 --membind=4 \
  ./a64fx/fp8-gemm/bench_fp8_stream 32768 4096
```

`fp8_matrix_prepare_fast` replaces rare exponent-zero/NaN codes with signed
minimum-normal codes and records exact corrections grouped by N128 tile. The
assembly hot loop then uses eight independent normal-value decode chains;
corrections are applied in FP32 after accumulation.

## A64FX results

Full 128 MiB stream, N=32768, K=4096, one 12-core CMG:

| Decoder | GFLOP/s | Source GB/s | Semantics |
|---|---:|---:|---|
| Sparse-exception assembly | **188.4** | **94.2** | Exact E4M3FN |
| Normal-only assembly | 202.2 | 101.2 | Invalid for exponent-zero input |
| FP16-bit decode, FP32 FMA | 159.2 | 79.6 | Canonicalized + corrections |
| FP32 LUT gather | 74.1 | 37.1 | Exact E4M3FN |
| Inline exact arithmetic | 52.9 | 26.4 | Exact E4M3FN |

Layer-0 DeepSeek attention samples contain 0.014--0.017% exponent-zero values,
about 0.001% exact zeros, and no NaNs. The benchmark injects 0.0167% specials.

The kernel is compute/decode limited rather than HBM limited. For each 16
weights, the fast path needs five FL integer instructions plus one FP32 FMA.
With two FL pipes, this puts an optimistic 12-core issue ceiling near 280
GFLOP/s before weight loads, activation/scale multiplication, and exception
corrections. Consequently a 400 GFLOP/s bandwidth target is not feasible on
SVE 1.0 while preserving FP32 FMA and exact E4M3FN values.
