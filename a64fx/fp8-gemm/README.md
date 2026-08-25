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

`fp8_matrix_prepare_i8_tile` provides an optional offline signed-INT8
re-encoding. The maximum-throughput 128x128 form retains one byte per weight
and one FP32 scale per block. Its assembly hot loop is `LD1SB`, `SCVTF`, and
FP32 FMA. A 32x2 form uses four scales per two K rows for lower error.
Re-encoding never mutates the source matrix and is excluded from timing.

## A64FX results

Full 128 MiB stream, N=32768, K=4096, one 12-core CMG:

| Decoder | GFLOP/s | Source GB/s | Semantics |
|---|---:|---:|---|
| Sparse-exception assembly | **188.4** | **94.2** | Exact E4M3FN |
| Normal-only assembly | 202.2 | 101.2 | Invalid for exponent-zero input |
| FP16-bit decode, FP32 FMA | 159.2 | 79.6 | Canonicalized + corrections |
| FP32 LUT gather | 74.1 | 37.1 | Exact E4M3FN |
| Inline exact arithmetic | 52.9 | 26.4 | Exact E4M3FN |
| INT8 re-encoded, 128x128 | **419.4** | **209.8** | Lossy, FP32 FMA/accum |
| INT8 re-encoded, 32x2 | 373.2 | 198.3 | Lower error, FP32 FMA/accum |

Layer-0 DeepSeek attention samples contain 0.014--0.017% exponent-zero values,
about 0.001% exact zeros, and no NaNs. The benchmark injects 0.0167% specials.

The kernel is compute/decode limited rather than HBM limited. For each 16
weights, the fast path needs five FL integer instructions plus one FP32 FMA.
With two FL pipes, this puts an optimistic 12-core issue ceiling near 280
GFLOP/s before weight loads, activation/scale multiplication, and exception
corrections. Consequently a 400 GFLOP/s bandwidth target is not feasible on
SVE 1.0 while preserving FP32 FMA and exact E4M3FN values.

The 128x128 INT8 path exceeds 400 GFLOP/s by eliminating nonlinear decode.
Neither tested INT8 layout meets the selected 0.5% worst-case relative-L2
gate on the five layer-0 DeepSeek attention tensors: 128x128 MSE scaling
reaches 1.31%, and 32x2 absmax scaling reaches 0.92%. Both retain cosine
similarity above 0.9999. They therefore remain explicit experimental speed
paths, while sparse-exact FP8 is the accuracy fallback.

```sh
OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores \
  numactl --physcpubind=12-23 --membind=4 \
  ./bench_fp8_deepseek /local/u14346/ds4f-fp4-gemm/layer0.raw \
  /local/u14346/ds4f-fp4-gemm/layer0.manifest
```

`FP8_I8_GROUP`, `FP8_I8_LANES`, and `FP8_I8_POLICY=mse` select alternative
encodings. A third positional argument writes converted codes and scales to
the explicitly supplied directory.
