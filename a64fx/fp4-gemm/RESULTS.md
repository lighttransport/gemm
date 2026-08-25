# DeepSeek-V4-Flash Layer-0 Results

Measured on one A64FX node with 48 cores, Fujitsu Compiler 4.12.2,
`OMP_PROC_BIND=close`, and `OMP_PLACES=cores`. The 3.287 GiB layer-0 subset was
loaded into anonymous HBM; `MemAvailable` remained approximately 26.3 GiB.
Offline staging and quantization are excluded from GEMM timing.

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
