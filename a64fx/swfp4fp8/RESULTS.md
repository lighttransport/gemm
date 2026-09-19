# A64FX native results

Measured 2026-09-20 on `a27-0106c`, A64FX SVE 512-bit, with FCC
4.12.2 (20251113).  Commands used the checked-in Makefile and pinned compute
cores 12-59.

## Correctness

`make -C a64fx/swfp4fp8 test CC=fcc` passes:

- exhaustive decoder classification for all E4M3 codes and boundary checks for
  E2M1/E8M0;
- canonical and V100 QPN permutation round trips;
- all four exact formats against a double-accumulation reference at M=17;
- exact panel and row routes, N tails, and the FP16 adapter;
- FP4 SDOT accuracy: relative L2 `0.003153` on the deterministic test case.

Exact FP32 relative L2 was zero for FP4 and at most `1.47e-7` for FP8.  The
FP16 adapter measured `1.90e-4` relative L2.

## NUMA and thread scaling

The 384 MiB fresh-page SVE read control measured:

| Threads | Raw read GB/s | NVFP4 panel GB/s | QPN8 panel GB/s | MXFP4 panel GB/s | block-FP8 panel GB/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 44.1 | 0.75 | 1.23 | 0.85 | 0.50 |
| 12 | 225.3 | 8.40 | 13.83 | 9.40 | 5.76 |
| 24 | 430.7 | 11.81 | 21.46 | 11.30 | 10.69 |
| 48 | 756.2 | 10.21 | 17.82 | 10.02 | 16.00 |

Kernel entries are `N=5120,K=1536,M=1`; bandwidth includes compressed codes
and scale metadata once per call.  The decline from 24 to 48 threads on several
small-N cases is reproducible scheduling/decode saturation, not an HBM ceiling.

## Representative 48-core results

From `./bench_swfp4fp8 --quick --threads 48`:

| Format / shape `(N,K)` | M=1 | M=8 | M=16 | M=32 |
|---|---:|---:|---:|---:|
| NVFP4 `(3584,5120)` | 27.1 GB/s | 27.8 GB/s | 18.1 GB/s | - |
| QPN8 `(3584,5120)` | 44.1 GB/s | 34.2 GB/s | 20.5 GB/s | 11.3 GB/s |
| MXFP4 `(3584,5120)` | 28.0 GB/s | 26.9 GB/s | 17.6 GB/s | - |
| block FP8 `(3584,5120)` | 20.5 GB/s | 13.4 GB/s | 7.0 GB/s | 3.6 GB/s |

The full V100/A64FX shape sweep also completed.  At M=1 it measured 33.67
GB/s NVFP4, 58.51 GB/s QPN8, 29.95 GB/s MXFP4, and 24.42 GB/s block FP8 on
the `(62080,5120)` lm-head.  The best exact QPN8 M=1 result in the sweep was
53.39 GB/s on `(8704,5120)` aside from the wider lm-head case.

On the QKV shape `(5120,1536)`, the 48-core panel/row comparisons were:

- NVFP4: 10.21 vs 4.79 GB/s.
- QPN8: 17.82 vs 9.57 GB/s.
- MXFP4: 10.02 vs 4.74 GB/s.
- block FP8: 16.00 vs 9.01 GB/s.

The FP8 FTZ experiment reached 18.41 GB/s versus 17.82 GB/s exact QPN8 on
that shape.  FP4 SDOT was much slower (about 0.6 GB/s at 48 cores) because the
current experiment dynamically materializes group-local int8 vectors; it is an
accuracy/dataflow probe, not a dispatch candidate.

## Conclusions

- Fresh anonymous first-touch is mandatory: recycled heap placement initially
  reported roughly 93 GB/s, the bandwidth of one CMG, while the corrected
  control reaches 756 GB/s.
- These exact kernels are decode/issue-bound at only 1-6% of measured HBM read
  bandwidth.  Absolute V100 bandwidth numbers are therefore not comparable.
- Reusing each decoded vector over M improves compute throughput, but repeated
  eight-row blocks reduce logical bandwidth for larger M.
- A 16-row SVE specialization caused severe FCC spills on large K.  Composing
  two spill-free eight-row calls reduced the `(8192,4096),M=16` NVFP4 time from
  about 14.2 ms to 2.7 ms.
- The next optimization should target fewer FP32 decode instructions (or an
  L1-tiled exact decode reused across larger M), not additional HBM tuning.
