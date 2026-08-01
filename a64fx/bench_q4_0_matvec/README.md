# Q4_0 matvec benchmark on A64FX

Compares the following matvec paths for Q4_0 weights on actual A64FX hardware:

| Path | Description | Notes |
|------|-------------|-------|
| `fp32 FMA` | Gold standard, fp32 weights + fp32 FMA | Baseline |
| `int8 SDOT (on-fly)` | Dequant Q4_0 to int8 in hot loop, then SDOT | Slower due to dequant overhead |
| `int8 SDOT (preq)` | Pre-dequant to int8 once, reuse for many matvecs | **Production path, 40x faster than fp32** |
| `int16 SDOT` | Pre-dequant to int16, then H→D SDOT | 2x slower than int8 (half products/dot) |
| `fp16 FMA` | Pre-dequant to fp16, then FMA | 2x slower than int8 (half products/dot) |

## Build

### Cross-compile (from x86 host)

```sh
# Default (prequant int8 only)
aarch64-linux-gnu-gcc-14 -O3 -static -march=armv8.2-a+sve -ffp-contract=fast \
    -D_GNU_SOURCE -I../../common bench_q4_0_matvec.c -lm -lpthread -o bench_q4_0_matvec

# With on-the-fly int8 path enabled
make ONTHEFLY=1
```

### Native (on A64FX)

```sh
gcc-14 -O3 -march=armv8.2-a+sve -ffp-contract=fast -D_GNU_SOURCE \
    -I../../common bench_q4_0_matvec.c -lm -o bench_q4_0_matvec
```

## Run

```sh
./bench_q4_0_matvec
```

## Output

The benchmark tests matrix sizes from 512×512 up to 8192×8192 and reports
per-path:
- **Cycles**: total cycles over all reps (read from `cntvct_el0`)
- **Time(ms)**: per-call wall time
- **GFLOPS**: effective throughput (2·M·N ops per call)
- **Speedup**: vs fp32 FMA baseline
- **OK**: whether output matches fp32 reference within tolerance
- **MaxErr**: max absolute error vs reference

## Measured results (native A64FX, `fcc -Nclang -O3`, single thread)

Measured on a native A64FX node (`fcc -Nclang -O3 -march=armv8.2-a+sve
-ffp-contract=fast`, `OMP_NUM_THREADS=1`). All paths verify against the fp32
reference (`OK`). Effective GFLOPS for the int8 prequant path (prepack done
once in warmup, not in the timed loop):

| Size (rows×cols) | fp32 GFLOPS | int8 preq | speedup | int16 | fp16 |
|------------------|-------------|-----------|---------|-------|------|
| 512×512   | 6.2 | 44.8 | 7.2×  | 2.9× | 2.1× |
| 1024×1024 | 6.2 | 62.6 | 10.3× | 4.0× | 2.6× |
| 2048×2048 | 6.2 | **76.6** | **12.5×** ← peak | 1.9× | 1.5× |
| 4096×4096 | 5.5 | 30.5 | 5.6×  | 2.1× | 1.6× |
| 8192×4096 | 5.5 | 30.0 | 5.5×  | 2.1× | 1.4× |
| 4096×8192 | 5.8 | 29.3 | 5.1×  | 1.9× | 1.2× |

**int8 SDOT prequant peaks at ~12.5× (≈77 GFLOPS) at 2048², not the
qlair-predicted 30-50×.** Two reasons: (1) the fp32 baseline here is itself
SVE-vectorized (qlair's 40× compared against a slower baseline); (2) at ≥4096²
the working set exceeds L2 and the matvec goes **bandwidth-bound** — and the
int8 cache is 1 byte/weight versus Q4_0's 0.56 byte/weight, so it reads *more*
bytes and loses its compute edge (12.5× @2048² → ~5.5× at 4096²+). The win is
real but is a mid-size, compute-bound effect.

Quantization error vs the fp32 reference: int8 prequant MaxErr 9.9–49 (within
the 5% tolerance); int16 MaxErr 0.005–0.029 (256× finer); fp16 MaxErr 0.2–2.5.

> Counter note: `cntfrq_el0` reads **100 MHz** on this node, so the benchmark's
> "Cycles" column is 100 MHz timer ticks, not CPU cycles. The GFLOPS / Time /
> Speedup columns are wall-time–derived and correct.

The int8 SDOT path achieves the highest throughput because int8 SDOT does
**64 int8 products per instruction** (4 int8 × 16 int32 lanes), while int16
SDOT and fp16 FMA do only **32 products per instruction**. Both at 2
ops/cycle issue rate, int8 has 2x the peak products/cycle of int16/fp16.

## Why these paths exist

- **fp32 FMA**: gold standard for precision, no quantization error
- **int8 SDOT (preq)**: highest throughput, 0.5-2% quantization error acceptable for inference
- **int16 SDOT**: higher precision than int8 (less quantization), 2x slower
- **fp16 FMA**: no quantization error, 2x slower than int8, overflow risk for large K

For LLM inference, the int8 prequant path is the recommended choice.
For numerical-sensitive workloads, fp32 or fp16 may be preferred.

## Cross-compile toolchain

The benchmark uses the `cntvct_el0` virtual counter for timing.
This is available on all AArch64 systems with a fixed-frequency counter.
On the measured A64FX node the counter runs at **100 MHz** (firmware-set), so
the "Cycles" column is 100 MHz ticks, not CPU cycles — use the Time/GFLOPS
columns (wall-time–derived) for performance.
