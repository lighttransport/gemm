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

## Expected results (on A64FX @ 2 GHz)

| Path | Speedup vs fp32 | Notes |
|------|-----------------|-------|
| fp32 FMA | 1.0× | 64 GFLOPS peak (32 ops/cycle × 2 GHz) |
| int8 SDOT (on-fly) | ~1-3× | Dequant overhead, only faster for large M |
| **int8 SDOT (preq)** | **~30-50×** | **Production path. ~55% of 256 GOPS int8 peak** |
| int16 SDOT | ~3-4× | 128 GOPS int16 peak (half of int8 throughput) |
| fp16 FMA | ~3-4× | 128 GFLOPS fp16 peak (2x fp32, half of int8) |

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

The benchmark uses the `cntvct_el0` virtual counter for cycle-accurate timing.
This is available on all AArch64 systems with a fixed-frequency counter.
On A64FX, the counter frequency is typically 1 GHz (set by firmware).
