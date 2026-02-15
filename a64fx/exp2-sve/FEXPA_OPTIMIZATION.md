# FEXPA-based exp2 Optimization for A64FX

## Summary

Achieved **5.64 Gelem/s** (53% of theoretical peak) using FEXPA with 8x unrolling.

| Method | cyc/elem | Gelem/s | ULP | Notes |
|--------|----------|---------|-----|-------|
| **FEXPA 8x unroll** | **0.360** | **5.56** | 246 | Best balance, degree-1 |
| FEXPA 16x unroll | 0.355 | 5.64 | 246 | Marginal gain over 8x |
| FEXPA 4x unroll | 0.448 | 4.46 | 246 | Original version |
| FEXPA accurate | 0.543 | 3.69 | 1 | Degree-2 Taylor, ULP=1 |
| Estrin Pipe8 | 0.716 | 2.79 | 13007 | Previous best |

### Speedup vs Estrin
- **FEXPA 8x**: 50% faster (0.360 vs 0.716 cyc/elem)
- **FEXPA 4x**: 37% faster (0.448 vs 0.716 cyc/elem)
- **FEXPA accurate**: 24% faster with perfect ULP=1

## The Problem: FL* Pipe Bottleneck

Traditional exp2 implementations require:
- `frintm` (floor) - 9 cycles, FL* pipe
- `fcvtzs` (float-to-int) - 9 cycles, FL* pipe

A64FX has only **1 FL\* pipe** vs **2 FLA pipes**, making these operations the bottleneck.

## The Solution: FEXPA Instruction

FEXPA (Floating-point Exponential Accelerator) is an SVE instruction with a hardwired 64-entry lookup table:

```
T[i] = 2^(i/64)  for i = 0..63
```

### The Magic Shift Constant

```c
shift = 0x48481fc0 = 204927.0f
```

This constant is carefully chosen so that:

```
z = x + shift
FEXPA(z) = 2^x  (directly, no floor/int conversion needed!)
```

### How It Works

```
Input mantissa relationship:
  +64 mantissa bits → FEXPA output ×2   (2^1.0)
  +32 mantissa bits → FEXPA output ×√2  (2^0.5)
  +8  mantissa bits → FEXPA output ×2^(1/8)
```

When `x` is added to `shift`:
- Integer part of x → mantissa increases by 64 per unit
- Fractional part → mantissa increases proportionally
- FEXPA reads mantissa bits and outputs 2^x via table lookup

### Cycle Comparison

| Operation | Cycles | Pipe |
|-----------|--------|------|
| **Old: frintm + fcvtzs** | **18** | FL* (bottleneck) |
| **New: fadd + fexpa** | **13** | FLA (fadd=9, fexpa=4) |

### A64FX FLA Latencies

All FP operations have **9-cycle latency** on A64FX:
- `fadd`, `fsub`, `fmul`, `fmla`: 9 cycles
- `fexpa`: 4 cycles

With 2 FLA pipes, throughput = 2 ops/cycle, but latency hiding is critical.

## Algorithm

```
exp2(x) using FEXPA:
  1. z = x + shift           // fadd, FLA pipe
  2. n = z - shift           // fsub, FLA pipe (rounded integer)
  3. r = x - n               // fsub, FLA pipe (small remainder)
  4. scale = FEXPA(z)        // fexpa, FLA pipe (≈ 2^n)
  5. poly = r × ln(2)        // first-order Taylor correction
  6. result = scale × (1 + poly)
```

No FL* pipe operations required!

## Implementation Notes

### Register Allocation (Critical!)

**Bug found:** Using z8-z15 (callee-saved registers) without preserving them caused timer corruption.

**Solution:** Use only caller-saved registers:
- z0-z7: temporaries (caller-saved)
- z16-z31: temporaries (caller-saved)
- Avoid z8-z15 unless saved/restored

### Files

- `exp2_fexpa_u8.S` - **Recommended**: 8x unroll, 5.56 Gelem/s, ULP=246
- `exp2_fexpa_u16.S` - 16x unroll, 5.64 Gelem/s, ULP=246
- `exp2_fexpa_pipe.S` - 16x interleaved, 5.64 Gelem/s, ULP=246
- `exp2_fexpa_opt.S` - 4x unroll, 4.46 Gelem/s, ULP=246
- `exp2_fexpa_accurate.S` - 4x unroll, degree-2 Taylor, ULP=1

## Unroll Factor Analysis

### Critical Path

```
fadd (9) → fsub (9) → fsub (9) → fmul (9) → fmla (9) = 45 cycles
                 ↘ fexpa (4) ↗
```

- **Critical path**: 45 cycles per vector
- **Throughput**: 6 FLA ops / 2 pipes = 3 cycles/vector
- **Theoretical peak**: 16 elem / 3 cyc × 2 GHz = **10.67 Gelem/s**
- **Required unroll**: 45 / 3 = **15 vectors** to saturate pipes

### Benchmark Results

| Unroll | L1 (32KB) | L2 (2MB) | Memory (16MB) |
|--------|-----------|----------|---------------|
| 4x     | 5.42 (51%) | 3.09 (29%) | 4.46 (42%) |
| **8x** | **6.22 (58%)** | **5.62 (53%)** | **5.56 (52%)** |
| 16x    | 6.42 (60%) | 5.66 (53%) | 5.59 (52%) |

### Key Findings

1. **8x unroll is the sweet spot**: +24% over 4x (4.46 → 5.56 Gelem/s)
2. **16x provides marginal gain**: only +1.4% over 8x
3. **L1 peak = 60%**: Instruction scheduling limits throughput, not memory
4. **Dependency chains limit parallelism**: Even with 16 vectors, can't reach 100%

### Why Not 100% Peak?

The 6 FLA ops per vector have dependencies:
```
fadd → fsub → fsub → fmul → fmla (chain of 5)
     ↘ fexpa (parallel after fadd)
```

With 8 independent vectors, we have 8 × 5 = 40 dependent ops in chains.
But each chain still has 45-cycle latency that can't be fully hidden.

### Build

```bash
fcc -Nclang -O3 -march=armv8.2-a+sve -c exp2_fexpa_opt.S
```

## Accuracy vs Speed Trade-off

| Version | Polynomial | ULP | cyc/elem | Speedup vs Estrin |
|---------|------------|-----|----------|-------------------|
| FEXPA fast | degree-1 (r×ln2) | 246 | 0.445 | 37% |
| **FEXPA accurate** | **degree-2 Taylor** | **1** | **0.543** | **24%** |
| FEXPA glibc | degree-4 | 1 | 0.785 | 9% |

### Recommended Versions

1. **For maximum speed** (softmax in neural networks): Use `exp2_fexpa_u8.S`
   - 8x unroll, 5.56 Gelem/s, 52% of peak
   - ULP=246 ≈ 0.003% relative error
   - Best balance of performance vs code size

2. **For accuracy + speed**: Use `exp2_fexpa_accurate.S`
   - ULP=1 (perfect single-precision accuracy)
   - 3.69 Gelem/s, 24% faster than Estrin

### Polynomial Coefficients (Degree-2)

**Important**: Taylor coefficients work better with FEXPA than minimax coefficients.

```
Degree-2 Taylor for 2^r ≈ 1 + c0*r + c1*r²:
  c0 = ln(2)     = 0.693147 (0x3f317218)
  c1 = ln(2)²/2  = 0.240227 (0x3e75fdf0)
```

Minimax coefficients (optimized for polynomial approximation alone) gave ULP=1354 because
they don't account for the interaction with FEXPA's table lookup. Taylor coefficients
properly complement FEXPA's approximation to achieve ULP=1.

## Verification

```bash
./bench_unroll
# Output:
# === Correctness + Performance (n=4194304, 16.8 MB) ===
# Method                  Time     Throughput         Cycles   Peak%  ULP
# 4x unroll           0.940 ms   4.46 Gelem/s  0.448 cyc/elem   41.8%  ULP=246
# 8x unroll           0.754 ms   5.56 Gelem/s  0.360 cyc/elem   52.1%  ULP=246
# 16x unroll          0.751 ms   5.59 Gelem/s  0.358 cyc/elem   52.4%  ULP=246
# 16x interleaved     0.743 ms   5.64 Gelem/s  0.355 cyc/elem   52.9%  ULP=246
#
# === L1 resident (n=8192, 32.0 KB) ===
# 8x unroll           0.001 ms   6.22 Gelem/s  0.322 cyc/elem   58.3%
# 16x unroll          0.001 ms   6.42 Gelem/s  0.312 cyc/elem   60.2%
```

## References

- glibc SVE exp2f: `sysdeps/aarch64/fpu/exp2f_sve.c`
- ARM FEXPA documentation: ARM Architecture Reference Manual
- A64FX Microarchitecture Manual (FL* vs FLA pipe latencies)
