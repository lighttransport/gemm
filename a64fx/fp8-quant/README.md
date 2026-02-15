# FP8 Quantization Kernels for A64FX

SVE-optimized FP16→FP8 and FP32→FP8 quantization kernels for Fujitsu A64FX processor.

## Supported Formats

| Format | Layout | Exponent Bias | Max Value | Min Normal |
|--------|--------|---------------|-----------|------------|
| E5M2 | 1 sign + 5 exp + 2 mant | 15 | 57344 | 6.1e-5 |
| E4M3 | 1 sign + 4 exp + 3 mant | 7 | 448 | 1/64 |

## Performance Results

### A64FX Configuration
- SVE Vector Length: 512 bits (32 FP16 or 16 FP32 per vector)
- CPU Frequency: 2.0 GHz
- Key Latencies: LD=11cy, FCVT=9cy, MUL=9cy, ADD=5cy, BITOP=4cy

### FP16 → E5M2 Performance

| Kernel | Cycles/Element | Speedup vs Scalar |
|--------|----------------|-------------------|
| Scalar Reference | ~15 | 1x |
| SVE Original | 0.75 | 20x |
| SVE Unroll x4 | 0.42 | 36x |
| **SVE Unroll x8** | **0.39** | **38x** |
| SVE SW Pipeline | 0.47 | 32x |

### FP32 → E5M2 Performance

| Kernel | Cycles/Element | Speedup vs Scalar |
|--------|----------------|-------------------|
| Scalar Reference | ~20 | 1x |
| SVE Original | 1.88 | 11x |
| SVE Unroll x4 | 1.01 | 20x |
| **SVE Unroll x8** | **0.91** | **22x** |
| SVE SW Pipeline | 1.29 | 15x |
| SVE FCVT (asm) | 2.21 | 9x |

### E4M3 Kernels
E4M3 kernels require scalar fallback for subnormal handling, resulting in lower speedups:
- FP16→E4M3: ~7x speedup
- FP32→E4M3: ~6x speedup

## Optimization Techniques

### 1. Loop Unrolling (Best Performance)
Issue multiple loads early to hide the 11-cycle load latency:
```c
// x8 unroll: issue 8 loads, then process all
svuint16_t x0 = svld1_u16(pg16, src + i);
svuint16_t x1 = svld1_u16(pg16, src + i + vl16);
// ... x2-x7
// Process x0-x7 while loads complete
```

**Results:**
- x4 unroll: 1.8x over original SVE
- x8 unroll: 1.9-2.1x over original SVE

### 2. Software Pipelining
Overlap load/compute/store across iterations:
```
Iteration N: Load(N+2), Compute(N+1), Store(N)
```

**Results:**
- 1.45-1.60x speedup (less than unrolling due to overhead)

### 3. Hardware FCVT Approach (Inline Assembly)
Uses SVE FCVT instruction for FP32→FP16, then bit ops for FP16→E5M2:
```asm
fcvt z0.h, p0/m, z0.s    // FP32 -> FP16 (9 cycles)
// Then extract/round mantissa, combine with sign
```

**Why FCVT is slower than pure bit manipulation:**
- FCVT has 9-cycle latency, adding to critical path
- Still need bit manipulation after FCVT for mantissa rounding
- Pure bit manipulation with x8 unroll better hides 11-cycle load latency
- Result: 2.21 vs 0.91 cycles/elem

**Compiler issues:**
- `svcvt_f16_f32_x` intrinsic compiles but fails to link (undefined symbol)
- `svcvt_f16_f32_m` causes internal compiler error in fcc
- Workaround: Use inline assembly with explicit register allocation

## Algorithm Details

### FP16 → E5M2 Conversion
FP16 and E5M2 share the same exponent bias (15), simplifying conversion:
1. Extract sign, exponent, mantissa from FP16
2. Shift exponent to E5M2 position (bits 2-6)
3. Round mantissa from 10 bits to 2 bits
4. Handle mantissa overflow (increment exponent)
5. Clamp exponent overflow to max

### FP32 → E5M2 Conversion
Requires exponent bias adjustment (FP32 bias=127, E5M2 bias=15):
1. Clamp absolute value to E5M2 range
2. Extract and adjust exponent: `e5m2_exp = fp32_exp - 112`
3. Round mantissa from 23 bits to 2 bits
4. Handle overflow/underflow

## File Structure

```
fp8-quant/
├── fp8_quant.h          # Main header with function declarations
├── fp8_quant.c          # Original SVE kernels + scalar reference
├── fp8_quant_opt.h      # Unroll x4 declarations
├── fp8_quant_opt.c      # Unroll x4 implementations
├── fp8_quant_opt8.h     # Unroll x8 declarations
├── fp8_quant_opt8.c     # Unroll x8 implementations
├── fp8_quant_swp.h      # Software pipelining declarations
├── fp8_quant_swp.c      # Software pipelining implementations
├── bench_fp8.c          # Basic validation benchmark
├── bench_compare.c      # Unroll comparison benchmark
├── bench_all.c          # Complete optimization comparison
├── Makefile             # Build with fcc compiler
└── README.md            # This file
```

## Building and Running

```bash
# Build all benchmarks
make all

# Run complete comparison
./bench_all

# Run basic validation
./bench_fp8
```

## Compiler Notes

Using Fujitsu fcc native compiler on A64FX:
```
fcc -O3 -march=armv8.2-a+sve -Kopenmp -Kfast
```

### Known Issues
1. **FP16 SVE intrinsics**: `svcvt_f16_f32_*` intrinsics don't link properly
2. **Internal compiler errors**: `svcvt_f16_f32_m` causes ICE, use `_x` or `_z` variants
3. **Goto with SVE types**: Avoid goto that bypasses SVE variable declarations

## Conclusion

**Recommended kernel: x8 unrolled version**
- FP16→E5M2: **0.39 cycles/element** (2.56 elements/cycle, 38x vs scalar)
- FP32→E5M2: **0.91 cycles/element** (1.10 elements/cycle, 22x vs scalar)

### Key Findings

1. **Simple unrolling wins**: x8 unroll outperforms both SW pipelining and hardware FCVT
2. **Load latency is the bottleneck**: 11-cycle LD latency dominates; issuing 8 loads early hides it
3. **FCVT not beneficial**: Hardware FP32→FP16 adds 9 cycles to critical path without reducing work
4. **Pure bit manipulation is efficient**: Same exponent bias (15) for FP16/E5M2 simplifies conversion

### Performance Hierarchy
```
FP32 → E5M2:
  x8 unroll (0.91 cy/elem) > x4 unroll (1.01) > SW pipe (1.29) > Original (1.88) > FCVT (2.21)

FP16 → E5M2:
  x8 unroll (0.39 cy/elem) > x4 unroll (0.42) > SW pipe (0.47) > Original (0.74)
```
