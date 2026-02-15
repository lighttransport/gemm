# FP8 to FP16/FP32 Conversion Kernels for A64FX

## Overview

This document summarizes the performance of FP8 (E4M3 and E5M2) to FP16/FP32 conversion kernels on Fujitsu A64FX processor with 512-bit SVE.

## FP8 Formats

| Format | Sign | Exponent | Mantissa | Bias | Special Values |
|--------|------|----------|----------|------|----------------|
| E4M3   | 1    | 4        | 3        | 7    | NaN only (no Inf) |
| E5M2   | 1    | 5        | 2        | 15   | Inf and NaN |

## Implementation Approaches

### 1. LUT-based Gather (Recommended)
- 256-entry lookup table (1KB for 32-bit entries)
- Uses SVE `ld1_gather_u32offset` instruction
- Fits entirely in L1 cache (16 cache lines)

### 2. SVE Bit Arithmetic (FL* pipe)
- Vectorized bit manipulation using SVE instructions
- Extracts sign/exp/mantissa, adjusts bias, recombines
- Falls back to LUT for subnormal numbers

### 3. Scalar (EX* pipe)
- Reference implementation using ARM64 base instructions
- Not vectorized, included for correctness verification

## Performance Results (A64FX @ 2.0 GHz)

### FP8 -> FP32 Conversion (16 elements per SVE op)

| Conversion | Method | cyc/elem | cyc/16elem | cyc/32elem |
|------------|--------|----------|------------|------------|
| **E4M3 -> FP32** | Gather (LUT) | 0.70 | **11.2** | 22.4 |
| | SVE (FL*) | 1.71 | 27.4 | 54.7 |
| | Scalar (EX*) | 14.50 | 232.0 | 464.1 |
| **E5M2 -> FP32** | Gather (LUT) | 0.70 | **11.2** | 22.4 |
| | SVE (FL*) | 1.71 | 27.4 | 54.7 |
| | Scalar (EX*) | 14.65 | 234.4 | 468.8 |

### FP8 -> FP16 Conversion (16 elements per SVE op, packed to 32 FP16)

| Conversion | Method | cyc/elem | cyc/16elem | cyc/32elem |
|------------|--------|----------|------------|------------|
| **E4M3 -> FP16** | Gather (LUT) | 0.75 | **12.0** | 24.1 |
| | SVE (FL*) | 1.79 | 28.7 | 57.4 |
| | Scalar (EX*) | 12.83 | 205.3 | 410.5 |
| **E5M2 -> FP16** | Gather (LUT) | 0.76 | **12.2** | 24.4 |
| | SVE (FL*) | 1.62 | 26.0 | 52.0 |
| | Scalar (EX*) | 7.92 | 126.7 | 253.3 |

## Latency Analysis

### A64FX Memory Subsystem
- L1 Load Bandwidth: 128 bytes/cycle (2 × 64B pipes)
- L1 Load Latency: 11 cycles
- Gather Latency: ~9 cycles per element (pipelined)

### Gather Latency Model
```
Theoretical max: 11 + (N_loads - 1) = 11 + 15 = 26 cycles (16 loads to different cache lines)
Observed:        ~11-12 cycles for 16 elements
```

**Why observed is better than theoretical max:**
1. LUT is only 1KB (16 cache lines × 64B)
2. High cache line reuse - 256 entries across 16 lines = 16 entries/line
3. A64FX has 2 L1 load pipes that can work in parallel
4. Good pipelining when multiple accesses hit same cache line

### SVE Bit Arithmetic Breakdown (~27-28 cycles for 16 elements)
```
Instruction          Cycles (approx)
------------------------------------
ld1b                 1    - load 16 FP8 bytes
sunpklo ×2           2    - unpack to 32-bit
lsr, and ×3          3    - extract sign, exp, mant
cmpeq ×2             2    - check zero/max exp
add, lsl ×3          3    - compute new exp/mant
orr ×3               3    - combine fields
sel ×2               2    - handle special cases
st1w                 1    - store 16 FP32
------------------------------------
Total: ~18-20 instructions ≈ 27-28 cycles
```

## Conversion Formulas

### E4M3 -> FP32
```
Normal (exp != 0, exp != 15):
  fp32_exp = fp8_exp + 120  (bias adjustment: 127 - 7)
  fp32_mant = fp8_mant << 20
  result = (sign << 31) | (fp32_exp << 23) | fp32_mant

Zero (exp == 0, mant == 0):
  result = sign << 31

Subnormal (exp == 0, mant != 0):
  Normalize and adjust exponent (use LUT)

NaN (exp == 15):
  result = (sign << 31) | (0xFF << 23) | (mant << 20)
```

### E5M2 -> FP32
```
Normal (exp != 0, exp != 31):
  fp32_exp = fp8_exp + 112  (bias adjustment: 127 - 15)
  fp32_mant = fp8_mant << 21
  result = (sign << 31) | (fp32_exp << 23) | fp32_mant

Zero (exp == 0, mant == 0):
  result = sign << 31

Subnormal (exp == 0, mant != 0):
  Normalize and adjust exponent (use LUT)

Inf/NaN (exp == 31):
  result = (sign << 31) | (0xFF << 23) | (mant << 21)
```

## Recommendations

1. **Use LUT-based gather** for best performance (~11 cycles per 16 elements)
   - LUT fits in L1 cache
   - 2.4× faster than SVE bit arithmetic
   - 20× faster than scalar

2. **Use SVE bit arithmetic** when:
   - LUT memory overhead is unacceptable
   - Need to use FL* pipes while EX*/LD pipes are busy
   - Building mixed-precision compute kernels

3. **Throughput estimates** (single core @ 2 GHz):
   - Gather: 16 elem / 11 cyc = 1.45 elem/cycle = 2.9 GB/s (FP8 input)
   - SVE: 16 elem / 27 cyc = 0.59 elem/cycle = 1.2 GB/s (FP8 input)

## Files

- `fp8_convert.h` - Header with scalar reference and function declarations
- `fp8_convert.c` - Implementation of all conversion methods
- `bench_fp8_convert.c` - Benchmark program
- `Makefile` - Build with `fcc -Ofast -Kfast,openmp -Nclang`

## Build and Run

```bash
make clean && make
./bench_fp8_convert
```
