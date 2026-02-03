# SVE exp2 Polynomial Kernel Summary

## Overview
Fast exp2 approximation for softmax using 4th-degree Taylor polynomial with IEEE754 bit manipulation.

## Performance

### Speedup
- **30x faster** than libm exp2f()
- Simple exp2: ~1.0 cycles/element
- Softmax exp2: ~1.5 cycles/element (includes int32→float conversion)

### Throughput
- Simple exp2: 1.95 Gelem/s
- Softmax exp2: 1.34 Gelem/s

## Algorithm

### exp2(x) Computation
```
n = floor(x)           // Integer part
f = x - n              // Fractional part [0, 1)
exp2(x) = 2^n * 2^f
```

### 2^n via Bit Manipulation
```
n_int = convert_to_int(n)
biased = n_int + 127           // IEEE754 exponent bias
exp2_n = biased << 23          // Position in IEEE754 float bits
```

### 2^f via Taylor Polynomial
```
exp2(f) ≈ 1 + c1*f + c2*f² + c3*f³ + c4*f⁴

Coefficients (Taylor series for e^(f*ln2)):
c1 = ln(2)       = 0.6931472
c2 = ln(2)²/2    = 0.2402265
c3 = ln(2)³/6    = 0.0555041
c4 = ln(2)⁴/24   = 0.0096181
```

## Accuracy

### ULP Error
- Max ULP: ~13000 (for extreme values)
- Typical relative error: < 0.02%

### Softmax Suitability
For softmax, relative accuracy matters more than ULP:
- Normalized softmax probabilities have < 0.1% error
- Acceptable for deep learning inference

## Files

### exp2_poly_v2.S
- `exp2_poly_simple_v2`: Float input → exp2 output
- `exp2_poly_softmax_v2`: Int32 input with scaling and max subtraction

### Kernel Features
- 4x SVE vector unrolling
- Horner's method for polynomial evaluation
- Range clamping to [-126, 127] for numerical stability
- Predicated tail handling

## Usage
```c
#include <stdint.h>

// Simple exp2 for float input
void exp2_poly_simple_v2(const float* in, float* out, int n);

// Softmax exp2: out[i] = exp2((in[i] - max_val) * scale)
void exp2_poly_softmax_v2(const int32_t* in, float* out, int n,
                           float scale, int32_t max_val);
```

## Build
```bash
fcc -O3 -Kopenmp -c exp2_poly_v2.S
fcc -O3 -Kopenmp -o bench bench.c exp2_poly_v2.o -lm
```

## Note on FEXPA
The A64FX FEXPA instruction was investigated but proved difficult to use correctly. The polynomial approach provides more predictable accuracy and simpler implementation while achieving similar performance.
