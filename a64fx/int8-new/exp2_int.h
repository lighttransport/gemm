// exp2_int.h - Integer exp2() Approximation for Softmax
// Optimized for A64FX SVE 512-bit
//
// Key insight: exp(x) = 2^(x * log2(e))
// So we compute exp2(y) where y = x * log2(e)
//
// For softmax: exp(score - max) = exp2((score - max) * log2(e))
//
// Algorithm:
// 1. Split y into integer n and fractional f: y = n + f where f in [0, 1)
// 2. 2^n: bit shift (right for n < 0)
// 3. 2^f: polynomial approximation
// 4. Result = 2^n * 2^f

#ifndef EXP2_INT_H
#define EXP2_INT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Fixed-Point Format
// =============================================================================
//
// Input x: Q8.8 format (8 integer bits, 8 fractional bits)
//   - Range: [-128, 127] with 1/256 precision
//   - For softmax: typically [-8, 0] -> x in [-2048, 0]
//
// Output: Q16.16 format
//   - 2^(x/256) * 65536
//   - Range: [~22, 65536] for x in [-2048, 0]
//
// log2(e) = 1.4427 in Q8.8 = 369

#define EXP2_LOG2E_Q8_8  369    // log2(e) * 256 = 1.4427 * 256

// =============================================================================
// Scalar exp2 Functions
// =============================================================================

// Fast exp2 using bit manipulation + polynomial
// Input: x in Q8.8 (x/256 = real value)
// Output: 2^(x/256) in Q16.16
static inline int32_t exp2_int32(int32_t x) {
    // Clamp to valid range: [-8, 0] in real = [-2048, 0] in Q8.8
    if (x < -2048) return 0;
    if (x > 0) x = 0;

    // Split into integer n and fractional f
    // x is negative, so n = floor(x/256) is negative
    // f = x - n*256 is in [0, 255]
    int32_t n = x >> 8;           // Integer part (negative)
    int32_t f = x & 0xFF;         // Fractional part [0, 255]

    // If x is negative and has fractional part, adjust
    // e.g., x = -300: n = -2, f = -300 - (-512) = 212 (wrong)
    // Actually for arithmetic right shift, x = -300 >> 8 = -2, and -300 & 255 = 212
    // But we want: x = -300/256 = -1.17, so n = -2, f = 0.83
    // 212/256 = 0.83 is correct!

    // For negative x: n is floor(x/256), f = (x - n*256)/256 is in [0,1)
    // x >> 8 gives floor(x/256) for negative x due to arithmetic shift
    // x & 0xFF gives the fractional part correctly

    // 2^f for f in [0, 255] representing [0, 1)
    // Polynomial: 2^f' ≈ 1 + 0.6931*f' + 0.2402*f'² + 0.0555*f'³
    // where f' = f/256
    //
    // In Q16.16 output:
    // result = 65536 * (1 + c1*(f/256) + c2*(f/256)² + c3*(f/256)³)
    // result = 65536 + c1*65536*f/256 + c2*65536*f²/65536 + c3*65536*f³/16777216
    // result = 65536 + 256*c1*f + c2*f²/1024 + c3*f³/256/1024
    //
    // Coefficients:
    // c1 = 0.6931472 -> 256*c1*65536/256 = 45426 (multiply by f, shift by 8)
    // c2 = 0.2402265 -> 65536*c2 = 15743 (multiply by f², shift by 16)
    // c3 = 0.0555041 -> 65536*c3 = 3638 (multiply by f³, shift by 24)

    // Compute 2^f in Q16.16
    // term0 = 65536 (1.0)
    // term1 = 45426 * f >> 8 (≈ 0.693 * f scaled)
    // term2 = 15743 * f² >> 16
    // term3 = 3638 * f³ >> 24

    int64_t f64 = f;
    int64_t f2 = f64 * f64;          // f² in [0, 65025]
    int64_t f3 = f2 * f64;           // f³ in [0, 16581375]

    int32_t result = 65536;          // 1.0 in Q16.16
    result += (int32_t)((45426 * f64) >> 8);
    result += (int32_t)((15743 * f2) >> 16);
    result += (int32_t)((3638 * f3) >> 24);

    // 2^n for negative n: right shift
    if (n < 0) {
        int shift = -n;
        if (shift >= 16) return 0;   // Underflow
        result >>= shift;
    }
    // For positive n (shouldn't happen in softmax): left shift
    // Not needed for softmax since x <= 0

    return result;
}

// exp(x) using exp2: exp(x) = 2^(x * log2(e))
// Input: x in Q8.8
// Output: exp(x/256) in Q16.16
static inline int32_t exp_via_exp2(int32_t x) {
    // y = x * log2(e)
    // x is Q8.8, log2(e) is Q8.8 (369)
    // y = x * 369 >> 8 to stay in Q8.8
    int32_t y = (x * EXP2_LOG2E_Q8_8) >> 8;
    return exp2_int32(y);
}

// =============================================================================
// SVE Vectorized exp2 (inline assembly)
// =============================================================================

// Process 16 INT32 values at once
// Input: x[16] in Q8.8
// Output: out[16] = 2^(x/256) in Q16.16
void exp2_int32_sve(const int32_t* x, int32_t* out, int n);

// =============================================================================
// Pre-computed LUT for 2^f where f in [0,1)
// =============================================================================
// 256 entries: exp2_frac_lut[i] = 2^(i/256) * 65536
// Range: [65536, 131071] for i in [0, 255]

extern const int32_t exp2_frac_lut[256];

// LUT-based exp2 (faster but less accurate)
static inline int32_t exp2_int32_lut(int32_t x) {
    if (x < -2048) return 0;
    if (x > 0) x = 0;

    int32_t n = x >> 8;
    uint32_t f = (uint32_t)(x & 0xFF);

    // Linear interpolation between LUT entries
    int32_t y0 = exp2_frac_lut[f];
    int32_t y1 = (f < 255) ? exp2_frac_lut[f + 1] : exp2_frac_lut[255];

    // No interpolation needed since f is exact index
    int32_t result = y0;

    // Apply 2^n
    if (n < 0) {
        int shift = -n;
        if (shift >= 16) return 0;
        result >>= shift;
    }

    return result;
}

// =============================================================================
// Softmax-Specific Functions
// =============================================================================

// Compute softmax numerators: exp(S - max) for a row
// S: INT32 scores [n]
// max: row maximum (INT32)
// out: INT32 exp values [n] in Q16.16
// scale: Q16.16 scale to convert S to softmax input range
void softmax_exp_row(const int32_t* S, int32_t max_val, int32_t* out,
                     int32_t scale, int n);

// SVE version
void softmax_exp_row_sve(const int32_t* S, int32_t max_val, int32_t* out,
                         int32_t scale, int n);

// Find max of INT32 array using SVE
int32_t smax_int32_sve(const int32_t* x, int n);

// Sum of INT32 array using SVE (with saturation check)
int64_t sum_int32_sve(const int32_t* x, int n);

#ifdef __cplusplus
}
#endif

#endif // EXP2_INT_H
