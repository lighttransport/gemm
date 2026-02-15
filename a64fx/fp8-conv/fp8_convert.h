/*
 * FP8 to FP16/FP32 Conversion Kernels for A64FX
 *
 * FP8 E4M3: 1 sign, 4 exponent (bias=7), 3 mantissa
 * FP8 E5M2: 1 sign, 5 exponent (bias=15), 2 mantissa
 *
 * FP16: 1 sign, 5 exponent (bias=15), 10 mantissa
 * FP32: 1 sign, 8 exponent (bias=127), 23 mantissa
 */

#ifndef FP8_CONVERT_H
#define FP8_CONVERT_H

#include <stdint.h>
#include <arm_sve.h>

// FP8 type definitions
typedef uint8_t fp8_e4m3_t;
typedef uint8_t fp8_e5m2_t;

// Bias values
#define FP8_E4M3_BIAS 7
#define FP8_E5M2_BIAS 15
#define FP16_BIAS 15
#define FP32_BIAS 127

// ============================================================================
// Lookup Tables (for gather-based conversion)
// ============================================================================
extern uint16_t fp8_e4m3_to_fp16_lut[256];
extern uint32_t fp8_e4m3_to_fp32_lut[256];
extern uint16_t fp8_e5m2_to_fp16_lut[256];
extern uint32_t fp8_e5m2_to_fp32_lut[256];

void init_fp8_luts(void);

// ============================================================================
// Scalar Reference Implementations
// ============================================================================

// E4M3 conversions
static inline uint16_t fp8_e4m3_to_fp16_scalar(fp8_e4m3_t x) {
    uint8_t sign = (x >> 7) & 1;
    uint8_t exp = (x >> 3) & 0xF;
    uint8_t mant = x & 0x7;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            return (uint16_t)sign << 15;
        }
        // Subnormal: normalize
        // E4M3 subnormal = 2^(-6) * (0.mant)
        // Need to convert to FP16 normal
        int shift = 0;
        while ((mant & 0x4) == 0) {
            mant <<= 1;
            shift++;
        }
        mant &= 0x3; // Remove implicit 1
        exp = FP16_BIAS - FP8_E4M3_BIAS - shift;
        return ((uint16_t)sign << 15) | ((uint16_t)exp << 10) | ((uint16_t)mant << 7);
    }
    if (exp == 15) {
        // E4M3 has no Inf, all exp=15 are NaN
        return ((uint16_t)sign << 15) | (0x1F << 10) | ((uint16_t)mant << 7);
    }

    // Normal number
    uint16_t new_exp = exp + (FP16_BIAS - FP8_E4M3_BIAS); // exp + 8
    return ((uint16_t)sign << 15) | (new_exp << 10) | ((uint16_t)mant << 7);
}

static inline uint32_t fp8_e4m3_to_fp32_scalar(fp8_e4m3_t x) {
    uint8_t sign = (x >> 7) & 1;
    uint8_t exp = (x >> 3) & 0xF;
    uint8_t mant = x & 0x7;

    if (exp == 0) {
        if (mant == 0) {
            return (uint32_t)sign << 31;
        }
        // Subnormal
        int shift = 0;
        while ((mant & 0x4) == 0) {
            mant <<= 1;
            shift++;
        }
        mant &= 0x3;
        exp = FP32_BIAS - FP8_E4M3_BIAS - shift;
        return ((uint32_t)sign << 31) | ((uint32_t)exp << 23) | ((uint32_t)mant << 20);
    }
    if (exp == 15) {
        return ((uint32_t)sign << 31) | (0xFF << 23) | ((uint32_t)mant << 20);
    }

    uint32_t new_exp = exp + (FP32_BIAS - FP8_E4M3_BIAS); // exp + 120
    return ((uint32_t)sign << 31) | (new_exp << 23) | ((uint32_t)mant << 20);
}

// E5M2 conversions
static inline uint16_t fp8_e5m2_to_fp16_scalar(fp8_e5m2_t x) {
    uint8_t sign = (x >> 7) & 1;
    uint8_t exp = (x >> 2) & 0x1F;
    uint8_t mant = x & 0x3;

    if (exp == 0) {
        if (mant == 0) {
            return (uint16_t)sign << 15;
        }
        // Subnormal
        int shift = 0;
        while ((mant & 0x2) == 0) {
            mant <<= 1;
            shift++;
        }
        mant &= 0x1;
        exp = FP16_BIAS - FP8_E5M2_BIAS - shift; // 0 - shift
        return ((uint16_t)sign << 15) | ((uint16_t)exp << 10) | ((uint16_t)mant << 8);
    }
    if (exp == 31) {
        // Inf or NaN
        return ((uint16_t)sign << 15) | (0x1F << 10) | ((uint16_t)mant << 8);
    }

    // Normal: same bias, just shift mantissa
    return ((uint16_t)sign << 15) | ((uint16_t)exp << 10) | ((uint16_t)mant << 8);
}

static inline uint32_t fp8_e5m2_to_fp32_scalar(fp8_e5m2_t x) {
    uint8_t sign = (x >> 7) & 1;
    uint8_t exp = (x >> 2) & 0x1F;
    uint8_t mant = x & 0x3;

    if (exp == 0) {
        if (mant == 0) {
            return (uint32_t)sign << 31;
        }
        // Subnormal
        int shift = 0;
        while ((mant & 0x2) == 0) {
            mant <<= 1;
            shift++;
        }
        mant &= 0x1;
        exp = FP32_BIAS - FP8_E5M2_BIAS - shift; // 112 - shift
        return ((uint32_t)sign << 31) | ((uint32_t)exp << 23) | ((uint32_t)mant << 21);
    }
    if (exp == 31) {
        return ((uint32_t)sign << 31) | (0xFF << 23) | ((uint32_t)mant << 21);
    }

    uint32_t new_exp = exp + (FP32_BIAS - FP8_E5M2_BIAS); // exp + 112
    return ((uint32_t)sign << 31) | (new_exp << 23) | ((uint32_t)mant << 21);
}

// ============================================================================
// LUT-based Gather Implementations (uses FCVT-like approach via table)
// ============================================================================

void fp8_e4m3_to_fp16_gather(const fp8_e4m3_t* src, uint16_t* dst, int n);
void fp8_e4m3_to_fp32_gather(const fp8_e4m3_t* src, uint32_t* dst, int n);
void fp8_e5m2_to_fp16_gather(const fp8_e5m2_t* src, uint16_t* dst, int n);
void fp8_e5m2_to_fp32_gather(const fp8_e5m2_t* src, uint32_t* dst, int n);

// ============================================================================
// Bit Arithmetic - ARM64 Base Instructions (EX* pipe)
// ============================================================================

void fp8_e4m3_to_fp16_base(const fp8_e4m3_t* src, uint16_t* dst, int n);
void fp8_e4m3_to_fp32_base(const fp8_e4m3_t* src, uint32_t* dst, int n);
void fp8_e5m2_to_fp16_base(const fp8_e5m2_t* src, uint16_t* dst, int n);
void fp8_e5m2_to_fp32_base(const fp8_e5m2_t* src, uint32_t* dst, int n);

// ============================================================================
// Bit Arithmetic - SVE Instructions (FL* pipe)
// ============================================================================

void fp8_e4m3_to_fp16_sve(const fp8_e4m3_t* src, uint16_t* dst, int n);
void fp8_e4m3_to_fp32_sve(const fp8_e4m3_t* src, uint32_t* dst, int n);
void fp8_e5m2_to_fp16_sve(const fp8_e5m2_t* src, uint16_t* dst, int n);
void fp8_e5m2_to_fp32_sve(const fp8_e5m2_t* src, uint32_t* dst, int n);

#endif // FP8_CONVERT_H
