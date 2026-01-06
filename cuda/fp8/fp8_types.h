/*
 * FP8 Type Definitions and Conversion Functions
 * Supports both E4M3 and E5M2 formats
 *
 * E4M3: sign(1) + exponent(4) + mantissa(3), bias=7, max=448
 * E5M2: sign(1) + exponent(5) + mantissa(2), bias=15, max=57344
 */

#ifndef FP8_TYPES_H
#define FP8_TYPES_H

#include <stdint.h>
#include <math.h>

typedef uint8_t fp8_e4m3_t;
typedef uint8_t fp8_e5m2_t;

/* E4M3 Constants */
#define FP8_E4M3_EXP_BITS  4
#define FP8_E4M3_MANT_BITS 3
#define FP8_E4M3_BIAS      7
#define FP8_E4M3_MAX_EXP   15
#define FP8_E4M3_MAX       448.0f

/* E5M2 Constants */
#define FP8_E5M2_EXP_BITS  5
#define FP8_E5M2_MANT_BITS 2
#define FP8_E5M2_BIAS      15
#define FP8_E5M2_MAX_EXP   31
#define FP8_E5M2_MAX       57344.0f

/* Convert float to FP8 E4M3 */
static inline fp8_e4m3_t float_to_fp8_e4m3(float f) {
    if (f != f) return 0x7F; /* NaN -> 0x7F (all mantissa bits set) */
    if (f == 0.0f) return 0x00;

    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + FP8_E4M3_BIAS;
    uint32_t mant = (bits >> 20) & 0x7; /* Take top 3 bits of mantissa */

    /* Handle overflow */
    if (exp >= 15) {
        exp = 15;
        mant = 0x6; /* Max normal value, not NaN */
    }
    /* Handle underflow */
    if (exp <= 0) {
        return (sign << 7); /* Zero with sign */
    }

    return (sign << 7) | ((exp & 0xF) << 3) | (mant & 0x7);
}

/* Convert FP8 E4M3 to float */
static inline float fp8_e4m3_to_float(fp8_e4m3_t x) {
    uint32_t sign = (x >> 7) & 1;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;

    /* Handle special cases */
    if (exp == 15 && mant == 0x7) {
        return sign ? -NAN : NAN;
    }
    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    /* Denormal */
    if (exp == 0) {
        float result = ldexpf((float)mant / 8.0f, -6);
        return sign ? -result : result;
    }

    /* Normal */
    float mantissa = 1.0f + (float)mant / 8.0f;
    float result = ldexpf(mantissa, exp - FP8_E4M3_BIAS);
    return sign ? -result : result;
}

/* Convert float to FP8 E5M2 */
static inline fp8_e5m2_t float_to_fp8_e5m2(float f) {
    if (f != f) return 0x7F; /* NaN */
    if (isinf(f)) return (f < 0) ? 0xFF : 0x7F; /* +/- Inf */
    if (f == 0.0f) return 0x00;

    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + FP8_E5M2_BIAS;
    uint32_t mant = (bits >> 21) & 0x3; /* Take top 2 bits of mantissa */

    /* Handle overflow -> Inf */
    if (exp >= 31) {
        return (sign << 7) | 0x7C; /* Infinity */
    }
    /* Handle underflow */
    if (exp <= 0) {
        return (sign << 7); /* Zero with sign */
    }

    return (sign << 7) | ((exp & 0x1F) << 2) | (mant & 0x3);
}

/* Convert FP8 E5M2 to float */
static inline float fp8_e5m2_to_float(fp8_e5m2_t x) {
    uint32_t sign = (x >> 7) & 1;
    uint32_t exp = (x >> 2) & 0x1F;
    uint32_t mant = x & 0x3;

    /* Handle special cases */
    if (exp == 31) {
        if (mant == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    }
    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    /* Denormal */
    if (exp == 0) {
        float result = ldexpf((float)mant / 4.0f, -14);
        return sign ? -result : result;
    }

    /* Normal */
    float mantissa = 1.0f + (float)mant / 4.0f;
    float result = ldexpf(mantissa, exp - FP8_E5M2_BIAS);
    return sign ? -result : result;
}

#endif /* FP8_TYPES_H */
