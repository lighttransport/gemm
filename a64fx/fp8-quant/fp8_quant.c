#include "fp8_quant.h"
#include <arm_sve.h>
#include <math.h>

// ============================================================================
// FP8 E4M3 Constants
// ============================================================================
// E4M3: 1 sign, 4 exp, 3 mantissa, bias=7
// Max representable: 448.0 (0x7F = 0_1111_110, avoiding NaN encoding 0x7F)
// Min normal: 2^-6 = 0.015625
// Min subnormal: 2^-9 = 0.001953125

#define FP8_E4M3_EXP_BIAS     7
#define FP8_E4M3_MAX_VAL      448.0f
#define FP8_E4M3_MIN_NORMAL   0.015625f

// ============================================================================
// FP8 E5M2 Constants
// ============================================================================
// E5M2: 1 sign, 5 exp, 2 mantissa, bias=15
// Max representable: 57344.0
// Min normal: 2^-14
// Min subnormal: 2^-16

#define FP8_E5M2_EXP_BIAS     15
#define FP8_E5M2_MAX_VAL      57344.0f
#define FP8_E5M2_MIN_NORMAL   6.103515625e-5f

// ============================================================================
// Scalar Reference Implementations
// ============================================================================

fp8_e4m3_t fp16_to_fp8_e4m3_scalar(uint16_t x) {
    // Extract FP16 components: 1 sign, 5 exp, 10 mantissa, bias=15
    uint32_t sign = (x >> 15) & 1;
    uint32_t exp = (x >> 10) & 0x1F;
    uint32_t mant = x & 0x3FF;

    // Handle special cases
    if (exp == 0x1F) {
        // Inf/NaN -> max value with sign (E4M3 uses 0x7F for NaN)
        return (sign << 7) | 0x7E;  // Return max finite value
    }

    if (exp == 0 && mant == 0) {
        // Zero
        return sign << 7;
    }

    // Convert exponent from FP16 bias (15) to E4M3 bias (7)
    int32_t unbiased_exp = (int32_t)exp - 15;
    int32_t e4m3_exp = unbiased_exp + 7;

    // Handle overflow (E4M3 max exp is 15, max value with mant=6 is 448)
    if (e4m3_exp > 15) {
        return (sign << 7) | 0x7E;  // Overflow -> max value
    }

    if (e4m3_exp <= 0) {
        // Subnormal or underflow
        if (e4m3_exp < -3) {
            return sign << 7;  // Underflow to zero
        }
        // Subnormal: shift mantissa
        uint32_t shift = 1 - e4m3_exp;
        mant = (mant | 0x400) >> (shift + 7);  // Add implicit 1, shift
        return (sign << 7) | (mant & 0x7);
    }

    // Normal number: round mantissa from 10 bits to 3 bits
    uint32_t round_bit = (mant >> 6) & 1;
    uint32_t e4m3_mant = (mant >> 7) + round_bit;

    // Handle mantissa overflow from rounding
    if (e4m3_mant > 7) {
        e4m3_mant = 0;
        e4m3_exp++;
    }

    // Check for overflow after rounding (exp > 15, or exp=15 & mant=7 is NaN)
    if (e4m3_exp > 15 || (e4m3_exp == 15 && e4m3_mant > 6)) {
        return (sign << 7) | 0x7E;  // Clamp to max value
    }

    return (sign << 7) | (e4m3_exp << 3) | e4m3_mant;
}

fp8_e5m2_t fp16_to_fp8_e5m2_scalar(uint16_t x) {
    // Extract FP16 components
    uint32_t sign = (x >> 15) & 1;
    uint32_t exp = (x >> 10) & 0x1F;
    uint32_t mant = x & 0x3FF;

    // Handle special cases
    if (exp == 0x1F) {
        if (mant != 0) {
            return (sign << 7) | 0x7F;  // NaN
        }
        return (sign << 7) | 0x7C;  // Inf
    }

    if (exp == 0 && mant == 0) {
        return sign << 7;  // Zero
    }

    // E5M2 and FP16 have the same exponent bias (15)
    // Just need to round mantissa from 10 bits to 2 bits
    uint32_t round_bit = (mant >> 7) & 1;
    uint32_t e5m2_mant = (mant >> 8) + round_bit;
    uint32_t e5m2_exp = exp;

    if (e5m2_mant > 3) {
        e5m2_mant = 0;
        e5m2_exp++;
        if (e5m2_exp >= 31) {
            return (sign << 7) | 0x7C;  // Overflow to Inf
        }
    }

    return (sign << 7) | (e5m2_exp << 2) | e5m2_mant;
}

fp8_e4m3_t fp32_to_fp8_e4m3_scalar(float x) {
    union { float f; uint32_t u; } conv = {x};
    uint32_t bits = conv.u;

    uint32_t sign = (bits >> 31) & 1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;

    // Handle special cases
    if (exp == 0xFF) {
        return (sign << 7) | 0x7E;  // Inf/NaN -> max
    }

    if (exp == 0 && mant == 0) {
        return sign << 7;  // Zero
    }

    // Convert exp from FP32 bias (127) to E4M3 bias (7)
    int32_t unbiased_exp = (int32_t)exp - 127;
    int32_t e4m3_exp = unbiased_exp + 7;

    // Handle overflow
    if (e4m3_exp > 15) {
        return (sign << 7) | 0x7E;
    }

    if (e4m3_exp <= 0) {
        if (e4m3_exp < -3) {
            return sign << 7;  // Underflow
        }
        uint32_t shift = 1 - e4m3_exp;
        mant = (mant | 0x800000) >> (shift + 20);
        return (sign << 7) | (mant & 0x7);
    }

    // Round from 23 bits to 3 bits
    uint32_t round_bit = (mant >> 19) & 1;
    uint32_t e4m3_mant = (mant >> 20) + round_bit;

    if (e4m3_mant > 7) {
        e4m3_mant = 0;
        e4m3_exp++;
    }

    // Check for overflow after rounding
    if (e4m3_exp > 15 || (e4m3_exp == 15 && e4m3_mant > 6)) {
        return (sign << 7) | 0x7E;
    }

    return (sign << 7) | (e4m3_exp << 3) | e4m3_mant;
}

fp8_e5m2_t fp32_to_fp8_e5m2_scalar(float x) {
    union { float f; uint32_t u; } conv = {x};
    uint32_t bits = conv.u;

    uint32_t sign = (bits >> 31) & 1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp == 0xFF) {
        if (mant != 0) {
            return (sign << 7) | 0x7F;  // NaN
        }
        return (sign << 7) | 0x7C;  // Inf
    }

    if (exp == 0 && mant == 0) {
        return sign << 7;
    }

    // Convert exp from FP32 bias (127) to E5M2 bias (15)
    int32_t unbiased_exp = (int32_t)exp - 127;
    int32_t e5m2_exp = unbiased_exp + 15;

    if (e5m2_exp >= 31) {
        return (sign << 7) | 0x7C;  // Overflow to Inf
    }
    if (e5m2_exp <= 0) {
        if (e5m2_exp < -2) {
            return sign << 7;  // Underflow
        }
        uint32_t shift = 1 - e5m2_exp;
        mant = (mant | 0x800000) >> (shift + 21);
        return (sign << 7) | (mant & 0x3);
    }

    // Round from 23 bits to 2 bits
    uint32_t round_bit = (mant >> 20) & 1;
    uint32_t e5m2_mant = (mant >> 21) + round_bit;

    if (e5m2_mant > 3) {
        e5m2_mant = 0;
        e5m2_exp++;
        if (e5m2_exp >= 31) {
            return (sign << 7) | 0x7C;
        }
    }

    return (sign << 7) | (e5m2_exp << 2) | e5m2_mant;
}

// ============================================================================
// Dequantization (for validation)
// ============================================================================

float fp8_e4m3_to_fp32_scalar(fp8_e4m3_t x) {
    uint32_t sign = (x >> 7) & 1;
    uint32_t exp = (x >> 3) & 0xF;
    uint32_t mant = x & 0x7;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Subnormal
        float val = mant * powf(2.0f, -9.0f);
        return sign ? -val : val;
    }
    if (exp == 15 && mant == 7) {
        return NAN;  // Special NaN encoding
    }

    int32_t unbiased = (int32_t)exp - 7;
    float val = (1.0f + mant / 8.0f) * powf(2.0f, unbiased);
    return sign ? -val : val;
}

float fp8_e5m2_to_fp32_scalar(fp8_e5m2_t x) {
    uint32_t sign = (x >> 7) & 1;
    uint32_t exp = (x >> 2) & 0x1F;
    uint32_t mant = x & 0x3;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = mant * powf(2.0f, -16.0f);
        return sign ? -val : val;
    }
    if (exp == 31) {
        if (mant != 0) return NAN;
        return sign ? -INFINITY : INFINITY;
    }

    int32_t unbiased = (int32_t)exp - 15;
    float val = (1.0f + mant / 4.0f) * powf(2.0f, unbiased);
    return sign ? -val : val;
}

// Software FP32 to FP16 conversion
static uint16_t fp32_to_fp16_sw(float f) {
    union { float f; uint32_t u; } conv = {f};
    uint32_t bits = conv.u;

    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;

    if (exp <= 0) {
        return sign;  // Underflow to zero
    } else if (exp >= 31) {
        return sign | 0x7C00;  // Overflow to Inf
    }
    return sign | (exp << 10) | mant;
}

uint16_t fp8_e4m3_to_fp16_scalar(fp8_e4m3_t x) {
    float f = fp8_e4m3_to_fp32_scalar(x);
    return fp32_to_fp16_sw(f);
}

uint16_t fp8_e5m2_to_fp16_scalar(fp8_e5m2_t x) {
    float f = fp8_e5m2_to_fp32_scalar(x);
    return fp32_to_fp16_sw(f);
}

// ============================================================================
// SVE-Optimized FP16 -> FP8 E4M3
// ============================================================================
// Strategy: Use SVE to process 32 FP16 values -> 32 FP8 values per iteration
// A64FX has 512-bit vectors = 32 x FP16 = 64 x FP8

void fp16_to_fp8_e4m3_sve(const uint16_t* src, fp8_e4m3_t* dst, size_t n) {
    // For subnormal handling complexity, use scalar for small values
    // SVE handles normal range efficiently
    size_t i = 0;

    svbool_t pg16 = svptrue_b16();
    size_t vl16 = svcnth();

    // Constants
    svuint16_t sign_mask = svdup_u16(0x8000);
    svuint16_t exp_mask = svdup_u16(0x7C00);
    svuint16_t mant_mask = svdup_u16(0x03FF);

    // Max E4M3 value is 448 = 0x5F00 in FP16
    // Min normal E4M3 = 2^-6 = 0.015625, in FP16 this is exp=9 = 0x2400
    svuint16_t max_fp16 = svdup_u16(0x5F00);
    svuint16_t min_normal_fp16 = svdup_u16(0x2400);  // 2^-6 in FP16 = E4M3 exp=1
    svuint16_t e4m3_max = svdup_u16(0x7E);

    for (; i + vl16 <= n; i += vl16) {
        svuint16_t x = svld1_u16(pg16, src + i);

        // Extract sign and absolute value
        svuint16_t sign = svand_u16_x(pg16, x, sign_mask);
        svuint16_t abs_x = svand_n_u16_x(pg16, x, 0x7FFF);
        svuint16_t sign8 = svlsr_n_u16_x(pg16, sign, 8);

        // Check overflow/underflow BEFORE processing
        svbool_t overflow = svcmpgt_u16(pg16, abs_x, max_fp16);
        svbool_t underflow = svcmplt_u16(pg16, abs_x, min_normal_fp16);
        svbool_t is_zero = svcmpeq_u16(pg16, abs_x, svdup_u16(0));

        // Extract components from clamped value
        svuint16_t clamped = svsel_u16(overflow, max_fp16, abs_x);
        svuint16_t exp16 = svlsr_n_u16_x(pg16, svand_u16_x(pg16, clamped, exp_mask), 10);
        svuint16_t mant16 = svand_u16_x(pg16, clamped, mant_mask);

        // Convert exponent: E4M3_exp = FP16_exp - 8
        svint16_t exp_s = svsub_n_s16_x(pg16, svreinterpret_s16_u16(exp16), 8);
        exp_s = svmax_n_s16_x(pg16, exp_s, 1);  // Min exp=1 for normal range
        svuint16_t e4m3_exp = svreinterpret_u16_s16(exp_s);

        // Round mantissa from 10 bits to 3 bits
        svuint16_t round_bit = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant16, 6), 1);
        svuint16_t e4m3_mant = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant16, 7), round_bit);

        // Handle mantissa overflow from rounding
        svbool_t mant_overflow = svcmpgt_u16(pg16, e4m3_mant, svdup_u16(7));
        e4m3_mant = svsel_u16(mant_overflow, svdup_u16(0), e4m3_mant);
        e4m3_exp = svadd_u16_x(pg16, e4m3_exp,
                               svsel_u16(mant_overflow, svdup_u16(1), svdup_u16(0)));

        // Clamp to valid range
        svbool_t final_overflow = svcmpgt_u16(pg16, e4m3_exp, svdup_u16(15));
        svbool_t at_max = svand_b_z(pg16, svcmpeq_u16(pg16, e4m3_exp, svdup_u16(15)),
                                    svcmpgt_u16(pg16, e4m3_mant, svdup_u16(6)));
        final_overflow = svorr_b_z(pg16, final_overflow, at_max);
        e4m3_exp = svsel_u16(final_overflow, svdup_u16(15), e4m3_exp);
        e4m3_mant = svsel_u16(final_overflow, svdup_u16(6), e4m3_mant);

        // Combine: (sign >> 8) | (exp << 3) | mant
        svuint16_t result = svorr_u16_x(pg16, sign8,
                           svorr_u16_x(pg16, svlsl_n_u16_x(pg16, e4m3_exp, 3), e4m3_mant));

        // Handle special cases: zeros, underflow, overflow
        result = svsel_u16(is_zero, sign8, result);
        result = svsel_u16(overflow, svorr_u16_x(pg16, sign8, e4m3_max), result);

        // For underflow (subnormal range), use scalar fallback via storing intermediate
        // For now, mark underflow values to be processed by scalar loop
        // This is a simplification - a full implementation would handle subnormals in SVE
        svuint16_t underflow_marker = svdup_u16(0xFF);
        result = svsel_u16(underflow, underflow_marker, result);

        // Narrow to 8-bit
        svuint8_t result8 = svuzp1_u8(svreinterpret_u8_u16(result),
                                       svreinterpret_u8_u16(result));

        svbool_t pg8_half = svwhilelt_b8(0UL, vl16);
        svst1_u8(pg8_half, dst + i, result8);
    }

    // Fix underflow markers and handle remaining elements with scalar
    for (size_t j = 0; j < i; j++) {
        if (dst[j] == 0xFF) {
            dst[j] = fp16_to_fp8_e4m3_scalar(src[j]);
        }
    }

    for (; i < n; i++) {
        dst[i] = fp16_to_fp8_e4m3_scalar(src[i]);
    }
}

// ============================================================================
// SVE-Optimized FP16 -> FP8 E5M2
// ============================================================================

void fp16_to_fp8_e5m2_sve(const uint16_t* src, fp8_e5m2_t* dst, size_t n) {
    size_t i = 0;

    svbool_t pg16 = svptrue_b16();
    size_t vl16 = svcnth();

    // E5M2 has same exponent range as FP16, just smaller mantissa
    svuint16_t sign_mask = svdup_u16(0x8000);
    svuint16_t exp_mask = svdup_u16(0x7C00);
    svuint16_t mant_mask = svdup_u16(0x03FF);

    for (; i + vl16 <= n; i += vl16) {
        svuint16_t x = svld1_u16(pg16, src + i);

        // Extract components
        svuint16_t sign = svand_u16_x(pg16, x, sign_mask);
        svuint16_t exp16 = svand_u16_x(pg16, x, exp_mask);
        svuint16_t mant16 = svand_u16_x(pg16, x, mant_mask);

        // E5M2 exponent = FP16 exponent (same bias of 15)
        svuint16_t e5m2_exp = svlsr_n_u16_x(pg16, exp16, 10);

        // Round mantissa from 10 bits to 2 bits
        // Take top 2 bits, round based on bit 7
        svuint16_t round_bit = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant16, 7), 1);
        svuint16_t e5m2_mant = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant16, 8), round_bit);

        // Handle mantissa overflow
        svbool_t mant_overflow = svcmpgt_u16(pg16, e5m2_mant, svdup_u16(3));
        e5m2_mant = svsel_u16(mant_overflow, svdup_u16(0), e5m2_mant);
        e5m2_exp = svadd_u16_x(pg16, e5m2_exp,
                              svsel_u16(mant_overflow, svdup_u16(1), svdup_u16(0)));

        // Handle exponent overflow -> Inf
        svbool_t exp_overflow = svcmpgt_u16(pg16, e5m2_exp, svdup_u16(30));
        e5m2_exp = svsel_u16(exp_overflow, svdup_u16(31), e5m2_exp);
        e5m2_mant = svsel_u16(exp_overflow, svdup_u16(0), e5m2_mant);

        // Handle Inf/NaN passthrough
        svbool_t is_special = svcmpeq_u16(pg16, svlsr_n_u16_x(pg16, exp16, 10), svdup_u16(31));
        svbool_t is_nan = svand_b_z(pg16, is_special,
                                    svcmpne_u16(pg16, mant16, svdup_u16(0)));
        e5m2_exp = svsel_u16(is_special, svdup_u16(31), e5m2_exp);
        e5m2_mant = svsel_u16(is_nan, svdup_u16(3),
                             svsel_u16(is_special, svdup_u16(0), e5m2_mant));

        // Combine: (sign >> 8) | (exp << 2) | mant
        svuint16_t sign8 = svlsr_n_u16_x(pg16, sign, 8);
        svuint16_t result = svorr_u16_x(pg16, sign8,
                           svorr_u16_x(pg16, svlsl_n_u16_x(pg16, e5m2_exp, 2), e5m2_mant));

        // Narrow to 8-bit
        svuint8_t result8 = svuzp1_u8(svreinterpret_u8_u16(result),
                                       svreinterpret_u8_u16(result));

        svbool_t pg8_half = svwhilelt_b8(0UL, vl16);
        svst1_u8(pg8_half, dst + i, result8);
    }

    // Handle remaining
    for (; i < n; i++) {
        dst[i] = fp16_to_fp8_e5m2_scalar(src[i]);
    }
}

// ============================================================================
// SVE-Optimized FP32 -> FP8 E4M3
// ============================================================================

void fp32_to_fp8_e4m3_sve(const float* src, fp8_e4m3_t* dst, size_t n) {
    size_t i = 0;

    svbool_t pg32 = svptrue_b32();
    size_t vl32 = svcntw();

    // Max E4M3 value is 448.0, min normal is 2^-6 = 0.015625
    svfloat32_t max_val = svdup_f32(448.0f);
    svfloat32_t min_normal = svdup_f32(0.015625f);  // 2^-6
    svuint32_t e4m3_max = svdup_u32(0x7E);

    for (; i + vl32 <= n; i += vl32) {
        svfloat32_t x = svld1_f32(pg32, src + i);

        svuint32_t bits = svreinterpret_u32_f32(x);
        svuint32_t sign = svand_n_u32_x(pg32, bits, 0x80000000);
        svuint32_t sign8 = svlsr_n_u32_x(pg32, sign, 24);

        svfloat32_t abs_x = svabs_f32_x(pg32, x);

        // Check for overflow, underflow, zero
        svbool_t overflow = svcmpgt_f32(pg32, abs_x, max_val);
        svbool_t underflow = svcmplt_f32(pg32, abs_x, min_normal);
        svbool_t is_zero = svcmpeq_f32(pg32, abs_x, svdup_f32(0.0f));

        // Clamp to valid range
        abs_x = svmin_f32_x(pg32, abs_x, max_val);
        abs_x = svmax_f32_x(pg32, abs_x, min_normal);
        svuint32_t abs_bits = svreinterpret_u32_f32(abs_x);

        // Extract exponent and mantissa
        svuint32_t exp32 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, abs_bits, 23), 0xFF);
        svuint32_t mant32 = svand_n_u32_x(pg32, abs_bits, 0x7FFFFF);

        // Convert exponent: E4M3_exp = FP32_exp - 120
        svint32_t exp_s = svsub_n_s32_x(pg32, svreinterpret_s32_u32(exp32), 120);
        exp_s = svmax_n_s32_x(pg32, exp_s, 1);  // Min exp=1 for normal
        svuint32_t e4m3_exp = svreinterpret_u32_s32(exp_s);

        // Round mantissa from 23 bits to 3 bits
        svuint32_t round_bit = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant32, 19), 1);
        svuint32_t e4m3_mant = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant32, 20), round_bit);

        // Handle mantissa overflow from rounding
        svbool_t mant_overflow = svcmpgt_u32(pg32, e4m3_mant, svdup_u32(7));
        e4m3_mant = svsel_u32(mant_overflow, svdup_u32(0), e4m3_mant);
        e4m3_exp = svadd_u32_x(pg32, e4m3_exp,
                              svsel_u32(mant_overflow, svdup_u32(1), svdup_u32(0)));

        // Clamp to max (exp=15, mant=6)
        svbool_t final_overflow = svcmpgt_u32(pg32, e4m3_exp, svdup_u32(15));
        svbool_t at_max = svand_b_z(pg32, svcmpeq_u32(pg32, e4m3_exp, svdup_u32(15)),
                                    svcmpgt_u32(pg32, e4m3_mant, svdup_u32(6)));
        final_overflow = svorr_b_z(pg32, final_overflow, at_max);
        e4m3_exp = svsel_u32(final_overflow, svdup_u32(15), e4m3_exp);
        e4m3_mant = svsel_u32(final_overflow, svdup_u32(6), e4m3_mant);

        // Combine result
        svuint32_t result32 = svorr_u32_x(pg32, sign8,
                              svorr_u32_x(pg32, svlsl_n_u32_x(pg32, e4m3_exp, 3), e4m3_mant));

        // Handle special cases
        result32 = svsel_u32(is_zero, sign8, result32);
        result32 = svsel_u32(overflow, svorr_u32_x(pg32, sign8, e4m3_max), result32);

        // Mark underflow for scalar fixup
        result32 = svsel_u32(underflow, svdup_u32(0xFF), result32);

        // Narrow from 32-bit to 8-bit
        svuint16_t r16 = svuzp1_u16(svreinterpret_u16_u32(result32),
                                     svreinterpret_u16_u32(result32));
        svuint8_t r8 = svuzp1_u8(svreinterpret_u8_u16(r16),
                                  svreinterpret_u8_u16(r16));

        svbool_t pg8_quarter = svwhilelt_b8(0UL, vl32);
        svst1_u8(pg8_quarter, dst + i, r8);
    }

    // Fix underflow markers and handle remaining
    for (size_t j = 0; j < i; j++) {
        if (dst[j] == 0xFF) {
            dst[j] = fp32_to_fp8_e4m3_scalar(src[j]);
        }
    }

    for (; i < n; i++) {
        dst[i] = fp32_to_fp8_e4m3_scalar(src[i]);
    }
}

// ============================================================================
// SVE-Optimized FP32 -> FP8 E5M2
// ============================================================================

void fp32_to_fp8_e5m2_sve(const float* src, fp8_e5m2_t* dst, size_t n) {
    size_t i = 0;

    svbool_t pg32 = svptrue_b32();
    size_t vl32 = svcntw();

    svfloat32_t max_val = svdup_f32(57344.0f);
    // Min normal for E5M2: 2^-14 = 6.1e-5
    // Min subnormal: 2^-16 = 1.5e-5
    // Values smaller than 2^-17 should underflow to zero
    svfloat32_t min_subnormal = svdup_f32(7.6293945e-6f);  // 2^-17

    for (; i + vl32 <= n; i += vl32) {
        svfloat32_t x = svld1_f32(pg32, src + i);

        svuint32_t bits = svreinterpret_u32_f32(x);
        svuint32_t sign = svand_n_u32_x(pg32, bits, 0x80000000);
        svuint32_t sign8 = svlsr_n_u32_x(pg32, sign, 24);

        svfloat32_t abs_x = svabs_f32_x(pg32, x);

        // Check for underflow (very small values)
        svbool_t underflow = svcmplt_f32(pg32, abs_x, min_subnormal);

        // Check for overflow
        svbool_t overflow = svcmpgt_f32(pg32, abs_x, max_val);

        // Check for zero
        svbool_t is_zero = svcmpeq_f32(pg32, abs_x, svdup_f32(0.0f));

        // Clamp
        abs_x = svmin_f32_x(pg32, abs_x, max_val);
        svuint32_t abs_bits = svreinterpret_u32_f32(abs_x);

        svuint32_t exp32 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, abs_bits, 23), 0xFF);
        svuint32_t mant32 = svand_n_u32_x(pg32, abs_bits, 0x7FFFFF);

        // E5M2_exp = FP32_exp - 112
        svint32_t exp_s = svsub_n_s32_x(pg32, svreinterpret_s32_u32(exp32), 112);

        svbool_t exp_neg = svcmplt_s32(pg32, exp_s, svdup_s32(1));
        exp_s = svsel_s32(exp_neg, svdup_s32(0), exp_s);
        svuint32_t e5m2_exp = svreinterpret_u32_s32(exp_s);

        // Round mantissa from 23 bits to 2 bits
        svuint32_t round_bit = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant32, 20), 1);
        svuint32_t e5m2_mant = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant32, 21), round_bit);

        // Handle mantissa overflow
        svbool_t mant_overflow = svcmpgt_u32(pg32, e5m2_mant, svdup_u32(3));
        e5m2_mant = svsel_u32(mant_overflow, svdup_u32(0), e5m2_mant);
        e5m2_exp = svadd_u32_x(pg32, e5m2_exp,
                              svsel_u32(mant_overflow, svdup_u32(1), svdup_u32(0)));
        e5m2_exp = svmin_n_u32_x(pg32, e5m2_exp, 30);

        // Combine
        svuint32_t result32 = svorr_u32_x(pg32, sign8,
                              svorr_u32_x(pg32, svlsl_n_u32_x(pg32, e5m2_exp, 2), e5m2_mant));

        // Handle zeros and underflows
        result32 = svsel_u32(is_zero, sign8, result32);
        result32 = svsel_u32(underflow, sign8, result32);

        // Handle overflow -> Inf (0x7C with sign)
        result32 = svsel_u32(overflow, svorr_n_u32_x(pg32, sign8, 0x7C), result32);

        // Narrow
        svuint16_t r16 = svuzp1_u16(svreinterpret_u16_u32(result32),
                                     svreinterpret_u16_u32(result32));
        svuint8_t r8 = svuzp1_u8(svreinterpret_u8_u16(r16),
                                  svreinterpret_u8_u16(r16));

        svbool_t pg8_quarter = svwhilelt_b8(0UL, vl32);
        svst1_u8(pg8_quarter, dst + i, r8);
    }

    for (; i < n; i++) {
        dst[i] = fp32_to_fp8_e5m2_scalar(src[i]);
    }
}
