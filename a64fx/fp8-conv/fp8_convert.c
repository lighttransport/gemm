/*
 * FP8 to FP16/FP32 Conversion Kernels Implementation
 */

#include "fp8_convert.h"
#include <arm_sve.h>

// ============================================================================
// Lookup Tables
// ============================================================================

uint16_t fp8_e4m3_to_fp16_lut[256];
uint32_t fp8_e4m3_to_fp32_lut[256];
uint16_t fp8_e5m2_to_fp16_lut[256];
uint32_t fp8_e5m2_to_fp32_lut[256];

// Extended LUT for FP16 (stored as 32-bit for gather compatibility)
uint32_t fp8_e4m3_to_fp16_lut32[256];
uint32_t fp8_e5m2_to_fp16_lut32[256];

void init_fp8_luts(void) {
    for (int i = 0; i < 256; i++) {
        fp8_e4m3_to_fp16_lut[i] = fp8_e4m3_to_fp16_scalar((fp8_e4m3_t)i);
        fp8_e4m3_to_fp32_lut[i] = fp8_e4m3_to_fp32_scalar((fp8_e4m3_t)i);
        fp8_e5m2_to_fp16_lut[i] = fp8_e5m2_to_fp16_scalar((fp8_e5m2_t)i);
        fp8_e5m2_to_fp32_lut[i] = fp8_e5m2_to_fp32_scalar((fp8_e5m2_t)i);
        // Extended LUTs for gather
        fp8_e4m3_to_fp16_lut32[i] = fp8_e4m3_to_fp16_lut[i];
        fp8_e5m2_to_fp16_lut32[i] = fp8_e5m2_to_fp16_lut[i];
    }
}

// ============================================================================
// LUT-based Gather Implementations
// Uses SVE 32-bit gather loads (16-bit gather not available on A64FX)
// ============================================================================

void fp8_e4m3_to_fp16_gather(const fp8_e4m3_t* src, uint16_t* dst, int n) {
    int i = 0;
    svbool_t pg = svptrue_b32();
    int vl = svcntw();  // 32-bit vector length

    for (; i + vl <= n; i += vl) {
        // Load FP8 values as bytes and zero-extend to 32-bit
        svbool_t pg8 = svwhilelt_b8(0, vl);
        svuint8_t fp8_bytes = svld1_u8(pg8, src + i);
        svuint16_t indices16 = svunpklo_u16(fp8_bytes);
        svuint32_t indices = svunpklo_u32(indices16);

        // Convert to byte offsets (*4 for uint32)
        svuint32_t byte_offsets = svlsl_n_u32_x(pg, indices, 2);

        // Gather from 32-bit LUT
        svuint32_t result32 = svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp16_lut32, byte_offsets);

        // Pack back to 16-bit and store
        svuint16_t result16 = svuzp1_u16(svreinterpret_u16(result32), svreinterpret_u16(result32));
        svst1_u16(svwhilelt_b16(0, vl), dst + i, result16);
    }

    // Remainder
    for (; i < n; i++) {
        dst[i] = fp8_e4m3_to_fp16_lut[src[i]];
    }
}

void fp8_e4m3_to_fp32_gather(const fp8_e4m3_t* src, uint32_t* dst, int n) {
    int i = 0;
    svbool_t pg = svptrue_b32();
    int vl = svcntw();

    for (; i + vl <= n; i += vl) {
        // Load FP8 values as bytes
        svbool_t pg8 = svwhilelt_b8(0, vl);
        svuint8_t fp8_bytes = svld1_u8(pg8, src + i);

        // Zero-extend to 32-bit
        svuint16_t indices16 = svunpklo_u16(fp8_bytes);
        svuint32_t indices = svunpklo_u32(indices16);
        svuint32_t byte_offsets = svlsl_n_u32_x(pg, indices, 2); // *4 for uint32

        // Gather from LUT
        svuint32_t result = svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, byte_offsets);
        svst1_u32(pg, dst + i, result);
    }

    for (; i < n; i++) {
        dst[i] = fp8_e4m3_to_fp32_lut[src[i]];
    }
}

void fp8_e5m2_to_fp16_gather(const fp8_e5m2_t* src, uint16_t* dst, int n) {
    int i = 0;
    svbool_t pg = svptrue_b32();
    int vl = svcntw();

    for (; i + vl <= n; i += vl) {
        svbool_t pg8 = svwhilelt_b8(0, vl);
        svuint8_t fp8_bytes = svld1_u8(pg8, src + i);
        svuint16_t indices16 = svunpklo_u16(fp8_bytes);
        svuint32_t indices = svunpklo_u32(indices16);
        svuint32_t byte_offsets = svlsl_n_u32_x(pg, indices, 2);
        svuint32_t result32 = svld1_gather_u32offset_u32(pg, fp8_e5m2_to_fp16_lut32, byte_offsets);
        svuint16_t result16 = svuzp1_u16(svreinterpret_u16(result32), svreinterpret_u16(result32));
        svst1_u16(svwhilelt_b16(0, vl), dst + i, result16);
    }

    for (; i < n; i++) {
        dst[i] = fp8_e5m2_to_fp16_lut[src[i]];
    }
}

void fp8_e5m2_to_fp32_gather(const fp8_e5m2_t* src, uint32_t* dst, int n) {
    int i = 0;
    svbool_t pg = svptrue_b32();
    int vl = svcntw();

    for (; i + vl <= n; i += vl) {
        svbool_t pg8 = svwhilelt_b8(0, vl);
        svuint8_t fp8_bytes = svld1_u8(pg8, src + i);
        svuint16_t indices16 = svunpklo_u16(fp8_bytes);
        svuint32_t indices = svunpklo_u32(indices16);
        svuint32_t byte_offsets = svlsl_n_u32_x(pg, indices, 2);
        svuint32_t result = svld1_gather_u32offset_u32(pg, fp8_e5m2_to_fp32_lut, byte_offsets);
        svst1_u32(pg, dst + i, result);
    }

    for (; i < n; i++) {
        dst[i] = fp8_e5m2_to_fp32_lut[src[i]];
    }
}

// ============================================================================
// Bit Arithmetic - ARM64 Base Instructions (uses EX* pipe)
// Uses inline scalar reference for correctness
// ============================================================================

void fp8_e4m3_to_fp16_base(const fp8_e4m3_t* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = fp8_e4m3_to_fp16_scalar(src[i]);
    }
}

void fp8_e4m3_to_fp32_base(const fp8_e4m3_t* src, uint32_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = fp8_e4m3_to_fp32_scalar(src[i]);
    }
}

void fp8_e5m2_to_fp16_base(const fp8_e5m2_t* src, uint16_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = fp8_e5m2_to_fp16_scalar(src[i]);
    }
}

void fp8_e5m2_to_fp32_base(const fp8_e5m2_t* src, uint32_t* dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = fp8_e5m2_to_fp32_scalar(src[i]);
    }
}

// ============================================================================
// Bit Arithmetic - SVE Instructions (uses FL* pipe)
// Vectorized bit manipulation - uses LUT for subnormals (exp=0, mant!=0)
// ============================================================================

void fp8_e4m3_to_fp16_sve(const fp8_e4m3_t* src, uint16_t* dst, int n) {
    int i = 0;
    int vl = svcntw(); // Use 32-bit vector length for easier byte handling

    svuint32_t bias_adj = svdup_u32(8);      // FP16_BIAS - FP8_E4M3_BIAS
    svuint32_t nan_exp = svdup_u32(0x1F << 10);

    for (; i + vl <= n; i += vl) {
        svbool_t pg = svptrue_b32();
        svbool_t pg8 = svwhilelt_b8(0, vl);

        // Load bytes and zero-extend to 32-bit for arithmetic
        svuint8_t bytes = svld1_u8(pg8, src + i);
        svuint16_t h = svunpklo_u16(bytes);
        svuint32_t x = svunpklo_u32(h);

        // Extract fields
        svuint32_t sign = svlsr_n_u32_x(pg, x, 7);
        sign = svlsl_n_u32_x(pg, sign, 15);

        svuint32_t exp = svlsr_n_u32_x(pg, x, 3);
        exp = svand_n_u32_x(pg, exp, 0xF);

        svuint32_t mant = svand_n_u32_x(pg, x, 0x7);

        // Check for special cases
        svbool_t is_zero_exp = svcmpeq_n_u32(pg, exp, 0);
        svbool_t is_max_exp = svcmpeq_n_u32(pg, exp, 15);
        svbool_t is_zero = svand_b_z(pg, is_zero_exp, svcmpeq_n_u32(pg, mant, 0));
        svbool_t is_subnormal = svand_b_z(pg, is_zero_exp, svcmpne_n_u32(pg, mant, 0));

        // Normal path: result = sign | ((exp + 8) << 10) | (mant << 7)
        svuint32_t new_exp = svadd_u32_x(pg, exp, bias_adj);
        new_exp = svlsl_n_u32_x(pg, new_exp, 10);
        svuint32_t new_mant = svlsl_n_u32_x(pg, mant, 7);
        svuint32_t result = svorr_u32_x(pg, sign, new_exp);
        result = svorr_u32_x(pg, result, new_mant);

        // Handle zero (exp=0, mant=0)
        result = svsel_u32(is_zero, sign, result);

        // Handle max exponent (NaN for E4M3)
        svuint32_t nan_result = svorr_u32_x(pg, sign, nan_exp);
        nan_result = svorr_u32_x(pg, nan_result, new_mant);
        result = svsel_u32(is_max_exp, nan_result, result);

        // Handle subnormals using LUT gather
        if (svptest_any(pg, is_subnormal)) {
            svuint32_t lut_offsets = svlsl_n_u32_x(pg, x, 2);
            svuint32_t lut_result = svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp16_lut32, lut_offsets);
            result = svsel_u32(is_subnormal, lut_result, result);
        }

        // Pack to 16-bit and store
        svuint16_t result16 = svuzp1_u16(svreinterpret_u16(result), svreinterpret_u16(result));
        svst1_u16(svwhilelt_b16(0, vl), dst + i, result16);
    }

    // Remainder using scalar
    for (; i < n; i++) {
        dst[i] = fp8_e4m3_to_fp16_scalar(src[i]);
    }
}

void fp8_e4m3_to_fp32_sve(const fp8_e4m3_t* src, uint32_t* dst, int n) {
    int i = 0;
    int vl = svcntw();

    svuint32_t bias_adj = svdup_u32(120);    // FP32_BIAS - FP8_E4M3_BIAS
    svuint32_t nan_exp = svdup_u32(0xFF << 23);

    for (; i + vl <= n; i += vl) {
        svbool_t pg = svptrue_b32();
        svbool_t pg8 = svwhilelt_b8(0, vl);

        // Load bytes and zero-extend to 32-bit
        svuint8_t bytes = svld1_u8(pg8, src + i);
        svuint16_t h = svunpklo_u16(bytes);
        svuint32_t x = svunpklo_u32(h);

        // Extract fields
        svuint32_t sign = svlsr_n_u32_x(pg, x, 7);
        sign = svlsl_n_u32_x(pg, sign, 31);

        svuint32_t exp = svlsr_n_u32_x(pg, x, 3);
        exp = svand_n_u32_x(pg, exp, 0xF);

        svuint32_t mant = svand_n_u32_x(pg, x, 0x7);

        // Check special cases
        svbool_t is_zero_exp = svcmpeq_n_u32(pg, exp, 0);
        svbool_t is_max_exp = svcmpeq_n_u32(pg, exp, 15);
        svbool_t is_zero = svand_b_z(pg, is_zero_exp, svcmpeq_n_u32(pg, mant, 0));
        svbool_t is_subnormal = svand_b_z(pg, is_zero_exp, svcmpne_n_u32(pg, mant, 0));

        // Normal path
        svuint32_t new_exp = svadd_u32_x(pg, exp, bias_adj);
        new_exp = svlsl_n_u32_x(pg, new_exp, 23);
        svuint32_t new_mant = svlsl_n_u32_x(pg, mant, 20);
        svuint32_t result = svorr_u32_x(pg, sign, new_exp);
        result = svorr_u32_x(pg, result, new_mant);

        // Zero
        result = svsel_u32(is_zero, sign, result);

        // NaN
        svuint32_t nan_result = svorr_u32_x(pg, sign, nan_exp);
        nan_result = svorr_u32_x(pg, nan_result, new_mant);
        result = svsel_u32(is_max_exp, nan_result, result);

        // Handle subnormals using LUT gather
        if (svptest_any(pg, is_subnormal)) {
            svuint32_t lut_offsets = svlsl_n_u32_x(pg, x, 2);
            svuint32_t lut_result = svld1_gather_u32offset_u32(pg, fp8_e4m3_to_fp32_lut, lut_offsets);
            result = svsel_u32(is_subnormal, lut_result, result);
        }

        svst1_u32(pg, dst + i, result);
    }

    for (; i < n; i++) {
        dst[i] = fp8_e4m3_to_fp32_scalar(src[i]);
    }
}

void fp8_e5m2_to_fp16_sve(const fp8_e5m2_t* src, uint16_t* dst, int n) {
    int i = 0;
    int vl = svcntw();

    // E5M2 to FP16: same bias, just shift mantissa by 8
    for (; i + vl <= n; i += vl) {
        svbool_t pg = svptrue_b32();
        svbool_t pg8 = svwhilelt_b8(0, vl);

        svuint8_t bytes = svld1_u8(pg8, src + i);
        svuint16_t h = svunpklo_u16(bytes);
        svuint32_t x = svunpklo_u32(h);

        // Extract fields
        svuint32_t sign = svlsr_n_u32_x(pg, x, 7);
        sign = svlsl_n_u32_x(pg, sign, 15);

        svuint32_t exp = svlsr_n_u32_x(pg, x, 2);
        exp = svand_n_u32_x(pg, exp, 0x1F);

        svuint32_t mant = svand_n_u32_x(pg, x, 0x3);

        // Check special cases
        svbool_t is_zero_exp = svcmpeq_n_u32(pg, exp, 0);
        svbool_t is_zero = svand_b_z(pg, is_zero_exp, svcmpeq_n_u32(pg, mant, 0));
        svbool_t is_subnormal = svand_b_z(pg, is_zero_exp, svcmpne_n_u32(pg, mant, 0));

        // Normal/Inf/NaN path (same structure due to same exponent bits)
        svuint32_t new_exp = svlsl_n_u32_x(pg, exp, 10);
        svuint32_t new_mant = svlsl_n_u32_x(pg, mant, 8);
        svuint32_t result = svorr_u32_x(pg, sign, new_exp);
        result = svorr_u32_x(pg, result, new_mant);

        // Zero
        result = svsel_u32(is_zero, sign, result);

        // Handle subnormals using LUT gather
        if (svptest_any(pg, is_subnormal)) {
            svuint32_t lut_offsets = svlsl_n_u32_x(pg, x, 2);
            svuint32_t lut_result = svld1_gather_u32offset_u32(pg, fp8_e5m2_to_fp16_lut32, lut_offsets);
            result = svsel_u32(is_subnormal, lut_result, result);
        }

        // Pack to 16-bit and store
        svuint16_t result16 = svuzp1_u16(svreinterpret_u16(result), svreinterpret_u16(result));
        svst1_u16(svwhilelt_b16(0, vl), dst + i, result16);
    }

    for (; i < n; i++) {
        dst[i] = fp8_e5m2_to_fp16_scalar(src[i]);
    }
}

void fp8_e5m2_to_fp32_sve(const fp8_e5m2_t* src, uint32_t* dst, int n) {
    int i = 0;
    int vl = svcntw();

    svuint32_t bias_adj = svdup_u32(112);    // FP32_BIAS - FP8_E5M2_BIAS
    svuint32_t nan_exp = svdup_u32(0xFF << 23);

    for (; i + vl <= n; i += vl) {
        svbool_t pg = svptrue_b32();
        svbool_t pg8 = svwhilelt_b8(0, vl);

        svuint8_t bytes = svld1_u8(pg8, src + i);
        svuint16_t h = svunpklo_u16(bytes);
        svuint32_t x = svunpklo_u32(h);

        // Extract fields
        svuint32_t sign = svlsr_n_u32_x(pg, x, 7);
        sign = svlsl_n_u32_x(pg, sign, 31);

        svuint32_t exp = svlsr_n_u32_x(pg, x, 2);
        exp = svand_n_u32_x(pg, exp, 0x1F);

        svuint32_t mant = svand_n_u32_x(pg, x, 0x3);

        // Check special cases
        svbool_t is_zero_exp = svcmpeq_n_u32(pg, exp, 0);
        svbool_t is_max_exp = svcmpeq_n_u32(pg, exp, 31);
        svbool_t is_zero = svand_b_z(pg, is_zero_exp, svcmpeq_n_u32(pg, mant, 0));
        svbool_t is_subnormal = svand_b_z(pg, is_zero_exp, svcmpne_n_u32(pg, mant, 0));

        // Normal path
        svuint32_t new_exp = svadd_u32_x(pg, exp, bias_adj);
        new_exp = svlsl_n_u32_x(pg, new_exp, 23);
        svuint32_t new_mant = svlsl_n_u32_x(pg, mant, 21);
        svuint32_t result = svorr_u32_x(pg, sign, new_exp);
        result = svorr_u32_x(pg, result, new_mant);

        // Zero
        result = svsel_u32(is_zero, sign, result);

        // Inf/NaN
        svuint32_t nan_result = svorr_u32_x(pg, sign, nan_exp);
        nan_result = svorr_u32_x(pg, nan_result, new_mant);
        result = svsel_u32(is_max_exp, nan_result, result);

        // Handle subnormals using LUT gather
        if (svptest_any(pg, is_subnormal)) {
            svuint32_t lut_offsets = svlsl_n_u32_x(pg, x, 2);
            svuint32_t lut_result = svld1_gather_u32offset_u32(pg, fp8_e5m2_to_fp32_lut, lut_offsets);
            result = svsel_u32(is_subnormal, lut_result, result);
        }

        svst1_u32(pg, dst + i, result);
    }

    for (; i < n; i++) {
        dst[i] = fp8_e5m2_to_fp32_scalar(src[i]);
    }
}
