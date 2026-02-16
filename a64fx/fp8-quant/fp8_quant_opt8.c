#include "fp8_quant_opt8.h"
#include <arm_sve.h>

// ============================================================================
// FP16 -> E5M2: x8 Unroll for maximum latency hiding
// ============================================================================
// A64FX has 11-cycle load latency, 2 LD units, 4 ALU pipes
// With x8 unroll: issue 8 loads (4 cycles), then 11 cycles of compute
// This keeps ALU busy while waiting for load results

void fp16_to_fp8_e5m2_sve_unroll8(const uint16_t* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    svbool_t pg16 = svptrue_b16();
    size_t vl16 = svcnth();   // 32 elements
    size_t vl8 = vl16 * 8;    // 256 elements per iteration

    // Constants hoisted out of loop
    svuint16_t mask_sign = svdup_u16(0x8000);
    svuint16_t mask_exp  = svdup_u16(0x7C00);
    svuint16_t mask_mant = svdup_u16(0x03FF);
    svuint16_t zero = svdup_u16(0);
    svuint16_t three = svdup_u16(3);
    svuint16_t four = svdup_u16(4);
    svuint16_t exp_max = svdup_u16(124);

    // Main loop: process 256 elements (8 vectors) per iteration
    for (; i + vl8 <= n; i += vl8) {
        // ===== PHASE 1: Issue all 8 loads early =====
        svuint16_t x0 = svld1_u16(pg16, src + i);
        svuint16_t x1 = svld1_u16(pg16, src + i + vl16);
        svuint16_t x2 = svld1_u16(pg16, src + i + vl16*2);
        svuint16_t x3 = svld1_u16(pg16, src + i + vl16*3);
        svuint16_t x4 = svld1_u16(pg16, src + i + vl16*4);
        svuint16_t x5 = svld1_u16(pg16, src + i + vl16*5);
        svuint16_t x6 = svld1_u16(pg16, src + i + vl16*6);
        svuint16_t x7 = svld1_u16(pg16, src + i + vl16*7);

        // ===== PHASE 2: Extract sign (4 cycles bitop) =====
        svuint16_t sign0 = svand_u16_x(pg16, x0, mask_sign);
        svuint16_t sign1 = svand_u16_x(pg16, x1, mask_sign);
        svuint16_t sign2 = svand_u16_x(pg16, x2, mask_sign);
        svuint16_t sign3 = svand_u16_x(pg16, x3, mask_sign);
        svuint16_t sign4 = svand_u16_x(pg16, x4, mask_sign);
        svuint16_t sign5 = svand_u16_x(pg16, x5, mask_sign);
        svuint16_t sign6 = svand_u16_x(pg16, x6, mask_sign);
        svuint16_t sign7 = svand_u16_x(pg16, x7, mask_sign);

        // ===== PHASE 3: Extract exponent =====
        svuint16_t exp0 = svand_u16_x(pg16, x0, mask_exp);
        svuint16_t exp1 = svand_u16_x(pg16, x1, mask_exp);
        svuint16_t exp2 = svand_u16_x(pg16, x2, mask_exp);
        svuint16_t exp3 = svand_u16_x(pg16, x3, mask_exp);
        svuint16_t exp4 = svand_u16_x(pg16, x4, mask_exp);
        svuint16_t exp5 = svand_u16_x(pg16, x5, mask_exp);
        svuint16_t exp6 = svand_u16_x(pg16, x6, mask_exp);
        svuint16_t exp7 = svand_u16_x(pg16, x7, mask_exp);

        // ===== PHASE 4: Extract mantissa =====
        svuint16_t mant0 = svand_u16_x(pg16, x0, mask_mant);
        svuint16_t mant1 = svand_u16_x(pg16, x1, mask_mant);
        svuint16_t mant2 = svand_u16_x(pg16, x2, mask_mant);
        svuint16_t mant3 = svand_u16_x(pg16, x3, mask_mant);
        svuint16_t mant4 = svand_u16_x(pg16, x4, mask_mant);
        svuint16_t mant5 = svand_u16_x(pg16, x5, mask_mant);
        svuint16_t mant6 = svand_u16_x(pg16, x6, mask_mant);
        svuint16_t mant7 = svand_u16_x(pg16, x7, mask_mant);

        // ===== PHASE 5: Shift sign to bit 7 =====
        sign0 = svlsr_n_u16_x(pg16, sign0, 8);
        sign1 = svlsr_n_u16_x(pg16, sign1, 8);
        sign2 = svlsr_n_u16_x(pg16, sign2, 8);
        sign3 = svlsr_n_u16_x(pg16, sign3, 8);
        sign4 = svlsr_n_u16_x(pg16, sign4, 8);
        sign5 = svlsr_n_u16_x(pg16, sign5, 8);
        sign6 = svlsr_n_u16_x(pg16, sign6, 8);
        sign7 = svlsr_n_u16_x(pg16, sign7, 8);

        // ===== PHASE 6: Shift exponent =====
        exp0 = svlsr_n_u16_x(pg16, exp0, 8);
        exp1 = svlsr_n_u16_x(pg16, exp1, 8);
        exp2 = svlsr_n_u16_x(pg16, exp2, 8);
        exp3 = svlsr_n_u16_x(pg16, exp3, 8);
        exp4 = svlsr_n_u16_x(pg16, exp4, 8);
        exp5 = svlsr_n_u16_x(pg16, exp5, 8);
        exp6 = svlsr_n_u16_x(pg16, exp6, 8);
        exp7 = svlsr_n_u16_x(pg16, exp7, 8);

        // ===== PHASE 7: Round mantissa =====
        svuint16_t round0 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant0, 7), 1);
        svuint16_t round1 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant1, 7), 1);
        svuint16_t round2 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant2, 7), 1);
        svuint16_t round3 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant3, 7), 1);
        svuint16_t round4 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant4, 7), 1);
        svuint16_t round5 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant5, 7), 1);
        svuint16_t round6 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant6, 7), 1);
        svuint16_t round7 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant7, 7), 1);

        mant0 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant0, 8), round0);
        mant1 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant1, 8), round1);
        mant2 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant2, 8), round2);
        mant3 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant3, 8), round3);
        mant4 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant4, 8), round4);
        mant5 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant5, 8), round5);
        mant6 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant6, 8), round6);
        mant7 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant7, 8), round7);

        // ===== PHASE 8: Handle mantissa overflow =====
        svbool_t ovf0 = svcmpgt_u16(pg16, mant0, three);
        svbool_t ovf1 = svcmpgt_u16(pg16, mant1, three);
        svbool_t ovf2 = svcmpgt_u16(pg16, mant2, three);
        svbool_t ovf3 = svcmpgt_u16(pg16, mant3, three);
        svbool_t ovf4 = svcmpgt_u16(pg16, mant4, three);
        svbool_t ovf5 = svcmpgt_u16(pg16, mant5, three);
        svbool_t ovf6 = svcmpgt_u16(pg16, mant6, three);
        svbool_t ovf7 = svcmpgt_u16(pg16, mant7, three);

        mant0 = svsel_u16(ovf0, zero, mant0);
        mant1 = svsel_u16(ovf1, zero, mant1);
        mant2 = svsel_u16(ovf2, zero, mant2);
        mant3 = svsel_u16(ovf3, zero, mant3);
        mant4 = svsel_u16(ovf4, zero, mant4);
        mant5 = svsel_u16(ovf5, zero, mant5);
        mant6 = svsel_u16(ovf6, zero, mant6);
        mant7 = svsel_u16(ovf7, zero, mant7);

        exp0 = svadd_u16_x(pg16, exp0, svsel_u16(ovf0, four, zero));
        exp1 = svadd_u16_x(pg16, exp1, svsel_u16(ovf1, four, zero));
        exp2 = svadd_u16_x(pg16, exp2, svsel_u16(ovf2, four, zero));
        exp3 = svadd_u16_x(pg16, exp3, svsel_u16(ovf3, four, zero));
        exp4 = svadd_u16_x(pg16, exp4, svsel_u16(ovf4, four, zero));
        exp5 = svadd_u16_x(pg16, exp5, svsel_u16(ovf5, four, zero));
        exp6 = svadd_u16_x(pg16, exp6, svsel_u16(ovf6, four, zero));
        exp7 = svadd_u16_x(pg16, exp7, svsel_u16(ovf7, four, zero));

        // ===== PHASE 9: Clamp to Inf =====
        svbool_t inf0 = svcmpgt_u16(pg16, exp0, exp_max);
        svbool_t inf1 = svcmpgt_u16(pg16, exp1, exp_max);
        svbool_t inf2 = svcmpgt_u16(pg16, exp2, exp_max);
        svbool_t inf3 = svcmpgt_u16(pg16, exp3, exp_max);
        svbool_t inf4 = svcmpgt_u16(pg16, exp4, exp_max);
        svbool_t inf5 = svcmpgt_u16(pg16, exp5, exp_max);
        svbool_t inf6 = svcmpgt_u16(pg16, exp6, exp_max);
        svbool_t inf7 = svcmpgt_u16(pg16, exp7, exp_max);

        exp0 = svsel_u16(inf0, exp_max, exp0);
        exp1 = svsel_u16(inf1, exp_max, exp1);
        exp2 = svsel_u16(inf2, exp_max, exp2);
        exp3 = svsel_u16(inf3, exp_max, exp3);
        exp4 = svsel_u16(inf4, exp_max, exp4);
        exp5 = svsel_u16(inf5, exp_max, exp5);
        exp6 = svsel_u16(inf6, exp_max, exp6);
        exp7 = svsel_u16(inf7, exp_max, exp7);

        mant0 = svsel_u16(inf0, zero, mant0);
        mant1 = svsel_u16(inf1, zero, mant1);
        mant2 = svsel_u16(inf2, zero, mant2);
        mant3 = svsel_u16(inf3, zero, mant3);
        mant4 = svsel_u16(inf4, zero, mant4);
        mant5 = svsel_u16(inf5, zero, mant5);
        mant6 = svsel_u16(inf6, zero, mant6);
        mant7 = svsel_u16(inf7, zero, mant7);

        // ===== PHASE 10: Combine =====
        svuint16_t r0 = svorr_u16_x(pg16, sign0, svorr_u16_x(pg16, exp0, mant0));
        svuint16_t r1 = svorr_u16_x(pg16, sign1, svorr_u16_x(pg16, exp1, mant1));
        svuint16_t r2 = svorr_u16_x(pg16, sign2, svorr_u16_x(pg16, exp2, mant2));
        svuint16_t r3 = svorr_u16_x(pg16, sign3, svorr_u16_x(pg16, exp3, mant3));
        svuint16_t r4 = svorr_u16_x(pg16, sign4, svorr_u16_x(pg16, exp4, mant4));
        svuint16_t r5 = svorr_u16_x(pg16, sign5, svorr_u16_x(pg16, exp5, mant5));
        svuint16_t r6 = svorr_u16_x(pg16, sign6, svorr_u16_x(pg16, exp6, mant6));
        svuint16_t r7 = svorr_u16_x(pg16, sign7, svorr_u16_x(pg16, exp7, mant7));

        // ===== PHASE 11: Pack and store =====
        svuint8_t out0 = svuzp1_u8(svreinterpret_u8_u16(r0), svreinterpret_u8_u16(r0));
        svuint8_t out1 = svuzp1_u8(svreinterpret_u8_u16(r1), svreinterpret_u8_u16(r1));
        svuint8_t out2 = svuzp1_u8(svreinterpret_u8_u16(r2), svreinterpret_u8_u16(r2));
        svuint8_t out3 = svuzp1_u8(svreinterpret_u8_u16(r3), svreinterpret_u8_u16(r3));
        svuint8_t out4 = svuzp1_u8(svreinterpret_u8_u16(r4), svreinterpret_u8_u16(r4));
        svuint8_t out5 = svuzp1_u8(svreinterpret_u8_u16(r5), svreinterpret_u8_u16(r5));
        svuint8_t out6 = svuzp1_u8(svreinterpret_u8_u16(r6), svreinterpret_u8_u16(r6));
        svuint8_t out7 = svuzp1_u8(svreinterpret_u8_u16(r7), svreinterpret_u8_u16(r7));

        svbool_t pg8 = svwhilelt_b8((uint64_t)0, vl16);
        svst1_u8(pg8, dst + i, out0);
        svst1_u8(pg8, dst + i + vl16, out1);
        svst1_u8(pg8, dst + i + vl16*2, out2);
        svst1_u8(pg8, dst + i + vl16*3, out3);
        svst1_u8(pg8, dst + i + vl16*4, out4);
        svst1_u8(pg8, dst + i + vl16*5, out5);
        svst1_u8(pg8, dst + i + vl16*6, out6);
        svst1_u8(pg8, dst + i + vl16*7, out7);
    }

    // Cleanup with single-vector loop
    for (; i + vl16 <= n; i += vl16) {
        svuint16_t x = svld1_u16(pg16, src + i);
        svuint16_t sign = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x, mask_sign), 8);
        svuint16_t exp = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x, mask_exp), 8);
        svuint16_t mant = svand_u16_x(pg16, x, mask_mant);

        svuint16_t round = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant, 7), 1);
        mant = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant, 8), round);

        svbool_t ovf = svcmpgt_u16(pg16, mant, three);
        mant = svsel_u16(ovf, zero, mant);
        exp = svadd_u16_x(pg16, exp, svsel_u16(ovf, four, zero));

        svbool_t inf = svcmpgt_u16(pg16, exp, exp_max);
        exp = svsel_u16(inf, exp_max, exp);
        mant = svsel_u16(inf, zero, mant);

        svuint16_t r = svorr_u16_x(pg16, sign, svorr_u16_x(pg16, exp, mant));
        svuint8_t out = svuzp1_u8(svreinterpret_u8_u16(r), svreinterpret_u8_u16(r));

        svbool_t pg8 = svwhilelt_b8((uint64_t)0, vl16);
        svst1_u8(pg8, dst + i, out);
    }

    // Scalar tail
    for (; i < n; i++) {
        uint16_t x = src[i];
        uint16_t sign = (x >> 8) & 0x80;
        uint16_t exp = (x >> 8) & 0x7C;
        uint16_t mant = x & 0x3FF;
        uint16_t round_bit = (mant >> 7) & 1;
        mant = (mant >> 8) + round_bit;
        if (mant > 3) { mant = 0; exp += 4; }
        if (exp > 124) { exp = 124; mant = 0; }
        dst[i] = sign | exp | mant;
    }
}

// ============================================================================
// FP32 -> E5M2: x8 Unroll
// ============================================================================

void fp32_to_fp8_e5m2_sve_unroll8(const float* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    svbool_t pg32 = svptrue_b32();
    size_t vl32 = svcntw();   // 16 elements
    size_t vl8 = vl32 * 8;    // 128 elements per iteration

    svfloat32_t max_val = svdup_f32(57344.0f);
    svfloat32_t min_val = svdup_f32(6.1e-5f);
    svuint32_t zero32 = svdup_u32(0);
    svuint32_t one32 = svdup_u32(1);
    svuint32_t three32 = svdup_u32(3);

    for (; i + vl8 <= n; i += vl8) {
        // ===== Load 8 vectors =====
        svfloat32_t f0 = svld1_f32(pg32, src + i);
        svfloat32_t f1 = svld1_f32(pg32, src + i + vl32);
        svfloat32_t f2 = svld1_f32(pg32, src + i + vl32*2);
        svfloat32_t f3 = svld1_f32(pg32, src + i + vl32*3);
        svfloat32_t f4 = svld1_f32(pg32, src + i + vl32*4);
        svfloat32_t f5 = svld1_f32(pg32, src + i + vl32*5);
        svfloat32_t f6 = svld1_f32(pg32, src + i + vl32*6);
        svfloat32_t f7 = svld1_f32(pg32, src + i + vl32*7);

        // ===== Extract sign =====
        svuint32_t b0 = svreinterpret_u32_f32(f0);
        svuint32_t b1 = svreinterpret_u32_f32(f1);
        svuint32_t b2 = svreinterpret_u32_f32(f2);
        svuint32_t b3 = svreinterpret_u32_f32(f3);
        svuint32_t b4 = svreinterpret_u32_f32(f4);
        svuint32_t b5 = svreinterpret_u32_f32(f5);
        svuint32_t b6 = svreinterpret_u32_f32(f6);
        svuint32_t b7 = svreinterpret_u32_f32(f7);

        svuint32_t sign0 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b0, 0x80000000), 24);
        svuint32_t sign1 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b1, 0x80000000), 24);
        svuint32_t sign2 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b2, 0x80000000), 24);
        svuint32_t sign3 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b3, 0x80000000), 24);
        svuint32_t sign4 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b4, 0x80000000), 24);
        svuint32_t sign5 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b5, 0x80000000), 24);
        svuint32_t sign6 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b6, 0x80000000), 24);
        svuint32_t sign7 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b7, 0x80000000), 24);

        // ===== Clamp and check underflow =====
        svfloat32_t a0 = svmin_f32_x(pg32, svabs_f32_x(pg32, f0), max_val);
        svfloat32_t a1 = svmin_f32_x(pg32, svabs_f32_x(pg32, f1), max_val);
        svfloat32_t a2 = svmin_f32_x(pg32, svabs_f32_x(pg32, f2), max_val);
        svfloat32_t a3 = svmin_f32_x(pg32, svabs_f32_x(pg32, f3), max_val);
        svfloat32_t a4 = svmin_f32_x(pg32, svabs_f32_x(pg32, f4), max_val);
        svfloat32_t a5 = svmin_f32_x(pg32, svabs_f32_x(pg32, f5), max_val);
        svfloat32_t a6 = svmin_f32_x(pg32, svabs_f32_x(pg32, f6), max_val);
        svfloat32_t a7 = svmin_f32_x(pg32, svabs_f32_x(pg32, f7), max_val);

        svbool_t uf0 = svcmplt_f32(pg32, a0, min_val);
        svbool_t uf1 = svcmplt_f32(pg32, a1, min_val);
        svbool_t uf2 = svcmplt_f32(pg32, a2, min_val);
        svbool_t uf3 = svcmplt_f32(pg32, a3, min_val);
        svbool_t uf4 = svcmplt_f32(pg32, a4, min_val);
        svbool_t uf5 = svcmplt_f32(pg32, a5, min_val);
        svbool_t uf6 = svcmplt_f32(pg32, a6, min_val);
        svbool_t uf7 = svcmplt_f32(pg32, a7, min_val);

        svuint32_t ab0 = svreinterpret_u32_f32(a0);
        svuint32_t ab1 = svreinterpret_u32_f32(a1);
        svuint32_t ab2 = svreinterpret_u32_f32(a2);
        svuint32_t ab3 = svreinterpret_u32_f32(a3);
        svuint32_t ab4 = svreinterpret_u32_f32(a4);
        svuint32_t ab5 = svreinterpret_u32_f32(a5);
        svuint32_t ab6 = svreinterpret_u32_f32(a6);
        svuint32_t ab7 = svreinterpret_u32_f32(a7);

        // ===== Extract and convert exponent =====
        svuint32_t exp0 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab0, 23), 112);
        svuint32_t exp1 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab1, 23), 112);
        svuint32_t exp2 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab2, 23), 112);
        svuint32_t exp3 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab3, 23), 112);
        svuint32_t exp4 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab4, 23), 112);
        svuint32_t exp5 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab5, 23), 112);
        svuint32_t exp6 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab6, 23), 112);
        svuint32_t exp7 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab7, 23), 112);

        // ===== Round mantissa =====
        svuint32_t mant0 = svand_n_u32_x(pg32, ab0, 0x7FFFFF);
        svuint32_t mant1 = svand_n_u32_x(pg32, ab1, 0x7FFFFF);
        svuint32_t mant2 = svand_n_u32_x(pg32, ab2, 0x7FFFFF);
        svuint32_t mant3 = svand_n_u32_x(pg32, ab3, 0x7FFFFF);
        svuint32_t mant4 = svand_n_u32_x(pg32, ab4, 0x7FFFFF);
        svuint32_t mant5 = svand_n_u32_x(pg32, ab5, 0x7FFFFF);
        svuint32_t mant6 = svand_n_u32_x(pg32, ab6, 0x7FFFFF);
        svuint32_t mant7 = svand_n_u32_x(pg32, ab7, 0x7FFFFF);

        svuint32_t round0 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant0, 20), 1);
        svuint32_t round1 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant1, 20), 1);
        svuint32_t round2 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant2, 20), 1);
        svuint32_t round3 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant3, 20), 1);
        svuint32_t round4 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant4, 20), 1);
        svuint32_t round5 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant5, 20), 1);
        svuint32_t round6 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant6, 20), 1);
        svuint32_t round7 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant7, 20), 1);

        mant0 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant0, 21), round0);
        mant1 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant1, 21), round1);
        mant2 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant2, 21), round2);
        mant3 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant3, 21), round3);
        mant4 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant4, 21), round4);
        mant5 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant5, 21), round5);
        mant6 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant6, 21), round6);
        mant7 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant7, 21), round7);

        // ===== Handle overflow =====
        svbool_t ovf0 = svcmpgt_u32(pg32, mant0, three32);
        svbool_t ovf1 = svcmpgt_u32(pg32, mant1, three32);
        svbool_t ovf2 = svcmpgt_u32(pg32, mant2, three32);
        svbool_t ovf3 = svcmpgt_u32(pg32, mant3, three32);
        svbool_t ovf4 = svcmpgt_u32(pg32, mant4, three32);
        svbool_t ovf5 = svcmpgt_u32(pg32, mant5, three32);
        svbool_t ovf6 = svcmpgt_u32(pg32, mant6, three32);
        svbool_t ovf7 = svcmpgt_u32(pg32, mant7, three32);

        mant0 = svsel_u32(ovf0, zero32, mant0);
        mant1 = svsel_u32(ovf1, zero32, mant1);
        mant2 = svsel_u32(ovf2, zero32, mant2);
        mant3 = svsel_u32(ovf3, zero32, mant3);
        mant4 = svsel_u32(ovf4, zero32, mant4);
        mant5 = svsel_u32(ovf5, zero32, mant5);
        mant6 = svsel_u32(ovf6, zero32, mant6);
        mant7 = svsel_u32(ovf7, zero32, mant7);

        exp0 = svadd_u32_x(pg32, exp0, svsel_u32(ovf0, one32, zero32));
        exp1 = svadd_u32_x(pg32, exp1, svsel_u32(ovf1, one32, zero32));
        exp2 = svadd_u32_x(pg32, exp2, svsel_u32(ovf2, one32, zero32));
        exp3 = svadd_u32_x(pg32, exp3, svsel_u32(ovf3, one32, zero32));
        exp4 = svadd_u32_x(pg32, exp4, svsel_u32(ovf4, one32, zero32));
        exp5 = svadd_u32_x(pg32, exp5, svsel_u32(ovf5, one32, zero32));
        exp6 = svadd_u32_x(pg32, exp6, svsel_u32(ovf6, one32, zero32));
        exp7 = svadd_u32_x(pg32, exp7, svsel_u32(ovf7, one32, zero32));

        // Clamp exp
        exp0 = svmin_n_u32_x(pg32, exp0, 30);
        exp1 = svmin_n_u32_x(pg32, exp1, 30);
        exp2 = svmin_n_u32_x(pg32, exp2, 30);
        exp3 = svmin_n_u32_x(pg32, exp3, 30);
        exp4 = svmin_n_u32_x(pg32, exp4, 30);
        exp5 = svmin_n_u32_x(pg32, exp5, 30);
        exp6 = svmin_n_u32_x(pg32, exp6, 30);
        exp7 = svmin_n_u32_x(pg32, exp7, 30);

        // ===== Combine =====
        svuint32_t r0 = svorr_u32_x(pg32, sign0, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp0, 2), mant0));
        svuint32_t r1 = svorr_u32_x(pg32, sign1, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp1, 2), mant1));
        svuint32_t r2 = svorr_u32_x(pg32, sign2, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp2, 2), mant2));
        svuint32_t r3 = svorr_u32_x(pg32, sign3, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp3, 2), mant3));
        svuint32_t r4 = svorr_u32_x(pg32, sign4, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp4, 2), mant4));
        svuint32_t r5 = svorr_u32_x(pg32, sign5, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp5, 2), mant5));
        svuint32_t r6 = svorr_u32_x(pg32, sign6, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp6, 2), mant6));
        svuint32_t r7 = svorr_u32_x(pg32, sign7, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp7, 2), mant7));

        r0 = svsel_u32(uf0, sign0, r0);
        r1 = svsel_u32(uf1, sign1, r1);
        r2 = svsel_u32(uf2, sign2, r2);
        r3 = svsel_u32(uf3, sign3, r3);
        r4 = svsel_u32(uf4, sign4, r4);
        r5 = svsel_u32(uf5, sign5, r5);
        r6 = svsel_u32(uf6, sign6, r6);
        r7 = svsel_u32(uf7, sign7, r7);

        // ===== Pack 32->8 and store =====
        svuint16_t h0 = svuzp1_u16(svreinterpret_u16_u32(r0), svreinterpret_u16_u32(r0));
        svuint16_t h1 = svuzp1_u16(svreinterpret_u16_u32(r1), svreinterpret_u16_u32(r1));
        svuint16_t h2 = svuzp1_u16(svreinterpret_u16_u32(r2), svreinterpret_u16_u32(r2));
        svuint16_t h3 = svuzp1_u16(svreinterpret_u16_u32(r3), svreinterpret_u16_u32(r3));
        svuint16_t h4 = svuzp1_u16(svreinterpret_u16_u32(r4), svreinterpret_u16_u32(r4));
        svuint16_t h5 = svuzp1_u16(svreinterpret_u16_u32(r5), svreinterpret_u16_u32(r5));
        svuint16_t h6 = svuzp1_u16(svreinterpret_u16_u32(r6), svreinterpret_u16_u32(r6));
        svuint16_t h7 = svuzp1_u16(svreinterpret_u16_u32(r7), svreinterpret_u16_u32(r7));

        svuint8_t out0 = svuzp1_u8(svreinterpret_u8_u16(h0), svreinterpret_u8_u16(h0));
        svuint8_t out1 = svuzp1_u8(svreinterpret_u8_u16(h1), svreinterpret_u8_u16(h1));
        svuint8_t out2 = svuzp1_u8(svreinterpret_u8_u16(h2), svreinterpret_u8_u16(h2));
        svuint8_t out3 = svuzp1_u8(svreinterpret_u8_u16(h3), svreinterpret_u8_u16(h3));
        svuint8_t out4 = svuzp1_u8(svreinterpret_u8_u16(h4), svreinterpret_u8_u16(h4));
        svuint8_t out5 = svuzp1_u8(svreinterpret_u8_u16(h5), svreinterpret_u8_u16(h5));
        svuint8_t out6 = svuzp1_u8(svreinterpret_u8_u16(h6), svreinterpret_u8_u16(h6));
        svuint8_t out7 = svuzp1_u8(svreinterpret_u8_u16(h7), svreinterpret_u8_u16(h7));

        svbool_t pg8 = svwhilelt_b8((uint64_t)0, vl32);
        svst1_u8(pg8, dst + i, out0);
        svst1_u8(pg8, dst + i + vl32, out1);
        svst1_u8(pg8, dst + i + vl32*2, out2);
        svst1_u8(pg8, dst + i + vl32*3, out3);
        svst1_u8(pg8, dst + i + vl32*4, out4);
        svst1_u8(pg8, dst + i + vl32*5, out5);
        svst1_u8(pg8, dst + i + vl32*6, out6);
        svst1_u8(pg8, dst + i + vl32*7, out7);
    }

    // Single-vector cleanup
    for (; i + vl32 <= n; i += vl32) {
        svfloat32_t f = svld1_f32(pg32, src + i);
        svuint32_t b = svreinterpret_u32_f32(f);
        svuint32_t sign = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b, 0x80000000), 24);

        svfloat32_t a = svmin_f32_x(pg32, svabs_f32_x(pg32, f), max_val);
        svbool_t uf = svcmplt_f32(pg32, a, min_val);
        svuint32_t ab = svreinterpret_u32_f32(a);

        svuint32_t exp = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab, 23), 112);
        svuint32_t mant = svand_n_u32_x(pg32, ab, 0x7FFFFF);
        svuint32_t round = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant, 20), 1);
        mant = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant, 21), round);

        svbool_t ovf = svcmpgt_u32(pg32, mant, three32);
        mant = svsel_u32(ovf, zero32, mant);
        exp = svadd_u32_x(pg32, exp, svsel_u32(ovf, one32, zero32));
        exp = svmin_n_u32_x(pg32, exp, 30);

        svuint32_t r = svorr_u32_x(pg32, sign, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp, 2), mant));
        r = svsel_u32(uf, sign, r);

        svuint16_t h = svuzp1_u16(svreinterpret_u16_u32(r), svreinterpret_u16_u32(r));
        svuint8_t out = svuzp1_u8(svreinterpret_u8_u16(h), svreinterpret_u8_u16(h));

        svbool_t pg8 = svwhilelt_b8((uint64_t)0, vl32);
        svst1_u8(pg8, dst + i, out);
    }

    // Scalar tail
    for (; i < n; i++) {
        union { float f; uint32_t u; } c = {src[i]};
        uint32_t sign = (c.u >> 24) & 0x80;
        float a = c.f < 0 ? -c.f : c.f;
        if (a > 57344.0f) a = 57344.0f;
        if (a < 6.1e-5f) { dst[i] = sign; continue; }

        union { float f; uint32_t u; } ac = {a};
        uint32_t exp = ((ac.u >> 23) & 0xFF) - 112;
        uint32_t mant = ac.u & 0x7FFFFF;
        uint32_t round = (mant >> 20) & 1;
        mant = (mant >> 21) + round;
        if (mant > 3) { mant = 0; exp++; }
        if (exp > 30) exp = 30;
        dst[i] = sign | (exp << 2) | mant;
    }
}
