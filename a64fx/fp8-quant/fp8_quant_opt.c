#include "fp8_quant_opt.h"
#include <arm_sve.h>

// ============================================================================
// FP16 -> E5M2: Unroll x4 to hide 11-cycle load latency
// ============================================================================
// Strategy: Issue 4 loads early, then process while loads complete
// A64FX: 2 LD units, 4 ALU pipes - can sustain 2 loads + 4 ops per cycle

void fp16_to_fp8_e5m2_sve_unroll4(const uint16_t* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    svbool_t pg16 = svptrue_b16();
    size_t vl16 = svcnth();  // 32 elements
    size_t vl4 = vl16 * 4;   // 128 elements per iteration

    // Constants hoisted out of loop
    svuint16_t mask_sign = svdup_u16(0x8000);
    svuint16_t mask_exp  = svdup_u16(0x7C00);
    svuint16_t mask_mant = svdup_u16(0x03FF);
    svuint16_t zero = svdup_u16(0);
    svuint16_t three = svdup_u16(3);
    svuint16_t exp31 = svdup_u16(31);

    // Main loop: process 128 elements (4 vectors) per iteration
    for (; i + vl4 <= n; i += vl4) {
        // ===== PHASE 1: Issue all loads early =====
        // Loads have 11 cycle latency - issue them first
        svuint16_t x0 = svld1_u16(pg16, src + i);
        svuint16_t x1 = svld1_u16(pg16, src + i + vl16);
        svuint16_t x2 = svld1_u16(pg16, src + i + vl16*2);
        svuint16_t x3 = svld1_u16(pg16, src + i + vl16*3);

        // ===== PHASE 2: Extract sign (4 cycles each, can overlap) =====
        svuint16_t sign0 = svand_u16_x(pg16, x0, mask_sign);
        svuint16_t sign1 = svand_u16_x(pg16, x1, mask_sign);
        svuint16_t sign2 = svand_u16_x(pg16, x2, mask_sign);
        svuint16_t sign3 = svand_u16_x(pg16, x3, mask_sign);

        // ===== PHASE 3: Extract exponent =====
        svuint16_t exp0 = svand_u16_x(pg16, x0, mask_exp);
        svuint16_t exp1 = svand_u16_x(pg16, x1, mask_exp);
        svuint16_t exp2 = svand_u16_x(pg16, x2, mask_exp);
        svuint16_t exp3 = svand_u16_x(pg16, x3, mask_exp);

        // ===== PHASE 4: Extract mantissa =====
        svuint16_t mant0 = svand_u16_x(pg16, x0, mask_mant);
        svuint16_t mant1 = svand_u16_x(pg16, x1, mask_mant);
        svuint16_t mant2 = svand_u16_x(pg16, x2, mask_mant);
        svuint16_t mant3 = svand_u16_x(pg16, x3, mask_mant);

        // ===== PHASE 5: Shift sign to bit 7 position =====
        sign0 = svlsr_n_u16_x(pg16, sign0, 8);
        sign1 = svlsr_n_u16_x(pg16, sign1, 8);
        sign2 = svlsr_n_u16_x(pg16, sign2, 8);
        sign3 = svlsr_n_u16_x(pg16, sign3, 8);

        // ===== PHASE 6: Shift exponent to position =====
        // E5M2 exp same bias as FP16, just shift from bits 10-14 to bits 2-6
        exp0 = svlsr_n_u16_x(pg16, exp0, 8);  // Now in bits 2-6
        exp1 = svlsr_n_u16_x(pg16, exp1, 8);
        exp2 = svlsr_n_u16_x(pg16, exp2, 8);
        exp3 = svlsr_n_u16_x(pg16, exp3, 8);

        // ===== PHASE 7: Round mantissa from 10 bits to 2 bits =====
        // Take bits 8-9, round based on bit 7
        svuint16_t round0 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant0, 7), 1);
        svuint16_t round1 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant1, 7), 1);
        svuint16_t round2 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant2, 7), 1);
        svuint16_t round3 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant3, 7), 1);

        mant0 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant0, 8), round0);
        mant1 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant1, 8), round1);
        mant2 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant2, 8), round2);
        mant3 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant3, 8), round3);

        // ===== PHASE 8: Handle mantissa overflow (mant > 3) =====
        svbool_t ovf0 = svcmpgt_u16(pg16, mant0, three);
        svbool_t ovf1 = svcmpgt_u16(pg16, mant1, three);
        svbool_t ovf2 = svcmpgt_u16(pg16, mant2, three);
        svbool_t ovf3 = svcmpgt_u16(pg16, mant3, three);

        mant0 = svsel_u16(ovf0, zero, mant0);
        mant1 = svsel_u16(ovf1, zero, mant1);
        mant2 = svsel_u16(ovf2, zero, mant2);
        mant3 = svsel_u16(ovf3, zero, mant3);

        // Increment exponent on overflow (add 4 since exp is in bits 2-6)
        exp0 = svadd_u16_x(pg16, exp0, svsel_u16(ovf0, svdup_u16(4), zero));
        exp1 = svadd_u16_x(pg16, exp1, svsel_u16(ovf1, svdup_u16(4), zero));
        exp2 = svadd_u16_x(pg16, exp2, svsel_u16(ovf2, svdup_u16(4), zero));
        exp3 = svadd_u16_x(pg16, exp3, svsel_u16(ovf3, svdup_u16(4), zero));

        // ===== PHASE 9: Clamp exponent overflow to Inf =====
        // If exp > 31<<2 = 124, clamp to Inf (exp=31, mant=0)
        svbool_t inf0 = svcmpgt_u16(pg16, exp0, svdup_u16(124));
        svbool_t inf1 = svcmpgt_u16(pg16, exp1, svdup_u16(124));
        svbool_t inf2 = svcmpgt_u16(pg16, exp2, svdup_u16(124));
        svbool_t inf3 = svcmpgt_u16(pg16, exp3, svdup_u16(124));

        exp0 = svsel_u16(inf0, svdup_u16(124), exp0);
        exp1 = svsel_u16(inf1, svdup_u16(124), exp1);
        exp2 = svsel_u16(inf2, svdup_u16(124), exp2);
        exp3 = svsel_u16(inf3, svdup_u16(124), exp3);

        mant0 = svsel_u16(inf0, zero, mant0);
        mant1 = svsel_u16(inf1, zero, mant1);
        mant2 = svsel_u16(inf2, zero, mant2);
        mant3 = svsel_u16(inf3, zero, mant3);

        // ===== PHASE 10: Combine sign | exp | mant =====
        svuint16_t r0 = svorr_u16_x(pg16, sign0, svorr_u16_x(pg16, exp0, mant0));
        svuint16_t r1 = svorr_u16_x(pg16, sign1, svorr_u16_x(pg16, exp1, mant1));
        svuint16_t r2 = svorr_u16_x(pg16, sign2, svorr_u16_x(pg16, exp2, mant2));
        svuint16_t r3 = svorr_u16_x(pg16, sign3, svorr_u16_x(pg16, exp3, mant3));

        // ===== PHASE 11: Pack 16-bit to 8-bit and store =====
        svuint8_t out0 = svuzp1_u8(svreinterpret_u8_u16(r0), svreinterpret_u8_u16(r0));
        svuint8_t out1 = svuzp1_u8(svreinterpret_u8_u16(r1), svreinterpret_u8_u16(r1));
        svuint8_t out2 = svuzp1_u8(svreinterpret_u8_u16(r2), svreinterpret_u8_u16(r2));
        svuint8_t out3 = svuzp1_u8(svreinterpret_u8_u16(r3), svreinterpret_u8_u16(r3));

        svbool_t pg8 = svwhilelt_b8((uint64_t)0, vl16);
        svst1_u8(pg8, dst + i, out0);
        svst1_u8(pg8, dst + i + vl16, out1);
        svst1_u8(pg8, dst + i + vl16*2, out2);
        svst1_u8(pg8, dst + i + vl16*3, out3);
    }

    // Cleanup: process remaining with single-vector loop
    for (; i + vl16 <= n; i += vl16) {
        svuint16_t x = svld1_u16(pg16, src + i);
        svuint16_t sign = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x, mask_sign), 8);
        svuint16_t exp = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x, mask_exp), 8);
        svuint16_t mant = svand_u16_x(pg16, x, mask_mant);

        svuint16_t round = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant, 7), 1);
        mant = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant, 8), round);

        svbool_t ovf = svcmpgt_u16(pg16, mant, three);
        mant = svsel_u16(ovf, zero, mant);
        exp = svadd_u16_x(pg16, exp, svsel_u16(ovf, svdup_u16(4), zero));

        svbool_t inf = svcmpgt_u16(pg16, exp, svdup_u16(124));
        exp = svsel_u16(inf, svdup_u16(124), exp);
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
// FP32 -> E5M2: Unroll x4
// ============================================================================

void fp32_to_fp8_e5m2_sve_unroll4(const float* src, uint8_t* dst, size_t n) {
    size_t i = 0;
    svbool_t pg32 = svptrue_b32();
    size_t vl32 = svcntw();  // 16 elements
    size_t vl4 = vl32 * 4;   // 64 elements per iteration

    svfloat32_t max_val = svdup_f32(57344.0f);
    svfloat32_t min_val = svdup_f32(6.1e-5f);  // Min normal E5M2

    for (; i + vl4 <= n; i += vl4) {
        // ===== Load 4 vectors =====
        svfloat32_t f0 = svld1_f32(pg32, src + i);
        svfloat32_t f1 = svld1_f32(pg32, src + i + vl32);
        svfloat32_t f2 = svld1_f32(pg32, src + i + vl32*2);
        svfloat32_t f3 = svld1_f32(pg32, src + i + vl32*3);

        // ===== Extract sign =====
        svuint32_t b0 = svreinterpret_u32_f32(f0);
        svuint32_t b1 = svreinterpret_u32_f32(f1);
        svuint32_t b2 = svreinterpret_u32_f32(f2);
        svuint32_t b3 = svreinterpret_u32_f32(f3);

        svuint32_t sign0 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b0, 0x80000000), 24);
        svuint32_t sign1 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b1, 0x80000000), 24);
        svuint32_t sign2 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b2, 0x80000000), 24);
        svuint32_t sign3 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b3, 0x80000000), 24);

        // ===== Get absolute value and clamp =====
        svfloat32_t a0 = svmin_f32_x(pg32, svabs_f32_x(pg32, f0), max_val);
        svfloat32_t a1 = svmin_f32_x(pg32, svabs_f32_x(pg32, f1), max_val);
        svfloat32_t a2 = svmin_f32_x(pg32, svabs_f32_x(pg32, f2), max_val);
        svfloat32_t a3 = svmin_f32_x(pg32, svabs_f32_x(pg32, f3), max_val);

        // Check underflow
        svbool_t uf0 = svcmplt_f32(pg32, a0, min_val);
        svbool_t uf1 = svcmplt_f32(pg32, a1, min_val);
        svbool_t uf2 = svcmplt_f32(pg32, a2, min_val);
        svbool_t uf3 = svcmplt_f32(pg32, a3, min_val);

        svuint32_t ab0 = svreinterpret_u32_f32(a0);
        svuint32_t ab1 = svreinterpret_u32_f32(a1);
        svuint32_t ab2 = svreinterpret_u32_f32(a2);
        svuint32_t ab3 = svreinterpret_u32_f32(a3);

        // ===== Extract and convert exponent =====
        // E5M2_exp = FP32_exp - 112 (bias conversion: 127-15=112)
        svuint32_t exp0 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab0, 23), 112);
        svuint32_t exp1 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab1, 23), 112);
        svuint32_t exp2 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab2, 23), 112);
        svuint32_t exp3 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab3, 23), 112);

        // ===== Round mantissa from 23 bits to 2 bits =====
        svuint32_t mant0 = svand_n_u32_x(pg32, ab0, 0x7FFFFF);
        svuint32_t mant1 = svand_n_u32_x(pg32, ab1, 0x7FFFFF);
        svuint32_t mant2 = svand_n_u32_x(pg32, ab2, 0x7FFFFF);
        svuint32_t mant3 = svand_n_u32_x(pg32, ab3, 0x7FFFFF);

        svuint32_t round0 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant0, 20), 1);
        svuint32_t round1 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant1, 20), 1);
        svuint32_t round2 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant2, 20), 1);
        svuint32_t round3 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant3, 20), 1);

        mant0 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant0, 21), round0);
        mant1 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant1, 21), round1);
        mant2 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant2, 21), round2);
        mant3 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant3, 21), round3);

        // ===== Handle mantissa overflow =====
        svbool_t ovf0 = svcmpgt_u32(pg32, mant0, svdup_u32(3));
        svbool_t ovf1 = svcmpgt_u32(pg32, mant1, svdup_u32(3));
        svbool_t ovf2 = svcmpgt_u32(pg32, mant2, svdup_u32(3));
        svbool_t ovf3 = svcmpgt_u32(pg32, mant3, svdup_u32(3));

        mant0 = svsel_u32(ovf0, svdup_u32(0), mant0);
        mant1 = svsel_u32(ovf1, svdup_u32(0), mant1);
        mant2 = svsel_u32(ovf2, svdup_u32(0), mant2);
        mant3 = svsel_u32(ovf3, svdup_u32(0), mant3);

        exp0 = svadd_u32_x(pg32, exp0, svsel_u32(ovf0, svdup_u32(1), svdup_u32(0)));
        exp1 = svadd_u32_x(pg32, exp1, svsel_u32(ovf1, svdup_u32(1), svdup_u32(0)));
        exp2 = svadd_u32_x(pg32, exp2, svsel_u32(ovf2, svdup_u32(1), svdup_u32(0)));
        exp3 = svadd_u32_x(pg32, exp3, svsel_u32(ovf3, svdup_u32(1), svdup_u32(0)));

        // Clamp exp to 30 max (31 = Inf)
        exp0 = svmin_n_u32_x(pg32, exp0, 30);
        exp1 = svmin_n_u32_x(pg32, exp1, 30);
        exp2 = svmin_n_u32_x(pg32, exp2, 30);
        exp3 = svmin_n_u32_x(pg32, exp3, 30);

        // ===== Combine and handle underflow =====
        svuint32_t r0 = svorr_u32_x(pg32, sign0, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp0, 2), mant0));
        svuint32_t r1 = svorr_u32_x(pg32, sign1, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp1, 2), mant1));
        svuint32_t r2 = svorr_u32_x(pg32, sign2, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp2, 2), mant2));
        svuint32_t r3 = svorr_u32_x(pg32, sign3, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp3, 2), mant3));

        r0 = svsel_u32(uf0, sign0, r0);
        r1 = svsel_u32(uf1, sign1, r1);
        r2 = svsel_u32(uf2, sign2, r2);
        r3 = svsel_u32(uf3, sign3, r3);

        // ===== Pack 32->16->8 and store =====
        svuint16_t h0 = svuzp1_u16(svreinterpret_u16_u32(r0), svreinterpret_u16_u32(r0));
        svuint16_t h1 = svuzp1_u16(svreinterpret_u16_u32(r1), svreinterpret_u16_u32(r1));
        svuint16_t h2 = svuzp1_u16(svreinterpret_u16_u32(r2), svreinterpret_u16_u32(r2));
        svuint16_t h3 = svuzp1_u16(svreinterpret_u16_u32(r3), svreinterpret_u16_u32(r3));

        svuint8_t out0 = svuzp1_u8(svreinterpret_u8_u16(h0), svreinterpret_u8_u16(h0));
        svuint8_t out1 = svuzp1_u8(svreinterpret_u8_u16(h1), svreinterpret_u8_u16(h1));
        svuint8_t out2 = svuzp1_u8(svreinterpret_u8_u16(h2), svreinterpret_u8_u16(h2));
        svuint8_t out3 = svuzp1_u8(svreinterpret_u8_u16(h3), svreinterpret_u8_u16(h3));

        svbool_t pg8 = svwhilelt_b8((uint64_t)0, vl32);
        svst1_u8(pg8, dst + i, out0);
        svst1_u8(pg8, dst + i + vl32, out1);
        svst1_u8(pg8, dst + i + vl32*2, out2);
        svst1_u8(pg8, dst + i + vl32*3, out3);
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

        svbool_t ovf = svcmpgt_u32(pg32, mant, svdup_u32(3));
        mant = svsel_u32(ovf, svdup_u32(0), mant);
        exp = svadd_u32_x(pg32, exp, svsel_u32(ovf, svdup_u32(1), svdup_u32(0)));
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
