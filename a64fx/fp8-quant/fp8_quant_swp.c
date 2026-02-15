#include "fp8_quant_swp.h"
#include <arm_sve.h>

// Helper: scalar FP16->E5M2 conversion
static inline uint8_t fp16_to_e5m2_scalar(uint16_t x) {
    uint16_t sign = (x >> 8) & 0x80;
    uint16_t exp = (x >> 8) & 0x7C;
    uint16_t mant = x & 0x3FF;
    uint16_t round_bit = (mant >> 7) & 1;
    mant = (mant >> 8) + round_bit;
    if (mant > 3) { mant = 0; exp += 4; }
    if (exp > 124) { exp = 124; mant = 0; }
    return sign | exp | mant;
}

// Helper: scalar FP32->E5M2 conversion
static inline uint8_t fp32_to_e5m2_scalar(float f) {
    union { float fv; uint32_t u; } c = {f};
    uint32_t sign = (c.u >> 24) & 0x80;
    float a = f < 0 ? -f : f;
    if (a > 57344.0f) a = 57344.0f;
    if (a < 6.1e-5f) return sign;
    union { float fv; uint32_t u; } ac = {a};
    uint32_t exp = ((ac.u >> 23) & 0xFF) - 112;
    uint32_t mant = ac.u & 0x7FFFFF;
    uint32_t round = (mant >> 20) & 1;
    mant = (mant >> 21) + round;
    if (mant > 3) { mant = 0; exp++; }
    if (exp > 30) exp = 30;
    return sign | (exp << 2) | mant;
}

// ============================================================================
// FP16 -> E5M2: Software Pipelined
// ============================================================================

void fp16_to_fp8_e5m2_sve_swp(const uint16_t* src, uint8_t* dst, size_t n) {
    svbool_t pg16 = svptrue_b16();
    size_t vl16 = svcnth();

    svuint16_t mask_sign = svdup_u16(0x8000);
    svuint16_t mask_exp  = svdup_u16(0x7C00);
    svuint16_t mask_mant = svdup_u16(0x03FF);
    svuint16_t zero = svdup_u16(0);
    svuint16_t three = svdup_u16(3);
    svuint16_t four = svdup_u16(4);
    svuint16_t exp_max = svdup_u16(124);
    svbool_t pg8 = svwhilelt_b8((uint64_t)0, vl16);

    size_t i = 0;

    // Need at least 3 vectors for pipelining
    if (n >= vl16 * 3) {
        // PROLOGUE: Load first two, process first
        svuint16_t x0 = svld1_u16(pg16, src);
        svuint16_t x1 = svld1_u16(pg16, src + vl16);

        svuint16_t sign0 = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x0, mask_sign), 8);
        svuint16_t exp0 = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x0, mask_exp), 8);
        svuint16_t mant0 = svand_u16_x(pg16, x0, mask_mant);
        svuint16_t round0 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant0, 7), 1);
        mant0 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant0, 8), round0);
        svbool_t ovf0 = svcmpgt_u16(pg16, mant0, three);
        mant0 = svsel_u16(ovf0, zero, mant0);
        exp0 = svadd_u16_x(pg16, exp0, svsel_u16(ovf0, four, zero));
        svbool_t inf0 = svcmpgt_u16(pg16, exp0, exp_max);
        exp0 = svsel_u16(inf0, exp_max, exp0);
        mant0 = svsel_u16(inf0, zero, mant0);
        svuint16_t r0 = svorr_u16_x(pg16, sign0, svorr_u16_x(pg16, exp0, mant0));
        svuint8_t out0 = svuzp1_u8(svreinterpret_u8_u16(r0), svreinterpret_u8_u16(r0));

        i = vl16 * 2;

        // MAIN LOOP: Load(i), Compute(x1), Store(out0)
        for (; i + vl16 <= n; i += vl16) {
            svuint16_t x2 = svld1_u16(pg16, src + i);

            svuint16_t sign1 = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x1, mask_sign), 8);
            svuint16_t exp1 = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x1, mask_exp), 8);
            svuint16_t mant1 = svand_u16_x(pg16, x1, mask_mant);
            svuint16_t round1 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant1, 7), 1);
            mant1 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant1, 8), round1);
            svbool_t ovf1 = svcmpgt_u16(pg16, mant1, three);
            mant1 = svsel_u16(ovf1, zero, mant1);
            exp1 = svadd_u16_x(pg16, exp1, svsel_u16(ovf1, four, zero));
            svbool_t inf1 = svcmpgt_u16(pg16, exp1, exp_max);
            exp1 = svsel_u16(inf1, exp_max, exp1);
            mant1 = svsel_u16(inf1, zero, mant1);
            svuint16_t r1 = svorr_u16_x(pg16, sign1, svorr_u16_x(pg16, exp1, mant1));
            svuint8_t out1 = svuzp1_u8(svreinterpret_u8_u16(r1), svreinterpret_u8_u16(r1));

            svst1_u8(pg8, dst + i - vl16 * 2, out0);

            x1 = x2;
            out0 = out1;
        }

        // EPILOGUE: Store second-to-last, process and store last
        svst1_u8(pg8, dst + i - vl16 * 2, out0);

        svuint16_t sign1 = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x1, mask_sign), 8);
        svuint16_t exp1 = svlsr_n_u16_x(pg16, svand_u16_x(pg16, x1, mask_exp), 8);
        svuint16_t mant1 = svand_u16_x(pg16, x1, mask_mant);
        svuint16_t round1 = svand_n_u16_x(pg16, svlsr_n_u16_x(pg16, mant1, 7), 1);
        mant1 = svadd_u16_x(pg16, svlsr_n_u16_x(pg16, mant1, 8), round1);
        svbool_t ovf1 = svcmpgt_u16(pg16, mant1, three);
        mant1 = svsel_u16(ovf1, zero, mant1);
        exp1 = svadd_u16_x(pg16, exp1, svsel_u16(ovf1, four, zero));
        svbool_t inf1 = svcmpgt_u16(pg16, exp1, exp_max);
        exp1 = svsel_u16(inf1, exp_max, exp1);
        mant1 = svsel_u16(inf1, zero, mant1);
        svuint16_t r1 = svorr_u16_x(pg16, sign1, svorr_u16_x(pg16, exp1, mant1));
        svuint8_t out1 = svuzp1_u8(svreinterpret_u8_u16(r1), svreinterpret_u8_u16(r1));
        svst1_u8(pg8, dst + i - vl16, out1);
    } else {
        // Small size fallback - simple loop
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
            svst1_u8(pg8, dst + i, out);
        }
    }

    // Scalar tail
    for (; i < n; i++) {
        dst[i] = fp16_to_e5m2_scalar(src[i]);
    }
}

// ============================================================================
// FP32 -> E5M2: Software Pipelined
// ============================================================================

void fp32_to_fp8_e5m2_sve_swp(const float* src, uint8_t* dst, size_t n) {
    svbool_t pg32 = svptrue_b32();
    size_t vl32 = svcntw();

    svfloat32_t max_val = svdup_f32(57344.0f);
    svfloat32_t min_val = svdup_f32(6.1e-5f);
    svuint32_t zero32 = svdup_u32(0);
    svuint32_t one32 = svdup_u32(1);
    svuint32_t three32 = svdup_u32(3);
    svbool_t pg8 = svwhilelt_b8((uint64_t)0, vl32);

    size_t i = 0;

    if (n >= vl32 * 3) {
        // PROLOGUE
        svfloat32_t f0 = svld1_f32(pg32, src);
        svfloat32_t f1 = svld1_f32(pg32, src + vl32);

        svuint32_t b0 = svreinterpret_u32_f32(f0);
        svuint32_t sign0 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b0, 0x80000000), 24);
        svfloat32_t a0 = svmin_f32_x(pg32, svabs_f32_x(pg32, f0), max_val);
        svbool_t uf0 = svcmplt_f32(pg32, a0, min_val);
        svuint32_t ab0 = svreinterpret_u32_f32(a0);
        svuint32_t exp0 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab0, 23), 112);
        svuint32_t mant0 = svand_n_u32_x(pg32, ab0, 0x7FFFFF);
        svuint32_t round0 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant0, 20), 1);
        mant0 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant0, 21), round0);
        svbool_t ovf0 = svcmpgt_u32(pg32, mant0, three32);
        mant0 = svsel_u32(ovf0, zero32, mant0);
        exp0 = svadd_u32_x(pg32, exp0, svsel_u32(ovf0, one32, zero32));
        exp0 = svmin_n_u32_x(pg32, exp0, 30);
        svuint32_t r0 = svorr_u32_x(pg32, sign0, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp0, 2), mant0));
        r0 = svsel_u32(uf0, sign0, r0);
        svuint16_t h0 = svuzp1_u16(svreinterpret_u16_u32(r0), svreinterpret_u16_u32(r0));
        svuint8_t out0 = svuzp1_u8(svreinterpret_u8_u16(h0), svreinterpret_u8_u16(h0));

        i = vl32 * 2;

        // MAIN LOOP
        for (; i + vl32 <= n; i += vl32) {
            svfloat32_t f2 = svld1_f32(pg32, src + i);

            svuint32_t b1 = svreinterpret_u32_f32(f1);
            svuint32_t sign1 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b1, 0x80000000), 24);
            svfloat32_t a1 = svmin_f32_x(pg32, svabs_f32_x(pg32, f1), max_val);
            svbool_t uf1 = svcmplt_f32(pg32, a1, min_val);
            svuint32_t ab1 = svreinterpret_u32_f32(a1);
            svuint32_t exp1 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab1, 23), 112);
            svuint32_t mant1 = svand_n_u32_x(pg32, ab1, 0x7FFFFF);
            svuint32_t round1 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant1, 20), 1);
            mant1 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant1, 21), round1);
            svbool_t ovf1 = svcmpgt_u32(pg32, mant1, three32);
            mant1 = svsel_u32(ovf1, zero32, mant1);
            exp1 = svadd_u32_x(pg32, exp1, svsel_u32(ovf1, one32, zero32));
            exp1 = svmin_n_u32_x(pg32, exp1, 30);
            svuint32_t r1 = svorr_u32_x(pg32, sign1, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp1, 2), mant1));
            r1 = svsel_u32(uf1, sign1, r1);
            svuint16_t h1 = svuzp1_u16(svreinterpret_u16_u32(r1), svreinterpret_u16_u32(r1));
            svuint8_t out1 = svuzp1_u8(svreinterpret_u8_u16(h1), svreinterpret_u8_u16(h1));

            svst1_u8(pg8, dst + i - vl32 * 2, out0);

            f1 = f2;
            out0 = out1;
        }

        // EPILOGUE
        svst1_u8(pg8, dst + i - vl32 * 2, out0);

        svuint32_t b1 = svreinterpret_u32_f32(f1);
        svuint32_t sign1 = svlsr_n_u32_x(pg32, svand_n_u32_x(pg32, b1, 0x80000000), 24);
        svfloat32_t a1 = svmin_f32_x(pg32, svabs_f32_x(pg32, f1), max_val);
        svbool_t uf1 = svcmplt_f32(pg32, a1, min_val);
        svuint32_t ab1 = svreinterpret_u32_f32(a1);
        svuint32_t exp1 = svsub_n_u32_x(pg32, svlsr_n_u32_x(pg32, ab1, 23), 112);
        svuint32_t mant1 = svand_n_u32_x(pg32, ab1, 0x7FFFFF);
        svuint32_t round1 = svand_n_u32_x(pg32, svlsr_n_u32_x(pg32, mant1, 20), 1);
        mant1 = svadd_u32_x(pg32, svlsr_n_u32_x(pg32, mant1, 21), round1);
        svbool_t ovf1 = svcmpgt_u32(pg32, mant1, three32);
        mant1 = svsel_u32(ovf1, zero32, mant1);
        exp1 = svadd_u32_x(pg32, exp1, svsel_u32(ovf1, one32, zero32));
        exp1 = svmin_n_u32_x(pg32, exp1, 30);
        svuint32_t r1 = svorr_u32_x(pg32, sign1, svorr_u32_x(pg32, svlsl_n_u32_x(pg32, exp1, 2), mant1));
        r1 = svsel_u32(uf1, sign1, r1);
        svuint16_t h1 = svuzp1_u16(svreinterpret_u16_u32(r1), svreinterpret_u16_u32(r1));
        svuint8_t out1 = svuzp1_u8(svreinterpret_u8_u16(h1), svreinterpret_u8_u16(h1));
        svst1_u8(pg8, dst + i - vl32, out1);
    } else {
        // Small size fallback
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
            svst1_u8(pg8, dst + i, out);
        }
    }

    // Scalar tail
    for (; i < n; i++) {
        dst[i] = fp32_to_e5m2_scalar(src[i]);
    }
}

// ============================================================================
// FP32 -> E5M2: Using hardware FCVT via inline assembly
// Complete conversion in single asm block to avoid memory traffic
// ============================================================================

// Full FP32 -> E5M2 conversion in pure asm
// z0: input FP32, z1: max_val, z2: working reg
// Result: E5M2 bytes stored to dst
static void fp32_to_e5m2_asm_block(const float* in, uint8_t* out, float max_v) {
    // Constants for E5M2 conversion
    // FP16: 1 sign + 5 exp + 10 mant
    // E5M2: 1 sign + 5 exp + 2 mant (same bias 15)
    asm volatile(
        // Setup
        "ptrue p0.s\n\t"
        "ld1w {z0.s}, p0/z, [%0]\n\t"         // Load FP32

        // Extract and save sign (bit 31 -> bit 7 of result)
        "lsr z3.s, z0.s, #24\n\t"             // Sign to bits 0-7
        "and z3.s, z3.s, #0x80\n\t"           // Keep only bit 7

        // Absolute value and clamp
        "dup z1.s, %w2\n\t"                   // max value = 57344
        "fabs z0.s, p0/m, z0.s\n\t"           // |x|
        "fmin z0.s, p0/m, z0.s, z1.s\n\t"     // min(|x|, max)

        // FP32 -> FP16 (hardware conversion)
        "fcvt z0.h, p0/m, z0.s\n\t"

        // Now z0 has FP16 in lower 16 bits of each 32-bit lane
        // FP16 format: [15:sign][14:10:exp][9:0:mant]

        // Extract exponent: (fp16 >> 10) & 0x1F, then shift to bits 2-6
        // = (fp16 & 0x7C00) >> 8
        "mov z1.d, z0.d\n\t"                  // copy
        "and z1.s, z1.s, #0x7C00\n\t"         // exp field
        "lsr z1.s, z1.s, #8\n\t"              // exp in bits 2-6

        // Extract mantissa: fp16 & 0x3FF
        "mov z2.d, z0.d\n\t"                  // copy
        "and z2.s, z2.s, #0x3FF\n\t"          // mant field

        // Round mantissa from 10 bits to 2 bits
        // round_bit = (mant >> 7) & 1
        // e5m2_mant = (mant >> 8) + round_bit
        "lsr z4.s, z2.s, #7\n\t"
        "and z4.s, z4.s, #1\n\t"              // round bit
        "lsr z2.s, z2.s, #8\n\t"              // mant >> 8
        "add z2.s, z2.s, z4.s\n\t"            // + round

        // Handle mantissa overflow (mant > 3)
        "mov z4.s, #3\n\t"
        "cmpgt p1.s, p0/z, z2.s, z4.s\n\t"    // mant > 3?
        "mov z2.s, p1/m, #0\n\t"              // mant = 0 if overflow
        "mov z4.s, #4\n\t"
        "add z1.s, p1/m, z1.s, z4.s\n\t"      // exp += 4 if overflow

        // Clamp exp to max (124 = 31 << 2)
        "mov z4.s, #124\n\t"
        "cmpgt p1.s, p0/z, z1.s, z4.s\n\t"    // exp > 124?
        "mov z1.s, p1/m, #124\n\t"            // clamp exp
        "mov z2.s, p1/m, #0\n\t"              // mant = 0 if exp overflow

        // Combine: sign | exp | mant
        "orr z1.d, z1.d, z2.d\n\t"            // exp | mant
        "orr z0.d, z3.d, z1.d\n\t"            // sign | exp | mant

        // Pack 32-bit -> 16-bit -> 8-bit
        "uzp1 z0.h, z0.h, z0.h\n\t"
        "uzp1 z0.b, z0.b, z0.b\n\t"

        // Store 16 bytes
        "ptrue p1.b, vl16\n\t"
        "st1b {z0.b}, p1, [%1]"
        :
        : "r"(in), "r"(out), "r"(*(uint32_t*)&max_v)
        : "z0", "z1", "z2", "z3", "z4", "p0", "p1", "memory"
    );
}

void fp32_to_fp8_e5m2_sve_fcvt(const float* src, uint8_t* dst, size_t n) {
    size_t vl32 = 16;  // A64FX: 16 FP32 per vector
    float max_val = 57344.0f;

    size_t i = 0;
    for (; i + vl32 <= n; i += vl32) {
        fp32_to_e5m2_asm_block(src + i, dst + i, max_val);
    }

    // Scalar tail
    for (; i < n; i++) {
        dst[i] = fp32_to_e5m2_scalar(src[i]);
    }
}
