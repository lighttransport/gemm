/*
 * FP8->FP32 Direct Bitwise Conversion (no LUT)
 *
 * FP8 E4M3: 1 sign + 4 exp (bias=7) + 3 mantissa
 * FP32:     1 sign + 8 exp (bias=127) + 23 mantissa
 *
 * Conversion (normal case):
 *   fp32_sign = fp8_sign << 24
 *   fp32_exp = (fp8_exp + 120) << 23
 *   fp32_mant = fp8_mant << 20
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <arm_sve.h>
#include <math.h>

#define ALIGN 64

typedef uint8_t fp8_e4m3_t;
uint32_t fp8_to_fp32_lut[256] __attribute__((aligned(64)));

static inline uint64_t get_ticks(void) {
    uint64_t val;
    __asm__ volatile("isb" ::: "memory");
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    __asm__ volatile("isb" ::: "memory");
    return val;
}

void init_lut(void) {
    for (int i = 0; i < 256; i++) {
        uint8_t sign = (i >> 7) & 1;
        uint8_t exp = (i >> 3) & 0xF;
        uint8_t mant = i & 0x7;
        uint32_t fp32;
        if (exp == 0) {
            if (mant == 0) fp32 = (uint32_t)sign << 31;
            else {
                int shift = 0;
                uint8_t m = mant;
                while ((m & 0x4) == 0) { m <<= 1; shift++; }
                fp32 = ((uint32_t)sign << 31) | ((127 - 6 - shift) << 23) | ((uint32_t)(m & 3) << 21);
            }
        } else if (exp == 15 && mant == 7) {
            fp32 = ((uint32_t)sign << 31) | 0x7FC00000;
        } else {
            fp32 = ((uint32_t)sign << 31) | ((exp + 120) << 23) | ((uint32_t)mant << 20);
        }
        fp8_to_fp32_lut[i] = fp32;
    }
}

// Direct bitwise conversion (scalar, for verification)
static inline uint32_t fp8_to_fp32_direct(uint8_t fp8) {
    uint32_t sign = (fp8 >> 7) & 1;
    uint32_t exp = (fp8 >> 3) & 0xF;
    uint32_t mant = fp8 & 0x7;

    if (exp == 0) {
        // Zero or subnormal (treat as zero for simplicity)
        return sign << 31;
    }
    // Normal: fp32_exp = fp8_exp + 120, fp32_mant = fp8_mant << 20
    return (sign << 31) | ((exp + 120) << 23) | (mant << 20);
}

// SVE gather conversion
void convert_sve_gather(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t indices = svld1ub_u32(pg, src + i);
        svuint32_t fp32_bits = svld1_gather_index(pg, fp8_to_fp32_lut, indices);
        svst1(pg, dst + i, svreinterpret_f32(fp32_bits));
    }
}

// SVE direct bitwise conversion (no gather!)
void convert_sve_direct(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();
    svuint32_t bias = svdup_u32(120 << 23);

    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);

        // Load FP8 bytes, zero-extend to u32
        svuint32_t fp8 = svld1ub_u32(pg, src + i);

        // Extract sign: (fp8 >> 7) & 1, then shift to bit 31
        svuint32_t sign = svand_x(pg, svlsr_x(pg, fp8, 7), svdup_u32(1));
        sign = svlsl_x(pg, sign, 31);

        // Extract exp: (fp8 >> 3) & 0xF
        svuint32_t exp = svand_x(pg, svlsr_x(pg, fp8, 3), svdup_u32(0xF));

        // Extract mant: fp8 & 0x7
        svuint32_t mant = svand_x(pg, fp8, svdup_u32(0x7));

        // Build FP32: sign | ((exp + 120) << 23) | (mant << 20)
        svuint32_t fp32_exp = svadd_x(pg, exp, svdup_u32(120));
        fp32_exp = svlsl_x(pg, fp32_exp, 23);
        svuint32_t fp32_mant = svlsl_x(pg, mant, 20);

        svuint32_t result = svorr_x(pg, sign, fp32_exp);
        result = svorr_x(pg, result, fp32_mant);

        // Handle zero case: if exp==0, result should be just sign
        svbool_t is_zero = svcmpeq(pg, exp, svdup_u32(0));
        result = svsel(is_zero, sign, result);

        svst1(pg, dst + i, svreinterpret_f32(result));
    }
}

// SVE direct - optimized (fewer operations)
void convert_sve_direct_opt(const fp8_e4m3_t* src, float* dst, int64_t count) {
    int64_t vl = svcntw();

    for (int64_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);
        svuint32_t fp8 = svld1ub_u32(pg, src + i);

        // sign bit: (fp8 & 0x80) << 24
        svuint32_t sign = svand_x(pg, fp8, svdup_u32(0x80));
        sign = svlsl_x(pg, sign, 24);

        // exp field: ((fp8 & 0x78) + (120 << 3)) << 20
        svuint32_t exp_field = svand_x(pg, fp8, svdup_u32(0x78));
        exp_field = svadd_x(pg, exp_field, svdup_u32(120 << 3));
        exp_field = svlsl_x(pg, exp_field, 20);

        // mant field: (fp8 & 0x7) << 20
        svuint32_t mant = svand_x(pg, fp8, svdup_u32(0x7));
        mant = svlsl_x(pg, mant, 20);

        svuint32_t result = svorr_x(pg, sign, exp_field);
        result = svorr_x(pg, result, mant);

        // Zero handling: if exp_field becomes 120<<23, check if original exp was 0
        svuint32_t orig_exp = svand_x(pg, fp8, svdup_u32(0x78));
        svbool_t is_zero = svcmpeq(pg, orig_exp, svdup_u32(0));
        result = svsel(is_zero, sign, result);

        svst1(pg, dst + i, svreinterpret_f32(result));
    }
}

int main() {
    printf("=== Direct FP8->FP32 Conversion (No LUT) ===\n\n");
    init_lut();

    int64_t count = 384 * 512;  // 196,608 elements
    int iters = 100;

    printf("Elements: %ld\n\n", count);

    fp8_e4m3_t* src = aligned_alloc(ALIGN, count);
    float* dst_gather = aligned_alloc(ALIGN, count * sizeof(float));
    float* dst_direct = aligned_alloc(ALIGN, count * sizeof(float));

    srand(42);
    for (int64_t i = 0; i < count; i++) {
        src[i] = ((rand() % 14 + 1) << 3) | (rand() & 7);  // Avoid exp=0
    }

    uint64_t t0, t1;

    // SVE gather
    printf("1. SVE gather (LUT):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_sve_gather(src, dst_gather, count);
    }
    t1 = get_ticks();
    printf("   %lu ticks (%lu per iter)\n", t1-t0, (t1-t0)/iters);

    // SVE direct
    printf("2. SVE direct (bitwise):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_sve_direct(src, dst_direct, count);
    }
    t1 = get_ticks();
    printf("   %lu ticks (%lu per iter)\n", t1-t0, (t1-t0)/iters);

    // SVE direct optimized
    printf("3. SVE direct (optimized):\n");
    t0 = get_ticks();
    for (int it = 0; it < iters; it++) {
        convert_sve_direct_opt(src, dst_direct, count);
    }
    t1 = get_ticks();
    printf("   %lu ticks (%lu per iter)\n\n", t1-t0, (t1-t0)/iters);

    // Verify correctness
    printf("Verifying correctness...\n");
    convert_sve_gather(src, dst_gather, count);
    convert_sve_direct(src, dst_direct, count);

    int errors = 0;
    for (int64_t i = 0; i < count && errors < 10; i++) {
        if (dst_gather[i] != dst_direct[i]) {
            printf("  Mismatch at %ld: LUT=%.6f, direct=%.6f (fp8=0x%02x)\n",
                   i, dst_gather[i], dst_direct[i], src[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  All values match!\n");
    else printf("  Found %d mismatches\n", errors);

    free(src); free(dst_gather); free(dst_direct);
    return 0;
}
