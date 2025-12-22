// exp_intrinsics_opt.c
// Optimized SVE intrinsics version - process 4 vectors per iteration like asm

#include <arm_sve.h>
#include <stddef.h>

// Constants
static const float INV_LN2_F32 = 1.4426950408889634f;
static const float LN2_F32 = 0.6931471805599453f;

static const double INV_LN2_F64 = 1.4426950408889634;
static const double LN2_F64 = 0.6931471805599453;

// ============================================
// FP32 exp with 5-term polynomial - 4x unrolled
// ============================================
void exp_f32_poly5_intrin_opt(const float* restrict input, float* restrict output, size_t count) {
    size_t vl = svcntw();
    size_t vl4 = vl * 4;

    // Constants broadcast
    svfloat32_t inv_ln2 = svdup_f32(INV_LN2_F32);
    svfloat32_t ln2 = svdup_f32(LN2_F32);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t half = svdup_f32(0.5f);
    svfloat32_t c3 = svdup_f32(1.0f/6.0f);
    svfloat32_t c4 = svdup_f32(1.0f/24.0f);
    svfloat32_t c5 = svdup_f32(1.0f/120.0f);
    svint32_t bias = svdup_s32(127);

    svbool_t pg = svptrue_b32();
    size_t i = 0;

    // Main loop: 4 vectors at a time
    for (; i + vl4 <= count; i += vl4) {
        // Load 4 vectors
        svfloat32_t x0 = svld1_f32(pg, &input[i]);
        svfloat32_t x1 = svld1_f32(pg, &input[i + vl]);
        svfloat32_t x2 = svld1_f32(pg, &input[i + vl*2]);
        svfloat32_t x3 = svld1_f32(pg, &input[i + vl*3]);

        // n = round(x / ln2)
        svfloat32_t n0 = svrintn_f32_x(pg, svmul_f32_x(pg, x0, inv_ln2));
        svfloat32_t n1 = svrintn_f32_x(pg, svmul_f32_x(pg, x1, inv_ln2));
        svfloat32_t n2 = svrintn_f32_x(pg, svmul_f32_x(pg, x2, inv_ln2));
        svfloat32_t n3 = svrintn_f32_x(pg, svmul_f32_x(pg, x3, inv_ln2));

        // r = x - n * ln2
        svfloat32_t r0 = svmls_f32_x(pg, x0, n0, ln2);
        svfloat32_t r1 = svmls_f32_x(pg, x1, n1, ln2);
        svfloat32_t r2 = svmls_f32_x(pg, x2, n2, ln2);
        svfloat32_t r3 = svmls_f32_x(pg, x3, n3, ln2);

        // 2^n
        svint32_t ni0 = svlsl_n_s32_x(pg, svadd_s32_x(pg, svcvt_s32_f32_x(pg, n0), bias), 23);
        svint32_t ni1 = svlsl_n_s32_x(pg, svadd_s32_x(pg, svcvt_s32_f32_x(pg, n1), bias), 23);
        svint32_t ni2 = svlsl_n_s32_x(pg, svadd_s32_x(pg, svcvt_s32_f32_x(pg, n2), bias), 23);
        svint32_t ni3 = svlsl_n_s32_x(pg, svadd_s32_x(pg, svcvt_s32_f32_x(pg, n3), bias), 23);

        svfloat32_t two_n0 = svreinterpret_f32_s32(ni0);
        svfloat32_t two_n1 = svreinterpret_f32_s32(ni1);
        svfloat32_t two_n2 = svreinterpret_f32_s32(ni2);
        svfloat32_t two_n3 = svreinterpret_f32_s32(ni3);

        // 5-term Horner polynomial
        svfloat32_t p0 = svmad_f32_x(pg, c5, r0, c4);
        svfloat32_t p1 = svmad_f32_x(pg, c5, r1, c4);
        svfloat32_t p2 = svmad_f32_x(pg, c5, r2, c4);
        svfloat32_t p3 = svmad_f32_x(pg, c5, r3, c4);

        p0 = svmad_f32_x(pg, p0, r0, c3);
        p1 = svmad_f32_x(pg, p1, r1, c3);
        p2 = svmad_f32_x(pg, p2, r2, c3);
        p3 = svmad_f32_x(pg, p3, r3, c3);

        p0 = svmad_f32_x(pg, p0, r0, half);
        p1 = svmad_f32_x(pg, p1, r1, half);
        p2 = svmad_f32_x(pg, p2, r2, half);
        p3 = svmad_f32_x(pg, p3, r3, half);

        p0 = svmad_f32_x(pg, p0, r0, one);
        p1 = svmad_f32_x(pg, p1, r1, one);
        p2 = svmad_f32_x(pg, p2, r2, one);
        p3 = svmad_f32_x(pg, p3, r3, one);

        p0 = svmad_f32_x(pg, p0, r0, one);
        p1 = svmad_f32_x(pg, p1, r1, one);
        p2 = svmad_f32_x(pg, p2, r2, one);
        p3 = svmad_f32_x(pg, p3, r3, one);

        // result = 2^n * p
        svfloat32_t res0 = svmul_f32_x(pg, p0, two_n0);
        svfloat32_t res1 = svmul_f32_x(pg, p1, two_n1);
        svfloat32_t res2 = svmul_f32_x(pg, p2, two_n2);
        svfloat32_t res3 = svmul_f32_x(pg, p3, two_n3);

        // Store 4 vectors
        svst1_f32(pg, &output[i], res0);
        svst1_f32(pg, &output[i + vl], res1);
        svst1_f32(pg, &output[i + vl*2], res2);
        svst1_f32(pg, &output[i + vl*3], res3);
    }

    // Tail: remaining elements
    for (; i < count; i += vl) {
        svbool_t pg_tail = svwhilelt_b32(i, count);

        svfloat32_t x = svld1_f32(pg_tail, &input[i]);
        svfloat32_t n = svrintn_f32_x(pg_tail, svmul_f32_x(pg_tail, x, inv_ln2));
        svfloat32_t r = svmls_f32_x(pg_tail, x, n, ln2);

        svint32_t ni = svlsl_n_s32_x(pg_tail, svadd_s32_x(pg_tail, svcvt_s32_f32_x(pg_tail, n), bias), 23);
        svfloat32_t two_n = svreinterpret_f32_s32(ni);

        svfloat32_t p = svmad_f32_x(pg_tail, c5, r, c4);
        p = svmad_f32_x(pg_tail, p, r, c3);
        p = svmad_f32_x(pg_tail, p, r, half);
        p = svmad_f32_x(pg_tail, p, r, one);
        p = svmad_f32_x(pg_tail, p, r, one);

        svfloat32_t result = svmul_f32_x(pg_tail, p, two_n);
        svst1_f32(pg_tail, &output[i], result);
    }
}

// ============================================
// FP64 exp with 5-term polynomial - 4x unrolled
// ============================================
void exp_f64_poly5_intrin_opt(const double* restrict input, double* restrict output, size_t count) {
    size_t vl = svcntd();
    size_t vl4 = vl * 4;

    svfloat64_t inv_ln2 = svdup_f64(INV_LN2_F64);
    svfloat64_t ln2 = svdup_f64(LN2_F64);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t half = svdup_f64(0.5);
    svfloat64_t c3 = svdup_f64(1.0/6.0);
    svfloat64_t c4 = svdup_f64(1.0/24.0);
    svfloat64_t c5 = svdup_f64(1.0/120.0);
    svint64_t bias = svdup_s64(1023);

    svbool_t pg = svptrue_b64();
    size_t i = 0;

    for (; i + vl4 <= count; i += vl4) {
        svfloat64_t x0 = svld1_f64(pg, &input[i]);
        svfloat64_t x1 = svld1_f64(pg, &input[i + vl]);
        svfloat64_t x2 = svld1_f64(pg, &input[i + vl*2]);
        svfloat64_t x3 = svld1_f64(pg, &input[i + vl*3]);

        svfloat64_t n0 = svrintn_f64_x(pg, svmul_f64_x(pg, x0, inv_ln2));
        svfloat64_t n1 = svrintn_f64_x(pg, svmul_f64_x(pg, x1, inv_ln2));
        svfloat64_t n2 = svrintn_f64_x(pg, svmul_f64_x(pg, x2, inv_ln2));
        svfloat64_t n3 = svrintn_f64_x(pg, svmul_f64_x(pg, x3, inv_ln2));

        svfloat64_t r0 = svmls_f64_x(pg, x0, n0, ln2);
        svfloat64_t r1 = svmls_f64_x(pg, x1, n1, ln2);
        svfloat64_t r2 = svmls_f64_x(pg, x2, n2, ln2);
        svfloat64_t r3 = svmls_f64_x(pg, x3, n3, ln2);

        svint64_t ni0 = svlsl_n_s64_x(pg, svadd_s64_x(pg, svcvt_s64_f64_x(pg, n0), bias), 52);
        svint64_t ni1 = svlsl_n_s64_x(pg, svadd_s64_x(pg, svcvt_s64_f64_x(pg, n1), bias), 52);
        svint64_t ni2 = svlsl_n_s64_x(pg, svadd_s64_x(pg, svcvt_s64_f64_x(pg, n2), bias), 52);
        svint64_t ni3 = svlsl_n_s64_x(pg, svadd_s64_x(pg, svcvt_s64_f64_x(pg, n3), bias), 52);

        svfloat64_t two_n0 = svreinterpret_f64_s64(ni0);
        svfloat64_t two_n1 = svreinterpret_f64_s64(ni1);
        svfloat64_t two_n2 = svreinterpret_f64_s64(ni2);
        svfloat64_t two_n3 = svreinterpret_f64_s64(ni3);

        svfloat64_t p0 = svmad_f64_x(pg, c5, r0, c4);
        svfloat64_t p1 = svmad_f64_x(pg, c5, r1, c4);
        svfloat64_t p2 = svmad_f64_x(pg, c5, r2, c4);
        svfloat64_t p3 = svmad_f64_x(pg, c5, r3, c4);

        p0 = svmad_f64_x(pg, p0, r0, c3);
        p1 = svmad_f64_x(pg, p1, r1, c3);
        p2 = svmad_f64_x(pg, p2, r2, c3);
        p3 = svmad_f64_x(pg, p3, r3, c3);

        p0 = svmad_f64_x(pg, p0, r0, half);
        p1 = svmad_f64_x(pg, p1, r1, half);
        p2 = svmad_f64_x(pg, p2, r2, half);
        p3 = svmad_f64_x(pg, p3, r3, half);

        p0 = svmad_f64_x(pg, p0, r0, one);
        p1 = svmad_f64_x(pg, p1, r1, one);
        p2 = svmad_f64_x(pg, p2, r2, one);
        p3 = svmad_f64_x(pg, p3, r3, one);

        p0 = svmad_f64_x(pg, p0, r0, one);
        p1 = svmad_f64_x(pg, p1, r1, one);
        p2 = svmad_f64_x(pg, p2, r2, one);
        p3 = svmad_f64_x(pg, p3, r3, one);

        svfloat64_t res0 = svmul_f64_x(pg, p0, two_n0);
        svfloat64_t res1 = svmul_f64_x(pg, p1, two_n1);
        svfloat64_t res2 = svmul_f64_x(pg, p2, two_n2);
        svfloat64_t res3 = svmul_f64_x(pg, p3, two_n3);

        svst1_f64(pg, &output[i], res0);
        svst1_f64(pg, &output[i + vl], res1);
        svst1_f64(pg, &output[i + vl*2], res2);
        svst1_f64(pg, &output[i + vl*3], res3);
    }

    for (; i < count; i += vl) {
        svbool_t pg_tail = svwhilelt_b64(i, count);

        svfloat64_t x = svld1_f64(pg_tail, &input[i]);
        svfloat64_t n = svrintn_f64_x(pg_tail, svmul_f64_x(pg_tail, x, inv_ln2));
        svfloat64_t r = svmls_f64_x(pg_tail, x, n, ln2);

        svint64_t ni = svlsl_n_s64_x(pg_tail, svadd_s64_x(pg_tail, svcvt_s64_f64_x(pg_tail, n), bias), 52);
        svfloat64_t two_n = svreinterpret_f64_s64(ni);

        svfloat64_t p = svmad_f64_x(pg_tail, c5, r, c4);
        p = svmad_f64_x(pg_tail, p, r, c3);
        p = svmad_f64_x(pg_tail, p, r, half);
        p = svmad_f64_x(pg_tail, p, r, one);
        p = svmad_f64_x(pg_tail, p, r, one);

        svfloat64_t result = svmul_f64_x(pg_tail, p, two_n);
        svst1_f64(pg_tail, &output[i], result);
    }
}
