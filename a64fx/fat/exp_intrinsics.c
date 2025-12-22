// exp_intrinsics.c
// SVE intrinsics version of exp() with 1-5 term polynomials
// Compare compiler output vs hand-written assembly

#include <arm_sve.h>
#include <stddef.h>

// Constants for exp computation
static const float INV_LN2_F32 = 1.4426950408889634f;  // 1/ln(2)
static const float LN2_F32 = 0.6931471805599453f;      // ln(2)

static const double INV_LN2_F64 = 1.4426950408889634;
static const double LN2_F64 = 0.6931471805599453;

// ============================================
// FP32 exp with 1-term polynomial
// ============================================
void exp_f32_poly1_intrin(const float* restrict input, float* restrict output, size_t count) {
    size_t vl = svcntw();

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);

        // Load input
        svfloat32_t x = svld1_f32(pg, &input[i]);

        // n = round(x / ln2)
        svfloat32_t n = svmul_f32_x(pg, x, svdup_f32(INV_LN2_F32));
        n = svrintn_f32_x(pg, n);

        // r = x - n * ln2
        svfloat32_t r = svmls_f32_x(pg, x, n, svdup_f32(LN2_F32));

        // 2^n via IEEE754 exponent manipulation
        svint32_t ni = svcvt_s32_f32_x(pg, n);
        ni = svadd_s32_x(pg, ni, svdup_s32(127));
        ni = svlsl_n_s32_x(pg, ni, 23);
        svfloat32_t two_n = svreinterpret_f32_s32(ni);

        // 1-term polynomial: p = 1 + r
        svfloat32_t p = svadd_f32_x(pg, svdup_f32(1.0f), r);

        // result = 2^n * p
        svfloat32_t result = svmul_f32_x(pg, p, two_n);

        svst1_f32(pg, &output[i], result);
    }
}

// ============================================
// FP32 exp with 2-term polynomial
// ============================================
void exp_f32_poly2_intrin(const float* restrict input, float* restrict output, size_t count) {
    size_t vl = svcntw();

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);

        svfloat32_t x = svld1_f32(pg, &input[i]);

        svfloat32_t n = svmul_f32_x(pg, x, svdup_f32(INV_LN2_F32));
        n = svrintn_f32_x(pg, n);

        svfloat32_t r = svmls_f32_x(pg, x, n, svdup_f32(LN2_F32));

        svint32_t ni = svcvt_s32_f32_x(pg, n);
        ni = svadd_s32_x(pg, ni, svdup_s32(127));
        ni = svlsl_n_s32_x(pg, ni, 23);
        svfloat32_t two_n = svreinterpret_f32_s32(ni);

        // 2-term: p = 1 + r(1 + r/2) = 1 + r + r²/2
        svfloat32_t p = svmad_f32_x(pg, svdup_f32(0.5f), r, svdup_f32(1.0f));
        p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));

        svfloat32_t result = svmul_f32_x(pg, p, two_n);
        svst1_f32(pg, &output[i], result);
    }
}

// ============================================
// FP32 exp with 3-term polynomial
// ============================================
void exp_f32_poly3_intrin(const float* restrict input, float* restrict output, size_t count) {
    size_t vl = svcntw();
    const float c3 = 1.0f/6.0f;

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);

        svfloat32_t x = svld1_f32(pg, &input[i]);

        svfloat32_t n = svmul_f32_x(pg, x, svdup_f32(INV_LN2_F32));
        n = svrintn_f32_x(pg, n);

        svfloat32_t r = svmls_f32_x(pg, x, n, svdup_f32(LN2_F32));

        svint32_t ni = svcvt_s32_f32_x(pg, n);
        ni = svadd_s32_x(pg, ni, svdup_s32(127));
        ni = svlsl_n_s32_x(pg, ni, 23);
        svfloat32_t two_n = svreinterpret_f32_s32(ni);

        // 3-term Horner: ((1/6)*r + 0.5)*r + 1)*r + 1
        svfloat32_t p = svmad_f32_x(pg, svdup_f32(c3), r, svdup_f32(0.5f));
        p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));
        p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));

        svfloat32_t result = svmul_f32_x(pg, p, two_n);
        svst1_f32(pg, &output[i], result);
    }
}

// ============================================
// FP32 exp with 4-term polynomial
// ============================================
void exp_f32_poly4_intrin(const float* restrict input, float* restrict output, size_t count) {
    size_t vl = svcntw();
    const float c3 = 1.0f/6.0f;
    const float c4 = 1.0f/24.0f;

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);

        svfloat32_t x = svld1_f32(pg, &input[i]);

        svfloat32_t n = svmul_f32_x(pg, x, svdup_f32(INV_LN2_F32));
        n = svrintn_f32_x(pg, n);

        svfloat32_t r = svmls_f32_x(pg, x, n, svdup_f32(LN2_F32));

        svint32_t ni = svcvt_s32_f32_x(pg, n);
        ni = svadd_s32_x(pg, ni, svdup_s32(127));
        ni = svlsl_n_s32_x(pg, ni, 23);
        svfloat32_t two_n = svreinterpret_f32_s32(ni);

        // 4-term Horner
        svfloat32_t p = svmad_f32_x(pg, svdup_f32(c4), r, svdup_f32(c3));
        p = svmad_f32_x(pg, p, r, svdup_f32(0.5f));
        p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));
        p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));

        svfloat32_t result = svmul_f32_x(pg, p, two_n);
        svst1_f32(pg, &output[i], result);
    }
}

// ============================================
// FP32 exp with 5-term polynomial
// ============================================
void exp_f32_poly5_intrin(const float* restrict input, float* restrict output, size_t count) {
    size_t vl = svcntw();
    const float c3 = 1.0f/6.0f;
    const float c4 = 1.0f/24.0f;
    const float c5 = 1.0f/120.0f;

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b32(i, count);

        svfloat32_t x = svld1_f32(pg, &input[i]);

        svfloat32_t n = svmul_f32_x(pg, x, svdup_f32(INV_LN2_F32));
        n = svrintn_f32_x(pg, n);

        svfloat32_t r = svmls_f32_x(pg, x, n, svdup_f32(LN2_F32));

        svint32_t ni = svcvt_s32_f32_x(pg, n);
        ni = svadd_s32_x(pg, ni, svdup_s32(127));
        ni = svlsl_n_s32_x(pg, ni, 23);
        svfloat32_t two_n = svreinterpret_f32_s32(ni);

        // 5-term Horner
        svfloat32_t p = svmad_f32_x(pg, svdup_f32(c5), r, svdup_f32(c4));
        p = svmad_f32_x(pg, p, r, svdup_f32(c3));
        p = svmad_f32_x(pg, p, r, svdup_f32(0.5f));
        p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));
        p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));

        svfloat32_t result = svmul_f32_x(pg, p, two_n);
        svst1_f32(pg, &output[i], result);
    }
}

// ============================================
// FP64 exp with 1-term polynomial
// ============================================
void exp_f64_poly1_intrin(const double* restrict input, double* restrict output, size_t count) {
    size_t vl = svcntd();

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b64(i, count);

        svfloat64_t x = svld1_f64(pg, &input[i]);

        svfloat64_t n = svmul_f64_x(pg, x, svdup_f64(INV_LN2_F64));
        n = svrintn_f64_x(pg, n);

        svfloat64_t r = svmls_f64_x(pg, x, n, svdup_f64(LN2_F64));

        svint64_t ni = svcvt_s64_f64_x(pg, n);
        ni = svadd_s64_x(pg, ni, svdup_s64(1023));
        ni = svlsl_n_s64_x(pg, ni, 52);
        svfloat64_t two_n = svreinterpret_f64_s64(ni);

        svfloat64_t p = svadd_f64_x(pg, svdup_f64(1.0), r);

        svfloat64_t result = svmul_f64_x(pg, p, two_n);
        svst1_f64(pg, &output[i], result);
    }
}

// ============================================
// FP64 exp with 2-term polynomial
// ============================================
void exp_f64_poly2_intrin(const double* restrict input, double* restrict output, size_t count) {
    size_t vl = svcntd();

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b64(i, count);

        svfloat64_t x = svld1_f64(pg, &input[i]);

        svfloat64_t n = svmul_f64_x(pg, x, svdup_f64(INV_LN2_F64));
        n = svrintn_f64_x(pg, n);

        svfloat64_t r = svmls_f64_x(pg, x, n, svdup_f64(LN2_F64));

        svint64_t ni = svcvt_s64_f64_x(pg, n);
        ni = svadd_s64_x(pg, ni, svdup_s64(1023));
        ni = svlsl_n_s64_x(pg, ni, 52);
        svfloat64_t two_n = svreinterpret_f64_s64(ni);

        svfloat64_t p = svmad_f64_x(pg, svdup_f64(0.5), r, svdup_f64(1.0));
        p = svmad_f64_x(pg, p, r, svdup_f64(1.0));

        svfloat64_t result = svmul_f64_x(pg, p, two_n);
        svst1_f64(pg, &output[i], result);
    }
}

// ============================================
// FP64 exp with 3-term polynomial
// ============================================
void exp_f64_poly3_intrin(const double* restrict input, double* restrict output, size_t count) {
    size_t vl = svcntd();
    const double c3 = 1.0/6.0;

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b64(i, count);

        svfloat64_t x = svld1_f64(pg, &input[i]);

        svfloat64_t n = svmul_f64_x(pg, x, svdup_f64(INV_LN2_F64));
        n = svrintn_f64_x(pg, n);

        svfloat64_t r = svmls_f64_x(pg, x, n, svdup_f64(LN2_F64));

        svint64_t ni = svcvt_s64_f64_x(pg, n);
        ni = svadd_s64_x(pg, ni, svdup_s64(1023));
        ni = svlsl_n_s64_x(pg, ni, 52);
        svfloat64_t two_n = svreinterpret_f64_s64(ni);

        svfloat64_t p = svmad_f64_x(pg, svdup_f64(c3), r, svdup_f64(0.5));
        p = svmad_f64_x(pg, p, r, svdup_f64(1.0));
        p = svmad_f64_x(pg, p, r, svdup_f64(1.0));

        svfloat64_t result = svmul_f64_x(pg, p, two_n);
        svst1_f64(pg, &output[i], result);
    }
}

// ============================================
// FP64 exp with 4-term polynomial
// ============================================
void exp_f64_poly4_intrin(const double* restrict input, double* restrict output, size_t count) {
    size_t vl = svcntd();
    const double c3 = 1.0/6.0;
    const double c4 = 1.0/24.0;

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b64(i, count);

        svfloat64_t x = svld1_f64(pg, &input[i]);

        svfloat64_t n = svmul_f64_x(pg, x, svdup_f64(INV_LN2_F64));
        n = svrintn_f64_x(pg, n);

        svfloat64_t r = svmls_f64_x(pg, x, n, svdup_f64(LN2_F64));

        svint64_t ni = svcvt_s64_f64_x(pg, n);
        ni = svadd_s64_x(pg, ni, svdup_s64(1023));
        ni = svlsl_n_s64_x(pg, ni, 52);
        svfloat64_t two_n = svreinterpret_f64_s64(ni);

        svfloat64_t p = svmad_f64_x(pg, svdup_f64(c4), r, svdup_f64(c3));
        p = svmad_f64_x(pg, p, r, svdup_f64(0.5));
        p = svmad_f64_x(pg, p, r, svdup_f64(1.0));
        p = svmad_f64_x(pg, p, r, svdup_f64(1.0));

        svfloat64_t result = svmul_f64_x(pg, p, two_n);
        svst1_f64(pg, &output[i], result);
    }
}

// ============================================
// FP64 exp with 5-term polynomial
// ============================================
void exp_f64_poly5_intrin(const double* restrict input, double* restrict output, size_t count) {
    size_t vl = svcntd();
    const double c3 = 1.0/6.0;
    const double c4 = 1.0/24.0;
    const double c5 = 1.0/120.0;

    for (size_t i = 0; i < count; i += vl) {
        svbool_t pg = svwhilelt_b64(i, count);

        svfloat64_t x = svld1_f64(pg, &input[i]);

        svfloat64_t n = svmul_f64_x(pg, x, svdup_f64(INV_LN2_F64));
        n = svrintn_f64_x(pg, n);

        svfloat64_t r = svmls_f64_x(pg, x, n, svdup_f64(LN2_F64));

        svint64_t ni = svcvt_s64_f64_x(pg, n);
        ni = svadd_s64_x(pg, ni, svdup_s64(1023));
        ni = svlsl_n_s64_x(pg, ni, 52);
        svfloat64_t two_n = svreinterpret_f64_s64(ni);

        svfloat64_t p = svmad_f64_x(pg, svdup_f64(c5), r, svdup_f64(c4));
        p = svmad_f64_x(pg, p, r, svdup_f64(c3));
        p = svmad_f64_x(pg, p, r, svdup_f64(0.5));
        p = svmad_f64_x(pg, p, r, svdup_f64(1.0));
        p = svmad_f64_x(pg, p, r, svdup_f64(1.0));

        svfloat64_t result = svmul_f64_x(pg, p, two_n);
        svst1_f64(pg, &output[i], result);
    }
}
