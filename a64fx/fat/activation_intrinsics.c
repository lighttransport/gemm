// activation_intrinsics.c
// SVE intrinsics implementations of activation functions
// GELU, GELU (tanh approx), QuickGELU, SiLU
// Targeting A64FX with 512-bit SVE

#include <arm_sve.h>
#include <math.h>
#include <stddef.h>

// ============================================
// Constants
// ============================================

// GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
// sqrt(2) = 1.4142135623730951
// 1/sqrt(2) = 0.7071067811865476

// GELU tanh approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// sqrt(2/pi) = 0.7978845608028654
// 0.044715

// QuickGELU: x * sigmoid(1.702 * x)

// SiLU (Swish): x * sigmoid(x)

// ============================================
// Helper: Fast exponential approximation (FP32)
// Uses polynomial approximation
// ============================================

// exp(x) approximation using range reduction and polynomial
// exp(x) = 2^n * exp(r) where x = n*ln(2) + r, |r| <= ln(2)/2
static inline svfloat32_t exp_f32_approx(svbool_t pg, svfloat32_t x) {
    const float log2e = 1.4426950408889634f;
    const float ln2_hi = 0.6931471805599453f;
    const float ln2_lo = 1.4286068203094172e-6f;

    // Clamp to avoid overflow/underflow
    x = svmax_f32_x(pg, x, svdup_f32(-88.0f));
    x = svmin_f32_x(pg, x, svdup_f32(88.0f));

    // n = round(x / ln(2))
    svfloat32_t n = svrintn_f32_x(pg, svmul_f32_x(pg, x, svdup_f32(log2e)));

    // r = x - n * ln(2)
    svfloat32_t r = svsub_f32_x(pg, x, svmul_f32_x(pg, n, svdup_f32(ln2_hi)));
    r = svsub_f32_x(pg, r, svmul_f32_x(pg, n, svdup_f32(ln2_lo)));

    // Polynomial approximation for exp(r): 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    svfloat32_t r2 = svmul_f32_x(pg, r, r);
    svfloat32_t p = svdup_f32(1.0f / 120.0f);
    p = svmad_f32_x(pg, p, r, svdup_f32(1.0f / 24.0f));
    p = svmad_f32_x(pg, p, r, svdup_f32(1.0f / 6.0f));
    p = svmad_f32_x(pg, p, r, svdup_f32(0.5f));
    p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));
    p = svmad_f32_x(pg, p, r, svdup_f32(1.0f));

    // Scale by 2^n using floating-point bit manipulation
    svint32_t ni = svcvt_s32_f32_x(pg, n);
    ni = svadd_s32_x(pg, ni, svdup_s32(127));  // Add bias
    ni = svlsl_n_s32_x(pg, ni, 23);            // Shift to exponent position
    svfloat32_t scale = svreinterpret_f32_s32(ni);

    return svmul_f32_x(pg, p, scale);
}

// exp(x) approximation for FP64
static inline svfloat64_t exp_f64_approx(svbool_t pg, svfloat64_t x) {
    const double log2e = 1.4426950408889634;
    const double ln2_hi = 0.6931471805599453;
    const double ln2_lo = 2.3190468138462996e-17;

    x = svmax_f64_x(pg, x, svdup_f64(-708.0));
    x = svmin_f64_x(pg, x, svdup_f64(708.0));

    svfloat64_t n = svrintn_f64_x(pg, svmul_f64_x(pg, x, svdup_f64(log2e)));

    svfloat64_t r = svsub_f64_x(pg, x, svmul_f64_x(pg, n, svdup_f64(ln2_hi)));
    r = svsub_f64_x(pg, r, svmul_f64_x(pg, n, svdup_f64(ln2_lo)));

    // Higher order polynomial for FP64
    svfloat64_t p = svdup_f64(1.0 / 5040.0);
    p = svmad_f64_x(pg, p, r, svdup_f64(1.0 / 720.0));
    p = svmad_f64_x(pg, p, r, svdup_f64(1.0 / 120.0));
    p = svmad_f64_x(pg, p, r, svdup_f64(1.0 / 24.0));
    p = svmad_f64_x(pg, p, r, svdup_f64(1.0 / 6.0));
    p = svmad_f64_x(pg, p, r, svdup_f64(0.5));
    p = svmad_f64_x(pg, p, r, svdup_f64(1.0));
    p = svmad_f64_x(pg, p, r, svdup_f64(1.0));

    svint64_t ni = svcvt_s64_f64_x(pg, n);
    ni = svadd_s64_x(pg, ni, svdup_s64(1023));
    ni = svlsl_n_s64_x(pg, ni, 52);
    svfloat64_t scale = svreinterpret_f64_s64(ni);

    return svmul_f64_x(pg, p, scale);
}

// ============================================
// Helper: Sigmoid
// sigmoid(x) = 1 / (1 + exp(-x))
// ============================================

static inline svfloat32_t sigmoid_f32(svbool_t pg, svfloat32_t x) {
    svfloat32_t neg_x = svneg_f32_x(pg, x);
    svfloat32_t exp_neg_x = exp_f32_approx(pg, neg_x);
    svfloat32_t one = svdup_f32(1.0f);
    svfloat32_t denom = svadd_f32_x(pg, one, exp_neg_x);

    // Use reciprocal estimate + Newton-Raphson
    svfloat32_t inv = svrecpe_f32(denom);
    inv = svmul_f32_x(pg, inv, svrecps_f32(denom, inv));
    inv = svmul_f32_x(pg, inv, svrecps_f32(denom, inv));

    return inv;
}

static inline svfloat64_t sigmoid_f64(svbool_t pg, svfloat64_t x) {
    svfloat64_t neg_x = svneg_f64_x(pg, x);
    svfloat64_t exp_neg_x = exp_f64_approx(pg, neg_x);
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t denom = svadd_f64_x(pg, one, exp_neg_x);

    svfloat64_t inv = svrecpe_f64(denom);
    inv = svmul_f64_x(pg, inv, svrecps_f64(denom, inv));
    inv = svmul_f64_x(pg, inv, svrecps_f64(denom, inv));
    inv = svmul_f64_x(pg, inv, svrecps_f64(denom, inv));

    return inv;
}

// ============================================
// Helper: tanh approximation
// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
//         = 1 - 2 / (exp(2x) + 1)
//         = 2 * sigmoid(2x) - 1
// ============================================

static inline svfloat32_t tanh_f32(svbool_t pg, svfloat32_t x) {
    svfloat32_t two_x = svmul_f32_x(pg, x, svdup_f32(2.0f));
    svfloat32_t sig = sigmoid_f32(pg, two_x);
    return svsub_f32_x(pg, svmul_f32_x(pg, sig, svdup_f32(2.0f)), svdup_f32(1.0f));
}

static inline svfloat64_t tanh_f64(svbool_t pg, svfloat64_t x) {
    svfloat64_t two_x = svmul_f64_x(pg, x, svdup_f64(2.0));
    svfloat64_t sig = sigmoid_f64(pg, two_x);
    return svsub_f64_x(pg, svmul_f64_x(pg, sig, svdup_f64(2.0)), svdup_f64(1.0));
}

// ============================================
// Helper: erf approximation (Abramowitz & Stegun)
// erf(x) ≈ 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
// where t = 1/(1 + p*x), p = 0.3275911
// ============================================

static inline svfloat32_t erf_f32(svbool_t pg, svfloat32_t x) {
    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    // Save sign and work with absolute value
    svfloat32_t sign = svdup_f32(1.0f);
    svbool_t neg = svcmplt_f32(pg, x, svdup_f32(0.0f));
    sign = svsel_f32(neg, svdup_f32(-1.0f), sign);
    svfloat32_t ax = svabs_f32_x(pg, x);

    // t = 1 / (1 + p * |x|)
    svfloat32_t denom = svmad_f32_x(pg, ax, svdup_f32(p), svdup_f32(1.0f));
    svfloat32_t t = svrecpe_f32(denom);
    t = svmul_f32_x(pg, t, svrecps_f32(denom, t));
    t = svmul_f32_x(pg, t, svrecps_f32(denom, t));

    // Polynomial: a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    svfloat32_t poly = svdup_f32(a5);
    poly = svmad_f32_x(pg, poly, t, svdup_f32(a4));
    poly = svmad_f32_x(pg, poly, t, svdup_f32(a3));
    poly = svmad_f32_x(pg, poly, t, svdup_f32(a2));
    poly = svmad_f32_x(pg, poly, t, svdup_f32(a1));
    poly = svmul_f32_x(pg, poly, t);

    // exp(-x^2)
    svfloat32_t x2 = svmul_f32_x(pg, ax, ax);
    svfloat32_t exp_neg_x2 = exp_f32_approx(pg, svneg_f32_x(pg, x2));

    // erf = sign * (1 - poly * exp(-x^2))
    svfloat32_t result = svsub_f32_x(pg, svdup_f32(1.0f), svmul_f32_x(pg, poly, exp_neg_x2));
    return svmul_f32_x(pg, sign, result);
}

static inline svfloat64_t erf_f64(svbool_t pg, svfloat64_t x) {
    const double p = 0.3275911;
    const double a1 = 0.254829592;
    const double a2 = -0.284496736;
    const double a3 = 1.421413741;
    const double a4 = -1.453152027;
    const double a5 = 1.061405429;

    svfloat64_t sign = svdup_f64(1.0);
    svbool_t neg = svcmplt_f64(pg, x, svdup_f64(0.0));
    sign = svsel_f64(neg, svdup_f64(-1.0), sign);
    svfloat64_t ax = svabs_f64_x(pg, x);

    svfloat64_t denom = svmad_f64_x(pg, ax, svdup_f64(p), svdup_f64(1.0));
    svfloat64_t t = svrecpe_f64(denom);
    t = svmul_f64_x(pg, t, svrecps_f64(denom, t));
    t = svmul_f64_x(pg, t, svrecps_f64(denom, t));
    t = svmul_f64_x(pg, t, svrecps_f64(denom, t));

    svfloat64_t poly = svdup_f64(a5);
    poly = svmad_f64_x(pg, poly, t, svdup_f64(a4));
    poly = svmad_f64_x(pg, poly, t, svdup_f64(a3));
    poly = svmad_f64_x(pg, poly, t, svdup_f64(a2));
    poly = svmad_f64_x(pg, poly, t, svdup_f64(a1));
    poly = svmul_f64_x(pg, poly, t);

    svfloat64_t x2 = svmul_f64_x(pg, ax, ax);
    svfloat64_t exp_neg_x2 = exp_f64_approx(pg, svneg_f64_x(pg, x2));

    svfloat64_t result = svsub_f64_x(pg, svdup_f64(1.0), svmul_f64_x(pg, poly, exp_neg_x2));
    return svmul_f64_x(pg, sign, result);
}

// ============================================
// GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
// ============================================

void gelu_f32_intrin(const float* input, float* output, size_t n) {
    const float inv_sqrt2 = 0.7071067811865476f;

    size_t vl = svcntw();
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    for (; i + vl <= n; i += vl) {
        svfloat32_t x = svld1_f32(pg, &input[i]);

        // x / sqrt(2)
        svfloat32_t x_scaled = svmul_f32_x(pg, x, svdup_f32(inv_sqrt2));

        // erf(x / sqrt(2))
        svfloat32_t erf_val = erf_f32(pg, x_scaled);

        // 0.5 * x * (1 + erf)
        svfloat32_t one_plus_erf = svadd_f32_x(pg, svdup_f32(1.0f), erf_val);
        svfloat32_t result = svmul_f32_x(pg, x, svmul_f32_x(pg, svdup_f32(0.5f), one_plus_erf));

        svst1_f32(pg, &output[i], result);
    }

    // Handle remainder
    if (i < n) {
        svbool_t pg_rem = svwhilelt_b32(i, n);
        svfloat32_t x = svld1_f32(pg_rem, &input[i]);
        svfloat32_t x_scaled = svmul_f32_x(pg_rem, x, svdup_f32(inv_sqrt2));
        svfloat32_t erf_val = erf_f32(pg_rem, x_scaled);
        svfloat32_t one_plus_erf = svadd_f32_x(pg_rem, svdup_f32(1.0f), erf_val);
        svfloat32_t result = svmul_f32_x(pg_rem, x, svmul_f32_x(pg_rem, svdup_f32(0.5f), one_plus_erf));
        svst1_f32(pg_rem, &output[i], result);
    }
}

void gelu_f64_intrin(const double* input, double* output, size_t n) {
    const double inv_sqrt2 = 0.7071067811865476;

    size_t vl = svcntd();
    size_t i = 0;

    svbool_t pg = svptrue_b64();

    for (; i + vl <= n; i += vl) {
        svfloat64_t x = svld1_f64(pg, &input[i]);
        svfloat64_t x_scaled = svmul_f64_x(pg, x, svdup_f64(inv_sqrt2));
        svfloat64_t erf_val = erf_f64(pg, x_scaled);
        svfloat64_t one_plus_erf = svadd_f64_x(pg, svdup_f64(1.0), erf_val);
        svfloat64_t result = svmul_f64_x(pg, x, svmul_f64_x(pg, svdup_f64(0.5), one_plus_erf));
        svst1_f64(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b64(i, n);
        svfloat64_t x = svld1_f64(pg_rem, &input[i]);
        svfloat64_t x_scaled = svmul_f64_x(pg_rem, x, svdup_f64(inv_sqrt2));
        svfloat64_t erf_val = erf_f64(pg_rem, x_scaled);
        svfloat64_t one_plus_erf = svadd_f64_x(pg_rem, svdup_f64(1.0), erf_val);
        svfloat64_t result = svmul_f64_x(pg_rem, x, svmul_f64_x(pg_rem, svdup_f64(0.5), one_plus_erf));
        svst1_f64(pg_rem, &output[i], result);
    }
}

// ============================================
// GELU tanh approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================

void gelu_tanh_f32_intrin(const float* input, float* output, size_t n) {
    const float sqrt_2_pi = 0.7978845608028654f;
    const float coef = 0.044715f;

    size_t vl = svcntw();
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    for (; i + vl <= n; i += vl) {
        svfloat32_t x = svld1_f32(pg, &input[i]);

        // x^3
        svfloat32_t x2 = svmul_f32_x(pg, x, x);
        svfloat32_t x3 = svmul_f32_x(pg, x2, x);

        // x + 0.044715 * x^3
        svfloat32_t inner = svmad_f32_x(pg, x3, svdup_f32(coef), x);

        // sqrt(2/pi) * (x + 0.044715 * x^3)
        inner = svmul_f32_x(pg, inner, svdup_f32(sqrt_2_pi));

        // tanh(...)
        svfloat32_t tanh_val = tanh_f32(pg, inner);

        // 0.5 * x * (1 + tanh)
        svfloat32_t one_plus_tanh = svadd_f32_x(pg, svdup_f32(1.0f), tanh_val);
        svfloat32_t result = svmul_f32_x(pg, x, svmul_f32_x(pg, svdup_f32(0.5f), one_plus_tanh));

        svst1_f32(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b32(i, n);
        svfloat32_t x = svld1_f32(pg_rem, &input[i]);
        svfloat32_t x2 = svmul_f32_x(pg_rem, x, x);
        svfloat32_t x3 = svmul_f32_x(pg_rem, x2, x);
        svfloat32_t inner = svmad_f32_x(pg_rem, x3, svdup_f32(coef), x);
        inner = svmul_f32_x(pg_rem, inner, svdup_f32(sqrt_2_pi));
        svfloat32_t tanh_val = tanh_f32(pg_rem, inner);
        svfloat32_t one_plus_tanh = svadd_f32_x(pg_rem, svdup_f32(1.0f), tanh_val);
        svfloat32_t result = svmul_f32_x(pg_rem, x, svmul_f32_x(pg_rem, svdup_f32(0.5f), one_plus_tanh));
        svst1_f32(pg_rem, &output[i], result);
    }
}

void gelu_tanh_f64_intrin(const double* input, double* output, size_t n) {
    const double sqrt_2_pi = 0.7978845608028654;
    const double coef = 0.044715;

    size_t vl = svcntd();
    size_t i = 0;

    svbool_t pg = svptrue_b64();

    for (; i + vl <= n; i += vl) {
        svfloat64_t x = svld1_f64(pg, &input[i]);
        svfloat64_t x2 = svmul_f64_x(pg, x, x);
        svfloat64_t x3 = svmul_f64_x(pg, x2, x);
        svfloat64_t inner = svmad_f64_x(pg, x3, svdup_f64(coef), x);
        inner = svmul_f64_x(pg, inner, svdup_f64(sqrt_2_pi));
        svfloat64_t tanh_val = tanh_f64(pg, inner);
        svfloat64_t one_plus_tanh = svadd_f64_x(pg, svdup_f64(1.0), tanh_val);
        svfloat64_t result = svmul_f64_x(pg, x, svmul_f64_x(pg, svdup_f64(0.5), one_plus_tanh));
        svst1_f64(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b64(i, n);
        svfloat64_t x = svld1_f64(pg_rem, &input[i]);
        svfloat64_t x2 = svmul_f64_x(pg_rem, x, x);
        svfloat64_t x3 = svmul_f64_x(pg_rem, x2, x);
        svfloat64_t inner = svmad_f64_x(pg_rem, x3, svdup_f64(coef), x);
        inner = svmul_f64_x(pg_rem, inner, svdup_f64(sqrt_2_pi));
        svfloat64_t tanh_val = tanh_f64(pg_rem, inner);
        svfloat64_t one_plus_tanh = svadd_f64_x(pg_rem, svdup_f64(1.0), tanh_val);
        svfloat64_t result = svmul_f64_x(pg_rem, x, svmul_f64_x(pg_rem, svdup_f64(0.5), one_plus_tanh));
        svst1_f64(pg_rem, &output[i], result);
    }
}

// ============================================
// QuickGELU: x * sigmoid(1.702 * x)
// ============================================

void quickgelu_f32_intrin(const float* input, float* output, size_t n) {
    const float alpha = 1.702f;

    size_t vl = svcntw();
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    for (; i + vl <= n; i += vl) {
        svfloat32_t x = svld1_f32(pg, &input[i]);

        // 1.702 * x
        svfloat32_t scaled = svmul_f32_x(pg, x, svdup_f32(alpha));

        // sigmoid(1.702 * x)
        svfloat32_t sig = sigmoid_f32(pg, scaled);

        // x * sigmoid(1.702 * x)
        svfloat32_t result = svmul_f32_x(pg, x, sig);

        svst1_f32(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b32(i, n);
        svfloat32_t x = svld1_f32(pg_rem, &input[i]);
        svfloat32_t scaled = svmul_f32_x(pg_rem, x, svdup_f32(alpha));
        svfloat32_t sig = sigmoid_f32(pg_rem, scaled);
        svfloat32_t result = svmul_f32_x(pg_rem, x, sig);
        svst1_f32(pg_rem, &output[i], result);
    }
}

void quickgelu_f64_intrin(const double* input, double* output, size_t n) {
    const double alpha = 1.702;

    size_t vl = svcntd();
    size_t i = 0;

    svbool_t pg = svptrue_b64();

    for (; i + vl <= n; i += vl) {
        svfloat64_t x = svld1_f64(pg, &input[i]);
        svfloat64_t scaled = svmul_f64_x(pg, x, svdup_f64(alpha));
        svfloat64_t sig = sigmoid_f64(pg, scaled);
        svfloat64_t result = svmul_f64_x(pg, x, sig);
        svst1_f64(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b64(i, n);
        svfloat64_t x = svld1_f64(pg_rem, &input[i]);
        svfloat64_t scaled = svmul_f64_x(pg_rem, x, svdup_f64(alpha));
        svfloat64_t sig = sigmoid_f64(pg_rem, scaled);
        svfloat64_t result = svmul_f64_x(pg_rem, x, sig);
        svst1_f64(pg_rem, &output[i], result);
    }
}

// ============================================
// SiLU (Swish): x * sigmoid(x)
// ============================================

void silu_f32_intrin(const float* input, float* output, size_t n) {
    size_t vl = svcntw();
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    for (; i + vl <= n; i += vl) {
        svfloat32_t x = svld1_f32(pg, &input[i]);
        svfloat32_t sig = sigmoid_f32(pg, x);
        svfloat32_t result = svmul_f32_x(pg, x, sig);
        svst1_f32(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b32(i, n);
        svfloat32_t x = svld1_f32(pg_rem, &input[i]);
        svfloat32_t sig = sigmoid_f32(pg_rem, x);
        svfloat32_t result = svmul_f32_x(pg_rem, x, sig);
        svst1_f32(pg_rem, &output[i], result);
    }
}

void silu_f64_intrin(const double* input, double* output, size_t n) {
    size_t vl = svcntd();
    size_t i = 0;

    svbool_t pg = svptrue_b64();

    for (; i + vl <= n; i += vl) {
        svfloat64_t x = svld1_f64(pg, &input[i]);
        svfloat64_t sig = sigmoid_f64(pg, x);
        svfloat64_t result = svmul_f64_x(pg, x, sig);
        svst1_f64(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b64(i, n);
        svfloat64_t x = svld1_f64(pg_rem, &input[i]);
        svfloat64_t sig = sigmoid_f64(pg_rem, x);
        svfloat64_t result = svmul_f64_x(pg_rem, x, sig);
        svst1_f64(pg_rem, &output[i], result);
    }
}

// ============================================
// 2x unrolled versions for better performance
// ============================================

void silu_f32_unroll2_intrin(const float* input, float* output, size_t n) {
    size_t vl = svcntw();
    size_t vl2 = vl * 2;
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    for (; i + vl2 <= n; i += vl2) {
        svfloat32_t x0 = svld1_f32(pg, &input[i]);
        svfloat32_t x1 = svld1_f32(pg, &input[i + vl]);

        svfloat32_t sig0 = sigmoid_f32(pg, x0);
        svfloat32_t sig1 = sigmoid_f32(pg, x1);

        svfloat32_t result0 = svmul_f32_x(pg, x0, sig0);
        svfloat32_t result1 = svmul_f32_x(pg, x1, sig1);

        svst1_f32(pg, &output[i], result0);
        svst1_f32(pg, &output[i + vl], result1);
    }

    // Handle remainder
    for (; i < n; i += vl) {
        svbool_t pg_rem = svwhilelt_b32(i, n);
        svfloat32_t x = svld1_f32(pg_rem, &input[i]);
        svfloat32_t sig = sigmoid_f32(pg_rem, x);
        svfloat32_t result = svmul_f32_x(pg_rem, x, sig);
        svst1_f32(pg_rem, &output[i], result);
    }
}

void quickgelu_f32_unroll2_intrin(const float* input, float* output, size_t n) {
    const float alpha = 1.702f;

    size_t vl = svcntw();
    size_t vl2 = vl * 2;
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    for (; i + vl2 <= n; i += vl2) {
        svfloat32_t x0 = svld1_f32(pg, &input[i]);
        svfloat32_t x1 = svld1_f32(pg, &input[i + vl]);

        svfloat32_t scaled0 = svmul_f32_x(pg, x0, svdup_f32(alpha));
        svfloat32_t scaled1 = svmul_f32_x(pg, x1, svdup_f32(alpha));

        svfloat32_t sig0 = sigmoid_f32(pg, scaled0);
        svfloat32_t sig1 = sigmoid_f32(pg, scaled1);

        svfloat32_t result0 = svmul_f32_x(pg, x0, sig0);
        svfloat32_t result1 = svmul_f32_x(pg, x1, sig1);

        svst1_f32(pg, &output[i], result0);
        svst1_f32(pg, &output[i + vl], result1);
    }

    for (; i < n; i += vl) {
        svbool_t pg_rem = svwhilelt_b32(i, n);
        svfloat32_t x = svld1_f32(pg_rem, &input[i]);
        svfloat32_t scaled = svmul_f32_x(pg_rem, x, svdup_f32(alpha));
        svfloat32_t sig = sigmoid_f32(pg_rem, scaled);
        svfloat32_t result = svmul_f32_x(pg_rem, x, sig);
        svst1_f32(pg_rem, &output[i], result);
    }
}

// ============================================
// SwiGLU: x * gate * sigmoid(gate) = x * SiLU(gate)
// Common in LLaMA, Mistral, etc.
// ============================================

void swiglu_f32_intrin(const float* x, const float* gate, float* output, size_t n) {
    size_t vl = svcntw();
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    for (; i + vl <= n; i += vl) {
        svfloat32_t vx = svld1_f32(pg, &x[i]);
        svfloat32_t vgate = svld1_f32(pg, &gate[i]);

        // SiLU(gate) = gate * sigmoid(gate)
        svfloat32_t sig = sigmoid_f32(pg, vgate);
        svfloat32_t silu_gate = svmul_f32_x(pg, vgate, sig);

        // output = x * SiLU(gate)
        svfloat32_t result = svmul_f32_x(pg, vx, silu_gate);

        svst1_f32(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1_f32(pg_rem, &x[i]);
        svfloat32_t vgate = svld1_f32(pg_rem, &gate[i]);
        svfloat32_t sig = sigmoid_f32(pg_rem, vgate);
        svfloat32_t silu_gate = svmul_f32_x(pg_rem, vgate, sig);
        svfloat32_t result = svmul_f32_x(pg_rem, vx, silu_gate);
        svst1_f32(pg_rem, &output[i], result);
    }
}

void swiglu_f64_intrin(const double* x, const double* gate, double* output, size_t n) {
    size_t vl = svcntd();
    size_t i = 0;

    svbool_t pg = svptrue_b64();

    for (; i + vl <= n; i += vl) {
        svfloat64_t vx = svld1_f64(pg, &x[i]);
        svfloat64_t vgate = svld1_f64(pg, &gate[i]);

        svfloat64_t sig = sigmoid_f64(pg, vgate);
        svfloat64_t silu_gate = svmul_f64_x(pg, vgate, sig);
        svfloat64_t result = svmul_f64_x(pg, vx, silu_gate);

        svst1_f64(pg, &output[i], result);
    }

    if (i < n) {
        svbool_t pg_rem = svwhilelt_b64(i, n);
        svfloat64_t vx = svld1_f64(pg_rem, &x[i]);
        svfloat64_t vgate = svld1_f64(pg_rem, &gate[i]);
        svfloat64_t sig = sigmoid_f64(pg_rem, vgate);
        svfloat64_t silu_gate = svmul_f64_x(pg_rem, vgate, sig);
        svfloat64_t result = svmul_f64_x(pg_rem, vx, silu_gate);
        svst1_f64(pg_rem, &output[i], result);
    }
}

// 2x unrolled version for better performance
void swiglu_f32_unroll2_intrin(const float* x, const float* gate, float* output, size_t n) {
    size_t vl = svcntw();
    size_t vl2 = vl * 2;
    size_t i = 0;

    svbool_t pg = svptrue_b32();

    for (; i + vl2 <= n; i += vl2) {
        svfloat32_t vx0 = svld1_f32(pg, &x[i]);
        svfloat32_t vx1 = svld1_f32(pg, &x[i + vl]);
        svfloat32_t vgate0 = svld1_f32(pg, &gate[i]);
        svfloat32_t vgate1 = svld1_f32(pg, &gate[i + vl]);

        svfloat32_t sig0 = sigmoid_f32(pg, vgate0);
        svfloat32_t sig1 = sigmoid_f32(pg, vgate1);

        svfloat32_t silu0 = svmul_f32_x(pg, vgate0, sig0);
        svfloat32_t silu1 = svmul_f32_x(pg, vgate1, sig1);

        svfloat32_t result0 = svmul_f32_x(pg, vx0, silu0);
        svfloat32_t result1 = svmul_f32_x(pg, vx1, silu1);

        svst1_f32(pg, &output[i], result0);
        svst1_f32(pg, &output[i + vl], result1);
    }

    for (; i < n; i += vl) {
        svbool_t pg_rem = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1_f32(pg_rem, &x[i]);
        svfloat32_t vgate = svld1_f32(pg_rem, &gate[i]);
        svfloat32_t sig = sigmoid_f32(pg_rem, vgate);
        svfloat32_t silu_gate = svmul_f32_x(pg_rem, vgate, sig);
        svfloat32_t result = svmul_f32_x(pg_rem, vx, silu_gate);
        svst1_f32(pg_rem, &output[i], result);
    }
}
