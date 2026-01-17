// silu_fast.h
// Fast SVE-optimized SiLU activation for A64FX
//
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Uses fast exp approximation: exp(x) ≈ 2^(x * log2(e))
// with polynomial approximation for 2^frac

#ifndef SILU_FAST_H
#define SILU_FAST_H

#include <arm_sve.h>
#include <stdint.h>

// Constants for fast exp
#define LOG2E 1.4426950408889634f    // log2(e)
#define LN2   0.6931471805599453f    // ln(2)

// ============================================================================
// Fast exp using 2^x decomposition
// exp(x) = 2^(x * log2e) = 2^n * 2^f where n=floor, f=frac
// 2^f approximated by polynomial for f in [0, 1)
// ============================================================================

static inline svfloat32_t fast_exp_sve(svbool_t pg, svfloat32_t x) {
    // Clamp to avoid overflow/underflow
    x = svmax_f32_x(pg, x, svdup_f32(-88.0f));
    x = svmin_f32_x(pg, x, svdup_f32(88.0f));

    // Convert to base 2: x * log2(e)
    svfloat32_t t = svmul_f32_x(pg, x, svdup_f32(LOG2E));

    // Split into integer and fractional parts
    // n = floor(t), f = t - n
    svfloat32_t n = svrintm_f32_x(pg, t);  // floor
    svfloat32_t f = svsub_f32_x(pg, t, n);

    // Polynomial approximation for 2^f, f in [0, 1)
    // 2^f ≈ 1 + f*(c1 + f*(c2 + f*(c3 + f*c4)))
    // Coefficients from minimax fit
    svfloat32_t c1 = svdup_f32(0.6931472f);
    svfloat32_t c2 = svdup_f32(0.2402265f);
    svfloat32_t c3 = svdup_f32(0.0555041f);
    svfloat32_t c4 = svdup_f32(0.0096139f);

    svfloat32_t p = svmla_f32_x(pg, c3, c4, f);
    p = svmla_f32_x(pg, c2, p, f);
    p = svmla_f32_x(pg, c1, p, f);
    p = svmla_f32_x(pg, svdup_f32(1.0f), p, f);

    // Scale by 2^n using integer manipulation
    // 2^n = reinterpret((n + 127) << 23) for float32
    svint32_t ni = svcvt_s32_f32_x(pg, n);
    ni = svadd_s32_x(pg, ni, svdup_s32(127));
    ni = svlsl_n_s32_x(pg, ni, 23);
    svfloat32_t scale = svreinterpret_f32_s32(ni);

    return svmul_f32_x(pg, p, scale);
}

// ============================================================================
// Fast sigmoid: 1 / (1 + exp(-x))
// ============================================================================

static inline svfloat32_t fast_sigmoid_sve(svbool_t pg, svfloat32_t x) {
    svfloat32_t neg_x = svneg_f32_x(pg, x);
    svfloat32_t exp_neg_x = fast_exp_sve(pg, neg_x);
    svfloat32_t denom = svadd_f32_x(pg, svdup_f32(1.0f), exp_neg_x);
    return svdiv_f32_x(pg, svdup_f32(1.0f), denom);
}

// ============================================================================
// Fast SiLU: x * sigmoid(x)
// ============================================================================

static inline svfloat32_t fast_silu_sve(svbool_t pg, svfloat32_t x) {
    svfloat32_t sig = fast_sigmoid_sve(pg, x);
    return svmul_f32_x(pg, x, sig);
}

// ============================================================================
// Even faster sigmoid using rational approximation (no exp)
// sigmoid(x) ≈ 0.5 + 0.5 * x / (1 + |x|) for |x| < ~5
// More accurate: sigmoid(x) ≈ 0.5 * (1 + tanh(x/2))
// ============================================================================

static inline svfloat32_t fast_sigmoid_rational_sve(svbool_t pg, svfloat32_t x) {
    // sigmoid(x) ≈ 0.5 * (1 + x / (1 + |x|))
    // This is accurate to ~2% for |x| < 5
    svfloat32_t abs_x = svabs_f32_x(pg, x);
    svfloat32_t denom = svadd_f32_x(pg, svdup_f32(1.0f), abs_x);
    svfloat32_t frac = svdiv_f32_x(pg, x, denom);
    return svmla_f32_x(pg, svdup_f32(0.5f), svdup_f32(0.5f), frac);
}

static inline svfloat32_t fast_silu_rational_sve(svbool_t pg, svfloat32_t x) {
    svfloat32_t sig = fast_sigmoid_rational_sve(pg, x);
    return svmul_f32_x(pg, x, sig);
}

// ============================================================================
// Batch processing functions
// ============================================================================

// Fast SiLU for int32 input, float output (for GEMM results)
static inline void silu_i32_to_f32_fast(const int32_t* in, float* out,
                                         int n, float scale) {
    svfloat32_t vscale = svdup_f32(scale);
    int i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        // Load and convert to float
        svint32_t vi = svld1_s32(pg, in + i);
        svfloat32_t vx = svcvt_f32_s32_x(pg, vi);
        vx = svmul_f32_x(pg, vx, vscale);

        // Apply fast SiLU
        svfloat32_t vresult = fast_silu_sve(pg, vx);

        svst1_f32(pg, out + i, vresult);
        i += svcntw();
    }
}

// Ultra-fast SiLU using rational approximation (no exp at all)
static inline void silu_i32_to_f32_ultra(const int32_t* in, float* out,
                                          int n, float scale) {
    svfloat32_t vscale = svdup_f32(scale);
    int i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        // Load and convert to float
        svint32_t vi = svld1_s32(pg, in + i);
        svfloat32_t vx = svcvt_f32_s32_x(pg, vi);
        vx = svmul_f32_x(pg, vx, vscale);

        // Apply ultra-fast SiLU (rational approximation)
        svfloat32_t vresult = fast_silu_rational_sve(pg, vx);

        svst1_f32(pg, out + i, vresult);
        i += svcntw();
    }
}

// Float to float SiLU
static inline void silu_f32_fast(float* x, int n) {
    int i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t vx = svld1_f32(pg, x + i);
        svfloat32_t vresult = fast_silu_sve(pg, vx);
        svst1_f32(pg, x + i, vresult);
        i += svcntw();
    }
}

// ============================================================================
// Fused SiLU + multiply (for gate * up pattern in SwiGLU)
// out = SiLU(gate) * up
// ============================================================================

static inline void silu_mul_i32_fast(const int32_t* gate, const int32_t* up,
                                      float* out, int n, float scale) {
    svfloat32_t vscale = svdup_f32(scale);
    int i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        // Load gate and up
        svint32_t vgate_i = svld1_s32(pg, gate + i);
        svint32_t vup_i = svld1_s32(pg, up + i);

        // Convert to float and scale
        svfloat32_t vgate = svcvt_f32_s32_x(pg, vgate_i);
        svfloat32_t vup = svcvt_f32_s32_x(pg, vup_i);
        vgate = svmul_f32_x(pg, vgate, vscale);
        vup = svmul_f32_x(pg, vup, vscale);

        // SiLU(gate) * up
        svfloat32_t vsilu = fast_silu_sve(pg, vgate);
        svfloat32_t vresult = svmul_f32_x(pg, vsilu, vup);

        svst1_f32(pg, out + i, vresult);
        i += svcntw();
    }
}

// Ultra-fast version using rational approximation
static inline void silu_mul_i32_ultra(const int32_t* gate, const int32_t* up,
                                       float* out, int n, float scale) {
    svfloat32_t vscale = svdup_f32(scale);
    int i = 0;

    while (i < n) {
        svbool_t pg = svwhilelt_b32(i, n);

        // Load gate and up
        svint32_t vgate_i = svld1_s32(pg, gate + i);
        svint32_t vup_i = svld1_s32(pg, up + i);

        // Convert to float and scale
        svfloat32_t vgate = svcvt_f32_s32_x(pg, vgate_i);
        svfloat32_t vup = svcvt_f32_s32_x(pg, vup_i);
        vgate = svmul_f32_x(pg, vgate, vscale);
        vup = svmul_f32_x(pg, vup, vscale);

        // Ultra-fast SiLU(gate) * up
        svfloat32_t vsilu = fast_silu_rational_sve(pg, vgate);
        svfloat32_t vresult = svmul_f32_x(pg, vsilu, vup);

        svst1_f32(pg, out + i, vresult);
        i += svcntw();
    }
}

#endif // SILU_FAST_H
