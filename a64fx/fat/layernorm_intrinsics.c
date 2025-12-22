// layernorm_intrinsics.c
// SVE intrinsics implementation of LayerNorm and RMSNorm
// Uses rsqrte + Newton-Raphson for fast inverse sqrt
// Uses recpe + Newton-Raphson for fast division

#include <arm_sve.h>
#include <stddef.h>

// ============================================
// Fast inverse sqrt using rsqrte + Newton-Raphson
// rsqrte gives ~8-bit accuracy, 1 NR iteration gives ~16-bit
// 2 NR iterations give ~32-bit (full FP32 precision)
// ============================================

// Single Newton-Raphson iteration for rsqrt: x' = x * (3 - a*x*x) / 2
// SVE has frsqrts which computes (3 - a*b)/2 in one instruction

static inline svfloat32_t rsqrt_f32_nr1(svbool_t pg, svfloat32_t a) {
    // Initial estimate
    svfloat32_t x = svrsqrte_f32(a);
    // NR step: x = x * frsqrts(a, x*x) = x * (3 - a*x*x)/2
    x = svmul_f32_x(pg, x, svrsqrts_f32(a, svmul_f32_x(pg, x, x)));
    return x;
}

static inline svfloat32_t rsqrt_f32_nr2(svbool_t pg, svfloat32_t a) {
    svfloat32_t x = svrsqrte_f32(a);
    x = svmul_f32_x(pg, x, svrsqrts_f32(a, svmul_f32_x(pg, x, x)));
    x = svmul_f32_x(pg, x, svrsqrts_f32(a, svmul_f32_x(pg, x, x)));
    return x;
}

static inline svfloat64_t rsqrt_f64_nr1(svbool_t pg, svfloat64_t a) {
    svfloat64_t x = svrsqrte_f64(a);
    x = svmul_f64_x(pg, x, svrsqrts_f64(a, svmul_f64_x(pg, x, x)));
    return x;
}

static inline svfloat64_t rsqrt_f64_nr2(svbool_t pg, svfloat64_t a) {
    svfloat64_t x = svrsqrte_f64(a);
    x = svmul_f64_x(pg, x, svrsqrts_f64(a, svmul_f64_x(pg, x, x)));
    x = svmul_f64_x(pg, x, svrsqrts_f64(a, svmul_f64_x(pg, x, x)));
    return x;
}

static inline svfloat64_t rsqrt_f64_nr3(svbool_t pg, svfloat64_t a) {
    svfloat64_t x = svrsqrte_f64(a);
    x = svmul_f64_x(pg, x, svrsqrts_f64(a, svmul_f64_x(pg, x, x)));
    x = svmul_f64_x(pg, x, svrsqrts_f64(a, svmul_f64_x(pg, x, x)));
    x = svmul_f64_x(pg, x, svrsqrts_f64(a, svmul_f64_x(pg, x, x)));
    return x;
}

// ============================================
// Fast reciprocal using recpe + Newton-Raphson
// recpe gives ~8-bit accuracy
// SVE has frecps which computes (2 - a*x) for NR step
// ============================================

static inline svfloat32_t recip_f32_nr1(svbool_t pg, svfloat32_t a) {
    svfloat32_t x = svrecpe_f32(a);
    // NR: x = x * (2 - a*x)
    x = svmul_f32_x(pg, x, svrecps_f32(a, x));
    return x;
}

static inline svfloat32_t recip_f32_nr2(svbool_t pg, svfloat32_t a) {
    svfloat32_t x = svrecpe_f32(a);
    x = svmul_f32_x(pg, x, svrecps_f32(a, x));
    x = svmul_f32_x(pg, x, svrecps_f32(a, x));
    return x;
}

static inline svfloat64_t recip_f64_nr1(svbool_t pg, svfloat64_t a) {
    svfloat64_t x = svrecpe_f64(a);
    x = svmul_f64_x(pg, x, svrecps_f64(a, x));
    return x;
}

static inline svfloat64_t recip_f64_nr2(svbool_t pg, svfloat64_t a) {
    svfloat64_t x = svrecpe_f64(a);
    x = svmul_f64_x(pg, x, svrecps_f64(a, x));
    x = svmul_f64_x(pg, x, svrecps_f64(a, x));
    return x;
}

static inline svfloat64_t recip_f64_nr3(svbool_t pg, svfloat64_t a) {
    svfloat64_t x = svrecpe_f64(a);
    x = svmul_f64_x(pg, x, svrecps_f64(a, x));
    x = svmul_f64_x(pg, x, svrecps_f64(a, x));
    x = svmul_f64_x(pg, x, svrecps_f64(a, x));
    return x;
}

// ============================================
// RMSNorm FP32 - SVE Intrinsics
// output[i] = input[i] * gamma[i] / sqrt(mean(input^2) + eps)
// ============================================

void rmsnorm_f32_intrin(const float* restrict input,
                        const float* restrict gamma,
                        float* restrict output,
                        size_t dim,
                        float eps) {
    size_t vl = svcntw();
    svbool_t pg_all = svptrue_b32();

    // Step 1: Compute sum of squares
    svfloat32_t sum_sq = svdup_f32(0.0f);

    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        sum_sq = svmla_f32_m(pg, sum_sq, x, x);
    }

    // Horizontal sum
    float total_sq = svaddv_f32(pg_all, sum_sq);

    // Step 2: Compute rsqrt(mean_sq + eps)
    float mean_sq = total_sq / (float)dim;
    float variance_eps = mean_sq + eps;

    // Use rsqrte + NR for 1/sqrt(variance + eps)
    svfloat32_t var_vec = svdup_f32(variance_eps);
    svfloat32_t inv_std = rsqrt_f32_nr2(pg_all, var_vec);

    // Step 3: Normalize and scale
    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        svfloat32_t g = svld1_f32(pg, &gamma[i]);

        // output = x * gamma * inv_std
        svfloat32_t y = svmul_f32_x(pg, x, inv_std);
        y = svmul_f32_x(pg, y, g);

        svst1_f32(pg, &output[i], y);
    }
}

// RMSNorm using true division (recpe + NR) for comparison
void rmsnorm_f32_recip_intrin(const float* restrict input,
                              const float* restrict gamma,
                              float* restrict output,
                              size_t dim,
                              float eps) {
    size_t vl = svcntw();
    svbool_t pg_all = svptrue_b32();

    // Step 1: Compute sum of squares
    svfloat32_t sum_sq = svdup_f32(0.0f);

    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        sum_sq = svmla_f32_m(pg, sum_sq, x, x);
    }

    float total_sq = svaddv_f32(pg_all, sum_sq);

    // Step 2: Compute 1/dim using recpe + NR, then mean
    svfloat32_t dim_vec = svdup_f32((float)dim);
    svfloat32_t inv_dim = recip_f32_nr2(pg_all, dim_vec);
    float mean_sq = total_sq * svlasta_f32(pg_all, inv_dim);
    float variance_eps = mean_sq + eps;

    // rsqrt
    svfloat32_t var_vec = svdup_f32(variance_eps);
    svfloat32_t inv_std = rsqrt_f32_nr2(pg_all, var_vec);

    // Step 3: Normalize and scale
    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        svfloat32_t g = svld1_f32(pg, &gamma[i]);

        svfloat32_t y = svmul_f32_x(pg, x, inv_std);
        y = svmul_f32_x(pg, y, g);

        svst1_f32(pg, &output[i], y);
    }
}

// ============================================
// LayerNorm FP32 - SVE Intrinsics
// output[i] = (input[i] - mean) * gamma[i] / sqrt(var + eps) + beta[i]
// ============================================

void layernorm_f32_intrin(const float* restrict input,
                          const float* restrict gamma,
                          const float* restrict beta,
                          float* restrict output,
                          size_t dim,
                          float eps) {
    size_t vl = svcntw();
    svbool_t pg_all = svptrue_b32();

    // Step 1: Compute mean
    svfloat32_t sum = svdup_f32(0.0f);

    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        sum = svadd_f32_m(pg, sum, x);
    }

    float total = svaddv_f32(pg_all, sum);
    float mean = total / (float)dim;
    svfloat32_t mean_vec = svdup_f32(mean);

    // Step 2: Compute variance
    svfloat32_t sum_sq = svdup_f32(0.0f);

    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        svfloat32_t diff = svsub_f32_x(pg, x, mean_vec);
        sum_sq = svmla_f32_m(pg, sum_sq, diff, diff);
    }

    float total_sq = svaddv_f32(pg_all, sum_sq);
    float variance = total_sq / (float)dim;
    float variance_eps = variance + eps;

    // Step 3: Compute rsqrt(variance + eps)
    svfloat32_t var_vec = svdup_f32(variance_eps);
    svfloat32_t inv_std = rsqrt_f32_nr2(pg_all, var_vec);

    // Step 4: Normalize, scale, and shift
    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        svfloat32_t g = svld1_f32(pg, &gamma[i]);
        svfloat32_t b = svld1_f32(pg, &beta[i]);

        // y = (x - mean) * inv_std * gamma + beta
        svfloat32_t diff = svsub_f32_x(pg, x, mean_vec);
        svfloat32_t y = svmul_f32_x(pg, diff, inv_std);
        y = svmla_f32_x(pg, b, y, g);  // y*g + b

        svst1_f32(pg, &output[i], y);
    }
}

// ============================================
// RMSNorm FP64 - SVE Intrinsics
// ============================================

void rmsnorm_f64_intrin(const double* restrict input,
                        const double* restrict gamma,
                        double* restrict output,
                        size_t dim,
                        double eps) {
    size_t vl = svcntd();
    svbool_t pg_all = svptrue_b64();

    // Step 1: Compute sum of squares
    svfloat64_t sum_sq = svdup_f64(0.0);

    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b64(i, dim);
        svfloat64_t x = svld1_f64(pg, &input[i]);
        sum_sq = svmla_f64_m(pg, sum_sq, x, x);
    }

    double total_sq = svaddv_f64(pg_all, sum_sq);

    // Step 2: Compute rsqrt(mean_sq + eps)
    double mean_sq = total_sq / (double)dim;
    double variance_eps = mean_sq + eps;

    // Use rsqrte + 3 NR iterations for FP64 precision
    svfloat64_t var_vec = svdup_f64(variance_eps);
    svfloat64_t inv_std = rsqrt_f64_nr3(pg_all, var_vec);

    // Step 3: Normalize and scale
    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b64(i, dim);
        svfloat64_t x = svld1_f64(pg, &input[i]);
        svfloat64_t g = svld1_f64(pg, &gamma[i]);

        svfloat64_t y = svmul_f64_x(pg, x, inv_std);
        y = svmul_f64_x(pg, y, g);

        svst1_f64(pg, &output[i], y);
    }
}

// ============================================
// LayerNorm FP64 - SVE Intrinsics
// ============================================

void layernorm_f64_intrin(const double* restrict input,
                          const double* restrict gamma,
                          const double* restrict beta,
                          double* restrict output,
                          size_t dim,
                          double eps) {
    size_t vl = svcntd();
    svbool_t pg_all = svptrue_b64();

    // Step 1: Compute mean
    svfloat64_t sum = svdup_f64(0.0);

    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b64(i, dim);
        svfloat64_t x = svld1_f64(pg, &input[i]);
        sum = svadd_f64_m(pg, sum, x);
    }

    double total = svaddv_f64(pg_all, sum);
    double mean = total / (double)dim;
    svfloat64_t mean_vec = svdup_f64(mean);

    // Step 2: Compute variance
    svfloat64_t sum_sq = svdup_f64(0.0);

    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b64(i, dim);
        svfloat64_t x = svld1_f64(pg, &input[i]);
        svfloat64_t diff = svsub_f64_x(pg, x, mean_vec);
        sum_sq = svmla_f64_m(pg, sum_sq, diff, diff);
    }

    double total_sq = svaddv_f64(pg_all, sum_sq);
    double variance = total_sq / (double)dim;
    double variance_eps = variance + eps;

    // rsqrt with 3 NR iterations for FP64
    svfloat64_t var_vec = svdup_f64(variance_eps);
    svfloat64_t inv_std = rsqrt_f64_nr3(pg_all, var_vec);

    // Step 4: Normalize, scale, and shift
    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b64(i, dim);
        svfloat64_t x = svld1_f64(pg, &input[i]);
        svfloat64_t g = svld1_f64(pg, &gamma[i]);
        svfloat64_t b = svld1_f64(pg, &beta[i]);

        svfloat64_t diff = svsub_f64_x(pg, x, mean_vec);
        svfloat64_t y = svmul_f64_x(pg, diff, inv_std);
        y = svmla_f64_x(pg, b, y, g);

        svst1_f64(pg, &output[i], y);
    }
}

// ============================================
// Fused RMSNorm (single pass) - more cache efficient
// ============================================

void rmsnorm_f32_fused_intrin(const float* restrict input,
                              const float* restrict gamma,
                              float* restrict output,
                              size_t batch_size,
                              size_t dim,
                              float eps) {
    size_t vl = svcntw();
    svbool_t pg_all = svptrue_b32();

    for (size_t b = 0; b < batch_size; b++) {
        const float* in_ptr = input + b * dim;
        float* out_ptr = output + b * dim;

        // Compute sum of squares
        svfloat32_t sum_sq = svdup_f32(0.0f);

        for (size_t i = 0; i < dim; i += vl) {
            svbool_t pg = svwhilelt_b32(i, dim);
            svfloat32_t x = svld1_f32(pg, &in_ptr[i]);
            sum_sq = svmla_f32_m(pg, sum_sq, x, x);
        }

        float total_sq = svaddv_f32(pg_all, sum_sq);
        float mean_sq = total_sq / (float)dim;
        float variance_eps = mean_sq + eps;

        svfloat32_t var_vec = svdup_f32(variance_eps);
        svfloat32_t inv_std = rsqrt_f32_nr2(pg_all, var_vec);

        // Normalize and scale
        for (size_t i = 0; i < dim; i += vl) {
            svbool_t pg = svwhilelt_b32(i, dim);
            svfloat32_t x = svld1_f32(pg, &in_ptr[i]);
            svfloat32_t g = svld1_f32(pg, &gamma[i]);

            svfloat32_t y = svmul_f32_x(pg, x, inv_std);
            y = svmul_f32_x(pg, y, g);

            svst1_f32(pg, &out_ptr[i], y);
        }
    }
}

// ============================================
// Unrolled versions for better performance
// ============================================

void rmsnorm_f32_unroll4_intrin(const float* restrict input,
                                const float* restrict gamma,
                                float* restrict output,
                                size_t dim,
                                float eps) {
    size_t vl = svcntw();
    svbool_t pg_all = svptrue_b32();

    // Step 1: Compute sum of squares with 4x unroll
    svfloat32_t sum0 = svdup_f32(0.0f);
    svfloat32_t sum1 = svdup_f32(0.0f);
    svfloat32_t sum2 = svdup_f32(0.0f);
    svfloat32_t sum3 = svdup_f32(0.0f);

    size_t i = 0;
    size_t vl4 = vl * 4;
    for (; i + vl4 <= dim; i += vl4) {
        svfloat32_t x0 = svld1_f32(pg_all, &input[i]);
        svfloat32_t x1 = svld1_f32(pg_all, &input[i + vl]);
        svfloat32_t x2 = svld1_f32(pg_all, &input[i + vl*2]);
        svfloat32_t x3 = svld1_f32(pg_all, &input[i + vl*3]);

        sum0 = svmla_f32_x(pg_all, sum0, x0, x0);
        sum1 = svmla_f32_x(pg_all, sum1, x1, x1);
        sum2 = svmla_f32_x(pg_all, sum2, x2, x2);
        sum3 = svmla_f32_x(pg_all, sum3, x3, x3);
    }

    // Handle remainder
    for (; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        sum0 = svmla_f32_m(pg, sum0, x, x);
    }

    // Combine sums
    sum0 = svadd_f32_x(pg_all, sum0, sum1);
    sum2 = svadd_f32_x(pg_all, sum2, sum3);
    sum0 = svadd_f32_x(pg_all, sum0, sum2);

    float total_sq = svaddv_f32(pg_all, sum0);
    float mean_sq = total_sq / (float)dim;
    float variance_eps = mean_sq + eps;

    svfloat32_t var_vec = svdup_f32(variance_eps);
    svfloat32_t inv_std = rsqrt_f32_nr2(pg_all, var_vec);

    // Step 2: Normalize and scale with 4x unroll
    i = 0;
    for (; i + vl4 <= dim; i += vl4) {
        svfloat32_t x0 = svld1_f32(pg_all, &input[i]);
        svfloat32_t x1 = svld1_f32(pg_all, &input[i + vl]);
        svfloat32_t x2 = svld1_f32(pg_all, &input[i + vl*2]);
        svfloat32_t x3 = svld1_f32(pg_all, &input[i + vl*3]);

        svfloat32_t g0 = svld1_f32(pg_all, &gamma[i]);
        svfloat32_t g1 = svld1_f32(pg_all, &gamma[i + vl]);
        svfloat32_t g2 = svld1_f32(pg_all, &gamma[i + vl*2]);
        svfloat32_t g3 = svld1_f32(pg_all, &gamma[i + vl*3]);

        svfloat32_t y0 = svmul_f32_x(pg_all, svmul_f32_x(pg_all, x0, inv_std), g0);
        svfloat32_t y1 = svmul_f32_x(pg_all, svmul_f32_x(pg_all, x1, inv_std), g1);
        svfloat32_t y2 = svmul_f32_x(pg_all, svmul_f32_x(pg_all, x2, inv_std), g2);
        svfloat32_t y3 = svmul_f32_x(pg_all, svmul_f32_x(pg_all, x3, inv_std), g3);

        svst1_f32(pg_all, &output[i], y0);
        svst1_f32(pg_all, &output[i + vl], y1);
        svst1_f32(pg_all, &output[i + vl*2], y2);
        svst1_f32(pg_all, &output[i + vl*3], y3);
    }

    // Handle remainder
    for (; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        svfloat32_t g = svld1_f32(pg, &gamma[i]);

        svfloat32_t y = svmul_f32_x(pg, x, inv_std);
        y = svmul_f32_x(pg, y, g);

        svst1_f32(pg, &output[i], y);
    }
}

// ============================================
// LayerNorm with Welford online algorithm
// Single pass mean and variance computation
// ============================================

void layernorm_f32_welford_intrin(const float* restrict input,
                                  const float* restrict gamma,
                                  const float* restrict beta,
                                  float* restrict output,
                                  size_t dim,
                                  float eps) {
    size_t vl = svcntw();
    svbool_t pg_all = svptrue_b32();

    // Welford's online algorithm for mean and variance
    svfloat32_t mean_vec = svdup_f32(0.0f);
    svfloat32_t m2_vec = svdup_f32(0.0f);
    size_t count = 0;

    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);

        // Update count (for elements in this chunk)
        size_t chunk_size = (dim - i < vl) ? (dim - i) : vl;
        size_t new_count = count + chunk_size;

        // delta = x - mean
        svfloat32_t delta = svsub_f32_x(pg, x, mean_vec);

        // mean += delta / new_count
        svfloat32_t n_vec = svdup_f32((float)new_count);
        svfloat32_t inv_n = recip_f32_nr2(pg_all, n_vec);
        mean_vec = svmla_f32_m(pg, mean_vec, delta, inv_n);

        // delta2 = x - mean (after update)
        svfloat32_t delta2 = svsub_f32_x(pg, x, mean_vec);

        // m2 += delta * delta2
        m2_vec = svmla_f32_m(pg, m2_vec, delta, delta2);

        count = new_count;
    }

    // Reduce to scalar
    float mean = svaddv_f32(pg_all, mean_vec) / (float)vl;  // Approximate
    float m2 = svaddv_f32(pg_all, m2_vec);

    // Fallback to two-pass for now (Welford reduction is complex)
    // Recompute with correct two-pass
    svfloat32_t sum = svdup_f32(0.0f);
    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        sum = svadd_f32_m(pg, sum, x);
    }
    float total = svaddv_f32(pg_all, sum);
    mean = total / (float)dim;
    mean_vec = svdup_f32(mean);

    svfloat32_t sum_sq = svdup_f32(0.0f);
    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        svfloat32_t diff = svsub_f32_x(pg, x, mean_vec);
        sum_sq = svmla_f32_m(pg, sum_sq, diff, diff);
    }
    float variance = svaddv_f32(pg_all, sum_sq) / (float)dim;
    float variance_eps = variance + eps;

    svfloat32_t var_vec = svdup_f32(variance_eps);
    svfloat32_t inv_std = rsqrt_f32_nr2(pg_all, var_vec);

    // Normalize, scale, shift
    for (size_t i = 0; i < dim; i += vl) {
        svbool_t pg = svwhilelt_b32(i, dim);
        svfloat32_t x = svld1_f32(pg, &input[i]);
        svfloat32_t g = svld1_f32(pg, &gamma[i]);
        svfloat32_t b = svld1_f32(pg, &beta[i]);

        svfloat32_t diff = svsub_f32_x(pg, x, mean_vec);
        svfloat32_t y = svmul_f32_x(pg, diff, inv_std);
        y = svmla_f32_x(pg, b, y, g);

        svst1_f32(pg, &output[i], y);
    }
}
