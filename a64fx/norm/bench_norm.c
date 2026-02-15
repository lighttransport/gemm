#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "norm_sve.h"
#include "sve_math.h"
#include <arm_sve.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Timer ── */
static inline uint64_t rdtsc(void) {
    uint64_t c;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(c));
    return c;
}
static inline uint64_t rdfreq(void) {
    uint64_t f;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}

/* ── Random fill ── */
static void fill_random(float *buf, int n, float lo, float hi, unsigned seed) {
    srand(seed);
    float range = hi - lo;
    for (int i = 0; i < n; i++)
        buf[i] = lo + range * ((float)rand() / RAND_MAX);
}

/* ── FP16 conversion helpers ── */
static void fp32_to_fp16(const float *src, uint16_t *dst, int n) {
    for (int i = 0; i < n; i++) {
        _Float16 h = (_Float16)src[i];
        uint16_t u;
        memcpy(&u, &h, 2);
        dst[i] = u;
    }
}

static float fp16_to_fp32(uint16_t u) {
    _Float16 h;
    memcpy(&h, &u, 2);
    return (float)h;
}


/* ════════════════════════════════════════════════════════════════
 * FRSQRTE + Newton-Raphson precision test
 * ════════════════════════════════════════════════════════════════ */

static void test_rsqrt_precision(void) {
    printf("\n=== FRSQRTE + Newton-Raphson Precision Test ===\n");

    float test_vals[] = { 0.001f, 0.1f, 0.5f, 1.0f, 2.0f, 5.0f,
                          10.0f, 100.0f, 1000.0f, 65504.0f };
    int n = sizeof(test_vals) / sizeof(test_vals[0]);

    svbool_t pg = svptrue_b32();

    printf("  %-12s  %-12s  %-12s  %-12s  %-12s\n",
           "x", "ref", "0-NR", "1-NR", "2-NR");

    for (int i = 0; i < n; i++) {
        float x = test_vals[i];
        double ref = 1.0 / sqrt((double)x);

        svfloat32_t vx = svdup_f32(x);

        /* 0 NR steps */
        svfloat32_t v0 = svrsqrte_f32(vx);
        float buf[16] __attribute__((aligned(256)));
        svst1_f32(pg, buf, v0);
        float r0 = buf[0];

        /* 1 NR step */
        svfloat32_t dx2 = svmul_f32_x(pg, vx, v0);
        svfloat32_t step = svrsqrts_f32(dx2, v0);
        svfloat32_t v1 = svmul_f32_x(pg, v0, step);
        svst1_f32(pg, buf, v1);
        float r1 = buf[0];

        /* 2 NR steps (full precision) — inline expansion */
        svfloat32_t v2 = svrsqrte_f32(vx);
        {
            svfloat32_t t1 = svmul_f32_x(pg, vx, v2);
            v2 = svmul_f32_x(pg, v2, svrsqrts_f32(t1, v2));
            svfloat32_t t2 = svmul_f32_x(pg, vx, v2);
            v2 = svmul_f32_x(pg, v2, svrsqrts_f32(t2, v2));
        }
        svst1_f32(pg, buf, v2);
        float r2 = buf[0];

        double e0 = fabs(((double)r0 - ref) / ref);
        double e1 = fabs(((double)r1 - ref) / ref);
        double e2 = fabs(((double)r2 - ref) / ref);

        printf("  %10.3f  %12.8f  %.2e  %.2e  %.2e %s\n",
               x, ref, e0, e1, e2,
               e2 < 1e-6 ? "OK" : "FAIL");
    }
}


/* ════════════════════════════════════════════════════════════════
 * RMSNorm accuracy test
 * ════════════════════════════════════════════════════════════════ */

static void test_rmsnorm_accuracy(int N) {
    printf("\n=== RMSNorm Accuracy Test: N=%d ===\n", N);

    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_sca = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_sve = (float *)aligned_alloc(256, (size_t)N * sizeof(float));

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    float eps = 1e-5f;

    rmsnorm_ref_f64(x, y_ref, gamma, eps, N);
    rmsnorm_scalar_f32(x, y_sca, gamma, eps, N);
    rmsnorm_fwd_f32(x, y_sve, gamma, eps, N);

    float max_err_sca = 0.0f, max_err_sve = 0.0f;
    for (int i = 0; i < N; i++) {
        float e_sca = fabsf(y_sca[i] - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        float e_sve = fabsf(y_sve[i] - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        if (e_sca > max_err_sca) max_err_sca = e_sca;
        if (e_sve > max_err_sve) max_err_sve = e_sve;
    }

    printf("  scalar max rel err: %.2e %s\n", max_err_sca,
           max_err_sca < 1e-5 ? "OK" : "FAIL");
    printf("  SVE    max rel err: %.2e %s\n", max_err_sve,
           max_err_sve < 1e-5 ? "OK" : "FAIL");

    /* Test gamma=NULL */
    rmsnorm_ref_f64(x, y_ref, NULL, eps, N);
    rmsnorm_fwd_f32(x, y_sve, NULL, eps, N);
    float max_err_null = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = fabsf(y_sve[i] - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        if (e > max_err_null) max_err_null = e;
    }
    printf("  SVE (gamma=NULL): max rel err: %.2e %s\n", max_err_null,
           max_err_null < 1e-5 ? "OK" : "FAIL");

    free(x); free(gamma); free(y_ref); free(y_sca); free(y_sve);
}


/* ════════════════════════════════════════════════════════════════
 * LayerNorm accuracy test
 * ════════════════════════════════════════════════════════════════ */

static void test_layernorm_accuracy(int N) {
    printf("\n=== LayerNorm Accuracy Test: N=%d ===\n", N);

    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta  = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_sca = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_sve = (float *)aligned_alloc(256, (size_t)N * sizeof(float));

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    fill_random(beta, N, -0.5f, 0.5f, 456);
    float eps = 1e-5f;

    layernorm_ref_f64(x, y_ref, gamma, beta, eps, N);
    layernorm_scalar_f32(x, y_sca, gamma, beta, eps, N);
    layernorm_fwd_f32(x, y_sve, gamma, beta, eps, N);

    float max_err_sca = 0.0f, max_err_sve = 0.0f;
    for (int i = 0; i < N; i++) {
        float e_sca = fabsf(y_sca[i] - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        float e_sve = fabsf(y_sve[i] - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        if (e_sca > max_err_sca) max_err_sca = e_sca;
        if (e_sve > max_err_sve) max_err_sve = e_sve;
    }

    /* Note: scalar uses naive var=E[x^2]-E[x]^2 with single accumulator,
     * so catastrophic cancellation at large N is expected. */
    printf("  scalar max rel err: %.2e %s\n", max_err_sca,
           max_err_sca < 2e-3 ? "OK" : "FAIL");
    printf("  SVE    max rel err: %.2e %s\n", max_err_sve,
           max_err_sve < 1e-4 ? "OK" : "FAIL");

    /* Test gamma=NULL, beta=NULL */
    layernorm_ref_f64(x, y_ref, NULL, NULL, eps, N);
    layernorm_fwd_f32(x, y_sve, NULL, NULL, eps, N);
    float max_err_null = 0.0f;
    for (int i = 0; i < N; i++) {
        float e = fabsf(y_sve[i] - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        if (e > max_err_null) max_err_null = e;
    }
    printf("  SVE (no affine): max rel err: %.2e %s\n", max_err_null,
           max_err_null < 1e-4 ? "OK" : "FAIL");

    free(x); free(gamma); free(beta); free(y_ref); free(y_sca); free(y_sve);
}


/* ════════════════════════════════════════════════════════════════
 * FP16 accuracy tests
 * ════════════════════════════════════════════════════════════════ */

static void test_rmsnorm_fp16_accuracy(int N) {
    printf("\n=== RMSNorm FP16 Accuracy Test: N=%d ===\n", N);

    float *x_f32    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    uint16_t *x_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));
    uint16_t *y_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));
    float *gamma    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));

    fill_random(x_f32, N, -2.0f, 2.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    fp32_to_fp16(x_f32, x_f16, N);
    float eps = 1e-5f;

    /* Reconstruct fp32 from fp16 for reference */
    float *x_recon = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    for (int i = 0; i < N; i++)
        x_recon[i] = fp16_to_fp32(x_f16[i]);

    rmsnorm_ref_f64(x_recon, y_ref, gamma, eps, N);
    rmsnorm_fwd_f16(x_f16, y_f16, gamma, eps, N);

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float yf = fp16_to_fp32(y_f16[i]);
        float e = fabsf(yf - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        if (e > max_err) max_err = e;
    }

    printf("  max rel err: %.2e %s\n", max_err,
           max_err < 1e-2 ? "OK" : "FAIL");

    free(x_f32); free(x_f16); free(y_f16); free(gamma); free(y_ref); free(x_recon);
}

static void test_layernorm_fp16_accuracy(int N) {
    printf("\n=== LayerNorm FP16 Accuracy Test: N=%d ===\n", N);

    float *x_f32    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    uint16_t *x_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));
    uint16_t *y_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));
    float *gamma    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));

    fill_random(x_f32, N, -2.0f, 2.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    fill_random(beta, N, -0.5f, 0.5f, 456);
    fp32_to_fp16(x_f32, x_f16, N);
    float eps = 1e-5f;

    float *x_recon = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    for (int i = 0; i < N; i++)
        x_recon[i] = fp16_to_fp32(x_f16[i]);

    layernorm_ref_f64(x_recon, y_ref, gamma, beta, eps, N);
    layernorm_fwd_f16(x_f16, y_f16, gamma, beta, eps, N);

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float yf = fp16_to_fp32(y_f16[i]);
        float e = fabsf(yf - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        if (e > max_err) max_err = e;
    }

    printf("  max rel err: %.2e %s\n", max_err,
           max_err < 1e-2 ? "OK" : "FAIL");

    free(x_f32); free(x_f16); free(y_f16); free(gamma); free(beta); free(y_ref); free(x_recon);
}


/* ════════════════════════════════════════════════════════════════
 * Edge case tests
 * ════════════════════════════════════════════════════════════════ */

static void test_edge_cases(void) {
    printf("\n=== Edge Case Tests ===\n");
    float eps = 1e-5f;

    /* N=1 */
    {
        float x = 3.0f, gamma = 2.0f, beta = 1.0f;
        float y_rms, y_ln;
        rmsnorm_fwd_f32(&x, &y_rms, &gamma, eps, 1);
        layernorm_fwd_f32(&x, &y_ln, &gamma, &beta, eps, 1);
        /* RMSNorm(3) with gamma=2: 2 * 3 / sqrt(9 + eps) ~ 2.0 */
        float rms_ref = 2.0f * 3.0f / sqrtf(9.0f + eps);
        /* LayerNorm(3) with gamma=2,beta=1: (3-3)/std*2+1 = 1.0 */
        float ln_ref = 1.0f;  /* (x-mean)*inv_std = 0, so gamma*0+beta = beta */
        printf("  N=1 RMSNorm: got=%.6f ref=%.6f %s\n", y_rms, rms_ref,
               fabsf(y_rms - rms_ref) < 1e-4 ? "OK" : "FAIL");
        printf("  N=1 LayerNorm: got=%.6f ref=%.6f %s\n", y_ln, ln_ref,
               fabsf(y_ln - ln_ref) < 1e-4 ? "OK" : "FAIL");
    }

    /* All-zero input */
    {
        float x[16] = {0};
        float gamma[16], y[16];
        for (int i = 0; i < 16; i++) gamma[i] = 1.0f;
        rmsnorm_fwd_f32(x, y, gamma, eps, 16);
        int all_near_zero = 1;
        for (int i = 0; i < 16; i++)
            if (fabsf(y[i]) > 1e-3f) all_near_zero = 0;
        printf("  all-zero RMSNorm: %s\n", all_near_zero ? "OK" : "FAIL");
    }

    /* All-same input: LayerNorm should give beta */
    {
        int N = 128;
        float *x = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
        float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
        float *beta = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
        float *y = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
        for (int i = 0; i < N; i++) { x[i] = 5.0f; gamma[i] = 1.0f; beta[i] = 0.5f; }
        layernorm_fwd_f32(x, y, gamma, beta, eps, N);
        int ok = 1;
        for (int i = 0; i < N; i++)
            if (fabsf(y[i] - 0.5f) > 1e-2f) ok = 0;
        printf("  all-same LayerNorm: %s (y[0]=%.6f, expected 0.5)\n",
               ok ? "OK" : "FAIL", y[0]);
        free(x); free(gamma); free(beta); free(y);
    }
}


/* ════════════════════════════════════════════════════════════════
 * RMSNorm backward gradient check
 * ════════════════════════════════════════════════════════════════ */

static void test_rmsnorm_gradient(int N) {
    printf("\n=== RMSNorm Gradient Check: N=%d ===\n", N);

    float *x      = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *dy     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma  = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *dx     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *dgamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y      = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_plus = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_minus= (float *)aligned_alloc(256, (size_t)N * sizeof(float));

    float eps = 1e-5f;
    fill_random(x, N, -2.0f, 2.0f, 42);
    fill_random(dy, N, -1.0f, 1.0f, 123);
    fill_random(gamma, N, 0.5f, 2.0f, 456);
    memset(dgamma, 0, (size_t)N * sizeof(float));

    /* Compute forward to get inv_rms */
    rmsnorm_fwd_f32(x, y, gamma, eps, N);
    double sum_sq = 0.0;
    for (int i = 0; i < N; i++) sum_sq += (double)x[i] * (double)x[i];
    float inv_rms = 1.0f / sqrtf((float)(sum_sq / N) + eps);

    /* Analytical backward */
    rmsnorm_bwd_f32(dy, x, dx, gamma, dgamma, inv_rms, N);

    /* Finite difference check for dx */
    float h = 1e-3f;
    int check_indices[] = { 0, N / 4, N / 2, N - 1 };
    int nchecks = sizeof(check_indices) / sizeof(check_indices[0]);

    printf("  --- dx gradient check ---\n");
    for (int c = 0; c < nchecks; c++) {
        int idx = check_indices[c];
        if (idx >= N) continue;

        float orig = x[idx];
        x[idx] = orig + h;
        rmsnorm_scalar_f32(x, y_plus, gamma, eps, N);
        x[idx] = orig - h;
        rmsnorm_scalar_f32(x, y_minus, gamma, eps, N);
        x[idx] = orig;

        /* loss = sum(dy * y), so dloss/dx[idx] = sum(dy * dy/dx[idx]) */
        float fd = 0.0f;
        for (int i = 0; i < N; i++)
            fd += dy[i] * (y_plus[i] - y_minus[i]) / (2.0f * h);

        float abs_err = fabsf(dx[idx] - fd);
        float rel_err = (fabsf(fd) > 1e-4f) ? abs_err / fabsf(fd) : abs_err;
        int ok = (abs_err < 5e-3f) || (rel_err < 0.01f);

        printf("  dx[%4d]: analytical=%10.6f  fd=%10.6f  abs=%.2e rel=%.2e %s\n",
               idx, dx[idx], fd, abs_err, rel_err, ok ? "OK" : "FAIL");
    }

    /* Finite difference check for dgamma */
    printf("  --- dgamma gradient check ---\n");
    for (int c = 0; c < nchecks; c++) {
        int idx = check_indices[c];
        if (idx >= N) continue;

        float orig = gamma[idx];
        gamma[idx] = orig + h;
        rmsnorm_scalar_f32(x, y_plus, gamma, eps, N);
        gamma[idx] = orig - h;
        rmsnorm_scalar_f32(x, y_minus, gamma, eps, N);
        gamma[idx] = orig;

        float fd = 0.0f;
        for (int i = 0; i < N; i++)
            fd += dy[i] * (y_plus[i] - y_minus[i]) / (2.0f * h);

        float abs_err = fabsf(dgamma[idx] - fd);
        float rel_err = (fabsf(fd) > 1e-4f) ? abs_err / fabsf(fd) : abs_err;
        int ok = (abs_err < 5e-3f) || (rel_err < 0.01f);

        printf("  dgamma[%4d]: analytical=%10.6f  fd=%10.6f  abs=%.2e rel=%.2e %s\n",
               idx, dgamma[idx], fd, abs_err, rel_err, ok ? "OK" : "FAIL");
    }

    free(x); free(dy); free(gamma); free(dx); free(dgamma);
    free(y); free(y_plus); free(y_minus);
}


/* ════════════════════════════════════════════════════════════════
 * LayerNorm backward gradient check
 * ════════════════════════════════════════════════════════════════ */

static void test_layernorm_gradient(int N) {
    printf("\n=== LayerNorm Gradient Check: N=%d ===\n", N);

    float *x      = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *dy     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma  = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta   = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *dx     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *dgamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *dbeta  = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y      = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_plus = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_minus= (float *)aligned_alloc(256, (size_t)N * sizeof(float));

    float eps = 1e-5f;
    fill_random(x, N, -2.0f, 2.0f, 42);
    fill_random(dy, N, -1.0f, 1.0f, 123);
    fill_random(gamma, N, 0.5f, 2.0f, 456);
    fill_random(beta, N, -0.5f, 0.5f, 789);
    memset(dgamma, 0, (size_t)N * sizeof(float));
    memset(dbeta, 0, (size_t)N * sizeof(float));

    /* Compute forward to get mean and inv_std */
    layernorm_fwd_f32(x, y, gamma, beta, eps, N);
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < N; i++) {
        sum += (double)x[i];
        sum_sq += (double)x[i] * (double)x[i];
    }
    float mean = (float)(sum / N);
    float var = (float)(sum_sq / N) - mean * mean;
    float inv_std = 1.0f / sqrtf(var + eps);

    /* Analytical backward */
    layernorm_bwd_f32(dy, x, dx, gamma, dgamma, dbeta, mean, inv_std, N);

    /* Finite difference check for dx */
    float h = 1e-3f;
    int check_indices[] = { 0, N / 4, N / 2, N - 1 };
    int nchecks = sizeof(check_indices) / sizeof(check_indices[0]);

    printf("  --- dx gradient check ---\n");
    for (int c = 0; c < nchecks; c++) {
        int idx = check_indices[c];
        if (idx >= N) continue;

        float orig = x[idx];
        x[idx] = orig + h;
        layernorm_scalar_f32(x, y_plus, gamma, beta, eps, N);
        x[idx] = orig - h;
        layernorm_scalar_f32(x, y_minus, gamma, beta, eps, N);
        x[idx] = orig;

        float fd = 0.0f;
        for (int i = 0; i < N; i++)
            fd += dy[i] * (y_plus[i] - y_minus[i]) / (2.0f * h);

        float abs_err = fabsf(dx[idx] - fd);
        float rel_err = (fabsf(fd) > 1e-4f) ? abs_err / fabsf(fd) : abs_err;
        int ok = (abs_err < 5e-3f) || (rel_err < 0.01f);

        printf("  dx[%4d]: analytical=%10.6f  fd=%10.6f  abs=%.2e rel=%.2e %s\n",
               idx, dx[idx], fd, abs_err, rel_err, ok ? "OK" : "FAIL");
    }

    /* Finite difference check for dgamma */
    printf("  --- dgamma gradient check ---\n");
    for (int c = 0; c < nchecks; c++) {
        int idx = check_indices[c];
        if (idx >= N) continue;

        float orig = gamma[idx];
        gamma[idx] = orig + h;
        layernorm_scalar_f32(x, y_plus, gamma, beta, eps, N);
        gamma[idx] = orig - h;
        layernorm_scalar_f32(x, y_minus, gamma, beta, eps, N);
        gamma[idx] = orig;

        float fd = 0.0f;
        for (int i = 0; i < N; i++)
            fd += dy[i] * (y_plus[i] - y_minus[i]) / (2.0f * h);

        float abs_err = fabsf(dgamma[idx] - fd);
        float rel_err = (fabsf(fd) > 1e-4f) ? abs_err / fabsf(fd) : abs_err;
        int ok = (abs_err < 5e-3f) || (rel_err < 0.01f);

        printf("  dgamma[%4d]: analytical=%10.6f  fd=%10.6f  abs=%.2e rel=%.2e %s\n",
               idx, dgamma[idx], fd, abs_err, rel_err, ok ? "OK" : "FAIL");
    }

    /* Finite difference check for dbeta */
    printf("  --- dbeta gradient check ---\n");
    for (int c = 0; c < nchecks; c++) {
        int idx = check_indices[c];
        if (idx >= N) continue;

        float orig = beta[idx];
        beta[idx] = orig + h;
        layernorm_scalar_f32(x, y_plus, gamma, beta, eps, N);
        beta[idx] = orig - h;
        layernorm_scalar_f32(x, y_minus, gamma, beta, eps, N);
        beta[idx] = orig;

        float fd = 0.0f;
        for (int i = 0; i < N; i++)
            fd += dy[i] * (y_plus[i] - y_minus[i]) / (2.0f * h);

        float abs_err = fabsf(dbeta[idx] - fd);
        float rel_err = (fabsf(fd) > 1e-4f) ? abs_err / fabsf(fd) : abs_err;
        int ok = (abs_err < 5e-3f) || (rel_err < 0.01f);

        printf("  dbeta[%4d]: analytical=%10.6f  fd=%10.6f  abs=%.2e rel=%.2e %s\n",
               idx, dbeta[idx], fd, abs_err, rel_err, ok ? "OK" : "FAIL");
    }

    free(x); free(dy); free(gamma); free(beta);
    free(dx); free(dgamma); free(dbeta);
    free(y); free(y_plus); free(y_minus);
}


/* ════════════════════════════════════════════════════════════════
 * Quantized output accuracy tests
 * ════════════════════════════════════════════════════════════════ */

static void test_rmsnorm_f32_int8_accuracy(int N) {
    printf("\n=== RMSNorm FP32→INT8 Accuracy: N=%d ===\n", N);

    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    int8_t *y_i8 = (int8_t *)aligned_alloc(256, (size_t)N);

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    float eps = 1e-5f;

    /* Reference: double-precision norm */
    rmsnorm_ref_f64(x, y_ref, gamma, eps, N);

    /* Quantized kernel */
    float scale = rmsnorm_fwd_f32_int8(x, y_i8, gamma, eps, N);

    /* Dequantize and compare.
     * For INT8: max abs error should be <= 0.5 * quantization step = 0.5 * scale.
     * We also check that the normalized RMSE is small. */
    float max_abs_err = 0.0f;
    float sum_sq_err = 0.0f, sum_sq_ref = 0.0f;
    int max_abs_val = 0;
    for (int i = 0; i < N; i++) {
        float dequant = (float)y_i8[i] * scale;
        float ref = y_ref[i];
        float ae = fabsf(dequant - ref);
        if (ae > max_abs_err) max_abs_err = ae;
        sum_sq_err += ae * ae;
        sum_sq_ref += ref * ref;
        int av = abs(y_i8[i]);
        if (av > max_abs_val) max_abs_val = av;
    }
    float nrmse = sqrtf(sum_sq_err / (sum_sq_ref + 1e-20f));
    float step = scale;  /* quantization step = absmax/127 */
    int abs_ok = max_abs_err <= step * 1.01f;  /* allow tiny fp rounding */

    printf("  scale=%.6f  max|y_i8|=%d  max_abs_err=%.6f (step=%.6f)  NRMSE=%.4f  %s\n",
           scale, max_abs_val, max_abs_err, step, nrmse,
           (abs_ok && nrmse < 0.01f) ? "OK" : "FAIL");

    free(x); free(gamma); free(y_ref); free(y_i8);
}

static void test_layernorm_f32_int8_accuracy(int N) {
    printf("\n=== LayerNorm FP32→INT8 Accuracy: N=%d ===\n", N);

    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta  = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    int8_t *y_i8 = (int8_t *)aligned_alloc(256, (size_t)N);

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    fill_random(beta, N, -0.5f, 0.5f, 456);
    float eps = 1e-5f;

    layernorm_ref_f64(x, y_ref, gamma, beta, eps, N);
    float scale = layernorm_fwd_f32_int8(x, y_i8, gamma, beta, eps, N);

    float max_abs_err = 0.0f;
    float sum_sq_err = 0.0f, sum_sq_ref = 0.0f;
    for (int i = 0; i < N; i++) {
        float dequant = (float)y_i8[i] * scale;
        float ref = y_ref[i];
        float ae = fabsf(dequant - ref);
        if (ae > max_abs_err) max_abs_err = ae;
        sum_sq_err += ae * ae;
        sum_sq_ref += ref * ref;
    }
    float nrmse = sqrtf(sum_sq_err / (sum_sq_ref + 1e-20f));
    float step = scale;
    int abs_ok = max_abs_err <= step * 1.01f;

    printf("  scale=%.6f  max_abs_err=%.6f (step=%.6f)  NRMSE=%.4f  %s\n",
           scale, max_abs_err, step, nrmse,
           (abs_ok && nrmse < 0.01f) ? "OK" : "FAIL");

    free(x); free(gamma); free(beta); free(y_ref); free(y_i8);
}

static void test_rmsnorm_f16_int8_accuracy(int N) {
    printf("\n=== RMSNorm FP16→INT8 Accuracy: N=%d ===\n", N);

    float *x_f32    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    uint16_t *x_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));
    float *gamma    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    int8_t *y_i8    = (int8_t *)aligned_alloc(256, (size_t)N);

    fill_random(x_f32, N, -2.0f, 2.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    fp32_to_fp16(x_f32, x_f16, N);
    float eps = 1e-5f;

    /* Reconstruct fp32 from fp16 for reference */
    float *x_recon = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    for (int i = 0; i < N; i++)
        x_recon[i] = fp16_to_fp32(x_f16[i]);

    rmsnorm_ref_f64(x_recon, y_ref, gamma, eps, N);
    float scale = rmsnorm_fwd_f16_int8(x_f16, y_i8, gamma, eps, N);

    float max_abs_err = 0.0f;
    float sum_sq_err = 0.0f, sum_sq_ref = 0.0f;
    for (int i = 0; i < N; i++) {
        float dequant = (float)y_i8[i] * scale;
        float ref = y_ref[i];
        float ae = fabsf(dequant - ref);
        if (ae > max_abs_err) max_abs_err = ae;
        sum_sq_err += ae * ae;
        sum_sq_ref += ref * ref;
    }
    float nrmse = sqrtf(sum_sq_err / (sum_sq_ref + 1e-20f));
    float step = scale;
    int abs_ok = max_abs_err <= step * 1.01f;

    printf("  scale=%.6f  max_abs_err=%.6f (step=%.6f)  NRMSE=%.4f  %s\n",
           scale, max_abs_err, step, nrmse,
           (abs_ok && nrmse < 0.01f) ? "OK" : "FAIL");

    free(x_f32); free(x_f16); free(gamma); free(y_ref); free(y_i8); free(x_recon);
}

static void test_rmsnorm_f32_f16_accuracy(int N) {
    printf("\n=== RMSNorm FP32→FP16 Accuracy: N=%d ===\n", N);

    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    uint16_t *y_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    float eps = 1e-5f;

    rmsnorm_ref_f64(x, y_ref, gamma, eps, N);
    rmsnorm_fwd_f32_f16(x, y_f16, gamma, eps, N);

    float max_rel_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float yf = fp16_to_fp32(y_f16[i]);
        float e = fabsf(yf - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        if (e > max_rel_err) max_rel_err = e;
    }

    printf("  max_rel_err=%.2e %s\n", max_rel_err,
           max_rel_err < 5e-4 ? "OK" : "FAIL");

    free(x); free(gamma); free(y_ref); free(y_f16);
}

static void test_layernorm_f32_f16_accuracy(int N) {
    printf("\n=== LayerNorm FP32→FP16 Accuracy: N=%d ===\n", N);

    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta  = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y_ref = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    uint16_t *y_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    fill_random(beta, N, -0.5f, 0.5f, 456);
    float eps = 1e-5f;

    layernorm_ref_f64(x, y_ref, gamma, beta, eps, N);
    layernorm_fwd_f32_f16(x, y_f16, gamma, beta, eps, N);

    float max_rel_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float yf = fp16_to_fp32(y_f16[i]);
        float e = fabsf(yf - y_ref[i]) / (fabsf(y_ref[i]) + 1e-10f);
        if (e > max_rel_err) max_rel_err = e;
    }

    printf("  max_rel_err=%.2e %s\n", max_rel_err,
           max_rel_err < 5e-4 ? "OK" : "FAIL");

    free(x); free(gamma); free(beta); free(y_ref); free(y_f16);
}

static void test_int8_edge_cases(void) {
    printf("\n=== INT8 Edge Cases ===\n");
    float eps = 1e-5f;

    /* All-zero input: absmax=0 → scale=1, y=0 */
    {
        int N = 128;
        float *x = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
        float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
        int8_t *y = (int8_t *)aligned_alloc(256, (size_t)N);
        memset(x, 0, (size_t)N * sizeof(float));
        for (int i = 0; i < N; i++) gamma[i] = 1.0f;

        float scale = rmsnorm_fwd_f32_int8(x, y, gamma, eps, N);
        int all_zero = 1;
        for (int i = 0; i < N; i++)
            if (y[i] != 0) all_zero = 0;
        printf("  all-zero: scale=%.2f all_y_zero=%s %s\n",
               scale, all_zero ? "yes" : "no",
               (all_zero && scale == 1.0f) ? "OK" : "FAIL");

        free(x); free(gamma); free(y);
    }

    /* Verify symmetry: max y_i8 should be +127 or -127 */
    {
        int N = 256;
        float *x = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
        float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
        int8_t *y = (int8_t *)aligned_alloc(256, (size_t)N);
        fill_random(x, N, -3.0f, 3.0f, 42);
        for (int i = 0; i < N; i++) gamma[i] = 1.0f;

        rmsnorm_fwd_f32_int8(x, y, gamma, eps, N);
        int max_abs = 0;
        for (int i = 0; i < N; i++) {
            int av = abs(y[i]);
            if (av > max_abs) max_abs = av;
        }
        printf("  symmetry: max|y_i8|=%d %s\n", max_abs,
               max_abs == 127 ? "OK" : "FAIL");

        free(x); free(gamma); free(y);
    }
}


/* ════════════════════════════════════════════════════════════════
 * Performance benchmark helpers
 * ════════════════════════════════════════════════════════════════ */

static void bench_norm(const char *label, int N, int reps,
                       void (*fn)(const float *, float *, const float *,
                                  const float *, float, int),
                       int has_beta, int npasses) {
    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *y     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta  = has_beta ? (float *)aligned_alloc(256, (size_t)N * sizeof(float)) : NULL;

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    if (beta) fill_random(beta, N, -0.5f, 0.5f, 456);
    float eps = 1e-5f;
    uint64_t freq = rdfreq();

    /* Warmup */
    for (int r = 0; r < 3; r++)
        fn(x, y, gamma, beta, eps, N);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++)
        fn(x, y, gamma, beta, eps, N);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    /* Bytes: pass1 reads x (4N), pass2 reads x+gamma(+beta) and writes y */
    double bytes = (double)N * 4.0 * npasses;
    double bw_b_cy = bytes / cy;
    double elem_cy = (double)N / cy;

    printf("  %-28s N=%5d: %8.1f cy  %6.2f us  %5.1f B/cy  %5.1f elem/cy\n",
           label, N, cy, us, bw_b_cy, elem_cy);

    free(x); free(y); free(gamma); free(beta);
}

/* Wrapper for rmsnorm_fwd_f32 to match layernorm signature */
static void rmsnorm_fwd_f32_wrap(const float *x, float *y, const float *gamma,
                                 const float *beta, float eps, int N) {
    (void)beta;
    rmsnorm_fwd_f32(x, y, gamma, eps, N);
}

/* Wrapper for scalar versions */
static void rmsnorm_scalar_wrap(const float *x, float *y, const float *gamma,
                                const float *beta, float eps, int N) {
    (void)beta;
    rmsnorm_scalar_f32(x, y, gamma, eps, N);
}

static void bench_norm_f16(const char *label, int N, int reps,
                           void (*fn)(const uint16_t *, uint16_t *,
                                      const float *, const float *,
                                      float, int),
                           int has_beta, int npasses) {
    float *x_f32    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    uint16_t *x_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));
    uint16_t *y_f16 = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));
    float *gamma    = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta     = has_beta ? (float *)aligned_alloc(256, (size_t)N * sizeof(float)) : NULL;

    fill_random(x_f32, N, -2.0f, 2.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    if (beta) fill_random(beta, N, -0.5f, 0.5f, 456);
    fp32_to_fp16(x_f32, x_f16, N);
    float eps = 1e-5f;
    uint64_t freq = rdfreq();

    for (int r = 0; r < 3; r++)
        fn(x_f16, y_f16, gamma, beta, eps, N);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++)
        fn(x_f16, y_f16, gamma, beta, eps, N);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double bytes = (double)N * 2.0 * npasses + (double)N * 4.0;  /* fp16 rd/wr + gamma fp32 */
    double bw_b_cy = bytes / cy;
    double elem_cy = (double)N / cy;

    printf("  %-28s N=%5d: %8.1f cy  %6.2f us  %5.1f B/cy  %5.1f elem/cy\n",
           label, N, cy, us, bw_b_cy, elem_cy);

    free(x_f32); free(x_f16); free(y_f16); free(gamma); free(beta);
}

/* Wrapper for rmsnorm_fwd_f16 */
static void rmsnorm_fwd_f16_wrap(const uint16_t *x, uint16_t *y,
                                 const float *gamma, const float *beta,
                                 float eps, int N) {
    (void)beta;
    rmsnorm_fwd_f16(x, y, gamma, eps, N);
}

/* ── INT8 output benchmark ── */
static void bench_norm_int8(const char *label, int N, int reps,
                            float (*fn)(const float *, int8_t *, const float *,
                                        const float *, float, int),
                            int has_beta) {
    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    int8_t *y    = (int8_t *)aligned_alloc(256, (size_t)N);
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta  = has_beta ? (float *)aligned_alloc(256, (size_t)N * sizeof(float)) : NULL;

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    if (beta) fill_random(beta, N, -0.5f, 0.5f, 456);
    float eps = 1e-5f;
    uint64_t freq = rdfreq();

    for (int r = 0; r < 3; r++)
        fn(x, y, gamma, beta, eps, N);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++)
        fn(x, y, gamma, beta, eps, N);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    /* 3 passes over x(4N) + gamma(4N on passes 2,3) + write(N) */
    double bytes_read = (double)N * 4.0 * 3.0 + (double)N * 4.0 * 2.0;
    double bytes_write = (double)N * 1.0;
    double bw_b_cy = (bytes_read + bytes_write) / cy;

    printf("  %-28s N=%5d: %8.1f cy  %6.2f us  %5.1f B/cy\n",
           label, N, cy, us, bw_b_cy);

    free(x); free(y); free(gamma); free(beta);
}

/* Wrappers to match the fn pointer for INT8 benchmarks */
static float rmsnorm_fwd_f32_int8_wrap(const float *x, int8_t *y,
                                        const float *gamma, const float *beta,
                                        float eps, int N) {
    (void)beta;
    return rmsnorm_fwd_f32_int8(x, y, gamma, eps, N);
}

static float layernorm_fwd_f32_int8_wrap(const float *x, int8_t *y,
                                          const float *gamma, const float *beta,
                                          float eps, int N) {
    return layernorm_fwd_f32_int8(x, y, gamma, beta, eps, N);
}

/* ── FP32→FP16 output benchmark ── */
static void bench_norm_f32_f16(const char *label, int N, int reps,
                               void (*fn)(const float *, uint16_t *,
                                          const float *, const float *,
                                          float, int),
                               int has_beta) {
    float *x     = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    uint16_t *y  = (uint16_t *)aligned_alloc(256, (size_t)N * sizeof(uint16_t));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));
    float *beta  = has_beta ? (float *)aligned_alloc(256, (size_t)N * sizeof(float)) : NULL;

    fill_random(x, N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    if (beta) fill_random(beta, N, -0.5f, 0.5f, 456);
    float eps = 1e-5f;
    uint64_t freq = rdfreq();

    for (int r = 0; r < 3; r++)
        fn(x, y, gamma, beta, eps, N);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++)
        fn(x, y, gamma, beta, eps, N);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    /* 2 passes: read x(4N)+gamma(4N), write y_f16(2N) */
    double bytes = (double)N * 4.0 * 2.0 + (double)N * 4.0 + (double)N * 2.0;
    double bw_b_cy = bytes / cy;

    printf("  %-28s N=%5d: %8.1f cy  %6.2f us  %5.1f B/cy\n",
           label, N, cy, us, bw_b_cy);

    free(x); free(y); free(gamma); free(beta);
}

/* Wrappers for FP32→FP16 benchmarks */
static void rmsnorm_fwd_f32_f16_wrap(const float *x, uint16_t *y,
                                      const float *gamma, const float *beta,
                                      float eps, int N) {
    (void)beta;
    rmsnorm_fwd_f32_f16(x, y, gamma, eps, N);
}

static void layernorm_fwd_f32_f16_wrap(const float *x, uint16_t *y,
                                        const float *gamma, const float *beta,
                                        float eps, int N) {
    layernorm_fwd_f32_f16(x, y, gamma, beta, eps, N);
}


#ifdef _OPENMP

/* Wrapper so layernorm_batch_fwd_f32 matches rmsnorm signature */
static void layernorm_batch_fwd_f32_wrap(const float *x, float *y,
                                         const float *gamma,
                                         float eps, int M, int N) {
    /* Use gamma as beta=NULL for benchmarking — just measuring throughput */
    layernorm_batch_fwd_f32(x, y, gamma, NULL, eps, M, N);
}

static void bench_batch(const char *label, int N, int M, int reps,
                        void (*fn)(const float *, float *, const float *,
                                   float, int, int)) {
    float *x     = (float *)aligned_alloc(256, (size_t)M * N * sizeof(float));
    float *y     = (float *)aligned_alloc(256, (size_t)M * N * sizeof(float));
    float *gamma = (float *)aligned_alloc(256, (size_t)N * sizeof(float));

    fill_random(x, M * N, -3.0f, 3.0f, 42);
    fill_random(gamma, N, 0.5f, 2.0f, 123);
    float eps = 1e-5f;
    uint64_t freq = rdfreq();

    for (int r = 0; r < 3; r++)
        fn(x, y, gamma, eps, M, N);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++)
        fn(x, y, gamma, eps, M, N);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double rows_per_s = (double)M / (us / 1e6);

    int nthreads = 1;
    #pragma omp parallel
    { nthreads = omp_get_num_threads(); }

    printf("  %-28s N=%5d M=%5d: %10.1f cy  %8.1f us  %8.0f rows/s  (%d threads)\n",
           label, N, M, cy, us, rows_per_s, nthreads);

    free(x); free(y); free(gamma);
}
#endif


/* ════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    printf("SVE LayerNorm/RMSNorm Benchmark for A64FX\n");
    printf("VL = %d floats (%d bits)\n", (int)svcntw(), (int)svcntw() * 32);
    printf("Timer freq = %lu Hz\n", rdfreq());

    /* ── FRSQRTE precision ── */
    test_rsqrt_precision();

    /* ── Accuracy tests ── */
    int test_sizes[] = { 15, 16, 127, 128, 256, 768, 1024, 4096, 8192 };
    int ntests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < ntests; t++)
        test_rmsnorm_accuracy(test_sizes[t]);

    for (int t = 0; t < ntests; t++)
        test_layernorm_accuracy(test_sizes[t]);

    /* FP16 accuracy */
    int fp16_sizes[] = { 128, 256, 1024, 4096 };
    int nfp16 = sizeof(fp16_sizes) / sizeof(fp16_sizes[0]);
    for (int t = 0; t < nfp16; t++) {
        test_rmsnorm_fp16_accuracy(fp16_sizes[t]);
        test_layernorm_fp16_accuracy(fp16_sizes[t]);
    }

    /* Edge cases */
    test_edge_cases();

    /* ── Quantized output accuracy ── */
    printf("\n\n════ QUANTIZED OUTPUT TESTS ════\n");

    int quant_sizes[] = { 128, 256, 768, 1024, 4096, 8192 };
    int nquant = sizeof(quant_sizes) / sizeof(quant_sizes[0]);

    for (int t = 0; t < nquant; t++)
        test_rmsnorm_f32_int8_accuracy(quant_sizes[t]);
    for (int t = 0; t < nquant; t++)
        test_layernorm_f32_int8_accuracy(quant_sizes[t]);

    /* FP16→INT8 */
    int f16i8_sizes[] = { 128, 256, 1024, 4096 };
    int nf16i8 = sizeof(f16i8_sizes) / sizeof(f16i8_sizes[0]);
    for (int t = 0; t < nf16i8; t++)
        test_rmsnorm_f16_int8_accuracy(f16i8_sizes[t]);

    /* FP32→FP16 */
    for (int t = 0; t < nquant; t++)
        test_rmsnorm_f32_f16_accuracy(quant_sizes[t]);
    for (int t = 0; t < nquant; t++)
        test_layernorm_f32_f16_accuracy(quant_sizes[t]);

    /* INT8 edge cases */
    test_int8_edge_cases();

    /* ── Gradient checks ── */
    test_rmsnorm_gradient(256);
    test_rmsnorm_gradient(1024);
    test_layernorm_gradient(256);
    test_layernorm_gradient(1024);

    /* ── Performance benchmarks ── */
    printf("\n=== Performance Benchmarks (single core) ===\n");

    int bench_sizes[] = { 256, 512, 768, 1024, 2048, 4096, 5120, 8192 };
    int nbench = sizeof(bench_sizes) / sizeof(bench_sizes[0]);

    printf("\n--- RMSNorm Forward ---\n");
    for (int b = 0; b < nbench; b++) {
        int N = bench_sizes[b];
        int reps = (N >= 4096) ? 5000 : 10000;
        bench_norm("scalar", N, reps, rmsnorm_scalar_wrap, 0, 3);
        bench_norm("SVE f32", N, reps, rmsnorm_fwd_f32_wrap, 0, 3);
        bench_norm_f16("SVE f16", N, reps, rmsnorm_fwd_f16_wrap, 0, 3);
        printf("\n");
    }

    printf("\n--- LayerNorm Forward ---\n");
    for (int b = 0; b < nbench; b++) {
        int N = bench_sizes[b];
        int reps = (N >= 4096) ? 5000 : 10000;
        bench_norm("scalar", N, reps, layernorm_scalar_f32, 1, 4);
        bench_norm("SVE f32", N, reps, layernorm_fwd_f32, 1, 4);
        bench_norm_f16("SVE f16", N, reps, layernorm_fwd_f16, 1, 4);
        printf("\n");
    }

    printf("\n--- RMSNorm Fused Output Variants ---\n");
    for (int b = 0; b < nbench; b++) {
        int N = bench_sizes[b];
        int reps = (N >= 4096) ? 5000 : 10000;
        bench_norm("SVE f32→f32", N, reps, rmsnorm_fwd_f32_wrap, 0, 3);
        bench_norm_f32_f16("SVE f32→f16", N, reps, rmsnorm_fwd_f32_f16_wrap, 0);
        bench_norm_int8("SVE f32→int8", N, reps, rmsnorm_fwd_f32_int8_wrap, 0);
        printf("\n");
    }

    printf("\n--- LayerNorm Fused Output Variants ---\n");
    for (int b = 0; b < nbench; b++) {
        int N = bench_sizes[b];
        int reps = (N >= 4096) ? 5000 : 10000;
        bench_norm("SVE f32→f32", N, reps, layernorm_fwd_f32, 1, 4);
        bench_norm_f32_f16("SVE f32→f16", N, reps, layernorm_fwd_f32_f16_wrap, 1);
        bench_norm_int8("SVE f32→int8", N, reps, layernorm_fwd_f32_int8_wrap, 1);
        printf("\n");
    }

    /* ── Batch + OMP scaling ── */
#ifdef _OPENMP
    printf("\n=== Batch OMP Scaling ===\n");
    int batch_N[] = { 1024, 4096 };
    int batch_M[] = { 128, 1024 };
    for (int ni = 0; ni < 2; ni++) {
        for (int mi = 0; mi < 2; mi++) {
            int N = batch_N[ni], M = batch_M[mi];
            int reps = (M >= 1024) ? 5 : 20;
            bench_batch("RMSNorm batch", N, M, reps, rmsnorm_batch_fwd_f32);
            bench_batch("LayerNorm batch", N, M, reps, layernorm_batch_fwd_f32_wrap);
        }
    }
#endif

    printf("\nDone.\n");
    return 0;
}
