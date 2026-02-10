#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdint.h>
#include <sched.h>

#include "cross_entropy_sve.h"
#include "sve_math.h"
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

/* ── Random logits ── */
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
 * Accuracy tests
 * ════════════════════════════════════════════════════════════════ */

static void test_accuracy(int V) {
    printf("\n=== Accuracy Test: V=%d ===\n", V);

    float *logits = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    fill_random(logits, V, -10.0f, 10.0f, 42);

    int targets[] = { 0, V / 2, V - 1, 7 };
    int ntargets = sizeof(targets) / sizeof(targets[0]);
    if (targets[3] >= V) targets[3] = V / 3;

    for (int t = 0; t < ntargets; t++) {
        int tgt = targets[t];
        double ref   = cross_entropy_ref_f64(logits, tgt, V);
        float  scalar = cross_entropy_scalar_f32(logits, tgt, V);
        float  sve   = cross_entropy_fwd_f32(logits, tgt, V);

        double err_scalar = fabs((double)scalar - ref) / fabs(ref);
        double err_sve    = fabs((double)sve    - ref) / fabs(ref);

        printf("  target=%6d: ref=%.10f  scalar=%.6f (err=%.2e)  SVE=%.6f (err=%.2e) %s\n",
               tgt, ref, scalar, err_scalar, sve, err_sve,
               err_sve < 0.01 ? "OK" : "FAIL");
    }

    /* Edge: uniform logits → loss = log(V) */
    for (int i = 0; i < V; i++) logits[i] = 1.0f;
    float uniform_loss = cross_entropy_fwd_f32(logits, 0, V);
    float expected = logf((float)V);
    printf("  uniform:    SVE=%.6f  expected=log(%d)=%.6f  err=%.2e %s\n",
           uniform_loss, V, expected,
           fabsf(uniform_loss - expected) / fabsf(expected),
           fabsf(uniform_loss - expected) / fabsf(expected) < 0.01 ? "OK" : "FAIL");

    /* Edge: one-hot logits (target=0 has 100, rest=-100) → loss ≈ 0 */
    for (int i = 0; i < V; i++) logits[i] = -100.0f;
    logits[0] = 100.0f;
    float onehot_loss = cross_entropy_fwd_f32(logits, 0, V);
    printf("  one-hot:    SVE=%.6f  expected≈0.0  %s\n",
           onehot_loss, fabsf(onehot_loss) < 0.01f ? "OK" : "FAIL");

    free(logits);
}

/* ════════════════════════════════════════════════════════════════
 * Blocked forward accuracy test
 * ════════════════════════════════════════════════════════════════ */

static void test_blocked_accuracy(int V) {
    printf("\n=== Blocked Forward Accuracy Test: V=%d ===\n", V);

    float *logits = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    fill_random(logits, V, -10.0f, 10.0f, 42);

    int targets[] = { 0, V / 2, V - 1, 7 };
    int ntargets = sizeof(targets) / sizeof(targets[0]);
    if (targets[3] >= V) targets[3] = V / 3;

    for (int t = 0; t < ntargets; t++) {
        int tgt = targets[t];
        double ref      = cross_entropy_ref_f64(logits, tgt, V);
        float  twopass  = cross_entropy_fwd_f32(logits, tgt, V);
        float  blocked  = cross_entropy_fwd_blocked_f32(logits, tgt, V);

        double err_2p = fabs((double)twopass - ref) / fabs(ref);
        double err_bl = fabs((double)blocked - ref) / fabs(ref);

        printf("  target=%6d: ref=%.10f  2pass=%.6f (err=%.2e)  blocked=%.6f (err=%.2e) %s\n",
               tgt, ref, twopass, err_2p, blocked, err_bl,
               err_bl < 0.01 ? "OK" : "FAIL");
    }

    free(logits);
}

/* ════════════════════════════════════════════════════════════════
 * FP16 accuracy test
 * ════════════════════════════════════════════════════════════════ */

static void test_fp16_accuracy(int V) {
    printf("\n=== FP16 Accuracy Test: V=%d ===\n", V);

    float *logits_f32 = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    uint16_t *logits_f16 = (uint16_t *)aligned_alloc(256, (size_t)V * sizeof(uint16_t));

    /* Use smaller range for fp16 (max ~65504) */
    fill_random(logits_f32, V, -5.0f, 5.0f, 123);
    fp32_to_fp16(logits_f32, logits_f16, V);

    /* Reconstruct fp32 from fp16 for reference */
    float *logits_recon = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    for (int i = 0; i < V; i++)
        logits_recon[i] = fp16_to_fp32(logits_f16[i]);

    int tgt = V / 2;
    double ref       = cross_entropy_ref_f64(logits_recon, tgt, V);
    float  sve_f16   = cross_entropy_fwd_f16(logits_f16, tgt, V);
    float  sve_f32   = cross_entropy_fwd_f32(logits_recon, tgt, V);

    printf("  target=%d:\n", tgt);
    printf("    ref(f64)     = %.10f\n", ref);
    printf("    SVE(fp32)    = %.6f (err=%.2e)\n", sve_f32,
           fabs((double)sve_f32 - ref) / fabs(ref));
    printf("    SVE(fp16in)  = %.6f (err=%.2e)\n", sve_f16,
           fabs((double)sve_f16 - ref) / fabs(ref));

    free(logits_f32);
    free(logits_f16);
    free(logits_recon);
}

/* ════════════════════════════════════════════════════════════════
 * Blocked FP16 accuracy test
 * ════════════════════════════════════════════════════════════════ */

static void test_blocked_fp16_accuracy(int V) {
    printf("\n=== Blocked FP16 Accuracy Test: V=%d ===\n", V);

    float *logits_f32 = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    uint16_t *logits_f16 = (uint16_t *)aligned_alloc(256, (size_t)V * sizeof(uint16_t));

    fill_random(logits_f32, V, -5.0f, 5.0f, 123);
    fp32_to_fp16(logits_f32, logits_f16, V);

    float *logits_recon = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    for (int i = 0; i < V; i++)
        logits_recon[i] = fp16_to_fp32(logits_f16[i]);

    int targets[] = { 0, V / 2, V - 1 };
    int ntargets = sizeof(targets) / sizeof(targets[0]);

    for (int t = 0; t < ntargets; t++) {
        int tgt = targets[t];
        double ref         = cross_entropy_ref_f64(logits_recon, tgt, V);
        float  twopass_f16 = cross_entropy_fwd_f16(logits_f16, tgt, V);
        float  blocked_f16 = cross_entropy_fwd_blocked_f16(logits_f16, tgt, V);

        double err_2p = fabs((double)twopass_f16 - ref) / fabs(ref);
        double err_bl = fabs((double)blocked_f16 - ref) / fabs(ref);

        printf("  target=%6d: ref=%.10f  2pass=%.6f (err=%.2e)  blocked=%.6f (err=%.2e) %s\n",
               tgt, ref, twopass_f16, err_2p, blocked_f16, err_bl,
               err_bl < 0.01 ? "OK" : "FAIL");
    }

    free(logits_f32);
    free(logits_f16);
    free(logits_recon);
}

/* ════════════════════════════════════════════════════════════════
 * FP16 gradient check (finite difference)
 * ════════════════════════════════════════════════════════════════ */

static void test_gradient_f16(int V, const char *label,
                               float (*fwd_bwd)(const uint16_t *, int, int, float *)) {
    printf("\n=== FP16 Gradient Check (%s): V=%d ===\n", label, V);

    float *logits_f32 = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    uint16_t *logits_f16 = (uint16_t *)aligned_alloc(256, (size_t)V * sizeof(uint16_t));
    float *grad = (float *)aligned_alloc(256, (size_t)V * sizeof(float));

    fill_random(logits_f32, V, -5.0f, 5.0f, 77);
    fp32_to_fp16(logits_f32, logits_f16, V);

    /* Reconstruct fp32 from fp16 for finite difference reference */
    float *logits_recon = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    for (int i = 0; i < V; i++)
        logits_recon[i] = fp16_to_fp32(logits_f16[i]);

    int tgt = V / 3;

    float loss = fwd_bwd(logits_f16, tgt, V, grad);
    printf("  loss = %.6f\n", loss);

    /* Check a few gradient elements via finite difference on reconstructed fp32 */
    float eps = 1e-3f;
    int check_indices[] = { 0, tgt, V / 2, V - 1 };
    int nchecks = sizeof(check_indices) / sizeof(check_indices[0]);

    float max_rel_err = 0.0f;
    for (int c = 0; c < nchecks; c++) {
        int idx = check_indices[c];
        if (idx >= V) continue;

        float orig = logits_recon[idx];
        logits_recon[idx] = orig + eps;
        float lp = cross_entropy_scalar_f32(logits_recon, tgt, V);
        logits_recon[idx] = orig - eps;
        float lm = cross_entropy_scalar_f32(logits_recon, tgt, V);
        logits_recon[idx] = orig;

        float fd = (lp - lm) / (2.0f * eps);
        float abs_err = fabsf(grad[idx] - fd);
        float rel_err = (fabsf(fd) > 1e-4f) ?
            abs_err / fabsf(fd) : abs_err;
        int ok = (abs_err < 5e-4f) || (rel_err < 0.01f);

        printf("  grad[%6d]: analytical=%10.6f  fd=%10.6f  abs=%.2e rel=%.2e %s\n",
               idx, grad[idx], fd, abs_err, rel_err, ok ? "OK" : "FAIL");
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    double gsum = 0.0;
    for (int i = 0; i < V; i++) gsum += (double)grad[i];
    printf("  sum(grad) = %.6e (should be ~0) %s\n", gsum,
           fabs(gsum) < 0.01 ? "OK" : "WARN");

    free(logits_f32);
    free(logits_f16);
    free(logits_recon);
    free(grad);
}

/* ════════════════════════════════════════════════════════════════
 * Gradient check (finite difference)
 * ════════════════════════════════════════════════════════════════ */

static void test_gradient(int V, const char *label,
                          float (*fwd_bwd)(const float *, int, int, float *)) {
    printf("\n=== Gradient Check (%s): V=%d ===\n", label, V);

    float *logits = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    float *grad   = (float *)aligned_alloc(256, (size_t)V * sizeof(float));

    fill_random(logits, V, -5.0f, 5.0f, 77);
    int tgt = V / 3;

    float loss = fwd_bwd(logits, tgt, V, grad);
    printf("  loss = %.6f\n", loss);

    /* Check a few gradient elements via finite difference */
    float eps = 1e-3f;
    int check_indices[] = { 0, tgt, V / 2, V - 1 };
    int nchecks = sizeof(check_indices) / sizeof(check_indices[0]);

    float max_rel_err = 0.0f;
    for (int c = 0; c < nchecks; c++) {
        int idx = check_indices[c];
        if (idx >= V) continue;

        float orig = logits[idx];
        logits[idx] = orig + eps;
        float lp = cross_entropy_scalar_f32(logits, tgt, V);
        logits[idx] = orig - eps;
        float lm = cross_entropy_scalar_f32(logits, tgt, V);
        logits[idx] = orig;

        float fd = (lp - lm) / (2.0f * eps);
        float abs_err = fabsf(grad[idx] - fd);
        float rel_err = (fabsf(fd) > 1e-4f) ?
            abs_err / fabsf(fd) : abs_err;
        int ok = (abs_err < 5e-4f) || (rel_err < 0.01f);

        printf("  grad[%6d]: analytical=%10.6f  fd=%10.6f  abs=%.2e rel=%.2e %s\n",
               idx, grad[idx], fd, abs_err, rel_err, ok ? "OK" : "FAIL");
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    /* Check grad sums to ~0 (softmax sums to 1, minus 1 at target = 0) */
    double gsum = 0.0;
    for (int i = 0; i < V; i++) gsum += (double)grad[i];
    printf("  sum(grad) = %.6e (should be ~0) %s\n", gsum,
           fabs(gsum) < 0.01 ? "OK" : "WARN");

    free(logits);
    free(grad);
}

/* ════════════════════════════════════════════════════════════════
 * SVE log accuracy test
 * ════════════════════════════════════════════════════════════════ */

static void test_log_accuracy(void) {
    printf("\n=== SVE log2/log Accuracy Test ===\n");
    #include <arm_sve.h>

    float test_vals[] = { 0.001f, 0.1f, 0.5f, 1.0f, 1.5f, 2.0f,
                          3.14159f, 10.0f, 100.0f, 65504.0f };
    int n = sizeof(test_vals) / sizeof(test_vals[0]);

    float max_log2_err = 0.0f, max_ln_err = 0.0f;

    for (int i = 0; i < n; i++) {
        float x = test_vals[i];

        svbool_t pg = svptrue_b32();
        svfloat32_t vx = svdup_f32(x);
        svfloat32_t vlog2 = sve_log2_f32(pg, vx);
        svfloat32_t vln   = sve_log_f32(pg, vx);

        float buf[16] __attribute__((aligned(256)));
        svst1_f32(pg, buf, vlog2);
        float sve_log2_val = buf[0];

        svst1_f32(pg, buf, vln);
        float sve_ln_val = buf[0];

        float ref_log2 = log2f(x);
        float ref_ln   = logf(x);

        float err_log2 = (fabsf(ref_log2) > 1e-7f) ?
            fabsf(sve_log2_val - ref_log2) / fabsf(ref_log2) :
            fabsf(sve_log2_val - ref_log2);
        float err_ln = (fabsf(ref_ln) > 1e-7f) ?
            fabsf(sve_ln_val - ref_ln) / fabsf(ref_ln) :
            fabsf(sve_ln_val - ref_ln);

        if (err_log2 > max_log2_err) max_log2_err = err_log2;
        if (err_ln > max_ln_err) max_ln_err = err_ln;

        printf("  x=%10.4f: log2: SVE=%.6f ref=%.6f (err=%.2e)  "
               "ln: SVE=%.6f ref=%.6f (err=%.2e)\n",
               x, sve_log2_val, ref_log2, err_log2,
               sve_ln_val, ref_ln, err_ln);
    }
    printf("  Max relative error: log2=%.2e  ln=%.2e\n",
           max_log2_err, max_ln_err);
}

/* ════════════════════════════════════════════════════════════════
 * Performance benchmark helpers
 * ════════════════════════════════════════════════════════════════ */

static void bench_single(const char *label, int V, int reps,
                         float (*fn)(const float *, int, int),
                         int npasses) {
    float *logits = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    fill_random(logits, V, -10.0f, 10.0f, 42);
    int target = V / 2;
    uint64_t freq = rdfreq();

    /* Warmup */
    volatile float dummy = 0;
    for (int r = 0; r < 3; r++)
        dummy += fn(logits, target, V);

    /* Timed */
    uint64_t t0 = rdtsc();
    float loss = 0;
    for (int r = 0; r < reps; r++)
        loss += fn(logits, target, V);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double tps = 1e6 / us;
    double bytes = (double)V * 4.0 * npasses;
    double bw_b_cy = bytes / cy;

    printf("  %-26s V=%6d: %8.1f cy  %6.1f us  %6.0f tok/s  %5.2f B/cy  loss=%.4f\n",
           label, V, cy, us, tps, bw_b_cy, loss / reps);

    free(logits);
}

static void bench_fwd_bwd(const char *label, int V, int reps,
                          float (*fn)(const float *, int, int, float *),
                          int npasses) {
    float *logits = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    float *grad   = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    fill_random(logits, V, -10.0f, 10.0f, 42);
    int target = V / 2;
    uint64_t freq = rdfreq();

    volatile float dummy = 0;
    for (int r = 0; r < 3; r++)
        dummy += fn(logits, target, V, grad);

    uint64_t t0 = rdtsc();
    float loss = 0;
    for (int r = 0; r < reps; r++)
        loss += fn(logits, target, V, grad);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double tps = 1e6 / us;
    /* For blocked: 2 reads + 1 write = 3 passes equivalent;
     * for 3-pass: 3 reads + 1 write = 4 passes.
     * Report raw B/cy based on npasses provided. */
    double bytes = (double)V * 4.0 * npasses;
    double bw_b_cy = bytes / cy;

    printf("  %-26s V=%6d: %8.1f cy  %6.1f us  %6.0f tok/s  %5.2f B/cy  loss=%.4f\n",
           label, V, cy, us, tps, bw_b_cy, loss / reps);

    free(logits);
    free(grad);
}

static void bench_scalar(const char *label, int V, int reps) {
    float *logits = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    fill_random(logits, V, -10.0f, 10.0f, 42);
    int target = V / 2;
    uint64_t freq = rdfreq();

    volatile float dummy = 0;
    for (int r = 0; r < 3; r++)
        dummy += cross_entropy_scalar_f32(logits, target, V);

    uint64_t t0 = rdtsc();
    float loss = 0;
    for (int r = 0; r < reps; r++)
        loss += cross_entropy_scalar_f32(logits, target, V);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double tps = 1e6 / us;

    printf("  %-26s V=%6d: %8.1f cy  %6.1f us  %6.0f tok/s  loss=%.4f\n",
           label, V, cy, us, tps, loss / reps);

    free(logits);
}

static void bench_fp16(const char *label, int V, int reps) {
    float *logits_f32 = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    uint16_t *logits_f16 = (uint16_t *)aligned_alloc(256, (size_t)V * sizeof(uint16_t));
    fill_random(logits_f32, V, -5.0f, 5.0f, 42);
    fp32_to_fp16(logits_f32, logits_f16, V);
    int target = V / 2;
    uint64_t freq = rdfreq();

    volatile float dummy = 0;
    for (int r = 0; r < 3; r++)
        dummy += cross_entropy_fwd_f16(logits_f16, target, V);

    uint64_t t0 = rdtsc();
    float loss = 0;
    for (int r = 0; r < reps; r++)
        loss += cross_entropy_fwd_f16(logits_f16, target, V);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double tps = 1e6 / us;
    double bytes = (double)V * 2.0 * 2.0;     /* 2 passes, 2 bytes/elem */
    double bw_b_cy = bytes / cy;

    printf("  %-26s V=%6d: %8.1f cy  %6.1f us  %6.0f tok/s  %5.2f B/cy  loss=%.4f\n",
           label, V, cy, us, tps, bw_b_cy, loss / reps);

    free(logits_f32);
    free(logits_f16);
}

static void bench_fp16_single(const char *label, int V, int reps,
                               float (*fn)(const uint16_t *, int, int),
                               int npasses) {
    float *logits_f32 = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    uint16_t *logits_f16 = (uint16_t *)aligned_alloc(256, (size_t)V * sizeof(uint16_t));
    fill_random(logits_f32, V, -5.0f, 5.0f, 42);
    fp32_to_fp16(logits_f32, logits_f16, V);
    int target = V / 2;
    uint64_t freq = rdfreq();

    volatile float dummy = 0;
    for (int r = 0; r < 3; r++)
        dummy += fn(logits_f16, target, V);

    uint64_t t0 = rdtsc();
    float loss = 0;
    for (int r = 0; r < reps; r++)
        loss += fn(logits_f16, target, V);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double tps = 1e6 / us;
    double bytes = (double)V * 2.0 * npasses;
    double bw_b_cy = bytes / cy;

    printf("  %-26s V=%6d: %8.1f cy  %6.1f us  %6.0f tok/s  %5.2f B/cy  loss=%.4f\n",
           label, V, cy, us, tps, bw_b_cy, loss / reps);

    free(logits_f32);
    free(logits_f16);
}

static void bench_fwd_bwd_f16(const char *label, int V, int reps,
                                float (*fn)(const uint16_t *, int, int, float *),
                                int npasses_read, int npasses_write) {
    float *logits_f32 = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    uint16_t *logits_f16 = (uint16_t *)aligned_alloc(256, (size_t)V * sizeof(uint16_t));
    float *grad = (float *)aligned_alloc(256, (size_t)V * sizeof(float));
    fill_random(logits_f32, V, -5.0f, 5.0f, 42);
    fp32_to_fp16(logits_f32, logits_f16, V);
    int target = V / 2;
    uint64_t freq = rdfreq();

    volatile float dummy = 0;
    for (int r = 0; r < 3; r++)
        dummy += fn(logits_f16, target, V, grad);

    uint64_t t0 = rdtsc();
    float loss = 0;
    for (int r = 0; r < reps; r++)
        loss += fn(logits_f16, target, V, grad);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double tps = 1e6 / us;
    /* fp16 reads (2B) + fp32 writes (4B) */
    double bytes = (double)V * (2.0 * npasses_read + 4.0 * npasses_write);
    double bw_b_cy = bytes / cy;

    printf("  %-26s V=%6d: %8.1f cy  %6.1f us  %6.0f tok/s  %5.2f B/cy  loss=%.4f\n",
           label, V, cy, us, tps, bw_b_cy, loss / reps);

    free(logits_f32);
    free(logits_f16);
    free(grad);
}

static void bench_batch(int V, int batch, int reps) {
    float *logits = (float *)aligned_alloc(256, (int64_t)batch * V * sizeof(float));
    int *targets = (int *)malloc(batch * sizeof(int));
    float *losses = (float *)malloc(batch * sizeof(float));

    for (int t = 0; t < batch; t++) {
        fill_random(logits + (int64_t)t * V, V, -10.0f, 10.0f, 42 + t);
        targets[t] = t % V;
    }
    uint64_t freq = rdfreq();

    cross_entropy_batch_f32(logits, targets, losses, batch, V);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++)
        cross_entropy_batch_f32(logits, targets, losses, batch, V);
    uint64_t t1 = rdtsc();

    double cy = (double)(t1 - t0) / reps;
    double us = cy / ((double)freq / 1e6);
    double tps = (double)batch / (us / 1e6);

    printf("  batch=%-4d V=%6d: %.0f cy  %.0f us  %.0f tok/s\n",
           batch, V, cy, us, tps);

    free(logits);
    free(targets);
    free(losses);
}

/* ════════════════════════════════════════════════════════════════
 * Multi-core scaling benchmark (OMP)
 * ════════════════════════════════════════════════════════════════ */

#ifdef _OPENMP

static void bench_multicore_fp32(const char *label, int V, int batch, int reps,
                                  float (*fn)(const float *, int, int)) {
    float *logits = (float *)aligned_alloc(256, (int64_t)batch * V * sizeof(float));
    int *targets = (int *)malloc(batch * sizeof(int));
    volatile float *losses = (float *)malloc(batch * sizeof(float));

    for (int t = 0; t < batch; t++) {
        fill_random(logits + (int64_t)t * V, V, -10.0f, 10.0f, 42 + t);
        targets[t] = t % V;
    }
    uint64_t freq = rdfreq();

    int nc_list[] = {1, 2, 4, 8, 12};
    int nnc = sizeof(nc_list) / sizeof(nc_list[0]);

    printf("\n  %-20s V=%d batch=%d reps=%d\n", label, V, batch, reps);
    printf("  %4s  %10s  %10s  %10s  %8s\n", "NC", "cy", "us", "tok/s", "eff%");

    double base_tps = 0;

    for (int c = 0; c < nnc; c++) {
        int nc = nc_list[c];
        omp_set_num_threads(nc);

        /* Warmup */
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < batch; t++)
            losses[t] = fn(logits + (int64_t)t * V, targets[t], V);

        uint64_t t0 = rdtsc();
        for (int r = 0; r < reps; r++) {
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < batch; t++)
                losses[t] = fn(logits + (int64_t)t * V, targets[t], V);
        }
        uint64_t t1 = rdtsc();

        double cy = (double)(t1 - t0) / reps;
        double us = cy / ((double)freq / 1e6);
        double tps = (double)batch / (us / 1e6);

        if (c == 0) base_tps = tps;
        double eff = (tps / ((double)nc * base_tps)) * 100.0;

        printf("  %4d  %10.0f  %10.1f  %10.0f  %7.1f%%\n",
               nc, cy, us, tps, eff);
    }

    free(logits);
    free(targets);
    free((void *)losses);
}

static void bench_multicore_fp16(const char *label, int V, int batch, int reps) {
    float *logits_f32 = (float *)aligned_alloc(256, (int64_t)batch * V * sizeof(float));
    uint16_t *logits_f16 = (uint16_t *)aligned_alloc(256, (int64_t)batch * V * sizeof(uint16_t));
    int *targets = (int *)malloc(batch * sizeof(int));
    volatile float *losses = (float *)malloc(batch * sizeof(float));

    for (int t = 0; t < batch; t++) {
        fill_random(logits_f32 + (int64_t)t * V, V, -5.0f, 5.0f, 42 + t);
        fp32_to_fp16(logits_f32 + (int64_t)t * V,
                     logits_f16 + (int64_t)t * V, V);
        targets[t] = t % V;
    }
    uint64_t freq = rdfreq();

    int nc_list[] = {1, 2, 4, 8, 12};
    int nnc = sizeof(nc_list) / sizeof(nc_list[0]);

    printf("\n  %-20s V=%d batch=%d reps=%d\n", label, V, batch, reps);
    printf("  %4s  %10s  %10s  %10s  %8s\n", "NC", "cy", "us", "tok/s", "eff%");

    double base_tps = 0;

    for (int c = 0; c < nnc; c++) {
        int nc = nc_list[c];
        omp_set_num_threads(nc);

        /* Warmup */
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < batch; t++)
            losses[t] = cross_entropy_fwd_f16(
                logits_f16 + (int64_t)t * V, targets[t], V);

        uint64_t t0 = rdtsc();
        for (int r = 0; r < reps; r++) {
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < batch; t++)
                losses[t] = cross_entropy_fwd_f16(
                    logits_f16 + (int64_t)t * V, targets[t], V);
        }
        uint64_t t1 = rdtsc();

        double cy = (double)(t1 - t0) / reps;
        double us = cy / ((double)freq / 1e6);
        double tps = (double)batch / (us / 1e6);

        if (c == 0) base_tps = tps;
        double eff = (tps / ((double)nc * base_tps)) * 100.0;

        printf("  %4d  %10.0f  %10.1f  %10.0f  %7.1f%%\n",
               nc, cy, us, tps, eff);
    }

    free(logits_f32);
    free(logits_f16);
    free(targets);
    free((void *)losses);
}

/* ════════════════════════════════════════════════════════════════
 * Cross-CMG NUMA-local scaling benchmark
 *
 * Shared node cores: 12-59 across CMGs 4-7 (12 cores per CMG)
 *   CMG4: cores 12-23    CMG5: cores 24-35
 *   CMG6: cores 36-47    CMG7: cores 48-59
 *
 * Same-CMG:  all threads on CMG4 (cores 12..12+NC-1)
 * Cross-CMG: threads spread round-robin across 4 CMGs
 *            e.g. NC=4 → {12,24,36,48} = 1 per CMG
 *                 NC=8 → {12,24,36,48,13,25,37,49} = 2 per CMG
 * ════════════════════════════════════════════════════════════════ */

static const int CMG_BASE[4] = {12, 24, 36, 48};

/* Pin OMP threads: all on CMG4 */
static void pin_same_cmg(int nc) {
    omp_set_num_threads(nc);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(12 + tid, &cs);
        sched_setaffinity(0, sizeof(cs), &cs);
    }
}

/* Pin OMP threads: round-robin across 4 CMGs */
static void pin_cross_cmg(int nc) {
    omp_set_num_threads(nc);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int cmg = tid % 4;
        int local = tid / 4;
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(CMG_BASE[cmg] + local, &cs);
        sched_setaffinity(0, sizeof(cs), &cs);
    }
}

static void bench_numa_fp32(const char *label, int V, int batch, int reps,
                             float (*fn)(const float *, int, int),
                             void (*pin_fn)(int), const int *nc_list, int nnc) {
    float *logits = (float *)aligned_alloc(256, (int64_t)batch * V * sizeof(float));
    int *targets = (int *)malloc(batch * sizeof(int));
    volatile float *losses = (float *)malloc(batch * sizeof(float));

    for (int t = 0; t < batch; t++) {
        fill_random(logits + (int64_t)t * V, V, -10.0f, 10.0f, 42 + t);
        targets[t] = t % V;
    }
    uint64_t freq = rdfreq();

    printf("\n  %-28s V=%d batch=%d\n", label, V, batch);
    printf("  %4s  %10s  %10s  %10s  %8s  %8s\n",
           "NC", "cy", "us", "tok/s", "eff%", "tok/s/c");

    double base_tps_per_core = 0;

    for (int c = 0; c < nnc; c++) {
        int nc = nc_list[c];
        pin_fn(nc);

        /* Warmup */
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < batch; t++)
            losses[t] = fn(logits + (int64_t)t * V, targets[t], V);

        uint64_t t0 = rdtsc();
        for (int r = 0; r < reps; r++) {
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < batch; t++)
                losses[t] = fn(logits + (int64_t)t * V, targets[t], V);
        }
        uint64_t t1 = rdtsc();

        double cy = (double)(t1 - t0) / reps;
        double us = cy / ((double)freq / 1e6);
        double tps = (double)batch / (us / 1e6);
        double tps_per_core = tps / nc;

        if (c == 0) base_tps_per_core = tps_per_core;
        double eff = (tps_per_core / base_tps_per_core) * 100.0;

        printf("  %4d  %10.0f  %10.1f  %10.0f  %7.1f%%  %8.0f\n",
               nc, cy, us, tps, eff, tps_per_core);
    }

    free(logits);
    free(targets);
    free((void *)losses);
}

static void bench_numa_fp16(const char *label, int V, int batch, int reps,
                             void (*pin_fn)(int), const int *nc_list, int nnc) {
    float *logits_f32 = (float *)aligned_alloc(256, (int64_t)batch * V * sizeof(float));
    uint16_t *logits_f16 = (uint16_t *)aligned_alloc(256, (int64_t)batch * V * sizeof(uint16_t));
    int *targets = (int *)malloc(batch * sizeof(int));
    volatile float *losses = (float *)malloc(batch * sizeof(float));

    for (int t = 0; t < batch; t++) {
        fill_random(logits_f32 + (int64_t)t * V, V, -5.0f, 5.0f, 42 + t);
        fp32_to_fp16(logits_f32 + (int64_t)t * V,
                     logits_f16 + (int64_t)t * V, V);
        targets[t] = t % V;
    }
    uint64_t freq = rdfreq();

    printf("\n  %-28s V=%d batch=%d\n", label, V, batch);
    printf("  %4s  %10s  %10s  %10s  %8s  %8s\n",
           "NC", "cy", "us", "tok/s", "eff%", "tok/s/c");

    double base_tps_per_core = 0;

    for (int c = 0; c < nnc; c++) {
        int nc = nc_list[c];
        pin_fn(nc);

        #pragma omp parallel for schedule(static)
        for (int t = 0; t < batch; t++)
            losses[t] = cross_entropy_fwd_f16(
                logits_f16 + (int64_t)t * V, targets[t], V);

        uint64_t t0 = rdtsc();
        for (int r = 0; r < reps; r++) {
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < batch; t++)
                losses[t] = cross_entropy_fwd_f16(
                    logits_f16 + (int64_t)t * V, targets[t], V);
        }
        uint64_t t1 = rdtsc();

        double cy = (double)(t1 - t0) / reps;
        double us = cy / ((double)freq / 1e6);
        double tps = (double)batch / (us / 1e6);
        double tps_per_core = tps / nc;

        if (c == 0) base_tps_per_core = tps_per_core;
        double eff = (tps_per_core / base_tps_per_core) * 100.0;

        printf("  %4d  %10.0f  %10.1f  %10.0f  %7.1f%%  %8.0f\n",
               nc, cy, us, tps, eff, tps_per_core);
    }

    free(logits_f32);
    free(logits_f16);
    free(targets);
    free((void *)losses);
}

#endif /* _OPENMP */

/* ════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    printf("SVE Cross-Entropy Loss Benchmark for A64FX\n");
    printf("VL = %d floats (%d bits)\n", (int)svcntw(), (int)svcntw() * 32);
    printf("Timer freq = %lu Hz\n", rdfreq());

    /* ── Accuracy tests ── */
    test_log_accuracy();

    int test_vocabs[] = { 1024, 32000, 128256, 151936 };
    int nvocabs = sizeof(test_vocabs) / sizeof(test_vocabs[0]);

    for (int v = 0; v < nvocabs; v++)
        test_accuracy(test_vocabs[v]);

    /* Blocked accuracy */
    for (int v = 0; v < nvocabs; v++)
        test_blocked_accuracy(test_vocabs[v]);

    test_fp16_accuracy(32000);
    test_fp16_accuracy(151936);

    /* Blocked FP16 accuracy */
    test_blocked_fp16_accuracy(32000);
    test_blocked_fp16_accuracy(151936);

    /* Gradient checks for both 3-pass and blocked fwd+bwd (f32) */
    test_gradient(1024, "3-pass", cross_entropy_fwd_bwd_f32);
    test_gradient(32000, "3-pass", cross_entropy_fwd_bwd_f32);
    test_gradient(1024, "blocked", cross_entropy_fwd_bwd_blocked_f32);
    test_gradient(32000, "blocked", cross_entropy_fwd_bwd_blocked_f32);

    /* Gradient checks for fp16 fwd+bwd (3-pass and blocked) */
    test_gradient_f16(1024, "fp16 3-pass", cross_entropy_fwd_bwd_f16);
    test_gradient_f16(32000, "fp16 3-pass", cross_entropy_fwd_bwd_f16);
    test_gradient_f16(1024, "fp16 blocked", cross_entropy_fwd_bwd_blocked_f16);
    test_gradient_f16(32000, "fp16 blocked", cross_entropy_fwd_bwd_blocked_f16);

    /* ── Performance benchmarks ── */
    printf("\n=== Performance Benchmarks (single core) ===\n");

    int bench_vocabs[] = { 32000, 128256, 151936, 256000 };
    int nbench = sizeof(bench_vocabs) / sizeof(bench_vocabs[0]);

    printf("\n--- Forward only ---\n");
    for (int v = 0; v < nbench; v++) {
        int V = bench_vocabs[v];
        int reps = (V >= 128000) ? 200 : 500;
        bench_scalar("scalar", V, reps);
        bench_single("SVE 2-pass+pf+8x", V, reps,
                     cross_entropy_fwd_f32, 2);
        bench_single("SVE blocked f32", V, reps,
                     cross_entropy_fwd_blocked_f32, 2);
        bench_fp16("SVE fp16-in 2-pass", V, reps);
        bench_fp16_single("SVE fp16-in blocked", V, reps,
                          cross_entropy_fwd_blocked_f16, 2);
        printf("\n");
    }

    printf("\n--- Forward + Backward (f32) ---\n");
    for (int v = 0; v < nbench; v++) {
        int V = bench_vocabs[v];
        int reps = (V >= 128000) ? 200 : 500;
        bench_fwd_bwd("SVE 3-pass f32", V, reps,
                      cross_entropy_fwd_bwd_f32, 3);
        bench_fwd_bwd("SVE blocked f32 fwd+bwd", V, reps,
                      cross_entropy_fwd_bwd_blocked_f32, 3);
    }

    printf("\n--- Forward + Backward (fp16) ---\n");
    for (int v = 0; v < nbench; v++) {
        int V = bench_vocabs[v];
        int reps = (V >= 128000) ? 200 : 500;
        bench_fwd_bwd_f16("SVE 3-pass fp16", V, reps,
                          cross_entropy_fwd_bwd_f16, 3, 1);
        bench_fwd_bwd_f16("SVE blocked fp16 fwd+bwd", V, reps,
                          cross_entropy_fwd_bwd_blocked_f16, 2, 1);
    }

    /* ── Batch benchmarks ── */
    printf("\n--- Batch (V=151936) ---\n");
    int batches[] = { 1, 128, 1024 };
    for (int b = 0; b < 3; b++) {
        int reps = (batches[b] >= 1024) ? 5 : 20;
        bench_batch(151936, batches[b], reps);
    }

    /* ── Multi-core scaling (OMP only) ── */
#ifdef _OPENMP
    printf("\n=== Multi-core Scaling (V=151936) ===\n");
    bench_multicore_fp32("SVE 2-pass fp32", 151936, 128, 20,
                         cross_entropy_fwd_f32);
    bench_multicore_fp32("SVE blocked fp32", 151936, 128, 20,
                         cross_entropy_fwd_blocked_f32);
    bench_multicore_fp16("SVE fp16-in (LD1H)", 151936, 128, 20);

    /* ── NUMA cross-CMG scaling ── */
    int nc_same[]  = {1, 2, 4, 8, 12};
    int nc_cross[] = {4, 8, 12, 16, 24, 48};

    printf("\n=== Same-CMG Scaling (CMG4 only, V=151936) ===\n");
    bench_numa_fp32("same-CMG 2-pass fp32", 151936, 128, 20,
                     cross_entropy_fwd_f32, pin_same_cmg, nc_same, 5);
    bench_numa_fp32("same-CMG blocked fp32", 151936, 128, 20,
                     cross_entropy_fwd_blocked_f32, pin_same_cmg, nc_same, 5);
    bench_numa_fp16("same-CMG fp16-in", 151936, 128, 20,
                     pin_same_cmg, nc_same, 5);

    printf("\n=== Cross-CMG Scaling (4 CMGs, V=151936) ===\n");
    bench_numa_fp32("cross-CMG 2-pass fp32", 151936, 128, 20,
                     cross_entropy_fwd_f32, pin_cross_cmg, nc_cross, 6);
    bench_numa_fp32("cross-CMG blocked fp32", 151936, 128, 20,
                     cross_entropy_fwd_blocked_f32, pin_cross_cmg, nc_cross, 6);
    bench_numa_fp16("cross-CMG fp16-in", 151936, 128, 20,
                     pin_cross_cmg, nc_cross, 6);
#endif

    printf("\nDone.\n");
    return 0;
}
