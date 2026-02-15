/*
 * Benchmark + correctness test for SVE sincos and RoPE kernels
 *
 * Tests:
 *   1. Accuracy vs libm sinf/cosf for various angle ranges
 *   2. Sincos throughput (elements/sec, cycles/element)
 *   3. Fused RoPE throughput (elements/sec)
 *   4. Comparison: poly vs ftmad
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#include "sincos.h"

/* ---- Timer ---- */
static inline uint64_t rdtsc(void)
{
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t rdfreq(void)
{
    uint64_t val;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(val));
    return val;
}

/* ---- Helpers ---- */
static float randf(float lo, float hi)
{
    return lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

static float ulp_error(float computed, float reference)
{
    if (reference == computed) return 0.0f;
    if (isnan(reference) || isnan(computed)) return INFINITY;
    /* ULP distance */
    float ulp = scalbnf(1.0f, ilogbf(reference) - 23);
    if (ulp == 0.0f) ulp = FLT_MIN;
    return fabsf(computed - reference) / fabsf(ulp);
}

/* ---- Accuracy test ---- */
static void test_accuracy(const char *name,
                          void (*sincos_fn)(const float *, float *, float *, int64_t),
                          float lo, float hi, int n)
{
    float *theta  = (float *)aligned_alloc(64, n * sizeof(float));
    float *s_sve  = (float *)aligned_alloc(64, n * sizeof(float));
    float *c_sve  = (float *)aligned_alloc(64, n * sizeof(float));

    for (int i = 0; i < n; i++)
        theta[i] = randf(lo, hi);

    sincos_fn(theta, s_sve, c_sve, n);

    float max_ulp_sin = 0, max_ulp_cos = 0;
    double avg_ulp_sin = 0, avg_ulp_cos = 0;
    float worst_angle_sin = 0, worst_angle_cos = 0;

    for (int i = 0; i < n; i++) {
        float s_ref = sinf(theta[i]);
        float c_ref = cosf(theta[i]);

        float ue_s = ulp_error(s_sve[i], s_ref);
        float ue_c = ulp_error(c_sve[i], c_ref);

        avg_ulp_sin += ue_s;
        avg_ulp_cos += ue_c;

        if (ue_s > max_ulp_sin) {
            max_ulp_sin = ue_s;
            worst_angle_sin = theta[i];
        }
        if (ue_c > max_ulp_cos) {
            max_ulp_cos = ue_c;
            worst_angle_cos = theta[i];
        }
    }
    avg_ulp_sin /= n;
    avg_ulp_cos /= n;

    printf("  %-8s [%8.2f, %8.2f]: sin max=%.1f avg=%.2f ULP | "
           "cos max=%.1f avg=%.2f ULP",
           name, lo, hi,
           max_ulp_sin, avg_ulp_sin,
           max_ulp_cos, avg_ulp_cos);

    if (max_ulp_sin > 4.0f || max_ulp_cos > 4.0f)
        printf("  ** HIGH ERROR **");
    printf("\n");

    /* Print worst cases if error is large */
    if (max_ulp_sin > 2.0f) {
        float s_ref = sinf(worst_angle_sin);
        printf("    worst sin: theta=%.8e  got=%.8e  ref=%.8e  (%.1f ULP)\n",
               worst_angle_sin, s_sve[0], s_ref, max_ulp_sin);
    }
    if (max_ulp_cos > 2.0f) {
        float c_ref = cosf(worst_angle_cos);
        printf("    worst cos: theta=%.8e  got=%.8e  ref=%.8e  (%.1f ULP)\n",
               worst_angle_cos, c_sve[0], c_ref, max_ulp_cos);
    }

    free(theta);
    free(s_sve);
    free(c_sve);
}

/* ---- Spot check: print a few values ---- */
static void spot_check(const char *name,
                       void (*sincos_fn)(const float *, float *, float *, int64_t))
{
    float angles[] = {0.0f, 0.1f, 0.5f, 1.0f, M_PI/4, M_PI/2,
                      M_PI, 3*M_PI/2, 2*M_PI, 10.0f, 100.0f, -0.5f, -3.0f};
    int n = sizeof(angles)/sizeof(angles[0]);
    float s[16], c[16];  /* padded */

    sincos_fn(angles, s, c, n);

    printf("\n  %-8s spot check:\n", name);
    printf("  %12s %12s %12s %12s %12s\n",
           "theta", "sin(SVE)", "sin(ref)", "cos(SVE)", "cos(ref)");
    for (int i = 0; i < n; i++) {
        printf("  %12.6f %12.8f %12.8f %12.8f %12.8f\n",
               angles[i], s[i], sinf(angles[i]), c[i], cosf(angles[i]));
    }
}

/* ---- Throughput benchmark: sincos ---- */
static void bench_sincos(const char *name,
                         void (*sincos_fn)(const float *, float *, float *, int64_t),
                         int n, int reps)
{
    float *theta = (float *)aligned_alloc(64, n * sizeof(float));
    float *s     = (float *)aligned_alloc(64, n * sizeof(float));
    float *c     = (float *)aligned_alloc(64, n * sizeof(float));

    for (int i = 0; i < n; i++)
        theta[i] = randf(-100.0f, 100.0f);

    /* Warmup */
    for (int r = 0; r < 3; r++)
        sincos_fn(theta, s, c, n);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++)
        sincos_fn(theta, s, c, n);
    uint64_t t1 = rdtsc();

    double cycles = (double)(t1 - t0);
    double cy_per_elem = cycles / ((double)n * reps);
    double freq = (double)rdfreq();
    double elems_per_sec = (double)n * reps / ((t1 - t0) / freq);

    printf("  %-8s N=%5d: %6.2f cy/elem  %.1f Melem/s\n",
           name, n, cy_per_elem, elems_per_sec / 1e6);

    free(theta);
    free(s);
    free(c);
}

/* ---- Throughput benchmark: RoPE ---- */
static void bench_rope(const char *name,
                       void (*rope_fn)(float *, const float *, int64_t),
                       int dim, int reps)
{
    float *x     = (float *)aligned_alloc(64, dim * sizeof(float));
    float *theta = (float *)aligned_alloc(64, (dim/2) * sizeof(float));

    for (int i = 0; i < dim; i++)
        x[i] = randf(-1.0f, 1.0f);
    for (int i = 0; i < dim/2; i++)
        theta[i] = randf(-10.0f, 10.0f);

    /* Warmup */
    for (int r = 0; r < 3; r++)
        rope_fn(x, theta, dim);

    uint64_t t0 = rdtsc();
    for (int r = 0; r < reps; r++)
        rope_fn(x, theta, dim);
    uint64_t t1 = rdtsc();

    double cycles = (double)(t1 - t0);
    double cy_per_elem = cycles / ((double)dim * reps);
    double freq = (double)rdfreq();
    double elems_per_sec = (double)dim * reps / ((t1 - t0) / freq);

    printf("  %-8s dim=%4d: %6.2f cy/elem  %.1f Melem/s\n",
           name, dim, cy_per_elem, elems_per_sec / 1e6);

    free(x);
    free(theta);
}

/* ---- RoPE correctness test ---- */
static void test_rope_accuracy(const char *name,
                               void (*rope_fn)(float *, const float *, int64_t),
                               int dim)
{
    float *x_sve   = (float *)aligned_alloc(64, dim * sizeof(float));
    float *x_ref   = (float *)aligned_alloc(64, dim * sizeof(float));
    float *theta   = (float *)aligned_alloc(64, (dim/2) * sizeof(float));

    for (int i = 0; i < dim; i++)
        x_sve[i] = x_ref[i] = randf(-1.0f, 1.0f);
    for (int i = 0; i < dim/2; i++)
        theta[i] = randf(-10.0f, 10.0f);

    /* SVE */
    rope_fn(x_sve, theta, dim);

    /* Reference */
    for (int i = 0; i < dim/2; i++) {
        float s = sinf(theta[i]);
        float c = cosf(theta[i]);
        float e = x_ref[2*i];
        float o = x_ref[2*i+1];
        x_ref[2*i]   = e * c - o * s;
        x_ref[2*i+1] = e * s + o * c;
    }

    float max_err = 0;
    for (int i = 0; i < dim; i++) {
        float err = fabsf(x_sve[i] - x_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("  %-8s dim=%4d: max abs error = %.2e %s\n",
           name, dim, max_err,
           max_err < 1e-5f ? "OK" : "** FAIL **");

    free(x_sve);
    free(x_ref);
    free(theta);
}

/* ---- Main ---- */
int main(int argc, char **argv)
{
    srand(42);

    printf("=== SVE sincos accuracy ===\n");
    int ntest = 10000;

    /* Test various ranges */
    float ranges[][2] = {
        {-1.0f, 1.0f},
        {-M_PI, M_PI},
        {-10.0f, 10.0f},
        {-100.0f, 100.0f},
        {-1000.0f, 1000.0f},
        {0.0f, 0.001f},        /* tiny angles (RoPE high freq) */
    };
    int nranges = sizeof(ranges) / sizeof(ranges[0]);

    for (int r = 0; r < nranges; r++) {
        test_accuracy("poly", sve_sincos_poly_f32, ranges[r][0], ranges[r][1], ntest);
        test_accuracy("ftmad", sve_sincos_ftmad_f32, ranges[r][0], ranges[r][1], ntest);
    }

    printf("\n=== Spot check ===\n");
    spot_check("poly", sve_sincos_poly_f32);
    spot_check("ftmad", sve_sincos_ftmad_f32);

    printf("\n=== RoPE accuracy ===\n");
    int dims[] = {16, 64, 128, 256, 512};
    for (int d = 0; d < 5; d++) {
        test_rope_accuracy("poly", sve_rope_poly_f32, dims[d]);
        test_rope_accuracy("ftmad", sve_rope_ftmad_f32, dims[d]);
    }

    printf("\n=== Sincos throughput ===\n");
    int sizes[] = {16, 64, 256, 1024, 4096, 16384};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < nsizes; i++) {
        int reps = 100000 / (sizes[i] / 16 + 1);
        if (reps < 100) reps = 100;
        bench_sincos("poly", sve_sincos_poly_f32, sizes[i], reps);
        bench_sincos("ftmad", sve_sincos_ftmad_f32, sizes[i], reps);
        printf("\n");
    }

    printf("=== RoPE throughput ===\n");
    int rdims[] = {64, 128, 256, 512, 1024, 4096};
    for (int d = 0; d < 6; d++) {
        int reps = 100000 / (rdims[d] / 64 + 1);
        if (reps < 100) reps = 100;
        bench_rope("poly", sve_rope_poly_f32, rdims[d], reps);
        bench_rope("ftmad", sve_rope_ftmad_f32, rdims[d], reps);
        printf("\n");
    }

    return 0;
}
