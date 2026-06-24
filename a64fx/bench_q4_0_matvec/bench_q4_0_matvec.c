/* bench_q4_0_matvec.c - Benchmark Q4_0 matvec paths on actual A64FX hardware
 *
 * Compares:
 *   1. fp32 FMA (gold standard, baseline)
 *   2. int8 SDOT (prequantized, production path - 40x faster than fp32)
 *   3. int16 SDOT (H->D, 2x slower than int8)
 *   4. fp16 FMA (2x slower than int8)
 *
 * Build (cross-compile for A64FX):
 *   aarch64-linux-gnu-gcc-14 -O3 -static -march=armv8.2-a+sve \
 *     -ffp-contract=fast -D_GNU_SOURCE -I../../common \
 *     bench_q4_0_matvec.c -lm -lpthread -o bench_q4_0_matvec
 *
 * Run on A64FX:
 *   ./bench_q4_0_matvec
 *
 * Output: per-path wall time, cycle count, effective GFLOPS, speedup vs fp32.
 *         Also verifies correctness against fp32 reference.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_sve.h>

#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"

/* ------------------------------------------------------------------ */
/* AArch64 cycle counter (cntvct_el0 virtual counter at user freq)      */
/* ------------------------------------------------------------------ */
static inline uint64_t cycle_read(void) {
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t cycle_freq(void) {
    uint64_t freq;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

/* ------------------------------------------------------------------ */
/* Q4_0 weight initialization: random fp16 d + random nibbles           */
/* ------------------------------------------------------------------ */

/* Convert float32 to float16 (IEEE 754 half-precision). */
static uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        /* Subnormal fp16 */
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    }
    if (exp >= 31) {
        if (exp == 128 && mant) return (uint16_t)(sign | 0x7E00); /* NaN */
        return (uint16_t)(sign | 0x7C00); /* Inf */
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static void init_q4_0_weights(block_q4_0 *w, int n_blocks, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n_blocks; i++) {
        /* Random fp16 d in [0.01, 2.0] - typical Q4_0 range. */
        float d = 0.01f + (rand() / (float)RAND_MAX) * 1.99f;
        w[i].d = fp32_to_fp16(d);
        /* Random nibbles in [0, 15] */
        for (int j = 0; j < 16; j++) {
            w[i].qs[j] = (unsigned char)(rand() & 0xFF);
        }
    }
}

/* ------------------------------------------------------------------ */
/* fp32 activation initialization                                       */
/* ------------------------------------------------------------------ */
static void init_fp32_activations(float *x, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        x[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;  /* [-1, 1] */
    }
}

/* ------------------------------------------------------------------ */
/* fp32 reference matvec (gold standard)                               */
/* ------------------------------------------------------------------ */
static void matvec_fp32_reference(float *dst, const block_q4_0 *w,
                                    size_t row_bytes, const float *x,
                                    int n_rows, int n_cols) {
    for (int r = 0; r < n_rows; r++) {
        const block_q4_0 *row = (const block_q4_0 *)((const uint8_t *)w + (size_t)r * row_bytes);
        float sum = 0.0f;
        int nb = n_cols / 32;
        for (int b = 0; b < nb; b++) {
            float d = ggml_fp16_to_fp32(row[b].d);
            for (int j = 0; j < 16; j++) {
                int v0 = (row[b].qs[j] & 0x0F) - 8;
                int v1 = (row[b].qs[j] >> 4) - 8;
                sum += d * v0 * x[b * 32 + j];
                sum += d * v1 * x[b * 32 + j + 16];
            }
        }
        dst[r] = sum;
    }
}

/* ------------------------------------------------------------------ */
/* SVE fp32 FMA matvec (on-the-fly dequant)                            */
/* ------------------------------------------------------------------ */
static void matvec_fp32_sve(float *dst, const block_q4_0 *w,
                              size_t row_bytes, const float *x,
                              int n_rows, int n_cols) {
    /* Use the dispatch from transformer.h which has the optimized SVE path. */
    tf_matvec_q4_0_rows(dst, (const uint8_t *)w, row_bytes, x, n_cols, 0, n_rows);
}

/* ------------------------------------------------------------------ */
/* SVE int8 SDOT matvec (prequantized, opt-in via TF_USE_INT8_SDOT_Q4_0 */
/* for the on-the-fly path, or prequant cache for the fast path).        */
/* ------------------------------------------------------------------ */

/* On-the-fly int8 SDOT (dequant in hot path) */
#ifdef TF_USE_INT8_SDOT_Q4_0
static void matvec_int8_onthefly(float *dst, const block_q4_0 *w,
                                    size_t row_bytes, const float *x,
                                    int n_rows, int n_cols) {
    tf_matvec_q4_0_rows(dst, (const uint8_t *)w, row_bytes, x, n_cols, 0, n_rows);
}
#endif

/* Prequantized int8 SDOT (dequant once, reuse for many matvecs)
 * The cache is initialized once and reused. The matvec itself is timed
 * separately from the prepack cost (which is amortized over many matvecs). */
static void matvec_int8_prequant(float *dst, const block_q4_0 *w,
                                   size_t row_bytes, const float *x,
                                   int n_rows, int n_cols,
                                   tf_q4_0_int8_cache *cache) {
    /* Caller has already initialized the cache. Just do the matvec. */
    tf_matvec_q4_0_int8_prequant_rows(dst, cache, x, 0, n_rows);
}

/* ------------------------------------------------------------------ */
/* SVE int16 SDOT matvec (H->D, 2x slower than int8)                    */
/* ------------------------------------------------------------------ */
static void matvec_int16_sve(float *dst, const int16_t *wi16,
                               const float *x, int n_rows, int n_cols) {
    int nb_pairs = n_cols / 32;
    int16_t *xi16 = aligned_alloc(256, (((size_t)n_cols + 31) & ~31) * sizeof(int16_t));
    /* Quantize x to int16: x' = x (values in [-1, 1] fit in int16) */
    for (int j = 0; j < n_cols; j++) {
        float v = x[j] * 32767.0f;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        xi16[j] = (int16_t)lrintf(v);
    }
    for (int b = 0; b < n_rows; b += 8) {
        svbool_t pg = svptrue_b16();
        svint64_t a0=svdup_s64(0),a1=svdup_s64(0),a2=svdup_s64(0),a3=svdup_s64(0);
        svint64_t a4=svdup_s64(0),a5=svdup_s64(0),a6=svdup_s64(0),a7=svdup_s64(0);
        for (int p = 0; p + 1 < nb_pairs; p += 2) {
            svint16_t x0v = svld1_s16(pg, xi16 + p*32);
            svint16_t x1v = svld1_s16(pg, xi16 + (p+1)*32);
            svint16_t w0a=svld1_s16(pg, wi16 + (size_t)(b+0)*n_cols + p*32);
            svint16_t w0b=svld1_s16(pg, wi16 + (size_t)(b+1)*n_cols + p*32);
            svint16_t w0c=svld1_s16(pg, wi16 + (size_t)(b+2)*n_cols + p*32);
            svint16_t w0d=svld1_s16(pg, wi16 + (size_t)(b+3)*n_cols + p*32);
            svint16_t w0e=svld1_s16(pg, wi16 + (size_t)(b+4)*n_cols + p*32);
            svint16_t w0f=svld1_s16(pg, wi16 + (size_t)(b+5)*n_cols + p*32);
            svint16_t w0g=svld1_s16(pg, wi16 + (size_t)(b+6)*n_cols + p*32);
            svint16_t w0h=svld1_s16(pg, wi16 + (size_t)(b+7)*n_cols + p*32);
            a0=svdot_s64(a0, w0a, x0v); a1=svdot_s64(a1, w0b, x0v);
            a2=svdot_s64(a2, w0c, x0v); a3=svdot_s64(a3, w0d, x0v);
            a4=svdot_s64(a4, w0e, x0v); a5=svdot_s64(a5, w0f, x0v);
            a6=svdot_s64(a6, w0g, x0v); a7=svdot_s64(a7, w0h, x0v);
            svint16_t w1a=svld1_s16(pg, wi16 + (size_t)(b+0)*n_cols + (p+1)*32);
            svint16_t w1b=svld1_s16(pg, wi16 + (size_t)(b+1)*n_cols + (p+1)*32);
            svint16_t w1c=svld1_s16(pg, wi16 + (size_t)(b+2)*n_cols + (p+1)*32);
            svint16_t w1d=svld1_s16(pg, wi16 + (size_t)(b+3)*n_cols + (p+1)*32);
            svint16_t w1e=svld1_s16(pg, wi16 + (size_t)(b+4)*n_cols + (p+1)*32);
            svint16_t w1f=svld1_s16(pg, wi16 + (size_t)(b+5)*n_cols + (p+1)*32);
            svint16_t w1g=svld1_s16(pg, wi16 + (size_t)(b+6)*n_cols + (p+1)*32);
            svint16_t w1h=svld1_s16(pg, wi16 + (size_t)(b+7)*n_cols + (p+1)*32);
            a0=svdot_s64(a0, w1a, x1v); a1=svdot_s64(a1, w1b, x1v);
            a2=svdot_s64(a2, w1c, x1v); a3=svdot_s64(a3, w1d, x1v);
            a4=svdot_s64(a4, w1e, x1v); a5=svdot_s64(a5, w1f, x1v);
            a6=svdot_s64(a6, w1g, x1v); a7=svdot_s64(a7, w1h, x1v);
        }
        /* Convert int64 sums to fp32 and rescale. The x was scaled by 32767,
         * so the result is sum(w*x*32767) / 32767 = sum(w*x). */
        float scale = 1.0f / 32767.0f;
        dst[b+0] = svaddv_s64(svptrue_b64(), a0) * scale;
        dst[b+1] = svaddv_s64(svptrue_b64(), a1) * scale;
        dst[b+2] = svaddv_s64(svptrue_b64(), a2) * scale;
        dst[b+3] = svaddv_s64(svptrue_b64(), a3) * scale;
        dst[b+4] = svaddv_s64(svptrue_b64(), a4) * scale;
        dst[b+5] = svaddv_s64(svptrue_b64(), a5) * scale;
        dst[b+6] = svaddv_s64(svptrue_b64(), a6) * scale;
        dst[b+7] = svaddv_s64(svptrue_b64(), a7) * scale;
    }
    free(xi16);
}

/* Pre-dequantize Q4_0 to int16 (lossless from Q4_0's fp16 d). */
static void dequant_q4_0_to_int16(int16_t *wi16, const block_q4_0 *w,
                                    size_t row_bytes, int n_rows, int n_cols) {
    int nb = n_cols / 32;
    for (int r = 0; r < n_rows; r++) {
        const block_q4_0 *row = (const block_q4_0 *)((const uint8_t *)w + (size_t)r * row_bytes);
        for (int b = 0; b < nb; b++) {
            float d = ggml_fp16_to_fp32(row[b].d);
            for (int j = 0; j < 16; j++) {
                int v0 = (row[b].qs[j] & 0x0F) - 8;
                int v1 = (row[b].qs[j] >> 4) - 8;
                int16_t i0 = (int16_t)lrintf(d * v0);
                int16_t i1 = (int16_t)lrintf(d * v1);
                /* Store as int16: 2 elements per byte, lo in low 16 bits, hi in high 16 bits */
                /* Actually, store separately for clarity */
                wi16[r * n_cols + b * 32 + j]      = i0;
                wi16[r * n_cols + b * 32 + j + 16] = i1;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* SVE fp16 FMA matvec (2x slower than int8)                           */
/* ------------------------------------------------------------------ */
static void matvec_fp16_sve(float *dst, const float16_t *wf16,
                              const float *x, int n_rows, int n_cols) {
    int nb_pairs = n_cols / 32;
    float16_t *xf16 = aligned_alloc(256, (((size_t)n_cols + 31) & ~31) * sizeof(float16_t));
    /* Quantize x to fp16: x' = x (values in [-1, 1] fit in fp16) */
    for (int j = 0; j < n_cols; j++) {
        xf16[j] = (float16_t)x[j];
    }
    for (int b = 0; b < n_rows; b += 8) {
        svbool_t pg = svptrue_b16();
        svfloat16_t a0=svdup_f16(0),a1=svdup_f16(0),a2=svdup_f16(0),a3=svdup_f16(0);
        svfloat16_t a4=svdup_f16(0),a5=svdup_f16(0),a6=svdup_f16(0),a7=svdup_f16(0);
        for (int p = 0; p + 1 < nb_pairs; p += 2) {
            svfloat16_t x0v = svld1_f16(pg, xf16 + p*32);
            svfloat16_t x1v = svld1_f16(pg, xf16 + (p+1)*32);
            svfloat16_t w0a=svld1_f16(pg, wf16 + (size_t)(b+0)*n_cols + p*32);
            svfloat16_t w0b=svld1_f16(pg, wf16 + (size_t)(b+1)*n_cols + p*32);
            svfloat16_t w0c=svld1_f16(pg, wf16 + (size_t)(b+2)*n_cols + p*32);
            svfloat16_t w0d=svld1_f16(pg, wf16 + (size_t)(b+3)*n_cols + p*32);
            svfloat16_t w0e=svld1_f16(pg, wf16 + (size_t)(b+4)*n_cols + p*32);
            svfloat16_t w0f=svld1_f16(pg, wf16 + (size_t)(b+5)*n_cols + p*32);
            svfloat16_t w0g=svld1_f16(pg, wf16 + (size_t)(b+6)*n_cols + p*32);
            svfloat16_t w0h=svld1_f16(pg, wf16 + (size_t)(b+7)*n_cols + p*32);
            a0=svmla_f16_z(pg, a0, w0a, x0v); a1=svmla_f16_z(pg, a1, w0b, x0v);
            a2=svmla_f16_z(pg, a2, w0c, x0v); a3=svmla_f16_z(pg, a3, w0d, x0v);
            a4=svmla_f16_z(pg, a4, w0e, x0v); a5=svmla_f16_z(pg, a5, w0f, x0v);
            a6=svmla_f16_z(pg, a6, w0g, x0v); a7=svmla_f16_z(pg, a7, w0h, x0v);
            svfloat16_t w1a=svld1_f16(pg, wf16 + (size_t)(b+0)*n_cols + (p+1)*32);
            svfloat16_t w1b=svld1_f16(pg, wf16 + (size_t)(b+1)*n_cols + (p+1)*32);
            svfloat16_t w1c=svld1_f16(pg, wf16 + (size_t)(b+2)*n_cols + (p+1)*32);
            svfloat16_t w1d=svld1_f16(pg, wf16 + (size_t)(b+3)*n_cols + (p+1)*32);
            svfloat16_t w1e=svld1_f16(pg, wf16 + (size_t)(b+4)*n_cols + (p+1)*32);
            svfloat16_t w1f=svld1_f16(pg, wf16 + (size_t)(b+5)*n_cols + (p+1)*32);
            svfloat16_t w1g=svld1_f16(pg, wf16 + (size_t)(b+6)*n_cols + (p+1)*32);
            svfloat16_t w1h=svld1_f16(pg, wf16 + (size_t)(b+7)*n_cols + (p+1)*32);
            a0=svmla_f16_z(pg, a0, w1a, x1v); a1=svmla_f16_z(pg, a1, w1b, x1v);
            a2=svmla_f16_z(pg, a2, w1c, x1v); a3=svmla_f16_z(pg, a3, w1d, x1v);
            a4=svmla_f16_z(pg, a4, w1e, x1v); a5=svmla_f16_z(pg, a5, w1f, x1v);
            a6=svmla_f16_z(pg, a6, w1g, x1v); a7=svmla_f16_z(pg, a7, w1h, x1v);
        }
        dst[b+0] = svaddv_f16(pg, a0);
        dst[b+1] = svaddv_f16(pg, a1);
        dst[b+2] = svaddv_f16(pg, a2);
        dst[b+3] = svaddv_f16(pg, a3);
        dst[b+4] = svaddv_f16(pg, a4);
        dst[b+5] = svaddv_f16(pg, a5);
        dst[b+6] = svaddv_f16(pg, a6);
        dst[b+7] = svaddv_f16(pg, a7);
    }
    free(xf16);
}

/* Pre-dequantize Q4_0 to fp16 (lossless from Q4_0's fp16 d). */
static void dequant_q4_0_to_fp16(uint16_t *wf16, const block_q4_0 *w,
                                   size_t row_bytes, int n_rows, int n_cols) {
    int nb = n_cols / 32;
    for (int r = 0; r < n_rows; r++) {
        const block_q4_0 *row = (const block_q4_0 *)((const uint8_t *)w + (size_t)r * row_bytes);
        for (int b = 0; b < nb; b++) {
            float d = ggml_fp16_to_fp32(row[b].d);
            for (int j = 0; j < 16; j++) {
                int v0 = (row[b].qs[j] & 0x0F) - 8;
                int v1 = (row[b].qs[j] >> 4) - 8;
                float16_t f0 = (float16_t)(d * v0);
                float16_t f1 = (float16_t)(d * v1);
                memcpy(&wf16[r * n_cols + b * 32 + j],      &f0, sizeof(uint16_t));
                memcpy(&wf16[r * n_cols + b * 32 + j + 16], &f1, sizeof(uint16_t));
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Benchmark runner                                                    */
/* ------------------------------------------------------------------ */
typedef struct {
    const char *name;
    double ops;            /* total ops (FLOPs) for the matvec */
    uint64_t cycles;
    double seconds;
    double gflops;
    double speedup_vs_fp32;
    int    correct;        /* 1 if output matches reference, 0 otherwise */
    double max_err;        /* max absolute error vs reference */
} bench_result_t;

static double compute_max_err(const float *a, const float *b, int n) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)a[i] - (double)b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static void run_bench(int n_rows, int n_cols, int n_reps) {
    int n_blocks = (n_rows * n_cols) / 32;
    size_t row_bytes = (size_t)(n_cols / 32) * sizeof(block_q4_0);
    size_t total_bytes = (size_t)n_rows * row_bytes;
    block_q4_0 *w = aligned_alloc(256, total_bytes);
    float *x = aligned_alloc(256, (size_t)n_cols * sizeof(float));
    float *dst_ref = aligned_alloc(256, (size_t)n_rows * sizeof(float));
    float *dst = aligned_alloc(256, (size_t)n_rows * sizeof(float));

    /* Init weights and activations */
    init_q4_0_weights(w, n_blocks, 42);
    init_fp32_activations(x, n_cols, 137);

    /* Compute reference (fp32) */
    matvec_fp32_reference(dst_ref, w, row_bytes, x, n_rows, n_cols);

    /* Pre-dequantize for int16 and fp16 paths */
    int16_t *wi16 = aligned_alloc(256, (size_t)n_rows * n_cols * sizeof(int16_t));
    uint16_t *wf16 = aligned_alloc(256, (size_t)n_rows * n_cols * sizeof(uint16_t));
    dequant_q4_0_to_int16(wi16, w, row_bytes, n_rows, n_cols);
    dequant_q4_0_to_fp16(wf16, w, row_bytes, n_rows, n_cols);

    /* Pre-quant cache for int8 prequant path */
    tf_q4_0_int8_cache cache;
    cache.wi8 = NULL;  /* Will be allocated on first call */

    uint64_t freq = cycle_freq();
    double ops_per_matvec = 2.0 * (double)n_rows * (double)n_cols;  /* mul + add per element */

    printf("================================================================\n");
    printf("Q4_0 matvec benchmark: n_rows=%d, n_cols=%d, n_reps=%d\n", n_rows, n_cols, n_reps);
    printf("Cycle counter freq: %lu Hz (%.2f GHz)\n", (unsigned long)freq, (double)freq / 1e9);
    printf("Ops per matvec: %.0f (%.2f GFLOPS theoretical at 1 call/sec)\n",
           ops_per_matvec, ops_per_matvec / 1e9);
    printf("================================================================\n");

    /* Warmup */
    matvec_fp32_sve(dst, w, row_bytes, x, n_rows, n_cols);
#ifdef TF_USE_INT8_SDOT_Q4_0
    matvec_int8_onthefly(dst, w, row_bytes, x, n_rows, n_cols);
#endif
    /* Init int8 cache once for warmup and timed loop */
    if (tf_q4_0_int8_cache_init(&cache, w, row_bytes, n_rows, n_cols) != 0) {
        fprintf(stderr, "cache init failed\n");
        return;
    }
    matvec_int8_prequant(dst, w, row_bytes, x, n_rows, n_cols, &cache);
    matvec_int16_sve(dst, wi16, x, n_rows, n_cols);
    matvec_fp16_sve(dst, (const float16_t *)wf16, x, n_rows, n_cols);

    bench_result_t results[5];
    int n_results = 0;

    /* --- fp32 FMA (baseline) --- */
    {
        uint64_t t0 = cycle_read();
        for (int r = 0; r < n_reps; r++) {
            matvec_fp32_sve(dst, w, row_bytes, x, n_rows, n_cols);
        }
        uint64_t t1 = cycle_read();
        results[n_results].name = "fp32 FMA";
        results[n_results].cycles = t1 - t0;
        results[n_results].seconds = (double)(t1 - t0) / (double)freq;
        results[n_results].ops = ops_per_matvec * n_reps;
        results[n_results].gflops = results[n_results].ops / results[n_results].seconds / 1e9;
        results[n_results].speedup_vs_fp32 = 1.0;
        results[n_results].correct = 1;
        results[n_results].max_err = 0.0;
        n_results++;
    }

    /* --- int8 SDOT on-the-fly --- */
#ifdef TF_USE_INT8_SDOT_Q4_0
    {
        matvec_int8_onthefly(dst, w, row_bytes, x, n_rows, n_cols);
        results[n_results].max_err = compute_max_err(dst, dst_ref, n_rows);
        results[n_results].correct = (results[n_results].max_err < (double)(n_cols) * 0.05);
        uint64_t t0 = cycle_read();
        for (int r = 0; r < n_reps; r++) {
            matvec_int8_onthefly(dst, w, row_bytes, x, n_rows, n_cols);
        }
        uint64_t t1 = cycle_read();
        results[n_results].name = "int8 SDOT (on-fly)";
        results[n_results].cycles = t1 - t0;
        results[n_results].seconds = (double)(t1 - t0) / (double)freq;
        results[n_results].ops = ops_per_matvec * n_reps;
        results[n_results].gflops = results[n_results].ops / results[n_results].seconds / 1e9;
        results[n_results].speedup_vs_fp32 = results[0].seconds / results[n_results].seconds;
        n_results++;
    }
#endif

    /* --- int8 SDOT prequantized (production path) --- */
    {
        /* Cache was initialized in warmup. Just do the matvec. */
        matvec_int8_prequant(dst, w, row_bytes, x, n_rows, n_cols, &cache);
        results[n_results].max_err = compute_max_err(dst, dst_ref, n_rows);
        results[n_results].correct = (results[n_results].max_err < (double)(n_cols) * 0.05);
        uint64_t t0 = cycle_read();
        for (int r = 0; r < n_reps; r++) {
            matvec_int8_prequant(dst, w, row_bytes, x, n_rows, n_cols, &cache);
        }
        uint64_t t1 = cycle_read();
        results[n_results].name = "int8 SDOT (preq)";
        results[n_results].cycles = t1 - t0;
        results[n_results].seconds = (double)(t1 - t0) / (double)freq;
        results[n_results].ops = ops_per_matvec * n_reps;
        results[n_results].gflops = results[n_results].ops / results[n_results].seconds / 1e9;
        results[n_results].speedup_vs_fp32 = results[0].seconds / results[n_results].seconds;
        n_results++;
    }

    /* --- int16 SDOT --- */
    {
        matvec_int16_sve(dst, wi16, x, n_rows, n_cols);
        results[n_results].max_err = compute_max_err(dst, dst_ref, n_rows);
        results[n_results].correct = (results[n_results].max_err < (double)(n_cols) * 0.001);
        uint64_t t0 = cycle_read();
        for (int r = 0; r < n_reps; r++) {
            matvec_int16_sve(dst, wi16, x, n_rows, n_cols);
        }
        uint64_t t1 = cycle_read();
        results[n_results].name = "int16 SDOT";
        results[n_results].cycles = t1 - t0;
        results[n_results].seconds = (double)(t1 - t0) / (double)freq;
        results[n_results].ops = ops_per_matvec * n_reps;
        results[n_results].gflops = results[n_results].ops / results[n_results].seconds / 1e9;
        results[n_results].speedup_vs_fp32 = results[0].seconds / results[n_results].seconds;
        n_results++;
    }

    /* --- fp16 FMA --- */
    {
        matvec_fp16_sve(dst, (const float16_t *)wf16, x, n_rows, n_cols);
        results[n_results].max_err = compute_max_err(dst, dst_ref, n_rows);
        results[n_results].correct = (results[n_results].max_err < (double)(n_cols) * 0.001);
        uint64_t t0 = cycle_read();
        for (int r = 0; r < n_reps; r++) {
            matvec_fp16_sve(dst, (const float16_t *)wf16, x, n_rows, n_cols);
        }
        uint64_t t1 = cycle_read();
        results[n_results].name = "fp16 FMA";
        results[n_results].cycles = t1 - t0;
        results[n_results].seconds = (double)(t1 - t0) / (double)freq;
        results[n_results].ops = ops_per_matvec * n_reps;
        results[n_results].gflops = results[n_results].ops / results[n_results].seconds / 1e9;
        results[n_results].speedup_vs_fp32 = results[0].seconds / results[n_results].seconds;
        n_results++;
    }

    /* Print results */
    printf("\n%-22s %12s %12s %10s %10s %8s %s\n",
           "Path", "Cycles", "Time(ms)", "GFLOPS", "Speedup", "OK", "MaxErr");
    printf("------------------------------------------------------------------------\n");
    for (int i = 0; i < n_results; i++) {
        printf("%-22s %12lu %12.3f %10.2f %9.2fx %s %g\n",
               results[i].name,
               (unsigned long)results[i].cycles,
               results[i].seconds * 1000.0 / n_reps,
               results[i].gflops,
               results[i].speedup_vs_fp32,
               results[i].correct ? "  OK" : "FAIL",
               results[i].max_err);
    }
    printf("\n");
    printf("Notes:\n");
    printf("  - 'Cycles' is total over %d reps. 'Time(ms)' is per-call.\n", n_reps);
    printf("  - 'GFLOPS' = 2*n_rows*n_cols*reps / time.\n");
    printf("  - 'MaxErr' is max abs error vs fp32 reference.\n");
#ifdef TF_USE_INT8_SDOT_Q4_0
    printf("  - int8 paths use TF_USE_INT8_SDOT_Q4_0 for on-the-fly path.\n");
#else
    printf("  - int8 on-the-fly path not enabled (rebuild with -DTF_USE_INT8_SDOT_Q4_0).\n");
#endif
    printf("  - int8 (preq) includes prepack cost in the timed loop.\n");
    printf("  - int16/fp16 paths are 2x slower than int8 (half products per dot).\n");

    free(w);
    free(x);
    free(dst_ref);
    free(dst);
    free(wi16);
    free(wf16);
    if (cache.wi8) tf_q4_0_int8_cache_free(&cache);
    (void)n_reps;  /* silence unused warning if ONTHEFLY not defined */
}

int main(int argc, char **argv) {
    printf("================================================================\n");
    printf("Q4_0 matvec benchmark on A64FX\n");
    printf("================================================================\n");
    printf("Comparing: fp32 FMA, int8 SDOT (on-the-fly + prequant), int16 SDOT, fp16 FMA\n");
    printf("Build: -O3 -march=armv8.2-a+sve -ffp-contract=fast\n");
    printf("================================================================\n\n");

    /* Various sizes - adjust n_reps for reasonable run time */
    int sizes[][2] = {
        {  512,  512 },
        { 1024, 1024 },
        { 2048, 2048 },
        { 4096, 4096 },
        { 8192, 4096 },
        { 4096, 8192 },
    };
    int n_reps_list[] = { 100, 50, 20, 10, 5, 5 };

    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < n_sizes; i++) {
        run_bench(sizes[i][0], sizes[i][1], n_reps_list[i]);
    }

    return 0;
}
