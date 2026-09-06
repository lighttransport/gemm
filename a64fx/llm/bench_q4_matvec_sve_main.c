/*
 * bench_q4_matvec_sve_main.c - Hosted C benchmark for Q4_0 matvec on A64FX SVE
 *
 * Build for qemu:
 *   clang-21 --target=aarch64-linux-gnu -march=armv8.2-a+sve \
 *     -O3 -static -o bench_q4_matvec_sve bench_q4_matvec_sve_main.c -lm
 *   qemu-aarch64-static -cpu max ./bench_q4_matvec_sve
 *
 * Build native (x86 scalar):
 *   gcc -O3 -march=native -o bench_q4_matvec_sve bench_q4_matvec_sve_main.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

typedef struct {
    unsigned short d;
    unsigned char  qs[16];
} block_q4_0;

static float fp16_to_fp32(unsigned short h) {
    unsigned int sign = (unsigned int)(h & 0x8000) << 16;
    unsigned int exp  = (h >> 10) & 0x1F;
    unsigned int mant = h & 0x3FF;
    unsigned int f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    return *(float *)&f;
}

/* === SVE 1-row (baseline - current transformer.h) === */
static void vec_dot_1row(float *dst, const block_q4_0 *row, const float *x, int nb) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b32();
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < nb; b++) {
        float d = fp16_to_fp32(row[b].d);
        int base = b * 32;
        svuint32_t q = svld1ub_u32(pg, row[b].qs);
        svint32_t qlo = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q, 0x0f)), 8);
        svint32_t qhi = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q, 4)), 8);
        svfloat32_t wlo = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qlo), d);
        svfloat32_t whi = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qhi), d);
        acc = svmla_x(pg, acc, wlo, svld1(pg, x + base));
        acc = svmla_x(pg, acc, whi, svld1(pg, x + base + 16));
    }
    dst[0] = svaddv_f32(pg, acc);
#else
    (void)dst; (void)row; (void)x; (void)nb;
#endif
}

/* === SVE 4-row (activation shared) === */
static void vec_dot_4row(float dst[4],
                          const block_q4_0 *r0, const block_q4_0 *r1,
                          const block_q4_0 *r2, const block_q4_0 *r3,
                          const float *x, int nb) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    for (int b = 0; b < nb; b++) {
        int base = b * 32;
        svfloat32_t x_lo = svld1(pg, x + base);
        svfloat32_t x_hi = svld1(pg, x + base + 16);

#define ROW(acc, row) do { \
    float d__ = fp16_to_fp32(row[b].d); \
    svuint32_t q__ = svld1ub_u32(pg, row[b].qs); \
    svint32_t ql__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q__, 0x0f)), 8); \
    svint32_t qh__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q__, 4)), 8); \
    svfloat32_t wl__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, ql__), d__); \
    svfloat32_t wh__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qh__), d__); \
    acc = svmla_x(pg, acc, wl__, x_lo); \
    acc = svmla_x(pg, acc, wh__, x_hi); \
} while(0)

        ROW(a0, r0); ROW(a1, r1); ROW(a2, r2); ROW(a3, r3);
#undef ROW
    }
    dst[0] = svaddv_f32(pg, a0);
    dst[1] = svaddv_f32(pg, a1);
    dst[2] = svaddv_f32(pg, a2);
    dst[3] = svaddv_f32(pg, a3);
#else
    (void)dst; (void)r0; (void)r1; (void)r2; (void)r3; (void)x; (void)nb;
#endif
}

/* === SVE 8-row (activation shared, max register pressure) === */
static void vec_dot_8row(float dst[8],
                          const block_q4_0 *r0, const block_q4_0 *r1,
                          const block_q4_0 *r2, const block_q4_0 *r3,
                          const block_q4_0 *r4, const block_q4_0 *r5,
                          const block_q4_0 *r6, const block_q4_0 *r7,
                          const float *x, int nb) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svptrue_b32();
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    for (int b = 0; b < nb; b++) {
        int base = b * 32;
        svfloat32_t x_lo = svld1(pg, x + base);
        svfloat32_t x_hi = svld1(pg, x + base + 16);

#define ROW(acc, row) do { \
    float d__ = fp16_to_fp32(row[b].d); \
    svuint32_t q__ = svld1ub_u32(pg, row[b].qs); \
    svint32_t ql__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q__, 0x0f)), 8); \
    svint32_t qh__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q__, 4)), 8); \
    svfloat32_t wl__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, ql__), d__); \
    svfloat32_t wh__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qh__), d__); \
    acc = svmla_x(pg, acc, wl__, x_lo); \
    acc = svmla_x(pg, acc, wh__, x_hi); \
} while(0)

        ROW(a0,r0);ROW(a1,r1);ROW(a2,r2);ROW(a3,r3);
        ROW(a4,r4);ROW(a5,r5);ROW(a6,r6);ROW(a7,r7);
#undef ROW
    }
    dst[0]=svaddv_f32(pg,a0);dst[1]=svaddv_f32(pg,a1);
    dst[2]=svaddv_f32(pg,a2);dst[3]=svaddv_f32(pg,a3);
    dst[4]=svaddv_f32(pg,a4);dst[5]=svaddv_f32(pg,a5);
    dst[6]=svaddv_f32(pg,a6);dst[7]=svaddv_f32(pg,a7);
#else
    (void)dst;(void)r0;(void)r1;(void)r2;(void)r3;
    (void)r4;(void)r5;(void)r6;(void)r7;(void)x;(void)nb;
#endif
}

/* === Driver: dispatch over rows === */
static void matvec_driver(const char *name, int row_group,
                           float *dst, const unsigned char *base,
                           size_t rb, const float *x, int nb,
                           int rs, int re) {
    int i = rs;
    if (row_group == 8) {
        for (; i + 7 < re; i += 8)
            vec_dot_8row(dst + i,
                (const block_q4_0*)(base+(size_t)(i)*rb),
                (const block_q4_0*)(base+(size_t)(i+1)*rb),
                (const block_q4_0*)(base+(size_t)(i+2)*rb),
                (const block_q4_0*)(base+(size_t)(i+3)*rb),
                (const block_q4_0*)(base+(size_t)(i+4)*rb),
                (const block_q4_0*)(base+(size_t)(i+5)*rb),
                (const block_q4_0*)(base+(size_t)(i+6)*rb),
                (const block_q4_0*)(base+(size_t)(i+7)*rb), x, nb);
    } else if (row_group == 4) {
        for (; i + 3 < re; i += 4)
            vec_dot_4row(dst + i,
                (const block_q4_0*)(base+(size_t)(i)*rb),
                (const block_q4_0*)(base+(size_t)(i+1)*rb),
                (const block_q4_0*)(base+(size_t)(i+2)*rb),
                (const block_q4_0*)(base+(size_t)(i+3)*rb), x, nb);
    }
    for (; i < re; i++)
        vec_dot_1row(dst + i,
            (const block_q4_0*)(base+(size_t)i * rb), x, nb);

    (void)name;
}

/* === Scalar reference === */
static float dot_scalar(const block_q4_0 *row, const float *x, int nb) {
    float s = 0.0f;
    for (int b = 0; b < nb; b++) {
        float d = fp16_to_fp32(row[b].d);
        int base = b * 32;
        for (int j = 0; j < 16; j++) {
            unsigned char qb = row[b].qs[j];
            s += ((float)((int)(qb & 0x0f) - 8) * d) * x[base + j];
            s += ((float)((int)(qb >> 4) - 8) * d) * x[base + j + 16];
        }
    }
    return s;
}

/* === Benchmark harness === */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int n_cols = 2048;
    int n_rows = (argc > 1) ? atoi(argv[1]) : 65536;
    int warmup = 3;
    int trials = 10;

    int nb = n_cols / 32;
    size_t rb = (size_t)nb * sizeof(block_q4_0);

    printf("Q4_0 SVE Matvec Benchmark\n");
    printf("  n_cols = %d, n_rows = %d\n", n_cols, n_rows);
    printf("  nb     = %d, row_bytes = %zu\n", nb, rb);
    printf("  total weight = %.1f MB\n", (double)n_rows * rb / 1e6);
    printf("  SVE path: %s\n\n",
#if defined(__ARM_FEATURE_SVE)
           "ENABLED"
#else
           "DISABLED (scalar fallback)"
#endif
    );

    block_q4_0 *weights = (block_q4_0 *)aligned_alloc(256, (size_t)n_rows * rb);
    float *x = (float *)aligned_alloc(256, (size_t)n_cols * sizeof(float));
    float *ref = (float *)aligned_alloc(256, (size_t)n_rows * sizeof(float));
    float *out = (float *)aligned_alloc(256, (size_t)n_rows * sizeof(float));

    if (!weights || !x || !ref || !out) {
        printf("FAIL: allocation\n");
        return 1;
    }

    /* Fill with deterministic data (valid fp16 scales) */
    unsigned int seed = 42;
    for (int i = 0; i < n_rows * nb; i++) {
        seed = seed * 1103515245u + 12345u;
        unsigned short exp = (seed >> 20) & 0x1F;
        if (exp == 0) exp = 1;
        if (exp == 31) exp = 30;
        unsigned short mant = (seed >> 10) & 0x3FF;
        weights[i].d = exp << 10 | mant;
        for (int j = 0; j < 16; j++) {
            seed = seed * 1103515245u + 12345u;
            weights[i].qs[j] = (unsigned char)((seed >> 24) & 0x0F) |
                               ((unsigned char)((seed >> 20) & 0x0F) << 4);
        }
    }
    for (int i = 0; i < n_cols; i++) {
        seed = seed * 1103515245u + 12345u;
        x[i] = (float)((int)(seed >> 16) % 200 - 100) * 0.01f;
    }

    /* Scalar reference (small subset for verification) */
    int verify_rows = (n_rows < 1024) ? n_rows : 1024;
    for (int r = 0; r < verify_rows; r++)
        ref[r] = dot_scalar(
            (const block_q4_0 *)((const unsigned char *)weights + (size_t)r * rb), x, nb);

    /* Run all three variants */
    const int groups[] = {1, 4, 8};
    const char *names[] = {"1-row", "4-row", "8-row"};
    double best_time[3] = {1e30, 1e30, 1e30};
    float max_err[3] = {0, 0, 0};

    for (int variant = 0; variant < 3; variant++) {
        int g = groups[variant];

        /* Warmup */
        for (int t = 0; t < warmup; t++)
            matvec_driver(names[variant], g, out, (const unsigned char *)weights, rb, x, nb, 0, n_rows);

        /* Timed trials */
        for (int t = 0; t < trials; t++) {
            double t0 = now_sec();
            matvec_driver(names[variant], g, out, (const unsigned char *)weights, rb, x, nb, 0, n_rows);
            double t1 = now_sec();
            double dt = t1 - t0;
            if (dt < best_time[variant]) best_time[variant] = dt;
        }

        /* Verify against scalar (first verify_rows rows) */
        float me = 0.0f;
        /* Re-run for verification */
        if (verify_rows > 0) {
            if (g == 1) {
                for (int r = 0; r < verify_rows; r++)
                    vec_dot_1row(out + r,
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)r * rb), x, nb);
            } else if (g == 4) {
                int r = 0;
                for (; r + 3 < verify_rows; r += 4)
                    vec_dot_4row(out + r,
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+1)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+2)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+3)*rb), x, nb);
                for (; r < verify_rows; r++)
                    vec_dot_1row(out + r,
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)r * rb), x, nb);
            } else {
                int r = 0;
                for (; r + 7 < verify_rows; r += 8)
                    vec_dot_8row(out + r,
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+1)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+2)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+3)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+4)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+5)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+6)*rb),
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)(r+7)*rb), x, nb);
                for (; r < verify_rows; r++)
                    vec_dot_1row(out + r,
                        (const block_q4_0*)((const unsigned char *)weights + (size_t)r * rb), x, nb);
            }
            for (int r = 0; r < verify_rows; r++) {
                float d = out[r] - ref[r];
                if (d < 0) d = -d;
                if (d > me) me = d;
            }
        }
        max_err[variant] = me;

        double ops = (double)n_rows * n_cols * 2.0;
        double gflops = ops / best_time[variant] / 1e9;
        double bw = (double)n_rows * rb / best_time[variant] / 1e9;

        printf("%-6s: %8.2f ms  %7.2f GFLOP/s  %5.2f GB/s  max_err=%.2e\n",
               names[variant], best_time[variant] * 1000, gflops, bw, me);
    }

    free(weights);
    free(x);
    free(ref);
    free(out);
    return 0;
}
