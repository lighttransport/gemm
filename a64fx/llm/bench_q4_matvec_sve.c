/*
 * a64fx/llm/bench_q4_matvec_sve.c
 *
 * Standalone Q4_0 matvec benchmark comparing 1-row, 4-row, and 8-row SVE.
 *
 * Build (x86 cross for qlair):
 *   clang-21 --target=aarch64-linux-gnu -march=armv8.2-a+sve \
 *     -O3 -static -nostdlib -ffreestanding -o bench_q4_matvec_sve.elf \
 *     bench_q4_matvec_sve.c
 *
 * Run with qlair:
 *   ~/work/clair/main/build-arm64-sve-check/qlair -n 50M bench_q4_matvec_sve.elf
 */

typedef __SIZE_TYPE__ size_t;
typedef unsigned long uint64_t;

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/* Test status written before exit (visible to qlair) */
volatile unsigned int test_status __attribute__((section(".test_output")));

/* Block Q4_0 structure (matches ggml) */
typedef struct {
    unsigned short d;         /* fp16 scale */
    unsigned char  qs[16];    /* nibbles */
} block_q4_0;

/* IEEE-754 fp16 -> fp32 */
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

/* ================================================================
 * IMPLEMENTATION 1: Single-row SVE (baseline - current code)
 * ================================================================ */
static void vec_dot_q4_0_1row_sve(float *dst, const block_q4_0 *row,
                                   const float *x, int nb) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg_all = svptrue_b32();
    svfloat32_t acc = svdup_f32(0.0f);
    if (nb > 0) __builtin_prefetch(row->qs, 0, 0);
    for (int b = 0; b < nb; b++) {
        float d = fp16_to_fp32(row[b].d);
        int base = b * 32;
        /* Load 16 bytes from qs using svld1ub_u32 with full predicate */
        svuint32_t q = svld1ub_u32(pg_all, row[b].qs);
        /* Extract nibbles */
        svuint32_t lo = svand_n_u32_x(pg_all, q, 0x0f);
        svuint32_t hi = svlsr_n_u32_x(pg_all, q, 4);
        /* Dequant: (nibble - 8) * d */
        svint32_t ilo = svsub_n_s32_x(pg_all, svreinterpret_s32_u32(lo), 8);
        svint32_t ihi = svsub_n_s32_x(pg_all, svreinterpret_s32_u32(hi), 8);
        /* Convert to float and multiply by scale */
        svfloat32_t wlo = svmul_n_f32_x(pg_all, svcvt_f32_s32_x(pg_all, ilo), d);
        svfloat32_t whi = svmul_n_f32_x(pg_all, svcvt_f32_s32_x(pg_all, ihi), d);
        /* FMA with activation */
        acc = svmla_x(pg_all, acc, wlo, svld1(pg_all, x + base));
        acc = svmla_x(pg_all, acc, whi, svld1(pg_all, x + base + 16));
        if (b + 1 < nb) __builtin_prefetch(row[b+1].qs, 0, 0);
    }
    dst[0] = svaddv_f32(pg_all, acc);
#else
    (void)dst; (void)row; (void)x; (void)nb;
#endif
}

/* ================================================================
 * IMPLEMENTATION 2: 4-row SVE (activation shared)
 * ================================================================ */
static void vec_dot_q4_0_4row_sve(float *dst,
                                   const block_q4_0 *r0, const block_q4_0 *r1,
                                   const block_q4_0 *r2, const block_q4_0 *r3,
                                   const float *x, int nb) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    if (nb > 0) {
        __builtin_prefetch(r0->qs, 0, 0);
        __builtin_prefetch(r1->qs, 0, 0);
        __builtin_prefetch(r2->qs, 0, 0);
        __builtin_prefetch(r3->qs, 0, 0);
    }
    for (int b = 0; b < nb; b++) {
        int base = b * 32;
        svfloat32_t x_lo = svld1(pg, x + base);
        svfloat32_t x_hi = svld1(pg, x + base + 16);

#define DOT4_ROW(acc, row) do { \
    float d__ = fp16_to_fp32(row[b].d); \
    svuint32_t q__ = svld1ub_u32(pg, row[b].qs); \
    svint32_t ql__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q__, 0x0f)), 8); \
    svint32_t qh__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q__, 4)), 8); \
    svfloat32_t wl__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, ql__), d__); \
    svfloat32_t wh__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qh__), d__); \
    acc = svmla_x(pg, acc, wl__, x_lo); \
    acc = svmla_x(pg, acc, wh__, x_hi); \
} while(0)

        DOT4_ROW(a0, r0); DOT4_ROW(a1, r1);
        DOT4_ROW(a2, r2); DOT4_ROW(a3, r3);
#undef DOT4_ROW
        if (b + 1 < nb) {
            __builtin_prefetch(r0[b+1].qs, 0, 0);
            __builtin_prefetch(r1[b+1].qs, 0, 0);
            __builtin_prefetch(r2[b+1].qs, 0, 0);
            __builtin_prefetch(r3[b+1].qs, 0, 0);
        }
    }
    dst[0] = svaddv_f32(pg, a0); dst[1] = svaddv_f32(pg, a1);
    dst[2] = svaddv_f32(pg, a2); dst[3] = svaddv_f32(pg, a3);
#else
    (void)dst; (void)r0; (void)r1; (void)r2; (void)r3; (void)x; (void)nb;
#endif
}

/* ================================================================
 * IMPLEMENTATION 3: 8-row SVE (activation shared)
 * ================================================================ */
static void vec_dot_q4_0_8row_sve(float *dst,
                                   const block_q4_0 *r0, const block_q4_0 *r1,
                                   const block_q4_0 *r2, const block_q4_0 *r3,
                                   const block_q4_0 *r4, const block_q4_0 *r5,
                                   const block_q4_0 *r6, const block_q4_0 *r7,
                                   const float *x, int nb) {
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
    svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
    svfloat32_t a4 = svdup_f32(0.0f), a5 = svdup_f32(0.0f);
    svfloat32_t a6 = svdup_f32(0.0f), a7 = svdup_f32(0.0f);
    svbool_t pg = svptrue_b32();
    if (nb > 0) {
        __builtin_prefetch(r0->qs, 0, 0);
        __builtin_prefetch(r1->qs, 0, 0);
        __builtin_prefetch(r2->qs, 0, 0);
        __builtin_prefetch(r3->qs, 0, 0);
        __builtin_prefetch(r4->qs, 0, 0);
        __builtin_prefetch(r5->qs, 0, 0);
        __builtin_prefetch(r6->qs, 0, 0);
        __builtin_prefetch(r7->qs, 0, 0);
    }
    for (int b = 0; b < nb; b++) {
        int base = b * 32;
        svfloat32_t x_lo = svld1(pg, x + base);
        svfloat32_t x_hi = svld1(pg, x + base + 16);

#define DOT8_ROW(acc, row) do { \
    float d__ = fp16_to_fp32(row[b].d); \
    svuint32_t q__ = svld1ub_u32(pg, row[b].qs); \
    svint32_t ql__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svand_n_u32_x(pg, q__, 0x0f)), 8); \
    svint32_t qh__ = svsub_n_s32_x(pg, svreinterpret_s32_u32(svlsr_n_u32_x(pg, q__, 4)), 8); \
    svfloat32_t wl__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, ql__), d__); \
    svfloat32_t wh__ = svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, qh__), d__); \
    acc = svmla_x(pg, acc, wl__, x_lo); \
    acc = svmla_x(pg, acc, wh__, x_hi); \
} while(0)

        DOT8_ROW(a0, r0); DOT8_ROW(a1, r1);
        DOT8_ROW(a2, r2); DOT8_ROW(a3, r3);
        DOT8_ROW(a4, r4); DOT8_ROW(a5, r5);
        DOT8_ROW(a6, r6); DOT8_ROW(a7, r7);
#undef DOT8_ROW
        if (b + 1 < nb) {
            __builtin_prefetch(r0[b+1].qs, 0, 0);
            __builtin_prefetch(r1[b+1].qs, 0, 0);
            __builtin_prefetch(r2[b+1].qs, 0, 0);
            __builtin_prefetch(r3[b+1].qs, 0, 0);
            __builtin_prefetch(r4[b+1].qs, 0, 0);
            __builtin_prefetch(r5[b+1].qs, 0, 0);
            __builtin_prefetch(r6[b+1].qs, 0, 0);
            __builtin_prefetch(r7[b+1].qs, 0, 0);
        }
    }
    dst[0]=svaddv_f32(pg,a0);dst[1]=svaddv_f32(pg,a1);
    dst[2]=svaddv_f32(pg,a2);dst[3]=svaddv_f32(pg,a3);
    dst[4]=svaddv_f32(pg,a4);dst[5]=svaddv_f32(pg,a5);
    dst[6]=svaddv_f32(pg,a6);dst[7]=svaddv_f32(pg,a7);
#else
    (void)dst; (void)r0; (void)r1; (void)r2; (void)r3;
    (void)r4; (void)r5; (void)r6; (void)r7; (void)x; (void)nb;
#endif
}

/* ================================================================
 * IMPLEMENTATION 4: 8-row SVE SDOT (int8 dot product)
 * Dequant Q4_0 to int8 in 4-block chunks, then use SVE SDOT against
 * pre-quantized int8 activations. Requires int8 x (e.g. per-tensor
 * PTQ with scale 64-256 for normalized activations).
 * ================================================================ */
typedef int8_t q4_int8_t;
static void dequant_8rows_int8_4blk(const block_q4_0 *w, q4_int8_t *dst, int n_cols) {
    svbool_t pg = svptrue_b8();
    svbool_t pg32 = svwhilelt_b8(0, 32);
    int nb = n_cols / 32;
    for (int r = 0; r < 8; r++) {
        const block_q4_0 *row = (const block_q4_0 *)((const char *)w + r * (nb * sizeof(block_q4_0)));
        q4_int8_t *drow = dst + r * n_cols;
        for (int b = 0; b < nb; b += 4) {
            svuint8_t q0=svld1_u8(pg,row[b].qs),q1=svld1_u8(pg,row[b+1].qs);
            svuint8_t q2=svld1_u8(pg,row[b+2].qs),q3=svld1_u8(pg,row[b+3].qs);
            svuint8_t lo0=svand_n_u8_x(pg,q0,0x0f),hi0=svlsr_n_u8_x(pg,q0,4);
            svuint8_t lo1=svand_n_u8_x(pg,q1,0x0f),hi1=svlsr_n_u8_x(pg,q1,4);
            svuint8_t lo2=svand_n_u8_x(pg,q2,0x0f),hi2=svlsr_n_u8_x(pg,q2,4);
            svuint8_t lo3=svand_n_u8_x(pg,q3,0x0f),hi3=svlsr_n_u8_x(pg,q3,4);
            svuint8_t wu0=svzip1_u8(lo0,hi0),wu1=svzip1_u8(lo1,hi1);
            svuint8_t wu2=svzip1_u8(lo2,hi2),wu3=svzip1_u8(lo3,hi3);
            svint8_t ws0=svsub_n_s8_x(pg,svreinterpret_s8_u8(wu0),8);
            svint8_t ws1=svsub_n_s8_x(pg,svreinterpret_s8_u8(wu1),8);
            svint8_t ws2=svsub_n_s8_x(pg,svreinterpret_s8_u8(wu2),8);
            svint8_t ws3=svsub_n_s8_x(pg,svreinterpret_s8_u8(wu3),8);
            svst1_s8(pg32,drow+(b+0)*32,ws0);svst1_s8(pg32,drow+(b+1)*32,ws1);
            svst1_s8(pg32,drow+(b+2)*32,ws2);svst1_s8(pg32,drow+(b+3)*32,ws3);
        }
    }
}
static void vec_dot_q4_0_8row_sdot(q4_int8_t *wbuf, const q4_int8_t *xi8, int *dst,
                                     const block_q4_0 *r0, const block_q4_0 *r1,
                                     const block_q4_0 *r2, const block_q4_0 *r3,
                                     const block_q4_0 *r4, const block_q4_0 *r5,
                                     const block_q4_0 *r6, const block_q4_0 *r7,
                                     int nb) {
    (void)wbuf; (void)xi8; (void)dst; (void)r0; (void)r1; (void)r2; (void)r3;
    (void)r4; (void)r5; (void)r6; (void)r7; (void)nb;
#if defined(__ARM_FEATURE_SVE)
    dequant_8rows_int8_4blk(r0, wbuf, nb*32);
    svbool_t pg32 = svwhilelt_b8(0, 32);
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
    svint32_t a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
    for (int b = 0; b < nb; b++) {
        svint8_t xv = svld1_s8(pg32, xi8 + b*32);
        svint8_t w0=svld1_s8(pg32, wbuf+0*nb*32+b*32);
        svint8_t w1=svld1_s8(pg32, wbuf+1*nb*32+b*32);
        svint8_t w2=svld1_s8(pg32, wbuf+2*nb*32+b*32);
        svint8_t w3=svld1_s8(pg32, wbuf+3*nb*32+b*32);
        svint8_t w4=svld1_s8(pg32, wbuf+4*nb*32+b*32);
        svint8_t w5=svld1_s8(pg32, wbuf+5*nb*32+b*32);
        svint8_t w6=svld1_s8(pg32, wbuf+6*nb*32+b*32);
        svint8_t w7=svld1_s8(pg32, wbuf+7*nb*32+b*32);
        a0=svdot_s32(a0, w0, xv); a1=svdot_s32(a1, w1, xv);
        a2=svdot_s32(a2, w2, xv); a3=svdot_s32(a3, w3, xv);
        a4=svdot_s32(a4, w4, xv); a5=svdot_s32(a5, w5, xv);
        a6=svdot_s32(a6, w6, xv); a7=svdot_s32(a7, w7, xv);
    }
    dst[0] = svaddv_s32(svptrue_b32(), a0);
    dst[1] = svaddv_s32(svptrue_b32(), a1);
    dst[2] = svaddv_s32(svptrue_b32(), a2);
    dst[3] = svaddv_s32(svptrue_b32(), a3);
    dst[4] = svaddv_s32(svptrue_b32(), a4);
    dst[5] = svaddv_s32(svptrue_b32(), a5);
    dst[6] = svaddv_s32(svptrue_b32(), a6);
    dst[7] = svaddv_s32(svptrue_b32(), a7);
#endif
}

/* ================================================================
 * Matvec drivers
 * ================================================================ */
static void matvec_1row(float *dst, const unsigned char *base,
                         size_t rb, const float *x, int nb, int rs, int re) {
    for (int i = rs; i < re; i++)
        vec_dot_q4_0_1row_sve(dst + i,
            (const block_q4_0 *)(base + (size_t)i * rb), x, nb);
}

static void matvec_4row(float *dst, const unsigned char *base,
                         size_t rb, const float *x, int nb, int rs, int re) {
    int i = rs;
    for (; i + 3 < re; i += 4)
        vec_dot_q4_0_4row_sve(dst + i,
            (const block_q4_0 *)(base + (size_t)(i)*rb),
            (const block_q4_0 *)(base + (size_t)(i+1)*rb),
            (const block_q4_0 *)(base + (size_t)(i+2)*rb),
            (const block_q4_0 *)(base + (size_t)(i+3)*rb), x, nb);
    for (; i < re; i++)
        vec_dot_q4_0_1row_sve(dst + i,
            (const block_q4_0 *)(base + (size_t)i * rb), x, nb);
}

static void matvec_8row(float *dst, const unsigned char *base,
                         size_t rb, const float *x, int nb, int rs, int re) {
    int i = rs;
    for (; i + 7 < re; i += 8)
        vec_dot_q4_0_8row_sve(dst + i,
            (const block_q4_0 *)(base + (size_t)(i)*rb),
            (const block_q4_0 *)(base + (size_t)(i+1)*rb),
            (const block_q4_0 *)(base + (size_t)(i+2)*rb),
            (const block_q4_0 *)(base + (size_t)(i+3)*rb),
            (const block_q4_0 *)(base + (size_t)(i+4)*rb),
            (const block_q4_0 *)(base + (size_t)(i+5)*rb),
            (const block_q4_0 *)(base + (size_t)(i+6)*rb),
            (const block_q4_0 *)(base + (size_t)(i+7)*rb), x, nb);
    for (; i < re; i++)
        vec_dot_q4_0_1row_sve(dst + i,
            (const block_q4_0 *)(base + (size_t)i * rb), x, nb);
}

/* Scalar reference */
static float vec_dot_q4_0_scalar(const block_q4_0 *row, const float *x, int nb) {
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

/* ================================================================
 * _start entry point (freestanding, SVC #0 exit)
 * ================================================================ */
void _start(void) {
    int n_cols = 2048;  /* Gemma4 31B hidden dim */
    int n_rows = 128;  /* must fit on freestanding stack */
    int nb = n_cols / 32;
    size_t rb = (size_t)nb * sizeof(block_q4_0);

    /* Stack-allocated arrays (must fit within 256KB stack) */
    unsigned char wbuf[128 * 64 * sizeof(block_q4_0)];  /* 128 * 1152 = 144KB */
    float xbuf[2048];
    float ref[128];
    float out[128];

    block_q4_0 *weights = (block_q4_0 *)wbuf;
    float *x = xbuf;
    float *dst_ref = ref;
    float *dst_out = out;

    /* Fill with deterministic pseudo-random data
     * Scale: generate reasonable fp16 values (exp in [1..30]) */
    unsigned int seed = 42;
    for (int i = 0; i < n_rows * nb; i++) {
        seed = seed * 1103515245u + 12345u;
        unsigned short exp = (unsigned short)((seed >> 20) & 0x1F);
        if (exp == 0) exp = 1;       /* avoid zero/subnormal */
        if (exp == 31) exp = 30;     /* avoid inf/nan */
        unsigned short mant = (unsigned short)((seed >> 10) & 0x3FF);
        weights[i].d = exp << 10 | mant;  /* valid fp16, positive */
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

    /* Scalar reference */
    for (int r = 0; r < n_rows; r++)
        dst_ref[r] = vec_dot_q4_0_scalar(
            (const block_q4_0 *)(wbuf + (size_t)r * rb), x, nb);

    /* === Run 1-row (store for comparison) === */
    matvec_1row(dst_out, wbuf, rb, x, nb, 0, n_rows);
    int ok1 = 1;
    float max_err1 = 0.0f;
    for (int r = 0; r < n_rows; r++) {
        float d = dst_out[r] - dst_ref[r];
        if (d < 0) d = -d;
        if (d > max_err1) max_err1 = d;
        if (d > 0.5f) { ok1 = 0; }
    }
    /* Copy 1-row results to ref for subsequent comparisons */
    for (int r = 0; r < n_rows; r++) dst_ref[r] = dst_out[r];

    /* === Benchmark: 4-row (compare against 1-row) === */
    matvec_4row(dst_out, wbuf, rb, x, nb, 0, n_rows);
    int ok4 = 1;
    float max_err4 = 0.0f;
    for (int r = 0; r < n_rows; r++) {
        float d = dst_out[r] - dst_ref[r];
        if (d < 0) d = -d;
        if (d > max_err4) max_err4 = d;
        if (d > 1.0e-2f) { ok4 = 0; }
    }

    /* === Benchmark: 8-row (compare against 1-row) === */
    matvec_8row(dst_out, wbuf, rb, x, nb, 0, n_rows);
    int ok8 = 1;
    float max_err8 = 0.0f;
    for (int r = 0; r < n_rows; r++) {
        float d = dst_out[r] - dst_ref[r];
        if (d < 0) d = -d;
        if (d > max_err8) max_err8 = d; if (d > 1.0e-2f) { ok8 = 0; }
    }

    /* Encode results: lower 16 bits = approximate max error * 1e6 */
    unsigned int err_code = (unsigned int)(max_err1 * 1e6f);
    if (err_code > 0xFFFF) err_code = 0xFFFF;
    test_status = (ok1 ? 0x00010000u : 0) | (ok4 ? 0x00000100u : 0) | (ok8 ? 0x00000001u : 0) | err_code;

    /* Count large mismatches (threshold 0.1) */
    int mismatches = 0;
    for (int r = 0; r < n_rows; r++) {
        float d = dst_out[r] - dst_ref[r];
        if (d < 0) d = -d;
        if (d > 0.1f) mismatches++;
    }
    uint64_t exit_code = (uint64_t)(mismatches > 254 ? 254 : mismatches);

    register uint64_t x0 asm("x0") = exit_code;
    register uint64_t x8 asm("x8") = 93;  /* __NR_exit */
    __asm__ volatile("svc #0" : : "r"(x0), "r"(x8) : "memory");

    /* Never reached */
    while (1) __asm__ volatile("wfi");
}
