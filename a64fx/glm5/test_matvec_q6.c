#define _GNU_SOURCE
/* Custom A64FX Q6 experiment.
 *
 * Per Q6_GROUP_SIZE weights: 6 packed bits/weight plus one f32 scale.
 * The default group 128 is 100 B (0.78125 B/weight including metadata).
 * Codes are offset-binary 0..63 for signed quants -32..31. The kernel dots
 * unsigned codes directly with int16 activations and folds -32*sum(xq).
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *     -Wall -Wextra -Wpedantic -I../../common test_matvec_q6.c \
 *     matvec_q6_a8.S -lm -o test_matvec_q6
 */
#include <arm_sve.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "glm5_int8.h"

#ifndef Q6_GROUP_SIZE
#define Q6_GROUP_SIZE 128
#endif

enum {
    Q6_COLS = 6144, Q6_ROWS = 2048, Q6_GS = Q6_GROUP_SIZE,
    Q6_CHUNKS = Q6_GS / 64, Q6_NB = Q6_COLS / Q6_GS
};

_Static_assert(Q6_GS == 64 || Q6_GS == 128 || Q6_GS == 256,
        "supported Q6 groups are 64, 128, and 256");

#ifndef Q6_PAD_BYTES
#define Q6_PAD_BYTES 0
#endif

typedef struct {
    uint8_t ql[Q6_GS / 2];
    uint8_t qh[Q6_GS / 4];
    float d;
#if Q6_PAD_BYTES > 0
    uint8_t pad[Q6_PAD_BYTES];
#endif
} q6_block;

_Static_assert(sizeof(q6_block) == 3 * Q6_GS / 4 + 4 + Q6_PAD_BYTES,
        "unexpected q6_block padding");

static volatile double g_sink;

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1.0e-9;
}

static size_t env_size(const char *name, size_t def) {
    const char *s = getenv(name);
    if (!s || !*s) return def;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 0);
    return errno || end == s || *end ? def : (size_t)v;
}

static int env_int(const char *name, int def) {
    const char *s = getenv(name);
    if (!s || !*s) return def;
    char *end = NULL;
    long v = strtol(s, &end, 0);
    return end == s || *end ? def : (int)v;
}

static uint32_t lcg32(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s;
}

static void q6_pack_codes(q6_block *b, const uint8_t code[Q6_GS], float d) {
    memset(b, 0, sizeof(*b));
    for (int i = 0; i < Q6_GS / 2; i++)
        b->ql[i] = (uint8_t)((code[2 * i] & 15u) | ((code[2 * i + 1] & 15u) << 4));
    for (int i = 0; i < Q6_GS; i++)
        b->qh[i >> 2] |= (uint8_t)(((code[i] >> 4) & 3u) << (2 * (i & 3)));
    b->d = d;
}

static void q6_unpack_codes(const q6_block *b, uint8_t code[Q6_GS]) {
    for (int i = 0; i < Q6_GS / 2; i++) {
        code[2 * i] = b->ql[i] & 15u;
        code[2 * i + 1] = b->ql[i] >> 4;
    }
    for (int i = 0; i < Q6_GS; i++)
        code[i] |= (uint8_t)(((b->qh[i >> 2] >> (2 * (i & 3))) & 3u) << 4);
}

static inline svuint8_t q6_decode_u8(const q6_block *b,
        int sub, svuint8_t qh_index, svuint8_t qh_shift) {
    const svbool_t pb = svptrue_b8();
    const svbool_t p16 = svwhilelt_b8((uint64_t)0, (uint64_t)16);
    const svbool_t p32 = svwhilelt_b8((uint64_t)0, (uint64_t)32);
    const svuint8_t ql = svld1_u8(p32, b->ql + sub * 32);
    const svuint8_t low = svzip1_u8(svand_n_u8_x(pb, ql, 15), svlsr_n_u8_x(pb, ql, 4));
    const svuint8_t qh_packed = svld1_u8(p16, b->qh + sub * 16);
    svuint8_t qh = svtbl_u8(qh_packed, qh_index);
    qh = svand_n_u8_x(pb, svlsr_u8_x(pb, qh, qh_shift), 3);
    return svorr_u8_x(pb, low, svlsl_n_u8_x(pb, qh, 4));
}

static inline void q6_decode_pair(const q6_block *b,
        int sub, svuint8_t qh_index, svuint8_t qh_shift,
        svint16_t *lo, svint16_t *hi) {
    const svuint8_t q = q6_decode_u8(b, sub, qh_index, qh_shift);
    *lo = svreinterpret_s16_u16(svunpklo_u16(q));
    *hi = svreinterpret_s16_u16(svunpkhi_u16(q));
}

static void q6_quantize_x8(const int16_t *restrict x16,
        int8_t *restrict x8, float *restrict xd, int nb) {
    for (int g = 0; g < nb; g++) {
        int amax = 0;
        for (int i = 0; i < Q6_GS; i++) {
            const int v = x16[g * Q6_GS + i];
            const int av = v < 0 ? -v : v;
            if (av > amax) amax = av;
        }
        const float d = amax ? (float)amax / 127.0f : 0.0f;
        const float inv = amax ? 127.0f / (float)amax : 0.0f;
        for (int i = 0; i < Q6_GS; i++) {
            int q = (int)lrintf((float)x16[g * Q6_GS + i] * inv);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            x8[g * Q6_GS + i] = (int8_t)q;
        }
        xd[g] = d;
    }
}

__attribute__((noinline))
static void q6_matvec_8row(float *restrict dst,
        const q6_block *w0, const q6_block *w1, const q6_block *w2, const q6_block *w3,
        const q6_block *w4, const q6_block *w5, const q6_block *w6, const q6_block *w7,
        const int16_t *restrict xq, const int64_t *restrict xgsum, int nb) {
    const svbool_t pd = svptrue_b64();
    const svbool_t ph = svptrue_b16();
    const svbool_t pb = svptrue_b8();
    const svuint8_t lane = svindex_u8(0, 1);
    const svuint8_t qh_index = svlsr_n_u8_x(pb, lane, 2);
    const svuint8_t qh_shift = svlsl_n_u8_x(pb, svand_n_u8_x(pb, lane, 3), 1);
    svfloat64_t a0 = svdup_f64(0), a1 = svdup_f64(0), a2 = svdup_f64(0), a3 = svdup_f64(0);
    svfloat64_t a4 = svdup_f64(0), a5 = svdup_f64(0), a6 = svdup_f64(0), a7 = svdup_f64(0);
    double bc0 = 0, bc1 = 0, bc2 = 0, bc3 = 0, bc4 = 0, bc5 = 0, bc6 = 0, bc7 = 0;
    for (int g = 0; g < nb; g++) {
        svint64_t d0 = svdup_s64(0), d1 = svdup_s64(0), d2 = svdup_s64(0), d3 = svdup_s64(0);
        svint64_t d4 = svdup_s64(0), d5 = svdup_s64(0), d6 = svdup_s64(0), d7 = svdup_s64(0);
        for (int sub = 0; sub < Q6_CHUNKS; sub++) {
            const svint16_t xl = svld1_s16(ph, xq + g * Q6_GS + sub * 64);
            const svint16_t xh = svld1_s16(ph, xq + g * Q6_GS + sub * 64 + 32);
#define Q6_DOT(B, D) do { \
            svint16_t ql_, qh_; \
            q6_decode_pair((B) + g, sub, qh_index, qh_shift, &ql_, &qh_); \
            D = svdot_s64(D, ql_, xl); \
            D = svdot_s64(D, qh_, xh); \
        } while (0)
            Q6_DOT(w0, d0); Q6_DOT(w1, d1); Q6_DOT(w2, d2); Q6_DOT(w3, d3);
            Q6_DOT(w4, d4); Q6_DOT(w5, d5); Q6_DOT(w6, d6); Q6_DOT(w7, d7);
#undef Q6_DOT
        }
        const double sc0 = w0[g].d, sc1 = w1[g].d, sc2 = w2[g].d, sc3 = w3[g].d;
        const double sc4 = w4[g].d, sc5 = w5[g].d, sc6 = w6[g].d, sc7 = w7[g].d;
        const double xg = (double)xgsum[g];
        a0 = svmla_n_f64_x(pd, a0, svcvt_f64_s64_x(pd, d0), sc0);
        a1 = svmla_n_f64_x(pd, a1, svcvt_f64_s64_x(pd, d1), sc1);
        a2 = svmla_n_f64_x(pd, a2, svcvt_f64_s64_x(pd, d2), sc2);
        a3 = svmla_n_f64_x(pd, a3, svcvt_f64_s64_x(pd, d3), sc3);
        a4 = svmla_n_f64_x(pd, a4, svcvt_f64_s64_x(pd, d4), sc4);
        a5 = svmla_n_f64_x(pd, a5, svcvt_f64_s64_x(pd, d5), sc5);
        a6 = svmla_n_f64_x(pd, a6, svcvt_f64_s64_x(pd, d6), sc6);
        a7 = svmla_n_f64_x(pd, a7, svcvt_f64_s64_x(pd, d7), sc7);
        bc0 += sc0 * xg; bc1 += sc1 * xg; bc2 += sc2 * xg; bc3 += sc3 * xg;
        bc4 += sc4 * xg; bc5 += sc5 * xg; bc6 += sc6 * xg; bc7 += sc7 * xg;
    }
    dst[0] = (float)(svaddv_f64(pd, a0) - 32.0 * bc0);
    dst[1] = (float)(svaddv_f64(pd, a1) - 32.0 * bc1);
    dst[2] = (float)(svaddv_f64(pd, a2) - 32.0 * bc2);
    dst[3] = (float)(svaddv_f64(pd, a3) - 32.0 * bc3);
    dst[4] = (float)(svaddv_f64(pd, a4) - 32.0 * bc4);
    dst[5] = (float)(svaddv_f64(pd, a5) - 32.0 * bc5);
    dst[6] = (float)(svaddv_f64(pd, a6) - 32.0 * bc6);
    dst[7] = (float)(svaddv_f64(pd, a7) - 32.0 * bc7);
}

__attribute__((noinline))
static void q6_matvec_a8_8row(float *restrict dst,
        const q6_block *w0, const q6_block *w1, const q6_block *w2, const q6_block *w3,
        const q6_block *w4, const q6_block *w5, const q6_block *w6, const q6_block *w7,
        const int8_t *restrict xq, const float *restrict xd, int nb) {
    const svbool_t pf = svptrue_b32();
    const svbool_t pb = svptrue_b8();
    const svuint8_t lane = svindex_u8(0, 1);
    const svuint8_t qh_index = svlsr_n_u8_x(pb, lane, 2);
    const svuint8_t qh_shift = svlsl_n_u8_x(pb, svand_n_u8_x(pb, lane, 3), 1);
    svfloat32_t a0 = svdup_f32(0), a1 = svdup_f32(0), a2 = svdup_f32(0), a3 = svdup_f32(0);
    svfloat32_t a4 = svdup_f32(0), a5 = svdup_f32(0), a6 = svdup_f32(0), a7 = svdup_f32(0);
    for (int g = 0; g < nb; g++) {
        const float xsc = xd[g];
#define Q6_A8_ROW(B, A) do { \
            svint32_t di_ = svdup_s32(0); \
            for (int sub_ = 0; sub_ < Q6_CHUNKS; sub_++) { \
                const svint8_t xv_ = svld1_s8(pb, xq + g * Q6_GS + sub_ * 64); \
                const svuint8_t qu_ = q6_decode_u8((B) + g, sub_, qh_index, qh_shift); \
                const svint8_t qs_ = svreinterpret_s8_u8(svsub_n_u8_x(pb, qu_, 32)); \
                di_ = svdot_s32(di_, qs_, xv_); \
            } \
            const float sc_ = (B)[g].d * xsc; \
            A = svmla_n_f32_x(pf, A, svcvt_f32_s32_x(pf, di_), sc_); \
        } while (0)
        Q6_A8_ROW(w0, a0); Q6_A8_ROW(w1, a1);
        Q6_A8_ROW(w2, a2); Q6_A8_ROW(w3, a3);
        Q6_A8_ROW(w4, a4); Q6_A8_ROW(w5, a5);
        Q6_A8_ROW(w6, a6); Q6_A8_ROW(w7, a7);
#undef Q6_A8_ROW
    }
    dst[0] = svaddv_f32(pf, a0); dst[1] = svaddv_f32(pf, a1);
    dst[2] = svaddv_f32(pf, a2); dst[3] = svaddv_f32(pf, a3);
    dst[4] = svaddv_f32(pf, a4); dst[5] = svaddv_f32(pf, a5);
    dst[6] = svaddv_f32(pf, a6); dst[7] = svaddv_f32(pf, a7);
}

typedef struct {
    const q6_block *w[8];
    const int8_t *xq;
    const float *xd;
    int nb;
} q6_a8_asm_job;

extern void glm5_matvec_q6_a8_asm(float *dst, const q6_a8_asm_job *job);

__attribute__((noinline))
static void q6_matvec_a8_asm_8row(float *restrict dst,
        const q6_block *w0, const q6_block *w1, const q6_block *w2, const q6_block *w3,
        const q6_block *w4, const q6_block *w5, const q6_block *w6, const q6_block *w7,
        const int8_t *restrict xq, const float *restrict xd, int nb) {
    const q6_a8_asm_job job = {{w0, w1, w2, w3, w4, w5, w6, w7}, xq, xd, nb};
    glm5_matvec_q6_a8_asm(dst, &job);
}

static float q6_dot_scalar(const q6_block *row, const int16_t *xq, int nb) {
    double sum = 0;
    uint8_t code[Q6_GS];
    for (int g = 0; g < nb; g++) {
        q6_unpack_codes(row + g, code);
        int64_t dot = 0;
        for (int i = 0; i < Q6_GS; i++)
            dot += ((int)code[i] - 32) * (int)xq[g * Q6_GS + i];
        sum += (double)dot * row[g].d;
    }
    return (float)sum;
}

__attribute__((noinline))
static void raw_q6_block(float *dst, const uint8_t *p, int bytes) {
    const svbool_t pg = svptrue_b64();
    const int vb = (int)svcntb();
    svuint64_t a0 = svdup_u64(0), a1 = svdup_u64(0), a2 = svdup_u64(0), a3 = svdup_u64(0);
    svuint64_t a4 = svdup_u64(0), a5 = svdup_u64(0), a6 = svdup_u64(0), a7 = svdup_u64(0);
    int c = 0;
    for (; c + 8 * vb <= bytes; c += 8 * vb) {
        a0 = svadd_u64_x(pg, a0, svld1_u64(pg, (const uint64_t *)(p + c + 0 * vb)));
        a1 = svadd_u64_x(pg, a1, svld1_u64(pg, (const uint64_t *)(p + c + 1 * vb)));
        a2 = svadd_u64_x(pg, a2, svld1_u64(pg, (const uint64_t *)(p + c + 2 * vb)));
        a3 = svadd_u64_x(pg, a3, svld1_u64(pg, (const uint64_t *)(p + c + 3 * vb)));
        a4 = svadd_u64_x(pg, a4, svld1_u64(pg, (const uint64_t *)(p + c + 4 * vb)));
        a5 = svadd_u64_x(pg, a5, svld1_u64(pg, (const uint64_t *)(p + c + 5 * vb)));
        a6 = svadd_u64_x(pg, a6, svld1_u64(pg, (const uint64_t *)(p + c + 6 * vb)));
        a7 = svadd_u64_x(pg, a7, svld1_u64(pg, (const uint64_t *)(p + c + 7 * vb)));
    }
    uint64_t tail = 0;
    for (; c < bytes; c++) tail += p[c];
    dst[0] = (float)(svaddv_u64(pg, a0) + tail); dst[1] = (float)svaddv_u64(pg, a1);
    dst[2] = (float)svaddv_u64(pg, a2); dst[3] = (float)svaddv_u64(pg, a3);
    dst[4] = (float)svaddv_u64(pg, a4); dst[5] = (float)svaddv_u64(pg, a5);
    dst[6] = (float)svaddv_u64(pg, a6); dst[7] = (float)svaddv_u64(pg, a7);
}

static int quality_check(void) {
    enum { ROWS = 256, TRIALS = 8 };
    const size_t wcount = (size_t)ROWS * Q6_COLS;
    uint8_t *w8 = mmap(NULL, wcount, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *s8 = mmap(NULL, (size_t)ROWS * Q6_NB * sizeof(float), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    q6_block *w6 = mmap(NULL, (size_t)ROWS * Q6_NB * sizeof(q6_block), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int16_t *xq = mmap(NULL, (size_t)Q6_COLS * sizeof(int16_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int64_t *xg = mmap(NULL, (size_t)Q6_NB * sizeof(int64_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int8_t *x8 = mmap(NULL, (size_t)Q6_COLS * sizeof(int8_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *xd = mmap(NULL, (size_t)Q6_NB * sizeof(float), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (w8 == MAP_FAILED || s8 == MAP_FAILED || w6 == MAP_FAILED ||
            xq == MAP_FAILED || xg == MAP_FAILED || x8 == MAP_FAILED ||
            xd == MAP_FAILED) return 2;

    uint32_t rng = 7;
    double werr2 = 0, wref2 = 0;
    for (int r = 0; r < ROWS; r++) {
        for (int g = 0; g < Q6_NB; g++) {
            int raw[Q6_GS], amax = 1;
            for (int i = 0; i < Q6_GS; i++) {
                int v = 0;
                for (int k = 0; k < 4; k++) v += (int)(lcg32(&rng) >> 24);
                raw[i] = v - 510;
                if (abs(raw[i]) > amax) amax = abs(raw[i]);
            }
            const float base_d = (float)(1 + (lcg32(&rng) & 2047)) * 5.0e-7f;
            uint8_t code[Q6_GS];
            for (int i = 0; i < Q6_GS; i++) {
                int q8 = (int)lrintf((float)raw[i] * 127.0f / amax);
                if (q8 > 127) q8 = 127; if (q8 < -127) q8 = -127;
                w8[(size_t)r * Q6_COLS + g * Q6_GS + i] = (uint8_t)(q8 + 128);
            }
            s8[(size_t)r * Q6_NB + g] = base_d;
            int qamax = 1;
            for (int i = 0; i < Q6_GS; i++) {
                int q8 = (int)w8[(size_t)r * Q6_COLS + g * Q6_GS + i] - 128;
                if (abs(q8) > qamax) qamax = abs(q8);
            }
            const float d6 = base_d * (float)qamax / 31.0f;
            for (int i = 0; i < Q6_GS; i++) {
                int q8 = (int)w8[(size_t)r * Q6_COLS + g * Q6_GS + i] - 128;
                int q6 = (int)lrintf((float)q8 * 31.0f / qamax);
                if (q6 > 31) q6 = 31; if (q6 < -31) q6 = -31;
                code[i] = (uint8_t)(q6 + 32);
                const double ref = (double)q8 * base_d;
                const double got = (double)q6 * d6;
                werr2 += (got - ref) * (got - ref);
                wref2 += ref * ref;
            }
            q6_pack_codes(w6 + (size_t)r * Q6_NB + g, code, d6);
        }
    }

    double a16_err2 = 0, a8_err2 = 0, out_ref2 = 0;
    double impl_err2 = 0, impl_ref2 = 0, asm_err2 = 0;
    double a16_max = 0, a8_max = 0, impl_max = 0, asm_max = 0;
    for (int tr = 0; tr < TRIALS; tr++) {
        for (int c = 0; c < Q6_COLS; c++) xq[c] = (int16_t)((int)(lcg32(&rng) % 4001) - 2000);
        for (int g = 0; g < Q6_NB; g++) {
            int64_t sum = 0;
            for (int i = 0; i < Q6_GS; i++) sum += xq[g * Q6_GS + i];
            xg[g] = sum;
        }
        q6_quantize_x8(xq, x8, xd, Q6_NB);
        for (int r = 0; r < ROWS; r += 8) {
            float ref[8], got16[8], got8[8], gotasm[8];
            const uint8_t *wr = w8 + (size_t)r * Q6_COLS;
            const float *sr = s8 + (size_t)r * Q6_NB;
            glm5_matvec_int16sdot_8row(ref,
                    wr, wr + Q6_COLS, wr + 2 * Q6_COLS, wr + 3 * Q6_COLS,
                    wr + 4 * Q6_COLS, wr + 5 * Q6_COLS, wr + 6 * Q6_COLS, wr + 7 * Q6_COLS,
                    sr, sr + Q6_NB, sr + 2 * Q6_NB, sr + 3 * Q6_NB,
                    sr + 4 * Q6_NB, sr + 5 * Q6_NB, sr + 6 * Q6_NB, sr + 7 * Q6_NB,
                    Q6_GS, 0, xq, xg, Q6_COLS);
            const q6_block *qr = w6 + (size_t)r * Q6_NB;
            q6_matvec_8row(got16,
                    qr, qr + Q6_NB, qr + 2 * Q6_NB, qr + 3 * Q6_NB,
                    qr + 4 * Q6_NB, qr + 5 * Q6_NB, qr + 6 * Q6_NB, qr + 7 * Q6_NB,
                    xq, xg, Q6_NB);
            q6_matvec_a8_8row(got8,
                    qr, qr + Q6_NB, qr + 2 * Q6_NB, qr + 3 * Q6_NB,
                    qr + 4 * Q6_NB, qr + 5 * Q6_NB, qr + 6 * Q6_NB, qr + 7 * Q6_NB,
                    x8, xd, Q6_NB);
            q6_matvec_a8_asm_8row(gotasm,
                    qr, qr + Q6_NB, qr + 2 * Q6_NB, qr + 3 * Q6_NB,
                    qr + 4 * Q6_NB, qr + 5 * Q6_NB, qr + 6 * Q6_NB, qr + 7 * Q6_NB,
                    x8, xd, Q6_NB);
            for (int j = 0; j < 8; j++) {
                const float scalar = q6_dot_scalar(qr + j * Q6_NB, xq, Q6_NB);
                const double e16 = (double)got16[j] - ref[j];
                const double e8 = (double)got8[j] - ref[j];
                const double ie = (double)got16[j] - scalar;
                const double ae = (double)gotasm[j] - got8[j];
                a16_err2 += e16 * e16; a8_err2 += e8 * e8;
                out_ref2 += (double)ref[j] * ref[j];
                impl_err2 += ie * ie; impl_ref2 += (double)scalar * scalar;
                asm_err2 += ae * ae;
                if (fabs(e16) > a16_max) a16_max = fabs(e16);
                if (fabs(e8) > a8_max) a8_max = fabs(e8);
                if (fabs(ie) > impl_max) impl_max = fabs(ie);
                if (fabs(ae) > asm_max) asm_max = fabs(ae);
            }
        }
    }
    printf("QUALITY q6_bytes_per_weight=%.4f weight_rel_l2=%.8g "
            "a16_output_rel_l2=%.8g a16_max_abs=%.8g "
            "a8_output_rel_l2=%.8g a8_max_abs=%.8g "
            "a16_impl_rel_l2=%.8g a16_impl_max_abs=%.8g "
            "a8_asm_rel_l2=%.8g a8_asm_max_abs=%.8g\n",
            (double)sizeof(q6_block) / Q6_GS, sqrt(werr2 / wref2),
            sqrt(a16_err2 / out_ref2), a16_max,
            sqrt(a8_err2 / out_ref2), a8_max,
            sqrt(impl_err2 / impl_ref2), impl_max,
            sqrt(asm_err2 / out_ref2), asm_max);
    munmap(w8, wcount); munmap(s8, (size_t)ROWS * Q6_NB * sizeof(float));
    munmap(w6, (size_t)ROWS * Q6_NB * sizeof(q6_block));
    munmap(xq, (size_t)Q6_COLS * sizeof(int16_t)); munmap(xg, (size_t)Q6_NB * sizeof(int64_t));
    munmap(x8, (size_t)Q6_COLS * sizeof(int8_t)); munmap(xd, (size_t)Q6_NB * sizeof(float));
    return 0;
}

int main(void) {
    int rc = quality_check();
    if (rc) return rc;
    const char *mode = getenv("MODE");
    if (mode && strcmp(mode, "check") == 0) return 0;
    const int reps = env_int("REPS", 3);
    const size_t row_bytes = (size_t)Q6_NB * sizeof(q6_block);
    const size_t tensor_bytes = (size_t)Q6_ROWS * row_bytes;
    const size_t target = env_size("Q6_BYTES", (size_t)2 << 30);
    size_t ntens = target / tensor_bytes;
    if (ntens < 1) ntens = 1;
    const int rows = (int)(ntens * Q6_ROWS);
    const size_t physical_bytes = ntens * tensor_bytes;
    const size_t logical_bytes = (size_t)rows * Q6_COLS;
    q6_block *w = mmap(NULL, physical_bytes, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *y = mmap(NULL, (size_t)rows * sizeof(float), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int16_t *xq = mmap(NULL, (size_t)Q6_COLS * sizeof(int16_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int64_t *xg = mmap(NULL, (size_t)Q6_NB * sizeof(int64_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int8_t *x8 = mmap(NULL, (size_t)Q6_COLS * sizeof(int8_t), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    float *xd = mmap(NULL, (size_t)Q6_NB * sizeof(float), PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (w == MAP_FAILED || y == MAP_FAILED || xq == MAP_FAILED || xg == MAP_FAILED ||
            x8 == MAP_FAILED || xd == MAP_FAILED) {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return 2;
    }
    for (int c = 0; c < Q6_COLS; c++) xq[c] = (int16_t)((c * 37) % 4001 - 2000);
    for (int g = 0; g < Q6_NB; g++) {
        int64_t sum = 0;
        for (int i = 0; i < Q6_GS; i++) sum += xq[g * Q6_GS + i];
        xg[g] = sum;
    }
    q6_quantize_x8(xq, x8, xd, Q6_NB);
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
        int nb8 = Q6_ROWS / 8, per = (nb8 + nthr - 1) / nthr;
        int b0 = tid * per, b1 = b0 + per > nb8 ? nb8 : b0 + per;
        for (size_t t = 0; t < ntens; t++) {
            for (int bi = b0; bi < b1; bi++) {
                int r0 = (int)(t * Q6_ROWS) + bi * 8;
                for (int j = 0; j < 8; j++) {
                    uint32_t rng = 1u + (uint32_t)(r0 + j) * 2654435761u;
                    q6_block *row = w + (size_t)(r0 + j) * Q6_NB;
                    for (int g = 0; g < Q6_NB; g++) {
                        uint8_t code[Q6_GS];
                        for (int i = 0; i < Q6_GS; i++) code[i] = (uint8_t)(lcg32(&rng) >> 26);
                        q6_pack_codes(row + g, code, (float)(1 + (lcg32(&rng) & 1023)) * 1.0e-6f);
                    }
                }
            }
        }
    }

    const int do_raw = mode && strcmp(mode, "raw") == 0;
    const int do_a8 = mode && strcmp(mode, "q6a8") == 0;
    const int do_a8asm = mode && strcmp(mode, "q6a8asm") == 0;
    const char *mode_name = do_raw ? "raw" : do_a8 ? "q6a8" :
            do_a8asm ? "q6a8asm" : "q6a16";
    printf("CONFIG mode=%s threads=%d rows=%d cols=%d physical=%.3f_GiB logical=%.3f_GiB reps=%d\n",
            mode_name, omp_get_max_threads(), rows, Q6_COLS,
            (double)physical_bytes / ((size_t)1 << 30),
            (double)logical_bytes / ((size_t)1 << 30), reps);
    double best = 1.0e100, total = 0;
    for (int it = 0; it < reps; it++) {
        double sink = 0, t0 = now_sec();
#pragma omp parallel reduction(+:sink)
        {
            int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
            int nb8 = Q6_ROWS / 8, per = (nb8 + nthr - 1) / nthr;
            int b0 = tid * per, b1 = b0 + per > nb8 ? nb8 : b0 + per;
            for (size_t t = 0; t < ntens; t++) {
                for (int bi = b0; bi < b1; bi++) {
                    int r = (int)(t * Q6_ROWS) + bi * 8;
                    q6_block *wr = w + (size_t)r * Q6_NB;
                    float *dst = y + r;
                    if (do_raw) raw_q6_block(dst, (const uint8_t *)wr, (int)(8 * row_bytes));
                    else if (do_a8asm) q6_matvec_a8_asm_8row(dst,
                            wr, wr + Q6_NB, wr + 2 * Q6_NB, wr + 3 * Q6_NB,
                            wr + 4 * Q6_NB, wr + 5 * Q6_NB, wr + 6 * Q6_NB, wr + 7 * Q6_NB,
                            x8, xd, Q6_NB);
                    else if (do_a8) q6_matvec_a8_8row(dst,
                            wr, wr + Q6_NB, wr + 2 * Q6_NB, wr + 3 * Q6_NB,
                            wr + 4 * Q6_NB, wr + 5 * Q6_NB, wr + 6 * Q6_NB, wr + 7 * Q6_NB,
                            x8, xd, Q6_NB);
                    else q6_matvec_8row(dst,
                            wr, wr + Q6_NB, wr + 2 * Q6_NB, wr + 3 * Q6_NB,
                            wr + 4 * Q6_NB, wr + 5 * Q6_NB, wr + 6 * Q6_NB, wr + 7 * Q6_NB,
                            xq, xg, Q6_NB);
                    sink += dst[0] + dst[1] + dst[2] + dst[3] + dst[4] + dst[5] + dst[6] + dst[7];
                }
            }
        }
        const double dt = now_sec() - t0;
        if (dt < best) best = dt;
        total += dt; g_sink += sink;
        printf("RESULT rep=%d seconds=%.6f physical_GB/s=%.1f logical_GB/s=%.1f sink=%.9g\n",
                it, dt, (double)physical_bytes / dt / 1.0e9,
                (double)logical_bytes / dt / 1.0e9, sink);
    }
    printf("SUMMARY mode=%s best_physical_GB/s=%.1f mean_physical_GB/s=%.1f best_logical_GB/s=%.1f mean_logical_GB/s=%.1f\n",
            mode_name, (double)physical_bytes / best / 1.0e9,
            (double)physical_bytes / (total / reps) / 1.0e9,
            (double)logical_bytes / best / 1.0e9,
            (double)logical_bytes / (total / reps) / 1.0e9);
    printf("DONE g_sink=%.17g\n", g_sink);
    return 0;
}
