#ifndef K3_QUANT_H
#define K3_QUANT_H

/* Small, source-faithful GGML quantized matrix interface for K3.
 *
 * The GGUF releases keep the dense weights in Q8_0 and the expert weights in
 * IQ1_S/IQ2_XS, with a few IQ2_XXS/IQ3_XXS tensors selected by calibration.
 * This header deliberately keeps the storage format unchanged.  Reference,
 * A16-SVE, and Q8-SDOT paths share one interface so the runner can switch
 * kernels without changing GGUF staging.
 */
#define GGML_DEQUANT_IMPLEMENTATION
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "../../common/ggml_dequant.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

typedef enum {
    K3_Q_F32 = 1,
    K3_Q_Q8_0,
    K3_Q_IQ1_S,
    K3_Q_IQ2_XS,
    K3_Q_IQ2_XXS,
    K3_Q_IQ3_XXS,
    K3_Q_BF16,
} k3_quant_type;

typedef struct {
    const uint8_t *data;
    int type;
    int rows;
    int cols;
    size_t row_bytes;
} k3_quant_matrix;

typedef enum {
    K3_QUANT_REFERENCE = 0,
    K3_QUANT_SVE_A16 = 1,
    K3_QUANT_SVE_Q8 = 2,
} k3_quant_kernel_mode;

typedef struct {
    int16_t *a16;
    int8_t *q8;
    int cols;
    float scale;      /* scale of the activation the selected mode uses */
    float scale_a16;  /* valid whenever a16_ready is set */
    int a16_ready;    /* the int16 activation holds this call's input */
} k3_quant_workspace;

/* Decode LUTs are small enough to stay resident in the shared L2/L1 working
 * set and remove the scalar grid/sign expansion from every output row. */
static int8_t k3_iq1_lut[2048][8];
static int8_t k3_iq2_lut[65536][8];
/* Sign patterns as multiplicands, so the sign application that IQ2_XXS and
 * IQ3_XXS need per element becomes one vector multiply instead of eight
 * predicated scalar negations. */
static int8_t k3_iq_sign_lut[256][8];
static int8_t k3_iq_ones[32];
static int8_t k3_iq_ones64[64];
static pthread_once_t k3_quant_lut_once = PTHREAD_ONCE_INIT;

/* Every grid entry is eight bytes wide, so a whole group moves as one 64-bit
 * copy.  The scalar j-loops this replaces were the dominant cost in the IQ
 * kernels: eight byte loads and eight byte stores to stage what is a single
 * aligned doubleword. */
static inline void k3_quant_copy8(void *dst, const void *src) {
    memcpy(dst, src, 8);
}

static void k3_quant_init_luts(void) {
    for (int s = 0; s < 256; ++s)
        for (int j = 0; j < 8; ++j)
            k3_iq_sign_lut[s][j] = (s & kmask_iq2xs[j]) ? -1 : 1;
    for (int j = 0; j < 32; ++j) k3_iq_ones[j] = 1;
    for (int j = 0; j < 64; ++j) k3_iq_ones64[j] = 1;
    for (int i = 0; i < 2048; ++i) {
        const int8_t *g = (const int8_t *)(iq1s_grid + i);
        memcpy(k3_iq1_lut[i], g, 8);
    }
    for (int code = 0; code < 65536; ++code) {
        const uint8_t *g = (const uint8_t *)(iq2xs_grid + (code & 511));
        uint8_t signs = ksigns_iq2xs[code >> 9];
        for (int j = 0; j < 8; ++j)
            k3_iq2_lut[code][j] = (int8_t)
                ((signs & kmask_iq2xs[j]) ? -(int)g[j] : (int)g[j]);
    }
}

static inline void k3_quant_ensure_luts(void) {
    (void)pthread_once(&k3_quant_lut_once, k3_quant_init_luts);
}

/* omp_set_num_threads on every matvec forces the runtime to reconsider the
 * team even when the count has not changed; a decode step issues hundreds of
 * these.  Profiling put __kmp_fork_barrier at ~13% of cycles. */
static inline void k3_quant_set_threads(int threads) {
#if defined(_OPENMP)
    int want = threads > 0 ? threads : 1;
    if (omp_get_max_threads() != want) omp_set_num_threads(want);
#else
    (void)threads;
#endif
}

static inline int k3_quant_type_from_name(const char *name) {
    if (!strcmp(name, "F32")) return K3_Q_F32;
    if (!strcmp(name, "BF16")) return K3_Q_BF16;
    if (!strcmp(name, "Q8_0")) return K3_Q_Q8_0;
    if (!strcmp(name, "IQ1_S")) return K3_Q_IQ1_S;
    if (!strcmp(name, "IQ2_XS")) return K3_Q_IQ2_XS;
    if (!strcmp(name, "IQ2_XXS")) return K3_Q_IQ2_XXS;
    if (!strcmp(name, "IQ3_XXS")) return K3_Q_IQ3_XXS;
    return 0;
}

static inline const char *k3_quant_type_name(int type) {
    switch (type) {
    case K3_Q_F32: return "F32";
    case K3_Q_BF16: return "BF16";
    case K3_Q_Q8_0: return "Q8_0";
    case K3_Q_IQ1_S: return "IQ1_S";
    case K3_Q_IQ2_XS: return "IQ2_XS";
    case K3_Q_IQ2_XXS: return "IQ2_XXS";
    case K3_Q_IQ3_XXS: return "IQ3_XXS";
    default: return "UNKNOWN";
    }
}

static inline uint32_t k3_quant_ggml_type(int type) {
    switch (type) {
    case K3_Q_Q8_0: return GGML_TYPE_Q8_0;
    case K3_Q_IQ1_S: return GGML_TYPE_IQ1_S;
    case K3_Q_IQ2_XS: return GGML_TYPE_IQ2_XS;
    case K3_Q_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
    case K3_Q_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
    default: return 0xffffffffu;
    }
}

static inline size_t k3_quant_row_bytes(int type, int cols) {
    if (type == K3_Q_F32) return (size_t)cols * sizeof(float);
    if (type == K3_Q_BF16) return (size_t)cols * sizeof(uint16_t);
    uint32_t gt = k3_quant_ggml_type(type);
    return gt == 0xffffffffu ? 0 : dequant_row_size(gt, cols);
}

static inline int k3_quant_valid_shape(int type, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return 0;
    if (type == K3_Q_F32 || type == K3_Q_BF16) return 1;
    uint32_t gt = k3_quant_ggml_type(type);
    if (gt == 0xffffffffu) return 0;
    int block = (type == K3_Q_Q8_0) ? 32 : 256;
    return cols % block == 0;
}

static inline float k3_quant_dot_row_ref(const uint8_t *row, int type,
                                         const float *x, int cols) {
    if (type == K3_Q_F32) {
        const float *w = (const float *)row;
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) sum += (double)w[i] * x[i];
        return (float)sum;
    }
    if (type == K3_Q_BF16) {
        const uint16_t *w = (const uint16_t *)row;
        double sum = 0.0;
        for (int i = 0; i < cols; ++i)
            sum += (double)bf16_to_f32_scalar(w[i]) * x[i];
        return (float)sum;
    }
    float *tmp = (float *)malloc((size_t)cols * sizeof(*tmp));
    if (!tmp) return NAN;
    if (dequant_row(k3_quant_ggml_type(type), row, tmp, cols)) {
        free(tmp);
        return NAN;
    }
    double sum = 0.0;
    for (int i = 0; i < cols; ++i) sum += (double)tmp[i] * x[i];
    free(tmp);
    return (float)sum;
}

static inline int k3_quant_dequant_row(float *out, const uint8_t *row,
                                       int type, int cols) {
    if (!out || !row || cols <= 0) return -1;
    if (type == K3_Q_F32) {
        memcpy(out, row, (size_t)cols * sizeof(float));
        return 0;
    }
    if (type == K3_Q_BF16) {
        const uint16_t *w = (const uint16_t *)row;
        for (int i = 0; i < cols; ++i) out[i] = bf16_to_f32_scalar(w[i]);
        return 0;
    }
    uint32_t gt = k3_quant_ggml_type(type);
    if (gt == 0xffffffffu || dequant_row(gt, row, out, cols)) return -1;
    return 0;
}

static inline float k3_quant_dot_row(const uint8_t *row, int type,
                                     const float *x, int cols) {
    /* Keep the first production path source-faithful.  This also handles the
     * mixed IQ types present in the downloaded checkpoints without silently
     * treating one format as another. */
    return k3_quant_dot_row_ref(row, type, x, cols);
}

static inline int k3_quant_workspace_prepare(k3_quant_workspace *ws,
                                              int cols, int mode) {
    /* The workspace only holds the quantized activation, so 32-column
     * granularity is enough for Q8_0.  Per-format block alignment is the
     * caller's contract and is checked by k3_quant_valid_shape; requiring 256
     * here silently pushed every Q8_0 tensor with cols < 256 (attn_k_b is
     * 4096x128) onto the scalar reference path. */
    if (!ws || cols <= 0 || cols % 32) return -1;
    if (ws->cols >= cols && ws->a16 &&
        (mode == K3_QUANT_SVE_A16 || ws->q8)) {
        return 0;
    }
    free(ws->a16); free(ws->q8);
    memset(ws, 0, sizeof(*ws));
    ws->cols = cols;
    /* The int16 activation is always allocated: Q8_0 blocks are 32 elements,
     * so 32 int16 lanes fill a whole 512-bit A64FX vector while 32 int8 lanes
     * fill only half of one.  Q8_0 therefore wants the a16 kernel even when
     * the IQ tensors in the same layer are running in q8 mode. */
    ws->a16 = (int16_t *)malloc((size_t)cols * sizeof(*ws->a16));
    if (!ws->a16) return -1;
    if (mode == K3_QUANT_SVE_Q8) {
        ws->q8 = (int8_t *)malloc((size_t)cols * sizeof(*ws->q8));
        if (!ws->q8) { free(ws->a16); ws->a16 = NULL; return -1; }
    }
    return 0;
}

static inline void k3_quant_workspace_free(k3_quant_workspace *ws) {
    if (!ws) return;
    free(ws->a16); free(ws->q8); memset(ws, 0, sizeof(*ws));
}

static inline float k3_quant_fabs(float x) { return x < 0.0f ? -x : x; }

static inline void k3_quant_prepare_a16(k3_quant_workspace *ws,
                                        const float *x, int cols) {
    float amax = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float a = k3_quant_fabs(x[i]); if (a > amax) amax = a;
    }
    ws->scale = amax > 0.0f ? amax / 32767.0f : 0.0f;
    float inv = amax > 0.0f ? 32767.0f / amax : 0.0f;
    for (int i = 0; i < cols; ++i) {
        int v = (int)lrintf(x[i] * inv);
        ws->a16[i] = (int16_t)(v > 32767 ? 32767 : v < -32767 ? -32767 : v);
    }
}

static inline void k3_quant_prepare_q8(k3_quant_workspace *ws,
                                       const float *x, int cols) {
    float amax = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float a = k3_quant_fabs(x[i]); if (a > amax) amax = a;
    }
    ws->scale = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    for (int i = 0; i < cols; ++i) {
        int v = (int)lrintf(x[i] * inv);
        ws->q8[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
}

/* IQ2_XS carries a separate scale for each half of a 32-element group.  Doing
 * that as two 16-lane dots used a quarter of a 512-bit vector per operation
 * and paid two reductions.  One 32-lane svdot produces eight int32 lanes —
 * lanes 0-3 are the low half, lanes 4-7 the high half — so scaling the lanes
 * in place lets a single svaddv finish both halves. */
static inline int64_t k3_quant_dot_i8_split16(const int8_t *w, const int8_t *x,
                                              int s_lo, int s_hi) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svwhilelt_b8(0, 32);
    svint32_t acc = svdot_s32(svdup_s32(0), svld1_s8(pg, w), svld1_s8(pg, x));
    svbool_t lanes = svwhilelt_b32(0, 8);
    svbool_t lo = svwhilelt_b32(0, 4);
    svint32_t sc = svsel_s32(lo, svdup_s32(s_lo), svdup_s32(s_hi));
    return (int64_t)svaddv_s32(lanes, svmul_s32_x(lanes, acc, sc));
#else
    int64_t lo = 0, hi = 0;
    for (int i = 0; i < 16; ++i) {
        lo += (int)w[i] * x[i];
        hi += (int)w[i + 16] * x[i + 16];
    }
    return lo * s_lo + hi * s_hi;
#endif
}

/* Two consecutive 32-element groups in one 64-lane svdot.  A 32-element group
 * fills only half an A64FX vector; a pair fills it exactly.  svdot folds each
 * four bytes into one int32 lane, so group A lands in lanes 0-7 and group B in
 * lanes 8-15, and per-group integer scales apply by selecting on lane index —
 * one dot and one reduction where there were two of each. */
static inline int64_t k3_quant_dot_i8_pair32(const int8_t *w, const int8_t *x,
                                             int s_a, int s_b) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svwhilelt_b8(0, 64);
    svint32_t acc = svdot_s32(svdup_s32(0), svld1_s8(pg, w), svld1_s8(pg, x));
    svbool_t lanes = svwhilelt_b32(0, 16);
    svbool_t first = svwhilelt_b32(0, 8);
    svint32_t sc = svsel_s32(first, svdup_s32(s_a), svdup_s32(s_b));
    return (int64_t)svaddv_s32(lanes, svmul_s32_x(lanes, acc, sc));
#else
    int64_t a = 0, b = 0;
    for (int i = 0; i < 32; ++i) {
        a += (int)w[i] * x[i];
        b += (int)w[i + 32] * x[i + 32];
    }
    return a * s_a + b * s_b;
#endif
}

/* IQ2_XS again, but over a pair of groups: 64 bytes give 16 int32 lanes and
 * the format's four half-group scales land on lanes 0-3, 4-7, 8-11, 12-15. */
static inline int64_t k3_quant_dot_i8_quad16(const int8_t *w, const int8_t *x,
                                             const int *s) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svwhilelt_b8(0, 64);
    svint32_t acc = svdot_s32(svdup_s32(0), svld1_s8(pg, w), svld1_s8(pg, x));
    svint32_t sc = svdup_s32(s[3]);
    sc = svsel_s32(svwhilelt_b32(0, 12), svdup_s32(s[2]), sc);
    sc = svsel_s32(svwhilelt_b32(0, 8), svdup_s32(s[1]), sc);
    sc = svsel_s32(svwhilelt_b32(0, 4), svdup_s32(s[0]), sc);
    svbool_t lanes = svwhilelt_b32(0, 16);
    return (int64_t)svaddv_s32(lanes, svmul_s32_x(lanes, acc, sc));
#else
    int64_t total = 0;
    for (int q = 0; q < 4; ++q) {
        int64_t p = 0;
        for (int i = 0; i < 16; ++i) p += (int)w[16 * q + i] * x[16 * q + i];
        total += p * s[q];
    }
    return total;
#endif
}

#if defined(__ARM_FEATURE_SVE)
/* A16 group accumulator.  Unlike the q8 path there is nothing to gain from
 * pairing groups: 32 int16 activations already fill a 512-bit vector.  The
 * wins here are staging the grid as int8 doublewords and widening it in the
 * load (svld1sb_s16), plus keeping the per-group scaling in the vector domain
 * so a block needs one svaddv instead of eight.
 *
 * svdot_s64 folds four int16 pairs into each int64 lane, so a 32-element group
 * occupies lanes 0-7 and IQ2_XS's two half-group scales split at lane 4. */
static inline svint64_t k3_quant_acc_a16(svint64_t vacc, const int8_t *w,
                                         const int16_t *x, svint64_t scale) {
    svbool_t pg = svwhilelt_b16(0, 32);
    svint64_t d = svdot_s64(svdup_s64(0), svld1sb_s16(pg, w), svld1_s16(pg, x));
    return svmla_s64_x(svptrue_b64(), vacc, d, scale);
}

static inline svint64_t k3_quant_a16_split_scale(int s_lo, int s_hi) {
    return svsel_s64(svwhilelt_b64(0, 4), svdup_s64(s_lo), svdup_s64(s_hi));
}

static inline int64_t k3_quant_acc_a16_total(svint64_t vacc) {
    return svaddv_s64(svptrue_b64(), vacc);
}
#endif

#if defined(__ARM_FEATURE_SVE)
/* Accumulator forms of the two dot helpers.  Reducing to a scalar per group
 * costs an svaddv plus a SIMD->GPR move, which profiling showed to be the two
 * hottest instructions in the IQ kernels.  Staying in the vector domain across
 * a whole 256-element block leaves one reduction per block instead of four.
 * Lane magnitudes stay far inside int32: four int8 products (<=64516) times a
 * group scale (<=31) times four pairs is ~8e6. */
static inline svint32_t k3_quant_acc_pair32(svint32_t vacc, const int8_t *w,
                                            const int8_t *x, int s_a, int s_b) {
    svbool_t pg = svwhilelt_b8(0, 64);
    svbool_t lanes = svwhilelt_b32(0, 16);
    svint32_t acc = svdot_s32(svdup_s32(0), svld1_s8(pg, w), svld1_s8(pg, x));
    svint32_t sc = svsel_s32(svwhilelt_b32(0, 8), svdup_s32(s_a), svdup_s32(s_b));
    return svadd_s32_x(lanes, vacc, svmul_s32_x(lanes, acc, sc));
}

static inline svint32_t k3_quant_acc_quad16(svint32_t vacc, const int8_t *w,
                                            const int8_t *x, const int *s) {
    svbool_t pg = svwhilelt_b8(0, 64);
    svbool_t lanes = svwhilelt_b32(0, 16);
    svint32_t acc = svdot_s32(svdup_s32(0), svld1_s8(pg, w), svld1_s8(pg, x));
    svint32_t sc = svdup_s32(s[3]);
    sc = svsel_s32(svwhilelt_b32(0, 12), svdup_s32(s[2]), sc);
    sc = svsel_s32(svwhilelt_b32(0, 8), svdup_s32(s[1]), sc);
    sc = svsel_s32(svwhilelt_b32(0, 4), svdup_s32(s[0]), sc);
    return svadd_s32_x(lanes, vacc, svmul_s32_x(lanes, acc, sc));
}

static inline int64_t k3_quant_acc_total(svint32_t vacc) {
    return (int64_t)svaddv_s32(svwhilelt_b32(0, 16), vacc);
}
#endif

/* Multiply n grid bytes by their +-1 sign pattern in one vector op. */
static inline void k3_quant_apply_signs_n(int8_t *qw, const int8_t *sg, int n) {
#if defined(__ARM_FEATURE_SVE)
    svbool_t pg = svwhilelt_b8(0, n);
    svst1_s8(pg, qw, svmul_s8_x(pg, svld1_s8(pg, qw), svld1_s8(pg, sg)));
#else
    for (int j = 0; j < n; ++j) qw[j] = (int8_t)(qw[j] * sg[j]);
#endif
}

static inline void k3_quant_apply_signs32(int8_t *qw, const int8_t *sg) {
    k3_quant_apply_signs_n(qw, sg, 32);
}

static inline void k3_quant_apply_signs64(int8_t *qw, const int8_t *sg) {
    k3_quant_apply_signs_n(qw, sg, 64);
}

static inline int64_t k3_quant_dot_i16(const int16_t *a, const int16_t *b, int n) {
#if defined(__ARM_FEATURE_SVE)
    svint64_t acc = svdup_s64(0);
    for (int i = 0; i < n; i += (int)svcnth()) {
        svbool_t pg = svwhilelt_b16(i, n);
        acc = svdot_s64(acc, svld1_s16(pg, a + i), svld1_s16(pg, b + i));
    }
    return svaddv_s64(svptrue_b64(), acc);
#else
    int64_t acc = 0; for (int i = 0; i < n; ++i) acc += (int32_t)a[i] * b[i]; return acc;
#endif
}

static inline int64_t k3_quant_dot_i8(const int8_t *a, const int8_t *b, int n) {
#if defined(__ARM_FEATURE_SVE)
    svint32_t acc = svdup_s32(0);
    for (int i = 0; i < n; i += (int)svcntb()) {
        svbool_t pg = svwhilelt_b8(i, n);
        acc = svdot_s32(acc, svld1_s8(pg, a + i), svld1_s8(pg, b + i));
    }
    return (int64_t)svaddv_s32(svptrue_b32(), acc);
#else
    int64_t acc = 0; for (int i = 0; i < n; ++i) acc += (int)a[i] * b[i]; return acc;
#endif
}

static inline float k3_quant_iq2_xs_row(const block_iq2_xs *w,
                                        const int16_t *x, float xd, int nb) {
    float out = 0.0f; int8_t qw[32];
    for (int b = 0; b < nb; ++b) {
        int64_t part = 0;
#if defined(__ARM_FEATURE_SVE)
        svint64_t vacc = svdup_s64(0);
#endif
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                uint16_t code = w[b].qs[4 * ib + l];
                k3_quant_copy8(qw + 8 * l, k3_iq2_lut[code]);
            }
            int s0 = 2 * (w[b].scales[ib] & 15) + 1;
            int s1 = 2 * (w[b].scales[ib] >> 4) + 1;
#if defined(__ARM_FEATURE_SVE)
            vacc = k3_quant_acc_a16(vacc, qw, x + 256 * b + 32 * ib,
                                    k3_quant_a16_split_scale(s0, s1));
#else
            for (int j = 0; j < 16; ++j) {
                part += (int)qw[j] * x[256 * b + 32 * ib + j] * s0;
                part += (int)qw[j + 16] * x[256 * b + 32 * ib + j + 16] * s1;
            }
#endif
        }
#if defined(__ARM_FEATURE_SVE)
        part = k3_quant_acc_a16_total(vacc);
#endif
        out += ggml_fp16_to_fp32(w[b].d) * xd * (float)part * 0.125f;
    }
    return out;
}

static inline float k3_quant_iq2_xxs_row(const block_iq2_xxs *w,
                                         const int16_t *x, float xd, int nb) {
    /* (0.5 + k) * 0.25 is (1 + 2k) * 0.125, so the group scale is an integer
     * and the whole block can accumulate before one reduction. */
    float out = 0.0f; uint32_t aux[2]; int8_t qw[32], sg[32];
    for (int b = 0; b < nb; ++b) {
        int64_t part = 0;
#if defined(__ARM_FEATURE_SVE)
        svint64_t vacc = svdup_s64(0);
#endif
        for (int ib = 0; ib < 8; ++ib) {
            memcpy(aux, w[b].qs + 4 * ib, sizeof(aux));
            for (int l = 0; l < 4; ++l) {
                k3_quant_copy8(qw + 8 * l,
                               iq2xxs_grid + ((const uint8_t *)aux)[l]);
                k3_quant_copy8(sg + 8 * l,
                               k3_iq_sign_lut[ksigns_iq2xs[(aux[1] >> (7 * l)) & 127]]);
            }
            k3_quant_apply_signs32(qw, sg);
            int s = 1 + 2 * (int)(aux[1] >> 28);
#if defined(__ARM_FEATURE_SVE)
            vacc = k3_quant_acc_a16(vacc, qw, x + 256 * b + 32 * ib,
                                    svdup_s64(s));
#else
            for (int j = 0; j < 32; ++j)
                part += (int)qw[j] * x[256 * b + 32 * ib + j] * s;
#endif
        }
#if defined(__ARM_FEATURE_SVE)
        part = k3_quant_acc_a16_total(vacc);
#endif
        out += ggml_fp16_to_fp32(w[b].d) * xd * 0.125f * (float)part;
    }
    return out;
}

static inline float k3_quant_iq1_s_row(const block_iq1_s *w,
                                       const int16_t *x, float xd, int nb) {
    /* Both terms carry an integer group scale, so the block accumulates once
     * for the weight product and once for sum(x) against ones. */
    float out = 0.0f; const uint8_t *qs; const uint16_t *qh;
    int8_t qw[32];
    for (int b = 0; b < nb; ++b) {
        const block_iq1_s *wb = w + b; qs = wb->qs; qh = wb->qh;
        int64_t acc = 0, dacc = 0;
#if defined(__ARM_FEATURE_SVE)
        svint64_t vacc = svdup_s64(0), vdacc = svdup_s64(0);
#endif
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                int idx = qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8);
                k3_quant_copy8(qw + 8 * l, k3_iq1_lut[idx]);
            }
            int s = 2 * ((qh[ib] >> 12) & 7) + 1;
            int sd = (qh[ib] & 0x8000) ? -s : s;
            const int16_t *xg = x + 256 * b + 32 * ib;
#if defined(__ARM_FEATURE_SVE)
            vacc = k3_quant_acc_a16(vacc, qw, xg, svdup_s64(s));
            vdacc = k3_quant_acc_a16(vdacc, k3_iq_ones, xg, svdup_s64(sd));
#else
            for (int j = 0; j < 32; ++j) {
                acc += (int)qw[j] * xg[j] * s;
                dacc += (int)xg[j] * sd;
            }
#endif
            qs += 4;
        }
#if defined(__ARM_FEATURE_SVE)
        acc = k3_quant_acc_a16_total(vacc);
        dacc = k3_quant_acc_a16_total(vdacc);
#endif
        out += ggml_fp16_to_fp32(wb->d) * xd *
               ((float)acc + 0.125f * (float)dacc);
    }
    return out;
}

static inline float k3_quant_iq3_xxs_row(const block_iq3_xxs *w,
                                         const int16_t *x, float xd, int nb) {
    float out = 0.0f; int8_t qw[32], sg[32];
    for (int b = 0; b < nb; ++b) {
        const uint8_t *q3 = w[b].qs, *gas = w[b].qs + 64;
        int64_t part = 0;
#if defined(__ARM_FEATURE_SVE)
        svint64_t vacc = svdup_s64(0);
#endif
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t aux; memcpy(&aux, gas + 4 * ib, sizeof(aux));
            const uint8_t *q3h = q3 + 8 * ib;
            for (int l = 0; l < 4; ++l) {
                /* IQ3 packs two 4-byte grid entries per group. */
                memcpy(qw + 8 * l, iq3xxs_grid + q3h[2 * l], 4);
                memcpy(qw + 8 * l + 4, iq3xxs_grid + q3h[2 * l + 1], 4);
                k3_quant_copy8(sg + 8 * l,
                               k3_iq_sign_lut[ksigns_iq2xs[(aux >> (7 * l)) & 127]]);
            }
            k3_quant_apply_signs32(qw, sg);
            int s = 2 * (int)(aux >> 28) + 1;
#if defined(__ARM_FEATURE_SVE)
            vacc = k3_quant_acc_a16(vacc, qw, x + 256 * b + 32 * ib,
                                    svdup_s64(s));
#else
            for (int j = 0; j < 32; ++j)
                part += (int)qw[j] * x[256 * b + 32 * ib + j] * s;
#endif
        }
#if defined(__ARM_FEATURE_SVE)
        part = k3_quant_acc_a16_total(vacc);
#endif
        out += ggml_fp16_to_fp32(w[b].d) * xd * (float)part * 0.25f;
    }
    return out;
}

static inline float k3_quant_iq2_xs_q8_row(const block_iq2_xs *w,
                                           const int8_t *x, float xd, int nb) {
    /* IQ2_XS keeps the fully merged grid+sign LUT (512 KiB) even though the
     * same substitution is a loss for IQ2_XXS.  Measured both ways: folding
     * IQ2_XS down to the 4 KiB grid plus a sign multiply costs ~7%, while
     * expanding IQ2_XXS up to a 256 KiB merged table costs ~9%.  Table size
     * alone does not predict this; each format had to be measured. */
    float out = 0.0f; int8_t qw[64];
    for (int b = 0; b < nb; ++b) {
        int64_t part = 0;
#if defined(__ARM_FEATURE_SVE)
        svint32_t vacc = svdup_s32(0);
#endif
        for (int ib = 0; ib < 8; ib += 2) {
            int s[4];
            for (int h = 0; h < 2; ++h) {
                for (int l = 0; l < 4; ++l) {
                    uint16_t code = w[b].qs[4 * (ib + h) + l];
                    k3_quant_copy8(qw + 32 * h + 8 * l, k3_iq2_lut[code]);
                }
                s[2 * h] = 2 * (w[b].scales[ib + h] & 15) + 1;
                s[2 * h + 1] = 2 * (w[b].scales[ib + h] >> 4) + 1;
            }
#if defined(__ARM_FEATURE_SVE)
            vacc = k3_quant_acc_quad16(vacc, qw, x + 256 * b + 32 * ib, s);
#else
            part += k3_quant_dot_i8_quad16(qw, x + 256 * b + 32 * ib, s);
#endif
        }
#if defined(__ARM_FEATURE_SVE)
        part = k3_quant_acc_total(vacc);
#endif
        out += ggml_fp16_to_fp32(w[b].d) * xd * (float)part * 0.125f;
    }
    return out;
}

static inline float k3_quant_iq2_xxs_q8_row(const block_iq2_xxs *w,
                                            const int8_t *x, float xd, int nb) {
    /* The per-group multiplier (0.5 + k) * 0.25 is (1 + 2k) * 0.125, so the
     * group scale is an integer and two groups can share one 64-lane dot. */
    float out = 0.0f; uint32_t aux[2]; int8_t qw[64], sg[64];
    for (int b = 0; b < nb; ++b) {
        int64_t part = 0;
#if defined(__ARM_FEATURE_SVE)
        svint32_t vacc = svdup_s32(0);
#endif
        for (int ib = 0; ib < 8; ib += 2) {
            int s[2];
            for (int h = 0; h < 2; ++h) {
                memcpy(aux, w[b].qs + 4 * (ib + h), sizeof(aux));
                int8_t *qh8 = qw + 32 * h, *sh8 = sg + 32 * h;
                for (int l = 0; l < 4; ++l) {
                    k3_quant_copy8(qh8 + 8 * l,
                                   iq2xxs_grid + ((const uint8_t *)aux)[l]);
                    k3_quant_copy8(sh8 + 8 * l,
                                   k3_iq_sign_lut[ksigns_iq2xs[(aux[1] >> (7 * l)) & 127]]);
                }
                s[h] = 1 + 2 * (int)(aux[1] >> 28);
            }
            k3_quant_apply_signs64(qw, sg);
#if defined(__ARM_FEATURE_SVE)
            vacc = k3_quant_acc_pair32(vacc, qw, x + 256 * b + 32 * ib,
                                       s[0], s[1]);
#else
            part += k3_quant_dot_i8_pair32(qw, x + 256 * b + 32 * ib,
                                           s[0], s[1]);
#endif
        }
#if defined(__ARM_FEATURE_SVE)
        part = k3_quant_acc_total(vacc);
#endif
        out += ggml_fp16_to_fp32(w[b].d) * xd * 0.125f * (float)part;
    }
    return out;
}

static inline float k3_quant_iq1_s_q8_row(const block_iq1_s *w,
                                          const int8_t *x, float xd, int nb) {
    /* Both terms of the IQ1 group sum, S*<w,x> and S*delta*sum(x), carry an
     * integer group scale, so a pair of groups shares one 64-lane dot for the
     * product and another for the sum-against-ones. */
    float out = 0.0f; int8_t qw[64];
    for (int b = 0; b < nb; ++b) {
        const uint8_t *qs = w[b].qs; const uint16_t *qh = w[b].qh;
        int64_t acc = 0, dacc = 0;
#if defined(__ARM_FEATURE_SVE)
        svint32_t vacc = svdup_s32(0), vdacc = svdup_s32(0);
#endif
        for (int ib = 0; ib < 8; ib += 2) {
            int s[2], sd[2];
            for (int h = 0; h < 2; ++h) {
                const uint8_t *qsh = qs + 4 * (ib + h);
                uint16_t qhh = qh[ib + h];
                for (int l = 0; l < 4; ++l) {
                    int idx = qsh[l] | (((qhh >> (3 * l)) & 7) << 8);
                    k3_quant_copy8(qw + 32 * h + 8 * l, k3_iq1_lut[idx]);
                }
                s[h] = 2 * ((qhh >> 12) & 7) + 1;
                sd[h] = (qhh & 0x8000) ? -s[h] : s[h];
            }
            const int8_t *xg = x + 256 * b + 32 * ib;
#if defined(__ARM_FEATURE_SVE)
            vacc = k3_quant_acc_pair32(vacc, qw, xg, s[0], s[1]);
            /* sum(x) per group, scaled by S*sign, in the same shape */
            vdacc = k3_quant_acc_pair32(vdacc, k3_iq_ones64, xg, sd[0], sd[1]);
#else
            acc += k3_quant_dot_i8_pair32(qw, xg, s[0], s[1]);
            dacc += k3_quant_dot_i8_pair32(k3_iq_ones64, xg, sd[0], sd[1]);
#endif
        }
#if defined(__ARM_FEATURE_SVE)
        acc = k3_quant_acc_total(vacc);
        dacc = k3_quant_acc_total(vdacc);
#endif
        out += ggml_fp16_to_fp32(w[b].d) * xd *
               ((float)acc + 0.125f * (float)dacc);
    }
    return out;
}

static inline float k3_quant_iq3_xxs_q8_row(const block_iq3_xxs *w,
                                            const int8_t *x, float xd, int nb) {
    float out = 0.0f; int8_t qw[64], sg[64];
    for (int b = 0; b < nb; ++b) {
        const uint8_t *q3 = w[b].qs, *gas = w[b].qs + 64; int64_t part = 0;
#if defined(__ARM_FEATURE_SVE)
        svint32_t vacc = svdup_s32(0);
#endif
        for (int ib = 0; ib < 8; ib += 2) {
            int s[2];
            for (int h = 0; h < 2; ++h) {
                uint32_t aux; memcpy(&aux, gas + 4 * (ib + h), sizeof(aux));
                int8_t *qh8 = qw + 32 * h, *sh8 = sg + 32 * h;
                const uint8_t *q3h = q3 + 8 * (ib + h);
                for (int l = 0; l < 4; ++l) {
                    /* IQ3 packs two 4-byte grid entries per group. */
                    memcpy(qh8 + 8 * l, iq3xxs_grid + q3h[2 * l], 4);
                    memcpy(qh8 + 8 * l + 4, iq3xxs_grid + q3h[2 * l + 1], 4);
                    k3_quant_copy8(sh8 + 8 * l,
                                   k3_iq_sign_lut[ksigns_iq2xs[(aux >> (7 * l)) & 127]]);
                }
                s[h] = 2 * (int)(aux >> 28) + 1;
            }
            k3_quant_apply_signs64(qw, sg);
#if defined(__ARM_FEATURE_SVE)
            vacc = k3_quant_acc_pair32(vacc, qw, x + 256 * b + 32 * ib,
                                       s[0], s[1]);
#else
            part += k3_quant_dot_i8_pair32(qw, x + 256 * b + 32 * ib,
                                           s[0], s[1]);
#endif
        }
#if defined(__ARM_FEATURE_SVE)
        part = k3_quant_acc_total(vacc);
#endif
        out += ggml_fp16_to_fp32(w[b].d) * xd * (float)part * 0.25f;
    }
    return out;
}

static inline float k3_quant_q8_0_a16_row(const block_q8_0 *w,
                                          const int16_t *x, float xd, int cols) {
#if defined(__ARM_FEATURE_SVE)
    /* Widen the stored bytes in the load itself.  The old scalar 32-entry
     * staging loop cost more than the dot product it fed.
     *
     * The per-block reduction stays in the vector domain: `svaddv` followed by
     * moving the result to a GPR is a long-latency SIMD->GPR crossing, and at
     * cols=7168 that is 224 of them per row.  Converting each block's int64
     * lanes to double and folding the block scale in with svmla leaves exactly
     * one horizontal reduction per row. */
    const svbool_t pg = svwhilelt_b16(0, 32);
    const svbool_t pd = svptrue_b64();
    svfloat64_t facc = svdup_f64(0.0);
    for (int b = 0; b < cols / 32; ++b) {
        svint64_t acc = svdot_s64(svdup_s64(0),
                                  svld1sb_s16(pg, w[b].qs),
                                  svld1_s16(pg, x + 32 * b));
        facc = svmla_n_f64_x(pd, facc, svcvt_f64_s64_x(pd, acc),
                             (double)ggml_fp16_to_fp32(w[b].d));
    }
    return (float)(svaddv_f64(pd, facc) * (double)xd);
#else
    float out = 0.0f; int16_t qw[32];
    for (int b = 0; b < cols / 32; ++b) {
        for (int j = 0; j < 32; ++j) qw[j] = w[b].qs[j];
        out += ggml_fp16_to_fp32(w[b].d) * xd *
               (float)k3_quant_dot_i16(qw, x + 32 * b, 32);
    }
    return out;
#endif
}

/* Q8_0 W8A16: eight output rows share one pass over the int16 activation.
 * This is the fast Q8_0 path on A64FX — one 32-element block is exactly one
 * 512-bit vector of int16, so svdot_s64 runs at full width, whereas the int8
 * form leaves half the lanes idle.  Eight rows give the per-block svaddv
 * reductions independent chains to overlap. */
static inline void k3_quant_q8_0_a16_rows8(const uint8_t *base, size_t row_bytes,
                                           int nrows, const int16_t *x, float xd,
                                           int cols, float *out) {
#if defined(__ARM_FEATURE_SVE)
    const svbool_t pg = svwhilelt_b16(0, 32);
    float acc[8] = {0.0f};
    for (int b = 0; b < cols / 32; ++b) {
        svint16_t xv = svld1_s16(pg, x + 32 * b);
        for (int r = 0; r < nrows; ++r) {
            const block_q8_0 *w =
                (const block_q8_0 *)(base + (size_t)r * row_bytes);
            svint64_t d = svdot_s64(svdup_s64(0), svld1sb_s16(pg, w[b].qs), xv);
            acc[r] += ggml_fp16_to_fp32(w[b].d) *
                      (float)svaddv_s64(svptrue_b64(), d);
        }
    }
    for (int r = 0; r < nrows; ++r) out[r] = acc[r] * xd;
#else
    for (int r = 0; r < nrows; ++r) {
        const block_q8_0 *w = (const block_q8_0 *)(base + (size_t)r * row_bytes);
        float sum = 0.0f;
        for (int b = 0; b < cols / 32; ++b) {
            int64_t part = 0;
            for (int j = 0; j < 32; ++j) part += (int)w[b].qs[j] * x[32 * b + j];
            sum += ggml_fp16_to_fp32(w[b].d) * (float)part;
        }
        out[r] = sum * xd;
    }
#endif
}

/* Q8_0 W8A8: eight output rows share one pass over the quantized activation.
 * Each block owns a scale, so the int32 reduction cannot be deferred across
 * blocks; what the blocking buys is one activation load per eight rows and
 * eight independent svaddv chains to cover the reduction latency.  Loading
 * w[b].qs straight into the vector also removes the per-block 32-byte memcpy
 * the single-row kernel needed to reach an aligned staging buffer. */
static inline void k3_quant_q8_0_q8_rows8(const uint8_t *base, size_t row_bytes,
                                          int nrows, const int8_t *x, float xd,
                                          int cols, float *out) {
#if defined(__ARM_FEATURE_SVE)
    const svbool_t pg = svwhilelt_b8(0, 32);
    float acc[8] = {0.0f};
    for (int b = 0; b < cols / 32; ++b) {
        svint8_t xv = svld1_s8(pg, x + 32 * b);
        for (int r = 0; r < nrows; ++r) {
            const block_q8_0 *w =
                (const block_q8_0 *)(base + (size_t)r * row_bytes);
            svint32_t d = svdot_s32(svdup_s32(0), svld1_s8(pg, w[b].qs), xv);
            acc[r] += ggml_fp16_to_fp32(w[b].d) *
                      (float)svaddv_s32(svptrue_b32(), d);
        }
    }
    for (int r = 0; r < nrows; ++r) out[r] = acc[r] * xd;
#else
    for (int r = 0; r < nrows; ++r) {
        const block_q8_0 *w = (const block_q8_0 *)(base + (size_t)r * row_bytes);
        float sum = 0.0f;
        for (int b = 0; b < cols / 32; ++b) {
            int32_t part = 0;
            for (int j = 0; j < 32; ++j) part += (int)w[b].qs[j] * x[32 * b + j];
            sum += ggml_fp16_to_fp32(w[b].d) * (float)part;
        }
        out[r] = sum * xd;
    }
#endif
}

static inline float k3_quant_q8_0_q8_row(const block_q8_0 *w,
                                         const int8_t *x, float xd, int cols) {
    float out = 0.0f;
    k3_quant_q8_0_q8_rows8((const uint8_t *)w, 0, 1, x, xd, cols, &out);
    return out;
}

static inline float k3_quant_sve_a16_row(const uint8_t *row, int type,
                                         const k3_quant_workspace *ws, int cols) {
    int nb = cols / 256;
    if (type == K3_Q_Q8_0)
        return k3_quant_q8_0_a16_row((const block_q8_0 *)row, ws->a16, ws->scale, cols);
    if (type == K3_Q_IQ1_S)
        return k3_quant_iq1_s_row((const block_iq1_s *)row, ws->a16, ws->scale, nb);
    if (type == K3_Q_IQ2_XS)
        return k3_quant_iq2_xs_row((const block_iq2_xs *)row, ws->a16, ws->scale, nb);
    if (type == K3_Q_IQ2_XXS)
        return k3_quant_iq2_xxs_row((const block_iq2_xxs *)row, ws->a16, ws->scale, nb);
    return k3_quant_iq3_xxs_row((const block_iq3_xxs *)row, ws->a16, ws->scale, nb);
}

/* Scalar, source-faithful path.  Kept exact so the kernel tests have a
 * reference to gate the SDOT kernels against. */
static inline int k3_quant_matvec_ref(float *out, const k3_quant_matrix *m,
                                      const float *x, int threads) {
    if (!m || !out || !x || !k3_quant_valid_shape(m->type, m->rows, m->cols))
        return -1;
    if (m->row_bytes != k3_quant_row_bytes(m->type, m->cols)) return -1;
#if defined(_OPENMP)
    k3_quant_set_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < m->rows; ++r)
        out[r] = k3_quant_dot_row(m->data + (size_t)r * m->row_bytes,
                                  m->type, x, m->cols);
    return 0;
}

/* Row blocking for Q8_0 is a measured trade, not a given: eight rows hold
 * eight weight streams in flight, which helps the reduction chains but costs
 * L1 residency.  Default off after benchmarking a real IQ1 layer; set
 * K3_Q8_ROWS8=1 to re-measure. */
static inline int k3_quant_q8_0_rows8_enabled(void) {
    /* Resolved once: this is consulted inside the parallel row loop. */
    static int cached = -1;
    if (cached < 0) {
        const char *s = getenv("K3_Q8_ROWS8");
        cached = (s && *s == '1') ? 1 : 0;
    }
    return cached;
}

/* Run one matvec against an activation that the caller already quantized.
 * Projections issue many eight-row blocks against the same input; preparing
 * the workspace per block re-scanned the whole activation once per block. */
static inline int k3_quant_matvec_ws(float *out, const k3_quant_matrix *m,
                                     const float *x,
                                     const k3_quant_workspace *ws,
                                     int threads, int mode) {
    if (!m || !out || !ws || !k3_quant_valid_shape(m->type, m->rows, m->cols))
        return -1;
    if (ws->cols < m->cols) return -1;
    if (m->type == K3_Q_Q8_0 && ws->a16 && ws->a16_ready) {
        int blocks = (m->rows + 7) / 8;
#if defined(_OPENMP)
        k3_quant_set_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int rb = 0; rb < blocks; ++rb) {
            int r = rb * 8;
            int nr = m->rows - r < 8 ? m->rows - r : 8;
            if (k3_quant_q8_0_rows8_enabled())
                k3_quant_q8_0_a16_rows8(m->data + (size_t)r * m->row_bytes,
                                        m->row_bytes, nr, ws->a16,
                                        ws->scale_a16, m->cols, out + r);
            else
                for (int i = 0; i < nr; ++i)
                    out[r + i] = k3_quant_q8_0_a16_row(
                        (const block_q8_0 *)(m->data +
                            (size_t)(r + i) * m->row_bytes),
                        ws->a16, ws->scale_a16, m->cols);
        }
        return 0;
    }
    /* Note: a `threads <= 1` serial fast path to skip the one-thread parallel
     * region the full runner enters per eight-row task was tried both as an
     * `if` clause on the pragma and as a separate branch over a shared
     * dispatch helper.  Both cost ~6% on the 47-thread path, because the
     * inline chain below lets the compiler hoist the mode/type tests out of
     * the loop and neither form preserves that.  Left alone deliberately. */
#if defined(_OPENMP)
    k3_quant_set_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < m->rows; ++r) {
        const uint8_t *row = m->data + (size_t)r * m->row_bytes;
        if (mode == K3_QUANT_SVE_A16 &&
            (m->type == K3_Q_Q8_0 || m->type == K3_Q_IQ1_S ||
             m->type == K3_Q_IQ2_XS || m->type == K3_Q_IQ2_XXS))
            out[r] = k3_quant_sve_a16_row(row, m->type, ws, m->cols);
        else if (mode == K3_QUANT_SVE_A16 && m->type == K3_Q_IQ3_XXS)
            out[r] = k3_quant_iq3_xxs_row((const block_iq3_xxs *)row, ws->a16, ws->scale, m->cols / 256);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_IQ1_S)
            out[r] = k3_quant_iq1_s_q8_row((const block_iq1_s *)row, ws->q8, ws->scale, m->cols / 256);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_IQ2_XS)
            out[r] = k3_quant_iq2_xs_q8_row((const block_iq2_xs *)row, ws->q8, ws->scale, m->cols / 256);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_IQ2_XXS)
            out[r] = k3_quant_iq2_xxs_q8_row((const block_iq2_xxs *)row, ws->q8, ws->scale, m->cols / 256);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_IQ3_XXS)
            out[r] = k3_quant_iq3_xxs_q8_row((const block_iq3_xxs *)row, ws->q8, ws->scale, m->cols / 256);
        else out[r] = k3_quant_dot_row_ref(row, m->type, x, m->cols);
    }
    return 0;
}

static inline int k3_quant_matvec_mode(float *out, const k3_quant_matrix *m,
                                       const float *x, int threads,
                                       int mode) {
    if (mode == K3_QUANT_REFERENCE)
        return k3_quant_matvec_ref(out, m, x, threads);
    if (!m || !out || !x || !k3_quant_valid_shape(m->type, m->rows, m->cols))
        return -1;
    k3_quant_ensure_luts();
    k3_quant_workspace ws = {0};
    if (k3_quant_workspace_prepare(&ws, m->cols, mode)) return -1;
    /* Quantize only the activation form the selected kernel reads.  Doing both
     * costs a full extra pass over cols, which is visible on wide, short
     * tensors (attn_output is 598x12288). */
    int needs_q8 = mode == K3_QUANT_SVE_Q8 && m->type != K3_Q_Q8_0;
    if (!needs_q8) {
        k3_quant_prepare_a16(&ws, x, m->cols);
        ws.scale_a16 = ws.scale;
    } else {
        k3_quant_prepare_q8(&ws, x, m->cols);
    }
    ws.a16_ready = !needs_q8;
    int rc = k3_quant_matvec_ws(out, m, x, &ws, threads, mode);
    k3_quant_workspace_free(&ws);
    return rc;
}

/* Resolved once: the full runner reads this per eight-row task, which showed
 * up as ~1% of cycles in getenv alone. */
static inline int k3_quant_kernel_mode_env(void) {
    static int cached = -1;
    int m = cached;
    if (m < 0) {
        const char *s = getenv("K3_QUANT_KERNEL");
        m = K3_QUANT_SVE_Q8;
        if (s && !strcmp(s, "reference")) m = K3_QUANT_REFERENCE;
        else if (s && !strcmp(s, "sve-a16")) m = K3_QUANT_SVE_A16;
        cached = m;
    }
    return m;
}

/* The default entry point takes the SDOT path.  It used to route every caller
 * to the scalar reference, which malloc'd an fp32 row inside the OpenMP loop;
 * on a real IQ1 layer that is 11x slower than sve-q8. */
static inline int k3_quant_matvec(float *out, const k3_quant_matrix *m,
                                  const float *x, int threads) {
    if (!m || !out || !x || !k3_quant_valid_shape(m->type, m->rows, m->cols))
        return -1;
    if (m->row_bytes != k3_quant_row_bytes(m->type, m->cols)) return -1;
    int mode = k3_quant_kernel_mode_env();
    if (mode != K3_QUANT_REFERENCE &&
        !k3_quant_matvec_mode(out, m, x, threads, mode))
        return 0;
    return k3_quant_matvec_ref(out, m, x, threads);
}

#endif
