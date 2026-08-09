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

/* Optional packed form for the wide IQ projections.  GGUF keeps each row
 * compressed, but the decode kernel consumes four int8 values per output row
 * at a time.  This layout stores one 16-row tile as
 *
 *     [group 0, columns 0..3, rows 0..15][columns 4..7, ...] ...
 *
 * so every inner-loop weight access is a contiguous 64-byte SVE load.  The
 * packed buffer is deliberately separate from k3_quant_matrix: callers can
 * use it as a cache without changing the on-disk format or the reference
 * path.  It is most useful when a layer's weights are reused for many
 * decode tokens.
 */
typedef struct {
    int8_t *data;
    int rows, cols, type;
    size_t tile_bytes;
    int nibble;
    int8_t *scales;       /* per tile/group: s0[16], s1[16] */
    uint16_t *ds;         /* per tile/block: d[16] */
    size_t scale_tile_bytes;
    size_t d_tile_bytes;
} k3_quant_packed;

static inline void k3_quant_packed_free(k3_quant_packed *p) {
    if (!p) return;
    free(p->data);
    free(p->scales);
    free(p->ds);
    memset(p, 0, sizeof(*p));
}

static inline void *k3_quant_alloc_aligned(size_t bytes) {
    void *ptr = NULL;
    if (bytes && posix_memalign(&ptr, 64u, bytes) != 0) return NULL;
    return ptr;
}

static inline int k3_quant_pack_iq_rows16(k3_quant_packed *p,
                                          const k3_quant_matrix *m) {
    if (!p || !m || m->rows <= 0 || m->cols <= 0 || (m->rows & 15) ||
        m->cols % 256 ||
        (m->type != K3_Q_IQ1_S && m->type != K3_Q_IQ2_XS &&
         m->type != K3_Q_IQ2_XXS)) return -1;
    k3_quant_ensure_luts();
    const int nb = m->cols / 256;
    const int tiles = m->rows / 16;
    const size_t group_bytes = 8u * 64u;
    const size_t tile_bytes = (size_t)nb * 8u * group_bytes;
    int8_t *data = (int8_t *)k3_quant_alloc_aligned((size_t)tiles * tile_bytes);
    int8_t *scales = (int8_t *)k3_quant_alloc_aligned((size_t)tiles * nb * 8u * 32u);
    uint16_t *ds = (uint16_t *)k3_quant_alloc_aligned((size_t)tiles * nb * 16u * sizeof(*ds));
    if (!data || !scales || !ds) { free(data); free(scales); free(ds); return -1; }
    memset(p, 0, sizeof(*p));
    p->data = data; p->rows = m->rows; p->cols = m->cols;
    p->type = m->type; p->tile_bytes = tile_bytes; p->nibble = 0;
    p->scales = scales; p->ds = ds;
    p->scale_tile_bytes = (size_t)nb * 8u * 32u;
    p->d_tile_bytes = (size_t)nb * 16u * sizeof(*ds);

#if defined(_OPENMP)
    /* Parallel first-touch distributes the expanded cache across CMGs. */
    #pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < tiles; ++t) {
        for (int b = 0; b < nb; ++b) {
            for (int ib = 0; ib < 8; ++ib) {
                int8_t *dst = data + (size_t)t * tile_bytes +
                              ((size_t)b * 8u + (size_t)ib) * group_bytes;
                for (int r = 0; r < 16; ++r) {
                    const uint8_t *row = m->data +
                        (size_t)(t * 16 + r) * m->row_bytes;
                    int8_t q[32];
                    if (m->type == K3_Q_IQ1_S) {
                        const block_iq1_s *w = (const block_iq1_s *)row + b;
                        memcpy(ds + ((size_t)t * nb + b) * 16u + r, &w->d, sizeof(w->d));
                        int8_t *sm = scales + (((size_t)t * nb + b) * 8u + ib) * 32u;
                        sm[r] = (int8_t)(2 * ((w->qh[ib] >> 12) & 7) + 1);
                        sm[16 + r] = (w->qh[ib] & 0x8000) ? -sm[r] : sm[r];
                        for (int l = 0; l < 4; ++l) {
                            int idx = w->qs[4 * ib + l] |
                                (((w->qh[ib] >> (3 * l)) & 7) << 8);
                            memcpy(q + 8 * l, k3_iq1_lut[idx], 8);
                        }
                    } else if (m->type == K3_Q_IQ2_XS) {
                        const block_iq2_xs *w = (const block_iq2_xs *)row + b;
                        memcpy(ds + ((size_t)t * nb + b) * 16u + r, &w->d, sizeof(w->d));
                        int8_t *sm = scales + (((size_t)t * nb + b) * 8u + ib) * 32u;
                        sm[r] = (int8_t)(2 * (w->scales[ib] & 15) + 1);
                        sm[16 + r] = (int8_t)(2 * (w->scales[ib] >> 4) + 1);
                        for (int l = 0; l < 4; ++l)
                            memcpy(q + 8 * l,
                                   k3_iq2_lut[w->qs[4 * ib + l]], 8);
                    } else {
                        const block_iq2_xxs *w = (const block_iq2_xxs *)row + b;
                        uint32_t aux[2];
                        memcpy(aux, w->qs + 4 * ib, sizeof aux);
                        for (int l = 0; l < 4; ++l) {
                            memcpy(q + 8 * l,
                                   iq2xxs_grid + ((const uint8_t *)aux)[l], 8);
                            const int8_t *sg = k3_iq_sign_lut[
                                ksigns_iq2xs[(aux[1] >> (7 * l)) & 127]];
                            for (int j = 0; j < 8; ++j)
                                q[8 * l + j] = (int8_t)(q[8 * l + j] * sg[j]);
                        }
                    }
                    for (int c = 0; c < 8; ++c)
                        memcpy(dst + (size_t)c * 64u + (size_t)r * 4u,
                               q + 4 * c, 4);
                }
            }
        }
    }
    return 0;
}

/* Compress an already validated IQ1 row16 tile to signed nibbles.  The IQ1
 * grid values are in [-8, 7], so this is lossless and cuts the steady-state
 * weight stream in half.  IQ2_XS includes values outside this range and must
 * retain its byte-packed representation. */
static inline int k3_quant_pack_iq_rows16_nibble(k3_quant_packed *p,
                                                  const k3_quant_matrix *m) {
    k3_quant_packed tmp = {0};
    if (!m || m->type != K3_Q_IQ1_S) return -1;
    if (k3_quant_pack_iq_rows16(&tmp, m)) return -1;
    size_t byte_tile = tmp.tile_bytes;
    size_t n = (size_t)(m->rows / 16) * byte_tile;
    int8_t *data = (int8_t *)k3_quant_alloc_aligned(n / 2);
    if (!data) { k3_quant_packed_free(&tmp); return -1; }
    for (size_t i = 0; i < n; i += 2) {
        uint8_t a = (uint8_t)tmp.data[i] & 15u;
        uint8_t b = (uint8_t)tmp.data[i + 1] & 15u;
        data[i / 2] = (int8_t)(a | (uint8_t)(b << 4));
    }
    free(tmp.data);
    memset(p, 0, sizeof(*p));
    p->data = data; p->rows = m->rows; p->cols = m->cols;
    p->type = m->type; p->tile_bytes = byte_tile / 2; p->nibble = 1;
    p->scales = tmp.scales; p->ds = tmp.ds;
    p->scale_tile_bytes = tmp.scale_tile_bytes;
    p->d_tile_bytes = tmp.d_tile_bytes;
    return 0;
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

/* Same block fold as the q8 path, in the int64/f64 domain the a16 kernels
 * accumulate in: one horizontal reduction per row rather than per block. */
static inline svfloat64_t k3_quant_fold_block_a16(svfloat64_t facc,
                                                  svint64_t vacc, float scale) {
    svbool_t pd = svptrue_b64();
    return svmla_n_f64_x(pd, facc, svcvt_f64_s64_x(pd, vacc), (double)scale);
}

static inline float k3_quant_fold_total_a16(svfloat64_t facc) {
    return (float)svaddv_f64(svptrue_b64(), facc);
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

/* Fold one block's int32 lane accumulator into a float accumulator, scaled by
 * that block's own scale.  Lane summation is linear, so
 *   sum_b d_b * sum_lanes(vacc_b) == sum_lanes(sum_b d_b * vacc_b),
 * which lets a whole row finish with a single svaddv instead of one per
 * 256-element block -- twelve of them per row at cols=3072. */
static inline svfloat32_t k3_quant_fold_block(svfloat32_t facc, svint32_t vacc,
                                              float scale) {
    svbool_t lanes = svwhilelt_b32(0, 16);
    return svmla_n_f32_x(lanes, facc, svcvt_f32_s32_x(lanes, vacc), scale);
}

static inline float k3_quant_fold_total(svfloat32_t facc) {
    return svaddv_f32(svwhilelt_b32(0, 16), facc);
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
#if defined(__ARM_FEATURE_SVE)
    svfloat64_t facc = svdup_f64(0.0);
#endif
    for (int b = 0; b < nb; ++b) {
        int64_t part = 0; (void)part;
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
        facc = k3_quant_fold_block_a16(facc, vacc, ggml_fp16_to_fp32(w[b].d));
#else
        out += ggml_fp16_to_fp32(w[b].d) * (float)part;
#endif
    }
#if defined(__ARM_FEATURE_SVE)
    out = k3_quant_fold_total_a16(facc);
#endif
    return out * xd * 0.125f;
}

static inline float k3_quant_iq2_xxs_row(const block_iq2_xxs *w,
                                         const int16_t *x, float xd, int nb) {
    /* (0.5 + k) * 0.25 is (1 + 2k) * 0.125, so the group scale is an integer
     * and the whole block can accumulate before one reduction. */
    float out = 0.0f; uint32_t aux[2]; int8_t qw[32], sg[32];
#if defined(__ARM_FEATURE_SVE)
    svfloat64_t facc = svdup_f64(0.0);
#endif
    for (int b = 0; b < nb; ++b) {
        int64_t part = 0; (void)part;
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
        facc = k3_quant_fold_block_a16(facc, vacc, ggml_fp16_to_fp32(w[b].d));
#else
        out += ggml_fp16_to_fp32(w[b].d) * (float)part;
#endif
    }
#if defined(__ARM_FEATURE_SVE)
    out = k3_quant_fold_total_a16(facc);
#endif
    return out * xd * 0.125f;
}

static inline float k3_quant_iq1_s_row(const block_iq1_s *w,
                                       const int16_t *x, float xd, int nb) {
    /* Both terms carry an integer group scale, so the block accumulates once
     * for the weight product and once for sum(x) against ones. */
    float out = 0.0f; const uint8_t *qs; const uint16_t *qh;
    int8_t qw[32];
#if defined(__ARM_FEATURE_SVE)
    svfloat64_t facc = svdup_f64(0.0);
#endif
    for (int b = 0; b < nb; ++b) {
        const block_iq1_s *wb = w + b; qs = wb->qs; qh = wb->qh;
        int64_t acc = 0, dacc = 0; (void)acc; (void)dacc;
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
        /* acc + 0.125*dacc == (8*acc + dacc) / 8, and both are integers, so the
         * two accumulators combine with a shift and an add before a *single*
         * fold.  Folding them separately costs two f64 converts per block and
         * measured ~10% slower than the per-block svaddv it replaced. */
        svint64_t vcomb = svadd_s64_x(svptrue_b64(),
                                      svlsl_n_s64_x(svptrue_b64(), vacc, 3),
                                      vdacc);
        facc = k3_quant_fold_block_a16(facc, vcomb,
                                       ggml_fp16_to_fp32(wb->d) * 0.125f);
#else
        out += ggml_fp16_to_fp32(wb->d) *
               ((float)acc + 0.125f * (float)dacc);
#endif
    }
#if defined(__ARM_FEATURE_SVE)
    out = k3_quant_fold_total_a16(facc);
#endif
    return out * xd;
}

static inline float k3_quant_iq3_xxs_row(const block_iq3_xxs *w,
                                         const int16_t *x, float xd, int nb) {
    float out = 0.0f; int8_t qw[32], sg[32];
#if defined(__ARM_FEATURE_SVE)
    svfloat64_t facc = svdup_f64(0.0);
#endif
    for (int b = 0; b < nb; ++b) {
        const uint8_t *q3 = w[b].qs, *gas = w[b].qs + 64;
        int64_t part = 0; (void)part;
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
        facc = k3_quant_fold_block_a16(facc, vacc, ggml_fp16_to_fp32(w[b].d));
#else
        out += ggml_fp16_to_fp32(w[b].d) * (float)part;
#endif
    }
#if defined(__ARM_FEATURE_SVE)
    out = k3_quant_fold_total_a16(facc);
#endif
    return out * xd * 0.25f;
}

static inline float k3_quant_iq2_xs_q8_row(const block_iq2_xs *w,
                                           const int8_t *x, float xd, int nb) {
    /* IQ2_XS keeps the fully merged grid+sign LUT (512 KiB) even though the
     * same substitution is a loss for IQ2_XXS.  Measured both ways: folding
     * IQ2_XS down to the 4 KiB grid plus a sign multiply costs ~7%, while
     * expanding IQ2_XXS up to a 256 KiB merged table costs ~9%.  Table size
     * alone does not predict this; each format had to be measured. */
    float out = 0.0f; int8_t qw[64];
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t facc = svdup_f32(0.0f);
#endif
    for (int b = 0; b < nb; ++b) {
#if defined(__ARM_FEATURE_SVE)
        svint32_t vacc = svdup_s32(0);
#else
        int64_t part = 0;
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
        facc = k3_quant_fold_block(facc, vacc, ggml_fp16_to_fp32(w[b].d));
#else
        out += ggml_fp16_to_fp32(w[b].d) * (float)part;
#endif
    }
#if defined(__ARM_FEATURE_SVE)
    out = k3_quant_fold_total(facc);
#endif
    return out * xd * 0.125f;
}

static inline float k3_quant_iq2_xxs_q8_row(const block_iq2_xxs *w,
                                            const int8_t *x, float xd, int nb) {
    /* The per-group multiplier (0.5 + k) * 0.25 is (1 + 2k) * 0.125, so the
     * group scale is an integer and two groups can share one 64-lane dot. */
    float out = 0.0f; uint32_t aux[2]; int8_t qw[64], sg[64];
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t facc = svdup_f32(0.0f);
#endif
    for (int b = 0; b < nb; ++b) {
#if defined(__ARM_FEATURE_SVE)
        svint32_t vacc = svdup_s32(0);
#else
        int64_t part = 0;
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
        facc = k3_quant_fold_block(facc, vacc, ggml_fp16_to_fp32(w[b].d));
#else
        out += ggml_fp16_to_fp32(w[b].d) * (float)part;
#endif
    }
#if defined(__ARM_FEATURE_SVE)
    out = k3_quant_fold_total(facc);
#endif
    return out * xd * 0.125f;
}

#if defined(__ARM_FEATURE_SVE)
/* IQ1_S with sixteen output rows in lanes: the kernel ends in a plain vector
 * store and performs no horizontal reduction at all.
 *
 * Storage stays compressed -- unpacking IQ1_S to int8 would be a 5x expansion
 * and defeat the format -- so the row-major to lane-major transpose happens in
 * registers, as a strided gather out of the 16x32 unpack tile.
 *
 * a64fx/llm/WS3_GEMM_findings.md rejected this shape for bf16: 2x in isolation,
 * 0.60-1.0x at 48 threads, because that GEMM is memory-system-bound. IQ1_S is
 * 1.56 bits/weight, so it is unpack-bound instead and the result inverts --
 * measured 1.35x at 47 threads on 28672x3072 (ffn_down_exps), rel_l2 1.1e-07. */
#define K3_IQ_ROWS 16
static inline void k3_quant_iq1_s_q8_rows16(float *out, const uint8_t *base,
                                            size_t row_bytes, const int8_t *x,
                                            float xd, int nb) {
    const svbool_t l16 = svwhilelt_b32(0, K3_IQ_ROWS);
    const svbool_t p8 = svptrue_b8();
    const svuint32_t idx0 = svindex_u32(0, 32), idx1 = svindex_u32(4, 32);
    const svuint32_t idx2 = svindex_u32(8, 32), idx3 = svindex_u32(12, 32);
    svfloat32_t facc = svdup_f32(0.0f);
    int8_t tile[K3_IQ_ROWS * 32];
    float drow[K3_IQ_ROWS];
    int srow[K3_IQ_ROWS], sdrow[K3_IQ_ROWS];

    for (int b = 0; b < nb; ++b) {
        svint32_t bacc = svdup_s32(0);
        for (int r = 0; r < K3_IQ_ROWS; ++r)
            drow[r] = ggml_fp16_to_fp32(
                ((const block_iq1_s *)(base + (size_t)r * row_bytes))[b].d);

        for (int ib = 0; ib < 8; ++ib) {
            for (int r = 0; r < K3_IQ_ROWS; ++r) {
                const block_iq1_s *wb =
                    (const block_iq1_s *)(base + (size_t)r * row_bytes) + b;
                const uint8_t *qs = wb->qs + 4 * ib;
                uint16_t qh = wb->qh[ib];
                for (int l = 0; l < 4; ++l) {
                    int idx = qs[l] | (((qh >> (3 * l)) & 7) << 8);
                    k3_quant_copy8(tile + r * 32 + 8 * l, k3_iq1_lut[idx]);
                }
                srow[r] = 2 * ((qh >> 12) & 7) + 1;
                sdrow[r] = (qh & 0x8000) ? -srow[r] : srow[r];
            }
            /* svld1rq replicates 16 activation bytes to every 128-bit segment
             * and svdot_lane selects the quad, so the activation never leaves
             * the vector domain; a scalar load plus svdup would reintroduce the
             * GPR crossing this kernel exists to avoid. */
            svint32_t gacc = svdup_s32(0);
            const int8_t *xg = x + 256 * b + 32 * ib;
            for (int c = 0; c < 2; ++c) {
                svint8_t xq = svld1rq_s8(p8, xg + 16 * c);
                svuint32_t i0 = idx0, i1 = idx1, i2 = idx2, i3 = idx3;
                if (c) {
                    i0 = svadd_n_u32_x(l16, i0, 16);
                    i1 = svadd_n_u32_x(l16, i1, 16);
                    i2 = svadd_n_u32_x(l16, i2, 16);
                    i3 = svadd_n_u32_x(l16, i3, 16);
                }
                svint8_t w0 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i0));
                svint8_t w1 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i1));
                svint8_t w2 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i2));
                svint8_t w3 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i3));
                gacc = svdot_lane_s32(gacc, w0, xq, 0);
                gacc = svdot_lane_s32(gacc, w1, xq, 1);
                gacc = svdot_lane_s32(gacc, w2, xq, 2);
                gacc = svdot_lane_s32(gacc, w3, xq, 3);
            }
            /* sum(x) over the group is shared by all sixteen rows; the delta
             * term folds in as (8*acc + delta)/8 to keep the scales integral. */
            svint32_t xs0 = svld1sb_s32(svptrue_b32(), xg);
            svint32_t xs1 = svld1sb_s32(svptrue_b32(), xg + 16);
            int32_t xs = (int32_t)svaddv_s32(svptrue_b32(), xs0) +
                         (int32_t)svaddv_s32(svptrue_b32(), xs1);
            bacc = svmla_x(l16, bacc, svlsl_n_s32_x(l16, gacc, 3),
                           svld1_s32(l16, srow));
            bacc = svmla_x(l16, bacc, svdup_s32(xs), svld1_s32(l16, sdrow));
        }
        facc = svmla_x(l16, facc, svcvt_f32_s32_x(l16, bacc),
                       svmul_n_f32_x(l16, svld1(l16, drow), 0.125f));
    }
    svst1(l16, out, svmul_n_f32_x(l16, facc, xd));
}
/* IQ2_XS, same shape.  Its two half-group scales cover k 0-15 and 16-31, i.e.
 * quads 0-3 and 4-7, so the group needs two accumulators before the per-row
 * scale vectors apply. */
static inline void k3_quant_iq2_xs_q8_rows16(float *out, const uint8_t *base,
                                             size_t row_bytes, const int8_t *x,
                                             float xd, int nb) {
    const svbool_t l16 = svwhilelt_b32(0, K3_IQ_ROWS);
    const svbool_t p8 = svptrue_b8();
    const svuint32_t idx00 = svindex_u32(0, 32), idx01 = svindex_u32(4, 32);
    const svuint32_t idx02 = svindex_u32(8, 32), idx03 = svindex_u32(12, 32);
    const svuint32_t idx10 = svindex_u32(16, 32), idx11 = svindex_u32(20, 32);
    const svuint32_t idx12 = svindex_u32(24, 32), idx13 = svindex_u32(28, 32);
    svfloat32_t facc = svdup_f32(0.0f);
    int8_t tile[K3_IQ_ROWS * 32];
    float drow[K3_IQ_ROWS];
    int s0row[K3_IQ_ROWS], s1row[K3_IQ_ROWS];

    for (int b = 0; b < nb; ++b) {
        svint32_t bacc = svdup_s32(0);
        for (int r = 0; r < K3_IQ_ROWS; ++r)
            drow[r] = ggml_fp16_to_fp32(
                ((const block_iq2_xs *)(base + (size_t)r * row_bytes))[b].d);

        for (int ib = 0; ib < 8; ++ib) {
            for (int r = 0; r < K3_IQ_ROWS; ++r) {
                const block_iq2_xs *wb =
                    (const block_iq2_xs *)(base + (size_t)r * row_bytes) + b;
                for (int l = 0; l < 4; ++l)
                    k3_quant_copy8(tile + r * 32 + 8 * l,
                                   k3_iq2_lut[wb->qs[4 * ib + l]]);
                s0row[r] = 2 * (wb->scales[ib] & 15) + 1;
                s1row[r] = 2 * (wb->scales[ib] >> 4) + 1;
            }
            const int8_t *xg = x + 256 * b + 32 * ib;
            for (int c = 0; c < 2; ++c) {
                svint8_t xq = svld1rq_s8(p8, xg + 16 * c);
                svuint32_t i0 = c ? idx10 : idx00, i1 = c ? idx11 : idx01;
                svuint32_t i2 = c ? idx12 : idx02, i3 = c ? idx13 : idx03;
                svint8_t w0 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i0));
                svint8_t w1 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i1));
                svint8_t w2 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i2));
                svint8_t w3 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i3));
                svint32_t gacc = svdup_s32(0);
                gacc = svdot_lane_s32(gacc, w0, xq, 0);
                gacc = svdot_lane_s32(gacc, w1, xq, 1);
                gacc = svdot_lane_s32(gacc, w2, xq, 2);
                gacc = svdot_lane_s32(gacc, w3, xq, 3);
                bacc = svmla_x(l16, bacc, gacc,
                               svld1_s32(l16, c ? s1row : s0row));
            }
        }
        facc = svmla_x(l16, facc, svcvt_f32_s32_x(l16, bacc),
                       svmul_n_f32_x(l16, svld1(l16, drow), 0.125f));
    }
    svst1(l16, out, svmul_n_f32_x(l16, facc, xd));
}

/* Packed counterpart: each four-column slice of a 16-row group is already
 * contiguous, so the eight weight operands below are ordinary SVE loads. */
#if defined(__ARM_FEATURE_SVE)
static inline svint8_t k3_quant_unpack_iq_nibbles(const int8_t *src,
                                                   svbool_t all,
                                                   svbool_t p32,
                                                   svuint8_t mask_lo,
                                                   svbool_t odd) {
    svuint8_t packed = svld1_u8(p32, (const uint8_t *)src);
    packed = svsel_u8(p32, packed, svdup_u8(0));
    svuint8_t dup = svzip1_u8(packed, packed);
    svuint8_t lo = svand_u8_x(all, dup, mask_lo);
    svuint8_t hi = svlsr_n_u8_x(all, dup, 4);
    svint8_t idx = svreinterpret_s8_u8(svsel_u8(odd, hi, lo));
    /* Four-bit values use two's-complement nibbles. */
    idx = svlsl_n_s8_x(all, idx, 4);
    return svasr_n_s8_x(all, idx, 4);
}

static inline svint32_t k3_quant_dot_lane4(svint32_t acc, svint8_t w,
                                           svint8_t x, int lane) {
    switch (lane) {
    case 0: return svdot_lane_s32(acc, w, x, 0);
    case 1: return svdot_lane_s32(acc, w, x, 1);
    case 2: return svdot_lane_s32(acc, w, x, 2);
    default: return svdot_lane_s32(acc, w, x, 3);
    }
}

#endif
static inline void k3_quant_iq2_xxs_q8_rows16(float *, const uint8_t *,
                                              size_t, const int8_t *, float,
                                              int);
static inline void k3_quant_iq_packed_rows16(float * restrict out,
                                             const k3_quant_packed * restrict p,
                                             const k3_quant_matrix * restrict m,
                                             const int8_t * restrict x, float xd) {
    /* The packed layout is [group][4-column slice][row][4 bytes].  Use it
     * directly: this removes eight LUT expansions and four gather loads per
     * row/group from the decode path.  The original matrix is still needed
     * for FP16 group scales and IQ1/IQ2 per-group metadata. */
    if (!p || !p->data || !m || m->rows != 16 || m->cols <= 0) return;
#if defined(__ARM_FEATURE_SVE)
    const svbool_t l16 = svwhilelt_b32(0, 16);
    const svbool_t p8 = svptrue_b8();
    const svbool_t p32 = svwhilelt_b8(0, 32);
    const svuint8_t nibble_mask = svld1_u8(p8, ((const uint8_t[]){
        15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,
        15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,
        15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0,
        15,0,15,0,15,0,15,0,15,0,15,0,15,0,15,0 }));
    const svuint8_t nibble_shift = svld1_u8(p8, ((const uint8_t[]){
        0,4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,
        0,4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,
        0,4,0,4,0,4,0,4,0,4,0,4,0,4,0,4,
        0,4,0,4,0,4,0,4,0,4,0,4,0,4,0,4 }));
    const svbool_t odd = svcmpeq_n_u8(p8, nibble_shift, 4);
    const int nb = m->cols / 256;
    const size_t wstride = p->nibble ? 32u : 64u;
    for (int t = 0; t < m->rows; t += 16) {
        svfloat32_t facc = svdup_f32(0.0f);
        float drow[16];
        for (int b = 0; b < nb; ++b) {
            svint32_t bacc = svdup_s32(0);
            for (int r = 0; r < 16; ++r) {
                if (p->ds)
                    drow[r] = ggml_fp16_to_fp32(p->ds[(size_t)b * 16u + r]);
                else {
                    const uint8_t *row = m->data +
                        (size_t)(t + r) * m->row_bytes;
                    if (m->type == K3_Q_IQ1_S)
                        drow[r] = ggml_fp16_to_fp32(
                            (((const block_iq1_s *)row) + b)->d);
                    else
                        drow[r] = ggml_fp16_to_fp32(
                            (((const block_iq2_xs *)row) + b)->d);
                }
            }
            for (int ib = 0; ib < 8; ++ib) {
                const int8_t *group = p->data +
                    (size_t)(t / 16) * p->tile_bytes +
                    ((size_t)b * 8u + (size_t)ib) *
                    (p->nibble ? 256u : 512u);
                const int8_t *xg = x + 256 * b + 32 * ib;
                const int8_t *sm = p->scales ?
                    p->scales + ((size_t)b * 8u + ib) * 32u : NULL;
                svint32_t vs0 = sm ? svld1sb_s32(l16, sm) : svdup_s32(1);
                svint32_t vs1 = sm ? svld1sb_s32(l16, sm + 16) : svdup_s32(1);
                svint8_t xlo = svld1rq_s8(p8, xg);
                svint8_t xhi = svld1rq_s8(p8, xg + 16);
                if (m->type == K3_Q_IQ2_XS) {
                    svint32_t lo = svdup_s32(0), hi = svdup_s32(0);
                    svint8_t wv0 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 0u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 0u * wstride);
                    svint8_t wv1 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 1u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 1u * wstride);
                    svint8_t wv2 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 2u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 2u * wstride);
                    svint8_t wv3 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 3u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 3u * wstride);
                    svint8_t wv4 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 4u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 4u * wstride);
                    svint8_t wv5 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 5u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 5u * wstride);
                    svint8_t wv6 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 6u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 6u * wstride);
                    svint8_t wv7 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 7u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 7u * wstride);
                    lo = svdot_lane_s32(lo, wv0, xlo, 0);
                    lo = svdot_lane_s32(lo, wv1, xlo, 1);
                    lo = svdot_lane_s32(lo, wv2, xlo, 2);
                    lo = svdot_lane_s32(lo, wv3, xlo, 3);
                    hi = svdot_lane_s32(hi, wv4, xhi, 0);
                    hi = svdot_lane_s32(hi, wv5, xhi, 1);
                    hi = svdot_lane_s32(hi, wv6, xhi, 2);
                    hi = svdot_lane_s32(hi, wv7, xhi, 3);
                    bacc = svmla_x(l16, bacc, lo, vs0);
                    bacc = svmla_x(l16, bacc, hi, vs1);
                } else {
                    svint32_t acc = svdup_s32(0);
                    svint8_t wv0 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 0u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 0u * wstride);
                    svint8_t wv1 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 1u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 1u * wstride);
                    svint8_t wv2 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 2u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 2u * wstride);
                    svint8_t wv3 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 3u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 3u * wstride);
                    svint8_t wv4 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 4u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 4u * wstride);
                    svint8_t wv5 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 5u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 5u * wstride);
                    svint8_t wv6 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 6u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 6u * wstride);
                    svint8_t wv7 = p->nibble ? k3_quant_unpack_iq_nibbles(group + 7u * wstride, p8, p32, nibble_mask, odd) : svld1_s8(p8, group + 7u * wstride);
                    acc = svdot_lane_s32(acc, wv0, xlo, 0);
                    acc = svdot_lane_s32(acc, wv1, xlo, 1);
                    acc = svdot_lane_s32(acc, wv2, xlo, 2);
                    acc = svdot_lane_s32(acc, wv3, xlo, 3);
                    acc = svdot_lane_s32(acc, wv4, xhi, 0);
                    acc = svdot_lane_s32(acc, wv5, xhi, 1);
                    acc = svdot_lane_s32(acc, wv6, xhi, 2);
                    acc = svdot_lane_s32(acc, wv7, xhi, 3);
                    if (m->type == K3_Q_IQ1_S) {
                        svint32_t xs0 = svld1sb_s32(svptrue_b32(), xg);
                        svint32_t xs1 = svld1sb_s32(svptrue_b32(), xg + 16);
                        int32_t xs = (int32_t)svaddv_s32(svptrue_b32(), xs0) +
                                     (int32_t)svaddv_s32(svptrue_b32(), xs1);
                        bacc = svmla_x(l16, bacc, svlsl_n_s32_x(l16, acc, 3),
                                       vs0);
                        bacc = svmla_x(l16, bacc, svdup_s32(xs), vs1);
                    } else {
                        bacc = svmla_x(l16, bacc, acc, vs0);
                    }
                }
            }
            facc = svmla_x(l16, facc, svcvt_f32_s32_x(l16, bacc),
                           svmul_n_f32_x(l16, svld1(l16, drow), 0.125f));
        }
        svst1(l16, out + t, svmul_n_f32_x(l16, facc, xd));
    }
#else
    (void)out; (void)p; (void)m; (void)x; (void)xd;
#endif
}

static inline int k3_quant_matvec_packed_ws(float *out,
                                            const k3_quant_matrix *m,
                                            const k3_quant_packed *p,
                                            const k3_quant_workspace *ws) {
    if (!out || !m || !p || !ws || m->rows != 16 ||
        (m->type != K3_Q_IQ1_S && m->type != K3_Q_IQ2_XS) ||
        !ws->q8) return -1;
    k3_quant_iq_packed_rows16(out, p, m, ws->q8, ws->scale);
    return 0;
}


/* Token-major prefill companion to the decode entry point above.  Every
 * activation owns a prepared workspace; flattening token and row-tile work
 * keeps a single OpenMP team live and exposes enough independent tiles when a
 * TP shard has fewer rows than cores.  out_stride is measured in floats. */
static inline int k3_quant_matvec_packed_batch(float *out, size_t out_stride,
                                               const k3_quant_matrix *m,
                                               const k3_quant_packed *p,
                                               const k3_quant_workspace *ws,
                                               int batch) {
    if (!out || !m || !p || !ws || batch < 1 || (m->rows & 15) ||
        out_stride < (size_t)m->rows || p->tile_bytes == 0) return -1;
    int rc = 0;
    int tiles = m->rows / 16;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) reduction(|:rc)
#endif
    for (int task = 0; task < batch * tiles; ++task) {
        int token = task / tiles;
        int tile_index = task - token * tiles;
        int r = tile_index * 16;
        k3_quant_matrix sub = *m;
        k3_quant_packed tile = *p;
        sub.data += (size_t)r * m->row_bytes;
        sub.rows = 16;
        tile.data += (size_t)tile_index * p->tile_bytes;
        tile.rows = 16;
        if (tile.scales)
            tile.scales += (size_t)tile_index * p->scale_tile_bytes;
        if (tile.ds)
            tile.ds += (size_t)tile_index * p->d_tile_bytes / sizeof(*tile.ds);
        if (k3_quant_matvec_packed_ws(out + (size_t)token * out_stride + r,
                                      &sub, &tile, ws + token)) rc |= 1;
    }
    return rc ? -1 : 0;
}

/* IQ2_XXS: one integer scale per group, (0.5+k)*0.25 rewritten as
 * (1+2k)*0.125, plus the sign pattern applied to the unpack tile. */
static inline void k3_quant_iq2_xxs_q8_rows16(float *out, const uint8_t *base,
                                              size_t row_bytes, const int8_t *x,
                                              float xd, int nb) {
    const svbool_t l16 = svwhilelt_b32(0, K3_IQ_ROWS);
    const svbool_t p8 = svptrue_b8();
    const svuint32_t idx00 = svindex_u32(0, 32), idx01 = svindex_u32(4, 32);
    const svuint32_t idx02 = svindex_u32(8, 32), idx03 = svindex_u32(12, 32);
    const svuint32_t idx10 = svindex_u32(16, 32), idx11 = svindex_u32(20, 32);
    const svuint32_t idx12 = svindex_u32(24, 32), idx13 = svindex_u32(28, 32);
    svfloat32_t facc = svdup_f32(0.0f);
    int8_t tile[K3_IQ_ROWS * 32], sgt[K3_IQ_ROWS * 32];
    float drow[K3_IQ_ROWS];
    int srow[K3_IQ_ROWS];
    uint32_t aux[2];

    for (int b = 0; b < nb; ++b) {
        svint32_t bacc = svdup_s32(0);
        for (int r = 0; r < K3_IQ_ROWS; ++r)
            drow[r] = ggml_fp16_to_fp32(
                ((const block_iq2_xxs *)(base + (size_t)r * row_bytes))[b].d);

        for (int ib = 0; ib < 8; ++ib) {
            for (int r = 0; r < K3_IQ_ROWS; ++r) {
                const block_iq2_xxs *wb =
                    (const block_iq2_xxs *)(base + (size_t)r * row_bytes) + b;
                memcpy(aux, wb->qs + 4 * ib, sizeof(aux));
                int8_t *qr = tile + r * 32, *sr = sgt + r * 32;
                for (int l = 0; l < 4; ++l) {
                    k3_quant_copy8(qr + 8 * l,
                                   iq2xxs_grid + ((const uint8_t *)aux)[l]);
                    k3_quant_copy8(sr + 8 * l,
                                   k3_iq_sign_lut[ksigns_iq2xs[(aux[1] >> (7 * l)) & 127]]);
                }
                srow[r] = 1 + 2 * (int)(aux[1] >> 28);
            }
            /* Sign the whole tile in full-width steps.  Doing it per row costs
             * sixteen half-vector multiplies per group where the paired
             * per-row kernel needed one full-width one per two groups, and that
             * alone made a first version of this kernel slower than what it
             * replaced. */
            for (int o = 0; o < K3_IQ_ROWS * 32; o += 64)
                k3_quant_apply_signs64(tile + o, sgt + o);
            svint32_t gacc = svdup_s32(0);
            const int8_t *xg = x + 256 * b + 32 * ib;
            for (int c = 0; c < 2; ++c) {
                svint8_t xq = svld1rq_s8(p8, xg + 16 * c);
                svuint32_t i0 = c ? idx10 : idx00, i1 = c ? idx11 : idx01;
                svuint32_t i2 = c ? idx12 : idx02, i3 = c ? idx13 : idx03;
                svint8_t w0 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i0));
                svint8_t w1 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i1));
                svint8_t w2 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i2));
                svint8_t w3 = svreinterpret_s8_u32(svld1_gather_u32offset_u32(l16, (const uint32_t *)tile, i3));
                gacc = svdot_lane_s32(gacc, w0, xq, 0);
                gacc = svdot_lane_s32(gacc, w1, xq, 1);
                gacc = svdot_lane_s32(gacc, w2, xq, 2);
                gacc = svdot_lane_s32(gacc, w3, xq, 3);
            }
            bacc = svmla_x(l16, bacc, gacc, svld1_s32(l16, srow));
        }
        facc = svmla_x(l16, facc, svcvt_f32_s32_x(l16, bacc),
                       svmul_n_f32_x(l16, svld1(l16, drow), 0.125f));
    }
    svst1(l16, out, svmul_n_f32_x(l16, facc, xd));
}
#endif

static inline float k3_quant_iq1_s_q8_row(const block_iq1_s *w,
                                          const int8_t *x, float xd, int nb) {
    /* Both terms of the IQ1 group sum, S*<w,x> and S*delta*sum(x), carry an
     * integer group scale, so a pair of groups shares one 64-lane dot for the
     * product and another for the sum-against-ones. */
    float out = 0.0f; int8_t qw[64];
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t facc = svdup_f32(0.0f);
#endif
    for (int b = 0; b < nb; ++b) {
        const uint8_t *qs = w[b].qs; const uint16_t *qh = w[b].qh;
#if defined(__ARM_FEATURE_SVE)
        svint32_t vacc = svdup_s32(0), vdacc = svdup_s32(0);
#else
        int64_t acc = 0, dacc = 0;
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
        /* acc + 0.125*dacc == (8*acc + dacc) / 8; combining the accumulators
         * with a shift and an add costs one fold per block instead of two. */
        svbool_t lanes16 = svwhilelt_b32(0, 16);
        svint32_t vcomb = svadd_s32_x(lanes16,
                                      svlsl_n_s32_x(lanes16, vacc, 3), vdacc);
        facc = k3_quant_fold_block(facc, vcomb,
                                   ggml_fp16_to_fp32(w[b].d) * 0.125f);
#else
        out += ggml_fp16_to_fp32(w[b].d) *
               ((float)acc + 0.125f * (float)dacc);
#endif
    }
#if defined(__ARM_FEATURE_SVE)
    out = k3_quant_fold_total(facc);
#endif
    return out * xd;
}

static inline float k3_quant_iq3_xxs_q8_row(const block_iq3_xxs *w,
                                            const int8_t *x, float xd, int nb) {
    float out = 0.0f; int8_t qw[64], sg[64];
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t facc = svdup_f32(0.0f);
#endif
    for (int b = 0; b < nb; ++b) {
        const uint8_t *q3 = w[b].qs, *gas = w[b].qs + 64;
#if defined(__ARM_FEATURE_SVE)
        svint32_t vacc = svdup_s32(0);
#else
        int64_t part = 0;
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
        facc = k3_quant_fold_block(facc, vacc, ggml_fp16_to_fp32(w[b].d));
#else
        out += ggml_fp16_to_fp32(w[b].d) * (float)part;
#endif
    }
#if defined(__ARM_FEATURE_SVE)
    out = k3_quant_fold_total(facc);
#endif
    return out * xd * 0.25f;
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

/* Two-token prefill tile.  Q8 weights arrive from HBM once and are reused
 * against two activation vectors; activation reloads hit the much smaller
 * token workspace.  Each token retains the decode kernel's block order. */
static inline void k3_quant_q8_0_a16_rows8_batch2(
        const uint8_t *base, size_t row_bytes, int nrows,
        const k3_quant_workspace *ws, int ntokens, int cols,
        float *out, size_t out_stride) {
#if defined(__ARM_FEATURE_SVE)
    const svbool_t pg = svwhilelt_b16(0, 32);
    const svbool_t pd = svptrue_b64();
    for (int r = 0; r < nrows; ++r) {
        svfloat64_t f0 = svdup_f64(0.0), f1 = svdup_f64(0.0);
        for (int b = 0; b < cols / 32; ++b) {
            const block_q8_0 *w =
                (const block_q8_0 *)(base + (size_t)r * row_bytes);
            svint16_t wv = svld1sb_s16(pg, w[b].qs);
            double wd = (double)ggml_fp16_to_fp32(w[b].d);
            svint64_t d0 = svdot_s64(svdup_s64(0), wv,
                svld1_s16(pg, ws[0].a16 + 32 * b));
            f0 = svmla_n_f64_x(pd, f0, svcvt_f64_s64_x(pd, d0), wd);
            if (ntokens == 2) {
                svint64_t d1 = svdot_s64(svdup_s64(0), wv,
                    svld1_s16(pg, ws[1].a16 + 32 * b));
                f1 = svmla_n_f64_x(pd, f1, svcvt_f64_s64_x(pd, d1), wd);
            }
        }
        out[r] = (float)(svaddv_f64(pd, f0) * (double)ws[0].scale_a16);
        if (ntokens == 2)
            out[out_stride + r] =
                (float)(svaddv_f64(pd, f1) * (double)ws[1].scale_a16);
    }
#else
    for (int t = 0; t < ntokens; ++t)
        k3_quant_q8_0_a16_rows8(base, row_bytes, nrows, ws[t].a16,
                                ws[t].scale_a16, cols,
                                out + (size_t)t * out_stride);
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
#if !defined(_OPENMP)
    (void)threads;
#endif
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
#if !defined(_OPENMP)
    (void)threads;
#endif
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
#if defined(__ARM_FEATURE_SVE)
    if (mode == K3_QUANT_SVE_Q8 && m->rows >= K3_IQ_ROWS &&
        (m->type == K3_Q_IQ1_S || m->type == K3_Q_IQ2_XS ||
         m->type == K3_Q_IQ2_XXS)) {
        int blocks = m->rows / K3_IQ_ROWS, nb = m->cols / 256;
#if defined(_OPENMP)
        k3_quant_set_threads(threads);
#pragma omp parallel for schedule(static)
#endif
        for (int rb = 0; rb < blocks; ++rb) {
            int r = rb * K3_IQ_ROWS;
            const uint8_t *rp = m->data + (size_t)r * m->row_bytes;
            if (m->type == K3_Q_IQ1_S)
                k3_quant_iq1_s_q8_rows16(out + r, rp, m->row_bytes,
                                         ws->q8, ws->scale, nb);
            else if (m->type == K3_Q_IQ2_XS)
                k3_quant_iq2_xs_q8_rows16(out + r, rp, m->row_bytes,
                                          ws->q8, ws->scale, nb);
            else
                k3_quant_iq2_xxs_q8_rows16(out + r, rp, m->row_bytes,
                                           ws->q8, ws->scale, nb);
        }
        /* Rows past the last full sixteen fall back to the per-row kernels. */
        for (int r = blocks * K3_IQ_ROWS; r < m->rows; ++r) {
            const uint8_t *rp = m->data + (size_t)r * m->row_bytes;
            if (m->type == K3_Q_IQ1_S)
                out[r] = k3_quant_iq1_s_q8_row((const block_iq1_s *)rp,
                                               ws->q8, ws->scale, nb);
            else if (m->type == K3_Q_IQ2_XS)
                out[r] = k3_quant_iq2_xs_q8_row((const block_iq2_xs *)rp,
                                                ws->q8, ws->scale, nb);
            else
                out[r] = k3_quant_iq2_xxs_q8_row((const block_iq2_xxs *)rp,
                                                 ws->q8, ws->scale, nb);
        }
        return 0;
    }
#endif
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

/* Batch Q8_0 projections without opening one OpenMP region per token.  Q8_0
 * consumes the activation's A16 workspace in both sve-a16 and sve-q8 modes,
 * so each (token,row8) task is numerically identical to matvec_ws. */
static inline int k3_quant_q8_0_matvec_batch(float *out, size_t out_stride,
                                             const k3_quant_matrix *m,
                                             const k3_quant_workspace *ws,
                                             int batch, int threads) {
    (void)threads;
    if (!out || !m || !ws || m->type != K3_Q_Q8_0 || batch < 1 ||
        out_stride < (size_t)m->rows ||
        m->row_bytes != k3_quant_row_bytes(m->type, m->cols)) return -1;
    for (int b = 0; b < batch; ++b)
        if (!ws[b].a16 || !ws[b].a16_ready || ws[b].cols < m->cols) return -1;
    int blocks = (m->rows + 7) / 8;
    int token_groups = (batch + 1) / 2;
#if defined(_OPENMP)
    k3_quant_set_threads(threads);
    #pragma omp parallel for schedule(static)
#endif
    for (int task = 0; task < token_groups * blocks; ++task) {
        int token = (task / blocks) * 2;
        int r = (task % blocks) * 8;
        int ntokens = batch - token < 2 ? batch - token : 2;
        int nr = m->rows - r < 8 ? m->rows - r : 8;
        float *dst = out + (size_t)token * out_stride + r;
        k3_quant_q8_0_a16_rows8_batch2(
            m->data + (size_t)r * m->row_bytes, m->row_bytes, nr,
            ws + token, ntokens, m->cols, dst, out_stride);
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

/* GGUF MoE expert pool view: [cols, rows_per_expert, experts].  Selecting an
 * expert is only an offset calculation; the selected plane remains in the
 * original compressed format and uses the regular 2-D kernels. */
static inline int k3_quant_matvec_expert3d(float *out, const uint8_t *base,
                                           int type, int cols,
                                           int rows_per_expert, int experts,
                                           int expert, const float *x,
                                           int threads, int mode) {
    if (!base || expert < 0 || expert >= experts || cols <= 0 ||
        rows_per_expert <= 0 || k3_quant_row_bytes(type, cols) == 0)
        return -1;
    size_t row_bytes = k3_quant_row_bytes(type, cols);
    k3_quant_matrix plane = {
        base + (size_t)expert * (size_t)rows_per_expert * row_bytes,
        type, rows_per_expert, cols, row_bytes
    };
    return k3_quant_matvec_mode(out, &plane, x, threads, mode);
}

/* Select a contiguous intermediate-channel slice from an expert plane.  The
 * w1/w3 GGUF tensors are [input_cols, intermediate_rows, experts], so their
 * TP ownership is a row slice.  Keeping this as a view avoids expanding or
 * copying the complete expert during rank-local staging. */
static inline int k3_quant_matvec_expert3d_rowslice(
        float *out, const uint8_t *base, int type, int cols,
        int rows_per_expert, int experts, int expert, int row_start,
        int row_count, const float *x, int threads, int mode) {
    if (!base || expert < 0 || expert >= experts || row_start < 0 ||
        row_count <= 0 || row_start > rows_per_expert - row_count ||
        cols <= 0 || k3_quant_row_bytes(type, cols) == 0)
        return -1;
    size_t row_bytes = k3_quant_row_bytes(type, cols);
    const uint8_t *plane = base + (size_t)expert * (size_t)rows_per_expert * row_bytes;
    k3_quant_matrix slice = {
        plane + (size_t)row_start * row_bytes, type, row_count, cols, row_bytes
    };
    return k3_quant_matvec_mode(out, &slice, x, threads, mode);
}

/* Select a contiguous input-channel slice from an expert plane.  The w2
 * GGUF tensors are [intermediate_cols, output_rows, experts].  A column slice
 * is quantization-block aligned and therefore can be represented by a row
 * pointer plus a shortened row; no full-plane transpose is required. */
static inline int k3_quant_matvec_expert3d_colslice(
        float *out, const uint8_t *base, int type, int cols_per_expert,
        int rows_per_expert, int experts, int expert, int col_start,
        int col_count, const float *x, int threads, int mode) {
    int block = (type == K3_Q_Q8_0) ? 32 : 256;
    if (!base || expert < 0 || expert >= experts || col_start < 0 ||
        col_count <= 0 || col_start > cols_per_expert - col_count ||
        (col_start % block) || (col_count % block) || cols_per_expert <= 0 ||
        rows_per_expert <= 0 || k3_quant_row_bytes(type, cols_per_expert) == 0)
        return -1;
    size_t full_row_bytes = k3_quant_row_bytes(type, cols_per_expert);
    size_t slice_row_bytes = k3_quant_row_bytes(type, col_count);
    size_t col_bytes = (size_t)(col_start / block) *
                       (full_row_bytes / (size_t)(cols_per_expert / block));
    const uint8_t *plane = base + (size_t)expert *
                           (size_t)rows_per_expert * full_row_bytes;
    k3_quant_matrix slice = {
        plane + col_bytes, type, rows_per_expert, col_count, slice_row_bytes
    };
    /* Each row is strided by the original full row width, so a compact matrix
     * view is valid only when the slice is the complete row.  The TP runtime
     * uses the packed row-slice helper below for partial w2 columns. */
    if (col_start == 0 && col_count == cols_per_expert)
        return k3_quant_matvec_mode(out, &slice, x, threads, mode);
#if defined(_OPENMP)
    k3_quant_set_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows_per_expert; ++r) {
        const uint8_t *row = plane + (size_t)r * full_row_bytes + col_bytes;
        k3_quant_matrix one = {row, type, 1, col_count, slice_row_bytes};
        (void)k3_quant_matvec_mode(out + r, &one, x, 1, mode);
    }
    return 0;
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
