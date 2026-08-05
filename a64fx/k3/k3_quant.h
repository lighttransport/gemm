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
    float scale;
} k3_quant_workspace;

/* Decode LUTs are small enough to stay resident in the shared L2/L1 working
 * set and remove the scalar grid/sign expansion from every output row. */
static int8_t k3_iq1_lut[2048][8];
static int8_t k3_iq2_lut[65536][8];
static pthread_once_t k3_quant_lut_once = PTHREAD_ONCE_INIT;

static void k3_quant_init_luts(void) {
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

static inline int k3_quant_matvec(float *out, const k3_quant_matrix *m,
                                  const float *x, int threads);

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
    if (!ws || cols <= 0 || cols % 256) return -1;
    if (ws->cols >= cols && ((mode == K3_QUANT_SVE_A16 && ws->a16) ||
                             (mode == K3_QUANT_SVE_Q8 && ws->q8))) {
        return 0;
    }
    free(ws->a16); free(ws->q8);
    memset(ws, 0, sizeof(*ws));
    ws->cols = cols;
    if (mode == K3_QUANT_SVE_A16)
        ws->a16 = (int16_t *)malloc((size_t)cols * sizeof(*ws->a16));
    else if (mode == K3_QUANT_SVE_Q8)
        ws->q8 = (int8_t *)malloc((size_t)cols * sizeof(*ws->q8));
    return (mode == K3_QUANT_SVE_A16 ? (void *)ws->a16 : (void *)ws->q8) ? 0 : -1;
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
    float out = 0.0f; int16_t qw[32];
    for (int b = 0; b < nb; ++b) {
        int64_t part = 0;
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                uint16_t code = w[b].qs[4 * ib + l];
                for (int j = 0; j < 8; ++j) qw[8 * l + j] = k3_iq2_lut[code][j];
            }
            int s0 = 2 * (w[b].scales[ib] & 15) + 1;
            int s1 = 2 * (w[b].scales[ib] >> 4) + 1;
            part += k3_quant_dot_i16(qw, x + 256 * b + 32 * ib, 16) * s0;
            part += k3_quant_dot_i16(qw + 16, x + 256 * b + 32 * ib + 16, 16) * s1;
        }
        out += ggml_fp16_to_fp32(w[b].d) * xd * (float)part * 0.125f;
    }
    return out;
}

static inline float k3_quant_iq2_xxs_row(const block_iq2_xxs *w,
                                         const int16_t *x, float xd, int nb) {
    float out = 0.0f; uint32_t aux[2]; int16_t qw[32];
    for (int b = 0; b < nb; ++b) {
        for (int ib = 0; ib < 8; ++ib) {
            memcpy(aux, w[b].qs + 4 * ib, sizeof(aux));
            int64_t part = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *g = (const uint8_t *)(iq2xxs_grid + ((const uint8_t *)aux)[l]);
                uint8_t signs = ksigns_iq2xs[(aux[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j)
                    qw[8 * l + j] = (int16_t)((signs & kmask_iq2xs[j]) ? -(int)g[j] : (int)g[j]);
            }
            part += k3_quant_dot_i16(qw, x + 256 * b + 32 * ib, 32);
            out += ggml_fp16_to_fp32(w[b].d) * xd *
                   (0.5f + (float)(aux[1] >> 28)) * 0.25f * (float)part;
        }
    }
    return out;
}

static inline float k3_quant_iq1_s_row(const block_iq1_s *w,
                                       const int16_t *x, float xd, int nb) {
    float out = 0.0f; const uint8_t *qs; const uint16_t *qh;
    int16_t qw[32];
    for (int b = 0; b < nb; ++b) {
        const block_iq1_s *wb = w + b; qs = wb->qs; qh = wb->qh;
        for (int ib = 0; ib < 8; ++ib) {
            int64_t part = 0, xs = 0;
            float delta = qh[ib] & 0x8000 ? -0.125f : 0.125f;
            for (int l = 0; l < 4; ++l) {
                int idx = qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8);
                for (int j = 0; j < 8; ++j) qw[8 * l + j] = k3_iq1_lut[idx][j];
            }
            part = k3_quant_dot_i16(qw, x + 256 * b + 32 * ib, 32);
            for (int j = 0; j < 32; ++j) xs += x[256 * b + 32 * ib + j];
            out += ggml_fp16_to_fp32(wb->d) * xd *
                   (float)(2 * ((qh[ib] >> 12) & 7) + 1) *
                   ((float)part + delta * (float)xs);
            qs += 4;
        }
    }
    return out;
}

static inline float k3_quant_iq3_xxs_row(const block_iq3_xxs *w,
                                         const int16_t *x, float xd, int nb) {
    float out = 0.0f; int16_t qw[32];
    for (int b = 0; b < nb; ++b) {
        const uint8_t *q3 = w[b].qs, *gas = w[b].qs + 64;
        int64_t part = 0;
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t aux; memcpy(&aux, gas + 4 * ib, sizeof(aux));
            for (int l = 0; l < 4; ++l) {
                const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + q3[8 * ib + 2 * l]);
                const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + q3[8 * ib + 2 * l + 1]);
                uint8_t s = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                for (int j = 0; j < 4; ++j) {
                    qw[8 * l + j] = (int16_t)((s & kmask_iq2xs[j]) ? -(int)g1[j] : (int)g1[j]);
                    qw[8 * l + j + 4] = (int16_t)((s & kmask_iq2xs[j + 4]) ? -(int)g2[j] : (int)g2[j]);
                }
            }
            part += k3_quant_dot_i16(qw, x + 256 * b + 32 * ib, 32) * (2 * (int)(aux >> 28) + 1);
        }
        out += ggml_fp16_to_fp32(w[b].d) * xd * (float)part * 0.25f;
    }
    return out;
}

static inline float k3_quant_iq2_xs_q8_row(const block_iq2_xs *w,
                                           const int8_t *x, float xd, int nb) {
    float out = 0.0f; int8_t qw[32];
    for (int b = 0; b < nb; ++b) {
        int64_t part = 0;
        for (int ib = 0; ib < 8; ++ib) {
            for (int l = 0; l < 4; ++l) {
                uint16_t code = w[b].qs[4 * ib + l];
                for (int j = 0; j < 8; ++j) qw[8 * l + j] = k3_iq2_lut[code][j];
            }
            part += k3_quant_dot_i8(qw, x + 256 * b + 32 * ib, 16) * (2 * (w[b].scales[ib] & 15) + 1);
            part += k3_quant_dot_i8(qw + 16, x + 256 * b + 32 * ib + 16, 16) * (2 * (w[b].scales[ib] >> 4) + 1);
        }
        out += ggml_fp16_to_fp32(w[b].d) * xd * (float)part * 0.125f;
    }
    return out;
}

static inline float k3_quant_iq2_xxs_q8_row(const block_iq2_xxs *w,
                                            const int8_t *x, float xd, int nb) {
    float out = 0.0f; uint32_t aux[2]; int8_t qw[32];
    for (int b = 0; b < nb; ++b) {
        for (int ib = 0; ib < 8; ++ib) {
            memcpy(aux, w[b].qs + 4 * ib, sizeof(aux));
            int64_t part = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *g = (const uint8_t *)(iq2xxs_grid + ((const uint8_t *)aux)[l]);
                uint8_t signs = ksigns_iq2xs[(aux[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j)
                    qw[8 * l + j] = (int8_t)((signs & kmask_iq2xs[j]) ? -(int)g[j] : (int)g[j]);
            }
            part += k3_quant_dot_i8(qw, x + 256 * b + 32 * ib, 32);
            out += ggml_fp16_to_fp32(w[b].d) * xd *
                   (0.5f + (float)(aux[1] >> 28)) * 0.25f * (float)part;
        }
    }
    return out;
}

static inline float k3_quant_iq1_s_q8_row(const block_iq1_s *w,
                                           const int8_t *x, float xd, int nb) {
    float out = 0.0f; int8_t qw[32];
    for (int b = 0; b < nb; ++b) {
        const uint8_t *qs = w[b].qs; const uint16_t *qh = w[b].qh;
        for (int ib = 0; ib < 8; ++ib) {
            int8_t delta_sign = qh[ib] & 0x8000 ? -1 : 1;
            int64_t part = 0, xs = 0;
            for (int l = 0; l < 4; ++l) {
                int idx = qs[l] | (((qh[ib] >> (3 * l)) & 7) << 8);
                for (int j = 0; j < 8; ++j) qw[8 * l + j] = k3_iq1_lut[idx][j];
            }
            part = k3_quant_dot_i8(qw, x + 256 * b + 32 * ib, 32);
            for (int j = 0; j < 32; ++j) xs += x[256 * b + 32 * ib + j];
            out += ggml_fp16_to_fp32(w[b].d) * xd *
                   (float)(2 * ((qh[ib] >> 12) & 7) + 1) *
                   ((float)part + 0.125f * (float)delta_sign * (float)xs);
            qs += 4;
        }
    }
    return out;
}

static inline float k3_quant_iq3_xxs_q8_row(const block_iq3_xxs *w,
                                            const int8_t *x, float xd, int nb) {
    float out = 0.0f; int8_t qw[32];
    for (int b = 0; b < nb; ++b) {
        const uint8_t *q3 = w[b].qs, *gas = w[b].qs + 64; int64_t part = 0;
        for (int ib = 0; ib < 8; ++ib) {
            uint32_t aux; memcpy(&aux, gas + 4 * ib, sizeof(aux));
            for (int l = 0; l < 4; ++l) {
                const uint8_t *g1 = (const uint8_t *)(iq3xxs_grid + q3[8 * ib + 2 * l]);
                const uint8_t *g2 = (const uint8_t *)(iq3xxs_grid + q3[8 * ib + 2 * l + 1]);
                uint8_t s = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                for (int j = 0; j < 4; ++j) {
                    qw[8 * l + j] = (int8_t)((s & kmask_iq2xs[j]) ? -(int)g1[j] : (int)g1[j]);
                    qw[8 * l + j + 4] = (int8_t)((s & kmask_iq2xs[j + 4]) ? -(int)g2[j] : (int)g2[j]);
                }
            }
            part += k3_quant_dot_i8(qw, x + 256 * b + 32 * ib, 32) * (2 * (int)(aux >> 28) + 1);
        }
        out += ggml_fp16_to_fp32(w[b].d) * xd * (float)part * 0.25f;
    }
    return out;
}

static inline float k3_quant_q8_0_a16_row(const block_q8_0 *w,
                                          const int16_t *x, float xd, int cols) {
    float out = 0.0f; int16_t qw[32];
    for (int b = 0; b < cols / 32; ++b) {
        for (int j = 0; j < 32; ++j) qw[j] = w[b].qs[j];
        out += ggml_fp16_to_fp32(w[b].d) * xd *
               (float)k3_quant_dot_i16(qw, x + 32 * b, 32);
    }
    return out;
}

static inline float k3_quant_q8_0_q8_row(const block_q8_0 *w,
                                         const int8_t *x, float xd, int cols) {
    float out = 0.0f; int8_t qw[32];
    for (int b = 0; b < cols / 32; ++b) {
        memcpy(qw, w[b].qs, sizeof(qw));
        out += ggml_fp16_to_fp32(w[b].d) * xd *
               (float)k3_quant_dot_i8(qw, x + 32 * b, 32);
    }
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

static inline int k3_quant_matvec_mode(float *out, const k3_quant_matrix *m,
                                       const float *x, int threads,
                                       int mode) {
    if (mode == K3_QUANT_REFERENCE) return k3_quant_matvec(out, m, x, threads);
    if (!m || !out || !x || !k3_quant_valid_shape(m->type, m->rows, m->cols)) return -1;
    k3_quant_ensure_luts();
    k3_quant_workspace ws = {0};
    if (k3_quant_workspace_prepare(&ws, m->cols, mode)) return -1;
    if (mode == K3_QUANT_SVE_A16) k3_quant_prepare_a16(&ws, x, m->cols);
    else k3_quant_prepare_q8(&ws, x, m->cols);
#if defined(_OPENMP)
    omp_set_num_threads(threads > 0 ? threads : 1);
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < m->rows; ++r) {
        const uint8_t *row = m->data + (size_t)r * m->row_bytes;
        if (mode == K3_QUANT_SVE_A16 &&
            (m->type == K3_Q_Q8_0 || m->type == K3_Q_IQ1_S ||
             m->type == K3_Q_IQ2_XS || m->type == K3_Q_IQ2_XXS))
            out[r] = k3_quant_sve_a16_row(row, m->type, &ws, m->cols);
        else if (mode == K3_QUANT_SVE_A16 && m->type == K3_Q_IQ3_XXS)
            out[r] = k3_quant_iq3_xxs_row((const block_iq3_xxs *)row, ws.a16, ws.scale, m->cols / 256);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_Q8_0)
            out[r] = k3_quant_q8_0_q8_row((const block_q8_0 *)row, ws.q8, ws.scale, m->cols);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_IQ1_S)
            out[r] = k3_quant_iq1_s_q8_row((const block_iq1_s *)row, ws.q8, ws.scale, m->cols / 256);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_IQ2_XS)
            out[r] = k3_quant_iq2_xs_q8_row((const block_iq2_xs *)row, ws.q8, ws.scale, m->cols / 256);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_IQ2_XXS)
            out[r] = k3_quant_iq2_xxs_q8_row((const block_iq2_xxs *)row, ws.q8, ws.scale, m->cols / 256);
        else if (mode == K3_QUANT_SVE_Q8 && m->type == K3_Q_IQ3_XXS)
            out[r] = k3_quant_iq3_xxs_q8_row((const block_iq3_xxs *)row, ws.q8, ws.scale, m->cols / 256);
        else out[r] = k3_quant_dot_row_ref(row, m->type, x, m->cols);
    }
    k3_quant_workspace_free(&ws);
    return 0;
}

static inline int k3_quant_kernel_mode_env(void) {
    const char *s = getenv("K3_QUANT_KERNEL");
    if (s && !strcmp(s, "reference")) return K3_QUANT_REFERENCE;
    if (s && !strcmp(s, "sve-q8")) return K3_QUANT_SVE_Q8;
    return K3_QUANT_SVE_A16;
}

static inline int k3_quant_matvec(float *out, const k3_quant_matrix *m,
                                  const float *x, int threads) {
    if (!m || !out || !x || !k3_quant_valid_shape(m->type, m->rows, m->cols))
        return -1;
    if (m->row_bytes != k3_quant_row_bytes(m->type, m->cols)) return -1;
#if defined(_OPENMP)
    omp_set_num_threads(threads > 0 ? threads : 1);
#pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < m->rows; ++r)
        out[r] = k3_quant_dot_row(m->data + (size_t)r * m->row_bytes,
                                  m->type, x, m->cols);
    return 0;
}

#endif
