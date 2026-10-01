/* Reuse the validated GLM-5.2 A64FX mixed-IQ kernels in the GLM-5.3 graph.
 * Keeping this bridge in one translation unit avoids duplicating the large IQ
 * lookup tables in every consumer of glm53f_expert_kern.h. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GLM5_IMPL
#include "../../common/glm5.h"
#include "glm53f_iq_bridge.h"
#include "kern/glm53f_kern.h"
#include "glm53f_iq_fast.h"
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>

int glm53f_iq_type_supported(int type) {
    return type == GLM53F_GGML_Q4_K || type == GLM53F_GGML_Q5_K ||
           type == GLM53F_GGML_Q6_K ||
           type == GLM53F_GGML_IQ2_XS ||
           type == GLM53F_GGML_IQ3_XXS ||
           type == GLM53F_GGML_IQ4_XS;
}

size_t glm53f_iq_row_size(int type, int columns) {
    return columns > 0 ? dequant_row_size((uint32_t)type, columns) : 0;
}

static float q6_k_q8_row(const block_q6_K *w,
                         const glm5_iq_q8_block *x, int blocks) {
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svbool_t p4 = svwhilelt_b32(0, 4), p32bytes = svwhilelt_b8(0, 32);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float ds = ggml_fp16_to_fp32(w[b].d) * x[b].d;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql = w[b].ql + half * 64;
            const svuint8_t qh = svld1_u8(p32bytes, w[b].qh + half * 32);
            const int8_t *sc = w[b].scales + half * 8;
            for (int g = 0; g < 4; ++g) {
                svuint8_t q = svld1_u8(p32bytes, ql + ((g & 1) ? 32 : 0));
                q = (g >= 2) ? svlsr_n_u8_x(p8, q, 4) : svand_n_u8_x(p8, q, 15);
                svuint8_t hi = svand_n_u8_x(p8, svlsr_n_u8_x(p8, qh, 2 * g), 3);
                svint8_t qs = svreinterpret_s8_u8(svsub_n_u8_x(
                    p8, svorr_u8_x(p8, q, svlsl_n_u8_x(p8, hi, 4)), 32));
                const svint8_t xv = svld1_s8(p32bytes,
                    x[b].q + half * 128 + g * 32);
                const svint32_t dot = svdot_s32(svdup_s32(0), qs, xv);
                const svint32_t scale = svsel_s32(p4, svdup_s32(sc[2 * g]),
                                                       svdup_s32(sc[2 * g + 1]));
                acc = svmla_n_f32_x(p32, acc,
                    svcvt_f32_s32_x(p32, svmul_s32_x(p32, dot, scale)), ds);
            }
        }
    }
    return svaddv_f32(p32, acc);
}

static float q5_k_q8_row(const block_q5_K *w,
                         const glm5_iq_q8_block *x, int blocks) {
    const svbool_t p8 = svptrue_b8();
    const svbool_t p32 = svptrue_b32();
    const svbool_t p32bytes = svwhilelt_b8(0, 32);
    const svbool_t lo8 = svwhilelt_b32(0, 8);
    const svint8_t ones = svdup_s8(1);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float d = ggml_fp16_to_fp32(w[b].d);
        const float dm = ggml_fp16_to_fp32(w[b].dmin);
        svint32_t dots = svdup_s32(0), mins = svdup_s32(0);
        int is = 0;
        uint8_t bit0 = 1, bit1 = 2;
        for (int g = 0; g < 256; g += 64, is += 2, bit0 <<= 2, bit1 <<= 2) {
            uint8_t sc0, min0, sc1, min1;
            get_scale_min_k4(is, w[b].scales, &sc0, &min0);
            get_scale_min_k4(is + 1, w[b].scales, &sc1, &min1);
            const svuint8_t packed = svld1_u8(p32bytes, w[b].qs + (g >> 1));
            const svuint8_t high = svld1_u8(p32bytes, w[b].qh);
            svuint8_t uq0 = svand_n_u8_x(p8, packed, 15);
            svuint8_t uq1 = svlsr_n_u8_x(p8, packed, 4);
            uq0 = svadd_u8_x(p8, uq0, svsel_u8(svcmpne_n_u8(p32bytes,
                svand_n_u8_x(p8, high, bit0), 0), svdup_u8(16), svdup_u8(0)));
            uq1 = svadd_u8_x(p8, uq1, svsel_u8(svcmpne_n_u8(p32bytes,
                svand_n_u8_x(p8, high, bit1), 0), svdup_u8(16), svdup_u8(0)));
            const svint8_t qv = svreinterpret_s8_u8(svsplice_u8(p32bytes, uq0, uq1));
            const svint8_t xv = svld1_s8(p8, x[b].q + g);
            const svint32_t dot = svdot_s32(svdup_s32(0),qv,xv);
            const svint32_t sx = svdot_s32(svdup_s32(0),ones,xv);
            const svint32_t scv=svsel_s32(lo8,svdup_s32(sc0),svdup_s32(sc1));
            const svint32_t mnv=svsel_s32(lo8,svdup_s32(min0),svdup_s32(min1));
            dots = svmla_s32_x(p32, dots, dot, scv);
            mins = svmla_s32_x(p32, mins, sx, mnv);
        }
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, dots), x[b].d*d);
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, mins), -x[b].d*dm);
    }
    return svaddv_f32(p32, acc);
}

/* Q4_K is especially convenient with the existing Q8 activation blocks: the
 * affine correction only needs the sum of each 32-byte activation slice and
 * the main term maps directly to SVE SDOT. */
static float q4_k_q8_row(const block_q4_K *w,
                         const glm5_iq_q8_block *x, int blocks) {
    const svbool_t p8 = svptrue_b8();
    const svbool_t p32 = svptrue_b32();
    const svint8_t ones = svdup_s8(1);
    const svbool_t p32bytes = svwhilelt_b8(0, 32);
    const svbool_t lo8 = svwhilelt_b32(0, 8);
    svfloat32_t acc = svdup_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
        const float d = ggml_fp16_to_fp32(w[b].d);
        const float dm = ggml_fp16_to_fp32(w[b].dmin);
        svint32_t dots = svdup_s32(0), mins = svdup_s32(0);
        int is = 0;
        for (int g = 0; g < 256; g += 64, is += 2) {
            uint8_t sc0, min0, sc1, min1;
            get_scale_min_k4(is, w[b].scales, &sc0, &min0);
            get_scale_min_k4(is + 1, w[b].scales, &sc1, &min1);
            const svuint8_t packed = svld1_u8(p32bytes,
                                               w[b].qs + (g >> 1));
            const svint8_t q0 = svreinterpret_s8_u8(
                svand_n_u8_x(p8, packed, 15));
            const svint8_t q1 = svreinterpret_s8_u8(
                svlsr_n_u8_x(p8, packed, 4));
            const svint8_t qv = svsplice_s8(p32bytes, q0, q1);
            const svint8_t xv=svld1_s8(p8,x[b].q+g);
            const svint32_t dot=svdot_s32(svdup_s32(0),qv,xv);
            const svint32_t sx=svdot_s32(svdup_s32(0),ones,xv);
            const svint32_t scv=svsel_s32(lo8,svdup_s32(sc0),svdup_s32(sc1));
            const svint32_t mnv=svsel_s32(lo8,svdup_s32(min0),svdup_s32(min1));
            dots = svmla_s32_x(p32, dots, dot, scv);
            mins = svmla_s32_x(p32, mins, sx, mnv);
        }
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, dots), x[b].d*d);
        acc = svmla_n_f32_x(p32, acc, svcvt_f32_s32_x(p32, mins), -x[b].d*dm);
    }
    return svaddv_f32(p32, acc);
}

static float iq_row(int type, const uint8_t *row,
                    const glm5_iq_q8_block *xq, int blocks) {
    if (type == GLM53F_GGML_Q4_K)
        return q4_k_q8_row((const block_q4_K *)row, xq, blocks);
    if (type == GLM53F_GGML_Q5_K)
        return q5_k_q8_row((const block_q5_K *)row, xq, blocks);
    if (type == GLM53F_GGML_Q6_K)
        return q6_k_q8_row((const block_q6_K *)row, xq, blocks);
    glm5_tensor tensor = {0};
    tensor.type = (glm5_qtype)type;
    return glm5_iq_q8_row(&tensor, row, xq, blocks);
}

/* llama.cpp's Q8_0 matrix path does not multiply the stored Q8 weights by
 * the original F32 activation.  It first quantizes each 32-value activation
 * block to Q8_0 (including the fp16-rounded scale), then performs Q8 x Q8.
 * Preserve that contract here; the distinction is visible after mHC even
 * though both variants are close for an isolated projection. */
static void quantize_q8_0_activation(block_q8_0 *q, const float *x, int n) {
    const svbool_t pg = svptrue_b32();
    const int vl = (int)svcntw();
    for (int b = 0; b < n / 32; ++b) {
        const float *xb = x + 32 * b;
        svfloat32_t vmax = svdup_f32(0.0f);
        for (int j = 0; j < 32; j += vl)
            vmax = svmax_f32_x(pg, vmax,
                svabs_f32_x(pg, svld1_f32(pg, xb + j)));
        const float amax = svmaxv_f32(pg, vmax);
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        q[b].d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < 32; j += vl) {
            svfloat32_t v = svmul_n_f32_x(pg, svld1_f32(pg, xb + j), id);
            svint32_t iv = svcvt_s32_f32_x(pg, svrintn_f32_x(pg, v));
            svst1b_s32(pg, q[b].qs + j, iv);
        }
    }
}

static float q8_0_q8_0_row(const block_q8_0 *w,
                            const block_q8_0 *x, int blocks) {
    const svbool_t p8 = svptrue_b8();
    const svbool_t lo32 = svptrue_pat_b8(SV_VL32);
    const svbool_t hi32 = svnot_b_z(p8, lo32);
    const svbool_t p32 = svptrue_b32();
    const svbool_t lo8 = svptrue_pat_b32(SV_VL8);
    const svbool_t hi8 = svnot_b_z(p32, lo8);
    svfloat32_t acc = svdup_f32(0.0f);
    int b = 0;
    for (; b + 1 < blocks; b += 2) {
        svint8_t wq = svld1_s8(lo32, w[b].qs);
        wq = svadd_s8_x(p8, wq, svld1_s8(hi32, w[b].qs + 2));
        svint8_t xq = svld1_s8(lo32, x[b].qs);
        xq = svadd_s8_x(p8, xq, svld1_s8(hi32, x[b].qs + 2));
        const float s0 = ggml_fp16_to_fp32(w[b].d) *
                         ggml_fp16_to_fp32(x[b].d);
        const float s1 = ggml_fp16_to_fp32(w[b + 1].d) *
                         ggml_fp16_to_fp32(x[b + 1].d);
        const svfloat32_t scale = svdup_f32_m(
            svdup_f32_z(lo8, s0), hi8, s1);
        acc = svmla_f32_m(p32, acc,
            svcvt_f32_s32_x(p32, svdot_s32(svdup_s32(0), wq, xq)), scale);
    }
    float sum = svaddv_f32(p32, acc);
    if (b < blocks) {
        int dot = 0;
        for (int j = 0; j < 32; ++j)
            dot += (int)w[b].qs[j] * (int)x[b].qs[j];
        sum += (float)dot * ggml_fp16_to_fp32(w[b].d) *
               ggml_fp16_to_fp32(x[b].d);
    }
    return sum;
}

int glm53f_native_type_supported(int type) {
    return glm53f_iq_type_supported(type) || type == GLM53F_GGML_Q8_0 ||
           type == GLM53F_NATIVE_Q8_0R || type == GLM53F_NATIVE_Q8_0R16;
}

/* A prepared activation carries both llama.cpp activation contracts: Q8_K
 * style 256-value blocks for K/IQ weights and Q8_0 32-value blocks for Q8_0
 * weights.  Only the formats requested by the caller are populated. */
typedef struct {
    int columns, has_q8k, has_q80;
    glm5_iq_q8_block *q8k;
    block_q8_0 *q80;
    int8_t *xq;      /* Q8_0 values, contiguous (repacked-weight path). */
    float *xpat;     /* per 64 values: 8 lanes of d[2k], 8 lanes of d[2k+1]. */
    float *xd;       /* per 32 values: Q8_0 scale (16-row panel path). */
} native_act;

static size_t align256(size_t n) { return (n + 255) & ~(size_t)255; }

size_t glm53f_native_act_bytes(int columns) {
    if (columns < 32 || columns % 32) return 0;
    return align256(sizeof(native_act)) +
           align256((size_t)(columns / 256 + 1) * sizeof(glm5_iq_q8_block)) +
           align256((size_t)(columns / 32) * sizeof(block_q8_0)) +
           align256((size_t)columns) +
           align256((size_t)(columns / 64 + 1) * 16 * sizeof(float)) +
           align256((size_t)(columns / 32) * sizeof(float));
}

int glm53f_native_act_prepare(void *storage, const float *input, int columns,
                              int need_q8k, int need_q80) {
    if (!storage || !input || columns < 32 || columns % 32 ||
        (need_q8k && columns % 256)) return -1;
    native_act *a = storage;
    unsigned char *base = storage;
    a->columns = columns;
    a->q8k = (glm5_iq_q8_block *)(base + align256(sizeof(native_act)));
    a->q80 = (block_q8_0 *)((unsigned char *)a->q8k +
        align256((size_t)(columns / 256 + 1) * sizeof(glm5_iq_q8_block)));
    a->xq = (int8_t *)((unsigned char *)a->q80 +
        align256((size_t)(columns / 32) * sizeof(block_q8_0)));
    a->xpat = (float *)((unsigned char *)a->xq + align256((size_t)columns));
    a->xd = (float *)((unsigned char *)a->xpat +
        align256((size_t)(columns / 64 + 1) * 16 * sizeof(float)));
    a->has_q8k = a->has_q80 = 0;
    if (need_q8k) {
        pthread_once(&glm5_iq_lut_once, glm5_iq_init_luts);
        glm5_iq_quant_q8(a->q8k, input, columns);
        a->has_q8k = 1;
    }
    if (need_q80) {
        quantize_q8_0_activation(a->q80, input, columns);
        for (int b = 0; b < columns / 32; ++b) {
            memcpy(a->xq + 32 * b, a->q80[b].qs, 32);
            const float d = ggml_fp16_to_fp32(a->q80[b].d);
            a->xd[b] = d;
            for (int j = 0; j < 8; ++j)
                a->xpat[(b >> 1) * 16 + (b & 1) * 8 + j] = d;
        }
        a->has_q80 = 1;
    }
    return 0;
}

/* Orphaned (team) version of glm53f_native_act_prepare: every thread of the enclosing parallel region calls it, the
 * blocks are quantized in parallel and the activation is ready (barrier included) when it returns.  Same block-wise
 * arithmetic as the serial version, so results are bit-identical. */
int glm53f_native_act_prepare_team(void *storage, const float *input, int columns, int need_q8k, int need_q80) {
    int bad = (!storage || !input || columns < 32 || columns % 32 || (need_q8k && columns % 256));
    if (bad) return -1;
    native_act *a = storage;
#pragma omp single
    {
        unsigned char *base = storage;
        a->columns = columns;
        a->q8k = (glm5_iq_q8_block *)(base + align256(sizeof(native_act)));
        a->q80 = (block_q8_0 *)((unsigned char *)a->q8k + align256((size_t)(columns / 256 + 1) * sizeof(glm5_iq_q8_block)));
        a->xq = (int8_t *)((unsigned char *)a->q80 + align256((size_t)(columns / 32) * sizeof(block_q8_0)));
        a->xpat = (float *)((unsigned char *)a->xq + align256((size_t)columns));
        a->xd = (float *)((unsigned char *)a->xpat + align256((size_t)(columns / 64 + 1) * 16 * sizeof(float)));
        a->has_q8k = need_q8k != 0;
        a->has_q80 = need_q80 != 0;
        if (need_q8k) pthread_once(&glm5_iq_lut_once, glm5_iq_init_luts);
    }
    if (need_q8k) {
#pragma omp for schedule(static) nowait
        for (int blk = 0; blk < columns / 256; ++blk) glm5_iq_quant_q8(a->q8k + blk, input + 256 * blk, 256);
    }
    if (need_q80) {
#pragma omp for schedule(static)
        for (int blk = 0; blk < columns / 32; ++blk) {
            quantize_q8_0_activation(a->q80 + blk, input + 32 * blk, 32);
            memcpy(a->xq + 32 * blk, a->q80[blk].qs, 32);
            const float d = ggml_fp16_to_fp32(a->q80[blk].d);
            a->xd[blk] = d;
            for (int j = 0; j < 8; ++j) a->xpat[(blk >> 1) * 16 + (blk & 1) * 8 + j] = d;
        }
    } else {
#pragma omp barrier
    }
    return 0;
}

/* Repacked Q8_0 row: int8 q[columns] followed by float d[columns / 32].
 * The fp16 -> fp32 scale conversion is exact, so the arithmetic is the same
 * as the GGUF block layout; only the memory layout is SVE friendly. */
size_t glm53f_native_row_size(int type, int columns) {
    if (type == GLM53F_NATIVE_Q8_0R16)
        return columns > 0 && columns % 128 == 0 ?
               gk_panel_bytes_q8_0r16(columns) / 16 : 0;
    if (type == GLM53F_NATIVE_Q8_0R)
        return columns > 0 && columns % 64 == 0 ?
               (size_t)columns + (size_t)(columns / 32) * sizeof(float) : 0;
    return glm53f_iq_row_size(type, columns);
}

static int native_repack_impl(int type, const uint8_t *source, int rows,
                              int columns, uint8_t **output, int *output_type,
                              int allow_panel) {
    if (!source || !output || !output_type || rows < 1) return -1;
    if (type != GLM53F_GGML_Q8_0 || columns % 64 ||
        getenv("GLM53F_NATIVE_NO_REPACK")) {
        *output = NULL;
        *output_type = type;
        return 0;
    }
    const char *panel = getenv("GLM53F_NATIVE_Q8_PANEL");
    if (allow_panel && panel && strcmp(panel, "1") == 0 && columns % 128 == 0 &&
        rows % 16 == 0) {
        const size_t pb = gk_panel_bytes_q8_0r16(columns);
        const size_t sb = (size_t)(columns / 32) * sizeof(block_q8_0);
        uint8_t *p = NULL;
        if (posix_memalign((void **)&p, 256, (size_t)(rows / 16) * pb)) return -1;
#pragma omp parallel for schedule(static)
        for (int panel_index = 0; panel_index < rows / 16; ++panel_index) {
            uint8_t *dst = p + (size_t)panel_index * pb;
            const uint8_t *src = source + (size_t)panel_index * 16 * sb;
            for (int b = 0; b < columns / 32; ++b) {
                uint8_t *blk = dst + (size_t)b * 576;
                float *dw = (float *)(blk + 512);
                for (int r = 0; r < 16; ++r) {
                    const block_q8_0 *w = (const block_q8_0 *)(src + (size_t)r * sb);
                    dw[r] = ggml_fp16_to_fp32(w[b].d);
                    for (int j = 0; j < 8; ++j)
                        memcpy(blk + j * 64 + r * 4, w[b].qs + j * 4, 4);
                }
            }
        }
        *output = p;
        *output_type = GLM53F_NATIVE_Q8_0R16;
        return 0;
    }
    const size_t rb = glm53f_native_row_size(GLM53F_NATIVE_Q8_0R, columns);
    const size_t sb = (size_t)(columns / 32) * sizeof(block_q8_0);
    uint8_t *p = NULL;
    if (posix_memalign((void **)&p, 256, (size_t)rows * rb)) return -1;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const block_q8_0 *src = (const block_q8_0 *)(source + (size_t)r * sb);
        int8_t *q = (int8_t *)(p + (size_t)r * rb);
        float *d = (float *)(p + (size_t)r * rb + columns);
        for (int b = 0; b < columns / 32; ++b) {
            memcpy(q + 32 * b, src[b].qs, 32);
            d[b] = ggml_fp16_to_fp32(src[b].d);
        }
    }
    *output = p;
    *output_type = GLM53F_NATIVE_Q8_0R;
    return 0;
}

int glm53f_native_repack(int type, const uint8_t *source, int rows,
                         int columns, uint8_t **output, int *output_type) {
    return native_repack_impl(type, source, rows, columns, output, output_type, 1);
}

int glm53f_native_repack_rowwise(int type, const uint8_t *source, int rows,
                                 int columns, uint8_t **output, int *output_type) {
    return native_repack_impl(type, source, rows, columns, output, output_type, 0);
}

#ifndef GLM53F_Q8R_PF
#define GLM53F_Q8R_PF 16384 /* synthetic sweep: 16 KiB ahead with pldl2keep is ~20% faster than none on 7-12 MB matvecs */
#endif
#ifndef GLM53F_Q8R_PF_LVL
#define GLM53F_Q8R_PF_LVL 2
#endif
/* Up to four repacked rows share each activation vector.  Per lane this is
 * acc += float(sdot) * (dw * dx), exactly q8_0_q8_0_row's operation order. */
static inline void q8_0r_rows(float *out, const uint8_t *w, size_t rb,
                              int nrows, const native_act *a) {
    const int columns = a->columns, pairs = columns / 64;
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
    const svbool_t lo8 = svptrue_pat_b32(SV_VL8);
    const int8_t *q[4];
    const float *d[4];
    for (int r = 0; r < 4; ++r) {
        const int rr = r < nrows ? r : 0;
        q[r] = (const int8_t *)(w + (size_t)rr * rb);
        d[r] = (const float *)(w + (size_t)rr * rb + columns);
    }
    svfloat32_t acc0 = svdup_f32(0.0f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    for (int k = 0; k < pairs; ++k) {
#if GLM53F_Q8R_PF
        if ((k & 3) == 0) {   /* one 256 B line per row every four 64-byte steps, GLM53F_Q8R_PF bytes ahead */
            __builtin_prefetch(q[0] + 64 * k + GLM53F_Q8R_PF, 0, GLM53F_Q8R_PF_LVL);
            if (nrows > 1) __builtin_prefetch(q[1] + 64 * k + GLM53F_Q8R_PF, 0, GLM53F_Q8R_PF_LVL);
            if (nrows > 2) __builtin_prefetch(q[2] + 64 * k + GLM53F_Q8R_PF, 0, GLM53F_Q8R_PF_LVL);
            if (nrows > 3) __builtin_prefetch(q[3] + 64 * k + GLM53F_Q8R_PF, 0, GLM53F_Q8R_PF_LVL);
        }
#endif
        const svint8_t xv = svld1_s8(p8, a->xq + 64 * k);
        const svfloat32_t xs = svld1_f32(p32, a->xpat + 16 * k);
#define GLM53F_Q8R_ROW(R, ACC) do { \
        const svint8_t wv = svld1_s8(p8, q[R] + 64 * k); \
        const svfloat32_t ws = svsel_f32(lo8, svdup_f32(d[R][2 * k]), \
                                         svdup_f32(d[R][2 * k + 1])); \
        ACC = svmla_f32_m(p32, ACC, \
            svcvt_f32_s32_x(p32, svdot_s32(svdup_s32(0), wv, xv)), \
            svmul_f32_x(p32, ws, xs)); \
    } while (0)
        GLM53F_Q8R_ROW(0, acc0);
        if (nrows > 1) GLM53F_Q8R_ROW(1, acc1);
        if (nrows > 2) GLM53F_Q8R_ROW(2, acc2);
        if (nrows > 3) GLM53F_Q8R_ROW(3, acc3);
#undef GLM53F_Q8R_ROW
    }
    out[0] = svaddv_f32(p32, acc0);
    if (nrows > 1) out[1] = svaddv_f32(p32, acc1);
    if (nrows > 2) out[2] = svaddv_f32(p32, acc2);
    if (nrows > 3) out[3] = svaddv_f32(p32, acc3);
}

static inline float native_row(int type, const uint8_t *row,
                               const native_act *a) {
    if (type == GLM53F_NATIVE_Q8_0R) {
        float y;
        q8_0r_rows(&y, row, 0, 1, a);
        return y;
    }
    if (type == GLM53F_GGML_Q8_0)
        return q8_0_q8_0_row((const block_q8_0 *)row, a->q80, a->columns / 32);
    return iq_row(type, row, a->q8k, a->columns / 256);
}

static inline void q8_0r16_rows(float *out, const uint8_t *weight,
                               int rows, int columns, int r,
                               const native_act *a) {
    const gk_act ga = {.columns = columns, .xq = a->xq, .xd = a->xd};
    const gk_mv gm = {.w = weight,
                      .row_bytes = glm53f_native_row_size(GLM53F_NATIVE_Q8_0R16, columns),
                      .rows = rows, .columns = columns, .a = &ga, .y = out};
    gk_q8_0r16_v3pf16k(&gm, r, r + 16);
}

/* Four rows x four tokens. Reuse every weight load across four independent
 * accumulators without changing the decode kernel's per-lane FMA order. */
static inline void q8_0r_tile4(float *out, int stride, const uint8_t *w,
                              size_t rb, const native_act *a[4]) {
    const int columns = a[0]->columns;
    const svbool_t p8 = svptrue_b8(), pg = svptrue_b32();
    const svbool_t lo8 = svptrue_pat_b32(SV_VL8);
#define Q8R_ACC(R) svfloat32_t a##R##0 = svdup_f32(0), a##R##1 = a##R##0, \
                              a##R##2 = a##R##0, a##R##3 = a##R##0
    Q8R_ACC(0); Q8R_ACC(1); Q8R_ACC(2); Q8R_ACC(3);
#undef Q8R_ACC
    for (int k = 0; k < columns / 64; ++k) {
#define Q8R_X(T) const svint8_t x##T = svld1_s8(p8, a[T]->xq + 64 * k); \
                 const svfloat32_t s##T = svld1_f32(pg, a[T]->xpat + 16 * k)
        Q8R_X(0); Q8R_X(1); Q8R_X(2); Q8R_X(3);
#undef Q8R_X
#define Q8R_FMA(R,T) a##R##T = svmla_f32_m(pg, a##R##T, \
    svcvt_f32_s32_x(pg, svdot_s32(svdup_s32(0), wv, x##T)), \
    svmul_f32_x(pg, ws, s##T))
#define Q8R_ROW(R) do { \
    const float *d = (const float *)(w + (size_t)(R) * rb + columns); \
    const svint8_t wv = svld1_s8(p8, (const int8_t *)(w + (size_t)(R) * rb) + 64 * k); \
    const svfloat32_t ws = svsel_f32(lo8, svdup_f32(d[2*k]), svdup_f32(d[2*k+1])); \
    Q8R_FMA(R,0); Q8R_FMA(R,1); Q8R_FMA(R,2); Q8R_FMA(R,3); \
} while (0)
        Q8R_ROW(0); Q8R_ROW(1); Q8R_ROW(2); Q8R_ROW(3);
#undef Q8R_ROW
#undef Q8R_FMA
    }
#define Q8R_STORE(R) do { \
    out[R] = svaddv_f32(pg, a##R##0); \
    out[(size_t)stride + R] = svaddv_f32(pg, a##R##1); \
    out[(size_t)2 * stride + R] = svaddv_f32(pg, a##R##2); \
    out[(size_t)3 * stride + R] = svaddv_f32(pg, a##R##3); \
} while (0)
    Q8R_STORE(0); Q8R_STORE(1); Q8R_STORE(2); Q8R_STORE(3);
#undef Q8R_STORE
}

static int native_is_q80(int type) {
    return type == GLM53F_GGML_Q8_0 || type == GLM53F_NATIVE_Q8_0R ||
           type == GLM53F_NATIVE_Q8_0R16;
}

static int native_check(int type, int columns, const native_act *a) {
    if (!glm53f_native_type_supported(type) || !a || a->columns != columns)
        return -1;
    return native_is_q80(type) ? (a->has_q80 ? 0 : -1)
                               : (a->has_q8k ? 0 : -1);
}

/* Orphaned work-sharing: call from every thread of an existing team.  The
 * trailing implicit barrier publishes all outputs to the team. */
int glm53f_native_matvec_team(const glm53f_native_matrix *m, int count,
                              const void *activation) {
    const native_act *a = activation;
    int total = 0, bad = 0, group[8], start[9];
    size_t rb[8];
    if (!m || count < 1 || count > 8) return -1;
    for (int i = 0; i < count; ++i) {
        rb[i] = glm53f_native_row_size(m[i].type, m[i].columns);
        bad |= !m[i].output || !m[i].weight || m[i].rows < 1 || !rb[i] ||
               native_check(m[i].type, m[i].columns, a);
        /* Work items are four-row groups for repacked Q8_0, rows otherwise. */
        group[i] = m[i].type == GLM53F_NATIVE_Q8_0R16 ? 16 :
                   m[i].type == GLM53F_NATIVE_Q8_0R ? 4 : 1;
        bad |= group[i] == 16 && m[i].rows % 16 != 0;
        start[i] = total;
        total += (m[i].rows + group[i] - 1) / group[i];
    }
    start[count] = total;
    if (bad) return -1;
#pragma omp for schedule(static)
    for (int q = 0; q < total; ++q) {
        int i = 0;
        while (q >= start[i + 1]) ++i;
        const int r = (q - start[i]) * group[i];
        if (group[i] == 16) {
            q8_0r16_rows(m[i].output, m[i].weight, m[i].rows,
                          m[i].columns, r, a);
        } else if (group[i] == 4) {
            const int n = m[i].rows - r < 4 ? m[i].rows - r : 4;
            q8_0r_rows(m[i].output + r, m[i].weight + (size_t)r * rb[i],
                       rb[i], n, a);
        } else {
            m[i].output[r] = native_row(m[i].type,
                m[i].weight + (size_t)r * rb[i], a);
        }
    }
    return 0;
}

int glm53f_native_matvec_n(const glm53f_native_matrix *m, int count,
                           const float *input) {
    int need_q8k = 0, need_q80 = 0, columns, rc = 0;
    if (!m || count < 1 || count > 8 || !input) return -1;
    columns = m[0].columns;
    for (int i = 0; i < count; ++i) {
        if (m[i].columns != columns ||
            !glm53f_native_type_supported(m[i].type)) return -1;
        if (native_is_q80(m[i].type)) need_q80 = 1; else need_q8k = 1;
    }
    size_t bytes = glm53f_native_act_bytes(columns);
    void *act = NULL;
    if (!bytes || posix_memalign(&act, 256, bytes)) return -1;
    if (glm53f_native_act_prepare(act, input, columns, need_q8k, need_q80)) {
        free(act);
        return -1;
    }
#pragma omp parallel reduction(|:rc)
    rc |= glm53f_native_matvec_team(m, count, act) != 0;
    free(act);
    return rc ? -1 : 0;
}

int glm53f_native_matvec_batch_team(const glm53f_native_matrix *m, int count,
        const void *activation, size_t activation_stride, int tokens) {
    int total = 0, group[8], start[9];
    size_t rb[8];
    if (!m || count < 1 || count > 8 || !activation || tokens < 1 ||
        activation_stride < glm53f_native_act_bytes(m[0].columns)) return -1;
    for (int i = 0; i < count; ++i) {
        rb[i] = glm53f_native_row_size(m[i].type, m[i].columns);
        if (!m[i].output || !m[i].weight || m[i].rows < 1 || !rb[i]) return -1;
        for (int t = 0; t < tokens; ++t)
            if (native_check(m[i].type, m[i].columns,
                    (const native_act *)((const uint8_t *)activation +
                                          (size_t)t * activation_stride))) return -1;
        group[i] = m[i].type == GLM53F_NATIVE_Q8_0R16 ? 16 :
                   m[i].type == GLM53F_NATIVE_Q8_0R ? 4 : 1;
        if (group[i] == 16 && m[i].rows % 16 != 0) return -1;
        start[i] = total;
        total += (m[i].rows + group[i] - 1) / group[i];
    }
    start[count] = total;
#pragma omp for collapse(2) schedule(static)
    for (int q = 0; q < total; ++q)
        for (int t = 0; t < tokens; t += 4) {
            int i = 0;
            while (q >= start[i + 1]) ++i;
            const int r = (q - start[i]) * group[i];
            const int nr = m[i].rows - r < group[i] ? m[i].rows - r : group[i];
            const int nt = tokens - t < 4 ? tokens - t : 4;
            const uint8_t *row = m[i].weight + (size_t)r * rb[i];
            float *out = m[i].output + (size_t)t * m[i].rows + r;
            const native_act *a[4];
            for (int j = 0; j < nt; ++j)
                a[j] = (const native_act *)((const uint8_t *)activation +
                                            (size_t)(t + j) * activation_stride);
            if (group[i] == 16) {
                for (int j = 0; j < nt; ++j)
                    q8_0r16_rows(m[i].output + (size_t)(t + j) * m[i].rows,
                        m[i].weight, m[i].rows, m[i].columns, r, a[j]);
            } else if (group[i] == 4 && nr == 4 && nt == 4)
                q8_0r_tile4(out, m[i].rows, row, rb[i], a);
            else for (int j = 0; j < nt; ++j) {
                if (group[i] == 4)
                    q8_0r_rows(out + (size_t)j * m[i].rows, row, rb[i], nr, a[j]);
                else out[(size_t)j * m[i].rows] = native_row(m[i].type, row, a[j]);
            }
        }
    return 0;
}

int glm53f_native_matvec_batch(const glm53f_native_matrix *m, int count,
        const float *input, int tokens) {
    if (!m || count < 1 || count > 8 || !input || tokens < 1) return -1;
    int columns = m[0].columns, need_q8k = 0, need_q80 = 0, bad = 0;
    size_t stride = glm53f_native_act_bytes(columns);
    if (!stride || (size_t)tokens > SIZE_MAX / stride) return -1;
    for (int i = 0; i < count; ++i) {
        if (m[i].columns != columns || !glm53f_native_row_size(m[i].type, columns) ||
            !glm53f_native_type_supported(m[i].type)) return -1;
        if (native_is_q80(m[i].type)) need_q80 = 1; else need_q8k = 1;
    }
    if (need_q8k && columns % 256) return -1;
    uint8_t *act = NULL;
    if (posix_memalign((void **)&act, 256, stride * (size_t)tokens)) return -1;
#pragma omp parallel reduction(|:bad)
    {
#pragma omp for schedule(static)
        for (int t = 0; t < tokens; ++t)
            bad |= glm53f_native_act_prepare(act + (size_t)t * stride,
                input + (size_t)t * columns, columns, need_q8k, need_q80) != 0;
        bad |= glm53f_native_matvec_batch_team(m, count, act, stride, tokens) != 0;
    }
    free(act);
    return bad ? -1 : 0;
}

int glm53f_iq_matvec_2(
        float *output0, const uint8_t *weight0, int weight0_type,
        float *output1, const uint8_t *weight1, int weight1_type,
        int rows, int columns, const float *input) {
    if (native_is_q80(weight0_type) ||
        (output1 && native_is_q80(weight1_type))) {
        if (!output0 || !weight0 || (!!output1 != !!weight1)) return -1;
        glm53f_native_matrix m[2] = {
            {output0, weight0, weight0_type, rows, columns},
            {output1, weight1, weight1_type, rows, columns}};
        return glm53f_native_matvec_n(m, output1 ? 2 : 1, input);
    }
    if (!output0 || !weight0 || !input || rows < 1 || columns < 256 ||
        columns % 256 || !glm53f_iq_type_supported(weight0_type) ||
        (!!output1 != !!weight1) ||
        (output1 && !glm53f_iq_type_supported(weight1_type))) return -1;
    size_t row0 = dequant_row_size((uint32_t)weight0_type, columns);
    size_t row1 = output1 ? dequant_row_size((uint32_t)weight1_type, columns) : 0;
    glm5_iq_q8_block *input_q = malloc(
        (size_t)(columns / 256) * sizeof(*input_q));
    if (!row0 || (output1 && !row1) || !input_q) {
        free(input_q);
        return -1;
    }
    pthread_once(&glm5_iq_lut_once, glm5_iq_init_luts);
    glm5_iq_quant_q8(input_q, input, columns);
#pragma omp parallel for schedule(static)
    for (int row = 0; row < rows; ++row) {
        output0[row] = iq_row(weight0_type, weight0 + (size_t)row * row0,
                              input_q, columns / 256);
        if (output1)
            output1[row] = iq_row(weight1_type,
                weight1 + (size_t)row * row1, input_q, columns / 256);
    }
    free(input_q);
    return 0;
}

int glm53f_iq_matvec(
        float *output, const uint8_t *weight, int weight_type,
        int rows, int columns, const float *input) {
    if (weight_type == GLM53F_NATIVE_Q8_0R) {
        glm53f_native_matrix m = {output, weight, weight_type, rows, columns};
        return glm53f_native_matvec_n(&m, 1, input);
    }
    if (weight_type == GLM53F_GGML_Q8_0) {
        if (!output || !weight || !input || rows < 1 || columns < 32 ||
            columns % 32) return -1;
        const size_t row = (size_t)(columns / 32) * sizeof(block_q8_0);
        block_q8_0 *input_q = malloc(row);
        if (!input_q) return -1;
        quantize_q8_0_activation(input_q, input, columns);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r)
            output[r] = q8_0_q8_0_row(
                (const block_q8_0 *)(weight + (size_t)r * row),
                input_q, columns / 32);
        free(input_q);
        return 0;
    }
    return glm53f_iq_matvec_2(output, weight, weight_type,
                              NULL, NULL, 0, rows, columns, input);
}

double glm53f_iq_stage_us[6]; /* GLM53F_IQ_TIMING: quant+prepare, gate/up, swiglu, act quant, down+tail, calls */
static inline double iq_stamp(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }
/* ---- CMG-affine placement of routed-expert parts (GLM53F_IQ_AFFINE=1) ----------------------------------------
 * The decode kernels split every part's rows over the four CMGs in proportion to their thread counts, so the rows a
 * CMG's threads read can live in that CMG's HBM stack.  glm53f_iq_place_part() binds the row ranges (call it before the
 * pages are first touched); iq_expert_fast() uses the same split. */
static void iq_cmg_bounds(int rows, int nt, int *bound /* [5] */) {
    const int per = 12; /* cores per CMG; OMP places=cores, close: thread i sits on CMG i/12 */
    int cum = 0;
    bound[0] = 0;
    for (int c = 0; c < 4; ++c) {
        int n = nt - c * per; n = n < 0 ? 0 : (n > per ? per : n);
        cum += n;
        bound[c + 1] = c == 3 ? rows : (int)((long long)rows * cum / nt) & ~1;
    }
}

int glm53f_iq_affine_enabled(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("GLM53F_IQ_AFFINE"); v = e && *e ? atoi(e) : 0; }
    return v;
}

static void iq_bind_rows(const uint8_t *base, size_t row_bytes, const int *bound) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    for (int c = 0; c < 4; ++c) {
        uintptr_t lo = ((uintptr_t)base + (size_t)bound[c] * row_bytes + page - 1) & ~(page - 1);
        uintptr_t hi = ((uintptr_t)base + (size_t)bound[c + 1] * row_bytes) & ~(page - 1);
        if (c == 0) lo = (uintptr_t)base & ~(page - 1);
        if (hi <= lo) continue;
        unsigned long mask = 1UL << (4 + c); /* compute CMGs are NUMA nodes 4..7 */
        syscall(SYS_mbind, (void *)lo, hi - lo, 2 /* MPOL_BIND */, &mask, 64UL, 0);
    }
}

void glm53f_iq_place_part(const uint8_t *gate_up, size_t gate_row_bytes, int gate_rows, const uint8_t *down,
                          size_t down_row_bytes, int down_rows) {
    int bound[5];
    const int nt = omp_get_max_threads();
    iq_cmg_bounds(gate_rows, nt, bound);
    iq_bind_rows(gate_up, gate_row_bytes, bound);
    iq_cmg_bounds(down_rows, nt, bound);
    iq_bind_rows(down, down_row_bytes, bound);
}

/* Decode routed-expert step on the native Q4_K/Q5_K rows: input quantisation, gate/up, SwiGLU+quantisation and down
 * projection in one parallel region (three barriers, no serial sections). */
static int iq_expert_fast(float *output, const glm53f_iq_part *parts, const float *weights, int count,
                          const float *input, const glm53f_iq_shared *sh) {
    enum { HIDDEN = 4096, GU_STRIDE = 1024 };
    size_t gate_rb[9], down_rb[9];
    for (int k = 0; k < count; ++k) {
        gate_rb[k] = dequant_row_size((uint32_t)parts[k].gate_type, HIDDEN);
        down_rb[k] = dequant_row_size((uint32_t)parts[k].down_type, parts[k].inter);
    }
    const int timing = getenv("GLM53F_IQ_TIMING") != NULL;
    int affine = glm53f_iq_affine_enabled() && omp_get_max_threads() >= 37 && omp_get_max_threads() <= 48 && count >= 1;
    for (int k = 0; k < count && affine; ++k) affine = parts[k].inter == 256;
    double ts0 = timing ? iq_stamp() : 0, ts1 = 0, ts2 = 0, ts3 = 0;
    iqf_act in_a[HIDDEN / 256] __attribute__((aligned(64)));
    iqf_act act_a[9][2] __attribute__((aligned(64)));
    float gate_up[9 * GU_STRIDE] __attribute__((aligned(256)));
    int total_gate_rows = 0, bad = 0;
    for (int k = 0; k < count; ++k) total_gate_rows += 2 * parts[k].inter;
#pragma omp parallel reduction(|:bad)
    {
#pragma omp for schedule(static)
        for (int b = 0; b < HIDDEN / 256; ++b) {
            iqf_src_block tmp;
            glm5_iq_quant_q8((glm5_iq_q8_block *)&tmp, input + 256 * b, 256);
            iqf_prepare(&in_a[b], &tmp, 1);
        }
        if (sh) bad |= glm53f_native_act_prepare_team(sh->act_x, input, HIDDEN, sh->x_q8k, sh->x_q80) != 0;
#pragma omp master
        if (timing) ts1 = iq_stamp();
        if (affine) {
            /* threads of CMG c stream rows [bound[c], bound[c+1]) of every part, split evenly among the CMG's threads */
            const int nt = omp_get_num_threads(), tid = omp_get_thread_num(), cmg = tid / 12 > 3 ? 3 : tid / 12;
            const int nc = cmg == 3 ? nt - 36 : 12, j = tid - cmg * 12;
            int bound[5], rows_c = 0;
            iq_cmg_bounds(parts[0].inter * 2, nt, bound); /* all parts share the same shape */
            rows_c = bound[cmg + 1] - bound[cmg];
            const long long total = (long long)rows_c * count;
            const long long t0 = total * j / nc, t1 = total * (j + 1) / nc;
            for (long long q = t0; q < t1;) {
                const int k = (int)(q / rows_c), r = bound[cmg] + (int)(q % rows_c);
                long long n = (k + 1) * (long long)rows_c - q; if (q + n > t1) n = t1 - q;
                iqf_rows(gate_up + (size_t)k * GU_STRIDE + r, parts[k].gate_up + (size_t)r * gate_rb[k], gate_rb[k], (int)n,
                         in_a, HIDDEN / 256, parts[k].gate_type == GLM53F_GGML_Q5_K);
                q += n;
            }
        } else {
            const int nt = omp_get_num_threads(), tid = omp_get_thread_num();
            const int q0 = (int)((long long)total_gate_rows * tid / nt), q1 = (int)((long long)total_gate_rows * (tid + 1) / nt);
            int base = 0, k = 0;
            for (int q = q0; q < q1;) {
                while (q >= base + 2 * parts[k].inter) base += 2 * parts[k++].inter;
                const int r = q - base, n = (base + 2 * parts[k].inter < q1 ? base + 2 * parts[k].inter : q1) - q;
                iqf_rows(gate_up + (size_t)k * GU_STRIDE + r, parts[k].gate_up + (size_t)r * gate_rb[k], gate_rb[k], n,
                         in_a, HIDDEN / 256, parts[k].gate_type == GLM53F_GGML_Q5_K);
                q += n;
            }
        }
        if (sh) {
            bad |= glm53f_native_matvec_team(sh->gu, 2, sh->act_x) != 0; /* ends with a barrier */
        } else {
#pragma omp barrier
        }
#pragma omp master
        if (timing) ts2 = iq_stamp();
#pragma omp for schedule(static) nowait
        for (int task = 0; task < count * 2; ++task) {
            const int k = task >> 1, blk = task & 1;
            if (blk < parts[k].inter / 256)
                iqf_swiglu_block(&act_a[k][blk], gate_up + (size_t)k * GU_STRIDE + blk * 256,
                                 gate_up + (size_t)k * GU_STRIDE + parts[k].inter + blk * 256);
        }
        if (sh) {
#pragma omp for schedule(static)
            for (int i = 0; i < sh->rows; ++i) { /* same clamps / formula as nsh_accumulate */
                float g = sh->gu[0].output[i], u = sh->gu[1].output[i];
                if (g > 10) g = 10;
                if (g < -100) g = -100;
                if (u > 10) u = 10;
                if (u < -10) u = -10;
                sh->act[i] = (g / (1 + expf(-g))) * u;
            }
            bad |= glm53f_native_act_prepare_team(sh->act_h, sh->act, sh->rows, sh->h_q8k, sh->h_q80) != 0;
            bad |= glm53f_native_matvec_team(&sh->dn, 1, sh->act_h) != 0;
        } else {
#pragma omp barrier
        }
#pragma omp master
        if (timing) ts3 = iq_stamp();
        {
            const int nt = omp_get_num_threads(), tid = omp_get_thread_num();
            int r0, r1;
            if (affine) {
                const int cmg = tid / 12 > 3 ? 3 : tid / 12, j = tid - cmg * 12;
                const int nc = cmg == 3 ? nt - 36 : 12;
                int bound[5];
                iq_cmg_bounds(HIDDEN, nt, bound);
                const int rc = bound[cmg + 1] - bound[cmg];
                r0 = bound[cmg] + (int)((long long)rc * j / nc); r1 = bound[cmg] + (int)((long long)rc * (j + 1) / nc);
            } else {
                r0 = (int)((long long)HIDDEN * tid / nt); r1 = (int)((long long)HIDDEN * (tid + 1) / nt);
            }
            float tmp[9][64];
            for (int r = r0; r < r1; r += 64) {
                const int n = r1 - r < 64 ? r1 - r : 64;
                for (int k = 0; k < count; ++k)
                    iqf_rows(tmp[k], parts[k].down + (size_t)r * down_rb[k], down_rb[k], n, act_a[k],
                             parts[k].inter / 256, parts[k].down_type == GLM53F_GGML_Q5_K);
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < count; ++k) sum += weights[k] * tmp[k][i];
                    output[r + i] = sh ? sum + sh->dn.output[r + i] : sum;
                }
            }
        }
    }
    if (bad) return -1;
    if (timing) {
        const double te = iq_stamp();
        glm53f_iq_stage_us[0] += ts1 - ts0; glm53f_iq_stage_us[1] += ts2 - ts1; glm53f_iq_stage_us[2] += ts3 - ts2;
        glm53f_iq_stage_us[4] += te - ts3; glm53f_iq_stage_us[5] += 1;
    }
    return 0;
}

static int iq_expert_weighted_impl(
        float *output, const glm53f_iq_part *parts, const float *weights,
        int count, const float *input, float *gate_up, float *activation, int fast_override) {
    enum { HIDDEN = 4096, GU_STRIDE = 1024, ACT_STRIDE = 512 };
    glm5_iq_q8_block input_q[HIDDEN / 256];
    glm5_iq_q8_block act_q[9][ACT_STRIDE / 256];
    size_t gate_rb[9], down_rb[9];
    int gate_blocks = HIDDEN / 256;
    if (!output || !parts || !weights || !input || !gate_up || !activation ||
        count < 1 || count > 9) return -1;
    pthread_once(&glm5_iq_lut_once, glm5_iq_init_luts);
    const int timing = getenv("GLM53F_IQ_TIMING") != NULL;
    double ts0 = timing ? iq_stamp() : 0, ts1 = 0, ts2 = 0, ts3 = 0, ts4 = 0;
    glm5_iq_quant_q8(input_q, input, HIDDEN);
    for (int k = 0; k < count; ++k) {
        if (!glm53f_iq_type_supported(parts[k].gate_type) ||
            !glm53f_iq_type_supported(parts[k].down_type) ||
            parts[k].inter < 256 || parts[k].inter > ACT_STRIDE ||
            parts[k].inter % 256) return -1;
        gate_rb[k] = dequant_row_size((uint32_t)parts[k].gate_type, HIDDEN);
        down_rb[k] = dequant_row_size((uint32_t)parts[k].down_type, parts[k].inter);
    }
    /* Full-width Q4_K/Q5_K row kernels (glm53f_iq_fast.h); GLM53F_IQ_FAST=0 keeps the reference loops. */
    static int fast_env = -1;
    if (fast_env < 0) { const char *e = getenv("GLM53F_IQ_FAST"); fast_env = !e || !*e ? 1 : atoi(e); iqf_init(); }
    int fast = fast_override >= 0 ? fast_override : fast_env == 1;
    for (int k = 0; k < count; ++k)
        fast &= (parts[k].gate_type == GLM53F_GGML_Q4_K || parts[k].gate_type == GLM53F_GGML_Q5_K) &&
                (parts[k].down_type == GLM53F_GGML_Q4_K || parts[k].down_type == GLM53F_GGML_Q5_K) &&
                parts[k].inter <= 512;
    iqf_act in_a[HIDDEN / 256] __attribute__((aligned(64)));
    iqf_act act_a[9][2] __attribute__((aligned(64)));
    if (fast) iqf_prepare(in_a, (const iqf_src_block *)input_q, gate_blocks);
    int total_gate_rows = 0;
    for (int k = 0; k < count; ++k) total_gate_rows += 2 * parts[k].inter;
    if (timing) ts1 = iq_stamp();
#pragma omp parallel
    {
        if (fast) {
            const int nt = omp_get_num_threads(), tid = omp_get_thread_num();
            const int q0 = (int)((long long)total_gate_rows * tid / nt), q1 = (int)((long long)total_gate_rows * (tid + 1) / nt);
            int base = 0, k = 0;
            for (int q = q0; q < q1;) {
                while (q >= base + 2 * parts[k].inter) base += 2 * parts[k++].inter;
                const int r = q - base, n = (base + 2 * parts[k].inter < q1 ? base + 2 * parts[k].inter : q1) - q;
                iqf_rows(gate_up + (size_t)k * GU_STRIDE + r, parts[k].gate_up + (size_t)r * gate_rb[k], gate_rb[k], n,
                         in_a, gate_blocks, parts[k].gate_type == GLM53F_GGML_Q5_K);
                q += n;
            }
#pragma omp barrier
        } else
#pragma omp for schedule(static)
        for (int q = 0; q < total_gate_rows; ++q) {
            int k = 0, r = q;
            while (r >= 2 * parts[k].inter) r -= 2 * parts[k++].inter;
            gate_up[(size_t)k * GU_STRIDE + r] = iq_row(
                parts[k].gate_type, parts[k].gate_up + (size_t)r * gate_rb[k],
                input_q, gate_blocks);
        }
#pragma omp master
        if (timing) ts2 = iq_stamp();
#pragma omp for schedule(static)
        for (int q = 0; q < count * ACT_STRIDE; ++q) {
            int k = q / ACT_STRIDE, i = q - k * ACT_STRIDE;
            if (i < parts[k].inter) {
                float g = gate_up[(size_t)k * GU_STRIDE + i];
                float u = gate_up[(size_t)k * GU_STRIDE + parts[k].inter + i];
                if (g > 10) g = 10; if (g < -100) g = -100;
                if (u > 10) u = 10; if (u < -10) u = -10;
                activation[(size_t)k * ACT_STRIDE + i] =
                    (g / (1.0f + expf(-g))) * u;
            }
        }
#pragma omp master
        if (timing) ts3 = iq_stamp();
#pragma omp barrier
#pragma omp single
        for (int k = 0; k < count; ++k) {
            glm5_iq_quant_q8(act_q[k], activation + (size_t)k * ACT_STRIDE, parts[k].inter);
            if (fast) iqf_prepare(act_a[k], (const iqf_src_block *)act_q[k], parts[k].inter / 256);
        }
#pragma omp master
        if (timing) ts4 = iq_stamp();
        if (fast) {
            const int nt = omp_get_num_threads(), tid = omp_get_thread_num();
            const int r0 = (int)((long long)HIDDEN * tid / nt), r1 = (int)((long long)HIDDEN * (tid + 1) / nt);
            float tmp[9][64];
            for (int r = r0; r < r1; r += 64) {
                const int n = r1 - r < 64 ? r1 - r : 64;
                for (int k = 0; k < count; ++k)
                    iqf_rows(tmp[k], parts[k].down + (size_t)r * down_rb[k], down_rb[k], n, act_a[k],
                             parts[k].inter / 256, parts[k].down_type == GLM53F_GGML_Q5_K);
                for (int i = 0; i < n; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < count; ++k) sum += weights[k] * tmp[k][i];
                    output[r + i] = sum;
                }
            }
        } else
#pragma omp for schedule(static)
        for (int r = 0; r < HIDDEN; ++r) {
            float sum = 0.0f;
            for (int k = 0; k < count; ++k)
                sum += weights[k] * iq_row(
                    parts[k].down_type,
                    parts[k].down + (size_t)r * down_rb[k], act_q[k],
                    parts[k].inter / 256);
            output[r] = sum;
        }
    }
    if (timing) {
        const double te = iq_stamp();
        glm53f_iq_stage_us[0] += ts1 - ts0; glm53f_iq_stage_us[1] += ts2 - ts1; glm53f_iq_stage_us[2] += ts3 - ts2;
        glm53f_iq_stage_us[3] += ts4 - ts3; glm53f_iq_stage_us[4] += te - ts4; glm53f_iq_stage_us[5] += 1;
    }
    return 0;
}

int glm53f_iq_expert_weighted(
        float *output, const glm53f_iq_part *parts, const float *weights,
        int count, const float *input, float *gate_up, float *activation) {
    static int mode = -1;
    if (mode < 0) { const char *e = getenv("GLM53F_IQ_FAST"); mode = !e || !*e ? 1 : atoi(e); iqf_init(); }
    int eligible = mode != 0 && count >= 1 && count <= 9;
    for (int k = 0; k < count && eligible; ++k)
        eligible = (parts[k].gate_type == GLM53F_GGML_Q4_K || parts[k].gate_type == GLM53F_GGML_Q5_K) &&
                   (parts[k].down_type == GLM53F_GGML_Q4_K || parts[k].down_type == GLM53F_GGML_Q5_K) && parts[k].inter <= 512 &&
                   parts[k].inter % 256 == 0;
    if (mode == 1) {
        if (eligible) return iq_expert_fast(output, parts, weights, count, input, NULL);
        return iq_expert_weighted_impl(output, parts, weights, count, input, gate_up, activation, 0);
    }
    if (mode == 0) return iq_expert_weighted_impl(output, parts, weights, count, input, gate_up, activation, 0);
    /* verify: run the reference loops into scratch and compare the fast result against them */
    static float ref[4096], gu2[9 * 1024], act2[9 * 512];
    static double worst; static long calls;
    if (iq_expert_weighted_impl(ref, parts, weights, count, input, gu2, act2, 0)) return -1;
    if (eligible ? iq_expert_fast(output, parts, weights, count, input, NULL) : iq_expert_weighted_impl(output, parts, weights, count, input, gate_up, activation, 0)) return -1;
    double se = 0, sr = 0;
    for (int i = 0; i < 4096; ++i) { double d = output[i] - ref[i]; se += d * d; sr += (double)ref[i] * ref[i]; }
    double rel = sqrt(se / (sr + 1e-30));
    if (rel > worst) worst = rel;
    if (++calls % 2000 == 0) fprintf(stderr, "GLM53F_IQ_FAST_VERIFY calls=%ld worst_rel_l2=%.3e last=%.3e\n", calls, worst, rel);
    return 0;
}

/* Routed experts plus the native Q8_0 shared expert in one parallel region; returns -1 if the parts are not eligible
 * for the fast kernels (the caller then falls back to the separate paths). */
int glm53f_iq_expert_weighted_shared(float *output, const glm53f_iq_part *parts, const float *weights, int count,
                                     const float *input, const glm53f_iq_shared *sh) {
    static int mode = -1;
    if (mode < 0) { const char *e = getenv("GLM53F_IQ_FAST"); mode = !e || !*e ? 1 : atoi(e); iqf_init(); }
    if (mode != 1 || !sh || count < 1 || count > 9) return -1;
    for (int k = 0; k < count; ++k)
        if (!((parts[k].gate_type == GLM53F_GGML_Q4_K || parts[k].gate_type == GLM53F_GGML_Q5_K) &&
              (parts[k].down_type == GLM53F_GGML_Q4_K || parts[k].down_type == GLM53F_GGML_Q5_K) && parts[k].inter <= 512 &&
              parts[k].inter % 256 == 0)) return -1;
    return iq_expert_fast(output, parts, weights, count, input, sh);
}
