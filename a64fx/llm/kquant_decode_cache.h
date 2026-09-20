/* Decode-oriented, lossless weight repacks for compact Q5_K and IQ4_XS.
 *
 * Include this after transformer.h so the GGML blocks, A8 activation blocks,
 * quantizer, FP16 conversion, scale unpacker, and A64FX SVE intrinsics are
 * available. Both layouts store groups of eight rows, then 256-column blocks.
 * Callers must use dimensions divisible by eight rows and 256 columns.
 */
#ifndef A64FX_KQUANT_DECODE_CACHE_H
#define A64FX_KQUANT_DECODE_CACHE_H

#define TF_KQUANT_CACHE_LAYOUT_VERSION 1
#define TF_KQUANT_CACHE_ROWS 8
#define TF_KQUANT_CACHE_COLS 256

typedef struct {
    float d, dmin;
    uint8_t scales[8];
    uint8_t mins[8];
} packed_q5r_header;

_Static_assert(sizeof(packed_q5r_header) == 24, "packed Q5R header size");

static size_t packed_q5r_block_bytes(void) {
    return 8 * sizeof(packed_q5r_header) + 8 * 256;
}

static size_t packed_q5r_bytes(int rows, int cols) {
    if (rows <= 0 || cols <= 0 || rows % TF_KQUANT_CACHE_ROWS ||
        cols % TF_KQUANT_CACHE_COLS) return 0;
    return (size_t)(rows / TF_KQUANT_CACHE_ROWS) *
           (size_t)(cols / TF_KQUANT_CACHE_COLS) * packed_q5r_block_bytes();
}

#ifndef TF_KQUANT_CACHE_LAYOUT_ONLY
static int pack_q5r(uint8_t *dst, const block_q5_K *src, int rows, int cols) {
    if (!dst || !src || !packed_q5r_bytes(rows, cols)) return -1;
    int nb = cols / 256;
    size_t bb = packed_q5r_block_bytes();
#pragma omp parallel for schedule(static)
    for (int rg = 0; rg < rows / 8; rg++) {
        for (int b = 0; b < nb; b++) {
            uint8_t *block = dst + ((size_t)rg * nb + b) * bb;
            packed_q5r_header *headers = (packed_q5r_header *)block;
            int8_t *q = (int8_t *)(block + 8 * sizeof(*headers));
            for (int rr = 0; rr < 8; rr++) {
                const block_q5_K *wb = src + (size_t)(rg * 8 + rr) * nb + b;
                headers[rr].d = ggml_fp16_to_fp32(wb->d);
                headers[rr].dmin = ggml_fp16_to_fp32(wb->dmin);
                for (int i = 0; i < 8; i++)
                    get_scale_min_k4(i, wb->scales,
                                     &headers[rr].scales[i], &headers[rr].mins[i]);
                for (int g = 0; g < 4; g++) {
                    int8_t *qrow = q + (g * 8 + rr) * 64;
                    for (int k = 0; k < 32; k++) {
                        uint8_t v = wb->qs[g * 32 + k];
                        qrow[k] = (int8_t)((v & 15) |
                            (((wb->qh[k] >> (2 * g)) & 1) << 4));
                        qrow[32 + k] = (int8_t)((v >> 4) |
                            (((wb->qh[k] >> (2 * g + 1)) & 1) << 4));
                    }
                }
            }
        }
    }
    return 0;
}
#endif

#ifndef TF_KQUANT_CACHE_PACK_ONLY
static inline void packed_q5r_dot8(float out[8], const uint8_t *weights,
                                   const tf_kquant_a8_block *x, int nb) {
    const svbool_t p8 = svptrue_b8(), pg = svptrue_b32();
    const svbool_t first8 = svwhilelt_b32(0, 8);
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    svfloat32_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    float c4 = 0, c5 = 0, c6 = 0, c7 = 0;
    size_t bb = packed_q5r_block_bytes();
    for (int b = 0; b < nb; b++) {
        const uint8_t *block = weights + (size_t)b * bb;
        const packed_q5r_header *headers = (const packed_q5r_header *)block;
        const int8_t *q = (const int8_t *)(block + 8 * sizeof(*headers));
        svint32_t i0 = svdup_s32(0), i1 = i0, i2 = i0, i3 = i0;
        svint32_t i4 = i0, i5 = i0, i6 = i0, i7 = i0;
        for (int g = 0; g < 4; g++) {
            svint8_t xv = svld1_s8(p8, x[b].q + g * 64);
#define PACKED_Q5R_ROW(R, IA, C) do { \
    const packed_q5r_header *h = &headers[R]; \
    uint8_t s0 = h->scales[2 * g], s1 = h->scales[2 * g + 1]; \
    uint8_t m0 = h->mins[2 * g], m1 = h->mins[2 * g + 1]; \
    svint32_t dot = svdot_s32(svdup_s32(0), \
        svld1_s8(p8, q + (g * 8 + (R)) * 64), xv); \
    IA = svmla_s32_x(pg, IA, dot, \
        svsel_s32(first8, svdup_s32(s0), svdup_s32(s1))); \
    C -= h->dmin * x[b].d[0] * \
        ((float)m0 * x[b].sum[2 * g] + (float)m1 * x[b].sum[2 * g + 1]); \
} while (0)
            PACKED_Q5R_ROW(0, i0, c0); PACKED_Q5R_ROW(1, i1, c1);
            PACKED_Q5R_ROW(2, i2, c2); PACKED_Q5R_ROW(3, i3, c3);
            PACKED_Q5R_ROW(4, i4, c4); PACKED_Q5R_ROW(5, i5, c5);
            PACKED_Q5R_ROW(6, i6, c6); PACKED_Q5R_ROW(7, i7, c7);
#undef PACKED_Q5R_ROW
        }
#define PACKED_Q5R_SCALE(R, A, I) \
    A = svmla_n_f32_x(pg, A, svcvt_f32_s32_x(pg, I), \
        headers[R].d * x[b].d[0])
        PACKED_Q5R_SCALE(0, a0, i0); PACKED_Q5R_SCALE(1, a1, i1);
        PACKED_Q5R_SCALE(2, a2, i2); PACKED_Q5R_SCALE(3, a3, i3);
        PACKED_Q5R_SCALE(4, a4, i4); PACKED_Q5R_SCALE(5, a5, i5);
        PACKED_Q5R_SCALE(6, a6, i6); PACKED_Q5R_SCALE(7, a7, i7);
#undef PACKED_Q5R_SCALE
    }
    out[0] = svaddv_f32(pg, a0) + c0; out[1] = svaddv_f32(pg, a1) + c1;
    out[2] = svaddv_f32(pg, a2) + c2; out[3] = svaddv_f32(pg, a3) + c3;
    out[4] = svaddv_f32(pg, a4) + c4; out[5] = svaddv_f32(pg, a5) + c5;
    out[6] = svaddv_f32(pg, a6) + c6; out[7] = svaddv_f32(pg, a7) + c7;
}

static int run_packed_q5r(float *y, const uint8_t *weights,
                          const float *x, int rows, int cols) {
    if (!y || !weights || !x || !packed_q5r_bytes(rows, cols)) return -1;
    int nb = cols / 256;
    size_t row_group_bytes = (size_t)nb * packed_q5r_block_bytes();
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        int g0 = (rows / 8) * tid / nt;
        int g1 = (rows / 8) * (tid + 1) / nt;
        tf_kquant_a8_block *qx = alloca((size_t)nb * sizeof(*qx));
        tf_kquant_quant_a8(qx, x, cols);
        for (int g = g0; g < g1; g++)
            packed_q5r_dot8(y + g * 8, weights + (size_t)g * row_group_bytes,
                            qx, nb);
    }
    return 0;
}
#endif

typedef struct {
    float d;
    int8_t scales[8];
    uint8_t padding[4];
} packed_iq4r_header;

_Static_assert(sizeof(packed_iq4r_header) == 16, "packed IQ4R header size");

#ifndef TF_KQUANT_CACHE_LAYOUT_ONLY
static const int8_t packed_iq4r_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};
#endif

static size_t packed_iq4r_block_bytes(void) {
    return 8 * sizeof(packed_iq4r_header) + 8 * 256;
}

static size_t packed_iq4r_bytes(int rows, int cols) {
    if (rows <= 0 || cols <= 0 || rows % TF_KQUANT_CACHE_ROWS ||
        cols % TF_KQUANT_CACHE_COLS) return 0;
    return (size_t)(rows / TF_KQUANT_CACHE_ROWS) *
           (size_t)(cols / TF_KQUANT_CACHE_COLS) * packed_iq4r_block_bytes();
}

#ifndef TF_KQUANT_CACHE_LAYOUT_ONLY
static int pack_iq4r(uint8_t *dst, const block_iq4_xs *src,
                     int rows, int cols) {
    if (!dst || !src || !packed_iq4r_bytes(rows, cols)) return -1;
    int nb = cols / 256;
    size_t bb = packed_iq4r_block_bytes();
#pragma omp parallel for schedule(static)
    for (int rg = 0; rg < rows / 8; rg++) {
        for (int b = 0; b < nb; b++) {
            uint8_t *block = dst + ((size_t)rg * nb + b) * bb;
            packed_iq4r_header *headers = (packed_iq4r_header *)block;
            int8_t *q = (int8_t *)(block + 8 * sizeof(*headers));
            for (int rr = 0; rr < 8; rr++) {
                const block_iq4_xs *wb = src + (size_t)(rg * 8 + rr) * nb + b;
                headers[rr].d = ggml_fp16_to_fp32(wb->d);
                uint16_t high = wb->scales_h;
                for (int ib = 0; ib < 8; ib++) {
                    headers[rr].scales[ib] = (int8_t)(
                        (((wb->scales_l[ib / 2] >> (4 * (ib & 1))) & 15) |
                         (((high >> (2 * ib)) & 3) << 4)) - 32);
                    int8_t *qrow = q + ((ib / 2) * 8 + rr) * 64 + (ib & 1) * 32;
                    const uint8_t *packed = wb->qs + ib * 16;
                    for (int k = 0; k < 16; k++) {
                        qrow[k] = packed_iq4r_values[packed[k] & 15];
                        qrow[16 + k] = packed_iq4r_values[packed[k] >> 4];
                    }
                }
            }
        }
    }
    return 0;
}
#endif

#ifndef TF_KQUANT_CACHE_PACK_ONLY
static inline void packed_iq4r_dot8(float out[8], const uint8_t *weights,
                                    const tf_kquant_a8_block *x, int nb) {
    const svbool_t p8 = svptrue_b8(), pg = svptrue_b32();
    const svbool_t first8 = svwhilelt_b32(0, 8);
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    svfloat32_t a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    size_t bb = packed_iq4r_block_bytes();
    for (int b = 0; b < nb; b++) {
        const uint8_t *block = weights + (size_t)b * bb;
        const packed_iq4r_header *headers = (const packed_iq4r_header *)block;
        const int8_t *q = (const int8_t *)(block + 8 * sizeof(*headers));
        svint32_t i0 = svdup_s32(0), i1 = i0, i2 = i0, i3 = i0;
        svint32_t i4 = i0, i5 = i0, i6 = i0, i7 = i0;
        for (int g = 0; g < 4; g++) {
            svint8_t xv = svld1_s8(p8, x[b].q + g * 64);
#define PACKED_IQ4R_ROW(R, IA) do { \
    const packed_iq4r_header *h = &headers[R]; \
    svint32_t dot = svdot_s32(svdup_s32(0), \
        svld1_s8(p8, q + (g * 8 + (R)) * 64), xv); \
    IA = svmla_s32_x(pg, IA, dot, svsel_s32(first8, \
        svdup_s32(h->scales[2 * g]), svdup_s32(h->scales[2 * g + 1]))); \
} while (0)
            PACKED_IQ4R_ROW(0, i0); PACKED_IQ4R_ROW(1, i1);
            PACKED_IQ4R_ROW(2, i2); PACKED_IQ4R_ROW(3, i3);
            PACKED_IQ4R_ROW(4, i4); PACKED_IQ4R_ROW(5, i5);
            PACKED_IQ4R_ROW(6, i6); PACKED_IQ4R_ROW(7, i7);
#undef PACKED_IQ4R_ROW
        }
#define PACKED_IQ4R_SCALE(R, A, I) \
    A = svmla_n_f32_x(pg, A, svcvt_f32_s32_x(pg, I), \
        headers[R].d * x[b].d[0])
        PACKED_IQ4R_SCALE(0, a0, i0); PACKED_IQ4R_SCALE(1, a1, i1);
        PACKED_IQ4R_SCALE(2, a2, i2); PACKED_IQ4R_SCALE(3, a3, i3);
        PACKED_IQ4R_SCALE(4, a4, i4); PACKED_IQ4R_SCALE(5, a5, i5);
        PACKED_IQ4R_SCALE(6, a6, i6); PACKED_IQ4R_SCALE(7, a7, i7);
#undef PACKED_IQ4R_SCALE
    }
    out[0] = svaddv_f32(pg, a0); out[1] = svaddv_f32(pg, a1);
    out[2] = svaddv_f32(pg, a2); out[3] = svaddv_f32(pg, a3);
    out[4] = svaddv_f32(pg, a4); out[5] = svaddv_f32(pg, a5);
    out[6] = svaddv_f32(pg, a6); out[7] = svaddv_f32(pg, a7);
}

static int run_packed_iq4r(float *y, const uint8_t *weights,
                           const float *x, int rows, int cols) {
    if (!y || !weights || !x || !packed_iq4r_bytes(rows, cols)) return -1;
    int nb = cols / 256;
    size_t row_group_bytes = (size_t)nb * packed_iq4r_block_bytes();
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        int g0 = (rows / 8) * tid / nt;
        int g1 = (rows / 8) * (tid + 1) / nt;
        tf_kquant_a8_block *qx = alloca((size_t)nb * sizeof(*qx));
        tf_kquant_quant_a8(qx, x, cols);
        for (int g = g0; g < g1; g++)
            packed_iq4r_dot8(y + g * 8, weights + (size_t)g * row_group_bytes,
                             qx, nb);
    }
    return 0;
}
#endif

#endif
