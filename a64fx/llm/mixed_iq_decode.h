/* A64FX SVE-512 fused IQ lookup / FP32 dot products.
 * Included from transformer.h after ggml_dequant.h and arm_sve.h.
 * FP32 activations by default; Q8 activations and lossless expanded FFN
 * weights are separate, explicit opt-ins.
 */
#ifndef A64FX_MIXED_IQ_DECODE_H
#define A64FX_MIXED_IQ_DECODE_H

static uint64_t tf_mixed_sign64[256], tf_mixed_ksign64[128];
static pthread_once_t tf_mixed_sign_once = PTHREAD_ONCE_INIT;
static int tf_mixed_iq_q8_enabled;
typedef struct { float d; int8_t q[256]; } tf_mixed_q8_block;
typedef struct { int8_t q[256]; float d[16]; } tf_mixed_weight_block;

static void tf_mixed_quant_q8(tf_mixed_q8_block *out, const float *x, int n) {
    svbool_t pg = svptrue_b32();
    for (int b = 0; b < n / 256; b++) {
        svfloat32_t vmax = svdup_f32(0);
        for (int k = 0; k < 256; k += 16)
            vmax = svmax_f32_x(pg, vmax, svabs_f32_x(pg, svld1_f32(pg, x+b*256+k)));
        float max = svmaxv_f32(pg, vmax);
        out[b].d = max > 0 ? max / 127.f : 0;
        float inv = max > 0 ? 127.f / max : 0;
        for (int k = 0; k < 256; k += 16) {
            svint32_t q = svcvt_s32_f32_x(pg, svrintn_f32_x(pg,
                svmul_n_f32_x(pg, svld1_f32(pg, x+b*256+k), inv)));
            q = svmin_n_s32_x(pg, svmax_n_s32_x(pg, q, -127), 127);
            svst1b_s32(pg, out[b].q+k, q);
        }
    }
}

static void tf_mixed_sign_init(void) {
    for (int s = 0; s < 256; s++) {
        uint64_t bits = 0;
        for (int j = 0; j < 8; j++)
            if (s & (1 << j)) bits |= UINT64_C(255) << (8 * j);
        tf_mixed_sign64[s] = bits;
    }
    for (int s = 0; s < 128; s++)
        tf_mixed_ksign64[s] = tf_mixed_sign64[ksigns_iq2xs[s]];
}

static inline int tf_mixed_iq_expandable(uint32_t type) {
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
           type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S;
}

static inline int tf_mixed_iq_supported(uint32_t type, int n) {
    return n > 0 && n % 256 == 0 && svcntb() == 64 &&
        (tf_mixed_iq_expandable(type) || type == GGML_TYPE_Q2_K ||
         type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M);
}

static inline __attribute__((always_inline)) float tf_mixed_iq1_dot_sve(
    const void *weights, const float *x, int n, uint32_t type) {
    const svbool_t p8 = svptrue_b8(), pg = svptrue_b32(), p64 = svptrue_b64();
    const svbool_t first4 = svwhilelt_b64(0, 4);
    const svuint64_t sh3 = svmul_n_u64_x(p64, svand_n_u64_x(p64, svindex_u64(0, 1), 3), 3);
    const svuint64_t sh4 = svindex_u64(0, 4);
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int b = 0; b < n/256; b++) {
        for (int g = 0; g < 4; g++) {
            svuint64_t hi, ix, negative;
            float s0, s1, s2, s3;
            if (type == GGML_TYPE_IQ1_S) {
                const block_iq1_s *w = (const block_iq1_s *)weights + b;
                hi = svsel_u64(first4, svdup_u64(w->qh[g*2]), svdup_u64(w->qh[g*2+1]));
                ix = svorr_u64_x(p64, svld1ub_u64(p64, w->qs+g*8), svlsl_n_u64_x(p64,
                    svand_n_u64_x(p64, svlsr_u64_x(p64, hi, sh3), 7), 8));
                negative = svand_n_u64_x(p64, hi, 0x8000);
                float d = ggml_fp16_to_fp32(w->d) * 0.125f;
                s0 = s1 = d * (1 + 2*((w->qh[g*2]>>12)&7));
                s2 = s3 = d * (1 + 2*((w->qh[g*2+1]>>12)&7));
            } else {
                const block_iq1_m *w = (const block_iq1_m *)weights + b;
                uint16_t sc[4];
                uint32_t qh;
                memcpy(sc, w->scales, sizeof(sc));
                memcpy(&qh, w->qh+g*4, sizeof(qh));
                uint16_t dh = (sc[0]>>12) | ((sc[1]>>8)&0xf0) | ((sc[2]>>4)&0xf00) | (sc[3]&0xf000);
                hi = svlsr_u64_x(p64, svdup_u64(qh), sh4);
                ix = svorr_u64_x(p64, svld1ub_u64(p64, w->qs+g*8),
                    svlsl_n_u64_x(p64, svand_n_u64_x(p64, hi, 7), 8));
                negative = svand_n_u64_x(p64, hi, 8);
                float d = ggml_fp16_to_fp32(dh) * 0.125f;
                s0 = d * (1 + 2*((sc[g]>>0)&7));
                s1 = d * (1 + 2*((sc[g]>>3)&7));
                s2 = d * (1 + 2*((sc[g]>>6)&7));
                s3 = d * (1 + 2*((sc[g]>>9)&7));
            }
            svint8_t grid = svreinterpret_s8_u64(svld1_gather_u64index_u64(p64, iq1s_grid, ix));
            /* Exactly encode grid +/- 1/8 as (8*grid +/- 1)/8. */
            svint8_t delta = svreinterpret_s8_u64(svsel_u64(svcmpne_n_u64(p64, negative, 0),
                svdup_u64(UINT64_MAX), svdup_u64(UINT64_C(0x0101010101010101))));
            svint8_t q = svadd_s8_x(p8, svlsl_n_s8_x(p8, grid, 3), delta);
            svint16_t lo = svunpklo_s16(q), hi16 = svunpkhi_s16(q);
            const float *v = x+b*256+g*64;
            a0 = svmla_f32_x(pg, a0, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, svunpklo_s32(lo)), s0), svld1_f32(pg,v));
            a1 = svmla_f32_x(pg, a1, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, svunpkhi_s32(lo)), s1), svld1_f32(pg,v+16));
            a2 = svmla_f32_x(pg, a2, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, svunpklo_s32(hi16)), s2), svld1_f32(pg,v+32));
            a3 = svmla_f32_x(pg, a3, svmul_n_f32_x(pg, svcvt_f32_s32_x(pg, svunpkhi_s32(hi16)), s3), svld1_f32(pg,v+48));
        }
    }
    return svaddv_f32(pg, svadd_f32_x(pg, svadd_f32_x(pg,a0,a1), svadd_f32_x(pg,a2,a3)));
}

static inline float tf_mixed_q2_k_dot_sve(const block_q2_K *w, const float *x, int n) {
    svbool_t pg = svptrue_b32();
    svfloat32_t a0 = svdup_f32(0), a1 = a0;
    for (int b = 0; b < n/256; b++) {
        float d = ggml_fp16_to_fp32(w[b].d), dm = ggml_fp16_to_fp32(w[b].dmin);
        for (int h = 0; h < 2; h++) {
            svuint32_t q0 = svld1ub_u32(pg, w[b].qs + h*32);
            svuint32_t q1 = svld1ub_u32(pg, w[b].qs + h*32 + 16);
            for (int g = 0; g < 4; g++) {
                int s0 = w[b].scales[h*8 + g*2], s1 = w[b].scales[h*8 + g*2+1];
                svfloat32_t v0 = svcvt_f32_u32_x(pg, svand_n_u32_x(pg,
                    svlsr_u32_x(pg, q0, svdup_u32(2*g)), 3));
                svfloat32_t v1 = svcvt_f32_u32_x(pg, svand_n_u32_x(pg,
                    svlsr_u32_x(pg, q1, svdup_u32(2*g)), 3));
                v0 = svsub_n_f32_x(pg, svmul_n_f32_x(pg, v0, d*(s0&15)), dm*(s0>>4));
                v1 = svsub_n_f32_x(pg, svmul_n_f32_x(pg, v1, d*(s1&15)), dm*(s1>>4));
                a0 = svmla_f32_x(pg, a0, v0, svld1_f32(pg, x+b*256+h*128+g*32));
                a1 = svmla_f32_x(pg, a1, v1, svld1_f32(pg, x+b*256+h*128+g*32+16));
            }
        }
    }
    return svaddv_f32(pg, svadd_f32_x(pg, a0, a1));
}

static inline __attribute__((always_inline)) float tf_mixed_iq_dot_sve(const void *weights, const float *x,
                                       int n, uint32_t type, const tf_mixed_q8_block *qx) {
    pthread_once(&tf_mixed_sign_once, tf_mixed_sign_init);
    const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32(), p64 = svptrue_b64();
    const svuint64_t shift2 = svindex_u64(0, 2);
    const svuint32_t shift1 = svindex_u32(0, 1);
    static const uint64_t sh7_data[8] = {0, 7, 14, 21, 0, 7, 14, 21};
    const svuint64_t sh7 = svld1_u64(p64, sh7_data);
    const svuint64_t sh8 = svand_n_u64_x(p64, svindex_u64(0, 8), 31);
    const svbool_t first4 = svwhilelt_b64(0, 4);
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int b = 0; b < n / 256; b++) {
        for (int g = 0; g < 4; g++) {
            svuint64_t grid, signs;
            float s0, s1, s2, s3;
            if (type == GGML_TYPE_IQ3_S) {
                const block_iq3_s *w = (const block_iq3_s *)weights + b;
                unsigned high = w->qh[2*g] | ((unsigned)w->qh[2*g+1] << 8);
                svuint32_t ix = svorr_u32_x(p32, svld1ub_u32(p32, w->qs + 16*g),
                    svlsl_n_u32_x(p32, svand_n_u32_x(p32,
                        svlsr_u32_x(p32, svdup_u32(high), shift1), 1), 8));
                grid = svreinterpret_u64_u32(svld1_gather_u32index_u32(p32, iq3s_grid, ix));
                signs = svld1_gather_u64index_u64(p64, tf_mixed_sign64,
                    svld1ub_u64(p64, w->signs + 8*g));
                float d = ggml_fp16_to_fp32(w->d);
                s0 = s1 = d * (1 + 2 * (w->scales[g] & 15));
                s2 = s3 = d * (1 + 2 * (w->scales[g] >> 4));
            } else if (type == GGML_TYPE_IQ2_S) {
                const block_iq2_s *w = (const block_iq2_s *)weights + b;
                unsigned high = w->qh[2*g] | ((unsigned)w->qh[2*g+1] << 8);
                svuint64_t ix = svorr_u64_x(p64, svld1ub_u64(p64, w->qs + 8*g),
                    svlsl_n_u64_x(p64, svand_n_u64_x(p64,
                        svlsr_u64_x(p64, svdup_u64(high), shift2), 3), 8));
                grid = svld1_gather_u64index_u64(p64, iq2s_grid, ix);
                signs = svld1_gather_u64index_u64(p64, tf_mixed_sign64,
                    svld1ub_u64(p64, w->qs + 32 + 8*g));
                float d = ggml_fp16_to_fp32(w->d) * 0.125f;
                s0 = d * (1 + 2 * (w->scales[2*g] & 15));
                s1 = d * (1 + 2 * (w->scales[2*g] >> 4));
                s2 = d * (1 + 2 * (w->scales[2*g+1] & 15));
                s3 = d * (1 + 2 * (w->scales[2*g+1] >> 4));
            } else if (type == GGML_TYPE_IQ2_XS) {
                const block_iq2_xs *w = (const block_iq2_xs *)weights + b;
                svuint64_t q = svld1uh_u64(p64, w->qs + 8*g);
                grid = svld1_gather_u64index_u64(p64, iq2xs_grid, svand_n_u64_x(p64, q, 511));
                signs = svld1_gather_u64index_u64(p64, tf_mixed_ksign64, svlsr_n_u64_x(p64, q, 9));
                float d = ggml_fp16_to_fp32(w->d) * 0.125f;
                s0 = d * (1 + 2 * (w->scales[2*g] & 15));
                s1 = d * (1 + 2 * (w->scales[2*g] >> 4));
                s2 = d * (1 + 2 * (w->scales[2*g+1] & 15));
                s3 = d * (1 + 2 * (w->scales[2*g+1] >> 4));
            } else {
                const block_iq2_xxs *w = (const block_iq2_xxs *)weights + b;
                uint32_t raw[4];
                memcpy(raw, w->qs + 8*g, sizeof(raw));
                svuint64_t indices = svsel_u64(first4, svdup_u64(raw[0]), svdup_u64(raw[2]));
                grid = svld1_gather_u64index_u64(p64, iq2xxs_grid,
                    svand_n_u64_x(p64, svlsr_u64_x(p64, indices, sh8), 255));
                svuint64_t aux = svsel_u64(first4, svdup_u64(raw[1]), svdup_u64(raw[3]));
                signs = svld1_gather_u64index_u64(p64, tf_mixed_ksign64,
                    svand_n_u64_x(p64, svlsr_u64_x(p64, aux, sh7), 127));
                float d = ggml_fp16_to_fp32(w->d) * 0.125f;
                s0 = s1 = d * (1 + 2 * (raw[1] >> 28));
                s2 = s3 = d * (1 + 2 * (raw[3] >> 28));
            }
            svint8_t q = svsub_s8_x(p8,
                svreinterpret_s8_u64(sveor_u64_x(p64, grid, signs)),
                svreinterpret_s8_u64(signs));
            if (qx) {
                svint32_t dot = svdot_s32(svdup_s32(0), q, svld1_s8(p8, qx[b].q + g*64));
                svfloat32_t scale = svsel_f32(svwhilelt_b32(0, 8),
                    svsel_f32(svwhilelt_b32(0, 4), svdup_f32(s0), svdup_f32(s1)),
                    svsel_f32(svwhilelt_b32(0, 12), svdup_f32(s2), svdup_f32(s3)));
                a0 = svmla_f32_x(p32, a0, svcvt_f32_s32_x(p32, dot),
                                 svmul_n_f32_x(p32, scale, qx[b].d));
                continue;
            }
            svint16_t lo = svunpklo_s16(q), hi = svunpkhi_s16(q);
            svfloat32_t w0 = svcvt_f32_s32_x(p32, svunpklo_s32(lo));
            svfloat32_t w1 = svcvt_f32_s32_x(p32, svunpkhi_s32(lo));
            svfloat32_t w2 = svcvt_f32_s32_x(p32, svunpklo_s32(hi));
            svfloat32_t w3 = svcvt_f32_s32_x(p32, svunpkhi_s32(hi));
            const float *v = x + b*256 + g*64;
            a0 = svmla_f32_x(p32, a0, svmul_n_f32_x(p32, w0, s0), svld1_f32(p32, v));
            a1 = svmla_f32_x(p32, a1, svmul_n_f32_x(p32, w1, s1), svld1_f32(p32, v+16));
            a2 = svmla_f32_x(p32, a2, svmul_n_f32_x(p32, w2, s2), svld1_f32(p32, v+32));
            a3 = svmla_f32_x(p32, a3, svmul_n_f32_x(p32, w3, s3), svld1_f32(p32, v+48));
        }
    }
    return svaddv_f32(p32, svadd_f32_x(p32,
        svadd_f32_x(p32, a0, a1), svadd_f32_x(p32, a2, a3)));
}

/* Constant type at each call site lets the compiler specialize the inner
 * loop and hoist the block scale without per-group format branches. */
static void tf_mixed_iq_rows(float *dst, const void *weights, const float *x,
                             int n, uint32_t type, int start, int end) {
    if (type == GGML_TYPE_IQ1_S) {
        for (int r = start; r < end; r++)
            dst[r] = tf_mixed_iq1_dot_sve((const block_iq1_s *)weights+(size_t)r*(n/256), x,n,GGML_TYPE_IQ1_S);
        return;
    }
    if (type == GGML_TYPE_IQ1_M) {
        for (int r = start; r < end; r++)
            dst[r] = tf_mixed_iq1_dot_sve((const block_iq1_m *)weights+(size_t)r*(n/256), x,n,GGML_TYPE_IQ1_M);
        return;
    }
    if (type == GGML_TYPE_Q2_K) {
        for (int r = start; r < end; r++)
            dst[r] = tf_mixed_q2_k_dot_sve((const block_q2_K *)weights + (size_t)r*(n/256), x, n);
        return;
    }
    tf_mixed_q8_block *qx = NULL;
    if (tf_mixed_iq_q8_enabled) {
        qx = (tf_mixed_q8_block *)alloca((size_t)(n/256) * sizeof(*qx));
        tf_mixed_quant_q8(qx, x, n);
    }
#define TF_MIXED_ROWS_CASE(TYPE, BLOCK, QX) \
    case TYPE: \
        for (int r = start; r < end; r++) \
            dst[r] = tf_mixed_iq_dot_sve((const BLOCK *)weights + (size_t)r * (n/256), x, n, TYPE, QX); \
        break
    if (qx) {
        switch (type) {
            TF_MIXED_ROWS_CASE(GGML_TYPE_IQ2_XXS, block_iq2_xxs, qx);
            TF_MIXED_ROWS_CASE(GGML_TYPE_IQ2_XS, block_iq2_xs, qx);
            TF_MIXED_ROWS_CASE(GGML_TYPE_IQ2_S, block_iq2_s, qx);
            TF_MIXED_ROWS_CASE(GGML_TYPE_IQ3_S, block_iq3_s, qx);
        }
    } else {
        switch (type) {
            TF_MIXED_ROWS_CASE(GGML_TYPE_IQ2_XXS, block_iq2_xxs, NULL);
            TF_MIXED_ROWS_CASE(GGML_TYPE_IQ2_XS, block_iq2_xs, NULL);
            TF_MIXED_ROWS_CASE(GGML_TYPE_IQ2_S, block_iq2_s, NULL);
            TF_MIXED_ROWS_CASE(GGML_TYPE_IQ3_S, block_iq3_s, NULL);
        }
    }
#undef TF_MIXED_ROWS_CASE
}

/* Lossless codebook expansion: retain the original 16-value scales rather
 * than fitting a new per-row quantizer. Reject any non-exact reconstruction. */
static int tf_mixed_expand_row(tf_mixed_weight_block *dst, const void *src,
                                uint32_t type, int n) {
    size_t rb = tf_row_bytes(type, 256);
    float decoded[256];
    for (int b = 0; b < n/256; b++) {
        const uint8_t *raw = (const uint8_t *)src + (size_t)b * rb;
        dequant_row(type, raw, decoded, 256);
        for (int g = 0; g < 16; g++) {
            float d;
            if (type == GGML_TYPE_IQ3_S) {
                const block_iq3_s *w = (const block_iq3_s *)raw;
                int sc = (w->scales[g/4] >> ((g%4)/2*4)) & 15;
                d = ggml_fp16_to_fp32(w->d) * (1 + 2*sc);
            } else if (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS) {
                const uint8_t *scales = type == GGML_TYPE_IQ2_S ?
                    ((const block_iq2_s *)raw)->scales : ((const block_iq2_xs *)raw)->scales;
                uint16_t dh;
                memcpy(&dh, raw, sizeof(dh));
                d = ggml_fp16_to_fp32(dh) * (1 + 2*((scales[g/2] >> (g%2*4)) & 15)) * 0.125f;
            } else if (type == GGML_TYPE_IQ2_XXS) {
                const block_iq2_xxs *w = (const block_iq2_xxs *)raw;
                uint32_t aux;
                memcpy(&aux, (const uint8_t *)w->qs + (g/2)*8 + 4, sizeof(aux));
                d = ggml_fp16_to_fp32(w->d) * (1 + 2*(aux>>28)) * 0.125f;
            } else return -1;
            if (!isfinite(d)) return -1;
            dst[b].d[g] = d;
            float inv = d ? 1.f/d : 0;
            for (int j = 0; j < 16; j++) {
                float value = decoded[g*16+j];
                float q = nearbyintf(value * inv);
                if (!isfinite(q) || q < -127 || q > 127 || q*d != value) return -1;
                dst[b].q[g*16+j] = (int8_t)q;
            }
        }
    }
    return 0;
}

static inline __attribute__((always_inline)) float tf_mixed_cached_dot(
    const tf_mixed_weight_block *w, const float *x, const tf_mixed_q8_block *qx, int nb) {
    svbool_t pg = svptrue_b32(), p8 = svptrue_b8();
    svuint32_t scale_ix = svlsr_n_u32_x(pg, svindex_u32(0, 1), 2);
    svfloat32_t a0 = svdup_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int b = 0; b < nb; b++) {
        for (int g = 0; g < 4; g++) {
            if (qx) {
                svint32_t dot = svdot_s32(svdup_s32(0),
                    svld1_s8(p8, w[b].q+g*64), svld1_s8(p8, qx[b].q+g*64));
                svfloat32_t scale = svtbl_f32(svld1rq_f32(pg, w[b].d+g*4), scale_ix);
                a0 = svmla_f32_x(pg, a0, svcvt_f32_s32_x(pg, dot),
                                 svmul_n_f32_x(pg, scale, qx[b].d));
            } else {
                const int8_t *q = w[b].q + g*64;
                const float *v = x + b*256 + g*64, *d = w[b].d + g*4;
                a0 = svmla_f32_x(pg, a0, svmul_n_f32_x(pg,
                    svcvt_f32_s32_x(pg, svld1sb_s32(pg, q)), d[0]), svld1_f32(pg, v));
                a1 = svmla_f32_x(pg, a1, svmul_n_f32_x(pg,
                    svcvt_f32_s32_x(pg, svld1sb_s32(pg, q+16)), d[1]), svld1_f32(pg, v+16));
                a2 = svmla_f32_x(pg, a2, svmul_n_f32_x(pg,
                    svcvt_f32_s32_x(pg, svld1sb_s32(pg, q+32)), d[2]), svld1_f32(pg, v+32));
                a3 = svmla_f32_x(pg, a3, svmul_n_f32_x(pg,
                    svcvt_f32_s32_x(pg, svld1sb_s32(pg, q+48)), d[3]), svld1_f32(pg, v+48));
            }
        }
    }
    return svaddv_f32(pg, svadd_f32_x(pg, svadd_f32_x(pg, a0, a1), svadd_f32_x(pg, a2, a3)));
}

static void tf_mixed_cached_rows(float *dst, const void *cache, const float *x,
                                 int n, int start, int end) {
    const tf_mixed_weight_block *w = (const tf_mixed_weight_block *)cache;
    int nb = n/256;
    if (tf_mixed_iq_q8_enabled) {
        tf_mixed_q8_block *qx = (tf_mixed_q8_block *)alloca((size_t)nb*sizeof(*qx));
        tf_mixed_quant_q8(qx, x, n);
        for (int r = start; r < end; r++)
            dst[r] = tf_mixed_cached_dot(w+(size_t)r*nb, x, qx, nb);
    } else {
        for (int r = start; r < end; r++)
            dst[r] = tf_mixed_cached_dot(w+(size_t)r*nb, x, NULL, nb);
    }
}

#endif
