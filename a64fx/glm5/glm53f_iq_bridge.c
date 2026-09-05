/* Reuse the validated GLM-5.2 A64FX mixed-IQ kernels in the GLM-5.3 graph.
 * Keeping this bridge in one translation unit avoids duplicating the large IQ
 * lookup tables in every consumer of glm53f_expert_kern.h. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GLM5_IMPL
#include "../../common/glm5.h"
#include "glm53f_iq_bridge.h"

int glm53f_iq_type_supported(int type) {
    return type == GLM53F_GGML_Q4_K || type == GLM53F_GGML_Q5_K ||
           type == GLM53F_GGML_Q6_K ||
           type == GLM53F_GGML_IQ2_XS ||
           type == GLM53F_GGML_IQ3_XXS ||
           type == GLM53F_GGML_IQ4_XS;
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
    static const uint8_t idx0[64] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
    static const uint8_t idx1[64] = {
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    const svuint8_t vi0=svld1_u8(p8,idx0), vi1=svld1_u8(p8,idx1);
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
            const svint8_t qv = svreinterpret_s8_u8(svorr_u8_x(p8,
                svtbl_u8(uq0,vi0), svtbl_u8(uq1,vi1)));
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
    static const uint8_t idx0[64] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
    static const uint8_t idx1[64] = {
        255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    const svuint8_t vi0=svld1_u8(p8,idx0), vi1=svld1_u8(p8,idx1);
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
            const svint8_t qv=svorr_s8_x(p8,svtbl_s8(q0,vi0),svtbl_s8(q1,vi1));
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

int glm53f_iq_expert_weighted(
        float *output, const glm53f_iq_part *parts, const float *weights,
        int count, const float *input, float *gate_up, float *activation) {
    enum { HIDDEN = 4096, GU_STRIDE = 1024, ACT_STRIDE = 512 };
    glm5_iq_q8_block input_q[HIDDEN / 256];
    glm5_iq_q8_block act_q[9][ACT_STRIDE / 256];
    size_t gate_rb[9], down_rb[9];
    int gate_blocks = HIDDEN / 256;
    if (!output || !parts || !weights || !input || !gate_up || !activation ||
        count < 1 || count > 9) return -1;
    pthread_once(&glm5_iq_lut_once, glm5_iq_init_luts);
    glm5_iq_quant_q8(input_q, input, HIDDEN);
    for (int k = 0; k < count; ++k) {
        if (!glm53f_iq_type_supported(parts[k].gate_type) ||
            !glm53f_iq_type_supported(parts[k].down_type) ||
            parts[k].inter < 256 || parts[k].inter > ACT_STRIDE ||
            parts[k].inter % 256) return -1;
        gate_rb[k] = dequant_row_size((uint32_t)parts[k].gate_type, HIDDEN);
        down_rb[k] = dequant_row_size((uint32_t)parts[k].down_type, parts[k].inter);
    }
#pragma omp parallel
    {
        int total_gate_rows = 0;
        for (int k = 0; k < count; ++k) total_gate_rows += 2 * parts[k].inter;
#pragma omp for schedule(static)
        for (int q = 0; q < total_gate_rows; ++q) {
            int k = 0, r = q;
            while (r >= 2 * parts[k].inter) r -= 2 * parts[k++].inter;
            gate_up[(size_t)k * GU_STRIDE + r] = iq_row(
                parts[k].gate_type, parts[k].gate_up + (size_t)r * gate_rb[k],
                input_q, gate_blocks);
        }
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
#pragma omp single
        for (int k = 0; k < count; ++k)
            glm5_iq_quant_q8(act_q[k], activation + (size_t)k * ACT_STRIDE,
                             parts[k].inter);
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
    return 0;
}
