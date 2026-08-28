#ifndef GLM53F_EXPERT_KERN_H
#define GLM53F_EXPERT_KERN_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    const uint8_t *gate_up;
    const float *gate_up_scale;
    const uint8_t *down;
    const float *down_scale;
    int inter;
} glm53f_expert_part;

#if defined(__ARM_FEATURE_SVE)
static inline svfloat32_t glm53f_fp8_e4m3_bits(svbool_t pg, const uint8_t *w, int c) {
    svuint32_t q = svld1ub_u32(pg, w + c);
    svuint32_t sign = svlsl_n_u32_x(pg, svand_n_u32_x(pg, q, 0x80), 24);
    svuint32_t exponent = svand_n_u32_x(pg, svlsr_n_u32_x(pg, q, 3), 15);
    svuint32_t mantissa = svand_n_u32_x(pg, q, 7);
    svuint32_t normal = svorr_u32_x(
        pg, sign, svlsl_n_u32_x(pg, svadd_n_u32_x(pg, exponent, 120), 23));
    normal = svorr_u32_x(pg, normal, svlsl_n_u32_x(pg, mantissa, 20));
    svfloat32_t subnormal = svmul_n_f32_x(
        pg, svcvt_f32_u32_x(pg, mantissa), 1.0f / 512.0f);
    svuint32_t bits = svsel_u32(
        svcmpne_n_u32(pg, exponent, 0), normal,
        svorr_u32_x(pg, svreinterpret_u32_f32(subnormal), sign));
    svbool_t nan = svand_b_z(
        pg, svcmpeq_n_u32(pg, exponent, 15), svcmpeq_n_u32(pg, mantissa, 7));
    bits = svsel_u32(nan, svorr_n_u32_x(pg, sign, 0x7fc00000u), bits);
    return svreinterpret_f32_u32(bits);
}

static inline void glm53f_matvec_fp8_bits_8(
        float *dst, const uint8_t *w, const float *scale,
        const float *x, int cols) {
    svfloat32_t a0 = svdup_f32(0), a1 = svdup_f32(0);
    svfloat32_t a2 = svdup_f32(0), a3 = svdup_f32(0);
    svfloat32_t a4 = svdup_f32(0), a5 = svdup_f32(0);
    svfloat32_t a6 = svdup_f32(0), a7 = svdup_f32(0);
    int vl = (int)svcntw();
    for (int b = 0; b < cols; b += 128) {
        int end = b + 128 < cols ? b + 128 : cols;
        int block = b / 128;
        for (int c = b; c < end; c += vl) {
            svbool_t pg = svwhilelt_b32(c, end);
            svfloat32_t xv = svld1(pg, x + c);
            svfloat32_t xs = svmul_n_f32_x(pg, xv, scale[block]);
#define GLM53F_FP8_ROW(J, A) \
            A = svmla_x(pg, A, glm53f_fp8_e4m3_bits( \
                pg, w + (size_t)(J) * cols, c), xs)
            GLM53F_FP8_ROW(0, a0); GLM53F_FP8_ROW(1, a1);
            GLM53F_FP8_ROW(2, a2); GLM53F_FP8_ROW(3, a3);
            GLM53F_FP8_ROW(4, a4); GLM53F_FP8_ROW(5, a5);
            GLM53F_FP8_ROW(6, a6); GLM53F_FP8_ROW(7, a7);
#undef GLM53F_FP8_ROW
        }
    }
    svbool_t pt = svptrue_b32();
    dst[0] = svaddv_f32(pt, a0); dst[1] = svaddv_f32(pt, a1);
    dst[2] = svaddv_f32(pt, a2); dst[3] = svaddv_f32(pt, a3);
    dst[4] = svaddv_f32(pt, a4); dst[5] = svaddv_f32(pt, a5);
    dst[6] = svaddv_f32(pt, a6); dst[7] = svaddv_f32(pt, a7);
}

/* Run `batch` independent quarter experts concurrently. Threads are divided
 * into equal teams; batch=4 and 48 threads maps one 12-core team to each CMG. */
static inline void glm53f_expert_batch_bits(
        const glm53f_expert_part *part, int batch, const float *x,
        float *up, float *act, float *y) {
    enum { HIDDEN = 4096, INTER_STRIDE = 512, GATE_UP_STRIDE = 1024 };
#pragma omp parallel
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        int lanes = nth / batch;
        int task = lanes ? tid / lanes : batch;
        int lane = lanes ? tid % lanes : 0;
        if (task < batch) {
            int inter = part[task].inter, gate_up = 2 * inter;
            for (int bi = lane; bi < gate_up / 8; bi += lanes) {
                int r = bi * 8;
                glm53f_matvec_fp8_bits_8(
                    up + (size_t)task * GATE_UP_STRIDE + r,
                    part[task].gate_up + (size_t)r * HIDDEN,
                    part[task].gate_up_scale + (size_t)(r / 128) * (HIDDEN / 128),
                    x, HIDDEN);
            }
        }
#pragma omp barrier
        if (task < batch) {
            int inter = part[task].inter;
            for (int i = lane; i < inter; i += lanes) {
                float g = up[(size_t)task * GATE_UP_STRIDE + i];
                float u = up[(size_t)task * GATE_UP_STRIDE + inter + i];
                if (g > 10) g = 10;
                if (g < -100) g = -100;
                if (u > 10) u = 10;
                if (u < -10) u = -10;
                act[(size_t)task * INTER_STRIDE + i] = (g / (1 + expf(-g))) * u;
            }
        }
#pragma omp barrier
        if (task < batch) {
            int inter = part[task].inter;
            for (int bi = lane; bi < HIDDEN / 8; bi += lanes) {
                int r = bi * 8;
                glm53f_matvec_fp8_bits_8(
                    y + (size_t)task * HIDDEN + r,
                    part[task].down + (size_t)r * inter,
                    part[task].down_scale + (size_t)(r / 128) * (inter / 128),
                    act + (size_t)task * INTER_STRIDE, inter);
            }
        }
    }
}
#endif

#endif
