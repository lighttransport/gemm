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

static inline float glm53f_fp8_e4m3_scalar(uint8_t q) {
    int sign = q >> 7, exponent = (q >> 3) & 15, mantissa = q & 7;
    float v;
    if (!exponent) v = (float)mantissa / 512.0f;
    else if (exponent == 15 && mantissa == 7) return NAN;
    else v = ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -v : v;
}

static inline float glm53f_dot_fp8_block128(
        const uint8_t *w, const float *scale, const float *x, int cols) {
    double acc = 0;
    for (int c = 0; c < cols; ++c)
        acc += (double)glm53f_fp8_e4m3_scalar(w[c]) * scale[c / 128] * x[c];
    return (float)acc;
}

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
    /* The staged checkpoint contract excludes E4M3 NaNs (0x7f/0xff).
     * Avoid two compares and a select in the decode-token inner loop. */
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

static inline void glm53f_mv_fp8_block128_bits(
        float *y, const uint8_t *w, const float *scale,
        const float *x, int rows, int cols) {
    int blocks = (cols + 127) / 128, n8 = rows / 8;
#pragma omp parallel for schedule(static)
    for (int bi = 0; bi < n8; ++bi) {
        int r = bi * 8;
        glm53f_matvec_fp8_bits_8(
            y + r, w + (size_t)r * cols,
            scale + (size_t)(r / 128) * blocks, x, cols);
    }
    for (int r = n8 * 8; r < rows; ++r)
        y[r] = glm53f_dot_fp8_block128(
            w + (size_t)r * cols, scale + (size_t)(r / 128) * blocks, x, cols);
}

static inline void glm53f_mv_fp8_block128_bits_2(
        float *y0, const uint8_t *w0, const float *s0, int rows0,
        float *y1, const uint8_t *w1, const float *s1, int rows1,
        const float *x, int cols) {
    int blocks = (cols + 127) / 128, n0 = rows0 / 8, n1 = rows1 / 8;
#pragma omp parallel for schedule(static)
    for (int bi = 0; bi < n0 + n1; ++bi) {
        int second = bi >= n0, r = (second ? bi - n0 : bi) * 8;
        const uint8_t *w = second ? w1 : w0;
        const float *s = second ? s1 : s0;
        float *y = second ? y1 : y0;
        glm53f_matvec_fp8_bits_8(
            y + r, w + (size_t)r * cols,
            s + (size_t)(r / 128) * blocks, x, cols);
    }
    for (int r = n0 * 8; r < rows0; ++r)
        y0[r] = glm53f_dot_fp8_block128(
            w0 + (size_t)r * cols, s0 + (size_t)(r / 128) * blocks, x, cols);
    for (int r = n1 * 8; r < rows1; ++r)
        y1[r] = glm53f_dot_fp8_block128(
            w1 + (size_t)r * cols, s1 + (size_t)(r / 128) * blocks, x, cols);
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
        int total_units = 0, task = batch, lane = 0, lanes = 0, prefix = 0;
        for (int j = 0; j < batch; ++j) total_units += part[j].inter / 128;
        for (int j = 0; j < batch; ++j) {
            int units = part[j].inter / 128;
            int begin = nth * prefix / total_units;
            int end = nth * (prefix + units) / total_units;
            if (tid >= begin && tid < end) { task = j; lane = tid - begin; lanes = end - begin; }
            prefix += units;
        }
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
