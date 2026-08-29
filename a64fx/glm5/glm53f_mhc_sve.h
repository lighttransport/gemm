#ifndef GLM53F_MHC_SVE_H
#define GLM53F_MHC_SVE_H

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include "../../common/glm53f_ref.h"

enum {
    GLM53F_MHC_STREAMS = 4,
    GLM53F_MHC_WIDTH = 4096,
    GLM53F_MHC_FLAT = GLM53F_MHC_STREAMS * GLM53F_MHC_WIDTH,
    GLM53F_MHC_MIX = (2 + GLM53F_MHC_STREAMS) * GLM53F_MHC_STREAMS
};

typedef struct {
    const uint16_t *fn;
    const float *base;
    const float *scale;
} glm53f_mhc_site;

typedef struct {
    float collapsed[GLM53F_MHC_WIDTH];
    float normalized[GLM53F_MHC_WIDTH];
    float residual[GLM53F_MHC_FLAT];
    float post[GLM53F_MHC_STREAMS];
    float combine[GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS];
} glm53f_mhc_scratch;

static inline float glm53f_mhc_dot_bf16_sve(
        const uint16_t *weight, const float *x, int n) {
    svfloat32_t acc = svdup_f32(0);
    int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svuint32_t bits = svlsl_n_u32_x(pg, svld1uh_u32(pg, weight + i), 16);
        acc = svmla_x(pg, acc, svreinterpret_f32_u32(bits), svld1(pg, x + i));
    }
    return svaddv_f32(svptrue_b32(), acc);
}

static inline void glm53f_mhc_pre_sve(
        glm53f_mhc_scratch *scratch, const float *streams,
        const glm53f_mhc_site *site, const uint16_t *norm) {
    double sumsq = 0.0;
    float logits[GLM53F_MHC_MIX];
#pragma omp parallel for reduction(+:sumsq)
    for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
        sumsq += (double)streams[i] * streams[i];
    float inv = 1.0f / sqrtf((float)(sumsq / GLM53F_MHC_FLAT) + 1e-5f);
#pragma omp parallel for schedule(static)
    for (int m = 0; m < GLM53F_MHC_MIX; ++m)
        logits[m] = glm53f_mhc_dot_bf16_sve(
            site->fn + (size_t)m * GLM53F_MHC_FLAT, streams,
            GLM53F_MHC_FLAT) * inv;
    for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
        logits[k] = glm53f_sigmoid(logits[k] * site->scale[0] + site->base[k]) + 1e-6f;
        scratch->post[k] = 2.0f * glm53f_sigmoid(
            logits[GLM53F_MHC_STREAMS + k] * site->scale[1] +
            site->base[GLM53F_MHC_STREAMS + k]);
    }
    for (int m = 0; m < GLM53F_MHC_STREAMS * GLM53F_MHC_STREAMS; ++m)
        scratch->combine[m] = logits[2 * GLM53F_MHC_STREAMS + m] *
                              site->scale[2] + site->base[2 * GLM53F_MHC_STREAMS + m];
    glm53f_mhc_sinkhorn(scratch->combine, GLM53F_MHC_STREAMS, 20, 1e-6f);
#pragma omp parallel for schedule(static)
    for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
        float value = 0.0f;
        for (int k = 0; k < GLM53F_MHC_STREAMS; ++k)
            value += logits[k] * streams[(size_t)k * GLM53F_MHC_WIDTH + d];
        scratch->collapsed[d] = value;
    }
    memcpy(scratch->residual, streams, sizeof(scratch->residual));
    glm53f_rmsnorm_bf16(scratch->normalized, scratch->collapsed, norm,
                        GLM53F_MHC_WIDTH, 1e-5f);
}

static inline void glm53f_mhc_post_sve(
        float *streams, const float *sublayer, const glm53f_mhc_scratch *scratch) {
    glm53f_mhc_post(streams, scratch->residual, sublayer, scratch->post,
                    scratch->combine, GLM53F_MHC_STREAMS, GLM53F_MHC_WIDTH);
}

#endif
