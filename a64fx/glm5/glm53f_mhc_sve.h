#ifndef GLM53F_MHC_SVE_H
#define GLM53F_MHC_SVE_H

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>
#include "../../common/glm53f_ref.h"

/* The default remains the validated implementation.  The fused variant keeps
 * the mHC norm reduction and 24-row projection in one OpenMP team, avoiding a
 * fork/join on every mHC invocation during scalar decode. */
#ifndef GLM53F_MHC_FUSED
#define GLM53F_MHC_FUSED 0
#endif
#ifndef GLM53F_MHC_POST_FLOAT
#define GLM53F_MHC_POST_FLOAT 0
#endif

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

/* mHC has only 24 projection rows. Keep all rows parallel while reusing each
 * BF16 weight vector across the four verification positions. */
static inline void glm53f_mhc_mv_batch4(
        float *out, const uint16_t *weight, const float *input, int tokens) {
#pragma omp parallel for schedule(static)
    for (int row = 0; row < GLM53F_MHC_MIX; ++row) {
        svfloat32_t a0 = svdup_f32(0.0f), a1 = svdup_f32(0.0f);
        svfloat32_t a2 = svdup_f32(0.0f), a3 = svdup_f32(0.0f);
        const uint16_t *w = weight + (size_t)row * GLM53F_MHC_FLAT;
        int vl = (int)svcntw();
        for (int i = 0; i < GLM53F_MHC_FLAT; i += vl) {
            svbool_t pg = svwhilelt_b32(i, GLM53F_MHC_FLAT);
            svuint32_t bits = svlsl_n_u32_x(pg, svld1uh_u32(pg, w + i), 16);
            svfloat32_t wf = svreinterpret_f32_u32(bits);
            a0 = svmla_x(pg, a0, wf, svld1(pg, input + i));
            if (tokens > 1) a1 = svmla_x(pg, a1, wf,
                svld1(pg, input + GLM53F_MHC_FLAT + i));
            if (tokens > 2) a2 = svmla_x(pg, a2, wf,
                svld1(pg, input + 2 * GLM53F_MHC_FLAT + i));
            if (tokens > 3) a3 = svmla_x(pg, a3, wf,
                svld1(pg, input + 3 * GLM53F_MHC_FLAT + i));
        }
        svbool_t all = svptrue_b32();
        out[row] = svaddv_f32(all, a0);
        if (tokens > 1) out[GLM53F_MHC_MIX + row] = svaddv_f32(all, a1);
        if (tokens > 2) out[2 * GLM53F_MHC_MIX + row] = svaddv_f32(all, a2);
        if (tokens > 3) out[3 * GLM53F_MHC_MIX + row] = svaddv_f32(all, a3);
    }
}

static inline void glm53f_mhc_pre_sve(
        glm53f_mhc_scratch *scratch, const float *streams,
        const glm53f_mhc_site *site, const uint16_t *norm) {
    double sumsq = 0.0;
    float logits[GLM53F_MHC_MIX];
#if GLM53F_MHC_FUSED
#pragma omp parallel shared(sumsq,logits)
    {
#pragma omp for reduction(+:sumsq) schedule(static)
        for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
            sumsq += (double)streams[i] * streams[i];
#pragma omp barrier
#pragma omp single
        { sumsq = (double)(1.0f / sqrtf((float)(sumsq / GLM53F_MHC_FLAT) + 1e-5f)); }
#pragma omp barrier
#pragma omp for schedule(static)
        for (int m = 0; m < GLM53F_MHC_MIX; ++m)
            logits[m] = glm53f_mhc_dot_bf16_sve(
                site->fn + (size_t)m * GLM53F_MHC_FLAT, streams,
                GLM53F_MHC_FLAT) * (float)sumsq;
    }
#else
#pragma omp parallel for reduction(+:sumsq)
    for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
        sumsq += (double)streams[i] * streams[i];
    float inv = 1.0f / sqrtf((float)(sumsq / GLM53F_MHC_FLAT) + 1e-5f);
#pragma omp parallel for schedule(static)
    for (int m = 0; m < GLM53F_MHC_MIX; ++m)
        logits[m] = glm53f_mhc_dot_bf16_sve(
            site->fn + (size_t)m * GLM53F_MHC_FLAT, streams,
            GLM53F_MHC_FLAT) * inv;
#endif
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

static inline void glm53f_mhc_pre_batch_sve(
        glm53f_mhc_scratch *scratch, const float *streams,
        const glm53f_mhc_site *site, const uint16_t *norm, int tokens,
        size_t scratch_stride, float *normalized) {
    float inv[5],logits[5*GLM53F_MHC_MIX];
    for(int t=0;t<tokens;t++){double sumsq=0;const float*s=streams+(size_t)t*GLM53F_MHC_FLAT;
#pragma omp parallel for reduction(+:sumsq)
        for(int i=0;i<GLM53F_MHC_FLAT;i++)sumsq+=(double)s[i]*s[i];inv[t]=1.0f/sqrtf((float)(sumsq/GLM53F_MHC_FLAT)+1e-5f);}
    int n=tokens<5?tokens:4;glm53f_mhc_mv_batch4(logits,site->fn,streams,n);if(tokens==5){const float*s=streams+(size_t)4*GLM53F_MHC_FLAT;float*z=logits+(size_t)4*GLM53F_MHC_MIX;
#pragma omp parallel for schedule(static)
        for(int m=0;m<GLM53F_MHC_MIX;m++)z[m]=glm53f_mhc_dot_bf16_sve(site->fn+(size_t)m*GLM53F_MHC_FLAT,s,GLM53F_MHC_FLAT);}
    for(int t=0;t<tokens;t++){glm53f_mhc_scratch*q=(glm53f_mhc_scratch*)((unsigned char*)scratch+(size_t)t*scratch_stride);const float*s=streams+(size_t)t*GLM53F_MHC_FLAT;float*z=logits+(size_t)t*GLM53F_MHC_MIX;for(int m=0;m<GLM53F_MHC_MIX;m++)z[m]*=inv[t];for(int k=0;k<GLM53F_MHC_STREAMS;k++){z[k]=glm53f_sigmoid(z[k]*site->scale[0]+site->base[k])+1e-6f;q->post[k]=2.0f*glm53f_sigmoid(z[GLM53F_MHC_STREAMS+k]*site->scale[1]+site->base[GLM53F_MHC_STREAMS+k]);}for(int m=0;m<GLM53F_MHC_STREAMS*GLM53F_MHC_STREAMS;m++)q->combine[m]=z[2*GLM53F_MHC_STREAMS+m]*site->scale[2]+site->base[2*GLM53F_MHC_STREAMS+m];glm53f_mhc_sinkhorn(q->combine,GLM53F_MHC_STREAMS,20,1e-6f);
#pragma omp parallel for schedule(static)
        for(int d=0;d<GLM53F_MHC_WIDTH;d++){float v=0;for(int k=0;k<GLM53F_MHC_STREAMS;k++)v+=z[k]*s[(size_t)k*GLM53F_MHC_WIDTH+d];q->collapsed[d]=v;}memcpy(q->residual,s,sizeof(q->residual));glm53f_rmsnorm_bf16(q->normalized,q->collapsed,norm,GLM53F_MHC_WIDTH,1e-5f);memcpy(normalized+(size_t)t*GLM53F_MHC_WIDTH,q->normalized,GLM53F_MHC_WIDTH*4);}
}

static inline void glm53f_mhc_post_sve(
        float *streams, const float *sublayer, const glm53f_mhc_scratch *scratch) {
#pragma omp parallel for collapse(2) schedule(static)
        for (int k = 0; k < GLM53F_MHC_STREAMS; ++k)
        for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
#if GLM53F_MHC_POST_FLOAT
            float v = scratch->post[k] * sublayer[d];
            for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                v += scratch->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                     scratch->residual[(size_t)j * GLM53F_MHC_WIDTH + d];
#else
            double v = (double)scratch->post[k] * sublayer[d];
            for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                v += (double)scratch->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                     scratch->residual[(size_t)j * GLM53F_MHC_WIDTH + d];
#endif
            streams[(size_t)k * GLM53F_MHC_WIDTH + d] = (float)v;
        }
}

/* Batch-only post mix.  Verification positions are independent at this
 * point, so distribute the (otherwise scalar) mHC post over token/head
 * pairs.  The inner accumulation order is identical to glm53f_mhc_post,
 * preserving target-token decisions while removing four serial calls. */
static inline void glm53f_mhc_post_batch_sve(
        float *streams, const float *sublayer,
        const glm53f_mhc_scratch *scratch, int tokens, size_t stride) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < tokens; ++t)
        for (int k = 0; k < GLM53F_MHC_STREAMS; ++k) {
            float *dst = streams + (size_t)t * GLM53F_MHC_FLAT +
                         (size_t)k * GLM53F_MHC_WIDTH;
            const float *res = (const float *)((const unsigned char *)scratch +
                              (size_t)t * stride + offsetof(glm53f_mhc_scratch, residual));
            const glm53f_mhc_scratch *s = (const glm53f_mhc_scratch *)
                              ((const unsigned char *)scratch + (size_t)t * stride);
            for (int d = 0; d < GLM53F_MHC_WIDTH; ++d) {
#if GLM53F_MHC_POST_FLOAT
                float v = s->post[k] * sublayer[(size_t)t * GLM53F_MHC_WIDTH + d];
                for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                    v += s->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                         res[(size_t)j * GLM53F_MHC_WIDTH + d];
#else
                double v = (double)s->post[k] * sublayer[(size_t)t * GLM53F_MHC_WIDTH + d];
                for (int j = 0; j < GLM53F_MHC_STREAMS; ++j)
                    v += (double)s->combine[(size_t)j * GLM53F_MHC_STREAMS + k] *
                         res[(size_t)j * GLM53F_MHC_WIDTH + d];
#endif
                dst[d] = (float)v;
            }
        }
}

#endif
