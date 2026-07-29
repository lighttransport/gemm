#ifndef K3_MOE_H
#define K3_MOE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "k3_kernels.h"
#include "ggml_dequant.h"

typedef struct {
    const uint8_t *packed;
    const uint8_t *scale;
    int rows;
    int cols;
} k3_mxfp4_matrix;

#define K3_MOE_MAX_BATCH 256
#define K3_MXFP4_TILE_K 512

/* Slot-major dispatch buffers have capacity=batch for each local expert. */
static inline int k3_moe_build_dispatch(const int *route_experts,
                                        const float *route_weights,
                                        int batch, int topk,
                                        const int *local_expert_ids, int nlocal,
                                        int *counts, int *token_ids,
                                        float *token_weights) {
    for (int s = 0; s < nlocal; ++s) counts[s] = 0;
    int assignments = 0;
    for (int t = 0; t < batch; ++t) {
        for (int k = 0; k < topk; ++k) {
            int expert = route_experts[t * topk + k], slot = -1;
            for (int s = 0; s < nlocal; ++s)
                if (local_expert_ids[s] == expert) { slot = s; break; }
            if (slot < 0) continue;
            int p = counts[slot]++;
            if (p >= batch) return -1;
            token_ids[slot * batch + p] = t;
            token_weights[slot * batch + p] = route_weights[t * topk + k];
            ++assignments;
        }
    }
    return assignments;
}

static inline void k3_moe_gather(float *dst, const float *src,
                                 const int *token_ids, int count, int width) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < count; ++p)
        memcpy(dst + (size_t)p * width, src + (size_t)token_ids[p] * width,
               (size_t)width * sizeof(float));
}

static inline void k3_moe_scatter_add(float *dst, const float *src,
                                      const int *token_ids, const float *weights,
                                      int count, int width) {
    for (int p = 0; p < count; ++p) {
        float *d = dst + (size_t)token_ids[p] * width;
        const float *s = src + (size_t)p * width;
        float w = weights[p];
#if defined(__ARM_FEATURE_SVE)
        int vl = (int)svcntw();
        for (int i = 0; i < width; i += vl) {
            svbool_t pg = svwhilelt_b32(i, width);
            svfloat32_t v = svmla_n_f32_x(pg, svld1(pg, d + i), svld1(pg, s + i), w);
            svst1(pg, d + i, v);
        }
#else
        for (int i = 0; i < width; ++i) d[i] += w * s[i];
#endif
    }
}

static inline void k3_mxfp4_group_svtbl(float *y, int ystride,
                                         const uint8_t *w, const uint8_t *s,
                                         const float *x, int xstride,
                                         int batch, int k) {
    size_t rb = (size_t)k / 2, sb = (size_t)k / 32;
    const uint8_t *w0=w,*w1=w+rb,*w2=w+2*rb,*w3=w+3*rb;
    const uint8_t *w4=w+4*rb,*w5=w+5*rb,*w6=w+6*rb,*w7=w+7*rb;
    const uint8_t *s0=s,*s1=s+sb,*s2=s+2*sb,*s3=s+3*sb;
    const uint8_t *s4=s+4*sb,*s5=s+5*sb,*s6=s+6*sb,*s7=s+7*sb;
    int m = 0;
    for (; m + 1 < batch; m += 2)
        matvec_mxfp4_8row_2x(y+(size_t)m*ystride,y+(size_t)(m+1)*ystride,
                             w0,w1,w2,w3,w4,w5,w6,w7,
                             s0,s1,s2,s3,s4,s5,s6,s7,
                             x+(size_t)m*xstride,x+(size_t)(m+1)*xstride,k);
    for (; m < batch; ++m)
        matvec_mxfp4_8row(y+(size_t)m*ystride,w0,w1,w2,w3,w4,w5,w6,w7,
                          s0,s1,s2,s3,s4,s5,s6,s7,x+(size_t)m*xstride,k);
}

#if defined(__ARM_FEATURE_SVE)
/* Dequantize each 8-row MXFP4 tile once to an L1-resident BF16 pair-vector
 * panel, then reuse it for three tokens at a time.  This is lossless in weight
 * conversion; only the K-tile reduction order differs from the decode kernel. */
static inline void k3_mxfp4_group_tile(float *y, int ystride,
                                       const uint8_t *w, const uint8_t *s,
                                       const float *x, int xstride,
                                       int batch, int k) {
    const int tk = K3_MXFP4_TILE_K;
    uint16_t pv[4 * 2 * K3_MXFP4_TILE_K] __attribute__((aligned(256)));
    float acc[K3_MOE_MAX_BATCH][8];
    memset(acc, 0, (size_t)batch * 8 * sizeof(float));
    size_t rb = (size_t)k / 2, sb = (size_t)k / 32;
    svbool_t pg = svptrue_b32(), ph = svptrue_b16();
    svfloat32_t kv = svld1(pg, ds4f_kvalues_mxfp4_f32);
    int vl = (int)svcntw();
    for (int k0 = 0; k0 < k; k0 += tk) {
        int klen = k - k0 < tk ? k - k0 : tk;
        for (int pr = 0; pr < 4; ++pr) {
            uint16_t *pb = pv + (size_t)pr * 2 * tk;
            const uint8_t *wa = w + (size_t)(2 * pr) * rb;
            const uint8_t *wb = wa + rb;
            const uint8_t *sa = s + (size_t)(2 * pr) * sb;
            const uint8_t *ssb = sa + sb;
            for (int c = 0; c < klen; c += vl) {
                int col = k0 + c, blk = col >> 5;
                int high = (col & 16) != 0;
                svuint32_t ra = svld1ub_u32(pg, wa + (size_t)blk * 16);
                svuint32_t rbv = svld1ub_u32(pg, wb + (size_t)blk * 16);
                svuint32_t na = high
                    ? svand_n_u32_x(pg, svlsr_n_u32_x(pg, ra, 4), 15)
                    : svand_n_u32_x(pg, ra, 15);
                svuint32_t nb = high
                    ? svand_n_u32_x(pg, svlsr_n_u32_x(pg, rbv, 4), 15)
                    : svand_n_u32_x(pg, rbv, 15);
                svfloat32_t fa = svmul_n_f32_x(pg, svtbl_f32(kv, na),
                                               ggml_e8m0_to_fp32(sa[blk]));
                svfloat32_t fb = svmul_n_f32_x(pg, svtbl_f32(kv, nb),
                                               ggml_e8m0_to_fp32(ssb[blk]));
                svuint16_t a16 = svreinterpret_u16_u32(svlsr_n_u32_x(
                    pg, svreinterpret_u32_f32(fa), 16));
                svuint16_t b16 = svreinterpret_u16_u32(svlsr_n_u32_x(
                    pg, svreinterpret_u32_f32(fb), 16));
                svuint16_t ca = svuzp1_u16(a16, a16);
                svuint16_t cb = svuzp1_u16(b16, b16);
                svst1_u16(ph, pb + 2 * c, svzip1_u16(ca, cb));
            }
        }
        const uint16_t *p0 = pv, *p2 = pv + 2 * tk;
        const uint16_t *p4 = pv + 4 * tk, *p6 = pv + 6 * tk;
        int m = 0;
        for (; m + 2 < batch; m += 3)
            matvec_bf16_8x3_pv_acc(acc[m], acc[m + 1], acc[m + 2],
                                    p0, p2, p4, p6,
                                    x + (size_t)m * xstride + k0,
                                    x + (size_t)(m + 1) * xstride + k0,
                                    x + (size_t)(m + 2) * xstride + k0, klen);
        for (; m < batch; ++m)
            matvec_bf16_8row_pv_acc(acc[m], p0, p2, p4, p6,
                                     x + (size_t)m * xstride + k0, klen);
    }
    for (int m = 0; m < batch; ++m)
        memcpy(y + (size_t)m * ystride, acc[m], 8 * sizeof(float));
}
#endif

static inline void k3_mxfp4_group_batch(float *y, int ystride,
                                        const uint8_t *w, const uint8_t *s,
                                        const float *x, int xstride,
                                        int batch, int k, int tile_threshold) {
#if defined(__ARM_FEATURE_SVE)
    if (tile_threshold > 0 && batch >= tile_threshold &&
        batch <= K3_MOE_MAX_BATCH) {
        k3_mxfp4_group_tile(y, ystride, w, s, x, xstride, batch, k);
        return;
    }
#else
    (void)tile_threshold;
#endif
    k3_mxfp4_group_svtbl(y, ystride, w, s, x, xstride, batch, k);
}

static inline void k3_mxfp4_gemm_mode(float *y, const k3_mxfp4_matrix *matrix,
                                      const float *x, int batch, int threads,
                                      int tile_threshold) {
    int groups = matrix->rows / 8, k = matrix->cols;
    size_t wr = (size_t)k / 2, sr = (size_t)k / 32;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for (int g = 0; g < groups; ++g) {
        int r = g * 8;
        k3_mxfp4_group_batch(y+r,matrix->rows,
                             matrix->packed+(size_t)r*wr,
                             matrix->scale+(size_t)r*sr,
                             x,k,batch,k,tile_threshold);
    }
}

/* Fuse w1+w3 into one workshare; both consume the same gathered token tile. */
static inline void k3_mxfp4_gemm2_mode(float *y1, const k3_mxfp4_matrix *m1,
                                  float *y3, const k3_mxfp4_matrix *m3,
                                  const float *x, int batch, int threads,
                                  int tile_threshold) {
    int groups = m1->rows / 8, k = m1->cols;
    size_t wr = (size_t)k/2, sr = (size_t)k/32;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for (int task = 0; task < groups * 2; ++task) {
        int which = task / groups, r = (task % groups) * 8;
        const k3_mxfp4_matrix *m = which ? m3 : m1;
        float *y = which ? y3 : y1;
        k3_mxfp4_group_batch(y+r,m->rows,m->packed+(size_t)r*wr,
                             m->scale+(size_t)r*sr,x,k,batch,k,tile_threshold);
    }
}

static inline void k3_expert_forward_mxfp4_mode(float *out,
                                           const k3_mxfp4_matrix *w1,
                                           const k3_mxfp4_matrix *w2,
                                           const k3_mxfp4_matrix *w3,
                                           const float *x, int batch,
                                           float *gate, float *up,
                                           int threads, int tile_threshold) {
    k3_mxfp4_gemm2_mode(gate,w1,up,w3,x,batch,threads,tile_threshold);
    int n = batch * w1->rows;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i)
        gate[i] = 4.0f*tanhf(gate[i]*0.25f)*k3_sigmoidf(gate[i])
                *25.0f*tanhf(up[i]*0.04f);
    k3_mxfp4_gemm_mode(out,w2,gate,batch,threads,tile_threshold);
}

static inline void k3_expert_forward_mxfp4(float *out,
                                           const k3_mxfp4_matrix *w1,
                                           const k3_mxfp4_matrix *w2,
                                           const k3_mxfp4_matrix *w3,
                                           const float *x, int batch,
                                           float *gate, float *up,
                                           int threads) {
    k3_expert_forward_mxfp4_mode(out,w1,w2,w3,x,batch,gate,up,threads,8);
}

static inline void k3_moe_forward_sparse_partitioned(
        float *dst, const k3_mxfp4_matrix *w1,
        const k3_mxfp4_matrix *w2, const k3_mxfp4_matrix *w3,
        int nlocal, const int *counts, const int *token_ids,
        const float *token_weights, const float *gathered, int batch,
        float *gate, float *up, float *expert_out,
        int threads, int tile_threshold, const int *active, int nactive) {
    (void)nlocal;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel
    {
        int tid=omp_get_thread_num(),nth=omp_get_num_threads();
        int groups=nactive<4?nactive:4;
        int group=tid*groups/nth;
        int first=group*nth/groups, last=(group+1)*nth/groups;
        int lane=tid-first, group_threads=last-first;
        for(int ai=group;ai<nactive;ai+=groups){
            int e=active[ai],m=counts[e];
            for(int which=0;which<2;++which){
                const k3_mxfp4_matrix*mat=which?&w3[e]:&w1[e];
                float*y=(which?up:gate)+(size_t)e*batch*K3_EXPERT_INTER;
                const float*x=gathered+(size_t)e*batch*K3_LATENT;
                size_t wr=(size_t)mat->cols/2,sr=(size_t)mat->cols/32;
                for(int r=lane*8;r<mat->rows;r+=group_threads*8)
                    k3_mxfp4_group_batch(y+r,K3_EXPERT_INTER,
                        mat->packed+(size_t)r*wr,mat->scale+(size_t)r*sr,
                        x,K3_LATENT,m,mat->cols,tile_threshold);
            }
        }
#pragma omp barrier
        for(int ai=group;ai<nactive;ai+=groups){
            int e=active[ai],n=counts[e]*K3_EXPERT_INTER;
            float*g=gate+(size_t)e*batch*K3_EXPERT_INTER;
            const float*u=up+(size_t)e*batch*K3_EXPERT_INTER;
            for(int i=lane;i<n;i+=group_threads)
                g[i]=4.0f*tanhf(g[i]*.25f)*k3_sigmoidf(g[i])
                    *25.0f*tanhf(u[i]*.04f);
        }
#pragma omp barrier
        for(int ai=group;ai<nactive;ai+=groups){
            int e=active[ai],m=counts[e];const k3_mxfp4_matrix*mat=&w2[e];
            float*y=expert_out+(size_t)e*batch*K3_LATENT;
            const float*x=gate+(size_t)e*batch*K3_EXPERT_INTER;
            size_t wr=(size_t)mat->cols/2,sr=(size_t)mat->cols/32;
            for(int r=lane*8;r<mat->rows;r+=group_threads*8)
                k3_mxfp4_group_batch(y+r,K3_LATENT,
                    mat->packed+(size_t)r*wr,mat->scale+(size_t)r*sr,
                    x,K3_EXPERT_INTER,m,mat->cols,tile_threshold);
        }
    }
#else
    (void)nlocal;
    for(int ai=0;ai<nactive;++ai){int e=active[ai];k3_expert_forward_mxfp4_mode(
        expert_out+(size_t)e*batch*K3_LATENT,&w1[e],&w2[e],&w3[e],
        gathered+(size_t)e*batch*K3_LATENT,counts[e],
        gate+(size_t)e*batch*K3_EXPERT_INTER,
        up+(size_t)e*batch*K3_EXPERT_INTER,1,tile_threshold);}
#endif
    memset(dst,0,(size_t)batch*K3_LATENT*sizeof(float));
    for(int ai=0;ai<nactive;++ai){int e=active[ai];k3_moe_scatter_add(dst,
        expert_out+(size_t)e*batch*K3_LATENT,token_ids+e*batch,
        token_weights+e*batch,counts[e],K3_LATENT);}
}

/* Execute all local expert buckets with two global workshares.  Buffers are
 * expert-major with batch-capacity slots: [nlocal][batch][width].  Sharing the
 * OpenMP teams is important for sparse decode, where most non-empty experts
 * carry one token and per-expert parallel-region setup otherwise dominates. */
static inline void k3_moe_forward_local_mxfp4(
        float *dst, const k3_mxfp4_matrix *w1,
        const k3_mxfp4_matrix *w2, const k3_mxfp4_matrix *w3,
        int nlocal, const int *counts, const int *token_ids,
        const float *token_weights, const float *gathered, int batch,
        float *gate, float *up, float *expert_out,
        int threads, int tile_threshold) {
    int active[nlocal],nactive=0;
    for(int e=0;e<nlocal;++e)if(counts[e])active[nactive++]=e;
    if(nactive==4&&threads>=16){
        k3_moe_forward_sparse_partitioned(dst,w1,w2,w3,nlocal,counts,
            token_ids,token_weights,gathered,batch,gate,up,expert_out,
            threads,tile_threshold,active,nactive);
        return;
    }
    const int g13 = w1[0].rows / 8, g2 = w2[0].rows / 8;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for (int task = 0; task < nlocal * g13 * 2; ++task) {
        int expert = task / (g13 * 2);
        int rem = task % (g13 * 2), which = rem / g13, r = (rem % g13) * 8;
        int m = counts[expert];
        if (!m) continue;
        const k3_mxfp4_matrix *matrix = which ? &w3[expert] : &w1[expert];
        float *ybase = (which ? up : gate)
            + (size_t)expert * batch * K3_EXPERT_INTER;
        const float *xbase = gathered
            + (size_t)expert * batch * K3_LATENT;
        size_t wr = (size_t)matrix->cols / 2, sr = (size_t)matrix->cols / 32;
        k3_mxfp4_group_batch(ybase + r, K3_EXPERT_INTER,
                             matrix->packed + (size_t)r * wr,
                             matrix->scale + (size_t)r * sr,
                             xbase, K3_LATENT, m, matrix->cols, tile_threshold);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < nlocal * batch * K3_EXPERT_INTER; ++i) {
        int expert = i / (batch * K3_EXPERT_INTER);
        int rem = i % (batch * K3_EXPERT_INTER);
        if (rem >= counts[expert] * K3_EXPERT_INTER) continue;
        gate[i] = 4.0f * tanhf(gate[i] * 0.25f) * k3_sigmoidf(gate[i])
                * 25.0f * tanhf(up[i] * 0.04f);
    }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int task = 0; task < nlocal * g2; ++task) {
        int expert = task / g2, r = (task % g2) * 8;
        int m = counts[expert];
        if (!m) continue;
        const k3_mxfp4_matrix *matrix = &w2[expert];
        float *ybase = expert_out + (size_t)expert * batch * K3_LATENT;
        const float *xbase = gate
            + (size_t)expert * batch * K3_EXPERT_INTER;
        size_t wr = (size_t)matrix->cols / 2, sr = (size_t)matrix->cols / 32;
        k3_mxfp4_group_batch(ybase + r, K3_LATENT,
                             matrix->packed + (size_t)r * wr,
                             matrix->scale + (size_t)r * sr,
                             xbase, K3_EXPERT_INTER, m, matrix->cols,
                             tile_threshold);
    }
    memset(dst, 0, (size_t)batch * K3_LATENT * sizeof(float));
    for (int expert = 0; expert < nlocal; ++expert)
        k3_moe_scatter_add(dst,
            expert_out + (size_t)expert * batch * K3_LATENT,
            token_ids + expert * batch, token_weights + expert * batch,
            counts[expert], K3_LATENT);
}

#endif
