#ifndef K3_MOE_H
#define K3_MOE_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "k3_kernels.h"
#include "k3_dense.h"
#include "ggml_dequant.h"

typedef struct {
    const uint8_t *packed;
    const uint8_t *scale;
    int rows;
    int cols;
    /* Optional per-CMG copies for the decode expert path.  Zero-initialized
     * aggregate literals keep the normal path unchanged. */
    const uint8_t *packed_cmg[4];
    const uint8_t *scale_cmg[4];
} k3_mxfp4_matrix;

static inline void k3_mxfp4_thread_weights(const k3_mxfp4_matrix *m,
                                            const uint8_t **packed,
                                            const uint8_t **scale) {
    *packed = m->packed;
    *scale = m->scale;
#if defined(_OPENMP)
    int cmg = omp_get_thread_num() / 12;
    if (cmg > 3) cmg = 3;
    if (m->packed_cmg[cmg]) {
        *packed = m->packed_cmg[cmg];
        *scale = m->scale_cmg[cmg];
    }
#endif
}

static inline void k3_mxfp4_group_batch(float *y, int ystride,
        const uint8_t *w, const uint8_t *s, const float *x, int xstride,
        int batch, int k, int tile_threshold);

static inline void k3_mxfp4_gemm_mode(float *y,
        const k3_mxfp4_matrix *matrix, const float *x, int batch,
        int threads, int tile_threshold);

static inline size_t k3_mxfp4_matrix_bytes(int rows, int cols) {
    return (size_t)rows * (cols / 2 + cols / 32);
}

/* Offline-style BF16 -> OCP MXFP4 conversion used by bounded real-weight
 * probes and the future stage path.  Each 32-value block uses the smallest
 * E8M0 power-of-two scale whose E2M1 maximum (12) covers the block. */
static inline void k3_mxfp4_quantize_bf16(uint8_t *packed, uint8_t *scale,
        const uint16_t *src, int rows, int cols) {
    int blocks=cols/32;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<rows;++r){const uint16_t *row=src+(size_t)r*cols;
        uint8_t *qr=packed+(size_t)r*cols/2,*sr=scale+(size_t)r*blocks;
        for(int b=0;b<blocks;++b){float amax=0;
            for(int j=0;j<32;++j)amax=fmaxf(amax,fabsf(bf16_to_f32_scalar(row[b*32+j])));
            int e=0;if(amax>0) e=(int)ceilf(log2f(amax/12.0f));
            if(e < -126)e=-126;if(e>127)e=127;sr[b]=(uint8_t)(e+127);
            float inv=1.0f/ldexpf(1.0f,e);
            for(int j=0;j<16;++j){uint8_t code[2];
                for(int h=0;h<2;++h){float v=bf16_to_f32_scalar(row[b*32+j+h*16])*inv;
                    int sign=v<0, best=0;float av=fabsf(v),err=av;
                    for(int q=1;q<8;++q){float d=fabsf(av-ds4f_kvalues_mxfp4_f32[q]);if(d<err){err=d;best=q;}}
                    code[h]=(uint8_t)(best|(sign&&best?8:0));}
                qr[b*16+j]=(uint8_t)(code[0]|(code[1]<<4));}}
    }
}

#define K3_EXPERT_TP_BLOCK 32
#define K3_MOE_REDUCE_FLOATS (K3_LATENT + K3_HIDDEN)
/* Decode-only routed-down association: apply the expert's route weight to the
 * two activation vectors once per 32-channel block instead of multiplying it
 * into each of eight row scales.  The operation is algebraically identical;
 * rounding differs because the multiply moves before the dot product. */
static int k3_tp_scale_activation = 0;

/* At TP=96 every rank owns exactly one native MXFP4 scale group from every
 * expert.  Smaller jobs emulate the same architecture with a contiguous
 * multiple of 32 intermediate channels per rank. */
static inline int k3_expert_tp_layout_valid(const k3_mxfp4_matrix *w1,
                                             const k3_mxfp4_matrix *w2,
                                             const k3_mxfp4_matrix *w3) {
    return w1 && w2 && w3 && w1->rows == w3->rows &&
        w1->rows > 0 && !(w1->rows % K3_EXPERT_TP_BLOCK) &&
        w1->cols == K3_LATENT && w3->cols == K3_LATENT &&
        w2->rows == K3_LATENT && w2->cols == w1->rows;
}

static inline int k3_expert_tp_forward_mxfp4(
        float *partial_latent, const k3_mxfp4_matrix *w1,
        const k3_mxfp4_matrix *w2, const k3_mxfp4_matrix *w3,
        const float *latent, int batch, float *gate, float *up,
        int threads, int tile_threshold);

/* Local contributions are concatenated so routed latent and shared hidden use
 * one network synchronization.  The caller invokes exactly one sum-allreduce
 * over K3_MOE_REDUCE_FLOATS floats, then finishes routed_up locally. */
static inline void k3_moe_pack_reduce(float *reduce,
                                      const float *routed_latent_partial,
                                      const float *shared_hidden_partial) {
    memcpy(reduce, routed_latent_partial, K3_LATENT * sizeof(float));
    memcpy(reduce + K3_LATENT, shared_hidden_partial,
           K3_HIDDEN * sizeof(float));
}

static inline void k3_moe_finish_reduce_q8(
        float *hidden_out, float *norm_scratch, int8_t *q_scratch,
        const float *reduced, const float *routed_norm_weight,
        const k3_q8_matrix *replicated_routed_up, float eps, int threads) {
    k3_rmsnorm_sve(norm_scratch, reduced, routed_norm_weight, K3_LATENT, eps);
    const float *shared = reduced + K3_LATENT;
    k3_matvec_q8_bias(hidden_out, replicated_routed_up, norm_scratch,
                      q_scratch, shared, threads);
}

static inline void k3_moe_finish_reduce_bf16(
        float *hidden_out, float *norm_scratch, const float *reduced,
        const float *routed_norm_weight,
        const k3_bf16_matrix *replicated_routed_up, float eps, int threads) {
    k3_rmsnorm_sve(norm_scratch, reduced, routed_norm_weight, K3_LATENT, eps);
    int groups=K3_HIDDEN/8;
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#else
    (void)threads;
#endif
    for(int g=0;g<groups;++g){int r=g*8;const uint16_t*w=
        replicated_routed_up->weight+(size_t)r*K3_LATENT;
        matvec_bf16_8row(hidden_out+r,w,w+K3_LATENT,w+2*K3_LATENT,
            w+3*K3_LATENT,w+4*K3_LATENT,w+5*K3_LATENT,
            w+6*K3_LATENT,w+7*K3_LATENT,norm_scratch,K3_LATENT);
        for(int j=0;j<8;++j)hidden_out[r+j]+=reduced[K3_LATENT+r+j];}
}

static inline void k3_moe_finish_reduce_q8p16(float*hidden_out,
        float*norm_scratch,int8_t*q_scratch,const float*reduced,
        const float*routed_norm_weight,const int8_t*routed_up_packed,
        const float*routed_up_scale,float eps,int threads){
    k3_rmsnorm_sve(norm_scratch,reduced,routed_norm_weight,K3_LATENT,eps);
    k3_matvec_q8p16_bias(hidden_out,routed_up_packed,routed_up_scale,
        K3_HIDDEN,K3_LATENT,norm_scratch,q_scratch,reduced+K3_LATENT,threads);
}

static inline void k3_moe_finish_reduce_mxfp4(float *hidden_out,
        float *norm_scratch, const float *reduced,
        const float *routed_norm_weight,
        const k3_mxfp4_matrix *replicated_routed_up, float eps, int threads) {
    k3_rmsnorm_sve(norm_scratch,reduced,routed_norm_weight,K3_LATENT,eps);
    k3_mxfp4_gemm_mode(hidden_out,replicated_routed_up,norm_scratch,1,threads,0);
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<K3_HIDDEN;++i)hidden_out[i]+=reduced[K3_LATENT+i];
}

static inline void k3_moe_finish_reduce_q8w16(float *hidden_out,
        float *norm_scratch, const float *reduced,
        const float *routed_norm_weight,
        const k3_q8pv_matrix *replicated_routed_up, float eps, int threads) {
    k3_rmsnorm_sve(norm_scratch,reduced,routed_norm_weight,K3_LATENT,eps);
    /* More workers do not raise this bandwidth-bound projection on A64FX;
     * leave application cores available for transport progress. */
    if (threads > K3_Q8W16_UP_THREADS)
        threads = K3_Q8W16_UP_THREADS;
    k3_matvec_q8pv16_f32_bias(hidden_out,replicated_routed_up,norm_scratch,
                              reduced+K3_LATENT,threads);
}

#ifndef K3_MOE_MAX_BATCH
#define K3_MOE_MAX_BATCH 256
#endif
#ifndef K3_MXFP4_TILE_K_SMALL
#define K3_MXFP4_TILE_K_SMALL 512
#endif
#ifndef K3_MXFP4_TILE_K_LARGE
#define K3_MXFP4_TILE_K_LARGE 3072
#endif
#ifndef K3_MXFP4_PREFETCH_BLOCKS
#define K3_MXFP4_PREFETCH_BLOCKS 16
#endif
#ifndef K3_SITU_FEXPA
#define K3_SITU_FEXPA 1
#endif
#ifndef K3_TP_FUSED_DOWN32
#define K3_TP_FUSED_DOWN32 1
#endif

static inline void k3_moe_situ(float *gate, const float *up, int n,
                               int threads) {
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#endif
#if defined(__ARM_FEATURE_SVE) && K3_SITU_FEXPA
    int vl=(int)svcntw();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<n;i+=vl)
        k3_situ_fast_sve(gate+i,gate+i,up+i,n-i<vl?n-i:vl);
#else
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<n;++i)
        gate[i]=4.0f*tanhf(gate[i]*.25f)*k3_sigmoidf(gate[i])
                *25.0f*tanhf(up[i]*.04f);
#endif
}

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

#if defined(__ARM_FEATURE_SVE)
/* K3 decode-specialized copy of the native MXFP4 kernel.  Keeping it local
 * permits A64FX scheduling/prefetch experiments without changing every DS4F
 * backend that consumes the shared reference kernel. */
/* e8m0 -> f32 through a 256-entry table.  ggml_e8m0_to_fp32() computes the same
 * bits with a shift, but lands them in a GPR, so svmla_n then pays a GPR->FPR
 * domain crossing 8 times per 32-value block -- the A64FX trap already recorded
 * for svdupq-from-GPR.  Indexing a table puts the scale straight in an FP
 * register.  Bit-identical by construction (entry i is (uint32)i<<23), measured
 * 20.26 -> 26.98 Gmac/s single-thread, matching the no-scale upper bound. */
typedef union { uint32_t u; float f; } k3_e8m0_cvt;
#define K3_E8(i)    {(uint32_t)(i) << 23}
#define K3_E8_4(i)  K3_E8(i),K3_E8((i)+1),K3_E8((i)+2),K3_E8((i)+3)
#define K3_E8_16(i) K3_E8_4(i),K3_E8_4((i)+4),K3_E8_4((i)+8),K3_E8_4((i)+12)
#define K3_E8_64(i) K3_E8_16(i),K3_E8_16((i)+16),K3_E8_16((i)+32),K3_E8_16((i)+48)
static const k3_e8m0_cvt k3_e8m0_tab[256] = {
    K3_E8_64(0), K3_E8_64(64), K3_E8_64(128), K3_E8_64(192)
};
#undef K3_E8
#undef K3_E8_4
#undef K3_E8_16
#undef K3_E8_64

static inline void k3_matvec_mxfp4_8row(float *dst,
        const uint8_t *w0,const uint8_t *w1,const uint8_t *w2,const uint8_t *w3,
        const uint8_t *w4,const uint8_t *w5,const uint8_t *w6,const uint8_t *w7,
        const uint8_t *s0,const uint8_t *s1,const uint8_t *s2,const uint8_t *s3,
        const uint8_t *s4,const uint8_t *s5,const uint8_t *s6,const uint8_t *s7,
        const float*x,int k){
    svbool_t pg=svptrue_b32();svfloat32_t kv=svld1(pg,ds4f_kvalues_mxfp4_f32);
    svfloat32_t a0=svdup_f32(0),a1=svdup_f32(0),a2=svdup_f32(0),a3=svdup_f32(0);
    svfloat32_t a4=svdup_f32(0),a5=svdup_f32(0),a6=svdup_f32(0),a7=svdup_f32(0);
    int nb=k/32;
#pragma clang loop unroll_count(2)
    for(int b=0;b<nb;++b){
#if K3_MXFP4_PREFETCH_BLOCKS > 0
        int pb=b+K3_MXFP4_PREFETCH_BLOCKS;
        if(pb<nb){__builtin_prefetch(w0+(size_t)pb*16,0,2);__builtin_prefetch(w1+(size_t)pb*16,0,2);
            __builtin_prefetch(w2+(size_t)pb*16,0,2);__builtin_prefetch(w3+(size_t)pb*16,0,2);
            __builtin_prefetch(w4+(size_t)pb*16,0,2);__builtin_prefetch(w5+(size_t)pb*16,0,2);
            __builtin_prefetch(w6+(size_t)pb*16,0,2);__builtin_prefetch(w7+(size_t)pb*16,0,2);}
#endif
        svfloat32_t xl=svld1(pg,x+(size_t)b*32),xh=svld1(pg,x+(size_t)b*32+16);
#define K3_MXROW(W,S,A) do{svuint32_t z=svld1ub_u32(pg,(W)+(size_t)b*16); \
        svuint32_t lo=svand_n_u32_x(pg,z,15),hi=svand_n_u32_x(pg,svlsr_n_u32_x(pg,z,4),15); \
        svfloat32_t p=svmul_x(pg,svtbl_f32(kv,lo),xl);p=svmla_x(pg,p,svtbl_f32(kv,hi),xh); \
        (A)=svmla_n_f32_x(pg,(A),p,k3_e8m0_tab[(S)[b]].f);}while(0)
        K3_MXROW(w0,s0,a0);K3_MXROW(w1,s1,a1);K3_MXROW(w2,s2,a2);K3_MXROW(w3,s3,a3);
        K3_MXROW(w4,s4,a4);K3_MXROW(w5,s5,a5);K3_MXROW(w6,s6,a6);K3_MXROW(w7,s7,a7);
#undef K3_MXROW
    }
    dst[0]=svaddv(pg,a0);dst[1]=svaddv(pg,a1);dst[2]=svaddv(pg,a2);dst[3]=svaddv(pg,a3);
    dst[4]=svaddv(pg,a4);dst[5]=svaddv(pg,a5);dst[6]=svaddv(pg,a6);dst[7]=svaddv(pg,a7);
}

#endif

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
#if defined(__ARM_FEATURE_SVE)
        k3_matvec_mxfp4_8row(y+(size_t)m*ystride,w0,w1,w2,w3,w4,w5,w6,w7,
                          s0,s1,s2,s3,s4,s5,s6,s7,x+(size_t)m*xstride,k);
#else
        matvec_mxfp4_8row(y+(size_t)m*ystride,w0,w1,w2,w3,w4,w5,w6,w7,
                          s0,s1,s2,s3,s4,s5,s6,s7,x+(size_t)m*xstride,k);
#endif
}

/* Small routed buckets do not repay a full latent gather. Keep the same 2-token
 * microkernel but take token rows through the compact dispatch index. */
static inline void k3_mxfp4_group_svtbl_indexed(float *y,int ystride,
        const uint8_t*w,const uint8_t*s,const float*x,int xstride,
        const int*token_ids,int batch,int k){
    size_t rb=(size_t)k/2,sb=(size_t)k/32;
    const uint8_t*w0=w,*w1=w+rb,*w2=w+2*rb,*w3=w+3*rb;
    const uint8_t*w4=w+4*rb,*w5=w+5*rb,*w6=w+6*rb,*w7=w+7*rb;
    const uint8_t*s0=s,*s1=s+sb,*s2=s+2*sb,*s3=s+3*sb;
    const uint8_t*s4=s+4*sb,*s5=s+5*sb,*s6=s+6*sb,*s7=s+7*sb;int m=0;
    for(;m+1<batch;m+=2)matvec_mxfp4_8row_2x(y+(size_t)m*ystride,
        y+(size_t)(m+1)*ystride,w0,w1,w2,w3,w4,w5,w6,w7,
        s0,s1,s2,s3,s4,s5,s6,s7,x+(size_t)token_ids[m]*xstride,
        x+(size_t)token_ids[m+1]*xstride,k);
    for(;m<batch;++m)
#if defined(__ARM_FEATURE_SVE)
        k3_matvec_mxfp4_8row(y+(size_t)m*ystride,w0,w1,w2,w3,w4,w5,w6,w7,
            s0,s1,s2,s3,s4,s5,s6,s7,x+(size_t)token_ids[m]*xstride,k);
#else
        matvec_mxfp4_8row(y+(size_t)m*ystride,w0,w1,w2,w3,w4,w5,w6,w7,
            s0,s1,s2,s3,s4,s5,s6,s7,x+(size_t)token_ids[m]*xstride,k);
#endif
}

#if defined(__ARM_FEATURE_SVE)
/* Dequantize each 8-row MXFP4 tile once to an L1-resident BF16 pair-vector
 * panel, then reuse it for three tokens at a time.  This is lossless in weight
 * conversion; only the K-tile reduction order differs from the decode kernel. */
static inline void k3_mxfp4_group_tile(float *y, int ystride,
                                       const uint8_t *w, const uint8_t *s,
                                       const float *x, int xstride,
                                       int batch, int k) {
    const int tk = batch >= 24 ? K3_MXFP4_TILE_K_LARGE
                               : K3_MXFP4_TILE_K_SMALL;
    uint16_t pv[4 * 2 * K3_MXFP4_TILE_K_LARGE] __attribute__((aligned(256)));
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
    k3_moe_situ(gate,up,n,threads);
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

static inline int k3_expert_tp_forward_mxfp4(
        float *partial_latent, const k3_mxfp4_matrix *w1,
        const k3_mxfp4_matrix *w2, const k3_mxfp4_matrix *w3,
        const float *latent, int batch, float *gate, float *up,
        int threads, int tile_threshold) {
    if (!k3_expert_tp_layout_valid(w1, w2, w3)) return -1;
    k3_expert_forward_mxfp4_mode(partial_latent, w1, w2, w3, latent,
        batch, gate, up, threads, tile_threshold);
    return 0;
}

static inline int k3_expert_tp_selected_layout_valid(
        const k3_mxfp4_matrix *w1, const k3_mxfp4_matrix *w2,
        const k3_mxfp4_matrix *w3, int selected) {
    if(selected<1||!w1||!w2||!w3)return 0;int local=w1[0].rows;
    for(int e=0;e<selected;++e)if(w1[e].rows!=local||
        !k3_expert_tp_layout_valid(&w1[e],&w2[e],&w3[e]))return 0;
    return 1;
}

#if defined(__ARM_FEATURE_SVE)
/* Expert-TP routed-down: fuse the router-weighted sum while accumulators are
 * still vectors. Native 32-channel groups are accumulated directly, so TP72's
 * 64-channel ranks retain the same eight final horizontal reductions as TP96. */
static inline void k3_expert_tp_down_selected_sve(float *out,
        const uint8_t *const *packed_experts,
        const uint8_t *const *scale_experts,
        const float *gate,
        const float *route_weight,int selected,int local,int row){
    svbool_t pg=svptrue_b32();svfloat32_t kv=svld1(pg,ds4f_kvalues_mxfp4_f32);
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    size_t wr=(size_t)local/2,sr=(size_t)local/32;
    for(int e=0;e<selected;++e){
        const uint8_t *wbase=packed_experts[e]+(size_t)row*wr;
        const uint8_t *sbase=scale_experts[e]+(size_t)row*sr;
        for(int b=0;b<local;b+=K3_EXPERT_TP_BLOCK){
            const uint8_t*w=wbase+b/2;
            const uint8_t*s=sbase+b/32;
            const float*x=gate+(size_t)e*local+b;
            svfloat32_t xl=svld1(pg,x),xh=svld1(pg,x+16);float rw=route_weight[e];
            if(k3_tp_scale_activation){xl=svmul_n_f32_x(pg,xl,rw);
                xh=svmul_n_f32_x(pg,xh,rw);rw=1.0f;}
#define K3_TP_DOWN_ROW(R,A) do{svuint32_t z=svld1ub_u32(pg,w+(size_t)(R)*wr); \
        svuint32_t lo=svand_n_u32_x(pg,z,15),hi=svand_n_u32_x(pg,svlsr_n_u32_x(pg,z,4),15); \
        svfloat32_t p=svmul_f32_x(pg,svtbl_f32(kv,lo),xl); \
        p=svmla_f32_x(pg,p,svtbl_f32(kv,hi),xh); \
        A=svmla_n_f32_x(pg,A,p,rw*k3_e8m0_tab[s[(size_t)(R)*sr]].f);}while(0)
        K3_TP_DOWN_ROW(0,a0);K3_TP_DOWN_ROW(1,a1);K3_TP_DOWN_ROW(2,a2);K3_TP_DOWN_ROW(3,a3);
        K3_TP_DOWN_ROW(4,a4);K3_TP_DOWN_ROW(5,a5);K3_TP_DOWN_ROW(6,a6);K3_TP_DOWN_ROW(7,a7);
#undef K3_TP_DOWN_ROW
        }
    }
    out[0]=svaddv(pg,a0);out[1]=svaddv(pg,a1);out[2]=svaddv(pg,a2);out[3]=svaddv(pg,a3);
    out[4]=svaddv(pg,a4);out[5]=svaddv(pg,a5);out[6]=svaddv(pg,a6);out[7]=svaddv(pg,a7);
}

/* Prefill form of the TP=96 routed-down kernel. Expert activations are packed
 * by dispatch position rather than by expert id; fuse the top-k weighted sum
 * directly so no [assignments,7168] expert-output buffer or scatter pass is
 * required. */
static inline void k3_expert_tp_down_routed_sve(float *out,
        const k3_mxfp4_matrix *w2,const int *route_experts,
        const int *positions,const float *route_weight,int topk,
        const float *gate,int local,int row){
    svbool_t pg=svptrue_b32();svfloat32_t kv=svld1(pg,ds4f_kvalues_mxfp4_f32);
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0,a4=a0,a5=a0,a6=a0,a7=a0;
    size_t wr=(size_t)local/2,sr=(size_t)local/32;
    for(int k=0;k<topk;++k)for(int b=0;b<local;b+=K3_EXPERT_TP_BLOCK){int e=route_experts[k],pos=positions[k];
        const uint8_t*w=w2[e].packed+(size_t)row*wr+b/2;
        const uint8_t*s=w2[e].scale+(size_t)row*sr+b/32;
        const float*x=gate+(size_t)pos*local+b;
        svfloat32_t xl=svld1(pg,x),xh=svld1(pg,x+16);float rw=route_weight[k];
#define K3_TP_PREFILL_DOWN_ROW(R,A) do{svuint32_t z=svld1ub_u32(pg,w+(size_t)(R)*wr); \
        svuint32_t lo=svand_n_u32_x(pg,z,15),hi=svand_n_u32_x(pg,svlsr_n_u32_x(pg,z,4),15); \
        svfloat32_t p=svmul_f32_x(pg,svtbl_f32(kv,lo),xl); \
        p=svmla_f32_x(pg,p,svtbl_f32(kv,hi),xh); \
        A=svmla_n_f32_x(pg,A,p,rw*ggml_e8m0_to_fp32(s[(size_t)(R)*sr]));}while(0)
        K3_TP_PREFILL_DOWN_ROW(0,a0);K3_TP_PREFILL_DOWN_ROW(1,a1);
        K3_TP_PREFILL_DOWN_ROW(2,a2);K3_TP_PREFILL_DOWN_ROW(3,a3);
        K3_TP_PREFILL_DOWN_ROW(4,a4);K3_TP_PREFILL_DOWN_ROW(5,a5);
        K3_TP_PREFILL_DOWN_ROW(6,a6);K3_TP_PREFILL_DOWN_ROW(7,a7);
#undef K3_TP_PREFILL_DOWN_ROW
    }
    out[0]=svaddv(pg,a0);out[1]=svaddv(pg,a1);out[2]=svaddv(pg,a2);out[3]=svaddv(pg,a3);
    out[4]=svaddv(pg,a4);out[5]=svaddv(pg,a5);out[6]=svaddv(pg,a6);out[7]=svaddv(pg,a7);
}
#endif

/* Expert-TP prefill over a token chunk. W1/W3 retain expert buckets so their
 * long-K weights are reused across routed tokens. W2 is fused by token/top-k,
 * eliminating the otherwise dominant expert-output materialization. Scratch:
 * counts[nexpert], offsets[nexpert+1], positions/token_ids[batch*topk],
 * gathered[batch*topk,7168], gate/up[batch*topk,local]. */
static inline int k3_expert_tp_prefill_mxfp4(float *partial,
        const k3_mxfp4_matrix *w1,const k3_mxfp4_matrix *w2,
        const k3_mxfp4_matrix *w3,int nexpert,const int *route_experts,
        const float *route_weight,int batch,int topk,const float *latent,
        int *counts,int *offsets,int *positions,int *token_ids,float *gathered,
        float *gate,float *up,int threads,int tile_threshold){
    if(!partial||!route_experts||!route_weight||!latent||!counts||!offsets||
        !positions||!token_ids||!gathered||!gate||!up||nexpert<1||nexpert>4096||
        batch<1||topk<1||topk>nexpert||batch>INT_MAX/topk||threads<1||
        !k3_expert_tp_selected_layout_valid(w1,w2,w3,nexpert))return-1;
    /* At 1K, expert buckets average ~18 tokens.  The live 12-rank sweep put
     * threshold 4 at 5,972 critical-rank tok/s versus 5,863 for threshold 8.
     * Smaller chunks retain 8: threshold 4 regressed M=64/256. */
    if(batch>=1024&&tile_threshold==8)tile_threshold=4;
    int local=w1[0].rows;if(local<K3_EXPERT_TP_BLOCK||local%K3_EXPERT_TP_BLOCK)return-1;
    memset(counts,0,(size_t)nexpert*sizeof(*counts));
    for(int i=0;i<batch*topk;++i){int e=route_experts[i];if(e<0||e>=nexpert)return-1;counts[e]++;}
    offsets[0]=0;for(int e=0;e<nexpert;++e)offsets[e+1]=offsets[e]+counts[e];
    int cursor[nexpert];memcpy(cursor,offsets,(size_t)nexpert*sizeof(*cursor));
    for(int i=0;i<batch*topk;++i){int pos=cursor[route_experts[i]]++;positions[i]=pos;token_ids[pos]=i/topk;}
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#pragma omp parallel
    {
#pragma omp for schedule(static)
#endif
    for(int i=0;i<batch*topk;++i){int e=route_experts[i];if(tile_threshold>0&&counts[e]>=tile_threshold)
        memcpy(gathered+(size_t)positions[i]*K3_LATENT,
            latent+(size_t)(i/topk)*K3_LATENT,(size_t)K3_LATENT*sizeof(float));}
    int g13=local/8;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for(int task=0;task<nexpert*g13;++task){int e=task/g13,r=(task%g13)*8,m=counts[e];if(!m)continue;
        size_t wr=(size_t)K3_LATENT/2,sr=(size_t)K3_LATENT/32;
        float*g=gate+(size_t)offsets[e]*local+r,*u=up+(size_t)offsets[e]*local+r;
        if(tile_threshold>0&&m>=tile_threshold){const float*x=gathered+(size_t)offsets[e]*K3_LATENT;
            k3_mxfp4_group_batch(g,local,w1[e].packed+(size_t)r*wr,
                w1[e].scale+(size_t)r*sr,x,K3_LATENT,m,K3_LATENT,tile_threshold);
            k3_mxfp4_group_batch(u,local,w3[e].packed+(size_t)r*wr,
                w3[e].scale+(size_t)r*sr,x,K3_LATENT,m,K3_LATENT,tile_threshold);
        }else{const int*ids=token_ids+offsets[e];
            k3_mxfp4_group_svtbl_indexed(g,local,w1[e].packed+(size_t)r*wr,
                w1[e].scale+(size_t)r*sr,latent,K3_LATENT,ids,m,K3_LATENT);
            k3_mxfp4_group_svtbl_indexed(u,local,w3[e].packed+(size_t)r*wr,
                w3[e].scale+(size_t)r*sr,latent,K3_LATENT,ids,m,K3_LATENT);}}
#if defined(__ARM_FEATURE_SVE) && K3_SITU_FEXPA
    {int total=batch*topk*local,vl=(int)svcntw();
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for(int i=0;i<total;i+=vl)k3_situ_fast_sve(gate+i,gate+i,up+i,total-i<vl?total-i:vl);}
#else
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for(int i=0;i<batch*topk*local;++i)gate[i]=4.0f*tanhf(gate[i]*.25f)*
        k3_sigmoidf(gate[i])*25.0f*tanhf(up[i]*.04f);
#endif
    int g2=K3_LATENT/8;
#if defined(__ARM_FEATURE_SVE)
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for(int task=0;task<batch*g2;++task){int t=task/g2,row=(task%g2)*8;
        k3_expert_tp_down_routed_sve(partial+(size_t)t*K3_LATENT+row,w2,
            route_experts+(size_t)t*topk,positions+(size_t)t*topk,
            route_weight+(size_t)t*topk,topk,gate,local,row);}
#else
    (void)partial;return-1;
#endif
#if defined(_OPENMP)
    }
#endif
    return 0;
}

/* Orphaned workshares: every thread in an existing OpenMP team must call it. */
static inline void k3_expert_tp_forward_selected_team_mxfp4(
        float *latent_partial, const k3_mxfp4_matrix *w1,
        const k3_mxfp4_matrix *w2, const k3_mxfp4_matrix *w3,
    const float *route_weight, int selected, const float *latent,
        float *gate, float *up) {
    int local=w1[0].rows;
    int g13=local/8,g2=K3_LATENT/8;
    const uint8_t *w2_packed[selected], *w2_scale[selected];
    for (int e = 0; e < selected; ++e)
        k3_mxfp4_thread_weights(&w2[e], &w2_packed[e], &w2_scale[e]);
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for(int task=0;task<selected*g13;++task){int e=task/g13,r=(task%g13)*8;
        float*g=gate+(size_t)e*local+r,*u=up+(size_t)e*local+r;
        const uint8_t *m1_packed, *m1_scale, *m3_packed, *m3_scale;
        k3_mxfp4_thread_weights(&w1[e], &m1_packed, &m1_scale);
        k3_mxfp4_thread_weights(&w3[e], &m3_packed, &m3_scale);
        size_t wr=(size_t)K3_LATENT/2,sr=(size_t)K3_LATENT/32;
        k3_mxfp4_group_batch(g,local,m1_packed+(size_t)r*wr,
            m1_scale+(size_t)r*sr,latent,K3_LATENT,1,K3_LATENT,0);
        k3_mxfp4_group_batch(u,local,m3_packed+(size_t)r*wr,
            m3_scale+(size_t)r*sr,latent,K3_LATENT,1,K3_LATENT,0);
#if defined(__ARM_FEATURE_SVE) && K3_SITU_FEXPA
        k3_situ_fast_sve(g,g,u,8);
#else
        for(int j=0;j<8;++j)g[j]=4.0f*tanhf(g[j]*.25f)*
            k3_sigmoidf(g[j])*25.0f*tanhf(u[j]*.04f);
#endif
    }
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for(int gr=0;gr<g2;++gr){int r=gr*8;
#if defined(__ARM_FEATURE_SVE) && K3_TP_FUSED_DOWN32
        if(local%K3_EXPERT_TP_BLOCK==0){k3_expert_tp_down_selected_sve(latent_partial+r,
                w2_packed,w2_scale,gate,route_weight,selected,local,r);continue;}
#endif
        float sum[8]={0},tmp[8];
        for(int e=0;e<selected;++e){const k3_mxfp4_matrix*m=&w2[e];size_t wr=(size_t)local/2,sr=(size_t)local/32;
            k3_mxfp4_group_batch(tmp,8,m->packed+(size_t)r*wr,
                m->scale+(size_t)r*sr,gate+(size_t)e*local,local,1,local,0);
            for(int j=0;j<8;++j)sum[j]+=route_weight[e]*tmp[j];}
        for(int j=0;j<8;++j)latent_partial[r+j]=sum[j];}
}

/* Decode path for the 16 selected experts on one intermediate-TP rank.  The
 * team is shared across experts, avoiding 48-thread startup/workshare overhead
 * for each tiny 32-channel slice at TP=96. */
static inline int k3_expert_tp_forward_selected_mxfp4(
        float *latent_partial, const k3_mxfp4_matrix *w1,
        const k3_mxfp4_matrix *w2, const k3_mxfp4_matrix *w3,
        const float *route_weight, int selected, const float *latent,
        float *gate, float *up, float *expert_out, int threads) {
    if(!k3_expert_tp_selected_layout_valid(w1,w2,w3,selected))return-1;
#if defined(_OPENMP)
    (void)expert_out;omp_set_num_threads(threads);
#pragma omp parallel
    k3_expert_tp_forward_selected_team_mxfp4(latent_partial,w1,w2,w3,
        route_weight,selected,latent,gate,up);
#else
    (void)threads;
    int local = w1[0].rows;
    for(int e=0;e<selected;++e){if(k3_expert_tp_forward_mxfp4(
        expert_out+(size_t)e*K3_LATENT,&w1[e],&w2[e],&w3[e],latent,1,
        gate+(size_t)e*local,up+(size_t)e*local,1,0))return-1;}
    for(int i=0;i<K3_LATENT;++i){float sum=0;for(int e=0;e<selected;++e)
        sum+=route_weight[e]*expert_out[(size_t)e*K3_LATENT+i];latent_partial[i]=sum;}
#endif
    return 0;
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
#if defined(__ARM_FEATURE_SVE) && K3_SITU_FEXPA
            int vl=(int)svcntw();
            for(int i=lane*vl;i<n;i+=group_threads*vl)
                k3_situ_fast_sve(g+i,g+i,u+i,n-i<vl?n-i:vl);
#else
            for(int i=lane;i<n;i+=group_threads)
                g[i]=4.0f*tanhf(g[i]*.25f)*k3_sigmoidf(g[i])
                    *25.0f*tanhf(u[i]*.04f);
#endif
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
#if defined(__ARM_FEATURE_SVE) && K3_SITU_FEXPA
    {int total=nlocal*batch*K3_EXPERT_INTER,vl=(int)svcntw();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<total;i+=vl){int expert=i/(batch*K3_EXPERT_INTER);
        int rem=i%(batch*K3_EXPERT_INTER);
        if(rem<counts[expert]*K3_EXPERT_INTER)
            k3_situ_fast_sve(gate+i,gate+i,up+i,total-i<vl?total-i:vl);}}
#else
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<nlocal*batch*K3_EXPERT_INTER;++i){
        int expert=i/(batch*K3_EXPERT_INTER),rem=i%(batch*K3_EXPERT_INTER);
        if(rem<counts[expert]*K3_EXPERT_INTER)
            gate[i]=4.0f*tanhf(gate[i]*.25f)*k3_sigmoidf(gate[i])
                    *25.0f*tanhf(up[i]*.04f);}
#endif
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
