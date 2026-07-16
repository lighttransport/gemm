/*
 * glm5_impl.h - GLM-5.2 (text) forward graph + synthetic allocator.
 *
 * Included by the EP runner (a64fx/glm5/glm5_ep_runner.c) and single-node drivers.
 * Provides: glm5_arena_size, glm5_alloc_synth, glm5_free, glm5_forward_token.
 *
 * Parallelism:
 *   - EXPERT-PARALLEL (always): expert e owned by rank e%ep_size (slot e/ep_size);
 *     routed-expert partials accumulate into s_route, one m->ar_cb per MoE layer sums
 *     them across the group (== tp_allreduce_sum).
 *   - TENSOR-PARALLEL (opt-in, GLM5_TP / GLM5_TP_*): shards the replicated dense so full
 *     dims fit at 96 nodes. The TP group == the EP group (same N ranks). Each shard is
 *     reconstructed bit-exactly by the same all-reduce:
 *       TP_ATTN   : q heads split (glm5_shard_heads) -> wq owns those rows, wo owns those
 *                   input columns; per-rank partial o_proj -> ar_cb sum. kv heads + MSA
 *                   index stay replicated (small).
 *       TP_SHARED : shared-expert inter split -> partial folded into s_route (one reduce).
 *       TP_FFN    : dense-layer (0..2) FFN inter split -> partial down-proj -> ar_cb sum.
 *       TP_HEAD   : lm_head vocab split -> local (val,global-idx) argmax -> ar_argmax merge.
 *       TP_EMBED  : embed vocab split (real-weight gen only; synth seeds activations).
 *     Sharding is detected in the forward by range<full (set in glm5_alloc_synth); no extra
 *     flags threaded. Every reduce makes the result identical on every rank -> lockstep.
 *
 *   - MSA (MiniMax Sparse Attention, opt-in GLM5_MSA, sparse layers L>=n_dense_layers):
 *     a small index projection (4 q heads / 1 k head MQA, dim 128) + QK-norm + partial
 *     RoPE scores 128-token blocks (max over block of summed index-head dots); the top-K
 *     blocks (+ init + local) are selected and the GQA attention runs only over their
 *     positions. Below the block threshold it is exactly full attention (byte-identical).
 *
 * Weights SYNTHETIC bf16; logits meaningless. Targets: memory fit, NaN-free, cross-rank
 * lockstep argmax. Offline (no uTofu) the EP+TP reconstruction can be checked single-
 * process by summing N rank-shards (see a64fx/glm5/glm5_tp_selftest.c).
 */
#ifndef GLM5_IMPL_H
#define GLM5_IMPL_H

#include "glm5.h"
#include "glm5_mxfp8.h"   /* FP8 E4GLM5 + E8M0 per-32 block scale matvec */
#include "glm5_int8.h"    /* INT8 w8a16 (offset-binary, per-row group/channel f32 scale) matvec */
#ifdef _OPENMP
#include <omp.h>
#endif
/* parallelize a matvec/down-proj only when the output dim is large enough that the
 * fork/join barrier is amortized (small projections stay serial). */
#define GLM5_PAR_MIN 512

static int glm5_envi(const char*k,int d){ const char*v=getenv(k); return (v&&*v)?atoi(v):d; }
static inline void glm5_shard_shared_inter(int total,int rank,int size,int*r0,int*rows){
    const char*v=getenv("GLM5_TP_SHARED_BLOCK");
    int block=(v&&*v)?atoi(v):128;  /* 0 = exact even rows; default keeps faster scale-aligned shards */
    if(block>0) glm5_shard_blocks(total,block,rank,size,r0,rows);
    else glm5_shard(total,rank,size,r0,rows);
}
static inline double glm5_prof_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static inline void glm5_prof_add(glm5_model*m,int phase,double t0){ m->prof[phase]+=glm5_prof_now()-t0; }

#if defined(GLM5_USE_FAPP) && GLM5_USE_FAPP
extern void fapp_start(const char*, int, int) __attribute__((weak));
extern void fapp_stop(const char*, int, int) __attribute__((weak));
static inline int glm5_fapp_on(void){
    static int on=-1;
    if(on<0) on=glm5_envi("GLM5_FAPP",0) && fapp_start && fapp_stop;
    return on;
}
#define GLM5_FAPP_START(name) do{ if(glm5_fapp_on()) fapp_start((name),1,0); }while(0)
#define GLM5_FAPP_STOP(name)  do{ if(glm5_fapp_on()) fapp_stop ((name),1,0); }while(0)
#else
#define GLM5_FAPP_START(name) do{}while(0)
#define GLM5_FAPP_STOP(name)  do{}while(0)
#endif

#if defined(__ARM_FEATURE_SVE)
/* 2^x via the SVE FEXPA accelerator (~5 instrs): round-to-table-index, svexpa table lookup,
 * first-order residual correction 2^r ~= 1 + r*ln2. ~0.01% rel error — plenty for router sigmoids. */
static inline svfloat32_t glm5_exp2_fexpa(svbool_t pg, svfloat32_t x){
    const float shift_f=204927.0f;                 /* 0x48481fc0: FEXPA-compatible rounding shift */
    svfloat32_t shift=svdup_f32(shift_f);
    svfloat32_t z=svadd_f32_x(pg,x,shift);
    svfloat32_t n=svsub_f32_x(pg,z,shift);
    svfloat32_t r=svsub_f32_x(pg,x,n);
    svfloat32_t scale=svexpa_f32(svreinterpret_u32_f32(z));
    svfloat32_t corr=svmla_n_f32_x(pg,svdup_f32(1.0f),r,0.6931471805599453f);
    return svmul_f32_x(pg,scale,corr);
}
#endif
/* in-place sigmoid 1/(1+e^-x) over a row of n floats. SVE: exp(-x)=2^(-x*log2e) via FEXPA +
 * one Newton step on the reciprocal. Scalar fallback uses expf. */
static inline void glm5_sigmoid_row(float*rl,int n){
#if defined(__ARM_FEATURE_SVE)
    int vl=(int)svcntw();
    for(int e=0;e<n;e+=vl){
        svbool_t pg=svwhilelt_b32(e,n);
        svfloat32_t v=svld1_f32(pg,&rl[e]);
        svfloat32_t t=svmax_n_f32_x(pg,svmin_n_f32_x(pg,svmul_n_f32_x(pg,v,-1.4426950408889634f),80.f),-80.f);
        svfloat32_t en=glm5_exp2_fexpa(pg,t);
        svfloat32_t den=svadd_n_f32_x(pg,en,1.0f);
        svfloat32_t inv=svrecpe_f32(den); inv=svmul_f32_x(pg,inv,svrecps_f32(den,inv));
        svst1_f32(pg,&rl[e],inv);
    }
#else
    for(int e=0;e<n;e++) rl[e]=1.0f/(1.0f+expf(-rl[e]));
#endif
}

/* ===================== scalar/codec helpers ===================== */
static inline uint16_t glm5_f2bf(float f){
    uint32_t u; memcpy(&u,&f,4);
    if(((u>>23)&0xff)==0xff) return (uint16_t)(u>>16);
    uint32_t r=u+0x7fffu+((u>>16)&1u); return (uint16_t)(r>>16);
}
static inline float glm5_bf2f(uint16_t h){ return bf16_to_f32_scalar(h); }
/* IEEE fp16 (half) KV codec: ~11-bit mantissa vs bf16's 8 -> higher KV fidelity (the model
 * computes K/V in f32, so bf16 storage already quantizes; fp16 is the higher-precision option
 * for the quality A/B vs int4). Same 2 bytes/elem as bf16. A64FX has native fp16. */
static inline uint16_t glm5_f2h(float f){ __fp16 h=(__fp16)f; uint16_t u; memcpy(&u,&h,2); return u; }
static inline float    glm5_h2f(uint16_t u){ __fp16 h; memcpy(&h,&u,2); return (float)h; }
/* uint16 KV path encode/decode, bf16 (default) or fp16 (GLM5_KV_FP16=1). */
static inline uint16_t glm5_kv_enc(const glm5_model*m,float f){ return m->kv_fp16 ? glm5_f2h(f) : glm5_f2bf(f); }
static inline float    glm5_kv_dec(const glm5_model*m,uint16_t u){ return m->kv_fp16 ? glm5_h2f(u) : glm5_bf2f(u); }

static inline float glm5_dot_f32_opt(const float*a,const float*b,int n,int use_sve){
#if defined(__ARM_FEATURE_SVE)
    if(use_sve){
        int vl=(int)svcntw();
        /* 4 independent accumulators break the loop-carried svmla dependency chain (A64FX FMA
         * latency ~9 cyc, 2 FLA pipes) -> ~2x on the isolated dot (qlair: single-acc was ~71%
         * data-dependency stalls). GLM5_DOT_ACC4=0 restores the single-acc path for A/B. */
        static int acc4=-1;
        if(acc4<0){ const char*e=getenv("GLM5_DOT_ACC4"); acc4=e?atoi(e):1; }
        if(acc4){
            svfloat32_t a0=svdup_f32(0.0f),a1=a0,a2=a0,a3=a0;
            svbool_t pg=svptrue_b32();
            int i=0;
            for(;i+4*vl<=n;i+=4*vl){
                a0=svmla_f32_x(pg,a0,svld1(pg,a+i),      svld1(pg,b+i));
                a1=svmla_f32_x(pg,a1,svld1(pg,a+i+vl),   svld1(pg,b+i+vl));
                a2=svmla_f32_x(pg,a2,svld1(pg,a+i+2*vl), svld1(pg,b+i+2*vl));
                a3=svmla_f32_x(pg,a3,svld1(pg,a+i+3*vl), svld1(pg,b+i+3*vl));
            }
            for(;i<n;i+=vl){ svbool_t pt=svwhilelt_b32(i,n); a0=svmla_f32_x(pt,a0,svld1(pt,a+i),svld1(pt,b+i)); }
            a0=svadd_f32_x(pg,svadd_f32_x(pg,a0,a1),svadd_f32_x(pg,a2,a3));
            return svaddv_f32(pg,a0);
        }
        svfloat32_t acc=svdup_f32(0.0f);
        for(int i=0;i<n;i+=vl){
            svbool_t pg=svwhilelt_b32(i,n);
            acc=svmla_f32_x(pg,acc,svld1(pg,a+i),svld1(pg,b+i));
        }
        return svaddv_f32(svptrue_b32(),acc);
    }
#else
    (void)use_sve;
#endif
    double d=0.0; for(int i=0;i<n;i++) d+=(double)a[i]*b[i]; return (float)d;
}
static inline void glm5_scale_f32(float*x,float s,int n){
#if defined(__ARM_FEATURE_SVE)
    int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n);
        svst1(pg,x+i,svmul_n_f32_x(pg,svld1(pg,x+i),s));
    }
#else
    for(int i=0;i<n;i++) x[i]*=s;
#endif
}
static inline void glm5_axpy_f32(float*y,const float*x,float a,int n){
#if defined(__ARM_FEATURE_SVE)
    int vl=(int)svcntw();
    /* 4-way unroll: the online-softmax AXPY has NO reduction dependency, but the single-op loop is
     * store/load-pipeline bound on A64FX; unrolling hides it -> measured 1.93x NATIVELY (K0, job
     * 49420133). Bit-identical to the 1-op loop (no reassociation). NOTE qlair modelled this as only
     * 1.03x -> qlair under-models memory-pipeline effects; trust the NATIVE probe here. Same
     * GLM5_DOT_ACC4=0 escape hatch restores the 1-op loop for the A/B. */
    static int acc4=-1;
    if(acc4<0){ const char*e=getenv("GLM5_DOT_ACC4"); acc4=e?atoi(e):1; }
    if(acc4){
        svbool_t pg=svptrue_b32(); int i=0;
        for(;i+4*vl<=n;i+=4*vl){
            svst1(pg,y+i,     svmla_n_f32_x(pg,svld1(pg,y+i),      svld1(pg,x+i),a));
            svst1(pg,y+i+vl,  svmla_n_f32_x(pg,svld1(pg,y+i+vl),   svld1(pg,x+i+vl),a));
            svst1(pg,y+i+2*vl,svmla_n_f32_x(pg,svld1(pg,y+i+2*vl), svld1(pg,x+i+2*vl),a));
            svst1(pg,y+i+3*vl,svmla_n_f32_x(pg,svld1(pg,y+i+3*vl), svld1(pg,x+i+3*vl),a));
        }
        for(;i<n;i+=vl){ svbool_t q=svwhilelt_b32(i,n); svst1(q,y+i,svmla_n_f32_x(q,svld1(q,y+i),svld1(q,x+i),a)); }
        return;
    }
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n);
        svfloat32_t yv=svld1(pg,y+i);
        yv=svmla_n_f32_x(pg,yv,svld1(pg,x+i),a);
        svst1(pg,y+i,yv);
    }
#else
    for(int i=0;i<n;i++) y[i]+=a*x[i];
#endif
}

/* ===================== int4-KV codec (GLM5_INT4_KV, for 1M context) =====================
 * Per group of n (== head_dim) values: signed int4 (+/-7), scale = absmax/7 stored bf16.
 * ~3.9x smaller than bf16 (n/2 bytes + 1 bf16 scale vs 2n bytes). Used for the k/v/idx
 * caches; dequant happens on read in the attention/index loops (decode is latency-bound,
 * scalar matches the existing bf16 inner loops). n must be even. */
static inline uint16_t glm5_q4_pack(uint8_t*restrict q, const float*v, int n){
    float amax=0; for(int i=0;i<n;i++){ float a=fabsf(v[i]); if(a>amax)amax=a; }
    float inv = amax>0 ? 7.0f/amax : 0.0f;
    for(int i=0;i<n;i+=2){
        int a=(int)lrintf(v[i]*inv);   if(a>7)a=7; else if(a<-7)a=-7;
        int b=(int)lrintf(v[i+1]*inv); if(b>7)b=7; else if(b<-7)b=-7;
        q[i>>1]=(uint8_t)((a&0xF)|((b&0xF)<<4));
    }
    return glm5_f2bf(amax/7.0f);
}
static inline float glm5_q4_dot(const uint8_t*restrict q, uint16_t scbf, const float*restrict x, int n){
    double d=0; for(int i=0;i<n;i+=2){ uint8_t by=q[i>>1];
        int a=(int)(int8_t)(by<<4)>>4, b=(int)(int8_t)by>>4;   /* sign-extend low/high nibble */
        d += (double)a*x[i] + (double)b*x[i+1]; }
    return (float)(d*glm5_bf2f(scbf));
}
static inline void glm5_q4_axpy(float*restrict out, const uint8_t*restrict q, uint16_t scbf, float w, int n){
    float ws=w*glm5_bf2f(scbf);
    for(int i=0;i<n;i+=2){ uint8_t by=q[i>>1];
        int a=(int)(int8_t)(by<<4)>>4, b=(int)(int8_t)by>>4;
        out[i]+=ws*a; out[i+1]+=ws*b; }
}
static inline void glm5_q4_unpack(float*restrict out, const uint8_t*restrict q, uint16_t scbf, int n){
    float sc=glm5_bf2f(scbf);
    for(int i=0;i<n;i+=2){ uint8_t by=q[i>>1];
        int a=(int)(int8_t)(by<<4)>>4, b=(int)(int8_t)by>>4;
        out[i]=sc*a; out[i+1]=sc*b; }
}

static uint64_t glm5_sm;
static inline double glm5_sm_next(void){
    glm5_sm += 0x9E3779B97F4A7C15ull; uint64_t z=glm5_sm;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; z^=z>>31;
    return (double)(z>>11)/(double)(1ull<<53);
}
static inline void glm5_fill_bf16(uint16_t*w,size_t n,float amp){ for(size_t i=0;i<n;i++) w[i]=glm5_f2bf((float)((glm5_sm_next()*2.0-1.0)*amp)); }
static inline void glm5_fill_f32 (float   *w,size_t n,float amp){ for(size_t i=0;i<n;i++) w[i]=(float)((glm5_sm_next()*2.0-1.0)*amp); }

static inline void glm5_mv_f32(float*restrict y,const float*W,const float*x,int rows,int cols){
    for(int r=0;r<rows;r++){ const float*w=W+(size_t)r*cols; double s=0; for(int i=0;i<cols;i++) s+=(double)w[i]*x[i]; y[r]=(float)s; }
}
#if defined(__ARM_FEATURE_SVE)
static inline void glm5_bf16_4row_3x_acc(float*a0,float*a1,float*a2,
        const uint16_t*w0,const uint16_t*w1,const uint16_t*w2,const uint16_t*w3,
        const float*x0,const float*x1,const float*x2,int n){
    svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0);
    svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
    svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0);
    int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n);
        svfloat32_t x0v=svld1(pg,x0+i),x1v=svld1(pg,x1+i),x2v=svld1(pg,x2+i);
        svfloat32_t v0=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w0+i),16));
        svfloat32_t v1=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w1+i),16));
        svfloat32_t v2=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w2+i),16));
        svfloat32_t v3=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w3+i),16));
        a00=svmla_x(pg,a00,v0,x0v); a01=svmla_x(pg,a01,v1,x0v); a02=svmla_x(pg,a02,v2,x0v); a03=svmla_x(pg,a03,v3,x0v);
        a10=svmla_x(pg,a10,v0,x1v); a11=svmla_x(pg,a11,v1,x1v); a12=svmla_x(pg,a12,v2,x1v); a13=svmla_x(pg,a13,v3,x1v);
        a20=svmla_x(pg,a20,v0,x2v); a21=svmla_x(pg,a21,v1,x2v); a22=svmla_x(pg,a22,v2,x2v); a23=svmla_x(pg,a23,v3,x2v);
    }
    svbool_t pt=svptrue_b32();
    a0[0]+=svaddv_f32(pt,a00); a0[1]+=svaddv_f32(pt,a01); a0[2]+=svaddv_f32(pt,a02); a0[3]+=svaddv_f32(pt,a03);
    a1[0]+=svaddv_f32(pt,a10); a1[1]+=svaddv_f32(pt,a11); a1[2]+=svaddv_f32(pt,a12); a1[3]+=svaddv_f32(pt,a13);
    a2[0]+=svaddv_f32(pt,a20); a2[1]+=svaddv_f32(pt,a21); a2[2]+=svaddv_f32(pt,a22); a2[3]+=svaddv_f32(pt,a23);
}
/* 4 bf16 weight rows x 4 activation tokens: each widened weight vector feeds 4 FMAs (vs 3 in
 * the 3x kernel) -> 25% less relative weight-tile traffic. 16 acc + 4 x + 4 w = 24 SVE regs. */
static inline void glm5_bf16_4row_4x_acc(float*a0,float*a1,float*a2,float*a3,
        const uint16_t*w0,const uint16_t*w1,const uint16_t*w2,const uint16_t*w3,
        const float*x0,const float*x1,const float*x2,const float*x3,int n){
    svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0);
    svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
    svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0);
    svfloat32_t a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0);
    int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n);
        svfloat32_t v0=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w0+i),16));
        svfloat32_t v1=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w1+i),16));
        svfloat32_t v2=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w2+i),16));
        svfloat32_t v3=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w3+i),16));
        svfloat32_t xv=svld1(pg,x0+i);
        a00=svmla_x(pg,a00,v0,xv); a01=svmla_x(pg,a01,v1,xv); a02=svmla_x(pg,a02,v2,xv); a03=svmla_x(pg,a03,v3,xv);
        xv=svld1(pg,x1+i);
        a10=svmla_x(pg,a10,v0,xv); a11=svmla_x(pg,a11,v1,xv); a12=svmla_x(pg,a12,v2,xv); a13=svmla_x(pg,a13,v3,xv);
        xv=svld1(pg,x2+i);
        a20=svmla_x(pg,a20,v0,xv); a21=svmla_x(pg,a21,v1,xv); a22=svmla_x(pg,a22,v2,xv); a23=svmla_x(pg,a23,v3,xv);
        xv=svld1(pg,x3+i);
        a30=svmla_x(pg,a30,v0,xv); a31=svmla_x(pg,a31,v1,xv); a32=svmla_x(pg,a32,v2,xv); a33=svmla_x(pg,a33,v3,xv);
    }
    svbool_t pt=svptrue_b32();
    a0[0]+=svaddv_f32(pt,a00); a0[1]+=svaddv_f32(pt,a01); a0[2]+=svaddv_f32(pt,a02); a0[3]+=svaddv_f32(pt,a03);
    a1[0]+=svaddv_f32(pt,a10); a1[1]+=svaddv_f32(pt,a11); a1[2]+=svaddv_f32(pt,a12); a1[3]+=svaddv_f32(pt,a13);
    a2[0]+=svaddv_f32(pt,a20); a2[1]+=svaddv_f32(pt,a21); a2[2]+=svaddv_f32(pt,a22); a2[3]+=svaddv_f32(pt,a23);
    a3[0]+=svaddv_f32(pt,a30); a3[1]+=svaddv_f32(pt,a31); a3[2]+=svaddv_f32(pt,a32); a3[3]+=svaddv_f32(pt,a33);
}
/* 4 bf16 weight rows x 5 activation tokens: each widened weight vector feeds 5 FMAs.
 * 20 acc + 4 w + 1 reused x = 25 SVE regs. */
static inline void glm5_bf16_4row_5x_acc(float*a0,float*a1,float*a2,float*a3,float*a4,
        const uint16_t*w0,const uint16_t*w1,const uint16_t*w2,const uint16_t*w3,
        const float*x0,const float*x1,const float*x2,const float*x3,const float*x4,int n){
    svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0);
    svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
    svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0);
    svfloat32_t a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0);
    svfloat32_t a40=svdup_f32(0),a41=svdup_f32(0),a42=svdup_f32(0),a43=svdup_f32(0);
    int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n);
        svfloat32_t v0=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w0+i),16));
        svfloat32_t v1=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w1+i),16));
        svfloat32_t v2=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w2+i),16));
        svfloat32_t v3=svreinterpret_f32(svlsl_x(pg,svld1uh_u32(pg,w3+i),16));
        svfloat32_t xv=svld1(pg,x0+i);
        a00=svmla_x(pg,a00,v0,xv); a01=svmla_x(pg,a01,v1,xv); a02=svmla_x(pg,a02,v2,xv); a03=svmla_x(pg,a03,v3,xv);
        xv=svld1(pg,x1+i);
        a10=svmla_x(pg,a10,v0,xv); a11=svmla_x(pg,a11,v1,xv); a12=svmla_x(pg,a12,v2,xv); a13=svmla_x(pg,a13,v3,xv);
        xv=svld1(pg,x2+i);
        a20=svmla_x(pg,a20,v0,xv); a21=svmla_x(pg,a21,v1,xv); a22=svmla_x(pg,a22,v2,xv); a23=svmla_x(pg,a23,v3,xv);
        xv=svld1(pg,x3+i);
        a30=svmla_x(pg,a30,v0,xv); a31=svmla_x(pg,a31,v1,xv); a32=svmla_x(pg,a32,v2,xv); a33=svmla_x(pg,a33,v3,xv);
        xv=svld1(pg,x4+i);
        a40=svmla_x(pg,a40,v0,xv); a41=svmla_x(pg,a41,v1,xv); a42=svmla_x(pg,a42,v2,xv); a43=svmla_x(pg,a43,v3,xv);
    }
    svbool_t pt=svptrue_b32();
    a0[0]+=svaddv_f32(pt,a00); a0[1]+=svaddv_f32(pt,a01); a0[2]+=svaddv_f32(pt,a02); a0[3]+=svaddv_f32(pt,a03);
    a1[0]+=svaddv_f32(pt,a10); a1[1]+=svaddv_f32(pt,a11); a1[2]+=svaddv_f32(pt,a12); a1[3]+=svaddv_f32(pt,a13);
    a2[0]+=svaddv_f32(pt,a20); a2[1]+=svaddv_f32(pt,a21); a2[2]+=svaddv_f32(pt,a22); a2[3]+=svaddv_f32(pt,a23);
    a3[0]+=svaddv_f32(pt,a30); a3[1]+=svaddv_f32(pt,a31); a3[2]+=svaddv_f32(pt,a32); a3[3]+=svaddv_f32(pt,a33);
    a4[0]+=svaddv_f32(pt,a40); a4[1]+=svaddv_f32(pt,a41); a4[2]+=svaddv_f32(pt,a42); a4[3]+=svaddv_f32(pt,a43);
}
static inline void glm5_qk4_f32(float s[4][4], const float*q0,const float*q1,const float*q2,const float*q3,
        const float*k0,const float*k1,const float*k2,const float*k3,int n){
    svfloat32_t c00=svdup_f32(0),c01=c00,c02=c00,c03=c00, c10=c00,c11=c00,c12=c00,c13=c00;
    svfloat32_t c20=c00,c21=c00,c22=c00,c23=c00, c30=c00,c31=c00,c32=c00,c33=c00;
    int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n);
        svfloat32_t qa=svld1(pg,q0+i),qb=svld1(pg,q1+i),qc=svld1(pg,q2+i),qd=svld1(pg,q3+i);
        svfloat32_t ka=svld1(pg,k0+i);
        c00=svmla_f32_x(pg,c00,qa,ka); c10=svmla_f32_x(pg,c10,qb,ka); c20=svmla_f32_x(pg,c20,qc,ka); c30=svmla_f32_x(pg,c30,qd,ka);
        svfloat32_t kb=svld1(pg,k1+i);
        c01=svmla_f32_x(pg,c01,qa,kb); c11=svmla_f32_x(pg,c11,qb,kb); c21=svmla_f32_x(pg,c21,qc,kb); c31=svmla_f32_x(pg,c31,qd,kb);
        svfloat32_t kc=svld1(pg,k2+i);
        c02=svmla_f32_x(pg,c02,qa,kc); c12=svmla_f32_x(pg,c12,qb,kc); c22=svmla_f32_x(pg,c22,qc,kc); c32=svmla_f32_x(pg,c32,qd,kc);
        svfloat32_t kd=svld1(pg,k3+i);
        c03=svmla_f32_x(pg,c03,qa,kd); c13=svmla_f32_x(pg,c13,qb,kd); c23=svmla_f32_x(pg,c23,qc,kd); c33=svmla_f32_x(pg,c33,qd,kd);
    }
    svbool_t pt=svptrue_b32();
    s[0][0]=svaddv_f32(pt,c00); s[0][1]=svaddv_f32(pt,c01); s[0][2]=svaddv_f32(pt,c02); s[0][3]=svaddv_f32(pt,c03);
    s[1][0]=svaddv_f32(pt,c10); s[1][1]=svaddv_f32(pt,c11); s[1][2]=svaddv_f32(pt,c12); s[1][3]=svaddv_f32(pt,c13);
    s[2][0]=svaddv_f32(pt,c20); s[2][1]=svaddv_f32(pt,c21); s[2][2]=svaddv_f32(pt,c22); s[2][3]=svaddv_f32(pt,c23);
    s[3][0]=svaddv_f32(pt,c30); s[3][1]=svaddv_f32(pt,c31); s[3][2]=svaddv_f32(pt,c32); s[3][3]=svaddv_f32(pt,c33);
}
static inline void glm5_qk4_int16(float s[4][4], const float*q0,const float*q1,const float*q2,const float*q3,
        const float*k0,const float*k1,const float*k2,const float*k3,int n){
    int16_t qv[4][512], kv[4][512]; float qs[4], ks[4];
    const float*Q[4]={q0,q1,q2,q3}, *K[4]={k0,k1,k2,k3};
    for(int r=0;r<4;r++){
        float mx=1e-20f; for(int i=0;i<n;i++){ float a=fabsf(Q[r][i]); if(a>mx) mx=a; }
        float inv=32767.0f/mx; qs[r]=mx/32767.0f;
        for(int i=0;i<n;i++){ int v=(int)lrintf(Q[r][i]*inv); v=v>32767?32767:(v<-32767?-32767:v); qv[r][i]=(int16_t)v; }
        mx=1e-20f; for(int i=0;i<n;i++){ float a=fabsf(K[r][i]); if(a>mx) mx=a; }
        inv=32767.0f/mx; ks[r]=mx/32767.0f;
        for(int i=0;i<n;i++){ int v=(int)lrintf(K[r][i]*inv); v=v>32767?32767:(v<-32767?-32767:v); kv[r][i]=(int16_t)v; }
    }
    svint64_t c00=svdup_s64(0),c01=c00,c02=c00,c03=c00, c10=c00,c11=c00,c12=c00,c13=c00;
    svint64_t c20=c00,c21=c00,c22=c00,c23=c00, c30=c00,c31=c00,c32=c00,c33=c00;
    int vh=(int)svcnth();
    for(int i=0;i<n;i+=vh){
        svbool_t pg=svwhilelt_b16(i,n);
        svint16_t qa=svld1_s16(pg,qv[0]+i),qb=svld1_s16(pg,qv[1]+i),qc=svld1_s16(pg,qv[2]+i),qd=svld1_s16(pg,qv[3]+i);
        svint16_t ka=svld1_s16(pg,kv[0]+i);
        c00=svdot_s64(c00,qa,ka); c10=svdot_s64(c10,qb,ka); c20=svdot_s64(c20,qc,ka); c30=svdot_s64(c30,qd,ka);
        svint16_t kb=svld1_s16(pg,kv[1]+i);
        c01=svdot_s64(c01,qa,kb); c11=svdot_s64(c11,qb,kb); c21=svdot_s64(c21,qc,kb); c31=svdot_s64(c31,qd,kb);
        svint16_t kc=svld1_s16(pg,kv[2]+i);
        c02=svdot_s64(c02,qa,kc); c12=svdot_s64(c12,qb,kc); c22=svdot_s64(c22,qc,kc); c32=svdot_s64(c32,qd,kc);
        svint16_t kd=svld1_s16(pg,kv[3]+i);
        c03=svdot_s64(c03,qa,kd); c13=svdot_s64(c13,qb,kd); c23=svdot_s64(c23,qc,kd); c33=svdot_s64(c33,qd,kd);
    }
    svbool_t pt=svptrue_b64();
    s[0][0]=(float)svaddv_s64(pt,c00)*qs[0]*ks[0]; s[0][1]=(float)svaddv_s64(pt,c01)*qs[0]*ks[1]; s[0][2]=(float)svaddv_s64(pt,c02)*qs[0]*ks[2]; s[0][3]=(float)svaddv_s64(pt,c03)*qs[0]*ks[3];
    s[1][0]=(float)svaddv_s64(pt,c10)*qs[1]*ks[0]; s[1][1]=(float)svaddv_s64(pt,c11)*qs[1]*ks[1]; s[1][2]=(float)svaddv_s64(pt,c12)*qs[1]*ks[2]; s[1][3]=(float)svaddv_s64(pt,c13)*qs[1]*ks[3];
    s[2][0]=(float)svaddv_s64(pt,c20)*qs[2]*ks[0]; s[2][1]=(float)svaddv_s64(pt,c21)*qs[2]*ks[1]; s[2][2]=(float)svaddv_s64(pt,c22)*qs[2]*ks[2]; s[2][3]=(float)svaddv_s64(pt,c23)*qs[2]*ks[3];
    s[3][0]=(float)svaddv_s64(pt,c30)*qs[3]*ks[0]; s[3][1]=(float)svaddv_s64(pt,c31)*qs[3]*ks[1]; s[3][2]=(float)svaddv_s64(pt,c32)*qs[3]*ks[2]; s[3][3]=(float)svaddv_s64(pt,c33)*qs[3]*ks[3];
}
static inline void glm5_f32_4row_3x_acc(float*a0,float*a1,float*a2,
        const float*w0,const float*w1,const float*w2,const float*w3,
        const float*x0,const float*x1,const float*x2,int n){
    svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0);
    svfloat32_t a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
    svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0);
    int vl=(int)svcntw();
    for(int i=0;i<n;i+=vl){
        svbool_t pg=svwhilelt_b32(i,n);
        svfloat32_t x0v=svld1(pg,x0+i),x1v=svld1(pg,x1+i),x2v=svld1(pg,x2+i);
        svfloat32_t v0=svld1(pg,w0+i),v1=svld1(pg,w1+i),v2=svld1(pg,w2+i),v3=svld1(pg,w3+i);
        a00=svmla_x(pg,a00,v0,x0v); a01=svmla_x(pg,a01,v1,x0v); a02=svmla_x(pg,a02,v2,x0v); a03=svmla_x(pg,a03,v3,x0v);
        a10=svmla_x(pg,a10,v0,x1v); a11=svmla_x(pg,a11,v1,x1v); a12=svmla_x(pg,a12,v2,x1v); a13=svmla_x(pg,a13,v3,x1v);
        a20=svmla_x(pg,a20,v0,x2v); a21=svmla_x(pg,a21,v1,x2v); a22=svmla_x(pg,a22,v2,x2v); a23=svmla_x(pg,a23,v3,x2v);
    }
    svbool_t pt=svptrue_b32();
    a0[0]+=svaddv_f32(pt,a00); a0[1]+=svaddv_f32(pt,a01); a0[2]+=svaddv_f32(pt,a02); a0[3]+=svaddv_f32(pt,a03);
    a1[0]+=svaddv_f32(pt,a10); a1[1]+=svaddv_f32(pt,a11); a1[2]+=svaddv_f32(pt,a12); a1[3]+=svaddv_f32(pt,a13);
    a2[0]+=svaddv_f32(pt,a20); a2[1]+=svaddv_f32(pt,a21); a2[2]+=svaddv_f32(pt,a22); a2[3]+=svaddv_f32(pt,a23);
}
#endif
static inline void glm5_rmsnorm_gemma(float*out,const float*x,const uint16_t*w,int n,float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float inv=(float)(1.0/sqrt(ss/n+eps)); for(int i=0;i<n;i++) out[i]=x[i]*inv*glm5_bf2f(w[i]);
}
static inline void glm5_rmsnorm_head(float*v,const uint16_t*w,int n,float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)v[i]*v[i];
    float inv=(float)(1.0/sqrt(ss/n+eps)); for(int i=0;i<n;i++) v[i]=v[i]*inv*glm5_bf2f(w[i]);
}
/* SwiGLU-OAI (GPT-OSS "swigluoai"): glu(gate)·(up+1), gate clamped <=lim, up clamped
 * to [-lim,lim]. glu(g)=g·sigmoid(alpha·g). The (up+1) term is part of the OAI variant. */
static inline float glm5_swiglu_oai(float gate,float up,float alpha,float lim){
    (void)alpha; (void)lim;
    return gate/(1.0f+expf(-gate))*up;
}
static inline void glm5_rope_head(float*v,const float*cosp,const float*sinp,int rotary_dim){
    int half=rotary_dim/2;
    for(int k=0;k<half;k++){ float c=cosp[k],s=sinp[k],a=v[k],b=v[k+half]; v[k]=a*c-b*s; v[k+half]=a*s+b*c; }
}
static inline void glm5_rope_interleaved(float*v,const float*cosp,const float*sinp,int rotary_dim){
    int half=rotary_dim/2;
    for(int k=0;k<half;k++){ float c=cosp[k],s=sinp[k],a=v[2*k],b=v[2*k+1]; v[2*k]=a*c-b*s; v[2*k+1]=a*s+b*c; }
}

/* ===================== pinned spin pool (EXPERIMENTAL, GLM5_POOL=1, default OFF) ========
 * Intent: beat the OpenMP 48-thread cross-CMG fork-join regression with a persistent
 * pinned spin pool (pin within the job cpuset = cores 12-59 on A64FX; dispatch via an
 * atomic seq, release/acquire ordering; disjoint row-slices -> bit-identical to serial).
 * STATUS (2026-06-15): NOT working — node-dependent hangs and no speedup even when it
 * runs (OpenMP-12 stays the practical best, ~7x). Root cause undiagnosed (instrumentation
 * hung). Reaching full-48 likely needs NUMA-interleaved weights (each CMG reads local
 * HBM) + pool-stability work. Default is OpenMP (glm5_g_pool stays NULL unless GLM5_POOL=1). */
#define GLM5_POOL_MAXT 64
struct glm5_pool {   /* glm5.h forward-declares `typedef struct glm5_pool glm5_pool;` */
    void (*fn)(void*,int,int); void *arg; int nthr;
    pthread_t th[GLM5_POOL_MAXT]; int core[GLM5_POOL_MAXT];
    _Atomic long seq; _Atomic long wdone[GLM5_POOL_MAXT]; volatile int stop;
};
static glm5_pool *glm5_g_pool=NULL;   /* one model per process; matvecs dispatch here */
static int glm5_pool_dbg=0;
/* GLM5_DUMMY: idealized-compute ceiling. matvecs STREAM the weight bytes at full HBM BW
 * (touch every cache line, no bf16->f32 widening, no FMA) -> models the actual memory
 * read with compute removed; comm (ar_cb) + dispatch stay real. Reveals the
 * comm+mem+dispatch floor = the practical tok/s ceiling if matvec were BW-perfect. */
static int glm5_dummy=0;
static _Atomic long glm5_dbg_disp=0, glm5_dbg_main=0, glm5_dbg_wrk=0;

static inline void glm5_cpu_relax(void){ __asm__ __volatile__("yield":::"memory"); }
static void glm5_pin(int core){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(core,&s); sched_setaffinity(0,sizeof s,&s); }
typedef struct { glm5_pool *p; int tid; } glm5_wctx;
static void* glm5_worker(void *a){
    glm5_wctx *w=a; glm5_pool *p=w->p; int tid=w->tid; glm5_pin(p->core[tid]); long last=0;
    if(glm5_pool_dbg) fprintf(stderr,"[pool] worker %d started on core %d\n",tid,p->core[tid]);
    for(;;){
        while(atomic_load_explicit(&p->seq,memory_order_acquire)==last && !p->stop) glm5_cpu_relax();
        if(p->stop) break; last=atomic_load_explicit(&p->seq,memory_order_acquire);
        p->fn(p->arg,tid,p->nthr);
        atomic_store_explicit(&p->wdone[tid],last,memory_order_release);
    }
    return NULL;
}
static glm5_pool* glm5_pool_create(int nthr){
    if(nthr<1)nthr=1; if(nthr>GLM5_POOL_MAXT)nthr=GLM5_POOL_MAXT;
    { const char*e=getenv("GLM5_POOL_DEBUG"); glm5_pool_dbg=(e&&*e)?atoi(e):0; }
    /* pin within the job's actual cpuset (Fugaku: cores 12-59, not 0-N); cpus[] are
     * the allowed cores in ascending order -> tid t pins to cpus[t]. On A64FX the 48
     * compute cores group as 12 per CMG, so ascending order fills CMG0 first. */
    cpu_set_t allowed; CPU_ZERO(&allowed);
    int cpus[512], nc=0;
    if(sched_getaffinity(0,sizeof allowed,&allowed)==0)
        for(int c=0;c<512 && nc<GLM5_POOL_MAXT;c++) if(CPU_ISSET(c,&allowed)) cpus[nc++]=c;
    if(nc<1){ for(int c=0;c<GLM5_POOL_MAXT;c++) cpus[c]=c; nc=GLM5_POOL_MAXT; }
    if(nthr>nc) nthr=nc;
    glm5_pool *p=glm5_acalloc(1,sizeof *p); p->nthr=nthr; atomic_store(&p->seq,0); p->stop=0;
    for(int t=0;t<nthr;t++){ p->core[t]=cpus[t]; atomic_store(&p->wdone[t],0); }
    glm5_pin(p->core[0]);
    for(int t=1;t<nthr;t++){ glm5_wctx *w=glm5_amalloc(sizeof *w); w->p=p; w->tid=t;
        int rc=pthread_create(&p->th[t],NULL,glm5_worker,w);
        if(glm5_pool_dbg) fprintf(stderr,"[pool] create worker %d rc=%d\n",t,rc); }
    if(glm5_pool_dbg) fprintf(stderr,"[pool] created nthr=%d\n",nthr);
    return p;
}
static void glm5_pool_run(glm5_pool *p, void(*fn)(void*,int,int), void *arg){
    if(!p || p->nthr<=1){ fn(arg,0,1); return; }
    if(glm5_pool_dbg) atomic_fetch_add_explicit(&glm5_dbg_disp,1,memory_order_relaxed);
    p->fn=fn; p->arg=arg;
    long s=atomic_load_explicit(&p->seq,memory_order_relaxed)+1;
    atomic_store_explicit(&p->seq,s,memory_order_release);   /* dispatch (publishes fn/arg) */
    fn(arg,0,p->nthr);                                       /* main = tid 0 */
    for(int t=1;t<p->nthr;t++) while(atomic_load_explicit(&p->wdone[t],memory_order_acquire)<s) glm5_cpu_relax();
}
static void glm5_pool_destroy(glm5_pool *p){
    if(!p) return;
    if(glm5_pool_dbg) fprintf(stderr,"[pool] nthr=%d dispatches=%ld main_blocks=%ld worker_blocks=%ld\n",
                            p->nthr,(long)glm5_dbg_disp,(long)glm5_dbg_main,(long)glm5_dbg_wrk);
    p->stop=1; atomic_fetch_add(&p->seq,1);
    for(int t=1;t<p->nthr;t++) pthread_join(p->th[t],NULL);
    glm5_afree(p);
}

/* matvec worker: y[rows] = W[rows,cols](bf16) . x[cols], rows partitioned by 8-blocks */
typedef struct { float *y; const uint16_t *W; const float *x; int rows, cols; } glm5_mvjob;
static void glm5_mv_worker(void *a,int tid,int nthr){
    glm5_mvjob *j=a; int nb=j->rows/8, per=(nb+nthr-1)/nthr, b0=tid*per, b1=b0+per; if(b1>nb)b1=nb;
    for(int bi=b0;bi<b1;bi++){ int r=bi*8; const uint16_t *b=j->W+(size_t)r*j->cols;
        matvec_bf16_8row(j->y+r,b,b+j->cols,b+2*(size_t)j->cols,b+3*(size_t)j->cols,
                         b+4*(size_t)j->cols,b+5*(size_t)j->cols,b+6*(size_t)j->cols,b+7*(size_t)j->cols,j->x,j->cols); }
    if(tid==0) for(int r=nb*8;r<j->rows;r++) j->y[r]=vec_dot_bf16_f32(j->W+(size_t)r*j->cols,j->x,j->cols);
    if(glm5_pool_dbg) atomic_fetch_add_explicit(tid==0?&glm5_dbg_main:&glm5_dbg_wrk,(long)(b1-b0),memory_order_relaxed);
}

/* y[rows] = W[rows,cols](bf16) . x[cols] over 8-row blocks (disjoint outputs -> bit-
 * identical to serial). Default parallelism is OpenMP (validated ~7x @12 threads = 1 CMG;
 * the cross-CMG fork-join wall caps it there). The experimental pinned pool (GLM5_POOL=1)
 * is selected when glm5_g_pool is set. */
static void glm5_mv_bf16(float*restrict y,const uint16_t*W,const float*x,int rows,int cols){
    if(glm5_dummy){   /* stream every cache line of W at full BW; no widen/FMA (compute idealized) */
        double acc=0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
        for(int r=0;r<rows;r++){ const uint16_t*w=W+(size_t)r*cols; uint32_t s=0; for(int i=0;i<cols;i+=32) s+=w[i]; acc+=s; }
        float v=(float)(((long)acc)&1)*1e-30f; for(int r=0;r<rows;r++) y[r]=v;   /* defeat DCE, ~0 */
        return;
    }
    if(glm5_g_pool && rows>=GLM5_PAR_MIN){ glm5_mvjob j={y,(const uint16_t*)W,x,rows,cols}; glm5_pool_run(glm5_g_pool,glm5_mv_worker,&j); return; }
    int nb=rows/8;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){ int r=bi*8; const uint16_t*b=W+(size_t)r*cols;
        matvec_bf16_8row(y+r,b,b+cols,b+2*(size_t)cols,b+3*(size_t)cols,
                         b+4*(size_t)cols,b+5*(size_t)cols,b+6*(size_t)cols,b+7*(size_t)cols,x,cols); }
    for(int r=nb*8;r<rows;r++) y[r]=vec_dot_bf16_f32(W+(size_t)r*cols,x,cols);
}

static inline float glm5_dot_mxfp8_f32scale_row(const uint8_t*w,const float*s,const float*x,int cols,const uint32_t*lut){
    double a=0;
    for(int b=0;b<cols;b+=128){
        float sc=s[b/128];
        int e=b+128<cols?b+128:cols;
        for(int c=b;c<e;c++){ float wf; uint32_t u=lut[w[c]]; memcpy(&wf,&u,4); a+=(double)wf*sc*x[c]; }
    }
    return (float)a;
}

#if defined(__ARM_FEATURE_SVE)
static inline void glm5_matvec_mxfp8_f32scale_8row(float*restrict dst,
        const uint8_t*w0,const uint8_t*w1,const uint8_t*w2,const uint8_t*w3,
        const uint8_t*w4,const uint8_t*w5,const uint8_t*w6,const uint8_t*w7,
        const float*s0,const float*s1,const float*s2,const float*s3,
        const float*s4,const float*s5,const float*s6,const float*s7,
        const float*x,int cols,const uint32_t*lut){
    svfloat32_t a0=svdup_f32(0.f),a1=svdup_f32(0.f),a2=svdup_f32(0.f),a3=svdup_f32(0.f);
    svfloat32_t a4=svdup_f32(0.f),a5=svdup_f32(0.f),a6=svdup_f32(0.f),a7=svdup_f32(0.f);
    int vl=(int)svcntw();
    #define GLM5_FP8_F32S_ROW(WP,SC,ACC) do{ \
        svuint32_t idx=svld1ub_u32(pg,&(WP)[c]); \
        svfloat32_t wv=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,lut,idx)); \
        ACC=svmla_x(pg,ACC,wv,svmul_n_f32_x(pg,xv,(SC))); \
    }while(0)
    for(int b=0;b<cols;b+=128){
        int bend=b+128<cols?b+128:cols, blk=b/128;
        float c0=s0[blk],c1=s1[blk],c2=s2[blk],c3=s3[blk];
        float c4=s4[blk],c5=s5[blk],c6=s6[blk],c7=s7[blk];
        for(int c=b;c<bend;c+=vl){
            svbool_t pg=svwhilelt_b32(c,bend);
            svfloat32_t xv=svld1(pg,&x[c]);
            GLM5_FP8_F32S_ROW(w0,c0,a0); GLM5_FP8_F32S_ROW(w1,c1,a1);
            GLM5_FP8_F32S_ROW(w2,c2,a2); GLM5_FP8_F32S_ROW(w3,c3,a3);
            GLM5_FP8_F32S_ROW(w4,c4,a4); GLM5_FP8_F32S_ROW(w5,c5,a5);
            GLM5_FP8_F32S_ROW(w6,c6,a6); GLM5_FP8_F32S_ROW(w7,c7,a7);
        }
    }
    #undef GLM5_FP8_F32S_ROW
    svbool_t pt=svptrue_b32();
    dst[0]=svaddv_f32(pt,a0); dst[1]=svaddv_f32(pt,a1); dst[2]=svaddv_f32(pt,a2); dst[3]=svaddv_f32(pt,a3);
    dst[4]=svaddv_f32(pt,a4); dst[5]=svaddv_f32(pt,a5); dst[6]=svaddv_f32(pt,a6); dst[7]=svaddv_f32(pt,a7);
}
#endif

static inline void glm5_mxfp8_f32scale_decode_row_bf16(uint16_t*restrict dst, const uint8_t*restrict w,
        const float*restrict s, int col0, int kl, const uint32_t*restrict lut){
#if defined(__ARM_FEATURE_SVE)
    const int vl=(int)svcntw();
    for(int b=0;b<kl;b+=128){
        int absc=col0+b;
        int bend=b+128<kl?b+128:kl;
        float sc=s[absc/128];
        for(int c=b;c<bend;c+=vl){
            svbool_t pg=svwhilelt_b32(c,bend);
            svuint32_t idx=svld1ub_u32(pg,&w[c]);
            svfloat32_t wf=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,lut,idx));
            svuint32_t bits=svreinterpret_u32_f32(svmul_n_f32_x(pg,wf,sc));
            svuint32_t lsb=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1);
            svuint32_t rnd=svadd_u32_x(pg,bits,svadd_n_u32_x(pg,lsb,0x7fffu));
            svst1h_u32(pg,&dst[c],svlsr_n_u32_x(pg,rnd,16));
        }
    }
#else
    for(int u=0;u<kl;u++){
        float wf; uint32_t bits=lut[w[u]]; memcpy(&wf,&bits,4);
        dst[u]=glm5_f2bf(wf*s[(col0+u)/128]);
    }
#endif
}

/* INT8 decode one row's [col0,col0+kl) -> bf16 tile: dst = bf16((byte-128)*scale[(col0+c)/gs]).
 * Unlike FP8 (exact in bf16), int8*scale rounds to bf16 (round-to-nearest-even), matching the
 * w8a16 reference. col0 is relative to the first copied scale group, so the first
 * local group may be partial for an unaligned TP column shard. */
static inline void glm5_int8_decode_row_bf16(uint16_t*restrict dst, const uint8_t*restrict w,
        const float*restrict s, int gs, int col0, int kl){
#if defined(__ARM_FEATURE_SVE)
    const int vl=(int)svcntw();
    for(int b=0;b<kl;){
        int blk=(col0+b)/gs, bend=(blk+1)*gs-col0; if(bend>kl)bend=kl; float sc=s[blk];
        for(int c=b;c<bend;c+=vl){
            svbool_t pg=svwhilelt_b32(c,bend);
            svuint32_t bz=svld1ub_u32(pg,&w[c]);
            svfloat32_t wf=svcvt_f32_s32_x(pg, svsub_n_s32_x(pg, svreinterpret_s32_u32(bz),128));
            svuint32_t bits=svreinterpret_u32_f32(svmul_n_f32_x(pg,wf,sc));
            svuint32_t lsb=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1);
            svuint32_t rnd=svadd_u32_x(pg,bits,svadd_n_u32_x(pg,lsb,0x7fffu));
            svst1h_u32(pg,&dst[c],svlsr_n_u32_x(pg,rnd,16));
        }
        b=bend;
    }
#else
    for(int u=0;u<kl;u++) dst[u]=glm5_f2bf((float)((int)w[u]-128)*s[(col0+u)/gs]);
#endif
}

/* GLM5.2 FP8 matvec: F8_E4M3 weights with F32 scale_inv per 128x128 block.
 * The loader expands each block-row scale to one local row, so the kernel can
 * group eight output rows while preserving arbitrary TP row shards. */
static void glm5_mv_mxfp8(glm5_model*m, float*restrict y, const uint8_t*W, const uint8_t*S, const float*x, int rows, int cols){
    const uint32_t*lut=m->fp8_lut; const float*Sc=(const float*)S; int sb=(cols+127)/128;
    if(glm5_dummy){ double acc=0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
        for(int r=0;r<rows;r++){ const uint8_t*w=W+(size_t)r*cols; uint32_t s=0; for(int i=0;i<cols;i+=64) s+=w[i]; acc+=s; }
        float v=(float)(((long)acc)&1)*1e-30f; for(int r=0;r<rows;r++) y[r]=v; return; }
#if defined(__ARM_FEATURE_SVE)
    static int use_8row=-1;
    if(use_8row<0) use_8row=glm5_envi("GLM5_FP8_8ROW",1);
    if(use_8row){
        int nb=rows/8;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
        for(int bi=0;bi<nb;bi++){
            int r=bi*8;
            const uint8_t*w=W+(size_t)r*cols;
            const float*s=Sc+(size_t)r*sb;
            glm5_matvec_mxfp8_f32scale_8row(y+r,
                w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
                w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
                s,s+sb,s+2*(size_t)sb,s+3*(size_t)sb,
                s+4*(size_t)sb,s+5*(size_t)sb,s+6*(size_t)sb,s+7*(size_t)sb,
                x,cols,lut);
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
        for(int r=nb*8;r<rows;r++)
            y[r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,Sc+(size_t)r*sb,x,cols,lut);
        return;
    }
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
    for(int r=0;r<rows;r++)
        y[r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,Sc+(size_t)r*sb,x,cols,lut);
}
/* GLM5.2 INT8 matvec: offset-binary int8 weights (q=byte-128) with per-row F32 group scale.
 * gs = scale group size in cols (128 group / =cols per-channel); sb = groups per row. */
static void glm5_mv_int8(glm5_model*m, float*restrict y, const uint8_t*W, const float*S, int gs, int qg0, const float*x, int rows, int cols){
    int sb=(qg0+cols+gs-1)/gs;
    if(glm5_dummy){ for(int r=0;r<rows;r++) y[r]=0.f; (void)m; return; }
#if defined(__ARM_FEATURE_SVE)
    static int use_8row=-1;
    if(use_8row<0) use_8row=glm5_envi("GLM5_INT8_8ROW",1);
    if(use_8row){
        int nb=rows/8;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
        for(int bi=0;bi<nb;bi++){
            int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb;
            glm5_matvec_int8_8row(y+r,
                w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
                w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
                s,s+sb,s+2*(size_t)sb,s+3*(size_t)sb,
                s+4*(size_t)sb,s+5*(size_t)sb,s+6*(size_t)sb,s+7*(size_t)sb,
                gs,qg0,x,cols);
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
        for(int r=nb*8;r<rows;r++) y[r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,qg0,x,cols);
        return;
    }
#endif
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
    for(int r=0;r<rows;r++) y[r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,qg0,x,cols);
}
/* GLM5.2 INT8 w8a8 SDOT matvec (M=1 decode): dynamically quantize the activation to int8 ONCE
 * (per-vector symmetric absmax), then contract with SVE SDOT instead of the w8a16 per-byte
 * u8->f32 convert. Eliminates the convert-throughput bottleneck of glm5_mv_int8 for M=1. Lossier
 * (activation rounded to int8) -> opt-in via GLM5_MV_SDOT. */
static void glm5_mv_int8_sdot(glm5_model*m, float*restrict y, const uint8_t*W, const float*S, int gs, int qg0, const float*x, int rows, int cols){
    int sb=(qg0+cols+gs-1)/gs;
    if(glm5_dummy){ for(int r=0;r<rows;r++) y[r]=0.f; (void)m; return; }
#if defined(__ARM_FEATURE_SVE)
    int8_t *xq=(int8_t*)malloc((size_t)cols);
    if(!xq){ glm5_mv_int8(m,y,W,S,gs,qg0,x,rows,cols); return; }
    float amax=1e-20f;
    for(int c=0;c<cols;c++){ float a=fabsf(x[c]); if(a>amax)amax=a; }
    float inv=127.0f/amax, xsc=amax/127.0f;
    for(int c=0;c<cols;c++){ int v=(int)lrintf(x[c]*inv); v=v>127?127:(v<-127?-127:v); xq[c]=(int8_t)v; }
    int nb=rows/8;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb; float tmp[8];
        glm5_matvec_int8_sdot_8row(tmp,
            w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
            w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
            s,s+sb,s+2*(size_t)sb,s+3*(size_t)sb,
            s+4*(size_t)sb,s+5*(size_t)sb,s+6*(size_t)sb,s+7*(size_t)sb,
            gs,qg0,xq,cols);
        for(int j=0;j<8;j++) y[r+j]=xsc*tmp[j];
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
    for(int r=nb*8;r<rows;r++) y[r]=xsc*glm5_dot_int8_sdot_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,qg0,xq,cols);
    free(xq);
    return;
#else
    glm5_mv_int8(m,y,W,S,gs,qg0,x,rows,cols);
#endif
}
/* GLM5.2 INT8 w8a16-MIMIC via int16 SDOT (M=1 decode): quantize the activation to int16 (near-
 * lossless, ~3e-5/elem) instead of int8, then contract with svdot_s64. Matches the w8a16 reference to
 * ~1e-4 (unlike w8a8's ~8% on outlier-heavy activations) while still avoiding the per-byte f32 convert.
 * The QuantTrio GLM-5.2-Int8 checkpoint IS w8a16, so this is the accuracy-preserving fast path. */
/* Per-group int16 activation sums for the bias-fold widen: xg[blk] = sum of xq over the columns
 * of quant-group blk (matching the (gs,qg0) grouping the SDOT kernels use). Computed once per
 * activation vector and reused across every weight row. See glm5_matvec_int16sdot_8row. */
static inline void glm5_i16_group_sums(int64_t*restrict xg, const int16_t*restrict xq, int gs, int qg0, int cols){
    for(int b=0;b<cols;){ int blk=(qg0+b)/gs, bend=(blk+1)*gs-qg0; if(bend>cols)bend=cols;
        long s=0; for(int c=b;c<bend;c++) s+=xq[c]; xg[blk]=s; b=bend; }
}
static void glm5_mv_int16_sdot(glm5_model*m, float*restrict y, const uint8_t*W, const float*S, int gs, int qg0, const float*x, int rows, int cols){
    int sb=(qg0+cols+gs-1)/gs;
    if(glm5_dummy){ for(int r=0;r<rows;r++) y[r]=0.f; (void)m; return; }
#if defined(__ARM_FEATURE_SVE)
    int16_t *xq=(int16_t*)malloc((size_t)cols*2);
    int64_t *xg=(int64_t*)malloc((size_t)sb*sizeof(int64_t));
    if(!xq||!xg){ free(xq); free(xg); glm5_mv_int8(m,y,W,S,gs,qg0,x,rows,cols); return; }
    float amax=1e-20f;
    for(int c=0;c<cols;c++){ float a=fabsf(x[c]); if(a>amax)amax=a; }
    float inv=32767.0f/amax, xsc=amax/32767.0f;
    for(int c=0;c<cols;c++){ int v=(int)lrintf(x[c]*inv); v=v>32767?32767:(v<-32767?-32767:v); xq[c]=(int16_t)v; }
    glm5_i16_group_sums(xg,xq,gs,qg0,cols);
    int nb=rows/8;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb; float tmp[8];
        glm5_matvec_int16sdot_8row(tmp,
            w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
            w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
            s,s+sb,s+2*(size_t)sb,s+3*(size_t)sb,
            s+4*(size_t)sb,s+5*(size_t)sb,s+6*(size_t)sb,s+7*(size_t)sb,
            gs,qg0,xq,xg,cols);
        for(int j=0;j<8;j++) y[r+j]=xsc*tmp[j];
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
    for(int r=nb*8;r<rows;r++) y[r]=xsc*glm5_dot_int16sdot_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,qg0,xq,cols);
    free(xq); free(xg);
    return;
#else
    glm5_mv_int8(m,y,W,S,gs,qg0,x,rows,cols);
#endif
}
static inline float glm5_dot_f32_f32(const float *a,const float *b,int n){
#if defined(__ARM_FEATURE_SVE)
    svfloat32_t acc=svdup_f32(0.f);
    for(int i=0;i<n;i+=(int)svcntw()){
        svbool_t pg=svwhilelt_b32(i,n);
        acc=svmla_f32_m(pg,acc,svld1(pg,a+i),svld1(pg,b+i));
    }
    return svaddv_f32(svptrue_b32(),acc);
#else
    double acc=0.0; for(int i=0;i<n;i++) acc+=(double)a[i]*b[i]; return (float)acc;
#endif
}

/* Mixed-IQ decode kernel.  llama.cpp normally pairs the IQ blocks with a
 * Q8_K activation.  On A64FX we use a per-256-element int16 activation
 * instead: it is effectively lossless for inference activations while still
 * allowing SVE svdot_s64 to contract the small integer grids directly.  The
 * IQ block scale and sub-block scale are folded after the integer dot. */
typedef struct { float d; int16_t q[256]; } glm5_iq_a16_block;

static inline int64_t glm5_iq_dot_i16(const int16_t*a,const int16_t*b,int n){
#if defined(__ARM_FEATURE_SVE)
    svint64_t acc=svdup_s64(0);
    for(int i=0;i<n;i+=(int)svcnth()){
        svbool_t pg=svwhilelt_b16(i,n);
        acc=svdot_s64(acc,svld1_s16(pg,a+i),svld1_s16(pg,b+i));
    }
    return svaddv_s64(svptrue_b64(),acc);
#else
    int64_t acc=0; for(int i=0;i<n;i++) acc+=(int32_t)a[i]*b[i]; return acc;
#endif
}

static void glm5_iq_quant_a16(glm5_iq_a16_block*q,const float*x,int cols){
    for(int b=0;b<cols/256;b++){
        float amax=0.f;
        for(int j=0;j<256;j++){ float a=fabsf(x[256*b+j]); if(a>amax)amax=a; }
        q[b].d=amax>0.f?amax/32767.f:0.f;
        float inv=amax>0.f?32767.f/amax:0.f;
        for(int j=0;j<256;j++){
            int v=(int)lrintf(x[256*b+j]*inv);
            v=v>32767?32767:(v<-32767?-32767:v);
            q[b].q[j]=(int16_t)v;
        }
    }
}

static float glm5_iq2_xs_a16_row(const block_iq2_xs*w,const glm5_iq_a16_block*x,int nb){
    double sum=0.0; int16_t qw[32];
    for(int b=0;b<nb;b++){
        int64_t ibsum=0;
        for(int ib=0;ib<8;ib++){
            for(int l=0;l<4;l++){
                uint16_t code=w[b].qs[4*ib+l];
                const uint8_t*grid=(const uint8_t*)(iq2xs_grid+(code&511));
                uint8_t signs=ksigns_iq2xs[code>>9];
                for(int j=0;j<8;j++) qw[8*l+j]=(int16_t)(
                    (signs&kmask_iq2xs[j])?-(int)grid[j]:(int)grid[j]);
            }
            int ls1=2*(w[b].scales[ib]&15)+1;
            int ls2=2*(w[b].scales[ib]>>4)+1;
            const int16_t*xq=x[b].q+32*ib;
            ibsum+=glm5_iq_dot_i16(qw,xq,16)*ls1;
            ibsum+=glm5_iq_dot_i16(qw+16,xq+16,16)*ls2;
        }
        sum+=(double)ggml_fp16_to_fp32(w[b].d)*x[b].d*(double)ibsum*0.125;
    }
    return (float)sum;
}

static float glm5_iq3_xxs_a16_row(const block_iq3_xxs*w,const glm5_iq_a16_block*x,int nb){
    double sum=0.0; int16_t qw[32];
    for(int b=0;b<nb;b++){
        const uint8_t*q3=w[b].qs;
        const uint8_t*gas=w[b].qs+64;
        int64_t ibsum=0;
        for(int ib=0;ib<8;ib++){
            uint32_t aux; memcpy(&aux,gas+4*ib,4);
            for(int l=0;l<4;l++){
                const uint8_t*g1=(const uint8_t*)(iq3xxs_grid+q3[8*ib+2*l]);
                const uint8_t*g2=(const uint8_t*)(iq3xxs_grid+q3[8*ib+2*l+1]);
                uint8_t signs=ksigns_iq2xs[(aux>>(7*l))&127];
                for(int j=0;j<4;j++){
                    qw[8*l+j]=(int16_t)((signs&kmask_iq2xs[j])?-(int)g1[j]:(int)g1[j]);
                    qw[8*l+j+4]=(int16_t)((signs&kmask_iq2xs[j+4])?-(int)g2[j]:(int)g2[j]);
                }
            }
            ibsum+=glm5_iq_dot_i16(qw,x[b].q+32*ib,32)*(2*(int)(aux>>28)+1);
        }
        sum+=(double)ggml_fp16_to_fp32(w[b].d)*x[b].d*(double)ibsum*0.25;
    }
    return (float)sum;
}

static float glm5_iq4_xs_a16_row(const block_iq4_xs*w,const glm5_iq_a16_block*x,int nb){
    double sum=0.0; int16_t qw[32];
    for(int b=0;b<nb;b++){
        const uint8_t*qs=w[b].qs; uint16_t h=w[b].scales_h;
        int64_t ibsum=0;
        for(int ib=0;ib<8;ib++){
            for(int j=0;j<16;j++){
                qw[j]=kvalues_iq4nl[qs[j]&15];
                qw[j+16]=kvalues_iq4nl[qs[j]>>4];
            }
            int ls=((w[b].scales_l[ib/2]>>(4*(ib&1)))&15)|((h&3)<<4);
            h>>=2;
            ibsum+=glm5_iq_dot_i16(qw,x[b].q+32*ib,32)*(ls-32);
            qs+=16;
        }
        sum+=(double)ggml_fp16_to_fp32(w[b].d)*x[b].d*(double)ibsum;
    }
    return (float)sum;
}

static int glm5_mv_iq_a16(float*restrict y,const glm5_tensor*t,const float*x,int rows,int cols){
    if(cols%256) return -1;
    if(t->type!=GLM5_IQ2_XS&&t->type!=GLM5_IQ3_XXS&&t->type!=GLM5_IQ4_XS) return -1;
    int nb=cols/256;
    glm5_iq_a16_block*xq=malloc((size_t)nb*sizeof(*xq));
    if(!xq) return -1;
    glm5_iq_quant_a16(xq,x,cols);
    size_t rb=dequant_row_size((uint32_t)t->type,cols);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows>=GLM5_PAR_MIN)
#endif
    for(int r=0;r<rows;r++){
        const uint8_t*w=(const uint8_t*)t->w+(size_t)r*rb;
        if(t->type==GLM5_IQ2_XS) y[r]=glm5_iq2_xs_a16_row((const block_iq2_xs*)w,xq,nb);
        else if(t->type==GLM5_IQ3_XXS) y[r]=glm5_iq3_xxs_a16_row((const block_iq3_xxs*)w,xq,nb);
        else y[r]=glm5_iq4_xs_a16_row((const block_iq4_xs*)w,xq,nb);
    }
    free(xq); return 0;
}
/* Source-faithful mixed-IQ baseline.  A row is dequantized into
 * thread-local scratch and immediately contracted, so compressed expert
 * weights never expand persistently in HBM. */
static void glm5_mv_ggml(float*restrict y,const glm5_tensor*t,const float*x,int rows,int cols){
    const size_t rb=dequant_row_size((uint32_t)t->type,cols);
    if(!rb){ for(int r=0;r<rows;r++) y[r]=NAN; return; }
#ifdef _OPENMP
    #pragma omp parallel if(rows>=GLM5_PAR_MIN)
    {
        float*tmp=malloc((size_t)cols*4);
        #pragma omp for schedule(static)
        for(int r=0;r<rows;r++){
            const uint8_t*src=(const uint8_t*)t->w+(size_t)r*rb;
            y[r]=(!tmp||dequant_row((uint32_t)t->type,src,tmp,cols))?NAN:
                 glm5_dot_f32_f32(tmp,x,cols);
        }
        free(tmp);
    }
#else
    float*tmp=malloc((size_t)cols*4);
    for(int r=0;r<rows;r++){
        const uint8_t*src=(const uint8_t*)t->w+(size_t)r*rb;
        y[r]=(!tmp||dequant_row((uint32_t)t->type,src,tmp,cols))?NAN:
             glm5_dot_f32_f32(tmp,x,cols);
    }
    free(tmp);
#endif
}
/* matvec dispatch by weight type (bf16 / MXFP8 / INT8 / GGML IQ) */
static void glm5_mv(glm5_model*m, float*restrict y, const glm5_tensor*t, const float*x, int rows, int cols){
    if(t->type==GLM5_MXFP8) glm5_mv_mxfp8(m,y,(const uint8_t*)t->w,t->scale,x,rows,cols);
    else if(t->type==GLM5_INT8){
        /* GLM5_MV_SDOT: 0=w8a16 f32 (shipped), 1=w8a8 int8-SDOT (fast, lossy ~8%),
         * 2=w8a16-mimic int16-SDOT (fast + accurate, matches the w8a16 checkpoint). Default off. */
        static int mvsd=-2;
        if(mvsd==-2) mvsd=glm5_envi("GLM5_MV_SDOT",0);
        if(mvsd==2) glm5_mv_int16_sdot(m,y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,x,rows,cols);
        else if(mvsd==1) glm5_mv_int8_sdot(m,y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,x,rows,cols);
        else glm5_mv_int8(m,y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,x,rows,cols);
    }
    else if(glm5_is_ggml_q(t->type)){
        static int iqref=-1,iqcheck=-1;
        /* The row-dequant path is currently faster at production thread count;
         * set GLM5_IQ_REF=0 to exercise the accuracy-preserving A16 SDOT path. */
        if(iqref<0) iqref=glm5_envi("GLM5_IQ_REF",1);
        if(iqcheck<0) iqcheck=glm5_envi("GLM5_IQ_CHECK",0);
        int fallback=iqref||glm5_mv_iq_a16(y,t,x,rows,cols);
        if(fallback) glm5_mv_ggml(y,t,x,rows,cols);
        else if(iqcheck){
            /* First-call accuracy gate against source-faithful F32 dequant. */
            static atomic_int checked=0;
            if(atomic_exchange(&checked,1)==0){
                float*ref=malloc((size_t)rows*sizeof(float));
                if(ref){
                    glm5_mv_ggml(ref,t,x,rows,cols);
                    double e2=0,r2=0,mx=0;
                    for(int r=0;r<rows;r++){
                        double e=(double)y[r]-ref[r],a=fabs(e);
                        e2+=e*e; r2+=(double)ref[r]*ref[r]; if(a>mx)mx=a;
                    }
                    double rel=sqrt(e2/(r2+1e-30));
                    fprintf(stderr,"[glm52-iq] a16/ref rows=%d cols=%d rel_l2=%.3e max_abs=%.3e %s\n",
                            rows,cols,rel,mx,(isfinite(rel)&&rel<5e-4)?"OK":"FAIL");
                    if(!isfinite(rel)||rel>=5e-4) abort();
                    free(ref);
                }
            }
        }
    }
    else glm5_mv_bf16(y,(const uint16_t*)t->w,x,rows,cols);
}

/* ===================== arena fit estimate (synth uses malloc) ===================== */
static size_t glm5_arena_size(const glm5_config*c,int ep_rank,int ep_size){
    int H=c->hidden, QD=glm5_q_dim(c), KVD=glm5_kv_dim(c);
    int no=glm5_n_owned(c->n_experts,ep_rank,ep_size);
    int qh0,qh1; glm5_shard_heads(c->n_heads,ep_rank,ep_size,&qh0,&qh1);
    int tp=glm5_envi("GLM5_TP",0);
    int tp_attn=glm5_envi("GLM5_TP_ATTN",tp), tp_sh=glm5_envi("GLM5_TP_SHARED",tp);
    int tp_ffn=glm5_envi("GLM5_TP_FFN",tp), tp_head=glm5_envi("GLM5_TP_HEAD",tp), tp_emb=glm5_envi("GLM5_TP_EMBED",tp);
    int qrows = tp_attn ? (qh1-qh0)*c->head_dim : QD;
    int sh_r0_est=0, shrows=c->moe_inter; if(tp_sh) glm5_shard_shared_inter(c->moe_inter,ep_rank,ep_size,&sh_r0_est,&shrows);
    int ffrows = tp_ffn ? (c->dense_inter+ep_size-1)/ep_size : c->dense_inter;
    int hrows = tp_head ? (c->vocab+ep_size-1)/ep_size : c->vocab;
    int erows = tp_emb ? (c->vocab+ep_size-1)/ep_size : c->vocab;
    int int4_kv=glm5_envi("GLM5_INT4_KV",0);
    int cp_on=glm5_envi("GLM5_CP",0) && ep_size>1;
    int cp_nslot=glm5_cp_nslot(c->max_pos,c->msa_block_size,ep_size,cp_on);
    size_t kv_cache = int4_kv ? (size_t)cp_nslot*(KVD/2 + 2) : (size_t)cp_nslot*KVD*2;
    size_t idx_cache = int4_kv ? (size_t)cp_nslot*(c->index_dim/2 + 2) : (size_t)cp_nslot*c->index_dim*2;
    size_t attn = (size_t)qrows*H*2 + 2*(size_t)KVD*H*2 + (size_t)H*qrows*2 + 2*(size_t)H*2 + 2*(size_t)c->head_dim*2;
    attn += kv_cache;
    size_t msa = (size_t)glm5_idx_q_dim(c)*H*2 + (size_t)c->msa_index_dim*H*2 + 2*(size_t)c->msa_index_dim*2 + idx_cache;
    size_t per_moe = attn + msa + (size_t)c->n_experts*H*4 + (size_t)c->n_experts*4
                   + 2*(size_t)shrows*H*2 + (size_t)H*shrows*2
                   + (size_t)no*3*(size_t)c->moe_inter*H*2;
    size_t per_dense = attn + 2*(size_t)ffrows*H*2 + (size_t)H*ffrows*2;
    int n_dense = c->n_dense_layers<c->n_layers?c->n_dense_layers:c->n_layers;
    int n_moe = c->n_layers-n_dense;
    size_t total = (size_t)n_dense*per_dense + (size_t)n_moe*per_moe;
    total += (size_t)erows*H*2 + (size_t)hrows*H*2 + (size_t)H*2;
    total += (size_t)c->max_pos*(c->rotary_dim/2)*4*2;
    return total;
}

/* ===================== synthetic model ===================== */
static void glm5_free_mstream(glm5_model*m);
static void glm5_free(glm5_model*m){
    if(!m) return;
    if(m->ms) glm5_free_mstream(m);
    if(m->pool){ if(glm5_g_pool==m->pool) glm5_g_pool=NULL; glm5_pool_destroy(m->pool); m->pool=NULL; }
    for(int l=0;l<m->cfg.n_layers;l++){
        glm5_layer*L=&m->layers[l];
        if(m->owns_weights){
            glm5_afree(L->input_norm);glm5_afree(L->post_norm);
            /* .scale (MXFP8 e8m0) is NULL for bf16 tensors; glm5_afree() of NULL is a no-op */
            glm5_afree(L->wq_a.w);glm5_afree(L->wq_a.scale);glm5_afree(L->wq_b.w);glm5_afree(L->wq_b.scale);
            glm5_afree(L->wkv_a.w);glm5_afree(L->wkv_a.scale);glm5_afree(L->wkv_b.w);glm5_afree(L->wkv_b.scale);
            glm5_afree(L->wq.w);glm5_afree(L->wq.scale);glm5_afree(L->wk.w);glm5_afree(L->wk.scale);glm5_afree(L->wv.w);glm5_afree(L->wv.scale);
            glm5_afree(L->wo.w);glm5_afree(L->wo.scale);glm5_afree(L->q_a_norm);glm5_afree(L->kv_a_norm);glm5_afree(L->q_norm);glm5_afree(L->k_norm);
        }
        glm5_afree(L->kv_cache);glm5_afree(L->k_cache);glm5_afree(L->v_cache);
        glm5_afree(L->k_q4);glm5_afree(L->v_q4);glm5_afree(L->k_qs);glm5_afree(L->v_qs);glm5_afree(L->idx_q4);glm5_afree(L->idx_qs);
        if(glm5_is_moe(&m->cfg,l)){
            if(m->owns_weights){
                glm5_afree(L->idx_wq_b.w);glm5_afree(L->idx_wq_b.scale);glm5_afree(L->idx_wproj.w);glm5_afree(L->idx_wproj.scale);
                glm5_afree(L->idx_wq.w);glm5_afree(L->idx_wq.scale);glm5_afree(L->idx_wk.w);glm5_afree(L->idx_wk.scale);
                glm5_afree(L->idx_q_norm);glm5_afree(L->idx_k_norm);glm5_afree(L->idx_k_bias);
                glm5_afree(L->gate.w);glm5_afree(L->gate_bias);
                glm5_afree(L->sh_w1.w);glm5_afree(L->sh_w1.scale);glm5_afree(L->sh_w3.w);glm5_afree(L->sh_w3.scale);glm5_afree(L->sh_w2.w);glm5_afree(L->sh_w2.scale);
                if(L->ex_w1){ for(int s=0;s<L->n_owned;s++){ glm5_afree(L->ex_w1[s].w);glm5_afree(L->ex_w1[s].scale);glm5_afree(L->ex_w3[s].w);glm5_afree(L->ex_w3[s].scale);glm5_afree(L->ex_w2[s].w);glm5_afree(L->ex_w2[s].scale);} }
            }
            glm5_afree(L->idx_k_cache);
            if(m->owns_weights){ glm5_afree(L->ex_w1);glm5_afree(L->ex_w3);glm5_afree(L->ex_w2);glm5_afree(L->owned_eid); }
        } else if(m->owns_weights) { glm5_afree(L->ff_gate.w);glm5_afree(L->ff_gate.scale);glm5_afree(L->ff_up.w);glm5_afree(L->ff_up.scale);glm5_afree(L->ff_down.w);glm5_afree(L->ff_down.scale); }
    }
    glm5_afree(m->layers);
    if(m->owns_weights){ glm5_afree(m->embed);glm5_afree(m->head.w);glm5_afree(m->out_norm);glm5_afree(m->rope_cos);glm5_afree(m->rope_sin); }
    glm5_afree(m->s_norm);glm5_afree(m->s_q);glm5_afree(m->s_k);glm5_afree(m->s_v);glm5_afree(m->s_kvb);glm5_afree(m->s_qabs);glm5_afree(m->s_ctx);glm5_afree(m->s_attn);glm5_afree(m->s_o);
    glm5_afree(m->s_idx_q);glm5_afree(m->s_idx_k);glm5_afree(m->s_blk_score);glm5_afree(m->s_blk_sel);glm5_afree(m->s_attn_score);
    glm5_afree(m->s_router);glm5_afree(m->s_shg);glm5_afree(m->s_shu);glm5_afree(m->s_sh);glm5_afree(m->s_moe);
    glm5_afree(m->s_exg);glm5_afree(m->s_exu);glm5_afree(m->s_route);glm5_afree(m->s_ff_g);glm5_afree(m->s_ff_u);glm5_afree(m->s_ff);glm5_afree(m->s_logits);
    glm5_afree(m);
}

/* Context-tiered KV init (call once after ep_rank/ep_size set, before glm5_alloc_kv).
 * Default: a single binary auto-picks the prefill algorithm by context length -- start in Tier A
 * (cp_on=0 bf16, KV replicated per rank, no per-token CP combine) while the un-sharded KV fits a
 * per-rank budget, then transition (glm5_prefill_to_cp) to Tier B (cp_on=1 int4, CP-sharded) at
 * T_cp. If the whole context fits un-sharded, it stays Tier A throughout (no CP overhead).
 * GLM5_CP_THRESHOLD: 0=auto (default), >0=explicit T_cp positions, <0=static (legacy env CP/int4,
 * for forced-CP A/B reference). GLM5_KV_BUDGET_GB tunes the auto budget. */
/* read a /proc/meminfo field (e.g. "MemAvailable") in bytes; 0 if unavailable. */
static long glm5_meminfo_bytes(const char*key){
    FILE*f=fopen("/proc/meminfo","r"); if(!f) return 0;
    char line[256]; size_t kl=strlen(key); long kb=0;
    while(fgets(line,sizeof line,f)){
        if(!strncmp(line,key,kl) && line[kl]==':'){
            const char*p=line+kl+1; while(*p&&(*p<'0'||*p>'9'))p++; kb=atol(p); break;
        }
    }
    fclose(f); return kb*1024L;
}
static void glm5_kv_init(glm5_model*m){
    const glm5_config*c=&m->cfg;
    m->kv_fp16 =glm5_envi("GLM5_KV_FP16",0);
    m->cp_block=c->msa_block_size;             /* CP shard granularity == MSA block (128) */
    int ctx=c->max_pos, thr=glm5_envi("GLM5_CP_THRESHOLD",0);
    if(m->ep_size<=1 || thr<0){                /* static / legacy: env-driven, no tiering */
        m->int4_kv=glm5_envi("GLM5_INT4_KV",0);
        m->cp_on  =glm5_envi("GLM5_CP",0) && m->ep_size>1;
        m->cp_nslot=glm5_cp_nslot(ctx,m->cp_block,m->ep_size,m->cp_on);
        m->msa_on =glm5_envi("GLM5_MSA",0);
        m->T_cp=0; return;
    }
    /* per-position un-sharded bf16 KV bytes across all layers (latent KV + MSA index on MoE). */
    int KVD=glm5_kv_cache_dim(c), ID=c->index_dim;
    int n_dense=c->n_dense_layers<c->n_layers?c->n_dense_layers:c->n_layers, n_moe=c->n_layers-n_dense;
    long per_pos=(long)c->n_layers*KVD*2 + (long)n_moe*ID*2;
    long T;
    if(thr>0) T=thr;                           /* explicit tuning override: T_cp in positions */
    else {
        /* Derive the Tier-A budget from real free memory: kv_init runs before weights load, so
         * MemAvailable is the whole node — subtract the weights-to-come (glm5_arena_size with a
         * tiny max_pos ~= weights only) and the real rope buffers, keep a safety headroom. */
        long avail=glm5_meminfo_bytes("MemAvailable");
        long budget;
        if(avail<=0) budget=(long)4<<30;       /* /proc/meminfo unreadable: fall back to 4 GB */
        else {
            glm5_config tiny=*c; tiny.max_pos=128;             /* KV/rope negligible -> ~weights */
            /* Converted mixed-IQ blobs are already rank-sliced, and arena_sz is
             * seeded from the exact blob size before kv_init.  The legacy
             * estimator assumes BF16 experts and would overstate EP12 by ~5x. */
            long reserve=m->arena_sz>0?(long)m->arena_sz:
                         (long)glm5_arena_size(&tiny,m->ep_rank,m->ep_size);
            reserve += (long)c->max_pos*(c->rotary_dim/2)*4*2; /* real rope_cos/sin */
            budget = avail - reserve - avail/10;               /* 10% headroom for scratch/mstream */
            if(budget < (1L<<30)) budget = 1L<<30;             /* floor: at least 1 GB of Tier-A KV */
        }
        int bgb=glm5_envi("GLM5_KV_BUDGET_GB",0);              /* optional hard cap (0 = no cap) */
        if(bgb>0 && ((long)bgb<<30) < budget) budget=(long)bgb<<30;
        T = budget/(per_pos>0?per_pos:1);
    }
    if(T<m->cp_block) T=m->cp_block;
    T=(T/m->cp_block)*m->cp_block;             /* block-align so Tier-A slots are CP-block aligned */
    m->int4_kv=0; m->cp_on=0;                   /* start in Tier A (bf16, replicated) */
    if(T>=ctx){ m->cp_nslot=ctx; m->T_cp=0; m->msa_on=0; }  /* fits un-sharded: Tier A only, no MSA */
    else      { m->cp_nslot=(int)T; m->T_cp=(int)T; m->msa_on=1; }  /* tiered: MSA on (Tier B needs it) */
}
/* allocate this layer's KV cache (bf16 or int4, sized to cp_nslot owned slots). */
static void glm5_alloc_kv(glm5_model*m,glm5_layer*L,int is_moe,size_t*used){
    const glm5_config*c=&m->cfg; int KVD=glm5_kv_cache_dim(c), KVH=1, ID=c->index_dim;
    size_t ns=(size_t)m->cp_nslot;
    if(m->int4_kv){
        L->k_q4=glm5_acalloc(ns*(KVD/2),1);
        L->k_qs=glm5_acalloc(ns*KVH,2);
        *used += ns*(KVD/2)+ns*KVH*2;
        if(is_moe){ L->idx_q4=glm5_acalloc(ns*(ID/2),1); L->idx_qs=glm5_acalloc(ns,2); *used += ns*(ID/2)+ns*2; }
    } else {
        L->kv_cache=glm5_acalloc(ns*KVD,2); *used += ns*KVD*2;
        if(is_moe){ L->idx_k_cache=glm5_acalloc(ns*ID,2); *used += ns*ID*2; }
    }
}

/* Tier A -> Tier B transition (called once when pos reaches m->T_cp). Each rank currently holds
 * the full [0,upto) KV un-sharded (Tier A: bf16, slot=pos). This keeps only this rank's CP-owned
 * blocks (b%ep_size==ep_rank), repacked int4 at the block-cyclic CP slot, for both the latent KV
 * and the MSA index keys; frees the larger Tier-A buffers; then flips cp_on/int4_kv. Purely local
 * (no collective) -- but must run on every rank in lockstep before the next chunk's collectives. */
static void glm5_prefill_to_cp(glm5_model*m,int upto){
    const glm5_config*c=&m->cfg;
    int KVD=glm5_kv_cache_dim(c), ID=c->index_dim, B=m->cp_block, eps=m->ep_size, er=m->ep_rank;
    int nslotB=glm5_cp_nslot(c->max_pos,B,eps,1);
    float kv[1024], ik[256];   /* KVD<=576, ID<=128 */
    for(int l=0;l<c->n_layers;l++){
        glm5_layer*L=&m->layers[l]; int is_moe=glm5_is_moe(c,l);
        /* idx cache exists only for layers that actually had a Tier-A MSA index buffer (real-weight
         * load allocates idx for has_full_indexer layers, not all MoE) -- key off the buffer, not
         * is_moe, so this matches exactly which layers Tier B allocates idx_q4 for. */
        int has_idx = (L->idx_k_cache!=NULL);
        uint8_t *nk=glm5_acalloc((size_t)nslotB*(KVD/2),1); uint16_t *nks=glm5_acalloc((size_t)nslotB,2);
        uint8_t *niq=NULL; uint16_t *niqs=NULL;
        if(has_idx){ niq=glm5_acalloc((size_t)nslotB*(ID/2),1); niqs=glm5_acalloc((size_t)nslotB,2); }
        for(int pos=0;pos<upto;pos++){
            int b=pos/B; if(b%eps!=er) continue;                 /* keep only my CP-owned blocks */
            int newslot=(b/eps)*B + (pos%B);                     /* == glm5_cp_slot under cp_on=1 */
            const uint16_t*kc=L->kv_cache+(size_t)pos*KVD;
            for(int i=0;i<KVD;i++) kv[i]=glm5_kv_dec(m,kc[i]);
            nks[newslot]=glm5_q4_pack(nk+(size_t)newslot*(KVD/2),kv,KVD);
            if(has_idx){
                const uint16_t*ic=L->idx_k_cache+(size_t)pos*ID;
                for(int i=0;i<ID;i++) ik[i]=glm5_kv_dec(m,ic[i]);
                niqs[newslot]=glm5_q4_pack(niq+(size_t)newslot*(ID/2),ik,ID);
            }
        }
        (void)is_moe;
        glm5_afree(L->kv_cache); L->kv_cache=NULL;
        glm5_afree(L->idx_k_cache); L->idx_k_cache=NULL;
        L->k_q4=nk; L->k_qs=nks; L->idx_q4=niq; L->idx_qs=niqs;
    }
    m->int4_kv=1; m->cp_on=1; m->cp_nslot=nslotB;
}

/* Group merge (Phase 2), local expert step: shrink this rank's expert ownership from ep_size to
 * new_eps (= 2*ep_size for a pairwise merge of adjacent groups), with this rank's new group-local
 * index new_ep_rank. Each routed expert was independently staged in BOTH sibling subgroups, so the
 * new owner e%new_eps already holds it -- drop the redundant experts locally (NO network transfer)
 * and recompact ex_w[] and owned_eid. Slot order stays ascending-by-eid, matching the e/ep_size
 * arithmetic the forward uses. Caller re-shards KV + rebuilds collectives for the merged group. */
static void glm5_group_expert_drop(glm5_model*m,int new_eps,int new_ep_rank){
    glm5_config*c=&m->cfg;
    for(int l=0;l<c->n_layers;l++){
        if(!glm5_is_moe(c,l)) continue;
        glm5_layer*L=&m->layers[l]; int keep=0;
        for(int s=0;s<L->n_owned;s++){
            int e=L->owned_eid[s];
            if(e%new_eps==new_ep_rank){
                if(keep!=s){ L->ex_w1[keep]=L->ex_w1[s]; L->ex_w3[keep]=L->ex_w3[s]; L->ex_w2[keep]=L->ex_w2[s]; L->owned_eid[keep]=e; }
                keep++;
            } else {
                glm5_afree(L->ex_w1[s].w); glm5_afree(L->ex_w1[s].scale);
                glm5_afree(L->ex_w3[s].w); glm5_afree(L->ex_w3[s].scale);
                glm5_afree(L->ex_w2[s].w); glm5_afree(L->ex_w2[s].scale);
            }
        }
        L->n_owned=keep;
    }
    m->ep_size=new_eps; m->ep_rank=new_ep_rank;
}

static void glm5_alloc_scratch(glm5_model*m,int hrows){
    const glm5_config*cfg=&m->cfg;
    const int H=cfg->hidden, QD=glm5_q_dim(cfg), AD=glm5_attn_dim(cfg);
    const int KVC=glm5_kv_cache_dim(cfg), IQD=glm5_idx_q_dim(cfg), ID=cfg->index_dim;
    int local_heads=cfg->n_heads;
    if(m->layers && cfg->n_layers>0){
        local_heads=m->layers[0].qh1-m->layers[0].qh0;
        if(local_heads<1) local_heads=1;
    }
    /* Attention scratch sized for FULL n_heads, not local_heads: the (optional) MTP draft block is
     * loaded REPLICATED (all heads on every rank) so its forward writes n_heads worth of per-head
     * scratch. Main layers use nown=local_heads<=n_heads -> fits. Cost ~+0.9MB/rank (negligible). */
    int sh_heads=cfg->n_heads;
    m->s_norm=glm5_amalloc(H*4); m->s_q=glm5_amalloc(QD*4); m->s_k=glm5_amalloc(KVC*4); m->s_v=glm5_amalloc(AD*4);
    m->s_kvb=glm5_amalloc((size_t)sh_heads*(cfg->qk_nope_dim+cfg->v_head_dim)*4);
    m->s_qabs=glm5_amalloc((size_t)sh_heads*cfg->kv_lora*4);
    m->s_ctx=glm5_amalloc((size_t)sh_heads*cfg->kv_lora*4);
    m->s_attn=glm5_amalloc(AD*4); m->s_o=glm5_amalloc(H*4);
    m->s_idx_q=glm5_amalloc((size_t)IQD*4); m->s_idx_k=glm5_amalloc((size_t)ID*4);
    m->s_blk_score=glm5_amalloc((size_t)cfg->max_pos*4); m->s_blk_sel=glm5_amalloc((size_t)cfg->max_pos*sizeof(int));
    m->s_attn_score=glm5_amalloc((size_t)cfg->max_pos*sh_heads*4);
    m->s_router=glm5_amalloc((size_t)cfg->n_experts*4); m->s_shg=glm5_amalloc(cfg->moe_inter*4); m->s_shu=glm5_amalloc(cfg->moe_inter*4);
    m->s_sh=glm5_amalloc(H*4); m->s_moe=glm5_amalloc(H*4); m->s_exg=glm5_amalloc(cfg->moe_inter*4); m->s_exu=glm5_amalloc(cfg->moe_inter*4); m->s_route=glm5_amalloc(H*4);
    m->s_ff_g=glm5_amalloc(cfg->dense_inter*4); m->s_ff_u=glm5_amalloc(cfg->dense_inter*4); m->s_ff=glm5_amalloc(H*4);
    /* full-vocab sized (>= hrows) so decode sampling can gather the sharded logits in place */
    m->s_logits=glm5_amalloc((size_t)(hrows>cfg->vocab?hrows:cfg->vocab)*4);
}

static glm5_model* glm5_clone_runtime(glm5_model*src){
    if(!src) return NULL;
    glm5_model*m=glm5_acalloc(1,sizeof(glm5_model)); if(!m) return NULL;
    *m=*src;
    m->owns_weights=0; m->pool=NULL; m->ms=NULL; m->arena=NULL; m->arena_sz=0; m->arena_used=0;
    m->bytes_read=0; memset(m->prof,0,sizeof(m->prof));
    m->layers=glm5_acalloc(src->cfg.n_layers,sizeof(glm5_layer));
    if(!m->layers){ glm5_afree(m); return NULL; }
    size_t used=0;
    for(int l=0;l<src->cfg.n_layers;l++){
        m->layers[l]=src->layers[l];
        glm5_layer*L=&m->layers[l];
        L->kv_cache=NULL; L->k_cache=NULL; L->v_cache=NULL; L->idx_k_cache=NULL;
        L->k_q4=NULL; L->v_q4=NULL; L->k_qs=NULL; L->v_qs=NULL; L->idx_q4=NULL; L->idx_qs=NULL;
        int has_idx_cache = src->layers[l].idx_k_cache || src->layers[l].idx_q4 || src->layers[l].idx_qs;
        glm5_alloc_kv(m,L,has_idx_cache,&used);
    }
    glm5_alloc_scratch(m,src->head.rows);
    m->arena_used=used; m->arena_sz=used;
    return m;
}

static glm5_model* glm5_alloc_synth(glm5_config cfg,int ep_rank,int ep_size,int n_threads,int n_cmgs){
    glm5_model*m=glm5_acalloc(1,sizeof(glm5_model)); if(!m) return NULL;
    m->cfg=cfg; m->ep_rank=ep_rank; m->ep_size=ep_size; m->n_threads=n_threads; m->n_cmgs=n_cmgs;
    m->bf16_mv_qt=GLM5_BF16; m->owns_weights=1; glm5_kv_init(m);
    const int H=cfg.hidden, QD=glm5_q_dim(&cfg), AD=glm5_attn_dim(&cfg);
    const int KVC=glm5_kv_cache_dim(&cfg), KVD=KVC, VD=cfg.v_head_dim, QHD=cfg.qk_head_dim, HD=QHD;
    const int IQD=glm5_idx_q_dim(&cfg), ID=cfg.index_dim, half=cfg.qk_rope_dim/2;
    /* TP flags (TP_ATTN forced off under CP: each rank needs all query heads to merge position-shards) */
    int tp=glm5_envi("GLM5_TP",0);
    int tp_attn=glm5_envi("GLM5_TP_ATTN",tp) && !m->cp_on; int tp_sh=glm5_envi("GLM5_TP_SHARED",tp);
    int tp_ffn=glm5_envi("GLM5_TP_FFN",tp), tp_head=glm5_envi("GLM5_TP_HEAD",tp), tp_emb=glm5_envi("GLM5_TP_EMBED",tp);
    int qh0,qh1; if(tp_attn) glm5_shard_heads(cfg.n_heads,ep_rank,ep_size,&qh0,&qh1); else { qh0=0; qh1=cfg.n_heads; }
    int qrows=(qh1-qh0)*QHD, arows=(qh1-qh0)*VD, kvb_rows=(qh1-qh0)*(cfg.qk_nope_dim+VD);
    int sh_r0,sh_rows; if(tp_sh) glm5_shard_shared_inter(cfg.moe_inter,ep_rank,ep_size,&sh_r0,&sh_rows); else { sh_r0=0; sh_rows=cfg.moe_inter; }
    int ff_r0,ff_rows; if(tp_ffn) glm5_shard(cfg.dense_inter,ep_rank,ep_size,&ff_r0,&ff_rows); else { ff_r0=0; ff_rows=cfg.dense_inter; }
    int hr0,hrows; if(tp_head) glm5_shard(cfg.vocab,ep_rank,ep_size,&hr0,&hrows); else { hr0=0; hrows=cfg.vocab; }
    int er0,erows; if(tp_emb) glm5_shard(cfg.vocab,ep_rank,ep_size,&er0,&erows); else { er0=0; erows=cfg.vocab; }
    size_t used=0;
    #define BF(p,n,amp) do{ (p)=glm5_amalloc((size_t)(n)*2); glm5_fill_bf16((p),(n),(amp)); used+=(size_t)(n)*2; }while(0)
    #define FZ(p,n,amp) do{ (p)=glm5_amalloc((size_t)(n)*4); glm5_fill_f32 ((p),(n),(amp)); used+=(size_t)(n)*4; }while(0)

    m->rope_cos=glm5_amalloc((size_t)cfg.max_pos*half*4); m->rope_sin=glm5_amalloc((size_t)cfg.max_pos*half*4);
    for(int p=0;p<cfg.max_pos;p++) for(int k=0;k<half;k++){
        double invf=pow((double)cfg.rope_theta,-2.0*k/(double)cfg.rotary_dim); double a=p*invf;
        m->rope_cos[(size_t)p*half+k]=(float)cos(a); m->rope_sin[(size_t)p*half+k]=(float)sin(a); }

    uint16_t*eb; BF(eb,(size_t)erows*H,0.05f); m->embed=eb; m->emb_r0=er0; m->emb_rows=erows;
    uint16_t*hd; BF(hd,(size_t)hrows*H,0.05f); m->head.w=hd; m->head.type=GLM5_BF16; m->head.rows=hrows; m->head.cols=H; m->head_r0=hr0;
    BF(m->out_norm,H,0.1f);

    m->layers=glm5_acalloc(cfg.n_layers,sizeof(glm5_layer));
    for(int l=0;l<cfg.n_layers;l++){
        glm5_layer*L=&m->layers[l]; int is_moe=glm5_is_moe(&cfg,l);
        BF(L->input_norm,H,0.1f); BF(L->post_norm,H,0.1f);
        uint16_t*p;
        BF(p,(size_t)cfg.q_lora*H,0.03f); L->wq_a=(glm5_tensor){p,NULL,GLM5_BF16,cfg.q_lora,H};
        BF(p,(size_t)qrows*cfg.q_lora,0.03f); L->wq_b=(glm5_tensor){p,NULL,GLM5_BF16,qrows,cfg.q_lora};
        BF(p,(size_t)KVD*H,0.03f); L->wkv_a=(glm5_tensor){p,NULL,GLM5_BF16,KVD,H};
        BF(p,(size_t)kvb_rows*cfg.kv_lora,0.03f); L->wkv_b=(glm5_tensor){p,NULL,GLM5_BF16,kvb_rows,cfg.kv_lora};
        BF(p,(size_t)H*arows,0.03f); L->wo=(glm5_tensor){p,NULL,GLM5_BF16,H,arows};
        BF(L->q_a_norm,cfg.q_lora,0.1f); BF(L->kv_a_norm,cfg.kv_lora,0.1f);
        L->qh0=qh0; L->qh1=qh1;
        glm5_alloc_kv(m,L,is_moe,&used);
        if(is_moe){
            /* MSA indexer (replicated) */
            BF(p,(size_t)IQD*H,0.03f); L->idx_wq=(glm5_tensor){p,NULL,GLM5_BF16,IQD,H};
            BF(p,(size_t)ID*H,0.03f);  L->idx_wk=(glm5_tensor){p,NULL,GLM5_BF16,ID,H};
            BF(L->idx_q_norm,ID,0.1f); BF(L->idx_k_norm,ID,0.1f);
            /* MoE */
            BF(p,(size_t)cfg.n_experts*H,0.03f); L->gate=(glm5_tensor){p,NULL,GLM5_BF16,cfg.n_experts,H};
            FZ(L->gate_bias,cfg.n_experts,0.1f);
            BF(p,(size_t)sh_rows*H,0.03f); L->sh_w1=(glm5_tensor){p,NULL,GLM5_BF16,sh_rows,H};
            BF(p,(size_t)sh_rows*H,0.03f); L->sh_w3=(glm5_tensor){p,NULL,GLM5_BF16,sh_rows,H};
            BF(p,(size_t)H*sh_rows,0.03f); L->sh_w2=(glm5_tensor){p,NULL,GLM5_BF16,H,sh_rows};
            L->sh_r0=sh_r0; L->sh_rows=sh_rows;
            int no=glm5_n_owned(cfg.n_experts,ep_rank,ep_size);
            L->n_owned=no; L->owned_eid=glm5_amalloc((size_t)(no>0?no:1)*sizeof(int));
            L->ex_w1=glm5_amalloc((size_t)(no>0?no:1)*sizeof(glm5_tensor)); L->ex_w3=glm5_amalloc((size_t)(no>0?no:1)*sizeof(glm5_tensor)); L->ex_w2=glm5_amalloc((size_t)(no>0?no:1)*sizeof(glm5_tensor));
            int s=0; for(int e=0;e<cfg.n_experts;e++) if(e%ep_size==ep_rank){ L->owned_eid[s]=e;
                BF(p,(size_t)cfg.moe_inter*H,0.03f); L->ex_w1[s]=(glm5_tensor){p,NULL,GLM5_BF16,cfg.moe_inter,H};
                BF(p,(size_t)cfg.moe_inter*H,0.03f); L->ex_w3[s]=(glm5_tensor){p,NULL,GLM5_BF16,cfg.moe_inter,H};
                BF(p,(size_t)H*cfg.moe_inter,0.03f); L->ex_w2[s]=(glm5_tensor){p,NULL,GLM5_BF16,H,cfg.moe_inter}; s++; }
        } else {
            BF(p,(size_t)ff_rows*H,0.03f); L->ff_gate=(glm5_tensor){p,NULL,GLM5_BF16,ff_rows,H};
            BF(p,(size_t)ff_rows*H,0.03f); L->ff_up  =(glm5_tensor){p,NULL,GLM5_BF16,ff_rows,H};
            BF(p,(size_t)H*ff_rows,0.03f); L->ff_down=(glm5_tensor){p,NULL,GLM5_BF16,H,ff_rows};
            L->ff_r0=ff_r0; L->ff_rows=ff_rows;
        }
    }
    glm5_alloc_scratch(m,hrows);
    m->arena_used=used; m->arena_sz=used;
    glm5_init_fp8_lut(m->fp8_lut);
    glm5_dummy=glm5_envi("GLM5_DUMMY",0);
    if(glm5_envi("GLM5_POOL",0)) m->pool=glm5_g_pool=glm5_pool_create(m->n_threads);  /* experimental pinned pool; default OpenMP */
    return m;
    #undef BF
    #undef FZ
}

/* ===================== real-weight loader (staged blob + manifest) ===================== */
#include <sys/mman.h>
typedef struct { char name[300]; uint64_t off; size_t nbytes; int f32; glm5_qtype qtype; int nd; long shape[5]; } glm5_ent;
static glm5_qtype glm5_manifest_qtype(const char*s){
    if(!strcmp(s,"Q8_0"))return GLM5_Q8_0;
    if(!strcmp(s,"Q2_K"))return GLM5_Q2_K;
    if(!strcmp(s,"Q3_K"))return GLM5_Q3_K;
    if(!strcmp(s,"Q4_K"))return GLM5_Q4_K;
    if(!strcmp(s,"Q5_K"))return GLM5_Q5_K;
    if(!strcmp(s,"Q6_K"))return GLM5_Q6_K;
    if(!strcmp(s,"IQ2_XS"))return GLM5_IQ2_XS;
    if(!strcmp(s,"IQ3_XXS"))return GLM5_IQ3_XXS;
    if(!strcmp(s,"IQ4_XS"))return GLM5_IQ4_XS;
    return GLM5_BF16;
}
static glm5_ent* glm5_find(glm5_ent*es,int n,const char*nm){ for(int i=0;i<n;i++) if(!strcmp(es[i].name,nm)) return &es[i]; return NULL; }
static glm5_ent* glm5_req(glm5_ent*es,int n,const char*nm){ glm5_ent*e=glm5_find(es,n,nm); if(!e) fprintf(stderr,"glm5_load: MISSING tensor %s\n",nm); return e; }
static void glm5_blob_dontneed(const uint8_t*base,uint64_t off,size_t len){
    long ps=sysconf(_SC_PAGESIZE); if(ps<=0) ps=4096;
    uintptr_t a=(uintptr_t)(base+off), b=a+len, aa=a&~(uintptr_t)(ps-1), bb=(b+(uintptr_t)ps-1)&~(uintptr_t)(ps-1);
    if(bb>aa) madvise((void*)aa,bb-aa,MADV_DONTNEED);
}
/* copy helpers from the blob mmap (base); return malloc'd buffer (caller frees). */
static void* glm5_cp_full(const uint8_t*base,const glm5_ent*e){ void*d=glm5_amalloc(e->nbytes); memcpy(d,base+e->off,e->nbytes); glm5_blob_dontneed(base,e->off,e->nbytes); return d; }
static void* glm5_cp_rows(const uint8_t*base,const glm5_ent*e,int r0,int nrows,int cols,int esz){
    size_t rb=(size_t)cols*esz, off=e->off+(size_t)r0*rb, nb=(size_t)nrows*rb;
    void*d=glm5_amalloc(nb); memcpy(d,base+off,nb); glm5_blob_dontneed(base,off,nb); return d; }
static void* glm5_cp_cols(const uint8_t*base,const glm5_ent*e,int Rtot,int c0,int ncols,int Ctot,int esz){
    void*d=glm5_amalloc((size_t)Rtot*ncols*esz); uint8_t*dp=d; const uint8_t*sp=base+e->off;
    for(int r=0;r<Rtot;r++) memcpy(dp+(size_t)r*ncols*esz, sp+((size_t)r*Ctot+c0)*esz, (size_t)ncols*esz);
    glm5_blob_dontneed(base,e->off,e->nbytes); return d; }
static uint8_t* glm5_cp_scale_f32_blocks(const uint8_t*base,const glm5_ent*e,int r0,int rows,int c0,int cols,int Rtot,int Ctot){
    int sc=(Ctot+127)/128, cb0=c0/128, ncb=(cols+127)/128;
    float *d=glm5_amalloc((size_t)rows*ncb*4);
    const float *sp=(const float*)(base+e->off);
    (void)Rtot;
    for(int r=0;r<rows;r++){
        const float *sr=sp+(size_t)((r0+r)/128)*sc+cb0;
        memcpy(d+(size_t)r*ncb, sr, (size_t)ncb*4);
    }
    glm5_blob_dontneed(base,e->off,e->nbytes);
    return (uint8_t*)d;
}
/* INT8 weight_scale (BF16 [Rtot, ng]) -> F32 [nrows, ngk], slicing rows [r0,r0+nrows) and
 * groups [g0,g0+ngk). Per actual row (no 128-row block sharing), so the matvec reads
 * scale[localrow*ngk + col/gs] directly. */
static uint8_t* glm5_cp_scale_bf16(const uint8_t*base,const glm5_ent*e,int r0,int nrows,int g0,int ngk,int ng){
    float *d=glm5_amalloc((size_t)nrows*ngk*4);
    const uint16_t *sp=(const uint16_t*)(base+e->off);
    for(int r=0;r<nrows;r++) for(int g=0;g<ngk;g++){
        uint32_t bits=(uint32_t)sp[(size_t)(r0+r)*ng + g0 + g]<<16; float f; memcpy(&f,&bits,4);
        d[(size_t)r*ngk+g]=f;
    }
    glm5_blob_dontneed(base,e->off,e->nbytes);
    return (uint8_t*)d;
}

/* Load weight `name` (bf16, or FP8 if a `name_scale_inv` companion exists) with TP slicing.
 * mode 0 = full [Rtot,Ctot]; 1 = row-shard rows[r0,r0+nr) of [Rtot,Ctot]; 2 = col-shard
 * cols[c0,c0+nc) of [Rtot,Ctot]. GLM5.2 FP8 stores F8_E4M3 weights plus F32 scale_inv
 * blocks [ceil(rows/128), ceil(cols/128)]. The loader expands those block scales to one
 * scale row per local weight row, so row-shards may start at arbitrary head boundaries. */
static glm5_tensor glm5_load_w(glm5_ent*es,int n,const uint8_t*base,const char*name,int mode,
                           int r0,int nr,int c0,int nc,int Rtot,int Ctot,int*ok,size_t*used){
    char sn[416]; snprintf(sn,sizeof sn,"%s_scale_inv",name);
    char pn[416]; snprintf(pn,sizeof pn,"%s_packed",name);
    char qn[416]; snprintf(qn,sizeof qn,"%s_scale",name);
    glm5_ent*we=glm5_find(es,n,name); glm5_ent*se=glm5_find(es,n,sn);
    glm5_ent*pe=glm5_find(es,n,pn); glm5_ent*qe=glm5_find(es,n,qn);
    glm5_tensor t; t.w=NULL; t.scale=NULL; t.type=GLM5_BF16; t.rows=0; t.cols=0; t.qg=0; t.qg0=0;
    if(we && glm5_is_ggml_q(we->qtype)){
        int er=we->nd>0?(int)we->shape[0]:Rtot, ec=we->nd>1?(int)we->shape[1]:Ctot;
        if(mode!=0 || er!=Rtot || ec!=Ctot ||
           we->nbytes!=(size_t)er*dequant_row_size((uint32_t)we->qtype,ec)){
            fprintf(stderr,"glm5_load: invalid compressed tensor %s shape=%dx%d bytes=%zu mode=%d\n",
                    name,er,ec,we->nbytes,mode); *ok=0; return t;
        }
        t.w=glm5_cp_full(base,we); t.type=we->qtype; t.rows=er; t.cols=ec;
        *used+=we->nbytes; return t;
    }
    /* INT8 (compressed-tensors w8a16): name_packed (int8 bytes [Rtot,Ctot]) + name_scale (bf16
     * [Rtot,ng]); offset-binary q=byte-128, group size gs=Ctot/ng (128 group / =Ctot channel). */
    if(pe && qe){
        int ng = qe->nd>=2 ? (int)qe->shape[1] : 1; if(ng<1) ng=1;
        int gs = Ctot/ng; if(gs<1) gs=Ctot;
        t.type=GLM5_INT8; t.qg=gs;
        if(mode==0){ t.w=glm5_cp_full(base,pe); t.rows=Rtot; t.cols=Ctot; *used+=(size_t)Rtot*Ctot;
            t.scale=glm5_cp_scale_bf16(base,qe,0,Rtot,0,ng,ng); *used+=(size_t)Rtot*ng*4; }
        else if(mode==1){ t.w=glm5_cp_rows(base,pe,r0,nr,Ctot,1); t.rows=nr; t.cols=Ctot; *used+=(size_t)nr*Ctot;
            t.scale=glm5_cp_scale_bf16(base,qe,r0,nr,0,ng,ng); *used+=(size_t)nr*ng*4; }
        else {
            int g0=c0/gs, g1=(c0+nc+gs-1)/gs, ngk=g1-g0;
            t.w=glm5_cp_cols(base,pe,Rtot,c0,nc,Ctot,1); t.rows=Rtot; t.cols=nc; *used+=(size_t)Rtot*nc;
            t.scale=glm5_cp_scale_bf16(base,qe,0,Rtot,g0,ngk,ng); *used+=(size_t)Rtot*ngk*4;
            t.qg0=c0-g0*gs;
        }
        return t;
    }
    if(!we){ fprintf(stderr,"glm5_load: MISSING %s\n",name); *ok=0; return t; }
    int esz = se ? 1 : 2;  t.type = se ? GLM5_MXFP8 : GLM5_BF16;
    int er=we->nd>0?(int)we->shape[0]:Rtot, ec=we->nd>1?(int)we->shape[1]:Ctot;
    int local_row=(mode==1 && er==nr && ec==Ctot);
    int local_col=(mode==2 && er==Rtot && ec==nc);
    if(mode==0 || local_row || local_col){
        t.w=glm5_cp_full(base,we); t.rows=er; t.cols=ec; *used+=we->nbytes;
    } else if(mode==1){ t.w=glm5_cp_rows(base,we,r0,nr,Ctot,esz); t.rows=nr; t.cols=Ctot; *used+=(size_t)nr*Ctot*esz; }
    else { t.w=glm5_cp_cols(base,we,Rtot,c0,nc,Ctot,esz); t.rows=Rtot; t.cols=nc; *used+=(size_t)Rtot*nc*esz; }
    if(se){
        if(!se->f32){ fprintf(stderr,"glm5_load: FP8 scale_inv %s is not F32\n",sn); *ok=0; return t; }
        if(mode==0){ t.scale=glm5_cp_scale_f32_blocks(base,se,0,Rtot,0,Ctot,Rtot,Ctot); *used+=(size_t)Rtot*((Ctot+127)/128)*4; }
        else if(mode==1){ t.scale=glm5_cp_scale_f32_blocks(base,se,r0,nr,0,Ctot,Rtot,Ctot); *used+=(size_t)nr*((Ctot+127)/128)*4; }
        else {
            if((c0%128) && (c0%128)+nc>128){ fprintf(stderr,"glm5_load: FP8 col-shard %s crosses an unaligned 128-col scale block (c0=%d nc=%d)\n",name,c0,nc); *ok=0; return t; }
            t.scale=glm5_cp_scale_f32_blocks(base,se,0,Rtot,c0,nc,Rtot,Ctot); *used+=(size_t)Rtot*((nc+127)/128)*4;
        }
    }
    return t;
}

/* Build an glm5_model from this rank's staged blob (GLM5_STAGE_DIR/rank<rr>.{blob,manifest}).
 * Dense tensors are TP-sliced into the arena per the same ranges as glm5_alloc_synth; routed
 * experts are the owned ones in the blob. Returns NULL on any missing/short tensor. */
/* Load ONE transformer layer's real weights (name prefix model.layers.<l>.) into L. Shared by the
 * main layer loop and the MTP block (layer 78). is_moe/has_idx are computed by the caller. */
static void glm5_load_one_layer(glm5_model*m, glm5_layer*L, int l, int is_moe, int has_idx,
    glm5_ent*es, int n, const uint8_t*base,
    int qh0,int qh1,int qrows,int arows,int sh_r0,int sh_rows,int ff_r0,int ff_rows,
    int*ok,size_t*used){
    const glm5_config*cfg=&m->cfg;
    const int H=cfg->hidden, QD=glm5_q_dim(cfg), AD=glm5_attn_dim(cfg), KVC=glm5_kv_cache_dim(cfg);
    const int VD=cfg->v_head_dim, QHD=cfg->qk_head_dim, IQD=glm5_idx_q_dim(cfg), ID=cfg->index_dim;
    char nb[512];
    #define REQ2(nm) ({ glm5_ent*_e=glm5_req(es,n,(nm)); if(!_e){ *ok=0; } _e; })
    #define LN(suf) (snprintf(nb,sizeof nb,"model.layers.%d." suf,l),nb)
    { glm5_ent*e=REQ2(LN("input_layernorm.weight")); if(e) L->input_norm=glm5_cp_full(base,e); }
    { glm5_ent*e=REQ2(LN("post_attention_layernorm.weight")); if(e) L->post_norm=glm5_cp_full(base,e); }
    L->wq_a=glm5_load_w(es,n,base,LN("self_attn.q_a_proj.weight"),0,0,0,0,0,cfg->q_lora,H,ok,used);
    L->wq_b=glm5_load_w(es,n,base,LN("self_attn.q_b_proj.weight"),1,qh0*QHD,qrows,0,0,QD,cfg->q_lora,ok,used);
    L->wkv_a=glm5_load_w(es,n,base,LN("self_attn.kv_a_proj_with_mqa.weight"),0,0,0,0,0,KVC,H,ok,used);
    L->wkv_b=glm5_load_w(es,n,base,LN("self_attn.kv_b_proj.weight"),1,qh0*(cfg->qk_nope_dim+VD),arows ? (qh1-qh0)*(cfg->qk_nope_dim+VD) : 0,0,0,cfg->n_heads*(cfg->qk_nope_dim+VD),cfg->kv_lora,ok,used);
    L->wo=glm5_load_w(es,n,base,LN("self_attn.o_proj.weight"),2,0,0,qh0*VD,arows,H,AD,ok,used);
    { glm5_ent*e=REQ2(LN("self_attn.q_a_layernorm.weight")); if(e) L->q_a_norm=glm5_cp_full(base,e); }
    { glm5_ent*e=REQ2(LN("self_attn.kv_a_layernorm.weight")); if(e) L->kv_a_norm=glm5_cp_full(base,e); }
    L->qh0=qh0; L->qh1=qh1;
    glm5_alloc_kv(m,L,is_moe && has_idx,used);
    if(is_moe){
        if(has_idx){
            L->idx_wq_b=glm5_load_w(es,n,base,LN("self_attn.indexer.wq_b.weight"),0,0,0,0,0,IQD,cfg->q_lora,ok,used);
            L->idx_wk=glm5_load_w(es,n,base,LN("self_attn.indexer.wk.weight"),0,0,0,0,0,ID,H,ok,used);
            L->idx_wproj=glm5_load_w(es,n,base,LN("self_attn.indexer.weights_proj.weight"),0,0,0,0,0,cfg->index_n_heads,H,ok,used);
            { glm5_ent*e=REQ2(LN("self_attn.indexer.k_norm.weight")); if(e) L->idx_k_norm=glm5_cp_full(base,e); }
            { glm5_ent*e=REQ2(LN("self_attn.indexer.k_norm.bias")); if(e) L->idx_k_bias=glm5_cp_full(base,e); }
        }
        { glm5_ent*e=REQ2(LN("mlp.gate.weight")); if(e){
            void*gw;
            if(e->f32){ size_t ne=(size_t)cfg->n_experts*H; uint16_t*d=glm5_amalloc(ne*2);
                const uint32_t*s=(const uint32_t*)(base+e->off);
                for(size_t i=0;i<ne;i++){ uint32_t b=s[i]; d[i]=(uint16_t)((b+0x8000u+((b>>16)&1))>>16); }
                glm5_blob_dontneed(base,e->off,e->nbytes); gw=d;
            } else gw=glm5_cp_full(base,e);
            L->gate=(glm5_tensor){gw,NULL,GLM5_BF16,cfg->n_experts,H}; } }
        { glm5_ent*e=REQ2(LN("mlp.gate.e_score_correction_bias")); if(e) L->gate_bias=glm5_cp_full(base,e); }
        L->sh_w1=glm5_load_w(es,n,base,LN("mlp.shared_experts.gate_proj.weight"),1,sh_r0,sh_rows,0,0,cfg->moe_inter,H,ok,used);
        L->sh_w3=glm5_load_w(es,n,base,LN("mlp.shared_experts.up_proj.weight"),1,sh_r0,sh_rows,0,0,cfg->moe_inter,H,ok,used);
        L->sh_w2=glm5_load_w(es,n,base,LN("mlp.shared_experts.down_proj.weight"),2,0,0,sh_r0,sh_rows,H,cfg->moe_inter,ok,used);
        L->sh_r0=sh_r0; L->sh_rows=sh_rows;
        int no=glm5_n_owned(cfg->n_experts,m->ep_rank,m->ep_size); L->n_owned=no;
        L->owned_eid=glm5_amalloc((size_t)(no>0?no:1)*sizeof(int));
        L->ex_w1=glm5_amalloc((size_t)(no>0?no:1)*sizeof(glm5_tensor)); L->ex_w3=glm5_amalloc((size_t)(no>0?no:1)*sizeof(glm5_tensor)); L->ex_w2=glm5_amalloc((size_t)(no>0?no:1)*sizeof(glm5_tensor));
        int s=0; for(int e2=0;e2<cfg->n_experts && *ok;e2++) if(e2%m->ep_size==m->ep_rank){ L->owned_eid[s]=e2;
            snprintf(nb,sizeof nb,"model.layers.%d.mlp.experts.%d.gate_proj.weight",l,e2); L->ex_w1[s]=glm5_load_w(es,n,base,nb,0,0,0,0,0,cfg->moe_inter,H,ok,used);
            snprintf(nb,sizeof nb,"model.layers.%d.mlp.experts.%d.up_proj.weight",l,e2); L->ex_w3[s]=glm5_load_w(es,n,base,nb,0,0,0,0,0,cfg->moe_inter,H,ok,used);
            snprintf(nb,sizeof nb,"model.layers.%d.mlp.experts.%d.down_proj.weight",l,e2); L->ex_w2[s]=glm5_load_w(es,n,base,nb,0,0,0,0,0,H,cfg->moe_inter,ok,used);
            s++; }
    } else {
        L->ff_gate=glm5_load_w(es,n,base,LN("mlp.gate_proj.weight"),1,ff_r0,ff_rows,0,0,cfg->dense_inter,H,ok,used);
        L->ff_up  =glm5_load_w(es,n,base,LN("mlp.up_proj.weight"),1,ff_r0,ff_rows,0,0,cfg->dense_inter,H,ok,used);
        L->ff_down=glm5_load_w(es,n,base,LN("mlp.down_proj.weight"),2,0,0,ff_r0,ff_rows,H,cfg->dense_inter,ok,used);
        L->ff_r0=ff_r0; L->ff_rows=ff_rows;
    }
    #undef LN
    #undef REQ2
}
/* MTP real-weight load: the checkpoint's layer 78 (a full MoE block w/ indexer) + the fusion tensors
 * (enorm/hnorm/eh_proj) + shared_head.norm. Called from glm5_load_real while the blob is mapped, gated
 * GLM5_MTP. Stage with GLM5_STAGE_LAYERS=79, or GLM5_STAGE_MTP=1 for truncated runs, so
 * model.layers.78.* is in the rank blobs. */
static int glm5_load_mtp_real(glm5_model*m, glm5_ent*es, int n, const uint8_t*base,
    int qh0,int qh1,int qrows,int arows,int sh_r0,int sh_rows,size_t*used){
    const glm5_config*cfg=&m->cfg; const int H=cfg->hidden; int ok=1, L78=78;
    /* Load the MTP draft block REPLICATED (no TP col-shard). The MTP layer's o_proj and
     * shared_experts.down_proj are PER-CHANNEL int8 (scale ng=1 -> gs=cols), which the col-shard path
     * rejects (nc%gs != 0). It's a small draft head, so replicating attn+shared (full arows/sh_rows,
     * qh0=0) is fine and means the draft forward needs no o_proj all-reduce; routed experts stay
     * EP-sharded via glm5_load_one_layer's e%ep_size ownership. (Ignore the caller's TP-shard params.) */
    (void)qh0;(void)qh1;(void)qrows;(void)arows;(void)sh_r0;(void)sh_rows;
    m->mtp_layer=glm5_acalloc(1,sizeof(glm5_layer));
    glm5_load_one_layer(m,m->mtp_layer,L78,/*is_moe*/1,/*has_idx*/1,es,n,base,
                        0,cfg->n_heads,glm5_q_dim(cfg),glm5_attn_dim(cfg),0,cfg->moe_inter,0,0,&ok,used);
    char nb[512];
    #define MREQ(suf) ({ snprintf(nb,sizeof nb,"model.layers.%d." suf,L78); glm5_ent*_e=glm5_req(es,n,nb); if(!_e) ok=0; _e; })
    { glm5_ent*e=MREQ("enorm.weight");            if(e) m->mtp_enorm=glm5_cp_full(base,e); }
    { glm5_ent*e=MREQ("hnorm.weight");            if(e) m->mtp_hnorm=glm5_cp_full(base,e); }
    { glm5_ent*e=MREQ("shared_head.norm.weight"); if(e) m->mtp_out_norm=glm5_cp_full(base,e); }
    #undef MREQ
    snprintf(nb,sizeof nb,"model.layers.%d.eh_proj.weight",L78);
    m->mtp_eh=glm5_load_w(es,n,base,nb,0,0,0,0,0,H,2*H,&ok,used);  /* [H, 2H], replicated */
    if(!ok){ fprintf(stderr,"glm5_load_mtp: rank %d incomplete (layer %d missing)\n",m->ep_rank,L78); return -1; }
    return 0;
}
static glm5_model* glm5_load_real(glm5_config cfg,int ep_rank,int ep_size,const char*blob_dir,int n_threads,int n_cmgs){
    char bdir[1024]; if(blob_dir&&*blob_dir) snprintf(bdir,sizeof bdir,"%s",blob_dir);
    else { const char*e=getenv("GLM5_STAGE_DIR"); snprintf(bdir,sizeof bdir,"%s",(e&&*e)?e:"/local/glm5"); }
    char bp[1100],mp[1100]; snprintf(bp,sizeof bp,"%s/rank%02d.blob",bdir,ep_rank); snprintf(mp,sizeof mp,"%s/rank%02d.manifest",bdir,ep_rank);
    FILE*mf=fopen(mp,"r"); if(!mf){ fprintf(stderr,"glm5_load: cannot open %s\n",mp); return NULL; }
    char line[1024]; int cap=4096,n=0,glm52_manifest=0; glm5_ent*es=glm5_amalloc((size_t)cap*sizeof(glm5_ent));
    while(fgets(line,sizeof line,mf)){
        if(line[0]=='#'){ if(strstr(line,"glm52-a64fx-ep12-v1"))glm52_manifest=1; continue; }
        if(line[0]=='\n') continue;
        if(n>=cap){ cap*=2; es=realloc(es,(size_t)cap*sizeof(glm5_ent)); }
        glm5_ent*e=&es[n]; char dt[32]; int pos=0,cnt;
        if(sscanf(line,"%llu %zu %31s %d%n",(unsigned long long*)&e->off,&e->nbytes,dt,&e->nd,&cnt)!=4) continue;
        pos=cnt; e->f32=(dt[0]=='F'&&dt[1]=='3'); e->qtype=glm5_manifest_qtype(dt);
        for(int d=0;d<e->nd&&d<5;d++){ long v; int c2; sscanf(line+pos," %ld%n",&v,&c2); e->shape[d]=v; pos+=c2; }
        while(line[pos]==' ')pos++; char*nl=strchr(line+pos,'\n'); if(nl)*nl=0; snprintf(e->name,sizeof e->name,"%s",line+pos); n++;
    }
    fclose(mf);
    int bfd=open(bp,O_RDONLY); if(bfd<0){ fprintf(stderr,"glm5_load: cannot open %s\n",bp); glm5_afree(es); return NULL; }
    struct stat sb; fstat(bfd,&sb); size_t bsz=sb.st_size;
    for(int i=0;i<n;i++) if(es[i].off>bsz || es[i].nbytes>bsz-es[i].off){
        fprintf(stderr,"glm5_load: manifest tensor %s exceeds blob (%llu+%zu > %zu)\n",
                es[i].name,(unsigned long long)es[i].off,es[i].nbytes,bsz);
        close(bfd); glm5_afree(es); return NULL;
    }
    const uint8_t*base=mmap(NULL,bsz,PROT_READ,MAP_PRIVATE,bfd,0);
    if(base==MAP_FAILED){ fprintf(stderr,"glm5_load: mmap failed\n"); close(bfd); glm5_afree(es); return NULL; }

    glm5_model*m=glm5_acalloc(1,sizeof(glm5_model)); m->cfg=cfg; m->ep_rank=ep_rank; m->ep_size=ep_size; m->n_threads=n_threads; m->n_cmgs=n_cmgs; m->bf16_mv_qt=GLM5_BF16; m->owns_weights=1;
    if(glm52_manifest) m->arena_sz=bsz; /* exact compressed/rank-sliced weight reserve for KV tiering */
    glm5_kv_init(m);
    const int H=cfg.hidden, QD=glm5_q_dim(&cfg), AD=glm5_attn_dim(&cfg);
    const int KVC=glm5_kv_cache_dim(&cfg), VD=cfg.v_head_dim, QHD=cfg.qk_head_dim;
    const int IQD=glm5_idx_q_dim(&cfg), ID=cfg.index_dim, half=cfg.qk_rope_dim/2;
    int tp=glm5_envi("GLM5_TP",0);
    int tp_attn=glm5_envi("GLM5_TP_ATTN",tp) && !m->cp_on; int tp_sh=glm5_envi("GLM5_TP_SHARED",tp),tp_ffn=glm5_envi("GLM5_TP_FFN",tp),tp_head=glm5_envi("GLM5_TP_HEAD",tp),tp_emb=glm5_envi("GLM5_TP_EMBED",tp);
    int qh0,qh1; if(tp_attn) glm5_shard_heads(cfg.n_heads,ep_rank,ep_size,&qh0,&qh1); else { qh0=0; qh1=cfg.n_heads; } int qrows=(qh1-qh0)*QHD, arows=(qh1-qh0)*VD;
    int sh_r0,sh_rows; if(tp_sh) glm5_shard_shared_inter(cfg.moe_inter,ep_rank,ep_size,&sh_r0,&sh_rows); else { sh_r0=0; sh_rows=cfg.moe_inter; }
    int ff_r0,ff_rows; if(tp_ffn) glm5_shard(cfg.dense_inter,ep_rank,ep_size,&ff_r0,&ff_rows); else { ff_r0=0; ff_rows=cfg.dense_inter; }
    int hr0,hrows; if(tp_head) glm5_shard(cfg.vocab,ep_rank,ep_size,&hr0,&hrows); else { hr0=0; hrows=cfg.vocab; }
    int er0,erows; if(tp_emb) glm5_shard(cfg.vocab,ep_rank,ep_size,&er0,&erows); else { er0=0; erows=cfg.vocab; }
    size_t used=0; int ok=1;
    #define REQ(nm) ({ glm5_ent*_e=glm5_req(es,n,(nm)); if(!_e){ ok=0; } _e; })
    char nb[512];

    m->rope_cos=glm5_amalloc((size_t)cfg.max_pos*half*4); m->rope_sin=glm5_amalloc((size_t)cfg.max_pos*half*4);
    for(int p=0;p<cfg.max_pos;p++) for(int k=0;k<half;k++){ double invf=pow((double)cfg.rope_theta,-2.0*k/(double)cfg.rotary_dim),a=p*invf;
        m->rope_cos[(size_t)p*half+k]=(float)cos(a); m->rope_sin[(size_t)p*half+k]=(float)sin(a); }

    { glm5_ent*e=REQ("model.embed_tokens.weight"); if(e){
        if(e->nd>=2&&e->shape[0]==erows&&e->shape[1]==H) m->embed=glm5_cp_full(base,e);
        else m->embed=glm5_cp_rows(base,e,er0,erows,H,2);
        used+=(size_t)erows*H*2; } m->emb_r0=er0; m->emb_rows=erows; }
    { glm5_ent*e=REQ("lm_head.weight"); if(e){
        if(e->nd>=2&&e->shape[0]==hrows&&e->shape[1]==H) m->head.w=glm5_cp_full(base,e);
        else m->head.w=glm5_cp_rows(base,e,hr0,hrows,H,2);
        used+=(size_t)hrows*H*2; } m->head.type=GLM5_BF16; m->head.rows=hrows; m->head.cols=H; m->head_r0=hr0; }
    { glm5_ent*e=REQ("model.norm.weight"); if(e) m->out_norm=glm5_cp_full(base,e); }

    m->layers=glm5_acalloc(cfg.n_layers,sizeof(glm5_layer));
    for(int l=0;l<cfg.n_layers&&ok;l++)
        glm5_load_one_layer(m,&m->layers[l],l,glm5_is_moe(&cfg,l),glm5_has_full_indexer(&cfg,l),
                            es,n,base,qh0,qh1,qrows,arows,sh_r0,sh_rows,ff_r0,ff_rows,&ok,&used);
    /* MTP block (checkpoint layer 78) — draft head for speculative decode. GLM5_MTP=1 (needs the
     * layer staged: GLM5_STAGE_LAYERS=79, or GLM5_STAGE_MTP=1 with a truncated main model).
     * Failure is non-fatal (MTP just stays disabled). */
    if(ok && glm5_envi("GLM5_MTP",0)){
        if(glm5_load_mtp_real(m,es,n,base,qh0,qh1,qrows,arows,sh_r0,sh_rows,&used)==0){
            if(ep_rank==0) fprintf(stderr,"glm5: MTP block loaded (checkpoint layer 78)\n");
        } else { m->mtp_layer=NULL; if(ep_rank==0) fprintf(stderr,"glm5: MTP load failed -> disabled\n"); }
    }
    #undef REQ
    munmap((void*)base,bsz); close(bfd); glm5_afree(es);
    if(!ok){ fprintf(stderr,"glm5_load: rank %d incomplete (missing tensors)\n",ep_rank); glm5_free(m); return NULL; }
    glm5_alloc_scratch(m,hrows);
    m->arena_used=used; m->arena_sz=used;
    glm5_init_fp8_lut(m->fp8_lut);
    glm5_dummy=glm5_envi("GLM5_DUMMY",0);
    if(glm5_envi("GLM5_POOL",0)) m->pool=glm5_g_pool=glm5_pool_create(m->n_threads);  /* experimental pinned pool; default OpenMP */
    return m;
}

/* Phase 2: re-slice this rank's TP-sharded DENSE weights for a new ep (after a group merge), reading
 * the FULL tensors from this node's local blob (glm5_stage stages dense un-sharded). In-place: frees
 * each old slice before installing the new one, so no 2x-weight spike. Routed experts are handled by
 * glm5_group_expert_drop; norms and the replicated attn projections (wq_a/wkv_a) are ep-invariant.
 * blob_rank = this rank's ORIGINAL staging index (MyRank % initial_ep_size); the blob holds the full
 * dense + a superset of the experts. Returns 0 on success. */
static void glm5_reslice_w(glm5_tensor*t,glm5_ent*es,int n,const uint8_t*base,const char*name,int mode,
                           int r0,int nr,int c0,int nc,int Rtot,int Ctot,int*ok){
    glm5_afree(t->w); glm5_afree(t->scale); size_t u=0;
    *t=glm5_load_w(es,n,base,name,mode,r0,nr,c0,nc,Rtot,Ctot,ok,&u);
}
static int glm5_group_tp_reslice(glm5_model*m,const char*blob_dir,int blob_rank,int new_eps,int new_ep_rank){
    glm5_config*cfg=&m->cfg;
    char bdir[1024]; if(blob_dir&&*blob_dir) snprintf(bdir,sizeof bdir,"%s",blob_dir);
    else { const char*e=getenv("GLM5_STAGE_DIR"); snprintf(bdir,sizeof bdir,"%s",(e&&*e)?e:"/local/glm5"); }
    char bp[1100],mp[1100]; snprintf(bp,sizeof bp,"%s/rank%02d.blob",bdir,blob_rank); snprintf(mp,sizeof mp,"%s/rank%02d.manifest",bdir,blob_rank);
    FILE*mf=fopen(mp,"r"); if(!mf){ fprintf(stderr,"reslice: cannot open %s\n",mp); return -1; }
    char line[1024]; int cap=4096,n=0; glm5_ent*es=glm5_amalloc((size_t)cap*sizeof(glm5_ent));
    while(fgets(line,sizeof line,mf)){
        if(line[0]=='#'||line[0]=='\n') continue;
        if(n>=cap){ cap*=2; es=realloc(es,(size_t)cap*sizeof(glm5_ent)); }
        glm5_ent*e=&es[n]; char dt[32]; int pos=0,cnt;
        if(sscanf(line,"%llu %zu %31s %d%n",(unsigned long long*)&e->off,&e->nbytes,dt,&e->nd,&cnt)!=4) continue;
        pos=cnt; e->f32=(dt[0]=='F'&&dt[1]=='3'); for(int d=0;d<e->nd&&d<5;d++){ long v; int c2; sscanf(line+pos," %ld%n",&v,&c2); e->shape[d]=v; pos+=c2; }
        while(line[pos]==' ')pos++; char*nl=strchr(line+pos,'\n'); if(nl)*nl=0; snprintf(e->name,sizeof e->name,"%s",line+pos); n++;
    }
    fclose(mf);
    int bfd=open(bp,O_RDONLY); if(bfd<0){ fprintf(stderr,"reslice: cannot open %s\n",bp); glm5_afree(es); return -1; }
    struct stat sb; fstat(bfd,&sb); size_t bsz=sb.st_size;
    /* Merge re-slice re-reads the dense tensors from the (possibly-evicted) /local blob WHILE the
     * 16 GB model weights are resident on a 31 GB node -> the 26 GB mmap thrashes against them
     * (~87 MB/s random faults, ~300 s, variable per node -> rank desync trips the 300 s gbarrier).
     * Hint sequential readahead so faults coalesce; the real safety is the raised gbarrier timeout. */
    posix_fadvise(bfd,0,bsz,POSIX_FADV_SEQUENTIAL);
    const uint8_t*base=mmap(NULL,bsz,PROT_READ,MAP_PRIVATE,bfd,0);
    if(base==MAP_FAILED){ fprintf(stderr,"reslice: mmap failed\n"); close(bfd); glm5_afree(es); return -1; }
    madvise((void*)base,bsz,MADV_SEQUENTIAL);
    const int H=cfg->hidden, QD=glm5_q_dim(cfg), AD=glm5_attn_dim(cfg), VD=cfg->v_head_dim, QHD=cfg->qk_head_dim;
    int tp=glm5_envi("GLM5_TP",0);
    int tp_attn=glm5_envi("GLM5_TP_ATTN",tp) && !m->cp_on, tp_sh=glm5_envi("GLM5_TP_SHARED",tp);
    int tp_ffn=glm5_envi("GLM5_TP_FFN",tp), tp_head=glm5_envi("GLM5_TP_HEAD",tp), tp_emb=glm5_envi("GLM5_TP_EMBED",tp);
    int qh0,qh1; if(tp_attn) glm5_shard_heads(cfg->n_heads,new_ep_rank,new_eps,&qh0,&qh1); else { qh0=0; qh1=cfg->n_heads; } int qrows=(qh1-qh0)*QHD, arows=(qh1-qh0)*VD; (void)qrows;
    int sh_r0,sh_rows; if(tp_sh) glm5_shard_shared_inter(cfg->moe_inter,new_ep_rank,new_eps,&sh_r0,&sh_rows); else { sh_r0=0; sh_rows=cfg->moe_inter; }
    int ff_r0,ff_rows; if(tp_ffn) glm5_shard(cfg->dense_inter,new_ep_rank,new_eps,&ff_r0,&ff_rows); else { ff_r0=0; ff_rows=cfg->dense_inter; }
    int hr0,hrows; if(tp_head) glm5_shard(cfg->vocab,new_ep_rank,new_eps,&hr0,&hrows); else { hr0=0; hrows=cfg->vocab; }
    int er0,erows; if(tp_emb) glm5_shard(cfg->vocab,new_ep_rank,new_eps,&er0,&erows); else { er0=0; erows=cfg->vocab; }
    int ok=1; char nb[512];
    if(tp_emb){ glm5_ent*e=glm5_req(es,n,"model.embed_tokens.weight"); if(e){ glm5_afree(m->embed); m->embed=glm5_cp_rows(base,e,er0,erows,H,2); } m->emb_r0=er0; m->emb_rows=erows; }
    if(tp_head){ glm5_reslice_w(&m->head,es,n,base,"lm_head.weight",1,hr0,hrows,0,0,cfg->vocab,H,&ok); m->head_r0=hr0; }
    for(int l=0;l<cfg->n_layers&&ok;l++){
        glm5_layer*L=&m->layers[l]; int is_moe=glm5_is_moe(cfg,l);
        #define RLN(suf) (snprintf(nb,sizeof nb,"model.layers.%d." suf,l),nb)
        if(tp_attn){
            glm5_reslice_w(&L->wq_b,es,n,base,RLN("self_attn.q_b_proj.weight"),1,qh0*QHD,(qh1-qh0)*QHD,0,0,QD,cfg->q_lora,&ok);
            glm5_reslice_w(&L->wkv_b,es,n,base,RLN("self_attn.kv_b_proj.weight"),1,qh0*(cfg->qk_nope_dim+VD),(qh1-qh0)*(cfg->qk_nope_dim+VD),0,0,cfg->n_heads*(cfg->qk_nope_dim+VD),cfg->kv_lora,&ok);
            glm5_reslice_w(&L->wo,es,n,base,RLN("self_attn.o_proj.weight"),2,0,0,qh0*VD,arows,H,AD,&ok);
            L->qh0=qh0; L->qh1=qh1;
        }
        if(is_moe){
            if(tp_sh){
                glm5_reslice_w(&L->sh_w1,es,n,base,RLN("mlp.shared_experts.gate_proj.weight"),1,sh_r0,sh_rows,0,0,cfg->moe_inter,H,&ok);
                glm5_reslice_w(&L->sh_w3,es,n,base,RLN("mlp.shared_experts.up_proj.weight"),1,sh_r0,sh_rows,0,0,cfg->moe_inter,H,&ok);
                glm5_reslice_w(&L->sh_w2,es,n,base,RLN("mlp.shared_experts.down_proj.weight"),2,0,0,sh_r0,sh_rows,H,cfg->moe_inter,&ok);
                L->sh_r0=sh_r0; L->sh_rows=sh_rows;
            }
        } else if(tp_ffn){
            glm5_reslice_w(&L->ff_gate,es,n,base,RLN("mlp.gate_proj.weight"),1,ff_r0,ff_rows,0,0,cfg->dense_inter,H,&ok);
            glm5_reslice_w(&L->ff_up,es,n,base,RLN("mlp.up_proj.weight"),1,ff_r0,ff_rows,0,0,cfg->dense_inter,H,&ok);
            glm5_reslice_w(&L->ff_down,es,n,base,RLN("mlp.down_proj.weight"),2,0,0,ff_r0,ff_rows,H,cfg->dense_inter,&ok);
            L->ff_r0=ff_r0; L->ff_rows=ff_rows;
        }
        #undef RLN
    }
    munmap((void*)base,bsz); close(bfd); glm5_afree(es);
    return ok?0:-1;
}

/* ===================== KV cache accessors (bf16 or int4, CP slot-mapped) ===================== */
/* dot(q_head, k[pos,kvh]) without the 1/sqrt(d) scale; reads bf16 or int4 at the CP slot. */
static inline float glm5_kdot(glm5_model*m,glm5_layer*L,int t,int kvh,const float*qh,int HD,int KVH){
    long sl=glm5_cp_slot(m,t);
    if(m->int4_kv) return glm5_q4_dot(L->k_q4+sl*(size_t)(KVH*HD/2)+(size_t)kvh*(HD/2), L->k_qs[sl*KVH+kvh], qh, HD);
    const uint16_t*kt=L->k_cache+sl*(size_t)(KVH*HD)+(size_t)kvh*HD; double d=0; for(int i=0;i<HD;i++) d+=(double)qh[i]*glm5_kv_dec(m,kt[i]); return (float)d;
}
/* oh[i] += w * v[pos,kvh][i] */
static inline void glm5_vaxpy(glm5_model*m,glm5_layer*L,int t,int kvh,float w,float*oh,int HD,int KVH){
    long sl=glm5_cp_slot(m,t);
    if(m->int4_kv){ glm5_q4_axpy(oh, L->v_q4+sl*(size_t)(KVH*HD/2)+(size_t)kvh*(HD/2), L->v_qs[sl*KVH+kvh], w, HD); return; }
    const uint16_t*vt=L->v_cache+sl*(size_t)(KVH*HD)+(size_t)kvh*HD; for(int i=0;i<HD;i++) oh[i]+=w*glm5_kv_dec(m,vt[i]);
}
/* GLM-5.2 MLA stores one latent KV vector [kv_lora + qk_rope] per token. */
static inline void glm5_store_latent_kv(glm5_model*m,glm5_layer*L,int t,const float*kv,int KVC){
    long sl=glm5_cp_slot(m,t);
    if(m->int4_kv){ L->k_qs[sl]=glm5_q4_pack(L->k_q4+sl*(size_t)(KVC/2),kv,KVC); return; }
    for(int i=0;i<KVC;i++) L->kv_cache[sl*(size_t)KVC+i]=glm5_kv_enc(m,kv[i]);
}
static inline void glm5_load_latent_kv(glm5_model*m,glm5_layer*L,int t,float*kv,int KVC){
    long sl=glm5_cp_slot(m,t);
    if(m->int4_kv){ glm5_q4_unpack(kv,L->k_q4+sl*(size_t)(KVC/2),L->k_qs[sl],KVC); return; }
    const uint16_t*kc=L->kv_cache+sl*(size_t)KVC;
    for(int i=0;i<KVC;i++) kv[i]=glm5_kv_dec(m,kc[i]);
}
/* dot(idx_q_head, idx_k[pos]) for MSA block scoring (1 MQA key head) */
static inline float glm5_idxdot(glm5_model*m,glm5_layer*L,int t,const float*qh,int ID){
    long sl=glm5_cp_slot(m,t);
    if(m->int4_kv) return glm5_q4_dot(L->idx_q4+sl*(size_t)(ID/2), L->idx_qs[sl], qh, ID);
    const uint16_t*kt=L->idx_k_cache+sl*(size_t)ID; float d=0; for(int i=0;i<ID;i++) d+=qh[i]*glm5_kv_dec(m,kt[i]); return d;
}

/* ===================== MSA: build the selected-position list for a sparse layer ===================== */
/* Returns nsel positions in sel[] (this rank's OWNED positions under CP); computes idx_q/idx_k,
 * stores idx_k at the owner's slot. When the block count is <= K+local+init (or MSA off) it
 * returns the full causal range (owned subset under CP). Under CP each rank scores only its
 * owned blocks, then blk_reduce_cb (all-reduce MAX) gives every rank the same global selection;
 * the cross-rank attention partials are merged later by kv_combine_cb. */
static int glm5_msa_select(glm5_model*m,glm5_layer*L,const float*xn,const float*q_lat,int pos,int msa_on,int*sel){
    const glm5_config*c=&m->cfg;
    const int ID=c->msa_index_dim, IH=c->msa_n_index_heads, B=c->msa_block_size, half=c->rotary_dim/2;
    if(!msa_on){
        int nsel=0;
        for(int t=0;t<=pos;t++) if(glm5_cp_mine(m,t)) sel[nsel++]=t;
        return nsel;
    }
    const float*cosp=&m->rope_cos[(size_t)pos*half], *sinp=&m->rope_sin[(size_t)pos*half];
    /* index projections (replicated): idx_q [IH*ID], idx_k [ID] (MQA, 1 head).
     * Real GLM-5.2 stores indexer.wq_b as [IH*ID, q_lora], fed by q_a latent.
     * Synthetic tests keep the older idx_wq [IH*ID, hidden] direct path. */
    float*iq=m->s_idx_q, *ik=m->s_idx_k;
    if(L->idx_wq_b.w) glm5_mv(m,iq,&L->idx_wq_b,q_lat,IH*ID,c->q_lora);
    else              glm5_mv(m,iq,&L->idx_wq,xn,IH*ID,c->hidden);
    glm5_mv(m,ik,&L->idx_wk,xn,ID,c->hidden);
    for(int h=0;h<IH;h++){ float*qh=iq+h*ID; if(L->idx_q_norm) glm5_rmsnorm_head(qh,L->idx_q_norm,ID,c->norm_eps); glm5_rope_head(qh,cosp,sinp,c->rotary_dim); }
    glm5_rmsnorm_head(ik,L->idx_k_norm,ID,c->norm_eps); glm5_rope_head(ik,cosp,sinp,c->rotary_dim);
    if(glm5_cp_mine(m,pos)){ long sl=glm5_cp_slot(m,pos);            /* store this pos's index key (owner only) */
        if(m->int4_kv) L->idx_qs[sl]=glm5_q4_pack(L->idx_q4+sl*(size_t)(ID/2), ik, ID);
        else for(int i=0;i<ID;i++) L->idx_k_cache[sl*(size_t)ID+i]=glm5_kv_enc(m,ik[i]); }

    int nblk=pos/B+1;
    int keep=c->msa_topk_blocks+c->msa_local_block+c->msa_init_block;
    int dense_sel = (!msa_on || nblk<=keep);   /* attend all causal positions */
    if(dense_sel){
        int nsel=0;
        for(int t=0;t<=pos;t++) if(glm5_cp_mine(m,t)) sel[nsel++]=t;
        return nsel;
    }
    char*selb=(char*)m->s_blk_sel;             /* reuse first nblk bytes as a block-selected bitmap */
    /* per-block score = max over positions of sum_h dot(idx_q_h, idx_k[t]); only OWN blocks.
     * GLM5_MSA_BLOCK_REP=1 scores one representative token per block for synthetic 1M
     * stress runs; it is an approximation and should not be used for quality validation. */
    float*bs=m->s_blk_score;
    int block_rep=glm5_envi("GLM5_MSA_BLOCK_REP",0);
    int score_blocks=glm5_envi("GLM5_MSA_SCORE_BLOCKS",0);
    int score_b0=(score_blocks>0 && nblk>score_blocks) ? nblk-score_blocks : 0;
    for(int b=0;b<nblk;b++){
        if(m->cp_on && b%m->ep_size!=m->ep_rank){ bs[b]=-1e30f; continue; }   /* not my block */
        if(b<score_b0 && b>=c->msa_init_block){ bs[b]=-1e30f; continue; }
        int t0=b*B, t1=t0+B; if(t1>pos+1)t1=pos+1; float best=-1e30f;
        if(block_rep){
            int t=t1-1; float scr=0;
            for(int h=0;h<IH;h++) scr+=glm5_idxdot(m,L,t,iq+h*ID,ID);
            best=scr;
        } else {
            for(int t=t0;t<t1;t++){ float scr=0; for(int h=0;h<IH;h++) scr+=glm5_idxdot(m,L,t,iq+h*ID,ID); if(scr>best)best=scr; }
        }
        bs[b]=best;
    }
    if(m->cp_on && m->blk_reduce_cb) m->blk_reduce_cb(bs,nblk,m->blk_reduce_ctx);  /* global block scores */
    for(int b=0;b<nblk;b++) selb[b]=0;
    for(int b=0;b<c->msa_init_block && b<nblk;b++) selb[b]=1;                 /* init blocks */
    for(int b=nblk-c->msa_local_block;b<nblk;b++) if(b>=0) selb[b]=1;          /* local blocks */
    for(int pick=0;pick<c->msa_topk_blocks;pick++){                            /* top-K by score */
        int best=-1; float bv=-1e30f;
        for(int b=0;b<nblk;b++){ if(selb[b])continue; if(bs[b]>bv){bv=bs[b];best=b;} }
        if(best<0)break; selb[best]=1;
    }
    /* gather selected positions (causal); under CP keep only my owned blocks. Write past the
     * nblk-byte bitmap region, then memmove down. */
    int nsel=0;
    for(int b=0;b<nblk;b++){ if(!selb[b])continue;
        if(m->cp_on && b%m->ep_size!=m->ep_rank) continue;     /* another rank owns this block */
        int t0=b*B,t1=t0+B; if(t1>pos+1)t1=pos+1;
        for(int t=t0;t<t1;t++) sel[nblk+nsel++]=t; }
    memmove(sel,sel+nblk,(size_t)nsel*sizeof(int));
    return nsel;
}

static glm5_tensor glm5_tensor_rows(const glm5_tensor*t,int r0,int rows);
static void glm5_tensor_tmul_rows(const glm5_model*m,float*y,const glm5_tensor*t,int r0,int rows,const float*x);

/* ===================== decode sampling (lockstep) =====================
 * Operates on the full, replicated logit vector in m->s_logits (identical on every rank).
 * With an identical rng seed + identical history on all ranks, every rank picks the SAME
 * token, so the replicated-dense / lockstep invariant is preserved exactly. Only used when
 * the lm_head is replicated (head.rows==vocab); the sharded-head path falls back to argmax. */
static inline double glm5_rng_unif(uint64_t*s){          /* xorshift64* -> [0,1) */
    uint64_t x=*s; x^=x>>12; x^=x<<25; x^=x>>27; *s=x;
    return (double)((x*0x2545F4914F6CDD1DULL)>>11) * (1.0/9007199254740992.0);
}
static const float*g_glm5_samp_lg;                       /* qsort key (decode is single-threaded) */
static int glm5_samp_cmp(const void*a,const void*b){
    float x=g_glm5_samp_lg[*(const int*)a], y=g_glm5_samp_lg[*(const int*)b];
    return (x<y)-(x>y);                                  /* descending */
}
static int glm5_sample_logits(glm5_model*m,int vocab){
    float*lg=m->s_logits;
    /* 1. repetition penalty over generated history (CTRL/HF-style: each UNIQUE token penalized
     * ONCE -- not per-occurrence, which would crush frequent tokens by pen^count and degenerate). */
    if(m->samp_rep_pen>1.0f && m->samp_hist){
        float p=m->samp_rep_pen;
        for(int i=0;i<m->samp_hist_n;i++){ int t=m->samp_hist[i];
            if((unsigned)t>=(unsigned)vocab) continue;
            int first=1; for(int j=0;j<i;j++) if(m->samp_hist[j]==t){ first=0; break; }
            if(first){ float v=lg[t]; lg[t]=(v>0.0f)?v/p:v*p; } }
    }
    /* 2. temperature + stable softmax (rewrite lg in place as probabilities) */
    float it=1.0f/m->samp_temp;
    float mx=lg[0]; for(int v=1;v<vocab;v++) if(lg[v]>mx) mx=lg[v];
    double sum=0; for(int v=0;v<vocab;v++){ float e=expf((lg[v]-mx)*it); lg[v]=e; sum+=e; }
    float inv=(float)(1.0/sum); for(int v=0;v<vocab;v++) lg[v]*=inv;
    /* 3. top-p nucleus: sort indices by prob desc, accumulate to top_p */
    int*idx=m->samp_idx; if(!idx) return 0;
    for(int v=0;v<vocab;v++) idx[v]=v;
    g_glm5_samp_lg=lg; qsort(idx,(size_t)vocab,sizeof(int),glm5_samp_cmp);
    double cum=1.0; int nuc=vocab;
    if(m->samp_topp<1.0f){ cum=0.0; for(int k=0;k<vocab;k++){ cum+=lg[idx[k]]; if(cum>=(double)m->samp_topp){ nuc=k+1; break; } } }
    /* 4. sample within the (renormalized) nucleus */
    double r=glm5_rng_unif(&m->samp_rng)*cum, acc=0; int pick=idx[0];
    for(int k=0;k<nuc;k++){ acc+=lg[idx[k]]; if(acc>=r){ pick=idx[k]; break; } }
    return pick;
}

/* ===================== forward (one token at position pos) ===================== */
static int glm5_forward_token(glm5_model*m,float*x,int pos){
    const glm5_config*c=&m->cfg;
    const int H=c->hidden, AD=glm5_attn_dim(c);
    const int KVC=glm5_kv_cache_dim(c), half=c->qk_rope_dim/2;
    const float ascale=1.0f/sqrtf((float)c->qk_head_dim);
    float*xn=m->s_norm,*q=m->s_q,*kv=m->s_k,*kvb=m->s_kvb,*attn=m->s_attn,*ao=m->s_o,*score=m->s_attn_score;
    float*qlat=m->s_idx_q; /* >= q_lora */
    const int msa_on=m->msa_on;
    const int attn_window=glm5_envi("GLM5_ATTN_WINDOW",0);
    const int dense_window=glm5_envi("GLM5_DENSE_ATTN_WINDOW",attn_window);
    const int sparse_window=glm5_envi("GLM5_SPARSE_ATTN_WINDOW",attn_window);
    m->bytes_read=0;
    int last_msa_nsel=0;
    for(int l=0;l<c->n_layers;l++){
        glm5_layer*L=&m->layers[l]; int is_moe=glm5_is_moe(c,l);
        const int qh0=L->qh0, qh1=L->qh1, nown=qh1-qh0, qrows=nown*c->qk_head_dim, arows=nown*c->v_head_dim;
        const int tp_attn=(arows<AD);
        const float*cosp=&m->rope_cos[(size_t)pos*half], *sinp=&m->rope_sin[(size_t)pos*half];
        double pt=glm5_prof_now();
        GLM5_FAPP_START("glm5_dec_qkv");
        glm5_rmsnorm_gemma(xn,x,L->input_norm,H,c->norm_eps);
        glm5_mv(m,qlat,&L->wq_a,xn,c->q_lora,H);
        glm5_rmsnorm_head(qlat,L->q_a_norm,c->q_lora,c->norm_eps);
        glm5_mv(m,q,&L->wq_b,qlat,qrows,c->q_lora);
        for(int hh=0;hh<nown;hh++) glm5_rope_interleaved(q+hh*c->qk_head_dim+c->qk_nope_dim,cosp,sinp,c->qk_rope_dim);

        glm5_mv(m,kv,&L->wkv_a,xn,KVC,H);
        glm5_rmsnorm_head(kv,L->kv_a_norm,c->kv_lora,c->norm_eps);
        glm5_rope_interleaved(kv+c->kv_lora,cosp,sinp,c->qk_rope_dim);
        if(glm5_cp_mine(m,pos)) glm5_store_latent_kv(m,L,pos,kv,KVC);
        GLM5_FAPP_STOP("glm5_dec_qkv");
        glm5_prof_add(m,GLM5_P_QKV,pt);

        pt=glm5_prof_now();
        int*selp=m->s_blk_sel, nsel=0;
        int full_idx = is_moe && glm5_has_full_indexer(c,l);
        if(is_moe && msa_on && full_idx){
            nsel=glm5_msa_select(m,L,xn,qlat,pos,msa_on,selp);
            last_msa_nsel=nsel;
        } else if(is_moe && msa_on && last_msa_nsel>0){
            /* GLM-5.2 sparse layers between full-indexer layers reuse the most
             * recent selected block set. Before the first sparse full-indexer,
             * fall through to dense/full attention below. */
            nsel=last_msa_nsel;
        } else {
            int win=is_moe ? sparse_window : dense_window;
            int t0=(win>0 && pos+1>win) ? pos+1-win : 0;
            for(int t=t0;t<=pos;t++) if(glm5_cp_mine(m,t)) selp[nsel++]=t;
        }
        glm5_prof_add(m,GLM5_P_MSA_INDEX,pt);
        pt=glm5_prof_now();
        GLM5_FAPP_START("glm5_dec_attn");
        float hmx[64], hse[64];
        const int kvb_stride=c->qk_nope_dim+c->v_head_dim;
        int absorb=glm5_envi("GLM5_ABSORB_ATTN",1) && nsel>=glm5_envi("GLM5_ABSORB_MINSEL",32);
        int absorb_sve_dot=glm5_envi("GLM5_ABSORB_SVE_DOT",1);
        if(absorb){
            for(int hh=0;hh<nown;hh++){
                hmx[hh]=-1e30f; hse[hh]=0.0f;
                float*qa=m->s_qabs+(size_t)hh*c->kv_lora;
                float*ctx=m->s_ctx+(size_t)hh*c->kv_lora;
                const float*qh=q+hh*c->qk_head_dim;
                glm5_tensor_tmul_rows(m,qa,&L->wkv_b,hh*kvb_stride,c->qk_nope_dim,qh);
                for(int i=0;i<c->kv_lora;i++) ctx[i]=0.0f;
                float*oh=attn+hh*c->v_head_dim;
                for(int i=0;i<c->v_head_dim;i++) oh[i]=0.0f;
            }
            for(int j=0;j<nsel;j++){
                glm5_load_latent_kv(m,L,selp[j],kv,KVC);
                for(int hh=0;hh<nown;hh++){
                    const float*qh=q+hh*c->qk_head_dim;
                    const float*qa=m->s_qabs+(size_t)hh*c->kv_lora;
                    double d=(double)glm5_dot_f32_opt(qa,kv,c->kv_lora,absorb_sve_dot);
                    for(int i=0;i<c->qk_rope_dim;i++) d+=(double)qh[c->qk_nope_dim+i]*kv[c->kv_lora+i];
                    float s=(float)d*ascale;
                    float*ctx=m->s_ctx+(size_t)hh*c->kv_lora;
                    if(s>hmx[hh]){
                        float r=(hmx[hh]>-1e20f)?expf(hmx[hh]-s):0.0f;
                        hse[hh]*=r;
                        glm5_scale_f32(ctx,r,c->kv_lora);
                        hmx[hh]=s;
                    }
                    float e=expf(s-hmx[hh]);
                    hse[hh]+=e;
                    glm5_axpy_f32(ctx,kv,e,c->kv_lora);
                }
            }
            for(int hh=0;hh<nown;hh++){
                glm5_tensor tv=glm5_tensor_rows(&L->wkv_b,hh*kvb_stride+c->qk_nope_dim,c->v_head_dim);
                glm5_mv(m,attn+hh*c->v_head_dim,&tv,m->s_ctx+(size_t)hh*c->kv_lora,c->v_head_dim,c->kv_lora);
            }
        } else {
            const int kvb_rows=nown*kvb_stride;
            for(int hh=0;hh<nown;hh++){
                hmx[hh]=-1e30f; hse[hh]=0.0f;
                float*oh=attn+hh*c->v_head_dim;
                for(int i=0;i<c->v_head_dim;i++) oh[i]=0.0f;
            }
            for(int j=0;j<nsel;j++){
                glm5_load_latent_kv(m,L,selp[j],kv,KVC);
                glm5_mv(m,kvb,&L->wkv_b,kv,kvb_rows,c->kv_lora);
                for(int hh=0;hh<nown;hh++){
                    const float*qh=q+hh*c->qk_head_dim;
                    const float*kn=kvb+hh*kvb_stride;
                    double d=0;
                    for(int i=0;i<c->qk_nope_dim;i++) d+=(double)qh[i]*kn[i];
                    for(int i=0;i<c->qk_rope_dim;i++) d+=(double)qh[c->qk_nope_dim+i]*kv[c->kv_lora+i];
                    float s=(float)d*ascale;
                    score[(size_t)hh*c->max_pos+j]=s;
                    if(s>hmx[hh]) hmx[hh]=s;
                }
            }
            for(int j=0;j<nsel;j++){
                glm5_load_latent_kv(m,L,selp[j],kv,KVC);
                glm5_mv(m,kvb,&L->wkv_b,kv,kvb_rows,c->kv_lora);
                for(int hh=0;hh<nown;hh++){
                    float e=expf(score[(size_t)hh*c->max_pos+j]-hmx[hh]);
                    hse[hh]+=e;
                    const float*vv=kvb+hh*kvb_stride+c->qk_nope_dim;
                    float*oh=attn+hh*c->v_head_dim;
                    for(int i=0;i<c->v_head_dim;i++) oh[i]+=e*vv[i];
                }
            }
        }
        if(m->cp_on && m->kv_combine_cb) m->kv_combine_cb(attn,hmx,hse,nown,c->v_head_dim,m->kv_combine_ctx);
        else for(int hh=0;hh<nown;hh++){ float inv=1.0f/(hse[hh]>0?hse[hh]:1); float*oh=attn+hh*c->v_head_dim; for(int i=0;i<c->v_head_dim;i++) oh[i]*=inv; }
        GLM5_FAPP_STOP("glm5_dec_attn");
        glm5_prof_add(m,GLM5_P_ATTN,pt);
        pt=glm5_prof_now();
        GLM5_FAPP_START("glm5_dec_o_proj");
        glm5_mv(m,ao,&L->wo,attn,H,arows);
        if(tp_attn && m->ar_cb) m->ar_cb(ao,H,m->ar_ctx);
        for(int i=0;i<H;i++) x[i]+=ao[i];
        GLM5_FAPP_STOP("glm5_dec_o_proj");
        glm5_prof_add(m,GLM5_P_OPROJ,pt);
        /* FFN / MoE */
        pt=glm5_prof_now();
        float*h2=m->s_norm; glm5_rmsnorm_gemma(h2,x,L->post_norm,H,c->norm_eps);
        glm5_prof_add(m,GLM5_P_OTHER,pt);
        if(is_moe){
            const int tp_sh=(L->sh_rows<c->moe_inter);
            pt=glm5_prof_now();
            GLM5_FAPP_START("glm5_dec_router");
            float*rl=m->s_router; glm5_mv(m,rl,&L->gate,h2,c->n_experts,H);
            glm5_sigmoid_row(rl,c->n_experts);   /* FEXPA SVE sigmoid (was scalar expf loop) */
            int selx[64]; float selw[64]; int na=c->n_active>64?64:c->n_active;
            for(int a=0;a<na;a++){ int best=-1; float bv=-1e30f;
                for(int e=0;e<c->n_experts;e++){ int used=0; for(int j=0;j<a;j++) if(selx[j]==e){used=1;break;} if(used)continue;
                    float v=rl[e]+L->gate_bias[e]; if(v>bv){bv=v;best=e;} } selx[a]=best; selw[a]=rl[best]; }
            float wsum=0; for(int a=0;a<na;a++) wsum+=selw[a]; if(wsum<=0)wsum=1;
            float*route=m->s_route; for(int i=0;i<H;i++) route[i]=0;
            GLM5_FAPP_STOP("glm5_dec_router");
            glm5_prof_add(m,GLM5_P_ROUTER,pt);
            pt=glm5_prof_now();
            GLM5_FAPP_START("glm5_dec_experts");
            for(int a=0;a<na;a++){ int e=selx[a]; if(e%m->ep_size!=m->ep_rank) continue; int slot=e/m->ep_size;
                float w=selw[a]/wsum*c->routed_scale;
                glm5_mv(m,m->s_exg,&L->ex_w1[slot],h2,c->moe_inter,H);
                glm5_mv(m,m->s_exu,&L->ex_w3[slot],h2,c->moe_inter,H);
                for(int i=0;i<c->moe_inter;i++) m->s_exg[i]=glm5_swiglu_oai(m->s_exg[i],m->s_exu[i],c->swiglu_alpha,c->swiglu_limit);
                glm5_mv(m,m->s_moe,&L->ex_w2[slot],m->s_exg,H,c->moe_inter);
                glm5_axpy_f32(route, m->s_moe, w, H); }   /* SVE: was scalar route[i]+=w*s_moe[i] */
            GLM5_FAPP_STOP("glm5_dec_experts");
            glm5_prof_add(m,GLM5_P_EXPERTS,pt);
            /* shared expert: TP-sharded -> fold partial into route[] (one reduce); else replicated -> add after */
            int overlap=(m->ar_async_start && !tp_sh);
            if(overlap) m->ar_async_start(route,H,m->ar_async_ctx);
            pt=glm5_prof_now();
            GLM5_FAPP_START("glm5_dec_shared");
            glm5_mv(m,m->s_shg,&L->sh_w1,h2,L->sh_rows,H);
            glm5_mv(m,m->s_shu,&L->sh_w3,h2,L->sh_rows,H);
            for(int i=0;i<L->sh_rows;i++) m->s_shg[i]=glm5_swiglu_oai(m->s_shg[i],m->s_shu[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_mv(m,m->s_sh,&L->sh_w2,m->s_shg,H,L->sh_rows);
            if(tp_sh) for(int i=0;i<H;i++) route[i]+=m->s_sh[i];
            GLM5_FAPP_STOP("glm5_dec_shared");
            glm5_prof_add(m,GLM5_P_SHARED,pt);
            if(overlap) m->ar_wait(m->ar_async_ctx);
            else if(m->ar_cb) m->ar_cb(route,H,m->ar_ctx);     /* EP-sum routed (+ shared if TP) */
            for(int i=0;i<H;i++) x[i]+=route[i] + (tp_sh?0.0f:m->s_sh[i]);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            pt=glm5_prof_now();
            GLM5_FAPP_START("glm5_dec_dense_ffn");
            glm5_mv(m,m->s_ff_g,&L->ff_gate,h2,L->ff_rows,H);
            glm5_mv(m,m->s_ff_u,&L->ff_up,h2,L->ff_rows,H);
            for(int i=0;i<L->ff_rows;i++) m->s_ff_g[i]=glm5_swiglu_oai(m->s_ff_g[i],m->s_ff_u[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_mv(m,m->s_ff,&L->ff_down,m->s_ff_g,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(m->s_ff,H,m->ar_ctx);
            for(int i=0;i<H;i++) x[i]+=m->s_ff[i];
            GLM5_FAPP_STOP("glm5_dec_dense_ffn");
            glm5_prof_add(m,GLM5_P_DENSE_FFN,pt);
        }
    }
    /* head: vocab-shard partial logits -> (TP_HEAD) global argmax via ar_argmax, else full */
    double pt=glm5_prof_now();
    GLM5_FAPP_START("glm5_dec_head");
    float*h2=m->s_norm; glm5_rmsnorm_gemma(h2,x,m->out_norm,H,c->norm_eps);
    int hrows=m->head.rows;
    /* sampling: needs the FULL distribution on every rank. Replicated head -> s_logits is already
     * full; vocab-SHARDED head -> scatter each shard to its global offset + EP all-reduce to gather
     * the full logit vector identically on every rank (lockstep), then sample. */
    if(m->samp_temp>0.0f){
        if(hrows<c->vocab){
            memset(m->s_logits,0,(size_t)c->vocab*4);
            glm5_mv(m,m->s_logits+m->head_r0,&m->head,h2,hrows,H);
            if(m->ar_cb) m->ar_cb(m->s_logits,c->vocab,m->ar_ctx);
        } else glm5_mv(m,m->s_logits,&m->head,h2,hrows,H);
        int t=glm5_sample_logits(m,c->vocab);
        GLM5_FAPP_STOP("glm5_dec_head");
        glm5_prof_add(m,GLM5_P_HEAD,pt);
        return t;
    }
    glm5_mv(m,m->s_logits,&m->head,h2,hrows,H);   /* greedy: shard (or full) at offset 0 */
    /* GLM5_LOGIT_DUMP=<file>: append the full logit vector per forward_token call (rank 0, replicated
     * head only). For the SDOT-vs-w8a16 numeric A/B (cosine/rms) fed identical inputs (teacher-forcing
     * / prefill-only) so there is no autoregressive divergence. */
    if(m->ep_rank==0 && hrows>=c->vocab){ static FILE*ldf=(FILE*)-1;
        if(ldf==(FILE*)-1){ const char*e=getenv("GLM5_LOGIT_DUMP"); ldf=(e&&*e)?fopen(e,"wb"):NULL; }
        if(ldf){ fwrite(m->s_logits,4,(size_t)c->vocab,ldf); fflush(ldf); } }
    int la=0; float lv=m->s_logits[0]; for(int i=1;i<hrows;i++) if(m->s_logits[i]>lv){lv=m->s_logits[i];la=i;}
    int32_t gidx=m->head_r0+la; float gval=lv;
    if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);  /* TP_HEAD merge */
    GLM5_FAPP_STOP("glm5_dec_head");
    glm5_prof_add(m,GLM5_P_HEAD,pt);
    return gidx;
}

/* ===================== multi-stream batched decode (raise M -> amortize dispatch+comm) =====
 * N concurrent sequences per forward. The dense projections (qkv/o/shared/ffn/head) become
 * batched GEMMs (weight read ONCE per K-tile, reused across the N tokens), and the EP
 * all-reduce fires ONCE per layer over the whole [N,hidden] partial -> dispatch + comm
 * amortized N-fold (the costs the dummy ceiling showed dominate single-stream). Each stream
 * keeps its own KV cache + position; attention is per-stream (full causal; MSA selection is
 * deferred in the batched path). Routed experts stay M=1 per (token,expert) for v1 (grouping
 * is a follow-on) but the dominant comm is amortized. */
typedef struct {
    int n;
    uint16_t *kc, *vc;        /* [n][n_layers][max_pos][kv_dim] bf16 per-stream KV caches */
    float *xn,*q,*k,*v,*qlat,*kvb,*attn,*o,*h2,*router,*route,*shg,*shu,*ffg,*ffu,*tmp2;  /* token-major batched scratch */
    float *exg,*exu,*emoe;    /* per-token expert scratch (M=1) */
    int *bk; float *bw; int *bcnt;   /* expert grouping: per-owned-slot token buckets [n_experts*N]/[n_experts] */
    float *logits,*sc; int sc_stride;
    int *psel, *pnsel;        /* prefill: per-token MSA selected positions [n*maxsel] + counts [n] (kc/vc NULL) */
    int maxsel;
    int *gsel; float *gselw;  /* router top-k per token [n*8] (heap; prefill N can exceed 64) */
    /* batched-parallel MSA select (prefill, non-CP): batched idx_q/idx_k projections + per-token
     * block bitmap, so the O(pos) block scoring runs OpenMP-parallel over the chunk. */
    float *piq, *pik;         /* [n*idx_q_dim], [n*index_dim] batched index projections */
    char  *pbit; int nblkmax; /* [n*nblkmax] per-token block bitmap (nblkmax=ceil(max_pos/block)) */
    float *hmx, *hse;         /* [n*64] per-token CP softmax max/sumexp -> deferred ordered combine */
    float *slog;              /* [n*vocab] full-logit gather for batched SAMPLING (one AR, not n) */
    const int *const *hist; const int *hist_n;  /* per-stream rep-penalty history (caller-set; may be NULL) */
    const int *sid;           /* batch entry -> KV stream map (caller-set; NULL = identity).
                               * This is valid for independent service slots and prompt-style
                               * teacher-forcing rows. It does NOT make a same-stream speculative
                               * chain equivalent to autoregressive decode: later rows do not consume
                               * earlier rows' newly computed hidden states layer-by-layer. */
} glm5_mstream;

/* Y[N,rows] (token-major) = X[N,cols] . W[rows,cols]^T. K-tiled (8 W rows for a tile stay
 * in L1 across the N tokens -> weight read amortized over N). OpenMP over 8-row blocks. */
static void glm5_gemm_bf16(float*restrict Y, const uint16_t*W, const float*X, int N, int rows, int cols){
    if(glm5_dummy){
        double acc=0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
        for(int r=0;r<rows;r++){ const uint16_t*w=W+(size_t)r*cols; uint32_t s=0; for(int i=0;i<cols;i+=32) s+=w[i]; acc+=s; }
        float vv=(float)(((long)acc)&1)*1e-30f; for(size_t i=0;i<(size_t)N*rows;i++) Y[i]=vv; return;
    }
    int nb=rows/8, TILE=512; if(TILE>cols)TILE=cols;
    static int tok=-1; if(tok<0){ tok=glm5_envi("GLM5_BF16_GEMM_TOK",3); if(tok!=3&&tok!=4&&tok!=5) tok=3; }
#ifdef _OPENMP
    /* parallelize on TOTAL work, not rows: the MoE gate is [256,6144] (rows<GLM5_PAR_MIN) but a
     * huge GEMM -> the old rows>=512 guard ran it SINGLE-THREADED (~16 Gop/s, ~26% of prefill). */
    #pragma omp parallel for schedule(static) if((long)nb*(size_t)cols>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const uint16_t*w=W+(size_t)r*cols;
        float acc[N][8];
        for(int t=0;t<N;t++) for(int j=0;j<8;j++) acc[t][j]=0.f;
        for(int k0=0;k0<cols;k0+=TILE){ int kl=cols-k0<TILE?cols-k0:TILE; const uint16_t*tw=w+k0;
            int t=0;
#if defined(__ARM_FEATURE_SVE)
            if(tok==5){
                for(;t+4<N;t+=5){
                    const float*x0=X+(size_t)t*cols+k0,*x1=X+(size_t)(t+1)*cols+k0,*x2=X+(size_t)(t+2)*cols+k0,*x3=X+(size_t)(t+3)*cols+k0,*x4=X+(size_t)(t+4)*cols+k0;
                    glm5_bf16_4row_5x_acc(acc[t],acc[t+1],acc[t+2],acc[t+3],acc[t+4],tw,tw+cols,tw+2*(size_t)cols,tw+3*(size_t)cols,x0,x1,x2,x3,x4,kl);
                    glm5_bf16_4row_5x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,acc[t+3]+4,acc[t+4]+4,tw+4*(size_t)cols,tw+5*(size_t)cols,tw+6*(size_t)cols,tw+7*(size_t)cols,x0,x1,x2,x3,x4,kl);
                }
            } else if(tok==4){
                for(;t+3<N;t+=4){
                    const float*x0=X+(size_t)t*cols+k0,*x1=X+(size_t)(t+1)*cols+k0,*x2=X+(size_t)(t+2)*cols+k0,*x3=X+(size_t)(t+3)*cols+k0;
                    glm5_bf16_4row_4x_acc(acc[t],acc[t+1],acc[t+2],acc[t+3],tw,tw+cols,tw+2*(size_t)cols,tw+3*(size_t)cols,x0,x1,x2,x3,kl);
                    glm5_bf16_4row_4x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,acc[t+3]+4,tw+4*(size_t)cols,tw+5*(size_t)cols,tw+6*(size_t)cols,tw+7*(size_t)cols,x0,x1,x2,x3,kl);
                }
            }
            for(;t+2<N;t+=3){
                glm5_bf16_4row_3x_acc(acc[t],acc[t+1],acc[t+2],tw,tw+cols,tw+2*(size_t)cols,tw+3*(size_t)cols,
                                      X+(size_t)t*cols+k0,X+(size_t)(t+1)*cols+k0,X+(size_t)(t+2)*cols+k0,kl);
                glm5_bf16_4row_3x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,tw+4*(size_t)cols,tw+5*(size_t)cols,tw+6*(size_t)cols,tw+7*(size_t)cols,
                                      X+(size_t)t*cols+k0,X+(size_t)(t+1)*cols+k0,X+(size_t)(t+2)*cols+k0,kl);
            }
#endif
            for(;t<N;t++){ float tmp[8];
                matvec_bf16_8row(tmp,tw,tw+cols,tw+2*(size_t)cols,tw+3*(size_t)cols,
                                 tw+4*(size_t)cols,tw+5*(size_t)cols,tw+6*(size_t)cols,tw+7*(size_t)cols,
                                 X+(size_t)t*cols+k0,kl);
                for(int j=0;j<8;j++) acc[t][j]+=tmp[j]; } }
        for(int t=0;t<N;t++){ float*y=Y+(size_t)t*rows+r; for(int j=0;j<8;j++) y[j]=acc[t][j]; }
    }
    for(int r=nb*8;r<rows;r++) for(int t=0;t<N;t++) Y[(size_t)t*rows+r]=vec_dot_bf16_f32(W+(size_t)r*cols,X+(size_t)t*cols,cols);
}

static void glm5_gemm_f32(float*restrict Y, const float*W, const float*X, int N, int rows, int cols){
    int nb=rows/8;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const float*w=W+(size_t)r*cols;
        float acc[N][8];
        for(int t=0;t<N;t++) for(int j=0;j<8;j++) acc[t][j]=0.f;
        int t=0;
#if defined(__ARM_FEATURE_SVE)
        for(;t+2<N;t+=3){
            glm5_f32_4row_3x_acc(acc[t],acc[t+1],acc[t+2],w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,
                                 X+(size_t)t*cols,X+(size_t)(t+1)*cols,X+(size_t)(t+2)*cols,cols);
            glm5_f32_4row_3x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
                                 X+(size_t)t*cols,X+(size_t)(t+1)*cols,X+(size_t)(t+2)*cols,cols);
        }
#endif
        for(;t<N;t++){ float tmp[8];
            for(int j=0;j<8;j++){ const float*wr=w+(size_t)j*cols; double s=0; for(int c=0;c<cols;c++) s+=(double)wr[c]*X[(size_t)t*cols+c]; tmp[j]=(float)s; }
            for(int j=0;j<8;j++) acc[t][j]+=tmp[j]; }
        for(t=0;t<N;t++){ float*y=Y+(size_t)t*rows+r; for(int j=0;j<8;j++) y[j]=acc[t][j]; }
    }
    for(int r=nb*8;r<rows;r++) for(int t=0;t<N;t++){ const float*w=W+(size_t)r*cols; const float*x=X+(size_t)t*cols; double s=0; for(int c=0;c<cols;c++) s+=(double)w[c]*x[c]; Y[(size_t)t*rows+r]=(float)s; }
}

/* batched MXFP8 GEMM: Y[N,rows] = decode(W fp8 + S e8m0) . X[N,cols]. For N>1 the expensive
 * part is the FP8 LUT-gather decode -- so we decode each 8-row x TILE block ONCE into a bf16
 * tile (FP8*2^k is exact in bf16), then run N cheap matvec_bf16_8row over it. This amortizes
 * the decode by N (the multi-stream win). N==1 uses the direct fused kernel (no tile staging). */
#define GLM5_MXG_TILE 512
static void glm5_gemm_mxfp8(glm5_model*m, float*restrict Y, const uint8_t*W, const uint8_t*S, const float*X, int N, int rows, int cols){
    if(glm5_dummy){ double acc=0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
        for(int r=0;r<rows;r++){ const uint8_t*w=W+(size_t)r*cols; uint32_t s=0; for(int i=0;i<cols;i+=64) s+=w[i]; acc+=s; }
        float v=(float)(((long)acc)&1)*1e-30f; for(size_t i=0;i<(size_t)N*rows;i++) Y[i]=v; return; }
    if(N<=1){ glm5_mv_mxfp8(m,Y,W,S,X,rows,cols); return; }
    const uint32_t*lut=m->fp8_lut;
    const float*Sc=(const float*)S;
    const int sb=(cols+127)/128;
    const int nb=rows/8;
    /* The inner matvec is L1-load-bound on the bf16 weight tile (an f32 tile measured slower).
     * So block several tokens per weight-vector load to cut relative weight traffic and expose
     * more independent FMA chains. GLM5_FP8_GEMM_TOK selects the token block: 5 (default, 20 acc
     * +4 w +1 x = 25 SVE regs), 4, or 3 (previous kernel). All bit-exact. Kernel bench N=256:
     * tok=3 230 -> tok=4 295 -> tok=5 330 Gop/s; full 96n prefill tok=3 14.5 -> tok=5 16.7 tok/s. */
    static int tok=-1; if(tok<0){ tok=glm5_envi("GLM5_FP8_GEMM_TOK",5); if(tok<3||tok>5) tok=5; }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8;
        uint16_t tile[8*GLM5_MXG_TILE];
        float acc[N][8];
        for(int t=0;t<N;t++) for(int j=0;j<8;j++) acc[t][j]=0.0f;
        for(int k0=0;k0<cols;k0+=GLM5_MXG_TILE){
            int kl=cols-k0<GLM5_MXG_TILE?cols-k0:GLM5_MXG_TILE;
            for(int j=0;j<8;j++){
                const uint8_t*wrow=W+(size_t)(r+j)*cols+k0;
                const float*srow=Sc+(size_t)(r+j)*sb;
                uint16_t*trow=tile+(size_t)j*kl;
                glm5_mxfp8_f32scale_decode_row_bf16(trow,wrow,srow,k0,kl,lut);
            }
            int t=0;
#if defined(__ARM_FEATURE_SVE)
            if(tok==5){
                for(;t+4<N;t+=5){
                    const float*x0=X+(size_t)t*cols+k0,*x1=X+(size_t)(t+1)*cols+k0,*x2=X+(size_t)(t+2)*cols+k0,*x3=X+(size_t)(t+3)*cols+k0,*x4=X+(size_t)(t+4)*cols+k0;
                    glm5_bf16_4row_5x_acc(acc[t],acc[t+1],acc[t+2],acc[t+3],acc[t+4],tile,tile+kl,tile+2*(size_t)kl,tile+3*(size_t)kl,x0,x1,x2,x3,x4,kl);
                    glm5_bf16_4row_5x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,acc[t+3]+4,acc[t+4]+4,tile+4*(size_t)kl,tile+5*(size_t)kl,tile+6*(size_t)kl,tile+7*(size_t)kl,x0,x1,x2,x3,x4,kl);
                }
            } else if(tok==4){
                for(;t+3<N;t+=4){
                    const float*x0=X+(size_t)t*cols+k0,*x1=X+(size_t)(t+1)*cols+k0,*x2=X+(size_t)(t+2)*cols+k0,*x3=X+(size_t)(t+3)*cols+k0;
                    glm5_bf16_4row_4x_acc(acc[t],acc[t+1],acc[t+2],acc[t+3],tile,tile+kl,tile+2*(size_t)kl,tile+3*(size_t)kl,x0,x1,x2,x3,kl);
                    glm5_bf16_4row_4x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,acc[t+3]+4,tile+4*(size_t)kl,tile+5*(size_t)kl,tile+6*(size_t)kl,tile+7*(size_t)kl,x0,x1,x2,x3,kl);
                }
            }
            for(;t+2<N;t+=3){
                glm5_bf16_4row_3x_acc(acc[t],acc[t+1],acc[t+2],tile,tile+kl,tile+2*(size_t)kl,tile+3*(size_t)kl,
                                      X+(size_t)t*cols+k0,X+(size_t)(t+1)*cols+k0,X+(size_t)(t+2)*cols+k0,kl);
                glm5_bf16_4row_3x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,tile+4*(size_t)kl,tile+5*(size_t)kl,tile+6*(size_t)kl,tile+7*(size_t)kl,
                                      X+(size_t)t*cols+k0,X+(size_t)(t+1)*cols+k0,X+(size_t)(t+2)*cols+k0,kl);
            }
#endif
            for(;t<N;t++){
                float tmp[8];
                matvec_bf16_8row(tmp,
                    tile,tile+kl,tile+2*(size_t)kl,tile+3*(size_t)kl,
                    tile+4*(size_t)kl,tile+5*(size_t)kl,tile+6*(size_t)kl,tile+7*(size_t)kl,
                    X+(size_t)t*cols+k0,kl);
                for(int j=0;j<8;j++) acc[t][j]+=tmp[j];
            }
        }
        for(int t=0;t<N;t++){ float*y=Y+(size_t)t*rows+r; for(int j=0;j<8;j++) y[j]=acc[t][j]; }
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
    for(int r=nb*8;r<rows;r++){
        for(int t=0;t<N;t++) Y[(size_t)t*rows+r]=glm5_dot_mxfp8_f32scale_row(W+(size_t)r*cols,Sc+(size_t)r*sb,X+(size_t)t*cols,cols,lut);
    }
}
/* INT8 batched GEMM: decode each 8-row x TILE block once into a bf16 tile, then run the same
 * bf16 multi-token tile kernels as the FP8 path (amortizes decode by N). gs = scale group size. */
static void glm5_gemm_int8(glm5_model*m, float*restrict Y, const uint8_t*W, const float*S, int gs, int qg0, const float*X, int N, int rows, int cols){
    if(glm5_dummy){ for(size_t i=0;i<(size_t)N*rows;i++) Y[i]=0.f; (void)m; return; }
    if(N<=1){ glm5_mv_int8(m,Y,W,S,gs,qg0,X,rows,cols); return; }
    const int sb=(qg0+cols+gs-1)/gs;
    const int nb=rows/8;
    static int tok=-1; if(tok<0){ tok=glm5_envi("GLM5_FP8_GEMM_TOK",5); if(tok<3||tok>5) tok=5; }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8;
        uint16_t tile[8*GLM5_MXG_TILE];
        float acc[N][8];
        for(int t=0;t<N;t++) for(int j=0;j<8;j++) acc[t][j]=0.0f;
        for(int k0=0;k0<cols;k0+=GLM5_MXG_TILE){
            int kl=cols-k0<GLM5_MXG_TILE?cols-k0:GLM5_MXG_TILE;
            for(int j=0;j<8;j++){
                const uint8_t*wrow=W+(size_t)(r+j)*cols+k0;
                const float*srow=S+(size_t)(r+j)*sb;
                glm5_int8_decode_row_bf16(tile+(size_t)j*kl,wrow,srow,gs,qg0+k0,kl);
            }
            int t=0;
#if defined(__ARM_FEATURE_SVE)
            if(tok==5){
                for(;t+4<N;t+=5){
                    const float*x0=X+(size_t)t*cols+k0,*x1=X+(size_t)(t+1)*cols+k0,*x2=X+(size_t)(t+2)*cols+k0,*x3=X+(size_t)(t+3)*cols+k0,*x4=X+(size_t)(t+4)*cols+k0;
                    glm5_bf16_4row_5x_acc(acc[t],acc[t+1],acc[t+2],acc[t+3],acc[t+4],tile,tile+kl,tile+2*(size_t)kl,tile+3*(size_t)kl,x0,x1,x2,x3,x4,kl);
                    glm5_bf16_4row_5x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,acc[t+3]+4,acc[t+4]+4,tile+4*(size_t)kl,tile+5*(size_t)kl,tile+6*(size_t)kl,tile+7*(size_t)kl,x0,x1,x2,x3,x4,kl);
                }
            } else if(tok==4){
                for(;t+3<N;t+=4){
                    const float*x0=X+(size_t)t*cols+k0,*x1=X+(size_t)(t+1)*cols+k0,*x2=X+(size_t)(t+2)*cols+k0,*x3=X+(size_t)(t+3)*cols+k0;
                    glm5_bf16_4row_4x_acc(acc[t],acc[t+1],acc[t+2],acc[t+3],tile,tile+kl,tile+2*(size_t)kl,tile+3*(size_t)kl,x0,x1,x2,x3,kl);
                    glm5_bf16_4row_4x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,acc[t+3]+4,tile+4*(size_t)kl,tile+5*(size_t)kl,tile+6*(size_t)kl,tile+7*(size_t)kl,x0,x1,x2,x3,kl);
                }
            }
            for(;t+2<N;t+=3){
                glm5_bf16_4row_3x_acc(acc[t],acc[t+1],acc[t+2],tile,tile+kl,tile+2*(size_t)kl,tile+3*(size_t)kl,
                                      X+(size_t)t*cols+k0,X+(size_t)(t+1)*cols+k0,X+(size_t)(t+2)*cols+k0,kl);
                glm5_bf16_4row_3x_acc(acc[t]+4,acc[t+1]+4,acc[t+2]+4,tile+4*(size_t)kl,tile+5*(size_t)kl,tile+6*(size_t)kl,tile+7*(size_t)kl,
                                      X+(size_t)t*cols+k0,X+(size_t)(t+1)*cols+k0,X+(size_t)(t+2)*cols+k0,kl);
            }
#endif
            for(;t<N;t++){
                float tmp[8];
                matvec_bf16_8row(tmp,tile,tile+kl,tile+2*(size_t)kl,tile+3*(size_t)kl,
                    tile+4*(size_t)kl,tile+5*(size_t)kl,tile+6*(size_t)kl,tile+7*(size_t)kl,
                    X+(size_t)t*cols+k0,kl);
                for(int j=0;j<8;j++) acc[t][j]+=tmp[j];
            }
        }
        for(int t=0;t<N;t++){ float*y=Y+(size_t)t*rows+r; for(int j=0;j<8;j++) y[j]=acc[t][j]; }
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
    for(int r=nb*8;r<rows;r++)
        for(int t=0;t<N;t++) Y[(size_t)t*rows+r]=glm5_dot_int8_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,qg0,X+(size_t)t*cols,cols);
}
/* INT8 w8a8 sdot GEMM: dynamically quantize activations to int8 per token, then use the SVE
 * SDOT (4-way int8->int32, 64 MACs/instr) for the contraction. Weights are offset-binary so the
 * signed value is byte^0x80. Per group g: d = SDOT_c( (wr[c]^0x80), xq[c] ) -> 16 int32 lanes;
 * convert and fold the group scale into an f32 accumulator (svaddv is linear, so the per-group
 * horizontal reduce becomes ONE reduce per (row,token)). y = xsc[t] * sum_g wsc[r,g]*groupdot.
 * Lossier than the w8a16 path (activations rounded to int8) -> opt-in via GLM5_INT8_SDOT. */
static void glm5_gemm_int8_sdot(glm5_model*m, float*restrict Y, const uint8_t*W, const float*S, int gs, int qg0,
                                const float*X, int N, int rows, int cols){
    int sb=(qg0+cols+gs-1)/gs;
    int8_t *Xq=(int8_t*)malloc((size_t)N*cols); float *xsc=(float*)malloc((size_t)N*sizeof(float));
    if(!Xq||!xsc){ free(Xq); free(xsc); glm5_gemm_int8(m,Y,W,S,gs,qg0,X,N,rows,cols); return; }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(N>=4)
#endif
    for(int t=0;t<N;t++){
        const float*xt=X+(size_t)t*cols; float amax=1e-20f;
        for(int c=0;c<cols;c++){ float a=fabsf(xt[c]); if(a>amax)amax=a; }
        float inv=127.0f/amax; xsc[t]=amax/127.0f; int8_t*xq=Xq+(size_t)t*cols;
        for(int c=0;c<cols;c++){ int v=(int)lrintf(xt[c]*inv); v=v>127?127:(v<-127?-127:v); xq[c]=(int8_t)v; }
    }
#if defined(__ARM_FEATURE_SVE)
    int vb=(int)svcntb(); svbool_t pf=svptrue_b32();
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
    for(int r=0;r<rows;r++){
        const uint8_t*wr=W+(size_t)r*cols; const float*sr=S+(size_t)r*sb;
        for(int t=0;t<N;t++){
            const int8_t*xq=Xq+(size_t)t*cols;
            svfloat32_t facc=svdup_f32(0.f);
            for(int g=0;g<sb;g++){
                int k0=g*gs-qg0; if(k0<0)k0=0;
                int k1=(g+1)*gs-qg0; if(k1>cols)k1=cols;
                int c=k0; if(k0>=k1) continue;
                svint32_t d0=svdup_s32(0),d1=svdup_s32(0),d2=svdup_s32(0),d3=svdup_s32(0);
                #define GLM5_SDOTC(D,CC) do{ svint8_t wv=svreinterpret_s8_u8(sveor_n_u8_x(pf8,svld1_u8(pf8,&wr[CC]),0x80)); \
                    D=svdot_s32(D,wv,svld1_s8(pf8,&xq[CC])); }while(0)
                svbool_t pf8=svptrue_b8();
                for(;c+4*vb<=k1;c+=4*vb){ GLM5_SDOTC(d0,c); GLM5_SDOTC(d1,c+vb); GLM5_SDOTC(d2,c+2*vb); GLM5_SDOTC(d3,c+3*vb); }
                #undef GLM5_SDOTC
                for(;c<k1;c+=vb){ svbool_t pg=svwhilelt_b8(c,k1);
                    svint8_t wv=svreinterpret_s8_u8(sveor_n_u8_x(pg,svld1_u8(pg,&wr[c]),0x80));
                    d0=svdot_s32(d0,wv,svld1_s8(pg,&xq[c])); }
                svint32_t d=svadd_s32_x(pf,svadd_s32_x(pf,d0,d1),svadd_s32_x(pf,d2,d3));
                facc=svmla_n_f32_x(pf,facc,svcvt_f32_s32_x(pf,d),sr[g]);
            }
            Y[(size_t)t*rows+r]=xsc[t]*svaddv_f32(pf,facc);
        }
    }
#else
    glm5_gemm_int8(m,Y,W,S,gs,qg0,X,N,rows,cols);
#endif
    free(Xq); free(xsc);
}
static void glm5_quantize_i16_acts(int16_t *restrict Xq, float *restrict xsc, const float *X, int N, int cols){
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(N>=4)
#endif
    for(int t=0;t<N;t++){
        const float*xt=X+(size_t)t*cols; float amax=1e-20f;
        for(int c=0;c<cols;c++){ float a=fabsf(xt[c]); if(a>amax)amax=a; }
        float inv=32767.0f/amax; xsc[t]=amax/32767.0f; int16_t*xq=Xq+(size_t)t*cols;
        for(int c=0;c<cols;c++){ int v=(int)lrintf(xt[c]*inv); v=v>32767?32767:(v<-32767?-32767:v); xq[c]=(int16_t)v; }
    }
}

#if defined(__ARM_FEATURE_SVE)
static inline void glm5_gemm_int16sdot_block8(float*restrict Y, int yrows, int r,
        const uint8_t*w, const float*s, int sb, int gs, int qg0,
        const int16_t *Xq, const int64_t *Xg, const float *xsc, int N, int cols){
    const uint8_t*w0=w,*w1=w+cols,*w2=w+2*(size_t)cols,*w3=w+3*(size_t)cols,*w4=w+4*(size_t)cols,*w5=w+5*(size_t)cols,*w6=w+6*(size_t)cols,*w7=w+7*(size_t)cols;
    const float*s0=s,*s1=s+sb,*s2=s+2*(size_t)sb,*s3=s+3*(size_t)sb,*s4=s+4*(size_t)sb,*s5=s+5*(size_t)sb,*s6=s+6*(size_t)sb,*s7=s+7*(size_t)sb;
    int t=0;
    for(;t+4<N;t+=5){
        float acc[5][8]; for(int u=0;u<5;u++) for(int j=0;j<8;j++) acc[u][j]=0.f;
        const int16_t*q0=Xq+(size_t)t*cols,*q1=Xq+(size_t)(t+1)*cols,*q2=Xq+(size_t)(t+2)*cols,*q3=Xq+(size_t)(t+3)*cols,*q4=Xq+(size_t)(t+4)*cols;
        const int64_t*g0=Xg+(size_t)t*sb,*g1=Xg+(size_t)(t+1)*sb,*g2=Xg+(size_t)(t+2)*sb,*g3=Xg+(size_t)(t+3)*sb,*g4=Xg+(size_t)(t+4)*sb;
        glm5_int16sdot_4row_5x(acc[0],acc[1],acc[2],acc[3],acc[4], w0,w1,w2,w3, s0,s1,s2,s3, gs,qg0, q0,q1,q2,q3,q4, g0,g1,g2,g3,g4, cols);
        glm5_int16sdot_4row_5x(acc[0]+4,acc[1]+4,acc[2]+4,acc[3]+4,acc[4]+4, w4,w5,w6,w7, s4,s5,s6,s7, gs,qg0, q0,q1,q2,q3,q4, g0,g1,g2,g3,g4, cols);
        for(int u=0;u<5;u++){ float xs=xsc[t+u]; for(int j=0;j<8;j++) Y[(size_t)(t+u)*yrows+r+j]=xs*acc[u][j]; }
    }
    int rem=N-t;
    if(rem>=3){
        float acc[4][8]; for(int u=0;u<4;u++) for(int j=0;j<8;j++) acc[u][j]=0.f;
        const int16_t*q0=Xq+(size_t)t*cols,*q1=Xq+(size_t)(t+1)*cols;
        const int16_t*q2=Xq+(size_t)(t+(rem>2?2:1))*cols;
        const int16_t*q3=Xq+(size_t)(t+(rem>3?3:(rem>2?2:1)))*cols;
        const int64_t*g0=Xg+(size_t)t*sb,*g1=Xg+(size_t)(t+1)*sb;
        const int64_t*g2=Xg+(size_t)(t+(rem>2?2:1))*sb;
        const int64_t*g3=Xg+(size_t)(t+(rem>3?3:(rem>2?2:1)))*sb;
        glm5_int16sdot_4row_4x(acc[0],acc[1],acc[2],acc[3], w0,w1,w2,w3, s0,s1,s2,s3, gs,qg0, q0,q1,q2,q3, g0,g1,g2,g3, cols);
        glm5_int16sdot_4row_4x(acc[0]+4,acc[1]+4,acc[2]+4,acc[3]+4, w4,w5,w6,w7, s4,s5,s6,s7, gs,qg0, q0,q1,q2,q3, g0,g1,g2,g3, cols);
        for(int u=0;u<rem;u++){ float xs=xsc[t+u]; for(int j=0;j<8;j++) Y[(size_t)(t+u)*yrows+r+j]=xs*acc[u][j]; }
        t=N;
    }
    for(;t<N;t++){
        float tmp[8];
        glm5_matvec_int16sdot_8row(tmp, w0,w1,w2,w3,w4,w5,w6,w7, s0,s1,s2,s3,s4,s5,s6,s7, gs, qg0, Xq+(size_t)t*cols, Xg+(size_t)t*sb, cols);
        float xs=xsc[t]; for(int j=0;j<8;j++) Y[(size_t)t*yrows+r+j]=xs*tmp[j];
    }
}
#endif

static int glm5_gemm_int16sdot_preq(float*restrict Y, const uint8_t*W, const float*S, int gs, int qg0,
                                    const int16_t *Xq, const float *xsc, int N, int rows, int cols){
#if defined(__ARM_FEATURE_SVE)
    const int sb=(qg0+cols+gs-1)/gs;
    const int nb=rows/8;
    int64_t *Xg=(int64_t*)malloc((size_t)N*sb*sizeof(int64_t));
    if(!Xg) return 0;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)N>=4)
#endif
    for(int t=0;t<N;t++) glm5_i16_group_sums(Xg+(size_t)t*sb,Xq+(size_t)t*cols,gs,qg0,cols);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb;
        glm5_gemm_int16sdot_block8(Y,rows,r,w,s,sb,gs,qg0,Xq,Xg,xsc,N,cols);
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
    for(int r=nb*8;r<rows;r++)
        for(int t=0;t<N;t++) Y[(size_t)t*rows+r]=xsc[t]*glm5_dot_int16sdot_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,qg0,Xq+(size_t)t*cols,cols);
    free(Xg);
    return 1;
#else
    (void)Y; (void)W; (void)S; (void)gs; (void)qg0; (void)Xq; (void)xsc; (void)N; (void)rows; (void)cols;
    return 0;
#endif
}

static int glm5_gemm2_int16sdot_preq(float*restrict Y0, const uint8_t*W0, const float*S0, int gs0, int qg00,
                                     float*restrict Y1, const uint8_t*W1, const float*S1, int gs1, int qg01,
                                     const int16_t *Xq, const float *xsc, int N, int rows, int cols){
#if defined(__ARM_FEATURE_SVE)
    const int sb0=(qg00+cols+gs0-1)/gs0, sb1=(qg01+cols+gs1-1)/gs1;
    const int nb=rows/8;
    int64_t *Xg0=(int64_t*)malloc((size_t)N*sb0*sizeof(int64_t));
    int64_t *Xg1=(int64_t*)malloc((size_t)N*sb1*sizeof(int64_t));
    if(!Xg0||!Xg1){ free(Xg0); free(Xg1); return 0; }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)N>=4)
#endif
    for(int t=0;t<N;t++){ glm5_i16_group_sums(Xg0+(size_t)t*sb0,Xq+(size_t)t*cols,gs0,qg00,cols);
                          glm5_i16_group_sums(Xg1+(size_t)t*sb1,Xq+(size_t)t*cols,gs1,qg01,cols); }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8;
        glm5_gemm_int16sdot_block8(Y0,rows,r,W0+(size_t)r*cols,S0+(size_t)r*sb0,sb0,gs0,qg00,Xq,Xg0,xsc,N,cols);
        glm5_gemm_int16sdot_block8(Y1,rows,r,W1+(size_t)r*cols,S1+(size_t)r*sb1,sb1,gs1,qg01,Xq,Xg1,xsc,N,cols);
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
    for(int r=nb*8;r<rows;r++){
        for(int t=0;t<N;t++){
            Y0[(size_t)t*rows+r]=xsc[t]*glm5_dot_int16sdot_row(W0+(size_t)r*cols,S0+(size_t)r*sb0,gs0,qg00,Xq+(size_t)t*cols,cols);
            Y1[(size_t)t*rows+r]=xsc[t]*glm5_dot_int16sdot_row(W1+(size_t)r*cols,S1+(size_t)r*sb1,gs1,qg01,Xq+(size_t)t*cols,cols);
        }
    }
    free(Xg0); free(Xg1);
    return 1;
#else
    (void)Y0; (void)W0; (void)S0; (void)gs0; (void)qg00; (void)Y1; (void)W1; (void)S1; (void)gs1; (void)qg01; (void)Xq; (void)xsc; (void)N; (void)rows; (void)cols;
    return 0;
#endif
}

/* INT16 w8a16-MIMIC prefill GEMM: quantize the N activations to int16 ONCE (per-token symmetric, near-
 * lossless), then contract with the register-blocked svdot_s64 4row×5tok kernel (weight widened in-reg
 * once, reused across 5 tokens). ~2x denser than the w8a16 bf16-tile FMA (svdot_s64 32 macs/instr vs
 * f32 svmla 16) AND accurate — the prefill counterpart of glm5_mv_int16_sdot. Applies to all int8
 * tensors. Y is token-major [N,rows] (Y[t*rows+r]). */
static void glm5_gemm_int16sdot(glm5_model*m, float*restrict Y, const uint8_t*W, const float*S, int gs, int qg0,
                                const float*X, int N, int rows, int cols){
    if(N<=1){ glm5_mv_int16_sdot(m,Y,W,S,gs,qg0,X,rows,cols); return; }
    int16_t *Xq=(int16_t*)malloc((size_t)N*cols*2); float *xsc=(float*)malloc((size_t)N*sizeof(float));
    if(!Xq||!xsc){ free(Xq); free(xsc); glm5_gemm_int8(m,Y,W,S,gs,qg0,X,N,rows,cols); return; }
    glm5_quantize_i16_acts(Xq,xsc,X,N,cols);
    if(!glm5_gemm_int16sdot_preq(Y,W,S,gs,qg0,Xq,xsc,N,rows,cols))
        glm5_gemm_int8(m,Y,W,S,gs,qg0,X,N,rows,cols);
    free(Xq); free(xsc);
}
static int glm5_gemm2_int16sdot(float*restrict Y0, const glm5_tensor*t0, float*restrict Y1, const glm5_tensor*t1,
                                const float*X, int N, int rows, int cols){
    if(N<=1 || rows<=0 || cols<=0) return 0;
    int16_t *Xq=(int16_t*)malloc((size_t)N*cols*2); float *xsc=(float*)malloc((size_t)N*sizeof(float));
    if(!Xq||!xsc){ free(Xq); free(xsc); return 0; }
    glm5_quantize_i16_acts(Xq,xsc,X,N,cols);
    int ok=glm5_gemm2_int16sdot_preq(Y0,(const uint8_t*)t0->w,(const float*)t0->scale,t0->qg,t0->qg0,
                                     Y1,(const uint8_t*)t1->w,(const float*)t1->scale,t1->qg,t1->qg0,
                                     Xq,xsc,N,rows,cols);
    free(Xq); free(xsc);
    return ok;
}
/* INT8 w8a8 register-blocked prefill GEMM: quantize N acts to int8 once, then the register-blocked
 * svdot_s32 4row×5tok kernel (weight in-reg reused across 5 tokens). 4x denser than the w8a16 bf16-tile
 * FMA and register-blocked (unlike the shipped glm5_gemm_int8_sdot which re-reads weights per token) —
 * the FASTEST prefill GEMM, but LOSSY (int8 activations). Y token-major [N,rows]. */
static void glm5_gemm_int8sdot_rb(glm5_model*m, float*restrict Y, const uint8_t*W, const float*S, int gs, int qg0,
                                  const float*X, int N, int rows, int cols){
    if(N<=1){ glm5_mv_int8_sdot(m,Y,W,S,gs,qg0,X,rows,cols); return; }
    const int sb=(qg0+cols+gs-1)/gs;
    int8_t *Xq=(int8_t*)malloc((size_t)N*cols); float *xsc=(float*)malloc((size_t)N*sizeof(float));
    if(!Xq||!xsc){ free(Xq); free(xsc); glm5_gemm_int8(m,Y,W,S,gs,qg0,X,N,rows,cols); return; }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(N>=4)
#endif
    for(int t=0;t<N;t++){
        const float*xt=X+(size_t)t*cols; float amax=1e-20f;
        for(int c=0;c<cols;c++){ float a=fabsf(xt[c]); if(a>amax)amax=a; }
        float inv=127.0f/amax; xsc[t]=amax/127.0f; int8_t*xq=Xq+(size_t)t*cols;
        for(int c=0;c<cols;c++){ int v=(int)lrintf(xt[c]*inv); v=v>127?127:(v<-127?-127:v); xq[c]=(int8_t)v; }
    }
#if defined(__ARM_FEATURE_SVE)
    const int nb=rows/8;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)rows>=GLM5_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const uint8_t*w=W+(size_t)r*cols; const float*s=S+(size_t)r*sb;
        const uint8_t*w0=w,*w1=w+cols,*w2=w+2*(size_t)cols,*w3=w+3*(size_t)cols,*w4=w+4*(size_t)cols,*w5=w+5*(size_t)cols,*w6=w+6*(size_t)cols,*w7=w+7*(size_t)cols;
        const float*s0=s,*s1=s+sb,*s2=s+2*(size_t)sb,*s3=s+3*(size_t)sb,*s4=s+4*(size_t)sb,*s5=s+5*(size_t)sb,*s6=s+6*(size_t)sb,*s7=s+7*(size_t)sb;
        int t=0;
        for(;t+4<N;t+=5){
            float acc[5][8]; for(int u=0;u<5;u++) for(int j=0;j<8;j++) acc[u][j]=0.f;
            const int8_t*q0=Xq+(size_t)t*cols,*q1=Xq+(size_t)(t+1)*cols,*q2=Xq+(size_t)(t+2)*cols,*q3=Xq+(size_t)(t+3)*cols,*q4=Xq+(size_t)(t+4)*cols;
            glm5_int8sdot_4row_5x(acc[0],acc[1],acc[2],acc[3],acc[4], w0,w1,w2,w3, s0,s1,s2,s3, gs,qg0, q0,q1,q2,q3,q4, cols);
            glm5_int8sdot_4row_5x(acc[0]+4,acc[1]+4,acc[2]+4,acc[3]+4,acc[4]+4, w4,w5,w6,w7, s4,s5,s6,s7, gs,qg0, q0,q1,q2,q3,q4, cols);
            for(int u=0;u<5;u++){ float xs=xsc[t+u]; for(int j=0;j<8;j++) Y[(size_t)(t+u)*rows+r+j]=xs*acc[u][j]; }
        }
        for(;t<N;t++){
            float tmp[8];
            glm5_matvec_int8_sdot_8row(tmp, w0,w1,w2,w3,w4,w5,w6,w7, s0,s1,s2,s3,s4,s5,s6,s7, gs, qg0, Xq+(size_t)t*cols, cols);
            float xs=xsc[t]; for(int j=0;j<8;j++) Y[(size_t)t*rows+r+j]=xs*tmp[j];
        }
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows-nb*8>=GLM5_PAR_MIN)
#endif
    for(int r=nb*8;r<rows;r++)
        for(int t=0;t<N;t++) Y[(size_t)t*rows+r]=xsc[t]*glm5_dot_int8_sdot_row(W+(size_t)r*cols,S+(size_t)r*sb,gs,qg0,Xq+(size_t)t*cols,cols);
#else
    glm5_gemm_int8(m,Y,W,S,gs,qg0,X,N,rows,cols);
#endif
    free(Xq); free(xsc);
}
static void glm5_gemm_ggml(float*restrict Y,const glm5_tensor*t,const float*X,
                           int N,int rows,int cols){
    if(N<=1){ glm5_mv_ggml(Y,t,X,rows,cols); return; }
    const size_t rb=dequant_row_size((uint32_t)t->type,cols);
    if(!rb){ for(size_t i=0;i<(size_t)N*rows;i++)Y[i]=NAN; return; }
#ifdef _OPENMP
    #pragma omp parallel if((long)rows*cols>=GLM5_PAR_MIN)
    {
        float*tmp=malloc((size_t)cols*4);
        #pragma omp for schedule(static)
        for(int r=0;r<rows;r++){
            const uint8_t*src=(const uint8_t*)t->w+(size_t)r*rb;
            int bad=!tmp||dequant_row((uint32_t)t->type,src,tmp,cols);
            for(int n=0;n<N;n++) Y[(size_t)n*rows+r]=bad?NAN:
                glm5_dot_f32_f32(tmp,X+(size_t)n*cols,cols);
        }
        free(tmp);
    }
#else
    float*tmp=malloc((size_t)cols*4);
    for(int r=0;r<rows;r++){
        const uint8_t*src=(const uint8_t*)t->w+(size_t)r*rb;
        int bad=!tmp||dequant_row((uint32_t)t->type,src,tmp,cols);
        for(int n=0;n<N;n++) Y[(size_t)n*rows+r]=bad?NAN:
            glm5_dot_f32_f32(tmp,X+(size_t)n*cols,cols);
    }
    free(tmp);
#endif
}
/* batched GEMM dispatch by weight type */
static void glm5_gemm(glm5_model*m, float*restrict Y, const glm5_tensor*t, const float*X, int N, int rows, int cols){
    if(t->type==GLM5_MXFP8) glm5_gemm_mxfp8(m,Y,(const uint8_t*)t->w,t->scale,X,N,rows,cols);
    else if(t->type==GLM5_INT8){
        /* GLM5_GEMM_SDOT: 0=w8a16 bf16-tile (shipped), 2=int16 w8a16-mimic svdot_s64 (accurate+dense),
         * 1=int8 w8a8 svdot (fastest, lossy). Default off (=0) keeps the auto w8a8-for-experts logic. */
        static int gsd=-2, mink=0;
        if(gsd==-2){ gsd=glm5_envi("GLM5_GEMM_SDOT",0); mink=glm5_envi("GLM5_GEMM_SDOT_MINK",0); }
        /* int16 SDOT helps ALL int8 GEMMs net (12L e2e prefill 1.30x; a cols-based size gate was tested
         * and REGRESSED, so default MINK=0 = no gate). GLM5_GEMM_SDOT_MINK stays as a tuning override. */
        if(gsd==2 && N>1 && cols>=mink){ glm5_gemm_int16sdot(m,Y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,X,N,rows,cols); return; }
        if(gsd==1 && N>1){ glm5_gemm_int8sdot_rb(m,Y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,X,N,rows,cols); return; }  /* register-blocked int8 w8a8 (fastest, lossy) */
        /* NB: a mixed mode (int8 w8a8 experts + int16 dense) was tested and gave ZERO e2e gain — the
         * experts are HBM-BW-bound in the runner (22 owned experts stream ≫L2 from HBM); int8 and int16
         * read the SAME int8 weight bytes, so int8's higher COMPUTE density (a win only in the L2-resident
         * kernel bench) buys nothing when memory-bound. Not worth the lossy-experts risk. */
        /* w8a8 sdot wins only for per-channel tensors (routed experts: 1.77x); for group-128 it
         * loses to the tuned w8a16 bf16-FMA kernel. AUTO-enable it for the experts only in the
         * large-chunk prefill regime (chunk >= GLM5_INT8_SDOT_MIN, default 1024 tokens) where the
         * throughput win is worth the w8a8 activation rounding; small chunks/decode stay w8a16.
         * GLM5_INT8_SDOT overrides: 0 = never, 1 = always (N>1); unset = auto-by-chunk-size. */
        static int mode=-2, thr=1024;
        if(mode==-2){ const char*e=getenv("GLM5_INT8_SDOT"); mode=(e&&*e)?atoi(e):-1; thr=glm5_envi("GLM5_INT8_SDOT_MIN",1024); }
        int use_sdot = ((t->qg>=cols) && N>1 && (mode==1 || (mode<0 && m->prefill_ntok>=thr)));
        if(use_sdot && N>1) glm5_gemm_int8sdot_rb(m,Y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,X,N,rows,cols);  /* register-blocked (~3x the old glm5_gemm_int8_sdot) */
        else glm5_gemm_int8(m,Y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,X,N,rows,cols);
    }
    else if(glm5_is_ggml_q(t->type)) glm5_gemm_ggml(Y,t,X,N,rows,cols);
    else glm5_gemm_bf16(Y,(const uint16_t*)t->w,X,N,rows,cols);
}
static int glm5_gemm_sdot_override(glm5_model*m, float*restrict Y, const glm5_tensor*t, const float*X, int N, int rows, int cols, int mode){
    if(mode<0 || t->type!=GLM5_INT8 || N<=1) return 0;
    if(mode==1){ glm5_gemm_int8sdot_rb(m,Y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,X,N,rows,cols); return 1; }
    if(mode==2){ glm5_gemm_int16sdot(m,Y,(const uint8_t*)t->w,(const float*)t->scale,t->qg,t->qg0,X,N,rows,cols); return 1; }
    return 0;
}
static int glm5_gemm_pair(glm5_model*m, float*restrict Y0, const glm5_tensor*t0,
                          float*restrict Y1, const glm5_tensor*t1,
                          const float*X, int N, int rows, int cols){
    (void)m;
    if(t0->type==GLM5_INT8 && t1->type==GLM5_INT8 && t0->rows==rows && t1->rows==rows &&
       t0->cols==cols && t1->cols==cols && N>1){
        static int gsd=-2, mink=0;
        if(gsd==-2){ gsd=glm5_envi("GLM5_GEMM_SDOT",0); mink=glm5_envi("GLM5_GEMM_SDOT_MINK",0); }
        if(gsd==2 && cols>=mink)
            return glm5_gemm2_int16sdot(Y0,t0,Y1,t1,X,N,rows,cols);
    }
    return 0;
}
static inline float glm5_tensor_get(const glm5_model*m,const glm5_tensor*t,int r,int c){
    if(t->type==GLM5_MXFP8){
        const uint8_t*w=(const uint8_t*)t->w; const float*sc=(const float*)t->scale; int sb=(t->cols+127)/128;
        float wf; uint32_t u=m->fp8_lut[w[(size_t)r*t->cols+c]]; memcpy(&wf,&u,4);
        return wf*sc[(size_t)r*sb+c/128];
    }
    if(t->type==GLM5_INT8){
        const uint8_t*w=(const uint8_t*)t->w; const float*sc=(const float*)t->scale; int sb=(t->qg0+t->cols+t->qg-1)/t->qg;
        return (float)((int)w[(size_t)r*t->cols+c]-128) * sc[(size_t)r*sb + (t->qg0+c)/t->qg];
    }
    if(glm5_is_ggml_q(t->type)){
        size_t rb=dequant_row_size((uint32_t)t->type,t->cols);
        float*tmp=malloc((size_t)t->cols*4); if(!tmp)return NAN;
        int bad=dequant_row((uint32_t)t->type,(const uint8_t*)t->w+(size_t)r*rb,tmp,t->cols);
        float v=bad?NAN:tmp[c]; free(tmp); return v;
    }
    return glm5_bf2f(((const uint16_t*)t->w)[(size_t)r*t->cols+c]);
}
static glm5_tensor glm5_tensor_rows(const glm5_tensor*t,int r0,int rows){
    glm5_tensor u=*t; u.rows=rows;
    if(t->type==GLM5_MXFP8){
        u.w=(uint8_t*)t->w+(size_t)r0*t->cols;
        u.scale=t->scale+(size_t)r0*((t->cols+127)/128)*4;
    } else if(t->type==GLM5_INT8){
        u.w=(uint8_t*)t->w+(size_t)r0*t->cols;
        u.scale=t->scale+(size_t)r0*((t->qg0+t->cols+t->qg-1)/t->qg)*4;   /* f32 scale, bytes */
    } else if(glm5_is_ggml_q(t->type)){
        u.w=(uint8_t*)t->w+(size_t)r0*dequant_row_size((uint32_t)t->type,t->cols);
    } else u.w=(uint16_t*)t->w+(size_t)r0*t->cols;
    return u;
}
static void glm5_tensor_tmul_rows(const glm5_model*m,float*y,const glm5_tensor*t,int r0,int rows,const float*x){
    const int cols=t->cols;
    for(int k=0;k<cols;k++) y[k]=0.0f;
    if(t->type==GLM5_MXFP8){
        const uint8_t*w=(const uint8_t*)t->w+(size_t)r0*cols;
        const float*sc=(const float*)t->scale+(size_t)r0*((cols+127)/128);
        int sb=(cols+127)/128;
        for(int i=0;i<rows;i++){
            float xi=x[i]; if(xi==0.0f) continue;
            const uint8_t*row=w+(size_t)i*cols;
            const float*srow=sc+(size_t)i*sb;
#if defined(__ARM_FEATURE_SVE)
            int vl=(int)svcntw();
            for(int b=0;b<sb;b++){
                float scale=srow[b]*xi; int k0=b*128, k1=k0+128<cols?k0+128:cols;
                for(int u=k0;u<k1;u+=vl){
                    svbool_t pg=svwhilelt_b32(u,k1);
                    svfloat32_t yv=svld1(pg,y+u);
                    svuint32_t idx=svld1ub_u32(pg,row+u);
                    svfloat32_t wv=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,m->fp8_lut,idx));
                    yv=svmla_n_f32_x(pg,yv,wv,scale);
                    svst1(pg,y+u,yv);
                }
            }
#else
            for(int b=0;b<sb;b++){
                float scale=srow[b]; int k0=b*128, k1=k0+128<cols?k0+128:cols;
                for(int u=k0;u<k1;u++){ float wf; uint32_t bits=m->fp8_lut[row[u]]; memcpy(&wf,&bits,4); y[u]+=xi*wf*scale; }
            }
#endif
        }
    } else if(t->type==GLM5_INT8){
        const uint8_t*w=(const uint8_t*)t->w+(size_t)r0*cols;
        const float*sc=(const float*)t->scale+(size_t)r0*((t->qg0+cols+t->qg-1)/t->qg);
        int sb=(t->qg0+cols+t->qg-1)/t->qg, gs=t->qg, qg0=t->qg0;
        for(int i=0;i<rows;i++){
            float xi=x[i]; if(xi==0.0f) continue;
            const uint8_t*row=w+(size_t)i*cols; const float*srow=sc+(size_t)i*sb;
            for(int b=0;b<sb;b++){
                float scale=srow[b]*xi; int k0=b*gs-qg0; if(k0<0)k0=0;
                int k1=(b+1)*gs-qg0; if(k1>cols)k1=cols;
                if(k0>=k1) continue;
#if defined(__ARM_FEATURE_SVE)
                int vl=(int)svcntw();
                for(int u=k0;u<k1;u+=vl){
                    svbool_t pg=svwhilelt_b32(u,k1);
                    svfloat32_t yv=svld1(pg,y+u);
                    svuint32_t bz=svld1ub_u32(pg,row+u);
                    svint32_t q=svsub_n_s32_x(pg,svreinterpret_s32_u32(bz),128);
                    yv=svmla_n_f32_x(pg,yv,svcvt_f32_s32_x(pg,q),scale);
                    svst1(pg,y+u,yv);
                }
#else
                for(int u=k0;u<k1;u++) y[u]+=(float)((int)row[u]-128)*scale;
#endif
            }
        }
    } else {
        const uint16_t*w=(const uint16_t*)t->w+(size_t)r0*cols;
        for(int i=0;i<rows;i++){
            float xi=x[i]; if(xi==0.0f) continue;
            const uint16_t*row=w+(size_t)i*cols;
#ifdef _OPENMP
            #pragma omp simd
#endif
            for(int k=0;k<cols;k++) y[k]+=xi*glm5_bf2f(row[k]);
        }
    }
}

static void glm5_prefill_absorb_token(glm5_model*m, glm5_layer*L, const float*qb, float*ab,
        const int*sel, int ns, float*hmx, float*hse, float*qabs, float*ctxb,
        float*kvtmp, int nown, int qrows, int arows, int qk_mode){
    (void)qrows; (void)arows;
    const glm5_config*c=&m->cfg;
    const int KVC=glm5_kv_cache_dim(c), kvb_stride=c->qk_nope_dim+c->v_head_dim;
    const float ascale=1.0f/sqrtf((float)c->qk_head_dim);
    int absorb_sve_dot=glm5_envi("GLM5_ABSORB_SVE_DOT",1);
    for(int hh=0;hh<nown;hh++){
        hmx[hh]=-1e30f; hse[hh]=0.0f;
        const float*qh=qb+hh*c->qk_head_dim;
        float*qa=qabs+(size_t)hh*c->kv_lora;
        glm5_tensor_tmul_rows(m,qa,&L->wkv_b,hh*kvb_stride,c->qk_nope_dim,qh);
        float*oh=ab+hh*c->v_head_dim; for(int i=0;i<c->v_head_dim;i++) oh[i]=0.0f;
    }
    for(int hh=0;hh<nown;hh++){ float*ctx=ctxb+(size_t)hh*c->kv_lora; for(int i=0;i<c->kv_lora;i++) ctx[i]=0.0f; }
#if defined(__ARM_FEATURE_SVE)
    if(qk_mode>0 && c->kv_lora==512){
        int j=0;
        for(;j+3<ns;j+=4){
            float kv4[4][576], scr[4][4];
            for(int u=0;u<4;u++) glm5_load_latent_kv(m,L,sel[j+u],kv4[u],KVC);
            int hh=0;
            for(;hh+3<nown;hh+=4){
                const float*q0=qabs+(size_t)(hh+0)*c->kv_lora,*q1=qabs+(size_t)(hh+1)*c->kv_lora;
                const float*q2=qabs+(size_t)(hh+2)*c->kv_lora,*q3=qabs+(size_t)(hh+3)*c->kv_lora;
                if(qk_mode==2) glm5_qk4_int16(scr,q0,q1,q2,q3,kv4[0],kv4[1],kv4[2],kv4[3],c->kv_lora);
                else           glm5_qk4_f32  (scr,q0,q1,q2,q3,kv4[0],kv4[1],kv4[2],kv4[3],c->kv_lora);
                for(int h=0;h<4;h++){
                    const float*qh=qb+(hh+h)*c->qk_head_dim;
                    float*ctx=ctxb+(size_t)(hh+h)*c->kv_lora;
                    for(int u=0;u<4;u++){
                        double d=(double)scr[h][u];
                        for(int i=0;i<c->qk_rope_dim;i++) d+=(double)qh[c->qk_nope_dim+i]*kv4[u][c->kv_lora+i];
                        float s=(float)d*ascale;
                        if(s>hmx[hh+h]){ float r=(hmx[hh+h]>-1e20f)?expf(hmx[hh+h]-s):0.0f; hse[hh+h]*=r; glm5_scale_f32(ctx,r,c->kv_lora); hmx[hh+h]=s; }
                        float e=expf(s-hmx[hh+h]); hse[hh+h]+=e; glm5_axpy_f32(ctx,kv4[u],e,c->kv_lora);
                    }
                }
            }
            for(;hh<nown;hh++){
                const float*qh=qb+hh*c->qk_head_dim, *qa=qabs+(size_t)hh*c->kv_lora;
                float*ctx=ctxb+(size_t)hh*c->kv_lora;
                for(int u=0;u<4;u++){
                    double d=(double)glm5_dot_f32_opt(qa,kv4[u],c->kv_lora,absorb_sve_dot);
                    for(int i=0;i<c->qk_rope_dim;i++) d+=(double)qh[c->qk_nope_dim+i]*kv4[u][c->kv_lora+i];
                    float s=(float)d*ascale;
                    if(s>hmx[hh]){ float r=(hmx[hh]>-1e20f)?expf(hmx[hh]-s):0.0f; hse[hh]*=r; glm5_scale_f32(ctx,r,c->kv_lora); hmx[hh]=s; }
                    float e=expf(s-hmx[hh]); hse[hh]+=e; glm5_axpy_f32(ctx,kv4[u],e,c->kv_lora);
                }
            }
        }
        for(;j<ns;j++){
            float*kv=kvtmp; glm5_load_latent_kv(m,L,sel[j],kv,KVC);
            for(int hh=0;hh<nown;hh++){
                const float*qh=qb+hh*c->qk_head_dim, *qa=qabs+(size_t)hh*c->kv_lora;
                float*ctx=ctxb+(size_t)hh*c->kv_lora;
                double d=(double)glm5_dot_f32_opt(qa,kv,c->kv_lora,absorb_sve_dot);
                for(int i=0;i<c->qk_rope_dim;i++) d+=(double)qh[c->qk_nope_dim+i]*kv[c->kv_lora+i];
                float s=(float)d*ascale;
                if(s>hmx[hh]){ float r=(hmx[hh]>-1e20f)?expf(hmx[hh]-s):0.0f; hse[hh]*=r; glm5_scale_f32(ctx,r,c->kv_lora); hmx[hh]=s; }
                float e=expf(s-hmx[hh]); hse[hh]+=e; glm5_axpy_f32(ctx,kv,e,c->kv_lora);
            }
        }
    } else
#endif
    {
        for(int j=0;j<ns;j++){
            float*kv=kvtmp; glm5_load_latent_kv(m,L,sel[j],kv,KVC);
            for(int hh=0;hh<nown;hh++){
                const float*qh=qb+hh*c->qk_head_dim, *qa=qabs+(size_t)hh*c->kv_lora;
                float*ctx=ctxb+(size_t)hh*c->kv_lora;
                double d=(double)glm5_dot_f32_opt(qa,kv,c->kv_lora,absorb_sve_dot);
                for(int i=0;i<c->qk_rope_dim;i++) d+=(double)qh[c->qk_nope_dim+i]*kv[c->kv_lora+i];
                float s=(float)d*ascale;
                if(s>hmx[hh]){ float r=(hmx[hh]>-1e20f)?expf(hmx[hh]-s):0.0f; hse[hh]*=r; glm5_scale_f32(ctx,r,c->kv_lora); hmx[hh]=s; }
                float e=expf(s-hmx[hh]); hse[hh]+=e; glm5_axpy_f32(ctx,kv,e,c->kv_lora);
            }
        }
    }
    for(int hh=0;hh<nown;hh++){
        glm5_tensor tv=glm5_tensor_rows(&L->wkv_b,hh*kvb_stride+c->qk_nope_dim,c->v_head_dim);
        glm5_mv(m,ab+hh*c->v_head_dim,&tv,ctxb+(size_t)hh*c->kv_lora,c->v_head_dim,c->kv_lora);
    }
}

static void glm5_free_mstream(glm5_model*m){
    glm5_mstream*ms=(glm5_mstream*)m->ms; if(!ms) return;
    glm5_afree(ms->kc);glm5_afree(ms->vc);glm5_afree(ms->xn);glm5_afree(ms->q);glm5_afree(ms->k);glm5_afree(ms->v);glm5_afree(ms->qlat);glm5_afree(ms->kvb);glm5_afree(ms->attn);glm5_afree(ms->o);
    glm5_afree(ms->h2);glm5_afree(ms->router);glm5_afree(ms->route);glm5_afree(ms->shg);glm5_afree(ms->shu);glm5_afree(ms->ffg);glm5_afree(ms->ffu);
    glm5_afree(ms->tmp2);glm5_afree(ms->exg);glm5_afree(ms->exu);glm5_afree(ms->emoe);glm5_afree(ms->bk);glm5_afree(ms->bw);glm5_afree(ms->bcnt);glm5_afree(ms->logits);glm5_afree(ms->sc);
    glm5_afree(ms->psel);glm5_afree(ms->pnsel);glm5_afree(ms->gsel);glm5_afree(ms->gselw);
    glm5_afree(ms->piq);glm5_afree(ms->pik);glm5_afree(ms->pbit);
    glm5_afree(ms->hmx);glm5_afree(ms->hse);glm5_afree(ms->slog);
    glm5_afree(ms); m->ms=NULL;
}
/* per_stream_kv=1: multi-stream decode (own KV per stream). 0: chunked prefill (shared model
 * KV cache; allocate per-token MSA selection buffers psel/pnsel instead). */
static int glm5_alloc_mstream_ex(glm5_model*m,int N,int per_stream_kv){
    const glm5_config*c=&m->cfg; int H=c->hidden,QD=glm5_q_dim(c),KVD=glm5_kv_dim(c),hrows=m->head.rows;
    glm5_mstream*ms=glm5_acalloc(1,sizeof *ms); if(!ms) return -1; ms->n=N;
    size_t per=(size_t)c->n_layers*c->max_pos*KVD;
    if(per_stream_kv){ ms->kc=glm5_acalloc((size_t)N*per,2);   /* latent-MLA: kc only (no vc; halves KV memory) */
           ms->slog=glm5_amalloc((size_t)N*c->vocab*4); }      /* batched-sampling full-logit gather */
    else {
           int maxsel=(c->msa_topk_blocks+c->msa_local_block+c->msa_init_block+1)*c->msa_block_size;
           int attn_window=glm5_envi("GLM5_ATTN_WINDOW",0);
           int dense_window=glm5_envi("GLM5_DENSE_ATTN_WINDOW",attn_window);
           int sparse_window=glm5_envi("GLM5_SPARSE_ATTN_WINDOW",attn_window);
           if(dense_window>maxsel) maxsel=dense_window;
           if(sparse_window>maxsel) maxsel=sparse_window;
           int env_maxsel=glm5_envi("GLM5_PREFILL_MAXSEL",0);
           if(env_maxsel>maxsel) maxsel=env_maxsel;
           if(maxsel<1) maxsel=1;
           if(maxsel>c->max_pos) maxsel=c->max_pos;
           ms->maxsel=maxsel;
           ms->psel=glm5_amalloc((size_t)N*ms->maxsel*sizeof(int)); ms->pnsel=glm5_amalloc((size_t)N*sizeof(int));
           ms->nblkmax=(c->max_pos+c->msa_block_size-1)/c->msa_block_size;
           ms->piq=glm5_amalloc((size_t)N*glm5_idx_q_dim(c)*4); ms->pik=glm5_amalloc((size_t)N*c->msa_index_dim*4);
           ms->pbit=glm5_amalloc((size_t)N*ms->nblkmax); }
    /* router top-k scratch: used by BOTH prefill_chunk and batch_decode_mla */
    ms->gsel=glm5_amalloc((size_t)N*8*sizeof(int)); ms->gselw=glm5_amalloc((size_t)N*8*sizeof(float));
    ms->xn=glm5_amalloc((size_t)N*H*4); ms->q=glm5_amalloc((size_t)N*QD*4); ms->k=glm5_amalloc((size_t)N*KVD*4); ms->v=glm5_amalloc((size_t)N*KVD*4);
    int kvb_scratch = c->qk_nope_dim+c->v_head_dim; if(kvb_scratch<2*c->kv_lora) kvb_scratch=2*c->kv_lora;
    ms->qlat=glm5_amalloc((size_t)N*c->q_lora*4); ms->kvb=glm5_amalloc((size_t)N*c->n_heads*kvb_scratch*4);
    ms->attn=glm5_amalloc((size_t)N*QD*4); ms->o=glm5_amalloc((size_t)N*H*4); ms->h2=glm5_amalloc((size_t)N*H*4);
    ms->hmx=glm5_amalloc((size_t)N*64*4); ms->hse=glm5_amalloc((size_t)N*64*4);
    ms->router=glm5_amalloc((size_t)N*c->n_experts*4); ms->route=glm5_amalloc((size_t)N*H*4);
    ms->shg=glm5_amalloc((size_t)N*c->moe_inter*4); ms->shu=glm5_amalloc((size_t)N*c->moe_inter*4);
    ms->ffg=glm5_amalloc((size_t)N*c->dense_inter*4); ms->ffu=glm5_amalloc((size_t)N*c->dense_inter*4);
    ms->tmp2=glm5_amalloc((size_t)N*H*4);
    ms->exg=glm5_amalloc((size_t)c->moe_inter*4); ms->exu=glm5_amalloc((size_t)c->moe_inter*4); ms->emoe=glm5_amalloc((size_t)H*4);
    ms->bk=glm5_amalloc((size_t)c->n_experts*N*sizeof(int)); ms->bw=glm5_amalloc((size_t)c->n_experts*N*4); ms->bcnt=glm5_amalloc((size_t)c->n_experts*sizeof(int));
    ms->logits=glm5_amalloc((size_t)N*hrows*4);
    ms->sc_stride = per_stream_kv ? c->max_pos : c->n_heads*(ms->maxsel>0?ms->maxsel:c->max_pos);
    ms->sc=glm5_amalloc((size_t)N*ms->sc_stride*4);
    int kvok = per_stream_kv ? (ms->kc&&ms->slog) : (ms->psel&&ms->pnsel);
    if(!kvok||!ms->logits||!ms->qlat||!ms->kvb||!ms->sc||!ms->hmx||!ms->hse){ m->ms=ms; glm5_free_mstream(m); return -1; }
    m->ms=ms; return 0;
}
static int glm5_alloc_mstream(glm5_model*m,int N){ return glm5_alloc_mstream_ex(m,N,1); }

/* one batched decode step: N tokens (X token-major [N,hidden]) at positions pos[t] -> out[t]=argmax. */
static void glm5_forward_batch_decode(glm5_model*m, float*X, int N, const int*pos, int*out){
    const glm5_config*c=&m->cfg; const int H=c->hidden,HD=c->head_dim,QH=c->n_heads,KVH=c->n_kv_heads;
    const int KVD=glm5_kv_dim(c),grp=QH/KVH,half=c->rotary_dim/2; const float ascale=1.0f/sqrtf((float)HD);
    glm5_mstream*ms=(glm5_mstream*)m->ms; size_t per=(size_t)c->n_layers*c->max_pos*KVD;
    for(int l=0;l<c->n_layers;l++){
        glm5_layer*L=&m->layers[l]; int is_moe=glm5_is_moe(c,l);
        const int qh0=L->qh0,qh1=L->qh1,nown=qh1-qh0,qrows=nown*HD; const int tp_attn=(qrows<QH*HD);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<N;t++) glm5_rmsnorm_gemma(ms->xn+(size_t)t*H, X+(size_t)t*H, L->input_norm, H, c->norm_eps);
        glm5_gemm(m,ms->q,&L->wq,ms->xn,N,qrows,H);
        glm5_gemm(m,ms->k,&L->wk,ms->xn,N,KVD,H);
        glm5_gemm(m,ms->v,&L->wv,ms->xn,N,KVD,H);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<N;t++){ int p=pos[t]; const float*cosp=&m->rope_cos[(size_t)p*half],*sinp=&m->rope_sin[(size_t)p*half];
            float*qb=ms->q+(size_t)t*qrows,*kb=ms->k+(size_t)t*KVD,*vb=ms->v+(size_t)t*KVD;
            for(int hh=0;hh<nown;hh++){ float*qh=qb+hh*HD; if(c->use_qk_norm) glm5_rmsnorm_head(qh,L->q_norm,HD,c->norm_eps); glm5_rope_head(qh,cosp,sinp,c->rotary_dim); }
            for(int kh=0;kh<KVH;kh++){ float*kk=kb+kh*HD; if(c->use_qk_norm) glm5_rmsnorm_head(kk,L->k_norm,HD,c->norm_eps); glm5_rope_head(kk,cosp,sinp,c->rotary_dim); }
            uint16_t*kc=ms->kc+(size_t)t*per+(size_t)l*c->max_pos*KVD+(size_t)p*KVD;
            uint16_t*vc=ms->vc+(size_t)t*per+(size_t)l*c->max_pos*KVD+(size_t)p*KVD;
            for(int i=0;i<KVD;i++){ kc[i]=glm5_f2bf(kb[i]); vc[i]=glm5_f2bf(vb[i]); } }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<N;t++){ int p=pos[t]; float*qb=ms->q+(size_t)t*qrows,*ab=ms->attn+(size_t)t*qrows,*sc=ms->sc+(size_t)t*c->max_pos;
            uint16_t*kcl=ms->kc+(size_t)t*per+(size_t)l*c->max_pos*KVD,*vcl=ms->vc+(size_t)t*per+(size_t)l*c->max_pos*KVD;
            for(int hh=0;hh<nown;hh++){ int hgl=qh0+hh; float*qh=qb+hh*HD; int kvh=hgl/grp; float mx=-1e30f;
                for(int tt=0;tt<=p;tt++){ const uint16_t*kt=kcl+(size_t)tt*KVD+kvh*HD; double d=0; for(int i=0;i<HD;i++) d+=(double)qh[i]*glm5_bf2f(kt[i]); float s=(float)d*ascale; sc[tt]=s; if(s>mx)mx=s; }
                double sum=0; for(int tt=0;tt<=p;tt++){ float e=expf(sc[tt]-mx); sc[tt]=e; sum+=e; }
                float inv=(float)(1.0/(sum>0?sum:1)); float*oh=ab+hh*HD; for(int i=0;i<HD;i++) oh[i]=0;
                for(int tt=0;tt<=p;tt++){ float w=sc[tt]*inv; const uint16_t*vt=vcl+(size_t)tt*KVD+kvh*HD; for(int i=0;i<HD;i++) oh[i]+=w*glm5_bf2f(vt[i]); } } }
        glm5_gemm(m,ms->o,&L->wo,ms->attn,N,H,qrows);
        if(tp_attn && m->ar_cb) m->ar_cb(ms->o,N*H,m->ar_ctx);
        for(size_t i=0;i<(size_t)N*H;i++) X[i]+=ms->o[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<N;t++) glm5_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, L->post_norm, H, c->norm_eps);
        if(is_moe){
            const int tp_sh=(L->sh_rows<c->moe_inter);
            for(size_t i=0;i<(size_t)N*H;i++) ms->route[i]=0;
            int na=c->n_active>8?8:c->n_active; int sel_all[64*8]; float selw_all[64*8];
            /* router + select: batched F32 gate GEMM, then parallel sigmoid/top-k. */
            glm5_gemm_f32(ms->router,(float*)L->gate.w,ms->h2,N,c->n_experts,H);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int t=0;t<N;t++){ float*rl=ms->router+(size_t)t*c->n_experts;
                for(int e=0;e<c->n_experts;e++) rl[e]=1.0f/(1.0f+expf(-rl[e]));
                int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int best=-1; float bv=-1e30f; for(int e=0;e<c->n_experts;e++){ int used=0; for(int j=0;j<a;j++) if(sel[j]==e){used=1;break;} if(used)continue; float vv=rl[e]+L->gate_bias[e]; if(vv>bv){bv=vv;best=e;} } sel[a]=best; sw[a]=rl[best]; }
                float wsum=0; for(int a=0;a<na;a++) wsum+=sw[a]; if(wsum<=0)wsum=1; for(int a=0;a<na;a++) sw[a]=sw[a]/wsum*c->routed_scale; }
            /* EXPERT GROUPING: bucket tokens by owned expert, run ONE M=g GEMM per owned
             * expert (weight read once for the whole group) instead of M=1 per (token,expert). */
            for(int s=0;s<L->n_owned;s++) ms->bcnt[s]=0;
            for(int t=0;t<N;t++){ int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int e=sel[a]; if(e%m->ep_size!=m->ep_rank) continue; int slot=e/m->ep_size;
                    int g=ms->bcnt[slot]++; ms->bk[(size_t)slot*N+g]=t; ms->bw[(size_t)slot*N+g]=sw[a]; } }
            for(int s=0;s<L->n_owned;s++){ int g=ms->bcnt[s]; if(g==0) continue;
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*N+i]; memcpy(ms->o+(size_t)i*H, ms->h2+(size_t)t*H, (size_t)H*4); }
                if(!glm5_gemm_pair(m,ms->shg,&L->ex_w1[s],ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H)){
                    glm5_gemm(m,ms->shg,&L->ex_w1[s],ms->o,g,c->moe_inter,H);
                    glm5_gemm(m,ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H);
                }
                for(size_t i=0;i<(size_t)g*c->moe_inter;i++) ms->shg[i]=glm5_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
                glm5_gemm(m,ms->tmp2,&L->ex_w2[s],ms->shg,g,H,c->moe_inter);
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*N+i]; float w=ms->bw[(size_t)s*N+i]; float*rt=ms->route+(size_t)t*H,*dn=ms->tmp2+(size_t)i*H;
                    for(int j=0;j<H;j++) rt[j]+=w*dn[j]; } }
            /* comm-overlap: issue the routed reduce on the comm-driver thread, compute the
             * (replicated) shared expert during it, then wait. Needs shared replicated (!tp_sh)
             * so it is independent of the route buffer being reduced. */
            int overlap = (m->ar_async_start && !tp_sh);
            if(overlap) m->ar_async_start(ms->route,N*H,m->ar_async_ctx);
            if(!glm5_gemm_pair(m,ms->shg,&L->sh_w1,ms->shu,&L->sh_w3,ms->h2,N,L->sh_rows,H)){
                glm5_gemm(m,ms->shg,&L->sh_w1,ms->h2,N,L->sh_rows,H);
                glm5_gemm(m,ms->shu,&L->sh_w3,ms->h2,N,L->sh_rows,H);
            }
            for(size_t i=0;i<(size_t)N*L->sh_rows;i++) ms->shg[i]=glm5_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_gemm(m,ms->tmp2,&L->sh_w2,ms->shg,N,H,L->sh_rows);   /* shared-out [N,H] */
            if(overlap){ m->ar_wait(m->ar_async_ctx); }
            else { if(tp_sh) for(size_t i=0;i<(size_t)N*H;i++) ms->route[i]+=ms->tmp2[i];
                   if(m->ar_cb) m->ar_cb(ms->route,N*H,m->ar_ctx); }
            for(size_t i=0;i<(size_t)N*H;i++) X[i]+=ms->route[i] + (tp_sh?0.0f:ms->tmp2[i]);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            if(!glm5_gemm_pair(m,ms->ffg,&L->ff_gate,ms->ffu,&L->ff_up,ms->h2,N,L->ff_rows,H)){
                glm5_gemm(m,ms->ffg,&L->ff_gate,ms->h2,N,L->ff_rows,H);
                glm5_gemm(m,ms->ffu,&L->ff_up,ms->h2,N,L->ff_rows,H);
            }
            for(size_t i=0;i<(size_t)N*L->ff_rows;i++) ms->ffg[i]=glm5_swiglu_oai(ms->ffg[i],ms->ffu[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_gemm(m,ms->tmp2,&L->ff_down,ms->ffg,N,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(ms->tmp2,N*H,m->ar_ctx);
            for(size_t i=0;i<(size_t)N*H;i++) X[i]+=ms->tmp2[i];
        }
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<N;t++) glm5_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, m->out_norm, H, c->norm_eps);
    int hrows=m->head.rows; glm5_gemm(m,ms->logits,&m->head,ms->h2,N,hrows,H);
    for(int t=0;t<N;t++){ float*lg=ms->logits+(size_t)t*hrows; int la=0; float bv=lg[0]; for(int i=1;i<hrows;i++) if(lg[i]>bv){bv=lg[i];la=i;}
        int32_t gidx=m->head_r0+la; float gval=bv;
        if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);
        out[t]=gidx; }
}

/* Batched-parallel MSA block selection for a prefill chunk (non-CP). Replaces the per-token
 * sequential glm5_msa_select: (A) batch idx_q/idx_k projections M=S; (B) parallel norm+RoPE +
 * store idx_k for all S positions; (C) parallel per-token block scoring (the O(pos) hot loop)
 * + top-k. Mathematically identical to the sequential path -- the block scores read the same
 * fully-stored idx_k cache; only the work is parallelized over the chunk + amortized GEMMs. */
static void glm5_msa_prefill_select(glm5_model*m, glm5_layer*L, int p0, int S, int msa_on){
    const glm5_config*c=&m->cfg; glm5_mstream*ms=(glm5_mstream*)m->ms;
    const int ID=c->msa_index_dim, IH=c->msa_n_index_heads, B=c->msa_block_size, half=c->rotary_dim/2, IQD=IH*ID;
    const int keep=c->msa_topk_blocks+c->msa_local_block+c->msa_init_block;
    if(!msa_on){
        for(int t=0;t<S;t++){
            int p=p0+t, n=0;
            int*sel=ms->psel+(size_t)t*ms->maxsel;
            for(int tt=0;tt<=p;tt++) sel[n++]=tt;
            ms->pnsel[t]=n;
        }
        return;
    }
    glm5_gemm(m,ms->piq,&L->idx_wq,ms->xn,S,IQD,c->hidden);     /* A: batched index projections */
    glm5_gemm(m,ms->pik,&L->idx_wk,ms->xn,S,ID,c->hidden);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<S;t++){ int p=p0+t; const float*cosp=&m->rope_cos[(size_t)p*half],*sinp=&m->rope_sin[(size_t)p*half];
        float*iq=ms->piq+(size_t)t*IQD,*ik=ms->pik+(size_t)t*ID;
        for(int h=0;h<IH;h++){ float*qh=iq+h*ID; glm5_rmsnorm_head(qh,L->idx_q_norm,ID,c->norm_eps); glm5_rope_head(qh,cosp,sinp,c->rotary_dim); }
        glm5_rmsnorm_head(ik,L->idx_k_norm,ID,c->norm_eps); glm5_rope_head(ik,cosp,sinp,c->rotary_dim);
        long sl=glm5_cp_slot(m,p);   /* B: store idx_k (non-CP -> sl==p, every rank stores) */
        if(m->int4_kv) L->idx_qs[sl]=glm5_q4_pack(L->idx_q4+sl*(size_t)(ID/2),ik,ID);
        else for(int i=0;i<ID;i++) L->idx_k_cache[sl*(size_t)ID+i]=glm5_kv_enc(m,ik[i]); }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<S;t++){ int p=p0+t, nblk=p/B+1;     /* C: parallel score + select per token */
        const float*iq=ms->piq+(size_t)t*IQD; int*sel=ms->psel+(size_t)t*ms->maxsel;
        if(!msa_on || nblk<=keep){ int n=0;
            for(int b=0;b<nblk;b++){ int t0=b*B,t1=t0+B; if(t1>p+1)t1=p+1; for(int tt=t0;tt<t1;tt++) sel[n++]=tt; }
            ms->pnsel[t]=n; continue; }
        float*bs=ms->sc+(size_t)t*c->max_pos; char*selb=ms->pbit+(size_t)t*ms->nblkmax;
        for(int b=0;b<nblk;b++){ int t0=b*B,t1=t0+B; if(t1>p+1)t1=p+1; float best=-1e30f;
            for(int tt=t0;tt<t1;tt++){ float scr=0; for(int h=0;h<IH;h++) scr+=glm5_idxdot(m,L,tt,iq+h*ID,ID); if(scr>best)best=scr; }
            bs[b]=best; }
        for(int b=0;b<nblk;b++) selb[b]=0;
        for(int b=0;b<c->msa_init_block && b<nblk;b++) selb[b]=1;
        for(int b=nblk-c->msa_local_block;b<nblk;b++) if(b>=0) selb[b]=1;
        for(int pick=0;pick<c->msa_topk_blocks;pick++){ int best=-1; float bv=-1e30f;
            for(int b=0;b<nblk;b++){ if(selb[b])continue; if(bs[b]>bv){bv=bs[b];best=b;} } if(best<0)break; selb[best]=1; }
        int n=0; for(int b=0;b<nblk;b++){ if(!selb[b])continue; int t0=b*B,t1=t0+B; if(t1>p+1)t1=p+1; for(int tt=t0;tt<t1;tt++) sel[n++]=tt; }
        ms->pnsel[t]=n; }
}

/* ===================== chunked batched prefill (Lever 1, >=100 tok/s target) =====================
 * S consecutive tokens of ONE sequence at positions [p0,p0+S), X token-major [S,hidden]. Writes
 * the model's SHARED KV cache (bf16/fp16/int4, CP-aware) and does causal attention (token i sees
 * [0,p0+i]; MSA top-k per token on sparse layers). The projection/FFN/expert GEMMs run M=S with
 * ONE all-reduce per layer per chunk -> comm + weight-read + FP8-decode all amortized S-fold.
 * Returns the LAST token's argmax (next-token prediction); other tokens' logits aren't needed.
 * Prefill is compute-bound -> ideally run WITHOUT CP (replicated/TP KV); CP works but adds a
 * per-token collective in the MSA select. Requires glm5_alloc_mstream_ex(m,S,0). */
static int glm5_forward_prefill_chunk(glm5_model*m, float*X, int S, int p0, int need_head){
    m->prefill_ntok=S;            /* expose chunk size to the GEMM dispatch (expert sdot gate) */
    const glm5_config*c=&m->cfg;
    const int H=c->hidden, AD=glm5_attn_dim(c);
    const int KVC=glm5_kv_cache_dim(c), half=c->qk_rope_dim/2;
    const float ascale=1.0f/sqrtf((float)c->qk_head_dim);
    glm5_mstream*ms=(glm5_mstream*)m->ms;
    const int msa_on=m->msa_on;
    const int attn_window=glm5_envi("GLM5_ATTN_WINDOW",0);
    const int dense_window=glm5_envi("GLM5_DENSE_ATTN_WINDOW",attn_window);
    const int sparse_window=glm5_envi("GLM5_SPARSE_ATTN_WINDOW",attn_window);
    static int qkv_gsd=-2, oproj_gsd=-2;
    if(qkv_gsd==-2) qkv_gsd=glm5_envi("GLM5_QKV_GEMM_SDOT",-1);
    if(oproj_gsd==-2) oproj_gsd=glm5_envi("GLM5_OPROJ_GEMM_SDOT",-1);
    for(int l=0;l<c->n_layers;l++){
        glm5_layer*L=&m->layers[l]; int is_moe=glm5_is_moe(c,l);
        const int qh0=L->qh0,qh1=L->qh1,nown=qh1-qh0,qrows=nown*c->qk_head_dim, arows=nown*c->v_head_dim;
        const int tp_attn=(arows<AD), kvb_stride=c->qk_nope_dim+c->v_head_dim;
        double pt=glm5_prof_now();
        GLM5_FAPP_START("glm5_qkv");
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++) glm5_rmsnorm_gemma(ms->xn+(size_t)t*H, X+(size_t)t*H, L->input_norm, H, c->norm_eps);
        if(!glm5_gemm_sdot_override(m,ms->qlat,&L->wq_a,ms->xn,S,c->q_lora,H,qkv_gsd))
            glm5_gemm(m,ms->qlat,&L->wq_a,ms->xn,S,c->q_lora,H);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++) glm5_rmsnorm_head(ms->qlat+(size_t)t*c->q_lora,L->q_a_norm,c->q_lora,c->norm_eps);
        if(!glm5_gemm_sdot_override(m,ms->q,&L->wq_b,ms->qlat,S,qrows,c->q_lora,qkv_gsd))
            glm5_gemm(m,ms->q,&L->wq_b,ms->qlat,S,qrows,c->q_lora);
        if(!glm5_gemm_sdot_override(m,ms->k,&L->wkv_a,ms->xn,S,KVC,H,qkv_gsd))
            glm5_gemm(m,ms->k,&L->wkv_a,ms->xn,S,KVC,H);
        /* RoPE and latent KV store. GLM5.2 stores [kv_lora, qk_rope] per position. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++){
            int p=p0+t; const float*cosp=&m->rope_cos[(size_t)p*half],*sinp=&m->rope_sin[(size_t)p*half];
            float*qb=ms->q+(size_t)t*qrows, *kv=ms->k+(size_t)t*KVC;
            for(int hh=0;hh<nown;hh++) glm5_rope_interleaved(qb+hh*c->qk_head_dim+c->qk_nope_dim,cosp,sinp,c->qk_rope_dim);
            glm5_rmsnorm_head(kv,L->kv_a_norm,c->kv_lora,c->norm_eps);
            glm5_rope_interleaved(kv+c->kv_lora,cosp,sinp,c->qk_rope_dim);
            if(glm5_cp_mine(m,p)) glm5_store_latent_kv(m,L,p,kv,KVC);
        }
        GLM5_FAPP_STOP("glm5_qkv");
        glm5_prof_add(m,GLM5_P_QKV,pt);
        pt=glm5_prof_now();
        GLM5_FAPP_START("glm5_msa_index");
        /* Selection lists: dense layers use an optional local window; sparse layers use MSA
         * when enabled, otherwise the sparse window. This keeps long-context prefill bounded. */
        for(int t=0;t<S;t++){
            int p=p0+t, n=0, *sel=ms->psel+(size_t)t*ms->maxsel;
            if(is_moe && msa_on && glm5_has_full_indexer(c,l)){
                ms->pnsel[t]=glm5_msa_select(m,L,ms->xn+(size_t)t*H,ms->qlat+(size_t)t*c->q_lora,p,msa_on,sel);
                continue;
            }
            int win=is_moe ? sparse_window : dense_window;
            int t0=(win>0 && p+1>win) ? p+1-win : 0;
            int maxsel=ms->maxsel;
            if(win<=0 && p+1>maxsel) t0=p+1-maxsel;
            for(int tt=t0;tt<=p;tt++) if(glm5_cp_mine(m,tt) && n<maxsel) sel[n++]=tt;
            ms->pnsel[t]=n;
        }
        GLM5_FAPP_STOP("glm5_msa_index");
        glm5_prof_add(m,GLM5_P_MSA_INDEX,pt);
        pt=glm5_prof_now();
        GLM5_FAPP_START("glm5_attn");
        /* Absorbed MLA attention. Scores use (W_nope^T q_nope) dot kv_lora plus
         * q_rope dot k_rope. Values accumulate a weighted latent context and apply
         * the per-head value rows of wkv_b once, instead of expanding wkv_b per key. */
        static int qk_mode=-1; if(qk_mode<0){ qk_mode=glm5_envi("GLM5_ATTN_QK",0); if(qk_mode<0||qk_mode>2) qk_mode=0; }
/* The per-token local flash-attention math below touches no uTofu, so it is fully parallel
 * even under CP. Per-token softmax stats (hmx/hse) and the unnormalized output go to per-token
 * scratch; the CP combine (which DOES call uTofu) is deferred to an ordered serial loop after
 * this region so every rank issues collectives in identical token order. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++){ float*qb=ms->q+(size_t)t*qrows,*ab=ms->attn+(size_t)t*arows;
            const int*sel=ms->psel+(size_t)t*ms->maxsel; int ns=ms->pnsel[t];
            float*hmx=ms->hmx+(size_t)t*64,*hse=ms->hse+(size_t)t*64;
            float*qabs=ms->kvb+(size_t)t*c->n_heads*(2*c->kv_lora);
            float*ctxb=qabs+(size_t)c->n_heads*c->kv_lora;
            glm5_prefill_absorb_token(m,L,qb,ab,sel,ns,hmx,hse,qabs,ctxb,ms->v+(size_t)t*KVC,nown,qrows,arows,qk_mode);
        }
        /* deferred combine/normalize. Under CP this calls uTofu: the batched cb merges the whole
         * chunk in 2 collectives (bit-identical to the per-token loop, which is kept as fallback);
         * both are deterministic (same buffer layout/order on every rank). Non-CP just normalizes. */
        if(m->cp_on && m->kv_combine_batch_cb){
            m->kv_combine_batch_cb(ms->attn,ms->hmx,ms->hse,S,nown,c->v_head_dim,arows,64,m->kv_combine_batch_ctx);
        } else if(m->cp_on && m->kv_combine_cb){
            for(int t=0;t<S;t++){
                float*ab=ms->attn+(size_t)t*arows;
                m->kv_combine_cb(ab,ms->hmx+(size_t)t*64,ms->hse+(size_t)t*64,nown,c->v_head_dim,m->kv_combine_ctx);
            }
        } else {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int t=0;t<S;t++){
                float*ab=ms->attn+(size_t)t*arows; const float*hse=ms->hse+(size_t)t*64;
                for(int hh=0;hh<nown;hh++){ float inv=1.0f/(hse[hh]>0?hse[hh]:1); float*oh=ab+hh*c->v_head_dim; for(int i=0;i<c->v_head_dim;i++) oh[i]*=inv; }
            }
        }
        GLM5_FAPP_STOP("glm5_attn");
        glm5_prof_add(m,GLM5_P_ATTN,pt);
        pt=glm5_prof_now();
        GLM5_FAPP_START("glm5_o_proj");
        GLM5_FAPP_START("glm5_o_proj_gemm");
        if(!glm5_gemm_sdot_override(m,ms->o,&L->wo,ms->attn,S,H,arows,oproj_gsd))
            glm5_gemm(m,ms->o,&L->wo,ms->attn,S,H,arows);
        GLM5_FAPP_STOP("glm5_o_proj_gemm");
        if(tp_attn && m->ar_cb){ GLM5_FAPP_START("glm5_o_proj_ar"); m->ar_cb(ms->o,S*H,m->ar_ctx); GLM5_FAPP_STOP("glm5_o_proj_ar"); }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
        for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->o[i];
        GLM5_FAPP_STOP("glm5_o_proj");
        glm5_prof_add(m,GLM5_P_OPROJ,pt);
        /* post-norm + MoE/FFN (M=S; identical to batch_decode) */
        pt=glm5_prof_now();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++) glm5_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, L->post_norm, H, c->norm_eps);
        glm5_prof_add(m,GLM5_P_OTHER,pt);
        if(is_moe){
            const int tp_sh=(L->sh_rows<c->moe_inter);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*H;i++) ms->route[i]=0;
            int na=c->n_active>8?8:c->n_active; int*sel_all=ms->gsel; float*selw_all=ms->gselw;  /* [S*8] heap (S may exceed 64) */
            pt=glm5_prof_now();
            GLM5_FAPP_START("glm5_router");
            /* gate is bf16 (GLM5_BF16); use the type-dispatched GEMM like the single-token path
             * (glm5_mv at the decode router). The old glm5_gemm_f32((float*)gate.w) read the bf16
             * buffer as f32 -> 2x over-read past the tensor + wrong logits. */
            double t_gate=glm5_prof_now();
            glm5_gemm(m,ms->router,&L->gate,ms->h2,S,c->n_experts,H);
            { static int gdb=0; if(getenv("GLM5_ROUTER_DBG") && m->ep_rank==0 && gdb<2){ fprintf(stderr,"ROUTER_DBG gate_gemm=%.4f s (S=%d rows=%d cols=%d) then loop...\n",glm5_prof_now()-t_gate,S,c->n_experts,H); gdb++; } }
            double t_loop=glm5_prof_now();
            #pragma omp parallel for schedule(static) if((long)S*c->n_experts>=GLM5_PAR_MIN)
            for(int t=0;t<S;t++){ float*rl=ms->router+(size_t)t*c->n_experts;
                /* per-token FEXPA sigmoid + greedy top-k + normalize; independent across tokens,
                 * so this loop is parallelized (was single-threaded -> ~25% of full-model prefill). */
                glm5_sigmoid_row(rl,c->n_experts);
                int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int best=-1; float bv=-1e30f; for(int e=0;e<c->n_experts;e++){ int used=0; for(int j=0;j<a;j++) if(sel[j]==e){used=1;break;} if(used)continue; float vv=rl[e]+L->gate_bias[e]; if(vv>bv){bv=vv;best=e;} } sel[a]=best; sw[a]=rl[best]; }
                float wsum=0; for(int a=0;a<na;a++) wsum+=sw[a]; if(wsum<=0)wsum=1; for(int a=0;a<na;a++) sw[a]=sw[a]/wsum*c->routed_scale; }
            { static int ldb=0; if(getenv("GLM5_ROUTER_DBG") && m->ep_rank==0 && ldb<2){ fprintf(stderr,"ROUTER_DBG sigmoid+topk loop=%.4f s\n",glm5_prof_now()-t_loop); ldb++; } }
            GLM5_FAPP_STOP("glm5_router");
            glm5_prof_add(m,GLM5_P_ROUTER,pt);
            pt=glm5_prof_now();
            GLM5_FAPP_START("glm5_experts");
            for(int s=0;s<L->n_owned;s++) ms->bcnt[s]=0;
            for(int t=0;t<S;t++){ int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int e=sel[a]; if(e%m->ep_size!=m->ep_rank) continue; int slot=e/m->ep_size;
                    int g=ms->bcnt[slot]++; ms->bk[(size_t)slot*S+g]=t; ms->bw[(size_t)slot*S+g]=sw[a]; } }
            for(int s=0;s<L->n_owned;s++){ int g=ms->bcnt[s]; if(g==0) continue;
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*S+i]; memcpy(ms->o+(size_t)i*H, ms->h2+(size_t)t*H, (size_t)H*4); }
                if(!glm5_gemm_pair(m,ms->shg,&L->ex_w1[s],ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H)){
                    glm5_gemm(m,ms->shg,&L->ex_w1[s],ms->o,g,c->moe_inter,H);
                    glm5_gemm(m,ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H);
                }
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) if((long)g*c->moe_inter>=GLM5_PAR_MIN)
#endif
                for(size_t i=0;i<(size_t)g*c->moe_inter;i++) ms->shg[i]=glm5_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
                glm5_gemm(m,ms->tmp2,&L->ex_w2[s],ms->shg,g,H,c->moe_inter);
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*S+i]; float w=ms->bw[(size_t)s*S+i];
                    glm5_axpy_f32(ms->route+(size_t)t*H, ms->tmp2+(size_t)i*H, w, H); } }   /* SVE: was scalar rt[j]+=w*dn[j] */
            GLM5_FAPP_STOP("glm5_experts");
            glm5_prof_add(m,GLM5_P_EXPERTS,pt);
            pt=glm5_prof_now();
            GLM5_FAPP_START("glm5_shared");
            int overlap = (m->ar_async_start && !tp_sh);
            if(overlap){ GLM5_FAPP_START("glm5_shared_ar"); m->ar_async_start(ms->route,S*H,m->ar_async_ctx); GLM5_FAPP_STOP("glm5_shared_ar"); }
            GLM5_FAPP_START("glm5_shared_gemm");
            static int shared_gsd=-2;
            if(shared_gsd==-2) shared_gsd=glm5_envi("GLM5_SHARED_GEMM_SDOT",-1);
            int shared_i8 = (shared_gsd==1 && S>1 &&
                             L->sh_w1.type==GLM5_INT8 && L->sh_w3.type==GLM5_INT8 && L->sh_w2.type==GLM5_INT8);
            if(shared_i8){
                glm5_gemm_int8sdot_rb(m,ms->shg,(const uint8_t*)L->sh_w1.w,(const float*)L->sh_w1.scale,L->sh_w1.qg,L->sh_w1.qg0,ms->h2,S,L->sh_rows,H);
                glm5_gemm_int8sdot_rb(m,ms->shu,(const uint8_t*)L->sh_w3.w,(const float*)L->sh_w3.scale,L->sh_w3.qg,L->sh_w3.qg0,ms->h2,S,L->sh_rows,H);
            } else if(!glm5_gemm_pair(m,ms->shg,&L->sh_w1,ms->shu,&L->sh_w3,ms->h2,S,L->sh_rows,H)){
                glm5_gemm(m,ms->shg,&L->sh_w1,ms->h2,S,L->sh_rows,H);
                glm5_gemm(m,ms->shu,&L->sh_w3,ms->h2,S,L->sh_rows,H);
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*L->sh_rows>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*L->sh_rows;i++) ms->shg[i]=glm5_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
            if(shared_i8)
                glm5_gemm_int8sdot_rb(m,ms->tmp2,(const uint8_t*)L->sh_w2.w,(const float*)L->sh_w2.scale,L->sh_w2.qg,L->sh_w2.qg0,ms->shg,S,H,L->sh_rows);
            else
                glm5_gemm(m,ms->tmp2,&L->sh_w2,ms->shg,S,H,L->sh_rows);
            GLM5_FAPP_STOP("glm5_shared_gemm");
            if(overlap){ GLM5_FAPP_START("glm5_shared_ar"); m->ar_wait(m->ar_async_ctx); GLM5_FAPP_STOP("glm5_shared_ar"); }
            else { if(tp_sh) for(size_t i=0;i<(size_t)S*H;i++) ms->route[i]+=ms->tmp2[i];
                   if(m->ar_cb){ GLM5_FAPP_START("glm5_shared_ar"); m->ar_cb(ms->route,S*H,m->ar_ctx); GLM5_FAPP_STOP("glm5_shared_ar"); } }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->route[i] + (tp_sh?0.0f:ms->tmp2[i]);
            GLM5_FAPP_STOP("glm5_shared");
            glm5_prof_add(m,GLM5_P_SHARED,pt);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            pt=glm5_prof_now();
            GLM5_FAPP_START("glm5_dense_ffn");
            GLM5_FAPP_START("glm5_dense_gemm");
            if(!glm5_gemm_pair(m,ms->ffg,&L->ff_gate,ms->ffu,&L->ff_up,ms->h2,S,L->ff_rows,H)){
                glm5_gemm(m,ms->ffg,&L->ff_gate,ms->h2,S,L->ff_rows,H);
                glm5_gemm(m,ms->ffu,&L->ff_up,ms->h2,S,L->ff_rows,H);
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*L->ff_rows>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*L->ff_rows;i++) ms->ffg[i]=glm5_swiglu_oai(ms->ffg[i],ms->ffu[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_gemm(m,ms->tmp2,&L->ff_down,ms->ffg,S,H,L->ff_rows);
            GLM5_FAPP_STOP("glm5_dense_gemm");
            if(tp_ffn && m->ar_cb){ GLM5_FAPP_START("glm5_dense_ar"); m->ar_cb(ms->tmp2,S*H,m->ar_ctx); GLM5_FAPP_STOP("glm5_dense_ar"); }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->tmp2[i];
            GLM5_FAPP_STOP("glm5_dense_ffn");
            glm5_prof_add(m,GLM5_P_DENSE_FFN,pt);
        }
    }
    /* head: LAST token only (next-token prediction) */
    if(!need_head) return -1;
    double pt=glm5_prof_now();
    int hrows=m->head.rows; glm5_rmsnorm_gemma(ms->h2, X+(size_t)(S-1)*H, m->out_norm, H, c->norm_eps);
    glm5_gemm(m,ms->logits,&m->head,ms->h2,1,hrows,H);
    float*lg=ms->logits; int la=0; float bv=lg[0]; for(int i=1;i<hrows;i++) if(lg[i]>bv){bv=lg[i];la=i;}
    int32_t gidx=m->head_r0+la; float gval=bv;
    if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);
    glm5_prof_add(m,GLM5_P_HEAD,pt);
    return gidx;
}

/* ============ query-sequence-parallel prefill (GLM5_PREFILL_SP) ============
 * Rank r owns the BLOCK-CONTIGUOUS query slice [q0,q1) of the S-token chunk and computes the
 * ATTENTION HALF of every layer (input-norm, QKV, RoPE/KV, selection, absorbed-MLA attention,
 * o_proj) ONLY for its slice — the compute that the classic path replicates on every rank
 * (and that head-TP caps at 64 ways with 32/96 ranks idle) parallelizes N-way and BALANCED,
 * which removes the straggler barrier measured as ~40%% of the prefill wall.
 *
 * v1 REQUIRES fully replicated weights (load with GLM5_TP=0: attention/shared/dense/head;
 * int8@96n ~26 GB fits), CP off, MSA off. Cross-rank coupling reuses ar_cb ONLY, via
 * ZERO-EXTENDED sums that act as allgathers (0 + v = v exactly, and glm5_kv_enc is
 * deterministic, so results stay bit-identical to the classic chunk path per token):
 *   (1) latent-KV gather  : home rows staged at GLOBAL row offsets in ms->k, rest zeroed,
 *                           ar_cb over [S,KVC] (small: 576 vs H=6144), then EVERY rank
 *                           encodes+stores all S positions (queries attend earlier slices
 *                           of the SAME chunk).
 *   (2) attention-out gather: home o_proj rows staged at global offsets in ms->o, ar_cb over
 *                           [S,H] (replaces the classic o_proj AR 1:1 in bytes).
 * The MoE/dense/head tail then runs the CLASSIC full-S code verbatim on the coherent X
 * (router/experts/shared identical, one route AR). Net vs classic TP1: same 2 big ARs +1 small
 * per MoE layer, but the replicated attention-half compute is spread N-way and balanced.
 * The gather seams are where the follow-up REAL transports (multi-TNI allgather / MoE a2a,
 * moe_dispatch_bench emit/waitall) drop in to cut the remaining wire ~15x (prefill_sim lever d). */
static int glm5_forward_prefill_chunk_sp(glm5_model*m, float*X, int S, int p0, int need_head){
    m->prefill_ntok=S;
    const glm5_config*c=&m->cfg;
    const int H=c->hidden, AD=glm5_attn_dim(c);
    const int KVC=glm5_kv_cache_dim(c), KVD=glm5_kv_dim(c), half=c->qk_rope_dim/2;
    const float ascale=1.0f/sqrtf((float)c->qk_head_dim);
    glm5_mstream*ms=(glm5_mstream*)m->ms;
    const int N=m->ep_size, R=m->ep_rank;
    const int q0=(int)((long)S*R/N), q1=(int)((long)S*(R+1)/N), Sl=q1-q0;
    const int attn_window=glm5_envi("GLM5_ATTN_WINDOW",0);
    if(m->cp_on || m->msa_on) return -2;                 /* v1 scope: CP/MSA off */
    for(int l=0;l<c->n_layers;l++){
        glm5_layer*L=&m->layers[l]; int is_moe=glm5_is_moe(c,l);
        const int qh0=L->qh0,qh1=L->qh1,nown=qh1-qh0,qrows=nown*c->qk_head_dim, arows=nown*c->v_head_dim;
        const int kvb_stride=c->qk_nope_dim+c->v_head_dim; (void)qh0;
        if(arows<AD) return -3;                          /* needs replicated attention (GLM5_TP_ATTN=0) */
        double pt=glm5_prof_now();
        /* --- SP attention half: home rows only, staged at LOCAL indices --- */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<Sl;t++) glm5_rmsnorm_gemma(ms->xn+(size_t)t*H, X+(size_t)(q0+t)*H, L->input_norm, H, c->norm_eps);
        glm5_gemm(m,ms->qlat,&L->wq_a,ms->xn,Sl,c->q_lora,H);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<Sl;t++) glm5_rmsnorm_head(ms->qlat+(size_t)t*c->q_lora,L->q_a_norm,c->q_lora,c->norm_eps);
        glm5_gemm(m,ms->q,&L->wq_b,ms->qlat,Sl,qrows,c->q_lora);
        /* latent KV: compute home rows into a GLOBAL-row staging (ms->k [S,KVD]), zero elsewhere */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if((long)S*KVD>=GLM5_PAR_MIN)
#endif
        for(size_t i=0;i<(size_t)S*KVD;i++) ms->k[i]=0.f;
        glm5_gemm(m,ms->k+(size_t)q0*KVD,&L->wkv_a,ms->xn,Sl,KVC,H);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<Sl;t++){
            int p=p0+q0+t; const float*cosp=&m->rope_cos[(size_t)p*half],*sinp=&m->rope_sin[(size_t)p*half];
            float*qb=ms->q+(size_t)t*qrows, *kv=ms->k+(size_t)(q0+t)*KVD;
            for(int hh=0;hh<nown;hh++) glm5_rope_interleaved(qb+hh*c->qk_head_dim+c->qk_nope_dim,cosp,sinp,c->qk_rope_dim);
            glm5_rmsnorm_head(kv,L->kv_a_norm,c->kv_lora,c->norm_eps);
            glm5_rope_interleaved(kv+c->kv_lora,cosp,sinp,c->qk_rope_dim);
        }
        /* (1) latent-KV gather (zero-extended sum == allgather), then ALL ranks store ALL rows */
        if(m->ar_cb) m->ar_cb(ms->k,S*KVD,m->ar_ctx);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++) glm5_store_latent_kv(m,L,p0+t,ms->k+(size_t)t*KVD,KVC);
        glm5_prof_add(m,GLM5_P_QKV,pt);
        pt=glm5_prof_now();
        /* selection: full causal (optional window), positions are GLOBAL */
        for(int t=0;t<Sl;t++){
            int p=p0+q0+t, n=0, *sel=ms->psel+(size_t)t*ms->maxsel;
            int t0=(attn_window>0 && p+1>attn_window) ? p+1-attn_window : 0;
            if(attn_window<=0 && p+1>ms->maxsel) t0=p+1-ms->maxsel;
            for(int tt=t0;tt<=p;tt++) if(n<ms->maxsel) sel[n++]=tt;
            ms->pnsel[t]=n;
        }
        glm5_prof_add(m,GLM5_P_MSA_INDEX,pt);
        pt=glm5_prof_now();
        static int qk_mode=-1; if(qk_mode<0){ qk_mode=glm5_envi("GLM5_ATTN_QK",0); if(qk_mode<0||qk_mode>2) qk_mode=0; }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<Sl;t++){ float*qb=ms->q+(size_t)t*qrows,*ab=ms->attn+(size_t)t*arows;
            const int*sel=ms->psel+(size_t)t*ms->maxsel; int ns=ms->pnsel[t];
            float*hmx=ms->hmx+(size_t)t*64,*hse=ms->hse+(size_t)t*64;
            float*qabs=ms->kvb+(size_t)t*c->n_heads*(2*c->kv_lora);
            float*ctxb=qabs+(size_t)c->n_heads*c->kv_lora;
            glm5_prefill_absorb_token(m,L,qb,ab,sel,ns,hmx,hse,qabs,ctxb,ms->v+(size_t)t*KVD,nown,qrows,arows,qk_mode);
            for(int hh=0;hh<nown;hh++){
                float inv=1.0f/(hse[hh]>0?hse[hh]:1); float*oh=ab+hh*c->v_head_dim;
                for(int i=0;i<c->v_head_dim;i++) oh[i]*=inv;
            }
        }
        glm5_prof_add(m,GLM5_P_ATTN,pt);
        pt=glm5_prof_now();
        /* (2) o_proj on home rows at GLOBAL staging offsets, zero-extended gather, full X update */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
        for(size_t i=0;i<(size_t)S*H;i++) ms->o[i]=0.f;
        glm5_gemm(m,ms->o+(size_t)q0*H,&L->wo,ms->attn,Sl,H,arows);
        if(m->ar_cb) m->ar_cb(ms->o,S*H,m->ar_ctx);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
        for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->o[i];
        glm5_prof_add(m,GLM5_P_OPROJ,pt);
        /* --- classic full-S tail: post-norm + MoE / dense FFN (mirrors glm5_forward_prefill_chunk) --- */
        pt=glm5_prof_now();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++) glm5_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, L->post_norm, H, c->norm_eps);
        glm5_prof_add(m,GLM5_P_OTHER,pt);
        if(is_moe){
            const int tp_sh=(L->sh_rows<c->moe_inter);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*H;i++) ms->route[i]=0;
            int na=c->n_active>8?8:c->n_active; int*sel_all=ms->gsel; float*selw_all=ms->gselw;
            pt=glm5_prof_now();
            glm5_gemm(m,ms->router,&L->gate,ms->h2,S,c->n_experts,H);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*c->n_experts>=GLM5_PAR_MIN)
#endif
            for(int t=0;t<S;t++){ float*rl=ms->router+(size_t)t*c->n_experts;
                glm5_sigmoid_row(rl,c->n_experts);
                int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int best=-1; float bv=-1e30f; for(int e=0;e<c->n_experts;e++){ int used=0; for(int j=0;j<a;j++) if(sel[j]==e){used=1;break;} if(used)continue; float vv=rl[e]+L->gate_bias[e]; if(vv>bv){bv=vv;best=e;} } sel[a]=best; sw[a]=rl[best]; }
                float wsum=0; for(int a=0;a<na;a++) wsum+=sw[a]; if(wsum<=0)wsum=1; for(int a=0;a<na;a++) sw[a]=sw[a]/wsum*c->routed_scale; }
            glm5_prof_add(m,GLM5_P_ROUTER,pt);
            pt=glm5_prof_now();
            for(int s=0;s<L->n_owned;s++) ms->bcnt[s]=0;
            for(int t=0;t<S;t++){ int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int e=sel[a]; if(e%m->ep_size!=m->ep_rank) continue; int slot=e/m->ep_size;
                    int g=ms->bcnt[slot]++; ms->bk[(size_t)slot*S+g]=t; ms->bw[(size_t)slot*S+g]=sw[a]; } }
            for(int s=0;s<L->n_owned;s++){ int g=ms->bcnt[s]; if(g==0) continue;
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*S+i]; memcpy(ms->o+(size_t)i*H, ms->h2+(size_t)t*H, (size_t)H*4); }
                if(!glm5_gemm_pair(m,ms->shg,&L->ex_w1[s],ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H)){
                    glm5_gemm(m,ms->shg,&L->ex_w1[s],ms->o,g,c->moe_inter,H);
                    glm5_gemm(m,ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H);
                }
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) if((long)g*c->moe_inter>=GLM5_PAR_MIN)
#endif
                for(size_t i=0;i<(size_t)g*c->moe_inter;i++) ms->shg[i]=glm5_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
                glm5_gemm(m,ms->tmp2,&L->ex_w2[s],ms->shg,g,H,c->moe_inter);
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*S+i]; float w=ms->bw[(size_t)s*S+i];
                    glm5_axpy_f32(ms->route+(size_t)t*H, ms->tmp2+(size_t)i*H, w, H); } }
            glm5_prof_add(m,GLM5_P_EXPERTS,pt);
            pt=glm5_prof_now();
            if(!glm5_gemm_pair(m,ms->shg,&L->sh_w1,ms->shu,&L->sh_w3,ms->h2,S,L->sh_rows,H)){
                glm5_gemm(m,ms->shg,&L->sh_w1,ms->h2,S,L->sh_rows,H);
                glm5_gemm(m,ms->shu,&L->sh_w3,ms->h2,S,L->sh_rows,H);
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*L->sh_rows>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*L->sh_rows;i++) ms->shg[i]=glm5_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_gemm(m,ms->tmp2,&L->sh_w2,ms->shg,S,H,L->sh_rows);
            if(tp_sh) for(size_t i=0;i<(size_t)S*H;i++) ms->route[i]+=ms->tmp2[i];
            if(m->ar_cb) m->ar_cb(ms->route,S*H,m->ar_ctx);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->route[i] + (tp_sh?0.0f:ms->tmp2[i]);
            glm5_prof_add(m,GLM5_P_SHARED,pt);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            pt=glm5_prof_now();
            if(!glm5_gemm_pair(m,ms->ffg,&L->ff_gate,ms->ffu,&L->ff_up,ms->h2,S,L->ff_rows,H)){
                glm5_gemm(m,ms->ffg,&L->ff_gate,ms->h2,S,L->ff_rows,H);
                glm5_gemm(m,ms->ffu,&L->ff_up,ms->h2,S,L->ff_rows,H);
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*L->ff_rows>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*L->ff_rows;i++) ms->ffg[i]=glm5_swiglu_oai(ms->ffg[i],ms->ffu[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_gemm(m,ms->tmp2,&L->ff_down,ms->ffg,S,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(ms->tmp2,S*H,m->ar_ctx);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)S*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->tmp2[i];
            glm5_prof_add(m,GLM5_P_DENSE_FFN,pt);
        }
    }
    if(!need_head) return -1;
    double pt=glm5_prof_now();
    int hrows=m->head.rows; glm5_rmsnorm_gemma(ms->h2, X+(size_t)(S-1)*H, m->out_norm, H, c->norm_eps);
    glm5_gemm(m,ms->logits,&m->head,ms->h2,1,hrows,H);
    float*lg=ms->logits; int la=0; float bv=lg[0]; for(int i=1;i<hrows;i++) if(lg[i]>bv){bv=lg[i];la=i;}
    int32_t gidx=m->head_r0+la; float gval=bv;
    if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);
    glm5_prof_add(m,GLM5_P_HEAD,pt);
    return gidx;
}

/* ============ multi-stream MLA decode (GLM5_BATCH_DECODE) ============
 * M independent decode streams in ONE forward. Per-stream latent KV lives in ms->kc (CP OFF ->
 * full KV local, so attention needs NO collective); the only per-layer collective is the MoE
 * all-reduce, now amortized over M tokens (the decode-throughput win). Greedy: out[t]=argmax.
 * v1 mirrors the validated glm5_forward_prefill_chunk; deltas vs it = per-stream pos[t],
 * per-stream KV store/load, per-stream full-causal selection, per-stream head. MSA/index-sparse
 * selection is deferred (full causal); routed-expert grouping is inherited from prefill_chunk.
 * Validate at M=1 vs glm5_forward_token (argmax) via glm5_batch_decode_selfcheck. */
static void glm5_forward_batch_decode_mla(glm5_model*m, float*X, int M, const int*pos, int*out){
    const glm5_config*c=&m->cfg;
    const int H=c->hidden, AD=glm5_attn_dim(c);
    const int KVC=glm5_kv_cache_dim(c), half=c->qk_rope_dim/2;
    const float ascale=1.0f/sqrtf((float)c->qk_head_dim);
    glm5_mstream*ms=(glm5_mstream*)m->ms;
    const size_t per=(size_t)c->n_layers*c->max_pos*KVC;     /* per-stream kc stride */
    const int absorb_sve_dot=glm5_envi("GLM5_ABSORB_SVE_DOT",1);
    for(int l=0;l<c->n_layers;l++){
        glm5_layer*L=&m->layers[l]; int is_moe=glm5_is_moe(c,l);
        const int qh0=L->qh0,qh1=L->qh1,nown=qh1-qh0,qrows=nown*c->qk_head_dim, arows=nown*c->v_head_dim;
        const int tp_attn=(arows<AD), kvb_stride=c->qk_nope_dim+c->v_head_dim; (void)qh0;
        double pt=glm5_prof_now();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<M;t++) glm5_rmsnorm_gemma(ms->xn+(size_t)t*H, X+(size_t)t*H, L->input_norm, H, c->norm_eps);
        glm5_gemm(m,ms->qlat,&L->wq_a,ms->xn,M,c->q_lora,H);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<M;t++) glm5_rmsnorm_head(ms->qlat+(size_t)t*c->q_lora,L->q_a_norm,c->q_lora,c->norm_eps);
        glm5_gemm(m,ms->q,&L->wq_b,ms->qlat,M,qrows,c->q_lora);
        glm5_gemm(m,ms->k,&L->wkv_a,ms->xn,M,KVC,H);
        /* RoPE + per-stream latent-KV store into ms->kc[stream][layer][pos[t]] */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<M;t++){
            int p=pos[t]; const float*cosp=&m->rope_cos[(size_t)p*half],*sinp=&m->rope_sin[(size_t)p*half];
            float*qb=ms->q+(size_t)t*qrows, *kv=ms->k+(size_t)t*KVC;
            for(int hh=0;hh<nown;hh++) glm5_rope_interleaved(qb+hh*c->qk_head_dim+c->qk_nope_dim,cosp,sinp,c->qk_rope_dim);
            glm5_rmsnorm_head(kv,L->kv_a_norm,c->kv_lora,c->norm_eps);
            glm5_rope_interleaved(kv+c->kv_lora,cosp,sinp,c->qk_rope_dim);
            int st=ms->sid?ms->sid[t]:t;                       /* KV stream of this entry */
            uint16_t*kc=ms->kc+(size_t)st*per+(size_t)l*c->max_pos*KVC+(size_t)p*KVC;
            for(int i=0;i<KVC;i++) kc[i]=glm5_kv_enc(m,kv[i]);
        }
        glm5_prof_add(m,GLM5_P_QKV,pt);
        /* per-stream absorbed-MLA attention over own KV [0..pos[t]] (CP OFF, no collective) */
        pt=glm5_prof_now();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<M;t++){
            int p=pos[t];
            float*qb=ms->q+(size_t)t*qrows,*ab=ms->attn+(size_t)t*arows;
            float*qabs=ms->kvb+(size_t)t*c->n_heads*(2*c->kv_lora); float*ctxb=qabs+(size_t)c->n_heads*c->kv_lora;
            float*hmx=ms->hmx+(size_t)t*64,*hse=ms->hse+(size_t)t*64;
            int st=ms->sid?ms->sid[t]:t;                       /* KV stream of this entry */
            float*kv=ms->v+(size_t)t*KVC; const uint16_t*kcl=ms->kc+(size_t)st*per+(size_t)l*c->max_pos*KVC;
            for(int hh=0;hh<nown;hh++){
                hmx[hh]=-1e30f; hse[hh]=0.0f;
                glm5_tensor_tmul_rows(m,qabs+(size_t)hh*c->kv_lora,&L->wkv_b,hh*kvb_stride,c->qk_nope_dim,qb+hh*c->qk_head_dim);
                float*oh=ab+hh*c->v_head_dim; for(int i=0;i<c->v_head_dim;i++) oh[i]=0.0f;
                float*ctx=ctxb+(size_t)hh*c->kv_lora; for(int i=0;i<c->kv_lora;i++) ctx[i]=0.0f;
            }
            for(int j=0;j<=p;j++){
                const uint16_t*kcj=kcl+(size_t)j*KVC; for(int i=0;i<KVC;i++) kv[i]=glm5_kv_dec(m,kcj[i]);
                for(int hh=0;hh<nown;hh++){
                    const float*qh=qb+hh*c->qk_head_dim; const float*qa=qabs+(size_t)hh*c->kv_lora;
                    double d=(double)glm5_dot_f32_opt(qa,kv,c->kv_lora,absorb_sve_dot);
                    for(int i=0;i<c->qk_rope_dim;i++) d+=(double)qh[c->qk_nope_dim+i]*kv[c->kv_lora+i];
                    float s=(float)d*ascale; float*ctx=ctxb+(size_t)hh*c->kv_lora;
                    if(s>hmx[hh]){ float r=(hmx[hh]>-1e20f)?expf(hmx[hh]-s):0.0f; hse[hh]*=r; glm5_scale_f32(ctx,r,c->kv_lora); hmx[hh]=s; }
                    float e=expf(s-hmx[hh]); hse[hh]+=e; glm5_axpy_f32(ctx,kv,e,c->kv_lora);
                }
            }
            for(int hh=0;hh<nown;hh++){
                float inv=1.0f/(hse[hh]>0?hse[hh]:1); float*ctx=ctxb+(size_t)hh*c->kv_lora; glm5_scale_f32(ctx,inv,c->kv_lora);
                glm5_tensor tv=glm5_tensor_rows(&L->wkv_b,hh*kvb_stride+c->qk_nope_dim,c->v_head_dim);
                glm5_mv(m,ab+hh*c->v_head_dim,&tv,ctx,c->v_head_dim,c->kv_lora);
            }
        }
        glm5_prof_add(m,GLM5_P_ATTN,pt);
        pt=glm5_prof_now();
        glm5_gemm(m,ms->o,&L->wo,ms->attn,M,H,arows);
        /* CUT 2 ALL-REDUCES/LAYER -> 1: this attention o-proj all-reduce only exists when heads are
         * SHARDED (tp_attn). It cannot be fused with the MoE all-reduce (the post-attn RMSNorm between
         * them is nonlinear). Decode is comm-bound, so the right choice is REPLICATED attention
         * (load with GLM5_TP_ATTN=0 -> qh1==n_heads -> tp_attn=0 here -> NO attention AR -> 1 AR/MoE
         * layer, ~1.95x decode per the sim). Prefill keeps sharding (bandwidth-bound). */
        if(tp_attn && m->ar_cb) m->ar_cb(ms->o,M*H,m->ar_ctx);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if((long)M*H>=GLM5_PAR_MIN)
#endif
        for(size_t i=0;i<(size_t)M*H;i++) X[i]+=ms->o[i];
        glm5_prof_add(m,GLM5_P_OPROJ,pt);
        pt=glm5_prof_now();
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<M;t++) glm5_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, L->post_norm, H, c->norm_eps);
        glm5_prof_add(m,GLM5_P_OTHER,pt);
        if(is_moe){
            const int tp_sh=(L->sh_rows<c->moe_inter);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)M*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)M*H;i++) ms->route[i]=0;
            int na=c->n_active>8?8:c->n_active; int*sel_all=ms->gsel; float*selw_all=ms->gselw;
            pt=glm5_prof_now();
            glm5_gemm(m,ms->router,&L->gate,ms->h2,M,c->n_experts,H);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)M*c->n_experts>=GLM5_PAR_MIN)
#endif
            for(int t=0;t<M;t++){ float*rl=ms->router+(size_t)t*c->n_experts; glm5_sigmoid_row(rl,c->n_experts);
                int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int best=-1; float bv=-1e30f; for(int e=0;e<c->n_experts;e++){ int used=0; for(int j=0;j<a;j++) if(sel[j]==e){used=1;break;} if(used)continue; float vv=rl[e]+L->gate_bias[e]; if(vv>bv){bv=vv;best=e;} } sel[a]=best; sw[a]=rl[best]; }
                float wsum=0; for(int a=0;a<na;a++) wsum+=sw[a]; if(wsum<=0)wsum=1; for(int a=0;a<na;a++) sw[a]=sw[a]/wsum*c->routed_scale; }
            glm5_prof_add(m,GLM5_P_ROUTER,pt);
            pt=glm5_prof_now();
            for(int s=0;s<L->n_owned;s++) ms->bcnt[s]=0;
            for(int t=0;t<M;t++){ int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int e=sel[a]; if(e%m->ep_size!=m->ep_rank) continue; int slot=e/m->ep_size;
                    int g=ms->bcnt[slot]++; ms->bk[(size_t)slot*M+g]=t; ms->bw[(size_t)slot*M+g]=sw[a]; } }
            for(int s=0;s<L->n_owned;s++){ int g=ms->bcnt[s]; if(g==0) continue;
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*M+i]; memcpy(ms->o+(size_t)i*H, ms->h2+(size_t)t*H, (size_t)H*4); }
                if(!glm5_gemm_pair(m,ms->shg,&L->ex_w1[s],ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H)){
                    glm5_gemm(m,ms->shg,&L->ex_w1[s],ms->o,g,c->moe_inter,H);
                    glm5_gemm(m,ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H);
                }
#ifdef _OPENMP
                #pragma omp parallel for schedule(static) if((long)g*c->moe_inter>=GLM5_PAR_MIN)
#endif
                for(size_t i=0;i<(size_t)g*c->moe_inter;i++) ms->shg[i]=glm5_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
                glm5_gemm(m,ms->tmp2,&L->ex_w2[s],ms->shg,g,H,c->moe_inter);
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*M+i]; float w=ms->bw[(size_t)s*M+i];
                    glm5_axpy_f32(ms->route+(size_t)t*H, ms->tmp2+(size_t)i*H, w, H); } }
            glm5_prof_add(m,GLM5_P_EXPERTS,pt);
            pt=glm5_prof_now();
            int overlap = (m->ar_async_start && !tp_sh);
            if(overlap) m->ar_async_start(ms->route,M*H,m->ar_async_ctx);
            if(!glm5_gemm_pair(m,ms->shg,&L->sh_w1,ms->shu,&L->sh_w3,ms->h2,M,L->sh_rows,H)){
                glm5_gemm(m,ms->shg,&L->sh_w1,ms->h2,M,L->sh_rows,H);
                glm5_gemm(m,ms->shu,&L->sh_w3,ms->h2,M,L->sh_rows,H);
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)M*L->sh_rows>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)M*L->sh_rows;i++) ms->shg[i]=glm5_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_gemm(m,ms->tmp2,&L->sh_w2,ms->shg,M,H,L->sh_rows);
            if(overlap){ m->ar_wait(m->ar_async_ctx); }
            else { if(tp_sh) for(size_t i=0;i<(size_t)M*H;i++) ms->route[i]+=ms->tmp2[i];
                   if(m->ar_cb) m->ar_cb(ms->route,M*H,m->ar_ctx); }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)M*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)M*H;i++) X[i]+=ms->route[i] + (tp_sh?0.0f:ms->tmp2[i]);
            glm5_prof_add(m,GLM5_P_SHARED,pt);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            pt=glm5_prof_now();
            if(!glm5_gemm_pair(m,ms->ffg,&L->ff_gate,ms->ffu,&L->ff_up,ms->h2,M,L->ff_rows,H)){
                glm5_gemm(m,ms->ffg,&L->ff_gate,ms->h2,M,L->ff_rows,H);
                glm5_gemm(m,ms->ffu,&L->ff_up,ms->h2,M,L->ff_rows,H);
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)M*L->ff_rows>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)M*L->ff_rows;i++) ms->ffg[i]=glm5_swiglu_oai(ms->ffg[i],ms->ffu[i],c->swiglu_alpha,c->swiglu_limit);
            glm5_gemm(m,ms->tmp2,&L->ff_down,ms->ffg,M,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(ms->tmp2,M*H,m->ar_ctx);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) if((long)M*H>=GLM5_PAR_MIN)
#endif
            for(size_t i=0;i<(size_t)M*H;i++) X[i]+=ms->tmp2[i];
            glm5_prof_add(m,GLM5_P_DENSE_FFN,pt);
        }
    }
    /* head: ALL M streams, greedy argmax (+ TP_HEAD merge per stream) or sampling */
    double pt=glm5_prof_now(); int hrows=m->head.rows;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<M;t++) glm5_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, m->out_norm, H, c->norm_eps);
    glm5_gemm(m,ms->logits,&m->head,ms->h2,M,hrows,H);
    if(m->samp_temp>0.0f){
        /* batched sampling: gather ALL streams' full logits with ONE all-reduce (not M),
         * then sample per stream in stream order with the shared-seed RNG -- every rank
         * holds identical full logits and advances the RNG identically => lockstep,
         * matching glm5_forward_token's sampling contract. Per-stream rep-penalty
         * history comes from ms->hist/hist_n (caller-set; NULL => no history). */
        float*SL=ms->slog;
        if(hrows<c->vocab){
            memset(SL,0,(size_t)M*c->vocab*4);
            for(int t=0;t<M;t++) memcpy(SL+(size_t)t*c->vocab+m->head_r0, ms->logits+(size_t)t*hrows, (size_t)hrows*4);
            if(m->ar_cb) m->ar_cb(SL,M*c->vocab,m->ar_ctx);
        } else for(int t=0;t<M;t++) memcpy(SL+(size_t)t*c->vocab, ms->logits+(size_t)t*hrows, (size_t)c->vocab*4);
        for(int t=0;t<M;t++){
            memcpy(m->s_logits, SL+(size_t)t*c->vocab, (size_t)c->vocab*4);
            m->samp_hist   = ms->hist   ? ms->hist[t]   : NULL;
            m->samp_hist_n = ms->hist_n ? ms->hist_n[t] : 0;
            out[t]=glm5_sample_logits(m,c->vocab);
        }
        glm5_prof_add(m,GLM5_P_HEAD,pt);
        return;
    }
    if(hrows<c->vocab && m->ar_argmax_n_cb){
        /* batched TP_HEAD merge: all M streams' (val,idx) pairs in ONE collective */
        float vi[2*64];   /* M<=64 streams */
        int MM=M>64?64:M;
        for(int t=0;t<MM;t++){
            float*lg=ms->logits+(size_t)t*hrows; int la=0; float bv=lg[0];
            for(int i=1;i<hrows;i++) if(lg[i]>bv){bv=lg[i];la=i;}
            int32_t gidx=m->head_r0+la; vi[2*t]=bv; memcpy(&vi[2*t+1],&gidx,4);
        }
        m->ar_argmax_n_cb(vi,MM,m->ar_argmax_n_ctx);
        for(int t=0;t<MM;t++){ int32_t gi; memcpy(&gi,&vi[2*t+1],4); out[t]=gi; }
        glm5_prof_add(m,GLM5_P_HEAD,pt);
        return;
    }
    for(int t=0;t<M;t++){
        float*lg=ms->logits+(size_t)t*hrows; int la=0; float bv=lg[0];
        for(int i=1;i<hrows;i++) if(lg[i]>bv){bv=lg[i];la=i;}
        int32_t gidx=m->head_r0+la; float gval=bv;
        if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);
        out[t]=gidx;
    }
    glm5_prof_add(m,GLM5_P_HEAD,pt);
}

/* ===================== MTP (multi-token prediction, checkpoint layer 78) =====================
 * Draft head: predicts token t+2 from (residual hidden at position t+1's producer, embedding of
 * token t+1):  x = eh_proj . [ enorm(emb(tok)) ; hnorm(h) ]  ->  one full transformer block
 * (own latent-KV cache, EP-sharded MoE)  ->  shared out_norm + lm_head.
 * The block runs through a 1-LAYER MODEL VIEW so it reuses the entire validated
 * glm5_forward_token path (absorbed-MLA attention, MoE ar_cb, TP_HEAD argmax merge). */
static glm5_model glm5_mtp_view(glm5_model*m){
    glm5_model v=*m;
    v.layers=m->mtp_layer; v.cfg.n_layers=1; v.cfg.n_dense_layers=0;  /* the block is MoE */
    v.mtp_layer=NULL; v.ms=NULL;
    if(m->mtp_out_norm) v.out_norm=m->mtp_out_norm;  /* MTP uses shared_head.norm, not model.norm */
    return v;
}
/* argmax draft token. h = the residual hidden (post-all-layers, PRE-out_norm) of the forward
 * that produced `tok` (the caller's activation buffer after glm5_forward_token /
 * glm5_forward_batch_decode_mla — both leave exactly this in X). pos = tok's position.
 * xb: caller scratch, 2*hidden floats. Draft is greedy (verification decides acceptance). */
static int glm5_mtp_draft(glm5_model*m,const float*h,int tok,int pos,float*xb){
    const glm5_config*c=&m->cfg; const int H=c->hidden;
    if(!m->mtp_layer) return -1;
    float*x=m->s_ff;                                   /* [dense_inter] scratch >= H */
    if(tok<0||tok>=c->vocab) tok=0;
    if(m->emb_rows<c->vocab){                          /* TP_EMBED: owned rows + AR sum */
        for(int i=0;i<H;i++) x[i]=0.f;
        if(tok>=m->emb_r0 && tok<m->emb_r0+m->emb_rows){
            const uint16_t*row=m->embed+(size_t)(tok-m->emb_r0)*H;
            for(int i=0;i<H;i++) x[i]=glm5_bf2f(row[i]);
        }
        if(m->ar_cb) m->ar_cb(x,H,m->ar_ctx);
    } else {
        const uint16_t*row=m->embed+(size_t)tok*H;
        for(int i=0;i<H;i++) x[i]=glm5_bf2f(row[i]);
    }
    glm5_rmsnorm_gemma(xb,   x, m->mtp_enorm, H, c->norm_eps);
    glm5_rmsnorm_gemma(xb+H, h, m->mtp_hnorm, H, c->norm_eps);
    glm5_mv(m,x,&m->mtp_eh,xb,H,2*H);
    glm5_model v=glm5_mtp_view(m);
    float t0=v.samp_temp; v.samp_temp=0;               /* draft is always greedy */
    int d=glm5_forward_token(&v,x,pos); (void)t0;
    return d;
}
/* Synthetic MTP block for simulator/lockstep validation (no real layer-78 weights needed).
 * Same glm5_sm reset discipline as the main model applies (caller's responsibility in
 * multi-rank single-address-space sims). Real weights: stage with GLM5_STAGE_LAYERS=79 and
 * load model.layers.78.{enorm,hnorm,eh_proj,block...} (loader TODO, job-validated). */
static int glm5_alloc_mtp_synth(glm5_model*m){
    glm5_config c1=m->cfg; c1.n_layers=1; c1.n_dense_layers=0;
    glm5_model*t=glm5_alloc_synth(c1,m->ep_rank,m->ep_size,m->n_threads,m->n_cmgs);
    if(!t) return -1;
    m->mtp_layer=&t->layers[0];                        /* wrapper model intentionally kept (sim-only) */
    const int H=m->cfg.hidden;
    m->mtp_enorm=glm5_amalloc((size_t)H*2); glm5_fill_bf16(m->mtp_enorm,H,0.1f);
    m->mtp_hnorm=glm5_amalloc((size_t)H*2); glm5_fill_bf16(m->mtp_hnorm,H,0.1f);
    uint16_t*p=glm5_amalloc((size_t)H*2*H*2); if(!p) return -1;
    glm5_fill_bf16(p,(size_t)H*2*H,0.03f);
    m->mtp_eh=(glm5_tensor){p,NULL,GLM5_BF16,H,2*H};
    return 0;
}

#endif /* GLM5_IMPL_H */
