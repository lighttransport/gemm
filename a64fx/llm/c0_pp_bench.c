/* C0 follow-up: confirm SINGLE-NODE position-parallel attention captures the
 * win for the 27B head config (n_heads/rank = 2-3, not 16 -> no qpkd K_DP).
 *
 * Today: tf_attn_worker parallelises over HEADS -> a rank's 3 local heads run on
 * 3 of 48 threads, each serial over all S. C0 measured that = 6.13 ms/layer @8192.
 *
 * Fix: parallelise over (head x position-chunk) -> all 48 threads busy, each does
 * 1 head over S/(48/n_heads) positions, with an intra-rank online-softmax merge
 * across the threads sharing a head. Same idea the K_DP path uses for 9B (+211%).
 * Expect ~16x for 3 heads (48/3) -> ~0.4 ms/layer -> ctx8192 decode ~34 tok/s.
 *
 * This bench implements a faithful SVE F16 position-parallel attention and times
 * it head-parallel (HP, today) vs position-parallel (PP) for the 27B shape.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <arm_sve.h>

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec + (double)t.tv_nsec*1e-9; }

#define HD 256
#define NKV 4
#define GQA 6
#define NTH 48
#define SMAX 8192

static float *g_q, *g_xb2;
static uint16_t *g_kc, *g_vc;
static size_t g_kv_dim;
/* per-thread partials: [tid][HD] o, plus m,l */
static float *g_o; static float *g_m; static float *g_l;

static uint16_t f2h(float v){ __fp16 h=(__fp16)v; uint16_t u; memcpy(&u,&h,2); return u; }

/* QK score for head h at position p: dot(q_h, K[p][kv_h]) * scale, F16 K. */
static inline float qk_score(const float *q_h, const uint16_t *kbase, float scale){
    svbool_t pg = svptrue_b32();
    svfloat32_t acc = svdup_f32(0.0f);
    for (int d=0; d<HD; d+=16){
        svfloat32_t qv = svld1_f32(pg, q_h + d);
        svuint32_t  ku = svld1uh_u32(pg, kbase + d);
        svfloat32_t kv = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(ku));
        acc = svmla_x(pg, acc, qv, kv);
    }
    return svaddv_f32(pg, acc) * scale;
}
/* o[0..HD) += a * V[p][kv_h], F16 V. */
static inline void av_acc(float *o, const uint16_t *vbase, float a){
    svbool_t pg = svptrue_b32();
    svfloat32_t av = svdup_f32(a);
    for (int d=0; d<HD; d+=16){
        svfloat32_t ov = svld1_f32(pg, o + d);
        svuint32_t  vu = svld1uh_u32(pg, vbase + d);
        svfloat32_t vv = svcvt_f32_f16_x(pg, svreinterpret_f16_u32(vu));
        ov = svmla_x(pg, ov, av, vv);
        svst1_f32(pg, o + d, ov);
    }
}

/* One thread's partial over head h, positions [p_lo,p_hi): two-pass softmax,
 * writes (m,l,o) for slot `slot`. */
static void partial(int h, int p_lo, int p_hi, int slot, float scale){
    int kv_h = h / GQA;
    const float *q_h = g_q + (size_t)h*HD;
    float m = -INFINITY;
    for (int p=p_lo; p<p_hi; p++){
        const uint16_t *kb = g_kc + (size_t)p*g_kv_dim + (size_t)kv_h*HD;
        float s = qk_score(q_h, kb, scale);
        if (s>m) m=s;
    }
    float *o = g_o + (size_t)slot*HD;
    memset(o, 0, HD*sizeof(float));
    float l = 0.0f;
    for (int p=p_lo; p<p_hi; p++){
        const uint16_t *kb = g_kc + (size_t)p*g_kv_dim + (size_t)kv_h*HD;
        float s = qk_score(q_h, kb, scale);
        float e = expf(s - m);
        l += e;
        const uint16_t *vb = g_vc + (size_t)p*g_kv_dim + (size_t)kv_h*HD;
        av_acc(o, vb, e);
    }
    g_m[slot]=m; g_l[slot]=l;
}

/* Head-parallel (today): n_heads on n_heads threads, each full S. */
static double run_HP(int n_heads, int seq_len, int iters, int warm){
    double out=0;
    #pragma omp parallel num_threads(NTH)
    {
        int tid=omp_get_thread_num();
        int active = tid < n_heads;
        float sc = 1.0f/sqrtf(HD);
        for(int it=0; it<warm; it++){
            if(active) partial(tid,0,seq_len,tid,sc);
            #pragma omp barrier
        }
        #pragma omp barrier
        double t0=0; if(tid==0) t0=now_s();
        #pragma omp barrier
        for(int it=0; it<iters; it++){
            if(active) partial(tid,0,seq_len,tid,sc);
            #pragma omp barrier
        }
        if(tid==0){ out=(now_s()-t0)/iters*1000.0; }
    }
    return out;
}

/* Position-parallel: each head split across (NTH/n_heads) threads; intra-rank
 * online-softmax merge across the threads sharing a head. */
static double run_PP(int n_heads, int seq_len, int iters, int warm){
    double out=0;
    int tph = NTH / n_heads;           /* threads per head */
    if (tph < 1) tph = 1;
    #pragma omp parallel num_threads(NTH)
    {
        int tid=omp_get_thread_num();
        int h = tid / tph;             /* which head this thread serves */
        int c = tid % tph;             /* chunk index within the head */
        int active = h < n_heads;
        int p_lo = active ? (int)((long)c*seq_len/tph) : 0;
        int p_hi = active ? (int)((long)(c+1)*seq_len/tph) : 0;
        float scale = 1.0f/sqrtf(HD);
        for(int w=0; w<warm+iters; w++){
            int timed = (w>=warm);
            double t0=0;
            #pragma omp barrier
            if(timed && tid==0) t0=now_s();
            if(active) partial(h, p_lo, p_hi, tid, scale);
            #pragma omp barrier
            /* merge: thread c==0 of each head combines its tph chunks online */
            if(active && c==0){
                float gm=-INFINITY;
                for(int k=0;k<tph;k++){ float mk=g_m[tid+k]; if(mk>gm) gm=mk; }
                float gl=0; float *go=g_xb2+(size_t)h*HD; memset(go,0,HD*sizeof(float));
                for(int k=0;k<tph;k++){
                    float w2=expf(g_m[tid+k]-gm); gl+=w2*g_l[tid+k];
                    float *ok=g_o+(size_t)(tid+k)*HD;
                    for(int d=0;d<HD;d++) go[d]+=w2*ok[d];
                }
                float inv = gl>0?1.0f/gl:0.0f;
                for(int d=0;d<HD;d++) go[d]*=inv;
            }
            #pragma omp barrier
            if(timed && tid==0) out += (now_s()-t0)*1000.0;
        }
        if(tid==0) out/=iters;
    }
    return out;
}

int main(void){
    g_kv_dim=(size_t)NKV*HD;
    g_q  = aligned_alloc(256,(size_t)24*HD*sizeof(float));
    g_xb2= aligned_alloc(256,(size_t)24*HD*sizeof(float));
    g_o  = aligned_alloc(256,(size_t)NTH*HD*sizeof(float));
    g_m  = aligned_alloc(256,(size_t)NTH*sizeof(float));
    g_l  = aligned_alloc(256,(size_t)NTH*sizeof(float));
    g_kc = aligned_alloc(256,(size_t)SMAX*g_kv_dim*sizeof(uint16_t));
    g_vc = aligned_alloc(256,(size_t)SMAX*g_kv_dim*sizeof(uint16_t));
    for(int i=0;i<24*HD;i++) g_q[i]=0.02f*((i%11)-5);
    for(size_t i=0;i<(size_t)SMAX*g_kv_dim;i++){ g_kc[i]=f2h(0.015f*((int)(i%13)-6)); g_vc[i]=f2h(0.010f*((int)(i%17)-8)); }

    printf("# C0-PP  HD=%d NKV=%d GQA=%d NTH=%d kv=F16\n",HD,NKV,GQA,NTH);
    printf("# cfg                 nh    S    HP_ms    PP_ms   speedup   PP*16(ms/tok)\n");
    struct{int nh,S;} cfgs[]={{3,8192},{2,8192},{3,4096},{3,2048},{2,4096}};
    for(int i=0;i<(int)(sizeof(cfgs)/sizeof(cfgs[0]));i++){
        int it = cfgs[i].S>=4096?120:300;
        double hp=run_HP(cfgs[i].nh,cfgs[i].S,it,20);
        double pp=run_PP(cfgs[i].nh,cfgs[i].S,it,20);
        printf("nh=%-2d S=%-5d        %3d  %5d  %7.4f  %7.4f   %6.2fx   %7.2f\n",
               cfgs[i].nh,cfgs[i].S,cfgs[i].nh,cfgs[i].S,hp,pp,hp/pp,pp*16);
        fflush(stdout);
    }
    return 0;
}
