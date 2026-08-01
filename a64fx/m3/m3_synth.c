/*
 * m3_synth.c - MiniMax-M3 (text) SINGLE-NODE synthetic forward, native A64FX.
 *
 * v0 milestone: validate the M3 forward GRAPH + the bf16 SVE matvec kernels run
 * NaN-free on one native A64FX node. Weights are SYNTHETIC (deterministic junk),
 * so logits are meaningless — the targets are: compiles with fcc, runs to
 * completion, NaNs=0, sane timing. No MPI, no EP/TP, no MSA block-selection yet
 * (sparse layers run FULL causal attention here; MSA top-k selection + the
 * clair/msa kernels land next). The full model does NOT fit one node, so use the
 * truncation knobs.
 *
 * Build (native A64FX, inside an interactive alloc):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *       -Wno-unused-function -I <repo> -o m3_synth a64fx/m3/m3_synth.c -lm
 * Cross-compile-check (login node): same flags with fccpx.
 *
 * Run:
 *   M3_LAYERS=4 M3_EXPERTS=8 M3_MAXPOS=512 ./m3_synth
 * Env:
 *   M3_LAYERS  n_layers override          (default 4;  0 = full 60 — won't fit)
 *   M3_EXPERTS n_experts override         (default 8;  full = 128 — won't fit 1 node)
 *   M3_MAXPOS  KV capacity                (default 512)
 *   M3_PREFILL synthetic prefill tokens   (default 8)
 *   M3_DECODE  synthetic decode tokens    (default 8)
 *   LLM_THREADS / OMP_NUM_THREADS         (OpenMP, default leaves it to the RT)
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "common/m3.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- helpers ---- */
static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int envi(const char*k,int d){ const char*v=getenv(k); return (v&&*v)?atoi(v):d; }

/* round-to-nearest-even f32 -> bf16 */
static inline uint16_t f2bf(float f){
    uint32_t u; memcpy(&u,&f,4);
    if (((u>>23)&0xff)==0xff) return (uint16_t)(u>>16);       /* inf/nan: keep top */
    uint32_t r = u + 0x7fffu + ((u>>16)&1u);
    return (uint16_t)(r>>16);
}
static inline float bf2f(uint16_t h){ return bf16_to_f32_scalar(h); }

/* splitmix64 -> deterministic synthetic weights/activations in [-1,1] */
static uint64_t g_sm;
static inline double sm_next(void){
    g_sm += 0x9E3779B97F4A7C15ull; uint64_t z=g_sm;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; z^=z>>31;
    return (double)(z>>11)/(double)(1ull<<53);
}
static void fill_bf16(uint16_t*w,size_t n,float amp){ for(size_t i=0;i<n;i++) w[i]=f2bf((float)((sm_next()*2.0-1.0)*amp)); }
static void fill_f32 (float   *w,size_t n,float amp){ for(size_t i=0;i<n;i++) w[i]=(float)((sm_next()*2.0-1.0)*amp); }

/* y[rows] = W[rows,cols](bf16) . x[cols](f32) */
static void mv_bf16(float*restrict y,const uint16_t*W,const float*x,int rows,int cols){
    int r=0;
    for(;r+8<=rows;r+=8){
        const uint16_t*b=W+(size_t)r*cols;
        matvec_bf16_8row(y+r,b,b+cols,b+2*(size_t)cols,b+3*(size_t)cols,
                         b+4*(size_t)cols,b+5*(size_t)cols,b+6*(size_t)cols,b+7*(size_t)cols,x,cols);
    }
    for(;r<rows;r++) y[r]=vec_dot_bf16_f32(W+(size_t)r*cols,x,cols);
}
/* f32 weight matvec (router gate, small) */
static void mv_f32(float*restrict y,const float*W,const float*x,int rows,int cols){
    for(int r=0;r<rows;r++){ const float*w=W+(size_t)r*cols; double s=0; for(int i=0;i<cols;i++) s+=(double)w[i]*x[i]; y[r]=(float)s; }
}

/* Gemma RMSNorm: out = x/rms(x) * (1 + w),  w bf16 [n] */
static void rmsnorm_gemma(float*out,const float*x,const uint16_t*w,int n,float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float inv=(float)(1.0/sqrt(ss/n+eps));
    for(int i=0;i<n;i++) out[i]=x[i]*inv*(1.0f+bf2f(w[i]));
}
/* plain RMSNorm over a head_dim slice with bf16 weight (QK-norm, per head) */
static void rmsnorm_head(float*v,const uint16_t*w,int n,float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)v[i]*v[i];
    float inv=(float)(1.0/sqrt(ss/n+eps));
    for(int i=0;i<n;i++) v[i]=v[i]*inv*(1.0f+bf2f(w[i]));   /* Gemma-style 1+w */
}

/* SwiGLU-OAI: inter = silu_oai(clamp(gate,<=lim)) * clamp(up,[-lim,lim]) */
static inline float swiglu_oai(float gate,float up,float alpha,float lim){
    float g = gate>lim?lim:gate;
    float u = up>lim?lim:(up<-lim?-lim:up);
    float s = g/(1.0f+expf(-alpha*g));     /* g*sigmoid(alpha*g) */
    return s*u;
}

/* partial RoPE on a head vector [head_dim]: rotate first rotary_dim dims as
 * rotary_dim/2 pairs (i, i+half), pass the rest through. cos/sin[pos*half+k]. */
static void rope_head(float*v,const float*cosp,const float*sinp,int rotary_dim){
    int half=rotary_dim/2;
    for(int k=0;k<half;k++){
        float c=cosp[k],s=sinp[k];
        float a=v[k],b=v[k+half];
        v[k]     = a*c - b*s;
        v[k+half]= a*s + b*c;
    }
}

/* ---- per-layer synthetic weights (truncated/small to fit one node) ---- */
typedef struct {
    uint16_t *in_norm,*post_norm;
    uint16_t *wq,*wk,*wv,*wo,*q_norm,*k_norm;     /* GQA */
    /* MoE (sparse layers) */
    float    *gate; float *gate_bias;             /* f32 router */
    uint16_t *sh_w1,*sh_w3,*sh_w2;                /* shared expert */
    uint16_t **ex_w1,**ex_w3,**ex_w2;             /* routed experts [n_experts] */
    /* dense FFN (dense layers) */
    uint16_t *ff_g,*ff_u,*ff_d;
    /* KV cache */
    uint16_t *kc,*vc;                             /* [max_pos, kv_dim] */
    int is_moe;
} layer_t;

int main(void){
    m3_config c = m3_default_config();
    int L      = envi("M3_LAYERS", 4);   if (L>0) c.n_layers=L;
    int NE     = envi("M3_EXPERTS",8);   if (NE>0) c.n_experts=NE;
    c.max_pos  = envi("M3_MAXPOS",512);
    int prefill= envi("M3_PREFILL",8);
    int decode = envi("M3_DECODE",8);
    if (prefill+decode > c.max_pos){ fprintf(stderr,"prefill+decode > max_pos\n"); return 2; }
    if (c.n_active > c.n_experts) c.n_active = c.n_experts;

    const int H=c.hidden, HD=c.head_dim, QH=c.n_heads, KVH=c.n_kv_heads;
    const int QD=m3_q_dim(&c), KVD=m3_kv_dim(&c);
    const int grp=QH/KVH;                          /* GQA group size (16) */
    const float attn_scale=1.0f/sqrtf((float)HD);
    const int half=c.rotary_dim/2;

#ifdef _OPENMP
    int thr=envi("LLM_THREADS",envi("OMP_NUM_THREADS",omp_get_max_threads()));
    omp_set_num_threads(thr);
#endif
    printf("=== M3 synth single-node forward ===\n");
    printf("layers=%d hidden=%d heads=%d/%d head_dim=%d experts=%d active=%d dense_layers=%d\n",
           c.n_layers,H,QH,KVH,HD,c.n_experts,c.n_active,c.n_dense_layers);
    printf("q_dim=%d kv_dim=%d moe_inter=%d dense_inter=%d vocab=%d max_pos=%d\n",
           QD,KVD,c.moe_inter,c.dense_inter,c.vocab,c.max_pos);

    g_sm=0xD3F00Dull;
    double tb0=now_sec();

    /* RoPE tables */
    float *rcos=malloc((size_t)c.max_pos*half*4), *rsin=malloc((size_t)c.max_pos*half*4);
    for(int p=0;p<c.max_pos;p++) for(int k=0;k<half;k++){
        double invf=pow((double)c.rope_theta, -2.0*k/(double)c.rotary_dim);
        double ang=p*invf; rcos[(size_t)p*half+k]=(float)cos(ang); rsin[(size_t)p*half+k]=(float)sin(ang);
    }

    /* embed + head + final norm */
    uint16_t *embed=malloc((size_t)c.vocab*H*2); fill_bf16(embed,(size_t)c.vocab*H,0.05f);
    uint16_t *head =malloc((size_t)c.vocab*H*2); fill_bf16(head ,(size_t)c.vocab*H,0.05f);
    uint16_t *onorm=malloc((size_t)H*2);         fill_bf16(onorm,H,0.1f);

    layer_t *ly=calloc(c.n_layers,sizeof(layer_t));
    for(int l=0;l<c.n_layers;l++){
        layer_t*L_=&ly[l]; L_->is_moe=m3_is_moe(&c,l);
        L_->in_norm=malloc(H*2);  fill_bf16(L_->in_norm,H,0.1f);
        L_->post_norm=malloc(H*2);fill_bf16(L_->post_norm,H,0.1f);
        L_->wq=malloc((size_t)QD*H*2);  fill_bf16(L_->wq,(size_t)QD*H,0.03f);
        L_->wk=malloc((size_t)KVD*H*2); fill_bf16(L_->wk,(size_t)KVD*H,0.03f);
        L_->wv=malloc((size_t)KVD*H*2); fill_bf16(L_->wv,(size_t)KVD*H,0.03f);
        L_->wo=malloc((size_t)H*QD*2);  fill_bf16(L_->wo,(size_t)H*QD,0.03f);
        L_->q_norm=malloc(HD*2); fill_bf16(L_->q_norm,HD,0.1f);
        L_->k_norm=malloc(HD*2); fill_bf16(L_->k_norm,HD,0.1f);
        L_->kc=calloc((size_t)c.max_pos*KVD,2);
        L_->vc=calloc((size_t)c.max_pos*KVD,2);
        if(L_->is_moe){
            L_->gate=malloc((size_t)c.n_experts*H*4); fill_f32(L_->gate,(size_t)c.n_experts*H,0.03f);
            L_->gate_bias=malloc((size_t)c.n_experts*4); fill_f32(L_->gate_bias,c.n_experts,0.1f);
            L_->sh_w1=malloc((size_t)c.moe_inter*H*2); fill_bf16(L_->sh_w1,(size_t)c.moe_inter*H,0.03f);
            L_->sh_w3=malloc((size_t)c.moe_inter*H*2); fill_bf16(L_->sh_w3,(size_t)c.moe_inter*H,0.03f);
            L_->sh_w2=malloc((size_t)H*c.moe_inter*2); fill_bf16(L_->sh_w2,(size_t)H*c.moe_inter,0.03f);
            L_->ex_w1=malloc(c.n_experts*sizeof(void*));
            L_->ex_w3=malloc(c.n_experts*sizeof(void*));
            L_->ex_w2=malloc(c.n_experts*sizeof(void*));
            for(int e=0;e<c.n_experts;e++){
                L_->ex_w1[e]=malloc((size_t)c.moe_inter*H*2); fill_bf16(L_->ex_w1[e],(size_t)c.moe_inter*H,0.03f);
                L_->ex_w3[e]=malloc((size_t)c.moe_inter*H*2); fill_bf16(L_->ex_w3[e],(size_t)c.moe_inter*H,0.03f);
                L_->ex_w2[e]=malloc((size_t)H*c.moe_inter*2); fill_bf16(L_->ex_w2[e],(size_t)H*c.moe_inter,0.03f);
            }
        } else {
            L_->ff_g=malloc((size_t)c.dense_inter*H*2); fill_bf16(L_->ff_g,(size_t)c.dense_inter*H,0.03f);
            L_->ff_u=malloc((size_t)c.dense_inter*H*2); fill_bf16(L_->ff_u,(size_t)c.dense_inter*H,0.03f);
            L_->ff_d=malloc((size_t)H*c.dense_inter*2); fill_bf16(L_->ff_d,(size_t)H*c.dense_inter,0.03f);
        }
    }
    printf("alloc+fill: %.2fs\n", now_sec()-tb0);

    /* scratch */
    float *xn=malloc(H*4), *q=malloc(QD*4), *kbuf=malloc(KVD*4), *vbuf=malloc(KVD*4);
    float *attn=malloc(QD*4), *ao=malloc(H*4), *h2=malloc(H*4);
    float *scorew=malloc((size_t)c.max_pos*4);
    float *shg=malloc(c.moe_inter*4), *shu=malloc(c.moe_inter*4), *moe=malloc(H*4);
    float *exg=malloc(c.moe_inter*4), *exu=malloc(c.moe_inter*4);
    float *ffg=malloc(c.dense_inter*4), *ffu=malloc(c.dense_inter*4);
    float *rlog=malloc(c.n_experts*4);
    float *logits=malloc((size_t)c.vocab*4);
    float *x=malloc(H*4);

    /* forward one token at position pos; x[H] in/out hidden; returns argmax */
    #define FORWARD_BODY                                                                      \
    for(int l=0;l<c.n_layers;l++){                                                            \
        layer_t*Ly=&ly[l];                                                                    \
        rmsnorm_gemma(xn,x,Ly->in_norm,H,c.norm_eps);                                         \
        mv_bf16(q,Ly->wq,xn,QD,H);                                                            \
        mv_bf16(kbuf,Ly->wk,xn,KVD,H);                                                        \
        mv_bf16(vbuf,Ly->wv,xn,KVD,H);                                                        \
        for(int h=0;h<QH;h++){ float*qh=q+h*HD; if(c.use_qk_norm) rmsnorm_head(qh,Ly->q_norm,HD,c.norm_eps);\
            rope_head(qh,&rcos[(size_t)pos*half],&rsin[(size_t)pos*half],c.rotary_dim); }     \
        for(int kh=0;kh<KVH;kh++){ float*kk=kbuf+kh*HD; if(c.use_qk_norm) rmsnorm_head(kk,Ly->k_norm,HD,c.norm_eps);\
            rope_head(kk,&rcos[(size_t)pos*half],&rsin[(size_t)pos*half],c.rotary_dim); }     \
        for(int i=0;i<KVD;i++){ Ly->kc[(size_t)pos*KVD+i]=f2bf(kbuf[i]); Ly->vc[(size_t)pos*KVD+i]=f2bf(vbuf[i]); } \
        for(int h=0;h<QH;h++){ float*qh=q+h*HD; int kvh=h/grp; float mx=-1e30f;               \
            for(int t=0;t<=pos;t++){ const uint16_t*kt=Ly->kc+(size_t)t*KVD+kvh*HD; double d=0;\
                for(int i=0;i<HD;i++) d+=(double)qh[i]*bf2f(kt[i]); float s=(float)d*attn_scale; scorew[t]=s; if(s>mx)mx=s; }\
            double sum=0; for(int t=0;t<=pos;t++){ float e=expf(scorew[t]-mx); scorew[t]=e; sum+=e; }   \
            float inv=(float)(1.0/(sum>0?sum:1)); float*oh=attn+h*HD; for(int i=0;i<HD;i++) oh[i]=0;    \
            for(int t=0;t<=pos;t++){ float w=scorew[t]*inv; const uint16_t*vt=Ly->vc+(size_t)t*KVD+kvh*HD;\
                for(int i=0;i<HD;i++) oh[i]+=w*bf2f(vt[i]); } }                                \
        mv_bf16(ao,Ly->wo,attn,H,QD);                                                         \
        for(int i=0;i<H;i++) x[i]+=ao[i];                                                     \
        rmsnorm_gemma(h2,x,Ly->post_norm,H,c.norm_eps);                                       \
        if(Ly->is_moe){                                                                       \
            mv_f32(rlog,Ly->gate,h2,c.n_experts,H);                                           \
            for(int e=0;e<c.n_experts;e++) rlog[e]=1.0f/(1.0f+expf(-rlog[e]));                 \
            int sel[64]; float selw[64];                                                      \
            for(int a=0;a<c.n_active;a++){ int best=-1; float bv=-1e30f;                       \
                for(int e=0;e<c.n_experts;e++){ int used=0; for(int j=0;j<a;j++) if(sel[j]==e){used=1;break;} \
                    if(used)continue; float v=rlog[e]+Ly->gate_bias[e]; if(v>bv){bv=v;best=e;} } \
                sel[a]=best; selw[a]=rlog[best]; }                                             \
            float wsum=0; for(int a=0;a<c.n_active;a++) wsum+=selw[a]; if(wsum<=0)wsum=1;       \
            for(int i=0;i<H;i++) moe[i]=0;                                                     \
            for(int a=0;a<c.n_active;a++){ int e=sel[a]; float w=selw[a]/wsum*c.routed_scale;  \
                mv_bf16(exg,Ly->ex_w1[e],h2,c.moe_inter,H); mv_bf16(exu,Ly->ex_w3[e],h2,c.moe_inter,H);\
                for(int i=0;i<c.moe_inter;i++) exg[i]=swiglu_oai(exg[i],exu[i],c.swiglu_alpha,c.swiglu_limit);\
                for(int i=0;i<H;i++){ const uint16_t*wr=Ly->ex_w2[e]+(size_t)i*c.moe_inter; moe[i]+=w*vec_dot_bf16_f32(wr,exg,c.moe_inter);} } \
            mv_bf16(shg,Ly->sh_w1,h2,c.moe_inter,H); mv_bf16(shu,Ly->sh_w3,h2,c.moe_inter,H);  \
            for(int i=0;i<c.moe_inter;i++) shg[i]=swiglu_oai(shg[i],shu[i],c.swiglu_alpha,c.swiglu_limit);\
            for(int i=0;i<H;i++){ const uint16_t*wr=Ly->sh_w2+(size_t)i*c.moe_inter; x[i]+=moe[i]+vec_dot_bf16_f32(wr,shg,c.moe_inter);} \
        } else {                                                                              \
            mv_bf16(ffg,Ly->ff_g,h2,c.dense_inter,H); mv_bf16(ffu,Ly->ff_u,h2,c.dense_inter,H);\
            for(int i=0;i<c.dense_inter;i++) ffg[i]=swiglu_oai(ffg[i],ffu[i],c.swiglu_alpha,c.swiglu_limit);\
            for(int i=0;i<H;i++){ const uint16_t*wr=Ly->ff_d+(size_t)i*c.dense_inter; x[i]+=vec_dot_bf16_f32(wr,ffg,c.dense_inter);} \
        }                                                                                     \
    }                                                                                         \
    rmsnorm_gemma(h2,x,onorm,H,c.norm_eps);                                                   \
    mv_bf16(logits,head,h2,c.vocab,H);                                                        \
    { int am=0; float bv=logits[0]; for(int i=1;i<c.vocab;i++) if(logits[i]>bv){bv=logits[i];am=i;} argmax=am; }

    int nan_count=0; double xnorm=0; int argmax=0;
    /* prefill */
    double tp0=now_sec();
    for(int p=0;p<prefill;p++){
        for(int i=0;i<H;i++) x[i]=(float)(sm_next()*2.0-1.0);
        int pos=p; FORWARD_BODY
        for(int i=0;i<H;i++){ if(!(x[i]==x[i])) nan_count++; xnorm+=(double)x[i]*x[i]; }
    }
    double tp=now_sec()-tp0;
    /* decode */
    double td0=now_sec(); int last=argmax;
    for(int g=0;g<decode;g++){
        for(int i=0;i<H;i++) x[i]=(float)(sm_next()*2.0-1.0);
        int pos=prefill+g; FORWARD_BODY last=argmax;
        for(int i=0;i<H;i++){ if(!(x[i]==x[i])) nan_count++; }
    }
    double td=now_sec()-td0;

    printf("prefill: %d tok  %.1f ms/tok  %.2f tok/s\n", prefill, tp/prefill*1e3, prefill/tp);
    printf("decode:  %d tok  %.1f ms/tok  %.2f tok/s\n", decode,  td/decode *1e3, decode /td);
    printf("last argmax=%d  ||x||=%.3e  NaNs=%d\n", last, sqrt(xnorm), nan_count);
    printf("%s\n", nan_count==0 ? "OK: forward ran NaN-free" : "FAIL: NaNs present");
    return nan_count==0?0:1;
}
