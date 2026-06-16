/*
 * m3_impl.h - MiniMax-M3 (text) forward graph + synthetic allocator.
 *
 * Included by the EP runner (a64fx/m3/m3_ep_runner.c) and single-node drivers.
 * Provides: m3_arena_size, m3_alloc_synth, m3_free, m3_forward_token.
 *
 * Parallelism:
 *   - EXPERT-PARALLEL (always): expert e owned by rank e%ep_size (slot e/ep_size);
 *     routed-expert partials accumulate into s_route, one m->ar_cb per MoE layer sums
 *     them across the group (== tp_allreduce_sum).
 *   - TENSOR-PARALLEL (opt-in, M3_TP / M3_TP_*): shards the replicated dense so full
 *     dims fit at 96 nodes. The TP group == the EP group (same N ranks). Each shard is
 *     reconstructed bit-exactly by the same all-reduce:
 *       TP_ATTN   : q heads split (m3_shard_heads) -> wq owns those rows, wo owns those
 *                   input columns; per-rank partial o_proj -> ar_cb sum. kv heads + MSA
 *                   index stay replicated (small).
 *       TP_SHARED : shared-expert inter split -> partial folded into s_route (one reduce).
 *       TP_FFN    : dense-layer (0..2) FFN inter split -> partial down-proj -> ar_cb sum.
 *       TP_HEAD   : lm_head vocab split -> local (val,global-idx) argmax -> ar_argmax merge.
 *       TP_EMBED  : embed vocab split (real-weight gen only; synth seeds activations).
 *     Sharding is detected in the forward by range<full (set in m3_alloc_synth); no extra
 *     flags threaded. Every reduce makes the result identical on every rank -> lockstep.
 *
 *   - MSA (MiniMax Sparse Attention, opt-in M3_MSA, sparse layers L>=n_dense_layers):
 *     a small index projection (4 q heads / 1 k head MQA, dim 128) + QK-norm + partial
 *     RoPE scores 128-token blocks (max over block of summed index-head dots); the top-K
 *     blocks (+ init + local) are selected and the GQA attention runs only over their
 *     positions. Below the block threshold it is exactly full attention (byte-identical).
 *
 * Weights SYNTHETIC bf16; logits meaningless. Targets: memory fit, NaN-free, cross-rank
 * lockstep argmax. Offline (no uTofu) the EP+TP reconstruction can be checked single-
 * process by summing N rank-shards (see a64fx/m3/m3_tp_selftest.c).
 */
#ifndef M3_IMPL_H
#define M3_IMPL_H

#include "m3.h"
#include "m3_mxfp8.h"   /* FP8 E4M3 + E8M0 per-32 block scale matvec */
#ifdef _OPENMP
#include <omp.h>
#endif
/* parallelize a matvec/down-proj only when the output dim is large enough that the
 * fork/join barrier is amortized (small projections stay serial). */
#define M3_PAR_MIN 512

static int m3_envi(const char*k,int d){ const char*v=getenv(k); return (v&&*v)?atoi(v):d; }

/* ===================== scalar/codec helpers ===================== */
static inline uint16_t m3_f2bf(float f){
    uint32_t u; memcpy(&u,&f,4);
    if(((u>>23)&0xff)==0xff) return (uint16_t)(u>>16);
    uint32_t r=u+0x7fffu+((u>>16)&1u); return (uint16_t)(r>>16);
}
static inline float m3_bf2f(uint16_t h){ return bf16_to_f32_scalar(h); }
/* IEEE fp16 (half) KV codec: ~11-bit mantissa vs bf16's 8 -> higher KV fidelity (the model
 * computes K/V in f32, so bf16 storage already quantizes; fp16 is the higher-precision option
 * for the quality A/B vs int4). Same 2 bytes/elem as bf16. A64FX has native fp16. */
static inline uint16_t m3_f2h(float f){ __fp16 h=(__fp16)f; uint16_t u; memcpy(&u,&h,2); return u; }
static inline float    m3_h2f(uint16_t u){ __fp16 h; memcpy(&h,&u,2); return (float)h; }
/* uint16 KV path encode/decode, bf16 (default) or fp16 (M3_KV_FP16=1). */
static inline uint16_t m3_kv_enc(const m3_model*m,float f){ return m->kv_fp16 ? m3_f2h(f) : m3_f2bf(f); }
static inline float    m3_kv_dec(const m3_model*m,uint16_t u){ return m->kv_fp16 ? m3_h2f(u) : m3_bf2f(u); }

/* ===================== int4-KV codec (M3_INT4_KV, for 1M context) =====================
 * Per group of n (== head_dim) values: signed int4 (+/-7), scale = absmax/7 stored bf16.
 * ~3.9x smaller than bf16 (n/2 bytes + 1 bf16 scale vs 2n bytes). Used for the k/v/idx
 * caches; dequant happens on read in the attention/index loops (decode is latency-bound,
 * scalar matches the existing bf16 inner loops). n must be even. */
static inline uint16_t m3_q4_pack(uint8_t*restrict q, const float*v, int n){
    float amax=0; for(int i=0;i<n;i++){ float a=fabsf(v[i]); if(a>amax)amax=a; }
    float inv = amax>0 ? 7.0f/amax : 0.0f;
    for(int i=0;i<n;i+=2){
        int a=(int)lrintf(v[i]*inv);   if(a>7)a=7; else if(a<-7)a=-7;
        int b=(int)lrintf(v[i+1]*inv); if(b>7)b=7; else if(b<-7)b=-7;
        q[i>>1]=(uint8_t)((a&0xF)|((b&0xF)<<4));
    }
    return m3_f2bf(amax/7.0f);
}
static inline float m3_q4_dot(const uint8_t*restrict q, uint16_t scbf, const float*restrict x, int n){
    double d=0; for(int i=0;i<n;i+=2){ uint8_t by=q[i>>1];
        int a=(int)(int8_t)(by<<4)>>4, b=(int)(int8_t)by>>4;   /* sign-extend low/high nibble */
        d += (double)a*x[i] + (double)b*x[i+1]; }
    return (float)(d*m3_bf2f(scbf));
}
static inline void m3_q4_axpy(float*restrict out, const uint8_t*restrict q, uint16_t scbf, float w, int n){
    float ws=w*m3_bf2f(scbf);
    for(int i=0;i<n;i+=2){ uint8_t by=q[i>>1];
        int a=(int)(int8_t)(by<<4)>>4, b=(int)(int8_t)by>>4;
        out[i]+=ws*a; out[i+1]+=ws*b; }
}

static uint64_t m3_sm;
static inline double m3_sm_next(void){
    m3_sm += 0x9E3779B97F4A7C15ull; uint64_t z=m3_sm;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; z^=z>>31;
    return (double)(z>>11)/(double)(1ull<<53);
}
static inline void m3_fill_bf16(uint16_t*w,size_t n,float amp){ for(size_t i=0;i<n;i++) w[i]=m3_f2bf((float)((m3_sm_next()*2.0-1.0)*amp)); }
static inline void m3_fill_f32 (float   *w,size_t n,float amp){ for(size_t i=0;i<n;i++) w[i]=(float)((m3_sm_next()*2.0-1.0)*amp); }

static inline void m3_mv_f32(float*restrict y,const float*W,const float*x,int rows,int cols){
    for(int r=0;r<rows;r++){ const float*w=W+(size_t)r*cols; double s=0; for(int i=0;i<cols;i++) s+=(double)w[i]*x[i]; y[r]=(float)s; }
}
static inline void m3_rmsnorm_gemma(float*out,const float*x,const uint16_t*w,int n,float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float inv=(float)(1.0/sqrt(ss/n+eps)); for(int i=0;i<n;i++) out[i]=x[i]*inv*(1.0f+m3_bf2f(w[i]));
}
static inline void m3_rmsnorm_head(float*v,const uint16_t*w,int n,float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)v[i]*v[i];
    float inv=(float)(1.0/sqrt(ss/n+eps)); for(int i=0;i<n;i++) v[i]=v[i]*inv*(1.0f+m3_bf2f(w[i]));
}
/* SwiGLU-OAI (GPT-OSS "swigluoai"): glu(gate)·(up+1), gate clamped <=lim, up clamped
 * to [-lim,lim]. glu(g)=g·sigmoid(alpha·g). The (up+1) term is part of the OAI variant. */
static inline float m3_swiglu_oai(float gate,float up,float alpha,float lim){
    float g=gate>lim?lim:gate; float u=up>lim?lim:(up<-lim?-lim:up);
    float s=g/(1.0f+expf(-alpha*g)); return s*(u+1.0f);
}
static inline void m3_rope_head(float*v,const float*cosp,const float*sinp,int rotary_dim){
    int half=rotary_dim/2;
    for(int k=0;k<half;k++){ float c=cosp[k],s=sinp[k],a=v[k],b=v[k+half]; v[k]=a*c-b*s; v[k+half]=a*s+b*c; }
}

/* ===================== pinned spin pool (EXPERIMENTAL, M3_POOL=1, default OFF) ========
 * Intent: beat the OpenMP 48-thread cross-CMG fork-join regression with a persistent
 * pinned spin pool (pin within the job cpuset = cores 12-59 on A64FX; dispatch via an
 * atomic seq, release/acquire ordering; disjoint row-slices -> bit-identical to serial).
 * STATUS (2026-06-15): NOT working — node-dependent hangs and no speedup even when it
 * runs (OpenMP-12 stays the practical best, ~7x). Root cause undiagnosed (instrumentation
 * hung). Reaching full-48 likely needs NUMA-interleaved weights (each CMG reads local
 * HBM) + pool-stability work. Default is OpenMP (m3_g_pool stays NULL unless M3_POOL=1). */
#define M3_POOL_MAXT 64
struct m3_pool {   /* m3.h forward-declares `typedef struct m3_pool m3_pool;` */
    void (*fn)(void*,int,int); void *arg; int nthr;
    pthread_t th[M3_POOL_MAXT]; int core[M3_POOL_MAXT];
    _Atomic long seq; _Atomic long wdone[M3_POOL_MAXT]; volatile int stop;
};
static m3_pool *m3_g_pool=NULL;   /* one model per process; matvecs dispatch here */
static int m3_pool_dbg=0;
/* M3_DUMMY: idealized-compute ceiling. matvecs STREAM the weight bytes at full HBM BW
 * (touch every cache line, no bf16->f32 widening, no FMA) -> models the actual memory
 * read with compute removed; comm (ar_cb) + dispatch stay real. Reveals the
 * comm+mem+dispatch floor = the practical tok/s ceiling if matvec were BW-perfect. */
static int m3_dummy=0;
static _Atomic long m3_dbg_disp=0, m3_dbg_main=0, m3_dbg_wrk=0;

static inline void m3_cpu_relax(void){ __asm__ __volatile__("yield":::"memory"); }
static void m3_pin(int core){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(core,&s); sched_setaffinity(0,sizeof s,&s); }
typedef struct { m3_pool *p; int tid; } m3_wctx;
static void* m3_worker(void *a){
    m3_wctx *w=a; m3_pool *p=w->p; int tid=w->tid; m3_pin(p->core[tid]); long last=0;
    if(m3_pool_dbg) fprintf(stderr,"[pool] worker %d started on core %d\n",tid,p->core[tid]);
    for(;;){
        while(atomic_load_explicit(&p->seq,memory_order_acquire)==last && !p->stop) m3_cpu_relax();
        if(p->stop) break; last=atomic_load_explicit(&p->seq,memory_order_acquire);
        p->fn(p->arg,tid,p->nthr);
        atomic_store_explicit(&p->wdone[tid],last,memory_order_release);
    }
    return NULL;
}
static m3_pool* m3_pool_create(int nthr){
    if(nthr<1)nthr=1; if(nthr>M3_POOL_MAXT)nthr=M3_POOL_MAXT;
    { const char*e=getenv("M3_POOL_DEBUG"); m3_pool_dbg=(e&&*e)?atoi(e):0; }
    /* pin within the job's actual cpuset (Fugaku: cores 12-59, not 0-N); cpus[] are
     * the allowed cores in ascending order -> tid t pins to cpus[t]. On A64FX the 48
     * compute cores group as 12 per CMG, so ascending order fills CMG0 first. */
    cpu_set_t allowed; CPU_ZERO(&allowed);
    int cpus[512], nc=0;
    if(sched_getaffinity(0,sizeof allowed,&allowed)==0)
        for(int c=0;c<512 && nc<M3_POOL_MAXT;c++) if(CPU_ISSET(c,&allowed)) cpus[nc++]=c;
    if(nc<1){ for(int c=0;c<M3_POOL_MAXT;c++) cpus[c]=c; nc=M3_POOL_MAXT; }
    if(nthr>nc) nthr=nc;
    m3_pool *p=calloc(1,sizeof *p); p->nthr=nthr; atomic_store(&p->seq,0); p->stop=0;
    for(int t=0;t<nthr;t++){ p->core[t]=cpus[t]; atomic_store(&p->wdone[t],0); }
    m3_pin(p->core[0]);
    for(int t=1;t<nthr;t++){ m3_wctx *w=malloc(sizeof *w); w->p=p; w->tid=t;
        int rc=pthread_create(&p->th[t],NULL,m3_worker,w);
        if(m3_pool_dbg) fprintf(stderr,"[pool] create worker %d rc=%d\n",t,rc); }
    if(m3_pool_dbg) fprintf(stderr,"[pool] created nthr=%d\n",nthr);
    return p;
}
static void m3_pool_run(m3_pool *p, void(*fn)(void*,int,int), void *arg){
    if(!p || p->nthr<=1){ fn(arg,0,1); return; }
    if(m3_pool_dbg) atomic_fetch_add_explicit(&m3_dbg_disp,1,memory_order_relaxed);
    p->fn=fn; p->arg=arg;
    long s=atomic_load_explicit(&p->seq,memory_order_relaxed)+1;
    atomic_store_explicit(&p->seq,s,memory_order_release);   /* dispatch (publishes fn/arg) */
    fn(arg,0,p->nthr);                                       /* main = tid 0 */
    for(int t=1;t<p->nthr;t++) while(atomic_load_explicit(&p->wdone[t],memory_order_acquire)<s) m3_cpu_relax();
}
static void m3_pool_destroy(m3_pool *p){
    if(!p) return;
    if(m3_pool_dbg) fprintf(stderr,"[pool] nthr=%d dispatches=%ld main_blocks=%ld worker_blocks=%ld\n",
                            p->nthr,(long)m3_dbg_disp,(long)m3_dbg_main,(long)m3_dbg_wrk);
    p->stop=1; atomic_fetch_add(&p->seq,1);
    for(int t=1;t<p->nthr;t++) pthread_join(p->th[t],NULL);
    free(p);
}

/* matvec worker: y[rows] = W[rows,cols](bf16) . x[cols], rows partitioned by 8-blocks */
typedef struct { float *y; const uint16_t *W; const float *x; int rows, cols; } m3_mvjob;
static void m3_mv_worker(void *a,int tid,int nthr){
    m3_mvjob *j=a; int nb=j->rows/8, per=(nb+nthr-1)/nthr, b0=tid*per, b1=b0+per; if(b1>nb)b1=nb;
    for(int bi=b0;bi<b1;bi++){ int r=bi*8; const uint16_t *b=j->W+(size_t)r*j->cols;
        matvec_bf16_8row(j->y+r,b,b+j->cols,b+2*(size_t)j->cols,b+3*(size_t)j->cols,
                         b+4*(size_t)j->cols,b+5*(size_t)j->cols,b+6*(size_t)j->cols,b+7*(size_t)j->cols,j->x,j->cols); }
    if(tid==0) for(int r=nb*8;r<j->rows;r++) j->y[r]=vec_dot_bf16_f32(j->W+(size_t)r*j->cols,j->x,j->cols);
    if(m3_pool_dbg) atomic_fetch_add_explicit(tid==0?&m3_dbg_main:&m3_dbg_wrk,(long)(b1-b0),memory_order_relaxed);
}

/* y[rows] = W[rows,cols](bf16) . x[cols] over 8-row blocks (disjoint outputs -> bit-
 * identical to serial). Default parallelism is OpenMP (validated ~7x @12 threads = 1 CMG;
 * the cross-CMG fork-join wall caps it there). The experimental pinned pool (M3_POOL=1)
 * is selected when m3_g_pool is set. */
static void m3_mv_bf16(float*restrict y,const uint16_t*W,const float*x,int rows,int cols){
    if(m3_dummy){   /* stream every cache line of W at full BW; no widen/FMA (compute idealized) */
        double acc=0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static) if(rows>=M3_PAR_MIN)
#endif
        for(int r=0;r<rows;r++){ const uint16_t*w=W+(size_t)r*cols; uint32_t s=0; for(int i=0;i<cols;i+=32) s+=w[i]; acc+=s; }
        float v=(float)(((long)acc)&1)*1e-30f; for(int r=0;r<rows;r++) y[r]=v;   /* defeat DCE, ~0 */
        return;
    }
    if(m3_g_pool && rows>=M3_PAR_MIN){ m3_mvjob j={y,(const uint16_t*)W,x,rows,cols}; m3_pool_run(m3_g_pool,m3_mv_worker,&j); return; }
    int nb=rows/8;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows>=M3_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){ int r=bi*8; const uint16_t*b=W+(size_t)r*cols;
        matvec_bf16_8row(y+r,b,b+cols,b+2*(size_t)cols,b+3*(size_t)cols,
                         b+4*(size_t)cols,b+5*(size_t)cols,b+6*(size_t)cols,b+7*(size_t)cols,x,cols); }
    for(int r=nb*8;r<rows;r++) y[r]=vec_dot_bf16_f32(W+(size_t)r*cols,x,cols);
}

/* MXFP8 matvec: y[rows] = decode(W fp8 + S e8m0) . x[cols], 8-row blocks via the kernel. */
static void m3_mv_mxfp8(m3_model*m, float*restrict y, const uint8_t*W, const uint8_t*S, const float*x, int rows, int cols){
    const uint32_t*lut=m->fp8_lut; int sb=cols/M3_MX_BLK, nb=rows/8;
    if(m3_dummy){ double acc=0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static) if(rows>=M3_PAR_MIN)
#endif
        for(int r=0;r<rows;r++){ const uint8_t*w=W+(size_t)r*cols; uint32_t s=0; for(int i=0;i<cols;i+=64) s+=w[i]; acc+=s; }
        float v=(float)(((long)acc)&1)*1e-30f; for(int r=0;r<rows;r++) y[r]=v; return; }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(rows>=M3_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){ int r=bi*8; const uint8_t*w=W+(size_t)r*cols,*s=S+(size_t)r*sb;
        m3_matvec_mxfp8_8row(y+r, w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
                             s,s+sb,s+2*sb,s+3*sb,s+4*sb,s+5*sb,s+6*sb,s+7*sb, x, cols, lut); }
    for(int r=nb*8;r<rows;r++){ const uint8_t*w=W+(size_t)r*cols,*s=S+(size_t)r*sb; double a=0;
        for(int b=0;b<cols;b+=M3_MX_BLK){ float sc=m3_e8m0(s[b/M3_MX_BLK]); int e=b+M3_MX_BLK<cols?b+M3_MX_BLK:cols;
            for(int c=b;c<e;c++){ float wf; uint32_t u=lut[w[c]]; memcpy(&wf,&u,4); a+=(double)wf*sc*x[c]; } } y[r]=(float)a; }
}
/* matvec dispatch by weight type (bf16 or MXFP8) */
static void m3_mv(m3_model*m, float*restrict y, const m3_tensor*t, const float*x, int rows, int cols){
    if(t->type==M3_MXFP8) m3_mv_mxfp8(m,y,(const uint8_t*)t->w,t->scale,x,rows,cols);
    else m3_mv_bf16(y,(const uint16_t*)t->w,x,rows,cols);
}

/* ===================== arena fit estimate (synth uses malloc) ===================== */
static size_t m3_arena_size(const m3_config*c,int ep_rank,int ep_size){
    int H=c->hidden, QD=m3_q_dim(c), KVD=m3_kv_dim(c);
    int no=m3_n_owned(c->n_experts,ep_rank,ep_size);
    int qh0,qh1; m3_shard_heads(c->n_heads,ep_rank,ep_size,&qh0,&qh1);
    int tp=m3_envi("M3_TP",0);
    int tp_attn=m3_envi("M3_TP_ATTN",tp), tp_sh=m3_envi("M3_TP_SHARED",tp);
    int tp_ffn=m3_envi("M3_TP_FFN",tp), tp_head=m3_envi("M3_TP_HEAD",tp), tp_emb=m3_envi("M3_TP_EMBED",tp);
    int qrows = tp_attn ? (qh1-qh0)*c->head_dim : QD;
    int shrows = tp_sh ? (c->moe_inter+ep_size-1)/ep_size : c->moe_inter;
    int ffrows = tp_ffn ? (c->dense_inter+ep_size-1)/ep_size : c->dense_inter;
    int hrows = tp_head ? (c->vocab+ep_size-1)/ep_size : c->vocab;
    int erows = tp_emb ? (c->vocab+ep_size-1)/ep_size : c->vocab;
    size_t attn = (size_t)qrows*H*2 + 2*(size_t)KVD*H*2 + (size_t)H*qrows*2 + 2*(size_t)H*2 + 2*(size_t)c->head_dim*2;
    attn += 2*(size_t)c->max_pos*KVD*2;
    size_t msa = (size_t)m3_idx_q_dim(c)*H*2 + (size_t)c->msa_index_dim*H*2 + 2*(size_t)c->msa_index_dim*2 + (size_t)c->max_pos*c->msa_index_dim*2;
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
static void m3_free_mstream(m3_model*m);
static void m3_free(m3_model*m){
    if(!m) return;
    if(m->ms) m3_free_mstream(m);
    if(m->pool){ if(m3_g_pool==m->pool) m3_g_pool=NULL; m3_pool_destroy(m->pool); m->pool=NULL; }
    for(int l=0;l<m->cfg.n_layers;l++){
        m3_layer*L=&m->layers[l];
        free(L->input_norm);free(L->post_norm);
        free(L->wq.w);free(L->wk.w);free(L->wv.w);free(L->wo.w);free(L->q_norm);free(L->k_norm);
        free(L->k_cache);free(L->v_cache);
        free(L->k_q4);free(L->v_q4);free(L->k_qs);free(L->v_qs);free(L->idx_q4);free(L->idx_qs);
        if(m3_is_moe(&m->cfg,l)){
            free(L->idx_wq.w);free(L->idx_wk.w);free(L->idx_q_norm);free(L->idx_k_norm);free(L->idx_k_cache);
            free(L->gate.w);free(L->gate_bias);free(L->sh_w1.w);free(L->sh_w3.w);free(L->sh_w2.w);
            if(L->ex_w1){ for(int s=0;s<L->n_owned;s++){ free(L->ex_w1[s].w);free(L->ex_w3[s].w);free(L->ex_w2[s].w);} }
            free(L->ex_w1);free(L->ex_w3);free(L->ex_w2);free(L->owned_eid);
        } else { free(L->ff_gate.w);free(L->ff_up.w);free(L->ff_down.w); }
    }
    free(m->layers);free(m->embed);free(m->head.w);free(m->out_norm);free(m->rope_cos);free(m->rope_sin);
    free(m->s_norm);free(m->s_q);free(m->s_k);free(m->s_v);free(m->s_attn);free(m->s_o);
    free(m->s_idx_q);free(m->s_idx_k);free(m->s_blk_score);free(m->s_blk_sel);
    free(m->s_router);free(m->s_shg);free(m->s_shu);free(m->s_sh);free(m->s_moe);
    free(m->s_exg);free(m->s_exu);free(m->s_route);free(m->s_ff_g);free(m->s_ff_u);free(m->s_ff);free(m->s_logits);
    free(m);
}

/* init int4-KV / CP flags from env (call once after ep_rank/ep_size set, before m3_alloc_kv). */
static void m3_kv_init(m3_model*m){
    const m3_config*c=&m->cfg;
    m->int4_kv =m3_envi("M3_INT4_KV",0);       /* int4 KV (independent of CP; both for 1M) */
    m->kv_fp16 =m3_envi("M3_KV_FP16",0);        /* uint16 KV path: fp16 (1) vs bf16 (0) */
    m->cp_on   =m3_envi("M3_CP",0) && m->ep_size>1;
    m->cp_block=c->msa_block_size;             /* CP shard granularity == MSA block (128) */
    m->cp_nslot=m3_cp_nslot(c->max_pos,m->cp_block,m->ep_size,m->cp_on);
}
/* allocate this layer's KV cache (bf16 or int4, sized to cp_nslot owned slots). */
static void m3_alloc_kv(m3_model*m,m3_layer*L,int is_moe,size_t*used){
    const m3_config*c=&m->cfg; int KVD=m3_kv_dim(c), KVH=c->n_kv_heads, ID=c->msa_index_dim;
    size_t ns=(size_t)m->cp_nslot;
    if(m->int4_kv){
        L->k_q4=calloc(ns*(KVD/2),1); L->v_q4=calloc(ns*(KVD/2),1);
        L->k_qs=calloc(ns*KVH,2);     L->v_qs=calloc(ns*KVH,2);
        *used += 2*(ns*(KVD/2)+ns*KVH*2);
        if(is_moe){ L->idx_q4=calloc(ns*(ID/2),1); L->idx_qs=calloc(ns,2); *used += ns*(ID/2)+ns*2; }
    } else {
        L->k_cache=calloc(ns*KVD,2); L->v_cache=calloc(ns*KVD,2); *used += 2*ns*KVD*2;
        if(is_moe){ L->idx_k_cache=calloc(ns*ID,2); *used += ns*ID*2; }
    }
}

static m3_model* m3_alloc_synth(m3_config cfg,int ep_rank,int ep_size,int n_threads,int n_cmgs){
    m3_model*m=calloc(1,sizeof(m3_model)); if(!m) return NULL;
    m->cfg=cfg; m->ep_rank=ep_rank; m->ep_size=ep_size; m->n_threads=n_threads; m->n_cmgs=n_cmgs;
    m->bf16_mv_qt=M3_BF16; m3_kv_init(m);
    const int H=cfg.hidden, QD=m3_q_dim(&cfg), KVD=m3_kv_dim(&cfg), HD=cfg.head_dim;
    const int IQD=m3_idx_q_dim(&cfg), ID=cfg.msa_index_dim, half=cfg.rotary_dim/2;
    /* TP flags (TP_ATTN forced off under CP: each rank needs all query heads to merge position-shards) */
    int tp=m3_envi("M3_TP",0);
    int tp_attn=m3_envi("M3_TP_ATTN",tp) && !m->cp_on; int tp_sh=m3_envi("M3_TP_SHARED",tp);
    int tp_ffn=m3_envi("M3_TP_FFN",tp), tp_head=m3_envi("M3_TP_HEAD",tp), tp_emb=m3_envi("M3_TP_EMBED",tp);
    int qh0,qh1; if(tp_attn) m3_shard_heads(cfg.n_heads,ep_rank,ep_size,&qh0,&qh1); else { qh0=0; qh1=cfg.n_heads; }
    int qrows=(qh1-qh0)*HD;
    int sh_r0,sh_rows; if(tp_sh) m3_shard(cfg.moe_inter,ep_rank,ep_size,&sh_r0,&sh_rows); else { sh_r0=0; sh_rows=cfg.moe_inter; }
    int ff_r0,ff_rows; if(tp_ffn) m3_shard(cfg.dense_inter,ep_rank,ep_size,&ff_r0,&ff_rows); else { ff_r0=0; ff_rows=cfg.dense_inter; }
    int hr0,hrows; if(tp_head) m3_shard(cfg.vocab,ep_rank,ep_size,&hr0,&hrows); else { hr0=0; hrows=cfg.vocab; }
    int er0,erows; if(tp_emb) m3_shard(cfg.vocab,ep_rank,ep_size,&er0,&erows); else { er0=0; erows=cfg.vocab; }
    size_t used=0;
    #define BF(p,n,amp) do{ (p)=malloc((size_t)(n)*2); m3_fill_bf16((p),(n),(amp)); used+=(size_t)(n)*2; }while(0)
    #define FZ(p,n,amp) do{ (p)=malloc((size_t)(n)*4); m3_fill_f32 ((p),(n),(amp)); used+=(size_t)(n)*4; }while(0)

    m->rope_cos=malloc((size_t)cfg.max_pos*half*4); m->rope_sin=malloc((size_t)cfg.max_pos*half*4);
    for(int p=0;p<cfg.max_pos;p++) for(int k=0;k<half;k++){
        double invf=pow((double)cfg.rope_theta,-2.0*k/(double)cfg.rotary_dim); double a=p*invf;
        m->rope_cos[(size_t)p*half+k]=(float)cos(a); m->rope_sin[(size_t)p*half+k]=(float)sin(a); }

    uint16_t*eb; BF(eb,(size_t)erows*H,0.05f); m->embed=eb; m->emb_r0=er0; m->emb_rows=erows;
    uint16_t*hd; BF(hd,(size_t)hrows*H,0.05f); m->head.w=hd; m->head.type=M3_BF16; m->head.rows=hrows; m->head.cols=H; m->head_r0=hr0;
    BF(m->out_norm,H,0.1f);

    m->layers=calloc(cfg.n_layers,sizeof(m3_layer));
    for(int l=0;l<cfg.n_layers;l++){
        m3_layer*L=&m->layers[l]; int is_moe=m3_is_moe(&cfg,l);
        BF(L->input_norm,H,0.1f); BF(L->post_norm,H,0.1f);
        uint16_t*p;
        BF(p,(size_t)qrows*H,0.03f); L->wq=(m3_tensor){p,NULL,M3_BF16,qrows,H};
        BF(p,(size_t)KVD*H,0.03f);   L->wk=(m3_tensor){p,NULL,M3_BF16,KVD,H};
        BF(p,(size_t)KVD*H,0.03f);   L->wv=(m3_tensor){p,NULL,M3_BF16,KVD,H};
        BF(p,(size_t)H*qrows,0.03f); L->wo=(m3_tensor){p,NULL,M3_BF16,H,qrows};
        BF(L->q_norm,HD,0.1f); BF(L->k_norm,HD,0.1f);
        L->qh0=qh0; L->qh1=qh1;
        m3_alloc_kv(m,L,is_moe,&used);
        if(is_moe){
            /* MSA indexer (replicated) */
            BF(p,(size_t)IQD*H,0.03f); L->idx_wq=(m3_tensor){p,NULL,M3_BF16,IQD,H};
            BF(p,(size_t)ID*H,0.03f);  L->idx_wk=(m3_tensor){p,NULL,M3_BF16,ID,H};
            BF(L->idx_q_norm,ID,0.1f); BF(L->idx_k_norm,ID,0.1f);
            /* MoE */
            float*g; FZ(g,(size_t)cfg.n_experts*H,0.03f); L->gate=(m3_tensor){g,NULL,M3_F32,cfg.n_experts,H};
            FZ(L->gate_bias,cfg.n_experts,0.1f);
            BF(p,(size_t)sh_rows*H,0.03f); L->sh_w1=(m3_tensor){p,NULL,M3_BF16,sh_rows,H};
            BF(p,(size_t)sh_rows*H,0.03f); L->sh_w3=(m3_tensor){p,NULL,M3_BF16,sh_rows,H};
            BF(p,(size_t)H*sh_rows,0.03f); L->sh_w2=(m3_tensor){p,NULL,M3_BF16,H,sh_rows};
            L->sh_r0=sh_r0; L->sh_rows=sh_rows;
            int no=m3_n_owned(cfg.n_experts,ep_rank,ep_size);
            L->n_owned=no; L->owned_eid=malloc((size_t)(no>0?no:1)*sizeof(int));
            L->ex_w1=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor)); L->ex_w3=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor)); L->ex_w2=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor));
            int s=0; for(int e=0;e<cfg.n_experts;e++) if(e%ep_size==ep_rank){ L->owned_eid[s]=e;
                BF(p,(size_t)cfg.moe_inter*H,0.03f); L->ex_w1[s]=(m3_tensor){p,NULL,M3_BF16,cfg.moe_inter,H};
                BF(p,(size_t)cfg.moe_inter*H,0.03f); L->ex_w3[s]=(m3_tensor){p,NULL,M3_BF16,cfg.moe_inter,H};
                BF(p,(size_t)H*cfg.moe_inter,0.03f); L->ex_w2[s]=(m3_tensor){p,NULL,M3_BF16,H,cfg.moe_inter}; s++; }
        } else {
            BF(p,(size_t)ff_rows*H,0.03f); L->ff_gate=(m3_tensor){p,NULL,M3_BF16,ff_rows,H};
            BF(p,(size_t)ff_rows*H,0.03f); L->ff_up  =(m3_tensor){p,NULL,M3_BF16,ff_rows,H};
            BF(p,(size_t)H*ff_rows,0.03f); L->ff_down=(m3_tensor){p,NULL,M3_BF16,H,ff_rows};
            L->ff_r0=ff_r0; L->ff_rows=ff_rows;
        }
    }
    m->s_norm=malloc(H*4); m->s_q=malloc(QD*4); m->s_k=malloc(KVD*4); m->s_v=malloc(KVD*4);
    m->s_attn=malloc(QD*4); m->s_o=malloc(H*4);
    m->s_idx_q=malloc((size_t)IQD*4); m->s_idx_k=malloc((size_t)ID*4);
    m->s_blk_score=malloc((size_t)cfg.max_pos*4); m->s_blk_sel=malloc((size_t)cfg.max_pos*sizeof(int));
    m->s_router=malloc((size_t)cfg.n_experts*4); m->s_shg=malloc(cfg.moe_inter*4); m->s_shu=malloc(cfg.moe_inter*4);
    m->s_sh=malloc(H*4); m->s_moe=malloc(H*4); m->s_exg=malloc(cfg.moe_inter*4); m->s_exu=malloc(cfg.moe_inter*4); m->s_route=malloc(H*4);
    m->s_ff_g=malloc(cfg.dense_inter*4); m->s_ff_u=malloc(cfg.dense_inter*4); m->s_ff=malloc(H*4);
    m->s_logits=malloc((size_t)hrows*4);
    m->arena_used=used; m->arena_sz=used;
    m3_init_fp8_lut(m->fp8_lut);
    m3_dummy=m3_envi("M3_DUMMY",0);
    if(m3_envi("M3_POOL",0)) m->pool=m3_g_pool=m3_pool_create(m->n_threads);  /* experimental pinned pool; default OpenMP */
    return m;
    #undef BF
    #undef FZ
}

/* ===================== real-weight loader (staged blob + manifest) ===================== */
#include <sys/mman.h>
typedef struct { char name[300]; uint64_t off; size_t nbytes; int f32; int nd; long shape[5]; } m3_ent;
static m3_ent* m3_find(m3_ent*es,int n,const char*nm){ for(int i=0;i<n;i++) if(!strcmp(es[i].name,nm)) return &es[i]; return NULL; }
static m3_ent* m3_req(m3_ent*es,int n,const char*nm){ m3_ent*e=m3_find(es,n,nm); if(!e) fprintf(stderr,"m3_load: MISSING tensor %s\n",nm); return e; }
/* copy helpers from the blob mmap (base); return malloc'd buffer (caller frees). */
static void* m3_cp_full(const uint8_t*base,const m3_ent*e){ void*d=malloc(e->nbytes); memcpy(d,base+e->off,e->nbytes); return d; }
static void* m3_cp_rows(const uint8_t*base,const m3_ent*e,int r0,int nrows,int cols,int esz){
    size_t rb=(size_t)cols*esz; void*d=malloc((size_t)nrows*rb); memcpy(d,base+e->off+(size_t)r0*rb,(size_t)nrows*rb); return d; }
static void* m3_cp_cols(const uint8_t*base,const m3_ent*e,int Rtot,int c0,int ncols,int Ctot,int esz){
    void*d=malloc((size_t)Rtot*ncols*esz); uint8_t*dp=d; const uint8_t*sp=base+e->off;
    for(int r=0;r<Rtot;r++) memcpy(dp+(size_t)r*ncols*esz, sp+((size_t)r*Ctot+c0)*esz, (size_t)ncols*esz); return d; }

/* Load weight `name` (bf16, or MXFP8 if a `name_scale_inv` companion exists) with TP slicing.
 * mode 0 = full [Rtot,Ctot]; 1 = row-shard rows[r0,r0+nr) of [Rtot,Ctot]; 2 = col-shard
 * cols[c0,c0+nc) of [Rtot,Ctot]. MXFP8: FP8 weight (1 B) + E8M0 scale (1 B, Ctot/32 per row);
 * col-shard requires c0,nc multiples of 32 (true for head_dim 128; ep_size=1 uses full). */
static m3_tensor m3_load_w(m3_ent*es,int n,const uint8_t*base,const char*name,int mode,
                           int r0,int nr,int c0,int nc,int Rtot,int Ctot,int*ok,size_t*used){
    char sn[416]; snprintf(sn,sizeof sn,"%s_scale_inv",name);
    m3_ent*we=m3_find(es,n,name); m3_ent*se=m3_find(es,n,sn);
    m3_tensor t; t.w=NULL; t.scale=NULL; t.type=M3_BF16; t.rows=0; t.cols=0;
    if(!we){ fprintf(stderr,"m3_load: MISSING %s\n",name); *ok=0; return t; }
    int esz = se ? 1 : 2;  t.type = se ? M3_MXFP8 : M3_BF16;
    if(mode==0){ t.w=m3_cp_full(base,we); t.rows=Rtot; t.cols=Ctot; *used+=we->nbytes; }
    else if(mode==1){ t.w=m3_cp_rows(base,we,r0,nr,Ctot,esz); t.rows=nr; t.cols=Ctot; *used+=(size_t)nr*Ctot*esz; }
    else { t.w=m3_cp_cols(base,we,Rtot,c0,nc,Ctot,esz); t.rows=Rtot; t.cols=nc; *used+=(size_t)Rtot*nc*esz; }
    if(se){ int sc=Ctot/M3_MX_BLK;
        if(mode==0) t.scale=(uint8_t*)m3_cp_full(base,se);
        else if(mode==1) t.scale=(uint8_t*)m3_cp_rows(base,se,r0,nr,sc,1);
        else t.scale=(uint8_t*)m3_cp_cols(base,se,Rtot,c0/M3_MX_BLK,nc/M3_MX_BLK,sc,1); }
    return t;
}

/* Build an m3_model from this rank's staged blob (M3_STAGE_DIR/rank<rr>.{blob,manifest}).
 * Dense tensors are TP-sliced into the arena per the same ranges as m3_alloc_synth; routed
 * experts are the owned ones in the blob. Returns NULL on any missing/short tensor. */
static m3_model* m3_load_real(m3_config cfg,int ep_rank,int ep_size,const char*blob_dir,int n_threads,int n_cmgs){
    char bdir[1024]; if(blob_dir&&*blob_dir) snprintf(bdir,sizeof bdir,"%s",blob_dir);
    else { const char*e=getenv("M3_STAGE_DIR"); snprintf(bdir,sizeof bdir,"%s",(e&&*e)?e:"/local/m3"); }
    char bp[1100],mp[1100]; snprintf(bp,sizeof bp,"%s/rank%02d.blob",bdir,ep_rank); snprintf(mp,sizeof mp,"%s/rank%02d.manifest",bdir,ep_rank);
    FILE*mf=fopen(mp,"r"); if(!mf){ fprintf(stderr,"m3_load: cannot open %s\n",mp); return NULL; }
    char line[1024]; int cap=4096,n=0; m3_ent*es=malloc((size_t)cap*sizeof(m3_ent));
    while(fgets(line,sizeof line,mf)){
        if(line[0]=='#'||line[0]=='\n') continue;
        if(n>=cap){ cap*=2; es=realloc(es,(size_t)cap*sizeof(m3_ent)); }
        m3_ent*e=&es[n]; char dt[32]; int pos=0,cnt;
        if(sscanf(line,"%llu %zu %31s %d%n",(unsigned long long*)&e->off,&e->nbytes,dt,&e->nd,&cnt)!=4) continue;
        pos=cnt; e->f32=(dt[0]=='F'&&dt[1]=='3'); for(int d=0;d<e->nd&&d<5;d++){ long v; int c2; sscanf(line+pos," %ld%n",&v,&c2); e->shape[d]=v; pos+=c2; }
        while(line[pos]==' ')pos++; char*nl=strchr(line+pos,'\n'); if(nl)*nl=0; snprintf(e->name,sizeof e->name,"%s",line+pos); n++;
    }
    fclose(mf);
    int bfd=open(bp,O_RDONLY); if(bfd<0){ fprintf(stderr,"m3_load: cannot open %s\n",bp); free(es); return NULL; }
    struct stat sb; fstat(bfd,&sb); size_t bsz=sb.st_size;
    const uint8_t*base=mmap(NULL,bsz,PROT_READ,MAP_PRIVATE,bfd,0);
    if(base==MAP_FAILED){ fprintf(stderr,"m3_load: mmap failed\n"); close(bfd); free(es); return NULL; }

    m3_model*m=calloc(1,sizeof(m3_model)); m->cfg=cfg; m->ep_rank=ep_rank; m->ep_size=ep_size; m->n_threads=n_threads; m->n_cmgs=n_cmgs; m->bf16_mv_qt=M3_BF16; m3_kv_init(m);
    const int H=cfg.hidden, QD=m3_q_dim(&cfg), KVD=m3_kv_dim(&cfg), HD=cfg.head_dim;
    const int IQD=m3_idx_q_dim(&cfg), ID=cfg.msa_index_dim, half=cfg.rotary_dim/2;
    int tp=m3_envi("M3_TP",0);
    int tp_attn=m3_envi("M3_TP_ATTN",tp) && !m->cp_on; int tp_sh=m3_envi("M3_TP_SHARED",tp),tp_ffn=m3_envi("M3_TP_FFN",tp),tp_head=m3_envi("M3_TP_HEAD",tp),tp_emb=m3_envi("M3_TP_EMBED",tp);
    int qh0,qh1; if(tp_attn) m3_shard_heads(cfg.n_heads,ep_rank,ep_size,&qh0,&qh1); else { qh0=0; qh1=cfg.n_heads; } int qrows=(qh1-qh0)*HD;
    int sh_r0,sh_rows; if(tp_sh) m3_shard(cfg.moe_inter,ep_rank,ep_size,&sh_r0,&sh_rows); else { sh_r0=0; sh_rows=cfg.moe_inter; }
    int ff_r0,ff_rows; if(tp_ffn) m3_shard(cfg.dense_inter,ep_rank,ep_size,&ff_r0,&ff_rows); else { ff_r0=0; ff_rows=cfg.dense_inter; }
    int hr0,hrows; if(tp_head) m3_shard(cfg.vocab,ep_rank,ep_size,&hr0,&hrows); else { hr0=0; hrows=cfg.vocab; }
    int er0,erows; if(tp_emb) m3_shard(cfg.vocab,ep_rank,ep_size,&er0,&erows); else { er0=0; erows=cfg.vocab; }
    size_t used=0; int ok=1;
    #define REQ(nm) ({ m3_ent*_e=m3_req(es,n,(nm)); if(!_e){ ok=0; } _e; })
    char nb[512];

    m->rope_cos=malloc((size_t)cfg.max_pos*half*4); m->rope_sin=malloc((size_t)cfg.max_pos*half*4);
    for(int p=0;p<cfg.max_pos;p++) for(int k=0;k<half;k++){ double invf=pow((double)cfg.rope_theta,-2.0*k/(double)cfg.rotary_dim),a=p*invf;
        m->rope_cos[(size_t)p*half+k]=(float)cos(a); m->rope_sin[(size_t)p*half+k]=(float)sin(a); }

    { m3_ent*e=REQ("language_model.model.embed_tokens.weight"); if(e){ m->embed=m3_cp_rows(base,e,er0,erows,H,2); used+=(size_t)erows*H*2; } m->emb_r0=er0; m->emb_rows=erows; }
    { m3_ent*e=REQ("language_model.lm_head.weight"); if(e){ m->head.w=m3_cp_rows(base,e,hr0,hrows,H,2); used+=(size_t)hrows*H*2; } m->head.type=M3_BF16; m->head.rows=hrows; m->head.cols=H; m->head_r0=hr0; }
    { m3_ent*e=REQ("language_model.model.norm.weight"); if(e) m->out_norm=m3_cp_full(base,e); }

    m->layers=calloc(cfg.n_layers,sizeof(m3_layer));
    for(int l=0;l<cfg.n_layers&&ok;l++){
        m3_layer*L=&m->layers[l]; int is_moe=m3_is_moe(&cfg,l);
        #define LN(suf) (snprintf(nb,sizeof nb,"language_model.model.layers.%d." suf,l),nb)
        { m3_ent*e=REQ(LN("input_layernorm.weight")); if(e) L->input_norm=m3_cp_full(base,e); }
        { m3_ent*e=REQ(LN("post_attention_layernorm.weight")); if(e) L->post_norm=m3_cp_full(base,e); }
        L->wq=m3_load_w(es,n,base,LN("self_attn.q_proj.weight"),1,qh0*HD,qrows,0,0,QD,H,&ok,&used);
        L->wk=m3_load_w(es,n,base,LN("self_attn.k_proj.weight"),0,0,0,0,0,KVD,H,&ok,&used);
        L->wv=m3_load_w(es,n,base,LN("self_attn.v_proj.weight"),0,0,0,0,0,KVD,H,&ok,&used);
        L->wo=m3_load_w(es,n,base,LN("self_attn.o_proj.weight"),2,0,0,qh0*HD,qrows,H,QD,&ok,&used);
        { m3_ent*e=REQ(LN("self_attn.q_norm.weight")); if(e) L->q_norm=m3_cp_full(base,e); }
        { m3_ent*e=REQ(LN("self_attn.k_norm.weight")); if(e) L->k_norm=m3_cp_full(base,e); }
        L->qh0=qh0; L->qh1=qh1;
        m3_alloc_kv(m,L,is_moe,&used);
        if(is_moe){
            L->idx_wq=m3_load_w(es,n,base,LN("self_attn.index_q_proj.weight"),0,0,0,0,0,IQD,H,&ok,&used);
            L->idx_wk=m3_load_w(es,n,base,LN("self_attn.index_k_proj.weight"),0,0,0,0,0,ID,H,&ok,&used);
            { m3_ent*e=REQ(LN("self_attn.index_q_norm.weight")); if(e) L->idx_q_norm=m3_cp_full(base,e); }
            { m3_ent*e=REQ(LN("self_attn.index_k_norm.weight")); if(e) L->idx_k_norm=m3_cp_full(base,e); }
            { m3_ent*e=REQ(LN("block_sparse_moe.gate.weight")); if(e) L->gate=(m3_tensor){m3_cp_full(base,e),NULL,M3_F32,cfg.n_experts,H}; }
            { m3_ent*e=REQ(LN("block_sparse_moe.e_score_correction_bias")); if(e) L->gate_bias=m3_cp_full(base,e); }
            L->sh_w1=m3_load_w(es,n,base,LN("block_sparse_moe.shared_experts.gate_proj.weight"),1,sh_r0,sh_rows,0,0,cfg.moe_inter,H,&ok,&used);
            L->sh_w3=m3_load_w(es,n,base,LN("block_sparse_moe.shared_experts.up_proj.weight"),1,sh_r0,sh_rows,0,0,cfg.moe_inter,H,&ok,&used);
            L->sh_w2=m3_load_w(es,n,base,LN("block_sparse_moe.shared_experts.down_proj.weight"),2,0,0,sh_r0,sh_rows,H,cfg.moe_inter,&ok,&used);
            L->sh_r0=sh_r0; L->sh_rows=sh_rows;
            int no=m3_n_owned(cfg.n_experts,ep_rank,ep_size); L->n_owned=no;
            L->owned_eid=malloc((size_t)(no>0?no:1)*sizeof(int));
            L->ex_w1=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor)); L->ex_w3=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor)); L->ex_w2=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor));
            int s=0; for(int e2=0;e2<cfg.n_experts&&ok;e2++) if(e2%ep_size==ep_rank){ L->owned_eid[s]=e2;
                snprintf(nb,sizeof nb,"language_model.model.layers.%d.block_sparse_moe.experts.%d.w1.weight",l,e2); L->ex_w1[s]=m3_load_w(es,n,base,nb,0,0,0,0,0,cfg.moe_inter,H,&ok,&used);
                snprintf(nb,sizeof nb,"language_model.model.layers.%d.block_sparse_moe.experts.%d.w3.weight",l,e2); L->ex_w3[s]=m3_load_w(es,n,base,nb,0,0,0,0,0,cfg.moe_inter,H,&ok,&used);
                snprintf(nb,sizeof nb,"language_model.model.layers.%d.block_sparse_moe.experts.%d.w2.weight",l,e2); L->ex_w2[s]=m3_load_w(es,n,base,nb,0,0,0,0,0,H,cfg.moe_inter,&ok,&used);
                s++; }
        } else {
            L->ff_gate=m3_load_w(es,n,base,LN("mlp.gate_proj.weight"),1,ff_r0,ff_rows,0,0,cfg.dense_inter,H,&ok,&used);
            L->ff_up  =m3_load_w(es,n,base,LN("mlp.up_proj.weight"),1,ff_r0,ff_rows,0,0,cfg.dense_inter,H,&ok,&used);
            L->ff_down=m3_load_w(es,n,base,LN("mlp.down_proj.weight"),2,0,0,ff_r0,ff_rows,H,cfg.dense_inter,&ok,&used);
            L->ff_r0=ff_r0; L->ff_rows=ff_rows;
        }
        #undef LN
    }
    #undef REQ
    munmap((void*)base,bsz); close(bfd); free(es);
    if(!ok){ fprintf(stderr,"m3_load: rank %d incomplete (missing tensors)\n",ep_rank); m3_free(m); return NULL; }
    /* scratch (same as synth) */
    m->s_norm=malloc(H*4); m->s_q=malloc(QD*4); m->s_k=malloc(KVD*4); m->s_v=malloc(KVD*4);
    m->s_attn=malloc(QD*4); m->s_o=malloc(H*4);
    m->s_idx_q=malloc((size_t)IQD*4); m->s_idx_k=malloc((size_t)ID*4);
    m->s_blk_score=malloc((size_t)cfg.max_pos*4); m->s_blk_sel=malloc((size_t)cfg.max_pos*sizeof(int));
    m->s_router=malloc((size_t)cfg.n_experts*4); m->s_shg=malloc(cfg.moe_inter*4); m->s_shu=malloc(cfg.moe_inter*4);
    m->s_sh=malloc(H*4); m->s_moe=malloc(H*4); m->s_exg=malloc(cfg.moe_inter*4); m->s_exu=malloc(cfg.moe_inter*4); m->s_route=malloc(H*4);
    m->s_ff_g=malloc(cfg.dense_inter*4); m->s_ff_u=malloc(cfg.dense_inter*4); m->s_ff=malloc(H*4);
    m->s_logits=malloc((size_t)hrows*4);
    m->arena_used=used; m->arena_sz=used;
    m3_init_fp8_lut(m->fp8_lut);
    m3_dummy=m3_envi("M3_DUMMY",0);
    if(m3_envi("M3_POOL",0)) m->pool=m3_g_pool=m3_pool_create(m->n_threads);  /* experimental pinned pool; default OpenMP */
    return m;
}

/* ===================== KV cache accessors (bf16 or int4, CP slot-mapped) ===================== */
/* dot(q_head, k[pos,kvh]) without the 1/sqrt(d) scale; reads bf16 or int4 at the CP slot. */
static inline float m3_kdot(m3_model*m,m3_layer*L,int t,int kvh,const float*qh,int HD,int KVH){
    long sl=m3_cp_slot(m,t);
    if(m->int4_kv) return m3_q4_dot(L->k_q4+sl*(size_t)(KVH*HD/2)+(size_t)kvh*(HD/2), L->k_qs[sl*KVH+kvh], qh, HD);
    const uint16_t*kt=L->k_cache+sl*(size_t)(KVH*HD)+(size_t)kvh*HD; double d=0; for(int i=0;i<HD;i++) d+=(double)qh[i]*m3_kv_dec(m,kt[i]); return (float)d;
}
/* oh[i] += w * v[pos,kvh][i] */
static inline void m3_vaxpy(m3_model*m,m3_layer*L,int t,int kvh,float w,float*oh,int HD,int KVH){
    long sl=m3_cp_slot(m,t);
    if(m->int4_kv){ m3_q4_axpy(oh, L->v_q4+sl*(size_t)(KVH*HD/2)+(size_t)kvh*(HD/2), L->v_qs[sl*KVH+kvh], w, HD); return; }
    const uint16_t*vt=L->v_cache+sl*(size_t)(KVH*HD)+(size_t)kvh*HD; for(int i=0;i<HD;i++) oh[i]+=w*m3_kv_dec(m,vt[i]);
}
/* dot(idx_q_head, idx_k[pos]) for MSA block scoring (1 MQA key head) */
static inline float m3_idxdot(m3_model*m,m3_layer*L,int t,const float*qh,int ID){
    long sl=m3_cp_slot(m,t);
    if(m->int4_kv) return m3_q4_dot(L->idx_q4+sl*(size_t)(ID/2), L->idx_qs[sl], qh, ID);
    const uint16_t*kt=L->idx_k_cache+sl*(size_t)ID; float d=0; for(int i=0;i<ID;i++) d+=qh[i]*m3_kv_dec(m,kt[i]); return d;
}

/* ===================== MSA: build the selected-position list for a sparse layer ===================== */
/* Returns nsel positions in sel[] (this rank's OWNED positions under CP); computes idx_q/idx_k,
 * stores idx_k at the owner's slot. When the block count is <= K+local+init (or MSA off) it
 * returns the full causal range (owned subset under CP). Under CP each rank scores only its
 * owned blocks, then blk_reduce_cb (all-reduce MAX) gives every rank the same global selection;
 * the cross-rank attention partials are merged later by kv_combine_cb. */
static int m3_msa_select(m3_model*m,m3_layer*L,const float*xn,int pos,int msa_on,int*sel){
    const m3_config*c=&m->cfg;
    const int ID=c->msa_index_dim, IH=c->msa_n_index_heads, B=c->msa_block_size, half=c->rotary_dim/2;
    const float*cosp=&m->rope_cos[(size_t)pos*half], *sinp=&m->rope_sin[(size_t)pos*half];
    /* index projections (replicated): idx_q [IH*ID], idx_k [ID] (MQA, 1 head) */
    float*iq=m->s_idx_q, *ik=m->s_idx_k;
    m3_mv(m,iq,&L->idx_wq,xn,IH*ID,c->hidden);
    m3_mv(m,ik,&L->idx_wk,xn,ID,c->hidden);
    for(int h=0;h<IH;h++){ float*qh=iq+h*ID; m3_rmsnorm_head(qh,L->idx_q_norm,ID,c->norm_eps); m3_rope_head(qh,cosp,sinp,c->rotary_dim); }
    m3_rmsnorm_head(ik,L->idx_k_norm,ID,c->norm_eps); m3_rope_head(ik,cosp,sinp,c->rotary_dim);
    if(m3_cp_mine(m,pos)){ long sl=m3_cp_slot(m,pos);            /* store this pos's index key (owner only) */
        if(m->int4_kv) L->idx_qs[sl]=m3_q4_pack(L->idx_q4+sl*(size_t)(ID/2), ik, ID);
        else for(int i=0;i<ID;i++) L->idx_k_cache[sl*(size_t)ID+i]=m3_kv_enc(m,ik[i]); }

    int nblk=pos/B+1;
    int keep=c->msa_topk_blocks+c->msa_local_block+c->msa_init_block;
    int dense_sel = (!msa_on || nblk<=keep);   /* attend all causal positions */
    char*selb=(char*)m->s_blk_sel;             /* reuse first nblk bytes as a block-selected bitmap */
    if(dense_sel){ for(int b=0;b<nblk;b++) selb[b]=1; }
    else {
        /* per-block score = max over positions of sum_h dot(idx_q_h, idx_k[t]); only OWN blocks */
        float*bs=m->s_blk_score;
        for(int b=0;b<nblk;b++){
            if(m->cp_on && b%m->ep_size!=m->ep_rank){ bs[b]=-1e30f; continue; }   /* not my block */
            int t0=b*B, t1=t0+B; if(t1>pos+1)t1=pos+1; float best=-1e30f;
            for(int t=t0;t<t1;t++){ float scr=0; for(int h=0;h<IH;h++) scr+=m3_idxdot(m,L,t,iq+h*ID,ID); if(scr>best)best=scr; }
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

/* ===================== forward (one token at position pos) ===================== */
static int m3_forward_token(m3_model*m,float*x,int pos){
    const m3_config*c=&m->cfg;
    const int H=c->hidden, HD=c->head_dim, QH=c->n_heads, KVH=c->n_kv_heads;
    const int KVD=m3_kv_dim(c), grp=QH/KVH, half=c->rotary_dim/2;
    const float ascale=1.0f/sqrtf((float)HD);
    const int msa_on=m3_envi("M3_MSA",1);
    float*xn=m->s_norm,*q=m->s_q,*kb=m->s_k,*vb=m->s_v,*attn=m->s_attn,*ao=m->s_o,*sc=m->s_blk_score;
    m->bytes_read=0;
    for(int l=0;l<c->n_layers;l++){
        m3_layer*L=&m->layers[l]; int is_moe=m3_is_moe(c,l);
        const int qh0=L->qh0, qh1=L->qh1, nown=qh1-qh0, qrows=nown*HD;
        const int tp_attn=(qrows<QH*HD);
        const float*cosp=&m->rope_cos[(size_t)pos*half], *sinp=&m->rope_sin[(size_t)pos*half];
        m3_rmsnorm_gemma(xn,x,L->input_norm,H,c->norm_eps);
        /* q: owned heads only (TP_ATTN) or all; k/v: replicated (full kv heads) */
        m3_mv(m,q,&L->wq,xn,qrows,H);
        m3_mv(m,kb,&L->wk,xn,KVD,H);
        m3_mv(m,vb,&L->wv,xn,KVD,H);
        for(int hh=0;hh<nown;hh++){ float*qh=q+hh*HD; if(c->use_qk_norm) m3_rmsnorm_head(qh,L->q_norm,HD,c->norm_eps); m3_rope_head(qh,cosp,sinp,c->rotary_dim); }
        for(int kh=0;kh<KVH;kh++){ float*kk=kb+kh*HD; if(c->use_qk_norm) m3_rmsnorm_head(kk,L->k_norm,HD,c->norm_eps); m3_rope_head(kk,cosp,sinp,c->rotary_dim); }
        /* store k/v at the owner's slot (bf16 or int4); under CP only the owning rank stores */
        if(m3_cp_mine(m,pos)){ long sl=m3_cp_slot(m,pos);
            if(m->int4_kv) for(int kh=0;kh<KVH;kh++){ L->k_qs[sl*KVH+kh]=m3_q4_pack(L->k_q4+sl*(size_t)(KVD/2)+(size_t)kh*(HD/2),kb+kh*HD,HD);
                                                      L->v_qs[sl*KVH+kh]=m3_q4_pack(L->v_q4+sl*(size_t)(KVD/2)+(size_t)kh*(HD/2),vb+kh*HD,HD); }
            else for(int i=0;i<KVD;i++){ L->k_cache[sl*(size_t)KVD+i]=m3_kv_enc(m,kb[i]); L->v_cache[sl*(size_t)KVD+i]=m3_kv_enc(m,vb[i]); } }
        /* selected positions (MSA on sparse layers; full causal otherwise); under CP = owned subset */
        int*selp=m->s_blk_sel;
        int nsel; int sparse=is_moe;       /* M3: sparse layers == moe layers (L>=n_dense) */
        if(sparse) nsel=m3_msa_select(m,L,xn,pos,msa_on,selp);
        else { nsel=0; for(int t=0;t<=pos;t++) if(m3_cp_mine(m,t)) selp[nsel++]=t; }
        /* GQA attention over selected positions; flash form (unnormalized), owned heads -> attn[hh*HD].
         * Under CP the per-rank partials (oh,max,sumexp) are merged by kv_combine_cb. */
        float hmx[64], hse[64];
        for(int hh=0;hh<nown;hh++){ int h=qh0+hh; float*qh=q+hh*HD; int kvh=h/grp; float mx=-1e30f;
            for(int j=0;j<nsel;j++){ float s=m3_kdot(m,L,selp[j],kvh,qh,HD,KVH)*ascale; sc[j]=s; if(s>mx)mx=s; }
            double sum=0; float*oh=attn+hh*HD; for(int i=0;i<HD;i++) oh[i]=0;
            for(int j=0;j<nsel;j++){ float e=expf(sc[j]-mx); sum+=e; m3_vaxpy(m,L,selp[j],kvh,e,oh,HD,KVH); }
            hmx[hh]=(nsel>0?mx:-1e30f); hse[hh]=(float)sum;          /* unnormalized; finalize below */
        }
        if(m->cp_on && m->kv_combine_cb) m->kv_combine_cb(attn,hmx,hse,nown,HD,m->kv_combine_ctx);
        else for(int hh=0;hh<nown;hh++){ float inv=1.0f/(hse[hh]>0?hse[hh]:1); float*oh=attn+hh*HD; for(int i=0;i<HD;i++) oh[i]*=inv; }
        /* o_proj (wo cols = owned head cols); partial under TP_ATTN -> ar_cb sum (CP: full, no reduce) */
        m3_mv(m,ao,&L->wo,attn,H,qrows);
        if(tp_attn && m->ar_cb) m->ar_cb(ao,H,m->ar_ctx);
        for(int i=0;i<H;i++) x[i]+=ao[i];
        /* FFN / MoE */
        float*h2=m->s_norm; m3_rmsnorm_gemma(h2,x,L->post_norm,H,c->norm_eps);
        if(is_moe){
            const int tp_sh=(L->sh_rows<c->moe_inter);
            float*rl=m->s_router; m3_mv_f32(rl,(float*)L->gate.w,h2,c->n_experts,H);
            for(int e=0;e<c->n_experts;e++) rl[e]=1.0f/(1.0f+expf(-rl[e]));
            int selx[64]; float selw[64]; int na=c->n_active>64?64:c->n_active;
            for(int a=0;a<na;a++){ int best=-1; float bv=-1e30f;
                for(int e=0;e<c->n_experts;e++){ int used=0; for(int j=0;j<a;j++) if(selx[j]==e){used=1;break;} if(used)continue;
                    float v=rl[e]+L->gate_bias[e]; if(v>bv){bv=v;best=e;} } selx[a]=best; selw[a]=rl[best]; }
            float wsum=0; for(int a=0;a<na;a++) wsum+=selw[a]; if(wsum<=0)wsum=1;
            float*route=m->s_route; for(int i=0;i<H;i++) route[i]=0;
            for(int a=0;a<na;a++){ int e=selx[a]; if(e%m->ep_size!=m->ep_rank) continue; int slot=e/m->ep_size;
                float w=selw[a]/wsum*c->routed_scale;
                m3_mv(m,m->s_exg,&L->ex_w1[slot],h2,c->moe_inter,H);
                m3_mv(m,m->s_exu,&L->ex_w3[slot],h2,c->moe_inter,H);
                for(int i=0;i<c->moe_inter;i++) m->s_exg[i]=m3_swiglu_oai(m->s_exg[i],m->s_exu[i],c->swiglu_alpha,c->swiglu_limit);
                m3_mv(m,m->s_moe,&L->ex_w2[slot],m->s_exg,H,c->moe_inter);
                for(int i=0;i<H;i++) route[i]+=w*m->s_moe[i]; }
            /* shared expert: TP-sharded -> fold partial into route[] (one reduce); else replicated -> add after */
            m3_mv(m,m->s_shg,&L->sh_w1,h2,L->sh_rows,H);
            m3_mv(m,m->s_shu,&L->sh_w3,h2,L->sh_rows,H);
            for(int i=0;i<L->sh_rows;i++) m->s_shg[i]=m3_swiglu_oai(m->s_shg[i],m->s_shu[i],c->swiglu_alpha,c->swiglu_limit);
            m3_mv(m,m->s_sh,&L->sh_w2,m->s_shg,H,L->sh_rows);
            if(tp_sh) for(int i=0;i<H;i++) route[i]+=m->s_sh[i];
            if(m->ar_cb) m->ar_cb(route,H,m->ar_ctx);     /* EP-sum routed (+ shared if TP) */
            for(int i=0;i<H;i++) x[i]+=route[i] + (tp_sh?0.0f:m->s_sh[i]);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            m3_mv(m,m->s_ff_g,&L->ff_gate,h2,L->ff_rows,H);
            m3_mv(m,m->s_ff_u,&L->ff_up,h2,L->ff_rows,H);
            for(int i=0;i<L->ff_rows;i++) m->s_ff_g[i]=m3_swiglu_oai(m->s_ff_g[i],m->s_ff_u[i],c->swiglu_alpha,c->swiglu_limit);
            m3_mv(m,m->s_ff,&L->ff_down,m->s_ff_g,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(m->s_ff,H,m->ar_ctx);
            for(int i=0;i<H;i++) x[i]+=m->s_ff[i];
        }
    }
    /* head: vocab-shard partial logits -> (TP_HEAD) global argmax via ar_argmax, else full */
    float*h2=m->s_norm; m3_rmsnorm_gemma(h2,x,m->out_norm,H,c->norm_eps);
    int hrows=m->head.rows; m3_mv(m,m->s_logits,&m->head,h2,hrows,H);
    int la=0; float lv=m->s_logits[0]; for(int i=1;i<hrows;i++) if(m->s_logits[i]>lv){lv=m->s_logits[i];la=i;}
    int32_t gidx=m->head_r0+la; float gval=lv;
    if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);  /* TP_HEAD merge */
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
    float *xn,*q,*k,*v,*attn,*o,*h2,*router,*route,*shg,*shu,*ffg,*ffu,*tmp2;  /* token-major batched scratch */
    float *exg,*exu,*emoe;    /* per-token expert scratch (M=1) */
    int *bk; float *bw; int *bcnt;   /* expert grouping: per-owned-slot token buckets [n_experts*N]/[n_experts] */
    float *logits,*sc;
    int *psel, *pnsel;        /* prefill: per-token MSA selected positions [n*maxsel] + counts [n] (kc/vc NULL) */
    int maxsel;
    int *gsel; float *gselw;  /* router top-k per token [n*8] (heap; prefill N can exceed 64) */
} m3_mstream;

/* Y[N,rows] (token-major) = X[N,cols] . W[rows,cols]^T. K-tiled (8 W rows for a tile stay
 * in L1 across the N tokens -> weight read amortized over N). OpenMP over 8-row blocks. */
static void m3_gemm_bf16(float*restrict Y, const uint16_t*W, const float*X, int N, int rows, int cols){
    if(m3_dummy){
        double acc=0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static) if((long)rows>=M3_PAR_MIN)
#endif
        for(int r=0;r<rows;r++){ const uint16_t*w=W+(size_t)r*cols; uint32_t s=0; for(int i=0;i<cols;i+=32) s+=w[i]; acc+=s; }
        float vv=(float)(((long)acc)&1)*1e-30f; for(size_t i=0;i<(size_t)N*rows;i++) Y[i]=vv; return;
    }
    int nb=rows/8, TILE=512; if(TILE>cols)TILE=cols;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)rows>=M3_PAR_MIN)
#endif
    for(int bi=0;bi<nb;bi++){
        int r=bi*8; const uint16_t*w=W+(size_t)r*cols;
        for(int t=0;t<N;t++){ float*y=Y+(size_t)t*rows+r; for(int j=0;j<8;j++) y[j]=0.f; }
        for(int k0=0;k0<cols;k0+=TILE){ int kl=cols-k0<TILE?cols-k0:TILE; const uint16_t*tw=w+k0;
            for(int t=0;t<N;t++){ float tmp[8];
                matvec_bf16_8row(tmp,tw,tw+cols,tw+2*(size_t)cols,tw+3*(size_t)cols,
                                 tw+4*(size_t)cols,tw+5*(size_t)cols,tw+6*(size_t)cols,tw+7*(size_t)cols,
                                 X+(size_t)t*cols+k0,kl);
                float*y=Y+(size_t)t*rows+r; for(int j=0;j<8;j++) y[j]+=tmp[j]; } }
    }
    for(int r=nb*8;r<rows;r++) for(int t=0;t<N;t++) Y[(size_t)t*rows+r]=vec_dot_bf16_f32(W+(size_t)r*cols,X+(size_t)t*cols,cols);
}

/* batched MXFP8 GEMM: Y[N,rows] = decode(W fp8 + S e8m0) . X[N,cols]. For N>1 the expensive
 * part is the FP8 LUT-gather decode -- so we decode each 8-row x TILE block ONCE into a bf16
 * tile (FP8*2^k is exact in bf16), then run N cheap matvec_bf16_8row over it. This amortizes
 * the decode by N (the multi-stream win). N==1 uses the direct fused kernel (no tile staging). */
#define M3_MXG_TILE 512
static void m3_gemm_mxfp8(m3_model*m, float*restrict Y, const uint8_t*W, const uint8_t*S, const float*X, int N, int rows, int cols){
    const uint32_t*lut=m->fp8_lut; int sb=cols/M3_MX_BLK, nb=rows/8;
    if(m3_dummy){ double acc=0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static) if((long)rows>=M3_PAR_MIN)
#endif
        for(int r=0;r<rows;r++){ const uint8_t*w=W+(size_t)r*cols; uint32_t s=0; for(int i=0;i<cols;i+=64) s+=w[i]; acc+=s; }
        float v=(float)(((long)acc)&1)*1e-30f; for(size_t i=0;i<(size_t)N*rows;i++) Y[i]=v; return; }
    if(N==1){  /* direct fused: no tile, decode==matvec are one pass */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if((long)rows>=M3_PAR_MIN)
#endif
        for(int bi=0;bi<nb;bi++){ int r=bi*8; const uint8_t*w=W+(size_t)r*cols,*s=S+(size_t)r*sb;
            m3_matvec_mxfp8_8row(Y+r, w,w+cols,w+2*(size_t)cols,w+3*(size_t)cols,w+4*(size_t)cols,w+5*(size_t)cols,w+6*(size_t)cols,w+7*(size_t)cols,
                                 s,s+sb,s+2*sb,s+3*sb,s+4*sb,s+5*sb,s+6*sb,s+7*sb, X, cols, lut); }
    } else {
        int TILE=M3_MXG_TILE<cols?M3_MXG_TILE:cols;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if((long)rows>=M3_PAR_MIN)
#endif
        for(int bi=0;bi<nb;bi++){ int r=bi*8; const uint8_t*w=W+(size_t)r*cols,*s=S+(size_t)r*sb;
            uint16_t tile[8*M3_MXG_TILE];   /* decoded bf16 weights, 8KB/thread, L1-resident */
            for(int t=0;t<N;t++){ float*y=Y+(size_t)t*rows+r; for(int j=0;j<8;j++) y[j]=0.f; }
            for(int k0=0;k0<cols;k0+=TILE){ int kl=cols-k0<TILE?cols-k0:TILE;
                for(int j=0;j<8;j++) m3_mxfp8_decode_row_bf16(tile+(size_t)j*kl, w+(size_t)j*cols, s+(size_t)j*sb, k0, kl, lut);
                for(int t=0;t<N;t++){ float tmp[8];
                    matvec_bf16_8row(tmp, tile,tile+kl,tile+2*kl,tile+3*kl,tile+4*kl,tile+5*kl,tile+6*kl,tile+7*kl, X+(size_t)t*cols+k0, kl);
                    float*y=Y+(size_t)t*rows+r; for(int j=0;j<8;j++) y[j]+=tmp[j]; } }
        }
    }
    for(int r=nb*8;r<rows;r++){ const uint8_t*w=W+(size_t)r*cols,*s=S+(size_t)r*sb;
        for(int t=0;t<N;t++){ const float*x=X+(size_t)t*cols; double a=0;
            for(int b=0;b<cols;b+=M3_MX_BLK){ float sc=m3_e8m0(s[b/M3_MX_BLK]); int e=b+M3_MX_BLK<cols?b+M3_MX_BLK:cols;
                for(int c=b;c<e;c++){ float wf; uint32_t u=lut[w[c]]; memcpy(&wf,&u,4); a+=(double)wf*sc*x[c]; } } Y[(size_t)t*rows+r]=(float)a; } }
}
/* batched GEMM dispatch by weight type */
static void m3_gemm(m3_model*m, float*restrict Y, const m3_tensor*t, const float*X, int N, int rows, int cols){
    if(t->type==M3_MXFP8) m3_gemm_mxfp8(m,Y,(const uint8_t*)t->w,t->scale,X,N,rows,cols);
    else m3_gemm_bf16(Y,(const uint16_t*)t->w,X,N,rows,cols);
}

static void m3_free_mstream(m3_model*m){
    m3_mstream*ms=(m3_mstream*)m->ms; if(!ms) return;
    free(ms->kc);free(ms->vc);free(ms->xn);free(ms->q);free(ms->k);free(ms->v);free(ms->attn);free(ms->o);
    free(ms->h2);free(ms->router);free(ms->route);free(ms->shg);free(ms->shu);free(ms->ffg);free(ms->ffu);
    free(ms->tmp2);free(ms->exg);free(ms->exu);free(ms->emoe);free(ms->bk);free(ms->bw);free(ms->bcnt);free(ms->logits);free(ms->sc);
    free(ms->psel);free(ms->pnsel);free(ms->gsel);free(ms->gselw);
    free(ms); m->ms=NULL;
}
/* per_stream_kv=1: multi-stream decode (own KV per stream). 0: chunked prefill (shared model
 * KV cache; allocate per-token MSA selection buffers psel/pnsel instead). */
static int m3_alloc_mstream_ex(m3_model*m,int N,int per_stream_kv){
    const m3_config*c=&m->cfg; int H=c->hidden,QD=m3_q_dim(c),KVD=m3_kv_dim(c),hrows=m->head.rows;
    m3_mstream*ms=calloc(1,sizeof *ms); if(!ms) return -1; ms->n=N;
    size_t per=(size_t)c->n_layers*c->max_pos*KVD;
    if(per_stream_kv){ ms->kc=calloc((size_t)N*per,2); ms->vc=calloc((size_t)N*per,2); }
    else { ms->maxsel=(c->msa_topk_blocks+c->msa_local_block+c->msa_init_block+1)*c->msa_block_size;
           ms->psel=malloc((size_t)N*ms->maxsel*sizeof(int)); ms->pnsel=malloc((size_t)N*sizeof(int));
           ms->gsel=malloc((size_t)N*8*sizeof(int)); ms->gselw=malloc((size_t)N*8*sizeof(float)); }
    ms->xn=malloc((size_t)N*H*4); ms->q=malloc((size_t)N*QD*4); ms->k=malloc((size_t)N*KVD*4); ms->v=malloc((size_t)N*KVD*4);
    ms->attn=malloc((size_t)N*QD*4); ms->o=malloc((size_t)N*H*4); ms->h2=malloc((size_t)N*H*4);
    ms->router=malloc((size_t)N*c->n_experts*4); ms->route=malloc((size_t)N*H*4);
    ms->shg=malloc((size_t)N*c->moe_inter*4); ms->shu=malloc((size_t)N*c->moe_inter*4);
    ms->ffg=malloc((size_t)N*c->dense_inter*4); ms->ffu=malloc((size_t)N*c->dense_inter*4);
    ms->tmp2=malloc((size_t)N*H*4);
    ms->exg=malloc((size_t)c->moe_inter*4); ms->exu=malloc((size_t)c->moe_inter*4); ms->emoe=malloc((size_t)H*4);
    ms->bk=malloc((size_t)c->n_experts*N*sizeof(int)); ms->bw=malloc((size_t)c->n_experts*N*4); ms->bcnt=malloc((size_t)c->n_experts*sizeof(int));
    ms->logits=malloc((size_t)N*hrows*4); ms->sc=malloc((size_t)N*c->max_pos*4);
    int kvok = per_stream_kv ? (ms->kc&&ms->vc) : (ms->psel&&ms->pnsel);
    if(!kvok||!ms->logits){ m->ms=ms; m3_free_mstream(m); return -1; }
    m->ms=ms; return 0;
}
static int m3_alloc_mstream(m3_model*m,int N){ return m3_alloc_mstream_ex(m,N,1); }

/* one batched decode step: N tokens (X token-major [N,hidden]) at positions pos[t] -> out[t]=argmax. */
static void m3_forward_batch_decode(m3_model*m, float*X, int N, const int*pos, int*out){
    const m3_config*c=&m->cfg; const int H=c->hidden,HD=c->head_dim,QH=c->n_heads,KVH=c->n_kv_heads;
    const int KVD=m3_kv_dim(c),grp=QH/KVH,half=c->rotary_dim/2; const float ascale=1.0f/sqrtf((float)HD);
    m3_mstream*ms=(m3_mstream*)m->ms; size_t per=(size_t)c->n_layers*c->max_pos*KVD;
    for(int l=0;l<c->n_layers;l++){
        m3_layer*L=&m->layers[l]; int is_moe=m3_is_moe(c,l);
        const int qh0=L->qh0,qh1=L->qh1,nown=qh1-qh0,qrows=nown*HD; const int tp_attn=(qrows<QH*HD);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<N;t++) m3_rmsnorm_gemma(ms->xn+(size_t)t*H, X+(size_t)t*H, L->input_norm, H, c->norm_eps);
        m3_gemm(m,ms->q,&L->wq,ms->xn,N,qrows,H);
        m3_gemm(m,ms->k,&L->wk,ms->xn,N,KVD,H);
        m3_gemm(m,ms->v,&L->wv,ms->xn,N,KVD,H);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<N;t++){ int p=pos[t]; const float*cosp=&m->rope_cos[(size_t)p*half],*sinp=&m->rope_sin[(size_t)p*half];
            float*qb=ms->q+(size_t)t*qrows,*kb=ms->k+(size_t)t*KVD,*vb=ms->v+(size_t)t*KVD;
            for(int hh=0;hh<nown;hh++){ float*qh=qb+hh*HD; if(c->use_qk_norm) m3_rmsnorm_head(qh,L->q_norm,HD,c->norm_eps); m3_rope_head(qh,cosp,sinp,c->rotary_dim); }
            for(int kh=0;kh<KVH;kh++){ float*kk=kb+kh*HD; if(c->use_qk_norm) m3_rmsnorm_head(kk,L->k_norm,HD,c->norm_eps); m3_rope_head(kk,cosp,sinp,c->rotary_dim); }
            uint16_t*kc=ms->kc+(size_t)t*per+(size_t)l*c->max_pos*KVD+(size_t)p*KVD;
            uint16_t*vc=ms->vc+(size_t)t*per+(size_t)l*c->max_pos*KVD+(size_t)p*KVD;
            for(int i=0;i<KVD;i++){ kc[i]=m3_f2bf(kb[i]); vc[i]=m3_f2bf(vb[i]); } }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<N;t++){ int p=pos[t]; float*qb=ms->q+(size_t)t*qrows,*ab=ms->attn+(size_t)t*qrows,*sc=ms->sc+(size_t)t*c->max_pos;
            uint16_t*kcl=ms->kc+(size_t)t*per+(size_t)l*c->max_pos*KVD,*vcl=ms->vc+(size_t)t*per+(size_t)l*c->max_pos*KVD;
            for(int hh=0;hh<nown;hh++){ int hgl=qh0+hh; float*qh=qb+hh*HD; int kvh=hgl/grp; float mx=-1e30f;
                for(int tt=0;tt<=p;tt++){ const uint16_t*kt=kcl+(size_t)tt*KVD+kvh*HD; double d=0; for(int i=0;i<HD;i++) d+=(double)qh[i]*m3_bf2f(kt[i]); float s=(float)d*ascale; sc[tt]=s; if(s>mx)mx=s; }
                double sum=0; for(int tt=0;tt<=p;tt++){ float e=expf(sc[tt]-mx); sc[tt]=e; sum+=e; }
                float inv=(float)(1.0/(sum>0?sum:1)); float*oh=ab+hh*HD; for(int i=0;i<HD;i++) oh[i]=0;
                for(int tt=0;tt<=p;tt++){ float w=sc[tt]*inv; const uint16_t*vt=vcl+(size_t)tt*KVD+kvh*HD; for(int i=0;i<HD;i++) oh[i]+=w*m3_bf2f(vt[i]); } } }
        m3_gemm(m,ms->o,&L->wo,ms->attn,N,H,qrows);
        if(tp_attn && m->ar_cb) m->ar_cb(ms->o,N*H,m->ar_ctx);
        for(size_t i=0;i<(size_t)N*H;i++) X[i]+=ms->o[i];
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<N;t++) m3_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, L->post_norm, H, c->norm_eps);
        if(is_moe){
            const int tp_sh=(L->sh_rows<c->moe_inter);
            for(size_t i=0;i<(size_t)N*H;i++) ms->route[i]=0;
            int na=c->n_active>8?8:c->n_active; int sel_all[64*8]; float selw_all[64*8];
            /* router + select: parallel over streams (F32 gate matvec, no nested matvec) */
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int t=0;t<N;t++){ float*rl=ms->router+(size_t)t*c->n_experts; m3_mv_f32(rl,(float*)L->gate.w,ms->h2+(size_t)t*H,c->n_experts,H);
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
                m3_gemm(m,ms->shg,&L->ex_w1[s],ms->o,g,c->moe_inter,H);
                m3_gemm(m,ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H);
                for(size_t i=0;i<(size_t)g*c->moe_inter;i++) ms->shg[i]=m3_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
                m3_gemm(m,ms->tmp2,&L->ex_w2[s],ms->shg,g,H,c->moe_inter);
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*N+i]; float w=ms->bw[(size_t)s*N+i]; float*rt=ms->route+(size_t)t*H,*dn=ms->tmp2+(size_t)i*H;
                    for(int j=0;j<H;j++) rt[j]+=w*dn[j]; } }
            /* comm-overlap: issue the routed reduce on the comm-driver thread, compute the
             * (replicated) shared expert during it, then wait. Needs shared replicated (!tp_sh)
             * so it is independent of the route buffer being reduced. */
            int overlap = (m->ar_async_start && !tp_sh);
            if(overlap) m->ar_async_start(ms->route,N*H,m->ar_async_ctx);
            m3_gemm(m,ms->shg,&L->sh_w1,ms->h2,N,L->sh_rows,H);
            m3_gemm(m,ms->shu,&L->sh_w3,ms->h2,N,L->sh_rows,H);
            for(size_t i=0;i<(size_t)N*L->sh_rows;i++) ms->shg[i]=m3_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
            m3_gemm(m,ms->tmp2,&L->sh_w2,ms->shg,N,H,L->sh_rows);   /* shared-out [N,H] */
            if(overlap){ m->ar_wait(m->ar_async_ctx); }
            else { if(tp_sh) for(size_t i=0;i<(size_t)N*H;i++) ms->route[i]+=ms->tmp2[i];
                   if(m->ar_cb) m->ar_cb(ms->route,N*H,m->ar_ctx); }
            for(size_t i=0;i<(size_t)N*H;i++) X[i]+=ms->route[i] + (tp_sh?0.0f:ms->tmp2[i]);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            m3_gemm(m,ms->ffg,&L->ff_gate,ms->h2,N,L->ff_rows,H);
            m3_gemm(m,ms->ffu,&L->ff_up,ms->h2,N,L->ff_rows,H);
            for(size_t i=0;i<(size_t)N*L->ff_rows;i++) ms->ffg[i]=m3_swiglu_oai(ms->ffg[i],ms->ffu[i],c->swiglu_alpha,c->swiglu_limit);
            m3_gemm(m,ms->tmp2,&L->ff_down,ms->ffg,N,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(ms->tmp2,N*H,m->ar_ctx);
            for(size_t i=0;i<(size_t)N*H;i++) X[i]+=ms->tmp2[i];
        }
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<N;t++) m3_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, m->out_norm, H, c->norm_eps);
    int hrows=m->head.rows; m3_gemm(m,ms->logits,&m->head,ms->h2,N,hrows,H);
    for(int t=0;t<N;t++){ float*lg=ms->logits+(size_t)t*hrows; int la=0; float bv=lg[0]; for(int i=1;i<hrows;i++) if(lg[i]>bv){bv=lg[i];la=i;}
        int32_t gidx=m->head_r0+la; float gval=bv;
        if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);
        out[t]=gidx; }
}

/* ===================== chunked batched prefill (Lever 1, >=100 tok/s target) =====================
 * S consecutive tokens of ONE sequence at positions [p0,p0+S), X token-major [S,hidden]. Writes
 * the model's SHARED KV cache (bf16/fp16/int4, CP-aware) and does causal attention (token i sees
 * [0,p0+i]; MSA top-k per token on sparse layers). The projection/FFN/expert GEMMs run M=S with
 * ONE all-reduce per layer per chunk -> comm + weight-read + FP8-decode all amortized S-fold.
 * Returns the LAST token's argmax (next-token prediction); other tokens' logits aren't needed.
 * Prefill is compute-bound -> ideally run WITHOUT CP (replicated/TP KV); CP works but adds a
 * per-token collective in the MSA select. Requires m3_alloc_mstream_ex(m,S,0). */
static int m3_forward_prefill_chunk(m3_model*m, float*X, int S, int p0){
    const m3_config*c=&m->cfg; const int H=c->hidden,HD=c->head_dim,QH=c->n_heads,KVH=c->n_kv_heads;
    const int KVD=m3_kv_dim(c),grp=QH/KVH,half=c->rotary_dim/2; const float ascale=1.0f/sqrtf((float)HD);
    m3_mstream*ms=(m3_mstream*)m->ms; const int msa_on=m3_envi("M3_MSA",1);
    for(int l=0;l<c->n_layers;l++){
        m3_layer*L=&m->layers[l]; int is_moe=m3_is_moe(c,l), sparse=is_moe;
        const int qh0=L->qh0,qh1=L->qh1,nown=qh1-qh0,qrows=nown*HD; const int tp_attn=(qrows<QH*HD);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++) m3_rmsnorm_gemma(ms->xn+(size_t)t*H, X+(size_t)t*H, L->input_norm, H, c->norm_eps);
        m3_gemm(m,ms->q,&L->wq,ms->xn,S,qrows,H);
        m3_gemm(m,ms->k,&L->wk,ms->xn,S,KVD,H);
        m3_gemm(m,ms->v,&L->wv,ms->xn,S,KVD,H);
        /* qk-norm + RoPE(pos=p0+t); store K/V to the shared cache at pos p0+t (owner only under CP) */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++){ int p=p0+t; const float*cosp=&m->rope_cos[(size_t)p*half],*sinp=&m->rope_sin[(size_t)p*half];
            float*qb=ms->q+(size_t)t*qrows,*kb=ms->k+(size_t)t*KVD,*vb=ms->v+(size_t)t*KVD;
            for(int hh=0;hh<nown;hh++){ float*qh=qb+hh*HD; if(c->use_qk_norm) m3_rmsnorm_head(qh,L->q_norm,HD,c->norm_eps); m3_rope_head(qh,cosp,sinp,c->rotary_dim); }
            for(int kh=0;kh<KVH;kh++){ float*kk=kb+kh*HD; if(c->use_qk_norm) m3_rmsnorm_head(kk,L->k_norm,HD,c->norm_eps); m3_rope_head(kk,cosp,sinp,c->rotary_dim); }
            if(m3_cp_mine(m,p)){ long sl=m3_cp_slot(m,p);
                if(m->int4_kv) for(int kh=0;kh<KVH;kh++){ L->k_qs[sl*KVH+kh]=m3_q4_pack(L->k_q4+sl*(size_t)(KVD/2)+(size_t)kh*(HD/2),kb+kh*HD,HD);
                                                          L->v_qs[sl*KVH+kh]=m3_q4_pack(L->v_q4+sl*(size_t)(KVD/2)+(size_t)kh*(HD/2),vb+kh*HD,HD); }
                else for(int i=0;i<KVD;i++){ L->k_cache[sl*(size_t)KVD+i]=m3_kv_enc(m,kb[i]); L->v_cache[sl*(size_t)KVD+i]=m3_kv_enc(m,vb[i]); } } }
        /* MSA selection per token (sequential: m3_msa_select stores idx[p0+t] then selects, so
         * earlier intra-chunk tokens' idx are visible; uses shared scratch -> not parallel). */
        if(sparse) for(int t=0;t<S;t++){ int*sel=ms->psel+(size_t)t*ms->maxsel;
            ms->pnsel[t]=m3_msa_select(m,L,ms->xn+(size_t)t*H,p0+t,msa_on,sel); }
        /* causal attention per token (flash form); sparse -> sel list, dense -> contiguous [0,p] */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++){ int p=p0+t; float*qb=ms->q+(size_t)t*qrows,*ab=ms->attn+(size_t)t*qrows,*sc=ms->sc+(size_t)t*c->max_pos;
            const int*sel=ms->psel+(size_t)t*ms->maxsel; int ns=sparse?ms->pnsel[t]:0;
            float hmx[64],hse[64];
            for(int hh=0;hh<nown;hh++){ int hgl=qh0+hh; float*qh=qb+hh*HD; int kvh=hgl/grp; float mx=-1e30f; double sum=0; float*oh=ab+hh*HD; for(int i=0;i<HD;i++) oh[i]=0;
                if(sparse){
                    for(int j=0;j<ns;j++){ float s=m3_kdot(m,L,sel[j],kvh,qh,HD,KVH)*ascale; sc[j]=s; if(s>mx)mx=s; }
                    for(int j=0;j<ns;j++){ float e=expf(sc[j]-mx); sum+=e; m3_vaxpy(m,L,sel[j],kvh,e,oh,HD,KVH); }
                    hmx[hh]=(ns>0?mx:-1e30f);
                } else {
                    int any=0;
                    for(int tt=0;tt<=p;tt++){ if(!m3_cp_mine(m,tt))continue; float s=m3_kdot(m,L,tt,kvh,qh,HD,KVH)*ascale; sc[tt]=s; if(s>mx)mx=s; any=1; }
                    for(int tt=0;tt<=p;tt++){ if(!m3_cp_mine(m,tt))continue; float e=expf(sc[tt]-mx); sum+=e; m3_vaxpy(m,L,tt,kvh,e,oh,HD,KVH); }
                    hmx[hh]=(any?mx:-1e30f);
                }
                hse[hh]=(float)sum; }
            if(!(m->cp_on && m->kv_combine_cb)) for(int hh=0;hh<nown;hh++){ float inv=1.0f/(hse[hh]>0?hse[hh]:1); float*oh=ab+hh*HD; for(int i=0;i<HD;i++) oh[i]*=inv; }
            else m->kv_combine_cb(ab,hmx,hse,nown,HD,m->kv_combine_ctx); }
        m3_gemm(m,ms->o,&L->wo,ms->attn,S,H,qrows);
        if(tp_attn && m->ar_cb) m->ar_cb(ms->o,S*H,m->ar_ctx);
        for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->o[i];
        /* post-norm + MoE/FFN (M=S; identical to batch_decode) */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int t=0;t<S;t++) m3_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, L->post_norm, H, c->norm_eps);
        if(is_moe){
            const int tp_sh=(L->sh_rows<c->moe_inter);
            for(size_t i=0;i<(size_t)S*H;i++) ms->route[i]=0;
            int na=c->n_active>8?8:c->n_active; int*sel_all=ms->gsel; float*selw_all=ms->gselw;  /* [S*8] heap (S may exceed 64) */
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int t=0;t<S;t++){ float*rl=ms->router+(size_t)t*c->n_experts; m3_mv_f32(rl,(float*)L->gate.w,ms->h2+(size_t)t*H,c->n_experts,H);
                for(int e=0;e<c->n_experts;e++) rl[e]=1.0f/(1.0f+expf(-rl[e]));
                int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int best=-1; float bv=-1e30f; for(int e=0;e<c->n_experts;e++){ int used=0; for(int j=0;j<a;j++) if(sel[j]==e){used=1;break;} if(used)continue; float vv=rl[e]+L->gate_bias[e]; if(vv>bv){bv=vv;best=e;} } sel[a]=best; sw[a]=rl[best]; }
                float wsum=0; for(int a=0;a<na;a++) wsum+=sw[a]; if(wsum<=0)wsum=1; for(int a=0;a<na;a++) sw[a]=sw[a]/wsum*c->routed_scale; }
            for(int s=0;s<L->n_owned;s++) ms->bcnt[s]=0;
            for(int t=0;t<S;t++){ int*sel=sel_all+t*na; float*sw=selw_all+t*na;
                for(int a=0;a<na;a++){ int e=sel[a]; if(e%m->ep_size!=m->ep_rank) continue; int slot=e/m->ep_size;
                    int g=ms->bcnt[slot]++; ms->bk[(size_t)slot*S+g]=t; ms->bw[(size_t)slot*S+g]=sw[a]; } }
            for(int s=0;s<L->n_owned;s++){ int g=ms->bcnt[s]; if(g==0) continue;
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*S+i]; memcpy(ms->o+(size_t)i*H, ms->h2+(size_t)t*H, (size_t)H*4); }
                m3_gemm(m,ms->shg,&L->ex_w1[s],ms->o,g,c->moe_inter,H);
                m3_gemm(m,ms->shu,&L->ex_w3[s],ms->o,g,c->moe_inter,H);
                for(size_t i=0;i<(size_t)g*c->moe_inter;i++) ms->shg[i]=m3_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
                m3_gemm(m,ms->tmp2,&L->ex_w2[s],ms->shg,g,H,c->moe_inter);
                for(int i=0;i<g;i++){ int t=ms->bk[(size_t)s*S+i]; float w=ms->bw[(size_t)s*S+i]; float*rt=ms->route+(size_t)t*H,*dn=ms->tmp2+(size_t)i*H;
                    for(int j=0;j<H;j++) rt[j]+=w*dn[j]; } }
            int overlap = (m->ar_async_start && !tp_sh);
            if(overlap) m->ar_async_start(ms->route,S*H,m->ar_async_ctx);
            m3_gemm(m,ms->shg,&L->sh_w1,ms->h2,S,L->sh_rows,H);
            m3_gemm(m,ms->shu,&L->sh_w3,ms->h2,S,L->sh_rows,H);
            for(size_t i=0;i<(size_t)S*L->sh_rows;i++) ms->shg[i]=m3_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
            m3_gemm(m,ms->tmp2,&L->sh_w2,ms->shg,S,H,L->sh_rows);
            if(overlap){ m->ar_wait(m->ar_async_ctx); }
            else { if(tp_sh) for(size_t i=0;i<(size_t)S*H;i++) ms->route[i]+=ms->tmp2[i];
                   if(m->ar_cb) m->ar_cb(ms->route,S*H,m->ar_ctx); }
            for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->route[i] + (tp_sh?0.0f:ms->tmp2[i]);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            m3_gemm(m,ms->ffg,&L->ff_gate,ms->h2,S,L->ff_rows,H);
            m3_gemm(m,ms->ffu,&L->ff_up,ms->h2,S,L->ff_rows,H);
            for(size_t i=0;i<(size_t)S*L->ff_rows;i++) ms->ffg[i]=m3_swiglu_oai(ms->ffg[i],ms->ffu[i],c->swiglu_alpha,c->swiglu_limit);
            m3_gemm(m,ms->tmp2,&L->ff_down,ms->ffg,S,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(ms->tmp2,S*H,m->ar_ctx);
            for(size_t i=0;i<(size_t)S*H;i++) X[i]+=ms->tmp2[i];
        }
    }
    /* head: LAST token only (next-token prediction) */
    int hrows=m->head.rows; m3_rmsnorm_gemma(ms->h2, X+(size_t)(S-1)*H, m->out_norm, H, c->norm_eps);
    m3_gemm(m,ms->logits,&m->head,ms->h2,1,hrows,H);
    float*lg=ms->logits; int la=0; float bv=lg[0]; for(int i=1;i<hrows;i++) if(lg[i]>bv){bv=lg[i];la=i;}
    int32_t gidx=m->head_r0+la; float gval=bv;
    if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);
    return gidx;
}

#endif /* M3_IMPL_H */
