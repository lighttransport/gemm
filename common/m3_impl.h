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

static m3_model* m3_alloc_synth(m3_config cfg,int ep_rank,int ep_size,int n_threads,int n_cmgs){
    m3_model*m=calloc(1,sizeof(m3_model)); if(!m) return NULL;
    m->cfg=cfg; m->ep_rank=ep_rank; m->ep_size=ep_size; m->n_threads=n_threads; m->n_cmgs=n_cmgs;
    m->bf16_mv_qt=M3_BF16;
    const int H=cfg.hidden, QD=m3_q_dim(&cfg), KVD=m3_kv_dim(&cfg), HD=cfg.head_dim;
    const int IQD=m3_idx_q_dim(&cfg), ID=cfg.msa_index_dim, half=cfg.rotary_dim/2;
    /* TP flags */
    int tp=m3_envi("M3_TP",0);
    int tp_attn=m3_envi("M3_TP_ATTN",tp), tp_sh=m3_envi("M3_TP_SHARED",tp);
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
        L->k_cache=calloc((size_t)cfg.max_pos*KVD,2); L->v_cache=calloc((size_t)cfg.max_pos*KVD,2);
        used+=2*(size_t)cfg.max_pos*KVD*2;
        if(is_moe){
            /* MSA indexer (replicated) */
            BF(p,(size_t)IQD*H,0.03f); L->idx_wq=(m3_tensor){p,NULL,M3_BF16,IQD,H};
            BF(p,(size_t)ID*H,0.03f);  L->idx_wk=(m3_tensor){p,NULL,M3_BF16,ID,H};
            BF(L->idx_q_norm,ID,0.1f); BF(L->idx_k_norm,ID,0.1f);
            L->idx_k_cache=calloc((size_t)cfg.max_pos*ID,2); used+=(size_t)cfg.max_pos*ID*2;
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

    m3_model*m=calloc(1,sizeof(m3_model)); m->cfg=cfg; m->ep_rank=ep_rank; m->ep_size=ep_size; m->n_threads=n_threads; m->n_cmgs=n_cmgs; m->bf16_mv_qt=M3_BF16;
    const int H=cfg.hidden, QD=m3_q_dim(&cfg), KVD=m3_kv_dim(&cfg), HD=cfg.head_dim;
    const int IQD=m3_idx_q_dim(&cfg), ID=cfg.msa_index_dim, half=cfg.rotary_dim/2;
    int tp=m3_envi("M3_TP",0);
    int tp_attn=m3_envi("M3_TP_ATTN",tp),tp_sh=m3_envi("M3_TP_SHARED",tp),tp_ffn=m3_envi("M3_TP_FFN",tp),tp_head=m3_envi("M3_TP_HEAD",tp),tp_emb=m3_envi("M3_TP_EMBED",tp);
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
        { m3_ent*e=REQ(LN("self_attn.q_proj.weight")); if(e){ L->wq=(m3_tensor){m3_cp_rows(base,e,qh0*HD,qrows,H,2),NULL,M3_BF16,qrows,H}; used+=(size_t)qrows*H*2; } }
        { m3_ent*e=REQ(LN("self_attn.k_proj.weight")); if(e) L->wk=(m3_tensor){m3_cp_full(base,e),NULL,M3_BF16,KVD,H}; }
        { m3_ent*e=REQ(LN("self_attn.v_proj.weight")); if(e) L->wv=(m3_tensor){m3_cp_full(base,e),NULL,M3_BF16,KVD,H}; }
        { m3_ent*e=REQ(LN("self_attn.o_proj.weight")); if(e){ L->wo=(m3_tensor){m3_cp_cols(base,e,H,qh0*HD,qrows,QD,2),NULL,M3_BF16,H,qrows}; used+=(size_t)H*qrows*2; } }
        { m3_ent*e=REQ(LN("self_attn.q_norm.weight")); if(e) L->q_norm=m3_cp_full(base,e); }
        { m3_ent*e=REQ(LN("self_attn.k_norm.weight")); if(e) L->k_norm=m3_cp_full(base,e); }
        L->qh0=qh0; L->qh1=qh1;
        L->k_cache=calloc((size_t)cfg.max_pos*KVD,2); L->v_cache=calloc((size_t)cfg.max_pos*KVD,2);
        if(is_moe){
            { m3_ent*e=REQ(LN("self_attn.index_q_proj.weight")); if(e) L->idx_wq=(m3_tensor){m3_cp_full(base,e),NULL,M3_BF16,IQD,H}; }
            { m3_ent*e=REQ(LN("self_attn.index_k_proj.weight")); if(e) L->idx_wk=(m3_tensor){m3_cp_full(base,e),NULL,M3_BF16,ID,H}; }
            { m3_ent*e=REQ(LN("self_attn.index_q_norm.weight")); if(e) L->idx_q_norm=m3_cp_full(base,e); }
            { m3_ent*e=REQ(LN("self_attn.index_k_norm.weight")); if(e) L->idx_k_norm=m3_cp_full(base,e); }
            L->idx_k_cache=calloc((size_t)cfg.max_pos*ID,2);
            { m3_ent*e=REQ(LN("block_sparse_moe.gate.weight")); if(e) L->gate=(m3_tensor){m3_cp_full(base,e),NULL,M3_F32,cfg.n_experts,H}; }
            { m3_ent*e=REQ(LN("block_sparse_moe.e_score_correction_bias")); if(e) L->gate_bias=m3_cp_full(base,e); }
            { m3_ent*e=REQ(LN("block_sparse_moe.shared_experts.gate_proj.weight")); if(e){ L->sh_w1=(m3_tensor){m3_cp_rows(base,e,sh_r0,sh_rows,H,2),NULL,M3_BF16,sh_rows,H}; used+=(size_t)sh_rows*H*2; } }
            { m3_ent*e=REQ(LN("block_sparse_moe.shared_experts.up_proj.weight"));   if(e){ L->sh_w3=(m3_tensor){m3_cp_rows(base,e,sh_r0,sh_rows,H,2),NULL,M3_BF16,sh_rows,H}; used+=(size_t)sh_rows*H*2; } }
            { m3_ent*e=REQ(LN("block_sparse_moe.shared_experts.down_proj.weight")); if(e){ L->sh_w2=(m3_tensor){m3_cp_cols(base,e,H,sh_r0,sh_rows,cfg.moe_inter,2),NULL,M3_BF16,H,sh_rows}; used+=(size_t)H*sh_rows*2; } }
            L->sh_r0=sh_r0; L->sh_rows=sh_rows;
            int no=m3_n_owned(cfg.n_experts,ep_rank,ep_size); L->n_owned=no;
            L->owned_eid=malloc((size_t)(no>0?no:1)*sizeof(int));
            L->ex_w1=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor)); L->ex_w3=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor)); L->ex_w2=malloc((size_t)(no>0?no:1)*sizeof(m3_tensor));
            int s=0; for(int e2=0;e2<cfg.n_experts&&ok;e2++) if(e2%ep_size==ep_rank){ L->owned_eid[s]=e2;
                snprintf(nb,sizeof nb,"language_model.model.layers.%d.block_sparse_moe.experts.%d.w1.weight",l,e2); { m3_ent*e=REQ(nb); if(e){ L->ex_w1[s]=(m3_tensor){m3_cp_full(base,e),NULL,M3_BF16,cfg.moe_inter,H}; used+=e->nbytes; } }
                snprintf(nb,sizeof nb,"language_model.model.layers.%d.block_sparse_moe.experts.%d.w3.weight",l,e2); { m3_ent*e=REQ(nb); if(e){ L->ex_w3[s]=(m3_tensor){m3_cp_full(base,e),NULL,M3_BF16,cfg.moe_inter,H}; used+=e->nbytes; } }
                snprintf(nb,sizeof nb,"language_model.model.layers.%d.block_sparse_moe.experts.%d.w2.weight",l,e2); { m3_ent*e=REQ(nb); if(e){ L->ex_w2[s]=(m3_tensor){m3_cp_full(base,e),NULL,M3_BF16,H,cfg.moe_inter}; used+=e->nbytes; } }
                s++; }
        } else {
            { m3_ent*e=REQ(LN("mlp.gate_proj.weight")); if(e){ L->ff_gate=(m3_tensor){m3_cp_rows(base,e,ff_r0,ff_rows,H,2),NULL,M3_BF16,ff_rows,H}; used+=(size_t)ff_rows*H*2; } }
            { m3_ent*e=REQ(LN("mlp.up_proj.weight"));   if(e){ L->ff_up  =(m3_tensor){m3_cp_rows(base,e,ff_r0,ff_rows,H,2),NULL,M3_BF16,ff_rows,H}; used+=(size_t)ff_rows*H*2; } }
            { m3_ent*e=REQ(LN("mlp.down_proj.weight")); if(e){ L->ff_down=(m3_tensor){m3_cp_cols(base,e,H,ff_r0,ff_rows,cfg.dense_inter,2),NULL,M3_BF16,H,ff_rows}; used+=(size_t)H*ff_rows*2; } }
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
    m3_dummy=m3_envi("M3_DUMMY",0);
    if(m3_envi("M3_POOL",0)) m->pool=m3_g_pool=m3_pool_create(m->n_threads);  /* experimental pinned pool; default OpenMP */
    return m;
}

/* ===================== MSA: build the selected-position list for a sparse layer ===================== */
/* Returns nsel positions in sel[]; computes idx_q/idx_k, stores idx_k_cache[pos]. When the
 * block count is <= K+local+init (or MSA off) it returns the full causal range [0..pos]. */
static int m3_msa_select(m3_model*m,m3_layer*L,const float*xn,int pos,int msa_on,int*sel){
    const m3_config*c=&m->cfg;
    const int ID=c->msa_index_dim, IH=c->msa_n_index_heads, B=c->msa_block_size, half=c->rotary_dim/2;
    const float*cosp=&m->rope_cos[(size_t)pos*half], *sinp=&m->rope_sin[(size_t)pos*half];
    /* index projections (replicated): idx_q [IH*ID], idx_k [ID] (MQA, 1 head) */
    float*iq=m->s_idx_q, *ik=m->s_idx_k;
    m3_mv_bf16(iq,(uint16_t*)L->idx_wq.w,xn,IH*ID,c->hidden);
    m3_mv_bf16(ik,(uint16_t*)L->idx_wk.w,xn,ID,c->hidden);
    for(int h=0;h<IH;h++){ float*qh=iq+h*ID; m3_rmsnorm_head(qh,L->idx_q_norm,ID,c->norm_eps); m3_rope_head(qh,cosp,sinp,c->rotary_dim); }
    m3_rmsnorm_head(ik,L->idx_k_norm,ID,c->norm_eps); m3_rope_head(ik,cosp,sinp,c->rotary_dim);
    for(int i=0;i<ID;i++) L->idx_k_cache[(size_t)pos*ID+i]=m3_f2bf(ik[i]);

    int nblk=pos/B+1;
    int keep=c->msa_topk_blocks+c->msa_local_block+c->msa_init_block;
    if(!msa_on || nblk<=keep){ for(int t=0;t<=pos;t++) sel[t]=t; return pos+1; }
    /* per-block score = max over positions in block of sum_h dot(idx_q_h, idx_k_cache[t]) */
    float*bs=m->s_blk_score;
    for(int b=0;b<nblk;b++){
        int t0=b*B, t1=t0+B; if(t1>pos+1)t1=pos+1; float best=-1e30f;
        for(int t=t0;t<t1;t++){ const uint16_t*kt=L->idx_k_cache+(size_t)t*ID; float sc=0;
            for(int h=0;h<IH;h++){ const float*qh=iq+h*ID; float d=0; for(int i=0;i<ID;i++) d+=qh[i]*m3_bf2f(kt[i]); sc+=d; }
            if(sc>best)best=sc; }
        bs[b]=best;
    }
    char*selb=(char*)m->s_blk_sel;    /* reuse first nblk bytes as a block-selected bitmap */
    for(int b=0;b<nblk;b++) selb[b]=0;
    for(int b=0;b<c->msa_init_block && b<nblk;b++) selb[b]=1;                 /* init blocks */
    for(int b=nblk-c->msa_local_block;b<nblk;b++) if(b>=0) selb[b]=1;          /* local blocks */
    for(int pick=0;pick<c->msa_topk_blocks;pick++){                            /* top-K by score */
        int best=-1; float bv=-1e30f;
        for(int b=0;b<nblk;b++){ if(selb[b])continue; if(bs[b]>bv){bv=bs[b];best=b;} }
        if(best<0)break; selb[best]=1;
    }
    /* gather selected positions (causal) into sel[] AFTER reading the bitmap (separate region) */
    int nsel=0;
    for(int b=0;b<nblk;b++){ if(!selb[b])continue; int t0=b*B,t1=t0+B; if(t1>pos+1)t1=pos+1;
        for(int t=t0;t<t1;t++) sel[nblk+nsel++]=t; }       /* write past the nblk-byte bitmap */
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
        m3_mv_bf16(q,(uint16_t*)L->wq.w,xn,qrows,H);
        m3_mv_bf16(kb,(uint16_t*)L->wk.w,xn,KVD,H);
        m3_mv_bf16(vb,(uint16_t*)L->wv.w,xn,KVD,H);
        for(int hh=0;hh<nown;hh++){ float*qh=q+hh*HD; if(c->use_qk_norm) m3_rmsnorm_head(qh,L->q_norm,HD,c->norm_eps); m3_rope_head(qh,cosp,sinp,c->rotary_dim); }
        for(int kh=0;kh<KVH;kh++){ float*kk=kb+kh*HD; if(c->use_qk_norm) m3_rmsnorm_head(kk,L->k_norm,HD,c->norm_eps); m3_rope_head(kk,cosp,sinp,c->rotary_dim); }
        for(int i=0;i<KVD;i++){ L->k_cache[(size_t)pos*KVD+i]=m3_f2bf(kb[i]); L->v_cache[(size_t)pos*KVD+i]=m3_f2bf(vb[i]); }
        /* selected positions (MSA on sparse layers; full causal otherwise) */
        int*selp=m->s_blk_sel;
        int nsel; int sparse=is_moe;       /* M3: sparse layers == moe layers (L>=n_dense) */
        if(sparse) nsel=m3_msa_select(m,L,xn,pos,msa_on,selp);
        else { for(int t=0;t<=pos;t++) selp[t]=t; nsel=pos+1; }
        /* GQA attention over selected positions, owned heads -> attn[hh*HD] */
        for(int hh=0;hh<nown;hh++){ int h=qh0+hh; float*qh=q+hh*HD; int kvh=h/grp; float mx=-1e30f;
            for(int j=0;j<nsel;j++){ int t=selp[j]; const uint16_t*kt=L->k_cache+(size_t)t*KVD+kvh*HD; double d=0;
                for(int i=0;i<HD;i++) d+=(double)qh[i]*m3_bf2f(kt[i]); float s=(float)d*ascale; sc[j]=s; if(s>mx)mx=s; }
            double sum=0; for(int j=0;j<nsel;j++){ float e=expf(sc[j]-mx); sc[j]=e; sum+=e; }
            float inv=(float)(1.0/(sum>0?sum:1)); float*oh=attn+hh*HD; for(int i=0;i<HD;i++) oh[i]=0;
            for(int j=0;j<nsel;j++){ int t=selp[j]; float w=sc[j]*inv; const uint16_t*vt=L->v_cache+(size_t)t*KVD+kvh*HD; for(int i=0;i<HD;i++) oh[i]+=w*m3_bf2f(vt[i]); } }
        /* o_proj (wo cols = owned head cols); partial under TP_ATTN -> ar_cb sum */
        m3_mv_bf16(ao,(uint16_t*)L->wo.w,attn,H,qrows);
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
                m3_mv_bf16(m->s_exg,(uint16_t*)L->ex_w1[slot].w,h2,c->moe_inter,H);
                m3_mv_bf16(m->s_exu,(uint16_t*)L->ex_w3[slot].w,h2,c->moe_inter,H);
                for(int i=0;i<c->moe_inter;i++) m->s_exg[i]=m3_swiglu_oai(m->s_exg[i],m->s_exu[i],c->swiglu_alpha,c->swiglu_limit);
                m3_mv_bf16(m->s_moe,(uint16_t*)L->ex_w2[slot].w,m->s_exg,H,c->moe_inter);
                for(int i=0;i<H;i++) route[i]+=w*m->s_moe[i]; }
            /* shared expert: TP-sharded -> fold partial into route[] (one reduce); else replicated -> add after */
            m3_mv_bf16(m->s_shg,(uint16_t*)L->sh_w1.w,h2,L->sh_rows,H);
            m3_mv_bf16(m->s_shu,(uint16_t*)L->sh_w3.w,h2,L->sh_rows,H);
            for(int i=0;i<L->sh_rows;i++) m->s_shg[i]=m3_swiglu_oai(m->s_shg[i],m->s_shu[i],c->swiglu_alpha,c->swiglu_limit);
            m3_mv_bf16(m->s_sh,(uint16_t*)L->sh_w2.w,m->s_shg,H,L->sh_rows);
            if(tp_sh) for(int i=0;i<H;i++) route[i]+=m->s_sh[i];
            if(m->ar_cb) m->ar_cb(route,H,m->ar_ctx);     /* EP-sum routed (+ shared if TP) */
            for(int i=0;i<H;i++) x[i]+=route[i] + (tp_sh?0.0f:m->s_sh[i]);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            m3_mv_bf16(m->s_ff_g,(uint16_t*)L->ff_gate.w,h2,L->ff_rows,H);
            m3_mv_bf16(m->s_ff_u,(uint16_t*)L->ff_up.w,h2,L->ff_rows,H);
            for(int i=0;i<L->ff_rows;i++) m->s_ff_g[i]=m3_swiglu_oai(m->s_ff_g[i],m->s_ff_u[i],c->swiglu_alpha,c->swiglu_limit);
            m3_mv_bf16(m->s_ff,(uint16_t*)L->ff_down.w,m->s_ff_g,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(m->s_ff,H,m->ar_ctx);
            for(int i=0;i<H;i++) x[i]+=m->s_ff[i];
        }
    }
    /* head: vocab-shard partial logits -> (TP_HEAD) global argmax via ar_argmax, else full */
    float*h2=m->s_norm; m3_rmsnorm_gemma(h2,x,m->out_norm,H,c->norm_eps);
    int hrows=m->head.rows; m3_mv_bf16(m->s_logits,(uint16_t*)m->head.w,h2,hrows,H);
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
    float *logits,*sc;
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

static void m3_free_mstream(m3_model*m){
    m3_mstream*ms=(m3_mstream*)m->ms; if(!ms) return;
    free(ms->kc);free(ms->vc);free(ms->xn);free(ms->q);free(ms->k);free(ms->v);free(ms->attn);free(ms->o);
    free(ms->h2);free(ms->router);free(ms->route);free(ms->shg);free(ms->shu);free(ms->ffg);free(ms->ffu);
    free(ms->tmp2);free(ms->exg);free(ms->exu);free(ms->emoe);free(ms->logits);free(ms->sc);
    free(ms); m->ms=NULL;
}
static int m3_alloc_mstream(m3_model*m,int N){
    const m3_config*c=&m->cfg; int H=c->hidden,QD=m3_q_dim(c),KVD=m3_kv_dim(c),hrows=m->head.rows;
    m3_mstream*ms=calloc(1,sizeof *ms); if(!ms) return -1; ms->n=N;
    size_t per=(size_t)c->n_layers*c->max_pos*KVD;
    ms->kc=calloc((size_t)N*per,2); ms->vc=calloc((size_t)N*per,2);
    ms->xn=malloc((size_t)N*H*4); ms->q=malloc((size_t)N*QD*4); ms->k=malloc((size_t)N*KVD*4); ms->v=malloc((size_t)N*KVD*4);
    ms->attn=malloc((size_t)N*QD*4); ms->o=malloc((size_t)N*H*4); ms->h2=malloc((size_t)N*H*4);
    ms->router=malloc((size_t)N*c->n_experts*4); ms->route=malloc((size_t)N*H*4);
    ms->shg=malloc((size_t)N*c->moe_inter*4); ms->shu=malloc((size_t)N*c->moe_inter*4);
    ms->ffg=malloc((size_t)N*c->dense_inter*4); ms->ffu=malloc((size_t)N*c->dense_inter*4);
    ms->tmp2=malloc((size_t)N*H*4);
    ms->exg=malloc((size_t)c->moe_inter*4); ms->exu=malloc((size_t)c->moe_inter*4); ms->emoe=malloc((size_t)H*4);
    ms->logits=malloc((size_t)N*hrows*4); ms->sc=malloc((size_t)N*c->max_pos*4);
    if(!ms->kc||!ms->vc||!ms->logits){ m->ms=ms; m3_free_mstream(m); return -1; }
    m->ms=ms; return 0;
}

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
        m3_gemm_bf16(ms->q,(uint16_t*)L->wq.w,ms->xn,N,qrows,H);
        m3_gemm_bf16(ms->k,(uint16_t*)L->wk.w,ms->xn,N,KVD,H);
        m3_gemm_bf16(ms->v,(uint16_t*)L->wv.w,ms->xn,N,KVD,H);
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
        m3_gemm_bf16(ms->o,(uint16_t*)L->wo.w,ms->attn,N,H,qrows);
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
            /* owned-expert compute: serial over streams (shared expert scratch + parallel matvec) */
            for(int t=0;t<N;t++){ int*sel=sel_all+t*na; float*sw=selw_all+t*na; float*h2t=ms->h2+(size_t)t*H,*rt=ms->route+(size_t)t*H;
                for(int a=0;a<na;a++){ int e=sel[a]; if(e%m->ep_size!=m->ep_rank) continue; int slot=e/m->ep_size; float w=sw[a];
                    m3_mv_bf16(ms->exg,(uint16_t*)L->ex_w1[slot].w,h2t,c->moe_inter,H);
                    m3_mv_bf16(ms->exu,(uint16_t*)L->ex_w3[slot].w,h2t,c->moe_inter,H);
                    for(int i=0;i<c->moe_inter;i++) ms->exg[i]=m3_swiglu_oai(ms->exg[i],ms->exu[i],c->swiglu_alpha,c->swiglu_limit);
                    m3_mv_bf16(ms->emoe,(uint16_t*)L->ex_w2[slot].w,ms->exg,H,c->moe_inter);
                    for(int i=0;i<H;i++) rt[i]+=w*ms->emoe[i]; } }
            m3_gemm_bf16(ms->shg,(uint16_t*)L->sh_w1.w,ms->h2,N,L->sh_rows,H);
            m3_gemm_bf16(ms->shu,(uint16_t*)L->sh_w3.w,ms->h2,N,L->sh_rows,H);
            for(size_t i=0;i<(size_t)N*L->sh_rows;i++) ms->shg[i]=m3_swiglu_oai(ms->shg[i],ms->shu[i],c->swiglu_alpha,c->swiglu_limit);
            m3_gemm_bf16(ms->tmp2,(uint16_t*)L->sh_w2.w,ms->shg,N,H,L->sh_rows);   /* shared-out [N,H] */
            if(tp_sh) for(size_t i=0;i<(size_t)N*H;i++) ms->route[i]+=ms->tmp2[i];
            if(m->ar_cb) m->ar_cb(ms->route,N*H,m->ar_ctx);                        /* ONE reduce for N tokens */
            for(size_t i=0;i<(size_t)N*H;i++) X[i]+=ms->route[i] + (tp_sh?0.0f:ms->tmp2[i]);
        } else {
            const int tp_ffn=(L->ff_rows<c->dense_inter);
            m3_gemm_bf16(ms->ffg,(uint16_t*)L->ff_gate.w,ms->h2,N,L->ff_rows,H);
            m3_gemm_bf16(ms->ffu,(uint16_t*)L->ff_up.w,ms->h2,N,L->ff_rows,H);
            for(size_t i=0;i<(size_t)N*L->ff_rows;i++) ms->ffg[i]=m3_swiglu_oai(ms->ffg[i],ms->ffu[i],c->swiglu_alpha,c->swiglu_limit);
            m3_gemm_bf16(ms->tmp2,(uint16_t*)L->ff_down.w,ms->ffg,N,H,L->ff_rows);
            if(tp_ffn && m->ar_cb) m->ar_cb(ms->tmp2,N*H,m->ar_ctx);
            for(size_t i=0;i<(size_t)N*H;i++) X[i]+=ms->tmp2[i];
        }
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<N;t++) m3_rmsnorm_gemma(ms->h2+(size_t)t*H, X+(size_t)t*H, m->out_norm, H, c->norm_eps);
    int hrows=m->head.rows; m3_gemm_bf16(ms->logits,(uint16_t*)m->head.w,ms->h2,N,hrows,H);
    for(int t=0;t<N;t++){ float*lg=ms->logits+(size_t)t*hrows; int la=0; float bv=lg[0]; for(int i=1;i<hrows;i++) if(lg[i]>bv){bv=lg[i];la=i;}
        int32_t gidx=m->head_r0+la; float gval=bv;
        if(hrows<c->vocab && m->ar_argmax_cb) m->ar_argmax_cb(&gval,&gidx,m->ar_argmax_ctx);
        out[t]=gidx; }
}

#endif /* M3_IMPL_H */
