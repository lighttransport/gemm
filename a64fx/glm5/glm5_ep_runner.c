/*
 * glm5_ep_runner.c - GLM-5.2 (text) expert-parallel runner, A64FX.
 *
 * Multi-node EP decode/prefill over pure uTofu (mpiexec places one rank/node).
 * Each rank owns experts e with e % N == rank; router/shared/attention/dense/head
 * are replicated. The per-MoE-layer combine is one tp_allreduce_sum over the routed
 * partial [hidden]. Default path uses staged real BF16 weights from /local/glm5.
 * GLM5_REAL=0 keeps the synthetic structural benchmark path available.
 *
 * Build:  make -C a64fx/llm glm5_ep_runner   (rule mirrors ds4f_ep_runner; -ltofucom)
 * Run:    mpiexec -n N [-vcoordfile vc] build/glm5_ep_runner   (after tofu_topo_helper)
 *
 * Env: LLM_THREADS, GLM5_PREFILL(8), GLM5_DECODE(16), GLM5_MAXPOS(2048), GLM5_LAYERS(0=full),
 *      GLM5_EXPERTS(0=full 256).
 */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <utofu.h>

#define GLM5_IMPL
#include "glm5.h"
#include "glm5_impl.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define MAX_NODES 512   /* >= largest EP run (384-node prefill); arrays are O(MAX_NODES) tiny */
#define RUN_STAG  DEMO_STAG
#define WAIT_TIMEOUT_SEC 300.0

static FILE *g_log = NULL;
static void logmsg(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    if (g_log) { va_list ap2; va_copy(ap2, ap); vfprintf(g_log, fmt, ap2); va_end(ap2); fflush(g_log); }
    vfprintf(stderr, fmt, ap); va_end(ap);
}
static void die(const char *what, int rc) { logmsg("FATAL: %s (rc=%d)\n", what, rc); exit(1); }
static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void prof_snapshot(glm5_model*m,double dst[GLM5_NPHASE]){
    for(int i=0;i<GLM5_NPHASE;i++) dst[i]=m->prof[i];
}
static void prof_log_delta(const char*label,const double a[GLM5_NPHASE],
                           const double b[GLM5_NPHASE],int ntok,double wall,double comm){
    double sum=0.0;
    for(int i=0;i<GLM5_NPHASE;i++) if(b[i]>a[i]) sum+=b[i]-a[i];
    logmsg("PROFILE %s wall=%.6f s tokens=%d measured=%.6f s comm=%.6f s\n",label,wall,ntok,sum,comm);
    for(int i=0;i<GLM5_NPHASE;i++){
        double d=b[i]-a[i];
        if(d<=0.0) continue;
        logmsg("PROFILE %s %-10s total=%.6f s ms/tok=%.3f pct=%.1f\n",
               label,glm5_prof_names[i],d,ntok>0?d*1e3/ntok:0.0,sum>0?100.0*d/sum:0.0);
    }
}
static void prof_file_delta(FILE*f,const char*label,const double a[GLM5_NPHASE],
                            const double b[GLM5_NPHASE],int ntok,double wall,double comm){
    double sum=0.0;
    for(int i=0;i<GLM5_NPHASE;i++) if(b[i]>a[i]) sum+=b[i]-a[i];
    fprintf(f,"PROFILE %s wall=%.6f s tokens=%d measured=%.6f s comm=%.6f s\n",label,wall,ntok,sum,comm);
    for(int i=0;i<GLM5_NPHASE;i++){
        double d=b[i]-a[i];
        if(d<=0.0) continue;
        fprintf(f,"PROFILE %s %-10s total=%.6f s ms/tok=%.3f pct=%.1f\n",
                label,glm5_prof_names[i],d,ntok>0?d*1e3/ntok:0.0,sum>0?100.0*d/sum:0.0);
    }
}
static int envi(const char*k,int d){ const char*v=getenv(k); return (v&&*v)?atoi(v):d; }
static size_t rss_bytes(void){ FILE*f=fopen("/proc/self/statm","r"); if(!f)return 0; long tot=0,res=0; if(fscanf(f,"%ld %ld",&tot,&res)!=2)res=0; fclose(f); return (size_t)res*(size_t)sysconf(_SC_PAGESIZE); }

/* deterministic synthetic activations, identical on every rank */
static uint64_t sm_state;
static double sm_next(void){ sm_state+=0x9E3779B97F4A7C15ull; uint64_t z=sm_state;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull; z^=z>>31; return (double)(z>>11)/(double)(1ull<<53); }

/* ---- topology (tofu_topo.txt) ---- */
static const char *topo_path(void){ const char*t=getenv("TOFU_TOPO_PATH"); return (t&&*t)?t:TOPO_PATH; }
static int read_topo(uint8_t coords[][TOFU_NCOORDS]){
    const char*tp=topo_path(); FILE*f=fopen(tp,"r");
    if(!f){ fprintf(stderr,"cannot open %s (run tofu_topo_helper first)\n",tp); exit(1); }
    int n=0; char line[256];
    while(fgets(line,sizeof line,f)){
        if(line[0]=='#'||line[0]=='\n') continue;
        if(n>=MAX_NODES){ fprintf(stderr,"too many nodes\n"); exit(1); }
        unsigned r,cc[TOFU_NCOORDS];
        if(sscanf(line,"%u %u %u %u %u %u %u",&r,&cc[0],&cc[1],&cc[2],&cc[3],&cc[4],&cc[5])!=7){ fprintf(stderr,"malformed line: %s",line); exit(1); }
        if((int)r!=n){ fprintf(stderr,"%s ranks out of order\n",tp); exit(1); }
        for(int k=0;k<TOFU_NCOORDS;k++) coords[n][k]=(uint8_t)cc[k]; n++;
    }
    fclose(f); if(n<1){ fprintf(stderr,"%s lists %d node(s)\n",tp,n); exit(1); } return n;
}

/* ---- uTofu state ---- */
static int             N, MyRank;
static char           *Region;
static size_t          SEND_OFF, BAR_BASE, SlotSend, SlotB;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t   Base;
static utofu_vcq_id_t  PeerVcq[MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static uint64_t        Bt = 1;

static inline size_t bar_recv_off(int s){ return BAR_BASE+(size_t)s*SlotB; }
static inline size_t bar_go_off(void)   { return BAR_BASE+(size_t)N*SlotB; }
static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len, int drain){
    int rc; void*cb;
    for(;;){ rc=utofu_put(Vcq,pv,s,d,len,0,FLAGS,NULL); if(rc!=UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq,0,&cb); }
    if(rc!=UTOFU_SUCCESS) die("utofu_put",rc);
    if(drain){ do{ rc=utofu_poll_tcq(Vcq,0,&cb);}while(rc==UTOFU_ERR_NOT_FOUND); if(rc!=UTOFU_SUCCESS) die("utofu_poll_tcq",rc); }
}
static void wait_ge(volatile uint64_t*q, uint64_t v, const char*what){ double ts=now_sec(); while(*q<v) if(now_sec()-ts>WAIT_TIMEOUT_SEC) die(what,-1); }
static void barrier_robust(int robust){
    uint64_t t=++Bt; char*sb=Region+SEND_OFF;
    if(MyRank==0){
        for(int s=1;s<N;s++) wait_ge((volatile uint64_t*)(Region+bar_recv_off(s)),t,"barrier fan-in");
        for(int s=1;s<N;s++){ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[s],Base+SEND_OFF,PeerBase[s]+bar_go_off(),8,1); }
    } else {
        volatile uint64_t*go=(volatile uint64_t*)(Region+bar_go_off()); double ts=now_sec();
        do{ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[0],Base+SEND_OFF,PeerBase[0]+bar_recv_off(MyRank),8,1);
            if(!robust){ wait_ge(go,t,"barrier release"); break; }
            for(int a=0;a<50&&*go<t;a++) usleep(2000);
            if(now_sec()-ts>WAIT_TIMEOUT_SEC) die("bootstrap barrier timeout",-1);
        }while(*go<t);
    }
}
/* Always robust: at large rank counts the N-1 -> 0 fan-in is an incast that congests rank 0's
 * TNI, so single-shot (non-robust) puts drop and the barrier hangs (seen as a 384-node
 * "barrier fan-in"/"barrier release" timeout). The robust path retries the fan-in put until
 * release, which is why the bootstrap barriers (barrier_robust(1)) already survive 384 ranks.
 * No barrier() is on a per-token hot path (those use tp_allreduce, robust by default), so the
 * retry's usleep granularity costs nothing here. */
static void barrier(void){ barrier_robust(1); }

/* ---- EP combine all-reduce callback ---- */
static double g_ar_secs=0.0; static long g_ar_calls=0, g_ar_frags=0;
static void ep_ar_callback(float*buf,int count,void*ctx){
    tp_comm*c=(tp_comm*)ctx; int mc=c->max_count>0?c->max_count:count; double t0=now_sec();
    long nf=0;
    for(int off=0;off<count;){ int n=count-off; if(n>mc)n=mc; tp_allreduce_sum(c,buf+off,n); off+=n; nf++; }
    g_ar_secs+=now_sec()-t0; g_ar_calls++;
    g_ar_frags+=nf;
}
/* (val,global-idx) argmax all-reduce (TP_HEAD vocab-shard logits merge) */
static void ep_argmax_callback(float*val,int32_t*idx,void*ctx){
    double t0=now_sec(); tp_allreduce_argmax((tp_comm*)ctx,val,idx); g_ar_secs+=now_sec()-t0; g_ar_calls++;
    g_ar_frags++;
}
/* ---- CP (context-parallel KV) callbacks ---- */
/* all-reduce MAX of per-block index scores so every rank derives the same global top-k. */
static void ep_blk_reduce(float*scores,int nblk,void*ctx){
    tp_comm*c=(tp_comm*)ctx; int mc=c->max_count>0?c->max_count:nblk; double t0=now_sec();
    long nf=0;
    for(int off=0;off<nblk;){ int n=nblk-off; if(n>mc)n=mc; tp_allreduce_max(c,scores+off,n); off+=n; nf++; }
    g_ar_secs+=now_sec()-t0; g_ar_calls++;
    g_ar_frags+=nf;
}
/* flash-combine the per-rank partial attention (out unnormalized, max, sumexp) across ranks:
 *   gmx=max_r mx_r;  s_r=exp(mx_r-gmx);  out = (sum_r s_r*out_r)/(sum_r s_r*se_r). */
static float *g_kvbuf=NULL;
static void ep_kv_combine(float*out,float*mx,float*se,int nh,int hd,void*ctx){
    tp_comm*c=(tp_comm*)ctx; int mc=c->max_count>0?c->max_count:1; double t0=now_sec();
    float gmx[64]; for(int h=0;h<nh;h++) gmx[h]=mx[h];
    tp_allreduce_max(c,gmx,nh);                              /* global per-head max */
    long nf=1;
    int cnt=nh+nh*hd; float*buf=g_kvbuf;                     /* [se(nh) | out(nh*hd)] rescaled */
    for(int h=0;h<nh;h++){ float s=expf(mx[h]-gmx[h]); buf[h]=se[h]*s;
        float*o=out+h*hd,*b=buf+nh+(size_t)h*hd; for(int i=0;i<hd;i++) b[i]=o[i]*s; }
    for(int off=0;off<cnt;){ int n=cnt-off; if(n>mc)n=mc; tp_allreduce_sum(c,buf+off,n); off+=n; nf++; }
    for(int h=0;h<nh;h++){ float inv=1.0f/(buf[h]>0?buf[h]:1.0f);
        float*o=out+h*hd,*b=buf+nh+(size_t)h*hd; for(int i=0;i<hd;i++) o[i]=b[i]*inv; }
    g_ar_secs+=now_sec()-t0; g_ar_calls++;
    g_ar_frags+=nf;
}

/* ---- comm-overlap driver thread: only it touches uTofu during an overlapped reduce ----
 * The batched MoE issues the routed-expert all-reduce here (ar_async_start), computes the
 * replicated shared expert on the OpenMP pool, then ar_wait. The route reduce (comm thread)
 * runs concurrently with the shared GEMMs (compute threads) -> comm hidden under compute.
 * Safe because uTofu use is never concurrent: the per-layer o_proj reduce (main thread) and
 * the route reduce (this thread) are sequential, and the main thread does no uTofu while the
 * shared expert computes. */
static tp_comm *g_comm_ctx=NULL;
static _Atomic int g_comm_go=0, g_comm_done=1, g_comm_stop=0;
static float *g_comm_buf=NULL; static int g_comm_count=0;
static pthread_t g_comm_th;
static void* comm_driver(void *a){ (void)a;
    for(;;){
        while(!atomic_load_explicit(&g_comm_go,memory_order_acquire) && !atomic_load_explicit(&g_comm_stop,memory_order_acquire))
            __asm__ __volatile__("yield":::"memory");
        if(atomic_load_explicit(&g_comm_stop,memory_order_acquire)) break;
        atomic_store_explicit(&g_comm_go,0,memory_order_relaxed);
        ep_ar_callback(g_comm_buf,g_comm_count,g_comm_ctx);
        atomic_store_explicit(&g_comm_done,1,memory_order_release);
    }
    return NULL;
}
static void ar_async_start_cb(float *buf,int count,void *ctx){
    g_comm_buf=buf; g_comm_count=count; g_comm_ctx=(tp_comm*)ctx;
    atomic_store_explicit(&g_comm_done,0,memory_order_relaxed);
    atomic_store_explicit(&g_comm_go,1,memory_order_release);
}
static void ar_wait_cb(void *ctx){ (void)ctx; while(!atomic_load_explicit(&g_comm_done,memory_order_acquire)) __asm__ __volatile__("yield":::"memory"); }

/* token id -> input embedding (the forward's first op is input_layernorm, so the raw
 * widened embed row is the activation it expects). Under TP_EMBED the owner fills its
 * vocab-shard row and the ar_cb SUMs (zeros elsewhere) -> full embedding, bit-exact. */
static void embed_lookup(glm5_model*m,int tok,float*x){
    int H=m->cfg.hidden;
    if(tok<0||tok>=m->cfg.vocab) tok=0;
    if(m->emb_rows<m->cfg.vocab){
        for(int i=0;i<H;i++) x[i]=0.f;
        if(tok>=m->emb_r0 && tok<m->emb_r0+m->emb_rows){
            const uint16_t*row=m->embed+(size_t)(tok-m->emb_r0)*H;
            for(int i=0;i<H;i++) x[i]=glm5_bf2f(row[i]);
        }
        if(m->ar_cb) m->ar_cb(x,H,m->ar_ctx);
        return;
    }
    const uint16_t*row=m->embed+(size_t)tok*H;
    for(int i=0;i<H;i++) x[i]=glm5_bf2f(row[i]);
}
#define GLM5_EOS_ID0 154820
#define GLM5_EOS_ID1 154827
#define GLM5_EOS_ID2 154829

typedef struct {
    int *ids;
    int n;
} id_prompt;

typedef struct {
    glm5_model *m;
    float *x;
    int *gen;
    int req, n_prompt, ng, cur, done, nan;
} cb_slot;

static int parse_id_line(char *line,int **out_ids,int *out_n){
    char *p=line, *end=NULL;
    while(*p==' '||*p=='\t'||*p=='\r'||*p=='\n') p++;
    if(!*p || *p=='#') return 0;
    int cap=64,n=0,*ids=glm5_amalloc((size_t)cap*sizeof(int));
    while(*p){
        long v=strtol(p,&end,10);
        if(end==p) break;
        if(n>=cap){ cap*=2; ids=realloc(ids,(size_t)cap*sizeof(int)); }
        ids[n++]=(int)v;
        p=end;
        while(*p==' '||*p=='\t'||*p=='\r'||*p=='\n') p++;
    }
    if(n<1){ glm5_afree(ids); return 0; }
    *out_ids=ids; *out_n=n; return 1;
}

static int load_prompt_batch(const char*path,id_prompt **out){
    FILE*f=fopen(path,"r"); if(!f) return -1;
    int cap=16,n=0; id_prompt *ps=glm5_amalloc((size_t)cap*sizeof(id_prompt));
    char *line=NULL; size_t linecap=0;
    while(getline(&line,&linecap,f)>0){
        int *ids=NULL, ni=0;
        if(!parse_id_line(line,&ids,&ni)) continue;
        if(n>=cap){ cap*=2; ps=realloc(ps,(size_t)cap*sizeof(id_prompt)); }
        ps[n++]=(id_prompt){ids,ni};
    }
    free(line); fclose(f); *out=ps; return n;
}

static void prof_sum_models(cb_slot*s,int ns,double dst[GLM5_NPHASE]){
    for(int i=0;i<GLM5_NPHASE;i++) dst[i]=0.0;
    for(int j=0;j<ns;j++) if(s[j].m) for(int i=0;i<GLM5_NPHASE;i++) dst[i]+=s[j].m->prof[i];
}

static void cbatch_write_req(const char*prefix,int req,const int*gen,int ng){
    if(MyRank!=0 || !prefix || !*prefix) return;
    char path[512]; snprintf(path,sizeof path,"%s_%03d.txt",prefix,req);
    FILE*f=fopen(path,"w");
    if(!f) return;
    for(int i=0;i<ng;i++) fprintf(f,"%d%s",gen[i],i+1<ng?" ":"\n");
    fclose(f);
}

static int cbatch_start(cb_slot*s,const id_prompt*p,int req,int max_new,int C,double *prefill_sec,double *prefill_ar,long *prefill_calls){
    s->req=req; s->n_prompt=p->n; s->ng=0; s->done=0; s->nan=0; s->cur=0;
    g_ar_secs=0; g_ar_calls=0; g_ar_frags=0;
    double t0=now_sec();
    int last=-1;
    for(int i=0;i<p->n;i++){ embed_lookup(s->m,p->ids[i],s->x); last=glm5_forward_token(s->m,s->x,i); }
    double dt=now_sec()-t0;
    *prefill_sec+=dt; *prefill_ar+=g_ar_secs; *prefill_calls+=g_ar_calls;
    s->cur=last;
    (void)max_new; (void)C;
    return 0;
}

static int run_cbatch(glm5_model*root,const char*batch_file,const char*out_prefix,int max_new,int C){
    id_prompt *prompts=NULL;
    int n_req=load_prompt_batch(batch_file,&prompts);
    if(n_req<1) die("empty GLM5_CBATCH_PROMPTS",-1);
    int slots=envi("GLM5_CBATCH_SLOTS",n_req);
    if(slots<1) slots=1; if(slots>n_req) slots=n_req;
    if(MyRank==0) logmsg("cbatch: requests=%d slots=%d max_new=%d max_pos=%d prompts=%s\n",
                         n_req,slots,max_new,root->cfg.max_pos,batch_file);
    for(int r=0;r<n_req;r++) if(prompts[r].n+max_new>root->cfg.max_pos)
        die("cbatch prompt+max_new exceeds max_pos",-1);

    cb_slot *S=glm5_acalloc((size_t)slots,sizeof(cb_slot));
    for(int s=0;s<slots;s++){
        S[s].m=glm5_clone_runtime(root);
        if(!S[s].m) die("glm5_clone_runtime",-1);
        S[s].x=glm5_amalloc((size_t)C*4);
        S[s].gen=glm5_amalloc((size_t)(max_new>0?max_new:1)*sizeof(int));
        S[s].req=-1;
    }

    double prof0[GLM5_NPHASE], prof1[GLM5_NPHASE];
    prof_sum_models(S,slots,prof0);
    double pf_sec=0.0,pf_ar=0.0,svc_ar=0.0; long pf_calls=0,svc_calls=0;
    int next=0,done=0,active=0,total_gen=0,total_nan=0,total_prompt=0;
    for(int s=0;s<slots && next<n_req;s++,next++){
        total_prompt+=prompts[next].n;
        cbatch_start(&S[s],&prompts[next],next,max_new,C,&pf_sec,&pf_ar,&pf_calls);
        active++;
    }
    barrier();
    g_ar_secs=0; g_ar_calls=0; g_ar_frags=0;
    double svc0=now_sec();
    while(done<n_req){
        int progressed=0;
        for(int s=0;s<slots;s++){
            cb_slot *q=&S[s];
            if(q->req<0) continue;
            progressed=1;
            if(q->ng<max_new) q->gen[q->ng++]=q->cur;
            int eos=(q->cur==GLM5_EOS_ID0||q->cur==GLM5_EOS_ID1||q->cur==GLM5_EOS_ID2);
            if(eos || q->ng>=max_new){
                total_gen+=q->ng; total_nan+=q->nan;
                cbatch_write_req(out_prefix,q->req,q->gen,q->ng);
                if(MyRank==0){
                    char buf[3000]; int o=0;
                    for(int i=0;i<q->ng&&o<2900;i++) o+=snprintf(buf+o,sizeof(buf)-o,"%d ",q->gen[i]);
                    logmsg("CBATCH_IDS req=%d n=%d %s\n",q->req,q->ng,buf);
                }
                q->req=-1; done++; active--;
                if(next<n_req){
                    total_prompt+=prompts[next].n;
                    double save_ar=g_ar_secs; long save_calls=g_ar_calls, save_frags=g_ar_frags;
                    cbatch_start(q,&prompts[next],next,max_new,C,&pf_sec,&pf_ar,&pf_calls);
                    g_ar_secs=save_ar; g_ar_calls=save_calls; g_ar_frags=save_frags;
                    next++; active++;
                }
                continue;
            }
            embed_lookup(q->m,q->cur,q->x);
            q->cur=glm5_forward_token(q->m,q->x,q->n_prompt+q->ng-1);
            for(int i=0;i<C;i++) if(!(q->x[i]==q->x[i])) q->nan++;
        }
        if(!progressed && active==0 && next<n_req){
            total_prompt+=prompts[next].n;
            double save_ar=g_ar_secs; long save_calls=g_ar_calls, save_frags=g_ar_frags;
            cbatch_start(&S[0],&prompts[next],next,max_new,C,&pf_sec,&pf_ar,&pf_calls);
            g_ar_secs=save_ar; g_ar_calls=save_calls; g_ar_frags=save_frags;
            next++; active++;
        }
    }
    double svc_dt=now_sec()-svc0; svc_ar=g_ar_secs; svc_calls=g_ar_calls;
    barrier();
    prof_sum_models(S,slots,prof1);
    if(MyRank==0){
        logmsg("cbatch: prefill %d tok %.2f tok/s comm %.1f%% calls=%ld\n",
               total_prompt,pf_sec>0?total_prompt/pf_sec:0.0,pf_sec>0?100.0*pf_ar/pf_sec:0.0,pf_calls);
        logmsg("cbatch: service decode %d tok %.2f agg tok/s %.2f tok/s/slot comm %.1f%% calls=%ld NaNs=%d\n",
               total_gen,svc_dt>0?total_gen/svc_dt:0.0,svc_dt>0?total_gen/(svc_dt*slots):0.0,
               svc_dt>0?100.0*svc_ar/svc_dt:0.0,svc_calls,total_nan);
        prof_log_delta("cbatch_total",prof0,prof1,total_prompt+total_gen,pf_sec+svc_dt,pf_ar+svc_ar);
        logmsg("SENTINEL glm5_cbatch_%dn=done\n",N);
    }
    for(int s=0;s<slots;s++){ glm5_afree(S[s].gen); glm5_afree(S[s].x); glm5_free(S[s].m); }
    glm5_afree(S);
    for(int r=0;r<n_req;r++) glm5_afree(prompts[r].ids);
    glm5_afree(prompts);
    return 0;
}

int main(void){
    int rc;
    int n_threads=envi("LLM_THREADS",12), n_cmgs=envi("GLM5_CMGS",4);
    int prefill=envi("GLM5_PREFILL",8), maxgen=envi("GLM5_DECODE",16), maxpos=envi("GLM5_MAXPOS",2048);
    int prefill_only=envi("GLM5_PREFILL_ONLY",0), prefill_synth=envi("GLM5_PREFILL_SYNTH",0);
    if(prefill_synth>0) prefill=prefill_synth;
    if(prefill_only) maxgen=0;
    int start_pos=envi("GLM5_START_POS",0); if(start_pos<0) start_pos=0;
    int layers=envi("GLM5_LAYERS",0), nexp=envi("GLM5_EXPERTS",0);
    int mstream=envi("GLM5_MSTREAM",1); if(mstream<1)mstream=1; if(mstream>64)mstream=64;
    const char*ar_env=getenv("GLM5_AR_TOKENS");
    int ar_tokens=(ar_env&&*ar_env)?atoi(ar_env):0;
    int ar_auto_cap=envi("GLM5_AR_AUTO_CAP",256);
    int pchunk0=envi("GLM5_PCHUNK",0);
    if(!ar_env && (envi("GLM5_CP",0) || maxpos>65536)){
        ar_tokens=1;  /* small registered slots for long-context/CP stability */
    } else {
        if(ar_tokens<mstream) ar_tokens=mstream;
        if(pchunk0>ar_tokens) ar_tokens=pchunk0;
        if(!ar_env && ar_auto_cap>0 && ar_tokens>ar_auto_cap) ar_tokens=ar_auto_cap;
    }
    if(ar_tokens<1) ar_tokens=1;
    int ar_hard_cap=envi("GLM5_AR_HARD_CAP",64);
    if(ar_hard_cap>0 && ar_tokens>ar_hard_cap) ar_tokens=ar_hard_cap;

    utofu_tni_id_t*tni_ids=NULL; size_t num_tnis=0;
    rc=utofu_get_onesided_tnis(&tni_ids,&num_tnis); if(rc!=UTOFU_SUCCESS) die("utofu_get_onesided_tnis",rc);
    if(num_tnis<1) die("no onesided TNIs",-1);
    uint8_t my_coords[TOFU_NCOORDS]={0};
    rc=utofu_query_my_coords(my_coords); if(rc!=UTOFU_SUCCESS) die("utofu_query_my_coords",rc);
    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N=read_topo(topo);
    int rpn=envi("GLM5_RANKS_PER_NODE",1); if(rpn<1) rpn=1;
    MyRank=-1;
    if(rpn>1){
        /* multiple ranks/node share the same Tofu node coords, so coord-matching is ambiguous;
         * take MyRank from the MPI rank (topo is rank-ordered, validated by read_topo). */
        const char*rk[]={"PMIX_RANK","OMPI_COMM_WORLD_RANK","PMI_RANK","GLM5_EP_RANK",NULL};
        for(int i=0;rk[i];i++){ const char*e=getenv(rk[i]); if(e&&*e){ MyRank=atoi(e); break; } }
        if(MyRank<0||MyRank>=N) die("GLM5_RANKS_PER_NODE>1: no valid MPI rank env",-1);
        if(memcmp(topo[MyRank],my_coords,TOFU_NCOORDS)!=0) die("MPI rank vs topo coords mismatch",-1);
    } else {
        for(int r=0;r<N;r++) if(memcmp(topo[r],my_coords,TOFU_NCOORDS)==0) MyRank=r;
    }
    if(MyRank==-1){ fprintf(stderr,"my coords not in %s\n",topo_path()); exit(1); }

    { char en[64]; snprintf(en,sizeof en,"glm5_ep_stderr_rank%02d.txt",MyRank); if(!freopen(en,"w",stderr)){} setvbuf(stderr,NULL,_IOLBF,0); }
    if(MyRank==0) g_log=fopen("glm5_ep_rank00.txt","w");

    int ep_rank=MyRank, ep_size=N;
    glm5_config cfg=glm5_default_config(); cfg.max_pos=maxpos;
    if(layers>0) cfg.n_layers=layers;
    if(nexp>0)   cfg.n_experts=nexp;
    if(cfg.n_active>cfg.n_experts) cfg.n_active=cfg.n_experts;
    if(start_pos+prefill+maxgen>cfg.max_pos) die("start_pos+prefill+maxgen exceeds max_pos",-1);

    int no=glm5_n_owned(cfg.n_experts,ep_rank,ep_size);
    size_t arena_est=glm5_arena_size(&cfg,ep_rank,ep_size);
    if(MyRank==0)
        logmsg("=== GLM5 EP synthetic runner: %d ranks ===\n"
               "layers=%d hidden=%d experts=%d active=%d owned~%d  heads=%d/%d head_dim=%d\n"
               "threads=%d start_pos=%d prefill=%d decode=%d max_pos=%d  arena~%.2f GB/node (synth malloc)\n",
               N,cfg.n_layers,cfg.hidden,cfg.n_experts,cfg.n_active,no,
               cfg.n_heads,cfg.n_kv_heads,cfg.head_dim,n_threads,start_pos,prefill,maxgen,maxpos,
               arena_est/(1024.0*1024.0*1024.0));
    if(MyRank==0){ FILE*mf=fopen("/proc/meminfo","r"); if(mf){ char line[128];
        while(fgets(line,sizeof line,mf)) if(!strncmp(line,"MemTotal",8)||!strncmp(line,"MemFree",7)||!strncmp(line,"MemAvailable",12)){
            for(char*p=line;*p;p++) if(*p=='\n')*p=0; logmsg("NODE_MEMINFO %s\n",line);} fclose(mf);} }

    int real_weights=envi("GLM5_REAL",1);
    const char*blob_dir=getenv("GLM5_STAGE_DIR");
    double ta0=now_sec();
    glm5_model*m=real_weights ? glm5_load_real(cfg,ep_rank,ep_size,blob_dir,n_threads,n_cmgs)
                            : glm5_alloc_synth(cfg,ep_rank,ep_size,n_threads,n_cmgs);
    if(!m){ fprintf(stderr,"rank %d: model %s failed\n",MyRank,real_weights?"load":"alloc"); exit(1); }
    double ta1=now_sec();
    { char tn[64]; snprintf(tn,sizeof tn,"glm5_ep_load_rank%02d.txt",MyRank); FILE*tf=fopen(tn,"w");
      if(tf){ fprintf(tf,"rank %d: alloc=%.2fs arena_used=%.2f GB RSS=%.2f GB owned=%d/layer\n",MyRank,ta1-ta0,m->arena_used/1e9,rss_bytes()/1e9,no); fclose(tf);} }

    /* ---- barrier region + VCQ ---- */
    SlotSend=DEMO_CACHE_LINE; SlotB=DEMO_CACHE_LINE; SEND_OFF=0; BAR_BASE=SlotSend;
    size_t region_sz=BAR_BASE+(size_t)(N+1)*SlotB;
    if(posix_memalign((void**)&Region,DEMO_CACHE_LINE,region_sz)!=0) die("posix_memalign",-1);
    memset(Region,0,region_sz);
    /* Co-located ranks (same node coords) must be distinguishable on the wire: give each its own
     * TNI by local rank, and address each peer via ITS local-rank TNI. A64FX exposes 6 TNIs, so
     * up to 6 ranks/node. (rpn==1 keeps the original single-TNI path.) */
    int local_rank = MyRank % rpn;
    if((size_t)rpn > num_tnis){ fprintf(stderr,"GLM5_RANKS_PER_NODE %d > TNIs %zu\n",rpn,num_tnis); die("ranks/node>TNIs",-1); }
    utofu_tni_id_t tni=tni_ids[local_rank];
    rc=utofu_create_vcq_with_cmp_id(tni,DEMO_CMP_ID,0,&Vcq); if(rc!=UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id",rc);
    utofu_vcq_id_t my_real; rc=utofu_query_vcq_id(Vcq,&my_real); if(rc!=UTOFU_SUCCESS) die("utofu_query_vcq_id",rc);
    rc=utofu_reg_mem_with_stag(Vcq,Region,region_sz,RUN_STAG,0,&Base); if(rc!=UTOFU_SUCCESS) die("utofu_reg_mem_with_stag",rc);
    for(int r=0;r<N;r++){
        if(r==MyRank){ PeerVcq[r]=my_real; PeerBase[r]=Base; continue; }
        rc=utofu_construct_vcq_id(topo[r],tni_ids[r%rpn],DEMO_CQ_ID,DEMO_CMP_ID,&PeerVcq[r]); if(rc!=UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)",rc);
        utofu_set_vcq_id_path(&PeerVcq[r],NULL);
        rc=utofu_query_stadd(PeerVcq[r],RUN_STAG,&PeerBase[r]); if(rc!=UTOFU_SUCCESS) die("utofu_query_stadd(peer)",rc);
    }
    glm5_afree(tni_ids);
    barrier_robust(1);

    static tp_comm comm;
    if(MyRank==0) logmsg("allreduce: max_count=%d floats ar_tokens=%d pchunk=%d mstream=%d ar_auto_cap=%d ar_hard_cap=%d\n",
                         cfg.hidden*ar_tokens,ar_tokens,pchunk0,mstream,ar_auto_cap,ar_hard_cap);
    if(tp_comm_init(&comm,Vcq,PeerVcq,MyRank,N,cfg.hidden*ar_tokens,barrier)!=0) die("tp_comm_init",-1);
    m->ar_cb=ep_ar_callback; m->ar_ctx=&comm;
    m->ar_argmax_cb=ep_argmax_callback; m->ar_argmax_ctx=&comm;
    /* CP callbacks are wired unconditionally so a mid-run Tier A->B transition can turn CP on.
     * They stay dormant while m->cp_on==0 (the forward gates the combine/block-reduce on it). */
    g_kvbuf=(float*)glm5_amalloc((size_t)(cfg.n_heads + cfg.n_heads*cfg.head_dim)*sizeof(float));
    m->blk_reduce_cb=ep_blk_reduce; m->blk_reduce_ctx=&comm;
    m->kv_combine_cb=ep_kv_combine; m->kv_combine_ctx=&comm;
    if(MyRank==0){
        if(m->T_cp>0) logmsg("CP TIERED: Tier A (cp_on=0 bf16, %d slots) -> transition at pos=%d -> Tier B (CP int4, block=%d over %d ranks)\n",
                             m->cp_nslot,m->T_cp,m->cp_block,N);
        else if(m->cp_on) logmsg("CP ON: KV sharded block-cyclic (block=%d) over %d ranks, %d slots/rank, int4_kv=%d\n",
                             m->cp_block,N,m->cp_nslot,m->int4_kv);
        else logmsg("CP OFF: un-sharded KV, %d slots/rank, msa_on=%d (single-tier)\n",m->cp_nslot,m->msa_on);
    }
    if(envi("GLM5_COMM_OVERLAP",0)){
        atomic_store(&g_comm_stop,0); atomic_store(&g_comm_done,1); atomic_store(&g_comm_go,0);
        if(pthread_create(&g_comm_th,NULL,comm_driver,NULL)==0){
            m->ar_async_start=ar_async_start_cb; m->ar_wait=ar_wait_cb; m->ar_async_ctx=&comm;
            if(MyRank==0) logmsg("comm-overlap ON (dedicated comm-driver thread)\n");
        } else if(MyRank==0) logmsg("comm-overlap: pthread_create failed; running synchronous\n");
    }
    barrier_robust(1);
    if(MyRank==0) logmsg("all %d ranks past bootstrap barrier; starting prefill\n",N);

    int C=cfg.hidden; float*x=(float*)glm5_amalloc((size_t)C*4);

    /* ---- GLM5_MSTREAM=N: batched multi-stream decode (synthetic) -> aggregate tok/s ----
     * N concurrent streams per forward: dense GEMMs M=N + ONE EP all-reduce per layer for
     * all N tokens -> dispatch + comm amortized N-fold. Measures the structural throughput
     * lever the dummy ceiling pointed to. */
    if(mstream>1){
        die("GLM5_MSTREAM batched decode still uses legacy QKV tensors; disabled until ported to GLM5.2 MLA", -1);
        int NS=mstream;
        if(glm5_alloc_mstream(m,NS)) die("alloc_mstream",-1);
        float*X=(float*)glm5_amalloc((size_t)NS*C*4);
        int*pos=(int*)glm5_amalloc((size_t)NS*sizeof(int)),*out=(int*)glm5_amalloc((size_t)NS*sizeof(int));
        if(prefill+maxgen+8>cfg.max_pos){ if(MyRank==0) logmsg("mstream: maxpos too small\n"); }
        sm_state=0xD3F00D; for(int t=0;t<NS;t++) pos[t]=0;
        for(int g=0;g<4;g++){ for(int i=0;i<NS*C;i++) X[i]=(float)(sm_next()*0.2-0.1); glm5_forward_batch_decode(m,X,NS,pos,out); for(int t=0;t<NS;t++) pos[t]++; }
        barrier();
        double t0=now_sec(); g_ar_secs=0; g_ar_calls=0; g_ar_frags=0; int nan=0;
        for(int g=0;g<maxgen;g++){ for(int i=0;i<NS*C;i++) X[i]=(float)(sm_next()*0.2-0.1); glm5_forward_batch_decode(m,X,NS,pos,out); for(int t=0;t<NS;t++) pos[t]++;
            for(int i=0;i<NS*C;i++) if(!(X[i]==X[i])) nan++; }
        double dt=now_sec()-t0, ar=g_ar_secs;
        barrier();
        if(MyRank==0){
            logmsg("\n=== MSTREAM N=%d on %d nodes ===\n",NS,N);
            logmsg("steps=%d  %.1f ms/step  AGG %.2f tok/s  per-stream %.2f  comm %.1f%%  out0=%d NaNs=%d\n",
                   maxgen, dt/maxgen*1e3, (double)maxgen*NS/dt, (double)maxgen/dt, 100.0*ar/dt, out[0], nan);
            logmsg("SENTINEL glm5_mstream_%dn_N%d=done\n",N,NS);
        }
        glm5_afree(X);glm5_afree(pos);glm5_afree(out); glm5_afree(x); glm5_free(m); return 0;
    }

    /* ---- continuous batch gen-mode: each active request owns KV/scratch but shares weights.
     * GLM5_CBATCH_PROMPTS is a text file: one whitespace-separated token-id prompt per line.
     * GLM5_CBATCH_SLOTS limits concurrent in-flight requests; completed slots immediately
     * accept the next queued prompt. This is scheduler/interleaved decode first, not fused MLA. */
    const char*cbatch_file=getenv("GLM5_CBATCH_PROMPTS");
    if(cbatch_file&&*cbatch_file){
        int max_new=envi("GLM5_MAX_NEW",64);
        const char*out_prefix=getenv("GLM5_CBATCH_OUT_PREFIX");
        if(!out_prefix||!*out_prefix) out_prefix="glm5_cbatch_gen";
        int rc2=run_cbatch(m,cbatch_file,out_prefix,max_new,C);
        glm5_afree(x); glm5_free(m); return rc2;
    }

    /* ---- gen-mode: GLM5_PROMPT_IDS set -> real prompt prefill + greedy decode ----
     * Every rank reads the SAME prompt file and (under TP_HEAD) computes the SAME
     * global argmax -> identical token feedback -> lockstep, no extra broadcast. */
    const char*prompt_file=getenv("GLM5_PROMPT_IDS");
    const char*gen_out=getenv("GLM5_GEN_OUT");
    if(prompt_file&&*prompt_file){
        int max_new=envi("GLM5_MAX_NEW",64);
        int min_new=envi("GLM5_MIN_NEW",0);
        FILE*pf=fopen(prompt_file,"r"); if(!pf) die("cannot open GLM5_PROMPT_IDS",-1);
        int cap=1024,n_prompt=0,*prompt=glm5_amalloc((size_t)cap*sizeof(int)),v;
        while(fscanf(pf,"%d",&v)==1){ if(n_prompt>=cap){cap*=2;prompt=realloc(prompt,(size_t)cap*sizeof(int));} prompt[n_prompt++]=v; }
        fclose(pf); if(n_prompt<1) die("empty prompt",-1);
        if(n_prompt+max_new>cfg.max_pos) max_new=cfg.max_pos-n_prompt;
        if(min_new<0) min_new=0; if(min_new>max_new) min_new=max_new;
        if(MyRank==0) logmsg("gen: prompt=%d tok, max_new=%d, max_pos=%d\n",n_prompt,max_new,cfg.max_pos);
        int pf_last=-1; double t0=now_sec();
        double prof_gen0[GLM5_NPHASE], prof_gen_pf[GLM5_NPHASE], prof_gen_dec[GLM5_NPHASE];
        prof_snapshot(m,prof_gen0);
        g_ar_secs=0; g_ar_calls=0; g_ar_frags=0;
        int pchunk=envi("GLM5_PCHUNK",0);   /* Lever 1: chunked batched prefill (M=S) */
        if(pchunk>0){
            if(glm5_alloc_mstream_ex(m,pchunk,0)) die("alloc prefill chunk",-1);
            float*Xc=(float*)glm5_amalloc((size_t)pchunk*C*4);
            for(int p0=0;p0<n_prompt;p0+=pchunk){ int S=n_prompt-p0; if(S>pchunk)S=pchunk;
                for(int t=0;t<S;t++) embed_lookup(m,prompt[p0+t],Xc+(size_t)t*C);
                int a=glm5_forward_prefill_chunk(m,Xc,S,p0,p0+S>=n_prompt);
                if(a>=0) pf_last=a; }
            glm5_afree(Xc); glm5_free_mstream(m);
            if(MyRank==0) logmsg("prefill: chunked M=%d\n",pchunk);
        } else
        for(int p=0;p<n_prompt;p++){ embed_lookup(m,prompt[p],x); pf_last=glm5_forward_token(m,x,p); }
        prof_snapshot(m,prof_gen_pf);
        double tpf=now_sec()-t0;
        double gen_pf_ar=g_ar_secs; long gen_pf_calls=g_ar_calls, gen_pf_frags=g_ar_frags;
        if(prefill_only){
            barrier();
            if(MyRank==0){
                logmsg("gen_prefill_only: %d tok %.2f tok/s comm %.1f%% calls=%ld frags=%ld argmax=%d\n",
                       n_prompt,tpf>0?n_prompt/tpf:0.0,tpf>0?100.0*gen_pf_ar/tpf:0.0,gen_pf_calls,gen_pf_frags,pf_last);
                prof_log_delta("gen_prefill",prof_gen0,prof_gen_pf,n_prompt,tpf,gen_pf_ar);
                logmsg("SENTINEL glm5_prefill_%dn=done\n",N);
            }
            glm5_afree(prompt); glm5_afree(x); glm5_free(m); return 0;
        }
        int *gen=glm5_amalloc((size_t)(max_new>0?max_new:1)*sizeof(int)),ng=0,cur=pf_last,nan=0;
        g_ar_secs=0; g_ar_calls=0; g_ar_frags=0;
        double td0=now_sec();
        for(int g=0;g<max_new;g++){ gen[ng++]=cur; if((cur==GLM5_EOS_ID0||cur==GLM5_EOS_ID1||cur==GLM5_EOS_ID2) && ng>=min_new) break;
            embed_lookup(m,cur,x); cur=glm5_forward_token(m,x,n_prompt+g);
            for(int i=0;i<C;i++) if(!(x[i]==x[i])) nan++; }
        double td=now_sec()-td0;
        double gen_d_ar=g_ar_secs; long gen_d_calls=g_ar_calls, gen_d_frags=g_ar_frags;
        prof_snapshot(m,prof_gen_dec);
        barrier();
        if(MyRank==0){
            logmsg("gen: prefill %.2f tok/s comm %.1f%% calls=%ld frags=%ld, decode %d tok %.2f tok/s comm %.1f%% calls=%ld frags=%ld, NaNs=%d\n",
                   n_prompt/tpf,100.0*gen_pf_ar/tpf,gen_pf_calls,gen_pf_frags,ng,td>0?ng/td:0.0,td>0?100.0*gen_d_ar/td:0.0,gen_d_calls,gen_d_frags,nan);
            prof_log_delta("gen_prefill",prof_gen0,prof_gen_pf,n_prompt,tpf,gen_pf_ar);
            prof_log_delta("gen_decode",prof_gen_pf,prof_gen_dec,ng,td,gen_d_ar);
            char buf[6000]; int o=0; for(int i=0;i<ng&&o<5900;i++) o+=snprintf(buf+o,sizeof(buf)-o,"%d ",gen[i]);
            logmsg("GEN_IDS %s\n",buf);
            if(gen_out&&*gen_out){ FILE*gf=fopen(gen_out,"w"); if(gf){ for(int i=0;i<ng;i++) fprintf(gf,"%d%s",gen[i],i+1<ng?" ":"\n"); fclose(gf); logmsg("gen: wrote %d ids to %s\n",ng,gen_out);} }
            logmsg("SENTINEL glm5_gen_%dn=done\n",N);
        }
        glm5_afree(gen); glm5_afree(prompt); glm5_afree(x); glm5_free(m); return 0;
    }

    /* ---- synthetic-token prefill benchmark: uses embeddings but avoids a huge prompt file. ---- */
    if(prefill_synth>0){
        int pchunk=envi("GLM5_PCHUNK",0);
        double prof0s[GLM5_NPHASE], prof_pfs[GLM5_NPHASE];
        prof_snapshot(m,prof0s);
        double t0=now_sec(); g_ar_secs=0; g_ar_calls=0; g_ar_frags=0; int pf_last=-1, nan=0;
        if(pchunk>0){
            if(glm5_alloc_mstream_ex(m,pchunk,0)) die("alloc prefill chunk",-1);
            float*Xc=(float*)glm5_amalloc((size_t)pchunk*C*4);
            for(int p0=0;p0<prefill;p0+=pchunk){
                int S=prefill-p0; if(S>pchunk)S=pchunk;
                for(int t=0;t<S;t++){
                    int tok=(p0+t)*1315423911u % (unsigned)m->cfg.vocab;
                    embed_lookup(m,tok,Xc+(size_t)t*C);
                }
                /* Tier A->B: re-shard the [0,p0) history BEFORE any chunk that would store a
                 * position >= T_cp (so the Tier-A buffer never overflows). Lockstep on all ranks. */
                if(!m->cp_on && m->T_cp>0 && p0+S>m->T_cp){
                    double tt=now_sec(); glm5_prefill_to_cp(m,p0);
                    barrier();
                    if(MyRank==0) logmsg("prefill_tier: A->B re-shard at pos=%d (%.3f s) -> CP int4 %d slots/rank\n",
                                         p0,now_sec()-tt,m->cp_nslot);
                }
                int a=glm5_forward_prefill_chunk(m,Xc,S,p0,p0+S>=prefill);
                if(a>=0) pf_last=a;
                for(size_t i=0;i<(size_t)S*C;i++) if(!(Xc[i]==Xc[i])) nan++;
                if(MyRank==0 && envi("GLM5_PREFILL_ROLLING",1)){
                    double dt=now_sec()-t0;
                    logmsg("prefill_progress: %d/%d tok elapsed=%.3f rate=%.2f tok/s RSS=%.2f GB\n",
                           p0+S,prefill,dt,dt>0?(p0+S)/dt:0.0,rss_bytes()/1e9);
                }
            }
            glm5_afree(Xc); glm5_free_mstream(m);
        } else {
            for(int p=0;p<prefill;p++){
                int tok=p*1315423911u % (unsigned)m->cfg.vocab;
                embed_lookup(m,tok,x); pf_last=glm5_forward_token(m,x,p);
                for(int i=0;i<C;i++) if(!(x[i]==x[i])) nan++;
            }
        }
        double tpf=now_sec()-t0, ar=g_ar_secs; long calls=g_ar_calls, frags=g_ar_frags;
        prof_snapshot(m,prof_pfs); barrier();
        if(MyRank==0){
            logmsg("prefill_synth: %d tok %.2f tok/s comm %.1f%% calls=%ld frags=%ld pchunk=%d argmax=%d NaNs=%d RSS=%.2f GB\n",
                   prefill,tpf>0?prefill/tpf:0.0,tpf>0?100.0*ar/tpf:0.0,calls,frags,pchunk,pf_last,nan,rss_bytes()/1e9);
            prof_log_delta("prefill_synth",prof0s,prof_pfs,prefill,tpf,ar);
            logmsg("SENTINEL glm5_prefill_%dn=%s\n",N,nan==0?"done":"NAN");
        }
        glm5_afree(x); glm5_free(m); return nan==0?0:1;
    }

    /* ---- prefill (synthetic, identical activations on every rank) ---- */
    double prof0[GLM5_NPHASE], prof_pf[GLM5_NPHASE], prof_dec[GLM5_NPHASE];
    prof_snapshot(m,prof0);
    double t_pf0=now_sec(); g_ar_secs=0; g_ar_calls=0; g_ar_frags=0; int nan_count=0; double xnorm=0; int pf_last=-1;
    sm_state=0xD3F00D;
    for(int p=0;p<prefill;p++){
        for(int i=0;i<C;i++) x[i]=(float)(sm_next()*2.0-1.0);
        pf_last=glm5_forward_token(m,x,start_pos+p);
        for(int i=0;i<C;i++){ if(!(x[i]==x[i])) nan_count++; xnorm+=(double)x[i]*x[i]; }
    }
    double t_pf=now_sec()-t_pf0; double pf_ar=g_ar_secs; long pf_calls=g_ar_calls, pf_frags=g_ar_frags;
    prof_snapshot(m,prof_pf);
    barrier();

    /* ---- decode ---- */
    double t_d0=now_sec(); g_ar_secs=0; g_ar_calls=0; g_ar_frags=0; int last=pf_last;
    for(int g=0;g<maxgen;g++){
        int pos=start_pos+prefill+g; for(int i=0;i<C;i++) x[i]=(float)(sm_next()*2.0-1.0);
        last=glm5_forward_token(m,x,pos);
        for(int i=0;i<C;i++) if(!(x[i]==x[i])) nan_count++;
    }
    double t_d=now_sec()-t_d0; double d_ar=g_ar_secs;
    prof_snapshot(m,prof_dec);
    barrier();

    { char rn[64]; snprintf(rn,sizeof rn,"glm5_ep_perf_rank%02d.txt",MyRank); FILE*rf=fopen(rn,"w");
      if(rf){ fprintf(rf,"rank %d/%d owned=%d RSS=%.2f GB\n",MyRank,N,no,rss_bytes()/1e9);
        if(prefill>0) fprintf(rf,"prefill: %d tok %.1f ms/tok %.2f tok/s comm %.1f%% argmax=%d\n",prefill,t_pf/prefill*1e3,prefill/t_pf,100.0*pf_ar/t_pf,pf_last);
        if(maxgen>0)  fprintf(rf,"decode:  %d tok %.1f ms/tok %.2f tok/s comm %.1f%%\n",maxgen,t_d/maxgen*1e3,maxgen/t_d,100.0*d_ar/t_d);
        if(prefill>0) prof_file_delta(rf,"prefill",prof0,prof_pf,prefill,t_pf,pf_ar);
        if(maxgen>0)  prof_file_delta(rf,"decode",prof_pf,prof_dec,maxgen,t_d,d_ar);
        fprintf(rf,"last argmax=%d (identical across ranks == lockstep ok)  NaNs=%d ||x||=%.3e\n",last,nan_count,sqrt(xnorm)); fclose(rf);} }
    if(MyRank==0){
        logmsg("\n=== rank0 summary (%d nodes, EP all-reduce combine) ===\n",N);
        if(prefill>0) logmsg("prefill: %d tok %.1f ms/tok %.2f tok/s comm %.1f%% (ar_calls=%ld frags=%ld argmax=%d)\n",prefill,t_pf/prefill*1e3,prefill/t_pf,100.0*pf_ar/t_pf,pf_calls,pf_frags,pf_last);
        if(maxgen>0)  logmsg("decode:  %d tok %.1f ms/tok %.2f tok/s comm %.1f%%\n",maxgen,t_d/maxgen*1e3,maxgen/t_d,100.0*d_ar/t_d);
        if(prefill>0) prof_log_delta("prefill",prof0,prof_pf,prefill,t_pf,pf_ar);
        if(maxgen>0)  prof_log_delta("decode",prof_pf,prof_dec,maxgen,t_d,d_ar);
        logmsg("last argmax=%d  NaNs=%d  RSS=%.2f GB\n",last,nan_count,rss_bytes()/1e9);
        logmsg("SENTINEL glm5_ep_%dn=%s\n",N,nan_count==0?"done":"NAN");
    }
    glm5_afree(x); glm5_free(m);
    return 0;
}
