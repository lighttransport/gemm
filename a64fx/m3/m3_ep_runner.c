/*
 * m3_ep_runner.c - MiniMax-M3 (text) synthetic expert-parallel runner, A64FX.
 *
 * Multi-node EP decode/prefill over pure uTofu (mpiexec places one rank/node).
 * Each rank owns experts e with e % N == rank; router/shared/attention/dense/head
 * are replicated. The per-MoE-layer combine is one tp_allreduce_sum over the routed
 * partial [hidden]. Weights SYNTHETIC; validation = per-node memory fit, cross-rank
 * lockstep (identical step count + identical synthetic argmax), and tok/s split into
 * compute vs all-reduce. The uTofu/topology/barrier scaffold is the ds4f_ep_runner
 * one (proven to 96+ nodes); only the model calls (ds4f_* -> m3_*) differ.
 *
 * Build:  make -C a64fx/llm m3_ep_runner   (rule mirrors ds4f_ep_runner; -ltofucom)
 * Run:    mpiexec -n N [-vcoordfile vc] build/m3_ep_runner   (after tofu_topo_helper)
 *
 * Env: LLM_THREADS, M3_PREFILL(8), M3_DECODE(16), M3_MAXPOS(512), M3_LAYERS(0=full),
 *      M3_EXPERTS(0=full 128).
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

#define M3_IMPL
#include "m3.h"
#include "m3_impl.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define MAX_NODES 128
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
static void barrier(void){ barrier_robust(0); }

/* ---- EP combine all-reduce callback ---- */
static double g_ar_secs=0.0; static long g_ar_calls=0;
static void ep_ar_callback(float*buf,int count,void*ctx){
    tp_comm*c=(tp_comm*)ctx; int mc=c->max_count>0?c->max_count:count; double t0=now_sec();
    for(int off=0;off<count;){ int n=count-off; if(n>mc)n=mc; tp_allreduce_sum(c,buf+off,n); off+=n; }
    g_ar_secs+=now_sec()-t0; g_ar_calls++;
}
/* (val,global-idx) argmax all-reduce (TP_HEAD vocab-shard logits merge) */
static void ep_argmax_callback(float*val,int32_t*idx,void*ctx){
    double t0=now_sec(); tp_allreduce_argmax((tp_comm*)ctx,val,idx); g_ar_secs+=now_sec()-t0; g_ar_calls++;
}

/* token id -> input embedding (the forward's first op is input_layernorm, so the raw
 * widened embed row is the activation it expects). Under TP_EMBED the owner fills its
 * vocab-shard row and the ar_cb SUMs (zeros elsewhere) -> full embedding, bit-exact. */
static void embed_lookup(m3_model*m,int tok,float*x){
    int H=m->cfg.hidden;
    if(tok<0||tok>=m->cfg.vocab) tok=0;
    if(m->emb_rows<m->cfg.vocab){
        for(int i=0;i<H;i++) x[i]=0.f;
        if(tok>=m->emb_r0 && tok<m->emb_r0+m->emb_rows){
            const uint16_t*row=m->embed+(size_t)(tok-m->emb_r0)*H;
            for(int i=0;i<H;i++) x[i]=m3_bf2f(row[i]);
        }
        if(m->ar_cb) m->ar_cb(x,H,m->ar_ctx);
        return;
    }
    const uint16_t*row=m->embed+(size_t)tok*H;
    for(int i=0;i<H;i++) x[i]=m3_bf2f(row[i]);
}
#define M3_EOS_ID 200020

int main(void){
    int rc;
    int n_threads=envi("LLM_THREADS",48), n_cmgs=envi("M3_CMGS",4);
    int prefill=envi("M3_PREFILL",8), maxgen=envi("M3_DECODE",16), maxpos=envi("M3_MAXPOS",512);
    int layers=envi("M3_LAYERS",0), nexp=envi("M3_EXPERTS",0);

    utofu_tni_id_t*tni_ids=NULL; size_t num_tnis=0;
    rc=utofu_get_onesided_tnis(&tni_ids,&num_tnis); if(rc!=UTOFU_SUCCESS) die("utofu_get_onesided_tnis",rc);
    if(num_tnis<1) die("no onesided TNIs",-1);
    uint8_t my_coords[TOFU_NCOORDS]={0};
    rc=utofu_query_my_coords(my_coords); if(rc!=UTOFU_SUCCESS) die("utofu_query_my_coords",rc);
    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N=read_topo(topo); MyRank=-1;
    for(int r=0;r<N;r++) if(memcmp(topo[r],my_coords,TOFU_NCOORDS)==0) MyRank=r;
    if(MyRank==-1){ fprintf(stderr,"my coords not in %s\n",topo_path()); exit(1); }

    { char en[64]; snprintf(en,sizeof en,"m3_ep_stderr_rank%02d.txt",MyRank); if(!freopen(en,"w",stderr)){} setvbuf(stderr,NULL,_IOLBF,0); }
    if(MyRank==0) g_log=fopen("m3_ep_rank00.txt","w");

    int ep_rank=MyRank, ep_size=N;
    m3_config cfg=m3_default_config(); cfg.max_pos=maxpos;
    if(layers>0) cfg.n_layers=layers;
    if(nexp>0)   cfg.n_experts=nexp;
    if(cfg.n_active>cfg.n_experts) cfg.n_active=cfg.n_experts;
    if(prefill+maxgen>cfg.max_pos) die("prefill+maxgen exceeds max_pos",-1);

    int no=m3_n_owned(cfg.n_experts,ep_rank,ep_size);
    size_t arena_est=m3_arena_size(&cfg,ep_rank,ep_size);
    if(MyRank==0)
        logmsg("=== M3 EP synthetic runner: %d ranks ===\n"
               "layers=%d hidden=%d experts=%d active=%d owned~%d  heads=%d/%d head_dim=%d\n"
               "threads=%d prefill=%d decode=%d max_pos=%d  arena~%.2f GB/node (synth malloc)\n",
               N,cfg.n_layers,cfg.hidden,cfg.n_experts,cfg.n_active,no,
               cfg.n_heads,cfg.n_kv_heads,cfg.head_dim,n_threads,prefill,maxgen,maxpos,
               arena_est/(1024.0*1024.0*1024.0));
    if(MyRank==0){ FILE*mf=fopen("/proc/meminfo","r"); if(mf){ char line[128];
        while(fgets(line,sizeof line,mf)) if(!strncmp(line,"MemTotal",8)||!strncmp(line,"MemFree",7)||!strncmp(line,"MemAvailable",12)){
            for(char*p=line;*p;p++) if(*p=='\n')*p=0; logmsg("NODE_MEMINFO %s\n",line);} fclose(mf);} }

    int real_weights=envi("M3_REAL",0);
    const char*blob_dir=getenv("M3_STAGE_DIR");
    double ta0=now_sec();
    m3_model*m=real_weights ? m3_load_real(cfg,ep_rank,ep_size,blob_dir,n_threads,n_cmgs)
                            : m3_alloc_synth(cfg,ep_rank,ep_size,n_threads,n_cmgs);
    if(!m){ fprintf(stderr,"rank %d: model %s failed\n",MyRank,real_weights?"load":"alloc"); exit(1); }
    double ta1=now_sec();
    { char tn[64]; snprintf(tn,sizeof tn,"m3_ep_load_rank%02d.txt",MyRank); FILE*tf=fopen(tn,"w");
      if(tf){ fprintf(tf,"rank %d: alloc=%.2fs arena_used=%.2f GB RSS=%.2f GB owned=%d/layer\n",MyRank,ta1-ta0,m->arena_used/1e9,rss_bytes()/1e9,no); fclose(tf);} }

    /* ---- barrier region + VCQ ---- */
    SlotSend=DEMO_CACHE_LINE; SlotB=DEMO_CACHE_LINE; SEND_OFF=0; BAR_BASE=SlotSend;
    size_t region_sz=BAR_BASE+(size_t)(N+1)*SlotB;
    if(posix_memalign((void**)&Region,DEMO_CACHE_LINE,region_sz)!=0) die("posix_memalign",-1);
    memset(Region,0,region_sz);
    utofu_tni_id_t tni=tni_ids[0];
    rc=utofu_create_vcq_with_cmp_id(tni,DEMO_CMP_ID,0,&Vcq); if(rc!=UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id",rc);
    utofu_vcq_id_t my_real; rc=utofu_query_vcq_id(Vcq,&my_real); if(rc!=UTOFU_SUCCESS) die("utofu_query_vcq_id",rc);
    rc=utofu_reg_mem_with_stag(Vcq,Region,region_sz,RUN_STAG,0,&Base); if(rc!=UTOFU_SUCCESS) die("utofu_reg_mem_with_stag",rc);
    for(int r=0;r<N;r++){
        if(r==MyRank){ PeerVcq[r]=my_real; PeerBase[r]=Base; continue; }
        rc=utofu_construct_vcq_id(topo[r],tni,DEMO_CQ_ID,DEMO_CMP_ID,&PeerVcq[r]); if(rc!=UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)",rc);
        utofu_set_vcq_id_path(&PeerVcq[r],NULL);
        rc=utofu_query_stadd(PeerVcq[r],RUN_STAG,&PeerBase[r]); if(rc!=UTOFU_SUCCESS) die("utofu_query_stadd(peer)",rc);
    }
    free(tni_ids);
    barrier_robust(1);

    static tp_comm comm;
    if(tp_comm_init(&comm,Vcq,PeerVcq,MyRank,N,cfg.hidden,barrier)!=0) die("tp_comm_init",-1);
    m->ar_cb=ep_ar_callback; m->ar_ctx=&comm;
    m->ar_argmax_cb=ep_argmax_callback; m->ar_argmax_ctx=&comm;
    barrier_robust(1);
    if(MyRank==0) logmsg("all %d ranks past bootstrap barrier; starting prefill\n",N);

    int C=cfg.hidden; float*x=(float*)aligned_alloc(256,(size_t)C*4);

    /* ---- gen-mode: M3_PROMPT_IDS set -> real prompt prefill + greedy decode ----
     * Every rank reads the SAME prompt file and (under TP_HEAD) computes the SAME
     * global argmax -> identical token feedback -> lockstep, no extra broadcast. */
    const char*prompt_file=getenv("M3_PROMPT_IDS");
    const char*gen_out=getenv("M3_GEN_OUT");
    if(prompt_file&&*prompt_file){
        int max_new=envi("M3_MAX_NEW",64);
        FILE*pf=fopen(prompt_file,"r"); if(!pf) die("cannot open M3_PROMPT_IDS",-1);
        int cap=1024,n_prompt=0,*prompt=malloc((size_t)cap*sizeof(int)),v;
        while(fscanf(pf,"%d",&v)==1){ if(n_prompt>=cap){cap*=2;prompt=realloc(prompt,(size_t)cap*sizeof(int));} prompt[n_prompt++]=v; }
        fclose(pf); if(n_prompt<1) die("empty prompt",-1);
        if(n_prompt+max_new>cfg.max_pos) max_new=cfg.max_pos-n_prompt;
        if(MyRank==0) logmsg("gen: prompt=%d tok, max_new=%d, max_pos=%d\n",n_prompt,max_new,cfg.max_pos);
        int pf_last=-1; double t0=now_sec();
        for(int p=0;p<n_prompt;p++){ embed_lookup(m,prompt[p],x); pf_last=m3_forward_token(m,x,p); }
        int *gen=malloc((size_t)(max_new>0?max_new:1)*sizeof(int)),ng=0,cur=pf_last,nan=0;
        double td0=now_sec();
        for(int g=0;g<max_new;g++){ gen[ng++]=cur; if(cur==M3_EOS_ID) break;
            embed_lookup(m,cur,x); cur=m3_forward_token(m,x,n_prompt+g);
            for(int i=0;i<C;i++) if(!(x[i]==x[i])) nan++; }
        double td=now_sec()-td0, tpf=td0-t0;
        barrier();
        if(MyRank==0){
            logmsg("gen: prefill %.2f tok/s, decode %d tok %.2f tok/s, NaNs=%d\n",
                   n_prompt/tpf, ng, td>0?ng/td:0.0, nan);
            char buf[6000]; int o=0; for(int i=0;i<ng&&o<5900;i++) o+=snprintf(buf+o,sizeof(buf)-o,"%d ",gen[i]);
            logmsg("GEN_IDS %s\n",buf);
            if(gen_out&&*gen_out){ FILE*gf=fopen(gen_out,"w"); if(gf){ for(int i=0;i<ng;i++) fprintf(gf,"%d%s",gen[i],i+1<ng?" ":"\n"); fclose(gf); logmsg("gen: wrote %d ids to %s\n",ng,gen_out);} }
            logmsg("SENTINEL m3_gen_%dn=done\n",N);
        }
        free(gen); free(prompt); free(x); m3_free(m); return 0;
    }

    /* ---- prefill (synthetic, identical activations on every rank) ---- */
    double t_pf0=now_sec(); g_ar_secs=0; g_ar_calls=0; int nan_count=0; double xnorm=0; int pf_last=-1;
    sm_state=0xD3F00D;
    for(int p=0;p<prefill;p++){
        for(int i=0;i<C;i++) x[i]=(float)(sm_next()*2.0-1.0);
        pf_last=m3_forward_token(m,x,p);
        for(int i=0;i<C;i++){ if(!(x[i]==x[i])) nan_count++; xnorm+=(double)x[i]*x[i]; }
    }
    double t_pf=now_sec()-t_pf0; double pf_ar=g_ar_secs; long pf_calls=g_ar_calls;
    barrier();

    /* ---- decode ---- */
    double t_d0=now_sec(); g_ar_secs=0; g_ar_calls=0; int last=pf_last;
    for(int g=0;g<maxgen;g++){
        int pos=prefill+g; for(int i=0;i<C;i++) x[i]=(float)(sm_next()*2.0-1.0);
        last=m3_forward_token(m,x,pos);
        for(int i=0;i<C;i++) if(!(x[i]==x[i])) nan_count++;
    }
    double t_d=now_sec()-t_d0; double d_ar=g_ar_secs;
    barrier();

    { char rn[64]; snprintf(rn,sizeof rn,"m3_ep_perf_rank%02d.txt",MyRank); FILE*rf=fopen(rn,"w");
      if(rf){ fprintf(rf,"rank %d/%d owned=%d RSS=%.2f GB\n",MyRank,N,no,rss_bytes()/1e9);
        if(prefill>0) fprintf(rf,"prefill: %d tok %.1f ms/tok %.2f tok/s comm %.1f%% argmax=%d\n",prefill,t_pf/prefill*1e3,prefill/t_pf,100.0*pf_ar/t_pf,pf_last);
        if(maxgen>0)  fprintf(rf,"decode:  %d tok %.1f ms/tok %.2f tok/s comm %.1f%%\n",maxgen,t_d/maxgen*1e3,maxgen/t_d,100.0*d_ar/t_d);
        fprintf(rf,"last argmax=%d (identical across ranks == lockstep ok)  NaNs=%d ||x||=%.3e\n",last,nan_count,sqrt(xnorm)); fclose(rf);} }
    if(MyRank==0){
        logmsg("\n=== rank0 summary (%d nodes, EP all-reduce combine) ===\n",N);
        if(prefill>0) logmsg("prefill: %d tok %.1f ms/tok %.2f tok/s comm %.1f%% (ar_calls=%ld argmax=%d)\n",prefill,t_pf/prefill*1e3,prefill/t_pf,100.0*pf_ar/t_pf,pf_calls,pf_last);
        if(maxgen>0)  logmsg("decode:  %d tok %.1f ms/tok %.2f tok/s comm %.1f%%\n",maxgen,t_d/maxgen*1e3,maxgen/t_d,100.0*d_ar/t_d);
        logmsg("last argmax=%d  NaNs=%d  RSS=%.2f GB\n",last,nan_count,rss_bytes()/1e9);
        logmsg("SENTINEL m3_ep_%dn=%s\n",N,nan_count==0?"done":"NAN");
    }
    free(x); m3_free(m);
    return 0;
}
