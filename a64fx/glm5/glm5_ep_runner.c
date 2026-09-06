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
#include <sys/syscall.h>
#include <sys/stat.h>
#include <utofu.h>

#define GLM5_IMPL
#include "glm5.h"
#include "glm5_impl.h"
#include "../../common/glm5_bpe.h"
#include "../../common/glm5_chat_template.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define MAX_NODES 512   /* >= largest EP run (384-node prefill); arrays are O(MAX_NODES) tiny */
#define RUN_STAG  DEMO_STAG
#define WAIT_TIMEOUT_SEC 900.0  /* generous: the Phase-2 merge re-slice re-reads the dense blob from
                                 * /local (~300 s, per-node variable); barriers are infrequent and the
                                 * per-token allreduce has its own timeout, so this only delays detecting
                                 * a genuine (rare) barrier hang. */

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
static void *glm5_arealloc(void *old,size_t old_bytes,size_t new_bytes){
    void *p=glm5_amalloc(new_bytes);
    if(!p) return NULL;
    if(old){ memcpy(p,old,old_bytes<new_bytes?old_bytes:new_bytes); glm5_afree(old); }
    return p;
}

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

/* ===================== data-parallel groups ===================== */
/* The N global ranks are split into G independent groups of GSize contiguous ranks; each group is a
 * complete EP model (ep_size=GSize) prefilling its own sequence. Per-sequence collectives are scoped
 * to the group (group-local tp_comm + gbarrier) so groups run fully async; only the bootstrap/final
 * barrier is global. gbarrier uses a slot region disjoint from the global barrier's. */
static int GId, GBase, GSize=1, GRank;
static size_t GBAR_BASE;
static uint64_t Gt=1;
static inline size_t gbar_recv_off(int s){ return GBAR_BASE+(size_t)s*SlotB; }
static inline size_t gbar_go_off(void)   { return GBAR_BASE+(size_t)N*SlotB; }
static void gbarrier(void){
    if(GSize<=1){ return; }
    uint64_t t=++Gt; char*sb=Region+SEND_OFF; int root=GBase;
    if(MyRank==root){
        for(int s=1;s<GSize;s++) wait_ge((volatile uint64_t*)(Region+gbar_recv_off(s)),t,"gbarrier fan-in");
        for(int s=1;s<GSize;s++){ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[root+s],Base+SEND_OFF,PeerBase[root+s]+gbar_go_off(),8,1); }
    } else {
        volatile uint64_t*go=(volatile uint64_t*)(Region+gbar_go_off()); double ts=now_sec();
        do{ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[root],Base+SEND_OFF,PeerBase[root]+gbar_recv_off(GRank),8,1);
            for(int a=0;a<50&&*go<t;a++) usleep(2000);
            if(now_sec()-ts>WAIT_TIMEOUT_SEC) die("gbarrier timeout",-1);
        }while(*go<t);
    }
}
/* Pick the largest group count G in {4,2,1} (dividing N) whose per-group Tier-A KV budget holds the
 * target context un-sharded. Mirrors glm5_kv_init's MemAvailable budget, but at ep_size=GSize (the
 * per-rank weight footprint grows as groups shrink, so smaller groups have a lower context ceiling). */
static int pick_groups(const glm5_config*cfg,int n,int target_ctx){
    int KVD=glm5_kv_cache_dim(cfg), ID=cfg->index_dim;
    int n_dense=cfg->n_dense_layers<cfg->n_layers?cfg->n_dense_layers:cfg->n_layers;
    int n_moe=cfg->n_layers-n_dense;
    long per_pos=(long)cfg->n_layers*KVD*2 + (long)n_moe*ID*2; if(per_pos<1)per_pos=1;
    long avail=glm5_meminfo_bytes("MemAvailable");
    if(avail<=0 || target_ctx<=0) return 1;            /* can't decide -> safe single group */
    int cands[3]={4,2,1};
    for(int i=0;i<3;i++){
        int G=cands[i]; if(n%G) continue; int gsize=n/G;
        glm5_config tiny=*cfg; tiny.max_pos=128;        /* KV/rope negligible -> ~weights at ep_size */
        long reserve=(long)glm5_arena_size(&tiny,0,gsize) + (long)cfg->max_pos*(cfg->rotary_dim/2)*4*2;
        long budget=avail-reserve-avail/10;
        if(budget<(1L<<30)) continue;                   /* group too small for weights + 1 GB KV */
        if((long)target_ctx <= budget/per_pos) return G;
    }
    return 1;
}

/* ===================== Phase 2: dynamic group merge ===================== */
#define KV_STAG 11
/* Tier-A KV propagation at a pairwise merge: the surviving (even/lower) subgroup holds the
 * replicated KV; copy it to the sibling (odd/upper) subgroup pairwise by local index over uTofu.
 * Call AFTER GBase/GSize/GRank are updated to the merged group so gbarrier spans all 2*old_gsize
 * ranks. upto = filled positions (only the [0,upto) prefix is transferred). */
/* Transfer one cache buffer's filled prefix from the even (survivor) subgroup to the odd sibling,
 * pairwise, with dc-civac coherence around the uTofu put. All group ranks must call this the same
 * number of times (kv_cache exists on every layer; idx_k_cache only on indexer layers - structural,
 * so consistent across ranks). */
static void glm5_kv_xfer(char*buf,long reg_bytes,long copy_bytes,int stag,int even,int partner){
    utofu_stadd_t st; int rc=utofu_reg_mem_with_stag(Vcq,buf,reg_bytes,stag,0,&st);
    if(rc!=UTOFU_SUCCESS) die("kv_xfer reg",rc);
    for(long off=0;off<copy_bytes;off+=64) __asm__ __volatile__("dc civac, %0"::"r"(buf+off):"memory");
    __asm__ __volatile__("dsb sy":::"memory");
    gbarrier();                                        /* both ends registered + flushed */
    if(even && copy_bytes>0){
        utofu_stadd_t dst; rc=utofu_query_stadd(PeerVcq[partner],stag,&dst); if(rc!=UTOFU_SUCCESS) die("kv_xfer query",rc);
        const long CH=4L<<20;
        for(long off=0;off<copy_bytes;off+=CH){ long n=copy_bytes-off; if(n>CH)n=CH; put_issue(PeerVcq[partner],st+off,dst+off,(size_t)n,1); }
    }
    gbarrier();                                        /* put landed in receiver DRAM */
    if(!even){ for(long off=0;off<copy_bytes;off+=64) __asm__ __volatile__("dc civac, %0"::"r"(buf+off):"memory"); __asm__ __volatile__("dsb sy":::"memory"); }
    utofu_dereg_mem(Vcq,st,0);
}
static void glm5_group_kv_propagate(glm5_model*m,int old_gsize,int new_base,int upto){
    const glm5_config*c=&m->cfg; int KVD=glm5_kv_cache_dim(c), ID=c->index_dim;
    int li=(MyRank-new_base)%old_gsize;                /* local index within subgroup */
    int even=(MyRank-new_base)<old_gsize;              /* even subgroup survives, sends */
    int partner=even ? (new_base+old_gsize+li) : (new_base+li);
    long ns=(long)m->cp_nslot;                         /* Tier-A buffers are cp_nslot (=T_cp) slots, NOT max_pos */
    for(int l=0;l<c->n_layers;l++){
        glm5_layer*L=&m->layers[l];
        if(L->kv_cache)     glm5_kv_xfer((char*)L->kv_cache,    ns*KVD*2,(long)upto*KVD*2,KV_STAG,  even,partner);
        if(L->idx_k_cache)  glm5_kv_xfer((char*)L->idx_k_cache, ns*ID*2, (long)upto*ID*2, KV_STAG+1,even,partner);  /* MSA index keys (tiered runs) */
    }
}
/* Merge this group with its sibling (pairwise) -> group of size 2*GSize. Local expert drop + KV
 * propagation to the sibling + rebuild group-scoped collectives. Survivor = even subgroup's sequence. */
static void glm5_group_merge(glm5_model*m,tp_comm*comm,int ar_floats,int upto,const char*blob_dir,int orig_gsize){
    int old_gsize=GSize, new_gsize=GSize*2;
    int new_GId=GId/2, new_base=new_GId*new_gsize, new_GRank=MyRank-new_base;
    GId=new_GId; GBase=new_base; GSize=new_gsize; GRank=new_GRank;   /* group identity FIRST */
    glm5_group_expert_drop(m,new_gsize,new_GRank);                   /* local: routed experts, sets ep_* */
    if(glm5_group_tp_reslice(m,blob_dir,MyRank%orig_gsize,new_gsize,new_GRank)!=0) die("merge tp_reslice",-1); /* local: TP dense from blob */
    glm5_group_kv_propagate(m,old_gsize,new_base,upto);             /* network: even KV -> odd sibling */
    tp_comm_free(comm);
    if(tp_comm_init(comm,Vcq,PeerVcq+GBase,GRank,GSize,ar_floats,gbarrier)!=0) die("merge tp_comm_init",-1);
}

/* ===================== KV cache persistence (prompt/prefix caching) ===================== */
/* Save the processed [0,total) Tier-A KV (bf16, slot=pos; replicated across ranks, so only rank 0
 * writes) to <dir>/kv.bin with a 4-int header [total,n_layers,KVD,ep_size]. Lets an expensive system
 * prompt be processed once and re-loaded on later runs instead of recomputed. */
static void glm5_kv_save(glm5_model*m,const char*dir,int total){
    const glm5_config*c=&m->cfg; int KVD=glm5_kv_cache_dim(c);
    char fn[1100]; snprintf(fn,sizeof fn,"%s/kv.bin",dir);
    FILE*f=fopen(fn,"wb"); if(!f){ logmsg("kv_save: cannot open %s\n",fn); return; }
    int hdr[4]={total,c->n_layers,KVD,m->ep_size}; fwrite(hdr,sizeof(int),4,f);
    size_t per=(size_t)total*KVD;
    for(int l=0;l<c->n_layers;l++) if(m->layers[l].kv_cache) fwrite(m->layers[l].kv_cache,2,per,f);
    fclose(f);
    logmsg("kv_save: wrote %d positions x %d layers x %d (%.2f GB) -> %s\n",
           total,c->n_layers,KVD,(double)c->n_layers*per*2/1e9,fn);
}
/* Read <dir>/kv.bin into each layer's Tier-A kv_cache[0:P*KVD]; returns P (positions) or -1. All ranks
 * read the same file (Tier-A KV is replicated). Caller resumes prefill from start_pos=P. */
static int glm5_kv_load(glm5_model*m,const char*dir){
    const glm5_config*c=&m->cfg; int KVD=glm5_kv_cache_dim(c);
    char fn[1100]; snprintf(fn,sizeof fn,"%s/kv.bin",dir);
    FILE*f=fopen(fn,"rb"); if(!f){ if(MyRank==0) logmsg("kv_load: cannot open %s\n",fn); return -1; }
    int hdr[4]; if(fread(hdr,sizeof(int),4,f)!=4){ fclose(f); return -1; }
    int P=hdr[0];
    if(hdr[1]!=c->n_layers || hdr[2]!=KVD){ if(MyRank==0) logmsg("kv_load: layout mismatch (layers %d/%d KVD %d/%d)\n",hdr[1],c->n_layers,hdr[2],KVD); fclose(f); return -1; }
    if(P<=0 || P>c->max_pos){ fclose(f); return -1; }
    size_t per=(size_t)P*KVD;
    for(int l=0;l<c->n_layers;l++) if(m->layers[l].kv_cache){ if(fread(m->layers[l].kv_cache,2,per,f)!=per){ fclose(f); return -1; } }
    fclose(f);
    if(MyRank==0) logmsg("kv_load: restored %d positions x %d layers from %s\n",P,c->n_layers,fn);
    return P;
}

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
/* batched merge: M streams' (val,idx) pairs in ONE collective (batched-decode head) */
static void ep_argmax_n_callback(float*vi,int n,void*ctx){
    double t0=now_sec(); tp_allreduce_argmax_n((tp_comm*)ctx,vi,n); g_ar_secs+=now_sec()-t0; g_ar_calls++;
    g_ar_frags++;
}

/* ===================== GLM5_AR_PROBE: decode all-reduce calibration ===================== */
/* Measures the PRODUCTION tp_allreduce on the real machine to calibrate the local models:
 *   decode_sim.py : recalibrate(ar_ms=us_per_ar/1000)  (or round_ms=round_us/1000)
 *   qlair         : cross-check tools/qlair/tofu/qlair-tofu.hh latency constants and the
 *                   MOE_COMM_RESULTS wire floors against the measured per-AR anatomy.
 * Sweeps robust mode {1 prod, 2 lean, 0 passive} x payload {fp32,bf16} x M in {1,2,8,16,32}
 * (message = [M,hidden] f32, the batched-decode AR), plus argmax/argmax_n head merges and a
 * 78-AR back-to-back "decode token" sequence (the comm term decode_sim anchors on: measured
 * 0.25 tok/s @96n == 153 AR x ~26 ms). All ranks run identical sequences (lockstep); rank 0
 * reports grep-able "ARPROBE," CSV lines. Values in buf saturate to inf after a few sums --
 * harmless (inf+inf=inf, no NaNs) and timing-neutral. Runs INSTEAD of prefill/decode
 * (GLM5_AR_PROBE=1; pair with GLM5_LAYERS=1 GLM5_DUMMY=1 to make the model alloc trivial). */
static void run_ar_probe(tp_comm*c,int H){
    int reps=envi("GLM5_PROBE_REPS",50), wu=envi("GLM5_PROBE_WARMUP",5);
    static const int Ms[]={1,2,8,16,32}; const int nM=5;
    int rounds=0; for(int x=1;x<c->pof2;x<<=1) rounds++;
    float*buf=glm5_amalloc((size_t)c->max_count*4);
    if(!buf) die("ar_probe alloc",-1);
    if(MyRank==0) logmsg("ARPROBE,begin,N=%d,rounds=%d,max_count=%d,reps=%d\n",c->nprocs,rounds,c->max_count,reps);
    int save_robust=c->robust, save_bf16=c->use_bf16;
    static const int RB[3]={1,2,0};
    for(int ri=0;ri<3;ri++){ c->robust=RB[ri];
        for(int bf=0;bf<=1;bf++){ c->use_bf16=bf;
            for(int mi=0;mi<nM;mi++){ int count=Ms[mi]*H; if(count>c->max_count) continue;
                for(int i=0;i<count;i++) buf[i]=(float)((i%13)+1);
                for(int w=0;w<wu;w++) tp_allreduce_sum(c,buf,count);
                barrier();
                double t0=now_sec();
                for(int r=0;r<reps;r++) tp_allreduce_sum(c,buf,count);
                double us=(now_sec()-t0)/reps*1e6;
                if(MyRank==0) logmsg("ARPROBE,sum,N=%d,robust=%d,bf16=%d,M=%d,bytes=%d,us_per_ar=%.1f,round_us=%.1f\n",
                                     c->nprocs,c->robust,bf,Ms[mi],count*4,us,rounds>0?us/rounds:us);
            }
        }
        c->use_bf16=0;
        { float v=(float)MyRank; int32_t idx=MyRank;              /* per-stream head merge */
          for(int w=0;w<wu;w++) tp_allreduce_argmax(c,&v,&idx);
          barrier(); double t0=now_sec();
          for(int r=0;r<reps;r++) tp_allreduce_argmax(c,&v,&idx);
          double us=(now_sec()-t0)/reps*1e6;
          if(MyRank==0) logmsg("ARPROBE,argmax,N=%d,robust=%d,us_per_ar=%.1f\n",c->nprocs,c->robust,us); }
        { float vi[64];                                           /* batched head merge, 32 pairs */
          for(int k=0;k<32;k++){ vi[2*k]=(float)((MyRank*7+k)%11); int32_t ii=MyRank; memcpy(&vi[2*k+1],&ii,4); }
          for(int w=0;w<wu;w++) tp_allreduce_argmax_n(c,vi,32);
          barrier(); double t0=now_sec();
          for(int r=0;r<reps;r++) tp_allreduce_argmax_n(c,vi,32);
          double us=(now_sec()-t0)/reps*1e6;
          if(MyRank==0) logmsg("ARPROBE,argmax_n32,N=%d,robust=%d,us_per_ar=%.1f\n",c->nprocs,c->robust,us); }
        for(int mi=0;mi<nM;mi++){ int count=Ms[mi]*H; if(count>c->max_count) continue;
            int tok_reps=reps/10>3?reps/10:3;                     /* one decode token = 78 ARs */
            for(int i=0;i<count;i++) buf[i]=(float)((i%13)+1);
            barrier(); double t0=now_sec();
            for(int r=0;r<tok_reps;r++) for(int l=0;l<78;l++) tp_allreduce_sum(c,buf,count);
            double ms=(now_sec()-t0)/tok_reps*1e3;
            if(MyRank==0) logmsg("ARPROBE,token78,N=%d,robust=%d,M=%d,ms_per_token_comm=%.2f,tok_s_comm_bound=%.3f\n",
                                 c->nprocs,c->robust,Ms[mi],ms,Ms[mi]/(ms/1e3));
        }
    }
    c->robust=save_robust; c->use_bf16=save_bf16;
    glm5_afree(buf);
    if(MyRank==0) logmsg("ARPROBE,end\n");
}
/* 2-level (hierarchical) AR probe: same metrics as run_ar_probe but via tp_allreduce_sum_2d over
 * row(B)+col(A) sub-comms.  Logged as ARPROBE2D,* so a run with GLM5_AR_PROBE=1 GLM5_AR_2D=A emits
 * both ARPROBE,sum and ARPROBE2D,sum lines for a same-allocation flat-vs-hierarchical comparison.
 * Covers the decode-relevant sizes (M=1,2,8) + the 78-AR/token comm-bound projection; robust/bf16
 * follow the production decode path (robust=2, bf16=1).  See a64fx/glm5/DECODE_SCALING.md. */
static void run_ar_probe_2d(tp_comm*row,tp_comm*col,int H,int A,int B){
    int reps=envi("GLM5_PROBE_REPS",50), wu=envi("GLM5_PROBE_WARMUP",5);
    static const int Ms[]={1,2,8,16,32}; const int nM=5;
    int max_count=row->max_count; float*buf=glm5_amalloc((size_t)max_count*4);
    if(!buf) die("ar_probe_2d alloc",-1);
    row->robust=col->robust=2; row->use_bf16=col->use_bf16=1;             /* production decode AR config */
    if(MyRank==0) logmsg("ARPROBE2D,begin,N=%d,A=%d,B=%d,rounds=%d+%d,max_count=%d,reps=%d\n",
                         row->nprocs*0+A*B,A,B,col->nrounds,row->nrounds,max_count,reps);
    for(int mi=0;mi<nM;mi++){ int count=Ms[mi]*H; if(count>max_count) continue;
        for(int i=0;i<count;i++) buf[i]=(float)((i%13)+1);
        for(int w=0;w<wu;w++) tp_allreduce_sum_2d(row,col,buf,count);
        barrier();
        double t0=now_sec();
        for(int r=0;r<reps;r++) tp_allreduce_sum_2d(row,col,buf,count);
        double us=(now_sec()-t0)/reps*1e6;
        if(MyRank==0) logmsg("ARPROBE2D,sum,A=%d,B=%d,M=%d,bytes=%d,us_per_ar=%.1f\n",A,B,Ms[mi],count*4,us);
    }
    for(int mi=0;mi<nM;mi++){ int count=Ms[mi]*H; if(count>max_count) continue;
        int tok_reps=reps/10>3?reps/10:3;                                /* one decode token = 78 ARs */
        for(int i=0;i<count;i++) buf[i]=(float)((i%13)+1);
        barrier(); double t0=now_sec();
        for(int r=0;r<tok_reps;r++) for(int l=0;l<78;l++) tp_allreduce_sum_2d(row,col,buf,count);
        double ms=(now_sec()-t0)/tok_reps*1e3;
        if(MyRank==0) logmsg("ARPROBE2D,token78,A=%d,B=%d,M=%d,ms_per_token_comm=%.2f,tok_s_comm_bound=%.3f\n",
                             A,B,Ms[mi],ms,Ms[mi]/(ms/1e3));
    }
    glm5_afree(buf);
    if(MyRank==0) logmsg("ARPROBE2D,end\n");
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
    for(int h=0;h<nh;h++){
        /* same guard as ep_kv_combine_batch: clamp the (mathematically <=0) exponent so a bf16-
         * rounded global max cannot overflow expf, and skip empty-selection heads (se==0) whose
         * 0*inf would be NaN.  See the 512K long-context NaN note in CTX_BUFFER_LAYOUT.md. */
        float d=mx[h]-gmx[h]; if(d>0.0f) d=0.0f;
        float s=(se[h]>0.0f)?expf(d):0.0f;
        buf[h]=se[h]*s;
        float*o=out+h*hd,*b=buf+nh+(size_t)h*hd; for(int i=0;i<hd;i++) b[i]=o[i]*s; }
    for(int off=0;off<cnt;){ int n=cnt-off; if(n>mc)n=mc; tp_allreduce_sum(c,buf+off,n); off+=n; nf++; }
    for(int h=0;h<nh;h++){ float inv=1.0f/(buf[h]>0?buf[h]:1.0f);
        float*o=out+h*hd,*b=buf+nh+(size_t)h*hd; for(int i=0;i<hd;i++) o[i]=b[i]*inv; }
    g_ar_secs+=now_sec()-t0; g_ar_calls++;
    g_ar_frags+=nf;
}
/* batched flash-combine for a whole prefill chunk (S tokens): collapses 2*S per-token collectives
 * into ONE allreduce_max over [S*nh] + ONE allreduce_sum over [S*(nh+nh*hd)]. The per-(t,h,i)
 * arithmetic is byte-for-byte ep_kv_combine, so the result is bit-identical to the per-token loop;
 * allreduce reduces each element independently under a fixed rank schedule, so concatenating the
 * tokens (and the flat max_count fragmentation that straddles token boundaries) changes nothing.
 * mx/se share mxse_stride; out uses out_stride. Local rescale/normalize are OpenMP-parallel here
 * (the per-token path can't parallelize them -- they sit between the two collectives). */
static float *g_kvmax=NULL;   /* [S*nh] gathered per-(t,h) global max */
static void ep_kv_combine_batch(float*out,float*mx,float*se,int S,int nh,int hd,
                                int out_stride,int mxse_stride,void*ctx){
    tp_comm*c=(tp_comm*)ctx; int mc=c->max_count>0?c->max_count:1; double tc=0;
    long nf=0; const size_t blk=(size_t)nh+(size_t)nh*hd;
    /* (1) gather strided mx -> contiguous [S*nh], one fragmented allreduce_max (keep result). */
    float*gmx=g_kvmax;
    for(int t=0;t<S;t++){ const float*mt=mx+(size_t)t*mxse_stride; float*gt=gmx+(size_t)t*nh;
        for(int h=0;h<nh;h++) gt[h]=mt[h]; }
    { double t0=now_sec(); int cnt=S*nh; for(int off=0;off<cnt;){ int n=cnt-off; if(n>mc)n=mc; tp_allreduce_max(c,gmx+off,n); off+=n; nf++; } tc+=now_sec()-t0; }
    /* (2) rescale + pack each token's [se(nh) | out(nh*hd)] block (purely local). */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<S;t++){
        const float*mt=mx+(size_t)t*mxse_stride,*st=se+(size_t)t*mxse_stride,*gt=gmx+(size_t)t*nh;
        const float*ot=out+(size_t)t*out_stride; float*bt=g_kvbuf+(size_t)t*blk;
        for(int h=0;h<nh;h++){
            /* gt is the GLOBAL max, so mt-gt <= 0 mathematically; a positive can only come from the
             * all-reduce's bf16 rounding of the max (TP_AR_BF16=1).  Unclamped that overflows expf
             * to +inf and, for a rank with an EMPTY local selection (mt=-1e30, st=0), yields
             * 0*inf = NaN.  That is the 512K long-context NaN: at extreme ctx the ~18 MSA-selected
             * blocks spread thinly over the ranks, so some (token,head) has no owned block on this
             * rank and hits the sentinel.  Clamp the exponent and skip empty heads. */
            float d=mt[h]-gt[h]; if(d>0.0f) d=0.0f;
            float s=(st[h]>0.0f)?expf(d):0.0f;
            bt[h]=st[h]*s;
            const float*o=ot+(size_t)h*hd; float*b=bt+nh+(size_t)h*hd; for(int i=0;i<hd;i++) b[i]=o[i]*s; } }
    /* (3) one flat fragmented allreduce_sum over the whole [S*blk] payload. */
    { double t0=now_sec(); size_t cnt=(size_t)S*blk; for(size_t off=0;off<cnt;){ size_t n=cnt-off; if(n>(size_t)mc)n=(size_t)mc;
        tp_allreduce_sum(c,g_kvbuf+off,(int)n); off+=n; nf++; } tc+=now_sec()-t0; }
    /* (4) normalize back into out (purely local). */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int t=0;t<S;t++){
        const float*bt=g_kvbuf+(size_t)t*blk; float*ot=out+(size_t)t*out_stride;
        for(int h=0;h<nh;h++){ float inv=1.0f/(bt[h]>0?bt[h]:1.0f);
            const float*b=bt+nh+(size_t)h*hd; float*o=ot+(size_t)h*hd; for(int i=0;i<hd;i++) o[i]=b[i]*inv; } }
    g_ar_secs+=tc; g_ar_calls++;  /* time ONLY the collectives (not the local pack/normalize); one combine/chunk vs S */
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
/* Pick the comm-driver's dedicated core: GLM5_COMM_CORE if set, else the HIGHEST core in this
 * process's cpuset (A64FX job cpuset = 12-59, so core 59). The OMP pool must be sized to cores-1
 * and pinned to the LOWER cores (OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores) so the
 * comm thread never shares a core with a compute thread -> the spin-wait AR stops starving. */
static int glm5_comm_pick_core(void){
    const char*e=getenv("GLM5_COMM_CORE"); if(e&&*e) return atoi(e);
    cpu_set_t s; CPU_ZERO(&s); int hi=-1;
    if(sched_getaffinity(0,sizeof s,&s)==0) for(int c=0;c<512;c++) if(CPU_ISSET(c,&s)) hi=c;
    return hi;   /* highest allowed core; -1 -> no pin */
}
static void* comm_driver(void *a){ (void)a;
    int cc=glm5_comm_pick_core();
    if(cc>=0){ cpu_set_t s; CPU_ZERO(&s); CPU_SET(cc,&s); sched_setaffinity(0,sizeof s,&s);
        if(MyRank==0) logmsg("comm-driver pinned to core %d\n",cc); }
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
/* GLM5_BATCH_SELFCHECK=<tok>: validate the multi-stream MLA decode (glm5_forward_batch_decode_mla)
 * against single-stream glm5_forward_token at M=1, pos=0 (argmax must match). The cheap 1-node gate
 * before trusting M>1 (which still needs a job for routing/KV-independence). Off unless tok>0. */
static void glm5_batch_selfcheck(glm5_model*m){
    int tok=envi("GLM5_BATCH_SELFCHECK",0); if(tok<=0) return;
    if(glm5_alloc_mstream_ex(m,1,1)){ if(MyRank==0) logmsg("BATCH_SELFCHECK: mstream alloc failed\n"); return; }
    int H=m->cfg.hidden, pos0=0, out=-1; float*x=glm5_amalloc((size_t)H*4);
    embed_lookup(m,tok,x); int single=glm5_forward_token(m,x,0);
    embed_lookup(m,tok,x); glm5_forward_batch_decode_mla(m,x,1,&pos0,&out);
    glm5_afree(x); glm5_free_mstream(m);
    int li=0; while(li<m->cfg.n_layers-1 && !glm5_is_moe(&m->cfg,li)) li++;   /* first MoE layer */
    int nown=m->layers[li].qh1-m->layers[li].qh0, shard=(nown<m->cfg.n_heads);
    if(MyRank==0){
        logmsg("BATCH_SELFCHECK tok=%d single=%d batched=%d %s\n",
               tok,single,out,single==out?"MATCH (M=1 ok; M>1 needs a job)":"*** MISMATCH ***");
        logmsg("BATCH_DECODE AR/MoE-layer=%d (attention %s) -> replicate attention (GLM5_TP_ATTN=0) for 1 AR/layer ~1.95x\n",
               shard?2:1, shard?"sharded":"replicated");
    }
}
/* GLM5_BATCH_SELFCHECK2=<tok>: numerics diff of the FUSED batch layer (GLM5_BD_FUSED path)
 * vs the legacy batched path at M=2 (same-stream verifier shape: sid={0,0}, pos={0,1}).
 * Runs the identical inputs through both (KV rewritten deterministically), reports the
 * max relative X diff and whether the two argmax pairs match. */
static void glm5_batch_selfcheck2(glm5_model*m){
    int tok=envi("GLM5_BATCH_SELFCHECK2",0); if(tok<=0) return;
    if(glm5_alloc_mstream_ex(m,2,1)){ if(MyRank==0) logmsg("BATCH_SELFCHECK2: alloc failed\n"); return; }
    glm5_mstream*ms=(glm5_mstream*)m->ms;
    static const int sid2[2]={0,0}; ms->sid=sid2;
    int H=m->cfg.hidden, pos2[2]={0,1}, outA[2]={-1,-1}, outB[2]={-1,-1};
    float*X=glm5_amalloc((size_t)2*H*4), *XA=glm5_amalloc((size_t)2*H*4);
    embed_lookup(m,tok,X); embed_lookup(m,tok+1,X+H);
    int save=m->bd_fused;
    m->bd_fused=1; glm5_forward_batch_decode_mla(m,X,2,pos2,outA);
    memcpy(XA,X,(size_t)2*H*4);
    embed_lookup(m,tok,X); embed_lookup(m,tok+1,X+H);
    m->bd_fused=0; glm5_forward_batch_decode_mla(m,X,2,pos2,outB);
    m->bd_fused=save;
    double mx=0,rn=0,dn=0;
    for(int i=0;i<2*H;i++){ double d=fabs((double)XA[i]-X[i]); dn+=d*d; rn+=(double)X[i]*X[i]; if(d>mx)mx=d; }
    if(MyRank==0) logmsg("BATCH_SELFCHECK2 tok=%d fused=(%d,%d) legacy=(%d,%d) rel_l2=%.3e max_abs=%.3e %s\n",
                         tok,outA[0],outA[1],outB[0],outB[1],sqrt(dn/(rn+1e-30)),mx,
                         (outA[0]==outB[0]&&outA[1]==outB[1])?"ARGMAX-MATCH":"*** ARGMAX-MISMATCH ***");
    /* M=1 control: same diff with a single token — separates M=2-lane bugs from
     * layer-code (kernel-order) deltas that exist at every M */
    {
        int p0=0, oA=-1, oB=-1;
        embed_lookup(m,tok,X);
        m->bd_fused=1; glm5_forward_batch_decode_mla(m,X,1,&p0,&oA);
        memcpy(XA,X,(size_t)H*4);
        embed_lookup(m,tok,X);
        m->bd_fused=0; glm5_forward_batch_decode_mla(m,X,1,&p0,&oB);
        m->bd_fused=save;
        double mx1=0,rn1=0,dn1=0;
        for(int i=0;i<H;i++){ double d=fabs((double)XA[i]-X[i]); dn1+=d*d; rn1+=(double)X[i]*X[i]; if(d>mx1)mx1=d; }
        if(MyRank==0) logmsg("BATCH_SELFCHECK2-M1 tok=%d fused=%d legacy=%d rel_l2=%.3e max_abs=%.3e %s\n",
                             tok,oA,oB,sqrt(dn1/(rn1+1e-30)),mx1,oA==oB?"ARGMAX-MATCH":"*** ARGMAX-MISMATCH ***");
    }
    glm5_afree(X); glm5_afree(XA); glm5_free_mstream(m);
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
        if(n>=cap){ int oc=cap; cap*=2; ids=glm5_arealloc(ids,(size_t)oc*sizeof(int),(size_t)cap*sizeof(int)); }
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
        if(n>=cap){ int oc=cap; cap*=2; ps=glm5_arealloc(ps,(size_t)oc*sizeof(id_prompt),(size_t)cap*sizeof(id_prompt)); }
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

/* token id -> embedding WITHOUT the TP_EMBED all-reduce (owned rows only; zeros elsewhere).
 * The batched decode path gathers M of these and fires ONE ar over [M,hidden] instead of M. */
static void embed_partial(glm5_model*m,int tok,float*x){
    int H=m->cfg.hidden;
    if(tok<0||tok>=m->cfg.vocab) tok=0;
    if(m->emb_rows<m->cfg.vocab){
        memset(x,0,(size_t)H*4);
        if(tok>=m->emb_r0 && tok<m->emb_r0+m->emb_rows){
            const uint16_t*row=m->embed+(size_t)(tok-m->emb_r0)*H;
            for(int i=0;i<H;i++) x[i]=glm5_bf2f(row[i]);
        }
        return;
    }
    const uint16_t*row=m->embed+(size_t)tok*H;
    for(int i=0;i<H;i++) x[i]=glm5_bf2f(row[i]);
}

/* GLM5_BATCH_DECODE: copy a slot model's prompt latent-KV (positions 0..npos-1, all layers)
 * into the shared mstream's per-stream cache. Bridges the validated per-slot prefill (clone
 * L->kv_cache, same glm5_kv_enc encoding) to the batched kernel's ms->kc. bf16/fp16 KV, CP off. */
static void cbatch_kv_to_stream(glm5_model*root,glm5_model*src,int stream,int npos){
    glm5_mstream*ms=(glm5_mstream*)root->ms; const glm5_config*c=&root->cfg;
    const int KVC=glm5_kv_cache_dim(c); const size_t per=(size_t)c->n_layers*c->max_pos*KVC;
    for(int l=0;l<c->n_layers;l++)
        memcpy(ms->kc+(size_t)stream*per+(size_t)l*c->max_pos*KVC,
               src->layers[l].kv_cache,(size_t)npos*KVC*2);
}

static void cbatch_stream_to_kv(glm5_model*root,int stream,int npos){
    glm5_mstream*ms=(glm5_mstream*)root->ms; const glm5_config*c=&root->cfg;
    const int KVC=glm5_kv_cache_dim(c); const size_t per=(size_t)c->n_layers*c->max_pos*KVC;
    for(int l=0;l<c->n_layers;l++)
        memcpy(root->layers[l].kv_cache,
               ms->kc+(size_t)stream*per+(size_t)l*c->max_pos*KVC,
               (size_t)npos*KVC*2);
}

static void copy_runtime_kv(glm5_model*dst,glm5_model*src,int npos){
    const glm5_config*c=&src->cfg;
    const int KVC=glm5_kv_cache_dim(c), ID=c->index_dim;
    for(int l=0;l<c->n_layers;l++){
        if(dst->layers[l].kv_cache && src->layers[l].kv_cache)
            memcpy(dst->layers[l].kv_cache,src->layers[l].kv_cache,(size_t)npos*KVC*2);
        if(dst->layers[l].idx_k_cache && src->layers[l].idx_k_cache)
            memcpy(dst->layers[l].idx_k_cache,src->layers[l].idx_k_cache,(size_t)npos*ID*2);
    }
}

static void glm5_spec_selftest(glm5_model*m,int cur,int n_prompt,int C){
    if(!envi("GLM5_SPEC_SELFTEST",0)) return;
    if(m->cp_on || m->int4_kv){ if(MyRank==0) logmsg("SPEC_SELFTEST: skip (needs CP off + bf16 KV)\n"); return; }
    glm5_model*seq=glm5_clone_runtime(m);
    if(!seq){ if(MyRank==0) logmsg("SPEC_SELFTEST: clone failed\n"); return; }
    copy_runtime_kv(seq,m,n_prompt);
    glm5_model*chk=glm5_clone_runtime(m);
    if(chk) copy_runtime_kv(chk,m,n_prompt);
    float *xs=glm5_amalloc((size_t)C*4), *h0=glm5_amalloc((size_t)C*4);
    float *h1=glm5_amalloc((size_t)C*4), *xb=glm5_amalloc((size_t)2*C*4), *xc=glm5_amalloc((size_t)2*C*4);
    int out_seq0=-1,out_seq1=-1,out_b[2]={-1,-1}, pos[2]={n_prompt,n_prompt+1}, sid[2]={0,0};
    embed_lookup(seq,cur,xs);
    out_seq0=glm5_forward_token(seq,xs,n_prompt);
    memcpy(h0,xs,(size_t)C*4);
    embed_lookup(seq,out_seq0,xs);
    out_seq1=glm5_forward_token(seq,xs,n_prompt+1);
    memcpy(h1,xs,(size_t)C*4);
    if(glm5_alloc_mstream_ex(m,2,1)){
        if(MyRank==0) logmsg("SPEC_SELFTEST: mstream alloc failed\n");
    } else {
        cbatch_kv_to_stream(m,m,0,n_prompt);
        embed_partial(m,cur,xb);
        embed_partial(m,out_seq0,xb+C);
        if(m->emb_rows<m->cfg.vocab && m->ar_cb) m->ar_cb(xb,2*C,m->ar_ctx);
        glm5_mstream*ms=(glm5_mstream*)m->ms;
        ms->sid=sid;
        glm5_forward_batch_decode_mla(m,xb,2,pos,out_b);
        ms->sid=NULL;
        double s0=0.0,d0=0.0,s1=0.0,d1=0.0,m0=0.0,m1=0.0;
        for(int i=0;i<C;i++){
            double a0=h0[i], b0=xb[i], e0=a0-b0;
            double a1=h1[i], b1=xb[C+i], e1=a1-b1;
            s0+=a0*a0; d0+=e0*e0; if(fabs(e0)>m0) m0=fabs(e0);
            s1+=a1*a1; d1+=e1*e1; if(fabs(e1)>m1) m1=fabs(e1);
        }
        glm5_free_mstream(m);
        if(MyRank==0) logmsg("SPEC_SELFTEST seq=(%d,%d) batch=(%d,%d) %s\n",
                             out_seq0,out_seq1,out_b[0],out_b[1],
                             (out_seq0==out_b[0] && out_seq1==out_b[1])?"MATCH":"*** MISMATCH ***");
        if(MyRank==0) logmsg("SPEC_SELFTEST hidden_relerr row0=%.3e max=%.3e row1=%.3e max=%.3e\n",
                             s0>0.0?sqrt(d0/s0):0.0,m0,s1>0.0?sqrt(d1/s1):0.0,m1);
    }
    if(chk){
        embed_lookup(chk,cur,xc);
        embed_lookup(chk,out_seq0,xc+C);
        if(glm5_alloc_mstream_ex(chk,2,0)){
            if(MyRank==0) logmsg("SPEC_SELFTEST: prefill mstream alloc failed\n");
        } else {
            int out_c=glm5_forward_prefill_chunk(chk,xc,2,n_prompt,1);
            double s=0.0,d=0.0,mx=0.0;
            for(int i=0;i<C;i++){
                double a=h1[i], b=xc[C+i], e=a-b;
                s+=a*a; d+=e*e; if(fabs(e)>mx) mx=fabs(e);
            }
            glm5_free_mstream(chk);
            if(MyRank==0) logmsg("SPEC_SELFTEST prefill2 last=%d expected=%d %s hidden_relerr=%.3e max=%.3e\n",
                                 out_c,out_seq1,out_c==out_seq1?"MATCH":"*** MISMATCH ***",
                                 s>0.0?sqrt(d/s):0.0,mx);
        }
        glm5_free(chk);
    }
    glm5_afree(xs); glm5_afree(h0); glm5_afree(h1); glm5_afree(xb); glm5_afree(xc); glm5_free(seq);
}

static int glm5_run_spec_verify(glm5_model*m,int *gen,int *ng,int max_new,int min_new,
                                int n_prompt,int *curp,float*x,int C,int *nanp,
                                long *hitp,long *totp){
    if(m->cp_on || m->int4_kv || !m->mtp_layer) return 0;
    int K=envi("GLM5_SPEC_K",1); if(K<1)K=1; if(K>2)K=2;   /* K=2: chained 2nd draft, M=3 verify */
    if(glm5_alloc_mstream_ex(m,K+1,1)) return 0;
    cbatch_kv_to_stream(m,m,0,n_prompt);
    float *BX=glm5_amalloc((size_t)(K+1)*C*4), *xb=glm5_amalloc((size_t)2*C*4), *xb2=glm5_amalloc((size_t)2*C*4);
    int pos[3], out[3], sid[3]={0,0,0};
    glm5_mstream*ms=(glm5_mstream*)m->ms;
    ms->sid=sid;
    int cur=*curp, p=n_prompt, draft=glm5_mtp_draft(m,x,cur,p,xb);
    int warm=envi("GLM5_SPEC_WARMUP",8), min_alpha=envi("GLM5_SPEC_MIN_ALPHA_PCT",25);
    long hit=0, tot=0, steps=0; int aborted=0;
    while(*ng<max_new){
        if(draft<0) break;
        int d2=-1;
        if(K==2) d2=glm5_mtp_draft(m,xb,draft,p+1,xb2);   /* chained: drafter hidden feeds draft 2 */
        int M=(d2>=0)?3:2;
        embed_partial(m,cur,BX);
        embed_partial(m,draft,BX+C);
        if(M==3) embed_partial(m,d2,BX+2*C);
        if(m->emb_rows<m->cfg.vocab && m->ar_cb) m->ar_cb(BX,M*C,m->ar_ctx);
        pos[0]=p; pos[1]=p+1; pos[2]=p+2;
        glm5_forward_batch_decode_mla(m,BX,M,pos,out);
        for(int i=0;i<M*C;i++) if(!(BX[i]==BX[i])){ (*nanp)++; break; }
        gen[(*ng)++]=cur;
        int eos0=(cur==GLM5_EOS_ID0||cur==GLM5_EOS_ID1||cur==GLM5_EOS_ID2);
        if(eos0 && *ng>=min_new){ cur=out[0]; p++; memcpy(x,BX,(size_t)C*4); break; }
        tot++;
        if(out[0]==draft && *ng<max_new){
            hit++;
            gen[(*ng)++]=draft;
            int eos1=(draft==GLM5_EOS_ID0||draft==GLM5_EOS_ID1||draft==GLM5_EOS_ID2);
            if(eos1 && *ng>=min_new){ cur=out[1]; p+=2; memcpy(x,BX+C,(size_t)C*4); break; }
            if(M==3 && *ng<max_new){
                tot++;                                     /* 2nd draft judged only after 1st accept */
                if(out[1]==d2){
                    hit++;
                    gen[(*ng)++]=d2;
                    int eos2=(d2==GLM5_EOS_ID0||d2==GLM5_EOS_ID1||d2==GLM5_EOS_ID2);
                    cur=out[2]; p+=3; memcpy(x,BX+2*C,(size_t)C*4);
                    if(eos2 && *ng>=min_new) break;
                } else { cur=out[1]; p+=2; memcpy(x,BX+C,(size_t)C*4); }
            } else { cur=out[1]; p+=2; memcpy(x,BX+C,(size_t)C*4); }
        } else {
            cur=out[0]; p++; memcpy(x,BX,(size_t)C*4);
        }
        draft=glm5_mtp_draft(m,x,cur,p,xb);
        steps++;
        if(warm>0 && tot>=warm && hit*100 < (long)min_alpha*tot){
            cbatch_stream_to_kv(m,0,p);
            aborted=1;
            if(MyRank==0) logmsg("SPEC_VERIFY abort: alpha %ld/%ld below %d%% after warmup; restored %d KV positions\n",
                                 hit,tot,min_alpha,p);
            break;
        }
    }
    ms->sid=NULL;
    glm5_free_mstream(m);
    glm5_afree(BX); glm5_afree(xb); glm5_afree(xb2);
    *curp=cur; *hitp=hit; *totp=tot;
    if(MyRank==0) logmsg("SPEC_VERIFY steps=%ld emitted=%d accept=%ld/%ld %.1f%%\n",
                         steps,*ng,hit,tot,tot?100.0*(double)hit/(double)tot:0.0);
    return aborted?2:1;
}

static int cbatch_start(cb_slot*s,const id_prompt*p,int req,int max_new,int C,double *prefill_sec,double *prefill_ar,long *prefill_calls){
    s->req=req; s->n_prompt=p->n; s->ng=0; s->done=0; s->nan=0; s->cur=0;
    g_ar_secs=0; g_ar_calls=0; g_ar_frags=0;
    double t0=now_sec();
    int last=-1, chunk=envi("GLM5_PCHUNK",512); if(chunk<1)chunk=1;
    if(glm5_alloc_mstream_ex(s->m,chunk,0)) die("cbatch alloc prefill chunk",-1);
    float*Xc=glm5_amalloc((size_t)chunk*C*4);
    for(int p0=0;p0<p->n;p0+=chunk){ int S=p->n-p0; if(S>chunk)S=chunk;
        for(int t=0;t<S;t++) embed_lookup(s->m,p->ids[p0+t],Xc+(size_t)t*C);
        if(!s->m->cp_on && s->m->T_cp>0 && p0+S>s->m->T_cp){
            double tt=now_sec(); glm5_prefill_to_cp(s->m,p0); gbarrier();
            if(GRank==0) logmsg("cbatch_tier: req=%d A->B at pos=%d (%.3f s) CP %s %d slots/rank\n",
                                req,p0,now_sec()-tt,s->m->int4_kv?"int4":"bf16",s->m->cp_nslot);
        }
        int a=glm5_forward_prefill_chunk(s->m,Xc,S,p0,p0+S>=p->n); if(a>=0)last=a;
        memcpy(s->x,Xc+(size_t)(S-1)*C,(size_t)C*4);
    }
    glm5_afree(Xc); glm5_free_mstream(s->m);
    double dt=now_sec()-t0;
    *prefill_sec+=dt; *prefill_ar+=g_ar_secs; *prefill_calls+=g_ar_calls;
    s->cur=last;
    (void)max_new; (void)C;
    return 0;
}

static int run_cbatch(glm5_model*root,const char*batch_file,const char*out_prefix,int max_new,int C);

static void touch_range(void*p,size_t n){
    volatile unsigned char*q=(volatile unsigned char*)p;
    if(!q)return; for(size_t i=0;i<n;i+=4096) q[i]=(unsigned char)(q[i]^1u);
    if(n)q[n-1]=(unsigned char)(q[n-1]^1u);
}
static void touch_context_kv(glm5_model*m){
    const glm5_config*c=&m->cfg; size_t ns=(size_t)m->cp_nslot;
    int KVD=glm5_kv_cache_dim(c),ID=c->index_dim;
    for(int l=0;l<c->n_layers;l++){ glm5_layer*L=&m->layers[l]; int moe=glm5_is_moe(c,l);
        if(m->int4_kv){ touch_range(L->k_q4,ns*(KVD/2)); touch_range(L->k_qs,ns*2);
            if(moe){ touch_range(L->idx_q4,ns*(ID/2)); touch_range(L->idx_qs,ns*2); } }
        else { touch_range(L->kv_cache,ns*KVD*2); if(moe)touch_range(L->idx_k_cache,ns*ID*2); }
    }
}
static void run_serve_capacity_probe(glm5_model*root,int slots){
    glm5_model**p=glm5_acalloc((size_t)slots,sizeof(*p));
    for(int s=0;s<slots;s++){
        p[s]=glm5_clone_runtime(root); if(!p[s])die("serve capacity clone",-1);
        if(!p[s]->cp_on && p[s]->T_cp>0)glm5_prefill_to_cp(p[s],0);
        touch_context_kv(p[s]); gbarrier();
    }
    float mb=(float)(glm5_meminfo_bytes("MemAvailable")/(1024L*1024L)), neg=-mb;
    tp_allreduce_max((tp_comm*)root->ar_ctx,&neg,1); float min_mb=-neg;
    if(MyRank==0)logmsg("SERVE_CAPACITY slots=%d ctx=%d tier=%s min_MemAvailable=%.0f MB\n",
                        slots,root->cfg.max_pos,p[0]->int4_kv?"int4":"bf16",min_mb);
    int floor_mb=envi("GLM5_SERVE_MIN_AVAILABLE_MB",1024);
    for(int s=0;s<slots;s++)glm5_free(p[s]); glm5_afree(p); gbarrier();
    if(min_mb<(float)floor_mb)die("serve capacity below MemAvailable floor",-1);
}

/* Persistent filesystem control plane. Only rank 0 polls; a one-float allreduce broadcasts the
 * sequence to the other ranks. Request files are atomically renamed into place by the HTTP
 * process and contain: SEQ MAX_NEW SLOTS PROMPTS_PATH OUT_PREFIX. */
static int run_serve(glm5_model*root,const char*dir,int max_slots,int C){
    const float stop_cmd=16777215.0f; /* largest exactly represented positive integer below 2^24 */
    if(envi("GLM5_SERVE_CAPACITY_PROBE",0))run_serve_capacity_probe(root,max_slots);
    if(max_slots<1)max_slots=1;
    if(MyRank==0){ mkdir(dir,0700); char p[512]; snprintf(p,sizeof p,"%s/ready",dir);
        FILE*f=fopen(p,"w"); if(f){ fprintf(f,"pid=%ld slots=%d max_pos=%d\n",(long)getpid(),max_slots,root->cfg.max_pos); fclose(f); } }
    barrier(); long last=0;
    if(MyRank==0) logmsg("serve: ready dir=%s slots=%d max_pos=%d\n",dir,max_slots,root->cfg.max_pos);
    for(;;){
        float cmd=0.0f;
        if(MyRank==0){
            char stop[512],req[512]; snprintf(stop,sizeof stop,"%s/stop",dir);
            if(access(stop,F_OK)==0) cmd=stop_cmd;
            else { snprintf(req,sizeof req,"%s/request",dir); FILE*f=fopen(req,"r");
                long seq=0; if(f){ if(fscanf(f,"%ld",&seq)==1 && seq>last) cmd=(float)seq; fclose(f); } }
        }
        tp_allreduce_max((tp_comm*)root->ar_ctx,&cmd,1);
        if(cmd==stop_cmd) break;
        long seq=(long)cmd; if(seq<=last){ usleep(50000); continue; }
        char req[512],prompts[512],prefix[512]; int max_new=0,slots=0; long file_seq=0;
        /* LLIO metadata visibility can lag an atomic rename on sibling compute nodes. Rank 0
         * already observed the new sequence, but peers may briefly open the previous descriptor
         * (or no descriptor). Retry until this exact sequence and its prompt file are visible. */
        snprintf(req,sizeof req,"%s/request",dir); int parsed=0;
        for(int a=0;a<300&&!parsed;a++){
            FILE*f=fopen(req,"r"); int nf=0;
            if(f){ nf=fscanf(f,"%ld %d %d %511s %511s",&file_seq,&max_new,&slots,prompts,prefix); fclose(f); }
            struct stat st;
            if(nf==5 && file_seq==seq && stat(prompts,&st)==0 && st.st_size>0) parsed=1;
            else usleep(10000);
        }
        if(!parsed) die("serve request visibility timeout",-1);
        if(file_seq!=seq || max_new<1 || slots<1 || slots>max_slots) die("serve invalid request",-1);
        barrier();
        char sb[32]; snprintf(sb,sizeof sb,"%d",slots); setenv("GLM5_CBATCH_SLOTS",sb,1);
        int rc=run_cbatch(root,prompts,prefix,max_new,C); barrier();
        if(MyRank==0){ char tmp[512],done[512]; snprintf(tmp,sizeof tmp,"%s/done.tmp",dir); snprintf(done,sizeof done,"%s/done",dir);
            FILE*df=fopen(tmp,"w"); if(df){ fprintf(df,"%ld %d\n",seq,rc); fclose(df); rename(tmp,done); } }
        last=seq;
    }
    if(MyRank==0) logmsg("serve: stopped after seq=%ld\n",last);
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

    /* GLM5_BATCH_DECODE=1: ONE glm5_forward_batch_decode_mla over all active slots per step
     * (one AR/layer serves M tokens) instead of M single-token forwards (M ARs/layer). */
    int bd=envi("GLM5_BATCH_DECODE",0);
    float *BX=NULL; int *bdx=NULL,*bpos=NULL,*bout=NULL,*bhn=NULL; const int**bh=NULL;
    if(bd){
        if(root->int4_kv) die("GLM5_BATCH_DECODE needs bf16/fp16 KV (GLM5_INT4_KV=0)",-1);
        if(root->cp_on)   die("GLM5_BATCH_DECODE needs CP off",-1);
        if(glm5_alloc_mstream_ex(root,slots,1)) die("GLM5_BATCH_DECODE mstream alloc",-1);
        BX=glm5_amalloc((size_t)slots*C*4);
        bdx=glm5_amalloc((size_t)slots*sizeof(int));  bpos=glm5_amalloc((size_t)slots*sizeof(int));
        bout=glm5_amalloc((size_t)slots*sizeof(int)); bhn=glm5_amalloc((size_t)slots*sizeof(int));
        bh=glm5_amalloc((size_t)slots*sizeof(const int*));
        if(MyRank==0) logmsg("cbatch: BATCH_DECODE on (M<=%d per forward, per-stream KV %.2f GB)\n",
                             slots,(double)slots*root->cfg.n_layers*root->cfg.max_pos*glm5_kv_cache_dim(&root->cfg)*2/1e9);
    }

    double prof0[GLM5_NPHASE], prof1[GLM5_NPHASE];
    prof_sum_models(S,slots,prof0);
    double pf_sec=0.0,pf_ar=0.0,svc_ar=0.0; long pf_calls=0,svc_calls=0;
    int next=0,done=0,active=0,total_gen=0,total_nan=0,total_prompt=0;
    for(int s=0;s<slots && next<n_req;s++,next++){
        total_prompt+=prompts[next].n;
        cbatch_start(&S[s],&prompts[next],next,max_new,C,&pf_sec,&pf_ar,&pf_calls);
        if(bd) cbatch_kv_to_stream(root,S[s].m,s,S[s].n_prompt);
        active++;
    }
    barrier();
    g_ar_secs=0; g_ar_calls=0; g_ar_frags=0;
    double svc0=now_sec();
    while(bd && done<n_req){
        /* ---- batched service loop: emit/retire/refill per slot, then ONE forward for all ---- */
        int M=0;
        for(int s=0;s<slots;s++){
            cb_slot *q=&S[s];
            if(q->req<0) continue;
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
                    cbatch_kv_to_stream(root,q->m,s,q->n_prompt);
                    next++; active++;
                }
                continue;   /* refilled slot joins the batch next iteration (as per-slot path) */
            }
            bdx[M]=s; M++;
        }
        if(M==0) continue;  /* retire/refill pass only; loop re-evaluates (no active streams) */
        glm5_mstream*ms=(glm5_mstream*)root->ms;
        for(int i=0;i<M;i++){
            cb_slot *q=&S[bdx[i]];
            embed_partial(root,q->cur,BX+(size_t)i*C);
            bpos[i]=q->n_prompt+q->ng-1;
            bh[i]=q->gen; bhn[i]=q->ng;
        }
        /* TP_EMBED: one AR gathers all M embeddings (embed_partial left owned rows only) */
        if(root->emb_rows<root->cfg.vocab && root->ar_cb) root->ar_cb(BX,M*C,root->ar_ctx);
        ms->hist=bh; ms->hist_n=bhn;                 /* per-stream repetition-penalty history */
        glm5_forward_batch_decode_mla(root,BX,M,bpos,bout);
        for(int i=0;i<M;i++){
            cb_slot *q=&S[bdx[i]]; q->cur=bout[i];
            for(int k=0;k<C;k++) if(!(BX[(size_t)i*C+k]==BX[(size_t)i*C+k])){ q->nan++; break; }
        }
    }
    while(!bd && done<n_req){
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
            q->m->samp_hist=q->gen; q->m->samp_hist_n=q->ng;   /* per-slot repetition penalty */
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
    if(bd){ glm5_afree(BX); glm5_afree(bdx); glm5_afree(bpos); glm5_afree(bout);
            glm5_afree(bhn); glm5_afree((void*)bh); glm5_free_mstream(root); }
    for(int s=0;s<slots;s++){ glm5_afree(S[s].gen); glm5_afree(S[s].x); glm5_free(S[s].m); }
    glm5_afree(S);
    for(int r=0;r<n_req;r++) glm5_afree(prompts[r].ids);
    glm5_afree(prompts);
    return 0;
}

/* ---- CLI front-end: map --flags to the existing GLM5_* env config, so the runner is driven by
 * command-line args instead of `export` soup. Named flags cover the common knobs; `--set K=V` reaches
 * any of the ~100 GLM5_* vars. Parsed FIRST in main (before any envi() read) via setenv, so the 100+
 * existing config sites are untouched and no-args behavior is byte-identical (backward compatible). */
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
/* NUMA-local weight placement (the measured 1.40x e2e, bit-identical lever): interleave THIS process's
 * future allocations across all CMGs -- in-process equivalent of `numactl --interleave=all`, so no
 * wrapper is needed. OMP thread affinity is read by the runtime at init, so the launch script still
 * exports OMP_PROC_BIND=close / OMP_PLACES=cores (best-effort setenv here as a fallback). Default ON. */
static void glm5_apply_numa(int on){
    /* Thread pinning is NOT optional and must not ride on the interleave flag: first-touch
     * placement is only meaningful if a thread stays on one CMG.  (Previously the `if(!on)
     * return` below skipped these too, so GLM5_NUMA=0 silently unpinned the threads and
     * scattered placement.) */
    setenv("OMP_PROC_BIND","close",0);           /* 0 = don't override if the script already set it */
    setenv("OMP_PLACES","cores",0);
    if(!on) return;
    /* Interleave over the NUMA nodes that hold OUR cpus only.  On A64FX nodes 0-3 are the
     * tiny assistant-core nodes (~700 MB each); a ~0UL mask makes the kernel round-robin
     * weight pages onto them until they fill.  NOTE: page placement only follows this
     * policy under XOS demand paging (XOS_MMM_L_PAGING_POLICY=demand:demand:demand, set
     * by the launcher); the default prepage policy puts every heap page on the allocating
     * thread's CMG and caps multi-CMG streaming at ~94 GB/s (measured; demand = 843). */
    unsigned long nodemask=0;
    cpu_set_t aff;
    if(sched_getaffinity(0,sizeof aff,&aff)==0){
        for(int nd=0;nd<64;nd++){
            char p[128]; snprintf(p,sizeof p,"/sys/devices/system/node/node%d/cpulist",nd);
            FILE*f=fopen(p,"r"); if(!f) continue;
            char buf[256]={0};
            if(fgets(buf,sizeof buf,f)){
                char*s=buf;
                while(*s && *s!='\n'){
                    char*e; long a=strtol(s,&e,10); if(e==s) break;
                    long b=a; s=e;
                    if(*s=='-'){ b=strtol(s+1,&e,10); s=e; }
                    for(long cc=a;cc<=b;cc++)
                        if(cc>=0 && cc<CPU_SETSIZE && CPU_ISSET((int)cc,&aff)){ nodemask|=1UL<<nd; break; }
                    if(*s==',') s++; else break;
                }
            }
            fclose(f);
        }
    }
    if(!nodemask) nodemask=~0UL;
    syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask, (unsigned long)(8*sizeof nodemask));
}
/* Production defaults for the 12-node GLM-5.2 Q2 path, baked so the runner reproduces the tuned
 * numbers in GLM52_Q2_12N.md with ZERO env and ZERO flags.  Each is setenv(...,0) = no-overwrite,
 * so a real env var (debug) still wins, and a CLI flag (which setenv(...,1) overwrites) wins over
 * both.  This is the pragmatic bridge: the deep forward/kernel code keeps reading glm5_envi, but
 * the value now comes from here or from a flag -- the operator never has to export anything.
 * FLIB_BARRIER / OMP_* / XOS_MMM_L_PAGING_POLICY are library-runtime env set by the launcher
 * before the process starts (they cannot be program args) -- see run_glm52_q2_12n.sh. */
static void glm5_bake_defaults(void){
    static const char *kv[][2] = {
        {"GLM5_REAL","1"},                                  /* staged real BF16/IQ weights */
        {"GLM5_TP","1"},{"GLM5_TP_ATTN","1"},{"GLM5_TP_SHARED","1"},
        {"GLM5_TP_FFN","1"},{"GLM5_TP_HEAD","1"},{"GLM5_TP_EMBED","1"},
        {"GLM5_PREFILL_GROUPS","1"},                        /* one group over the 12 ranks */
        {"GLM5_IQ_MODE","1"},{"GLM5_IQ_REF","1"},           /* q8 SDOT expert kernels */
        {"GLM5_DENSE_I8","1"},                              /* hoisted-a16 int8 dense (was manual) */
        {"TP_AR_BF16","1"},{"TP_AR_ROBUST","2"},            /* lean bf16 allreduce */
        {"GLM5_BF16_GEMM_TOK","5"},                         /* 5-token BF16 GEMM register block */
        {"GLM5_ATTN_QK","1"},
        {"GLM5_CP_THRESHOLD","0"},                          /* 0 = auto KV tiering (long-ctx safe;
                                                             * Tier-A-only when it fits, identical
                                                             * to the legacy -1 static path) */
        {"GLM5_MAXPOS","2304"},{"GLM5_PCHUNK","512"},
        {NULL,NULL}
    };
    for(int i=0;kv[i][0];i++) setenv(kv[i][0],kv[i][1],0);
}
static void glm5_cli_usage(void){
    fprintf(stderr,
      "glm5_ep_runner [--flags]  (arg-driven; tuned 12n-Q2 defaults are baked in -- no env needed)\n"
      "\n model / run:\n"
      "  --model DIR         weights dir          --stage-dir DIR   node-local staged blobs\n"
      "  --layers N          0=full (78)          --experts N       0=full (256)\n"
      "  --threads N         OMP/worker threads   --maxpos N (--ctx N) KV positions (default 2304)\n"
      "  --pchunk N          prefill chunk (512)  --prefill-only[=1] prefill benchmark, no decode\n"
      "  --numa[=0|1]        NUMA-interleave weights (default ON; bit-identical 1.40x lever)\n"
      "  --real N            0=synthetic structural benchmark path\n"
      "\n generation:\n"
      "  --max-new N         decode tokens        --min-new N       floor decode tokens\n"
      "  --prompt-ids FILE   space-separated ids  --prompt-text TEXT   render+tokenize GLM5.3F\n"
      "  --prompt-tokens FILE  packed uint32 prompt\n"
      "  --gen-out FILE      write generated ids  --gen-new N       generate-after-prefill count\n"
      "  --kv-save FILE      save KV after prefill --kv-load FILE    resume from saved KV\n"
      "  --temp F --topp F --rep-pen F --seed N    sampler (temp<=0 => greedy)\n"
      "\n perf / quality:\n"
      "  --active-experts N  top-k MoE (default 8 exact; 3 => ~20 tok/s, changes output)\n"
      "  --dense-i8[=0|1]    hoisted int8 dense (default ON)   --dense-i8-gs N  group size (64; 256=prefill+)\n"
      "  --gemm-sdot N       prefill/batched GEMM (2=int16, lossless 1.2-1.4x)  --sdot N  M=1 decode (0=w8a16)\n"
      "\n long context (256K+):\n"
      "  --cp-threshold N    KV tier switch position (0=auto budget; -1=legacy static off)\n"
      "  --kv-budget-gb N    Tier-A bf16 KV cap/node (0=auto from MemAvailable)\n"
      "  --kv-tier-bf16[=1]  exact bf16 CP-sharded Tier-B (default int4; fits <=256K on 12n)\n"
      "  --stable-outputs[=1]  fixed-tree + BF16 Tier-B + serial absorbed attention\n"
      "\n comm calibration:\n"
      "  --ar-probe[=1]      all-reduce latency probe (skips prefill/decode)\n"
      "  --ar-2d A           also probe the 2-level AR: A groups of N/A (A must divide the group)\n"
      "\n batched serving / misc:\n"
      "  --batch-decode[=1]  --overlap[=1]  --slots N  --prompts FILE  --out-prefix PATH\n"
      "  --serve-dir DIR     persistent request directory  --nshards N --ep-size N\n"
      "  --set KEY=VAL       escape hatch: set any GLM5_* var directly\n");
}
static void glm5_cli(int argc,char**argv){
    /* GLM5_NUMA=0 disables the process-wide MPOL_INTERLEAVE.  That interleave dates from the
     * XOS *prepage* era, when first-touch put every weight page on the allocating thread's CMG
     * (1 CMG = ~94 GB/s) so round-robin was a 1.40x win.  Under demand paging (the launcher's
     * default since 2026-07-16) first-touch places each thread's rows on ITS OWN CMG, and the
     * interleave instead forces ~75% of every read across the CMG ring. */
    int numa=envi("GLM5_NUMA",1);
    for(int i=1;i<argc;i++){
        char*a=argv[i]; if(strncmp(a,"--",2)) continue; a+=2;
        char*eq=strchr(a,'='); char*val=NULL;
        if(eq){ *eq=0; val=eq+1; }
        /* Accept NEGATIVE values: "-1" starts with '-', so a plain argv[i+1][0]!='-' test silently
         * dropped it and left the flag unset (e.g. `--cp-threshold -1` became a no-op that kept the
         * baked default).  Treat a leading '-' followed by a digit/dot as a value, not a flag. */
        else if(i+1<argc && (argv[i+1][0]!='-' ||
                             ((argv[i+1][1]>='0'&&argv[i+1][1]<='9')||argv[i+1][1]=='.'))){ val=argv[++i]; }
        if(!strcmp(a,"help")){ glm5_cli_usage(); exit(0); }
        if(!strcmp(a,"numa")){ numa=val?atoi(val):1; continue; }
        /* boolean flags default to "1" when given bare (--dense-i8 == --dense-i8=1) */
        #define BFLAG(flag,var) if(!strcmp(a,flag)){ setenv(var,val?val:"1",1); continue; }
        BFLAG("batch-decode","GLM5_BATCH_DECODE") BFLAG("overlap","GLM5_COMM_OVERLAP")
        BFLAG("dense-i8","GLM5_DENSE_I8")         BFLAG("prefill-only","GLM5_PREFILL_ONLY")
        BFLAG("kv-tier-bf16","GLM5_KV_TIER_BF16") BFLAG("stable-outputs","GLM5_STABLE_OUTPUTS") BFLAG("ar-probe","GLM5_AR_PROBE")
        #undef BFLAG
        if(!strcmp(a,"set")&&val){ char*e=strchr(val,'='); if(e){*e=0; setenv(val,e+1,1);} continue; }
        #define MAP(flag,var) if(!strcmp(a,flag)){ if(val) setenv(var,val,1); continue; }
        /* model / run */
        MAP("model","GLM5_MODEL_DIR")   MAP("layers","GLM5_LAYERS")   MAP("experts","GLM5_EXPERTS")
        MAP("maxpos","GLM5_MAXPOS")     MAP("ctx","GLM5_MAXPOS")      MAP("threads","LLM_THREADS")
        MAP("pchunk","GLM5_PCHUNK")     MAP("stage-dir","GLM5_STAGE_DIR") MAP("real","GLM5_REAL")
        MAP("status-dir","GLM5_STATUS_DIR") MAP("prefill-synth","GLM5_PREFILL_SYNTH")
        MAP("tp","GLM5_TP")             MAP("tp-shared","GLM5_TP_SHARED")
        /* generation */
        MAP("max-new","GLM5_MAX_NEW")   MAP("min-new","GLM5_MIN_NEW") MAP("gen-new","GLM5_GEN_NEW")
        MAP("prompt-ids","GLM5_PROMPT_IDS") MAP("prompt-text","GLM5_PROMPT_TEXT") MAP("prompt-tokens","GLM5_PROMPT_TOKENS")
        MAP("gen-out","GLM5_GEN_OUT")   MAP("kv-save","GLM5_KV_SAVE") MAP("kv-load","GLM5_KV_LOAD")
        MAP("temp","GLM5_TEMP")         MAP("topp","GLM5_TOPP")       MAP("rep-pen","GLM5_REP_PEN")
        MAP("seed","GLM5_SEED")
        /* perf / quality */
        MAP("active-experts","GLM5_ACTIVE_EXPERTS") MAP("dense-i8-gs","GLM5_DENSE_I8_GS")
        MAP("gemm-sdot","GLM5_GEMM_SDOT") MAP("sdot","GLM5_MV_SDOT")
        /* long context */
        MAP("cp-threshold","GLM5_CP_THRESHOLD") MAP("kv-budget-gb","GLM5_KV_BUDGET_GB")
        /* comm calibration */
        MAP("ar-2d","GLM5_AR_2D")
        /* batched serving / misc */
        MAP("slots","GLM5_CBATCH_SLOTS") MAP("prompts","GLM5_CBATCH_PROMPTS")
        MAP("out-prefix","GLM5_CBATCH_OUT_PREFIX")
        MAP("serve-dir","GLM5_SERVE_DIR")
        MAP("nshards","GLM5_NSHARDS")    MAP("ep-size","GLM5_EP_SIZE")
        #undef MAP
        fprintf(stderr,"glm5_ep_runner: unknown flag --%s (try --help)\n",a);
        exit(2);
    }
    if(envi("GLM5_STABLE_OUTPUTS",0)){
        setenv("TP_AR_DETERMINISTIC","1",1);
        /* BF16 Tier-B is the quality default, but full-weight 256K exceeds the
         * 32 GB HBM margin. Preserve an explicit --kv-tier-bf16=0 override. */
        if(!getenv("GLM5_KV_TIER_BF16")) setenv("GLM5_KV_TIER_BF16","1",1);
        setenv("GLM5_INT8_SDOT","0",1);
        setenv("GLM5_GEMM_SDOT","0",1);
        setenv("GLM5_MV_SDOT","0",1);
        setenv("GLM5_DENSE_I8","0",1);
    }
    glm5_apply_numa(numa);
}

int main(int argc,char**argv){
    glm5_bake_defaults();
    glm5_cli(argc,argv);
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
    int ar_auto_cap=envi("GLM5_AR_AUTO_CAP",512);
    int pchunk0=envi("GLM5_PCHUNK",0);
    int cp_lean_slot=(envi("GLM5_CP",0) || maxpos>65536);
    if(!ar_env && cp_lean_slot){
        ar_tokens=1;  /* provisional lean slot; resized below (post-cfg) to cover the chunk combine */
    } else {
        if(ar_tokens<mstream) ar_tokens=mstream;
        if(pchunk0>ar_tokens) ar_tokens=pchunk0;
        if(!ar_env && ar_auto_cap>0 && ar_tokens>ar_auto_cap) ar_tokens=ar_auto_cap;
    }
    if(ar_tokens<1) ar_tokens=1;
    int ar_hard_cap=envi("GLM5_AR_HARD_CAP",512);  /* bigger AR messages = fewer uTofu puts: cap
        64->512 cut prefill comm ~2.5x (24%->11%, +13% tok/s) at ~144MB region vs 18MB. Latency-bound
        at small messages. Raise to the chunk size (one message/allreduce) where memory allows. */
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
    glm5_config cfg=glm5_default_config(); cfg.max_pos=maxpos;
    if(layers>0) cfg.n_layers=layers;
    if(nexp>0)   cfg.n_experts=nexp;
    { int na=envi("GLM5_ACTIVE_EXPERTS",cfg.n_active); if(na>0) cfg.n_active=na; }
    if(cfg.n_active>cfg.n_experts) cfg.n_active=cfg.n_experts;
    /* data-parallel groups: G independent models over the N ranks (group size from ctx+MemAvailable;
     * GLM5_PREFILL_GROUPS overrides). ep_rank/ep_size become group-local so each group is complete. */
    int G=pick_groups(&cfg,N,prefill);
    { int ge=envi("GLM5_PREFILL_GROUPS",0); if(ge>0) G=ge; }
    if(G<1) G=1; while(N%G) G--;
    GSize=N/G; GId=MyRank/GSize; GBase=GId*GSize; GRank=MyRank-GBase;
    int ep_rank=GRank, ep_size=GSize;
    if(GRank==0){ char ln[64]; if(GId==0) snprintf(ln,sizeof ln,"glm5_ep_rank00.txt");
                  else snprintf(ln,sizeof ln,"glm5_ep_g%02d_rank00.txt",GId); g_log=fopen(ln,"w"); }
    if(start_pos+prefill+maxgen>cfg.max_pos) die("start_pos+prefill+maxgen exceeds max_pos",-1);
    /* CP/long-context keep the registered allreduce slot lean, but it must still cover the
     * per-token attention flash-combine (n_heads + n_heads*v_head_dim floats reduced per token).
     * If max_count < that payload, the per-token reduce already fragments and the batched combine
     * cannot merge -> no comm win. Size the slot to a chunk's combine payload, bounded by the hard
     * cap (region ~ (1+TP_AR_NSTEP)*hidden*hard_cap*4 ~ 18 MB at cap 64 -> no extra memory risk
     * vs the non-CP path). Only when the user didn't pin GLM5_AR_TOKENS. */
    if(!ar_env && cp_lean_slot){
        int chunk=pchunk0>0?pchunk0:mstream; if(chunk<1)chunk=1;
        long need=(long)cfg.n_heads*(1+cfg.v_head_dim)*chunk;   /* (nh + nh*hd) floats per token */
        int t=(int)((need+cfg.hidden-1)/cfg.hidden); if(t<1)t=1;
        if(ar_hard_cap>0 && t>ar_hard_cap) t=ar_hard_cap;
        ar_tokens=t;
    }

    int no=glm5_n_owned(cfg.n_experts,ep_rank,ep_size);
    size_t arena_est=glm5_arena_size(&cfg,ep_rank,ep_size);
    if(MyRank==0)
        logmsg("=== GLM5 EP runner: %d ranks, %d groups x %d (group-parallel prefill) ===\n",N,N/GSize,GSize);
    if(GRank==0)
        logmsg("group %d/%d (ranks %d..%d): layers=%d hidden=%d experts=%d active=%d owned~%d ep_size=%d\n"
               "threads=%d start_pos=%d prefill=%d decode=%d max_pos=%d  arena~%.2f GB/node\n",
               GId,N/GSize,GBase,GBase+GSize-1,cfg.n_layers,cfg.hidden,cfg.n_experts,cfg.n_active,no,ep_size,
               n_threads,start_pos,prefill,maxgen,maxpos,arena_est/(1024.0*1024.0*1024.0));
    if(GRank==0){ FILE*mf=fopen("/proc/meminfo","r"); if(mf){ char line[128];
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
    GBAR_BASE=BAR_BASE+(size_t)(N+1)*SlotB;                /* group-barrier slots, disjoint from global */
    size_t region_sz=GBAR_BASE+(size_t)(N+1)*SlotB;
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
    if(GRank==0) logmsg("allreduce: max_count=%d floats ar_tokens=%d pchunk=%d mstream=%d ar_auto_cap=%d ar_hard_cap=%d\n",
                         cfg.hidden*ar_tokens,ar_tokens,pchunk0,mstream,ar_auto_cap,ar_hard_cap);
    if(tp_comm_init(&comm,Vcq,PeerVcq+GBase,GRank,GSize,cfg.hidden*ar_tokens,gbarrier)!=0) die("tp_comm_init",-1);
    m->ar_cb=ep_ar_callback; m->ar_ctx=&comm;
    m->ar_argmax_cb=ep_argmax_callback; m->ar_argmax_ctx=&comm;
    m->ar_argmax_n_cb=ep_argmax_n_callback; m->ar_argmax_n_ctx=&comm;
    if(envi("GLM5_AR_PROBE",0)){       /* comm calibration only; skips prefill/decode */
        run_ar_probe(&comm,cfg.hidden);
        /* GLM5_AR_2D=A: also probe the 2-level AR (row=B contiguous, col=A stride-B). Frees the flat
         * comm's TP_AR_STAG region first so the 2D row can re-register it; needs A | GSize. */
        int a2d=envi("GLM5_AR_2D",0);
        if(a2d>0){
            if(GSize%a2d!=0){ if(MyRank==0) logmsg("ARPROBE2D,skip: A=%d does not divide group %d\n",a2d,GSize); }
            else {
                tp_comm_free(&comm); barrier();
                static tp_comm row,col;
                if(tp_comm_init_2d(&row,&col,Vcq,PeerVcq+GBase,GRank,GSize,a2d,cfg.hidden*ar_tokens,gbarrier)!=0)
                    die("tp_comm_init_2d",-1);
                run_ar_probe_2d(&row,&col,cfg.hidden,a2d,GSize/a2d);
                tp_comm_free_2d(&row,&col);
            }
        }
        if(MyRank==0) logmsg("SENTINEL glm5_ar_probe_%dn=done\n",N);
        barrier();
        return 0;
    }
    /* ---- LOCKSTEP: make the Tier A->B transition point identical on every rank ----
     * glm5_kv_init derives T_cp from THIS NODE's /proc/meminfo MemAvailable, which genuinely
     * differs across nodes (each staged a different 24.5 GB blob, so page cache differs).  The
     * transition flips m->cp_on, which GATES a collective (the CP kv-combine), so a divergent
     * T_cp makes some ranks issue the combine while others do not -> mismatched collectives ->
     * the all-reduce/barrier waits out -> "barrier fan-in (rc=-1)".  Block alignment (128 pos ~
     * 14 MB of budget) hides small skew, but not the 100s of MB staging can leave.
     * Reduce to the MIN across ranks (min via max-of-negation).  Safe wrt the ALREADY-allocated
     * Tier-A KV: every rank's buffer was sized for its own T_cp >= the min (a rank that needed no
     * tiering allocated the full ctx), so shrinking the transition point never overflows.
     * A rank that thought it fit (T_cp==0) must also switch on MSA once any rank must tier. */
    /* UNCONDITIONAL: the reduce itself must be lockstep, so it may not be gated on any per-rank
     * value (T_cp/msa_on are exactly the values that diverge).  All-ranks-fit collapses to the
     * sentinel and changes nothing. */
    {   float v = (m->T_cp>0) ? (float)m->T_cp : 1e30f;   /* +big sentinel = "no tiering needed" */
        float nv = -v; tp_allreduce_max(&comm,&nv,1); v = -nv;      /* -> global MIN */
        if(v < 1e29f){
            int T=(int)v;
            if(m->T_cp==0){ m->msa_on=1; }   /* this rank fit, but another must tier: join it */
            if(T!=m->T_cp && GRank==0)
                logmsg("T_cp sync: local %d -> global min %d (per-node MemAvailable skew)\n",m->T_cp,T);
            m->T_cp=T;
        }
    }
    /* decode sampling (off by default; temp<=0 => greedy argmax). Lockstep: identical seed on
     * every rank => identical token. Only effective when the lm_head is replicated. */
    { const char*te=getenv("GLM5_TEMP"); const char*tp=getenv("GLM5_TOPP"); const char*rp=getenv("GLM5_REP_PEN");
      const char*sd=getenv("GLM5_SEED");
      m->samp_temp   = (te&&*te)? (float)atof(te) : 0.0f;
      m->samp_topp   = (tp&&*tp)? (float)atof(tp) : 1.0f;
      m->samp_rep_pen= (rp&&*rp)? (float)atof(rp) : 1.0f;
      uint64_t seed  = (sd&&*sd)? (uint64_t)strtoull(sd,NULL,10) : 0x9E3779B97F4A7C15ULL;
      if(!seed) seed = 0x9E3779B97F4A7C15ULL;
      m->samp_rng=seed; m->samp_hist=NULL; m->samp_hist_n=0; m->samp_idx=NULL;
      if(m->samp_temp>0.0f){
          m->samp_idx=(int*)malloc((size_t)m->cfg.vocab*sizeof(int));
          if(MyRank==0) logmsg("sampling ON: temp=%.3f top_p=%.3f rep_pen=%.3f seed=%llu (head %s)\n",
              m->samp_temp,m->samp_topp,m->samp_rep_pen,(unsigned long long)seed,
              m->head.rows==m->cfg.vocab?"replicated":"sharded->full-logit gather");
      }
    }
    glm5_batch_selfcheck(m);   /* GLM5_BATCH_SELFCHECK=<tok>: M=1 batched-decode == single-stream */
    glm5_batch_selfcheck2(m);  /* GLM5_BATCH_SELFCHECK2=<tok>: fused-batch vs legacy at M=2 */
    /* CP callbacks are wired unconditionally so a mid-run Tier A->B transition can turn CP on.
     * They stay dormant while m->cp_on==0 (the forward gates the combine/block-reduce on it). */
    /* combine scratch sized for a whole chunk (S<=pchunk0): g_kvbuf holds the packed
     * [S*(nh+nh*hd)] sum payload, g_kvmax the [S*nh] gathered max. NOTE: under CP ar_tokens is
     * forced to 1 (small registered slot), so size by the CHUNK, not ar_tokens. The batch combine
     * is only called from the prefill-chunk path, where S<=pchunk0; >=1 covers the no-chunk case. */
    int kv_chunk=pchunk0>1?pchunk0:1; if(kv_chunk<mstream) kv_chunk=mstream;
    g_kvbuf=(float*)glm5_amalloc((size_t)kv_chunk*(cfg.n_heads + cfg.n_heads*cfg.head_dim)*sizeof(float));
    g_kvmax=(float*)glm5_amalloc((size_t)kv_chunk*cfg.n_heads*sizeof(float));
    m->blk_reduce_cb=ep_blk_reduce; m->blk_reduce_ctx=&comm;
    m->kv_combine_cb=ep_kv_combine; m->kv_combine_ctx=&comm;
    /* batched chunk combine (default on); GLM5_CP_COMBINE_BATCH=0 leaves it NULL -> the forward
     * falls back to the per-token kv_combine_cb loop (clean A/B without a recompile). */
    m->kv_combine_batch_cb=NULL; m->kv_combine_batch_ctx=NULL;
    if(envi("GLM5_CP_COMBINE_BATCH",1)){ m->kv_combine_batch_cb=ep_kv_combine_batch; m->kv_combine_batch_ctx=&comm; }
    if(MyRank==0){
        if(m->T_cp>0) logmsg("CP TIERED: Tier A (cp_on=0 bf16, %d slots) -> transition at pos=%d -> Tier B (CP %s, block=%d over %d ranks)\n",
                             m->cp_nslot,m->T_cp,envi("GLM5_KV_TIER_BF16",0)?"bf16":"int4",m->cp_block,N);
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

    const char*serve_dir=getenv("GLM5_SERVE_DIR");
    if(serve_dir&&*serve_dir){
        int serve_slots=envi("GLM5_CBATCH_SLOTS",1);
        int rc2=run_serve(m,serve_dir,serve_slots,C);
        glm5_afree(x); glm5_free(m); return rc2;
    }

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
    const char*prompt_text=getenv("GLM5_PROMPT_TEXT");
    const char*gen_out=getenv("GLM5_GEN_OUT");
    if((prompt_file&&*prompt_file)||(prompt_text&&*prompt_text)){
        int max_new=envi("GLM5_MAX_NEW",64);
        int min_new=envi("GLM5_MIN_NEW",0);
        int cap=1024,n_prompt=0,*prompt=glm5_amalloc((size_t)cap*sizeof(int)),v;
        if(prompt_file&&*prompt_file){
            FILE*pf=fopen(prompt_file,"r"); if(!pf) die("cannot open GLM5_PROMPT_IDS",-1);
            while(fscanf(pf,"%d",&v)==1){ if(n_prompt>=cap){int oc=cap;cap*=2;prompt=glm5_arealloc(prompt,(size_t)oc*sizeof(int),(size_t)cap*sizeof(int));} prompt[n_prompt++]=v; }
            fclose(pf);
        } else {
            const char *tok_path=getenv("GLM5_TOKENIZER");
            char default_tok[4096];
            if(!tok_path||!*tok_path){ const char *home=getenv("HOME"); snprintf(default_tok,sizeof default_tok,"%s/models/glm53f/tokenizer.json",home?home:""); tok_path=default_tok; }
            glm5_bpe bpe; char *rendered; size_t text_len=strlen(prompt_text), render_cap=text_len+512;
            glm5_chat_message msg={"user",prompt_text,NULL};
            if(glm5_bpe_load(tok_path,&bpe)!=0) die("cannot load GLM5_TOKENIZER",-1);
            rendered=malloc(render_cap); if(!rendered) die("prompt template allocation",-1);
            if(glm5_chat_template_render(&msg,1,getenv("GLM5_REASONING_EFFORT"),1,rendered,render_cap)<0){ free(rendered);glm5_bpe_free(&bpe);die("prompt template too long",-1); }
            int need=(int)(render_cap/2); if(need<1024)need=1024;
            prompt=glm5_arealloc(prompt,(size_t)cap*sizeof(int),(size_t)need*sizeof(int)); cap=need;
            n_prompt=glm5_bpe_encode(&bpe,rendered,prompt,cap);
            free(rendered); glm5_bpe_free(&bpe);
            if(n_prompt<1) die("empty tokenized prompt",-1);
            if(MyRank==0) logmsg("prompt-text: tokenizer=%s rendered=%zu bytes tokens=%d\n",tok_path,strlen(prompt_text),n_prompt);
        }
        if(n_prompt<1) die("empty prompt",-1);
        /* A prompt longer than the context makes max_new negative, which downstream
           becomes an undersized allocation and a corrupted heap far from the cause.
           Fail here with the two numbers the user needs (raise --ctx). */
        if(n_prompt>=cfg.max_pos){
            if(MyRank==0) logmsg("fatal: prompt=%d tok >= --ctx/max_pos=%d; raise --ctx to at least %d\n",
                                 n_prompt,cfg.max_pos,n_prompt+64);
            die("prompt exceeds context",-1);
        }
        if(n_prompt+max_new>cfg.max_pos) max_new=cfg.max_pos-n_prompt;
        if(min_new<0) min_new=0; if(min_new>max_new) min_new=max_new;
        if(MyRank==0) logmsg("gen: prompt=%d tok, max_new=%d, max_pos=%d\n",n_prompt,max_new,cfg.max_pos);
        int pf_last=-1; double t0=now_sec();
        double prof_gen0[GLM5_NPHASE], prof_gen_pf[GLM5_NPHASE], prof_gen_dec[GLM5_NPHASE];
        prof_snapshot(m,prof_gen0);
        g_ar_secs=0; g_ar_calls=0; g_ar_frags=0;
            int pchunk=envi("GLM5_PCHUNK",0);   /* Lever 1: chunked batched prefill (M=S) */
        /* GLM5_PREFILL_SP=1: query-sequence-parallel chunk (attention half sharded by home
         * query slice, N-way; needs GLM5_TP=0 replicated weights). Fallback: classic chunk. */
        int prefill_sp=envi("GLM5_PREFILL_SP",0);
        /* MTP prefill support: the draft block keeps its OWN latent-KV cache, so drafting
         * during decode attends garbage for every prompt position unless we back-fill it.
         * Capture the last-layer residual of every prompt position here (the chunk buffer
         * already holds exactly that on return), then after prefill run the drafter over
         * the prompt once to populate its KV (side computation; main KV untouched). */
        float*mtp_h=(envi("GLM5_MTP",0)||envi("GLM5_SPEC",0))?(float*)glm5_amalloc((size_t)n_prompt*C*4):NULL;
        if(pchunk>0){
            if(glm5_alloc_mstream_ex(m,pchunk,0)) die("alloc prefill chunk",-1);
            float*Xc=(float*)glm5_amalloc((size_t)pchunk*C*4);
            for(int p0=0;p0<n_prompt;p0+=pchunk){ int S=n_prompt-p0; if(S>pchunk)S=pchunk;
                for(int t=0;t<S;t++) embed_lookup(m,prompt[p0+t],Xc+(size_t)t*C);
                /* Tier A->B: re-shard the [0,p0) history before this chunk stores a
                 * position at or beyond T_cp.  Keep this in lockstep with the
                 * synthetic prefill path: CP callbacks are installed before the
                 * prompt loop, but remain dormant until this transition. */
                if(!m->cp_on && m->T_cp>0 && p0+S>m->T_cp){
                    double tt=now_sec(); glm5_prefill_to_cp(m,p0);
                    barrier();
                    if(MyRank==0) logmsg("prefill_tier: A->B re-shard at pos=%d (%.3f s) -> CP %s %d slots/rank\n",
                                         p0,now_sec()-tt,m->int4_kv?"int4":"bf16",m->cp_nslot);
                }
                int a=prefill_sp?glm5_forward_prefill_chunk_sp(m,Xc,S,p0,p0+S>=n_prompt)
                                :glm5_forward_prefill_chunk(m,Xc,S,p0,p0+S>=n_prompt);
                if(prefill_sp && a<-1) die("GLM5_PREFILL_SP needs GLM5_TP=0, CP/MSA off",a);
                if(mtp_h) memcpy(mtp_h+(size_t)p0*C,Xc,(size_t)S*C*4);
                if(a>=0) pf_last=a; }
            glm5_afree(Xc); glm5_free_mstream(m);
            if(MyRank==0) logmsg("prefill: chunked M=%d%s\n",pchunk,prefill_sp?" (query-SP)":"");
        } else {
        /* GLM5_TF_CHECK: teacher-forcing accuracy -- does argmax at pos p predict prompt[p+1]?
         * A correct LM scores ~40-80%; a broken forward ~0%. Compares int8 vs bf16 to localize. */
        int tf_check=envi("GLM5_TF_CHECK",0), tf_ok=0, tf_tot=0;
        for(int p=0;p<n_prompt;p++){
            /* The token-by-token fallback must perform the same tier transition as
             * chunked prefill before storing the first Tier-B position. */
            if(!m->cp_on && m->T_cp>0 && p>=m->T_cp){
                double tt=now_sec(); glm5_prefill_to_cp(m,p);
                barrier();
                if(MyRank==0) logmsg("prefill_tier: A->B re-shard at pos=%d (%.3f s) -> CP %s %d slots/rank\n",
                                     p,now_sec()-tt,m->int4_kv?"int4":"bf16",m->cp_nslot);
            }
            embed_lookup(m,prompt[p],x); pf_last=glm5_forward_token(m,x,p);
            if(mtp_h) memcpy(mtp_h+(size_t)p*C,x,(size_t)C*4);
            if(tf_check && p+1<n_prompt){ tf_tot++; if(pf_last==prompt[p+1]) tf_ok++;
                if(MyRank==0 && p<12) logmsg("TF p=%d pred=%d actual=%d %s\n",p,pf_last,prompt[p+1],pf_last==prompt[p+1]?"HIT":"."); } }
        if(tf_check && MyRank==0) logmsg("TF_ACCURACY %d/%d = %.1f%% (argmax(p)==prompt[p+1]; real LM ~40-80%%, broken ~0%%)\n",
                                         tf_ok,tf_tot, tf_tot?100.0*tf_ok/tf_tot:0.0);
        }
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
        glm5_spec_selftest(m,pf_last,n_prompt,C);
        int *gen=glm5_amalloc((size_t)(max_new>0?max_new:1)*sizeof(int)),ng=0,cur=pf_last,nan=0;
        /* MTP (GLM5_MTP / GLM5_SPEC): each decode step, draft the NEXT token from the just-produced
         * token + its residual hidden (glm5_mtp_draft), and measure acceptance alpha = P(draft == the
         * real next token). This is BYTE-IDENTICAL to plain decode (the draft is a side computation
         * that appends only to the MTP block's own KV, never the main KV). alpha is the gate: if
         * >=~0.7, gamma=1 spec decode yields E[tokens]=1+alpha per BATCHED-K=2 verify (the speedup
         * step; this loop is the correctness-first driver + alpha measurement). */
        int mtp_on=(m->mtp_layer!=NULL) && (envi("GLM5_MTP",0)||envi("GLM5_SPEC",0));
        float *xb=mtp_on?(float*)glm5_amalloc((size_t)2*C*4):NULL;
        int prev_draft=-1; long mtp_hit=0,mtp_tot=0;
        if(mtp_on && mtp_h){
            /* Back-fill the draft block's KV over the prompt: draft(h_t, token at t+1, pos t+1)
             * for every prompt position (slot 0 seeded from a zero hidden).  Lockstep on all
             * ranks (the draft forward carries the usual EP/TP collectives). */
            double tf0=now_sec();
            float*hz=(float*)glm5_amalloc((size_t)C*4); memset(hz,0,(size_t)C*4);
            glm5_mtp_draft(m,hz,prompt[0],0,xb);
            long dtf_ok=0,dtf_tot=0;
            for(int t=0;t<n_prompt;t++){
                int d=glm5_mtp_draft(m,mtp_h+(size_t)t*C,(t+1<n_prompt)?prompt[t+1]:cur,t+1,xb);
                if(t+2<n_prompt){ dtf_tot++; if(d==prompt[t+2]) dtf_ok++;
                    if(MyRank==0 && t<6 && envi("GLM5_MTP_DBG",0))
                        logmsg("MTP-TF t=%d draft=%d actual=%d %s\n",t,d,prompt[t+2],d==prompt[t+2]?"HIT":"."); }
            }
            glm5_afree(hz);
            if(MyRank==0) logmsg("MTP prefill: draft KV filled for %d positions in %.2fs; drafter TF %ld/%ld = %.1f%%\n",
                                 n_prompt+1,now_sec()-tf0,dtf_ok,dtf_tot,dtf_tot?100.0*dtf_ok/dtf_tot:0.0);
        }
        if(mtp_h){ glm5_afree(mtp_h); mtp_h=NULL; }
        if(mtp_on && MyRank==0) logmsg("MTP draft on: measuring acceptance alpha over %d decode steps\n",max_new);
        g_ar_secs=0; g_ar_calls=0; g_ar_frags=0;
        double td0=now_sec();
        int spec_verify=envi("GLM5_SPEC_VERIFY",0) && mtp_on && m->samp_temp<=0.0f;
        if(spec_verify){
            if(MyRank==0) logmsg("SPEC_VERIFY on: experimental K=2 batched verifier (greedy only)\n");
            int sv=glm5_run_spec_verify(m,gen,&ng,max_new,min_new,n_prompt,&cur,x,C,&nan,&mtp_hit,&mtp_tot);
            if(!sv){
                if(MyRank==0) logmsg("SPEC_VERIFY unavailable -> falling back to side-draft decode\n");
                spec_verify=0;
            } else if(sv==2){
                if(MyRank==0) logmsg("SPEC_VERIFY fell back at generated=%d\n",ng);
                spec_verify=0;
            }
        }
        double tl_emb=0,tl_fw=0,tl_nan=0,tl_mtp=0;   /* GLM5_TOK_TRACE: where the token loop's wall goes */
        if(!spec_verify){
            for(int g=ng;g<max_new;g++){ gen[ng++]=cur; if((cur==GLM5_EOS_ID0||cur==GLM5_EOS_ID1||cur==GLM5_EOS_ID2) && ng>=min_new) break;
                m->samp_hist=gen; m->samp_hist_n=ng;   /* repetition penalty over tokens so far */
                double tt0=now_sec();
                embed_lookup(m,cur,x);
                double tt1=now_sec(); tl_emb+=tt1-tt0;
                cur=glm5_forward_token(m,x,n_prompt+g);
                double tt2=now_sec(); tl_fw+=tt2-tt1;
                if(mtp_on){
                    if(prev_draft>=0){ mtp_tot++; if(prev_draft==cur) mtp_hit++;
                        if(MyRank==0 && mtp_tot<=10 && envi("GLM5_MTP_DBG",0))
                            logmsg("MTP dbg %ld: draft=%d actual=%d %s\n",
                                   mtp_tot,prev_draft,cur,prev_draft==cur?"HIT":".");
                    }
                    prev_draft=glm5_mtp_draft(m,x,cur,n_prompt+g+1,xb);  /* draft the token at n_prompt+g+2 */
                }
                double tt3=now_sec(); tl_mtp+=tt3-tt2;
                for(int i=0;i<C;i++) if(!(x[i]==x[i])) nan++;
                tl_nan+=now_sec()-tt3; }
        }
        double td=now_sec()-td0;
        if(MyRank==0 && envi("GLM5_TOK_TRACE",0) && ng>0)
            logmsg("[tok-trace] ms/tok: embed %.2f | forward_token %.2f | mtp %.2f | nan %.2f | loop-total %.2f\n"
                   "[tok-trace] layers: wall %.2f ms/tok | buckets-in-layer %.2f ms/tok | UNBUCKETED-IN-LAYER %.2f ms/tok (%.1f us/layer over %ld)\n",
                   tl_emb*1e3/ng, tl_fw*1e3/ng, tl_mtp*1e3/ng, tl_nan*1e3/ng, td*1e3/ng,
                   glm5_lay_wall*1e3/ng, glm5_lay_buck*1e3/ng, (glm5_lay_wall-glm5_lay_buck)*1e3/ng,
                   glm5_lay_n?(glm5_lay_wall-glm5_lay_buck)*1e6/glm5_lay_n:0.0, glm5_lay_n);
        double mtp_alpha=mtp_tot?(double)mtp_hit/mtp_tot:0.0;
        double gen_d_ar=g_ar_secs; long gen_d_calls=g_ar_calls, gen_d_frags=g_ar_frags;
        prof_snapshot(m,prof_gen_dec);
        barrier();
        if(MyRank==0){
            logmsg("gen: prefill %.2f tok/s comm %.1f%% calls=%ld frags=%ld, decode %d tok %.2f tok/s comm %.1f%% calls=%ld frags=%ld, NaNs=%d\n",
                   n_prompt/tpf,100.0*gen_pf_ar/tpf,gen_pf_calls,gen_pf_frags,ng,td>0?ng/td:0.0,td>0?100.0*gen_d_ar/td:0.0,gen_d_calls,gen_d_frags,nan);
            prof_log_delta("gen_prefill",prof_gen0,prof_gen_pf,n_prompt,tpf,gen_pf_ar);
            prof_log_delta("gen_decode",prof_gen_pf,prof_gen_dec,ng,td,gen_d_ar);
            if(mtp_on) logmsg("MTP_ALPHA %ld/%ld = %.1f%% (draft==next real token) -> spec E[tok/verify]=%.2f (gate >=0.7)\n",
                              mtp_hit,mtp_tot,100.0*mtp_alpha,1.0+mtp_alpha);
            char buf[6000]; int o=0; for(int i=0;i<ng&&o<5900;i++) o+=snprintf(buf+o,sizeof(buf)-o,"%d ",gen[i]);
            logmsg("GEN_IDS %s\n",buf);
            if(gen_out&&*gen_out){ FILE*gf=fopen(gen_out,"w"); if(gf){ for(int i=0;i<ng;i++) fprintf(gf,"%d%s",gen[i],i+1<ng?" ":"\n"); fclose(gf); logmsg("gen: wrote %d ids to %s\n",ng,gen_out);} }
            logmsg("SENTINEL glm5_gen_%dn=done\n",N);
        }
        glm5_afree(gen); glm5_afree(xb); glm5_afree(prompt); glm5_afree(x); glm5_free(m); return 0;
    }

    /* ---- synthetic-token prefill benchmark: uses embeddings but avoids a huge prompt file. ---- */
    if(prefill_synth>0){
        /* Same trap as the --prompt-ids path: writing past max_pos is an OOB KV write. */
        if(start_pos+prefill>cfg.max_pos){
            if(MyRank==0) logmsg("fatal: start_pos=%d + prefill=%d > --ctx/max_pos=%d; raise --ctx\n",
                                 start_pos,prefill,cfg.max_pos);
            die("synth prefill exceeds context",-1);
        }
        int pchunk=envi("GLM5_PCHUNK",0);
        double prof0s[GLM5_NPHASE], prof_pfs[GLM5_NPHASE];
        prof_snapshot(m,prof0s);
        double t0=now_sec(); g_ar_secs=0; g_ar_calls=0; g_ar_frags=0; int pf_last=-1, nan=0;
        if(pchunk>0){
            if(glm5_alloc_mstream_ex(m,pchunk,0)) die("alloc prefill chunk",-1);
            float*Xc=(float*)glm5_amalloc((size_t)pchunk*C*4);
            int orig_gsize=GSize;                       /* group size before any Phase-2 merges */
            int mat[16],matn=0,mi=0; { const char*ms=getenv("GLM5_MERGE_AT");  /* test: forced merge positions */
                if(ms&&*ms){ char b[256]; snprintf(b,sizeof b,"%s",ms); for(char*t=strtok(b,":,");t&&matn<16;t=strtok(NULL,":,")) mat[matn++]=atoi(t); } }  /* ':' too: pjsub -x splits on ',' */
            /* Optional real token stream (GLM5_PROMPT_TOKENS = binary uint32 file, e.g. tokenized
             * repo source). Cycled if shorter than prefill; falls back to the deterministic hash. */
            uint32_t*ptok=NULL; long ptn=0; { const char*ptf=getenv("GLM5_PROMPT_TOKENS");
                if(ptf&&*ptf){ FILE*pf=fopen(ptf,"rb"); if(pf){ fseek(pf,0,SEEK_END); ptn=ftell(pf)/4; fseek(pf,0,SEEK_SET);
                    if(ptn>0){ ptok=(uint32_t*)glm5_amalloc((size_t)ptn*4); if(fread(ptok,4,ptn,pf)!=(size_t)ptn){ glm5_afree(ptok); ptok=NULL; ptn=0; } } fclose(pf);
                    if(GRank==0) logmsg("prompt tokens: %s (%ld tok, cycled)\n",ptf,ptn); } } }
            /* KV cache load (prompt caching): restore a saved system-prompt KV and resume from P.
             * Tokens are indexed by ABSOLUTE position (pa) so save/load/full-recompute agree. */
            { const char*kvl=getenv("GLM5_KV_LOAD"); if(kvl&&*kvl){ int P=glm5_kv_load(m,kvl); if(P>0) start_pos=P; } }
            int pend=start_pos+prefill;
            for(int p0=0;p0<prefill;p0+=pchunk){
                int S=prefill-p0; if(S>pchunk)S=pchunk; int pa=start_pos+p0;   /* pa = absolute position */
                /* Phase 2: pairwise group merge at a (forced) merge point while a bigger group exists.
                 * Survivor = even/lower subgroup; its KV is propagated to the sibling, concurrency halves. */
                while(!m->cp_on && GSize<N && mi<matn && pa+S>mat[mi]){
                    int og=GSize; double tt=now_sec();
                    glm5_group_merge(m,&comm,cfg.hidden*ar_tokens,pa,blob_dir,orig_gsize);
                    if(GRank==0) logmsg("group_merge: %dx%d -> %dx%d at pos=%d (%.3f s) seq=%d\n",
                                        N/og,og,N/GSize,GSize,pa,now_sec()-tt,GBase/orig_gsize);
                    mi++;
                }
                for(int t=0;t<S;t++){
                    long seq=(pa+t)+(long)(GBase/orig_gsize)*0x9E3779B1u;     /* absolute position + group offset */
                    int tok = ptok ? (int)(ptok[((seq%ptn)+ptn)%ptn] % (unsigned)m->cfg.vocab)
                                   : (int)((unsigned)seq*1315423911u % (unsigned)m->cfg.vocab);
                    embed_lookup(m,tok,Xc+(size_t)t*C);
                }
                /* Tier A->B: re-shard the [0,pa) history BEFORE any chunk that would store a
                 * position >= T_cp (so the Tier-A buffer never overflows). Lockstep on all ranks. */
                if(!m->cp_on && m->T_cp>0 && pa+S>m->T_cp){
                    double tt=now_sec(); glm5_prefill_to_cp(m,pa);
                    gbarrier();
                    if(GRank==0) logmsg("prefill_tier: A->B re-shard at pos=%d (%.3f s) -> CP %s %d slots/rank\n",
                                         pa,now_sec()-tt,m->int4_kv?"int4":"bf16",m->cp_nslot);
                }
                int a=envi("GLM5_PREFILL_SP",0)?glm5_forward_prefill_chunk_sp(m,Xc,S,pa,pa+S>=pend)
                                              :glm5_forward_prefill_chunk(m,Xc,S,pa,pa+S>=pend);
                if(a<-1) die("GLM5_PREFILL_SP needs GLM5_TP=0, CP/MSA off",a);
                if(a>=0) pf_last=a;
                { long chunk_nan=0; int first_nan_t=-1, nan_toks=0;   /* per-chunk NaN localization */
                  for(int t=0;t<S;t++){ int tn=0; for(int i=0;i<C;i++) if(!(Xc[(size_t)t*C+i]==Xc[(size_t)t*C+i])) tn++;
                      if(tn){ if(first_nan_t<0) first_nan_t=t; chunk_nan+=tn; nan_toks++; } }
                  nan+=chunk_nan;
                  if(chunk_nan && GRank==0)
                      logmsg("[nan-chunk] pos=[%d,%d) first_nan_pos=%d nan_toks=%d/%d elems=%ld\n",
                             pa,pa+S,pa+first_nan_t,nan_toks,S,chunk_nan); }
                if(GRank==0 && envi("GLM5_PREFILL_ROLLING",1)){
                    double dt=now_sec()-t0;
                    logmsg("prefill_progress: %d/%d tok elapsed=%.3f rate=%.2f tok/s RSS=%.2f GB\n",
                           pa+S,pend,dt,dt>0?(p0+S)/dt:0.0,rss_bytes()/1e9);
                }
            }
            /* KV cache save (prompt caching): persist the processed [0,pend) Tier-A KV for reuse. */
            { const char*kvs=getenv("GLM5_KV_SAVE"); if(kvs&&*kvs && MyRank==0) glm5_kv_save(m,kvs,pend); }
            glm5_afree(Xc); glm5_free_mstream(m);
            /* Generation tail: greedy-decode GLM5_GEN_NEW tokens after the prefill (reuses the loaded
             * KV + the just-prefilled query). All ranks decode (forward_token is collective); group
             * rank 0 logs/writes GEN_IDS for detokenization. */
            int gen_new=envi("GLM5_GEN_NEW",0);
            if(gen_new>0 && pf_last>=0){
                int*gen=glm5_amalloc((size_t)gen_new*4); int ng=0, cur=pf_last, pos=pend;
                double tg=now_sec();
                for(int g=0; g<gen_new && pos<m->cfg.max_pos; g++){
                    gen[ng++]=cur;
                    if(cur==GLM5_EOS_ID0||cur==GLM5_EOS_ID1||cur==GLM5_EOS_ID2) break;
                    m->samp_hist=gen; m->samp_hist_n=ng;   /* repetition penalty over tokens so far */
                    embed_lookup(m,cur,x); cur=glm5_forward_token(m,x,pos++);
                }
                if(GRank==0){
                    logmsg("gen: %d tokens (%.2f tok/s) from pos=%d\n",ng,ng/(now_sec()-tg+1e-9),pend);
                    char buf[16000]; int o=0; for(int i=0;i<ng&&o<15900;i++) o+=snprintf(buf+o,sizeof buf-o,"%d ",gen[i]);
                    logmsg("GEN_IDS %s\n",buf);
                    const char*go=getenv("GLM5_GEN_OUT"); if(go&&*go){ FILE*gf=fopen(go,"w"); if(gf){ for(int i=0;i<ng;i++) fprintf(gf,"%d ",gen[i]); fclose(gf); logmsg("gen: wrote %d ids -> %s\n",ng,go);} }
                }
                glm5_afree(gen);
            }
        } else {
            for(int p=0;p<prefill;p++){
                int tok=(unsigned)(p+(unsigned)GId*0x9E3779B1u)*1315423911u % (unsigned)m->cfg.vocab;
                embed_lookup(m,tok,x); pf_last=glm5_forward_token(m,x,p);
                for(int i=0;i<C;i++) if(!(x[i]==x[i])) nan++;
            }
        }
        double tpf=now_sec()-t0, ar=g_ar_secs; long calls=g_ar_calls, frags=g_ar_frags;
        prof_snapshot(m,prof_pfs); gbarrier();
        if(GRank==0){
            logmsg("prefill_synth: gid=%d/%d gsize=%d %d tok %.2f tok/s comm %.1f%% calls=%ld frags=%ld pchunk=%d argmax=%d NaNs=%d RSS=%.2f GB\n",
                   GId,N/GSize,GSize,prefill,tpf>0?prefill/tpf:0.0,tpf>0?100.0*ar/tpf:0.0,calls,frags,pchunk,pf_last,nan,rss_bytes()/1e9);
            prof_log_delta("prefill_synth",prof0s,prof_pfs,prefill,tpf,ar);
            logmsg("SENTINEL glm5_prefill_g%dn=%s\n",GSize,nan==0?"done":"NAN");
        }
        barrier();   /* global: all groups done before teardown (clean VCQ dereg / aggregate) */
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
