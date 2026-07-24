/* laguna_tofu.inc - minimal single-EP-group uTofu bootstrap + generate driver.
 * Included at the bottom of laguna_s21_ep_runner.c (one translation unit).
 * Distilled from a64fx/glm5/glm5_ep_runner.c (rpn=1, one group, no DP merge). */
#include <utofu.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define LAGUNA_MAX_NODES 64
#define RUN_STAG  DEMO_STAG
#define WAIT_TIMEOUT_SEC 900.0

static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t   Base;
static utofu_vcq_id_t  PeerVcq[LAGUNA_MAX_NODES];
static utofu_stadd_t   PeerBase[LAGUNA_MAX_NODES];
static int             N, MyRank;
static char           *Region;
static size_t          SEND_OFF, BAR_BASE, SlotB;
static uint64_t        Bt = 1;

static void die(const char *what, int rc) { logmsg("FATAL: %s (rc=%d)\n", what, rc); exit(1); }
static inline size_t bar_recv_off(int s){ return BAR_BASE+(size_t)s*SlotB; }
static inline size_t bar_go_off(void)   { return BAR_BASE+(size_t)N*SlotB; }

static int read_topo(uint8_t coords[][TOFU_NCOORDS]){
    const char*t=getenv("TOFU_TOPO_PATH"); const char*tp=(t&&*t)?t:TOPO_PATH;
    FILE*f=fopen(tp,"r");
    if(!f){ fprintf(stderr,"cannot open %s (run tofu_topo_helper first)\n",tp); exit(1); }
    int n=0; char line[256];
    while(fgets(line,sizeof line,f)){
        if(line[0]=='#') continue;
        int r,cc[TOFU_NCOORDS];
        if(sscanf(line,"%d %d %d %d %d %d %d",&r,&cc[0],&cc[1],&cc[2],&cc[3],&cc[4],&cc[5])!=7) continue;
        for(int k=0;k<TOFU_NCOORDS;k++) coords[n][k]=(uint8_t)cc[k];
        n++;
    }
    fclose(f); return n;
}
static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len, int drain){
    int rc; void*cb;
    for(;;){ rc=utofu_put(Vcq,pv,s,d,len,0,FLAGS,NULL); if(rc!=UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq,0,&cb); }
    if(rc!=UTOFU_SUCCESS) die("utofu_put",rc);
    if(drain){ do{ rc=utofu_poll_tcq(Vcq,0,&cb);}while(rc==UTOFU_ERR_NOT_FOUND); if(rc!=UTOFU_SUCCESS) die("utofu_poll_tcq",rc); }
}
static void wait_ge(volatile uint64_t*q, uint64_t v, const char*what){
    double ts=now_sec(); while(*q<v) if(now_sec()-ts>WAIT_TIMEOUT_SEC) die(what,-1);
}
static void barrier(void){
    uint64_t t=++Bt; char*sb=Region+SEND_OFF;
    if(MyRank==0){
        for(int s=1;s<N;s++) wait_ge((volatile uint64_t*)(Region+bar_recv_off(s)),t,"barrier fan-in");
        for(int s=1;s<N;s++){ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[s],Base+SEND_OFF,PeerBase[s]+bar_go_off(),8,1); }
    } else {
        volatile uint64_t*go=(volatile uint64_t*)(Region+bar_go_off()); double ts=now_sec();
        do{ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[0],Base+SEND_OFF,PeerBase[0]+bar_recv_off(MyRank),8,1);
            for(int a=0;a<50&&*go<t;a++) usleep(2000);
            if(now_sec()-ts>WAIT_TIMEOUT_SEC) die("barrier timeout",-1);
        }while(*go<t);
    }
}

static tp_comm g_comm;
static double g_ar_secs=0; static long g_ar_calls=0;
static void ep_ar(void *ctx, float *buf, int count){
    tp_comm *c=(tp_comm*)ctx; int mc=c->max_count>0?c->max_count:count;
    double t=now_sec();
    for(int off=0;off<count;){ int nn=count-off; if(nn>mc)nn=mc; tp_allreduce_sum(c,buf+off,nn); off+=nn; }
    g_ar_secs+=now_sec()-t; g_ar_calls++;
}

/* Synchronous allreduce via the async interface: launch() does the blocking
 * reduce, join() is a no-op.  (A dedicated comm-thread overlap was tried and
 * reverted: on this OMP runtime the comm thread is starved of CPU and can't
 * drive uTofu promptly -- the allreduce ballooned ~100x, net loss.) */
static void ar_launch(void *ctx, float *buf, int count){ (void)ctx; ep_ar(&g_comm, buf, count); }
static void ar_join(void *ctx){ (void)ctx; }
static laguna_async_ar g_aar = { ar_launch, ar_join, NULL };
static void ar_thread_start(void){ if(N<=1){ g_aar.launch=NULL; g_aar.join=NULL; } }
static void ar_thread_stop(void){ }

static void tofu_bootstrap(void){
    int rc;
    utofu_tni_id_t*tni_ids=NULL; size_t num_tnis=0;
    rc=utofu_get_onesided_tnis(&tni_ids,&num_tnis); if(rc!=UTOFU_SUCCESS) die("utofu_get_onesided_tnis",rc);
    if(num_tnis<1) die("no onesided TNIs",-1);
    uint8_t my_coords[TOFU_NCOORDS]={0};
    rc=utofu_query_my_coords(my_coords); if(rc!=UTOFU_SUCCESS) die("utofu_query_my_coords",rc);
    static uint8_t topo[LAGUNA_MAX_NODES][TOFU_NCOORDS];
    N=read_topo(topo);
    if(N<1||N>LAGUNA_MAX_NODES) die("bad node count from topo",N);
    MyRank=-1;
    for(int r=0;r<N;r++) if(memcmp(topo[r],my_coords,TOFU_NCOORDS)==0) MyRank=r;
    if(MyRank==-1){ fprintf(stderr,"my coords not in topo\n"); exit(1); }
    g_rank=MyRank;
    { char en[64]; snprintf(en,sizeof en,"laguna_stderr_rank%02d.txt",MyRank); if(!freopen(en,"w",stderr)){} setvbuf(stderr,NULL,_IOLBF,0); }
    if(MyRank==0) g_log=fopen("laguna_rank00.txt","w");

    SlotB=DEMO_CACHE_LINE; SEND_OFF=0; BAR_BASE=SlotB;
    size_t region_sz=BAR_BASE+(size_t)(N+1)*SlotB;
    if(posix_memalign((void**)&Region,DEMO_CACHE_LINE,region_sz)!=0) die("posix_memalign",-1);
    memset(Region,0,region_sz);
    utofu_tni_id_t tni=tni_ids[0];
    rc=utofu_create_vcq_with_cmp_id(tni,DEMO_CMP_ID,0,&Vcq); if(rc!=UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id",rc);
    utofu_vcq_id_t my_real; rc=utofu_query_vcq_id(Vcq,&my_real); if(rc!=UTOFU_SUCCESS) die("utofu_query_vcq_id",rc);
    rc=utofu_reg_mem_with_stag(Vcq,Region,region_sz,RUN_STAG,0,&Base); if(rc!=UTOFU_SUCCESS) die("utofu_reg_mem_with_stag",rc);
    for(int r=0;r<N;r++){
        if(r==MyRank){ PeerVcq[r]=my_real; PeerBase[r]=Base; continue; }
        rc=utofu_construct_vcq_id(topo[r],tni_ids[0],DEMO_CQ_ID,DEMO_CMP_ID,&PeerVcq[r]); if(rc!=UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)",rc);
        utofu_set_vcq_id_path(&PeerVcq[r],NULL);
        rc=utofu_query_stadd(PeerVcq[r],RUN_STAG,&PeerBase[r]); if(rc!=UTOFU_SUCCESS) die("utofu_query_stadd(peer)",rc);
    }
    free(tni_ids);
    barrier();
    /* max_count sized for the batched chunked-prefill allreduce (C*hidden) so the
     * whole chunk reduces in one sync; decode's per-token [hidden] reduce still fits. */
    if(tp_comm_init(&g_comm,Vcq,PeerVcq,MyRank,N,LAGUNA_PCHUNK*LAGUNA_HIDDEN,barrier)!=0) die("tp_comm_init",-1);
}

/* Isolate pure collective cost from load-imbalance wait: barrier before each
 * timed allreduce so every rank enters together (=> wire+sync only), vs a
 * back-to-back loop (=> what the decode loop actually pays). No weights needed. */
static int run_ar_probe(int argc, char **argv){
    int reps = 2000, count = LAGUNA_HIDDEN;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc) reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--count")&&i+1<argc) count=atoi(argv[++i]);
    }
    if(count>LAGUNA_HIDDEN) count=LAGUNA_HIDDEN;
    tofu_bootstrap();
    float *buf=malloc((size_t)count*sizeof(float));
    for(int i=0;i<count;i++) buf[i]=0.001f*i;
    /* warmup */
    for(int i=0;i<50;i++) ep_ar(&g_comm,buf,count);
    barrier();
    /* (a) back-to-back: includes any drift/imbalance the decode loop sees */
    double t0=now_sec();
    for(int i=0;i<reps;i++) ep_ar(&g_comm,buf,count);
    double bb=(now_sec()-t0)/reps;
    barrier();
    /* (b) barrier-synced: all ranks enter together => pure collective cost */
    double acc=0;
    for(int i=0;i<reps;i++){ barrier(); double s=now_sec(); ep_ar(&g_comm,buf,count); acc+=now_sec()-s; }
    double sy=acc/reps;
    if(MyRank==0)
        logmsg("AR_PROBE count=%d reps=%d: back_to_back=%.1f us  barrier_synced=%.1f us  "
               "=> per-layer decode comm would be %.1f us; 47 layers = %.2f ms/token\n",
               count,reps,bb*1e6,sy*1e6,bb*1e6,47*bb*1e3);
    barrier();
    return 0;
}

static int run_generate(int argc, char **argv){
    const char *ids_path=NULL, *stage_dir=getenv("LAGUNA_STAGE_DIR"), *gen_out=NULL;
    int max_new=32, maxpos=0, n_layers=LAGUNA_LAYERS;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--ids")&&i+1<argc) ids_path=argv[++i];
        else if(!strcmp(argv[i],"--max-new")&&i+1<argc) max_new=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--maxpos")&&i+1<argc) maxpos=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--layers")&&i+1<argc) n_layers=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--stage-dir")&&i+1<argc) stage_dir=argv[++i];
        else if(!strcmp(argv[i],"--gen-out")&&i+1<argc) gen_out=argv[++i];
        else { fprintf(stderr,"unknown flag %s\n",argv[i]); return 2; }
    }
    if(!ids_path){ fprintf(stderr,"--generate requires --ids FILE\n"); return 2; }
    if(!stage_dir) stage_dir="/local/laguna-s21";
    if(n_layers<1||n_layers>LAGUNA_LAYERS) n_layers=LAGUNA_LAYERS;

    g_dbg=envi("LAGUNA_DEBUG",0);
    tofu_bootstrap();

    int *prompt=NULL; int n_prompt=load_ids(ids_path,&prompt);
    if(n_prompt<1){ fprintf(stderr,"no prompt ids in %s\n",ids_path); return 2; }
    if(maxpos<=0) maxpos=n_prompt+max_new+8;
    if(n_prompt+max_new>maxpos){ fprintf(stderr,"prompt+max_new exceeds maxpos\n"); return 2; }

    double t0=now_sec();
    laguna_stage st;
    /* The fp8 build re-quantizes every hot tensor at load (experts -> int8-per-block,
     * linears -> int8), so it needs no blob-sized arena -- except in FP8_EXACT mode,
     * where the e4m3 expert bytes are read hot and want NUMA-local pages. */
    int use_arena = 1;
#if defined(LAGUNA_FP8)
    { const char *e=getenv("LAGUNA_FP8_EXACT"); use_arena = (e && atoi(e)) ? 1 : 0; }
#endif
    if(stage_load(&st,stage_dir,MyRank,use_arena)!=0){ fprintf(stderr,"rank %d: stage load failed\n",MyRank); return 1; }
    laguna_model m; model_build(&m,&st,n_layers,maxpos,MyRank,N);
#if defined(LAGUNA_FP8)
    fp8_release_blob(&m,&st);   /* weights are re-quantized; drop the mapping */
#endif
    laguna_scratch sc; scratch_alloc(&sc,maxpos);
    ar_thread_start();   /* dedicated comm core for overlapped allreduce */
    double t1=now_sec();
    logmsg("rank %d: loaded %ld tensors, %.2f GB blob, %d layers, maxpos %d in %.1fs\n",
           MyRank,st.n_ents,st.blob_bytes/1e9,n_layers,maxpos,t1-t0);
    /* All ranks must enter the forward together: load time varies per rank (a lot
     * for the ~27GB bf16 build under memory pressure), and without this barrier a
     * fast rank hits the first MoE-layer allreduce and times out (60s) waiting on
     * a still-loading peer. */
    barrier();

    /* eos tokens (config: [2,24]) */
    const int eos0=2, eos1=24;
    int *gen=malloc((size_t)max_new*sizeof(int)); int ngen=0;
    float *x=malloc(LAGUNA_HIDDEN*sizeof(float));
    int total_nan=0;

    /* prefill: chunked over the prompt except the last token, which is a per-token
     * forward to produce the first-generation logits.  --pchunk 1 forces the old
     * per-token path for A/B. */
    double pf0=now_sec();
    int pchunk=envi("LAGUNA_PCHUNK", LAGUNA_PCHUNK);
    if(pchunk<1) pchunk=1; if(pchunk>LAGUNA_PCHUNK) pchunk=LAGUNA_PCHUNK;
    int nprefill = n_prompt-1;   /* last token done per-token for logits */
    float *Xc = malloc((size_t)LAGUNA_PCHUNK*LAGUNA_HIDDEN*sizeof(float));
    int p=0;
    if(pchunk>1){
        for(; p+1<=nprefill; ){
            int C = nprefill-p; if(C>pchunk)C=pchunk; if(C<=0) break;
            for(int c=0;c<C;++c) embed_token(&m,prompt[p+c],Xc+(size_t)c*LAGUNA_HIDDEN);
            forward_prefill_chunk(&m,&sc,Xc,p,C,&g_aar);
            p+=C;
            if(MyRank==0 && (p%2048<C)) logmsg("prefill progress: %d/%d\n",p,n_prompt);
        }
    }
    for(; p<n_prompt; ++p){   /* remaining tokens (incl. the last) per-token */
        embed_token(&m,prompt[p],x);
        int need_logits=(p==n_prompt-1)||g_dbg;
        forward_token(&m,&sc,x,p,&g_aar,need_logits);
        total_nan+=count_nan(x,LAGUNA_HIDDEN);
    }
    free(Xc);
    double pf1=now_sec();
    int next=argmax(sc.logits,LAGUNA_VOCAB);
    if(MyRank==0){
        /* top-3 for sanity */
        /* top-5 logits for sanity */
        for(int t=0;t<5;t++){ int a=argmax(sc.logits,LAGUNA_VOCAB); logmsg("  top%d: id=%d logit=%.3f\n",t,a,sc.logits[a]); sc.logits[a]=-1e30f; }
        int a=next;
        logmsg("prefill %d tok in %.2fs (%.1f tok/s); argmax first token = %d nan=%d\n",
               n_prompt,pf1-pf0,n_prompt/(pf1-pf0),a,total_nan);
        extern double g_c_qkv,g_c_attn,g_c_op,g_c_router,g_c_expert,g_c_shared;
        logmsg("  chunk phases (s): qkv=%.1f attn=%.1f o_proj+rms=%.1f router=%.1f (expert=%.1f) shared/mlp=%.1f\n",
               g_c_qkv,g_c_attn,g_c_op,g_c_router-g_c_expert,g_c_expert,g_c_shared);
    }
    /* decode loop */
    g_ar_secs=0; g_ar_calls=0;   /* measure decode-only comm */
    double dc0=now_sec();
    int pos=n_prompt;
    for(int g=0; g<max_new; ++g){
        gen[ngen++]=next;
        if(next==eos0||next==eos1) break;
        embed_token(&m,next,x);
        forward_token(&m,&sc,x,pos,&g_aar,1);
        total_nan+=count_nan(x,LAGUNA_HIDDEN);
        next=argmax(sc.logits,LAGUNA_VOCAB);
        pos++;
    }
    double dc1=now_sec();
    if(MyRank==0){
        int nth=1;
#ifdef _OPENMP
        nth=omp_get_max_threads();
#endif
        logmsg("decode %d tok in %.2fs (%.1f tok/s) total_nan=%d comm=%.2fs (%.1f%%) threads=%d\n",
               ngen,dc1-dc0,ngen/(dc1-dc0>0?dc1-dc0:1),total_nan,g_ar_secs,
               (dc1-dc0)>0?100.0*g_ar_secs/(dc1-dc0):0.0,nth);
        printf("GEN_IDS");
        for(int i=0;i<ngen;i++) printf(" %d",gen[i]);
        printf("\n"); fflush(stdout);
        if(gen_out){ FILE*f=fopen(gen_out,"w"); if(f){ for(int i=0;i<ngen;i++) fprintf(f,"%d%s",gen[i],i+1<ngen?" ":"\n"); fclose(f);} }
    }
    ar_thread_stop();
    barrier();
    return total_nan>0 ? 3 : 0;
}
