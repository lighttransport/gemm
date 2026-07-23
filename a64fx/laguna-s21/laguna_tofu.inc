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
    if(tp_comm_init(&g_comm,Vcq,PeerVcq,MyRank,N,LAGUNA_HIDDEN,barrier)!=0) die("tp_comm_init",-1);
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
    if(stage_load(&st,stage_dir,MyRank)!=0){ fprintf(stderr,"rank %d: stage load failed\n",MyRank); return 1; }
    laguna_model m; model_build(&m,&st,n_layers,maxpos,MyRank,N);
    laguna_scratch sc; scratch_alloc(&sc,maxpos);
    double t1=now_sec();
    logmsg("rank %d: loaded %ld tensors, %.2f GB blob, %d layers, maxpos %d in %.1fs\n",
           MyRank,st.n_ents,st.blob_bytes/1e9,n_layers,maxpos,t1-t0);

    /* eos tokens (config: [2,24]) */
    const int eos0=2, eos1=24;
    int *gen=malloc((size_t)max_new*sizeof(int)); int ngen=0;
    float *x=malloc(LAGUNA_HIDDEN*sizeof(float));
    int total_nan=0;

    /* prefill: process prompt tokens; only compute logits at the last one */
    double pf0=now_sec();
    for(int p=0;p<n_prompt;p++){
        embed_token(&m,prompt[p],x);
        int need_logits=(p==n_prompt-1)||g_dbg;
        forward_token(&m,&sc,x,p,ep_ar,&g_comm,need_logits);
        total_nan+=count_nan(x,LAGUNA_HIDDEN);
        if(g_dbg&&MyRank==0){ int a=argmax(sc.logits,LAGUNA_VOCAB); logmsg("PREFILL pos%d in_tok=%d -> argmax=%d (logit %.2f)\n",p,prompt[p],a,sc.logits[a]); }
    }
    double pf1=now_sec();
    int next=argmax(sc.logits,LAGUNA_VOCAB);
    if(MyRank==0){
        /* top-3 for sanity */
        /* top-5 logits for sanity */
        for(int t=0;t<5;t++){ int a=argmax(sc.logits,LAGUNA_VOCAB); logmsg("  top%d: id=%d logit=%.3f\n",t,a,sc.logits[a]); sc.logits[a]=-1e30f; }
        int a=next;
        logmsg("prefill %d tok in %.2fs (%.1f tok/s); argmax first token = %d nan=%d\n",
               n_prompt,pf1-pf0,n_prompt/(pf1-pf0),a,total_nan);
    }
    /* decode loop */
    double dc0=now_sec();
    int pos=n_prompt;
    for(int g=0; g<max_new; ++g){
        gen[ngen++]=next;
        if(next==eos0||next==eos1) break;
        embed_token(&m,next,x);
        forward_token(&m,&sc,x,pos,ep_ar,&g_comm,1);
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
    barrier();
    return total_nan>0 ? 3 : 0;
}
