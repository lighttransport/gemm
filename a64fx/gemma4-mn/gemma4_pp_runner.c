/* gemma4_pp_runner.c - pipeline-parallel (PP) multinode runner for Gemma-4 12B BF16.
 *
 * One rank/node = one pipeline stage owning layers [r*L/N, (r+1)*L/N). MPI only places
 * ranks (no MPI_Init); all comm is uTofu one-sided Put (modeled on ds4f_ep_runner.c).
 * Per token (LOCKSTEP, single stream): rank 0 embeds -> forward_partial -> Put the 15KB
 * hidden to rank 1; rank k recv -> set_hidden -> forward_partial -> Put to k+1; the last
 * rank recv -> forward_partial(.., n_layers) [final norm] -> compute_logits -> argmax ->
 * Put the next token back to rank 0. rank 0 waits that token before sending the next
 * hidden, so each hidden slot is consumed before overwrite (no ring needed).
 *
 * Run: (after tofu_topo_helper writes tofu_topo.txt) mpiexec -np N -vcoordfile vc \
 *        ./gemma4_pp_runner <gguf> <blob_dir> [prompt_ids_file] [maxgen]
 * Build: see run/Makefile -- fcc ... -lm -lpthread -lhwb -ltofucom
 */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include "gemma4_pp_load.h"
#include "../utofu-tests/tofu_demo.h"   /* DEMO_CMP_ID, DEMO_CQ_ID, DEMO_STAG, TOFU_NCOORDS, DEMO_CACHE_LINE, TOPO_PATH */

#define MAX_NODES 128
#define RUN_STAG  DEMO_STAG
#define WAIT_TIMEOUT_SEC 300.0
static void die(const char *what, int rc){ fprintf(stderr,"FATAL: %s (rc=%d)\n",what,rc); exit(1); }
static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

/* ---- topology ---- */
static int read_topo(uint8_t coords[][TOFU_NCOORDS]){
    const char *tp = getenv("TOFU_TOPO_PATH"); if(!tp||!*tp) tp = TOPO_PATH;
    FILE *f = fopen(tp,"r"); if(!f){ fprintf(stderr,"cannot open %s (run tofu_topo_helper first)\n",tp); exit(1); }
    int n=0; char line[256];
    while(fgets(line,sizeof line,f)){
        if(line[0]=='#'||line[0]=='\n') continue;
        unsigned r,c[TOFU_NCOORDS];
        if(sscanf(line,"%u %u %u %u %u %u %u",&r,&c[0],&c[1],&c[2],&c[3],&c[4],&c[5])!=7){ fprintf(stderr,"malformed: %s",line); exit(1); }
        if((int)r!=n){ fprintf(stderr,"ranks out of order\n"); exit(1); }
        for(int k=0;k<TOFU_NCOORDS;k++) coords[n][k]=(uint8_t)c[k];
        n++;
    }
    fclose(f); if(n<1){ fprintf(stderr,"empty topo\n"); exit(1); } return n;
}

/* ---- uTofu state ---- */
static int N, MyRank;
static char *Region;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t Base;
static utofu_vcq_id_t PeerVcq[MAX_NODES];
static utofu_stadd_t PeerBase[MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
/* region offsets (all cache-line aligned) */
static size_t SEND_OFF, BAR_BASE, HRECV_OFF, HSEQ_OFF, TTOK_OFF, TSEQ_OFF, HSEND_OFF, SCRATCH_OFF, REGION_SZ;
static int n_embd_g;
static uint64_t Bt = 1;
static inline size_t alignup(size_t x){ return (x + DEMO_CACHE_LINE-1) & ~(size_t)(DEMO_CACHE_LINE-1); }
static inline size_t bar_recv_off(int s){ return BAR_BASE + (size_t)s*DEMO_CACHE_LINE; }
static inline size_t bar_go_off(void){ return BAR_BASE + (size_t)N*DEMO_CACHE_LINE; }

static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len, int drain){
    int rc; void *cb;
    for(;;){ rc=utofu_put(Vcq,pv,s,d,len,0,FLAGS,NULL); if(rc!=UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq,0,&cb); }
    if(rc!=UTOFU_SUCCESS) die("utofu_put",rc);
    if(drain){ do{ rc=utofu_poll_tcq(Vcq,0,&cb); }while(rc==UTOFU_ERR_NOT_FOUND); if(rc!=UTOFU_SUCCESS) die("poll_tcq",rc); }
}
static void wait_ge(volatile uint64_t *q, uint64_t v, const char *what){
    double ts=now_sec(); while(*q<v) if(now_sec()-ts>WAIT_TIMEOUT_SEC) die(what,-1);
}
static void barrier_robust(int robust){
    uint64_t t=++Bt; char *sb=Region+SEND_OFF;
    if(MyRank==0){
        for(int s=1;s<N;s++) wait_ge((volatile uint64_t*)(Region+bar_recv_off(s)),t,"barrier fan-in");
        for(int s=1;s<N;s++){ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[s],Base+SEND_OFF,PeerBase[s]+bar_go_off(),8,1); }
    } else {
        volatile uint64_t *go=(volatile uint64_t*)(Region+bar_go_off()); double ts=now_sec();
        do{ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[0],Base+SEND_OFF,PeerBase[0]+bar_recv_off(MyRank),8,1);
            if(!robust){ wait_ge(go,t,"barrier release"); break; }
            for(int a=0;a<50 && *go<t;a++) usleep(2000);
            if(now_sec()-ts>WAIT_TIMEOUT_SEC) die("barrier timeout",-1);
        }while(*go<t);
    }
}

int main(int argc,char**argv){
    if(argc<3){ fprintf(stderr,"usage: %s <gguf> <blob_dir> [prompt_ids_file] [maxgen]\n",argv[0]); return 1; }
    const char *gguf=argv[1], *blob_dir=argv[2];
    const char *prompt_file = argc>3 ? argv[3] : NULL;
    int maxgen = argc>4 ? atoi(argv[4]) : 32;
    int nthr = getenv("LLM_THREADS")?atoi(getenv("LLM_THREADS")):48;
    int max_seq = getenv("MAX_SEQ")?atoi(getenv("MAX_SEQ")):2048;
    int rc;

    /* ---- uTofu bootstrap ---- */
    utofu_tni_id_t *tni_ids=NULL; size_t num_tnis=0;
    rc=utofu_get_onesided_tnis(&tni_ids,&num_tnis); if(rc!=UTOFU_SUCCESS) die("get_onesided_tnis",rc);
    if(num_tnis<1) die("no onesided TNIs",-1);
    uint8_t my_coords[TOFU_NCOORDS]={0};
    rc=utofu_query_my_coords(my_coords); if(rc!=UTOFU_SUCCESS) die("query_my_coords",rc);
    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N=read_topo(topo);
    MyRank=-1; for(int r=0;r<N;r++) if(memcmp(topo[r],my_coords,TOFU_NCOORDS)==0) MyRank=r;
    if(MyRank<0) die("my coords not in topo",-1);

    /* ---- load this stage's shard ---- */
    char blob[1100],mani[1100];
    snprintf(blob,sizeof blob,"%s/rank%02d.blob",blob_dir,MyRank);
    snprintf(mani,sizeof mani,"%s/rank%02d.manifest",blob_dir,MyRank);
    int L0,L1;
    transformer_model *m=gemma4_pp_load(gguf,blob,mani,MyRank,N,max_seq,&L0,&L1);
    if(!m) die("pp load failed",-1);
    transformer_set_threads(m,nthr);
    int nl=m->n_layers; n_embd_g=m->n_embd;
    int is_first=(MyRank==0), is_last=(MyRank==N-1);

    /* ---- region layout ---- */
    size_t hbytes=alignup((size_t)n_embd_g*sizeof(float));
    SEND_OFF=0;
    BAR_BASE=DEMO_CACHE_LINE;
    HRECV_OFF=BAR_BASE+(size_t)(N+1)*DEMO_CACHE_LINE;
    HSEQ_OFF =HRECV_OFF+hbytes;
    TTOK_OFF =HSEQ_OFF+DEMO_CACHE_LINE;
    TSEQ_OFF =TTOK_OFF+DEMO_CACHE_LINE;
    HSEND_OFF=TSEQ_OFF+DEMO_CACHE_LINE;
    SCRATCH_OFF=HSEND_OFF+hbytes;          /* 8B seq/token send scratch */
    REGION_SZ=SCRATCH_OFF+DEMO_CACHE_LINE;
    if(posix_memalign((void**)&Region,DEMO_CACHE_LINE,REGION_SZ)!=0) die("posix_memalign",-1);
    memset(Region,0,REGION_SZ);

    utofu_tni_id_t tni=tni_ids[0];
    rc=utofu_create_vcq_with_cmp_id(tni,DEMO_CMP_ID,0,&Vcq); if(rc!=UTOFU_SUCCESS) die("create_vcq",rc);
    utofu_vcq_id_t my_real; rc=utofu_query_vcq_id(Vcq,&my_real); if(rc!=UTOFU_SUCCESS) die("query_vcq_id",rc);
    rc=utofu_reg_mem_with_stag(Vcq,Region,REGION_SZ,RUN_STAG,0,&Base); if(rc!=UTOFU_SUCCESS) die("reg_mem",rc);
    for(int r=0;r<N;r++){
        if(r==MyRank){ PeerVcq[r]=my_real; PeerBase[r]=Base; continue; }
        rc=utofu_construct_vcq_id(topo[r],tni,DEMO_CQ_ID,DEMO_CMP_ID,&PeerVcq[r]); if(rc!=UTOFU_SUCCESS) die("construct_vcq_id",rc);
        utofu_set_vcq_id_path(&PeerVcq[r],NULL);
        rc=utofu_query_stadd(PeerVcq[r],RUN_STAG,&PeerBase[r]); if(rc!=UTOFU_SUCCESS) die("query_stadd",rc);
    }
    free(tni_ids);
    barrier_robust(1);
    if(MyRank==0) fprintf(stderr,"[pp] %d stages up; layers/stage ~%d; starting\n",N,nl/N);

    /* ---- prompt ---- */
    int prompt[2048]; int P=0;
    if(prompt_file){ FILE*pf=fopen(prompt_file,"r"); if(pf){ int v; while(P<2048 && fscanf(pf,"%d",&v)==1) prompt[P++]=v; fclose(pf); } }
    if(P==0){ int dflt[]={2,651,6037,576,6081,603,476,6892,576}; P=sizeof dflt/sizeof dflt[0]; for(int i=0;i<P;i++) prompt[i]=dflt[i]; }
    int total=P+maxgen;
    int *gen=(int*)calloc(maxgen,sizeof(int)); int ngen=0;

    volatile uint64_t *hseq=(volatile uint64_t*)(Region+HSEQ_OFF);
    volatile uint64_t *tseq=(volatile uint64_t*)(Region+TSEQ_OFF);
    float *hrecv=(float*)(Region+HRECV_OFF);
    float *hsend=(float*)(Region+HSEND_OFF);
    uint64_t *scr=(uint64_t*)(Region+SCRATCH_OFF);
    double t0=now_sec(), t_pf=now_sec();
    double acc_fwd=0, acc_wait=0; int use_persist=(getenv("GEMMA4_PP_PERSIST")&&atoi(getenv("GEMMA4_PP_PERSIST")));

    for(int p=0; p<total; p++){
        if(is_first){
            int t; double tw0=now_sec();
            if(p==0) t=prompt[0];
            else { wait_ge(tseq,(uint64_t)p,"rank0 token wait");           /* token from p-1 */
                   int nt=(int)*(volatile uint64_t*)(Region+TTOK_OFF);
                   t = (p<P)? prompt[p] : nt; }
            acc_wait+=now_sec()-tw0;
            transformer_embed_token(m,t);
            double tf0=now_sec();
            (use_persist?transformer_forward_partial_persistent(m,p,L0,L1):transformer_forward_partial(m,p,L0,L1));
            acc_fwd+=now_sec()-tf0;
            memcpy(hsend,m->x,(size_t)n_embd_g*sizeof(float));
            put_issue(PeerVcq[1],Base+HSEND_OFF,PeerBase[1]+HRECV_OFF,(size_t)n_embd_g*sizeof(float),0);
            *scr=(uint64_t)(p+1); put_issue(PeerVcq[1],Base+SCRATCH_OFF,PeerBase[1]+HSEQ_OFF,8,1);
        } else {
            double tw0=now_sec();
            wait_ge(hseq,(uint64_t)(p+1),"hidden wait");
            acc_wait+=now_sec()-tw0;
            transformer_set_hidden(m,hrecv);
            double tf0=now_sec();
            (use_persist?transformer_forward_partial_persistent(m,p,L0,L1):transformer_forward_partial(m,p,L0,L1));
            acc_fwd+=now_sec()-tf0;
            if(is_last){
                float *lg=transformer_compute_logits(m);
                int nt=0; float mx=lg[0]; for(int i=1;i<m->n_vocab;i++) if(lg[i]>mx){mx=lg[i];nt=i;}
                if(p>=P-1 && ngen<maxgen) gen[ngen++]=nt;
                if(p==P-1) t_pf=now_sec();   /* prefill (P prompt tokens) done */
                /* send next token to rank 0: token value then seq (ordered per VCQ pair) */
                *scr=(uint64_t)nt; put_issue(PeerVcq[0],Base+SCRATCH_OFF,PeerBase[0]+TTOK_OFF,8,1);  /* drain: scratch reused next */
                *scr=(uint64_t)(p+1); put_issue(PeerVcq[0],Base+SCRATCH_OFF,PeerBase[0]+TSEQ_OFF,8,1);
            } else {
                memcpy(hsend,m->x,(size_t)n_embd_g*sizeof(float));
                put_issue(PeerVcq[MyRank+1],Base+HSEND_OFF,PeerBase[MyRank+1]+HRECV_OFF,(size_t)n_embd_g*sizeof(float),0);
                *scr=(uint64_t)(p+1); put_issue(PeerVcq[MyRank+1],Base+SCRATCH_OFF,PeerBase[MyRank+1]+HSEQ_OFF,8,1);
            }
        }
    }
    double tend=now_sec(), dt=tend-t0;
    fprintf(stderr,"[pp] rank%d: fwd %.3fs wait %.3fs (over %d toks, %.3fs/fwd)\n",
            MyRank,acc_fwd,acc_wait,total,acc_fwd/total);
    if(is_last){
        double pf_t = t_pf - t0, dec_t = tend - t_pf;          /* prefill=P prompt toks, decode=maxgen */
        double pf_tps = pf_t>0 ? P/pf_t : 0, dec_tps = dec_t>0 ? maxgen/dec_t : 0;
        fprintf(stderr,"[pp] N=%d prefill %.2f tok/s (%d tok, %.3fs)  decode %.2f tok/s (%d tok, %.3fs)\n",
                N,pf_tps,P,pf_t,dec_tps,maxgen,dec_t);
        printf("PPGEN N=%d prefill_tps=%.3f decode_tps=%.3f n=%d tokens:",N,pf_tps,dec_tps,ngen);
        for(int i=0;i<ngen;i++) printf(" %d",gen[i]);
        printf("\n");
        /* also write to a shared-FS file: mpiexec does NOT reliably forward rank stdout */
        const char *rf = getenv("GEMMA4_RESULT_FILE"); if(!rf||!*rf) rf="gemma4_pp_result.txt";
        FILE *f=fopen(rf,"w");
        if(f){ fprintf(f,"PPGEN N=%d prefill_tps=%.4f decode_tps=%.4f P=%d maxgen=%d n=%d tokens:",
                       N,pf_tps,dec_tps,P,maxgen,ngen);
               for(int i=0;i<ngen;i++) fprintf(f," %d",gen[i]); fprintf(f,"\n"); fclose(f); }
    }
    barrier_robust(1);
    return 0;
}
