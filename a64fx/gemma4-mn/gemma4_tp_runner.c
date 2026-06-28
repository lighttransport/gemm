/* gemma4_tp_runner.c - tensor-parallel (TP) multinode runner for Gemma-4 12B BF16.
 *
 * Every rank/node holds a ROW/COL slice of EVERY weight (Megatron sharding, see
 * gemma4_stage.c "tp" + TP_DESIGN.md) and runs the FULL forward in lockstep; the sharded
 * matvecs combine via uTofu recursive-doubling all-reduce (tp_allreduce.h). MPI only places
 * ranks (no MPI_Init). Per token (identical on every rank): embed -> forward (all layers,
 * 2 all-reduce-SUM/layer) -> vocab-parallel lm_head slice -> local argmax -> all-reduce
 * argmax -> global token. No pipeline handoff; the token stream is identical on all ranks.
 *
 * Run: (after tofu_topo_helper writes tofu_topo.txt) mpiexec -np N -vcoordfile vc \
 *        ./gemma4_tp_runner <gguf> <blob_dir> [prompt_ids_file] [maxgen]
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *        -I../../common gemma4_tp_runner.c -lm -lpthread -lhwb -ltofucom -o gemma4_tp_runner
 */
#define _GNU_SOURCE
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
#include "gemma4_tp_load.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define MAX_NODES 128
#define RUN_STAG  DEMO_STAG
#define WAIT_TIMEOUT_SEC 300.0
static void die(const char *what, int rc){ fprintf(stderr,"FATAL: %s (rc=%d)\n",what,rc); exit(1); }
static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

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

/* ---- uTofu state (barrier region only; tp_allreduce.h owns its own region) ---- */
static int N, MyRank;
static char *Region;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t Base;
static utofu_vcq_id_t PeerVcq[MAX_NODES];
static utofu_stadd_t PeerBase[MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static size_t SEND_OFF, BAR_BASE, REGION_SZ;
static uint64_t Bt = 1;
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
static void barrier_robust(void){
    uint64_t t=++Bt; char *sb=Region+SEND_OFF;
    if(MyRank==0){
        for(int s=1;s<N;s++) wait_ge((volatile uint64_t*)(Region+bar_recv_off(s)),t,"barrier fan-in");
        for(int s=1;s<N;s++){ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[s],Base+SEND_OFF,PeerBase[s]+bar_go_off(),8,1); }
    } else {
        volatile uint64_t *go=(volatile uint64_t*)(Region+bar_go_off()); double ts=now_sec();
        do{ *(volatile uint64_t*)sb=t; put_issue(PeerVcq[0],Base+SEND_OFF,PeerBase[0]+bar_recv_off(MyRank),8,1);
            for(int a=0;a<50 && *go<t;a++) usleep(2000);
            if(now_sec()-ts>WAIT_TIMEOUT_SEC) die("barrier timeout",-1);
        }while(*go<t);
    }
}

/* transformer_set_tp callback: SUM-reduce a partial across all TP ranks (o-proj/down).
 * TP_SKIP_AR=1 makes it a no-op -> WRONG output, but isolates compute-only time so we can
 * measure the comm fraction (== the ceiling that batched/amortized decode could reach). */
static int g_skip_ar = -1;
static void ar_sum_cb(float *buf, int count, void *ctx){
    if(g_skip_ar<0){ const char*e=getenv("TP_SKIP_AR"); g_skip_ar = (e&&atoi(e))?1:0; }
    if(g_skip_ar) return;
    tp_allreduce_sum((tp_comm*)ctx, buf, count);
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

    /* ---- load this rank's TP shard ---- */
    char blob[1100],mani[1100];
    snprintf(blob,sizeof blob,"%s/rank%02d.blob",blob_dir,MyRank);
    snprintf(mani,sizeof mani,"%s/rank%02d.manifest",blob_dir,MyRank);
    transformer_model *m=gemma4_tp_load(gguf,blob,mani,MyRank,N,max_seq);
    if(!m) die("tp load failed",-1);
    transformer_set_threads(m,nthr);
    int n_embd_g=m->n_embd, V=m->n_vocab;

    /* ---- barrier region ---- */
    SEND_OFF=0; BAR_BASE=DEMO_CACHE_LINE; REGION_SZ=BAR_BASE+(size_t)(N+1)*DEMO_CACHE_LINE;
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
    barrier_robust();

    /* ---- TP all-reduce comm (its own registered region) ---- */
    /* batched forward reduces [M,n_embd] at once -> size for the largest batch (cap 32). */
    int batch_M = getenv("GEMMA4_TP_BATCH") ? atoi(getenv("GEMMA4_TP_BATCH")) : 0;
    int ar_max = n_embd_g * (batch_M > 1 ? (batch_M < 64 ? batch_M : 64) : 1);
    static tp_comm comm;
    if(tp_comm_init(&comm, Vcq, PeerVcq, MyRank, N, ar_max, barrier_robust)!=0) die("tp_comm_init",-1);
    transformer_set_tp(m, MyRank, N, ar_sum_cb, &comm);
    barrier_robust();
    if(MyRank==0) fprintf(stderr,"[tp] %d ranks up; sharded forward; starting\n",N);

    /* ---- prompt ---- */
    int prompt[2048]; int P=0;
    if(prompt_file){ FILE*pf=fopen(prompt_file,"r"); if(pf){ int v; while(P<2048 && fscanf(pf,"%d",&v)==1) prompt[P++]=v; fclose(pf); } }
    if(P==0){ int dflt[]={2,651,6037,576,6081,603,476,6892,576}; P=sizeof dflt/sizeof dflt[0]; for(int i=0;i<P;i++) prompt[i]=dflt[i]; }
    int total=P+maxgen;
    int *gen=(int*)calloc(maxgen,sizeof(int)); int ngen=0;
    long v0=(long)MyRank*V/N, v1=(long)(MyRank+1)*V/N;

    /* ---- batched-throughput mode (comm amortized over M tokens/forward) ---- */
    if(batch_M > 1){
        int M = batch_M;
        int *btok=(int*)malloc((size_t)M*sizeof(int));
        for(int i=0;i<M;i++) btok[i]=prompt[i%P];
        /* correctness: batched prefill of the real prompt -> argmax(last) == M=1 first gen */
        if(!transformer_prefill_gemm(m, prompt, P, 0)){
            if(MyRank==0) fprintf(stderr,"[tp] BATCH unavailable (prefill_gemm NULL: PLE model?)\n");
            free(btok); barrier_robust(); return 0;
        }
        int chk=0; { float mx=m->logits[0]; for(int i=1;i<V;i++) if(m->logits[i]>mx){mx=m->logits[i];chk=i;} }
        transformer_prefill_gemm(m, btok, M, 0);      /* warm */
        barrier_robust();
        int reps=8; double tb=now_sec();
        for(int r=0;r<reps;r++) transformer_prefill_gemm(m, btok, M, 0);
        double el=now_sec()-tb; double tps=el>0?(double)reps*M/el:0;
        if(MyRank==0){
            fprintf(stderr,"[tp] N=%d BATCH M=%d: %.2f tok/s (%d forwards x %d tok in %.3fs)  prompt-next-argmax=%d\n",
                    N,M,tps,reps,M,el,chk);
            const char*rf=getenv("GEMMA4_RESULT_FILE"); if(!rf||!*rf) rf="gemma4_tp_result.txt";
            FILE*f=fopen(rf,"w");
            if(f){ fprintf(f,"TPBATCH N=%d M=%d batched_tps=%.4f prompt_next_argmax=%d\n",N,M,tps,chk); fclose(f); }
            printf("TPBATCH N=%d M=%d batched_tps=%.3f prompt_next_argmax=%d\n",N,M,tps,chk);
        }
        free(btok); barrier_robust(); return 0;
    }

    double t0=now_sec(), t_pf=now_sec();
    int next=prompt[0];
    for(int p=0; p<total; p++){
        int t = (p<P)? prompt[p] : next;
        transformer_embed_token(m,t);
        transformer_forward_partial(m,p,0,m->n_layers);          /* full TP forward */
        transformer_compute_logits_slice(m,(int)v0,(int)v1);     /* local vocab slice */
        float mx=m->logits[0]; int32_t gi=(int32_t)v0;
        for(long i=1;i<v1-v0;i++) if(m->logits[i]>mx){ mx=m->logits[i]; gi=(int32_t)(v0+i); }
        tp_allreduce_argmax(&comm,&mx,&gi);                      /* global argmax token */
        next=(int)gi;
        if(p>=P-1 && ngen<maxgen) gen[ngen++]=next;
        if(p==P-1) t_pf=now_sec();
    }
    double tend=now_sec(), dt=tend-t0;
    if(MyRank==0){
        double pf_t=t_pf-t0, dec_t=tend-t_pf;
        double pf_tps=pf_t>0?P/pf_t:0, dec_tps=dec_t>0?maxgen/dec_t:0;
        fprintf(stderr,"[tp] N=%d prefill %.2f tok/s (%d tok, %.3fs)  decode %.2f tok/s (%d tok, %.3fs)\n",
                N,pf_tps,P,pf_t,dec_tps,maxgen,dec_t);
        printf("TPGEN N=%d prefill_tps=%.3f decode_tps=%.3f n=%d tokens:",N,pf_tps,dec_tps,ngen);
        for(int i=0;i<ngen;i++) printf(" %d",gen[i]);
        printf("\n");
        const char *rf=getenv("GEMMA4_RESULT_FILE"); if(!rf||!*rf) rf="gemma4_tp_result.txt";
        FILE*f=fopen(rf,"w");
        if(f){ fprintf(f,"TPGEN N=%d prefill_tps=%.4f decode_tps=%.4f P=%d maxgen=%d n=%d tokens:",
                       N,pf_tps,dec_tps,P,maxgen,ngen);
               for(int i=0;i<ngen;i++) fprintf(f," %d",gen[i]); fprintf(f,"\n"); fclose(f); }
    }
    (void)dt; (void)n_embd_g;
    barrier_robust();
    return 0;
}
