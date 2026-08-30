#define _GNU_SOURCE
#include <mpi.h>
#ifdef Q38FN_HAVE_FAPP
#include <fj_tool/fapp.h>
#pragma weak fapp_start
#pragma weak fapp_stop
#endif
#define Q38FN_TP_BLOB_IMPLEMENTATION
#define Q38FN_TP_RUNTIME_IMPLEMENTATION
#include "../common/q38fn_tp_runtime.h"
#define GLM5_BPE_IMPLEMENTATION
#include "../common/glm5_bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>

#ifdef Q38FN_USE_UTOFU
#include <utofu.h>
#include "../a64fx/utofu-tests/tofu_demo.h"
#include "q38fn_utofu_transport.h"
#endif

typedef struct {
    double seconds;
    long calls;
#ifdef Q38FN_USE_UTOFU
    utofu_vcq_hdl_t vcq;
    q38fn_utofu_transport *tofu;
    int initialized;
#endif
} q38fn_comm;
static MPI_Comm q38fn_tp_mpi_comm=MPI_COMM_WORLD;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void q38fn_sum(float*v,int n,void*opaque){q38fn_comm*p=opaque;double t=now();
#ifdef Q38FN_USE_UTOFU
    q38fn_utofu_transport_sum(p->tofu,v,n);
#else
    MPI_Allreduce(MPI_IN_PLACE,v,n,MPI_FLOAT,MPI_SUM,q38fn_tp_mpi_comm);
#endif
    p->seconds+=now()-t;p->calls++;}
static void q38fn_argmax(float*v,int*i,void*opaque){q38fn_comm*p=opaque;double t=now();
#ifdef Q38FN_USE_UTOFU
    if(getenv("Q38FN_TP_UTOFU_ARGMAX")){
        int32_t index=*i;q38fn_utofu_transport_argmax(p->tofu,v,&index);*i=(int)index;
    }else{
        /* The TP12 non-power-of-two uTofu argmax occasionally returned the
         * float payload bits as an out-of-range token index.  This collective
         * is only eight bytes once per token; use MPI MAXLOC by default while
         * retaining uTofu for the thousands of bandwidth-critical sums. */
        struct{float v;int i;}in={*v,*i},out;
        MPI_Allreduce(&in,&out,1,MPI_FLOAT_INT,MPI_MAXLOC,q38fn_tp_mpi_comm);
        *v=out.v;*i=out.i;
    }
#else
    struct{float v;int i;}in={*v,*i},out;MPI_Allreduce(&in,&out,1,MPI_FLOAT_INT,MPI_MAXLOC,q38fn_tp_mpi_comm);*v=out.v;*i=out.i;
#endif
    p->seconds+=now()-t;p->calls++;}
static void die(const char*s,int rank){
    fprintf(stderr,"q38fn_tp rank=%d: %s\n",rank,s);fflush(stderr);
    const char*d=getenv("Q38FN_TP_DIAG_DIR");
    if(d&&*d){char p[4096];if(snprintf(p,sizeof(p),"%s/die-rank-%02d.log",d,rank)<(int)sizeof(p)){FILE*f=fopen(p,"w");if(f){fprintf(f,"q38fn_tp rank=%d: %s\n",rank,s);fclose(f);}}}
    MPI_Abort(MPI_COMM_WORLD,1);
}
static uint64_t checksum(const float*x,size_t n){uint64_t h=UINT64_C(1469598103934665603);for(size_t i=0;i<n;i++){uint32_t u;memcpy(&u,x+i,4);for(int b=0;b<4;b++){h^=(u>>(8*b))&255u;h*=UINT64_C(1099511628211);}}return h;}
static int finite_vector(const float*x,size_t n){for(size_t i=0;i<n;i++)if(!isfinite(x[i]))return 0;return 1;}
static void trace_stats(FILE*f,int layer,const float*x,size_t n){if(!f)return;size_t bad=0;float ma=0;for(size_t i=0;i<n;i++){if(!isfinite(x[i]))bad++;else if(fabsf(x[i])>ma)ma=fabsf(x[i]);}fprintf(f,"stats layer=%d nonfinite=%zu maxabs=%.9g\n",layer,bad,ma);fflush(f);}
static void trace_component(FILE*f,const char*name,const float*x,size_t n){if(!f)return;size_t bad=0;float ma=0;for(size_t i=0;i<n;i++){if(!isfinite(x[i]))bad++;else if(fabsf(x[i])>ma)ma=fabsf(x[i]);}fprintf(f,"component=%s nonfinite=%zu maxabs=%.9g checksum=%016llx",name,bad,ma,(unsigned long long)checksum(x,n));if(n<=4)for(size_t i=0;i<n;i++)fprintf(f," value%zu=%.9g",i,x[i]);fputc('\n',f);fflush(f);}
static int dump_vector(int rank,const char*name,const float*x,size_t n){const char*d=getenv("Q38FN_DUMP_DIR");char p[4096];FILE*f;if(rank||!d||!*d)return 0;if(snprintf(p,sizeof(p),"%s/tp-%s.f32",d,name)>=(int)sizeof(p)||(f=fopen(p,"wb"))==NULL)return-1;return fwrite(x,sizeof(*x),n,f)==n&&!fclose(f)?0:-1;}

#ifdef Q38FN_USE_UTOFU
static void mpi_barrier_callback(void){MPI_Barrier(q38fn_tp_mpi_comm);}
static int q38fn_comm_init(q38fn_comm*c,int rank,int ranks)
{
    /* Decode performs thousands of short reductions.  The lean robust mode
     * drains/invalidate-polls periodically instead of on every spin; TP12
     * FAPP runs found 32 spins best while retaining the robust receive path. */
    if(!getenv("TP_AR_ROBUST"))setenv("TP_AR_ROBUST","2",0);
    if(!getenv("TP_AR_POLL_SPINS"))setenv("TP_AR_POLL_SPINS","32",0);
    utofu_tni_id_t*tnis=NULL;size_t ntnis=0;utofu_vcq_id_t mine,peers[Q38FN_TP_RANKS];
    int tni_rc=utofu_get_onesided_tnis(&tnis,&ntnis);
    if(tni_rc!=UTOFU_SUCCESS||ntnis<1){fprintf(stderr,"q38fn utofu: get tnis rc=%d count=%zu\n",tni_rc,ntnis);fflush(stderr);return-1;}
    int rc=utofu_create_vcq_with_cmp_id(tnis[DEMO_TNI_INDEX],DEMO_CMP_ID,0,&c->vcq);free(tnis);
    if(rc!=UTOFU_SUCCESS){fprintf(stderr,"q38fn utofu: create vcq rc=%d\n",rc);fflush(stderr);return-1;}
    rc=utofu_query_vcq_id(c->vcq,&mine);
    if(rc!=UTOFU_SUCCESS){fprintf(stderr,"q38fn utofu: query vcq rc=%d\n",rc);fflush(stderr);return-1;}
    rc=MPI_Allgather(&mine,(int)sizeof mine, MPI_BYTE, peers,(int)sizeof mine,MPI_BYTE,q38fn_tp_mpi_comm);
    if(rc!=MPI_SUCCESS){fprintf(stderr,"q38fn utofu: allgather rc=%d\n",rc);fflush(stderr);return-1;}
    for(int i=0;i<ranks;i++)utofu_set_vcq_id_path(&peers[i],NULL);
    c->tofu=q38fn_utofu_transport_create(c->vcq,peers,rank,ranks,QTP_HC,mpi_barrier_callback);
    if(!c->tofu){fprintf(stderr,"q38fn utofu: tp_comm_init failed rank=%d ranks=%d\n",rank,ranks);fflush(stderr);return-1;}
    c->initialized=1;return 0;
}
static void q38fn_comm_close(q38fn_comm*c){if(c->initialized){q38fn_utofu_transport_free(c->tofu);c->tofu=NULL;utofu_free_vcq(c->vcq);c->initialized=0;}}
#else
static int q38fn_comm_init(q38fn_comm*c,int rank,int ranks){(void)c;(void)rank;(void)ranks;return 0;}
static void q38fn_comm_close(q38fn_comm*c){(void)c;}
#endif

static FILE*open_trace(int rank)
{
    const char*d=getenv("Q38FN_TP_TRACE_DIR");if(!d||!*d)return NULL;
    char path[4096];int n=snprintf(path,sizeof(path),"%s/rank-%02d.log",d,rank);
    if(n<0||(size_t)n>=sizeof(path)){errno=ENAMETOOLONG;return NULL;}return fopen(path,"w");
}

typedef struct {
    pthread_t thread; pthread_mutex_t lock; pthread_cond_t wake,done;
    q38fn_tp_model *model; uint64_t token,previous,previous2;
    float *output; int pending,ready,stop,rc;
} ngram_worker;
static void *ngram_worker_main(void *opaque){ngram_worker*w=opaque;pthread_mutex_lock(&w->lock);for(;;){while(!w->pending&&!w->stop)pthread_cond_wait(&w->wake,&w->lock);if(w->stop)break;uint64_t t=w->token,p=w->previous,p2=w->previous2;float*out=w->output;w->pending=0;pthread_mutex_unlock(&w->lock);int rc=q38fn_tp_ngram_local(w->model,t,p,p2,out);pthread_mutex_lock(&w->lock);w->rc=rc;w->ready=1;pthread_cond_signal(&w->done);}pthread_mutex_unlock(&w->lock);return NULL;}
static int ngram_worker_init(ngram_worker*w,q38fn_tp_model*m){memset(w,0,sizeof(*w));w->model=m;if(pthread_mutex_init(&w->lock,NULL)||pthread_cond_init(&w->wake,NULL)||pthread_cond_init(&w->done,NULL)||pthread_create(&w->thread,NULL,ngram_worker_main,w))return-1;return 0;}
static void ngram_worker_start(ngram_worker*w,uint64_t t,uint64_t p,uint64_t p2,float*out){pthread_mutex_lock(&w->lock);w->token=t;w->previous=p;w->previous2=p2;w->output=out;w->ready=0;w->pending=1;pthread_cond_signal(&w->wake);pthread_mutex_unlock(&w->lock);}
static int ngram_worker_wait(ngram_worker*w){pthread_mutex_lock(&w->lock);while(!w->ready)pthread_cond_wait(&w->done,&w->lock);int rc=w->rc;w->ready=0;pthread_mutex_unlock(&w->lock);if(!rc)q38fn_tp_ngram_reduce(w->model,w->output);return rc;}
static void ngram_worker_close(ngram_worker*w){pthread_mutex_lock(&w->lock);w->stop=1;pthread_cond_signal(&w->wake);pthread_mutex_unlock(&w->lock);pthread_join(w->thread,NULL);pthread_cond_destroy(&w->done);pthread_cond_destroy(&w->wake);pthread_mutex_destroy(&w->lock);}

int main(int argc,char**argv)
{
    int rank,ranks,world_rank,world_ranks,stage=0,pipeline=0;
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);MPI_Comm_size(MPI_COMM_WORLD,&world_ranks);
    rank=world_rank;ranks=world_ranks;
    if(getenv("Q38FN_TP_PIPELINE")&&world_ranks>Q38FN_TP_RANKS&&world_ranks%Q38FN_TP_RANKS==0){
        pipeline=1;stage=world_rank/Q38FN_TP_RANKS;rank=world_rank%Q38FN_TP_RANKS;ranks=Q38FN_TP_RANKS;
        MPI_Comm_split(MPI_COMM_WORLD,stage,rank,&q38fn_tp_mpi_comm);
    }
    const char*model_dir=NULL,*local_base="/local/u14346/q38fn-tp",*prompt="Write a complete C11 program that prints the first ten Fibonacci numbers. Return only code.";
    int max_gen=32,max_seq=512,layers=Q38FN_LAYERS,probe=0,probe_token=42,prompt_repeat=1;
    for(int a=1;a<argc;a++){
        if(!strcmp(argv[a],"--local-base")&&a+1<argc)local_base=argv[++a];
        else if(!strcmp(argv[a],"--prompt")&&a+1<argc)prompt=argv[++a];
        else if(!strcmp(argv[a],"--max-gen")&&a+1<argc)max_gen=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--max-seq")&&a+1<argc)max_seq=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--prompt-repeat")&&a+1<argc)prompt_repeat=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--layers")&&a+1<argc)layers=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--probe-token")&&a+1<argc){probe=1;probe_token=atoi(argv[++a]);}
        else if(argv[a][0]!='-'&&!model_dir)model_dir=argv[a];else die("invalid arguments",rank);
    }
    if(!model_dir||ranks!=Q38FN_TP_RANKS||layers<1||layers>Q38FN_LAYERS||prompt_repeat<1||
       (pipeline&&world_ranks/Q38FN_TP_RANKS!=3))die("requires MODEL, configured TP rank count, and valid arguments",world_rank);
    int layer_start=pipeline?stage*(Q38FN_LAYERS/3):0;
    int layer_end=pipeline?(stage+1)*(Q38FN_LAYERS/3):layers;
    FILE*trace=open_trace(world_rank);q38fn_comm comm={0};if(q38fn_comm_init(&comm,rank,ranks))die("communication init failed",world_rank);q38fn_tp_model m;
    int profile=getenv("Q38FN_TP_PROFILE")!=NULL;double layer_s[Q38FN_LAYERS]={0};long layer_calls[Q38FN_LAYERS]={0};
    double load0=now();if(q38fn_tp_model_open(&m,local_base,rank,ranks,q38fn_sum,q38fn_argmax,&comm))die("blob load failed",rank);double load_s=now()-load0;
    int skip_ngram=getenv("Q38FN_TP_SKIP_NGRAM")!=NULL;
    ngram_worker ngworker;int ngworker_ready=(!pipeline||stage==0)&&!skip_ngram;if(ngworker_ready&&ngram_worker_init(&ngworker,&m))die("ngram worker init",world_rank);
    if(trace){fprintf(trace,"rank=%d transport=%s load=%.6f\n",rank,
#ifdef Q38FN_USE_UTOFU
        "utofu",
#else
        "mpi",
#endif
        load_s);fflush(trace);}
    q38fn_tp_delta_state delta[Q38FN_LAYERS]={{0}};q38fn_tp_attention_state attention[Q38FN_LAYERS]={{0}};q38fn_tp_ple_state ple={0};
    for(int L=layer_start;L<layer_end;L++){if(q38fn_layer_is_full_attention((size_t)L)){if(q38fn_tp_attention_init(&attention[L],(size_t)max_seq))die("attention state alloc",world_rank);}else if(q38fn_tp_delta_init(&delta[L]))die("delta state alloc",world_rank);}
    if(layer_start<=Q38FN_PLE_LAYER&&layer_end>Q38FN_PLE_LAYER&&q38fn_tp_ple_init(&ple))die("PLE state alloc",world_rank);
    float*hyper=malloc((size_t)Q38FN_HC_COUNT*Q38FN_HIDDEN*4);if(!hyper)die("activation alloc",rank);
    if(pipeline&&probe)die("pipeline probe mode is not supported",world_rank);
    if(probe){float emb[Q38FN_HIDDEN],ng[Q38FN_HIDDEN];if(q38fn_tp_embedding(&m,probe_token,emb))die("probe embedding",rank);for(int s=0;s<4;s++)memcpy(hyper+s*Q38FN_HIDDEN,emb,sizeof(emb));double t=now();for(int L=0;L<layers;L++){double lt=profile?now():0;if(L==Q38FN_PLE_LAYER){memset(ng,0,sizeof(ng));if(q38fn_tp_ple_apply(&m,L,&ple,(uint64_t)probe_token,ng,hyper))die("PLE probe",rank);}if(q38fn_layer_is_full_attention((size_t)L)){if(q38fn_tp_attention_layer(&m,L,&attention[L],hyper))die("attention probe",rank);}else if(profile&&L==0){float x[Q38FN_HIDDEN],y[Q38FN_HIDDEN],inj[4];if(qtp_hc(&m,L,"attn",hyper,x,inj))die("hc attn probe",rank);trace_component(trace,"hc_attn_x",x,Q38FN_HIDDEN);trace_component(trace,"hc_attn_inj",inj,4);if(dump_vector(rank,"hc-attn-x",x,Q38FN_HIDDEN)||dump_vector(rank,"hc-attn-inj",inj,4))die("dump hc attn",rank);if(qtp_delta(&m,L,&delta[L],x,y))die("delta probe",rank);trace_component(trace,"delta_y",y,Q38FN_HIDDEN);if(dump_vector(rank,"delta-y",y,Q38FN_HIDDEN))die("dump delta",rank);qtp_residual(hyper,y,inj);trace_component(trace,"after_attn_residual",hyper,QTP_HC);if(dump_vector(rank,"after-attn-residual",hyper,QTP_HC))die("dump residual",rank);if(qtp_hc(&m,L,"mlp",hyper,x,inj))die("hc mlp probe",rank);trace_component(trace,"hc_mlp_x",x,Q38FN_HIDDEN);trace_component(trace,"hc_mlp_inj",inj,4);if(dump_vector(rank,"hc-mlp-x",x,Q38FN_HIDDEN)||dump_vector(rank,"hc-mlp-inj",inj,4))die("dump hc mlp",rank);if(qtp_moe(&m,L,x,y))die("moe probe",rank);trace_component(trace,"moe_y",y,Q38FN_HIDDEN);if(dump_vector(rank,"moe-y",y,Q38FN_HIDDEN))die("dump moe",rank);qtp_residual(hyper,y,inj);if(dump_vector(rank,"layer-0",hyper,QTP_HC))die("dump layer",rank);}else if(q38fn_tp_linear_layer(&m,L,&delta[L],hyper))die("linear probe",rank);if(!finite_vector(hyper,QTP_HC))die("nonfinite probe output",rank);if(profile){layer_s[L]+=now()-lt;layer_calls[L]++;trace_stats(trace,L,hyper,(size_t)QTP_HC);}}double elapsed=now()-t;if(trace){fprintf(trace,"probe layers=%d checksum=%016llx elapsed=%.6f comm=%.6f calls=%ld\n",layers,(unsigned long long)checksum(hyper,(size_t)QTP_HC),elapsed,comm.seconds,comm.calls);fflush(trace);}if(!rank)fprintf(stderr,"Q38FN_TP_PROBE layers=%d checksum=%016llx elapsed=%.6f comm=%.6f calls=%ld load=%.3f\n",layers,(unsigned long long)checksum(hyper,(size_t)QTP_HC),elapsed,comm.seconds,comm.calls,load_s);goto cleanup;}
    glm5_bpe tok={0};int32_t*tokens=NULL;int nt=0,current=-1;float current_logit=0;
    if(!world_rank){char path[4096];snprintf(path,sizeof(path),"%s/tokenizer.json",model_dir);if(glm5_bpe_load(path,&tok))die("tokenizer",world_rank);tok.im_start=248045;tok.im_end=248046;tok.think=248068;tok.end_think=248069;tok.endoftext=248044;size_t plen=strlen(prompt),cap=plen*(size_t)prompt_repeat+(size_t)prompt_repeat+128;char*body=malloc(plen*(size_t)prompt_repeat+(size_t)prompt_repeat),*rendered=malloc(cap);tokens=malloc((size_t)max_seq*4);if(!body||!rendered||!tokens)die("prompt allocation",world_rank);size_t used=0;for(int r=0;r<prompt_repeat;r++){if(r)body[used++]=' ';memcpy(body+used,prompt,plen);used+=plen;}body[used]=0;snprintf(rendered,cap,"<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n",body);free(body);nt=glm5_bpe_encode(&tok,rendered,(int*)tokens,max_seq);free(rendered);if(nt<1||nt+max_gen>max_seq)die("prompt",world_rank);}
    MPI_Bcast(&nt,1,MPI_INT,0,MPI_COMM_WORLD);if(trace){fprintf(trace,"prompt_tokens=%d repeat=%d max_gen=%d\n",nt,prompt_repeat,max_gen);fflush(trace);}uint64_t prev=Q38FN_EOS,prev2=Q38FN_EOS;double start=now(),decode0=0;int generated=0;
    int fapp_decode=0;
    for(int pos=0;pos<nt+max_gen;pos++){
        int token;if(pos<nt){if(!world_rank)token=tokens[pos];MPI_Bcast(&token,1,MPI_INT,0,MPI_COMM_WORLD);}else token=current;
        if(pos==nt){decode0=now();
#ifdef Q38FN_HAVE_FAPP
            if(getenv("Q38FN_TP_FAPP")&&fapp_start){fapp_start("decode",0,0);fapp_decode=1;}
#endif
        }
        if(pos>=nt&&!world_rank){char piece[4096];int decoded=glm5_bpe_decode_token(&tok,token,piece,sizeof(piece));if(decoded>=0){fputs(piece,stdout);fflush(stdout);}if(trace){fprintf(trace,"token pos=%d id=%d logit=%.9g decoded=%d piece=",pos-nt,token,current_logit,decoded);if(decoded>=0)fputs(piece,trace);fputc('\n',trace);fflush(trace);}generated++;}
        if(pos>=nt&&!getenv("Q38FN_TP_IGNORE_EOS")&&
           (token==Q38FN_EOS||token==248046))break;
        float emb[Q38FN_HIDDEN],ng[Q38FN_HIDDEN];
        if(!pipeline||stage==0){if(ngworker_ready)ngram_worker_start(&ngworker,(uint64_t)token,prev,prev2,ng);if(q38fn_tp_embedding(&m,token,emb))die("embedding",world_rank);for(int s=0;s<4;s++)memcpy(hyper+s*Q38FN_HIDDEN,emb,sizeof(emb));}
        if(pipeline&&stage>0){if(!rank)MPI_Recv(hyper,QTP_HC,MPI_FLOAT,(stage-1)*Q38FN_TP_RANKS,700+stage,MPI_COMM_WORLD,MPI_STATUS_IGNORE);MPI_Bcast(hyper,QTP_HC,MPI_FLOAT,0,q38fn_tp_mpi_comm);}
        for(int L=layer_start;L<layer_end;L++){double lt=profile?now():0;if(L==Q38FN_PLE_LAYER&&!skip_ngram&&(ngram_worker_wait(&ngworker)||q38fn_tp_ple_apply(&m,L,&ple,(uint64_t)token,ng,hyper)))die("PLE",world_rank);int layer_rc=q38fn_layer_is_full_attention((size_t)L)?q38fn_tp_attention_layer(&m,L,&attention[L],hyper):q38fn_tp_linear_layer(&m,L,&delta[L],hyper);if(layer_rc){char why[96];snprintf(why,sizeof(why),"%s stage=%d layer=%d",q38fn_layer_is_full_attention((size_t)L)?"attention":"linear",stage,L);die(why,world_rank);}if(!finite_vector(hyper,QTP_HC))die("nonfinite layer output",world_rank);if(profile){layer_s[L]+=now()-lt;layer_calls[L]++;}}
        if(pipeline&&stage<2&& !rank)MPI_Send(hyper,QTP_HC,MPI_FLOAT,(stage+1)*Q38FN_TP_RANKS,701+stage,MPI_COMM_WORLD);
        if(!pipeline||stage==2){float hidden[Q38FN_HIDDEN];
            if(layers!=Q38FN_LAYERS)die("head requires all layers",world_rank);
            if(q38fn_tp_final(&m,hyper,hidden))die("final mixer failed",world_rank);
            if(!finite_vector(hidden,Q38FN_HIDDEN))die("nonfinite final mixer",world_rank);
            if(q38fn_tp_head(&m,hidden,&current,&current_logit))die("lm head failed",world_rank);
            if(current<0||current>=Q38FN_VOCAB)die("invalid head token",world_rank);
            if(!isfinite(current_logit))die("nonfinite head logit",world_rank);}
        if(pipeline){MPI_Bcast(&current,1,MPI_INT,2*Q38FN_TP_RANKS,MPI_COMM_WORLD);MPI_Bcast(&current_logit,1,MPI_FLOAT,2*Q38FN_TP_RANKS,MPI_COMM_WORLD);}
        prev2=prev;prev=(uint64_t)token;
    }
#ifdef Q38FN_HAVE_FAPP
    if(fapp_decode&&fapp_stop)fapp_stop("decode",0,0);
#else
    (void)fapp_decode;
#endif
    MPI_Barrier(MPI_COMM_WORLD);if(!world_rank){double end=now(),decode_s=end-decode0;fprintf(stderr,"\nQ38FN_TP generated=%d prefill=%.3f decode=%.3f tok_s=%.6f comm=%.3f calls=%ld load=%.3f pipeline=%d skip_ngram=%d\n",generated,decode0-start,decode_s,generated/decode_s,comm.seconds,comm.calls,load_s,pipeline,skip_ngram);if(trace){fprintf(trace,"summary generated=%d prefill=%.6f decode=%.6f tok_s=%.6f comm=%.6f calls=%ld load=%.6f pipeline=%d skip_ngram=%d\n",generated,decode0-start,decode_s,generated/decode_s,comm.seconds,comm.calls,load_s,pipeline,skip_ngram);q38fn_tp_profile_report(trace);fflush(trace);}}
    if(!world_rank){glm5_bpe_free(&tok);free(tokens);}
cleanup:if(trace&&profile){for(int L=0;L<layers;L++)if(layer_calls[L])fprintf(trace,"layer=%d seconds=%.9f calls=%ld avg_ms=%.6f\n",L,layer_s[L],layer_calls[L],1e3*layer_s[L]/layer_calls[L]);fflush(trace);}
    /* Profiling can bypass teardown to avoid perturbing the measured region and
       to work around allocator errors in experimental model paths. */
    if (!getenv("Q38FN_TP_SKIP_CLEANUP")) {
        if(ngworker_ready)ngram_worker_close(&ngworker);free(hyper);q38fn_tp_ple_close(&ple);
        for(int L=layer_start;L<layer_end;L++){q38fn_tp_delta_close(&delta[L]);q38fn_tp_attention_close(&attention[L]);}
        q38fn_tp_model_close(&m);q38fn_comm_close(&comm);if(trace)fclose(trace);
    }
    if(pipeline)MPI_Comm_free(&q38fn_tp_mpi_comm);
    MPI_Finalize();return 0;
}
