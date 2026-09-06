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
#define Q38FN_SPEC_IMPLEMENTATION
#include "../common/q38fn_spec.h"
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
/* Set only around the serialized prompt loop.  PLE state remains sequential;
 * these pointers merely supply projections computed ahead of time. */
static const float *q38fn_prefill_ple_key=NULL,*q38fn_prefill_ple_value=NULL;
static void q38fn_sum(float*v,int n,void*opaque){q38fn_comm*p=opaque;double t=now();
#ifdef Q38FN_USE_UTOFU
    if(getenv("Q38FN_TP_MPI_SUM"))
        MPI_Allreduce(MPI_IN_PLACE,v,n,MPI_FLOAT,MPI_SUM,q38fn_tp_mpi_comm);
    else q38fn_utofu_transport_sum(p->tofu,v,n);
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
static void q38fn_argmax_n(float*v,int32_t*i,int n,void*opaque){q38fn_comm*p=opaque;double t=now();
#ifdef Q38FN_USE_UTOFU
    if(getenv("Q38FN_TP_UTOFU_ARGMAX"))q38fn_utofu_transport_argmax_n(p->tofu,v,i,n);
    else{
        struct pair{float v;int i;}in[Q38FN_SPEC_MAX_WIDTH],out[Q38FN_SPEC_MAX_WIDTH];
        for(int k=0;k<n;k++){in[k].v=v[k];in[k].i=i[k];}
        MPI_Allreduce(in,out,n,MPI_FLOAT_INT,MPI_MAXLOC,q38fn_tp_mpi_comm);
        for(int k=0;k<n;k++){v[k]=out[k].v;i[k]=out[k].i;}
    }
#else
    struct pair{float v;int i;}in[Q38FN_SPEC_MAX_WIDTH],out[Q38FN_SPEC_MAX_WIDTH];
    for(int k=0;k<n;k++){in[k].v=v[k];in[k].i=i[k];}
    MPI_Allreduce(in,out,n,MPI_FLOAT_INT,MPI_MAXLOC,q38fn_tp_mpi_comm);
    for(int k=0;k<n;k++){v[k]=out[k].v;i[k]=out[k].i;}
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
static void trace_stats(FILE*f,int layer,const float*x,size_t n){if(!f)return;size_t bad=0;float ma=0;for(size_t i=0;i<n;i++){if(!isfinite(x[i]))bad++;else if(fabsf(x[i])>ma)ma=fabsf(x[i]);}fprintf(f,"stats layer=%d nonfinite=%zu maxabs=%.9g checksum=%016llx\n",layer,bad,ma,(unsigned long long)checksum(x,n));fflush(f);}
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

typedef struct {
    q38fn_tp_model *model;
    q38fn_tp_delta_state *delta;
    q38fn_tp_attention_state *attention;
    q38fn_tp_ple_state *ple;
    float *hyper;
    int layers;
    int skip_ngram;
    int rank;
} q38fn_decode_step;

static uint64_t q38fn_target_state_checksum(const q38fn_decode_step *step)
{
    uint64_t h=UINT64_C(1469598103934665603);
    size_t dc=q38fn_tp_delta_snapshot_bytes()/sizeof(float);
    for(int layer=0;layer<step->layers;layer++){
        if(q38fn_layer_is_full_attention((size_t)layer)){
            const q38fn_tp_attention_state *s=&step->attention[layer];
            size_t n=s->length*(size_t)Q38FN_KV_HEADS*Q38FN_HEAD_DIM;
            h^=checksum(s->keys,n);h*=UINT64_C(1099511628211);
            h^=checksum(s->values,n);h*=UINT64_C(1099511628211);
            h^=(uint64_t)s->length;h*=UINT64_C(1099511628211);
        }else{
            size_t conv=(size_t)Q38FN_LINEAR_CONV_DIM*4;
            h^=checksum(step->delta[layer].conv,conv);h*=UINT64_C(1099511628211);
            h^=checksum(step->delta[layer].recurrent,dc-conv);h*=UINT64_C(1099511628211);
        }
    }
    h^=checksum(step->ple->conv,(size_t)QTP_HC*9);h*=UINT64_C(1099511628211);
    h^=step->ple->previous;h*=UINT64_C(1099511628211);
    h^=step->ple->previous2;return h;
}

static int q38fn_run_target_body(q38fn_decode_step *step, int32_t token,
                                 const float *ngram, const float *embedding_in,
                                 float *output_hyper)
{
    float embedding[Q38FN_HIDDEN];
    const float *embedding_ptr=embedding_in;
    if(!embedding_ptr){if(q38fn_tp_embedding(step->model,token,embedding))return-1;embedding_ptr=embedding;}
    for(int stream=0;stream<Q38FN_HC_COUNT;stream++)
        memcpy(step->hyper+(size_t)stream*Q38FN_HIDDEN,embedding_ptr,(size_t)Q38FN_HIDDEN*sizeof(float));
    for(int layer=0;layer<step->layers;layer++){
        if(layer==Q38FN_PLE_LAYER&&!step->skip_ngram&&
           (q38fn_tp_ple_apply)(step->model,layer,step->ple,(uint64_t)(uint32_t)token,
                              ngram,q38fn_prefill_ple_key,q38fn_prefill_ple_value,step->hyper))return-1;
        int rc=q38fn_layer_is_full_attention((size_t)layer)?
            q38fn_tp_attention_layer(step->model,layer,&step->attention[layer],step->hyper):
            q38fn_tp_linear_layer(step->model,layer,&step->delta[layer],step->hyper);
        if(rc||!finite_vector(step->hyper,QTP_HC))return-1;
    }
    if(step->layers!=Q38FN_LAYERS)return-1;
    memcpy(output_hyper,step->hyper,(size_t)QTP_HC*sizeof(float));return 0;
}
static int q38fn_run_target_token(q38fn_decode_step *step, int32_t token,
                                  const float *ngram, const float *embedding,
                                  int32_t *next,
                                  float *next_logit)
{
    float output_hyper[QTP_HC],hidden[Q38FN_HIDDEN];int id;
    if(q38fn_run_target_body(step,token,ngram,embedding,output_hyper)||
       q38fn_tp_final(step->model,output_hyper,hidden)||
       q38fn_tp_head(step->model,hidden,&id,next_logit))return-1;
    *next=(int32_t)id;return id>=0&&id<Q38FN_VOCAB&&isfinite(*next_logit)?0:-1;
}

static int q38fn_run_mtp_draft(q38fn_tp_model *model,q38fn_tp_mtp_state *state,
                               int32_t token,const float *previous_hyper,
                               float *output_hyper,int32_t *next,float *logit)
{
    return q38fn_tp_mtp_draft(model,state,token,previous_hyper,
                              output_hyper,next,logit);
}

/* Execute a speculative block layer-major.  Candidates remain ordered inside
 * each recurrent/attention layer, preserving exact autoregressive state, but
 * successive candidates now reuse the same layer's resident weight working
 * set and expose a natural boundary for width-aware projection kernels. */
static int q38fn_run_target_window(q38fn_decode_step *step,
                                   const int32_t *tokens, const float *ngram,
                                   uint32_t width, float *output_hyper,
                                   unsigned char *delta_journal, size_t delta_bytes,
                                   unsigned char *ple_journal, size_t ple_bytes)
{
    if(!step||!tokens||!ngram||!output_hyper||
       ((delta_journal==NULL)!=(ple_journal==NULL))||
       width<1||width>Q38FN_SPEC_MAX_WIDTH||step->layers!=Q38FN_LAYERS)return-1;
    /* The recurrent target is exact only when every token completes all
     * layers before the next token starts.  Keep the layer-major scheduler as
     * an explicit kernel-development experiment until its cross-layer state
     * dependencies are fully characterized. */
    if(!getenv("Q38FN_TP_LAYER_MAJOR")){
        for(uint32_t pos=0;pos<width;pos++){
            float *candidate=output_hyper+(size_t)pos*QTP_HC;
            if(q38fn_run_target_body(step,tokens[pos],
                                  ngram+(size_t)pos*Q38FN_HIDDEN,NULL,candidate))return-1;
            for(int layer=0;layer<step->layers;layer++)
                if(delta_journal&&!q38fn_layer_is_full_attention((size_t)layer)&&
                   q38fn_tp_delta_save(&step->delta[layer],
                     delta_journal+((size_t)pos*Q38FN_LAYERS+layer)*delta_bytes,
                     delta_bytes))return-1;
            if(ple_journal&&q38fn_tp_ple_save(step->ple,
               ple_journal+(size_t)pos*ple_bytes,ple_bytes))return-1;
        }
        return 0;
    }
    float embeddings[Q38FN_SPEC_MAX_WIDTH][Q38FN_HIDDEN];
    if(q38fn_tp_embedding_window(step->model,tokens,width,embeddings[0]))return-1;
    for(uint32_t pos=0;pos<width;pos++){
        for(int stream=0;stream<Q38FN_HC_COUNT;stream++)
            memcpy(output_hyper+(size_t)pos*QTP_HC+(size_t)stream*Q38FN_HIDDEN,
                   embeddings[pos],sizeof embeddings[pos]);
    }
    for(int layer=0;layer<step->layers;layer++){
        for(uint32_t pos=0;pos<width;pos++){
            float *candidate=output_hyper+(size_t)pos*QTP_HC;
            if(layer==Q38FN_PLE_LAYER&&!step->skip_ngram&&
               q38fn_tp_ple_apply(step->model,layer,step->ple,
                                  (uint64_t)(uint32_t)tokens[pos],
                                  ngram+(size_t)pos*Q38FN_HIDDEN,candidate))return-1;
            if(ple_journal&&layer==Q38FN_PLE_LAYER&&q38fn_tp_ple_save(step->ple,
               ple_journal+(size_t)pos*ple_bytes,ple_bytes))return-1;
        }
        if(q38fn_layer_is_full_attention((size_t)layer)){
            if(!getenv("Q38FN_TP_ATTN_WINDOW")){
                for(uint32_t pos=0;pos<width;pos++)if(q38fn_tp_attention_layer(
                   step->model,layer,&step->attention[layer],
                   output_hyper+(size_t)pos*QTP_HC))return-1;
            }else if(qtp_attention_layer_window(step->model,layer,
                                                &step->attention[layer],
                                                output_hyper,(int)width))return-1;
        }else for(uint32_t pos=0;pos<width;pos++){
            float *candidate=output_hyper+(size_t)pos*QTP_HC;
            if(q38fn_tp_linear_layer(step->model,layer,
                                     &step->delta[layer],candidate)||
               (delta_journal&&q38fn_tp_delta_save(&step->delta[layer],
                  delta_journal+((size_t)pos*Q38FN_LAYERS+layer)*delta_bytes,
                  delta_bytes)))return-1;
        }
        for(uint32_t pos=0;pos<width;pos++)if(!finite_vector(
           output_hyper+(size_t)pos*QTP_HC,QTP_HC))return-1;
    }
    memcpy(step->hyper,output_hyper+(size_t)(width-1)*QTP_HC,
           (size_t)QTP_HC*sizeof(float));
    return 0;
}

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
    int max_gen=32,max_seq=512,layers=Q38FN_LAYERS,probe=0,probe_token=42,prompt_repeat=1,spec_width=1,spec_adaptive=0;
    for(int a=1;a<argc;a++){
        if(!strcmp(argv[a],"--local-base")&&a+1<argc)local_base=argv[++a];
        else if(!strcmp(argv[a],"--prompt")&&a+1<argc)prompt=argv[++a];
        else if(!strcmp(argv[a],"--max-gen")&&a+1<argc)max_gen=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--max-seq")&&a+1<argc)max_seq=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--prompt-repeat")&&a+1<argc)prompt_repeat=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--layers")&&a+1<argc)layers=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--spec-width")&&a+1<argc)spec_width=atoi(argv[++a]);
        else if(!strcmp(argv[a],"--spec-adaptive"))spec_adaptive=1;
        else if(!strcmp(argv[a],"--probe-token")&&a+1<argc){probe=1;probe_token=atoi(argv[++a]);}
        else if(argv[a][0]!='-'&&!model_dir)model_dir=argv[a];else die("invalid arguments",rank);
    }
    if(!model_dir||ranks!=Q38FN_TP_RANKS||layers<1||layers>Q38FN_LAYERS||prompt_repeat<1||
       (spec_width!=1&&spec_width!=4&&spec_width!=8)||
       (pipeline&&world_ranks/Q38FN_TP_RANKS!=3))die("requires MODEL, configured TP rank count, and valid arguments",world_rank);
    int layer_start=pipeline?stage*(Q38FN_LAYERS/3):0;
    int layer_end=pipeline?(stage+1)*(Q38FN_LAYERS/3):layers;
    FILE*trace=open_trace(world_rank);q38fn_comm comm={0};if(q38fn_comm_init(&comm,rank,ranks))die("communication init failed",world_rank);q38fn_tp_model m;
    int profile=getenv("Q38FN_TP_PROFILE")!=NULL;double layer_s[Q38FN_LAYERS]={0};long layer_calls[Q38FN_LAYERS]={0};
    double load0=now();if(q38fn_tp_model_open(&m,local_base,rank,ranks,q38fn_sum,q38fn_argmax,&comm))die("blob load failed",rank);q38fn_tp_model_set_argmax_n(&m,q38fn_argmax_n);double load_s=now()-load0;
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
    if(!world_rank){char path[4096];snprintf(path,sizeof(path),"%s/tokenizer.json",model_dir);int tok_rc=glm5_bpe_load(path,&tok);if(tok_rc){fprintf(stderr,"q38fn_tp rank=0: tokenizer load failed path=%s rc=%d\n",path,tok_rc);fflush(stderr);die("tokenizer",world_rank);}tok.im_start=248045;tok.im_end=248046;tok.think=248068;tok.end_think=248069;tok.endoftext=248044;size_t plen=strlen(prompt),cap=plen*(size_t)prompt_repeat+(size_t)prompt_repeat+128;char*body=malloc(plen*(size_t)prompt_repeat+(size_t)prompt_repeat),*rendered=malloc(cap);tokens=malloc((size_t)max_seq*4);if(!body||!rendered||!tokens)die("prompt allocation",world_rank);size_t used=0;for(int r=0;r<prompt_repeat;r++){if(r)body[used++]=' ';memcpy(body+used,prompt,plen);used+=plen;}body[used]=0;snprintf(rendered,cap,"<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n",body);free(body);nt=glm5_bpe_encode(&tok,rendered,(int*)tokens,max_seq);free(rendered);if(nt<1||nt+max_gen>max_seq)die("prompt",world_rank);}
    MPI_Bcast(&nt,1,MPI_INT,0,MPI_COMM_WORLD);if(trace){fprintf(trace,"prompt_tokens=%d repeat=%d max_gen=%d\n",nt,prompt_repeat,max_gen);fflush(trace);}uint64_t prev=Q38FN_EOS,prev2=Q38FN_EOS;double start=now(),decode0=0;int generated=0;
    int fapp_decode=0;
    if(spec_width>1){
        if(pipeline)die("speculative decode is not supported with pipeline mode",world_rank);
        q38fn_decode_step step={&m,delta,attention,&ple,hyper,layers,skip_ngram,world_rank};
        q38fn_spec_history history;q38fn_spec_history_init(&history);
        q38fn_tp_mtp_state mtp={0};int mtp_ready=getenv("Q38FN_TP_ENABLE_MTP")&&
            q38fn_tp_blob_find(&m.blob,"mtp.fc_hidden.weight")!=NULL;
        q38fn_tp_model mtp_model=m;
        if(mtp_ready){
            mtp_model.head_logits=malloc(m.head_logits_count*sizeof(*m.head_logits));
            if(!mtp_model.head_logits||q38fn_tp_mtp_init(&mtp,(size_t)max_seq))
                die("MTP state allocation",world_rank);
        }
        float previous_hyper[QTP_HC];int have_previous_hyper=0;
        float ng[Q38FN_SPEC_MAX_WIDTH][Q38FN_HIDDEN];
        int prefill_width=1;const char*pw=getenv("Q38FN_TP_PREFILL_WIDTH");
        if(pw&&!mtp_ready){prefill_width=atoi(pw);if(prefill_width<1||prefill_width>Q38FN_SPEC_MAX_WIDTH)die("invalid prefill width",world_rank);}
        if(prefill_width>1){
            float phyper[Q38FN_SPEC_MAX_WIDTH][QTP_HC];
            int32_t ptokens[Q38FN_SPEC_MAX_WIDTH];
            for(int base=0;base<nt;base+=prefill_width){
                uint32_t width=(uint32_t)(nt-base<prefill_width?nt-base:prefill_width);
                if(!world_rank)memcpy(ptokens,tokens+base,(size_t)width*sizeof(*ptokens));
                MPI_Bcast(ptokens,(int)width,MPI_INT,0,MPI_COMM_WORLD);
                if(skip_ngram)memset(ng,0,(size_t)width*sizeof ng[0]);
                else if(q38fn_tp_ngram_window(&m,ptokens,width,prev,prev2,ng[0]))die("prefill window ngram",world_rank);
                if(q38fn_run_target_window(&step,ptokens,ng[0],width,phyper[0],
                                           NULL,0,NULL,0))die("prefill target window",world_rank);
                for(uint32_t i=0;i<width;i++){
                    if(q38fn_spec_history_append(&history,ptokens[i]))die("history capacity",world_rank);
                    prev2=prev;prev=(uint64_t)(uint32_t)ptokens[i];
                }
                memcpy(previous_hyper,phyper[width-1],sizeof previous_hyper);have_previous_hyper=1;
            }
            float hidden[Q38FN_HIDDEN];int id;
            if(q38fn_tp_final(&m,previous_hyper,hidden)||q38fn_tp_head(&m,hidden,&id,&current_logit))die("prefill window head",world_rank);
            current=(int32_t)id;
        }else {
            /* Embedding lookup is independent across prompt positions.  Batch
             * only this part of the otherwise strictly sequential prefill:
             * the PLE convolution and every decoder state remain in token
             * order, so this does not use the unsafe layer-major scheduler. */
            float *prefill_embeddings=NULL;
            float *prefill_ple_key=NULL,*prefill_ple_value=NULL;
            float *prefill_ngrams=NULL;
            if(!getenv("Q38FN_TP_DISABLE_EMBED_BATCH")){
                prefill_embeddings=malloc((size_t)nt*Q38FN_HIDDEN*sizeof(float));
                prefill_ple_key=malloc((size_t)nt*QTP_HC*sizeof(float));
                prefill_ple_value=malloc((size_t)nt*Q38FN_HIDDEN*sizeof(float));
                int32_t *batch_tokens=malloc((size_t)nt*sizeof(*batch_tokens));
                if(!prefill_embeddings||!prefill_ple_key||!prefill_ple_value||!batch_tokens)die("prefill cache",world_rank);
                if(!world_rank)memcpy(batch_tokens,tokens,(size_t)nt*sizeof(*batch_tokens));
                MPI_Bcast(batch_tokens,nt,MPI_INT,0,MPI_COMM_WORLD);
                for(int base=0;base<nt;base+=Q38FN_SPEC_MAX_WIDTH){
                    uint32_t width=(uint32_t)(nt-base<Q38FN_SPEC_MAX_WIDTH?nt-base:Q38FN_SPEC_MAX_WIDTH);
                    if(q38fn_tp_embedding_window(&m,batch_tokens+base,width,
                                                 prefill_embeddings+(size_t)base*Q38FN_HIDDEN))
                        die("prefill embedding batch",world_rank);
                }
                for(int base=0;base<nt;base+=Q38FN_SPEC_MAX_WIDTH){
                    uint32_t width=(uint32_t)(nt-base<Q38FN_SPEC_MAX_WIDTH?nt-base:Q38FN_SPEC_MAX_WIDTH);
                    if(q38fn_tp_ple_project_window(&m,Q38FN_PLE_LAYER,
                         prefill_embeddings+(size_t)base*Q38FN_HIDDEN,width,
                         prefill_ple_key+(size_t)base*QTP_HC,
                         prefill_ple_value+(size_t)base*Q38FN_HIDDEN))
                        die("prefill PLE projection batch",world_rank);
                }
                if(!skip_ngram){
                    prefill_ngrams=malloc((size_t)nt*Q38FN_HIDDEN*sizeof(float));
                    if(!prefill_ngrams)die("prefill ngram cache",world_rank);
                    uint64_t nprev=Q38FN_EOS,nprev2=Q38FN_EOS;
                    for(int base=0;base<nt;base+=Q38FN_SPEC_MAX_WIDTH){
                        uint32_t width=(uint32_t)(nt-base<Q38FN_SPEC_MAX_WIDTH?nt-base:Q38FN_SPEC_MAX_WIDTH);
                        if(q38fn_tp_ngram_window(&m,batch_tokens+base,width,nprev,nprev2,
                                                 prefill_ngrams+(size_t)base*Q38FN_HIDDEN))
                            die("prefill ngram batch",world_rank);
                        for(uint32_t i=0;i<width;i++){nprev2=nprev;nprev=(uint64_t)(uint32_t)batch_tokens[base+(int)i];}
                    }
                }
                free(batch_tokens);
            }
            for(int pos=0;pos<nt;pos++){
            int32_t token;if(!world_rank)token=tokens[pos];MPI_Bcast(&token,1,MPI_INT,0,MPI_COMM_WORLD);
            if(mtp_ready&&have_previous_hyper){float ignored_hyper[QTP_HC],ignored_logit;int32_t ignored_token;
                uint64_t state_before=getenv("Q38FN_TP_MTP_STATE_CHECK")?q38fn_target_state_checksum(&step):0;
                uint64_t hyper_before=getenv("Q38FN_TP_MTP_STATE_CHECK")?checksum(previous_hyper,QTP_HC):0;
                uint64_t live_before=getenv("Q38FN_TP_MTP_STATE_CHECK")?checksum(hyper,QTP_HC):0;
                uint64_t logits_before=getenv("Q38FN_TP_MTP_STATE_CHECK")?checksum(m.head_logits,m.head_logits_count):0;
                if(q38fn_run_mtp_draft(&mtp_model,&mtp,token,previous_hyper,ignored_hyper,&ignored_token,&ignored_logit))die("MTP prefill",world_rank);
                if(trace&&getenv("Q38FN_TP_MTP_STATE_CHECK")){uint64_t state_after=q38fn_target_state_checksum(&step),hyper_after=checksum(previous_hyper,QTP_HC),live_after=checksum(hyper,QTP_HC),logits_after=checksum(m.head_logits,m.head_logits_count);fprintf(trace,"mtp_state pos=%d state=%016llx/%016llx hyper=%016llx/%016llx live=%016llx/%016llx logits=%016llx/%016llx mode=%d\n",pos,(unsigned long long)state_before,(unsigned long long)state_after,(unsigned long long)hyper_before,(unsigned long long)hyper_after,(unsigned long long)live_before,(unsigned long long)live_after,(unsigned long long)logits_before,(unsigned long long)logits_after,qtp_mtp_mode);fflush(trace);}}
            if(skip_ngram)memset(ng[0],0,sizeof ng[0]);
            else if(prefill_ngrams)memcpy(ng[0],prefill_ngrams+(size_t)pos*Q38FN_HIDDEN,
                                          sizeof ng[0]);
            else if(q38fn_tp_ngram_window(&m,&token,1,prev,prev2,ng[0]))die("prefill ngram",world_rank);
            q38fn_prefill_ple_key=prefill_ple_key?(prefill_ple_key+(size_t)pos*QTP_HC):NULL;
            q38fn_prefill_ple_value=prefill_ple_value?(prefill_ple_value+(size_t)pos*Q38FN_HIDDEN):NULL;
            if(q38fn_run_target_token(&step,token,ng[0],
                                      prefill_embeddings?prefill_embeddings+(size_t)pos*Q38FN_HIDDEN:NULL,
                                      (int32_t*)&current,&current_logit))die("prefill target",world_rank);
            memcpy(previous_hyper,hyper,sizeof previous_hyper);have_previous_hyper=1;
            if(q38fn_spec_history_append(&history,token))die("history capacity",world_rank);
            prev2=prev;prev=(uint64_t)(uint32_t)token;
            }
            q38fn_prefill_ple_key=q38fn_prefill_ple_value=NULL;
            free(prefill_embeddings);
            free(prefill_ple_key);free(prefill_ple_value);
            free(prefill_ngrams);
        }
        decode0=now();
#ifdef Q38FN_HAVE_FAPP
        if(getenv("Q38FN_TP_FAPP")&&fapp_start){fapp_start("decode",0,0);fapp_decode=1;}
#endif
        size_t dsb=q38fn_tp_delta_snapshot_bytes(),psb=q38fn_tp_ple_snapshot_bytes();
        unsigned char*djournal=malloc((size_t)spec_width*Q38FN_LAYERS*dsb);
        unsigned char*pjournal=malloc((size_t)spec_width*psb);
        unsigned char*dbase=malloc((size_t)Q38FN_LAYERS*dsb);
        unsigned char*pbase=malloc(psb);
        if(!djournal||!pjournal||!dbase||!pbase)die("speculative journal allocation",world_rank);
        q38fn_spec_stats spec_stats={0};q38fn_spec_controller spec_controller;q38fn_spec_controller_init(&spec_controller);
        while(generated<max_gen){
            int round_width=spec_adaptive?(int)q38fn_spec_controller_width(&spec_controller,(uint32_t)spec_width):spec_width;
            double proposal0=now();
            q38fn_spec_candidates candidates={0};candidates.width=1;candidates.token[0]=current;candidates.source[0]=Q38FN_SPEC_SOURCE_TARGET;
            q38fn_spec_history draft_history=history;
            if(q38fn_spec_history_append(&draft_history,current))die("draft history capacity",world_rank);
            int32_t drafted[Q38FN_SPEC_MAX_WIDTH-1];
            uint32_t n=q38fn_spec_history_draft(&draft_history,drafted,(uint32_t)round_width-1);
            for(uint32_t i=0;i<n;i++){candidates.token[1+i]=drafted[i];candidates.source[1+i]=Q38FN_SPEC_SOURCE_HISTORY;}
            candidates.width+=n;
            size_t mtp_base=mtp.attention.length;
            if(mtp_ready&&!getenv("Q38FN_TP_MTP_PREFILL_ONLY")){float mtp_in[QTP_HC],mtp_out[QTP_HC];memcpy(mtp_in,previous_hyper,sizeof mtp_in);candidates.width=(uint32_t)round_width;
                for(int i=0;i<round_width;i++){int32_t prediction;float prediction_logit;if(q38fn_run_mtp_draft(&mtp_model,&mtp,candidates.token[i],mtp_in,mtp_out,&prediction,&prediction_logit))die("MTP draft",world_rank);if(i+1<round_width&&(uint32_t)(i+1)>n){candidates.token[i+1]=prediction;candidates.source[i+1]=Q38FN_SPEC_SOURCE_MTP;}memcpy(mtp_in,mtp_out,sizeof mtp_in);}}
            double proposal_s=now()-proposal0,verify0=now();uint64_t base_prev=prev,base_prev2=prev2;
            size_t attn_base[Q38FN_LAYERS];for(int L=0;L<layers;L++)attn_base[L]=attention[L].length;
            if(candidates.width>1){
                for(int L=0;L<layers;L++)if(!q38fn_layer_is_full_attention((size_t)L)&&
                   q38fn_tp_delta_save(&delta[L],dbase+(size_t)L*dsb,dsb))die("delta base snapshot",world_rank);
                if(q38fn_tp_ple_save(&ple,pbase,psb))die("PLE base snapshot",world_rank);
            }
            if(skip_ngram)memset(ng,0,(size_t)candidates.width*sizeof ng[0]);
            else if(q38fn_tp_ngram_window(&m,candidates.token,candidates.width,prev,prev2,ng[0]))die("speculative ngram",world_rank);
            int32_t target[Q38FN_SPEC_MAX_WIDTH];float target_logit[Q38FN_SPEC_MAX_WIDTH];
            float target_hyper[Q38FN_SPEC_MAX_WIDTH][QTP_HC];
            float target_hidden[Q38FN_SPEC_MAX_WIDTH][Q38FN_HIDDEN];
            if(candidates.width==1){
                if(q38fn_run_target_token(&step,candidates.token[0],ng[0],NULL,target,target_logit))
                    die("adaptive scalar target",world_rank);
                memcpy(target_hyper[0],hyper,sizeof target_hyper[0]);
            }else{
                if(q38fn_run_target_window(&step,candidates.token,ng[0],candidates.width,
                                           target_hyper[0],djournal,dsb,pjournal,psb))
                    die("speculative target window",world_rank);
                if(getenv("Q38FN_TP_BATCHED_HEAD")){
                    if(q38fn_tp_final_window(&m,target_hyper[0],candidates.width,target_hidden[0]))die("speculative final mixer",world_rank);
                    if(q38fn_tp_head_window(&m,target_hidden[0],candidates.width,target,target_logit))die("speculative head",world_rank);
                }else for(uint32_t pos=0;pos<candidates.width;pos++){
                    int id=-1;
                    if(q38fn_tp_final(&m,target_hyper[pos],target_hidden[pos])||
                       q38fn_tp_head(&m,target_hidden[pos],&id,target_logit+pos))
                        die("speculative scalar head",world_rank);
                    target[pos]=(int32_t)id;
                }
            }
            double verify_s=now()-verify0;q38fn_spec_acceptance accepted;
            q38fn_spec_accept(&candidates,target,target_logit,&accepted);
            double commit0=now();
            /* Speculative kernels may have hidden scratch/state dependencies.
             * Restore the complete pre-block recurrent state and replay the
             * accepted prefix through the canonical scalar path.  This is the
             * exactness oracle used while each batched component is qualified. */
            uint32_t remain=(uint32_t)(max_gen-generated),canonical=0;
            int32_t replay_next;float replay_logit;
            if(candidates.width==1){
                canonical=remain?1:0;replay_next=target[0];replay_logit=target_logit[0];
            }else{
                for(int L=0;L<layers;L++)if(q38fn_layer_is_full_attention((size_t)L))
                    attention[L].length=attn_base[L];
                else if(q38fn_tp_delta_restore(&delta[L],dbase+(size_t)L*dsb,dsb))
                    die("delta base restore",world_rank);
                if(q38fn_tp_ple_restore(&ple,pbase,psb))die("PLE base restore",world_rank);
                replay_next=current;replay_logit=current_logit;
                uint64_t replay_prev=base_prev,replay_prev2=base_prev2;
                while(canonical<candidates.width&&canonical<remain){
                    int32_t token=candidates.token[canonical];float replay_ng[Q38FN_HIDDEN];
                    if(skip_ngram)memset(replay_ng,0,sizeof replay_ng);
                    else if(q38fn_tp_ngram(&m,(uint64_t)(uint32_t)token,
                                           replay_prev,replay_prev2,replay_ng))
                        die("canonical replay ngram",world_rank);
                    if(q38fn_run_target_token(&step,token,replay_ng,NULL,&replay_next,&replay_logit))
                        die("canonical replay target",world_rank);
                    target[canonical]=replay_next;target_logit[canonical]=replay_logit;
                    canonical++;
                    replay_prev2=replay_prev;replay_prev=(uint64_t)(uint32_t)token;
                    if(!getenv("Q38FN_TP_IGNORE_EOS")&&
                       (token==Q38FN_EOS||token==248046))break;
                    if(canonical<candidates.width&&candidates.token[canonical]!=replay_next)break;
                }
            }
            accepted.committed=canonical;accepted.carry=replay_next;
            accepted.carry_logit=replay_logit;
            if(mtp_ready)mtp.attention.length=mtp_base+canonical;
            memcpy(previous_hyper,hyper,sizeof previous_hyper);
            prev=base_prev;prev2=base_prev2;
            int stop=0;
            for(uint32_t i=0;i<accepted.committed;i++){
                int32_t token=candidates.token[i];
                if(!world_rank){char piece[4096];int decoded=glm5_bpe_decode_token(&tok,token,piece,sizeof(piece));if(decoded>=0){fputs(piece,stdout);fflush(stdout);}if(trace){fprintf(trace,"token pos=%d id=%d logit=%.9g source=%u accepted=%u/%u piece=",generated,token,i?target_logit[i-1]:current_logit,candidates.source[i],accepted.committed,candidates.width);if(decoded>=0)fputs(piece,trace);fputc('\n',trace);fflush(trace);}}
                if(q38fn_spec_history_append(&history,token))die("history capacity",world_rank);
                prev2=prev;prev=(uint64_t)(uint32_t)token;generated++;
                if(!getenv("Q38FN_TP_IGNORE_EOS")&&(token==Q38FN_EOS||token==248046)){stop=1;break;}
            }
            current=accepted.carry;current_logit=accepted.carry_logit;
            double commit_s=now()-commit0;q38fn_spec_stats_record(&spec_stats,&candidates,&accepted,proposal_s,verify_s,commit_s);
            if(spec_adaptive)q38fn_spec_controller_record(&spec_controller,candidates.width,accepted.committed,proposal_s+verify_s+commit_s);
            if(stop)break;
        }
        if(!world_rank){fprintf(stderr,"\nQ38FN_SPEC width=%d adaptive=%d rounds=%llu committed=%llu mean_accept=%.3f proposal=%.6f verify=%.6f commit=%.6f history=%llu mtp=%llu histogram=",spec_width,spec_adaptive,(unsigned long long)spec_stats.rounds,(unsigned long long)spec_stats.committed,spec_stats.rounds?(double)spec_stats.committed/spec_stats.rounds:0.0,spec_stats.proposal_seconds,spec_stats.verify_seconds,spec_stats.commit_seconds,(unsigned long long)spec_stats.history_proposed,(unsigned long long)spec_stats.mtp_proposed);for(int i=0;i<=spec_width;i++)fprintf(stderr,"%s%d:%llu",i?",":"",i,(unsigned long long)spec_stats.histogram[i]);fputc('\n',stderr);}
        free(pbase);free(dbase);free(pjournal);free(djournal);if(mtp_ready){q38fn_tp_mtp_close(&mtp);free(mtp_model.head_logits);}goto decode_done;
    }
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
decode_done:
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
