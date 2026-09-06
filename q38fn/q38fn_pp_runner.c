/* Twelve-rank Qwen3.8-Flash-Next pipeline decoder. */
#define _GNU_SOURCE
#include <mpi.h>
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_arch.h"
#define Q38FN_RUNTIME_IMPLEMENTATION
#include "../common/q38fn_runtime.h"
#define GLM5_BPE_IMPLEMENTATION
#include "../common/glm5_bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

static double seconds(void)
{
    struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec + now.tv_nsec * 1.0e-9;
}

static void fail(const char *message, int rank)
{
    fprintf(stderr, "q38fn_pp rank=%d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const char *model = NULL, *prompt = "Write a C11 function that returns the nth Fibonacci number.";
    const char *local_base = "/local/u14346/q38fn";
    int max_gen = 32, max_seq = 512, rank, ranks;
    glm53f_st_context *ctx;
    glm5_bpe tokenizer = {0};
    int32_t *tokens = NULL; int token_count = 0, current = -1;
    float *hyper;
    q38fn_delta_state delta[4] = {{0}};
    q38fn_attention_state attention[4] = {{0}};
    q38fn_ple_state ple = {0};
    double ngram_seconds=0.0,receive_seconds=0.0,layer_seconds=0.0;
    double send_seconds=0.0,endpoint_seconds=0.0;

    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    {
        const char *trace_dir = getenv("Q38FN_TRACE_DIR");
        if (trace_dir) {
            char trace_path[4096];
            snprintf(trace_path, sizeof(trace_path), "%s/preload-rank-%02d.log", trace_dir, rank);
            if (!freopen(trace_path, "w", stderr)) fail("trace log open failed", rank);
            setvbuf(stderr, NULL, _IONBF, 0);
            if (rank == 0) {
                snprintf(trace_path, sizeof(trace_path), "%s/generated.txt", trace_dir);
                if (!freopen(trace_path, "w", stdout)) fail("generation log open failed", rank);
                setvbuf(stdout, NULL, _IONBF, 0);
            }
            fprintf(stderr, "Q38FN_START rank=%d ranks=%d\n", rank, ranks);
        }
    }
    for (int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"--prompt")&&i+1<argc)prompt=argv[++i];
        else if(!strcmp(argv[i],"--max-gen")&&i+1<argc)max_gen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-seq")&&i+1<argc)max_seq=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--local-base")&&i+1<argc)local_base=argv[++i];
        else if(argv[i][0]!='-'&&!model)model=argv[i]; else fail("invalid arguments",rank);
    }
    if(!model||ranks!=12||max_gen<1||max_seq<2)fail("requires MODEL and exactly 12 ranks",rank);
    {unsigned long mask=0xf0UL;(void)syscall(SYS_set_mempolicy,3,&mask,8UL);}
    char payload[4096];snprintf(payload,sizeof(payload),"%s/rank-%02d",local_base,rank);
    setenv("GLM53F_ST_PAYLOAD_DIR",payload,1);
    ctx=glm53f_st_open(model); if(!ctx)fail("checkpoint open failed",rank);
    hyper=(float*)malloc((size_t)Q38FN_HC_COUNT*Q38FN_HIDDEN*sizeof(*hyper));
    if(!hyper)fail("activation allocation failed",rank);
    int layer0=rank*4;
    for(int local=0;local<4;++local){int layer=layer0+local;if(q38fn_layer_is_full_attention((size_t)layer)){if(q38fn_attention_state_init(&attention[local],(size_t)max_seq)!=0)fail("KV allocation failed",rank);}else if(q38fn_delta_state_init(&delta[local])!=0)fail("delta allocation failed",rank);}
    if(rank==0&&q38fn_ple_state_init(&ple)!=0)fail("PLE state allocation failed",rank);
    double preload_start=seconds();
    for(int local=0;local<4;++local){if(q38fn_preload_layer(ctx,layer0+local)!=0)fail("layer preload failed",rank);fprintf(stderr,"Q38FN_PRELOAD_PROGRESS rank=%d layer=%d bytes=%zu\n",rank,layer0+local,q38fn_cached_weight_bytes());}
    if(q38fn_preload_ngram_owner(ctx,rank,ranks)!=0)fail("ngram preload failed",rank);
    fprintf(stderr,"Q38FN_PRELOAD_PROGRESS rank=%d ngram=resident bytes=%zu\n",rank,q38fn_cached_weight_bytes());
    /* The full input embedding is 1.27 GB but decode touches one 5 KiB row.
     * Leave it in the node-local payload and let the row cache retain only
     * prompt/generated rows; rank 0 otherwise crosses the A64FX HBM limit. */
    if(rank==ranks-1){
        const char*finals[]={"model.language_model.hyper_connection_mixer.hc_norm.weight","model.language_model.hyper_connection_mixer.input_mix_weight_down.weight","model.language_model.hyper_connection_mixer.input_mix_weight_up.weight","lm_head.weight"};
        for(size_t i=0;i<sizeof(finals)/sizeof(finals[0]);++i)if(q38fn_preload_tensor(ctx,finals[i])!=0)fail("final preload failed",rank);
    }
    fprintf(stderr,"Q38FN_PRELOAD rank=%d layers=%d-%d bytes=%zu seconds=%.3f payload=%s\n",rank,layer0,layer0+4,q38fn_cached_weight_bytes(),seconds()-preload_start,payload);
    if(rank==0){
        char path[4096]; snprintf(path,sizeof(path),"%s/tokenizer.json",model);
        if(glm5_bpe_load(path,&tokenizer)!=0)fail("tokenizer load failed",rank);
        tokenizer.im_start=248045;tokenizer.im_end=248046;tokenizer.think=248068;
        tokenizer.end_think=248069;tokenizer.endoftext=248044;
        size_t cap=strlen(prompt)+128;char*rendered=(char*)malloc(cap);
        tokens=(int32_t*)malloc((size_t)max_seq*sizeof(*tokens));
        snprintf(rendered,cap,"<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n",prompt);
        token_count=glm5_bpe_encode(&tokenizer,rendered,(int*)tokens,max_seq);free(rendered);
        if(token_count<1||token_count+max_gen>max_seq)fail("prompt too long",rank);
        fprintf(stderr,"q38fn_pp prompt_tokens=%d max_gen=%d\n",token_count,max_gen);
    }
    MPI_Bcast(&token_count,1,MPI_INT,0,MPI_COMM_WORLD);
    double start=seconds(),decode_start=0.0; int generated=0;uint64_t previous=Q38FN_EOS,previous2=Q38FN_EOS;
    for(int position=0;position<token_count+max_gen;++position){
        int token;
        if(position<token_count){if(rank==0)token=tokens[position];MPI_Bcast(&token,1,MPI_INT,0,MPI_COMM_WORLD);}else token=current;
        if(position==token_count)decode_start=seconds();
        if(position>=token_count&&rank==0){char piece[4096];if(glm5_bpe_decode_token(&tokenizer,token,piece,sizeof(piece))>=0)fputs(piece,stdout);else printf("<%d>",token);fflush(stdout);generated++;}
        if(position>=token_count&&(token==Q38FN_EOS||token==248046))break;
        float local_ngram[Q38FN_HIDDEN]={0},ngram[Q38FN_HIDDEN];uint64_t ngram_rows[Q38FN_NGRAM_HEADS];
        q38fn_ngram_rows((uint64_t)token,previous,previous2,ngram_rows);
        for(int head=0;head<Q38FN_NGRAM_HEADS;++head){uint64_t shard=ngram_rows[head]/Q38FN_NGRAM_ROWS_PER_SHARD;uint64_t row=ngram_rows[head]%Q38FN_NGRAM_ROWS_PER_SHARD;char name[224];snprintf(name,sizeof(name),"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%llu.weight",(unsigned long long)shard);if(q38fn_ngram_owner((int)shard)==rank&&q38fn_read_bf16_row(ctx,name,(size_t)row,Q38FN_NGRAM_HEAD_DIM,local_ngram+head*Q38FN_NGRAM_HEAD_DIM)!=0)fail("local ngram read failed",rank);}
        double phase=seconds();
        MPI_Allreduce(local_ngram,ngram,Q38FN_HIDDEN,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
        ngram_seconds+=seconds()-phase;phase=seconds();
        if(rank==0){
            float embedding[Q38FN_HIDDEN];
            if(q38fn_read_bf16_row(ctx,"model.language_model.embed_tokens.weight",(size_t)token,Q38FN_HIDDEN,embedding)!=0)fail("embedding read failed",rank);
            for(int s=0;s<Q38FN_HC_COUNT;++s)memcpy(hyper+s*Q38FN_HIDDEN,embedding,sizeof(embedding));
        }else MPI_Recv(hyper,Q38FN_HC_COUNT*Q38FN_HIDDEN,MPI_FLOAT,rank-1,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        receive_seconds+=seconds()-phase;phase=seconds();
        for(int local=0;local<4;++local){
            int layer=layer0+local, experts[Q38FN_ACTIVE_EXPERTS];
            if(layer==Q38FN_PLE_LAYER&&q38fn_ple_apply_embedding(ctx,layer,&ple,(uint64_t)token,ngram,hyper)!=0)fail("PLE step failed",rank);
            if(q38fn_layer_is_full_attention((size_t)layer)){
                if(q38fn_attention_layer_step(ctx,layer,&attention[local],hyper,experts)!=0)fail("attention layer failed",rank);
            }else if(q38fn_linear_layer_step(ctx,layer,&delta[local],hyper,experts)!=0)fail("linear layer failed",rank);
        }
        layer_seconds+=seconds()-phase;phase=seconds();
        if(rank<ranks-1)MPI_Send(hyper,Q38FN_HC_COUNT*Q38FN_HIDDEN,MPI_FLOAT,rank+1,100,MPI_COMM_WORLD);
        send_seconds+=seconds()-phase;phase=seconds();
        if(rank==ranks-1){float hidden[Q38FN_HIDDEN],logit;if(q38fn_final_mix(ctx,hyper,hidden)!=0||q38fn_lm_head_argmax(ctx,hidden,&current,&logit)!=0)fail("logit projection failed",rank);}
        endpoint_seconds+=seconds()-phase;
        MPI_Bcast(&current,1,MPI_INT,ranks-1,MPI_COMM_WORLD);
        previous2=previous;previous=(uint64_t)token;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    {
        double timing[5]={ngram_seconds,receive_seconds,layer_seconds,send_seconds,endpoint_seconds};
        double all_timings[12][5];
        MPI_Gather(timing,5,MPI_DOUBLE,all_timings,5,MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(rank==0){double end=seconds();putchar('\n');fprintf(stderr,"q38fn_pp generated=%d elapsed=%.3f prefill=%.3f decode=%.3f decode_tok_s=%.6f\n",generated,end-start,decode_start-start,end-decode_start,generated/(end-decode_start));
            for(int r=0;r<ranks;++r)fprintf(stderr,"Q38FN_PP_TIMING rank=%d ngram=%.6f recv=%.6f layers=%.6f send=%.6f endpoint=%.6f\n",r,all_timings[r][0],all_timings[r][1],all_timings[r][2],all_timings[r][3],all_timings[r][4]);}
    }
    for(int local=0;local<4;++local){q38fn_delta_state_destroy(&delta[local]);q38fn_attention_state_destroy(&attention[local]);}
    q38fn_ple_state_destroy(&ple);free(hyper);free(tokens);glm5_bpe_free(&tokenizer);glm53f_st_close(ctx);MPI_Finalize();return 0;
}
