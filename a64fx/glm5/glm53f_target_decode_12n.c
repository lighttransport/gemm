/* Persistent real-weight 45-layer GLM-5.3F greedy target decode. */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/glm53f_safetensors.h"
#include "glm53f_dense_ffn_12n.h"
#include "glm53f_embedding_12n.h"
#include "glm53f_kda_12n.h"
#include "glm53f_moe_stage_12n.h"
#include "glm53f_sparse_12n.h"
#include "glm53f_target_head_12n.h"
#include "glm53f_target_layer_12n.h"

enum { LAYERS = 45, HIDDEN = 4096, STREAMS = 4, FLAT = 16384, MIX = 24 };

static void *a256(size_t bytes) {
    void *p = NULL;
    return posix_memalign(&p, 256, bytes) ? NULL : p;
}

static void *load_exact(glm53f_st_context *st, const char *name, size_t bytes) {
    const st_tensor_info *tensor = glm53f_st_find(st, name, NULL);
    void *p = a256(bytes);
    if (!tensor || tensor->nbytes != bytes || !p ||
        glm53f_st_read(st, name, 0, p, bytes)) {
        fprintf(stderr, "target stack load failed: %s\n", name);
        free(p);
        return NULL;
    }
    return p;
}

static int load_layer(glm53f_st_context *st, int layer,
                      glm53f_target_layer_weights_12n *w) {
    char name[256];
    memset(w, 0, sizeof(*w));
#define SITE(FIELD, WHICH) do { \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_fn",layer); \
    w->FIELD.fn=load_exact(st,name,(size_t)MIX*FLAT*sizeof(uint16_t)); \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_base",layer); \
    w->FIELD.base=load_exact(st,name,MIX*sizeof(float)); \
    snprintf(name,sizeof(name),"model.language_model.layers.%d.hc_" WHICH "_scale",layer); \
    w->FIELD.scale=load_exact(st,name,3*sizeof(float)); \
} while(0)
    SITE(attention_mhc,"attn"); SITE(ffn_mhc,"ffn");
#undef SITE
    snprintf(name,sizeof(name),"model.language_model.layers.%d.input_layernorm.weight",layer);
    w->input_norm=load_exact(st,name,HIDDEN*sizeof(uint16_t));
    snprintf(name,sizeof(name),"model.language_model.layers.%d.post_attention_layernorm.weight",layer);
    w->post_attention_norm=load_exact(st,name,HIDDEN*sizeof(uint16_t));
    return w->attention_mhc.fn&&w->attention_mhc.base&&w->attention_mhc.scale&&
           w->ffn_mhc.fn&&w->ffn_mhc.base&&w->ffn_mhc.scale&&
           w->input_norm&&w->post_attention_norm?0:-1;
}

int main(int argc, char **argv) {
    int rank, ranks, token, steps, capacity;
    glm53f_st_context *st;
    glm53f_target_layer_weights_12n layer_weight[LAYERS];
    glm53f_kda_context_12n *kda[LAYERS] = {0};
    glm53f_sparse_context_12n *sparse[LAYERS] = {0};
    glm53f_dense_ffn_context_12n *dense[3] = {0};
    glm53f_moe_stage_context_12n *moe;
    glm53f_embedding_context_12n *embedding;
    glm53f_target_head_context_12n *head;
    glm53f_target_layer_scratch_12n *scratch;
    float *streams;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 4 || ranks != 12) {
        if (!rank) fprintf(stderr,"usage: %s MODEL ROUTED_STAGE SHARED_STAGE [token=1] [steps=1]\n",argv[0]);
        MPI_Abort(MPI_COMM_WORLD,2);
    }
    token=argc>4?atoi(argv[4]):1;steps=argc>5?atoi(argv[5]):1;
    if(token<0||token>=154880||steps<1||steps>32)MPI_Abort(MPI_COMM_WORLD,2);
    capacity=steps;
    st=glm53f_st_open(argv[1]);if(!st)MPI_Abort(MPI_COMM_WORLD,2);
    for(int l=0;l<LAYERS;l++)if(load_layer(st,l,&layer_weight[l]))MPI_Abort(MPI_COMM_WORLD,2);
    glm53f_st_close(st);
    embedding=glm53f_embedding_create_12n(argv[1]);
    head=glm53f_target_head_create_12n(argv[1]);
    for(int l=0;l<LAYERS;l++){
        if(l%4==3)sparse[l]=glm53f_sparse_create_12n(argv[1],l,capacity);
        else kda[l]=glm53f_kda_create_12n(argv[1],l);
    }
    for(int l=0;l<3;l++)dense[l]=glm53f_dense_ffn_create_12n(argv[1],l);
    moe=glm53f_moe_stage_create_12n(argv[2],argv[3],argv[1],3,42);
    scratch=a256(sizeof(*scratch));streams=a256((size_t)FLAT*sizeof(float));
    if(!embedding||!head||!moe||!scratch||!streams)MPI_Abort(MPI_COMM_WORLD,2);
    MPI_Barrier(MPI_COMM_WORLD);double begin=MPI_Wtime();
    for(int step=0;step<steps;step++){
        if(glm53f_embedding_streams_12n(embedding,token,streams))MPI_Abort(MPI_COMM_WORLD,2);
        for(int l=0;l<LAYERS;l++){
            glm53f_target_sublayer_12n attention=kda[l]?glm53f_kda_sublayer_12n:glm53f_sparse_sublayer_12n;
            void*attention_context=kda[l]?(void*)kda[l]:(void*)sparse[l];
            glm53f_target_sublayer_12n ffn;
            void*ffn_context;
            if(l<3){ffn=glm53f_dense_ffn_sublayer_12n;ffn_context=dense[l];}
            else{glm53f_moe_stage_set_layer_12n(moe,l);ffn=glm53f_moe_stage_sublayer_12n;ffn_context=moe;}
            if(glm53f_target_layer_forward_12n(streams,&layer_weight[l],attention,
                    attention_context,ffn,ffn_context,scratch))MPI_Abort(MPI_COMM_WORLD,2);
        }
        float value;
        if(glm53f_target_head_argmax_12n(head,streams,&token,&value))MPI_Abort(MPI_COMM_WORLD,2);
        if(!rank)printf("GLM53F_TARGET_TOKEN step=%d token=%d logit=%.9g\n",step,token,value);
    }
    double elapsed=MPI_Wtime()-begin,max_elapsed;MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank)printf("GLM53F_TARGET_DECODE_12N steps=%d ms_tok=%.3f tok_s=%.3f final_token=%d PASS\n",steps,max_elapsed*1e3/steps,steps/max_elapsed,token);
    glm53f_moe_stage_free_12n(moe);glm53f_target_head_free_12n(head);glm53f_embedding_free_12n(embedding);
    MPI_Finalize();return 0;
}
