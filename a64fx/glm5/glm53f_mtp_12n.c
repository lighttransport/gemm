#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef GLM53F_EXTERNAL_ST_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#endif
#include <arm_sve.h>
#include <mpi.h>
#include <omp.h>
#ifndef __ARM_FEATURE_SVE
#define __ARM_FEATURE_SVE 1
#endif
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include "glm53f_embedding_12n.h"
#include "glm53f_expert_kern.h"
#include "glm53f_moe_stage_12n.h"
#include "glm53f_mtp_12n.h"
#include "glm53f_sparse_12n.h"
#include "glm53f_target_head_12n.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MTP_H = 4096, MTP_STREAMS = 4 };

struct glm53f_mtp_context_12n {
    int rank, ranks, row0, rows;
    uint16_t *enorm, *hnorm, *input_norm, *post_norm, *eh;
    glm53f_embedding_context_12n *embedding;
    glm53f_sparse_context_12n *attention;
    glm53f_moe_stage_context_12n *moe;
    glm53f_target_head_context_12n *head;
    float *embed_streams, *pair, *fusion, *normalized, *sublayer, *head_streams;
};

static void *mtp_a256(size_t n){void*p=NULL;return posix_memalign(&p,256,n)?NULL:p;}
static void mtp_norm(float*out,const float*x,const uint16_t*w){double ss=0;
#pragma omp parallel for reduction(+:ss)
    for(int i=0;i<MTP_H;i++)ss+=(double)x[i]*x[i];float inv=1.0f/sqrtf((float)(ss/MTP_H)+1e-5f);
#pragma omp parallel for schedule(static)
    for(int i=0;i<MTP_H;i++)out[i]=x[i]*inv*glm53f_bf16_to_f32(w[i]);}

glm53f_mtp_context_12n*glm53f_mtp_create_12n(const char*model,const char*routed,const char*shared,int capacity){glm53f_mtp_context_12n*c=calloc(1,sizeof(*c));glm53f_st_context*st;char name[128];if(!c)return NULL;MPI_Comm_rank(MPI_COMM_WORLD,&c->rank);MPI_Comm_size(MPI_COMM_WORLD,&c->ranks);if(c->ranks!=12||capacity<1)goto fail;c->row0=(int)((long long)MTP_H*c->rank/c->ranks);c->rows=(int)((long long)MTP_H*(c->rank+1)/c->ranks)-c->row0;st=glm53f_st_open(model);if(!st)goto fail;
#define ALLOC_READ(F,N,Z,O) do{c->F=mtp_a256(Z);if(!c->F||glm53f_st_read(st,N,O,c->F,Z))goto fail_st;}while(0)
    ALLOC_READ(enorm,"model.language_model.layers.45.enorm.weight",MTP_H*2,0);ALLOC_READ(hnorm,"model.language_model.layers.45.hnorm.weight",MTP_H*2,0);ALLOC_READ(input_norm,"model.language_model.layers.45.input_layernorm.weight",MTP_H*2,0);ALLOC_READ(post_norm,"model.language_model.layers.45.post_attention_layernorm.weight",MTP_H*2,0);snprintf(name,sizeof(name),"model.language_model.layers.45.eh_proj.weight");ALLOC_READ(eh,name,(size_t)c->rows*2*MTP_H*2,(size_t)c->row0*2*MTP_H*2);
#undef ALLOC_READ
    glm53f_st_close(st);c->embedding=glm53f_embedding_create_12n(model);c->attention=glm53f_sparse_create_12n(model,45,capacity);c->moe=glm53f_moe_stage_create_12n(routed,shared,model,45,1);c->head=glm53f_target_head_create_with_norm_12n(model,"model.language_model.layers.45.shared_head.norm.weight");c->embed_streams=mtp_a256((size_t)MTP_STREAMS*MTP_H*4);c->pair=mtp_a256((size_t)2*MTP_H*4);c->fusion=mtp_a256(MTP_H*4);c->normalized=mtp_a256(MTP_H*4);c->sublayer=mtp_a256(MTP_H*4);c->head_streams=mtp_a256((size_t)MTP_STREAMS*MTP_H*4);if(!c->embedding||!c->attention||!c->moe||!c->head||!c->embed_streams||!c->pair||!c->fusion||!c->normalized||!c->sublayer||!c->head_streams)goto fail;return c;
fail_st:glm53f_st_close(st);fail:glm53f_mtp_free_12n(c);return NULL;}

int glm53f_mtp_forward_12n(glm53f_mtp_context_12n*c,int token,const float*hidden,int*draft,float*logit,float*draft_hidden){if(!c||!hidden||!draft||!logit)return-1;if(glm53f_embedding_streams_12n(c->embedding,token,c->embed_streams))return-1;mtp_norm(c->pair,c->embed_streams,c->enorm);mtp_norm(c->pair+MTP_H,hidden,c->hnorm);
#pragma omp parallel for schedule(static)
    for(int r=0;r<c->rows;r++)c->fusion[c->row0+r]=glm53f_dot_bf16_sve(c->eh+(size_t)r*2*MTP_H,c->pair,2*MTP_H);int counts[12],displs[12];for(int r=0;r<c->ranks;r++){displs[r]=(int)((long long)MTP_H*r/c->ranks);counts[r]=(int)((long long)MTP_H*(r+1)/c->ranks)-displs[r];}if(MPI_Allgatherv(MPI_IN_PLACE,0,MPI_FLOAT,c->fusion,counts,displs,MPI_FLOAT,MPI_COMM_WORLD)!=MPI_SUCCESS)return-1;memcpy(c->head_streams,c->fusion,MTP_H*4);mtp_norm(c->normalized,c->fusion,c->input_norm);if(glm53f_sparse_sublayer_12n(c->attention,c->sublayer,c->normalized))return-1;
#pragma omp parallel for schedule(static)
    for(int i=0;i<MTP_H;i++)c->fusion[i]+=c->sublayer[i];mtp_norm(c->normalized,c->fusion,c->post_norm);glm53f_moe_stage_set_layer_12n(c->moe,45);if(glm53f_moe_stage_sublayer_12n(c->moe,c->sublayer,c->normalized))return-1;
#pragma omp parallel for schedule(static)
    for(int i=0;i<MTP_H;i++)c->fusion[i]+=c->sublayer[i];if(draft_hidden)memcpy(draft_hidden,c->fusion,MTP_H*4);for(int s=0;s<MTP_STREAMS;s++)memcpy(c->head_streams+(size_t)s*MTP_H,c->fusion,MTP_H*4);return glm53f_target_head_argmax_12n(c->head,c->head_streams,draft,logit);}
int glm53f_mtp_length_12n(const glm53f_mtp_context_12n*c){return c?glm53f_sparse_length_12n(c->attention):-1;}
int glm53f_mtp_restore_length_12n(glm53f_mtp_context_12n*c,int length){return c?glm53f_sparse_restore_length_12n(c->attention,length):-1;}
void glm53f_mtp_free_12n(glm53f_mtp_context_12n*c){if(!c)return;free(c->head_streams);free(c->sublayer);free(c->normalized);free(c->fusion);free(c->pair);free(c->embed_streams);glm53f_target_head_free_12n(c->head);glm53f_moe_stage_free_12n(c->moe);glm53f_sparse_free_12n(c->attention);glm53f_embedding_free_12n(c->embedding);free(c->eh);free(c->post_norm);free(c->input_norm);free(c->hnorm);free(c->enorm);free(c);}
