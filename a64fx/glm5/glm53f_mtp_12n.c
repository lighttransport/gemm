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

enum { MTP_H = 4096, MTP_STREAMS = 4, MTP_PRIME_TILE = 64 };

struct glm53f_mtp_context_12n {
    int rank, ranks, row0, rows, capacity;
    uint16_t *enorm, *hnorm, *input_norm, *post_norm, *eh;
    glm53f_embedding_context_12n *embedding;
    glm53f_sparse_context_12n *attention;
    glm53f_moe_stage_context_12n *moe;
    glm53f_target_head_context_12n *head;
    float *embed_streams, *pair, *fusion, *normalized, *sublayer, *head_streams;
    float *prime_storage;
};

static void *mtp_a256(size_t n){void*p=NULL;return posix_memalign(&p,256,n)?NULL:p;}
static void mtp_norm(float*out,const float*x,const uint16_t*w){double ss=0;
#pragma omp parallel for reduction(+:ss)
    for(int i=0;i<MTP_H;i++)ss+=(double)x[i]*x[i];float inv=1.0f/sqrtf((float)(ss/MTP_H)+1e-5f);
#pragma omp parallel for schedule(static)
    for(int i=0;i<MTP_H;i++)out[i]=x[i]*inv*glm53f_bf16_to_f32(w[i]);}

glm53f_mtp_context_12n*glm53f_mtp_create_12n(const char*model,const char*routed,const char*shared,int capacity){glm53f_mtp_context_12n*c=calloc(1,sizeof(*c));glm53f_st_context*st;char name[128];if(!c)return NULL;MPI_Comm_rank(MPI_COMM_WORLD,&c->rank);MPI_Comm_size(MPI_COMM_WORLD,&c->ranks);if(c->ranks!=12||capacity<1)goto fail;c->capacity=capacity;c->row0=(int)((long long)MTP_H*c->rank/c->ranks);c->rows=(int)((long long)MTP_H*(c->rank+1)/c->ranks)-c->row0;st=glm53f_st_open(model);if(!st)goto fail;
#define ALLOC_READ(F,N,Z,O) do { \
    c->F = mtp_a256(Z); \
    if (!c->F) { \
        fprintf(stderr, "GLM53F_MTP_CREATE_FAIL rank=%d phase=allocation tensor=%s bytes=%zu\n", \
                c->rank, N, (size_t)(Z)); \
        goto fail_st; \
    } \
    int read_rc = glm53f_st_read(st, N, O, c->F, Z); \
    if (read_rc) { \
        fprintf(stderr, "GLM53F_MTP_CREATE_FAIL rank=%d phase=read tensor=%s offset=%zu bytes=%zu rc=%d\n", \
                c->rank, N, (size_t)(O), (size_t)(Z), read_rc); \
        goto fail_st; \
    } \
} while (0)
    ALLOC_READ(enorm,"model.language_model.layers.45.enorm.weight",MTP_H*2,0);ALLOC_READ(hnorm,"model.language_model.layers.45.hnorm.weight",MTP_H*2,0);ALLOC_READ(input_norm,"model.language_model.layers.45.input_layernorm.weight",MTP_H*2,0);ALLOC_READ(post_norm,"model.language_model.layers.45.post_attention_layernorm.weight",MTP_H*2,0);snprintf(name,sizeof(name),"model.language_model.layers.45.eh_proj.weight");ALLOC_READ(eh,name,(size_t)c->rows*2*MTP_H*2,(size_t)c->row0*2*MTP_H*2);
#undef ALLOC_READ
    glm53f_st_close(st);c->embedding=glm53f_embedding_create_12n(model);c->attention=glm53f_sparse_create_12n(model,45,capacity);c->moe=glm53f_moe_stage_create_12n(routed,shared,model,45,1);c->head=glm53f_target_head_create_with_norm_12n(model,"model.language_model.layers.45.shared_head.norm.weight");c->embed_streams=mtp_a256((size_t)MTP_STREAMS*MTP_H*4);c->pair=mtp_a256((size_t)2*MTP_H*4);c->fusion=mtp_a256(MTP_H*4);c->normalized=mtp_a256(MTP_H*4);c->sublayer=mtp_a256(MTP_H*4);c->head_streams=mtp_a256((size_t)MTP_STREAMS*MTP_H*4);if(!c->embedding||!c->attention||!c->moe||!c->head||!c->embed_streams||!c->pair||!c->fusion||!c->normalized||!c->sublayer||!c->head_streams)goto fail;return c;
fail_st:glm53f_st_close(st);
fail:
    fprintf(stderr, "GLM53F_MTP_CREATE_FAIL rank=%d capacity=%d ranks=%d enorm=%d hnorm=%d input_norm=%d post_norm=%d eh=%d embedding=%d attention=%d moe=%d head=%d buffers=%d/%d/%d/%d/%d/%d\n",
            c->rank, capacity, c->ranks, !!c->enorm, !!c->hnorm, !!c->input_norm,
            !!c->post_norm, !!c->eh, !!c->embedding, !!c->attention, !!c->moe,
            !!c->head, !!c->embed_streams, !!c->pair, !!c->fusion,
            !!c->normalized, !!c->sublayer, !!c->head_streams);
    glm53f_mtp_free_12n(c);
    return NULL;
}

static int mtp_run(glm53f_mtp_context_12n*c,int token,const float*hidden,int*draft,float*logit,float*draft_hidden,int cache_only){if(!c||!hidden||(!cache_only&&(!draft||!logit)))return-1;if(glm53f_embedding_streams_12n(c->embedding,token,c->embed_streams))return-1;
    /* Training shifts the token embedding by one position.  There is no
     * predecessor at MTP position zero, so the checkpoint contract masks that
     * embedding before enorm (the previous target hidden state is retained). */
    if(glm53f_sparse_length_12n(c->attention)==0)
        memset(c->embed_streams,0,(size_t)MTP_STREAMS*MTP_H*4);
    mtp_norm(c->pair,c->embed_streams,c->enorm);mtp_norm(c->pair+MTP_H,hidden,c->hnorm);
#pragma omp parallel for schedule(static)
    for(int r=0;r<c->rows;r++)c->fusion[c->row0+r]=glm53f_dot_bf16_sve(c->eh+(size_t)r*2*MTP_H,c->pair,2*MTP_H);int counts[12],displs[12];for(int r=0;r<c->ranks;r++){displs[r]=(int)((long long)MTP_H*r/c->ranks);counts[r]=(int)((long long)MTP_H*(r+1)/c->ranks)-displs[r];}if(MPI_Allgatherv(MPI_IN_PLACE,0,MPI_FLOAT,c->fusion,counts,displs,MPI_FLOAT,MPI_COMM_WORLD)!=MPI_SUCCESS)return-1;memcpy(c->head_streams,c->fusion,MTP_H*4);mtp_norm(c->normalized,c->fusion,c->input_norm);if(cache_only)return glm53f_sparse_cache_append_12n(c->attention,c->normalized);if(glm53f_sparse_sublayer_12n(c->attention,c->sublayer,c->normalized))return-1;
#pragma omp parallel for schedule(static)
    for(int i=0;i<MTP_H;i++)c->fusion[i]+=c->sublayer[i];mtp_norm(c->normalized,c->fusion,c->post_norm);glm53f_moe_stage_set_layer_12n(c->moe,45);if(glm53f_moe_stage_sublayer_12n(c->moe,c->sublayer,c->normalized))return-1;
#pragma omp parallel for schedule(static)
    for(int i=0;i<MTP_H;i++)c->fusion[i]+=c->sublayer[i];if(draft_hidden)memcpy(draft_hidden,c->fusion,MTP_H*4);for(int s=0;s<MTP_STREAMS;s++)memcpy(c->head_streams+(size_t)s*MTP_H,c->fusion,MTP_H*4);return glm53f_target_head_argmax_12n(c->head,c->head_streams,draft,logit);}
int glm53f_mtp_forward_12n(glm53f_mtp_context_12n*c,int token,const float*hidden,int*draft,float*logit,float*draft_hidden){return mtp_run(c,token,hidden,draft,logit,draft_hidden,0);}
int glm53f_mtp_cache_append_12n(glm53f_mtp_context_12n*c,int token,const float*hidden){return mtp_run(c,token,hidden,NULL,NULL,NULL,1);}
/* Prefill-only: preserve scalar norms, dot chains and cache updates. The
 * gathered layout is rank-major; unpacking restores token-major fusion rows. */
int glm53f_mtp_cache_append_batch_12n(glm53f_mtp_context_12n *c,
        const int *tokens, const float *hidden, int count) {
    if (!c || !tokens || !hidden || count < 1 || count > MTP_PRIME_TILE ||
        count > c->capacity - glm53f_mtp_length_12n(c)) return -1;
    if (!c->prime_storage) {
        c->prime_storage = mtp_a256((size_t)MTP_PRIME_TILE * MTP_H * 8 * sizeof(float));
        if (!c->prime_storage) return -1;
    }
    float *embed = c->prime_storage;
    float *pair = embed + (size_t)MTP_PRIME_TILE * MTP_H * 4;
    float *local = pair + (size_t)MTP_PRIME_TILE * MTP_H * 2;
    float *gather = local + (size_t)MTP_PRIME_TILE * MTP_H;
    if (glm53f_embedding_streams_packed_12n(c->embedding, tokens, count, embed)) return -1;
    int base = glm53f_mtp_length_12n(c);
    if (!base) memset(embed, 0, (size_t)MTP_STREAMS * MTP_H * sizeof(float));
    for (int t = 0; t < count; ++t) {
        mtp_norm(pair + (size_t)t * 2 * MTP_H,
                 embed + (size_t)t * MTP_STREAMS * MTP_H, c->enorm);
        mtp_norm(pair + ((size_t)t * 2 + 1) * MTP_H,
                 hidden + (size_t)t * MTP_H, c->hnorm);
    }
#pragma omp parallel for collapse(2) schedule(static)
    for (int block = 0; block < c->rows / 4; ++block)
        for (int base_token = 0; base_token < count; base_token += 4) {
            int n = count - base_token; if (n > 4) n = 4;
            glm53f_matvec_bf16_4x4(local + (size_t)base_token * c->rows + block * 4,
                c->rows, c->eh + (size_t)block * 4 * 2 * MTP_H,
                pair + (size_t)base_token * 2 * MTP_H, n, 2 * MTP_H);
        }
#pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < count; ++t)
        for (int r = c->rows / 4 * 4; r < c->rows; ++r)
            local[(size_t)t * c->rows + r] = glm53f_dot_bf16_sve(
                c->eh + (size_t)r * 2 * MTP_H, pair + (size_t)t * 2 * MTP_H, 2 * MTP_H);
    int counts[12], offsets[12], offset = 0;
    for (int r = 0; r < c->ranks; ++r) {
        int row0 = (int)((long long)MTP_H * r / c->ranks);
        int rows = (int)((long long)MTP_H * (r + 1) / c->ranks) - row0;
        offsets[r] = offset; counts[r] = count * rows; offset += counts[r];
    }
    if (MPI_Allgatherv(local, count * c->rows, MPI_FLOAT, gather, counts,
                       offsets, MPI_FLOAT, MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
    for (int t = 0; t < count; ++t) {
        for (int r = 0; r < c->ranks; ++r) {
            int row0 = (int)((long long)MTP_H * r / c->ranks);
            int rows = counts[r] / count;
            memcpy(c->fusion + row0, gather + offsets[r] + (size_t)t * rows,
                   (size_t)rows * sizeof(float));
        }
        memcpy(c->head_streams, c->fusion, MTP_H * sizeof(float));
        /* Fusion is complete, so its input-pair storage can be reused. */
        mtp_norm(pair + (size_t)t * MTP_H, c->fusion, c->input_norm);
    }
    return glm53f_sparse_cache_append_batch_12n(c->attention, pair, count);
}
int glm53f_mtp_length_12n(const glm53f_mtp_context_12n*c){return c?glm53f_sparse_length_12n(c->attention):-1;}
int glm53f_mtp_head_hidden_12n(const glm53f_mtp_context_12n *c, float *hidden) {
    return c ? glm53f_target_head_hidden_12n(c->head, hidden, 1) : -1;
}
int glm53f_mtp_restore_length_12n(glm53f_mtp_context_12n*c,int length){return c?glm53f_sparse_restore_length_12n(c->attention,length):-1;}
void glm53f_mtp_free_12n(glm53f_mtp_context_12n*c){if(!c)return;free(c->prime_storage);free(c->head_streams);free(c->sublayer);free(c->normalized);free(c->fusion);free(c->pair);free(c->embed_streams);glm53f_target_head_free_12n(c->head);glm53f_moe_stage_free_12n(c->moe);glm53f_sparse_free_12n(c->attention);glm53f_embedding_free_12n(c->embedding);free(c->eh);free(c->post_norm);free(c->input_norm);free(c->hnorm);free(c->enorm);free(c);}
