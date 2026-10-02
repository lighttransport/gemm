#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef GLM53F_EXTERNAL_ST_IMPLEMENTATION
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#endif
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"
#include "glm53f_embedding_12n.h"
#include "glm53f_pp_f32.h"
#include <mpi.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { GLM53F_EMBED_HIDDEN = 4096, GLM53F_EMBED_STREAMS = 4,
       GLM53F_EMBED_VOCAB = 154880 };

struct glm53f_embedding_context_12n {
    int rank, ranks, row0, rows;
    const glm53f_dist *dist;
    uint16_t *weight;
    float *q2_weight;
    float row[GLM53F_EMBED_HIDDEN];
};

static glm53f_embedding_context_12n *embedding_create(const glm53f_dist *dist, const char *model, const char *native_stage) {
    glm53f_embedding_context_12n *c = calloc(1, sizeof(*c));
    glm53f_st_context *st;
    if (!c) return NULL;
    if(dist){if(!dist->initialized||dist->config.layout!=GLM53F_PP3_TP4||dist->map.stage!=0||!native_stage)goto fail;
        c->dist=dist;c->rank=dist->map.tp_rank;c->ranks=dist->map.tp_size;
    }else{MPI_Comm_rank(MPI_COMM_WORLD,&c->rank);MPI_Comm_size(MPI_COMM_WORLD,&c->ranks);if(c->ranks!=12)goto fail;}
    c->row0 = (int)((long long)GLM53F_EMBED_VOCAB * c->rank / c->ranks);
    c->rows = (int)((long long)GLM53F_EMBED_VOCAB * (c->rank + 1) / c->ranks) - c->row0;
    if(dist){if(glm53f_pp_f32_load(dist,native_stage,"EMBED","token_embd.weight",c->row0,c->rows,GLM53F_EMBED_HIDDEN,&c->q2_weight))goto fail;return c;}
    const char *stage = getenv("GLM53F_Q2_EMBED_STAGE");
    if (stage && *stage) {
        char path[4096];
        FILE *f;
        snprintf(path, sizeof(path), "%s/rank%02d.f32", stage, c->rank);
        f = fopen(path, "rb");
        if (!f) goto fail;
        if (posix_memalign((void **)&c->q2_weight, 256,
                           (size_t)c->rows * GLM53F_EMBED_HIDDEN * sizeof(float))) {
            fclose(f); goto fail;
        }
        if (fread(c->q2_weight, sizeof(float),
                  (size_t)c->rows * GLM53F_EMBED_HIDDEN, f) !=
            (size_t)c->rows * GLM53F_EMBED_HIDDEN) {
            fclose(f); goto fail;
        }
        fclose(f);
        return c;
    }
    if (posix_memalign((void **)&c->weight, 256,
                       (size_t)c->rows * GLM53F_EMBED_HIDDEN * sizeof(uint16_t))) goto fail;
    st = glm53f_st_open(model);
    if (!st || glm53f_st_read(st, "model.language_model.embed_tokens.weight",
        (size_t)c->row0 * GLM53F_EMBED_HIDDEN * sizeof(uint16_t), c->weight,
        (size_t)c->rows * GLM53F_EMBED_HIDDEN * sizeof(uint16_t))) {
        glm53f_st_close(st);
        goto fail;
    }
    glm53f_st_close(st);
    return c;
fail:
    glm53f_embedding_free_12n(c);
    return NULL;
}

glm53f_embedding_context_12n *glm53f_embedding_create_12n(const char *model){return embedding_create(NULL,model,NULL);}
glm53f_embedding_context_12n *glm53f_embedding_create_dist(const glm53f_dist *dist,const char *model,const char *native_stage){return dist?embedding_create(dist,model,native_stage):NULL;}

void glm53f_embedding_free_12n(glm53f_embedding_context_12n *c) {
    if (!c) return;
    free(c->weight);
    free(c->q2_weight);
    free(c);
}

int glm53f_embedding_streams_12n(
        glm53f_embedding_context_12n *c, int token, float *streams) {
    int owner = -1;
    if (!c || token < 0 || token >= GLM53F_EMBED_VOCAB || !streams) return -1;
    for (int rank = 0; rank < c->ranks; ++rank) {
        int begin = (int)((long long)GLM53F_EMBED_VOCAB * rank / c->ranks);
        int end = (int)((long long)GLM53F_EMBED_VOCAB * (rank + 1) / c->ranks);
        if (token >= begin && token < end) { owner = rank; break; }
    }
    if (owner < 0) return -1;
    if (owner == c->rank) {
        if (c->q2_weight) {
            const float *src = c->q2_weight + (size_t)(token - c->row0) * GLM53F_EMBED_HIDDEN;
            memcpy(c->row, src, sizeof(c->row));
        } else {
            const uint16_t *src = c->weight + (size_t)(token - c->row0) * GLM53F_EMBED_HIDDEN;
            for (int i = 0; i < GLM53F_EMBED_HIDDEN; ++i)
                c->row[i] = glm53f_bf16_to_f32(src[i]);
        }
    }
    if (MPI_Bcast(c->row, GLM53F_EMBED_HIDDEN, MPI_FLOAT, owner,
                  c->dist?c->dist->tp:MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
    for (int stream = 0; stream < GLM53F_EMBED_STREAMS; ++stream)
        memcpy(streams + (size_t)stream * GLM53F_EMBED_HIDDEN,
               c->row, sizeof(c->row));
    return 0;
}

/* PP prefill broadcasts packed rows once per owner, then restores original
 * token order and duplicates four streams locally. FP32 bits are untouched. */
int glm53f_embedding_streams_batch_12n(glm53f_embedding_context_12n *c,
        const int *ids,int tokens,float *streams) {
    if(!c||!ids||!streams||tokens<1||tokens>4096)return-1;
    if(!c->dist){for(int t=0;t<tokens;t++)if(glm53f_embedding_streams_12n(c,ids[t],streams+(size_t)t*GLM53F_EMBED_STREAMS*GLM53F_EMBED_HIDDEN))return-1;return 0;}
    for(int t=0;t<tokens;t++)if(ids[t]<0||ids[t]>=GLM53F_EMBED_VOCAB)return-1;
    float *packed=malloc((size_t)tokens*GLM53F_EMBED_HIDDEN*sizeof(float));if(!packed)return-1;
    for(int owner=0;owner<c->ranks;owner++){
        int begin=(int)((long long)GLM53F_EMBED_VOCAB*owner/c->ranks),end=(int)((long long)GLM53F_EMBED_VOCAB*(owner+1)/c->ranks),count=0;
        for(int t=0;t<tokens;t++)if(ids[t]>=begin&&ids[t]<end){
            if(owner==c->rank)memcpy(packed+(size_t)count*GLM53F_EMBED_HIDDEN,c->q2_weight+(size_t)(ids[t]-c->row0)*GLM53F_EMBED_HIDDEN,GLM53F_EMBED_HIDDEN*sizeof(float));count++;}
        if(!count)continue;
        if(MPI_Bcast(packed,count*GLM53F_EMBED_HIDDEN,MPI_FLOAT,owner,c->dist->tp)!=MPI_SUCCESS){free(packed);return-1;}
        int i=0;for(int t=0;t<tokens;t++)if(ids[t]>=begin&&ids[t]<end){
            for(int h=0;h<GLM53F_EMBED_STREAMS;h++)memcpy(streams+((size_t)t*GLM53F_EMBED_STREAMS+h)*GLM53F_EMBED_HIDDEN,packed+(size_t)i*GLM53F_EMBED_HIDDEN,GLM53F_EMBED_HIDDEN*sizeof(float));i++;}
    }
    free(packed);return 0;
}
