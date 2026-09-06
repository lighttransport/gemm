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
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { GLM53F_EMBED_HIDDEN = 4096, GLM53F_EMBED_STREAMS = 4,
       GLM53F_EMBED_VOCAB = 154880 };

struct glm53f_embedding_context_12n {
    int rank, ranks, row0, rows;
    uint16_t *weight;
    float row[GLM53F_EMBED_HIDDEN];
};

glm53f_embedding_context_12n *glm53f_embedding_create_12n(const char *model) {
    glm53f_embedding_context_12n *c = calloc(1, sizeof(*c));
    glm53f_st_context *st;
    if (!c) return NULL;
    MPI_Comm_rank(MPI_COMM_WORLD, &c->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c->ranks);
    if (c->ranks != 12) goto fail;
    c->row0 = (int)((long long)GLM53F_EMBED_VOCAB * c->rank / c->ranks);
    c->rows = (int)((long long)GLM53F_EMBED_VOCAB * (c->rank + 1) / c->ranks) - c->row0;
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

void glm53f_embedding_free_12n(glm53f_embedding_context_12n *c) {
    if (!c) return;
    free(c->weight);
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
        const uint16_t *src = c->weight + (size_t)(token - c->row0) * GLM53F_EMBED_HIDDEN;
        for (int i = 0; i < GLM53F_EMBED_HIDDEN; ++i)
            c->row[i] = glm53f_bf16_to_f32(src[i]);
    }
    if (MPI_Bcast(c->row, GLM53F_EMBED_HIDDEN, MPI_FLOAT, owner,
                  MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
    for (int stream = 0; stream < GLM53F_EMBED_STREAMS; ++stream)
        memcpy(streams + (size_t)stream * GLM53F_EMBED_HIDDEN,
               c->row, sizeof(c->row));
    return 0;
}
