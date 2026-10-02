#ifndef GLM53F_EMBEDDING_12N_H
#define GLM53F_EMBEDDING_12N_H
#include "glm53f_dist.h"

typedef struct glm53f_embedding_context_12n glm53f_embedding_context_12n;

glm53f_embedding_context_12n *glm53f_embedding_create_12n(
    const char *model_dir);
/* Stage0 only; borrows dist and requires a PP vocabulary image. */
glm53f_embedding_context_12n *glm53f_embedding_create_dist(
    const glm53f_dist *dist, const char *model_dir, const char *native_stage);
void glm53f_embedding_free_12n(glm53f_embedding_context_12n *context);
int glm53f_embedding_streams_12n(
    glm53f_embedding_context_12n *context, int token, float *streams);

int glm53f_embedding_streams_batch_12n(
    glm53f_embedding_context_12n *context, const int *ids, int tokens, float *streams);

#endif
