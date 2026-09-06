#ifndef GLM53F_EMBEDDING_12N_H
#define GLM53F_EMBEDDING_12N_H

typedef struct glm53f_embedding_context_12n glm53f_embedding_context_12n;

glm53f_embedding_context_12n *glm53f_embedding_create_12n(
    const char *model_dir);
void glm53f_embedding_free_12n(glm53f_embedding_context_12n *context);
int glm53f_embedding_streams_12n(
    glm53f_embedding_context_12n *context, int token, float *streams);

#endif
