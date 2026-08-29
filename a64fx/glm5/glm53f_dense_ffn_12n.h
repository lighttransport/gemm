#ifndef GLM53F_DENSE_FFN_12N_H
#define GLM53F_DENSE_FFN_12N_H

typedef struct glm53f_dense_ffn_context_12n glm53f_dense_ffn_context_12n;

glm53f_dense_ffn_context_12n *glm53f_dense_ffn_create_12n(
    const char *model_dir, int layer);
void glm53f_dense_ffn_free_12n(glm53f_dense_ffn_context_12n *context);
int glm53f_dense_ffn_sublayer_12n(
    void *context, float *output, const float *normalized_input);
int glm53f_dense_ffn_sublayer_batch_12n(
    glm53f_dense_ffn_context_12n *context, float *output,
    const float *normalized_input, int tokens);

#endif
