#ifndef GLM53F_DENSE_FFN_12N_H
#define GLM53F_DENSE_FFN_12N_H
#include "glm53f_dist.h"

typedef struct glm53f_dense_ffn_context_12n glm53f_dense_ffn_context_12n;

glm53f_dense_ffn_context_12n *glm53f_dense_ffn_create_12n(
    const char *model_dir, int layer);
/* Borrow dist until free. PP requires an explicit stage-native image and an
 * owned dense layer; incompatible manifests fail before payload allocation. */
glm53f_dense_ffn_context_12n *glm53f_dense_ffn_create_dist(
    const glm53f_dist *dist, const char *model_dir,
    const char *native_stage, int layer);
void glm53f_dense_ffn_free_12n(glm53f_dense_ffn_context_12n *context);
/* Native prefill capacity selected before construction; FP8 remains at four. */
int glm53f_dense_ffn_batch_capacity_12n(const glm53f_dense_ffn_context_12n *context);
/* Select a tile within an existing allocation for baseline/candidate checks. */
int glm53f_dense_ffn_set_batch_tile_12n(glm53f_dense_ffn_context_12n *context, int tile);
int glm53f_dense_ffn_sublayer_12n(
    void *context, float *output, const float *normalized_input);
int glm53f_dense_ffn_sublayer_batch_12n(
    glm53f_dense_ffn_context_12n *context, float *output,
    const float *normalized_input, int tokens);

#endif
