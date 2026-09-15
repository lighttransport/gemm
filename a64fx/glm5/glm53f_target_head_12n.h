#ifndef GLM53F_TARGET_HEAD_12N_H
#define GLM53F_TARGET_HEAD_12N_H

typedef struct glm53f_target_head_context_12n glm53f_target_head_context_12n;

glm53f_target_head_context_12n *glm53f_target_head_create_12n(
    const char *model_dir);
glm53f_target_head_context_12n *glm53f_target_head_create_with_norm_12n(
    const char *model_dir, const char *norm_tensor);
void glm53f_target_head_free_12n(glm53f_target_head_context_12n *context);
/* Diagnostic view of the most recent scalar vocabulary shard; invalidated
 * by the next head evaluation. No communication and no ownership transfer. */
const float *glm53f_target_head_logits_12n(const glm53f_target_head_context_12n *context,
                                          int *first, int *count);
int glm53f_target_head_argmax_12n(
    glm53f_target_head_context_12n *context, const float *streams,
    int *token, float *logit);
int glm53f_target_head_argmax_batch_12n(
    glm53f_target_head_context_12n *context, const float *streams, int tokens,
    int *token, float *logit);
void glm53f_target_head_last_phase_12n(
    const glm53f_target_head_context_12n *context, double phase_seconds[3]);

#endif
