#ifndef GLM53F_KDA_12N_H
#define GLM53F_KDA_12N_H
#include <stddef.h>

typedef struct glm53f_kda_context_12n glm53f_kda_context_12n;

glm53f_kda_context_12n *glm53f_kda_create_12n(
    const char *model_dir, int layer);
void glm53f_kda_reset_12n(glm53f_kda_context_12n *context);
void glm53f_kda_free_12n(glm53f_kda_context_12n *context);
int glm53f_kda_sublayer_12n(
    void *context, float *output, const float *normalized_input);
int glm53f_kda_sublayer_batch_12n(
    glm53f_kda_context_12n *context, float *output,
    const float *normalized_input, int tokens);
int glm53f_kda_sublayer_batch_capture_12n(
    glm53f_kda_context_12n *context, float *output,
    const float *normalized_input, int tokens,
    void *state_after_each_token, size_t state_stride);
void glm53f_kda_last_phase_12n(
    const glm53f_kda_context_12n *context, double phase_seconds[3]);
size_t glm53f_kda_state_bytes_12n(const glm53f_kda_context_12n *context);
int glm53f_kda_save_state_12n(
    const glm53f_kda_context_12n *context, void *snapshot, size_t bytes);
int glm53f_kda_restore_state_12n(
    glm53f_kda_context_12n *context, const void *snapshot, size_t bytes);

#endif
