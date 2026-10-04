#ifndef GLM53F_KDA_12N_H
#define GLM53F_KDA_12N_H
#include <stddef.h>
#include "glm53f_prefill.h"
#include "glm53f_dist.h"

typedef struct glm53f_kda_context_12n glm53f_kda_context_12n;

glm53f_kda_context_12n *glm53f_kda_create_12n(
    const char *model_dir, int layer);
/* Borrows dist for its lifetime; requires an owned KDA layer and PP native image. */
glm53f_kda_context_12n *glm53f_kda_create_dist(
    const glm53f_dist *dist, const char *model_dir, const char *native_stage, int layer);
void glm53f_kda_reset_12n(glm53f_kda_context_12n *context);
void glm53f_kda_configure_prefill_12n(glm53f_kda_context_12n *context,
                                     const glm53f_prefill_config *config);
void glm53f_kda_free_12n(glm53f_kda_context_12n *context);
int glm53f_kda_convert_int8_12n(glm53f_kda_context_12n *context);
int glm53f_kda_sublayer_12n(
    void *context, float *output, const float *normalized_input);
int glm53f_kda_sublayer_batch_12n(
    glm53f_kda_context_12n *context, float *output,
    const float *normalized_input, int tokens);
/* Token tile of the batched KDA prefill path (buffers and GEMM scratch are sized for it). */
enum { GLM53F_KDA_TILE_TOKENS = 64 };
/* Builds the prefill panel GEMM copies now (model load) instead of inside the first prefill call. */
void glm53f_kda_prewarm_12n(glm53f_kda_context_12n *c);
/* Registers this layer's decode front matrices in the prefetch plan (glm53f_pf_plan.h). */
void glm53f_kda_prefetch_plan_12n(const glm53f_kda_context_12n *c);
int glm53f_kda_fusable_12n(const glm53f_kda_context_12n *c);
void glm53f_kda_team_12n(void *context, const float *x);
/* Prefill only: leave per-rank partial outputs unreduced (the caller runs the collective). */
void glm53f_kda_set_defer_reduce_12n(int on);
int glm53f_kda_sublayer_batch_capture_12n(
    glm53f_kda_context_12n *context, float *output,
    const float *normalized_input, int tokens,
    void *state_after_each_token, size_t state_stride);
void glm53f_kda_last_phase_12n(
    const glm53f_kda_context_12n *context, double phase_seconds[3]);
void glm53f_kda_last_detail_12n(
    const glm53f_kda_context_12n *context, double phase_seconds[5]);
int glm53f_kda_head_range_12n(const glm53f_kda_context_12n *context, int *first, int *count);
size_t glm53f_kda_state_bytes_12n(const glm53f_kda_context_12n *context);
int glm53f_kda_save_state_12n(
    const glm53f_kda_context_12n *context, void *snapshot, size_t bytes);
int glm53f_kda_restore_state_12n(
    glm53f_kda_context_12n *context, const void *snapshot, size_t bytes);

#endif
