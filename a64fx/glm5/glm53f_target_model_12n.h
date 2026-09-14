#ifndef GLM53F_TARGET_MODEL_12N_H
#define GLM53F_TARGET_MODEL_12N_H
#include "glm53f_prefill.h"

typedef struct glm53f_target_model_12n glm53f_target_model_12n;
typedef struct glm53f_target_snapshot_12n glm53f_target_snapshot_12n;

glm53f_target_model_12n *glm53f_target_model_create_12n(
    const char *model_dir, const char *routed_stage,
    const char *shared_stage, int capacity);
void glm53f_target_model_free_12n(glm53f_target_model_12n *model);
int glm53f_target_model_convert_int8_12n(glm53f_target_model_12n *model);
int glm53f_target_model_convert_kda_int8_12n(glm53f_target_model_12n *model);
int glm53f_target_model_touch_cache_12n(glm53f_target_model_12n *model);
/* Diagnostics only: each rank creates/checks PREFIX.rankNN.bin. Stream every
 * token's complete final-layer hidden streams, then persistent KDA/sparse state.
 * Writes are exclusive (never overwrite an existing trace); I/O is not a speed run. */
int glm53f_target_trace_open_12n(glm53f_target_model_12n *model, const char *prefix, int compare);
int glm53f_target_trace_close_12n(glm53f_target_model_12n *model);
int glm53f_target_model_step_12n(
    glm53f_target_model_12n *model, int input_token,
    int *next_token, float *next_logit, float *target_hidden);
/* Prompt-only calls (all output pointers NULL) accept up to 256 positions.
 * Calls returning logits, hidden states, or snapshots retain the limit five. */
int glm53f_target_model_step_batch_12n(
    glm53f_target_model_12n *model, const int *input_tokens, int tokens,
    int *next_tokens, float *next_logits, float *target_hidden,
    glm53f_target_snapshot_12n **state_after_each_token);
void glm53f_target_profile_reset_12n(glm53f_target_model_12n *model);
void glm53f_target_profile_report_12n(
    const glm53f_target_model_12n *model, const char *label);
glm53f_target_snapshot_12n *glm53f_target_snapshot_create_12n(
    const glm53f_target_model_12n *model);
void glm53f_target_snapshot_free_12n(glm53f_target_snapshot_12n *snapshot);
int glm53f_target_snapshot_save_12n(
    const glm53f_target_model_12n *model, glm53f_target_snapshot_12n *snapshot);
int glm53f_target_snapshot_restore_12n(
    glm53f_target_model_12n *model, const glm53f_target_snapshot_12n *snapshot);

#endif
