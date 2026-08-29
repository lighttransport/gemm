#ifndef GLM53F_TARGET_MODEL_12N_H
#define GLM53F_TARGET_MODEL_12N_H

typedef struct glm53f_target_model_12n glm53f_target_model_12n;
typedef struct glm53f_target_snapshot_12n glm53f_target_snapshot_12n;

glm53f_target_model_12n *glm53f_target_model_create_12n(
    const char *model_dir, const char *routed_stage,
    const char *shared_stage, int capacity);
void glm53f_target_model_free_12n(glm53f_target_model_12n *model);
int glm53f_target_model_step_12n(
    glm53f_target_model_12n *model, int input_token,
    int *next_token, float *next_logit, float *target_hidden);
glm53f_target_snapshot_12n *glm53f_target_snapshot_create_12n(
    const glm53f_target_model_12n *model);
void glm53f_target_snapshot_free_12n(glm53f_target_snapshot_12n *snapshot);
int glm53f_target_snapshot_save_12n(
    const glm53f_target_model_12n *model, glm53f_target_snapshot_12n *snapshot);
int glm53f_target_snapshot_restore_12n(
    glm53f_target_model_12n *model, const glm53f_target_snapshot_12n *snapshot);

#endif
