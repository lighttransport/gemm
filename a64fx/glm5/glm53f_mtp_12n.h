#ifndef GLM53F_MTP_12N_H
#define GLM53F_MTP_12N_H

typedef struct glm53f_mtp_context_12n glm53f_mtp_context_12n;

glm53f_mtp_context_12n *glm53f_mtp_create_12n(
    const char *model_dir, const char *routed_stage,
    const char *shared_stage, int capacity);
void glm53f_mtp_free_12n(glm53f_mtp_context_12n *context);
int glm53f_mtp_forward_12n(
    glm53f_mtp_context_12n *context, int input_token,
    const float *target_hidden, int *draft_token, float *draft_logit,
    float *draft_hidden);
int glm53f_mtp_length_12n(const glm53f_mtp_context_12n *context);
int glm53f_mtp_restore_length_12n(
    glm53f_mtp_context_12n *context, int length);

#endif
