#ifndef GLM53F_KDA_12N_H
#define GLM53F_KDA_12N_H

typedef struct glm53f_kda_context_12n glm53f_kda_context_12n;

glm53f_kda_context_12n *glm53f_kda_create_12n(
    const char *model_dir, int layer);
void glm53f_kda_reset_12n(glm53f_kda_context_12n *context);
void glm53f_kda_free_12n(glm53f_kda_context_12n *context);
int glm53f_kda_sublayer_12n(
    void *context, float *output, const float *normalized_input);
void glm53f_kda_last_phase_12n(
    const glm53f_kda_context_12n *context, double phase_seconds[3]);

#endif
