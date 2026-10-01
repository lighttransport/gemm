#ifndef GLM53F_MOE_STAGE_12N_H
#define GLM53F_MOE_STAGE_12N_H
#include "glm53f_prefill.h"

typedef struct glm53f_moe_stage_context_12n glm53f_moe_stage_context_12n;
void glm53f_moe_configure_prefill_12n(glm53f_moe_stage_context_12n *context,
                                     const glm53f_prefill_config *config);

glm53f_moe_stage_context_12n *glm53f_moe_stage_create_12n(
    const char *routed_stage, const char *shared_stage,
    const char *model_dir, int first_layer, int layer_count);
/* Prefetch-plan registration for MoE layer `layer` (see glm53f_pf_plan.h). */
void glm53f_moe_stage_prefetch_plan_12n(glm53f_moe_stage_context_12n *c, int layer);
/* Called by every thread after mHC normalization in an enclosing team. */
void glm53f_moe_router_team_12n(void *context, const float *input);
void glm53f_moe_stage_set_layer_12n(
    glm53f_moe_stage_context_12n *context, int layer);
/* Destructive only to the private anonymous copy; /local files stay FP8. */
int glm53f_moe_stage_convert_int8_12n(glm53f_moe_stage_context_12n *context);
int glm53f_moe_stage_sublayer_12n(
    void *context, float *output, const float *normalized_input);
int glm53f_moe_stage_sublayer_batch_12n(
    glm53f_moe_stage_context_12n *context, float *output,
    const float *normalized_input, int tokens);
void glm53f_moe_stage_profile_reset_12n(glm53f_moe_stage_context_12n *context);
void glm53f_moe_stage_profile_report_12n(
    const glm53f_moe_stage_context_12n *context, long positions,
    const char *label);
void glm53f_moe_stage_free_12n(glm53f_moe_stage_context_12n *context);

#endif
