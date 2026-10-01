#ifndef GLM53F_LOOKUP_SPEC_12N_H
#define GLM53F_LOOKUP_SPEC_12N_H
#include "glm53f_target_model_12n.h"
typedef struct glm53f_lookup_workspace_12n glm53f_lookup_workspace_12n;
typedef struct {
    int accepted, proposed, cycles, fallback_tokens;
    double lookup_seconds, verify_seconds, rollback_seconds, plain_seconds;
} glm53f_lookup_stats_12n;
glm53f_lookup_workspace_12n *glm53f_lookup_workspace_create_12n(
    const glm53f_target_model_12n *model, int history_capacity);
void glm53f_lookup_workspace_free_12n(glm53f_lookup_workspace_12n *workspace);
/* Model is primed through all prompt tokens; first is its greedy prediction.
 * Returns exactly transitions new predictions in ids[1..], counting every
 * lookup, verifier, snapshot restore and fallback in caller's wall time.
 * Plain reference seconds/token enables adaptive fallback; zero keeps lookup. */
int glm53f_lookup_decode_12n(glm53f_target_model_12n *model,
    glm53f_lookup_workspace_12n *workspace, const int *prompt, int prompt_count,
    int first, int transitions, int depth, double plain_seconds_per_token,
    int *ids, glm53f_lookup_stats_12n *stats,
    void (*observer)(void *, int), void *observer_context);
#endif
