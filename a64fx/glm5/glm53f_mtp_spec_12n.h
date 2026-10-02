#ifndef GLM53F_MTP_SPEC_12N_H
#define GLM53F_MTP_SPEC_12N_H
#include "glm53f_target_model_12n.h"
#include "glm53f_mtp_12n.h"

typedef struct glm53f_mtp_spec_workspace_12n glm53f_mtp_spec_workspace_12n;
typedef struct {
    int accepted, proposed, cycles, fallback_tokens, cache_synchronized;
    double draft_seconds, verify_seconds, replay_seconds, plain_seconds;
} glm53f_mtp_spec_stats_12n;

glm53f_mtp_spec_workspace_12n *glm53f_mtp_spec_workspace_create_12n(
    const glm53f_target_model_12n *model);
void glm53f_mtp_spec_workspace_free_12n(glm53f_mtp_spec_workspace_12n *workspace);
/* Target is primed through prompt_count positions; MTP has prompt_count-1
 * teacher-forced pairs. parent_hidden is the actual final prompt hidden state
 * after the target output norm; chained draft hiddens use the MTP head norm.
 * Each cycle verifies the known next input and its drafts together. ids[0] is
 * first; ids[1..transitions] count only delivered target predictions.
 * A positive plain cost enables rank-consistent recurring fallback. Fallback
 * clears the MTP cache; re-prime before another speculative sequence. */
int glm53f_mtp_spec_decode_12n(glm53f_target_model_12n *model,
    glm53f_mtp_context_12n *mtp, glm53f_mtp_spec_workspace_12n *workspace,
    int prompt_count, const float *parent_hidden, int first, int transitions,
    int depth, double plain_seconds_per_token, int *ids,
    glm53f_mtp_spec_stats_12n *stats,
    void (*observer)(void *, int), void *observer_context);
#endif
