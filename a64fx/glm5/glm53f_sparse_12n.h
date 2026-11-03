#ifndef GLM53F_SPARSE_12N_H
#define GLM53F_SPARSE_12N_H
#include <stddef.h>

typedef struct glm53f_sparse_context_12n glm53f_sparse_context_12n;
typedef struct {
    int count, anchor, selected[2048];
} glm53f_sparse_selection_state;
int glm53f_sparse_set_policy_12n(glm53f_sparse_context_12n *context,
                                int top_k, int refresh);
void glm53f_sparse_selection_save_12n(const glm53f_sparse_context_12n *context,
                                     glm53f_sparse_selection_state *state);
void glm53f_sparse_selection_at_12n(const glm53f_sparse_context_12n *context,
    int length, glm53f_sparse_selection_state *state);
void glm53f_sparse_selection_restore_12n(glm53f_sparse_context_12n *context,
                                        const glm53f_sparse_selection_state *state);
void glm53f_sparse_selection_history_save_12n(
    const glm53f_sparse_context_12n *context,
    glm53f_sparse_selection_state history[8]);
void glm53f_sparse_selection_history_restore_12n(
    glm53f_sparse_context_12n *context,
    const glm53f_sparse_selection_state history[8]);

glm53f_sparse_context_12n *glm53f_sparse_create_12n(
    const char *model_dir, int layer, int capacity);
void glm53f_sparse_reset_12n(glm53f_sparse_context_12n *context);
void glm53f_sparse_free_12n(glm53f_sparse_context_12n *context);
int glm53f_sparse_sublayer_12n(
    void *context, float *output, const float *normalized_input);
/* Append only persistent KV/indexer state; no query or attention output. */
int glm53f_sparse_cache_append_12n(
    glm53f_sparse_context_12n *context, const float *normalized_input);
int glm53f_sparse_sublayer_batch_12n(
    glm53f_sparse_context_12n *context, float *output,
    const float *normalized_input, int tokens);
int glm53f_sparse_length_12n(const glm53f_sparse_context_12n *context);
int glm53f_sparse_restore_length_12n(
    glm53f_sparse_context_12n *context, int length);
int glm53f_sparse_is_context_parallel_12n(
    const glm53f_sparse_context_12n *context);
size_t glm53f_sparse_cache_bytes_12n(
    const glm53f_sparse_context_12n *context);
/* Cumulative seconds: projections, indexer, packing, MLA, output, collective.
 * Enabled with GLM53F_SPARSE_DETAIL; excludes model loading. */
void glm53f_sparse_profile_12n(glm53f_sparse_context_12n *context,
                              double seconds[6], int reset);

#endif
