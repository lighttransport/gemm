#ifndef GLM53F_SPARSE_12N_H
#define GLM53F_SPARSE_12N_H
#include <stddef.h>

typedef struct glm53f_sparse_context_12n glm53f_sparse_context_12n;
typedef struct glm53f_sparse_prefill_workspace_12n glm53f_sparse_prefill_workspace_12n;
struct glm53f_state_io;
int glm53f_sparse_state_io_12n(const glm53f_sparse_context_12n *context,
                               struct glm53f_state_io *io);
glm53f_sparse_prefill_workspace_12n *glm53f_sparse_prefill_workspace_create_12n(void);
void glm53f_sparse_prefill_workspace_free_12n(glm53f_sparse_prefill_workspace_12n *workspace);
/* At most 32 positions. A model shares one workspace across its sparse layers. */
int glm53f_sparse_prefill_12n(glm53f_sparse_context_12n *context,
    glm53f_sparse_prefill_workspace_12n *workspace, float *output,
    const float *normalized_input, int tokens);
enum { GLM53F_SPARSE_PROFILE_PHASES = 7 };
/* Seconds: front, index (including score exchange), pack, MLA, output,
 * output reduction, and unsplit context-parallel front/index/MLA. */
void glm53f_sparse_profile_add_12n(const glm53f_sparse_context_12n *context,
                                  double *seconds);
void glm53f_sparse_profile_reset_12n(glm53f_sparse_context_12n *context);

glm53f_sparse_context_12n *glm53f_sparse_create_12n(
    const char *model_dir, int layer, int capacity);
/* BF16 affects CP latent rows only; index/pool state remains FP32. */
glm53f_sparse_context_12n *glm53f_sparse_create_format_12n(
    const char *model_dir, int layer, int capacity, int latent_bf16);
int glm53f_sparse_convert_int8_12n(glm53f_sparse_context_12n *context);
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
/* Commit zero-length cache pages before a capacity check; rejects live state. */
int glm53f_sparse_touch_cache_12n(glm53f_sparse_context_12n *context);

#endif
