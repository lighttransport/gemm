#ifndef GLM53F_TARGET_MODEL_12N_H
#define GLM53F_TARGET_MODEL_12N_H
#include "glm53f_prefill.h"
#include "glm53f_dist.h"
#include "glm53f_pp_images.h"

typedef struct glm53f_target_model_12n glm53f_target_model_12n;
typedef struct glm53f_target_snapshot_12n glm53f_target_snapshot_12n;

glm53f_target_model_12n *glm53f_target_model_create_12n(
    const char *model_dir, const char *routed_stage,
    const char *shared_stage, int capacity);
/* Collective on dist->world. Owned-layer PP graph requires complete native images and conservative
 * memory preflight. Borrows dist/images paths; one resident graph per process. */
glm53f_target_model_12n *glm53f_target_model_create_dist(
    glm53f_dist *dist, const glm53f_pp_images *images, int capacity);
void glm53f_target_model_free_12n(glm53f_target_model_12n *model);
/* Configure only with an empty/reset model. Named recipes enable the legacy
 * process-wide prefill switches; numerical fast features remain per model. */
int glm53f_target_model_configure_prefill_12n(glm53f_target_model_12n *model,
                                             const glm53f_prefill_config *config);
int glm53f_target_model_convert_int8_12n(glm53f_target_model_12n *model);
int glm53f_target_model_convert_kda_int8_12n(glm53f_target_model_12n *model);
int glm53f_target_model_touch_cache_12n(glm53f_target_model_12n *model);
/* Must be called while the model is empty. Storage is separate from the
 * owner-distributed cache and is safe for 512K-capacity runs. */
int glm53f_target_model_set_cp_hot_prefix_12n(glm53f_target_model_12n *model,
                                              int hot_prefix);
const float *glm53f_target_model_logits_12n(const glm53f_target_model_12n *model,
                                           int *first, int *count);
/* Evaluate the last completed token without advancing any cache. Invalid
 * before the first step or immediately after snapshot restoration. */
int glm53f_target_model_readout_12n(glm53f_target_model_12n *model,
                                    int *token, float *logit);
/* Diagnostics only: each rank creates/checks PREFIX.rankNN.bin. Stream every
 * token's complete final-layer hidden streams, then persistent KDA/sparse state.
 * Writes are exclusive (never overwrite an existing trace); I/O is not a speed run.
 * compare=0 writes, 1 is byte-exact, 2 checks typed FP32 fields at rel-L2<=1e-3
 * with exact metadata and reports selected-index changes separately. */
int glm53f_target_export_fields_12n(glm53f_target_model_12n *model, const char *prefix);
int glm53f_target_trace_open_12n(glm53f_target_model_12n *model, const char *prefix, int compare);
int glm53f_target_trace_close_12n(glm53f_target_model_12n *model);
int glm53f_target_place_weights_12n(glm53f_target_model_12n *m);
int glm53f_target_model_step_12n(
    glm53f_target_model_12n *model, int input_token,
    int *next_token, float *next_logit, float *target_hidden);
/* Greedy transitions after an already predicted first token. ids[0] receives
 * first; ids[1..transitions] are the following predictions. Optional observer
 * runs on the controller after each transition. The persistent executor keeps
 * one OpenMP team alive for this entire sequence. */
int glm53f_target_decode_sequence_12n(glm53f_target_model_12n *model,
    int first, int transitions, int *ids,
    void (*observer)(void *, int), void *observer_context);
/* Execute one four-stream position over an owned contiguous layer interval,
 * retaining scalar decode arithmetic and model-owned readout storage. */
int glm53f_target_model_layers_scalar_12n(glm53f_target_model_12n *model,
    float *streams, int first_layer, int end_layer);
/* Prompt-only calls accept up to 256 positions, or the compiled
 * GLM53F_PREFILL_MAX_TOKENS capacity with the fast recipe.
 * Calls returning logits, hidden states, or snapshots retain the limit five. */
int glm53f_target_model_step_batch_12n(
    glm53f_target_model_12n *model, const int *input_tokens, int tokens,
    int *next_tokens, float *next_logits, float *target_hidden,
    glm53f_target_snapshot_12n **state_after_each_token);
/* Produce token-major four-stream embeddings without advancing model state. */
int glm53f_target_model_embed_batch_12n(glm53f_target_model_12n *model,
    const int *input_tokens, int tokens, float *streams);
/* Execute an owned layer interval in place on tokens*4*4096 FP32 streams.
 * Performs neither embedding nor final normalization/head. The model retains
 * the last token independently of caller/transfer-buffer lifetime. Requires
 * the configured wide recipe above256 positions; no verification snapshots
 * or legacy trace stream. PP constructors will restrict the owned interval. */
int glm53f_target_model_layers_batch_12n(glm53f_target_model_12n *model,
    float *streams, int tokens, int first_layer, int end_layer);
/* Prompt-only execution with every position's collapsed target hidden state.
 * Caller supplies tokens*4096 floats. Uses the same prefill recipe/panels as
 * step_batch; logits and verification snapshots retain their five-token limit. */
int glm53f_target_model_prefill_hidden_12n(glm53f_target_model_12n *model,
    const int *input_tokens, int tokens, float *target_hidden);
/* MTP uses post-output-norm hiddens. Normalize prompt means in place, or copy
 * the actual normalized hiddens of the last scalar/verification head call. */
int glm53f_target_model_normalize_hidden_12n(const glm53f_target_model_12n *model,
    float *target_hidden, int tokens);
int glm53f_target_model_head_hidden_12n(const glm53f_target_model_12n *model,
    float *target_hidden, int tokens);
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
