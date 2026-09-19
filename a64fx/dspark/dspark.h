#ifndef A64FX_DSPARK_H
#define A64FX_DSPARK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DSPARK_BLOCK_SIZE 7
#define DSPARK_TARGET_TAPS 5

typedef struct dspark_model dspark_model;
typedef struct dspark_state dspark_state;

typedef enum {
    DSPARK_BACKEND_AUTO = 0,
    DSPARK_BACKEND_SCALAR = 1,
    DSPARK_BACKEND_SVE = 2
} dspark_backend;

typedef enum {
    DSPARK_OK = 0,
    DSPARK_EINVAL = 1,
    DSPARK_ENOMEM = 2,
    DSPARK_EIO = 3,
    DSPARK_EFORMAT = 4,
    DSPARK_EUNSUPPORTED = 5,
    DSPARK_ERANGE = 6
} dspark_status;

typedef struct {
    dspark_backend backend;
    int threads;                 /* 0 selects min(48, online CPUs). */
} dspark_load_options;

typedef struct {
    size_t max_context_tokens;   /* 0 selects 8192; maximum is 262144. */
} dspark_state_options;

typedef struct {
    size_t count;
    int32_t token_ids[DSPARK_BLOCK_SIZE];
    float confidence[DSPARK_BLOCK_SIZE];
    float selected_logits[DSPARK_BLOCK_SIZE];
} dspark_proposal;

/* Validate configs, index entries, tensor names, dtypes, and shapes only. */
int dspark_validate_files(const char *draft_dir, const char *target_dir,
                          char *error, size_t error_size);

int dspark_model_load(dspark_model **out, const char *draft_dir,
                      const char *target_dir,
                      const dspark_load_options *options,
                      char *error, size_t error_size);
void dspark_model_free(dspark_model *model);

int dspark_state_create(dspark_state **out, const dspark_model *model,
                        const dspark_state_options *options,
                        char *error, size_t error_size);
void dspark_state_free(dspark_state *state);
void dspark_state_reset(dspark_state *state);
int dspark_state_truncate(dspark_state *state, size_t context_tokens);
size_t dspark_state_context_tokens(const dspark_state *state);

/*
 * taps[i] is [n_tokens, stride] FP32 and contains the post-layer output for
 * target layer target_layer_ids[i].  The published model order is
 * {5, 19, 33, 47, 61}.  Appends are committed atomically.
 */
int dspark_state_append_target(dspark_state *state,
                               const float *const taps[DSPARK_TARGET_TAPS],
                               size_t n_tokens, size_t stride,
                               char *error, size_t error_size);

/* Does not change the persistent context cursor or KV caches. */
int dspark_state_propose(dspark_state *state, int32_t anchor_token,
                         dspark_proposal *proposal,
                         char *error, size_t error_size);

const int *dspark_target_layer_ids(const dspark_model *model);
size_t dspark_hidden_size(const dspark_model *model);
size_t dspark_vocab_size(const dspark_model *model);
dspark_backend dspark_model_backend(const dspark_model *model);
const char *dspark_backend_name(dspark_backend backend);

#ifdef __cplusplus
}
#endif
#endif
