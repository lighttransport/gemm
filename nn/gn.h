/* SPDX-License-Identifier: MIT
 * Native policy/value network. All tensor interfaces are contiguous NHWC.
 * This implementation is independent of shogi engines and ML frameworks. */
#ifndef GEMM_GN_H
#define GEMM_GN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct gn_model gn_model;
typedef struct {
    uint32_t version, side, inputs, actions, channels, blocks;
    uint32_t attention_every, head_dim, value_channels, value_hidden;
    uint64_t seed;
    size_t memory_limit;
} gn_config;
typedef struct {
    float policy, value, grad_norm;
    uint64_t step;
} gn_metrics;
gn_config gn_default_config(void);
/* backend: cpu (FP32), cuda, hip; cuda-fp32/hip-fp32 are diagnostic paths.
 * Unsupported backends fail explicitly; there is no silent CPU fallback. */
gn_model *gn_create(const gn_config *config, const char *backend, int device);
void gn_destroy(gn_model *model);
const char *gn_error(void);
const gn_config *gn_configuration(const gn_model *model);
size_t gn_parameter_count(const gn_model *model);
size_t gn_memory_used(const gn_model *model);
uint64_t gn_step(const gn_model *model);
/* policy [batch, side*side, actions], wdl [batch,3] probabilities. */
int gn_infer(gn_model *, size_t batch, const float *input, float *policy, float *wdl);
/* Policy targets: -1 means illegal; nonnegative probabilities sum to one.
 * WDL targets: 0=win, 1=draw, 2=loss, from the player-to-move perspective.
 * backward accumulates parameter gradients; update averages accumulated batches
 * by sample count, clips global norm, and applies decoupled AdamW. */
int gn_backward(gn_model *, size_t batch, const float *input, const float *policy_targets,
                const uint32_t *wdl_targets, gn_metrics *metrics);
int gn_update(gn_model *, float learning_rate, float weight_decay, float clip_norm,
              gn_metrics *metrics);
void gn_zero_grad(gn_model *);
int gn_save(const gn_model *, const char *path);
gn_model *gn_load(const char *path, const char *backend, int device);
/* Diagnostics and independent reference validation. */
size_t gn_tensor_count(const gn_model *);
/* Replay sampling shares the checkpointed RNG state. */
uint64_t gn_random(gn_model *);
const char *gn_tensor(gn_model *, size_t index, size_t *rows, size_t *cols, float **data,
                      float **gradient);
#ifdef __cplusplus
}
#endif
#endif
