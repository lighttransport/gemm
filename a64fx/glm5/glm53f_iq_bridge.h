#ifndef GLM53F_IQ_BRIDGE_H
#define GLM53F_IQ_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

typedef struct glm53f_iq_cache glm53f_iq_cache;
/* One cache per owning MoE context; serialize calls using the same cache.
 * NULL is a supported no-cache fallback. Limits include the lookup table. */
glm53f_iq_cache *glm53f_iq_cache_create(size_t byte_limit);
void glm53f_iq_cache_free(glm53f_iq_cache *cache);
size_t glm53f_iq_cache_bytes(const glm53f_iq_cache *cache);

enum {
    GLM53F_GGML_Q2_K = 10,
    GLM53F_GGML_Q3_K = 11,
    GLM53F_GGML_Q4_K = 12,
    GLM53F_GGML_Q5_K = 13,
    GLM53F_GGML_Q6_K = 14,
    GLM53F_GGML_IQ2_XS = 17,
    GLM53F_GGML_IQ3_XXS = 18,
    GLM53F_GGML_IQ4_XS = 23
};

typedef struct {
    const uint8_t *gate_up;
    const uint8_t *down;
    int gate_type;
    int down_type;
    int inter;
} glm53f_iq_part;

int glm53f_iq_type_supported(int type);
int glm53f_iq_expert_weighted(
    float *output, const glm53f_iq_part *parts, const float *weights,
    int count, const float *input, float *gate_up, float *activation);

/* Token-major arrays use nine expert slots per token. Scratch capacities:
 * tokens*9*1024 gate/up, tokens*9*512 activation, tokens*9*4096 results. */
int glm53f_iq_expert_weighted_batch(float *output,
    const glm53f_iq_part *parts, const float *weights, const int *counts,
    int tokens, const float *input, float *gate_up, float *activation,
    float *expert_output);
int glm53f_iq_expert_weighted_batch_cached(float *output,
    const glm53f_iq_part *parts, const float *weights, const int *counts,
    int tokens, const float *input, float *gate_up, float *activation,
    float *expert_output, glm53f_iq_cache *cache);

#endif
