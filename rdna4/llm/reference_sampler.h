#ifndef HIP_LLM_REFERENCE_SAMPLER_H
#define HIP_LLM_REFERENCE_SAMPLER_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct hllm_sampler hllm_sampler;
typedef struct {
    uint32_t seed;
    int top_k;                 /* <=0 disables top-k; no implicit cap */
    float top_p, min_p, temperature;
    int penalty_last_n;        /* 0 disables history */
    float repetition, frequency, presence;
} hllm_sampler_config;

void hllm_sampler_defaults(hllm_sampler_config *config);
hllm_sampler *hllm_sampler_create(const hllm_sampler_config *config, int n_vocab);
hllm_sampler *hllm_sampler_clone(const hllm_sampler *sampler);
void hllm_sampler_free(hllm_sampler *sampler);
void hllm_sampler_reset(hllm_sampler *sampler);
/* Selection advances RNG only. Accept explicitly commits penalty history. */
int hllm_sampler_sample(hllm_sampler *sampler, const float *logits);
int hllm_sampler_accept(hllm_sampler *sampler, int token);
/* Diagnostic processed candidate order, logits and probabilities. */
int hllm_sampler_candidates(const hllm_sampler *sampler, int *ids,
                            float *logits, float *probabilities, int capacity);
#ifdef __cplusplus
}
#endif
#endif
