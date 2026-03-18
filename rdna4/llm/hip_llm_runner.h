/*
 * hip_llm_runner.h - HIP/ROCm LLM inference runner for Qwen3-style transformers
 *
 * Uses HIPRTC to compile HIP C kernels at runtime (no hipcc needed).
 * Supports F16 weights on GPU, F32 compute. Targets RDNA4.
 *
 * Usage:
 *   hip_llm_runner *r = hip_llm_init(0, 1);
 *   hip_llm_load_weights(r, gguf, 4096);
 *   float *out = hip_llm_forward(r, token_id, position);
 *   hip_llm_free(r);
 */
#ifndef HIP_LLM_RUNNER_H
#define HIP_LLM_RUNNER_H

#include <stdint.h>
#include "../../common/gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_llm_runner hip_llm_runner;

/* Initialize HIP context + compile kernels via HIPRTC for the given device.
 * Returns NULL on failure. verbose: 0=quiet, 1=info, 2=debug */
hip_llm_runner *hip_llm_init(int device_id, int verbose);

/* Load model weights from GGUF onto GPU. max_seq_len <= 0 uses model default.
 * Returns 0 on success, -1 on error. */
int hip_llm_load_weights(hip_llm_runner *r, gguf_context *gguf, int max_seq_len);

/* Run one token through the transformer. Returns pointer to F32 hidden state [n_embd].
 * The returned pointer is valid until the next call (host-side buffer). */
float *hip_llm_forward(hip_llm_runner *r, int32_t token_id, int position);

/* Run one token and return logits [n_vocab]. Applies lm_head after hidden state.
 * The returned pointer is valid until the next call (host-side buffer). */
float *hip_llm_forward_logits(hip_llm_runner *r, int32_t token_id, int position);

/* Run forward pass with a pre-computed F32 embedding [n_embd] instead of token lookup.
 * Used to inject vision embeddings. embd_stride is the stride between embeddings
 * (>= n_embd; extra data used for deepstack injection). */
float *hip_llm_forward_embd(hip_llm_runner *r, const float *embd, int embd_stride, int position);
float *hip_llm_forward_embd_logits(hip_llm_runner *r, const float *embd, int embd_stride, int position);

/* Free all GPU resources and the runner. */
void hip_llm_free(hip_llm_runner *r);

/* Reset all SSM state (conv + recurrent). Call between conversations for hybrid models. */
void hip_llm_reset_state(hip_llm_runner *r);

/* Read last hidden state (d_x) from GPU into dst. n = n_embd. */
int hip_llm_read_hidden(const hip_llm_runner *r, float *dst, int n);

/* Enable per-layer debug output (print hidden state norm after each layer). */
void hip_llm_set_debug(hip_llm_runner *r, int debug_layers);

/* Set max layers to process (0 = all). For debugging: run only first N layers. */
void hip_llm_set_max_layers(hip_llm_runner *r, int max_layers);

/* Query model dimensions (valid after load_weights). */
int hip_llm_n_embd(const hip_llm_runner *r);
int hip_llm_n_layers(const hip_llm_runner *r);
int hip_llm_n_vocab(const hip_llm_runner *r);
int hip_llm_max_seq_len(const hip_llm_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* HIP_LLM_RUNNER_H */
