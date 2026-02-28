/*
 * cuda_llm_runner.h - CUDA LLM inference runner for Qwen3-style transformers
 *
 * Uses NVRTC to compile CUDA C kernels at runtime (no nvcc needed).
 * Supports F16 weights on GPU, F32 compute. Works on sm_70+ (V100 through Blackwell).
 *
 * Usage:
 *   cuda_llm_runner *r = cuda_llm_init(0, 1);
 *   cuda_llm_load_weights(r, gguf, 4096);
 *   float *out = cuda_llm_forward(r, token_id, position);
 *   cuda_llm_free(r);
 */
#ifndef CUDA_LLM_RUNNER_H
#define CUDA_LLM_RUNNER_H

#include <stdint.h>
#include "../../common/gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_llm_runner cuda_llm_runner;

/* Initialize CUDA context + compile kernels via NVRTC for the given device.
 * Returns NULL on failure. verbose: 0=quiet, 1=info, 2=debug */
cuda_llm_runner *cuda_llm_init(int device_id, int verbose);

/* Load model weights from GGUF onto GPU. max_seq_len <= 0 uses model default.
 * Returns 0 on success, -1 on error. */
int cuda_llm_load_weights(cuda_llm_runner *r, gguf_context *gguf, int max_seq_len);

/* Run one token through the transformer. Returns pointer to F32 hidden state [n_embd].
 * The returned pointer is valid until the next call (host-side buffer). */
float *cuda_llm_forward(cuda_llm_runner *r, int32_t token_id, int position);

/* Run one token and return logits [n_vocab]. Applies lm_head after hidden state.
 * The returned pointer is valid until the next call (host-side buffer). */
float *cuda_llm_forward_logits(cuda_llm_runner *r, int32_t token_id, int position);

/* Run forward pass with a pre-computed F32 embedding [n_embd] instead of token lookup.
 * Used to inject vision embeddings. embd_stride is the stride between embeddings
 * (>= n_embd; extra data used for deepstack injection). */
float *cuda_llm_forward_embd(cuda_llm_runner *r, const float *embd, int embd_stride, int position);
float *cuda_llm_forward_embd_logits(cuda_llm_runner *r, const float *embd, int embd_stride, int position);

/* Free all GPU resources and the runner. */
void cuda_llm_free(cuda_llm_runner *r);

/* Reset all SSM state (conv + recurrent). Call between conversations for hybrid models. */
void cuda_llm_reset_state(cuda_llm_runner *r);

/* Read last hidden state (d_x) from GPU into dst. n = n_embd. */
int cuda_llm_read_hidden(const cuda_llm_runner *r, float *dst, int n);

/* Enable per-layer debug output (print hidden state norm after each layer). */
void cuda_llm_set_debug(cuda_llm_runner *r, int debug_layers);

/* Set max layers to process (0 = all). For debugging: run only first N layers. */
void cuda_llm_set_max_layers(cuda_llm_runner *r, int max_layers);

/* Query model dimensions (valid after load_weights). */
int cuda_llm_n_embd(const cuda_llm_runner *r);
int cuda_llm_n_layers(const cuda_llm_runner *r);
int cuda_llm_n_vocab(const cuda_llm_runner *r);
int cuda_llm_max_seq_len(const cuda_llm_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_LLM_RUNNER_H */
