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

/* Load Qwen3 dense weights from a safetensors file (text-encoder path). */
int hip_llm_load_weights_qwen3_safetensors(hip_llm_runner *r, const char *model_path, int max_seq_len);

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

/* Batched embedding forward: feed M pre-computed embeddings through the
 * batched prefill path in one shot (positions [position_start, +M)). Same
 * row layout as hip_llm_forward_embd — first n_embd floats per row are the
 * main embedding, remainder (when embd_stride > n_embd) are deepstack slices.
 * Falls back to per-token loop if M doesn't fit the batched path. Returns
 * the last row's hidden state [n_embd] (host-side buffer, valid until next
 * call). */
float *hip_llm_forward_batch_embd(hip_llm_runner *r, const float *embds,
                                    int M, int embd_stride, int position_start);

/* Batched forward over a contiguous run of n_tokens starting at position_start.
 * Returns the LAST token's hidden state [n_embd] (or logits [n_vocab] for the
 * _logits variant). Buffer is host-side, valid until the next call. Phase 1 is
 * a thin wrapper over per-token forward; later phases route to a true batched
 * (WMMA GEMM + flash-attention) path when n_tokens is large. */
float *hip_llm_forward_batch(hip_llm_runner *r, const int32_t *tokens,
                             int n_tokens, int position_start);
float *hip_llm_forward_batch_logits(hip_llm_runner *r, const int32_t *tokens,
                                    int n_tokens, int position_start);

/* Free all GPU resources and the runner. */
void hip_llm_free(hip_llm_runner *r);

/* Free only GPU weight/activation buffers (keep module, stream, context alive).
 * Call this to reclaim VRAM for another model while keeping the HIP context valid. */
void hip_llm_offload(hip_llm_runner *r);

/* Reset all SSM state (conv + recurrent). Call between conversations for hybrid models. */
void hip_llm_reset_state(hip_llm_runner *r);

/* Read last hidden state (d_x) from GPU into dst. n = n_embd. */
int hip_llm_read_hidden(const hip_llm_runner *r, float *dst, int n);

/* Text-encoder hidden snapshots: select up to 3 layers, then read their
 * captured per-token hidden states after each forward. */
int hip_llm_set_hidden_snapshot_layers(hip_llm_runner *r, const int *layers, int n_slots);
int hip_llm_read_hidden_snapshots(const hip_llm_runner *r, float *dst, int n_slots, int n);

/* Enable per-layer debug output (print hidden state norm after each layer). */
void hip_llm_set_debug(hip_llm_runner *r, int debug_layers);

/* Set max layers to process (0 = all). For debugging: run only first N layers. */
void hip_llm_set_max_layers(hip_llm_runner *r, int max_layers);

/* Phase 2: runtime toggle for the batched dense (hipBLASLt) prefill path. When
 * 0, hip_llm_forward_batch{,_logits} fall back to per-token loop. When 1 (the
 * default if eligible at load), they route through the batched path for
 * n_tokens >= LLM_GEMM_M_THRESHOLD. */
void hip_llm_set_batched_path(hip_llm_runner *r, int enable);
int  hip_llm_batched_path_available(const hip_llm_runner *r);

/* A/B verify the GPU matvec kernel for `weight_type` against a CPU reference.
 *
 * Generates a deterministic random raw block-quant matrix [n_rows, n_cols]
 * (runner layout for `weight_type`; Q8_0 uses the runner's padded 36-byte
 * block) and an F32 input vector x[n_cols], runs
 * the HIP matvec, then dequantizes the same bytes on CPU via `cpu_dequant_row`
 * and does a scalar dot-product per row. Reports rel-L2 vs the CPU reference
 * and the max absolute per-row error.
 *
 * `cpu_dequant_row` must dequantize exactly `n` elements (= n_cols here) of
 * `weight_type` from `src` into `dst`. Caller supplies the function pointer
 * (typically `dequantize_row_<type>` from common/ggml_dequant.h) so the runner
 * doesn't link in the full dequant implementation set.
 *
 * Constraint: n_cols must be a multiple of the block-element count for
 * `weight_type` (32 for IQ4_NL etc.; 256 for K-/IQ-super-block types).
 *
 * No model needs to be loaded — only `hip_llm_init` must have succeeded so
 * the HIPRTC module is available. Returns 0 on success (out values written),
 * -1 if the type has no registered HIP matvec or shape is invalid, -2 on
 * any HIP allocation/launch error. */
int hip_llm_verify_quant_matvec(
        hip_llm_runner *r, int weight_type,
        void (*cpu_dequant_row)(const void *src, float *dst, int n),
        int n_rows, int n_cols,
        double *out_rel_l2, double *out_max_abs);

/* Microbenchmark the GPU matvec kernel for `weight_type` on deterministic
 * random raw block-quant bytes. No model needs to be loaded. Reports average
 * HIP event time per launch in milliseconds. */
int hip_llm_bench_quant_matvec(
        hip_llm_runner *r, int weight_type,
        int n_rows, int n_cols,
        int warmup, int iters,
        float *out_ms);

/* Query model dimensions (valid after load_weights). */
int hip_llm_n_embd(const hip_llm_runner *r);
int hip_llm_n_layers(const hip_llm_runner *r);
int hip_llm_n_vocab(const hip_llm_runner *r);
int hip_llm_max_seq_len(const hip_llm_runner *r);

#ifdef __cplusplus
}
#endif

#endif /* HIP_LLM_RUNNER_H */
