/*
 * hip_qwen35_st.h - Qwen3.5 (hybrid Gated-DeltaNet + gated full attention, e.g. Qwen3.5-2B)
 * text tower on RDNA4 / ROCm, loaded DIRECTLY from HF safetensors (no GGUF).
 *
 * HIP port of cuda_llm_load_weights_qwen35_safetensors + cuda_llm_prefill_hidden_ex
 * (cuda/llm, see cuda/llm/QWEN35_ST.md).  Used as the praxis-1-droid text encoder:
 * cond = hidden_states[-3] (output of layer index 21, pre final norm), all tokens.
 *
 * Self-contained: HIPRTC kernels (WMMA f16 GEMM, fp32 accumulate, fp32 activations),
 * linked with rdna4/rocew.c only.  Does not touch the hip_llm_runner monolith.
 */
#ifndef HIP_QWEN35_ST_H
#define HIP_QWEN35_ST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_qwen35 hip_qwen35;

/* device_id: HIP device. verbose: 0 quiet, 1 info, 2 debug. NULL on failure. */
hip_qwen35 *hip_qwen35_init(int device_id, int verbose);

/* Load Qwen3.5 text tower from an HF safetensors dir (config.json + model*.safetensors,
 * tensors model.language_model.*).  bf16 weights -> f16 on the GPU, RMSNorm weights +1,
 * A_log -> -exp.  max_seq_len bounds start_pos + n_tokens (KV cache size). 0 on success. */
int hip_qwen35_load_safetensors(hip_qwen35 *r, const char *model_path, int max_seq_len);

/* Run the first n_run_layers (<=0: all) layers on token_ids.
 *  start_pos == 0 resets the recurrent/conv state, > 0 continues a previous prefill.
 *  dump_only_slot >= 0: out_per_layer receives ONLY that slot [n_tokens][n_embd]
 *     (0 = embeddings, k = output of layer k-1); < 0: all [(n_run+1)][n_tokens][n_embd].
 *  out_normed (optional): [n_tokens][n_embd] = output of the last run layer after final RMSNorm.
 * Returns 0 on success. */
int hip_qwen35_prefill_hidden_ex(hip_qwen35 *r, const int32_t *token_ids, int n_tokens,
                                 int n_run_layers, int start_pos, int dump_only_slot,
                                 float *out_per_layer, float *out_normed);

/* Convenience: cond = hidden_states[-3] (22 layers, slot 22, pre-norm) for n_tokens ids
 * (caller pads), out [n_tokens][n_embd]. */
int hip_qwen35_cond(hip_qwen35 *r, const int32_t *token_ids, int n_tokens, float *out);

int  hip_qwen35_n_embd(const hip_qwen35 *r);
int  hip_qwen35_n_layers(const hip_qwen35 *r);
int  hip_qwen35_max_seq_len(const hip_qwen35 *r);
void hip_qwen35_free(hip_qwen35 *r);

#ifdef __cplusplus
}
#endif
#endif
