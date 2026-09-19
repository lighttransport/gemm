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
#include "../../common/glm5next.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_llm_runner hip_llm_runner;
typedef struct hip_llm_state_snapshot hip_llm_state_snapshot;

typedef enum {
    HIP_LLM_MOE_AUTO = 0,
    HIP_LLM_MOE_HYBRID,
    HIP_LLM_MOE_CPU,
    HIP_LLM_MOE_GPU_STREAM,
} hip_llm_moe_mode;

typedef enum {
    HIP_LLM_KV_AUTO = 0,
    HIP_LLM_KV_F32,
    HIP_LLM_KV_F16,
    /* Explicit llama.cpp-compatible full-attention cache: q8_0 K/q4_0 V. */
    HIP_LLM_KV_Q8_0_Q4_0,
    /* Q8_0 for both K and V; supported by Qwen3.5 hybrid attention. */
    HIP_LLM_KV_Q8_0_Q8_0,
} hip_llm_kv_cache_type;

typedef enum {
    HIP_LLM_DECODE_KERNEL_DEFAULT = 0,
    HIP_LLM_DECODE_KERNEL_NATIVE,
    HIP_LLM_DECODE_KERNEL_DP4A2,
    HIP_LLM_DECODE_KERNEL_AUTO,
} hip_llm_decode_kernel_mode;

typedef enum {
    HIP_LLM_DECODE_LAYOUT_NATIVE = 0,
    HIP_LLM_DECODE_LAYOUT_AUTO_REPACK,
} hip_llm_decode_layout_mode;

typedef struct {
    uint32_t struct_size;
    int max_seq_len;              /* <= 0: model default */
    hip_llm_moe_mode moe_mode;
    uint64_t moe_cache_bytes;     /* 0: consume safe remaining VRAM */
    int moe_cpu_threads;          /* <= 0: physical cores */
    uint64_t host_register_bytes; /* 0: automatic */
    uint64_t gpu_reserve_bytes;   /* 0: default 1 GiB */
    int qwen4_prefill_staging;    /* opt-in two-tier Qwen4 batched prefill */
    uint64_t qwen4_prefill_stage_bytes; /* 0: 512 MiB when staging is enabled */
    hip_llm_kv_cache_type kv_cache_type;
    hip_llm_decode_kernel_mode decode_kernel_mode;
    hip_llm_decode_layout_mode decode_layout_mode;
    const char *decode_layout_cache_path; /* borrowed for the duration of load */
    uint64_t decode_layout_budget_bytes;
    /* Prefill microbatch size. 0 keeps the backend default (512 for Qwen3.5). */
    int prefill_batch_tokens;
    /* 0: quality-safe scalar dispatcher; 1: enable Qwen3.5 batched path. */
    int qwen35_batched_prefill;
    /* Experimental BF16 projection GEMMs during Qwen3.5 prefill only.
     * Changes quantized projection arithmetic; decode remains unchanged. */
    int qwen35_prefill_bf16;
    /* Opt-in graph-safe Q8/Q8 decode; arithmetic matches uncaptured decode. */
    int qwen35_decode_graph;
    /* Diagnostic reference arithmetic profile; whole-model parity is WIP. */
    int qwen35_reference_math;
    /* Native Q8/Q8 D=256 decode using the pinned gfx1201 attention order. */
    int qwen35_native_q8_attention;
    /* Extend native Q8/Q8 attention to prefill (also enables native decode). */
    int qwen35_native_q8_prefill;
    int qwen35_native_q2k; /* precise Q2_K x Q8_1 decode on gfx1201 */
    int qwen35_native_mmvq; /* also IQ2_XXS/XS/S, IQ3_XXS/S with precise Q8_1 staging */
} hip_llm_load_options;

typedef struct {
    uint64_t tokens;
    uint64_t assignments;
    uint64_t gpu_assignments;
    uint64_t cpu_assignments;
    uint64_t skipped_assignments;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t cache_evictions;
    uint64_t h2d_bytes;
    uint64_t stage_h2d_bytes;
    uint64_t stage_waves;
    uint64_t stage_promotions;
    uint64_t stage_fallbacks;
    double cpu_ms;
    double copy_ms;
    double gpu_moe_ms;
} hip_llm_moe_stats;

typedef struct {
    uint32_t struct_size;
    uint64_t total_bytes;
    uint64_t free_bytes;
    uint64_t peak_used_bytes;
} hip_llm_vram_stats;

/* Contract exported by a Qwen4exp NextN/MTP sidecar.  The sidecar shares the
 * trunk embedding and output head, so it must be checked before allocating
 * its independent layer state on the GPU. */
typedef struct {
    int layer_index;
    int n_embd;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int n_experts;
    int n_experts_used;
    int expert_ff;
    int hc_count;
    int hc_low_rank;
} hip_llm_qwen4_nextn_info;

void hip_llm_load_options_default(hip_llm_load_options *options);

/* Validate a standalone Qwen4exp NextN sidecar.  This only inspects GGUF
 * metadata/tensor shapes; it does not allocate device memory. */
int hip_llm_qwen4_nextn_inspect(const gguf_shards *sidecar,
                                 hip_llm_qwen4_nextn_info *info,
                                 char *error, size_t error_cap);

/* Load the complete Qwen4 NextN layer, independent KV state and expert cache.
 * Shared embedding/output weights are borrowed from the loaded trunk. */
int hip_llm_load_qwen4_nextn_fusion(hip_llm_runner *r,
                                    const gguf_shards *sidecar,
                                    char *error, size_t error_cap);
/* Diagnostic oracle interface: hidden arrays contain hc_count*n_embd floats. */
int hip_llm_qwen4_nextn_logits(hip_llm_runner *r, int32_t token, int position,
                               const float *hidden, float *out_hidden, float *logits);
int hip_llm_verify_qwen4_nextn(hip_llm_runner *r, const gguf_shards *sidecar,
                              const gguf_shards *target, int steps);
typedef struct {
    int32_t tokens[33];
    int emitted, accepted, drafted, processed;
    int32_t pending;
    int stopped;
    double draft_ms, verify_ms;
} hip_llm_qwen4_mtp_result;
/* Enable only after sidecar loading. K is supplied per round. */
int hip_llm_qwen4_mtp_enable(hip_llm_runner *r);
/* Before load: reserve NextN/QSA/state memory from the expert-cache budget. */
int hip_llm_qwen4_mtp_configure(hip_llm_runner *r, size_t cache_bytes, int draft);
int hip_llm_qwen4_mtp_set_verify(hip_llm_runner *r, int window);
/* Borrowed host logits for the current committed target state. */
float *hip_llm_current_logits(hip_llm_runner *r);
/* Dense Qwen3.5/3.8 NextN owns its weights and KV. Proposals never advance
 * target state or sampler RNG; the caller must verify every proposed token.
 * The draft starts a fresh KV prefix at the first generation position. */
int hip_llm_qwen35_mtp_load(hip_llm_runner *r, const char *path,
                           char *error, size_t error_cap);
int hip_llm_qwen35_mtp_propose(hip_llm_runner *r, int32_t anchor, int position,
                              int count, int32_t *drafts);
/* Verify a window, then commit exactly its accepted input prefix before any
 * new proposal/target forward. Requires Q8/Q8 KV and the native decode graph.
 * The first window fixes capacity (1..16 rows); later windows may be smaller.
 * Logits are row-major and borrowed until the next verification call.
 * Commit accepts 1..rows inputs, including the anchor. On an execution error
 * reset the target before reuse; partially executed GPU work is not reusable. */
float *hip_llm_qwen35_mtp_verify(hip_llm_runner *r, const int32_t *tokens,
                                int rows, int position);
int hip_llm_qwen35_mtp_commit(hip_llm_runner *r, int processed);

/* Qwen3.8 DFlash2 block-diffusion drafter. Target verification remains exact;
 * only verified target tokens are committed to the caller. */
int hip_llm_qwen35_dflash2_load(hip_llm_runner *r, const char *path,
                               char *error, size_t error_cap);
int hip_llm_qwen35_dflash2_propose(hip_llm_runner *r, int32_t anchor,
                                  int position, int count, int32_t *drafts);
int hip_llm_qwen35_dflash2_commit(hip_llm_runner *r, int position,
                                 int processed);
/* Run target forward and return only the greedy token; avoids a full-vocab
 * device-to-host copy during speculative verification. */
int hip_llm_forward_argmax(hip_llm_runner *r, int32_t token_id, int position);
int hip_llm_qwen4_exact_enable(hip_llm_runner *r);
int hip_llm_verify_qwen4_mtp(hip_llm_runner *r, int32_t anchor, int position, int draft);
int hip_llm_verify_qwen4_qsa(hip_llm_runner *r);
int hip_llm_qwen4_mtp_step(hip_llm_runner *r, int32_t anchor, int position,
                           int draft, int max_emit, const int32_t *stop_ids,
                           int n_stop, hip_llm_qwen4_mtp_result *result);
/* Apply the measured Qwen4 coding-decode routing profile without relying on
 * process environment variables. Call after loading weights. */
void hip_llm_set_qwen4_coding_profile(hip_llm_runner *r);
void hip_llm_set_qwen4_batched_prefill(hip_llm_runner *r, int enabled);
void hip_llm_set_qwen4_batch_request_tokens(hip_llm_runner *r, int n_tokens);

/* Initialize HIP context + compile kernels via HIPRTC for the given device.
 * Returns NULL on failure. verbose: 0=quiet, 1=info, 2=debug */
hip_llm_runner *hip_llm_init(int device_id, int verbose);

/* Load model weights from GGUF onto GPU. max_seq_len <= 0 uses model default.
 * Returns 0 on success, -1 on error. */
int hip_llm_load_weights(hip_llm_runner *r, gguf_context *gguf, int max_seq_len);

/* Load a logical model spanning one or more GGUF shards.  The caller owns the
 * shard mappings and must keep them alive until hip_llm_offload/free. */
int hip_llm_load_weights_sharded(hip_llm_runner *r, gguf_shards *model,
                                 const hip_llm_load_options *options);

int hip_llm_get_moe_stats(const hip_llm_runner *r, hip_llm_moe_stats *stats);
/* Current free/total VRAM plus high-water usage sampled after forwards. */
int hip_llm_get_vram_stats(const hip_llm_runner *r, hip_llm_vram_stats *stats);
void hip_llm_reset_moe_stats(hip_llm_runner *r);
int hip_llm_verify_moe_native(hip_llm_runner *r, int M);
int hip_llm_verify_ssm_projections(hip_llm_runner *r, int M);
int hip_llm_verify_qwen4_ple_split(hip_llm_runner *r, int rows);
int hip_llm_verify_hc_batch(hip_llm_runner *r, int batch,
                            double *out_rel_l2, double *out_max_abs);

/* Verify the standalone GLM5Next KDA recurrent step against its scalar CPU
 * oracle. This does not require model weights to be loaded. */
int hip_llm_verify_glm5next_kda(hip_llm_runner *r, int head_dim,
                                double *out_rel_l2, double *out_max_abs);
int hip_llm_verify_glm5next_kda_heads(hip_llm_runner *r, int n_heads, int head_dim,
                                      double *out_rel_l2, double *out_max_abs);
int hip_llm_verify_glm5next_dsa_attention(hip_llm_runner *r, int n_heads,
                                          int kv_dim, int value_dim, int n_tokens,
                                          double *out_rel_l2, double *out_max_abs);
int hip_llm_verify_glm5next_model_matvec(hip_llm_runner *r, gguf_shards *model,
                                         int layer, double *out_rel_l2,
                                         double *out_max_abs);
/* Stage all three real KDA input projections together and execute them in one
 * stream.  This is the first model-weighted GPU graph boundary; it is kept as
 * a public verifier until the complete staged layer is wired into forward(). */
int hip_llm_verify_glm5next_model_kda_projections(hip_llm_runner *r,
                                                   gguf_shards *model, int layer,
                                                   double *out_rel_l2,
                                                   double *out_max_abs,
                                                   double *out_ms);
int hip_llm_verify_glm5next_model_kda_layer(hip_llm_runner *r, gguf_shards *model,
                                            int layer, double *out_rel_l2,
                                            double *out_max_abs, double *out_ms);
int hip_llm_verify_glm5next_model_dsa_layer(hip_llm_runner *r, gguf_shards *model,
                                            int layer, double *out_rel_l2,
                                            double *out_max_abs, double *out_ms);

/* Load Qwen3 dense weights from a safetensors file (text-encoder path). */
int hip_llm_load_weights_qwen3_safetensors(hip_llm_runner *r, const char *model_path, int max_seq_len);

/* Run one token through the transformer. Returns pointer to F32 hidden state [n_embd].
 * The returned pointer is valid until the next call (host-side buffer). */
float *hip_llm_forward(hip_llm_runner *r, int32_t token_id, int position);

/* Run one token and return logits [n_vocab]. Applies lm_head after hidden state.
 * The returned pointer is valid until the next call (host-side buffer). */
float *hip_llm_forward_logits(hip_llm_runner *r, int32_t token_id, int position);

/* Run the optional GLM5Next speculative/NextN block using the latest trunk
 * hidden state and the token that should be embedded by the draft head.  The
 * returned logits are valid until the next call. */
float *hip_llm_forward_nextn_logits(hip_llm_runner *r, int32_t prev_token, int position);

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
void hip_llm_set_decode_mode(hip_llm_runner *r, int enabled);

/* Save/restore recurrent state at a prompt boundary. KV entries remain in
 * their positional device cache, so this snapshots only hybrid SSM/PLE state.
 * The opaque snapshot is owned by the caller and may be reused across turns. */
hip_llm_state_snapshot *hip_llm_snapshot_state(hip_llm_runner *r);
/* Snapshot only the attention KV slots in [start_pos, start_pos+n_positions).
 * This bounded transaction is intended for speculative draft verification. */
hip_llm_state_snapshot *hip_llm_snapshot_state_window(hip_llm_runner *r,
                                                       int start_pos,
                                                       int n_positions);
int hip_llm_restore_state(hip_llm_runner *r, const hip_llm_state_snapshot *snapshot);
void hip_llm_free_state_snapshot(hip_llm_state_snapshot *snapshot);

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

/* Verify batched MoE top-k routing, including expert ids above 255. */
int hip_llm_verify_moe_routing(hip_llm_runner *r, int n_experts, int n_used);

/* Query model dimensions (valid after load_weights). */
int hip_llm_n_embd(const hip_llm_runner *r);
int hip_llm_n_layers(const hip_llm_runner *r);
int hip_llm_n_vocab(const hip_llm_runner *r);
int hip_llm_max_seq_len(const hip_llm_runner *r);

/* Inspect the GLM5Next GGUF contract without initializing HIP.  This is used
 * by CPU bring-up and by launchers to reject an incomplete model cleanly. */
int hip_llm_glm5next_inspect(gguf_shards *model, glm5next_config *config,
                             glm5next_state_layout *layout, int max_seq_len,
                             char *error, size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif /* HIP_LLM_RUNNER_H */
