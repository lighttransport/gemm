// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan LLM Runner - GPU-accelerated transformer inference for Qwen2/3-style models.
// Mirrors the CPU reference in common/transformer.h.
//
#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

#include "deps/vulkan-runner.hh"

extern "C" {
#include "../common/gguf_loader.h"
#include "../common/ggml_dequant.h"
#include "../common/transformer.h"
}

class VulkanLLMRunner {
public:
    VulkanLLMRunner();
    ~VulkanLLMRunner();

    // Initialize Vulkan device
    bool initialize(int device_id = 0, bool verbose = false);

    // Load weights from GGUF model file
    bool loadWeights(gguf_context *g);

    // Forward pass: single token decode with M-RoPE positions.
    // Returns pointer to logits [n_vocab] (valid until next call).
    // If compute_logits is false, returns hidden state [n_embd].
    float *forward(int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w,
                   bool compute_logits = true);

    // Forward pass with pre-computed embedding (for vision tokens)
    float *forwardEmbd(const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w,
                       bool compute_logits = true);

    // Simple forward (all positions same)
    float *forward(int32_t token_id, int position, bool compute_logits = true) {
        return forward(token_id, position, position, position, position, compute_logits);
    }

    // Enable batched prefill (allocates scratch buffers)
    bool enablePrefill(int max_batch_size);

    // Batched prefill from token IDs. Returns logits of last token.
    float *prefillTokens(const int32_t *tokens, int n_tokens,
                         const int *cache_pos, const int *pos_t,
                         const int *pos_h, const int *pos_w);

    // Batched prefill from embeddings. Returns logits of last token.
    // embd: [n_tokens * embd_stride], ds_embds: optional deepstack pointers per token.
    float *prefillEmbds(const float *embd, int n_tokens, int embd_stride,
                        const int *cache_pos, const int *pos_t,
                        const int *pos_h, const int *pos_w,
                        const float **ds_embds = nullptr, int ds_embd_stride = 0);

    // Set deepstack embedding stride (call before vision token forwardEmbd calls)
    void setDeepstackStride(int stride) {
        ds_embd_stride_ = stride;
        if (cpu_model_) cpu_model_->ds_embd_stride = stride;
    }

    std::string getLastError() const { return last_error_; }
    std::string getDeviceName() const;

    // Model parameters (public for test access)
    int n_layers() const { return n_layers_; }
    int n_embd() const { return n_embd_; }
    int n_vocab() const { return n_vocab_; }
    int n_heads() const { return n_heads_; }
    int n_kv_heads() const { return n_kv_heads_; }
    int head_dim() const { return head_dim_; }

private:
    using Runner = vl_cpp::vulkan::VulkanComputeRunner;
    using Pipeline = Runner::ComputePipeline;
    using BufInfo = Runner::BufferInfo;

    Runner runner_;
    bool initialized_ = false;
    bool weights_loaded_ = false;
    bool cpu_fallback_moe_ = false;
    bool fused_mode_ = false;  // when true, dispatches skip begin/end/wait

    // Helper: bind pipeline and descriptors (uses dynamic descriptors in fused mode)
    void bindPipelineAndDescriptors(Pipeline &pipe, const std::vector<BufInfo> &bufs);
    transformer_model *cpu_model_ = nullptr;
    std::string last_error_;
    std::string shader_dir_;

    // Model hyperparameters
    int n_layers_ = 0;
    int n_embd_ = 0;
    int n_heads_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int n_ff_ = 0;
    int n_ff_expert_ = 0;
    int n_expert_ = 0;
    int n_expert_used_ = 0;
    bool use_moe_ = false;
    int n_vocab_ = 0;
    int max_seq_len_ = 0;
    float rope_freq_base_ = 5000000.0f;
    float rms_norm_eps_ = 1e-6f;
    int use_mrope_ = 0;
    int mrope_sections_[4] = {};
    int n_deepstack_ = 0;
    int ds_embd_stride_ = 0;
    bool ds_active_ = false;
    BufInfo buf_ds_slices_[3] = {};  // up to 3 deepstack slices

    // Pipelines
    Pipeline pipe_rmsnorm_;
    Pipeline pipe_matvec_;
    Pipeline pipe_matvec_q8_0_;
    Pipeline pipe_matvec_f16_;
    Pipeline pipe_silu_mul_;
    Pipeline pipe_rope_neox_;
    Pipeline pipe_rope_mrope_;
    Pipeline pipe_rope_neox_batch_;
    Pipeline pipe_rope_mrope_batch_;
    Pipeline pipe_qknorm_;
    Pipeline pipe_kv_store_;
    Pipeline pipe_kv_store_batch_;
    Pipeline pipe_attn_decode_;
    Pipeline pipe_attn_prefill_;
    Pipeline pipe_add_;
    Pipeline pipe_dequant_q8_0_;
    Pipeline pipe_matmul_coopmat_nt_;
    bool pipelines_created_ = false;

    // Weight buffers on GPU
    std::map<std::string, BufInfo> weight_bufs_;
    std::map<std::string, uint32_t> weight_types_;  // GGML type per weight (0=F32, GGML_TYPE_Q8_0=raw)

    // Per-layer weight buffer names
    struct LayerWeightNames {
        std::string attn_norm, attn_q, attn_k, attn_v;
        std::string attn_q_norm, attn_k_norm;
        std::string attn_output;
        std::string ffn_norm, ffn_gate, ffn_up, ffn_down;
        std::string ffn_gate_inp, ffn_up_exps, ffn_gate_exps, ffn_down_exps;
    };
    std::vector<LayerWeightNames> layer_names_;

    // Scratch buffers on GPU (persistent across forward calls)
    BufInfo buf_x_;        // [n_embd]
    BufInfo buf_xb_;       // [n_embd]
    BufInfo buf_xb2_;      // [n_embd]
    BufInfo buf_q_;        // [n_embd] (n_heads * head_dim)
    BufInfo buf_k_;        // [kv_dim]
    BufInfo buf_v_;        // [kv_dim]
    BufInfo buf_ffn_gate_; // [n_ff]
    BufInfo buf_ffn_up_;   // [n_ff]
    BufInfo buf_moe_router_; // [n_expert]
    BufInfo buf_logits_;   // [n_vocab]
    bool scratch_created_ = false;

    // KV cache buffers on GPU: [n_layers] each [max_seq_len * kv_dim]
    std::vector<BufInfo> kv_key_bufs_;
    std::vector<BufInfo> kv_val_bufs_;

    // Prefill batch scratch buffers
    int max_prefill_tokens_ = 0;
    bool prefill_enabled_ = false;
    BufInfo buf_pfx_;          // [max_prefill * n_embd]
    BufInfo buf_pfxb_;         // [max_prefill * n_embd]
    BufInfo buf_pfq_;          // [max_prefill * q_dim]
    BufInfo buf_pfk_;          // [max_prefill * kv_dim]
    BufInfo buf_pfv_;          // [max_prefill * kv_dim]
    BufInfo buf_pf_attn_out_;  // [max_prefill * q_dim]
    BufInfo buf_pf_ffn_gate_;  // [max_prefill * n_ff]
    BufInfo buf_pf_ffn_up_;    // [max_prefill * n_ff]
    BufInfo buf_positions_;    // [max_prefill * 3] (t,h,w)
    BufInfo buf_cache_pos_;    // [max_prefill]
    BufInfo buf_weight_f16_;   // 96MB scratch for dequanted weights

    // CPU-side logits for download
    std::vector<float> logits_cpu_;
    std::vector<float> hidden_cpu_;
    std::vector<float> moe_router_cpu_;
    std::vector<float> moe_accum_cpu_;
    std::vector<float> moe_tmp_cpu_;

    // Helper: dequantize and upload tensor
    bool uploadTensor(const std::string &name, const qtensor &t);
    bool uploadTensorRaw(const std::string &name, const qtensor &t);
    bool uploadTensorF16(const std::string &name, const qtensor &t);
    std::vector<float> dequantFull(const qtensor &t);

    // Create pipelines
    bool createPipelines();

    // Buffer helpers
    BufInfo createGpuBuffer(size_t size);
    void destroyGpuBuffer(BufInfo &buf);
    bool uploadToBuffer(BufInfo &buf, const void *data, size_t size);
    bool downloadFromBuffer(const BufInfo &buf, void *data, size_t size);

    // Dispatch helpers
    bool dispatchRmsNorm(const BufInfo &src, BufInfo &dst, const BufInfo &weight,
                         uint32_t n_tokens, uint32_t dim, float eps);
    bool dispatchMatvec(const BufInfo &x, const BufInfo &W, BufInfo &dst,
                        uint32_t N, uint32_t K, uint32_t row_offset = 0);
    bool dispatchSiluMul(BufInfo &gate, const BufInfo &up, uint32_t n);
    bool dispatchRopeNeox(BufInfo &vec, uint32_t n_heads, uint32_t head_dim,
                          int position, float freq_base);
    bool dispatchRopeMrope(BufInfo &vec, uint32_t n_heads, uint32_t head_dim,
                           int pos_t, int pos_h, int pos_w, float freq_base,
                           int sect0, int sect1, int sect2);
    bool dispatchQkNorm(BufInfo &vec, const BufInfo &weight,
                        uint32_t n_heads, uint32_t head_dim, float eps);
    bool dispatchKvStore(const BufInfo &k_vec, const BufInfo &v_vec,
                         BufInfo &k_cache, BufInfo &v_cache,
                         uint32_t cache_pos, uint32_t kv_dim);
    bool dispatchAttnDecode(const BufInfo &q, const BufInfo &k_cache,
                            const BufInfo &v_cache, BufInfo &out,
                            uint32_t n_heads, uint32_t n_kv_heads,
                            uint32_t head_dim, uint32_t seq_len, float scale);
    bool dispatchAdd(BufInfo &dst, const BufInfo &src, uint32_t n);

    // Auto-dispatch matvec: picks Q8_0 or F32 based on weight_types_
    bool dispatchMatvecAuto(const BufInfo &x, const std::string &w_name, BufInfo &dst,
                            uint32_t N, uint32_t K, uint32_t row_offset = 0);

    // F16 matvec dispatch
    bool dispatchMatvecF16(const BufInfo &x, const BufInfo &W_f16, BufInfo &dst,
                           uint32_t N, uint32_t K, uint32_t row_offset = 0);

    // Q8_0 matvec dispatch
    bool dispatchMatvecQ8(const BufInfo &x, const BufInfo &W_raw, BufInfo &dst,
                          uint32_t N, uint32_t K, uint32_t row_offset = 0);

    // Dequant Q8_0 -> F16
    bool dispatchDequantQ8(const BufInfo &src, BufInfo &dst,
                           uint32_t n_rows, uint32_t n_cols);

    // Coopmat NT GEMM: C[M,N] = A[M,K] * B^T[N,K]
    bool dispatchMatmulCoopmatNT(const BufInfo &A, const BufInfo &B, BufInfo &C,
                                  uint32_t M, uint32_t N, uint32_t K);

    // Batched RoPE
    bool dispatchRopeNeoxBatch(BufInfo &vec, const BufInfo &positions,
                               uint32_t n_tokens, uint32_t n_heads,
                               uint32_t head_dim, float freq_base);
    bool dispatchRopeMropeBatch(BufInfo &vec, const BufInfo &positions,
                                uint32_t n_tokens, uint32_t n_heads,
                                uint32_t head_dim, float freq_base,
                                int sect0, int sect1, int sect2);

    // Batched KV store
    bool dispatchKvStoreBatch(const BufInfo &k, const BufInfo &v,
                              BufInfo &k_cache, BufInfo &v_cache,
                              const BufInfo &cache_positions,
                              uint32_t n_tokens, uint32_t kv_dim);

    // Prefill attention
    bool dispatchAttnPrefill(const BufInfo &q, const BufInfo &k_cache,
                             const BufInfo &v_cache, BufInfo &out,
                             const BufInfo &cache_positions,
                             uint32_t n_tokens, uint32_t n_heads,
                             uint32_t n_kv_heads, uint32_t head_dim,
                             uint32_t max_seq_len, float scale);

    // Internal forward pass (after x_ is loaded)
    float *forwardInternal(int cache_pos, int pos_t, int pos_h, int pos_w,
                           bool compute_logits);

    // Batched forward pass
    float *forwardBatch(int n_tokens, const int *cache_positions,
                        const int *pos_t, const int *pos_h, const int *pos_w,
                        bool compute_logits);
};
