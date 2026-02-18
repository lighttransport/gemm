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
}

#ifndef TRANSFORMER_H
typedef struct {
    void    *data;
    uint32_t type;
    int      n_rows;
    int      n_cols;
} qtensor;
#endif

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

    // Set deepstack embedding stride (call before vision token forwardEmbd calls)
    void setDeepstackStride(int stride) { ds_embd_stride_ = stride; }

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
    std::string last_error_;
    std::string shader_dir_;

    // Model hyperparameters
    int n_layers_ = 0;
    int n_embd_ = 0;
    int n_heads_ = 0;
    int n_kv_heads_ = 0;
    int head_dim_ = 0;
    int n_ff_ = 0;
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
    Pipeline pipe_silu_mul_;
    Pipeline pipe_rope_neox_;
    Pipeline pipe_rope_mrope_;
    Pipeline pipe_qknorm_;
    Pipeline pipe_kv_store_;
    Pipeline pipe_attn_decode_;
    Pipeline pipe_add_;
    bool pipelines_created_ = false;

    // Weight buffers on GPU
    std::map<std::string, BufInfo> weight_bufs_;

    // Per-layer weight buffer names
    struct LayerWeightNames {
        std::string attn_norm, attn_q, attn_k, attn_v;
        std::string attn_q_norm, attn_k_norm;
        std::string attn_output;
        std::string ffn_norm, ffn_gate, ffn_up, ffn_down;
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
    BufInfo buf_logits_;   // [n_vocab]
    bool scratch_created_ = false;

    // KV cache buffers on GPU: [n_layers] each [max_seq_len * kv_dim]
    std::vector<BufInfo> kv_key_bufs_;
    std::vector<BufInfo> kv_val_bufs_;

    // CPU-side logits for download
    std::vector<float> logits_cpu_;
    std::vector<float> hidden_cpu_;

    // Helper: dequantize and upload tensor
    bool uploadTensor(const std::string &name, const qtensor &t);
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
                        uint32_t N, uint32_t K);
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

    // Internal forward pass (after x_ is loaded)
    float *forwardInternal(int cache_pos, int pos_t, int pos_h, int pos_w,
                           bool compute_logits);
};
