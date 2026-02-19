// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan Vision Encoder for Qwen3-VL
// GPU-accelerated ViT inference using Vulkan compute shaders.
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

// qtensor is defined in transformer.h; replicate here to avoid pulling in full transformer
#ifndef TRANSFORMER_H
typedef struct {
    void    *data;
    uint32_t type;
    int      n_rows;
    int      n_cols;
} qtensor;
#endif

class VulkanVisionEncoder {
public:
    VulkanVisionEncoder();
    ~VulkanVisionEncoder();

    // Initialize Vulkan device
    bool initialize(int device_id = 0, bool verbose = false);

    // Load weights from GGUF mmproj file
    bool loadWeights(gguf_context *g);

    enum AttentionMode { ATTN_CPU = 0, ATTN_NAIVE_GPU = 1, ATTN_FLASH_GPU = 2 };

    // Set attention implementation (default: ATTN_FLASH_GPU)
    void setAttentionMode(AttentionMode mode) { attn_mode_ = mode; }

    // Encode image. rgb_norm is [height * width * 3] normalized float RGB.
    // Returns result buffer (caller must free with delete[]).
    float *encode(const float *rgb_norm, int width, int height);

    std::string getLastError() const { return last_error_; }
    std::string getDeviceName() const;

private:
    using Runner = vl_cpp::vulkan::VulkanComputeRunner;
    using Pipeline = Runner::ComputePipeline;
    using BufInfo = Runner::BufferInfo;

    Runner runner_;
    bool initialized_ = false;
    bool weights_loaded_ = false;
    std::string last_error_;
    std::string shader_dir_;
    AttentionMode attn_mode_ = ATTN_FLASH_GPU;

    // Model hyperparameters
    int n_blocks_ = 0;
    int dim_ = 0;
    int n_heads_ = 0;
    int head_dim_ = 0;
    int ffn_dim_ = 0;
    int patch_size_ = 0;
    int image_size_ = 0;
    int n_patches_max_ = 0;
    int proj_dim_ = 0;
    int spatial_merge_ = 0;
    float ln_eps_ = 1e-6f;

    // DeepStack info
    int n_deepstack_ = 0;
    std::vector<int> deepstack_indices_;

    // Pipelines (one per shader)
    Pipeline pipe_add_;
    Pipeline pipe_gelu_;
    Pipeline pipe_matmul_bias_;
    Pipeline pipe_layernorm_;
    Pipeline pipe_softmax_;
    Pipeline pipe_patch_embed_;
    Pipeline pipe_add_transposed_;
    Pipeline pipe_spatial_merge_;
    Pipeline pipe_attn_naive_;
    Pipeline pipe_attn_flash_;
    Pipeline pipe_rope_vision_;
    bool pipelines_created_ = false;

    // Weight buffers on GPU (name -> buffer)
    std::map<std::string, BufInfo> weight_bufs_;

    // Per-block weight buffer names
    struct BlockWeightNames {
        std::string qkv_w, qkv_b, out_w, out_b;
        std::string ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b;
        std::string ln1_w, ln1_b, ln2_w, ln2_b;
    };
    std::vector<BlockWeightNames> block_names_;

    struct DeepstackWeightNames {
        std::string fc1_w, fc1_b, fc2_w, fc2_b;
        std::string norm_w, norm_b;
    };
    std::vector<DeepstackWeightNames> ds_names_;

    // Helper: dequantize a qtensor and upload to GPU buffer
    bool uploadTensor(const std::string &name, const qtensor &t);

    // Helper: dequantize entire tensor (all rows) to flat f32
    std::vector<float> dequantFull(const qtensor &t);

    // Create all pipelines
    bool createPipelines();

    // Dispatch helpers
    bool dispatchAdd(BufInfo &dst, const BufInfo &src, uint32_t n);
    bool dispatchGelu(BufInfo &buf, uint32_t n);
    bool dispatchMatmulBias(const BufInfo &A, const BufInfo &W, BufInfo &C, const BufInfo &bias,
                            uint32_t M, uint32_t N, uint32_t K, bool has_bias);
    bool dispatchLayernorm(const BufInfo &src, BufInfo &dst,
                           const BufInfo &weight, const BufInfo &bias,
                           uint32_t n_tokens, uint32_t dim, float eps);
    bool dispatchSoftmax(BufInfo &data, uint32_t n_rows, uint32_t n_cols);
    bool dispatchPatchEmbed(const BufInfo &image, const BufInfo &kernel,
                            const BufInfo &bias, BufInfo &output,
                            uint32_t n_patches, uint32_t dim, uint32_t kernel_size,
                            uint32_t gw, uint32_t width, uint32_t patch_sz);
    bool dispatchAddTransposed(BufInfo &hidden, const BufInfo &pos_emb,
                               uint32_t n_patches, uint32_t dim,
                               uint32_t gw, uint32_t orig_gw, uint32_t orig_n);
    bool dispatchSpatialMerge(const BufInfo &src, BufInfo &dst,
                              uint32_t gw, uint32_t gh, uint32_t dim, uint32_t sm);
    bool dispatchAttnNaive(const BufInfo &qkv, BufInfo &out,
                           uint32_t n_patches, uint32_t dim,
                           uint32_t n_heads, uint32_t head_dim, float scale);
    bool dispatchAttnFlash(const BufInfo &qkv, BufInfo &out,
                           uint32_t n_patches, uint32_t dim,
                           uint32_t n_heads, uint32_t head_dim, float scale);
    bool dispatchRopeVision(BufInfo &qkv, const BufInfo &pos_ids,
                            uint32_t n_patches, uint32_t dim,
                            uint32_t n_heads, uint32_t head_dim,
                            uint32_t sect_size, float freq_base);

    // Buffer creation helper
    BufInfo createGpuBuffer(size_t size);
    void destroyGpuBuffer(BufInfo &buf);

    // Upload data to buffer
    bool uploadToBuffer(BufInfo &buf, const void *data, size_t size);
    bool downloadFromBuffer(const BufInfo &buf, void *data, size_t size);
};
