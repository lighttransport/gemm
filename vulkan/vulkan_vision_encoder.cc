// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan Vision Encoder implementation
//

#include "vulkan_vision_encoder.hh"

#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>

extern "C" {
#include "../common/profiler.h"
}

/* GPU vision profiling macros */
#ifdef PROFILER_H
#define GVIT_PROF_BEGIN(name, layer, op, prec) prof_begin(name, "gpu_vision", layer, op, prec)
#define GVIT_PROF_END(name, flops, iops) prof_end(name, flops, iops)
#else
#define GVIT_PROF_BEGIN(name, layer, op, prec) ((void)0)
#define GVIT_PROF_END(name, flops, iops) ((void)0)
#endif

// Load SPIR-V binary
static std::vector<uint32_t> load_spirv(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> spirv(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(spirv.data()), size);
    return spirv;
}

// Helper to create standard storage buffer bindings
static std::vector<VkDescriptorSetLayoutBinding> makeBindings(int count) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(count);
    for (int i = 0; i < count; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }
    return bindings;
}

VulkanVisionEncoder::VulkanVisionEncoder() {}

VulkanVisionEncoder::~VulkanVisionEncoder() {
    // Destroy weight buffers
    for (auto &kv : weight_bufs_) {
        runner_.destroyBuffer(kv.second);
    }
    weight_bufs_.clear();

    if (pipelines_created_) {
        runner_.destroyComputePipeline(pipe_add_);
        runner_.destroyComputePipeline(pipe_gelu_);
        runner_.destroyComputePipeline(pipe_matmul_bias_);
        runner_.destroyComputePipeline(pipe_layernorm_);
        runner_.destroyComputePipeline(pipe_softmax_);
        runner_.destroyComputePipeline(pipe_patch_embed_);
        runner_.destroyComputePipeline(pipe_add_transposed_);
        runner_.destroyComputePipeline(pipe_spatial_merge_);
        runner_.destroyComputePipeline(pipe_attn_naive_);
        runner_.destroyComputePipeline(pipe_attn_flash_);
        runner_.destroyComputePipeline(pipe_rope_vision_);
    }

    if (initialized_) {
        runner_.cleanup();
    }
}

bool VulkanVisionEncoder::initialize(int device_id, bool verbose) {
    if (!vl_cpp::vulkan::InitializeVulkan()) {
        last_error_ = "Failed to initialize Vulkan";
        return false;
    }

    if (!runner_.initialize(false)) {
        last_error_ = "Failed to initialize runner: " + runner_.getLastError();
        return false;
    }

    if (static_cast<uint32_t>(device_id) >= runner_.getDeviceCount()) {
        last_error_ = "Invalid device ID";
        return false;
    }

    if (!runner_.selectDevice(device_id)) {
        last_error_ = "Failed to select device: " + runner_.getLastError();
        return false;
    }

    if (verbose) {
        std::cerr << "Vulkan device: " << runner_.getDeviceName(device_id) << "\n";
    }

    initialized_ = true;
    return true;
}

std::string VulkanVisionEncoder::getDeviceName() const {
    if (!initialized_) return "not initialized";
    return runner_.getDeviceName(0);
}

VulkanVisionEncoder::BufInfo VulkanVisionEncoder::createGpuBuffer(size_t size) {
    BufInfo buf{};
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!runner_.createBuffer(size, usage, props, buf)) {
        last_error_ = "Failed to create buffer of size " + std::to_string(size);
    }
    return buf;
}

void VulkanVisionEncoder::destroyGpuBuffer(BufInfo &buf) {
    runner_.destroyBuffer(buf);
    buf = {};
}

bool VulkanVisionEncoder::uploadToBuffer(BufInfo &buf, const void *data, size_t size) {
    void *ptr = nullptr;
    if (!runner_.mapBuffer(buf, &ptr)) {
        last_error_ = "Failed to map buffer for upload";
        return false;
    }
    std::memcpy(ptr, data, size);
    runner_.unmapBuffer(buf);
    return true;
}

bool VulkanVisionEncoder::downloadFromBuffer(const BufInfo &buf, void *data, size_t size) {
    void *ptr = nullptr;
    if (!runner_.mapBuffer(buf, &ptr)) {
        last_error_ = "Failed to map buffer for download";
        return false;
    }
    std::memcpy(data, ptr, size);
    runner_.unmapBuffer(buf);
    return true;
}

std::vector<float> VulkanVisionEncoder::dequantFull(const qtensor &t) {
    if (!t.data) return {};
    int n_rows = t.n_rows;
    int n_cols = t.n_cols;

    // For 4D tensors like patch_embd_w, compute actual total elements
    int total = n_rows * n_cols;
    std::vector<float> out(total);

    // Determine block_size and type_size for row stride
    int block_size = 1, type_size = 4;
    switch (t.type) {
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q6_K: block_size = 256; type_size = 210; break;
        case GGML_TYPE_F32:  block_size = 1;   type_size = 4;   break;
        case GGML_TYPE_F16:  block_size = 1;   type_size = 2;   break;
        default: break;
    }

    size_t row_bytes = (size_t)((n_cols + block_size - 1) / block_size) * type_size;

    for (int row = 0; row < n_rows; row++) {
        const void *row_data = (const uint8_t *)t.data + row * row_bytes;
        dequant_row(t.type, row_data, out.data() + row * n_cols, n_cols);
    }
    return out;
}

bool VulkanVisionEncoder::uploadTensor(const std::string &name, const qtensor &t) {
    auto data = dequantFull(t);
    if (data.empty()) return false;

    size_t bytes = data.size() * sizeof(float);
    auto buf = createGpuBuffer(bytes);
    if (buf.buffer == VK_NULL_HANDLE) return false;

    if (!uploadToBuffer(buf, data.data(), bytes)) {
        runner_.destroyBuffer(buf);
        return false;
    }

    weight_bufs_[name] = buf;
    return true;
}

bool VulkanVisionEncoder::createPipelines() {
    auto load = [&](const std::string &name) -> std::vector<uint32_t> {
        std::string path = shader_dir_ + "/shaders/" + name + ".spv";
        auto spirv = load_spirv(path);
        if (spirv.empty()) {
            last_error_ = "Failed to load shader: " + path;
        }
        return spirv;
    };

    // add_f32: 2 buffers, push_constant {uint n}
    {
        auto spirv = load("add_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), sizeof(uint32_t), pipe_add_)) {
            last_error_ = "Failed to create add pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // gelu_f32: 1 buffer, push_constant {uint n}
    {
        auto spirv = load("gelu_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(1), sizeof(uint32_t), pipe_gelu_)) {
            last_error_ = "Failed to create gelu pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // matmul_bias_f32: 4 buffers (A, W, C, bias), push_constant {M, N, K, has_bias}
    {
        auto spirv = load("matmul_bias_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(4), 4 * sizeof(uint32_t), pipe_matmul_bias_)) {
            last_error_ = "Failed to create matmul_bias pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // layernorm_f32: 4 buffers (src, dst, weight, bias), push_constant {n_tokens, dim, eps}
    {
        auto spirv = load("layernorm_f32");
        if (spirv.empty()) return false;
        // Push constants: 2*uint + 1*float = 12 bytes
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(4), 12, pipe_layernorm_)) {
            last_error_ = "Failed to create layernorm pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // softmax_f32: 1 buffer, push_constant {n_rows, n_cols}
    {
        auto spirv = load("softmax_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(1), 2 * sizeof(uint32_t), pipe_softmax_)) {
            last_error_ = "Failed to create softmax pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // patch_embed_f32: 4 buffers (image, kernel, bias, output), push_constant {n_patches, dim, kernel_size, gw, width, patch_size}
    {
        auto spirv = load("patch_embed_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(4), 6 * sizeof(uint32_t), pipe_patch_embed_)) {
            last_error_ = "Failed to create patch_embed pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // add_transposed_f32: 2 buffers, push_constant {n_patches, dim, gw, orig_gw, orig_n}
    {
        auto spirv = load("add_transposed_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 5 * sizeof(uint32_t), pipe_add_transposed_)) {
            last_error_ = "Failed to create add_transposed pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // spatial_merge_f32: 2 buffers (src, dst), push_constant {gw, gh, dim, sm}
    {
        auto spirv = load("spatial_merge_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 4 * sizeof(uint32_t), pipe_spatial_merge_)) {
            last_error_ = "Failed to create spatial_merge pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // attn_naive_f32: 2 buffers (qkv, out), push_constant {n_patches, dim, n_heads, head_dim, scale}
    {
        auto spirv = load("attn_naive_f32");
        if (spirv.empty()) return false;
        // 4 uints + 1 float = 20 bytes
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 20, pipe_attn_naive_)) {
            last_error_ = "Failed to create attn_naive pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // attn_flash_f32: 2 buffers (qkv, out), push_constant {n_patches, dim, n_heads, head_dim, scale}
    {
        auto spirv = load("attn_flash_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 20, pipe_attn_flash_)) {
            last_error_ = "Failed to create attn_flash pipeline: " + runner_.getLastError();
            return false;
        }
    }

    // rope_vision_f32: 2 buffers (qkv, pos_ids), push_constant {n_patches, dim, head_dim, n_heads, sect_size, freq_base}
    {
        auto spirv = load("rope_vision_f32");
        if (spirv.empty()) return false;
        // 5 uints + 1 float = 24 bytes
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 24, pipe_rope_vision_)) {
            last_error_ = "Failed to create rope_vision pipeline: " + runner_.getLastError();
            return false;
        }
    }

    pipelines_created_ = true;
    return true;
}

// ---- Dispatch helpers ----

bool VulkanVisionEncoder::dispatchAdd(BufInfo &dst, const BufInfo &src, uint32_t n) {
    std::vector<BufInfo> bufs = {dst, src};
    runner_.updateDescriptorSet(pipe_add_, bufs);

    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_add_);
    runner_.bindDescriptorSets(pipe_add_);
    runner_.pushConstants(pipe_add_, &n, sizeof(n));
    runner_.dispatch((n + 255) / 256);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchGelu(BufInfo &buf, uint32_t n) {
    std::vector<BufInfo> bufs = {buf};
    runner_.updateDescriptorSet(pipe_gelu_, bufs);

    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_gelu_);
    runner_.bindDescriptorSets(pipe_gelu_);
    runner_.pushConstants(pipe_gelu_, &n, sizeof(n));
    runner_.dispatch((n + 255) / 256);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchMatmulBias(const BufInfo &A, const BufInfo &W,
                                              BufInfo &C, const BufInfo &bias,
                                              uint32_t M, uint32_t N, uint32_t K,
                                              bool has_bias) {
    std::vector<BufInfo> bufs = {A, W, C, bias};
    runner_.updateDescriptorSet(pipe_matmul_bias_, bufs);

    struct { uint32_t M, N, K, has_bias; } pc = {M, N, K, has_bias ? 1u : 0u};

    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_matmul_bias_);
    runner_.bindDescriptorSets(pipe_matmul_bias_);
    runner_.pushConstants(pipe_matmul_bias_, &pc, sizeof(pc));
    runner_.dispatch((N + 15) / 16, (M + 15) / 16);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchLayernorm(const BufInfo &src, BufInfo &dst,
                                             const BufInfo &weight, const BufInfo &bias,
                                             uint32_t n_tokens, uint32_t dim, float eps) {
    std::vector<BufInfo> bufs = {src, dst, weight, bias};
    runner_.updateDescriptorSet(pipe_layernorm_, bufs);

    struct { uint32_t n_tokens, dim; float eps; } pc = {n_tokens, dim, eps};

    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_layernorm_);
    runner_.bindDescriptorSets(pipe_layernorm_);
    runner_.pushConstants(pipe_layernorm_, &pc, sizeof(pc));
    runner_.dispatch(n_tokens);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchSoftmax(BufInfo &data, uint32_t n_rows, uint32_t n_cols) {
    std::vector<BufInfo> bufs = {data};
    runner_.updateDescriptorSet(pipe_softmax_, bufs);

    struct { uint32_t n_rows, n_cols; } pc = {n_rows, n_cols};

    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_softmax_);
    runner_.bindDescriptorSets(pipe_softmax_);
    runner_.pushConstants(pipe_softmax_, &pc, sizeof(pc));
    runner_.dispatch(n_rows);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchPatchEmbed(const BufInfo &image, const BufInfo &kernel,
                                              const BufInfo &bias, BufInfo &output,
                                              uint32_t n_patches, uint32_t dim,
                                              uint32_t kernel_size, uint32_t gw,
                                              uint32_t width, uint32_t patch_sz) {
    std::vector<BufInfo> bufs = {image, kernel, bias, output};
    runner_.updateDescriptorSet(pipe_patch_embed_, bufs);

    struct { uint32_t n_patches, dim, kernel_size, gw, width, patch_size; } pc = {
        n_patches, dim, kernel_size, gw, width, patch_sz
    };

    uint32_t total = n_patches * dim;
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_patch_embed_);
    runner_.bindDescriptorSets(pipe_patch_embed_);
    runner_.pushConstants(pipe_patch_embed_, &pc, sizeof(pc));
    runner_.dispatch((total + 255) / 256);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchAddTransposed(BufInfo &hidden, const BufInfo &pos_emb,
                                                 uint32_t n_patches, uint32_t dim,
                                                 uint32_t gw, uint32_t orig_gw, uint32_t orig_n) {
    std::vector<BufInfo> bufs = {hidden, pos_emb};
    runner_.updateDescriptorSet(pipe_add_transposed_, bufs);

    struct { uint32_t n_patches, dim, gw, orig_gw, orig_n; } pc = {
        n_patches, dim, gw, orig_gw, orig_n
    };

    uint32_t total = n_patches * dim;
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_add_transposed_);
    runner_.bindDescriptorSets(pipe_add_transposed_);
    runner_.pushConstants(pipe_add_transposed_, &pc, sizeof(pc));
    runner_.dispatch((total + 255) / 256);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchSpatialMerge(const BufInfo &src, BufInfo &dst,
                                                uint32_t gw, uint32_t gh,
                                                uint32_t dim, uint32_t sm) {
    std::vector<BufInfo> bufs = {src, dst};
    runner_.updateDescriptorSet(pipe_spatial_merge_, bufs);

    struct { uint32_t gw, gh, dim, sm; } pc = {gw, gh, dim, sm};

    uint32_t mgw = gw / sm, mgh = gh / sm;
    uint32_t total = mgw * mgh * dim * sm * sm;
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_spatial_merge_);
    runner_.bindDescriptorSets(pipe_spatial_merge_);
    runner_.pushConstants(pipe_spatial_merge_, &pc, sizeof(pc));
    runner_.dispatch((total + 255) / 256);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchAttnNaive(const BufInfo &qkv, BufInfo &out,
                                             uint32_t n_patches, uint32_t dim,
                                             uint32_t n_heads, uint32_t head_dim,
                                             float scale) {
    std::vector<BufInfo> bufs = {qkv, out};
    runner_.updateDescriptorSet(pipe_attn_naive_, bufs);

    struct { uint32_t n_patches, dim, n_heads, head_dim; float scale; } pc = {
        n_patches, dim, n_heads, head_dim, scale
    };

    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_attn_naive_);
    runner_.bindDescriptorSets(pipe_attn_naive_);
    runner_.pushConstants(pipe_attn_naive_, &pc, sizeof(pc));
    // Dispatch: one WG per (query, head)
    runner_.dispatch(n_patches, n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchAttnFlash(const BufInfo &qkv, BufInfo &out,
                                             uint32_t n_patches, uint32_t dim,
                                             uint32_t n_heads, uint32_t head_dim,
                                             float scale) {
    std::vector<BufInfo> bufs = {qkv, out};
    runner_.updateDescriptorSet(pipe_attn_flash_, bufs);

    struct { uint32_t n_patches, dim, n_heads, head_dim; float scale; } pc = {
        n_patches, dim, n_heads, head_dim, scale
    };

    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_attn_flash_);
    runner_.bindDescriptorSets(pipe_attn_flash_);
    runner_.pushConstants(pipe_attn_flash_, &pc, sizeof(pc));
    // Dispatch: one WG per (query, head)
    runner_.dispatch(n_patches, n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanVisionEncoder::dispatchRopeVision(BufInfo &qkv, const BufInfo &pos_ids,
                                              uint32_t n_patches, uint32_t dim,
                                              uint32_t n_heads, uint32_t head_dim,
                                              uint32_t sect_size, float freq_base) {
    std::vector<BufInfo> bufs = {qkv, pos_ids};
    runner_.updateDescriptorSet(pipe_rope_vision_, bufs);

    struct { uint32_t n_patches, dim, head_dim, n_heads, sect_size; float freq_base; } pc = {
        n_patches, dim, head_dim, n_heads, sect_size, freq_base
    };

    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_rope_vision_);
    runner_.bindDescriptorSets(pipe_rope_vision_);
    runner_.pushConstants(pipe_rope_vision_, &pc, sizeof(pc));
    // Dispatch: one WG per (patch, head), local_size_x = 32 covers all pairs
    runner_.dispatch(n_patches, n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

// ---- Weight loading ----

static int vit_get_int(const gguf_context *g, const char *key, int def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_UINT32) return (int)g->kv[idx].value.u32;
    if (g->kv[idx].type == GGUF_TYPE_INT32) return g->kv[idx].value.i32;
    return def;
}

static float vit_get_float(const gguf_context *g, const char *key, float def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_FLOAT32) return g->kv[idx].value.f32;
    return def;
}

static int vit_find_tensor(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

static qtensor vit_load_qt(const gguf_context *g, const char *name) {
    qtensor t = {};
    int idx = vit_find_tensor(g, name);
    if (idx < 0) return t;
    t.data = gguf_tensor_data(g, idx);
    t.type = g->tensors[idx].type;
    t.n_cols = (int)g->tensors[idx].dims[0];
    t.n_rows = (g->tensors[idx].n_dims >= 2) ? (int)g->tensors[idx].dims[1] : 1;
    return t;
}

bool VulkanVisionEncoder::loadWeights(gguf_context *g) {
    if (!initialized_) {
        last_error_ = "Not initialized";
        return false;
    }

    // Read hyperparams
    n_blocks_    = vit_get_int(g, "clip.vision.block_count", 24);
    dim_         = vit_get_int(g, "clip.vision.embedding_length", 1024);
    n_heads_     = vit_get_int(g, "clip.vision.attention.head_count", 16);
    ffn_dim_     = vit_get_int(g, "clip.vision.feed_forward_length", 4096);
    patch_size_  = vit_get_int(g, "clip.vision.patch_size", 16);
    image_size_  = vit_get_int(g, "clip.vision.image_size", 768);
    proj_dim_    = vit_get_int(g, "clip.vision.projection_dim", 2048);
    spatial_merge_ = vit_get_int(g, "clip.vision.spatial_merge_size", 2);
    ln_eps_      = vit_get_float(g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    head_dim_    = dim_ / n_heads_;

    int gs = image_size_ / patch_size_;
    n_patches_max_ = gs * gs;

    std::cerr << "vulkan vision: dim=" << dim_ << " heads=" << n_heads_
              << " blocks=" << n_blocks_ << " ffn=" << ffn_dim_
              << " patch=" << patch_size_ << " image=" << image_size_
              << " proj=" << proj_dim_ << "\n";

    // Create pipelines (needs shader_dir_)
    // Determine shader directory: look for shaders/ relative to executable or current dir
    shader_dir_ = ".";
    if (!createPipelines()) return false;

    // Upload patch embedding (dual conv2d: add both kernels together on CPU)
    {
        qtensor pw0 = vit_load_qt(g, "v.patch_embd.weight");
        qtensor pw1 = vit_load_qt(g, "v.patch_embd.weight.1");
        if (pw0.data) {
            int total = patch_size_ * patch_size_ * 3 * dim_;
            std::vector<float> kernel(total);
            dequant_row(pw0.type, pw0.data, kernel.data(), total);
            if (pw1.data) {
                std::vector<float> kernel1(total);
                dequant_row(pw1.type, pw1.data, kernel1.data(), total);
                for (int i = 0; i < total; i++) kernel[i] += kernel1[i];
                std::cerr << "vulkan vision: loaded dual conv2d patch embeddings\n";
            }
            size_t bytes = total * sizeof(float);
            auto buf = createGpuBuffer(bytes);
            uploadToBuffer(buf, kernel.data(), bytes);
            weight_bufs_["patch_embd_w"] = buf;
        }

        qtensor pb = vit_load_qt(g, "v.patch_embd.bias");
        if (pb.data) uploadTensor("patch_embd_b", pb);
    }

    // Position embedding
    {
        qtensor pe = vit_load_qt(g, "v.position_embd.weight");
        if (pe.data) {
            int total = dim_ * n_patches_max_;
            std::vector<float> pos(total);
            dequant_row(pe.type, pe.data, pos.data(), total);
            size_t bytes = total * sizeof(float);
            auto buf = createGpuBuffer(bytes);
            uploadToBuffer(buf, pos.data(), bytes);
            weight_bufs_["position_embd"] = buf;
        }
    }

    // Blocks
    block_names_.resize(n_blocks_);
    for (int l = 0; l < n_blocks_; l++) {
        char name[128];
        auto &bn = block_names_[l];

        auto upload_blk = [&](const char *suffix, std::string &key) {
            snprintf(name, sizeof(name), "v.blk.%d.%s", l, suffix);
            key = name;
            qtensor t = vit_load_qt(g, name);
            if (t.data) uploadTensor(key, t);
        };

        upload_blk("attn_qkv.weight", bn.qkv_w);
        upload_blk("attn_qkv.bias", bn.qkv_b);
        upload_blk("attn_out.weight", bn.out_w);
        upload_blk("attn_out.bias", bn.out_b);
        upload_blk("ffn_up.weight", bn.ffn_up_w);
        upload_blk("ffn_up.bias", bn.ffn_up_b);
        upload_blk("ffn_down.weight", bn.ffn_down_w);
        upload_blk("ffn_down.bias", bn.ffn_down_b);
        upload_blk("ln1.weight", bn.ln1_w);
        upload_blk("ln1.bias", bn.ln1_b);
        upload_blk("ln2.weight", bn.ln2_w);
        upload_blk("ln2.bias", bn.ln2_b);

        if (l == 0 || (l + 1) % 6 == 0 || l == n_blocks_ - 1) {
            std::cerr << "  uploaded block " << l << "/" << n_blocks_ << "\n";
        }
    }

    // DeepStack
    int idx = gguf_find_key(g, "clip.vision.is_deepstack_layers");
    if (idx >= 0) {
        uint8_t *flags = (uint8_t *)g->kv[idx].value.arr.data;
        int n = (int)g->kv[idx].value.arr.n;
        for (int i = 0; i < n; i++) {
            if (flags[i]) deepstack_indices_.push_back(i);
        }
        n_deepstack_ = (int)deepstack_indices_.size();

        ds_names_.resize(n_deepstack_);
        for (int si = 0; si < n_deepstack_; si++) {
            int li = deepstack_indices_[si];
            char name[128];
            auto &dn = ds_names_[si];

            auto upload_ds = [&](const char *suffix, std::string &key) {
                snprintf(name, sizeof(name), "v.deepstack.%d.%s", li, suffix);
                key = name;
                qtensor t = vit_load_qt(g, name);
                if (t.data) uploadTensor(key, t);
            };

            upload_ds("fc1.weight", dn.fc1_w);
            upload_ds("fc1.bias", dn.fc1_b);
            upload_ds("fc2.weight", dn.fc2_w);
            upload_ds("fc2.bias", dn.fc2_b);
            upload_ds("norm.weight", dn.norm_w);
            upload_ds("norm.bias", dn.norm_b);
        }
        std::cerr << "  uploaded " << n_deepstack_ << " deepstack layers\n";
    }

    // Post LN
    {
        qtensor t;
        t = vit_load_qt(g, "v.post_ln.weight"); if (t.data) uploadTensor("post_ln_w", t);
        t = vit_load_qt(g, "v.post_ln.bias");   if (t.data) uploadTensor("post_ln_b", t);
    }

    // MM projection
    {
        qtensor t;
        t = vit_load_qt(g, "mm.0.weight"); if (t.data) uploadTensor("mm0_w", t);
        t = vit_load_qt(g, "mm.0.bias");   if (t.data) uploadTensor("mm0_b", t);
        t = vit_load_qt(g, "mm.2.weight"); if (t.data) uploadTensor("mm2_w", t);
        t = vit_load_qt(g, "mm.2.bias");   if (t.data) uploadTensor("mm2_b", t);
    }

    weights_loaded_ = true;
    std::cerr << "vulkan vision: all weights uploaded to GPU\n";
    return true;
}

// ---- Encode ----

float *VulkanVisionEncoder::encode(const float *rgb_norm, int width, int height) {
    if (!weights_loaded_) {
        last_error_ = "Weights not loaded";
        return nullptr;
    }

    int ps = patch_size_;
    int dim = dim_;
    int n_heads = n_heads_;
    int head_dim = head_dim_;
    int ffn_dim = ffn_dim_;
    int gw = width / ps;
    int gh = height / ps;
    int n_patches = gw * gh;
    int sm = spatial_merge_;
    int merged_dim = dim * sm * sm;
    int n_merged = n_patches / (sm * sm);
    int orig_gw = image_size_ / ps;

    std::cerr << "vulkan vision: encoding " << width << "x" << height
              << " (" << n_patches << " patches, " << n_merged << " merged)\n";

    // Create temporary GPU buffers
    auto buf_hidden  = createGpuBuffer(n_patches * dim * sizeof(float));
    auto buf_hidden2 = createGpuBuffer(n_patches * dim * sizeof(float));
    auto buf_qkv     = createGpuBuffer(n_patches * 3 * dim * sizeof(float));
    auto buf_attn_out = createGpuBuffer(n_patches * dim * sizeof(float));
    auto buf_ffn     = createGpuBuffer(n_patches * ffn_dim * sizeof(float));
    auto buf_ln      = createGpuBuffer(n_patches * dim * sizeof(float));
    auto buf_att     = createGpuBuffer(n_patches * n_patches * sizeof(float));
    auto buf_merge   = createGpuBuffer(n_merged * merged_dim * sizeof(float));
    // Deepstack feature buffers (one per deepstack layer, for concat)
    std::vector<BufInfo> buf_ds_feats;
    buf_ds_feats.reserve(n_deepstack_);
    auto buf_image   = createGpuBuffer(width * height * 3 * sizeof(float));

    // For attention: Q, K, V head buffers
    auto buf_q_head  = createGpuBuffer(n_patches * head_dim * sizeof(float));
    auto buf_k_head  = createGpuBuffer(n_patches * head_dim * sizeof(float));
    auto buf_v_head  = createGpuBuffer(n_patches * head_dim * sizeof(float));
    auto buf_att_head_out = createGpuBuffer(n_patches * head_dim * sizeof(float));

    // Dummy bias buffer for matmul without bias
    auto buf_dummy_bias = createGpuBuffer(std::max(merged_dim, std::max(3 * dim, ffn_dim)) * sizeof(float));

    // (deepstack features are stored per-layer, no accumulator needed)

    // Upload image
    uploadToBuffer(buf_image, rgb_norm, width * height * 3 * sizeof(float));

    // Create M-RoPE position IDs: [4][n_patches] = {p_t, p_h, p_w, p_e}
    std::vector<int32_t> pos_ids(4 * n_patches, 0);
    for (int py = 0; py < gh; py++) {
        for (int px = 0; px < gw; px++) {
            int p = py * gw + px;
            pos_ids[0 * n_patches + p] = py;  // p_t
            pos_ids[1 * n_patches + p] = px;  // p_h
            pos_ids[2 * n_patches + p] = py;  // p_w
            pos_ids[3 * n_patches + p] = px;  // p_e
        }
    }
    auto buf_pos_ids = createGpuBuffer(4 * n_patches * sizeof(int32_t));
    uploadToBuffer(buf_pos_ids, pos_ids.data(), 4 * n_patches * sizeof(int32_t));

    // 1. Patch embedding
    std::cerr << "  patch embedding...\n";
    GVIT_PROF_BEGIN("patch_embed", -1, "conv2d", "FP32");
    {
        uint32_t kernel_size = ps * ps * 3;
        dispatchPatchEmbed(buf_image, weight_bufs_["patch_embd_w"],
                           weight_bufs_["patch_embd_b"], buf_hidden,
                           n_patches, dim, kernel_size, gw, width, ps);
    }
    GVIT_PROF_END("patch_embed", 2.0 * n_patches * dim * ps * ps * 3, 0);

    // 2. Position embeddings
    std::cerr << "  position embeddings...\n";
    GVIT_PROF_BEGIN("pos_embd", -1, "add", "FP32");
    dispatchAddTransposed(buf_hidden, weight_bufs_["position_embd"],
                          n_patches, dim, gw, orig_gw, n_patches_max_);
    GVIT_PROF_END("pos_embd", (double)n_patches * dim, 0);

    // 3. ViT blocks
    for (int l = 0; l < n_blocks_; l++) {
        if (l == 0 || l == n_blocks_ - 1 || (l + 1) % 6 == 0)
            std::cerr << "  vit block " << l << "/" << n_blocks_ << "\n";

        auto &bn = block_names_[l];

        // LayerNorm1
        GVIT_PROF_BEGIN("ln1", l, "layernorm", "FP32");
        dispatchLayernorm(buf_hidden, buf_ln,
                          weight_bufs_[bn.ln1_w], weight_bufs_[bn.ln1_b],
                          n_patches, dim, ln_eps_);
        GVIT_PROF_END("ln1", 5.0 * n_patches * dim, 0);

        // QKV projection: [n_patches, dim] -> [n_patches, 3*dim]
        GVIT_PROF_BEGIN("qkv_matmul", l, "matmul", "FP32");
        dispatchMatmulBias(buf_ln, weight_bufs_[bn.qkv_w], buf_qkv,
                           weight_bufs_[bn.qkv_b],
                           n_patches, 3 * dim, dim, true);
        GVIT_PROF_END("qkv_matmul", 2.0 * n_patches * 3 * dim * dim, 0);

        // M-RoPE on Q and K
        GVIT_PROF_BEGIN("mrope", l, "rope", "FP32");
        dispatchRopeVision(buf_qkv, buf_pos_ids,
                           n_patches, dim, n_heads, head_dim,
                           head_dim / 4, 10000.0f);
        GVIT_PROF_END("mrope", (double)n_patches * n_heads * (head_dim / 2) * 8.0 * 2, 0);

        // Multi-head attention
        GVIT_PROF_BEGIN("attention", l, "attention", "FP32");
        {
            float scale = 1.0f / sqrtf((float)head_dim);
            if (attn_mode_ == ATTN_FLASH_GPU) {
                dispatchAttnFlash(buf_qkv, buf_attn_out,
                                  n_patches, dim, n_heads, head_dim, scale);
            } else if (attn_mode_ == ATTN_NAIVE_GPU) {
                dispatchAttnNaive(buf_qkv, buf_attn_out,
                                  n_patches, dim, n_heads, head_dim, scale);
            } else {
                // CPU fallback
                std::vector<float> qkv_data(n_patches * 3 * dim);
                downloadFromBuffer(buf_qkv, qkv_data.data(), qkv_data.size() * sizeof(float));

                std::vector<float> attn_result(n_patches * dim, 0.0f);
                std::vector<float> att_scores(n_patches * n_patches);

                for (int h = 0; h < n_heads; h++) {
                    for (int qi = 0; qi < n_patches; qi++) {
                        const float *q_h = qkv_data.data() + qi * 3 * dim + h * head_dim;
                        for (int ki = 0; ki < n_patches; ki++) {
                            const float *k_h = qkv_data.data() + ki * 3 * dim + dim + h * head_dim;
                            float score = 0.0f;
                            for (int d2 = 0; d2 < head_dim; d2++) score += q_h[d2] * k_h[d2];
                            att_scores[qi * n_patches + ki] = score * scale;
                        }
                    }
                    for (int qi = 0; qi < n_patches; qi++) {
                        float *row = att_scores.data() + qi * n_patches;
                        float max_val = row[0];
                        for (int j = 1; j < n_patches; j++) max_val = std::max(max_val, row[j]);
                        float sum = 0.0f;
                        for (int j = 0; j < n_patches; j++) { row[j] = expf(row[j] - max_val); sum += row[j]; }
                        for (int j = 0; j < n_patches; j++) row[j] /= sum;
                    }
                    for (int qi = 0; qi < n_patches; qi++) {
                        float *out_h = attn_result.data() + qi * dim + h * head_dim;
                        for (int vi = 0; vi < n_patches; vi++) {
                            const float *v_h = qkv_data.data() + vi * 3 * dim + 2 * dim + h * head_dim;
                            float a = att_scores[qi * n_patches + vi];
                            for (int d2 = 0; d2 < head_dim; d2++) out_h[d2] += a * v_h[d2];
                        }
                    }
                }
                uploadToBuffer(buf_attn_out, attn_result.data(), attn_result.size() * sizeof(float));
            }
        }

        GVIT_PROF_END("attention", 2.0 * n_heads * (double)n_patches * n_patches * head_dim * 2, 0);

        // Attention output projection
        GVIT_PROF_BEGIN("attn_out", l, "matmul", "FP32");
        dispatchMatmulBias(buf_attn_out, weight_bufs_[bn.out_w], buf_hidden2,
                           weight_bufs_[bn.out_b],
                           n_patches, dim, dim, true);
        GVIT_PROF_END("attn_out", 2.0 * n_patches * dim * dim, 0);

        // Residual: hidden += hidden2
        dispatchAdd(buf_hidden, buf_hidden2, n_patches * dim);

        // LayerNorm2
        GVIT_PROF_BEGIN("ln2", l, "layernorm", "FP32");
        dispatchLayernorm(buf_hidden, buf_ln,
                          weight_bufs_[bn.ln2_w], weight_bufs_[bn.ln2_b],
                          n_patches, dim, ln_eps_);
        GVIT_PROF_END("ln2", 5.0 * n_patches * dim, 0);

        // FFN up
        GVIT_PROF_BEGIN("ffn_up", l, "matmul", "FP32");
        dispatchMatmulBias(buf_ln, weight_bufs_[bn.ffn_up_w], buf_ffn,
                           weight_bufs_[bn.ffn_up_b],
                           n_patches, ffn_dim, dim, true);
        GVIT_PROF_END("ffn_up", 2.0 * n_patches * ffn_dim * dim, 0);

        // GELU
        GVIT_PROF_BEGIN("gelu", l, "gelu", "FP32");
        dispatchGelu(buf_ffn, n_patches * ffn_dim);
        GVIT_PROF_END("gelu", 8.0 * n_patches * ffn_dim, 0);

        // FFN down
        GVIT_PROF_BEGIN("ffn_down", l, "matmul", "FP32");
        dispatchMatmulBias(buf_ffn, weight_bufs_[bn.ffn_down_w], buf_hidden2,
                           weight_bufs_[bn.ffn_down_b],
                           n_patches, dim, ffn_dim, true);
        GVIT_PROF_END("ffn_down", 2.0 * n_patches * dim * ffn_dim, 0);

        // Residual
        dispatchAdd(buf_hidden, buf_hidden2, n_patches * dim);

        // DeepStack
        for (int ds = 0; ds < n_deepstack_; ds++) {
            if (deepstack_indices_[ds] != l) continue;

            std::cerr << "  deepstack at layer " << l << "\n";
            auto &dn = ds_names_[ds];

            // Spatial merge hidden -> merge_buf
            dispatchSpatialMerge(buf_hidden, buf_merge, gw, gh, dim, sm);

            // LayerNorm on merge_buf (before fc1)
            auto buf_ds_ln = createGpuBuffer(n_merged * merged_dim * sizeof(float));
            GVIT_PROF_BEGIN("ds_ln", l, "layernorm", "FP32");
            dispatchLayernorm(buf_merge, buf_ds_ln,
                              weight_bufs_[dn.norm_w], weight_bufs_[dn.norm_b],
                              n_merged, merged_dim, ln_eps_);
            GVIT_PROF_END("ds_ln", 5.0 * n_merged * merged_dim, 0);

            // fc1
            auto buf_ds_tmp = createGpuBuffer(n_merged * merged_dim * sizeof(float));
            GVIT_PROF_BEGIN("ds_fc1", l, "matmul", "FP32");
            dispatchMatmulBias(buf_ds_ln, weight_bufs_[dn.fc1_w], buf_ds_tmp,
                               weight_bufs_[dn.fc1_b],
                               n_merged, merged_dim, merged_dim, true);
            GVIT_PROF_END("ds_fc1", 2.0 * n_merged * merged_dim * merged_dim, 0);

            // GELU
            GVIT_PROF_BEGIN("ds_gelu", l, "gelu", "FP32");
            dispatchGelu(buf_ds_tmp, n_merged * merged_dim);
            GVIT_PROF_END("ds_gelu", 8.0 * n_merged * merged_dim, 0);

            // fc2: [merged_dim -> proj_dim]
            auto buf_ds_out = createGpuBuffer(n_merged * proj_dim_ * sizeof(float));
            GVIT_PROF_BEGIN("ds_fc2", l, "matmul", "FP32");
            dispatchMatmulBias(buf_ds_tmp, weight_bufs_[dn.fc2_w], buf_ds_out,
                               weight_bufs_[dn.fc2_b],
                               n_merged, proj_dim_, merged_dim, true);
            GVIT_PROF_END("ds_fc2", 2.0 * n_merged * proj_dim_ * merged_dim, 0);

            // Store deepstack feature (keep buffer alive for final concat)
            buf_ds_feats.push_back(buf_ds_out);

            destroyGpuBuffer(buf_ds_tmp);
            destroyGpuBuffer(buf_ds_ln);
        }
    }

    // 4. Post LayerNorm
    std::cerr << "  post layernorm...\n";
    GVIT_PROF_BEGIN("post_ln", -1, "layernorm", "FP32");
    dispatchLayernorm(buf_hidden, buf_hidden,
                      weight_bufs_["post_ln_w"], weight_bufs_["post_ln_b"],
                      n_patches, dim, ln_eps_);
    GVIT_PROF_END("post_ln", 5.0 * n_patches * dim, 0);

    // 5. Spatial merge
    std::cerr << "  spatial merge...\n";
    dispatchSpatialMerge(buf_hidden, buf_merge, gw, gh, dim, sm);

    // 6. MM projection
    std::cerr << "  mm projection...\n";
    auto buf_mm = createGpuBuffer(n_merged * merged_dim * sizeof(float));
    GVIT_PROF_BEGIN("mm0", -1, "matmul", "FP32");
    dispatchMatmulBias(buf_merge, weight_bufs_["mm0_w"], buf_mm,
                       weight_bufs_["mm0_b"],
                       n_merged, merged_dim, merged_dim, true);
    GVIT_PROF_END("mm0", 2.0 * n_merged * merged_dim * merged_dim, 0);

    GVIT_PROF_BEGIN("mm_gelu", -1, "gelu", "FP32");
    dispatchGelu(buf_mm, n_merged * merged_dim);
    GVIT_PROF_END("mm_gelu", 8.0 * n_merged * merged_dim, 0);

    auto buf_result = createGpuBuffer(n_merged * proj_dim_ * sizeof(float));
    GVIT_PROF_BEGIN("mm2", -1, "matmul", "FP32");
    dispatchMatmulBias(buf_mm, weight_bufs_["mm2_w"], buf_result,
                       weight_bufs_["mm2_b"],
                       n_merged, proj_dim_, merged_dim, true);
    GVIT_PROF_END("mm2", 2.0 * n_merged * proj_dim_ * merged_dim, 0);

    // 7. Concat main embeddings + deepstack features per token
    // Layout: [main_proj_dim, ds0_proj_dim, ds1_proj_dim, ...] for each token
    int total_embd = proj_dim_ * (1 + (int)buf_ds_feats.size());
    float *result = new float[n_merged * total_embd];

    // Download main result
    std::vector<float> main_result(n_merged * proj_dim_);
    downloadFromBuffer(buf_result, main_result.data(), n_merged * proj_dim_ * sizeof(float));

    // Download each deepstack feature
    std::vector<std::vector<float>> ds_results(buf_ds_feats.size());
    for (size_t d = 0; d < buf_ds_feats.size(); d++) {
        ds_results[d].resize(n_merged * proj_dim_);
        downloadFromBuffer(buf_ds_feats[d], ds_results[d].data(), n_merged * proj_dim_ * sizeof(float));
    }

    // Interleave: for each token, [main, ds0, ds1, ...]
    for (int t = 0; t < n_merged; t++) {
        float *dst = result + t * total_embd;
        memcpy(dst, main_result.data() + t * proj_dim_, proj_dim_ * sizeof(float));
        for (size_t d = 0; d < buf_ds_feats.size(); d++) {
            memcpy(dst + (1 + d) * proj_dim_,
                   ds_results[d].data() + t * proj_dim_,
                   proj_dim_ * sizeof(float));
        }
    }

    std::cerr << "  vulkan vision encoding done: " << n_merged << " tokens of dim " << total_embd
              << " (main " << proj_dim_ << " + " << buf_ds_feats.size() << " deepstack)\n";

    // Cleanup temp buffers
    destroyGpuBuffer(buf_hidden);
    destroyGpuBuffer(buf_hidden2);
    destroyGpuBuffer(buf_qkv);
    destroyGpuBuffer(buf_attn_out);
    destroyGpuBuffer(buf_ffn);
    destroyGpuBuffer(buf_ln);
    destroyGpuBuffer(buf_att);
    destroyGpuBuffer(buf_merge);
    for (auto &b : buf_ds_feats) destroyGpuBuffer(b);
    destroyGpuBuffer(buf_image);
    destroyGpuBuffer(buf_q_head);
    destroyGpuBuffer(buf_k_head);
    destroyGpuBuffer(buf_v_head);
    destroyGpuBuffer(buf_att_head_out);
    destroyGpuBuffer(buf_dummy_bias);
    destroyGpuBuffer(buf_pos_ids);
    destroyGpuBuffer(buf_mm);
    destroyGpuBuffer(buf_result);

    return result;
}
