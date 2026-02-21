// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan LLM Runner implementation
//

#include "vulkan_llm_runner.hh"

#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

extern "C" {
#include "../common/profiler.h"
}

/* GPU LLM profiling macros */
#ifdef PROFILER_H
#define GLLM_PROF_BEGIN(name, layer, op, prec) prof_begin(name, "gpu_llm", layer, op, prec)
#define GLLM_PROF_END(name, flops, iops) prof_end(name, flops, iops)
#else
#define GLLM_PROF_BEGIN(name, layer, op, prec) ((void)0)
#define GLLM_PROF_END(name, flops, iops) ((void)0)
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

VulkanLLMRunner::VulkanLLMRunner() {}

VulkanLLMRunner::~VulkanLLMRunner() {
    if (cpu_model_) {
        transformer_free(cpu_model_);
        cpu_model_ = nullptr;
    }

    // Destroy weight buffers
    for (auto &kv : weight_bufs_) {
        runner_.destroyBuffer(kv.second);
    }
    weight_bufs_.clear();

    // Destroy scratch buffers
    if (scratch_created_) {
        runner_.destroyBuffer(buf_x_);
        runner_.destroyBuffer(buf_xb_);
        runner_.destroyBuffer(buf_xb2_);
        runner_.destroyBuffer(buf_q_);
        runner_.destroyBuffer(buf_k_);
        runner_.destroyBuffer(buf_v_);
        runner_.destroyBuffer(buf_ffn_gate_);
        runner_.destroyBuffer(buf_ffn_up_);
        runner_.destroyBuffer(buf_moe_router_);
        runner_.destroyBuffer(buf_logits_);
        for (int d = 0; d < 3; d++) {
            if (buf_ds_slices_[d].buffer != VK_NULL_HANDLE)
                runner_.destroyBuffer(buf_ds_slices_[d]);
        }
    }

    // Destroy KV cache
    for (auto &b : kv_key_bufs_) runner_.destroyBuffer(b);
    for (auto &b : kv_val_bufs_) runner_.destroyBuffer(b);

    // Destroy prefill buffers
    if (prefill_enabled_) {
        runner_.destroyBuffer(buf_pfx_);
        runner_.destroyBuffer(buf_pfxb_);
        runner_.destroyBuffer(buf_pfq_);
        runner_.destroyBuffer(buf_pfk_);
        runner_.destroyBuffer(buf_pfv_);
        runner_.destroyBuffer(buf_pf_attn_out_);
        runner_.destroyBuffer(buf_pf_ffn_gate_);
        runner_.destroyBuffer(buf_pf_ffn_up_);
        runner_.destroyBuffer(buf_positions_);
        runner_.destroyBuffer(buf_cache_pos_);
        runner_.destroyBuffer(buf_weight_f16_);
    }

    if (pipelines_created_) {
        runner_.destroyComputePipeline(pipe_rmsnorm_);
        runner_.destroyComputePipeline(pipe_matvec_);
        runner_.destroyComputePipeline(pipe_matvec_q8_0_);
        runner_.destroyComputePipeline(pipe_silu_mul_);
        runner_.destroyComputePipeline(pipe_rope_neox_);
        runner_.destroyComputePipeline(pipe_rope_mrope_);
        runner_.destroyComputePipeline(pipe_rope_neox_batch_);
        runner_.destroyComputePipeline(pipe_rope_mrope_batch_);
        runner_.destroyComputePipeline(pipe_qknorm_);
        runner_.destroyComputePipeline(pipe_kv_store_);
        runner_.destroyComputePipeline(pipe_kv_store_batch_);
        runner_.destroyComputePipeline(pipe_attn_decode_);
        runner_.destroyComputePipeline(pipe_attn_prefill_);
        runner_.destroyComputePipeline(pipe_add_);
        runner_.destroyComputePipeline(pipe_dequant_q8_0_);
        runner_.destroyComputePipeline(pipe_matmul_coopmat_nt_);
    }

    if (initialized_) {
        runner_.cleanup();
    }
}

bool VulkanLLMRunner::initialize(int device_id, bool verbose) {
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
        std::cerr << "Vulkan LLM device: " << runner_.getDeviceName(device_id) << "\n";
    }
    initialized_ = true;
    return true;
}

std::string VulkanLLMRunner::getDeviceName() const {
    if (!initialized_) return "not initialized";
    return runner_.getDeviceName(0);
}

VulkanLLMRunner::BufInfo VulkanLLMRunner::createGpuBuffer(size_t size) {
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

void VulkanLLMRunner::destroyGpuBuffer(BufInfo &buf) {
    runner_.destroyBuffer(buf);
    buf = {};
}

bool VulkanLLMRunner::uploadToBuffer(BufInfo &buf, const void *data, size_t size) {
    void *ptr = nullptr;
    if (!runner_.mapBuffer(buf, &ptr)) {
        last_error_ = "Failed to map buffer for upload";
        return false;
    }
    std::memcpy(ptr, data, size);
    runner_.unmapBuffer(buf);
    return true;
}

bool VulkanLLMRunner::downloadFromBuffer(const BufInfo &buf, void *data, size_t size) {
    void *ptr = nullptr;
    if (!runner_.mapBuffer(buf, &ptr)) {
        last_error_ = "Failed to map buffer for download";
        return false;
    }
    std::memcpy(data, ptr, size);
    runner_.unmapBuffer(buf);
    return true;
}

std::vector<float> VulkanLLMRunner::dequantFull(const qtensor &t) {
    if (!t.data) return {};
    int n_rows = t.n_rows;
    int n_cols = t.n_cols;
    int total = n_rows * n_cols;
    std::vector<float> out(total);

    int block_size = 1, type_size = 4;
    switch (t.type) {
        case GGML_TYPE_Q2_K: block_size = 256; type_size = 84;  break;
        case GGML_TYPE_Q3_K: block_size = 256; type_size = 110; break;
        case GGML_TYPE_Q8_0: block_size = 32;  type_size = 34;  break;
        case GGML_TYPE_Q4_K: block_size = 256; type_size = 144; break;
        case GGML_TYPE_Q5_K: block_size = 256; type_size = 176; break;
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

bool VulkanLLMRunner::uploadTensor(const std::string &name, const qtensor &t) {
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
    weight_types_[name] = GGML_TYPE_F32;  // dequanted to F32
    return true;
}

bool VulkanLLMRunner::uploadTensorRaw(const std::string &name, const qtensor &t) {
    if (!t.data) return false;
    int block_size = 32, type_size = 34;  // Q8_0
    size_t row_bytes = (size_t)((t.n_cols + block_size - 1) / block_size) * type_size;
    size_t total_bytes = (size_t)t.n_rows * row_bytes;

    // Vulkan storage buffers need at least 4-byte alignment
    size_t aligned_bytes = (total_bytes + 3) & ~3ULL;
    auto buf = createGpuBuffer(aligned_bytes);
    if (buf.buffer == VK_NULL_HANDLE) return false;

    if (!uploadToBuffer(buf, t.data, total_bytes)) {
        runner_.destroyBuffer(buf);
        return false;
    }

    weight_bufs_[name] = buf;
    weight_types_[name] = t.type;
    return true;
}

bool VulkanLLMRunner::createPipelines() {
    auto load = [&](const std::string &name) -> std::vector<uint32_t> {
        std::string path = shader_dir_ + "/shaders/" + name + ".spv";
        auto spirv = load_spirv(path);
        if (spirv.empty()) {
            last_error_ = "Failed to load shader: " + path;
        }
        return spirv;
    };

    // rmsnorm_f32: 3 buffers, push {n_tokens, dim, eps} = 12 bytes
    {
        auto spirv = load("rmsnorm_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(3), 12, pipe_rmsnorm_))
            return false;
    }
    // matvec_f32: 3 buffers, push {N, K, row0} = 12 bytes
    {
        auto spirv = load("matvec_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(3), 12, pipe_matvec_))
            return false;
    }
    // matvec_q8_0_f32: 3 buffers, push {N, K, row0} = 12 bytes
    {
        auto spirv = load("matvec_q8_0_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(3), 12, pipe_matvec_q8_0_))
            return false;
    }
    // silu_mul_f32: 2 buffers, push {n} = 4 bytes
    {
        auto spirv = load("silu_mul_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 4, pipe_silu_mul_))
            return false;
    }
    // rope_neox_f32: 1 buffer, push {n_heads, head_dim, position, freq_base} = 16 bytes
    {
        auto spirv = load("rope_neox_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(1), 16, pipe_rope_neox_))
            return false;
    }
    // rope_mrope_f32: 1 buffer, push = 36 bytes
    {
        auto spirv = load("rope_mrope_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(1), 36, pipe_rope_mrope_))
            return false;
    }
    // rope_neox_batch_f32: 2 buffers, push {n_tokens, n_heads, head_dim, freq_base} = 16 bytes
    {
        auto spirv = load("rope_neox_batch_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 16, pipe_rope_neox_batch_))
            return false;
    }
    // rope_mrope_batch_f32: 2 buffers, push {n_tokens, n_heads, head_dim, freq_base, sect0, sect1, sect2} = 28 bytes
    {
        auto spirv = load("rope_mrope_batch_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 28, pipe_rope_mrope_batch_))
            return false;
    }
    // qknorm_f32: 2 buffers, push {n_heads, head_dim, eps} = 12 bytes
    {
        auto spirv = load("qknorm_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 12, pipe_qknorm_))
            return false;
    }
    // kv_cache_store_f32: 4 buffers, push {cache_pos, kv_dim} = 8 bytes
    {
        auto spirv = load("kv_cache_store_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(4), 8, pipe_kv_store_))
            return false;
    }
    // kv_cache_store_batch_f32: 5 buffers, push {n_tokens, kv_dim} = 8 bytes
    {
        auto spirv = load("kv_cache_store_batch_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(5), 8, pipe_kv_store_batch_))
            return false;
    }
    // attn_decode_f32: 4 buffers, push {n_heads, n_kv_heads, head_dim, seq_len, max_seq_len, scale} = 24 bytes
    {
        auto spirv = load("attn_decode_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(4), 24, pipe_attn_decode_))
            return false;
    }
    // attn_prefill_f32: 5 buffers, push {n_tokens, n_heads, n_kv_heads, head_dim, scale} = 20 bytes
    {
        auto spirv = load("attn_prefill_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(5), 20, pipe_attn_prefill_))
            return false;
    }
    // matmul_coopmat_nt_f16: 3 buffers, push {M, N, K} = 12 bytes
    {
        auto spirv = load("matmul_coopmat_nt_f16");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(3), 12, pipe_matmul_coopmat_nt_))
            return false;
    }
    // dequant_q8_0_f16: 2 buffers, push {n_rows, n_cols} = 8 bytes
    {
        auto spirv = load("dequant_q8_0_f16");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 8, pipe_dequant_q8_0_))
            return false;
    }
    // add_f32: reuse existing
    {
        auto spirv = load("add_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(2), 4, pipe_add_))
            return false;
    }

    pipelines_created_ = true;
    return true;
}

// ---- Dispatch helpers ----

bool VulkanLLMRunner::dispatchRmsNorm(const BufInfo &src, BufInfo &dst, const BufInfo &weight,
                                       uint32_t n_tokens, uint32_t dim, float eps) {
    std::vector<BufInfo> bufs = {src, dst, weight};
    runner_.updateDescriptorSet(pipe_rmsnorm_, bufs);
    struct { uint32_t n_tokens, dim; float eps; } pc = {n_tokens, dim, eps};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_rmsnorm_);
    runner_.bindDescriptorSets(pipe_rmsnorm_);
    runner_.pushConstants(pipe_rmsnorm_, &pc, sizeof(pc));
    runner_.dispatch(n_tokens);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchMatvec(const BufInfo &x, const BufInfo &W, BufInfo &dst,
                                      uint32_t N, uint32_t K, uint32_t row_offset) {
    std::vector<BufInfo> bufs = {x, W, dst};
    runner_.updateDescriptorSet(pipe_matvec_, bufs);
    struct { uint32_t N, K, row0; } pc = {N, K, row_offset};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_matvec_);
    runner_.bindDescriptorSets(pipe_matvec_);
    runner_.pushConstants(pipe_matvec_, &pc, sizeof(pc));
    runner_.dispatch(N);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchMatvecAuto(const BufInfo &x, const std::string &w_name, BufInfo &dst,
                                          uint32_t N, uint32_t K, uint32_t row_offset) {
    auto it = weight_types_.find(w_name);
    if (it != weight_types_.end() && it->second == GGML_TYPE_Q8_0) {
        return dispatchMatvecQ8(x, weight_bufs_[w_name], dst, N, K, row_offset);
    }
    return dispatchMatvec(x, weight_bufs_[w_name], dst, N, K, row_offset);
}

bool VulkanLLMRunner::dispatchMatvecQ8(const BufInfo &x, const BufInfo &W_raw, BufInfo &dst,
                                        uint32_t N, uint32_t K, uint32_t row_offset) {
    std::vector<BufInfo> bufs = {x, W_raw, dst};
    runner_.updateDescriptorSet(pipe_matvec_q8_0_, bufs);
    struct { uint32_t N, K, row0; } pc = {N, K, row_offset};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_matvec_q8_0_);
    runner_.bindDescriptorSets(pipe_matvec_q8_0_);
    runner_.pushConstants(pipe_matvec_q8_0_, &pc, sizeof(pc));
    runner_.dispatch(N);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchSiluMul(BufInfo &gate, const BufInfo &up, uint32_t n) {
    std::vector<BufInfo> bufs = {gate, up};
    runner_.updateDescriptorSet(pipe_silu_mul_, bufs);
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_silu_mul_);
    runner_.bindDescriptorSets(pipe_silu_mul_);
    runner_.pushConstants(pipe_silu_mul_, &n, sizeof(n));
    runner_.dispatch((n + 255) / 256);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchRopeNeox(BufInfo &vec, uint32_t n_heads, uint32_t head_dim,
                                        int position, float freq_base) {
    std::vector<BufInfo> bufs = {vec};
    runner_.updateDescriptorSet(pipe_rope_neox_, bufs);
    struct { uint32_t n_heads, head_dim; int32_t position; float freq_base; } pc = {
        n_heads, head_dim, position, freq_base
    };
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_rope_neox_);
    runner_.bindDescriptorSets(pipe_rope_neox_);
    runner_.pushConstants(pipe_rope_neox_, &pc, sizeof(pc));
    runner_.dispatch(n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchRopeMrope(BufInfo &vec, uint32_t n_heads, uint32_t head_dim,
                                         int pos_t, int pos_h, int pos_w, float freq_base,
                                         int sect0, int sect1, int sect2) {
    std::vector<BufInfo> bufs = {vec};
    runner_.updateDescriptorSet(pipe_rope_mrope_, bufs);
    struct {
        uint32_t n_heads, head_dim;
        int32_t pos_t, pos_h, pos_w;
        float freq_base;
        int32_t sect0, sect1, sect2;
    } pc = {n_heads, head_dim, pos_t, pos_h, pos_w, freq_base, sect0, sect1, sect2};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_rope_mrope_);
    runner_.bindDescriptorSets(pipe_rope_mrope_);
    runner_.pushConstants(pipe_rope_mrope_, &pc, sizeof(pc));
    runner_.dispatch(n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchQkNorm(BufInfo &vec, const BufInfo &weight,
                                      uint32_t n_heads, uint32_t head_dim, float eps) {
    std::vector<BufInfo> bufs = {vec, weight};
    runner_.updateDescriptorSet(pipe_qknorm_, bufs);
    struct { uint32_t n_heads, head_dim; float eps; } pc = {n_heads, head_dim, eps};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_qknorm_);
    runner_.bindDescriptorSets(pipe_qknorm_);
    runner_.pushConstants(pipe_qknorm_, &pc, sizeof(pc));
    runner_.dispatch(n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchKvStore(const BufInfo &k_vec, const BufInfo &v_vec,
                                       BufInfo &k_cache, BufInfo &v_cache,
                                       uint32_t cache_pos, uint32_t kv_dim) {
    std::vector<BufInfo> bufs = {k_vec, v_vec, k_cache, v_cache};
    runner_.updateDescriptorSet(pipe_kv_store_, bufs);
    struct { uint32_t cache_pos, kv_dim; } pc = {cache_pos, kv_dim};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_kv_store_);
    runner_.bindDescriptorSets(pipe_kv_store_);
    runner_.pushConstants(pipe_kv_store_, &pc, sizeof(pc));
    runner_.dispatch((kv_dim + 255) / 256);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchAttnDecode(const BufInfo &q, const BufInfo &k_cache,
                                          const BufInfo &v_cache, BufInfo &out,
                                          uint32_t n_heads, uint32_t n_kv_heads,
                                          uint32_t head_dim, uint32_t seq_len, float scale) {
    std::vector<BufInfo> bufs = {q, k_cache, v_cache, out};
    runner_.updateDescriptorSet(pipe_attn_decode_, bufs);
    struct { uint32_t n_heads, n_kv_heads, head_dim, seq_len, max_seq_len; float scale; } pc = {
        n_heads, n_kv_heads, head_dim, seq_len, (uint32_t)max_seq_len_, scale
    };
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_attn_decode_);
    runner_.bindDescriptorSets(pipe_attn_decode_);
    runner_.pushConstants(pipe_attn_decode_, &pc, sizeof(pc));
    runner_.dispatch(n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchAdd(BufInfo &dst, const BufInfo &src, uint32_t n) {
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

bool VulkanLLMRunner::dispatchRopeNeoxBatch(BufInfo &vec, const BufInfo &positions,
                                             uint32_t n_tokens, uint32_t n_heads,
                                             uint32_t head_dim, float freq_base) {
    std::vector<BufInfo> bufs = {vec, positions};
    runner_.updateDescriptorSet(pipe_rope_neox_batch_, bufs);
    struct { uint32_t n_tokens, n_heads, head_dim; float freq_base; } pc = {
        n_tokens, n_heads, head_dim, freq_base
    };
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_rope_neox_batch_);
    runner_.bindDescriptorSets(pipe_rope_neox_batch_);
    runner_.pushConstants(pipe_rope_neox_batch_, &pc, sizeof(pc));
    runner_.dispatch(n_tokens * n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchRopeMropeBatch(BufInfo &vec, const BufInfo &positions,
                                              uint32_t n_tokens, uint32_t n_heads,
                                              uint32_t head_dim, float freq_base,
                                              int sect0, int sect1, int sect2) {
    std::vector<BufInfo> bufs = {vec, positions};
    runner_.updateDescriptorSet(pipe_rope_mrope_batch_, bufs);
    struct {
        uint32_t n_tokens, n_heads, head_dim;
        float freq_base;
        int32_t sect0, sect1, sect2;
    } pc = {n_tokens, n_heads, head_dim, freq_base, sect0, sect1, sect2};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_rope_mrope_batch_);
    runner_.bindDescriptorSets(pipe_rope_mrope_batch_);
    runner_.pushConstants(pipe_rope_mrope_batch_, &pc, sizeof(pc));
    runner_.dispatch(n_tokens * n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchKvStoreBatch(const BufInfo &k, const BufInfo &v,
                                            BufInfo &k_cache, BufInfo &v_cache,
                                            const BufInfo &cache_positions,
                                            uint32_t n_tokens, uint32_t kv_dim) {
    std::vector<BufInfo> bufs = {k, v, k_cache, v_cache, cache_positions};
    runner_.updateDescriptorSet(pipe_kv_store_batch_, bufs);
    struct { uint32_t n_tokens, kv_dim; } pc = {n_tokens, kv_dim};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_kv_store_batch_);
    runner_.bindDescriptorSets(pipe_kv_store_batch_);
    runner_.pushConstants(pipe_kv_store_batch_, &pc, sizeof(pc));
    runner_.dispatch(n_tokens);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchAttnPrefill(const BufInfo &q, const BufInfo &k_cache,
                                           const BufInfo &v_cache, BufInfo &out,
                                           const BufInfo &cache_positions,
                                           uint32_t n_tokens, uint32_t n_heads,
                                           uint32_t n_kv_heads, uint32_t head_dim,
                                           uint32_t max_seq_len, float scale) {
    (void)max_seq_len;
    std::vector<BufInfo> bufs = {q, k_cache, v_cache, out, cache_positions};
    runner_.updateDescriptorSet(pipe_attn_prefill_, bufs);
    struct { uint32_t n_tokens, n_heads, n_kv_heads, head_dim; float scale; } pc = {
        n_tokens, n_heads, n_kv_heads, head_dim, scale
    };
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_attn_prefill_);
    runner_.bindDescriptorSets(pipe_attn_prefill_);
    runner_.pushConstants(pipe_attn_prefill_, &pc, sizeof(pc));
    runner_.dispatch(n_tokens * n_heads);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchMatmulCoopmatNT(const BufInfo &A, const BufInfo &B, BufInfo &C,
                                                uint32_t M, uint32_t N, uint32_t K) {
    std::vector<BufInfo> bufs = {A, B, C};
    runner_.updateDescriptorSet(pipe_matmul_coopmat_nt_, bufs);
    struct { uint32_t M, N, K; } pc = {M, N, K};
    // Dispatch: grid = ceil(N/64) x ceil(M/128)
    uint32_t grid_x = (N + 63) / 64;
    uint32_t grid_y = (M + 127) / 128;
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_matmul_coopmat_nt_);
    runner_.bindDescriptorSets(pipe_matmul_coopmat_nt_);
    runner_.pushConstants(pipe_matmul_coopmat_nt_, &pc, sizeof(pc));
    runner_.dispatch(grid_x, grid_y);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

bool VulkanLLMRunner::dispatchDequantQ8(const BufInfo &src, BufInfo &dst,
                                         uint32_t n_rows, uint32_t n_cols) {
    std::vector<BufInfo> bufs = {src, dst};
    runner_.updateDescriptorSet(pipe_dequant_q8_0_, bufs);
    struct { uint32_t n_rows, n_cols; } pc = {n_rows, n_cols};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_dequant_q8_0_);
    runner_.bindDescriptorSets(pipe_dequant_q8_0_);
    runner_.pushConstants(pipe_dequant_q8_0_, &pc, sizeof(pc));
    runner_.dispatch(n_rows);
    runner_.endRecordingAndSubmit();
    return runner_.waitForCompletion();
}

// ---- Weight loading ----

static int llm_get_int(const gguf_context *g, const char *key, int def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_UINT32) return (int)g->kv[idx].value.u32;
    if (g->kv[idx].type == GGUF_TYPE_INT32) return g->kv[idx].value.i32;
    return def;
}

static float llm_get_float(const gguf_context *g, const char *key, float def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_FLOAT32) return g->kv[idx].value.f32;
    return def;
}

static int llm_find_tensor(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

static qtensor llm_load_qt(const gguf_context *g, const char *name) {
    qtensor t = {};
    int idx = llm_find_tensor(g, name);
    if (idx < 0) return t;
    t.data = gguf_tensor_data(g, idx);
    t.type = g->tensors[idx].type;
    t.n_cols = (int)g->tensors[idx].dims[0];
    t.n_rows = 1;
    for (uint32_t d = 1; d < g->tensors[idx].n_dims; d++) {
        t.n_rows *= (int)g->tensors[idx].dims[d];
    }
    return t;
}

static void llm_softmax(float *x, int n) {
    if (!x || n <= 0) return;
    float vmax = x[0];
    for (int i = 1; i < n; i++) vmax = std::max(vmax, x[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - vmax);
        sum += x[i];
    }
    if (sum <= 0.0f) return;
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

bool VulkanLLMRunner::loadWeights(gguf_context *g) {
    if (!initialized_) {
        last_error_ = "Not initialized";
        return false;
    }
    if (cpu_model_) {
        transformer_free(cpu_model_);
        cpu_model_ = nullptr;
    }
    cpu_fallback_moe_ = false;

    // Detect architecture
    const char *arch = "qwen2";
    bool is_moe_arch = false;
    if (gguf_find_key(g, "qwen3vlmoe.block_count") >= 0) {
        arch = "qwen3vlmoe";
        is_moe_arch = true;
    } else if (gguf_find_key(g, "qwen3moe.block_count") >= 0) {
        arch = "qwen3moe";
        is_moe_arch = true;
    } else if (gguf_find_key(g, "qwen3vl.block_count") >= 0) {
        arch = "qwen3vl";
    } else if (gguf_find_key(g, "qwen3.block_count") >= 0) {
        arch = "qwen3";
    }

    char kbuf[128];
    #define ARCH_KEY(suffix) (snprintf(kbuf, sizeof(kbuf), "%s." suffix, arch), kbuf)

    n_embd_      = llm_get_int(g, ARCH_KEY("embedding_length"), 4096);
    n_heads_     = llm_get_int(g, ARCH_KEY("attention.head_count"), 32);
    n_kv_heads_  = llm_get_int(g, ARCH_KEY("attention.head_count_kv"), 8);
    n_layers_    = llm_get_int(g, ARCH_KEY("block_count"), 36);
    n_ff_        = llm_get_int(g, ARCH_KEY("feed_forward_length"), 12288);
    n_expert_    = llm_get_int(g, ARCH_KEY("expert_count"), 0);
    n_expert_used_ = llm_get_int(g, ARCH_KEY("expert_used_count"), 0);
    n_ff_expert_ = llm_get_int(g, ARCH_KEY("expert_feed_forward_length"), 0);
    use_moe_ = (n_expert_ > 0);
    n_vocab_     = llm_get_int(g, ARCH_KEY("vocab_size"), 0);
    rms_norm_eps_ = llm_get_float(g, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);
    rope_freq_base_ = llm_get_float(g, ARCH_KEY("rope.freq_base"), 5000000.0f);
    head_dim_    = llm_get_int(g, ARCH_KEY("attention.key_length"), n_embd_ / n_heads_);
    int model_ctx_len = llm_get_int(g, ARCH_KEY("context_length"), 0);
    int requested_ctx = 0;
    if (const char *env_ctx = std::getenv("GEMM_MAX_SEQ_LEN")) {
        requested_ctx = std::atoi(env_ctx);
    }
    const int default_ctx = 4096;
    if (requested_ctx > 0) {
        max_seq_len_ = requested_ctx;
    } else if (model_ctx_len > 0) {
        max_seq_len_ = std::min(model_ctx_len, default_ctx);
    } else {
        max_seq_len_ = default_ctx;
    }
    if (model_ctx_len > 0 && max_seq_len_ > model_ctx_len) {
        max_seq_len_ = model_ctx_len;
    }
    if (max_seq_len_ <= 0) {
        last_error_ = "Invalid context length resolved from GGUF/env";
        return false;
    }
    if (model_ctx_len > 0 && max_seq_len_ < model_ctx_len) {
        std::cerr << "vulkan llm: runtime context capped to " << max_seq_len_
                  << " (model supports " << model_ctx_len
                  << "). Set GEMM_MAX_SEQ_LEN to override.\n";
    }

    // M-RoPE sections
    use_mrope_ = 0;
    memset(mrope_sections_, 0, sizeof(mrope_sections_));
    {
        int idx = gguf_find_key(g, ARCH_KEY("rope.dimension_sections"));
        if (idx >= 0 && g->kv[idx].type == GGUF_TYPE_ARRAY &&
            g->kv[idx].value.arr.type == GGUF_TYPE_INT32) {
            int n = (int)g->kv[idx].value.arr.n;
            if (n > 4) n = 4;
            int32_t *data = (int32_t *)g->kv[idx].value.arr.data;
            for (int i = 0; i < n; i++) mrope_sections_[i] = data[i];
            int sect_sum = mrope_sections_[0] + mrope_sections_[1] + mrope_sections_[2] + mrope_sections_[3];
            if (sect_sum > 0) {
                use_mrope_ = 1;
                std::cerr << "vulkan llm: M-RoPE sections=[" << mrope_sections_[0] << ", "
                          << mrope_sections_[1] << ", " << mrope_sections_[2] << ", "
                          << mrope_sections_[3] << "]\n";
            }
        }
    }
    n_deepstack_ = llm_get_int(g, ARCH_KEY("n_deepstack_layers"), 0);
    if (n_deepstack_ > 3) n_deepstack_ = 3;
    if (n_deepstack_ > 0) {
        std::cerr << "vulkan llm: n_deepstack=" << n_deepstack_ << "\n";
    }

    #undef ARCH_KEY

    if (use_moe_) {
        if (n_expert_used_ <= 0 || n_expert_used_ > n_expert_) {
            last_error_ = "Invalid expert_used_count in GGUF";
            return false;
        }
        if (n_ff_expert_ <= 0) {
            last_error_ = "Invalid expert_feed_forward_length in GGUF";
            return false;
        }
    } else {
        n_expert_used_ = 0;
        n_ff_expert_ = 0;
    }

    int kv_dim = n_kv_heads_ * head_dim_;

    std::cerr << "vulkan llm: arch=" << arch << " n_embd=" << n_embd_ << " n_heads=" << n_heads_
              << " n_kv_heads=" << n_kv_heads_ << " n_layers=" << n_layers_ << " n_ff=" << n_ff_
              << " head_dim=" << head_dim_ << " n_vocab=" << n_vocab_;
    if (use_moe_) {
        std::cerr << " n_expert=" << n_expert_ << " n_expert_used=" << n_expert_used_
                  << " n_ff_expert=" << n_ff_expert_;
    }
    std::cerr << "\n";

    if (is_moe_arch && !use_moe_) {
        last_error_ = std::string("Architecture '") + arch + "' detected but MoE metadata is incomplete";
        return false;
    }

    if (use_moe_) {
        const char *moe_cpu_fallback = std::getenv("GEMM_MOE_CPU_FALLBACK");
        if (!(moe_cpu_fallback && std::atoi(moe_cpu_fallback) != 0)) {
            std::cerr << "vulkan llm: MoE GPU path enabled\n";
        } else {
            std::cerr << "vulkan llm: MoE architecture detected; enabling CPU fallback path\n";
            const int fallback_ctx = std::min(max_seq_len_, 1024);
            cpu_model_ = transformer_load(g, fallback_ctx);
            if (!cpu_model_) {
                last_error_ = "Failed to initialize CPU fallback model for MoE architecture";
                return false;
            }
            cpu_model_->ds_embd_stride = ds_embd_stride_;
            n_layers_ = cpu_model_->n_layers;
            n_embd_ = cpu_model_->n_embd;
            n_heads_ = cpu_model_->n_heads;
            n_kv_heads_ = cpu_model_->n_kv_heads;
            head_dim_ = cpu_model_->head_dim;
            n_ff_ = cpu_model_->n_ff;
            n_vocab_ = cpu_model_->n_vocab;
            max_seq_len_ = cpu_model_->max_seq_len;
            cpu_fallback_moe_ = true;
            weights_loaded_ = true;
            return true;
        }
    }
    if (is_moe_arch && use_moe_ && n_ff_ <= 0) {
        // MoE models may omit dense feed_forward_length; avoid accidental zero-sized buffers.
        n_ff_ = n_ff_expert_;
    }

    // Create pipelines
    shader_dir_ = ".";
    if (!createPipelines()) return false;

    // Upload global tensors
    std::cerr << "vulkan llm: uploading token_embd...\n";
    {
        qtensor t = llm_load_qt(g, "token_embd.weight");
        if (!t.data) {
            last_error_ = "Missing required tensor token_embd.weight";
            return false;
        }
        if (n_vocab_ == 0) n_vocab_ = t.n_rows;
        if (!uploadTensor("token_embd", t)) {
            last_error_ = "Failed to upload required tensor token_embd.weight";
            return false;
        }
    }
    {
        qtensor t = llm_load_qt(g, "output_norm.weight");
        if (!t.data) {
            last_error_ = "Missing required tensor output_norm.weight";
            return false;
        }
        if (!uploadTensor("output_norm", t)) {
            last_error_ = "Failed to upload required tensor output_norm.weight";
            return false;
        }
    }
    {
        qtensor t = llm_load_qt(g, "output.weight");
        if (t.data) {
            bool ok = (t.type == GGML_TYPE_Q8_0) ? uploadTensorRaw("output", t) : uploadTensor("output", t);
            if (!ok) {
                last_error_ = "Failed to upload tensor output.weight";
                return false;
            }
        } else {
            // Weight tying: output shares token_embd
            std::cerr << "vulkan llm: weight-tied output (using token_embd)\n";
        }
    }
    if (n_vocab_ <= 0) {
        last_error_ = "Invalid vocabulary size";
        return false;
    }

    // Per-layer tensors
    layer_names_.resize(n_layers_);
    bool missing_required = false;
    for (int l = 0; l < n_layers_; l++) {
        char name[128];
        auto &ln = layer_names_[l];

        auto upload_blk = [&](const char *suffix, std::string &key, bool required) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, suffix);
            key = name;
            qtensor t = llm_load_qt(g, name);
            if (t.data) {
                bool ok;
                if (t.type == GGML_TYPE_Q8_0) {
                    ok = uploadTensorRaw(key, t);
                } else {
                    ok = uploadTensor(key, t);
                }
                if (!ok && required) {
                    if (!missing_required) {
                        last_error_ = std::string("Failed to upload required tensor ") + name;
                    }
                    missing_required = true;
                }
            } else if (required) {
                std::cerr << "vulkan llm: missing required " << name << "\n";
                if (!missing_required) {
                    last_error_ = std::string("Missing required tensor ") + name;
                }
                missing_required = true;
            }
        };

        auto check_dims = [&](const char *suffix, int n_dims_min,
                              int64_t d0, int64_t d1, int64_t d2) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, suffix);
            int tidx = llm_find_tensor(g, name);
            if (tidx < 0) return;
            const gguf_tensor_info &ti = g->tensors[tidx];
            if ((int)ti.n_dims < n_dims_min) {
                if (!missing_required) {
                    last_error_ = std::string("Unexpected tensor rank for ") + name;
                }
                missing_required = true;
                return;
            }
            if (d0 >= 0 && (int64_t)ti.dims[0] != d0) {
                if (!missing_required) {
                    last_error_ = std::string("Unexpected dim[0] for ") + name;
                }
                missing_required = true;
            }
            if (d1 >= 0 && (int64_t)ti.dims[1] != d1) {
                if (!missing_required) {
                    last_error_ = std::string("Unexpected dim[1] for ") + name;
                }
                missing_required = true;
            }
            if (d2 >= 0 && (int64_t)ti.dims[2] != d2) {
                if (!missing_required) {
                    last_error_ = std::string("Unexpected dim[2] for ") + name;
                }
                missing_required = true;
            }
        };

        upload_blk("attn_norm",   ln.attn_norm, true);
        upload_blk("attn_q",      ln.attn_q, true);
        upload_blk("attn_k",      ln.attn_k, true);
        upload_blk("attn_v",      ln.attn_v, true);
        upload_blk("attn_q_norm", ln.attn_q_norm, false);
        upload_blk("attn_k_norm", ln.attn_k_norm, false);
        upload_blk("attn_output", ln.attn_output, true);
        upload_blk("ffn_norm",    ln.ffn_norm, true);
        if (use_moe_) {
            upload_blk("ffn_gate_inp",  ln.ffn_gate_inp, true);
            upload_blk("ffn_up_exps",   ln.ffn_up_exps, true);
            upload_blk("ffn_gate_exps", ln.ffn_gate_exps, true);
            upload_blk("ffn_down_exps", ln.ffn_down_exps, true);
            check_dims("ffn_gate_inp", 2, n_embd_, n_expert_, -1);
            check_dims("ffn_up_exps", 3, n_embd_, n_ff_expert_, n_expert_);
            check_dims("ffn_gate_exps", 3, n_embd_, n_ff_expert_, n_expert_);
            check_dims("ffn_down_exps", 3, n_ff_expert_, n_embd_, n_expert_);

            // Keep dense tensors optional in MoE models.
            upload_blk("ffn_gate",    ln.ffn_gate, false);
            upload_blk("ffn_up",      ln.ffn_up, false);
            upload_blk("ffn_down",    ln.ffn_down, false);
        } else {
            upload_blk("ffn_gate",    ln.ffn_gate, true);
            upload_blk("ffn_up",      ln.ffn_up, true);
            upload_blk("ffn_down",    ln.ffn_down, true);
        }

        if (l == 0 || (l + 1) % 7 == 0 || l == n_layers_ - 1) {
            std::cerr << "  uploaded layer " << l << "/" << n_layers_ << "\n";
        }
    }
    if (missing_required) {
        return false;
    }

    // Create scratch buffers
    int ffn_work = n_ff_;
    if (use_moe_ && n_ff_expert_ > ffn_work) ffn_work = n_ff_expert_;
    if (ffn_work <= 0) ffn_work = 1;
    int moe_router = use_moe_ ? n_expert_ : 1;
    if (moe_router <= 0) moe_router = 1;

    int q_dim = n_heads_ * head_dim_;
    buf_x_        = createGpuBuffer(n_embd_ * sizeof(float));
    buf_xb_       = createGpuBuffer(n_embd_ * sizeof(float));
    buf_xb2_      = createGpuBuffer(q_dim * sizeof(float));
    buf_q_        = createGpuBuffer(q_dim * sizeof(float));
    buf_k_        = createGpuBuffer(kv_dim * sizeof(float));
    buf_v_        = createGpuBuffer(kv_dim * sizeof(float));
    buf_ffn_gate_ = createGpuBuffer((size_t)ffn_work * sizeof(float));
    buf_ffn_up_   = createGpuBuffer((size_t)ffn_work * sizeof(float));
    buf_moe_router_ = createGpuBuffer((size_t)moe_router * sizeof(float));
    buf_logits_   = createGpuBuffer(n_vocab_ * sizeof(float));
    for (int d = 0; d < n_deepstack_; d++) {
        buf_ds_slices_[d] = createGpuBuffer(n_embd_ * sizeof(float));
    }
    scratch_created_ = true;

    // Create KV cache
    kv_key_bufs_.resize(n_layers_);
    kv_val_bufs_.resize(n_layers_);
    for (int l = 0; l < n_layers_; l++) {
        size_t cache_size = (size_t)max_seq_len_ * kv_dim * sizeof(float);
        kv_key_bufs_[l] = createGpuBuffer(cache_size);
        kv_val_bufs_[l] = createGpuBuffer(cache_size);

        // Zero-init
        std::vector<float> zeros(max_seq_len_ * kv_dim, 0.0f);
        uploadToBuffer(kv_key_bufs_[l], zeros.data(), cache_size);
        uploadToBuffer(kv_val_bufs_[l], zeros.data(), cache_size);
    }

    logits_cpu_.resize(n_vocab_);
    hidden_cpu_.resize(n_embd_);
    moe_router_cpu_.resize((size_t)moe_router);
    moe_accum_cpu_.resize((size_t)n_embd_);
    moe_tmp_cpu_.resize((size_t)n_embd_);

    weights_loaded_ = true;
    std::cerr << "vulkan llm: all weights uploaded to GPU\n";
    return true;
}

// ---- Forward pass ----

float *VulkanLLMRunner::forward(int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w,
                                 bool compute_logits) {
    if (cpu_fallback_moe_) {
        if (!cpu_model_) {
            last_error_ = "CPU fallback model is not initialized";
            return nullptr;
        }
        if (compute_logits) {
            if (!cpu_model_->use_mrope) {
                return transformer_forward_logits(cpu_model_, token_id, cache_pos);
            }
            return transformer_forward_logits_pos(cpu_model_, token_id, cache_pos, pos_t, pos_h, pos_w);
        }
        if (!cpu_model_->use_mrope) {
            return transformer_forward(cpu_model_, token_id, cache_pos);
        }
        return transformer_forward_pos(cpu_model_, token_id, cache_pos, pos_t, pos_h, pos_w);
    }
    if (!weights_loaded_) {
        last_error_ = "Weights are not loaded";
        return nullptr;
    }
    if (token_id < 0 || token_id >= n_vocab_) {
        last_error_ = "token_id out of range";
        std::cerr << "vulkan llm: token_id " << token_id << " out of range [0, " << n_vocab_ << ")\n";
        return nullptr;
    }
    // Token embedding: read row from GPU weight buffer via CPU
    // (mapped memory, no dispatch needed)
    {
        void *ptr = nullptr;
        runner_.mapBuffer(weight_bufs_["token_embd"], &ptr);
        const float *embd_data = reinterpret_cast<const float *>(ptr);
        const float *row = embd_data + (size_t)token_id * n_embd_;
        uploadToBuffer(buf_x_, row, n_embd_ * sizeof(float));
        runner_.unmapBuffer(weight_bufs_["token_embd"]);
    }

    return forwardInternal(cache_pos, pos_t, pos_h, pos_w, compute_logits);
}

float *VulkanLLMRunner::forwardEmbd(const float *embd, int cache_pos, int pos_t, int pos_h, int pos_w,
                                     bool compute_logits) {
    if (cpu_fallback_moe_) {
        if (!cpu_model_) {
            last_error_ = "CPU fallback model is not initialized";
            return nullptr;
        }
        cpu_model_->ds_embd_stride = ds_embd_stride_;
        if (compute_logits) {
            if (!cpu_model_->use_mrope) {
                return transformer_forward_embd_logits(cpu_model_, embd, cache_pos);
            }
            return transformer_forward_embd_logits_pos(cpu_model_, embd, cache_pos, pos_t, pos_h, pos_w);
        }
        if (!cpu_model_->use_mrope) {
            return transformer_forward_embd(cpu_model_, embd, cache_pos);
        }
        return transformer_forward_embd_pos(cpu_model_, embd, cache_pos, pos_t, pos_h, pos_w);
    }
    if (!weights_loaded_) {
        last_error_ = "Weights are not loaded";
        return nullptr;
    }
    if (!embd) {
        last_error_ = "Null embedding pointer";
        return nullptr;
    }
    uploadToBuffer(buf_x_, embd, n_embd_ * sizeof(float));
    // Upload deepstack slices if stride indicates they're present
    ds_active_ = false;
    if (n_deepstack_ > 0 && ds_embd_stride_ > n_embd_) {
        for (int d = 0; d < n_deepstack_; d++) {
            const float *slice = embd + (1 + d) * n_embd_;
            uploadToBuffer(buf_ds_slices_[d], slice, n_embd_ * sizeof(float));
        }
        ds_active_ = true;
    }
    float *result = forwardInternal(cache_pos, pos_t, pos_h, pos_w, compute_logits);
    ds_active_ = false;
    return result;
}

float *VulkanLLMRunner::forwardInternal(int cache_pos, int pos_t, int pos_h, int pos_w,
                                         bool compute_logits) {
    if (cache_pos < 0 || cache_pos >= max_seq_len_) {
        last_error_ = "cache_pos out of range for allocated KV cache";
        std::cerr << "vulkan llm: cache_pos " << cache_pos
                  << " out of range [0, " << max_seq_len_ << ")\n";
        return nullptr;
    }
    int kv_dim = n_kv_heads_ * head_dim_;
    int seq_len = cache_pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim_);

    for (int l = 0; l < n_layers_; l++) {
        auto &ln = layer_names_[l];

        // --- Attention ---
        // RMSNorm(x -> xb)
        GLLM_PROF_BEGIN("attn_norm", l, "rmsnorm", "FP32");
        dispatchRmsNorm(buf_x_, buf_xb_, weight_bufs_[ln.attn_norm], 1, n_embd_, rms_norm_eps_);
        GLLM_PROF_END("attn_norm", 5.0 * n_embd_, 0);

        // Q/K/V projections
        int q_dim = n_heads_ * head_dim_;
        GLLM_PROF_BEGIN("q_proj", l, "matvec", "FP32");
        dispatchMatvecAuto(buf_xb_, ln.attn_q, buf_q_, q_dim, n_embd_);
        GLLM_PROF_END("q_proj", 2.0 * q_dim * n_embd_, 0);

        GLLM_PROF_BEGIN("k_proj", l, "matvec", "FP32");
        dispatchMatvecAuto(buf_xb_, ln.attn_k, buf_k_, kv_dim, n_embd_);
        GLLM_PROF_END("k_proj", 2.0 * kv_dim * n_embd_, 0);

        GLLM_PROF_BEGIN("v_proj", l, "matvec", "FP32");
        dispatchMatvecAuto(buf_xb_, ln.attn_v, buf_v_, kv_dim, n_embd_);
        GLLM_PROF_END("v_proj", 2.0 * kv_dim * n_embd_, 0);

        // QK-Norm (if present)
        GLLM_PROF_BEGIN("qk_norm", l, "rmsnorm", "FP32");
        if (weight_bufs_.count(ln.attn_q_norm) && weight_bufs_[ln.attn_q_norm].buffer != VK_NULL_HANDLE) {
            dispatchQkNorm(buf_q_, weight_bufs_[ln.attn_q_norm], n_heads_, head_dim_, rms_norm_eps_);
        }
        if (weight_bufs_.count(ln.attn_k_norm) && weight_bufs_[ln.attn_k_norm].buffer != VK_NULL_HANDLE) {
            dispatchQkNorm(buf_k_, weight_bufs_[ln.attn_k_norm], n_kv_heads_, head_dim_, rms_norm_eps_);
        }
        GLLM_PROF_END("qk_norm", 5.0 * (n_heads_ + n_kv_heads_) * head_dim_, 0);

        // RoPE
        GLLM_PROF_BEGIN("rope", l, "rope", "FP32");
        if (use_mrope_) {
            dispatchRopeMrope(buf_q_, n_heads_, head_dim_, pos_t, pos_h, pos_w,
                              rope_freq_base_, mrope_sections_[0], mrope_sections_[1], mrope_sections_[2]);
            dispatchRopeMrope(buf_k_, n_kv_heads_, head_dim_, pos_t, pos_h, pos_w,
                              rope_freq_base_, mrope_sections_[0], mrope_sections_[1], mrope_sections_[2]);
        } else {
            dispatchRopeNeox(buf_q_, n_heads_, head_dim_, pos_t, rope_freq_base_);
            dispatchRopeNeox(buf_k_, n_kv_heads_, head_dim_, pos_t, rope_freq_base_);
        }
        GLLM_PROF_END("rope", 8.0 * (n_heads_ + n_kv_heads_) * head_dim_ / 2, 0);

        // KV cache store
        dispatchKvStore(buf_k_, buf_v_, kv_key_bufs_[l], kv_val_bufs_[l], cache_pos, kv_dim);

        // Attention decode
        GLLM_PROF_BEGIN("attention", l, "attention", "FP32");
        dispatchAttnDecode(buf_q_, kv_key_bufs_[l], kv_val_bufs_[l], buf_xb2_,
                           n_heads_, n_kv_heads_, head_dim_, seq_len, scale);
        GLLM_PROF_END("attention", 2.0 * n_heads_ * seq_len * head_dim_ * 2, 0);

        // Output projection
        GLLM_PROF_BEGIN("out_proj", l, "matvec", "FP32");
        dispatchMatvecAuto(buf_xb2_, ln.attn_output, buf_xb_, n_embd_, q_dim);
        GLLM_PROF_END("out_proj", 2.0 * n_embd_ * q_dim, 0);

        // Residual: x += xb
        dispatchAdd(buf_x_, buf_xb_, n_embd_);

        // --- FFN ---
        // RMSNorm(x -> xb)
        GLLM_PROF_BEGIN("ffn_norm", l, "rmsnorm", "FP32");
        dispatchRmsNorm(buf_x_, buf_xb_, weight_bufs_[ln.ffn_norm], 1, n_embd_, rms_norm_eps_);
        GLLM_PROF_END("ffn_norm", 5.0 * n_embd_, 0);

        const bool layer_has_moe = use_moe_ &&
                                   !ln.ffn_gate_inp.empty() &&
                                   !ln.ffn_up_exps.empty() &&
                                   !ln.ffn_gate_exps.empty() &&
                                   !ln.ffn_down_exps.empty() &&
                                   weight_bufs_.count(ln.ffn_gate_inp) &&
                                   weight_bufs_.count(ln.ffn_up_exps) &&
                                   weight_bufs_.count(ln.ffn_gate_exps) &&
                                   weight_bufs_.count(ln.ffn_down_exps);
        if (layer_has_moe) {
            GLLM_PROF_BEGIN("ffn_gate_inp", l, "matvec", "FP32");
            dispatchMatvecAuto(buf_xb_, ln.ffn_gate_inp, buf_moe_router_, n_expert_, n_embd_);
            GLLM_PROF_END("ffn_gate_inp", 2.0 * n_expert_ * n_embd_, 0);

            downloadFromBuffer(buf_moe_router_, moe_router_cpu_.data(), (size_t)n_expert_ * sizeof(float));
            llm_softmax(moe_router_cpu_.data(), n_expert_);

            std::vector<int> top_idx(n_expert_used_);
            std::vector<float> top_w(n_expert_used_);
            float wsum = 0.0f;
            for (int i = 0; i < n_expert_used_; i++) {
                int best = -1;
                float best_w = -1.0f;
                for (int e = 0; e < n_expert_; e++) {
                    float w = moe_router_cpu_[e];
                    if (w > best_w) {
                        best_w = w;
                        best = e;
                    }
                }
                top_idx[i] = best;
                top_w[i] = best_w;
                wsum += best_w;
                if (best >= 0) moe_router_cpu_[best] = -1.0f;
            }
            if (wsum > 0.0f) {
                for (int i = 0; i < n_expert_used_; i++) top_w[i] /= wsum;
            }

            std::fill(moe_accum_cpu_.begin(), moe_accum_cpu_.end(), 0.0f);
            for (int ei = 0; ei < n_expert_used_; ei++) {
                int e = top_idx[ei];
                if (e < 0) continue;
                float ew = top_w[ei];
                uint32_t up_off = (uint32_t)(e * n_ff_expert_);
                uint32_t down_off = (uint32_t)(e * n_embd_);

                GLLM_PROF_BEGIN("ffn_up_exp", l, "matvec", "FP32");
                dispatchMatvecAuto(buf_xb_, ln.ffn_up_exps, buf_ffn_up_, n_ff_expert_, n_embd_, up_off);
                GLLM_PROF_END("ffn_up_exp", 2.0 * n_ff_expert_ * n_embd_, 0);

                GLLM_PROF_BEGIN("ffn_gate_exp", l, "matvec", "FP32");
                dispatchMatvecAuto(buf_xb_, ln.ffn_gate_exps, buf_ffn_gate_, n_ff_expert_, n_embd_, up_off);
                GLLM_PROF_END("ffn_gate_exp", 2.0 * n_ff_expert_ * n_embd_, 0);

                GLLM_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
                dispatchSiluMul(buf_ffn_gate_, buf_ffn_up_, n_ff_expert_);
                GLLM_PROF_END("silu_mul", 5.0 * n_ff_expert_, 0);

                GLLM_PROF_BEGIN("ffn_down_exp", l, "matvec", "FP32");
                dispatchMatvecAuto(buf_ffn_gate_, ln.ffn_down_exps, buf_xb_, n_embd_, n_ff_expert_, down_off);
                GLLM_PROF_END("ffn_down_exp", 2.0 * n_embd_ * n_ff_expert_, 0);

                downloadFromBuffer(buf_xb_, moe_tmp_cpu_.data(), (size_t)n_embd_ * sizeof(float));
                for (int i = 0; i < n_embd_; i++) {
                    moe_accum_cpu_[i] += ew * moe_tmp_cpu_[i];
                }
            }
            uploadToBuffer(buf_xb_, moe_accum_cpu_.data(), (size_t)n_embd_ * sizeof(float));
            dispatchAdd(buf_x_, buf_xb_, n_embd_);
        } else {
            // Dense SwiGLU fallback path
            if (!weight_bufs_.count(ln.ffn_gate) ||
                !weight_bufs_.count(ln.ffn_up) ||
                !weight_bufs_.count(ln.ffn_down)) {
                last_error_ = "Missing FFN tensors for dense fallback path";
                return nullptr;
            }
            GLLM_PROF_BEGIN("ffn_gate", l, "matvec", "FP32");
            dispatchMatvecAuto(buf_xb_, ln.ffn_gate, buf_ffn_gate_, n_ff_, n_embd_);
            GLLM_PROF_END("ffn_gate", 2.0 * n_ff_ * n_embd_, 0);

            GLLM_PROF_BEGIN("ffn_up", l, "matvec", "FP32");
            dispatchMatvecAuto(buf_xb_, ln.ffn_up, buf_ffn_up_, n_ff_, n_embd_);
            GLLM_PROF_END("ffn_up", 2.0 * n_ff_ * n_embd_, 0);

            GLLM_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
            dispatchSiluMul(buf_ffn_gate_, buf_ffn_up_, n_ff_);
            GLLM_PROF_END("silu_mul", 5.0 * n_ff_, 0);

            GLLM_PROF_BEGIN("ffn_down", l, "matvec", "FP32");
            dispatchMatvecAuto(buf_ffn_gate_, ln.ffn_down, buf_xb_, n_embd_, n_ff_);
            GLLM_PROF_END("ffn_down", 2.0 * n_embd_ * n_ff_, 0);

            dispatchAdd(buf_x_, buf_xb_, n_embd_);
        }

        // DeepStack injection (Qwen3-VL): add deepstack slice after early layers
        if (ds_active_ && l < n_deepstack_) {
            dispatchAdd(buf_x_, buf_ds_slices_[l], n_embd_);
        }

        if (l == 0 || l == n_layers_ - 1 || (l + 1) % 10 == 0) {
            // Download hidden state for debug norm
            downloadFromBuffer(buf_x_, hidden_cpu_.data(), n_embd_ * sizeof(float));
            float norm = 0.0f;
            for (int i = 0; i < n_embd_; i++) norm += hidden_cpu_[i] * hidden_cpu_[i];
            std::cerr << "  layer " << l << ": hidden norm = " << sqrtf(norm) << "\n";
        }
    }

    // Final RMSNorm
    GLLM_PROF_BEGIN("final_norm", -1, "rmsnorm", "FP32");
    dispatchRmsNorm(buf_x_, buf_xb_, weight_bufs_["output_norm"], 1, n_embd_, rms_norm_eps_);
    GLLM_PROF_END("final_norm", 5.0 * n_embd_, 0);

    if (compute_logits) {
        // LM head: logits = output_weight @ hidden
        GLLM_PROF_BEGIN("lm_head", -1, "matvec", "FP32");
        const std::string output_key = weight_bufs_.count("output") ? "output" : "token_embd";
        dispatchMatvecAuto(buf_xb_, output_key, buf_logits_, n_vocab_, n_embd_);
        GLLM_PROF_END("lm_head", 2.0 * n_vocab_ * n_embd_, 0);

        // Download logits
        downloadFromBuffer(buf_logits_, logits_cpu_.data(), n_vocab_ * sizeof(float));
        return logits_cpu_.data();
    } else {
        // Download hidden state
        downloadFromBuffer(buf_xb_, hidden_cpu_.data(), n_embd_ * sizeof(float));
        return hidden_cpu_.data();
    }
}

// ---- Batched Prefill ----

bool VulkanLLMRunner::enablePrefill(int max_batch_size) {
    if (!weights_loaded_) {
        last_error_ = "Weights must be loaded before enabling prefill";
        return false;
    }
    if (prefill_enabled_) return true;

    int N = max_batch_size;
    int q_dim = n_heads_ * head_dim_;
    int kv_dim = n_kv_heads_ * head_dim_;

    buf_pfx_         = createGpuBuffer((size_t)N * n_embd_ * sizeof(float));
    buf_pfxb_        = createGpuBuffer((size_t)N * n_embd_ * sizeof(float));
    buf_pfq_         = createGpuBuffer((size_t)N * q_dim * sizeof(float));
    buf_pfk_         = createGpuBuffer((size_t)N * kv_dim * sizeof(float));
    buf_pfv_         = createGpuBuffer((size_t)N * kv_dim * sizeof(float));
    buf_pf_attn_out_ = createGpuBuffer((size_t)N * q_dim * sizeof(float));
    buf_pf_ffn_gate_ = createGpuBuffer((size_t)N * n_ff_ * sizeof(float));
    buf_pf_ffn_up_   = createGpuBuffer((size_t)N * n_ff_ * sizeof(float));
    buf_positions_   = createGpuBuffer((size_t)N * 3 * sizeof(int));
    buf_cache_pos_   = createGpuBuffer((size_t)N * sizeof(int));

    // 96MB scratch for dequanted F16 weights (covers largest weight matrix)
    size_t weight_f16_size = 96 * 1024 * 1024;
    buf_weight_f16_  = createGpuBuffer(weight_f16_size);

    max_prefill_tokens_ = N;
    prefill_enabled_ = true;
    std::cerr << "vulkan llm: prefill enabled, max_batch=" << N << "\n";
    return true;
}

float *VulkanLLMRunner::prefillTokens(const int32_t *tokens, int n_tokens,
                                       const int *cache_pos, const int *pos_t,
                                       const int *pos_h, const int *pos_w) {
    if (!prefill_enabled_) {
        last_error_ = "Prefill not enabled";
        return nullptr;
    }
    if (n_tokens > max_prefill_tokens_) {
        last_error_ = "n_tokens exceeds max_prefill_tokens";
        return nullptr;
    }

    // Look up token embeddings and upload batch
    std::vector<float> embds((size_t)n_tokens * n_embd_);
    {
        void *ptr = nullptr;
        runner_.mapBuffer(weight_bufs_["token_embd"], &ptr);
        const float *embd_data = reinterpret_cast<const float *>(ptr);
        for (int i = 0; i < n_tokens; i++) {
            const float *row = embd_data + (size_t)tokens[i] * n_embd_;
            std::memcpy(embds.data() + (size_t)i * n_embd_, row, n_embd_ * sizeof(float));
        }
        runner_.unmapBuffer(weight_bufs_["token_embd"]);
    }
    uploadToBuffer(buf_pfx_, embds.data(), (size_t)n_tokens * n_embd_ * sizeof(float));

    // Upload positions and cache positions
    uploadToBuffer(buf_cache_pos_, cache_pos, n_tokens * sizeof(int));
    {
        std::vector<int> pos3(n_tokens * 3);
        for (int i = 0; i < n_tokens; i++) {
            pos3[i * 3 + 0] = pos_t[i];
            pos3[i * 3 + 1] = pos_h[i];
            pos3[i * 3 + 2] = pos_w[i];
        }
        uploadToBuffer(buf_positions_, pos3.data(), pos3.size() * sizeof(int));
    }

    return forwardBatch(n_tokens, cache_pos, pos_t, pos_h, pos_w, true);
}

float *VulkanLLMRunner::prefillEmbds(const float *embd, int n_tokens, int embd_stride,
                                      const int *cache_pos, const int *pos_t,
                                      const int *pos_h, const int *pos_w,
                                      const float **ds_embds, int ds_embd_stride_in) {
    if (!prefill_enabled_) {
        last_error_ = "Prefill not enabled";
        return nullptr;
    }
    if (n_tokens > max_prefill_tokens_) {
        last_error_ = "n_tokens exceeds max_prefill_tokens";
        return nullptr;
    }

    // Upload embeddings (extract n_embd from potentially larger stride)
    if (embd_stride == n_embd_) {
        uploadToBuffer(buf_pfx_, embd, (size_t)n_tokens * n_embd_ * sizeof(float));
    } else {
        std::vector<float> packed((size_t)n_tokens * n_embd_);
        for (int i = 0; i < n_tokens; i++) {
            std::memcpy(packed.data() + (size_t)i * n_embd_,
                        embd + (size_t)i * embd_stride,
                        n_embd_ * sizeof(float));
        }
        uploadToBuffer(buf_pfx_, packed.data(), packed.size() * sizeof(float));
    }

    // Upload deepstack slices for batch
    // For batched prefill, we need to handle deepstack differently:
    // We'll upload per-token deepstack slices and add them in forwardBatch.
    ds_embd_stride_ = ds_embd_stride_in > 0 ? ds_embd_stride_in : embd_stride;

    // Upload positions
    uploadToBuffer(buf_cache_pos_, cache_pos, n_tokens * sizeof(int));
    {
        std::vector<int> pos3(n_tokens * 3);
        for (int i = 0; i < n_tokens; i++) {
            pos3[i * 3 + 0] = pos_t[i];
            pos3[i * 3 + 1] = pos_h[i];
            pos3[i * 3 + 2] = pos_w[i];
        }
        uploadToBuffer(buf_positions_, pos3.data(), pos3.size() * sizeof(int));
    }

    // Store deepstack data pointers for forwardBatch to handle
    // For now we upload deepstack slices into ds_slices buffers per-layer in forwardBatch
    // We need to pass the raw embedding data through; store it temporarily
    // Actually for simplicity, handle deepstack in forwardBatch by uploading per-token
    // deepstack slices from CPU. This is small data (n_tokens * n_embd * n_ds * 4 bytes).
    // We'll pass ds_embds and embd to forwardBatch via member vars.

    // Store for forwardBatch to access
    const float *embd_for_ds = (ds_embds || (embd_stride > n_embd_ && n_deepstack_ > 0)) ? embd : nullptr;

    float *result = forwardBatch(n_tokens, cache_pos, pos_t, pos_h, pos_w, true);

    return result;
}

float *VulkanLLMRunner::forwardBatch(int n_tokens, const int *cache_positions,
                                       const int *pos_t, const int *pos_h, const int *pos_w,
                                       bool compute_logits) {
    if (!prefill_enabled_ || n_tokens <= 0) return nullptr;

    int kv_dim = n_kv_heads_ * head_dim_;
    int q_dim = n_heads_ * head_dim_;
    float scale = 1.0f / sqrtf((float)head_dim_);

    for (int l = 0; l < n_layers_; l++) {
        auto &ln = layer_names_[l];

        // --- Attention ---
        // Batched RMSNorm: [n_tokens, n_embd] -> [n_tokens, n_embd]
        dispatchRmsNorm(buf_pfx_, buf_pfxb_, weight_bufs_[ln.attn_norm], n_tokens, n_embd_, rms_norm_eps_);

        // Q/K/V projections via GEMM
        // Q[n_tok, q_dim] = xb[n_tok, n_embd] * W_q^T[q_dim, n_embd]
        auto it_q = weight_types_.find(ln.attn_q);
        bool q8_q = (it_q != weight_types_.end() && it_q->second == GGML_TYPE_Q8_0);
        if (q8_q) {
            dispatchDequantQ8(weight_bufs_[ln.attn_q], buf_weight_f16_, q_dim, n_embd_);
            dispatchMatmulCoopmatNT(buf_pfxb_, buf_weight_f16_, buf_pfq_, n_tokens, q_dim, n_embd_);
        } else {
            // Fallback: per-token matvec (slow but correct for F32 weights)
            for (int t = 0; t < n_tokens; t++) {
                // Would need offset views. For now, unsupported.
                last_error_ = "Batch GEMM requires Q8_0 weights";
                return nullptr;
            }
        }

        // K projection
        auto it_k = weight_types_.find(ln.attn_k);
        bool q8_k = (it_k != weight_types_.end() && it_k->second == GGML_TYPE_Q8_0);
        if (q8_k) {
            dispatchDequantQ8(weight_bufs_[ln.attn_k], buf_weight_f16_, kv_dim, n_embd_);
            dispatchMatmulCoopmatNT(buf_pfxb_, buf_weight_f16_, buf_pfk_, n_tokens, kv_dim, n_embd_);
        } else {
            last_error_ = "Batch GEMM requires Q8_0 weights";
            return nullptr;
        }

        // V projection
        auto it_v = weight_types_.find(ln.attn_v);
        bool q8_v = (it_v != weight_types_.end() && it_v->second == GGML_TYPE_Q8_0);
        if (q8_v) {
            dispatchDequantQ8(weight_bufs_[ln.attn_v], buf_weight_f16_, kv_dim, n_embd_);
            dispatchMatmulCoopmatNT(buf_pfxb_, buf_weight_f16_, buf_pfv_, n_tokens, kv_dim, n_embd_);
        } else {
            last_error_ = "Batch GEMM requires Q8_0 weights";
            return nullptr;
        }

        // QK-Norm (batched: treat as n_tokens*n_heads / n_tokens*n_kv_heads heads)
        if (weight_bufs_.count(ln.attn_q_norm) && weight_bufs_[ln.attn_q_norm].buffer != VK_NULL_HANDLE) {
            dispatchQkNorm(buf_pfq_, weight_bufs_[ln.attn_q_norm], n_tokens * n_heads_, head_dim_, rms_norm_eps_);
        }
        if (weight_bufs_.count(ln.attn_k_norm) && weight_bufs_[ln.attn_k_norm].buffer != VK_NULL_HANDLE) {
            dispatchQkNorm(buf_pfk_, weight_bufs_[ln.attn_k_norm], n_tokens * n_kv_heads_, head_dim_, rms_norm_eps_);
        }

        // Batched RoPE
        if (use_mrope_) {
            dispatchRopeMropeBatch(buf_pfq_, buf_positions_, n_tokens, n_heads_, head_dim_,
                                   rope_freq_base_, mrope_sections_[0], mrope_sections_[1], mrope_sections_[2]);
            dispatchRopeMropeBatch(buf_pfk_, buf_positions_, n_tokens, n_kv_heads_, head_dim_,
                                   rope_freq_base_, mrope_sections_[0], mrope_sections_[1], mrope_sections_[2]);
        } else {
            // For neox batch, positions buffer has 1 int per token (use pos_t slot)
            // Upload pos_t as single positions
            uploadToBuffer(buf_positions_, pos_t, n_tokens * sizeof(int));
            dispatchRopeNeoxBatch(buf_pfq_, buf_positions_, n_tokens, n_heads_, head_dim_, rope_freq_base_);
            dispatchRopeNeoxBatch(buf_pfk_, buf_positions_, n_tokens, n_kv_heads_, head_dim_, rope_freq_base_);
            // Restore 3-component positions for M-RoPE (if needed later)
            std::vector<int> pos3(n_tokens * 3);
            for (int i = 0; i < n_tokens; i++) {
                pos3[i * 3 + 0] = pos_t[i];
                pos3[i * 3 + 1] = pos_h[i];
                pos3[i * 3 + 2] = pos_w[i];
            }
            uploadToBuffer(buf_positions_, pos3.data(), pos3.size() * sizeof(int));
        }

        // Batched KV store
        dispatchKvStoreBatch(buf_pfk_, buf_pfv_, kv_key_bufs_[l], kv_val_bufs_[l],
                             buf_cache_pos_, n_tokens, kv_dim);

        // Prefill attention
        dispatchAttnPrefill(buf_pfq_, kv_key_bufs_[l], kv_val_bufs_[l], buf_pf_attn_out_,
                            buf_cache_pos_, n_tokens, n_heads_, n_kv_heads_, head_dim_,
                            max_seq_len_, scale);

        // Output projection: out[n_tok, n_embd] = attn_out[n_tok, q_dim] * W_o^T[n_embd, q_dim]
        auto it_o = weight_types_.find(ln.attn_output);
        if (it_o != weight_types_.end() && it_o->second == GGML_TYPE_Q8_0) {
            dispatchDequantQ8(weight_bufs_[ln.attn_output], buf_weight_f16_, n_embd_, q_dim);
            dispatchMatmulCoopmatNT(buf_pf_attn_out_, buf_weight_f16_, buf_pfxb_, n_tokens, n_embd_, q_dim);
        } else {
            last_error_ = "Batch GEMM requires Q8_0 weights";
            return nullptr;
        }

        // Residual: x += xb
        dispatchAdd(buf_pfx_, buf_pfxb_, n_tokens * n_embd_);

        // --- FFN ---
        dispatchRmsNorm(buf_pfx_, buf_pfxb_, weight_bufs_[ln.ffn_norm], n_tokens, n_embd_, rms_norm_eps_);

        // Dense FFN only for batch path (MoE not supported in batch)
        if (!weight_bufs_.count(ln.ffn_gate) || !weight_bufs_.count(ln.ffn_up) || !weight_bufs_.count(ln.ffn_down)) {
            last_error_ = "Batch prefill requires dense FFN weights";
            return nullptr;
        }

        // Gate
        auto it_gate = weight_types_.find(ln.ffn_gate);
        if (it_gate != weight_types_.end() && it_gate->second == GGML_TYPE_Q8_0) {
            dispatchDequantQ8(weight_bufs_[ln.ffn_gate], buf_weight_f16_, n_ff_, n_embd_);
            dispatchMatmulCoopmatNT(buf_pfxb_, buf_weight_f16_, buf_pf_ffn_gate_, n_tokens, n_ff_, n_embd_);
        } else {
            last_error_ = "Batch GEMM requires Q8_0 weights";
            return nullptr;
        }

        // Up
        auto it_up = weight_types_.find(ln.ffn_up);
        if (it_up != weight_types_.end() && it_up->second == GGML_TYPE_Q8_0) {
            dispatchDequantQ8(weight_bufs_[ln.ffn_up], buf_weight_f16_, n_ff_, n_embd_);
            dispatchMatmulCoopmatNT(buf_pfxb_, buf_weight_f16_, buf_pf_ffn_up_, n_tokens, n_ff_, n_embd_);
        } else {
            last_error_ = "Batch GEMM requires Q8_0 weights";
            return nullptr;
        }

        // SiLU(gate) * up
        dispatchSiluMul(buf_pf_ffn_gate_, buf_pf_ffn_up_, n_tokens * n_ff_);

        // Down
        auto it_down = weight_types_.find(ln.ffn_down);
        if (it_down != weight_types_.end() && it_down->second == GGML_TYPE_Q8_0) {
            dispatchDequantQ8(weight_bufs_[ln.ffn_down], buf_weight_f16_, n_embd_, n_ff_);
            dispatchMatmulCoopmatNT(buf_pf_ffn_gate_, buf_weight_f16_, buf_pfxb_, n_tokens, n_embd_, n_ff_);
        } else {
            last_error_ = "Batch GEMM requires Q8_0 weights";
            return nullptr;
        }

        // Residual
        dispatchAdd(buf_pfx_, buf_pfxb_, n_tokens * n_embd_);

        // DeepStack injection — upload and add per-token deepstack slices
        // This is handled by the caller setting up buf_pfx_ with deepstack-aware embeddings.
        // For now, deepstack is not supported in batch mode.
        // TODO: implement batched deepstack injection

        if (l == 0 || l == n_layers_ - 1 || (l + 1) % 10 == 0) {
            // Download last token's hidden state for debug norm
            std::vector<float> hid(n_embd_);
            size_t last_tok_offset = (size_t)(n_tokens - 1) * n_embd_ * sizeof(float);
            void *ptr = nullptr;
            runner_.mapBuffer(buf_pfx_, &ptr);
            std::memcpy(hid.data(), (const uint8_t *)ptr + last_tok_offset, n_embd_ * sizeof(float));
            runner_.unmapBuffer(buf_pfx_);
            float norm = 0.0f;
            for (int i = 0; i < n_embd_; i++) norm += hid[i] * hid[i];
            std::cerr << "  layer " << l << ": hidden norm = " << sqrtf(norm) << "\n";
        }
    }

    // Final RMSNorm on last token
    // Extract last token from batch into single-token buffer
    {
        void *ptr = nullptr;
        runner_.mapBuffer(buf_pfx_, &ptr);
        const float *last_tok = reinterpret_cast<const float *>(
            (const uint8_t *)ptr + (size_t)(n_tokens - 1) * n_embd_ * sizeof(float));
        uploadToBuffer(buf_x_, last_tok, n_embd_ * sizeof(float));
        runner_.unmapBuffer(buf_pfx_);
    }

    dispatchRmsNorm(buf_x_, buf_xb_, weight_bufs_["output_norm"], 1, n_embd_, rms_norm_eps_);

    if (compute_logits) {
        const std::string output_key = weight_bufs_.count("output") ? "output" : "token_embd";
        dispatchMatvecAuto(buf_xb_, output_key, buf_logits_, n_vocab_, n_embd_);
        downloadFromBuffer(buf_logits_, logits_cpu_.data(), n_vocab_ * sizeof(float));
        return logits_cpu_.data();
    } else {
        downloadFromBuffer(buf_xb_, hidden_cpu_.data(), n_embd_ * sizeof(float));
        return hidden_cpu_.data();
    }
}
