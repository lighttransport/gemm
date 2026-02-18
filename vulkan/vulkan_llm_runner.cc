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
        runner_.destroyBuffer(buf_logits_);
    }

    // Destroy KV cache
    for (auto &b : kv_key_bufs_) runner_.destroyBuffer(b);
    for (auto &b : kv_val_bufs_) runner_.destroyBuffer(b);

    if (pipelines_created_) {
        runner_.destroyComputePipeline(pipe_rmsnorm_);
        runner_.destroyComputePipeline(pipe_matvec_);
        runner_.destroyComputePipeline(pipe_silu_mul_);
        runner_.destroyComputePipeline(pipe_rope_neox_);
        runner_.destroyComputePipeline(pipe_rope_mrope_);
        runner_.destroyComputePipeline(pipe_qknorm_);
        runner_.destroyComputePipeline(pipe_kv_store_);
        runner_.destroyComputePipeline(pipe_attn_decode_);
        runner_.destroyComputePipeline(pipe_add_);
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
    // matvec_f32: 3 buffers, push {N, K} = 8 bytes
    {
        auto spirv = load("matvec_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(3), 8, pipe_matvec_))
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
    // attn_decode_f32: 4 buffers, push {n_heads, n_kv_heads, head_dim, seq_len, max_seq_len, scale} = 24 bytes
    {
        auto spirv = load("attn_decode_f32");
        if (spirv.empty()) return false;
        if (!runner_.createComputePipelineWithPushConstants(spirv, makeBindings(4), 24, pipe_attn_decode_))
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
                                      uint32_t N, uint32_t K) {
    std::vector<BufInfo> bufs = {x, W, dst};
    runner_.updateDescriptorSet(pipe_matvec_, bufs);
    struct { uint32_t N, K; } pc = {N, K};
    runner_.beginRecording();
    runner_.bindComputePipeline(pipe_matvec_);
    runner_.bindDescriptorSets(pipe_matvec_);
    runner_.pushConstants(pipe_matvec_, &pc, sizeof(pc));
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
    t.n_rows = (g->tensors[idx].n_dims >= 2) ? (int)g->tensors[idx].dims[1] : 1;
    return t;
}

bool VulkanLLMRunner::loadWeights(gguf_context *g) {
    if (!initialized_) {
        last_error_ = "Not initialized";
        return false;
    }

    // Detect architecture
    const char *arch = "qwen2";
    if (gguf_find_key(g, "qwen3vl.block_count") >= 0) arch = "qwen3vl";
    else if (gguf_find_key(g, "qwen3.block_count") >= 0) arch = "qwen3";

    char kbuf[128];
    #define ARCH_KEY(suffix) (snprintf(kbuf, sizeof(kbuf), "%s." suffix, arch), kbuf)

    n_embd_      = llm_get_int(g, ARCH_KEY("embedding_length"), 4096);
    n_heads_     = llm_get_int(g, ARCH_KEY("attention.head_count"), 32);
    n_kv_heads_  = llm_get_int(g, ARCH_KEY("attention.head_count_kv"), 8);
    n_layers_    = llm_get_int(g, ARCH_KEY("block_count"), 36);
    n_ff_        = llm_get_int(g, ARCH_KEY("feed_forward_length"), 12288);
    n_vocab_     = llm_get_int(g, ARCH_KEY("vocab_size"), 0);
    rms_norm_eps_ = llm_get_float(g, ARCH_KEY("attention.layer_norm_rms_epsilon"), 1e-6f);
    rope_freq_base_ = llm_get_float(g, ARCH_KEY("rope.freq_base"), 5000000.0f);
    head_dim_    = n_embd_ / n_heads_;
    max_seq_len_ = 1024; // default, can be made configurable

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
    #undef ARCH_KEY

    int kv_dim = n_kv_heads_ * head_dim_;

    std::cerr << "vulkan llm: arch=" << arch << " n_embd=" << n_embd_ << " n_heads=" << n_heads_
              << " n_kv_heads=" << n_kv_heads_ << " n_layers=" << n_layers_ << " n_ff=" << n_ff_
              << " head_dim=" << head_dim_ << " n_vocab=" << n_vocab_ << "\n";

    // Create pipelines
    shader_dir_ = ".";
    if (!createPipelines()) return false;

    // Upload global tensors
    std::cerr << "vulkan llm: uploading token_embd...\n";
    {
        qtensor t = llm_load_qt(g, "token_embd.weight");
        if (t.data) {
            if (n_vocab_ == 0) n_vocab_ = t.n_rows;
            uploadTensor("token_embd", t);
        }
    }
    {
        qtensor t = llm_load_qt(g, "output_norm.weight");
        if (t.data) uploadTensor("output_norm", t);
    }
    {
        qtensor t = llm_load_qt(g, "output.weight");
        if (t.data) {
            uploadTensor("output", t);
        } else {
            // Weight tying: output shares token_embd
            std::cerr << "vulkan llm: weight-tied output (using token_embd)\n";
        }
    }

    // Per-layer tensors
    layer_names_.resize(n_layers_);
    for (int l = 0; l < n_layers_; l++) {
        char name[128];
        auto &ln = layer_names_[l];

        auto upload_blk = [&](const char *suffix, std::string &key, bool required) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, suffix);
            key = name;
            qtensor t = llm_load_qt(g, name);
            if (t.data) {
                uploadTensor(key, t);
            } else if (required) {
                std::cerr << "vulkan llm: warning: missing " << name << "\n";
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
        upload_blk("ffn_gate",    ln.ffn_gate, true);
        upload_blk("ffn_up",      ln.ffn_up, true);
        upload_blk("ffn_down",    ln.ffn_down, true);

        if (l == 0 || (l + 1) % 7 == 0 || l == n_layers_ - 1) {
            std::cerr << "  uploaded layer " << l << "/" << n_layers_ << "\n";
        }
    }

    // Create scratch buffers
    buf_x_        = createGpuBuffer(n_embd_ * sizeof(float));
    buf_xb_       = createGpuBuffer(n_embd_ * sizeof(float));
    buf_xb2_      = createGpuBuffer(n_embd_ * sizeof(float));
    buf_q_        = createGpuBuffer(n_embd_ * sizeof(float));
    buf_k_        = createGpuBuffer(kv_dim * sizeof(float));
    buf_v_        = createGpuBuffer(kv_dim * sizeof(float));
    buf_ffn_gate_ = createGpuBuffer(n_ff_ * sizeof(float));
    buf_ffn_up_   = createGpuBuffer(n_ff_ * sizeof(float));
    buf_logits_   = createGpuBuffer(n_vocab_ * sizeof(float));
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

    weights_loaded_ = true;
    std::cerr << "vulkan llm: all weights uploaded to GPU\n";
    return true;
}

// ---- Forward pass ----

float *VulkanLLMRunner::forward(int32_t token_id, int cache_pos, int pos_t, int pos_h, int pos_w,
                                 bool compute_logits) {
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
    uploadToBuffer(buf_x_, embd, n_embd_ * sizeof(float));
    return forwardInternal(cache_pos, pos_t, pos_h, pos_w, compute_logits);
}

float *VulkanLLMRunner::forwardInternal(int cache_pos, int pos_t, int pos_h, int pos_w,
                                         bool compute_logits) {
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
        GLLM_PROF_BEGIN("q_proj", l, "matvec", "FP32");
        dispatchMatvec(buf_xb_, weight_bufs_[ln.attn_q], buf_q_, n_embd_, n_embd_);
        GLLM_PROF_END("q_proj", 2.0 * n_embd_ * n_embd_, 0);

        GLLM_PROF_BEGIN("k_proj", l, "matvec", "FP32");
        dispatchMatvec(buf_xb_, weight_bufs_[ln.attn_k], buf_k_, kv_dim, n_embd_);
        GLLM_PROF_END("k_proj", 2.0 * kv_dim * n_embd_, 0);

        GLLM_PROF_BEGIN("v_proj", l, "matvec", "FP32");
        dispatchMatvec(buf_xb_, weight_bufs_[ln.attn_v], buf_v_, kv_dim, n_embd_);
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
        dispatchMatvec(buf_xb2_, weight_bufs_[ln.attn_output], buf_xb_, n_embd_, n_embd_);
        GLLM_PROF_END("out_proj", 2.0 * n_embd_ * n_embd_, 0);

        // Residual: x += xb
        dispatchAdd(buf_x_, buf_xb_, n_embd_);

        // --- FFN ---
        // RMSNorm(x -> xb)
        GLLM_PROF_BEGIN("ffn_norm", l, "rmsnorm", "FP32");
        dispatchRmsNorm(buf_x_, buf_xb_, weight_bufs_[ln.ffn_norm], 1, n_embd_, rms_norm_eps_);
        GLLM_PROF_END("ffn_norm", 5.0 * n_embd_, 0);

        // Gate and Up projections
        GLLM_PROF_BEGIN("ffn_gate", l, "matvec", "FP32");
        dispatchMatvec(buf_xb_, weight_bufs_[ln.ffn_gate], buf_ffn_gate_, n_ff_, n_embd_);
        GLLM_PROF_END("ffn_gate", 2.0 * n_ff_ * n_embd_, 0);

        GLLM_PROF_BEGIN("ffn_up", l, "matvec", "FP32");
        dispatchMatvec(buf_xb_, weight_bufs_[ln.ffn_up], buf_ffn_up_, n_ff_, n_embd_);
        GLLM_PROF_END("ffn_up", 2.0 * n_ff_ * n_embd_, 0);

        // SiLU-Mul
        GLLM_PROF_BEGIN("silu_mul", l, "silu_mul", "FP32");
        dispatchSiluMul(buf_ffn_gate_, buf_ffn_up_, n_ff_);
        GLLM_PROF_END("silu_mul", 5.0 * n_ff_, 0);

        // Down projection
        GLLM_PROF_BEGIN("ffn_down", l, "matvec", "FP32");
        dispatchMatvec(buf_ffn_gate_, weight_bufs_[ln.ffn_down], buf_xb_, n_embd_, n_ff_);
        GLLM_PROF_END("ffn_down", 2.0 * n_embd_ * n_ff_, 0);

        // Residual: x += xb
        dispatchAdd(buf_x_, buf_xb_, n_embd_);

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
        const std::string &output_key = weight_bufs_.count("output") ? "output" : "token_embd";
        dispatchMatvec(buf_xb_, weight_bufs_[output_key], buf_logits_, n_vocab_, n_embd_);
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
