// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan Hunyuan3D-2.1 Runner implementation
//
// Pipeline: DINOv2 encoder -> DiT diffusion (flow matching) -> ShapeVAE -> MC mesh
// Uses pre-compiled SPIR-V shaders and HOST_VISIBLE SSBOs for simplicity.
//

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define MARCHING_CUBES_IMPLEMENTATION
#include "../../common/marching_cubes.h"

#include "vulkan_hy3d_runner.h"
#include "../deps/vulkan-runner.hh"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <ctime>

/* ======================================================================== */
/* Model constants                                                          */
/* ======================================================================== */

/* DINOv2-L config */
#define DINO_HIDDEN     1024
#define DINO_HEADS      16
#define DINO_HEAD_DIM   64
#define DINO_LAYERS     24
#define DINO_FFN        4096
#define DINO_PATCH      14
#define DINO_IMG_SIZE   518
#define DINO_NUM_PATCHES ((DINO_IMG_SIZE/DINO_PATCH)*(DINO_IMG_SIZE/DINO_PATCH)) /* 1369 */
#define DINO_SEQ_LEN    (DINO_NUM_PATCHES + 1)  /* 1370 */

/* DiT config */
#define DIT_INPUT_SIZE  4096
#define DIT_IN_CHANNELS 64
#define DIT_HIDDEN      2048
#define DIT_CONTEXT_DIM 1024
#define DIT_DEPTH       21
#define DIT_HEADS       16
#define DIT_HEAD_DIM    128
#define DIT_FFN         (DIT_HIDDEN * 4)    /* 8192 */
#define DIT_HALF_DEPTH  (DIT_DEPTH / 2)     /* 10 */
#define DIT_MOE_START   15
#define DIT_N_EXPERTS   8
#define DIT_MOE_TOP_K   2

/* ShapeVAE config */
#define VAE_NUM_LATENTS  4096
#define VAE_EMBED_DIM    64
#define VAE_WIDTH        1024
#define VAE_HEADS        16
#define VAE_HEAD_DIM     64
#define VAE_DEC_LAYERS   16
#define VAE_NUM_FREQS    8
#define VAE_FOURIER_DIM  51   /* 3*(2*8+1) */

/* ======================================================================== */
/* Types                                                                    */
/* ======================================================================== */

using VulkanRunner = vl_cpp::vulkan::VulkanComputeRunner;
using BufInfo = VulkanRunner::BufferInfo;
using Pipeline = VulkanRunner::ComputePipeline;

/* DINOv2 per-layer weights */
struct dino_layer_vk {
    BufInfo ln1_w, ln1_b;
    BufInfo q_w, q_b, k_w, k_b, v_w, v_b;
    BufInfo out_w, out_b;
    BufInfo ls1;
    BufInfo ln2_w, ln2_b;
    BufInfo fc1_w, fc1_b, fc2_w, fc2_b;
    BufInfo ls2;
    bool has_ls1 = false, has_ls2 = false;
};

/* DiT per-block weights */
struct dit_block_vk {
    BufInfo norm1_w, norm1_b;
    BufInfo sa_qkv_w;
    BufInfo sa_out_w, sa_out_b;
    BufInfo sa_q_norm_w, sa_k_norm_w;
    bool has_sa_q_norm = false, has_sa_k_norm = false;

    BufInfo norm2_w, norm2_b;
    BufInfo ca_q_w;
    BufInfo ca_kv_w;
    BufInfo ca_out_w, ca_out_b;
    BufInfo ca_q_norm_w, ca_k_norm_w;
    bool has_ca_q_norm = false, has_ca_k_norm = false;

    BufInfo norm3_w, norm3_b;
    bool has_norm3 = false;

    /* Regular MLP (blocks 0-14) */
    BufInfo mlp_fc1_w, mlp_fc1_b;
    BufInfo mlp_fc2_w, mlp_fc2_b;

    /* MoE (blocks 15-20) */
    bool use_moe = false;
    BufInfo moe_gate_w;
    BufInfo moe_expert_fc1_w[DIT_N_EXPERTS], moe_expert_fc1_b[DIT_N_EXPERTS];
    BufInfo moe_expert_fc2_w[DIT_N_EXPERTS], moe_expert_fc2_b[DIT_N_EXPERTS];
    BufInfo moe_shared_fc1_w, moe_shared_fc1_b;
    BufInfo moe_shared_fc2_w, moe_shared_fc2_b;

    /* Skip connection (blocks 11-20) */
    bool use_skip = false;
    BufInfo skip_linear_w, skip_linear_b;
    BufInfo skip_norm_w, skip_norm_b;
};

/* ShapeVAE transformer block weights */
struct vae_block_vk {
    BufInfo ln1_w, ln1_b;
    BufInfo qkv_w;
    BufInfo proj_w, proj_b;
    BufInfo q_norm_w, q_norm_b;
    BufInfo k_norm_w, k_norm_b;
    bool use_qk_norm = false;
    BufInfo ln2_w, ln2_b;
    BufInfo mlp_fc_w, mlp_fc_b;
    BufInfo mlp_proj_w, mlp_proj_b;
};

/* ShapeVAE geometry decoder weights */
struct vae_geo_decoder_vk {
    BufInfo query_proj_w, query_proj_b;
    BufInfo ln1_w, ln1_b;
    BufInfo ln2_w, ln2_b;
    BufInfo c_q_w;
    BufInfo c_kv_w;
    BufInfo c_proj_w, c_proj_b;
    BufInfo q_norm_w, q_norm_b;
    BufInfo k_norm_w, k_norm_b;
    bool use_qk_norm = false;
    BufInfo ln3_w, ln3_b;
    BufInfo mlp_fc_w, mlp_fc_b;
    BufInfo mlp_proj_w, mlp_proj_b;
    BufInfo ln_post_w, ln_post_b;
    bool has_ln_post = false;
    BufInfo output_w, output_b;
};

struct vulkan_hy3d_runner {
    VulkanRunner runner;
    int verbose;
    std::string shader_dir;

    /* Pipelines */
    Pipeline pipe_layernorm;         /* 4 bufs: src, dst, weight, bias; push: {n_tokens, dim, eps} = 12B */
    Pipeline pipe_gemm_f32;          /* 4 bufs: Y, W, X, bias; push: {n_out, n_in, n_tok, has_bias} = 16B */
    Pipeline pipe_gelu;              /* 1 buf: data; push: {n} = 4B */
    Pipeline pipe_add;               /* 2 bufs: dst, src; push: {n} = 4B */
    Pipeline pipe_layerscale_add;    /* 3 bufs: dst, src, scale; push: {n, dim} = 8B */
    Pipeline pipe_rms_norm;          /* 2 bufs: data, w; push: {n_tok, n_heads, head_dim, stride, eps} = 20B */
    Pipeline pipe_qk_layernorm;      /* 3 bufs: data, w, b; push: same as rms_norm = 20B */
    Pipeline pipe_cross_attn;        /* 4 bufs: out, Q, K, V; push: {q_len, kv_len, dim, n_heads, head_dim, scale} = 24B */
    Pipeline pipe_fourier_embed;     /* 3 bufs: out, coords, freqs; push: {N, num_freqs, out_dim} = 12B */
    Pipeline pipe_timestep_embed;    /* 1 buf: out; push: {t, dim} = 8B */
    Pipeline pipe_euler_step;        /* 2 bufs: x, v; push: {dt, n} = 8B */
    Pipeline pipe_cfg_combine;       /* 3 bufs: out, cond, uncond; push: {scale, n} = 8B */
    Pipeline pipe_split_qkv;         /* 4 bufs: Q, K, V, qkv; push: {N, H, HD} = 12B */
    Pipeline pipe_split_kv;          /* 3 bufs: K, V, kv; push: {M, H, HD} = 12B */
    Pipeline pipe_broadcast_add;     /* 2 bufs: dst, src; push: {n, dim} = 8B */
    Pipeline pipe_concat_first;      /* 3 bufs: out, token, seq; push: {N, dim} = 8B */
    Pipeline pipe_strip_first;       /* 2 bufs: dst, src; push: {N, dim} = 8B */
    Pipeline pipe_concat_last_dim;   /* 3 bufs: out, a, b; push: {N, dim} = 8B */
    Pipeline pipe_patch_embed;       /* 4 bufs: image, kernel_w, bias, output; push: {n_patches, dim, kernel_size, gw, width, patch_size} = 24B */
    Pipeline pipe_matmul_bias_f32;   /* 4 bufs: A, W, C, bias; push: {M, N, K, has_bias} = 16B */
    bool pipelines_created = false;

    /* DINOv2 weights */
    BufInfo dino_patch_w, dino_patch_b;
    BufInfo dino_pos_emb;
    BufInfo dino_cls_token;
    BufInfo dino_final_ln_w, dino_final_ln_b;
    dino_layer_vk dino_layers[DINO_LAYERS];

    /* DiT weights */
    BufInfo dit_x_emb_w, dit_x_emb_b;
    BufInfo dit_t_mlp0_w, dit_t_mlp0_b;
    BufInfo dit_t_mlp2_w, dit_t_mlp2_b;
    BufInfo dit_final_ln_w, dit_final_ln_b;
    BufInfo dit_final_linear_w, dit_final_linear_b;
    dit_block_vk dit_blocks[DIT_DEPTH];

    /* ShapeVAE weights */
    BufInfo vae_post_kl_w, vae_post_kl_b;
    vae_block_vk vae_blocks[VAE_DEC_LAYERS];
    vae_geo_decoder_vk vae_geo;
    BufInfo vae_fourier_freqs;

    /* Scratch buffers */
    BufInfo scratch[8];
    size_t scratch_size[8];

    /* Load status */
    bool dino_loaded = false, dit_loaded = false, vae_loaded = false;
    bool initialized = false;
};

/* ======================================================================== */
/* Helpers                                                                  */
/* ======================================================================== */

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

static BufInfo createGpuBuffer(VulkanRunner &runner, size_t size) {
    BufInfo buf{};
    if (size == 0) return buf;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    runner.createBuffer(size, usage, props, buf);
    return buf;
}

static bool uploadToBuffer(VulkanRunner &runner, BufInfo &buf, const void *data, size_t size) {
    void *ptr = nullptr;
    if (!runner.mapBuffer(buf, &ptr)) return false;
    std::memcpy(ptr, data, size);
    runner.unmapBuffer(buf);
    return true;
}

static bool downloadFromBuffer(VulkanRunner &runner, const BufInfo &buf, void *data, size_t size) {
    void *ptr = nullptr;
    if (!runner.mapBuffer(buf, &ptr)) return false;
    std::memcpy(data, ptr, size);
    runner.unmapBuffer(buf);
    return true;
}

static void ensureScratch(vulkan_hy3d_runner *r, int idx, size_t bytes) {
    if (r->scratch_size[idx] < bytes) {
        if (r->scratch[idx].buffer != VK_NULL_HANDLE)
            r->runner.destroyBuffer(r->scratch[idx]);
        r->scratch[idx] = createGpuBuffer(r->runner, bytes);
        r->scratch_size[idx] = bytes;
    }
}

/* Upload safetensors tensor as F32 to GPU */
static BufInfo stUploadF32(VulkanRunner &runner, st_context *st, const char *name, int verbose) {
    BufInfo buf{};
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [WARN] tensor '%s' not found\n", name);
        return buf;
    }
    void *data = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "F32") == 0) {
        buf = createGpuBuffer(runner, nbytes);
        uploadToBuffer(runner, buf, data, nbytes);
    } else if (strcmp(dtype, "F16") == 0) {
        /* Convert F16 -> F32 */
        size_t n = nbytes / sizeof(uint16_t);
        std::vector<float> f32(n);
        const uint16_t *f16 = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t sign = (f16[i] >> 15) & 0x1;
            uint32_t exp = (f16[i] >> 10) & 0x1f;
            uint32_t mant = f16[i] & 0x3ff;
            uint32_t f;
            if (exp == 0) f = sign << 31;
            else if (exp == 31) f = (sign << 31) | 0x7f800000 | (mant << 13);
            else f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            memcpy(&f32[i], &f, sizeof(float));
        }
        buf = createGpuBuffer(runner, n * sizeof(float));
        uploadToBuffer(runner, buf, f32.data(), n * sizeof(float));
    } else if (strcmp(dtype, "BF16") == 0) {
        size_t n = nbytes / sizeof(uint16_t);
        std::vector<float> f32(n);
        const uint16_t *bf16 = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t f = (uint32_t)bf16[i] << 16;
            memcpy(&f32[i], &f, sizeof(float));
        }
        buf = createGpuBuffer(runner, n * sizeof(float));
        uploadToBuffer(runner, buf, f32.data(), n * sizeof(float));
    }
    return buf;
}

/* Fuse 3 F32 weight tensors [dim, in_dim] into one [3*dim, in_dim] on GPU */
static BufInfo stFuse3F32(VulkanRunner &runner, st_context *st,
                           const char *name_a, const char *name_b, const char *name_c,
                           int verbose) {
    /* Upload each individually then concatenate on CPU */
    int ia = safetensors_find(st, name_a);
    int ib = safetensors_find(st, name_b);
    int ic = safetensors_find(st, name_c);
    BufInfo buf{};
    if (ia < 0 || ib < 0 || ic < 0) {
        if (verbose) fprintf(stderr, "  [WARN] fuse: missing tensor(s)\n");
        return buf;
    }

    /* Get all as F32 CPU vectors */
    auto getF32 = [&](int idx) -> std::vector<float> {
        void *data = safetensors_data(st, idx);
        size_t nbytes = safetensors_nbytes(st, idx);
        const char *dtype = safetensors_dtype(st, idx);
        if (strcmp(dtype, "F32") == 0) {
            size_t n = nbytes / sizeof(float);
            std::vector<float> v(n);
            memcpy(v.data(), data, nbytes);
            return v;
        } else if (strcmp(dtype, "F16") == 0) {
            size_t n = nbytes / sizeof(uint16_t);
            std::vector<float> v(n);
            const uint16_t *f16 = (const uint16_t *)data;
            for (size_t i = 0; i < n; i++) {
                uint32_t sign = (f16[i] >> 15) & 0x1;
                uint32_t exp = (f16[i] >> 10) & 0x1f;
                uint32_t mant = f16[i] & 0x3ff;
                uint32_t f;
                if (exp == 0) f = sign << 31;
                else if (exp == 31) f = (sign << 31) | 0x7f800000 | (mant << 13);
                else f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                memcpy(&v[i], &f, sizeof(float));
            }
            return v;
        }
        return {};
    };

    auto va = getF32(ia);
    auto vb = getF32(ib);
    auto vc = getF32(ic);
    if (va.empty() || vb.empty() || vc.empty()) return buf;

    std::vector<float> combined;
    combined.reserve(va.size() + vb.size() + vc.size());
    combined.insert(combined.end(), va.begin(), va.end());
    combined.insert(combined.end(), vb.begin(), vb.end());
    combined.insert(combined.end(), vc.begin(), vc.end());

    buf = createGpuBuffer(runner, combined.size() * sizeof(float));
    uploadToBuffer(runner, buf, combined.data(), combined.size() * sizeof(float));
    return buf;
}

/* Fuse 2 F32 weight tensors */
static BufInfo stFuse2F32(VulkanRunner &runner, st_context *st,
                           const char *name_a, const char *name_b,
                           int verbose) {
    int ia = safetensors_find(st, name_a);
    int ib = safetensors_find(st, name_b);
    BufInfo buf{};
    if (ia < 0 || ib < 0) return buf;

    auto getF32 = [&](int idx) -> std::vector<float> {
        void *data = safetensors_data(st, idx);
        size_t nbytes = safetensors_nbytes(st, idx);
        const char *dtype = safetensors_dtype(st, idx);
        if (strcmp(dtype, "F32") == 0) {
            size_t n = nbytes / sizeof(float);
            std::vector<float> v(n);
            memcpy(v.data(), data, nbytes);
            return v;
        } else if (strcmp(dtype, "F16") == 0) {
            size_t n = nbytes / sizeof(uint16_t);
            std::vector<float> v(n);
            const uint16_t *f16 = (const uint16_t *)data;
            for (size_t i = 0; i < n; i++) {
                uint32_t sign = (f16[i] >> 15) & 0x1;
                uint32_t exp = (f16[i] >> 10) & 0x1f;
                uint32_t mant = f16[i] & 0x3ff;
                uint32_t f;
                if (exp == 0) f = sign << 31;
                else if (exp == 31) f = (sign << 31) | 0x7f800000 | (mant << 13);
                else f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                memcpy(&v[i], &f, sizeof(float));
            }
            return v;
        }
        return {};
    };

    auto va = getF32(ia);
    auto vb = getF32(ib);
    if (va.empty() || vb.empty()) return buf;

    std::vector<float> combined;
    combined.reserve(va.size() + vb.size());
    combined.insert(combined.end(), va.begin(), va.end());
    combined.insert(combined.end(), vb.begin(), vb.end());

    buf = createGpuBuffer(runner, combined.size() * sizeof(float));
    uploadToBuffer(runner, buf, combined.data(), combined.size() * sizeof(float));
    return buf;
}

static bool isValid(const BufInfo &buf) {
    return buf.buffer != VK_NULL_HANDLE;
}

/* ======================================================================== */
/* Dispatch helpers                                                         */
/* ======================================================================== */

/* Generic dispatch: bind buffers, push constants, dispatch, wait */
static void dispatchSync(vulkan_hy3d_runner *r, Pipeline &pipe,
                          const std::vector<BufInfo> &bufs,
                          const void *push_data, uint32_t push_size,
                          uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    auto ds = r->runner.allocateAndUpdateDescriptorSet(pipe, bufs);
    r->runner.beginRecording();
    r->runner.bindComputePipeline(pipe);
    r->runner.bindDescriptorSetDynamic(pipe, ds);
    if (push_data && push_size > 0)
        r->runner.pushConstants(pipe, push_data, push_size);
    r->runner.dispatch(gx, gy, gz);
    r->runner.endRecordingAndSubmit();
    r->runner.waitForCompletion();
}

/* ---- Op wrappers ---- */

/* LayerNorm: dst = LN(src, weight, bias) */
static void opLayerNorm(vulkan_hy3d_runner *r, BufInfo &dst, BufInfo &src,
                          BufInfo &w, BufInfo &b, int n_tok, int dim) {
    struct { uint32_t n_tokens, dim; float eps; } pc = {(uint32_t)n_tok, (uint32_t)dim, 1e-6f};
    dispatchSync(r, r->pipe_layernorm, {src, dst, w, b}, &pc, sizeof(pc), (uint32_t)n_tok);
}

/* GEMM: Y = X @ W^T + bias.  W[n_out, n_in], X[n_tok, n_in], Y[n_tok, n_out]
 * Uses the gemm_f32_f32 shader: bindings = {Y, W, X, bias}, push = {n_out, n_in, n_tok, has_bias} */
static void opGemm(vulkan_hy3d_runner *r, BufInfo &Y, BufInfo &W, BufInfo &X, BufInfo &bias,
                     int n_out, int n_in, int n_tok) {
    struct { uint32_t n_out, n_in, n_tok, has_bias; } pc = {
        (uint32_t)n_out, (uint32_t)n_in, (uint32_t)n_tok, isValid(bias) ? 1u : 0u
    };
    uint32_t gx = (uint32_t)((n_out + 63) / 64);
    uint32_t gy = (uint32_t)((n_tok + 15) / 16);
    dispatchSync(r, r->pipe_gemm_f32, {Y, W, X, bias}, &pc, sizeof(pc), gx, gy);
}

/* GELU in-place */
static void opGelu(vulkan_hy3d_runner *r, BufInfo &x, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchSync(r, r->pipe_gelu, {x}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Element-wise add: dst += src */
static void opAdd(vulkan_hy3d_runner *r, BufInfo &dst, BufInfo &src, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchSync(r, r->pipe_add, {dst, src}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* LayerScale add: dst[i] += src[i] * scale[i % dim] */
static void opLayerscaleAdd(vulkan_hy3d_runner *r, BufInfo &dst, BufInfo &src,
                              BufInfo &scale, int n, int dim) {
    struct { uint32_t n, dim; } pc = {(uint32_t)n, (uint32_t)dim};
    dispatchSync(r, r->pipe_layerscale_add, {dst, src, scale}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Per-head RMSNorm */
static void opRmsNorm(vulkan_hy3d_runner *r, BufInfo &data, BufInfo &w,
                        int n_tok, int n_heads, int head_dim, int stride) {
    struct { uint32_t n_tok, n_heads, head_dim, stride; float eps; } pc = {
        (uint32_t)n_tok, (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)stride, 1e-6f
    };
    int total = n_tok * n_heads;
    dispatchSync(r, r->pipe_rms_norm, {data, w}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Per-head QK LayerNorm */
static void opQkLayerNorm(vulkan_hy3d_runner *r, BufInfo &data, BufInfo &w, BufInfo &b,
                            int n_tok, int n_heads, int head_dim, int stride) {
    struct { uint32_t n_tok, n_heads, head_dim, stride; float eps; } pc = {
        (uint32_t)n_tok, (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)stride, 1e-6f
    };
    int total = n_tok * n_heads;
    dispatchSync(r, r->pipe_qk_layernorm, {data, w, b}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Cross-attention (also used for self-attention when q_len == kv_len) */
static void opCrossAttn(vulkan_hy3d_runner *r, BufInfo &out,
                          BufInfo &Q, BufInfo &K, BufInfo &V,
                          int q_len, int kv_len, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    struct { uint32_t q_len, kv_len, dim, n_heads, head_dim; float scale; } pc = {
        (uint32_t)q_len, (uint32_t)kv_len, (uint32_t)dim, (uint32_t)n_heads, (uint32_t)head_dim, scale
    };
    dispatchSync(r, r->pipe_cross_attn, {out, Q, K, V}, &pc, sizeof(pc),
                 (uint32_t)n_heads, (uint32_t)q_len);
}

static void opSelfAttn(vulkan_hy3d_runner *r, BufInfo &out,
                         BufInfo &Q, BufInfo &K, BufInfo &V,
                         int n_tok, int dim, int n_heads, int head_dim) {
    opCrossAttn(r, out, Q, K, V, n_tok, n_tok, dim, n_heads, head_dim);
}

/* Fourier 3D embedding */
static void opFourierEmbed(vulkan_hy3d_runner *r, BufInfo &out, BufInfo &coords,
                             BufInfo &freqs, int N, int num_freqs, int out_dim) {
    struct { uint32_t N, num_freqs, out_dim; } pc = {(uint32_t)N, (uint32_t)num_freqs, (uint32_t)out_dim};
    dispatchSync(r, r->pipe_fourier_embed, {out, coords, freqs}, &pc, sizeof(pc), (uint32_t)((N + 255) / 256));
}

/* Sinusoidal timestep embedding */
static void opTimestepEmbed(vulkan_hy3d_runner *r, BufInfo &out, float t, int dim) {
    struct { float t; uint32_t dim; } pc = {t, (uint32_t)dim};
    int half = dim / 2;
    dispatchSync(r, r->pipe_timestep_embed, {out}, &pc, sizeof(pc), (uint32_t)((half + 255) / 256));
}

/* Euler step: x = x - dt * v */
static void opEulerStep(vulkan_hy3d_runner *r, BufInfo &x, BufInfo &v, float dt, int n) {
    struct { float dt; uint32_t n; } pc = {dt, (uint32_t)n};
    dispatchSync(r, r->pipe_euler_step, {x, v}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* CFG combine: out = uncond + scale * (cond - uncond) */
static void opCfgCombine(vulkan_hy3d_runner *r, BufInfo &out, BufInfo &cond,
                           BufInfo &uncond, float scale, int n) {
    struct { float scale; uint32_t n; } pc = {scale, (uint32_t)n};
    dispatchSync(r, r->pipe_cfg_combine, {out, cond, uncond}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Split interleaved QKV */
static void opSplitQkv(vulkan_hy3d_runner *r, BufInfo &Q, BufInfo &K, BufInfo &V,
                         BufInfo &qkv, int N, int H, int HD) {
    struct { uint32_t N, H, HD; } pc = {(uint32_t)N, (uint32_t)H, (uint32_t)HD};
    int total = N * H * HD;
    dispatchSync(r, r->pipe_split_qkv, {Q, K, V, qkv}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Split interleaved KV */
static void opSplitKv(vulkan_hy3d_runner *r, BufInfo &K, BufInfo &V,
                        BufInfo &kv, int M, int H, int HD) {
    struct { uint32_t M, H, HD; } pc = {(uint32_t)M, (uint32_t)H, (uint32_t)HD};
    int total = M * H * HD;
    dispatchSync(r, r->pipe_split_kv, {K, V, kv}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Broadcast add: dst[i] += src[i % dim] */
static void opBroadcastAdd(vulkan_hy3d_runner *r, BufInfo &dst, BufInfo &src, int n, int dim) {
    struct { uint32_t n, dim; } pc = {(uint32_t)n, (uint32_t)dim};
    dispatchSync(r, r->pipe_broadcast_add, {dst, src}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Concat first: prepend token to sequence */
static void opConcatFirst(vulkan_hy3d_runner *r, BufInfo &out, BufInfo &token,
                            BufInfo &seq, int seq_len, int dim) {
    struct { uint32_t N, dim; } pc = {(uint32_t)seq_len, (uint32_t)dim};
    int total = (seq_len + 1) * dim;
    dispatchSync(r, r->pipe_concat_first, {out, token, seq}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Strip first token */
static void opStripFirst(vulkan_hy3d_runner *r, BufInfo &dst, BufInfo &src, int N_plus_1, int dim) {
    /* N in the shader = N_plus_1 - 1 = number of output tokens */
    struct { uint32_t N, dim; } pc = {(uint32_t)(N_plus_1 - 1), (uint32_t)dim};
    int total = (N_plus_1 - 1) * dim;
    dispatchSync(r, r->pipe_strip_first, {dst, src}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Concat along last dim: [N, dim] + [N, dim] -> [N, 2*dim] */
static void opConcatLastDim(vulkan_hy3d_runner *r, BufInfo &out, BufInfo &a, BufInfo &b,
                              int N, int dim) {
    struct { uint32_t N, dim; } pc = {(uint32_t)N, (uint32_t)dim};
    int total = N * 2 * dim;
    dispatchSync(r, r->pipe_concat_last_dim, {out, a, b}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Buffer copy via CPU (HOST_VISIBLE, so direct map+memcpy) */
static void bufferCopy(VulkanRunner &runner, BufInfo &dst, BufInfo &src, size_t bytes) {
    void *sp = nullptr, *dp = nullptr;
    runner.mapBuffer(src, &sp);
    runner.mapBuffer(dst, &dp);
    memcpy(dp, sp, bytes);
    runner.unmapBuffer(src);
    runner.unmapBuffer(dst);
}

/* memset buffer to zero */
static void bufferZero(VulkanRunner &runner, BufInfo &buf, size_t bytes) {
    void *ptr = nullptr;
    runner.mapBuffer(buf, &ptr);
    memset(ptr, 0, bytes);
    runner.unmapBuffer(buf);
}

/* ======================================================================== */
/* Pipeline creation                                                        */
/* ======================================================================== */

static bool createPipelines(vulkan_hy3d_runner *r) {
    auto loadSpv = [&](const std::string &name) -> std::vector<uint32_t> {
        /* Try multiple search paths for .spv shaders */
        const std::string candidates[] = {
            r->shader_dir + "/" + name + ".spv",
            r->shader_dir + "/build/" + name + ".spv",
            "build/" + name + ".spv",
            name + ".spv",
        };
        for (auto &path : candidates) {
            auto spirv = load_spirv(path);
            if (!spirv.empty()) return spirv;
        }
        fprintf(stderr, "HY3D: failed to load shader: %s.spv (searched %s)\n",
                name.c_str(), r->shader_dir.c_str());
        return {};
    };

    auto createPipe = [&](const std::string &name, int nBuf, int pushSize, Pipeline &pipe) -> bool {
        auto spirv = loadSpv(name);
        if (spirv.empty()) return false;
        return r->runner.createComputePipelineWithPushConstants(spirv, makeBindings(nBuf), pushSize, pipe);
    };

    /* Common shaders */
    if (!createPipe("shaders/layernorm_f32",                   4, 12, r->pipe_layernorm)) return false;
    if (!createPipe("shaders/hy3d/gemm_f32_f32",              4, 16, r->pipe_gemm_f32)) return false;
    if (!createPipe("shaders/gelu_f32",                        1, 4,  r->pipe_gelu)) return false;
    if (!createPipe("shaders/add_f32",                         2, 4,  r->pipe_add)) return false;
    if (!createPipe("shaders/matmul_bias_f32",                 4, 16, r->pipe_matmul_bias_f32)) return false;
    if (!createPipe("shaders/patch_embed_f32",                 4, 24, r->pipe_patch_embed)) return false;

    /* HY3D-specific shaders */
    if (!createPipe("shaders/hy3d/layerscale_add_f32",         3, 8,  r->pipe_layerscale_add)) return false;
    if (!createPipe("shaders/hy3d/rms_norm_f32",               2, 20, r->pipe_rms_norm)) return false;
    if (!createPipe("shaders/hy3d/qk_layernorm_f32",           3, 20, r->pipe_qk_layernorm)) return false;
    if (!createPipe("shaders/hy3d/cross_attn_f32",             4, 24, r->pipe_cross_attn)) return false;
    if (!createPipe("shaders/hy3d/fourier_embed_3d_f32",       3, 12, r->pipe_fourier_embed)) return false;
    if (!createPipe("shaders/hy3d/timestep_embed_f32",         1, 8,  r->pipe_timestep_embed)) return false;
    if (!createPipe("shaders/hy3d/euler_step_f32",             2, 8,  r->pipe_euler_step)) return false;
    if (!createPipe("shaders/hy3d/cfg_combine_f32",            3, 8,  r->pipe_cfg_combine)) return false;
    if (!createPipe("shaders/hy3d/split_qkv_interleaved_f32",  4, 12, r->pipe_split_qkv)) return false;
    if (!createPipe("shaders/hy3d/split_kv_interleaved_f32",   3, 12, r->pipe_split_kv)) return false;
    if (!createPipe("shaders/hy3d/broadcast_add_f32",           2, 8,  r->pipe_broadcast_add)) return false;
    if (!createPipe("shaders/hy3d/concat_token_f32",            3, 8,  r->pipe_concat_first)) return false;
    if (!createPipe("shaders/hy3d/strip_first_token_f32",       2, 8,  r->pipe_strip_first)) return false;
    if (!createPipe("shaders/hy3d/concat_f32",                  3, 8,  r->pipe_concat_last_dim)) return false;

    r->pipelines_created = true;
    return true;
}

/* ======================================================================== */
/* Weight loading                                                           */
/* ======================================================================== */

static int loadDinoWeights(vulkan_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "HY3D: cannot open DINOv2: %s\n", path); return -1; }
    if (r->verbose) fprintf(stderr, "HY3D: loading DINOv2 from %s (%d tensors)\n", path, st->n_tensors);

    r->dino_patch_w = stUploadF32(r->runner, st, "main_image_encoder.model.embeddings.patch_embeddings.projection.weight", r->verbose);
    r->dino_patch_b = stUploadF32(r->runner, st, "main_image_encoder.model.embeddings.patch_embeddings.projection.bias", r->verbose);
    r->dino_pos_emb = stUploadF32(r->runner, st, "main_image_encoder.model.embeddings.position_embeddings", r->verbose);
    r->dino_cls_token = stUploadF32(r->runner, st, "main_image_encoder.model.embeddings.cls_token", r->verbose);

    for (int i = 0; i < DINO_LAYERS; i++) {
        char name[256];
        dino_layer_vk *l = &r->dino_layers[i];

        #define DINO_F32(field, suffix) \
            snprintf(name, sizeof(name), "main_image_encoder.model.encoder.layer.%d.%s", i, suffix); \
            l->field = stUploadF32(r->runner, st, name, r->verbose);

        DINO_F32(ln1_w, "norm1.weight");
        DINO_F32(ln1_b, "norm1.bias");
        DINO_F32(q_w,   "attention.attention.query.weight");
        DINO_F32(q_b,   "attention.attention.query.bias");
        DINO_F32(k_w,   "attention.attention.key.weight");
        DINO_F32(k_b,   "attention.attention.key.bias");
        DINO_F32(v_w,   "attention.attention.value.weight");
        DINO_F32(v_b,   "attention.attention.value.bias");
        DINO_F32(out_w, "attention.output.dense.weight");
        DINO_F32(out_b, "attention.output.dense.bias");
        DINO_F32(ls1,   "layer_scale1.lambda1");
        DINO_F32(ln2_w, "norm2.weight");
        DINO_F32(ln2_b, "norm2.bias");
        DINO_F32(fc1_w, "mlp.fc1.weight");
        DINO_F32(fc1_b, "mlp.fc1.bias");
        DINO_F32(fc2_w, "mlp.fc2.weight");
        DINO_F32(fc2_b, "mlp.fc2.bias");
        DINO_F32(ls2,   "layer_scale2.lambda1");
        #undef DINO_F32

        l->has_ls1 = isValid(l->ls1);
        l->has_ls2 = isValid(l->ls2);
    }

    r->dino_final_ln_w = stUploadF32(r->runner, st, "main_image_encoder.model.layernorm.weight", r->verbose);
    r->dino_final_ln_b = stUploadF32(r->runner, st, "main_image_encoder.model.layernorm.bias", r->verbose);
    if (!isValid(r->dino_final_ln_w)) {
        r->dino_final_ln_w = stUploadF32(r->runner, st, "main_image_encoder.model.norm.weight", r->verbose);
        r->dino_final_ln_b = stUploadF32(r->runner, st, "main_image_encoder.model.norm.bias", r->verbose);
    }

    safetensors_close(st);
    r->dino_loaded = true;
    if (r->verbose) fprintf(stderr, "HY3D: DINOv2 weights loaded\n");
    return 0;
}

static int loadDitWeights(vulkan_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "HY3D: cannot open DiT: %s\n", path); return -1; }
    if (r->verbose) fprintf(stderr, "HY3D: loading DiT from %s (%d tensors)\n", path, st->n_tensors);

    r->dit_x_emb_w = stUploadF32(r->runner, st, "x_embedder.weight", r->verbose);
    r->dit_x_emb_b = stUploadF32(r->runner, st, "x_embedder.bias", r->verbose);

    r->dit_t_mlp0_w = stUploadF32(r->runner, st, "t_embedder.mlp.0.weight", r->verbose);
    r->dit_t_mlp0_b = stUploadF32(r->runner, st, "t_embedder.mlp.0.bias", r->verbose);
    r->dit_t_mlp2_w = stUploadF32(r->runner, st, "t_embedder.mlp.2.weight", r->verbose);
    r->dit_t_mlp2_b = stUploadF32(r->runner, st, "t_embedder.mlp.2.bias", r->verbose);

    for (int i = 0; i < DIT_DEPTH; i++) {
        char name[256];
        dit_block_vk *b = &r->dit_blocks[i];

        #define DIT_F32(field, suffix) \
            snprintf(name, sizeof(name), "blocks.%d.%s", i, suffix); \
            b->field = stUploadF32(r->runner, st, name, r->verbose);

        DIT_F32(norm1_w,     "norm1.weight");
        DIT_F32(norm1_b,     "norm1.bias");

        /* Fuse Q/K/V into [3*dim, dim] */
        {
            char nq[256], nk[256], nv[256];
            snprintf(nq, sizeof(nq), "blocks.%d.attn1.to_q.weight", i);
            snprintf(nk, sizeof(nk), "blocks.%d.attn1.to_k.weight", i);
            snprintf(nv, sizeof(nv), "blocks.%d.attn1.to_v.weight", i);
            b->sa_qkv_w = stFuse3F32(r->runner, st, nq, nk, nv, r->verbose);
        }
        DIT_F32(sa_out_w,    "attn1.out_proj.weight");
        DIT_F32(sa_out_b,    "attn1.out_proj.bias");
        DIT_F32(sa_q_norm_w, "attn1.q_norm.weight");
        DIT_F32(sa_k_norm_w, "attn1.k_norm.weight");
        b->has_sa_q_norm = isValid(b->sa_q_norm_w);
        b->has_sa_k_norm = isValid(b->sa_k_norm_w);

        DIT_F32(norm2_w,     "norm2.weight");
        DIT_F32(norm2_b,     "norm2.bias");
        DIT_F32(ca_q_w,      "attn2.to_q.weight");
        /* Fuse K/V */
        {
            char nk[256], nv[256];
            snprintf(nk, sizeof(nk), "blocks.%d.attn2.to_k.weight", i);
            snprintf(nv, sizeof(nv), "blocks.%d.attn2.to_v.weight", i);
            b->ca_kv_w = stFuse2F32(r->runner, st, nk, nv, r->verbose);
        }
        DIT_F32(ca_out_w,    "attn2.out_proj.weight");
        DIT_F32(ca_out_b,    "attn2.out_proj.bias");
        DIT_F32(ca_q_norm_w, "attn2.q_norm.weight");
        DIT_F32(ca_k_norm_w, "attn2.k_norm.weight");
        b->has_ca_q_norm = isValid(b->ca_q_norm_w);
        b->has_ca_k_norm = isValid(b->ca_k_norm_w);

        DIT_F32(norm3_w,     "norm3.weight");
        DIT_F32(norm3_b,     "norm3.bias");
        b->has_norm3 = isValid(b->norm3_w);

        /* Skip connection (blocks 11-20) */
        b->use_skip = (i > DIT_HALF_DEPTH);
        if (b->use_skip) {
            DIT_F32(skip_linear_w, "skip_linear.weight");
            DIT_F32(skip_linear_b, "skip_linear.bias");
            DIT_F32(skip_norm_w,   "skip_norm.weight");
            DIT_F32(skip_norm_b,   "skip_norm.bias");
        }

        /* MoE vs regular MLP */
        b->use_moe = (i >= DIT_MOE_START);
        if (b->use_moe) {
            DIT_F32(moe_gate_w, "moe.gate.weight");
            for (int e = 0; e < DIT_N_EXPERTS; e++) {
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.0.proj.weight", i, e);
                b->moe_expert_fc1_w[e] = stUploadF32(r->runner, st, name, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.0.proj.bias", i, e);
                b->moe_expert_fc1_b[e] = stUploadF32(r->runner, st, name, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.2.weight", i, e);
                b->moe_expert_fc2_w[e] = stUploadF32(r->runner, st, name, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.2.bias", i, e);
                b->moe_expert_fc2_b[e] = stUploadF32(r->runner, st, name, r->verbose);
            }
            DIT_F32(moe_shared_fc1_w, "moe.shared_experts.net.0.proj.weight");
            DIT_F32(moe_shared_fc1_b, "moe.shared_experts.net.0.proj.bias");
            DIT_F32(moe_shared_fc2_w, "moe.shared_experts.net.2.weight");
            DIT_F32(moe_shared_fc2_b, "moe.shared_experts.net.2.bias");
        } else {
            DIT_F32(mlp_fc1_w, "mlp.fc1.weight");
            DIT_F32(mlp_fc1_b, "mlp.fc1.bias");
            DIT_F32(mlp_fc2_w, "mlp.fc2.weight");
            DIT_F32(mlp_fc2_b, "mlp.fc2.bias");
        }
        #undef DIT_F32
    }

    r->dit_final_ln_w = stUploadF32(r->runner, st, "final_layer.norm_final.weight", r->verbose);
    r->dit_final_ln_b = stUploadF32(r->runner, st, "final_layer.norm_final.bias", r->verbose);
    r->dit_final_linear_w = stUploadF32(r->runner, st, "final_layer.linear.weight", r->verbose);
    r->dit_final_linear_b = stUploadF32(r->runner, st, "final_layer.linear.bias", r->verbose);

    safetensors_close(st);
    r->dit_loaded = true;
    if (r->verbose) fprintf(stderr, "HY3D: DiT weights loaded\n");
    return 0;
}

static int loadVaeWeights(vulkan_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "HY3D: cannot open VAE: %s\n", path); return -1; }
    if (r->verbose) fprintf(stderr, "HY3D: loading ShapeVAE from %s (%d tensors)\n", path, st->n_tensors);

    r->vae_post_kl_w = stUploadF32(r->runner, st, "post_kl.weight", r->verbose);
    r->vae_post_kl_b = stUploadF32(r->runner, st, "post_kl.bias", r->verbose);

    for (int i = 0; i < VAE_DEC_LAYERS; i++) {
        char name[256];
        vae_block_vk *b = &r->vae_blocks[i];

        #define VAE_F32(field, suffix) \
            snprintf(name, sizeof(name), "transformer.resblocks.%d.%s", i, suffix); \
            b->field = stUploadF32(r->runner, st, name, r->verbose);

        VAE_F32(ln1_w,        "ln_1.weight");
        VAE_F32(ln1_b,        "ln_1.bias");
        VAE_F32(qkv_w,        "attn.c_qkv.weight");
        VAE_F32(proj_w,       "attn.c_proj.weight");
        VAE_F32(proj_b,       "attn.c_proj.bias");
        VAE_F32(q_norm_w,     "attn.attention.q_norm.weight");
        VAE_F32(q_norm_b,     "attn.attention.q_norm.bias");
        VAE_F32(k_norm_w,     "attn.attention.k_norm.weight");
        VAE_F32(k_norm_b,     "attn.attention.k_norm.bias");
        VAE_F32(ln2_w,        "ln_2.weight");
        VAE_F32(ln2_b,        "ln_2.bias");
        VAE_F32(mlp_fc_w,     "mlp.c_fc.weight");
        VAE_F32(mlp_fc_b,     "mlp.c_fc.bias");
        VAE_F32(mlp_proj_w,   "mlp.c_proj.weight");
        VAE_F32(mlp_proj_b,   "mlp.c_proj.bias");
        #undef VAE_F32

        b->use_qk_norm = isValid(b->q_norm_w);
    }

    /* Geometry decoder */
    vae_geo_decoder_vk *g = &r->vae_geo;
    #define GEO_F32(field, suffix) g->field = stUploadF32(r->runner, st, suffix, r->verbose);

    GEO_F32(query_proj_w, "geo_decoder.query_proj.weight");
    GEO_F32(query_proj_b, "geo_decoder.query_proj.bias");
    GEO_F32(ln1_w,        "geo_decoder.cross_attn_decoder.ln_1.weight");
    GEO_F32(ln1_b,        "geo_decoder.cross_attn_decoder.ln_1.bias");
    GEO_F32(ln2_w,        "geo_decoder.cross_attn_decoder.ln_2.weight");
    GEO_F32(ln2_b,        "geo_decoder.cross_attn_decoder.ln_2.bias");
    GEO_F32(c_q_w,        "geo_decoder.cross_attn_decoder.attn.c_q.weight");
    GEO_F32(c_kv_w,       "geo_decoder.cross_attn_decoder.attn.c_kv.weight");
    GEO_F32(c_proj_w,     "geo_decoder.cross_attn_decoder.attn.c_proj.weight");
    GEO_F32(c_proj_b,     "geo_decoder.cross_attn_decoder.attn.c_proj.bias");
    GEO_F32(q_norm_w,     "geo_decoder.cross_attn_decoder.attn.attention.q_norm.weight");
    GEO_F32(q_norm_b,     "geo_decoder.cross_attn_decoder.attn.attention.q_norm.bias");
    GEO_F32(k_norm_w,     "geo_decoder.cross_attn_decoder.attn.attention.k_norm.weight");
    GEO_F32(k_norm_b,     "geo_decoder.cross_attn_decoder.attn.attention.k_norm.bias");
    GEO_F32(ln3_w,        "geo_decoder.cross_attn_decoder.ln_3.weight");
    GEO_F32(ln3_b,        "geo_decoder.cross_attn_decoder.ln_3.bias");
    GEO_F32(mlp_fc_w,     "geo_decoder.cross_attn_decoder.mlp.c_fc.weight");
    GEO_F32(mlp_fc_b,     "geo_decoder.cross_attn_decoder.mlp.c_fc.bias");
    GEO_F32(mlp_proj_w,   "geo_decoder.cross_attn_decoder.mlp.c_proj.weight");
    GEO_F32(mlp_proj_b,   "geo_decoder.cross_attn_decoder.mlp.c_proj.bias");
    GEO_F32(ln_post_w,    "geo_decoder.ln_post.weight");
    GEO_F32(ln_post_b,    "geo_decoder.ln_post.bias");
    GEO_F32(output_w,     "geo_decoder.output_proj.weight");
    GEO_F32(output_b,     "geo_decoder.output_proj.bias");
    #undef GEO_F32

    g->use_qk_norm = isValid(g->q_norm_w);
    g->has_ln_post = isValid(g->ln_post_w);

    /* Pre-compute Fourier frequencies WITHOUT pi multiplier */
    float freqs[VAE_NUM_FREQS];
    for (int i = 0; i < VAE_NUM_FREQS; i++) {
        freqs[i] = powf(2.0f, (float)i);
    }
    r->vae_fourier_freqs = createGpuBuffer(r->runner, sizeof(freqs));
    uploadToBuffer(r->runner, r->vae_fourier_freqs, freqs, sizeof(freqs));

    safetensors_close(st);
    r->vae_loaded = true;
    if (r->verbose) fprintf(stderr, "HY3D: ShapeVAE weights loaded\n");
    return 0;
}

/* ======================================================================== */
/* Pipeline stages                                                          */
/* ======================================================================== */

/* Stage 1: DINOv2 forward pass */
static void runDinov2(vulkan_hy3d_runner *r, BufInfo &d_image, BufInfo &d_out) {
    const int seq = DINO_SEQ_LEN;
    const int dim = DINO_HIDDEN;
    const int heads = DINO_HEADS;
    const int hd = DINO_HEAD_DIM;
    const int ffn = DINO_FFN;
    const int ps = DINO_PATCH;
    const int gw = DINO_IMG_SIZE / ps;

    ensureScratch(r, 0, (size_t)seq * dim * sizeof(float));
    ensureScratch(r, 1, (size_t)3 * seq * dim * sizeof(float));
    ensureScratch(r, 2, (size_t)seq * dim * sizeof(float));
    ensureScratch(r, 3, (size_t)seq * ffn * sizeof(float));
    ensureScratch(r, 4, (size_t)seq * dim * sizeof(float));

    BufInfo &d_hidden = r->scratch[0];
    BufInfo &d_qkv    = r->scratch[1];
    BufInfo &d_attn   = r->scratch[2];
    BufInfo &d_mlp    = r->scratch[3];
    BufInfo &d_normed = r->scratch[4];

    /* 1. Patch embedding via matmul_bias_f32 (conv2d decomposed as GEMM for simplicity)
     *    OR use patch_embed_f32 shader if available.
     *    We use patch_embed_f32: bindings = {image, kernel_w, bias, output}
     *    push = {n_patches, dim, kernel_size, gw, width, patch_size} */
    {
        int n_patches = gw * gw;
        int kernel_size = ps * ps * 3;
        struct { uint32_t n_patches, dim, kernel_size, gw, width, patch_size; } pc = {
            (uint32_t)n_patches, (uint32_t)dim, (uint32_t)kernel_size, (uint32_t)gw,
            (uint32_t)DINO_IMG_SIZE, (uint32_t)ps
        };
        int total = n_patches * dim;
        dispatchSync(r, r->pipe_patch_embed, {d_image, r->dino_patch_w, r->dino_patch_b, d_hidden},
                     &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
    }

    /* 2. CLS token + position embeddings (do on CPU for simplicity since it's one-time) */
    {
        /* Download hidden, add CLS token at position 0, add position embeddings */
        size_t patch_bytes = (size_t)(gw * gw) * dim * sizeof(float);
        size_t seq_bytes = (size_t)seq * dim * sizeof(float);
        std::vector<float> hidden(seq * dim, 0.0f);

        /* Download patch embeddings (they occupy slots 1..1369) */
        void *ptr = nullptr;
        r->runner.mapBuffer(d_hidden, &ptr);
        memcpy(hidden.data() + dim, ptr, patch_bytes);  /* patches go to [1..1369] */
        r->runner.unmapBuffer(d_hidden);

        /* Download CLS token and add to position 0 */
        std::vector<float> cls(dim);
        downloadFromBuffer(r->runner, r->dino_cls_token, cls.data(), dim * sizeof(float));
        memcpy(hidden.data(), cls.data(), dim * sizeof(float));

        /* Download position embeddings and add */
        std::vector<float> pos(seq * dim);
        downloadFromBuffer(r->runner, r->dino_pos_emb, pos.data(), seq_bytes);
        for (int i = 0; i < seq * dim; i++)
            hidden[i] += pos[i];

        /* Upload back */
        uploadToBuffer(r->runner, d_hidden, hidden.data(), seq_bytes);
    }

    /* 3. Encoder layers */
    for (int li = 0; li < DINO_LAYERS; li++) {
        dino_layer_vk *l = &r->dino_layers[li];

        /* LN1 */
        opLayerNorm(r, d_normed, d_hidden, l->ln1_w, l->ln1_b, seq, dim);

        /* Q, K, V projections (separate GEMMs) */
        /* Split d_qkv into Q, K, V regions by offset:
         * Q: [0, seq*dim), K: [seq*dim, 2*seq*dim), V: [2*seq*dim, 3*seq*dim) */
        /* Since we can't offset within BufInfo easily, we need separate buffers or
         * use a single scratch and download/upload. For simplicity, use separate scratch
         * regions. d_qkv is large enough for Q, K, V individually but we need to bind
         * each as separate buffers. Use d_qkv, d_attn, d_normed as temp Q, K, V. */

        /* Actually, we need separate buffers for Q, K, V. Let's allocate from scratch:
         * d_qkv (scratch[1]) is 3*seq*dim floats, but we can't sub-allocate within Vulkan
         * binding. So let's use scratch[1] as Q, scratch[2] as K, and create temp V. */

        /* Simpler approach: compute Q into d_attn(scratch[2]), K into d_normed(scratch[4]),
         * V into d_qkv(scratch[1]). Then do attention. */
        BufInfo &d_Q = d_attn;    /* scratch[2] */
        BufInfo &d_K = d_normed;  /* scratch[4] - we already used it for LN output, but
                                     we'll recompute. Actually d_normed has LN output we need! */

        /* Better: use d_qkv for Q, d_attn for temp, and a portion approach won't work.
         * Let's just do: Q->d_qkv, then LN again for K, then LN again for V.
         * Or: save normed in d_mlp temporarily. */

        /* Most practical approach for HOST_VISIBLE synchronous execution:
         * Use 5 scratch buffers, each [seq, dim]. */
        /* scratch[0]=hidden, scratch[1] used as Q, scratch[2]=K, scratch[3]=V(temp), scratch[4]=normed */
        /* But scratch[3] is [seq*ffn] = much larger, so also fine as V. */

        /* Let's redefine: use scratch[1] as Q buffer [seq*dim subset of 3*seq*dim] */
        /* Q -> scratch[1], K -> scratch[2], V -> scratch[3] (first seq*dim of ffn buffer) */
        BufInfo &buf_Q = d_qkv;   /* scratch[1], only first seq*dim used */
        BufInfo &buf_K = d_attn;   /* scratch[2] */
        BufInfo &buf_V = d_mlp;    /* scratch[3], first seq*dim of ffn-sized buffer */

        /* We need d_normed as input for all three GEMMs, but we're overwriting scratch[2]/scratch[4].
         * d_normed is scratch[4]. Let's compute GEMM Q first (output to scratch[1]),
         * GEMM K (output to scratch[2]), GEMM V (output to scratch[3]).
         * All read from scratch[4] which we don't overwrite. */

        opGemm(r, buf_Q, l->q_w, d_normed, l->q_b, dim, dim, seq);
        opGemm(r, buf_K, l->k_w, d_normed, l->k_b, dim, dim, seq);
        opGemm(r, buf_V, l->v_w, d_normed, l->v_b, dim, dim, seq);

        /* Self-attention: output to d_normed (scratch[4]) since we're done with input */
        opSelfAttn(r, d_normed, buf_Q, buf_K, buf_V, seq, dim, heads, hd);

        /* Output projection: d_attn(scratch[2]) = out_w @ d_normed + out_b */
        opGemm(r, d_attn, l->out_w, d_normed, l->out_b, dim, dim, seq);

        /* LayerScale 1 + residual */
        if (l->has_ls1)
            opLayerscaleAdd(r, d_hidden, d_attn, l->ls1, seq * dim, dim);
        else
            opAdd(r, d_hidden, d_attn, seq * dim);

        /* LN2 -> MLP -> LayerScale + residual */
        opLayerNorm(r, d_normed, d_hidden, l->ln2_w, l->ln2_b, seq, dim);
        opGemm(r, d_mlp, l->fc1_w, d_normed, l->fc1_b, ffn, dim, seq);
        opGelu(r, d_mlp, seq * ffn);
        opGemm(r, d_attn, l->fc2_w, d_mlp, l->fc2_b, dim, ffn, seq);

        if (l->has_ls2)
            opLayerscaleAdd(r, d_hidden, d_attn, l->ls2, seq * dim, dim);
        else
            opAdd(r, d_hidden, d_attn, seq * dim);
    }

    /* 4. Final LN */
    if (isValid(r->dino_final_ln_w)) {
        opLayerNorm(r, d_out, d_hidden, r->dino_final_ln_w, r->dino_final_ln_b, seq, dim);
    } else {
        bufferCopy(r->runner, d_out, d_hidden, (size_t)seq * dim * sizeof(float));
    }
}

/* MoE forward (CPU gating, GPU expert compute) */
static void runDitMoe(vulkan_hy3d_runner *r, dit_block_vk *blk,
                       BufInfo &d_input, BufInfo &d_output,
                       int N_tok, BufInfo &d_exp_h, BufInfo &d_exp_o) {
    const int H_dim = DIT_HIDDEN;
    const int ffn = DIT_FFN;

    /* Step 1: Gate logits (small GEMM, download to CPU) */
    BufInfo d_gate = createGpuBuffer(r->runner, (size_t)N_tok * DIT_N_EXPERTS * sizeof(float));
    opGemm(r, d_gate, blk->moe_gate_w, d_input, blk->moe_expert_fc1_b[0] /* dummy, has_bias=0 check */,
           DIT_N_EXPERTS, H_dim, N_tok);
    /* Actually gate has no bias. We need a way to pass no-bias. Let's use a tiny dummy buffer. */
    /* The GEMM shader checks has_bias flag in push constants. When bias buffer is VK_NULL_HANDLE,
     * isValid returns false and has_bias will be 0. Let's fix the call: */
    {
        BufInfo no_bias{};
        struct { uint32_t n_out, n_in, n_tok, has_bias; } pc = {
            (uint32_t)DIT_N_EXPERTS, (uint32_t)H_dim, (uint32_t)N_tok, 0u
        };
        uint32_t gx = (uint32_t)((DIT_N_EXPERTS + 63) / 64);
        uint32_t gy = (uint32_t)((N_tok + 15) / 16);
        /* We need a valid bias buffer even if unused. Create a tiny one. */
        BufInfo dummy_bias = createGpuBuffer(r->runner, sizeof(float));
        dispatchSync(r, r->pipe_gemm_f32, {d_gate, blk->moe_gate_w, d_input, dummy_bias},
                     &pc, sizeof(pc), gx, gy);
        r->runner.destroyBuffer(dummy_bias);
    }

    /* Step 2: Download gate logits, softmax + top-2 on CPU */
    std::vector<float> gate_cpu(N_tok * DIT_N_EXPERTS);
    downloadFromBuffer(r->runner, d_gate, gate_cpu.data(), gate_cpu.size() * sizeof(float));
    r->runner.destroyBuffer(d_gate);

    std::vector<float> gate_weights(N_tok * DIT_N_EXPERTS, 0.0f);
    for (int t = 0; t < N_tok; t++) {
        float *row = gate_cpu.data() + t * DIT_N_EXPERTS;
        float mx = row[0];
        for (int e = 1; e < DIT_N_EXPERTS; e++)
            if (row[e] > mx) mx = row[e];
        float sum = 0.0f;
        float softmax_vals[DIT_N_EXPERTS];
        for (int e = 0; e < DIT_N_EXPERTS; e++) {
            softmax_vals[e] = expf(row[e] - mx);
            sum += softmax_vals[e];
        }
        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        for (int e = 0; e < DIT_N_EXPERTS; e++)
            softmax_vals[e] *= inv;

        int top_idx[DIT_MOE_TOP_K];
        float top_val[DIT_MOE_TOP_K];
        for (int k = 0; k < DIT_MOE_TOP_K; k++) {
            int best = -1;
            float best_v = -1e30f;
            for (int e = 0; e < DIT_N_EXPERTS; e++) {
                bool used = false;
                for (int kk = 0; kk < k; kk++)
                    if (top_idx[kk] == e) { used = true; break; }
                if (!used && softmax_vals[e] > best_v) {
                    best_v = softmax_vals[e];
                    best = e;
                }
            }
            top_idx[k] = best;
            top_val[k] = best_v;
        }

        for (int k = 0; k < DIT_MOE_TOP_K; k++)
            gate_weights[t * DIT_N_EXPERTS + top_idx[k]] = top_val[k];
    }

    /* Step 3: Zero accumulator */
    bufferZero(r->runner, d_output, (size_t)N_tok * H_dim * sizeof(float));

    /* Step 4: For each expert, compute output and weighted-add (on CPU) */
    for (int e = 0; e < DIT_N_EXPERTS; e++) {
        bool any_nonzero = false;
        for (int t = 0; t < N_tok; t++)
            if (gate_weights[t * DIT_N_EXPERTS + e] > 0.0f) { any_nonzero = true; break; }
        if (!any_nonzero) continue;

        opGemm(r, d_exp_h, blk->moe_expert_fc1_w[e], d_input, blk->moe_expert_fc1_b[e],
               ffn, H_dim, N_tok);
        opGelu(r, d_exp_h, N_tok * ffn);
        opGemm(r, d_exp_o, blk->moe_expert_fc2_w[e], d_exp_h, blk->moe_expert_fc2_b[e],
               H_dim, ffn, N_tok);

        /* Scale-add on CPU */
        std::vector<float> exp_out(N_tok * H_dim);
        std::vector<float> accum(N_tok * H_dim);
        downloadFromBuffer(r->runner, d_exp_o, exp_out.data(), exp_out.size() * sizeof(float));
        downloadFromBuffer(r->runner, d_output, accum.data(), accum.size() * sizeof(float));

        for (int t = 0; t < N_tok; t++) {
            float w = gate_weights[t * DIT_N_EXPERTS + e];
            if (w == 0.0f) continue;
            for (int j = 0; j < H_dim; j++)
                accum[t * H_dim + j] += w * exp_out[t * H_dim + j];
        }
        uploadToBuffer(r->runner, d_output, accum.data(), accum.size() * sizeof(float));
    }

    /* Step 5: Shared expert */
    opGemm(r, d_exp_h, blk->moe_shared_fc1_w, d_input, blk->moe_shared_fc1_b, ffn, H_dim, N_tok);
    opGelu(r, d_exp_h, N_tok * ffn);
    opGemm(r, d_exp_o, blk->moe_shared_fc2_w, d_exp_h, blk->moe_shared_fc2_b, H_dim, ffn, N_tok);
    opAdd(r, d_output, d_exp_o, N_tok * H_dim);
}

/* Stage 2: DiT single forward pass */
static void runDitForward(vulkan_hy3d_runner *r, BufInfo &d_latents,
                            float timestep, BufInfo &d_context,
                            BufInfo &d_output) {
    const int N = DIT_INPUT_SIZE;
    const int C = DIT_IN_CHANNELS;
    const int H_dim = DIT_HIDDEN;
    const int heads = DIT_HEADS;
    const int hd = DIT_HEAD_DIM;
    const int ffn = DIT_FFN;
    const int ctx_len = DINO_SEQ_LEN;
    const int N1 = N + 1;

    /* Allocate scratch buffers */
    size_t mlp_sz = (size_t)N1 * ffn * sizeof(float);
    size_t qkv_sz = (size_t)3 * N1 * H_dim * sizeof(float);
    size_t moe_scratch_sz = (size_t)N1 * (DIT_N_EXPERTS + ffn + H_dim) * sizeof(float);
    size_t shared1_sz = std::max(qkv_sz, std::max(mlp_sz, moe_scratch_sz));

    ensureScratch(r, 0, (size_t)N1 * H_dim * sizeof(float));         /* hidden */
    ensureScratch(r, 1, shared1_sz);                                   /* qkv/mlp/moe */
    ensureScratch(r, 2, (size_t)N1 * H_dim * sizeof(float));         /* attn_out */
    ensureScratch(r, 3, (size_t)N1 * H_dim * sizeof(float));         /* normed */
    ensureScratch(r, 4, (size_t)H_dim * sizeof(float) + (size_t)ffn * sizeof(float)); /* t_emb + t_mlp */
    ensureScratch(r, 5, (size_t)N1 * H_dim * sizeof(float));         /* cross_Q */
    ensureScratch(r, 6, (size_t)N1 * 2 * H_dim * sizeof(float));    /* cat_buf */
    ensureScratch(r, 7, (size_t)ctx_len * H_dim * sizeof(float) * 2  /* ca_K + ca_V */
                      + (size_t)N1 * H_dim * sizeof(float));          /* split_V */

    BufInfo &d_hidden  = r->scratch[0];
    BufInfo &d_qkv     = r->scratch[1];
    BufInfo &d_attn    = r->scratch[2];
    BufInfo &d_normed  = r->scratch[3];
    BufInfo &d_temb_buf = r->scratch[4];
    BufInfo &d_cross_Q = r->scratch[5];
    BufInfo &d_cat_buf = r->scratch[6];
    BufInfo &d_ca_kv_buf = r->scratch[7];

    /* We also need separate buffers for Q, K, V for self-attention and ca_K, ca_V.
     * For simplicity, allocate them as needed. */
    BufInfo d_Q  = createGpuBuffer(r->runner, (size_t)N1 * H_dim * sizeof(float));
    BufInfo d_K  = createGpuBuffer(r->runner, (size_t)N1 * H_dim * sizeof(float));
    BufInfo d_V  = createGpuBuffer(r->runner, (size_t)N1 * H_dim * sizeof(float));
    BufInfo d_ca_K = createGpuBuffer(r->runner, (size_t)ctx_len * H_dim * sizeof(float));
    BufInfo d_ca_V = createGpuBuffer(r->runner, (size_t)ctx_len * H_dim * sizeof(float));
    BufInfo d_ca_kv_raw = createGpuBuffer(r->runner, (size_t)ctx_len * 2 * H_dim * sizeof(float));

    /* Skip stack on CPU */
    size_t skip_entry_sz = (size_t)N1 * H_dim * sizeof(float);
    std::vector<float> skip_stack_cpu((DIT_HALF_DEPTH + 1) * N1 * H_dim);
    int skip_sp = 0;

    /* 1. Embed latents: [N, C] -> [N, H_dim] into d_normed temp */
    opGemm(r, d_normed, r->dit_x_emb_w, d_latents, r->dit_x_emb_b, H_dim, C, N);

    /* 2. Timestep embedding with GELU */
    BufInfo d_temb = createGpuBuffer(r->runner, (size_t)H_dim * sizeof(float));
    BufInfo d_tmlp = createGpuBuffer(r->runner, (size_t)ffn * sizeof(float));

    opTimestepEmbed(r, d_temb, timestep, H_dim);
    opGemm(r, d_tmlp, r->dit_t_mlp0_w, d_temb, r->dit_t_mlp0_b, ffn, H_dim, 1);
    opGelu(r, d_tmlp, ffn);
    opGemm(r, d_temb, r->dit_t_mlp2_w, d_tmlp, r->dit_t_mlp2_b, H_dim, ffn, 1);

    /* 3. Prepend timestep token: d_hidden = cat([t_emb, x], dim=0) -> [N+1, H_dim] */
    opConcatFirst(r, d_hidden, d_temb, d_normed, N, H_dim);

    /* 4. Transformer blocks */
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        dit_block_vk *blk = &r->dit_blocks[bi];

        /* Skip connection (blocks 11-20) */
        if (blk->use_skip && skip_sp > 0) {
            skip_sp--;
            float *skip_cpu = skip_stack_cpu.data() + (size_t)skip_sp * N1 * H_dim;

            /* Upload skip value to d_attn (temp) */
            uploadToBuffer(r->runner, d_attn, skip_cpu, skip_entry_sz);

            /* cat = concat([skip_value, x], dim=-1) -> [N1, 2*H_dim] */
            opConcatLastDim(r, d_cat_buf, d_attn, d_hidden, N1, H_dim);

            /* x = skip_linear(cat) -> [N1, H_dim] */
            opGemm(r, d_hidden, blk->skip_linear_w, d_cat_buf, blk->skip_linear_b, H_dim, 2 * H_dim, N1);

            /* x = skip_norm(x) */
            opLayerNorm(r, d_hidden, d_hidden, blk->skip_norm_w, blk->skip_norm_b, N1, H_dim);
        }

        /* Save hidden for skip connection (blocks 0..DIT_HALF_DEPTH) */
        if (bi <= DIT_HALF_DEPTH) {
            float *skip_cpu = skip_stack_cpu.data() + (size_t)skip_sp * N1 * H_dim;
            downloadFromBuffer(r->runner, d_hidden, skip_cpu, skip_entry_sz);
            skip_sp++;
        }

        /* === Self-attention === */
        opLayerNorm(r, d_normed, d_hidden, blk->norm1_w, blk->norm1_b, N1, H_dim);

        /* Fused QKV GEMM: [N1, dim] @ [3*dim, dim]^T -> [N1, 3*dim] */
        {
            BufInfo no_bias{};
            struct { uint32_t n_out, n_in, n_tok, has_bias; } pc = {
                (uint32_t)(3 * H_dim), (uint32_t)H_dim, (uint32_t)N1, 0u
            };
            uint32_t gx = (uint32_t)((3 * H_dim + 63) / 64);
            uint32_t gy = (uint32_t)((N1 + 15) / 16);
            BufInfo dummy_bias = createGpuBuffer(r->runner, sizeof(float));
            dispatchSync(r, r->pipe_gemm_f32, {d_qkv, blk->sa_qkv_w, d_normed, dummy_bias},
                         &pc, sizeof(pc), gx, gy);
            r->runner.destroyBuffer(dummy_bias);
        }

        /* Split interleaved QKV */
        opSplitQkv(r, d_Q, d_K, d_V, d_qkv, N1, heads, hd);

        /* QK RMSNorm */
        if (blk->has_sa_q_norm)
            opRmsNorm(r, d_Q, blk->sa_q_norm_w, N1, heads, hd, H_dim);
        if (blk->has_sa_k_norm)
            opRmsNorm(r, d_K, blk->sa_k_norm_w, N1, heads, hd, H_dim);

        /* Self-attention */
        opSelfAttn(r, d_attn, d_Q, d_K, d_V, N1, H_dim, heads, hd);

        /* Output projection + residual */
        opGemm(r, d_normed, blk->sa_out_w, d_attn, blk->sa_out_b, H_dim, H_dim, N1);
        opAdd(r, d_hidden, d_normed, N1 * H_dim);

        /* === Cross-attention === */
        opLayerNorm(r, d_normed, d_hidden, blk->norm2_w, blk->norm2_b, N1, H_dim);

        /* Q from hidden */
        {
            BufInfo no_bias{};
            struct { uint32_t n_out, n_in, n_tok, has_bias; } pc = {
                (uint32_t)H_dim, (uint32_t)H_dim, (uint32_t)N1, 0u
            };
            uint32_t gx = (uint32_t)((H_dim + 63) / 64);
            uint32_t gy = (uint32_t)((N1 + 15) / 16);
            BufInfo dummy_bias = createGpuBuffer(r->runner, sizeof(float));
            dispatchSync(r, r->pipe_gemm_f32, {d_cross_Q, blk->ca_q_w, d_normed, dummy_bias},
                         &pc, sizeof(pc), gx, gy);
            r->runner.destroyBuffer(dummy_bias);
        }
        if (blk->has_ca_q_norm)
            opRmsNorm(r, d_cross_Q, blk->ca_q_norm_w, N1, heads, hd, H_dim);

        /* KV from context: fused [2*dim, ctx_dim] */
        {
            BufInfo no_bias{};
            struct { uint32_t n_out, n_in, n_tok, has_bias; } pc = {
                (uint32_t)(2 * H_dim), (uint32_t)DIT_CONTEXT_DIM, (uint32_t)ctx_len, 0u
            };
            uint32_t gx = (uint32_t)((2 * H_dim + 63) / 64);
            uint32_t gy = (uint32_t)((ctx_len + 15) / 16);
            BufInfo dummy_bias = createGpuBuffer(r->runner, sizeof(float));
            dispatchSync(r, r->pipe_gemm_f32, {d_ca_kv_raw, blk->ca_kv_w, d_context, dummy_bias},
                         &pc, sizeof(pc), gx, gy);
            r->runner.destroyBuffer(dummy_bias);
        }
        opSplitKv(r, d_ca_K, d_ca_V, d_ca_kv_raw, ctx_len, heads, hd);

        if (blk->has_ca_k_norm)
            opRmsNorm(r, d_ca_K, blk->ca_k_norm_w, ctx_len, heads, hd, H_dim);

        /* Cross-attention */
        opCrossAttn(r, d_attn, d_cross_Q, d_ca_K, d_ca_V, N1, ctx_len, H_dim, heads, hd);

        /* Output projection + residual */
        opGemm(r, d_normed, blk->ca_out_w, d_attn, blk->ca_out_b, H_dim, H_dim, N1);
        opAdd(r, d_hidden, d_normed, N1 * H_dim);

        /* === MLP or MoE === */
        if (blk->has_norm3) {
            opLayerNorm(r, d_normed, d_hidden, blk->norm3_w, blk->norm3_b, N1, H_dim);
        } else {
            bufferCopy(r->runner, d_normed, d_hidden, (size_t)N1 * H_dim * sizeof(float));
        }

        if (blk->use_moe) {
            /* MoE output goes to d_qkv (large scratch), then add to hidden */
            BufInfo d_moe_out = createGpuBuffer(r->runner, (size_t)N1 * H_dim * sizeof(float));
            BufInfo d_exp_h = createGpuBuffer(r->runner, (size_t)N1 * ffn * sizeof(float));
            BufInfo d_exp_o = createGpuBuffer(r->runner, (size_t)N1 * H_dim * sizeof(float));
            runDitMoe(r, blk, d_normed, d_moe_out, N1, d_exp_h, d_exp_o);
            opAdd(r, d_hidden, d_moe_out, N1 * H_dim);
            r->runner.destroyBuffer(d_moe_out);
            r->runner.destroyBuffer(d_exp_h);
            r->runner.destroyBuffer(d_exp_o);
        } else {
            /* Regular MLP: fc1 -> GELU -> fc2 */
            opGemm(r, d_qkv, blk->mlp_fc1_w, d_normed, blk->mlp_fc1_b, ffn, H_dim, N1);
            opGelu(r, d_qkv, N1 * ffn);
            opGemm(r, d_normed, blk->mlp_fc2_w, d_qkv, blk->mlp_fc2_b, H_dim, ffn, N1);
            opAdd(r, d_hidden, d_normed, N1 * H_dim);
        }

        if (r->verbose)
            fprintf(stderr, "  block %2d done%s%s\n", bi,
                    blk->use_moe ? " [MoE]" : "", blk->use_skip ? " [skip]" : "");
    }

    /* 5. Final layer: strip timestep token, LN, Linear */
    opStripFirst(r, d_normed, d_hidden, N1, H_dim);

    /* Final LN */
    opLayerNorm(r, d_attn, d_normed, r->dit_final_ln_w, r->dit_final_ln_b, N, H_dim);

    /* Final linear: [N, H_dim] -> [N, C] */
    opGemm(r, d_output, r->dit_final_linear_w, d_attn, r->dit_final_linear_b, C, H_dim, N);

    /* Cleanup temp buffers */
    r->runner.destroyBuffer(d_Q);
    r->runner.destroyBuffer(d_K);
    r->runner.destroyBuffer(d_V);
    r->runner.destroyBuffer(d_ca_K);
    r->runner.destroyBuffer(d_ca_V);
    r->runner.destroyBuffer(d_ca_kv_raw);
    r->runner.destroyBuffer(d_temb);
    r->runner.destroyBuffer(d_tmlp);
}

/* Stage 3: ShapeVAE single transformer block */
static void runVaeBlock(vulkan_hy3d_runner *r, vae_block_vk *b,
                          BufInfo &d_in, BufInfo &d_out,
                          BufInfo &d_ln1, BufInfo &d_qkv, BufInfo &d_Q, BufInfo &d_K, BufInfo &d_V,
                          BufInfo &d_aout, BufInfo &d_proj, BufInfo &d_res1,
                          BufInfo &d_ln2, BufInfo &d_mlph, BufInfo &d_mlpo) {
    const int N = VAE_NUM_LATENTS;
    const int W = VAE_WIDTH;
    const int H = VAE_HEADS;
    const int HD = VAE_HEAD_DIM;
    const int MLP = 4 * W;

    /* LN1 */
    opLayerNorm(r, d_ln1, d_in, b->ln1_w, b->ln1_b, N, W);

    /* Fused QKV GEMM (no bias) */
    {
        BufInfo dummy_bias = createGpuBuffer(r->runner, sizeof(float));
        struct { uint32_t n_out, n_in, n_tok, has_bias; } pc = {
            (uint32_t)(3 * W), (uint32_t)W, (uint32_t)N, 0u
        };
        uint32_t gx = (uint32_t)((3 * W + 63) / 64);
        uint32_t gy = (uint32_t)((N + 15) / 16);
        dispatchSync(r, r->pipe_gemm_f32, {d_qkv, b->qkv_w, d_ln1, dummy_bias},
                     &pc, sizeof(pc), gx, gy);
        r->runner.destroyBuffer(dummy_bias);
    }

    /* Split interleaved QKV */
    opSplitQkv(r, d_Q, d_K, d_V, d_qkv, N, H, HD);

    /* QK normalization */
    if (b->use_qk_norm) {
        opQkLayerNorm(r, d_Q, b->q_norm_w, b->q_norm_b, N, H, HD, W);
        opQkLayerNorm(r, d_K, b->k_norm_w, b->k_norm_b, N, H, HD, W);
    }

    /* Self-attention */
    opSelfAttn(r, d_aout, d_Q, d_K, d_V, N, W, H, HD);

    /* Output projection */
    opGemm(r, d_proj, b->proj_w, d_aout, b->proj_b, W, W, N);

    /* Residual 1: res1 = input + proj */
    bufferCopy(r->runner, d_res1, d_in, (size_t)N * W * sizeof(float));
    opAdd(r, d_res1, d_proj, N * W);

    /* LN2 -> MLP -> Residual 2 */
    opLayerNorm(r, d_ln2, d_res1, b->ln2_w, b->ln2_b, N, W);
    opGemm(r, d_mlph, b->mlp_fc_w, d_ln2, b->mlp_fc_b, MLP, W, N);
    opGelu(r, d_mlph, N * MLP);
    opGemm(r, d_mlpo, b->mlp_proj_w, d_mlph, b->mlp_proj_b, W, MLP, N);

    /* Output = res1 + mlp_out */
    bufferCopy(r->runner, d_out, d_res1, (size_t)N * W * sizeof(float));
    opAdd(r, d_out, d_mlpo, N * W);
}

/* Stage 3: ShapeVAE decode + SDF query */
static void runShapeVae(vulkan_hy3d_runner *r, BufInfo &d_latents,
                          int grid_res, float *sdf_out) {
    const int N = VAE_NUM_LATENTS;
    const int E = VAE_EMBED_DIM;
    const int W = VAE_WIDTH;
    const int MLP = 4 * W;

    /* Allocate decoder buffers */
    BufInfo d_dec_a = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_dec_b = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));

    /* VAE block scratch buffers */
    BufInfo d_ln1   = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_qkv   = createGpuBuffer(r->runner, (size_t)N * 3 * W * sizeof(float));
    BufInfo d_Q     = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_K     = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_V     = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_aout  = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_proj  = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_res1  = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_ln2   = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_mlph  = createGpuBuffer(r->runner, (size_t)N * MLP * sizeof(float));
    BufInfo d_mlpo  = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));

    /* Post-KL projection: [N, E] -> [N, W] */
    opGemm(r, d_dec_a, r->vae_post_kl_w, d_latents, r->vae_post_kl_b, W, E, N);

    /* Run transformer blocks */
    BufInfo *d_cur = &d_dec_a;
    BufInfo *d_next = &d_dec_b;
    for (int i = 0; i < VAE_DEC_LAYERS; i++) {
        runVaeBlock(r, &r->vae_blocks[i], *d_cur, *d_next,
                    d_ln1, d_qkv, d_Q, d_K, d_V, d_aout, d_proj, d_res1,
                    d_ln2, d_mlph, d_mlpo);
        std::swap(d_cur, d_next);
        if (r->verbose)
            fprintf(stderr, "  VAE block %d done\n", i);
    }

    /* Query SDF at grid points in batches */
    int total_points = grid_res * grid_res * grid_res;
    int batch_size = 8192;
    float bounds[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    BufInfo d_coords    = createGpuBuffer(r->runner, (size_t)batch_size * 3 * sizeof(float));
    BufInfo d_fourier   = createGpuBuffer(r->runner, (size_t)batch_size * VAE_FOURIER_DIM * sizeof(float));
    BufInfo d_query_proj = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));
    BufInfo d_sdf_batch = createGpuBuffer(r->runner, (size_t)batch_size * sizeof(float));

    /* Geo decoder scratch */
    BufInfo d_g_ln1  = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));
    BufInfo d_g_ln2  = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_g_Q    = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));
    BufInfo d_g_KV   = createGpuBuffer(r->runner, (size_t)N * 2 * W * sizeof(float));
    BufInfo d_g_K    = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_g_V    = createGpuBuffer(r->runner, (size_t)N * W * sizeof(float));
    BufInfo d_g_aout = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));
    BufInfo d_g_proj = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));
    BufInfo d_g_res  = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));
    BufInfo d_g_ln3  = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));
    BufInfo d_g_mlph = createGpuBuffer(r->runner, (size_t)batch_size * 4 * W * sizeof(float));
    BufInfo d_g_mlpo = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));
    BufInfo d_g_post = createGpuBuffer(r->runner, (size_t)batch_size * W * sizeof(float));

    float dx = (bounds[3] - bounds[0]) / (float)(grid_res - 1);
    float dy = (bounds[4] - bounds[1]) / (float)(grid_res - 1);
    float dz = (bounds[5] - bounds[2]) / (float)(grid_res - 1);

    vae_geo_decoder_vk *g = &r->vae_geo;

    for (int start = 0; start < total_points; start += batch_size) {
        int count = std::min(batch_size, total_points - start);

        /* Generate 3D coordinates on CPU and upload */
        std::vector<float> coords(count * 3);
        for (int i = 0; i < count; i++) {
            int idx = start + i;
            int iz = idx % grid_res;
            int iy = (idx / grid_res) % grid_res;
            int ix = idx / (grid_res * grid_res);
            coords[i * 3 + 0] = bounds[0] + ix * dx;
            coords[i * 3 + 1] = bounds[1] + iy * dy;
            coords[i * 3 + 2] = bounds[2] + iz * dz;
        }
        uploadToBuffer(r->runner, d_coords, coords.data(), count * 3 * sizeof(float));

        /* Fourier embedding */
        opFourierEmbed(r, d_fourier, d_coords, r->vae_fourier_freqs,
                       count, VAE_NUM_FREQS, VAE_FOURIER_DIM);

        /* Query projection */
        opGemm(r, d_query_proj, g->query_proj_w, d_fourier, g->query_proj_b, W, VAE_FOURIER_DIM, count);

        /* Cross-attention geo decoder */
        opLayerNorm(r, d_g_ln1, d_query_proj, g->ln1_w, g->ln1_b, count, W);
        opLayerNorm(r, d_g_ln2, *d_cur, g->ln2_w, g->ln2_b, N, W);

        /* Q from queries, KV from latents */
        {
            BufInfo dummy_bias = createGpuBuffer(r->runner, sizeof(float));
            struct { uint32_t n_out, n_in, n_tok, has_bias; } pc;

            pc = {(uint32_t)W, (uint32_t)W, (uint32_t)count, 0u};
            dispatchSync(r, r->pipe_gemm_f32, {d_g_Q, g->c_q_w, d_g_ln1, dummy_bias},
                         &pc, sizeof(pc), (uint32_t)((W + 63) / 64), (uint32_t)((count + 15) / 16));

            pc = {(uint32_t)(2 * W), (uint32_t)W, (uint32_t)N, 0u};
            dispatchSync(r, r->pipe_gemm_f32, {d_g_KV, g->c_kv_w, d_g_ln2, dummy_bias},
                         &pc, sizeof(pc), (uint32_t)((2 * W + 63) / 64), (uint32_t)((N + 15) / 16));

            r->runner.destroyBuffer(dummy_bias);
        }

        /* Split KV */
        opSplitKv(r, d_g_K, d_g_V, d_g_KV, N, VAE_HEADS, VAE_HEAD_DIM);

        /* QK norm */
        if (g->use_qk_norm) {
            opQkLayerNorm(r, d_g_Q, g->q_norm_w, g->q_norm_b, count, VAE_HEADS, VAE_HEAD_DIM, W);
            opQkLayerNorm(r, d_g_K, g->k_norm_w, g->k_norm_b, N, VAE_HEADS, VAE_HEAD_DIM, W);
        }

        /* Cross-attention */
        opCrossAttn(r, d_g_aout, d_g_Q, d_g_K, d_g_V, count, N, W, VAE_HEADS, VAE_HEAD_DIM);

        /* Output projection + residual */
        opGemm(r, d_g_proj, g->c_proj_w, d_g_aout, g->c_proj_b, W, W, count);
        bufferCopy(r->runner, d_g_res, d_query_proj, (size_t)count * W * sizeof(float));
        opAdd(r, d_g_res, d_g_proj, count * W);

        /* MLP block */
        opLayerNorm(r, d_g_ln3, d_g_res, g->ln3_w, g->ln3_b, count, W);
        opGemm(r, d_g_mlph, g->mlp_fc_w, d_g_ln3, g->mlp_fc_b, 4 * W, W, count);
        opGelu(r, d_g_mlph, count * 4 * W);
        opGemm(r, d_g_mlpo, g->mlp_proj_w, d_g_mlph, g->mlp_proj_b, W, 4 * W, count);
        bufferCopy(r->runner, d_g_post, d_g_res, (size_t)count * W * sizeof(float));
        opAdd(r, d_g_post, d_g_mlpo, count * W);

        /* Post LN */
        if (g->has_ln_post) {
            opLayerNorm(r, d_g_ln1, d_g_post, g->ln_post_w, g->ln_post_b, count, W);
        } else {
            bufferCopy(r->runner, d_g_ln1, d_g_post, (size_t)count * W * sizeof(float));
        }

        /* Final output projection: [count, W] -> [count, 1] */
        opGemm(r, d_sdf_batch, g->output_w, d_g_ln1, g->output_b, 1, W, count);

        /* Download SDF values */
        downloadFromBuffer(r->runner, d_sdf_batch, sdf_out + start, count * sizeof(float));
    }

    /* Cleanup */
    r->runner.destroyBuffer(d_dec_a);
    r->runner.destroyBuffer(d_dec_b);
    r->runner.destroyBuffer(d_ln1);
    r->runner.destroyBuffer(d_qkv);
    r->runner.destroyBuffer(d_Q);
    r->runner.destroyBuffer(d_K);
    r->runner.destroyBuffer(d_V);
    r->runner.destroyBuffer(d_aout);
    r->runner.destroyBuffer(d_proj);
    r->runner.destroyBuffer(d_res1);
    r->runner.destroyBuffer(d_ln2);
    r->runner.destroyBuffer(d_mlph);
    r->runner.destroyBuffer(d_mlpo);
    r->runner.destroyBuffer(d_coords);
    r->runner.destroyBuffer(d_fourier);
    r->runner.destroyBuffer(d_query_proj);
    r->runner.destroyBuffer(d_sdf_batch);
    r->runner.destroyBuffer(d_g_ln1);
    r->runner.destroyBuffer(d_g_ln2);
    r->runner.destroyBuffer(d_g_Q);
    r->runner.destroyBuffer(d_g_KV);
    r->runner.destroyBuffer(d_g_K);
    r->runner.destroyBuffer(d_g_V);
    r->runner.destroyBuffer(d_g_aout);
    r->runner.destroyBuffer(d_g_proj);
    r->runner.destroyBuffer(d_g_res);
    r->runner.destroyBuffer(d_g_ln3);
    r->runner.destroyBuffer(d_g_mlph);
    r->runner.destroyBuffer(d_g_mlpo);
    r->runner.destroyBuffer(d_g_post);
}

/* ======================================================================== */
/* Random noise generation (CPU, Box-Muller)                                */
/* ======================================================================== */

static void generateRandn(float *buf, int n, uint32_t seed) {
    uint64_t s0 = seed ? seed : (uint64_t)time(NULL);
    uint64_t s1 = s0 ^ 0x6c62272e07bb0142ULL;

    for (int i = 0; i < n; i += 2) {
        uint64_t x = s0, y = s1; s0 = y;
        x ^= x << 23; s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r1 = s1 + y;

        x = s0; y = s1; s0 = y;
        x ^= x << 23; s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        uint64_t r2 = s1 + y;

        double u1 = ((r1 >> 11) + 0.5) / 2097152.0;
        double u2 = ((r2 >> 11) + 0.5) / 2097152.0;
        double rr = sqrt(-2.0 * log(u1));
        double theta = 2.0 * 3.141592653589793 * u2;
        buf[i] = (float)(rr * cos(theta));
        if (i + 1 < n) buf[i + 1] = (float)(rr * sin(theta));
    }
}

/* Simple CPU resize + normalize for DINOv2 input preprocessing */
static void resizeNormalize(const uint8_t *rgb, int w, int h,
                              float *out, int dw, int dh) {
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std_inv[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

    for (int dy = 0; dy < dh; dy++) {
        for (int dx = 0; dx < dw; dx++) {
            float sx = (float)dx * w / dw;
            float sy = (float)dy * h / dh;
            int ix = std::min((int)sx, w - 1);
            int iy = std::min((int)sy, h - 1);

            for (int c = 0; c < 3; c++) {
                float v = (float)rgb[(iy * w + ix) * 3 + c] / 255.0f;
                out[c * dh * dw + dy * dw + dx] = (v - mean[c]) * std_inv[c];
            }
        }
    }
}

/* ======================================================================== */
/* Public API                                                               */
/* ======================================================================== */

extern "C" {

vulkan_hy3d_runner *vulkan_hy3d_init(int device_id, int verbose) {
    if (!vl_cpp::vulkan::InitializeVulkan()) {
        fprintf(stderr, "HY3D: failed to initialize Vulkan\n");
        return nullptr;
    }

    vulkan_hy3d_runner *r = new vulkan_hy3d_runner();
    memset(r->scratch_size, 0, sizeof(r->scratch_size));
    for (int i = 0; i < 8; i++) r->scratch[i] = {};
    r->verbose = verbose;

    if (!r->runner.initialize(false)) {
        fprintf(stderr, "HY3D: failed to init Vulkan runner: %s\n", r->runner.getLastError().c_str());
        delete r;
        return nullptr;
    }
    if ((uint32_t)device_id >= r->runner.getDeviceCount()) {
        fprintf(stderr, "HY3D: invalid device ID %d\n", device_id);
        delete r;
        return nullptr;
    }
    if (!r->runner.selectDevice(device_id)) {
        fprintf(stderr, "HY3D: failed to select device: %s\n", r->runner.getLastError().c_str());
        delete r;
        return nullptr;
    }

    if (verbose)
        fprintf(stderr, "HY3D: using Vulkan device: %s\n", r->runner.getDeviceName(device_id).c_str());

    /* Determine shader directory relative to executable (heuristic) */
    r->shader_dir = ".";

    /* Create dynamic descriptor pool */
    if (!r->runner.createDynamicDescriptorPool(16384)) {
        fprintf(stderr, "HY3D: failed to create descriptor pool\n");
        delete r;
        return nullptr;
    }

    /* Create compute pipelines */
    if (!createPipelines(r)) {
        fprintf(stderr, "HY3D: failed to create pipelines\n");
        delete r;
        return nullptr;
    }

    r->initialized = true;
    return r;
}

int vulkan_hy3d_load_weights(vulkan_hy3d_runner *r,
                              const char *cond_path,
                              const char *model_path,
                              const char *vae_path) {
    if (!r) return -1;

    if (cond_path && loadDinoWeights(r, cond_path) != 0) return -1;
    if (model_path && loadDitWeights(r, model_path) != 0) return -1;
    if (vae_path && loadVaeWeights(r, vae_path) != 0) return -1;

    return 0;
}

hy3d_mesh vulkan_hy3d_predict(vulkan_hy3d_runner *r,
                               const uint8_t *rgb, int w, int h,
                               int n_steps, float guidance_scale,
                               int grid_res, uint32_t seed) {
    hy3d_mesh result = {};
    if (!r || !r->dino_loaded || !r->dit_loaded || !r->vae_loaded) {
        fprintf(stderr, "HY3D: runner not fully initialized\n");
        return result;
    }

    if (n_steps <= 0) n_steps = 30;
    if (guidance_scale <= 0.0f) guidance_scale = 7.5f;
    if (grid_res <= 0) grid_res = 256;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* ---- Stage 1: DINOv2 image encoding ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 1 - DINOv2 encoding...\n");

    /* Preprocess on CPU: resize to 518x518, ImageNet-normalize, CHW format */
    size_t img_floats = 3 * DINO_IMG_SIZE * DINO_IMG_SIZE;
    std::vector<float> image_f32(img_floats);
    resizeNormalize(rgb, w, h, image_f32.data(), DINO_IMG_SIZE, DINO_IMG_SIZE);

    BufInfo d_image = createGpuBuffer(r->runner, img_floats * sizeof(float));
    uploadToBuffer(r->runner, d_image, image_f32.data(), img_floats * sizeof(float));

    BufInfo d_dino_out = createGpuBuffer(r->runner, (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    runDinov2(r, d_image, d_dino_out);
    r->runner.destroyBuffer(d_image);

    /* ---- Stage 2: DiT diffusion with flow matching ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 2 - DiT diffusion (%d steps)...\n", n_steps);

    int latent_size = DIT_INPUT_SIZE * DIT_IN_CHANNELS;
    std::vector<float> noise_cpu(latent_size);
    generateRandn(noise_cpu.data(), latent_size, seed);

    BufInfo d_latents = createGpuBuffer(r->runner, (size_t)latent_size * sizeof(float));
    uploadToBuffer(r->runner, d_latents, noise_cpu.data(), (size_t)latent_size * sizeof(float));

    BufInfo d_pred_cond = createGpuBuffer(r->runner, (size_t)latent_size * sizeof(float));
    BufInfo d_pred_uncond = createGpuBuffer(r->runner, (size_t)latent_size * sizeof(float));
    BufInfo d_pred_combined = createGpuBuffer(r->runner, (size_t)latent_size * sizeof(float));

    /* Zero context for unconditional pass */
    BufInfo d_uncond_ctx = createGpuBuffer(r->runner, (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    bufferZero(r->runner, d_uncond_ctx, (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));

    for (int step = 0; step < n_steps; step++) {
        float t_current = 1.0f - (float)step / (float)n_steps;
        float t_next = 1.0f - (float)(step + 1) / (float)n_steps;
        float dt = t_current - t_next;

        if (r->verbose && (step % 5 == 0 || step == n_steps - 1))
            fprintf(stderr, "  step %d/%d (t=%.3f)\n", step + 1, n_steps, t_current);

        /* Conditional pass */
        runDitForward(r, d_latents, t_current, d_dino_out, d_pred_cond);

        /* Unconditional pass */
        runDitForward(r, d_latents, t_current, d_uncond_ctx, d_pred_uncond);

        /* CFG combination */
        opCfgCombine(r, d_pred_combined, d_pred_cond, d_pred_uncond,
                     guidance_scale, latent_size);

        /* Euler step */
        opEulerStep(r, d_latents, d_pred_combined, dt, latent_size);
    }

    r->runner.destroyBuffer(d_pred_cond);
    r->runner.destroyBuffer(d_pred_uncond);
    r->runner.destroyBuffer(d_pred_combined);
    r->runner.destroyBuffer(d_uncond_ctx);
    r->runner.destroyBuffer(d_dino_out);

    /* ---- Stage 3: ShapeVAE decode + SDF query ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 3 - ShapeVAE decode (grid %d^3)...\n", grid_res);

    int total_pts = grid_res * grid_res * grid_res;
    std::vector<float> sdf_grid(total_pts);
    runShapeVae(r, d_latents, grid_res, sdf_grid.data());
    r->runner.destroyBuffer(d_latents);

    /* ---- Stage 4: Marching cubes ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 4 - Marching cubes...\n");

    mc_mesh mc = mc_marching_cubes(sdf_grid.data(), grid_res, grid_res, grid_res, 0.0f, NULL);

    result.vertices = mc.vertices;
    result.triangles = mc.triangles;
    result.n_verts = mc.n_verts;
    result.n_tris = mc.n_tris;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose)
        fprintf(stderr, "HY3D: done in %.2fs (%d verts, %d tris)\n",
                elapsed, result.n_verts, result.n_tris);

    return result;
}

void vulkan_hy3d_free(vulkan_hy3d_runner *r) {
    if (!r) return;

    /* Free scratch buffers */
    for (int i = 0; i < 8; i++) {
        if (r->scratch[i].buffer != VK_NULL_HANDLE)
            r->runner.destroyBuffer(r->scratch[i]);
    }

    /* Note: Weight buffers and pipelines are cleaned up by the VulkanRunner destructor
     * and vulkan_hy3d_runner destructor. For a full cleanup we would iterate all buffers.
     * The Vulkan device destruction reclaims all GPU memory. */

    r->runner.destroyDynamicDescriptorPool();

    if (r->pipelines_created) {
        r->runner.destroyComputePipeline(r->pipe_layernorm);
        r->runner.destroyComputePipeline(r->pipe_gemm_f32);
        r->runner.destroyComputePipeline(r->pipe_gelu);
        r->runner.destroyComputePipeline(r->pipe_add);
        r->runner.destroyComputePipeline(r->pipe_layerscale_add);
        r->runner.destroyComputePipeline(r->pipe_rms_norm);
        r->runner.destroyComputePipeline(r->pipe_qk_layernorm);
        r->runner.destroyComputePipeline(r->pipe_cross_attn);
        r->runner.destroyComputePipeline(r->pipe_fourier_embed);
        r->runner.destroyComputePipeline(r->pipe_timestep_embed);
        r->runner.destroyComputePipeline(r->pipe_euler_step);
        r->runner.destroyComputePipeline(r->pipe_cfg_combine);
        r->runner.destroyComputePipeline(r->pipe_split_qkv);
        r->runner.destroyComputePipeline(r->pipe_split_kv);
        r->runner.destroyComputePipeline(r->pipe_broadcast_add);
        r->runner.destroyComputePipeline(r->pipe_concat_first);
        r->runner.destroyComputePipeline(r->pipe_strip_first);
        r->runner.destroyComputePipeline(r->pipe_concat_last_dim);
        r->runner.destroyComputePipeline(r->pipe_patch_embed);
        r->runner.destroyComputePipeline(r->pipe_matmul_bias_f32);
    }

    if (r->initialized) {
        r->runner.cleanup();
    }

    delete r;
}

/* ---- Per-stage verification API ---- */

int vulkan_hy3d_run_dinov2(vulkan_hy3d_runner *r,
                            const float *image_f32,
                            float *output) {
    if (!r || !r->dino_loaded) return -1;

    size_t img_bytes = (size_t)3 * DINO_IMG_SIZE * DINO_IMG_SIZE * sizeof(float);
    size_t out_bytes = (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float);

    BufInfo d_image = createGpuBuffer(r->runner, img_bytes);
    BufInfo d_out = createGpuBuffer(r->runner, out_bytes);

    uploadToBuffer(r->runner, d_image, image_f32, img_bytes);
    runDinov2(r, d_image, d_out);
    downloadFromBuffer(r->runner, d_out, output, out_bytes);

    r->runner.destroyBuffer(d_image);
    r->runner.destroyBuffer(d_out);
    return 0;
}

int vulkan_hy3d_run_vae(vulkan_hy3d_runner *r,
                         const float *latents,
                         int grid_res,
                         float *sdf_out) {
    if (!r || !r->vae_loaded) return -1;

    size_t lat_bytes = (size_t)VAE_NUM_LATENTS * VAE_EMBED_DIM * sizeof(float);
    BufInfo d_latents = createGpuBuffer(r->runner, lat_bytes);
    uploadToBuffer(r->runner, d_latents, latents, lat_bytes);

    runShapeVae(r, d_latents, grid_res, sdf_out);

    r->runner.destroyBuffer(d_latents);
    return 0;
}

int vulkan_hy3d_run_dit(vulkan_hy3d_runner *r,
                         const float *latents,
                         float timestep,
                         const float *context,
                         float *output) {
    if (!r || !r->dit_loaded) return -1;

    size_t lat_bytes = (size_t)DIT_INPUT_SIZE * DIT_IN_CHANNELS * sizeof(float);
    size_t ctx_bytes = (size_t)DINO_SEQ_LEN * DIT_CONTEXT_DIM * sizeof(float);
    size_t out_bytes = lat_bytes;

    BufInfo d_latents = createGpuBuffer(r->runner, lat_bytes);
    BufInfo d_context = createGpuBuffer(r->runner, ctx_bytes);
    BufInfo d_output  = createGpuBuffer(r->runner, out_bytes);

    uploadToBuffer(r->runner, d_latents, latents, lat_bytes);
    uploadToBuffer(r->runner, d_context, context, ctx_bytes);

    runDitForward(r, d_latents, timestep, d_context, d_output);

    downloadFromBuffer(r->runner, d_output, output, out_bytes);

    r->runner.destroyBuffer(d_latents);
    r->runner.destroyBuffer(d_context);
    r->runner.destroyBuffer(d_output);
    return 0;
}

} /* extern "C" */
