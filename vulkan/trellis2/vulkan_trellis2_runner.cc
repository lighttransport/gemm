// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan TRELLIS.2 Stage 1 Runner implementation
//
// Pipeline: DINOv3 encoder -> DiT diffusion (flow matching) -> Decoder -> occupancy
// Uses pre-compiled SPIR-V shaders and HOST_VISIBLE SSBOs for simplicity.
//

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

/* sparse3d.h and trellis2_shape_decoder.h reference ggml_dequant types.
 * Include header-only part for type definitions, provide minimal stub
 * for the implementation (shape decoder uses F32/F16 via safetensors,
 * not GGML quantized types). */
#include "../../common/ggml_dequant.h"
/* Stub: shape decoder never calls dequant_row with quantized types */
int dequant_row(uint32_t /*type*/, const void * /*src*/, float * /*dst*/, int /*n*/) { return 0; }

#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"

#include "vulkan_trellis2_runner.h"
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

/* DINOv3 ViT-L/16 */
#define DINO_HIDDEN     1024
#define DINO_HEADS      16
#define DINO_HEAD_DIM   64
#define DINO_LAYERS     24
#define DINO_FFN        4096
#define DINO_PATCH      16
#define DINO_IMG_SIZE   512
#define DINO_GRID       (DINO_IMG_SIZE / DINO_PATCH)  /* 32 */
#define DINO_N_PATCHES  (DINO_GRID * DINO_GRID)       /* 1024 */
#define DINO_N_STORAGE  4
#define DINO_SEQ_LEN    (1 + DINO_N_STORAGE + DINO_N_PATCHES)  /* 1029 */

/* DiT Stage 1 */
#define DIT_DIM         1536
#define DIT_HEADS       12
#define DIT_HEAD_DIM    128
#define DIT_FFN         8192
#define DIT_DEPTH       30
#define DIT_IN_CH       8
#define DIT_GRID        16
#define DIT_N_TOKENS    (DIT_GRID * DIT_GRID * DIT_GRID)  /* 4096 */
#define DIT_COND_DIM    DINO_HIDDEN                        /* 1024 */

/* ======================================================================== */
/* Types                                                                    */
/* ======================================================================== */

using VulkanRunner = vl_cpp::vulkan::VulkanComputeRunner;
using BufInfo = VulkanRunner::BufferInfo;
using Pipeline = VulkanRunner::ComputePipeline;

/* DiT per-block weights */
struct t2_dit_block_vk {
    /* Self-attention */
    BufInfo sa_qkv_w, sa_qkv_b;     /* [3*DIT_DIM, DIT_DIM], [3*DIT_DIM] */
    BufInfo sa_q_norm, sa_k_norm;    /* [DIT_HEADS * DIT_HEAD_DIM] */
    BufInfo sa_out_w, sa_out_b;
    /* Cross-attention */
    BufInfo norm2_w, norm2_b;
    BufInfo ca_q_w, ca_q_b;
    BufInfo ca_kv_w, ca_kv_b;
    BufInfo ca_q_norm, ca_k_norm;
    BufInfo ca_out_w, ca_out_b;
    /* MLP */
    BufInfo mlp_fc1_w, mlp_fc1_b;
    BufInfo mlp_fc2_w, mlp_fc2_b;
    /* Per-block modulation bias */
    BufInfo mod_bias;  /* [6*DIT_DIM] = [9216] */
};

/* Generic DiT model (shared structure for Stage 1/2/3) */
struct t2_dit_model_vk {
    int in_channels, out_channels;
    BufInfo t_fc1_w, t_fc1_b, t_fc2_w, t_fc2_b;  /* timestep MLP */
    BufInfo mod_w, mod_b;                           /* adaLN modulation */
    BufInfo x_emb_w, x_emb_b;                      /* input projection */
    BufInfo out_w, out_b;                           /* output projection */
    t2_dit_block_vk blocks[DIT_DEPTH];
    BufInfo rope_cos, rope_sin;                     /* precomputed for Stage 1, runtime for 2/3 */
    int n_freqs = 0;
    bool loaded = false;
};

/* DINOv3 per-layer weights */
struct t2_dino_layer_vk {
    BufInfo ln1_w, ln1_b;
    BufInfo q_w, q_b, k_w, k_b, v_w, v_b;  /* separate Q/K/V for RoPE */
    BufInfo out_w, out_b;
    BufInfo ls1, ls2;                        /* LayerScale gamma */
    BufInfo ln2_w, ln2_b;
    BufInfo fc1_w, fc1_b, fc2_w, fc2_b;
};

/* Decoder ResBlock weights */
struct t2_dec_resblock_vk {
    BufInfo gn1_w, gn1_b, conv1_w, conv1_b;
    BufInfo gn2_w, gn2_b, conv2_w, conv2_b;
};

struct vulkan_trellis2_runner {
    VulkanRunner runner;
    int verbose;
    std::string shader_dir;

    /* ---- Pipelines ---- */
    /* Reused from HY3D/common */
    Pipeline pipe_layernorm;         /* 4 bufs: src, dst, w, b; push: 12B */
    Pipeline pipe_gemm_f32;          /* 4 bufs: Y, W, X, bias; push: 16B */
    Pipeline pipe_gelu;              /* 1 buf: data; push: 4B */
    Pipeline pipe_add;               /* 2 bufs: dst, src; push: 4B */
    Pipeline pipe_rms_norm;          /* 2 bufs: data, w; push: 20B */
    Pipeline pipe_cross_attn;        /* 4 bufs: out, Q, K, V; push: 24B */
    Pipeline pipe_euler_step;        /* 2 bufs: x, v; push: 8B */
    Pipeline pipe_cfg_combine;       /* 3 bufs: out, cond, uncond; push: 8B */

    /* TRELLIS2-specific */
    Pipeline pipe_adaln;             /* 4 bufs: dst, src, shift, scale; push: 12B */
    Pipeline pipe_gated_add;         /* 3 bufs: dst, src, gate; push: 8B */
    Pipeline pipe_modulation;        /* 5 bufs: out, t_emb, W, b, blk_b; push: 8B */
    Pipeline pipe_rope_3d;           /* 3 bufs: data, cos, sin; push: 20B */
    Pipeline pipe_timestep_cossin;   /* 1 buf: out; push: 8B */
    Pipeline pipe_split_qkv_chunk;   /* 4 bufs: Q, K, V, qkv; push: 8B */
    Pipeline pipe_split_kv_chunk;    /* 3 bufs: K, V, kv; push: 8B */
    Pipeline pipe_layernorm_noaffine;/* 2 bufs: dst, src; push: 12B */
    Pipeline pipe_silu_inplace;      /* 1 buf: data; push: 4B */
    Pipeline pipe_conv3d;            /* 4 bufs: out, inp, w, bias; push: 24B */
    Pipeline pipe_channel_layernorm; /* 4 bufs: dst, src, w, b; push: 12B */
    Pipeline pipe_pixel_shuffle;     /* 2 bufs: dst, src; push: 16B */
    Pipeline pipe_self_attn_tiled;   /* 4 bufs: out, Q, K, V; push: 28B */

    /* Sparse decoder */
    Pipeline pipe_sparse_conv3d;     /* 7 bufs: out, feats, w, bias, keys, vals, coords; push: 16B */

    /* DINOv3-specific */
    Pipeline pipe_rope_2d_dinov3;    /* 4 bufs: Q, K, cos, sin; push: 20B */
    Pipeline pipe_dinov3_prepend;    /* 4 bufs: hidden, patches, cls, reg; push: 12B */
    Pipeline pipe_layerscale_add;    /* 3 bufs: dst, src, scale; push: 8B */
    Pipeline pipe_patch_embed;       /* 4 bufs: out, image, W, bias; push: 24B */
    bool pipelines_created = false;

    /* ---- DINOv3 weights ---- */
    BufInfo dino_patch_w, dino_patch_b;
    BufInfo dino_cls_token, dino_storage_tokens;
    t2_dino_layer_vk dino_layers[DINO_LAYERS];
    BufInfo dino_rope_cos, dino_rope_sin;
    bool dino_has_layerscale = false;

    /* ---- DiT models ---- */
    t2_dit_model_vk stage1_dit;   /* Stage 1: in=8, out=8, N=4096 */
    t2_dit_model_vk stage2_dit;   /* Stage 2: in=32, out=32, sparse N */
    t2_dit_model_vk stage3_dit;   /* Stage 3: in=64, out=32, sparse N */

    /* ---- Shape decoder (SC-VAE) ---- */
    /* Stored on CPU; ConvNeXt blocks dispatched to GPU, C2S runs on CPU */
    bool shape_dec_loaded = false;
    /* We use the CPU shape decoder from common/trellis2_shape_decoder.h
     * loaded at runtime. The Vulkan runner accelerates the ConvNeXt conv+MLP
     * but delegates C2S subdivision to CPU. */

    /* ---- Stage 1 Decoder weights ---- */
    BufInfo dec_conv_in_w, dec_conv_in_b;
    t2_dec_resblock_vk dec_middle[2];
    t2_dec_resblock_vk dec_res16[2];
    BufInfo dec_up1_w, dec_up1_b;
    t2_dec_resblock_vk dec_res32[2];
    BufInfo dec_up2_w, dec_up2_b;
    t2_dec_resblock_vk dec_res64[2];
    BufInfo dec_out_gn_w, dec_out_gn_b;
    BufInfo dec_out_conv_w, dec_out_conv_b;

    /* Scratch buffers */
    BufInfo scratch[12];
    size_t scratch_size[12] = {};

    /* Dummy buffer for unused bindings (Vulkan requires all to be valid) */
    BufInfo dummy_buf;

    /* Load status */
    bool dec_loaded = false;
    bool initialized = false;
};

/* ======================================================================== */
/* Helpers (mirrors HY3D runner pattern)                                    */
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
    /* Prefer DEVICE_LOCAL | HOST_VISIBLE (resizable BAR / SAM) for fast GPU access
     * with direct CPU mapping. Falls back to HOST_VISIBLE if not available. */
    VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!runner.createBuffer(size, usage, props, buf)) {
        /* Fallback: HOST_VISIBLE only (system RAM, accessible but slower for GPU) */
        props = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        runner.createBuffer(size, usage, props, buf);
    }
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

static void ensureScratch(vulkan_trellis2_runner *r, int idx, size_t bytes) {
    if (r->scratch_size[idx] < bytes) {
        if (r->scratch[idx].buffer != VK_NULL_HANDLE)
            r->runner.destroyBuffer(r->scratch[idx]);
        r->scratch[idx] = createGpuBuffer(r->runner, bytes);
        r->scratch_size[idx] = bytes;
    }
}

static bool isValid(const BufInfo &buf) {
    return buf.buffer != VK_NULL_HANDLE;
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

/* ======================================================================== */
/* Dispatch helpers                                                         */
/* ======================================================================== */

static void dispatchSync(vulkan_trellis2_runner *r, Pipeline &pipe,
                          const std::vector<BufInfo> &bufs,
                          const void *push_data, uint32_t push_size,
                          uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    /* Validate all buffer handles */
    for (size_t i = 0; i < bufs.size(); i++) {
        if (bufs[i].buffer == VK_NULL_HANDLE) {
            fprintf(stderr, "T2VK ERROR: dispatchSync: buffer %zu is VK_NULL_HANDLE (grid %u,%u,%u)\n",
                    i, gx, gy, gz);
            return;
        }
    }
    auto ds = r->runner.allocateAndUpdateDescriptorSet(pipe, bufs);
    if (ds == VK_NULL_HANDLE) {
        fprintf(stderr, "T2VK ERROR: descriptor alloc failed: %s\n",
                r->runner.getLastError().c_str());
        return;
    }
    r->runner.beginRecording();
    r->runner.bindComputePipeline(pipe);
    r->runner.bindDescriptorSetDynamic(pipe, ds);
    if (push_data && push_size > 0)
        r->runner.pushConstants(pipe, push_data, push_size);
    r->runner.dispatch(gx, gy, gz);
    r->runner.endRecordingAndSubmit();
    if (!r->runner.waitForCompletion()) {
        fprintf(stderr, "T2VK ERROR: GPU dispatch failed (grid %u,%u,%u): %s\n",
                gx, gy, gz, r->runner.getLastError().c_str());
    }
}

/* Forward declarations for ops used by opModulation */
static void opSiluInplace(vulkan_trellis2_runner *r, BufInfo &x, int n);
static void opGemm(vulkan_trellis2_runner *r, BufInfo &Y, BufInfo &W, BufInfo &X, BufInfo &bias,
                     int n_out, int n_in, int n_tok);
static void opAdd(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src, int n);
static void bufferCopy(VulkanRunner &runner, BufInfo &dst, BufInfo &src, size_t bytes);

/* ---- Op wrappers ---- */

/* AdaLN: dst = LN(src) * (1+scale) + shift */
static void opAdaLN(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src,
                     BufInfo &shift, BufInfo &scale, int n_tok, int dim) {
    struct { uint32_t n_tok, dim; float eps; } pc = {(uint32_t)n_tok, (uint32_t)dim, 1e-6f};
    dispatchSync(r, r->pipe_adaln, {dst, src, shift, scale}, &pc, sizeof(pc), (uint32_t)n_tok);
}

/* LayerNorm (affine): dst = LN(src, w, b) */
static void opLayerNorm(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src,
                          BufInfo &w, BufInfo &b, int n_tok, int dim) {
    struct { uint32_t n_tok, dim; float eps; } pc = {(uint32_t)n_tok, (uint32_t)dim, 1e-6f};
    dispatchSync(r, r->pipe_layernorm, {src, dst, w, b}, &pc, sizeof(pc), (uint32_t)n_tok);
}

/* LayerNorm (no affine): dst = LN(src) */
static void opLayerNormNoAffine(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src,
                                  int n_tok, int dim) {
    struct { uint32_t n_tok, dim; float eps; } pc = {(uint32_t)n_tok, (uint32_t)dim, 1e-6f};
    dispatchSync(r, r->pipe_layernorm_noaffine, {dst, src}, &pc, sizeof(pc), (uint32_t)n_tok);
}

/* GEMM: Y = X @ W^T + bias */
static void opGemm(vulkan_trellis2_runner *r, BufInfo &Y, BufInfo &W, BufInfo &X, BufInfo &bias,
                     int n_out, int n_in, int n_tok) {
    struct { uint32_t n_out, n_in, n_tok, has_bias; } pc = {
        (uint32_t)n_out, (uint32_t)n_in, (uint32_t)n_tok, isValid(bias) ? 1u : 0u
    };
    uint32_t gx = (uint32_t)((n_out + 63) / 64);
    uint32_t gy = (uint32_t)((n_tok + 15) / 16);
    /* Use dummy buffer for unused bias binding (Vulkan requires all bindings valid) */
    BufInfo &bias_buf = isValid(bias) ? bias : r->dummy_buf;
    dispatchSync(r, r->pipe_gemm_f32, {Y, W, X, bias_buf}, &pc, sizeof(pc), gx, gy);
}

/* GELU in-place */
static void opGelu(vulkan_trellis2_runner *r, BufInfo &x, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchSync(r, r->pipe_gelu, {x}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Element-wise add: dst += src */
static void opAdd(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchSync(r, r->pipe_add, {dst, src}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Gated add: dst += gate * src */
static void opGatedAdd(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src,
                         BufInfo &gate, int n, int dim) {
    struct { uint32_t n, dim; } pc = {(uint32_t)n, (uint32_t)dim};
    dispatchSync(r, r->pipe_gated_add, {dst, src, gate}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Modulation: out = SiLU(t_emb) @ W^T + bias + blk_bias
 * Decomposed into GEMM + add to avoid single-block matmul timeout on Vulkan */
static void opModulation(vulkan_trellis2_runner *r, BufInfo &out, BufInfo &t_emb_src,
                           BufInfo &W, BufInfo &bias, BufInfo &blk_bias,
                           int dim, int out_dim) {
    /* 1. Copy t_emb and apply SiLU in-place */
    BufInfo t_silu = createGpuBuffer(r->runner, (size_t)dim * sizeof(float));
    bufferCopy(r->runner, t_silu, t_emb_src, (size_t)dim * sizeof(float));
    opSiluInplace(r, t_silu, dim);

    /* 2. GEMM: out = t_silu @ W^T + bias  (n_tok=1) */
    opGemm(r, out, W, t_silu, bias, out_dim, dim, 1);

    /* 3. Add per-block bias: out += blk_bias */
    if (isValid(blk_bias))
        opAdd(r, out, blk_bias, out_dim);

    r->runner.destroyBuffer(t_silu);
}

/* Per-head RMSNorm */
static void opRmsNorm(vulkan_trellis2_runner *r, BufInfo &data, BufInfo &w,
                        int n_tok, int n_heads, int head_dim, int stride) {
    struct { uint32_t n_tok, n_heads, head_dim, stride; float eps; } pc = {
        (uint32_t)n_tok, (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)stride, 1e-6f
    };
    int total = n_tok * n_heads;
    dispatchSync(r, r->pipe_rms_norm, {data, w}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* 3D RoPE */
static void opRope3D(vulkan_trellis2_runner *r, BufInfo &data,
                       BufInfo &cos_tab, BufInfo &sin_tab,
                       int N, int dim, int n_heads, int head_dim, int n_freqs) {
    struct { uint32_t N, dim, n_heads, head_dim, n_freqs; } pc = {
        (uint32_t)N, (uint32_t)dim, (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)n_freqs
    };
    dispatchSync(r, r->pipe_rope_3d, {data, cos_tab, sin_tab}, &pc, sizeof(pc), (uint32_t)N);
}

/* Cross-attention */
static void opCrossAttn(vulkan_trellis2_runner *r, BufInfo &out,
                          BufInfo &Q, BufInfo &K, BufInfo &V,
                          int q_len, int kv_len, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    struct { uint32_t q_len, kv_len, dim, n_heads, head_dim; float scale; } pc = {
        (uint32_t)q_len, (uint32_t)kv_len, (uint32_t)dim, (uint32_t)n_heads, (uint32_t)head_dim, scale
    };
    dispatchSync(r, r->pipe_cross_attn, {out, Q, K, V}, &pc, sizeof(pc),
                 (uint32_t)n_heads, (uint32_t)q_len);
}

/* Self-attention: dispatched in batches of queries to avoid GPU timeout.
 * Uses tiled online-softmax with 32 threads per workgroup for parallel dot products. */
static void opSelfAttn(vulkan_trellis2_runner *r, BufInfo &out,
                         BufInfo &Q, BufInfo &K, BufInfo &V,
                         int n_tok, int dim, int n_heads, int head_dim) {
    if (n_tok <= 1024) {
        /* Small N: use shared-memory attention (faster) */
        opCrossAttn(r, out, Q, K, V, n_tok, n_tok, dim, n_heads, head_dim);
        return;
    }

    /* Large N: batch queries to avoid GPU timeout.
     * Each batch processes BATCH_SIZE queries across all heads.
     * With N=4096 and BATCH=256: 16 dispatches of (12, 256) = 3072 workgroups each. */
    const int BATCH_SIZE = 256;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int q_start = 0; q_start < n_tok; q_start += BATCH_SIZE) {
        int q_count = std::min(BATCH_SIZE, n_tok - q_start);
        struct { uint32_t q_start, q_count, kv_len, dim, n_heads, head_dim; float scale; } pc = {
            (uint32_t)q_start, (uint32_t)q_count, (uint32_t)n_tok,
            (uint32_t)dim, (uint32_t)n_heads, (uint32_t)head_dim, scale
        };
        dispatchSync(r, r->pipe_self_attn_tiled, {out, Q, K, V}, &pc, sizeof(pc),
                     (uint32_t)n_heads, (uint32_t)q_count);
    }
}

/* Timestep embedding: [cos, sin] order */
static void opTimestepEmbedCosSin(vulkan_trellis2_runner *r, BufInfo &out, float t, int dim) {
    struct { float t; uint32_t dim; } pc = {t, (uint32_t)dim};
    int half = dim / 2;
    dispatchSync(r, r->pipe_timestep_cossin, {out}, &pc, sizeof(pc), (uint32_t)((half + 255) / 256));
}

/* SiLU in-place */
static void opSiluInplace(vulkan_trellis2_runner *r, BufInfo &x, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchSync(r, r->pipe_silu_inplace, {x}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Split QKV (chunk) */
static void opSplitQkvChunk(vulkan_trellis2_runner *r, BufInfo &Q, BufInfo &K, BufInfo &V,
                              BufInfo &qkv, int N, int W) {
    struct { uint32_t N, W; } pc = {(uint32_t)N, (uint32_t)W};
    int total = N * W;
    dispatchSync(r, r->pipe_split_qkv_chunk, {Q, K, V, qkv}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Split KV (chunk) */
static void opSplitKvChunk(vulkan_trellis2_runner *r, BufInfo &K, BufInfo &V,
                             BufInfo &kv, int M, int W) {
    struct { uint32_t M, W; } pc = {(uint32_t)M, (uint32_t)W};
    int total = M * W;
    dispatchSync(r, r->pipe_split_kv_chunk, {K, V, kv}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Conv3D k=3 pad=1 stride=1 */
static void opConv3d(vulkan_trellis2_runner *r, BufInfo &out, BufInfo &inp,
                       BufInfo &w, BufInfo &bias, int Ci, int Co, int D, int H, int W) {
    struct { uint32_t Ci, Co, D, H, W, has_bias; } pc = {
        (uint32_t)Ci, (uint32_t)Co, (uint32_t)D, (uint32_t)H, (uint32_t)W,
        isValid(bias) ? 1u : 0u
    };
    int spatial = D * H * W;
    BufInfo &bias_buf = isValid(bias) ? bias : r->dummy_buf;
    dispatchSync(r, r->pipe_conv3d, {out, inp, w, bias_buf}, &pc, sizeof(pc),
                 (uint32_t)Co, (uint32_t)((spatial + 63) / 64));
}

/* Channel LayerNorm 3D */
static void opChannelLayerNorm3D(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src,
                                   BufInfo &w, BufInfo &b, int C, int spatial) {
    struct { uint32_t C, spatial, has_wb; } pc = {
        (uint32_t)C, (uint32_t)spatial, (isValid(w) ? 1u : 0u)
    };
    BufInfo &w_buf = isValid(w) ? w : r->dummy_buf;
    BufInfo &b_buf = isValid(b) ? b : r->dummy_buf;
    dispatchSync(r, r->pipe_channel_layernorm, {dst, src, w_buf, b_buf}, &pc, sizeof(pc),
                 (uint32_t)((spatial + 255) / 256));
}

/* Pixel shuffle 3D: [C*8, D, H, W] -> [C, 2D, 2H, 2W] */
static void opPixelShuffle3D(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src,
                               int C, int D, int H, int W) {
    struct { uint32_t C, D, H, W; } pc = {(uint32_t)C, (uint32_t)D, (uint32_t)H, (uint32_t)W};
    int total = C * 2 * D * 2 * H * 2 * W;
    dispatchSync(r, r->pipe_pixel_shuffle, {dst, src}, &pc, sizeof(pc), (uint32_t)((total + 255) / 256));
}

/* Euler step: x -= dt * v */
static void opEulerStep(vulkan_trellis2_runner *r, BufInfo &x, BufInfo &v, float dt, int n) {
    struct { float dt; uint32_t n; } pc = {dt, (uint32_t)n};
    dispatchSync(r, r->pipe_euler_step, {x, v}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* CFG combine: out = uncond + scale*(cond - uncond) */
static void opCfgCombine(vulkan_trellis2_runner *r, BufInfo &out, BufInfo &cond,
                           BufInfo &uncond, float scale, int n) {
    struct { float scale; uint32_t n; } pc = {scale, (uint32_t)n};
    dispatchSync(r, r->pipe_cfg_combine, {out, cond, uncond}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Buffer operations */
static void bufferCopy(VulkanRunner &runner, BufInfo &dst, BufInfo &src, size_t bytes) {
    void *sp = nullptr, *dp = nullptr;
    runner.mapBuffer(src, &sp);
    runner.mapBuffer(dst, &dp);
    memcpy(dp, sp, bytes);
    runner.unmapBuffer(src);
    runner.unmapBuffer(dst);
}

static void bufferZero(VulkanRunner &runner, BufInfo &buf, size_t bytes) {
    void *ptr = nullptr;
    runner.mapBuffer(buf, &ptr);
    memset(ptr, 0, bytes);
    runner.unmapBuffer(buf);
}

/* ---- DINOv3-specific ops ---- */

/* 2D RoPE for DINOv3 (rotate_half, patch tokens only) */
static void opRope2dDinov3(vulkan_trellis2_runner *r, BufInfo &Q, BufInfo &K,
                             BufInfo &cos_tab, BufInfo &sin_tab,
                             int n_patches, int n_prefix, int dim, int n_heads, int head_dim) {
    struct { uint32_t n_patches, n_prefix, dim, n_heads, head_dim; } pc = {
        (uint32_t)n_patches, (uint32_t)n_prefix, (uint32_t)dim,
        (uint32_t)n_heads, (uint32_t)head_dim
    };
    dispatchSync(r, r->pipe_rope_2d_dinov3, {Q, K, cos_tab, sin_tab}, &pc, sizeof(pc),
                 (uint32_t)n_patches);
}

/* Prepend CLS + register tokens */
static void opDinov3PrependTokens(vulkan_trellis2_runner *r, BufInfo &hidden,
                                    BufInfo &patches, BufInfo &cls, BufInfo &reg,
                                    int n_patches, int n_reg, int dim) {
    struct { uint32_t n_patches, n_reg, dim; } pc = {
        (uint32_t)n_patches, (uint32_t)n_reg, (uint32_t)dim
    };
    int n_prefix = 1 + n_reg;
    int total = (n_prefix + n_patches) * dim;
    dispatchSync(r, r->pipe_dinov3_prepend, {hidden, patches, cls, reg}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* LayerScale add: dst[i] += src[i] * scale[i % dim] */
static void opLayerscaleAdd(vulkan_trellis2_runner *r, BufInfo &dst, BufInfo &src,
                              BufInfo &scale, int n, int dim) {
    struct { uint32_t n, dim; } pc = {(uint32_t)n, (uint32_t)dim};
    dispatchSync(r, r->pipe_layerscale_add, {dst, src, scale}, &pc, sizeof(pc),
                 (uint32_t)((n + 255) / 256));
}

/* Patch embedding: conv2d-like from image patches */
static void opPatchEmbed(vulkan_trellis2_runner *r, BufInfo &out, BufInfo &image,
                           BufInfo &w, BufInfo &bias,
                           int n_patches, int dim, int patch_size, int grid_w, int img_w) {
    struct { uint32_t n_patches, dim, kernel_size, gw, width, patch_size; } pc = {
        (uint32_t)n_patches, (uint32_t)dim, (uint32_t)(patch_size * patch_size * 3),
        (uint32_t)grid_w, (uint32_t)img_w, (uint32_t)patch_size
    };
    dispatchSync(r, r->pipe_patch_embed, {image, w, bias, out}, &pc, sizeof(pc),
                 (uint32_t)((n_patches * dim + 255) / 256));
}

/* ======================================================================== */
/* Pipeline creation                                                        */
/* ======================================================================== */

static bool createPipelines(vulkan_trellis2_runner *r) {
    auto loadSpv = [&](const std::string &name) -> std::vector<uint32_t> {
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
        fprintf(stderr, "T2VK: failed to load shader: %s.spv (searched %s)\n",
                name.c_str(), r->shader_dir.c_str());
        return {};
    };

    auto createPipe = [&](const std::string &name, int nBuf, int pushSize, Pipeline &pipe) -> bool {
        auto spirv = loadSpv(name);
        if (spirv.empty()) return false;
        return r->runner.createComputePipelineWithPushConstants(spirv, makeBindings(nBuf), pushSize, pipe);
    };

    /* Reused common/HY3D shaders */
    if (!createPipe("shaders/layernorm_f32",                   4, 12, r->pipe_layernorm)) return false;
    if (!createPipe("shaders/hy3d/gemm_f32_f32",              4, 16, r->pipe_gemm_f32)) return false;
    if (!createPipe("shaders/gelu_f32",                        1,  4, r->pipe_gelu)) return false;
    if (!createPipe("shaders/add_f32",                         2,  4, r->pipe_add)) return false;
    if (!createPipe("shaders/hy3d/rms_norm_f32",               2, 20, r->pipe_rms_norm)) return false;
    if (!createPipe("shaders/hy3d/cross_attn_f32",             4, 24, r->pipe_cross_attn)) return false;
    if (!createPipe("shaders/hy3d/euler_step_f32",             2,  8, r->pipe_euler_step)) return false;
    if (!createPipe("shaders/hy3d/cfg_combine_f32",            3,  8, r->pipe_cfg_combine)) return false;

    /* TRELLIS2-specific shaders */
    if (!createPipe("shaders/trellis2/adaln_f32",              4, 12, r->pipe_adaln)) return false;
    if (!createPipe("shaders/trellis2/gated_add_f32",          3,  8, r->pipe_gated_add)) return false;
    if (!createPipe("shaders/trellis2/modulation_f32",         5,  8, r->pipe_modulation)) return false;
    if (!createPipe("shaders/trellis2/rope_3d_f32",            3, 20, r->pipe_rope_3d)) return false;
    if (!createPipe("shaders/trellis2/timestep_embed_cossin_f32", 1, 8, r->pipe_timestep_cossin)) return false;
    if (!createPipe("shaders/trellis2/split_qkv_chunk_f32",    4,  8, r->pipe_split_qkv_chunk)) return false;
    if (!createPipe("shaders/trellis2/split_kv_chunk_f32",     3,  8, r->pipe_split_kv_chunk)) return false;
    if (!createPipe("shaders/trellis2/layernorm_noaffine_f32",  2, 12, r->pipe_layernorm_noaffine)) return false;
    if (!createPipe("shaders/trellis2/silu_inplace_f32",        1,  4, r->pipe_silu_inplace)) return false;
    if (!createPipe("shaders/trellis2/conv3d_k3_f32",           4, 24, r->pipe_conv3d)) return false;
    if (!createPipe("shaders/trellis2/channel_layernorm_3d_f32", 4, 12, r->pipe_channel_layernorm)) return false;
    if (!createPipe("shaders/trellis2/pixel_shuffle_3d_f32",    2, 16, r->pipe_pixel_shuffle)) return false;
    if (!createPipe("shaders/trellis2/self_attn_tiled_f32",   4, 28, r->pipe_self_attn_tiled)) return false;

    /* Sparse decoder */
    if (!createPipe("shaders/trellis2/sparse_conv3d_f32",      7, 16, r->pipe_sparse_conv3d)) return false;

    /* DINOv3-specific */
    if (!createPipe("shaders/trellis2/rope_2d_dinov3_f32",     4, 20, r->pipe_rope_2d_dinov3)) return false;
    if (!createPipe("shaders/trellis2/dinov3_prepend_tokens_f32", 4, 12, r->pipe_dinov3_prepend)) return false;
    if (!createPipe("shaders/hy3d/layerscale_add_f32",          3,  8, r->pipe_layerscale_add)) return false;
    if (!createPipe("shaders/patch_embed_f32",                  4, 24, r->pipe_patch_embed)) return false;

    r->pipelines_created = true;

    /* Create a small dummy buffer for unused descriptor bindings */
    r->dummy_buf = createGpuBuffer(r->runner, 64);

    return true;
}

/* ======================================================================== */
/* Weight loading                                                           */
/* ======================================================================== */

/* Helper: get tensor data as F32 vector (handles F16/BF16/F32 conversion) */
static std::vector<float> stGetF32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return {};
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
    } else if (strcmp(dtype, "BF16") == 0) {
        size_t n = nbytes / sizeof(uint16_t);
        std::vector<float> v(n);
        const uint16_t *bf16 = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t f = (uint32_t)bf16[i] << 16;
            memcpy(&v[i], &f, sizeof(float));
        }
        return v;
    }
    return {};
}

/* Upload F32 data to GPU buffer */
static BufInfo uploadF32(VulkanRunner &runner, const float *data, size_t n) {
    BufInfo buf = createGpuBuffer(runner, n * sizeof(float));
    uploadToBuffer(runner, buf, data, n * sizeof(float));
    return buf;
}

static int loadDinov3Weights(vulkan_trellis2_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "T2VK: cannot open DINOv3: %s\n", path); return -1; }
    fprintf(stderr, "T2VK: loading DINOv3 from %s (%d tensors)\n", path, st->n_tensors);
    int v = r->verbose;

    /* Patch embedding + tokens (timm naming) */
    r->dino_patch_w = stUploadF32(r->runner, st, "patch_embed.proj.weight", v);
    r->dino_patch_b = stUploadF32(r->runner, st, "patch_embed.proj.bias", v);
    r->dino_cls_token = stUploadF32(r->runner, st, "cls_token", v);
    r->dino_storage_tokens = stUploadF32(r->runner, st, "reg_token", v);

    r->dino_has_layerscale = false;

    /* Per-layer weights */
    for (int L = 0; L < DINO_LAYERS; L++) {
        t2_dino_layer_vk &l = r->dino_layers[L];
        char name[256];

        #define DINO_LOAD(field, suffix) do { \
            snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix); \
            l.field = stUploadF32(r->runner, st, name, v >= 2 ? v : 0); \
        } while(0)

        DINO_LOAD(ln1_w, "norm1.weight");
        DINO_LOAD(ln1_b, "norm1.bias");

        /* Split fused QKV [3*dim, dim] into separate Q, K, V on CPU */
        snprintf(name, sizeof(name), "blocks.%d.attn.qkv.weight", L);
        auto qkv_f32 = stGetF32(st, name);
        if (!qkv_f32.empty()) {
            int dim = DINO_HIDDEN;
            l.q_w = uploadF32(r->runner, qkv_f32.data(), (size_t)dim * dim);
            l.k_w = uploadF32(r->runner, qkv_f32.data() + dim * dim, (size_t)dim * dim);
            l.v_w = uploadF32(r->runner, qkv_f32.data() + 2 * dim * dim, (size_t)dim * dim);
            /* timm QKV has no bias — create zeros */
            std::vector<float> zeros(dim, 0.0f);
            l.q_b = uploadF32(r->runner, zeros.data(), dim);
            l.k_b = uploadF32(r->runner, zeros.data(), dim);
            l.v_b = uploadF32(r->runner, zeros.data(), dim);
        }

        DINO_LOAD(out_w, "attn.proj.weight");
        DINO_LOAD(out_b, "attn.proj.bias");
        DINO_LOAD(ln2_w, "norm2.weight");
        DINO_LOAD(ln2_b, "norm2.bias");
        DINO_LOAD(fc1_w, "mlp.fc1.weight");
        DINO_LOAD(fc1_b, "mlp.fc1.bias");
        DINO_LOAD(fc2_w, "mlp.fc2.weight");
        DINO_LOAD(fc2_b, "mlp.fc2.bias");

        /* LayerScale (timm: gamma_1, gamma_2) */
        snprintf(name, sizeof(name), "blocks.%d.gamma_1", L);
        l.ls1 = stUploadF32(r->runner, st, name, 0);
        snprintf(name, sizeof(name), "blocks.%d.gamma_2", L);
        l.ls2 = stUploadF32(r->runner, st, name, 0);
        if (isValid(l.ls1)) r->dino_has_layerscale = true;

        #undef DINO_LOAD
    }

    /* Precompute 2D RoPE tables for 32×32 grid */
    {
        const int gh = DINO_IMG_SIZE / DINO_PATCH;  /* 32 */
        const int np = gh * gh;                      /* 1024 */
        const int hd = DINO_HEAD_DIM;               /* 64 */
        const float rope_theta = 100.0f;
        const int n_freqs = hd / 4;                 /* 16 */

        std::vector<float> inv_freq(n_freqs);
        for (int j = 0; j < n_freqs; j++)
            inv_freq[j] = 1.0f / powf(rope_theta, (float)(4 * j) / (float)hd);

        std::vector<float> cos_tab(np * hd), sin_tab(np * hd);
        for (int p = 0; p < np; p++) {
            int py = p / gh, px = p % gh;
            float cy = ((0.5f + py) / gh) * 2.0f - 1.0f;
            float cx = ((0.5f + px) / gh) * 2.0f - 1.0f;

            /* angles: [y_freqs(16), x_freqs(16)] tiled 2x = [y16, x16, y16, x16] = 64 */
            float angles[64];
            for (int j = 0; j < n_freqs; j++) {
                angles[j]              = 2.0f * 3.14159265358979f * cy * inv_freq[j];
                angles[n_freqs + j]    = 2.0f * 3.14159265358979f * cx * inv_freq[j];
                angles[2*n_freqs + j]  = angles[j];        /* tile */
                angles[3*n_freqs + j]  = angles[n_freqs + j];
            }
            for (int d = 0; d < hd; d++) {
                cos_tab[p * hd + d] = cosf(angles[d]);
                sin_tab[p * hd + d] = sinf(angles[d]);
            }
        }
        r->dino_rope_cos = uploadF32(r->runner, cos_tab.data(), np * hd);
        r->dino_rope_sin = uploadF32(r->runner, sin_tab.data(), np * hd);
        fprintf(stderr, "T2VK: DINOv3 RoPE tables uploaded (%d freqs, %dx%d grid)\n", n_freqs, gh, gh);
    }

    safetensors_close(st);
    fprintf(stderr, "T2VK: DINOv3 loaded (%d blocks, layerscale=%d)\n",
            DINO_LAYERS, r->dino_has_layerscale ? 1 : 0);
    return 0;
}

/* Generic DiT weight loader: loads into a t2_dit_model_vk struct.
 * label is for log messages (e.g. "Stage 1", "Stage 2"). */
static int loadDitModelWeights(vulkan_trellis2_runner *r, const char *path,
                                t2_dit_model_vk *m, const char *label) {
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "T2VK: cannot open %s DiT: %s\n", label, path); return -1; }
    fprintf(stderr, "T2VK: loading %s DiT from %s (%d tensors)\n", label, path, st->n_tensors);
    int v = r->verbose;

    /* Detect in/out channels from input_layer.weight shape */
    {
        int idx = safetensors_find(st, "input_layer.weight");
        if (idx >= 0 && safetensors_ndims(st, idx) >= 2) {
            const uint64_t *sh = safetensors_shape(st, idx);
            m->in_channels = (int)sh[1];
            fprintf(stderr, "T2VK:   in_channels=%d (from input_layer.weight[%llu, %llu])\n",
                    m->in_channels, (unsigned long long)sh[0], (unsigned long long)sh[1]);
        }
        idx = safetensors_find(st, "out_layer.weight");
        if (idx >= 0 && safetensors_ndims(st, idx) >= 2) {
            const uint64_t *sh = safetensors_shape(st, idx);
            m->out_channels = (int)sh[0];
            fprintf(stderr, "T2VK:   out_channels=%d\n", m->out_channels);
        }
    }

    /* Top-level */
    m->t_fc1_w = stUploadF32(r->runner, st, "t_embedder.mlp.0.weight", v);
    m->t_fc1_b = stUploadF32(r->runner, st, "t_embedder.mlp.0.bias", v);
    m->t_fc2_w = stUploadF32(r->runner, st, "t_embedder.mlp.2.weight", v);
    m->t_fc2_b = stUploadF32(r->runner, st, "t_embedder.mlp.2.bias", v);
    m->mod_w   = stUploadF32(r->runner, st, "adaLN_modulation.1.weight", v);
    m->mod_b   = stUploadF32(r->runner, st, "adaLN_modulation.1.bias", v);
    m->x_emb_w = stUploadF32(r->runner, st, "input_layer.weight", v);
    m->x_emb_b = stUploadF32(r->runner, st, "input_layer.bias", v);
    m->out_w   = stUploadF32(r->runner, st, "out_layer.weight", v);
    m->out_b   = stUploadF32(r->runner, st, "out_layer.bias", v);

    /* Per-block */
    for (int L = 0; L < DIT_DEPTH; L++) {
        t2_dit_block_vk &blk = m->blocks[L];
        char name[256];

        #define BLK(field, suffix) do { \
            snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix); \
            blk.field = stUploadF32(r->runner, st, name, v >= 2 ? v : 0); \
        } while(0)

        BLK(sa_qkv_w,    "self_attn.to_qkv.weight");
        BLK(sa_qkv_b,    "self_attn.to_qkv.bias");
        BLK(sa_q_norm,   "self_attn.q_rms_norm.gamma");
        BLK(sa_k_norm,   "self_attn.k_rms_norm.gamma");
        BLK(sa_out_w,    "self_attn.to_out.weight");
        BLK(sa_out_b,    "self_attn.to_out.bias");

        BLK(norm2_w,     "norm2.weight");
        BLK(norm2_b,     "norm2.bias");
        BLK(ca_q_w,      "cross_attn.to_q.weight");
        BLK(ca_q_b,      "cross_attn.to_q.bias");
        BLK(ca_kv_w,     "cross_attn.to_kv.weight");
        BLK(ca_kv_b,     "cross_attn.to_kv.bias");
        BLK(ca_q_norm,   "cross_attn.q_rms_norm.gamma");
        BLK(ca_k_norm,   "cross_attn.k_rms_norm.gamma");
        BLK(ca_out_w,    "cross_attn.to_out.weight");
        BLK(ca_out_b,    "cross_attn.to_out.bias");

        BLK(mlp_fc1_w,   "mlp.mlp.0.weight");
        BLK(mlp_fc1_b,   "mlp.mlp.0.bias");
        BLK(mlp_fc2_w,   "mlp.mlp.2.weight");
        BLK(mlp_fc2_b,   "mlp.mlp.2.bias");

        BLK(mod_bias,    "modulation");

        #undef BLK
    }

    m->n_freqs = DIT_HEAD_DIM / 6;  /* 21 */
    safetensors_close(st);
    m->loaded = true;
    fprintf(stderr, "T2VK: %s DiT loaded (%d blocks, in=%d, out=%d)\n",
            label, DIT_DEPTH, m->in_channels, m->out_channels);
    return 0;
}

/* Precompute 3D RoPE tables for a regular grid (Stage 1) */
static void precomputeGridRope(vulkan_trellis2_runner *r, t2_dit_model_vk *m,
                                int grid_size) {
    const int nt = grid_size * grid_size * grid_size;
    const int n_freqs = m->n_freqs;

    std::vector<float> freqs(n_freqs);
    for (int j = 0; j < n_freqs; j++)
        freqs[j] = 1.0f / powf(10000.0f, (float)j / (float)n_freqs);

    size_t table_sz = (size_t)nt * 3 * n_freqs;
    std::vector<float> cos_tab(table_sz), sin_tab(table_sz);

    for (int i = 0; i < nt; i++) {
        int z = i / (grid_size * grid_size), y = (i / grid_size) % grid_size, x = i % grid_size;
        float coords[3] = {(float)z, (float)y, (float)x};
        for (int axis = 0; axis < 3; axis++) {
            for (int j = 0; j < n_freqs; j++) {
                float theta = coords[axis] * freqs[j];
                cos_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = cosf(theta);
                sin_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = sinf(theta);
            }
        }
    }

    m->rope_cos = createGpuBuffer(r->runner, table_sz * sizeof(float));
    uploadToBuffer(r->runner, m->rope_cos, cos_tab.data(), table_sz * sizeof(float));
    m->rope_sin = createGpuBuffer(r->runner, table_sz * sizeof(float));
    uploadToBuffer(r->runner, m->rope_sin, sin_tab.data(), table_sz * sizeof(float));
    fprintf(stderr, "T2VK: RoPE tables uploaded (%d freqs/axis, grid=%d)\n", n_freqs, grid_size);
}

/* Compute 3D RoPE tables from sparse coordinates (Stage 2/3) */
static void computeSparseRope(vulkan_trellis2_runner *r, const int32_t *coords, int N,
                               int n_freqs, BufInfo &rope_cos, BufInfo &rope_sin) {
    std::vector<float> freqs(n_freqs);
    for (int j = 0; j < n_freqs; j++)
        freqs[j] = 1.0f / powf(10000.0f, (float)j / (float)n_freqs);

    size_t table_sz = (size_t)N * 3 * n_freqs;
    std::vector<float> cos_tab(table_sz), sin_tab(table_sz);

    for (int i = 0; i < N; i++) {
        /* coords[i*4 + 0] = batch, [1]=z, [2]=y, [3]=x */
        for (int axis = 0; axis < 3; axis++) {
            float c = (float)coords[i * 4 + 1 + axis];
            for (int j = 0; j < n_freqs; j++) {
                float theta = c * freqs[j];
                cos_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = cosf(theta);
                sin_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = sinf(theta);
            }
        }
    }

    rope_cos = uploadF32(r->runner, cos_tab.data(), table_sz);
    rope_sin = uploadF32(r->runner, sin_tab.data(), table_sz);
}

static int loadDitWeights(vulkan_trellis2_runner *r, const char *path) {
    r->stage1_dit.in_channels = DIT_IN_CH;
    r->stage1_dit.out_channels = DIT_IN_CH;
    int ret = loadDitModelWeights(r, path, &r->stage1_dit, "Stage 1");
    if (ret == 0)
        precomputeGridRope(r, &r->stage1_dit, DIT_GRID);
    return ret;
}

static int loadDecoderWeights(vulkan_trellis2_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "T2VK: cannot open decoder: %s\n", path); return -1; }
    fprintf(stderr, "T2VK: loading decoder from %s (%d tensors)\n", path, st->n_tensors);
    int v = r->verbose;

    r->dec_conv_in_w = stUploadF32(r->runner, st, "input_layer.weight", v);
    r->dec_conv_in_b = stUploadF32(r->runner, st, "input_layer.bias", v);

    #define LOAD_RES(rb, prefix) do { \
        char _n[256]; \
        snprintf(_n, sizeof(_n), "%snorm1.weight", prefix); (rb).gn1_w = stUploadF32(r->runner, st, _n, v>=2?v:0); \
        snprintf(_n, sizeof(_n), "%snorm1.bias", prefix);   (rb).gn1_b = stUploadF32(r->runner, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv1.weight", prefix); (rb).conv1_w = stUploadF32(r->runner, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv1.bias", prefix);   (rb).conv1_b = stUploadF32(r->runner, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%snorm2.weight", prefix); (rb).gn2_w = stUploadF32(r->runner, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%snorm2.bias", prefix);   (rb).gn2_b = stUploadF32(r->runner, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv2.weight", prefix); (rb).conv2_w = stUploadF32(r->runner, st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv2.bias", prefix);   (rb).conv2_b = stUploadF32(r->runner, st, _n, 0); \
    } while(0)

    LOAD_RES(r->dec_middle[0], "middle_block.0.");
    LOAD_RES(r->dec_middle[1], "middle_block.1.");
    LOAD_RES(r->dec_res16[0],  "blocks.0.");
    LOAD_RES(r->dec_res16[1],  "blocks.1.");
    r->dec_up1_w = stUploadF32(r->runner, st, "blocks.2.conv.weight", v);
    r->dec_up1_b = stUploadF32(r->runner, st, "blocks.2.conv.bias", v);
    LOAD_RES(r->dec_res32[0],  "blocks.3.");
    LOAD_RES(r->dec_res32[1],  "blocks.4.");
    r->dec_up2_w = stUploadF32(r->runner, st, "blocks.5.conv.weight", v);
    r->dec_up2_b = stUploadF32(r->runner, st, "blocks.5.conv.bias", v);
    LOAD_RES(r->dec_res64[0],  "blocks.6.");
    LOAD_RES(r->dec_res64[1],  "blocks.7.");
    r->dec_out_gn_w   = stUploadF32(r->runner, st, "out_layer.0.weight", v);
    r->dec_out_gn_b   = stUploadF32(r->runner, st, "out_layer.0.bias", v);
    r->dec_out_conv_w  = stUploadF32(r->runner, st, "out_layer.2.weight", v);
    r->dec_out_conv_b  = stUploadF32(r->runner, st, "out_layer.2.bias", v);

    #undef LOAD_RES

    safetensors_close(st);
    r->dec_loaded = true;
    fprintf(stderr, "T2VK: decoder loaded\n");
    return 0;
}

/* ======================================================================== */
/* DINOv3 forward pass                                                      */
/* ======================================================================== */

static void runDinov3(vulkan_trellis2_runner *r,
                        BufInfo &d_image, BufInfo &d_out) {
    const int dim = DINO_HIDDEN;        /* 1024 */
    const int heads = DINO_HEADS;       /* 16 */
    const int hd = DINO_HEAD_DIM;       /* 64 */
    const int ffn = DINO_FFN;           /* 4096 */
    const int ps = DINO_PATCH;          /* 16 */
    const int gw = DINO_IMG_SIZE / ps;  /* 32 */
    const int np = DINO_N_PATCHES;      /* 1024 */
    const int ns = DINO_N_STORAGE;      /* 4 */
    const int seq = DINO_SEQ_LEN;       /* 1029 */
    const int n_prefix = 1 + ns;        /* 5 */

    if (r->verbose) fprintf(stderr, "T2VK: DINOv3 forward (seq=%d)\n", seq);
    r->runner.resetDynamicDescriptorPool();

    /* Scratch buffers */
    BufInfo d_hidden = createGpuBuffer(r->runner, (size_t)seq * dim * sizeof(float));
    BufInfo d_Q      = createGpuBuffer(r->runner, (size_t)seq * dim * sizeof(float));
    BufInfo d_K      = createGpuBuffer(r->runner, (size_t)seq * dim * sizeof(float));
    BufInfo d_V      = createGpuBuffer(r->runner, (size_t)seq * dim * sizeof(float));
    BufInfo d_attn   = createGpuBuffer(r->runner, (size_t)seq * dim * sizeof(float));
    BufInfo d_mlp    = createGpuBuffer(r->runner, (size_t)seq * ffn * sizeof(float));
    BufInfo d_normed = createGpuBuffer(r->runner, (size_t)seq * dim * sizeof(float));

    /* 1. Patch embedding: image [3, 512, 512] -> patches [1024, 1024] */
    BufInfo d_patches = createGpuBuffer(r->runner, (size_t)np * dim * sizeof(float));
    opPatchEmbed(r, d_patches, d_image, r->dino_patch_w, r->dino_patch_b,
                 np, dim, ps, gw, DINO_IMG_SIZE);

    /* 2. Prepend CLS + register tokens -> hidden [1029, 1024] */
    opDinov3PrependTokens(r, d_hidden, d_patches, r->dino_cls_token,
                           r->dino_storage_tokens, np, ns, dim);
    r->runner.destroyBuffer(d_patches);

    /* 3. Transformer blocks */
    for (int li = 0; li < DINO_LAYERS; li++) {
        t2_dino_layer_vk &l = r->dino_layers[li];

        /* LN1 -> Q, K, V projections */
        opLayerNorm(r, d_normed, d_hidden, l.ln1_w, l.ln1_b, seq, dim);
        opGemm(r, d_Q, l.q_w, d_normed, l.q_b, dim, dim, seq);
        opGemm(r, d_K, l.k_w, d_normed, l.k_b, dim, dim, seq);
        opGemm(r, d_V, l.v_w, d_normed, l.v_b, dim, dim, seq);

        /* 2D RoPE on Q and K (patch tokens only, skip prefix) */
        opRope2dDinov3(r, d_Q, d_K, r->dino_rope_cos, r->dino_rope_sin,
                        np, n_prefix, dim, heads, hd);

        /* Self-attention (seq=1029 is small enough for shared-memory version) */
        opCrossAttn(r, d_attn, d_Q, d_K, d_V, seq, seq, dim, heads, hd);

        /* Output projection */
        opGemm(r, d_normed, l.out_w, d_attn, l.out_b, dim, dim, seq);

        /* LayerScale 1 + residual */
        if (isValid(l.ls1)) {
            opLayerscaleAdd(r, d_hidden, d_normed, l.ls1, seq * dim, dim);
        } else {
            opAdd(r, d_hidden, d_normed, seq * dim);
        }

        /* LN2 -> MLP -> LayerScale 2 + residual */
        opLayerNorm(r, d_normed, d_hidden, l.ln2_w, l.ln2_b, seq, dim);
        opGemm(r, d_mlp, l.fc1_w, d_normed, l.fc1_b, ffn, dim, seq);
        opGelu(r, d_mlp, seq * ffn);
        opGemm(r, d_normed, l.fc2_w, d_mlp, l.fc2_b, dim, ffn, seq);

        if (isValid(l.ls2)) {
            opLayerscaleAdd(r, d_hidden, d_normed, l.ls2, seq * dim, dim);
        } else {
            opAdd(r, d_hidden, d_normed, seq * dim);
        }

        /* Debug: print hidden state after each block */
        if (r->verbose >= 1) {
            float hs[4];
            downloadFromBuffer(r->runner, d_hidden, hs, 4 * sizeof(float));
            fprintf(stderr, "  DINO block %2d: h[0,:4]=%.6f %.6f %.6f %.6f\n",
                    li, hs[0], hs[1], hs[2], hs[3]);
        }

        /* Reset pool periodically to avoid exhaustion */
        if ((li & 7) == 7) r->runner.resetDynamicDescriptorPool();
    }

    /* 4. Final LayerNorm (no affine) */
    opLayerNormNoAffine(r, d_out, d_hidden, seq, dim);

    /* Free scratch buffers */
    r->runner.destroyBuffer(d_hidden);
    r->runner.destroyBuffer(d_Q);
    r->runner.destroyBuffer(d_K);
    r->runner.destroyBuffer(d_V);
    r->runner.destroyBuffer(d_attn);
    r->runner.destroyBuffer(d_mlp);
    r->runner.destroyBuffer(d_normed);
}

/* ======================================================================== */
/* DiT forward pass                                                         */
/* ======================================================================== */

/* Cross-attention KV cache: precomputed K,V for all 30 blocks.
 * Valid when conditioning doesn't change between forward passes. */
struct dit_kv_cache {
    BufInfo K[DIT_DEPTH];  /* [ctx_len, D] per block */
    BufInfo V[DIT_DEPTH];  /* [ctx_len, D] per block */
    bool valid = false;
};

/* Precompute cross-attention KV for all blocks. Allocates GPU buffers. */
static void precomputeKvCache(vulkan_trellis2_runner *r, t2_dit_model_vk &m,
                                BufInfo &d_cond, dit_kv_cache &cache) {
    const int D = DIT_DIM;
    const int H = DIT_HEADS;
    const int HD = DIT_HEAD_DIM;
    const int ctx_len = DINO_SEQ_LEN;
    size_t kv_sz = (size_t)ctx_len * D * sizeof(float);

    /* Temp buffer for fused KV projection */
    BufInfo d_kv_tmp = createGpuBuffer(r->runner, (size_t)ctx_len * 2 * D * sizeof(float));

    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        t2_dit_block_vk &blk = m.blocks[bi];

        cache.K[bi] = createGpuBuffer(r->runner, kv_sz);
        cache.V[bi] = createGpuBuffer(r->runner, kv_sz);

        /* KV projection: [ctx_len, cond_dim] @ W^T + b -> [ctx_len, 2*D] */
        opGemm(r, d_kv_tmp, blk.ca_kv_w, d_cond, blk.ca_kv_b, 2 * D, DIT_COND_DIM, ctx_len);
        opSplitKvChunk(r, cache.K[bi], cache.V[bi], d_kv_tmp, ctx_len, D);

        /* K RMSNorm */
        if (isValid(blk.ca_k_norm))
            opRmsNorm(r, cache.K[bi], blk.ca_k_norm, ctx_len, H, HD, D);

        if ((bi & 7) == 7) r->runner.resetDynamicDescriptorPool();
    }

    r->runner.destroyBuffer(d_kv_tmp);
    cache.valid = true;
}

static void freeKvCache(vulkan_trellis2_runner *r, dit_kv_cache &cache) {
    if (!cache.valid) return;
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        if (isValid(cache.K[bi])) r->runner.destroyBuffer(cache.K[bi]);
        if (isValid(cache.V[bi])) r->runner.destroyBuffer(cache.V[bi]);
    }
    cache.valid = false;
}

/* Generic DiT forward pass: works for Stage 1 (dense) and Stage 2/3 (sparse).
 * N = number of tokens, in_ch/out_ch from model, RoPE tables from model. */
/* kv_cache: if non-null and valid, skip cross-attention KV projection (use cached).
 *           if non-null and !valid, compute and cache (first call). */
static void runDitForwardGeneric(vulkan_trellis2_runner *r,
                                   BufInfo &d_x, float timestep,
                                   BufInfo &d_cond, BufInfo &d_output,
                                   int N, t2_dit_model_vk &m,
                                   BufInfo &rope_cos, BufInfo &rope_sin,
                                   dit_kv_cache *kv_cache = nullptr) {
    const int D = DIT_DIM;          /* 1536 */
    const int H = DIT_HEADS;        /* 12 */
    const int HD = DIT_HEAD_DIM;    /* 128 */
    const int FFN = DIT_FFN;        /* 8192 */
    const int in_ch = m.in_channels;
    const int out_ch = m.out_channels;
    const int ctx_len = DINO_SEQ_LEN; /* 1029 */
    const int n_freqs = m.n_freqs;

    /* Scratch layout */
    size_t qkv_sz = (size_t)3 * N * D * sizeof(float);
    size_t mlp_sz = (size_t)N * FFN * sizeof(float);
    size_t sh1 = std::max(qkv_sz, mlp_sz);
    size_t ca_kv_sz = (size_t)ctx_len * D * sizeof(float);

    ensureScratch(r, 0, (size_t)N * D * sizeof(float));
    ensureScratch(r, 1, sh1);
    ensureScratch(r, 2, (size_t)2 * N * D * sizeof(float));

    BufInfo d_hidden = r->scratch[0];
    BufInfo d_qkv    = r->scratch[1];
    BufInfo d_mlp    = r->scratch[1]; /* alias */
    BufInfo d_attn   = r->scratch[2];

    BufInfo d_normed = createGpuBuffer(r->runner, (size_t)N * D * sizeof(float));
    BufInfo d_mod    = createGpuBuffer(r->runner, (size_t)6 * D * sizeof(float));
    BufInfo d_temb   = createGpuBuffer(r->runner, (size_t)D * sizeof(float));
    BufInfo d_temb_raw = createGpuBuffer(r->runner, 256 * sizeof(float));
    BufInfo d_cross_Q = createGpuBuffer(r->runner, (size_t)N * D * sizeof(float));
    BufInfo d_ca_K   = createGpuBuffer(r->runner, ca_kv_sz);
    BufInfo d_ca_V   = createGpuBuffer(r->runner, ca_kv_sz);
    BufInfo d_split_V = createGpuBuffer(r->runner, (size_t)N * D * sizeof(float));

    /* Pre-allocate modulation sub-buffers (reused across all 30 blocks) */
    BufInfo d_shift_sa  = createGpuBuffer(r->runner, (size_t)D * sizeof(float));
    BufInfo d_scale_sa  = createGpuBuffer(r->runner, (size_t)D * sizeof(float));
    BufInfo d_gate_sa   = createGpuBuffer(r->runner, (size_t)D * sizeof(float));
    BufInfo d_shift_mlp = createGpuBuffer(r->runner, (size_t)D * sizeof(float));
    BufInfo d_scale_mlp = createGpuBuffer(r->runner, (size_t)D * sizeof(float));
    BufInfo d_gate_mlp  = createGpuBuffer(r->runner, (size_t)D * sizeof(float));

    r->runner.resetDynamicDescriptorPool();

    /* 1. Input embedding: [N, in_ch] -> [N, D] */
    opGemm(r, d_hidden, m.x_emb_w, d_x, m.x_emb_b, D, in_ch, N);

    /* 2. Timestep embedding: sinusoidal(256) -> MLP(256->D->D) */
    float t_scaled = timestep * 1000.0f;
    opTimestepEmbedCosSin(r, d_temb_raw, t_scaled, 256);
    opGemm(r, d_normed, m.t_fc1_w, d_temb_raw, m.t_fc1_b, D, 256, 1);
    opSiluInplace(r, d_normed, D);
    opGemm(r, d_temb, m.t_fc2_w, d_normed, m.t_fc2_b, D, D, 1);

    /* 3. Transformer blocks */
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        t2_dit_block_vk &blk = m.blocks[bi];

        /* 3a. Modulation */
        opModulation(r, d_mod, d_temb, m.mod_w, m.mod_b, blk.mod_bias, D, 6 * D);

        /* Copy modulation sub-regions from d_mod into pre-allocated buffers */
        {
            void *mod_ptr = nullptr;
            r->runner.mapBuffer(d_mod, &mod_ptr);
            float *mp = (float *)mod_ptr;

            void *p;
            r->runner.mapBuffer(d_shift_sa, &p);  memcpy(p, mp + 0*D, D*sizeof(float)); r->runner.unmapBuffer(d_shift_sa);
            r->runner.mapBuffer(d_scale_sa, &p);  memcpy(p, mp + 1*D, D*sizeof(float)); r->runner.unmapBuffer(d_scale_sa);
            r->runner.mapBuffer(d_gate_sa, &p);   memcpy(p, mp + 2*D, D*sizeof(float)); r->runner.unmapBuffer(d_gate_sa);
            r->runner.mapBuffer(d_shift_mlp, &p); memcpy(p, mp + 3*D, D*sizeof(float)); r->runner.unmapBuffer(d_shift_mlp);
            r->runner.mapBuffer(d_scale_mlp, &p); memcpy(p, mp + 4*D, D*sizeof(float)); r->runner.unmapBuffer(d_scale_mlp);
            r->runner.mapBuffer(d_gate_mlp, &p);  memcpy(p, mp + 5*D, D*sizeof(float)); r->runner.unmapBuffer(d_gate_mlp);

            r->runner.unmapBuffer(d_mod);
        }

        /* === Self-attention === */
        opAdaLN(r, d_normed, d_hidden, d_shift_sa, d_scale_sa, N, D);
        opGemm(r, d_qkv, blk.sa_qkv_w, d_normed, blk.sa_qkv_b, 3 * D, D, N);
        opSplitQkvChunk(r, d_cross_Q, d_normed, d_split_V, d_qkv, N, D);
        if (isValid(blk.sa_q_norm))
            opRmsNorm(r, d_cross_Q, blk.sa_q_norm, N, H, HD, D);
        if (isValid(blk.sa_k_norm))
            opRmsNorm(r, d_normed, blk.sa_k_norm, N, H, HD, D);
        opRope3D(r, d_cross_Q, rope_cos, rope_sin, N, D, H, HD, n_freqs);
        opRope3D(r, d_normed, rope_cos, rope_sin, N, D, H, HD, n_freqs);
        opSelfAttn(r, d_attn, d_cross_Q, d_normed, d_split_V, N, D, H, HD);
        opGemm(r, d_normed, blk.sa_out_w, d_attn, blk.sa_out_b, D, D, N);
        opGatedAdd(r, d_hidden, d_normed, d_gate_sa, N * D, D);

        /* === Cross-attention === */
        opLayerNorm(r, d_normed, d_hidden, blk.norm2_w, blk.norm2_b, N, D);
        opGemm(r, d_cross_Q, blk.ca_q_w, d_normed, blk.ca_q_b, D, D, N);
        if (isValid(blk.ca_q_norm))
            opRmsNorm(r, d_cross_Q, blk.ca_q_norm, N, H, HD, D);

        /* KV: use cache if available, otherwise compute from conditioning */
        BufInfo &use_K = (kv_cache && kv_cache->valid) ? kv_cache->K[bi] : d_ca_K;
        BufInfo &use_V = (kv_cache && kv_cache->valid) ? kv_cache->V[bi] : d_ca_V;
        if (!kv_cache || !kv_cache->valid) {
            opGemm(r, d_qkv, blk.ca_kv_w, d_cond, blk.ca_kv_b, 2 * D, DIT_COND_DIM, ctx_len);
            opSplitKvChunk(r, d_ca_K, d_ca_V, d_qkv, ctx_len, D);
            if (isValid(blk.ca_k_norm))
                opRmsNorm(r, d_ca_K, blk.ca_k_norm, ctx_len, H, HD, D);
        }
        opCrossAttn(r, d_attn, d_cross_Q, use_K, use_V, N, ctx_len, D, H, HD);
        opGemm(r, d_normed, blk.ca_out_w, d_attn, blk.ca_out_b, D, D, N);
        opAdd(r, d_hidden, d_normed, N * D);

        /* === MLP === */
        opAdaLN(r, d_normed, d_hidden, d_shift_mlp, d_scale_mlp, N, D);
        opGemm(r, d_mlp, blk.mlp_fc1_w, d_normed, blk.mlp_fc1_b, FFN, D, N);
        opGelu(r, d_mlp, N * FFN);
        opGemm(r, d_normed, blk.mlp_fc2_w, d_mlp, blk.mlp_fc2_b, D, FFN, N);
        opGatedAdd(r, d_hidden, d_normed, d_gate_mlp, N * D, D);

        if (r->verbose >= 1 && (bi == 0 || bi == DIT_DEPTH - 1)) {
            float hs[4];
            downloadFromBuffer(r->runner, d_hidden, hs, 4 * sizeof(float));
            fprintf(stderr, "  T2VK block %2d: h[:4]=%.4f %.4f %.4f %.4f\n",
                    bi, hs[0], hs[1], hs[2], hs[3]);
        }

        /* Reset pool periodically */
        if ((bi & 7) == 7) r->runner.resetDynamicDescriptorPool();
    }

    /* 4. Final LayerNorm (no affine) + output projection */
    opLayerNormNoAffine(r, d_hidden, d_hidden, N, D);
    opGemm(r, d_output, m.out_w, d_hidden, m.out_b, out_ch, D, N);

    r->runner.destroyBuffer(d_normed);
    r->runner.destroyBuffer(d_mod);
    r->runner.destroyBuffer(d_temb);
    r->runner.destroyBuffer(d_temb_raw);
    r->runner.destroyBuffer(d_cross_Q);
    r->runner.destroyBuffer(d_ca_K);
    r->runner.destroyBuffer(d_ca_V);
    r->runner.destroyBuffer(d_split_V);
    r->runner.destroyBuffer(d_shift_sa);
    r->runner.destroyBuffer(d_scale_sa);
    r->runner.destroyBuffer(d_gate_sa);
    r->runner.destroyBuffer(d_shift_mlp);
    r->runner.destroyBuffer(d_scale_mlp);
    r->runner.destroyBuffer(d_gate_mlp);
}

/* Stage 1 wrapper: fixed N=4096, precomputed RoPE */
static void runDitForward(vulkan_trellis2_runner *r,
                            BufInfo &d_x, float timestep,
                            BufInfo &d_cond, BufInfo &d_output,
                            dit_kv_cache *kv_cache = nullptr) {
    runDitForwardGeneric(r, d_x, timestep, d_cond, d_output,
                         DIT_N_TOKENS, r->stage1_dit,
                         r->stage1_dit.rope_cos, r->stage1_dit.rope_sin,
                         kv_cache);
}

/* ======================================================================== */
/* Decoder forward                                                          */
/* ======================================================================== */

static void runResblock(vulkan_trellis2_runner *r,
                          BufInfo &d_out, BufInfo &d_in,
                          t2_dec_resblock_vk &rb,
                          int C, int D, int H, int W) {
    int spatial = D * H * W;

    ensureScratch(r, 4, (size_t)C * spatial * sizeof(float));
    ensureScratch(r, 5, (size_t)C * spatial * sizeof(float));
    BufInfo d_h1 = r->scratch[4];
    BufInfo d_h2 = r->scratch[5];

    /* ChannelLN1 -> SiLU -> Conv1 */
    opChannelLayerNorm3D(r, d_h1, d_in, rb.gn1_w, rb.gn1_b, C, spatial);
    opSiluInplace(r, d_h1, C * spatial);
    opConv3d(r, d_h2, d_h1, rb.conv1_w, rb.conv1_b, C, C, D, H, W);

    /* ChannelLN2 -> SiLU -> Conv2 */
    opChannelLayerNorm3D(r, d_h1, d_h2, rb.gn2_w, rb.gn2_b, C, spatial);
    opSiluInplace(r, d_h1, C * spatial);
    opConv3d(r, d_out, d_h1, rb.conv2_w, rb.conv2_b, C, C, D, H, W);

    /* Skip (identity) */
    opAdd(r, d_out, d_in, C * spatial);
}

static void runDecoder(vulkan_trellis2_runner *r,
                         BufInfo &d_latent, BufInfo &d_output) {
    if (r->verbose) fprintf(stderr, "T2VK: decoder start\n");
    r->runner.resetDynamicDescriptorPool();

    /* Alloc main decoder buffers */
    ensureScratch(r, 6, (size_t)1024 * 16 * 16 * 16 * sizeof(float));
    ensureScratch(r, 7, (size_t)512 * 32 * 32 * 32 * sizeof(float));

    BufInfo d_buf_a = r->scratch[6];
    BufInfo d_buf_b = r->scratch[7];

    /* conv_in: [8, 16^3] -> [512, 16^3] */
    opConv3d(r, d_buf_a, d_latent, r->dec_conv_in_w, r->dec_conv_in_b, 8, 512, 16, 16, 16);

    /* middle + res16 blocks (512 ch, 16^3) */
    runResblock(r, d_buf_b, d_buf_a, r->dec_middle[0], 512, 16, 16, 16);
    runResblock(r, d_buf_a, d_buf_b, r->dec_middle[1], 512, 16, 16, 16);
    runResblock(r, d_buf_b, d_buf_a, r->dec_res16[0], 512, 16, 16, 16);
    runResblock(r, d_buf_a, d_buf_b, r->dec_res16[1], 512, 16, 16, 16);

    /* Up1: conv 512->1024, pixel_shuffle -> [128, 32^3] */
    opConv3d(r, d_buf_b, d_buf_a, r->dec_up1_w, r->dec_up1_b, 512, 1024, 16, 16, 16);
    opPixelShuffle3D(r, d_buf_a, d_buf_b, 128, 16, 16, 16);

    /* res32 blocks (128 ch, 32^3) */
    runResblock(r, d_buf_b, d_buf_a, r->dec_res32[0], 128, 32, 32, 32);
    runResblock(r, d_buf_a, d_buf_b, r->dec_res32[1], 128, 32, 32, 32);

    /* Up2: conv 128->256, pixel_shuffle -> [32, 64^3] */
    opConv3d(r, d_buf_b, d_buf_a, r->dec_up2_w, r->dec_up2_b, 128, 256, 32, 32, 32);

    /* Need bigger buffer for 64^3 */
    ensureScratch(r, 6, (size_t)32 * 64 * 64 * 64 * sizeof(float));
    d_buf_a = r->scratch[6];
    opPixelShuffle3D(r, d_buf_a, d_buf_b, 32, 32, 32, 32);

    /* res64 blocks (32 ch, 64^3) */
    ensureScratch(r, 7, (size_t)32 * 64 * 64 * 64 * sizeof(float));
    d_buf_b = r->scratch[7];
    runResblock(r, d_buf_b, d_buf_a, r->dec_res64[0], 32, 64, 64, 64);
    runResblock(r, d_buf_a, d_buf_b, r->dec_res64[1], 32, 64, 64, 64);

    /* Output: ChannelLN -> SiLU -> Conv3d(32->1) */
    opChannelLayerNorm3D(r, d_buf_b, d_buf_a, r->dec_out_gn_w, r->dec_out_gn_b, 32, 64*64*64);
    opSiluInplace(r, d_buf_b, 32 * 64 * 64 * 64);
    opConv3d(r, d_output, d_buf_b, r->dec_out_conv_w, r->dec_out_conv_b, 32, 1, 64, 64, 64);
}

/* ======================================================================== */
/* Sampling utilities                                                       */
/* ======================================================================== */

static uint64_t t2_rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; } t2_rng;
static uint64_t t2_next(t2_rng *rng) {
    uint64_t *s = rng->s;
    uint64_t result = t2_rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = t2_rotl(s[3], 45);
    return result;
}
static float t2_randn(t2_rng *rng) {
    double u1 = ((double)(t2_next(rng) >> 11) + 0.5) / (double)(1ULL << 53);
    double u2 = ((double)(t2_next(rng) >> 11) + 0.5) / (double)(1ULL << 53);
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}
static float t2_rescale(float t, float rescale_t) {
    return t * rescale_t / (1.0f + (rescale_t - 1.0f) * t);
}

/* ======================================================================== */
/* Public API                                                               */
/* ======================================================================== */

vulkan_trellis2_runner *vulkan_trellis2_init(int device_id, int verbose) {
    if (!vl_cpp::vulkan::InitializeVulkan()) {
        fprintf(stderr, "T2VK: failed to load Vulkan\n");
        return nullptr;
    }

    auto *r = new vulkan_trellis2_runner();
    memset(r->scratch_size, 0, sizeof(r->scratch_size));
    for (int i = 0; i < 12; i++) r->scratch[i] = {};
    r->verbose = verbose;

    if (!r->runner.initialize(false)) {
        fprintf(stderr, "T2VK: failed to initialize Vulkan runner: %s\n",
                r->runner.getLastError().c_str());
        delete r;
        return nullptr;
    }
    if ((uint32_t)device_id >= r->runner.getDeviceCount()) {
        fprintf(stderr, "T2VK: invalid device ID %d (have %u devices)\n",
                device_id, r->runner.getDeviceCount());
        delete r;
        return nullptr;
    }
    if (!r->runner.selectDevice(device_id)) {
        fprintf(stderr, "T2VK: failed to select device: %s\n",
                r->runner.getLastError().c_str());
        delete r;
        return nullptr;
    }
    if (verbose)
        fprintf(stderr, "T2VK: using device: %s\n", r->runner.getDeviceName(device_id).c_str());

    /* Create dynamic descriptor pool for dispatch operations */
    if (!r->runner.createDynamicDescriptorPool(16384)) {
        fprintf(stderr, "T2VK: failed to create descriptor pool\n");
        delete r;
        return nullptr;
    }

    /* Determine shader directory */
    const char *sd = getenv("TRELLIS2_SHADER_DIR");
    if (sd) {
        r->shader_dir = sd;
    } else {
        r->shader_dir = ".";
    }
    fprintf(stderr, "T2VK: shader dir: %s\n", r->shader_dir.c_str());

    if (!createPipelines(r)) {
        fprintf(stderr, "T2VK: failed to create compute pipelines\n");
        delete r;
        return nullptr;
    }

    r->initialized = true;
    fprintf(stderr, "T2VK: initialized on device %d\n", device_id);
    return r;
}

int vulkan_trellis2_load_weights(vulkan_trellis2_runner *r,
                                  const char *dinov3_path,
                                  const char *dit_path,
                                  const char *decoder_path) {
    if (dinov3_path && loadDinov3Weights(r, dinov3_path) != 0) return -1;
    if (dit_path && loadDitWeights(r, dit_path) != 0) return -1;
    if (decoder_path && loadDecoderWeights(r, decoder_path) != 0) return -1;
    return 0;
}

int vulkan_trellis2_run_dinov3(vulkan_trellis2_runner *r,
                                const float *image_f32,
                                float *output) {
    const int seq = DINO_SEQ_LEN;
    const int dim = DINO_HIDDEN;
    const int S = DINO_IMG_SIZE;

    /* Convert image from CHW [3, 512, 512] to HWC [512, 512, 3] for patch_embed shader */
    std::vector<float> hwc(3 * S * S);
    for (int y = 0; y < S; y++)
        for (int x = 0; x < S; x++)
            for (int c = 0; c < 3; c++)
                hwc[(y * S + x) * 3 + c] = image_f32[c * S * S + y * S + x];

    BufInfo d_image = createGpuBuffer(r->runner, (size_t)3 * S * S * sizeof(float));
    uploadToBuffer(r->runner, d_image, hwc.data(), (size_t)3 * S * S * sizeof(float));

    BufInfo d_out = createGpuBuffer(r->runner, (size_t)seq * dim * sizeof(float));

    runDinov3(r, d_image, d_out);

    downloadFromBuffer(r->runner, d_out, output, (size_t)seq * dim * sizeof(float));

    r->runner.destroyBuffer(d_image);
    r->runner.destroyBuffer(d_out);
    return 0;
}

int vulkan_trellis2_run_dit(vulkan_trellis2_runner *r,
                             const float *x_t, float timestep,
                             const float *features, float *output) {
    const int N = DIT_N_TOKENS, in_ch = DIT_IN_CH;
    const int ctx_len = DINO_SEQ_LEN;

    /* Transpose input from NCDHW [C, spatial] to [spatial, C] */
    std::vector<float> x_transposed(N * in_ch);
    for (int pos = 0; pos < N; pos++)
        for (int ch = 0; ch < in_ch; ch++)
            x_transposed[pos * in_ch + ch] = x_t[ch * N + pos];

    BufInfo d_x = createGpuBuffer(r->runner, (size_t)N * in_ch * sizeof(float));
    uploadToBuffer(r->runner, d_x, x_transposed.data(), N * in_ch * sizeof(float));

    BufInfo d_cond = createGpuBuffer(r->runner, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    uploadToBuffer(r->runner, d_cond, features, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));

    BufInfo d_out = createGpuBuffer(r->runner, (size_t)N * in_ch * sizeof(float));

    runDitForward(r, d_x, timestep, d_cond, d_out);

    /* Transpose output from [spatial, C] back to [C, spatial] NCDHW */
    std::vector<float> out_flat(N * in_ch);
    downloadFromBuffer(r->runner, d_out, out_flat.data(), N * in_ch * sizeof(float));
    for (int pos = 0; pos < N; pos++)
        for (int ch = 0; ch < in_ch; ch++)
            output[ch * N + pos] = out_flat[pos * in_ch + ch];

    r->runner.destroyBuffer(d_x);
    r->runner.destroyBuffer(d_cond);
    r->runner.destroyBuffer(d_out);
    return 0;
}

int vulkan_trellis2_run_decoder(vulkan_trellis2_runner *r,
                                 const float *latent, float *output) {
    BufInfo d_lat = createGpuBuffer(r->runner, (size_t)8 * 16 * 16 * 16 * sizeof(float));
    uploadToBuffer(r->runner, d_lat, latent, (size_t)8 * 16 * 16 * 16 * sizeof(float));

    BufInfo d_out = createGpuBuffer(r->runner, (size_t)64 * 64 * 64 * sizeof(float));

    runDecoder(r, d_lat, d_out);

    downloadFromBuffer(r->runner, d_out, output, (size_t)64 * 64 * 64 * sizeof(float));

    r->runner.destroyBuffer(d_lat);
    r->runner.destroyBuffer(d_out);
    return 0;
}

int vulkan_trellis2_run_stage1(vulkan_trellis2_runner *r,
                                const float *features,
                                const float *initial_noise,
                                float *output,
                                int n_steps,
                                float cfg_scale,
                                float cfg_rescale,
                                uint32_t seed) {
    const int N = DIT_N_TOKENS, in_ch = DIT_IN_CH;
    const int ctx_len = DINO_SEQ_LEN;
    const int n_elem = N * in_ch;
    const float rescale_t_val = 5.0f;
    const float guidance_min = 0.6f, guidance_max = 1.0f;
    const float sigma_min = 1e-5f;

    /* Initialize noise */
    std::vector<float> x(n_elem);
    if (initial_noise) {
        memcpy(x.data(), initial_noise, n_elem * sizeof(float));
    } else {
        t2_rng rng;
        rng.s[0] = seed; rng.s[1] = seed ^ 0x1234567890abcdefULL;
        rng.s[2] = seed ^ 0xfedcba0987654321ULL; rng.s[3] = seed ^ 0xdeadbeefcafebabeULL;
        for (int i = 0; i < 100; i++) t2_next(&rng); /* warm up */
        for (int i = 0; i < n_elem; i++) x[i] = t2_randn(&rng);
    }

    /* Upload conditioning features */
    BufInfo d_cond = createGpuBuffer(r->runner, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    uploadToBuffer(r->runner, d_cond, features, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));

    /* Zero conditioning for unconditioned pass */
    BufInfo d_zero_cond = createGpuBuffer(r->runner, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    bufferZero(r->runner, d_zero_cond, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));

    /* Precompute cross-attention KV cache */
    dit_kv_cache cond_kv = {}, uncond_kv = {};
    precomputeKvCache(r, r->stage1_dit, d_cond, cond_kv);
    precomputeKvCache(r, r->stage1_dit, d_zero_cond, uncond_kv);

    /* Pre-allocate GPU buffers (reused across steps) */
    BufInfo d_x = createGpuBuffer(r->runner, n_elem * sizeof(float));
    BufInfo d_v_out = createGpuBuffer(r->runner, n_elem * sizeof(float));

    std::vector<float> v_cond(n_elem), v_uncond(n_elem), pred_v(n_elem);

    /* Euler sampling loop */
    for (int step = 0; step < n_steps; step++) {
        float t_start = 1.0f - (float)step / (float)n_steps;
        float t_end = 1.0f - (float)(step + 1) / (float)n_steps;
        float t_cur = t2_rescale(t_start, rescale_t_val);
        float t_next = t2_rescale(t_end, rescale_t_val);

        bool apply_cfg = (t_cur >= guidance_min && t_cur <= guidance_max && cfg_scale != 1.0f);

        if (apply_cfg) {
            /* Transpose x to token-major for GPU */
            std::vector<float> x_tok(n_elem);
            for (int pos = 0; pos < N; pos++)
                for (int ch = 0; ch < in_ch; ch++)
                    x_tok[pos * in_ch + ch] = x[ch * N + pos];

            uploadToBuffer(r->runner, d_x, x_tok.data(), n_elem * sizeof(float));

            /* Conditioned forward (with KV cache) */
            runDitForward(r, d_x, t_cur, d_cond, d_v_out, &cond_kv);

            std::vector<float> v_tok(n_elem);
            downloadFromBuffer(r->runner, d_v_out, v_tok.data(), n_elem * sizeof(float));
            for (int pos = 0; pos < N; pos++)
                for (int ch = 0; ch < in_ch; ch++)
                    v_cond[ch * N + pos] = v_tok[pos * in_ch + ch];

            /* Unconditioned forward (with KV cache) */
            uploadToBuffer(r->runner, d_x, x_tok.data(), n_elem * sizeof(float));
            runDitForward(r, d_x, t_cur, d_zero_cond, d_v_out, &uncond_kv);

            downloadFromBuffer(r->runner, d_v_out, v_tok.data(), n_elem * sizeof(float));
            for (int pos = 0; pos < N; pos++)
                for (int ch = 0; ch < in_ch; ch++)
                    v_uncond[ch * N + pos] = v_tok[pos * in_ch + ch];

            /* CFG combination */
            for (int i = 0; i < n_elem; i++)
                pred_v[i] = cfg_scale * v_cond[i] + (1.0f - cfg_scale) * v_uncond[i];

            /* CFG rescaling */
            if (cfg_rescale > 0.0f) {
                float tc = sigma_min + (1.0f - sigma_min) * t_cur;
                float one_m_sm = 1.0f - sigma_min;

                double sum_pos = 0, sum2_pos = 0;
                double sum_cfg = 0, sum2_cfg = 0;
                for (int i = 0; i < n_elem; i++) {
                    float x0p = one_m_sm * x[i] - tc * v_cond[i];
                    float x0c = one_m_sm * x[i] - tc * pred_v[i];
                    sum_pos += x0p; sum2_pos += (double)x0p * x0p;
                    sum_cfg += x0c; sum2_cfg += (double)x0c * x0c;
                }
                double n_d = (double)n_elem;
                double std_pos = sqrt((sum2_pos - sum_pos * sum_pos / n_d) / (n_d - 1.0));
                double std_cfg = sqrt((sum2_cfg - sum_cfg * sum_cfg / n_d) / (n_d - 1.0));

                float ratio = (std_cfg > 1e-8) ? (float)(std_pos / std_cfg) : 1.0f;
                float sc = cfg_rescale * ratio + (1.0f - cfg_rescale);

                for (int i = 0; i < n_elem; i++) {
                    float x0c = one_m_sm * x[i] - tc * pred_v[i];
                    pred_v[i] = (one_m_sm * x[i] - sc * x0c) / tc;
                }
            }

            /* Euler step */
            float dt = t_cur - t_next;
            for (int i = 0; i < n_elem; i++)
                x[i] -= dt * pred_v[i];
        } else {
            /* No CFG: just conditioned prediction */
            std::vector<float> x_tok(n_elem);
            for (int pos = 0; pos < N; pos++)
                for (int ch = 0; ch < in_ch; ch++)
                    x_tok[pos * in_ch + ch] = x[ch * N + pos];

            uploadToBuffer(r->runner, d_x, x_tok.data(), n_elem * sizeof(float));
            runDitForward(r, d_x, t_cur, d_cond, d_v_out, &cond_kv);

            std::vector<float> v_tok(n_elem);
            downloadFromBuffer(r->runner, d_v_out, v_tok.data(), n_elem * sizeof(float));
            for (int pos = 0; pos < N; pos++)
                for (int ch = 0; ch < in_ch; ch++)
                    v_cond[ch * N + pos] = v_tok[pos * in_ch + ch];

            float dt = t_cur - t_next;
            for (int i = 0; i < n_elem; i++)
                x[i] -= dt * v_cond[i];
        }

        if (r->verbose >= 1)
            fprintf(stderr, "T2VK: step %d/%d (t=%.4f->%.4f, cfg=%s)\n",
                    step + 1, n_steps, t_cur, t_next, apply_cfg ? "yes" : "no");
    }

    /* Run decoder */
    std::vector<float> occ(64 * 64 * 64);
    BufInfo d_lat = createGpuBuffer(r->runner, n_elem * sizeof(float));
    uploadToBuffer(r->runner, d_lat, x.data(), n_elem * sizeof(float));
    BufInfo d_occ = createGpuBuffer(r->runner, 64 * 64 * 64 * sizeof(float));

    runDecoder(r, d_lat, d_occ);

    downloadFromBuffer(r->runner, d_occ, output, 64 * 64 * 64 * sizeof(float));

    r->runner.destroyBuffer(d_lat);
    r->runner.destroyBuffer(d_occ);
    r->runner.destroyBuffer(d_x);
    r->runner.destroyBuffer(d_v_out);
    r->runner.destroyBuffer(d_cond);
    r->runner.destroyBuffer(d_zero_cond);
    freeKvCache(r, cond_kv);
    freeKvCache(r, uncond_kv);

    return 0;
}

/* ======================================================================== */
/* Stage 2/3 API                                                            */
/* ======================================================================== */

int vulkan_trellis2_load_stage2(vulkan_trellis2_runner *r, const char *path) {
    return loadDitModelWeights(r, path, &r->stage2_dit, "Stage 2");
}

int vulkan_trellis2_load_stage3(vulkan_trellis2_runner *r, const char *path) {
    return loadDitModelWeights(r, path, &r->stage3_dit, "Stage 3");
}

int vulkan_trellis2_run_stage2_dit(vulkan_trellis2_runner *r,
                                     const float *x_t, float timestep,
                                     const float *cond_features,
                                     const int32_t *coords, int N,
                                     float *output) {
    if (!r->stage2_dit.loaded) {
        fprintf(stderr, "T2VK: Stage 2 weights not loaded\n");
        return -1;
    }

    const int in_ch = r->stage2_dit.in_channels;
    const int out_ch = r->stage2_dit.out_channels;
    const int ctx_len = DINO_SEQ_LEN;
    const int n_freqs = r->stage2_dit.n_freqs;

    /* Compute RoPE from sparse coords */
    BufInfo rope_cos, rope_sin;
    computeSparseRope(r, coords, N, n_freqs, rope_cos, rope_sin);

    /* Upload x_t (already token-major [N, in_ch] — no transpose needed for sparse) */
    BufInfo d_x = uploadF32(r->runner, x_t, (size_t)N * in_ch);
    BufInfo d_cond = uploadF32(r->runner, cond_features, (size_t)ctx_len * DIT_COND_DIM);
    BufInfo d_out = createGpuBuffer(r->runner, (size_t)N * out_ch * sizeof(float));

    runDitForwardGeneric(r, d_x, timestep, d_cond, d_out,
                         N, r->stage2_dit, rope_cos, rope_sin);

    downloadFromBuffer(r->runner, d_out, output, (size_t)N * out_ch * sizeof(float));

    r->runner.destroyBuffer(d_x);
    r->runner.destroyBuffer(d_cond);
    r->runner.destroyBuffer(d_out);
    r->runner.destroyBuffer(rope_cos);
    r->runner.destroyBuffer(rope_sin);
    return 0;
}

int vulkan_trellis2_run_stage3_dit(vulkan_trellis2_runner *r,
                                     const float *x_t, float timestep,
                                     const float *cond_features,
                                     const int32_t *coords, int N,
                                     float *output) {
    if (!r->stage3_dit.loaded) {
        fprintf(stderr, "T2VK: Stage 3 weights not loaded\n");
        return -1;
    }

    const int in_ch = r->stage3_dit.in_channels;
    const int out_ch = r->stage3_dit.out_channels;
    const int ctx_len = DINO_SEQ_LEN;
    const int n_freqs = r->stage3_dit.n_freqs;

    BufInfo rope_cos, rope_sin;
    computeSparseRope(r, coords, N, n_freqs, rope_cos, rope_sin);

    BufInfo d_x = uploadF32(r->runner, x_t, (size_t)N * in_ch);
    BufInfo d_cond = uploadF32(r->runner, cond_features, (size_t)ctx_len * DIT_COND_DIM);
    BufInfo d_out = createGpuBuffer(r->runner, (size_t)N * out_ch * sizeof(float));

    runDitForwardGeneric(r, d_x, timestep, d_cond, d_out,
                         N, r->stage3_dit, rope_cos, rope_sin);

    downloadFromBuffer(r->runner, d_out, output, (size_t)N * out_ch * sizeof(float));

    r->runner.destroyBuffer(d_x);
    r->runner.destroyBuffer(d_cond);
    r->runner.destroyBuffer(d_out);
    r->runner.destroyBuffer(rope_cos);
    r->runner.destroyBuffer(rope_sin);
    return 0;
}

/* ======================================================================== */
/* Sparse conv op (for shape decoder GPU acceleration)                      */
/* ======================================================================== */

static void opSparseConv3d(vulkan_trellis2_runner *r,
                            BufInfo &out, BufInfo &feats,
                            BufInfo &w, BufInfo &bias,
                            BufInfo &hash_keys, BufInfo &hash_vals,
                            BufInfo &coords, int N, int Ci, int Co, int hash_mask) {
    struct { uint32_t N, Ci, Co, hash_mask; } pc = {
        (uint32_t)N, (uint32_t)Ci, (uint32_t)Co, (uint32_t)hash_mask
    };
    uint32_t gx = (uint32_t)((Co + 31) / 32);  /* TILE_CO = 32 */
    uint32_t gy = (uint32_t)N;
    dispatchSync(r, r->pipe_sparse_conv3d, {out, feats, w, bias, hash_keys, hash_vals, coords},
                 &pc, sizeof(pc), gx, gy);
}

/* ======================================================================== */
/* Shape decoder (GPU ConvNeXt + CPU C2S subdivision)                       */
/* ======================================================================== */

/* Build hash table on CPU, upload keys+vals to GPU */
static void buildAndUploadHash(vulkan_trellis2_runner *r,
                                const int32_t *coords, int N,
                                BufInfo &d_keys, BufInfo &d_vals, int *out_mask) {
    sp3d_hash *h = sp3d_hash_build(coords, N);
    int cap = h->capacity;
    *out_mask = cap - 1;

    d_keys = createGpuBuffer(r->runner, (size_t)cap * sizeof(uint64_t));
    uploadToBuffer(r->runner, d_keys, h->keys, (size_t)cap * sizeof(uint64_t));
    d_vals = createGpuBuffer(r->runner, (size_t)cap * sizeof(int32_t));
    uploadToBuffer(r->runner, d_vals, h->vals, (size_t)cap * sizeof(int32_t));

    sp3d_hash_free(h);
}

/* Run one ConvNeXt block on GPU:
 *   sparse_conv3d → LayerNorm → GEMM(C→4C) → GELU → GEMM(4C→C) → residual add */
static void runConvNeXtBlockGpu(vulkan_trellis2_runner *r,
                                  BufInfo &d_feats, BufInfo &d_tmp, BufInfo &d_normed,
                                  BufInfo &d_mlp,
                                  BufInfo &d_conv_w, BufInfo &d_conv_b,
                                  BufInfo &d_norm_w, BufInfo &d_norm_b,
                                  BufInfo &d_mlp0_w, BufInfo &d_mlp0_b,
                                  BufInfo &d_mlp2_w, BufInfo &d_mlp2_b,
                                  BufInfo &d_hash_keys, BufInfo &d_hash_vals,
                                  BufInfo &d_coords,
                                  int N, int C, int hash_mask) {
    /* sparse conv: feats → tmp */
    opSparseConv3d(r, d_tmp, d_feats, d_conv_w, d_conv_b,
                   d_hash_keys, d_hash_vals, d_coords, N, C, C, hash_mask);

    /* layernorm: tmp → normed */
    opLayerNorm(r, d_normed, d_tmp, d_norm_w, d_norm_b, N, C);

    /* mlp: normed → 4C → C */
    opGemm(r, d_mlp, d_mlp0_w, d_normed, d_mlp0_b, 4 * C, C, N);
    opGelu(r, d_mlp, N * 4 * C);
    opGemm(r, d_tmp, d_mlp2_w, d_mlp, d_mlp2_b, C, 4 * C, N);

    /* residual: feats += tmp */
    opAdd(r, d_feats, d_tmp, N * C);
}

int vulkan_trellis2_run_shape_decoder(vulkan_trellis2_runner *r,
                                        const char *dec_path,
                                        const float *slat,
                                        const int32_t *coords, int N,
                                        float *out_feats,
                                        int32_t *out_coords,
                                        int *out_N) {
    /* Load decoder weights (CPU struct — we'll upload per-block to GPU) */
    t2_shape_dec *dec = t2_shape_dec_load(dec_path);
    if (!dec) {
        fprintf(stderr, "T2VK: failed to load shape decoder from %s\n", dec_path);
        return -1;
    }

    fprintf(stderr, "T2VK: shape decoder (GPU ConvNeXt + CPU C2S), N=%d\n", N);
    r->runner.resetDynamicDescriptorPool();

    int C = dec->channels[0]; /* 1024 */

    /* from_latent: Linear(32 → 1024) on GPU */
    BufInfo d_slat = uploadF32(r->runner, slat, (size_t)N * 32);
    BufInfo d_from_w = uploadF32(r->runner, dec->from_latent_w, (size_t)C * 32);
    BufInfo d_from_b = uploadF32(r->runner, dec->from_latent_b, C);
    BufInfo d_feats = createGpuBuffer(r->runner, (size_t)N * C * sizeof(float));

    opGemm(r, d_feats, d_from_w, d_slat, d_from_b, C, 32, N);
    r->runner.destroyBuffer(d_slat);
    r->runner.destroyBuffer(d_from_w);
    r->runner.destroyBuffer(d_from_b);

    /* Upload initial coords + build hash */
    BufInfo d_coords = createGpuBuffer(r->runner, (size_t)N * 4 * sizeof(int32_t));
    uploadToBuffer(r->runner, d_coords, coords, (size_t)N * 4 * sizeof(int32_t));
    BufInfo d_hash_keys, d_hash_vals;
    int hash_mask;
    buildAndUploadHash(r, coords, N, d_hash_keys, d_hash_vals, &hash_mask);

    /* Working copy of coords/feats for CPU C2S */
    std::vector<int32_t> cur_coords(coords, coords + N * 4);
    int cur_N = N;

    /* Process stages */
    for (int stage = 0; stage < dec->n_stages; stage++) {
        int nc = dec->n_convnext[stage];
        int ch = dec->channels[stage];

        fprintf(stderr, "  stage %d: %d ConvNeXt(%d), N=%d\n", stage, nc, ch, cur_N);

        /* Scratch buffers for this stage */
        BufInfo d_tmp    = createGpuBuffer(r->runner, (size_t)cur_N * ch * sizeof(float));
        BufInfo d_normed = createGpuBuffer(r->runner, (size_t)cur_N * ch * sizeof(float));
        BufInfo d_mlp    = createGpuBuffer(r->runner, (size_t)cur_N * 4 * ch * sizeof(float));

        /* Run ConvNeXt blocks on GPU */
        for (int b = 0; b < nc; b++) {
            t2sd_convnext &blk = dec->convnext[stage][b];

            /* Upload block weights */
            BufInfo d_cw  = uploadF32(r->runner, blk.conv_w, (size_t)ch * 27 * ch);
            BufInfo d_cb  = uploadF32(r->runner, blk.conv_b, ch);
            BufInfo d_nw  = uploadF32(r->runner, blk.norm_w, ch);
            BufInfo d_nb  = uploadF32(r->runner, blk.norm_b, ch);
            BufInfo d_m0w = uploadF32(r->runner, blk.mlp0_w, (size_t)4 * ch * ch);
            BufInfo d_m0b = uploadF32(r->runner, blk.mlp0_b, 4 * ch);
            BufInfo d_m2w = uploadF32(r->runner, blk.mlp2_w, (size_t)ch * 4 * ch);
            BufInfo d_m2b = uploadF32(r->runner, blk.mlp2_b, ch);

            runConvNeXtBlockGpu(r, d_feats, d_tmp, d_normed, d_mlp,
                                d_cw, d_cb, d_nw, d_nb,
                                d_m0w, d_m0b, d_m2w, d_m2b,
                                d_hash_keys, d_hash_vals, d_coords,
                                cur_N, ch, hash_mask);

            /* Free block weights */
            r->runner.destroyBuffer(d_cw);  r->runner.destroyBuffer(d_cb);
            r->runner.destroyBuffer(d_nw);  r->runner.destroyBuffer(d_nb);
            r->runner.destroyBuffer(d_m0w); r->runner.destroyBuffer(d_m0b);
            r->runner.destroyBuffer(d_m2w); r->runner.destroyBuffer(d_m2b);

            if ((b & 3) == 3) r->runner.resetDynamicDescriptorPool();
        }

        r->runner.destroyBuffer(d_tmp);
        r->runner.destroyBuffer(d_normed);
        r->runner.destroyBuffer(d_mlp);

        /* C2S subdivision (CPU) */
        if (dec->c2s[stage].conv1_w) {
            /* Download features from GPU */
            std::vector<float> cpu_feats(cur_N * ch);
            downloadFromBuffer(r->runner, d_feats, cpu_feats.data(),
                              (size_t)cur_N * ch * sizeof(float));

            /* Build CPU sparse tensor for C2S */
            sp3d_tensor *t_cpu = sp3d_create(cur_coords.data(), cpu_feats.data(),
                                              cur_N, ch, 1);

            /* Run C2S on CPU */
            sp3d_tensor *t_new = t2sd_c2s_forward(t_cpu, &dec->c2s[stage], 4);
            sp3d_free(t_cpu);

            int new_N = t_new->N;
            int new_C = t_new->C;
            fprintf(stderr, "    c2s -> N=%d, C=%d\n", new_N, new_C);

            /* Free old GPU buffers */
            r->runner.destroyBuffer(d_feats);
            r->runner.destroyBuffer(d_coords);
            r->runner.destroyBuffer(d_hash_keys);
            r->runner.destroyBuffer(d_hash_vals);

            /* Upload new data to GPU */
            d_feats = uploadF32(r->runner, t_new->feats, (size_t)new_N * new_C);
            d_coords = createGpuBuffer(r->runner, (size_t)new_N * 4 * sizeof(int32_t));
            uploadToBuffer(r->runner, d_coords, t_new->coords,
                          (size_t)new_N * 4 * sizeof(int32_t));
            buildAndUploadHash(r, t_new->coords, new_N,
                              d_hash_keys, d_hash_vals, &hash_mask);

            /* Update CPU state */
            cur_coords.assign(t_new->coords, t_new->coords + new_N * 4);
            cur_N = new_N;

            sp3d_free(t_new);
        }

        r->runner.resetDynamicDescriptorPool();
    }

    /* output_layer: Linear(64 → out_channels) on GPU */
    int out_ch = dec->out_channels;
    BufInfo d_out_w = uploadF32(r->runner, dec->output_w, (size_t)out_ch * dec->channels[dec->n_stages]);
    BufInfo d_out_b = uploadF32(r->runner, dec->output_b, out_ch);
    BufInfo d_output = createGpuBuffer(r->runner, (size_t)cur_N * out_ch * sizeof(float));

    opGemm(r, d_output, d_out_w, d_feats, d_out_b, out_ch, dec->channels[dec->n_stages], cur_N);

    /* Download results */
    downloadFromBuffer(r->runner, d_output, out_feats,
                      (size_t)cur_N * out_ch * sizeof(float));
    memcpy(out_coords, cur_coords.data(), (size_t)cur_N * 4 * sizeof(int32_t));
    *out_N = cur_N;

    /* Cleanup */
    r->runner.destroyBuffer(d_feats);
    r->runner.destroyBuffer(d_coords);
    r->runner.destroyBuffer(d_hash_keys);
    r->runner.destroyBuffer(d_hash_vals);
    r->runner.destroyBuffer(d_out_w);
    r->runner.destroyBuffer(d_out_b);
    r->runner.destroyBuffer(d_output);

    t2_shape_dec_free(dec);
    fprintf(stderr, "T2VK: shape decoder done, output N=%d, C=%d\n", cur_N, out_ch);
    return 0;
}

/* ---- Normalization constants for Stage 2/3 sparse latents ---- */
static const float shape_slat_mean[32] = {0.781296f, 0.018091f, -0.495192f, -0.558457f, 1.060530f, 0.093252f, 1.518149f, -0.933218f, -0.732996f, 2.604095f, -0.118341f, -2.143904f, 0.495076f, -2.179512f, -2.130751f, -0.996944f, 0.261421f, -2.217463f, 1.260067f, -0.150213f, 3.790713f, 1.481266f, -1.046058f, -1.523667f, -0.059621f, 2.220780f, 1.621212f, 0.877230f, 0.567247f, -3.175944f, -3.186688f, 1.578665f};
static const float shape_slat_std[32]  = {5.972266f, 4.706852f, 5.445010f, 5.209927f, 5.320220f, 4.547237f, 5.020802f, 5.444004f, 5.226681f, 5.683095f, 4.831436f, 5.286469f, 5.652043f, 5.367606f, 5.525084f, 4.730578f, 4.805265f, 5.124013f, 5.530808f, 5.619001f, 5.103930f, 5.417670f, 5.269677f, 5.547194f, 5.634698f, 5.235274f, 6.110351f, 5.511298f, 6.237273f, 4.879207f, 5.347008f, 5.405691f};
static const float tex_slat_mean[32] = {3.501659f, 2.212398f, 2.226094f, 0.251093f, -0.026248f, -0.687364f, 0.439898f, -0.928075f, 0.029398f, -0.339596f, -0.869527f, 1.038479f, -0.972385f, 0.126042f, -1.129303f, 0.455149f, -1.209521f, 2.069067f, 0.544735f, 2.569128f, -0.323407f, 2.293000f, -1.925608f, -1.217717f, 1.213905f, 0.971588f, -0.023631f, 0.106750f, 2.021786f, 0.250524f, -0.662387f, -0.768862f};
static const float tex_slat_std[32]  = {2.665652f, 2.743913f, 2.765121f, 2.595319f, 3.037293f, 2.291316f, 2.144656f, 2.911822f, 2.969419f, 2.501689f, 2.154811f, 3.163343f, 2.621215f, 2.381943f, 3.186697f, 3.021588f, 2.295916f, 3.234985f, 3.233086f, 2.260140f, 2.874801f, 2.810596f, 3.292720f, 2.674999f, 2.680878f, 2.372054f, 2.451546f, 2.353556f, 2.995195f, 2.379849f, 2.786195f, 2.775190f};

int vulkan_trellis2_run_stage2(vulkan_trellis2_runner *r,
                                const float *features,
                                const float *occupancy,
                                float *shape_latent,
                                int32_t *out_coords,
                                int *out_N,
                                int n_steps,
                                float cfg_scale,
                                float cfg_rescale,
                                uint32_t seed) {
    if (!r->stage2_dit.loaded) {
        fprintf(stderr, "T2VK: Stage 2 weights not loaded\n");
        return -1;
    }

    /* 1. Extract sparse coords from occupancy grid (threshold > 0) */
    int N_sparse = 0;
    for (int i = 0; i < 64 * 64 * 64; i++)
        if (occupancy[i] > 0.0f) N_sparse++;

    std::vector<int32_t> coords(N_sparse * 4);
    int idx = 0;
    for (int z = 0; z < 64; z++)
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++)
                if (occupancy[z * 64 * 64 + y * 64 + x] > 0.0f) {
                    coords[idx * 4 + 0] = 0;  /* batch */
                    coords[idx * 4 + 1] = z;
                    coords[idx * 4 + 2] = y;
                    coords[idx * 4 + 3] = x;
                    idx++;
                }

    fprintf(stderr, "T2VK: Stage 2: %d sparse voxels, %d steps, cfg=%.1f\n",
            N_sparse, n_steps, cfg_scale);

    const int ch = 32;
    const int n_elem = N_sparse * ch;
    const float rescale_t_val = 3.0f;
    const float guidance_min = 0.6f, guidance_max = 1.0f;
    const float sigma_min = 1e-5f;

    /* 2. Generate noise [N, 32] */
    std::vector<float> x(n_elem);
    {
        t2_rng rng;
        uint64_t s = seed + 1;
        rng.s[0] = s; rng.s[1] = s ^ 0x9E3779B97F4A7C15ULL;
        rng.s[2] = s ^ 0x6C62272E07BB0142ULL; rng.s[3] = s ^ 0xBF58476D1CE4E5B9ULL;
        for (int i = 0; i < 8; i++) t2_next(&rng);
        for (int i = 0; i < n_elem; i++) x[i] = t2_randn(&rng);
    }

    std::vector<float> v_cond(n_elem), v_uncond(n_elem);

    /* Upload conditioning */
    const int ctx_len = DINO_SEQ_LEN;
    BufInfo d_cond = uploadF32(r->runner, features, (size_t)ctx_len * DIT_COND_DIM);
    BufInfo d_zero_cond = createGpuBuffer(r->runner, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    bufferZero(r->runner, d_zero_cond, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));

    /* Precompute sparse RoPE once (coords don't change during sampling) */
    BufInfo rope_cos, rope_sin;
    computeSparseRope(r, coords.data(), N_sparse, r->stage2_dit.n_freqs, rope_cos, rope_sin);

    /* Precompute cross-attention KV cache for conditioned + unconditioned */
    dit_kv_cache cond_kv = {}, uncond_kv = {};
    precomputeKvCache(r, r->stage2_dit, d_cond, cond_kv);
    precomputeKvCache(r, r->stage2_dit, d_zero_cond, uncond_kv);
    fprintf(stderr, "T2VK: Stage 2 KV cache precomputed (30 blocks × 2)\n");

    /* Pre-allocate x_t and output buffers (reused across steps) */
    BufInfo d_x = createGpuBuffer(r->runner, n_elem * sizeof(float));
    BufInfo d_out = createGpuBuffer(r->runner, n_elem * sizeof(float));

    /* 3. Euler sampling loop */
    for (int step = 0; step < n_steps; step++) {
        float t_start = 1.0f - (float)step / (float)n_steps;
        float t_end = 1.0f - (float)(step + 1) / (float)n_steps;
        float t_cur = t2_rescale(t_start, rescale_t_val);
        float t_next = t2_rescale(t_end, rescale_t_val);

        bool apply_cfg = (t_cur >= guidance_min && t_cur <= guidance_max && cfg_scale != 1.0f);

        /* Upload x_t */
        uploadToBuffer(r->runner, d_x, x.data(), n_elem * sizeof(float));

        if (apply_cfg) {
            /* Conditioned forward (with KV cache) */
            runDitForwardGeneric(r, d_x, t_cur, d_cond, d_out,
                                 N_sparse, r->stage2_dit, rope_cos, rope_sin, &cond_kv);
            downloadFromBuffer(r->runner, d_out, v_cond.data(), n_elem * sizeof(float));

            /* Unconditioned forward (with KV cache) */
            uploadToBuffer(r->runner, d_x, x.data(), n_elem * sizeof(float));
            runDitForwardGeneric(r, d_x, t_cur, d_zero_cond, d_out,
                                 N_sparse, r->stage2_dit, rope_cos, rope_sin, &uncond_kv);
            downloadFromBuffer(r->runner, d_out, v_uncond.data(), n_elem * sizeof(float));

            /* CFG combine: pred = cfg*cond + (1-cfg)*uncond */
            for (int i = 0; i < n_elem; i++)
                v_uncond[i] = cfg_scale * v_cond[i] + (1.0f - cfg_scale) * v_uncond[i];

            /* CFG rescale */
            if (cfg_rescale > 0.0f) {
                float tc = sigma_min + (1.0f - sigma_min) * t_cur;
                float one_m_sm = 1.0f - sigma_min;
                double sum_pos = 0, sum2_pos = 0, sum_cfg = 0, sum2_cfg = 0;
                for (int i = 0; i < n_elem; i++) {
                    float x0p = one_m_sm * x[i] - tc * v_cond[i];
                    float x0c = one_m_sm * x[i] - tc * v_uncond[i];
                    sum_pos += x0p; sum2_pos += (double)x0p * x0p;
                    sum_cfg += x0c; sum2_cfg += (double)x0c * x0c;
                }
                double n_d = (double)n_elem;
                double std_pos = sqrt((sum2_pos - sum_pos * sum_pos / n_d) / (n_d - 1.0));
                double std_cfg = sqrt((sum2_cfg - sum_cfg * sum_cfg / n_d) / (n_d - 1.0));
                float ratio = (std_cfg > 1e-8) ? (float)(std_pos / std_cfg) : 1.0f;
                float sc = cfg_rescale * ratio + (1.0f - cfg_rescale);
                for (int i = 0; i < n_elem; i++) {
                    float x0c = one_m_sm * x[i] - tc * v_uncond[i];
                    v_uncond[i] = (one_m_sm * x[i] - sc * x0c) / tc;
                }
            }

            float dt = t_cur - t_next;
            for (int i = 0; i < n_elem; i++) x[i] -= dt * v_uncond[i];
        } else {
            runDitForwardGeneric(r, d_x, t_cur, d_cond, d_out,
                                 N_sparse, r->stage2_dit, rope_cos, rope_sin, &cond_kv);
            downloadFromBuffer(r->runner, d_out, v_cond.data(), n_elem * sizeof(float));
            float dt = t_cur - t_next;
            for (int i = 0; i < n_elem; i++) x[i] -= dt * v_cond[i];
        }

        if (r->verbose >= 1)
            fprintf(stderr, "  Stage2 step %d/%d  t=%.4f->%.4f  %s\n",
                    step + 1, n_steps, t_cur, t_next, apply_cfg ? "CFG" : "noG");
    }

    r->runner.destroyBuffer(d_x);
    r->runner.destroyBuffer(d_out);
    r->runner.destroyBuffer(d_cond);
    r->runner.destroyBuffer(d_zero_cond);
    r->runner.destroyBuffer(rope_cos);
    r->runner.destroyBuffer(rope_sin);
    freeKvCache(r, cond_kv);
    freeKvCache(r, uncond_kv);

    /* 4. Denormalize shape latent: slat = x * std + mean */
    for (int i = 0; i < N_sparse; i++)
        for (int c = 0; c < ch; c++)
            x[i * ch + c] = x[i * ch + c] * shape_slat_std[c] + shape_slat_mean[c];

    /* Output */
    memcpy(shape_latent, x.data(), n_elem * sizeof(float));
    memcpy(out_coords, coords.data(), N_sparse * 4 * sizeof(int32_t));
    *out_N = N_sparse;

    fprintf(stderr, "T2VK: Stage 2 complete (N=%d)\n", N_sparse);
    return 0;
}

int vulkan_trellis2_run_stage3(vulkan_trellis2_runner *r,
                                const float *features,
                                const float *shape_latent,
                                const int32_t *coords, int N,
                                float *tex_latent,
                                int n_steps,
                                uint32_t seed) {
    if (!r->stage3_dit.loaded) {
        fprintf(stderr, "T2VK: Stage 3 weights not loaded\n");
        return -1;
    }

    fprintf(stderr, "T2VK: Stage 3: %d sparse voxels, %d steps, no CFG\n", N, n_steps);

    const int noise_ch = 32;
    const int in_ch = 64;  /* [noise_32, shape_norm_32] */
    const float rescale_t_val = 3.0f;

    /* 1. Normalize shape latent for concatenation */
    std::vector<float> shape_norm(N * 32);
    for (int i = 0; i < N; i++)
        for (int c = 0; c < 32; c++)
            shape_norm[i * 32 + c] = (shape_latent[i * 32 + c] - shape_slat_mean[c]) / shape_slat_std[c];

    /* 2. Generate noise [N, 32] */
    std::vector<float> noise(N * noise_ch);
    {
        t2_rng rng;
        uint64_t s = seed + 2;
        rng.s[0] = s; rng.s[1] = s ^ 0x9E3779B97F4A7C15ULL;
        rng.s[2] = s ^ 0x6C62272E07BB0142ULL; rng.s[3] = s ^ 0xBF58476D1CE4E5B9ULL;
        for (int i = 0; i < 8; i++) t2_next(&rng);
        for (int i = 0; i < N * noise_ch; i++) noise[i] = t2_randn(&rng);
    }

    /* Upload conditioning + precompute RoPE + KV cache */
    const int ctx_len = DINO_SEQ_LEN;
    BufInfo d_cond = uploadF32(r->runner, features, (size_t)ctx_len * DIT_COND_DIM);
    BufInfo rope_cos, rope_sin;
    computeSparseRope(r, coords, N, r->stage3_dit.n_freqs, rope_cos, rope_sin);

    dit_kv_cache cond_kv = {};
    precomputeKvCache(r, r->stage3_dit, d_cond, cond_kv);
    fprintf(stderr, "T2VK: Stage 3 KV cache precomputed\n");

    std::vector<float> x_t(N * in_ch);
    std::vector<float> v(N * noise_ch);

    /* Pre-allocate GPU buffers (reused across steps) */
    BufInfo d_x = createGpuBuffer(r->runner, (size_t)N * in_ch * sizeof(float));
    BufInfo d_out = createGpuBuffer(r->runner, (size_t)N * noise_ch * sizeof(float));

    /* 3. Euler sampling loop (no CFG) */
    for (int step = 0; step < n_steps; step++) {
        float t_start = 1.0f - (float)step / (float)n_steps;
        float t_end = 1.0f - (float)(step + 1) / (float)n_steps;
        float t_cur = t2_rescale(t_start, rescale_t_val);
        float t_next = t2_rescale(t_end, rescale_t_val);

        /* Build x_t = [noise, shape_norm] per token */
        for (int i = 0; i < N; i++) {
            memcpy(x_t.data() + i * in_ch, noise.data() + i * noise_ch, noise_ch * sizeof(float));
            memcpy(x_t.data() + i * in_ch + noise_ch, shape_norm.data() + i * 32, 32 * sizeof(float));
        }

        uploadToBuffer(r->runner, d_x, x_t.data(), (size_t)N * in_ch * sizeof(float));

        runDitForwardGeneric(r, d_x, t_cur, d_cond, d_out,
                             N, r->stage3_dit, rope_cos, rope_sin, &cond_kv);
        downloadFromBuffer(r->runner, d_out, v.data(), (size_t)N * noise_ch * sizeof(float));

        /* Euler step: noise -= dt * v */
        float dt = t_cur - t_next;
        for (int i = 0; i < N * noise_ch; i++) noise[i] -= dt * v[i];

        if (r->verbose >= 1)
            fprintf(stderr, "  Stage3 step %d/%d  t=%.4f->%.4f\n",
                    step + 1, n_steps, t_cur, t_next);
    }

    r->runner.destroyBuffer(d_x);
    r->runner.destroyBuffer(d_out);
    r->runner.destroyBuffer(d_cond);
    r->runner.destroyBuffer(rope_cos);
    r->runner.destroyBuffer(rope_sin);
    freeKvCache(r, cond_kv);

    /* 4. Denormalize texture latent */
    for (int i = 0; i < N; i++)
        for (int c = 0; c < noise_ch; c++)
            noise[i * noise_ch + c] = noise[i * noise_ch + c] * tex_slat_std[c] + tex_slat_mean[c];

    memcpy(tex_latent, noise.data(), (size_t)N * noise_ch * sizeof(float));
    fprintf(stderr, "T2VK: Stage 3 complete\n");
    return 0;
}

void vulkan_trellis2_free(vulkan_trellis2_runner *r) {
    if (!r) return;
    for (int i = 0; i < 12; i++) {
        if (r->scratch[i].buffer != VK_NULL_HANDLE)
            r->runner.destroyBuffer(r->scratch[i]);
    }
    /* TODO: destroy all weight and pipeline buffers */
    delete r;
}
