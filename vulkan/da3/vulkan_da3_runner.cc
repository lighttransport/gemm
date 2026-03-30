// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan DA3 Depth Estimation Runner
//
// Pipeline: Preprocessing -> ViT backbone (12 transformer blocks) -> DPT head -> depth output
// Uses pre-compiled SPIR-V shaders and DEVICE_LOCAL SSBOs with batched dispatch.
// F32 weights (converted from F16/BF16 at load time).
//
// Port of hip_da3_runner.c for cross-platform Vulkan compute.
//

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#include "vulkan_da3_runner.h"
#include "../deps/vulkan-runner.hh"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <ctime>

/* ======================================================================== */
/* Types                                                                    */
/* ======================================================================== */

using VulkanRunner = vl_cpp::vulkan::VulkanComputeRunner;
using BufInfo = VulkanRunner::BufferInfo;
using Pipeline = VulkanRunner::ComputePipeline;

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

struct vulkan_da3_runner {
    VulkanRunner runner;
    int verbose;
    std::string shader_dir;

    /* Pipelines */
    Pipeline pipe_layernorm;          /* 4 bufs: src,dst,w,b; push: {n_tokens,dim,eps} = 12B */
    Pipeline pipe_matmul_bias;        /* 4 bufs: A,W,C,bias; push: {M,N,K,has_bias} = 16B */
    Pipeline pipe_gelu;               /* 1 buf: data; push: {n} = 4B */
    Pipeline pipe_add;                /* 2 bufs: dst,src; push: {n} = 4B */
    Pipeline pipe_layerscale_add;     /* 3 bufs: dst,src,scale; push: {n,dim} = 8B */
    Pipeline pipe_qk_layernorm;       /* 3 bufs: data,w,b; push: {n_tok,n_heads,head_dim,stride,eps} = 20B */
    Pipeline pipe_patch_embed;        /* 4 bufs: image,kernel,bias,output; push: 24B */

    /* DA3-specific pipelines */
    Pipeline pipe_resize_normalize;   /* 2 bufs: dst,src; push: 40B */
    Pipeline pipe_cls_pos_embed;      /* 3 bufs: hidden,cls,pos; push: {nt,dim} = 8B */
    Pipeline pipe_rope_2d;            /* 3 bufs: vec,pos_y,pos_x; push: {n_tok,n_heads,head_dim,stride,freq_base} = 20B */
    Pipeline pipe_swiglu;             /* 2 bufs: dst,gate_up; push: {hidden,n_tok} = 8B */
    Pipeline pipe_kv_transpose;       /* 3 bufs: K_t,V_t,qkv; push: {n_tok,dim,n_heads,head_dim} = 16B */
    Pipeline pipe_flash_attn;         /* 4 bufs: out,qkv,K_t,V_t; push: {n_tok,dim,n_heads,head_dim,scale} = 20B */
    Pipeline pipe_cat_local_global;   /* 3 bufs: dst,local,global; push: {nt,dim} = 8B */
    Pipeline pipe_dpt_cls_concat;     /* 2 bufs: dst,src; push: {np,dim} = 8B */
    Pipeline pipe_strided_layernorm;  /* 3 bufs: data,w,b; push: {np,dim,stride,eps} = 16B */
    Pipeline pipe_tok_to_chw;         /* 2 bufs: dst,src; push: {C,gH,gW} = 12B */
    Pipeline pipe_deconv_scatter;     /* 3 bufs: dst,Y,bias; push: 36B */
    Pipeline pipe_conv2d;             /* 4 bufs: dst,src,w,bias; push: 32B */
    Pipeline pipe_bilinear;           /* 2 bufs: dst,src; push: {C,Hi,Wi,Ho,Wo} = 20B */
    Pipeline pipe_depth_activation;   /* 1 buf: out; push: {hw} = 4B */
    Pipeline pipe_sinusoidal_uv;      /* 1 buf: data; push: {C,H,W,span_x,span_y,ratio} = 24B */
    Pipeline pipe_relu;               /* 1 buf: x; push: {n} = 4B */
    Pipeline pipe_channel_layernorm;  /* 4 bufs: dst,src,w,b; push: {C,HW,eps} = 12B */
    Pipeline pipe_silu;               /* 1 buf: x; push: {n} = 4B */
    Pipeline pipe_matmul_bias_f16;    /* 4 bufs: A(f32),W(f16),C(f32),bias(f32); push: 16B */
    Pipeline pipe_copy;               /* 2 bufs: dst,src; push: {n} = 4B */
    Pipeline pipe_qk_layernorm_offset; /* 3 bufs: data,w,b; push: {n_tok,n_heads,head_dim,stride,eps,base_offset} = 24B */

    /* Model params */
    int n_blocks, dim, n_heads, head_dim, ffn_hidden;
    int patch_size, image_size, grid_h, grid_w, n_patches, n_tokens;
    float ln_eps;
    int rope_start, qk_norm_start;
    int feature_layers[4];
    int use_swiglu;
    int head_features, head_out_channels[4];
    float image_mean[3], image_std[3];

    /* Backbone weights */
    struct Layer {
        BufInfo ln1_w, ln1_b, attn_qkv_w, attn_qkv_b;
        BufInfo attn_q_norm_w, attn_q_norm_b, attn_k_norm_w, attn_k_norm_b;
        BufInfo attn_out_w, attn_out_b;
        BufInfo ls1, ls2;
        BufInfo ln2_w, ln2_b;
        BufInfo ffn_gate_up_w, ffn_gate_up_b;
        BufInfo ffn_up_w, ffn_up_b;
        BufInfo ffn_down_w, ffn_down_b;
        bool has_qk_norm = false, has_swiglu = false;
        int qkv_rows = 0, out_rows = 0, ffn_gu_rows = 0, ffn_up_rows = 0, ffn_down_rows = 0;
    };
    std::vector<Layer> layers;
    BufInfo patch_embed_w, patch_embed_b, cls_token, pos_embed, camera_token;
    float *h_pos_embed_orig = nullptr;  /* CPU copy for interpolation */
    BufInfo backbone_norm_w, backbone_norm_b;
    bool has_camera_token = false;

    /* DPT head weights */
    struct DPT {
        BufInfo norm_w, norm_b;
        BufInfo proj_w[4], proj_b[4];
        int proj_rows[4];
        BufInfo upsample_0_w, upsample_0_b, upsample_1_w, upsample_1_b;
        BufInfo downsample_w, downsample_b;
        BufInfo adapter_w[4];
        BufInfo fuse_out_w[4], fuse_out_b[4];
        BufInfo fuse_rcu1_c1_w[4], fuse_rcu1_c1_b[4], fuse_rcu1_c2_w[4], fuse_rcu1_c2_b[4];
        BufInfo fuse_rcu2_c1_w[4], fuse_rcu2_c1_b[4], fuse_rcu2_c2_w[4], fuse_rcu2_c2_b[4];
        bool has_rcu1[4] = {}, has_rcu2[4] = {};
        BufInfo neck_w, neck_b, out_0_w, out_0_b, out_2_w, out_2_b;
        int out_mid = 0;
    } dpt;

    /* CameraDec weights (pose estimation) */
    struct CamDec {
        BufInfo backbone_norm_w, backbone_norm_b;   /* F32 */
        BufInfo mlp_w[2], mlp_b[2];                /* F32 [dim*2, dim] and [dim*2, dim*2] */
        BufInfo fc_t_w, fc_t_b;                    /* F32 [3, dim*2] (tiny, CPU matmul) */
        BufInfo fc_qvec_w, fc_qvec_b;              /* F32 [4, dim*2] */
        BufInfo fc_fov_w, fc_fov_b;                /* F32 [2, dim*2] */
        int mlp_dim = 0;
        bool loaded = false;
    } cam_dec;

    /* CameraEnc weights (pose conditioning) */
    struct CamEnc {
        BufInfo fc1_w, fc1_b, fc2_w, fc2_b;  /* pose MLP: 9 -> dim/2 -> dim */
        struct TrunkBlock {
            BufInfo ln1_w, ln1_b, attn_qkv_w, attn_qkv_b;
            BufInfo attn_out_w, attn_out_b;
            BufInfo ls1, ls2;
            BufInfo ln2_w, ln2_b;
            BufInfo ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b;
            int qkv_rows = 0, out_rows = 0, ffn_up_rows = 0;
        };
        std::vector<TrunkBlock> trunk;
        BufInfo trunk_norm_w, trunk_norm_b;
        BufInfo token_norm_w, token_norm_b;
        int trunk_dim = 0;
        bool loaded = false;
    } cam_enc;

    /* DPT Aux Branch weights (rays + sky segmentation) */
    struct DptAux {
        BufInfo fuse_out_w[4], fuse_out_b[4];
        BufInfo fuse_rcu1_c1_w[4], fuse_rcu1_c1_b[4];
        BufInfo fuse_rcu1_c2_w[4], fuse_rcu1_c2_b[4];
        BufInfo fuse_rcu2_c1_w[4], fuse_rcu2_c1_b[4];
        BufInfo fuse_rcu2_c2_w[4], fuse_rcu2_c2_b[4];
        bool has_rcu1[4] = {}, has_rcu2[4] = {};
        /* Per-level output conv chains (up to 5 Conv2d each) */
        BufInfo oc1_w[4][5], oc1_b[4][5];
        int oc1_ci[4][5] = {}, oc1_co[4][5] = {};
        int oc1_count[4] = {};
        /* output_conv2_aux: Conv2d(128,32,3) + ChannelLayerNorm(32) + ReLU + Conv2d(32,7,1) */
        BufInfo oc2_conv_w[4], oc2_conv_b[4];
        BufInfo oc2_gn_w[4], oc2_gn_b[4];
        BufInfo oc2_out_w[4], oc2_out_b[4];
        bool loaded = false;
    } dpt_aux;

    /* GSDPT weights (3D Gaussian estimation) */
    struct GSDPT {
        DPT dpt;                                    /* standard DPT weights for GS branch */
        BufInfo merger_w[3], merger_b[3];           /* images_merger Conv2d, F32 weights */
        int merger_ci[3] = {}, merger_co[3] = {};
        int gs_out_channels = 0;
        bool loaded = false;
    } gsdpt;

    /* Scratch buffers */
    BufInfo hidden, hidden2, local_hidden, ln_buf;
    BufInfo qkv, attn_out, ffn_buf, ffn_mid, proj_out;
    BufInfo pos_y, pos_x, pos_y_nd, pos_x_nd;
    BufInfo features[4];
    BufInfo img_norm, img_raw;
    size_t img_raw_cap = 0;
    BufInfo dpt_cat, dpt_ln, dpt_proj, dpt_chw;
    BufInfo dpt_spatial[4], dpt_adapted[4];
    BufInfo dpt_fused, dpt_tmp, dpt_tmp2, dpt_out;
    BufInfo result_buf;
    size_t result_cap = 0;

    /* Aux DPT scratch */
    BufInfo aux_out;          /* [7, sp_h[0], sp_w[0]] */
    BufInfo aux_scratch;      /* scratch for aux output conv chains */

    /* GSDPT scratch */
    BufInfo gs_merged;        /* images_merger output [128, mg_h, mg_w] */
    BufInfo gs_out;           /* [gs_oc, fh, fw] gaussian output */
    int gs_merger_h = 0, gs_merger_w = 0;

    BufInfo zero_bias;  /* pre-allocated zero buffer for no-bias Conv2d */

    /* Staging buffers for device-local transfers */
    BufInfo upload_staging;   /* persistent HOST_VISIBLE for image upload */
    BufInfo download_staging; /* persistent HOST_VISIBLE for result download */
    size_t upload_staging_cap = 0;
    size_t download_staging_cap = 0;

    /* Batch fence for fenced submit */
    VkFence batch_fence = VK_NULL_HANDLE;

    bool loaded = false;
    bool batched = false; /* when true, dispatchOp uses batched mode */
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
    runner.createDeviceLocalBuffer(size, buf);
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

static bool isValid(const BufInfo &buf) {
    return buf.buffer != VK_NULL_HANDLE;
}

/* Upload safetensors tensor as F32 to GPU */
static BufInfo stUploadF32(VulkanRunner &runner, st_context *st, const char *name, int verbose) {
    BufInfo buf{};
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose >= 2) fprintf(stderr, "  [WARN] tensor '%s' not found\n", name);
        return buf;
    }
    void *data = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "F32") == 0) {
        buf = createGpuBuffer(runner, nbytes);
        runner.uploadToDeviceLocal(buf, data, nbytes);
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
        runner.uploadToDeviceLocal(buf, f32.data(), n * sizeof(float));
    } else if (strcmp(dtype, "BF16") == 0) {
        size_t n = nbytes / sizeof(uint16_t);
        std::vector<float> f32(n);
        const uint16_t *bf16 = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            uint32_t f = (uint32_t)bf16[i] << 16;
            memcpy(&f32[i], &f, sizeof(float));
        }
        buf = createGpuBuffer(runner, n * sizeof(float));
        runner.uploadToDeviceLocal(buf, f32.data(), n * sizeof(float));
    }
    return buf;
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
/* Dispatch helpers                                                         */
/* ======================================================================== */

/* Batched dispatch: record into open command buffer with barrier, no sync.
 * Call beginBatch() before first dispatch, endBatch() after last. */
static void dispatchBatched(vulkan_da3_runner *r, Pipeline &pipe,
                              const std::vector<BufInfo> &bufs,
                              const void *push_data, uint32_t push_size,
                              uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    auto ds = r->runner.allocateAndUpdateDescriptorSet(pipe, bufs);
    r->runner.bindComputePipeline(pipe);
    r->runner.bindDescriptorSetDynamic(pipe, ds);
    if (push_data && push_size > 0)
        r->runner.pushConstants(pipe, push_data, push_size);
    r->runner.dispatch(gx, gy, gz);
    r->runner.computeBarrier();
}

static void beginBatch(vulkan_da3_runner *r) { r->runner.beginRecording(); }
static void endBatch(vulkan_da3_runner *r) {
    r->runner.endRecordingAndSubmit();
    r->runner.waitForCompletion();
    r->runner.resetDynamicDescriptorPool();
}

/* Dispatch: record + submit per dispatch, no waitForCompletion.
 * Vulkan guarantees submission-order execution on the same queue.
 * Only call gpuSync() before CPU reads (downloadFromBuffer). */
static void dispatchSync(vulkan_da3_runner *r, Pipeline &pipe,
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
    r->runner.computeBarrier();
    r->runner.endRecordingAndSubmit();
    r->runner.waitForCompletion();
}

static void gpuSync(vulkan_da3_runner *r) {
    r->runner.waitForCompletion();
}

/* Dispatch that routes to batched or sync mode based on r->batched flag */
static void dispatchOp(vulkan_da3_runner *r, Pipeline &pipe,
                        const std::vector<BufInfo> &bufs,
                        const void *push_data, uint32_t push_size,
                        uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    if (r->batched) {
        dispatchBatched(r, pipe, bufs, push_data, push_size, gx, gy, gz);
    } else {
        dispatchSync(r, pipe, bufs, push_data, push_size, gx, gy, gz);
    }
}

/* ---- Op wrappers ---- */

/* LayerNorm: dst = LN(src, weight, bias) */
static void opLayerNorm(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src,
                          BufInfo &w, BufInfo &b, int n_tok, int dim) {
    struct { uint32_t n_tokens, dim; float eps; } pc = {
        (uint32_t)n_tok, (uint32_t)dim, r->ln_eps
    };
    dispatchOp(r, r->pipe_layernorm, {src, dst, w, b}, &pc, sizeof(pc), (uint32_t)n_tok);
}

/* MatMul+Bias: C[M,N] = A[M,K] * W[N,K]^T + bias[N]
 * bindings: {A, W, C, bias}, push: {M, N, K, has_bias}
 * Grid: (ceil(N/16), ceil(M/16)) since workgroup is 16x16 */
static void opMatmulBias(vulkan_da3_runner *r, BufInfo &C, BufInfo &W, BufInfo &A, BufInfo &bias,
                           int M, int N, int K) {
    bool has_bias = isValid(bias);
    struct { uint32_t M, N, K_val, has_bias; } pc = {
        (uint32_t)M, (uint32_t)N, (uint32_t)K, has_bias ? 1u : 0u
    };
    uint32_t gx = (uint32_t)((N + 15) / 16);
    uint32_t gy = (uint32_t)((M + 15) / 16);
    /* Vulkan requires valid buffers for all bindings even when has_bias=0.
     * Use C as a dummy for the bias binding when no bias is provided. */
    BufInfo &bias_buf = has_bias ? bias : C;
    dispatchOp(r, r->pipe_matmul_bias, {A, W, C, bias_buf}, &pc, sizeof(pc), gx, gy);
}

/* MatMul+Bias with F16 weights: C[M,N] = A[M,K](f32) * W[N,K](f16)^T + bias[N](f32)
 * Same interface as opMatmulBias but uses the F16 weight pipeline. */
static void opMatmulBiasF16(vulkan_da3_runner *r, BufInfo &C, BufInfo &W, BufInfo &A, BufInfo &bias,
                               int M, int N, int K) {
    bool has_bias = isValid(bias);
    struct { uint32_t M, N, K_val, has_bias; } pc = {
        (uint32_t)M, (uint32_t)N, (uint32_t)K, has_bias ? 1u : 0u
    };
    uint32_t gx = (uint32_t)((N + 15) / 16);
    uint32_t gy = (uint32_t)((M + 15) / 16);
    BufInfo &bias_buf = has_bias ? bias : C;
    dispatchOp(r, r->pipe_matmul_bias_f16, {A, W, C, bias_buf}, &pc, sizeof(pc), gx, gy);
}

/* GELU in-place */
static void opGelu(vulkan_da3_runner *r, BufInfo &x, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchOp(r, r->pipe_gelu, {x}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* ReLU in-place */
static void opRelu(vulkan_da3_runner *r, BufInfo &x, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchOp(r, r->pipe_relu, {x}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* GPU buffer copy (batchable, unlike bufferCopy which uses CPU memcpy) */
static void opCopy(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src, int n_floats) {
    uint32_t pc = (uint32_t)n_floats;
    dispatchOp(r, r->pipe_copy, {dst, src}, &pc, sizeof(pc), (uint32_t)((n_floats + 255) / 256));
}

/* SiLU in-place: x = x * sigmoid(x) */
static void opSilu(vulkan_da3_runner *r, BufInfo &x, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchOp(r, r->pipe_silu, {x}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* Element-wise add: dst += src */
static void opAdd(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src, int n) {
    uint32_t pc = (uint32_t)n;
    dispatchOp(r, r->pipe_add, {dst, src}, &pc, sizeof(pc), (uint32_t)((n + 255) / 256));
}

/* LayerScale add: dst[i] += src[i] * scale[i % dim] */
static void opLayerscaleAdd(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src,
                              BufInfo &scale, int n, int dim) {
    struct { uint32_t n, dim; } pc = {(uint32_t)n, (uint32_t)dim};
    dispatchOp(r, r->pipe_layerscale_add, {dst, src, scale}, &pc, sizeof(pc),
                 (uint32_t)((n + 255) / 256));
}

/* Per-head QK LayerNorm */
static void opQkLayerNorm(vulkan_da3_runner *r, BufInfo &data, BufInfo &w, BufInfo &b,
                            int n_tok, int n_heads, int head_dim, int stride) {
    struct { uint32_t n_tok, n_heads, head_dim, stride; float eps; } pc = {
        (uint32_t)n_tok, (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)stride, r->ln_eps
    };
    int total = n_tok * n_heads;
    dispatchOp(r, r->pipe_qk_layernorm, {data, w, b}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Per-head QK LayerNorm with base_offset (for K portion of interleaved QKV) */
static void opQkLayerNormOffset(vulkan_da3_runner *r, BufInfo &data, BufInfo &w, BufInfo &b,
                                  int n_tok, int n_heads, int head_dim, int stride, float eps, int base_offset = 0) {
    struct { uint32_t n_tok, n_heads, head_dim, stride; float eps; uint32_t base_offset; } pc = {
        (uint32_t)n_tok, (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)stride, eps, (uint32_t)base_offset
    };
    int total = n_tok * n_heads;
    dispatchOp(r, r->pipe_qk_layernorm_offset, {data, w, b}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* 2D RoPE */
static void opRope2d(vulkan_da3_runner *r, BufInfo &vec, BufInfo &pos_y_buf, BufInfo &pos_x_buf,
                       int n_tok, int n_heads, int head_dim, int stride, float freq_base = 100.0f, int base_offset = 0) {
    struct { uint32_t n_tok, n_heads, head_dim, stride; float freq_base; uint32_t base_offset; } pc = {
        (uint32_t)n_tok, (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)stride, freq_base, (uint32_t)base_offset
    };
    /* One workgroup per token; threads = n_heads * (head_dim/4) */
    dispatchOp(r, r->pipe_rope_2d, {vec, pos_y_buf, pos_x_buf}, &pc, sizeof(pc),
                 (uint32_t)n_tok);
}

/* SwiGLU: dst = silu(gate) * up, from [n_tok, 2*hidden] gate_up */
static void opSwiglu(vulkan_da3_runner *r, BufInfo &dst, BufInfo &gate_up, int hidden, int n_tok) {
    struct { uint32_t hidden, n_tok; } pc = {(uint32_t)hidden, (uint32_t)n_tok};
    int total = hidden * n_tok;
    dispatchOp(r, r->pipe_swiglu, {dst, gate_up}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* KV transpose: extract K^T and V^T from interleaved QKV */
static void opKvTranspose(vulkan_da3_runner *r, BufInfo &K_t, BufInfo &V_t,
                            BufInfo &qkv_buf, int n_tok, int dim, int n_heads, int head_dim) {
    struct { uint32_t n_tok, dim, n_heads, head_dim; } pc = {
        (uint32_t)n_tok, (uint32_t)dim, (uint32_t)n_heads, (uint32_t)head_dim
    };
    int total = n_tok * dim;
    dispatchOp(r, r->pipe_kv_transpose, {K_t, V_t, qkv_buf}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Flash attention (tiled) */
static void opFlashAttn(vulkan_da3_runner *r, BufInfo &out, BufInfo &qkv_buf,
                          BufInfo &K_t, BufInfo &V_t,
                          int n_tok, int dim, int n_heads, int head_dim) {
    float scale = 1.0f / sqrtf((float)head_dim);
    struct { uint32_t n_tok, dim, n_heads, head_dim; float scale; } pc = {
        (uint32_t)n_tok, (uint32_t)dim, (uint32_t)n_heads, (uint32_t)head_dim, scale
    };
    int bq = 64;
    uint32_t gy = (uint32_t)((n_tok + bq - 1) / bq);
    dispatchOp(r, r->pipe_flash_attn, {out, qkv_buf, K_t, V_t}, &pc, sizeof(pc),
                 (uint32_t)n_heads, gy);
}

/* CLS + pos_embed */
static void opClsPosEmbed(vulkan_da3_runner *r, BufInfo &hidden_buf, BufInfo &cls_buf,
                            BufInfo &pos_buf, int nt, int dim) {
    struct { uint32_t nt, dim; } pc = {(uint32_t)nt, (uint32_t)dim};
    int total = nt * dim;
    dispatchOp(r, r->pipe_cls_pos_embed, {hidden_buf, cls_buf, pos_buf}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* cat_local_global: dst[t*2*dim+j] = local[j<dim] or global[j-dim] */
static void opCatLocalGlobal(vulkan_da3_runner *r, BufInfo &dst, BufInfo &local_buf,
                               BufInfo &global_buf, int nt, int dim) {
    struct { uint32_t nt, dim; } pc = {(uint32_t)nt, (uint32_t)dim};
    int total = nt * 2 * dim;
    dispatchOp(r, r->pipe_cat_local_global, {dst, local_buf, global_buf}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* DPT CLS concat: extract patch tokens from [nt, 2*dim] (skip CLS) */
static void opDptClsConcat(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src, int np, int dim) {
    struct { uint32_t np, dim; } pc = {(uint32_t)np, (uint32_t)dim};
    int total = np * 2 * dim;
    dispatchOp(r, r->pipe_dpt_cls_concat, {dst, src}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Strided LayerNorm: apply LN to second half of [np, 2*dim] in-place */
static void opStridedLayerNorm(vulkan_da3_runner *r, BufInfo &data, BufInfo &w, BufInfo &b,
                                 int np, int dim, int stride) {
    struct { uint32_t np, dim, stride; float eps; } pc = {
        (uint32_t)np, (uint32_t)dim, (uint32_t)stride, r->ln_eps
    };
    dispatchOp(r, r->pipe_strided_layernorm, {data, w, b}, &pc, sizeof(pc), (uint32_t)np);
}

/* Token-major [np, C] -> spatial CHW [C, H, W] */
static void opTokToChw(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src, int C, int gH, int gW) {
    struct { uint32_t C, gH, gW; } pc = {(uint32_t)C, (uint32_t)gH, (uint32_t)gW};
    int total = C * gH * gW;
    dispatchOp(r, r->pipe_tok_to_chw, {dst, src}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Conv2D */
static void opConv2d(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src,
                       BufInfo &w, BufInfo &bias,
                       int H, int W, int Ci, int Co, int kH, int kW, int stride, int pad) {
    struct { uint32_t H, W, Ci, Co, kH, kW, stride, pad; } pc = {
        (uint32_t)H, (uint32_t)W, (uint32_t)Ci, (uint32_t)Co,
        (uint32_t)kH, (uint32_t)kW, (uint32_t)stride, (uint32_t)pad
    };
    int Ho = (H + 2 * pad - kH) / stride + 1;
    int Wo = (W + 2 * pad - kW) / stride + 1;
    int total = Co * Ho * Wo;
    dispatchOp(r, r->pipe_conv2d, {dst, src, w, bias}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Deconv scatter: used after matmul for ConvTranspose2d */
static void opDeconvScatter(vulkan_da3_runner *r, BufInfo &dst, BufInfo &Y, BufInfo &bias,
                              int Co, int Hi, int Wi, int Ho, int Wo,
                              int kH, int kW, int stride_h, int stride_w) {
    struct { uint32_t Co, Hi, Wi, Ho, Wo, kH, kW, stride_h, stride_w; } pc = {
        (uint32_t)Co, (uint32_t)Hi, (uint32_t)Wi, (uint32_t)Ho, (uint32_t)Wo,
        (uint32_t)kH, (uint32_t)kW, (uint32_t)stride_h, (uint32_t)stride_w
    };
    int total = Co * Ho * Wo;
    dispatchOp(r, r->pipe_deconv_scatter, {dst, Y, bias}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Bilinear upsample */
static void opBilinear(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src,
                         int C, int Hi, int Wi, int Ho, int Wo) {
    struct { uint32_t C, Hi, Wi, Ho, Wo; } pc = {
        (uint32_t)C, (uint32_t)Hi, (uint32_t)Wi, (uint32_t)Ho, (uint32_t)Wo
    };
    int total = C * Ho * Wo;
    dispatchOp(r, r->pipe_bilinear, {dst, src}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Depth activation: out[i] = exp(out[i]), out[i+hw] = exp(out[i+hw]) + 1 */
static void opDepthActivation(vulkan_da3_runner *r, BufInfo &out, int hw) {
    uint32_t pc = (uint32_t)hw;
    dispatchOp(r, r->pipe_depth_activation, {out}, &pc, sizeof(pc),
                 (uint32_t)((hw + 255) / 256));
}

/* Sinusoidal UV positional embedding */
static void opSinusoidalUv(vulkan_da3_runner *r, BufInfo &data,
                             int C, int H, int W, float span_x, float span_y, float ratio) {
    struct { uint32_t C, H, W; float span_x, span_y, ratio; } pc = {
        (uint32_t)C, (uint32_t)H, (uint32_t)W, span_x, span_y, ratio
    };
    int total = C * H * W;
    dispatchOp(r, r->pipe_sinusoidal_uv, {data}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Channel LayerNorm */
static void opChannelLayerNorm(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src,
                                 BufInfo &w, BufInfo &b, int C, int HW) {
    struct { uint32_t C, HW; float eps; } pc = {(uint32_t)C, (uint32_t)HW, 1e-5f};
    dispatchOp(r, r->pipe_channel_layernorm, {dst, src, w, b}, &pc, sizeof(pc),
                 (uint32_t)((HW + 255) / 256));
}

/* Resize + normalize (preprocessing) */
static void opResizeNormalize(vulkan_da3_runner *r, BufInfo &dst, BufInfo &src,
                                int src_w, int src_h, int dst_w, int dst_h) {
    struct {
        uint32_t sw, sh, dw, dh;
        float m0, m1, m2, istd0, istd1, istd2;
    } pc = {
        (uint32_t)src_w, (uint32_t)src_h, (uint32_t)dst_w, (uint32_t)dst_h,
        r->image_mean[0], r->image_mean[1], r->image_mean[2],
        1.0f / r->image_std[0], 1.0f / r->image_std[1], 1.0f / r->image_std[2]
    };
    int total = dst_h * dst_w;
    dispatchOp(r, r->pipe_resize_normalize, {dst, src}, &pc, sizeof(pc),
                 (uint32_t)((total + 255) / 256));
}

/* Patch embedding */
/* DA3 patch_embed_conv2d: image[CHW] -> hidden[(1+pid)*dim+co] */
static void opPatchEmbed(vulkan_da3_runner *r, BufInfo &hidden, BufInfo &image,
                           BufInfo &kernel_w, BufInfo &bias,
                           int np, int dim, int ps, int gw, int img_w, int img_h) {
    struct { uint32_t gw, dim, ps, img_w, img_h; } pc = {
        (uint32_t)gw, (uint32_t)dim, (uint32_t)ps, (uint32_t)img_w, (uint32_t)img_h
    };
    /* One workgroup per patch, 256 threads iterate over output channels */
    dispatchOp(r, r->pipe_patch_embed, {hidden, image, kernel_w, bias}, &pc, sizeof(pc),
                 (uint32_t)np);
}

/* Deconv via GEMM + scatter: ConvTranspose2d */
static void opDeconvGemmScatter(vulkan_da3_runner *r, BufInfo &dst, BufInfo &X,
                                  BufInfo &W, BufInfo &bias, BufInfo &scratch,
                                  int Ci, int Co, int Hi, int Wi,
                                  int kH, int kW, int stride) {
    /* Step 1: GEMM: scratch[Hi*Wi, kH*kW*Co] = X[Hi*Wi, Ci] * W[kH*kW*Co, Ci]^T (F16 weights) */
    int N = kH * kW * Co;
    BufInfo null_bias{};
    opMatmulBiasF16(r, scratch, W, X, null_bias, Hi * Wi, N, Ci);

    /* Step 2: Scatter to spatial output */
    int Ho = (Hi - 1) * stride + kH;
    int Wo = (Wi - 1) * stride + kW;
    opDeconvScatter(r, dst, scratch, bias, Co, Hi, Wi, Ho, Wo, kH, kW, stride, stride);
}

/* RCU: relu -> conv3x3 -> relu -> conv3x3 + residual */
static void opRcu(vulkan_da3_runner *r, BufInfo &out, BufInfo &x,
                    BufInfo &c1w, BufInfo &c1b, BufInfo &c2w, BufInfo &c2b,
                    int C, int H, int W) {
    int sz = C * H * W;
    /* tmp2 = relu(x) */
    opCopy(r, r->dpt_tmp2, x, sz);
    opRelu(r, r->dpt_tmp2, sz);
    /* tmp = conv3x3(tmp2) */
    opConv2d(r, r->dpt_tmp, r->dpt_tmp2, c1w, c1b, H, W, C, C, 3, 3, 1, 1);
    /* tmp = relu(tmp) */
    opRelu(r, r->dpt_tmp, sz);
    /* out = conv3x3(tmp) */
    opConv2d(r, out, r->dpt_tmp, c2w, c2b, H, W, C, C, 3, 3, 1, 1);
    /* out += x (residual) */
    opAdd(r, out, x, sz);
}

/* RefineNet fusion block */
static void opRefineNet(vulkan_da3_runner *r,
                          BufInfo &feat, int fH, int fW,
                          BufInfo *deeper, int dH, int dW,
                          int features, BufInfo &result,
                          BufInfo &fuse_out_w, BufInfo &fuse_out_b,
                          BufInfo &rcu1_c1_w, BufInfo &rcu1_c1_b,
                          BufInfo &rcu1_c2_w, BufInfo &rcu1_c2_b,
                          BufInfo &rcu2_c1_w, BufInfo &rcu2_c1_b,
                          BufInfo &rcu2_c2_w, BufInfo &rcu2_c2_b,
                          bool has_rcu1, bool has_rcu2,
                          int out_h, int out_w) {
    int sz = features * fH * fW;

    if (deeper) {
        /* result = upsample(deeper) + RCU(lateral) */
        opBilinear(r, result, *deeper, features, dH, dW, fH, fW);
        if (has_rcu1) {
            opRcu(r, r->dpt_ln, feat, rcu1_c1_w, rcu1_c1_b, rcu1_c2_w, rcu1_c2_b, features, fH, fW);
            opAdd(r, result, r->dpt_ln, sz);
        } else {
            opAdd(r, result, feat, sz);
        }
    } else {
        opCopy(r, result, feat, sz);
    }

    if (has_rcu2) {
        opRcu(r, r->dpt_cat, result, rcu2_c1_w, rcu2_c1_b, rcu2_c2_w, rcu2_c2_b, features, fH, fW);
        opCopy(r, result, r->dpt_cat, sz);
    }

    /* Upsample + out_conv */
    if (out_h > 0 && (out_h != fH || out_w != fW)) {
        opBilinear(r, r->dpt_cat, result, features, fH, fW, out_h, out_w);
        opConv2d(r, r->dpt_ln, r->dpt_cat, fuse_out_w, fuse_out_b,
                 out_h, out_w, features, features, 1, 1, 1, 0);
        opCopy(r, result, r->dpt_ln, features * out_h * out_w);
    } else {
        opConv2d(r, r->dpt_cat, result, fuse_out_w, fuse_out_b, fH, fW, features, features, 1, 1, 1, 0);
        opCopy(r, result, r->dpt_cat, sz);
    }
}

/* RefineNet fusion with explicit weight pointers (for aux/gsdpt branches) */
static void gpuRefineNetW(vulkan_da3_runner *r,
                            BufInfo &feat, int fH, int fW,
                            BufInfo *deeper, int dH, int dW,
                            int features, BufInfo &result,
                            BufInfo &fuse_out_w, BufInfo &fuse_out_b,
                            BufInfo &rcu1_c1_w, BufInfo &rcu1_c1_b,
                            BufInfo &rcu1_c2_w, BufInfo &rcu1_c2_b,
                            BufInfo &rcu2_c1_w, BufInfo &rcu2_c1_b,
                            BufInfo &rcu2_c2_w, BufInfo &rcu2_c2_b,
                            bool has_rcu1, bool has_rcu2,
                            int out_h, int out_w) {
    opRefineNet(r, feat, fH, fW, deeper, dH, dW, features, result,
                fuse_out_w, fuse_out_b,
                rcu1_c1_w, rcu1_c1_b, rcu1_c2_w, rcu1_c2_b,
                rcu2_c1_w, rcu2_c1_b, rcu2_c2_w, rcu2_c2_b,
                has_rcu1, has_rcu2, out_h, out_w);
}

static void gpuRefineNet(vulkan_da3_runner *r, int stage,
                           BufInfo &feat, int fH, int fW,
                           BufInfo *deeper, int dH, int dW,
                           int features, BufInfo &result,
                           int out_h, int out_w) {
    auto &dw = r->dpt;
    opRefineNet(r, feat, fH, fW, deeper, dH, dW, features, result,
                dw.fuse_out_w[stage], dw.fuse_out_b[stage],
                dw.fuse_rcu1_c1_w[stage], dw.fuse_rcu1_c1_b[stage],
                dw.fuse_rcu1_c2_w[stage], dw.fuse_rcu1_c2_b[stage],
                dw.fuse_rcu2_c1_w[stage], dw.fuse_rcu2_c1_b[stage],
                dw.fuse_rcu2_c2_w[stage], dw.fuse_rcu2_c2_b[stage],
                dw.has_rcu1[stage], dw.has_rcu2[stage],
                out_h, out_w);
}

/* ======================================================================== */
/* Pipeline creation                                                        */
/* ======================================================================== */

static bool createPipelines(vulkan_da3_runner *r) {
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
        fprintf(stderr, "DA3: failed to load shader: %s.spv (searched %s)\n",
                name.c_str(), r->shader_dir.c_str());
        return {};
    };

    auto createPipe = [&](const std::string &name, int nBuf, int pushSize, Pipeline &pipe) -> bool {
        auto spirv = loadSpv(name);
        if (spirv.empty()) return false;
        return r->runner.createComputePipelineWithPushConstants(spirv, makeBindings(nBuf), pushSize, pipe);
    };

    /* Common/reused shaders */
    if (!createPipe("shaders/layernorm_f32",           4, 12, r->pipe_layernorm)) return false;
    if (!createPipe("shaders/matmul_bias_f32",         4, 16, r->pipe_matmul_bias)) return false;
    if (!createPipe("shaders/gelu_f32",                1, 4,  r->pipe_gelu)) return false;
    if (!createPipe("shaders/add_f32",                 2, 4,  r->pipe_add)) return false;
    if (!createPipe("shaders/hy3d/layerscale_add_f32", 3, 8,  r->pipe_layerscale_add)) return false;
    if (!createPipe("shaders/hy3d/qk_layernorm_f32",   3, 20, r->pipe_qk_layernorm)) return false;
    if (!createPipe("shaders/da3/qk_layernorm_offset_f32", 3, 24, r->pipe_qk_layernorm_offset)) return false;
    if (!createPipe("shaders/da3/patch_embed_conv2d_f32", 4, 20, r->pipe_patch_embed)) return false;

    /* DA3-specific shaders */
    if (!createPipe("shaders/da3/resize_normalize_f32",       2, 40, r->pipe_resize_normalize)) return false;
    if (!createPipe("shaders/da3/cls_pos_embed_f32",          3, 8,  r->pipe_cls_pos_embed)) return false;
    if (!createPipe("shaders/da3/rope_2d_f32",                3, 24, r->pipe_rope_2d)) return false;
    if (!createPipe("shaders/da3/swiglu_f32",                 2, 8,  r->pipe_swiglu)) return false;
    if (!createPipe("shaders/da3/kv_transpose_f32",           3, 16, r->pipe_kv_transpose)) return false;
    if (!createPipe("shaders/da3/flash_attn_tiled_f32",       4, 20, r->pipe_flash_attn)) return false;
    if (!createPipe("shaders/da3/cat_local_global_f32",       3, 8,  r->pipe_cat_local_global)) return false;
    if (!createPipe("shaders/da3/dpt_cls_concat_f32",         2, 8,  r->pipe_dpt_cls_concat)) return false;
    if (!createPipe("shaders/da3/strided_layernorm_f32",      3, 16, r->pipe_strided_layernorm)) return false;
    if (!createPipe("shaders/da3/dpt_tok_to_chw_f32",         2, 12, r->pipe_tok_to_chw)) return false;
    if (!createPipe("shaders/da3/deconv_scatter_f32",          3, 36, r->pipe_deconv_scatter)) return false;
    if (!createPipe("shaders/da3/conv2d_f32",                  4, 32, r->pipe_conv2d)) return false;
    if (!createPipe("shaders/da3/bilinear_upsample_f32",       2, 20, r->pipe_bilinear)) return false;
    if (!createPipe("shaders/da3/depth_activation_f32",        1, 4,  r->pipe_depth_activation)) return false;
    if (!createPipe("shaders/da3/sinusoidal_uv_posembed_f32",  1, 24, r->pipe_sinusoidal_uv)) return false;
    if (!createPipe("shaders/da3/relu_f32",                    1, 4,  r->pipe_relu)) return false;
    if (!createPipe("shaders/da3/channel_layernorm_f32",       4, 12, r->pipe_channel_layernorm)) return false;
    if (!createPipe("shaders/da3/silu_f32",                    1, 4,  r->pipe_silu)) return false;
    if (!createPipe("shaders/da3/matmul_bias_f16_f32",        4, 16, r->pipe_matmul_bias_f16)) return false;
    if (!createPipe("shaders/da3/copy_f32",                   2, 4,  r->pipe_copy)) return false;

    if (r->verbose >= 1) fprintf(stderr, "DA3: all pipelines created\n");
    return true;
}

/* ======================================================================== */
/* Safetensors name mapping (same logic as HIP version)                     */
/* ======================================================================== */

typedef struct { const char *st_suffix; const char *gguf_name; } st_name_map;

static st_name_map *build_st_name_map(const st_context *st, int *out_count) {
    const char *bb_prefix = nullptr;
    const char *hd_prefix = nullptr;
    static const char *bb_candidates[] = {
        "model.da3.backbone.pretrained.", "model.backbone.pretrained.",
        "backbone.pretrained.", "backbone.", "pretrained.", "encoder.", nullptr
    };
    static const char *hd_candidates[] = {
        "model.da3.head.", "model.head.", "head.", "depth_head.", "dpt_head.", nullptr
    };
    for (int c = 0; bb_candidates[c]; c++) {
        for (int i = 0; i < st->n_tensors; i++) {
            if (strstr(safetensors_name(st, i), "blocks.") &&
                strncmp(safetensors_name(st, i), bb_candidates[c], strlen(bb_candidates[c])) == 0) {
                bb_prefix = bb_candidates[c];
                goto bb_found;
            }
        }
    }
bb_found:
    for (int c = 0; hd_candidates[c]; c++) {
        for (int i = 0; i < st->n_tensors; i++) {
            if (strstr(safetensors_name(st, i), "projects.") &&
                strncmp(safetensors_name(st, i), hd_candidates[c], strlen(hd_candidates[c])) == 0) {
                hd_prefix = hd_candidates[c];
                goto hd_found;
            }
        }
    }
hd_found:

    static const struct { const char *st; const char *gg; } blk_map[] = {
        {"norm1.weight","ln1.weight"},{"norm1.bias","ln1.bias"},
        {"attn.qkv.weight","attn_qkv.weight"},{"attn.qkv.bias","attn_qkv.bias"},
        {"attn.q_norm.weight","attn_q_norm.weight"},{"attn.q_norm.bias","attn_q_norm.bias"},
        {"attn.k_norm.weight","attn_k_norm.weight"},{"attn.k_norm.bias","attn_k_norm.bias"},
        {"attn.proj.weight","attn_out.weight"},{"attn.proj.bias","attn_out.bias"},
        {"ls1.gamma","ls1"},{"ls2.gamma","ls2"},
        {"norm2.weight","ln2.weight"},{"norm2.bias","ln2.bias"},
        {"mlp.w12.weight","ffn_gate_up.weight"},{"mlp.w12.bias","ffn_gate_up.bias"},
        {"mlp.w3.weight","ffn_down.weight"},{"mlp.w3.bias","ffn_down.bias"},
        {"mlp.fc1.weight","ffn_up.weight"},{"mlp.fc1.bias","ffn_up.bias"},
        {"mlp.fc2.weight","ffn_down.weight"},{"mlp.fc2.bias","ffn_down.bias"},
        {nullptr, nullptr}
    };

    static const struct { const char *st; const char *gg; } rn_map[] = {
        {"out_conv.weight","out.weight"},{"out_conv.bias","out.bias"},
        {"resConfUnit1.conv1.weight","rcu1.conv1.weight"},{"resConfUnit1.conv1.bias","rcu1.conv1.bias"},
        {"resConfUnit1.conv2.weight","rcu1.conv2.weight"},{"resConfUnit1.conv2.bias","rcu1.conv2.bias"},
        {"resConfUnit2.conv1.weight","rcu2.conv1.weight"},{"resConfUnit2.conv1.bias","rcu2.conv1.bias"},
        {"resConfUnit2.conv2.weight","rcu2.conv2.weight"},{"resConfUnit2.conv2.bias","rcu2.conv2.bias"},
        {nullptr, nullptr}
    };

    static const struct { const char *st; const char *gg; } cam_dec_map[] = {
        {"backbone.0.weight","cam_dec.mlp.0.weight"},{"backbone.0.bias","cam_dec.mlp.0.bias"},
        {"backbone.2.weight","cam_dec.mlp.2.weight"},{"backbone.2.bias","cam_dec.mlp.2.bias"},
        {"fc_t.weight","cam_dec.fc_t.weight"},{"fc_t.bias","cam_dec.fc_t.bias"},
        {"fc_qvec.weight","cam_dec.fc_qvec.weight"},{"fc_qvec.bias","cam_dec.fc_qvec.bias"},
        {"fc_fov.0.weight","cam_dec.fc_fov.weight"},{"fc_fov.0.bias","cam_dec.fc_fov.bias"},
        {nullptr, nullptr}
    };

    static const struct { const char *st; const char *gg; } cam_enc_pose_map[] = {
        {"pose_branch.fc1.weight","cam_enc.fc1.weight"},{"pose_branch.fc1.bias","cam_enc.fc1.bias"},
        {"pose_branch.fc2.weight","cam_enc.fc2.weight"},{"pose_branch.fc2.bias","cam_enc.fc2.bias"},
        {"trunk_norm.weight","cam_enc.trunk_norm.weight"},{"trunk_norm.bias","cam_enc.trunk_norm.bias"},
        {"token_norm.weight","cam_enc.token_norm.weight"},{"token_norm.bias","cam_enc.token_norm.bias"},
        {nullptr, nullptr}
    };

    /* Allocate enough space (2x for safety since we add more mappings now) */
    st_name_map *map = (st_name_map *)calloc((size_t)st->n_tensors * 2, sizeof(st_name_map));
    int n = 0;

    for (int i = 0; i < st->n_tensors; i++) {
        const char *key = safetensors_name(st, i);
        char gguf_name[256] = {0};

        if (bb_prefix && strncmp(key, bb_prefix, strlen(bb_prefix)) == 0) {
            const char *s = key + strlen(bb_prefix);
            if (strcmp(s, "cls_token") == 0) strcpy(gguf_name, "da3.cls_token");
            else if (strcmp(s, "camera_token") == 0) strcpy(gguf_name, "da3.camera_token");
            else if (strcmp(s, "pos_embed") == 0) strcpy(gguf_name, "da3.pos_embed");
            else if (strcmp(s, "patch_embed.proj.weight") == 0) strcpy(gguf_name, "da3.patch_embed.weight");
            else if (strcmp(s, "patch_embed.proj.bias") == 0) strcpy(gguf_name, "da3.patch_embed.bias");
            else if (strcmp(s, "norm.weight") == 0) strcpy(gguf_name, "da3.backbone_norm.weight");
            else if (strcmp(s, "norm.bias") == 0) strcpy(gguf_name, "da3.backbone_norm.bias");
            else if (strncmp(s, "blocks.", 7) == 0) {
                int L_idx = 0;
                const char *rest = s + 7;
                while (*rest >= '0' && *rest <= '9') { L_idx = L_idx * 10 + (*rest - '0'); rest++; }
                if (*rest == '.') rest++;
                for (int j = 0; blk_map[j].st; j++) {
                    if (strcmp(rest, blk_map[j].st) == 0) {
                        snprintf(gguf_name, sizeof(gguf_name), "da3.blk.%d.%s", L_idx, blk_map[j].gg);
                        break;
                    }
                }
            }
        }
        /* CameraDec: model.da3.cam_dec.* or model.cam_dec.* */
        else if (strncmp(key, "model.da3.cam_dec.", 18) == 0 || strncmp(key, "model.cam_dec.", 14) == 0) {
            const char *s = strncmp(key, "model.da3.", 10) == 0 ? key + 18 : key + 14;
            for (int j = 0; cam_dec_map[j].st; j++) {
                if (strcmp(s, cam_dec_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.%s", cam_dec_map[j].gg); break; }
            }
        }
        /* CameraEnc: model.da3.cam_enc.* or model.cam_enc.* */
        else if (strncmp(key, "model.da3.cam_enc.", 18) == 0 || strncmp(key, "model.cam_enc.", 14) == 0) {
            const char *s = strncmp(key, "model.da3.", 10) == 0 ? key + 18 : key + 14;
            for (int j = 0; cam_enc_pose_map[j].st; j++) {
                if (strcmp(s, cam_enc_pose_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.%s", cam_enc_pose_map[j].gg); break; }
            }
            if (!gguf_name[0] && strncmp(s, "trunk.", 6) == 0) {
                int L_idx = 0;
                const char *rest = s + 6;
                while (*rest >= '0' && *rest <= '9') { L_idx = L_idx * 10 + (*rest - '0'); rest++; }
                if (*rest == '.') rest++;
                for (int j = 0; blk_map[j].st; j++) {
                    if (strcmp(rest, blk_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.cam_enc.trunk.%d.%s", L_idx, blk_map[j].gg); break; }
                }
            }
        }
        /* GSDPT: model.da3.gs_head.* or model.gs_head.* */
        else if (strncmp(key, "model.da3.gs_head.", 18) == 0 || strncmp(key, "model.gs_head.", 14) == 0) {
            const char *s = strncmp(key, "model.da3.", 10) == 0 ? key + 18 : key + 14;
            if (strncmp(s, "images_merger.", 14) == 0) {
                const char *ms = s + 14;
                int idx = -1;
                if (ms[0] >= '0' && ms[0] <= '9') { idx = ms[0] - '0'; if (idx == 2) idx = 1; else if (idx == 4) idx = 2; else if (idx != 0) idx = -1; }
                if (idx >= 0 && ms[1] == '.') snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.merger.%d.%s", idx, ms + 2);
            } else if (strcmp(s, "norm.weight") == 0) strcpy(gguf_name, "da3.gsdpt.head.norm.weight");
            else if (strcmp(s, "norm.bias") == 0) strcpy(gguf_name, "da3.gsdpt.head.norm.bias");
            else if (strncmp(s, "projects.", 9) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.proj.%d.%s", s[9]-'0', s+11); }
            else if (strncmp(s, "resize_layers.", 14) == 0) {
                int idx = s[14] - '0'; const char *wb = s + 16;
                if (idx == 0) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.upsample_0.%s", wb);
                else if (idx == 1) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.upsample_1.%s", wb);
                else if (idx == 3) snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.downsample.%s", wb);
            } else if (strncmp(s, "scratch.", 8) == 0) {
                const char *ss = s + 8;
                for (int li = 1; li <= 4; li++) { char pfx[32]; snprintf(pfx, sizeof(pfx), "layer%d_rn.weight", li);
                    if (strcmp(ss, pfx) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.adapter.%d.weight", li-1); break; } }
                if (!gguf_name[0]) { for (int ri = 1; ri <= 4; ri++) { char pfx[32]; snprintf(pfx, sizeof(pfx), "refinenet%d.", ri); size_t plen = strlen(pfx);
                    if (strncmp(ss, pfx, plen) == 0) { const char *rn = ss + plen;
                        for (int j = 0; rn_map[j].st; j++) { if (strcmp(rn, rn_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.gsdpt.head.fuse.%d.%s", ri-1, rn_map[j].gg); break; } } break; } } }
                if (!gguf_name[0]) { static const struct { const char *st; const char *gg; } gsout_map[] = {
                    {"output_conv1.weight","da3.gsdpt.head.neck.weight"},{"output_conv1.bias","da3.gsdpt.head.neck.bias"},
                    {"output_conv2.0.weight","da3.gsdpt.head.out_0.weight"},{"output_conv2.0.bias","da3.gsdpt.head.out_0.bias"},
                    {"output_conv2.2.weight","da3.gsdpt.head.out_2.weight"},{"output_conv2.2.bias","da3.gsdpt.head.out_2.bias"},
                    {nullptr, nullptr} };
                    for (int j = 0; gsout_map[j].st; j++) { if (strcmp(ss, gsout_map[j].st) == 0) { strcpy(gguf_name, gsout_map[j].gg); break; } } }
            }
        }
        else if (hd_prefix && strncmp(key, hd_prefix, strlen(hd_prefix)) == 0) {
            const char *s = key + strlen(hd_prefix);
            if (strcmp(s, "norm.weight") == 0) strcpy(gguf_name, "da3.head.norm.weight");
            else if (strcmp(s, "norm.bias") == 0) strcpy(gguf_name, "da3.head.norm.bias");
            else if (strncmp(s, "projects.", 9) == 0) {
                snprintf(gguf_name, sizeof(gguf_name), "da3.head.proj.%d.%s", s[9] - '0', s + 11);
            }
            else if (strncmp(s, "resize_layers.", 14) == 0) {
                int idx = s[14] - '0'; const char *wb = s + 16;
                if (idx == 0) snprintf(gguf_name, sizeof(gguf_name), "da3.head.upsample_0.%s", wb);
                else if (idx == 1) snprintf(gguf_name, sizeof(gguf_name), "da3.head.upsample_1.%s", wb);
                else if (idx == 3) snprintf(gguf_name, sizeof(gguf_name), "da3.head.downsample.%s", wb);
            }
            else if (strncmp(s, "scratch.", 8) == 0) {
                const char *ss = s + 8;
                for (int li = 1; li <= 4; li++) {
                    char pfx[32]; snprintf(pfx, sizeof(pfx), "layer%d_rn.weight", li);
                    if (strcmp(ss, pfx) == 0) {
                        snprintf(gguf_name, sizeof(gguf_name), "da3.head.adapter.%d.weight", li - 1);
                        break;
                    }
                }
                if (!gguf_name[0]) {
                    for (int ri = 1; ri <= 4; ri++) {
                        char pfx[32]; snprintf(pfx, sizeof(pfx), "refinenet%d.", ri);
                        size_t plen = strlen(pfx);
                        if (strncmp(ss, pfx, plen) == 0) {
                            const char *rn = ss + plen;
                            for (int j = 0; rn_map[j].st; j++) {
                                if (strcmp(rn, rn_map[j].st) == 0) {
                                    snprintf(gguf_name, sizeof(gguf_name), "da3.head.fuse.%d.%s", ri - 1, rn_map[j].gg);
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                /* Aux RefineNet: refinenetN_aux.* */
                if (!gguf_name[0]) { for (int ri = 1; ri <= 4; ri++) { char pfx[32]; snprintf(pfx, sizeof(pfx), "refinenet%d_aux.", ri); size_t plen = strlen(pfx);
                    if (strncmp(ss, pfx, plen) == 0) { const char *rn = ss + plen;
                        for (int j = 0; rn_map[j].st; j++) { if (strcmp(rn, rn_map[j].st) == 0) { snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_fuse.%d.%s", ri-1, rn_map[j].gg); break; } } break; } } }
                /* Aux output_conv1_aux: output_conv1_aux.LEVEL.CONV_IDX.weight/bias */
                if (!gguf_name[0] && strncmp(ss, "output_conv1_aux.", 17) == 0) {
                    int level = ss[17]-'0';
                    if (level >= 0 && level < 4 && ss[18] == '.') { int ci = ss[19]-'0';
                        if (ci >= 0 && ci <= 4 && ss[20] == '.') snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_oc1.%d.%d.%s", level, ci, ss+21); } }
                /* Aux output_conv2_aux: output_conv2_aux.LEVEL.STAGE.weight/bias */
                if (!gguf_name[0] && strncmp(ss, "output_conv2_aux.", 17) == 0) {
                    int level = ss[17]-'0';
                    if (level >= 0 && level < 4 && ss[18] == '.') { int si = ss[19]-'0';
                        if (ss[20] == '.') { const char *wb = ss + 21;
                            if (si == 0) snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_oc2.%d.conv.%s", level, wb);
                            else if (si == 2) snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_oc2.%d.gn.%s", level, wb);
                            else if (si == 5) snprintf(gguf_name, sizeof(gguf_name), "da3.head.aux_oc2.%d.out.%s", level, wb); } } }
                /* Main output convs (not aux) */
                if (!gguf_name[0] && !strstr(ss, "_aux")) {
                    static const struct { const char *st; const char *gg; } out_map[] = {
                        {"output_conv1.weight","da3.head.neck.weight"},
                        {"output_conv1.bias","da3.head.neck.bias"},
                        {"output_conv2.0.weight","da3.head.out_0.weight"},
                        {"output_conv2.0.bias","da3.head.out_0.bias"},
                        {"output_conv2.2.weight","da3.head.out_2.weight"},
                        {"output_conv2.2.bias","da3.head.out_2.bias"},
                        {nullptr, nullptr}
                    };
                    for (int j = 0; out_map[j].st; j++) {
                        if (strcmp(ss, out_map[j].st) == 0) {
                            strcpy(gguf_name, out_map[j].gg);
                            break;
                        }
                    }
                }
            }
        }

        if (gguf_name[0]) {
            map[n].st_suffix = key;
            map[n].gguf_name = strdup(gguf_name);
            n++;
        }
    }

    *out_count = n;
    return map;
}

/* Find safetensors tensor by mapped gguf_name, return info */
struct da3_tensor_info {
    void *data;
    size_t nbytes;
    const char *dtype;
    int n_dims;
    uint64_t shape[8]; /* reversed: shape[0]=cols, shape[1]=rows etc. */
    int n_cols, n_rows;
    bool valid;
};

static da3_tensor_info find_st_tensor(const st_context *st, const st_name_map *map, int map_count,
                                       const char *gguf_name) {
    da3_tensor_info info = {};
    for (int m = 0; m < map_count; m++) {
        if (strcmp(map[m].gguf_name, gguf_name) != 0) continue;
        int si = safetensors_find(st, map[m].st_suffix);
        if (si < 0) break;
        info.data = safetensors_data(st, si);
        info.nbytes = safetensors_nbytes(st, si);
        info.dtype = safetensors_dtype(st, si);
        info.n_dims = safetensors_ndims(st, si);
        const uint64_t *sh = safetensors_shape(st, si);
        for (int d = 0; d < info.n_dims; d++)
            info.shape[d] = sh[info.n_dims - 1 - d]; /* reverse: cols first */
        info.n_cols = (int)info.shape[0];
        info.n_rows = (info.n_dims >= 2) ? (int)info.shape[1] : 1;
        info.valid = true;
        break;
    }
    return info;
}

/* Upload tensor from safetensors to GPU as F32 */
static BufInfo stUploadF32Mapped(VulkanRunner &runner, const st_context *st,
                                   const st_name_map *map, int map_count,
                                   const char *gguf_name, int verbose) {
    da3_tensor_info ti = find_st_tensor(st, map, map_count, gguf_name);
    if (!ti.valid) {
        if (verbose >= 2) fprintf(stderr, "  [WARN] mapped tensor '%s' not found\n", gguf_name);
        return BufInfo{};
    }

    size_t nbytes = ti.nbytes;
    BufInfo buf{};

    if (strcmp(ti.dtype, "F32") == 0) {
        buf = createGpuBuffer(runner, nbytes);
        runner.uploadToDeviceLocal(buf, ti.data, nbytes);
    } else if (strcmp(ti.dtype, "F16") == 0) {
        size_t n = nbytes / sizeof(uint16_t);
        std::vector<float> f32(n);
        const uint16_t *f16 = (const uint16_t *)ti.data;
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
        runner.uploadToDeviceLocal(buf, f32.data(), n * sizeof(float));
    } else if (strcmp(ti.dtype, "BF16") == 0) {
        size_t n = nbytes / sizeof(uint16_t);
        std::vector<float> f32(n);
        const uint16_t *bf16 = (const uint16_t *)ti.data;
        for (size_t i = 0; i < n; i++) {
            uint32_t f = (uint32_t)bf16[i] << 16;
            memcpy(&f32[i], &f, sizeof(float));
        }
        buf = createGpuBuffer(runner, n * sizeof(float));
        runner.uploadToDeviceLocal(buf, f32.data(), n * sizeof(float));
    }
    return buf;
}

/* Upload tensor from safetensors to GPU as F16 (uint16 storage).
 * If source is F32, convert to F16 on CPU. If already F16, upload directly. */
static BufInfo stUploadF16Mapped(VulkanRunner &runner, const st_context *st,
                                   const st_name_map *map, int map_count,
                                   const char *gguf_name, int verbose) {
    da3_tensor_info ti = find_st_tensor(st, map, map_count, gguf_name);
    if (!ti.valid) {
        if (verbose >= 2) fprintf(stderr, "  [WARN] mapped tensor '%s' not found\n", gguf_name);
        return BufInfo{};
    }

    size_t n_elements = ti.nbytes / (strcmp(ti.dtype, "F16") == 0 ? 2 : 4);
    size_t f16_bytes = n_elements * sizeof(uint16_t);
    BufInfo buf = createGpuBuffer(runner, f16_bytes);

    if (strcmp(ti.dtype, "F16") == 0) {
        /* Already F16, upload directly */
        runner.uploadToDeviceLocal(buf, ti.data, f16_bytes);
    } else if (strcmp(ti.dtype, "F32") == 0) {
        /* Convert F32 -> F16 on CPU (matching HIP's hip_f32_to_f16 with denormals) */
        const float *f32 = (const float *)ti.data;
        std::vector<uint16_t> f16(n_elements);
        for (size_t i = 0; i < n_elements; i++) {
            uint32_t bits; memcpy(&bits, &f32[i], 4);
            uint16_t sign = (uint16_t)((bits >> 16) & 0x8000);
            int32_t exp = ((bits >> 23) & 0xff) - 127;
            uint32_t mant = bits & 0x7fffff;
            if (exp > 15) { f16[i] = sign | 0x7c00; }
            else if (exp < -14) {
                if (exp < -24) { f16[i] = sign; }
                else { mant |= 0x800000; mant >>= (-1 - exp); f16[i] = sign | (uint16_t)(mant >> 13); }
            }
            else { f16[i] = sign | (uint16_t)((exp + 15) << 10) | (uint16_t)(mant >> 13); }
        }
        runner.uploadToDeviceLocal(buf, f16.data(), f16_bytes);
    } else if (strcmp(ti.dtype, "BF16") == 0) {
        /* Convert BF16 -> F32 -> F16 on CPU */
        const uint16_t *bf16 = (const uint16_t *)ti.data;
        std::vector<uint16_t> f16(n_elements);
        for (size_t i = 0; i < n_elements; i++) {
            uint32_t f32_bits = (uint32_t)bf16[i] << 16;
            float fv; memcpy(&fv, &f32_bits, 4);
            uint32_t bits; memcpy(&bits, &fv, 4);
            uint32_t sign = (bits >> 31) & 1;
            int32_t exp = ((bits >> 23) & 0xff) - 127;
            uint32_t mant = bits & 0x7fffff;
            uint16_t h;
            if (exp > 15) h = (sign << 15) | 0x7c00;
            else if (exp < -14) h = (sign << 15);
            else h = (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
            f16[i] = h;
        }
        runner.uploadToDeviceLocal(buf, f16.data(), f16_bytes);
    }
    return buf;
}

/* F16 half-float to F32 conversion */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (exp == 0) f = sign << 31;
    else if (exp == 31) f = (sign << 31) | 0x7f800000 | (mant << 13);
    else f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

/* Upload ConvTranspose2d weight transposed for GEMM-based deconv (as F32).
 * Input layout: [Ci, Co, kH, kW].
 * Output layout: [kH*kW*Co, Ci] as F32 (GEMM W matrix). */
static BufInfo uploadDeconvWeightF32(VulkanRunner &runner, const st_context *st,
                                       const st_name_map *map, int map_count,
                                       const char *gguf_name, int Ci, int Co, int kH, int kW,
                                       int verbose) {
    da3_tensor_info ti = find_st_tensor(st, map, map_count, gguf_name);
    if (!ti.valid) return BufInfo{};

    int N = kH * kW * Co;
    int K = Ci;
    size_t total = (size_t)Ci * Co * kH * kW;

    /* Get F32 source */
    std::vector<float> f32_src;
    const float *src_f32 = nullptr;

    if (strcmp(ti.dtype, "F32") == 0) {
        src_f32 = (const float *)ti.data;
    } else if (strcmp(ti.dtype, "F16") == 0) {
        f32_src.resize(total);
        const uint16_t *src16 = (const uint16_t *)ti.data;
        for (size_t i = 0; i < total; i++) f32_src[i] = f16_to_f32(src16[i]);
        src_f32 = f32_src.data();
    } else if (strcmp(ti.dtype, "BF16") == 0) {
        f32_src.resize(total);
        const uint16_t *bf16 = (const uint16_t *)ti.data;
        for (size_t i = 0; i < total; i++) {
            uint32_t f = (uint32_t)bf16[i] << 16;
            memcpy(&f32_src[i], &f, sizeof(float));
        }
        src_f32 = f32_src.data();
    } else {
        return BufInfo{};
    }

    /* Transpose [Ci, Co, kH, kW] -> [kH*kW*Co, Ci] as F32 */
    std::vector<float> transposed((size_t)N * K);
    for (int ci = 0; ci < Ci; ci++) {
        for (int co = 0; co < Co; co++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw_i = 0; kw_i < kW; kw_i++) {
                    int src_idx = ci * (Co * kH * kW) + co * (kH * kW) + kh * kW + kw_i;
                    int dst_row = (kh * kW + kw_i) * Co + co;
                    int dst_idx = dst_row * Ci + ci;
                    transposed[dst_idx] = src_f32[src_idx];
                }
            }
        }
    }

    BufInfo buf = createGpuBuffer(runner, (size_t)N * K * sizeof(float));
    runner.uploadToDeviceLocal(buf, transposed.data(), (size_t)N * K * sizeof(float));
    return buf;
}

/* Upload ConvTranspose2d weight transposed for GEMM-based deconv (as F16).
 * Input layout: [Ci, Co, kH, kW].
 * Output layout: [kH*kW*Co, Ci] as F16 (GEMM W matrix). */
static BufInfo uploadDeconvWeightF16(VulkanRunner &runner, const st_context *st,
                                       const st_name_map *map, int map_count,
                                       const char *gguf_name, int Ci, int Co, int kH, int kW,
                                       int verbose) {
    da3_tensor_info ti = find_st_tensor(st, map, map_count, gguf_name);
    if (!ti.valid) return BufInfo{};

    int N = kH * kW * Co;
    int K = Ci;
    size_t total = (size_t)Ci * Co * kH * kW;

    /* Get F32 source */
    std::vector<float> f32_src;
    const float *src_f32 = nullptr;

    if (strcmp(ti.dtype, "F32") == 0) {
        src_f32 = (const float *)ti.data;
    } else if (strcmp(ti.dtype, "F16") == 0) {
        f32_src.resize(total);
        const uint16_t *src16 = (const uint16_t *)ti.data;
        for (size_t i = 0; i < total; i++) f32_src[i] = f16_to_f32(src16[i]);
        src_f32 = f32_src.data();
    } else if (strcmp(ti.dtype, "BF16") == 0) {
        f32_src.resize(total);
        const uint16_t *bf16 = (const uint16_t *)ti.data;
        for (size_t i = 0; i < total; i++) {
            uint32_t f = (uint32_t)bf16[i] << 16;
            memcpy(&f32_src[i], &f, sizeof(float));
        }
        src_f32 = f32_src.data();
    } else {
        return BufInfo{};
    }

    /* Transpose [Ci, Co, kH, kW] -> [kH*kW*Co, Ci] then convert to F16 */
    std::vector<uint16_t> transposed_f16((size_t)N * K);
    for (int ci = 0; ci < Ci; ci++) {
        for (int co = 0; co < Co; co++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw_i = 0; kw_i < kW; kw_i++) {
                    int src_idx = ci * (Co * kH * kW) + co * (kH * kW) + kh * kW + kw_i;
                    int dst_row = (kh * kW + kw_i) * Co + co;
                    int dst_idx = dst_row * Ci + ci;
                    float fv = src_f32[src_idx];
                    uint32_t bits; memcpy(&bits, &fv, 4);
                    uint16_t s16 = (uint16_t)((bits >> 16) & 0x8000);
                    int32_t exp = ((bits >> 23) & 0xff) - 127;
                    uint32_t mant = bits & 0x7fffff;
                    uint16_t h;
                    if (exp > 15) h = s16 | 0x7c00;
                    else if (exp < -14) {
                        if (exp < -24) h = s16;
                        else { mant |= 0x800000; mant >>= (-1 - exp); h = s16 | (uint16_t)(mant >> 13); }
                    }
                    else h = s16 | (uint16_t)((exp + 15) << 10) | (uint16_t)(mant >> 13);
                    transposed_f16[dst_idx] = h;
                }
            }
        }
    }

    BufInfo buf = createGpuBuffer(runner, (size_t)N * K * sizeof(uint16_t));
    runner.uploadToDeviceLocal(buf, transposed_f16.data(), (size_t)N * K * sizeof(uint16_t));
    return buf;
}

/* ======================================================================== */
/* Public API: load_safetensors                                             */
/* ======================================================================== */

int vulkan_da3_load_safetensors(vulkan_da3_runner *r, const char *st_path, const char *config_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "DA3: cannot open safetensors: %s\n", st_path); return -1; }

    if (r->verbose >= 1)
        fprintf(stderr, "DA3: safetensors: %d tensors from %s\n", st->n_tensors, st_path);

    int map_count = 0;
    st_name_map *map = build_st_name_map(st, &map_count);
    if (r->verbose >= 1) fprintf(stderr, "DA3: mapped %d tensors\n", map_count);

    /* Detect model params from tensor shapes */
    int embed_dim = 384, n_heads = 6, head_dim_val = 64, n_blocks = 12, ffn_hidden = 0;
    int head_features = 64;
    int head_oc[4] = {48, 96, 192, 384};
    int feature_layers[4] = {5, 7, 9, 11};
    int rope_start = 4, qknorm_start = 4;
    int has_swiglu = 0;

    /* Parse config.json if provided */
    if (config_path) {
        FILE *f = fopen(config_path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            char *buf = (char *)malloc(sz);
            size_t nr = fread(buf, 1, sz, f); (void)nr;
            fclose(f);
            json_val *root = json_parse(buf, (int)sz);
            if (root) {
                json_val *cfg = json_obj_get(root, "config");
                if (cfg) {
                    json_val *anyview = json_obj_get(cfg, "anyview");
                    json_val *cfg_src = anyview ? anyview : cfg;
                    json_val *head = json_obj_get(cfg_src, "head");
                    if (head) {
                        json_val *v = json_obj_get(head, "features");
                        if (v && v->type == JSON_NUMBER) head_features = (int)v->num;
                        json_val *oc_arr = json_obj_get(head, "out_channels");
                        if (oc_arr && oc_arr->type == JSON_ARRAY)
                            for (int i = 0; i < 4 && i < oc_arr->arr.count; i++)
                                head_oc[i] = (int)oc_arr->arr.items[i].num;
                    }
                    json_val *net = json_obj_get(cfg_src, "net");
                    if (net) {
                        json_val *ol = json_obj_get(net, "out_layers");
                        if (ol && ol->type == JSON_ARRAY)
                            for (int i = 0; i < 4 && i < ol->arr.count; i++)
                                feature_layers[i] = (int)ol->arr.items[i].num;
                        json_val *rs = json_obj_get(net, "rope_start");
                        if (rs && rs->type == JSON_NUMBER) rope_start = (int)rs->num;
                        json_val *qs = json_obj_get(net, "qknorm_start");
                        if (qs && qs->type == JSON_NUMBER) qknorm_start = (int)qs->num;
                    }
                }
                json_free(root);
            }
            free(buf);
        }
    }

    /* Detect from tensor shapes */
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "patch_embed.proj.weight")) { embed_dim = (int)safetensors_shape(st, i)[0]; break; }
    }
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "attn.q_norm.weight") && !strstr(nm, "_aux")) {
            int hd = (int)safetensors_shape(st, i)[0]; n_heads = embed_dim / hd; head_dim_val = hd; break;
        }
    }
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "blocks.0.mlp.w12.weight")) { ffn_hidden = (int)safetensors_shape(st, i)[0] / 2; has_swiglu = 1; break; }
        if (strstr(nm, "blocks.0.mlp.fc1.weight")) { ffn_hidden = (int)safetensors_shape(st, i)[0]; break; }
    }
    n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *p = strstr(nm, "blocks.");
        if (p && !strstr(nm, "_aux")) {
            p += 7; int blk = 0;
            while (*p >= '0' && *p <= '9') { blk = blk * 10 + (*p - '0'); p++; }
            if (blk + 1 > n_blocks) n_blocks = blk + 1;
        }
    }

    r->dim = embed_dim; r->n_heads = n_heads; r->head_dim = head_dim_val;
    r->n_blocks = n_blocks; r->ffn_hidden = ffn_hidden;
    r->patch_size = 14; r->image_size = 518; r->ln_eps = 1e-6f;
    r->rope_start = rope_start; r->qk_norm_start = qknorm_start;
    r->head_features = head_features; r->use_swiglu = has_swiglu;
    r->grid_h = r->image_size / r->patch_size; r->grid_w = r->grid_h;
    r->n_patches = r->grid_h * r->grid_w; r->n_tokens = r->n_patches + 1;
    for (int i = 0; i < 4; i++) r->feature_layers[i] = feature_layers[i];
    for (int i = 0; i < 4; i++) r->head_out_channels[i] = head_oc[i];
    r->image_mean[0] = 0.485f; r->image_mean[1] = 0.456f; r->image_mean[2] = 0.406f;
    r->image_std[0] = 0.229f; r->image_std[1] = 0.224f; r->image_std[2] = 0.225f;

    if (r->verbose >= 1) {
        fprintf(stderr, "DA3: model: dim=%d, n_heads=%d, head_dim=%d, n_blocks=%d, ffn=%d, swiglu=%d\n",
                embed_dim, n_heads, head_dim_val, n_blocks, ffn_hidden, has_swiglu);
    }

    int dim = r->dim;

    /* Upload backbone embeddings */
    r->cls_token = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cls_token", r->verbose);
    r->pos_embed = stUploadF32Mapped(r->runner, st, map, map_count, "da3.pos_embed", r->verbose);

    /* Save CPU copy of pos_embed for interpolation */
    if (isValid(r->pos_embed)) {
        size_t pe_sz = (size_t)(r->n_patches + 1) * dim;
        r->h_pos_embed_orig = (float *)malloc(pe_sz * sizeof(float));
        r->runner.downloadFromDeviceLocal(r->pos_embed, r->h_pos_embed_orig, pe_sz * sizeof(float));
    }

    r->patch_embed_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.patch_embed.weight", r->verbose);
    r->patch_embed_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.patch_embed.bias", r->verbose);

    /* Camera token: [2, dim] -- use index 0 for single-view */
    {
        da3_tensor_info ti = find_st_tensor(st, map, map_count, "da3.camera_token");
        if (ti.valid) {
            std::vector<float> cam_f32(dim);
            if (strcmp(ti.dtype, "F16") == 0) {
                const uint16_t *s16 = (const uint16_t *)ti.data;
                for (int ci = 0; ci < dim; ci++) cam_f32[ci] = f16_to_f32(s16[ci]);
            } else if (strcmp(ti.dtype, "BF16") == 0) {
                const uint16_t *bf16 = (const uint16_t *)ti.data;
                for (int ci = 0; ci < dim; ci++) {
                    uint32_t f = (uint32_t)bf16[ci] << 16;
                    memcpy(&cam_f32[ci], &f, sizeof(float));
                }
            } else {
                memcpy(cam_f32.data(), ti.data, (size_t)dim * sizeof(float));
            }
            r->camera_token = createGpuBuffer(r->runner, (size_t)dim * sizeof(float));
            r->runner.uploadToDeviceLocal(r->camera_token, cam_f32.data(), (size_t)dim * sizeof(float));
            r->has_camera_token = true;
            if (r->verbose >= 1) fprintf(stderr, "DA3: camera_token loaded (%d floats)\n", dim);
        }
    }

    /* Backbone norm */
    r->backbone_norm_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.backbone_norm.weight", r->verbose);
    r->backbone_norm_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.backbone_norm.bias", r->verbose);

    /* Upload transformer blocks */
    int nb = r->n_blocks;
    r->layers.resize(nb);
    for (int L = 0; L < nb; L++) {
        auto &ly = r->layers[L];
        char name[128];

#define VK_F32(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    ly.field = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
#define VK_F16(field, fmt, ...) \
    snprintf(name, sizeof(name), fmt, __VA_ARGS__); \
    ly.field = stUploadF16Mapped(r->runner, st, map, map_count, name, r->verbose);

        VK_F32(ln1_w, "da3.blk.%d.ln1.weight", L)
        VK_F32(ln1_b, "da3.blk.%d.ln1.bias", L)
        VK_F16(attn_qkv_w, "da3.blk.%d.attn_qkv.weight", L)
        { snprintf(name, sizeof(name), "da3.blk.%d.attn_qkv.weight", L);
          da3_tensor_info ti = find_st_tensor(st, map, map_count, name);
          ly.qkv_rows = ti.valid ? ti.n_rows : (3 * dim); }
        VK_F32(attn_qkv_b, "da3.blk.%d.attn_qkv.bias", L)
        VK_F32(attn_q_norm_w, "da3.blk.%d.attn_q_norm.weight", L)
        VK_F32(attn_q_norm_b, "da3.blk.%d.attn_q_norm.bias", L)
        VK_F32(attn_k_norm_w, "da3.blk.%d.attn_k_norm.weight", L)
        VK_F32(attn_k_norm_b, "da3.blk.%d.attn_k_norm.bias", L)
        ly.has_qk_norm = isValid(ly.attn_q_norm_w);
        VK_F16(attn_out_w, "da3.blk.%d.attn_out.weight", L)
        { snprintf(name, sizeof(name), "da3.blk.%d.attn_out.weight", L);
          da3_tensor_info ti = find_st_tensor(st, map, map_count, name);
          ly.out_rows = ti.valid ? ti.n_rows : dim; }
        VK_F32(attn_out_b, "da3.blk.%d.attn_out.bias", L)
        VK_F32(ls1, "da3.blk.%d.ls1", L)
        VK_F32(ls2, "da3.blk.%d.ls2", L)
        VK_F32(ln2_w, "da3.blk.%d.ln2.weight", L)
        VK_F32(ln2_b, "da3.blk.%d.ln2.bias", L)

        VK_F16(ffn_gate_up_w, "da3.blk.%d.ffn_gate_up.weight", L)
        if (isValid(ly.ffn_gate_up_w)) {
            snprintf(name, sizeof(name), "da3.blk.%d.ffn_gate_up.weight", L);
            da3_tensor_info ti = find_st_tensor(st, map, map_count, name);
            ly.ffn_gu_rows = ti.valid ? ti.n_rows : (2 * ffn_hidden);
            ly.has_swiglu = true;
        }
        VK_F32(ffn_gate_up_b, "da3.blk.%d.ffn_gate_up.bias", L)
        VK_F16(ffn_up_w, "da3.blk.%d.ffn_up.weight", L)
        if (isValid(ly.ffn_up_w)) {
            snprintf(name, sizeof(name), "da3.blk.%d.ffn_up.weight", L);
            da3_tensor_info ti = find_st_tensor(st, map, map_count, name);
            ly.ffn_up_rows = ti.valid ? ti.n_rows : ffn_hidden;
        }
        VK_F32(ffn_up_b, "da3.blk.%d.ffn_up.bias", L)
        VK_F16(ffn_down_w, "da3.blk.%d.ffn_down.weight", L)
        { snprintf(name, sizeof(name), "da3.blk.%d.ffn_down.weight", L);
          da3_tensor_info ti = find_st_tensor(st, map, map_count, name);
          ly.ffn_down_rows = ti.valid ? ti.n_rows : dim; }
        VK_F32(ffn_down_b, "da3.blk.%d.ffn_down.bias", L)

#undef VK_F32
#undef VK_F16
    }

    /* Upload DPT head weights */
    {
        auto &dw = r->dpt;
        dw.norm_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.norm.weight", r->verbose);
        dw.norm_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.norm.bias", r->verbose);

        for (int i = 0; i < 4; i++) {
            char name[128];
            snprintf(name, sizeof(name), "da3.head.proj.%d.weight", i);
            dw.proj_w[i] = stUploadF16Mapped(r->runner, st, map, map_count, name, r->verbose);
            { da3_tensor_info ti = find_st_tensor(st, map, map_count, name);
              dw.proj_rows[i] = ti.valid ? ti.n_rows : head_oc[i]; }
            snprintf(name, sizeof(name), "da3.head.proj.%d.bias", i);
            dw.proj_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
        }

        /* Spatial alignment: ConvTranspose2d weights (F16 for GEMM) */
        {
            int oc0 = r->head_out_channels[0];
            dw.upsample_0_w = uploadDeconvWeightF16(r->runner, st, map, map_count,
                                                      "da3.head.upsample_0.weight", oc0, oc0, 4, 4, r->verbose);
            dw.upsample_0_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.upsample_0.bias", r->verbose);
            int oc1 = r->head_out_channels[1];
            dw.upsample_1_w = uploadDeconvWeightF16(r->runner, st, map, map_count,
                                                      "da3.head.upsample_1.weight", oc1, oc1, 2, 2, r->verbose);
            dw.upsample_1_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.upsample_1.bias", r->verbose);
        }
        dw.downsample_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.downsample.weight", r->verbose);
        dw.downsample_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.downsample.bias", r->verbose);

        /* Adapter convolutions */
        for (int i = 0; i < 4; i++) {
            char name[128];
            snprintf(name, sizeof(name), "da3.head.adapter.%d.weight", i);
            dw.adapter_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
        }

        /* RefineNet fusion blocks */
        for (int i = 0; i < 4; i++) {
            char name[128];
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.weight", i);
            dw.fuse_out_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.out.bias", i);
            dw.fuse_out_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);

            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.weight", i);
            dw.fuse_rcu1_c1_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv1.bias", i);
            dw.fuse_rcu1_c1_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.weight", i);
            dw.fuse_rcu1_c2_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu1.conv2.bias", i);
            dw.fuse_rcu1_c2_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            dw.has_rcu1[i] = isValid(dw.fuse_rcu1_c1_w[i]);

            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.weight", i);
            dw.fuse_rcu2_c1_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv1.bias", i);
            dw.fuse_rcu2_c1_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.weight", i);
            dw.fuse_rcu2_c2_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            snprintf(name, sizeof(name), "da3.head.fuse.%d.rcu2.conv2.bias", i);
            dw.fuse_rcu2_c2_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            dw.has_rcu2[i] = isValid(dw.fuse_rcu2_c1_w[i]);
        }

        /* Output convolutions */
        dw.neck_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.neck.weight", r->verbose);
        dw.neck_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.neck.bias", r->verbose);
        dw.out_0_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.out_0.weight", r->verbose);
        dw.out_0_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.out_0.bias", r->verbose);
        { da3_tensor_info ti = find_st_tensor(st, map, map_count, "da3.head.out_0.weight");
          /* out_0.weight is [Co, Ci, kH, kW]; first dim in PyTorch order = shape[n_dims-1] after reversal */
          dw.out_mid = ti.valid ? (int)ti.shape[ti.n_dims - 1] : 32; }
        dw.out_2_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.out_2.weight", r->verbose);
        dw.out_2_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.head.out_2.bias", r->verbose);
    }

    /* Upload CameraDec weights */
    {
        da3_tensor_info ti = find_st_tensor(st, map, map_count, "da3.cam_dec.mlp.0.weight");
        if (ti.valid) {
            auto &cd = r->cam_dec;
            cd.backbone_norm_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.backbone_norm.weight", r->verbose);
            cd.backbone_norm_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.backbone_norm.bias", r->verbose);
            cd.mlp_w[0] = stUploadF16Mapped(r->runner, st, map, map_count, "da3.cam_dec.mlp.0.weight", r->verbose);
            cd.mlp_dim = ti.n_rows;
            cd.mlp_b[0] = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_dec.mlp.0.bias", r->verbose);
            cd.mlp_w[1] = stUploadF16Mapped(r->runner, st, map, map_count, "da3.cam_dec.mlp.2.weight", r->verbose);
            cd.mlp_b[1] = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_dec.mlp.2.bias", r->verbose);
            cd.fc_t_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_dec.fc_t.weight", r->verbose);
            cd.fc_t_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_dec.fc_t.bias", r->verbose);
            cd.fc_qvec_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_dec.fc_qvec.weight", r->verbose);
            cd.fc_qvec_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_dec.fc_qvec.bias", r->verbose);
            cd.fc_fov_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_dec.fc_fov.weight", r->verbose);
            cd.fc_fov_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_dec.fc_fov.bias", r->verbose);
            cd.loaded = true;
            if (r->verbose >= 1) fprintf(stderr, "DA3: CameraDec weights loaded (mlp_dim=%d)\n", cd.mlp_dim);
        }
    }

    /* Upload CameraEnc weights */
    {
        da3_tensor_info ti = find_st_tensor(st, map, map_count, "da3.cam_enc.fc1.weight");
        if (ti.valid) {
            auto &ce = r->cam_enc;
            ce.fc1_w = stUploadF16Mapped(r->runner, st, map, map_count, "da3.cam_enc.fc1.weight", r->verbose);
            ce.trunk_dim = ti.n_rows;
            ce.fc1_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_enc.fc1.bias", r->verbose);
            ce.fc2_w = stUploadF16Mapped(r->runner, st, map, map_count, "da3.cam_enc.fc2.weight", r->verbose);
            ce.fc2_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_enc.fc2.bias", r->verbose);
            ce.trunk_norm_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_enc.trunk_norm.weight", r->verbose);
            ce.trunk_norm_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_enc.trunk_norm.bias", r->verbose);
            ce.token_norm_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_enc.token_norm.weight", r->verbose);
            ce.token_norm_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.cam_enc.token_norm.bias", r->verbose);

            /* Count trunk blocks */
            int ntb = 0;
            for (int L = 0; L < 8; L++) {
                char tname[128];
                snprintf(tname, sizeof(tname), "da3.cam_enc.trunk.%d.ln1.weight", L);
                da3_tensor_info tti = find_st_tensor(st, map, map_count, tname);
                if (!tti.valid) break;
                ntb = L + 1;
            }

            if (ntb > 0) {
                ce.trunk.resize(ntb);
                for (int L = 0; L < ntb; L++) {
                    auto &ly = ce.trunk[L];
                    char name[128];
#define CE_F32(field, fmt, ...) snprintf(name, sizeof(name), fmt, __VA_ARGS__); ly.field = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
#define CE_F16(field, fmt, ...) snprintf(name, sizeof(name), fmt, __VA_ARGS__); ly.field = stUploadF16Mapped(r->runner, st, map, map_count, name, r->verbose);
                    CE_F32(ln1_w, "da3.cam_enc.trunk.%d.ln1.weight", L)
                    CE_F32(ln1_b, "da3.cam_enc.trunk.%d.ln1.bias", L)
                    CE_F16(attn_qkv_w, "da3.cam_enc.trunk.%d.attn_qkv.weight", L)
                    { snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.attn_qkv.weight", L);
                      da3_tensor_info qti = find_st_tensor(st, map, map_count, name);
                      ly.qkv_rows = qti.valid ? qti.n_rows : (3 * ce.trunk_dim); }
                    CE_F32(attn_qkv_b, "da3.cam_enc.trunk.%d.attn_qkv.bias", L)
                    CE_F16(attn_out_w, "da3.cam_enc.trunk.%d.attn_out.weight", L)
                    { snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.attn_out.weight", L);
                      da3_tensor_info oti = find_st_tensor(st, map, map_count, name);
                      ly.out_rows = oti.valid ? oti.n_rows : ce.trunk_dim; }
                    CE_F32(attn_out_b, "da3.cam_enc.trunk.%d.attn_out.bias", L)
                    CE_F32(ls1, "da3.cam_enc.trunk.%d.ls1", L)
                    CE_F32(ls2, "da3.cam_enc.trunk.%d.ls2", L)
                    CE_F32(ln2_w, "da3.cam_enc.trunk.%d.ln2.weight", L)
                    CE_F32(ln2_b, "da3.cam_enc.trunk.%d.ln2.bias", L)
                    CE_F16(ffn_up_w, "da3.cam_enc.trunk.%d.ffn_up.weight", L)
                    if (isValid(ly.ffn_up_w)) {
                        snprintf(name, sizeof(name), "da3.cam_enc.trunk.%d.ffn_up.weight", L);
                        da3_tensor_info fti = find_st_tensor(st, map, map_count, name);
                        ly.ffn_up_rows = fti.valid ? fti.n_rows : (4 * ce.trunk_dim);
                    }
                    CE_F32(ffn_up_b, "da3.cam_enc.trunk.%d.ffn_up.bias", L)
                    CE_F16(ffn_down_w, "da3.cam_enc.trunk.%d.ffn_down.weight", L)
                    CE_F32(ffn_down_b, "da3.cam_enc.trunk.%d.ffn_down.bias", L)
#undef CE_F32
#undef CE_F16
                }
            }
            ce.loaded = true;
            if (r->verbose >= 1)
                fprintf(stderr, "DA3: CameraEnc weights loaded (%d trunk blocks, dim=%d)\n", ntb, ce.trunk_dim);
        }
    }

    /* Upload DPT Aux Branch weights */
    {
        da3_tensor_info ti = find_st_tensor(st, map, map_count, "da3.head.aux_fuse.0.out.weight");
        if (ti.valid) {
            auto &aux = r->dpt_aux;
            char name[128];
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.out.weight", i);
                aux.fuse_out_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.out.bias", i);
                aux.fuse_out_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv1.weight", i);
                aux.fuse_rcu1_c1_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv1.bias", i);
                aux.fuse_rcu1_c1_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv2.weight", i);
                aux.fuse_rcu1_c2_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu1.conv2.bias", i);
                aux.fuse_rcu1_c2_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                aux.has_rcu1[i] = isValid(aux.fuse_rcu1_c1_w[i]);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv1.weight", i);
                aux.fuse_rcu2_c1_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv1.bias", i);
                aux.fuse_rcu2_c1_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv2.weight", i);
                aux.fuse_rcu2_c2_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_fuse.%d.rcu2.conv2.bias", i);
                aux.fuse_rcu2_c2_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                aux.has_rcu2[i] = isValid(aux.fuse_rcu2_c1_w[i]);
            }
            for (int lv = 0; lv < 4; lv++) {
                aux.oc1_count[lv] = 0;
                for (int ci = 0; ci < 5; ci++) {
                    snprintf(name, sizeof(name), "da3.head.aux_oc1.%d.%d.weight", lv, ci);
                    da3_tensor_info tti = find_st_tensor(st, map, map_count, name);
                    if (!tti.valid) break;
                    aux.oc1_w[lv][ci] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                    aux.oc1_co[lv][ci] = (tti.n_dims == 4) ? (int)tti.shape[tti.n_dims-1] : tti.n_rows;
                    aux.oc1_ci[lv][ci] = (tti.n_dims == 4) ? (int)tti.shape[tti.n_dims-2] : (tti.n_cols / 9);
                    snprintf(name, sizeof(name), "da3.head.aux_oc1.%d.%d.bias", lv, ci);
                    aux.oc1_b[lv][ci] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                    aux.oc1_count[lv] = ci + 1;
                }
            }
            for (int lv = 0; lv < 4; lv++) {
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.conv.weight", lv);
                aux.oc2_conv_w[lv] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.conv.bias", lv);
                aux.oc2_conv_b[lv] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.gn.weight", lv);
                aux.oc2_gn_w[lv] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.gn.bias", lv);
                aux.oc2_gn_b[lv] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.out.weight", lv);
                aux.oc2_out_w[lv] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.head.aux_oc2.%d.out.bias", lv);
                aux.oc2_out_b[lv] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            }
            aux.loaded = true;
            if (r->verbose >= 1) fprintf(stderr, "DA3: DPT Aux Branch weights loaded\n");
        }
    }

    /* Upload GSDPT weights */
    {
        da3_tensor_info ti = find_st_tensor(st, map, map_count, "da3.gsdpt.head.proj.0.weight");
        if (ti.valid) {
            auto &gs = r->gsdpt;
            auto &gdw = gs.dpt;
            char name[128];
            gdw.norm_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.norm.weight", r->verbose);
            gdw.norm_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.norm.bias", r->verbose);
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.gsdpt.head.proj.%d.weight", i);
                gdw.proj_w[i] = stUploadF16Mapped(r->runner, st, map, map_count, name, r->verbose);
                { da3_tensor_info pti = find_st_tensor(st, map, map_count, name); gdw.proj_rows[i] = pti.valid ? pti.n_rows : head_oc[i]; }
                snprintf(name, sizeof(name), "da3.gsdpt.head.proj.%d.bias", i);
                gdw.proj_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            }
            { int oc0 = r->head_out_channels[0];
              gdw.upsample_0_w = uploadDeconvWeightF16(r->runner, st, map, map_count,
                  "da3.gsdpt.head.upsample_0.weight", oc0, oc0, 4, 4, r->verbose);
              gdw.upsample_0_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.upsample_0.bias", r->verbose);
              int oc1 = r->head_out_channels[1];
              gdw.upsample_1_w = uploadDeconvWeightF16(r->runner, st, map, map_count,
                  "da3.gsdpt.head.upsample_1.weight", oc1, oc1, 2, 2, r->verbose);
              gdw.upsample_1_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.upsample_1.bias", r->verbose);
            }
            gdw.downsample_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.downsample.weight", r->verbose);
            gdw.downsample_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.downsample.bias", r->verbose);
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.gsdpt.head.adapter.%d.weight", i);
                gdw.adapter_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
            }
            for (int i = 0; i < 4; i++) {
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.out.weight", i);
                gdw.fuse_out_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.out.bias", i);
                gdw.fuse_out_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv1.weight", i);
                gdw.fuse_rcu1_c1_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv1.bias", i);
                gdw.fuse_rcu1_c1_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv2.weight", i);
                gdw.fuse_rcu1_c2_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu1.conv2.bias", i);
                gdw.fuse_rcu1_c2_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                gdw.has_rcu1[i] = isValid(gdw.fuse_rcu1_c1_w[i]);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv1.weight", i);
                gdw.fuse_rcu2_c1_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv1.bias", i);
                gdw.fuse_rcu2_c1_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv2.weight", i);
                gdw.fuse_rcu2_c2_w[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                snprintf(name, sizeof(name), "da3.gsdpt.head.fuse.%d.rcu2.conv2.bias", i);
                gdw.fuse_rcu2_c2_b[i] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                gdw.has_rcu2[i] = isValid(gdw.fuse_rcu2_c1_w[i]);
            }
            gdw.neck_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.neck.weight", r->verbose);
            gdw.neck_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.neck.bias", r->verbose);
            gdw.out_0_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.out_0.weight", r->verbose);
            gdw.out_0_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.out_0.bias", r->verbose);
            { da3_tensor_info oti = find_st_tensor(st, map, map_count, "da3.gsdpt.head.out_0.weight");
              gdw.out_mid = oti.valid ? (int)oti.shape[oti.n_dims - 1] : 32; }
            gdw.out_2_w = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.out_2.weight", r->verbose);
            gdw.out_2_b = stUploadF32Mapped(r->runner, st, map, map_count, "da3.gsdpt.head.out_2.bias", r->verbose);
            { da3_tensor_info oti = find_st_tensor(st, map, map_count, "da3.gsdpt.head.out_2.weight");
              gs.gs_out_channels = oti.valid ? (int)oti.shape[oti.n_dims - 1] : 38; }
            gs.merger_ci[0] = 3;  gs.merger_co[0] = 32;
            gs.merger_ci[1] = 32; gs.merger_co[1] = 64;
            gs.merger_ci[2] = 64; gs.merger_co[2] = 128;
            for (int mi = 0; mi < 3; mi++) {
                snprintf(name, sizeof(name), "da3.gsdpt.merger.%d.weight", mi);
                da3_tensor_info mti = find_st_tensor(st, map, map_count, name);
                if (mti.valid) {
                    gs.merger_w[mi] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                    gs.merger_co[mi] = (mti.n_dims == 4) ? (int)mti.shape[mti.n_dims-1] : mti.n_rows;
                    snprintf(name, sizeof(name), "da3.gsdpt.merger.%d.bias", mi);
                    gs.merger_b[mi] = stUploadF32Mapped(r->runner, st, map, map_count, name, r->verbose);
                }
            }
            gs.loaded = true;
            if (r->verbose >= 1)
                fprintf(stderr, "DA3: GSDPT weights loaded (out_ch=%d, out_mid=%d, merger=%d->%d->%d)\n",
                        gs.gs_out_channels, gdw.out_mid, gs.merger_co[0], gs.merger_co[1], gs.merger_co[2]);
        }
    }

    /* Allocate scratch buffers */
    int nt = r->n_tokens;
    int np = r->n_patches;
    int gh = r->grid_h;
    int gw = r->grid_w;
    int max_ffn = r->use_swiglu ? 2 * r->ffn_hidden : 4 * dim;

    r->hidden       = createGpuBuffer(r->runner, (size_t)nt * dim * sizeof(float));
    r->hidden2      = createGpuBuffer(r->runner, (size_t)nt * dim * sizeof(float));
    r->local_hidden = createGpuBuffer(r->runner, (size_t)nt * dim * sizeof(float));
    r->ln_buf       = createGpuBuffer(r->runner, (size_t)nt * dim * sizeof(float));
    r->qkv          = createGpuBuffer(r->runner, (size_t)nt * 3 * dim * sizeof(float));
    r->attn_out     = createGpuBuffer(r->runner, (size_t)nt * dim * sizeof(float));
    r->ffn_buf      = createGpuBuffer(r->runner, (size_t)nt * max_ffn * sizeof(float));
    r->ffn_mid      = createGpuBuffer(r->runner, (size_t)nt * r->ffn_hidden * sizeof(float));
    r->proj_out     = createGpuBuffer(r->runner, (size_t)nt * dim * sizeof(float));
    r->pos_y        = createGpuBuffer(r->runner, (size_t)nt * sizeof(int));
    r->pos_x        = createGpuBuffer(r->runner, (size_t)nt * sizeof(int));
    r->pos_y_nd     = createGpuBuffer(r->runner, (size_t)nt * sizeof(int));
    r->pos_x_nd     = createGpuBuffer(r->runner, (size_t)nt * sizeof(int));
    r->img_norm     = createGpuBuffer(r->runner, (size_t)3 * r->image_size * r->image_size * sizeof(float));

    for (int i = 0; i < 4; i++)
        r->features[i] = createGpuBuffer(r->runner, (size_t)nt * 2 * dim * sizeof(float));

    /* DPT scratch buffers */
    {
        int feat = r->head_features;
        int oc_max = r->head_out_channels[3];
        int sp_h[4], sp_w[4];
        sp_h[0] = sp_w[0] = (gh - 1) * 4 + 4;
        sp_h[1] = sp_w[1] = (gh - 1) * 2 + 2;
        sp_h[2] = sp_w[2] = gh;
        sp_h[3] = sp_w[3] = (gh + 2 - 3) / 2 + 1;
        int fused_hw = sp_h[0] * 2 * sp_w[0] * 2;
        int model_hw = r->image_size * r->image_size;
        size_t large_sz = (size_t)feat * fused_hw;
        if (large_sz < (size_t)feat * model_hw) large_sz = (size_t)feat * model_hw;
        if (large_sz < (size_t)np * 2 * dim) large_sz = (size_t)np * 2 * dim;

        r->dpt_cat  = createGpuBuffer(r->runner, large_sz * sizeof(float));
        r->dpt_ln   = createGpuBuffer(r->runner, large_sz * sizeof(float));
        r->dpt_proj = createGpuBuffer(r->runner, (size_t)np * oc_max * sizeof(float));
        r->dpt_chw  = createGpuBuffer(r->runner, (size_t)oc_max * gh * gh * sizeof(float));

        for (int i = 0; i < 4; i++) {
            int oc_i = r->head_out_channels[i];
            r->dpt_spatial[i] = createGpuBuffer(r->runner, (size_t)oc_i * sp_h[i] * sp_w[i] * sizeof(float));
            r->dpt_adapted[i] = createGpuBuffer(r->runner, (size_t)feat * sp_h[i] * sp_w[i] * sizeof(float));
        }

        r->dpt_fused = createGpuBuffer(r->runner, large_sz * sizeof(float));
        r->dpt_tmp   = createGpuBuffer(r->runner, large_sz * sizeof(float));
        r->dpt_tmp2  = createGpuBuffer(r->runner, large_sz * sizeof(float));

        size_t out_sz = (size_t)2 * model_hw;
        if (out_sz < (size_t)2 * fused_hw) out_sz = (size_t)2 * fused_hw;
        r->dpt_out = createGpuBuffer(r->runner, out_sz * sizeof(float));

        /* Pre-allocate zero buffer for no-bias Conv2d (avoids per-dispatch alloc) */
        {
            size_t zb_size = (size_t)feat * sizeof(float);
            r->zero_bias = createGpuBuffer(r->runner, zb_size);
            std::vector<float> zeros(feat, 0.0f);
            r->runner.uploadToDeviceLocal(r->zero_bias, zeros.data(), zb_size);
        }

        /* Aux DPT scratch buffers */
        if (r->dpt_aux.loaded) {
            int max_hw = sp_h[0] * sp_w[0];
            r->aux_out = createGpuBuffer(r->runner, (size_t)7 * max_hw * sizeof(float));
            r->aux_scratch = createGpuBuffer(r->runner, (size_t)256 * max_hw * sizeof(float));
            if (r->verbose >= 1) fprintf(stderr, "DA3: DPT Aux scratch allocated\n");
        }

        /* GSDPT scratch buffers */
        if (r->gsdpt.loaded) {
            int mg_h = r->image_size, mg_w = r->image_size;
            for (int s = 0; s < 3; s++) { mg_h = (mg_h + 2*1 - 3)/2 + 1; mg_w = (mg_w + 2*1 - 3)/2 + 1; }
            r->gs_merger_h = mg_h; r->gs_merger_w = mg_w;
            r->gs_merged = createGpuBuffer(r->runner, (size_t)128 * mg_h * mg_w * sizeof(float));
            int gs_oc = r->gsdpt.gs_out_channels; if (gs_oc < 2) gs_oc = 38;
            int max_hw = sp_h[0] * sp_w[0];
            r->gs_out = createGpuBuffer(r->runner, (size_t)gs_oc * max_hw * sizeof(float));
            if (r->verbose >= 1)
                fprintf(stderr, "DA3: GSDPT scratch allocated (%d channels, merger=%dx%d)\n", gs_oc, mg_h, mg_w);
        }
    }

    /* Upload position arrays for RoPE: CLS at (0,0), patches at (y+1, x+1) */
    {
        std::vector<int> py(nt, 0), px(nt, 0), py_nd(nt, 0), px_nd(nt, 0);
        for (int p = 0; p < np; p++) {
            py[1 + p] = p / r->grid_w + 1;
            px[1 + p] = p % r->grid_w + 1;
            py_nd[1 + p] = 1;
            px_nd[1 + p] = 1;
        }
        r->runner.uploadToDeviceLocal(r->pos_y, py.data(), (size_t)nt * sizeof(int));
        r->runner.uploadToDeviceLocal(r->pos_x, px.data(), (size_t)nt * sizeof(int));
        r->runner.uploadToDeviceLocal(r->pos_y_nd, py_nd.data(), (size_t)nt * sizeof(int));
        r->runner.uploadToDeviceLocal(r->pos_x_nd, px_nd.data(), (size_t)nt * sizeof(int));
    }

    /* Clean up name map */
    for (int i = 0; i < map_count; i++) free((void *)map[i].gguf_name);
    free(map);
    safetensors_close(st);

    r->loaded = true;

    if (r->verbose >= 1) {
        fprintf(stderr, "DA3: loaded %d blocks, dim=%d, tokens=%d, swiglu=%d (safetensors)\n",
                nb, dim, nt, r->use_swiglu);
        fprintf(stderr, "DA3: modules: cam_dec=%d cam_enc=%d dpt_aux=%d gsdpt=%d\n",
                (int)r->cam_dec.loaded, (int)r->cam_enc.loaded, (int)r->dpt_aux.loaded, (int)r->gsdpt.loaded);
    }
    return 0;
}

/* ======================================================================== */
/* Bicubic interpolation for position embedding                             */
/* ======================================================================== */

static float bicubic_weight(float x) {
    float ax = fabsf(x);
    if (ax <= 1.0f) return ((1.25f * ax - 2.25f) * ax) * ax + 1.0f;
    if (ax < 2.0f) return ((-0.75f * ax + 3.75f) * ax - 6.0f) * ax + 3.0f;
    return 0.0f;
}

static float *interpolate_pos_embed_cpu(const float *pos_embed_flat,
                                          int M, int h0, int w0, int dim) {
    float interp_offset = 0.1f;
    float sx = (float)(w0 + interp_offset) / M;
    float sy = (float)(h0 + interp_offset) / M;
    float *out = (float *)calloc((size_t)h0 * w0 * dim, sizeof(float));
    for (int y = 0; y < h0; y++) {
        float fy = ((float)y + 0.5f) / sy - 0.5f;
        if (fy < 0) fy = 0; if (fy > M - 1) fy = (float)(M - 1);
        int iy = (int)floorf(fy);
        float dy = fy - iy;
        for (int x = 0; x < w0; x++) {
            float fx = ((float)x + 0.5f) / sx - 0.5f;
            if (fx < 0) fx = 0; if (fx > M - 1) fx = (float)(M - 1);
            int ix = (int)floorf(fx);
            float dx = fx - ix;
            for (int d = 0; d < dim; d++) {
                float val = 0.0f;
                for (int jj = -1; jj <= 2; jj++) {
                    int sy_idx = iy + jj; if (sy_idx < 0) sy_idx = 0; if (sy_idx >= M) sy_idx = M - 1;
                    float wy = bicubic_weight(dy - jj);
                    for (int ii = -1; ii <= 2; ii++) {
                        int sx_idx = ix + ii; if (sx_idx < 0) sx_idx = 0; if (sx_idx >= M) sx_idx = M - 1;
                        float wx = bicubic_weight(dx - ii);
                        val += pos_embed_flat[sy_idx * M * dim + sx_idx * dim + d] * wy * wx;
                    }
                }
                out[y * w0 * dim + x * dim + d] = val;
            }
        }
    }
    return out;
}

static int nearest_multiple(int x, int p) {
    int down = (x / p) * p;
    int up = down + p;
    return (up - x <= x - down) ? up : down;
}

/* ======================================================================== */
/* CameraEnc: pose[9] -> MLP -> trunk(ViT blocks) -> camera_token[dim]     */
/* ======================================================================== */

/* CameraEnc: encode a camera pose into a conditioning token.
 * The output token replaces the default camera_token in the backbone. */
static void run_camera_enc(vulkan_da3_runner *r, const float *pose_in, BufInfo &cam_token_out) {
    auto &ce = r->cam_enc;
    int dim = ce.trunk_dim;
    int ntb = (int)ce.trunk.size();

    /* Upload pose[9] to a small temp buffer */
    BufInfo d_pose = createGpuBuffer(r->runner, 9 * sizeof(float));
    r->runner.uploadToDeviceLocal(d_pose, pose_in, 9 * sizeof(float));

    /* pose_branch MLP: [1, 9] -> fc1 -> GELU -> fc2 -> [1, dim] */
    opMatmulBiasF16(r, r->attn_out, ce.fc1_w, d_pose, ce.fc1_b, 1, dim, 9);
    opGelu(r, r->attn_out, dim);
    opMatmulBiasF16(r, r->proj_out, ce.fc2_w, r->attn_out, ce.fc2_b, 1, dim, dim);

    /* token_norm: LayerNorm on [1, dim] */
    opLayerNorm(r, r->ln_buf, r->proj_out, ce.token_norm_w, ce.token_norm_b, 1, dim);

    /* Copy to hidden2 for trunk processing */
    opCopy(r, r->hidden2, r->ln_buf, dim);

    /* trunk: ViT blocks (no RoPE, no QK-norm, single token) */
    for (int L = 0; L < ntb; L++) {
        auto &ly = ce.trunk[L];

        /* LN1 + self-attention */
        opLayerNorm(r, r->ln_buf, r->hidden2, ly.ln1_w, ly.ln1_b, 1, dim);
        opMatmulBiasF16(r, r->qkv, ly.attn_qkv_w, r->ln_buf, ly.attn_qkv_b, 1, ly.qkv_rows, dim);

        /* For single token, attention is trivial: output = V (softmax of [1x1] = 1) */
        /* V starts at offset 2*dim in the QKV buffer. Use GPU copy with offset. */
        {
            std::vector<float> qkv_tmp(3 * dim);
            r->runner.downloadFromDeviceLocal(r->qkv, qkv_tmp.data(), 3 * dim * sizeof(float));
            r->runner.uploadToDeviceLocal(r->attn_out, qkv_tmp.data() + 2 * dim, (size_t)dim * sizeof(float));
        }

        /* Output projection */
        opMatmulBiasF16(r, r->proj_out, ly.attn_out_w, r->attn_out, ly.attn_out_b, 1, dim, dim);

        /* LayerScale 1 + residual */
        opLayerscaleAdd(r, r->hidden2, r->proj_out, ly.ls1, dim, dim);

        /* LN2 + FFN (GELU MLP) */
        opLayerNorm(r, r->ln_buf, r->hidden2, ly.ln2_w, ly.ln2_b, 1, dim);
        opMatmulBiasF16(r, r->ffn_buf, ly.ffn_up_w, r->ln_buf, ly.ffn_up_b, 1, ly.ffn_up_rows, dim);
        opGelu(r, r->ffn_buf, ly.ffn_up_rows);
        opMatmulBiasF16(r, r->proj_out, ly.ffn_down_w, r->ffn_buf, ly.ffn_down_b, 1, dim, ly.ffn_up_rows);

        /* LayerScale 2 + residual */
        opLayerscaleAdd(r, r->hidden2, r->proj_out, ly.ls2, dim, dim);
    }

    /* trunk_norm: LayerNorm on final output -> cam_token_out */
    opLayerNorm(r, cam_token_out, r->hidden2, ce.trunk_norm_w, ce.trunk_norm_b, 1, dim);

    r->runner.destroyBuffer(d_pose);

    if (r->verbose >= 1) {
        float dbg[5];
        r->runner.downloadFromDeviceLocal(cam_token_out, dbg, sizeof(dbg));
        fprintf(stderr, "DA3: CameraEnc output[0:5] = %.4f %.4f %.4f %.4f %.4f\n",
                dbg[0], dbg[1], dbg[2], dbg[3], dbg[4]);
    }
}

/* ======================================================================== */
/* Public API: predict_full                                                 */
/* ======================================================================== */

da3_full_result vulkan_da3_predict_full(vulkan_da3_runner *r, const uint8_t *rgb,
                                          int img_w, int img_h, int output_flags,
                                          const float *pose_in) {
    da3_full_result result = {};
    if (!r || !r->loaded) return result;

    int dim = r->dim, ps = r->patch_size;
    struct timespec ts;
    double t0, t1;

    /* Compute target size */
    int process_res = 504;
    int longest = (img_w > img_h) ? img_w : img_h;
    float scale = (float)process_res / (float)longest;
    int new_w = (int)(img_w * scale + 0.5f); if (new_w < 1) new_w = 1;
    int new_h = (int)(img_h * scale + 0.5f); if (new_h < 1) new_h = 1;
    int target_w = nearest_multiple(new_w, ps); if (target_w < ps) target_w = ps;
    int target_h = nearest_multiple(new_h, ps); if (target_h < ps) target_h = ps;

    int gh = target_h / ps, gw = target_w / ps;
    int np = gh * gw, nt = np + 1;

    if (r->verbose >= 1)
        fprintf(stderr, "DA3: input %dx%d -> target %dx%d (grid %dx%d, %d patches)\n",
                img_w, img_h, target_w, target_h, gw, gh, np);

    /* Clamp if needed */
    int orig_nt = r->n_tokens;
    if (nt > orig_nt) {
        fprintf(stderr, "DA3: WARNING: grid %dx%d (%d tokens) exceeds allocated %d tokens, clamping\n",
                gw, gh, nt, orig_nt);
        gh = r->grid_h; gw = r->grid_w; np = r->n_patches; nt = r->n_tokens;
        target_h = gh * ps; target_w = gw * ps;
    }

    /* ---- GPU: Preprocess + Patch Embed + CLS + PosEmbed ---- */
    clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    /* Upload raw image via staging buffer */
    size_t img_bytes = (size_t)img_w * img_h * 3;
    if (img_bytes > r->img_raw_cap) {
        if (isValid(r->img_raw)) r->runner.destroyBuffer(r->img_raw);
        r->img_raw = createGpuBuffer(r->runner, img_bytes);
        r->img_raw_cap = img_bytes;
    }
    /* Ensure staging buffer is large enough */
    if (img_bytes > r->upload_staging_cap) {
        if (isValid(r->upload_staging)) r->runner.destroyBuffer(r->upload_staging);
        r->runner.createStagingBuffer(img_bytes, r->upload_staging);
        r->upload_staging_cap = img_bytes;
    }
    /* Upload image to staging */
    {
        void *ptr = nullptr;
        r->runner.mapBuffer(r->upload_staging, &ptr);
        memcpy(ptr, rgb, img_bytes);
        r->runner.unmapBuffer(r->upload_staging);
    }

    /* GPU copy: staging -> img_raw, then preprocessing dispatches */
    r->batched = true;
    r->runner.beginRecording();
    r->runner.recordCopyBuffer(r->upload_staging, r->img_raw, img_bytes);
    r->runner.computeBarrier();

    /* Resize + normalize */
    opResizeNormalize(r, r->img_norm, r->img_raw, img_w, img_h, target_w, target_h);

    /* Patch embed: img_norm[3, target_h, target_w] -> hidden[np, dim] at offset dim (after CLS slot) */
    /* Note: patch_embed writes to hidden starting from patch 0. CLS is prepended by cls_pos_embed. */
    opPatchEmbed(r, r->hidden, r->img_norm, r->patch_embed_w, r->patch_embed_b,
                 np, dim, ps, gw, target_w, target_h);

    /* End preprocessing batch (resize+normalize, patch_embed are batched) */
    r->runner.endRecordingAndSubmitFenced(r->batch_fence);
    r->runner.waitForFence(r->batch_fence);
    r->runner.resetFence(r->batch_fence);
    r->runner.resetDynamicDescriptorPool();
    r->batched = false;

    /* Interpolate pos_embed for non-square grid (CPU work + device-local uploads) */
    {
        int M = r->grid_h;
        if (gh != M || gw != M) {
            float *orig_pe = r->h_pos_embed_orig;
            float *cls_pe = orig_pe;
            float *patch_pe = orig_pe + dim;
            float *interp_pe = interpolate_pos_embed_cpu(patch_pe, M, gh, gw, dim);
            std::vector<float> new_pe((size_t)nt * dim);
            memcpy(new_pe.data(), cls_pe, (size_t)dim * sizeof(float));
            for (int p = 0; p < np; p++)
                memcpy(new_pe.data() + (1 + p) * dim, interp_pe + p * dim, (size_t)dim * sizeof(float));
            r->runner.uploadToDeviceLocal(r->pos_embed, new_pe.data(), (size_t)nt * dim * sizeof(float));
            free(interp_pe);
        }

        /* Rebuild position arrays for RoPE with new grid */
        if (gh != r->grid_h || gw != r->grid_w) {
            std::vector<int> py(nt, 0), px(nt, 0), py_nd(nt, 0), px_nd(nt, 0);
            for (int p = 0; p < np; p++) {
                py[1 + p] = p / gw + 1; px[1 + p] = p % gw + 1;
                py_nd[1 + p] = 1; px_nd[1 + p] = 1;
            }
            r->runner.uploadToDeviceLocal(r->pos_y, py.data(), (size_t)nt * sizeof(int));
            r->runner.uploadToDeviceLocal(r->pos_x, px.data(), (size_t)nt * sizeof(int));
            r->runner.uploadToDeviceLocal(r->pos_y_nd, py_nd.data(), (size_t)nt * sizeof(int));
            r->runner.uploadToDeviceLocal(r->pos_x_nd, px_nd.data(), (size_t)nt * sizeof(int));
        }

        /* CLS prepend + pos_embed add (not batched, runs sync) */
        opClsPosEmbed(r, r->hidden, r->cls_token, r->pos_embed, nt, dim);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    fprintf(stderr, "DA3: GPU preprocess+embed: %.1f ms\n", (t1 - t0) * 1000);

    /* ---- GPU: Transformer Blocks ---- */
    clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;
    int stride_3dim = 3 * dim;

    /* If pose_in is provided and CameraEnc is loaded, compute conditioned camera token */
    BufInfo d_cond_cam_token{};
    bool have_cond_cam = false;
    if (pose_in && r->cam_enc.loaded) {
        d_cond_cam_token = createGpuBuffer(r->runner, (size_t)dim * sizeof(float));
        run_camera_enc(r, pose_in, d_cond_cam_token);
        have_cond_cam = true;
        if (r->verbose >= 1) fprintf(stderr, "DA3: CameraEnc: pose-conditioned camera token computed\n");
    }

    /* dump hidden before backbone (verbose >= 3) */
    if (r->verbose >= 3) {
        std::vector<float> h_dbg(nt * dim);
        r->runner.downloadFromDeviceLocal(r->hidden, h_dbg.data(), nt * dim * sizeof(float));
        float mn = h_dbg[0], mx = h_dbg[0], sum = 0;
        for (int i = 0; i < nt * dim; i++) { if (h_dbg[i]<mn) mn=h_dbg[i]; if (h_dbg[i]>mx) mx=h_dbg[i]; sum+=h_dbg[i]; }
        fprintf(stderr, "DA3: after embed: min=%.4f max=%.4f mean=%.6f\n", mn, mx, sum/(nt*dim));
        fprintf(stderr, "DA3: hidden[0:5] = %.4f %.4f %.4f %.4f %.4f\n", h_dbg[0], h_dbg[1], h_dbg[2], h_dbg[3], h_dbg[4]);
        fprintf(stderr, "DA3: hidden[dim:dim+5] = %.4f %.4f %.4f %.4f %.4f\n", h_dbg[dim], h_dbg[dim+1], h_dbg[dim+2], h_dbg[dim+3], h_dbg[dim+4]);
        fflush(stderr);
    }

    /* Begin batched backbone dispatch */
    r->batched = true;
    r->runner.beginRecording();

    /* Initialize local_hidden = hidden */
    opCopy(r, r->local_hidden, r->hidden, nt * dim);

    for (int L = 0; L < r->n_blocks; L++) {
        auto &ly = r->layers[L];
        int is_global = (L >= r->rope_start && (L % 2) == 1);

        /* Camera token injection at rope_start layer:
         * Use conditioned token (from CameraEnc) if available, else default camera_token */
        if (L == r->rope_start) {
            BufInfo *cam_src = have_cond_cam ? &d_cond_cam_token :
                               (r->has_camera_token ? &r->camera_token : nullptr);
            if (cam_src) {
                /* GPU-side copy of camera token to hidden[0:dim] */
                opCopy(r, r->hidden, *cam_src, dim);
            }
        }

        /* LayerNorm -> QKV projection */
        opLayerNorm(r, r->ln_buf, r->hidden, ly.ln1_w, ly.ln1_b, nt, dim);
        opMatmulBiasF16(r, r->qkv, ly.attn_qkv_w, r->ln_buf, ly.attn_qkv_b, nt, ly.qkv_rows, dim);

        /* QK LayerNorm */
        if (L >= r->qk_norm_start && ly.has_qk_norm) {
            /* Q norm: base_offset=0 (Q starts at beginning of each token's QKV) */
            opQkLayerNormOffset(r, r->qkv, ly.attn_q_norm_w, ly.attn_q_norm_b,
                                nt, r->n_heads, r->head_dim, stride_3dim, r->ln_eps, 0);
            /* K norm: base_offset=dim (K starts at offset dim in each token's QKV) */
            opQkLayerNormOffset(r, r->qkv, ly.attn_k_norm_w, ly.attn_k_norm_b,
                                nt, r->n_heads, r->head_dim, stride_3dim, r->ln_eps, dim);
        }

        /* RoPE 2D */
        if (L >= r->rope_start) {
            BufInfo &use_py = is_global ? r->pos_y_nd : r->pos_y;
            BufInfo &use_px = is_global ? r->pos_x_nd : r->pos_x;
            /* Q RoPE: base_offset=0 */
            opRope2d(r, r->qkv, use_py, use_px, nt, r->n_heads, r->head_dim, stride_3dim, 100.0f, 0);
            /* K RoPE: base_offset=dim */
            opRope2d(r, r->qkv, use_py, use_px, nt, r->n_heads, r->head_dim, stride_3dim, 100.0f, dim);
        }

        /* KV transpose + Flash Attention */
        {
            /* Use ffn_buf as scratch for K_t and V_t */
            /* K_t[n_heads, n_tok, head_dim], V_t[n_heads, n_tok, head_dim] */
            /* ffn_buf needs to hold at least 2 * nt * dim floats */
            opKvTranspose(r, r->ffn_buf, r->ffn_mid, r->qkv, nt, dim, r->n_heads, r->head_dim);
            opFlashAttn(r, r->attn_out, r->qkv, r->ffn_buf, r->ffn_mid,
                        nt, dim, r->n_heads, r->head_dim);
        }

        /* Attention output projection */
        opMatmulBiasF16(r, r->proj_out, ly.attn_out_w, r->attn_out, ly.attn_out_b,
                     nt, ly.out_rows, dim);

        /* Residual with layer scale */
        if (isValid(ly.ls1)) {
            opLayerscaleAdd(r, r->hidden, r->proj_out, ly.ls1, nt * dim, dim);
        } else {
            opAdd(r, r->hidden, r->proj_out, nt * dim);
        }

        /* FFN */
        opLayerNorm(r, r->ln_buf, r->hidden, ly.ln2_w, ly.ln2_b, nt, dim);

        if (ly.has_swiglu && isValid(ly.ffn_gate_up_w)) {
            /* SwiGLU: gate_up = W12 * x + b12, then swiglu, then down */
            opMatmulBiasF16(r, r->ffn_buf, ly.ffn_gate_up_w, r->ln_buf, ly.ffn_gate_up_b,
                         nt, ly.ffn_gu_rows, dim);
            int hid = ly.ffn_gu_rows / 2;
            opSwiglu(r, r->ffn_mid, r->ffn_buf, hid, nt);
            opMatmulBiasF16(r, r->proj_out, ly.ffn_down_w, r->ffn_mid, ly.ffn_down_b,
                         nt, ly.ffn_down_rows, hid);
        } else if (isValid(ly.ffn_up_w)) {
            /* GELU FFN: up -> gelu -> down */
            opMatmulBiasF16(r, r->ffn_buf, ly.ffn_up_w, r->ln_buf, ly.ffn_up_b,
                         nt, ly.ffn_up_rows, dim);
            opGelu(r, r->ffn_buf, nt * ly.ffn_up_rows);
            opMatmulBiasF16(r, r->proj_out, ly.ffn_down_w, r->ffn_buf, ly.ffn_down_b,
                         nt, ly.ffn_down_rows, ly.ffn_up_rows);
        }

        /* FFN residual with layer scale */
        if (isValid(ly.ls2)) {
            opLayerscaleAdd(r, r->hidden, r->proj_out, ly.ls2, nt * dim, dim);
        } else {
            opAdd(r, r->hidden, r->proj_out, nt * dim);
        }

        /* Update local_hidden after each local (non-global) block */
        if (!is_global)
            opCopy(r, r->local_hidden, r->hidden, nt * dim);

        /* Save features as [local_x, x] per token */
        for (int fi = 0; fi < 4; fi++) {
            if (L == r->feature_layers[fi]) {
                opCatLocalGlobal(r, r->features[fi], r->local_hidden, r->hidden, nt, dim);
            }
        }

        /* Per-block backbone stats */
        if (r->verbose >= 2) {
            if (r->batched) {
                r->runner.endRecordingAndSubmitFenced(r->batch_fence);
                r->runner.waitForFence(r->batch_fence);
                r->runner.resetFence(r->batch_fence);
            }
            std::vector<float> h_dbg(nt * dim);
            r->runner.downloadFromDeviceLocal(r->hidden, h_dbg.data(), nt * dim * sizeof(float));
            float mn = h_dbg[0], mx = h_dbg[0], sum = 0;
            for (int i = 0; i < nt * dim; i++) { if (h_dbg[i]<mn) mn=h_dbg[i]; if (h_dbg[i]>mx) mx=h_dbg[i]; sum+=h_dbg[i]; }
            fprintf(stderr, "DA3: block %2d (%s): min=%8.4f max=%8.4f mean=%10.6f\n", L, is_global?"global":"local ", mn, mx, sum/(nt*dim));
            fflush(stderr);
            if (r->batched) {
                r->runner.resetDynamicDescriptorPool();
                r->runner.beginRecording();
            }
        }
    }

    /* End batched backbone dispatch */
    r->runner.endRecordingAndSubmitFenced(r->batch_fence);
    r->runner.waitForFence(r->batch_fence);
    r->runner.resetFence(r->batch_fence);
    r->runner.resetDynamicDescriptorPool();
    r->batched = false;

    /* Free conditioned camera token buffer if allocated */
    if (have_cond_cam && isValid(d_cond_cam_token))
        r->runner.destroyBuffer(d_cond_cam_token);

    clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    fprintf(stderr, "DA3: GPU backbone (%d blocks): %.1f ms\n", r->n_blocks, (t1 - t0) * 1000);

    /* dump backbone hidden stats (verbose >= 3) */
    if (r->verbose >= 3) {
        std::vector<float> h_dbg(nt * dim);
        r->runner.downloadFromDeviceLocal(r->hidden, h_dbg.data(), nt * dim * sizeof(float));
        float mn = h_dbg[0], mx = h_dbg[0], sum = 0;
        for (int i = 0; i < nt * dim; i++) { if (h_dbg[i]<mn) mn=h_dbg[i]; if (h_dbg[i]>mx) mx=h_dbg[i]; sum+=h_dbg[i]; }
        fprintf(stderr, "DA3: backbone hidden: min=%.4f max=%.4f mean=%.6f\n", mn, mx, sum/(nt*dim));
    }

    /* ---- CameraDec (Phase 1): Pose Estimation ---- */
    if (r->cam_dec.loaded && (output_flags & DA3_OUTPUT_POSE)) {
        clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

        auto &cd = r->cam_dec;
        int mlp_dim = cd.mlp_dim;

        /* backbone_norm(CLS token) -> MLP -> 3 FC heads (CPU) */
        opLayerNorm(r, r->ln_buf, r->hidden, cd.backbone_norm_w, cd.backbone_norm_b, 1, dim);
        opMatmulBiasF16(r, r->attn_out, cd.mlp_w[0], r->ln_buf, cd.mlp_b[0], 1, mlp_dim, dim);
        opGelu(r, r->attn_out, mlp_dim);
        opMatmulBiasF16(r, r->proj_out, cd.mlp_w[1], r->attn_out, cd.mlp_b[1], 1, mlp_dim, mlp_dim);
        opGelu(r, r->proj_out, mlp_dim);

        /* Download MLP output and FC weights to CPU for tiny matmuls */
        std::vector<float> h_mlp(mlp_dim);
        r->runner.downloadFromDeviceLocal(r->proj_out, h_mlp.data(), (size_t)mlp_dim * sizeof(float));

        std::vector<float> h_fc_t_w(3*mlp_dim), h_fc_t_b(3);
        std::vector<float> h_fc_q_w(4*mlp_dim), h_fc_q_b(4);
        std::vector<float> h_fc_f_w(2*mlp_dim), h_fc_f_b(2);
        r->runner.downloadFromDeviceLocal( cd.fc_t_w, h_fc_t_w.data(), 3*mlp_dim*sizeof(float));
        r->runner.downloadFromDeviceLocal( cd.fc_t_b, h_fc_t_b.data(), 3*sizeof(float));
        r->runner.downloadFromDeviceLocal( cd.fc_qvec_w, h_fc_q_w.data(), 4*mlp_dim*sizeof(float));
        r->runner.downloadFromDeviceLocal( cd.fc_qvec_b, h_fc_q_b.data(), 4*sizeof(float));
        r->runner.downloadFromDeviceLocal( cd.fc_fov_w, h_fc_f_w.data(), 2*mlp_dim*sizeof(float));
        r->runner.downloadFromDeviceLocal( cd.fc_fov_b, h_fc_f_b.data(), 2*sizeof(float));

        for (int o = 0; o < 3; o++) { float s = h_fc_t_b[o]; for (int k = 0; k < mlp_dim; k++) s += h_fc_t_w[o*mlp_dim+k]*h_mlp[k]; result.pose[o] = s; }
        for (int o = 0; o < 4; o++) { float s = h_fc_q_b[o]; for (int k = 0; k < mlp_dim; k++) s += h_fc_q_w[o*mlp_dim+k]*h_mlp[k]; result.pose[3+o] = s; }
        for (int o = 0; o < 2; o++) { float s = h_fc_f_b[o]; for (int k = 0; k < mlp_dim; k++) s += h_fc_f_w[o*mlp_dim+k]*h_mlp[k]; result.pose[7+o] = s; }

        result.has_pose = 1;

        clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (r->verbose >= 1) {
            fprintf(stderr, "DA3: CameraDec (pose): %.1f ms\n", (t1 - t0) * 1000);
            fprintf(stderr, "DA3:   t=[%.4f,%.4f,%.4f] q=[%.4f,%.4f,%.4f,%.4f] fov=[%.4f,%.4f]\n",
                    result.pose[0], result.pose[1], result.pose[2],
                    result.pose[3], result.pose[4], result.pose[5], result.pose[6],
                    result.pose[7], result.pose[8]);
        }
    }

    /* ---- GPU: DPT Head ---- */
    clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

    if (!isValid(r->dpt.proj_w[0])) {
        fprintf(stderr, "DA3: no DPT head weights\n");
        return result;
    }

    /* Begin batched DPT head dispatch */
    r->batched = true;
    r->runner.beginRecording();

    auto &dw = r->dpt;
    int feat = r->head_features;
    int head_dim_in = dim * 2;
    int sp_h[4], sp_w[4];
    sp_h[0] = (gh - 1) * 4 + 4; sp_w[0] = (gw - 1) * 4 + 4;
    sp_h[1] = (gh - 1) * 2 + 2; sp_w[1] = (gw - 1) * 2 + 2;
    sp_h[2] = gh;                 sp_w[2] = gw;
    sp_h[3] = (gh + 2 - 3) / 2 + 1; sp_w[3] = (gw + 2 - 3) / 2 + 1;

    /* Token processing + projection for each feature level */
    for (int fi = 0; fi < 4; fi++) {
        int oc_val = r->head_out_channels[fi];

        /* Extract patch tokens (skip CLS) from features[fi] */
        opDptClsConcat(r, r->dpt_cat, r->features[fi], np, dim);

        /* Apply backbone_norm to second half (global features) */
        if (isValid(r->backbone_norm_w)) {
            opStridedLayerNorm(r, r->dpt_cat, r->backbone_norm_w, r->backbone_norm_b,
                               np, dim, head_dim_in);
        }

        /* Head norm */
        if (isValid(dw.norm_w)) {
            opLayerNorm(r, r->dpt_ln, r->dpt_cat, dw.norm_w, dw.norm_b, np, head_dim_in);
        } else {
            opCopy(r, r->dpt_ln, r->dpt_cat, np * head_dim_in);
        }

        /* Project: [np, 2*dim] -> [np, oc_val] */
        opMatmulBiasF16(r, r->dpt_proj, dw.proj_w[fi], r->dpt_ln, dw.proj_b[fi],
                     np, oc_val, head_dim_in);

        /* DPT projection stats */
        if (r->verbose >= 3 && fi == 0) {
            if (r->batched) { r->runner.endRecordingAndSubmitFenced(r->batch_fence); r->runner.waitForFence(r->batch_fence); r->runner.resetFence(r->batch_fence); }
            int pn = np * oc_val;
            std::vector<float> dbg(pn);
            r->runner.downloadFromDeviceLocal( r->dpt_proj, dbg.data(), pn * sizeof(float));
            float mn=dbg[0], mx=dbg[0], sm=0;
            for (int i=0;i<pn;i++){if(dbg[i]<mn)mn=dbg[i];if(dbg[i]>mx)mx=dbg[i];sm+=dbg[i];}
            fprintf(stderr, "DA3: DPT fi=0 proj[%d]: min=%.4f max=%.4f mean=%.6f\n", pn, mn, mx, sm/pn);
            fflush(stderr);
            if (r->batched) { r->runner.resetDynamicDescriptorPool(); r->runner.beginRecording(); }
        }

        /* Spatial alignment */
        if (fi == 0) {
            /* ConvTranspose2d 4x4 stride 4: [np, oc] -> [oc, (gh-1)*4+4, (gw-1)*4+4] */
            opDeconvGemmScatter(r, r->dpt_spatial[0], r->dpt_proj,
                                dw.upsample_0_w, dw.upsample_0_b, r->dpt_ln,
                                oc_val, oc_val, gh, gw, 4, 4, 4);
        } else if (fi == 1) {
            /* ConvTranspose2d 2x2 stride 2 */
            opDeconvGemmScatter(r, r->dpt_spatial[1], r->dpt_proj,
                                dw.upsample_1_w, dw.upsample_1_b, r->dpt_ln,
                                oc_val, oc_val, gh, gw, 2, 2, 2);
        } else if (fi == 2) {
            /* Identity: tok_to_chw */
            opTokToChw(r, r->dpt_chw, r->dpt_proj, oc_val, gh, gw);
            opCopy(r, r->dpt_spatial[2], r->dpt_chw, oc_val * gh * gw);
        } else {
            /* Conv2d 3x3 stride 2 (downsample) */
            opTokToChw(r, r->dpt_chw, r->dpt_proj, oc_val, gh, gw);
            opConv2d(r, r->dpt_spatial[3], r->dpt_chw, dw.downsample_w, dw.downsample_b,
                     gh, gw, oc_val, oc_val, 3, 3, 2, 1);
        }

        /* Adapter: Conv2d 3x3, no bias (create zero bias buffer since shader always reads bias) */
        {            opConv2d(r, r->dpt_adapted[fi], r->dpt_spatial[fi], dw.adapter_w[fi], r->zero_bias,
                     sp_h[fi], sp_w[fi], oc_val, feat, 3, 3, 1, 1);
        }
    }

    /* adapted stats (verbose >= 3) */
    if (r->verbose >= 3) {
        if (r->batched) { r->runner.endRecordingAndSubmitFenced(r->batch_fence); r->runner.waitForFence(r->batch_fence); r->runner.resetFence(r->batch_fence); }
        for (int fi2 = 0; fi2 < 4; fi2++) {
            int an = feat * sp_h[fi2] * sp_w[fi2];
            std::vector<float> dbg(an);
            r->runner.downloadFromDeviceLocal( r->dpt_adapted[fi2], dbg.data(), an * sizeof(float));
            float mn=dbg[0], mx=dbg[0], sm=0;
            for (int i=0;i<an;i++){if(dbg[i]<mn)mn=dbg[i];if(dbg[i]>mx)mx=dbg[i];sm+=dbg[i];}
            fprintf(stderr, "DA3: adapted[%d][%d]: min=%.4f max=%.4f mean=%.6f\n", fi2, an, mn, mx, sm/an);
        }
        fflush(stderr);
        if (r->batched) { r->runner.resetDynamicDescriptorPool(); r->runner.beginRecording(); }
    }

    /* Bottom-up RefineNet fusion */
    gpuRefineNet(r, 3, r->dpt_adapted[3], sp_h[3], sp_w[3],
                 nullptr, 0, 0, feat, r->dpt_fused, sp_h[2], sp_w[2]);
    int fh = sp_h[2], fw = sp_w[2];

    gpuRefineNet(r, 2, r->dpt_adapted[2], sp_h[2], sp_w[2],
                 &r->dpt_fused, fh, fw, feat, r->dpt_fused, sp_h[1], sp_w[1]);
    fh = sp_h[1]; fw = sp_w[1];

    gpuRefineNet(r, 1, r->dpt_adapted[1], sp_h[1], sp_w[1],
                 &r->dpt_fused, fh, fw, feat, r->dpt_fused, sp_h[0], sp_w[0]);
    fh = sp_h[0]; fw = sp_w[0];

    gpuRefineNet(r, 0, r->dpt_adapted[0], sp_h[0], sp_w[0],
                 &r->dpt_fused, fh, fw, feat, r->dpt_fused, sp_h[0] * 2, sp_w[0] * 2);
    fh = sp_h[0] * 2; fw = sp_w[0] * 2;

    /* fused stats after RefineNet (verbose >= 3) */
    if (r->verbose >= 3) {
        if (r->batched) { r->runner.endRecordingAndSubmitFenced(r->batch_fence); r->runner.waitForFence(r->batch_fence); r->runner.resetFence(r->batch_fence); }
        int fn = feat * fh * fw;
        std::vector<float> dbg(fn);
        r->runner.downloadFromDeviceLocal( r->dpt_fused, dbg.data(), fn * sizeof(float));
        float mn=dbg[0], mx=dbg[0], sm=0;
        for (int i=0;i<fn;i++){if(dbg[i]<mn)mn=dbg[i];if(dbg[i]>mx)mx=dbg[i];sm+=dbg[i];}
        fprintf(stderr, "DA3: fused[%d] (%dx%d): min=%.4f max=%.4f mean=%.6f\n", fn, fh, fw, mn, mx, sm/fn);
        fflush(stderr);
        if (r->batched) { r->runner.resetDynamicDescriptorPool(); r->runner.beginRecording(); }
    }

    /* Output convolutions */
    int feat_half = feat / 2; if (feat_half < 1) feat_half = 1;
    int out_mid = dw.out_mid > 0 ? dw.out_mid : feat_half;

    /* 1. Neck conv at fused resolution (no ReLU) */
    opConv2d(r, r->dpt_tmp, r->dpt_fused, dw.neck_w, dw.neck_b,
             fh, fw, feat, feat_half, 3, 3, 1, 1);

    /* neck output (verbose >= 3) */
    if (r->verbose >= 3) {
        if (r->batched) { r->runner.endRecordingAndSubmitFenced(r->batch_fence); r->runner.waitForFence(r->batch_fence); r->runner.resetFence(r->batch_fence); }
        int nn = feat_half * fh * fw;
        std::vector<float> dbg(nn);
        r->runner.downloadFromDeviceLocal( r->dpt_tmp, dbg.data(), nn * sizeof(float));
        float mn=dbg[0], mx=dbg[0], sm=0;
        for (int i=0;i<nn;i++){if(dbg[i]<mn)mn=dbg[i];if(dbg[i]>mx)mx=dbg[i];sm+=dbg[i];}
        fprintf(stderr, "DA3: neck[%d] (%dx%d): min=%.4f max=%.4f mean=%.6f\n", nn, fh, fw, mn, mx, sm/nn);
        fflush(stderr);
        if (r->batched) { r->runner.resetDynamicDescriptorPool(); r->runner.beginRecording(); }
    }
    /* 2. Bilinear upsample to model resolution */
    int model_h = gh * ps, model_w = gw * ps;
    opBilinear(r, r->dpt_tmp2, r->dpt_tmp, feat_half, fh, fw, model_h, model_w);

    /* 3. Sinusoidal UV positional embedding */
    {
        float aspect = (float)model_w / (float)model_h;
        float diag = sqrtf(aspect * aspect + 1.0f);
        float span_x = aspect / diag;
        float span_y = 1.0f / diag;
        opSinusoidalUv(r, r->dpt_tmp2, feat_half, model_h, model_w, span_x, span_y, 0.1f);
    }

    /* 4. out_0: Conv3x3 + ReLU */
    opConv2d(r, r->dpt_tmp, r->dpt_tmp2, dw.out_0_w, dw.out_0_b,
             model_h, model_w, feat_half, out_mid, 3, 3, 1, 1);
    opRelu(r, r->dpt_tmp, out_mid * model_h * model_w);

    /* out_0+relu (verbose >= 3) */
    if (r->verbose >= 3) {
        if (r->batched) { r->runner.endRecordingAndSubmitFenced(r->batch_fence); r->runner.waitForFence(r->batch_fence); r->runner.resetFence(r->batch_fence); }
        int nn = out_mid * model_h * model_w;
        std::vector<float> dbg(nn);
        r->runner.downloadFromDeviceLocal( r->dpt_tmp, dbg.data(), nn * sizeof(float));
        float mn=dbg[0], mx=dbg[0], sm=0;
        for (int i=0;i<nn;i++){if(dbg[i]<mn)mn=dbg[i];if(dbg[i]>mx)mx=dbg[i];sm+=dbg[i];}
        fprintf(stderr, "DA3: out_0+relu[%d] (%dx%d, out_mid=%d): min=%.4f max=%.4f mean=%.6f\n", nn, model_h, model_w, out_mid, mn, mx, sm/nn);
        fflush(stderr);
        if (r->batched) { r->runner.resetDynamicDescriptorPool(); r->runner.beginRecording(); }
    }
    /* 5. out_2: Conv1x1 -> 2 channels */
    opConv2d(r, r->dpt_out, r->dpt_tmp, dw.out_2_w, dw.out_2_b,
             model_h, model_w, out_mid, 2, 1, 1, 1, 0);
    fh = model_h; fw = model_w;

    /* Depth activation */
    opDepthActivation(r, r->dpt_out, fh * fw);

    /* depth activation output stats (verbose >= 3) */
    if (r->verbose >= 3) {
        if (r->batched) { r->runner.endRecordingAndSubmitFenced(r->batch_fence); r->runner.waitForFence(r->batch_fence); r->runner.resetFence(r->batch_fence); }
        int dn = 2 * fh * fw;
        std::vector<float> dbg(dn);
        r->runner.downloadFromDeviceLocal( r->dpt_out, dbg.data(), dn * sizeof(float));
        float mn=dbg[0], mx=dbg[0], sm=0;
        for (int i=0;i<fh*fw;i++){if(dbg[i]<mn)mn=dbg[i];if(dbg[i]>mx)mx=dbg[i];sm+=dbg[i];}
        fprintf(stderr, "DA3: depth logits: min=%.4f max=%.4f mean=%.6f\n", mn, mx, sm/(fh*fw));
        fflush(stderr);
        if (r->batched) { r->runner.resetDynamicDescriptorPool(); r->runner.beginRecording(); }
    }

    /* Bilinear upsample to original resolution */
    {
        size_t result_sz = (size_t)2 * img_h * img_w * sizeof(float);
        int npix = img_w * img_h;
        if (result_sz > r->result_cap) {
            if (isValid(r->result_buf)) r->runner.destroyBuffer(r->result_buf);
            r->result_buf = createGpuBuffer(r->runner, result_sz);
            r->result_cap = result_sz;
        }
        opBilinear(r, r->result_buf, r->dpt_out, 2, fh, fw, img_h, img_w);

        /* Ensure download staging buffer is large enough */
        if (result_sz > r->download_staging_cap) {
            if (isValid(r->download_staging)) r->runner.destroyBuffer(r->download_staging);
            r->runner.createStagingBuffer(result_sz, r->download_staging);
            r->download_staging_cap = result_sz;
        }

        /* Record copy from device-local result_buf to staging */
        r->runner.recordCopyBuffer(r->result_buf, r->download_staging, result_sz);

        /* End DPT batch */
        r->runner.endRecordingAndSubmitFenced(r->batch_fence);
        r->runner.waitForFence(r->batch_fence);
        r->runner.resetFence(r->batch_fence);
        r->runner.resetDynamicDescriptorPool();
        r->batched = false;

        /* Read from staging */
        result.width = img_w;
        result.height = img_h;
        result.depth = (float *)malloc((size_t)npix * sizeof(float));
        result.confidence = (float *)malloc((size_t)npix * sizeof(float));
        {
            void *ptr = nullptr;
            r->runner.mapBuffer(r->download_staging, &ptr);
            memcpy(result.depth, ptr, (size_t)npix * sizeof(float));
            memcpy(result.confidence, (float *)ptr + npix, (size_t)npix * sizeof(float));
            r->runner.unmapBuffer(r->download_staging);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
    fprintf(stderr, "DA3: GPU DPT head: %.1f ms\n", (t1 - t0) * 1000);

    /* ---- Aux DPT (Phase 2): Rays + Sky Segmentation ---- */
    if (r->dpt_aux.loaded && (output_flags & DA3_OUTPUT_RAYS)) {
        clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

        auto &aux = r->dpt_aux;

        /* Bottom-up aux RefineNet fusion (reuses dpt_adapted from main DPT) */
        /* Level 3 (deepest) */
        gpuRefineNetW(r, r->dpt_adapted[3], sp_h[3], sp_w[3],
                       nullptr, 0, 0, feat, r->dpt_fused,
                       aux.fuse_out_w[3], aux.fuse_out_b[3],
                       aux.fuse_rcu1_c1_w[3], aux.fuse_rcu1_c1_b[3],
                       aux.fuse_rcu1_c2_w[3], aux.fuse_rcu1_c2_b[3],
                       aux.fuse_rcu2_c1_w[3], aux.fuse_rcu2_c1_b[3],
                       aux.fuse_rcu2_c2_w[3], aux.fuse_rcu2_c2_b[3],
                       aux.has_rcu1[3], aux.has_rcu2[3],
                       sp_h[2], sp_w[2]);
        int aux_fh = sp_h[2], aux_fw = sp_w[2];

        /* Level 2 */
        gpuRefineNetW(r, r->dpt_adapted[2], sp_h[2], sp_w[2],
                       &r->dpt_fused, aux_fh, aux_fw, feat, r->dpt_fused,
                       aux.fuse_out_w[2], aux.fuse_out_b[2],
                       aux.fuse_rcu1_c1_w[2], aux.fuse_rcu1_c1_b[2],
                       aux.fuse_rcu1_c2_w[2], aux.fuse_rcu1_c2_b[2],
                       aux.fuse_rcu2_c1_w[2], aux.fuse_rcu2_c1_b[2],
                       aux.fuse_rcu2_c2_w[2], aux.fuse_rcu2_c2_b[2],
                       aux.has_rcu1[2], aux.has_rcu2[2],
                       sp_h[1], sp_w[1]);
        aux_fh = sp_h[1]; aux_fw = sp_w[1];

        /* Level 1 */
        gpuRefineNetW(r, r->dpt_adapted[1], sp_h[1], sp_w[1],
                       &r->dpt_fused, aux_fh, aux_fw, feat, r->dpt_fused,
                       aux.fuse_out_w[1], aux.fuse_out_b[1],
                       aux.fuse_rcu1_c1_w[1], aux.fuse_rcu1_c1_b[1],
                       aux.fuse_rcu1_c2_w[1], aux.fuse_rcu1_c2_b[1],
                       aux.fuse_rcu2_c1_w[1], aux.fuse_rcu2_c1_b[1],
                       aux.fuse_rcu2_c2_w[1], aux.fuse_rcu2_c2_b[1],
                       aux.has_rcu1[1], aux.has_rcu2[1],
                       sp_h[0], sp_w[0]);
        aux_fh = sp_h[0]; aux_fw = sp_w[0];

        /* Level 0 */
        gpuRefineNetW(r, r->dpt_adapted[0], sp_h[0], sp_w[0],
                       &r->dpt_fused, aux_fh, aux_fw, feat, r->dpt_fused,
                       aux.fuse_out_w[0], aux.fuse_out_b[0],
                       aux.fuse_rcu1_c1_w[0], aux.fuse_rcu1_c1_b[0],
                       aux.fuse_rcu1_c2_w[0], aux.fuse_rcu1_c2_b[0],
                       aux.fuse_rcu2_c1_w[0], aux.fuse_rcu2_c1_b[0],
                       aux.fuse_rcu2_c2_w[0], aux.fuse_rcu2_c2_b[0],
                       aux.has_rcu1[0], aux.has_rcu2[0],
                       sp_h[0] * 2, sp_w[0] * 2);

        /* Output conv chain on last level (level 3, finest resolution) */
        {
            int lv = 3;
            int oh = sp_h[0], ow = sp_w[0];

            /* output_conv1_aux: chain of Conv2d (no activation) */
            BufInfo *cur = &r->aux_scratch;
            opCopy(r, r->aux_scratch, r->dpt_fused, feat * oh * ow);
            int ci = feat;
            for (int ci_idx = 0; ci_idx < aux.oc1_count[lv]; ci_idx++) {
                int co = aux.oc1_co[lv][ci_idx];
                BufInfo &dst_buf = (ci_idx % 2 == 0) ? r->dpt_tmp : r->aux_scratch;
                BufInfo &src_buf = (ci_idx == 0) ? *cur : ((ci_idx % 2 == 0) ? r->aux_scratch : r->dpt_tmp);
                opConv2d(r, dst_buf, src_buf,
                         aux.oc1_w[lv][ci_idx], aux.oc1_b[lv][ci_idx],
                         oh, ow, ci, co, 3, 3, 1, 1);
                ci = co;
                cur = &dst_buf;
            }

            /* output_conv2_aux: Conv2d(128,32,3) + ChannelLayerNorm(32) + ReLU + Conv2d(32,7,1) */
            opConv2d(r, r->dpt_tmp2, *cur,
                     aux.oc2_conv_w[lv], aux.oc2_conv_b[lv],
                     oh, ow, ci, 32, 3, 3, 1, 1);
            /* Channel LayerNorm (create default w=1,b=0 if not loaded) */
            {
                BufInfo ln_w = aux.oc2_gn_w[lv];
                BufInfo ln_b = aux.oc2_gn_b[lv];
                bool created_w = false, created_b = false;
                if (!isValid(ln_w)) {
                    ln_w = createGpuBuffer(r->runner, 32 * sizeof(float));
                    std::vector<float> ones(32, 1.0f);
                    r->runner.uploadToDeviceLocal(ln_w, ones.data(), 32 * sizeof(float));
                    created_w = true;
                }
                if (!isValid(ln_b)) {
                    ln_b = createGpuBuffer(r->runner, 32 * sizeof(float));
                    std::vector<float> zeros_b(32, 0.0f);
                    r->runner.uploadToDeviceLocal(ln_b, zeros_b.data(), 32 * sizeof(float));
                    created_b = true;
                }
                opChannelLayerNorm(r, r->aux_scratch, r->dpt_tmp2, ln_w, ln_b, 32, oh * ow);
                if (created_w) r->runner.destroyBuffer(ln_w);
                if (created_b) r->runner.destroyBuffer(ln_b);
            }
            /* ReLU */
            opRelu(r, r->aux_scratch, 32 * oh * ow);
            /* Final conv: 32 -> 7 channels */
            opConv2d(r, r->aux_out, r->aux_scratch,
                     aux.oc2_out_w[lv], aux.oc2_out_b[lv],
                     oh, ow, 32, 7, 1, 1, 1, 0);
        }

        /* Bilinear upsample 7 channels to original resolution */
        int npix = img_w * img_h;
        {
            int oh = sp_h[0], ow = sp_w[0];
            size_t aux_full_sz = (size_t)7 * npix * sizeof(float);
            if (aux_full_sz > r->result_cap) {
                if (isValid(r->result_buf)) r->runner.destroyBuffer(r->result_buf);
                r->result_buf = createGpuBuffer(r->runner, aux_full_sz);
                r->result_cap = aux_full_sz;
            }
            opBilinear(r, r->result_buf, r->aux_out, 7, oh, ow, img_h, img_w);

            std::vector<float> h_aux_full((size_t)7 * npix);
            r->runner.downloadFromDeviceLocal( r->result_buf, h_aux_full.data(), aux_full_sz);

            /* Split: rays = channels 0-5 (linear), ray_confidence = channel 6 (expp1) */
            result.rays = (float *)malloc((size_t)6 * npix * sizeof(float));
            result.ray_confidence = (float *)malloc((size_t)npix * sizeof(float));
            memcpy(result.rays, h_aux_full.data(), (size_t)6 * npix * sizeof(float));
            for (int i = 0; i < npix; i++)
                result.ray_confidence[i] = expf(h_aux_full[6 * npix + i]) + 1.0f;
            result.sky_seg = nullptr;
            result.has_rays = 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (r->verbose >= 1) fprintf(stderr, "DA3: Aux DPT (rays+sky): %.1f ms\n", (t1 - t0) * 1000);
    }

    /* ---- GSDPT (Phase 4): 3D Gaussian Estimation ---- */
    if (r->gsdpt.loaded && (output_flags & DA3_OUTPUT_GAUSSIANS)) {
        clock_gettime(CLOCK_MONOTONIC, &ts); t0 = ts.tv_sec + ts.tv_nsec * 1e-9;

        auto &gs = r->gsdpt;
        auto &gdw = gs.dpt;
        int gs_oc = gs.gs_out_channels;
        if (gs_oc < 2) gs_oc = 38;

        /* Run images_merger: 3 stride-2 Conv2d on normalized image */
        {
            BufInfo *cur_buf = &r->img_norm;
            BufInfo *bufs[2] = {&r->dpt_tmp, &r->dpt_tmp2};
            int mh = target_h, mw = target_w;
            for (int mi = 0; mi < 3; mi++) {
                if (!isValid(gs.merger_w[mi])) break;
                int mci = gs.merger_ci[mi];
                int mco = gs.merger_co[mi];
                int moh = (mh + 2 * 1 - 3) / 2 + 1;
                int mow = (mw + 2 * 1 - 3) / 2 + 1;
                BufInfo &dst = (mi == 2) ? r->gs_merged : *bufs[mi % 2];
                opConv2d(r, dst, *cur_buf, gs.merger_w[mi], gs.merger_b[mi],
                         mh, mw, mci, mco, 3, 3, 2, 1);
                /* SiLU activation after first two conv layers (not after last) */
                if (mi < 2)
                    opSilu(r, dst, mco * moh * mow);
                cur_buf = &dst;
                mh = moh; mw = mow;
            }
        }

        /* GSDPT token processing + projection (same pipeline as main DPT, with GS weights) */
        for (int fi = 0; fi < 4; fi++) {
            int oc_val = r->head_out_channels[fi];

            opDptClsConcat(r, r->dpt_cat, r->features[fi], np, dim);
            if (isValid(gdw.norm_w))
                opLayerNorm(r, r->dpt_ln, r->dpt_cat, gdw.norm_w, gdw.norm_b, np, head_dim_in);
            else
                opCopy(r, r->dpt_ln, r->dpt_cat, np * head_dim_in);

            opMatmulBiasF16(r, r->dpt_proj, gdw.proj_w[fi], r->dpt_ln, gdw.proj_b[fi],
                         np, oc_val, head_dim_in);

            if (fi == 0)
                opDeconvGemmScatter(r, r->dpt_spatial[0], r->dpt_proj,
                                    gdw.upsample_0_w, gdw.upsample_0_b, r->dpt_ln,
                                    oc_val, oc_val, gh, gw, 4, 4, 4);
            else if (fi == 1)
                opDeconvGemmScatter(r, r->dpt_spatial[1], r->dpt_proj,
                                    gdw.upsample_1_w, gdw.upsample_1_b, r->dpt_ln,
                                    oc_val, oc_val, gh, gw, 2, 2, 2);
            else if (fi == 2) {
                opTokToChw(r, r->dpt_chw, r->dpt_proj, oc_val, gh, gw);
                opCopy(r, r->dpt_spatial[2], r->dpt_chw, oc_val * gh * gw);
            } else {
                opTokToChw(r, r->dpt_chw, r->dpt_proj, oc_val, gh, gw);
                opConv2d(r, r->dpt_spatial[3], r->dpt_chw, gdw.downsample_w, gdw.downsample_b,
                         gh, gw, oc_val, oc_val, 3, 3, 2, 1);
            }

            /* Adapter conv (no bias) */            opConv2d(r, r->dpt_adapted[fi], r->dpt_spatial[fi], gdw.adapter_w[fi], r->zero_bias,
                     sp_h[fi], sp_w[fi], oc_val, feat, 3, 3, 1, 1);
        }

        /* Bottom-up RefineNet fusion with GSDPT weights */
        gpuRefineNetW(r, r->dpt_adapted[3], sp_h[3], sp_w[3],
                       nullptr, 0, 0, feat, r->dpt_fused,
                       gdw.fuse_out_w[3], gdw.fuse_out_b[3],
                       gdw.fuse_rcu1_c1_w[3], gdw.fuse_rcu1_c1_b[3],
                       gdw.fuse_rcu1_c2_w[3], gdw.fuse_rcu1_c2_b[3],
                       gdw.fuse_rcu2_c1_w[3], gdw.fuse_rcu2_c1_b[3],
                       gdw.fuse_rcu2_c2_w[3], gdw.fuse_rcu2_c2_b[3],
                       gdw.has_rcu1[3], gdw.has_rcu2[3],
                       sp_h[2], sp_w[2]);
        int gs_fh = sp_h[2], gs_fw = sp_w[2];
        gpuRefineNetW(r, r->dpt_adapted[2], sp_h[2], sp_w[2],
                       &r->dpt_fused, gs_fh, gs_fw, feat, r->dpt_fused,
                       gdw.fuse_out_w[2], gdw.fuse_out_b[2],
                       gdw.fuse_rcu1_c1_w[2], gdw.fuse_rcu1_c1_b[2],
                       gdw.fuse_rcu1_c2_w[2], gdw.fuse_rcu1_c2_b[2],
                       gdw.fuse_rcu2_c1_w[2], gdw.fuse_rcu2_c1_b[2],
                       gdw.fuse_rcu2_c2_w[2], gdw.fuse_rcu2_c2_b[2],
                       gdw.has_rcu1[2], gdw.has_rcu2[2],
                       sp_h[1], sp_w[1]);
        gs_fh = sp_h[1]; gs_fw = sp_w[1];
        gpuRefineNetW(r, r->dpt_adapted[1], sp_h[1], sp_w[1],
                       &r->dpt_fused, gs_fh, gs_fw, feat, r->dpt_fused,
                       gdw.fuse_out_w[1], gdw.fuse_out_b[1],
                       gdw.fuse_rcu1_c1_w[1], gdw.fuse_rcu1_c1_b[1],
                       gdw.fuse_rcu1_c2_w[1], gdw.fuse_rcu1_c2_b[1],
                       gdw.fuse_rcu2_c1_w[1], gdw.fuse_rcu2_c1_b[1],
                       gdw.fuse_rcu2_c2_w[1], gdw.fuse_rcu2_c2_b[1],
                       gdw.has_rcu1[1], gdw.has_rcu2[1],
                       sp_h[0], sp_w[0]);
        gs_fh = sp_h[0]; gs_fw = sp_w[0];
        gpuRefineNetW(r, r->dpt_adapted[0], sp_h[0], sp_w[0],
                       &r->dpt_fused, gs_fh, gs_fw, feat, r->dpt_fused,
                       gdw.fuse_out_w[0], gdw.fuse_out_b[0],
                       gdw.fuse_rcu1_c1_w[0], gdw.fuse_rcu1_c1_b[0],
                       gdw.fuse_rcu1_c2_w[0], gdw.fuse_rcu1_c2_b[0],
                       gdw.fuse_rcu2_c1_w[0], gdw.fuse_rcu2_c1_b[0],
                       gdw.fuse_rcu2_c2_w[0], gdw.fuse_rcu2_c2_b[0],
                       gdw.has_rcu1[0], gdw.has_rcu2[0],
                       sp_h[0] * 2, sp_w[0] * 2);
        gs_fh = sp_h[0]; gs_fw = sp_w[0];

        /* Output convolutions */
        int gs_feat_half = feat / 2; if (gs_feat_half < 1) gs_feat_half = 1;
        int gs_out_mid = gdw.out_mid > 0 ? gdw.out_mid : gs_feat_half;

        /* Neck: Conv2d(feat, feat/2, 3), NO ReLU */
        opConv2d(r, r->dpt_tmp, r->dpt_fused, gdw.neck_w, gdw.neck_b,
                 gs_fh, gs_fw, feat, gs_feat_half, 3, 3, 1, 1);

        /* Inject merger features: upsample + add */
        if (isValid(r->gs_merged) && r->gs_merger_h > 0) {
            opBilinear(r, r->dpt_tmp2, r->gs_merged,
                       gs_feat_half, r->gs_merger_h, r->gs_merger_w, gs_fh, gs_fw);
            opAdd(r, r->dpt_tmp, r->dpt_tmp2, gs_feat_half * gs_fh * gs_fw);
        }

        /* out_0: Conv2d(feat/2, out_mid, 3) + ReLU */
        opConv2d(r, r->dpt_tmp2, r->dpt_tmp, gdw.out_0_w, gdw.out_0_b,
                 gs_fh, gs_fw, gs_feat_half, gs_out_mid, 3, 3, 1, 1);
        opRelu(r, r->dpt_tmp2, gs_out_mid * gs_fh * gs_fw);

        /* out_2: Conv1x1(out_mid, gs_oc) */
        opConv2d(r, r->gs_out, r->dpt_tmp2, gdw.out_2_w, gdw.out_2_b,
                 gs_fh, gs_fw, gs_out_mid, gs_oc, 1, 1, 1, 0);

        /* Bilinear upsample to original resolution */
        int npix = img_w * img_h;
        {
            size_t gs_full_sz = (size_t)gs_oc * npix * sizeof(float);
            if (gs_full_sz > r->result_cap) {
                if (isValid(r->result_buf)) r->runner.destroyBuffer(r->result_buf);
                r->result_buf = createGpuBuffer(r->runner, gs_full_sz);
                r->result_cap = gs_full_sz;
            }
            opBilinear(r, r->result_buf, r->gs_out, gs_oc, gs_fh, gs_fw, img_h, img_w);
            result.gaussians = (float *)malloc(gs_full_sz);
            r->runner.downloadFromDeviceLocal( r->result_buf, result.gaussians, gs_full_sz);
            result.has_gaussians = 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &ts); t1 = ts.tv_sec + ts.tv_nsec * 1e-9;
        if (r->verbose >= 1)
            fprintf(stderr, "DA3: GSDPT (%d channels): %.1f ms\n", gs_oc, (t1 - t0) * 1000);
    }

    return result;
}

/* ======================================================================== */
/* Public API: predict (depth-only convenience wrapper)                     */
/* ======================================================================== */

da3_vk_result vulkan_da3_predict(vulkan_da3_runner *r, const uint8_t *rgb, int w, int h) {
    da3_full_result full = vulkan_da3_predict_full(r, rgb, w, h, DA3_OUTPUT_DEPTH, nullptr);
    da3_vk_result result = {};
    result.depth = full.depth;
    result.confidence = full.confidence;
    result.width = full.width;
    result.height = full.height;
    /* Don't free depth/confidence since they're transferred to the simple result */
    free(full.rays); free(full.ray_confidence); free(full.sky_seg);
    free(full.gaussians); free(full.metric_depth);
    return result;
}

/* ======================================================================== */
/* Public API: init                                                         */
/* ======================================================================== */

vulkan_da3_runner *vulkan_da3_init(int device_id, int verbose, const char *shader_dir) {
    auto *r = new vulkan_da3_runner();
    r->verbose = verbose;
    r->shader_dir = shader_dir ? shader_dir : ".";

    if (!vl_cpp::vulkan::InitializeVulkan()) {
        fprintf(stderr, "DA3: failed to initialize Vulkan loader\n");
        delete r;
        return nullptr;
    }

    if (!r->runner.initialize(false)) {
        fprintf(stderr, "DA3: Vulkan runner initialization failed\n");
        delete r;
        return nullptr;
    }

    if ((uint32_t)device_id >= r->runner.getDeviceCount()) {
        fprintf(stderr, "DA3: invalid device ID %d (have %u)\n", device_id, r->runner.getDeviceCount());
        delete r;
        return nullptr;
    }
    if (!r->runner.selectDevice(device_id)) {
        fprintf(stderr, "DA3: failed to select device %d\n", device_id);
        delete r;
        return nullptr;
    }

    r->runner.createDynamicDescriptorPool(1024);

    if (!createPipelines(r)) {
        fprintf(stderr, "DA3: pipeline creation failed\n");
        delete r;
        return nullptr;
    }

    r->runner.createFence(r->batch_fence);

    if (verbose >= 1) fprintf(stderr, "DA3: Vulkan runner initialized (device %d)\n", device_id);
    return r;
}

/* ======================================================================== */
/* Public API: free                                                         */
/* ======================================================================== */

void vulkan_da3_free(vulkan_da3_runner *r) {
    if (!r) return;

    /* Free CPU pos_embed copy */
    free(r->h_pos_embed_orig);

    /* Destroy staging buffers */
    if (isValid(r->upload_staging)) r->runner.destroyBuffer(r->upload_staging);
    if (isValid(r->download_staging)) r->runner.destroyBuffer(r->download_staging);

    /* Destroy batch fence */
    if (r->batch_fence != VK_NULL_HANDLE) r->runner.destroyFence(r->batch_fence);

    /* The VulkanRunner destructor handles Vulkan cleanup */
    delete r;
}
