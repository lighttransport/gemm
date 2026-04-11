// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan Flux.2 Klein Text-to-Image Runner
//
// Pipeline: Input projection -> DiT (5 double + 20 single blocks) -> Output projection
// Uses pre-compiled SPIR-V shaders and DEVICE_LOCAL SSBOs with batched dispatch.
// F32 weights on GPU, F32 compute.
//
// Port of cuda_flux2_runner.h for cross-platform Vulkan compute.
//

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
/* ggml_dequant implementation is in ggml_dequant_impl.c (C, uses 'restrict') */
#include "../../common/ggml_dequant.h"
#define FLUX2_DIT_IMPLEMENTATION
#include "../../common/flux2_klein_dit.h"
#define FLUX2_VAE_IMPLEMENTATION
#include "../../common/flux2_klein_vae.h"

#include "vulkan_flux2_runner.h"
#include "../deps/vulkan-runner.hh"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <ctime>

/* ======================================================================== */
/* Types                                                                    */
/* ======================================================================== */

using VulkanRunner = vl_cpp::vulkan::VulkanComputeRunner;
using BufInfo = VulkanRunner::BufferInfo;
using Pipeline = VulkanRunner::ComputePipeline;

/* ======================================================================== */
/* GPU weight structures                                                    */
/* ======================================================================== */

struct flux2_vk_stream_t {
    BufInfo qkv_w;     /* [3H, H] F32 */
    BufInfo proj_w;    /* [H, H] F32 */
    BufInfo mlp_up_w;  /* [2*n_ff, H] F32 */
    BufInfo mlp_dn_w;  /* [H, n_ff] F32 */
    BufInfo q_norm;    /* [head_dim] F32 */
    BufInfo k_norm;    /* [head_dim] F32 */
};

struct flux2_vk_dblk_t {
    flux2_vk_stream_t img, txt;
};

struct flux2_vk_sblk_t {
    BufInfo linear1_w;   /* [3H+2*n_ff, H] F32 */
    BufInfo l2_attn_w;   /* [H, H] F32 */
    BufInfo l2_mlp_w;    /* [H, n_ff] F32 */
    BufInfo q_norm;      /* [head_dim] F32 */
    BufInfo k_norm;      /* [head_dim] F32 */
};

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

struct vulkan_flux2_runner {
    VulkanRunner runner;
    int verbose;
    std::string shader_dir;

    /* Pipelines */
    Pipeline pipe_gemm;           /* 4 bufs: W,X,Y,bias; push: {n_out,n_in,n_tok,has_bias} = 16B */
    Pipeline pipe_adaln;          /* 4 bufs: x,out,shift,scale; push: {N,dim} = 8B */
    Pipeline pipe_rmsnorm_ph;     /* 2 bufs: x,w; push: {N,n_heads,head_dim} = 12B */
    Pipeline pipe_rope_img;       /* 1 buf: x; push: {n_tok,n_heads,head_dim,lat_w,theta} = 20B */
    Pipeline pipe_rope_txt;       /* 1 buf: x; push: {n_tok,n_heads,head_dim,theta} = 16B */
    Pipeline pipe_swiglu;         /* 2 bufs: in,out; push: {n_tok,mlp_h} = 8B */
    Pipeline pipe_gated_add;      /* 3 bufs: x,proj,gate; push: {N,dim} = 8B */
    Pipeline pipe_bf16_trunc;     /* 1 buf: x; push: {n} = 4B */
    Pipeline pipe_attn;           /* 4 bufs: Q,K,V,out; push: {N,n_heads,head_dim,scale} = 16B */
    Pipeline pipe_silu;           /* 1 buf: x; push: {n} = 4B */
    Pipeline pipe_add;            /* 2 bufs: dst,src; push: {n} = 4B */
    Pipeline pipe_layernorm;      /* 4 bufs: src,dst,w,b; push: {n_tokens,dim,eps} = 12B */
    /* VAE pipelines */
    Pipeline pipe_vae_im2col;     /* 2 bufs: in,col; push: {ci,H,W} = 12B */
    Pipeline pipe_vae_transpose;  /* 2 bufs: in,out; push: {rows,cols} = 8B */
    Pipeline pipe_vae_add_bias;   /* 2 bufs: out,bias; push: {co,spatial} = 8B */
    Pipeline pipe_vae_conv1;      /* 4 bufs: in,out,w,bias; push: {ci,sp,co,has_bias} = 16B */
    Pipeline pipe_vae_gnsilu;     /* 4 bufs: in,out,gamma,beta; push: {C,sp,ng,silu} = 16B */
    Pipeline pipe_vae_up2x;       /* 2 bufs: in,out; push: {C,H,W} = 12B */

    /* CPU model (for arch params + VAE fallback) */
    flux2_dit_model *dit;
    flux2_vae_model *vae;

    int H, nH, hd, n_ff, pin, txt_dim, n_dbl, n_sgl;

    /* Global GPU weights */
    BufInfo d_img_in_w, d_img_in_b, d_txt_in_w, d_txt_in_b;
    BufInfo d_t_fc1_w, d_t_fc1_b, d_t_fc2_w, d_t_fc2_b;
    BufInfo d_mod_img_w, d_mod_txt_w, d_mod_sgl_w;
    BufInfo d_out_mod_w, d_out_proj_w;

    std::vector<flux2_vk_dblk_t> gpu_dblk;
    std::vector<flux2_vk_sblk_t> gpu_sblk;

    /* Activation buffers */
    BufInfo d_img, d_txt, d_joint;
    BufInfo d_temb, d_temb_silu;
    BufInfo d_mod_img_v, d_mod_txt_v, d_mod_sgl_v;
    BufInfo d_q, d_k, d_v, d_attn_out;
    BufInfo d_scratch1, d_scratch2, d_scratch3;
    BufInfo d_img_in_buf, d_txt_in_buf;
    BufInfo d_traw;
    int max_tok;
    bool gpu_loaded;
};

/* ======================================================================== */
/* Utility                                                                  */
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

static bool isValid(const BufInfo &buf) {
    return buf.buffer != VK_NULL_HANDLE;
}

static BufInfo uploadF32(VulkanRunner &runner, const float *data, int n) {
    if (!data || n <= 0) return BufInfo{};
    size_t bytes = (size_t)n * sizeof(float);
    BufInfo buf = createGpuBuffer(runner, bytes);
    if (isValid(buf))
        runner.uploadToDeviceLocal(buf, data, bytes);
    return buf;
}

static BufInfo uploadMat(VulkanRunner &runner, const flux2_mat *m) {
    return uploadF32(runner, m->w, m->rows * m->cols);
}

/* ======================================================================== */
/* Dispatch helpers                                                         */
/* ======================================================================== */

static void dispatchBatched(vulkan_flux2_runner *r, Pipeline &pipe,
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

static void beginBatch(vulkan_flux2_runner *r) { r->runner.beginRecording(); }
static void endBatch(vulkan_flux2_runner *r) {
    r->runner.endRecordingAndSubmit();
    r->runner.waitForCompletion();
    r->runner.resetDynamicDescriptorPool();
}

/* ---- Op wrappers ---- */

/* For offset buffers: create a sub-buffer view at byte offset.
 * Vulkan doesn't support arbitrary offsets on SSBOs directly.
 * We work around this by creating separate BufInfo for each weight slice at upload time,
 * or by using offset-aware dispatch patterns. For simplicity, weight slices are
 * pre-split during upload (QKV split into separate Q, K, V buffers). */

/* We need separate Q, K, V weight buffers per block since Vulkan SSBOs
 * don't support arbitrary pointer offsets like CUDA device pointers.
 * The stream block stores them as a combined [3H, H] QKV matrix on CPU,
 * so we split at upload time. */

struct flux2_vk_stream_split_t {
    BufInfo q_w, k_w, v_w;  /* [H, H] each */
    BufInfo proj_w;          /* [H, H] */
    BufInfo mlp_up_w;        /* [2*n_ff, H] */
    BufInfo mlp_dn_w;        /* [H, n_ff] */
    BufInfo q_norm, k_norm;  /* [head_dim] each */
};

struct flux2_vk_dblk_split_t {
    flux2_vk_stream_split_t img, txt;
};

struct flux2_vk_sblk_split_t {
    BufInfo q_w, k_w, v_w, gu_w;  /* linear1 split: [H,H], [H,H], [H,H], [2*n_ff,H] */
    BufInfo l2_attn_w, l2_mlp_w;
    BufInfo q_norm, k_norm;
};

/* Extended runner with split weights */
struct vulkan_flux2_runner_ext {
    std::vector<flux2_vk_dblk_split_t> dblk;
    std::vector<flux2_vk_sblk_split_t> sblk;
};

/* Global map to store per-runner extended data */
static std::map<vulkan_flux2_runner*, vulkan_flux2_runner_ext> g_ext_map;

static vulkan_flux2_runner_ext *get_ext(vulkan_flux2_runner *r) {
    return &g_ext_map[r];
}

static void op_gemm(vulkan_flux2_runner *r, BufInfo &Y, BufInfo &W,
                    BufInfo &X, BufInfo &bias, int n_out, int n_in, int n_tok) {
    struct { uint32_t n_out, n_in, n_tok, has_bias; } pc;
    pc.n_out = n_out; pc.n_in = n_in; pc.n_tok = n_tok;
    pc.has_bias = isValid(bias) ? 1 : 0;
    uint32_t gx = (n_out + 63) / 64;
    uint32_t gy = (n_tok + 15) / 16;
    dispatchBatched(r, r->pipe_gemm, {W, X, Y, bias}, &pc, sizeof(pc), gx, gy);
}

static void op_adaln(vulkan_flux2_runner *r, BufInfo &out, BufInfo &x,
                     BufInfo &shift, BufInfo &scale, int N, int dim) {
    struct { uint32_t N, dim; } pc;
    pc.N = N; pc.dim = dim;
    dispatchBatched(r, r->pipe_adaln, {x, out, shift, scale}, &pc, sizeof(pc), (uint32_t)N);
}

static void op_silu(vulkan_flux2_runner *r, BufInfo &x, int n) {
    struct { uint32_t n; } pc;
    pc.n = n;
    dispatchBatched(r, r->pipe_silu, {x}, &pc, sizeof(pc), (uint32_t)((n+255)/256));
}

static void op_add(vulkan_flux2_runner *r, BufInfo &dst, BufInfo &src, int n) {
    struct { uint32_t n; } pc;
    pc.n = n;
    dispatchBatched(r, r->pipe_add, {dst, src}, &pc, sizeof(pc), (uint32_t)((n+255)/256));
}

static void op_rmsnorm_ph(vulkan_flux2_runner *r, BufInfo &x, BufInfo &w,
                          int N, int n_heads, int head_dim) {
    struct { uint32_t N, n_heads, head_dim; } pc;
    pc.N = N; pc.n_heads = n_heads; pc.head_dim = head_dim;
    dispatchBatched(r, r->pipe_rmsnorm_ph, {x, w}, &pc, sizeof(pc),
                    (uint32_t)N, (uint32_t)n_heads);
}

static void op_gated_add(vulkan_flux2_runner *r, BufInfo &x, BufInfo &proj,
                         BufInfo &gate, int N, int dim) {
    struct { uint32_t N, dim; } pc;
    pc.N = N; pc.dim = dim;
    int total = N * dim;
    dispatchBatched(r, r->pipe_gated_add, {x, proj, gate}, &pc, sizeof(pc),
                    (uint32_t)((total+255)/256));
}

static void op_attn(vulkan_flux2_runner *r, BufInfo &out, BufInfo &q,
                    BufInfo &k, BufInfo &v, int n_tok, int n_heads, int head_dim) {
    struct { uint32_t N, n_heads, head_dim; float scale; } pc;
    pc.N = n_tok; pc.n_heads = n_heads; pc.head_dim = head_dim;
    pc.scale = 1.0f / sqrtf((float)head_dim);
    uint32_t gy = (n_tok + 3) / 4;  /* 4 queries per workgroup */
    dispatchBatched(r, r->pipe_attn, {q, k, v, out}, &pc, sizeof(pc),
                    (uint32_t)n_heads, gy);
}

static void op_swiglu(vulkan_flux2_runner *r, BufInfo &out, BufInfo &in,
                      int n_tok, int mlp_h) {
    struct { uint32_t n_tok, mlp_h; } pc;
    pc.n_tok = n_tok; pc.mlp_h = mlp_h;
    uint32_t gx = (mlp_h + 255) / 256;
    dispatchBatched(r, r->pipe_swiglu, {in, out}, &pc, sizeof(pc), (uint32_t)n_tok, gx);
}

static void op_rope_img(vulkan_flux2_runner *r, BufInfo &x, int n_tok,
                        int n_heads, int hd, int lat_w, float theta) {
    struct { uint32_t n_tok, n_heads, head_dim, lat_w; float theta; } pc;
    pc.n_tok = n_tok; pc.n_heads = n_heads; pc.head_dim = hd;
    pc.lat_w = lat_w; pc.theta = theta;
    uint32_t gx = (hd/2 + 31) / 32;
    dispatchBatched(r, r->pipe_rope_img, {x}, &pc, sizeof(pc),
                    gx, (uint32_t)n_heads, (uint32_t)n_tok);
}

static void op_rope_txt(vulkan_flux2_runner *r, BufInfo &x, int n_tok,
                        int n_heads, int hd, float theta) {
    struct { uint32_t n_tok, n_heads, head_dim; float theta; } pc;
    pc.n_tok = n_tok; pc.n_heads = n_heads; pc.head_dim = hd;
    pc.theta = theta;
    uint32_t gx = (hd/2 + 31) / 32;
    dispatchBatched(r, r->pipe_rope_txt, {x}, &pc, sizeof(pc),
                    gx, (uint32_t)n_heads, (uint32_t)n_tok);
}

/* Sinusoidal timestep embedding (CPU, small) */
static void flux2_ts_embed(float *out, float t, int dim) {
    int half = dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i * logf(10000.0f) / (float)half);
        out[i] = cosf(t * freq);
        out[half + i] = sinf(t * freq);
    }
}

/* ======================================================================== */
/* Pipeline creation                                                        */
/* ======================================================================== */

static bool createPipelines(vulkan_flux2_runner *r) {
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
        fprintf(stderr, "flux2: failed to load shader: %s.spv (searched %s)\n",
                name.c_str(), r->shader_dir.c_str());
        return {};
    };

    auto createPipe = [&](const std::string &name, int nBuf, int pushSize, Pipeline &pipe) -> bool {
        auto spirv = loadSpv(name);
        if (spirv.empty()) return false;
        return r->runner.createComputePipelineWithPushConstants(spirv, makeBindings(nBuf), pushSize, pipe);
    };

    /* Flux2-specific shaders */
    if (!createPipe("shaders/flux2_gemm_f32",              4, 16, r->pipe_gemm)) return false;
    if (!createPipe("shaders/flux2_adaln_f32",             4, 8,  r->pipe_adaln)) return false;
    if (!createPipe("shaders/flux2_rmsnorm_per_head_f32",  2, 12, r->pipe_rmsnorm_ph)) return false;
    if (!createPipe("shaders/flux2_rope_img_f32",          1, 20, r->pipe_rope_img)) return false;
    if (!createPipe("shaders/flux2_rope_txt_f32",          1, 16, r->pipe_rope_txt)) return false;
    if (!createPipe("shaders/flux2_swiglu_f32",            2, 8,  r->pipe_swiglu)) return false;
    if (!createPipe("shaders/flux2_gated_add_f32",         3, 8,  r->pipe_gated_add)) return false;
    if (!createPipe("shaders/flux2_bf16_trunc_f32",        1, 4,  r->pipe_bf16_trunc)) return false;
    if (!createPipe("shaders/flux2_attn_flash_f32",        4, 16, r->pipe_attn)) return false;

    /* Reuse common shaders */
    if (!createPipe("shaders/add_f32",                     2, 4,  r->pipe_add)) return false;
    if (!createPipe("shaders/silu_f32",                    1, 4,  r->pipe_silu)) return false;

    /* VAE shaders */
    if (!createPipe("shaders/vae_conv2d_3x3_f32",         4, 20, r->pipe_vae_im2col)) return false;  /* reuses conv3 pipeline slot */
    if (!createPipe("shaders/vae_conv2d_1x1_f32",         4, 16, r->pipe_vae_conv1)) return false;
    if (!createPipe("shaders/vae_groupnorm_silu_f32",      4, 16, r->pipe_vae_gnsilu)) return false;
    if (!createPipe("shaders/vae_upsample2x_f32",          2, 12, r->pipe_vae_up2x)) return false;

    if (r->verbose >= 1) fprintf(stderr, "flux2: all pipelines created\n");
    return true;
}

/* ======================================================================== */
/* Init                                                                     */
/* ======================================================================== */

vulkan_flux2_runner *vulkan_flux2_init(int device_id, int verbose, const char *shader_dir) {
    auto *r = new vulkan_flux2_runner();
    r->verbose = verbose;
    r->shader_dir = shader_dir ? shader_dir : ".";
    r->dit = nullptr;
    r->vae = nullptr;
    r->max_tok = 0;
    r->gpu_loaded = false;

    if (!vl_cpp::vulkan::InitializeVulkan()) {
        fprintf(stderr, "flux2: failed to initialize Vulkan loader\n");
        delete r;
        return nullptr;
    }

    if (!r->runner.initialize(false)) {
        fprintf(stderr, "flux2: Vulkan init failed\n");
        delete r;
        return nullptr;
    }

    if (!r->runner.selectDevice((uint32_t)device_id)) {
        fprintf(stderr, "flux2: failed to select device %d\n", device_id);
        delete r;
        return nullptr;
    }

    if (verbose >= 1)
        fprintf(stderr, "flux2: device %d = %s\n", device_id,
                r->runner.getDeviceName(device_id).c_str());

    /* Create dynamic descriptor pool */
    r->runner.createDynamicDescriptorPool(4096);

    if (!createPipelines(r)) {
        fprintf(stderr, "flux2: pipeline creation failed\n");
        delete r;
        return nullptr;
    }

    if (verbose >= 1) fprintf(stderr, "flux2: init OK\n");
    return r;
}

/* ======================================================================== */
/* Weight upload                                                            */
/* ======================================================================== */

static void upload_stream_split(VulkanRunner &runner, flux2_vk_stream_split_t &gs,
                                const flux2_stream_block *sb, int H, int hd) {
    /* Split QKV [3H, H] -> Q [H,H], K [H,H], V [H,H] */
    int n = H * H;
    float *q = (float *)malloc(n * sizeof(float));
    float *k = (float *)malloc(n * sizeof(float));
    float *v = (float *)malloc(n * sizeof(float));
    for (int r = 0; r < H; r++) {
        memcpy(q + r*H, sb->qkv.w + r*H,     H * sizeof(float));  /* rows [0,H) */
    }
    for (int r = 0; r < H; r++) {
        memcpy(k + r*H, sb->qkv.w + (H+r)*H, H * sizeof(float));  /* rows [H,2H) */
    }
    for (int r = 0; r < H; r++) {
        memcpy(v + r*H, sb->qkv.w + (2*H+r)*H, H * sizeof(float));  /* rows [2H,3H) */
    }
    gs.q_w = uploadF32(runner, q, n);
    gs.k_w = uploadF32(runner, k, n);
    gs.v_w = uploadF32(runner, v, n);
    free(q); free(k); free(v);

    gs.proj_w   = uploadMat(runner, &sb->proj);
    gs.mlp_up_w = uploadMat(runner, &sb->mlp_up);
    gs.mlp_dn_w = uploadMat(runner, &sb->mlp_down);
    gs.q_norm   = uploadF32(runner, sb->q_norm, hd);
    gs.k_norm   = uploadF32(runner, sb->k_norm, hd);
}

int vulkan_flux2_load_dit(vulkan_flux2_runner *r, const char *path) {
    if (r->verbose >= 1) fprintf(stderr, "flux2: loading DiT %s\n", path);

    r->dit = flux2_dit_load_safetensors(path);
    if (!r->dit) return -1;

    flux2_dit_model *m = r->dit;
    r->H = m->hidden_dim;  r->nH = m->n_heads;  r->hd = m->head_dim;
    r->n_ff = m->n_ff;     r->pin = m->patch_in_channels;
    r->txt_dim = m->txt_dim;
    r->n_dbl = m->n_double_blocks;
    r->n_sgl = m->n_single_blocks;
    int H = r->H, n_ff = r->n_ff, hd = r->hd;

    if (r->verbose >= 1)
        fprintf(stderr, "flux2: uploading weights to GPU (H=%d, %d+%d blocks)...\n",
                H, r->n_dbl, r->n_sgl);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Global weights */
    r->d_img_in_w  = uploadMat(r->runner, &m->img_in);
    r->d_img_in_b  = m->img_in_b ? uploadF32(r->runner, m->img_in_b, H) : BufInfo{};
    r->d_txt_in_w  = uploadMat(r->runner, &m->txt_in);
    r->d_txt_in_b  = m->txt_in_b ? uploadF32(r->runner, m->txt_in_b, H) : BufInfo{};
    r->d_t_fc1_w   = uploadMat(r->runner, &m->time_in_lin1);
    r->d_t_fc1_b   = m->time_in_lin1_b ? uploadF32(r->runner, m->time_in_lin1_b, H) : BufInfo{};
    r->d_t_fc2_w   = uploadMat(r->runner, &m->time_in_lin2);
    r->d_t_fc2_b   = m->time_in_lin2_b ? uploadF32(r->runner, m->time_in_lin2_b, H) : BufInfo{};
    r->d_mod_img_w = uploadMat(r->runner, &m->mod_img);
    r->d_mod_txt_w = uploadMat(r->runner, &m->mod_txt);
    r->d_mod_sgl_w = uploadMat(r->runner, &m->mod_sgl);
    r->d_out_mod_w = uploadMat(r->runner, &m->out_mod);
    r->d_out_proj_w= uploadMat(r->runner, &m->out_proj);

    /* Double-stream blocks (split QKV) */
    auto *ext = get_ext(r);
    ext->dblk.resize(r->n_dbl);
    for (int i = 0; i < r->n_dbl; i++) {
        upload_stream_split(r->runner, ext->dblk[i].img, &m->dblk[i].img, H, hd);
        upload_stream_split(r->runner, ext->dblk[i].txt, &m->dblk[i].txt, H, hd);
        if (r->verbose >= 2) fprintf(stderr, "\r  double block %d/%d", i+1, r->n_dbl);
    }

    /* Single-stream blocks (split linear1 and linear2) */
    ext->sblk.resize(r->n_sgl);
    for (int i = 0; i < r->n_sgl; i++) {
        /* linear1 is [3H+2*n_ff, H]. Split rows: Q[H,H], K[H,H], V[H,H], gate_up[2*n_ff,H] */
        const flux2_mat *l1 = &m->sblk[i].linear1;
        int l1_cols = l1->cols;  /* H */
        int nq = H * l1_cols, nk = H * l1_cols, nv = H * l1_cols, ngu = 2*n_ff*l1_cols;
        float *q = (float *)malloc(nq * sizeof(float));
        float *k = (float *)malloc(nk * sizeof(float));
        float *v = (float *)malloc(nv * sizeof(float));
        float *gu = (float *)malloc(ngu * sizeof(float));
        memcpy(q,  l1->w,                              nq * sizeof(float));
        memcpy(k,  l1->w + (size_t)H * l1_cols,        nk * sizeof(float));
        memcpy(v,  l1->w + (size_t)2*H * l1_cols,      nv * sizeof(float));
        memcpy(gu, l1->w + (size_t)3*H * l1_cols,      ngu * sizeof(float));
        ext->sblk[i].q_w  = uploadF32(r->runner, q, nq);
        ext->sblk[i].k_w  = uploadF32(r->runner, k, nk);
        ext->sblk[i].v_w  = uploadF32(r->runner, v, nv);
        ext->sblk[i].gu_w = uploadF32(r->runner, gu, ngu);
        free(q); free(k); free(v); free(gu);

        /* linear2 is [H, H+n_ff]. Split columns: attn[H,H], mlp[H,n_ff] */
        const flux2_mat *l2 = &m->sblk[i].linear2;
        int l2_in = l2->cols;
        float *attn = (float *)malloc((size_t)H * H * sizeof(float));
        float *mlp_ = (float *)malloc((size_t)H * n_ff * sizeof(float));
        for (int r2 = 0; r2 < H; r2++) {
            memcpy(attn + (size_t)r2 * H, l2->w + (size_t)r2 * l2_in, H * sizeof(float));
            memcpy(mlp_ + (size_t)r2 * n_ff, l2->w + (size_t)r2 * l2_in + H, n_ff * sizeof(float));
        }
        ext->sblk[i].l2_attn_w = uploadF32(r->runner, attn, H * H);
        ext->sblk[i].l2_mlp_w  = uploadF32(r->runner, mlp_, H * n_ff);
        free(attn); free(mlp_);

        ext->sblk[i].q_norm = uploadF32(r->runner, m->sblk[i].q_norm, hd);
        ext->sblk[i].k_norm = uploadF32(r->runner, m->sblk[i].k_norm, hd);
        if (r->verbose >= 2) fprintf(stderr, "\r  single block %d/%d", i+1, r->n_sgl);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "\nflux2: weights uploaded in %.1f s\n", dt);

    r->gpu_loaded = true;
    return 0;
}

int vulkan_flux2_load_vae(vulkan_flux2_runner *r, const char *path) {
    if (r->verbose >= 1) fprintf(stderr, "flux2: loading VAE %s\n", path);
    r->vae = flux2_vae_load(path);
    return r->vae ? 0 : -1;
}

/* ======================================================================== */
/* Activation buffer allocation                                             */
/* ======================================================================== */

static void flux2_alloc_bufs(vulkan_flux2_runner *r, int n_img, int n_txt) {
    int n_tot = n_img + n_txt;
    int H = r->H, n_ff = r->n_ff;

    if (r->max_tok >= n_tot) return;

    /* Free old buffers */
    auto freeB = [&](BufInfo &b) { if (isValid(b)) { r->runner.destroyBuffer(b); b = BufInfo{}; } };
    freeB(r->d_img); freeB(r->d_txt); freeB(r->d_joint);
    freeB(r->d_temb); freeB(r->d_temb_silu);
    freeB(r->d_mod_img_v); freeB(r->d_mod_txt_v); freeB(r->d_mod_sgl_v);
    freeB(r->d_q); freeB(r->d_k); freeB(r->d_v); freeB(r->d_attn_out);
    freeB(r->d_scratch1); freeB(r->d_scratch2); freeB(r->d_scratch3);
    freeB(r->d_img_in_buf); freeB(r->d_txt_in_buf); freeB(r->d_traw);

    size_t F = sizeof(float);
    r->d_img        = createGpuBuffer(r->runner, (size_t)n_img * H * F);
    r->d_txt        = createGpuBuffer(r->runner, (size_t)n_txt * H * F);
    r->d_joint      = createGpuBuffer(r->runner, (size_t)n_tot * H * F);
    r->d_temb       = createGpuBuffer(r->runner, (size_t)H * F);
    r->d_temb_silu  = createGpuBuffer(r->runner, (size_t)H * F);
    r->d_mod_img_v  = createGpuBuffer(r->runner, (size_t)6 * H * F);
    r->d_mod_txt_v  = createGpuBuffer(r->runner, (size_t)6 * H * F);
    r->d_mod_sgl_v  = createGpuBuffer(r->runner, (size_t)3 * H * F);
    r->d_q          = createGpuBuffer(r->runner, (size_t)n_tot * H * F);
    r->d_k          = createGpuBuffer(r->runner, (size_t)n_tot * H * F);
    r->d_v          = createGpuBuffer(r->runner, (size_t)n_tot * H * F);
    r->d_attn_out   = createGpuBuffer(r->runner, (size_t)n_tot * H * F);
    int lin1_dim = 3 * H + 2 * n_ff;
    r->d_scratch1   = createGpuBuffer(r->runner, (size_t)n_tot * lin1_dim * F);
    r->d_scratch2   = createGpuBuffer(r->runner, (size_t)n_tot * 2 * n_ff * F);
    int s3 = n_ff > H ? n_ff : H;
    r->d_scratch3   = createGpuBuffer(r->runner, (size_t)n_tot * s3 * F);
    r->d_img_in_buf = createGpuBuffer(r->runner, (size_t)n_img * r->pin * F);
    r->d_txt_in_buf = createGpuBuffer(r->runner, (size_t)n_txt * r->txt_dim * F);
    r->d_traw       = createGpuBuffer(r->runner, 256 * F);

    r->max_tok = n_tot;
    if (r->verbose >= 1)
        fprintf(stderr, "flux2: allocated activation buffers for %d tokens\n", n_tot);
}

/* ======================================================================== */
/* GPU DiT forward pass                                                     */
/* ======================================================================== */

/* Note: Vulkan doesn't support sub-buffer offsets on SSBOs easily.
 * For modulation vectors (6*H partitioned into 6 x H slices),
 * we upload/download to split buffers. For simplicity in this initial version,
 * we compute modulations on CPU and upload the split vectors. */

int vulkan_flux2_dit_step(vulkan_flux2_runner *r,
                          const float *img_tokens, int n_img,
                          const float *txt_tokens, int n_txt,
                          float timestep, float guidance, float *out) {
    (void)guidance;
    if (!r->dit || !r->gpu_loaded) {
        fprintf(stderr, "flux2: DiT not loaded\n");
        return -1;
    }

    int H = r->H, nH = r->nH, hd = r->hd, n_ff = r->n_ff;
    int n_tot = n_img + n_txt;
    size_t F = sizeof(float);
    auto *ext = get_ext(r);
    BufInfo null_buf{};

    flux2_alloc_bufs(r, n_img, n_txt);

    /* Upload inputs */
    r->runner.uploadToDeviceLocal(r->d_img_in_buf, img_tokens, (size_t)n_img * r->pin * F);
    r->runner.uploadToDeviceLocal(r->d_txt_in_buf, txt_tokens, (size_t)n_txt * r->txt_dim * F);

    /* 1. Timestep embedding (CPU -> upload) */
    float t_raw[256];
    flux2_ts_embed(t_raw, timestep * 1000.0f, 256);
    r->runner.uploadToDeviceLocal(r->d_traw, t_raw, 256 * F);

    /* Batched GPU execution */
    beginBatch(r);

    /* temb = SiLU(time_in_lin1 @ t_raw + b1) */
    op_gemm(r, r->d_temb, r->d_t_fc1_w, r->d_traw, r->d_t_fc1_b, H, 256, 1);
    op_silu(r, r->d_temb, H);
    op_gemm(r, r->d_temb_silu, r->d_t_fc2_w, r->d_temb, r->d_t_fc2_b, H, H, 1);

    /* 2. Project img and txt tokens */
    op_gemm(r, r->d_img, r->d_img_in_w, r->d_img_in_buf, r->d_img_in_b, H, r->pin, n_img);
    op_gemm(r, r->d_txt, r->d_txt_in_w, r->d_txt_in_buf, r->d_txt_in_b, H, r->txt_dim, n_txt);

    endBatch(r);

    /* 3. Compute global modulations on CPU (avoids sub-buffer offset issues) */
    /* Download temb_silu, compute SiLU, then modulations on CPU, upload split vectors */
    float *h_temb = (float *)malloc(H * F);
    r->runner.downloadFromDeviceLocal(r->d_temb_silu, h_temb, H * F);

    /* SiLU on CPU */
    for (int i = 0; i < H; i++) h_temb[i] = h_temb[i] / (1.0f + expf(-h_temb[i]));

    /* mod_img = mod_img_w @ silu_temb, etc. */
    float *h_mod_img = (float *)calloc(6 * H, sizeof(float));
    float *h_mod_txt = (float *)calloc(6 * H, sizeof(float));
    float *h_mod_sgl = (float *)calloc(3 * H, sizeof(float));

    flux2_dit_model *m = r->dit;
    flux2_gemm(h_mod_img, h_temb, 1, H, &m->mod_img, NULL);
    flux2_gemm(h_mod_txt, h_temb, 1, H, &m->mod_txt, NULL);
    flux2_gemm(h_mod_sgl, h_temb, 1, H, &m->mod_sgl, NULL);

    /* Split modulation vectors into separate buffers for adaLN */
    /* Each mod is [6H]: shift_a(H), scale_a(H), gate_a(H), shift_f(H), scale_f(H), gate_f(H) */
    BufInfo mi_shift_a = uploadF32(r->runner, h_mod_img,       H);
    BufInfo mi_scale_a = uploadF32(r->runner, h_mod_img + H,   H);
    BufInfo mi_gate_a  = uploadF32(r->runner, h_mod_img + 2*H, H);
    BufInfo mi_shift_f = uploadF32(r->runner, h_mod_img + 3*H, H);
    BufInfo mi_scale_f = uploadF32(r->runner, h_mod_img + 4*H, H);
    BufInfo mi_gate_f  = uploadF32(r->runner, h_mod_img + 5*H, H);

    BufInfo mt_shift_a = uploadF32(r->runner, h_mod_txt,       H);
    BufInfo mt_scale_a = uploadF32(r->runner, h_mod_txt + H,   H);
    BufInfo mt_gate_a  = uploadF32(r->runner, h_mod_txt + 2*H, H);
    BufInfo mt_shift_f = uploadF32(r->runner, h_mod_txt + 3*H, H);
    BufInfo mt_scale_f = uploadF32(r->runner, h_mod_txt + 4*H, H);
    BufInfo mt_gate_f  = uploadF32(r->runner, h_mod_txt + 5*H, H);

    BufInfo ms_shift = uploadF32(r->runner, h_mod_sgl,       H);
    BufInfo ms_scale = uploadF32(r->runner, h_mod_sgl + H,   H);
    BufInfo ms_gate  = uploadF32(r->runner, h_mod_sgl + 2*H, H);

    free(h_temb); free(h_mod_img); free(h_mod_txt); free(h_mod_sgl);

    /* Compute lat_w for RoPE */
    int lat_w_p = (int)sqrtf((float)n_img);
    if (lat_w_p * lat_w_p != n_img) lat_w_p = n_img;
    float theta = FLUX2_ROPE_THETA;

    /* ---- Double-stream blocks ---- */
    for (int bi = 0; bi < r->n_dbl; bi++) {
        auto &b = ext->dblk[bi];

        beginBatch(r);

        /* IMG stream */
        op_adaln(r, r->d_scratch1, r->d_img, mi_shift_a, mi_scale_a, n_img, H);
        op_gemm(r, r->d_q, b.img.q_w, r->d_scratch1, null_buf, H, H, n_img);
        op_gemm(r, r->d_k, b.img.k_w, r->d_scratch1, null_buf, H, H, n_img);
        op_gemm(r, r->d_v, b.img.v_w, r->d_scratch1, null_buf, H, H, n_img);

        /* TXT stream */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_a, mt_scale_a, n_txt, H);

        endBatch(r);

        /* TXT Q/K/V need to be written to offset positions in d_q/d_k/d_v.
         * Since we can't offset SSBOs, we use separate buffers and copy.
         * For simplicity: use d_scratch2 as temp for txt Q/K/V, then copy. */
        /* Actually, we need to rethink this. The attention needs joint [img,txt] Q/K/V.
         * We'll compute img Q/K/V into d_q/d_k/d_v (first n_img rows),
         * then txt Q/K/V into separate temp buffers, and concat via CPU download/upload.
         * This is expensive but correct for the initial version. */

        /* For now, use CPU-fallback attention for joint sequences */
        /* TODO: Implement proper buffer concatenation kernel */

        BufInfo txt_q_tmp = createGpuBuffer(r->runner, (size_t)n_txt * H * F);
        BufInfo txt_k_tmp = createGpuBuffer(r->runner, (size_t)n_txt * H * F);
        BufInfo txt_v_tmp = createGpuBuffer(r->runner, (size_t)n_txt * H * F);

        beginBatch(r);
        op_gemm(r, txt_q_tmp, b.txt.q_w, r->d_scratch1, null_buf, H, H, n_txt);
        op_gemm(r, txt_k_tmp, b.txt.k_w, r->d_scratch1, null_buf, H, H, n_txt);
        op_gemm(r, txt_v_tmp, b.txt.v_w, r->d_scratch1, null_buf, H, H, n_txt);

        /* RMSNorm + RoPE on img Q/K */
        op_rmsnorm_ph(r, r->d_q, b.img.q_norm, n_img, nH, hd);
        op_rmsnorm_ph(r, r->d_k, b.img.k_norm, n_img, nH, hd);
        op_rmsnorm_ph(r, txt_q_tmp, b.txt.q_norm, n_txt, nH, hd);
        op_rmsnorm_ph(r, txt_k_tmp, b.txt.k_norm, n_txt, nH, hd);

        op_rope_img(r, r->d_q, n_img, nH, hd, lat_w_p, theta);
        op_rope_img(r, r->d_k, n_img, nH, hd, lat_w_p, theta);
        op_rope_txt(r, txt_q_tmp, n_txt, nH, hd, theta);
        op_rope_txt(r, txt_k_tmp, n_txt, nH, hd, theta);
        endBatch(r);

        /* Concatenate img + txt into joint Q/K/V for attention */
        /* Download img and txt parts, concat on CPU, re-upload */
        size_t img_bytes = (size_t)n_img * H * F;
        size_t txt_bytes = (size_t)n_txt * H * F;
        size_t tot_bytes = (size_t)n_tot * H * F;

        BufInfo joint_q = createGpuBuffer(r->runner, tot_bytes);
        BufInfo joint_k = createGpuBuffer(r->runner, tot_bytes);
        BufInfo joint_v = createGpuBuffer(r->runner, tot_bytes);

        /* Use staging buffers for concat */
        float *hq = (float *)malloc(tot_bytes);
        float *hk = (float *)malloc(tot_bytes);
        float *hv = (float *)malloc(tot_bytes);
        r->runner.downloadFromDeviceLocal(r->d_q, hq, img_bytes);
        r->runner.downloadFromDeviceLocal(txt_q_tmp, hq + n_img * H, txt_bytes);
        r->runner.downloadFromDeviceLocal(r->d_k, hk, img_bytes);
        r->runner.downloadFromDeviceLocal(txt_k_tmp, hk + n_img * H, txt_bytes);
        r->runner.downloadFromDeviceLocal(r->d_v, hv, img_bytes);
        r->runner.downloadFromDeviceLocal(txt_v_tmp, hv + n_img * H, txt_bytes);
        r->runner.uploadToDeviceLocal(joint_q, hq, tot_bytes);
        r->runner.uploadToDeviceLocal(joint_k, hk, tot_bytes);
        r->runner.uploadToDeviceLocal(joint_v, hv, tot_bytes);
        free(hq); free(hk); free(hv);
        r->runner.destroyBuffer(txt_q_tmp);
        r->runner.destroyBuffer(txt_k_tmp);
        r->runner.destroyBuffer(txt_v_tmp);

        /* Joint attention */
        beginBatch(r);
        op_attn(r, r->d_attn_out, joint_q, joint_k, joint_v, n_tot, nH, hd);

        /* Output projections (img portion is first n_img rows, txt is rest) */
        /* For projection, we need just img rows and txt rows separately.
         * Download attn_out, split, project separately. */
        endBatch(r);

        /* Split attn_out -> img_attn, txt_attn via CPU */
        BufInfo img_attn = createGpuBuffer(r->runner, img_bytes);
        BufInfo txt_attn = createGpuBuffer(r->runner, txt_bytes);
        float *ha = (float *)malloc(tot_bytes);
        r->runner.downloadFromDeviceLocal(r->d_attn_out, ha, tot_bytes);
        r->runner.uploadToDeviceLocal(img_attn, ha, img_bytes);
        r->runner.uploadToDeviceLocal(txt_attn, ha + n_img * H, txt_bytes);
        free(ha);
        r->runner.destroyBuffer(joint_q);
        r->runner.destroyBuffer(joint_k);
        r->runner.destroyBuffer(joint_v);

        beginBatch(r);
        op_gemm(r, r->d_scratch1, b.img.proj_w, img_attn, null_buf, H, H, n_img);
        op_gemm(r, r->d_scratch2, b.txt.proj_w, txt_attn, null_buf, H, H, n_txt);

        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_a, n_img, H);
        op_gated_add(r, r->d_txt, r->d_scratch2, mt_gate_a, n_txt, H);

        /* FFN img */
        op_adaln(r, r->d_scratch1, r->d_img, mi_shift_f, mi_scale_f, n_img, H);
        op_gemm(r, r->d_scratch2, b.img.mlp_up_w, r->d_scratch1, null_buf, 2*n_ff, H, n_img);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_img, n_ff);
        op_gemm(r, r->d_scratch1, b.img.mlp_dn_w, r->d_scratch3, null_buf, H, n_ff, n_img);
        op_gated_add(r, r->d_img, r->d_scratch1, mi_gate_f, n_img, H);

        /* FFN txt */
        op_adaln(r, r->d_scratch1, r->d_txt, mt_shift_f, mt_scale_f, n_txt, H);
        op_gemm(r, r->d_scratch2, b.txt.mlp_up_w, r->d_scratch1, null_buf, 2*n_ff, H, n_txt);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_txt, n_ff);
        op_gemm(r, r->d_scratch1, b.txt.mlp_dn_w, r->d_scratch3, null_buf, H, n_ff, n_txt);
        op_gated_add(r, r->d_txt, r->d_scratch1, mt_gate_f, n_txt, H);
        endBatch(r);

        r->runner.destroyBuffer(img_attn);
        r->runner.destroyBuffer(txt_attn);

        if (r->verbose >= 2)
            fprintf(stderr, "\r  dit double block %d/%d", bi+1, r->n_dbl);
    }

    /* ---- Single-stream blocks ---- */
    /* Concatenate: joint = [txt, img] */
    {
        size_t txt_bytes = (size_t)n_txt * H * F;
        size_t img_bytes = (size_t)n_img * H * F;
        float *hj = (float *)malloc((size_t)n_tot * H * F);
        r->runner.downloadFromDeviceLocal(r->d_txt, hj, txt_bytes);
        r->runner.downloadFromDeviceLocal(r->d_img, hj + n_txt * H, img_bytes);
        r->runner.uploadToDeviceLocal(r->d_joint, hj, (size_t)n_tot * H * F);
        free(hj);
    }

    for (int bi = 0; bi < r->n_sgl; bi++) {
        auto &b = ext->sblk[bi];

        beginBatch(r);
        op_adaln(r, r->d_scratch1, r->d_joint, ms_shift, ms_scale, n_tot, H);

        op_gemm(r, r->d_q, b.q_w, r->d_scratch1, null_buf, H, H, n_tot);
        op_gemm(r, r->d_k, b.k_w, r->d_scratch1, null_buf, H, H, n_tot);
        op_gemm(r, r->d_v, b.v_w, r->d_scratch1, null_buf, H, H, n_tot);
        op_gemm(r, r->d_scratch2, b.gu_w, r->d_scratch1, null_buf, 2*n_ff, H, n_tot);

        op_rmsnorm_ph(r, r->d_q, b.q_norm, n_tot, nH, hd);
        op_rmsnorm_ph(r, r->d_k, b.k_norm, n_tot, nH, hd);
        endBatch(r);

        /* RoPE: txt tokens at front [0..n_txt), img after.
         * Since we can't offset, download, split, RoPE separately, re-upload.
         * For the single-stream case with joint Q/K, this is simpler:
         * the full buffer has txt first then img. We need per-portion RoPE. */
        /* TODO: Add offset support or a concat_rope shader.
         * For now: download Q/K, apply CPU RoPE, re-upload. */
        {
            size_t tot_bytes = (size_t)n_tot * H * F;
            float *hq = (float *)malloc(tot_bytes);
            float *hk = (float *)malloc(tot_bytes);
            r->runner.downloadFromDeviceLocal(r->d_q, hq, tot_bytes);
            r->runner.downloadFromDeviceLocal(r->d_k, hk, tot_bytes);

            /* CPU RoPE for txt portion [0, n_txt) */
            flux2_rope_txt(hq, n_txt, nH, hd, theta);
            flux2_rope_txt(hk, n_txt, nH, hd, theta);
            /* CPU RoPE for img portion [n_txt, n_tot) */
            flux2_rope_img(hq + (size_t)n_txt * H, n_img, nH, hd,
                          lat_w_p, lat_w_p, theta);
            flux2_rope_img(hk + (size_t)n_txt * H, n_img, nH, hd,
                          lat_w_p, lat_w_p, theta);

            r->runner.uploadToDeviceLocal(r->d_q, hq, tot_bytes);
            r->runner.uploadToDeviceLocal(r->d_k, hk, tot_bytes);
            free(hq); free(hk);
        }

        beginBatch(r);
        op_attn(r, r->d_attn_out, r->d_q, r->d_k, r->d_v, n_tot, nH, hd);
        op_swiglu(r, r->d_scratch3, r->d_scratch2, n_tot, n_ff);
        op_gemm(r, r->d_scratch1, b.l2_attn_w, r->d_attn_out, null_buf, H, H, n_tot);
        op_gemm(r, r->d_scratch2, b.l2_mlp_w, r->d_scratch3, null_buf, H, n_ff, n_tot);
        op_add(r, r->d_scratch1, r->d_scratch2, n_tot * H);
        op_gated_add(r, r->d_joint, r->d_scratch1, ms_gate, n_tot, H);
        endBatch(r);

        if (r->verbose >= 2)
            fprintf(stderr, "\r  dit single block %d/%d", bi+1, r->n_sgl);
    }

    /* ---- Output ---- */
    /* Extract img portion from joint = joint[n_txt:] */
    {
        size_t tot_bytes = (size_t)n_tot * H * F;
        size_t img_bytes = (size_t)n_img * H * F;
        float *hj = (float *)malloc(tot_bytes);
        r->runner.downloadFromDeviceLocal(r->d_joint, hj, tot_bytes);
        /* img portion starts at offset n_txt*H */
        BufInfo d_img_out = createGpuBuffer(r->runner, img_bytes);
        r->runner.uploadToDeviceLocal(d_img_out, hj + n_txt * H, img_bytes);
        free(hj);

        /* Final adaLN: compute out_mod on CPU */
        float *h_out_mod = (float *)calloc(2 * H, sizeof(float));
        float *h_silu_temb = (float *)malloc(H * F);
        r->runner.downloadFromDeviceLocal(r->d_temb_silu, h_silu_temb, H * F);
        for (int i = 0; i < H; i++) h_silu_temb[i] = h_silu_temb[i] / (1.0f + expf(-h_silu_temb[i]));
        flux2_gemm(h_out_mod, h_silu_temb, 1, H, &m->out_mod, NULL);
        free(h_silu_temb);

        BufInfo out_shift = uploadF32(r->runner, h_out_mod, H);
        BufInfo out_scale = uploadF32(r->runner, h_out_mod + H, H);
        free(h_out_mod);

        beginBatch(r);
        op_adaln(r, r->d_scratch2, d_img_out, out_shift, out_scale, n_img, H);
        op_gemm(r, r->d_attn_out, r->d_out_proj_w, r->d_scratch2, null_buf, r->pin, H, n_img);
        endBatch(r);

        /* Download result */
        r->runner.downloadFromDeviceLocal(r->d_attn_out, out, (size_t)n_img * r->pin * F);

        r->runner.destroyBuffer(d_img_out);
        r->runner.destroyBuffer(out_shift);
        r->runner.destroyBuffer(out_scale);
    }

    /* Cleanup modulation buffers */
    auto freeB = [&](BufInfo &b) { if (isValid(b)) { r->runner.destroyBuffer(b); b = BufInfo{}; } };
    freeB(mi_shift_a); freeB(mi_scale_a); freeB(mi_gate_a);
    freeB(mi_shift_f); freeB(mi_scale_f); freeB(mi_gate_f);
    freeB(mt_shift_a); freeB(mt_scale_a); freeB(mt_gate_a);
    freeB(mt_shift_f); freeB(mt_scale_f); freeB(mt_gate_f);
    freeB(ms_shift); freeB(ms_scale); freeB(ms_gate);

    if (r->verbose >= 2)
        fprintf(stderr, "\nflux2: dit step done\n");

    return 0;
}

/* ======================================================================== */
/* Free                                                                     */
/* ======================================================================== */

void vulkan_flux2_free(vulkan_flux2_runner *r) {
    if (!r) return;

    auto freeB = [&](BufInfo &b) { if (isValid(b)) r->runner.destroyBuffer(b); };

    /* Free ext block weights */
    auto *ext = get_ext(r);
    for (auto &d : ext->dblk) {
        freeB(d.img.q_w); freeB(d.img.k_w); freeB(d.img.v_w);
        freeB(d.img.proj_w); freeB(d.img.mlp_up_w); freeB(d.img.mlp_dn_w);
        freeB(d.img.q_norm); freeB(d.img.k_norm);
        freeB(d.txt.q_w); freeB(d.txt.k_w); freeB(d.txt.v_w);
        freeB(d.txt.proj_w); freeB(d.txt.mlp_up_w); freeB(d.txt.mlp_dn_w);
        freeB(d.txt.q_norm); freeB(d.txt.k_norm);
    }
    for (auto &s : ext->sblk) {
        freeB(s.q_w); freeB(s.k_w); freeB(s.v_w); freeB(s.gu_w);
        freeB(s.l2_attn_w); freeB(s.l2_mlp_w);
        freeB(s.q_norm); freeB(s.k_norm);
    }

    /* Free global weights */
    freeB(r->d_img_in_w); freeB(r->d_img_in_b);
    freeB(r->d_txt_in_w); freeB(r->d_txt_in_b);
    freeB(r->d_t_fc1_w); freeB(r->d_t_fc1_b);
    freeB(r->d_t_fc2_w); freeB(r->d_t_fc2_b);
    freeB(r->d_mod_img_w); freeB(r->d_mod_txt_w); freeB(r->d_mod_sgl_w);
    freeB(r->d_out_mod_w); freeB(r->d_out_proj_w);

    /* Free activation buffers */
    freeB(r->d_img); freeB(r->d_txt); freeB(r->d_joint);
    freeB(r->d_temb); freeB(r->d_temb_silu);
    freeB(r->d_mod_img_v); freeB(r->d_mod_txt_v); freeB(r->d_mod_sgl_v);
    freeB(r->d_q); freeB(r->d_k); freeB(r->d_v); freeB(r->d_attn_out);
    freeB(r->d_scratch1); freeB(r->d_scratch2); freeB(r->d_scratch3);
    freeB(r->d_img_in_buf); freeB(r->d_txt_in_buf); freeB(r->d_traw);

    if (r->dit) flux2_dit_free(r->dit);
    if (r->vae) flux2_vae_free(r->vae);
    r->runner.destroyDynamicDescriptorPool();
    r->runner.cleanup();

    /* Remove from ext map */
    g_ext_map.erase(r);

    delete r;
}

/* ---- Vulkan VAE op wrappers ---- */

/* Conv3x3 via tiled shared-memory kernel */
static void vk_vae_conv3(vulkan_flux2_runner *r, BufInfo &out, BufInfo &in,
                          BufInfo &w, BufInfo &bias,
                          int ci, int H, int W, int co) {
    struct { uint32_t ci, H, W, co, has_bias; } pc;
    pc.ci = ci; pc.H = H; pc.W = W; pc.co = co;
    pc.has_bias = isValid(bias) ? 1 : 0;
    uint32_t tiles_w = (W + 15) / 16;
    uint32_t tiles_h = (H + 15) / 16;
    uint32_t n_tiles = tiles_w * tiles_h;
    dispatchBatched(r, r->pipe_vae_im2col, {in, out, w, bias}, &pc, sizeof(pc),
                    n_tiles, (uint32_t)co);
}

static void vk_vae_conv1(vulkan_flux2_runner *r, BufInfo &out, BufInfo &in,
                          BufInfo &w, BufInfo &bias,
                          int ci, int spatial, int co) {
    struct { uint32_t ci, spatial, co, has_bias; } pc;
    pc.ci = ci; pc.spatial = spatial; pc.co = co;
    pc.has_bias = isValid(bias) ? 1 : 0;
    int total = co * spatial;
    dispatchBatched(r, r->pipe_vae_conv1, {in, out, w, bias}, &pc, sizeof(pc),
                    (uint32_t)((total + 255) / 256));
}

static void vk_vae_gnsilu(vulkan_flux2_runner *r, BufInfo &out, BufInfo &in,
                           BufInfo &gamma, BufInfo &beta,
                           int C, int spatial, int ng, int do_silu) {
    struct { uint32_t C, spatial, num_groups, apply_silu; } pc;
    pc.C = C; pc.spatial = spatial; pc.num_groups = ng; pc.apply_silu = do_silu;
    dispatchBatched(r, r->pipe_vae_gnsilu, {in, out, gamma, beta}, &pc, sizeof(pc),
                    (uint32_t)ng);
}

static void vk_vae_up2x(vulkan_flux2_runner *r, BufInfo &out, BufInfo &in,
                         int C, int H, int W) {
    struct { uint32_t C, H, W; } pc;
    pc.C = C; pc.H = H; pc.W = W;
    int total = C * H * W;
    dispatchBatched(r, r->pipe_vae_up2x, {in, out}, &pc, sizeof(pc),
                    (uint32_t)((total + 255) / 256));
}

/* Self-contained resblock: manages its own batch boundaries */
static void vk_vae_resblock(vulkan_flux2_runner *r,
                             BufInfo &out, BufInfo &x,
                             BufInfo &n1w, BufInfo &n1b, BufInfo &c1w, BufInfo &c1b,
                             BufInfo &n2w, BufInfo &n2b, BufInfo &c2w, BufInfo &c2b,
                             BufInfo &skipw, BufInfo &skipb,
                             int ci, int co, int H, int W, int ng) {
    int spatial = H * W;
    size_t F = sizeof(float);

    /* Batch 1: GN+SiLU -> Conv -> GN+SiLU -> Conv */
    BufInfo tmp1 = createGpuBuffer(r->runner, (size_t)ci * spatial * F);
    BufInfo tmp2 = createGpuBuffer(r->runner, (size_t)co * spatial * F);
    BufInfo tmp3 = createGpuBuffer(r->runner, (size_t)co * spatial * F);
    BufInfo tmp4 = createGpuBuffer(r->runner, (size_t)co * spatial * F);

    beginBatch(r);
    vk_vae_gnsilu(r, tmp1, x, n1w, n1b, ci, spatial, ng, 1);
    vk_vae_conv3(r, tmp2, tmp1, c1w, c1b, ci, H, W, co);
    vk_vae_gnsilu(r, tmp3, tmp2, n2w, n2b, co, spatial, ng, 1);
    vk_vae_conv3(r, tmp4, tmp3, c2w, c2b, co, H, W, co);
    /* Skip connection */
    if (isValid(skipw)) {
        vk_vae_conv1(r, out, x, skipw, skipb, ci, spatial, co);
    }
    endBatch(r);

    r->runner.destroyBuffer(tmp1);
    r->runner.destroyBuffer(tmp2);
    r->runner.destroyBuffer(tmp3);

    /* Identity skip: copy via CPU */
    if (!isValid(skipw)) {
        size_t bytes = (size_t)co * spatial * F;
        float *tmp_h = (float *)malloc(bytes);
        r->runner.downloadFromDeviceLocal(x, tmp_h, bytes);
        r->runner.uploadToDeviceLocal(out, tmp_h, bytes);
        free(tmp_h);
    }

    /* Batch 2: out += residual */
    beginBatch(r);
    op_add(r, out, tmp4, co * spatial);
    endBatch(r);
    r->runner.destroyBuffer(tmp4);
}

/* ---- GPU VAE decode ---- */
int vulkan_flux2_vae_decode(vulkan_flux2_runner *r,
                            const float *latent, int lat_h, int lat_w,
                            float *out_rgb) {
    if (!r->vae) {
        fprintf(stderr, "flux2: VAE not loaded\n");
        return -1;
    }
    flux2_vae_model *m = r->vae;
    int lc = m->latent_channels;
    int ng = m->num_groups;
    int h = lat_h, w = lat_w;
    size_t F = sizeof(float);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Stage 0: BN denorm on CPU (small, one-time) */
    float *latent_bn = (float *)malloc((size_t)lc * h * w * F);
    if (m->bn_mean && m->bn_var) {
        int ps = 2;
        int lat_h_p = lat_h / ps, lat_w_p = lat_w / ps;
        for (int ch = 0; ch < lc; ch++)
            for (int pr = 0; pr < ps; pr++)
                for (int pc_ = 0; pc_ < ps; pc_++)  {
                    int bn_ch = ch * (ps * ps) + pr * ps + pc_;
                    float mn = m->bn_mean[bn_ch];
                    float sd = sqrtf(m->bn_var[bn_ch] + m->bn_eps);
                    for (int hp = 0; hp < lat_h_p; hp++)
                        for (int wp = 0; wp < lat_w_p; wp++) {
                            int sh = hp * ps + pr, sw = wp * ps + pc_;
                            size_t idx = (size_t)ch * lat_h * lat_w + (size_t)sh * lat_w + sw;
                            latent_bn[idx] = latent[idx] * sd + mn;
                        }
                }
    } else {
        memcpy(latent_bn, latent, (size_t)lc * h * w * F);
    }

    BufInfo d_x = uploadF32(r->runner, latent_bn, lc * h * w);
    free(latent_bn);

    /* Stage 1: post_quant_conv (1x1) */
    if (m->pqc_w) {
        BufInfo pw = uploadF32(r->runner, m->pqc_w, lc * lc);
        BufInfo pb = uploadF32(r->runner, m->pqc_b, lc);
        BufInfo d_tmp = createGpuBuffer(r->runner, (size_t)lc * h * w * F);
        beginBatch(r);
        vk_vae_conv1(r, d_tmp, d_x, pw, pb, lc, h * w, lc);
        endBatch(r);
        r->runner.destroyBuffer(d_x); r->runner.destroyBuffer(pw); r->runner.destroyBuffer(pb);
        d_x = d_tmp;
    }

    /* Stage 2: conv_in (3x3) */
    int c = m->conv_in_out_ch;
    {
        BufInfo cw = uploadF32(r->runner, m->conv_in_w, c * lc * 9);
        BufInfo cb = uploadF32(r->runner, m->conv_in_b, c);
        BufInfo d_tmp = createGpuBuffer(r->runner, (size_t)c * h * w * F);
        beginBatch(r);
        vk_vae_conv3(r, d_tmp, d_x, cw, cb, lc, h, w, c);
        endBatch(r);
        r->runner.destroyBuffer(d_x); r->runner.destroyBuffer(cw); r->runner.destroyBuffer(cb);
        d_x = d_tmp;
    }

    if (r->verbose >= 1) fprintf(stderr, "flux2: VAE conv_in done [%d,%d,%d]\n", c, h, w);

    /* Helper to upload resblock and run */
    auto runResblock = [&](const flux2_vae_resblock *rb) {
        int ci = rb->c_in, co = rb->c_out;
        BufInfo n1w = uploadF32(r->runner, rb->norm1_w, ci);
        BufInfo n1b = uploadF32(r->runner, rb->norm1_b, ci);
        BufInfo c1w = uploadF32(r->runner, rb->conv1_w, co * ci * 9);
        BufInfo c1b = uploadF32(r->runner, rb->conv1_b, co);
        BufInfo n2w = uploadF32(r->runner, rb->norm2_w, co);
        BufInfo n2b = uploadF32(r->runner, rb->norm2_b, co);
        BufInfo c2w = uploadF32(r->runner, rb->conv2_w, co * co * 9);
        BufInfo c2b = uploadF32(r->runner, rb->conv2_b, co);
        BufInfo sw = rb->skip_w ? uploadF32(r->runner, rb->skip_w, co * ci) : BufInfo{};
        BufInfo sb = rb->skip_b ? uploadF32(r->runner, rb->skip_b, co) : BufInfo{};
        BufInfo d_tmp = createGpuBuffer(r->runner, (size_t)co * h * w * F);
        vk_vae_resblock(r, d_tmp, d_x, n1w, n1b, c1w, c1b, n2w, n2b, c2w, c2b, sw, sb,
                         ci, co, h, w, ng);
        r->runner.destroyBuffer(d_x); d_x = d_tmp; c = co;
        r->runner.destroyBuffer(n1w); r->runner.destroyBuffer(n1b);
        r->runner.destroyBuffer(c1w); r->runner.destroyBuffer(c1b);
        r->runner.destroyBuffer(n2w); r->runner.destroyBuffer(n2b);
        r->runner.destroyBuffer(c2w); r->runner.destroyBuffer(c2b);
        if (isValid(sw)) r->runner.destroyBuffer(sw);
        if (isValid(sb)) r->runner.destroyBuffer(sb);
    };

    /* Stage 3: mid_block */
    runResblock(&m->mid_res0);
    /* Mid attention: CPU fallback (only 1 call, small spatial at lowest res) */
    {
        size_t sz = (size_t)c * h * w * F;
        float *hx = (float *)malloc(sz);
        r->runner.downloadFromDeviceLocal(d_x, hx, sz);
        float *ho = (float *)malloc(sz);
        flux2_vae_mid_attn_forward(ho, hx, &m->mid_attn, h, w, ng);
        r->runner.destroyBuffer(d_x);
        d_x = uploadF32(r->runner, ho, c * h * w);
        free(hx); free(ho);
    }
    runResblock(&m->mid_res1);

    if (r->verbose >= 1) fprintf(stderr, "flux2: VAE mid_block done\n");

    /* Stage 4: up_blocks */
    for (int bi = 0; bi < 4; bi++) {
        runResblock(&m->up_res[bi][0]);
        runResblock(&m->up_res[bi][1]);
        if (m->up_has_sample[bi]) {
            BufInfo d_up = createGpuBuffer(r->runner, (size_t)c * 2*h * 2*w * F);
            beginBatch(r);
            vk_vae_up2x(r, d_up, d_x, c, h, w);
            endBatch(r);
            h *= 2; w *= 2;
            BufInfo usw = uploadF32(r->runner, m->up_sample[bi].conv_w, c * c * 9);
            BufInfo usb = uploadF32(r->runner, m->up_sample[bi].conv_b, c);
            BufInfo d_tmp = createGpuBuffer(r->runner, (size_t)c * h * w * F);
            beginBatch(r);
            vk_vae_conv3(r, d_tmp, d_up, usw, usb, c, h, w, c);
            endBatch(r);
            r->runner.destroyBuffer(d_up); r->runner.destroyBuffer(d_x);
            r->runner.destroyBuffer(usw); r->runner.destroyBuffer(usb);
            d_x = d_tmp;
        }
        if (r->verbose >= 1)
            fprintf(stderr, "flux2: VAE up_block[%d] done [%d,%d,%d]\n", bi, c, h, w);
    }

    /* Stage 5: norm_out -> SiLU -> conv_out */
    {
        BufInfo now = uploadF32(r->runner, m->norm_out_w, c);
        BufInfo nob = uploadF32(r->runner, m->norm_out_b, c);
        BufInfo d_normed = createGpuBuffer(r->runner, (size_t)c * h * w * F);
        beginBatch(r);
        vk_vae_gnsilu(r, d_normed, d_x, now, nob, c, h*w, ng, 1);
        endBatch(r);
        r->runner.destroyBuffer(d_x); r->runner.destroyBuffer(now); r->runner.destroyBuffer(nob);

        BufInfo cow = uploadF32(r->runner, m->conv_out_w, 3 * c * 9);
        BufInfo cob = uploadF32(r->runner, m->conv_out_b, 3);
        BufInfo d_rgb = createGpuBuffer(r->runner, (size_t)3 * h * w * F);
        beginBatch(r);
        vk_vae_conv3(r, d_rgb, d_normed, cow, cob, c, h, w, 3);
        endBatch(r);
        r->runner.destroyBuffer(d_normed); r->runner.destroyBuffer(cow); r->runner.destroyBuffer(cob);

        r->runner.downloadFromDeviceLocal(d_rgb, out_rgb, (size_t)3 * h * w * F);
        r->runner.destroyBuffer(d_rgb);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose >= 1)
        fprintf(stderr, "flux2: Vulkan GPU VAE decode done (%.1f s)\n", dt);

    return 0;
}
