// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Vulkan Qwen-Image Text-to-Image Runner
//
// Pipeline: Input projection -> DiT (60 dual-stream blocks) -> Output projection
// Uses pre-compiled SPIR-V shaders and DEVICE_LOCAL SSBOs with batched dispatch.
// F32 weights on GPU, F32 compute.
//
// Port of hip_qimg_runner.c for cross-platform Vulkan compute.
//

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"

#include "vulkan_qimg_runner.h"
#include "../deps/vulkan-runner.hh"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
/* FP8 E4M3 -> F32 CPU conversion                                          */
/* ======================================================================== */

static float fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) {
        f = ldexpf((float)mant / 8.0f, -6);
    } else if (exp == 15 && mant == 7) {
        return 0.0f;  /* NaN -> 0 */
    } else {
        f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -f : f;
}

static float vk_fp8_to_f32_lut[256];
static int vk_fp8_to_f32_lut_init = 0;

static void init_fp8_to_f32_lut(void) {
    if (vk_fp8_to_f32_lut_init) return;
    for (int i = 0; i < 256; i++)
        vk_fp8_to_f32_lut[i] = fp8_e4m3_to_f32((uint8_t)i);
    vk_fp8_to_f32_lut_init = 1;
}

/* ======================================================================== */
/* Per-block GPU weight struct                                              */
/* ======================================================================== */

struct qimg_vk_block_t {
    BufInfo attn_q_w, attn_q_b, attn_k_w, attn_k_b, attn_v_w, attn_v_b;
    BufInfo attn_out_w, attn_out_b;
    BufInfo attn_add_q_w, attn_add_q_b, attn_add_k_w, attn_add_k_b;
    BufInfo attn_add_v_w, attn_add_v_b, attn_add_out_w, attn_add_out_b;
    BufInfo norm_q_w, norm_k_w, norm_added_q_w, norm_added_k_w;
    BufInfo img_mod_w, img_mod_b;
    BufInfo img_mlp_fc1_w, img_mlp_fc1_b, img_mlp_fc2_w, img_mlp_fc2_b;
    BufInfo txt_mod_w, txt_mod_b;
    BufInfo txt_mlp_fc1_w, txt_mlp_fc1_b, txt_mlp_fc2_w, txt_mlp_fc2_b;
};

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

struct vulkan_qimg_runner {
    VulkanRunner runner;
    int verbose;
    std::string shader_dir;

    /* Pipelines */
    Pipeline pipe_gemm;
    Pipeline pipe_adaln;
    Pipeline pipe_rmsnorm_ph;
    Pipeline pipe_rope_2d;
    Pipeline pipe_rope_1d;
    Pipeline pipe_gated_add;
    Pipeline pipe_bf16_trunc;
    Pipeline pipe_attn;
    Pipeline pipe_silu;
    Pipeline pipe_gelu;
    Pipeline pipe_add;
    Pipeline pipe_layernorm;
    Pipeline pipe_rmsnorm_weighted;
    Pipeline pipe_patchify;
    Pipeline pipe_unpatchify;
    Pipeline pipe_euler_step;
    Pipeline pipe_cfg_combine;

    /* VAE pipelines */
    Pipeline pipe_vae_conv2d;
    Pipeline pipe_vae_rmsnorm;
    Pipeline pipe_vae_silu;
    Pipeline pipe_vae_up2x;

    /* DiT config */
    int dim, n_heads, head_dim, n_blocks;
    int in_ch, txt_dim, mlp_h;

    /* Safetensors context (mmap'd) */
    void *dit_st;
    void *vae_st;

    /* Global GPU weights */
    BufInfo d_img_in_w, d_img_in_b;
    BufInfo d_txt_in_w, d_txt_in_b;
    BufInfo d_txt_norm_w;
    BufInfo d_t_fc1_w, d_t_fc1_b;
    BufInfo d_t_fc2_w, d_t_fc2_b;
    BufInfo d_norm_out_w, d_norm_out_b;
    BufInfo d_proj_out_w, d_proj_out_b;
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

/* ======================================================================== */
/* Dispatch helpers                                                         */
/* ======================================================================== */

static void dispatchBatched(vulkan_qimg_runner *r, Pipeline &pipe,
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

static void beginBatch(vulkan_qimg_runner *r) { r->runner.beginRecording(); }
static void endBatch(vulkan_qimg_runner *r) {
    r->runner.endRecordingAndSubmit();
    r->runner.waitForCompletion();
    r->runner.resetDynamicDescriptorPool();
}

/* ======================================================================== */
/* FP8/BF16/F16/F32 -> F32 CPU dequant + upload                            */
/* ======================================================================== */

static BufInfo qimg_st_upload_f32(VulkanRunner &runner, st_context *st,
                                  const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return BufInfo{};
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];
    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    float *f32 = (float *)malloc(n * sizeof(float));
    if (!f32) return BufInfo{};

    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        init_fp8_to_f32_lut();
        for (size_t i = 0; i < n; i++)
            f32[i] = vk_fp8_to_f32_lut[src[i]];
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, src, n * 4);
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else if (strcmp(dtype, "F16") == 0) {
        const uint16_t *f16 = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = f16[i];
            uint32_t s = (bits >> 15) & 1;
            uint32_t e = (bits >> 10) & 0x1F;
            uint32_t m = bits & 0x3FF;
            uint32_t f;
            if (e == 0) f = s << 31;
            else if (e == 31) f = (s << 31) | (0xFF << 23) | (m << 13);
            else f = (s << 31) | ((e + 112) << 23) | (m << 13);
            memcpy(&f32[i], &f, 4);
        }
    } else {
        fprintf(stderr, "vulkan_qimg: unsupported dtype '%s' for %s\n", dtype, name);
        free(f32);
        return BufInfo{};
    }

    BufInfo buf = uploadF32(runner, f32, (int)n);
    free(f32);
    return buf;
}

/* Upload 3D conv weight -> 2D by taking last temporal slice */
static BufInfo qimg_upload_conv3d(VulkanRunner &runner, st_context *st,
                                  const char *name,
                                  int *out_co, int *out_ci) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return BufInfo{};
    const uint64_t *shape = safetensors_shape(st, idx);
    int co = (int)shape[0], ci = (int)shape[1], kd = (int)shape[2];
    int kh = (int)shape[3], kw = (int)shape[4];
    if (out_co) *out_co = co;
    if (out_ci) *out_ci = ci;
    size_t n2d = (size_t)co * ci * kh * kw;
    const uint16_t *bf = (const uint16_t *)safetensors_data(st, idx);
    int d_last = kd - 1;
    float *w2d = (float *)malloc(n2d * sizeof(float));
    for (int o = 0; o < co; o++)
        for (int i = 0; i < ci; i++)
            for (int h = 0; h < kh; h++)
                for (int w = 0; w < kw; w++) {
                    size_t idx3 = ((((size_t)o*ci+i)*kd+d_last)*kh+h)*kw+w;
                    uint32_t bits = (uint32_t)bf[idx3] << 16;
                    float f; memcpy(&f, &bits, 4);
                    w2d[(((size_t)o*ci+i)*kh+h)*kw+w] = f;
                }
    BufInfo buf = uploadF32(runner, w2d, (int)n2d);
    free(w2d);
    return buf;
}

/* ======================================================================== */
/* Op wrappers                                                              */
/* ======================================================================== */

static void op_gemm(vulkan_qimg_runner *r, BufInfo &Y, BufInfo &W,
                    BufInfo &X, BufInfo &bias, int n_out, int n_in, int n_tok) {
    struct { uint32_t n_out, n_in, n_tok, has_bias; } pc;
    pc.n_out = n_out; pc.n_in = n_in; pc.n_tok = n_tok;
    pc.has_bias = isValid(bias) ? 1 : 0;
    uint32_t gx = (n_out + 63) / 64;
    uint32_t gy = (n_tok + 15) / 16;
    dispatchBatched(r, r->pipe_gemm, {W, X, Y, bias}, &pc, sizeof(pc), gx, gy);
}

static void op_bf16_trunc(vulkan_qimg_runner *r, BufInfo &x, int n) {
    struct { uint32_t n; } pc;
    pc.n = n;
    dispatchBatched(r, r->pipe_bf16_trunc, {x}, &pc, sizeof(pc),
                    (uint32_t)((n + 255) / 256));
}

static void op_gemm_bf16(vulkan_qimg_runner *r, BufInfo &Y, BufInfo &W,
                         BufInfo &X, BufInfo &bias,
                         int n_out, int n_in, int n_tok) {
    op_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
    op_bf16_trunc(r, Y, n_out * n_tok);
}

static void op_silu(vulkan_qimg_runner *r, BufInfo &x, int n) {
    struct { uint32_t n; } pc;
    pc.n = n;
    dispatchBatched(r, r->pipe_silu, {x}, &pc, sizeof(pc),
                    (uint32_t)((n + 255) / 256));
}

static void op_gelu(vulkan_qimg_runner *r, BufInfo &x, int n) {
    struct { uint32_t n; } pc;
    pc.n = n;
    dispatchBatched(r, r->pipe_gelu, {x}, &pc, sizeof(pc),
                    (uint32_t)((n + 255) / 256));
}

static void op_add(vulkan_qimg_runner *r, BufInfo &dst, BufInfo &src, int n) {
    struct { uint32_t n; } pc;
    pc.n = n;
    dispatchBatched(r, r->pipe_add, {dst, src}, &pc, sizeof(pc),
                    (uint32_t)((n + 255) / 256));
}

static void op_adaln(vulkan_qimg_runner *r, BufInfo &out, BufInfo &x,
                     BufInfo &shift, BufInfo &scale, int N, int dim) {
    struct { uint32_t N, dim; } pc;
    pc.N = N; pc.dim = dim;
    dispatchBatched(r, r->pipe_adaln, {x, out, shift, scale}, &pc, sizeof(pc),
                    (uint32_t)N);
}

static void op_rmsnorm_ph(vulkan_qimg_runner *r, BufInfo &x, BufInfo &w,
                          int N, int n_heads, int head_dim) {
    struct { uint32_t N, n_heads, head_dim; } pc;
    pc.N = N; pc.n_heads = n_heads; pc.head_dim = head_dim;
    dispatchBatched(r, r->pipe_rmsnorm_ph, {x, w}, &pc, sizeof(pc),
                    (uint32_t)N, (uint32_t)n_heads);
}

static void op_rmsnorm_weighted(vulkan_qimg_runner *r, BufInfo &x, BufInfo &w,
                                int N, int dim) {
    struct { uint32_t N, dim; } pc;
    pc.N = N; pc.dim = dim;
    dispatchBatched(r, r->pipe_rmsnorm_weighted, {x, w}, &pc, sizeof(pc),
                    (uint32_t)N);
}

static void op_gated_add(vulkan_qimg_runner *r, BufInfo &x, BufInfo &proj,
                         BufInfo &gate, int N, int dim) {
    struct { uint32_t N, dim; } pc;
    pc.N = N; pc.dim = dim;
    int total = N * dim;
    dispatchBatched(r, r->pipe_gated_add, {x, proj, gate}, &pc, sizeof(pc),
                    (uint32_t)((total + 255) / 256));
}

static void op_attn(vulkan_qimg_runner *r, BufInfo &out, BufInfo &q,
                    BufInfo &k, BufInfo &v, int n_tok, int n_heads, int head_dim) {
    struct { uint32_t N, n_heads, head_dim; float scale; } pc;
    pc.N = n_tok; pc.n_heads = n_heads; pc.head_dim = head_dim;
    pc.scale = 1.0f / sqrtf((float)head_dim);
    uint32_t gy = (n_tok + 3) / 4;
    dispatchBatched(r, r->pipe_attn, {q, k, v, out}, &pc, sizeof(pc),
                    (uint32_t)n_heads, gy);
}

static void op_rope_2d(vulkan_qimg_runner *r, BufInfo &q, BufInfo &k,
                       int n_tok, int nh, int hd,
                       int hp, int wp, int t_dim, int h_dim, int w_dim,
                       float theta) {
    struct { uint32_t n_tok, nh, hd, hp, wp, td, hdim, wdim; float theta; } pc;
    pc.n_tok = n_tok; pc.nh = nh; pc.hd = hd;
    pc.hp = hp; pc.wp = wp;
    pc.td = t_dim; pc.hdim = h_dim; pc.wdim = w_dim;
    pc.theta = theta;
    dispatchBatched(r, r->pipe_rope_2d, {q, k}, &pc, sizeof(pc),
                    (uint32_t)n_tok, (uint32_t)nh);
}

static void op_rope_1d(vulkan_qimg_runner *r, BufInfo &q, BufInfo &k,
                       int n_tok, int nh, int hd, int txt_start,
                       int t_dim, int h_dim, int w_dim, float theta) {
    struct { uint32_t n_tok, nh, hd, txt_start, td, hdim, wdim; float theta; } pc;
    pc.n_tok = n_tok; pc.nh = nh; pc.hd = hd;
    pc.txt_start = txt_start;
    pc.td = t_dim; pc.hdim = h_dim; pc.wdim = w_dim;
    pc.theta = theta;
    dispatchBatched(r, r->pipe_rope_1d, {q, k}, &pc, sizeof(pc),
                    (uint32_t)n_tok, (uint32_t)nh);
}

static void op_patchify(vulkan_qimg_runner *r, BufInfo &out, BufInfo &in,
                        int n_img, int in_ch) {
    struct { uint32_t n_img, in_ch; } pc;
    pc.n_img = n_img; pc.in_ch = in_ch;
    dispatchBatched(r, r->pipe_patchify, {in, out}, &pc, sizeof(pc),
                    (uint32_t)((n_img * in_ch + 255) / 256));
}

static void op_unpatchify(vulkan_qimg_runner *r, BufInfo &out, BufInfo &in,
                          int n_img, int in_ch) {
    struct { uint32_t n_img, in_ch; } pc;
    pc.n_img = n_img; pc.in_ch = in_ch;
    dispatchBatched(r, r->pipe_unpatchify, {in, out}, &pc, sizeof(pc),
                    (uint32_t)((n_img * in_ch + 255) / 256));
}

static void op_euler_step(vulkan_qimg_runner *r, BufInfo &x, BufInfo &v,
                          float dt, int n) {
    struct { float dt; uint32_t n; } pc;
    pc.dt = dt; pc.n = n;
    dispatchBatched(r, r->pipe_euler_step, {x, v}, &pc, sizeof(pc),
                    (uint32_t)((n + 255) / 256));
}

static void op_cfg_combine(vulkan_qimg_runner *r, BufInfo &out,
                           BufInfo &cond, BufInfo &uncond,
                           float cfg_scale, int n) {
    struct { float cfg_scale; uint32_t n; } pc;
    pc.cfg_scale = cfg_scale; pc.n = n;
    dispatchBatched(r, r->pipe_cfg_combine, {out, cond, uncond}, &pc, sizeof(pc),
                    (uint32_t)((n + 255) / 256));
}

/* ---- VAE op wrappers ---- */

static void vk_vae_conv2d(vulkan_qimg_runner *r, BufInfo &out, BufInfo &inp,
                          BufInfo &w, BufInfo &b,
                          int ci, int h, int w_s, int co, int kh, int kw,
                          int rep_pad) {
    struct { uint32_t ci, h, ws, co, kh, kw, rp; } pc;
    pc.ci = ci; pc.h = h; pc.ws = w_s; pc.co = co;
    pc.kh = kh; pc.kw = kw; pc.rp = rep_pad;
    int total = co * h * w_s;
    dispatchBatched(r, r->pipe_vae_conv2d, {out, inp, w, b}, &pc, sizeof(pc),
                    (uint32_t)((total + 255) / 256));
}

static void vk_vae_rmsnorm(vulkan_qimg_runner *r, BufInfo &out, BufInfo &inp,
                           BufInfo &gamma, int C, int spatial) {
    struct { uint32_t C, sp; } pc;
    pc.C = C; pc.sp = spatial;
    dispatchBatched(r, r->pipe_vae_rmsnorm, {out, inp, gamma}, &pc, sizeof(pc),
                    (uint32_t)((spatial + 255) / 256));
}

static void vk_vae_silu(vulkan_qimg_runner *r, BufInfo &x, int n) {
    struct { uint32_t n; } pc;
    pc.n = n;
    dispatchBatched(r, r->pipe_vae_silu, {x}, &pc, sizeof(pc),
                    (uint32_t)((n + 255) / 256));
}

static void vk_vae_up2x(vulkan_qimg_runner *r, BufInfo &out, BufInfo &in,
                         int C, int H, int W) {
    struct { uint32_t C, H, W; } pc;
    pc.C = C; pc.H = H; pc.W = W;
    int total = C * H * W;
    dispatchBatched(r, r->pipe_vae_up2x, {in, out}, &pc, sizeof(pc),
                    (uint32_t)((total + 255) / 256));
}

/* ======================================================================== */
/* Block weight load/free                                                   */
/* ======================================================================== */

static void qimg_free_block(vulkan_qimg_runner *r, qimg_vk_block_t *b) {
    BufInfo *bufs = (BufInfo *)b;
    int n = sizeof(qimg_vk_block_t) / sizeof(BufInfo);
    for (int i = 0; i < n; i++) {
        if (isValid(bufs[i])) {
            r->runner.destroyBuffer(bufs[i]);
            bufs[i] = BufInfo{};
        }
    }
}

static int qimg_load_block(vulkan_qimg_runner *r, int block_idx,
                           qimg_vk_block_t *b) {
    st_context *st = (st_context *)r->dit_st;
    char name[256];
    int ok = 1;

    #define BLK_W(field, suffix) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        b->field = qimg_st_upload_f32(r->runner, st, name); \
        if (!isValid(b->field)) ok = 0; \
    } } while(0)

    BLK_W(attn_q_w, "attn.to_q.weight"); BLK_W(attn_q_b, "attn.to_q.bias");
    BLK_W(attn_k_w, "attn.to_k.weight"); BLK_W(attn_k_b, "attn.to_k.bias");
    BLK_W(attn_v_w, "attn.to_v.weight"); BLK_W(attn_v_b, "attn.to_v.bias");
    BLK_W(attn_out_w, "attn.to_out.0.weight"); BLK_W(attn_out_b, "attn.to_out.0.bias");

    BLK_W(attn_add_q_w, "attn.add_q_proj.weight"); BLK_W(attn_add_q_b, "attn.add_q_proj.bias");
    BLK_W(attn_add_k_w, "attn.add_k_proj.weight"); BLK_W(attn_add_k_b, "attn.add_k_proj.bias");
    BLK_W(attn_add_v_w, "attn.add_v_proj.weight"); BLK_W(attn_add_v_b, "attn.add_v_proj.bias");
    BLK_W(attn_add_out_w, "attn.to_add_out.weight"); BLK_W(attn_add_out_b, "attn.to_add_out.bias");

    BLK_W(norm_q_w, "attn.norm_q.weight");
    BLK_W(norm_k_w, "attn.norm_k.weight");
    BLK_W(norm_added_q_w, "attn.norm_added_q.weight");
    BLK_W(norm_added_k_w, "attn.norm_added_k.weight");

    BLK_W(img_mod_w, "img_mod.1.weight"); BLK_W(img_mod_b, "img_mod.1.bias");
    BLK_W(img_mlp_fc1_w, "img_mlp.net.0.proj.weight"); BLK_W(img_mlp_fc1_b, "img_mlp.net.0.proj.bias");
    BLK_W(img_mlp_fc2_w, "img_mlp.net.2.weight"); BLK_W(img_mlp_fc2_b, "img_mlp.net.2.bias");

    BLK_W(txt_mod_w, "txt_mod.1.weight"); BLK_W(txt_mod_b, "txt_mod.1.bias");
    BLK_W(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight"); BLK_W(txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias");
    BLK_W(txt_mlp_fc2_w, "txt_mlp.net.2.weight"); BLK_W(txt_mlp_fc2_b, "txt_mlp.net.2.bias");

    #undef BLK_W

    if (!ok) {
        qimg_free_block(r, b);
        return -1;
    }
    return 0;
}

/* ======================================================================== */
/* Pipeline creation                                                        */
/* ======================================================================== */

static bool createPipelines(vulkan_qimg_runner *r) {
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
        fprintf(stderr, "qimg: failed to load shader: %s.spv (searched %s)\n",
                name.c_str(), r->shader_dir.c_str());
        return {};
    };

    auto createPipe = [&](const std::string &name, int nBuf, int pushSize,
                          Pipeline &pipe) -> bool {
        auto spirv = loadSpv(name);
        if (spirv.empty()) return false;
        return r->runner.createComputePipelineWithPushConstants(
            spirv, makeBindings(nBuf), pushSize, pipe);
    };

    /* Reused from Flux.2 */
    if (!createPipe("shaders/flux2_gemm_f32",              4, 16, r->pipe_gemm)) return false;
    if (!createPipe("shaders/flux2_adaln_f32",             4, 8,  r->pipe_adaln)) return false;
    if (!createPipe("shaders/flux2_rmsnorm_per_head_f32",  2, 12, r->pipe_rmsnorm_ph)) return false;
    if (!createPipe("shaders/flux2_gated_add_f32",         3, 8,  r->pipe_gated_add)) return false;
    if (!createPipe("shaders/flux2_bf16_trunc_f32",        1, 4,  r->pipe_bf16_trunc)) return false;
    if (!createPipe("shaders/flux2_attn_flash_f32",        4, 16, r->pipe_attn)) return false;
    if (!createPipe("shaders/flux2_swiglu_f32",            2, 8,  r->pipe_silu)) return false;

    /* Shared shaders */
    if (!createPipe("shaders/gelu_f32",                    1, 4,  r->pipe_gelu)) return false;
    if (!createPipe("shaders/silu_f32",                    1, 4,  r->pipe_silu)) return false;
    if (!createPipe("shaders/layernorm_f32",               4, 12, r->pipe_layernorm)) return false;
    if (!createPipe("shaders/add_f32",                     2, 4,  r->pipe_add)) return false;

    /* Qimg-specific shaders */
    if (!createPipe("shaders/qimg_rope_2d_f32",            2, 36, r->pipe_rope_2d)) return false;
    if (!createPipe("shaders/qimg_rope_1d_f32",            2, 32, r->pipe_rope_1d)) return false;
    if (!createPipe("shaders/qimg_patchify_f32",           2, 8,  r->pipe_patchify)) return false;
    if (!createPipe("shaders/qimg_unpatchify_f32",         2, 8,  r->pipe_unpatchify)) return false;
    if (!createPipe("shaders/qimg_euler_step_f32",         2, 8,  r->pipe_euler_step)) return false;
    if (!createPipe("shaders/qimg_cfg_combine_f32",        3, 8,  r->pipe_cfg_combine)) return false;
    if (!createPipe("shaders/qimg_rmsnorm_weighted_f32",   2, 8,  r->pipe_rmsnorm_weighted)) return false;

    /* VAE shaders */
    if (!createPipe("shaders/qimg_vae_conv2d_f32",         4, 28, r->pipe_vae_conv2d)) return false;
    if (!createPipe("shaders/qimg_vae_rmsnorm_f32",        3, 8,  r->pipe_vae_rmsnorm)) return false;
    if (!createPipe("shaders/qimg_vae_silu_f32",           1, 4,  r->pipe_vae_silu)) return false;
    if (!createPipe("shaders/qimg_vae_upsample2x_f32",    2, 12, r->pipe_vae_up2x)) return false;

    if (r->verbose >= 1) fprintf(stderr, "qimg: all pipelines created\n");
    return true;
}

/* ======================================================================== */
/* Init                                                                     */
/* ======================================================================== */

vulkan_qimg_runner *vulkan_qimg_init(int device_id, int verbose,
                                     const char *shader_dir) {
    auto *r = new vulkan_qimg_runner();
    r->verbose = verbose;
    r->shader_dir = shader_dir ? shader_dir : ".";
    r->dit_st = nullptr;
    r->vae_st = nullptr;
    r->dim = 3072; r->n_heads = 24; r->head_dim = 128;
    r->n_blocks = 0;
    r->in_ch = 64; r->txt_dim = 3584; r->mlp_h = 12288;

    if (!vl_cpp::vulkan::InitializeVulkan()) {
        fprintf(stderr, "qimg: failed to initialize Vulkan loader\n");
        delete r;
        return nullptr;
    }

    if (!r->runner.initialize(false)) {
        fprintf(stderr, "qimg: Vulkan init failed\n");
        delete r;
        return nullptr;
    }

    if (!r->runner.selectDevice((uint32_t)device_id)) {
        fprintf(stderr, "qimg: failed to select device %d\n", device_id);
        delete r;
        return nullptr;
    }

    if (verbose >= 1)
        fprintf(stderr, "qimg: device %d = %s\n", device_id,
                r->runner.getDeviceName(device_id).c_str());

    r->runner.createDynamicDescriptorPool(4096);

    if (!createPipelines(r)) {
        fprintf(stderr, "qimg: pipeline creation failed\n");
        delete r;
        return nullptr;
    }

    if (verbose >= 1) fprintf(stderr, "qimg: init OK\n");
    return r;
}

/* ======================================================================== */
/* Load DiT                                                                 */
/* ======================================================================== */

int vulkan_qimg_load_dit(vulkan_qimg_runner *r, const char *path) {
    if (r->verbose >= 1)
        fprintf(stderr, "qimg: loading DiT %s\n", path);

    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->dit_st = st;

    /* Count blocks */
    r->n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *bp = strstr(nm, "transformer_blocks.");
        if (bp) {
            int blk = atoi(bp + 19);
            if (blk + 1 > r->n_blocks) r->n_blocks = blk + 1;
        }
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Upload global weights */
    r->d_img_in_w  = qimg_st_upload_f32(r->runner, st, "img_in.weight");
    r->d_img_in_b  = qimg_st_upload_f32(r->runner, st, "img_in.bias");
    r->d_txt_in_w  = qimg_st_upload_f32(r->runner, st, "txt_in.weight");
    r->d_txt_in_b  = qimg_st_upload_f32(r->runner, st, "txt_in.bias");
    r->d_txt_norm_w = qimg_st_upload_f32(r->runner, st, "txt_norm.weight");
    r->d_t_fc1_w   = qimg_st_upload_f32(r->runner, st,
                         "time_text_embed.timestep_embedder.linear_1.weight");
    r->d_t_fc1_b   = qimg_st_upload_f32(r->runner, st,
                         "time_text_embed.timestep_embedder.linear_1.bias");
    r->d_t_fc2_w   = qimg_st_upload_f32(r->runner, st,
                         "time_text_embed.timestep_embedder.linear_2.weight");
    r->d_t_fc2_b   = qimg_st_upload_f32(r->runner, st,
                         "time_text_embed.timestep_embedder.linear_2.bias");
    r->d_norm_out_w = qimg_st_upload_f32(r->runner, st, "norm_out.linear.weight");
    r->d_norm_out_b = qimg_st_upload_f32(r->runner, st, "norm_out.linear.bias");
    r->d_proj_out_w = qimg_st_upload_f32(r->runner, st, "proj_out.weight");
    r->d_proj_out_b = qimg_st_upload_f32(r->runner, st, "proj_out.bias");

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    if (r->verbose >= 1)
        fprintf(stderr, "qimg: loaded %d blocks, dim=%d (%.1f s)\n",
                r->n_blocks, r->dim, dt);
    return 0;
}

/* ======================================================================== */
/* Load VAE                                                                 */
/* ======================================================================== */

int vulkan_qimg_load_vae(vulkan_qimg_runner *r, const char *path) {
    if (r->verbose >= 1)
        fprintf(stderr, "qimg: loading VAE %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->vae_st = st;
    if (r->verbose >= 1)
        fprintf(stderr, "qimg: VAE loaded (%d tensors)\n", st->n_tensors);
    return 0;
}

/* ======================================================================== */
/* DiT single step                                                          */
/* ======================================================================== */

int vulkan_qimg_dit_step(vulkan_qimg_runner *r,
                         const float *img_tokens, int n_img,
                         const float *txt_tokens, int n_txt,
                         float timestep, float *out) {
    if (!r->dit_st) {
        fprintf(stderr, "qimg: DiT not loaded\n");
        return -1;
    }

    int dim = r->dim;
    int nh = r->n_heads, hd = r->head_dim;
    int in_ch = r->in_ch, txt_dim = r->txt_dim, mlp_h = r->mlp_h;
    int n_total = n_img + n_txt;
    size_t F = sizeof(float);
    BufInfo null_buf{};

    /* Allocate activation buffers */
    BufInfo d_img       = createGpuBuffer(r->runner, (size_t)n_img * dim * F);
    BufInfo d_txt       = createGpuBuffer(r->runner, (size_t)n_txt * dim * F);
    BufInfo d_t_emb     = createGpuBuffer(r->runner, (size_t)dim * F);
    BufInfo d_img_in    = createGpuBuffer(r->runner, (size_t)n_img * in_ch * F);
    BufInfo d_txt_in    = createGpuBuffer(r->runner, (size_t)n_txt * txt_dim * F);

    /* Upload inputs */
    r->runner.uploadToDeviceLocal(d_img_in, img_tokens,
                                  (size_t)n_img * in_ch * F);
    r->runner.uploadToDeviceLocal(d_txt_in, txt_tokens,
                                  (size_t)n_txt * txt_dim * F);

    /* BF16 truncate inputs (match ComfyUI precision) */
    beginBatch(r);
    op_bf16_trunc(r, d_img_in, n_img * in_ch);
    op_bf16_trunc(r, d_txt_in, n_txt * txt_dim);
    endBatch(r);

    /* 1. Timestep embedding: sinusoidal(256) -> SiLU(GEMM) -> GEMM */
    float t_sin[256];
    {
        int half = 128;
        for (int i = 0; i < half; i++) {
            float freq = expf(-(float)i / (float)half * logf(10000.0f));
            float angle = timestep * freq;
            t_sin[i]        = cosf(angle);
            t_sin[half + i] = sinf(angle);
        }
    }
    BufInfo d_t_sin = uploadF32(r->runner, t_sin, 256);

    BufInfo d_t_emb2 = createGpuBuffer(r->runner, (size_t)dim * F);

    beginBatch(r);
    op_gemm_bf16(r, d_t_emb, r->d_t_fc1_w, d_t_sin, r->d_t_fc1_b,
                 dim, 256, 1);
    op_silu(r, d_t_emb, dim);
    op_gemm_bf16(r, d_t_emb2, r->d_t_fc2_w, d_t_emb, r->d_t_fc2_b,
                 dim, dim, 1);
    endBatch(r);

    r->runner.destroyBuffer(d_t_emb);
    r->runner.destroyBuffer(d_t_sin);
    d_t_emb = d_t_emb2;

    /* 2. Text input: RMSNorm -> Linear */
    beginBatch(r);
    if (isValid(r->d_txt_norm_w))
        op_rmsnorm_weighted(r, d_txt_in, r->d_txt_norm_w, n_txt, txt_dim);
    op_gemm_bf16(r, d_txt, r->d_txt_in_w, d_txt_in, r->d_txt_in_b,
                 dim, txt_dim, n_txt);
    endBatch(r);
    r->runner.destroyBuffer(d_txt_in);

    /* 3. Image input: GEMM(64->3072) */
    beginBatch(r);
    op_gemm_bf16(r, d_img, r->d_img_in_w, d_img_in, r->d_img_in_b,
                 dim, in_ch, n_img);
    endBatch(r);
    r->runner.destroyBuffer(d_img_in);

    /* BF16 truncation after projection */
    beginBatch(r);
    op_bf16_trunc(r, d_img, n_img * dim);
    op_bf16_trunc(r, d_txt, n_txt * dim);
    op_bf16_trunc(r, d_t_emb, dim);
    endBatch(r);

    /* Scratch buffers */
    size_t max_n = (size_t)(n_img > n_txt ? n_img : n_txt);
    BufInfo d_scratch1 = createGpuBuffer(r->runner, (size_t)n_total * dim * F);
    BufInfo d_scratch2 = createGpuBuffer(r->runner, (size_t)n_total * dim * F);
    BufInfo d_scratch3 = createGpuBuffer(r->runner, max_n * mlp_h * F);

    /* Separate img/txt Q/K/V buffers for GEMM output */
    BufInfo d_img_q = createGpuBuffer(r->runner, (size_t)n_img * dim * F);
    BufInfo d_img_k = createGpuBuffer(r->runner, (size_t)n_img * dim * F);
    BufInfo d_img_v = createGpuBuffer(r->runner, (size_t)n_img * dim * F);
    BufInfo d_txt_q = createGpuBuffer(r->runner, (size_t)n_txt * dim * F);
    BufInfo d_txt_k = createGpuBuffer(r->runner, (size_t)n_txt * dim * F);
    BufInfo d_txt_v = createGpuBuffer(r->runner, (size_t)n_txt * dim * F);
    BufInfo d_attn_out = createGpuBuffer(r->runner, (size_t)n_total * dim * F);

    /* RoPE params */
    int hp_rope = (int)sqrtf((float)n_img);
    int wp_rope = n_img / hp_rope;
    float rope_theta = 10000.0f;
    int t_dim_rope = 16, h_dim_rope = 56, w_dim_rope = 56;

    /* 4. Process all blocks */
    for (int L = 0; L < r->n_blocks; L++) {
        if (r->verbose && (L % 10 == 0 || L == r->n_blocks - 1))
            fprintf(stderr, "\r  vulkan_qimg: block %d/%d", L + 1, r->n_blocks);

        /* Load block weights on-demand */
        qimg_vk_block_t blk;
        memset(&blk, 0, sizeof(blk));
        if (qimg_load_block(r, L, &blk) != 0) {
            fprintf(stderr, "\nqimg: failed to load block %d\n", L);
            break;
        }

        /* Image modulation: SiLU(t_emb) -> Linear -> 6*dim */
        BufInfo d_t_silu = createGpuBuffer(r->runner, (size_t)dim * F);
        {
            /* Copy t_emb to d_t_silu via CPU */
            float *h_temb = (float *)malloc(dim * F);
            r->runner.downloadFromDeviceLocal(d_t_emb, h_temb, dim * F);
            r->runner.uploadToDeviceLocal(d_t_silu, h_temb, dim * F);
            free(h_temb);
        }

        BufInfo d_img_mod = createGpuBuffer(r->runner, (size_t)6 * dim * F);
        BufInfo d_txt_mod = createGpuBuffer(r->runner, (size_t)6 * dim * F);

        beginBatch(r);
        op_silu(r, d_t_silu, dim);
        op_gemm_bf16(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b,
                     6 * dim, dim, 1);
        op_gemm_bf16(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b,
                     6 * dim, dim, 1);
        endBatch(r);

        r->runner.destroyBuffer(d_t_silu);

        /* Download modulations, split into 6 parts on CPU, re-upload.
         * This avoids SSBO offset issues. */
        float *h_img_mod = (float *)malloc(6 * dim * F);
        float *h_txt_mod = (float *)malloc(6 * dim * F);
        r->runner.downloadFromDeviceLocal(d_img_mod, h_img_mod, 6 * dim * F);
        r->runner.downloadFromDeviceLocal(d_txt_mod, h_txt_mod, 6 * dim * F);
        r->runner.destroyBuffer(d_img_mod);
        r->runner.destroyBuffer(d_txt_mod);

        /* Split modulation into 6 x dim */
        BufInfo img_sh1 = uploadF32(r->runner, h_img_mod + 0 * dim, dim);
        BufInfo img_sc1 = uploadF32(r->runner, h_img_mod + 1 * dim, dim);
        BufInfo img_g1  = uploadF32(r->runner, h_img_mod + 2 * dim, dim);
        BufInfo img_sh2 = uploadF32(r->runner, h_img_mod + 3 * dim, dim);
        BufInfo img_sc2 = uploadF32(r->runner, h_img_mod + 4 * dim, dim);
        BufInfo img_g2  = uploadF32(r->runner, h_img_mod + 5 * dim, dim);

        BufInfo txt_sh1 = uploadF32(r->runner, h_txt_mod + 0 * dim, dim);
        BufInfo txt_sc1 = uploadF32(r->runner, h_txt_mod + 1 * dim, dim);
        BufInfo txt_g1  = uploadF32(r->runner, h_txt_mod + 2 * dim, dim);
        BufInfo txt_sh2 = uploadF32(r->runner, h_txt_mod + 3 * dim, dim);
        BufInfo txt_sc2 = uploadF32(r->runner, h_txt_mod + 4 * dim, dim);
        BufInfo txt_g2  = uploadF32(r->runner, h_txt_mod + 5 * dim, dim);

        free(h_img_mod);
        free(h_txt_mod);

        /* adaLN image -> d_scratch1, text -> d_scratch2 */
        beginBatch(r);
        op_adaln(r, d_scratch1, d_img, img_sh1, img_sc1, n_img, dim);
        op_adaln(r, d_scratch2, d_txt, txt_sh1, txt_sc1, n_txt, dim);

        /* Image QKV -> separate buffers */
        op_gemm_bf16(r, d_img_q, blk.attn_q_w, d_scratch1, blk.attn_q_b,
                     dim, dim, n_img);
        op_gemm_bf16(r, d_img_k, blk.attn_k_w, d_scratch1, blk.attn_k_b,
                     dim, dim, n_img);
        op_gemm_bf16(r, d_img_v, blk.attn_v_w, d_scratch1, blk.attn_v_b,
                     dim, dim, n_img);

        /* Text QKV -> separate buffers */
        op_gemm_bf16(r, d_txt_q, blk.attn_add_q_w, d_scratch2, blk.attn_add_q_b,
                     dim, dim, n_txt);
        op_gemm_bf16(r, d_txt_k, blk.attn_add_k_w, d_scratch2, blk.attn_add_k_b,
                     dim, dim, n_txt);
        op_gemm_bf16(r, d_txt_v, blk.attn_add_v_w, d_scratch2, blk.attn_add_v_b,
                     dim, dim, n_txt);

        /* QK RMSNorm per head */
        op_rmsnorm_ph(r, d_img_q, blk.norm_q_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_img_k, blk.norm_k_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_txt_q, blk.norm_added_q_w, n_txt, nh, hd);
        op_rmsnorm_ph(r, d_txt_k, blk.norm_added_k_w, n_txt, nh, hd);

        /* RoPE */
        op_rope_2d(r, d_img_q, d_img_k, n_img, nh, hd,
                   hp_rope, wp_rope, t_dim_rope, h_dim_rope, w_dim_rope,
                   rope_theta);
        {
            int txt_start = hp_rope > wp_rope ? hp_rope / 2 : wp_rope / 2;
            op_rope_1d(r, d_txt_q, d_txt_k, n_txt, nh, hd, txt_start,
                       t_dim_rope, h_dim_rope, w_dim_rope, rope_theta);
        }
        endBatch(r);

        /* Concatenate img + txt Q/K/V for joint attention via CPU */
        size_t img_bytes = (size_t)n_img * dim * F;
        size_t txt_bytes = (size_t)n_txt * dim * F;
        size_t tot_bytes = (size_t)n_total * dim * F;

        BufInfo joint_q = createGpuBuffer(r->runner, tot_bytes);
        BufInfo joint_k = createGpuBuffer(r->runner, tot_bytes);
        BufInfo joint_v = createGpuBuffer(r->runner, tot_bytes);

        {
            float *hq = (float *)malloc(tot_bytes);
            float *hk = (float *)malloc(tot_bytes);
            float *hv = (float *)malloc(tot_bytes);
            /* txt first, then img (matching HIP runner layout) */
            r->runner.downloadFromDeviceLocal(d_txt_q, hq, txt_bytes);
            r->runner.downloadFromDeviceLocal(d_img_q, hq + n_txt * dim, img_bytes);
            r->runner.downloadFromDeviceLocal(d_txt_k, hk, txt_bytes);
            r->runner.downloadFromDeviceLocal(d_img_k, hk + n_txt * dim, img_bytes);
            r->runner.downloadFromDeviceLocal(d_txt_v, hv, txt_bytes);
            r->runner.downloadFromDeviceLocal(d_img_v, hv + n_txt * dim, img_bytes);
            r->runner.uploadToDeviceLocal(joint_q, hq, tot_bytes);
            r->runner.uploadToDeviceLocal(joint_k, hk, tot_bytes);
            r->runner.uploadToDeviceLocal(joint_v, hv, tot_bytes);
            free(hq); free(hk); free(hv);
        }

        /* Joint attention */
        beginBatch(r);
        op_attn(r, d_attn_out, joint_q, joint_k, joint_v, n_total, nh, hd);
        endBatch(r);

        r->runner.destroyBuffer(joint_q);
        r->runner.destroyBuffer(joint_k);
        r->runner.destroyBuffer(joint_v);

        /* Split attn_out -> img_attn (at offset n_txt), txt_attn (at 0) */
        BufInfo d_img_attn = createGpuBuffer(r->runner, img_bytes);
        BufInfo d_txt_attn = createGpuBuffer(r->runner, txt_bytes);
        {
            float *ha = (float *)malloc(tot_bytes);
            r->runner.downloadFromDeviceLocal(d_attn_out, ha, tot_bytes);
            r->runner.uploadToDeviceLocal(d_txt_attn, ha, txt_bytes);
            r->runner.uploadToDeviceLocal(d_img_attn, ha + n_txt * dim, img_bytes);
            free(ha);
        }

        /* Output projections */
        beginBatch(r);
        op_gemm_bf16(r, d_scratch1, blk.attn_out_w, d_img_attn, blk.attn_out_b,
                     dim, dim, n_img);
        op_gemm_bf16(r, d_scratch2, blk.attn_add_out_w, d_txt_attn,
                     blk.attn_add_out_b, dim, dim, n_txt);

        /* Gated residual */
        op_gated_add(r, d_img, d_scratch1, img_g1, n_img, dim);
        op_gated_add(r, d_txt, d_scratch2, txt_g1, n_txt, dim);

        /* MLP: Image (GELU) */
        op_adaln(r, d_scratch1, d_img, img_sh2, img_sc2, n_img, dim);
        op_gemm_bf16(r, d_scratch3, blk.img_mlp_fc1_w, d_scratch1,
                     blk.img_mlp_fc1_b, mlp_h, dim, n_img);
        op_gelu(r, d_scratch3, n_img * mlp_h);
        op_gemm_bf16(r, d_scratch1, blk.img_mlp_fc2_w, d_scratch3,
                     blk.img_mlp_fc2_b, dim, mlp_h, n_img);
        op_gated_add(r, d_img, d_scratch1, img_g2, n_img, dim);

        /* MLP: Text (GELU) */
        op_adaln(r, d_scratch2, d_txt, txt_sh2, txt_sc2, n_txt, dim);
        op_gemm_bf16(r, d_scratch3, blk.txt_mlp_fc1_w, d_scratch2,
                     blk.txt_mlp_fc1_b, mlp_h, dim, n_txt);
        op_gelu(r, d_scratch3, n_txt * mlp_h);
        op_gemm_bf16(r, d_scratch2, blk.txt_mlp_fc2_w, d_scratch3,
                     blk.txt_mlp_fc2_b, dim, mlp_h, n_txt);
        op_gated_add(r, d_txt, d_scratch2, txt_g2, n_txt, dim);

        /* BF16 truncation */
        op_bf16_trunc(r, d_img, n_img * dim);
        op_bf16_trunc(r, d_txt, n_txt * dim);
        endBatch(r);

        /* Free modulation buffers */
        auto freeB = [&](BufInfo &b) {
            if (isValid(b)) { r->runner.destroyBuffer(b); b = BufInfo{}; }
        };
        freeB(img_sh1); freeB(img_sc1); freeB(img_g1);
        freeB(img_sh2); freeB(img_sc2); freeB(img_g2);
        freeB(txt_sh1); freeB(txt_sc1); freeB(txt_g1);
        freeB(txt_sh2); freeB(txt_sc2); freeB(txt_g2);
        freeB(d_img_attn); freeB(d_txt_attn);

        /* Free block weights */
        qimg_free_block(r, &blk);
    }
    if (r->verbose) fprintf(stderr, "\n");

    /* 5. Final output: adaLN -> proj_out */
    {
        /* silu(t_emb) on CPU */
        float *h_temb = (float *)malloc(dim * F);
        r->runner.downloadFromDeviceLocal(d_t_emb, h_temb, dim * F);
        for (int i = 0; i < dim; i++)
            h_temb[i] = h_temb[i] / (1.0f + expf(-h_temb[i]));

        BufInfo d_t_silu = uploadF32(r->runner, h_temb, dim);
        free(h_temb);

        /* norm_out: GEMM -> 2*dim (scale, shift) */
        BufInfo d_final_mod = createGpuBuffer(r->runner, (size_t)2 * dim * F);
        beginBatch(r);
        op_gemm_bf16(r, d_final_mod, r->d_norm_out_w, d_t_silu, r->d_norm_out_b,
                     2 * dim, dim, 1);
        endBatch(r);
        r->runner.destroyBuffer(d_t_silu);

        /* Download and split scale/shift */
        float *h_final_mod = (float *)malloc(2 * dim * F);
        r->runner.downloadFromDeviceLocal(d_final_mod, h_final_mod, 2 * dim * F);
        r->runner.destroyBuffer(d_final_mod);

        BufInfo f_scale = uploadF32(r->runner, h_final_mod, dim);
        BufInfo f_shift = uploadF32(r->runner, h_final_mod + dim, dim);
        free(h_final_mod);

        BufInfo d_out = createGpuBuffer(r->runner, (size_t)n_img * in_ch * F);

        beginBatch(r);
        op_adaln(r, d_scratch1, d_img, f_shift, f_scale, n_img, dim);
        op_gemm_bf16(r, d_out, r->d_proj_out_w, d_scratch1, r->d_proj_out_b,
                     in_ch, dim, n_img);
        endBatch(r);

        /* Download result */
        r->runner.downloadFromDeviceLocal(d_out, out,
                                          (size_t)n_img * in_ch * F);

        r->runner.destroyBuffer(f_scale);
        r->runner.destroyBuffer(f_shift);
        r->runner.destroyBuffer(d_out);
    }

    /* Cleanup */
    auto freeB = [&](BufInfo &b) {
        if (isValid(b)) { r->runner.destroyBuffer(b); b = BufInfo{}; }
    };
    freeB(d_img); freeB(d_txt); freeB(d_t_emb);
    freeB(d_scratch1); freeB(d_scratch2); freeB(d_scratch3);
    freeB(d_img_q); freeB(d_img_k); freeB(d_img_v);
    freeB(d_txt_q); freeB(d_txt_k); freeB(d_txt_v);
    freeB(d_attn_out);

    return 0;
}

/* ======================================================================== */
/* VAE ResBlock helper                                                      */
/* ======================================================================== */

static BufInfo vae_resblock_gpu(vulkan_qimg_runner *r, BufInfo &x,
                                BufInfo &n1_g, BufInfo &c1_w, BufInfo &c1_b,
                                BufInfo &n2_g, BufInfo &c2_w, BufInfo &c2_b,
                                BufInfo &sc_w, BufInfo &sc_b,
                                int ci, int co, int h, int w) {
    int sp = h * w;
    size_t F = sizeof(float);

    BufInfo tmp   = createGpuBuffer(r->runner, (size_t)ci * sp * F);
    BufInfo c1out = createGpuBuffer(r->runner, (size_t)co * sp * F);

    beginBatch(r);
    vk_vae_rmsnorm(r, tmp, x, n1_g, ci, sp);
    vk_vae_silu(r, tmp, ci * sp);
    vk_vae_conv2d(r, c1out, tmp, c1_w, c1_b, ci, h, w, co, 3, 3, 0);
    endBatch(r);
    r->runner.destroyBuffer(tmp);

    tmp = createGpuBuffer(r->runner, (size_t)co * sp * F);
    BufInfo c2out = createGpuBuffer(r->runner, (size_t)co * sp * F);

    beginBatch(r);
    vk_vae_rmsnorm(r, tmp, c1out, n2_g, co, sp);
    vk_vae_silu(r, tmp, co * sp);
    vk_vae_conv2d(r, c2out, tmp, c2_w, c2_b, co, h, w, co, 3, 3, 0);
    endBatch(r);
    r->runner.destroyBuffer(tmp);
    r->runner.destroyBuffer(c1out);

    BufInfo out_buf = createGpuBuffer(r->runner, (size_t)co * sp * F);
    if (isValid(sc_w)) {
        /* Skip connection with conv */
        beginBatch(r);
        vk_vae_conv2d(r, out_buf, x, sc_w, sc_b, ci, h, w, co, 1, 1, 0);
        op_add(r, out_buf, c2out, co * sp);
        endBatch(r);
    } else {
        /* Identity skip: copy x, then add residual */
        size_t bytes = (size_t)co * sp * F;
        float *tmp_h = (float *)malloc(bytes);
        r->runner.downloadFromDeviceLocal(x, tmp_h, bytes);
        r->runner.uploadToDeviceLocal(out_buf, tmp_h, bytes);
        free(tmp_h);

        beginBatch(r);
        op_add(r, out_buf, c2out, co * sp);
        endBatch(r);
    }
    r->runner.destroyBuffer(c2out);

    return out_buf;
}

/* ======================================================================== */
/* VAE decode on GPU                                                        */
/* ======================================================================== */

int vulkan_qimg_vae_decode(vulkan_qimg_runner *r,
                           const float *latent, int lat_h, int lat_w,
                           float *out_rgb) {
    st_context *st = (st_context *)r->vae_st;
    if (!st) {
        fprintf(stderr, "qimg: VAE not loaded\n");
        return -1;
    }

    int h = lat_h, w = lat_w, c = 16;
    size_t F = sizeof(float);
    fprintf(stderr, "vulkan_qimg_vae: decoding [%d, %d, %d] on GPU\n", c, h, w);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Upload latent */
    BufInfo d_x = uploadF32(r->runner, latent, c * h * w);

    /* post_quant_conv */
    {
        BufInfo d_pqc_w = qimg_st_upload_f32(r->runner, st, "conv2.weight");
        BufInfo d_pqc_b = qimg_st_upload_f32(r->runner, st, "conv2.bias");
        if (isValid(d_pqc_w)) {
            BufInfo d_tmp = createGpuBuffer(r->runner, (size_t)c * h * w * F);
            beginBatch(r);
            vk_vae_conv2d(r, d_tmp, d_x, d_pqc_w, d_pqc_b, c, h, w, c, 1, 1, 0);
            endBatch(r);
            r->runner.destroyBuffer(d_x);
            d_x = d_tmp;
            r->runner.destroyBuffer(d_pqc_w);
            r->runner.destroyBuffer(d_pqc_b);
        }
    }

    /* decoder.conv1: 16->384, 3x3 */
    {
        int co_c1, ci_c1;
        BufInfo d_c1_w = qimg_upload_conv3d(r->runner, st,
                                            "decoder.conv1.weight",
                                            &co_c1, &ci_c1);
        BufInfo d_c1_b = qimg_st_upload_f32(r->runner, st,
                                            "decoder.conv1.bias");
        c = co_c1;
        BufInfo d_tmp = createGpuBuffer(r->runner, (size_t)c * h * w * F);
        beginBatch(r);
        vk_vae_conv2d(r, d_tmp, d_x, d_c1_w, d_c1_b, ci_c1, h, w, c, 3, 3, 0);
        endBatch(r);
        r->runner.destroyBuffer(d_x);
        d_x = d_tmp;
        r->runner.destroyBuffer(d_c1_w);
        r->runner.destroyBuffer(d_c1_b);
    }
    fprintf(stderr, "  after conv1: [%d, %d, %d]\n", c, h, w);

    /* Helper macro to load resblock weights from safetensors */
    #define LOAD_RB_VK(pfx_str, n1, c1w, c1b, n2, c2w, c2b, scw, scb) \
        BufInfo n1, c1w, c1b, n2, c2w, c2b, scw{}, scb{}; \
        { char _nm[256]; \
          snprintf(_nm, sizeof(_nm), "%s.residual.0.gamma", pfx_str); \
          n1 = qimg_st_upload_f32(r->runner, st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.2.weight", pfx_str); \
          { int _co, _ci; c1w = qimg_upload_conv3d(r->runner, st, _nm, &_co, &_ci); } \
          snprintf(_nm, sizeof(_nm), "%s.residual.2.bias", pfx_str); \
          c1b = qimg_st_upload_f32(r->runner, st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.3.gamma", pfx_str); \
          n2 = qimg_st_upload_f32(r->runner, st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.6.weight", pfx_str); \
          { int _co2, _ci2; c2w = qimg_upload_conv3d(r->runner, st, _nm, &_co2, &_ci2); } \
          snprintf(_nm, sizeof(_nm), "%s.residual.6.bias", pfx_str); \
          c2b = qimg_st_upload_f32(r->runner, st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.shortcut.weight", pfx_str); \
          if (safetensors_find(st, _nm) >= 0) { \
              scw = qimg_st_upload_f32(r->runner, st, _nm); \
              snprintf(_nm, sizeof(_nm), "%s.shortcut.bias", pfx_str); \
              scb = qimg_st_upload_f32(r->runner, st, _nm); } }

    #define FREE_RB_VK(n1, c1w, c1b, n2, c2w, c2b, scw, scb) do { \
        r->runner.destroyBuffer(n1); r->runner.destroyBuffer(c1w); \
        r->runner.destroyBuffer(c1b); r->runner.destroyBuffer(n2); \
        r->runner.destroyBuffer(c2w); r->runner.destroyBuffer(c2b); \
        if (isValid(scw)) r->runner.destroyBuffer(scw); \
        if (isValid(scb)) r->runner.destroyBuffer(scb); \
    } while(0)

    /* mid.0 */
    {
        LOAD_RB_VK("decoder.middle.0", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
        BufInfo d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b,
                                         n2, c2w, c2b, scw, scb, c, c, h, w);
        r->runner.destroyBuffer(d_x); d_x = d_tmp;
        FREE_RB_VK(n1, c1w, c1b, n2, c2w, c2b, scw, scb);
    }

    /* Middle attention: CPU fallback for spatial self-attention */
    {
        int spatial = h * w;
        BufInfo d_gn_g = qimg_st_upload_f32(r->runner, st,
                             "decoder.middle.1.norm.gamma");
        BufInfo d_qkv_w = qimg_st_upload_f32(r->runner, st,
                             "decoder.middle.1.to_qkv.weight");
        BufInfo d_qkv_b = qimg_st_upload_f32(r->runner, st,
                             "decoder.middle.1.to_qkv.bias");
        BufInfo d_proj_w = qimg_st_upload_f32(r->runner, st,
                             "decoder.middle.1.proj.weight");
        BufInfo d_proj_b = qimg_st_upload_f32(r->runner, st,
                             "decoder.middle.1.proj.bias");

        BufInfo d_normed = createGpuBuffer(r->runner, (size_t)c * spatial * F);
        beginBatch(r);
        vk_vae_rmsnorm(r, d_normed, d_x, d_gn_g, c, spatial);
        endBatch(r);
        r->runner.destroyBuffer(d_gn_g);

        BufInfo d_qkv = createGpuBuffer(r->runner, (size_t)3 * c * spatial * F);
        beginBatch(r);
        vk_vae_conv2d(r, d_qkv, d_normed, d_qkv_w, d_qkv_b,
                      c, h, w, 3 * c, 1, 1, 0);
        endBatch(r);
        r->runner.destroyBuffer(d_normed);
        r->runner.destroyBuffer(d_qkv_w);
        r->runner.destroyBuffer(d_qkv_b);

        /* CPU attention */
        float *h_qkv = (float *)malloc((size_t)3 * c * spatial * F);
        r->runner.downloadFromDeviceLocal(d_qkv, h_qkv,
                                          (size_t)3 * c * spatial * F);
        r->runner.destroyBuffer(d_qkv);

        /* Transpose from [3c, spatial] to [spatial, 3c] */
        float *h_qkv_t = (float *)malloc((size_t)3 * c * spatial * F);
        for (int s = 0; s < spatial; s++)
            for (int ch = 0; ch < 3 * c; ch++)
                h_qkv_t[s * 3 * c + ch] = h_qkv[ch * spatial + s];
        free(h_qkv);

        /* Split Q/K/V */
        float *h_q = (float *)malloc((size_t)spatial * c * F);
        float *h_k = (float *)malloc((size_t)spatial * c * F);
        float *h_v = (float *)malloc((size_t)spatial * c * F);
        for (int s = 0; s < spatial; s++) {
            memcpy(h_q + (size_t)s * c,
                   h_qkv_t + (size_t)s * 3 * c, (size_t)c * F);
            memcpy(h_k + (size_t)s * c,
                   h_qkv_t + (size_t)s * 3 * c + c, (size_t)c * F);
            memcpy(h_v + (size_t)s * c,
                   h_qkv_t + (size_t)s * 3 * c + 2 * c, (size_t)c * F);
        }

        /* Naive attention on CPU */
        float *h_attn = (float *)malloc((size_t)spatial * c * F);
        float scale_at = 1.0f / sqrtf((float)c);
        for (int i = 0; i < spatial; i++) {
            float mx = -1e30f;
            for (int j = 0; j < spatial; j++) {
                float dot = 0;
                for (int d = 0; d < c; d++)
                    dot += h_q[i * c + d] * h_k[j * c + d];
                dot *= scale_at;
                if (dot > mx) mx = dot;
            }
            float esum = 0;
            memset(h_attn + i * c, 0, (size_t)c * F);
            for (int j = 0; j < spatial; j++) {
                float dot = 0;
                for (int d = 0; d < c; d++)
                    dot += h_q[i * c + d] * h_k[j * c + d];
                float w_at = expf(dot * scale_at - mx);
                esum += w_at;
                for (int d = 0; d < c; d++)
                    h_attn[i * c + d] += w_at * h_v[j * c + d];
            }
            float inv = 1.0f / esum;
            for (int d = 0; d < c; d++)
                h_attn[i * c + d] *= inv;
        }
        free(h_qkv_t); free(h_q); free(h_k); free(h_v);

        /* Transpose back to [c, spatial] */
        float *h_attn_chw = (float *)malloc((size_t)c * spatial * F);
        for (int s = 0; s < spatial; s++)
            for (int ch = 0; ch < c; ch++)
                h_attn_chw[ch * spatial + s] = h_attn[s * c + ch];
        free(h_attn);

        BufInfo d_attn_chw = uploadF32(r->runner, h_attn_chw, c * spatial);
        free(h_attn_chw);

        BufInfo d_proj_out_v = createGpuBuffer(r->runner,
                                               (size_t)c * spatial * F);
        beginBatch(r);
        vk_vae_conv2d(r, d_proj_out_v, d_attn_chw, d_proj_w, d_proj_b,
                      c, h, w, c, 1, 1, 0);
        /* Residual: d_x += d_proj_out_v */
        op_add(r, d_x, d_proj_out_v, c * spatial);
        endBatch(r);

        r->runner.destroyBuffer(d_attn_chw);
        r->runner.destroyBuffer(d_proj_w);
        r->runner.destroyBuffer(d_proj_b);
        r->runner.destroyBuffer(d_proj_out_v);
    }

    /* mid.2 */
    {
        LOAD_RB_VK("decoder.middle.2", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
        BufInfo d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b,
                                         n2, c2w, c2b, scw, scb, c, c, h, w);
        r->runner.destroyBuffer(d_x); d_x = d_tmp;
        FREE_RB_VK(n1, c1w, c1b, n2, c2w, c2b, scw, scb);
    }
    fprintf(stderr, "  after middle: [%d, %d, %d]\n", c, h, w);

    /* Upsample blocks 0-14 */
    for (int i = 0; i < 15; i++) {
        char pfx[128];
        snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.residual.2.weight", i);
        if (safetensors_find(st, pfx) >= 0) {
            char rb_pfx[128];
            snprintf(rb_pfx, sizeof(rb_pfx), "decoder.upsamples.%d", i);
            LOAD_RB_VK(rb_pfx, n1, c1w, c1b, n2, c2w, c2b, scw, scb);

            snprintf(pfx, sizeof(pfx),
                     "decoder.upsamples.%d.residual.2.weight", i);
            int _idx = safetensors_find(st, pfx);
            int new_co = (int)safetensors_shape(st, _idx)[0];
            int old_ci = c;
            BufInfo d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b,
                                             n2, c2w, c2b, scw, scb,
                                             old_ci, new_co, h, w);
            r->runner.destroyBuffer(d_x); d_x = d_tmp;
            c = new_co;
            FREE_RB_VK(n1, c1w, c1b, n2, c2w, c2b, scw, scb);
        }

        snprintf(pfx, sizeof(pfx),
                 "decoder.upsamples.%d.resample.1.weight", i);
        if (safetensors_find(st, pfx) >= 0) {
            BufInfo rs_w = qimg_st_upload_f32(r->runner, st, pfx);
            snprintf(pfx, sizeof(pfx),
                     "decoder.upsamples.%d.resample.1.bias", i);
            BufInfo rs_b = qimg_st_upload_f32(r->runner, st, pfx);

            BufInfo d_up = createGpuBuffer(r->runner,
                               (size_t)c * (h * 2) * (w * 2) * F);
            beginBatch(r);
            vk_vae_up2x(r, d_up, d_x, c, h, w);
            endBatch(r);
            h *= 2; w *= 2;

            snprintf(pfx, sizeof(pfx),
                     "decoder.upsamples.%d.resample.1.weight", i);
            int rs_idx = safetensors_find(st, pfx);
            int new_c = (int)safetensors_shape(st, rs_idx)[0];

            BufInfo d_tmp = createGpuBuffer(r->runner,
                                (size_t)new_c * h * w * F);
            beginBatch(r);
            vk_vae_conv2d(r, d_tmp, d_up, rs_w, rs_b, c, h, w,
                          new_c, 3, 3, 0);
            endBatch(r);

            r->runner.destroyBuffer(d_up);
            r->runner.destroyBuffer(d_x);
            r->runner.destroyBuffer(rs_w);
            r->runner.destroyBuffer(rs_b);
            d_x = d_tmp;
            c = new_c;
            fprintf(stderr, "  upsample %d: [%d, %d, %d]\n", i, c, h, w);
        }
    }
    #undef LOAD_RB_VK
    #undef FREE_RB_VK

    /* Head: RMSNorm -> SiLU -> Conv(96->3) */
    {
        BufInfo d_gn = qimg_st_upload_f32(r->runner, st,
                                          "decoder.head.0.gamma");
        int spatial = h * w;

        BufInfo d_tmp = createGpuBuffer(r->runner, (size_t)c * spatial * F);
        beginBatch(r);
        vk_vae_rmsnorm(r, d_tmp, d_x, d_gn, c, spatial);
        vk_vae_silu(r, d_tmp, c * spatial);
        endBatch(r);
        r->runner.destroyBuffer(d_gn);

        int head_co, head_ci;
        BufInfo d_hw = qimg_upload_conv3d(r->runner, st,
                                          "decoder.head.2.weight",
                                          &head_co, &head_ci);
        BufInfo d_hb = qimg_st_upload_f32(r->runner, st,
                                          "decoder.head.2.bias");

        BufInfo d_rgb = createGpuBuffer(r->runner, (size_t)3 * spatial * F);
        beginBatch(r);
        vk_vae_conv2d(r, d_rgb, d_tmp, d_hw, d_hb, c, h, w, 3, 3, 3, 0);
        endBatch(r);

        r->runner.destroyBuffer(d_tmp);
        r->runner.destroyBuffer(d_x);
        r->runner.destroyBuffer(d_hw);
        r->runner.destroyBuffer(d_hb);

        r->runner.downloadFromDeviceLocal(d_rgb, out_rgb,
                                          (size_t)3 * h * w * F);
        r->runner.destroyBuffer(d_rgb);
        c = 3;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "vulkan_qimg_vae: decode complete [%d, %d, %d] (%.1f s)\n",
            c, h, w, dt);
    return 0;
}

/* ======================================================================== */
/* Free                                                                     */
/* ======================================================================== */

void vulkan_qimg_free(vulkan_qimg_runner *r) {
    if (!r) return;

    auto freeB = [&](BufInfo &b) {
        if (isValid(b)) r->runner.destroyBuffer(b);
    };

    /* Free global weights */
    freeB(r->d_img_in_w); freeB(r->d_img_in_b);
    freeB(r->d_txt_in_w); freeB(r->d_txt_in_b);
    freeB(r->d_txt_norm_w);
    freeB(r->d_t_fc1_w); freeB(r->d_t_fc1_b);
    freeB(r->d_t_fc2_w); freeB(r->d_t_fc2_b);
    freeB(r->d_norm_out_w); freeB(r->d_norm_out_b);
    freeB(r->d_proj_out_w); freeB(r->d_proj_out_b);

    if (r->dit_st) safetensors_close((st_context *)r->dit_st);
    if (r->vae_st) safetensors_close((st_context *)r->vae_st);

    r->runner.destroyDynamicDescriptorPool();
    r->runner.cleanup();

    delete r;
}
