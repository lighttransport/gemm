/*
 * cuda_qimg_runner.h - CUDA Qwen-Image text-to-image runner
 *
 * Uses NVRTC to compile kernels at runtime. Processes the 60-block MMDiT
 * by loading one block at a time (dequant FP8→F16 on CPU, upload, compute,
 * free) to fit within 16GB VRAM.
 *
 * Weight format: FP8 E4M3 safetensors (from Comfy-Org/Qwen-Image_ComfyUI)
 *
 * Pipeline:
 *   1. Text encoder (CPU) → hidden states [N_txt, 3584]
 *   2. MMDiT denoising (GPU) × N_steps → latent [16, H/8, W/8]
 *   3. VAE decoder (GPU) → RGB image [3, H, W]
 *
 * Usage:
 *   cuda_qimg_runner *r = cuda_qimg_init(0, 1);
 *   cuda_qimg_load_dit(r, "qwen_image_fp8_e4m3fn.safetensors");
 *   cuda_qimg_load_vae(r, "qwen_image_vae.safetensors");
 *   cuda_qimg_dit_step(r, img_tok, n_img, txt_tok, n_txt, t, out);
 *   cuda_qimg_free(r);
 */
#ifndef CUDA_QIMG_RUNNER_H
#define CUDA_QIMG_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_qimg_runner cuda_qimg_runner;

cuda_qimg_runner *cuda_qimg_init(int device_id, int verbose);
int  cuda_qimg_load_dit(cuda_qimg_runner *r, const char *safetensors_path);
int  cuda_qimg_load_vae(cuda_qimg_runner *r, const char *safetensors_path);
void cuda_qimg_free(cuda_qimg_runner *r);

/* Run single DiT denoising step on GPU.
 *   img_tokens: [n_img, 64] patchified latent (CPU)
 *   txt_tokens: [n_txt, 3584] text hidden states (CPU)
 *   out: [n_img, 64] velocity prediction (CPU, pre-allocated)
 *   Returns 0 on success. */
int cuda_qimg_dit_step(cuda_qimg_runner *r,
                       const float *img_tokens, int n_img,
                       const float *txt_tokens, int n_txt,
                       float timestep, float *out);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef CUDA_QIMG_RUNNER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_kernels_common.h"
#include "../../common/safetensors.h"

/* ---- FP8 E4M3 → F32 CPU conversion ---- */

static float fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) {
        /* Subnormal: value = (-1)^s × 2^(1-7) × (0.mant) */
        f = ldexpf((float)mant / 8.0f, -6);
    } else if (exp == 15 && mant == 7) {
        /* NaN */
        uint32_t bits = (sign << 31) | 0x7FC00000;
        memcpy(&f, &bits, 4);
        return f;
    } else {
        /* Normal: value = (-1)^s × 2^(exp-7) × (1 + mant/8) */
        f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -f : f;
}

/* ---- Kernel source for Qwen-Image specific ops ---- */

static const char *qimg_kernel_src =
"\n/* ---- Qwen-Image specific kernels ---- */\n"

/* Per-head RMSNorm: x[N, dim], w[head_dim], n_heads, head_dim */
"__global__ void rmsnorm_per_head_f32(float *__restrict__ x,\n"
"    const float *__restrict__ w, int N, int n_heads, int head_dim) {\n"
"    int tok = blockIdx.x;\n"
"    int h = blockIdx.y;\n"
"    if (tok >= N || h >= n_heads) return;\n"
"    int dim = n_heads * head_dim;\n"
"    float *hd = x + tok * dim + h * head_dim;\n"
"    float ss = 0.0f;\n"
"    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)\n"
"        ss += hd[i] * hd[i];\n"
"    /* Warp reduce */\n"
"    for (int m = 16; m > 0; m >>= 1)\n"
"        ss += __shfl_xor_sync(0xFFFFFFFF, ss, m);\n"
"    float inv = rsqrtf(ss / (float)head_dim + 1e-6f);\n"
"    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)\n"
"        hd[i] *= inv * w[i];\n"
"}\n"

/* adaLN modulate: out = LN(x) * (1+scale) + shift */
"__global__ void adaln_modulate_f32(float *__restrict__ out,\n"
"    const float *__restrict__ x, const float *__restrict__ shift,\n"
"    const float *__restrict__ scale, int N, int dim) {\n"
"    int tok = blockIdx.x;\n"
"    if (tok >= N) return;\n"
"    extern __shared__ float sdata[];\n"
"    int tid = threadIdx.x;\n"
"    /* Compute mean */\n"
"    float s = 0;\n"
"    for (int i = tid; i < dim; i += blockDim.x) s += x[tok*dim+i];\n"
"    sdata[tid] = s; __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float mean = sdata[0] / (float)dim;\n"
"    __syncthreads();\n"
"    /* Compute var */\n"
"    s = 0;\n"
"    for (int i = tid; i < dim; i += blockDim.x) { float d = x[tok*dim+i] - mean; s += d*d; }\n"
"    sdata[tid] = s; __syncthreads();\n"
"    for (int r = blockDim.x/2; r > 0; r >>= 1) { if (tid < r) sdata[tid] += sdata[tid+r]; __syncthreads(); }\n"
"    float inv = rsqrtf(sdata[0] / (float)dim + 1e-6f);\n"
"    /* Apply modulation */\n"
"    for (int i = tid; i < dim; i += blockDim.x)\n"
"        out[tok*dim+i] = ((x[tok*dim+i] - mean) * inv) * (1.0f + scale[i]) + shift[i];\n"
"}\n"

/* Gated residual: x += gate * proj */
"__global__ void gated_add_f32(float *__restrict__ x,\n"
"    const float *__restrict__ proj, const float *__restrict__ gate,\n"
"    int N, int dim) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= N * dim) return;\n"
"    int col = i % dim;\n"
"    x[i] += gate[col] * proj[i];\n"
"}\n"

/* Patchify: latent [C, H, W] → tokens [H/ps*W/ps, C*ps*ps] */
"__global__ void patchify_f32(float *__restrict__ out,\n"
"    const float *__restrict__ latent, int C, int H, int W, int ps) {\n"
"    int tok = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int hp = H / ps, wp = W / ps;\n"
"    if (tok >= hp * wp) return;\n"
"    int py = tok / wp, px = tok % wp;\n"
"    int tok_dim = C * ps * ps;\n"
"    int idx = 0;\n"
"    for (int c = 0; c < C; c++)\n"
"        for (int dy = 0; dy < ps; dy++)\n"
"            for (int dx = 0; dx < ps; dx++)\n"
"                out[tok * tok_dim + idx++] = latent[c * H * W + (py*ps+dy) * W + (px*ps+dx)];\n"
"}\n"

/* Unpatchify: tokens [seq, C*ps*ps] → latent [C, H, W] */
"__global__ void unpatchify_f32(float *__restrict__ out,\n"
"    const float *__restrict__ tokens, int C, int H, int W, int ps) {\n"
"    int tok = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int hp = H / ps, wp = W / ps;\n"
"    if (tok >= hp * wp) return;\n"
"    int py = tok / wp, px = tok % wp;\n"
"    int tok_dim = C * ps * ps;\n"
"    int idx = 0;\n"
"    for (int c = 0; c < C; c++)\n"
"        for (int dy = 0; dy < ps; dy++)\n"
"            for (int dx = 0; dx < ps; dx++)\n"
"                out[c * H * W + (py*ps+dy) * W + (px*ps+dx)] = tokens[tok * tok_dim + idx++];\n"
"}\n"

/* Euler step: x = x + dt * v */
"__global__ void euler_step_f32(float *__restrict__ x,\n"
"    const float *__restrict__ v, float dt, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) x[i] += dt * v[i];\n"
"}\n"

/* Simple self-attention: Q[N,dim], K[N,dim], V[N,dim] → out[N,dim]
 * One block per head, iterates over all query positions */
"__global__ void simple_attn_f32(float *__restrict__ out,\n"
"    const float *__restrict__ Q, const float *__restrict__ K,\n"
"    const float *__restrict__ V,\n"
"    int N, int n_heads, int head_dim) {\n"
"    int h = blockIdx.x;\n"
"    int dim = n_heads * head_dim;\n"
"    float scale = rsqrtf((float)head_dim);\n"
"    /* Process one query at a time per head */\n"
"    for (int qi = 0; qi < N; qi++) {\n"
"        const float *q = Q + qi * dim + h * head_dim;\n"
"        /* Find max for numerical stability */\n"
"        float mx = -1e30f;\n"
"        for (int j = 0; j < N; j++) {\n"
"            const float *k = K + j * dim + h * head_dim;\n"
"            float s = 0;\n"
"            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)\n"
"                s += q[d] * k[d];\n"
"            for (int m = 16; m > 0; m >>= 1)\n"
"                s += __shfl_xor_sync(0xFFFFFFFF, s, m);\n"
"            s *= scale;\n"
"            if (s > mx) mx = s;\n"
"        }\n"
"        /* Softmax + weighted sum */\n"
"        float exp_sum = 0;\n"
"        float acc[128];\n"  /* max head_dim = 128 */
"        for (int d = 0; d < head_dim; d++) acc[d] = 0;\n"
"        for (int j = 0; j < N; j++) {\n"
"            const float *k = K + j * dim + h * head_dim;\n"
"            float s = 0;\n"
"            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)\n"
"                s += q[d] * k[d];\n"
"            for (int m = 16; m > 0; m >>= 1)\n"
"                s += __shfl_xor_sync(0xFFFFFFFF, s, m);\n"
"            float w = expf(s * scale - mx);\n"
"            exp_sum += w;\n"
"            const float *v = V + j * dim + h * head_dim;\n"
"            for (int d = 0; d < head_dim; d++) acc[d] += w * v[d];\n"
"        }\n"
"        float inv = 1.0f / exp_sum;\n"
"        float *dst = out + qi * dim + h * head_dim;\n"
"        for (int d = 0; d < head_dim; d++) dst[d] = acc[d] * inv;\n"
"    }\n"
"}\n"

"}\n"; /* close extern "C" from cuda_kernels_common_src */


/* ---- Per-block GPU weight struct ---- */

typedef struct {
    CUdeviceptr attn_q_w, attn_q_b, attn_k_w, attn_k_b, attn_v_w, attn_v_b;
    CUdeviceptr attn_out_w, attn_out_b;
    CUdeviceptr attn_add_q_w, attn_add_q_b, attn_add_k_w, attn_add_k_b;
    CUdeviceptr attn_add_v_w, attn_add_v_b, attn_add_out_w, attn_add_out_b;
    CUdeviceptr norm_q_w, norm_k_w, norm_added_q_w, norm_added_k_w;
    CUdeviceptr img_mod_w, img_mod_b;
    CUdeviceptr img_mlp_fc1_w, img_mlp_fc1_b, img_mlp_fc2_w, img_mlp_fc2_b;
    CUdeviceptr txt_mod_w, txt_mod_b;
    CUdeviceptr txt_mlp_fc1_w, txt_mlp_fc1_b, txt_mlp_fc2_w, txt_mlp_fc2_b;
} qimg_block_gpu;


/* ---- Runner struct ---- */

struct cuda_qimg_runner {
    CUdevice device;
    CUcontext ctx;
    CUstream stream;
    CUmodule module;
    int sm_version;
    int verbose;

    /* Kernel handles */
    CUfunction gemm_f16_f32;
    CUfunction layernorm_f32;
    CUfunction gelu_f32;
    CUfunction silu_f32;
    CUfunction attn_prefill_f32;
    CUfunction add_bias_f32;
    CUfunction rmsnorm_per_head;
    CUfunction adaln_modulate;
    CUfunction gated_add;
    CUfunction patchify;
    CUfunction unpatchify;
    CUfunction euler_step;

    /* DiT config */
    int dit_dim, dit_n_heads, dit_head_dim, dit_n_blocks;
    int dit_in_ch, dit_txt_dim, dit_mlp_h;

    /* Safetensors context (mmap'd, kept for block-by-block loading) */
    void *dit_st;
    void *vae_st;

    /* Persistent GPU: global weights (~50MB F16) */
    CUdeviceptr d_img_in_w, d_img_in_b;
    CUdeviceptr d_txt_in_w, d_txt_in_b;
    CUdeviceptr d_txt_norm_w;
    CUdeviceptr d_t_fc1_w, d_t_fc1_b;
    CUdeviceptr d_t_fc2_w, d_t_fc2_b;
    CUdeviceptr d_norm_out_w, d_norm_out_b;
    CUdeviceptr d_proj_out_w, d_proj_out_b;
};


/* ---- Safetensor FP8→F16 upload helpers ---- */

/* Upload safetensor as F16 weight (dequant FP8→F32→F16 on CPU).
 * Safetensor shape [n_out, n_in] matches gemm_f16_f32 layout. */
static CUdeviceptr qimg_st_upload_f16(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return 0;
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];

    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    uint16_t *f16 = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        for (size_t i = 0; i < n; i++)
            f16[i] = cu_f32_to_f16(fp8_e4m3_to_f32(src[i]));
    } else if (strcmp(dtype, "F16") == 0) {
        memcpy(f16, src, n * 2);
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            float f; memcpy(&f, &bits, 4);
            f16[i] = cu_f32_to_f16(f);
        }
    } else if (strcmp(dtype, "F32") == 0) {
        const float *f32 = (const float *)src;
        for (size_t i = 0; i < n; i++)
            f16[i] = cu_f32_to_f16(f32[i]);
    } else {
        fprintf(stderr, "qimg: unsupported dtype '%s' for %s\n", dtype, name);
        free(f16);
        return 0;
    }

    CUdeviceptr d;
    cuMemAlloc(&d, n * sizeof(uint16_t));
    cuMemcpyHtoD(d, f16, n * sizeof(uint16_t));
    free(f16);
    return d;
}

/* Upload safetensor as F32 (for biases, norms) */
static CUdeviceptr qimg_st_upload_f32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return 0;
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];

    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    float *f32 = (float *)malloc(n * sizeof(float));
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        for (size_t i = 0; i < n; i++)
            f32[i] = fp8_e4m3_to_f32(src[i]);
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
            /* F16→F32 */
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
    }

    CUdeviceptr d;
    cuMemAlloc(&d, n * sizeof(float));
    cuMemcpyHtoD(d, f32, n * sizeof(float));
    free(f32);
    return d;
}


/* ---- Load/free one DiT block ---- */

static void qimg_load_block(cuda_qimg_runner *r, int block_idx, qimg_block_gpu *b) {
    st_context *st = (st_context *)r->dit_st;
    char name[256];

    #define BLK_F16(field, suffix) do { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        b->field = qimg_st_upload_f16(st, name); \
    } while(0)
    #define BLK_F32(field, suffix) do { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        b->field = qimg_st_upload_f32(st, name); \
    } while(0)

    BLK_F16(attn_q_w, "attn.to_q.weight"); BLK_F32(attn_q_b, "attn.to_q.bias");
    BLK_F16(attn_k_w, "attn.to_k.weight"); BLK_F32(attn_k_b, "attn.to_k.bias");
    BLK_F16(attn_v_w, "attn.to_v.weight"); BLK_F32(attn_v_b, "attn.to_v.bias");
    BLK_F16(attn_out_w, "attn.to_out.0.weight"); BLK_F32(attn_out_b, "attn.to_out.0.bias");

    BLK_F16(attn_add_q_w, "attn.add_q_proj.weight"); BLK_F32(attn_add_q_b, "attn.add_q_proj.bias");
    BLK_F16(attn_add_k_w, "attn.add_k_proj.weight"); BLK_F32(attn_add_k_b, "attn.add_k_proj.bias");
    BLK_F16(attn_add_v_w, "attn.add_v_proj.weight"); BLK_F32(attn_add_v_b, "attn.add_v_proj.bias");
    BLK_F16(attn_add_out_w, "attn.to_add_out.weight"); BLK_F32(attn_add_out_b, "attn.to_add_out.bias");

    BLK_F32(norm_q_w, "attn.norm_q.weight");
    BLK_F32(norm_k_w, "attn.norm_k.weight");
    BLK_F32(norm_added_q_w, "attn.norm_added_q.weight");
    BLK_F32(norm_added_k_w, "attn.norm_added_k.weight");

    BLK_F16(img_mod_w, "img_mod.1.weight"); BLK_F32(img_mod_b, "img_mod.1.bias");
    BLK_F16(img_mlp_fc1_w, "img_mlp.net.0.proj.weight"); BLK_F32(img_mlp_fc1_b, "img_mlp.net.0.proj.bias");
    BLK_F16(img_mlp_fc2_w, "img_mlp.net.2.weight"); BLK_F32(img_mlp_fc2_b, "img_mlp.net.2.bias");

    BLK_F16(txt_mod_w, "txt_mod.1.weight"); BLK_F32(txt_mod_b, "txt_mod.1.bias");
    BLK_F16(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight"); BLK_F32(txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias");
    BLK_F16(txt_mlp_fc2_w, "txt_mlp.net.2.weight"); BLK_F32(txt_mlp_fc2_b, "txt_mlp.net.2.bias");

    #undef BLK_F16
    #undef BLK_F32
}

static void qimg_free_block(qimg_block_gpu *b) {
    CUdeviceptr *ptrs = (CUdeviceptr *)b;
    int n = sizeof(qimg_block_gpu) / sizeof(CUdeviceptr);
    for (int i = 0; i < n; i++) {
        if (ptrs[i]) { cuMemFree(ptrs[i]); ptrs[i] = 0; }
    }
}


/* ---- Op launch helpers ---- */

static void op_gemm(cuda_qimg_runner *r, CUdeviceptr Y, CUdeviceptr W,
                    CUdeviceptr X, CUdeviceptr bias,
                    int n_out, int n_in, int n_tok) {
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    /* gemm_f16_f32: each block handles 256 output cols × 16 tokens,
     * 128 threads (4 warps), shared mem = 16 rows × 16 cols of F32 */
    unsigned gx = (unsigned)((n_out + 255) / 256);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    unsigned smem = 16 * 16 * sizeof(float);
    cuLaunchKernel(r->gemm_f16_f32, gx, gy, 1, 128, 1, 1,
                   smem, r->stream, args, NULL);
}

static void op_silu(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->silu_f32, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void op_gelu(cuda_qimg_runner *r, CUdeviceptr x, int n) {
    void *args[] = {&x, &n};
    cuLaunchKernel(r->gelu_f32, (unsigned)((n+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void op_adaln(cuda_qimg_runner *r, CUdeviceptr out, CUdeviceptr x,
                     CUdeviceptr shift, CUdeviceptr scale, int N, int dim) {
    void *args[] = {&out, &x, &shift, &scale, &N, &dim};
    cuLaunchKernel(r->adaln_modulate, (unsigned)N, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static void op_rmsnorm_ph(cuda_qimg_runner *r, CUdeviceptr x, CUdeviceptr w,
                          int N, int n_heads, int head_dim) {
    void *args[] = {&x, &w, &N, &n_heads, &head_dim};
    cuLaunchKernel(r->rmsnorm_per_head, (unsigned)N, (unsigned)n_heads, 1,
                   32, 1, 1, 0, r->stream, args, NULL);
}

static void op_gated_add(cuda_qimg_runner *r, CUdeviceptr x, CUdeviceptr proj,
                         CUdeviceptr gate, int N, int dim) {
    int total = N * dim;
    void *args[] = {&x, &proj, &gate, &N, &dim};
    cuLaunchKernel(r->gated_add, (unsigned)((total+255)/256), 1, 1, 256, 1, 1,
                   0, r->stream, args, NULL);
}

static void op_attn(cuda_qimg_runner *r, CUdeviceptr out, CUdeviceptr q,
                    CUdeviceptr k, CUdeviceptr v,
                    int n_tok, int n_heads, int head_dim) {
    /* simple_attn_f32: one block per head, single thread (sequential per head) */
    void *args[] = {&out, &q, &k, &v, &n_tok, &n_heads, &head_dim};
    cuLaunchKernel(r->attn_prefill_f32,
                   (unsigned)n_heads, 1, 1,
                   32, 1, 1,  /* 1 warp */
                   0, r->stream, args, NULL);
}


/* ---- Init ---- */

cuda_qimg_runner *cuda_qimg_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_qimg: cuewInit failed\n");
        return NULL;
    }
    CU_CHECK_NULL(cuInit(0));

    CUdevice dev;
    CU_CHECK_NULL(cuDeviceGet(&dev, device_id));

    int sm_major, sm_minor;
    cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    int sm = sm_major * 10 + sm_minor;

    CUcontext ctx;
    CU_CHECK_NULL(cuCtxCreate(&ctx, 0, dev));
    CUstream stream;
    CU_CHECK_NULL(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    if (verbose) {
        char name[256]; cuDeviceGetName(name, sizeof(name), dev);
        size_t mem; cuDeviceTotalMem(&mem, dev);
        fprintf(stderr, "cuda_qimg: %s (sm_%d, %.1f GB)\n", name, sm, (float)mem/(1<<30));
    }

    /* Compile kernels */
    size_t len1 = strlen(cuda_kernels_common_src);
    size_t len2 = strlen(qimg_kernel_src);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, cuda_kernels_common_src, len1);
    memcpy(full_src + len1, qimg_kernel_src, len2);
    full_src[len1 + len2] = '\0';

    CUmodule module;
    int rc = cu_compile_kernels(&module, dev, full_src, "qimg.cu", verbose, "cuda_qimg");
    free(full_src);
    if (rc < 0) return NULL;

    cuda_qimg_runner *r = (cuda_qimg_runner *)calloc(1, sizeof(*r));
    r->device = dev; r->ctx = ctx; r->stream = stream;
    r->module = module; r->sm_version = sm; r->verbose = verbose;

    #define GET(field, name) cuModuleGetFunction(&r->field, module, name)
    GET(gemm_f16_f32, "gemm_f16_f32");
    GET(layernorm_f32, "layernorm_f32");
    GET(gelu_f32, "gelu_f32");
    GET(silu_f32, "silu_f32");
    GET(attn_prefill_f32, "simple_attn_f32");
    GET(add_bias_f32, "add_bias_f32");
    GET(rmsnorm_per_head, "rmsnorm_per_head_f32");
    GET(adaln_modulate, "adaln_modulate_f32");
    GET(gated_add, "gated_add_f32");
    GET(patchify, "patchify_f32");
    GET(unpatchify, "unpatchify_f32");
    GET(euler_step, "euler_step_f32");
    #undef GET

    if (verbose) fprintf(stderr, "cuda_qimg: kernels compiled OK\n");
    return r;
}


/* ---- Load DiT from FP8 safetensors ---- */

int cuda_qimg_load_dit(cuda_qimg_runner *r, const char *path) {
    fprintf(stderr, "cuda_qimg: loading DiT %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->dit_st = st;

    r->dit_dim = 3072; r->dit_n_heads = 24; r->dit_head_dim = 128;
    r->dit_in_ch = 64; r->dit_txt_dim = 3584; r->dit_mlp_h = 12288;

    /* Count blocks */
    r->dit_n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *bp = strstr(nm, "transformer_blocks.");
        if (bp) {
            int blk = atoi(bp + 19);
            if (blk + 1 > r->dit_n_blocks) r->dit_n_blocks = blk + 1;
        }
    }

    /* Upload global weights (permanent) */
    r->d_img_in_w = qimg_st_upload_f16(st, "img_in.weight");
    r->d_img_in_b = qimg_st_upload_f32(st, "img_in.bias");
    r->d_txt_in_w = qimg_st_upload_f16(st, "txt_in.weight");
    r->d_txt_in_b = qimg_st_upload_f32(st, "txt_in.bias");
    r->d_txt_norm_w = qimg_st_upload_f32(st, "txt_norm.weight");
    r->d_t_fc1_w = qimg_st_upload_f16(st, "time_text_embed.timestep_embedder.linear_1.weight");
    r->d_t_fc1_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_1.bias");
    r->d_t_fc2_w = qimg_st_upload_f16(st, "time_text_embed.timestep_embedder.linear_2.weight");
    r->d_t_fc2_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_2.bias");
    r->d_norm_out_w = qimg_st_upload_f16(st, "norm_out.linear.weight");
    r->d_norm_out_b = qimg_st_upload_f32(st, "norm_out.linear.bias");
    r->d_proj_out_w = qimg_st_upload_f16(st, "proj_out.weight");
    r->d_proj_out_b = qimg_st_upload_f32(st, "proj_out.bias");

    fprintf(stderr, "cuda_qimg: loaded %d blocks, dim=%d, global weights on GPU\n",
            r->dit_n_blocks, r->dit_dim);
    return 0;
}

int cuda_qimg_load_vae(cuda_qimg_runner *r, const char *path) {
    fprintf(stderr, "cuda_qimg: loading VAE %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->vae_st = st;
    fprintf(stderr, "cuda_qimg: VAE loaded (%d tensors)\n", st->n_tensors);
    return 0;
}

void cuda_qimg_free(cuda_qimg_runner *r) {
    if (!r) return;
    CUdeviceptr *globals = &r->d_img_in_w;
    for (int i = 0; i < 13; i++) { if (globals[i]) cuMemFree(globals[i]); }
    if (r->dit_st) safetensors_close((st_context *)r->dit_st);
    if (r->vae_st) safetensors_close((st_context *)r->vae_st);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->ctx) cuCtxDestroy(r->ctx);
    free(r);
}


/* ---- DiT single step (full 60-block forward on GPU) ---- */

int cuda_qimg_dit_step(cuda_qimg_runner *r,
                       const float *img_tokens, int n_img,
                       const float *txt_tokens, int n_txt,
                       float timestep, float *out) {
    int dim = r->dit_dim;
    int nh = r->dit_n_heads, hd = r->dit_head_dim;
    int in_ch = r->dit_in_ch, txt_dim = r->dit_txt_dim, mlp_h = r->dit_mlp_h;
    int n_total = n_img + n_txt;
    CUstream s = r->stream;

    /* Allocate GPU activation buffers */
    CUdeviceptr d_img, d_txt, d_t_emb;
    cuMemAlloc(&d_img, (size_t)n_img * dim * sizeof(float));
    cuMemAlloc(&d_txt, (size_t)n_txt * dim * sizeof(float));
    cuMemAlloc(&d_t_emb, (size_t)dim * sizeof(float));

    /* Upload inputs */
    CUdeviceptr d_img_in, d_txt_in;
    cuMemAlloc(&d_img_in, (size_t)n_img * in_ch * sizeof(float));
    cuMemcpyHtoD(d_img_in, img_tokens, (size_t)n_img * in_ch * sizeof(float));
    cuMemAlloc(&d_txt_in, (size_t)n_txt * txt_dim * sizeof(float));
    cuMemcpyHtoD(d_txt_in, txt_tokens, (size_t)n_txt * txt_dim * sizeof(float));

    /* 1. Timestep embedding: sinusoidal(256) → SiLU(GEMM) → GEMM */
    float t_sin[256];
    int half = 128;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i / (float)half * logf(10000.0f));
        float angle = timestep * freq;
        t_sin[i] = sinf(angle);
        t_sin[half + i] = cosf(angle);
    }
    CUdeviceptr d_t_sin;
    cuMemAlloc(&d_t_sin, 256 * sizeof(float));
    cuMemcpyHtoD(d_t_sin, t_sin, 256 * sizeof(float));

    op_gemm(r, d_t_emb, r->d_t_fc1_w, d_t_sin, r->d_t_fc1_b, dim, 256, 1);
    op_silu(r, d_t_emb, dim);
    CUdeviceptr d_t_emb2;
    cuMemAlloc(&d_t_emb2, (size_t)dim * sizeof(float));
    op_gemm(r, d_t_emb2, r->d_t_fc2_w, d_t_emb, r->d_t_fc2_b, dim, dim, 1);
    cuMemFree(d_t_emb); d_t_emb = d_t_emb2;
    cuMemFree(d_t_sin);

    /* 2. Text input: RMSNorm → GEMM (TODO: RMSNorm on GPU, for now use layernorm) */
    /* Simplification: upload pre-normed text from CPU for now */
    /* Actually, let's do the RMSNorm + projection on GPU */
    /* For now: just project (skip txt_norm — text encoder already applied RMSNorm) */
    op_gemm(r, d_txt, r->d_txt_in_w, d_txt_in, r->d_txt_in_b, dim, txt_dim, n_txt);
    cuMemFree(d_txt_in);

    /* 3. Image input: GEMM(64→3072) */
    op_gemm(r, d_img, r->d_img_in_w, d_img_in, r->d_img_in_b, dim, in_ch, n_img);
    cuMemFree(d_img_in);

    /* Scratch buffers */
    CUdeviceptr d_scratch1, d_scratch2, d_scratch3;
    size_t max_scratch = (size_t)n_total * dim * sizeof(float);
    cuMemAlloc(&d_scratch1, max_scratch);
    cuMemAlloc(&d_scratch2, max_scratch);
    size_t ffn_scratch = (size_t)(n_img > n_txt ? n_img : n_txt) * mlp_h * sizeof(float);
    cuMemAlloc(&d_scratch3, ffn_scratch);

    /* Modulation scratch */
    CUdeviceptr d_mod;
    cuMemAlloc(&d_mod, (size_t)6 * dim * sizeof(float));

    /* Joint Q/K/V buffers */
    CUdeviceptr d_q, d_k, d_v, d_attn_out;
    cuMemAlloc(&d_q, (size_t)n_total * dim * sizeof(float));
    cuMemAlloc(&d_k, (size_t)n_total * dim * sizeof(float));
    cuMemAlloc(&d_v, (size_t)n_total * dim * sizeof(float));
    cuMemAlloc(&d_attn_out, (size_t)n_total * dim * sizeof(float));

    /* 4. Process all 60 blocks */
    for (int L = 0; L < r->dit_n_blocks; L++) {
        if (r->verbose && (L % 10 == 0 || L == r->dit_n_blocks - 1))
            fprintf(stderr, "\r  cuda_qimg: block %d/%d", L + 1, r->dit_n_blocks);

        /* Load block weights to GPU */
        qimg_block_gpu blk;
        memset(&blk, 0, sizeof(blk));
        qimg_load_block(r, L, &blk);

        /* -- Image modulation: SiLU(t_emb) → Linear → 6×dim -- */
        CUdeviceptr d_t_silu;
        cuMemAlloc(&d_t_silu, (size_t)dim * sizeof(float));
        cuMemcpyDtoD(d_t_silu, d_t_emb, (size_t)dim * sizeof(float));
        op_silu(r, d_t_silu, dim);

        /* Image mod */
        CUdeviceptr d_img_mod;
        cuMemAlloc(&d_img_mod, (size_t)6 * dim * sizeof(float));
        op_gemm(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b,
                6 * dim, dim, 1);

        /* Text mod */
        CUdeviceptr d_txt_mod;
        cuMemAlloc(&d_txt_mod, (size_t)6 * dim * sizeof(float));
        op_gemm(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b,
                6 * dim, dim, 1);
        cuMemFree(d_t_silu);

        /* -- adaLN + QKV projections -- */
        /* img_shift1 = d_img_mod[0..dim-1], img_scale1 = [dim..2*dim-1], etc. */
        CUdeviceptr img_sh1 = d_img_mod;
        CUdeviceptr img_sc1 = d_img_mod + (size_t)dim * sizeof(float);
        CUdeviceptr img_g1  = d_img_mod + (size_t)2 * dim * sizeof(float);
        CUdeviceptr img_sh2 = d_img_mod + (size_t)3 * dim * sizeof(float);
        CUdeviceptr img_sc2 = d_img_mod + (size_t)4 * dim * sizeof(float);
        CUdeviceptr img_g2  = d_img_mod + (size_t)5 * dim * sizeof(float);

        CUdeviceptr txt_sh1 = d_txt_mod;
        CUdeviceptr txt_sc1 = d_txt_mod + (size_t)dim * sizeof(float);
        CUdeviceptr txt_g1  = d_txt_mod + (size_t)2 * dim * sizeof(float);
        CUdeviceptr txt_sh2 = d_txt_mod + (size_t)3 * dim * sizeof(float);
        CUdeviceptr txt_sc2 = d_txt_mod + (size_t)4 * dim * sizeof(float);
        CUdeviceptr txt_g2  = d_txt_mod + (size_t)5 * dim * sizeof(float);

        /* adaLN image → d_scratch1 */
        op_adaln(r, d_scratch1, d_img, img_sh1, img_sc1, n_img, dim);
        /* adaLN text → d_scratch2 */
        op_adaln(r, d_scratch2, d_txt, txt_sh1, txt_sc1, n_txt, dim);

        /* Image QKV → offset into joint buffers at [n_txt:] */
        CUdeviceptr d_img_q = d_q + (size_t)n_txt * dim * sizeof(float);
        CUdeviceptr d_img_k = d_k + (size_t)n_txt * dim * sizeof(float);
        CUdeviceptr d_img_v = d_v + (size_t)n_txt * dim * sizeof(float);
        op_gemm(r, d_img_q, blk.attn_q_w, d_scratch1, blk.attn_q_b, dim, dim, n_img);
        op_gemm(r, d_img_k, blk.attn_k_w, d_scratch1, blk.attn_k_b, dim, dim, n_img);
        op_gemm(r, d_img_v, blk.attn_v_w, d_scratch1, blk.attn_v_b, dim, dim, n_img);

        /* Text QKV → offset at [0:n_txt] */
        CUdeviceptr d_txt_q = d_q;
        CUdeviceptr d_txt_k = d_k;
        CUdeviceptr d_txt_v = d_v;
        op_gemm(r, d_txt_q, blk.attn_add_q_w, d_scratch2, blk.attn_add_q_b, dim, dim, n_txt);
        op_gemm(r, d_txt_k, blk.attn_add_k_w, d_scratch2, blk.attn_add_k_b, dim, dim, n_txt);
        op_gemm(r, d_txt_v, blk.attn_add_v_w, d_scratch2, blk.attn_add_v_b, dim, dim, n_txt);

        /* QK RMSNorm (per-head) */
        op_rmsnorm_ph(r, d_img_q, blk.norm_q_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_img_k, blk.norm_k_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_txt_q, blk.norm_added_q_w, n_txt, nh, hd);
        op_rmsnorm_ph(r, d_txt_k, blk.norm_added_k_w, n_txt, nh, hd);

        /* TODO: RoPE (skip for now — will add later for accuracy) */

        /* Joint attention: Q/K/V already concatenated as [txt, img] */
        op_attn(r, d_attn_out, d_q, d_k, d_v, n_total, nh, hd);

        /* Output projections */
        CUdeviceptr d_img_attn = d_attn_out + (size_t)n_txt * dim * sizeof(float);
        CUdeviceptr d_txt_attn = d_attn_out;
        op_gemm(r, d_scratch1, blk.attn_out_w, d_img_attn, blk.attn_out_b, dim, dim, n_img);
        op_gemm(r, d_scratch2, blk.attn_add_out_w, d_txt_attn, blk.attn_add_out_b, dim, dim, n_txt);

        /* Gated residual */
        op_gated_add(r, d_img, d_scratch1, img_g1, n_img, dim);
        op_gated_add(r, d_txt, d_scratch2, txt_g1, n_txt, dim);

        /* -- MLP with adaLN -- */
        /* Image MLP */
        op_adaln(r, d_scratch1, d_img, img_sh2, img_sc2, n_img, dim);
        op_gemm(r, d_scratch3, blk.img_mlp_fc1_w, d_scratch1, blk.img_mlp_fc1_b,
                mlp_h, dim, n_img);
        op_gelu(r, d_scratch3, n_img * mlp_h);
        op_gemm(r, d_scratch1, blk.img_mlp_fc2_w, d_scratch3, blk.img_mlp_fc2_b,
                dim, mlp_h, n_img);
        op_gated_add(r, d_img, d_scratch1, img_g2, n_img, dim);

        /* Text MLP */
        op_adaln(r, d_scratch2, d_txt, txt_sh2, txt_sc2, n_txt, dim);
        op_gemm(r, d_scratch3, blk.txt_mlp_fc1_w, d_scratch2, blk.txt_mlp_fc1_b,
                mlp_h, dim, n_txt);
        op_gelu(r, d_scratch3, n_txt * mlp_h);
        op_gemm(r, d_scratch2, blk.txt_mlp_fc2_w, d_scratch3, blk.txt_mlp_fc2_b,
                dim, mlp_h, n_txt);
        op_gated_add(r, d_txt, d_scratch2, txt_g2, n_txt, dim);

        /* Free block weights */
        cuMemFree(d_img_mod); cuMemFree(d_txt_mod);
        qimg_free_block(&blk);

        cuStreamSynchronize(s);
    }
    if (r->verbose) fprintf(stderr, "\n");

    /* 5. Final output: adaLN → proj_out */
    {
        CUdeviceptr d_t_silu;
        cuMemAlloc(&d_t_silu, (size_t)dim * sizeof(float));
        cuMemcpyDtoD(d_t_silu, d_t_emb, (size_t)dim * sizeof(float));
        op_silu(r, d_t_silu, dim);
        CUdeviceptr d_final_mod;
        cuMemAlloc(&d_final_mod, (size_t)2 * dim * sizeof(float));
        op_gemm(r, d_final_mod, r->d_norm_out_w, d_t_silu, r->d_norm_out_b,
                2 * dim, dim, 1);
        cuMemFree(d_t_silu);

        CUdeviceptr f_shift = d_final_mod;
        CUdeviceptr f_scale = d_final_mod + (size_t)dim * sizeof(float);
        op_adaln(r, d_scratch1, d_img, f_shift, f_scale, n_img, dim);
        cuMemFree(d_final_mod);

        /* proj_out: [n_img, dim] → [n_img, in_ch] */
        CUdeviceptr d_out;
        cuMemAlloc(&d_out, (size_t)n_img * in_ch * sizeof(float));
        op_gemm(r, d_out, r->d_proj_out_w, d_scratch1, r->d_proj_out_b,
                in_ch, dim, n_img);

        /* Download result */
        cuMemcpyDtoH(out, d_out, (size_t)n_img * in_ch * sizeof(float));
        cuMemFree(d_out);
    }

    /* Cleanup */
    cuMemFree(d_img); cuMemFree(d_txt); cuMemFree(d_t_emb);
    cuMemFree(d_scratch1); cuMemFree(d_scratch2); cuMemFree(d_scratch3);
    cuMemFree(d_mod);
    cuMemFree(d_q); cuMemFree(d_k); cuMemFree(d_v); cuMemFree(d_attn_out);

    return 0;
}

#endif /* CUDA_QIMG_RUNNER_IMPLEMENTATION */
#endif /* CUDA_QIMG_RUNNER_H */
