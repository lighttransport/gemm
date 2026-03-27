/*
 * cuda_qimg_runner.h - CUDA Qwen-Image text-to-image runner
 *
 * Uses NVRTC to compile kernels at runtime. Processes the 60-block MMDiT
 * by loading one block at a time (dequant Q4_0→F16 on CPU, upload, compute,
 * free) to fit within 16GB VRAM.
 *
 * Pipeline:
 *   1. Text encoder (CPU) → hidden states [N_txt, 3584]
 *   2. MMDiT denoising (GPU) × N_steps → latent [16, H/8, W/8]
 *   3. VAE decoder (GPU) → RGB image [3, H, W]
 *
 * Usage:
 *   cuda_qimg_runner *r = cuda_qimg_init(0, 1);
 *   cuda_qimg_load_dit(r, "dit.gguf");
 *   cuda_qimg_load_vae(r, "vae.safetensors");
 *   float *rgb = cuda_qimg_generate(r, txt_hidden, n_txt, h, w, n_steps, seed);
 *   cuda_qimg_free(r);
 *
 * Requires: cuda.h, nvrtc.h (via cuew.h), gguf_loader.h, ggml_dequant.h,
 *           safetensors.h, cuda_kernels_common.h
 */
#ifndef CUDA_QIMG_RUNNER_H
#define CUDA_QIMG_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_qimg_runner cuda_qimg_runner;

cuda_qimg_runner *cuda_qimg_init(int device_id, int verbose);
int  cuda_qimg_load_dit(cuda_qimg_runner *r, const char *gguf_path);
int  cuda_qimg_load_vae(cuda_qimg_runner *r, const char *safetensors_path);
void cuda_qimg_free(cuda_qimg_runner *r);

/* Generate image from text hidden states.
 *   txt_hidden: [n_txt, 3584] F32 from text encoder (CPU memory)
 *   out_rgb:    [3, out_h, out_w] F32 (pre-allocated, CPU memory)
 *   Returns 0 on success. */
int cuda_qimg_generate(cuda_qimg_runner *r,
                       const float *txt_hidden, int n_txt,
                       int out_h, int out_w,
                       int n_steps, uint32_t seed,
                       float *out_rgb);

/* Run single DiT step for verification.
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
#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include "../../common/safetensors.h"
#include "../../common/qwen_image_scheduler.h"

/* ---- Kernel source for Qwen-Image specific ops ---- */

/* NOTE: qimg_kernel_src is appended AFTER cuda_kernels_common_src
 * which already has extern "C" { open. We just add more kernels
 * and close with } at the end. */
static const char *qimg_kernel_src =
"\n/* ---- Qwen-Image specific kernels ---- */\n"

/* Q4_0 dequantize kernel: [n_blocks, 18 bytes] → [n_elements] F32 */
"__global__ void dequant_q4_0_f32(const unsigned char *__restrict__ src,\n"
"    float *__restrict__ dst, int n_blocks) {\n"
"    int bid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (bid >= n_blocks) return;\n"
"    const unsigned char *blk = src + bid * 18;\n"
"    unsigned short d_bits = blk[0] | ((unsigned short)blk[1] << 8);\n"
"    float d;\n"
"    asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(d) : \"h\"(d_bits));\n"
"    for (int j = 0; j < 16; j++) {\n"
"        int byte = blk[2 + j];\n"
"        int lo = (byte & 0xF) - 8;\n"
"        int hi = (byte >> 4) - 8;\n"
"        dst[bid * 32 + j]      = (float)lo * d;\n"
"        dst[bid * 32 + j + 16] = (float)hi * d;\n"
"    }\n"
"}\n"

/* GroupNorm kernel for VAE */
"__global__ void groupnorm_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, const float *__restrict__ gamma,\n"
"    int C, int spatial, int groups) {\n"
"    int g = blockIdx.x;\n"
"    if (g >= groups) return;\n"
"    int cpg = C / groups;\n"
"    int n = cpg * spatial;\n"
"    float mean = 0, var = 0;\n"
"    for (int gc = 0; gc < cpg; gc++) {\n"
"        int ch = g * cpg + gc;\n"
"        for (int s = threadIdx.x; s < spatial; s += blockDim.x) {\n"
"            float v = inp[ch * spatial + s];\n"
"            mean += v;\n"
"            var += v * v;\n"
"        }\n"
"    }\n"
"    /* Warp reduce */\n"
"    for (int m = 16; m > 0; m >>= 1) {\n"
"        mean += __shfl_xor_sync(0xFFFFFFFF, mean, m);\n"
"        var += __shfl_xor_sync(0xFFFFFFFF, var, m);\n"
"    }\n"
"    mean /= (float)n;\n"
"    var = var / (float)n - mean * mean;\n"
"    float inv = rsqrtf(var + 1e-6f);\n"
"    for (int gc = 0; gc < cpg; gc++) {\n"
"        int ch = g * cpg + gc;\n"
"        float g_val = gamma ? gamma[ch] : 1.0f;\n"
"        for (int s = threadIdx.x; s < spatial; s += blockDim.x)\n"
"            out[ch * spatial + s] = (inp[ch * spatial + s] - mean) * inv * g_val;\n"
"    }\n"
"}\n"

/* Conv2D with replicate padding */
"__global__ void conv2d_replicate_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, const float *__restrict__ weight,\n"
"    const float *__restrict__ bias,\n"
"    int ci, int h, int w, int co, int kh, int kw) {\n"
"    int oc = blockIdx.x;\n"
"    int oy = blockIdx.y;\n"
"    int ox = threadIdx.x;\n"
"    if (oc >= co || oy >= h || ox >= w) return;\n"
"    int ph = (kh - 1) / 2, pw = (kw - 1) / 2;\n"
"    float sum = bias ? bias[oc] : 0.0f;\n"
"    for (int ic = 0; ic < ci; ic++)\n"
"        for (int fy = 0; fy < kh; fy++)\n"
"            for (int fx = 0; fx < kw; fx++) {\n"
"                int iy = oy + fy - ph, ix = ox + fx - pw;\n"
"                if (iy < 0) iy = 0; if (iy >= h) iy = h - 1;\n"
"                if (ix < 0) ix = 0; if (ix >= w) ix = w - 1;\n"
"                sum += inp[ic * h * w + iy * w + ix] *\n"
"                       weight[((oc * ci + ic) * kh + fy) * kw + fx];\n"
"            }\n"
"    out[oc * h * w + oy * w + ox] = sum;\n"
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

/* NN upsample 2×: [C, H, W] → [C, 2H, 2W] */
"__global__ void nn_upsample2x_f32(float *__restrict__ out,\n"
"    const float *__restrict__ inp, int C, int H, int W) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int oh = H * 2, ow = W * 2;\n"
"    int total = C * oh * ow;\n"
"    if (idx >= total) return;\n"
"    int c = idx / (oh * ow);\n"
"    int rem = idx % (oh * ow);\n"
"    int y = rem / ow, x = rem % ow;\n"
"    out[idx] = inp[c * H * W + (y/2) * W + (x/2)];\n"
"}\n"

/* Euler step: x = x + dt * v */
"__global__ void euler_step_f32(float *__restrict__ x,\n"
"    const float *__restrict__ v, float dt, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) x[i] += dt * v[i];\n"
"}\n"

"}\n"; /* end extern "C" */


/* ---- Runner struct ---- */

struct cuda_qimg_runner {
    CUdevice device;
    CUcontext ctx;
    CUstream stream;
    CUmodule module_common;  /* from cuda_kernels_common.h */
    CUmodule module_qimg;    /* our custom kernels */
    int sm_version;
    int verbose;

    /* Common kernels */
    CUfunction gemm_f16_f32;
    CUfunction layernorm_f32;
    CUfunction gelu_f32;
    CUfunction silu_f32;
    CUfunction attn_prefill_f32;
    CUfunction add_f32;
    CUfunction add_bias_f32;

    /* Qwen-Image specific kernels */
    CUfunction dequant_q4_0;
    CUfunction groupnorm;
    CUfunction conv2d_replicate;
    CUfunction patchify;
    CUfunction unpatchify;
    CUfunction nn_upsample2x;
    CUfunction euler_step;

    /* DiT config */
    int dit_dim;       /* 3072 */
    int dit_n_heads;   /* 24 */
    int dit_head_dim;  /* 128 */
    int dit_n_blocks;  /* 60 */
    int dit_in_ch;     /* 64 */
    int dit_txt_dim;   /* 3584 */
    int dit_mlp_h;     /* 12288 */

    /* GGUF context (kept for mmap weight access) */
    void *dit_gguf;

    /* Persistent GPU buffers for global weights (small, kept on GPU) */
    CUdeviceptr d_img_in_w, d_img_in_b;     /* [64, 3072] F16 + bias F32 */
    CUdeviceptr d_txt_in_w, d_txt_in_b;     /* [3584, 3072] F16 + bias F32 */
    CUdeviceptr d_txt_norm_w;                /* [3584] F32 */
    CUdeviceptr d_t_fc1_w, d_t_fc1_b;       /* timestep embedder */
    CUdeviceptr d_t_fc2_w, d_t_fc2_b;
    CUdeviceptr d_norm_out_w, d_norm_out_b;  /* final norm */
    CUdeviceptr d_proj_out_w, d_proj_out_b;  /* final projection */

    /* VAE safetensors context */
    void *vae_st;
};


/* ---- Implementation ---- */

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
        char name[256];
        cuDeviceGetName(name, sizeof(name), dev);
        size_t mem;
        cuDeviceTotalMem(&mem, dev);
        fprintf(stderr, "cuda_qimg: %s (sm_%d, %.1f GB)\n", name, sm, (float)mem/(1<<30));
    }

    /* Concatenate common + qimg kernel sources and compile once */
    CUmodule mod_common, mod_qimg;
    {
        size_t len1 = strlen(cuda_kernels_common_src);
        size_t len2 = strlen(qimg_kernel_src);
        char *full_src = (char *)malloc(len1 + len2 + 1);
        memcpy(full_src, cuda_kernels_common_src, len1);
        memcpy(full_src + len1, qimg_kernel_src, len2);
        full_src[len1 + len2] = '\0';
        int rc = cu_compile_kernels(&mod_common, dev, full_src,
                                     "qimg_kernels.cu", verbose, "cuda_qimg");
        free(full_src);
        if (rc < 0) {
            fprintf(stderr, "cuda_qimg: kernel compilation failed\n");
            return NULL;
        }
        mod_qimg = mod_common;  /* same module */
    }

    cuda_qimg_runner *r = (cuda_qimg_runner *)calloc(1, sizeof(*r));
    r->device = dev;
    r->ctx = ctx;
    r->stream = stream;
    r->module_common = mod_common;
    r->module_qimg = mod_qimg;
    r->sm_version = sm;
    r->verbose = verbose;

    /* Get kernel handles */
    #define GET_COMMON(field, name) cuModuleGetFunction(&r->field, mod_common, name)
    #define GET_QIMG(field, name) cuModuleGetFunction(&r->field, mod_qimg, name)

    GET_COMMON(gemm_f16_f32, "gemm_f16_f32");
    GET_COMMON(layernorm_f32, "layernorm_f32");
    GET_COMMON(gelu_f32, "gelu_f32");
    GET_COMMON(silu_f32, "silu_f32");
    GET_COMMON(attn_prefill_f32, "attn_prefill_f32");
    GET_COMMON(add_f32, "add_f32");
    GET_COMMON(add_bias_f32, "add_bias_f32");

    GET_QIMG(dequant_q4_0, "dequant_q4_0_f32");
    GET_QIMG(groupnorm, "groupnorm_f32");
    GET_QIMG(conv2d_replicate, "conv2d_replicate_f32");
    GET_QIMG(patchify, "patchify_f32");
    GET_QIMG(unpatchify, "unpatchify_f32");
    GET_QIMG(nn_upsample2x, "nn_upsample2x_f32");
    GET_QIMG(euler_step, "euler_step_f32");

    #undef GET_COMMON
    #undef GET_QIMG

    if (verbose) fprintf(stderr, "cuda_qimg: kernels compiled\n");
    return r;
}

/* Upload a weight tensor: dequant Q4_0/Q4_1/BF16/F32 → F16 on GPU.
 * Returns GPU pointer. Caller must cuMemFree. */
static CUdeviceptr qimg_upload_weight(cuda_qimg_runner *r,
                                       const void *data, uint32_t type,
                                       int n_rows, int n_cols) {
    int n = n_rows * n_cols;
    float *f32 = (float *)malloc((size_t)n * sizeof(float));

    /* Dequantize to F32 row by row */
    int block_size, type_size;
    switch (type) {
        case GGML_TYPE_Q4_0:    block_size = 32;  type_size = 18; break;
        case GGML_TYPE_Q4_1:    block_size = 32;  type_size = 20; break;
        case GGML_TYPE_Q8_0:    block_size = 32;  type_size = 34; break;
        case GGML_TYPE_F32:     block_size = 1;   type_size = 4;  break;
        case GGML_TYPE_F16:     block_size = 1;   type_size = 2;  break;
        case GGML_TYPE_BF16:    block_size = 1;   type_size = 2;  break;
        case GGML_TYPE_Q4_0_8_8: block_size = 32; type_size = 18; break;
        default: block_size = 1; type_size = 4; break;
    }
    size_t row_bytes = (size_t)((n_cols + block_size - 1) / block_size) * type_size;
    for (int row = 0; row < n_rows; row++) {
        const void *row_data = (const uint8_t *)data + row * row_bytes;
        dequant_row(type, row_data, f32 + row * n_cols, n_cols);
    }

    /* Convert F32 → F16 */
    uint16_t *f16 = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    for (int i = 0; i < n; i++)
        f16[i] = cu_f32_to_f16(f32[i]);
    free(f32);

    CUdeviceptr d;
    cuMemAlloc(&d, (size_t)n * sizeof(uint16_t));
    cuMemcpyHtoD(d, f16, (size_t)n * sizeof(uint16_t));
    free(f16);
    return d;
}

/* Upload bias (F32) to GPU */
static CUdeviceptr qimg_upload_bias(const void *data, uint32_t type, int n) {
    float *f32 = (float *)malloc((size_t)n * sizeof(float));
    dequant_row(type, data, f32, n);
    CUdeviceptr d;
    cuMemAlloc(&d, (size_t)n * sizeof(float));
    cuMemcpyHtoD(d, f32, (size_t)n * sizeof(float));
    free(f32);
    return d;
}

/* Find and upload a GGUF tensor */
static CUdeviceptr qimg_load_gguf_weight(cuda_qimg_runner *r,
                                          const char *name, int *out_rows, int *out_cols) {
    gguf_context *gguf = (gguf_context *)r->dit_gguf;
    int idx = -1;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        if (strcmp(gguf->tensors[i].name.str, name) == 0) { idx = (int)i; break; }
    }
    if (idx < 0) {
        fprintf(stderr, "cuda_qimg: missing tensor '%s'\n", name);
        return 0;
    }
    void *data = gguf_tensor_data(gguf, idx);
    int n_cols = (int)gguf->tensors[idx].dims[0];
    int n_rows = 1;
    for (uint32_t d = 1; d < gguf->tensors[idx].n_dims; d++)
        n_rows *= (int)gguf->tensors[idx].dims[d];
    if (out_rows) *out_rows = n_rows;
    if (out_cols) *out_cols = n_cols;

    if (gguf->tensors[idx].n_dims == 1) {
        /* Bias: upload as F32 */
        return qimg_upload_bias(data, gguf->tensors[idx].type, n_cols);
    }
    return qimg_upload_weight(r, data, gguf->tensors[idx].type, n_rows, n_cols);
}

int cuda_qimg_load_dit(cuda_qimg_runner *r, const char *gguf_path) {
    fprintf(stderr, "cuda_qimg: loading DiT %s\n", gguf_path);
    gguf_context *gguf = gguf_open(gguf_path, 1);
    if (!gguf) return -1;
    r->dit_gguf = gguf;

    /* Detect config */
    r->dit_dim = 3072; r->dit_n_heads = 24; r->dit_head_dim = 128;
    r->dit_in_ch = 64; r->dit_txt_dim = 3584; r->dit_mlp_h = 12288;

    /* Count blocks */
    r->dit_n_blocks = 0;
    for (uint64_t i = 0; i < gguf->n_tensors; i++) {
        const char *nm = gguf->tensors[i].name.str;
        const char *bp = strstr(nm, "transformer_blocks.");
        if (bp) {
            int blk = atoi(bp + 19);
            if (blk + 1 > r->dit_n_blocks) r->dit_n_blocks = blk + 1;
        }
    }
    fprintf(stderr, "cuda_qimg: %d blocks, dim=%d\n", r->dit_n_blocks, r->dit_dim);

    /* Upload global weights (small, kept on GPU permanently) */
    r->d_img_in_w = qimg_load_gguf_weight(r, "img_in.weight", NULL, NULL);
    r->d_img_in_b = qimg_load_gguf_weight(r, "img_in.bias", NULL, NULL);
    r->d_txt_in_w = qimg_load_gguf_weight(r, "txt_in.weight", NULL, NULL);
    r->d_txt_in_b = qimg_load_gguf_weight(r, "txt_in.bias", NULL, NULL);
    r->d_txt_norm_w = qimg_load_gguf_weight(r, "txt_norm.weight", NULL, NULL);
    r->d_t_fc1_w = qimg_load_gguf_weight(r, "time_text_embed.timestep_embedder.linear_1.weight", NULL, NULL);
    r->d_t_fc1_b = qimg_load_gguf_weight(r, "time_text_embed.timestep_embedder.linear_1.bias", NULL, NULL);
    r->d_t_fc2_w = qimg_load_gguf_weight(r, "time_text_embed.timestep_embedder.linear_2.weight", NULL, NULL);
    r->d_t_fc2_b = qimg_load_gguf_weight(r, "time_text_embed.timestep_embedder.linear_2.bias", NULL, NULL);
    r->d_norm_out_w = qimg_load_gguf_weight(r, "norm_out.linear.weight", NULL, NULL);
    r->d_norm_out_b = qimg_load_gguf_weight(r, "norm_out.linear.bias", NULL, NULL);
    r->d_proj_out_w = qimg_load_gguf_weight(r, "proj_out.weight", NULL, NULL);
    r->d_proj_out_b = qimg_load_gguf_weight(r, "proj_out.bias", NULL, NULL);

    fprintf(stderr, "cuda_qimg: global weights uploaded\n");
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
    /* Free GPU buffers */
    if (r->d_img_in_w) cuMemFree(r->d_img_in_w);
    if (r->d_img_in_b) cuMemFree(r->d_img_in_b);
    if (r->d_txt_in_w) cuMemFree(r->d_txt_in_w);
    if (r->d_txt_in_b) cuMemFree(r->d_txt_in_b);
    if (r->d_txt_norm_w) cuMemFree(r->d_txt_norm_w);
    if (r->d_t_fc1_w) cuMemFree(r->d_t_fc1_w);
    if (r->d_t_fc1_b) cuMemFree(r->d_t_fc1_b);
    if (r->d_t_fc2_w) cuMemFree(r->d_t_fc2_w);
    if (r->d_t_fc2_b) cuMemFree(r->d_t_fc2_b);
    if (r->d_norm_out_w) cuMemFree(r->d_norm_out_w);
    if (r->d_norm_out_b) cuMemFree(r->d_norm_out_b);
    if (r->d_proj_out_w) cuMemFree(r->d_proj_out_w);
    if (r->d_proj_out_b) cuMemFree(r->d_proj_out_b);
    if (r->dit_gguf) gguf_close((gguf_context *)r->dit_gguf);
    if (r->vae_st) safetensors_close((st_context *)r->vae_st);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->ctx) cuCtxDestroy(r->ctx);
    free(r);
}

/* Placeholder — full DiT step and generate will be filled next */
int cuda_qimg_dit_step(cuda_qimg_runner *r,
                       const float *img_tokens, int n_img,
                       const float *txt_tokens, int n_txt,
                       float timestep, float *out) {
    (void)r; (void)img_tokens; (void)n_img;
    (void)txt_tokens; (void)n_txt; (void)timestep; (void)out;
    fprintf(stderr, "cuda_qimg_dit_step: not yet implemented\n");
    return -1;
}

int cuda_qimg_generate(cuda_qimg_runner *r,
                       const float *txt_hidden, int n_txt,
                       int out_h, int out_w,
                       int n_steps, uint32_t seed,
                       float *out_rgb) {
    (void)r; (void)txt_hidden; (void)n_txt;
    (void)out_h; (void)out_w; (void)n_steps; (void)seed; (void)out_rgb;
    fprintf(stderr, "cuda_qimg_generate: not yet implemented\n");
    return -1;
}

#endif /* CUDA_QIMG_RUNNER_IMPLEMENTATION */
#endif /* CUDA_QIMG_RUNNER_H */
