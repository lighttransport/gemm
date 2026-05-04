/*
 * cuda_hy3d_runner.c - CUDA Hunyuan3D-2.1 via NVRTC-compiled kernels
 *
 * Pipeline: DINOv2 encoder -> DiT diffusion (flow matching) -> ShapeVAE -> MC mesh
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 *
 * Architecture fixes applied:
 *   1. DiT U-Net skip connections (blocks 11-20)
 *   2. DiT MoE in last 6 blocks (15-20): 8 experts + shared expert, top-2 gating
 *   3. DiT prepends timestep token (seq_len 4096 -> 4097, stripped at final layer)
 *   4. TimestepEmbedder uses GELU (not SiLU)
 *   5. VAE include_pi=false (freqs = 2^i, not pi*2^i)
 *   6. VAE tensor name mapping (transformer.resblocks, c_qkv, c_proj, etc.)
 *   7. VAE geo_decoder tensor names (cross_attn_decoder prefix, c_q/c_kv, etc.)
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define MARCHING_CUBES_IMPLEMENTATION
#include "../../common/marching_cubes.h"
#include "cuda_hy3d_runner.h"
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#include "../cuda_rng.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

/* Modular ops: kernel source strings + launch wrappers */
#include "cuda_hy3d_kernels.h"
/* Shared FP8/BF16 MMA kernel source (used by qimg/flux2). Appended in its
 * own extern "C" block below; provides per-row FP8 GEMM, BF16-pipe GEMM,
 * flash_attn_bf16/fp8, quantize_to_fp8_e4m3, reduce_max_abs_*. */
#include "../cuda_fp8_mma_kernels.h"
#include "cuda_hy3d_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* Env-var gated device-to-.npy debug dump. Set HY3D_DUMP_DIR=/path to enable;
 * individual stages write files like cuda_dinov2_hidden_0.npy into it. */
static void hy3d_dbg_dump_npy(CUstream stream, CUdeviceptr d_buf,
                              int rows, int cols, const char *fname) {
    const char *dir = getenv("HY3D_DUMP_DIR");
    if (!dir || !*dir) return;
    size_t n = (size_t)rows * (size_t)cols;
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) return;
    cuStreamSynchronize(stream);
    cuMemcpyDtoH(h, d_buf, n * sizeof(float));
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, fname);
    FILE *f = fopen(path, "wb");
    if (!f) { free(h); return; }
    static const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", rows, cols);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(h, sizeof(float), n, f);
    fclose(f);
    free(h);
}

static void hy3d_write_npy_f32_2d(const char *path, const float *buf, int rows, int cols) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    static const char magic[] = "\x93NUMPY";
    fwrite(magic, 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", rows, cols);
    int total = 10 + hlen + 1;
    int pad = ((total + 63) / 64) * 64 - total;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(buf, sizeof(float), (size_t)rows * (size_t)cols, f);
    fclose(f);
}

static float hy3d_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    union { uint32_t u; float f; } out;
    if (exp == 0) {
        if (mant == 0) {
            out.u = sign;
            return out.f;
        }
        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3FFu;
        exp = 1;
    } else if (exp == 0x1Fu) {
        out.u = sign | 0x7F800000u | (mant << 13);
        return out.f;
    }
    exp = exp + (127u - 15u);
    out.u = sign | (exp << 23) | (mant << 13);
    return out.f;
}

static void hy3d_make_dbg_name(char *dst, size_t dst_sz,
                               const char *base, const char *pass_tag) {
    if (pass_tag && *pass_tag) snprintf(dst, dst_sz, "%s_%s.npy", base, pass_tag);
    else snprintf(dst, dst_sz, "%s.npy", base);
}

static void hy3d_dbg_dump_tagged_npy(CUstream stream, CUdeviceptr d_buf,
                                     int rows, int cols,
                                     const char *base, const char *pass_tag) {
    char fname[256];
    hy3d_make_dbg_name(fname, sizeof(fname), base, pass_tag);
    hy3d_dbg_dump_npy(stream, d_buf, rows, cols, fname);
}

static void cpu_gemm_f32(float *Y, const float *W, const float *X,
                         int n_out, int n_in, int n_tok) {
    for (int t = 0; t < n_tok; t++) {
        const float *x = X + (size_t)t * n_in;
        float *y = Y + (size_t)t * n_out;
        for (int o = 0; o < n_out; o++) {
            const float *w = W + (size_t)o * n_in;
            float acc = 0.0f;
            for (int i = 0; i < n_in; i++) acc += w[i] * x[i];
            y[o] = acc;
        }
    }
}

static void cpu_dump_npy_f32_2d(const char *path, const float *buf, int rows, int cols) {
    hy3d_write_npy_f32_2d(path, buf, rows, cols);
}

static int hy3d_copy_gemm_weights_to_host_f32(float *dst, CUdeviceptr d_w,
                                              int n_out, int n_in, int use_f32_gemm) {
    size_t n = (size_t)n_out * (size_t)n_in;
    if (use_f32_gemm) {
        return cuMemcpyDtoH(dst, d_w, n * sizeof(float)) == CUDA_SUCCESS ? 0 : -1;
    }
    uint16_t *tmp = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!tmp) return -1;
    if (cuMemcpyDtoH(tmp, d_w, n * sizeof(uint16_t)) != CUDA_SUCCESS) {
        free(tmp);
        return -1;
    }
    for (size_t i = 0; i < n; i++) dst[i] = hy3d_f16_to_f32(tmp[i]);
    free(tmp);
    return 0;
}

static void hy3d_dump_latent(CUstream stream, CUdeviceptr d_latents,
                             int rows, int cols, const char *prefix, int step1) {
    if (!prefix || !*prefix) return;
    size_t n = (size_t)rows * (size_t)cols;
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) return;
    cuStreamSynchronize(stream);
    cuMemcpyDtoH(h, d_latents, n * sizeof(float));
    char path[1024];
    snprintf(path, sizeof(path), "%s_%03d.npy", prefix, step1);
    hy3d_write_npy_f32_2d(path, h, rows, cols);
    free(h);
}

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

#define HY3D_MAX_LATENT_DUMP_STEPS 64

/* ======================================================================== */
/* Internal structures                                                      */
/* ======================================================================== */

/* DINOv2 per-layer weights (on GPU, F16) */
typedef struct {
    CUdeviceptr ln1_w, ln1_b;      /* LayerNorm 1 */
    CUdeviceptr q_w, q_b;          /* Query projection */
    CUdeviceptr k_w, k_b;          /* Key projection */
    CUdeviceptr v_w, v_b;          /* Value projection */
    CUdeviceptr out_w, out_b;      /* Output projection */
    CUdeviceptr ls1;               /* LayerScale 1 */
    CUdeviceptr ln2_w, ln2_b;      /* LayerNorm 2 */
    CUdeviceptr fc1_w, fc1_b;      /* MLP FC1 */
    CUdeviceptr fc2_w, fc2_b;      /* MLP FC2 */
    CUdeviceptr ls2;               /* LayerScale 2 */
} dino_layer_gpu;

/* DiT per-block weights (on GPU, F16) */
typedef struct {
    /* Self-attention — fused QKV weight [3*dim, dim] for correct head interleaving */
    CUdeviceptr norm1_w, norm1_b;
    CUdeviceptr sa_qkv_w;              /* [3*2048, 2048] = concat(to_q.w, to_k.w, to_v.w) */
    CUdeviceptr sa_qkv_w_fp8, sa_qkv_w_scale;
    CUdeviceptr sa_out_w, sa_out_b;
    CUdeviceptr sa_out_w_fp8, sa_out_w_scale;
    CUdeviceptr sa_q_norm_w, sa_k_norm_w;  /* RMSNorm weight only */
    /* Cross-attention — Q/K/V separate to mirror the PyTorch reference
     * exactly before any view/split reshaping. */
    CUdeviceptr norm2_w, norm2_b;
    CUdeviceptr ca_q_w;                /* [2048, 2048] */
    CUdeviceptr ca_q_w_fp8, ca_q_w_scale;
    CUdeviceptr ca_k_w;                /* [2048, 1024] */
    CUdeviceptr ca_k_w_fp8, ca_k_w_scale;
    CUdeviceptr ca_v_w;                /* [2048, 1024] */
    CUdeviceptr ca_v_w_fp8, ca_v_w_scale;
    CUdeviceptr ca_out_w, ca_out_b;
    CUdeviceptr ca_out_w_fp8, ca_out_w_scale;
    CUdeviceptr ca_q_norm_w, ca_k_norm_w;
    /* Norm3 + MLP/MoE */
    CUdeviceptr norm3_w, norm3_b;
    /* Regular MLP (blocks 0-14) */
    CUdeviceptr mlp_fc1_w, mlp_fc1_b;
    CUdeviceptr mlp_fc1_w_fp8, mlp_fc1_w_scale;
    CUdeviceptr mlp_fc2_w, mlp_fc2_b;
    CUdeviceptr mlp_fc2_w_fp8, mlp_fc2_w_scale;
    /* MoE (blocks 15-20) */
    int use_moe;
    CUdeviceptr moe_gate_w;                                   /* [8, 2048] */
    CUdeviceptr moe_expert_fc1_w[DIT_N_EXPERTS];              /* [8192, 2048] each */
    CUdeviceptr moe_expert_fc1_b[DIT_N_EXPERTS];              /* [8192] each */
    CUdeviceptr moe_expert_fc2_w[DIT_N_EXPERTS];              /* [2048, 8192] each */
    CUdeviceptr moe_expert_fc2_b[DIT_N_EXPERTS];              /* [2048] each */
    /* Online-quantized FP8 mirrors of the expert weights + per-tensor scales.
     * Populated by quantize_dit_moe_to_fp8() at load time when ops.use_fp8_gemm
     * is enabled. NULL otherwise — dispatch falls through to the F16 path. */
    CUdeviceptr moe_expert_fc1_w_fp8[DIT_N_EXPERTS];
    CUdeviceptr moe_expert_fc1_w_scale[DIT_N_EXPERTS];
    CUdeviceptr moe_expert_fc2_w_fp8[DIT_N_EXPERTS];
    CUdeviceptr moe_expert_fc2_w_scale[DIT_N_EXPERTS];
    CUdeviceptr moe_shared_fc1_w, moe_shared_fc1_b;
    CUdeviceptr moe_shared_fc2_w, moe_shared_fc2_b;
    CUdeviceptr moe_shared_fc1_w_fp8, moe_shared_fc1_w_scale;
    CUdeviceptr moe_shared_fc2_w_fp8, moe_shared_fc2_w_scale;
    /* Skip connection (blocks 11-20) */
    int use_skip;
    CUdeviceptr skip_linear_w, skip_linear_b; /* [2048, 4096] */
    CUdeviceptr skip_norm_w, skip_norm_b;
} dit_block_gpu;

/* ShapeVAE transformer block weights (on GPU, F32) */
typedef struct {
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr qkv_w;             /* Fused QKV [3*W, W] */
    CUdeviceptr qkv_w_fp8, qkv_w_scale;
    CUdeviceptr proj_w, proj_b;
    CUdeviceptr proj_w_fp8, proj_w_scale;
    CUdeviceptr q_norm_w, q_norm_b;
    CUdeviceptr k_norm_w, k_norm_b;
    int use_qk_norm;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr mlp_fc_w, mlp_fc_b;
    CUdeviceptr mlp_fc_w_fp8, mlp_fc_w_scale;
    CUdeviceptr mlp_proj_w, mlp_proj_b;
    CUdeviceptr mlp_proj_w_fp8, mlp_proj_w_scale;
} vae_block_gpu;

/* ShapeVAE geometry decoder weights */
typedef struct {
    CUdeviceptr query_proj_w, query_proj_b;
    CUdeviceptr query_proj_w_fp8, query_proj_w_scale;
    CUdeviceptr ln1_w, ln1_b;     /* Query LN */
    CUdeviceptr ln2_w, ln2_b;     /* Key/Value LN */
    CUdeviceptr c_q_w;            /* Cross-attn Q proj */
    CUdeviceptr c_q_w_fp8, c_q_w_scale;
    CUdeviceptr c_kv_w;           /* Cross-attn KV proj [2*W, W] */
    CUdeviceptr c_kv_w_fp8, c_kv_w_scale;
    CUdeviceptr c_proj_w, c_proj_b;
    CUdeviceptr c_proj_w_fp8, c_proj_w_scale;
    CUdeviceptr q_norm_w, q_norm_b;
    CUdeviceptr k_norm_w, k_norm_b;
    int use_qk_norm;
    CUdeviceptr ln3_w, ln3_b;
    CUdeviceptr mlp_fc_w, mlp_fc_b;
    CUdeviceptr mlp_fc_w_fp8, mlp_fc_w_scale;
    CUdeviceptr mlp_proj_w, mlp_proj_b;
    CUdeviceptr mlp_proj_w_fp8, mlp_proj_w_scale;
    CUdeviceptr ln_post_w, ln_post_b;
    CUdeviceptr output_w, output_b;
} vae_geo_decoder_gpu;

struct cuda_hy3d_runner {
    /* CUDA context */
    CUdevice device;
    CUcontext ctx;
    CUstream stream;
    CUmodule module;
    int sm_version;
    int verbose;

    /* Modular ops context (all compiled kernel functions) */
    hy3d_ops ops;

    /* DINOv2 weights */
    CUdeviceptr dino_patch_w, dino_patch_b;
    CUdeviceptr dino_pos_emb;
    CUdeviceptr dino_cls_token;
    CUdeviceptr dino_final_ln_w, dino_final_ln_b;
    dino_layer_gpu dino_layers[DINO_LAYERS];

    /* DiT weights */
    CUdeviceptr dit_x_emb_w, dit_x_emb_b;
    CUdeviceptr dit_t_mlp0_w, dit_t_mlp0_b;
    CUdeviceptr dit_t_mlp2_w, dit_t_mlp2_b;
    CUdeviceptr dit_final_ln_w, dit_final_ln_b;
    CUdeviceptr dit_final_linear_w, dit_final_linear_b;
    dit_block_gpu dit_blocks[DIT_DEPTH];

    /* ShapeVAE weights */
    CUdeviceptr vae_post_kl_w, vae_post_kl_b;
    vae_block_gpu vae_blocks[VAE_DEC_LAYERS];
    vae_geo_decoder_gpu vae_geo;
    CUdeviceptr vae_fourier_freqs;  /* [num_freqs] precomputed */

    /* Scratch buffers (GPU) */
    CUdeviceptr scratch[8];        /* General purpose scratch buffers */
    size_t scratch_size[8];

    /* Per-step mesh dumping (diagnostic). When dump_every > 0, predict()
     * decodes the latent through ShapeVAE + marching cubes every N steps
     * and writes an OBJ file named <dump_prefix>_<step>.obj. */
    int dump_every;
    int dump_grid;        /* grid resolution for intermediate dumps */
    char dump_prefix[512];
    int latent_dump_count;
    int latent_dump_steps[HY3D_MAX_LATENT_DUMP_STEPS];  /* 1-based diffusion steps */
    char latent_dump_prefix[512];
    int velocity_dump_count;
    int velocity_dump_steps[HY3D_MAX_LATENT_DUMP_STEPS];
    char velocity_dump_prefix[512];
    float *init_latents;      /* host-side [4096,64] override */
    int init_latents_n;
    float *init_ctx_cond;     /* host-side [1370,1024] override */
    float *init_ctx_uncond;   /* host-side [1370,1024] override (optional) */
    int init_ctx_n;

    /* Pre-computed cross-attention K,V for DiT (constant across diffusion steps) */
    CUdeviceptr dit_ca_K[DIT_DEPTH];   /* [DINO_SEQ_LEN, DIT_HIDDEN] */
    CUdeviceptr dit_ca_V[DIT_DEPTH];   /* [DINO_SEQ_LEN, DIT_HIDDEN] */
    int ca_kv_precomputed;

    /* Load status */
    int dino_loaded, dit_loaded, vae_loaded;
};

static int hy3d_should_dump_latent(const struct cuda_hy3d_runner *r, int step1) {
    for (int i = 0; i < r->latent_dump_count; i++) {
        if (r->latent_dump_steps[i] == step1) return 1;
    }
    return 0;
}

static int hy3d_should_dump_velocity(const struct cuda_hy3d_runner *r, int step1) {
    for (int i = 0; i < r->velocity_dump_count; i++) {
        if (r->velocity_dump_steps[i] == step1) return 1;
    }
    return 0;
}

/* ======================================================================== */
/* Kernel compilation                                                       */
/* ======================================================================== */

/* Build the FP8e4m3 -> BF16 LUT used by gemm_bf16_pipe_f32. */
static void hy3d_init_fp8_to_bf16_lut(uint16_t out[256]) {
    for (int i = 0; i < 256; i++) {
        int sign = (i >> 7) & 1;
        int exp  = (i >> 3) & 0xF;
        int mant = i & 0x7;
        float v;
        if (exp == 0 && mant == 0) v = 0.f;
        else if (exp == 15 && mant == 7) v = 0.f; /* NaN */
        else if (exp == 0) v = ((float)mant / 8.f) * (1.f / 64.f);
        else v = (1.f + (float)mant / 8.f) * exp2f((float)(exp - 7));
        if (sign) v = -v;
        uint32_t b; memcpy(&b, &v, 4);
        uint32_t r = 0x7FFFu + ((b >> 16) & 1u);
        out[i] = (uint16_t)((b + r) >> 16);
    }
}

static int hy3d_compile_kernels(cuda_hy3d_runner *r) {
    /* Concat order (mirrors qimg / flux2):
     *   cuda_kernels_common_src   (opens extern "C")
     *   + cuda_hy3d_specific      (closes extern "C")
     *   + extern "C" {            (open second block)
     *   + fp8_mma_kernels_src     (FP8/BF16 MMA + flash attn, shared with qimg)
     *   + }                       (close second block)
     *
     * The second extern "C" block is required so NVRTC can find the FP8
     * kernel symbols by name. Failures to locate any of these later in
     * hy3d_ops_load() are non-fatal: the runtime falls back to the
     * existing scalar / tiled path. */
    /* fp8_mma_kernels_src expects a `to_bf16(float)` device fn to be in scope
     * (used by the *_bf16 output variants we don't dispatch but must compile).
     * Declare it inline at the top of our extern "C" block. */
    const char *mma_open  =
        "\nextern \"C\" {\n"
        "/* FP8 e4m3 -> BF16 LUT (uploaded after module load via cuModuleGetGlobal). */\n"
        "__device__ __constant__ unsigned short d_fp8_to_bf16_lut[256];\n"
        /* Returns BF16-truncated value as float (high 16 bits = BF16 bits, low 16 = 0).
         * Matches qimg's convention so gemm_bf16_pipe_*_f32 writebacks land
         * with the right interpretation in the F32 output buffer. NaN payload
         * is preserved as the canonical 0x7FC0_0000 quiet NaN. */
        "__device__ __forceinline__ float to_bf16(float f) {\n"
        "    unsigned int b; memcpy(&b, &f, 4);\n"
        "    if (((b >> 23) & 0xFF) == 0xFF && (b & 0x7FFFFF)) {\n"
        "        unsigned int qn = 0x7FC00000u; float r; memcpy(&r, &qn, 4); return r;\n"
        "    }\n"
        "    unsigned int rnd = 0x7FFFu + ((b >> 16) & 1u);\n"
        "    b = (b + rnd) & 0xFFFF0000u;\n"
        "    float out; memcpy(&out, &b, 4); return out;\n"
        "}\n";
    const char *mma_close = "\n} /* extern C (fp8_mma_kernels) */\n";
    size_t len1 = strlen(cuda_kernels_common_src);
    size_t len2 = strlen(cuda_hy3d_specific_kernels);
    size_t len3 = strlen(fp8_mma_kernels_src);
    size_t lo = strlen(mma_open);
    size_t lc = strlen(mma_close);
    char *full_src = (char *)malloc(len1 + len2 + lo + len3 + lc + 1);
    char *p = full_src;
    memcpy(p, cuda_kernels_common_src, len1);    p += len1;
    memcpy(p, cuda_hy3d_specific_kernels, len2); p += len2;
    memcpy(p, mma_open, lo);                     p += lo;
    memcpy(p, fp8_mma_kernels_src, len3);        p += len3;
    memcpy(p, mma_close, lc);                    p += lc;
    *p = '\0';

    r->sm_version = cu_compile_kernels(&r->module, r->device, full_src,
                                        "hy3d_kernels", r->verbose, "HY3D");
    free(full_src);
    if (r->sm_version < 0) return -1;

    /* Load all kernel functions via the modular ops API */
    if (hy3d_ops_load(&r->ops, r->module, r->sm_version) != 0) {
        fprintf(stderr, "HY3D: ops_load failed\n");
        return -1;
    }

    /* Upload FP8 e4m3 -> BF16 constant LUT consumed by gemm_bf16_pipe_f32. */
    {
        CUdeviceptr d_lut;
        size_t lut_sz;
        if (cuModuleGetGlobal(&d_lut, &lut_sz, r->module, "d_fp8_to_bf16_lut") == CUDA_SUCCESS &&
            lut_sz == 256 * sizeof(uint16_t)) {
            uint16_t lut[256];
            hy3d_init_fp8_to_bf16_lut(lut);
            cuMemcpyHtoD(d_lut, lut, sizeof(lut));
        }
    }

    return 0;
}

/* ======================================================================== */
/* Upload helpers                                                           */
/* ======================================================================== */

/* Upload safetensors tensor to GPU as F16 (converting F32->F16 if needed) */
static CUdeviceptr st_upload_f16(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [WARN] tensor '%s' not found\n", name);
        return 0;
    }
    void *data = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "BF16") == 0) {
        return cu_upload_raw(data, nbytes);
    } else if (strcmp(dtype, "F32") == 0) {
        /* Convert F32 to F16 on CPU then upload */
        size_t n = nbytes / sizeof(float);
        uint16_t *f16 = (uint16_t *)malloc(n * sizeof(uint16_t));
        const float *f32 = (const float *)data;
        for (size_t i = 0; i < n; i++) {
            f16[i] = cu_f32_to_f16(f32[i]);
        }
        CUdeviceptr d = cu_upload_raw(f16, n * sizeof(uint16_t));
        free(f16);
        return d;
    }
    if (verbose) fprintf(stderr, "  [WARN] tensor '%s' has unsupported dtype '%s'\n", name, dtype);
    return 0;
}

/* Upload safetensors tensor to GPU as F32 (converting F16->F32 if needed) */
static CUdeviceptr st_upload_f32(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [WARN] tensor '%s' not found\n", name);
        return 0;
    }
    void *data = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "F32") == 0) {
        return cu_upload_raw(data, nbytes);
    } else if (strcmp(dtype, "F16") == 0) {
        size_t n = nbytes / sizeof(uint16_t);
        float *f32 = (float *)malloc(n * sizeof(float));
        const uint16_t *f16 = (const uint16_t *)data;
        for (size_t i = 0; i < n; i++) {
            /* Simple F16 to F32 conversion */
            uint32_t sign = (f16[i] >> 15) & 0x1;
            uint32_t exp = (f16[i] >> 10) & 0x1f;
            uint32_t mant = f16[i] & 0x3ff;
            uint32_t f;
            if (exp == 0) {
                f = sign << 31;
            } else if (exp == 31) {
                f = (sign << 31) | 0x7f800000 | (mant << 13);
            } else {
                f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }
            memcpy(&f32[i], &f, sizeof(float));
        }
        CUdeviceptr d = cu_upload_raw(f32, n * sizeof(float));
        free(f32);
        return d;
    }
    return 0;
}

/* Fuse 3 F16 weight tensors [dim, in_dim] into one [3*dim, in_dim] on GPU */
static CUdeviceptr st_fuse_3_f16(st_context *st,
                                   const char *name_a, const char *name_b, const char *name_c,
                                   int verbose) {
    int ia = safetensors_find(st, name_a);
    int ib = safetensors_find(st, name_b);
    int ic = safetensors_find(st, name_c);
    if (ia < 0 || ib < 0 || ic < 0) {
        if (verbose) fprintf(stderr, "  [WARN] fuse: missing tensor(s)\n");
        return 0;
    }
    size_t na = safetensors_nbytes(st, ia);
    size_t nb = safetensors_nbytes(st, ib);
    size_t nc = safetensors_nbytes(st, ic);
    const char *da = safetensors_dtype(st, ia);

    /* All must be same dtype and size */
    if (na != nb || nb != nc) {
        if (verbose) fprintf(stderr, "  [WARN] fuse: size mismatch\n");
        return 0;
    }

    /* Convert to F16 if needed, then concatenate */
    size_t total = na + nb + nc;
    if (strcmp(da, "F16") == 0) {
        uint8_t *buf = (uint8_t *)malloc(total);
        memcpy(buf, safetensors_data(st, ia), na);
        memcpy(buf + na, safetensors_data(st, ib), nb);
        memcpy(buf + na + nb, safetensors_data(st, ic), nc);
        CUdeviceptr d = cu_upload_raw(buf, total);
        free(buf);
        return d;
    } else if (strcmp(da, "F32") == 0) {
        /* Convert each F32 tensor to F16, then concatenate */
        size_t n_each = na / sizeof(float);
        size_t n_total = n_each * 3;
        uint16_t *f16 = (uint16_t *)malloc(n_total * sizeof(uint16_t));
        const float *fa = (const float *)safetensors_data(st, ia);
        const float *fb = (const float *)safetensors_data(st, ib);
        const float *fc = (const float *)safetensors_data(st, ic);
        for (size_t i = 0; i < n_each; i++) f16[i] = cu_f32_to_f16(fa[i]);
        for (size_t i = 0; i < n_each; i++) f16[n_each + i] = cu_f32_to_f16(fb[i]);
        for (size_t i = 0; i < n_each; i++) f16[2 * n_each + i] = cu_f32_to_f16(fc[i]);
        CUdeviceptr d = cu_upload_raw(f16, n_total * sizeof(uint16_t));
        free(f16);
        return d;
    }
    return 0;
}

/* Allocate GPU buffer */
static CUdeviceptr gpu_alloc(size_t bytes) {
    CUdeviceptr d = 0;
    if (cuMemAlloc(&d, bytes) != CUDA_SUCCESS) {
        fprintf(stderr, "HY3D: GPU alloc failed for %zu bytes\n", bytes);
        return 0;
    }
    return d;
}

/* Online FP8 quantizer for an F16 GPU weight buffer.
 *   d_w_f16: input  [n_elem] uint16
 *   *out_d_w_fp8:   allocated here as [n_elem] uint8 (caller frees on shutdown)
 *   *out_d_w_scale: allocated here as [1]      float (caller frees on shutdown)
 * Returns 0 on success, -1 if any required kernel/alloc is missing — in which
 * case both out pointers are set to 0 and the caller falls back to F16.
 *
 * Pipeline: f16->f32 cast -> reduce_max_abs -> quantize_to_fp8_e4m3.
 * All scratch is freed before return. Stream-synchronizes once at the end so
 * the caller can immediately read scales / use FP8 buffers. */
static int hy3d_quantize_w_fp8(hy3d_ops *ops, CUstream stream,
                                CUdeviceptr d_w_f16, size_t n_elem,
                                CUdeviceptr *out_d_w_fp8,
                                CUdeviceptr *out_d_w_scale) {
    *out_d_w_fp8 = 0;
    *out_d_w_scale = 0;
    if (!ops || !d_w_f16 || n_elem == 0 ||
        !ops->f16_to_f32_buf || !ops->reduce_max_abs || !ops->quantize_fp8) {
        return -1;
    }
    CUdeviceptr d_f32 = 0, d_max = 0, d_fp8 = 0, d_scale = 0;
    if (cuMemAlloc(&d_f32, n_elem * sizeof(float)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_max, sizeof(float)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_fp8, n_elem) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_scale, sizeof(float)) != CUDA_SUCCESS) goto fail;

    int n = (int)n_elem;
    /* 1. f16 -> f32 */
    {
        unsigned grid = (unsigned)((n + 255) / 256);
        void *args[] = {&d_f32, &d_w_f16, &n};
        cuLaunchKernel(ops->f16_to_f32_buf, grid, 1, 1, 256, 1, 1, 0, stream, args, NULL);
    }
    /* 2. zero max + reduce */
    cuMemsetD8Async(d_max, 0, sizeof(float), stream);
    {
        unsigned grid = (unsigned)((n + 255) / 256);
        void *args[] = {&d_max, &d_f32, &n};
        cuLaunchKernel(ops->reduce_max_abs, grid, 1, 1, 256, 1, 1, 0, stream, args, NULL);
    }
    /* 3. quantize */
    {
        unsigned grid = (unsigned)((n + 255) / 256);
        void *args[] = {&d_fp8, &d_scale, &d_f32, &d_max, &n};
        cuLaunchKernel(ops->quantize_fp8, grid, 1, 1, 256, 1, 1, 0, stream, args, NULL);
    }
    cuStreamSynchronize(stream);
    cuMemFree(d_f32);
    cuMemFree(d_max);
    *out_d_w_fp8 = d_fp8;
    *out_d_w_scale = d_scale;
    return 0;
fail:
    if (d_f32)   cuMemFree(d_f32);
    if (d_max)   cuMemFree(d_max);
    if (d_fp8)   cuMemFree(d_fp8);
    if (d_scale) cuMemFree(d_scale);
    return -1;
}

/* Upload GEMM weight: F16 or F32 based on runner mode */
static CUdeviceptr st_upload_gemm_w(st_context *st, const char *name,
                                     int use_f32, int verbose) {
    return use_f32 ? st_upload_f32(st, name, verbose)
                   : st_upload_f16(st, name, verbose);
}

/* Fuse 3 weight tensors for GEMM: F16 or F32 based on runner mode */
static CUdeviceptr st_fuse_3_gemm_w(st_context *st,
                                      const char *a, const char *b, const char *c,
                                      int use_f32, int verbose) {
    if (use_f32) {
        /* Concatenate as F32 */
        int ia = safetensors_find(st, a);
        int ib = safetensors_find(st, b);
        int ic = safetensors_find(st, c);
        if (ia < 0 || ib < 0 || ic < 0) return 0;
        /* Get each as F32 on CPU */
        CUdeviceptr da = st_upload_f32(st, a, 0);
        CUdeviceptr db = st_upload_f32(st, b, 0);
        CUdeviceptr dc = st_upload_f32(st, c, 0);
        size_t na = safetensors_nbytes(st, ia);
        const char *dt = safetensors_dtype(st, ia);
        size_t n_elems = (strcmp(dt, "F16") == 0) ? na / 2 : na / 4;
        size_t bytes_each = n_elems * sizeof(float);
        /* Allocate combined GPU buffer and copy */
        CUdeviceptr d = gpu_alloc(bytes_each * 3);
        cuMemcpyDtoDAsync(d, da, bytes_each, 0);
        cuMemcpyDtoDAsync(d + bytes_each, db, bytes_each, 0);
        cuMemcpyDtoDAsync(d + 2 * bytes_each, dc, bytes_each, 0);
        cuMemFree(da); cuMemFree(db); cuMemFree(dc);
        return d;
    }
    return st_fuse_3_f16(st, a, b, c, verbose);
}

/* Fuse 2 weight tensors for GEMM: F16 or F32 */
static CUdeviceptr st_fuse_2_gemm_w(st_context *st,
                                      const char *a, const char *b,
                                      int use_f32, int verbose) {
    if (use_f32) {
        int ia = safetensors_find(st, a);
        int ib = safetensors_find(st, b);
        if (ia < 0 || ib < 0) return 0;
        CUdeviceptr da = st_upload_f32(st, a, 0);
        CUdeviceptr db = st_upload_f32(st, b, 0);
        size_t na = safetensors_nbytes(st, ia);
        const char *dt = safetensors_dtype(st, ia);
        size_t n_elems = (strcmp(dt, "F16") == 0) ? na / 2 : na / 4;
        size_t bytes_each = n_elems * sizeof(float);
        CUdeviceptr d = gpu_alloc(bytes_each * 2);
        cuMemcpyDtoDAsync(d, da, bytes_each, 0);
        cuMemcpyDtoDAsync(d + bytes_each, db, bytes_each, 0);
        cuMemFree(da); cuMemFree(db);
        return d;
    }
    /* F16 variant: concatenate raw F16 data */
    int ia = safetensors_find(st, a);
    int ib = safetensors_find(st, b);
    if (ia < 0 || ib < 0) return 0;
    size_t na = safetensors_nbytes(st, ia);
    size_t nb = safetensors_nbytes(st, ib);
    uint8_t *buf = (uint8_t *)malloc(na + nb);
    memcpy(buf, safetensors_data(st, ia), na);
    memcpy(buf + na, safetensors_data(st, ib), nb);
    CUdeviceptr d = cu_upload_raw(buf, na + nb);
    free(buf);
    return d;
}

/* ======================================================================== */
/* Scratch buffer management                                                */
/* ======================================================================== */

static void ensure_scratch(cuda_hy3d_runner *r, int idx, size_t bytes) {
    if (r->scratch_size[idx] < bytes) {
        if (r->scratch[idx]) cuMemFree(r->scratch[idx]);
        r->scratch[idx] = gpu_alloc(bytes);
        r->scratch_size[idx] = bytes;
    }
}

/* ======================================================================== */
/* Weight loading                                                           */
/* ======================================================================== */

static int load_dino_weights(cuda_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open DINOv2 weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading DINOv2 from %s (%d tensors)\n",
                             path, st->n_tensors);

    /* Embeddings */
    r->dino_patch_w = st_upload_f32(st, "main_image_encoder.model.embeddings.patch_embeddings.projection.weight", r->verbose);
    r->dino_patch_b = st_upload_f32(st, "main_image_encoder.model.embeddings.patch_embeddings.projection.bias", r->verbose);
    r->dino_pos_emb = st_upload_f32(st, "main_image_encoder.model.embeddings.position_embeddings", r->verbose);
    r->dino_cls_token = st_upload_f32(st, "main_image_encoder.model.embeddings.cls_token", r->verbose);

    /* Encoder layers
     * GEMM weights → F16 (gemm_f16_f32 expects half_raw)
     * LayerNorm weights, biases, LayerScale → F32 (layernorm_f32 expects float) */
    for (int i = 0; i < DINO_LAYERS; i++) {
        char name[256];
        dino_layer_gpu *l = &r->dino_layers[i];

        /* F32 uploads: LayerNorm, LayerScale, biases used by LN or standalone */
        #define DINO_F32(field, suffix) \
            snprintf(name, sizeof(name), "main_image_encoder.model.encoder.layer.%d.%s", i, suffix); \
            l->field = st_upload_f32(st, name, r->verbose);
        /* GEMM weight uploads: F16 or F32 depending on mode */
        int f32g = r->ops.use_f32_gemm;
        #define DINO_GW(field, suffix) \
            snprintf(name, sizeof(name), "main_image_encoder.model.encoder.layer.%d.%s", i, suffix); \
            l->field = st_upload_gemm_w(st, name, f32g, r->verbose);

        DINO_F32(ln1_w, "norm1.weight");
        DINO_F32(ln1_b, "norm1.bias");
        DINO_GW(q_w,   "attention.attention.query.weight");
        DINO_F32(q_b,   "attention.attention.query.bias");
        DINO_GW(k_w,   "attention.attention.key.weight");
        DINO_F32(k_b,   "attention.attention.key.bias");
        DINO_GW(v_w,   "attention.attention.value.weight");
        DINO_F32(v_b,   "attention.attention.value.bias");
        DINO_GW(out_w, "attention.output.dense.weight");
        DINO_F32(out_b, "attention.output.dense.bias");
        DINO_F32(ls1,   "layer_scale1.lambda1");
        DINO_F32(ln2_w, "norm2.weight");
        DINO_F32(ln2_b, "norm2.bias");
        DINO_GW(fc1_w, "mlp.fc1.weight");
        DINO_F32(fc1_b, "mlp.fc1.bias");
        DINO_GW(fc2_w, "mlp.fc2.weight");
        DINO_F32(fc2_b, "mlp.fc2.bias");
        DINO_F32(ls2,   "layer_scale2.lambda1");
        #undef DINO_F32
        #undef DINO_GW
    }

    /* Final LN (F32 — consumed by layernorm_f32) */
    r->dino_final_ln_w = st_upload_f32(st, "main_image_encoder.model.layernorm.weight", r->verbose);
    r->dino_final_ln_b = st_upload_f32(st, "main_image_encoder.model.layernorm.bias", r->verbose);
    if (!r->dino_final_ln_w) {
        r->dino_final_ln_w = st_upload_f32(st, "main_image_encoder.model.norm.weight", r->verbose);
        r->dino_final_ln_b = st_upload_f32(st, "main_image_encoder.model.norm.bias", r->verbose);
    }

    safetensors_close(st);
    r->dino_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: DINOv2 weights loaded\n");
    return 0;
}

static int load_dit_weights(cuda_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open DiT weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading DiT from %s (%d tensors)\n",
                             path, st->n_tensors);

    int f32g = r->ops.use_f32_gemm;

    /* x_embedder */
    r->dit_x_emb_w = st_upload_gemm_w(st, "x_embedder.weight", f32g, r->verbose);
    r->dit_x_emb_b = st_upload_f32(st, "x_embedder.bias", r->verbose);

    /* t_embedder */
    r->dit_t_mlp0_w = st_upload_gemm_w(st, "t_embedder.mlp.0.weight", f32g, r->verbose);
    r->dit_t_mlp0_b = st_upload_f32(st, "t_embedder.mlp.0.bias", r->verbose);
    r->dit_t_mlp2_w = st_upload_gemm_w(st, "t_embedder.mlp.2.weight", f32g, r->verbose);
    r->dit_t_mlp2_b = st_upload_f32(st, "t_embedder.mlp.2.bias", r->verbose);

    /* Blocks: GEMM weights F16 or F32 (switchable), LN/RMSNorm always F32 */
    for (int i = 0; i < DIT_DEPTH; i++) {
        char name[256];
        dit_block_gpu *b = &r->dit_blocks[i];
        #define DIT_GW(field, suffix) \
            snprintf(name, sizeof(name), "blocks.%d.%s", i, suffix); \
            b->field = st_upload_gemm_w(st, name, f32g, r->verbose);
        #define DIT_F32(field, suffix) \
            snprintf(name, sizeof(name), "blocks.%d.%s", i, suffix); \
            b->field = st_upload_f32(st, name, r->verbose);

        /* LayerNorm → F32 */
        DIT_F32(norm1_w,     "norm1.weight");
        DIT_F32(norm1_b,     "norm1.bias");
        /* Self-attn: fuse Q/K/V weights into [3*dim, dim] for correct head interleaving */
        {
            char nq[256], nk[256], nv[256];
            snprintf(nq, sizeof(nq), "blocks.%d.attn1.to_q.weight", i);
            snprintf(nk, sizeof(nk), "blocks.%d.attn1.to_k.weight", i);
            snprintf(nv, sizeof(nv), "blocks.%d.attn1.to_v.weight", i);
            b->sa_qkv_w = st_fuse_3_gemm_w(st, nq, nk, nv, f32g, r->verbose);
        }
        DIT_GW(sa_out_w,    "attn1.out_proj.weight");
        DIT_F32(sa_out_b,    "attn1.out_proj.bias");
        /* RMSNorm weights → F32 */
        DIT_F32(sa_q_norm_w, "attn1.q_norm.weight");
        DIT_F32(sa_k_norm_w, "attn1.k_norm.weight");

        DIT_F32(norm2_w,     "norm2.weight");
        DIT_F32(norm2_b,     "norm2.bias");
        DIT_GW(ca_q_w,      "attn2.to_q.weight");
        DIT_GW(ca_k_w,      "attn2.to_k.weight");
        DIT_GW(ca_v_w,      "attn2.to_v.weight");
        DIT_GW(ca_out_w,    "attn2.out_proj.weight");
        DIT_F32(ca_out_b,    "attn2.out_proj.bias");
        DIT_F32(ca_q_norm_w, "attn2.q_norm.weight");
        DIT_F32(ca_k_norm_w, "attn2.k_norm.weight");

        DIT_F32(norm3_w,     "norm3.weight");
        DIT_F32(norm3_b,     "norm3.bias");

        /* Skip connection (blocks 11-20) */
        b->use_skip = (i > DIT_HALF_DEPTH) ? 1 : 0;
        if (b->use_skip) {
            DIT_GW(skip_linear_w, "skip_linear.weight");
            DIT_F32(skip_linear_b, "skip_linear.bias");
            DIT_F32(skip_norm_w,   "skip_norm.weight");
            DIT_F32(skip_norm_b,   "skip_norm.bias");
        }

        /* MoE vs regular MLP */
        b->use_moe = (i >= DIT_MOE_START) ? 1 : 0;
        if (b->use_moe) {
            DIT_GW(moe_gate_w, "moe.gate.weight");
            for (int e = 0; e < DIT_N_EXPERTS; e++) {
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.0.proj.weight", i, e);
                b->moe_expert_fc1_w[e] = st_upload_gemm_w(st, name, f32g, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.0.proj.bias", i, e);
                b->moe_expert_fc1_b[e] = st_upload_f32(st, name, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.2.weight", i, e);
                b->moe_expert_fc2_w[e] = st_upload_gemm_w(st, name, f32g, r->verbose);
                snprintf(name, sizeof(name), "blocks.%d.moe.experts.%d.net.2.bias", i, e);
                b->moe_expert_fc2_b[e] = st_upload_f32(st, name, r->verbose);
            }
            DIT_GW(moe_shared_fc1_w, "moe.shared_experts.net.0.proj.weight");
            DIT_F32(moe_shared_fc1_b, "moe.shared_experts.net.0.proj.bias");
            DIT_GW(moe_shared_fc2_w, "moe.shared_experts.net.2.weight");
            DIT_F32(moe_shared_fc2_b, "moe.shared_experts.net.2.bias");
        } else {
            DIT_GW(mlp_fc1_w,   "mlp.fc1.weight");
            DIT_F32(mlp_fc1_b,   "mlp.fc1.bias");
            DIT_GW(mlp_fc2_w,   "mlp.fc2.weight");
            DIT_F32(mlp_fc2_b,   "mlp.fc2.bias");
        }
        #undef DIT_GW
        #undef DIT_F32
    }

    /* Final layer: LN → F32, linear → F16 */
    r->dit_final_ln_w = st_upload_f32(st, "final_layer.norm_final.weight", r->verbose);
    r->dit_final_ln_b = st_upload_f32(st, "final_layer.norm_final.bias", r->verbose);
    r->dit_final_linear_w = st_upload_gemm_w(st, "final_layer.linear.weight", f32g, r->verbose);
    r->dit_final_linear_b = st_upload_f32(st, "final_layer.linear.bias", r->verbose);

    safetensors_close(st);

    /* Online FP8 prequant of all hot DiT GEMM weights. Skipped silently when
     * use_fp8_gemm is off or when sm<89 / kernels are missing — caller falls
     * back to F16. */
    if (!f32g && r->ops.use_fp8_gemm && r->ops.gemm_fp8_perrow) {
        const size_t fc1_n   = (size_t)DIT_FFN * DIT_HIDDEN;        /* [8192, 2048] */
        const size_t fc2_n   = (size_t)DIT_HIDDEN * DIT_FFN;        /* [2048, 8192] */
        const size_t qkv_n   = (size_t)(3 * DIT_HIDDEN) * DIT_HIDDEN; /* [6144, 2048] */
        const size_t out_n   = (size_t)DIT_HIDDEN * DIT_HIDDEN;      /* [2048, 2048] */
        const size_t caq_n   = out_n;                                /* [2048, 2048] */
        const size_t cakv_n  = (size_t)DIT_HIDDEN * DINO_HIDDEN;     /* [2048, 1024] */
        int n_quant = 0;
        size_t bytes = 0;
        #define QFP8(field, n_elem) do { \
            if (b->field && hy3d_quantize_w_fp8(&r->ops, r->stream, \
                    b->field, (n_elem), &b->field##_fp8, &b->field##_scale) == 0) { \
                n_quant++; bytes += (n_elem); \
            } \
        } while (0)
        for (int i = 0; i < DIT_DEPTH; i++) {
            dit_block_gpu *b = &r->dit_blocks[i];
            QFP8(sa_qkv_w, qkv_n);
            QFP8(sa_out_w, out_n);
            QFP8(ca_q_w,   caq_n);
            QFP8(ca_k_w,   cakv_n);
            QFP8(ca_v_w,   cakv_n);
            QFP8(ca_out_w, out_n);
            if (b->use_moe) {
                for (int e = 0; e < DIT_N_EXPERTS; e++) {
                    if (b->moe_expert_fc1_w[e] &&
                        hy3d_quantize_w_fp8(&r->ops, r->stream,
                                             b->moe_expert_fc1_w[e], fc1_n,
                                             &b->moe_expert_fc1_w_fp8[e],
                                             &b->moe_expert_fc1_w_scale[e]) == 0)
                        { n_quant++; bytes += fc1_n; }
                    if (b->moe_expert_fc2_w[e] &&
                        hy3d_quantize_w_fp8(&r->ops, r->stream,
                                             b->moe_expert_fc2_w[e], fc2_n,
                                             &b->moe_expert_fc2_w_fp8[e],
                                             &b->moe_expert_fc2_w_scale[e]) == 0)
                        { n_quant++; bytes += fc2_n; }
                }
                /* Shared expert sees all tokens (dense path); same shapes as
                 * one MoE expert. */
                if (b->moe_shared_fc1_w &&
                    hy3d_quantize_w_fp8(&r->ops, r->stream,
                                         b->moe_shared_fc1_w, fc1_n,
                                         &b->moe_shared_fc1_w_fp8,
                                         &b->moe_shared_fc1_w_scale) == 0)
                    { n_quant++; bytes += fc1_n; }
                if (b->moe_shared_fc2_w &&
                    hy3d_quantize_w_fp8(&r->ops, r->stream,
                                         b->moe_shared_fc2_w, fc2_n,
                                         &b->moe_shared_fc2_w_fp8,
                                         &b->moe_shared_fc2_w_scale) == 0)
                    { n_quant++; bytes += fc2_n; }
            } else {
                QFP8(mlp_fc1_w, fc1_n);
                QFP8(mlp_fc2_w, fc2_n);
            }
        }
        #undef QFP8
        if (r->verbose)
            fprintf(stderr, "HY3D: online FP8 quant — %d weight buffers "
                            "(%.1f MB FP8)\n",
                    n_quant, (double)bytes / (1024.0*1024.0));
    }

    r->dit_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: DiT weights loaded\n");
    return 0;
}

/* Fix 6 & 7: VAE tensor name mapping */
static int load_vae_weights(cuda_hy3d_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "HY3D: cannot open VAE weights: %s\n", path);
        return -1;
    }
    if (r->verbose) fprintf(stderr, "HY3D: loading ShapeVAE from %s (%d tensors)\n",
                             path, st->n_tensors);

    /* Post-KL projection (GEMM: weight F16, bias F32) */
    r->vae_post_kl_w = st_upload_gemm_w(st, "post_kl.weight", r->ops.use_f32_gemm, r->verbose);
    r->vae_post_kl_b = st_upload_f32(st, "post_kl.bias", r->verbose);

    /* Fix 6: Transformer decoder blocks use transformer.resblocks.N prefix
     * GEMM weights → F16/F32 (switchable), biases/LN/QKnorm → F32 */
    int f32g = r->ops.use_f32_gemm;
    for (int i = 0; i < VAE_DEC_LAYERS; i++) {
        char name[256];
        vae_block_gpu *b = &r->vae_blocks[i];
        #define VAE_F16(field, suffix) \
            snprintf(name, sizeof(name), "transformer.resblocks.%d.%s", i, suffix); \
            b->field = st_upload_gemm_w(st, name, f32g, r->verbose);
        #define VAE_F32(field, suffix) \
            snprintf(name, sizeof(name), "transformer.resblocks.%d.%s", i, suffix); \
            b->field = st_upload_f32(st, name, r->verbose);

        VAE_F32(ln1_w,        "ln_1.weight");
        VAE_F32(ln1_b,        "ln_1.bias");
        VAE_F16(qkv_w,        "attn.c_qkv.weight");
        VAE_F16(proj_w,       "attn.c_proj.weight");
        VAE_F32(proj_b,       "attn.c_proj.bias");
        VAE_F32(q_norm_w,     "attn.attention.q_norm.weight");
        VAE_F32(q_norm_b,     "attn.attention.q_norm.bias");
        VAE_F32(k_norm_w,     "attn.attention.k_norm.weight");
        VAE_F32(k_norm_b,     "attn.attention.k_norm.bias");
        VAE_F32(ln2_w,        "ln_2.weight");
        VAE_F32(ln2_b,        "ln_2.bias");
        VAE_F16(mlp_fc_w,     "mlp.c_fc.weight");
        VAE_F32(mlp_fc_b,     "mlp.c_fc.bias");
        VAE_F16(mlp_proj_w,   "mlp.c_proj.weight");
        VAE_F32(mlp_proj_b,   "mlp.c_proj.bias");
        #undef VAE_F16
        #undef VAE_F32

        b->use_qk_norm = (b->q_norm_w != 0);
    }

    /* Fix 7: Geometry decoder
     * GEMM weights → F16, biases → F32; LN/QKnorm → F32 */
    vae_geo_decoder_gpu *g = &r->vae_geo;
    #define GEO_F16(field, suffix) g->field = st_upload_gemm_w(st, suffix, f32g, r->verbose);
    #define GEO_F32(field, suffix) g->field = st_upload_f32(st, suffix, r->verbose);

    GEO_F16(query_proj_w, "geo_decoder.query_proj.weight");
    GEO_F32(query_proj_b, "geo_decoder.query_proj.bias");
    GEO_F32(ln1_w,        "geo_decoder.cross_attn_decoder.ln_1.weight");
    GEO_F32(ln1_b,        "geo_decoder.cross_attn_decoder.ln_1.bias");
    GEO_F32(ln2_w,        "geo_decoder.cross_attn_decoder.ln_2.weight");
    GEO_F32(ln2_b,        "geo_decoder.cross_attn_decoder.ln_2.bias");
    GEO_F16(c_q_w,        "geo_decoder.cross_attn_decoder.attn.c_q.weight");
    GEO_F16(c_kv_w,       "geo_decoder.cross_attn_decoder.attn.c_kv.weight");
    GEO_F16(c_proj_w,     "geo_decoder.cross_attn_decoder.attn.c_proj.weight");
    GEO_F32(c_proj_b,     "geo_decoder.cross_attn_decoder.attn.c_proj.bias");
    GEO_F32(q_norm_w,     "geo_decoder.cross_attn_decoder.attn.attention.q_norm.weight");
    GEO_F32(q_norm_b,     "geo_decoder.cross_attn_decoder.attn.attention.q_norm.bias");
    GEO_F32(k_norm_w,     "geo_decoder.cross_attn_decoder.attn.attention.k_norm.weight");
    GEO_F32(k_norm_b,     "geo_decoder.cross_attn_decoder.attn.attention.k_norm.bias");
    GEO_F32(ln3_w,        "geo_decoder.cross_attn_decoder.ln_3.weight");
    GEO_F32(ln3_b,        "geo_decoder.cross_attn_decoder.ln_3.bias");
    GEO_F16(mlp_fc_w,     "geo_decoder.cross_attn_decoder.mlp.c_fc.weight");
    GEO_F32(mlp_fc_b,     "geo_decoder.cross_attn_decoder.mlp.c_fc.bias");
    GEO_F16(mlp_proj_w,   "geo_decoder.cross_attn_decoder.mlp.c_proj.weight");
    GEO_F32(mlp_proj_b,   "geo_decoder.cross_attn_decoder.mlp.c_proj.bias");
    GEO_F32(ln_post_w,    "geo_decoder.ln_post.weight");
    GEO_F32(ln_post_b,    "geo_decoder.ln_post.bias");
    GEO_F16(output_w,     "geo_decoder.output_proj.weight");
    GEO_F32(output_b,     "geo_decoder.output_proj.bias");
    #undef GEO_F16
    #undef GEO_F32

    g->use_qk_norm = (g->q_norm_w != 0);

    /* Fix 5: Pre-compute Fourier frequencies WITHOUT pi multiplier */
    float freqs[VAE_NUM_FREQS];
    for (int i = 0; i < VAE_NUM_FREQS; i++) {
        freqs[i] = powf(2.0f, (float)i);  /* 2^i, no pi factor */
    }
    r->vae_fourier_freqs = cu_upload_raw(freqs, sizeof(freqs));

    safetensors_close(st);

    /* Online FP8 prequant of VAE GEMM weights, mirrors load_dit_weights.
     * Eligible: n_in % 32 == 0 && n_out % 64 == 0. Skipped silently when
     * the runner is not on FP8 TC path. ~250 MB FP8. */
    if (r->ops.use_fp8_gemm && r->ops.f16_to_f32_buf && r->ops.reduce_max_abs &&
        r->ops.quantize_fp8) {
        const size_t W   = VAE_WIDTH;          /* 1024 */
        size_t n_qkv  = 3 * W * W;
        size_t n_proj = W * W;
        size_t n_fc   = 4 * W * W;
        size_t total_bytes = 0;
        int n_buf = 0;
        for (int i = 0; i < VAE_DEC_LAYERS; i++) {
            vae_block_gpu *b = &r->vae_blocks[i];
            if (b->qkv_w &&
                hy3d_quantize_w_fp8(&r->ops, r->stream, b->qkv_w, n_qkv,
                                    &b->qkv_w_fp8, &b->qkv_w_scale) == 0) {
                total_bytes += n_qkv; n_buf++;
            }
            if (b->proj_w &&
                hy3d_quantize_w_fp8(&r->ops, r->stream, b->proj_w, n_proj,
                                    &b->proj_w_fp8, &b->proj_w_scale) == 0) {
                total_bytes += n_proj; n_buf++;
            }
            if (b->mlp_fc_w &&
                hy3d_quantize_w_fp8(&r->ops, r->stream, b->mlp_fc_w, n_fc,
                                    &b->mlp_fc_w_fp8, &b->mlp_fc_w_scale) == 0) {
                total_bytes += n_fc; n_buf++;
            }
            if (b->mlp_proj_w &&
                hy3d_quantize_w_fp8(&r->ops, r->stream, b->mlp_proj_w, n_fc,
                                    &b->mlp_proj_w_fp8, &b->mlp_proj_w_scale) == 0) {
                total_bytes += n_fc; n_buf++;
            }
        }
        vae_geo_decoder_gpu *g = &r->vae_geo;
        size_t n_kv = 2 * W * W;
        if (g->c_q_w &&
            hy3d_quantize_w_fp8(&r->ops, r->stream, g->c_q_w, n_proj,
                                &g->c_q_w_fp8, &g->c_q_w_scale) == 0) {
            total_bytes += n_proj; n_buf++;
        }
        if (g->c_kv_w &&
            hy3d_quantize_w_fp8(&r->ops, r->stream, g->c_kv_w, n_kv,
                                &g->c_kv_w_fp8, &g->c_kv_w_scale) == 0) {
            total_bytes += n_kv; n_buf++;
        }
        if (g->c_proj_w &&
            hy3d_quantize_w_fp8(&r->ops, r->stream, g->c_proj_w, n_proj,
                                &g->c_proj_w_fp8, &g->c_proj_w_scale) == 0) {
            total_bytes += n_proj; n_buf++;
        }
        if (g->mlp_fc_w &&
            hy3d_quantize_w_fp8(&r->ops, r->stream, g->mlp_fc_w, n_fc,
                                &g->mlp_fc_w_fp8, &g->mlp_fc_w_scale) == 0) {
            total_bytes += n_fc; n_buf++;
        }
        if (g->mlp_proj_w &&
            hy3d_quantize_w_fp8(&r->ops, r->stream, g->mlp_proj_w, n_fc,
                                &g->mlp_proj_w_fp8, &g->mlp_proj_w_scale) == 0) {
            total_bytes += n_fc; n_buf++;
        }
        if (r->verbose && n_buf > 0) {
            fprintf(stderr, "HY3D: VAE FP8 prequant — %d weight buffers (%.1f MB FP8)\n",
                    n_buf, (double)total_bytes / (1024.0 * 1024.0));
        }
    }

    r->vae_loaded = 1;
    if (r->verbose) fprintf(stderr, "HY3D: ShapeVAE weights loaded\n");
    return 0;
}

/* ======================================================================== */
/* Pipeline stages                                                          */
/* ======================================================================== */

/* Stage 1: DINOv2 forward pass
 * Input:  d_image [3, 518, 518] F32 (normalized)
 * Output: d_out [1370, 1024] F32 */
static void run_dinov2(cuda_hy3d_runner *r, CUdeviceptr d_image, CUdeviceptr d_out) {
    hy3d_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int seq = DINO_SEQ_LEN;
    const int dim = DINO_HIDDEN;
    const int heads = DINO_HEADS;
    const int hd = DINO_HEAD_DIM;
    const int ffn = DINO_FFN;
    const int ps = DINO_PATCH;
    const int gw = DINO_IMG_SIZE / ps;  /* 37 */

    /* Scratch: 0=hidden[seq*dim], 1=qkv[3*seq*dim], 2=attn_out[seq*dim],
     *          3=mlp[seq*ffn], 4=normed[seq*dim] */
    ensure_scratch(r, 0, (size_t)seq * dim * sizeof(float));
    ensure_scratch(r, 1, (size_t)3 * seq * dim * sizeof(float));
    ensure_scratch(r, 2, (size_t)seq * dim * sizeof(float));
    ensure_scratch(r, 3, (size_t)seq * ffn * sizeof(float));
    ensure_scratch(r, 4, (size_t)seq * dim * sizeof(float));

    CUdeviceptr d_hidden = r->scratch[0];
    CUdeviceptr d_qkv    = r->scratch[1];
    CUdeviceptr d_attn   = r->scratch[2];
    CUdeviceptr d_mlp    = r->scratch[3];
    CUdeviceptr d_normed = r->scratch[4];

    /* 1. Patch embedding: conv2d -> [num_patches, dim]
     * Kernel: blockIdx.x = patch index, threadIdx.x loops over output channels */
    {
        int gw2 = gw;
        int dim2 = dim, ps2 = ps, img_w = DINO_IMG_SIZE;
        CUdeviceptr pw = r->dino_patch_w, pb = r->dino_patch_b;
        void *args[] = {&d_hidden, &d_image, &pw, &pb, &gw2, &dim2, &ps2, &img_w};
        cuLaunchKernel(ops->patch_embed,
                       (unsigned)(gw * gw), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* 2. CLS token + position embeddings */
    {
        int n_tok = seq, d2 = dim;
        CUdeviceptr cls = r->dino_cls_token, pos = r->dino_pos_emb;
        void *args[] = {&d_hidden, &cls, &pos, &n_tok, &d2};
        cuLaunchKernel(ops->cls_pos_embed,
                       (unsigned)((seq*dim+255)/256), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }
    hy3d_dbg_dump_npy(stream, d_hidden, seq, dim, "cuda_dinov2_hidden_0.npy");

    /* 3. Encoder layers */
    for (int li = 0; li < DINO_LAYERS; li++) {
        dino_layer_gpu *l = &r->dino_layers[li];

        /* LN1 -> Q,K,V -> Attention -> LayerScale + residual */
        op_layernorm(ops, stream, d_normed, d_hidden, l->ln1_w, l->ln1_b, seq, dim);

        /* Q, K, V projections */
        CUdeviceptr d_Q = d_qkv;
        CUdeviceptr d_K = d_qkv + (size_t)seq * dim * sizeof(float);
        CUdeviceptr d_V = d_qkv + (size_t)2 * seq * dim * sizeof(float);
        op_gemm(ops, stream, d_Q, l->q_w, d_normed, l->q_b, dim, dim, seq);
        op_gemm(ops, stream, d_K, l->k_w, d_normed, l->k_b, dim, dim, seq);
        op_gemm(ops, stream, d_V, l->v_w, d_normed, l->v_b, dim, dim, seq);

        /* Self-attention */
        op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, seq, dim, heads, hd);

        /* Output projection */
        op_gemm(ops, stream, d_normed, l->out_w, d_attn, l->out_b, dim, dim, seq);

        /* LayerScale 1 + residual */
        if (l->ls1)
            op_layerscale_add(ops, stream, d_hidden, d_normed, l->ls1, seq * dim, dim);
        else
            op_add(ops, stream, d_hidden, d_normed, seq * dim);

        /* LN2 -> MLP -> LayerScale + residual */
        op_layernorm(ops, stream, d_normed, d_hidden, l->ln2_w, l->ln2_b, seq, dim);
        op_gemm(ops, stream, d_mlp, l->fc1_w, d_normed, l->fc1_b, ffn, dim, seq);
        op_gelu(ops, stream, d_mlp, seq * ffn);
        op_gemm(ops, stream, d_normed, l->fc2_w, d_mlp, l->fc2_b, dim, ffn, seq);

        if (l->ls2)
            op_layerscale_add(ops, stream, d_hidden, d_normed, l->ls2, seq * dim, dim);
        else
            op_add(ops, stream, d_hidden, d_normed, seq * dim);

        /* HF hidden_states index = block_index + 1 */
        if (li == 11)
            hy3d_dbg_dump_npy(stream, d_hidden, seq, dim, "cuda_dinov2_hidden_12.npy");
        else if (li == 22)
            hy3d_dbg_dump_npy(stream, d_hidden, seq, dim, "cuda_dinov2_hidden_23.npy");
        else if (li == 23)
            hy3d_dbg_dump_npy(stream, d_hidden, seq, dim, "cuda_dinov2_hidden_24.npy");
    }

    /* 4. Final LN */
    if (r->dino_final_ln_w) {
        op_layernorm(ops, stream, d_out, d_hidden, r->dino_final_ln_w, r->dino_final_ln_b, seq, dim);
    } else {
        cuMemcpyDtoDAsync(d_out, d_hidden, (size_t)seq * dim * sizeof(float), stream);
    }
}

/* precompute_dit_ca_kv removed — K/V computed per-block inside run_dit_forward */

/*
 * MoE forward: simplified "run all experts on all tokens" approach.
 *
 * 1. gate_logits = x @ gate_w.T             -> [N, 8]
 * 2. Download gate_logits, compute softmax, top-2 mask on CPU
 * 3. Shared expert: shared_out = GELU(x @ shared_fc1_w.T + b) @ shared_fc2_w.T + b
 * 4. For each expert e (0..7):
 *      expert_out_e = GELU(x @ fc1_w_e.T + b_e) @ fc2_w_e.T + b_e
 *      output += expert_out_e * gate_weights[:, e]  (only top-2 are non-zero)
 * 5. output += shared_out
 *
 * The gate weights (top-2 per row, rest zero) are uploaded as [N] floats per expert.
 * We accumulate into d_output which starts as zeros.
 */
static void run_dit_moe(cuda_hy3d_runner *r, dit_block_gpu *blk,
                         CUdeviceptr d_input, CUdeviceptr d_output,
                         int N_tok, CUdeviceptr d_moe_scratch) {
    hy3d_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int H_dim = DIT_HIDDEN;
    const int ffn = DIT_FFN;

    /* Scratch layout within d_moe_scratch:
     * gate_logits  [N_tok * DIT_N_EXPERTS]    (~131KB)
     * expert_h     [N_tok * ffn]              (~134MB, reused per expert + shared)
     * expert_o     [N_tok * H_dim]            (~33.5MB, reused per expert + shared)
     * moe_idx      [N_tok * top_k]  int       (~32KB)
     * moe_w        [N_tok * top_k]  float     (~32KB)
     */
    size_t off = 0;
    CUdeviceptr d_gate   = d_moe_scratch + off; off += (size_t)N_tok * DIT_N_EXPERTS * sizeof(float);
    CUdeviceptr d_exp_h  = d_moe_scratch + off; off += (size_t)N_tok * ffn * sizeof(float);
    CUdeviceptr d_exp_o  = d_moe_scratch + off; off += (size_t)N_tok * H_dim * sizeof(float);
    CUdeviceptr d_moe_idx = d_moe_scratch + off; off += (size_t)N_tok * DIT_MOE_TOP_K * sizeof(int);
    CUdeviceptr d_moe_w   = d_moe_scratch + off; /* [N_tok * top_k] */

    /* Step 1: Compute gate logits: [N_tok, H_dim] @ [8, H_dim]^T -> [N_tok, 8] */
    op_gemm(ops, stream, d_gate, blk->moe_gate_w, d_input, 0,
            DIT_N_EXPERTS, H_dim, N_tok);

    /* Step 2: GPU-side softmax + top-k (was CPU bounce of [N_tok*8]). */
    int n_experts = DIT_N_EXPERTS;
    int top_k = DIT_MOE_TOP_K;
    if (ops->moe_topk_softmax) {
        void *args[] = {&d_moe_idx, &d_moe_w, &d_gate, &N_tok, &n_experts, &top_k};
        cuLaunchKernel(ops->moe_topk_softmax, (unsigned)N_tok, 1, 1,
                       32, 1, 1, 0, stream, args, NULL);
    } else {
        /* Fallback: CPU top-k path (matches former behaviour). */
        float *gate_cpu = (float *)malloc((size_t)N_tok * DIT_N_EXPERTS * sizeof(float));
        cuStreamSynchronize(stream);
        cuMemcpyDtoH(gate_cpu, d_gate, (size_t)N_tok * DIT_N_EXPERTS * sizeof(float));
        int *idx_cpu = (int *)malloc((size_t)N_tok * top_k * sizeof(int));
        float *w_cpu = (float *)malloc((size_t)N_tok * top_k * sizeof(float));
        for (int t = 0; t < N_tok; t++) {
            float *row = gate_cpu + t * DIT_N_EXPERTS;
            float mx = row[0];
            for (int e = 1; e < DIT_N_EXPERTS; e++) if (row[e] > mx) mx = row[e];
            float sum = 0.0f, sm[DIT_N_EXPERTS];
            for (int e = 0; e < DIT_N_EXPERTS; e++) { sm[e] = expf(row[e] - mx); sum += sm[e]; }
            float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
            for (int e = 0; e < DIT_N_EXPERTS; e++) sm[e] *= inv;
            for (int k = 0; k < top_k; k++) {
                int best = -1; float bv = -1.0f;
                for (int e = 0; e < DIT_N_EXPERTS; e++) if (sm[e] > bv) { bv = sm[e]; best = e; }
                idx_cpu[t*top_k+k] = best;
                w_cpu[t*top_k+k] = bv;
                if (best >= 0) sm[best] = -1.0f;
            }
        }
        cuMemcpyHtoDAsync(d_moe_idx, idx_cpu, (size_t)N_tok*top_k*sizeof(int), stream);
        cuMemcpyHtoDAsync(d_moe_w,   w_cpu,   (size_t)N_tok*top_k*sizeof(float), stream);
        cuStreamSynchronize(stream);
        free(gate_cpu); free(idx_cpu); free(w_cpu);
    }

    /* Step 3: Zero out accumulator (d_output) */
    cuMemsetD8Async(d_output, 0, (size_t)N_tok * H_dim * sizeof(float), stream);

    /* Step 4: For each expert, compute output and weighted-add ON GPU
     * (eliminates the per-expert ~33.5MB host roundtrip). */
    int H_dim_arg = H_dim;  /* drop const for kernel arg pointer */
    int total_acc = N_tok * H_dim;
    unsigned acc_grid = (unsigned)((total_acc + 255) / 256);
    for (int e = 0; e < DIT_N_EXPERTS; e++) {
        /* expert_out = GELU(x @ fc1_w_e.T + fc1_b_e) @ fc2_w_e.T + fc2_b_e
         * FP8 path used when w_fp8/scale were prequantized in load_dit_weights;
         * otherwise falls through to the F16 tiled path inside op_gemm_qw. */
        op_gemm_qw(ops, stream, d_exp_h,
                   blk->moe_expert_fc1_w[e],
                   blk->moe_expert_fc1_w_fp8[e], blk->moe_expert_fc1_w_scale[e],
                   d_input, blk->moe_expert_fc1_b[e], ffn, H_dim, N_tok);
        op_gelu(ops, stream, d_exp_h, N_tok * ffn);
        op_gemm_qw(ops, stream, d_exp_o,
                   blk->moe_expert_fc2_w[e],
                   blk->moe_expert_fc2_w_fp8[e], blk->moe_expert_fc2_w_scale[e],
                   d_exp_h, blk->moe_expert_fc2_b[e], H_dim, ffn, N_tok);

        if (ops->moe_scaled_accumulate) {
            int target = e;
            void *args[] = {&d_output, &d_exp_o, &d_moe_idx, &d_moe_w,
                            &target, &N_tok, &H_dim_arg, &top_k};
            cuLaunchKernel(ops->moe_scaled_accumulate, acc_grid, 1, 1,
                           256, 1, 1, 0, stream, args, NULL);
        } else {
            /* CPU fallback */
            cuStreamSynchronize(stream);
            size_t bytes = (size_t)N_tok * H_dim * sizeof(float);
            float *out = (float *)malloc(bytes);
            float *acc = (float *)malloc(bytes);
            int *idx = (int *)malloc((size_t)N_tok * top_k * sizeof(int));
            float *w  = (float *)malloc((size_t)N_tok * top_k * sizeof(float));
            cuMemcpyDtoH(out, d_exp_o, bytes);
            cuMemcpyDtoH(acc, d_output, bytes);
            cuMemcpyDtoH(idx, d_moe_idx, (size_t)N_tok*top_k*sizeof(int));
            cuMemcpyDtoH(w,   d_moe_w,   (size_t)N_tok*top_k*sizeof(float));
            for (int t = 0; t < N_tok; t++) {
                float ww = 0.0f;
                for (int k = 0; k < top_k; k++)
                    if (idx[t*top_k+k] == e) { ww = w[t*top_k+k]; break; }
                if (ww == 0.0f) continue;
                for (int j = 0; j < H_dim; j++)
                    acc[t*H_dim+j] += ww * out[t*H_dim+j];
            }
            cuMemcpyHtoDAsync(d_output, acc, bytes, stream);
            cuStreamSynchronize(stream);
            free(out); free(acc); free(idx); free(w);
        }
    }

    /* Step 5: Add shared expert output (reuse d_exp_h/d_exp_o buffers).
     * Same shape as a single MoE expert; per-row FP8 wins on N_tok=4097
     * (BF16-pipe MT4 needs %256 alignment). */
    op_gemm_qw(ops, stream, d_exp_h, blk->moe_shared_fc1_w,
               blk->moe_shared_fc1_w_fp8, blk->moe_shared_fc1_w_scale,
               d_input, blk->moe_shared_fc1_b, ffn, H_dim, N_tok);
    op_gelu(ops, stream, d_exp_h, N_tok * ffn);
    op_gemm_qw(ops, stream, d_exp_o, blk->moe_shared_fc2_w,
               blk->moe_shared_fc2_w_fp8, blk->moe_shared_fc2_w_scale,
               d_exp_h, blk->moe_shared_fc2_b, H_dim, ffn, N_tok);
    op_add(ops, stream, d_output, d_exp_o, N_tok * H_dim);
}

/* Stage 2: Single DiT forward pass
 *
 * Fix 1: U-Net skip connections (blocks 11-20)
 * Fix 2: MoE in blocks 15-20
 * Fix 3: Prepend timestep token -> seq_len = N+1, strip at final layer
 * Fix 4: TimestepEmbedder uses GELU (not SiLU)
 *
 * Input:  d_latents [4096, 64] F32, timestep (scalar), d_context [1370, 1024] F32
 * Output: d_output [4096, 64] F32 */
static void run_dit_forward(cuda_hy3d_runner *r, CUdeviceptr d_latents,
                             float timestep, CUdeviceptr d_context,
                             CUdeviceptr d_output, const char *pass_tag) {
    hy3d_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int N = DIT_INPUT_SIZE;       /* 4096 */
    const int C = DIT_IN_CHANNELS;      /* 64 */
    const int H_dim = DIT_HIDDEN;       /* 2048 */
    const int heads = DIT_HEADS;        /* 16 */
    const int hd = DIT_HEAD_DIM;        /* 128 */
    const int ffn = DIT_FFN;            /* 8192 */
    const int ctx_len = DINO_SEQ_LEN;   /* 1370 */
    const int N1 = N + 1;              /* 4097 (with prepended timestep token) */

    /* Scratch layout (using N1 = N+1 for transformer sequence length):
     * 0: hidden [(N1) * H_dim]         -- main hidden state
     * 1: Q/K/V [3 * N1 * H_dim]        -- QKV workspace
     * 2: attn_out [N1 * H_dim]
     * 3: mlp/moe scratch [see below]
     * 4: normed [N1 * H_dim]
     * 5: t_emb [H_dim] + t_mlp [ffn]
     * 6: cross_Q [N1 * H_dim] + cat_buf [N1 * 2*H_dim] + skip_values stack
     * 7: moe scratch [N1*(8+ffn+H_dim)*2+...] */

    /* MLP and MoE never active simultaneously — share scratch[3].
     * MLP needs: N1 * ffn  (134MB)
     * MoE needs: gate[N1*8] + expert_h[N1*ffn] + accum[N1*H_dim]  (168MB)
     *   (process one expert at a time, reuse expert_h for shared expert) */
    size_t mlp_sz = (size_t)N1 * ffn * sizeof(float);
    /* +(top_k * 2) for moe_idx[N1, top_k] (int) + moe_w[N1, top_k] (float) */
    size_t moe_scratch_sz = (size_t)N1 * (DIT_N_EXPERTS + ffn + H_dim + DIT_MOE_TOP_K * 2) * sizeof(float);
    size_t mlp_moe_sz = mlp_sz > moe_scratch_sz ? mlp_sz : moe_scratch_sz;
    /* For concat: [N1 * 2*H_dim] */
    size_t cat_buf_sz = (size_t)N1 * 2 * H_dim * sizeof(float);
    /* For cross-attention K,V: [ctx_len * H_dim] each, reused per block */
    size_t ca_kv_sz = (size_t)ctx_len * H_dim * sizeof(float);

    /* Memory-tight scratch layout — 4 GPU buffers, ~392MB total
     * 0: hidden [N1 * H_dim]                                = 33.5 MB
     * 1: QKV/MLP/MoE (shared) [max(3*N1*H_dim, moe_sz)]     = 168 MB
     * 2: attn_out/normed [2 * N1 * H_dim]                    = 67 MB
     * 3: cross_Q + cat_buf + ca_K + ca_V + t_emb             = ~123 MB
     *
     * scratch[1] is shared: QKV during attention, MLP/MoE during FFN.
     * scratch[2] holds both attn_out and normed (non-overlapping within each phase).
     */
    size_t qkv_sz = (size_t)3 * N1 * H_dim * sizeof(float);
    size_t shared1_sz = qkv_sz > mlp_moe_sz ? qkv_sz : mlp_moe_sz;
    /* buf3 layout: d_temb[H_dim] + d_tmlp[ffn] + d_cross_Q[N1*H_dim] + cat_buf[N1*2*H_dim]
     *              + ca_K[ctx*H_dim] + ca_V[ctx*H_dim] + split_V[N1*H_dim] */
    size_t buf3_sz = (size_t)(H_dim + ffn) * sizeof(float)
                   + (size_t)N1 * H_dim * sizeof(float)
                   + cat_buf_sz + 2 * ca_kv_sz
                   + (size_t)N1 * H_dim * sizeof(float); /* space for split V output */
    ensure_scratch(r, 0, (size_t)N1 * H_dim * sizeof(float));
    ensure_scratch(r, 1, shared1_sz);
    ensure_scratch(r, 2, (size_t)2 * N1 * H_dim * sizeof(float));
    ensure_scratch(r, 3, buf3_sz);

    CUdeviceptr d_hidden  = r->scratch[0];
    CUdeviceptr d_qkv     = r->scratch[1];  /* also used as d_mlp during MLP/MoE phase */
    CUdeviceptr d_mlp     = r->scratch[1];  /* shares with d_qkv */
    CUdeviceptr d_attn    = r->scratch[2];
    CUdeviceptr d_normed  = r->scratch[2] + (size_t)N1 * H_dim * sizeof(float);
    CUdeviceptr d_temb    = r->scratch[3];
    CUdeviceptr d_tmlp    = d_temb + (size_t)H_dim * sizeof(float);
    CUdeviceptr d_cross_Q = d_tmlp + (size_t)ffn * sizeof(float);
    CUdeviceptr d_cat_buf = d_cross_Q + (size_t)N1 * H_dim * sizeof(float);
    CUdeviceptr d_ca_K    = d_cat_buf + cat_buf_sz;
    CUdeviceptr d_ca_V    = d_ca_K + ca_kv_sz;
    CUdeviceptr d_moe_scratch = r->scratch[1]; /* shared with d_qkv/d_mlp */

    /* Skip stack: GPU-resident (was CPU RAM with sync DtoH/HtoD per push/pop;
     * ~370MB on GPU — trivially affordable, kills 1260 syncs + 40GB PCIe). */
    size_t skip_entry_sz = (size_t)N1 * H_dim * sizeof(float);
    int skip_stack_depth = DIT_HALF_DEPTH + 1;
    CUdeviceptr d_skip_stack = 0;
    if (cuMemAlloc(&d_skip_stack, (size_t)skip_stack_depth * skip_entry_sz) != CUDA_SUCCESS) {
        fprintf(stderr, "HY3D: failed to allocate %.1f MB GPU skip stack\n",
                (double)(skip_stack_depth * skip_entry_sz) / (1024.0 * 1024.0));
        return -1;
    }

    /* 1. Embed latents: [N, C] -> [N, H_dim] (into a temp buffer, not d_hidden yet) */
    /* We put the embedded result into d_normed temporarily */
    op_gemm(ops, stream, d_normed, r->dit_x_emb_w, d_latents, r->dit_x_emb_b,
            H_dim, C, N);

    /* 2. Fix 4: Timestep embedding with GELU (not SiLU)
     * sinusoidal -> Linear(2048 -> 8192) -> GELU -> Linear(8192 -> 2048) */
    op_timestep_embed(ops, stream, d_temb, timestep, H_dim);
    op_gemm(ops, stream, d_tmlp, r->dit_t_mlp0_w, d_temb, r->dit_t_mlp0_b,
            ffn, H_dim, 1);
    op_gelu(ops, stream, d_tmlp, ffn);  /* Fix 4: GELU, not SiLU */
    op_gemm(ops, stream, d_temb, r->dit_t_mlp2_w, d_tmlp, r->dit_t_mlp2_b,
            H_dim, ffn, 1);

    /* 3. Fix 3: Prepend timestep token to sequence
     * c = t_emb [1, H_dim]
     * x = d_normed [N, H_dim]   (embedded latents)
     * d_hidden = cat([c, x], dim=0)  -> [N+1, H_dim] */
    op_concat_first(ops, stream, d_hidden, d_temb, d_normed, N, H_dim);

    /* Now d_hidden has shape [N1, H_dim] = [4097, 2048] */
    if (r->verbose > 1) {
        cuStreamSynchronize(stream);
        float *hcpu = (float *)malloc((size_t)N1 * H_dim * sizeof(float));
        cuMemcpyDtoH(hcpu, d_hidden, (size_t)N1 * H_dim * sizeof(float));
        float mn = hcpu[0], mx = hcpu[0], sm = 0;
        for (int j = 0; j < N1 * H_dim; j++) {
            if (hcpu[j] < mn) mn = hcpu[j];
            if (hcpu[j] > mx) mx = hcpu[j];
            sm += hcpu[j];
        }
        float mean = sm / (float)(N1 * H_dim);
        float var = 0;
        for (int j = 0; j < N1 * H_dim; j++) { float d = hcpu[j] - mean; var += d*d; }
        fprintf(stderr, "  after_embed: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                mean, sqrtf(var / (float)(N1 * H_dim)), mn, mx);
        /* Token 0 (timestep) */
        float t0_mn = hcpu[0], t0_mx = hcpu[0], t0_sm = 0;
        for (int j = 0; j < H_dim; j++) {
            if (hcpu[j] < t0_mn) t0_mn = hcpu[j];
            if (hcpu[j] > t0_mx) t0_mx = hcpu[j];
            t0_sm += hcpu[j];
        }
        float t0_mean = t0_sm / H_dim;
        float t0_var = 0;
        for (int j = 0; j < H_dim; j++) { float d = hcpu[j] - t0_mean; t0_var += d*d; }
        fprintf(stderr, "    token[0]: mean=%.6f std=%.6f\n",
                t0_mean, sqrtf(t0_var / H_dim));
        float t1_sm = 0;
        for (int j = H_dim; j < 2*H_dim; j++) t1_sm += hcpu[j];
        float t1_mean = t1_sm / H_dim;
        float t1_var = 0;
        for (int j = H_dim; j < 2*H_dim; j++) { float d = hcpu[j] - t1_mean; t1_var += d*d; }
        fprintf(stderr, "    token[1]: mean=%.6f std=%.6f\n",
                t1_mean, sqrtf(t1_var / H_dim));
        free(hcpu);
    }

    if (r->verbose > 1)
        hy3d_dbg_dump_tagged_npy(stream, d_context, ctx_len, DIT_CONTEXT_DIM,
                                 "cuda_dit_context", pass_tag);

    /* Skip value stack pointer: blocks 0..DIT_HALF_DEPTH save hidden states */
    int skip_sp = 0;

    /* 4. Transformer blocks */
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        dit_block_gpu *blk = &r->dit_blocks[bi];

        /* Fix 1: Skip connection (blocks 11-20, layer > depth//2)
         * Before self-attention: pop skip value, concat, linear project, norm
         * Skip values stored in CPU RAM to save GPU memory */
        if (blk->use_skip && skip_sp > 0) {
            skip_sp--;
            CUdeviceptr d_skip_entry = d_skip_stack + (size_t)skip_sp * skip_entry_sz;

            /* cat = concat([skip_value, x], dim=-1)  -> [N1, 2*H_dim] */
            op_concat_last_dim(ops, stream, d_cat_buf, d_skip_entry, d_hidden, N1, H_dim);

            /* x = skip_linear(cat)  -> [N1, H_dim] */
            op_gemm(ops, stream, d_hidden, blk->skip_linear_w, d_cat_buf,
                    blk->skip_linear_b, H_dim, 2 * H_dim, N1);

            /* x = skip_norm(x) */
            op_layernorm(ops, stream, d_hidden, d_hidden,
                         blk->skip_norm_w, blk->skip_norm_b, N1, H_dim);
        }

        /* Save hidden state for skip connection (blocks 0..DIT_HALF_DEPTH) */
        if (bi <= DIT_HALF_DEPTH) {
            CUdeviceptr d_skip_entry = d_skip_stack + (size_t)skip_sp * skip_entry_sz;
            cuMemcpyDtoDAsync(d_skip_entry, d_hidden, skip_entry_sz, stream);
            skip_sp++;
        }

        /* === Self-attention ===
         * PyTorch Attention does: cat(to_q(x), to_k(x), to_v(x), dim=-1)
         *   → view(1, -1, H, 3*HD) → split(HD, dim=-1) → q_norm/k_norm → sdpa
         * The view+split interleaves Q/K/V per head. We replicate by using a
         * fused QKV weight [3*dim, dim] = concat(to_q.w, to_k.w, to_v.w) and
         * computing a single GEMM → [N1, 3*dim], then split_qkv_interleaved. */
        op_layernorm(ops, stream, d_normed, d_hidden, blk->norm1_w, blk->norm1_b, N1, H_dim);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_normed, N1, H_dim, "cuda_b0_norm1", pass_tag);
        }

        /* Fused QKV GEMM: [N1, dim] @ [3*dim, dim]^T → [N1, 3*dim] */
        op_gemm_qw_dit(ops, stream, d_qkv, blk->sa_qkv_w,
                   blk->sa_qkv_w_fp8, blk->sa_qkv_w_scale,
                   d_normed, 0, 3 * H_dim, H_dim, N1);

        /* Split interleaved QKV → per-head Q, K, V each [N1, dim]
         * IMPORTANT: d_qkv and d_mlp alias scratch[1], so split output must NOT
         * overlap with d_qkv input. Use:
         *   Q → scratch[2] (d_attn, free at this point)
         *   K → scratch[4] (d_normed, free at this point)
         *   V → end of scratch[3] (d_temb buffer has 123MB, V needs 33.5MB at offset ~90MB) */
        CUdeviceptr d_Q = d_attn;    /* scratch[2] */
        CUdeviceptr d_K = d_normed;  /* scratch[4] */
        CUdeviceptr d_V = d_ca_V + ca_kv_sz; /* after ca_V in scratch[3], ~100MB offset */
        op_split_qkv(ops, stream, d_Q, d_K, d_V, d_qkv, N1, heads, hd);

        /* Debug: compare Q/K/V after interleave */
        if (r->verbose > 1 && bi == 0) {
            cuStreamSynchronize(stream);
            float qv[8], kv[8], vv2[4];
            cuMemcpyDtoH(qv, d_Q, 8 * sizeof(float));
            cuMemcpyDtoH(kv, d_K, 8 * sizeof(float));
            cuMemcpyDtoH(vv2, d_V, 4 * sizeof(float));
            fprintf(stderr, "    b0 Q_interleaved[0,0:8]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    qv[0],qv[1],qv[2],qv[3],qv[4],qv[5],qv[6],qv[7]);
            fprintf(stderr, "    b0 K_interleaved[0,0:8]: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    kv[0],kv[1],kv[2],kv[3],kv[4],kv[5],kv[6],kv[7]);
            fprintf(stderr, "    b0 V_interleaved[0,0:4]: %.6f %.6f %.6f %.6f\n",
                    vv2[0],vv2[1],vv2[2],vv2[3]);
            /* Check raw QKV at position 256 (should be Q_raw head 2) */
            float raw[4];
            cuMemcpyDtoH(raw, d_qkv + 256*sizeof(float), 4*sizeof(float));
            fprintf(stderr, "    b0 QKV_raw[256:260]: %.6f %.6f %.6f %.6f\n",
                    raw[0],raw[1],raw[2],raw[3]);
            /* Check where V actually reads from */
            cuMemcpyDtoH(raw, d_qkv + 0*3*H_dim*sizeof(float) + 0*3*hd*sizeof(float) + 2*hd*sizeof(float), 4*sizeof(float));
            fprintf(stderr, "    b0 QKV[tok0,h0,2HD:2HD+4]: %.6f %.6f %.6f %.6f\n",
                    raw[0],raw[1],raw[2],raw[3]);
        }

        /* QK RMSNorm */
        if (blk->sa_q_norm_w)
            op_rms_norm(ops, stream, d_Q, blk->sa_q_norm_w, N1, heads, hd, H_dim);
        if (blk->sa_k_norm_w)
            op_rms_norm(ops, stream, d_K, blk->sa_k_norm_w, N1, heads, hd, H_dim);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_normed, N1, H_dim, "cuda_b0_attn1_in", pass_tag);
        }

        if (r->verbose > 1 && bi == 0) {
            cuStreamSynchronize(stream);
            float qn[4], kn[4], vv[4];
            cuMemcpyDtoH(qn, d_Q, 4 * sizeof(float));
            cuMemcpyDtoH(kn, d_K, 4 * sizeof(float));
            cuMemcpyDtoH(vv, d_V, 4 * sizeof(float));
            fprintf(stderr, "    b0 Q_normed[0,0:4]: %.6f %.6f %.6f %.6f\n", qn[0],qn[1],qn[2],qn[3]);
            fprintf(stderr, "    b0 K_normed[0,0:4]: %.6f %.6f %.6f %.6f\n", kn[0],kn[1],kn[2],kn[3]);
            fprintf(stderr, "    b0 V[0,0:4]:        %.6f %.6f %.6f %.6f\n", vv[0],vv[1],vv[2],vv[3]);
            /* V stats */
            float *vc = (float*)malloc((size_t)N1*H_dim*4);
            cuMemcpyDtoH(vc, d_V, (size_t)N1*H_dim*4);
            float vs=0; for(int j=0;j<N1*H_dim;j++) vs+=vc[j]*vc[j];
            fprintf(stderr, "    b0 V rms: %.6f\n", sqrtf(vs/(N1*H_dim)));
            free(vc);
        }

        op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, N1, H_dim, heads, hd);

        op_gemm_qw_dit(ops, stream, d_normed, blk->sa_out_w,
                   blk->sa_out_w_fp8, blk->sa_out_w_scale,
                   d_attn, blk->sa_out_b, H_dim, H_dim, N1);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_normed, N1, H_dim, "cuda_b0_attn1_out", pass_tag);
        }
        op_add(ops, stream, d_hidden, d_normed, N1 * H_dim);

        if (r->verbose > 1 && bi == 0) {
            cuStreamSynchronize(stream);
            /* Check attention output (before adding residual) */
            float *t = (float *)malloc((size_t)N1*H_dim*4);
            cuMemcpyDtoH(t, d_attn, (size_t)N1*H_dim*4);
            float sm=0,mn2=t[0],mx2=t[0];
            for (int j=0;j<N1*H_dim;j++){if(t[j]<mn2)mn2=t[j];if(t[j]>mx2)mx2=t[j];sm+=t[j];}
            float m=sm/(N1*H_dim),v=0;
            for(int j=0;j<N1*H_dim;j++){float d=t[j]-m;v+=d*d;}
            fprintf(stderr, "    b0 attn_raw: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                    m, sqrtf(v/(N1*H_dim)), mn2, mx2);
            /* Check sa output projection (d_normed after GEMM, before residual add) */
            cuMemcpyDtoH(t, d_normed, (size_t)N1*H_dim*4);
            sm=0;mn2=t[0];mx2=t[0];
            for (int j=0;j<N1*H_dim;j++){if(t[j]<mn2)mn2=t[j];if(t[j]>mx2)mx2=t[j];sm+=t[j];}
            m=sm/(N1*H_dim);v=0;
            for(int j=0;j<N1*H_dim;j++){float d=t[j]-m;v+=d*d;}
            fprintf(stderr, "    b0 sa_proj:  mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                    m, sqrtf(v/(N1*H_dim)), mn2, mx2);
            /* After residual add */
            cuMemcpyDtoH(t, d_hidden, (size_t)N1*H_dim*4);
            sm=0;mn2=t[0];mx2=t[0];
            for (int j=0;j<N1*H_dim;j++){if(t[j]<mn2)mn2=t[j];if(t[j]>mx2)mx2=t[j];sm+=t[j];}
            m=sm/(N1*H_dim);v=0;
            for(int j=0;j<N1*H_dim;j++){float d=t[j]-m;v+=d*d;}
            fprintf(stderr, "    b0 after_sa: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                    m, sqrtf(v/(N1*H_dim)), mn2, mx2);
            free(t);
        }

        /* === Cross-attention === */
        op_layernorm(ops, stream, d_normed, d_hidden, blk->norm2_w, blk->norm2_b, N1, H_dim);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_normed, N1, H_dim, "cuda_b0_norm2", pass_tag);
        }

        /* Q from hidden */
        op_gemm_qw_dit(ops, stream, d_cross_Q, blk->ca_q_w,
                   blk->ca_q_w_fp8, blk->ca_q_w_scale,
                   d_normed, 0, H_dim, H_dim, N1);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_cross_Q, N1, H_dim,
                                     "cuda_b0_attn2_to_q_raw", pass_tag);
        }
        if (blk->ca_q_norm_w)
            op_rms_norm(ops, stream, d_cross_Q, blk->ca_q_norm_w, N1, heads, hd, H_dim);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_cross_Q, N1, H_dim,
                                     "cuda_b0_attn2_to_q", pass_tag);
        }

        /* Reference-faithful K/V path: compute them separately, then compare or
         * combine later if needed. */
        const char *cpu_kv_env = getenv("HY3D_CPU_DIT_KV_GEMM");
        if (cpu_kv_env && cpu_kv_env[0] && cpu_kv_env[0] != '0') {
            float *h_k = (float *)malloc((size_t)ctx_len * H_dim * sizeof(float));
            float *h_v = (float *)malloc((size_t)ctx_len * H_dim * sizeof(float));
            float *h_ctx = (float *)malloc((size_t)ctx_len * DIT_CONTEXT_DIM * sizeof(float));
            if (h_k && h_v && h_ctx) {
                cuMemcpyDtoH(h_ctx, d_context, (size_t)ctx_len * DIT_CONTEXT_DIM * sizeof(float));
                /* Upload weights into host-visible buffers for a verification fallback. */
                float *w_k = (float *)malloc((size_t)H_dim * DIT_CONTEXT_DIM * sizeof(float));
                float *w_v = (float *)malloc((size_t)H_dim * DIT_CONTEXT_DIM * sizeof(float));
                if (w_k && w_v) {
                    if (hy3d_copy_gemm_weights_to_host_f32(w_k, blk->ca_k_w,
                                                           H_dim, DIT_CONTEXT_DIM,
                                                           r->ops.use_f32_gemm) == 0 &&
                        hy3d_copy_gemm_weights_to_host_f32(w_v, blk->ca_v_w,
                                                           H_dim, DIT_CONTEXT_DIM,
                                                           r->ops.use_f32_gemm) == 0) {
                        cpu_gemm_f32(h_k, w_k, h_ctx, H_dim, DIT_CONTEXT_DIM, ctx_len);
                        cpu_gemm_f32(h_v, w_v, h_ctx, H_dim, DIT_CONTEXT_DIM, ctx_len);
                        if (r->verbose > 1 && bi == 0) {
                            const char *dir = getenv("HY3D_DUMP_DIR");
                            if (dir && *dir) {
                                char path[1024];
                                char name[256];
                                hy3d_make_dbg_name(name, sizeof(name), "cpu_b0_attn2_to_k", pass_tag);
                                snprintf(path, sizeof(path), "%s/%s", dir, name);
                                cpu_dump_npy_f32_2d(path, h_k, ctx_len, H_dim);
                                hy3d_make_dbg_name(name, sizeof(name), "cpu_b0_attn2_to_v", pass_tag);
                                snprintf(path, sizeof(path), "%s/%s", dir, name);
                                cpu_dump_npy_f32_2d(path, h_v, ctx_len, H_dim);
                            }
                        }
                        cuMemcpyHtoDAsync(d_ca_K, h_k, (size_t)ctx_len * H_dim * sizeof(float), stream);
                        cuMemcpyHtoDAsync(d_ca_V, h_v, (size_t)ctx_len * H_dim * sizeof(float), stream);
                    } else {
                        fprintf(stderr, "HY3D: failed to stage attn2.to_k/to_v weights for CPU fallback\n");
                    }
                    free(w_k);
                    free(w_v);
                }
            }
            free(h_k);
            free(h_v);
            free(h_ctx);
        } else {
            op_gemm_qw_dit(ops, stream, d_ca_K, blk->ca_k_w,
                       blk->ca_k_w_fp8, blk->ca_k_w_scale,
                       d_context, 0, H_dim, DIT_CONTEXT_DIM, ctx_len);
            op_gemm_qw_dit(ops, stream, d_ca_V, blk->ca_v_w,
                       blk->ca_v_w_fp8, blk->ca_v_w_scale,
                       d_context, 0, H_dim, DIT_CONTEXT_DIM, ctx_len);
        }
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_ca_K, ctx_len, H_dim,
                                     "cuda_b0_attn2_to_k", pass_tag);
            hy3d_dbg_dump_tagged_npy(stream, d_ca_V, ctx_len, H_dim,
                                     "cuda_b0_attn2_to_v", pass_tag);
            hy3d_dbg_dump_tagged_npy(stream, d_cross_Q, N1, H_dim,
                                     "cuda_b0_attn2_q_norm", pass_tag);
        }

        /* PyTorch CrossAttention does:
         *   kv = cat(to_k(y), to_v(y), dim=-1)
         *   kv = kv.view(B, S, heads, 2*hd)
         *   k, v = split(kv, hd, dim=-1)
         * before k_norm / SDPA. Rebuild that interleaved per-head KV layout so
         * our K/V grouping matches the reference path. */
        op_concat_last_dim(ops, stream, d_qkv, d_ca_K, d_ca_V, ctx_len, H_dim);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_qkv, ctx_len, 2 * H_dim,
                                     "cuda_b0_attn2_kv_gemm", pass_tag);
        }
        op_split_kv(ops, stream, d_ca_K, d_ca_V, d_qkv, ctx_len, heads, hd);

        if (blk->ca_k_norm_w)
            op_rms_norm(ops, stream, d_ca_K, blk->ca_k_norm_w,
                        ctx_len, heads, hd, H_dim);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_ca_K, ctx_len, H_dim,
                                     "cuda_b0_attn2_k_norm", pass_tag);
        }

        op_cross_attn(ops, stream, d_attn, d_cross_Q, d_ca_K, d_ca_V,
                      N1, ctx_len, H_dim, heads, hd);

        op_gemm_qw_dit(ops, stream, d_normed, blk->ca_out_w,
                   blk->ca_out_w_fp8, blk->ca_out_w_scale,
                   d_attn, blk->ca_out_b, H_dim, H_dim, N1);
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_attn, N1, H_dim,
                                     "cuda_b0_attn2_out", pass_tag);
            hy3d_dbg_dump_tagged_npy(stream, d_normed, N1, H_dim,
                                     "cuda_b0_attn2_out_proj", pass_tag);
        }
        op_add(ops, stream, d_hidden, d_normed, N1 * H_dim);

        if (r->verbose > 1 && bi == 0) {
            cuStreamSynchronize(stream);
            float sm = 0; float *t2 = (float *)malloc((size_t)N1*H_dim*4);
            cuMemcpyDtoH(t2, d_hidden, (size_t)N1*H_dim*4);
            float mn2=t2[0], mx2=t2[0];
            for (int j=0;j<N1*H_dim;j++){if(t2[j]<mn2)mn2=t2[j];if(t2[j]>mx2)mx2=t2[j];sm+=t2[j];}
            float m2 = sm/(N1*H_dim); float v2=0;
            for(int j=0;j<N1*H_dim;j++){float d=t2[j]-m2;v2+=d*d;}
            fprintf(stderr, "    b0 after_ca: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
                    m2, sqrtf(v2/(N1*H_dim)), mn2, mx2);
            free(t2);
        }

        /* === MLP or MoE === */
        if (blk->norm3_w) {
            op_layernorm(ops, stream, d_normed, d_hidden, blk->norm3_w, blk->norm3_b, N1, H_dim);
        } else {
            cuMemcpyDtoDAsync(d_normed, d_hidden, (size_t)N1 * H_dim * sizeof(float), stream);
        }
        if (r->verbose > 1 && bi == 0) {
            hy3d_dbg_dump_tagged_npy(stream, d_normed, N1, H_dim, "cuda_b0_norm3", pass_tag);
        }

        if (blk->use_moe) {
            /* Fix 2: MoE with 8 experts + shared expert, top-2 gating.
             * Bug fix: d_moe_out must NOT alias d_moe_scratch. d_mlp and
             * d_moe_scratch both point to scratch[1]; d_exp_h inside
             * run_dit_moe sits at offset 131KB (size ~134MB) and overlaps
             * d_moe_out [0..33.5MB] at scratch[1] base, corrupting the
             * accumulator. Use d_attn (scratch[2], free during MLP phase). */
            CUdeviceptr d_moe_out = d_attn;
            run_dit_moe(r, blk, d_normed, d_moe_out, N1, d_moe_scratch);
            op_add(ops, stream, d_hidden, d_moe_out, N1 * H_dim);
        } else {
            /* Regular MLP: fc1 -> GELU -> fc2 */
            op_gemm_qw_dit(ops, stream, d_mlp, blk->mlp_fc1_w,
                       blk->mlp_fc1_w_fp8, blk->mlp_fc1_w_scale,
                       d_normed, blk->mlp_fc1_b, ffn, H_dim, N1);
            op_gelu(ops, stream, d_mlp, N1 * ffn);
            op_gemm_qw_dit(ops, stream, d_normed, blk->mlp_fc2_w,
                       blk->mlp_fc2_w_fp8, blk->mlp_fc2_w_scale,
                       d_mlp, blk->mlp_fc2_b, H_dim, ffn, N1);
            if (r->verbose > 1 && bi == 0) {
                hy3d_dbg_dump_tagged_npy(stream, d_normed, N1, H_dim, "cuda_b0_mlp_out", pass_tag);
            }
            op_add(ops, stream, d_hidden, d_normed, N1 * H_dim);
        }

        /* Per-block hidden dump (for error localization vs PyTorch) */
        if (bi == 0 || bi == 5 || bi == 10 || bi == 11 ||
            bi == 14 || bi == 15 || bi == 20) {
            char fname[64];
            if (pass_tag && *pass_tag) snprintf(fname, sizeof(fname), "cuda_dit_block_%d_%s.npy", bi, pass_tag);
            else snprintf(fname, sizeof(fname), "cuda_dit_block_%d.npy", bi);
            hy3d_dbg_dump_npy(stream, d_hidden, N1, H_dim, fname);
        }

        /* Per-block debug stats */
        if (r->verbose > 1) {
            cuStreamSynchronize(stream);
            float stats[4] = {0}; /* min, max, mean, std */
            float *hcpu = (float *)malloc((size_t)N1 * H_dim * sizeof(float));
            cuMemcpyDtoH(hcpu, d_hidden, (size_t)N1 * H_dim * sizeof(float));
            float mn = hcpu[0], mx = hcpu[0], sm = 0;
            for (int j = 0; j < N1 * H_dim; j++) {
                if (hcpu[j] < mn) mn = hcpu[j];
                if (hcpu[j] > mx) mx = hcpu[j];
                sm += hcpu[j];
            }
            float mean = sm / (float)(N1 * H_dim);
            float var = 0;
            for (int j = 0; j < N1 * H_dim; j++) {
                float d = hcpu[j] - mean; var += d * d;
            }
            float std = sqrtf(var / (float)(N1 * H_dim));
            fprintf(stderr, "  block %2d: mean=%.6f std=%.6f min=%.6f max=%.6f%s%s\n",
                    bi, mean, std, mn, mx,
                    blk->use_moe ? " [MoE]" : "", blk->use_skip ? " [skip]" : "");
            free(hcpu);
        }
    }

    /* 5. Final layer: strip timestep token, LN, Linear -> output [N, C]
     * Fix 3: Strip the first token (timestep) -> [N, H_dim] */
    op_strip_first(ops, stream, d_normed, d_hidden, N1, H_dim);
    /* d_normed is now [N, H_dim] */

    /* Apply final LN */
    CUdeviceptr d_ln_out = d_attn; /* reuse attn scratch, only need [N, H_dim] */
    op_layernorm(ops, stream, d_ln_out, d_normed,
                 r->dit_final_ln_w, r->dit_final_ln_b, N, H_dim);

    /* Final linear: [N, H_dim] -> [N, C] */
    op_gemm(ops, stream, d_output, r->dit_final_linear_w, d_ln_out,
            r->dit_final_linear_b, C, H_dim, N);

    cuMemFree(d_skip_stack);
}

/* Stage 3: ShapeVAE single transformer block */
static void run_vae_block(cuda_hy3d_runner *r, vae_block_gpu *b,
                           CUdeviceptr d_in, CUdeviceptr d_out,
                           CUdeviceptr d_scratch) {
    hy3d_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int N = VAE_NUM_LATENTS;
    const int W = VAE_WIDTH;
    const int H = VAE_HEADS;
    const int HD = VAE_HEAD_DIM;
    const int MLP = 4 * W;

    /* Scratch layout within d_scratch:
     * ln1_out [N*W], qkv [N*3*W], Q [N*W], K [N*W], V [N*W],
     * attn_out [N*W], proj [N*W], res1 [N*W],
     * ln2_out [N*W], mlp_h [N*MLP], mlp_out [N*W] */
    size_t off = 0;
    CUdeviceptr d_ln1   = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_qkv   = d_scratch + off; off += (size_t)N * 3 * W * sizeof(float);
    CUdeviceptr d_Q     = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_K     = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_V     = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_aout  = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_proj  = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_res1  = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_ln2   = d_scratch + off; off += (size_t)N * W * sizeof(float);
    CUdeviceptr d_mlph  = d_scratch + off; off += (size_t)N * MLP * sizeof(float);
    CUdeviceptr d_mlpo  = d_scratch + off;

    /* LN1 */
    op_layernorm(ops, stream, d_ln1, d_in, b->ln1_w, b->ln1_b, N, W);

    /* Fused QKV projection */
    op_gemm_qw_dit(ops, stream, d_qkv, b->qkv_w, b->qkv_w_fp8, b->qkv_w_scale,
                   d_ln1, 0, 3 * W, W, N);

    /* Split interleaved QKV */
    op_split_qkv(ops, stream, d_Q, d_K, d_V, d_qkv, N, H, HD);

    /* QK normalization */
    if (b->use_qk_norm) {
        op_qk_layernorm(ops, stream, d_Q, b->q_norm_w, b->q_norm_b, N, H, HD, W);
        op_qk_layernorm(ops, stream, d_K, b->k_norm_w, b->k_norm_b, N, H, HD, W);
    }

    /* Self-attention */
    op_self_attn(ops, stream, d_aout, d_Q, d_K, d_V, N, W, H, HD);

    /* Output projection */
    op_gemm_qw_dit(ops, stream, d_proj, b->proj_w, b->proj_w_fp8, b->proj_w_scale,
                   d_aout, b->proj_b, W, W, N);

    /* Residual 1: res1 = input + proj */
    cuMemcpyDtoDAsync(d_res1, d_in, (size_t)N * W * sizeof(float), stream);
    op_add(ops, stream, d_res1, d_proj, N * W);

    /* LN2 -> MLP -> Residual 2 */
    op_layernorm(ops, stream, d_ln2, d_res1, b->ln2_w, b->ln2_b, N, W);
    op_gemm_qw_dit(ops, stream, d_mlph, b->mlp_fc_w, b->mlp_fc_w_fp8, b->mlp_fc_w_scale,
                   d_ln2, b->mlp_fc_b, MLP, W, N);
    op_gelu(ops, stream, d_mlph, N * MLP);
    op_gemm_qw_dit(ops, stream, d_mlpo, b->mlp_proj_w, b->mlp_proj_w_fp8, b->mlp_proj_w_scale,
                   d_mlph, b->mlp_proj_b, W, MLP, N);

    /* Output = res1 + mlp_out */
    cuMemcpyDtoDAsync(d_out, d_res1, (size_t)N * W * sizeof(float), stream);
    op_add(ops, stream, d_out, d_mlpo, N * W);
}

/* Stage 3: ShapeVAE decode + SDF query
 * Input:  d_latents [4096, 64] F32
 * Output: sdf_grid [grid_res^3] F32 (on CPU) */
static void run_shapevae(cuda_hy3d_runner *r, CUdeviceptr d_latents,
                          int grid_res, float *sdf_out) {
    hy3d_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int N = VAE_NUM_LATENTS;
    const int E = VAE_EMBED_DIM;
    const int W = VAE_WIDTH;

    /* Allocate decoder buffers */
    CUdeviceptr d_dec_a = gpu_alloc((size_t)N * W * sizeof(float));
    CUdeviceptr d_dec_b = gpu_alloc((size_t)N * W * sizeof(float));

    /* VAE block scratch size (generous) */
    size_t block_scratch = (size_t)N * (W * 12 + 4 * W * 4) * sizeof(float);
    CUdeviceptr d_block_scratch = gpu_alloc(block_scratch);

    /* Post-KL projection: [N, E] -> [N, W] */
    op_gemm(ops, stream, d_dec_a, r->vae_post_kl_w, d_latents, r->vae_post_kl_b, W, E, N);

    /* Run transformer blocks */
    CUdeviceptr d_cur = d_dec_a;
    CUdeviceptr d_next = d_dec_b;
    for (int i = 0; i < VAE_DEC_LAYERS; i++) {
        run_vae_block(r, &r->vae_blocks[i], d_cur, d_next, d_block_scratch);
        CUdeviceptr tmp = d_cur; d_cur = d_next; d_next = tmp;
    }

    /* d_cur now contains decoded latents [N, W] */
    /* Query SDF at grid points in batches */
    int total_points = grid_res * grid_res * grid_res;
    int batch_size = 8192;  /* 8K points per batch (fits in 8GB GPU) */
    float bounds[6] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

    /* Allocate GPU buffers for SDF query */
    CUdeviceptr d_coords = gpu_alloc((size_t)batch_size * 3 * sizeof(float));
    CUdeviceptr d_fourier = gpu_alloc((size_t)batch_size * VAE_FOURIER_DIM * sizeof(float));
    CUdeviceptr d_query_proj = gpu_alloc((size_t)batch_size * W * sizeof(float));
    CUdeviceptr d_sdf_out = gpu_alloc((size_t)batch_size * sizeof(float));

    /* Geometry decoder scratch */
    size_t geo_scratch_sz = (size_t)batch_size * (W * 10 + 4 * W * 4) * sizeof(float);
    CUdeviceptr d_geo_scratch = gpu_alloc(geo_scratch_sz);

    float dx = (bounds[3] - bounds[0]) / (float)(grid_res - 1);
    float dy = (bounds[4] - bounds[1]) / (float)(grid_res - 1);
    float dz = (bounds[5] - bounds[2]) / (float)(grid_res - 1);

    for (int start = 0; start < total_points; start += batch_size) {
        int count = (start + batch_size <= total_points) ? batch_size : (total_points - start);

        /* Generate 3D coordinates on CPU and upload */
        float *coords = (float *)malloc((size_t)count * 3 * sizeof(float));
        for (int i = 0; i < count; i++) {
            int idx = start + i;
            int iz = idx % grid_res;
            int iy = (idx / grid_res) % grid_res;
            int ix = idx / (grid_res * grid_res);
            coords[i*3+0] = bounds[0] + ix * dx;
            coords[i*3+1] = bounds[1] + iy * dy;
            coords[i*3+2] = bounds[2] + iz * dz;
        }
        cuMemcpyHtoDAsync(d_coords, coords, (size_t)count * 3 * sizeof(float), stream);
        /* sync before free: HtoDAsync reads coords on the stream */
        cuStreamSynchronize(stream);
        free(coords);

        /* Fourier embedding */
        op_fourier_embed(ops, stream, d_fourier, d_coords, r->vae_fourier_freqs,
                         count, VAE_NUM_FREQS, VAE_FOURIER_DIM);

        /* Query projection: [count, 51] -> [count, W] */
        op_gemm(ops, stream, d_query_proj, r->vae_geo.query_proj_w, d_fourier,
                r->vae_geo.query_proj_b, W, VAE_FOURIER_DIM, count);

        /* Cross-attention with decoded latents */
        vae_geo_decoder_gpu *g = &r->vae_geo;

        /* LN on queries and latents */
        size_t geo_off = 0;
        CUdeviceptr d_g_ln1  = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_ln2  = d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        CUdeviceptr d_g_Q    = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_KV   = d_geo_scratch + geo_off; geo_off += (size_t)N * 2 * W * sizeof(float);
        CUdeviceptr d_g_K    = d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        CUdeviceptr d_g_V    = d_geo_scratch + geo_off; geo_off += (size_t)N * W * sizeof(float);
        CUdeviceptr d_g_aout = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_proj = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_res  = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_ln3  = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_mlph = d_geo_scratch + geo_off; geo_off += (size_t)count * 4 * W * sizeof(float);
        CUdeviceptr d_g_mlpo = d_geo_scratch + geo_off; geo_off += (size_t)count * W * sizeof(float);
        CUdeviceptr d_g_post = d_geo_scratch + geo_off;

        op_layernorm(ops, stream, d_g_ln1, d_query_proj, g->ln1_w, g->ln1_b, count, W);
        op_layernorm(ops, stream, d_g_ln2, d_cur, g->ln2_w, g->ln2_b, N, W);

        /* Q from queries, KV from latents */
        op_gemm_qw_dit(ops, stream, d_g_Q, g->c_q_w, g->c_q_w_fp8, g->c_q_w_scale,
                       d_g_ln1, 0, W, W, count);
        op_gemm_qw_dit(ops, stream, d_g_KV, g->c_kv_w, g->c_kv_w_fp8, g->c_kv_w_scale,
                       d_g_ln2, 0, 2 * W, W, N);

        /* Split KV */
        op_split_kv(ops, stream, d_g_K, d_g_V, d_g_KV, N, VAE_HEADS, VAE_HEAD_DIM);

        /* QK norm */
        if (g->use_qk_norm) {
            op_qk_layernorm(ops, stream, d_g_Q, g->q_norm_w, g->q_norm_b,
                            count, VAE_HEADS, VAE_HEAD_DIM, W);
            op_qk_layernorm(ops, stream, d_g_K, g->k_norm_w, g->k_norm_b,
                            N, VAE_HEADS, VAE_HEAD_DIM, W);
        }

        /* Cross-attention */
        op_cross_attn(ops, stream, d_g_aout, d_g_Q, d_g_K, d_g_V,
                      count, N, W, VAE_HEADS, VAE_HEAD_DIM);

        /* Output projection + residual */
        op_gemm_qw_dit(ops, stream, d_g_proj, g->c_proj_w, g->c_proj_w_fp8, g->c_proj_w_scale,
                       d_g_aout, g->c_proj_b, W, W, count);
        cuMemcpyDtoDAsync(d_g_res, d_query_proj, (size_t)count * W * sizeof(float), stream);
        op_add(ops, stream, d_g_res, d_g_proj, count * W);

        /* MLP block */
        op_layernorm(ops, stream, d_g_ln3, d_g_res, g->ln3_w, g->ln3_b, count, W);
        op_gemm_qw_dit(ops, stream, d_g_mlph, g->mlp_fc_w, g->mlp_fc_w_fp8, g->mlp_fc_w_scale,
                       d_g_ln3, g->mlp_fc_b, 4 * W, W, count);
        op_gelu(ops, stream, d_g_mlph, count * 4 * W);
        op_gemm_qw_dit(ops, stream, d_g_mlpo, g->mlp_proj_w, g->mlp_proj_w_fp8, g->mlp_proj_w_scale,
                       d_g_mlph, g->mlp_proj_b, W, 4 * W, count);
        cuMemcpyDtoDAsync(d_g_post, d_g_res, (size_t)count * W * sizeof(float), stream);
        op_add(ops, stream, d_g_post, d_g_mlpo, count * W);

        /* Post LN */
        if (g->ln_post_w) {
            op_layernorm(ops, stream, d_g_ln1, d_g_post, g->ln_post_w, g->ln_post_b, count, W);
        } else {
            cuMemcpyDtoDAsync(d_g_ln1, d_g_post, (size_t)count * W * sizeof(float), stream);
        }

        /* Final output projection: [count, W] -> [count, 1] */
        op_gemm(ops, stream, d_sdf_out, g->output_w, d_g_ln1, g->output_b, 1, W, count);

        /* Download SDF values */
        cuMemcpyDtoHAsync(sdf_out + start, d_sdf_out,
                          (size_t)count * sizeof(float), stream);
    }

    cuStreamSynchronize(stream);

    /* Cleanup */
    cuMemFree(d_dec_a);
    cuMemFree(d_dec_b);
    cuMemFree(d_block_scratch);
    cuMemFree(d_coords);
    cuMemFree(d_fourier);
    cuMemFree(d_query_proj);
    cuMemFree(d_sdf_out);
    cuMemFree(d_geo_scratch);
}

/* ======================================================================== */
/* Public API                                                               */
/* ======================================================================== */

cuda_hy3d_runner *cuda_hy3d_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "HY3D: cuew init failed\n");
        return NULL;
    }
    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "HY3D: cuInit failed\n");
        return NULL;
    }

    cuda_hy3d_runner *r = (cuda_hy3d_runner *)calloc(1, sizeof(cuda_hy3d_runner));
    r->verbose = verbose;

    CU_CHECK_NULL(cuDeviceGet(&r->device, device_id));
    CU_CHECK_NULL(cuCtxCreate(&r->ctx, 0, r->device));
    CU_CHECK_NULL(cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING));

    if (verbose) {
        char name[256];
        cuDeviceGetName(name, sizeof(name), r->device);
        fprintf(stderr, "HY3D: using GPU %d: %s\n", device_id, name);
    }

    if (hy3d_compile_kernels(r) != 0) {
        fprintf(stderr, "HY3D: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    /* Tensor-core dispatch policy (mirrors qimg's QIMG_* env vars).
     * Defaults are chosen to match qimg's recommended high-performance path:
     *   HY3D_BF16_ATTN=1 (sm>=80, head_dim=128 only — DiT)
     *   HY3D_FP8_GEMM=1  (sm>=89, GEMMs aligned to 16/32/64)
     * Verification escape hatches stay opt-in. */
    {
        int sm = r->sm_version;
        const char *env;

        env = getenv("HY3D_BF16_ATTN");
        r->ops.use_bf16_attn = (sm >= 80 && r->ops.flash_attn_bf16 && r->ops.cast_f32_to_bf16)
                               ? ((env && env[0] == '0') ? 0 : 1) : 0;

        env = getenv("HY3D_FP8_ATTN");
        r->ops.use_fp8_attn = (env && env[0] != '0' && sm >= 89 &&
                               r->ops.flash_attn_fp8 && r->ops.quantize_fp8 &&
                               r->ops.reduce_max_abs) ? 1 : 0;

        env = getenv("HY3D_FP8_GEMM");
        r->ops.use_fp8_gemm = (sm >= 89 && r->ops.gemm_fp8_perrow &&
                               r->ops.reduce_max_abs_per_row && r->ops.quantize_fp8)
                              ? ((env && env[0] == '0') ? 0 : 1) : 0;

        /* Default OFF: extending FP8 to DiT attention QKV/out + MLP fc1/fc2
         * causes per-tensor weight scale drift to accumulate across blocks
         * to the point of producing an all-negative SDF (verified at 5 and
         * 30 steps). MoE FP8 stays on the use_fp8_gemm flag and is safe. */
        env = getenv("HY3D_FP8_GEMM_ATTN_MLP");
        r->ops.use_fp8_gemm_attn_mlp = (env && env[0] != '0' &&
                                        r->ops.use_fp8_gemm) ? 1 : 0;

        /* BF16-pipe-scaled is the safe-and-fast path for DiT attn/MLP:
         * BF16 acts × FP8 weights with inline LUT dequant + per-tensor scale
         * applied at writeback. Avoids the per-tensor scale drift that breaks
         * the pure FP8 path on attn/MLP, while still cutting wall-clock vs
         * the F16-tiled fallback. Default ON for sm>=80 when prequant is
         * available. */
        env = getenv("HY3D_BF16_GEMM");
        r->ops.use_bf16_gemm = (sm >= 80 && r->ops.gemm_bf16_pipe_scaled &&
                                r->ops.use_fp8_gemm)
                               ? ((env && env[0] == '0') ? 0 : 1) : 0;

        env = getenv("HY3D_DISABLE_MT4");
        r->ops.disable_mt4 = (env && env[0] != '0') ? 1 : 0;

        if (verbose) {
            fprintf(stderr,
                "HY3D: tc dispatch — bf16_attn=%d fp8_attn=%d fp8_gemm=%d fp8_gemm_attn_mlp=%d bf16_gemm=%d mt4=%s (sm_%d)\n",
                r->ops.use_bf16_attn, r->ops.use_fp8_attn, r->ops.use_fp8_gemm,
                r->ops.use_fp8_gemm_attn_mlp,
                r->ops.use_bf16_gemm, r->ops.disable_mt4 ? "off" : "on", sm);
        }
    }

    return r;
}

void cuda_hy3d_set_dump(cuda_hy3d_runner *r, int every, int grid, const char *prefix) {
    if (!r) return;
    r->dump_every = every > 0 ? every : 0;
    r->dump_grid  = grid  > 0 ? grid  : 48;
    r->dump_prefix[0] = '\0';
    if (prefix && *prefix) {
        size_t n = strlen(prefix);
        if (n >= sizeof(r->dump_prefix)) n = sizeof(r->dump_prefix) - 1;
        memcpy(r->dump_prefix, prefix, n);
        r->dump_prefix[n] = '\0';
    }
}

void cuda_hy3d_set_latent_dump(cuda_hy3d_runner *r, const char *steps_csv, const char *prefix) {
    if (!r) return;
    r->latent_dump_count = 0;
    r->latent_dump_prefix[0] = '\0';
    if (prefix && *prefix) {
        size_t n = strlen(prefix);
        if (n >= sizeof(r->latent_dump_prefix)) n = sizeof(r->latent_dump_prefix) - 1;
        memcpy(r->latent_dump_prefix, prefix, n);
        r->latent_dump_prefix[n] = '\0';
    }
    if (!steps_csv || !*steps_csv) return;

    const char *p = steps_csv;
    while (*p && r->latent_dump_count < HY3D_MAX_LATENT_DUMP_STEPS) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0 && v <= 1000000) {
            int step = (int)v;
            int seen = 0;
            for (int i = 0; i < r->latent_dump_count; i++) {
                if (r->latent_dump_steps[i] == step) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) r->latent_dump_steps[r->latent_dump_count++] = step;
        }
        p = end;
        while (*p == ' ' || *p == ',') p++;
    }
}

void cuda_hy3d_set_velocity_dump(cuda_hy3d_runner *r, const char *steps_csv, const char *prefix) {
    if (!r) return;
    r->velocity_dump_count = 0;
    r->velocity_dump_prefix[0] = '\0';
    if (prefix && *prefix) {
        size_t n = strlen(prefix);
        if (n >= sizeof(r->velocity_dump_prefix)) n = sizeof(r->velocity_dump_prefix) - 1;
        memcpy(r->velocity_dump_prefix, prefix, n);
        r->velocity_dump_prefix[n] = '\0';
    }
    if (!steps_csv || !*steps_csv) return;

    const char *p = steps_csv;
    while (*p && r->velocity_dump_count < HY3D_MAX_LATENT_DUMP_STEPS) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0 && v <= 1000000) {
            int step = (int)v;
            int seen = 0;
            for (int i = 0; i < r->velocity_dump_count; i++) {
                if (r->velocity_dump_steps[i] == step) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) r->velocity_dump_steps[r->velocity_dump_count++] = step;
        }
        p = end;
        while (*p == ' ' || *p == ',') p++;
    }
}

int cuda_hy3d_set_init_latents(cuda_hy3d_runner *r, const float *latents, int n) {
    if (!r || !latents) return -1;
    const int expected = DIT_INPUT_SIZE * DIT_IN_CHANNELS;
    if (n != expected) return -1;
    float *tmp = (float *)malloc((size_t)n * sizeof(float));
    if (!tmp) return -1;
    memcpy(tmp, latents, (size_t)n * sizeof(float));
    free(r->init_latents);
    r->init_latents = tmp;
    r->init_latents_n = n;
    return 0;
}

int cuda_hy3d_set_init_contexts(cuda_hy3d_runner *r,
                                const float *cond,
                                const float *uncond,
                                int n) {
    if (!r || !cond) return -1;
    const int expected = DINO_SEQ_LEN * DIT_CONTEXT_DIM;
    if (n != expected) return -1;
    float *c = (float *)malloc((size_t)n * sizeof(float));
    if (!c) return -1;
    memcpy(c, cond, (size_t)n * sizeof(float));

    float *u = NULL;
    if (uncond) {
        u = (float *)malloc((size_t)n * sizeof(float));
        if (!u) {
            free(c);
            return -1;
        }
        memcpy(u, uncond, (size_t)n * sizeof(float));
    }

    free(r->init_ctx_cond);
    free(r->init_ctx_uncond);
    r->init_ctx_cond = c;
    r->init_ctx_uncond = u;
    r->init_ctx_n = n;
    return 0;
}

void cuda_hy3d_set_f32_gemm(cuda_hy3d_runner *r, int enable) {
    if (!r) return;
    r->ops.use_f32_gemm = enable;
    if (r->verbose)
        fprintf(stderr, "HY3D: GEMM mode set to %s\n",
                enable ? "F32 (PyTorch-compatible)" : "F16 (default)");
}

void cuda_hy3d_set_bf16_attn(cuda_hy3d_runner *r, int enable) {
    if (!r) return;
    /* Only honour ON when the kernels were actually loaded. */
    if (enable) {
        r->ops.use_bf16_attn = (r->ops.flash_attn_bf16 && r->ops.cast_f32_to_bf16) ? 1 : 0;
    } else {
        r->ops.use_bf16_attn = 0;
    }
    if (r->verbose)
        fprintf(stderr, "HY3D: bf16_attn = %d\n", r->ops.use_bf16_attn);
}

void cuda_hy3d_set_fp8_attn(cuda_hy3d_runner *r, int enable) {
    if (!r) return;
    if (enable) {
        r->ops.use_fp8_attn = (r->ops.flash_attn_fp8 && r->ops.quantize_fp8 && r->ops.reduce_max_abs) ? 1 : 0;
    } else {
        r->ops.use_fp8_attn = 0;
    }
    if (r->verbose)
        fprintf(stderr, "HY3D: fp8_attn = %d\n", r->ops.use_fp8_attn);
}

void cuda_hy3d_set_fp8_gemm(cuda_hy3d_runner *r, int enable) {
    if (!r) return;
    if (enable) {
        r->ops.use_fp8_gemm = (r->ops.gemm_fp8_perrow && r->ops.reduce_max_abs_per_row) ? 1 : 0;
    } else {
        r->ops.use_fp8_gemm = 0;
    }
    if (r->verbose)
        fprintf(stderr, "HY3D: fp8_gemm = %d\n", r->ops.use_fp8_gemm);
}

void cuda_hy3d_set_fp8_gemm_attn_mlp(cuda_hy3d_runner *r, int enable) {
    if (!r) return;
    r->ops.use_fp8_gemm_attn_mlp = (enable && r->ops.use_fp8_gemm) ? 1 : 0;
    if (r->verbose)
        fprintf(stderr, "HY3D: fp8_gemm_attn_mlp = %d\n", r->ops.use_fp8_gemm_attn_mlp);
}

void cuda_hy3d_set_bf16_gemm(cuda_hy3d_runner *r, int enable) {
    if (!r) return;
    if (enable) {
        r->ops.use_bf16_gemm = (r->ops.gemm_bf16_pipe_scaled && r->ops.use_fp8_gemm) ? 1 : 0;
    } else {
        r->ops.use_bf16_gemm = 0;
    }
    if (r->verbose)
        fprintf(stderr, "HY3D: bf16_gemm = %d\n", r->ops.use_bf16_gemm);
}

int cuda_hy3d_load_weights(cuda_hy3d_runner *r,
                           const char *conditioner_path,
                           const char *model_path,
                           const char *vae_path) {
    if (!r) return -1;

    if (conditioner_path && load_dino_weights(r, conditioner_path) != 0)
        return -1;
    if (model_path && load_dit_weights(r, model_path) != 0)
        return -1;
    if (vae_path && load_vae_weights(r, vae_path) != 0)
        return -1;

    return 0;
}

hy3d_mesh cuda_hy3d_predict(cuda_hy3d_runner *r,
                            const uint8_t *rgb, int w, int h,
                            int n_steps, float guidance_scale,
                            int grid_res, uint32_t seed) {
    hy3d_ops *ops = &r->ops;
    CUstream stream = r->stream;
    hy3d_mesh result = {0};
    if (!r || !r->dino_loaded || !r->dit_loaded || !r->vae_loaded) {
        fprintf(stderr, "HY3D: runner not fully initialized\n");
        return result;
    }

    if (n_steps <= 0) n_steps = 30;
    if (guidance_scale <= 0.0f) guidance_scale = 7.5f;
    if (grid_res <= 0) grid_res = 256;

    struct timespec t0, t1, ts1, ts2, ts3, ts4;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* ---- Stage 1: DINOv2 image encoding ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 1 - DINOv2 encoding...\n");

    /* Upload and preprocess image: resize to 518x518, normalize with ImageNet stats */
    CUdeviceptr d_rgb = gpu_alloc((size_t)w * h * 3);
    cuMemcpyHtoDAsync(d_rgb, rgb, (size_t)w * h * 3, stream);

    CUdeviceptr d_image = gpu_alloc((size_t)3 * DINO_IMG_SIZE * DINO_IMG_SIZE * sizeof(float));
    {
        int dw = DINO_IMG_SIZE, dh = DINO_IMG_SIZE;
        float mean0 = 0.485f, mean1 = 0.456f, mean2 = 0.406f;
        float istd0 = 1.0f/0.229f, istd1 = 1.0f/0.224f, istd2 = 1.0f/0.225f;
        void *args[] = {&d_image, &d_rgb, &w, &h, &dw, &dh,
                        &mean0, &mean1, &mean2, &istd0, &istd1, &istd2};
        cuLaunchKernel(ops->resize_normalize,
                       (unsigned)((dw*dh+255)/256), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* DINOv2 forward */
    CUdeviceptr d_dino_out = gpu_alloc((size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    run_dinov2(r, d_image, d_dino_out);
    if (r->init_ctx_cond && r->init_ctx_n == DINO_SEQ_LEN * DIT_CONTEXT_DIM) {
        cuMemcpyHtoDAsync(d_dino_out, r->init_ctx_cond,
                          (size_t)r->init_ctx_n * sizeof(float), stream);
        if (r->verbose) fprintf(stderr, "HY3D: using user-provided conditional context\n");
    }
    if (r->verbose > 1) {
        hy3d_dbg_dump_npy(stream, d_dino_out, DINO_SEQ_LEN, DIT_CONTEXT_DIM, "cuda_dino_context_after_override.npy");
    }
    cuMemFree(d_rgb);
    cuMemFree(d_image);
    cuStreamSynchronize(stream);
    clock_gettime(CLOCK_MONOTONIC, &ts1);

    /* ---- Stage 2: DiT diffusion with flow matching ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 2 - DiT diffusion (%d steps)...\n", n_steps);

    /* K,V computed per-block inside run_dit_forward (saves GPU memory) */

    /* Create initial noise */
    int latent_size = DIT_INPUT_SIZE * DIT_IN_CHANNELS;
    float *noise_cpu = (float *)malloc((size_t)latent_size * sizeof(float));
    if (r->init_latents && r->init_latents_n == latent_size) {
        memcpy(noise_cpu, r->init_latents, (size_t)latent_size * sizeof(float));
        if (r->verbose) fprintf(stderr, "HY3D: using user-provided initial latents\n");
    } else {
        uint64_t rng_seed = seed ? (uint64_t)seed : (uint64_t)time(NULL);
        cuda_rng_fill_normal_f32(noise_cpu, (size_t)latent_size, rng_seed, 0);
    }

    CUdeviceptr d_latents = gpu_alloc((size_t)latent_size * sizeof(float));
    cuMemcpyHtoDAsync(d_latents, noise_cpu, (size_t)latent_size * sizeof(float), stream);
    /* HtoDAsync reads from noise_cpu on the stream — sync before freeing
     * or the GPU copy races against libc recycling the host buffer. */
    cuStreamSynchronize(stream);
    free(noise_cpu);

    CUdeviceptr d_pred_cond = gpu_alloc((size_t)latent_size * sizeof(float));
    CUdeviceptr d_pred_uncond = gpu_alloc((size_t)latent_size * sizeof(float));
    CUdeviceptr d_pred_combined = gpu_alloc((size_t)latent_size * sizeof(float));

    /* Create zero context for unconditional pass */
    CUdeviceptr d_uncond_ctx = gpu_alloc((size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float));
    if (r->init_ctx_uncond && r->init_ctx_n == DINO_SEQ_LEN * DIT_CONTEXT_DIM) {
        cuMemcpyHtoDAsync(d_uncond_ctx, r->init_ctx_uncond,
                          (size_t)r->init_ctx_n * sizeof(float), stream);
        if (r->verbose) fprintf(stderr, "HY3D: using user-provided unconditional context\n");
    } else {
        cuMemsetD8Async(d_uncond_ctx, 0, (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float), stream);
    }

    /* Flow matching schedule — matches Hunyuan3DDiTFlowMatchingPipeline.__call__
     * (hy3dshape/pipelines.py line 736-764), NOT the parent class loop:
     *
     *   sigmas = np.linspace(0, 1, n_steps)          # ASCENDING from 0
     *   scheduler.set_timesteps(sigmas=sigmas)       # timesteps = sigmas*1000
     *   for t in timesteps:
     *       t_model = t / num_train_timesteps        # rescales back to [0, 1]
     *       v = model(latents, t_model, ctx)
     *       latents = latents + (sigmas[i+1] - sigmas[i]) * v
     *
     * Key facts:
     *   - The DiT is fed t in [0, 1], NOT [1, 1000]. The TimestepEmbedder in
     *     hunyuandit.py consumes it raw (no internal /1000).
     *   - Sigmas ASCEND 0 -> 1 with a sentinel 1.0 at position n_steps (so the
     *     final step's dsigma is 0 and does nothing).
     *   - Update direction is prev = sample + (sn - s) * v, i.e. additive
     *     with positive dsigma. Our op_euler_step does x -= dt*v, so we pass
     *     dt = sigma - sigma_next (negative) to get x = x - (s-sn)*v
     *                                              = x + (sn-s)*v. */
    for (int step = 0; step < n_steps; step++) {
        float sigma, sigma_next;
        if (n_steps == 1) {
            sigma = 0.0f;
            sigma_next = 1.0f;
        } else {
            sigma = (float)step / (float)(n_steps - 1);
            sigma_next = (step == n_steps - 1)
                              ? 1.0f  /* scheduler sentinel */
                              : (float)(step + 1) / (float)(n_steps - 1);
        }
        float dt         = sigma - sigma_next; /* negative except last step */

        /* DiT input timestep equals sigma (model was trained in [0, 1]). */
        float model_t = sigma;

        if (r->verbose && (step % 5 == 0 || step == n_steps - 1))
            fprintf(stderr, "  step %d/%d (t=%.4f)\n", step + 1, n_steps, model_t);

        if (r->latent_dump_count > 0 && hy3d_should_dump_latent(r, step + 1)) {
            hy3d_dump_latent(stream, d_latents, DIT_INPUT_SIZE, DIT_IN_CHANNELS,
                             r->latent_dump_prefix, step + 1);
            if (r->verbose)
                fprintf(stderr, "    dumped latent step %d -> %s_%03d.npy\n",
                        step + 1, r->latent_dump_prefix, step + 1);
        }

        /* Conditional pass */
        run_dit_forward(r, d_latents, model_t, d_dino_out, d_pred_cond, "cond");

        /* Unconditional pass */
        run_dit_forward(r, d_latents, model_t, d_uncond_ctx, d_pred_uncond, "uncond");

        /* CFG combination */
        op_cfg_combine(ops, stream, d_pred_combined, d_pred_cond, d_pred_uncond,
                       guidance_scale, latent_size);
        if (r->velocity_dump_count > 0 && hy3d_should_dump_velocity(r, step + 1)) {
            hy3d_dump_latent(stream, d_pred_combined, DIT_INPUT_SIZE, DIT_IN_CHANNELS,
                             r->velocity_dump_prefix, step + 1);
            if (r->verbose)
                fprintf(stderr, "    dumped velocity step %d -> %s_%03d.npy\n",
                        step + 1, r->velocity_dump_prefix, step + 1);
        }

        /* Euler step: x <- x - dt * v */
        op_euler_step(ops, stream, d_latents, d_pred_combined, dt, latent_size);

        /* Diagnostic per-step dump: decode current latent and write OBJ. */
        if (r->dump_every > 0 && ((step + 1) % r->dump_every == 0 ||
                                   step == n_steps - 1)) {
            int dg = r->dump_grid;
            int total = dg * dg * dg;
            float *sdf = (float *)malloc((size_t)total * sizeof(float));
            if (sdf) {
                run_shapevae(r, d_latents, dg, sdf);
                float mn = sdf[0], mx = sdf[0], sm = 0;
                int pos = 0, neg = 0;
                for (int i = 0; i < total; i++) {
                    float v = sdf[i];
                    if (v < mn) mn = v; if (v > mx) mx = v; sm += v;
                    if (v > 0) pos++; else if (v < 0) neg++;
                }
                mc_mesh mc = mc_marching_cubes(sdf, dg, dg, dg, 0.0f, NULL);
                char path[640];
                snprintf(path, sizeof(path), "%s_%03d.obj", r->dump_prefix, step + 1);
                if (mc.n_verts > 0) {
                    mc_write_obj(path, &mc);
                    if (r->verbose)
                        fprintf(stderr, "    dump step %d: sdf[%.3f,%.3f] pos=%d (%d v, %d t) -> %s\n",
                                step + 1, mn, mx, pos, mc.n_verts, mc.n_tris, path);
                } else if (r->verbose) {
                    fprintf(stderr, "    dump step %d: sdf[%.3f,%.3f] pos=%d neg=%d (empty mesh)\n",
                            step + 1, mn, mx, pos, neg);
                }
                free(mc.vertices);
                free(mc.triangles);
                free(sdf);
            }
        }
    }

    cuMemFree(d_pred_cond);
    cuMemFree(d_pred_uncond);
    cuMemFree(d_pred_combined);
    cuMemFree(d_uncond_ctx);
    cuMemFree(d_dino_out);
    cuStreamSynchronize(stream);
    clock_gettime(CLOCK_MONOTONIC, &ts2);

    /* ---- Stage 3: ShapeVAE decode + SDF query ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 3 - ShapeVAE decode (grid %d^3)...\n", grid_res);

    int total_pts = grid_res * grid_res * grid_res;
    float *sdf_grid = (float *)malloc((size_t)total_pts * sizeof(float));
    run_shapevae(r, d_latents, grid_res, sdf_grid);
    cuMemFree(d_latents);
    if (r->verbose) {
        float mn = sdf_grid[0], mx = sdf_grid[0], sm = 0;
        int pos = 0, neg = 0;
        for (int i = 0; i < total_pts; i++) {
            float v = sdf_grid[i];
            if (v < mn) mn = v; if (v > mx) mx = v; sm += v;
            if (v > 0) pos++; else if (v < 0) neg++;
        }
        fprintf(stderr, "HY3D: SDF stats: min=%.4f max=%.4f mean=%.4f pos=%d neg=%d\n",
                mn, mx, sm / total_pts, pos, neg);
    }

    cuStreamSynchronize(stream);
    clock_gettime(CLOCK_MONOTONIC, &ts3);

    /* ---- Stage 4: Marching cubes ---- */
    if (r->verbose) fprintf(stderr, "HY3D: Stage 4 - Marching cubes...\n");

    mc_mesh mc = mc_marching_cubes(sdf_grid, grid_res, grid_res, grid_res, 0.0f, NULL);
    free(sdf_grid);

    result.vertices = mc.vertices;
    result.triangles = mc.triangles;
    result.n_verts = mc.n_verts;
    result.n_tris = mc.n_tris;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    clock_gettime(CLOCK_MONOTONIC, &ts4);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (r->verbose) {
        double e1 = (ts1.tv_sec - t0.tv_sec)  + (ts1.tv_nsec - t0.tv_nsec) * 1e-9;
        double e2 = (ts2.tv_sec - ts1.tv_sec) + (ts2.tv_nsec - ts1.tv_nsec) * 1e-9;
        double e3 = (ts3.tv_sec - ts2.tv_sec) + (ts3.tv_nsec - ts2.tv_nsec) * 1e-9;
        double e4 = (ts4.tv_sec - ts3.tv_sec) + (ts4.tv_nsec - ts3.tv_nsec) * 1e-9;
        fprintf(stderr, "HY3D: stage timings — DINOv2=%.2fs DiT=%.2fs VAE=%.2fs MC=%.2fs\n",
                e1, e2, e3, e4);
        fprintf(stderr, "HY3D: done in %.2fs (%d verts, %d tris)\n",
                elapsed, result.n_verts, result.n_tris);
    }

    return result;
}

/* ======================================================================== */
/* Per-stage verification API                                               */
/* ======================================================================== */

int cuda_hy3d_run_dinov2(cuda_hy3d_runner *r,
                          const float *image_f32,
                          float *output) {
    if (!r || !r->dino_loaded) return -1;

    size_t img_bytes = (size_t)3 * DINO_IMG_SIZE * DINO_IMG_SIZE * sizeof(float);
    size_t out_bytes = (size_t)DINO_SEQ_LEN * DINO_HIDDEN * sizeof(float);

    CUdeviceptr d_image = gpu_alloc(img_bytes);
    CUdeviceptr d_out = gpu_alloc(out_bytes);

    cuMemcpyHtoDAsync(d_image, image_f32, img_bytes, r->stream);
    run_dinov2(r, d_image, d_out);
    cuMemcpyDtoHAsync(output, d_out, out_bytes, r->stream);
    cuStreamSynchronize(r->stream);

    cuMemFree(d_image);
    cuMemFree(d_out);
    return 0;
}

int cuda_hy3d_run_vae(cuda_hy3d_runner *r,
                       const float *latents,
                       int grid_res,
                       float *sdf_out) {
    if (!r || !r->vae_loaded) return -1;

    size_t lat_bytes = (size_t)VAE_NUM_LATENTS * VAE_EMBED_DIM * sizeof(float);
    CUdeviceptr d_latents = gpu_alloc(lat_bytes);
    cuMemcpyHtoDAsync(d_latents, latents, lat_bytes, r->stream);

    run_shapevae(r, d_latents, grid_res, sdf_out);

    cuMemFree(d_latents);
    return 0;
}

int cuda_hy3d_run_dit(cuda_hy3d_runner *r,
                       const float *latents,
                       float timestep,
                       const float *context,
                       float *output) {
    if (!r || !r->dit_loaded) return -1;

    size_t lat_bytes = (size_t)DIT_INPUT_SIZE * DIT_IN_CHANNELS * sizeof(float);
    size_t ctx_bytes = (size_t)DINO_SEQ_LEN * DIT_CONTEXT_DIM * sizeof(float);
    size_t out_bytes = lat_bytes;

    CUdeviceptr d_latents = gpu_alloc(lat_bytes);
    CUdeviceptr d_context = gpu_alloc(ctx_bytes);
    CUdeviceptr d_output  = gpu_alloc(out_bytes);

    cuMemcpyHtoDAsync(d_latents, latents, lat_bytes, r->stream);
    cuMemcpyHtoDAsync(d_context, context, ctx_bytes, r->stream);

    /* K,V computed per-block inside run_dit_forward (saves GPU memory) */
    run_dit_forward(r, d_latents, timestep, d_context, d_output, "single");

    cuMemcpyDtoHAsync(output, d_output, out_bytes, r->stream);
    cuStreamSynchronize(r->stream);

    cuMemFree(d_latents);
    cuMemFree(d_context);
    cuMemFree(d_output);
    return 0;
}

void cuda_hy3d_free(cuda_hy3d_runner *r) {
    if (!r) return;
    free(r->init_latents);
    r->init_latents = NULL;
    r->init_latents_n = 0;
    free(r->init_ctx_cond);
    free(r->init_ctx_uncond);
    r->init_ctx_cond = NULL;
    r->init_ctx_uncond = NULL;
    r->init_ctx_n = 0;
    r->velocity_dump_count = 0;
    r->latent_dump_count = 0;

    /* Free scratch buffers */
    for (int i = 0; i < 8; i++) {
        if (r->scratch[i]) cuMemFree(r->scratch[i]);
    }

    /* Free tensor-core dispatch scratch (BF16/FP8 attn buffers) */
    hy3d_ops_destroy(&r->ops);

    /* Free pre-computed K,V */
    for (int i = 0; i < DIT_DEPTH; i++) {
        if (r->dit_ca_K[i]) cuMemFree(r->dit_ca_K[i]);
        if (r->dit_ca_V[i]) cuMemFree(r->dit_ca_V[i]);
    }

    /* Free Fourier frequencies */
    if (r->vae_fourier_freqs) cuMemFree(r->vae_fourier_freqs);

    /* Note: individual weight buffers are GPU allocations that should also be freed.
     * For a full cleanup, iterate all CUdeviceptr fields. For now, destroying
     * the CUDA context reclaims all GPU memory. */

    if (r->module) cuModuleUnload(r->module);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->ctx) cuCtxDestroy(r->ctx);

    free(r);
}
