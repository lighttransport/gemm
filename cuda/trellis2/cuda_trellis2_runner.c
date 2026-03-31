/*
 * cuda_trellis2_runner.c - CUDA TRELLIS.2 Stage 1 via NVRTC-compiled kernels
 *
 * Pipeline: DINOv3 encoder -> DiT diffusion (flow matching) -> Decoder -> occupancy
 * Compiles with plain gcc (no nvcc). Uses cuew for dynamic CUDA/NVRTC loading.
 *
 * SPDX-License-Identifier: MIT
 */

#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
/* sparse3d.h provides sp3d_hash_build/free for shape decoder gather map */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"
#define T2_SHAPE_DEC_IMPLEMENTATION
#include "../../common/trellis2_shape_decoder.h"
#include "cuda_trellis2_runner.h"
#include "../cuew.h"
#include "../cuda_kernels_common.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

/* Include HY3D kernels for cross_attn, timestep_embed, euler_step, cfg_combine,
 * split_qkv, split_kv, broadcast_add, rms_norm — these are model-specific in HY3D
 * but reused here. We extract only the kernel source string (no ops header). */
#include "../hy3d/cuda_hy3d_kernels.h"

#include "cuda_trellis2_kernels.h"
#include "cuda_trellis2_ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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
/* GPU weight structures                                                    */
/* ======================================================================== */

typedef struct {
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr q_w, q_b, k_w, k_b, v_w, v_b;  /* separate Q/K/V for RoPE */
    CUdeviceptr q_norm_w, k_norm_w;
    CUdeviceptr out_w, out_b;
    CUdeviceptr ls1, ls2;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr fc1_w, fc1_b;
    CUdeviceptr fc2_w, fc2_b;
} dino_layer_gpu;

typedef struct {
    /* Self-attention */
    CUdeviceptr sa_qkv_w, sa_qkv_b;    /* [3*DIT_DIM, DIT_DIM], [3*DIT_DIM] */
    CUdeviceptr sa_q_norm, sa_k_norm;   /* [DIT_HEADS * DIT_HEAD_DIM] */
    CUdeviceptr sa_out_w, sa_out_b;
    /* Cross-attention */
    CUdeviceptr norm2_w, norm2_b;
    CUdeviceptr ca_q_w, ca_q_b;
    CUdeviceptr ca_kv_w, ca_kv_b;
    CUdeviceptr ca_q_norm, ca_k_norm;
    CUdeviceptr ca_out_w, ca_out_b;
    /* MLP */
    CUdeviceptr mlp_fc1_w, mlp_fc1_b;
    CUdeviceptr mlp_fc2_w, mlp_fc2_b;
    /* Per-block modulation bias */
    CUdeviceptr mod_bias;  /* [6*DIT_DIM] */
} dit_block_gpu;

/* CPU-side weight storage for layer streaming (offloading) */
typedef struct {
    void *sa_qkv_w, *sa_qkv_b, *sa_q_norm, *sa_k_norm, *sa_out_w, *sa_out_b;
    void *norm2_w, *norm2_b, *ca_q_w, *ca_q_b, *ca_kv_w, *ca_kv_b;
    void *ca_q_norm, *ca_k_norm, *ca_out_w, *ca_out_b;
    void *mlp_fc1_w, *mlp_fc1_b, *mlp_fc2_w, *mlp_fc2_b;
    void *mod_bias;
    /* Byte sizes for each weight */
    size_t sa_qkv_w_sz, sa_qkv_b_sz, sa_q_norm_sz, sa_k_norm_sz;
    size_t sa_out_w_sz, sa_out_b_sz, norm2_w_sz, norm2_b_sz;
    size_t ca_q_w_sz, ca_q_b_sz, ca_kv_w_sz, ca_kv_b_sz;
    size_t ca_q_norm_sz, ca_k_norm_sz, ca_out_w_sz, ca_out_b_sz;
    size_t mlp_fc1_w_sz, mlp_fc1_b_sz, mlp_fc2_w_sz, mlp_fc2_b_sz;
    size_t mod_bias_sz;
} dit_block_cpu;

/* Generic DiT model (shared by Stage 1 and Stage 2) */
typedef struct {
    int n_blocks;
    int model_channels;   /* D: 1536 */
    int n_heads;          /* 12 */
    int head_dim;         /* 128 */
    int ffn_hidden;       /* 8192 */
    int in_channels;      /* 8 for Stage 1, 32 for Stage 2, 64 for Stage 3 */
    int out_channels;     /* same as in_channels for Stage 1/2, 32 for Stage 3 */
    int cond_dim;         /* 1024 */
    CUdeviceptr t_fc1_w, t_fc1_b, t_fc2_w, t_fc2_b;
    CUdeviceptr mod_w, mod_b;
    CUdeviceptr x_emb_w, x_emb_b;
    CUdeviceptr out_w, out_b;
    dit_block_gpu *blocks;
    CUdeviceptr rope_cos, rope_sin;
    int n_rope_freqs, rope_axis_dim;
} dit_model_gpu;

typedef struct {
    CUdeviceptr conv_w, conv_b;
    CUdeviceptr gn1_w, gn1_b, conv1_w, conv1_b;
    CUdeviceptr gn2_w, gn2_b, conv2_w, conv2_b;
} dec_resblock_gpu;

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

struct cuda_trellis2_runner {
    CUdevice   device;
    CUcontext  context;
    CUstream   stream;
    CUmodule   module;
    t2_ops     ops;
    int        verbose;

    /* DINOv3 weights */
    CUdeviceptr dino_patch_w, dino_patch_b;
    CUdeviceptr dino_cls_token, dino_storage_tokens;
    CUdeviceptr dino_norm_w, dino_norm_b;
    dino_layer_gpu dino_layers[DINO_LAYERS];
    int dino_has_qk_norm, dino_has_layerscale;
    CUdeviceptr dino_rope_cos, dino_rope_sin;  /* [1024, 64] for 32x32 patches */

    /* Stage 1 DiT */
    CUdeviceptr dit_t_fc1_w, dit_t_fc1_b, dit_t_fc2_w, dit_t_fc2_b;
    CUdeviceptr dit_mod_w, dit_mod_b;
    CUdeviceptr dit_x_emb_w, dit_x_emb_b;
    CUdeviceptr dit_out_w, dit_out_b;
    dit_block_gpu dit_blocks[DIT_DEPTH];
    CUdeviceptr dit_rope_cos, dit_rope_sin;
    int dit_n_freqs, dit_axis_dim;
    int dit_use_f16;  /* 1 if DiT weights stored as F16 */
    dit_block_cpu dit_blocks_cpu[DIT_DEPTH]; /* CPU copies for layer streaming */

    /* Stage 2 shape flow DiT */
    dit_model_gpu stage2;
    dit_block_gpu stage2_blocks[DIT_DEPTH];
    dit_block_cpu stage2_blocks_cpu[DIT_DEPTH];
    int stage2_loaded;

    /* Stage 3 texture flow DiT (same architecture, in_channels=64) */
    dit_model_gpu stage3;
    dit_block_gpu stage3_blocks[DIT_DEPTH];
    dit_block_cpu stage3_blocks_cpu[DIT_DEPTH];
    int stage3_loaded;

    /* Layer streaming config (0 = all on GPU, >0 = max layers on GPU at once) */
    int max_gpu_layers;

    /* Decoder weights */
    CUdeviceptr dec_conv_in_w, dec_conv_in_b;
    dec_resblock_gpu dec_middle[2];
    dec_resblock_gpu dec_res16[2];
    CUdeviceptr dec_up1_w, dec_up1_b;
    dec_resblock_gpu dec_res32[2];
    CUdeviceptr dec_up2_w, dec_up2_b;
    dec_resblock_gpu dec_res64[2];
    CUdeviceptr dec_out_gn_w, dec_out_gn_b;
    CUdeviceptr dec_out_conv_w, dec_out_conv_b;

    /* Shape decoder (SC-VAE) */
    struct {
        CUdeviceptr from_latent_w, from_latent_b;  /* [1024, 32] */
        CUdeviceptr out_w, out_b;                   /* [7, 64] */
        struct {
            CUdeviceptr conv_w, conv_b;   /* [27, C, C] transposed for gather-GEMM */
            CUdeviceptr norm_w, norm_b;   /* [C] */
            CUdeviceptr mlp0_w, mlp0_b;   /* [4C, C] */
            CUdeviceptr mlp2_w, mlp2_b;   /* [C, 4C] */
        } convnext[4][16];  /* [stage][block] — max 16 blocks per stage */
        struct {
            CUdeviceptr norm1_w, norm1_b;
            CUdeviceptr conv1_w, conv1_b;  /* [27, C_out*8, C_in] transposed */
            CUdeviceptr conv2_w, conv2_b;  /* [27, C_out, C_out] transposed */
            CUdeviceptr subdiv_w, subdiv_b; /* [8, C_in] */
        } c2s[4];  /* one per stage */
        int n_convnext[4];  /* {4, 16, 8, 4} */
        int channels[5];    /* {1024, 512, 256, 128, 64} */
        int loaded;
    } shape_dec;

    /* CPU-side decoders for C2S operations */
    t2_shape_dec *shape_dec_cpu;
    t2_shape_dec *tex_dec_cpu;
    int tex_dec_loaded;

    /* Cross-attention KV cache (precomputed per-block, reused across timesteps) */
    CUdeviceptr ca_kv_cache_K[DIT_DEPTH];
    CUdeviceptr ca_kv_cache_V[DIT_DEPTH];
    int ca_kv_cache_n_blocks;
    int ca_kv_cache_model_id;  /* which model's weights were cached */
    int ca_kv_cache_valid;

    /* Scratch buffers (expanded to 12 for shape decoder use) */
    CUdeviceptr scratch[12];
    size_t      scratch_size[12];
};

static void ensure_scratch(cuda_trellis2_runner *r, int idx, size_t bytes) {
    if (r->scratch_size[idx] >= bytes) return;
    if (r->scratch[idx]) cuMemFree(r->scratch[idx]);
    CUresult err = cuMemAlloc(&r->scratch[idx], bytes);
    if (err != 0) {
        const char *name = NULL;
        cuGetErrorName(err, &name);
        size_t free_mem = 0, total_mem = 0;
        cuMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "T2: FATAL: cuMemAlloc(%zu bytes = %.1f MB) for scratch[%d] failed: %s "
                "(GPU free=%.0f MB / %.0f MB total)\n",
                bytes, bytes / 1048576.0, idx, name ? name : "?",
                free_mem / 1048576.0, total_mem / 1048576.0);
        r->scratch[idx] = 0;
        r->scratch_size[idx] = 0;
        return;
    }
    r->scratch_size[idx] = bytes;
}

/* ======================================================================== */
/* Layer streaming: upload/free individual blocks                           */
/* ======================================================================== */

/* Upload a single block's weights from CPU to GPU */
static void dit_block_upload_to_gpu(dit_block_gpu *gpu, const dit_block_cpu *cpu) {
    #define UP(field) if (cpu->field) gpu->field = cu_upload_raw(cpu->field, cpu->field##_sz); else gpu->field = 0
    UP(sa_qkv_w); UP(sa_qkv_b); UP(sa_q_norm); UP(sa_k_norm);
    UP(sa_out_w); UP(sa_out_b); UP(norm2_w); UP(norm2_b);
    UP(ca_q_w); UP(ca_q_b); UP(ca_kv_w); UP(ca_kv_b);
    UP(ca_q_norm); UP(ca_k_norm); UP(ca_out_w); UP(ca_out_b);
    UP(mlp_fc1_w); UP(mlp_fc1_b); UP(mlp_fc2_w); UP(mlp_fc2_b);
    UP(mod_bias);
    #undef UP
}

/* Free a block's GPU weights */
static void dit_block_free_gpu(dit_block_gpu *gpu) {
    #define FR(field) if (gpu->field) { cuMemFree(gpu->field); gpu->field = 0; }
    FR(sa_qkv_w); FR(sa_qkv_b); FR(sa_q_norm); FR(sa_k_norm);
    FR(sa_out_w); FR(sa_out_b); FR(norm2_w); FR(norm2_b);
    FR(ca_q_w); FR(ca_q_b); FR(ca_kv_w); FR(ca_kv_b);
    FR(ca_q_norm); FR(ca_k_norm); FR(ca_out_w); FR(ca_out_b);
    FR(mlp_fc1_w); FR(mlp_fc1_b); FR(mlp_fc2_w); FR(mlp_fc2_b);
    FR(mod_bias);
    #undef FR
}

/* Save CPU copy of data that was uploaded to GPU.
 * Returns malloc'd CPU copy (caller owns). */
static void *cpu_copy_from_gpu(CUdeviceptr d, size_t bytes) {
    if (!d || !bytes) return NULL;
    void *cpu = malloc(bytes);
    cuMemcpyDtoH(cpu, d, bytes);
    return cpu;
}

/* After loading a block to GPU, save CPU copies for streaming */
static void dit_block_save_cpu_copy(dit_block_cpu *cpu, const dit_block_gpu *gpu,
                                      int use_f16, int D, int H, int HD, int FFN, int COND_DIM) {
    /* Weight sizes depend on F16 vs F32 */
    size_t ws = use_f16 ? sizeof(uint16_t) : sizeof(float); /* weight element size */
    size_t bs = sizeof(float); /* bias always F32 */

    #define SV(field, sz) cpu->field##_sz = (sz); cpu->field = cpu_copy_from_gpu(gpu->field, (sz))
    SV(sa_qkv_w, (size_t)3*D*D*ws);  SV(sa_qkv_b, (size_t)3*D*bs);
    SV(sa_q_norm, (size_t)H*HD*bs);  SV(sa_k_norm, (size_t)H*HD*bs);
    SV(sa_out_w, (size_t)D*D*ws);    SV(sa_out_b, (size_t)D*bs);
    SV(norm2_w, (size_t)D*bs);       SV(norm2_b, (size_t)D*bs);
    SV(ca_q_w, (size_t)D*D*ws);      SV(ca_q_b, (size_t)D*bs);
    SV(ca_kv_w, (size_t)2*D*COND_DIM*ws); SV(ca_kv_b, (size_t)2*D*bs);
    SV(ca_q_norm, (size_t)H*HD*bs);  SV(ca_k_norm, (size_t)H*HD*bs);
    SV(ca_out_w, (size_t)D*D*ws);    SV(ca_out_b, (size_t)D*bs);
    SV(mlp_fc1_w, (size_t)FFN*D*ws); SV(mlp_fc1_b, (size_t)FFN*bs);
    SV(mlp_fc2_w, (size_t)D*FFN*ws); SV(mlp_fc2_b, (size_t)D*bs);
    SV(mod_bias, (size_t)6*D*bs);
    #undef SV
}

static void dit_block_cpu_free(dit_block_cpu *cpu) {
    free(cpu->sa_qkv_w); free(cpu->sa_qkv_b); free(cpu->sa_q_norm); free(cpu->sa_k_norm);
    free(cpu->sa_out_w); free(cpu->sa_out_b); free(cpu->norm2_w); free(cpu->norm2_b);
    free(cpu->ca_q_w); free(cpu->ca_q_b); free(cpu->ca_kv_w); free(cpu->ca_kv_b);
    free(cpu->ca_q_norm); free(cpu->ca_k_norm); free(cpu->ca_out_w); free(cpu->ca_out_b);
    free(cpu->mlp_fc1_w); free(cpu->mlp_fc1_b); free(cpu->mlp_fc2_w); free(cpu->mlp_fc2_b);
    free(cpu->mod_bias);
    memset(cpu, 0, sizeof(*cpu));
}

static void dbg4(const char *label, CUdeviceptr ptr, CUstream stream) {
    cuStreamSynchronize(stream);
    float d[4];
    cuMemcpyDtoH(d, ptr, 16);
    fprintf(stderr, "  %s: [:4]=%.6f %.6f %.6f %.6f\n", label, d[0], d[1], d[2], d[3]);
}

static double t2_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ======================================================================== */
/* Weight upload helpers                                                    */
/* ======================================================================== */

/* Upload safetensors tensor as F32 on GPU (handles F16, BF16, F32 source) */
static CUdeviceptr t2_upload_f32(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [MISSING] %s\n", name);
        return 0;
    }
    const char *dtype = safetensors_dtype(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    if (strcmp(dtype, "F32") == 0) {
        return cu_upload_raw(data, nbytes);
    }

    /* Convert to F32 on CPU, then upload */
    size_t n_elem;
    if (strcmp(dtype, "F16") == 0)      n_elem = nbytes / 2;
    else if (strcmp(dtype, "BF16") == 0) n_elem = nbytes / 2;
    else { fprintf(stderr, "  [SKIP] %s: unsupported dtype %s\n", name, dtype); return 0; }

    float *f32 = (float *)malloc(n_elem * sizeof(float));
    const uint16_t *src = (const uint16_t *)data;

    if (strcmp(dtype, "BF16") == 0) {
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else { /* F16 */
        for (size_t i = 0; i < n_elem; i++) {
            /* Simple F16->F32 */
            uint16_t h = src[i];
            uint32_t sign = ((uint32_t)h & 0x8000) << 16;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            uint32_t f;
            if (exp == 0) { if (mant == 0) f = sign; else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3ff; f = sign | ((exp + 127 - 15) << 23) | (mant << 13); } }
            else if (exp == 31) f = sign | 0x7f800000 | (mant << 13);
            else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            memcpy(&f32[i], &f, 4);
        }
    }

    CUdeviceptr d = cu_upload_raw(f32, n_elem * sizeof(float));
    free(f32);

    if (verbose >= 2) {
        const uint64_t *sh = safetensors_shape(st, idx);
        int nd = safetensors_ndims(st, idx);
        fprintf(stderr, "  %s: %s [", name, dtype);
        for (int d2 = 0; d2 < nd; d2++) fprintf(stderr, "%s%lu", d2?",":"", (unsigned long)sh[d2]);
        fprintf(stderr, "] -> F32 GPU (%.1f MB)\n", (float)(n_elem * 4) / (1024*1024));
    }
    return d;
}

/* Upload safetensors tensor as F16 on GPU (handles F16, BF16, F32 source).
 * Halves GPU memory compared to F32, enables MMA tensor core GEMM. */
static CUdeviceptr t2_upload_f16(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose) fprintf(stderr, "  [MISSING] %s\n", name);
        return 0;
    }
    const char *dtype = safetensors_dtype(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    if (strcmp(dtype, "F16") == 0) {
        /* Already F16 — upload directly */
        return cu_upload_raw(data, nbytes);
    }

    size_t n_elem;
    if (strcmp(dtype, "BF16") == 0)     n_elem = nbytes / 2;
    else if (strcmp(dtype, "F32") == 0) n_elem = nbytes / 4;
    else { fprintf(stderr, "  [SKIP] %s: unsupported dtype %s for F16\n", name, dtype); return 0; }

    uint16_t *f16 = (uint16_t *)malloc(n_elem * sizeof(uint16_t));
    if (strcmp(dtype, "BF16") == 0) {
        /* BF16 -> F16: convert via F32 intermediate */
        const uint16_t *src = (const uint16_t *)data;
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t bits = (uint32_t)src[i] << 16;
            float f; memcpy(&f, &bits, 4);
            /* F32 -> F16 conversion */
            uint32_t fb; memcpy(&fb, &f, 4);
            uint32_t sign = (fb >> 16) & 0x8000;
            int32_t exp = ((fb >> 23) & 0xff) - 127 + 15;
            uint32_t mant = (fb >> 13) & 0x3ff;
            if (exp <= 0) f16[i] = (uint16_t)sign;           /* underflow to zero */
            else if (exp >= 31) f16[i] = (uint16_t)(sign | 0x7c00); /* overflow to inf */
            else f16[i] = (uint16_t)(sign | (exp << 10) | mant);
        }
    } else { /* F32 -> F16 */
        const float *src = (const float *)data;
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t fb; memcpy(&fb, &src[i], 4);
            uint32_t sign = (fb >> 16) & 0x8000;
            int32_t exp = ((fb >> 23) & 0xff) - 127 + 15;
            uint32_t mant = (fb >> 13) & 0x3ff;
            if (exp <= 0) f16[i] = (uint16_t)sign;
            else if (exp >= 31) f16[i] = (uint16_t)(sign | 0x7c00);
            else f16[i] = (uint16_t)(sign | (exp << 10) | mant);
        }
    }

    CUdeviceptr d = cu_upload_raw(f16, n_elem * sizeof(uint16_t));
    free(f16);

    if (verbose >= 2) {
        const uint64_t *sh = safetensors_shape(st, idx);
        int nd = safetensors_ndims(st, idx);
        fprintf(stderr, "  %s: %s [", name, dtype);
        for (int d2 = 0; d2 < nd; d2++) fprintf(stderr, "%s%lu", d2?",":"", (unsigned long)sh[d2]);
        fprintf(stderr, "] -> F16 GPU (%.1f MB)\n", (float)(n_elem * 2) / (1024*1024));
    }
    return d;
}

/* ======================================================================== */
/* Init / Free                                                              */
/* ======================================================================== */

cuda_trellis2_runner *cuda_trellis2_init(int device_id, int verbose) {
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "T2: failed to load CUDA/NVRTC libraries\n");
        return NULL;
    }
    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "T2: cuInit failed (%d)\n", (int)err);
        return NULL;
    }

    cuda_trellis2_runner *r = (cuda_trellis2_runner *)calloc(1, sizeof(*r));
    r->verbose = verbose;

    CU_CHECK_NULL(cuDeviceGet(&r->device, device_id));
    CU_CHECK_NULL(cuCtxCreate(&r->context, 0, r->device));
    CU_CHECK_NULL(cuStreamCreate(&r->stream, CU_STREAM_NON_BLOCKING));

    /* Query GPU info */
    char name[256];
    cuDeviceGetName(name, sizeof(name), r->device);
    size_t mem;
    cuDeviceTotalMem(&mem, r->device);
    fprintf(stderr, "T2: GPU %d: %s (%.1f GB)\n", device_id, name, (double)mem / (1<<30));

    /* Compile kernels: common + HY3D shared ops + TRELLIS2-specific.
     * HY3D kernels end with "} extern C" so we strip the last line
     * and let trellis2 kernels close the block. */
    size_t hy3d_len = strlen(cuda_hy3d_specific_kernels);
    /* Find and trim the closing "}" from HY3D source */
    const char *hy3d_end = cuda_hy3d_specific_kernels + hy3d_len;
    while (hy3d_end > cuda_hy3d_specific_kernels && hy3d_end[-1] != '}')
        hy3d_end--;
    if (hy3d_end > cuda_hy3d_specific_kernels) hy3d_end--; /* skip the '}' itself */
    size_t hy3d_trimmed = (size_t)(hy3d_end - cuda_hy3d_specific_kernels);

    size_t total_len = strlen(cuda_kernels_common_src) + hy3d_trimmed
                     + strlen(cuda_trellis2_kernel_source) + 16;
    char *full_source = (char *)malloc(total_len);
    sprintf(full_source, "%s\n", cuda_kernels_common_src);
    size_t off = strlen(full_source);
    memcpy(full_source + off, cuda_hy3d_specific_kernels, hy3d_trimmed);
    off += hy3d_trimmed;
    sprintf(full_source + off, "\n%s", cuda_trellis2_kernel_source);

    int sm = cu_compile_kernels(&r->module, r->device, full_source, "trellis2",
                                verbose, "T2");
    free(full_source);
    if (sm < 0) {
        fprintf(stderr, "T2: kernel compilation failed\n");
        cuCtxDestroy(r->context);
        free(r);
        return NULL;
    }
    if (t2_ops_load(&r->ops, r->module, sm) != 0) {
        fprintf(stderr, "T2: failed to load kernel functions\n");
        cuModuleUnload(r->module);
        cuCtxDestroy(r->context);
        free(r);
        return NULL;
    }

    fprintf(stderr, "T2: kernels compiled for sm_%d\n", sm);
    return r;
}

void cuda_trellis2_set_f32_gemm(cuda_trellis2_runner *r, int enable) {
    r->ops.use_f32_gemm = enable;
}

void cuda_trellis2_set_max_gpu_layers(cuda_trellis2_runner *r, int n) {
    r->max_gpu_layers = n;
    if (n > 0)
        fprintf(stderr, "T2: layer streaming enabled: max %d layers on GPU\n", n);
}

void cuda_trellis2_free(cuda_trellis2_runner *r) {
    if (!r) return;
    for (int i = 0; i < 8; i++)
        if (r->scratch[i]) cuMemFree(r->scratch[i]);
    /* TODO: free all GPU weight buffers */
    cuModuleUnload(r->module);
    cuStreamDestroy(r->stream);
    cuCtxDestroy(r->context);
    free(r);
}

/* ======================================================================== */
/* Weight loading                                                           */
/* ======================================================================== */

static int load_dit_weights(cuda_trellis2_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading DiT from %s (%d tensors)\n", path, st->n_tensors);

    int v = r->verbose;
    int use_f16 = (r->ops.sm_version >= 70);

    /* Timestep MLP and modulation: always F32 (n_tok=1, modulation reads directly) */
    r->dit_t_fc1_w = t2_upload_f32(st, "t_embedder.mlp.0.weight", v);
    r->dit_t_fc1_b = t2_upload_f32(st, "t_embedder.mlp.0.bias", v);
    r->dit_t_fc2_w = t2_upload_f32(st, "t_embedder.mlp.2.weight", v);
    r->dit_t_fc2_b = t2_upload_f32(st, "t_embedder.mlp.2.bias", v);
    r->dit_mod_w   = t2_upload_f32(st, "adaLN_modulation.1.weight", v);
    r->dit_mod_b   = t2_upload_f32(st, "adaLN_modulation.1.bias", v);

    /* Input/output embedding and block weights: F16 if supported */
    #define UPW(name) (use_f16 ? t2_upload_f16(st, name, v) : t2_upload_f32(st, name, v))
    r->dit_x_emb_w = UPW("input_layer.weight");
    r->dit_x_emb_b = t2_upload_f32(st, "input_layer.bias", v);
    r->dit_out_w   = UPW("out_layer.weight");
    r->dit_out_b   = t2_upload_f32(st, "out_layer.bias", v);

    if (!r->dit_t_fc1_w) fprintf(stderr, "T2: WARNING: t_embedder missing\n");
    if (!r->dit_mod_w)   fprintf(stderr, "T2: WARNING: adaLN_modulation missing\n");
    if (!r->dit_x_emb_w) fprintf(stderr, "T2: WARNING: input_layer missing\n");

    /* Per-block */
    for (int L = 0; L < DIT_DEPTH; L++) {
        dit_block_gpu *blk = &r->dit_blocks[L];
        char name[256];

        #define BLKW(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                              UPW(name))
        #define BLKB(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                              t2_upload_f32(st, name, v >= 2 ? v : 0))

        blk->sa_qkv_w    = BLKW("self_attn.to_qkv.weight");
        blk->sa_qkv_b    = BLKB("self_attn.to_qkv.bias");
        blk->sa_q_norm   = BLKB("self_attn.q_rms_norm.gamma");
        blk->sa_k_norm   = BLKB("self_attn.k_rms_norm.gamma");
        blk->sa_out_w    = BLKW("self_attn.to_out.weight");
        blk->sa_out_b    = BLKB("self_attn.to_out.bias");

        blk->norm2_w     = BLKB("norm2.weight");
        blk->norm2_b     = BLKB("norm2.bias");
        blk->ca_q_w      = BLKW("cross_attn.to_q.weight");
        blk->ca_q_b      = BLKB("cross_attn.to_q.bias");
        blk->ca_kv_w     = BLKW("cross_attn.to_kv.weight");
        blk->ca_kv_b     = BLKB("cross_attn.to_kv.bias");
        blk->ca_q_norm   = BLKB("cross_attn.q_rms_norm.gamma");
        blk->ca_k_norm   = BLKB("cross_attn.k_rms_norm.gamma");
        blk->ca_out_w    = BLKW("cross_attn.to_out.weight");
        blk->ca_out_b    = BLKB("cross_attn.to_out.bias");

        blk->mlp_fc1_w   = BLKW("mlp.mlp.0.weight");
        blk->mlp_fc1_b   = BLKB("mlp.mlp.0.bias");
        blk->mlp_fc2_w   = BLKW("mlp.mlp.2.weight");
        blk->mlp_fc2_b   = BLKB("mlp.mlp.2.bias");

        blk->mod_bias    = BLKB("modulation");

        #undef BLKW
        #undef BLKB

        if (L == 0 && !blk->sa_qkv_w)
            fprintf(stderr, "T2: WARNING: block 0 self_attn missing\n");

        /* Save CPU copy for layer streaming */
        cuStreamSynchronize(0);
        dit_block_save_cpu_copy(&r->dit_blocks_cpu[L], blk, use_f16,
                                 DIT_DIM, DIT_HEADS, DIT_HEAD_DIM, DIT_FFN, DIT_COND_DIM);

        /* If streaming enabled, free GPU weights for blocks beyond limit */
        if (r->max_gpu_layers > 0 && L >= r->max_gpu_layers)
            dit_block_free_gpu(blk);
    }
    #undef UPW

    /* Precompute 3D RoPE tables on CPU, upload to GPU */
    {
        int gs = DIT_GRID, nt = DIT_N_TOKENS;
        int n_freqs = DIT_HEAD_DIM / 6;  /* 21 */
        int axis_dim = 2 * n_freqs;
        r->dit_n_freqs = n_freqs;
        r->dit_axis_dim = axis_dim;

        float *freqs = (float *)malloc((size_t)n_freqs * sizeof(float));
        for (int j = 0; j < n_freqs; j++)
            freqs[j] = 1.0f / powf(10000.0f, (float)j / (float)n_freqs);

        size_t table_sz = (size_t)nt * 3 * n_freqs * sizeof(float);
        float *cos_tab = (float *)malloc(table_sz);
        float *sin_tab = (float *)malloc(table_sz);

        for (int i = 0; i < nt; i++) {
            int z = i / (gs * gs), y = (i / gs) % gs, x = i % gs;
            float coords[3] = {(float)z, (float)y, (float)x};
            for (int axis = 0; axis < 3; axis++) {
                for (int j = 0; j < n_freqs; j++) {
                    float theta = coords[axis] * freqs[j];
                    cos_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = cosf(theta);
                    sin_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = sinf(theta);
                }
            }
        }

        r->dit_rope_cos = cu_upload_raw(cos_tab, table_sz);
        r->dit_rope_sin = cu_upload_raw(sin_tab, table_sz);
        free(freqs); free(cos_tab); free(sin_tab);
        fprintf(stderr, "T2: RoPE tables uploaded (%d freqs/axis)\n", n_freqs);
    }

    r->dit_use_f16 = use_f16;
    safetensors_close(st);
    { size_t fr = 0, to = 0; cuMemGetInfo(&fr, &to);
      int on_gpu = r->max_gpu_layers > 0 ? (r->max_gpu_layers < DIT_DEPTH ? r->max_gpu_layers : DIT_DEPTH) : DIT_DEPTH;
      fprintf(stderr, "T2: DiT loaded (%d blocks, %d on GPU, weights=%s, GPU %.0f/%.0f MB free)\n",
              DIT_DEPTH, on_gpu, use_f16 ? "F16" : "F32", fr/1048576.0, to/1048576.0); }
    return 0;
}

static int load_decoder_weights(cuda_trellis2_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading decoder from %s (%d tensors)\n", path, st->n_tensors);
    int v = r->verbose;

    r->dec_conv_in_w = t2_upload_f32(st, "input_layer.weight", v);
    r->dec_conv_in_b = t2_upload_f32(st, "input_layer.bias", v);

    /* Load ResBlock weights */
    #define LOAD_RES(rb, prefix) do { \
        char _n[256]; \
        snprintf(_n, sizeof(_n), "%snorm1.weight", prefix); (rb).gn1_w = t2_upload_f32(st, _n, v>=2?v:0); \
        snprintf(_n, sizeof(_n), "%snorm1.bias", prefix);   (rb).gn1_b = t2_upload_f32(st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv1.weight", prefix); (rb).conv1_w = t2_upload_f32(st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv1.bias", prefix);   (rb).conv1_b = t2_upload_f32(st, _n, 0); \
        snprintf(_n, sizeof(_n), "%snorm2.weight", prefix); (rb).gn2_w = t2_upload_f32(st, _n, 0); \
        snprintf(_n, sizeof(_n), "%snorm2.bias", prefix);   (rb).gn2_b = t2_upload_f32(st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv2.weight", prefix); (rb).conv2_w = t2_upload_f32(st, _n, 0); \
        snprintf(_n, sizeof(_n), "%sconv2.bias", prefix);   (rb).conv2_b = t2_upload_f32(st, _n, 0); \
    } while(0)

    LOAD_RES(r->dec_middle[0], "middle_block.0.");
    LOAD_RES(r->dec_middle[1], "middle_block.1.");
    LOAD_RES(r->dec_res16[0],  "blocks.0.");
    LOAD_RES(r->dec_res16[1],  "blocks.1.");
    r->dec_up1_w = t2_upload_f32(st, "blocks.2.conv.weight", v);
    r->dec_up1_b = t2_upload_f32(st, "blocks.2.conv.bias", v);
    LOAD_RES(r->dec_res32[0],  "blocks.3.");
    LOAD_RES(r->dec_res32[1],  "blocks.4.");
    r->dec_up2_w = t2_upload_f32(st, "blocks.5.conv.weight", v);
    r->dec_up2_b = t2_upload_f32(st, "blocks.5.conv.bias", v);
    LOAD_RES(r->dec_res64[0],  "blocks.6.");
    LOAD_RES(r->dec_res64[1],  "blocks.7.");
    r->dec_out_gn_w   = t2_upload_f32(st, "out_layer.0.weight", v);
    r->dec_out_gn_b   = t2_upload_f32(st, "out_layer.0.bias", v);
    r->dec_out_conv_w  = t2_upload_f32(st, "out_layer.2.weight", v);
    r->dec_out_conv_b  = t2_upload_f32(st, "out_layer.2.bias", v);

    #undef LOAD_RES

    safetensors_close(st);
    fprintf(stderr, "T2: decoder loaded\n");
    return 0;
}

static int load_dinov3_weights(cuda_trellis2_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading DINOv3 from %s (%d tensors)\n", path, st->n_tensors);
    int v = r->verbose;

    /* Patch embedding + tokens (timm naming) */
    r->dino_patch_w = t2_upload_f32(st, "patch_embed.proj.weight", v);
    r->dino_patch_b = t2_upload_f32(st, "patch_embed.proj.bias", v);
    r->dino_cls_token = t2_upload_f32(st, "cls_token", v);
    r->dino_storage_tokens = t2_upload_f32(st, "reg_token", v);

    r->dino_has_layerscale = 0;
    r->dino_has_qk_norm = 0;

    /* Per-layer weights (timm naming: fused QKV → split to separate Q/K/V) */
    for (int L = 0; L < DINO_LAYERS; L++) {
        dino_layer_gpu *l = &r->dino_layers[L];
        char name[256];

        #define DINO_LOAD(field, suffix) do { \
            snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix); \
            l->field = t2_upload_f32(st, name, v >= 2 ? v : 0); \
        } while(0)

        DINO_LOAD(ln1_w, "norm1.weight");
        DINO_LOAD(ln1_b, "norm1.bias");

        /* Split fused QKV [3*dim, dim] into separate Q, K, V on CPU */
        snprintf(name, sizeof(name), "blocks.%d.attn.qkv.weight", L);
        int idx = safetensors_find(st, name);
        if (idx >= 0) {
            size_t nbytes = safetensors_nbytes(st, idx);
            const char *dtype = safetensors_dtype(st, idx);
            void *data = safetensors_data(st, idx);
            int n_elem = (int)(nbytes / (strcmp(dtype, "F32") == 0 ? 4 : 2));
            int dim = DINO_HIDDEN;
            float *f32 = (float *)malloc(n_elem * sizeof(float));
            if (strcmp(dtype, "F32") == 0) memcpy(f32, data, n_elem * 4);
            else if (strcmp(dtype, "F16") == 0) {
                const uint16_t *src = (const uint16_t *)data;
                for (int i = 0; i < n_elem; i++) {
                    uint16_t h = src[i]; uint32_t sign = ((uint32_t)h & 0x8000) << 16;
                    uint32_t exp = (h>>10)&0x1f; uint32_t mant = h&0x3ff; uint32_t f;
                    if (exp==0) { if (mant==0) f=sign; else { exp=1; while(!(mant&0x400)){mant<<=1;exp--;} mant&=0x3ff; f=sign|((exp+127-15)<<23)|(mant<<13); }}
                    else if (exp==31) f=sign|0x7f800000|(mant<<13); else f=sign|((exp+127-15)<<23)|(mant<<13);
                    memcpy(&f32[i], &f, 4);
                }
            }
            /* Split: Q=[dim, dim], K=[dim, dim], V=[dim, dim] */
            l->q_w = cu_upload_raw(f32, (size_t)dim * dim * 4);
            l->k_w = cu_upload_raw(f32 + dim * dim, (size_t)dim * dim * 4);
            l->v_w = cu_upload_raw(f32 + 2 * dim * dim, (size_t)dim * dim * 4);
            free(f32);
            /* timm has no QKV bias — create zeros */
            float *zeros = (float *)calloc(dim, sizeof(float));
            l->q_b = cu_upload_raw(zeros, dim * 4);
            l->k_b = cu_upload_raw(zeros, dim * 4);
            l->v_b = cu_upload_raw(zeros, dim * 4);
            free(zeros);
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
        l->ls1 = t2_upload_f32(st, name, 0);
        snprintf(name, sizeof(name), "blocks.%d.gamma_2", L);
        l->ls2 = t2_upload_f32(st, name, 0);
        if (l->ls1) r->dino_has_layerscale = 1;

        /* QK norm (not present in timm ViT-L) */
        l->q_norm_w = 0; l->k_norm_w = 0;

        #undef DINO_LOAD
    }

    /* Precompute 2D RoPE tables for 32×32 grid */
    {
        int gh = DINO_IMG_SIZE / DINO_PATCH;  /* 32 */
        int np = gh * gh;  /* 1024 */
        int hd = DINO_HEAD_DIM;  /* 64 */
        float rope_theta = 100.0f;
        int n_freqs = hd / 4;  /* 16 */

        float *inv_freq = (float *)malloc(n_freqs * sizeof(float));
        for (int j = 0; j < n_freqs; j++)
            inv_freq[j] = 1.0f / powf(rope_theta, (float)(4 * j) / (float)hd);

        float *cos_tab = (float *)malloc((size_t)np * hd * sizeof(float));
        float *sin_tab = (float *)malloc((size_t)np * hd * sizeof(float));

        for (int p = 0; p < np; p++) {
            int py = p / gh, px = p % gh;
            float cy = ((0.5f + py) / gh) * 2.0f - 1.0f;
            float cx = ((0.5f + px) / gh) * 2.0f - 1.0f;

            /* angles: [y_freqs(16), x_freqs(16)] tiled 2× = [y16, x16, y16, x16] = 64 */
            float angles[64];
            for (int j = 0; j < n_freqs; j++) {
                angles[j]              = 2.0f * 3.14159265358979f * cy * inv_freq[j];
                angles[n_freqs + j]    = 2.0f * 3.14159265358979f * cx * inv_freq[j];
                angles[2*n_freqs + j]  = angles[j];       /* tile */
                angles[3*n_freqs + j]  = angles[n_freqs + j];
            }
            for (int d = 0; d < hd; d++) {
                cos_tab[p * hd + d] = cosf(angles[d]);
                sin_tab[p * hd + d] = sinf(angles[d]);
            }
        }
        r->dino_rope_cos = cu_upload_raw(cos_tab, (size_t)np * hd * sizeof(float));
        r->dino_rope_sin = cu_upload_raw(sin_tab, (size_t)np * hd * sizeof(float));
        free(inv_freq); free(cos_tab); free(sin_tab);
        fprintf(stderr, "T2: DINOv3 RoPE tables uploaded (16 freqs, 32x32 grid)\n");
    }

    safetensors_close(st);
    fprintf(stderr, "T2: DINOv3 loaded (%d blocks, layerscale=%d)\n",
            DINO_LAYERS, r->dino_has_layerscale);
    return 0;
}

/* Generic DiT weight loader (used for Stage 2 and Stage 3).
 * Both use identical architecture with different in_channels. */
static int load_dit_model_weights(cuda_trellis2_runner *r, const char *path,
                                    dit_model_gpu *m, dit_block_gpu *blocks_arr,
                                    dit_block_cpu *blocks_cpu_arr,
                                    const char *label) {
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading %s from %s (%d tensors)\n", label, path, st->n_tensors);
    int v = r->verbose;
    int use_f16 = (r->ops.sm_version >= 70);

    m->n_blocks = DIT_DEPTH;
    m->model_channels = DIT_DIM;
    m->n_heads = DIT_HEADS;
    m->head_dim = DIT_HEAD_DIM;
    m->ffn_hidden = DIT_FFN;
    m->cond_dim = DIT_COND_DIM;
    m->blocks = blocks_arr;

    /* Detect in_channels and out_channels from weight shapes */
    {
        int idx = safetensors_find(st, "input_layer.weight");
        if (idx >= 0) m->in_channels = (int)safetensors_shape(st, idx)[1];
        else m->in_channels = 32;
        idx = safetensors_find(st, "out_layer.weight");
        if (idx >= 0) m->out_channels = (int)safetensors_shape(st, idx)[0];
        else m->out_channels = m->in_channels;
    }

    /* Timestep MLP and modulation: keep F32 (n_tok=1, GEMM perf irrelevant).
     * Modulation kernel reads weights as F32 directly. */
    m->t_fc1_w = t2_upload_f32(st, "t_embedder.mlp.0.weight", v);
    m->t_fc1_b = t2_upload_f32(st, "t_embedder.mlp.0.bias", v);
    m->t_fc2_w = t2_upload_f32(st, "t_embedder.mlp.2.weight", v);
    m->t_fc2_b = t2_upload_f32(st, "t_embedder.mlp.2.bias", v);
    m->mod_w   = t2_upload_f32(st, "adaLN_modulation.1.weight", v);
    m->mod_b   = t2_upload_f32(st, "adaLN_modulation.1.bias", v);

    /* Input/output embedding: F16 weight, F32 bias */
    #define UPW(name) (use_f16 ? t2_upload_f16(st, name, v) : t2_upload_f32(st, name, v))
    m->x_emb_w = UPW("input_layer.weight");
    m->x_emb_b = t2_upload_f32(st, "input_layer.bias", v);
    m->out_w   = UPW("out_layer.weight");
    m->out_b   = t2_upload_f32(st, "out_layer.bias", v);

    for (int L = 0; L < m->n_blocks; L++) {
        dit_block_gpu *blk = &m->blocks[L];
        char name[256];
        #define BLK2W(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                               UPW(name))
        #define BLK2B(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                               t2_upload_f32(st, name, v >= 2 ? v : 0))
        blk->sa_qkv_w  = BLK2W("self_attn.to_qkv.weight");
        blk->sa_qkv_b  = BLK2B("self_attn.to_qkv.bias");
        blk->sa_q_norm  = BLK2B("self_attn.q_rms_norm.gamma");
        blk->sa_k_norm  = BLK2B("self_attn.k_rms_norm.gamma");
        blk->sa_out_w   = BLK2W("self_attn.to_out.weight");
        blk->sa_out_b   = BLK2B("self_attn.to_out.bias");
        blk->norm2_w    = BLK2B("norm2.weight");
        blk->norm2_b    = BLK2B("norm2.bias");
        blk->ca_q_w     = BLK2W("cross_attn.to_q.weight");
        blk->ca_q_b     = BLK2B("cross_attn.to_q.bias");
        blk->ca_kv_w    = BLK2W("cross_attn.to_kv.weight");
        blk->ca_kv_b    = BLK2B("cross_attn.to_kv.bias");
        blk->ca_q_norm  = BLK2B("cross_attn.q_rms_norm.gamma");
        blk->ca_k_norm  = BLK2B("cross_attn.k_rms_norm.gamma");
        blk->ca_out_w   = BLK2W("cross_attn.to_out.weight");
        blk->ca_out_b   = BLK2B("cross_attn.to_out.bias");
        blk->mlp_fc1_w  = BLK2W("mlp.mlp.0.weight");
        blk->mlp_fc1_b  = BLK2B("mlp.mlp.0.bias");
        blk->mlp_fc2_w  = BLK2W("mlp.mlp.2.weight");
        blk->mlp_fc2_b  = BLK2B("mlp.mlp.2.bias");
        blk->mod_bias   = BLK2B("modulation");
        #undef BLK2W
        #undef BLK2B

        /* Save CPU copy for layer streaming */
        cuStreamSynchronize(0);
        dit_block_save_cpu_copy(&blocks_cpu_arr[L], blk, use_f16,
                                 DIT_DIM, DIT_HEADS, DIT_HEAD_DIM, DIT_FFN, DIT_COND_DIM);

        if (r->max_gpu_layers > 0 && L >= r->max_gpu_layers)
            dit_block_free_gpu(blk);
    }
    #undef UPW

    m->rope_cos = 0; m->rope_sin = 0;
    m->n_rope_freqs = DIT_HEAD_DIM / 6;
    m->rope_axis_dim = 2 * m->n_rope_freqs;

    safetensors_close(st);
    { size_t fr = 0, to = 0; cuMemGetInfo(&fr, &to);
      int on_gpu = r->max_gpu_layers > 0 ? (r->max_gpu_layers < m->n_blocks ? r->max_gpu_layers : m->n_blocks) : m->n_blocks;
      fprintf(stderr, "T2: %s loaded (%d blocks, %d on GPU, in_ch=%d, weights=%s, GPU %.0f/%.0f MB free)\n",
              label, m->n_blocks, on_gpu, m->in_channels, use_f16 ? "F16" : "F32",
              fr/1048576.0, to/1048576.0); }
    return 0;
}

int cuda_trellis2_load_stage2(cuda_trellis2_runner *r, const char *stage2_path) {
    int ret = load_dit_model_weights(r, stage2_path, &r->stage2, r->stage2_blocks,
                                      r->stage2_blocks_cpu, "Stage 2");
    if (ret == 0) r->stage2_loaded = 1;
    return ret;
}

int cuda_trellis2_load_stage3(cuda_trellis2_runner *r, const char *stage3_path) {
    int ret = load_dit_model_weights(r, stage3_path, &r->stage3, r->stage3_blocks,
                                      r->stage3_blocks_cpu, "Stage 3");
    if (ret == 0) r->stage3_loaded = 1;
    return ret;
}

/* ---- Shape decoder weight loading ---- */

/* Upload sparse conv weight transposed: [out_C, 27, in_C] -> [27, out_C, in_C]
 * This layout enables gather-GEMM: for each k in 0..26, use weight[k*out_C*in_C] */
static CUdeviceptr t2_upload_conv_transposed(st_context *st, const char *name,
                                               int out_C, int in_C, int v,
                                               int use_f16) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { if (v) fprintf(stderr, "  [MISSING] %s\n", name); return 0; }
    const char *dtype = safetensors_dtype(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    /* First convert to F32 on CPU */
    size_t n_elem = (size_t)out_C * 27 * in_C;
    float *f32 = (float *)malloc(n_elem * sizeof(float));
    const uint16_t *src16 = (const uint16_t *)data;

    if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, data, n_elem * sizeof(float));
    } else if (strcmp(dtype, "BF16") == 0) {
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t bits = (uint32_t)src16[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else if (strcmp(dtype, "F16") == 0) {
        for (size_t i = 0; i < n_elem; i++) {
            uint16_t h = src16[i];
            uint32_t sign = ((uint32_t)h & 0x8000) << 16;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            uint32_t fb;
            if (exp == 0) { fb = sign; }
            else if (exp == 31) fb = sign | 0x7f800000 | (mant << 13);
            else fb = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            memcpy(&f32[i], &fb, 4);
        }
    } else {
        free(f32); return 0;
    }

    /* Transpose: [out_C, 27, in_C] -> [27, out_C, in_C] */
    float *trans = (float *)malloc(n_elem * sizeof(float));
    for (int o = 0; o < out_C; o++)
        for (int k = 0; k < 27; k++)
            memcpy(trans + ((size_t)k * out_C + o) * in_C,
                   f32 + ((size_t)o * 27 + k) * in_C,
                   (size_t)in_C * sizeof(float));
    free(f32);

    CUdeviceptr d;
    if (use_f16) {
        /* Convert transposed F32 to F16 */
        uint16_t *f16 = (uint16_t *)malloc(n_elem * sizeof(uint16_t));
        for (size_t i = 0; i < n_elem; i++) {
            uint32_t fb; memcpy(&fb, &trans[i], 4);
            uint32_t s = (fb >> 16) & 0x8000;
            int32_t e = ((fb >> 23) & 0xff) - 127 + 15;
            uint32_t m = (fb >> 13) & 0x3ff;
            if (e <= 0) f16[i] = (uint16_t)s;
            else if (e >= 31) f16[i] = (uint16_t)(s | 0x7c00);
            else f16[i] = (uint16_t)(s | (e << 10) | m);
        }
        d = cu_upload_raw(f16, n_elem * sizeof(uint16_t));
        free(f16);
    } else {
        d = cu_upload_raw(trans, n_elem * sizeof(float));
    }
    free(trans);

    if (v >= 2)
        fprintf(stderr, "  %s: [%d,27,%d] -> [27,%d,%d] %s GPU\n",
                name, out_C, in_C, out_C, in_C, use_f16 ? "F16" : "F32");
    return d;
}

int cuda_trellis2_load_shape_decoder(cuda_trellis2_runner *r, const char *path) {
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading shape decoder from %s (%d tensors)\n", path, st->n_tensors);
    int v = r->verbose;
    int use_f16 = (r->ops.sm_version >= 70);

    /* Weight names match: blocks.{stage}.{block}.{component} */
    #define UPW(n) (use_f16 ? t2_upload_f16(st, n, v) : t2_upload_f32(st, n, v))
    #define UPB(n) t2_upload_f32(st, n, v)

    r->shape_dec.from_latent_w = UPW("from_latent.weight");
    r->shape_dec.from_latent_b = UPB("from_latent.bias");
    r->shape_dec.out_w = UPW("output_layer.weight");
    r->shape_dec.out_b = UPB("output_layer.bias");

    /* Channel progression: 1024 -> 512 -> 256 -> 128 -> 64 */
    int ch[] = {1024, 512, 256, 128, 64};
    int nblk[] = {4, 16, 8, 4};
    memcpy(r->shape_dec.channels, ch, sizeof(ch));
    memcpy(r->shape_dec.n_convnext, nblk, sizeof(nblk));

    for (int s = 0; s < 4; s++) {
        int C = ch[s];

        /* ConvNeXt blocks */
        for (int b = 0; b < nblk[s]; b++) {
            char name[256];
            #define CN(suf) (snprintf(name, sizeof(name), "blocks.%d.%d.%s", s, b, suf), name)
            r->shape_dec.convnext[s][b].conv_w = t2_upload_conv_transposed(
                st, CN("conv.weight"), C, C, v, use_f16);
            r->shape_dec.convnext[s][b].conv_b = UPB(CN("conv.bias"));
            r->shape_dec.convnext[s][b].norm_w = UPB(CN("norm.weight"));
            r->shape_dec.convnext[s][b].norm_b = UPB(CN("norm.bias"));
            r->shape_dec.convnext[s][b].mlp0_w = UPW(CN("mlp.0.weight"));
            r->shape_dec.convnext[s][b].mlp0_b = UPB(CN("mlp.0.bias"));
            r->shape_dec.convnext[s][b].mlp2_w = UPW(CN("mlp.2.weight"));
            r->shape_dec.convnext[s][b].mlp2_b = UPB(CN("mlp.2.bias"));
            #undef CN
        }

        /* C2S upsample block */
        {
            char name[256];
            int C_out = ch[s + 1];
            #define C2S(suf) (snprintf(name, sizeof(name), "blocks.%d.%d.%s", s, nblk[s], suf), name)
            r->shape_dec.c2s[s].norm1_w = UPB(C2S("norm1.weight"));
            r->shape_dec.c2s[s].norm1_b = UPB(C2S("norm1.bias"));
            r->shape_dec.c2s[s].conv1_w = t2_upload_conv_transposed(
                st, C2S("conv1.weight"), C_out * 8, C, v, use_f16);
            r->shape_dec.c2s[s].conv1_b = UPB(C2S("conv1.bias"));
            r->shape_dec.c2s[s].conv2_w = t2_upload_conv_transposed(
                st, C2S("conv2.weight"), C_out, C_out, v, use_f16);
            r->shape_dec.c2s[s].conv2_b = UPB(C2S("conv2.bias"));
            /* to_subdiv may be missing (texture decoder) */
            snprintf(name, sizeof(name), "blocks.%d.%d.to_subdiv.weight", s, nblk[s]);
            int idx = safetensors_find(st, name);
            if (idx >= 0) {
                r->shape_dec.c2s[s].subdiv_w = UPW(name);
                snprintf(name, sizeof(name), "blocks.%d.%d.to_subdiv.bias", s, nblk[s]);
                r->shape_dec.c2s[s].subdiv_b = UPB(name);
            } else {
                r->shape_dec.c2s[s].subdiv_w = 0;
                r->shape_dec.c2s[s].subdiv_b = 0;
            }
            #undef C2S
        }
    }

    #undef UPW
    #undef UPB

    safetensors_close(st);
    r->shape_dec.loaded = 1;

    /* Also load full CPU decoder for C2S operations (needs F32 weights) */
    r->shape_dec_cpu = t2_shape_dec_load(path);
    if (!r->shape_dec_cpu)
        fprintf(stderr, "T2: WARNING: failed to load CPU shape decoder for C2S\n");

    fprintf(stderr, "T2: shape decoder loaded (GPU=%s, CPU for C2S)\n", use_f16 ? "F16" : "F32");
    return 0;
}

int cuda_trellis2_load_texture_decoder(cuda_trellis2_runner *r, const char *path) {
    /* Texture decoder uses the same SC-VAE architecture as shape decoder.
     * Load as CPU decoder (AVX2+pthreads optimized). */
    r->tex_dec_cpu = t2_shape_dec_load(path);
    if (!r->tex_dec_cpu) return -1;
    r->tex_dec_loaded = 1;
    fprintf(stderr, "T2: texture decoder loaded (out_ch=%d)\n", r->tex_dec_cpu->out_channels);
    return 0;
}

int cuda_trellis2_load_weights(cuda_trellis2_runner *r,
                                const char *dinov3_path,
                                const char *stage1_path,
                                const char *decoder_path) {
    /* GEMM mode is set per-model in each forward wrapper (run_dit_forward,
     * run_stage2_forward). Default to F32 for safety. */
    r->ops.use_f32_gemm = 1;

    if (stage1_path && load_dit_weights(r, stage1_path) != 0) return -1;
    if (decoder_path && load_decoder_weights(r, decoder_path) != 0) return -1;

    if (dinov3_path && load_dinov3_weights(r, dinov3_path) != 0) return -1;

    /* Stage 2 shape flow model loads with same code as Stage 1 — just different in_channels */
    r->stage2_loaded = 0;

    return 0;
}

/* ======================================================================== */
/* DINOv3 forward pass                                                      */
/* ======================================================================== */

static void run_dinov3(cuda_trellis2_runner *r, CUdeviceptr d_image, CUdeviceptr d_out) {
    t2_ops *ops = &r->ops;
    CUstream stream = r->stream;
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

    /* Scratch layout */
    ensure_scratch(r, 0, (size_t)seq * dim * sizeof(float));       /* hidden */
    ensure_scratch(r, 1, (size_t)3 * seq * dim * sizeof(float));   /* QKV */
    ensure_scratch(r, 2, (size_t)seq * dim * sizeof(float));       /* attn_out */
    ensure_scratch(r, 3, (size_t)seq * ffn * sizeof(float));       /* mlp */
    ensure_scratch(r, 4, (size_t)seq * dim * sizeof(float));       /* normed */

    CUdeviceptr d_hidden = r->scratch[0];
    CUdeviceptr d_qkv    = r->scratch[1];
    CUdeviceptr d_attn   = r->scratch[2];
    CUdeviceptr d_mlp    = r->scratch[3];
    CUdeviceptr d_normed = r->scratch[4];

    /* 1. Patch embedding */
    /* Write patches starting at offset n_prefix in hidden buffer */
    CUdeviceptr d_patches = d_hidden + (size_t)n_prefix * dim * sizeof(float);
    {
        int gw2 = gw, dim2 = dim, ps2 = ps, iw = DINO_IMG_SIZE;
        void *args[] = {&d_patches, &d_image, &r->dino_patch_w, &r->dino_patch_b,
                        &gw2, &dim2, &ps2, &iw};
        cuLaunchKernel(ops->dinov3_patch_embed, (unsigned)(gw * gw), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* 2. Prepend CLS + register tokens */
    {
        int np2 = np, ns2 = ns, dim2 = dim;
        void *args[] = {&d_hidden, &d_patches, &r->dino_cls_token, &r->dino_storage_tokens,
                        &np2, &ns2, &dim2};
        cuLaunchKernel(ops->dinov3_prepend_tokens,
                       (unsigned)((seq * dim + 255) / 256), 1, 1,
                       256, 1, 1, 0, stream, args, NULL);
    }

    /* 3. Transformer blocks */
    for (int li = 0; li < DINO_LAYERS; li++) {
        dino_layer_gpu *l = &r->dino_layers[li];

        /* LN1 → Q, K, V projections */
        t2_op_layernorm(ops, stream, d_normed, d_hidden, l->ln1_w, l->ln1_b, seq, dim);

        CUdeviceptr d_Q = d_qkv;
        CUdeviceptr d_K = d_qkv + (size_t)seq * dim * sizeof(float);
        CUdeviceptr d_V = d_qkv + (size_t)2 * seq * dim * sizeof(float);
        t2_op_gemm(ops, stream, d_Q, l->q_w, d_normed, l->q_b, dim, dim, seq);
        t2_op_gemm(ops, stream, d_K, l->k_w, d_normed, l->k_b, dim, dim, seq);
        t2_op_gemm(ops, stream, d_V, l->v_w, d_normed, l->v_b, dim, dim, seq);

        /* 2D RoPE on Q and K (patch tokens only, skip prefix) */
        {
            int np3 = np, npx = n_prefix, dim3 = dim, nh = heads, hd3 = hd;
            void *args[] = {&d_Q, &d_K, &r->dino_rope_cos, &r->dino_rope_sin,
                            &np3, &npx, &dim3, &nh, &hd3};
            cuLaunchKernel(ops->rope_2d_dinov3, (unsigned)np, 1, 1,
                           256, 1, 1, 0, stream, args, NULL);
        }

        /* Self-attention */
        t2_op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, seq, dim, heads, hd);

        /* Output projection */
        t2_op_gemm(ops, stream, d_normed, l->out_w, d_attn, l->out_b, dim, dim, seq);

        /* LayerScale 1 + residual */
        if (l->ls1) {
            /* layerscale_add: dst[i] += src[i] * scale[i % dim] */
            int n = seq * dim;
            void *ls_args[] = {&d_hidden, &d_normed, &l->ls1, &n, &dim};
            cuLaunchKernel(ops->layerscale_add, (unsigned)((n+255)/256), 1, 1,
                           256, 1, 1, 0, stream, ls_args, NULL);
        } else {
            t2_op_add(ops, stream, d_hidden, d_normed, seq * dim);
        }

        /* LN2 → MLP → LayerScale 2 + residual */
        t2_op_layernorm(ops, stream, d_normed, d_hidden, l->ln2_w, l->ln2_b, seq, dim);
        t2_op_gemm(ops, stream, d_mlp, l->fc1_w, d_normed, l->fc1_b, ffn, dim, seq);
        t2_op_gelu(ops, stream, d_mlp, seq * ffn);
        t2_op_gemm(ops, stream, d_normed, l->fc2_w, d_mlp, l->fc2_b, dim, ffn, seq);

        if (l->ls2) {
            int n = seq * dim;
            void *ls_args[] = {&d_hidden, &d_normed, &l->ls2, &n, &dim};
            cuLaunchKernel(ops->layerscale_add, (unsigned)((n+255)/256), 1, 1,
                           256, 1, 1, 0, stream, ls_args, NULL);
        } else {
            t2_op_add(ops, stream, d_hidden, d_normed, seq * dim);
        }
    }

    /* 4. Final unparameterized LayerNorm (for TRELLIS.2 compatibility) */
    {
        float eps = 1e-6f;
        int dim2 = dim;
        void *args[] = {&d_out, &d_hidden, &dim2, &eps};
        cuLaunchKernel(ops->layernorm_noaffine, (unsigned)seq, 1, 1,
                       256, 1, 1, 512 * sizeof(float), stream, args, NULL);
    }
}

/* ======================================================================== */
/* DiT forward pass (single denoising step)                                 */
/* ======================================================================== */

/* Generic DiT forward: works for both Stage 1 (N=4096, in=8) and Stage 2 (N=variable, in=32) */
static void run_dit_forward_generic(cuda_trellis2_runner *r,
                                      CUdeviceptr d_x, float timestep,
                                      CUdeviceptr d_cond,
                                      CUdeviceptr d_output,
                                      int N,                   /* token count */
                                      int in_ch,               /* input channels */
                                      int out_ch,              /* output channels (may differ from in_ch) */
                                      int n_blocks,
                                      CUdeviceptr t_fc1_w, CUdeviceptr t_fc1_b,
                                      CUdeviceptr t_fc2_w, CUdeviceptr t_fc2_b,
                                      CUdeviceptr mod_w, CUdeviceptr mod_b,
                                      CUdeviceptr x_emb_w, CUdeviceptr x_emb_b,
                                      CUdeviceptr out_w, CUdeviceptr out_b,
                                      dit_block_gpu *blocks,
                                      dit_block_cpu *blocks_cpu, /* CPU copies for streaming (NULL = all on GPU) */
                                      CUdeviceptr rope_cos, CUdeviceptr rope_sin,
                                      int n_freqs, int axis_dim,
                                      int model_id) {  /* 1=Stage1, 2=Stage2, 3=Stage3 */
    t2_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int D = DIT_DIM;        /* 1536 */
    const int H = DIT_HEADS;      /* 12 */
    const int HD = DIT_HEAD_DIM;  /* 128 */
    const int FFN = DIT_FFN;      /* 8192 */
    const int ctx_len = DINO_SEQ_LEN;  /* 1029 */

    /* Scratch layout:
     * 0: hidden [N * D]
     * 1: QKV / MLP shared [max(3*N*D, N*FFN)]
     * 2: attn_out + normed [2 * N * D]
     * 3: mod[6*D] + t_emb[D] + cross_Q[N*D] + ca_K[ctx*D] + ca_V[ctx*D] */
    size_t qkv_sz = (size_t)3 * N * D * sizeof(float);
    size_t mlp_sz = (size_t)N * FFN * sizeof(float);
    size_t ca_kv_gemm_sz = (size_t)ctx_len * 2 * D * sizeof(float);  /* cross-attn KV GEMM */
    size_t sh1 = qkv_sz > mlp_sz ? qkv_sz : mlp_sz;
    if (ca_kv_gemm_sz > sh1) sh1 = ca_kv_gemm_sz;
    size_t ca_kv_sz = (size_t)ctx_len * D * sizeof(float);
    /* t_emb MLP outputs D floats (not 256), needs D-sized buffer */
    size_t buf3_sz = (size_t)(6*D + D) * sizeof(float)
                   + (size_t)N * D * sizeof(float) + 2 * ca_kv_sz
                   + (size_t)N * D * sizeof(float); /* split V space */

    ensure_scratch(r, 0, (size_t)N * D * sizeof(float));
    ensure_scratch(r, 1, sh1);
    ensure_scratch(r, 2, (size_t)2 * N * D * sizeof(float));
    ensure_scratch(r, 3, buf3_sz);

    CUdeviceptr d_hidden = r->scratch[0];
    CUdeviceptr d_qkv    = r->scratch[1];
    CUdeviceptr d_mlp    = r->scratch[1]; /* alias */
    CUdeviceptr d_attn   = r->scratch[2];
    CUdeviceptr d_normed = r->scratch[2] + (size_t)N * D * sizeof(float);

    CUdeviceptr d_mod    = r->scratch[3];
    CUdeviceptr d_temb   = d_mod + (size_t)6 * D * sizeof(float);
    CUdeviceptr d_cross_Q = d_temb + (size_t)D * sizeof(float);  /* t_emb is D floats, not 256 */
    CUdeviceptr d_ca_K   = d_cross_Q + (size_t)N * D * sizeof(float);
    CUdeviceptr d_ca_V   = d_ca_K + ca_kv_sz;
    CUdeviceptr d_split_V = d_ca_V + ca_kv_sz;

    /* 1. Input embedding: [N, DIT_IN_CH] -> [N, D] */
    t2_op_gemm(ops, stream, d_hidden, x_emb_w, d_x, x_emb_b,
               D, in_ch, N);

    if (r->verbose >= 2) { cuStreamSynchronize(stream); float _d[4]; cuMemcpyDtoH(_d, d_hidden, 16); fprintf(stderr, "  input_emb: [:4]=%.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3]); }

    /* 2. Timestep embedding: sinusoidal(256) -> MLP(256->D->D) */
    /* TRELLIS.2: t*1000, [cos, sin] order (NOT [sin, cos] like HY3D) */
    {
        float t_scaled = timestep * 1000.0f;
        int te_dim = 256;
        void *te_args2[] = {&d_temb, &t_scaled, &te_dim};
        cuLaunchKernel(ops->timestep_embed_cossin, (unsigned)((128+255)/256), 1, 1,
                       256, 1, 1, 0, stream, te_args2, NULL);
    }

    if (r->verbose >= 2) { cuStreamSynchronize(stream); float _d[4]; cuMemcpyDtoH(_d, d_temb, 16); fprintf(stderr, "  sinusoidal_emb: [:4]=%.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3]); }

    /* t_fc1: [D, 256] */
    /* Timestep MLP always uses F32 GEMM (weights are F32, n_tok=1) */
    t2_op_gemm_f32(ops, stream, d_normed, t_fc1_w, d_temb, t_fc1_b,
               D, 256, 1);
    /* SiLU */
    t2_op_silu_inplace(ops, stream, d_normed, D);
    /* t_fc2: [D, D] */
    t2_op_gemm_f32(ops, stream, d_temb, t_fc2_w, d_normed, t_fc2_b,
               D, D, 1);
    /* Now d_temb = [D] timestep embedding */

    if (r->verbose >= 2) { cuStreamSynchronize(stream); float _d[4]; cuMemcpyDtoH(_d, d_temb, 16); fprintf(stderr, "  t_emb_mlp: [:4]=%.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3]); }

    /* Precompute cross-attention KV for all blocks if not cached.
     * KV depends on block weights (different per block) but NOT on timestep or x_t.
     * So we compute once and cache for all subsequent calls with same cond. */
    int use_kv_cache = (r->ca_kv_cache_valid && r->ca_kv_cache_model_id == model_id);
    if (!use_kv_cache && d_cond) {
        /* Allocate/reallocate cache */
        size_t kv_sz = (size_t)ctx_len * D * sizeof(float);
        if (r->ca_kv_cache_n_blocks != n_blocks) {
            for (int bi = 0; bi < DIT_DEPTH; bi++) {
                if (r->ca_kv_cache_K[bi]) cuMemFree(r->ca_kv_cache_K[bi]);
                if (r->ca_kv_cache_V[bi]) cuMemFree(r->ca_kv_cache_V[bi]);
                r->ca_kv_cache_K[bi] = 0; r->ca_kv_cache_V[bi] = 0;
            }
        }
        for (int bi = 0; bi < n_blocks; bi++) {
            dit_block_gpu *blk = &blocks[bi];
            int blk_streamed = 0;
            if (blocks_cpu && !blk->sa_qkv_w) {
                cuStreamSynchronize(stream);
                dit_block_upload_to_gpu(blk, &blocks_cpu[bi]);
                blk_streamed = 1;
            }
            if (!r->ca_kv_cache_K[bi]) cuMemAlloc(&r->ca_kv_cache_K[bi], kv_sz);
            if (!r->ca_kv_cache_V[bi]) cuMemAlloc(&r->ca_kv_cache_V[bi], kv_sz);
            /* KV GEMM: [ctx, cond_dim] -> [ctx, 2*D] */
            t2_op_gemm(ops, stream, d_qkv, blk->ca_kv_w, d_cond, blk->ca_kv_b,
                       2 * D, DIT_COND_DIM, ctx_len);
            t2_op_split_kv_chunk(ops, stream, r->ca_kv_cache_K[bi], r->ca_kv_cache_V[bi],
                                  d_qkv, ctx_len, D);
            if (blk->ca_k_norm)
                t2_op_rms_norm_perhead(ops, stream, r->ca_kv_cache_K[bi], blk->ca_k_norm,
                                      ctx_len, H, HD, D);
            if (blk_streamed) { cuStreamSynchronize(stream); dit_block_free_gpu(blk); }
        }
        r->ca_kv_cache_n_blocks = n_blocks;
        r->ca_kv_cache_model_id = model_id;
        r->ca_kv_cache_valid = 1;
        if (r->verbose >= 1)
            fprintf(stderr, "  Cross-attn KV cached for %d blocks\n", n_blocks);
    }

    /* 3. Transformer blocks */
    for (int bi = 0; bi < n_blocks; bi++) {
        dit_block_gpu *blk = &blocks[bi];
        int streamed = 0;  /* 1 if we uploaded this block from CPU */

        /* Layer streaming: if block not on GPU, upload from CPU */
        if (blocks_cpu && !blk->sa_qkv_w) {
            cuStreamSynchronize(stream);  /* wait for previous work before upload */
            dit_block_upload_to_gpu(blk, &blocks_cpu[bi]);
            streamed = 1;
        }

        /* 3a. Modulation: SiLU(t_emb) @ mod_w + mod_b + block_bias -> [6*D] */
        t2_op_modulation(ops, stream, d_mod, d_temb,
                         mod_w, mod_b, blk->mod_bias,
                         D, 6 * D);

        if (bi == 0 && r->verbose >= 2) { cuStreamSynchronize(stream); float _d[8]; cuMemcpyDtoH(_d, d_mod, 32); fprintf(stderr, "  mod_block0: [:8]=%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3],_d[4],_d[5],_d[6],_d[7]); }
        if (bi == 0 && r->verbose >= 2) { float _s[4]; cuMemcpyDtoH(_s, d_mod+(size_t)D*sizeof(float), 16); fprintf(stderr, "  scale_sa: [:4]=%.6f %.6f %.6f %.6f\n", _s[0],_s[1],_s[2],_s[3]); }

        /* Split mod into shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp */
        CUdeviceptr d_shift_sa  = d_mod;
        CUdeviceptr d_scale_sa  = d_mod + (size_t)D * sizeof(float);
        CUdeviceptr d_gate_sa   = d_mod + (size_t)2 * D * sizeof(float);
        CUdeviceptr d_shift_mlp = d_mod + (size_t)3 * D * sizeof(float);
        CUdeviceptr d_scale_mlp = d_mod + (size_t)4 * D * sizeof(float);
        CUdeviceptr d_gate_mlp  = d_mod + (size_t)5 * D * sizeof(float);

        /* === Self-attention === */
        /* adaLN: LN(hidden) * (1+scale) + shift */
        t2_op_adaln(ops, stream, d_normed, d_hidden,
                    d_shift_sa, d_scale_sa, N, D);

        if (bi == 0 && r->verbose >= 2) { cuStreamSynchronize(stream); float _d[4]; cuMemcpyDtoH(_d, d_normed, 16); fprintf(stderr, "  adaln_out: [:4]=%.6f %.6f %.6f %.6f\n", _d[0],_d[1],_d[2],_d[3]); }

        /* QKV projection: [N, D] -> [N, 3*D] */
        t2_op_gemm(ops, stream, d_qkv, blk->sa_qkv_w, d_normed, blk->sa_qkv_b,
                   3 * D, D, N);

        /* Split QKV -> Q, K, V (standard chunk, NOT interleaved)
         * IMPORTANT: Q must NOT alias d_attn since the attention kernel reads Q
         * and writes to d_attn simultaneously. Use d_cross_Q as temp for Q. */
        CUdeviceptr d_Q = d_cross_Q;  /* use cross_Q buffer as temp for self-attn Q */
        CUdeviceptr d_K = d_normed;
        CUdeviceptr d_V = d_split_V;
        t2_op_split_qkv_chunk(ops, stream, d_Q, d_K, d_V, d_qkv, N, D);

        if (bi==0 && r->verbose>=2) dbg4("Q_raw", d_Q, stream);
        if (bi==0 && r->verbose>=2) dbg4("K_raw", d_K, stream);
        if (bi==0 && r->verbose>=2) dbg4("V_raw", d_V, stream);

        /* QK RMSNorm (per-head, gamma [n_heads * head_dim]) */
        if (blk->sa_q_norm)
            t2_op_rms_norm_perhead(ops, stream, d_Q, blk->sa_q_norm, N, H, HD, D);
        if (blk->sa_k_norm)
            t2_op_rms_norm_perhead(ops, stream, d_K, blk->sa_k_norm, N, H, HD, D);

        if (bi==0 && r->verbose>=2) dbg4("Q_rmsnorm", d_Q, stream);
        if (bi==0 && r->verbose>=2) dbg4("K_rmsnorm", d_K, stream);

        /* 3D RoPE on Q and K */
        t2_op_rope_3d(ops, stream, d_Q, rope_cos, rope_sin,
                      N, D, H, HD, n_freqs, axis_dim);
        t2_op_rope_3d(ops, stream, d_K, rope_cos, rope_sin,
                      N, D, H, HD, n_freqs, axis_dim);

        if (bi==0 && r->verbose>=2) dbg4("Q_rope", d_Q, stream);
        if (bi==0 && r->verbose>=2) dbg4("K_rope", d_K, stream);

        /* Debug: save Q, K, V for token 0 to check attention */
        if (bi == 0 && r->verbose >= 2) {
            cuStreamSynchronize(stream);
            /* Save first head's Q for token 0 (128 floats) */
            float q0[128], k0[128], v0[128];
            cuMemcpyDtoH(q0, d_Q, 128 * sizeof(float));
            cuMemcpyDtoH(k0, d_K, 128 * sizeof(float));
            cuMemcpyDtoH(v0, d_V, 128 * sizeof(float));
            /* Compute Q[0] @ K[0]^T manually for head 0 */
            float dot = 0;
            for (int d = 0; d < 128; d++) dot += q0[d] * k0[d];
            dot /= sqrtf(128.0f);
            fprintf(stderr, "  MANUAL Q[0]@K[0]/sqrt(hd) = %.6f (should produce softmax weight for k=0)\n", dot);
            /* Also check how many Q/K elements are nonzero */
            int nz_q = 0, nz_k = 0;
            for (int d = 0; d < 128; d++) { if (q0[d] != 0.0f) nz_q++; if (k0[d] != 0.0f) nz_k++; }
            fprintf(stderr, "  Q[0] nonzero: %d/128, K[0] nonzero: %d/128\n", nz_q, nz_k);
        }

        /* Self-attention — output goes to d_attn (d_Q is separate buffer) */
        t2_op_self_attn(ops, stream, d_attn, d_Q, d_K, d_V, N, D, H, HD);

        if (bi==0 && r->verbose>=2) dbg4("attn_out", d_attn, stream);

        /* Output projection + gated residual */
        t2_op_gemm(ops, stream, d_normed, blk->sa_out_w, d_attn, blk->sa_out_b,
                   D, D, N);

        if (bi==0 && r->verbose>=2) dbg4("sa_proj", d_normed, stream);

        t2_op_gated_add(ops, stream, d_hidden, d_normed, d_gate_sa, N * D, D);

        if (bi==0 && r->verbose>=2) dbg4("h_after_sa", d_hidden, stream);


        /* === Cross-attention === */
        /* LayerNorm (affine) */
        t2_op_layernorm(ops, stream, d_normed, d_hidden,
                        blk->norm2_w, blk->norm2_b, N, D);

        /* Q from tokens */
        t2_op_gemm(ops, stream, d_cross_Q, blk->ca_q_w, d_normed, blk->ca_q_b,
                   D, D, N);
        if (blk->ca_q_norm)
            t2_op_rms_norm_perhead(ops, stream, d_cross_Q, blk->ca_q_norm, N, H, HD, D);

        /* KV from conditioning (cached — precomputed above) */
        CUdeviceptr d_ca_K_cached = r->ca_kv_cache_K[bi];
        CUdeviceptr d_ca_V_cached = r->ca_kv_cache_V[bi];

        /* Cross-attention (using cached KV) */
        t2_op_cross_attn(ops, stream, d_attn, d_cross_Q, d_ca_K_cached, d_ca_V_cached,
                         N, ctx_len, D, H, HD);

        if (bi==0 && r->verbose>=2) dbg4("cross_attn_out", d_attn, stream);

        /* Cross-attn output + residual (no gate) */
        t2_op_gemm(ops, stream, d_normed, blk->ca_out_w, d_attn, blk->ca_out_b,
                   D, D, N);

        if (bi==0 && r->verbose>=2) dbg4("ca_proj", d_normed, stream);

        t2_op_add(ops, stream, d_hidden, d_normed, N * D);

        if (bi==0 && r->verbose>=2) dbg4("h_after_ca", d_hidden, stream);

        /* === MLP === */
        t2_op_adaln(ops, stream, d_normed, d_hidden,
                    d_shift_mlp, d_scale_mlp, N, D);

        if (bi==0 && r->verbose>=2) dbg4("mlp_adaln", d_normed, stream);

        t2_op_gemm(ops, stream, d_mlp, blk->mlp_fc1_w, d_normed, blk->mlp_fc1_b,
                   FFN, D, N);
        t2_op_gelu(ops, stream, d_mlp, N * FFN);
        t2_op_gemm(ops, stream, d_normed, blk->mlp_fc2_w, d_mlp, blk->mlp_fc2_b,
                   D, FFN, N);

        if (bi==0 && r->verbose>=2) dbg4("mlp_out", d_normed, stream);

        t2_op_gated_add(ops, stream, d_hidden, d_normed, d_gate_mlp, N * D, D);

        if (bi==0 && r->verbose>=2) dbg4("h_after_mlp", d_hidden, stream);

        /* Per-block debug */
        if (r->verbose >= 2) {
            cuStreamSynchronize(stream);
            float hs[4];
            cuMemcpyDtoH(hs, d_hidden, 4 * sizeof(float));
            float *hc = (float *)malloc(4096 * sizeof(float));
            cuMemcpyDtoH(hc, d_hidden, 4096 * sizeof(float));
            double sm = 0, sm2 = 0;
            for (int j = 0; j < 4096; j++) { sm += hc[j]; sm2 += (double)hc[j]*hc[j]; }
            fprintf(stderr, "  block %2d: std=%10.4f  h[:4]=%.4f %.4f %.4f %.4f\n",
                    bi, sqrt(sm2/4096 - (sm/4096)*(sm/4096)), hs[0], hs[1], hs[2], hs[3]);
            free(hc);
        }

        /* Layer streaming: free GPU weights after processing this block */
        if (streamed) {
            cuStreamSynchronize(stream);
            dit_block_free_gpu(blk);
        }
    }

    /* 4. Final LayerNorm (no affine) + output projection */
    {
        float eps = 1e-6f;
        void *ln_args[] = {&d_hidden, &d_hidden, &D, &eps};
        cuLaunchKernel(ops->layernorm_noaffine, (unsigned)N, 1, 1,
                       256, 1, 1, 512 * sizeof(float), stream, ln_args, NULL);
    }
    t2_op_gemm(ops, stream, d_output, out_w, d_hidden, out_b,
               out_ch, D, N);
}

/* Stage 1 wrapper — uses F16 MMA if weights are F16, else F32 */
static void run_dit_forward(cuda_trellis2_runner *r,
                             CUdeviceptr d_x, float timestep,
                             CUdeviceptr d_cond, CUdeviceptr d_output) {
    /* Save GEMM mode and set appropriate mode for Stage 1 */
    int saved_f32 = r->ops.use_f32_gemm;
    int saved_mma = r->ops.use_mma_gemm;
    if (r->dit_use_f16) {
        r->ops.use_f32_gemm = 0;
        r->ops.use_mma_gemm = 1;
    } else {
        r->ops.use_f32_gemm = 1;
        r->ops.use_mma_gemm = 0;
    }

    run_dit_forward_generic(r, d_x, timestep, d_cond, d_output,
        DIT_N_TOKENS, DIT_IN_CH, DIT_IN_CH, DIT_DEPTH,
        r->dit_t_fc1_w, r->dit_t_fc1_b, r->dit_t_fc2_w, r->dit_t_fc2_b,
        r->dit_mod_w, r->dit_mod_b,
        r->dit_x_emb_w, r->dit_x_emb_b,
        r->dit_out_w, r->dit_out_b,
        r->dit_blocks, r->dit_blocks_cpu,
        r->dit_rope_cos, r->dit_rope_sin,
        r->dit_n_freqs, r->dit_axis_dim, 1);

    r->ops.use_f32_gemm = saved_f32;
    r->ops.use_mma_gemm = saved_mma;
}

/* Stage 2 wrapper — uses F16 MMA GEMM if available (weights stored as F16) */
/* Generic sparse DiT forward (used for Stage 2 and Stage 3) */
static void run_sparse_dit_forward(cuda_trellis2_runner *r,
                                     dit_model_gpu *m, dit_block_cpu *blocks_cpu,
                                     CUdeviceptr d_x, float timestep,
                                     CUdeviceptr d_cond, CUdeviceptr d_output,
                                     int N, CUdeviceptr rope_cos, CUdeviceptr rope_sin,
                                     int model_id) {
    /* Save GEMM mode and set F16 MMA if available */
    int saved_f32 = r->ops.use_f32_gemm;
    int saved_mma = r->ops.use_mma_gemm;
    if (r->ops.sm_version >= 70) {
        r->ops.use_f32_gemm = 0;
        r->ops.use_mma_gemm = 1;
    } else {
        r->ops.use_f32_gemm = 1;
        r->ops.use_mma_gemm = 0;
    }

    run_dit_forward_generic(r, d_x, timestep, d_cond, d_output,
        N, m->in_channels, m->out_channels, m->n_blocks,
        m->t_fc1_w, m->t_fc1_b, m->t_fc2_w, m->t_fc2_b,
        m->mod_w, m->mod_b,
        m->x_emb_w, m->x_emb_b,
        m->out_w, m->out_b,
        m->blocks, blocks_cpu,
        rope_cos, rope_sin,
        m->n_rope_freqs, m->rope_axis_dim, model_id);

    r->ops.use_f32_gemm = saved_f32;
    r->ops.use_mma_gemm = saved_mma;
}

/* ======================================================================== */
/* Decoder forward                                                          */
/* ======================================================================== */

static void run_resblock(cuda_trellis2_runner *r,
                          CUdeviceptr d_out, CUdeviceptr d_in,
                          const dec_resblock_gpu *rb,
                          int C, int D, int H, int W, int G) {
    t2_ops *ops = &r->ops;
    CUstream s = r->stream;
    int spatial = D * H * W;

    /* Ensure scratch[4,5] for resblock temps */
    ensure_scratch(r, 4, (size_t)C * spatial * sizeof(float));
    ensure_scratch(r, 5, (size_t)C * spatial * sizeof(float));
    CUdeviceptr d_h1 = r->scratch[4];
    CUdeviceptr d_h2 = r->scratch[5];

    /* ChannelLN1 -> SiLU -> Conv1 */
    t2_op_channel_layernorm_3d(ops, s, d_h1, d_in, rb->gn1_w, rb->gn1_b, C, spatial);
    t2_op_silu_inplace(ops, s, d_h1, C * spatial);
    t2_op_conv3d(ops, s, d_h2, d_h1, rb->conv1_w, rb->conv1_b, C, C, D, H, W);

    /* ChannelLN2 -> SiLU -> Conv2 */
    t2_op_channel_layernorm_3d(ops, s, d_h1, d_h2, rb->gn2_w, rb->gn2_b, C, spatial);
    t2_op_silu_inplace(ops, s, d_h1, C * spatial);
    t2_op_conv3d(ops, s, d_out, d_h1, rb->conv2_w, rb->conv2_b, C, C, D, H, W);

    /* Skip (identity) */
    t2_op_add(ops, s, d_out, d_in, C * spatial);
}

static void run_decoder(cuda_trellis2_runner *r,
                         CUdeviceptr d_latent, CUdeviceptr d_output) {
    t2_ops *ops = &r->ops;
    CUstream s = r->stream;
    int G = 32;

    /* Alloc main decoder buffers */
    ensure_scratch(r, 6, (size_t)1024 * 16 * 16 * 16 * sizeof(float));
    ensure_scratch(r, 7, (size_t)512 * 32 * 32 * 32 * sizeof(float));

    CUdeviceptr d_buf_a = r->scratch[6];
    CUdeviceptr d_buf_b = r->scratch[7];

    /* conv_in: [8, 16^3] -> [512, 16^3] */
    t2_op_conv3d(ops, s, d_buf_a, d_latent, r->dec_conv_in_w, r->dec_conv_in_b,
                 8, 512, 16, 16, 16);

    if (r->verbose >= 2) dbg4("dec_conv_in", d_buf_a, s);

    /* middle + res16 blocks (512 ch, 16^3) */
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_middle[0], 512, 16, 16, 16, G);
    if (r->verbose >= 2) dbg4("dec_mid0", d_buf_b, s);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_middle[1], 512, 16, 16, 16, G);
    if (r->verbose >= 2) dbg4("dec_mid1", d_buf_a, s);
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_res16[0], 512, 16, 16, 16, G);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_res16[1], 512, 16, 16, 16, G);

    /* Up1: conv 512->1024, pixel_shuffle -> [128, 32^3] */
    t2_op_conv3d(ops, s, d_buf_b, d_buf_a, r->dec_up1_w, r->dec_up1_b,
                 512, 1024, 16, 16, 16);
    t2_op_pixel_shuffle_3d(ops, s, d_buf_a, d_buf_b, 128, 16, 16, 16);

    /* res32 blocks (128 ch, 32^3) */
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_res32[0], 128, 32, 32, 32, G);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_res32[1], 128, 32, 32, 32, G);

    /* Up2: conv 128->256, pixel_shuffle -> [32, 64^3] */
    t2_op_conv3d(ops, s, d_buf_b, d_buf_a, r->dec_up2_w, r->dec_up2_b,
                 128, 256, 32, 32, 32);
    /* Need bigger buffer for 64^3 */
    ensure_scratch(r, 6, (size_t)32 * 64 * 64 * 64 * sizeof(float));
    d_buf_a = r->scratch[6];
    t2_op_pixel_shuffle_3d(ops, s, d_buf_a, d_buf_b, 32, 32, 32, 32);

    /* res64 blocks (32 ch, 64^3) */
    ensure_scratch(r, 7, (size_t)32 * 64 * 64 * 64 * sizeof(float));
    d_buf_b = r->scratch[7];
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_res64[0], 32, 64, 64, 64, G);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_res64[1], 32, 64, 64, 64, G);

    /* Output: GN -> SiLU -> Conv3d(32->1) */
    /* Output: ChannelLayerNorm -> SiLU -> Conv3d(32->1) */
    t2_op_channel_layernorm_3d(ops, s, d_buf_b, d_buf_a,
                                r->dec_out_gn_w, r->dec_out_gn_b, 32, 64*64*64);
    t2_op_silu_inplace(ops, s, d_buf_b, 32 * 64 * 64 * 64);
    t2_op_conv3d(ops, s, d_output, d_buf_b, r->dec_out_conv_w, r->dec_out_conv_b,
                 32, 1, 64, 64, 64);
}

/* ======================================================================== */
/* Full pipeline                                                            */
/* ======================================================================== */

/* Simple xoshiro256** */
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

float *cuda_trellis2_predict(cuda_trellis2_runner *r,
                              const uint8_t *rgb, int w, int h,
                              int n_steps, float cfg_scale,
                              uint32_t seed) {
    (void)rgb; (void)w; (void)h;
    fprintf(stderr, "T2: full predict not implemented yet (use per-stage APIs)\n");
    fprintf(stderr, "T2: DINOv3 GPU encoding is TODO — pass pre-computed features\n");
    return NULL;
}

/* ---- Per-stage APIs ---- */

void cuda_trellis2_invalidate_kv_cache(cuda_trellis2_runner *r) {
    r->ca_kv_cache_valid = 0;
    r->ca_kv_cache_model_id = 0;
}

int cuda_trellis2_run_dit(cuda_trellis2_runner *r,
                           const float *x_t, float timestep,
                           const float *cond_features, float *output) {
    /* Invalidate KV cache since cond may differ between CFG passes */
    cuda_trellis2_invalidate_kv_cache(r);
    CUstream s = r->stream;
    int N = DIT_N_TOKENS, in_ch = DIT_IN_CH;
    int ctx_len = DINO_SEQ_LEN;

    /* Transpose input from NCDHW [C, spatial] to [spatial, C] on CPU.
     * Official code: x.view(B, C, -1).permute(0, 2, 1) -> [B, N, C] */
    float *x_transposed = (float *)malloc((size_t)N * in_ch * sizeof(float));
    for (int pos = 0; pos < N; pos++)
        for (int ch = 0; ch < in_ch; ch++)
            x_transposed[pos * in_ch + ch] = x_t[ch * N + pos];

    CUdeviceptr d_x = cu_upload_raw(x_transposed, (size_t)N * in_ch * sizeof(float));
    free(x_transposed);
    CUdeviceptr d_cond = cu_upload_raw(cond_features, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)N * in_ch * sizeof(float));

    run_dit_forward(r, d_x, timestep, d_cond, d_out);

    /* Transpose output from [spatial, C] back to [C, spatial] NCDHW */
    float *out_flat = (float *)malloc((size_t)N * in_ch * sizeof(float));
    cuStreamSynchronize(s);
    cuMemcpyDtoH(out_flat, d_out, (size_t)N * in_ch * sizeof(float));
    for (int pos = 0; pos < N; pos++)
        for (int ch = 0; ch < in_ch; ch++)
            output[ch * N + pos] = out_flat[pos * in_ch + ch];
    free(out_flat);

    cuMemFree(d_x); cuMemFree(d_cond); cuMemFree(d_out);
    return 0;
}

int cuda_trellis2_run_decoder(cuda_trellis2_runner *r,
                               const float *latent, float *output) {
    CUstream s = r->stream;
    CUdeviceptr d_lat = cu_upload_raw(latent, (size_t)8 * 16 * 16 * 16 * sizeof(float));
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)64 * 64 * 64 * sizeof(float));

    run_decoder(r, d_lat, d_out);

    cuStreamSynchronize(s);
    cuMemcpyDtoH(output, d_out, (size_t)64 * 64 * 64 * sizeof(float));

    cuMemFree(d_lat); cuMemFree(d_out);
    return 0;
}

/* ======================================================================== */
/* Shape decoder forward (SC-VAE, gather-GEMM sparse conv)                  */
/* ======================================================================== */

/* Run one ConvNeXt block on GPU.
 * d_feats: [N, C] in/out features (modified in-place via residual)
 * d_gather_map: [N, 27] precomputed neighbor indices */
static void run_shape_convnext(cuda_trellis2_runner *r,
                                 CUdeviceptr d_feats, int N, int C,
                                 CUdeviceptr d_gather_map,
                                 CUdeviceptr conv_w, CUdeviceptr conv_b,
                                 CUdeviceptr norm_w, CUdeviceptr norm_b,
                                 CUdeviceptr mlp0_w, CUdeviceptr mlp0_b,
                                 CUdeviceptr mlp2_w, CUdeviceptr mlp2_b) {
    t2_ops *ops = &r->ops;
    CUstream s = r->stream;

    /* Scratch layout:
     * 8: [N, C] conv output / tmp
     * 9: [N, max(C, 4C)] gathered features / MLP buffer
     * 10: [N, max(C, out_C)] GEMM partial result */
    ensure_scratch(r, 8, (size_t)N * C * sizeof(float));
    ensure_scratch(r, 9, (size_t)N * 4 * C * sizeof(float));
    ensure_scratch(r, 10, (size_t)N * C * sizeof(float));

    CUdeviceptr d_tmp = r->scratch[8];
    CUdeviceptr d_gathered = r->scratch[9];
    CUdeviceptr d_partial = r->scratch[10];

    /* 1. Sparse conv: feats -> tmp */
    t2_op_sparse_conv3d(ops, s, d_tmp, d_feats, conv_w, conv_b,
                         d_gather_map, d_gathered, d_partial, N, C, C);

    /* 2. LayerNorm: tmp -> tmp */
    t2_op_layernorm(ops, s, d_tmp, d_tmp, norm_w, norm_b, N, C);

    /* 3. MLP: Linear(C->4C) -> GELU -> Linear(4C->C) */
    CUdeviceptr d_mlp = r->scratch[9];  /* reuse as [N, 4C] */
    t2_op_gemm(ops, s, d_mlp, mlp0_w, d_tmp, mlp0_b, 4 * C, C, N);
    t2_op_gelu(ops, s, d_mlp, N * 4 * C);
    t2_op_gemm(ops, s, d_tmp, mlp2_w, d_mlp, mlp2_b, C, 4 * C, N);

    /* 4. Residual: feats += tmp */
    t2_op_residual_add(ops, s, d_feats, d_tmp, N * C);
}

/* Run shape decoder forward pass.
 * slat: [N, 32] structured latent on CPU
 * coords: [N, 4] int32 on CPU
 * out_feats: caller-allocated [N_out, 7] on CPU
 * out_coords: caller-allocated [N_out, 4] int32 on CPU
 * Returns N_out (number of output voxels after C2S upsampling).
 * NOTE: Currently runs ConvNeXt blocks only (no C2S), outputs at same resolution. */
int cuda_trellis2_run_shape_decoder(cuda_trellis2_runner *r,
                                      const float *slat, const int32_t *coords, int N,
                                      float *out_feats, int32_t *out_coords,
                                      int *out_N) {
    if (!r->shape_dec.loaded) {
        fprintf(stderr, "T2: shape decoder not loaded\n"); return -1;
    }
    t2_ops *ops = &r->ops;
    CUstream s = r->stream;

    /* Use F16 MMA GEMM for shape decoder weights */
    int saved_f32 = ops->use_f32_gemm;
    int saved_mma = ops->use_mma_gemm;
    if (r->ops.sm_version >= 70) {
        ops->use_f32_gemm = 0;
        ops->use_mma_gemm = 1;
    }

    int C = r->shape_dec.channels[0];  /* 1024 */

    /* Upload input */
    CUdeviceptr d_slat = cu_upload_raw(slat, (size_t)N * 32 * sizeof(float));
    CUdeviceptr d_coords = cu_upload_raw(coords, (size_t)N * 4 * sizeof(int));

    /* from_latent: [N, 32] -> [N, 1024] */
    CUdeviceptr d_feats;
    cuMemAlloc(&d_feats, (size_t)N * C * sizeof(float));
    t2_op_gemm(ops, s, d_feats, r->shape_dec.from_latent_w, d_slat, r->shape_dec.from_latent_b,
               C, 32, N);
    cuMemFree(d_slat);

    fprintf(stderr, "T2: shape decoder: from_latent -> [%d, %d]\n", N, C);

    /* Build hash table on CPU, upload to GPU */
    /* We need the sp3d hash for neighbor lookup. Build on CPU, upload keys/vals. */
    /* Using the sp3d_hash from sparse3d.h */
    sp3d_hash *hash = sp3d_hash_build(coords, N);
    CUdeviceptr d_hash_keys = cu_upload_raw(hash->keys, (size_t)hash->capacity * sizeof(uint64_t));
    CUdeviceptr d_hash_vals = cu_upload_raw(hash->vals, (size_t)hash->capacity * sizeof(int32_t));
    int hash_cap = hash->capacity;

    /* Build gather map on GPU */
    CUdeviceptr d_gather_map;
    cuMemAlloc(&d_gather_map, (size_t)N * 27 * sizeof(int));
    t2_op_sparse_build_gather_map(ops, s, d_gather_map, d_coords, N,
                                   d_hash_keys, d_hash_vals, hash_cap);

    fprintf(stderr, "T2: shape decoder: gather map built (N=%d, hash_cap=%d)\n", N, hash_cap);

    /* Current CPU-side copies of features and coords for C2S round-trips */
    int32_t *cur_coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    memcpy(cur_coords, coords, (size_t)N * 4 * sizeof(int32_t));

    /* Run ConvNeXt blocks + C2S for each stage */
    for (int stage = 0; stage < 4; stage++) {
        int nblk = r->shape_dec.n_convnext[stage];
        C = r->shape_dec.channels[stage];
        fprintf(stderr, "T2: shape dec: stage %d: %d ConvNeXt(%d), N=%d\n",
                stage, nblk, C, N);

        struct timespec ts0; clock_gettime(CLOCK_MONOTONIC, &ts0);

        for (int b = 0; b < nblk; b++) {
            run_shape_convnext(r, d_feats, N, C,
                d_gather_map,
                r->shape_dec.convnext[stage][b].conv_w,
                r->shape_dec.convnext[stage][b].conv_b,
                r->shape_dec.convnext[stage][b].norm_w,
                r->shape_dec.convnext[stage][b].norm_b,
                r->shape_dec.convnext[stage][b].mlp0_w,
                r->shape_dec.convnext[stage][b].mlp0_b,
                r->shape_dec.convnext[stage][b].mlp2_w,
                r->shape_dec.convnext[stage][b].mlp2_b);
        }

        cuStreamSynchronize(s);
        struct timespec ts1; clock_gettime(CLOCK_MONOTONIC, &ts1);
        double dt_conv = (ts1.tv_sec - ts0.tv_sec) * 1000.0 + (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
        fprintf(stderr, "T2: shape dec: stage %d convnext %.1f ms\n", stage, dt_conv);

        /* C2S upsampling (hybrid: CPU subdivision + GPU conv) */
        if (stage < 3) {
            /* Download features to CPU */
            float *feats_cpu = (float *)malloc((size_t)N * C * sizeof(float));
            cuMemcpyDtoH(feats_cpu, d_feats, (size_t)N * C * sizeof(float));

            /* Build CPU sparse tensor for C2S */
            sp3d_tensor *t_cpu = sp3d_create(cur_coords, feats_cpu, N, C, 1);
            free(feats_cpu);

            /* Run C2S on CPU using the full CPU decoder's C2S weights */
            sp3d_tensor *t_new = t2sd_c2s_forward(t_cpu, &r->shape_dec_cpu->c2s[stage], 4);

            int N_new = t_new->N;
            int C_new = t_new->C;
            fprintf(stderr, "T2: shape dec: c2s %d->%d: N %d -> %d, C %d -> %d\n",
                    stage, stage + 1, N, N_new, C, C_new);

            /* Free old GPU buffers */
            cuMemFree(d_feats); cuMemFree(d_coords);
            cuMemFree(d_gather_map); cuMemFree(d_hash_keys); cuMemFree(d_hash_vals);
            sp3d_hash_free(hash); sp3d_free(t_cpu);

            /* Upload new features and coords to GPU */
            N = N_new; C = C_new;
            d_feats = cu_upload_raw(t_new->feats, (size_t)N * C * sizeof(float));
            d_coords = cu_upload_raw(t_new->coords, (size_t)N * 4 * sizeof(int));

            /* Save CPU coords for next round */
            free(cur_coords);
            cur_coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
            memcpy(cur_coords, t_new->coords, (size_t)N * 4 * sizeof(int32_t));

            /* Rebuild hash table and gather map */
            hash = sp3d_hash_build(t_new->coords, N);
            d_hash_keys = cu_upload_raw(hash->keys, (size_t)hash->capacity * sizeof(uint64_t));
            d_hash_vals = cu_upload_raw(hash->vals, (size_t)hash->capacity * sizeof(int32_t));
            hash_cap = hash->capacity;

            cuMemAlloc(&d_gather_map, (size_t)N * 27 * sizeof(int));
            t2_op_sparse_build_gather_map(ops, s, d_gather_map, d_coords, N,
                                           d_hash_keys, d_hash_vals, hash_cap);
            sp3d_free(t_new);

            struct timespec ts2; clock_gettime(CLOCK_MONOTONIC, &ts2);
            double dt_c2s = (ts2.tv_sec - ts1.tv_sec) * 1000.0 + (ts2.tv_nsec - ts1.tv_nsec) / 1e6;
            fprintf(stderr, "T2: shape dec: c2s %.1f ms, N=%d\n", dt_c2s, N);
        }
    }

    /* output_layer: [N, C=64] -> [N, out_ch] */
    int out_ch = r->shape_dec.channels[4]; /* 64 -> detect from weight */
    {
        /* Detect actual output channels from out_w shape:
         * For shape decoder: 7, for texture decoder: 6 */
        /* For now use a fixed value based on what was loaded */
        out_ch = 7;  /* shape decoder default */
    }

    cuStreamSynchronize(s);
    CUdeviceptr d_output;
    cuMemAlloc(&d_output, (size_t)N * out_ch * sizeof(float));
    t2_op_gemm(ops, s, d_output, r->shape_dec.out_w, d_feats, r->shape_dec.out_b,
               out_ch, C, N);
    cuStreamSynchronize(s);

    /* Download results — caller must have allocated enough for max possible N */
    cuMemcpyDtoH(out_feats, d_output, (size_t)N * out_ch * sizeof(float));
    memcpy(out_coords, cur_coords, (size_t)N * 4 * sizeof(int32_t));
    *out_N = N;
    cuMemFree(d_output);

    /* Cleanup */
    free(cur_coords);
    cuMemFree(d_feats);
    cuMemFree(d_coords);
    cuMemFree(d_gather_map);
    cuMemFree(d_hash_keys);
    cuMemFree(d_hash_vals);
    sp3d_hash_free(hash);

    ops->use_f32_gemm = saved_f32;
    ops->use_mma_gemm = saved_mma;
    return 0;
}

int cuda_trellis2_run_stage2_dit(cuda_trellis2_runner *r,
                                  const float *x_t, float timestep,
                                  const float *cond_features,
                                  const int32_t *coords, int N,
                                  float *output) {
    /* Invalidate cache since CFG uses different cond per pass */
    cuda_trellis2_invalidate_kv_cache(r);
    if (!r->stage2_loaded) {
        fprintf(stderr, "T2: Stage 2 not loaded\n"); return -1;
    }
    CUstream s = r->stream;
    dit_model_gpu *m = &r->stage2;
    int in_ch = m->in_channels;  /* 32 */
    int ctx_len = DINO_SEQ_LEN;

    /* Compute 3D RoPE tables from sparse coords on CPU */
    int n_freqs = m->n_rope_freqs;  /* 21 */
    float rope_theta = 10000.0f;
    float *freqs = (float *)malloc((size_t)n_freqs * sizeof(float));
    for (int j = 0; j < n_freqs; j++)
        freqs[j] = 1.0f / powf(rope_theta, (float)j / (float)n_freqs);

    size_t table_sz = (size_t)N * 3 * n_freqs * sizeof(float);
    float *cos_tab = (float *)malloc(table_sz);
    float *sin_tab = (float *)malloc(table_sz);

    for (int i = 0; i < N; i++) {
        /* coords[i] = (batch, z, y, x) */
        float cz = (float)coords[i * 4 + 1];
        float cy = (float)coords[i * 4 + 2];
        float cx = (float)coords[i * 4 + 3];
        float c[3] = {cz, cy, cx};
        for (int axis = 0; axis < 3; axis++) {
            for (int j = 0; j < n_freqs; j++) {
                float theta = c[axis] * freqs[j];
                cos_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = cosf(theta);
                sin_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = sinf(theta);
            }
        }
    }

    CUdeviceptr d_rope_cos = cu_upload_raw(cos_tab, table_sz);
    CUdeviceptr d_rope_sin = cu_upload_raw(sin_tab, table_sz);
    free(freqs); free(cos_tab); free(sin_tab);

    /* Upload input and conditioning */
    CUdeviceptr d_x = cu_upload_raw(x_t, (size_t)N * in_ch * sizeof(float));
    CUdeviceptr d_cond = cu_upload_raw(cond_features, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)N * in_ch * sizeof(float));

    /* Run Stage 2 DiT forward */
    run_sparse_dit_forward(r, m, r->stage2_blocks_cpu,
                            d_x, timestep, d_cond, d_out, N, d_rope_cos, d_rope_sin, 2);

    cuStreamSynchronize(s);
    cuMemcpyDtoH(output, d_out, (size_t)N * in_ch * sizeof(float));

    cuMemFree(d_x); cuMemFree(d_cond); cuMemFree(d_out);
    cuMemFree(d_rope_cos); cuMemFree(d_rope_sin);
    return 0;
}

int cuda_trellis2_run_stage3_dit(cuda_trellis2_runner *r,
                                  const float *x_t, float timestep,
                                  const float *cond_features,
                                  const int32_t *coords, int N,
                                  float *output) {
    if (!r->stage3_loaded) {
        fprintf(stderr, "T2: Stage 3 not loaded\n"); return -1;
    }
    CUstream s = r->stream;
    dit_model_gpu *m = &r->stage3;
    int in_ch = m->in_channels;  /* 64 (noise_32 + shape_slat_32) */
    int ctx_len = DINO_SEQ_LEN;

    /* Compute 3D RoPE tables from sparse coords (same as Stage 2) */
    int n_freqs = m->n_rope_freqs;
    float rope_theta = 10000.0f;
    float *freqs = (float *)malloc((size_t)n_freqs * sizeof(float));
    for (int j = 0; j < n_freqs; j++)
        freqs[j] = 1.0f / powf(rope_theta, (float)j / (float)n_freqs);

    size_t table_sz = (size_t)N * 3 * n_freqs * sizeof(float);
    float *cos_tab = (float *)malloc(table_sz);
    float *sin_tab = (float *)malloc(table_sz);
    for (int i = 0; i < N; i++) {
        float cz = (float)coords[i * 4 + 1];
        float cy = (float)coords[i * 4 + 2];
        float cx = (float)coords[i * 4 + 3];
        float c[3] = {cz, cy, cx};
        for (int axis = 0; axis < 3; axis++)
            for (int j = 0; j < n_freqs; j++) {
                float theta = c[axis] * freqs[j];
                cos_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = cosf(theta);
                sin_tab[(size_t)i * 3 * n_freqs + axis * n_freqs + j] = sinf(theta);
            }
    }
    CUdeviceptr d_rope_cos = cu_upload_raw(cos_tab, table_sz);
    CUdeviceptr d_rope_sin = cu_upload_raw(sin_tab, table_sz);
    free(freqs); free(cos_tab); free(sin_tab);

    /* Upload input [N, 64] and conditioning */
    CUdeviceptr d_x = cu_upload_raw(x_t, (size_t)N * in_ch * sizeof(float));
    CUdeviceptr d_cond = cu_upload_raw(cond_features, (size_t)ctx_len * DIT_COND_DIM * sizeof(float));
    int out_ch = m->out_channels;  /* 32 for Stage 3 */
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)N * out_ch * sizeof(float));

    run_sparse_dit_forward(r, m, r->stage3_blocks_cpu,
                            d_x, timestep, d_cond, d_out, N, d_rope_cos, d_rope_sin, 3);

    cuStreamSynchronize(s);
    cuMemcpyDtoH(output, d_out, (size_t)N * out_ch * sizeof(float));

    cuMemFree(d_x); cuMemFree(d_cond); cuMemFree(d_out);
    cuMemFree(d_rope_cos); cuMemFree(d_rope_sin);
    return 0;
}

int cuda_trellis2_run_dinov3(cuda_trellis2_runner *r,
                              const float *image_f32, float *output) {
    CUstream s = r->stream;
    int seq = DINO_SEQ_LEN;
    int dim = DINO_HIDDEN;

    /* Upload normalized image [3, 512, 512] */
    CUdeviceptr d_image = cu_upload_raw(image_f32,
        (size_t)3 * DINO_IMG_SIZE * DINO_IMG_SIZE * sizeof(float));
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, (size_t)seq * dim * sizeof(float));

    run_dinov3(r, d_image, d_out);

    cuStreamSynchronize(s);
    cuMemcpyDtoH(output, d_out, (size_t)seq * dim * sizeof(float));

    cuMemFree(d_image); cuMemFree(d_out);
    return 0;
}
