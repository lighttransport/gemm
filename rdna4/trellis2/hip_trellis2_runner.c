/*
 * hip_trellis2_runner.c - HIP/ROCm TRELLIS.2 Stage 1 runner (RDNA4)
 *
 * GPU-accelerated Stage 1: DINOv3 on CPU, DiT 30-block on GPU, Decoder on GPU.
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 * F16 GEMM weights + F32 compute. HIPRTC runtime compilation.
 *
 * Cross-attention KV cache: precomputed once per image, reused across 12 steps.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "hip_trellis2_runner.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#include "hip_trellis2_kernels.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* HIP_LAUNCH_PARAM_* constants (not in rocew.h; defined per ROCm hipModuleApi.h) */
#ifndef HIP_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)(uintptr_t)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void *)(uintptr_t)0x02)
#define HIP_LAUNCH_PARAM_END            ((void *)(uintptr_t)0x03)
#endif

/* ======================================================================== */
/* Model constants (Stage 1)                                               */
/* ======================================================================== */

#define DIT_DIM      1536
#define DIT_HEADS    12
#define DIT_HEAD_DIM 128
#define DIT_FFN      8192
#define DIT_DEPTH    30
#define DIT_IN_CH    8
#define DIT_GRID     16
#define DIT_N_TOK    (DIT_GRID * DIT_GRID * DIT_GRID)  /* 4096 */
#define DIT_COND_DIM 1024
#define DIT_T_HALF   128   /* sinusoidal embed half-dim (256/2) */

/* Decoder */
#define DEC_GRID   16
#define DEC_OUT_GRID 64

/* ======================================================================== */
/* GPU weight structures                                                    */
/* ======================================================================== */

/* F16 weight tensor + optional F32 bias */
typedef struct {
    void *w;   /* F16 weight, device pointer */
    void *b;   /* F32 bias, device pointer (may be NULL) */
} t2_wt;

typedef struct {
    /* Self-attention */
    t2_wt sa_qkv;          /* [3*DIT_DIM, DIT_DIM] */
    void *sa_q_norm;        /* [DIT_HEADS*DIT_HEAD_DIM] F32 */
    void *sa_k_norm;        /* [DIT_HEADS*DIT_HEAD_DIM] F32 */
    t2_wt sa_out;          /* [DIT_DIM, DIT_DIM] */
    /* Cross-attention */
    void *norm2_w, *norm2_b;  /* [DIT_DIM] F32 */
    t2_wt ca_q;             /* [DIT_DIM, DIT_DIM] */
    void *ca_q_norm;        /* [DIT_HEADS*DIT_HEAD_DIM] F32 */
    void *ca_k_norm;        /* [DIT_HEADS*DIT_HEAD_DIM] F32 */
    t2_wt ca_kv;            /* [2*DIT_DIM, DIT_COND_DIM] */
    t2_wt ca_out;           /* [DIT_DIM, DIT_DIM] */
    /* MLP */
    t2_wt mlp_fc1;          /* [DIT_FFN, DIT_DIM] */
    t2_wt mlp_fc2;          /* [DIT_DIM, DIT_FFN] */
    /* Per-block modulation bias */
    void *mod_bias;         /* [6*DIT_DIM] F32 */
} t2_block_gpu;

typedef struct {
    void *conv_w;  /* [Co, Ci, 3,3,3] F32 */
    void *conv_b;  /* [Co] F32 */
    /* ResBlock ChannelLN + Conv3D pairs */
    void *rb_ln1_w, *rb_ln1_b;
    void *rb_c1_w, *rb_c1_b;
    void *rb_ln2_w, *rb_ln2_b;
    void *rb_c2_w, *rb_c2_b;
} t2_dec_layer;

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

struct hip_trellis2_runner {
    int device_id, verbose;

    hipModule_t mod;

    /* GEMM and common ops (from hip_kernels_common_src) */
    hipFunction_t fn_gemm;           /* gemm_tiled_f32_f32 */
    hipFunction_t fn_layernorm;      /* layernorm_f32 */

    /* TRELLIS.2-specific ops */
    hipFunction_t fn_adaln;
    hipFunction_t fn_gated_add;
    hipFunction_t fn_residual_add;
    hipFunction_t fn_modulation;
    hipFunction_t fn_t_embed;        /* timestep_embed_cossin_f32 */
    hipFunction_t fn_silu;
    hipFunction_t fn_gelu;
    hipFunction_t fn_split_qkv;
    hipFunction_t fn_split_kv;
    hipFunction_t fn_rms_norm_ph;
    hipFunction_t fn_rope_3d;
    hipFunction_t fn_ln_noaffine;
    hipFunction_t fn_flash_sa;       /* flash_attn_sa_f32 */
    hipFunction_t fn_cross_attn;     /* cross_attn_tiled_f32 */
    hipFunction_t fn_broadcast_bias;

    /* Decoder ops */
    hipFunction_t fn_conv3d;
    hipFunction_t fn_ch_ln3d;
    hipFunction_t fn_pix_shuf3d;
    hipFunction_t fn_euler_step;
    hipFunction_t fn_cfg_combine;
    hipFunction_t fn_compute_x0;
    hipFunction_t fn_cfg_rescale;

    /* ---- DiT weights ---- */
    /* Global */
    t2_wt dit_input;         /* input_layer: [DIT_DIM, DIT_IN_CH] */
    t2_wt dit_t_fc1;         /* [DIT_DIM, DIT_T_HALF*2] */
    t2_wt dit_t_fc2;         /* [DIT_DIM, DIT_DIM] */
    t2_wt dit_ada_mod;       /* adaLN_modulation: [6*DIT_DIM, DIT_DIM] */
    t2_wt dit_out;           /* out_layer: [DIT_IN_CH, DIT_DIM] */
    /* 30 per-block weights */
    t2_block_gpu blocks[DIT_DEPTH];

    /* RoPE precomputed phases: cos/sin [N_TOK, 3, n_freqs] */
    void *d_rope_cos, *d_rope_sin;
    int n_rope_freqs;

    /* ---- Cross-attention KV cache ---- */
    void *ca_K_cache[DIT_DEPTH];  /* [DIT_COND_LEN, DIT_DIM] F32 */
    void *ca_V_cache[DIT_DEPTH];  /* [DIT_COND_LEN, DIT_DIM] F32 */
    int ca_kv_valid;
    int ca_kv_len;   /* number of conditioning tokens (1029) */

    /* ---- Decoder weights ---- */
    t2_dec_layer dec_input;       /* Conv3D input: [512, 8, 3,3,3] */
    t2_dec_layer dec_mid[2];      /* ResBlock3d × 2 at 16³ */
    t2_dec_layer dec_res16[2];    /* ResBlock3d × 2 at 16³ */
    void *dec_up1_w, *dec_up1_b; /* Conv3D for pixel_shuffle: [1024→128] */
    t2_dec_layer dec_res32[2];    /* ResBlock3d × 2 at 32³ */
    void *dec_up2_w, *dec_up2_b; /* Conv3D for pixel_shuffle: [256→32] */
    t2_dec_layer dec_res64[2];    /* ResBlock3d × 2 at 64³ */
    void *dec_out_ln_w, *dec_out_ln_b; /* ChannelLayerNorm */
    void *dec_out_w, *dec_out_b; /* Conv3D out: [1, 32, 3,3,3] */
    int dec_loaded;

    /* ---- Activation buffers ---- */
    void *d_h;         /* [DIT_N_TOK, DIT_DIM] main hidden state */
    void *d_ln_h;      /* [DIT_N_TOK, DIT_DIM] after adaLN */
    void *d_qkv;       /* [DIT_N_TOK, 3*DIT_DIM] QKV scratch */
    void *d_Q, *d_K, *d_V;  /* [DIT_N_TOK, DIT_DIM] */
    void *d_ca_Q;      /* [DIT_N_TOK, DIT_DIM] cross-attn Q */
    void *d_attn_out;  /* [DIT_N_TOK, DIT_DIM] attention output */
    void *d_mlp_mid;   /* [DIT_N_TOK, DIT_FFN] MLP intermediate */
    void *d_t_emb;     /* [DIT_T_HALF*2] sinusoidal embed */
    void *d_t_silu;    /* [DIT_DIM] after FC1+SiLU */
    void *d_ada_out;   /* [6*DIT_DIM] shared modulation */
    void *d_mod;       /* [6*DIT_DIM] per-block modulation */
    void *d_noise;     /* [DIT_N_TOK, DIT_IN_CH] input */
    void *d_vel;       /* [DIT_N_TOK, DIT_IN_CH] output velocity */
    void *d_cond;      /* [1029, DIT_COND_DIM] conditioning features */
    /* Decoder scratch */
    void *d_dec[4];    /* decoder intermediate volumes */
    size_t d_dec_sz[4];

    int dit_loaded;

    /* Block 0 debug sink (set by hip_trellis2_dump_b0_detail; zero otherwise) */
    hip_trellis2_b0_dbg *b0_dbg;
};

/* helper: hipMemcpy device buffer [n floats] to host if dst non-null */
static void dbg_dl(void *host, void *dev, size_t n) {
    if (host && dev) {
        hipMemcpy(host, dev, n * sizeof(float), hipMemcpyDeviceToHost);
    }
}

/* ======================================================================== */
/* Utilities                                                               */
/* ======================================================================== */

static void *gpu_upload_f32(const float *data, size_t n) {
    return hip_upload_raw(data, n * sizeof(float));
}

/* Convert BF16 array to F32 in place (CPU) */
static float *bf16_to_f32(const uint16_t *src, size_t n) {
    float *out = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)src[i] << 16;
        memcpy(&out[i], &bits, 4);
    }
    return out;
}

/* Convert F16 array to F32 (CPU) */
static float *f16_to_f32(const uint16_t *src, size_t n) {
    float *out = (float *)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        uint16_t h = src[i];
        uint32_t s = (uint32_t)(h & 0x8000) << 16;
        uint32_t e = (h >> 10) & 0x1f;
        uint32_t m = h & 0x3ff;
        uint32_t f;
        if (e == 0) f = m ? (s | ((127-14)<<23) | (m<<13)) : s;
        else if (e == 31) f = s | 0x7f800000 | (m << 13);
        else f = s | ((e + 127 - 15) << 23) | (m << 13);
        memcpy(&out[i], &f, 4);
    }
    return out;
}

/* Convert F32 to F16 (CPU) */
static uint16_t *f32_to_f16_buf(const float *src, size_t n) {
    uint16_t *out = (uint16_t *)malloc(n * sizeof(uint16_t));
    for (size_t i = 0; i < n; i++) {
        out[i] = hip_f32_to_f16(src[i]);
    }
    return out;
}

/* Upload safetensors tensor as F16 GPU (BF16/F32/F16 source) */
static void *st_upload_f16(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose >= 2) fprintf(stderr, "  [skip] %s\n", name);
        return NULL;
    }
    const char *dtype = safetensors_dtype(st, idx);
    size_t nb = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    uint16_t *f16 = NULL;
    size_t n_elem = 0;

    if (strcmp(dtype, "F16") == 0) {
        /* Already F16 — upload directly */
        void *d = hip_upload_raw(data, nb);
        if (verbose >= 2) fprintf(stderr, "  %s: F16 (%.1f MB)\n", name, nb/1048576.0f);
        return d;
    } else if (strcmp(dtype, "BF16") == 0) {
        n_elem = nb / 2;
        float *tmp = bf16_to_f32((const uint16_t *)data, n_elem);
        f16 = f32_to_f16_buf(tmp, n_elem);
        free(tmp);
    } else if (strcmp(dtype, "F32") == 0) {
        n_elem = nb / 4;
        f16 = f32_to_f16_buf((const float *)data, n_elem);
    } else {
        fprintf(stderr, "  [skip] %s: unsupported dtype %s\n", name, dtype);
        return NULL;
    }

    void *d = hip_upload_raw(f16, n_elem * sizeof(uint16_t));
    free(f16);
    if (verbose >= 2) fprintf(stderr, "  %s: %s->F16 (%.1f MB)\n", name, dtype,
                               n_elem * 2 / 1048576.0f);
    return d;
}

/* Upload safetensors tensor as F32 GPU */
static void *st_upload_f32(st_context *st, const char *name, int verbose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        if (verbose >= 2) fprintf(stderr, "  [skip] %s\n", name);
        return NULL;
    }
    const char *dtype = safetensors_dtype(st, idx);
    size_t nb = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);

    float *f32 = NULL;
    size_t n_elem = 0;

    if (strcmp(dtype, "F32") == 0) {
        void *d = hip_upload_raw(data, nb);
        if (verbose >= 2) fprintf(stderr, "  %s: F32 (%.1f MB)\n", name, nb/1048576.0f);
        return d;
    } else if (strcmp(dtype, "BF16") == 0) {
        n_elem = nb / 2;
        f32 = bf16_to_f32((const uint16_t *)data, n_elem);
    } else if (strcmp(dtype, "F16") == 0) {
        n_elem = nb / 2;
        f32 = f16_to_f32((const uint16_t *)data, n_elem);
    } else {
        fprintf(stderr, "  [skip] %s: unsupported dtype %s\n", name, dtype);
        return NULL;
    }

    void *d = hip_upload_raw(f32, n_elem * sizeof(float));
    free(f32);
    if (verbose >= 2) fprintf(stderr, "  %s: %s->F32 (%.1f MB)\n", name, dtype,
                               n_elem * 4 / 1048576.0f);
    return d;
}

/* Ensure scratch buffer is large enough, reallocate if needed */
static void *ensure_buf(void **buf, size_t bytes) {
    if (*buf) hipFree(*buf);
    hipMalloc(buf, bytes);
    return *buf;
}

static void free_hip(void **p) { if (*p) { hipFree(*p); *p = NULL; } }

/* ======================================================================== */
/* RoPE phase precomputation                                               */
/* ======================================================================== */

/* Compute 3D RoPE phases for a 16³ grid (complex-pair convention).
 * Output: cos/sin arrays of shape [N_TOK, 3, n_freqs] where n_freqs=21.
 * This matches TRELLIS.2's official implementation.
 */
static void compute_rope_phases(int grid, int n_freqs,
                                 float **out_cos, float **out_sin,
                                 int *out_n_tok) {
    int N = grid * grid * grid;
    *out_n_tok = N;
    size_t sz = (size_t)N * 3 * n_freqs;
    *out_cos = (float *)malloc(sz * sizeof(float));
    *out_sin = (float *)malloc(sz * sizeof(float));

    /* Frequency bands: freqs[j] = 1 / theta^(j/n_freqs), theta=10000
     * Must match CPU t2dit_precompute_rope exactly. */
    float freqs[21];
    for (int i = 0; i < n_freqs; i++) {
        freqs[i] = 1.0f / powf(10000.0f, (float)i / (float)n_freqs);
    }

    int tok = 0;
    for (int z = 0; z < grid; z++) {
        for (int y = 0; y < grid; y++) {
            for (int x = 0; x < grid; x++, tok++) {
                int coords[3] = {z, y, x};
                for (int axis = 0; axis < 3; axis++) {
                    float pos = (float)coords[axis];
                    for (int f = 0; f < n_freqs; f++) {
                        float angle = pos * freqs[f];
                        (*out_cos)[tok * 3 * n_freqs + axis * n_freqs + f] = cosf(angle);
                        (*out_sin)[tok * 3 * n_freqs + axis * n_freqs + f] = sinf(angle);
                    }
                }
            }
        }
    }
}

/* ======================================================================== */
/* Kernel launch helpers                                                    */
/* ======================================================================== */

static int gemm(hip_trellis2_runner *r, void *Y, const void *W, const void *X,
                const void *bias, int n_out, int n_in, int n_tok) {
    /* Grid: (ceil(n_out/64), ceil(n_tok/16)), Block: (16,16) */
    int gdx = (n_out + 63) / 64;
    int gdy = (n_tok + 15) / 16;
    void *args[] = { &Y, &W, &X, &bias, &n_out, &n_in, &n_tok };
    return hipModuleLaunchKernel(r->fn_gemm,
        gdx, gdy, 1, 16, 16, 1, 0, 0, args, NULL) == hipSuccess ? 0 : -1;
}

/* LAUNCH macro intentionally removed — use pointer-array style directly */

/* ======================================================================== */
/* Init / Free                                                             */
/* ======================================================================== */

hip_trellis2_runner *hip_trellis2_init(int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "T2-HIP: failed to load ROCm/HIPRTC libraries\n");
        return NULL;
    }
    HIP_CHECK_NULL(hipSetDevice(device_id));

    hip_trellis2_runner *r = (hip_trellis2_runner *)calloc(1, sizeof(*r));
    r->device_id = device_id;
    r->verbose = verbose;

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device_id);
    fprintf(stderr, "T2-HIP: GPU %d: %s (%.1f GB, %s)\n",
            device_id, prop.name,
            (double)prop.totalGlobalMem / (1<<30), prop.gcnArchName);

    /* Compile kernels: common + trellis2-specific */
    size_t src_len = strlen(hip_kernels_common_src) +
                     strlen(hip_trellis2_specific_kernels) + 16;
    char *full = (char *)malloc(src_len);
    snprintf(full, src_len, "%s%s", hip_kernels_common_src, hip_trellis2_specific_kernels);

    int ok = hip_compile_kernels(&r->mod, device_id, full, "trellis2", verbose, "T2-HIP");
    free(full);
    if (ok < 0) {
        fprintf(stderr, "T2-HIP: kernel compilation failed\n");
        free(r); return NULL;
    }

#define GET_FN(fn, name) do { \
    if (hipModuleGetFunction(&r->fn, r->mod, name) != hipSuccess) { \
        fprintf(stderr, "T2-HIP: missing kernel %s\n", name); \
        free(r); return NULL; \
    } \
} while(0)

#define LOAD_FN(field, name) do { \
    hipError_t _e = hipModuleGetFunction(&r->field, r->mod, name); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "T2-HIP: missing kernel '%s' (err=%d)\n", name, (int)_e); \
        free(r); return NULL; \
    } \
    if (verbose) fprintf(stderr, "  kernel OK: %s\n", name); \
} while(0)

    LOAD_FN(fn_gemm,         "gemm_tiled_f32_f32");
    LOAD_FN(fn_layernorm,    "layernorm_f32");
    LOAD_FN(fn_adaln,        "adaln_f32");
    LOAD_FN(fn_gated_add,    "gated_add_f32");
    LOAD_FN(fn_residual_add, "residual_add_f32");
    LOAD_FN(fn_modulation,   "modulation_f32");
    LOAD_FN(fn_t_embed,      "timestep_embed_cossin_f32");
    LOAD_FN(fn_silu,         "silu_inplace_f32");
    LOAD_FN(fn_gelu,         "gelu_inplace_f32");
    LOAD_FN(fn_split_qkv,   "split_qkv_chunk_f32");
    LOAD_FN(fn_split_kv,    "split_kv_chunk_f32");
    LOAD_FN(fn_rms_norm_ph,  "rms_norm_perhead_f32");
    LOAD_FN(fn_rope_3d,      "rope_3d_f32");
    LOAD_FN(fn_ln_noaffine,  "layernorm_noaffine_f32");
    LOAD_FN(fn_flash_sa,     "flash_attn_sa_f32");
    LOAD_FN(fn_cross_attn,   "cross_attn_tiled_f32");
    LOAD_FN(fn_broadcast_bias,"broadcast_bias_f32");
    LOAD_FN(fn_conv3d,       "conv3d_k3_f32");
    LOAD_FN(fn_ch_ln3d,      "channel_layernorm_3d_f32");
    LOAD_FN(fn_pix_shuf3d,   "pixel_shuffle_3d_f32");
    LOAD_FN(fn_euler_step,   "euler_step_f32");
    LOAD_FN(fn_cfg_combine,  "cfg_combine_f32");
    LOAD_FN(fn_compute_x0,   "compute_x0_f32");
    LOAD_FN(fn_cfg_rescale,  "cfg_rescale_f32");
#undef LOAD_FN

    fprintf(stderr, "T2-HIP: all %d kernels loaded OK\n", 24);

    /* Allocate activation buffers */
    hipMalloc(&r->d_h,        DIT_N_TOK * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_ln_h,     DIT_N_TOK * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_qkv,      DIT_N_TOK * 3 * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_Q,        DIT_N_TOK * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_K,        DIT_N_TOK * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_V,        DIT_N_TOK * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_ca_Q,     DIT_N_TOK * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_attn_out, DIT_N_TOK * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_mlp_mid,  DIT_N_TOK * DIT_FFN * sizeof(float));
    hipMalloc(&r->d_t_emb,    DIT_T_HALF * 2 * sizeof(float));
    hipMalloc(&r->d_t_silu,   DIT_DIM * sizeof(float));
    hipMalloc(&r->d_ada_out,  6 * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_mod,      6 * DIT_DIM * sizeof(float));
    hipMalloc(&r->d_noise,    DIT_N_TOK * DIT_IN_CH * sizeof(float));
    hipMalloc(&r->d_vel,      DIT_N_TOK * DIT_IN_CH * sizeof(float));
    hipMalloc(&r->d_cond,     1029 * DIT_COND_DIM * sizeof(float));

    return r;
}

void hip_trellis2_free(hip_trellis2_runner *r) {
    if (!r) return;

    /* DiT weights */
    free_hip(&r->dit_input.w); free_hip(&r->dit_input.b);
    free_hip(&r->dit_t_fc1.w); free_hip(&r->dit_t_fc1.b);
    free_hip(&r->dit_t_fc2.w); free_hip(&r->dit_t_fc2.b);
    free_hip(&r->dit_ada_mod.w); free_hip(&r->dit_ada_mod.b);
    free_hip(&r->dit_out.w); free_hip(&r->dit_out.b);
    free_hip(&r->d_rope_cos); free_hip(&r->d_rope_sin);

    for (int i = 0; i < DIT_DEPTH; i++) {
        t2_block_gpu *b = &r->blocks[i];
        free_hip(&b->sa_qkv.w); free_hip(&b->sa_qkv.b);
        free_hip(&b->sa_q_norm); free_hip(&b->sa_k_norm);
        free_hip(&b->sa_out.w); free_hip(&b->sa_out.b);
        free_hip(&b->norm2_w); free_hip(&b->norm2_b);
        free_hip(&b->ca_q.w); free_hip(&b->ca_q.b);
        free_hip(&b->ca_q_norm); free_hip(&b->ca_k_norm);
        free_hip(&b->ca_kv.w); free_hip(&b->ca_kv.b);
        free_hip(&b->ca_out.w); free_hip(&b->ca_out.b);
        free_hip(&b->mlp_fc1.w); free_hip(&b->mlp_fc1.b);
        free_hip(&b->mlp_fc2.w); free_hip(&b->mlp_fc2.b);
        free_hip(&b->mod_bias);
        free_hip(&r->ca_K_cache[i]);
        free_hip(&r->ca_V_cache[i]);
    }

    /* Activation buffers */
    free_hip(&r->d_h); free_hip(&r->d_ln_h); free_hip(&r->d_qkv);
    free_hip(&r->d_Q); free_hip(&r->d_K); free_hip(&r->d_V);
    free_hip(&r->d_ca_Q); free_hip(&r->d_attn_out); free_hip(&r->d_mlp_mid);
    free_hip(&r->d_t_emb); free_hip(&r->d_t_silu);
    free_hip(&r->d_ada_out); free_hip(&r->d_mod);
    free_hip(&r->d_noise); free_hip(&r->d_vel); free_hip(&r->d_cond);

    /* Decoder */
    for (int i = 0; i < 4; i++) free_hip(&r->d_dec[i]);

    if (r->mod) hipModuleUnload(r->mod);
    free(r);
}

void hip_trellis2_invalidate_kv_cache(hip_trellis2_runner *r) {
    r->ca_kv_valid = 0;
}

/* ======================================================================== */
/* Weight loading                                                          */
/* ======================================================================== */

int hip_trellis2_load_dit(hip_trellis2_runner *r, const char *path) {
    fprintf(stderr, "T2-HIP: loading DiT weights: %s\n", path);

    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "T2-HIP: failed to open %s\n", path); return -1; }

    int v = r->verbose;

    /* Global weights */
    r->dit_input.w  = st_upload_f32(st, "input_layer.weight", v);
    r->dit_input.b  = st_upload_f32(st, "input_layer.bias",   v);
    r->dit_t_fc1.w  = st_upload_f32(st, "t_embedder.mlp.0.weight", v);
    r->dit_t_fc1.b  = st_upload_f32(st, "t_embedder.mlp.0.bias",   v);
    r->dit_t_fc2.w  = st_upload_f32(st, "t_embedder.mlp.2.weight", v);
    r->dit_t_fc2.b  = st_upload_f32(st, "t_embedder.mlp.2.bias",   v);
    r->dit_ada_mod.w = st_upload_f32(st, "adaLN_modulation.1.weight", v);
    r->dit_ada_mod.b = st_upload_f32(st, "adaLN_modulation.1.bias",   v);
    r->dit_out.w    = st_upload_f32(st, "out_layer.weight", v);
    r->dit_out.b    = st_upload_f32(st, "out_layer.bias",   v);

    /* Per-block weights */
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        char name[128];
        t2_block_gpu *b = &r->blocks[bi];
        #define BN(n) (snprintf(name,sizeof(name),"blocks.%d.%s",bi,n), name)

        /* Correct tensor names from common/trellis2_dit.h */
        b->sa_qkv.w  = st_upload_f32(st, BN("self_attn.to_qkv.weight"), v);
        b->sa_qkv.b  = st_upload_f32(st, BN("self_attn.to_qkv.bias"),   v);
        b->sa_q_norm = st_upload_f32(st, BN("self_attn.q_rms_norm.gamma"), v);
        b->sa_k_norm = st_upload_f32(st, BN("self_attn.k_rms_norm.gamma"), v);
        b->sa_out.w  = st_upload_f32(st, BN("self_attn.to_out.weight"), v);
        b->sa_out.b  = st_upload_f32(st, BN("self_attn.to_out.bias"),   v);

        b->norm2_w   = st_upload_f32(st, BN("norm2.weight"), v);
        b->norm2_b   = st_upload_f32(st, BN("norm2.bias"),   v);
        b->ca_q.w    = st_upload_f32(st, BN("cross_attn.to_q.weight"), v);
        b->ca_q.b    = st_upload_f32(st, BN("cross_attn.to_q.bias"),   v);
        b->ca_q_norm = st_upload_f32(st, BN("cross_attn.q_rms_norm.gamma"), v);
        b->ca_k_norm = st_upload_f32(st, BN("cross_attn.k_rms_norm.gamma"), v);
        b->ca_kv.w   = st_upload_f32(st, BN("cross_attn.to_kv.weight"), v);
        b->ca_kv.b   = st_upload_f32(st, BN("cross_attn.to_kv.bias"),   v);
        b->ca_out.w  = st_upload_f32(st, BN("cross_attn.to_out.weight"), v);
        b->ca_out.b  = st_upload_f32(st, BN("cross_attn.to_out.bias"),   v);

        b->mlp_fc1.w = st_upload_f32(st, BN("mlp.mlp.0.weight"), v);
        b->mlp_fc1.b = st_upload_f32(st, BN("mlp.mlp.0.bias"),   v);
        b->mlp_fc2.w = st_upload_f32(st, BN("mlp.mlp.2.weight"), v);
        b->mlp_fc2.b = st_upload_f32(st, BN("mlp.mlp.2.bias"),   v);

        b->mod_bias  = st_upload_f32(st, BN("modulation"),            v);

        /* Allocate CA KV cache for this block */
        if (!r->ca_K_cache[bi])
            hipMalloc(&r->ca_K_cache[bi], 1029 * DIT_DIM * sizeof(float));
        if (!r->ca_V_cache[bi])
            hipMalloc(&r->ca_V_cache[bi], 1029 * DIT_DIM * sizeof(float));
        #undef BN
    }

    safetensors_close(st);

    /* Precompute 3D RoPE phases */
    r->n_rope_freqs = 21;  /* head_dim/2 / 3 ≈ 21 */
    float *cpu_cos, *cpu_sin;
    int n_tok;
    compute_rope_phases(DIT_GRID, r->n_rope_freqs, &cpu_cos, &cpu_sin, &n_tok);
    r->d_rope_cos = hip_upload_raw(cpu_cos, (size_t)n_tok * 3 * r->n_rope_freqs * sizeof(float));
    r->d_rope_sin = hip_upload_raw(cpu_sin, (size_t)n_tok * 3 * r->n_rope_freqs * sizeof(float));
    free(cpu_cos); free(cpu_sin);

    r->dit_loaded = 1;
    r->ca_kv_valid = 0;
    fprintf(stderr, "T2-HIP: DiT loaded (%d blocks, RoPE %d freqs)\n",
            DIT_DEPTH, r->n_rope_freqs);
    return 0;
}

/* Load decoder ResBlock weights */
static void load_dec_resblock(st_context *st, t2_dec_layer *rb,
                               const char *prefix, int v) {
    char name[256];
#define DN(suffix) (snprintf(name,sizeof(name),"%s.%s",prefix,suffix), name)
    rb->rb_ln1_w = st_upload_f32(st, DN("norm1.weight"), v);
    rb->rb_ln1_b = st_upload_f32(st, DN("norm1.bias"),   v);
    rb->rb_c1_w  = st_upload_f32(st, DN("conv1.weight"), v);
    rb->rb_c1_b  = st_upload_f32(st, DN("conv1.bias"),   v);
    rb->rb_ln2_w = st_upload_f32(st, DN("norm2.weight"), v);
    rb->rb_ln2_b = st_upload_f32(st, DN("norm2.bias"),   v);
    rb->rb_c2_w  = st_upload_f32(st, DN("conv2.weight"), v);
    rb->rb_c2_b  = st_upload_f32(st, DN("conv2.bias"),   v);
#undef DN
}

int hip_trellis2_load_decoder(hip_trellis2_runner *r, const char *path) {
    fprintf(stderr, "T2-HIP: loading decoder weights: %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "T2-HIP: failed to open %s\n", path); return -1; }

    int v = r->verbose;

    r->dec_input.conv_w = st_upload_f32(st, "input_layer.weight", v);
    r->dec_input.conv_b = st_upload_f32(st, "input_layer.bias",   v);
    load_dec_resblock(st, &r->dec_mid[0],   "middle_block.0", v);
    load_dec_resblock(st, &r->dec_mid[1],   "middle_block.1", v);
    load_dec_resblock(st, &r->dec_res16[0], "blocks.0", v);
    load_dec_resblock(st, &r->dec_res16[1], "blocks.1", v);
    r->dec_up1_w = st_upload_f32(st, "blocks.2.conv.weight", v);
    r->dec_up1_b = st_upload_f32(st, "blocks.2.conv.bias",   v);
    load_dec_resblock(st, &r->dec_res32[0], "blocks.3", v);
    load_dec_resblock(st, &r->dec_res32[1], "blocks.4", v);
    r->dec_up2_w = st_upload_f32(st, "blocks.5.conv.weight", v);
    r->dec_up2_b = st_upload_f32(st, "blocks.5.conv.bias",   v);
    load_dec_resblock(st, &r->dec_res64[0], "blocks.6", v);
    load_dec_resblock(st, &r->dec_res64[1], "blocks.7", v);
    r->dec_out_ln_w = st_upload_f32(st, "out_layer.0.weight", v);
    r->dec_out_ln_b = st_upload_f32(st, "out_layer.0.bias",   v);
    r->dec_out_w    = st_upload_f32(st, "out_layer.2.weight", v);
    r->dec_out_b    = st_upload_f32(st, "out_layer.2.bias",   v);

    safetensors_close(st);

    /* Allocate decoder scratch: 512@16³, 128@32³, 32@64³, and temp for pixel_shuffle */
    /* All 4 decoder buffers sized for the largest intermediate: 32ch × 64³ = 32 MB.
     * 1024ch × 16³ = 16 MB also fits. Unified size avoids aliasing bugs. */
    size_t dec_max = (size_t)32 * 64 * 64 * 64 * sizeof(float);
    for (int i = 0; i < 4; i++) hipMalloc(&r->d_dec[i], dec_max);
    r->d_dec_sz[0] = 512 * 16*16*16 * sizeof(float);
    r->d_dec_sz[1] = 512 * 16*16*16 * sizeof(float);
    r->d_dec_sz[2] = 128 * 32*32*32 * sizeof(float);
    r->d_dec_sz[3] =  32 * 64*64*64 * sizeof(float);

    r->dec_loaded = 1;
    fprintf(stderr, "T2-HIP: decoder loaded\n");
    return 0;
}

/* ======================================================================== */
/* Kernel launch wrappers — pointer-array style (kernelParams, not extra)  */
/* ======================================================================== */

/* layernorm_f32: dst[N,dim] = LN(src) * w + b */
static void run_layernorm(hip_trellis2_runner *r, void *dst, const void *src,
                           const void *w, const void *b, int N, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &w, &b, &dim, &eps};
    hipModuleLaunchKernel(r->fn_layernorm, N, 1, 1, 256, 1, 1,
                          256 * 2 * sizeof(float), 0, args, NULL);
}

/* adaln_f32: dst[N,dim] = LN_noaffine(src) * (1+scale) + shift */
static void run_adaln(hip_trellis2_runner *r, void *dst, const void *src,
                       const void *shift, const void *scale, int N, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &shift, &scale, &dim, &eps};
    hipModuleLaunchKernel(r->fn_adaln, N, 1, 1, 256, 1, 1,
                          256 * 2 * sizeof(float), 0, args, NULL);
}

/* gated_add: dst[N*dim] += gate[dim] * src[N*dim] */
static void run_gated_add(hip_trellis2_runner *r, void *dst, const void *src,
                            const void *gate, int N, int dim) {
    int n = N * dim;
    void *args[] = {&dst, &src, &gate, &n, &dim};
    hipModuleLaunchKernel(r->fn_gated_add, (n+255)/256, 1, 1, 256, 1, 1,
                          0, 0, args, NULL);
}

/* residual_add: dst[n] += src[n] */
static void run_residual_add(hip_trellis2_runner *r, void *dst, const void *src, int n) {
    void *args[] = {&dst, &src, &n};
    hipModuleLaunchKernel(r->fn_residual_add, (n+255)/256, 1, 1, 256, 1, 1,
                          0, 0, args, NULL);
}

/* silu_inplace_f32 */
static void run_silu(hip_trellis2_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_silu, (n+255)/256, 1, 1, 256, 1, 1,
                          0, 0, args, NULL);
}

/* gelu_inplace_f32 */
static void run_gelu(hip_trellis2_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_gelu, (n+255)/256, 1, 1, 256, 1, 1,
                          0, 0, args, NULL);
}

/* rms_norm_perhead_f32 */
static void run_rms_norm_ph(hip_trellis2_runner *r, void *data, const void *gamma,
                              int n_tok, int n_heads, int head_dim, int stride) {
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    void *args[] = {&data, &gamma, &n_tok, &n_heads, &head_dim, &stride, &eps};
    hipModuleLaunchKernel(r->fn_rms_norm_ph, (total+255)/256, 1, 1, 256, 1, 1,
                          0, 0, args, NULL);
}

/* rope_3d_f32: apply 3D RoPE to Q or K [N_tok, n_heads, head_dim] */
static void run_rope_3d(hip_trellis2_runner *r, void *data, int N) {
    int dim      = DIT_DIM;
    int n_heads  = DIT_HEADS;
    int head_dim = DIT_HEAD_DIM;
    int n_freqs  = r->n_rope_freqs;
    int axis_dim = DIT_GRID;
    void *args[] = {&data, &r->d_rope_cos, &r->d_rope_sin,
                    &N, &dim, &n_heads, &head_dim, &n_freqs, &axis_dim};
    hipModuleLaunchKernel(r->fn_rope_3d, N, 1, 1, 256, 1, 1, 0, 0, args, NULL);
}

/* flash_attn_sa_f32: warp-cooperative FA2 self-attention */
static void run_flash_sa(hip_trellis2_runner *r, void *out,
                          const void *Q, const void *K, const void *V, int N) {
    int n_heads  = DIT_HEADS;
    int head_dim = DIT_HEAD_DIM;
    int gY = (N + 3) / 4;  /* FA_WARPS=4 */
    size_t smem = 2 * 16 * 128 * sizeof(float);  /* FA_BKV=16, FA_HD=128 */
    void *args[] = {&out, &Q, &K, &V, &N, &n_heads, &head_dim};
    hipModuleLaunchKernel(r->fn_flash_sa, DIT_HEADS, gY, 1, 128, 1, 1,
                          smem, 0, args, NULL);
}

/* cross_attn_tiled_f32 */
static void run_cross_attn(hip_trellis2_runner *r, void *out,
                             const void *Q, const void *K, const void *V,
                             int q_len, int kv_len) {
    float scale  = 1.0f / sqrtf((float)DIT_HEAD_DIM);
    int dim      = DIT_DIM;
    int n_heads  = DIT_HEADS;
    int head_dim = DIT_HEAD_DIM;
    void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim, &n_heads, &head_dim, &scale};
    hipModuleLaunchKernel(r->fn_cross_attn, DIT_HEADS, q_len, 1, 128, 1, 1,
                          0, 0, args, NULL);
}

/* broadcast_bias_f32: out[tok*C + c] += bias[c] */
static void run_broadcast_bias(hip_trellis2_runner *r, void *out,
                                 const void *bias, int N, int C) {
    if (!bias) return;
    void *args[] = {&out, &bias, &N, &C};
    hipModuleLaunchKernel(r->fn_broadcast_bias, (N*C+255)/256, 1, 1, 256, 1, 1,
                          0, 0, args, NULL);
}

/* conv3d_k3_f32: 3D conv k=3, pad=1, NCDHW layout */
static void run_conv3d(hip_trellis2_runner *r, void *out, const void *inp,
                        const void *w, const void *b,
                        int Ci, int Co, int D, int H, int W) {
    int spatial = D * H * W;
    void *args[] = {&out, &inp, &w, &b, &Ci, &Co, &D, &H, &W};
    hipModuleLaunchKernel(r->fn_conv3d, Co, (spatial+63)/64, 1, 64, 1, 1,
                          0, 0, args, NULL);
}

/* channel_layernorm_3d_f32: LayerNorm over C per spatial position */
static void run_ch_ln3d(hip_trellis2_runner *r, void *dst, const void *src,
                          const void *w, const void *b, int C, int spatial) {
    int G = 0; /* unused */
    void *args[] = {&dst, &src, &w, &b, &C, &spatial, &G};
    hipModuleLaunchKernel(r->fn_ch_ln3d, spatial, 1, 1, 256, 1, 1,
                          256 * 2 * sizeof(float), 0, args, NULL);
}

/* pixel_shuffle_3d_f32: [C*8, D, H, W] → [C, 2D, 2H, 2W] */
static void run_pix_shuf3d(hip_trellis2_runner *r, void *dst, const void *src,
                              int C, int D, int H, int W) {
    int total = C * (2*D) * (2*H) * (2*W);
    void *args[] = {&dst, &src, &C, &D, &H, &W};
    hipModuleLaunchKernel(r->fn_pix_shuf3d, (total+255)/256, 1, 1, 256, 1, 1,
                          0, 0, args, NULL);
}

/* layernorm_noaffine_f32: LN without learned affine parameters */
static void run_ln_noaffine(hip_trellis2_runner *r, void *dst, const void *src,
                              int N, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &src, &dim, &eps};
    hipModuleLaunchKernel(r->fn_ln_noaffine, N, 1, 1, 256, 1, 1,
                          256 * 2 * sizeof(float), 0, args, NULL);
}

/* ======================================================================== */
/* Decoder ResBlock forward                                               */
/* ======================================================================== */

/* ResBlock3d: ChannelLN → SiLU → Conv3d → ChannelLN → SiLU → Conv3d + skip
 * Needs 3 distinct buffers: buf (skip), tmp1, tmp2. Caller must ensure no alias. */
static void run_dec_resblock(hip_trellis2_runner *r,
                               void *buf,   /* [C, D, H, W] in/out (skip preserved) */
                               void *tmp1,  /* [C, D, H, W] scratch */
                               void *tmp2,  /* [C, D, H, W] scratch */
                               t2_dec_layer *rb,
                               int C, int D, int H, int W) {
    int spatial = D * H * W;
    /* norm1 + SiLU -> tmp1 */
    run_ch_ln3d(r, tmp1, buf, rb->rb_ln1_w, rb->rb_ln1_b, C, spatial);
    run_silu(r, tmp1, C * spatial);
    /* conv1 tmp1 -> tmp2 */
    run_conv3d(r, tmp2, tmp1, rb->rb_c1_w, rb->rb_c1_b, C, C, D, H, W);
    /* norm2 + SiLU tmp2 -> tmp1 */
    run_ch_ln3d(r, tmp1, tmp2, rb->rb_ln2_w, rb->rb_ln2_b, C, spatial);
    run_silu(r, tmp1, C * spatial);
    /* conv2 tmp1 -> tmp2 */
    run_conv3d(r, tmp2, tmp1, rb->rb_c2_w, rb->rb_c2_b, C, C, D, H, W);
    /* skip connection: buf += tmp2 */
    run_residual_add(r, buf, tmp2, C * spatial);
}

/* ======================================================================== */
/* Decoder forward                                                         */
/* ======================================================================== */

int hip_trellis2_decode(hip_trellis2_runner *r, const float *latent, float *occupancy) {
    if (!r->dec_loaded) { fprintf(stderr, "T2-HIP: decoder not loaded\n"); return -1; }

    /* Caller passes channel-first (C,D,H,W) matching PyTorch Conv3d layout. */
    int D = 16, H = 16, W = 16;
    hipMemcpy(r->d_dec[0], latent, DIT_IN_CH * D*H*W * sizeof(float), hipMemcpyHostToDevice);

    /* input_layer: Conv3D [8 -> 512] */
    void *cur = r->d_dec[0];   /* [8, 16, 16, 16] */
    void *buf512 = r->d_dec[1]; /* [512, 16, 16, 16] */
    run_conv3d(r, buf512, cur, r->dec_input.conv_w, r->dec_input.conv_b, 8, 512, D, H, W);
    cur = buf512;

    /* All resblocks at every scale need 3 distinct buffers: buf, tmp1, tmp2.
     * After input_layer: cur=d_dec[1]. Use d_dec[2]/d_dec[3] as tmp1/tmp2. */

    /* middle_block: 2 × ResBlock3d(512) at 16³ */
    run_dec_resblock(r, cur, r->d_dec[2], r->d_dec[3], &r->dec_mid[0], 512, D, H, W);
    run_dec_resblock(r, cur, r->d_dec[2], r->d_dec[3], &r->dec_mid[1], 512, D, H, W);

    /* blocks.0-1: 2 × ResBlock3d(512) at 16³ */
    run_dec_resblock(r, cur, r->d_dec[2], r->d_dec[3], &r->dec_res16[0], 512, D, H, W);
    run_dec_resblock(r, cur, r->d_dec[2], r->d_dec[3], &r->dec_res16[1], 512, D, H, W);

    /* blocks.2: Conv3D(512→1024) + pixel_shuffle_3d(factor=2) -> [128, 32, 32, 32] */
    void *buf1024 = r->d_dec[0];  /* 32MB, large enough for 1024ch×16³=16 MB */
    run_conv3d(r, buf1024, cur, r->dec_up1_w, r->dec_up1_b, 512, 1024, D, H, W);
    void *buf128 = r->d_dec[1];
    run_pix_shuf3d(r, buf128, buf1024, 128, D, H, W);
    D = 32; H = 32; W = 32;
    cur = buf128;

    /* blocks.3-4: 2 × ResBlock3d(128) at 32³ (16 MB each) */
    run_dec_resblock(r, cur, r->d_dec[2], r->d_dec[3], &r->dec_res32[0], 128, D, H, W);
    run_dec_resblock(r, cur, r->d_dec[2], r->d_dec[3], &r->dec_res32[1], 128, D, H, W);

    /* blocks.5: Conv3D(128→256) + pixel_shuffle_3d -> [32, 64, 64, 64] */
    void *buf256 = r->d_dec[0];  /* 256ch×32³=32 MB, fits */
    run_conv3d(r, buf256, cur, r->dec_up2_w, r->dec_up2_b, 128, 256, D, H, W);
    void *buf32 = r->d_dec[1];
    run_pix_shuf3d(r, buf32, buf256, 32, D, H, W);
    D = 64; H = 64; W = 64;
    cur = buf32;

    /* blocks.6-7: 2 × ResBlock3d(32) at 64³ (32 MB each) */
    run_dec_resblock(r, cur, r->d_dec[2], r->d_dec[3], &r->dec_res64[0], 32, D, H, W);
    run_dec_resblock(r, cur, r->d_dec[2], r->d_dec[3], &r->dec_res64[1], 32, D, H, W);

    /* out_layer: ChannelLN(32) → SiLU → Conv3D(32→1) */
    void *buf_ln = r->d_dec[2];
    run_ch_ln3d(r, buf_ln, cur, r->dec_out_ln_w, r->dec_out_ln_b, 32, D*H*W);
    run_silu(r, buf_ln, 32 * D*H*W);
    void *buf_out = r->d_dec[3];
    run_conv3d(r, buf_out, buf_ln, r->dec_out_w, r->dec_out_b, 32, 1, D, H, W);

    hipDeviceSynchronize();

    /* Download [1, 64, 64, 64] */
    hipMemcpy(occupancy, buf_out, D*H*W * sizeof(float), hipMemcpyDeviceToHost);
    return 0;
}

/* ======================================================================== */
/* DiT forward                                                            */
/* ======================================================================== */

static int dit_forward(hip_trellis2_runner *r, int n_tok, int cond_len, int dump_block) {
    if (!r->dit_loaded) { fprintf(stderr, "T2-HIP: DiT not loaded\n"); return -1; }

    /* ---- Input embedding: [n_tok, IN_CH] × [DIT_DIM, IN_CH]^T -> [n_tok, DIT_DIM] ---- */
    gemm(r, r->d_h, r->dit_input.w, r->d_noise,
         r->dit_input.b, DIT_DIM, DIT_IN_CH, n_tok);
    if (r->b0_dbg) dbg_dl(r->b0_dbg->input_embed, r->d_h, (size_t)n_tok * DIT_DIM);

    /* ---- Timestep embedding (already uploaded to d_t_emb) ---- */
    /* FC1: [DIT_DIM, 256] -> [DIT_DIM], SiLU */
    gemm(r, r->d_t_silu, r->dit_t_fc1.w, r->d_t_emb,
         r->dit_t_fc1.b, DIT_DIM, DIT_T_HALF*2, 1);
    run_silu(r, r->d_t_silu, DIT_DIM);
    /* FC2: [DIT_DIM, DIT_DIM] -> [DIT_DIM]
     * Store FC2 output in d_ada_out (unused otherwise). Must not alias with
     * d_mod since per-block modulation writes to d_mod while reading d_t_out. */
    void *d_t_out = r->d_ada_out;
    gemm(r, d_t_out, r->dit_t_fc2.w, r->d_t_silu,
         r->dit_t_fc2.b, DIT_DIM, DIT_DIM, 1);

    /* ---- Precompute cross-attention KV cache if not valid ---- */
    if (!r->ca_kv_valid) {
        if (r->verbose) fprintf(stderr, "T2-HIP: building CA KV cache (%d tokens)...\n", cond_len);
        for (int bi = 0; bi < DIT_DEPTH; bi++) {
            t2_block_gpu *b = &r->blocks[bi];
            /* ca_kv_w: [2*DIT_DIM, DIT_COND_DIM] × cond [cond_len, DIT_COND_DIM]^T
             * -> [cond_len, 2*DIT_DIM] */
            gemm(r, r->d_V, b->ca_kv.w, r->d_cond,
                 NULL, 2*DIT_DIM, DIT_COND_DIM, cond_len);
            run_broadcast_bias(r, r->d_V, b->ca_kv.b, cond_len, 2*DIT_DIM);
            /* Split into K, V each [cond_len, DIT_DIM] */
            {
                void *K = r->ca_K_cache[bi], *V = r->ca_V_cache[bi];
                const void *kv = r->d_V;
                int M = cond_len, W = DIT_DIM;
                void *args[] = { &K, &V, &kv, &M, &W };
                hipModuleLaunchKernel(r->fn_split_kv,
                    (cond_len * DIT_DIM + 255) / 256, 1, 1, 256, 1, 1,
                    0, 0, args, NULL);
            }
            /* Apply per-head RMSNorm to K */
            run_rms_norm_ph(r, r->ca_K_cache[bi], b->ca_k_norm,
                             cond_len, DIT_HEADS, DIT_HEAD_DIM, DIT_DIM);
        }
        r->ca_kv_valid = 1;
        r->ca_kv_len = cond_len;
        if (r->verbose) fprintf(stderr, "T2-HIP: CA KV cache built\n");
    }

    /* ---- Per-block forward ---- */
    /* Pointer arithmetic helpers into d_ada_out/d_mod:
     * layout: [shift_sa|scale_sa|gate_sa|shift_mlp|scale_mlp|gate_mlp] each DIT_DIM */
    (void)r->d_ada_out; /* precomputed shared mod; per-block uses d_mod with blk_bias */

    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        t2_block_gpu *b = &r->blocks[bi];

        /* Per-block modulation: d_mod = d_ada_out + block.mod_bias */
        {
            void *out = r->d_mod;
            const void *t_emb = d_t_out, *mod_w = r->dit_ada_mod.w, *mod_b = r->dit_ada_mod.b;
            const void *blk_bias = b->mod_bias;
            int dim = DIT_DIM, out_dim = 6*DIT_DIM;
            void *args[] = { &out, &t_emb, &mod_w, &mod_b, &blk_bias, &dim, &out_dim };
            hipModuleLaunchKernel(r->fn_modulation, 1, 1, 1, 256, 1, 1,
                                  DIT_DIM * sizeof(float), 0, args, NULL);
        }
        float *mod = (float *)r->d_mod;
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->mod, r->d_mod, 6 * DIT_DIM);
        /* Pointers into mod [6*DIT_DIM]: shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp */
        void *shift_sa  = mod + 0*DIT_DIM;
        void *scale_sa  = mod + 1*DIT_DIM;
        void *gate_sa   = mod + 2*DIT_DIM;
        void *shift_mlp = mod + 3*DIT_DIM;
        void *scale_mlp = mod + 4*DIT_DIM;
        void *gate_mlp  = mod + 5*DIT_DIM;

        /* ---- Self-attention ---- */
        /* adaLN -> d_ln_h */
        run_adaln(r, r->d_ln_h, r->d_h, shift_sa, scale_sa, n_tok, DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->ln_h_sa, r->d_ln_h, (size_t)n_tok * DIT_DIM);
        /* QKV projection: [n_tok, DIT_DIM] × [3*DIT_DIM, DIT_DIM]^T -> [n_tok, 3*DIT_DIM] */
        gemm(r, r->d_qkv, b->sa_qkv.w, r->d_ln_h, NULL, 3*DIT_DIM, DIT_DIM, n_tok);
        run_broadcast_bias(r, r->d_qkv, b->sa_qkv.b, n_tok, 3*DIT_DIM);
        /* Split QKV */
        {
            void *Q = r->d_Q, *K = r->d_K, *V = r->d_V;
            const void *qkv = r->d_qkv;
            int N = n_tok, W = DIT_DIM;
            void *args[] = { &Q, &K, &V, &qkv, &N, &W };
            hipModuleLaunchKernel(r->fn_split_qkv,
                (n_tok * DIT_DIM + 255)/256, 1, 1, 256, 1, 1, 0, 0, args, NULL);
        }
        /* QK RMSNorm */
        run_rms_norm_ph(r, r->d_Q, b->sa_q_norm, n_tok, DIT_HEADS, DIT_HEAD_DIM, DIT_DIM);
        run_rms_norm_ph(r, r->d_K, b->sa_k_norm, n_tok, DIT_HEADS, DIT_HEAD_DIM, DIT_DIM);
        /* 3D RoPE on Q and K */
        run_rope_3d(r, r->d_Q, n_tok);
        run_rope_3d(r, r->d_K, n_tok);
        if (bi == 0 && r->b0_dbg) {
            dbg_dl(r->b0_dbg->q_post, r->d_Q, (size_t)n_tok * DIT_DIM);
            dbg_dl(r->b0_dbg->k_post, r->d_K, (size_t)n_tok * DIT_DIM);
            dbg_dl(r->b0_dbg->v,      r->d_V, (size_t)n_tok * DIT_DIM);
        }
        /* Flash attention -> attn_out */
        run_flash_sa(r, r->d_attn_out, r->d_Q, r->d_K, r->d_V, n_tok);
        /* Output projection: [n_tok, DIT_DIM] × [DIT_DIM, DIT_DIM]^T */
        gemm(r, r->d_ln_h, b->sa_out.w, r->d_attn_out, NULL, DIT_DIM, DIT_DIM, n_tok);
        run_broadcast_bias(r, r->d_ln_h, b->sa_out.b, n_tok, DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->sa_proj, r->d_ln_h, (size_t)n_tok * DIT_DIM);
        /* Gated residual: d_h += gate_sa * d_ln_h */
        run_gated_add(r, r->d_h, r->d_ln_h, gate_sa, n_tok, DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->h_post_sa, r->d_h, (size_t)n_tok * DIT_DIM);

        /* ---- Cross-attention ---- */
        /* norm2 -> d_ln_h */
        run_layernorm(r, r->d_ln_h, r->d_h, b->norm2_w, b->norm2_b, n_tok, DIT_DIM);
        /* Q projection: [n_tok, DIT_DIM] × [DIT_DIM, DIT_DIM]^T -> d_ca_Q */
        gemm(r, r->d_ca_Q, b->ca_q.w, r->d_ln_h, NULL, DIT_DIM, DIT_DIM, n_tok);
        run_broadcast_bias(r, r->d_ca_Q, b->ca_q.b, n_tok, DIT_DIM);
        /* Q RMSNorm */
        run_rms_norm_ph(r, r->d_ca_Q, b->ca_q_norm, n_tok, DIT_HEADS, DIT_HEAD_DIM, DIT_DIM);
        /* Cross-attention using cached K,V */
        run_cross_attn(r, r->d_attn_out, r->d_ca_Q,
                        r->ca_K_cache[bi], r->ca_V_cache[bi],
                        n_tok, r->ca_kv_len);
        /* Output projection */
        gemm(r, r->d_ln_h, b->ca_out.w, r->d_attn_out, NULL, DIT_DIM, DIT_DIM, n_tok);
        run_broadcast_bias(r, r->d_ln_h, b->ca_out.b, n_tok, DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->ca_proj, r->d_ln_h, (size_t)n_tok * DIT_DIM);
        /* Direct residual (no gate in cross-attn) */
        run_residual_add(r, r->d_h, r->d_ln_h, n_tok * DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->h_post_ca, r->d_h, (size_t)n_tok * DIT_DIM);

        /* ---- MLP ---- */
        /* adaLN (scale2/shift2) -> d_ln_h */
        run_adaln(r, r->d_ln_h, r->d_h, shift_mlp, scale_mlp, n_tok, DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->ln_h_mlp, r->d_ln_h, (size_t)n_tok * DIT_DIM);
        /* FC1: [n_tok, DIT_DIM] × [DIT_FFN, DIT_DIM]^T -> [n_tok, DIT_FFN] */
        gemm(r, r->d_mlp_mid, b->mlp_fc1.w, r->d_ln_h, NULL, DIT_FFN, DIT_DIM, n_tok);
        run_broadcast_bias(r, r->d_mlp_mid, b->mlp_fc1.b, n_tok, DIT_FFN);
        run_gelu(r, r->d_mlp_mid, n_tok * DIT_FFN);
        /* FC2: [n_tok, DIT_FFN] × [DIT_DIM, DIT_FFN]^T -> [n_tok, DIT_DIM] */
        gemm(r, r->d_ln_h, b->mlp_fc2.w, r->d_mlp_mid, NULL, DIT_DIM, DIT_FFN, n_tok);
        run_broadcast_bias(r, r->d_ln_h, b->mlp_fc2.b, n_tok, DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->mlp_proj, r->d_ln_h, (size_t)n_tok * DIT_DIM);
        /* Gated residual: d_h += gate_mlp * d_ln_h */
        run_gated_add(r, r->d_h, r->d_ln_h, gate_mlp, n_tok, DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->h_post_mlp, r->d_h, (size_t)n_tok * DIT_DIM);

        /* Optional block dump */
        if (dump_block == bi) {
            hipDeviceSynchronize();
            return bi;  /* signal caller to download d_h */
        }
    }

    /* ---- Final LayerNorm (no affine) ---- */
    run_ln_noaffine(r, r->d_ln_h, r->d_h, n_tok, DIT_DIM);

    /* ---- Output projection: [n_tok, DIT_DIM] × [DIT_IN_CH, DIT_DIM]^T ---- */
    gemm(r, r->d_vel, r->dit_out.w, r->d_ln_h, NULL, DIT_IN_CH, DIT_DIM, n_tok);
    run_broadcast_bias(r, r->d_vel, r->dit_out.b, n_tok, DIT_IN_CH);

    hipDeviceSynchronize();
    return DIT_DEPTH;  /* completed all blocks */
}

/* ======================================================================== */
/* Public API                                                              */
/* ======================================================================== */

int hip_trellis2_dit_step(hip_trellis2_runner *r,
                           const float *noise_flat,
                           const float *features,
                           float t,
                           float *out_vel) {
    if (!r->dit_loaded) { fprintf(stderr, "T2-HIP: DiT not loaded\n"); return -1; }

    int n_tok = DIT_N_TOK;
    int cond_len = 1029;

    /* Upload noise [n_tok, DIT_IN_CH] */
    hipMemcpy(r->d_noise, noise_flat, (size_t)n_tok * DIT_IN_CH * sizeof(float),
              hipMemcpyHostToDevice);

    /* Upload features [cond_len, DIT_COND_DIM] */
    hipMemcpy(r->d_cond, features, (size_t)cond_len * DIT_COND_DIM * sizeof(float),
              hipMemcpyHostToDevice);

    /* Timestep sinusoidal embed: [DIT_T_HALF*2] */
    {
        void *out = r->d_t_emb;
        float t_val = t * 1000.0f;  /* TRELLIS.2 scales t by 1000 */
        int dim = DIT_T_HALF * 2;
        void *args[] = { &out, &t_val, &dim };
        hipModuleLaunchKernel(r->fn_t_embed, (DIT_T_HALF+255)/256, 1, 1, 256, 1, 1,
                              0, 0, args, NULL);
    }

    int res = dit_forward(r, n_tok, cond_len, -1 /* no dump */);
    if (res < 0) return -1;

    /* Download velocity [n_tok, DIT_IN_CH] */
    hipMemcpy(out_vel, r->d_vel, (size_t)n_tok * DIT_IN_CH * sizeof(float),
              hipMemcpyDeviceToHost);
    return 0;
}

void hip_trellis2_invalidate_kv(hip_trellis2_runner *r) {
    if (r) r->ca_kv_valid = 0;
}

int hip_trellis2_dump_block(hip_trellis2_runner *r,
                              const float *noise_flat,
                              const float *features,
                              float t,
                              int block_idx,
                              float *out_hidden) {
    if (!r->dit_loaded) return -1;
    if (block_idx < 0 || block_idx >= DIT_DEPTH) return -1;

    int n_tok = DIT_N_TOK;
    int cond_len = 1029;

    hipMemcpy(r->d_noise, noise_flat, (size_t)n_tok * DIT_IN_CH * sizeof(float),
              hipMemcpyHostToDevice);
    hipMemcpy(r->d_cond, features, (size_t)cond_len * DIT_COND_DIM * sizeof(float),
              hipMemcpyHostToDevice);

    {
        void *out = r->d_t_emb;
        float t_val = t * 1000.0f;  /* TRELLIS.2 scales t by 1000 */
        int dim = DIT_T_HALF * 2;
        void *args[] = { &out, &t_val, &dim };
        hipModuleLaunchKernel(r->fn_t_embed, (DIT_T_HALF+255)/256, 1, 1, 256, 1, 1,
                              0, 0, args, NULL);
    }

    int res = dit_forward(r, n_tok, cond_len, block_idx);
    if (res < 0) return -1;

    /* Download hidden state [n_tok, DIT_DIM] */
    hipMemcpy(out_hidden, r->d_h, (size_t)n_tok * DIT_DIM * sizeof(float),
              hipMemcpyDeviceToHost);
    return 0;
}

int hip_trellis2_dump_b0_detail(hip_trellis2_runner *r,
                                 const float *noise_flat,
                                 const float *features,
                                 float t,
                                 hip_trellis2_b0_dbg *dbg) {
    if (!r || !r->dit_loaded || !dbg) return -1;
    int n_tok = DIT_N_TOK;
    int cond_len = 1029;

    hipMemcpy(r->d_noise, noise_flat, (size_t)n_tok * DIT_IN_CH * sizeof(float),
              hipMemcpyHostToDevice);
    hipMemcpy(r->d_cond, features, (size_t)cond_len * DIT_COND_DIM * sizeof(float),
              hipMemcpyHostToDevice);
    {
        void *out = r->d_t_emb;
        float t_val = t * 1000.0f;
        int dim = DIT_T_HALF * 2;
        void *args[] = { &out, &t_val, &dim };
        hipModuleLaunchKernel(r->fn_t_embed, (DIT_T_HALF+255)/256, 1, 1, 256, 1, 1,
                              0, 0, args, NULL);
    }
    r->b0_dbg = dbg;
    int res = dit_forward(r, n_tok, cond_len, 0 /* stop at block 0 */);
    r->b0_dbg = NULL;
    return (res < 0) ? -1 : 0;
}
