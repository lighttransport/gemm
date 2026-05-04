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
#include "../llm/mm_blaslt_bridge.h"
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

/* Weight tensor + optional bias.
 *   scale < 0  -> w is F32 (legacy path, also drives BF16 WMMA via in-flight truncation).
 *   scale >= 0 -> w is raw FP8 E4M3 bytes (1 byte/elem); reconstruct via lut[byte]*scale.
 */
typedef struct {
    void *w;          /* F32 weight (NULL if blaslt-only or fp8) */
    void *w_bf16;     /* BF16 weight (raw bytes, [n_out, n_in]); used by hipBLASLt path */
    void *b;          /* F32 bias, device pointer (may be NULL) */
    float scale;      /* -1.0f = F32/BF16 weight; >=0.0f = FP8 per-tensor scale */
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
    hipFunction_t fn_gemm;           /* gemm_tiled_f32_f32 (scalar F32 fallback) */
    hipFunction_t fn_gemm_wmma;      /* gemm_bf16w_bf16a_wmma_t (gfx12 only) */
    hipFunction_t fn_gemm_fp8_wmma;  /* gemm_fp8w_bf16a_wmma_t (gfx12 only) */
    hipFunction_t fn_pack_bf16;      /* pack_bf16_from_f32 — F32 -> BF16 for hipBLASLt */
    int use_wmma;                    /* 1 iff BF16 WMMA kernel is loaded & enabled */
    int use_fp8;                     /* 1 iff FP8 LUT uploaded and any FP8 weight loaded */
    int use_blaslt;                  /* 1 iff hipBLASLt path enabled (T2_BLASLT=1, default 1) */
    void *d_act_bf16;                /* scratch [4096*8192] BF16 (~64 MB) for activation cast */
    hipFunction_t fn_layernorm;      /* layernorm_f32 */
    hipFunction_t fn_layernorm_pack; /* layernorm_pack_bf16_f32 (fused) */
    hipFunction_t fn_adaln_pack;     /* adaln_pack_bf16_f32 (fused) */
    hipFunction_t fn_gated_layernorm_pack; /* gated_add+LN+pack fused */
    hipFunction_t fn_gated_adaln_pack;     /* gated_add+adaLN+pack fused */

    /* TRELLIS.2-specific ops */
    hipFunction_t fn_adaln;
    hipFunction_t fn_gated_add;
    hipFunction_t fn_gated_add_v4;   /* float4 2D variant (dim%4==0) */
    hipFunction_t fn_residual_add;
    hipFunction_t fn_modulation;
    hipFunction_t fn_modulation_par;
    hipFunction_t fn_mod_add_blkbias;
    hipFunction_t fn_t_embed;        /* timestep_embed_cossin_f32 */
    hipFunction_t fn_silu;
    hipFunction_t fn_gelu;
    hipFunction_t fn_gelu_pack;      /* gelu_pack_bf16_f32: fused GELU + F32->BF16 */
    hipFunction_t fn_split_qkv;
    hipFunction_t fn_split_kv;
    hipFunction_t fn_rms_norm_ph;
    hipFunction_t fn_rms_norm_ph_wave;  /* wave32 fast path; head_dim % 32 == 0 */
    hipFunction_t fn_rms_norm_rope_ph_wave;  /* fused RMSNorm + 3D RoPE wave32 */
    hipFunction_t fn_rope_3d;
    hipFunction_t fn_ln_noaffine;
    hipFunction_t fn_flash_sa;       /* flash_attn_sa_f32 (scalar fallback) */
    hipFunction_t fn_flash_sa_wmma;  /* flash_attn_sa_wmma_f32 (gfx12) */
    hipFunction_t fn_flash_sa_wmma_bc32;  /* flash_attn_sa_wmma_bc32_f32 (gfx12) */
    hipFunction_t fn_flash_sa_wmma_bc32_db;  /* flash_attn_sa_wmma_bc32_db_f32 (gfx12) */
    hipFunction_t fn_flash_sa_wmma_b16_db;   /* flash_attn_sa_wmma_b16_db_f32 (gfx12) */
    hipFunction_t fn_cross_attn;     /* cross_attn_tiled_f32 (scalar fallback) */
    hipFunction_t fn_cross_attn_wmma;/* cross_attn_wmma_f32 (gfx12) */
    hipFunction_t fn_cross_attn_wmma_bc32; /* cross_attn_wmma_bc32_f32 (gfx12) */
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
    void *d_mod_base;  /* [6*DIT_DIM] shared GEMV result (silu(t)*mod_w+mod_b) */
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

/* ---- FP8 E4M3 -> F32 host LUT (mirrors GPU constant memory `d_fp8_to_f32_lut`) ---- */
static float t2_fp8_to_f32_lut[256];
static int   t2_fp8_to_f32_lut_init = 0;

static float t2_fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) f = ldexpf((float)mant / 8.0f, -6);
    else if (exp == 15 && mant == 7) return 0.0f;  /* NaN -> 0 */
    else f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -f : f;
}

static void t2_init_fp8_lut(void) {
    if (t2_fp8_to_f32_lut_init) return;
    for (int i = 0; i < 256; i++)
        t2_fp8_to_f32_lut[i] = t2_fp8_e4m3_to_f32((uint8_t)i);
    t2_fp8_to_f32_lut_init = 1;
}

/* Upload weight tensor (.weight). On FP8 dtype, returns raw bytes and stores
 * the per-tensor scale (read from sibling "<base>.weight.scale" tensor) in
 * *out_scale. Otherwise returns F32 dequant and sets *out_scale = -1.0f.
 *
 * Caller knows the field name pattern (e.g. "blocks.N.<sub>.weight"). */
static void *st_upload_wt_w(st_context *st, const char *weight_name,
                            float *out_scale, int verbose) {
    *out_scale = -1.0f;
    int idx = safetensors_find(st, weight_name);
    if (idx < 0) {
        if (verbose >= 2) fprintf(stderr, "  [skip] %s\n", weight_name);
        return NULL;
    }
    const char *dtype = safetensors_dtype(st, idx);
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        /* Raw FP8 upload, sibling .scale tensor */
        size_t nb = safetensors_nbytes(st, idx);
        void *data = safetensors_data(st, idx);
        char scale_name[256];
        snprintf(scale_name, sizeof(scale_name), "%s.scale", weight_name);
        int sidx = safetensors_find(st, scale_name);
        float s = 1.0f;
        if (sidx >= 0) s = *(const float *)safetensors_data(st, sidx);
        else if (verbose >= 1)
            fprintf(stderr, "  [warn] %s has FP8 dtype but no sibling .scale; using 1.0\n",
                    weight_name);
        *out_scale = s;
        void *d = hip_upload_raw(data, nb);  /* 1 byte/element */
        if (verbose >= 2) fprintf(stderr, "  %s: FP8 (%.1f MB) scale=%.6g\n",
                                  weight_name, nb / 1048576.0f, (double)s);
        return d;
    }
    /* Fall through to F32 dequant */
    return st_upload_f32(st, weight_name, verbose);
}

/* Upload weight as BF16-raw if dtype is BF16, else fall back to F32 dequant.
 * Returns 0 on success. Sets *out_w_bf16 (BF16 path) OR *out_w_f32 (fallback)
 * exclusively — never both — so the caller knows which path will dispatch and
 * avoids holding two copies on-GPU. *out_scale is set to -1.0f (not FP8). */
static int st_upload_wt_bf16_pref(st_context *st, const char *weight_name,
                                  int prefer_bf16,
                                  void **out_w_bf16, void **out_w_f32,
                                  float *out_scale, int verbose) {
    *out_w_bf16 = NULL;
    *out_w_f32 = NULL;
    *out_scale = -1.0f;
    int idx = safetensors_find(st, weight_name);
    if (idx < 0) return 0;
    const char *dtype = safetensors_dtype(st, idx);
    if (prefer_bf16 &&
        (strcmp(dtype, "BF16") == 0 || strcmp(dtype, "BFloat16") == 0)) {
        size_t nb = safetensors_nbytes(st, idx);
        void *data = safetensors_data(st, idx);
        *out_w_bf16 = hip_upload_raw(data, nb);
        if (verbose >= 2) fprintf(stderr, "  %s: BF16 raw (%.1f MB)\n",
                                  weight_name, nb / 1048576.0f);
        return 0;
    }
    /* Non-BF16 (e.g. F8_E4M3 with sibling .scale, or F32) — defer to wt_w. */
    *out_w_f32 = st_upload_wt_w(st, weight_name, out_scale, verbose);
    return 0;
}

/* Upload raw BF16 bytes for hipBLASLt path. Returns NULL if tensor missing
 * or dtype is not BF16. Does NOT touch *out_scale. */
static void *st_upload_bf16_raw(st_context *st, const char *weight_name, int verbose) {
    int idx = safetensors_find(st, weight_name);
    if (idx < 0) return NULL;
    const char *dtype = safetensors_dtype(st, idx);
    if (strcmp(dtype, "BF16") != 0 && strcmp(dtype, "BFloat16") != 0) return NULL;
    size_t nb = safetensors_nbytes(st, idx);
    void *data = safetensors_data(st, idx);
    void *d = hip_upload_raw(data, nb);
    if (verbose >= 2) fprintf(stderr, "  %s: BF16 raw (%.1f MB)\n",
                              weight_name, nb / 1048576.0f);
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

static void run_broadcast_bias(hip_trellis2_runner *r, void *out,
                                const void *bias, int n_tok, int n_out);

/* GEMM dispatcher.
 *
 * Y[n_tok, n_out] = X[n_tok, n_in] @ W[n_out, n_in]^T + (bias ? bias : 0)
 *
 * `wt` carries the weight pointer plus an FP8/F32 discriminator:
 *   wt->scale >= 0  -> w is FP8 raw bytes; dispatch BF16-act × FP8-wt WMMA when eligible
 *   wt->scale <  0  -> w is F32; dispatch BF16 WMMA when eligible, else scalar F32
 *
 * The `bias` argument is kept separate (and is NULL at all current call sites,
 * since callers add bias via run_broadcast_bias afterward). */
static int gemm(hip_trellis2_runner *r, void *Y, const t2_wt *wt, const void *X,
                const void *bias, int n_out, int n_in, int n_tok) {
    const void *W = wt->w;

    /* hipBLASLt BF16 path: when a BF16 weight is preloaded and bridge is up.
     * Y is F32 [n_tok, n_out]; X is F32 [n_tok, n_in] — pack to BF16 scratch.
     * mm_blaslt_run_bf16(M=n_tok, N=n_out, K=n_in). */
    if (wt->w_bf16 && r->use_blaslt && n_tok >= 8) {
        int n_act = n_tok * n_in;
        int n_act4 = (n_act + 3) / 4;
        void *dst = r->d_act_bf16;
        void *src = (void *)X;
        void *args0[] = { &dst, &src, &n_act };
        if (hipModuleLaunchKernel(r->fn_pack_bf16,
                (n_act4 + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args0, NULL) != hipSuccess)
            return -1;
        if (mm_blaslt_run_bf16_bias(Y, wt->w_bf16, r->d_act_bf16, bias,
                                    n_tok, n_out, n_in, NULL) != 0) return -1;
        return 0;
    }

    /* FP8 (BF16-act × FP8-wt) WMMA path: kernel tile is 128x128 with 16x16x16
     * WMMA, so we need n_out % 16 == 0 and n_in % 16 == 0. The kernel guards
     * partial CTAs in n_tok / n_out but not in n_in (k loop is unconditional).
     * Aligned safetensors weights from quantize_dit_fp8.py satisfy n_out%64==0. */
    if (wt->scale >= 0.0f && r->use_fp8 && r->fn_gemm_fp8_wmma &&
        n_tok >= 16 && n_in >= 16 &&
        (n_out % 16) == 0 && (n_in % 16) == 0) {
        int gdx = (n_out + 127) / 128;
        int gdy = (n_tok + 127) / 128;
        float w_scale = wt->scale;
        void *args[] = { &Y, &W, &X, &bias, &n_out, &n_in, &n_tok, &w_scale };
        return hipModuleLaunchKernel(r->fn_gemm_fp8_wmma,
            gdx, gdy, 1, 256, 1, 1, 0, 0, args, NULL) == hipSuccess ? 0 : -1;
    }

    /* If we somehow reached here with FP8 weights, that means dimensions are
     * too small for the WMMA tile (n_in<16 or n_out<16). No FP8 fallback yet
     * — these tensors weren't quantized in the first place (input_layer,
     * out_layer have n_out=8). Fail loudly so we catch loader bugs. */
    if (wt->scale >= 0.0f) {
        fprintf(stderr,
                "T2-HIP: gemm() got FP8 weight at unsupported shape "
                "n_out=%d n_in=%d n_tok=%d (no FP8 fallback)\n",
                n_out, n_in, n_tok);
        return -1;
    }

    /* BF16 WMMA path (F32 weights truncated in flight): n_tok, n_in, n_out
     * all multiples of 16. */
    if (r->use_wmma && n_tok >= 16 && n_in >= 16 &&
        (n_out % 16) == 0 && (n_in % 16) == 0) {
        int gdx = (n_out + 127) / 128;
        int gdy = (n_tok + 127) / 128;
        void *args[] = { &Y, &W, &X, &bias, &n_out, &n_in, &n_tok };
        return hipModuleLaunchKernel(r->fn_gemm_wmma,
            gdx, gdy, 1, 256, 1, 1, 0, 0, args, NULL) == hipSuccess ? 0 : -1;
    }
    /* Scalar F32 fallback: Grid (ceil(n_out/64), ceil(n_tok/16)), Block (16,16) */
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
    LOAD_FN(fn_layernorm_pack, "layernorm_pack_bf16_f32");
    LOAD_FN(fn_gated_layernorm_pack, "gated_layernorm_pack_bf16_f32");
    LOAD_FN(fn_gated_adaln_pack,     "gated_adaln_pack_bf16_f32");
    LOAD_FN(fn_adaln,        "adaln_f32");
    LOAD_FN(fn_adaln_pack,   "adaln_pack_bf16_f32");
    LOAD_FN(fn_gated_add,    "gated_add_f32");
    LOAD_FN(fn_gated_add_v4, "gated_add_v4_f32");
    LOAD_FN(fn_residual_add, "residual_add_f32");
    LOAD_FN(fn_modulation,   "modulation_f32");
    LOAD_FN(fn_modulation_par, "modulation_par_f32");
    LOAD_FN(fn_mod_add_blkbias, "mod_add_blkbias_f32");
    LOAD_FN(fn_t_embed,      "timestep_embed_cossin_f32");
    LOAD_FN(fn_silu,         "silu_inplace_f32");
    LOAD_FN(fn_gelu,         "gelu_inplace_f32");
    LOAD_FN(fn_gelu_pack,    "gelu_pack_bf16_f32");
    LOAD_FN(fn_split_qkv,   "split_qkv_chunk_f32");
    LOAD_FN(fn_split_kv,    "split_kv_chunk_f32");
    LOAD_FN(fn_rms_norm_ph,  "rms_norm_perhead_f32");
    LOAD_FN(fn_rms_norm_ph_wave, "rms_norm_perhead_wave_f32");
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
    LOAD_FN(fn_pack_bf16,    "pack_bf16_from_f32");
#undef LOAD_FN

    /* Optional: BF16 WMMA kernels (gfx1200/gfx1201 only). Missing on other
     * archs is not an error — we fall back to scalar F32 paths. */
    r->use_wmma = 0;
    if (hipModuleGetFunction(&r->fn_gemm_wmma, r->mod, "gemm_bf16w_bf16a_wmma_t") == hipSuccess) {
        const char *env = getenv("T2_WMMA");
        if (!env || strcmp(env, "0") != 0) {
            r->use_wmma = 1;
            fprintf(stderr, "T2-HIP: BF16 WMMA GEMM enabled (gfx12 matrix cores)\n");
        } else {
            fprintf(stderr, "T2-HIP: BF16 WMMA GEMM disabled by T2_WMMA=0\n");
        }
    } else {
        fprintf(stderr, "T2-HIP: BF16 WMMA GEMM not available (non-gfx12 arch); using F32 fallback\n");
    }
    if (hipModuleGetFunction(&r->fn_flash_sa_wmma, r->mod, "flash_attn_sa_wmma_f32") != hipSuccess) {
        r->fn_flash_sa_wmma = NULL;
        fprintf(stderr, "T2-HIP: BF16 WMMA flash-attn not available; using scalar FA\n");
    } else if (r->use_wmma) {
        fprintf(stderr, "T2-HIP: BF16 WMMA flash-attn enabled\n");
    }
    r->fn_flash_sa_wmma_bc32 = NULL;
    r->fn_flash_sa_wmma_bc32_db = NULL;
    r->fn_flash_sa_wmma_b16_db = NULL;
    if (r->use_wmma) {
        const char *bc32 = getenv("T2_FA_BC32");
        int want = (bc32 == NULL) ? 1 : atoi(bc32);  /* default ON */
        if (want) {
            if (hipModuleGetFunction(&r->fn_flash_sa_wmma_bc32, r->mod,
                                     "flash_attn_sa_wmma_bc32_f32") == hipSuccess) {
                fprintf(stderr, "T2-HIP: BF16 WMMA flash-attn Bc=32 enabled%s\n",
                        bc32 ? " (T2_FA_BC32=1)" : " (default)");
            }
        }
        const char *db = getenv("T2_FA_DB");
        /* Default OFF: in microbench the doubled LDS (55 KB vs 38 KB) cut
         * CTA-per-WGP from 3 to 2, costing more occupancy than the prefetch
         * recovers. Kernel is preserved for future tile/occupancy retuning. */
        int want_db = (db == NULL) ? 0 : atoi(db);
        const char *b16db = getenv("T2_FA_B16DB");
        int want_b16db = (b16db == NULL) ? 0 : atoi(b16db);
        if (want_b16db) {
            if (hipModuleGetFunction(&r->fn_flash_sa_wmma_b16_db, r->mod,
                                     "flash_attn_sa_wmma_b16_db_f32") == hipSuccess) {
                fprintf(stderr, "T2-HIP: BF16 WMMA flash-attn Bc=16 KV double-buffer enabled (T2_FA_B16DB=1)\n");
            }
        }
        if (want_db) {
            if (hipModuleGetFunction(&r->fn_flash_sa_wmma_bc32_db, r->mod,
                                     "flash_attn_sa_wmma_bc32_db_f32") == hipSuccess) {
                fprintf(stderr, "T2-HIP: BF16 WMMA flash-attn Bc=32 KV double-buffer enabled%s\n",
                        db ? " (T2_FA_DB=1)" : " (default)");
            }
        }
    }
    if (hipModuleGetFunction(&r->fn_cross_attn_wmma, r->mod, "cross_attn_wmma_f32") != hipSuccess) {
        r->fn_cross_attn_wmma = NULL;
        fprintf(stderr, "T2-HIP: BF16 WMMA cross-attn not available; using scalar CA\n");
    } else if (r->use_wmma) {
        fprintf(stderr, "T2-HIP: BF16 WMMA cross-attn enabled\n");
    }
    r->fn_rms_norm_rope_ph_wave = NULL;
    {
        const char *e = getenv("T2_QKNORM_FUSE");
        int want = (e == NULL) ? 1 : atoi(e);
        if (want) {
            if (hipModuleGetFunction(&r->fn_rms_norm_rope_ph_wave, r->mod,
                                     "rms_norm_rope_perhead_wave_f32") == hipSuccess) {
                fprintf(stderr, "T2-HIP: fused RMSNorm+RoPE wave32 enabled%s\n",
                        e ? " (T2_QKNORM_FUSE=1)" : " (default)");
            }
        }
    }
    r->fn_cross_attn_wmma_bc32 = NULL;
    if (r->use_wmma && r->fn_cross_attn_wmma) {
        const char *cabc32 = getenv("T2_CA_BC32");
        int want_cabc32 = (cabc32 == NULL) ? 1 : atoi(cabc32);
        if (want_cabc32) {
            if (hipModuleGetFunction(&r->fn_cross_attn_wmma_bc32, r->mod,
                                     "cross_attn_wmma_bc32_f32") == hipSuccess) {
                fprintf(stderr, "T2-HIP: BF16 WMMA cross-attn Bc=32 enabled%s\n",
                        cabc32 ? " (T2_CA_BC32=1)" : " (default)");
            }
        }
    }

    /* Optional: BF16-act × FP8-wt WMMA (gfx12 only). Becomes active later in
     * load_dit if any tensor is FP8 dtype. */
    r->fn_gemm_fp8_wmma = NULL;
    r->use_fp8 = 0;
    if (hipModuleGetFunction(&r->fn_gemm_fp8_wmma, r->mod, "gemm_fp8w_bf16a_wmma_t") == hipSuccess) {
        /* Upload FP8 LUT to constant memory regardless of whether weights are
         * FP8 — cheap (1 KB) and lets load_dit decide later. */
        t2_init_fp8_lut();
        hipDeviceptr_t d_lut = 0;
        size_t lut_size = 0;
        if (hipModuleGetGlobal(&d_lut, &lut_size, r->mod, "d_fp8_to_f32_lut") == hipSuccess
            && lut_size == 256 * sizeof(float)) {
            hipMemcpyHtoD(d_lut, t2_fp8_to_f32_lut, 256 * sizeof(float));
            fprintf(stderr, "T2-HIP: FP8 WMMA GEMM kernel available (activates on FP8 weights)\n");
        } else {
            fprintf(stderr, "T2-HIP: FP8 LUT symbol missing; FP8 path disabled\n");
            r->fn_gemm_fp8_wmma = NULL;
        }
    }

    fprintf(stderr, "T2-HIP: all %d kernels loaded OK\n", 24);

    /* Optional: hipBLASLt BF16 GEMM bridge (default ON; T2_BLASLT=0 disables).
     * Activates only after load_dit populates wt->w_bf16 for eligible weights. */
    r->use_blaslt = 0;
    r->d_act_bf16 = NULL;
    {
        const char *env = getenv("T2_BLASLT");
        int want = (!env || strcmp(env, "0") != 0);
        if (want) {
            if (mm_blaslt_init() == 0) {
                /* Activation scratch: max n_tok * max n_in * sizeof(bf16) =
                 * 4096 * DIT_FFN(6144) * 2 = 48 MB. Round up to 64 MB. */
                size_t scratch_bytes = (size_t)4096 * 8192 * 2;
                if (hipMalloc(&r->d_act_bf16, scratch_bytes) == hipSuccess) {
                    r->use_blaslt = 1;
                    fprintf(stderr, "T2-HIP: hipBLASLt BF16 GEMM enabled (scratch %.0f MB)\n",
                            scratch_bytes / 1048576.0);
                } else {
                    fprintf(stderr, "T2-HIP: hipBLASLt scratch alloc failed; disabling\n");
                    mm_blaslt_destroy();
                }
            } else {
                fprintf(stderr, "T2-HIP: mm_blaslt_init failed; disabling hipBLASLt path\n");
            }
        } else {
            fprintf(stderr, "T2-HIP: hipBLASLt path disabled by T2_BLASLT=0\n");
        }
    }

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
    hipMalloc(&r->d_mod_base, 6 * DIT_DIM * sizeof(float));
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
        free_hip(&b->sa_qkv.w); free_hip(&b->sa_qkv.w_bf16); free_hip(&b->sa_qkv.b);
        free_hip(&b->sa_q_norm); free_hip(&b->sa_k_norm);
        free_hip(&b->sa_out.w); free_hip(&b->sa_out.w_bf16); free_hip(&b->sa_out.b);
        free_hip(&b->norm2_w); free_hip(&b->norm2_b);
        free_hip(&b->ca_q.w); free_hip(&b->ca_q.w_bf16); free_hip(&b->ca_q.b);
        free_hip(&b->ca_q_norm); free_hip(&b->ca_k_norm);
        free_hip(&b->ca_kv.w); free_hip(&b->ca_kv.w_bf16); free_hip(&b->ca_kv.b);
        free_hip(&b->ca_out.w); free_hip(&b->ca_out.w_bf16); free_hip(&b->ca_out.b);
        free_hip(&b->mlp_fc1.w); free_hip(&b->mlp_fc1.w_bf16); free_hip(&b->mlp_fc1.b);
        free_hip(&b->mlp_fc2.w); free_hip(&b->mlp_fc2.w_bf16); free_hip(&b->mlp_fc2.b);
        free_hip(&b->mod_bias);
        free_hip(&r->ca_K_cache[i]);
        free_hip(&r->ca_V_cache[i]);
    }

    /* Activation buffers */
    free_hip(&r->d_h); free_hip(&r->d_ln_h); free_hip(&r->d_qkv);
    free_hip(&r->d_Q); free_hip(&r->d_K); free_hip(&r->d_V);
    free_hip(&r->d_ca_Q); free_hip(&r->d_attn_out); free_hip(&r->d_mlp_mid);
    free_hip(&r->d_t_emb); free_hip(&r->d_t_silu);
    free_hip(&r->d_ada_out); free_hip(&r->d_mod); free_hip(&r->d_mod_base);
    free_hip(&r->d_noise); free_hip(&r->d_vel); free_hip(&r->d_cond);

    /* Decoder */
    for (int i = 0; i < 4; i++) free_hip(&r->d_dec[i]);

    /* hipBLASLt cleanup */
    if (r->d_act_bf16) { hipFree(r->d_act_bf16); r->d_act_bf16 = NULL; }
    if (r->use_blaslt) { mm_blaslt_destroy(); r->use_blaslt = 0; }

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

    /* Global weights. FP8-eligible 2D weights consumed by gemm() use
     * st_upload_wt_w (auto-detects F8_E4M3 dtype + sibling .scale tensor).
     * Other paths (small dims, modulation custom kernel) stay F32. */
    r->dit_input.w  = st_upload_f32(st, "input_layer.weight", v);  /* [1536,8] - too small */
    r->dit_input.scale = -1.0f;
    r->dit_input.b  = st_upload_f32(st, "input_layer.bias",   v);
    r->dit_t_fc1.w  = st_upload_wt_w(st, "t_embedder.mlp.0.weight", &r->dit_t_fc1.scale, v);
    r->dit_t_fc1.b  = st_upload_f32(st, "t_embedder.mlp.0.bias",   v);
    r->dit_t_fc2.w  = st_upload_wt_w(st, "t_embedder.mlp.2.weight", &r->dit_t_fc2.scale, v);
    r->dit_t_fc2.b  = st_upload_f32(st, "t_embedder.mlp.2.bias",   v);
    r->dit_ada_mod.w = st_upload_f32(st, "adaLN_modulation.1.weight", v);  /* fn_modulation custom kernel */
    r->dit_ada_mod.scale = -1.0f;
    r->dit_ada_mod.b = st_upload_f32(st, "adaLN_modulation.1.bias",   v);
    r->dit_out.w    = st_upload_f32(st, "out_layer.weight", v);  /* [8,1536] - too small */
    r->dit_out.scale = -1.0f;
    r->dit_out.b    = st_upload_f32(st, "out_layer.bias",   v);

    if (r->dit_t_fc1.scale >= 0.0f || r->dit_t_fc2.scale >= 0.0f) r->use_fp8 = 1;

    /* For hipBLASLt path, also upload raw BF16 weights for the large 2D
     * matmuls (skip t_embedder — n_tok=1 doesn't satisfy n_tok>=8 gate; skip
     * adaLN_modulation — consumed by fn_modulation custom kernel; skip
     * input_layer/out_layer — n_in/n_out=8 too small for tile). Per-block
     * weights are uploaded below. Only valid for native-BF16 safetensors. */

    /* Per-block weights */
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        char name[128];
        t2_block_gpu *b = &r->blocks[bi];
        #define BN(n) (snprintf(name,sizeof(name),"blocks.%d.%s",bi,n), name)

        /* Per-block large GEMM weights: prefer BF16-raw upload (hipBLASLt path).
         * Falls back to F32 only when dtype is FP8 or non-BF16. Avoids holding
         * both F32 and BF16 copies — saves ~5 GB VRAM and ~3× PCIe at load. */
        st_upload_wt_bf16_pref(st, BN("self_attn.to_qkv.weight"), r->use_blaslt,
                               &b->sa_qkv.w_bf16, &b->sa_qkv.w, &b->sa_qkv.scale, v);
        b->sa_qkv.b  = st_upload_f32(st, BN("self_attn.to_qkv.bias"),   v);
        b->sa_q_norm = st_upload_f32(st, BN("self_attn.q_rms_norm.gamma"), v);
        b->sa_k_norm = st_upload_f32(st, BN("self_attn.k_rms_norm.gamma"), v);
        st_upload_wt_bf16_pref(st, BN("self_attn.to_out.weight"), r->use_blaslt,
                               &b->sa_out.w_bf16, &b->sa_out.w, &b->sa_out.scale, v);
        b->sa_out.b  = st_upload_f32(st, BN("self_attn.to_out.bias"),   v);

        b->norm2_w   = st_upload_f32(st, BN("norm2.weight"), v);
        b->norm2_b   = st_upload_f32(st, BN("norm2.bias"),   v);
        st_upload_wt_bf16_pref(st, BN("cross_attn.to_q.weight"), r->use_blaslt,
                               &b->ca_q.w_bf16, &b->ca_q.w, &b->ca_q.scale, v);
        b->ca_q.b    = st_upload_f32(st, BN("cross_attn.to_q.bias"),   v);
        b->ca_q_norm = st_upload_f32(st, BN("cross_attn.q_rms_norm.gamma"), v);
        b->ca_k_norm = st_upload_f32(st, BN("cross_attn.k_rms_norm.gamma"), v);
        st_upload_wt_bf16_pref(st, BN("cross_attn.to_kv.weight"), r->use_blaslt,
                               &b->ca_kv.w_bf16, &b->ca_kv.w, &b->ca_kv.scale, v);
        b->ca_kv.b   = st_upload_f32(st, BN("cross_attn.to_kv.bias"),   v);
        st_upload_wt_bf16_pref(st, BN("cross_attn.to_out.weight"), r->use_blaslt,
                               &b->ca_out.w_bf16, &b->ca_out.w, &b->ca_out.scale, v);
        b->ca_out.b  = st_upload_f32(st, BN("cross_attn.to_out.bias"),   v);

        st_upload_wt_bf16_pref(st, BN("mlp.mlp.0.weight"), r->use_blaslt,
                               &b->mlp_fc1.w_bf16, &b->mlp_fc1.w, &b->mlp_fc1.scale, v);
        b->mlp_fc1.b = st_upload_f32(st, BN("mlp.mlp.0.bias"),   v);
        st_upload_wt_bf16_pref(st, BN("mlp.mlp.2.weight"), r->use_blaslt,
                               &b->mlp_fc2.w_bf16, &b->mlp_fc2.w, &b->mlp_fc2.scale, v);
        b->mlp_fc2.b = st_upload_f32(st, BN("mlp.mlp.2.bias"),   v);

        if (b->sa_qkv.scale >= 0.0f) r->use_fp8 = 1;

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

/* Fused layernorm + F32->BF16 pack: writes F32 dst AND BF16 dst_bf16. */
static void run_layernorm_pack(hip_trellis2_runner *r, void *dst, void *dst_bf16,
                                 const void *src, const void *w, const void *b,
                                 int N, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &dst_bf16, &src, &w, &b, &dim, &eps};
    hipModuleLaunchKernel(r->fn_layernorm_pack, N, 1, 1, 256, 1, 1,
                          256 * 2 * sizeof(float), 0, args, NULL);
}

/* Fused adaln + F32->BF16 pack: writes F32 dst AND BF16 dst_bf16. */
static void run_adaln_pack(hip_trellis2_runner *r, void *dst, void *dst_bf16,
                            const void *src, const void *shift, const void *scale,
                            int N, int dim) {
    float eps = 1e-6f;
    void *args[] = {&dst, &dst_bf16, &src, &shift, &scale, &dim, &eps};
    hipModuleLaunchKernel(r->fn_adaln_pack, N, 1, 1, 256, 1, 1,
                          256 * 2 * sizeof(float), 0, args, NULL);
}

/* Fused gated_add + layernorm + BF16 pack: x_io = x_io + gate*r; LN(x_io) -> dst, dst_bf16 */
static int run_gated_layernorm_pack(hip_trellis2_runner *r, void *x_io, void *dst,
                                     void *dst_bf16, const void *resid,
                                     const void *gate, const void *w, const void *b,
                                     int N, int dim) {
    if (!r->fn_gated_layernorm_pack) return -1;
    float eps = 1e-6f;
    void *args[] = {&x_io, &dst, &dst_bf16, (void *)&resid, (void *)&gate,
                    (void *)&w, (void *)&b, &dim, &eps};
    return hipModuleLaunchKernel(r->fn_gated_layernorm_pack, N, 1, 1, 256, 1, 1,
                                  256 * sizeof(float), 0, args, NULL) == hipSuccess
           ? 0 : -1;
}

/* Fused gated_add + adaLN + BF16 pack. */
static int run_gated_adaln_pack(hip_trellis2_runner *r, void *x_io, void *dst,
                                  void *dst_bf16, const void *resid,
                                  const void *gate, const void *shift,
                                  const void *scale, int N, int dim) {
    if (!r->fn_gated_adaln_pack) return -1;
    float eps = 1e-6f;
    void *args[] = {&x_io, &dst, &dst_bf16, (void *)&resid, (void *)&gate,
                    (void *)&shift, (void *)&scale, &dim, &eps};
    return hipModuleLaunchKernel(r->fn_gated_adaln_pack, N, 1, 1, 256, 1, 1,
                                  256 * 2 * sizeof(float), 0, args, NULL) == hipSuccess
           ? 0 : -1;
}

/* gated_add: dst[N*dim] += gate[dim] * src[N*dim] */
static void run_gated_add(hip_trellis2_runner *r, void *dst, const void *src,
                            const void *gate, int N, int dim) {
    /* Prefer the float4 2D variant when dim is a multiple of 4 — avoids the
     * `i % dim` modulo and issues 16-byte vector loads/stores. */
    if (r->fn_gated_add_v4 && (dim & 3) == 0) {
        void *args[] = {&dst, (void *)&src, (void *)&gate, &N, &dim};
        int dim4 = dim >> 2;
        int gx = (dim4 + 255) / 256;
        hipModuleLaunchKernel(r->fn_gated_add_v4, gx, N, 1, 256, 1, 1,
                              0, 0, args, NULL);
        return;
    }
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

/* gelu_pack_bf16_f32: fused GELU + F32->BF16 pack into r->d_act_bf16 scratch.
 * Used before the next BF16 GEMM (mlp_fc2) to skip the standalone pack_bf16
 * launch and halve memory traffic on the activation. */
static void run_gelu_pack(hip_trellis2_runner *r, void *dst_bf16,
                           const void *src_f32, int n) {
    int n4 = (n + 3) / 4;
    void *args[] = { &dst_bf16, (void *)&src_f32, &n };
    hipModuleLaunchKernel(r->fn_gelu_pack, (n4 + 255) / 256, 1, 1, 256, 1, 1,
                          0, 0, args, NULL);
}

/* BF16 GEMM with X already packed into r->d_act_bf16 (skip pack_bf16). */
static int gemm_bf16x_prepacked(hip_trellis2_runner *r, void *Y,
                                  const t2_wt *wt, const void *bias,
                                  int n_out, int n_in, int n_tok) {
    if (!(wt->w_bf16 && r->use_blaslt && n_tok >= 8)) {
        fprintf(stderr, "T2-HIP: gemm_bf16x_prepacked called without BF16 weight or hipBLASLt\n");
        return -1;
    }
    return mm_blaslt_run_bf16_bias(Y, wt->w_bf16, r->d_act_bf16, bias,
                                   n_tok, n_out, n_in, NULL);
}

/* BF16 GEMM with fused GELU + bias + BF16 D-output: D[bf16] = GELU(X*W^T + bias).
 * X is r->d_act_bf16, D is written to dst_bf16 directly. Replaces the
 * fc1(F32-out) + gelu_pack(F32→BF16) two-pass with a single kernel. */
static int gemm_bf16x_prepacked_gelu_bf16d(hip_trellis2_runner *r, void *dst_bf16,
                                             const t2_wt *wt, const void *bias,
                                             int n_out, int n_in, int n_tok) {
    if (!(wt->w_bf16 && r->use_blaslt && n_tok >= 8 && bias)) {
        fprintf(stderr, "T2-HIP: gemm_bf16x_prepacked_gelu_bf16d preconds failed\n");
        return -1;
    }
    return mm_blaslt_run_bf16_bias_gelu_bf16d(dst_bf16, wt->w_bf16, r->d_act_bf16,
                                                bias, n_tok, n_out, n_in, NULL);
}

/* Same as gemm_bf16x_prepacked but with fused residual: Y = X*W^T + bias + Y. */
static int gemm_bf16x_prepacked_residual(hip_trellis2_runner *r, void *Y,
                                          const t2_wt *wt, const void *bias,
                                          int n_out, int n_in, int n_tok) {
    if (!(wt->w_bf16 && r->use_blaslt && n_tok >= 8)) {
        fprintf(stderr, "T2-HIP: gemm_bf16x_prepacked_residual called without BF16 weight or hipBLASLt\n");
        return -1;
    }
    return mm_blaslt_run_bf16_bias_residual(Y, Y, wt->w_bf16, r->d_act_bf16,
                                             bias, n_tok, n_out, n_in, NULL);
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
    if (r->fn_rms_norm_ph_wave && (head_dim & 31) == 0 && head_dim <= 256) {
        hipModuleLaunchKernel(r->fn_rms_norm_ph_wave, total, 1, 1, 32, 1, 1,
                              0, 0, args, NULL);
        return;
    }
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

/* Fused RMSNorm + 3D RoPE: replaces run_rms_norm_ph + run_rope_3d for Q/K when
 * head_dim % 32 == 0. Single R+W pass over the [n_tok, n_heads, head_dim]
 * tensor; RoPE rotation done in-register via ds_swizzle on the wave32 layout. */
static int run_rms_norm_rope_ph(hip_trellis2_runner *r, void *data,
                                 const void *gamma, int n_tok, int n_heads,
                                 int head_dim, int stride) {
    if (!r->fn_rms_norm_rope_ph_wave || (head_dim & 31) != 0 || head_dim > 256) {
        return -1;
    }
    float eps = 1e-6f;
    int total = n_tok * n_heads;
    int n_freqs = r->n_rope_freqs;
    void *args[] = {&data, &gamma, &r->d_rope_cos, &r->d_rope_sin,
                    &n_tok, &n_heads, &head_dim, &stride, &n_freqs, &eps};
    return hipModuleLaunchKernel(r->fn_rms_norm_rope_ph_wave, total, 1, 1,
                                 32, 1, 1, 0, 0, args, NULL) == hipSuccess ? 0 : -1;
}

/* Self-attention dispatch: prefers gfx12 BF16 WMMA kernel when eligible.
 * `out_bf16` is an optional BF16 output co-write target (NULL to skip).
 * `qkv_stride` is the row stride of the Q/K/V tensors in elements; pass 0 for
 * the default contiguous layout (n_heads * head_dim). Use a non-zero stride
 * (e.g. 3*n_heads*head_dim) to read Q/K/V from a shared interleaved buffer.
 * Only the bc32 winner currently honors out_bf16/qkv_stride. */
static void run_flash_sa(hip_trellis2_runner *r, void *out, void *out_bf16,
                          const void *Q, const void *K, const void *V, int N,
                          int qkv_stride) {
    int n_heads  = DIT_HEADS;
    int head_dim = DIT_HEAD_DIM;
    void *args[] = {&out, &Q, &K, &V, &N, &n_heads, &head_dim};
    void *args_bc32[] = {&out, &Q, &K, &V, &N, &n_heads, &head_dim, &out_bf16,
                         &qkv_stride};
    if (r->use_wmma && r->fn_flash_sa_wmma_b16_db && head_dim == 128 && (N % 64) == 0) {
        hipModuleLaunchKernel(r->fn_flash_sa_wmma_b16_db, n_heads, N / 64, 1, 128, 1, 1,
                              0, 0, args, NULL);
        return;
    }
    if (r->use_wmma && r->fn_flash_sa_wmma_bc32_db && head_dim == 128 && (N % 64) == 0) {
        hipModuleLaunchKernel(r->fn_flash_sa_wmma_bc32_db, n_heads, N / 64, 1, 128, 1, 1,
                              0, 0, args, NULL);
        return;
    }
    if (r->use_wmma && r->fn_flash_sa_wmma_bc32 && head_dim == 128 && (N % 64) == 0) {
        hipModuleLaunchKernel(r->fn_flash_sa_wmma_bc32, n_heads, N / 64, 1, 128, 1, 1,
                              0, 0, args_bc32, NULL);
        return;
    }
    if (r->use_wmma && r->fn_flash_sa_wmma && head_dim == 128 && (N % 64) == 0) {
        hipModuleLaunchKernel(r->fn_flash_sa_wmma, n_heads, N / 64, 1, 128, 1, 1,
                              0, 0, args, NULL);
        return;
    }
    int gY = (N + 3) / 4;  /* FA_WARPS=4 */
    size_t smem = 2 * 16 * 128 * sizeof(float);  /* FA_BKV=16, FA_HD=128 */
    hipModuleLaunchKernel(r->fn_flash_sa, DIT_HEADS, gY, 1, 128, 1, 1,
                          smem, 0, args, NULL);
}

/* Cross-attention dispatch: WMMA path when eligible. `out_bf16` is an optional
 * BF16 co-write target (only honored by the bc32 variant; NULL to skip). */
static void run_cross_attn(hip_trellis2_runner *r, void *out, void *out_bf16,
                             const void *Q, const void *K, const void *V,
                             int q_len, int kv_len) {
    float scale  = 1.0f / sqrtf((float)DIT_HEAD_DIM);
    int dim      = DIT_DIM;
    int n_heads  = DIT_HEADS;
    int head_dim = DIT_HEAD_DIM;
    void *args[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim, &n_heads, &head_dim, &scale};
    void *args_bc32[] = {&out, &Q, &K, &V, &q_len, &kv_len, &dim, &n_heads, &head_dim, &scale, &out_bf16};
    if (r->use_wmma && r->fn_cross_attn_wmma_bc32 && head_dim == 128 && (q_len % 64) == 0) {
        hipModuleLaunchKernel(r->fn_cross_attn_wmma_bc32, n_heads, q_len / 64, 1, 128, 1, 1,
                              0, 0, args_bc32, NULL);
        return;
    }
    if (r->use_wmma && r->fn_cross_attn_wmma && head_dim == 128 && (q_len % 64) == 0) {
        hipModuleLaunchKernel(r->fn_cross_attn_wmma, n_heads, q_len / 64, 1, 128, 1, 1,
                              0, 0, args, NULL);
        return;
    }
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
    gemm(r, r->d_h, &r->dit_input, r->d_noise,
         r->dit_input.b, DIT_DIM, DIT_IN_CH, n_tok);
    if (r->b0_dbg) dbg_dl(r->b0_dbg->input_embed, r->d_h, (size_t)n_tok * DIT_DIM);

    /* ---- Timestep embedding (already uploaded to d_t_emb) ---- */
    /* FC1: [DIT_DIM, 256] -> [DIT_DIM], SiLU */
    gemm(r, r->d_t_silu, &r->dit_t_fc1, r->d_t_emb,
         r->dit_t_fc1.b, DIT_DIM, DIT_T_HALF*2, 1);
    run_silu(r, r->d_t_silu, DIT_DIM);
    /* FC2: [DIT_DIM, DIT_DIM] -> [DIT_DIM]
     * Store FC2 output in d_ada_out (unused otherwise). Must not alias with
     * d_mod since per-block modulation writes to d_mod while reading d_t_out. */
    void *d_t_out = r->d_ada_out;
    gemm(r, d_t_out, &r->dit_t_fc2, r->d_t_silu,
         r->dit_t_fc2.b, DIT_DIM, DIT_DIM, 1);

    /* ---- Precompute cross-attention KV cache if not valid ---- */
    if (!r->ca_kv_valid) {
        if (r->verbose) fprintf(stderr, "T2-HIP: building CA KV cache (%d tokens)...\n", cond_len);
        for (int bi = 0; bi < DIT_DEPTH; bi++) {
            t2_block_gpu *b = &r->blocks[bi];
            /* ca_kv_w: [2*DIT_DIM, DIT_COND_DIM] × cond [cond_len, DIT_COND_DIM]^T
             * -> [cond_len, 2*DIT_DIM] */
            gemm(r, r->d_V, &b->ca_kv, r->d_cond,
                 b->ca_kv.b, 2*DIT_DIM, DIT_COND_DIM, cond_len);
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

    /* Precompute shared GEMV once: d_mod_base = silu(t_emb)*mod_w + mod_b.
     * Per-block then adds the block-specific bias via a cheap pointwise kernel,
     * avoiding 30 redundant 12288×2048 GEMVs. */
    int mod_precomputed = 0;
    if (r->fn_mod_add_blkbias) {
        void *out = r->d_mod_base;
        const void *t_emb = d_t_out, *mod_w = r->dit_ada_mod.w, *mod_b = r->dit_ada_mod.b;
        const void *blk_bias_null = NULL;
        int dim = DIT_DIM, out_dim = 6*DIT_DIM;
        void *args[] = { &out, &t_emb, &mod_w, &mod_b, &blk_bias_null, &dim, &out_dim };
        int gx = (out_dim + 255) / 256;
        if (hipModuleLaunchKernel(r->fn_modulation_par, gx, 1, 1, 256, 1, 1,
                                   DIT_DIM * sizeof(float), 0, args, NULL) == hipSuccess) {
            mod_precomputed = 1;
        }
    }

    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        t2_block_gpu *b = &r->blocks[bi];

        /* Per-block modulation: d_mod = d_mod_base + block.mod_bias.
         * Falls back to the full per-block GEMV when the cheap add isn't loaded. */
        if (mod_precomputed) {
            void *out = r->d_mod;
            const void *base = r->d_mod_base;
            const void *blk_bias = b->mod_bias;
            int n = 6 * DIT_DIM;
            void *args[] = { &out, (void *)&base, (void *)&blk_bias, &n };
            int gx = (n + 255) / 256;
            hipModuleLaunchKernel(r->fn_mod_add_blkbias, gx, 1, 1, 256, 1, 1,
                                  0, 0, args, NULL);
        } else {
            void *out = r->d_mod;
            const void *t_emb = d_t_out, *mod_w = r->dit_ada_mod.w, *mod_b = r->dit_ada_mod.b;
            const void *blk_bias = b->mod_bias;
            int dim = DIT_DIM, out_dim = 6*DIT_DIM;
            void *args[] = { &out, &t_emb, &mod_w, &mod_b, &blk_bias, &dim, &out_dim };
            int gx = (out_dim + 255) / 256;
            hipModuleLaunchKernel(r->fn_modulation_par, gx, 1, 1, 256, 1, 1,
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

        /* Norm-fusion gate: when the BF16 hipBLASLt path is active, run a fused
         * (adaln|layernorm) + F32->BF16 pack kernel that also populates
         * r->d_act_bf16 in-place, letting the next GEMM skip pack_bf16. */
        static int norm_fuse_gate = -1;
        if (norm_fuse_gate < 0) {
            const char *e = getenv("T2_NORM_FUSE");
            norm_fuse_gate = (e && atoi(e) == 0) ? 0 : 1;
        }

        /* ---- Self-attention ---- */
        /* adaLN -> d_ln_h (and BF16 copy to scratch when fusion eligible) */
        if (norm_fuse_gate && r->use_blaslt && b->sa_qkv.w_bf16 && n_tok >= 8) {
            run_adaln_pack(r, r->d_ln_h, r->d_act_bf16, r->d_h,
                            shift_sa, scale_sa, n_tok, DIT_DIM);
            if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->ln_h_sa, r->d_ln_h, (size_t)n_tok * DIT_DIM);
            gemm_bf16x_prepacked(r, r->d_qkv, &b->sa_qkv, b->sa_qkv.b,
                                 3*DIT_DIM, DIT_DIM, n_tok);
        } else {
            run_adaln(r, r->d_ln_h, r->d_h, shift_sa, scale_sa, n_tok, DIT_DIM);
            if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->ln_h_sa, r->d_ln_h, (size_t)n_tok * DIT_DIM);
            gemm(r, r->d_qkv, &b->sa_qkv, r->d_ln_h, b->sa_qkv.b, 3*DIT_DIM, DIT_DIM, n_tok);
        }
        /* Split QKV */
        /* Skip the standalone split_qkv kernel when the bc32 FA + wave RMSNorm
         * paths are active: they accept a stride argument so Q/K/V can be read
         * directly from the [n_tok, 3*DIT_DIM] qkv buffer (offsets 0, DIT_DIM,
         * 2*DIT_DIM and stride=3*DIT_DIM). */
        static int qkv_unsplit_gate = -1;
        if (qkv_unsplit_gate < 0) {
            const char *e = getenv("T2_QKV_UNSPLIT");
            qkv_unsplit_gate = (e && atoi(e) == 0) ? 0 : 1;
        }
        int qkv_unsplit = (qkv_unsplit_gate
                           && r->fn_rms_norm_rope_ph_wave
                           && r->use_wmma && r->fn_flash_sa_wmma_bc32
                           && (n_tok % 64) == 0
                           && !r->fn_flash_sa_wmma_b16_db
                           && !r->fn_flash_sa_wmma_bc32_db);
        void *Q_ptr, *K_ptr, *V_ptr;
        int qkv_stride;
        if (qkv_unsplit) {
            Q_ptr = (char *)r->d_qkv + 0 * DIT_DIM * sizeof(float);
            K_ptr = (char *)r->d_qkv + 1 * DIT_DIM * sizeof(float);
            V_ptr = (char *)r->d_qkv + 2 * DIT_DIM * sizeof(float);
            qkv_stride = 3 * DIT_DIM;
        } else {
            void *Q = r->d_Q, *K = r->d_K, *V = r->d_V;
            const void *qkv = r->d_qkv;
            int N = n_tok, W = DIT_DIM;
            void *args[] = { &Q, &K, &V, &qkv, &N, &W };
            hipModuleLaunchKernel(r->fn_split_qkv,
                (n_tok * DIT_DIM + 255)/256, 1, 1, 256, 1, 1, 0, 0, args, NULL);
            Q_ptr = r->d_Q; K_ptr = r->d_K; V_ptr = r->d_V;
            qkv_stride = 0;  /* default = n_heads*head_dim */
        }
        /* QK RMSNorm + 3D RoPE (fused when available, falls back to two-pass) */
        if (run_rms_norm_rope_ph(r, Q_ptr, b->sa_q_norm, n_tok, DIT_HEADS,
                                  DIT_HEAD_DIM,
                                  qkv_unsplit ? qkv_stride : DIT_DIM) != 0) {
            /* Fallback path: two-pass requires non-strided buffer (no stride
             * arg in run_rope_3d), so split must be active in fallback. */
            run_rms_norm_ph(r, Q_ptr, b->sa_q_norm, n_tok, DIT_HEADS,
                            DIT_HEAD_DIM, DIT_DIM);
            run_rope_3d(r, Q_ptr, n_tok);
        }
        if (run_rms_norm_rope_ph(r, K_ptr, b->sa_k_norm, n_tok, DIT_HEADS,
                                  DIT_HEAD_DIM,
                                  qkv_unsplit ? qkv_stride : DIT_DIM) != 0) {
            run_rms_norm_ph(r, K_ptr, b->sa_k_norm, n_tok, DIT_HEADS,
                            DIT_HEAD_DIM, DIT_DIM);
            run_rope_3d(r, K_ptr, n_tok);
        }
        if (bi == 0 && r->b0_dbg) {
            /* Debug dumps assume contiguous Q/K/V — disabled in unsplit mode */
            if (!qkv_unsplit) {
                dbg_dl(r->b0_dbg->q_post, Q_ptr, (size_t)n_tok * DIT_DIM);
                dbg_dl(r->b0_dbg->k_post, K_ptr, (size_t)n_tok * DIT_DIM);
                dbg_dl(r->b0_dbg->v,      V_ptr, (size_t)n_tok * DIT_DIM);
            }
        }
        /* Flash attention -> attn_out (with optional fused BF16 co-write to scratch).
         * The bc32 winner kernel writes both F32 (for dbg) and BF16 (consumed by
         * the next sa_out GEMM, skipping pack_bf16). */
        int fa_fuse = (norm_fuse_gate && r->use_blaslt && b->sa_out.w_bf16
                       && n_tok >= 8 && r->use_wmma && r->fn_flash_sa_wmma_bc32
                       && (n_tok % 64) == 0
                       && !r->fn_flash_sa_wmma_b16_db && !r->fn_flash_sa_wmma_bc32_db);
        /* When fa_fuse is on and not in dbg mode, the F32 d_attn_out write is
         * unused (sa_out reads BF16 d_act_bf16). Pass NULL to skip the F32
         * store inside the kernel — saves ~32MB write per block. */
        int fa_skip_f32 = fa_fuse && !(bi == 0 && r->b0_dbg);
        run_flash_sa(r, fa_skip_f32 ? NULL : r->d_attn_out,
                      fa_fuse ? r->d_act_bf16 : NULL,
                      Q_ptr, K_ptr, V_ptr, n_tok, qkv_stride);
        /* Output projection: [n_tok, DIT_DIM] × [DIT_DIM, DIT_DIM]^T */
        if (fa_fuse) {
            gemm_bf16x_prepacked(r, r->d_ln_h, &b->sa_out, b->sa_out.b,
                                 DIT_DIM, DIT_DIM, n_tok);
        } else {
            gemm(r, r->d_ln_h, &b->sa_out, r->d_attn_out, b->sa_out.b, DIT_DIM, DIT_DIM, n_tok);
        }
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->sa_proj, r->d_ln_h, (size_t)n_tok * DIT_DIM);
        /* Gated residual: d_h += gate_sa * d_ln_h */
        run_gated_add(r, r->d_h, r->d_ln_h, gate_sa, n_tok, DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->h_post_sa, r->d_h, (size_t)n_tok * DIT_DIM);

        /* ---- Cross-attention ---- */
        /* norm2 -> d_ln_h (with optional fused BF16 pack into scratch) */
        if (norm_fuse_gate && r->use_blaslt && b->ca_q.w_bf16 && n_tok >= 8) {
            run_layernorm_pack(r, r->d_ln_h, r->d_act_bf16, r->d_h,
                                b->norm2_w, b->norm2_b, n_tok, DIT_DIM);
            gemm_bf16x_prepacked(r, r->d_ca_Q, &b->ca_q, b->ca_q.b,
                                 DIT_DIM, DIT_DIM, n_tok);
        } else {
            run_layernorm(r, r->d_ln_h, r->d_h, b->norm2_w, b->norm2_b, n_tok, DIT_DIM);
            gemm(r, r->d_ca_Q, &b->ca_q, r->d_ln_h, b->ca_q.b, DIT_DIM, DIT_DIM, n_tok);
        }
        /* Q RMSNorm */
        run_rms_norm_ph(r, r->d_ca_Q, b->ca_q_norm, n_tok, DIT_HEADS, DIT_HEAD_DIM, DIT_DIM);
        /* Cross-attention using cached K,V (with optional BF16 co-write). */
        int ca_fuse = (norm_fuse_gate && r->use_blaslt && b->ca_out.w_bf16
                       && n_tok >= 8 && r->use_wmma && r->fn_cross_attn_wmma_bc32
                       && (n_tok % 64) == 0);
        /* Skip F32 write when CA-fused and not dbg (ca_out reads BF16). */
        int ca_skip_f32 = ca_fuse && !(bi == 0 && r->b0_dbg);
        run_cross_attn(r, ca_skip_f32 ? NULL : r->d_attn_out,
                        ca_fuse ? r->d_act_bf16 : NULL,
                        r->d_ca_Q, r->ca_K_cache[bi], r->ca_V_cache[bi],
                        n_tok, r->ca_kv_len);
        /* Output projection + residual (fused via hipBLASLt beta=1 when active).
         * Cross-attn has no gate, so D = X*W^T + bias + d_h written into d_h. */
        static int ca_res_fuse_gate = -1;
        if (ca_res_fuse_gate < 0) {
            const char *e = getenv("T2_CA_RES_FUSE");
            ca_res_fuse_gate = (e && atoi(e) == 0) ? 0 : 1;
        }
        int ca_res_fuse = (ca_fuse && ca_res_fuse_gate && !(bi == 0 && r->b0_dbg));
        if (ca_res_fuse) {
            gemm_bf16x_prepacked_residual(r, r->d_h, &b->ca_out, b->ca_out.b,
                                           DIT_DIM, DIT_DIM, n_tok);
        } else if (ca_fuse) {
            gemm_bf16x_prepacked(r, r->d_ln_h, &b->ca_out, b->ca_out.b,
                                 DIT_DIM, DIT_DIM, n_tok);
        } else {
            gemm(r, r->d_ln_h, &b->ca_out, r->d_attn_out, b->ca_out.b, DIT_DIM, DIT_DIM, n_tok);
        }
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->ca_proj, r->d_ln_h, (size_t)n_tok * DIT_DIM);
        /* Direct residual (no gate in cross-attn) — folded into GEMM when fused */
        if (!ca_res_fuse) run_residual_add(r, r->d_h, r->d_ln_h, n_tok * DIT_DIM);
        if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->h_post_ca, r->d_h, (size_t)n_tok * DIT_DIM);

        /* ---- MLP ---- */
        /* adaLN (scale2/shift2) -> d_ln_h (with optional fused BF16 pack) */
        if (norm_fuse_gate && r->use_blaslt && b->mlp_fc1.w_bf16 && n_tok >= 8) {
            run_adaln_pack(r, r->d_ln_h, r->d_act_bf16, r->d_h,
                            shift_mlp, scale_mlp, n_tok, DIT_DIM);
            if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->ln_h_mlp, r->d_ln_h, (size_t)n_tok * DIT_DIM);
            gemm_bf16x_prepacked(r, r->d_mlp_mid, &b->mlp_fc1, b->mlp_fc1.b,
                                 DIT_FFN, DIT_DIM, n_tok);
        } else {
            run_adaln(r, r->d_ln_h, r->d_h, shift_mlp, scale_mlp, n_tok, DIT_DIM);
            if (bi == 0 && r->b0_dbg) dbg_dl(r->b0_dbg->ln_h_mlp, r->d_ln_h, (size_t)n_tok * DIT_DIM);
            gemm(r, r->d_mlp_mid, &b->mlp_fc1, r->d_ln_h, b->mlp_fc1.b, DIT_FFN, DIT_DIM, n_tok);
        }
        /* FC2: [n_tok, DIT_FFN] × [DIT_DIM, DIT_FFN]^T -> [n_tok, DIT_DIM]
         * When the BF16 hipBLASLt path is active, fuse GELU + F32->BF16 pack
         * into a single kernel that writes directly to the activation scratch,
         * then call hipBLASLt without a separate pack_bf16 launch. */
        static int gelu_fuse_gate = -1;
        if (gelu_fuse_gate < 0) {
            const char *e = getenv("T2_GELU_FUSE");
            gelu_fuse_gate = (e && atoi(e) == 0) ? 0 : 1;
        }
        if (gelu_fuse_gate && r->use_blaslt && b->mlp_fc2.w_bf16 && n_tok >= 8) {
            run_gelu_pack(r, r->d_act_bf16, r->d_mlp_mid, n_tok * DIT_FFN);
            gemm_bf16x_prepacked(r, r->d_ln_h, &b->mlp_fc2, b->mlp_fc2.b,
                                 DIT_DIM, DIT_FFN, n_tok);
        } else {
            run_gelu(r, r->d_mlp_mid, n_tok * DIT_FFN);
            gemm(r, r->d_ln_h, &b->mlp_fc2, r->d_mlp_mid, b->mlp_fc2.b, DIT_DIM, DIT_FFN, n_tok);
        }
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
    gemm(r, r->d_vel, &r->dit_out, r->d_ln_h, r->dit_out.b, DIT_IN_CH, DIT_DIM, n_tok);

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
