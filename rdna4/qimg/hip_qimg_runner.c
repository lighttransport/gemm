/*
 * hip_qimg_runner.c - HIP/ROCm Qwen-Image text-to-image runner (RDNA4)
 *
 * GPU-accelerated DiT (60 dual-stream blocks) with block-by-block streaming.
 * VAE decode on GPU.
 *
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 * F32 weights on GPU, F32 compute. Single-stream sequential kernel launches.
 *
 * Port of cuda_qimg_runner.h for AMD ROCm/HIP.
 *
 * SPDX-License-Identifier: MIT
 * Copyright 2025 - Present, Light Transport Entertainment Inc.
 */

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"

#include "hip_qimg_runner.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#include "hip_qimg_kernels.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- FP8 E4M3 → F32 CPU conversion ---- */

static float fp8_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    float f;
    if (exp == 0) {
        f = ldexpf((float)mant / 8.0f, -6);
    } else if (exp == 15 && mant == 7) {
        return 0.0f;  /* NaN → 0 */
    } else {
        f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    }
    return sign ? -f : f;
}

static float hip_fp8_to_f32_lut[256];
static int hip_fp8_to_f32_lut_init = 0;

static void init_fp8_to_f32_lut(void) {
    if (hip_fp8_to_f32_lut_init) return;
    for (int i = 0; i < 256; i++)
        hip_fp8_to_f32_lut[i] = fp8_e4m3_to_f32((uint8_t)i);
    hip_fp8_to_f32_lut_init = 1;
}

static uint8_t f32_to_fp8_e4m3_cpu(float f) {
    if (f > 448.0f) f = 448.0f;
    if (f < -448.0f) f = -448.0f;
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = bits >> 31;
    int e = (int)((bits >> 23) & 0xFFu) - 127;
    uint32_t mant = bits & 0x7FFFFFu;
    int fp8_exp = e + 7;
    if (e < -9) return (uint8_t)(sign << 7);
    if (fp8_exp <= 0) {
        uint32_t full_mant = mant | 0x800000u;
        int shift = 1 - fp8_exp + 20;
        if (shift >= 24) return (uint8_t)(sign << 7);
        uint32_t fp8_mant = (full_mant + (1u << (shift - 1))) >> shift;
        if (fp8_mant > 7) fp8_mant = 7;
        return (uint8_t)((sign << 7) | (fp8_mant & 7u));
    }
    if (fp8_exp >= 15) return (uint8_t)((sign << 7) | (15u << 3) | 6u);
    uint32_t fp8_mant = (mant + (1u << 19)) >> 20;
    if (fp8_mant > 7) { fp8_mant = 0; fp8_exp++; }
    if (fp8_exp >= 15) return (uint8_t)((sign << 7) | (15u << 3) | 6u);
    return (uint8_t)((sign << 7) | ((uint32_t)fp8_exp << 3) | (fp8_mant & 7u));
}

static int cmp_float_asc(const void *a, const void *b) {
    float fa = *(const float *)a, fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

/* ---- Per-block GPU weight struct ---- */

typedef struct {
    void *attn_q_w, *attn_q_b, *attn_k_w, *attn_k_b, *attn_v_w, *attn_v_b;
    void *attn_out_w, *attn_out_b;
    void *attn_add_q_w, *attn_add_q_b, *attn_add_k_w, *attn_add_k_b;
    void *attn_add_v_w, *attn_add_v_b, *attn_add_out_w, *attn_add_out_b;
    void *norm_q_w, *norm_k_w, *norm_added_q_w, *norm_added_k_w;
    void *img_mod_w, *img_mod_b;
    void *img_mlp_fc1_w, *img_mlp_fc1_b, *img_mlp_fc2_w, *img_mlp_fc2_b;
    void *txt_mod_w, *txt_mod_b;
    void *txt_mlp_fc1_w, *txt_mlp_fc1_b, *txt_mlp_fc2_w, *txt_mlp_fc2_b;
} qimg_block_gpu;

/* ---- Per-linear INT4 (Nunchaku/SVDQuant, W4A16) weight descriptor ----
 * Logical (de-swizzled) layout produced offline by tools/nunchaku_convert_logical.py and uploaded verbatim;
 * the dequant kernel unpacks contiguous nibbles, applies the per-(out, input-group) wscale, divides the BF16
 * activation by `smooth`, and adds the rank-128 BF16 low-rank (lora_up · (lora_down · x)) plus bias. */
typedef struct {
    void  *qint4;      /* device uint8 [n_out, n_in/2]  contiguous logical int4 nibble pack */
    float *wscale;     /* device f32   [n_out, n_in/group_size] */
    float *smooth;     /* device f32   [n_in] */
    void  *lora_down;  /* device bf16  [rank, n_in] */
    void  *lora_up;    /* device bf16  [n_out, rank] */
    float *bias;       /* device f32   [n_out] */
    int n_out, n_in, rank, group_size;
} qimg_int4_linear;

/* Upper bound on calibration slots = max blocks (64) * 12 main linears/block. */
#define QIMG_CALIB_MAXSLOT (64 * 12)

/* INT8 W8A8: 14 quantized linears per block (12 main + img_mod.1 + txt_mod.1). */
#define QIMG_I8_PER_BLOCK 14
static const char *const qimg_i8_suffix[QIMG_I8_PER_BLOCK] = {
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2",
    "img_mod.1", "txt_mod.1",
};

/* ---- Runner struct ---- */

struct hip_qimg_runner {
    int device_id;
    int verbose;

    hipModule_t mod;
    hipFunction_t fn_gemm, fn_gemm_fp8, fn_gemm_opt_fp8;
    hipFunction_t fn_gemm_fp8_wmma;  /* gfx12 BF16 act × FP8 wt matrix-core GEMM */
    hipFunction_t fn_gemm_fp8_fp8_pgr2;  /* gfx12 FP8 act × FP8 wt WMMA (hand-written) */
    /* Vendor (hipBLASLt/Tensile) FP8 GEMM, extracted to a .co. ~1.5x our
     * hand-written kernel; used for the FP8xFP8 path when QIMG_FP8_VENDOR is
     * set and the .co loads. Scalar scale only (scaleA=scaleB=1); bias added
     * separately via fn_add_bias. */
    hipModule_t fp8_vendor_mod;
    hipFunction_t fn_fp8_vendor;
    hipFunction_t fn_add_bias;       /* row-major Y[m,n]+=bias[n] */
    void *d_scale_one;               /* device float[4] = {1,1,1,1} for scaleA/B/C/D */
    int use_fp8_vendor;              /* 1 = vendor .co loaded and enabled */
    hipFunction_t fn_quantize_act_perrow;  /* F32 -> FP8 with per-row max-abs scale */
    hipFunction_t fn_quantize_act_scalar;  /* F32 -> FP8 with scalar scale=1, Comfy-style */
    hipFunction_t fn_quantize_act_clamp;   /* F32 -> FP8 with scale=max(1,maxabs/448) */
    hipFunction_t fn_layernorm, fn_silu, fn_gelu, fn_adaln, fn_gated_add;
    hipFunction_t fn_rmsnorm_ph, fn_flash_attn, fn_flash_attn_wmma, fn_flash_attn_wmma_pq, fn_flash_attn_wmma_sp;
    int use_attn_wmma; /* 1 = use BF16 WMMA flash attention (gfx12, head_dim=128) */
    int use_attn_wmma_pq; /* 1 = use the persistent-Q + double-buffered v2 kernel (opt-in QIMG_ATTN_V2) */
    int use_attn_wmma_sp; /* 1 = use the software-pipelined v3 kernel (opt-in QIMG_ATTN_V3) */
    hipFunction_t fn_qkv_perhead_maxabs, fn_q_quant_fp8, fn_kv_quant_repack_fp8, fn_flash_attn_fp8;
    hipFunction_t fn_q_quant_perrow, fn_k_quant_repack_perrow, fn_flash_attn_fp8_perrow;
    int use_attn_fp8;  /* 1 = QIMG_FP8_ATTN=1 enables FP8 WMMA flash attention */
    hipFunction_t fn_rope_2d, fn_rope_1d, fn_bf16_trunc, fn_add;
    hipFunction_t fn_dequant_int4_main, fn_expand_bf16, fn_gemm_int4w;  /* int4 dequant/expand + fused W4A16 GEMM */
    hipFunction_t fn_quant_act_int8, fn_gemm_w8a8, fn_gemm_w8a8_wmma, fn_gemm_w8a8_pgr2;  /* INT8 W8A8 GEMMs */
    hipFunction_t fn_gemm_int4w_g16;  /* simple RTN int4-g16 BF16-act WMMA GEMM (no LoRA/swizzle) */
    hipFunction_t fn_patchify, fn_unpatchify, fn_euler_step, fn_cfg_combine;
    hipFunction_t fn_rmsnorm_weighted, fn_fp8_roundtrip, fn_act_fp8_rt, fn_w_int8_rt, fn_w_int4_rt;
    int act_fp8_rt;  /* QIMG_ACT_FP8_RT=1: per-row fp8/448 act roundtrip pre-GEMM (CUDA repro) */
    int w_int8_rt;   /* QIMG_W_INT8_RT=1: int8 g64 weight roundtrip pre-GEMM (int8 quality check) */
    int w_int4_rt;   /* QIMG_W_INT4_RT=1: simple RTN int4 g16 weight roundtrip (int4 quality check) */
    /* VAE kernels */
    hipFunction_t fn_vae_conv2d, fn_vae_rmsnorm, fn_vae_silu, fn_vae_up2x;
    hipFunction_t fn_vae_conv2d_3x3_wmma, fn_vae_conv2d_1x1_wmma;
    hipFunction_t fn_vae_self_attn;
    hipFunction_t fn_vae_self_attn_qb;
    int use_vae_wmma; /* 1 = use BF16 WMMA conv2d when shapes align */

    /* DiT config */
    int dim, n_heads, head_dim, n_blocks;
    int in_ch, txt_dim, mlp_h;
    int use_fp8;  /* 1 = upload FP8 raw + use LUT GEMM (4x less VRAM) */
    int use_wmma; /* 1 = use BF16 WMMA matrix cores for FP8 GEMMs (gfx12 only) */
    int use_fp8_fp8w; /* 1 = use FP8xFP8 WMMA + act quant when n_tok % 128 == 0 (gfx12) */
    int fast_fp8_matrix_mult; /* 1 = ComfyUI --fast fp8_matrix_mult semantics */
    int prefer_bf16_wmma; /* 1 = QIMG_FP8_WMMA_BF16=1 forces BF16xFP8 even when FP8xFP8 would qualify */

    /* Persistent scratch for FP8xFP8 path. Sized lazily by op_gemm_fp8 to fit
     * (M_pad * max_n_in) bytes for d_act_fp8 and (M_pad * 4) for d_act_scales. */
    void *d_act_fp8;
    float *d_act_scales;
    size_t act_fp8_bytes;
    size_t act_scales_bytes;

    /* Persistent scratch for FP8 flash-attention. Three QKV FP8 buffers sized
     * to fit (max_n_tok × max_dim) bytes plus three per-head scale vectors. */
    void *d_fa_qfp8, *d_fa_kfp8, *d_fa_vfp8;
    float *d_fa_qs, *d_fa_ks, *d_fa_vs;
    size_t fa_qkv_bytes;
    size_t fa_scales_bytes;       /* size of d_fa_vs (per-head, small) */
    size_t fa_perrow_bytes;       /* size of d_fa_qs and d_fa_ks (per-row [n_heads,n_tok]) */

    /* Safetensors context (mmap'd) */
    void *dit_st;
    void *vae_st;

    /* Preloaded blocks on GPU */
    qimg_block_gpu *gpu_blocks;
    int n_preloaded;
    /* Persistent ping-pong buffers for streamed blocks (>= n_preloaded), reused
     * every denoise step instead of malloc/free churn. Async-prefetched on
     * copy_stream and double-buffered to overlap H2D with compute. */
    qimg_block_gpu stream_blk[2];
    int stream_blk_alloc;       /* 1 once both buffers' fields are allocated */
    /* INT4 (Nunchaku/SVDQuant, W4A16 logical layout) path — all blocks resident (no streaming). */
    qimg_int4_linear *int4_linears;  /* [n_blocks * QIMG_INT4_PER_BLOCK] per-block logical-linear descriptors */
    qimg_int4_linear *int4_mod;      /* [n_blocks * 2] img_mod/txt_mod RTN-int4 (rank0, no smooth) */
    float *i4_ldf, *i4_luf, *i4_dt, *i4_dly; size_t i4_dt_cap, i4_dly_cap;  /* persistent lora scratch */
    int use_int4;                    /* 1 when a logical-int4 DiT was loaded */
    /* INT8 SmoothQuant (W8A8) path: int8 weights stream via the fp8 byte path (same 1 B/param);
     * the small per-linear scales stay resident. d_xq_int8/d_x_iscale = per-token quant scratch. */
    int use_int8, use_int8_smooth;
    void **i8_ws, **i8_sm;           /* [n_blocks*QIMG_I8_PER_BLOCK] resident weight_scale[n_out] / smooth_scale[n_in] */
    void *d_xq_int8; float *d_x_iscale; size_t i8_xq_cap, i8_is_cap;
    hipStream_t copy_stream;
    hipEvent_t stream_copy_done[2];
    hipEvent_t stream_use_done[2];
    int use_block_stream_db;    /* 1 = async double-buffered streaming enabled */

    /* Global GPU weights */
    void *d_img_in_w, *d_img_in_b;
    void *d_txt_in_w, *d_txt_in_b;
    void *d_txt_norm_w;
    void *d_t_fc1_w, *d_t_fc1_b;
    void *d_t_fc2_w, *d_t_fc2_b;
    void *d_norm_out_w, *d_norm_out_b;
    void *d_proj_out_w, *d_proj_out_b;
    /* Per-global-weight FP8/F32 tags. Mixed-dtype checkpoints
     * (e.g. unsloth/Qwen-Image-2512-FP8) store some global GEMM weights as
     * BF16 instead of FP8. is_fp8_<name> = 1 if the device pointer is raw
     * FP8 bytes (use op_gemm_fp8); 0 if it's an F32 array (use op_gemm). */
    int is_fp8_img_in, is_fp8_txt_in;
    int is_fp8_t_fc1, is_fp8_t_fc2;
    int is_fp8_norm_out, is_fp8_proj_out;

    /* Optional diagnostics, enabled by env or test harness flags:
     * QIMG_PATH_STATS=1 prints and aggregates GEMM dispatches.
     * QIMG_FP8_QUANT_STATS=1 samples FP8xFP8 activation quantization error. */
    int path_stats_enabled;
    int quant_stats_enabled;
    int quant_stats_max;
    int path_prints;
    int quant_prints;
    uint64_t gemm_path_counts[6];
    uint64_t gemm_path_flops[6];
    uint64_t gemm_path_bytes[6];
    int mem_stats_enabled;
    int mem_report_printed;
    const char *fp8_fp8_allow;
    const char *fp8_fp8_deny;
    int fp8_fp8_block_min;
    int fp8_fp8_block_max;
    float fp8_quality_target_db;
    float fp8_act_scale_div;
    int fp8_act_scale_scalar;
    int fp8_act_scale_clamp;
    int current_block;
    char current_gemm_label[64];

    /* Activation-stats calibration (QIMG_CALIB_DUMP=<path>): per-(block,slot) per-input-channel
     * max-abs of each main linear's input, collected in the bf16/fp8 path to drive offline
     * SVDQuant smoothing. Lazily allocated on first hit; dumped (safetensors of F32 [n_in]) and
     * freed in hip_qimg_free. Slot = block*12 + op_proj slot (0..11). */
    const char *calib_dump_path;            /* non-NULL enables collection */
    hipFunction_t fn_calib_amax;            /* amax_per_col_f32 */
    float *calib_amax[QIMG_CALIB_MAXSLOT];  /* device [n_in] per main linear */
    int    calib_nin[QIMG_CALIB_MAXSLOT];   /* n_in per slot (0 = unallocated) */
    /* Modulation activation absmax: img_mod/txt_mod share the d_t_silu input [dim]; one [dim]
     * buffer per block, dumped as img_mod.1.amax + txt_mod.1.amax (the 990x-outlier layer). */
    float *calib_tsilu[64];                 /* device [dim] per block */
};

enum {
    QIMG_GEMM_F32 = 0,
    QIMG_GEMM_FP8_SCALAR,
    QIMG_GEMM_FP8_OPT,
    QIMG_GEMM_BF16_WMMA,
    QIMG_GEMM_FP8_FP8,
    QIMG_GEMM_PATH_COUNT
};

static const char *qimg_gemm_path_name(int path) {
    switch (path) {
    case QIMG_GEMM_F32:       return "f32";
    case QIMG_GEMM_FP8_SCALAR:return "fp8_scalar";
    case QIMG_GEMM_FP8_OPT:   return "fp8_lut128";
    case QIMG_GEMM_BF16_WMMA: return "bf16xfp8_wmma";
    case QIMG_GEMM_FP8_FP8:   return "fp8xfp8_wmma";
    default:                  return "unknown";
    }
}

static uint64_t qimg_ceil_div_u64(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static uint64_t qimg_estimate_gemm_global_bytes(int path, int n_tok, int n_out, int n_in) {
    uint64_t M = (uint64_t)n_tok, N = (uint64_t)n_out, K = (uint64_t)n_in;
    if (!M || !N || !K) return 0;

    uint64_t mt = qimg_ceil_div_u64(M, 128);
    uint64_t nt = qimg_ceil_div_u64(N, 128);
    uint64_t x_bpe = 4, w_bpe = 4;

    switch (path) {
    case QIMG_GEMM_FP8_SCALAR:
    case QIMG_GEMM_FP8_OPT:
    case QIMG_GEMM_BF16_WMMA:
        w_bpe = 1;
        break;
    case QIMG_GEMM_FP8_FP8:
        x_bpe = 1;
        w_bpe = 1;
        break;
    default:
        break;
    }

    /* GEMM kernel global traffic model:
     *   X is reread for each N tile, W is reread for each M tile, Y is written
     *   once. BF16xFP8 converts FP8 weights to BF16 inside the kernel while
     *   staging LDS, so there is no global BF16 weight copy in this model. */
    uint64_t bytes = 0;
    bytes += nt * M * K * x_bpe;
    bytes += mt * N * K * w_bpe;
    bytes += M * N * 4;              /* F32 output buffer */
    bytes += mt * N * 4;             /* bias reads, approximate */

    if (path == QIMG_GEMM_FP8_FP8) {
        /* Activation quantizer traffic plus per-row scale reloads at writeback. */
        bytes += M * K * 4;          /* read F32 activation */
        bytes += M * K;              /* write FP8 activation scratch */
        bytes += M * 4;              /* write scales */
        bytes += M * N * 4;          /* read row scale, worst-case uncached */
    }
    return bytes;
}

static void qimg_set_gemm_context(hip_qimg_runner *r, int block, const char *label) {
    if (!r) return;
    r->current_block = block;
    if (!label) label = "unnamed";
    snprintf(r->current_gemm_label, sizeof(r->current_gemm_label), "%s", label);
}

static void qimg_record_gemm_path(hip_qimg_runner *r, int path,
                                  int n_tok, int n_out, int n_in) {
    if (!r || path < 0 || path >= QIMG_GEMM_PATH_COUNT) return;
    r->gemm_path_counts[path]++;
    r->gemm_path_flops[path] += 2ULL * (uint64_t)n_tok * (uint64_t)n_out * (uint64_t)n_in;
    r->gemm_path_bytes[path] += qimg_estimate_gemm_global_bytes(path, n_tok, n_out, n_in);
    if (r->path_stats_enabled && r->path_prints < 240) {
        fprintf(stderr, "  gemm[%03d] block=%d %-18s path=%-15s M=%d N=%d K=%d\n",
                r->path_prints, r->current_block, r->current_gemm_label,
                qimg_gemm_path_name(path), n_tok, n_out, n_in);
        r->path_prints++;
    }
}

static int qimg_csv_has_label(const char *csv, const char *label) {
    if (!csv || !csv[0] || !label || !label[0]) return 0;
    const char *p = csv;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        const char *s = p;
        while (*p && *p != ',') p++;
        const char *e = p;
        while (e > s && (e[-1] == ' ' || e[-1] == '\t')) e--;
        size_t n = (size_t)(e - s);
        if ((n == 1 && s[0] == '*') ||
            (strlen(label) == n && strncmp(s, label, n) == 0))
            return 1;
    }
    return 0;
}

static int qimg_has_fp8_fp8_positive_filter(const hip_qimg_runner *r) {
    if (!r) return 0;
    if (r->fp8_fp8_allow && r->fp8_fp8_allow[0]) return 1;
    if (r->fp8_fp8_block_min >= 0 || r->fp8_fp8_block_max >= 0) return 1;
    return 0;
}

static int qimg_allow_fp8_fp8_for_current_label(hip_qimg_runner *r) {
    if (!r) return 0;
    if (!r->fast_fp8_matrix_mult &&
        r->fp8_quality_target_db > 0.0f && !qimg_has_fp8_fp8_positive_filter(r))
        return 0;
    if (r->current_block >= 0) {
        if (r->fp8_fp8_block_min >= 0 && r->current_block < r->fp8_fp8_block_min)
            return 0;
        if (r->fp8_fp8_block_max >= 0 && r->current_block > r->fp8_fp8_block_max)
            return 0;
    } else if (r->fp8_fp8_block_min >= 0 || r->fp8_fp8_block_max >= 0) {
        return 0;
    }
    if (qimg_csv_has_label(r->fp8_fp8_deny, r->current_gemm_label))
        return 0;
    if (r->fp8_fp8_allow && r->fp8_fp8_allow[0])
        return qimg_csv_has_label(r->fp8_fp8_allow, r->current_gemm_label);
    return 1;
}

static void qimg_print_gemm_summary(hip_qimg_runner *r) {
    if (!r || !r->path_stats_enabled) return;
    fprintf(stderr, "hip_qimg: GEMM path summary:");
    for (int i = 0; i < QIMG_GEMM_PATH_COUNT; i++)
        fprintf(stderr, " %s=%llu", qimg_gemm_path_name(i),
                (unsigned long long)r->gemm_path_counts[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "hip_qimg: GEMM traffic summary:");
    double total_bytes = 0.0, total_flops = 0.0;
    for (int i = 0; i < QIMG_GEMM_PATH_COUNT; i++) {
        total_bytes += (double)r->gemm_path_bytes[i];
        total_flops += (double)r->gemm_path_flops[i];
        if (r->gemm_path_counts[i]) {
            fprintf(stderr, " %s=%.2fGB/%.2fTF",
                    qimg_gemm_path_name(i),
                    (double)r->gemm_path_bytes[i] / 1.0e9,
                    (double)r->gemm_path_flops[i] / 1.0e12);
        }
    }
    fprintf(stderr, " total=%.2fGB/%.2fTF\n", total_bytes / 1.0e9, total_flops / 1.0e12);
    if (r->act_fp8_bytes || r->act_scales_bytes || r->fa_qkv_bytes || r->fa_perrow_bytes || r->fa_scales_bytes) {
        fprintf(stderr,
                "hip_qimg: persistent scratch: act_fp8=%.1fMB act_scales=%.1fMB "
                "fa_qkv=%.1fMB fa_perrow=%.1fMB fa_scales=%.3fMB\n",
                (double)r->act_fp8_bytes / (1024.0 * 1024.0),
                (double)r->act_scales_bytes / (1024.0 * 1024.0),
                (double)(3 * r->fa_qkv_bytes) / (1024.0 * 1024.0),
                (double)(2 * r->fa_perrow_bytes) / (1024.0 * 1024.0),
                (double)r->fa_scales_bytes / (1024.0 * 1024.0));
    }
}

static void qimg_maybe_quant_stats(hip_qimg_runner *r, void *X,
                                   int n_tok, int K, int M_pad) {
    if (!r || !r->quant_stats_enabled || r->quant_prints >= r->quant_stats_max)
        return;
    size_t n = (size_t)n_tok * (size_t)K;
    if (n == 0) return;
    float *h = (float *)malloc(n * sizeof(float));
    if (!h) return;
    hipMemcpy(h, X, n * sizeof(float), hipMemcpyDeviceToHost);
    init_fp8_to_f32_lut();

    double sum_row_max = 0.0, sum_mae = 0.0;
    float global_max = 0.0f, worst_row_max = 0.0f;
    int sat = 0, zeros = 0;
    const int nsamp_max = 8192;
    float *samples = (float *)malloc((size_t)nsamp_max * sizeof(float));
    int nsamp = 0;
    size_t stride = n / (size_t)nsamp_max;
    if (stride < 1) stride = 1;

    for (int m = 0; m < n_tok; m++) {
        const float *row = h + (size_t)m * K;
        float row_max = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = fabsf(row[k]);
            if (a > row_max) row_max = a;
        }
        if (row_max > global_max) global_max = row_max;
        if (row_max > worst_row_max) worst_row_max = row_max;
        sum_row_max += row_max;
        float s = r->fp8_act_scale_scalar ? 1.0f :
                  (r->fp8_act_scale_clamp ? (row_max > 448.0f ? row_max / 448.0f : 1.0f)
                                           : (row_max / r->fp8_act_scale_div));
        if (s < 1e-12f) s = 1e-12f;
        float inv_s = 1.0f / s;
        for (int k = 0; k < K; k++) {
            float scaled = row[k] * inv_s;
            if (fabsf(scaled) >= 448.0f) sat++;
            uint8_t q = f32_to_fp8_e4m3_cpu(scaled);
            if ((q & 0x7fu) == 0) zeros++;
            float recon = fp8_e4m3_to_f32(q) * s;
            float err = fabsf(recon - row[k]);
            sum_mae += err;
            size_t flat = (size_t)m * K + (size_t)k;
            if ((flat % stride) == 0 && nsamp < nsamp_max)
                samples[nsamp++] = err;
        }
    }
    float p99 = 0.0f;
    if (samples && nsamp > 0) {
        qsort(samples, (size_t)nsamp, sizeof(float), cmp_float_asc);
        int idx = (int)(0.99f * (float)(nsamp - 1));
        p99 = samples[idx];
    }

    /* Block-wise (per-row, per-K-group) and per-tensor reconstruction MAE,
     * to test whether finer activation-scale granularity reduces e4m3 quant
     * error vs the shipped per-row scale. scale_div / clamp follow the active
     * mode so the three numbers are directly comparable. Set the K-group with
     * QIMG_FP8_QUANT_STATS_GBLK (default 128). */
    int gblk = 128;
    {
        const char *gv = getenv("QIMG_FP8_QUANT_STATS_GBLK");
        if (gv && atoi(gv) > 0) gblk = atoi(gv);
    }
    double sum_mae_blk = 0.0, sum_mae_tensor = 0.0;
    {
        float st = r->fp8_act_scale_scalar ? 1.0f :
                   (r->fp8_act_scale_clamp ? (global_max > 448.0f ? global_max / 448.0f : 1.0f)
                                            : (global_max / r->fp8_act_scale_div));
        if (st < 1e-12f) st = 1e-12f;
        float inv_st = 1.0f / st;
        for (int m = 0; m < n_tok; m++) {
            const float *row = h + (size_t)m * K;
            for (int kb = 0; kb < K; kb += gblk) {
                int ke = kb + gblk; if (ke > K) ke = K;
                float gmax = 0.0f;
                for (int k = kb; k < ke; k++) { float a = fabsf(row[k]); if (a > gmax) gmax = a; }
                float sb = r->fp8_act_scale_scalar ? 1.0f :
                           (r->fp8_act_scale_clamp ? (gmax > 448.0f ? gmax / 448.0f : 1.0f)
                                                    : (gmax / r->fp8_act_scale_div));
                if (sb < 1e-12f) sb = 1e-12f;
                float inv_sb = 1.0f / sb;
                for (int k = kb; k < ke; k++) {
                    float qb = fp8_e4m3_to_f32(f32_to_fp8_e4m3_cpu(row[k] * inv_sb)) * sb;
                    sum_mae_blk += fabsf(qb - row[k]);
                    float qt = fp8_e4m3_to_f32(f32_to_fp8_e4m3_cpu(row[k] * inv_st)) * st;
                    sum_mae_tensor += fabsf(qt - row[k]);
                }
            }
        }
    }

    fprintf(stderr,
            "  fp8q[%03d] block=%d %-18s M=%d/%d K=%d row_max_avg=%.5g max=%.5g "
            "scale_mode=%s scale_div=%.1f mae=%.5g mae_blk%d=%.5g mae_tensor=%.5g "
            "p99=%.5g zeros=%.2f%% sat=%.2f%%\n",
            r->quant_prints, r->current_block, r->current_gemm_label,
            n_tok, M_pad, K, sum_row_max / (double)n_tok, global_max,
            r->fp8_act_scale_scalar ? "comfy" : (r->fp8_act_scale_clamp ? "clamp" : "perrow"),
            r->fp8_act_scale_div,
            sum_mae / (double)n, gblk, sum_mae_blk / (double)n,
            sum_mae_tensor / (double)n, p99, 100.0 * (double)zeros / (double)n,
            100.0 * (double)sat / (double)n);
    r->quant_prints++;
    free(samples);
    free(h);
}


/* ---- Upload helpers ---- */

/* Upload safetensor as F32 to GPU (handles FP8 E4M3, BF16, F16, F32 inputs) */
static void *qimg_st_upload_f32(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= shape[d];
    const uint8_t *src = (const uint8_t *)safetensors_data(st, idx);
    const char *dtype = safetensors_dtype(st, idx);

    float *f32 = (float *)malloc(n * sizeof(float));
    if (!f32) return NULL;

    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        init_fp8_to_f32_lut();
        for (size_t i = 0; i < n; i++)
            f32[i] = hip_fp8_to_f32_lut[src[i]];
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
        fprintf(stderr, "hip_qimg: unsupported dtype '%s' for %s\n", dtype, name);
        free(f32);
        return NULL;
    }

    void *d = NULL;
    if (hipMalloc(&d, n * sizeof(float)) != hipSuccess) {
        fprintf(stderr, "hip_qimg: hipMalloc(%.1f MB) FAILED for %s\n",
                (float)(n * 4) / (1 << 20), name);
        free(f32);
        return NULL;
    }
    hipMemcpy(d, f32, n * sizeof(float), hipMemcpyHostToDevice);
    free(f32);
    return d;
}

/* Upload safetensor as raw FP8 bytes to GPU (no conversion, 1 byte/element).
 * Mixed-dtype checkpoints call this only for tensors that are actually FP8;
 * BF16/F16/F32 global weights go through qimg_upload_weight_auto(). */
static void *qimg_st_upload_fp8_raw(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const char *dtype = safetensors_dtype(st, idx);
    if (strcmp(dtype, "F8_E4M3") != 0 && strcmp(dtype, "F8_E4M3FN") != 0) {
        fprintf(stderr, "hip_qimg: %s has dtype '%s' (expected F8_E4M3) — "
                "caller should route non-FP8 tensors through qimg_upload_weight_auto().\n",
                name, dtype);
        return NULL;
    }
    size_t nbytes = safetensors_nbytes(st, idx);
    const void *data = safetensors_data(st, idx);
    void *d = NULL;
    if (hipMalloc(&d, nbytes) != hipSuccess) {
        fprintf(stderr, "hip_qimg: hipMalloc(%.1f MB) FAILED for %s (fp8)\n",
                (float)nbytes / (1 << 20), name);
        return NULL;
    }
    hipMemcpy(d, data, nbytes, hipMemcpyHostToDevice);
    return d;
}

/* Upload weight: FP8 raw if use_fp8, else F32 */
/* Upload a tensor's raw bytes to device verbatim (dtype-agnostic) — for the logical INT4 bundle's packed
 * nibbles (uint8) and the BF16 low-rank factors, which the dequant kernel reinterprets. */
static void *qimg_st_upload_raw(st_context *st, const char *name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    size_t nbytes = safetensors_nbytes(st, idx);
    const void *data = safetensors_data(st, idx);
    void *d = NULL;
    if (hipMalloc(&d, nbytes) != hipSuccess) {
        fprintf(stderr, "hip_qimg: hipMalloc(%.1f MB) FAILED for %s (raw)\n", (float)nbytes / (1 << 20), name);
        return NULL;
    }
    hipMemcpy(d, data, nbytes, hipMemcpyHostToDevice);
    return d;
}

/* Upload one logical INT4 linear bundle (<key>.qint4/.wscale/.smooth/.lora_down/.lora_up/.bias) into a
 * descriptor; dims inferred from shapes (qint4 [n_out, n_in/2]; lora_up [n_out, rank]). Returns 0 on success.
 * Used by the Nunchaku-format DiT loader — all blocks preloaded (the INT4 model fits VRAM; no streaming). */
static int qimg_upload_int4_linear(st_context *st, const char *key, qimg_int4_linear *L) {
    char nm[256]; int idx;
    memset(L, 0, sizeof(*L)); L->group_size = 64;
    snprintf(nm, sizeof(nm), "%s.qint4", key);
    idx = safetensors_find(st, nm); if (idx < 0) return -1;
    const uint64_t *qs = safetensors_shape(st, idx);          /* [n_out, n_in/2] */
    L->n_out = (int)qs[0]; L->n_in = (int)(qs[1] * 2);
    /* Group size from wscale columns: Nunchaku g64 -> n_in/64 cols; simple g16 -> n_in/16. */
    snprintf(nm, sizeof(nm), "%s.wscale", key);
    { int widx = safetensors_find(st, nm);
      if (widx >= 0) { const uint64_t *ws = safetensors_shape(st, widx);
        int wcols = (int)ws[safetensors_ndims(st, widx) - 1];
        if (wcols > 0) L->group_size = L->n_in / wcols; } }
    snprintf(nm, sizeof(nm), "%s.lora_up", key);
    idx = safetensors_find(st, nm); if (idx >= 0) L->rank = (int)safetensors_shape(st, idx)[1];
    snprintf(nm, sizeof(nm), "%s.qint4", key);     L->qint4     = qimg_st_upload_raw(st, nm);
    snprintf(nm, sizeof(nm), "%s.wscale", key);    L->wscale    = (float *)qimg_st_upload_raw(st, nm);   /* bf16, kernel expands */
    snprintf(nm, sizeof(nm), "%s.smooth", key);    L->smooth    = (float *)qimg_st_upload_f32(st, nm);
    snprintf(nm, sizeof(nm), "%s.lora_down", key); L->lora_down = qimg_st_upload_raw(st, nm);
    snprintf(nm, sizeof(nm), "%s.lora_up", key);   L->lora_up   = qimg_st_upload_raw(st, nm);
    snprintf(nm, sizeof(nm), "%s.bias", key);      L->bias      = (float *)qimg_st_upload_f32(st, nm);
    if (!L->qint4 || !L->wscale || !L->bias) return -1;        /* mod: no smooth/lora (rank stays 0) */
    return 0;
}

static void *qimg_upload_weight(hip_qimg_runner *r, st_context *st, const char *name) {
    if (r->use_int8) return qimg_st_upload_raw(st, name);   /* int8 bytes (1 B/param, like fp8) */
    if (r->use_fp8) return qimg_st_upload_fp8_raw(st, name);
    return qimg_st_upload_f32(st, name);
}

/* Copy a tensor into an already-allocated device buffer `dest` (same shape as a
 * prior upload of an identically-shaped tensor). Used to re-stream the same set
 * of blocks every denoise step without re-malloc/free. `stream` may be a copy
 * stream for async overlap (host F32-convert temp is synced before reuse).
 * Returns 0 on success, -1 on failure. */
static int qimg_st_upload_fp8_raw_into(st_context *st, const char *name,
                                       void *dest, hipStream_t stream) {
    int idx = safetensors_find(st, name);
    if (idx < 0 || !dest) return -1;
    size_t nbytes = safetensors_nbytes(st, idx);
    const void *data = safetensors_data(st, idx);
    return hipMemcpyAsync(dest, data, nbytes, hipMemcpyHostToDevice, stream) == hipSuccess ? 0 : -1;
}

static int qimg_st_upload_f32_into(st_context *st, const char *name, void *dest) {
    /* Bias/norm tensors: small, host dtype-convert then a synchronous copy.
     * (Kept synchronous — the temp would otherwise need to outlive an async copy.) */
    void *tmp = qimg_st_upload_f32(st, name);  /* mallocs+copies a fresh device buffer */
    if (!tmp || !dest) { if (tmp) hipFree(tmp); return -1; }
    int idx = safetensors_find(st, name);
    const uint64_t *shape = safetensors_shape(st, idx);
    int ndims = safetensors_ndims(st, idx);
    size_t n = 1; for (int d = 0; d < ndims; d++) n *= shape[d];
    int rc = hipMemcpy(dest, tmp, n * sizeof(float), hipMemcpyDeviceToDevice) == hipSuccess ? 0 : -1;
    hipFree(tmp);
    return rc;
}

static int qimg_upload_weight_into(hip_qimg_runner *r, st_context *st,
                                   const char *name, void *dest, hipStream_t stream) {
    /* fp8_raw_into is a dtype-agnostic raw byte copy -> reuse it for int8 bytes too. */
    if (r->use_int8 || r->use_fp8) return qimg_st_upload_fp8_raw_into(st, name, dest, stream);
    return qimg_st_upload_f32_into(st, name, dest);
}

/* Upload a weight whose source dtype may be FP8 or BF16/F16/F32 (mixed-dtype
 * checkpoints like unsloth/Qwen-Image-2512-FP8). Returns a device pointer and
 * sets *out_is_fp8 to 1 when the data is FP8 raw bytes (caller dispatches
 * through the FP8 GEMM path) or 0 when it's an F32 array (caller uses the
 * F32 GEMM path). Returns NULL on failure. */
static void *qimg_upload_weight_auto(hip_qimg_runner *r, st_context *st,
                                      const char *name, int *out_is_fp8) {
    *out_is_fp8 = 0;
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
    const char *dtype = safetensors_dtype(st, idx);
    if (r->use_fp8 && (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0)) {
        *out_is_fp8 = 1;
        return qimg_st_upload_fp8_raw(st, name);
    }
    /* BF16/F16/F32 → F32 dequant on host (qimg_st_upload_f32 handles all dtypes) */
    return qimg_st_upload_f32(st, name);
}

/* Upload 3D conv weight → 2D by taking last temporal slice */
static void *qimg_upload_conv3d(st_context *st, const char *name,
                                 int *out_co, int *out_ci) {
    int idx = safetensors_find(st, name);
    if (idx < 0) return NULL;
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
    void *dp = NULL;
    hipMalloc(&dp, n2d * sizeof(float));
    hipMemcpy(dp, w2d, n2d * sizeof(float), hipMemcpyHostToDevice);
    free(w2d);
    return dp;
}

/* ---- Load/free one DiT block ---- */

static void qimg_free_block(qimg_block_gpu *b) {
    void **ptrs = (void **)b;
    int n = sizeof(qimg_block_gpu) / sizeof(void *);
    for (int i = 0; i < n; i++) {
        if (ptrs[i]) { hipFree(ptrs[i]); ptrs[i] = NULL; }
    }
}

/* Load all weights of a transformer block into `b`. If `b`'s fields are already
 * allocated (reuse=streaming buffer), copy into them on `stream` (no malloc/free);
 * otherwise malloc fresh (preload path). `stream` is only used for the reuse path. */
static int qimg_load_block_s(hip_qimg_runner *r, int block_idx, qimg_block_gpu *b,
                             hipStream_t stream) {
    st_context *st = (st_context *)r->dit_st;
    char name[256];
    int ok = 1;

    /* BLK_W: upload as FP8 raw or F32 depending on use_fp8 */
    #define BLK_W(field, suffix) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        if (b->field) { if (qimg_upload_weight_into(r, st, name, b->field, stream) != 0) ok = 0; } \
        else { b->field = qimg_upload_weight(r, st, name); if (!b->field) ok = 0; } \
    } } while(0)
    /* BLK_F32: always upload as F32 (biases, norms) */
    #define BLK_F32(field, suffix) do { if (ok) { \
        snprintf(name, sizeof(name), "transformer_blocks.%d." suffix, block_idx); \
        if (b->field) { if (qimg_st_upload_f32_into(st, name, b->field) != 0) ok = 0; } \
        else { b->field = qimg_st_upload_f32(st, name); if (!b->field) ok = 0; } \
    } } while(0)

    BLK_W(attn_q_w, "attn.to_q.weight"); BLK_F32(attn_q_b, "attn.to_q.bias");
    BLK_W(attn_k_w, "attn.to_k.weight"); BLK_F32(attn_k_b, "attn.to_k.bias");
    BLK_W(attn_v_w, "attn.to_v.weight"); BLK_F32(attn_v_b, "attn.to_v.bias");
    BLK_W(attn_out_w, "attn.to_out.0.weight"); BLK_F32(attn_out_b, "attn.to_out.0.bias");

    BLK_W(attn_add_q_w, "attn.add_q_proj.weight"); BLK_F32(attn_add_q_b, "attn.add_q_proj.bias");
    BLK_W(attn_add_k_w, "attn.add_k_proj.weight"); BLK_F32(attn_add_k_b, "attn.add_k_proj.bias");
    BLK_W(attn_add_v_w, "attn.add_v_proj.weight"); BLK_F32(attn_add_v_b, "attn.add_v_proj.bias");
    BLK_W(attn_add_out_w, "attn.to_add_out.weight"); BLK_F32(attn_add_out_b, "attn.to_add_out.bias");

    BLK_F32(norm_q_w, "attn.norm_q.weight");
    BLK_F32(norm_k_w, "attn.norm_k.weight");
    BLK_F32(norm_added_q_w, "attn.norm_added_q.weight");
    BLK_F32(norm_added_k_w, "attn.norm_added_k.weight");

    BLK_W(img_mod_w, "img_mod.1.weight"); BLK_F32(img_mod_b, "img_mod.1.bias");
    BLK_W(img_mlp_fc1_w, "img_mlp.net.0.proj.weight"); BLK_F32(img_mlp_fc1_b, "img_mlp.net.0.proj.bias");
    BLK_W(img_mlp_fc2_w, "img_mlp.net.2.weight"); BLK_F32(img_mlp_fc2_b, "img_mlp.net.2.bias");

    BLK_W(txt_mod_w, "txt_mod.1.weight"); BLK_F32(txt_mod_b, "txt_mod.1.bias");
    BLK_W(txt_mlp_fc1_w, "txt_mlp.net.0.proj.weight"); BLK_F32(txt_mlp_fc1_b, "txt_mlp.net.0.proj.bias");
    BLK_W(txt_mlp_fc2_w, "txt_mlp.net.2.weight"); BLK_F32(txt_mlp_fc2_b, "txt_mlp.net.2.bias");

    #undef BLK_W
    #undef BLK_F32

    if (!ok) {
        qimg_free_block(b);
        return -1;
    }
    return 0;
}

/* Preload path: fresh malloc, default stream. */
static int qimg_load_block(hip_qimg_runner *r, int block_idx, qimg_block_gpu *b) {
    return qimg_load_block_s(r, block_idx, b, 0);
}


/* ---- Kernel launch helpers ---- */

static void op_gemm(hip_qimg_runner *r, void *Y, void *W, void *X, void *bias,
                    int n_out, int n_in, int n_tok) {
    qimg_record_gemm_path(r, QIMG_GEMM_F32, n_tok, n_out, n_in);
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipModuleLaunchKernel(r->fn_gemm, gx, gy, 1, 16, 16, 1,
                          0, NULL, args, NULL);
}

/* Lazily (re)allocate the FP8 activation scratch + per-row scales to fit
 * (M_pad × n_in) bytes / (M_pad × 4) bytes. Grows but never shrinks. */
static int qimg_ensure_act_scratch(hip_qimg_runner *r, size_t M_pad, size_t K) {
    size_t need_fp8 = M_pad * K;
    size_t need_sc = M_pad * sizeof(float);
    if (need_fp8 > r->act_fp8_bytes) {
        if (r->d_act_fp8) hipFree(r->d_act_fp8);
        if (hipMalloc(&r->d_act_fp8, need_fp8) != hipSuccess) {
            fprintf(stderr, "hip_qimg: hipMalloc(%.1f MB) FAILED for FP8 act scratch\n",
                    (float)need_fp8 / (1 << 20));
            r->d_act_fp8 = NULL; r->act_fp8_bytes = 0;
            return -1;
        }
        r->act_fp8_bytes = need_fp8;
        if (r->mem_stats_enabled)
            fprintf(stderr, "hip_qimg: FP8xFP8 act scratch grew to %.1f MB (M_pad=%zu K=%zu)\n",
                    (double)need_fp8 / (1024.0 * 1024.0), M_pad, K);
    }
    if (need_sc > r->act_scales_bytes) {
        if (r->d_act_scales) hipFree(r->d_act_scales);
        if (hipMalloc((void **)&r->d_act_scales, need_sc) != hipSuccess) {
            fprintf(stderr, "hip_qimg: hipMalloc FAILED for FP8 act scales\n");
            r->d_act_scales = NULL; r->act_scales_bytes = 0;
            return -1;
        }
        r->act_scales_bytes = need_sc;
        if (r->mem_stats_enabled)
            fprintf(stderr, "hip_qimg: FP8xFP8 scale scratch grew to %.3f MB\n",
                    (double)need_sc / (1024.0 * 1024.0));
    }
    return 0;
}

/* FP8 weight GEMM: W is raw FP8 bytes, dequanted via GPU LUT.
 * Dispatch order:
 *   1. FP8×FP8 WMMA + activation FP8 quant (gfx12, n_tok % 128 == 0,
 *      n_out % 128 == 0, K % 32 == 0) — only for ComfyUI-compatible
 *      --fast fp8_matrix_mult mode.
 *   2. BF16×FP8 WMMA matrix-core (gfx12, n_tok ≥ 16) — default FP8
 *      checkpoint compute path when the kernel is present.
 *   3. 128×128 tiled scalar LUT (n_tok ≥ 16) — fallback path
 *   4. 16×16 scalar LUT — fallback for small token batches
 */
/* Launch the extracted vendor (Tensile) FP8 GEMM: D[M,N] = A[N,K] (op=T) @
 * B[M,K] (op=N), scalar scaleA/B (=1). Mirrors rdna4/fp8/bench_fp8_extracted.
 * A operand = weights W[n_out,n_in], B operand = activations X_fp8[n_tok,n_in].
 * Bias is NOT applied here (null slot); caller adds it separately. */
#ifndef HIP_LAUNCH_PARAM_BUFFER_POINTER
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void *)0x02)
#define HIP_LAUNCH_PARAM_END            ((void *)0x03)
#endif
static int qimg_launch_fp8_vendor(hip_qimg_runner *r, void *Y, void *W,
                                  void *X_fp8, void *bias, int n_out, int n_in, int n_tok) {
    /* 172-byte kernarg layout (offsets from rdna4/fp8/bench_fp8_extracted). */
    unsigned char ka[176];
    memset(ka, 0, sizeof(ka));
    #define KAU32(off, v) do { uint32_t t_ = (uint32_t)(v); memcpy(ka + (off), &t_, 4); } while (0)
    #define KAPTR(off, p)  do { void *t_ = (void *)(p);     memcpy(ka + (off), &t_, sizeof(void *)); } while (0)
    #define KAF32(off, v) do { float t_ = (float)(v);       memcpy(ka + (off), &t_, 4); } while (0)
    KAU32(0,  1u);            /* GEMM_INFO   */
    KAU32(4,  0x02200001u);   /* INTERNAL0   */
    KAU32(8,  0x08010008u);   /* INTERNAL1   */
    KAU32(12, (unsigned)(n_out / 128));  /* NUMWG */
    KAU32(16, (unsigned)n_out);          /* SIZES_FREE0 = N */
    KAU32(20, (unsigned)n_tok);          /* SIZES_FREE1 = M */
    KAU32(24, 1u);                       /* SIZES_FREE2 = batch */
    KAU32(28, (unsigned)n_in);           /* SIZES_SUM0  = K */
    KAPTR(32, Y);     /* D */
    KAPTR(40, Y);     /* C */
    KAPTR(48, W);     /* A = weights [N,K] */
    KAPTR(56, X_fp8); /* B = activations [M,K] */
    KAU32(64, (unsigned)n_out); /* stride D0 = N */
    KAU32(72, (unsigned)n_out); /* stride C0 = N */
    KAU32(80, (unsigned)n_in);  /* stride A0 = K */
    KAU32(88, (unsigned)n_in);  /* stride B0 = K */
    KAF32(96, 1.0f);  /* alpha */
    KAF32(100, 0.0f); /* beta  */
    KAPTR(104, r->d_scale_one); /* scaleA */
    KAPTR(112, r->d_scale_one); /* scaleB */
    KAPTR(120, r->d_scale_one); /* scaleC */
    KAPTR(128, r->d_scale_one); /* scaleD */
    /* Fuse the per-column F32 bias via the kernel's bias slot (sym has _Bias_).
     * BIAS_TYPE 0 = F32, STRIDE_BIAS 0 = broadcast over rows. scaleAV null. */
    if (bias) {
        KAPTR(144, bias);
        KAU32(152, 0u);  /* BIAS_TYPE = F32 */
        KAU32(156, 0u);  /* STRIDE_BIAS */
    }
    #undef KAU32
    #undef KAPTR
    #undef KAF32
    size_t arg_size = 172;
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, ka,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
                      HIP_LAUNCH_PARAM_END};
    hipError_t e = hipModuleLaunchKernel(r->fn_fp8_vendor,
                                         (unsigned)(n_out / 128), (unsigned)(n_tok / 128), 1,
                                         128, 1, 1, 0, NULL, NULL, (void **)config);
    return e == hipSuccess ? 0 : -1;
}

static void op_gemm_fp8(hip_qimg_runner *r, void *Y, void *W, void *X, void *bias,
                        int n_out, int n_in, int n_tok) {
    hipFunction_t quant_fn = r->fp8_act_scale_clamp ? r->fn_quantize_act_clamp :
                             (r->fp8_act_scale_scalar ? r->fn_quantize_act_scalar : r->fn_quantize_act_perrow);
    if (r->use_fp8_fp8w && !r->prefer_bf16_wmma && r->fn_gemm_fp8_fp8_pgr2 &&
        quant_fn && n_tok > 0 &&
        qimg_allow_fp8_fp8_for_current_label(r) &&
        (n_tok % 128) == 0 && (n_out % 128) == 0 && (n_in % 32) == 0) {
        int M_pad = n_tok;  /* already 128-aligned per gate */
        if (qimg_ensure_act_scratch(r, (size_t)M_pad, (size_t)n_in) == 0) {
            void *d_x_fp8 = r->d_act_fp8;
            float *d_scales = r->d_act_scales;
            qimg_record_gemm_path(r, QIMG_GEMM_FP8_FP8, n_tok, n_out, n_in);
            qimg_maybe_quant_stats(r, X, n_tok, n_in, M_pad);
            /* Quantize: F32 X[n_tok, n_in] -> FP8 + row writeback scale. */
            int K = n_in;
            if (r->use_fp8_vendor && r->fn_fp8_vendor) {
                /* Vendor (Tensile) FP8 GEMM: scalar scale=1 quantize, then GEMM,
                 * then a separate per-column bias add. The vendor kernel uses a
                 * scalar scaleA/B, so the activation must be scale=1; per-tensor
                 * == per-row quant error for these activations (see README), so
                 * this is quality-neutral vs the hand-written per-row path. */
                void *qa[] = {&X, &d_x_fp8, &d_scales, &n_tok, &K, &M_pad};
                hipModuleLaunchKernel(r->fn_quantize_act_scalar,
                                      (unsigned)M_pad, 1, 1, 256, 1, 1, 0, NULL, qa, NULL);
                if (qimg_launch_fp8_vendor(r, Y, W, d_x_fp8, bias, n_out, n_in, n_tok) == 0)
                    return;
                /* vendor launch failed -> fall through to hand-written kernel. */
            }
            if (r->fp8_act_scale_scalar || r->fp8_act_scale_clamp) {
                void *qa[] = {&X, &d_x_fp8, &d_scales, &n_tok, &K, &M_pad};
                hipModuleLaunchKernel(quant_fn,
                                      (unsigned)M_pad, 1, 1, 256, 1, 1,
                                      0, NULL, qa, NULL);
            } else {
                float scale_div = r->fp8_act_scale_div;
                void *qa[] = {&X, &d_x_fp8, &d_scales, &n_tok, &K, &M_pad, &scale_div};
                hipModuleLaunchKernel(r->fn_quantize_act_perrow,
                                      (unsigned)M_pad, 1, 1, 256, 1, 1,
                                      0, NULL, qa, NULL);
            }
            /* GEMM: FP8 act × FP8 wt -> F32, scaled writeback. */
            void *ga[] = {&Y, &W, &d_x_fp8, &bias, &d_scales,
                          &n_out, &n_in, &n_tok, &M_pad};
            unsigned gx = (unsigned)(n_out / 128);
            unsigned gy = (unsigned)(M_pad / 128);
            hipModuleLaunchKernel(r->fn_gemm_fp8_fp8_pgr2, gx, gy, 1, 128, 1, 1,
                                  0, NULL, ga, NULL);
            return;
        }
        /* Alloc failure: fall through to BF16xFP8 / scalar paths. */
    }
    if (r->use_wmma && r->fn_gemm_fp8_wmma && n_tok >= 16) {
        qimg_record_gemm_path(r, QIMG_GEMM_BF16_WMMA, n_tok, n_out, n_in);
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        unsigned gx = (unsigned)((n_out + 127) / 128);
        unsigned gy = (unsigned)((n_tok + 127) / 128);
        hipModuleLaunchKernel(r->fn_gemm_fp8_wmma, gx, gy, 1, 256, 1, 1,
                              0, NULL, args, NULL);
        return;
    }
    if (r->fn_gemm_opt_fp8 && n_tok >= 16) {
        qimg_record_gemm_path(r, QIMG_GEMM_FP8_OPT, n_tok, n_out, n_in);
        /* gemm_opt_fp8: Y[M,N] = X[M,K] × W[N,K]^T  (N=n_out, K=n_in, M=n_tok) */
        void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
        unsigned gx = (unsigned)((n_out + 127) / 128);
        unsigned gy = (unsigned)((n_tok + 127) / 128);
        hipModuleLaunchKernel(r->fn_gemm_opt_fp8, gx, gy, 1, 256, 1, 1,
                              0, NULL, args, NULL);
        return;  /* BF16 truncation fused into kernel */
    }
    qimg_record_gemm_path(r, QIMG_GEMM_FP8_SCALAR, n_tok, n_out, n_in);
    void *args[] = {&Y, &W, &X, &bias, &n_out, &n_in, &n_tok};
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    hipModuleLaunchKernel(r->fn_gemm_fp8, gx, gy, 1, 16, 16, 1,
                          0, NULL, args, NULL);
}

/* Weight GEMM: dispatch FP8 or F32 based on runner config */
static void op_wgemm(hip_qimg_runner *r, void *Y, void *W, void *X, void *bias,
                     int n_out, int n_in, int n_tok) {
    if (r->use_fp8) op_gemm_fp8(r, Y, W, X, bias, n_out, n_in, n_tok);
    else            op_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
}

static void op_bf16_trunc(hip_qimg_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_bf16_trunc, (unsigned)((n+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

/* Weight GEMM + BF16 truncation */
static void op_wgemm_bf16(hip_qimg_runner *r, void *Y, void *W, void *X, void *bias,
                          int n_out, int n_in, int n_tok) {
    if (r->act_fp8_rt && r->fn_act_fp8_rt) {  /* CUDA-repro: lossy fp8/448 acts, accurate GEMM */
        void *a[] = {&X, &n_in}; hipModuleLaunchKernel(r->fn_act_fp8_rt, (unsigned)n_tok, 1, 1, 256, 1, 1, 0, NULL, a, NULL);
    }
    if (r->w_int8_rt && r->fn_w_int8_rt && !r->use_fp8) {  /* int8 g64 weight precision check (bf16 src) */
        void *a[] = {&W, &n_in}; hipModuleLaunchKernel(r->fn_w_int8_rt, (unsigned)n_out, 1, 1, 1, 1, 1, 0, NULL, a, NULL);
    }
    if (r->w_int4_rt && r->fn_w_int4_rt && !r->use_fp8) {  /* simple RTN int4 g16 weight precision check */
        void *a[] = {&W, &n_in}; hipModuleLaunchKernel(r->fn_w_int4_rt, (unsigned)n_out, 1, 1, 1, 1, 1, 0, NULL, a, NULL);
    }
    hipFunction_t quant_fn = r->fp8_act_scale_clamp ? r->fn_quantize_act_clamp :
                             (r->fp8_act_scale_scalar ? r->fn_quantize_act_scalar : r->fn_quantize_act_perrow);
    int fp8_fp8_eligible = (r->use_fp8_fp8w && !r->prefer_bf16_wmma && r->fn_gemm_fp8_fp8_pgr2 &&
                             quant_fn && qimg_allow_fp8_fp8_for_current_label(r) &&
                             (n_tok % 128) == 0 && (n_out % 128) == 0 && (n_in % 32) == 0);
    if (r->use_fp8 && (r->fn_gemm_opt_fp8 || (r->use_wmma && r->fn_gemm_fp8_wmma) || fp8_fp8_eligible) && n_tok >= 16) {
        /* Optimized FP8 paths produce already-effectively-BF16 outputs:
         *  - gemm_opt_fp8 has fused BF16 trunc on writeback
         *  - WMMA path computes BF16×BF16 with F32 accum; downstream code is
         *    tolerant of either F32 or BF16-truncated outputs, so skip the
         *    extra trunc pass to save a kernel launch. */
        op_gemm_fp8(r, Y, W, X, bias, n_out, n_in, n_tok);
        return;
    }
    op_wgemm(r, Y, W, X, bias, n_out, n_in, n_tok);
    op_bf16_trunc(r, Y, n_out * n_tok);
}

/* Like op_wgemm_bf16 but with explicit per-weight FP8 flag. Used for global
 * GEMM weights when the checkpoint is mixed-dtype (some globals are BF16
 * stored as F32 device buffers, others are FP8 raw bytes). */
static void op_wgemm_bf16_auto(hip_qimg_runner *r, int is_fp8,
                                void *Y, void *W, void *X, void *bias,
                                int n_out, int n_in, int n_tok) {
    if (is_fp8) {
        op_wgemm_bf16(r, Y, W, X, bias, n_out, n_in, n_tok);
        return;
    }
    /* F32 weight path: gemm_f32_f32 + downstream BF16 trunc to keep
     * activation magnitudes consistent with the FP8 path's output. */
    op_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
    op_bf16_trunc(r, Y, n_out * n_tok);
}

/* F32-only GEMM + BF16 truncation (for non-weight GEMMs) */
static void op_gemm_bf16(hip_qimg_runner *r, void *Y, void *W, void *X, void *bias,
                         int n_out, int n_in, int n_tok) {
    op_gemm(r, Y, W, X, bias, n_out, n_in, n_tok);
    op_bf16_trunc(r, Y, n_out * n_tok);
}

/* Unfused W4A16 SVDQuant linear: Y[n_tok,n_out] = (W·wscale/smooth)@X + lora_up@(lora_down@X) + bias.
 * Verified-correct, perf-naive (per-call dense materialize + f32 GEMMs); the fused dequant-LDS WMMA kernel
 * replaces the W-materialize later. L = &int4_linears[block*PER_BLOCK + slot]. X is F32 [n_tok,n_in]. */
/* INT8 SmoothQuant W8A8 linear: quant X per-token to int8 (dividing by `smooth` if non-NULL),
 * int8xint8->int32 dp4a GEMM with fused dequant (xscale[t]*wscale[o]) + bias -> f32 Y[n_tok,n_out].
 * Wq is the (streamed) int8 weight [n_out,n_in]; wscale[n_out]/smooth[n_in] are resident. */
static void op_gemm_int8(hip_qimg_runner *r, void *Y, void *Wq, void *wscale, void *smooth,
                         void *X, void *bias, int n_out, int n_in, int n_tok) {
    size_t xqb = (size_t)n_tok * n_in;
    if (xqb > r->i8_xq_cap) { if (r->d_xq_int8) { hipDeviceSynchronize(); hipFree(r->d_xq_int8); }
        hipMalloc(&r->d_xq_int8, xqb); r->i8_xq_cap = xqb; }
    size_t isb = (size_t)n_tok * 4;
    if (isb > r->i8_is_cap) { if (r->d_x_iscale) { hipDeviceSynchronize(); hipFree(r->d_x_iscale); }
        hipMalloc((void **)&r->d_x_iscale, isb); r->i8_is_cap = isb; }
    void *qa[] = {&X, &smooth, &r->d_xq_int8, &r->d_x_iscale, &n_tok, &n_in};
    hipModuleLaunchKernel(r->fn_quant_act_int8, (unsigned)n_tok, 1, 1, 256, 1, 1, 0, NULL, qa, NULL);
    void *ga[] = {&Y, &Wq, &wscale, &r->d_xq_int8, &r->d_x_iscale, &bias, &n_out, &n_in, &n_tok};
    static int i8_dp4a = -1; if (i8_dp4a < 0) i8_dp4a = getenv("QIMG_INT8_DP4A") ? 1 : 0;
    if (r->fn_gemm_w8a8_pgr2 && !i8_dp4a && (n_tok % 128) == 0 && (n_out % 128) == 0 && (n_in % 32) == 0) {
        /* Pipelined int8 WMMA (double-buffered LDS) — the fast path for the big img linears. */
        void *gp[] = {&Y, &Wq, &r->d_xq_int8, &bias, &r->d_x_iscale, &wscale, &n_out, &n_in, &n_tok, &n_tok};
        hipModuleLaunchKernel(r->fn_gemm_w8a8_pgr2, (unsigned)(n_out / 128), (unsigned)(n_tok / 128), 1, 128, 1, 1, 0, NULL, gp, NULL);
    } else if (r->fn_gemm_w8a8_wmma && !i8_dp4a) {  /* simple int8 WMMA: any size (n_tok=1/19 tails), zero-padded — faster than scalar dp4a even with tile waste */
        hipModuleLaunchKernel(r->fn_gemm_w8a8_wmma, (unsigned)((n_out + 127) / 128), (unsigned)((n_tok + 127) / 128), 1, 256, 1, 1, 0, NULL, ga, NULL);
    } else {  /* scalar dp4a fallback (QIMG_INT8_DP4A=1 or no WMMA) */
        hipModuleLaunchKernel(r->fn_gemm_w8a8, (unsigned)((n_out + 127) / 128), (unsigned)n_tok, 1, 128, 1, 1, 0, NULL, ga, NULL);
    }
    /* One-shot kernel self-test: host reference = (Wq*wscale)·(x/smooth)+bias, vs the GEMM Y. */
    static int i8_dbg_done = 0;
    if (getenv("QIMG_INT8_DEBUG") && !i8_dbg_done && n_tok >= 64 && n_tok <= 256) {
        i8_dbg_done = 1; hipDeviceSynchronize();
        signed char *hW = malloc((size_t)n_out*n_in); float *hws = malloc((size_t)n_out*4);
        float *hsm = smooth ? malloc((size_t)n_in*4) : NULL, *hX = malloc((size_t)n_tok*n_in*4);
        float *hY = malloc((size_t)n_tok*n_out*4), *hb = bias ? malloc((size_t)n_out*4) : NULL;
        hipMemcpy(hW, Wq, (size_t)n_out*n_in, hipMemcpyDeviceToHost);
        hipMemcpy(hws, wscale, (size_t)n_out*4, hipMemcpyDeviceToHost);
        if (hsm) hipMemcpy(hsm, smooth, (size_t)n_in*4, hipMemcpyDeviceToHost);
        hipMemcpy(hX, X, (size_t)n_tok*n_in*4, hipMemcpyDeviceToHost);
        hipMemcpy(hY, Y, (size_t)n_tok*n_out*4, hipMemcpyDeviceToHost);
        if (hb) hipMemcpy(hb, bias, (size_t)n_out*4, hipMemcpyDeviceToHost);
        double dot=0, nr=0, ng=0; int t=0;
        for (int o=0;o<n_out;o++){ double acc=0; for(int k=0;k<n_in;k++){ double xv=hX[(size_t)t*n_in+k]; if(hsm)xv/=hsm[k];
            acc += (double)hW[(size_t)o*n_in+k]*hws[o]*xv; } if(hb)acc+=hb[o];
            double g=hY[(size_t)t*n_out+o]; dot+=acc*g; nr+=acc*acc; ng+=g*g; }
        fprintf(stderr,"[int8dbg] n_out=%d n_in=%d n_tok=%d smooth=%d cos(gpu,host)=%.5f ||host||=%.3g ||gpu||=%.3g\n",
                n_out,n_in,n_tok,smooth?1:0, dot/(sqrt(nr)*sqrt(ng)+1e-30), sqrt(nr), sqrt(ng));
        free(hW);free(hws);free(hX);free(hY); if(hsm)free(hsm); if(hb)free(hb);
    }
}

static void op_int4_linear(hip_qimg_runner *r, void *Y, void *X, const qimg_int4_linear *L, int n_tok) {
    int n_out = L->n_out, n_in = L->n_in, gs = L->group_size, rk = L->rank;
    /* Simple RTN int4-g16 path (no swizzle/LoRA/smooth): plain nibble + per-g16
     * bf16 scale via the validated gemm_int4w_g16 kernel. */
    if (gs == 16 && r->fn_gemm_int4w_g16) {
        void *ga[] = {&Y, (void*)&L->qint4, (void*)&L->wscale, &X, (void*)&L->bias, &n_out, &n_in, &n_tok};
        hipModuleLaunchKernel(r->fn_gemm_int4w_g16, (unsigned)((n_out+127)/128),
                              (unsigned)((n_tok+127)/128), 1, 256, 1, 1, 0, NULL, ga, NULL);
        return;
    }
    /* fused W4A16: dequant-in-LDS bf16 WMMA (main+smooth+bias) — replaces dense materialize */
    void *ga[]={&Y,(void*)&L->qint4,&X,(void*)&L->bias,(void*)&L->wscale,(void*)&L->smooth,&n_out,&n_in,&n_tok};
    hipModuleLaunchKernel(r->fn_gemm_int4w,(unsigned)((n_out+127)/128),(unsigned)((n_tok+127)/128),1,256,1,1,0,NULL,ga,NULL);
    if (rk > 0 && L->lora_down) {                                     /* rank-128 residual (skip for mod: rank 0) */
        if (!r->i4_ldf) { hipMalloc(&r->i4_ldf,(size_t)128*12288*4); hipMalloc(&r->i4_luf,(size_t)18432*128*4); }
        /* Growing these shared scratch buffers means hipFree+hipMalloc; sync
         * first so we never free a buffer a prior launch is still reading. Skip
         * and the GEMM faults at the first bigger output (img_mlp_fc1, 12288) and
         * the whole queue hangs. */
        if ((size_t)rk*n_tok > r->i4_dt_cap) { hipDeviceSynchronize(); hipFree(r->i4_dt); hipMalloc(&r->i4_dt,(size_t)rk*n_tok*4); r->i4_dt_cap=(size_t)rk*n_tok; }
        if ((size_t)n_out*n_tok > r->i4_dly_cap) { hipDeviceSynchronize(); hipFree(r->i4_dly); hipMalloc(&r->i4_dly,(size_t)n_out*n_tok*4); r->i4_dly_cap=(size_t)n_out*n_tok; }
        float *ldf=r->i4_ldf,*luf=r->i4_luf,*dt=r->i4_dt,*dly=r->i4_dly;
        int ldn=rk*n_in, lun=n_out*rk; void *e1[]={(void*)&L->lora_down,&ldf,&ldn}, *e2[]={(void*)&L->lora_up,&luf,&lun};
        hipModuleLaunchKernel(r->fn_expand_bf16,(unsigned)((ldn+255)/256),1,1,256,1,1,0,NULL,e1,NULL);
        hipModuleLaunchKernel(r->fn_expand_bf16,(unsigned)((lun+255)/256),1,1,256,1,1,0,NULL,e2,NULL);
        op_gemm(r, dt, ldf, X, NULL, rk, n_in, n_tok); op_gemm(r, dly, luf, dt, NULL, n_out, rk, n_tok);
        int ny=n_out*n_tok; void *aa[]={&Y,&dly,&ny}; hipModuleLaunchKernel(r->fn_add,(unsigned)((ny+255)/256),1,1,256,1,1,0,NULL,aa,NULL);
        /* Drain before the next linear reuses the shared dt/dly scratch: 60
         * blocks x ~6 unsynced launches each overruns the queue and deadlocks
         * at the final block. ~0.06 ms/step cost; safe. */
        hipDeviceSynchronize();
    }
}

/* Route a block projection: int4 descriptor[blk*12+slot] when a logical-INT4 DiT is loaded, else BF16 weight. */
static void op_proj(hip_qimg_runner *r, void *Y, void *W, void *X, void *bias,
                    int n_out, int n_in, int n_tok, int blk, int slot) {
    /* Calibration: accumulate this linear's per-input-channel activation max-abs (offline SVDQuant smoothing).
     * Hooks the activation X before the GEMM; works in the bf16/fp8 path (collected before any INT4 exists). */
    if (r->calib_dump_path && r->fn_calib_amax && X && blk >= 0 && slot >= 0
        && blk * 12 + slot < QIMG_CALIB_MAXSLOT) {
        int idx = blk * 12 + slot;
        if (!r->calib_amax[idx] && hipMalloc(&r->calib_amax[idx], (size_t)n_in * 4) == hipSuccess) {
            hipMemset(r->calib_amax[idx], 0, (size_t)n_in * 4);
            r->calib_nin[idx] = n_in;
        }
        if (r->calib_amax[idx]) {
            void *ca[] = {&r->calib_amax[idx], &X, &n_tok, &n_in};
            hipModuleLaunchKernel(r->fn_calib_amax, (unsigned)((n_in + 255) / 256), 1, 1, 256, 1, 1, 0, NULL, ca, NULL);
        }
    }
    if (r->use_int4 && r->int4_linears) { op_int4_linear(r, Y, X, &r->int4_linears[(size_t)blk*12 + slot], n_tok); return; }
    if (r->use_int8 && r->i8_ws) {   /* W is the streamed int8 weight; scales resident at [blk,slot] */
        size_t ix = (size_t)blk * QIMG_I8_PER_BLOCK + slot;
        op_gemm_int8(r, Y, W, r->i8_ws[ix], r->use_int8_smooth ? r->i8_sm[ix] : NULL, X, bias, n_out, n_in, n_tok);
        return;
    }
    op_wgemm_bf16(r, Y, W, X, bias, n_out, n_in, n_tok);
}

static void op_silu(hip_qimg_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_silu, (unsigned)((n+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

static void op_gelu(hip_qimg_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_gelu, (unsigned)((n+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

static void op_adaln(hip_qimg_runner *r, void *out, void *x,
                     void *shift, void *scale, int N, int dim) {
    void *args[] = {&out, &x, &shift, &scale, &N, &dim};
    hipModuleLaunchKernel(r->fn_adaln, (unsigned)N, 1, 1, 256, 1, 1,
                          256 * sizeof(float), NULL, args, NULL);
}

static void op_rmsnorm_ph(hip_qimg_runner *r, void *x, void *w,
                          int N, int n_heads, int head_dim) {
    void *args[] = {&x, &w, &N, &n_heads, &head_dim};
    hipModuleLaunchKernel(r->fn_rmsnorm_ph, (unsigned)N, (unsigned)n_heads, 1,
                          32, 1, 1, 0, NULL, args, NULL);
}

static void op_gated_add(hip_qimg_runner *r, void *x, void *proj,
                         void *gate, int N, int dim) {
    int total = N * dim;
    void *args[] = {&x, &proj, &gate, &N, &dim};
    hipModuleLaunchKernel(r->fn_gated_add, (unsigned)((total+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

static void op_attn(hip_qimg_runner *r, void *d_out, void *d_q,
                    void *d_k, void *d_v,
                    int n_tok, int n_heads, int head_dim) {
    if (r->use_attn_fp8 && head_dim == 128 && r->fn_flash_attn_fp8_perrow
        && getenv("QIMG_FP8_ATTN_PERROW")) {
        size_t qkv_bytes = (size_t)n_tok * n_heads * head_dim;
        size_t perrow_bytes = (size_t)n_heads * n_tok * sizeof(float);
        size_t vscales_bytes = (size_t)n_heads * sizeof(float);
        if (qkv_bytes > r->fa_qkv_bytes) {
            if (r->d_fa_qfp8) hipFree(r->d_fa_qfp8);
            if (r->d_fa_kfp8) hipFree(r->d_fa_kfp8);
            if (r->d_fa_vfp8) hipFree(r->d_fa_vfp8);
            hipMalloc(&r->d_fa_qfp8, qkv_bytes);
            hipMalloc(&r->d_fa_kfp8, qkv_bytes);
            hipMalloc(&r->d_fa_vfp8, qkv_bytes);
            r->fa_qkv_bytes = qkv_bytes;
        }
        if (perrow_bytes > r->fa_perrow_bytes) {
            if (r->d_fa_qs) hipFree(r->d_fa_qs);
            if (r->d_fa_ks) hipFree(r->d_fa_ks);
            hipMalloc((void **)&r->d_fa_qs, perrow_bytes);
            hipMalloc((void **)&r->d_fa_ks, perrow_bytes);
            r->fa_perrow_bytes = perrow_bytes;
        }
        if (vscales_bytes > r->fa_scales_bytes) {
            if (r->d_fa_vs) hipFree(r->d_fa_vs);
            hipMalloc((void **)&r->d_fa_vs, vscales_bytes);
            r->fa_scales_bytes = vscales_bytes;
        }
        void *d_qfp8 = r->d_fa_qfp8, *d_kfp8 = r->d_fa_kfp8, *d_vfp8 = r->d_fa_vfp8;
        void *d_qs = r->d_fa_qs, *d_ks = r->d_fa_ks, *d_vs = r->d_fa_vs;
        /* Per-row Q quant: grid (n_tok, n_heads), block 64. */
        {
            int n_th = 64;
            size_t smem = (size_t)n_th * sizeof(float);
            void *aq[] = {&d_qfp8, &d_qs, &d_q, &n_tok, &n_heads, &head_dim};
            hipModuleLaunchKernel(r->fn_q_quant_perrow,
                                  (unsigned)n_tok, (unsigned)n_heads, 1,
                                  (unsigned)n_th, 1, 1, smem, NULL, aq, NULL);
            void *ak[] = {&d_kfp8, &d_ks, &d_k, &n_tok, &n_heads, &head_dim};
            hipModuleLaunchKernel(r->fn_k_quant_repack_perrow,
                                  (unsigned)n_tok, (unsigned)n_heads, 1,
                                  (unsigned)n_th, 1, 1, smem, NULL, ak, NULL);
        }
        /* Per-head V max-abs (use 3-output kernel — outputs Q/K/V max but we ignore Q,K). */
        {
            int n_th = 256;
            size_t smem = (size_t)3 * n_th * sizeof(float);
            /* Reuse d_qs head[0..n_heads-1] is 'per-row'; need a separate small buffer
             * for the 3-output kernel. Hack: allocate temporary on-the-fly is fine here
             * since this is a tiny per-step alloc. */
            void *tmp_q = NULL, *tmp_k = NULL;
            hipMalloc(&tmp_q, (size_t)n_heads*sizeof(float));
            hipMalloc(&tmp_k, (size_t)n_heads*sizeof(float));
            void *args[] = {&tmp_q, &tmp_k, &d_vs, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
            hipModuleLaunchKernel(r->fn_qkv_perhead_maxabs, (unsigned)n_heads, 1, 1,
                                  (unsigned)n_th, 1, 1, smem, NULL, args, NULL);
            /* Quantize V into FP8 (head-major) using per-head v_scale. */
            void *av[] = {&d_vfp8, &d_v, &d_vs, &n_tok, &n_heads, &head_dim};
            hipModuleLaunchKernel(r->fn_kv_quant_repack_fp8, (unsigned)n_tok, (unsigned)n_heads, 1,
                                  64, 1, 1, 0, NULL, av, NULL);
            hipFree(tmp_q); hipFree(tmp_k);
        }
        {
            float inv_sqrtd = 1.0f / sqrtf((float)head_dim);
            unsigned gy = (unsigned)((n_tok + 63) / 64);
            void *afa[] = {&d_out, &d_qfp8, &d_kfp8, &d_vfp8, &d_qs, &d_ks, &d_vs,
                           &n_tok, &n_heads, &inv_sqrtd};
            hipModuleLaunchKernel(r->fn_flash_attn_fp8_perrow,
                                  (unsigned)n_heads, gy, 1, 128, 1, 1, 0, NULL, afa, NULL);
        }
        return;
    }
    /* Per-head FP8 FA path (default when QIMG_FP8_ATTN=1). Faster than per-row
     * but lower-quality scale grain. Reuses the persistent FA scratch. */
    if (r->use_attn_fp8 && head_dim == 128) {
        size_t qkv_bytes = (size_t)n_tok * n_heads * head_dim;
        size_t scales_bytes = (size_t)n_heads * sizeof(float);
        if (qkv_bytes > r->fa_qkv_bytes) {
            if (r->d_fa_qfp8) hipFree(r->d_fa_qfp8);
            if (r->d_fa_kfp8) hipFree(r->d_fa_kfp8);
            if (r->d_fa_vfp8) hipFree(r->d_fa_vfp8);
            hipMalloc(&r->d_fa_qfp8, qkv_bytes);
            hipMalloc(&r->d_fa_kfp8, qkv_bytes);
            hipMalloc(&r->d_fa_vfp8, qkv_bytes);
            r->fa_qkv_bytes = qkv_bytes;
        }
        if (scales_bytes > r->fa_scales_bytes) {
            if (r->d_fa_vs) hipFree(r->d_fa_vs);
            hipMalloc((void **)&r->d_fa_vs, scales_bytes);
            r->fa_scales_bytes = scales_bytes;
        }
        /* Reuse d_fa_qs / d_fa_ks (sized for per-row at most n_heads*n_tok floats);
         * for per-head we only touch first n_heads slots. Allocate small if missing. */
        if (!r->d_fa_qs) hipMalloc((void **)&r->d_fa_qs, scales_bytes);
        if (!r->d_fa_ks) hipMalloc((void **)&r->d_fa_ks, scales_bytes);
        void *d_qfp8 = r->d_fa_qfp8, *d_kfp8 = r->d_fa_kfp8, *d_vfp8 = r->d_fa_vfp8;
        void *d_qs = r->d_fa_qs, *d_ks = r->d_fa_ks, *d_vs = r->d_fa_vs;
        {
            int n_th = 256;
            size_t smem = (size_t)3 * n_th * sizeof(float);
            void *args[] = {&d_qs, &d_ks, &d_vs, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
            hipModuleLaunchKernel(r->fn_qkv_perhead_maxabs, (unsigned)n_heads, 1, 1,
                                  (unsigned)n_th, 1, 1, smem, NULL, args, NULL);
        }
        {
            void *aq[] = {&d_qfp8, &d_q, &d_qs, &n_tok, &n_heads, &head_dim};
            hipModuleLaunchKernel(r->fn_q_quant_fp8, (unsigned)n_tok, (unsigned)n_heads, 1,
                                  64, 1, 1, 0, NULL, aq, NULL);
            void *ak[] = {&d_kfp8, &d_k, &d_ks, &n_tok, &n_heads, &head_dim};
            hipModuleLaunchKernel(r->fn_kv_quant_repack_fp8, (unsigned)n_tok, (unsigned)n_heads, 1,
                                  64, 1, 1, 0, NULL, ak, NULL);
            void *av[] = {&d_vfp8, &d_v, &d_vs, &n_tok, &n_heads, &head_dim};
            hipModuleLaunchKernel(r->fn_kv_quant_repack_fp8, (unsigned)n_tok, (unsigned)n_heads, 1,
                                  64, 1, 1, 0, NULL, av, NULL);
        }
        {
            float h_qs_buf[64], h_ks_buf[64];
            int n_q = n_heads;
            hipMemcpy(h_qs_buf, d_qs, (size_t)n_q*sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(h_ks_buf, d_ks, (size_t)n_q*sizeof(float), hipMemcpyDeviceToHost);
            for (int h = 0; h < n_q; h++) h_qs_buf[h] *= h_ks_buf[h];
            hipMemcpy(d_qs, h_qs_buf, (size_t)n_q*sizeof(float), hipMemcpyHostToDevice);
            float inv_sqrtd = 1.0f / sqrtf((float)head_dim);
            unsigned gy = (unsigned)((n_tok + 63) / 64);
            void *afa[] = {&d_out, &d_qfp8, &d_kfp8, &d_vfp8, &d_qs, &d_vs,
                           &n_tok, &n_heads, &inv_sqrtd};
            hipModuleLaunchKernel(r->fn_flash_attn_fp8,
                                  (unsigned)n_heads, gy, 1, 128, 1, 1, 0, NULL, afa, NULL);
        }
        return;
    }
    if (r->use_attn_wmma_sp && r->fn_flash_attn_wmma_sp && head_dim == 128) {
        unsigned gy = (unsigned)((n_tok + 63) / 64);
        void *args[] = {&d_out, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
        hipModuleLaunchKernel(r->fn_flash_attn_wmma_sp,
                              (unsigned)n_heads, gy, 1,
                              128, 1, 1, 0, NULL, args, NULL);
        return;
    }
    if (r->use_attn_wmma_pq && r->fn_flash_attn_wmma_pq && head_dim == 128) {
        unsigned gy = (unsigned)((n_tok + 63) / 64);
        void *args[] = {&d_out, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
        hipModuleLaunchKernel(r->fn_flash_attn_wmma_pq,
                              (unsigned)n_heads, gy, 1,
                              128, 1, 1, 0, NULL, args, NULL);
        return;
    }
    if (r->use_attn_wmma && r->fn_flash_attn_wmma && head_dim == 128) {
        unsigned gy = (unsigned)((n_tok + 63) / 64);
        void *args[] = {&d_out, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
        hipModuleLaunchKernel(r->fn_flash_attn_wmma,
                              (unsigned)n_heads, gy, 1,
                              128, 1, 1, 0, NULL, args, NULL);
        return;
    }
    int fa2_warps = 4, fa2_bkv = 16;
    unsigned gy = (unsigned)((n_tok + fa2_warps - 1) / fa2_warps);
    unsigned n_threads = (unsigned)(32 * fa2_warps);
    size_t smem = (size_t)2 * fa2_bkv * 128 * sizeof(float);
    void *args[] = {&d_out, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
    hipModuleLaunchKernel(r->fn_flash_attn,
                          (unsigned)n_heads, gy, 1,
                          n_threads, 1, 1,
                          smem, NULL, args, NULL);
}

static void op_rmsnorm_weighted(hip_qimg_runner *r, void *x, void *w, int N, int dim) {
    void *args[] = {&x, &w, &N, &dim};
    hipModuleLaunchKernel(r->fn_rmsnorm_weighted, (unsigned)N, 1, 1,
                          256, 1, 1, 256 * sizeof(float), NULL, args, NULL);
}

/* ---- VAE kernel helpers ---- */

static void vae_op_conv2d(hip_qimg_runner *r, void *out, void *inp,
                          void *w, void *b,
                          int ci, int h, int w_s, int co, int kh, int kw, int rep_pad) {
    int sp = h * w_s;
    if (r->use_vae_wmma && (ci % 16) == 0 && (co % 16) == 0 && (sp % 16) == 0) {
        unsigned gx = (unsigned)(co / 16);
        unsigned gy = (unsigned)(sp / 16);
        if (kh == 3 && kw == 3) {
            void *args[] = {&out, &inp, &w, &b, &ci, &h, &w_s, &co, &rep_pad};
            hipModuleLaunchKernel(r->fn_vae_conv2d_3x3_wmma, gx, gy, 1, 32, 1, 1,
                                  0, NULL, args, NULL);
            return;
        }
        if (kh == 1 && kw == 1) {
            void *args[] = {&out, &inp, &w, &b, &ci, &h, &w_s, &co};
            hipModuleLaunchKernel(r->fn_vae_conv2d_1x1_wmma, gx, gy, 1, 32, 1, 1,
                                  0, NULL, args, NULL);
            return;
        }
    }
    int total = co * h * w_s;
    void *args[] = {&out, &inp, &w, &b, &ci, &h, &w_s, &co, &kh, &kw, &rep_pad};
    hipModuleLaunchKernel(r->fn_vae_conv2d, (unsigned)((total+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

static void vae_op_gn(hip_qimg_runner *r, void *out, void *inp,
                      void *gamma, int C, int spatial) {
    void *args[] = {&out, &inp, &gamma, &C, &spatial};
    hipModuleLaunchKernel(r->fn_vae_rmsnorm, (unsigned)((spatial+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

static void vae_op_silu(hip_qimg_runner *r, void *x, int n) {
    void *args[] = {&x, &n};
    hipModuleLaunchKernel(r->fn_vae_silu, (unsigned)((n+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
}

static void *vae_op_upsample(hip_qimg_runner *r, void *inp, int c, int h, int w) {
    int oh = h*2, ow = w*2;
    void *out = NULL;
    hipMalloc(&out, (size_t)c*oh*ow*sizeof(float));
    int total = c*oh*ow;
    void *args[] = {&out, &inp, &c, &h, &w};
    hipModuleLaunchKernel(r->fn_vae_up2x, (unsigned)((total+255)/256), 1, 1,
                          256, 1, 1, 0, NULL, args, NULL);
    return out;
}

/* GPU VAE ResBlock */
static void *vae_resblock_gpu(hip_qimg_runner *r, void *x,
                               void *n1_g, void *c1_w, void *c1_b,
                               void *n2_g, void *c2_w, void *c2_b,
                               void *sc_w, void *sc_b,
                               int ci, int co, int h, int w) {
    int sp = h * w;
    void *tmp = NULL; hipMalloc(&tmp, (size_t)ci*sp*sizeof(float));
    vae_op_gn(r, tmp, x, n1_g, ci, sp);
    vae_op_silu(r, tmp, ci*sp);
    void *c1_out = NULL; hipMalloc(&c1_out, (size_t)co*sp*sizeof(float));
    vae_op_conv2d(r, c1_out, tmp, c1_w, c1_b, ci, h, w, co, 3, 3, 0);
    hipFree(tmp);

    tmp = NULL; hipMalloc(&tmp, (size_t)co*sp*sizeof(float));
    vae_op_gn(r, tmp, c1_out, n2_g, co, sp);
    vae_op_silu(r, tmp, co*sp);
    void *c2_out = NULL; hipMalloc(&c2_out, (size_t)co*sp*sizeof(float));
    vae_op_conv2d(r, c2_out, tmp, c2_w, c2_b, co, h, w, co, 3, 3, 0);
    hipFree(tmp); hipFree(c1_out);

    void *out = NULL; hipMalloc(&out, (size_t)co*sp*sizeof(float));
    if (sc_w) {
        vae_op_conv2d(r, out, x, sc_w, sc_b, ci, h, w, co, 1, 1, 0);
        int n = co * sp;
        float dt = 1.0f;
        void *ea[] = {&out, &c2_out, &dt, &n};
        hipModuleLaunchKernel(r->fn_euler_step, (unsigned)((n+255)/256), 1, 1,
                              256, 1, 1, 0, NULL, ea, NULL);
    } else {
        int n = co * sp;
        hipMemcpy(out, x, (size_t)n * sizeof(float), hipMemcpyDeviceToDevice);
        float dt = 1.0f;
        void *ea[] = {&out, &c2_out, &dt, &n};
        hipModuleLaunchKernel(r->fn_euler_step, (unsigned)((n+255)/256), 1, 1,
                              256, 1, 1, 0, NULL, ea, NULL);
    }
    hipFree(c2_out);
    return out;
}


/* ---- Init ---- */

int g_hip_initialized = 0;  /* track if HIP was already init'd by another runner */
char g_hip_arch[64] = {0};  /* cached arch string (e.g. "gfx1201") */

/* Standalone correctness gate for gemm_int4w_g16_bf16a_wmma_t: pack a random
 * bf16 weight to simple RTN int4-g16 on the host, run the kernel, and compare to
 * a CPU reference using the SAME dequantized weights (so it isolates kernel
 * arithmetic/indexing, not quant error). Gated by QIMG_INT4_SELFTEST. */
static void qimg_int4_g16_selftest_one(hip_qimg_runner *r, int M, int K, int N) {
    if (!r->fn_gemm_int4w_g16) { fprintf(stderr, "int4-g16 selftest: kernel missing\n"); return; }
    float *X = (float*)malloc((size_t)M*K*4), *W = (float*)malloc((size_t)N*K*4);
    float *Wd = (float*)malloc((size_t)N*K*4); /* dequantized ref */
    unsigned char *q = (unsigned char*)calloc((size_t)N*(K/2),1);
    unsigned short *sc = (unsigned short*)malloc((size_t)N*(K/16)*2);
    unsigned int s=12345;
    #define RND ((s=s*1664525u+1013904223u),((float)(s>>8)*(1.0f/16777216.0f)*2.0f-1.0f))
    for (int i=0;i<M*K;i++) X[i]=RND;
    for (int i=0;i<N*K;i++) W[i]=RND;
    for (int o=0;o<N;o++) for (int g=0; g<K/16; g++) {
        float mx=0; for(int j=0;j<16;j++){float a=fabsf(W[o*K+g*16+j]); if(a>mx)mx=a;}
        float scale=mx/7.0f; if(scale<1e-12f)scale=1e-12f;
        unsigned int sb; float scf=scale; memcpy(&sb,&scf,4); sc[o*(K/16)+g]=(unsigned short)(sb>>16);
        unsigned int sbk=((unsigned int)sc[o*(K/16)+g])<<16; float scq; memcpy(&scq,&sbk,4); /* bf16-rounded scale */
        for (int j=0;j<16;j++){ int kp=g*16+j; int qq=(int)lroundf(W[o*K+kp]/scq); if(qq>7)qq=7; if(qq<-7)qq=-7;
            Wd[o*K+kp]=qq*scq; unsigned char b=q[o*(K/2)+(kp>>1)];
            if(kp&1) b=(b&0x0F)|((qq&0xF)<<4); else b=(b&0xF0)|(qq&0xF); q[o*(K/2)+(kp>>1)]=b; }
    }
    #undef RND
    double *Yr=(double*)malloc((size_t)M*N*8);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double a=0;for(int kk=0;kk<K;kk++)a+=(double)X[m*K+kk]*Wd[n*K+kk];Yr[m*N+n]=a;}
    void *dX,*dq,*dsc,*dY; hipMalloc(&dX,(size_t)M*K*4); hipMalloc(&dq,(size_t)N*(K/2)); hipMalloc(&dsc,(size_t)N*(K/16)*2); hipMalloc(&dY,(size_t)M*N*4);
    hipMemcpy(dX,X,(size_t)M*K*4,hipMemcpyHostToDevice); hipMemcpy(dq,q,(size_t)N*(K/2),hipMemcpyHostToDevice); hipMemcpy(dsc,sc,(size_t)N*(K/16)*2,hipMemcpyHostToDevice);
    void *bias=NULL; void *args[]={&dY,&dq,&dsc,&dX,&bias,&N,&K,&M};
    hipModuleLaunchKernel(r->fn_gemm_int4w_g16,(unsigned)((N+127)/128),(unsigned)((M+127)/128),1,256,1,1,0,NULL,args,NULL);
    hipError_t le = hipDeviceSynchronize();
    float *Yg=(float*)malloc((size_t)M*N*4); hipMemcpy(Yg,dY,(size_t)M*N*4,hipMemcpyDeviceToHost);
    double dot=0,nr=0,ng=0,mx=0; for(int i=0;i<M*N;i++){double a=Yr[i],b=Yg[i];dot+=a*b;nr+=a*a;ng+=b*b;double d=fabs(a-b);if(d>mx)mx=d;}
    double cosv = dot/(sqrt(nr)*sqrt(ng)+1e-30);
    fprintf(stderr,"hip_qimg: int4-g16 selftest M%-4d K%-5d N%-5d  cos=%.6f  max=%.3e  sync_err=%d  %s\n",
        M,K,N, cosv, mx, le, (le==0 && cosv>0.9999)?"PASS":"FAIL");
    hipFree(dX);hipFree(dq);hipFree(dsc);hipFree(dY); free(X);free(W);free(Wd);free(q);free(sc);free(Yr);free(Yg);
}
static void qimg_int4_g16_selftest(hip_qimg_runner *r) {
    /* Real DiT linear shapes (N=n_out, K=n_in) at n_tok 256 (img) and 12 (txt). */
    int shapes[][2] = {{3072,3072},{3072,3584},{12288,3072},{3072,12288},{18432,3072}};
    int Ms[3] = {256, 12, 1};   /* img, txt, modulation (n_tok=1) */
    for (int t = 0; t < 3; t++)
        for (int s = 0; s < 5; s++) qimg_int4_g16_selftest_one(r, Ms[t], shapes[s][1], shapes[s][0]);
}

hip_qimg_runner *hip_qimg_init(int device_id, int verbose) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_qimg: rocewInit failed (HIP/HIPRTC libraries not found)\n");
        return NULL;
    }
    HIP_CHECK_NULL(hipSetDevice(device_id));

    if (verbose) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, device_id);
        fprintf(stderr, "hip_qimg: %s (%.1f GB)\n",
                props.name, (float)props.totalGlobalMem / (1<<30));
    }

    /* Compile kernels */
    size_t len1 = strlen(hip_kernels_common_src);
    size_t len2 = strlen(hip_qimg_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, hip_kernels_common_src, len1);
    memcpy(full_src + len1, hip_qimg_specific_kernels, len2);
    full_src[len1 + len2] = '\0';

    hipModule_t mod;
    int compile_verbose = verbose;
    { const char *e = getenv("QIMG_COMPILE_VERBOSE");
      if (e) compile_verbose = atoi(e); }
    int rc = hip_compile_kernels(&mod, device_id, full_src, "qimg.hip", compile_verbose, "hip_qimg");
    free(full_src);
    if (rc < 0) return NULL;

    hip_qimg_runner *r = (hip_qimg_runner *)calloc(1, sizeof(*r));
    r->device_id = device_id;
    r->verbose = verbose;
    r->mod = mod;
    r->current_block = -1;
    snprintf(r->current_gemm_label, sizeof(r->current_gemm_label), "%s", "init");
    {
        const char *e = getenv("QIMG_PATH_STATS");
        r->path_stats_enabled = (e && !(strcmp(e, "0") == 0 || strcmp(e, "false") == 0));
        e = getenv("QIMG_FP8_QUANT_STATS");
        r->quant_stats_enabled = (e && !(strcmp(e, "0") == 0 || strcmp(e, "false") == 0));
        e = getenv("QIMG_MEM_STATS");
        r->mem_stats_enabled = (e && !(strcmp(e, "0") == 0 || strcmp(e, "false") == 0));
        r->quant_stats_max = 80;
        e = getenv("QIMG_FP8_QUANT_STATS_MAX");
        if (e && atoi(e) > 0) r->quant_stats_max = atoi(e);
        { const char *ar = getenv("QIMG_ACT_FP8_RT"); r->act_fp8_rt = (ar && atoi(ar)); }
        { const char *wr = getenv("QIMG_W_INT8_RT"); r->w_int8_rt = (wr && atoi(wr)); }
        { const char *wr = getenv("QIMG_W_INT4_RT"); r->w_int4_rt = (wr && atoi(wr)); }
        r->fp8_fp8_allow = getenv("QIMG_FP8_FP8_ALLOW");
        r->fp8_fp8_deny = getenv("QIMG_FP8_FP8_DENY");
        r->fp8_fp8_block_min = -1;
        r->fp8_fp8_block_max = -1;
        e = getenv("QIMG_FP8_FP8_BLOCK_MIN");
        if (e && atoi(e) >= 0) r->fp8_fp8_block_min = atoi(e);
        e = getenv("QIMG_FP8_FP8_BLOCK_MAX");
        if (e && atoi(e) >= 0) r->fp8_fp8_block_max = atoi(e);
        r->fp8_quality_target_db = 0.0f;
        e = getenv("QIMG_FP8_QUALITY_TARGET_DB");
        if (e && atof(e) > 0.0) r->fp8_quality_target_db = (float)atof(e);
        r->fp8_act_scale_div = 512.0f;
        e = getenv("QIMG_FP8_ACT_SCALE_DIV");
        if (e && atof(e) > 0.0) r->fp8_act_scale_div = (float)atof(e);
        r->fp8_act_scale_scalar = 0;
        r->fp8_act_scale_clamp = 0;
        e = getenv("QIMG_FP8_ACT_SCALE_MODE");
        if (e && (!strcmp(e, "comfy") || !strcmp(e, "scalar") || !strcmp(e, "scale1")))
            r->fp8_act_scale_scalar = 1;
        if (e && (!strcmp(e, "clamp") || !strcmp(e, "safe") || !strcmp(e, "safe_scalar")))
            r->fp8_act_scale_clamp = 1;
        r->calib_dump_path = getenv("QIMG_CALIB_DUMP");  /* enables per-linear activation max-abs collection */
        if (r->calib_dump_path && !r->calib_dump_path[0]) r->calib_dump_path = NULL;
    }

    #define GET(field, name) hipModuleGetFunction(&r->field, mod, name)
    GET(fn_gemm, "gemm_f32_f32");
    GET(fn_gemm_fp8, "gemm_fp8w_f32");
    GET(fn_gemm_opt_fp8, "gemm_opt_fp8");
    if (hipModuleGetFunction(&r->fn_gemm_fp8_wmma, mod, "gemm_fp8w_bf16a_wmma_t") != hipSuccess)
        r->fn_gemm_fp8_wmma = NULL;
    if (hipModuleGetFunction(&r->fn_gemm_fp8_fp8_pgr2, mod, "gemm_fp8_fp8w_perrow_pgr2") != hipSuccess)
        r->fn_gemm_fp8_fp8_pgr2 = NULL;
    if (hipModuleGetFunction(&r->fn_quantize_act_perrow, mod, "qimg_quantize_act_perrow_fp8") != hipSuccess)
        r->fn_quantize_act_perrow = NULL;
    if (hipModuleGetFunction(&r->fn_quantize_act_scalar, mod, "qimg_quantize_act_scalar_fp8") != hipSuccess)
        r->fn_quantize_act_scalar = NULL;
    if (hipModuleGetFunction(&r->fn_quantize_act_clamp, mod, "qimg_quantize_act_clamp_fp8") != hipSuccess)
        r->fn_quantize_act_clamp = NULL;
    if (hipModuleGetFunction(&r->fn_add_bias, mod, "qimg_add_bias_rowmajor_f32") != hipSuccess)
        r->fn_add_bias = NULL;
    GET(fn_layernorm, "layernorm_f32");
    GET(fn_dequant_int4_main, "dequant_int4_logical_main_f32");
    GET(fn_add, "add_inplace_f32");
    GET(fn_expand_bf16, "expand_bf16_f32");
    GET(fn_gemm_int4w, "gemm_int4w_bf16a_wmma_t");
    GET(fn_silu, "silu_f32");
    GET(fn_gelu, "gelu_f32");
    GET(fn_adaln, "adaln_modulate_f32");
    GET(fn_gated_add, "gated_add_f32");
    GET(fn_rmsnorm_ph, "rmsnorm_per_head_f32");
    GET(fn_flash_attn, "flash_attn_f32");
    if (hipModuleGetFunction(&r->fn_flash_attn_wmma, mod, "flash_attn_sa_wmma_f32") != hipSuccess)
        r->fn_flash_attn_wmma = NULL;
    if (hipModuleGetFunction(&r->fn_flash_attn_wmma_pq, mod, "flash_attn_sa_wmma_pq_f32") != hipSuccess)
        r->fn_flash_attn_wmma_pq = NULL;
    if (hipModuleGetFunction(&r->fn_flash_attn_wmma_sp, mod, "flash_attn_sa_wmma_sp_f32") != hipSuccess)
        r->fn_flash_attn_wmma_sp = NULL;
    GET(fn_rope_2d, "rope_2d_f32");
    GET(fn_rope_1d, "rope_1d_f32");
    GET(fn_bf16_trunc, "truncate_bf16_f32");
    GET(fn_patchify, "patchify_f32");
    GET(fn_unpatchify, "unpatchify_f32");
    GET(fn_euler_step, "euler_step_f32");
    GET(fn_cfg_combine, "cfg_combine_f32");
    GET(fn_rmsnorm_weighted, "rmsnorm_weighted_f32");
    GET(fn_fp8_roundtrip, "quantize_fp8_roundtrip_f32");
    GET(fn_act_fp8_rt, "act_fp8_roundtrip_perrow");
    GET(fn_w_int8_rt, "w_int8_roundtrip_g64");
    GET(fn_w_int4_rt, "w_int4_roundtrip_g16");
    GET(fn_gemm_int4w_g16, "gemm_int4w_g16_bf16a_wmma_t");
    GET(fn_calib_amax, "amax_per_col_f32");
    if (hipModuleGetFunction(&r->fn_quant_act_int8, mod, "quant_act_perrow_int8") != hipSuccess) r->fn_quant_act_int8 = NULL;
    if (hipModuleGetFunction(&r->fn_gemm_w8a8, mod, "gemm_w8a8_perrow_f32") != hipSuccess) r->fn_gemm_w8a8 = NULL;
    if (hipModuleGetFunction(&r->fn_gemm_w8a8_wmma, mod, "gemm_w8a8_wmma") != hipSuccess) r->fn_gemm_w8a8_wmma = NULL;
    if (hipModuleGetFunction(&r->fn_gemm_w8a8_pgr2, mod, "gemm_w8a8_pgr2") != hipSuccess) r->fn_gemm_w8a8_pgr2 = NULL;
    if (r->calib_dump_path)
        fprintf(stderr, "hip_qimg: QIMG_CALIB_DUMP active -> %s (collecting per-linear activation max-abs)\n",
                r->calib_dump_path);
    GET(fn_vae_conv2d, "vae_conv2d_f32");
    if (hipModuleGetFunction(&r->fn_vae_conv2d_3x3_wmma, mod, "vae_conv2d_3x3_wmma_f32") != hipSuccess)
        r->fn_vae_conv2d_3x3_wmma = NULL;
    if (hipModuleGetFunction(&r->fn_vae_conv2d_1x1_wmma, mod, "vae_conv2d_1x1_wmma_f32") != hipSuccess)
        r->fn_vae_conv2d_1x1_wmma = NULL;
    if (hipModuleGetFunction(&r->fn_vae_self_attn, mod, "vae_self_attn_f32") != hipSuccess)
        r->fn_vae_self_attn = NULL;
    if (hipModuleGetFunction(&r->fn_vae_self_attn_qb, mod, "vae_self_attn_qb_f32") != hipSuccess)
        r->fn_vae_self_attn_qb = NULL;
    if (hipModuleGetFunction(&r->fn_qkv_perhead_maxabs, mod, "qimg_qkv_perhead_maxabs_f32") != hipSuccess)
        r->fn_qkv_perhead_maxabs = NULL;
    if (hipModuleGetFunction(&r->fn_q_quant_fp8, mod, "qimg_q_quant_fp8") != hipSuccess)
        r->fn_q_quant_fp8 = NULL;
    if (hipModuleGetFunction(&r->fn_kv_quant_repack_fp8, mod, "qimg_kv_quant_repack_fp8") != hipSuccess)
        r->fn_kv_quant_repack_fp8 = NULL;
    if (hipModuleGetFunction(&r->fn_flash_attn_fp8, mod, "qimg_flash_attn_fp8_4w") != hipSuccess)
        r->fn_flash_attn_fp8 = NULL;
    if (hipModuleGetFunction(&r->fn_q_quant_perrow, mod, "qimg_q_quant_fp8_perrow") != hipSuccess)
        r->fn_q_quant_perrow = NULL;
    if (hipModuleGetFunction(&r->fn_k_quant_repack_perrow, mod, "qimg_k_quant_repack_fp8_perrow") != hipSuccess)
        r->fn_k_quant_repack_perrow = NULL;
    if (hipModuleGetFunction(&r->fn_flash_attn_fp8_perrow, mod, "qimg_flash_attn_fp8_4w_perrow") != hipSuccess)
        r->fn_flash_attn_fp8_perrow = NULL;
    {
        const char *e = getenv("QIMG_FP8_ATTN");
        r->use_attn_fp8 = (e && atoi(e) && r->fn_flash_attn_fp8 && r->fn_qkv_perhead_maxabs
                          && r->fn_q_quant_fp8 && r->fn_kv_quant_repack_fp8) ? 1 : 0;
        if (r->use_attn_fp8) fprintf(stderr, "hip_qimg: FP8 WMMA flash-attention enabled\n");
    }
    GET(fn_vae_rmsnorm, "vae_rmsnorm_f32");
    GET(fn_vae_silu, "vae_silu_f32");
    GET(fn_vae_up2x, "nn_upsample2x_f32");
    #undef GET

    /* Enable FP8 mode: upload FP8→F32 LUT to GPU constant memory */
    r->use_fp8 = 0;
    if (r->fn_gemm_fp8) {
        init_fp8_to_f32_lut();
        hipDeviceptr_t d_lut;
        size_t lut_size;
        hipError_t lut_err = hipModuleGetGlobal(&d_lut, &lut_size, mod, "d_fp8_to_f32_lut");
        if (lut_err == hipSuccess && lut_size == 256 * sizeof(float)) {
            hipMemcpyHtoD(d_lut, hip_fp8_to_f32_lut, 256 * sizeof(float));
            r->use_fp8 = 1;
            if (verbose)
                fprintf(stderr, "hip_qimg: FP8 LUT GEMM enabled (4x less VRAM per block)\n");
        }
    }
    if (getenv("QIMG_FORCE_F32W")) {  /* bf16/f32 ckpt: upload weights as f32, stream (1296MB/blk) */
        r->use_fp8 = 0;
        fprintf(stderr, "hip_qimg: QIMG_FORCE_F32W — fp8 disabled, weights uploaded as F32\n");
    }

    /* BF16xFP8 WMMA matrix-core path (gfx12). Default ON for FP8
     * checkpoints when the kernel is available; opt out with QIMG_FP8_WMMA=0. */
    r->use_wmma = 0;
    {
        const char *v = getenv("QIMG_FP8_WMMA");
        int want_wmma = !(v && (strcmp(v, "0") == 0 || strcmp(v, "false") == 0));
        if (want_wmma) {
            if (!r->fn_gemm_fp8_wmma) {
                if (v)
                    fprintf(stderr, "hip_qimg: WMMA kernel not present (need gfx12); ignoring QIMG_FP8_WMMA\n");
            } else if (!r->use_fp8) {
                if (v)
                    fprintf(stderr, "hip_qimg: QIMG_FP8_WMMA requires FP8 mode; ignoring\n");
            } else {
                r->use_wmma = 1;
                if (verbose)
                    fprintf(stderr, "hip_qimg: BF16xFP8 WMMA (gfx12 matrix cores) enabled\n");
            }
        }
        if (!r->use_wmma && r->fp8_quality_target_db > 0.0f && r->fn_gemm_fp8_wmma && r->use_fp8) {
            r->use_wmma = 1;
            if (verbose)
                fprintf(stderr, "hip_qimg: BF16xFP8 WMMA fallback enabled for FP8 quality target\n");
        }
    }

    /* FP8xFP8 WMMA path (gfx12). This is the ComfyUI --fast fp8_matrix_mult
     * path: activation clamp/cast to FP8, raw FP8 weights, unit scales by
     * default, BF16xFP8 fallback otherwise. QIMG_FP8_WMMA_BF16=1 forces the
     * BF16 path even when FP8xFP8 qualifies. */
    r->use_fp8_fp8w = 0;
    r->fast_fp8_matrix_mult = 0;
    r->prefer_bf16_wmma = 0;
    r->d_act_fp8 = NULL;
    r->d_act_scales = NULL;
    r->act_fp8_bytes = 0;
    r->d_fa_qfp8 = r->d_fa_kfp8 = r->d_fa_vfp8 = NULL;
    r->d_fa_qs = r->d_fa_ks = r->d_fa_vs = NULL;
    r->fa_qkv_bytes = 0;
    r->fa_scales_bytes = 0;
    r->fa_perrow_bytes = 0;
    r->act_scales_bytes = 0;
    {
        const char *v = getenv("QIMG_FAST_FP8_MATRIX_MULT");
        int fast_off = (v && (strcmp(v, "0") == 0 || strcmp(v, "false") == 0));
        int fast_on  = (v && !fast_off);
        r->fast_fp8_matrix_mult = fast_on ? 1 : 0;

        /* Quality-safe default preset (2026-05-23): when the user gives no
         * explicit FP8 fast configuration, route the perceptually-lossless
         * 48 dB tier to FP8xFP8 — img_mlp_fc1 in blocks >=8, clamp activation
         * quant. Measured 49.67 dB latent / PNG indistinguishable from the
         * pure BF16xFP8 default (54.7 dB PNG PSNR, max pixel diff 8/255), at
         * 29% of the WMMA-eligible GEMM pool. Earlier fc1 blocks (0..7) carry
         * the intrinsic e4m3 activation-mantissa error that floors at ~44 dB,
         * so they stay BF16xFP8. See README.native-fp8.md / project memory.
         * Opt out: --fast none (QIMG_FAST_FP8_MATRIX_MULT=0) or
         * QIMG_FP8_WMMA_BF16=1 restores the pure BF16xFP8 path. */
        if (!fast_on && !fast_off && r->use_fp8 &&
            !getenv("QIMG_FP8_WMMA_BF16") &&
            !(r->fp8_fp8_allow && r->fp8_fp8_allow[0]) &&
            !(r->fp8_fp8_deny && r->fp8_fp8_deny[0]) &&
            r->fp8_fp8_block_min < 0 && r->fp8_fp8_block_max < 0 &&
            r->fp8_quality_target_db <= 0.0f &&
            r->fn_gemm_fp8_fp8_pgr2 && r->fn_quantize_act_clamp) {
            r->fast_fp8_matrix_mult = 1;
            r->fp8_act_scale_clamp = 1;
            r->fp8_act_scale_scalar = 0;
            r->fp8_fp8_allow = "img_mlp_fc1";
            r->fp8_fp8_block_min = 8;
            if (verbose)
                fprintf(stderr, "hip_qimg: quality-safe FP8xFP8 default ON "
                        "(img_mlp_fc1 blocks>=8, clamp act; ~48 dB tier)\n");
        }

        if (r->fast_fp8_matrix_mult) {
            if (!r->fp8_act_scale_scalar && !r->fp8_act_scale_clamp)
                r->fp8_act_scale_scalar = 1;
            hipFunction_t quant_fn = r->fp8_act_scale_clamp ? r->fn_quantize_act_clamp :
                                     (r->fp8_act_scale_scalar ? r->fn_quantize_act_scalar : r->fn_quantize_act_perrow);
            if (!r->fn_gemm_fp8_fp8_pgr2 || !quant_fn) {
                fprintf(stderr, "hip_qimg: FP8xFP8 kernel not present (need gfx12); ignoring --fast fp8_matrix_mult\n");
            } else if (!r->use_fp8) {
                fprintf(stderr, "hip_qimg: --fast fp8_matrix_mult requires FP8 mode; ignoring\n");
            } else {
                r->use_fp8_fp8w = 1;
                if (verbose) {
                    const char *mode = r->fp8_act_scale_scalar ? "Comfy scale=1 clamp/cast" :
                                       (r->fp8_act_scale_clamp ? "safe scalar" : "per-row");
                    fprintf(stderr, "hip_qimg: ComfyUI --fast fp8_matrix_mult enabled: FP8xFP8 WMMA + %s act quant\n", mode);
                    if (r->fp8_fp8_allow && r->fp8_fp8_allow[0])
                        fprintf(stderr, "hip_qimg: FP8xFP8 allow labels: %s\n", r->fp8_fp8_allow);
                    if (r->fp8_fp8_deny && r->fp8_fp8_deny[0])
                        fprintf(stderr, "hip_qimg: FP8xFP8 deny labels: %s\n", r->fp8_fp8_deny);
                    if (r->fp8_fp8_block_min >= 0 || r->fp8_fp8_block_max >= 0)
                        fprintf(stderr, "hip_qimg: FP8xFP8 block range: %d..%d\n",
                                r->fp8_fp8_block_min, r->fp8_fp8_block_max);
                    if (r->fp8_quality_target_db > 0.0f)
                        fprintf(stderr, "hip_qimg: FP8 quality target: %.1f dB\n",
                                r->fp8_quality_target_db);
                    if (!r->fp8_act_scale_scalar && !r->fp8_act_scale_clamp && r->fp8_act_scale_div != 512.0f)
                        fprintf(stderr, "hip_qimg: FP8 activation scale divisor: %.1f\n", r->fp8_act_scale_div);
                }
            }
        }
        const char *vb = getenv("QIMG_FP8_WMMA_BF16");
        if (vb && !(strcmp(vb, "0") == 0 || strcmp(vb, "false") == 0)) {
            r->prefer_bf16_wmma = 1;
            if (verbose)
                fprintf(stderr, "hip_qimg: QIMG_FP8_WMMA_BF16=1 -- forcing BF16xFP8 path even where FP8xFP8 would qualify\n");
        }
    }

    /* Vendor (Tensile) FP8 GEMM for the FP8xFP8 fast path. Default ON when
     * --fast is enabled and the extracted .co is present (QIMG_FP8_VENDOR=0
     * forces the hand-written kernel). ~1.5x our kernel, matches torch. The
     * .co is a build artifact: regenerate via rdna4/fp8/extract_fp8_kernel.sh. */
    if (r->use_fp8_fp8w) {
        /* Opt-in (QIMG_FP8_VENDOR=1): the vendor kernel uses a scalar activation
         * scale, so it forces scale=1 quant (~1.5 dB below the hand-written
         * clamp path on knife-edge configs). Default OFF keeps the documented
         * quality-safe configs valid; enable for the ~1.5x GEMM / ~9% e2e win
         * at the 48 dB (perceptually-lossless) tier. */
        const char *vd = getenv("QIMG_FP8_VENDOR");
        int want = (vd && !(strcmp(vd, "0") == 0 || strcmp(vd, "false") == 0));
        if (want && r->fn_add_bias) {
            static const char *VSYM =
                "Cijk_Alik_Bljk_F8SS_BH_Bias_SHB_HA_S_SAB_SCD_SAV_UserArgs_"
                "MT128x128x64_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CLR1_CADS0_"
                "DTLA0_DTLB0_DTVA0_DTVB1_EPS0_FDSI0_GRPM1_GRVWA16_GRVWB16_GSUAMB_GLS0_"
                "ISA1201_IU1_K1_LDSTI0_LBSPPA256_LBSPPB0_LBSPPM0_LPA32_LPB0_LPM0_LRVW16_"
                "LWPMn1_MIAV1_MIWT4_4_MO40_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_"
                "NLCB2_ONLL0_PGR2_PLR1_PKA0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW4_SK0_SKFTR0_"
                "SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_VSn1_VWA4_VWB4_WSGRA0_"
                "WSGRB0_WS32_WG32_4_1";
            const char *co_env = getenv("QIMG_FP8_VENDOR_CO");
            const char *cands[3]; int nc = 0;
            if (co_env && co_env[0]) cands[nc++] = co_env;
            cands[nc++] = "../fp8/fp8_kernel_gfx1201.co";
            cands[nc++] = "rdna4/fp8/fp8_kernel_gfx1201.co";
            for (int ci = 0; ci < nc && !r->use_fp8_vendor; ci++) {
                FILE *fp = fopen(cands[ci], "rb");
                if (!fp) continue;
                fclose(fp);
                if (hipModuleLoad(&r->fp8_vendor_mod, cands[ci]) != hipSuccess) continue;
                if (hipModuleGetFunction(&r->fn_fp8_vendor, r->fp8_vendor_mod, VSYM) != hipSuccess) {
                    r->fn_fp8_vendor = NULL;
                    continue;
                }
                if (hipMalloc(&r->d_scale_one, 4 * sizeof(float)) == hipSuccess) {
                    float ones[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                    hipMemcpy(r->d_scale_one, ones, sizeof(ones), hipMemcpyHostToDevice);
                    r->use_fp8_vendor = 1;
                    if (verbose)
                        fprintf(stderr, "hip_qimg: FP8xFP8 using vendor Tensile kernel (%s)\n", cands[ci]);
                }
            }
            if (!r->use_fp8_vendor && verbose)
                fprintf(stderr, "hip_qimg: vendor FP8 .co not found; using hand-written FP8xFP8 kernel\n");
        }
    }

    /* BF16 WMMA flash-attention (gfx12, head_dim=128). Opt-in via QIMG_BF16_ATTN=1
     * (matches CUDA sibling's naming). Default ON when the kernel loaded. */
    r->use_attn_wmma = 0;
    if (r->fn_flash_attn_wmma) {
        const char *v = getenv("QIMG_BF16_ATTN");
        int enable = 1;
        if (v) enable = !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0);
        if (enable) {
            r->use_attn_wmma = 1;
            if (verbose)
                fprintf(stderr, "hip_qimg: BF16 WMMA flash-attention enabled\n");
        }
    }

    /* v2 flash-attention: persistent-Q-in-registers + double-buffered K/V (gfx12,
     * head_dim=128). DEFAULT ON when the kernel is loaded and WMMA attention is
     * enabled — numerically bit-identical to v1 but ~10% faster at 1024² (it freed
     * the 16 KB smQ so the load-latency hiding costs no occupancy). QIMG_ATTN_V2=0
     * reverts to the v1 kernel; QIMG_BF16_ATTN=0 (which clears use_attn_wmma) drops
     * all WMMA attention to the scalar fallback — and correctly suppresses v2 too,
     * because the default-enable hangs off the same use_attn_wmma umbrella. */
    r->use_attn_wmma_pq = 0;
    if (r->fn_flash_attn_wmma_pq && r->use_attn_wmma) {
        const char *v = getenv("QIMG_ATTN_V2");
        int enable = 1;
        if (v) enable = !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0);
        if (enable) {
            r->use_attn_wmma_pq = 1;
            if (verbose)
                fprintf(stderr, "hip_qimg: v2 flash-attention (persistent-Q + double-buffer) enabled [default]\n");
        }
    }

    /* v3 flash-attention: software-pipelined ("FA3-idea") lookahead-QK + triple-buffered
     * K/V (gfx12, head_dim=128). Opt-in A/B path via QIMG_ATTN_V3=1; default OFF. */
    r->use_attn_wmma_sp = 0;
    if (r->fn_flash_attn_wmma_sp) {
        const char *v = getenv("QIMG_ATTN_V3");
        if (v && !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0)) {
            r->use_attn_wmma_sp = 1;
            if (verbose)
                fprintf(stderr, "hip_qimg: v3 flash-attention (software-pipelined lookahead-QK) enabled\n");
        }
    }

    /* BF16 WMMA VAE conv2d (gfx12). Default ON when kernels loaded; opt out via QIMG_VAE_WMMA=0. */
    r->use_vae_wmma = 0;
    if (r->fn_vae_conv2d_3x3_wmma && r->fn_vae_conv2d_1x1_wmma) {
        const char *v = getenv("QIMG_VAE_WMMA");
        int enable = 1;
        if (v) enable = !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0);
        if (enable) {
            r->use_vae_wmma = 1;
            if (verbose)
                fprintf(stderr, "hip_qimg: BF16 WMMA VAE conv2d enabled\n");
        }
    }

    if (verbose) fprintf(stderr, "hip_qimg: kernels compiled OK\n");
    if (getenv("QIMG_INT4_SELFTEST")) qimg_int4_g16_selftest(r);
    return r;
}

/* Fill dst with the active HIP device's marketing name (e.g. "AMD Radeon RX
 * 9070 XT"). Returns 0 on success. Used by callers to prove the GEMM/VAE work
 * actually targets the GPU. */
int hip_qimg_device_name(const hip_qimg_runner *r, char *dst, size_t cap) {
    if (!r || !dst || cap == 0) return -1;
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, r->device_id) != hipSuccess) {
        snprintf(dst, cap, "(unknown)");
        return -1;
    }
    snprintf(dst, cap, "%s", props.name);
    return 0;
}

/* ---- Load DiT ---- */

/* The 12 logical linears per transformer block, in the order tools/nunchaku_convert_logical.py emits them
 * (fused QKV split into separate q/k/v slots); the per-linear key is "transformer_blocks.{b}.{suffix}". */
#define QIMG_INT4_PER_BLOCK 12
static const char *const qimg_int4_linear_suffix[QIMG_INT4_PER_BLOCK] = {
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2",
};

/* Load a Nunchaku/SVDQuant DiT in the offline-converted "logical" INT4 layout (W4A16). All blocks are
 * resident — the ~12 GB INT4 model fits 16 GB, so the block-streaming the FP8 path needs is gone. This loads
 * the per-block quantized-linear descriptors (the novel part); BF16 globals/norms/biases and the modulation
 * still reuse the existing global-upload flow shared with hip_qimg_load_dit (wired in a following step). */
int hip_qimg_load_dit_int4(hip_qimg_runner *r, const char *path) {
    fprintf(stderr, "hip_qimg: loading logical-INT4 DiT %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->dit_st = st; r->use_int4 = 1;
    size_t free_entry = 0, vram_total = 0; hipMemGetInfo(&free_entry, &vram_total);  /* for the residency delta */
    r->dim = 3072; r->n_heads = 24; r->head_dim = 128;
    r->in_ch = 64; r->txt_dim = 3584; r->mlp_h = 12288;
    r->n_blocks = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *bp = strstr(safetensors_name(st, i), "transformer_blocks.");
        if (bp) { int blk = atoi(bp + 19); if (blk + 1 > r->n_blocks) r->n_blocks = blk + 1; }
    }
    r->int4_linears = (qimg_int4_linear *)calloc((size_t)r->n_blocks * QIMG_INT4_PER_BLOCK, sizeof(qimg_int4_linear));
    if (!r->int4_linears) return -1;
    int loaded = 0;
    for (int b = 0; b < r->n_blocks; b++) {
        for (int j = 0; j < QIMG_INT4_PER_BLOCK; j++) {
            char key[160];
            snprintf(key, sizeof(key), "transformer_blocks.%d.%s", b, qimg_int4_linear_suffix[j]);
            if (qimg_upload_int4_linear(st, key, &r->int4_linears[(size_t)b * QIMG_INT4_PER_BLOCK + j]) == 0) loaded++;
        }
    }
    /* BF16 globals (converter passthrough) — reuse the auto-dtype path: all BF16 here -> f32 upload. */
    r->d_img_in_w = qimg_upload_weight_auto(r, st, "img_in.weight", &r->is_fp8_img_in);
    r->d_img_in_b = qimg_st_upload_f32(st, "img_in.bias");
    r->d_txt_in_w = qimg_upload_weight_auto(r, st, "txt_in.weight", &r->is_fp8_txt_in);
    r->d_txt_in_b = qimg_st_upload_f32(st, "txt_in.bias");
    r->d_txt_norm_w = qimg_st_upload_f32(st, "txt_norm.weight");
    r->d_t_fc1_w = qimg_upload_weight_auto(r, st, "time_text_embed.timestep_embedder.linear_1.weight", &r->is_fp8_t_fc1);
    r->d_t_fc1_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_1.bias");
    r->d_t_fc2_w = qimg_upload_weight_auto(r, st, "time_text_embed.timestep_embedder.linear_2.weight", &r->is_fp8_t_fc2);
    r->d_t_fc2_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_2.bias");
    r->d_norm_out_w = qimg_upload_weight_auto(r, st, "norm_out.linear.weight", &r->is_fp8_norm_out);
    r->d_norm_out_b = qimg_st_upload_f32(st, "norm_out.linear.bias");
    r->d_proj_out_w = qimg_upload_weight_auto(r, st, "proj_out.weight", &r->is_fp8_proj_out);
    r->d_proj_out_b = qimg_st_upload_f32(st, "proj_out.bias");
    if (!r->d_img_in_w || !r->d_txt_in_w || !r->d_proj_out_w)
        fprintf(stderr, "hip_qimg: int4 globals incomplete (some NULL); render needs all BF16 globals\n");
    /* Per-block BF16 passthrough (norms + modulation): qkv/mlp come from int4 descriptors, so blocks fully resident. */
    r->gpu_blocks = (qimg_block_gpu *)calloc((size_t)r->n_blocks, sizeof(qimg_block_gpu)); r->n_preloaded = r->n_blocks;
    r->int4_mod = (qimg_int4_linear *)calloc((size_t)r->n_blocks * 2, sizeof(qimg_int4_linear));
    for (int b = 0; b < r->n_blocks; b++) { qimg_block_gpu *g = &r->gpu_blocks[b]; char nm[160];
        #define MW(f,suf) do{snprintf(nm,sizeof nm,"transformer_blocks.%d." suf,b); g->f=qimg_st_upload_f32(st,nm);}while(0)
        MW(norm_q_w,"attn.norm_q.weight"); MW(norm_k_w,"attn.norm_k.weight");
        MW(norm_added_q_w,"attn.norm_added_q.weight"); MW(norm_added_k_w,"attn.norm_added_k.weight");
        #undef MW
        snprintf(nm,sizeof nm,"transformer_blocks.%d.img_mod.1",b); int e1=qimg_upload_int4_linear(st,nm,&r->int4_mod[2*b]);
        snprintf(nm,sizeof nm,"transformer_blocks.%d.txt_mod.1",b); int e2=qimg_upload_int4_linear(st,nm,&r->int4_mod[2*b+1]);
        if (e1||e2) fprintf(stderr, "hip_qimg: int4 block %d mod/norm incomplete\n", b);
    }
    if (r->verbose) {
        const qimg_int4_linear *s = &r->int4_linears[0];  /* sample: block 0, attn.to_q */
        fprintf(stderr, "hip_qimg: logical-INT4 — %d blocks, %d/%d linear descriptors loaded; "
                "sample to_q n_out=%d n_in=%d rank=%d group=%d\n",
                r->n_blocks, loaded, r->n_blocks * QIMG_INT4_PER_BLOCK, s->n_out, s->n_in, s->rank, s->group_size);
        size_t free_now = 0, t = 0; hipMemGetInfo(&free_now, &t);
        fprintf(stderr, "hip_qimg: logical-INT4 weights resident: %.0f MB (of %.0f MB VRAM; no block streaming)\n",
                (double)(free_entry - free_now) / 1e6, (double)vram_total / 1e6);
    }
    return (loaded == r->n_blocks * QIMG_INT4_PER_BLOCK) ? 0 : -1;
}

/* Deterministic on-GPU gate for the int4-logical "main" weight dequant. Assumes hip_qimg_load_dit_int4 has
 * run (descriptors resident, r->dit_st host bytes available). Runs fn_dequant_int4_main on block-0 attn.to_q's
 * packed nibbles+scales, copies the dense weight back, and compares element-wise against a host nibble-decode of
 * the *same* bytes — so any mismatch is purely a kernel-indexing bug (the converter's host decode is the proven
 * oracle). Bit-exact expected: identical float ops on identical operands. Returns 0 on success. */
int hip_qimg_test_int4_dequant(hip_qimg_runner *r) {
    if (!r->use_int4 || !r->int4_linears || !r->dit_st || !r->fn_dequant_int4_main) {
        fprintf(stderr, "hip_qimg: int4-dequant test prerequisites missing (load logical-INT4 DiT first)\n");
        return -1;
    }
    const char *key = "transformer_blocks.0.attn.to_q";   /* descriptor slot [0]; an unfused logical bundle */
    const qimg_int4_linear *L = &r->int4_linears[0];
    int n_out = L->n_out, n_in = L->n_in, gs = L->group_size, ng = n_in / gs;
    long total = (long)n_out * n_in;

    /* host-side raw bytes the descriptor was built from (still mapped in r->dit_st) */
    st_context *st = (st_context *)r->dit_st;
    char nm[256];
    snprintf(nm, sizeof(nm), "%s.qint4", key);  int iq = safetensors_find(st, nm);
    snprintf(nm, sizeof(nm), "%s.wscale", key); int iw = safetensors_find(st, nm);
    if (iq < 0 || iw < 0) { fprintf(stderr, "hip_qimg: int4-dequant test — host tensors for %s absent\n", key); return -1; }
    const unsigned char *qb = (const unsigned char *)safetensors_data(st, iq);
    const unsigned short *wsb = (const unsigned short *)safetensors_data(st, iw);  /* bf16 scales */
    #define WS(idx) ({unsigned _u=(unsigned)wsb[idx]<<16; float _f; memcpy(&_f,&_u,4); _f;})

    /* GPU dense weight */
    float *d_W = NULL;
    if (hipMalloc(&d_W, (size_t)total * sizeof(float)) != hipSuccess) { fprintf(stderr, "hip_qimg: int4-dequant test hipMalloc failed\n"); return -1; }
    void *smnull = NULL;  /* pure-decode check uses no smooth (bit-exact vs raw nibble*scale) */
    void *args[] = { (void *)&L->qint4, (void *)&L->wscale, &smnull, &d_W, &n_out, &n_in, &gs };
    hipModuleLaunchKernel(r->fn_dequant_int4_main, (unsigned)((total + 255) / 256), 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    hipDeviceSynchronize();
    float *h_W = (float *)malloc((size_t)total * sizeof(float));
    hipMemcpy(h_W, d_W, (size_t)total * sizeof(float), hipMemcpyDeviceToHost);

    /* host reference (the proven converter decode) + max-diff and norm */
    double max_abs_diff = 0.0, sumsq = 0.0; long n_nonfinite = 0;
    for (int o = 0; o < n_out; o++) {
        for (int i = 0; i < n_in; i++) {
            unsigned char byte = qb[(long)o * (n_in / 2) + (i >> 1)];
            int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
            if (nib >= 8) nib -= 16;
            float ref = (float)nib * WS((long)o * ng + (i / gs));
            float got = h_W[(long)o * n_in + i];
            double d = fabs((double)got - (double)ref);
            if (d > max_abs_diff) max_abs_diff = d;
            sumsq += (double)ref * ref;
            if (!isfinite(got)) n_nonfinite++;
        }
    }
    fprintf(stderr, "hip_qimg: int4-dequant gate — %s  Ŵ[%d,%d] group=%d  ||W||=%.1f  max|gpu-host|=%.3g  nonfinite=%ld\n",
            key, n_out, n_in, gs, sqrt(sumsq), max_abs_diff, n_nonfinite);

    /* int4-main forward hookup: feed the dense weight through the F32 GEMM (bias) on a small batch,
     * compare to a host matmul. lora/smoothing deferred — this just proves decode->GEMM wiring. */
    int n_tok = 4; float *xh = (float *)malloc((size_t)n_tok * n_in * sizeof(float));
    for (long k = 0; k < (long)n_tok * n_in; k++) xh[k] = ((k * 1103515245u + 12345u) % 2048) / 2048.0f - 0.5f;
    float *dx = NULL, *dy = NULL; hipMalloc(&dx, (size_t)n_tok*n_in*4); hipMalloc(&dy, (size_t)n_tok*n_out*4);
    hipMemcpy(dx, xh, (size_t)n_tok*n_in*4, hipMemcpyHostToDevice);
    /* GPU: re-dequant with smoothing folded, then GEMM+bias — main + smooth + bias on GPU */
    snprintf(nm, sizeof(nm), "%s.smooth", key); int ism = safetensors_find(st, nm); const float *smh = (const float *)safetensors_data(st, ism);
    void *args2[] = { (void *)&L->qint4, (void *)&L->wscale, (void *)&L->smooth, &d_W, &n_out, &n_in, &gs };
    hipModuleLaunchKernel(r->fn_dequant_int4_main, (unsigned)((total + 255) / 256), 1, 1, 256, 1, 1, 0, NULL, args2, NULL);
    op_gemm(r, dy, d_W, dx, L->bias, n_out, n_in, n_tok);  /* main+smooth+bias */
    /* + rank-128 low-rank residual on the RAW activation: y += lora_up @ (lora_down @ x). bf16->f32 for the f32 GEMMs. */
    int rk = L->rank; float *ldf = (float*)malloc((size_t)rk*n_in*4), *luf = (float*)malloc((size_t)n_out*rk*4);
    const uint16_t *ldb = (const uint16_t*)safetensors_data(st, safetensors_find(st, (snprintf(nm,sizeof nm,"%s.lora_down",key),nm)));
    const uint16_t *lub = (const uint16_t*)safetensors_data(st, safetensors_find(st, (snprintf(nm,sizeof nm,"%s.lora_up",key),nm)));
    for (long k=0;k<(long)rk*n_in;k++){unsigned u=ldb[k]<<16; memcpy(&ldf[k],&u,4);} for (long k=0;k<(long)n_out*rk;k++){unsigned u=lub[k]<<16; memcpy(&luf[k],&u,4);}
    float *dld=0,*dlu=0,*dt=0,*dly=0; hipMalloc(&dld,(size_t)rk*n_in*4);hipMalloc(&dlu,(size_t)n_out*rk*4);hipMalloc(&dt,(size_t)n_tok*rk*4);hipMalloc(&dly,(size_t)n_tok*n_out*4);
    hipMemcpy(dld,ldf,(size_t)rk*n_in*4,hipMemcpyHostToDevice); hipMemcpy(dlu,luf,(size_t)n_out*rk*4,hipMemcpyHostToDevice);
    op_gemm(r, dt, dld, dx, NULL, rk, n_in, n_tok); op_gemm(r, dly, dlu, dt, NULL, n_out, rk, n_tok);
    hipDeviceSynchronize();
    float *yh = (float *)malloc((size_t)n_tok*n_out*4); hipMemcpy(yh, dy, (size_t)n_tok*n_out*4, hipMemcpyDeviceToHost);
    float *lyh = (float *)malloc((size_t)n_tok*n_out*4); hipMemcpy(lyh, dly, (size_t)n_tok*n_out*4, hipMemcpyDeviceToHost);
    for (long k=0;k<(long)n_tok*n_out;k++) yh[k]+=lyh[k]; free(lyh);  /* GPU lora GEMMs + accumulate */
    float *bh = (float *)malloc(n_out*4); hipMemcpy(bh, L->bias, n_out*4, hipMemcpyDeviceToHost);
    double dot = 0, ng2 = 0, rg2 = 0;
    for (int t = 0; t < n_tok; t++) for (int o = 0; o < n_out; o++) {
        double acc = 0; for (int i = 0; i < n_in; i++) acc += (double)(h_W[(long)o*n_in+i]/smh[i]) * xh[(long)t*n_in+i];
        for (int rkk=0; rkk<rk; rkk++){ double d=0; for(int i=0;i<n_in;i++) d+=(double)ldf[(long)rkk*n_in+i]*xh[(long)t*n_in+i]; acc+=luf[(long)o*rk+rkk]*d; }
        double ref = acc + bh[o]; float got = yh[(long)t*n_out+o]; dot += ref*got; rg2 += ref*ref; ng2 += (double)got*got;
    }
    fprintf(stderr, "hip_qimg: int4 full linear — y[%d,%d] cos(gpu,host)=%.6f (main+smooth+bias+lora%d)\n",
            n_tok, n_out, dot / (sqrt(rg2)*sqrt(ng2) + 1e-30), rk);
    /* fused kernel main+smooth+bias (no lora) vs host main+bias */
    hipModuleLaunchKernel(r->fn_gemm_int4w,(unsigned)((n_out+127)/128),(unsigned)((n_tok+127)/128),1,256,1,1,0,NULL,
        (void*[]){&dy,(void*)&L->qint4,&dx,(void*)&L->bias,(void*)&L->wscale,(void*)&L->smooth,&n_out,&n_in,&n_tok},NULL);
    hipDeviceSynchronize(); float *fy=(float*)malloc((size_t)n_tok*n_out*4); hipMemcpy(fy,dy,(size_t)n_tok*n_out*4,hipMemcpyDeviceToHost);
    double fd=0,fg=0,fr=0; for(int t=0;t<n_tok;t++)for(int o=0;o<n_out;o++){double a=bh[o];for(int i=0;i<n_in;i++)a+=(double)(h_W[(long)o*n_in+i]/smh[i])*xh[(long)t*n_in+i];double g=fy[(long)t*n_out+o];fd+=a*g;fr+=a*a;fg+=g*g;}
    fprintf(stderr,"hip_qimg: fused int4 GEMM cos(fused,host main+smooth+bias)=%.6f\n",fd/(sqrt(fr)*sqrt(fg)+1e-30)); free(fy);
    free(bh); free(ldf); free(luf); hipFree(dld); hipFree(dlu); hipFree(dt); hipFree(dly);
    free(xh); free(yh); hipFree(dx); hipFree(dy);
    free(h_W); hipFree(d_W);
    return (max_abs_diff == 0.0 && n_nonfinite == 0) ? 0 : -1;
}

int hip_qimg_load_dit(hip_qimg_runner *r, const char *path) {
    fprintf(stderr, "hip_qimg: loading DiT %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->dit_st = st;

    r->dim = 3072; r->n_heads = 24; r->head_dim = 128;
    r->in_ch = 64; r->txt_dim = 3584; r->mlp_h = 12288;

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

    /* INT8 SmoothQuant (W8A8) detection: if .weight is I8, stream int8 weight bytes via the fp8
     * byte path and keep the small per-linear scales resident. Must precede global upload so
     * use_fp8=0 routes the bf16 globals through F32. */
    if (r->fn_quant_act_int8 && r->fn_gemm_w8a8) {
        int qi = safetensors_find(st, "transformer_blocks.0.attn.to_q.weight");
        if (qi >= 0 && strcmp(safetensors_dtype(st, qi), "I8") == 0) {
            r->use_int8 = 1; r->use_fp8 = 0; r->use_wmma = 0; r->use_fp8_fp8w = 0;
            r->use_int8_smooth = (safetensors_find(st, "transformer_blocks.0.attn.to_q.smooth_scale") >= 0);
            fprintf(stderr, "hip_qimg: INT8 W8A8 path%s (per-token act quant + per-out wscale; int8 weights streamed)\n",
                    r->use_int8_smooth ? " + SmoothQuant" : "");
        }
    }

    /* Upload global weights. Each *.weight may be FP8 or BF16/F32 depending on
     * the checkpoint (mixed-dtype like unsloth/Qwen-Image-2512-FP8). The auto
     * helper sets the per-weight is_fp8 flag, which is_fp8_<name> later uses
     * for FP8 vs F32 GEMM dispatch. Biases/norms always go through F32. */
    r->d_img_in_w = qimg_upload_weight_auto(r, st, "img_in.weight", &r->is_fp8_img_in);
    r->d_img_in_b = qimg_st_upload_f32(st, "img_in.bias");
    r->d_txt_in_w = qimg_upload_weight_auto(r, st, "txt_in.weight", &r->is_fp8_txt_in);
    r->d_txt_in_b = qimg_st_upload_f32(st, "txt_in.bias");
    r->d_txt_norm_w = qimg_st_upload_f32(st, "txt_norm.weight");
    r->d_t_fc1_w = qimg_upload_weight_auto(r, st,
        "time_text_embed.timestep_embedder.linear_1.weight", &r->is_fp8_t_fc1);
    r->d_t_fc1_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_1.bias");
    r->d_t_fc2_w = qimg_upload_weight_auto(r, st,
        "time_text_embed.timestep_embedder.linear_2.weight", &r->is_fp8_t_fc2);
    r->d_t_fc2_b = qimg_st_upload_f32(st, "time_text_embed.timestep_embedder.linear_2.bias");
    r->d_norm_out_w = qimg_upload_weight_auto(r, st,
        "norm_out.linear.weight", &r->is_fp8_norm_out);
    r->d_norm_out_b = qimg_st_upload_f32(st, "norm_out.linear.bias");
    r->d_proj_out_w = qimg_upload_weight_auto(r, st, "proj_out.weight", &r->is_fp8_proj_out);
    r->d_proj_out_b = qimg_st_upload_f32(st, "proj_out.bias");

    /* Diagnostic: report which globals are FP8 vs F32 */
    if (r->verbose) {
        int n_f32 = (!r->is_fp8_img_in) + (!r->is_fp8_txt_in) + (!r->is_fp8_t_fc1)
                  + (!r->is_fp8_t_fc2) + (!r->is_fp8_norm_out) + (!r->is_fp8_proj_out);
        if (n_f32 > 0) {
            fprintf(stderr, "hip_qimg: %d/6 global weights are F32 (BF16-stored): "
                    "img_in=%s txt_in=%s t_fc1=%s t_fc2=%s norm_out=%s proj_out=%s\n",
                    n_f32,
                    r->is_fp8_img_in ? "fp8" : "f32",
                    r->is_fp8_txt_in ? "fp8" : "f32",
                    r->is_fp8_t_fc1  ? "fp8" : "f32",
                    r->is_fp8_t_fc2  ? "fp8" : "f32",
                    r->is_fp8_norm_out ? "fp8" : "f32",
                    r->is_fp8_proj_out ? "fp8" : "f32");
        }
    }

    /* Bail early on any failed global weight upload — this happens when the
     * checkpoint is mixed-dtype (e.g. unsloth/Qwen-Image-2512-FP8 stores some
     * global weights as BF16 instead of FP8) and `qimg_st_upload_fp8_raw`
     * returns NULL with a printed error. Without this check the runner would
     * crash or hang in the GEMM kernel with a NULL device pointer. */
    {
        void *globals[] = {
            r->d_img_in_w,  r->d_img_in_b,  r->d_txt_in_w,  r->d_txt_in_b,
            r->d_txt_norm_w, r->d_t_fc1_w,  r->d_t_fc1_b,  r->d_t_fc2_w,
            r->d_t_fc2_b,   r->d_norm_out_w, r->d_norm_out_b,
            r->d_proj_out_w, r->d_proj_out_b
        };
        const char *names[] = {
            "img_in.weight", "img_in.bias", "txt_in.weight", "txt_in.bias",
            "txt_norm.weight", "time_text_embed.timestep_embedder.linear_1.weight",
            "time_text_embed.timestep_embedder.linear_1.bias",
            "time_text_embed.timestep_embedder.linear_2.weight",
            "time_text_embed.timestep_embedder.linear_2.bias",
            "norm_out.linear.weight", "norm_out.linear.bias",
            "proj_out.weight", "proj_out.bias"
        };
        for (int i = 0; i < 13; i++) {
            if (!globals[i]) {
                fprintf(stderr, "hip_qimg: global weight '%s' failed to upload — aborting load_dit.\n",
                        names[i]);
                return -1;
            }
        }
    }

    /* INT8: load the small per-linear scales (weight_scale[n_out] + smooth_scale[n_in]) resident
     * (~40 MB for all 60 blocks); only the int8 weight bytes are streamed. */
    if (r->use_int8) {
        size_t n = (size_t)r->n_blocks * QIMG_I8_PER_BLOCK;
        r->i8_ws = (void **)calloc(n, sizeof(void *));
        r->i8_sm = (void **)calloc(n, sizeof(void *));
        int miss = 0;
        for (int b = 0; b < r->n_blocks; b++)
            for (int s = 0; s < QIMG_I8_PER_BLOCK; s++) {
                char nm[256]; size_t ix = (size_t)b * QIMG_I8_PER_BLOCK + s;
                snprintf(nm, sizeof(nm), "transformer_blocks.%d.%s.weight_scale", b, qimg_i8_suffix[s]);
                r->i8_ws[ix] = qimg_st_upload_f32(st, nm);
                if (!r->i8_ws[ix]) miss++;
                if (r->use_int8_smooth) {
                    snprintf(nm, sizeof(nm), "transformer_blocks.%d.%s.smooth_scale", b, qimg_i8_suffix[s]);
                    r->i8_sm[ix] = qimg_st_upload_f32(st, nm);
                }
            }
        if (miss) fprintf(stderr, "hip_qimg: INT8 — %d weight_scale tensors missing (render will be wrong)\n", miss);
        else if (r->verbose) fprintf(stderr, "hip_qimg: INT8 — loaded %zu weight_scale%s vectors resident\n",
                                     n, r->use_int8_smooth ? "+smooth_scale" : "");
    }

    /* Preload as many blocks as fit in VRAM */
    {
        size_t free_mem = 0, total_mem = 0;
        hipMemGetInfo(&free_mem, &total_mem);
        /* FP8: ~324M params × 1 byte = ~324 MB/block (+ ~20MB F32 biases/norms)
         * F32:  ~324M params × 4 bytes = ~1296 MB/block */
        size_t block_bytes = (r->use_fp8 || r->use_int8) ? 344ULL * 1024 * 1024
                                        : 1296ULL * 1024 * 1024;
        /* Workspace budget for per-step activation allocations. At 512×512
         * peak usage is ~300 MB (Q/K/V 72 MB, scratch3 48 MB, scratch1/2
         * 18 MB each, img/txt 18 MB, joint attn_out 18 MB + slack). Keeping
         * a 512 MB reserve lets us preload ~4 more blocks than the old
         * conservative 2 GB reserve, eliminating ~340 ms/step of PCIe
         * streaming at 512×512 (and ~300 ms/step at 256×256). An env knob
         * `QIMG_WORKSPACE_MB` lets a caller raise this if they run at
         * higher resolutions where scratch grows. */
        size_t workspace = 512ULL * 1024 * 1024;
        {
            const char *v = getenv("QIMG_WORKSPACE_MB");
            if (v) {
                long mb = atol(v);
                if (mb >= 128 && mb <= 8192) workspace = (size_t)mb * 1024 * 1024;
            }
        }
        int max_preload = (free_mem > workspace)
            ? (int)((free_mem - workspace) / block_bytes) : 0;
        if (max_preload > r->n_blocks) max_preload = r->n_blocks;

        r->gpu_blocks = (qimg_block_gpu *)calloc((size_t)r->n_blocks,
                                                  sizeof(qimg_block_gpu));
        r->n_preloaded = max_preload;
        fprintf(stderr, "hip_qimg: preloading %d/%d blocks to GPU "
                "(%.1f GB free, %.0f MB/block)\n",
                max_preload, r->n_blocks,
                (float)free_mem / (1<<30), (float)block_bytes / (1<<20));

        for (int i = 0; i < max_preload; i++) {
            if (qimg_load_block(r, i, &r->gpu_blocks[i]) != 0) {
                fprintf(stderr, "hip_qimg: stopped preloading at block %d (OOM)\n", i);
                r->n_preloaded = i;
                break;
            }
        }
        hipDeviceSynchronize();

        hipMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "hip_qimg: after preload: %.1f GB free\n",
                (float)free_mem / (1<<30));
    }

    /* Streamed blocks (>= n_preloaded) reuse a persistent buffer + async copy
     * instead of malloc/copy/free every step. Default ON; QIMG_BLOCK_STREAM=0
     * reverts to the per-step malloc/free path. */
    r->use_block_stream_db = (r->n_preloaded < r->n_blocks);
    {
        const char *v = getenv("QIMG_BLOCK_STREAM");
        if (v && (strcmp(v, "0") == 0 || strcmp(v, "false") == 0))
            r->use_block_stream_db = 0;
    }
    if (r->use_block_stream_db && r->verbose)
        fprintf(stderr, "hip_qimg: persistent streamed-block buffer enabled (%d blocks streamed)\n",
                r->n_blocks - r->n_preloaded);

    fprintf(stderr, "hip_qimg: loaded %d blocks, dim=%d\n", r->n_blocks, r->dim);
    return 0;
}

/* ---- Load VAE ---- */

int hip_qimg_load_vae(hip_qimg_runner *r, const char *path) {
    fprintf(stderr, "hip_qimg: loading VAE %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    r->vae_st = st;
    fprintf(stderr, "hip_qimg: VAE loaded (%d tensors)\n", st->n_tensors);
    return 0;
}

/* Write collected calibration stats as a minimal safetensors of F32 [n_in] vectors keyed
 * transformer_blocks.{b}.{suffix}.amax — consumed by tools/svdquant_from_bf16.py --calib. */
static void qimg_calib_dump(hip_qimg_runner *r) {
    if (!r->calib_dump_path) return;
    size_t total_floats = 0; int present = 0;
    for (int idx = 0; idx < QIMG_CALIB_MAXSLOT; idx++)
        if (r->calib_amax[idx]) { total_floats += (size_t)r->calib_nin[idx]; present++; }
    for (int b = 0; b < 64; b++)                          /* tsilu -> img_mod.1 + txt_mod.1 (data written twice) */
        if (r->calib_tsilu[b]) { total_floats += (size_t)r->dim * 2; present += 2; }
    if (!present) { fprintf(stderr, "hip_qimg: calib dump: no stats collected (run with a bf16/fp8 DiT)\n"); return; }
    char *json = (char *)malloc((size_t)present * 256 + 64);
    float *data = (float *)malloc(total_floats * 4);
    if (!json || !data) { free(json); free(data); fprintf(stderr, "hip_qimg: calib dump: OOM\n"); return; }
    size_t jp = 0, foff = 0; int first = 1;
    jp += sprintf(json + jp, "{");
    for (int b = 0; b < QIMG_CALIB_MAXSLOT / QIMG_INT4_PER_BLOCK; b++)
        for (int s = 0; s < QIMG_INT4_PER_BLOCK; s++) {
            int idx = b * QIMG_INT4_PER_BLOCK + s;
            if (!r->calib_amax[idx]) continue;
            int nin = r->calib_nin[idx];
            hipMemcpy(data + foff, r->calib_amax[idx], (size_t)nin * 4, hipMemcpyDeviceToHost);
            jp += sprintf(json + jp,
                "%s\"transformer_blocks.%d.%s.amax\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[%zu,%zu]}",
                first ? "" : ",", b, qimg_int4_linear_suffix[s], nin, foff * 4, (foff + (size_t)nin) * 4);
            foff += (size_t)nin; first = 0;
        }
    for (int b = 0; b < 64; b++) {                        /* modulation: same d_t_silu amax for img_mod.1 + txt_mod.1 */
        if (!r->calib_tsilu[b]) continue;
        int nin = r->dim;
        float *hbuf = data + foff;
        hipMemcpy(hbuf, r->calib_tsilu[b], (size_t)nin * 4, hipMemcpyDeviceToHost);
        jp += sprintf(json + jp, "%s\"transformer_blocks.%d.img_mod.1.amax\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[%zu,%zu]}",
                      first ? "" : ",", b, nin, foff * 4, (foff + (size_t)nin) * 4);
        foff += (size_t)nin; first = 0;
        memcpy(data + foff, hbuf, (size_t)nin * 4);
        jp += sprintf(json + jp, ",\"transformer_blocks.%d.txt_mod.1.amax\":{\"dtype\":\"F32\",\"shape\":[%d],\"data_offsets\":[%zu,%zu]}",
                      b, nin, foff * 4, (foff + (size_t)nin) * 4);
        foff += (size_t)nin;
    }
    jp += sprintf(json + jp, "}");
    while (jp % 8) json[jp++] = ' ';                 /* pad header to 8-byte alignment */
    FILE *f = fopen(r->calib_dump_path, "wb");
    if (!f) { fprintf(stderr, "hip_qimg: calib dump: cannot open %s\n", r->calib_dump_path); free(json); free(data); return; }
    uint64_t hlen = (uint64_t)jp;
    fwrite(&hlen, 8, 1, f); fwrite(json, 1, jp, f); fwrite(data, 4, foff, f); fclose(f);
    fprintf(stderr, "hip_qimg: calib dump wrote %d vectors (%zu floats) -> %s\n", present, foff, r->calib_dump_path);
    free(json); free(data);
}

/* ---- Free ---- */

void hip_qimg_free(hip_qimg_runner *r) {
    if (!r) return;
    qimg_print_gemm_summary(r);
    qimg_calib_dump(r);
    for (int idx = 0; idx < QIMG_CALIB_MAXSLOT; idx++)
        if (r->calib_amax[idx]) { hipFree(r->calib_amax[idx]); r->calib_amax[idx] = NULL; }
    for (int b = 0; b < 64; b++)
        if (r->calib_tsilu[b]) { hipFree(r->calib_tsilu[b]); r->calib_tsilu[b] = NULL; }
    if (r->i8_ws || r->i8_sm) {
        size_t n = (size_t)r->n_blocks * QIMG_I8_PER_BLOCK;
        for (size_t i = 0; i < n; i++) {
            if (r->i8_ws && r->i8_ws[i]) hipFree(r->i8_ws[i]);
            if (r->i8_sm && r->i8_sm[i]) hipFree(r->i8_sm[i]);
        }
        free(r->i8_ws); free(r->i8_sm); r->i8_ws = r->i8_sm = NULL;
    }
    if (r->d_xq_int8) { hipFree(r->d_xq_int8); r->d_xq_int8 = NULL; }
    if (r->d_x_iscale) { hipFree(r->d_x_iscale); r->d_x_iscale = NULL; }
    qimg_free_block(&r->stream_blk[0]);
    qimg_free_block(&r->stream_blk[1]);
    if (r->gpu_blocks) {
        for (int i = 0; i < r->n_preloaded; i++)
            qimg_free_block(&r->gpu_blocks[i]);
        free(r->gpu_blocks);
    }
    /* Free global weights */
    void **globals[] = {
        &r->d_img_in_w, &r->d_img_in_b, &r->d_txt_in_w, &r->d_txt_in_b,
        &r->d_txt_norm_w, &r->d_t_fc1_w, &r->d_t_fc1_b, &r->d_t_fc2_w,
        &r->d_t_fc2_b, &r->d_norm_out_w, &r->d_norm_out_b,
        &r->d_proj_out_w, &r->d_proj_out_b
    };
    for (int i = 0; i < 13; i++) {
        if (*globals[i]) { hipFree(*globals[i]); *globals[i] = NULL; }
    }
    if (r->d_act_fp8) { hipFree(r->d_act_fp8); r->d_act_fp8 = NULL; }
    if (r->d_act_scales) { hipFree(r->d_act_scales); r->d_act_scales = NULL; }
    if (r->d_fa_qfp8) { hipFree(r->d_fa_qfp8); r->d_fa_qfp8 = NULL; }
    if (r->d_fa_kfp8) { hipFree(r->d_fa_kfp8); r->d_fa_kfp8 = NULL; }
    if (r->d_fa_vfp8) { hipFree(r->d_fa_vfp8); r->d_fa_vfp8 = NULL; }
    if (r->d_fa_qs) { hipFree(r->d_fa_qs); r->d_fa_qs = NULL; }
    if (r->d_fa_ks) { hipFree(r->d_fa_ks); r->d_fa_ks = NULL; }
    if (r->d_fa_vs) { hipFree(r->d_fa_vs); r->d_fa_vs = NULL; }
    if (r->dit_st) safetensors_close((st_context *)r->dit_st);
    if (r->vae_st) safetensors_close((st_context *)r->vae_st);
    if (r->mod) hipModuleUnload(r->mod);
    free(r);
}

/* Free resident DiT weights (int4/bf16) to reclaim VRAM before VAE decode. The
 * int4 model is 14 GB resident — keeping it through decode leaves too little for
 * the VAE on a 16 GB card. Safe to call after the denoise loop; DiT is unusable
 * afterward. */
void hip_qimg_unload_dit(hip_qimg_runner *r) {
    if (!r) return;
    hipDeviceSynchronize();
    /* Fused-QKV split linears alias one lora_down/lora_up/smooth across the 3
     * descriptors, so free each unique device pointer exactly once. */
    int n4 = r->int4_linears ? r->n_blocks * QIMG_INT4_PER_BLOCK : 0;
    int nm = r->int4_mod ? r->n_blocks * 2 : 0;
    size_t cap = (size_t)(n4 + nm) * 6 + 8;
    void **seen = (void **)malloc(cap * sizeof(void *)); size_t ns = 0;
    #define FREE1(p) do{ if(p){ size_t _j; for(_j=0;_j<ns;_j++) if(seen[_j]==(void*)(p)) break; \
        if(_j==ns){ hipFree(p); seen[ns++]=(void*)(p);} (p)=NULL; } }while(0)
    for (int i = 0; i < n4; i++) { qimg_int4_linear *L=&r->int4_linears[i];
        FREE1(L->qint4); FREE1(L->wscale); FREE1(L->smooth); FREE1(L->lora_down); FREE1(L->lora_up); FREE1(L->bias); }
    for (int i = 0; i < nm; i++) { qimg_int4_linear *L=&r->int4_mod[i];
        FREE1(L->qint4); FREE1(L->wscale); FREE1(L->smooth); FREE1(L->lora_down); FREE1(L->lora_up); FREE1(L->bias); }
    #undef FREE1
    free(seen);
    free(r->int4_linears); r->int4_linears = NULL;
    free(r->int4_mod); r->int4_mod = NULL;
    if (r->gpu_blocks) {
        for (int i = 0; i < r->n_preloaded; i++) qimg_free_block(&r->gpu_blocks[i]);
        free(r->gpu_blocks); r->gpu_blocks = NULL; r->n_preloaded = 0;
    }
    if (r->i4_ldf) { hipFree(r->i4_ldf); r->i4_ldf = NULL; }
    if (r->i4_luf) { hipFree(r->i4_luf); r->i4_luf = NULL; }
    if (r->i4_dt)  { hipFree(r->i4_dt);  r->i4_dt = NULL; r->i4_dt_cap = 0; }
    if (r->i4_dly) { hipFree(r->i4_dly); r->i4_dly = NULL; r->i4_dly_cap = 0; }
    r->n_blocks = 0; r->use_int4 = 0;
    hipDeviceSynchronize();
    { size_t f=0,t=0; hipMemGetInfo(&f,&t); fprintf(stderr,"hip_qimg: DiT unloaded; %.0f/%.0f MB free\n",f/1e6,t/1e6); }
}


/* ---- DiT single step ---- */

int hip_qimg_dit_step(hip_qimg_runner *r,
                      const float *img_tokens, int n_img,
                      const float *txt_tokens, int n_txt,
                      float timestep, float *out) {
    int dim = r->dim;
    int nh = r->n_heads, hd = r->head_dim;
    int in_ch = r->in_ch, txt_dim = r->txt_dim, mlp_h = r->mlp_h;
    int n_total = n_img + n_txt;

    /* Allocate GPU activation buffers */
    void *d_img = NULL, *d_txt = NULL, *d_t_emb = NULL;
    hipMalloc(&d_img, (size_t)n_img * dim * sizeof(float));
    hipMalloc(&d_txt, (size_t)n_txt * dim * sizeof(float));
    hipMalloc(&d_t_emb, (size_t)dim * sizeof(float));

    /* Upload inputs */
    void *d_img_in = NULL, *d_txt_in = NULL;
    hipMalloc(&d_img_in, (size_t)n_img * in_ch * sizeof(float));
    hipMemcpy(d_img_in, img_tokens, (size_t)n_img * in_ch * sizeof(float), hipMemcpyHostToDevice);
    hipMalloc(&d_txt_in, (size_t)n_txt * txt_dim * sizeof(float));
    hipMemcpy(d_txt_in, txt_tokens, (size_t)n_txt * txt_dim * sizeof(float), hipMemcpyHostToDevice);

    /* BF16 truncate inputs (match ComfyUI) */
    op_bf16_trunc(r, d_img_in, n_img * in_ch);
    op_bf16_trunc(r, d_txt_in, n_txt * txt_dim);

    /* 1. Timestep embedding: sinusoidal(256) → SiLU(GEMM) → GEMM */
    float t_sin[256];
    int half = 128;
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i / (float)half * logf(10000.0f));
        float angle = timestep * freq;
        t_sin[i]        = cosf(angle);
        t_sin[half + i] = sinf(angle);
    }
    void *d_t_sin = NULL;
    hipMalloc(&d_t_sin, 256 * sizeof(float));
    hipMemcpy(d_t_sin, t_sin, 256 * sizeof(float), hipMemcpyHostToDevice);

    qimg_set_gemm_context(r, -1, "time_fc1");
    op_wgemm_bf16_auto(r, r->is_fp8_t_fc1, d_t_emb, r->d_t_fc1_w, d_t_sin, r->d_t_fc1_b, dim, 256, 1);
    op_silu(r, d_t_emb, dim);
    void *d_t_emb2 = NULL;
    hipMalloc(&d_t_emb2, (size_t)dim * sizeof(float));
    qimg_set_gemm_context(r, -1, "time_fc2");
    op_wgemm_bf16_auto(r, r->is_fp8_t_fc2, d_t_emb2, r->d_t_fc2_w, d_t_emb, r->d_t_fc2_b, dim, dim, 1);
    hipFree(d_t_emb); d_t_emb = d_t_emb2;
    hipFree(d_t_sin);

    /* 2. Text input: RMSNorm → Linear */
    if (r->d_txt_norm_w) {
        op_rmsnorm_weighted(r, d_txt_in, r->d_txt_norm_w, n_txt, txt_dim);
    }
    qimg_set_gemm_context(r, -1, "txt_in");
    op_wgemm_bf16_auto(r, r->is_fp8_txt_in, d_txt, r->d_txt_in_w, d_txt_in, r->d_txt_in_b, dim, txt_dim, n_txt);
    hipFree(d_txt_in);

    /* 3. Image input: GEMM(64→3072) */
    qimg_set_gemm_context(r, -1, "img_in");
    op_wgemm_bf16_auto(r, r->is_fp8_img_in, d_img, r->d_img_in_w, d_img_in, r->d_img_in_b, dim, in_ch, n_img);
    hipFree(d_img_in);

    /* BF16 truncation after projection */
    op_bf16_trunc(r, d_img, n_img * dim);
    op_bf16_trunc(r, d_txt, n_txt * dim);
    op_bf16_trunc(r, d_t_emb, dim);

    /* Scratch buffers */
    void *d_scratch1 = NULL, *d_scratch2 = NULL, *d_scratch3 = NULL;
    size_t max_scratch = (size_t)n_total * dim * sizeof(float);
    hipMalloc(&d_scratch1, max_scratch);
    hipMalloc(&d_scratch2, max_scratch);
    size_t ffn_scratch = (size_t)(n_img > n_txt ? n_img : n_txt) * mlp_h * sizeof(float);
    hipMalloc(&d_scratch3, ffn_scratch);

    /* Joint Q/K/V buffers */
    void *d_q = NULL, *d_k = NULL, *d_v = NULL, *d_attn_out = NULL;
    hipMalloc(&d_q, (size_t)n_total * dim * sizeof(float));
    hipMalloc(&d_k, (size_t)n_total * dim * sizeof(float));
    hipMalloc(&d_v, (size_t)n_total * dim * sizeof(float));
    hipMalloc(&d_attn_out, (size_t)n_total * dim * sizeof(float));

    /* RoPE params */
    int hp_rope = (int)sqrtf((float)n_img);
    int wp_rope = n_img / hp_rope;
    float rope_theta = 10000.0f;
    int t_dim_rope = 16, h_dim_rope = 56, w_dim_rope = 56;

    /* Pre-size the int4 LoRA scratch once for this step's largest linear
     * (rank-128 down, mlp_h-wide up, max(n_img,n_txt) tokens), so the per-block
     * grow path never reallocs a buffer the GPU is mid-using. */
    if (r->use_int4) {
        int mt = n_img > n_txt ? n_img : n_txt;
        if (!r->i4_ldf) { hipMalloc(&r->i4_ldf,(size_t)128*12288*4); hipMalloc(&r->i4_luf,(size_t)18432*128*4); }
        size_t need_dt=(size_t)128*mt, need_dly=(size_t)mlp_h*mt;
        if (need_dt > r->i4_dt_cap)  { hipFree(r->i4_dt);  hipMalloc(&r->i4_dt,  need_dt*4);  r->i4_dt_cap=need_dt; }
        if (need_dly> r->i4_dly_cap) { hipFree(r->i4_dly); hipMalloc(&r->i4_dly, need_dly*4); r->i4_dly_cap=need_dly; }
    }

    /* Pre-allocate per-block modulation buffers (avoid malloc/free inside loop) */
    void *d_t_silu = NULL, *d_img_mod = NULL, *d_txt_mod = NULL;
    hipMalloc(&d_t_silu, (size_t)dim * sizeof(float));
    hipMalloc(&d_img_mod, (size_t)6 * dim * sizeof(float));
    hipMalloc(&d_txt_mod, (size_t)6 * dim * sizeof(float));
    if (r->mem_stats_enabled && !r->mem_report_printed) {
        size_t state_bytes = ((size_t)n_img + (size_t)n_txt) * (size_t)dim * sizeof(float)
                           + (size_t)dim * sizeof(float);
        size_t scratch_bytes = 2 * max_scratch + ffn_scratch;
        size_t qkv_bytes = 4 * (size_t)n_total * (size_t)dim * sizeof(float);
        size_t mod_bytes = ((size_t)1 + (size_t)6 + (size_t)6) * (size_t)dim * sizeof(float);
        size_t steady_bytes = state_bytes + scratch_bytes + qkv_bytes + mod_bytes;
        size_t fp8_img_fc1_scratch = (size_t)n_img * (size_t)mlp_h + (size_t)n_img * sizeof(float);
        size_t free_mem = 0, total_mem = 0;
        hipMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr,
                "hip_qimg: DiT workspace: n_img=%d n_txt=%d n_total=%d steady=%.1fMB "
                "(state=%.1f scratch=%.1f qkv/attn=%.1f mod=%.1f) fp8xfp8_max_extra=%.1fMB\n",
                n_img, n_txt, n_total,
                (double)steady_bytes / (1024.0 * 1024.0),
                (double)state_bytes / (1024.0 * 1024.0),
                (double)scratch_bytes / (1024.0 * 1024.0),
                (double)qkv_bytes / (1024.0 * 1024.0),
                (double)mod_bytes / (1024.0 * 1024.0),
                (double)fp8_img_fc1_scratch / (1024.0 * 1024.0));
        fprintf(stderr, "hip_qimg: device memory after DiT workspace alloc: %.1f/%.1f GB free\n",
                (double)free_mem / (1024.0 * 1024.0 * 1024.0),
                (double)total_mem / (1024.0 * 1024.0 * 1024.0));
        r->mem_report_printed = 1;
    }

    /* 4. Process all blocks */
    for (int L = 0; L < r->n_blocks; L++) {
        if (r->verbose && (L % 10 == 0 || L == r->n_blocks - 1))
            fprintf(stderr, "\r  hip_qimg: block %d/%d", L + 1, r->n_blocks);

        /* Use preloaded block if available, otherwise stream on-demand. Streamed
         * blocks reuse a persistent buffer (no per-step malloc/free) and copy
         * async on the default stream so the host queues ahead instead of
         * blocking — the dominant denoise overhead (see attention profile note). */
        qimg_block_gpu blk;
        int need_free = 0;
        if (r->use_int4) {
            /* INT4: weights live in int4_linears/int4_mod; gpu_blocks[L] holds only the
             * per-head norm weights (attn_q_w is intentionally NULL). Use it directly —
             * never stream (the int4 checkpoint has no fp8/bf16 QKV to stream). */
            blk = r->gpu_blocks[L];
        } else if (L < r->n_preloaded && r->gpu_blocks[L].attn_q_w) {
            blk = r->gpu_blocks[L];
        } else if (r->use_block_stream_db) {
            qimg_load_block_s(r, L, &r->stream_blk[0], 0);
            blk = r->stream_blk[0];
        } else {
            memset(&blk, 0, sizeof(blk));
            qimg_load_block(r, L, &blk);
            need_free = 1;
        }

        /* Image modulation: SiLU(t_emb) → Linear → 6×dim */
        hipMemcpyAsync(d_t_silu, d_t_emb, (size_t)dim * sizeof(float),
                       hipMemcpyDeviceToDevice, NULL);
        op_silu(r, d_t_silu, dim);

        /* Calibration: modulation input (d_t_silu, shared by img_mod/txt_mod) per-channel abs-max. */
        if (r->calib_dump_path && r->fn_calib_amax && L >= 0 && L < 64) {
            if (!r->calib_tsilu[L] && hipMalloc(&r->calib_tsilu[L], (size_t)dim * 4) == hipSuccess)
                hipMemset(r->calib_tsilu[L], 0, (size_t)dim * 4);
            if (r->calib_tsilu[L]) {
                int one = 1;
                void *ca[] = {&r->calib_tsilu[L], &d_t_silu, &one, &dim};
                hipModuleLaunchKernel(r->fn_calib_amax, (unsigned)((dim + 255) / 256), 1, 1, 256, 1, 1, 0, NULL, ca, NULL);
            }
        }

        qimg_set_gemm_context(r, L, "img_mod");
        if (r->use_int4) op_int4_linear(r, d_img_mod, d_t_silu, &r->int4_mod[2*L], 1);
        else if (r->use_int8 && r->i8_ws) op_gemm_int8(r, d_img_mod, blk.img_mod_w,
            r->i8_ws[(size_t)L*QIMG_I8_PER_BLOCK+12], r->use_int8_smooth ? r->i8_sm[(size_t)L*QIMG_I8_PER_BLOCK+12] : NULL,
            d_t_silu, blk.img_mod_b, 6 * dim, dim, 1);
        else op_wgemm_bf16(r, d_img_mod, blk.img_mod_w, d_t_silu, blk.img_mod_b, 6 * dim, dim, 1);
        qimg_set_gemm_context(r, L, "txt_mod");
        if (r->use_int4) op_int4_linear(r, d_txt_mod, d_t_silu, &r->int4_mod[2*L+1], 1);
        else if (r->use_int8 && r->i8_ws) op_gemm_int8(r, d_txt_mod, blk.txt_mod_w,
            r->i8_ws[(size_t)L*QIMG_I8_PER_BLOCK+13], r->use_int8_smooth ? r->i8_sm[(size_t)L*QIMG_I8_PER_BLOCK+13] : NULL,
            d_t_silu, blk.txt_mod_b, 6 * dim, dim, 1);
        else op_wgemm_bf16(r, d_txt_mod, blk.txt_mod_w, d_t_silu, blk.txt_mod_b, 6 * dim, dim, 1);

        /* Modulation offsets */
        #define MOD_OFF(base, idx) ((void *)((char *)(base) + (size_t)(idx) * dim * sizeof(float)))
        void *img_sh1 = MOD_OFF(d_img_mod, 0);
        void *img_sc1 = MOD_OFF(d_img_mod, 1);
        void *img_g1  = MOD_OFF(d_img_mod, 2);
        void *img_sh2 = MOD_OFF(d_img_mod, 3);
        void *img_sc2 = MOD_OFF(d_img_mod, 4);
        void *img_g2  = MOD_OFF(d_img_mod, 5);

        void *txt_sh1 = MOD_OFF(d_txt_mod, 0);
        void *txt_sc1 = MOD_OFF(d_txt_mod, 1);
        void *txt_g1  = MOD_OFF(d_txt_mod, 2);
        void *txt_sh2 = MOD_OFF(d_txt_mod, 3);
        void *txt_sc2 = MOD_OFF(d_txt_mod, 4);
        void *txt_g2  = MOD_OFF(d_txt_mod, 5);
        #undef MOD_OFF

        /* adaLN image → d_scratch1 */
        op_adaln(r, d_scratch1, d_img, img_sh1, img_sc1, n_img, dim);
        /* adaLN text → d_scratch2 */
        op_adaln(r, d_scratch2, d_txt, txt_sh1, txt_sc1, n_txt, dim);

        /* Image QKV → offset into joint buffers at [n_txt:] */
        void *d_img_q = (char *)d_q + (size_t)n_txt * dim * sizeof(float);
        void *d_img_k = (char *)d_k + (size_t)n_txt * dim * sizeof(float);
        void *d_img_v = (char *)d_v + (size_t)n_txt * dim * sizeof(float);
        qimg_set_gemm_context(r, L, "img_q");
        op_proj(r, d_img_q, blk.attn_q_w, d_scratch1, blk.attn_q_b, dim, dim, n_img, L, 0);
        qimg_set_gemm_context(r, L, "img_k");
        op_proj(r, d_img_k, blk.attn_k_w, d_scratch1, blk.attn_k_b, dim, dim, n_img, L, 1);
        qimg_set_gemm_context(r, L, "img_v");
        op_proj(r, d_img_v, blk.attn_v_w, d_scratch1, blk.attn_v_b, dim, dim, n_img, L, 2);

        /* Text QKV → offset at [0:n_txt] */
        void *d_txt_q = d_q;
        void *d_txt_k = d_k;
        void *d_txt_v = d_v;
        qimg_set_gemm_context(r, L, "txt_q");
        op_proj(r, d_txt_q, blk.attn_add_q_w, d_scratch2, blk.attn_add_q_b, dim, dim, n_txt, L, 4);
        qimg_set_gemm_context(r, L, "txt_k");
        op_proj(r, d_txt_k, blk.attn_add_k_w, d_scratch2, blk.attn_add_k_b, dim, dim, n_txt, L, 5);
        qimg_set_gemm_context(r, L, "txt_v");
        op_proj(r, d_txt_v, blk.attn_add_v_w, d_scratch2, blk.attn_add_v_b, dim, dim, n_txt, L, 6);

        /* QK RMSNorm */
        op_rmsnorm_ph(r, d_img_q, blk.norm_q_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_img_k, blk.norm_k_w, n_img, nh, hd);
        op_rmsnorm_ph(r, d_txt_q, blk.norm_added_q_w, n_txt, nh, hd);
        op_rmsnorm_ph(r, d_txt_k, blk.norm_added_k_w, n_txt, nh, hd);

        /* RoPE */
        {
            void *rope2d_args[] = {&d_img_q, &d_img_k,
                                   &n_img, &nh, &hd, &hp_rope, &wp_rope,
                                   &t_dim_rope, &h_dim_rope, &w_dim_rope, &rope_theta};
            hipModuleLaunchKernel(r->fn_rope_2d, (unsigned)n_img, 1, 1,
                                  (unsigned)nh, 1, 1, 0, NULL, rope2d_args, NULL);

            int txt_start = hp_rope > wp_rope ? hp_rope / 2 : wp_rope / 2;
            void *rope1d_args[] = {&d_txt_q, &d_txt_k,
                                   &n_txt, &nh, &hd, &txt_start,
                                   &t_dim_rope, &h_dim_rope, &w_dim_rope, &rope_theta};
            hipModuleLaunchKernel(r->fn_rope_1d, (unsigned)n_txt, 1, 1,
                                  (unsigned)nh, 1, 1, 0, NULL, rope1d_args, NULL);
        }

        /* Joint attention */
        op_attn(r, d_attn_out, d_q, d_k, d_v, n_total, nh, hd);

        /* Output projections */
        void *d_img_attn = (char *)d_attn_out + (size_t)n_txt * dim * sizeof(float);
        void *d_txt_attn = d_attn_out;
        qimg_set_gemm_context(r, L, "img_attn_out");
        op_proj(r, d_scratch1, blk.attn_out_w, d_img_attn, blk.attn_out_b, dim, dim, n_img, L, 3);
        qimg_set_gemm_context(r, L, "txt_attn_out");
        op_proj(r, d_scratch2, blk.attn_add_out_w, d_txt_attn, blk.attn_add_out_b, dim, dim, n_txt, L, 7);

        /* Gated residual */
        op_gated_add(r, d_img, d_scratch1, img_g1, n_img, dim);
        op_gated_add(r, d_txt, d_scratch2, txt_g1, n_txt, dim);

        /* MLP: Image (GELU) */
        op_adaln(r, d_scratch1, d_img, img_sh2, img_sc2, n_img, dim);
        qimg_set_gemm_context(r, L, "img_mlp_fc1");
        op_proj(r, d_scratch3, blk.img_mlp_fc1_w, d_scratch1, blk.img_mlp_fc1_b, mlp_h, dim, n_img, L, 8);
        op_gelu(r, d_scratch3, n_img * mlp_h);
        qimg_set_gemm_context(r, L, "img_mlp_fc2");
        op_proj(r, d_scratch1, blk.img_mlp_fc2_w, d_scratch3, blk.img_mlp_fc2_b, dim, mlp_h, n_img, L, 9);
        op_gated_add(r, d_img, d_scratch1, img_g2, n_img, dim);

        /* MLP: Text (GELU) */
        op_adaln(r, d_scratch2, d_txt, txt_sh2, txt_sc2, n_txt, dim);
        qimg_set_gemm_context(r, L, "txt_mlp_fc1");
        op_proj(r, d_scratch3, blk.txt_mlp_fc1_w, d_scratch2, blk.txt_mlp_fc1_b, mlp_h, dim, n_txt, L, 10);
        op_gelu(r, d_scratch3, n_txt * mlp_h);
        qimg_set_gemm_context(r, L, "txt_mlp_fc2");
        op_proj(r, d_scratch2, blk.txt_mlp_fc2_w, d_scratch3, blk.txt_mlp_fc2_b, dim, mlp_h, n_txt, L, 11);
        op_gated_add(r, d_txt, d_scratch2, txt_g2, n_txt, dim);

        /* BF16 truncation (image only — text is tiny, not worth a separate launch) */
        op_bf16_trunc(r, d_img, n_img * dim);

        /* Free on-demand block weights (async — no sync needed, hipFree is ordered) */
        if (need_free) {
            void **ptrs = (void **)&blk;
            int np = sizeof(qimg_block_gpu) / sizeof(void *);
            for (int pi = 0; pi < np; pi++)
                if (ptrs[pi]) hipFree(ptrs[pi]);
        }
    }
    if (r->verbose) fprintf(stderr, "\n");

    /* Free pre-allocated modulation buffers */
    hipFree(d_t_silu); hipFree(d_img_mod); hipFree(d_txt_mod);

    /* 5. Final output: adaLN → proj_out */
    {
        void *d_t_silu = NULL;
        hipMalloc(&d_t_silu, (size_t)dim * sizeof(float));
        hipMemcpy(d_t_silu, d_t_emb, (size_t)dim * sizeof(float), hipMemcpyDeviceToDevice);
        op_silu(r, d_t_silu, dim);
        void *d_final_mod = NULL;
        hipMalloc(&d_final_mod, (size_t)2 * dim * sizeof(float));
        qimg_set_gemm_context(r, -1, "norm_out_mod");
        op_wgemm_bf16_auto(r, r->is_fp8_norm_out, d_final_mod,
                           r->d_norm_out_w, d_t_silu, r->d_norm_out_b,
                           2 * dim, dim, 1);
        hipFree(d_t_silu);

        void *f_scale = d_final_mod;
        void *f_shift = (char *)d_final_mod + (size_t)dim * sizeof(float);
        op_adaln(r, d_scratch1, d_img, f_shift, f_scale, n_img, dim);
        hipFree(d_final_mod);

        void *d_out = NULL;
        hipMalloc(&d_out, (size_t)n_img * in_ch * sizeof(float));
        qimg_set_gemm_context(r, -1, "proj_out");
        op_wgemm_bf16_auto(r, r->is_fp8_proj_out, d_out,
                           r->d_proj_out_w, d_scratch1, r->d_proj_out_b,
                           in_ch, dim, n_img);

        hipDeviceSynchronize();
        hipMemcpy(out, d_out, (size_t)n_img * in_ch * sizeof(float), hipMemcpyDeviceToHost);
        hipFree(d_out);
    }

    /* Cleanup */
    hipFree(d_img); hipFree(d_txt); hipFree(d_t_emb);
    hipFree(d_scratch1); hipFree(d_scratch2); hipFree(d_scratch3);
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_attn_out);

    return 0;
}


/* ---- VAE decode on GPU ---- */

/* Env-gated per-stage VAE dump for layer-by-layer comparison vs the CUDA fp8
 * reference. When QIMG_VAE_DUMP_PREFIX is set, each decode stage's current
 * activation d_x is copied D->H and written as raw little-endian f32 [c,h,w]
 * (C-contiguous, channel-major) to "<prefix>_<name>.bin" — matching the layout
 * of the CUDA cuda_vae_*.npy dumps. No effect when the env var is unset. */
static void qimg_vae_dump(const char *prefix, const char *name,
                          void *d_buf, int c, int h, int w) {
    if (!prefix || !d_buf) return;
    size_t n = (size_t)c * h * w;
    float *host = (float *)malloc(n * sizeof(float));
    if (!host) return;
    hipDeviceSynchronize();
    hipMemcpy(host, d_buf, n * sizeof(float), hipMemcpyDeviceToHost);
    char path[1024];
    snprintf(path, sizeof(path), "%s_%s.bin", prefix, name);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite(host, sizeof(float), n, f);
        fclose(f);
        fprintf(stderr, "  [vae-dump] %s [%d,%d,%d]\n", path, c, h, w);
    } else {
        fprintf(stderr, "  [vae-dump] failed to open %s\n", path);
    }
    free(host);
}

int hip_qimg_vae_decode(hip_qimg_runner *r,
                        const float *latent, int lat_h, int lat_w,
                        float *out_rgb) {
    st_context *st = (st_context *)r->vae_st;
    if (!st) { fprintf(stderr, "hip_qimg: VAE not loaded\n"); return -1; }
    const char *vae_dump_prefix = getenv("QIMG_VAE_DUMP_PREFIX");

    /* Free DiT weights to make room for the VAE. INT4 keeps ~14 GB resident with
     * no streaming, so it must be freed here or every VAE hipMalloc fails (OOM ->
     * gray output). hip_qimg_unload_dit frees int4_linears/int4_mod + scratch +
     * gpu_blocks; for the fp8 path (no int4) fall back to the block-only free. */
    if (r->int4_linears || r->int4_mod) {
        hip_qimg_unload_dit(r);
    } else if (r->gpu_blocks) {
        for (int i = 0; i < r->n_preloaded; i++)
            qimg_free_block(&r->gpu_blocks[i]);
        r->n_preloaded = 0;
    }
    hipDeviceSynchronize();

    int h = lat_h, w = lat_w, c = 16;
    fprintf(stderr, "hip_qimg_vae: decoding [%d, %d, %d] on GPU\n", c, h, w);

    /* Upload latent */
    void *d_x = NULL;
    hipMalloc(&d_x, (size_t)c * h * w * sizeof(float));
    hipMemcpy(d_x, latent, (size_t)c * h * w * sizeof(float), hipMemcpyHostToDevice);

    /* post_quant_conv */
    void *d_pqc_w = qimg_st_upload_f32(st, "conv2.weight");
    void *d_pqc_b = qimg_st_upload_f32(st, "conv2.bias");
    if (d_pqc_w) {
        void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_pqc_w, d_pqc_b, c, h, w, c, 1, 1, 0);
        hipFree(d_x); d_x = d_tmp;
        hipFree(d_pqc_w); hipFree(d_pqc_b);
    }
    qimg_vae_dump(vae_dump_prefix, "post_quant", d_x, c, h, w);

    /* decoder.conv1: 16→384, 3×3 */
    int co_c1, ci_c1;
    void *d_c1_w = qimg_upload_conv3d(st, "decoder.conv1.weight", &co_c1, &ci_c1);
    void *d_c1_b = qimg_st_upload_f32(st, "decoder.conv1.bias");
    c = co_c1;
    {
        void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)c*h*w*sizeof(float));
        vae_op_conv2d(r, d_tmp, d_x, d_c1_w, d_c1_b, ci_c1, h, w, c, 3, 3, 0);
        hipFree(d_x); d_x = d_tmp;
        hipFree(d_c1_w); hipFree(d_c1_b);
    }
    fprintf(stderr, "  after conv1: [%d, %d, %d]\n", c, h, w);
    qimg_vae_dump(vae_dump_prefix, "conv1", d_x, c, h, w);

    /* Load resblock weights helper macro */
    #define LOAD_RB_NAMED(pfx_str, n1, c1w, c1b, n2, c2w, c2b, scw, scb) \
        void *n1, *c1w, *c1b, *n2, *c2w, *c2b, *scw = NULL, *scb = NULL; \
        { char _nm[256]; \
          snprintf(_nm, sizeof(_nm), "%s.residual.0.gamma", pfx_str); n1 = qimg_st_upload_f32(st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.2.weight", pfx_str); { int _co, _ci; c1w = qimg_upload_conv3d(st, _nm, &_co, &_ci); } \
          snprintf(_nm, sizeof(_nm), "%s.residual.2.bias", pfx_str); c1b = qimg_st_upload_f32(st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.3.gamma", pfx_str); n2 = qimg_st_upload_f32(st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.residual.6.weight", pfx_str); { int _co2, _ci2; c2w = qimg_upload_conv3d(st, _nm, &_co2, &_ci2); } \
          snprintf(_nm, sizeof(_nm), "%s.residual.6.bias", pfx_str); c2b = qimg_st_upload_f32(st, _nm); \
          snprintf(_nm, sizeof(_nm), "%s.shortcut.weight", pfx_str); \
          if (safetensors_find(st, _nm) >= 0) { scw = qimg_st_upload_f32(st, _nm); \
            snprintf(_nm, sizeof(_nm), "%s.shortcut.bias", pfx_str); scb = qimg_st_upload_f32(st, _nm); } }

    /* mid.0 */
    { LOAD_RB_NAMED("decoder.middle.0", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      void *d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      hipFree(d_x); d_x = d_tmp;
      hipFree(n1); hipFree(c1w); hipFree(c1b); hipFree(n2); hipFree(c2w); hipFree(c2b);
      if (scw) { hipFree(scw); } if (scb) { hipFree(scb); } }
    qimg_vae_dump(vae_dump_prefix, "middle_0", d_x, c, h, w);

    /* Middle attention: CPU fallback for spatial self-attention */
    {
        hipDeviceSynchronize();
        struct timespec _t0; clock_gettime(CLOCK_MONOTONIC, &_t0);
        int spatial = h * w;
        void *d_gn_g = qimg_st_upload_f32(st, "decoder.middle.1.norm.gamma");
        void *d_qkv_w = qimg_st_upload_f32(st, "decoder.middle.1.to_qkv.weight");
        void *d_qkv_b = qimg_st_upload_f32(st, "decoder.middle.1.to_qkv.bias");
        void *d_proj_w = qimg_st_upload_f32(st, "decoder.middle.1.proj.weight");
        void *d_proj_b = qimg_st_upload_f32(st, "decoder.middle.1.proj.bias");

        void *d_normed = NULL; hipMalloc(&d_normed, (size_t)c*spatial*sizeof(float));
        vae_op_gn(r, d_normed, d_x, d_gn_g, c, spatial);
        hipFree(d_gn_g);

        void *d_qkv = NULL; hipMalloc(&d_qkv, (size_t)3*c*spatial*sizeof(float));
        vae_op_conv2d(r, d_qkv, d_normed, d_qkv_w, d_qkv_b, c, h, w, 3*c, 1, 1, 0);
        hipFree(d_normed); hipFree(d_qkv_w); hipFree(d_qkv_b);

        /* GPU spatial self-attention: vae_self_attn_f32 reads QKV laid out as
         * [3*c, spatial], writes [c, spatial]. Single head, head_dim=c=384. */
        void *d_attn_chw = NULL; hipMalloc(&d_attn_chw, (size_t)c*spatial*sizeof(float));
        if (r->fn_vae_self_attn_qb && (c % 32) == 0) {
            int QB = 8, BKV = 16;
            size_t smem = (size_t)2 * BKV * c * sizeof(float);
            unsigned grid = (unsigned)((spatial + QB - 1) / QB);
            void *args[] = {&d_attn_chw, &d_qkv, &spatial, &c};
            hipModuleLaunchKernel(r->fn_vae_self_attn_qb,
                                  grid, 1, 1, (unsigned)(QB*32), 1, 1,
                                  smem, NULL, args, NULL);
        } else if (r->fn_vae_self_attn && (c % 32) == 0) {
            int BKV = 8;
            size_t smem = (size_t)2 * BKV * c * sizeof(float);
            void *args[] = {&d_attn_chw, &d_qkv, &spatial, &c};
            hipModuleLaunchKernel(r->fn_vae_self_attn,
                                  (unsigned)spatial, 1, 1, 32, 1, 1,
                                  smem, NULL, args, NULL);
        } else {
            /* Fallback: CPU attention (legacy) */
            float *h_qkv = (float *)malloc((size_t)3 * c * spatial * sizeof(float));
            hipMemcpy(h_qkv, d_qkv, (size_t)3 * c * spatial * sizeof(float), hipMemcpyDeviceToHost);
            float *h_attn_chw = (float *)malloc((size_t)c * spatial * sizeof(float));
            float scale_at = 1.0f / sqrtf((float)c);
            for (int i = 0; i < spatial; i++) {
                float mx = -1e30f;
                for (int j = 0; j < spatial; j++) {
                    float dot = 0;
                    for (int d = 0; d < c; d++) dot += h_qkv[d*spatial+i] * h_qkv[(c+d)*spatial+j];
                    dot *= scale_at;
                    if (dot > mx) mx = dot;
                }
                float esum = 0;
                float *o_row = (float *)alloca((size_t)c * sizeof(float));
                memset(o_row, 0, (size_t)c * sizeof(float));
                for (int j = 0; j < spatial; j++) {
                    float dot = 0;
                    for (int d = 0; d < c; d++) dot += h_qkv[d*spatial+i] * h_qkv[(c+d)*spatial+j];
                    float w_at = expf(dot * scale_at - mx);
                    esum += w_at;
                    for (int d = 0; d < c; d++) o_row[d] += w_at * h_qkv[(2*c+d)*spatial+j];
                }
                float inv = 1.0f / esum;
                for (int d = 0; d < c; d++) h_attn_chw[d*spatial+i] = o_row[d] * inv;
            }
            free(h_qkv);
            hipMemcpy(d_attn_chw, h_attn_chw, (size_t)c*spatial*sizeof(float), hipMemcpyHostToDevice);
            free(h_attn_chw);
        }
        hipFree(d_qkv);

        void *d_proj_out_v = NULL; hipMalloc(&d_proj_out_v, (size_t)c*spatial*sizeof(float));
        vae_op_conv2d(r, d_proj_out_v, d_attn_chw, d_proj_w, d_proj_b, c, h, w, c, 1, 1, 0);
        hipFree(d_attn_chw); hipFree(d_proj_w); hipFree(d_proj_b);

        /* Residual: d_x += d_proj_out_v */
        {
            int n = c * spatial;
            float dt_one = 1.0f;
            void *ea[] = {&d_x, &d_proj_out_v, &dt_one, &n};
            hipModuleLaunchKernel(r->fn_euler_step, (unsigned)((n+255)/256), 1, 1,
                                  256, 1, 1, 0, NULL, ea, NULL);
        }
        hipFree(d_proj_out_v);
        hipDeviceSynchronize();
        struct timespec _t1; clock_gettime(CLOCK_MONOTONIC, &_t1);
        double _ms = (_t1.tv_sec-_t0.tv_sec)*1e3 + (_t1.tv_nsec-_t0.tv_nsec)/1e6;
        fprintf(stderr, "  vae middle attention: %.0f ms (spatial=%d, c=%d)\n", _ms, h*w, c);
    }
    qimg_vae_dump(vae_dump_prefix, "middle_1", d_x, c, h, w);

    /* mid.2 */
    { LOAD_RB_NAMED("decoder.middle.2", n1, c1w, c1b, n2, c2w, c2b, scw, scb);
      void *d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b, scw, scb, c, c, h, w);
      hipFree(d_x); d_x = d_tmp;
      hipFree(n1); hipFree(c1w); hipFree(c1b); hipFree(n2); hipFree(c2w); hipFree(c2b);
      if (scw) { hipFree(scw); } if (scb) { hipFree(scb); } }
    fprintf(stderr, "  after middle: [%d, %d, %d]\n", c, h, w);
    qimg_vae_dump(vae_dump_prefix, "middle_2", d_x, c, h, w);

    /* Upsample blocks 0-14 */
    for (int i = 0; i < 15; i++) {
        char pfx[128];
        snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.residual.2.weight", i);
        if (safetensors_find(st, pfx) >= 0) {
            char rb_pfx[128];
            snprintf(rb_pfx, sizeof(rb_pfx), "decoder.upsamples.%d", i);
            LOAD_RB_NAMED(rb_pfx, n1, c1w, c1b, n2, c2w, c2b, scw, scb);
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.residual.2.weight", i);
            int _idx = safetensors_find(st, pfx);
            int new_co = (int)safetensors_shape(st, _idx)[0];
            int old_ci = c;
            void *d_tmp = vae_resblock_gpu(r, d_x, n1, c1w, c1b, n2, c2w, c2b,
                                           scw, scb, old_ci, new_co, h, w);
            hipFree(d_x); d_x = d_tmp;
            c = new_co;
            hipFree(n1); hipFree(c1w); hipFree(c1b);
            hipFree(n2); hipFree(c2w); hipFree(c2b);
            if (scw) { hipFree(scw); } if (scb) { hipFree(scb); }
        }

        snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.weight", i);
        if (safetensors_find(st, pfx) >= 0) {
            void *rs_w = qimg_st_upload_f32(st, pfx);
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.bias", i);
            void *rs_b = qimg_st_upload_f32(st, pfx);
            void *d_up = vae_op_upsample(r, d_x, c, h, w);
            h *= 2; w *= 2;
            snprintf(pfx, sizeof(pfx), "decoder.upsamples.%d.resample.1.weight", i);
            int rs_idx = safetensors_find(st, pfx);
            int new_c = (int)safetensors_shape(st, rs_idx)[0];
            void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)new_c*h*w*sizeof(float));
            vae_op_conv2d(r, d_tmp, d_up, rs_w, rs_b, c, h, w, new_c, 3, 3, 0);
            hipFree(d_up); hipFree(d_x);
            hipFree(rs_w); hipFree(rs_b);
            d_x = d_tmp; c = new_c;
            fprintf(stderr, "  upsample %d: [%d, %d, %d]\n", i, c, h, w);
        }
        if (vae_dump_prefix) {
            char nm[32]; snprintf(nm, sizeof(nm), "upsample_%d", i);
            qimg_vae_dump(vae_dump_prefix, nm, d_x, c, h, w);
        }
    }
    #undef LOAD_RB_NAMED

    /* Head: GroupNorm → SiLU → Conv(96→3) */
    {
        void *d_gn = qimg_st_upload_f32(st, "decoder.head.0.gamma");
        int spatial = h * w;
        void *d_tmp = NULL; hipMalloc(&d_tmp, (size_t)c*spatial*sizeof(float));
        vae_op_gn(r, d_tmp, d_x, d_gn, c, spatial);
        vae_op_silu(r, d_tmp, c * spatial);
        hipFree(d_gn);

        int head_co, head_ci;
        void *d_hw = qimg_upload_conv3d(st, "decoder.head.2.weight", &head_co, &head_ci);
        void *d_hb = qimg_st_upload_f32(st, "decoder.head.2.bias");
        void *d_rgb = NULL; hipMalloc(&d_rgb, (size_t)3*spatial*sizeof(float));
        vae_op_conv2d(r, d_rgb, d_tmp, d_hw, d_hb, c, h, w, 3, 3, 3, 0);
        hipFree(d_tmp); hipFree(d_x); hipFree(d_hw); hipFree(d_hb);
        d_x = d_rgb;
        c = 3;
    }

    hipDeviceSynchronize();
    hipMemcpy(out_rgb, d_x, (size_t)3 * h * w * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(d_x);

    fprintf(stderr, "hip_qimg_vae: decode complete [%d, %d, %d]\n", c, h, w);
    return 0;
}
