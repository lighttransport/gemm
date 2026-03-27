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

/* Generic DiT model (shared by Stage 1 and Stage 2) */
typedef struct {
    int n_blocks;
    int model_channels;   /* D: 1536 */
    int n_heads;          /* 12 */
    int head_dim;         /* 128 */
    int ffn_hidden;       /* 8192 */
    int in_channels;      /* 8 for Stage 1, 32 for Stage 2 */
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

    /* Stage 2 shape flow DiT */
    dit_model_gpu stage2;
    dit_block_gpu stage2_blocks[DIT_DEPTH];
    int stage2_loaded;

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

    /* Scratch buffers */
    CUdeviceptr scratch[8];
    size_t      scratch_size[8];
};

static void ensure_scratch(cuda_trellis2_runner *r, int idx, size_t bytes) {
    if (r->scratch_size[idx] >= bytes) return;
    if (r->scratch[idx]) cuMemFree(r->scratch[idx]);
    cuMemAlloc(&r->scratch[idx], bytes);
    r->scratch_size[idx] = bytes;
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

    /* Top-level */
    r->dit_t_fc1_w = t2_upload_f32(st, "t_embedder.mlp.0.weight", v);
    r->dit_t_fc1_b = t2_upload_f32(st, "t_embedder.mlp.0.bias", v);
    r->dit_t_fc2_w = t2_upload_f32(st, "t_embedder.mlp.2.weight", v);
    r->dit_t_fc2_b = t2_upload_f32(st, "t_embedder.mlp.2.bias", v);
    r->dit_mod_w   = t2_upload_f32(st, "adaLN_modulation.1.weight", v);
    r->dit_mod_b   = t2_upload_f32(st, "adaLN_modulation.1.bias", v);
    r->dit_x_emb_w = t2_upload_f32(st, "input_layer.weight", v);
    r->dit_x_emb_b = t2_upload_f32(st, "input_layer.bias", v);
    r->dit_out_w   = t2_upload_f32(st, "out_layer.weight", v);
    r->dit_out_b   = t2_upload_f32(st, "out_layer.bias", v);

    if (!r->dit_t_fc1_w) fprintf(stderr, "T2: WARNING: t_embedder missing\n");
    if (!r->dit_mod_w)   fprintf(stderr, "T2: WARNING: adaLN_modulation missing\n");
    if (!r->dit_x_emb_w) fprintf(stderr, "T2: WARNING: input_layer missing\n");

    /* Per-block */
    for (int L = 0; L < DIT_DEPTH; L++) {
        dit_block_gpu *blk = &r->dit_blocks[L];
        char name[256];

        #define BLK(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                              t2_upload_f32(st, name, v >= 2 ? v : 0))

        blk->sa_qkv_w    = BLK("self_attn.to_qkv.weight");
        blk->sa_qkv_b    = BLK("self_attn.to_qkv.bias");
        blk->sa_q_norm   = BLK("self_attn.q_rms_norm.gamma");
        blk->sa_k_norm   = BLK("self_attn.k_rms_norm.gamma");
        blk->sa_out_w    = BLK("self_attn.to_out.weight");
        blk->sa_out_b    = BLK("self_attn.to_out.bias");

        blk->norm2_w     = BLK("norm2.weight");
        blk->norm2_b     = BLK("norm2.bias");
        blk->ca_q_w      = BLK("cross_attn.to_q.weight");
        blk->ca_q_b      = BLK("cross_attn.to_q.bias");
        blk->ca_kv_w     = BLK("cross_attn.to_kv.weight");
        blk->ca_kv_b     = BLK("cross_attn.to_kv.bias");
        blk->ca_q_norm   = BLK("cross_attn.q_rms_norm.gamma");
        blk->ca_k_norm   = BLK("cross_attn.k_rms_norm.gamma");
        blk->ca_out_w    = BLK("cross_attn.to_out.weight");
        blk->ca_out_b    = BLK("cross_attn.to_out.bias");

        blk->mlp_fc1_w   = BLK("mlp.mlp.0.weight");
        blk->mlp_fc1_b   = BLK("mlp.mlp.0.bias");
        blk->mlp_fc2_w   = BLK("mlp.mlp.2.weight");
        blk->mlp_fc2_b   = BLK("mlp.mlp.2.bias");

        blk->mod_bias    = BLK("modulation");

        #undef BLK

        if (L == 0 && !blk->sa_qkv_w)
            fprintf(stderr, "T2: WARNING: block 0 self_attn missing\n");
    }

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

    safetensors_close(st);
    fprintf(stderr, "T2: DiT loaded (%d blocks)\n", DIT_DEPTH);
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

static int load_stage2_weights(cuda_trellis2_runner *r, const char *path) {
    /* Stage 2 uses identical weight naming as Stage 1, just in_channels=32 */
    st_context *st = safetensors_open(path);
    if (!st) return -1;
    fprintf(stderr, "T2: loading Stage 2 from %s (%d tensors)\n", path, st->n_tensors);
    int v = r->verbose;
    dit_model_gpu *m = &r->stage2;

    m->n_blocks = DIT_DEPTH;
    m->model_channels = DIT_DIM;
    m->n_heads = DIT_HEADS;
    m->head_dim = DIT_HEAD_DIM;
    m->ffn_hidden = DIT_FFN;
    m->cond_dim = DIT_COND_DIM;
    m->blocks = r->stage2_blocks;

    /* Detect in_channels from input_layer weight */
    {
        int idx = safetensors_find(st, "input_layer.weight");
        if (idx >= 0) m->in_channels = (int)safetensors_shape(st, idx)[1];
        else m->in_channels = 32;
    }

    m->t_fc1_w = t2_upload_f32(st, "t_embedder.mlp.0.weight", v);
    m->t_fc1_b = t2_upload_f32(st, "t_embedder.mlp.0.bias", v);
    m->t_fc2_w = t2_upload_f32(st, "t_embedder.mlp.2.weight", v);
    m->t_fc2_b = t2_upload_f32(st, "t_embedder.mlp.2.bias", v);
    m->mod_w   = t2_upload_f32(st, "adaLN_modulation.1.weight", v);
    m->mod_b   = t2_upload_f32(st, "adaLN_modulation.1.bias", v);
    m->x_emb_w = t2_upload_f32(st, "input_layer.weight", v);
    m->x_emb_b = t2_upload_f32(st, "input_layer.bias", v);
    m->out_w   = t2_upload_f32(st, "out_layer.weight", v);
    m->out_b   = t2_upload_f32(st, "out_layer.bias", v);

    for (int L = 0; L < m->n_blocks; L++) {
        dit_block_gpu *blk = &m->blocks[L];
        char name[256]; int idx;
        #define BLK2(suffix) (snprintf(name, sizeof(name), "blocks.%d.%s", L, suffix), \
                               t2_upload_f32(st, name, v >= 2 ? v : 0))
        blk->sa_qkv_w  = BLK2("self_attn.to_qkv.weight");
        blk->sa_qkv_b  = BLK2("self_attn.to_qkv.bias");
        blk->sa_q_norm  = BLK2("self_attn.q_rms_norm.gamma");
        blk->sa_k_norm  = BLK2("self_attn.k_rms_norm.gamma");
        blk->sa_out_w   = BLK2("self_attn.to_out.weight");
        blk->sa_out_b   = BLK2("self_attn.to_out.bias");
        blk->norm2_w    = BLK2("norm2.weight");
        blk->norm2_b    = BLK2("norm2.bias");
        blk->ca_q_w     = BLK2("cross_attn.to_q.weight");
        blk->ca_q_b     = BLK2("cross_attn.to_q.bias");
        blk->ca_kv_w    = BLK2("cross_attn.to_kv.weight");
        blk->ca_kv_b    = BLK2("cross_attn.to_kv.bias");
        blk->ca_q_norm  = BLK2("cross_attn.q_rms_norm.gamma");
        blk->ca_k_norm  = BLK2("cross_attn.k_rms_norm.gamma");
        blk->ca_out_w   = BLK2("cross_attn.to_out.weight");
        blk->ca_out_b   = BLK2("cross_attn.to_out.bias");
        blk->mlp_fc1_w  = BLK2("mlp.mlp.0.weight");
        blk->mlp_fc1_b  = BLK2("mlp.mlp.0.bias");
        blk->mlp_fc2_w  = BLK2("mlp.mlp.2.weight");
        blk->mlp_fc2_b  = BLK2("mlp.mlp.2.bias");
        blk->mod_bias   = BLK2("modulation");
        #undef BLK2
    }

    /* RoPE tables computed at runtime from sparse coords — set to NULL for now */
    m->rope_cos = 0; m->rope_sin = 0;
    m->n_rope_freqs = DIT_HEAD_DIM / 6;  /* 21 */
    m->rope_axis_dim = 2 * m->n_rope_freqs;

    safetensors_close(st);
    r->stage2_loaded = 1;
    fprintf(stderr, "T2: Stage 2 loaded (%d blocks, in_ch=%d)\n", m->n_blocks, m->in_channels);
    return 0;
}

int cuda_trellis2_load_stage2(cuda_trellis2_runner *r, const char *stage2_path) {
    return load_stage2_weights(r, stage2_path);
}

int cuda_trellis2_load_weights(cuda_trellis2_runner *r,
                                const char *dinov3_path,
                                const char *stage1_path,
                                const char *decoder_path) {
    /* Use F32 GEMM since all weights are converted to F32 */
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
                                      int n_blocks,
                                      CUdeviceptr t_fc1_w, CUdeviceptr t_fc1_b,
                                      CUdeviceptr t_fc2_w, CUdeviceptr t_fc2_b,
                                      CUdeviceptr mod_w, CUdeviceptr mod_b,
                                      CUdeviceptr x_emb_w, CUdeviceptr x_emb_b,
                                      CUdeviceptr out_w, CUdeviceptr out_b,
                                      dit_block_gpu *blocks,
                                      CUdeviceptr rope_cos, CUdeviceptr rope_sin,
                                      int n_freqs, int axis_dim) {
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
    size_t sh1 = qkv_sz > mlp_sz ? qkv_sz : mlp_sz;
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

    /* 2. Timestep embedding: sinusoidal(256) -> MLP(256->D->D) */
    /* TRELLIS.2: t*1000, [cos, sin] order (NOT [sin, cos] like HY3D) */
    {
        float t_scaled = timestep * 1000.0f;
        int half = 128;
        void *te_args[] = {&d_temb, &t_scaled, &half};  /* dim/2 = 128 for the kernel */
        int te_dim = 256;
        void *te_args2[] = {&d_temb, &t_scaled, &te_dim};
        cuLaunchKernel(ops->timestep_embed_cossin, (unsigned)((128+255)/256), 1, 1,
                       256, 1, 1, 0, stream, te_args2, NULL);
    }
    /* t_fc1: [D, 256] */
    t2_op_gemm(ops, stream, d_normed, t_fc1_w, d_temb, t_fc1_b,
               D, 256, 1);
    /* SiLU */
    t2_op_silu_inplace(ops, stream, d_normed, D);
    /* t_fc2: [D, D] */
    t2_op_gemm(ops, stream, d_temb, t_fc2_w, d_normed, t_fc2_b,
               D, D, 1);
    /* Now d_temb = [D] timestep embedding */

    /* 3. Transformer blocks */
    for (int bi = 0; bi < n_blocks; bi++) {
        dit_block_gpu *blk = &blocks[bi];

        /* 3a. Modulation: SiLU(t_emb) @ mod_w + mod_b + block_bias -> [6*D] */
        t2_op_modulation(ops, stream, d_mod, d_temb,
                         mod_w, mod_b, blk->mod_bias,
                         D, 6 * D);

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

        /* KV from conditioning (recompute per block — different weights) */
        /* KV GEMM: [ctx, cond_dim] -> [ctx, 2*D] */
        t2_op_gemm(ops, stream, d_qkv, blk->ca_kv_w, d_cond, blk->ca_kv_b,
                   2 * D, DIT_COND_DIM, ctx_len);
        /* Split KV (standard chunk, NOT interleaved) */
        t2_op_split_kv_chunk(ops, stream, d_ca_K, d_ca_V, d_qkv, ctx_len, D);
        if (blk->ca_k_norm)
            t2_op_rms_norm_perhead(ops, stream, d_ca_K, blk->ca_k_norm,
                                  ctx_len, H, HD, D);

        /* Cross-attention */
        t2_op_cross_attn(ops, stream, d_attn, d_cross_Q, d_ca_K, d_ca_V,
                         N, ctx_len, D, H, HD);

        /* Cross-attn output + residual (no gate) */
        t2_op_gemm(ops, stream, d_normed, blk->ca_out_w, d_attn, blk->ca_out_b,
                   D, D, N);
        t2_op_add(ops, stream, d_hidden, d_normed, N * D);

        /* === MLP === */
        t2_op_adaln(ops, stream, d_normed, d_hidden,
                    d_shift_mlp, d_scale_mlp, N, D);
        t2_op_gemm(ops, stream, d_mlp, blk->mlp_fc1_w, d_normed, blk->mlp_fc1_b,
                   FFN, D, N);
        t2_op_gelu(ops, stream, d_mlp, N * FFN);
        t2_op_gemm(ops, stream, d_normed, blk->mlp_fc2_w, d_mlp, blk->mlp_fc2_b,
                   D, FFN, N);
        t2_op_gated_add(ops, stream, d_hidden, d_normed, d_gate_mlp, N * D, D);

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
    }

    /* 4. Final LayerNorm (no affine) + output projection */
    {
        float eps = 1e-6f;
        void *ln_args[] = {&d_hidden, &d_hidden, &D, &eps};
        cuLaunchKernel(ops->layernorm_noaffine, (unsigned)N, 1, 1,
                       256, 1, 1, 512 * sizeof(float), stream, ln_args, NULL);
    }
    t2_op_gemm(ops, stream, d_output, out_w, d_hidden, out_b,
               in_ch, D, N);
}

/* Stage 1 wrapper */
static void run_dit_forward(cuda_trellis2_runner *r,
                             CUdeviceptr d_x, float timestep,
                             CUdeviceptr d_cond, CUdeviceptr d_output) {
    run_dit_forward_generic(r, d_x, timestep, d_cond, d_output,
        DIT_N_TOKENS, DIT_IN_CH, DIT_DEPTH,
        r->dit_t_fc1_w, r->dit_t_fc1_b, r->dit_t_fc2_w, r->dit_t_fc2_b,
        r->dit_mod_w, r->dit_mod_b,
        r->dit_x_emb_w, r->dit_x_emb_b,
        r->dit_out_w, r->dit_out_b,
        r->dit_blocks, r->dit_rope_cos, r->dit_rope_sin,
        r->dit_n_freqs, r->dit_axis_dim);
}

/* Stage 2 wrapper (variable N from sparse coords) */
static void run_stage2_forward(cuda_trellis2_runner *r,
                                CUdeviceptr d_x, float timestep,
                                CUdeviceptr d_cond, CUdeviceptr d_output,
                                int N, CUdeviceptr rope_cos, CUdeviceptr rope_sin) {
    dit_model_gpu *m = &r->stage2;
    run_dit_forward_generic(r, d_x, timestep, d_cond, d_output,
        N, m->in_channels, m->n_blocks,
        m->t_fc1_w, m->t_fc1_b, m->t_fc2_w, m->t_fc2_b,
        m->mod_w, m->mod_b,
        m->x_emb_w, m->x_emb_b,
        m->out_w, m->out_b,
        m->blocks, rope_cos, rope_sin,
        m->n_rope_freqs, m->rope_axis_dim);
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

int cuda_trellis2_run_dit(cuda_trellis2_runner *r,
                           const float *x_t, float timestep,
                           const float *cond_features, float *output) {
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
