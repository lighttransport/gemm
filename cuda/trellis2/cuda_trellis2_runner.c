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
    CUdeviceptr qkv_w, qkv_b;
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

    /* DiT weights */
    CUdeviceptr dit_t_fc1_w, dit_t_fc1_b, dit_t_fc2_w, dit_t_fc2_b;
    CUdeviceptr dit_mod_w, dit_mod_b;
    CUdeviceptr dit_x_emb_w, dit_x_emb_b;
    CUdeviceptr dit_out_w, dit_out_b;
    dit_block_gpu dit_blocks[DIT_DEPTH];
    CUdeviceptr dit_rope_cos, dit_rope_sin;  /* [DIT_N_TOKENS, 3, n_freqs] */
    int dit_n_freqs, dit_axis_dim;

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

int cuda_trellis2_load_weights(cuda_trellis2_runner *r,
                                const char *dinov3_path,
                                const char *stage1_path,
                                const char *decoder_path) {
    /* Use F32 GEMM since all weights are converted to F32 */
    r->ops.use_f32_gemm = 1;

    if (stage1_path && load_dit_weights(r, stage1_path) != 0) return -1;
    if (decoder_path && load_decoder_weights(r, decoder_path) != 0) return -1;

    /* DINOv3 loading would go here — skip for now (use CPU DINOv3 features) */
    if (dinov3_path)
        fprintf(stderr, "T2: DINOv3 GPU encoding not yet implemented (use CPU features)\n");

    return 0;
}

/* ======================================================================== */
/* DiT forward pass (single denoising step)                                 */
/* ======================================================================== */

static void run_dit_forward(cuda_trellis2_runner *r,
                             CUdeviceptr d_x, float timestep,
                             CUdeviceptr d_cond,
                             CUdeviceptr d_output) {
    t2_ops *ops = &r->ops;
    CUstream stream = r->stream;
    const int N = DIT_N_TOKENS;   /* 4096 */
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
    t2_op_gemm(ops, stream, d_hidden, r->dit_x_emb_w, d_x, r->dit_x_emb_b,
               D, DIT_IN_CH, N);

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
    t2_op_gemm(ops, stream, d_normed, r->dit_t_fc1_w, d_temb, r->dit_t_fc1_b,
               D, 256, 1);
    /* SiLU */
    t2_op_silu_inplace(ops, stream, d_normed, D);
    /* t_fc2: [D, D] */
    t2_op_gemm(ops, stream, d_temb, r->dit_t_fc2_w, d_normed, r->dit_t_fc2_b,
               D, D, 1);
    /* Now d_temb = [D] timestep embedding */

    /* 3. Transformer blocks */
    for (int bi = 0; bi < DIT_DEPTH; bi++) {
        dit_block_gpu *blk = &r->dit_blocks[bi];

        /* 3a. Modulation: SiLU(t_emb) @ mod_w + mod_b + block_bias -> [6*D] */
        t2_op_modulation(ops, stream, d_mod, d_temb,
                         r->dit_mod_w, r->dit_mod_b, blk->mod_bias,
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
        t2_op_rope_3d(ops, stream, d_Q, r->dit_rope_cos, r->dit_rope_sin,
                      N, D, H, HD, r->dit_n_freqs, r->dit_axis_dim);
        t2_op_rope_3d(ops, stream, d_K, r->dit_rope_cos, r->dit_rope_sin,
                      N, D, H, HD, r->dit_n_freqs, r->dit_axis_dim);

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
    t2_op_gemm(ops, stream, d_output, r->dit_out_w, d_hidden, r->dit_out_b,
               DIT_IN_CH, D, N);
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

    /* GN1 -> SiLU -> Conv1 */
    t2_op_groupnorm_3d(ops, s, d_h1, d_in, rb->gn1_w, rb->gn1_b, C, spatial, G);
    t2_op_silu_inplace(ops, s, d_h1, C * spatial);
    t2_op_conv3d(ops, s, d_h2, d_h1, rb->conv1_w, rb->conv1_b, C, C, D, H, W);

    /* GN2 -> SiLU -> Conv2 */
    t2_op_groupnorm_3d(ops, s, d_h1, d_h2, rb->gn2_w, rb->gn2_b, C, spatial, G);
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

    /* middle + res16 blocks (512 ch, 16^3) */
    run_resblock(r, d_buf_b, d_buf_a, &r->dec_middle[0], 512, 16, 16, 16, G);
    run_resblock(r, d_buf_a, d_buf_b, &r->dec_middle[1], 512, 16, 16, 16, G);
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
    t2_op_groupnorm_3d(ops, s, d_buf_b, d_buf_a,
                       r->dec_out_gn_w, r->dec_out_gn_b, 32, 64*64*64, G);
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
    (void)r; (void)image_f32; (void)output;
    fprintf(stderr, "T2: DINOv3 GPU not yet implemented\n");
    return -1;
}
