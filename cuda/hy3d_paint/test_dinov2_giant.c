/*
 * test_dinov2_giant.c - Standalone CUDA DINOv2-giant encoder runner
 *
 * Loads facebook/dinov2-giant from a local safetensors file, runs the
 * 40-layer encoder on a preprocessed input image, and dumps the last
 * hidden state as .npy for comparison against ref/hy3d/dump_dinov2_giant.py.
 *
 * Architecture differences vs the DINOv2-L we already run in cuda/hy3d:
 *   - 40 layers, 1536 hidden, 24 heads, head_dim 64
 *   - SwiGLU FFN (weights_in -> split_silu_gate -> weights_out) instead
 *     of GELU MLP
 *   - Position embedding is stored at 37x37 patches (518x518 image) but
 *     BitImageProcessor center-crops to 224x224 (16x16 patches). HF
 *     auto-interpolates pos_embed at run-time; we pre-interpolate host-
 *     side at load time so the kernel path stays unchanged.
 *
 * Usage:
 *   ./test_dinov2_giant <model.safetensors> <input.npy> [<out_prefix>]
 *     -> <out_prefix>_output.npy  [1, 257, 1536]
 *
 * Reference for diffing:
 *   uv run python dump_dinov2_giant.py \
 *     --model /mnt/disk01/models/dinov2-giant \
 *     --image <some png> --outdir /tmp/hy3d_dinov2g
 *
 * Build:
 *   make test_dinov2_giant
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_kernels_common.h"
#include "../hy3d/cuda_hy3d_kernels.h"
#include "cuda_paint_nn_kernels.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ==== Config (DINOv2-giant @ 224) ======================================= */

#define HIDDEN       1536
#define HEADS        24
#define HEAD_DIM     64        /* HIDDEN / HEADS */
#define LAYERS       40
#define FFN_HALF     4096      /* SwiGLU mid dim */
#define FFN_FULL     8192      /* 2 * FFN_HALF */
#define PATCH        14
#define IMG_SIZE     224
#define GRID         16        /* IMG_SIZE / PATCH */
#define NUM_PATCHES  (GRID * GRID)     /* 256 */
#define SEQ_LEN      (NUM_PATCHES + 1) /* 257 */

/* Stored pos_embed grid (from 518x518 training): 37x37 + CLS = 1370. */
#define STORED_PATCHES   1369
#define STORED_GRID      37
#define STORED_SEQ_LEN   (STORED_PATCHES + 1)

/* ==== Weight storage ==================================================== */

typedef struct {
    CUdeviceptr q_w, q_b;        /* F16 weight, F32 bias */
    CUdeviceptr k_w, k_b;
    CUdeviceptr v_w, v_b;
    CUdeviceptr out_w, out_b;
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr ls1, ls2;        /* LayerScale lambda */
    CUdeviceptr w_in_w, w_in_b;  /* SwiGLU weights_in [FFN_FULL, HIDDEN] */
    CUdeviceptr w_out_w, w_out_b;/* SwiGLU weights_out [HIDDEN, FFN_HALF] */
} layer_gpu;

typedef struct {
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUmodule mod;
    int sm;

    /* Kernels */
    CUfunction k_patch_embed;
    CUfunction k_cls_pos_embed;
    CUfunction k_layernorm;
    CUfunction k_gemm;          /* gemm_tiled_f16_f32 */
    CUfunction k_attn;          /* cross_attn_f32 (hy3d) */
    CUfunction k_layerscale_add;
    CUfunction k_add;
    CUfunction k_silu_gate;

    /* Embedding weights */
    CUdeviceptr patch_w;        /* F32 [1536, 3, 14, 14] */
    CUdeviceptr patch_b;        /* F32 [1536] */
    CUdeviceptr cls_token;      /* F32 [1536] */
    CUdeviceptr pos_embed;      /* F32 [SEQ_LEN * HIDDEN] after interpolation */

    /* Final LN */
    CUdeviceptr final_ln_w, final_ln_b;

    layer_gpu layers[LAYERS];

    /* Scratch */
    CUdeviceptr d_hidden;       /* [SEQ_LEN * HIDDEN] */
    CUdeviceptr d_normed;       /* [SEQ_LEN * HIDDEN] */
    CUdeviceptr d_qkv;          /* [3 * SEQ_LEN * HIDDEN] */
    CUdeviceptr d_attn;         /* [SEQ_LEN * HIDDEN] */
    CUdeviceptr d_h_in;         /* [SEQ_LEN * FFN_FULL] */
    CUdeviceptr d_h_mid;        /* [SEQ_LEN * FFN_HALF] */
} dinov2g;

/* ==== Utility: F32 -> F16 raw upload (reuses cu_f32_to_f16) ============= */

static CUdeviceptr upload_f16(const float *data, size_t nelem) {
    uint16_t *buf = (uint16_t *)malloc(nelem * sizeof(uint16_t));
    for (size_t i = 0; i < nelem; i++) buf[i] = cu_f32_to_f16(data[i]);
    CUdeviceptr d = 0;
    cuMemAlloc(&d, nelem * sizeof(uint16_t));
    cuMemcpyHtoD(d, buf, nelem * sizeof(uint16_t));
    free(buf);
    return d;
}

static CUdeviceptr upload_f32(const float *data, size_t nelem) {
    CUdeviceptr d = 0;
    cuMemAlloc(&d, nelem * sizeof(float));
    cuMemcpyHtoD(d, data, nelem * sizeof(float));
    return d;
}

/* Fetch a tensor from safetensors as F32 (converts from F16/BF16 if needed). */
static float *st_fetch_f32(st_context *st, const char *name,
                            size_t *out_nelem) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        fprintf(stderr, "ERROR: tensor '%s' not found\n", name);
        return NULL;
    }
    void *raw = safetensors_data(st, idx);
    size_t nbytes = safetensors_nbytes(st, idx);
    const char *dtype = safetensors_dtype(st, idx);
    size_t nelem = 0;
    float *out = NULL;
    if (strcmp(dtype, "F32") == 0) {
        nelem = nbytes / 4;
        out = (float *)malloc(nelem * sizeof(float));
        memcpy(out, raw, nbytes);
    } else if (strcmp(dtype, "F16") == 0) {
        nelem = nbytes / 2;
        out = (float *)malloc(nelem * sizeof(float));
        const uint16_t *src = (const uint16_t *)raw;
        for (size_t i = 0; i < nelem; i++) {
            uint16_t h = src[i];
            uint32_t sign = (uint32_t)(h & 0x8000) << 16;
            uint32_t exp  = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;
            uint32_t bits;
            if (exp == 0) {
                bits = sign;
            } else if (exp == 31) {
                bits = sign | 0x7F800000 | (mant << 13);
            } else {
                bits = sign | ((exp + 112) << 23) | (mant << 13);
            }
            memcpy(&out[i], &bits, 4);
        }
    } else {
        fprintf(stderr, "ERROR: tensor '%s' unsupported dtype %s\n", name, dtype);
        return NULL;
    }
    if (out_nelem) *out_nelem = nelem;
    return out;
}

/* Catmull-Rom bicubic kernel with a = -0.75 (PyTorch default). */
static inline float cubic_w(float t) {
    const float a = -0.75f;
    t = t < 0.f ? -t : t;
    if (t <= 1.f) return ((a + 2.f) * t - (a + 3.f)) * t * t + 1.f;
    if (t <= 2.f) return (((t - 5.f) * t + 8.f) * t - 4.f) * a;
    return 0.f;
}

/* Host-side bicubic (align_corners=False) interpolation of the stored
 * pos_embed, matching torch.nn.functional.interpolate(mode='bicubic').
 *   input  [1, 1370, 1536]  (1 CLS + 37x37 patches, row-major)
 *   output [SEQ_LEN * HIDDEN]  (1 CLS + 16x16 patches) */
static void interpolate_pos_embed(const float *src, float *dst) {
    memcpy(dst, src, HIDDEN * sizeof(float));
    const float *patch_src = src + HIDDEN;
    float *patch_dst = dst + HIDDEN;

    const float sy = (float)STORED_GRID / (float)GRID;
    const float sx = (float)STORED_GRID / (float)GRID;

    for (int oy = 0; oy < GRID; oy++) {
        float fy = ((float)oy + 0.5f) * sy - 0.5f;
        int iy0 = (int)floorf(fy);
        float ty = fy - (float)iy0;
        float wy[4] = {
            cubic_w(1.f + ty),
            cubic_w(ty),
            cubic_w(1.f - ty),
            cubic_w(2.f - ty),
        };
        int ys[4] = { iy0 - 1, iy0, iy0 + 1, iy0 + 2 };
        /* Clamp to [0, STORED_GRID - 1] — matches PyTorch's border reflect
         * (actually replicate padding for bicubic). */
        for (int i = 0; i < 4; i++) {
            if (ys[i] < 0) ys[i] = 0;
            if (ys[i] >= STORED_GRID) ys[i] = STORED_GRID - 1;
        }
        for (int ox = 0; ox < GRID; ox++) {
            float fx = ((float)ox + 0.5f) * sx - 0.5f;
            int ix0 = (int)floorf(fx);
            float tx = fx - (float)ix0;
            float wx[4] = {
                cubic_w(1.f + tx),
                cubic_w(tx),
                cubic_w(1.f - tx),
                cubic_w(2.f - tx),
            };
            int xs[4] = { ix0 - 1, ix0, ix0 + 1, ix0 + 2 };
            for (int i = 0; i < 4; i++) {
                if (xs[i] < 0) xs[i] = 0;
                if (xs[i] >= STORED_GRID) xs[i] = STORED_GRID - 1;
            }
            float *out = patch_dst + (oy * GRID + ox) * HIDDEN;
            for (int c = 0; c < HIDDEN; c++) {
                float acc = 0.f;
                for (int j = 0; j < 4; j++) {
                    const float *row = patch_src + ys[j] * STORED_GRID * HIDDEN;
                    float rv = 0.f;
                    for (int i = 0; i < 4; i++) {
                        rv += row[xs[i] * HIDDEN + c] * wx[i];
                    }
                    acc += rv * wy[j];
                }
                out[c] = acc;
            }
        }
    }
}

/* ==== Runner setup ====================================================== */

static int dinov2g_init(dinov2g *r) {
    memset(r, 0, sizeof(*r));
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return -1;
    }
    cuInit(0);
    cuDeviceGet(&r->dev, 0);
    char name[256]; cuDeviceGetName(name, sizeof(name), r->dev);
    fprintf(stderr, "DINOv2-giant: GPU %s\n", name);
    cuCtxCreate(&r->ctx, 0, r->dev);
    cuStreamCreate(&r->stream, 0);

    /* Concatenate common + hy3d + paint_nn kernel sources, all already
     * matched so that common opens extern "C", hy3d closes it, paint_nn
     * re-opens + closes. */
    size_t l1 = strlen(cuda_kernels_common_src);
    size_t l2 = strlen(cuda_hy3d_specific_kernels);
    size_t l3 = strlen(cuda_paint_nn_kernels_src);
    char *src = (char *)malloc(l1 + l2 + l3 + 1);
    memcpy(src,           cuda_kernels_common_src, l1);
    memcpy(src + l1,      cuda_hy3d_specific_kernels, l2);
    memcpy(src + l1 + l2, cuda_paint_nn_kernels_src,  l3);
    src[l1 + l2 + l3] = '\0';

    r->sm = cu_compile_kernels(&r->mod, r->dev, src,
                                "dinov2g_kernels", 1, "DINOv2-G");
    free(src);
    if (r->sm < 0) return -1;

    cuModuleGetFunction(&r->k_patch_embed,    r->mod, "patch_embed_conv2d");
    cuModuleGetFunction(&r->k_cls_pos_embed,  r->mod, "cls_pos_embed");
    cuModuleGetFunction(&r->k_layernorm,      r->mod, "layernorm_f32");
    cuModuleGetFunction(&r->k_gemm,           r->mod, "gemm_tiled_f16_f32");
    cuModuleGetFunction(&r->k_attn,           r->mod, "cross_attn_f32");
    cuModuleGetFunction(&r->k_layerscale_add, r->mod, "layerscale_add_f32");
    cuModuleGetFunction(&r->k_add,            r->mod, "add_f32");
    cuModuleGetFunction(&r->k_silu_gate,      r->mod, "split_silu_gate_f32");

    /* Scratch buffers */
    cuMemAlloc(&r->d_hidden, SEQ_LEN * HIDDEN * sizeof(float));
    cuMemAlloc(&r->d_normed, SEQ_LEN * HIDDEN * sizeof(float));
    cuMemAlloc(&r->d_qkv,    3 * SEQ_LEN * HIDDEN * sizeof(float));
    cuMemAlloc(&r->d_attn,   SEQ_LEN * HIDDEN * sizeof(float));
    cuMemAlloc(&r->d_h_in,   SEQ_LEN * FFN_FULL * sizeof(float));
    cuMemAlloc(&r->d_h_mid,  SEQ_LEN * FFN_HALF * sizeof(float));
    return 0;
}

static int dinov2g_load(dinov2g *r, const char *path) {
    fprintf(stderr, "DINOv2-giant: loading %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) {
        fprintf(stderr, "ERROR: cannot open %s\n", path); return -1;
    }
    fprintf(stderr, "DINOv2-giant: %d tensors\n", st->n_tensors);

    /* Embeddings */
    size_t n = 0;
    float *patch_w = st_fetch_f32(st, "embeddings.patch_embeddings.projection.weight", &n);
    float *patch_b = st_fetch_f32(st, "embeddings.patch_embeddings.projection.bias", NULL);
    float *cls     = st_fetch_f32(st, "embeddings.cls_token", NULL);
    float *pos_raw = st_fetch_f32(st, "embeddings.position_embeddings", NULL);
    if (!patch_w || !patch_b || !cls || !pos_raw) return -1;

    r->patch_w   = upload_f32(patch_w, (size_t)HIDDEN * 3 * PATCH * PATCH);
    r->patch_b   = upload_f32(patch_b, HIDDEN);
    r->cls_token = upload_f32(cls, HIDDEN);

    /* Interpolate pos_embed 1x1370x1536 -> 1x257x1536 */
    float *pos_small = (float *)malloc(SEQ_LEN * HIDDEN * sizeof(float));
    interpolate_pos_embed(pos_raw, pos_small);
    r->pos_embed = upload_f32(pos_small, SEQ_LEN * HIDDEN);
    free(pos_small);

    free(patch_w); free(patch_b); free(cls); free(pos_raw);

    /* Final LN */
    float *ln_w = st_fetch_f32(st, "layernorm.weight", NULL);
    float *ln_b = st_fetch_f32(st, "layernorm.bias", NULL);
    r->final_ln_w = upload_f32(ln_w, HIDDEN);
    r->final_ln_b = upload_f32(ln_b, HIDDEN);
    free(ln_w); free(ln_b);

    /* Per-layer weights */
    for (int i = 0; i < LAYERS; i++) {
        char nm[256];
        layer_gpu *l = &r->layers[i];
#define FETCH(name_suffix) st_fetch_f32(st, nm, NULL); \
        (void)sizeof(name_suffix)
#define FF32(field, suffix) do { \
            snprintf(nm, sizeof(nm), "encoder.layer.%d.%s", i, suffix); \
            float *t = st_fetch_f32(st, nm, NULL); \
            if (!t) return -1; \
            l->field = upload_f32(t, HIDDEN); \
            free(t); \
        } while (0)
#define FF16(field, suffix, n_elem) do { \
            snprintf(nm, sizeof(nm), "encoder.layer.%d.%s", i, suffix); \
            float *t = st_fetch_f32(st, nm, NULL); \
            if (!t) return -1; \
            l->field = upload_f16(t, (size_t)(n_elem)); \
            free(t); \
        } while (0)

        /* LayerNorms */
        FF32(ln1_w, "norm1.weight");
        FF32(ln1_b, "norm1.bias");
        FF32(ln2_w, "norm2.weight");
        FF32(ln2_b, "norm2.bias");
        /* LayerScale */
        FF32(ls1,  "layer_scale1.lambda1");
        FF32(ls2,  "layer_scale2.lambda1");
        /* Attention biases */
        FF32(q_b,   "attention.attention.query.bias");
        FF32(k_b,   "attention.attention.key.bias");
        FF32(v_b,   "attention.attention.value.bias");
        FF32(out_b, "attention.output.dense.bias");
        /* Attention weights as F16 */
        FF16(q_w,   "attention.attention.query.weight",  (size_t)HIDDEN * HIDDEN);
        FF16(k_w,   "attention.attention.key.weight",    (size_t)HIDDEN * HIDDEN);
        FF16(v_w,   "attention.attention.value.weight",  (size_t)HIDDEN * HIDDEN);
        FF16(out_w, "attention.output.dense.weight",     (size_t)HIDDEN * HIDDEN);
        /* MLP biases */
        snprintf(nm, sizeof(nm), "encoder.layer.%d.mlp.weights_in.bias", i);
        { float *t = st_fetch_f32(st, nm, NULL); l->w_in_b = upload_f32(t, FFN_FULL); free(t); }
        snprintf(nm, sizeof(nm), "encoder.layer.%d.mlp.weights_out.bias", i);
        { float *t = st_fetch_f32(st, nm, NULL); l->w_out_b = upload_f32(t, HIDDEN); free(t); }
        /* MLP weights as F16 */
        FF16(w_in_w,  "mlp.weights_in.weight",  (size_t)FFN_FULL * HIDDEN);
        FF16(w_out_w, "mlp.weights_out.weight", (size_t)HIDDEN * FFN_HALF);

#undef FF32
#undef FF16
#undef FETCH
    }
    safetensors_close(st);
    fprintf(stderr, "DINOv2-giant: weights loaded\n");
    return 0;
}

/* ==== Kernel launch wrappers =========================================== */

static void launch_layernorm(dinov2g *r, CUdeviceptr dst, CUdeviceptr src,
                              CUdeviceptr w, CUdeviceptr b, int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = { &dst, &src, &w, &b, &dim, &eps };
    cuLaunchKernel(r->k_layernorm, n_tok, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

/* Y[n_tok, n_out] = X[n_tok, n_in] @ W[n_out, n_in]^T + bias[n_out] */
static void launch_gemm(dinov2g *r, CUdeviceptr y, CUdeviceptr w, CUdeviceptr x,
                         CUdeviceptr bias, int n_out, int n_in, int n_tok) {
    void *args[] = { &y, &w, &x, &bias, &n_out, &n_in, &n_tok };
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    cuLaunchKernel(r->k_gemm, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
}

static void launch_attn(dinov2g *r, CUdeviceptr out, CUdeviceptr Q,
                         CUdeviceptr K, CUdeviceptr V, int n_tok) {
    int q_len = n_tok, kv_len = n_tok, dim = HIDDEN;
    int n_heads = HEADS, head_dim = HEAD_DIM;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    int nt = 128;
    size_t smem = (size_t)(kv_len + nt) * sizeof(float);
    void *args[] = { &out, &Q, &K, &V, &q_len, &kv_len, &dim,
                     &n_heads, &head_dim, &scale };
    cuLaunchKernel(r->k_attn, (unsigned)n_heads, (unsigned)q_len, 1,
                   (unsigned)nt, 1, 1, smem, r->stream, args, NULL);
}

static void launch_layerscale_add(dinov2g *r, CUdeviceptr dst, CUdeviceptr src,
                                    CUdeviceptr scale, int n, int dim) {
    void *args[] = { &dst, &src, &scale, &n, &dim };
    cuLaunchKernel(r->k_layerscale_add,
                   (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void launch_silu_gate(dinov2g *r, CUdeviceptr in, CUdeviceptr out,
                               int rows, int half_dim) {
    void *args[] = { &in, &out, &rows, &half_dim };
    unsigned grid = (unsigned)((rows * half_dim + 255) / 256);
    cuLaunchKernel(r->k_silu_gate, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
}

/* ==== Forward pass ====================================================== */

/* Forward decl (def below main-side) */
static void write_npy_f32(const char *path, const int *shape, int ndims,
                           const float *data);

/* Debug: env-gated dump of a device buffer to .npy */
static void dbg_dump(dinov2g *r, CUdeviceptr dp, int n_tok, int dim,
                      const char *name) {
    const char *dir = getenv("DINOV2G_DUMP_DIR");
    if (!dir) return;
    cuStreamSynchronize(r->stream);
    size_t n = (size_t)n_tok * dim;
    float *h = (float *)malloc(n * sizeof(float));
    cuMemcpyDtoH(h, dp, n * sizeof(float));
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.npy", dir, name);
    int sh[3] = {1, n_tok, dim};
    write_npy_f32(path, sh, 3, h);
    free(h);
}

static void dinov2g_run(dinov2g *r, const float *image_f32, float *out_f32) {
    /* 1. Upload image to device */
    size_t img_bytes = (size_t)3 * IMG_SIZE * IMG_SIZE * sizeof(float);
    CUdeviceptr d_image;
    cuMemAlloc(&d_image, img_bytes);
    cuMemcpyHtoDAsync(d_image, image_f32, img_bytes, r->stream);

    /* 2. Patch embedding -> d_hidden[1..SEQ_LEN-1, :] */
    {
        int gw = GRID, dim = HIDDEN, ps = PATCH, img_w = IMG_SIZE;
        CUdeviceptr pw = r->patch_w, pb = r->patch_b;
        void *args[] = { &r->d_hidden, &d_image, &pw, &pb,
                         &gw, &dim, &ps, &img_w };
        cuLaunchKernel(r->k_patch_embed, (unsigned)(GRID * GRID), 1, 1,
                       256, 1, 1, 0, r->stream, args, NULL);
    }
    /* 3. CLS + pos_embed */
    {
        int n_tok = SEQ_LEN, dim = HIDDEN;
        CUdeviceptr cls = r->cls_token, pos = r->pos_embed;
        void *args[] = { &r->d_hidden, &cls, &pos, &n_tok, &dim };
        cuLaunchKernel(r->k_cls_pos_embed,
                       (unsigned)((SEQ_LEN * HIDDEN + 255) / 256), 1, 1,
                       256, 1, 1, 0, r->stream, args, NULL);
    }
    dbg_dump(r, r->d_hidden, SEQ_LEN, HIDDEN, "patch_embed");

    /* 4. Encoder stack */
    CUdeviceptr d_Q = r->d_qkv;
    CUdeviceptr d_K = r->d_qkv + (size_t)SEQ_LEN * HIDDEN * sizeof(float);
    CUdeviceptr d_V = r->d_qkv + (size_t)2 * SEQ_LEN * HIDDEN * sizeof(float);

    for (int i = 0; i < LAYERS; i++) {
        layer_gpu *l = &r->layers[i];

        /* Self-attention */
        launch_layernorm(r, r->d_normed, r->d_hidden, l->ln1_w, l->ln1_b, SEQ_LEN, HIDDEN);
        launch_gemm(r, d_Q, l->q_w, r->d_normed, l->q_b, HIDDEN, HIDDEN, SEQ_LEN);
        launch_gemm(r, d_K, l->k_w, r->d_normed, l->k_b, HIDDEN, HIDDEN, SEQ_LEN);
        launch_gemm(r, d_V, l->v_w, r->d_normed, l->v_b, HIDDEN, HIDDEN, SEQ_LEN);
        launch_attn(r, r->d_attn, d_Q, d_K, d_V, SEQ_LEN);
        launch_gemm(r, r->d_normed, l->out_w, r->d_attn, l->out_b,
                    HIDDEN, HIDDEN, SEQ_LEN);
        launch_layerscale_add(r, r->d_hidden, r->d_normed, l->ls1,
                               SEQ_LEN * HIDDEN, HIDDEN);

        /* SwiGLU FFN */
        launch_layernorm(r, r->d_normed, r->d_hidden, l->ln2_w, l->ln2_b, SEQ_LEN, HIDDEN);
        launch_gemm(r, r->d_h_in, l->w_in_w, r->d_normed, l->w_in_b,
                    FFN_FULL, HIDDEN, SEQ_LEN);
        launch_silu_gate(r, r->d_h_in, r->d_h_mid, SEQ_LEN, FFN_HALF);
        launch_gemm(r, r->d_normed, l->w_out_w, r->d_h_mid, l->w_out_b,
                    HIDDEN, FFN_HALF, SEQ_LEN);
        launch_layerscale_add(r, r->d_hidden, r->d_normed, l->ls2,
                               SEQ_LEN * HIDDEN, HIDDEN);

        if (i == 0 || i == 10 || i == 20 || i == 30 || i == 39) {
            char nm[32]; snprintf(nm, sizeof(nm), "hidden_%d", i);
            dbg_dump(r, r->d_hidden, SEQ_LEN, HIDDEN, nm);
        }
    }

    /* 5. Final LN */
    CUdeviceptr d_out;
    cuMemAlloc(&d_out, SEQ_LEN * HIDDEN * sizeof(float));
    launch_layernorm(r, d_out, r->d_hidden, r->final_ln_w, r->final_ln_b,
                      SEQ_LEN, HIDDEN);
    cuStreamSynchronize(r->stream);
    cuMemcpyDtoH(out_f32, d_out, (size_t)SEQ_LEN * HIDDEN * sizeof(float));
    cuMemFree(d_out);
    cuMemFree(d_image);
}

/* ==== .npy reader + writer ============================================= */

static float *read_npy_f32(const char *path, int *shape_out, int *ndims_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return NULL; }
    char magic[6]; fread(magic, 1, 6, f);
    uint8_t ver[2]; fread(ver, 1, 2, f);
    uint16_t hlen; fread(&hlen, 2, 1, f);
    char *hdr = (char *)malloc(hlen + 1);
    fread(hdr, 1, hlen, f); hdr[hlen] = '\0';
    const char *sp = strstr(hdr, "'shape': (");
    size_t total = 1;
    int ndims = 0;
    if (sp) {
        sp += 10;
        while (*sp && *sp != ')') {
            if (*sp >= '0' && *sp <= '9') {
                int d = (int)strtol(sp, (char **)&sp, 10);
                shape_out[ndims++] = d;
                total *= (size_t)d;
            } else sp++;
        }
    }
    *ndims_out = ndims;
    free(hdr);
    float *data = (float *)malloc(total * sizeof(float));
    if (fread(data, sizeof(float), total, f) != total) {
        free(data); fclose(f); return NULL;
    }
    fclose(f);
    return data;
}

static void write_npy_f32(const char *path, const int *shape, int ndims,
                           const float *data) {
    FILE *f = fopen(path, "wb");
    fwrite("\x93NUMPY", 1, 6, f);
    uint8_t ver[2] = {1, 0}; fwrite(ver, 1, 2, f);
    char hdr[256], shape_s[128] = "";
    size_t total = 1;
    for (int i = 0; i < ndims; i++) {
        char tmp[32]; snprintf(tmp, sizeof(tmp), "%d, ", shape[i]);
        strcat(shape_s, tmp);
        total *= (size_t)shape[i];
    }
    int hlen = snprintf(hdr, sizeof(hdr),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }", shape_s);
    int tot = 10 + hlen + 1;
    int pad = ((tot + 63) / 64) * 64 - tot;
    uint16_t header_len = (uint16_t)(hlen + pad + 1);
    fwrite(&header_len, 2, 1, f);
    fwrite(hdr, 1, (size_t)hlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), total, f);
    fclose(f);
}

/* ==== main ============================================================= */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <dinov2-giant/model.safetensors> <input.npy> [<out_prefix>]\n"
            "  input.npy: [1, 3, 224, 224] float32 (from dump_dinov2_giant.py)\n",
            argv[0]);
        return 1;
    }
    const char *weights_path = argv[1];
    const char *input_path   = argv[2];
    const char *out_prefix   = argc >= 4 ? argv[3] : "/tmp/hy3d_dinov2g_cuda";

    int shape[4]; int ndims;
    float *input = read_npy_f32(input_path, shape, &ndims);
    if (!input) return 1;
    fprintf(stderr, "input %s shape=(", input_path);
    for (int i = 0; i < ndims; i++) fprintf(stderr, "%d%s", shape[i], i+1<ndims?", ":"");
    fprintf(stderr, ")\n");
    if (ndims != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != IMG_SIZE || shape[3] != IMG_SIZE) {
        fprintf(stderr, "ERROR: expected [1, 3, %d, %d]\n", IMG_SIZE, IMG_SIZE);
        return 1;
    }

    dinov2g r;
    if (dinov2g_init(&r) != 0) return 1;
    if (dinov2g_load(&r, weights_path) != 0) return 1;

    float *output = (float *)malloc((size_t)SEQ_LEN * HIDDEN * sizeof(float));
    dinov2g_run(&r, input, output);

    /* Basic stats */
    double mn = output[0], mx = output[0], sum = 0.0;
    for (int i = 0; i < SEQ_LEN * HIDDEN; i++) {
        if (output[i] < mn) mn = output[i];
        if (output[i] > mx) mx = output[i];
        sum += output[i];
    }
    double mean = sum / (double)(SEQ_LEN * HIDDEN);
    double var = 0.0;
    for (int i = 0; i < SEQ_LEN * HIDDEN; i++) {
        double d = output[i] - mean; var += d * d;
    }
    double std = sqrt(var / (double)(SEQ_LEN * HIDDEN));
    fprintf(stderr, "output: min=%.4f max=%.4f mean=%.4f std=%.4f\n",
            mn, mx, mean, std);

    char path[512];
    int out_shape[3] = {1, SEQ_LEN, HIDDEN};
    snprintf(path, sizeof(path), "%s_output.npy", out_prefix);
    write_npy_f32(path, out_shape, 3, output);
    fprintf(stderr, "Wrote %s\n", path);
    free(input); free(output);
    return 0;
}
