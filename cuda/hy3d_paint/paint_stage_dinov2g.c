/*
 * paint_stage_dinov2g.c - DINOv2-giant encoder as a per-stage TU.
 *
 * Lifted verbatim from test_dinov2_giant.c, exposed via the opaque API in
 * paint_stages.h so the pipeline orchestrator can drop the standalone main.
 *
 * Owns its own CUmodule (sole-includer of cuda_paint_nn_kernels.h /
 * cuda_fp8_mma_kernels.h / cuda_hy3d_kernels.h to avoid file-local helper
 * collisions with sibling stage TUs). SAFETENSORS_IMPLEMENTATION lives in
 * paint_runtime.c, so this TU only uses the safetensors API.
 */

#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_kernels_common.h"
#include "../hy3d/cuda_hy3d_kernels.h"
#include "../cuda_fp8_mma_kernels.h"
#include "cuda_paint_nn_kernels.h"
#include "../../common/safetensors.h"
#include "paint_stages.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define HIDDEN       1536
#define HEADS        24
#define HEAD_DIM     64
#define LAYERS       40
#define FFN_HALF     4096
#define FFN_FULL     8192
#define PATCH        14
#define IMG_SIZE     224
#define GRID         16
#define NUM_PATCHES  (GRID * GRID)
#define SEQ_LEN      (NUM_PATCHES + 1)

#define STORED_PATCHES   1369
#define STORED_GRID      37
#define STORED_SEQ_LEN   (STORED_PATCHES + 1)

typedef struct {
    CUdeviceptr q_w, q_b;
    CUdeviceptr k_w, k_b;
    CUdeviceptr v_w, v_b;
    CUdeviceptr out_w, out_b;
    CUdeviceptr ln1_w, ln1_b;
    CUdeviceptr ln2_w, ln2_b;
    CUdeviceptr ls1, ls2;
    CUdeviceptr w_in_w, w_in_b;
    CUdeviceptr w_out_w, w_out_b;
    CUdeviceptr q_w_fp8,   q_w_scale;
    CUdeviceptr k_w_fp8,   k_w_scale;
    CUdeviceptr v_w_fp8,   v_w_scale;
    CUdeviceptr out_w_fp8, out_w_scale;
    CUdeviceptr w_in_fp8,  w_in_scale;
    CUdeviceptr w_out_fp8, w_out_scale;
} layer_gpu;

struct paint_stage_dinov2g {
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    CUmodule mod;
    int sm;
    int owns_ctx;

    CUfunction k_patch_embed;
    CUfunction k_cls_pos_embed;
    CUfunction k_layernorm;
    CUfunction k_gemm;
    CUfunction k_attn;
    CUfunction k_layerscale_add;
    CUfunction k_add;
    CUfunction k_silu_gate;
    CUfunction k_gemm_bf16_mt4;
    CUfunction k_gemm_bf16;
    CUfunction k_flash_attn_hd64;
    CUfunction k_reduce_max_abs;
    CUfunction k_quantize_fp8;
    CUfunction k_cast_f32_to_bf16;
    CUfunction k_f16_to_f32;

    int use_bf16_attn;
    int use_bf16_gemm;
    int use_bf16_mt4;

    CUdeviceptr patch_w, patch_b, cls_token, pos_embed;
    CUdeviceptr final_ln_w, final_ln_b;

    layer_gpu layers[LAYERS];

    CUdeviceptr d_hidden, d_normed, d_qkv, d_attn, d_h_in, d_h_mid;
    CUdeviceptr d_qkv_bf16;
};

typedef struct paint_stage_dinov2g dinov2g;

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

static float *st_fetch_f32(st_context *st, const char *name, size_t *out_nelem) {
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
            if (exp == 0) bits = sign;
            else if (exp == 31) bits = sign | 0x7F800000 | (mant << 13);
            else bits = sign | ((exp + 112) << 23) | (mant << 13);
            memcpy(&out[i], &bits, 4);
        }
    } else {
        fprintf(stderr, "ERROR: tensor '%s' unsupported dtype %s\n", name, dtype);
        return NULL;
    }
    if (out_nelem) *out_nelem = nelem;
    return out;
}

static inline float cubic_w(float t) {
    const float a = -0.75f;
    t = t < 0.f ? -t : t;
    if (t <= 1.f) return ((a + 2.f) * t - (a + 3.f)) * t * t + 1.f;
    if (t <= 2.f) return (((t - 5.f) * t + 8.f) * t - 4.f) * a;
    return 0.f;
}

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
        float wy[4] = { cubic_w(1.f + ty), cubic_w(ty), cubic_w(1.f - ty), cubic_w(2.f - ty) };
        int ys[4] = { iy0 - 1, iy0, iy0 + 1, iy0 + 2 };
        for (int i = 0; i < 4; i++) {
            if (ys[i] < 0) ys[i] = 0;
            if (ys[i] >= STORED_GRID) ys[i] = STORED_GRID - 1;
        }
        for (int ox = 0; ox < GRID; ox++) {
            float fx = ((float)ox + 0.5f) * sx - 0.5f;
            int ix0 = (int)floorf(fx);
            float tx = fx - (float)ix0;
            float wx[4] = { cubic_w(1.f + tx), cubic_w(tx), cubic_w(1.f - tx), cubic_w(2.f - tx) };
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
                    for (int i = 0; i < 4; i++)
                        rv += row[xs[i] * HIDDEN + c] * wx[i];
                    acc += rv * wy[j];
                }
                out[c] = acc;
            }
        }
    }
}

static int dinov2g_quantize_w_fp8(dinov2g *r, CUdeviceptr d_w_f16, size_t n,
                                   CUdeviceptr *out_fp8, CUdeviceptr *out_scale) {
    *out_fp8 = 0; *out_scale = 0;
    if (!r->use_bf16_gemm) return -1;
    CUdeviceptr d_f32 = 0, d_max = 0, d_fp8 = 0, d_scale = 0;
    if (cuMemAlloc(&d_f32, n * sizeof(float)) != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_max, sizeof(float))     != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_fp8, n)                 != CUDA_SUCCESS) goto fail;
    if (cuMemAlloc(&d_scale, sizeof(float))   != CUDA_SUCCESS) goto fail;
    int ni = (int)n;
    {
        unsigned grid = (unsigned)((ni + 255) / 256);
        void *args[] = {&d_f32, &d_w_f16, &ni};
        cuLaunchKernel(r->k_f16_to_f32, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }
    cuMemsetD8Async(d_max, 0, sizeof(float), r->stream);
    {
        unsigned grid = (unsigned)((ni + 255) / 256);
        void *args[] = {&d_max, &d_f32, &ni};
        cuLaunchKernel(r->k_reduce_max_abs, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }
    {
        unsigned grid = (unsigned)((ni + 255) / 256);
        void *args[] = {&d_fp8, &d_scale, &d_f32, &d_max, &ni};
        cuLaunchKernel(r->k_quantize_fp8, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }
    cuStreamSynchronize(r->stream);
    cuMemFree(d_f32); cuMemFree(d_max);
    *out_fp8 = d_fp8; *out_scale = d_scale;
    return 0;
fail:
    if (d_f32)   cuMemFree(d_f32);
    if (d_max)   cuMemFree(d_max);
    if (d_fp8)   cuMemFree(d_fp8);
    if (d_scale) cuMemFree(d_scale);
    return -1;
}

static int dinov2g_load(dinov2g *r, const char *path) {
    fprintf(stderr, "DINOv2-giant: loading %s\n", path);
    st_context *st = safetensors_open(path);
    if (!st) { fprintf(stderr, "ERROR: cannot open %s\n", path); return -1; }
    fprintf(stderr, "DINOv2-giant: %d tensors\n", st->n_tensors);

    size_t n = 0;
    float *patch_w = st_fetch_f32(st, "embeddings.patch_embeddings.projection.weight", &n);
    float *patch_b = st_fetch_f32(st, "embeddings.patch_embeddings.projection.bias", NULL);
    float *cls     = st_fetch_f32(st, "embeddings.cls_token", NULL);
    float *pos_raw = st_fetch_f32(st, "embeddings.position_embeddings", NULL);
    if (!patch_w || !patch_b || !cls || !pos_raw) return -1;

    r->patch_w   = upload_f32(patch_w, (size_t)HIDDEN * 3 * PATCH * PATCH);
    r->patch_b   = upload_f32(patch_b, HIDDEN);
    r->cls_token = upload_f32(cls, HIDDEN);

    float *pos_small = (float *)malloc(SEQ_LEN * HIDDEN * sizeof(float));
    interpolate_pos_embed(pos_raw, pos_small);
    r->pos_embed = upload_f32(pos_small, SEQ_LEN * HIDDEN);
    free(pos_small);
    free(patch_w); free(patch_b); free(cls); free(pos_raw);

    float *ln_w = st_fetch_f32(st, "layernorm.weight", NULL);
    float *ln_b = st_fetch_f32(st, "layernorm.bias", NULL);
    r->final_ln_w = upload_f32(ln_w, HIDDEN);
    r->final_ln_b = upload_f32(ln_b, HIDDEN);
    free(ln_w); free(ln_b);

    for (int i = 0; i < LAYERS; i++) {
        char nm[256];
        layer_gpu *l = &r->layers[i];
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
        FF32(ln1_w, "norm1.weight"); FF32(ln1_b, "norm1.bias");
        FF32(ln2_w, "norm2.weight"); FF32(ln2_b, "norm2.bias");
        FF32(ls1, "layer_scale1.lambda1");
        FF32(ls2, "layer_scale2.lambda1");
        FF32(q_b,   "attention.attention.query.bias");
        FF32(k_b,   "attention.attention.key.bias");
        FF32(v_b,   "attention.attention.value.bias");
        FF32(out_b, "attention.output.dense.bias");
        FF16(q_w,   "attention.attention.query.weight",  (size_t)HIDDEN * HIDDEN);
        FF16(k_w,   "attention.attention.key.weight",    (size_t)HIDDEN * HIDDEN);
        FF16(v_w,   "attention.attention.value.weight",  (size_t)HIDDEN * HIDDEN);
        FF16(out_w, "attention.output.dense.weight",     (size_t)HIDDEN * HIDDEN);
        snprintf(nm, sizeof(nm), "encoder.layer.%d.mlp.weights_in.bias", i);
        { float *t = st_fetch_f32(st, nm, NULL); l->w_in_b = upload_f32(t, FFN_FULL); free(t); }
        snprintf(nm, sizeof(nm), "encoder.layer.%d.mlp.weights_out.bias", i);
        { float *t = st_fetch_f32(st, nm, NULL); l->w_out_b = upload_f32(t, HIDDEN); free(t); }
        FF16(w_in_w,  "mlp.weights_in.weight",  (size_t)FFN_FULL * HIDDEN);
        FF16(w_out_w, "mlp.weights_out.weight", (size_t)HIDDEN * FFN_HALF);
#undef FF32
#undef FF16
        if (r->use_bf16_gemm) {
            dinov2g_quantize_w_fp8(r, l->q_w,   (size_t)HIDDEN * HIDDEN, &l->q_w_fp8,   &l->q_w_scale);
            dinov2g_quantize_w_fp8(r, l->k_w,   (size_t)HIDDEN * HIDDEN, &l->k_w_fp8,   &l->k_w_scale);
            dinov2g_quantize_w_fp8(r, l->v_w,   (size_t)HIDDEN * HIDDEN, &l->v_w_fp8,   &l->v_w_scale);
            dinov2g_quantize_w_fp8(r, l->out_w, (size_t)HIDDEN * HIDDEN, &l->out_w_fp8, &l->out_w_scale);
            dinov2g_quantize_w_fp8(r, l->w_in_w,  (size_t)FFN_FULL * HIDDEN, &l->w_in_fp8,  &l->w_in_scale);
            dinov2g_quantize_w_fp8(r, l->w_out_w, (size_t)HIDDEN * FFN_HALF, &l->w_out_fp8, &l->w_out_scale);
        }
    }
    safetensors_close(st);
    fprintf(stderr, "DINOv2-giant: weights loaded%s\n",
            r->use_bf16_gemm ? " (+ FP8 prequant)" : "");
    return 0;
}

static void launch_layernorm(dinov2g *r, CUdeviceptr dst, CUdeviceptr src,
                              CUdeviceptr w, CUdeviceptr b, int n_tok, int dim) {
    float eps = 1e-6f;
    void *args[] = { &dst, &src, &w, &b, &dim, &eps };
    cuLaunchKernel(r->k_layernorm, n_tok, 1, 1, 256, 1, 1,
                   256 * sizeof(float), r->stream, args, NULL);
}

static void launch_gemm(dinov2g *r, CUdeviceptr y, CUdeviceptr w_f16,
                         CUdeviceptr w_fp8, CUdeviceptr w_scale,
                         CUdeviceptr x, CUdeviceptr bias,
                         int n_out, int n_in, int n_tok) {
    if (r->use_bf16_gemm && w_fp8 && w_scale &&
        n_tok >= 16 && (n_in % 32) == 0 && (n_out % 256) == 0) {
        void *args[] = {&y, &w_fp8, &x, &bias, &n_out, &n_in, &n_tok, &w_scale};
        unsigned gx = (unsigned)((n_out + 255) / 256);
        gx = (gx + 3u) & ~3u;
        if (r->use_bf16_mt4 && r->k_gemm_bf16_mt4) {
            unsigned gy = (unsigned)((n_tok + 63) / 64);
            gy = (gy + 3u) & ~3u;
            size_t smem = 4096 + 8192 * 2;
            cuLaunchKernel(r->k_gemm_bf16_mt4, gx, gy, 1, 128, 1, 1,
                           smem, r->stream, args, NULL);
            return;
        }
        if (r->k_gemm_bf16) {
            unsigned gy = (unsigned)((n_tok + 31) / 32);
            gy = (gy + 3u) & ~3u;
            size_t smem = 2048 + 8192 * 2;
            cuLaunchKernel(r->k_gemm_bf16, gx, gy, 1, 128, 1, 1,
                           smem, r->stream, args, NULL);
            return;
        }
    }
    void *args[] = { &y, &w_f16, &x, &bias, &n_out, &n_in, &n_tok };
    unsigned gx = (unsigned)((n_out + 63) / 64);
    unsigned gy = (unsigned)((n_tok + 15) / 16);
    cuLaunchKernel(r->k_gemm, gx, gy, 1, 16, 16, 1, 0, r->stream, args, NULL);
}

static void launch_cast_f32_to_bf16(dinov2g *r, CUdeviceptr dst, CUdeviceptr src,
                                     int n) {
    void *args[] = {&src, &dst, &n};
    cuLaunchKernel(r->k_cast_f32_to_bf16, (unsigned)((n + 255) / 256), 1, 1,
                   256, 1, 1, 0, r->stream, args, NULL);
}

static void launch_attn(dinov2g *r, CUdeviceptr out, CUdeviceptr Q,
                         CUdeviceptr K, CUdeviceptr V, int n_tok) {
    int dim = HIDDEN, n_heads = HEADS, head_dim = HEAD_DIM;
    if (r->use_bf16_attn && r->k_flash_attn_hd64 && r->d_qkv_bf16) {
        int n_elem = n_tok * dim;
        CUdeviceptr d_q = r->d_qkv_bf16;
        CUdeviceptr d_k = r->d_qkv_bf16 + (CUdeviceptr)((size_t)n_elem * sizeof(uint16_t));
        CUdeviceptr d_v = r->d_qkv_bf16 + (CUdeviceptr)((size_t)2 * n_elem * sizeof(uint16_t));
        launch_cast_f32_to_bf16(r, d_q, Q, n_elem);
        launch_cast_f32_to_bf16(r, d_k, K, n_elem);
        launch_cast_f32_to_bf16(r, d_v, V, n_elem);
        unsigned gy = (unsigned)((n_tok + 63) / 64);
        size_t smem = (size_t)(4 * 32 * 72 * 2);
        void *args[] = {&out, &d_q, &d_k, &d_v, &n_tok, &n_heads, &head_dim};
        cuLaunchKernel(r->k_flash_attn_hd64, (unsigned)n_heads, gy, 1,
                       128, 1, 1, smem, r->stream, args, NULL);
        return;
    }
    int q_len = n_tok, kv_len = n_tok;
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

paint_stage_dinov2g *paint_stage_dinov2g_create(CUdevice dev,
                                                 const char *weights_path) {
    dinov2g *r = (dinov2g *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->dev = dev;
    /* Reuse current ctx if any; else create. */
    if (cuCtxGetCurrent(&r->ctx) != CUDA_SUCCESS || r->ctx == NULL) {
        cuCtxCreate(&r->ctx, 0, dev);
        r->owns_ctx = 1;
    }
    cuStreamCreate(&r->stream, 0);

    const char *mma_open  =
        "\nextern \"C\" {\n"
        "__device__ __constant__ unsigned short d_fp8_to_bf16_lut[256];\n"
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
    size_t l1 = strlen(cuda_kernels_common_src);
    size_t l2 = strlen(cuda_hy3d_specific_kernels);
    size_t lo = strlen(mma_open);
    size_t lm = strlen(fp8_mma_kernels_src);
    size_t lc = strlen(mma_close);
    size_t l3 = strlen(cuda_paint_nn_kernels_src);
    char *src = (char *)malloc(l1 + l2 + lo + lm + lc + l3 + 1);
    char *p = src;
    memcpy(p, cuda_kernels_common_src, l1);    p += l1;
    memcpy(p, cuda_hy3d_specific_kernels, l2); p += l2;
    memcpy(p, mma_open, lo);                   p += lo;
    memcpy(p, fp8_mma_kernels_src, lm);        p += lm;
    memcpy(p, mma_close, lc);                  p += lc;
    memcpy(p, cuda_paint_nn_kernels_src, l3);  p += l3;
    *p = '\0';

    r->sm = cu_compile_kernels(&r->mod, r->dev, src,
                                "dinov2g_kernels", 1, "DINOv2-G");
    free(src);
    if (r->sm < 0) { free(r); return NULL; }

    cuModuleGetFunction(&r->k_patch_embed,    r->mod, "patch_embed_conv2d");
    cuModuleGetFunction(&r->k_cls_pos_embed,  r->mod, "cls_pos_embed");
    cuModuleGetFunction(&r->k_layernorm,      r->mod, "layernorm_f32");
    cuModuleGetFunction(&r->k_gemm,           r->mod, "gemm_tiled_f16_f32");
    cuModuleGetFunction(&r->k_attn,           r->mod, "cross_attn_f32");
    cuModuleGetFunction(&r->k_layerscale_add, r->mod, "layerscale_add_f32");
    cuModuleGetFunction(&r->k_add,            r->mod, "add_f32");
    cuModuleGetFunction(&r->k_silu_gate,      r->mod, "split_silu_gate_f32");
#define GET_OPT(name, fld) \
    if (cuModuleGetFunction(&r->fld, r->mod, name) != CUDA_SUCCESS) r->fld = NULL;
    GET_OPT("gemm_bf16_pipe_mt4_scaled_f32", k_gemm_bf16_mt4);
    GET_OPT("gemm_bf16_pipe_scaled_f32",     k_gemm_bf16);
    GET_OPT("flash_attn_bf16_hd64",          k_flash_attn_hd64);
    GET_OPT("reduce_max_abs_f32",            k_reduce_max_abs);
    GET_OPT("quantize_to_fp8_e4m3",          k_quantize_fp8);
    GET_OPT("cast_f32_to_bf16",              k_cast_f32_to_bf16);
    GET_OPT("f16_to_f32_buf",                k_f16_to_f32);
#undef GET_OPT

    {
        CUdeviceptr d_lut; size_t lut_sz;
        if (cuModuleGetGlobal(&d_lut, &lut_sz, r->mod, "d_fp8_to_bf16_lut") == CUDA_SUCCESS &&
            lut_sz == 256 * sizeof(uint16_t)) {
            uint16_t lut[256];
            for (int i = 0; i < 256; i++) {
                int sign = (i >> 7) & 1;
                int exp  = (i >> 3) & 0xF;
                int mant = i & 0x7;
                float v;
                if (exp == 0 && mant == 0) v = 0.f;
                else if (exp == 15 && mant == 7) v = 0.f;
                else if (exp == 0) v = ((float)mant / 8.f) * (1.f / 64.f);
                else v = (1.f + (float)mant / 8.f) * exp2f((float)(exp - 7));
                if (sign) v = -v;
                uint32_t b; memcpy(&b, &v, 4);
                uint32_t rb = 0x7FFFu + ((b >> 16) & 1u);
                lut[i] = (uint16_t)((b + rb) >> 16);
            }
            cuMemcpyHtoD(d_lut, lut, sizeof(lut));
        }
    }

    {
        const char *e;
        e = getenv("HY3D_BF16_ATTN");
        r->use_bf16_attn = (r->k_flash_attn_hd64 && r->k_cast_f32_to_bf16 &&
                            r->sm >= 80) ? ((e && e[0] == '0') ? 0 : 1) : 0;
        e = getenv("HY3D_BF16_GEMM");
        int bf16_gemm_ok = (r->k_gemm_bf16 && r->k_f16_to_f32 &&
                            r->k_reduce_max_abs && r->k_quantize_fp8 &&
                            r->sm >= 80);
        r->use_bf16_gemm = bf16_gemm_ok ? ((e && e[0] == '1') ? 1 : 0) : 0;
        e = getenv("HY3D_BF16_MT4");
        r->use_bf16_mt4 = (r->use_bf16_gemm && r->k_gemm_bf16_mt4)
                          ? ((e && e[0] == '0') ? 0 : 1) : 0;
        fprintf(stderr, "DINOv2-giant dispatch: BF16_ATTN=%d BF16_GEMM=%d MT4=%d sm=%d\n",
                r->use_bf16_attn, r->use_bf16_gemm, r->use_bf16_mt4, r->sm);
    }

    cuMemAlloc(&r->d_hidden, SEQ_LEN * HIDDEN * sizeof(float));
    cuMemAlloc(&r->d_normed, SEQ_LEN * HIDDEN * sizeof(float));
    cuMemAlloc(&r->d_qkv,    3 * SEQ_LEN * HIDDEN * sizeof(float));
    cuMemAlloc(&r->d_attn,   SEQ_LEN * HIDDEN * sizeof(float));
    cuMemAlloc(&r->d_h_in,   SEQ_LEN * FFN_FULL * sizeof(float));
    cuMemAlloc(&r->d_h_mid,  SEQ_LEN * FFN_HALF * sizeof(float));
    if (r->use_bf16_attn) {
        cuMemAlloc(&r->d_qkv_bf16,
                   (size_t)3 * SEQ_LEN * HIDDEN * sizeof(uint16_t));
    }

    if (dinov2g_load(r, weights_path) != 0) {
        paint_stage_dinov2g_destroy(r);
        return NULL;
    }
    return r;
}

void paint_stage_dinov2g_run(paint_stage_dinov2g *r,
                              const float *image_f32, float *out_f32) {
    size_t img_bytes = (size_t)3 * IMG_SIZE * IMG_SIZE * sizeof(float);
    CUdeviceptr d_image;
    cuMemAlloc(&d_image, img_bytes);
    cuMemcpyHtoDAsync(d_image, image_f32, img_bytes, r->stream);

    {
        int gw = GRID, dim = HIDDEN, ps = PATCH, img_w = IMG_SIZE;
        CUdeviceptr pw = r->patch_w, pb = r->patch_b;
        void *args[] = { &r->d_hidden, &d_image, &pw, &pb,
                         &gw, &dim, &ps, &img_w };
        cuLaunchKernel(r->k_patch_embed, (unsigned)(GRID * GRID), 1, 1,
                       256, 1, 1, 0, r->stream, args, NULL);
    }
    {
        int n_tok = SEQ_LEN, dim = HIDDEN;
        CUdeviceptr cls = r->cls_token, pos = r->pos_embed;
        void *args[] = { &r->d_hidden, &cls, &pos, &n_tok, &dim };
        cuLaunchKernel(r->k_cls_pos_embed,
                       (unsigned)((SEQ_LEN * HIDDEN + 255) / 256), 1, 1,
                       256, 1, 1, 0, r->stream, args, NULL);
    }

    CUdeviceptr d_Q = r->d_qkv;
    CUdeviceptr d_K = r->d_qkv + (size_t)SEQ_LEN * HIDDEN * sizeof(float);
    CUdeviceptr d_V = r->d_qkv + (size_t)2 * SEQ_LEN * HIDDEN * sizeof(float);

    for (int i = 0; i < LAYERS; i++) {
        layer_gpu *l = &r->layers[i];
        launch_layernorm(r, r->d_normed, r->d_hidden, l->ln1_w, l->ln1_b, SEQ_LEN, HIDDEN);
        launch_gemm(r, d_Q, l->q_w, l->q_w_fp8, l->q_w_scale,
                    r->d_normed, l->q_b, HIDDEN, HIDDEN, SEQ_LEN);
        launch_gemm(r, d_K, l->k_w, l->k_w_fp8, l->k_w_scale,
                    r->d_normed, l->k_b, HIDDEN, HIDDEN, SEQ_LEN);
        launch_gemm(r, d_V, l->v_w, l->v_w_fp8, l->v_w_scale,
                    r->d_normed, l->v_b, HIDDEN, HIDDEN, SEQ_LEN);
        launch_attn(r, r->d_attn, d_Q, d_K, d_V, SEQ_LEN);
        launch_gemm(r, r->d_normed, l->out_w, l->out_w_fp8, l->out_w_scale,
                    r->d_attn, l->out_b, HIDDEN, HIDDEN, SEQ_LEN);
        launch_layerscale_add(r, r->d_hidden, r->d_normed, l->ls1,
                               SEQ_LEN * HIDDEN, HIDDEN);

        launch_layernorm(r, r->d_normed, r->d_hidden, l->ln2_w, l->ln2_b, SEQ_LEN, HIDDEN);
        launch_gemm(r, r->d_h_in, l->w_in_w, l->w_in_fp8, l->w_in_scale,
                    r->d_normed, l->w_in_b, FFN_FULL, HIDDEN, SEQ_LEN);
        launch_silu_gate(r, r->d_h_in, r->d_h_mid, SEQ_LEN, FFN_HALF);
        launch_gemm(r, r->d_normed, l->w_out_w, l->w_out_fp8, l->w_out_scale,
                    r->d_h_mid, l->w_out_b, HIDDEN, FFN_HALF, SEQ_LEN);
        launch_layerscale_add(r, r->d_hidden, r->d_normed, l->ls2,
                               SEQ_LEN * HIDDEN, HIDDEN);
    }

    CUdeviceptr d_out;
    cuMemAlloc(&d_out, SEQ_LEN * HIDDEN * sizeof(float));
    launch_layernorm(r, d_out, r->d_hidden, r->final_ln_w, r->final_ln_b,
                      SEQ_LEN, HIDDEN);
    cuStreamSynchronize(r->stream);
    cuMemcpyDtoH(out_f32, d_out, (size_t)SEQ_LEN * HIDDEN * sizeof(float));
    cuMemFree(d_out);
    cuMemFree(d_image);
}

void paint_stage_dinov2g_destroy(paint_stage_dinov2g *r) {
    if (!r) return;
    /* Not bothering to free per-layer device buffers individually — the
     * runner is shut down with the context, mirroring sibling stages. */
    if (r->d_hidden)   cuMemFree(r->d_hidden);
    if (r->d_normed)   cuMemFree(r->d_normed);
    if (r->d_qkv)      cuMemFree(r->d_qkv);
    if (r->d_attn)     cuMemFree(r->d_attn);
    if (r->d_h_in)     cuMemFree(r->d_h_in);
    if (r->d_h_mid)    cuMemFree(r->d_h_mid);
    if (r->d_qkv_bf16) cuMemFree(r->d_qkv_bf16);
    if (r->mod)    cuModuleUnload(r->mod);
    if (r->stream) cuStreamDestroy(r->stream);
    if (r->owns_ctx && r->ctx) cuCtxDestroy(r->ctx);
    free(r);
}
