/* S3 dense FP8/E8M0 HIPRTC bring-up runner. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "hip_ds4f_dense.h"
#include "../../common/ds4f.h"
#include "hip_ds4f_kernels.h"
#include "../../rdna4/hip_kernels_common.h"
#include "../../rdna4/rocew.h"

#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../../rdna4/hip_runner_common.h"

#include <math.h>
#include <limits.h>

enum { HIP_DS4F_ASYNC_MAX = 2, HIP_DS4F_GEMM_MAX = 128 };

typedef struct {
    void *dw, *ds;
    const void *hw, *hs;
    int rows, cols, scale_cols, kind, owner;
} hip_ds4f_matrix;

enum {
    HIP_DS4F_MATRIX_FP8 = 0,
    HIP_DS4F_MATRIX_BF16 = 1,
    HIP_DS4F_MATRIX_FP8_BF16 = 2,
    HIP_DS4F_MATRIX_FP8_FP16 = 3,
    HIP_DS4F_MATRIX_FP8_ORDERED = 4,
    HIP_DS4F_MATRIX_MXFP4 = 5,
    HIP_DS4F_MATRIX_FP8_ROWSCALE = 6
};

static int matrix_is_bf16(int kind) {
    return kind == HIP_DS4F_MATRIX_BF16 || kind == HIP_DS4F_MATRIX_FP8_BF16;
}

static int matrix_is_fp16(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8_FP16;
}

static int matrix_is_fp8(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8 || kind == HIP_DS4F_MATRIX_FP8_ORDERED ||
           kind == HIP_DS4F_MATRIX_FP8_ROWSCALE;
}

static int matrix_is_mxfp4(int kind) {
    return kind == HIP_DS4F_MATRIX_MXFP4;
}

static int matrix_is_fp8_ordered(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8_ORDERED;
}

static int matrix_is_fp8_rowscale(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8_ROWSCALE;
}

static int matrix_is_fp8_promoted(int kind) {
    return kind == HIP_DS4F_MATRIX_FP8_BF16 ||
           kind == HIP_DS4F_MATRIX_FP8_FP16;
}

struct hip_ds4f_dense {
    ds4f_mem_pool *mem;
    int device_id;
    int verbose;
    hipModule_t module;
    hipFunction_t matvec;
    hipFunction_t bf16_matvec;
    hipFunction_t f16_matvec;
    hipFunction_t blockdiag_matvec;
    hipFunction_t gemm_fp8;
    hipFunction_t gemm_fp8_ordered;
    hipFunction_t gemm_mxfp4;
    hipFunction_t gemm_fp8_rowscale;
    hipFunction_t gemm_bf16;
    hipFunction_t gemm_f16;

    hip_ds4f_matrix *matrices;
    int n_matrices, cap_matrices;
    int current;
    void *dx, *dy;
    size_t x_bytes, y_bytes;

    hipStream_t stream;
    hipEvent_t done;
    int block_threads;
    int pending;
    int pending_id;
    int pending_rows;

    hipStream_t multi_stream[HIP_DS4F_ASYNC_MAX];
    hipEvent_t multi_done[HIP_DS4F_ASYNC_MAX];
    void *multi_dx[HIP_DS4F_ASYNC_MAX], *multi_dy[HIP_DS4F_ASYNC_MAX];
    size_t multi_x_bytes[HIP_DS4F_ASYNC_MAX], multi_y_bytes[HIP_DS4F_ASYNC_MAX];
    size_t multi_out_bytes[HIP_DS4F_ASYNC_MAX];
    float *multi_y[HIP_DS4F_ASYNC_MAX];
    int multi_pending, multi_n;

    void *gemm_dx, *gemm_dy;
    void *fp8_lut;
    size_t gemm_x_bytes, gemm_y_bytes;
    float *gemm_x_pack, *gemm_y_pack;
    size_t gemm_x_pack_bytes, gemm_y_pack_bytes;
    void *gemm_multi_dy[HIP_DS4F_GEMM_MAX];
    size_t gemm_multi_y_bytes[HIP_DS4F_GEMM_MAX];
    const ds4f_layer *stream_layer;
    void *stream_dw, *stream_ds;
};

static int valid_dims(int rows, int cols) {
    return rows > 0 && cols > 0 && rows <= INT_MAX / cols;
}

static void clear_matrices(hip_ds4f_dense *ctx) {
    for (int i = 0; i < ctx->n_matrices; ++i) {
        if (ctx->matrices[i].dw && !ctx->matrices[i].owner) hipFree(ctx->matrices[i].dw);
        if (ctx->matrices[i].ds && !ctx->matrices[i].owner) hipFree(ctx->matrices[i].ds);
    }
    ctx->n_matrices = 0;
    ctx->current = -1;
    ctx->stream_layer = NULL;
    if (ctx->stream_dw) hipFree(ctx->stream_dw);
    if (ctx->stream_ds) hipFree(ctx->stream_ds);
    ctx->stream_dw = ctx->stream_ds = NULL;
}

static void release_matrix(hip_ds4f_dense *ctx, int id) {
    if (!ctx || id < 0 || id >= ctx->n_matrices) return;
    if (ctx->matrices[id].dw && !ctx->matrices[id].owner) hipFree(ctx->matrices[id].dw);
    if (ctx->matrices[id].ds && !ctx->matrices[id].owner) hipFree(ctx->matrices[id].ds);
    memset(&ctx->matrices[id], 0, sizeof(ctx->matrices[id]));
    ctx->matrices[id].kind = -1;
}

static int append_device_matrix(hip_ds4f_dense *ctx, void *dw, void *ds,
                                const void *hw, const void *hs,
                                int rows, int cols, int scale_cols,
                                int kind, int owner) {
    int id = -1;
    for (int i = 0; i < ctx->n_matrices; ++i)
        if (!ctx->matrices[i].dw && !ctx->matrices[i].ds) { id = i; break; }
    if (id < 0 && ctx->n_matrices == ctx->cap_matrices) {
        int cap = ctx->cap_matrices ? ctx->cap_matrices * 2 : 8;
        if (cap < ctx->n_matrices || cap > INT_MAX / (int)sizeof(*ctx->matrices)) return -1;
        hip_ds4f_matrix *p = (hip_ds4f_matrix *)ds4f_mem_realloc(
            ctx->mem, ctx->matrices,
            (size_t)ctx->cap_matrices * sizeof(*ctx->matrices),
            (size_t)cap * sizeof(*ctx->matrices), 64);
        if (!p) return -1;
        ctx->matrices = p; ctx->cap_matrices = cap;
    }
    if (id < 0) id = ctx->n_matrices++;
    ctx->matrices[id] = (hip_ds4f_matrix){ dw, ds, hw, hs, rows, cols, scale_cols, kind, owner };
    ctx->current = id;
    return id;
}

hip_ds4f_dense *hip_ds4f_dense_create_ex(int device_id, int verbose, int precise_math) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_ds4f_dense: failed to initialize HIP/hipRTC\n");
        return NULL;
    }
    if (hipSetDevice(device_id) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: cannot select device %d\n", device_id);
        return NULL;
    }

    ds4f_mem_pool *mem = ds4f_mem_pool_create();
    if (!mem) return NULL;
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)ds4f_mem_calloc(mem, 1, sizeof(*ctx), 64);
    if (!ctx) {
        ds4f_mem_pool_destroy(mem);
        return NULL;
    }
    ctx->mem = mem;
    ctx->device_id = device_id;
    ctx->verbose = verbose;
    ctx->current = -1;
    /* One row is one block. 128 threads is the best measured gfx1201 point:
     * it halves the wave reduction/launch footprint versus the original 256,
     * while retaining enough lanes for the 1K--8K column projections. */
    ctx->block_threads = 128;
    const char *threads_env = getenv("DS4F_HIP_BLOCK_THREADS");
    if (threads_env) {
        int threads = atoi(threads_env);
        if (threads == 64 || threads == 128 || threads == 256)
            ctx->block_threads = threads;
        else
            fprintf(stderr, "hip_ds4f_dense: ignoring invalid DS4F_HIP_BLOCK_THREADS=%s (use 64, 128, or 256)\n",
                    threads_env);
    }
    size_t common_len = strlen(hip_kernels_common_src);
    size_t dense_len = strlen(hip_ds4f_dense_kernels_src);
    char *full_src = (char *)ds4f_mem_alloc(ctx->mem, common_len + dense_len + 2, 64, 0);
    if (!full_src) {
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    memcpy(full_src, hip_kernels_common_src, common_len);
    memcpy(full_src + common_len, hip_ds4f_dense_kernels_src, dense_len);
    full_src[common_len + dense_len] = '}';
    full_src[common_len + dense_len + 1] = '\0';
    if (hip_compile_kernels_ex(&ctx->module, device_id, full_src,
                            "hip_ds4f_dense.hip", verbose,
                            "hip_ds4f_dense", precise_math) < 0 ||
        hipModuleGetFunction(&ctx->matvec, ctx->module,
                             "ds4f_dense_fp8_matvec") != hipSuccess ||
        hipModuleGetFunction(&ctx->bf16_matvec, ctx->module,
                             "ds4f_dense_bf16_matvec") != hipSuccess ||
        hipModuleGetFunction(&ctx->f16_matvec, ctx->module,
                             "ds4f_dense_f16_matvec") != hipSuccess ||
        hipModuleGetFunction(&ctx->blockdiag_matvec, ctx->module,
                             "ds4f_dense_fp8_blockdiag") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_fp8, ctx->module,
                             "ds4f_dense_fp8_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_fp8_ordered, ctx->module,
                             "ds4f_dense_fp8_ordered_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_mxfp4, ctx->module,
                             "ds4f_dense_mxfp4_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_bf16, ctx->module,
                             "ds4f_dense_bf16_gemm") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_f16, ctx->module,
                             "gemm_tiled_f16_f32") != hipSuccess ||
        hipModuleGetFunction(&ctx->gemm_fp8_rowscale, ctx->module,
                             "ds4f_dense_fp8_rowscale_gemm") != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: failed to build matvec module\n");
        if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    {
        uint32_t lut[256];
        ds4f_init_fp8_e4m3fn_lut(lut);
        if (hipMalloc(&ctx->fp8_lut, sizeof(lut)) != hipSuccess ||
            hipMemcpy(ctx->fp8_lut, lut, sizeof(lut), hipMemcpyHostToDevice) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: FP8 LUT upload failed\n");
            if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
            if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
            ds4f_mem_pool_destroy(ctx->mem);
            return NULL;
        }
    }
    if (hipStreamCreate(&ctx->stream) != hipSuccess ||
        hipEventCreate(&ctx->done) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: failed to create stream/event\n");
        if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
        if (ctx->stream && hipStreamDestroy) hipStreamDestroy(ctx->stream);
        if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
        ds4f_mem_pool_destroy(ctx->mem);
        return NULL;
    }
    for (int i = 0; i < HIP_DS4F_ASYNC_MAX; ++i) {
        if (hipStreamCreate(&ctx->multi_stream[i]) != hipSuccess ||
            hipEventCreate(&ctx->multi_done[i]) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: failed to create async streams/events\n");
            for (int j = 0; j <= i; ++j) {
                if (ctx->multi_done[j] && hipEventDestroy) hipEventDestroy(ctx->multi_done[j]);
                if (ctx->multi_stream[j] && hipStreamDestroy) hipStreamDestroy(ctx->multi_stream[j]);
            }
            if (ctx->done && hipEventDestroy) hipEventDestroy(ctx->done);
            if (ctx->stream && hipStreamDestroy) hipStreamDestroy(ctx->stream);
            if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
            if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
            ds4f_mem_pool_destroy(ctx->mem);
            return NULL;
        }
    }
    return ctx;
}

hip_ds4f_dense *hip_ds4f_dense_create(int device_id, int verbose) {
    return hip_ds4f_dense_create_ex(device_id, verbose, -1);
}

void hip_ds4f_dense_destroy(hip_ds4f_dense *ctx) {
    if (!ctx) return;
    if (ctx->pending && ctx->done) hipEventSynchronize(ctx->done);
    if (ctx->multi_pending) {
        for (int i = 0; i < ctx->multi_n; ++i)
            if (ctx->multi_done[i]) hipEventSynchronize(ctx->multi_done[i]);
    }
    clear_matrices(ctx);
    if (ctx->dx) hipFree(ctx->dx);
    if (ctx->dy) hipFree(ctx->dy);
    if (ctx->gemm_dx) hipFree(ctx->gemm_dx);
    if (ctx->gemm_dy) hipFree(ctx->gemm_dy);
    if (ctx->fp8_lut) hipFree(ctx->fp8_lut);
    for (int i = 0; i < HIP_DS4F_GEMM_MAX; ++i)
        if (ctx->gemm_multi_dy[i]) hipFree(ctx->gemm_multi_dy[i]);
    for (int i = 0; i < HIP_DS4F_ASYNC_MAX; ++i) {
        if (ctx->multi_dx[i]) hipFree(ctx->multi_dx[i]);
        if (ctx->multi_dy[i]) hipFree(ctx->multi_dy[i]);
        if (ctx->multi_done[i] && hipEventDestroy) hipEventDestroy(ctx->multi_done[i]);
        if (ctx->multi_stream[i] && hipStreamDestroy) hipStreamDestroy(ctx->multi_stream[i]);
    }
    if (ctx->done && hipEventDestroy) hipEventDestroy(ctx->done);
    if (ctx->stream && hipStreamDestroy) hipStreamDestroy(ctx->stream);
    if (ctx->module && hipModuleUnload) hipModuleUnload(ctx->module);
    ds4f_mem_pool_destroy(ctx->mem);
}

static int ensure_vectors(hip_ds4f_dense *ctx, int rows, int cols) {
    size_t x_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * sizeof(float);
    if (ctx->dx && ctx->dy && ctx->x_bytes >= x_bytes && ctx->y_bytes >= y_bytes)
        return 0;

    void *dx = NULL, *dy = NULL;
    if (hipMalloc(&dx, x_bytes) != hipSuccess ||
        hipMalloc(&dy, y_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: vector allocation failed\n");
        if (dx) hipFree(dx);
        if (dy) hipFree(dy);
        return -1;
    }
    if (ctx->dx) hipFree(ctx->dx);
    if (ctx->dy) hipFree(ctx->dy);
    ctx->dx = dx; ctx->dy = dy;
    ctx->x_bytes = x_bytes; ctx->y_bytes = y_bytes;
    return 0;
}

static int ensure_multi_vectors(hip_ds4f_dense *ctx, int slot, int rows, int cols) {
    size_t x_bytes = (size_t)cols * sizeof(float);
    size_t y_bytes = (size_t)rows * sizeof(float);
    if (ctx->multi_dx[slot] && ctx->multi_dy[slot] &&
        ctx->multi_x_bytes[slot] >= x_bytes && ctx->multi_y_bytes[slot] >= y_bytes)
        return 0;
    void *dx = NULL, *dy = NULL;
    if (hipMalloc(&dx, x_bytes) != hipSuccess ||
        hipMalloc(&dy, y_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: async vector allocation failed\n");
        if (dx) hipFree(dx);
        if (dy) hipFree(dy);
        return -1;
    }
    if (ctx->multi_dx[slot]) hipFree(ctx->multi_dx[slot]);
    if (ctx->multi_dy[slot]) hipFree(ctx->multi_dy[slot]);
    ctx->multi_dx[slot] = dx;
    ctx->multi_dy[slot] = dy;
    ctx->multi_x_bytes[slot] = x_bytes;
    ctx->multi_y_bytes[slot] = y_bytes;
    return 0;
}

static int hip_ds4f_dense_add_storage(hip_ds4f_dense *ctx,
                                      const uint8_t *w, const uint8_t *s,
                                      int rows, int cols, int scale_cols,
                                      size_t w_bytes, size_t s_bytes, int kind) {
    if (!ctx || !w || !s || !valid_dims(rows, cols)) {
        fprintf(stderr, "hip_ds4f_dense: invalid matrix arguments\n");
        return -1;
    }
    if (ctx->pending || ctx->multi_pending) {
        fprintf(stderr, "hip_ds4f_dense: cannot add a matrix with work in flight\n");
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;

    void *dw = NULL, *ds = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess ||
        hipMalloc(&ds, s_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: matrix allocation failed\n");
        if (dw) hipFree(dw);
        if (ds) hipFree(ds);
        return -1;
    }
    if (hipMemcpy(dw, w, w_bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(ds, s, s_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: matrix upload failed\n");
        hipFree(dw); hipFree(ds);
        return -1;
    }
    int id = append_device_matrix(ctx, dw, ds, w, s, rows, cols,
                                  scale_cols, kind, 0);
    if (id < 0) { hipFree(dw); hipFree(ds); }
    return id;
}

int hip_ds4f_dense_add(hip_ds4f_dense *ctx,
                       const uint8_t *w, const uint8_t *s,
                       int rows, int cols) {
    int sc = (cols + 127) / 128;
    int sr = (rows + 127) / 128;
    return hip_ds4f_dense_add_storage(ctx, w, s, rows, cols, sc,
        (size_t)rows * (size_t)cols, (size_t)sr * (size_t)sc,
        HIP_DS4F_MATRIX_FP8);
}

int hip_ds4f_dense_bind_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!t) return -1;
    int id = hip_ds4f_dense_add(ctx, (const uint8_t *)t->w, t->scale,
                                t->rows, t->cols);
    if (id >= 0) t->gpu_id = id;
    return id;
}

int hip_ds4f_dense_bind_mxfp4_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_MXFP4 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || (t->cols & 31)) return -1;
    int id = hip_ds4f_dense_add_storage(ctx, (const uint8_t *)t->w,
        t->scale, t->rows, t->cols, t->cols / 32,
        (size_t)t->rows * (size_t)(t->cols / 2),
        (size_t)t->rows * (size_t)(t->cols / 32), HIP_DS4F_MATRIX_MXFP4);
    if (id >= 0) t->gpu_id = id;
    return id;
}

int hip_ds4f_dense_bind_mxfp4_widened_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    static const float lut[16] = { 0.f,1.f,2.f,3.f,4.f,6.f,8.f,12.f,
                                   0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-12.f };
    if (!ctx || !t || t->type != DS4F_MXFP4 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || (t->cols & 127)) return -1;
    int sc = (t->cols + 127) / 128;
    size_t wb = (size_t)t->rows * (size_t)t->cols;
    size_t sb = (size_t)t->rows * (size_t)sc;
    uint8_t *fw = (uint8_t *)ds4f_mem_alloc(ctx->mem, wb, 64, 0);
    uint8_t *fs = (uint8_t *)ds4f_mem_alloc(ctx->mem, sb, 64, 0);
    if (!fw || !fs) return -1;
    const uint8_t *w = (const uint8_t *)t->w, *s = t->scale;
    int rb = t->cols / 2, rs = t->cols / 32;
    for (int r = 0; r < t->rows; ++r) {
        for (int bc = 0; bc < sc; ++bc) {
            /* Raw MXFP4 stores the E8M0 byte one below the exponent byte
             * consumed by the regular FP8 kernels. Keep the exponent in
             * that byte domain while normalizing the FP8 payload. */
            int emax = 0;
            for (int b = 0; b < 4; ++b) {
                int e = s[(size_t)r * rs + bc * 4 + b];
                int se = e ? e - 1 : 0;
                if (se > emax) emax = se;
            }
            fs[(size_t)r * sc + bc] = (uint8_t)emax;
            for (int j = 0; j < 128; ++j) {
                int col = bc * 128 + j, block = col >> 5, p = col & 31;
                int off = p >> 1;
                uint8_t q = w[(size_t)r * rb + (size_t)block * 16 + off];
                int nib = (p & 1) ? q >> 4 : q & 15;
                int se = s[(size_t)r * rs + block] ?
                         (int)s[(size_t)r * rs + block] - 1 : 0;
                float v = s[(size_t)r * rs + block] == 0
                    ? 0.f : lut[nib] * ldexpf(1.0f, se - emax);
                fw[(size_t)r * t->cols + col] = hip_f32_to_fp8_e4m3(v);
            }
        }
    }
    int id = hip_ds4f_dense_add_storage(ctx, fw, fs, t->rows, t->cols,
        sc, wb, sb, HIP_DS4F_MATRIX_FP8_ROWSCALE);
    if (id >= 0) t->gpu_id = id;
    return id;
}

static int widen_mxfp4_host(const ds4f_tensor *t, uint8_t *fw, uint8_t *fs) {
    static const float lut[16] = { 0.f,1.f,2.f,3.f,4.f,6.f,8.f,12.f,
                                   0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-12.f };
    int sc = (t->cols + 127) / 128, rb = t->cols / 2, rs = t->cols / 32;
    const uint8_t *w = (const uint8_t *)t->w, *s = t->scale;
    for (int r = 0; r < t->rows; ++r) for (int bc = 0; bc < sc; ++bc) {
        int emax = 0;
        for (int b = 0; b < 4; ++b) {
            int e = s[(size_t)r * rs + bc * 4 + b];
            if (e && e - 1 > emax) emax = e - 1;
        }
        fs[(size_t)r * sc + bc] = (uint8_t)emax;
        for (int j = 0; j < 128; ++j) {
            int col = bc * 128 + j, block = col >> 5, p = col & 31;
            uint8_t q = w[(size_t)r * rb + (size_t)block * 16 + (p >> 1)];
            int nib = (p & 1) ? q >> 4 : q & 15;
            int e = s[(size_t)r * rs + block];
            float v = e ? lut[nib] * ldexpf(1.0f, (e - 1) - emax) : 0.f;
            fw[(size_t)r * t->cols + col] = hip_f32_to_fp8_e4m3(v);
        }
    }
    return 0;
}

static int stream_layer_impl(void *opaque, const ds4f_layer *layer, int raw) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !layer || ctx->pending || ctx->multi_pending) return -1;
    if (ctx->stream_layer == layer) return 0;
    if (ctx->stream_layer) {
        const ds4f_layer *old = ctx->stream_layer;
        const ds4f_tensor *old_ex[] = { old->ex_w1, old->ex_w2, old->ex_w3 };
        for (size_t w = 0; w < sizeof(old_ex) / sizeof(old_ex[0]); ++w)
            if (old_ex[w]) for (int e = 0; e < old->n_owned; ++e) {
                ds4f_tensor *t = (ds4f_tensor *)&old_ex[w][e];
                if (t->gpu_id >= 0) { release_matrix(ctx, t->gpu_id); t->gpu_id = -1; }
            }
    }
    ctx->stream_layer = NULL;
    if (ctx->stream_dw) hipFree(ctx->stream_dw);
    if (ctx->stream_ds) hipFree(ctx->stream_ds);
    ctx->stream_dw = ctx->stream_ds = NULL;
    const ds4f_tensor *ex[] = { layer->ex_w1, layer->ex_w2, layer->ex_w3 };
    size_t wtotal = 0, stotal = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi) {
        if (!ex[wi]) return -1;
        for (int e = 0; e < layer->n_owned; ++e) {
            const ds4f_tensor *t = &ex[wi][e];
            if (t->type != DS4F_MXFP4 || !valid_dims(t->rows, t->cols) || (t->cols & 127)) return -1;
            wtotal += raw ? (size_t)t->rows * (size_t)(t->cols / 2)
                          : (size_t)t->rows * (size_t)t->cols;
            stotal += raw ? (size_t)t->rows * (size_t)(t->cols / 32)
                          : (size_t)t->rows * (size_t)((t->cols + 127) / 128);
        }
    }
    uint8_t *hw = (uint8_t *)ds4f_mem_alloc(ctx->mem, wtotal, 256, 0);
    uint8_t *hs = (uint8_t *)ds4f_mem_alloc(ctx->mem, stotal, 256, 0);
    if (!hw || !hs || hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    size_t wo = 0, so = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi)
        for (int e = 0; e < layer->n_owned; ++e) {
            ds4f_tensor *t = (ds4f_tensor *)&ex[wi][e];
            size_t wb = raw ? (size_t)t->rows * (size_t)(t->cols / 2)
                            : (size_t)t->rows * (size_t)t->cols;
            size_t sb = raw ? (size_t)t->rows * (size_t)(t->cols / 32)
                            : (size_t)t->rows * (size_t)((t->cols + 127) / 128);
            if (raw) {
                memcpy(hw + wo, t->w, wb);
                memcpy(hs + so, t->scale, sb);
            } else if (widen_mxfp4_host(t, hw + wo, hs + so) != 0) return -1;
            t->gpu_id = -1;
            wo += wb; so += sb;
        }
    if (hipMalloc(&ctx->stream_dw, wtotal) != hipSuccess ||
        hipMalloc(&ctx->stream_ds, stotal) != hipSuccess ||
        hipMemcpy(ctx->stream_dw, hw, wtotal, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(ctx->stream_ds, hs, stotal, hipMemcpyHostToDevice) != hipSuccess) {
        if (ctx->stream_dw) hipFree(ctx->stream_dw);
        if (ctx->stream_ds) hipFree(ctx->stream_ds);
        ctx->stream_dw = ctx->stream_ds = NULL;
        return -1;
    }
    wo = so = 0;
    for (size_t wi = 0; wi < sizeof(ex) / sizeof(ex[0]); ++wi)
        for (int e = 0; e < layer->n_owned; ++e) {
            ds4f_tensor *t = (ds4f_tensor *)&ex[wi][e];
            size_t wb = raw ? (size_t)t->rows * (size_t)(t->cols / 2)
                            : (size_t)t->rows * (size_t)t->cols;
            size_t sb = raw ? (size_t)t->rows * (size_t)(t->cols / 32)
                            : (size_t)t->rows * (size_t)((t->cols + 127) / 128);
            int id = append_device_matrix(ctx, (uint8_t *)ctx->stream_dw + wo,
                (uint8_t *)ctx->stream_ds + so, hw + wo, hs + so,
                t->rows, t->cols, raw ? t->cols / 32 : (t->cols + 127) / 128,
                raw ? HIP_DS4F_MATRIX_MXFP4 : HIP_DS4F_MATRIX_FP8_ROWSCALE, 1);
            if (id < 0) return -1;
            t->gpu_id = id; wo += wb; so += sb;
        }
    ctx->stream_layer = layer;
    return 0;
}

int hip_ds4f_dense_stream_layer(void *opaque, const ds4f_layer *layer) {
    return stream_layer_impl(opaque, layer, 0);
}

int hip_ds4f_dense_stream_layer_raw(void *opaque, const ds4f_layer *layer) {
    return stream_layer_impl(opaque, layer, 1);
}

int hip_ds4f_dense_bind_fp8_ordered_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_FP8) return -1;
    int id = hip_ds4f_dense_bind_tensor(ctx, t);
    if (id >= 0) ctx->matrices[id].kind = HIP_DS4F_MATRIX_FP8_ORDERED;
    return id;
}

int hip_ds4f_dense_bind_bf16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_BF16 || !t->w ||
        !valid_dims(t->rows, t->cols) || ctx->pending || ctx->multi_pending)
        return -1;
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    size_t w_bytes = (size_t)t->rows * (size_t)t->cols * sizeof(uint16_t);
    void *dw = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess ||
        hipMemcpy(dw, t->w, w_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: BF16 matrix upload failed\n");
        if (dw) hipFree(dw);
        return -1;
    }
    if (ctx->n_matrices == ctx->cap_matrices) {
        int cap = ctx->cap_matrices ? ctx->cap_matrices * 2 : 8;
        if (cap < ctx->n_matrices || cap > INT_MAX / (int)sizeof(*ctx->matrices)) {
            hipFree(dw);
            return -1;
        }
        hip_ds4f_matrix *p = (hip_ds4f_matrix *)ds4f_mem_realloc(
            ctx->mem, ctx->matrices,
            (size_t)ctx->cap_matrices * sizeof(*ctx->matrices),
            (size_t)cap * sizeof(*ctx->matrices), 64);
        if (!p) {
            hipFree(dw);
            return -1;
        }
        ctx->matrices = p;
        ctx->cap_matrices = cap;
    }
    int id = ctx->n_matrices++;
    ctx->matrices[id] = (hip_ds4f_matrix){ dw, NULL, t->w, NULL,
                                           t->rows, t->cols, 0,
                                           HIP_DS4F_MATRIX_BF16 };
    ctx->current = id;
    t->gpu_id = id;
    return id;
}

static int fp8_scaled_to_f16(uint8_t w, uint8_t scale, uint16_t *out) {
    uint32_t bits = ds4f_fp8_e4m3fn_to_fp32_bits(w);
    float v;
    memcpy(&v, &bits, sizeof(v));
    v *= ggml_e8m0_to_fp32(scale);
    if (!isfinite(v) || fabsf(v) > 65504.0f) return -1;
    union { float f; uint32_t u; } fu = { v };
    uint32_t sign_bit = (fu.u >> 16) & 0x8000u;
    int fexp = (int)((fu.u >> 23) & 0xffu) - 127;
    uint32_t mantissa = fu.u & 0x7fffffu;
    uint16_t h;
    if (fexp < -24) h = (uint16_t)sign_bit;
    else if (fexp < -14) {
        mantissa |= 0x800000u;
        h = (uint16_t)(sign_bit | (mantissa >> (13 + (-14 - fexp))));
    } else {
        h = (uint16_t)(sign_bit | ((uint32_t)(fexp + 15) << 10) |
                       (mantissa >> 13));
    }
    /* The conversion is exact for the normal FP8 range. Check the edge cases
     * (FP8 subnormals combined with very small E8M0 scales) so this mode never
     * silently changes a weight and is unavailable rather than approximate. */
    uint32_t hb = h;
    uint32_t sign = (hb >> 15) & 1u, exp = (hb >> 10) & 31u, mant = hb & 1023u;
    float back;
    if (exp == 0) {
        if (mant == 0) back = 0.0f;
        else {
            int sh = 0; uint32_t m = mant;
            while ((m & 1024u) == 0) { m <<= 1; ++sh; }
            m &= 1023u;
            uint32_t fb = (sign << 31) | ((uint32_t)(113 - sh) << 23) | (m << 13);
            memcpy(&back, &fb, sizeof(back));
        }
    } else {
        uint32_t fb = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        memcpy(&back, &fb, sizeof(back));
    }
    if (back != v) return -1;
    *out = h;
    return 0;
}

int hip_ds4f_dense_bind_fp8_fp16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_FP8 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || ctx->pending || ctx->multi_pending)
        return -1;
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;

    size_t elems = (size_t)t->rows * (size_t)t->cols;
    size_t w_bytes = elems * sizeof(uint16_t);
    uint16_t *promoted = (uint16_t *)ds4f_mem_alloc(ctx->mem, w_bytes, 64, 0);
    if (!promoted) return -1;
    int scale_cols = (t->cols + 127) / 128;
    for (int r = 0; r < t->rows; ++r) {
        const uint8_t *wr = (const uint8_t *)t->w + (size_t)r * t->cols;
        const uint8_t *sr = t->scale + (size_t)(r >> 7) * scale_cols;
        uint16_t *dr = promoted + (size_t)r * t->cols;
        for (int c = 0; c < t->cols; ++c)
            if (fp8_scaled_to_f16(wr[c], sr[c >> 7], &dr[c]) != 0) {
                fprintf(stderr, "hip_ds4f_dense: FP8->FP16 exact conversion unavailable at row=%d col=%d w=0x%02x scale=0x%02x\n",
                        r, c, wr[c], sr[c >> 7]);
                return -1;
            }
    }

    void *dw = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess ||
        hipMemcpy(dw, promoted, w_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        if (dw) hipFree(dw);
        return -1;
    }
    if (ctx->n_matrices == ctx->cap_matrices) {
        int cap = ctx->cap_matrices ? ctx->cap_matrices * 2 : 8;
        if (cap < ctx->n_matrices || cap > INT_MAX / (int)sizeof(*ctx->matrices)) {
            hipFree(dw);
            return -1;
        }
        hip_ds4f_matrix *p = (hip_ds4f_matrix *)ds4f_mem_realloc(
            ctx->mem, ctx->matrices,
            (size_t)ctx->cap_matrices * sizeof(*ctx->matrices),
            (size_t)cap * sizeof(*ctx->matrices), 64);
        if (!p) {
            hipFree(dw);
            return -1;
        }
        ctx->matrices = p;
        ctx->cap_matrices = cap;
    }
    int id = ctx->n_matrices++;
    ctx->matrices[id] = (hip_ds4f_matrix){ dw, NULL, t->w, t->scale,
                                           t->rows, t->cols, scale_cols,
                                           HIP_DS4F_MATRIX_FP8_FP16 };
    ctx->current = id;
    t->gpu_id = id;
    return id;
}

int hip_ds4f_dense_bind_fp8_bf16_tensor(hip_ds4f_dense *ctx, ds4f_tensor *t) {
    if (!ctx || !t || t->type != DS4F_FP8 || !t->w || !t->scale ||
        !valid_dims(t->rows, t->cols) || ctx->pending || ctx->multi_pending)
        return -1;
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;

    size_t elems = (size_t)t->rows * (size_t)t->cols;
    size_t w_bytes = elems * sizeof(uint16_t);
    uint16_t *promoted = (uint16_t *)ds4f_mem_alloc(ctx->mem, w_bytes, 64, 0);
    if (!promoted) return -1;
    int scale_cols = (t->cols + 127) / 128;
    for (int r = 0; r < t->rows; ++r) {
        const uint8_t *wr = (const uint8_t *)t->w + (size_t)r * t->cols;
        const uint8_t *sr = t->scale + (size_t)(r >> 7) * scale_cols;
        uint16_t *dr = promoted + (size_t)r * t->cols;
        for (int c = 0; c < t->cols; ++c) {
            uint32_t bits = ds4f_fp8_e4m3fn_to_fp32_bits(wr[c]);
            float v;
            memcpy(&v, &bits, sizeof(v));
            v *= ggml_e8m0_to_fp32(sr[c >> 7]);
            memcpy(&bits, &v, sizeof(bits));
            dr[c] = (uint16_t)(bits >> 16);
        }
    }

    void *dw = NULL;
    if (hipMalloc(&dw, w_bytes) != hipSuccess ||
        hipMemcpy(dw, promoted, w_bytes, hipMemcpyHostToDevice) != hipSuccess) {
        if (dw) hipFree(dw);
        return -1;
    }
    if (ctx->n_matrices == ctx->cap_matrices) {
        int cap = ctx->cap_matrices ? ctx->cap_matrices * 2 : 8;
        if (cap < ctx->n_matrices || cap > INT_MAX / (int)sizeof(*ctx->matrices)) {
            hipFree(dw);
            return -1;
        }
        hip_ds4f_matrix *p = (hip_ds4f_matrix *)ds4f_mem_realloc(
            ctx->mem, ctx->matrices,
            (size_t)ctx->cap_matrices * sizeof(*ctx->matrices),
            (size_t)cap * sizeof(*ctx->matrices), 64);
        if (!p) {
            hipFree(dw);
            return -1;
        }
        ctx->matrices = p;
        ctx->cap_matrices = cap;
    }
    int id = ctx->n_matrices++;
    ctx->matrices[id] = (hip_ds4f_matrix){ dw, NULL, t->w, t->scale,
                                           t->rows, t->cols, scale_cols,
                                           HIP_DS4F_MATRIX_FP8_BF16 };
    ctx->current = id;
    t->gpu_id = id;
    return id;
}

int hip_ds4f_dense_load(hip_ds4f_dense *ctx,
                        const uint8_t *w, const uint8_t *s,
                        int rows, int cols) {
    if (!ctx || ctx->pending || ctx->multi_pending) {
        fprintf(stderr, "hip_ds4f_dense: invalid load or work in flight\n");
        return -1;
    }
    clear_matrices(ctx);
    return hip_ds4f_dense_add(ctx, w, s, rows, cols);
}

static int matrix_get(const hip_ds4f_dense *ctx, int id,
                      const hip_ds4f_matrix **out) {
    if (!ctx || id < 0 || id >= ctx->n_matrices || !out ||
        !ctx->matrices[id].dw) {
        fprintf(stderr, "hip_ds4f_dense: invalid matrix id %d\n", id);
        return -1;
    }
    *out = &ctx->matrices[id];
    return 0;
}

static int launch_matvec_async(hip_ds4f_dense *ctx, hipFunction_t fn, int id,
                               void *dw, void *ds, int rows, int cols,
                               int scale_cols, const float *x, int x_cols) {
    if (!ctx || !fn || !x || rows <= 0 || cols <= 0 || x_cols < cols) {
        fprintf(stderr, "hip_ds4f_dense: invalid matvec launch\n");
        return -1;
    }
    if (ctx->pending || ctx->multi_pending) {
        fprintf(stderr, "hip_ds4f_dense: previous matvec is still in flight\n");
        return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess ||
        ensure_vectors(ctx, rows, x_cols) != 0)
        return -1;
    if (hipMemcpyAsync(ctx->dx, x, (size_t)x_cols * sizeof(float),
                       hipMemcpyHostToDevice, ctx->stream) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: activation upload failed\n");
        hipStreamSynchronize(ctx->stream);
        return -1;
    }

    void *args[] = { &dw, &ds, &ctx->dx, &ctx->dy, &rows, &cols, &scale_cols };
    hipError_t err = hipModuleLaunchKernel(fn,
        (unsigned int)rows, 1, 1, (unsigned int)ctx->block_threads, 1, 1, 0,
        ctx->stream, args, NULL);
    if (err != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: kernel launch failed (%d)\n", (int)err);
        hipStreamSynchronize(ctx->stream);
        return -1;
    }
    if (hipEventRecord(ctx->done, ctx->stream) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: event record failed\n");
        hipStreamSynchronize(ctx->stream);
        return -1;
    }
    ctx->pending = 1;
    ctx->pending_id = id;
    ctx->pending_rows = rows;
    return 0;
}

int hip_ds4f_dense_matvec_id_async(hip_ds4f_dense *ctx, int id, const float *x) {
    const hip_ds4f_matrix *mat = NULL;
    if (!x || matrix_get(ctx, id, &mat) != 0) {
        fprintf(stderr, "hip_ds4f_dense: no matrix or invalid input\n");
        return -1;
    }
    hipFunction_t fn = matrix_is_fp16(mat->kind) ? ctx->f16_matvec :
                       matrix_is_bf16(mat->kind) ? ctx->bf16_matvec : ctx->matvec;
    return launch_matvec_async(ctx, fn, id, mat->dw, mat->ds,
                               mat->rows, mat->cols, mat->scale_cols,
                               x, mat->cols);
}

int hip_ds4f_dense_matvec_id(hip_ds4f_dense *ctx, int id,
                             const float *x, float *y) {
    if (hip_ds4f_dense_matvec_id_async(ctx, id, x) != 0) return -1;
    return hip_ds4f_dense_wait(ctx, y);
}

int hip_ds4f_dense_matvec_tensor(void *opaque, float *dst,
                                 const ds4f_tensor *t, const float *x) {
    if (!opaque || !t || (t->type != DS4F_FP8 && t->type != DS4F_BF16) ||
        t->gpu_id < 0)
        return -1;
    return hip_ds4f_dense_matvec_id((hip_ds4f_dense *)opaque, t->gpu_id, x, dst);
}

int hip_ds4f_dense_matvec_blockdiag(
    void *opaque, float *dst, const ds4f_tensor *t, const float *xbase,
    int gin, int glora, int goff) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    const hip_ds4f_matrix *mat = NULL;
    if (!ctx || !dst || !t || !xbase || t->type != DS4F_FP8 ||
        t->gpu_id < 0 || gin <= 0 || glora <= 0 || goff < 0 ||
        matrix_get(ctx, t->gpu_id, &mat) != 0 ||
        t->rows != mat->rows || t->cols != mat->cols) {
        fprintf(stderr, "hip_ds4f_dense: invalid block-diagonal matvec\n");
        return -1;
    }
    int groups_end = (goff + t->rows + glora - 1) / glora;
    if (groups_end <= 0 || groups_end > INT_MAX / gin) return -1;
    if (ctx->pending || ctx->multi_pending ||
        hipSetDevice(ctx->device_id) != hipSuccess ||
        ensure_vectors(ctx, t->rows, groups_end * gin) != 0)
        return -1;
    if (hipMemcpyAsync(ctx->dx, xbase,
                       (size_t)groups_end * (size_t)gin * sizeof(float),
                       hipMemcpyHostToDevice, ctx->stream) != hipSuccess)
        return -1;
    int rows = t->rows, cols = t->cols, scale_cols = mat->scale_cols;
    void *dw = mat->dw, *ds = mat->ds;
    void *args[] = { &dw, &ds, &ctx->dx, &ctx->dy,
                     &rows, &cols, &scale_cols, &gin, &glora, &goff };
    if (hipModuleLaunchKernel(ctx->blockdiag_matvec,
            (unsigned int)rows, 1, 1, (unsigned int)ctx->block_threads, 1, 1, 0,
            ctx->stream, args, NULL) != hipSuccess ||
        hipEventRecord(ctx->done, ctx->stream) != hipSuccess)
        return -1;
    ctx->pending = 1;
    ctx->pending_id = t->gpu_id;
    ctx->pending_rows = rows;
    return hip_ds4f_dense_wait(ctx, dst);
}

static int ensure_gemm_vectors(hip_ds4f_dense *ctx, size_t x_bytes, size_t y_bytes) {
    if (ctx->gemm_dx && ctx->gemm_dy && ctx->gemm_x_bytes >= x_bytes &&
        ctx->gemm_y_bytes >= y_bytes)
        return 0;
    void *dx = NULL, *dy = NULL;
    if (hipMalloc(&dx, x_bytes) != hipSuccess ||
        hipMalloc(&dy, y_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: batched GEMM allocation failed\n");
        if (dx) hipFree(dx);
        if (dy) hipFree(dy);
        return -1;
    }
    if (ctx->gemm_dx) hipFree(ctx->gemm_dx);
    if (ctx->gemm_dy) hipFree(ctx->gemm_dy);
    ctx->gemm_dx = dx; ctx->gemm_dy = dy;
    ctx->gemm_x_bytes = x_bytes; ctx->gemm_y_bytes = y_bytes;
    return 0;
}

static int ensure_gemm_x(hip_ds4f_dense *ctx, size_t x_bytes) {
    if (ctx->gemm_dx && ctx->gemm_x_bytes >= x_bytes) return 0;
    void *dx = NULL;
    if (hipMalloc(&dx, x_bytes) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: batched GEMM input allocation failed\n");
        return -1;
    }
    if (ctx->gemm_dx) hipFree(ctx->gemm_dx);
    ctx->gemm_dx = dx;
    ctx->gemm_x_bytes = x_bytes;
    return 0;
}

static int ensure_gemm_multi_outputs(hip_ds4f_dense *ctx, int n,
                                     const size_t *bytes) {
    for (int i = 0; i < n; ++i) {
        if (ctx->gemm_multi_dy[i] && ctx->gemm_multi_y_bytes[i] >= bytes[i])
            continue;
        void *dy = NULL;
        if (hipMalloc(&dy, bytes[i]) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: fused GEMM output allocation failed\n");
            if (dy) hipFree(dy);
            return -1;
        }
        if (ctx->gemm_multi_dy[i]) hipFree(ctx->gemm_multi_dy[i]);
        ctx->gemm_multi_dy[i] = dy;
        ctx->gemm_multi_y_bytes[i] = bytes[i];
    }
    return 0;
}

static int ensure_gemm_host_pack(hip_ds4f_dense *ctx, size_t x_bytes, size_t y_bytes) {
    if (x_bytes > ctx->gemm_x_pack_bytes) {
        float *p = (float *)ds4f_mem_alloc(ctx->mem, x_bytes, 64, 0);
        if (!p) return -1;
        ctx->gemm_x_pack = p; ctx->gemm_x_pack_bytes = x_bytes;
    }
    if (y_bytes > ctx->gemm_y_pack_bytes) {
        float *p = (float *)ds4f_mem_alloc(ctx->mem, y_bytes, 64, 0);
        if (!p) return -1;
        ctx->gemm_y_pack = p; ctx->gemm_y_pack_bytes = y_bytes;
    }
    return 0;
}

int hip_ds4f_dense_gemm_tensor(
    void *opaque, float *dst, const ds4f_tensor *t, const float *x,
    int M, int Ystride, int Xstride) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    const hip_ds4f_matrix *mat = NULL;
    int row0 = 0;
    if (!ctx || !dst || !x || !t || M < 1 || Ystride < t->rows ||
        Xstride < t->cols || t->gpu_id < 0 ||
        (t->type != DS4F_FP8 && t->type != DS4F_BF16 && t->type != DS4F_MXFP4) ||
        matrix_get(ctx, t->gpu_id, &mat) != 0 || mat->cols != t->cols ||
        (t->type == DS4F_FP8 && !matrix_is_fp8(mat->kind) &&
         !matrix_is_fp8_promoted(mat->kind)) ||
        (t->type == DS4F_BF16 && mat->kind != HIP_DS4F_MATRIX_BF16) ||
        (t->type == DS4F_MXFP4 && !matrix_is_mxfp4(mat->kind) &&
         !matrix_is_fp8(mat->kind))) {
        fprintf(stderr, "hip_ds4f_dense: invalid batched GEMM arguments\n");
        return -1;
    }
    if (mat->rows != t->rows) {
        if (matrix_is_fp8_promoted(mat->kind)) return -1;
        size_t wbpr = t->type == DS4F_FP8
            ? (size_t)t->cols : (size_t)t->cols * sizeof(uint16_t);
        if (!mat->hw || !t->w || (const uint8_t *)t->w < (const uint8_t *)mat->hw)
            return -1;
        size_t delta = (size_t)((const uint8_t *)t->w - (const uint8_t *)mat->hw);
        if (!wbpr || delta % wbpr != 0) return -1;
        row0 = (int)(delta / wbpr);
        if (row0 < 0 || row0 > mat->rows - t->rows) return -1;
        if (t->type == DS4F_FP8) {
            int sbc = mat->scale_cols;
            if (!mat->hs || row0 % 128 != 0 ||
                t->scale != (const uint8_t *)mat->hs +
                    (size_t)(row0 / 128) * (size_t)sbc)
                return -1;
        }
    }
    if (ctx->pending || ctx->multi_pending ||
        hipSetDevice(ctx->device_id) != hipSuccess)
        return -1;

    int N = t->rows, K = t->cols;
    size_t x_elems = (size_t)M * K, y_elems = (size_t)M * N;
    size_t x_bytes = x_elems * sizeof(float), y_bytes = y_elems * sizeof(float);
    const float *xh = x;
    float *yh = dst;
    if (Xstride != K || Ystride != N) {
        if (ensure_gemm_host_pack(ctx,
                                  Xstride != K ? x_bytes : 0,
                                  Ystride != N ? y_bytes : 0) != 0)
            return -1;
        if (Xstride != K) {
            for (int mm = 0; mm < M; mm++)
                memcpy(ctx->gemm_x_pack + (size_t)mm*K,
                       x + (size_t)mm*Xstride, (size_t)K*sizeof(float));
            xh = ctx->gemm_x_pack;
        }
        if (Ystride != N) yh = ctx->gemm_y_pack;
    }
    if (ensure_gemm_vectors(ctx, x_bytes, y_bytes) != 0) return -1;
    if (hipMemcpy(ctx->gemm_dx, xh, x_bytes, hipMemcpyHostToDevice) != hipSuccess)
        return -1;

    unsigned int gx = (unsigned int)((N + 63) / 64);
    unsigned int gy = (unsigned int)((M + 15) / 16);
    hipError_t err;
    if (matrix_is_mxfp4(mat->kind)) {
        void *dw = mat->dw, *ds = mat->ds;
        void *dx = ctx->gemm_dx, *dy = ctx->gemm_dy;
        int n_out = N, n_in = K, n_tok = M;
        void *args[] = { &dy, &dw, &ds, &dx, &n_out, &n_in, &n_tok };
        err = hipModuleLaunchKernel(ctx->gemm_mxfp4, gx, gy, 1, 16, 16, 1, 0,
                                    ctx->stream, args, NULL);
    } else if (matrix_is_fp8_rowscale(mat->kind)) {
        void *dw = (uint8_t *)mat->dw + (size_t)row0 * (size_t)K;
        void *ds = (uint8_t *)mat->ds + (size_t)row0 * (size_t)mat->scale_cols;
        void *dx = ctx->gemm_dx, *dy = ctx->gemm_dy, *lut = ctx->fp8_lut;
        int n_out = N, n_in = K, n_tok = M, scale_cols = mat->scale_cols;
        void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
        err = hipModuleLaunchKernel(ctx->gemm_fp8_rowscale, gx, gy, 1, 16, 16, 1, 0,
                                    ctx->stream, args, NULL);
    } else if (matrix_is_fp8_ordered(mat->kind)) {
        void *dw = (uint8_t *)mat->dw + (size_t)row0 * (size_t)K;
        void *ds = (uint8_t *)mat->ds + (size_t)(row0 / 128) * (size_t)mat->scale_cols;
        void *dx = ctx->gemm_dx, *dy = ctx->gemm_dy, *lut = ctx->fp8_lut;
        int n_out = N, n_in = K, n_tok = M, scale_cols = mat->scale_cols;
        void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
        err = hipModuleLaunchKernel(ctx->gemm_fp8_ordered, gx, gy, 1, 16, 16, 1, 0,
                                    ctx->stream, args, NULL);
    } else if (matrix_is_fp8(mat->kind)) {
        void *dw = (uint8_t *)mat->dw + (size_t)row0 * (size_t)K;
        void *ds = (uint8_t *)mat->ds + (size_t)(row0 / 128) * (size_t)mat->scale_cols;
        void *dx = ctx->gemm_dx, *dy = ctx->gemm_dy, *lut = ctx->fp8_lut;
        int n_out = N, n_in = K, n_tok = M, scale_cols = mat->scale_cols;
        void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
        err = hipModuleLaunchKernel(ctx->gemm_fp8, gx, gy, 1, 16, 16, 1, 0,
                                    ctx->stream, args, NULL);
    } else if (matrix_is_bf16(mat->kind)) {
        void *dw = (uint8_t *)mat->dw + (size_t)row0 * (size_t)K * sizeof(uint16_t);
        void *dx = ctx->gemm_dx, *dy = ctx->gemm_dy, *bias = NULL;
        int n_out = N, n_in = K, n_tok = M;
        void *args[] = { &dy, &dw, &dx, &bias, &n_out, &n_in, &n_tok };
        err = hipModuleLaunchKernel(ctx->gemm_bf16, gx, gy, 1, 16, 16, 1, 0,
                                    ctx->stream, args, NULL);
    } else {
        void *dw = (uint8_t *)mat->dw + (size_t)row0 * (size_t)K * sizeof(uint16_t);
        void *dx = ctx->gemm_dx, *dy = ctx->gemm_dy, *bias = NULL;
        int n_out = N, n_in = K, n_tok = M;
        void *args[] = { &dy, &dw, &dx, &bias, &n_out, &n_in, &n_tok };
        err = hipModuleLaunchKernel(ctx->gemm_f16, gx, gy, 1, 16, 16, 1, 0,
                                    ctx->stream, args, NULL);
    }
    if (err != hipSuccess || hipStreamSynchronize(ctx->stream) != hipSuccess)
        return -1;
    if (hipMemcpy(yh, ctx->gemm_dy, y_bytes, hipMemcpyDeviceToHost) != hipSuccess)
        return -1;
    if (Ystride != N) for (int mm = 0; mm < M; mm++)
        memcpy(dst + (size_t)mm*Ystride, yh + (size_t)mm*N, (size_t)N*sizeof(float));
    return 0;
}

int hip_ds4f_dense_gemm_tensors(
    void *opaque, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, const int *M, const int *Ystride,
    const int *Xstride, int n) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    const hip_ds4f_matrix *mat[HIP_DS4F_GEMM_MAX] = { NULL };
    size_t ybytes[HIP_DS4F_GEMM_MAX] = { 0 };
    if (!ctx || !dst || !t || !x || !M || !Ystride || !Xstride ||
        n < 1 || n > HIP_DS4F_GEMM_MAX || ctx->pending || ctx->multi_pending)
        return -1;
    int m0 = M[0], k0 = 0;
    if (m0 < 1 || !dst[0] || !x[0] || !t[0] || Xstride[0] < t[0]->cols ||
        t[0]->gpu_id < 0 || matrix_get(ctx, t[0]->gpu_id, &mat[0]) != 0 ||
        mat[0]->rows != t[0]->rows || mat[0]->cols != t[0]->cols ||
        (t[0]->type != DS4F_FP8 && t[0]->type != DS4F_BF16 && t[0]->type != DS4F_MXFP4) ||
        (t[0]->type == DS4F_FP8 && !matrix_is_fp8(mat[0]->kind) &&
         !matrix_is_fp8_promoted(mat[0]->kind)) ||
        (t[0]->type == DS4F_BF16 && mat[0]->kind != HIP_DS4F_MATRIX_BF16) ||
        (t[0]->type == DS4F_MXFP4 && !matrix_is_mxfp4(mat[0]->kind) &&
         !matrix_is_fp8_rowscale(mat[0]->kind)))
        return -1;
    k0 = t[0]->cols;
    for (int i = 0; i < n; ++i) {
        if (!dst[i] || !x[i] || !t[i] || M[i] < 1 ||
            t[i]->cols != k0 || Ystride[i] < t[i]->rows ||
            Xstride[i] < t[i]->cols || t[i]->gpu_id < 0 ||
            matrix_get(ctx, t[i]->gpu_id, &mat[i]) != 0 ||
            mat[i]->rows != t[i]->rows || mat[i]->cols != t[i]->cols ||
            (t[i]->type != DS4F_FP8 && t[i]->type != DS4F_BF16 && t[i]->type != DS4F_MXFP4) ||
            (t[i]->type == DS4F_FP8 && !matrix_is_fp8(mat[i]->kind) &&
             !matrix_is_fp8_promoted(mat[i]->kind)) ||
            (t[i]->type == DS4F_BF16 && mat[i]->kind != HIP_DS4F_MATRIX_BF16) ||
            (t[i]->type == DS4F_MXFP4 && !matrix_is_mxfp4(mat[i]->kind) &&
             !matrix_is_fp8_rowscale(mat[i]->kind)))
            return -1;
        ybytes[i] = (size_t)M[i] * (size_t)t[i]->rows * sizeof(float);
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    size_t xoff[HIP_DS4F_GEMM_MAX], xbytes = 0;
    for (int i = 0; i < n; ++i) {
        xoff[i] = xbytes;
        xbytes += (size_t)M[i] * (size_t)k0 * sizeof(float);
    }
    if (ensure_gemm_host_pack(ctx, xbytes, 0) != 0) return -1;
    for (int i = 0; i < n; ++i)
        for (int mm = 0; mm < M[i]; ++mm)
            memcpy((uint8_t *)ctx->gemm_x_pack + xoff[i] + (size_t)mm*k0*sizeof(float),
                   x[i] + (size_t)mm*Xstride[i], (size_t)k0*sizeof(float));
    if (ensure_gemm_x(ctx, xbytes) != 0 ||
        ensure_gemm_multi_outputs(ctx, n, ybytes) != 0)
        return -1;
    if (hipMemcpy(ctx->gemm_dx, ctx->gemm_x_pack, xbytes, hipMemcpyHostToDevice) != hipSuccess)
        return -1;
    for (int i = 0; i < n; ++i) {
        void *dw = mat[i]->dw, *ds = mat[i]->ds;
        void *dx = (uint8_t *)ctx->gemm_dx + xoff[i];
        void *dy = ctx->gemm_multi_dy[i];
        hipFunction_t fn;
        unsigned int gx = (unsigned int)((t[i]->rows + 63) / 64);
        unsigned int gy = (unsigned int)((M[i] + 15) / 16);
        hipError_t err;
        if (matrix_is_mxfp4(mat[i]->kind)) {
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            void *args[] = { &dy, &dw, &ds, &dx, &n_out, &n_in, &n_tok };
            fn = ctx->gemm_mxfp4;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8_rowscale(mat[i]->kind)) {
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            int scale_cols = mat[i]->scale_cols;
            void *lut = ctx->fp8_lut;
            void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
            fn = ctx->gemm_fp8_rowscale;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8_ordered(mat[i]->kind)) {
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            int scale_cols = mat[i]->scale_cols;
            void *lut = ctx->fp8_lut;
            void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
            fn = ctx->gemm_fp8_ordered;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_fp8(mat[i]->kind)) {
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            int scale_cols = mat[i]->scale_cols;
            void *lut = ctx->fp8_lut;
            void *args[] = { &dy, &dw, &ds, &dx, &lut, &n_out, &n_in, &n_tok, &scale_cols };
            fn = ctx->gemm_fp8;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else if (matrix_is_bf16(mat[i]->kind)) {
            void *bias = NULL;
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            void *args[] = { &dy, &dw, &dx, &bias, &n_out, &n_in, &n_tok };
            fn = ctx->gemm_bf16;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        } else {
            void *bias = NULL;
            int n_out = t[i]->rows, n_in = k0, n_tok = M[i];
            void *args[] = { &dy, &dw, &dx, &bias, &n_out, &n_in, &n_tok };
            fn = ctx->gemm_f16;
            err = hipModuleLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0,
                                        ctx->stream, args, NULL);
        }
        if (err != hipSuccess) return -1;
    }
    if (hipStreamSynchronize(ctx->stream) != hipSuccess) return -1;
    size_t max_ybytes = 0;
    for (int i = 0; i < n; ++i) if (ybytes[i] > max_ybytes) max_ybytes = ybytes[i];
    if (max_ybytes > 0 && ensure_gemm_host_pack(ctx, 0, max_ybytes) != 0) return -1;
    for (int i = 0; i < n; ++i) {
        float *yh = dst[i];
        if (Ystride[i] == t[i]->rows) {
            if (hipMemcpy(yh, ctx->gemm_multi_dy[i], ybytes[i], hipMemcpyDeviceToHost) != hipSuccess)
                return -1;
        } else {
            if (hipMemcpy(ctx->gemm_y_pack, ctx->gemm_multi_dy[i], ybytes[i],
                          hipMemcpyDeviceToHost) != hipSuccess)
                return -1;
            for (int mm = 0; mm < M[i]; ++mm)
                memcpy(yh + (size_t)mm*Ystride[i],
                       ctx->gemm_y_pack + (size_t)mm*t[i]->rows,
                       (size_t)t[i]->rows*sizeof(float));
        }
    }
    return 0;
}

int hip_ds4f_dense_matvec_tensors_async(
    void *opaque, float *const *dst, const ds4f_tensor *const *t,
    const float *const *x, int n) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !dst || !t || !x || n < 1 || n > HIP_DS4F_ASYNC_MAX ||
        ctx->pending || ctx->multi_pending) {
        fprintf(stderr, "hip_ds4f_dense: invalid async tensor batch\n");
        return -1;
    }
    const hip_ds4f_matrix *mat[HIP_DS4F_ASYNC_MAX] = { NULL, NULL };
    for (int i = 0; i < n; ++i) {
        if (!dst[i] || !x[i] || !t[i] ||
            (t[i]->type != DS4F_FP8 && t[i]->type != DS4F_BF16) ||
            t[i]->gpu_id < 0 || matrix_get(ctx, t[i]->gpu_id, &mat[i]) != 0 ||
            (t[i]->type == DS4F_FP8 && !matrix_is_fp8(mat[i]->kind) &&
             !matrix_is_fp8_promoted(mat[i]->kind)) ||
            (t[i]->type == DS4F_BF16 && mat[i]->kind != HIP_DS4F_MATRIX_BF16))
            return -1;
    }
    if (hipSetDevice(ctx->device_id) != hipSuccess) return -1;
    for (int i = 0; i < n; ++i) {
        if (ensure_multi_vectors(ctx, i, mat[i]->rows, mat[i]->cols) != 0)
            goto fail;
        if (hipMemcpyAsync(ctx->multi_dx[i], x[i],
                           (size_t)mat[i]->cols * sizeof(float),
                           hipMemcpyHostToDevice, ctx->multi_stream[i]) != hipSuccess)
            goto fail;
        void *args[] = { (void *)&mat[i]->dw, (void *)&mat[i]->ds,
                         &ctx->multi_dx[i], &ctx->multi_dy[i],
                         (void *)&mat[i]->rows, (void *)&mat[i]->cols,
                         (void *)&mat[i]->scale_cols };
        hipFunction_t fn = matrix_is_fp16(mat[i]->kind) ? ctx->f16_matvec :
                           matrix_is_bf16(mat[i]->kind) ? ctx->bf16_matvec : ctx->matvec;
        if (hipModuleLaunchKernel(fn,
                (unsigned int)mat[i]->rows, 1, 1, (unsigned int)ctx->block_threads, 1, 1, 0,
                ctx->multi_stream[i], args, NULL) != hipSuccess)
            goto fail;
        if (hipEventRecord(ctx->multi_done[i], ctx->multi_stream[i]) != hipSuccess)
            goto fail;
    }
    for (int i = 0; i < n; ++i) {
        ctx->multi_y[i] = dst[i];
        ctx->multi_out_bytes[i] = (size_t)mat[i]->rows * sizeof(float);
        ctx->multi_n = n;
    }
    ctx->multi_pending = 1;
    return 0;

fail:
    fprintf(stderr, "hip_ds4f_dense: async tensor launch failed\n");
    for (int i = 0; i < n; ++i)
        hipStreamSynchronize(ctx->multi_stream[i]);
    return -1;
}

int hip_ds4f_dense_wait_tensors(void *opaque) {
    hip_ds4f_dense *ctx = (hip_ds4f_dense *)opaque;
    if (!ctx || !ctx->multi_pending || ctx->multi_n < 1 ||
        ctx->multi_n > HIP_DS4F_ASYNC_MAX)
        return -1;
    int n = ctx->multi_n;
    for (int i = 0; i < n; ++i) {
        if (hipEventSynchronize(ctx->multi_done[i]) != hipSuccess ||
            hipMemcpy(ctx->multi_y[i], ctx->multi_dy[i], ctx->multi_out_bytes[i],
                      hipMemcpyDeviceToHost) != hipSuccess) {
            fprintf(stderr, "hip_ds4f_dense: async tensor wait/copy failed\n");
            return -1;
        }
    }
    ctx->multi_pending = 0;
    ctx->multi_n = 0;
    return 0;
}

int hip_ds4f_dense_matvec_loaded_async(hip_ds4f_dense *ctx, const float *x) {
    if (!ctx) return -1;
    return hip_ds4f_dense_matvec_id_async(ctx, ctx->current, x);
}

int hip_ds4f_dense_wait(hip_ds4f_dense *ctx, float *y) {
    if (!ctx || !ctx->pending || !y) {
        fprintf(stderr, "hip_ds4f_dense: no pending matvec or invalid output\n");
        return -1;
    }
    if (ctx->pending_rows <= 0) return -1;
    if (hipEventSynchronize(ctx->done) != hipSuccess ||
        hipMemcpy(y, ctx->dy, (size_t)ctx->pending_rows * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        fprintf(stderr, "hip_ds4f_dense: kernel synchronization/copy failed\n");
        return -1;
    }
    ctx->pending = 0;
    ctx->pending_rows = 0;
    return 0;
}

int hip_ds4f_dense_matvec_loaded(hip_ds4f_dense *ctx,
                                 const float *x, float *y) {
    if (hip_ds4f_dense_matvec_loaded_async(ctx, x) != 0) return -1;
    return hip_ds4f_dense_wait(ctx, y);
}

int hip_ds4f_dense_matvec(hip_ds4f_dense *ctx,
                          const uint8_t *w, const uint8_t *s,
                          const float *x, float *y,
                          int rows, int cols) {
    if (hip_ds4f_dense_load(ctx, w, s, rows, cols) < 0) return -1;
    return hip_ds4f_dense_matvec_loaded(ctx, x, y);
}
