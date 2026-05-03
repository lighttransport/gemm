/*
 * cublasew.c - Minimal dynamic cuBLAS loader without CUDA SDK headers
 */

#include "cublasew.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
typedef HMODULE cublas_lib_t;
#  define cublas_open(name) LoadLibraryA(name)
#  define cublas_sym(lib, name) GetProcAddress(lib, name)
#  define cublas_close(lib) FreeLibrary(lib)
#else
#  include <dlfcn.h>
typedef void *cublas_lib_t;
#  define cublas_open(name) dlopen(name, RTLD_NOW)
#  define cublas_sym(lib, name) dlsym(lib, name)
#  define cublas_close(lib) dlclose(lib)
#endif

typedef void *cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;
typedef int cudaDataType_t;
typedef int cublasComputeType_t;

enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUDA_R_32F = 0,
    CUDA_R_16F = 2,
    CUDA_R_16BF = 14,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_GEMM_DEFAULT = -1
};

/* cuBLAS-LT enums (from cublasLt.h, version >= 11.4) */
enum {
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6,

    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE = 0,
    CUBLASLT_MATMUL_DESC_SCALE_TYPE = 1,
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER = 8,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,

    CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,

    CUBLASLT_EPILOGUE_DEFAULT = 1,
    CUBLASLT_EPILOGUE_BIAS = 4
};

typedef void *cublasLtHandle_t;
typedef void *cublasLtMatmulDesc_t;
typedef void *cublasLtMatrixLayout_t;
typedef void *cublasLtMatmulPreference_t;

/* cublasLtMatmulHeuristicResult_t is a struct; we never inspect fields,
 * so use a 256-byte opaque buffer (actual size is ~80B). */
typedef struct { unsigned char opaque[256]; } cublasLtMatmulHeuristicResult_t;

typedef cublasStatus_t (*tcublasCreate_v2)(cublasHandle_t *);
typedef cublasStatus_t (*tcublasDestroy_v2)(cublasHandle_t);
typedef cublasStatus_t (*tcublasSetStream_v2)(cublasHandle_t, CUstream);
typedef cublasStatus_t (*tcublasSgemm_v2)(cublasHandle_t,
                                          cublasOperation_t,
                                          cublasOperation_t,
                                          int, int, int,
                                          const float *,
                                          const float *, int,
                                          const float *, int,
                                          const float *,
                                          float *, int);
typedef cublasStatus_t (*tcublasGemmEx)(cublasHandle_t,
                                        cublasOperation_t,
                                        cublasOperation_t,
                                        int, int, int,
                                        const void *,
                                        const void *, cudaDataType_t, int,
                                        const void *, cudaDataType_t, int,
                                        const void *,
                                        void *, cudaDataType_t, int,
                                        cublasComputeType_t, int);

static cublas_lib_t g_cublas_lib;
static tcublasCreate_v2 p_cublasCreate_v2;
static tcublasDestroy_v2 p_cublasDestroy_v2;
static tcublasSetStream_v2 p_cublasSetStream_v2;
static tcublasSgemm_v2 p_cublasSgemm_v2;
static tcublasGemmEx p_cublasGemmEx;
static int g_cublas_init_done;
static int g_cublas_available;

/* cuBLAS-LT function pointers */
typedef cublasStatus_t (*tcublasLtCreate)(cublasLtHandle_t *);
typedef cublasStatus_t (*tcublasLtDestroy)(cublasLtHandle_t);
typedef cublasStatus_t (*tcublasLtMatmulDescCreate)(cublasLtMatmulDesc_t *,
                                                    cublasComputeType_t,
                                                    cudaDataType_t);
typedef cublasStatus_t (*tcublasLtMatmulDescDestroy)(cublasLtMatmulDesc_t);
typedef cublasStatus_t (*tcublasLtMatmulDescSetAttribute)(cublasLtMatmulDesc_t,
                                                          int, const void *,
                                                          size_t);
typedef cublasStatus_t (*tcublasLtMatrixLayoutCreate)(cublasLtMatrixLayout_t *,
                                                     cudaDataType_t,
                                                     uint64_t, uint64_t,
                                                     int64_t);
typedef cublasStatus_t (*tcublasLtMatrixLayoutDestroy)(cublasLtMatrixLayout_t);
typedef cublasStatus_t (*tcublasLtMatrixLayoutSetAttribute)(cublasLtMatrixLayout_t,
                                                           int, const void *,
                                                           size_t);
typedef cublasStatus_t (*tcublasLtMatmulPreferenceCreate)(cublasLtMatmulPreference_t *);
typedef cublasStatus_t (*tcublasLtMatmulPreferenceDestroy)(cublasLtMatmulPreference_t);
typedef cublasStatus_t (*tcublasLtMatmulPreferenceSetAttribute)(cublasLtMatmulPreference_t,
                                                                int, const void *,
                                                                size_t);
typedef cublasStatus_t (*tcublasLtMatmulAlgoGetHeuristic)(cublasLtHandle_t,
                                                          cublasLtMatmulDesc_t,
                                                          cublasLtMatrixLayout_t,
                                                          cublasLtMatrixLayout_t,
                                                          cublasLtMatrixLayout_t,
                                                          cublasLtMatrixLayout_t,
                                                          cublasLtMatmulPreference_t,
                                                          int,
                                                          cublasLtMatmulHeuristicResult_t *,
                                                          int *);
typedef cublasStatus_t (*tcublasLtMatmul)(cublasLtHandle_t,
                                          cublasLtMatmulDesc_t,
                                          const void *,
                                          const void *, cublasLtMatrixLayout_t,
                                          const void *, cublasLtMatrixLayout_t,
                                          const void *,
                                          const void *, cublasLtMatrixLayout_t,
                                          void *, cublasLtMatrixLayout_t,
                                          const void *,
                                          void *, size_t,
                                          CUstream);

static cublas_lib_t g_cublaslt_lib;
static tcublasLtCreate p_cublasLtCreate;
static tcublasLtDestroy p_cublasLtDestroy;
static tcublasLtMatmulDescCreate p_cublasLtMatmulDescCreate;
static tcublasLtMatmulDescDestroy p_cublasLtMatmulDescDestroy;
static tcublasLtMatmulDescSetAttribute p_cublasLtMatmulDescSetAttribute;
static tcublasLtMatrixLayoutCreate p_cublasLtMatrixLayoutCreate;
static tcublasLtMatrixLayoutDestroy p_cublasLtMatrixLayoutDestroy;
static tcublasLtMatrixLayoutSetAttribute p_cublasLtMatrixLayoutSetAttribute;
static tcublasLtMatmulPreferenceCreate p_cublasLtMatmulPreferenceCreate;
static tcublasLtMatmulPreferenceDestroy p_cublasLtMatmulPreferenceDestroy;
static tcublasLtMatmulPreferenceSetAttribute p_cublasLtMatmulPreferenceSetAttribute;
static tcublasLtMatmulAlgoGetHeuristic p_cublasLtMatmulAlgoGetHeuristic;
static tcublasLtMatmul p_cublasLtMatmul;
static int g_cublaslt_init_done;
static int g_cublaslt_available;

#define CUBLASLT_WORKSPACE_BYTES (32u * 1024u * 1024u)
#define CUBLASLT_CACHE_MAX 96

typedef struct {
    int n_tok, n_out, n_in;
    int y_dtype;
    int has_w_scale, has_x_scale;
    int valid;
    /* cached per-shape objects */
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t a_layout, b_layout, d_layout;
    cublasLtMatmulHeuristicResult_t heur;
} lt_cache_entry;

struct cublasew_context {
    cublasHandle_t handle;
    cublasLtHandle_t lt_handle;
    CUstream stream;
    CUdeviceptr d_workspace;
    size_t workspace_bytes;
    /* Per-shape cache: amortizes descriptor creation + heuristic across calls. */
    lt_cache_entry cache[CUBLASLT_CACHE_MAX];
    int cache_n;
    cublasLtMatmulPreference_t pref;  /* shared across all cache entries */
};

static int cublasew_load_symbol(void **dst, const char *name) {
    *dst = cublas_sym(g_cublas_lib, name);
    return *dst ? 0 : -1;
}

int cublasewInit(void) {
    const char *names[] = {
        "libcublas.so.13",
        "libcublas.so.12",
        "libcublas.so.11",
        "libcublas.so",
        NULL
    };
    int i;

    if (g_cublas_init_done) return g_cublas_available ? 0 : -1;
    g_cublas_init_done = 1;

    for (i = 0; names[i]; i++) {
        g_cublas_lib = cublas_open(names[i]);
        if (g_cublas_lib) break;
    }
    if (!g_cublas_lib) return -1;

    if (cublasew_load_symbol((void **)&p_cublasCreate_v2, "cublasCreate_v2") != 0 ||
        cublasew_load_symbol((void **)&p_cublasDestroy_v2, "cublasDestroy_v2") != 0 ||
        cublasew_load_symbol((void **)&p_cublasSetStream_v2, "cublasSetStream_v2") != 0 ||
        cublasew_load_symbol((void **)&p_cublasSgemm_v2, "cublasSgemm_v2") != 0 ||
        cublasew_load_symbol((void **)&p_cublasGemmEx, "cublasGemmEx") != 0) {
        cublas_close(g_cublas_lib);
        g_cublas_lib = NULL;
        return -1;
    }

    g_cublas_available = 1;
    return 0;
}

static int cublaslt_load_symbol(void **dst, const char *name) {
    *dst = cublas_sym(g_cublaslt_lib, name);
    return *dst ? 0 : -1;
}

static int cublasewLtInit(void) {
    const char *names[] = {
        "libcublasLt.so.13",
        "libcublasLt.so.12",
        "libcublasLt.so.11",
        "libcublasLt.so",
        NULL
    };
    int i;
    if (g_cublaslt_init_done) return g_cublaslt_available ? 0 : -1;
    g_cublaslt_init_done = 1;
    for (i = 0; names[i]; i++) {
        g_cublaslt_lib = cublas_open(names[i]);
        if (g_cublaslt_lib) break;
    }
    if (!g_cublaslt_lib) return -1;

    if (cublaslt_load_symbol((void **)&p_cublasLtCreate, "cublasLtCreate") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtDestroy, "cublasLtDestroy") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatmulDescCreate, "cublasLtMatmulDescCreate") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatmulDescDestroy, "cublasLtMatmulDescDestroy") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatmulDescSetAttribute, "cublasLtMatmulDescSetAttribute") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatrixLayoutCreate, "cublasLtMatrixLayoutCreate") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatrixLayoutDestroy, "cublasLtMatrixLayoutDestroy") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatrixLayoutSetAttribute, "cublasLtMatrixLayoutSetAttribute") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatmulPreferenceCreate, "cublasLtMatmulPreferenceCreate") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatmulPreferenceDestroy, "cublasLtMatmulPreferenceDestroy") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatmulPreferenceSetAttribute, "cublasLtMatmulPreferenceSetAttribute") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatmulAlgoGetHeuristic, "cublasLtMatmulAlgoGetHeuristic") != 0 ||
        cublaslt_load_symbol((void **)&p_cublasLtMatmul, "cublasLtMatmul") != 0) {
        cublas_close(g_cublaslt_lib);
        g_cublaslt_lib = NULL;
        return -1;
    }
    g_cublaslt_available = 1;
    return 0;
}

int cublasewCreate(cublasew_context **out, CUstream stream) {
    cublasew_context *ctx;
    if (!out) return -1;
    *out = NULL;
    if (cublasewInit() != 0) return -1;

    ctx = (cublasew_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return -1;
    ctx->stream = stream;
    if (p_cublasCreate_v2(&ctx->handle) != CUBLAS_STATUS_SUCCESS) {
        free(ctx);
        return -1;
    }
    if (stream && p_cublasSetStream_v2(ctx->handle, stream) != CUBLAS_STATUS_SUCCESS) {
        p_cublasDestroy_v2(ctx->handle);
        free(ctx);
        return -1;
    }

    /* Optional: bring up cuBLAS-LT (best-effort) */
    if (cublasewLtInit() == 0) {
        if (p_cublasLtCreate(&ctx->lt_handle) == CUBLAS_STATUS_SUCCESS) {
            ctx->workspace_bytes = CUBLASLT_WORKSPACE_BYTES;
            if (cuMemAlloc(&ctx->d_workspace, ctx->workspace_bytes) != CUDA_SUCCESS) {
                ctx->d_workspace = 0;
                ctx->workspace_bytes = 0;
            }
            /* Build the shared preference once. */
            if (p_cublasLtMatmulPreferenceCreate(&ctx->pref) == CUBLAS_STATUS_SUCCESS) {
                size_t ws = ctx->workspace_bytes;
                p_cublasLtMatmulPreferenceSetAttribute(ctx->pref,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
            } else {
                ctx->pref = NULL;
            }
        } else {
            ctx->lt_handle = NULL;
        }
    }

    *out = ctx;
    return 0;
}

void cublasewDestroy(cublasew_context *ctx) {
    if (!ctx) return;
    /* Tear down per-shape cache */
    if (g_cublaslt_available) {
        int i;
        for (i = 0; i < ctx->cache_n; i++) {
            lt_cache_entry *e = &ctx->cache[i];
            if (!e->valid) continue;
            if (e->d_layout) p_cublasLtMatrixLayoutDestroy(e->d_layout);
            if (e->b_layout) p_cublasLtMatrixLayoutDestroy(e->b_layout);
            if (e->a_layout) p_cublasLtMatrixLayoutDestroy(e->a_layout);
            if (e->desc) p_cublasLtMatmulDescDestroy(e->desc);
        }
        if (ctx->pref) p_cublasLtMatmulPreferenceDestroy(ctx->pref);
    }
    if (ctx->d_workspace) cuMemFree(ctx->d_workspace);
    if (ctx->lt_handle && p_cublasLtDestroy) p_cublasLtDestroy(ctx->lt_handle);
    if (ctx->handle) p_cublasDestroy_v2(ctx->handle);
    free(ctx);
}

int cublasew_lt_available(cublasew_context *ctx) {
    return (ctx && ctx->lt_handle && g_cublaslt_available) ? 0 : -1;
}

int cublasewSetStream(cublasew_context *ctx, CUstream stream) {
    if (!ctx || !ctx->handle) return -1;
    ctx->stream = stream;
    return p_cublasSetStream_v2(ctx->handle, stream) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cublasew_gemm_f32_rowmajor_nt(cublasew_context *ctx,
                                  CUdeviceptr d_Y,
                                  CUdeviceptr d_W_f32,
                                  CUdeviceptr d_X_f32,
                                  int n_tok,
                                  int n_out,
                                  int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!ctx || !ctx->handle) return -1;
    return p_cublasSgemm_v2(ctx->handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            n_out, n_tok, n_in,
                            &alpha,
                            (const float *)(uintptr_t)d_W_f32, n_in,
                            (const float *)(uintptr_t)d_X_f32, n_in,
                            &beta,
                            (float *)(uintptr_t)d_Y, n_out) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cublasew_gemm_f16_f32_rowmajor_nt(cublasew_context *ctx,
                                      CUdeviceptr d_Y,
                                      CUdeviceptr d_W_f16,
                                      CUdeviceptr d_X_f32,
                                      int n_tok,
                                      int n_out,
                                      int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!ctx || !ctx->handle) return -1;
    /* Try mixed F16×F32 first (works on pre-Blackwell) */
    cublasStatus_t st = p_cublasGemmEx(ctx->handle,
                          CUBLAS_OP_T, CUBLAS_OP_N,
                          n_out, n_tok, n_in,
                          &alpha,
                          (const void *)(uintptr_t)d_W_f16, CUDA_R_16F, n_in,
                          (const void *)(uintptr_t)d_X_f32, CUDA_R_32F, n_in,
                          &beta,
                          (void *)(uintptr_t)d_Y, CUDA_R_32F, n_out,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 0;

    /* Blackwell fallback: caller must provide F16 input buffer via d_X_f16 */
    return -1;
}

int cublasew_gemm_f16_f16_f32_rowmajor_nt(cublasew_context *ctx,
                                           CUdeviceptr d_Y,
                                           CUdeviceptr d_W_f16,
                                           CUdeviceptr d_X_f16,
                                           int n_tok,
                                           int n_out,
                                           int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!ctx || !ctx->handle) return -1;
    return p_cublasGemmEx(ctx->handle,
                          CUBLAS_OP_T, CUBLAS_OP_N,
                          n_out, n_tok, n_in,
                          &alpha,
                          (const void *)(uintptr_t)d_W_f16, CUDA_R_16F, n_in,
                          (const void *)(uintptr_t)d_X_f16, CUDA_R_16F, n_in,
                          &beta,
                          (void *)(uintptr_t)d_Y, CUDA_R_32F, n_out,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cublasew_gemm_bf16_bf16_f32_rowmajor_nt(cublasew_context *ctx,
                                             CUdeviceptr d_Y,
                                             CUdeviceptr d_W_bf16,
                                             CUdeviceptr d_X_bf16,
                                             int n_tok,
                                             int n_out,
                                             int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!ctx || !ctx->handle) return -1;
    return p_cublasGemmEx(ctx->handle,
                          CUBLAS_OP_T, CUBLAS_OP_N,
                          n_out, n_tok, n_in,
                          &alpha,
                          (const void *)(uintptr_t)d_W_bf16, CUDA_R_16BF, n_in,
                          (const void *)(uintptr_t)d_X_bf16, CUDA_R_16BF, n_in,
                          &beta,
                          (void *)(uintptr_t)d_Y, CUDA_R_32F, n_out,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

/* ----- cuBLAS-LT FP8 e4m3 matmul ---------------------------------- */

/* Internal helper: build descriptors and run.
 *   D[N,M] col-major = A[K,N]^T * B[K,M] (col-major), in cuBLAS-LT terms:
 *   - A is K rows × N cols, ld=K, op=T   (this is row-major W [N,K])
 *   - B is K rows × M cols, ld=K, op=N   (this is row-major X [M,K])
 *   - D is N rows × M cols, ld=N         (this is row-major Y [M,N])
 *
 *   Hopper/Ada/Blackwell FP8 layout requirement: A op=T, B op=N, D col-major.
 */
/* Lookup or build a cache entry for (n_tok, n_out, n_in, y_dtype). The
 * scale/bias pointer attributes are baked into the desc — we assume the
 * caller passes the same persistent pointers for a given shape (true for
 * qimg's per-runner d_x_scale_f32 / d_w_scale_one). */
static lt_cache_entry *lt_cache_lookup_or_build(cublasew_context *ctx,
                                                int n_tok, int n_out, int n_in,
                                                cudaDataType_t y_dtype,
                                                CUdeviceptr d_w_scale_f32,
                                                CUdeviceptr d_x_scale_f32) {
    int i;
    int has_w_scale = d_w_scale_f32 ? 1 : 0;
    int has_x_scale = d_x_scale_f32 ? 1 : 0;
    for (i = 0; i < ctx->cache_n; i++) {
        lt_cache_entry *e = &ctx->cache[i];
        if (e->valid && e->n_tok == n_tok && e->n_out == n_out &&
            e->n_in == n_in && e->y_dtype == (int)y_dtype &&
            e->has_w_scale == has_w_scale && e->has_x_scale == has_x_scale)
            return e;
    }
    if (ctx->cache_n >= CUBLASLT_CACHE_MAX) return NULL;
    lt_cache_entry *e = &ctx->cache[ctx->cache_n];
    memset(e, 0, sizeof(*e));

    int op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    cudaDataType_t in_dt = CUDA_R_8F_E4M3;
    void *Wsp = (void *)(uintptr_t)d_w_scale_f32;
    void *Xsp = (void *)(uintptr_t)d_x_scale_f32;
    cublasStatus_t st;

    st = p_cublasLtMatmulDescCreate(&e->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    if (p_cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (p_cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (has_w_scale &&
        p_cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &Wsp, sizeof(Wsp)) != CUBLAS_STATUS_SUCCESS)
        goto fail;
    if (has_x_scale &&
        p_cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &Xsp, sizeof(Xsp)) != CUBLAS_STATUS_SUCCESS)
        goto fail;

    if (p_cublasLtMatrixLayoutCreate(&e->a_layout, in_dt, (uint64_t)n_in, (uint64_t)n_out, (int64_t)n_in) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (p_cublasLtMatrixLayoutCreate(&e->b_layout, in_dt, (uint64_t)n_in, (uint64_t)n_tok, (int64_t)n_in) != CUBLAS_STATUS_SUCCESS) goto fail;
    if (p_cublasLtMatrixLayoutCreate(&e->d_layout, y_dtype, (uint64_t)n_out, (uint64_t)n_tok, (int64_t)n_out) != CUBLAS_STATUS_SUCCESS) goto fail;

    int returned = 0;
    memset(&e->heur, 0, sizeof(e->heur));
    if (p_cublasLtMatmulAlgoGetHeuristic(ctx->lt_handle, e->desc,
                                         e->a_layout, e->b_layout,
                                         e->d_layout, e->d_layout,
                                         ctx->pref, 1, &e->heur, &returned) != CUBLAS_STATUS_SUCCESS)
        goto fail;
    if (returned <= 0) goto fail;

    e->n_tok = n_tok; e->n_out = n_out; e->n_in = n_in;
    e->y_dtype = (int)y_dtype;
    e->has_w_scale = has_w_scale;
    e->has_x_scale = has_x_scale;
    e->valid = 1;
    ctx->cache_n++;
    return e;

fail:
    if (e->d_layout) p_cublasLtMatrixLayoutDestroy(e->d_layout);
    if (e->b_layout) p_cublasLtMatrixLayoutDestroy(e->b_layout);
    if (e->a_layout) p_cublasLtMatrixLayoutDestroy(e->a_layout);
    if (e->desc) p_cublasLtMatmulDescDestroy(e->desc);
    memset(e, 0, sizeof(*e));
    return NULL;
}

static int cublasew_lt_fp8_run(cublasew_context *ctx,
                               CUdeviceptr d_Y,
                               cudaDataType_t y_dtype,
                               CUdeviceptr d_W_e4m3,
                               CUdeviceptr d_X_e4m3,
                               CUdeviceptr d_w_scale_f32,
                               CUdeviceptr d_x_scale_f32,
                               CUdeviceptr d_bias,
                               cudaDataType_t bias_dtype,
                               int n_tok, int n_out, int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    void *Wp = (void *)(uintptr_t)d_W_e4m3;
    void *Xp = (void *)(uintptr_t)d_X_e4m3;
    void *Yp = (void *)(uintptr_t)d_Y;
    cublasStatus_t st;
    (void)d_bias; (void)bias_dtype;  /* bias path is not cached; caller does post-add */

    if (!ctx || !ctx->lt_handle) return -1;
    if (!g_cublaslt_available) return -1;

    lt_cache_entry *e = lt_cache_lookup_or_build(ctx, n_tok, n_out, n_in,
                                                  y_dtype, d_w_scale_f32, d_x_scale_f32);
    if (!e) return -1;

    st = p_cublasLtMatmul(ctx->lt_handle, e->desc,
                          &alpha,
                          Wp, e->a_layout,
                          Xp, e->b_layout,
                          &beta,
                          Yp, e->d_layout,
                          Yp, e->d_layout,
                          (const void *)&e->heur.opaque[0],
                          (void *)(uintptr_t)ctx->d_workspace, ctx->workspace_bytes,
                          ctx->stream);
    if (st != CUBLAS_STATUS_SUCCESS) return -1;
    return 0;
}

int cublasew_gemm_fp8_e4m3_bf16out_rowmajor_nt(cublasew_context *ctx,
                                               CUdeviceptr d_Y_bf16,
                                               CUdeviceptr d_W_e4m3,
                                               CUdeviceptr d_X_e4m3,
                                               CUdeviceptr d_w_scale_f32,
                                               CUdeviceptr d_x_scale_f32,
                                               CUdeviceptr d_bias_bf16,
                                               int n_tok,
                                               int n_out,
                                               int n_in) {
    return cublasew_lt_fp8_run(ctx, d_Y_bf16, CUDA_R_16BF,
                               d_W_e4m3, d_X_e4m3,
                               d_w_scale_f32, d_x_scale_f32,
                               d_bias_bf16, CUDA_R_16BF,
                               n_tok, n_out, n_in);
}

int cublasew_gemm_fp8_e4m3_f32out_rowmajor_nt(cublasew_context *ctx,
                                              CUdeviceptr d_Y_f32,
                                              CUdeviceptr d_W_e4m3,
                                              CUdeviceptr d_X_e4m3,
                                              CUdeviceptr d_w_scale_f32,
                                              CUdeviceptr d_x_scale_f32,
                                              int n_tok,
                                              int n_out,
                                              int n_in) {
    /* Note: cuBLAS-LT FP8 typically requires BF16/FP16 D for FP8 inputs.
     * Use bf16out + a host-side BF16->F32 expand if F32 is needed.
     * We still try direct F32 here in case the version supports it. */
    return cublasew_lt_fp8_run(ctx, d_Y_f32, CUDA_R_32F,
                               d_W_e4m3, d_X_e4m3,
                               d_w_scale_f32, d_x_scale_f32,
                               0, 0,
                               n_tok, n_out, n_in);
}
