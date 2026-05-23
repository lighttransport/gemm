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
typedef int cublasSideMode_t;

enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_SIDE_LEFT = 0,   /* cublas*dgmm: C = diag(x) * A */
    CUBLAS_SIDE_RIGHT = 1,  /* cublas*dgmm: C = A * diag(x) */
    CUDA_R_32F = 0,
    CUDA_R_16F = 2,
    CUDA_R_16BF = 14,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_GEMM_DEFAULT = -1
};

/* cuBLAS-LT enums (from cublasLt.h, version >= 11.4) */
enum {
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
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
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,

    CUBLASLT_EPILOGUE_DEFAULT = 1,
    CUBLASLT_EPILOGUE_BIAS = 4,
    CUBLASLT_ORDER_ROW = 1
};

typedef void *cublasLtHandle_t;
typedef void *cublasLtMatmulDesc_t;
typedef void *cublasLtMatrixLayout_t;
typedef void *cublasLtMatmulPreference_t;

/* Semi-opaque cuBLASLt descriptors. The heuristic result stride must match
 * the CUDA headers; otherwise selecting heuristic entries beyond index 0 reads
 * garbage because cuBLASLt writes a compact array of these records. */
typedef struct { uint64_t data[8]; } cublasLtMatmulAlgo_t;
typedef struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

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
/* C[m,n] = diag(x) * A[m,n] (SIDE_LEFT) — column-major. Kernel-free per-row
 * scaling primitive; used to apply per-output-channel FP8 weight scales after
 * a per-tensor FP8 matmul. */
typedef cublasStatus_t (*tcublasSdgmm)(cublasHandle_t,
                                       cublasSideMode_t,
                                       int, int,
                                       const float *, int,
                                       const float *, int,
                                       float *, int);

static cublas_lib_t g_cublas_lib;
static tcublasCreate_v2 p_cublasCreate_v2;
static tcublasDestroy_v2 p_cublasDestroy_v2;
static tcublasSetStream_v2 p_cublasSetStream_v2;
static tcublasSgemm_v2 p_cublasSgemm_v2;
static tcublasGemmEx p_cublasGemmEx;
static tcublasSdgmm p_cublasSdgmm;
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
#define CUBLASLT_HEURISTIC_MAX 32

static size_t cublasewLtWorkspaceBytes(void) {
    const char *env = getenv("CUBLASEW_LT_WORKSPACE_BYTES");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) return (size_t)v;
    }
    return CUBLASLT_WORKSPACE_BYTES;
}

static int cublasewLtMinAlignmentBytes(void) {
    const char *env = getenv("CUBLASEW_LT_MIN_ALIGNMENT_BYTES");
    if (env && env[0]) {
        int v = atoi(env);
        if (v > 0) return v;
    }
    return 0;
}

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
    int allow_tf32;  /* route the plain f32 NT path to TF32 tensor cores */
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

static int cublasewLtAlgoIndex(void) {
    const char *env = getenv("CUBLASEW_LT_ALGO_INDEX");
    int idx = env && env[0] ? atoi(env) : 0;
    return idx < 0 ? 0 : idx;
}

static int cublasewLtGetHeuristic(cublasLtHandle_t lt_handle,
                                  cublasLtMatmulDesc_t desc,
                                  cublasLtMatrixLayout_t a_layout,
                                  cublasLtMatrixLayout_t b_layout,
                                  cublasLtMatrixLayout_t c_layout,
                                  cublasLtMatrixLayout_t d_layout,
                                  cublasLtMatmulPreference_t pref,
                                  cublasLtMatmulHeuristicResult_t *heur,
                                  const char *label,
                                  int n_tok,
                                  int n_out,
                                  int n_in) {
    cublasLtMatmulHeuristicResult_t all[CUBLASLT_HEURISTIC_MAX];
    const char *dbg = getenv("CUBLASEW_DEBUG_LT");
    int returned = 0;
    int algo_index = cublasewLtAlgoIndex();
    cublasStatus_t st;

    memset(all, 0, sizeof(all));
    st = p_cublasLtMatmulAlgoGetHeuristic(lt_handle, desc,
                                          a_layout, b_layout,
                                          c_layout, d_layout,
                                          pref, CUBLASLT_HEURISTIC_MAX,
                                          all, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned <= 0) {
        if (dbg)
            fprintf(stderr, "cublasew: cublasLt heuristic failed label=%s "
                    "status=%d returned=%d (M=%d N=%d K=%d)\n",
                    label ? label : "matmul", (int)st, returned,
                    n_tok, n_out, n_in);
        return -1;
    }
    if (algo_index >= returned) {
        if (dbg)
            fprintf(stderr, "cublasew: cublasLt algo index %d out of range "
                    "returned=%d label=%s (M=%d N=%d K=%d)\n",
                    algo_index, returned, label ? label : "matmul",
                    n_tok, n_out, n_in);
        return -1;
    }
    if (dbg)
        fprintf(stderr, "cublasew: cublasLt using algo index %d/%d label=%s "
                "(M=%d N=%d K=%d ws=%zu waves=%.3f state=%d)\n",
                algo_index, returned, label ? label : "matmul",
                n_tok, n_out, n_in, all[algo_index].workspaceSize,
                all[algo_index].wavesCount, (int)all[algo_index].state);
    *heur = all[algo_index];
    return 0;
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

    /* Optional: only the per-row FP8 path needs it; tolerate absence. */
    cublasew_load_symbol((void **)&p_cublasSdgmm, "cublasSdgmm");

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
    { const char *e = getenv("CUBLASEW_ALLOW_TF32");
      ctx->allow_tf32 = (e && atoi(e) != 0) ? 1 : 0; }
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
            ctx->workspace_bytes = cublasewLtWorkspaceBytes();
            if (cuMemAlloc(&ctx->d_workspace, ctx->workspace_bytes) != CUDA_SUCCESS) {
                ctx->d_workspace = 0;
                ctx->workspace_bytes = 0;
            }
            /* Build the shared preference once. */
            if (p_cublasLtMatmulPreferenceCreate(&ctx->pref) == CUBLAS_STATUS_SUCCESS) {
                size_t ws = ctx->workspace_bytes;
                p_cublasLtMatmulPreferenceSetAttribute(ctx->pref,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
                {
                    int min_align = cublasewLtMinAlignmentBytes();
                    if (min_align > 0) {
                        p_cublasLtMatmulPreferenceSetAttribute(ctx->pref,
                            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                            &min_align, sizeof(min_align));
                        p_cublasLtMatmulPreferenceSetAttribute(ctx->pref,
                            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                            &min_align, sizeof(min_align));
                        p_cublasLtMatmulPreferenceSetAttribute(ctx->pref,
                            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                            &min_align, sizeof(min_align));
                        p_cublasLtMatmulPreferenceSetAttribute(ctx->pref,
                            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                            &min_align, sizeof(min_align));
                    }
                }
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

void cublasew_set_tf32(cublasew_context *ctx, int enable) {
    if (ctx) ctx->allow_tf32 = enable ? 1 : 0;
}

int cublasew_gemm_f32_tf32_rowmajor_nt(cublasew_context *ctx,
                                       CUdeviceptr d_Y,
                                       CUdeviceptr d_W_f32,
                                       CUdeviceptr d_X_f32,
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
                          (const void *)(uintptr_t)d_W_f32, CUDA_R_32F, n_in,
                          (const void *)(uintptr_t)d_X_f32, CUDA_R_32F, n_in,
                          &beta,
                          (void *)(uintptr_t)d_Y, CUDA_R_32F, n_out,
                          CUBLAS_COMPUTE_32F_FAST_TF32,
                          CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
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
    /* Opt-in TF32 fast path (CUBLASEW_ALLOW_TF32=1 or cublasew_set_tf32).
     * Default stays on exact SGEMM — no behavior change for existing callers. */
    if (ctx->allow_tf32)
        return cublasew_gemm_f32_tf32_rowmajor_nt(ctx, d_Y, d_W_f32, d_X_f32,
                                                  n_tok, n_out, n_in);
    return p_cublasSgemm_v2(ctx->handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            n_out, n_tok, n_in,
                            &alpha,
                            (const float *)(uintptr_t)d_W_f32, n_in,
                            (const float *)(uintptr_t)d_X_f32, n_in,
                            &beta,
                            (float *)(uintptr_t)d_Y, n_out) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cublasew_gemm_f32_rowmajor_nt_beta1(cublasew_context *ctx,
                                        CUdeviceptr d_Y,
                                        CUdeviceptr d_W_f32,
                                        CUdeviceptr d_X_f32,
                                        int n_tok,
                                        int n_out,
                                        int n_in) {
    const float alpha = 1.0f;
    const float beta = 1.0f;
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

int cublasew_gemm_f32_lt_rowmajor_nt_beta1(cublasew_context *ctx,
                                           CUdeviceptr d_Y,
                                           CUdeviceptr d_W_f32,
                                           CUdeviceptr d_X_f32,
                                           int n_tok,
                                           int n_out,
                                           int n_in) {
    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasLtMatmulDesc_t desc = NULL;
    cublasLtMatrixLayout_t a_layout = NULL, b_layout = NULL, d_layout = NULL;
    cublasLtMatmulHeuristicResult_t heur;
    cublasStatus_t st;
    int op_n = CUBLAS_OP_N;
    int row_order = CUBLASLT_ORDER_ROW;
    void *Wp = (void *)(uintptr_t)d_W_f32;
    void *Xp = (void *)(uintptr_t)d_X_f32;
    void *Yp = (void *)(uintptr_t)d_Y;

    if (!ctx || !ctx->lt_handle) return -1;
    if (!g_cublaslt_available) return -1;

    st = p_cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    if (p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                          &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS)
        goto fail;
    if (p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                          &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS)
        goto fail;

    if (p_cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_32F,
                                     (uint64_t)n_tok, (uint64_t)n_in,
                                     (int64_t)n_in) != CUBLAS_STATUS_SUCCESS)
        goto fail;
    if (p_cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                           &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS)
        goto fail;
    if (p_cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_32F,
                                     (uint64_t)n_in, (uint64_t)n_out,
                                     (int64_t)n_in) != CUBLAS_STATUS_SUCCESS)
        goto fail;
    if (p_cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_32F,
                                     (uint64_t)n_tok, (uint64_t)n_out,
                                     (int64_t)n_out) != CUBLAS_STATUS_SUCCESS)
        goto fail;
    if (p_cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                           &row_order, sizeof(row_order)) != CUBLAS_STATUS_SUCCESS)
        goto fail;

    memset(&heur, 0, sizeof(heur));
    if (cublasewLtGetHeuristic(ctx->lt_handle, desc,
                               a_layout, b_layout,
                               d_layout, d_layout,
                               ctx->pref, &heur,
                               "f32_beta1_nt",
                               n_tok, n_out, n_in) != 0)
        goto fail;

    st = p_cublasLtMatmul(ctx->lt_handle, desc,
                          &alpha,
                          Xp, a_layout,
                          Wp, b_layout,
                          &beta,
                          Yp, d_layout,
                          Yp, d_layout,
                          (const void *)&heur.algo,
                          (void *)(uintptr_t)ctx->d_workspace,
                          ctx->workspace_bytes,
                          ctx->stream);
    if (d_layout) p_cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) p_cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) p_cublasLtMatrixLayoutDestroy(a_layout);
    if (desc) p_cublasLtMatmulDescDestroy(desc);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;

fail:
    if (d_layout) p_cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) p_cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) p_cublasLtMatrixLayoutDestroy(a_layout);
    if (desc) p_cublasLtMatmulDescDestroy(desc);
    return -1;
}

int cublasew_gemm_f32_lt_bias_rowmajor_nt(cublasew_context *ctx,
                                          CUdeviceptr d_Y,
                                          CUdeviceptr d_W_f32,
                                          CUdeviceptr d_X_f32,
                                          CUdeviceptr d_bias_f32,
                                          int n_tok,
                                          int n_out,
                                          int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasLtMatmulDesc_t desc = NULL;
    cublasLtMatrixLayout_t a_layout = NULL, b_layout = NULL, d_layout = NULL;
    cublasLtMatmulHeuristicResult_t heur;
    cublasStatus_t st;
    int op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    int epilogue = CUBLASLT_EPILOGUE_BIAS;
    void *biasp = (void *)(uintptr_t)d_bias_f32;
    void *Wp = (void *)(uintptr_t)d_W_f32;
    void *Xp = (void *)(uintptr_t)d_X_f32;
    void *Yp = (void *)(uintptr_t)d_Y;
    const char *dbg = getenv("CUBLASEW_DEBUG_LT");
    const char *fail_step = "precheck";
    int fail_status = 0;

    if (!ctx || !ctx->lt_handle || !d_bias_f32) return -1;
    if (!g_cublaslt_available) return -1;

    st = p_cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) { fail_step = "desc_create"; fail_status = st; goto fail; }
    if (p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                          &op_t, sizeof(op_t)) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "set_transa"; goto fail;
    }
    if (p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                          &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "set_transb"; goto fail;
    }
    if (p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                          &epilogue, sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "set_epilogue"; goto fail;
    }
    if (p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                          &biasp, sizeof(biasp)) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "set_bias"; goto fail;
    }

    if (p_cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_32F,
                                     (uint64_t)n_in, (uint64_t)n_out,
                                     (int64_t)n_in) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "a_layout_create"; goto fail;
    }
    if (p_cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_32F,
                                     (uint64_t)n_in, (uint64_t)n_tok,
                                     (int64_t)n_in) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "b_layout_create"; goto fail;
    }
    if (p_cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_32F,
                                     (uint64_t)n_out, (uint64_t)n_tok,
                                     (int64_t)n_out) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "d_layout_create"; goto fail;
    }

    memset(&heur, 0, sizeof(heur));
    if (cublasewLtGetHeuristic(ctx->lt_handle, desc,
                               a_layout, b_layout,
                               d_layout, d_layout,
                               ctx->pref, &heur,
                               "f32_bias_nt",
                               n_tok, n_out, n_in) != 0) {
        fail_step = "heuristic"; goto fail;
    }

    st = p_cublasLtMatmul(ctx->lt_handle, desc,
                          &alpha,
                          Wp, a_layout,
                          Xp, b_layout,
                          &beta,
                          Yp, d_layout,
                          Yp, d_layout,
                          (const void *)&heur.algo,
                          (void *)(uintptr_t)ctx->d_workspace,
                          ctx->workspace_bytes,
                          ctx->stream);
    if (d_layout) p_cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) p_cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) p_cublasLtMatrixLayoutDestroy(a_layout);
    if (desc) p_cublasLtMatmulDescDestroy(desc);
    if (dbg && st != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "cublasew: cublasLt f32 bias matmul status=%d\n", (int)st);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;

fail:
    if (dbg)
        fprintf(stderr, "cublasew: cublasLt f32 bias failed at %s status=%d "
                "(M=%d N=%d K=%d)\n",
                fail_step, fail_status, n_tok, n_out, n_in);
    if (d_layout) p_cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) p_cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) p_cublasLtMatrixLayoutDestroy(a_layout);
    if (desc) p_cublasLtMatmulDescDestroy(desc);
    return -1;
}

int cublasew_gemm_f32_lt_rowmajor_nt(cublasew_context *ctx,
                                     CUdeviceptr d_Y,
                                     CUdeviceptr d_W_f32,
                                     CUdeviceptr d_X_f32,
                                     int n_tok,
                                     int n_out,
                                     int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasLtMatmulDesc_t desc = NULL;
    cublasLtMatrixLayout_t a_layout = NULL, b_layout = NULL, d_layout = NULL;
    cublasLtMatmulHeuristicResult_t heur;
    cublasStatus_t st;
    int op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    void *Wp = (void *)(uintptr_t)d_W_f32;
    void *Xp = (void *)(uintptr_t)d_X_f32;
    void *Yp = (void *)(uintptr_t)d_Y;
    const char *dbg = getenv("CUBLASEW_DEBUG_LT");
    const char *fail_step = "precheck";
    int fail_status = 0;

    if (!ctx || !ctx->lt_handle) return -1;
    if (!g_cublaslt_available) return -1;

    st = p_cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) { fail_step = "desc_create"; fail_status = st; goto fail; }
    if (p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                          &op_t, sizeof(op_t)) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "set_transa"; goto fail;
    }
    if (p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                          &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "set_transb"; goto fail;
    }

    if (p_cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_32F,
                                     (uint64_t)n_in, (uint64_t)n_out,
                                     (int64_t)n_in) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "a_layout_create"; goto fail;
    }
    if (p_cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_32F,
                                     (uint64_t)n_in, (uint64_t)n_tok,
                                     (int64_t)n_in) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "b_layout_create"; goto fail;
    }
    if (p_cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_32F,
                                     (uint64_t)n_out, (uint64_t)n_tok,
                                     (int64_t)n_out) != CUBLAS_STATUS_SUCCESS) {
        fail_step = "d_layout_create"; goto fail;
    }

    memset(&heur, 0, sizeof(heur));
    if (cublasewLtGetHeuristic(ctx->lt_handle, desc,
                               a_layout, b_layout,
                               d_layout, d_layout,
                               ctx->pref, &heur,
                               "f32_nt",
                               n_tok, n_out, n_in) != 0) {
        fail_step = "heuristic"; goto fail;
    }

    st = p_cublasLtMatmul(ctx->lt_handle, desc,
                          &alpha,
                          Wp, a_layout,
                          Xp, b_layout,
                          &beta,
                          Yp, d_layout,
                          Yp, d_layout,
                          (const void *)&heur.algo,
                          (void *)(uintptr_t)ctx->d_workspace,
                          ctx->workspace_bytes,
                          ctx->stream);
    if (d_layout) p_cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) p_cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) p_cublasLtMatrixLayoutDestroy(a_layout);
    if (desc) p_cublasLtMatmulDescDestroy(desc);
    if (dbg && st != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "cublasew: cublasLt f32 matmul status=%d\n", (int)st);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;

fail:
    if (dbg)
        fprintf(stderr, "cublasew: cublasLt f32 failed at %s status=%d "
                "(M=%d N=%d K=%d)\n",
                fail_step, fail_status, n_tok, n_out, n_in);
    if (d_layout) p_cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) p_cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) p_cublasLtMatrixLayoutDestroy(a_layout);
    if (desc) p_cublasLtMatmulDescDestroy(desc);
    return -1;
}

int cublasew_gemm_f32_pedantic_rowmajor_nt(cublasew_context *ctx,
                                           CUdeviceptr d_Y,
                                           CUdeviceptr d_W_f32,
                                           CUdeviceptr d_X_f32,
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
                          (const void *)(uintptr_t)d_W_f32, CUDA_R_32F, n_in,
                          (const void *)(uintptr_t)d_X_f32, CUDA_R_32F, n_in,
                          &beta,
                          (void *)(uintptr_t)d_Y, CUDA_R_32F, n_out,
                          CUBLAS_COMPUTE_32F_PEDANTIC,
                          CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cublasew_gemm_f32_pedantic_rowmajor_nt_strided(cublasew_context *ctx,
                                                   CUdeviceptr d_Y,
                                                   CUdeviceptr d_W_f32,
                                                   int ld_w,
                                                   CUdeviceptr d_X_f32,
                                                   int ld_x,
                                                   int n_tok,
                                                   int n_out,
                                                   int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!ctx || !ctx->handle || ld_w < n_in || ld_x < n_in) return -1;
    return p_cublasGemmEx(ctx->handle,
                          CUBLAS_OP_T, CUBLAS_OP_N,
                          n_out, n_tok, n_in,
                          &alpha,
                          (const void *)(uintptr_t)d_W_f32, CUDA_R_32F, ld_w,
                          (const void *)(uintptr_t)d_X_f32, CUDA_R_32F, ld_x,
                          &beta,
                          (void *)(uintptr_t)d_Y, CUDA_R_32F, n_out,
                          CUBLAS_COMPUTE_32F_PEDANTIC,
                          CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cublasew_gemm_f32_pedantic_rowmajor_nn(cublasew_context *ctx,
                                           CUdeviceptr d_Y,
                                           int ld_y,
                                           CUdeviceptr d_A_f32,
                                           CUdeviceptr d_B_f32,
                                           int n_tok,
                                           int n_out,
                                           int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!ctx || !ctx->handle || ld_y < n_out) return -1;
    /* Row-major Y[M,N] = A[M,K] * B[K,N] is column-major
     * Y^T[N,M] = B^T[N,K] * A^T[K,M]. The row-major buffers already have
     * those transposed column-major layouts, so both cuBLAS operands are N.
     */
    return p_cublasGemmEx(ctx->handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          n_out, n_tok, n_in,
                          &alpha,
                          (const void *)(uintptr_t)d_B_f32, CUDA_R_32F, n_out,
                          (const void *)(uintptr_t)d_A_f32, CUDA_R_32F, n_in,
                          &beta,
                          (void *)(uintptr_t)d_Y, CUDA_R_32F, ld_y,
                          CUBLAS_COMPUTE_32F_PEDANTIC,
                          CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cublasew_gemm_f32_rowmajor_nn(cublasew_context *ctx,
                                  CUdeviceptr d_Y,
                                  int ld_y,
                                  CUdeviceptr d_A_f32,
                                  CUdeviceptr d_B_f32,
                                  int n_tok,
                                  int n_out,
                                  int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!ctx || !ctx->handle || ld_y < n_out) return -1;
    return p_cublasSgemm_v2(ctx->handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n_out, n_tok, n_in,
                            &alpha,
                            (const float *)(uintptr_t)d_B_f32, n_out,
                            (const float *)(uintptr_t)d_A_f32, n_in,
                            &beta,
                            (float *)(uintptr_t)d_Y, ld_y) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cublasew_gemm_f32_pedantic_rowmajor_nn_stridedB(cublasew_context *ctx,
                                                    CUdeviceptr d_Y,
                                                    int ld_y,
                                                    CUdeviceptr d_A_f32,
                                                    CUdeviceptr d_B_f32,
                                                    int ld_b,
                                                    int n_tok,
                                                    int n_out,
                                                    int n_in) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!ctx || !ctx->handle || ld_y < n_out || ld_b < n_out) return -1;
    return p_cublasGemmEx(ctx->handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          n_out, n_tok, n_in,
                          &alpha,
                          (const void *)(uintptr_t)d_B_f32, CUDA_R_32F, ld_b,
                          (const void *)(uintptr_t)d_A_f32, CUDA_R_32F, n_in,
                          &beta,
                          (void *)(uintptr_t)d_Y, CUDA_R_32F, ld_y,
                          CUBLAS_COMPUTE_32F_PEDANTIC,
                          CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS ? 0 : -1;
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
    const char *dbg = getenv("CUBLASEW_DEBUG_LT");
    const char *failpt = "?";

    st = p_cublasLtMatmulDescCreate(&e->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) { failpt = "DescCreate"; goto fail; }
    if (p_cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)) != CUBLAS_STATUS_SUCCESS) { failpt = "TRANSA"; goto fail; }
    if (p_cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)) != CUBLAS_STATUS_SUCCESS) { failpt = "TRANSB"; goto fail; }
    /* Per-tensor weight scale: bake a single A_SCALE_POINTER scalar into the
     * desc. (Per-row FP8 weight scaling is done outside this path as a post-GEMM
     * Sdgmm row-scale, since consumer GeForce cuBLAS-LT FP8 rejects vector A
     * scale modes.) */
    if (has_w_scale &&
        p_cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &Wsp, sizeof(Wsp)) != CUBLAS_STATUS_SUCCESS)
        { failpt = "A_SCALE_POINTER"; goto fail; }
    if (has_x_scale &&
        p_cublasLtMatmulDescSetAttribute(e->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &Xsp, sizeof(Xsp)) != CUBLAS_STATUS_SUCCESS)
        { failpt = "B_SCALE_POINTER"; goto fail; }

    if (p_cublasLtMatrixLayoutCreate(&e->a_layout, in_dt, (uint64_t)n_in, (uint64_t)n_out, (int64_t)n_in) != CUBLAS_STATUS_SUCCESS) { failpt = "a_layout"; goto fail; }
    if (p_cublasLtMatrixLayoutCreate(&e->b_layout, in_dt, (uint64_t)n_in, (uint64_t)n_tok, (int64_t)n_in) != CUBLAS_STATUS_SUCCESS) { failpt = "b_layout"; goto fail; }
    if (p_cublasLtMatrixLayoutCreate(&e->d_layout, y_dtype, (uint64_t)n_out, (uint64_t)n_tok, (int64_t)n_out) != CUBLAS_STATUS_SUCCESS) { failpt = "d_layout"; goto fail; }

    int returned = 0;
    memset(&e->heur, 0, sizeof(e->heur));
    st = p_cublasLtMatmulAlgoGetHeuristic(ctx->lt_handle, e->desc,
                                         e->a_layout, e->b_layout,
                                         e->d_layout, e->d_layout,
                                         ctx->pref, 1, &e->heur, &returned);
    if (st != CUBLAS_STATUS_SUCCESS) {
        if (dbg) fprintf(stderr, "cublasew: heuristic status=%d\n", (int)st);
        failpt = "heuristic"; goto fail;
    }
    if (returned <= 0) { failpt = "heuristic(0 algos)"; goto fail; }

    e->n_tok = n_tok; e->n_out = n_out; e->n_in = n_in;
    e->y_dtype = (int)y_dtype;
    e->has_w_scale = has_w_scale;
    e->has_x_scale = has_x_scale;
    e->valid = 1;
    ctx->cache_n++;
    return e;

fail:
    if (dbg) fprintf(stderr, "cublasew: lt fp8 build failed at %s "
                     "(shape %dx%dx%d y_dtype=%d)\n",
                     failpt, n_tok, n_out, n_in, (int)y_dtype);
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
                          (const void *)&alpha,
                          Wp, e->a_layout,
                          Xp, e->b_layout,
                          &beta,
                          Yp, e->d_layout,
                          Yp, e->d_layout,
                          (const void *)&e->heur.algo,
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

int cublasew_gemm_fp8_e4m3_f32out_wperrow_rowmajor_nt(cublasew_context *ctx,
                                                      CUdeviceptr d_Y_f32,
                                                      CUdeviceptr d_W_e4m3,
                                                      CUdeviceptr d_X_e4m3,
                                                      CUdeviceptr d_w_scale_vec_f32,
                                                      CUdeviceptr d_x_scale_f32,
                                                      int n_tok,
                                                      int n_out,
                                                      int n_in) {
    /* Per-row (per-output-channel) FP8 weight scaling on consumer GeForce.
     *
     * Consumer GeForce Blackwell cuBLAS-LT FP8 supports only a per-tensor A
     * scale (vector A_SCALE_MODE / alpha-device-vector are NOT_SUPPORTED at the
     * heuristic stage). But the per-row scale s[o] factors cleanly out of the
     * dot-product sum:
     *
     *   D[o,t] = s[o] * sum_k ( W_fp8[o,k] * x_scale * X_fp8[t,k] )
     *
     * so we (1) run the per-tensor FP8 matmul with NO A scale (the weight is
     * already FP8-quantized per row, x_scale applied to B) into an F32 D, then
     * (2) scale each output row o by s[o] with a kernel-free cublasSdgmm
     * (C = diag(s) * D, in-place). The precision win is captured at quantization
     * time — every weight row uses its own scale, so small-magnitude channels
     * keep the full e4m3 range a single per-tensor scale would crush.
     *
     * Output is F32 because Sdgmm is F32-typed and cublasew adds no conversion
     * kernels. D is column-major n_out x n_tok (row-major Y[n_tok, n_out]).
     * Returns -1 if Sdgmm is unavailable or the FP8 F32-out matmul is rejected,
     * so the caller can fall back to per-tensor. */
    if (!d_w_scale_vec_f32) return -1;  /* per-row scale vector is mandatory */
    if (!p_cublasSdgmm) return -1;      /* no kernel-free row-scale primitive */

    /* (1) per-tensor FP8 matmul, no A scale, B scale = x_scale, F32 output. */
    int rc = cublasew_lt_fp8_run(ctx, d_Y_f32, CUDA_R_32F,
                                 d_W_e4m3, d_X_e4m3,
                                 0, d_x_scale_f32,
                                 0, 0,
                                 n_tok, n_out, n_in);
    if (rc != 0) return -1;

    /* (2) row-scale: C = diag(s) * D, column-major D is n_out x n_tok. */
    cublasStatus_t st = p_cublasSdgmm(ctx->handle, CUBLAS_SIDE_LEFT,
                                      n_out, n_tok,
                                      (const float *)(uintptr_t)d_Y_f32, n_out,
                                      (const float *)(uintptr_t)d_w_scale_vec_f32, 1,
                                      (float *)(uintptr_t)d_Y_f32, n_out);
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
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
