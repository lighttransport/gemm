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
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
    CUBLAS_GEMM_DEFAULT = -1
};

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

struct cublasew_context {
    cublasHandle_t handle;
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

int cublasewCreate(cublasew_context **out, CUstream stream) {
    cublasew_context *ctx;
    if (!out) return -1;
    *out = NULL;
    if (cublasewInit() != 0) return -1;

    ctx = (cublasew_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return -1;
    if (p_cublasCreate_v2(&ctx->handle) != CUBLAS_STATUS_SUCCESS) {
        free(ctx);
        return -1;
    }
    if (stream && p_cublasSetStream_v2(ctx->handle, stream) != CUBLAS_STATUS_SUCCESS) {
        p_cublasDestroy_v2(ctx->handle);
        free(ctx);
        return -1;
    }
    *out = ctx;
    return 0;
}

void cublasewDestroy(cublasew_context *ctx) {
    if (!ctx) return;
    if (ctx->handle) p_cublasDestroy_v2(ctx->handle);
    free(ctx);
}

int cublasewSetStream(cublasew_context *ctx, CUstream stream) {
    if (!ctx || !ctx->handle) return -1;
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
