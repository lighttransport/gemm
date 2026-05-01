/*
 * bench_cuda_gemm.c
 *
 * Tensor-core GEMM benchmark for sm_120 (Blackwell GeForce, RTX 5060 Ti).
 *
 * Compares two backends per dtype:
 *   - ptx    : our PTX kernels in cuda_gemm_ptx_kernels.h, NVRTC-compiled
 *   - cublas : cuBLAS (f16/bf16 via cublasew GemmEx; fp8 via cuBLASLt)
 *
 * Layout: Y[M,N] = X[M,K] * W[N,K]^T   row-major; W stored "transposed".
 *
 * Reports ms / TFLOP/s / peak% / accuracy (cosine vs CPU FP32 reference on
 * a 64-row sample to keep CPU validation tractable on 8192^3).
 *
 * Peaks (RTX 5060 Ti, GeForce-throttled):
 *   f16  / bf16: 42 TFLOP/s
 *   fp8  e4m3 : 84 TFLOP/s
 *
 * Usage:
 *   ./bench_cuda_gemm --dtype f16  --mode all --shape all
 *   ./bench_cuda_gemm --dtype bf16 --mode ptx --shape mm0
 *   ./bench_cuda_gemm --dtype fp8  --mode cublas --m 4096 --n 4096 --k 4096
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>

#include "../cuew.h"
#include "../cublasew.h"

#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#include "cuda_gemm_ptx_kernels.h"

#define CPU_COMPUTE_IMPLEMENTATION
#include "../../common/cpu_compute.h"

/* ---------------------------------------------------------------------- */
/* Types and shape table                                                   */
/* ---------------------------------------------------------------------- */

typedef enum { DT_F16, DT_BF16, DT_FP8 } dtype_t;
typedef enum { MD_PTX, MD_CUBLAS, MD_CUTILE, MD_ALL } bmode_t;

typedef struct {
    const char *name;
    int M, N, K;
} shape_t;

static const shape_t g_shapes[] = {
    /* generic squares */
    {"square_1k", 1024, 1024, 1024},
    {"square_2k", 2048, 2048, 2048},
    {"square_4k", 4096, 4096, 4096},
    {"square_8k", 8192, 8192, 8192},
    /* Qwen3 / VLM shapes (mirrors rdna4/vlm/bench_vlm_gemm.c) */
    {"mm0",       1024, 4608, 4608},
    {"mm2",       1024, 1152, 4608},
    {"qkv",        512, 4096, 4096},
    {"attn_out",   512, 4096, 4096},
    {"ffn_up",     512, 11008, 4096},
    {"ffn_down",   512, 4096, 11008},
};
#define N_SHAPES ((int)(sizeof(g_shapes) / sizeof(g_shapes[0])))

static const char *dtype_name(dtype_t dt) {
    switch (dt) { case DT_F16: return "f16"; case DT_BF16: return "bf16"; case DT_FP8: return "fp8"; }
    return "?";
}
static double dtype_peak_tflops(dtype_t dt) {
    switch (dt) { case DT_F16: return 42.0; case DT_BF16: return 42.0; case DT_FP8: return 84.0; }
    return 0;
}
static int dtype_bytes(dtype_t dt) {
    switch (dt) { case DT_F16: case DT_BF16: return 2; case DT_FP8: return 1; }
    return 0;
}
static int dtype_k_alignment(dtype_t dt) {
    switch (dt) { case DT_F16: case DT_BF16: return 16; case DT_FP8: return 32; }
    return 16;
}

/* ---------------------------------------------------------------------- */
/* dtype conversions (host-side, for quantize-then-dequantize golden ref)  */
/* ---------------------------------------------------------------------- */

static uint16_t host_f32_to_bf16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    /* round-to-nearest-even */
    uint32_t lsb = (b >> 16) & 1;
    uint32_t round = 0x7FFF + lsb;
    return (uint16_t)((b + round) >> 16);
}
static float host_bf16_to_f32(uint16_t b) {
    uint32_t v = (uint32_t)b << 16; float f; memcpy(&f, &v, 4); return f;
}
static float host_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f_exp, f_mant;
    if (exp == 0 && mant == 0) {
        uint32_t bits = sign << 31; float f; memcpy(&f, &bits, 4); return f;
    } else if (exp == 0) {
        /* subnormal */
        float v = (float)mant / (float)(1 << 24);
        return sign ? -v : v;
    } else if (exp == 31) {
        f_exp = 255; f_mant = mant << 13;
    } else {
        f_exp = exp + (127 - 15); f_mant = mant << 13;
    }
    uint32_t bits = (sign << 31) | (f_exp << 23) | f_mant;
    float f; memcpy(&f, &bits, 4); return f;
}
static float host_fp8_e4m3_to_f32(uint8_t b) {
    if (b == 0x00 || b == 0x80) return 0.0f;
    int sign = (b >> 7) & 1;
    int exp  = (b >> 3) & 0xF;
    int mant = b & 0x7;
    float f;
    if (exp == 0) {
        /* subnormal: 2^-6 * (mant/8) */
        f = (float)mant * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else if (exp == 0xF && mant == 0x7) {
        /* NaN in e4m3 (no Inf encoding) */
        f = NAN;
    } else {
        int e = exp - 7;
        float scale = (e >= 0) ? (float)(1 << e) : 1.0f / (float)(1 << -e);
        f = (1.0f + (float)mant / 8.0f) * scale;
    }
    return sign ? -f : f;
}

/* ---------------------------------------------------------------------- */
/* Pack / unpack between F32 buffer and dtype byte buffer                  */
/* ---------------------------------------------------------------------- */

static void quantize_f32(const float *src, void *dst, size_t n, dtype_t dt) {
    if (dt == DT_F16) {
        uint16_t *q = (uint16_t *)dst;
        for (size_t i = 0; i < n; i++) q[i] = cu_f32_to_f16(src[i]);
    } else if (dt == DT_BF16) {
        uint16_t *q = (uint16_t *)dst;
        for (size_t i = 0; i < n; i++) q[i] = host_f32_to_bf16(src[i]);
    } else { /* fp8 e4m3 */
        uint8_t *q = (uint8_t *)dst;
        for (size_t i = 0; i < n; i++) q[i] = cu_f32_to_fp8_e4m3(src[i]);
    }
}

static void dequantize(const void *src, float *dst, size_t n, dtype_t dt) {
    if (dt == DT_F16) {
        const uint16_t *q = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) dst[i] = host_f16_to_f32(q[i]);
    } else if (dt == DT_BF16) {
        const uint16_t *q = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) dst[i] = host_bf16_to_f32(q[i]);
    } else {
        const uint8_t *q = (const uint8_t *)src;
        for (size_t i = 0; i < n; i++) dst[i] = host_fp8_e4m3_to_f32(q[i]);
    }
}

/* ---------------------------------------------------------------------- */
/* Validation                                                              */
/* ---------------------------------------------------------------------- */

typedef struct {
    int ok;
    float max_abs_err;
    double cos_sim;
} acc_result_t;

static acc_result_t validate(const float *Y_gpu, const float *Y_ref, int M, int N, int rows_to_check) {
    acc_result_t r = {0, 0.0f, 0.0};
    if (rows_to_check > M) rows_to_check = M;
    double dot = 0, ng = 0, nr = 0;
    float max_err = 0;
    for (int m = 0; m < rows_to_check; m++) {
        for (int n = 0; n < N; n++) {
            float g = Y_gpu[(size_t)m * N + n];
            float ref = Y_ref[(size_t)m * N + n];
            float e = fabsf(g - ref);
            if (e > max_err) max_err = e;
            dot += (double)g * ref;
            ng  += (double)g * g;
            nr  += (double)ref * ref;
        }
    }
    r.max_abs_err = max_err;
    r.cos_sim = (ng > 0 && nr > 0) ? dot / (sqrt(ng) * sqrt(nr)) : 0.0;
    r.ok = (r.cos_sim >= 0.999);
    return r;
}

/* ---------------------------------------------------------------------- */
/* cuBLASLt FP8 path (lifted from cuda/fp8/cublas_fp8_gemm.c)              */
/* ---------------------------------------------------------------------- */

typedef enum {
    LT_R_32F  = 0,
    LT_R_16F  = 2,
    LT_R_16BF = 14,
    LT_R_8F_E4M3 = 28,
} lt_dtype_t;
typedef enum {
    LT_COMPUTE_32F = 68,
} lt_compute_t;
typedef enum { LT_OP_N = 0, LT_OP_T = 1 } lt_op_t;
typedef enum {
    LT_MATMUL_DESC_TRANSA = 0,
    LT_MATMUL_DESC_TRANSB = 1,
    LT_MATMUL_DESC_A_SCALE_POINTER = 17,
    LT_MATMUL_DESC_B_SCALE_POINTER = 18,
    LT_MATMUL_DESC_D_SCALE_POINTER = 20,
    LT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
} lt_attr_t;

enum {
    LT_COMPUTE_32F_FAST_TF32 = 77,
};

typedef void *cublasLtHandle_t;
typedef void *cublasLtMatmulDesc_t;
typedef void *cublasLtMatrixLayout_t;
typedef void *cublasLtMatmulPreference_t;
typedef struct { void *algo; size_t workspaceSize; int status; float wavesCount; int reserved[4]; } lt_heuristic_t;

static void *g_lt_lib = NULL;
static int (*p_LtCreate)(cublasLtHandle_t *);
static int (*p_LtDestroy)(cublasLtHandle_t);
static int (*p_LtMatmulDescCreate)(cublasLtMatmulDesc_t *, lt_compute_t, lt_dtype_t);
static int (*p_LtMatmulDescDestroy)(cublasLtMatmulDesc_t);
static int (*p_LtMatmulDescSetAttr)(cublasLtMatmulDesc_t, lt_attr_t, const void *, size_t);
static int (*p_LtMatrixLayoutCreate)(cublasLtMatrixLayout_t *, lt_dtype_t, uint64_t, uint64_t, int64_t);
static int (*p_LtMatrixLayoutDestroy)(cublasLtMatrixLayout_t);
static int (*p_LtMatmulPrefCreate)(cublasLtMatmulPreference_t *);
static int (*p_LtMatmulPrefDestroy)(cublasLtMatmulPreference_t);
static int (*p_LtMatmulPrefSetAttr)(cublasLtMatmulPreference_t, lt_attr_t, const void *, size_t);
static int (*p_LtMatmulAlgoGetHeuristic)(cublasLtHandle_t, cublasLtMatmulDesc_t,
                                         cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                         cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                         cublasLtMatmulPreference_t, int, lt_heuristic_t *, int *);
static int (*p_LtMatmul)(cublasLtHandle_t, cublasLtMatmulDesc_t, const void *,
                         const void *, cublasLtMatrixLayout_t,
                         const void *, cublasLtMatrixLayout_t, const void *,
                         const void *, cublasLtMatrixLayout_t,
                         void *, cublasLtMatrixLayout_t,
                         const void *, void *, size_t, CUstream);

static int load_cublasLt(void) {
    if (g_lt_lib) return 0;
    const char *names[] = { "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so", NULL };
    for (int i = 0; names[i]; i++) {
        g_lt_lib = dlopen(names[i], RTLD_NOW);
        if (g_lt_lib) break;
    }
    if (!g_lt_lib) { fprintf(stderr, "  cuBLASLt: dlopen failed\n"); return -1; }
#define LT_SYM(name, fn) do { \
    *(void **)&p_##name = dlsym(g_lt_lib, fn); \
    if (!p_##name) { fprintf(stderr, "  cuBLASLt: missing %s\n", fn); return -1; } \
} while (0)
    LT_SYM(LtCreate,                "cublasLtCreate");
    LT_SYM(LtDestroy,               "cublasLtDestroy");
    LT_SYM(LtMatmulDescCreate,      "cublasLtMatmulDescCreate");
    LT_SYM(LtMatmulDescDestroy,     "cublasLtMatmulDescDestroy");
    LT_SYM(LtMatmulDescSetAttr,     "cublasLtMatmulDescSetAttribute");
    LT_SYM(LtMatrixLayoutCreate,    "cublasLtMatrixLayoutCreate");
    LT_SYM(LtMatrixLayoutDestroy,   "cublasLtMatrixLayoutDestroy");
    LT_SYM(LtMatmulPrefCreate,      "cublasLtMatmulPreferenceCreate");
    LT_SYM(LtMatmulPrefDestroy,     "cublasLtMatmulPreferenceDestroy");
    LT_SYM(LtMatmulPrefSetAttr,     "cublasLtMatmulPreferenceSetAttribute");
    LT_SYM(LtMatmulAlgoGetHeuristic,"cublasLtMatmulAlgoGetHeuristic");
    LT_SYM(LtMatmul,                "cublasLtMatmul");
#undef LT_SYM
    return 0;
}

/* Run FP8 GEMM via cuBLASLt and time it.
 * Inputs are E4M3 row-major buffers d_X[M,K], d_W[N,K]; output d_Y is FP32 [M,N].
 * cuBLASLt FP8 requires column-major-friendly layout; we use the trick from
 * cuda/fp8/cublas_fp8_gemm.c: pass W as A and X as B, with TRANSA so that
 * the column-major view yields D[N,M] = W * X^T but we only need the timing.
 *
 * For correctness: we compute Y = X @ W^T (row-major) -> in column-major,
 * Y^T[N,M] = W[N,K] @ (X[M,K])^T = W * X^T. So we pass A=W (no transpose), B=X
 * with TRANSB, and result lands in row-major D as Y[M,N] of column-major D[N,M].
 * Since D has layout dim N (rows in cm) x M (cols in cm), row-major reading
 * gives Y[m,n] at byte offset (n*M + m)*4 — i.e. column-major. We need to
 * transpose at validation time.
 */
static int cublas_lt_fp8_gemm_timed(CUstream stream,
                                     CUdeviceptr d_W, CUdeviceptr d_X, CUdeviceptr d_Y,
                                     int M, int N, int K,
                                     int warmup, int iters, float *avg_ms_out) {
    cublasLtHandle_t handle = NULL;
    if (p_LtCreate(&handle) != 0) return -1;

    cublasLtMatmulDesc_t desc = NULL;
    /* FP8 on cuBLAS requires FAST_TF32 compute and FP32 scale type. */
    if (p_LtMatmulDescCreate(&desc, LT_COMPUTE_32F_FAST_TF32, LT_R_32F) != 0) return -1;
    /* Math: D_cm[N,M] (= Y_rm[M,N]) = op(A) op(B). Choose op(A)=A^T with A_cm=W[K,N],
     * op(B)=B with B_cm=X[K,M]. Then op(A) is [N,K], op(B) is [K,M], D is [N,M]. */
    int ta = LT_OP_T, tb = LT_OP_N;
    p_LtMatmulDescSetAttr(desc, LT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    p_LtMatmulDescSetAttr(desc, LT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));

    /* FP8 requires scale pointers (device buffers holding 1.0). */
    CUdeviceptr d_scale_a, d_scale_b, d_scale_d;
    float ones[3] = { 1.0f, 1.0f, 1.0f };
    if (cuMemAlloc(&d_scale_a, sizeof(float)) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&d_scale_b, sizeof(float)) != CUDA_SUCCESS) return -1;
    if (cuMemAlloc(&d_scale_d, sizeof(float)) != CUDA_SUCCESS) return -1;
    cuMemcpyHtoD(d_scale_a, &ones[0], sizeof(float));
    cuMemcpyHtoD(d_scale_b, &ones[1], sizeof(float));
    cuMemcpyHtoD(d_scale_d, &ones[2], sizeof(float));
    p_LtMatmulDescSetAttr(desc, LT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_a, sizeof(d_scale_a));
    p_LtMatmulDescSetAttr(desc, LT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_b, sizeof(d_scale_b));
    p_LtMatmulDescSetAttr(desc, LT_MATMUL_DESC_D_SCALE_POINTER, &d_scale_d, sizeof(d_scale_d));

    /* A = W (row-major [N,K]) viewed as cm[K,N] ld=K
     * B = X (row-major [M,K]) viewed as cm[K,M] ld=K
     * D = Y (row-major [M,N]) viewed as cm[N,M] ld=N */
    cublasLtMatrixLayout_t lA = NULL, lB = NULL, lD = NULL;
    if (p_LtMatrixLayoutCreate(&lA, LT_R_8F_E4M3, K, N, K) != 0) return -1;
    if (p_LtMatrixLayoutCreate(&lB, LT_R_8F_E4M3, K, M, K) != 0) return -1;
    if (p_LtMatrixLayoutCreate(&lD, LT_R_32F,     N, M, N) != 0) return -1;

    cublasLtMatmulPreference_t pref = NULL;
    if (p_LtMatmulPrefCreate(&pref) != 0) return -1;
    size_t ws_bytes = 32 * 1024 * 1024;
    p_LtMatmulPrefSetAttr(pref, LT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));

    CUdeviceptr d_ws;
    if (cuMemAlloc(&d_ws, ws_bytes) != CUDA_SUCCESS) return -1;

    lt_heuristic_t heur = {0};
    int n_results = 0;
    if (p_LtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lD, lD, pref, 1, &heur, &n_results) != 0
        || n_results == 0) {
        fprintf(stderr, "  cuBLASLt: no FP8 algo found\n");
        cuMemFree(d_ws);
        return -1;
    }

    float alpha = 1.0f, beta = 0.0f;
    for (int i = 0; i < warmup; i++) {
        if (p_LtMatmul(handle, desc, &alpha,
                       (void *)(uintptr_t)d_W, lA,
                       (void *)(uintptr_t)d_X, lB, &beta,
                       (void *)(uintptr_t)d_Y, lD,
                       (void *)(uintptr_t)d_Y, lD,
                       &heur.algo, (void *)(uintptr_t)d_ws, ws_bytes, stream) != 0) {
            fprintf(stderr, "  cuBLASLt: matmul failed (warmup)\n");
            cuMemFree(d_ws);
            return -1;
        }
    }
    cuStreamSynchronize(stream);

    CUevent start, stop;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop,  CU_EVENT_DEFAULT);
    cuEventRecord(start, stream);
    for (int i = 0; i < iters; i++) {
        p_LtMatmul(handle, desc, &alpha,
                   (void *)(uintptr_t)d_W, lA,
                   (void *)(uintptr_t)d_X, lB, &beta,
                   (void *)(uintptr_t)d_Y, lD,
                   (void *)(uintptr_t)d_Y, lD,
                   &heur.algo, (void *)(uintptr_t)d_ws, ws_bytes, stream);
    }
    cuEventRecord(stop, stream);
    cuEventSynchronize(stop);
    float ms;
    cuEventElapsedTime(&ms, start, stop);
    *avg_ms_out = ms / iters;
    cuEventDestroy(start);
    cuEventDestroy(stop);

    cuMemFree(d_ws);
    cuMemFree(d_scale_a);
    cuMemFree(d_scale_b);
    cuMemFree(d_scale_d);
    p_LtMatmulPrefDestroy(pref);
    p_LtMatrixLayoutDestroy(lD);
    p_LtMatrixLayoutDestroy(lB);
    p_LtMatrixLayoutDestroy(lA);
    p_LtMatmulDescDestroy(desc);
    p_LtDestroy(handle);
    return 0;
}

/* ---------------------------------------------------------------------- */
/* PTX kernel module cache                                                 */
/* ---------------------------------------------------------------------- */

typedef enum { PTX_V1 = 1, PTX_V2 = 2, PTX_V3 = 3, PTX_V4 = 4, PTX_V5 = 5, PTX_V6 = 6, PTX_V7 = 7 } ptx_rev_t;

typedef struct {
    /* v1 modules + functions */
    CUmodule mod_f16, mod_bf16, mod_fp8;
    CUfunction f_f16, f_bf16, f_fp8;
    int built_f16, built_bf16, built_fp8;
    /* v2 modules + functions (f16/bf16 only) */
    CUmodule mod_f16_v2, mod_bf16_v2;
    CUfunction f_f16_v2, f_bf16_v2;
    int built_f16_v2, built_bf16_v2;
    /* v3 modules + functions (f16/bf16 only) */
    CUmodule mod_f16_v3, mod_bf16_v3;
    CUfunction f_f16_v3, f_bf16_v3;
    int built_f16_v3, built_bf16_v3;
    /* v4 modules + functions (f16/bf16 only) */
    CUmodule mod_f16_v4, mod_bf16_v4;
    CUfunction f_f16_v4, f_bf16_v4;
    int built_f16_v4, built_bf16_v4;
    /* v5 modules + functions (f16/bf16/fp8) */
    CUmodule mod_f16_v5, mod_bf16_v5, mod_fp8_v5;
    CUfunction f_f16_v5, f_bf16_v5, f_fp8_v5;
    int built_f16_v5, built_bf16_v5, built_fp8_v5;
    /* v6 modules + functions (f16/bf16 only) */
    CUmodule mod_f16_v6, mod_bf16_v6;
    CUfunction f_f16_v6, f_bf16_v6;
    int built_f16_v6, built_bf16_v6;
    /* v7 modules + functions (f16/bf16/fp8) */
    CUmodule mod_f16_v7, mod_bf16_v7, mod_fp8_v7;
    CUfunction f_f16_v7, f_bf16_v7, f_fp8_v7;
    int built_f16_v7, built_bf16_v7, built_fp8_v7;
} kernel_cache_t;

static int build_kernel(kernel_cache_t *kc, dtype_t dt, ptx_rev_t rev, CUdevice dev, int verbose) {
    CUmodule *mod;
    CUfunction *fn;
    const char *src, *name;
    int *built;
    if (rev == PTX_V7) {
        switch (dt) {
            case DT_F16:  mod = &kc->mod_f16_v7;  fn = &kc->f_f16_v7;  src = k_gemm_f16_v7_src;  name = "gemm_f16_v7";  built = &kc->built_f16_v7;  break;
            case DT_BF16: mod = &kc->mod_bf16_v7; fn = &kc->f_bf16_v7; src = k_gemm_bf16_v7_src; name = "gemm_bf16_v7"; built = &kc->built_bf16_v7; break;
            case DT_FP8:  mod = &kc->mod_fp8_v7;  fn = &kc->f_fp8_v7;  src = k_gemm_fp8_v7_src;  name = "gemm_fp8_v7";  built = &kc->built_fp8_v7;  break;
            default: return -1;
        }
    } else if (rev == PTX_V6) {
        switch (dt) {
            case DT_F16:  mod = &kc->mod_f16_v6;  fn = &kc->f_f16_v6;  src = k_gemm_f16_v6_src;  name = "gemm_f16_v6";  built = &kc->built_f16_v6;  break;
            case DT_BF16: mod = &kc->mod_bf16_v6; fn = &kc->f_bf16_v6; src = k_gemm_bf16_v6_src; name = "gemm_bf16_v6"; built = &kc->built_bf16_v6; break;
            case DT_FP8:  mod = &kc->mod_fp8;  fn = &kc->f_fp8;  src = k_gemm_fp8_src;  name = "gemm_fp8";  built = &kc->built_fp8;  break;
            default: return -1;
        }
    } else if (rev == PTX_V5) {
        switch (dt) {
            case DT_F16:  mod = &kc->mod_f16_v5;  fn = &kc->f_f16_v5;  src = k_gemm_f16_v5_src;  name = "gemm_f16_v5";  built = &kc->built_f16_v5;  break;
            case DT_BF16: mod = &kc->mod_bf16_v5; fn = &kc->f_bf16_v5; src = k_gemm_bf16_v5_src; name = "gemm_bf16_v5"; built = &kc->built_bf16_v5; break;
            case DT_FP8:  mod = &kc->mod_fp8_v5;  fn = &kc->f_fp8_v5;  src = k_gemm_fp8_v5_src;  name = "gemm_fp8_v5";  built = &kc->built_fp8_v5;  break;
            default: return -1;
        }
    } else if (rev == PTX_V4) {
        switch (dt) {
            case DT_F16:  mod = &kc->mod_f16_v4;  fn = &kc->f_f16_v4;  src = k_gemm_f16_v4_src;  name = "gemm_f16_v4";  built = &kc->built_f16_v4;  break;
            case DT_BF16: mod = &kc->mod_bf16_v4; fn = &kc->f_bf16_v4; src = k_gemm_bf16_v4_src; name = "gemm_bf16_v4"; built = &kc->built_bf16_v4; break;
            case DT_FP8:  mod = &kc->mod_fp8;  fn = &kc->f_fp8;  src = k_gemm_fp8_src;  name = "gemm_fp8";  built = &kc->built_fp8;  break;
            default: return -1;
        }
    } else if (rev == PTX_V3) {
        switch (dt) {
            case DT_F16:  mod = &kc->mod_f16_v3;  fn = &kc->f_f16_v3;  src = k_gemm_f16_v3_src;  name = "gemm_f16_v3";  built = &kc->built_f16_v3;  break;
            case DT_BF16: mod = &kc->mod_bf16_v3; fn = &kc->f_bf16_v3; src = k_gemm_bf16_v3_src; name = "gemm_bf16_v3"; built = &kc->built_bf16_v3; break;
            case DT_FP8:  mod = &kc->mod_fp8;  fn = &kc->f_fp8;  src = k_gemm_fp8_src;  name = "gemm_fp8";  built = &kc->built_fp8;  break;
            default: return -1;
        }
    } else if (rev == PTX_V2) {
        switch (dt) {
            case DT_F16:  mod = &kc->mod_f16_v2;  fn = &kc->f_f16_v2;  src = k_gemm_f16_v2_src;  name = "gemm_f16_v2";  built = &kc->built_f16_v2;  break;
            case DT_BF16: mod = &kc->mod_bf16_v2; fn = &kc->f_bf16_v2; src = k_gemm_bf16_v2_src; name = "gemm_bf16_v2"; built = &kc->built_bf16_v2; break;
            case DT_FP8:  /* v2 unsupported for fp8 — fall back to v1 */
                          mod = &kc->mod_fp8;  fn = &kc->f_fp8;  src = k_gemm_fp8_src;  name = "gemm_fp8";  built = &kc->built_fp8;  break;
            default: return -1;
        }
    } else {
        switch (dt) {
            case DT_F16:  mod = &kc->mod_f16;  fn = &kc->f_f16;  src = k_gemm_f16_src;  name = "gemm_f16";  built = &kc->built_f16;  break;
            case DT_BF16: mod = &kc->mod_bf16; fn = &kc->f_bf16; src = k_gemm_bf16_src; name = "gemm_bf16"; built = &kc->built_bf16; break;
            case DT_FP8:  mod = &kc->mod_fp8;  fn = &kc->f_fp8;  src = k_gemm_fp8_src;  name = "gemm_fp8";  built = &kc->built_fp8;  break;
            default: return -1;
        }
    }
    if (*built) return 0;
    if (cu_compile_kernels(mod, dev, src, name, verbose, "bench_cuda_gemm") < 0) return -1;
    if (cuModuleGetFunction(fn, *mod, name) != CUDA_SUCCESS) {
        fprintf(stderr, "  cuModuleGetFunction(%s) failed\n", name);
        return -1;
    }
    *built = 1;
    return 0;
}

/* ---------------------------------------------------------------------- */
/* Bench one (dtype, mode, shape) combo                                    */
/* ---------------------------------------------------------------------- */

typedef struct {
    int M, N, K;
    int warmup, iters;
    int verify;
    int verify_rows;
    int verbose;
    ptx_rev_t ptx_rev;
    cublasew_context *blas_ctx;
    kernel_cache_t *kc;
    CUdevice dev;
    CUstream stream;
    /* Persistent random source (F32) and CPU reference, allocated by caller. */
    float *X_f32, *W_f32;     /* quantize-then-dequantize representation */
    float *Y_ref;              /* CPU FP32 reference (verify_rows × N) */
    /* Persistent device buffers (allocated/sized per-shape upstream) */
    CUdeviceptr d_X, d_W, d_Y;
    void *Xq, *Wq;             /* host quantized buffers (size_dt) */
} bench_ctx_t;

static int run_ptx(bench_ctx_t *b, dtype_t dt, float *avg_ms_out) {
    ptx_rev_t rev = b->ptx_rev;
    /* fp8 has v1, v5, v7 only — fall back to v1 for other revs */
    if (dt == DT_FP8 && rev != PTX_V5 && rev != PTX_V7) rev = PTX_V1;
    if (build_kernel(b->kc, dt, rev, b->dev, b->verbose) != 0) return -1;
    CUfunction fn;
    if (rev == PTX_V7) {
        fn = (dt == DT_F16) ? b->kc->f_f16_v7 :
             (dt == DT_BF16) ? b->kc->f_bf16_v7 : b->kc->f_fp8_v7;
    } else if (rev == PTX_V6) {
        fn = (dt == DT_F16) ? b->kc->f_f16_v6 : b->kc->f_bf16_v6;
    } else if (rev == PTX_V5) {
        fn = (dt == DT_F16) ? b->kc->f_f16_v5 :
             (dt == DT_BF16) ? b->kc->f_bf16_v5 : b->kc->f_fp8_v5;
    } else if (rev == PTX_V4) {
        fn = (dt == DT_F16) ? b->kc->f_f16_v4 : b->kc->f_bf16_v4;
    } else if (rev == PTX_V3) {
        fn = (dt == DT_F16) ? b->kc->f_f16_v3 : b->kc->f_bf16_v3;
    } else if (rev == PTX_V2) {
        fn = (dt == DT_F16) ? b->kc->f_f16_v2 : b->kc->f_bf16_v2;
    } else {
        fn = (dt == DT_F16) ? b->kc->f_f16 :
             (dt == DT_BF16) ? b->kc->f_bf16 : b->kc->f_fp8;
    }
    int M = b->M, N = b->N, K = b->K;
    int gx, gy, block;
    size_t smem_bytes;
    if (rev == PTX_V7) {
        gx = (N + 127) / 128;
        gy = (M + 63) / 64;
        block = 256;
        smem_bytes = 2 * (64 * 32 + 128 * 32) * 2;  /* same SMEM as v5; 24 KiB */
    } else if (rev == PTX_V6) {
        gx = (N + 127) / 128;
        gy = (M + 63) / 64;
        block = 256;
        smem_bytes = 3 * (64 * 32 + 128 * 32) * 2;  /* 3 stages × 12 KiB = 36 KiB */
    } else if (rev == PTX_V5 || rev == PTX_V4 || rev == PTX_V3) {
        gx = (N + 127) / 128;
        gy = (M + 63) / 64;
        block = 256;
        smem_bytes = 2 * (64 * 32 + 128 * 32) * 2;  /* 2 stages × (sA+sB) halves = 24 KiB */
    } else if (rev == PTX_V2) {
        gx = (N + 127) / 128;
        gy = (M + 63) / 64;
        block = 256;
        smem_bytes = (64 * 32 + 128 * 32) * 2;
    } else {
        gx = (N + 255) / 256;
        gy = (M + 15) / 16;
        block = 128;
        smem_bytes = (dt == DT_FP8) ? 16 * 32 : 16 * 16 * 2;
    }
    void *args[] = { &b->d_Y, &b->d_X, &b->d_W, &M, &N, &K };

    /* Warmup */
    for (int i = 0; i < b->warmup; i++) {
        if (cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem_bytes, b->stream, args, NULL) != CUDA_SUCCESS) {
            fprintf(stderr, "  PTX: cuLaunchKernel failed\n"); return -1;
        }
    }
    cuStreamSynchronize(b->stream);

    CUevent start, stop;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop,  CU_EVENT_DEFAULT);
    cuEventRecord(start, b->stream);
    for (int i = 0; i < b->iters; i++) {
        cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem_bytes, b->stream, args, NULL);
    }
    cuEventRecord(stop, b->stream);
    cuEventSynchronize(stop);
    float ms;
    cuEventElapsedTime(&ms, start, stop);
    *avg_ms_out = ms / b->iters;
    cuEventDestroy(start);
    cuEventDestroy(stop);
    return 0;
}

static int run_cublas(bench_ctx_t *b, dtype_t dt, float *avg_ms_out) {
    int M = b->M, N = b->N, K = b->K;

    if (dt == DT_FP8) {
        return cublas_lt_fp8_gemm_timed(b->stream, b->d_W, b->d_X, b->d_Y,
                                        M, N, K, b->warmup, b->iters, avg_ms_out);
    }

    int (*fn)(cublasew_context *, CUdeviceptr, CUdeviceptr, CUdeviceptr, int, int, int) =
        (dt == DT_F16) ? cublasew_gemm_f16_f16_f32_rowmajor_nt
                       : cublasew_gemm_bf16_bf16_f32_rowmajor_nt;

    /* Warmup */
    for (int i = 0; i < b->warmup; i++) {
        if (fn(b->blas_ctx, b->d_Y, b->d_W, b->d_X, M, N, K) != 0) {
            fprintf(stderr, "  cuBLAS: gemm failed\n"); return -1;
        }
    }
    cuStreamSynchronize(b->stream);

    CUevent start, stop;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop,  CU_EVENT_DEFAULT);
    cuEventRecord(start, b->stream);
    for (int i = 0; i < b->iters; i++) {
        fn(b->blas_ctx, b->d_Y, b->d_W, b->d_X, M, N, K);
    }
    cuEventRecord(stop, b->stream);
    cuEventSynchronize(stop);
    float ms;
    cuEventElapsedTime(&ms, start, stop);
    *avg_ms_out = ms / b->iters;
    cuEventDestroy(start);
    cuEventDestroy(stop);
    return 0;
}

/* ---------------------------------------------------------------------- */
/* CUTLASS cutile mode (libcutile_gemm.so dlopen'd at runtime)            */
/* ---------------------------------------------------------------------- */

typedef int (*cutile_fn_t)(uintptr_t d_Y, uintptr_t d_W, uintptr_t d_X,
                           int M, int N, int K, void *stream);
static void *g_cutile_lib = NULL;
static cutile_fn_t p_cutile_f16, p_cutile_bf16, p_cutile_fp8;

static int cutile_load(void) {
    if (g_cutile_lib) return 0;
    const char *names[] = { "./libcutile_gemm.so", "libcutile_gemm.so", NULL };
    for (int i = 0; names[i]; i++) {
        g_cutile_lib = dlopen(names[i], RTLD_NOW);
        if (g_cutile_lib) break;
    }
    if (!g_cutile_lib) {
        fprintf(stderr, "cutile: libcutile_gemm.so not found (build with `make cutile`)\n");
        return -1;
    }
    p_cutile_f16  = (cutile_fn_t)dlsym(g_cutile_lib, "cutile_gemm_f16_f32");
    p_cutile_bf16 = (cutile_fn_t)dlsym(g_cutile_lib, "cutile_gemm_bf16_f32");
    p_cutile_fp8  = (cutile_fn_t)dlsym(g_cutile_lib, "cutile_gemm_fp8_e4m3_f32");
    if (!p_cutile_f16 || !p_cutile_bf16 || !p_cutile_fp8) {
        fprintf(stderr, "cutile: missing symbols in libcutile_gemm.so\n");
        return -1;
    }
    return 0;
}

static int run_cutile(bench_ctx_t *b, dtype_t dt, float *avg_ms_out) {
    if (cutile_load() != 0) return -1;
    cutile_fn_t fn = (dt == DT_F16)  ? p_cutile_f16
                   : (dt == DT_BF16) ? p_cutile_bf16
                                     : p_cutile_fp8;
    int M = b->M, N = b->N, K = b->K;

    for (int i = 0; i < b->warmup; i++) {
        if (fn(b->d_Y, b->d_W, b->d_X, M, N, K, b->stream) != 0) return -1;
    }
    cuStreamSynchronize(b->stream);

    CUevent start, stop;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop,  CU_EVENT_DEFAULT);
    cuEventRecord(start, b->stream);
    for (int i = 0; i < b->iters; i++) {
        fn(b->d_Y, b->d_W, b->d_X, M, N, K, b->stream);
    }
    cuEventRecord(stop, b->stream);
    cuEventSynchronize(stop);
    float ms;
    cuEventElapsedTime(&ms, start, stop);
    *avg_ms_out = ms / b->iters;
    cuEventDestroy(start);
    cuEventDestroy(stop);
    return 0;
}

/* Read GPU output and validate. For FP8 cuBLASLt, output is column-major
 * (N rows × M cols), so we transpose into a row-major buffer first. */
static acc_result_t read_and_validate(bench_ctx_t *b, dtype_t dt, bmode_t md, float *Y_host) {
    int M = b->M, N = b->N;
    cuMemcpyDtoH(Y_host, b->d_Y, (size_t)M * N * sizeof(float));

    if (dt == DT_FP8 && md == MD_CUBLAS) {
        /* Transpose first verify_rows × N from column-major[N,M] -> row-major[verify_rows, N] */
        float *tmp = (float *)malloc((size_t)b->verify_rows * N * sizeof(float));
        for (int m = 0; m < b->verify_rows; m++) {
            for (int n = 0; n < N; n++) tmp[(size_t)m * N + n] = Y_host[(size_t)n * M + m];
        }
        acc_result_t r = validate(tmp, b->Y_ref, M, N, b->verify_rows);
        free(tmp);
        return r;
    }
    return validate(Y_host, b->Y_ref, M, N, b->verify_rows);
}

/* ---------------------------------------------------------------------- */
/* Per-shape orchestration: alloc, fill, build CPU reference, run modes    */
/* ---------------------------------------------------------------------- */

static void print_row(dtype_t dt, bmode_t md, const char *shape_name,
                      int M, int N, int K, float ms, double tflops, double peak,
                      const acc_result_t *acc, int verify) {
    double pct = peak > 0 ? 100.0 * tflops / peak : 0.0;
    const char *status;
    if (verify) {
        if (acc->ok && pct >= 80.0) status = "PASS80 acc_ok";
        else if (acc->ok)           status = "FAIL80 acc_ok";
        else if (pct >= 80.0)       status = "PASS80 ACC_FAIL";
        else                        status = "FAIL80 ACC_FAIL";
    } else {
        status = (pct >= 80.0) ? "PASS80" : "FAIL80";
    }
    const char *mode_str = (md == MD_PTX) ? "ptx   "
                         : (md == MD_CUBLAS) ? "cublas"
                                             : "cutile";
    printf("dtype=%-4s mode=%s shape=%-9s M=%5d N=%5d K=%5d "
           "ms=%7.3f TFLOP/s=%6.2f peak=%5.1f%% %s",
           dtype_name(dt), mode_str, shape_name, M, N, K, ms, tflops, pct, status);
    if (verify) printf(" cos=%.5f max_err=%.4g", acc->cos_sim, acc->max_abs_err);
    printf("\n");
}

static int run_shape(dtype_t dt, bmode_t md, const shape_t *sh,
                     int warmup, int iters, int verify, int verify_rows, int verbose,
                     ptx_rev_t ptx_rev,
                     cublasew_context *blas_ctx, kernel_cache_t *kc,
                     CUdevice dev, CUstream stream) {
    int M = sh->M, N = sh->N, K = sh->K;
    int kal = dtype_k_alignment(dt);
    if (K % kal != 0) {
        fprintf(stderr, "skipping %s: K=%d not aligned to %d for dtype=%s\n",
                sh->name, K, kal, dtype_name(dt));
        return 0;
    }
    int dt_bytes = dtype_bytes(dt);

    /* Host F32 source (random) */
    float *X_f32 = (float *)malloc((size_t)M * K * sizeof(float));
    float *W_f32 = (float *)malloc((size_t)N * K * sizeof(float));
    if (!X_f32 || !W_f32) { fprintf(stderr, "OOM\n"); return -1; }
    for (size_t i = 0; i < (size_t)M * K; i++)
        X_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    /* Smaller W for FP8 (e4m3 range only ±448 but values up to ~1 keep us comfortably in range) */
    for (size_t i = 0; i < (size_t)N * K; i++)
        W_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    /* Quantize to dtype, then dequantize to f32 for golden CPU reference. */
    void *Xq = malloc((size_t)M * K * dt_bytes);
    void *Wq = malloc((size_t)N * K * dt_bytes);
    quantize_f32(X_f32, Xq, (size_t)M * K, dt);
    quantize_f32(W_f32, Wq, (size_t)N * K, dt);
    if (verify) {
        dequantize(Xq, X_f32, (size_t)M * K, dt);
        dequantize(Wq, W_f32, (size_t)N * K, dt);
    }

    /* CPU reference: only the first verify_rows rows of Y for tractability. */
    float *Y_ref = NULL;
    if (verify) {
        if (verify_rows > M) verify_rows = M;
        Y_ref = (float *)malloc((size_t)verify_rows * N * sizeof(float));
        if (verbose >= 1) fprintf(stderr, "  CPU ref: %d rows × N=%d × K=%d ...\n", verify_rows, N, K);
        cpu_gemm_f32(Y_ref, W_f32, NULL, X_f32, verify_rows, N, K, 0);
    }

    /* GPU buffers */
    CUdeviceptr d_X = 0, d_W = 0, d_Y = 0;
    cuMemAlloc(&d_X, (size_t)M * K * dt_bytes);
    cuMemAlloc(&d_W, (size_t)N * K * dt_bytes);
    cuMemAlloc(&d_Y, (size_t)M * N * sizeof(float));
    cuMemcpyHtoD(d_X, Xq, (size_t)M * K * dt_bytes);
    cuMemcpyHtoD(d_W, Wq, (size_t)N * K * dt_bytes);
    cuMemsetD8(d_Y, 0, (size_t)M * N * sizeof(float));

    bench_ctx_t b = { M, N, K, warmup, iters, verify, verify_rows, verbose,
                      ptx_rev, blas_ctx, kc, dev, stream,
                      X_f32, W_f32, Y_ref, d_X, d_W, d_Y, Xq, Wq };

    float *Y_host = (float *)malloc((size_t)M * N * sizeof(float));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double peak = dtype_peak_tflops(dt);

    if (md == MD_PTX || md == MD_ALL) {
        cuMemsetD8(d_Y, 0, (size_t)M * N * sizeof(float));
        float ms = 0;
        if (run_ptx(&b, dt, &ms) == 0) {
            double tflops = flops / (ms * 1e9);
            acc_result_t acc = {1, 0, 1.0};
            if (verify) acc = read_and_validate(&b, dt, MD_PTX, Y_host);
            print_row(dt, MD_PTX, sh->name, M, N, K, ms, tflops, peak, &acc, verify);
        }
    }
    if (md == MD_CUBLAS || md == MD_ALL) {
        cuMemsetD8(d_Y, 0, (size_t)M * N * sizeof(float));
        float ms = 0;
        if (run_cublas(&b, dt, &ms) == 0) {
            double tflops = flops / (ms * 1e9);
            acc_result_t acc = {1, 0, 1.0};
            if (verify) acc = read_and_validate(&b, dt, MD_CUBLAS, Y_host);
            print_row(dt, MD_CUBLAS, sh->name, M, N, K, ms, tflops, peak, &acc, verify);
        }
    }
    if (md == MD_CUTILE || md == MD_ALL) {
        cuMemsetD8(d_Y, 0, (size_t)M * N * sizeof(float));
        float ms = 0;
        if (run_cutile(&b, dt, &ms) == 0) {
            double tflops = flops / (ms * 1e9);
            acc_result_t acc = {1, 0, 1.0};
            if (verify) acc = read_and_validate(&b, dt, MD_CUTILE, Y_host);
            print_row(dt, MD_CUTILE, sh->name, M, N, K, ms, tflops, peak, &acc, verify);
        }
    }

    cuMemFree(d_X); cuMemFree(d_W); cuMemFree(d_Y);
    free(Y_host); free(Xq); free(Wq);
    free(X_f32); free(W_f32);
    if (Y_ref) free(Y_ref);
    return 0;
}

/* ---------------------------------------------------------------------- */
/* Argv + main                                                              */
/* ---------------------------------------------------------------------- */

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --dtype <f16|bf16|fp8>     dtype (default f16)\n");
    printf("  --mode  <ptx|cublas|all>   backend (default all)\n");
    printf("  --shape <name|all>         shape from table or 'all' (default all)\n");
    printf("  --m M --n N --k K          ad-hoc shape (overrides --shape)\n");
    printf("  --iters N                  bench iterations (default 100)\n");
    printf("  --warmup N                 warmup iterations (default 5)\n");
    printf("  --verify 0|1               run CPU validation (default 1)\n");
    printf("  --verify-rows N            CPU ref rows (default 64)\n");
    printf("  --verbose N                NVRTC verbosity (default 0)\n");
    printf("  --ptx-rev v1|v2|v3|v4|v5|v6 PTX kernel revision (default v5)\n");
    printf("  --list-shapes              list shape table and exit\n");
    printf("Available shapes:\n");
    for (int i = 0; i < N_SHAPES; i++)
        printf("  %-12s M=%5d N=%5d K=%5d\n", g_shapes[i].name,
               g_shapes[i].M, g_shapes[i].N, g_shapes[i].K);
}

int main(int argc, char **argv) {
    dtype_t dt = DT_F16;
    bmode_t md = MD_ALL;
    const char *shape_name = "all";
    int M = 0, N = 0, K = 0;
    int iters = 100, warmup = 5;
    int verify = 1, verify_rows = 64;
    int verbose = 0;
    ptx_rev_t ptx_rev = PTX_V7;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dtype") && i+1<argc) {
            const char *s = argv[++i];
            if (!strcmp(s, "f16")) dt = DT_F16;
            else if (!strcmp(s, "bf16")) dt = DT_BF16;
            else if (!strcmp(s, "fp8") || !strcmp(s, "fp8_e4m3")) dt = DT_FP8;
            else { fprintf(stderr, "unknown dtype %s\n", s); return 1; }
        } else if (!strcmp(argv[i], "--mode") && i+1<argc) {
            const char *s = argv[++i];
            if (!strcmp(s, "ptx")) md = MD_PTX;
            else if (!strcmp(s, "cublas")) md = MD_CUBLAS;
            else if (!strcmp(s, "cutile")) md = MD_CUTILE;
            else if (!strcmp(s, "all")) md = MD_ALL;
            else { fprintf(stderr, "unknown mode %s\n", s); return 1; }
        } else if (!strcmp(argv[i], "--shape") && i+1<argc) shape_name = argv[++i];
        else if (!strcmp(argv[i], "--m") && i+1<argc) M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n") && i+1<argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1<argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1<argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1<argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify") && i+1<argc) verify = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify-rows") && i+1<argc) verify_rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose") && i+1<argc) verbose = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ptx-rev") && i+1<argc) {
            const char *s = argv[++i];
            if (!strcmp(s, "v1")) ptx_rev = PTX_V1;
            else if (!strcmp(s, "v2")) ptx_rev = PTX_V2;
            else if (!strcmp(s, "v3")) ptx_rev = PTX_V3;
            else if (!strcmp(s, "v4")) ptx_rev = PTX_V4;
            else if (!strcmp(s, "v5")) ptx_rev = PTX_V5;
            else if (!strcmp(s, "v6")) ptx_rev = PTX_V6;
            else if (!strcmp(s, "v7")) ptx_rev = PTX_V7;
            else { fprintf(stderr, "unknown ptx-rev %s\n", s); return 1; }
        }
        else if (!strcmp(argv[i], "--list-shapes")) { print_usage(argv[0]); return 0; }
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }

    /* Init CUDA */
    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuewInit failed\n"); return 1;
    }
    if (cuInit(0) != CUDA_SUCCESS) { fprintf(stderr, "cuInit failed\n"); return 1; }
    CUdevice dev;
    CUcontext ctx;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) { fprintf(stderr, "cuDeviceGet failed\n"); return 1; }
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) { fprintf(stderr, "cuCtxCreate failed\n"); return 1; }

    char dev_name[256];
    int sm_major = 0, sm_minor = 0, sm_count = 0, clock_khz = 0;
    cuDeviceGetName(dev_name, sizeof(dev_name), dev);
    cuDeviceGetAttribute(&sm_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&sm_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    printf("Device: %s  sm_%d%d  SMs=%d  clock=%d MHz\n",
           dev_name, sm_major, sm_minor, sm_count, clock_khz / 1000);
    printf("Peak (assumed): f16/bf16=42 TFLOP/s   fp8=84 TFLOP/s\n");

    CUstream stream = NULL;
    cuStreamCreate(&stream, 0);

    /* cuBLAS */
    cublasew_context *blas_ctx = NULL;
    if (md == MD_CUBLAS || md == MD_ALL) {
        if (cublasewCreate(&blas_ctx, stream) != 0) {
            fprintf(stderr, "cublasewCreate failed; cuBLAS modes will be skipped.\n");
            blas_ctx = NULL;
        }
    }
    if ((md == MD_CUBLAS || md == MD_ALL) && dt == DT_FP8) {
        if (load_cublasLt() != 0) {
            fprintf(stderr, "cuBLASLt unavailable; FP8 cuBLAS mode will be skipped.\n");
        }
    }

    kernel_cache_t kc = {0};

    srand(42);

    if (M > 0 && N > 0 && K > 0) {
        shape_t adhoc = {"adhoc", M, N, K};
        run_shape(dt, md, &adhoc, warmup, iters, verify, verify_rows, verbose, ptx_rev, blas_ctx, &kc, dev, stream);
    } else if (!strcmp(shape_name, "all")) {
        for (int i = 0; i < N_SHAPES; i++) {
            run_shape(dt, md, &g_shapes[i], warmup, iters, verify, verify_rows, verbose, ptx_rev, blas_ctx, &kc, dev, stream);
        }
    } else {
        const shape_t *sh = NULL;
        for (int i = 0; i < N_SHAPES; i++) if (!strcmp(g_shapes[i].name, shape_name)) { sh = &g_shapes[i]; break; }
        if (!sh) { fprintf(stderr, "unknown shape %s\n", shape_name); return 1; }
        run_shape(dt, md, sh, warmup, iters, verify, verify_rows, verbose, ptx_rev, blas_ctx, &kc, dev, stream);
    }

    if (blas_ctx) cublasewDestroy(blas_ctx);
    if (kc.built_f16)     cuModuleUnload(kc.mod_f16);
    if (kc.built_bf16)    cuModuleUnload(kc.mod_bf16);
    if (kc.built_fp8)     cuModuleUnload(kc.mod_fp8);
    if (kc.built_f16_v2)  cuModuleUnload(kc.mod_f16_v2);
    if (kc.built_bf16_v2) cuModuleUnload(kc.mod_bf16_v2);
    if (kc.built_f16_v3)  cuModuleUnload(kc.mod_f16_v3);
    if (kc.built_bf16_v3) cuModuleUnload(kc.mod_bf16_v3);
    if (kc.built_f16_v4)  cuModuleUnload(kc.mod_f16_v4);
    if (kc.built_bf16_v4) cuModuleUnload(kc.mod_bf16_v4);
    if (kc.built_f16_v5)  cuModuleUnload(kc.mod_f16_v5);
    if (kc.built_bf16_v5) cuModuleUnload(kc.mod_bf16_v5);
    if (kc.built_f16_v6)  cuModuleUnload(kc.mod_f16_v6);
    if (kc.built_bf16_v6) cuModuleUnload(kc.mod_bf16_v6);
    cuStreamDestroy(stream);
    cuCtxDestroy(ctx);
    if (g_lt_lib) dlclose(g_lt_lib);
    return 0;
}
