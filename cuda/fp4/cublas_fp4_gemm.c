/*
 * cuBLASLt FP4 GEMM Benchmark  (Blackwell sm_120, e.g. RTX 5060 Ti)
 *
 * Revisits the question: is FP4 (e2m1) GEMM hardware-accelerated on consumer
 * Blackwell, and does NVFP4 differ from MXFP4?
 *
 *   NVFP4 = e2m1 data + 16-element block, UE4M3 scale  (CUBLASLT_..._VEC16_UE4M3)
 *   MXFP4 = e2m1 data + 32-element block, UE8M0 scale  (CUBLASLT_..._VEC32_UE8M0)
 *
 * Both go through the 5th-gen tensor-core block-scaled matmul on Blackwell.
 * cuBLASLt FP4 needs the *block-scaled* API (the per-tensor scale path that
 * works for FP8 on Hopper is unsupported on sm_120) -- see cublas_fp8_gemm.c.
 *
 * Strategy to keep this robust: every input element is the e2m1/e4m3/bf16
 * encoding of 1.0, and every block scale is the encoding of 1.0, so the
 * expected output is exactly K at every position -- a trivial correctness check
 * that is independent of the (swizzled) scale-factor layout.  We only need the
 * scale buffer to be at least as large as cuBLAS expects, so we over-allocate
 * and fill the whole thing with the 1.0 byte.
 *
 * Dynamically loads cuBLASLt (no CUDA SDK at compile time), like the FP8 bench.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

#include "cuew.h"

/* ---- cudaDataType_t (library_types.h) ---- */
typedef enum {
    CUDA_R_32F      = 0,
    CUDA_R_16F      = 2,
    CUDA_R_16BF     = 14,
    CUDA_R_8F_E4M3  = 28,   /* also UE4M3 (scale type for NVFP4) */
    CUDA_R_8F_E5M2  = 29,
    CUDA_R_8F_UE8M0 = 30,   /* scale type for MXFP4 */
    CUDA_R_4F_E2M1  = 33,   /* FP4 data */
} cudaDataType_t;

typedef enum {
    CUBLAS_COMPUTE_16F            = 64,
    CUBLAS_COMPUTE_32F            = 68,
    CUBLAS_COMPUTE_32F_FAST_TF32  = 77,
} cublasComputeType_t;

/* block-scale modes (cublasLt.h cublasLtMatmulMatrixScale_t) */
typedef enum {
    CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F   = 0,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3  = 1,  /* NVFP4 */
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0  = 2,  /* MXFP4 */
} cublasLtMatmulMatrixScale_t;

typedef enum {
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
} cublasLtMatmulPreferenceAttributes_t;

typedef enum {
    CUBLASLT_MATMUL_DESC_TRANSA          = 0,
    CUBLASLT_MATMUL_DESC_TRANSB          = 1,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE    = 31,
    CUBLASLT_MATMUL_DESC_B_SCALE_MODE    = 32,
} cublasLtMatmulDescAttributes_t;

typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 } cublasOperation_t;

typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;

typedef struct {
    uint64_t algo[8];        /* real cublasLtMatmulAlgo_t is uint64_t[8] (64 B), not a pointer */
    size_t   workspaceSize;
    int32_t  status;
    float    wavesCount;
    int      reserved[4];
} cublasLtMatmulHeuristicResult_t;

typedef int (*cublasLtCreate_t)(cublasLtHandle_t*);
typedef int (*cublasLtDestroy_t)(cublasLtHandle_t);
typedef int (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t);
typedef int (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
typedef int (*cublasLtMatmulDescSetAttribute_t)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void*, size_t);
typedef int (*cublasLtMatrixLayoutCreate_t)(cublasLtMatrixLayout_t*, cudaDataType_t, uint64_t, uint64_t, int64_t);
typedef int (*cublasLtMatrixLayoutDestroy_t)(cublasLtMatrixLayout_t);
typedef int (*cublasLtMatmulPreferenceCreate_t)(cublasLtMatmulPreference_t*);
typedef int (*cublasLtMatmulPreferenceDestroy_t)(cublasLtMatmulPreference_t);
typedef int (*cublasLtMatmulPreferenceSetAttribute_t)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void*, size_t);
typedef int (*cublasLtMatmulAlgoGetHeuristic_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t*, int*);
typedef int (*cublasLtMatmul_t)(cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
    const void*, void*, size_t, CUstream);

static cublasLtCreate_t                      p_cublasLtCreate;
static cublasLtDestroy_t                     p_cublasLtDestroy;
static cublasLtMatmulDescCreate_t            p_cublasLtMatmulDescCreate;
static cublasLtMatmulDescDestroy_t           p_cublasLtMatmulDescDestroy;
static cublasLtMatmulDescSetAttribute_t      p_cublasLtMatmulDescSetAttribute;
static cublasLtMatrixLayoutCreate_t          p_cublasLtMatrixLayoutCreate;
static cublasLtMatrixLayoutDestroy_t         p_cublasLtMatrixLayoutDestroy;
static cublasLtMatmulPreferenceCreate_t      p_cublasLtMatmulPreferenceCreate;
static cublasLtMatmulPreferenceDestroy_t     p_cublasLtMatmulPreferenceDestroy;
static cublasLtMatmulPreferenceSetAttribute_t p_cublasLtMatmulPreferenceSetAttribute;
static cublasLtMatmulAlgoGetHeuristic_t      p_cublasLtMatmulAlgoGetHeuristic;
static cublasLtMatmul_t                      p_cublasLtMatmul;

static void* cublas_lib = NULL;

static int load_cublas(void) {
    const char* names[] = { "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so", NULL };
    for (int i = 0; names[i]; i++) {
        cublas_lib = dlopen(names[i], RTLD_NOW);
        if (cublas_lib) { printf("Loaded cuBLAS: %s\n", names[i]); break; }
    }
    if (!cublas_lib) { fprintf(stderr, "Failed to load cuBLASLt\n"); return -1; }
    #define LOAD_FUNC(n) p_##n = (n##_t)dlsym(cublas_lib, #n); \
        if (!p_##n) { fprintf(stderr, "missing %s\n", #n); return -1; }
    LOAD_FUNC(cublasLtCreate); LOAD_FUNC(cublasLtDestroy);
    LOAD_FUNC(cublasLtMatmulDescCreate); LOAD_FUNC(cublasLtMatmulDescDestroy);
    LOAD_FUNC(cublasLtMatmulDescSetAttribute);
    LOAD_FUNC(cublasLtMatrixLayoutCreate); LOAD_FUNC(cublasLtMatrixLayoutDestroy);
    LOAD_FUNC(cublasLtMatmulPreferenceCreate); LOAD_FUNC(cublasLtMatmulPreferenceDestroy);
    LOAD_FUNC(cublasLtMatmulPreferenceSetAttribute);
    LOAD_FUNC(cublasLtMatmulAlgoGetHeuristic); LOAD_FUNC(cublasLtMatmul);
    #undef LOAD_FUNC
    return 0;
}

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { const char* s; cuGetErrorString(err, &s); \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, s); exit(1); } } while(0)

static float bf16_to_float(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16; float f; memcpy(&f, &bits, sizeof(f)); return f;
}

static int roundup(int x, int m) { return ((x + m - 1) / m) * m; }

/* ---- one benchmark configuration ---- */
typedef enum { SC_NONE = -1, SC_SCALAR = 0, SC_NVFP4 = 1, SC_MXFP4 = 2 } scale_kind_t;

typedef struct {
    const char*  label;
    cudaDataType_t ab_type;
    scale_kind_t scale_kind;   /* SC_NONE bf16, SC_SCALAR per-tensor fp8, SC_NVFP4/SC_MXFP4 block */
} bench_config_t;

typedef struct {
    const char* label;
    int   supported;
    int   verified;          /* 1 ok, 0 failed, -1 skipped */
    float max_diff;
    double avg_ms;
    double tflops;
} bench_result_t;

/* bytes per element *2 (to avoid fractional bytes for fp4) */
static size_t elem_bytes_x2(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_16BF:    return 4;  /* 2 bytes */
        case CUDA_R_8F_E4M3: return 2;  /* 1 byte  */
        case CUDA_R_4F_E2M1: return 1;  /* 0.5 byte */
        default:             return 4;
    }
}

/* fill a device buffer of `count` elements of type t with the encoding of 1.0 */
static void fill_ones(CUdeviceptr d, size_t count, cudaDataType_t t) {
    switch (t) {
        case CUDA_R_16BF:    CHECK_CUDA(cuMemsetD16(d, 0x3F80, count)); break;          /* bf16 1.0 */
        case CUDA_R_8F_E4M3: CHECK_CUDA(cuMemsetD8 (d, 0x38, count));   break;          /* e4m3 1.0 */
        case CUDA_R_4F_E2M1: CHECK_CUDA(cuMemsetD8 (d, 0x22, (count + 1) / 2)); break;  /* two e2m1 1.0 per byte */
        default: break;
    }
}

static int run_config(cublasLtHandle_t handle, const bench_config_t* cfg,
                      int M, int N, int K, int warmup, int iters, int skip_verify,
                      bench_result_t* res) {
    res->label = cfg->label;
    res->supported = 0; res->verified = -1; res->max_diff = 0; res->avg_ms = 0; res->tflops = 0;

    /* device buffers (TN layout: A[K,M] ld K, B[K,N] ld K, C[M,N] ld M) */
    size_t bytes_A = (size_t)M * K * elem_bytes_x2(cfg->ab_type) / 2;
    size_t bytes_B = (size_t)K * N * elem_bytes_x2(cfg->ab_type) / 2;
    size_t bytes_C = (size_t)M * N * sizeof(uint16_t);   /* bf16 output */
    CUdeviceptr d_A = 0, d_B = 0, d_C = 0, d_workspace = 0;
    CUdeviceptr d_scaleA = 0, d_scaleB = 0;          /* block scales (uint8) */
    CUdeviceptr d_fscaleA = 0, d_fscaleB = 0;        /* f32 per-tensor scales */
    CHECK_CUDA(cuMemAlloc(&d_A, bytes_A));
    CHECK_CUDA(cuMemAlloc(&d_B, bytes_B));
    CHECK_CUDA(cuMemAlloc(&d_C, bytes_C));
    fill_ones(d_A, (size_t)M * K, cfg->ab_type);
    fill_ones(d_B, (size_t)K * N, cfg->ab_type);
    CHECK_CUDA(cuMemsetD8(d_C, 0, bytes_C));

    /* Narrow precision (FP8/FP4) needs the TN layout (K-major A and B) plus
     * block scales.  Plain BF16 has no such constraint and is run through the
     * row-major-swap recipe that is known to work on this card, so the table
     * always has a real baseline next to the narrow rows. */
    int narrow = (cfg->scale_kind != SC_NONE);

    /* descriptor */
    cublasLtMatmulDesc_t desc;
    if (p_cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != 0) goto unsupported;
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    if (narrow) {   /* TN; bf16 NN-swap path leaves the defaults (N,N) like the standalone bench */
        p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    }

    if (cfg->scale_kind == SC_SCALAR) {
        float one = 1.0f;
        CHECK_CUDA(cuMemAlloc(&d_fscaleA, sizeof(float)));
        CHECK_CUDA(cuMemAlloc(&d_fscaleB, sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_fscaleA, &one, sizeof(float)));
        CHECK_CUDA(cuMemcpyHtoD(d_fscaleB, &one, sizeof(float)));
        p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_fscaleA, sizeof(d_fscaleA));
        p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_fscaleB, sizeof(d_fscaleB));
    } else if (cfg->scale_kind == SC_NVFP4 || cfg->scale_kind == SC_MXFP4) {
        int blk      = (cfg->scale_kind == SC_NVFP4) ? 16 : 32;
        uint8_t one  = (cfg->scale_kind == SC_NVFP4) ? 0x38 /* ue4m3 1.0 */ : 0x7F /* ue8m0 1.0 */;
        int mode     = (int)cfg->scale_kind;
        int Kb       = K / blk;
        /* cuBLAS expects a padded, swizzled scale tensor (rows padded to 128,
         * scale-cols padded to 4).  We fill uniformly with 1.0 and over-allocate
         * 2x + slack so the layout details don't matter. */
        size_t scaleA_bytes = (size_t)roundup(M, 128) * roundup(Kb, 4) * 2 + 4096;
        size_t scaleB_bytes = (size_t)roundup(N, 128) * roundup(Kb, 4) * 2 + 4096;
        CHECK_CUDA(cuMemAlloc(&d_scaleA, scaleA_bytes));
        CHECK_CUDA(cuMemAlloc(&d_scaleB, scaleB_bytes));
        CHECK_CUDA(cuMemsetD8(d_scaleA, one, scaleA_bytes));
        CHECK_CUDA(cuMemsetD8(d_scaleB, one, scaleB_bytes));
        p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
        p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
        p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA));
        p_cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB));
    }

    /* layouts + matmul operand order
     *   narrow (TN): D[M,N] = A[M,K]*B[K,N]; A,B K-major; call(A,B).
     *   bf16  (NN-swap): pass (B,A) with B as [N,K], A as [K,M], D as [N,M]. */
    cublasLtMatrixLayout_t lA, lB, lC;
    CUdeviceptr p1, p2; cublasLtMatrixLayout_t l1, l2;
    if (narrow) {
        p_cublasLtMatrixLayoutCreate(&lA, cfg->ab_type, K, M, K);
        p_cublasLtMatrixLayoutCreate(&lB, cfg->ab_type, K, N, K);
        p_cublasLtMatrixLayoutCreate(&lC, CUDA_R_16BF, M, N, M);
        p1 = d_A; l1 = lA; p2 = d_B; l2 = lB;
    } else {
        p_cublasLtMatrixLayoutCreate(&lA, cfg->ab_type, K, M, K);   /* A[M,K] row-major */
        p_cublasLtMatrixLayoutCreate(&lB, cfg->ab_type, N, K, N);   /* B[K,N] row-major */
        p_cublasLtMatrixLayoutCreate(&lC, CUDA_R_16BF, N, M, N);    /* D[M,N] row-major */
        p1 = d_B; l1 = lB; p2 = d_A; l2 = lA;
    }

    cublasLtMatmulPreference_t pref;
    p_cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_size = 64u * 1024 * 1024;
    CHECK_CUDA(cuMemAlloc(&d_workspace, ws_size));
    p_cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));

    cublasLtMatmulHeuristicResult_t heur;
    int n_algo = 0;
    int hstatus = p_cublasLtMatmulAlgoGetHeuristic(handle, desc, l1, l2, lC, lC, pref, 1, &heur, &n_algo);
    if (hstatus != 0 || n_algo == 0) {
        printf("  [%-6s] NO ALGORITHM (heuristic status=%d, count=%d)\n", cfg->label, hstatus, n_algo);
        p_cublasLtMatmulPreferenceDestroy(pref);
        p_cublasLtMatrixLayoutDestroy(lA); p_cublasLtMatrixLayoutDestroy(lB); p_cublasLtMatrixLayoutDestroy(lC);
        p_cublasLtMatmulDescDestroy(desc);
        goto cleanup_buffers;
    }
    res->supported = 1;

    float alpha = 1.0f, beta = 0.0f;
    CUevent ev0, ev1;
    CHECK_CUDA(cuEventCreate(&ev0, 0)); CHECK_CUDA(cuEventCreate(&ev1, 0));

    int matmul_err = 0;
    for (int i = 0; i < warmup; i++) {
        int e = p_cublasLtMatmul(handle, desc, &alpha, (void*)p1, l1, (void*)p2, l2,
                                 &beta, (void*)d_C, lC, (void*)d_C, lC,
                                 &heur.algo, (void*)d_workspace, ws_size, 0);
        if (e != 0) { matmul_err = e; break; }
    }
    if (matmul_err) {
        printf("  [%-6s] matmul FAILED at warmup (status=%d)\n", cfg->label, matmul_err);
        res->supported = 0;
        cuEventDestroy(ev0); cuEventDestroy(ev1);
        p_cublasLtMatmulPreferenceDestroy(pref);
        p_cublasLtMatrixLayoutDestroy(lA); p_cublasLtMatrixLayoutDestroy(lB); p_cublasLtMatrixLayoutDestroy(lC);
        p_cublasLtMatmulDescDestroy(desc);
        goto cleanup_buffers;
    }
    CHECK_CUDA(cuCtxSynchronize());

    CHECK_CUDA(cuEventRecord(ev0, 0));
    for (int i = 0; i < iters; i++) {
        p_cublasLtMatmul(handle, desc, &alpha, (void*)p1, l1, (void*)p2, l2,
                         &beta, (void*)d_C, lC, (void*)d_C, lC,
                         &heur.algo, (void*)d_workspace, ws_size, 0);
    }
    CHECK_CUDA(cuEventRecord(ev1, 0));
    CHECK_CUDA(cuEventSynchronize(ev1));
    float ms = 0; CHECK_CUDA(cuEventElapsedTime(&ms, ev0, ev1));
    res->avg_ms = ms / iters;
    res->tflops = (2.0 * M * N * K / (res->avg_ms / 1000.0)) / 1e12;

    /* verify: all inputs and scales are 1.0 -> every output element == K */
    if (!skip_verify) {
        uint16_t* h_C = (uint16_t*)malloc(bytes_C);
        CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, bytes_C));
        float expected = (float)K, maxd = 0;
        size_t ne = (size_t)M * N, nbad = 0;
        for (size_t i = 0; i < ne; i++) {
            float got = bf16_to_float(h_C[i]);
            float d = fabsf(got - expected);
            if (d > maxd) maxd = d;
            if (d > 0.05f * expected + 1.0f) nbad++;
        }
        res->max_diff = maxd;
        res->verified = (nbad == 0) ? 1 : 0;
        if (nbad) printf("  [%-6s] verify FAILED: %zu/%zu bad, max_diff=%.3f (expected %.1f)\n",
                         cfg->label, nbad, ne, maxd, expected);
        free(h_C);
    }

    printf("  [%-6s] %.3f ms   %.1f TFLOP/s%s\n", cfg->label, res->avg_ms, res->tflops,
           (res->verified == 1) ? "   (verified)" : (res->verified == 0 ? "   (VERIFY FAIL)" : ""));

    cuEventDestroy(ev0); cuEventDestroy(ev1);
    p_cublasLtMatmulPreferenceDestroy(pref);
    p_cublasLtMatrixLayoutDestroy(lA); p_cublasLtMatrixLayoutDestroy(lB); p_cublasLtMatrixLayoutDestroy(lC);
    p_cublasLtMatmulDescDestroy(desc);

cleanup_buffers:
    if (d_scaleA)   cuMemFree(d_scaleA);
    if (d_scaleB)   cuMemFree(d_scaleB);
    if (d_fscaleA)  cuMemFree(d_fscaleA);
    if (d_fscaleB)  cuMemFree(d_fscaleB);
    if (d_workspace) cuMemFree(d_workspace);
    cuMemFree(d_A); cuMemFree(d_B); cuMemFree(d_C);
    return res->supported;

unsupported:
    cuMemFree(d_A); cuMemFree(d_B); cuMemFree(d_C);
    return 0;
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    int skip_verify = 0, warmup = 10, iters = 100;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skip-verify")) skip_verify = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [-m M] [-n N] [-k K] [--iters N] [--skip-verify]\n", argv[0]);
            return 0;
        }
    }
    /* FP4 needs K divisible by 32 (MXFP4 block) and even (2 e2m1/byte) */
    if (K % 32 != 0) { int k2 = roundup(K, 32); printf("note: rounding K %d -> %d (mult of 32)\n", K, k2); K = k2; }

    printf("cuBLASLt FP4 GEMM benchmark   C[%d,%d] = A[%d,%d] x B[%d,%d]\n", M, N, M, K, K, N);

    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) { fprintf(stderr, "cuewInit failed\n"); return 1; }
    CHECK_CUDA(cuInit(0));
    CUdevice dev; CUcontext ctx;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    int major = 0, minor = 0, smcount = 0, clk = 0;
    char name[256] = {0};
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&smcount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&clk, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    cuDeviceGetName(name, sizeof(name), dev);
    printf("Device: %s  (SM %d.%d, %d SMs, %d MHz)\n\n", name, major, minor, smcount, clk / 1000);

    if (load_cublas() != 0) return 1;
    cublasLtHandle_t handle;
    if (p_cublasLtCreate(&handle) != 0) { fprintf(stderr, "cublasLtCreate failed\n"); return 1; }

    bench_config_t cfgs[] = {
        { "BF16",  CUDA_R_16BF,    SC_NONE   },
        { "FP8",   CUDA_R_8F_E4M3, SC_SCALAR },   /* per-tensor; likely unsupported on sm_120 */
        { "NVFP4", CUDA_R_4F_E2M1, SC_NVFP4  },   /* block 16, ue4m3 */
        { "MXFP4", CUDA_R_4F_E2M1, SC_MXFP4  },   /* block 32, ue8m0 */
    };
    int ncfg = (int)(sizeof(cfgs) / sizeof(cfgs[0]));
    bench_result_t results[8];

    printf("Running (warmup=%d, iters=%d, %s):\n", warmup, iters, skip_verify ? "no verify" : "verify");
    for (int i = 0; i < ncfg; i++)
        run_config(handle, &cfgs[i], M, N, K, warmup, iters, skip_verify, &results[i]);

    /* summary table */
    double bf16_tflops = 0;
    int any_narrow = 0;
    for (int i = 0; i < ncfg; i++) {
        if (results[i].supported && !strcmp(results[i].label, "BF16")) bf16_tflops = results[i].tflops;
        if (results[i].supported && strcmp(results[i].label, "BF16")) any_narrow = 1;
    }

    printf("\n=== Summary (M=N=K=%d) ===\n", M);
    printf("%-7s %-12s %10s %12s %10s\n", "format", "status", "ms", "TFLOP/s", "vs BF16");
    for (int i = 0; i < ncfg; i++) {
        bench_result_t* r = &results[i];
        if (!r->supported) {
            printf("%-7s %-12s %10s %12s %10s\n", r->label, "unsupported", "-", "-", "-");
        } else {
            char ratio[16] = "-";
            if (bf16_tflops > 0) snprintf(ratio, sizeof(ratio), "%.2fx", r->tflops / bf16_tflops);
            printf("%-7s %-12s %10.3f %12.1f %10s\n",
                   r->label, (r->verified == 0) ? "VERIFY-FAIL" : "ok",
                   r->avg_ms, r->tflops, ratio);
        }
    }
    printf("\nInterpretation:\n");
    if (any_narrow) {
        printf("  FP4 >> BF16 (~2-4x) => FP4 is HW-accelerated via cuBLAS on this SM.\n");
        printf("  NVFP4 ~ MXFP4 => both use the same 5th-gen tensor-core datapath.\n");
    } else {
        printf("  cuBLASLt exposes NO narrow-precision (FP8/FP4) GEMM on SM %d.%d\n", major, minor);
        printf("  (heuristic status 15 = NOT_SUPPORTED, even with block scaling + the\n");
        printf("  exact layout from NVIDIA's LtNvfp4Matmul sample).  This matches the\n");
        printf("  known 'cuBLAS FP8 unsupported on sm_120' result -- the cuBLAS narrow\n");
        printf("  GEMM kernels are gated to datacenter Blackwell (sm_100).\n");
        printf("  => cuBLAS not exposing FP4 does NOT mean the silicon lacks the datapath.\n");
        printf("     Run ./mma_fp4_probe to test the raw mma.sync e2m1 tensor-core path.\n");
    }

    p_cublasLtDestroy(handle);
    if (cublas_lib) dlclose(cublas_lib);
    cuCtxDestroy(ctx);
    return 0;
}
