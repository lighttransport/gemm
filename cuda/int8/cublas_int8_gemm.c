/*
 * cuBLAS INT8 GEMM Benchmark
 *
 * Uses cublasLtMatmul for INT8 matrix multiplication
 * Dynamically loads cuBLAS library (no CUDA SDK required at compile time)
 *
 * Computes: C[M,N] = A[M,K] * B[K,N]
 * - INT8 inputs (A, B)
 * - INT32 output (C)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>

#include "../cuew.h"

/* cuBLAS types and enums */
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;
typedef int cublasStatus_t;
typedef int cublasComputeType_t;
typedef int cudaDataType_t;
typedef int cublasLtOrder_t;

#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_COMPUTE_32I 72        /* INT32 compute for INT8 inputs */
#define CUDA_R_8I 3                  /* INT8 data type */
#define CUDA_R_32I 10                /* INT32 data type */
#define CUBLASLT_ORDER_COL 0         /* Column-major */
#define CUBLASLT_ORDER_ROW 1         /* Row-major */

/* cuBLAS function pointers */
typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t*);
typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t);
typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t);
typedef cublasStatus_t (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
typedef cublasStatus_t (*cublasLtMatrixLayoutCreate_t)(cublasLtMatrixLayout_t*, cudaDataType_t, uint64_t, uint64_t, int64_t);
typedef cublasStatus_t (*cublasLtMatrixLayoutDestroy_t)(cublasLtMatrixLayout_t);
typedef cublasStatus_t (*cublasLtMatmulPreferenceCreate_t)(cublasLtMatmulPreference_t*);
typedef cublasStatus_t (*cublasLtMatmulPreferenceDestroy_t)(cublasLtMatmulPreference_t);
typedef cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute_t)(cublasLtMatmulPreference_t, int, const void*, size_t);

typedef struct {
    int64_t algo;
    int tile[4];
    int stages;
    int customOption;
    int reserved[8];
} cublasLtMatmulAlgo_t;

typedef struct {
    cublasLtMatmulAlgo_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

typedef cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic_t)(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t,
    int, cublasLtMatmulHeuristicResult_t*, int*);

typedef cublasStatus_t (*cublasLtMatmul_t)(
    cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t,
    void*, cublasLtMatrixLayout_t,
    const cublasLtMatmulAlgo_t*, void*, size_t, CUstream);

/* Global function pointers */
static cublasLtCreate_t p_cublasLtCreate;
static cublasLtDestroy_t p_cublasLtDestroy;
static cublasLtMatmulDescCreate_t p_cublasLtMatmulDescCreate;
static cublasLtMatmulDescDestroy_t p_cublasLtMatmulDescDestroy;
static cublasLtMatrixLayoutCreate_t p_cublasLtMatrixLayoutCreate;
static cublasLtMatrixLayoutDestroy_t p_cublasLtMatrixLayoutDestroy;
static cublasLtMatmulPreferenceCreate_t p_cublasLtMatmulPreferenceCreate;
static cublasLtMatmulPreferenceDestroy_t p_cublasLtMatmulPreferenceDestroy;
static cublasLtMatmulPreferenceSetAttribute_t p_cublasLtMatmulPreferenceSetAttribute;
static cublasLtMatmulAlgoGetHeuristic_t p_cublasLtMatmulAlgoGetHeuristic;
static cublasLtMatmul_t p_cublasLtMatmul;

static void* cublas_lib = NULL;

static int load_cublas(void) {
    const char* lib_names[] = {
        "libcublasLt.so.13",  /* CUDA 13.x - required for SM 12.0 (Blackwell) */
        "libcublasLt.so.12",
        "libcublasLt.so.11",
        "libcublasLt.so",
        NULL
    };

    for (int i = 0; lib_names[i]; i++) {
        cublas_lib = dlopen(lib_names[i], RTLD_NOW);
        if (cublas_lib) {
            printf("Loaded cuBLAS: %s\n", lib_names[i]);
            break;
        }
    }

    if (!cublas_lib) {
        fprintf(stderr, "Failed to load cuBLAS library\n");
        return -1;
    }

    #define LOAD_FUNC(name) \
        p_##name = (name##_t)dlsym(cublas_lib, #name); \
        if (!p_##name) { \
            fprintf(stderr, "Failed to load %s\n", #name); \
            return -1; \
        }

    LOAD_FUNC(cublasLtCreate);
    LOAD_FUNC(cublasLtDestroy);
    LOAD_FUNC(cublasLtMatmulDescCreate);
    LOAD_FUNC(cublasLtMatmulDescDestroy);
    LOAD_FUNC(cublasLtMatrixLayoutCreate);
    LOAD_FUNC(cublasLtMatrixLayoutDestroy);
    LOAD_FUNC(cublasLtMatmulPreferenceCreate);
    LOAD_FUNC(cublasLtMatmulPreferenceDestroy);
    LOAD_FUNC(cublasLtMatmulPreferenceSetAttribute);
    LOAD_FUNC(cublasLtMatmulAlgoGetHeuristic);
    LOAD_FUNC(cublasLtMatmul);

    #undef LOAD_FUNC
    return 0;
}

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

/* Reference implementation for verification */
static void gemm_reference_s8(const int8_t* A, const int8_t* B, int32_t* C,
                               int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t sum = 0;
            for (int k = 0; k < K; k++) {
                sum += (int32_t)A[m * K + k] * (int32_t)B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -m M          Matrix M dimension (default: 4096)\n");
    printf("  -n N          Matrix N dimension (default: 4096)\n");
    printf("  -k K          Matrix K dimension (default: 4096)\n");
    printf("  --no-verify   Skip CPU verification\n");
    printf("  -h, --help    Show this help\n");
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    int skip_verify = 0;
    int warmup_iters = 5;
    int bench_iters = 100;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-verify") == 0) {
            skip_verify = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    printf("cuBLAS INT8 GEMM: C[%d,%d] = A[%d,%d] x B[%d,%d]\n\n", M, N, M, K, K, N);

    /* Initialize CUDA */
    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }

    CHECK_CUDA(cuInit(0));

    CUdevice device;
    CUcontext context;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    char device_name[256];
    CHECK_CUDA(cuDeviceGetName(device_name, sizeof(device_name), device));

    int major, minor, sm_count, clock_khz;
    CHECK_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    CHECK_CUDA(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    CHECK_CUDA(cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));

    printf("=== GPU Specifications ===\n");
    printf("Device: %s\n", device_name);
    printf("Compute Capability: SM %d.%d\n", major, minor);
    printf("SM Count: %d\n", sm_count);
    printf("GPU Clock: %d MHz\n\n", clock_khz / 1000);

    /* Load cuBLAS */
    if (load_cublas() != 0) {
        fprintf(stderr, "Failed to load cuBLAS\n");
        return 1;
    }

    /* Create cuBLAS handle */
    cublasLtHandle_t handle;
    CHECK_CUBLAS(p_cublasLtCreate(&handle));

    /* Allocate host memory */
    size_t size_A = (size_t)M * K * sizeof(int8_t);
    size_t size_B = (size_t)K * N * sizeof(int8_t);
    size_t size_C = (size_t)M * N * sizeof(int32_t);

    int8_t* h_A = (int8_t*)malloc(size_A);
    int8_t* h_B = (int8_t*)malloc(size_B);
    int32_t* h_C = (int32_t*)malloc(size_C);
    int32_t* h_C_ref = skip_verify ? NULL : (int32_t*)malloc(size_C);

    /* Initialize data */
    srand(42);
    for (size_t i = 0; i < (size_t)M * K; i++) {
        h_A[i] = (int8_t)(rand() % 7 - 3);  /* -3 to 3 */
    }
    for (size_t i = 0; i < (size_t)K * N; i++) {
        h_B[i] = (int8_t)(rand() % 7 - 3);
    }

    /* Compute reference */
    if (!skip_verify) {
        printf("Computing CPU reference...\n");
        gemm_reference_s8(h_A, h_B, h_C_ref, M, N, K);
    }

    /* Allocate device memory */
    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size_A));
    CHECK_CUDA(cuMemAlloc(&d_B, size_B));
    CHECK_CUDA(cuMemAlloc(&d_C, size_C));

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size_A));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size_B));
    CHECK_CUDA(cuMemsetD8(d_C, 0, size_C));

    /* Create matmul descriptor */
    cublasLtMatmulDesc_t matmul_desc;
    CHECK_CUBLAS(p_cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    /* Create matrix layouts (row-major: A[M,K], B[K,N], C[M,N]) */
    /* cuBLAS expects column-major, so we compute C^T = B^T * A^T */
    /* Which is equivalent to row-major C = A * B */
    cublasLtMatrixLayout_t layout_A, layout_B, layout_C;

    /* For row-major with cuBLAS: swap A and B, swap M and N */
    /* C[M,N] = A[M,K] * B[K,N] in row-major */
    /* = (B^T * A^T)^T in column-major terms */
    /* cuBLAS does: C = op(A) * op(B), with A[M,K], B[K,N] */
    /* For row-major, we pass: B'[N,K], A'[K,M], C'[N,M] as column-major */

    /* Layout: (datatype, rows, cols, leading_dim) in column-major view */
    /* A in row-major [M,K] -> seen as column-major [K,M] with ld=K */
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_A, CUDA_R_8I, K, M, K));
    /* B in row-major [K,N] -> seen as column-major [N,K] with ld=N */
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_B, CUDA_R_8I, N, K, N));
    /* C in row-major [M,N] -> seen as column-major [N,M] with ld=N */
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_32I, N, M, N));

    /* Create preference */
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(p_cublasLtMatmulPreferenceCreate(&preference));

    /* Set workspace size preference */
    size_t workspace_size = 32 * 1024 * 1024;  /* 32 MB */
    CUdeviceptr d_workspace;
    CHECK_CUDA(cuMemAlloc(&d_workspace, workspace_size));

    /* CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1 */
    CHECK_CUBLAS(p_cublasLtMatmulPreferenceSetAttribute(
        preference, 1, &workspace_size, sizeof(workspace_size)));

    /* Get best algorithm */
    cublasLtMatmulHeuristicResult_t heuristic;
    int num_results = 0;

    /* Note: For row-major C = A * B, we compute column-major C' = B' * A' */
    /* So we swap the order: layout_B (as A'), layout_A (as B'), layout_C */
    cublasStatus_t status = p_cublasLtMatmulAlgoGetHeuristic(
        handle, matmul_desc,
        layout_B, layout_A,  /* Swapped for row-major */
        layout_C, layout_C,
        preference,
        1, &heuristic, &num_results);

    if (status != CUBLAS_STATUS_SUCCESS || num_results == 0) {
        fprintf(stderr, "No suitable algorithm found (status=%d, results=%d)\n", status, num_results);
        fprintf(stderr, "INT8 GEMM may not be supported on this GPU configuration\n");
        return 1;
    }

    printf("Found cuBLAS algorithm (workspace: %zu bytes)\n\n", heuristic.workspaceSize);

    /* Scaling factors */
    int32_t alpha = 1;
    int32_t beta = 0;

    /* Create events for timing */
    CUevent start, stop;
    CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUBLAS(p_cublasLtMatmul(
            handle, matmul_desc,
            &alpha,
            (void*)d_B, layout_B,  /* Swapped */
            (void*)d_A, layout_A,  /* Swapped */
            &beta,
            (void*)d_C, layout_C,
            (void*)d_C, layout_C,
            &heuristic.algo,
            (void*)d_workspace, workspace_size,
            0));
    }
    CHECK_CUDA(cuCtxSynchronize());

    /* Benchmark */
    CHECK_CUDA(cuEventRecord(start, 0));
    for (int i = 0; i < bench_iters; i++) {
        CHECK_CUBLAS(p_cublasLtMatmul(
            handle, matmul_desc,
            &alpha,
            (void*)d_B, layout_B,
            (void*)d_A, layout_A,
            &beta,
            (void*)d_C, layout_C,
            (void*)d_C, layout_C,
            &heuristic.algo,
            (void*)d_workspace, workspace_size,
            0));
    }
    CHECK_CUDA(cuEventRecord(stop, 0));
    CHECK_CUDA(cuEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cuEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / bench_iters;

    /* Copy results back */
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size_C));

    /* Verify */
    if (!skip_verify) {
        int errors = 0;
        for (int i = 0; i < M * N && errors < 10; i++) {
            if (h_C[i] != h_C_ref[i]) {
                printf("Mismatch at %d: got %d, expected %d\n", i, h_C[i], h_C_ref[i]);
                errors++;
            }
        }
        if (errors == 0) {
            printf("Verification PASSED\n");
        } else {
            printf("Verification FAILED (%d errors shown)\n", errors);
        }
    } else {
        printf("Verification SKIPPED\n");
    }

    /* Calculate performance */
    double ops = 2.0 * M * N * K;
    double tops = ops / (avg_ms * 1e9);

    /* Estimate peak (rough) */
    int ops_per_sm = 8192;  /* Approximate for SM 12.0 */
    double peak_tops = (double)sm_count * ops_per_sm * (clock_khz / 1000) / 1e6;

    printf("\n=== Performance ===\n");
    printf("Time: %.3f ms (avg of %d runs)\n", avg_ms, bench_iters);
    printf("Throughput: %.2f TOPS\n", tops);
    printf("Estimated Peak: %.2f TOPS\n", peak_tops);
    printf("Efficiency: %.1f%%\n", 100.0 * tops / peak_tops);

    /* Cleanup */
    CHECK_CUDA(cuEventDestroy(start));
    CHECK_CUDA(cuEventDestroy(stop));
    CHECK_CUDA(cuMemFree(d_A));
    CHECK_CUDA(cuMemFree(d_B));
    CHECK_CUDA(cuMemFree(d_C));
    CHECK_CUDA(cuMemFree(d_workspace));

    p_cublasLtMatrixLayoutDestroy(layout_A);
    p_cublasLtMatrixLayoutDestroy(layout_B);
    p_cublasLtMatrixLayoutDestroy(layout_C);
    p_cublasLtMatmulDescDestroy(matmul_desc);
    p_cublasLtMatmulPreferenceDestroy(preference);
    p_cublasLtDestroy(handle);

    if (cublas_lib) dlclose(cublas_lib);

    CHECK_CUDA(cuCtxDestroy(context));

    free(h_A);
    free(h_B);
    free(h_C);
    if (h_C_ref) free(h_C_ref);

    return 0;
}
