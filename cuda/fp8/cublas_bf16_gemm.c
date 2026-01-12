/*
 * cuBLAS BF16 GEMM Benchmark
 * Based on working INT8 pattern
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

#include "cuew.h"

/* CUDA data types - from library_types.h */
typedef enum {
    CUDA_R_32F  = 0,
    CUDA_R_16F  = 2,
    CUDA_R_16BF = 14,
} cudaDataType_t;

/* cuBLAS compute types - from cublas_api.h */
typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_16BF = 75,
} cublasComputeType_t;

typedef enum {
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
} cublasLtMatmulPreferenceAttributes_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
} cublasOperation_t;

typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
} cublasStatus_t;

/* Opaque types */
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;

typedef struct {
    void* algo;
    size_t workspaceSize;
    cublasStatus_t status;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

/* Function pointer types */
typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t*);
typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t);
typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t*, cublasComputeType_t, cudaDataType_t);
typedef cublasStatus_t (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
typedef cublasStatus_t (*cublasLtMatrixLayoutCreate_t)(cublasLtMatrixLayout_t*, cudaDataType_t, uint64_t, uint64_t, int64_t);
typedef cublasStatus_t (*cublasLtMatrixLayoutDestroy_t)(cublasLtMatrixLayout_t);
typedef cublasStatus_t (*cublasLtMatmulPreferenceCreate_t)(cublasLtMatmulPreference_t*);
typedef cublasStatus_t (*cublasLtMatmulPreferenceDestroy_t)(cublasLtMatmulPreference_t);
typedef cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute_t)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void*, size_t);
typedef cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t*, int*);
typedef cublasStatus_t (*cublasLtMatmul_t)(cublasLtHandle_t, cublasLtMatmulDesc_t,
    const void*, const void*, cublasLtMatrixLayout_t, const void*, cublasLtMatrixLayout_t,
    const void*, const void*, cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
    const void*, void*, size_t, CUstream);

/* Function pointers */
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
        "libcublasLt.so.13",
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
        if (!p_##name) { fprintf(stderr, "Failed to load %s\n", #name); return -1; }

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
        const char* errStr; cuGetErrorString(err, &errStr); \
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

/* BF16 conversion */
static uint16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float bf16_to_float(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* CPU reference */
static void gemm_reference_bf16(const uint16_t* A, const uint16_t* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += bf16_to_float(A[i * K + k]) * bf16_to_float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

/* GPU specs */
typedef struct {
    int major, minor, sm_count, clock_mhz;
    char name[256];
} gpu_specs_t;

static gpu_specs_t get_gpu_specs(CUdevice dev) {
    gpu_specs_t s;
    cuDeviceGetAttribute(&s.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&s.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&s.sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&s.clock_mhz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    s.clock_mhz /= 1000;
    cuDeviceGetName(s.name, sizeof(s.name), dev);
    return s;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  -m M            Matrix M dimension (default: 4096)\n");
    printf("  -n N            Matrix N dimension (default: 4096)\n");
    printf("  -k K            Matrix K dimension (default: 4096)\n");
    printf("  --skip-verify   Skip CPU verification\n");
    printf("  -h, --help      Show help\n");
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    int skip_verify = 0;
    int warmup_iters = 10, bench_iters = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "--skip-verify") == 0) skip_verify = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    printf("cuBLAS BF16 GEMM: C[%d,%d] = A[%d,%d] x B[%d,%d]\n", M, N, M, K, K, N);

    /* Initialize CUDA */
    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 1;
    }
    CHECK_CUDA(cuInit(0));

    CUdevice dev;
    CUcontext ctx;
    CHECK_CUDA(cuDeviceGet(&dev, 0));
    CHECK_CUDA(cuCtxCreate(&ctx, 0, dev));

    gpu_specs_t specs = get_gpu_specs(dev);
    printf("\n=== GPU Specifications ===\n");
    printf("Device: %s\n", specs.name);
    printf("Compute Capability: SM %d.%d\n", specs.major, specs.minor);
    printf("SM Count: %d\n", specs.sm_count);
    printf("GPU Clock: %d MHz\n", specs.clock_mhz);

    /* BF16 tensor core ops per SM per cycle */
    int ops_per_sm = (specs.major >= 12) ? 2048 : (specs.major >= 9) ? 1024 : 512;
    double peak_tops = (double)specs.sm_count * specs.clock_mhz * ops_per_sm * 2.0 / 1e6;
    printf("Theoretical BF16 Peak: %.2f TOPS\n", peak_tops);

    if (load_cublas() != 0) return 1;

    cublasLtHandle_t handle;
    CHECK_CUBLAS(p_cublasLtCreate(&handle));

    /* Allocate host memory */
    size_t size_A = (size_t)M * K * sizeof(uint16_t);
    size_t size_B = (size_t)K * N * sizeof(uint16_t);
    size_t size_C = (size_t)M * N * sizeof(uint16_t);

    uint16_t* h_A = (uint16_t*)malloc(size_A);
    uint16_t* h_B = (uint16_t*)malloc(size_B);
    uint16_t* h_C = (uint16_t*)malloc(size_C);
    float* h_C_ref = skip_verify ? NULL : (float*)malloc((size_t)M * N * sizeof(float));

    /* Initialize random BF16 */
    srand(42);
    for (int i = 0; i < M * K; i++) {
        h_A[i] = float_to_bf16(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = float_to_bf16(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
    }

    if (!skip_verify) {
        printf("Computing CPU reference...\n");
        gemm_reference_bf16(h_A, h_B, h_C_ref, M, N, K);
    }

    /* Allocate device memory */
    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size_A));
    CHECK_CUDA(cuMemAlloc(&d_B, size_B));
    CHECK_CUDA(cuMemAlloc(&d_C, size_C));
    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size_A));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size_B));
    CHECK_CUDA(cuMemsetD8(d_C, 0, size_C));

    /* Create matmul descriptor - BF16 with FP32 compute */
    cublasLtMatmulDesc_t matmul_desc;
    CHECK_CUBLAS(p_cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    /* Matrix layouts - same pattern as working INT8 */
    /* Row-major A[M,K], B[K,N], C[M,N] -> column-major view */
    cublasLtMatrixLayout_t layout_A, layout_B, layout_C;
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_A, CUDA_R_16BF, K, M, K));
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_B, CUDA_R_16BF, N, K, N));
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_16BF, N, M, N));

    /* Preference */
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(p_cublasLtMatmulPreferenceCreate(&preference));

    size_t workspace_size = 32 * 1024 * 1024;
    CUdeviceptr d_workspace;
    CHECK_CUDA(cuMemAlloc(&d_workspace, workspace_size));
    CHECK_CUBLAS(p_cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    /* Get algorithm - swap A/B for row-major */
    cublasLtMatmulHeuristicResult_t heuristic;
    int num_results = 0;
    cublasStatus_t status = p_cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc,
        layout_B, layout_A, layout_C, layout_C,
        preference, 1, &heuristic, &num_results);

    if (status != CUBLAS_STATUS_SUCCESS || num_results == 0) {
        fprintf(stderr, "No algorithm found (status=%d, count=%d)\n", status, num_results);
        return 1;
    }
    printf("Found cuBLAS algorithm (workspace: %zu bytes)\n", heuristic.workspaceSize);

    float alpha = 1.0f, beta = 0.0f;

    /* Timing */
    CUevent start, stop;
    CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUBLAS(p_cublasLtMatmul(handle, matmul_desc, &alpha,
            (void*)d_B, layout_B, (void*)d_A, layout_A, &beta,
            (void*)d_C, layout_C, (void*)d_C, layout_C,
            &heuristic.algo, (void*)d_workspace, workspace_size, 0));
    }
    CHECK_CUDA(cuCtxSynchronize());

    /* Benchmark */
    CHECK_CUDA(cuEventRecord(start, 0));
    for (int i = 0; i < bench_iters; i++) {
        CHECK_CUBLAS(p_cublasLtMatmul(handle, matmul_desc, &alpha,
            (void*)d_B, layout_B, (void*)d_A, layout_A, &beta,
            (void*)d_C, layout_C, (void*)d_C, layout_C,
            &heuristic.algo, (void*)d_workspace, workspace_size, 0));
    }
    CHECK_CUDA(cuEventRecord(stop, 0));
    CHECK_CUDA(cuEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cuEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / bench_iters;

    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size_C));

    /* Verify */
    if (!skip_verify) {
        int errors = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < M && errors < 10; i++) {
            for (int j = 0; j < N && errors < 10; j++) {
                float got = bf16_to_float(h_C[j * M + i]);  /* column-major */
                float expected = h_C_ref[i * N + j];
                float diff = fabsf(got - expected);
                if (diff > max_diff) max_diff = diff;
                float rel = (fabsf(expected) > 1e-6f) ? diff / fabsf(expected) : diff;
                if (rel > 0.05f && diff > 0.1f) {
                    if (errors < 5) printf("Mismatch [%d,%d]: %.4f vs %.4f\n", i, j, got, expected);
                    errors++;
                }
            }
        }
        printf("%s (max_diff=%.4f)\n", errors ? "FAILED" : "\nVerification PASSED", max_diff);
    }

    /* Results */
    double flops = 2.0 * M * N * K;
    double tops = (flops / (avg_ms / 1000.0)) / 1e12;

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
    p_cublasLtMatmulPreferenceDestroy(preference);
    p_cublasLtMatrixLayoutDestroy(layout_A);
    p_cublasLtMatrixLayoutDestroy(layout_B);
    p_cublasLtMatrixLayoutDestroy(layout_C);
    p_cublasLtMatmulDescDestroy(matmul_desc);
    p_cublasLtDestroy(handle);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    if (cublas_lib) dlclose(cublas_lib);
    cuCtxDestroy(ctx);

    return 0;
}
