/*
 * cuBLAS FP8 GEMM Benchmark
 *
 * Uses cublasLt for FP8 (E4M3/E5M2) matrix multiplication on Blackwell GPUs.
 * Dynamically loads cuBLAS library - no CUDA SDK required at compile time.
 *
 * FP8 E4M3: 1 sign + 4 exponent + 3 mantissa (range: ±448, precision: ~0.125)
 * FP8 E5M2: 1 sign + 5 exponent + 2 mantissa (range: ±57344, precision: ~0.25)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <time.h>

#include "cuew.h"
#include "fp8_types.h"

/* cuBLAS data types */
typedef enum {
    CUDA_R_32F = 0,
    CUDA_R_16F = 2,
    CUDA_R_16BF = 14,   /* bfloat16 - required for FP8 output */
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
} cudaDataType_t;

typedef enum {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
} cublasComputeType_t;

typedef enum {
    CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
} cublasLtMatmulPreferenceAttributes_t;

typedef enum {
    CUBLASLT_MATMUL_DESC_TRANSA = 3,
    CUBLASLT_MATMUL_DESC_TRANSB = 4,
    CUBLASLT_MATMUL_DESC_EPILOGUE = 7,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER = 19,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER = 20,
} cublasLtMatmulDescAttributes_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
} cublasOperation_t;

typedef enum {
    CUBLASLT_EPILOGUE_DEFAULT = 1,
} cublasLtEpilogue_t;

/* Opaque types */
typedef void* cublasLtHandle_t;
typedef void* cublasLtMatmulDesc_t;
typedef void* cublasLtMatrixLayout_t;
typedef void* cublasLtMatmulPreference_t;

typedef struct {
    struct { uint64_t data[8]; } algo;
    size_t workspaceSize;
    int32_t status;
    float wavesCount;
    int reserved[4];
} cublasLtMatmulHeuristicResult_t;

/* Function pointer types */
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

/* Function pointers */
static cublasLtCreate_t p_cublasLtCreate;
static cublasLtDestroy_t p_cublasLtDestroy;
static cublasLtMatmulDescCreate_t p_cublasLtMatmulDescCreate;
static cublasLtMatmulDescDestroy_t p_cublasLtMatmulDescDestroy;
static cublasLtMatmulDescSetAttribute_t p_cublasLtMatmulDescSetAttribute;
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
    LOAD_FUNC(cublasLtMatmulDescSetAttribute);
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
    int err = (call); \
    if (err != 0) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

/* BF16 to float conversion */
static float bf16_to_float(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* CPU reference implementation */
static void gemm_reference_fp8(const uint8_t* A, const uint8_t* B, float* C,
                               int M, int N, int K, int e5m2) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = e5m2 ? fp8_e5m2_to_float(A[i * K + k]) : fp8_e4m3_to_float(A[i * K + k]);
                float b = e5m2 ? fp8_e5m2_to_float(B[j * K + k]) : fp8_e4m3_to_float(B[j * K + k]);
                sum += a * b;
            }
            C[i * N + j] = sum;
        }
    }
}

/* Get GPU specs */
typedef struct {
    int major, minor;
    int sm_count;
    int clock_mhz;
    int mem_bandwidth_gbps;
    char name[256];
} gpu_specs_t;

static gpu_specs_t get_gpu_specs(CUdevice dev) {
    gpu_specs_t specs;
    cuDeviceGetAttribute(&specs.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&specs.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&specs.sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&specs.clock_mhz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    specs.clock_mhz /= 1000;

    int mem_clock, mem_width;
    cuDeviceGetAttribute(&mem_clock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
    cuDeviceGetAttribute(&mem_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    specs.mem_bandwidth_gbps = (mem_clock / 1000) * (mem_width / 8) * 2 / 1000;

    cuDeviceGetName(specs.name, sizeof(specs.name), dev);
    return specs;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -m M          Matrix M dimension (default: 4096)\n");
    printf("  -n N          Matrix N dimension (default: 4096)\n");
    printf("  -k K          Matrix K dimension (default: 4096)\n");
    printf("  --e4m3        Use E4M3 format (default)\n");
    printf("  --e5m2        Use E5M2 format\n");
    printf("  --skip-verify Skip CPU verification\n");
    printf("  -h, --help    Show this help\n");
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    int use_e5m2 = 0;
    int skip_verify = 0;
    int warmup_iters = 10;
    int bench_iters = 100;

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--e4m3") == 0) {
            use_e5m2 = 0;
        } else if (strcmp(argv[i], "--e5m2") == 0) {
            use_e5m2 = 1;
        } else if (strcmp(argv[i], "--skip-verify") == 0) {
            skip_verify = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (M <= 0 || N <= 0 || K <= 0 || M > 16384 || N > 16384 || K > 16384) {
        fprintf(stderr, "Dimensions must be in 1..16384\n"); return 1;
    }

    const char* format_name = use_e5m2 ? "E5M2" : "E4M3";
    printf("cuBLAS FP8 %s GEMM: C[%d,%d] = A[%d,%d] x B[%d,%d]^T\n",
           format_name, M, N, M, K, N, K);

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

    /* Load cuBLAS */
    if (load_cublas() != 0) {
        return 1;
    }

    /* Create cuBLAS handle */
    cublasLtHandle_t handle;
    CHECK_CUBLAS(p_cublasLtCreate(&handle));

    /* Allocate host memory - BF16 output (2 bytes per element) */
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N * sizeof(uint16_t);  /* BF16 = 2 bytes */

    uint8_t* h_A = (uint8_t*)malloc(size_A);
    uint8_t* h_B = (uint8_t*)malloc(size_B);
    uint16_t* h_C = (uint16_t*)malloc(size_C);  /* BF16 output */
    float* h_C_ref = skip_verify ? NULL : (float*)malloc((size_t)M * N * sizeof(float));

    /* Initialize with random FP8 values */
    srand(42);
    for (size_t i = 0; i < size_A; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_A[i] = use_e5m2 ? float_to_fp8_e5m2(val) : float_to_fp8_e4m3(val);
    }
    for (size_t i = 0; i < size_B; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        h_B[i] = use_e5m2 ? float_to_fp8_e5m2(val) : float_to_fp8_e4m3(val);
    }

    /* Compute reference */
    if (!skip_verify) {
        printf("Computing CPU reference...\n");
        gemm_reference_fp8(h_A, h_B, h_C_ref, M, N, K, use_e5m2);
    }

    /* Allocate device memory */
    CUdeviceptr d_A, d_B, d_C;
    CHECK_CUDA(cuMemAlloc(&d_A, size_A));
    CHECK_CUDA(cuMemAlloc(&d_B, size_B));
    CHECK_CUDA(cuMemAlloc(&d_C, size_C));

    CHECK_CUDA(cuMemcpyHtoD(d_A, h_A, size_A));
    CHECK_CUDA(cuMemcpyHtoD(d_B, h_B, size_B));
    CHECK_CUDA(cuMemsetD8(d_C, 0, size_C));

    /* Allocate scale factors (required for FP8) */
    float h_scale_A = 1.0f;
    float h_scale_B = 1.0f;
    float h_scale_D = 1.0f;
    CUdeviceptr d_scale_A, d_scale_B, d_scale_D;
    CHECK_CUDA(cuMemAlloc(&d_scale_A, sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_scale_B, sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_scale_D, sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_scale_A, &h_scale_A, sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_scale_B, &h_scale_B, sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_scale_D, &h_scale_D, sizeof(float)));

    /* FP8 inputs, FP32 accumulation and scale type. */
    cublasLtMatmulDesc_t matmul_desc;
    CHECK_CUBLAS(p_cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    /* FP8 data type */
    cudaDataType_t fp8_type = use_e5m2 ? CUDA_R_8F_E5M2 : CUDA_R_8F_E4M3;

    /* C_rm[M,N] = X_rm[M,K] * W_rm[N,K]^T. In column-major:
     * C_cm[N,M] = W_cm[K,N]^T * X_cm[K,M]. */
    cublasOperation_t opT = CUBLAS_OP_T;
    cublasOperation_t opN = CUBLAS_OP_N;
    CHECK_CUBLAS(p_cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CHECK_CUBLAS(p_cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    /* Set scale pointers (required for FP8) */
    CHECK_CUBLAS(p_cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scale_A, sizeof(d_scale_A)));
    CHECK_CUBLAS(p_cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scale_B, sizeof(d_scale_B)));

    cublasLtMatrixLayout_t layout_A, layout_B, layout_C;

    /* A operand: W row-major [N,K] -> column-major [K,N], op=T. */
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_A, fp8_type, K, N, K));
    /* B operand: X row-major [M,K] -> column-major [K,M], op=N. */
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_B, fp8_type, K, M, K));
    /* C/D: row-major [M,N] -> column-major [N,M], BF16 output. */
    CHECK_CUBLAS(p_cublasLtMatrixLayoutCreate(&layout_C, CUDA_R_16BF, N, M, N));

    /* Create preference */
    cublasLtMatmulPreference_t preference;
    CHECK_CUBLAS(p_cublasLtMatmulPreferenceCreate(&preference));

    /* Set workspace size preference */
    size_t workspace_size = 32 * 1024 * 1024;  /* 32 MB */
    CUdeviceptr d_workspace;
    CHECK_CUDA(cuMemAlloc(&d_workspace, workspace_size));

    CHECK_CUBLAS(p_cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    /* Get algorithm */
    cublasLtMatmulHeuristicResult_t heuristic;
    int algo_count = 0;
    int status = p_cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc,
        layout_A, layout_B, layout_C, layout_C,
        preference, 1, &heuristic, &algo_count);

    if (status != 0 || algo_count == 0) {
        fprintf(stderr, "\n========================================\n");
        fprintf(stderr, "No FP8 %s algorithm for %dx%dx%d on SM %d.%d\n",
                format_name, M, N, K, specs.major, specs.minor);
        fprintf(stderr, "========================================\n\n");
        fprintf(stderr, "cuBLAS status: %d, algorithms found: %d\n\n", status, algo_count);
        fprintf(stderr, "No algorithm for this shape, format pair, and output type.\n");

        /* Cleanup and exit */
        p_cublasLtMatmulPreferenceDestroy(preference);
        p_cublasLtMatrixLayoutDestroy(layout_A);
        p_cublasLtMatrixLayoutDestroy(layout_B);
        p_cublasLtMatrixLayoutDestroy(layout_C);
        p_cublasLtMatmulDescDestroy(matmul_desc);
        p_cublasLtDestroy(handle);
        return 1;
    }

    printf("Found cuBLAS FP8 algorithm (workspace: %zu bytes)\n", heuristic.workspaceSize);

    /* Scaling factors for FP8 GEMM */
    float alpha = 1.0f;
    float beta = 0.0f;

    /* Create events for timing */
    CUevent start, stop;
    CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    /* Warmup */
    for (int i = 0; i < warmup_iters; i++) {
        CHECK_CUBLAS(p_cublasLtMatmul(
            handle, matmul_desc,
            &alpha,
            (void*)d_B, layout_A,
            (void*)d_A, layout_B,
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
            (void*)d_B, layout_A,
            (void*)d_A, layout_B,
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

    /* Copy result back */
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, size_C));

    int verified = 1;
    /* Verify against decoded FP8 inputs, allowing only BF16 output rounding. */
    if (!skip_verify) {
        int errors = 0;
        float max_diff = 0.0f;
        for (int i = 0; i < M && errors < 10; i++) {
            for (int j = 0; j < N && errors < 10; j++) {
                float got = bf16_to_float(h_C[i * N + j]);
                float expected = h_C_ref[i * N + j];
                float diff = fabsf(got - expected);
                float ref_abs = fabsf(expected);
                float rel_diff = (ref_abs > 1e-6f) ? diff / ref_abs : diff;
                if (diff > max_diff) max_diff = diff;
                /* FP8 has lower precision, use larger tolerance */
                if (!isfinite(got) || (rel_diff > 0.004f && diff > 1e-4f)) {
                    if (errors < 5) {
                        printf("Mismatch at [%d,%d]: got %.4f, expected %.4f (diff=%.4f)\n",
                               i, j, got, expected, diff);
                    }
                    errors++;
                }
            }
        }
        if (errors > 0) {
            verified = 0;
            printf("Verification FAILED (%d errors, max_diff=%.4f)\n", errors, max_diff);
        } else {
            printf("\nVerification PASSED (max_diff=%.4f)\n", max_diff);
        }
    }

    /* Performance metrics */
    double flops = 2.0 * M * N * K;
    double tops = (flops / (avg_ms / 1000.0)) / 1e12;

    printf("\n=== Performance ===\n");
    printf("Time: %.3f ms (avg of %d runs)\n", avg_ms, bench_iters);
    printf("Throughput: %.2f TOPS\n", tops);

    /* Cleanup */
    CHECK_CUDA(cuEventDestroy(start));
    CHECK_CUDA(cuEventDestroy(stop));
    CHECK_CUDA(cuMemFree(d_A));
    CHECK_CUDA(cuMemFree(d_B));
    CHECK_CUDA(cuMemFree(d_C));
    CHECK_CUDA(cuMemFree(d_workspace));
    CHECK_CUDA(cuMemFree(d_scale_A));
    CHECK_CUDA(cuMemFree(d_scale_B));
    CHECK_CUDA(cuMemFree(d_scale_D));

    p_cublasLtMatmulPreferenceDestroy(preference);
    p_cublasLtMatrixLayoutDestroy(layout_A);
    p_cublasLtMatrixLayoutDestroy(layout_B);
    p_cublasLtMatrixLayoutDestroy(layout_C);
    p_cublasLtMatmulDescDestroy(matmul_desc);
    p_cublasLtDestroy(handle);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    if (cublas_lib) dlclose(cublas_lib);
    cuCtxDestroy(ctx);

    return verified ? 0 : 1;
}
