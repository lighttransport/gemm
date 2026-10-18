/*
 * FP8 GEMM using CUDA with cuew - Tensor Core Version
 *
 * Computes: C[M,N] = A[M,K] * B[K,N]
 * - FP8 inputs (E4M3 or E5M2)
 * - FP32 accumulator
 * - Uses Tensor Core mma.sync instructions (SM 8.9+)
 *
 * Uses PTX kernel loaded via cuModuleLoadData (no CUDA SDK required)
 * Includes CUDA-event timing and CPU-reference verification
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../cuew.h"
#include "fp8_types.h"

/* Default matrix dimensions - must be multiples of tile sizes */
static int g_M = 128;
static int g_N = 128;
static int g_K = 128;

/* Benchmark settings */
static int g_warmup_iters = 5;
static int g_bench_iters = 100;
static int g_benchmark_mode = 0;
static int g_use_tensor_core = 1;  /* Use tensor cores by default */
static int g_tile_size = 64;      /* Tile size: 32, 64, or 128 */

#define CHECK_CUDA(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

/* GPU specifications */
typedef struct {
    int sm_count;
    int clock_mhz;
    int major;
    int minor;
    size_t mem_bandwidth_gbps;
} gpu_specs_t;

static void query_gpu_specs(CUdevice device, gpu_specs_t* specs) {
    int clock_khz, mem_clock_khz, mem_bus_width;
    CHECK_CUDA(cuDeviceGetAttribute(&specs->sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    CHECK_CUDA(cuDeviceGetAttribute(&clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));
    CHECK_CUDA(cuDeviceGetAttribute(&specs->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    CHECK_CUDA(cuDeviceGetAttribute(&specs->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    CHECK_CUDA(cuDeviceGetAttribute(&mem_clock_khz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
    CHECK_CUDA(cuDeviceGetAttribute(&mem_bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));

    specs->clock_mhz = clock_khz / 1000;
    specs->mem_bandwidth_gbps = (size_t)mem_clock_khz * 2 * mem_bus_width / 8 / 1000000;

}

static void print_gpu_specs(const char* device_name, const gpu_specs_t* specs) {
    printf("\n=== GPU Specifications ===\n");
    printf("Device: %s\n", device_name);
    printf("Compute Capability: SM %d.%d\n", specs->major, specs->minor);
    printf("SM Count: %d\n", specs->sm_count);
    printf("GPU Clock: %d MHz\n", specs->clock_mhz);
    printf("Memory Bandwidth: %zu GB/s\n", specs->mem_bandwidth_gbps);
    printf("\n");
}

/*
 * Tensor Core PTX Kernel for FP8 GEMM (E4M3)
 * Uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 * Block: 128 threads (4 warps), 32x32 output per block
 * Tiles: A=32x32 (row-major), B=32x32 (transposed/col-major in smem)
 */
#include "fp8_gemm_kernels.h"

static void gemm_reference_e4m3(const fp8_e4m3_t* A, const fp8_e4m3_t* B, float* C,
                                 int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                acc += fp8_e4m3_to_float(A[i * k + kk]) * fp8_e4m3_to_float(B[kk * n + j]);
            }
            C[i * n + j] = acc;
        }
    }
}

static void gemm_reference_e5m2(const fp8_e5m2_t* A, const fp8_e5m2_t* B, float* C,
                                 int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                acc += fp8_e5m2_to_float(A[i * k + kk]) * fp8_e5m2_to_float(B[kk * n + j]);
            }
            C[i * n + j] = acc;
        }
    }
}

typedef struct {
    double elapsed_ms;
    double gflops;
    int verified;
} benchmark_result_t;

static benchmark_result_t run_gemm_benchmark(
    const char* ptx_code, const char* kernel_name,
    const uint8_t* A, const uint8_t* B, float* C_ref,
    int m, int n, int k,
    int use_tc, /* use tensor core kernel */
    const char* format_name)
{
    benchmark_result_t result = {0};
    CUmodule module;
    CUfunction kernel_func;
    CUdeviceptr d_A, d_B, d_C;
    CUevent start, stop;
    float* h_C;

    printf("\n=== %s %s (%s) ===\n",
           g_benchmark_mode ? "Benchmarking" : "Testing",
           format_name,
           use_tc ? "Tensor Core" : "Naive");
    printf("Matrix size: %d x %d x %d\n", m, n, k);

    /* Load PTX */
    char error_log[4096] = {0};
    CUjit_option options[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void* option_values[] = { (void*)error_log, (void*)(size_t)sizeof(error_log) };

    CUresult res = cuModuleLoadDataEx(&module, ptx_code, 2, options, option_values);
    if (res != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "Failed to load PTX: %s\n", errStr);
        if (error_log[0]) fprintf(stderr, "JIT Error: %s\n", error_log);
        return result;
    }

    CHECK_CUDA(cuModuleGetFunction(&kernel_func, module, kernel_name));

    /* Allocate device memory */
    CHECK_CUDA(cuMemAlloc(&d_A, (size_t)m * k));
    CHECK_CUDA(cuMemAlloc(&d_B, (size_t)k * n));
    CHECK_CUDA(cuMemAlloc(&d_C, (size_t)m * n * sizeof(float)));
    CHECK_CUDA(cuMemcpyHtoD(d_A, A, (size_t)m * k));
    CHECK_CUDA(cuMemcpyHtoD(d_B, B, (size_t)k * n));
    CHECK_CUDA(cuMemsetD8(d_C, 0, (size_t)m * n * sizeof(float)));

    CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    unsigned int m_val = m, n_val = n, k_val = k;
    void* args[] = { &d_A, &d_B, &d_C, &m_val, &n_val, &k_val };

    unsigned int blockDimX, blockDimY, gridDimX, gridDimY;
    if (use_tc) {
        /* Tensor core block/grid configuration based on tile size */
        if (g_tile_size == 128) {
            /* 512 threads (16 warps), 128x128 output per block */
            blockDimX = 512; blockDimY = 1;
            gridDimX = (n + 127) / 128;
            gridDimY = (m + 127) / 128;
        } else if (g_tile_size == 64) {
            /* 512 threads (16 warps), 64x64 output per block */
            blockDimX = 512; blockDimY = 1;
            gridDimX = (n + 63) / 64;
            gridDimY = (m + 63) / 64;
        } else {
            /* 128 threads (4 warps), 32x32 output per block */
            blockDimX = 128; blockDimY = 1;
            gridDimX = (n + 31) / 32;
            gridDimY = (m + 31) / 32;
        }
    } else {
        /* Naive: 16x16 threads/block */
        blockDimX = 16; blockDimY = 16;
        gridDimX = (n + 15) / 16;
        gridDimY = (m + 15) / 16;
    }

    /* Warmup */
    for (int i = 0; i < g_warmup_iters; i++) {
        CHECK_CUDA(cuLaunchKernel(kernel_func, gridDimX, gridDimY, 1,
                                  blockDimX, blockDimY, 1, 0, NULL, args, NULL));
    }
    CHECK_CUDA(cuCtxSynchronize());

    /* Benchmark */
    CHECK_CUDA(cuEventRecord(start, NULL));
    for (int i = 0; i < g_bench_iters; i++) {
        CHECK_CUDA(cuLaunchKernel(kernel_func, gridDimX, gridDimY, 1,
                                  blockDimX, blockDimY, 1, 0, NULL, args, NULL));
    }
    CHECK_CUDA(cuEventRecord(stop, NULL));
    CHECK_CUDA(cuEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cuEventElapsedTime(&elapsed_ms, start, stop));
    result.elapsed_ms = elapsed_ms / g_bench_iters;

    double flops = 2.0 * m * n * k;
    result.gflops = flops / (result.elapsed_ms * 1e6);

    /* Verify */
    h_C = (float*)malloc((size_t)m * n * sizeof(float));
    CHECK_CUDA(cuMemcpyDtoH(h_C, d_C, (size_t)m * n * sizeof(float)));

    /* Compare arithmetic on decoded inputs, excluding FP8 quantization error. */
    float tolerance = 1e-5f * fmaxf(1.0f, sqrtf((float)k));
    float max_error = 0.0f;
    int errors = 0;
    for (int i = 0; i < m * n; i++) {
        float diff = fabsf(h_C[i] - C_ref[i]);
        if (diff > max_error) max_error = diff;
        if (diff > tolerance) errors++;
    }
    result.verified = (errors == 0);

    printf("Time: %.4f ms, Throughput: %.2f GFLOPS\n", result.elapsed_ms, result.gflops);
    printf("Verification: max_error=%.6f, %s\n", max_error, result.verified ? "PASSED" : "FAILED");

    free(h_C);
    CHECK_CUDA(cuEventDestroy(start));
    CHECK_CUDA(cuEventDestroy(stop));
    CHECK_CUDA(cuMemFree(d_A));
    CHECK_CUDA(cuMemFree(d_B));
    CHECK_CUDA(cuMemFree(d_C));
    CHECK_CUDA(cuModuleUnload(module));

    return result;
}

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -m <size>     M dimension (default: %d)\n", g_M);
    printf("  -n <size>     N dimension (default: %d)\n", g_N);
    printf("  -k <size>     K dimension (default: %d)\n", g_K);
    printf("  --tile <size> Tile size: 32, 64, or 128 (default: %d)\n", g_tile_size);
    printf("  -b, --bench   Benchmark mode\n");
    printf("  -t, --tc      Use Tensor Core (default)\n");
    printf("  --naive       Use naive scalar kernel\n");
    printf("  -i <iters>    Benchmark iterations (default: %d)\n", g_bench_iters);
    printf("  -h, --help    Show this help\n");
}

int main(int argc, char** argv) {
    CUdevice device;
    CUcontext context;
    int deviceCount;
    char deviceName[256];
    gpu_specs_t specs;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) g_M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) g_N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) g_K = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) g_bench_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tile") == 0 && i + 1 < argc) g_tile_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--bench") == 0) g_benchmark_mode = 1;
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tc") == 0) g_use_tensor_core = 1;
        else if (strcmp(argv[i], "--naive") == 0) g_use_tensor_core = 0;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        }
    }

    if (g_M <= 0 || g_N <= 0 || g_K <= 0 || g_M > 16384 || g_N > 16384 || g_K > 16384 || g_bench_iters <= 0) {
        fprintf(stderr, "Dimensions must be in 1..16384 and iterations positive\n"); return 1;
    }

    /* Validate tile size */
    if (g_tile_size != 32 && g_tile_size != 64 && g_tile_size != 128) {
        fprintf(stderr, "Invalid tile size %d (must be 32, 64, or 128)\n", g_tile_size);
        return 1;
    }

    /* Adjust dimensions for tensor core requirements */
    if (g_use_tensor_core) {
        g_M = ((g_M + g_tile_size - 1) / g_tile_size) * g_tile_size;
        g_N = ((g_N + g_tile_size - 1) / g_tile_size) * g_tile_size;
        g_K = ((g_K + 31) / 32) * 32;  /* K always processed in 32-element chunks */
    }

    printf("FP8 GEMM %s (%s, tile=%d)\n", g_benchmark_mode ? "Benchmark" : "Test",
           g_use_tensor_core ? "Tensor Core" : "Naive", g_tile_size);
    printf("==========================================\n");

    if (cuewInit(CUEW_INIT_CUDA) != CUEW_SUCCESS) {
        fprintf(stderr, "Failed to initialize cuew\n"); return 1;
    }

    CHECK_CUDA(cuInit(0));
    CHECK_CUDA(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "No CUDA devices\n"); return 1; }

    CHECK_CUDA(cuDeviceGet(&device, 0));
    CHECK_CUDA(cuDeviceGetName(deviceName, sizeof(deviceName), device));
    query_gpu_specs(device, &specs);
    print_gpu_specs(deviceName, &specs);

    if (g_use_tensor_core && specs.major * 10 + specs.minor < 89) {
        printf("Warning: GPU doesn't support FP8 tensor cores, falling back to naive\n");
        g_use_tensor_core = 0;
    }

    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    size_t size_A = (size_t)g_M * g_K;
    size_t size_B = (size_t)g_K * g_N;
    size_t size_C = (size_t)g_M * g_N;

    fp8_e4m3_t* A_e4m3 = (fp8_e4m3_t*)malloc(size_A);
    fp8_e4m3_t* B_e4m3 = (fp8_e4m3_t*)malloc(size_B);
    fp8_e5m2_t* A_e5m2 = (fp8_e5m2_t*)malloc(size_A);
    fp8_e5m2_t* B_e5m2 = (fp8_e5m2_t*)malloc(size_B);
    float* C_ref = (float*)malloc(size_C * sizeof(float));

    printf("Initializing data (%dx%dx%d)...\n", g_M, g_N, g_K);
    srand(42);
    for (size_t i = 0; i < size_A; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        A_e4m3[i] = float_to_fp8_e4m3(val);
        A_e5m2[i] = float_to_fp8_e5m2(val);
    }
    for (size_t i = 0; i < size_B; i++) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
        B_e4m3[i] = float_to_fp8_e4m3(val);
        B_e5m2[i] = float_to_fp8_e5m2(val);
    }

    const char *ptx_e4m3, *ptx_e5m2, *name_e4m3, *name_e5m2;
    if (g_use_tensor_core) {
        if (g_tile_size == 128) {
            ptx_e4m3 = ptx_gemm_tc128_e4m3; name_e4m3 = "gemm_fp8_tc128_e4m3";
            ptx_e5m2 = ptx_gemm_tc128_e5m2; name_e5m2 = "gemm_fp8_tc128_e5m2";
        } else if (g_tile_size == 64) {
            ptx_e4m3 = ptx_gemm_tc64_e4m3; name_e4m3 = "gemm_fp8_tc64_e4m3";
            ptx_e5m2 = ptx_gemm_tc64_e5m2; name_e5m2 = "gemm_fp8_tc64_e5m2";
        } else {
            ptx_e4m3 = ptx_gemm_tc_e4m3; name_e4m3 = "gemm_fp8_tc_e4m3";
            ptx_e5m2 = ptx_gemm_tc_e5m2; name_e5m2 = "gemm_fp8_tc_e5m2";
        }
    } else {
        ptx_e4m3 = ptx_gemm_naive_e4m3; name_e4m3 = "gemm_fp8_naive_e4m3";
        ptx_e5m2 = ptx_gemm_naive_e5m2; name_e5m2 = "gemm_fp8_naive_e5m2";
    }

    printf("\nComputing CPU reference...\n");
    gemm_reference_e4m3(A_e4m3, B_e4m3, C_ref, g_M, g_N, g_K);
    benchmark_result_t r1 = run_gemm_benchmark(ptx_e4m3, name_e4m3, A_e4m3, B_e4m3, C_ref,
                                                g_M, g_N, g_K, g_use_tensor_core, "FP8 E4M3");

    gemm_reference_e5m2(A_e5m2, B_e5m2, C_ref, g_M, g_N, g_K);
    benchmark_result_t r2 = run_gemm_benchmark(ptx_e5m2, name_e5m2, A_e5m2, B_e5m2, C_ref,
                                                g_M, g_N, g_K, g_use_tensor_core, "FP8 E5M2");

    printf("\n==========================================\n");
    printf("Summary (%dx%dx%d, %s)\n", g_M, g_N, g_K, g_use_tensor_core ? "Tensor Core" : "Naive");
    printf("==========================================\n");
    printf("Format  Time(ms)   GFLOPS   Verified\n");
    printf("E4M3    %8.4f  %8.2f  %s\n", r1.elapsed_ms, r1.gflops, r1.verified ? "PASS" : "FAIL");
    printf("E5M2    %8.4f  %8.2f  %s\n", r2.elapsed_ms, r2.gflops, r2.verified ? "PASS" : "FAIL");

    free(A_e4m3); free(B_e4m3); free(A_e5m2); free(B_e5m2); free(C_ref);
    CHECK_CUDA(cuCtxDestroy(context));
    printf("\nDone.\n");
    return (r1.verified && r2.verified) ? 0 : 1;
}
