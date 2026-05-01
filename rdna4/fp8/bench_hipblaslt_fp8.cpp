// bench_hipblaslt_fp8.cpp — hipBLASLt FP8 (e4m3) on gfx1201 to establish a
// vendor-tuned upper bound for the mm0 shape (1024 x 4608 x 4608).
//
// Layout: X[M,K] row-major, W[N,K] row-major, Y[M,N] row-major.
// hipBLASLt sees Y^T = W * X^T (column-major), so we set:
//   Op A = N (W as-is, lda=K, M=N, K=K)
//   Op B = T (X^T from row-major X)
//   matrix A: HIP_R_8F_E4M3, dims [N, K]
//   matrix B: HIP_R_8F_E4M3, dims [M, K]   (transposed via opB=T → effective B[K,M])
//   matrix D: HIP_R_32F,     dims [N, M]
//
// Build:
//   hipcc -O3 -std=c++17 bench_hipblaslt_fp8.cpp -o bench_hipblaslt_fp8 \
//         -L/opt/rocm/lib -lhipblaslt -lamdhip64

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <ctime>

#define HIP_CHECK(e) do { hipError_t _e=(e); if(_e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(_e));exit(1);} } while(0)
#define HBLT_CHECK(e) do { auto _e=(e); if(_e!=HIPBLAS_STATUS_SUCCESS){fprintf(stderr,"HBLT %s:%d status=%d\n",__FILE__,__LINE__,(int)_e);exit(1);} } while(0)

struct Shape { const char* name; int m, n, k; };
static const Shape kShapes[] = {
    {"mm0", 1024, 4608, 4608},
    {"mm2", 1024, 5120, 4608},
};

static double timer_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

__global__ void fill_fp8(uint8_t* p, int64_t n, uint32_t seed) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        uint32_t r = (uint32_t)(i + seed);
        r = r * 1103515245u + 12345u;
        // Build a simple FP8 e4m3 with small magnitude:
        // sign bit 0, exp 6..9 (≈ 2^-1..2^2), mantissa 0..7
        uint8_t exp4 = 6 + ((r >> 8) & 3);
        uint8_t mant3 = (r >> 12) & 7;
        uint8_t sign = (r >> 16) & 1;
        p[i] = (uint8_t)((sign << 7) | (exp4 << 3) | mant3);
    }
}

int main(int argc, char** argv) {
    const char* shape_name = "mm0";
    int iters = 200;
    int start_algo = 0;
    int end_algo = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--shape") && i + 1 < argc) shape_name = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--start") && i + 1 < argc) start_algo = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--end") && i + 1 < argc) end_algo = atoi(argv[++i]);
    }
    const Shape* sh = nullptr;
    for (auto& s : kShapes) if (!strcmp(s.name, shape_name)) sh = &s;
    if (!sh) { fprintf(stderr, "shape %s not found\n", shape_name); return 1; }

    int M = sh->m, N = sh->n, K = sh->k;
    printf("hipBLASLt FP8 e4m3 bench  shape=%s M=%d N=%d K=%d iters=%d\n",
           sh->name, M, N, K, iters);

    HIP_CHECK(hipSetDevice(0));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    size_t bytesX = (size_t)M * K;
    size_t bytesW = (size_t)N * K;
    size_t bytesY = (size_t)M * N * sizeof(float);
    uint8_t *dX, *dW; float *dY;
    HIP_CHECK(hipMalloc(&dX, bytesX));
    HIP_CHECK(hipMalloc(&dW, bytesW));
    HIP_CHECK(hipMalloc(&dY, bytesY));
    fill_fp8<<<512, 256, 0, stream>>>(dX, bytesX, 0x1234);
    fill_fp8<<<512, 256, 0, stream>>>(dW, bytesW, 0x5678);
    HIP_CHECK(hipMemsetAsync(dY, 0, bytesY, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    hipblasLtHandle_t handle;
    HBLT_CHECK(hipblasLtCreate(&handle));

    // hipBLASLt op-A=N (col-major view of W as [N,K]), op-B=T (X^T)
    // Effective: D[N,M] (col-major) = A[N,K] * B[K,M] = W[N,K] * X^T[K,M]
    hipblasLtMatrixLayout_t layoutA, layoutB, layoutC;
    HBLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_8F_E4M3, N, K, K));
    HBLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_8F_E4M3, K, M, K));
    HBLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F,     N, M, N));

    hipblasLtMatmulDesc_t matmul;
    HBLT_CHECK(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t opN = HIPBLAS_OP_N, opT = HIPBLAS_OP_T;
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));

    // Get heuristic-recommended algorithms.
    hipblasLtMatmulPreference_t pref;
    HBLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t workspace_size = 32 * 1024 * 1024;
    HBLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    void* workspace = nullptr;
    HIP_CHECK(hipMalloc(&workspace, workspace_size));

    // hipBLASLt FP8 SF8 kernels on gfx1201 expect scale pointers (HAS_ScaleAB / HAS_ScaleCD).
    // Default 1.0f scale for A,B,C,D so the kernels can read non-null device pointers.
    float *dScaleA, *dScaleB, *dScaleC, *dScaleD;
    HIP_CHECK(hipMalloc(&dScaleA, sizeof(float)));
    HIP_CHECK(hipMalloc(&dScaleB, sizeof(float)));
    HIP_CHECK(hipMalloc(&dScaleC, sizeof(float)));
    HIP_CHECK(hipMalloc(&dScaleD, sizeof(float)));
    {
        float one = 1.0f;
        HIP_CHECK(hipMemcpy(dScaleA, &one, sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dScaleB, &one, sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dScaleC, &one, sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dScaleD, &one, sizeof(float), hipMemcpyHostToDevice));
    }
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA)));
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB, sizeof(dScaleB)));
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, &dScaleC, sizeof(dScaleC)));
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(matmul, HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, &dScaleD, sizeof(dScaleD)));

    int returned = 0;
    std::vector<hipblasLtMatmulHeuristicResult_t> heur(32);
    auto heur_st = hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, layoutA, layoutB, layoutC, layoutC,
        pref, (int)heur.size(), heur.data(), &returned);
    printf("  heuristic status=%d candidates=%d\n", (int)heur_st, returned);
    fflush(stdout);
    if (returned == 0) {
        fprintf(stderr, "hipBLASLt: no FP8 algorithm available for shape %s\n", sh->name);
        return 2;
    }

    float alpha = 1.0f, beta = 0.0f;
    double best_tflops = 0.0;
    int best_idx = -1;
    double best_ms = 0.0;
    int max_algo = returned;
    if (end_algo < 0 || end_algo > max_algo) end_algo = max_algo;
    for (int i = start_algo; i < end_algo; i++) {
        printf("  trying algo[%2d]/%d ws=%zu ...", i, max_algo, heur[i].workspaceSize);
        fflush(stdout);
        // Validate algo is actually supported for this problem.
        size_t ws_needed = 0;
        auto sup = hipblaslt_ext::matmulIsAlgoSupported(
            handle, matmul, &alpha, layoutA, layoutB, &beta, layoutC, layoutC,
            heur[i].algo, ws_needed);
        if (sup != HIPBLAS_STATUS_SUCCESS) {
            printf(" not_supported=%d (skip)\n", (int)sup);
            fflush(stdout);
            continue;
        }
        if (ws_needed > workspace_size) {
            printf(" ws_needed=%zu > %zu (skip)\n", ws_needed, workspace_size);
            fflush(stdout);
            continue;
        }
        // Single launch first, check status.
        auto st = hipblasLtMatmul(handle, matmul, &alpha, dW, layoutA,
                                  dX, layoutB, &beta, dY, layoutC,
                                  dY, layoutC, &heur[i].algo,
                                  workspace, workspace_size, stream);
        if (st != HIPBLAS_STATUS_SUCCESS) {
            printf(" status=%d (skip)\n", (int)st);
            fflush(stdout);
            continue;
        }
        auto sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess) {
            printf(" sync_err=%s (skip)\n", hipGetErrorString(sync_err));
            fflush(stdout);
            continue;
        }
        // Warmup
        for (int w = 0; w < 3; w++) {
            hipblasLtMatmul(handle, matmul, &alpha, dW, layoutA,
                            dX, layoutB, &beta, dY, layoutC,
                            dY, layoutC, &heur[i].algo,
                            workspace, workspace_size, stream);
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        double t0 = timer_ms();
        for (int it = 0; it < iters; it++) {
            HBLT_CHECK(hipblasLtMatmul(handle, matmul, &alpha, dW, layoutA,
                                       dX, layoutB, &beta, dY, layoutC,
                                       dY, layoutC, &heur[i].algo,
                                       workspace, workspace_size, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        double ms = (timer_ms() - t0) / iters;
        double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
        printf(" %7.4f ms  %6.1f TFLOP/s\n", ms, tflops);
        fflush(stdout);
        if (tflops > best_tflops) { best_tflops = tflops; best_idx = i; best_ms = ms; }
    }
    printf("BEST hipBLASLt FP8 for %s: algo[%d] %.4f ms %.1f TFLOP/s\n",
           sh->name, best_idx, best_ms, best_tflops);

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatmulDescDestroy(matmul);
    hipblasLtMatrixLayoutDestroy(layoutA);
    hipblasLtMatrixLayoutDestroy(layoutB);
    hipblasLtMatrixLayoutDestroy(layoutC);
    hipblasLtDestroy(handle);
    hipFree(workspace);
    hipFree(dX); hipFree(dW); hipFree(dY);
    return 0;
}
