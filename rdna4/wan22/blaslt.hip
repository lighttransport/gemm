// SPDX-License-Identifier: MIT
// Cached hipBLASLt plans; tensor storage and workspace belong to PyTorch.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <tuple>
#include <cstdio>

extern "C" __global__ void wan22_q8_decode(_Float16 *, const unsigned char *, long);

struct WanPlan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;

    ~WanPlan() {
        if (c) hipblasLtMatrixLayoutDestroy(c);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (desc) hipblasLtMatmulDescDestroy(desc);
    }
};

struct WanBlas {
    hipblasLtHandle_t handle = nullptr;
    std::map<std::tuple<int, int, int, bool>, WanPlan> plans;
    ~WanBlas() {
        plans.clear();
        if (handle) hipblasLtDestroy(handle);
    }
};

#define WAN_BLAS_CHECK(call) do { \
    hipblasStatus_t status = (call); \
    if (status != HIPBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "Wan hipBLASLt: %s returned %d\n", #call, int(status)); \
        return 1000 + int(status); \
    } \
} while (0)

extern "C" int wan22_blas_create(void **context) {
    auto *state = new WanBlas;
    hipblasStatus_t status = hipblasLtCreate(&state->handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        delete state;
        return 1000 + int(status);
    }
    *context = state;
    return 0;
}

extern "C" void wan22_blas_destroy(void *context) {
    delete static_cast<WanBlas *>(context);
}

extern "C" int wan22_projection_blas(void *context, _Float16 *out, const _Float16 *input,
                                     const unsigned char *packed, _Float16 *scratch,
                                     const _Float16 *bias, void *workspace, size_t workspace_bytes,
                                     int m, int n, int k, void *stream_pointer) {
    if (!context || !out || !input || !packed || !scratch || m <= 0 || n <= 0 || k <= 0 || k % 32)
        return int(hipErrorInvalidValue);
    auto *state = static_cast<WanBlas *>(context);
    auto key = std::make_tuple(m, n, k, bias != nullptr);
    auto found = state->plans.find(key);
    if (found == state->plans.end()) {
        // Build locally so failures release partial descriptors and never cache invalid plans.
        WanPlan plan;
        WAN_BLAS_CHECK(hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        hipblasOperation_t trans_a = HIPBLAS_OP_T, trans_b = HIPBLAS_OP_N;
        WAN_BLAS_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                     &trans_a, sizeof(trans_a)));
        WAN_BLAS_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                     &trans_b, sizeof(trans_b)));
        if (bias) {
            hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_BIAS;
            hipDataType bias_type = HIP_R_16F;
            WAN_BLAS_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                         &epilogue, sizeof(epilogue)));
            WAN_BLAS_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                         &bias_type, sizeof(bias_type)));
            WAN_BLAS_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                         &bias, sizeof(bias)));
        }
        // Row-major X[M,K], W[N,K], Y[M,N] map to column-major Y^T = W * X^T.
        WAN_BLAS_CHECK(hipblasLtMatrixLayoutCreate(&plan.a, HIP_R_16F, k, n, k));
        WAN_BLAS_CHECK(hipblasLtMatrixLayoutCreate(&plan.b, HIP_R_16F, k, m, k));
        WAN_BLAS_CHECK(hipblasLtMatrixLayoutCreate(&plan.c, HIP_R_16F, n, m, n));
        hipblasLtMatmulPreference_t preference = nullptr;
        WAN_BLAS_CHECK(hipblasLtMatmulPreferenceCreate(&preference));
        hipblasStatus_t status = hipblasLtMatmulPreferenceSetAttribute(
            preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes));
        hipblasLtMatmulHeuristicResult_t result{};
        int returned = 0;
        if (status == HIPBLAS_STATUS_SUCCESS)
            status = hipblasLtMatmulAlgoGetHeuristic(state->handle, plan.desc, plan.a, plan.b,
                plan.c, plan.c, preference, 1, &result, &returned);
        hipblasLtMatmulPreferenceDestroy(preference);
        if (status != HIPBLAS_STATUS_SUCCESS || returned != 1 || result.state != HIPBLAS_STATUS_SUCCESS)
            return 1000 + int(status == HIPBLAS_STATUS_SUCCESS ? HIPBLAS_STATUS_NOT_SUPPORTED : status);
        plan.algo = result.algo;
        plan.workspace_bytes = result.workspaceSize;
        auto &cached = state->plans[key];
        cached.desc = plan.desc; cached.a = plan.a; cached.b = plan.b; cached.c = plan.c;
        cached.algo = plan.algo; cached.workspace_bytes = plan.workspace_bytes;
        plan.desc = nullptr; plan.a = nullptr; plan.b = nullptr; plan.c = nullptr;
        found = state->plans.find(key);
    }
    WanPlan &plan = found->second;
    if (plan.workspace_bytes > workspace_bytes || (plan.workspace_bytes && !workspace))
        return int(hipErrorInvalidValue);
    if (bias)
        WAN_BLAS_CHECK(hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                     &bias, sizeof(bias)));
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
    long count = long(n) * k;
    hipLaunchKernelGGL(wan22_q8_decode, dim3((count + 255) / 256), dim3(256), 0,
                       stream, scratch, packed, count);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) return int(error);
    const float alpha = 1, beta = 0;
    WAN_BLAS_CHECK(hipblasLtMatmul(state->handle, plan.desc, &alpha, scratch, plan.a,
        input, plan.b, &beta, out, plan.c, out, plan.c, &plan.algo,
        workspace, plan.workspace_bytes, stream));
    return int(hipGetLastError());
}
