// SPDX-License-Identifier: MIT
// BF16 ConvRot provider matching PyTorch ROCm's preferred hipBLASLt backend.
// GPU tensor storage and workspace are owned and budgeted by the native runner.
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace {
struct Plan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
    ~Plan() {
        if (c) hipblasLtMatrixLayoutDestroy(c);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (desc) hipblasLtMatmulDescDestroy(desc);
    }
};
struct State {
    hipblasLtHandle_t handle = nullptr;
    std::mutex mutex;
    std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans;
    ~State() {
        plans.clear();
        if (handle) hipblasLtDestroy(handle);
    }
};
#define LT_CHECK(call) do { int status = int(call); if (status) return status; } while (0)
}

extern "C" int video_bf16_lt_abi() { return 1; }

extern "C" int video_bf16_lt_create(void **context) {
    if (!context) return int(HIPBLAS_STATUS_INVALID_VALUE);
    try {
        auto state = std::make_unique<State>();
        LT_CHECK(hipblasLtCreate(&state->handle));
        *context = state.release();
        return 0;
    } catch (...) { return int(HIPBLAS_STATUS_ALLOC_FAILED); }
}
extern "C" void video_bf16_lt_destroy(void *context) { delete static_cast<State *>(context); }

extern "C" int video_bf16_lt_forward(void *context, void *out, const void *weight,
                                     const void *input, int m, int n, int k,
                                     void *workspace, size_t workspace_bytes, void *stream) {
    if (!context || !out || !weight || !input || m <= 0 || n <= 0 || k <= 0)
        return int(HIPBLAS_STATUS_INVALID_VALUE);
    try {
        auto &state = *static_cast<State *>(context);
        std::lock_guard<std::mutex> guard(state.mutex);
        auto key = std::make_tuple(m, n, k);
        auto found = state.plans.find(key);
        if (found == state.plans.end()) {
            auto plan = std::make_unique<Plan>();
            LT_CHECK(hipblasLtMatmulDescCreate(&plan->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            // Right-hand ConvRot matrix is [K,N], matching torch X @ rotation.
            // Although the Hadamard matrix is symmetric, using a transpose
            // selects a different reduction and changes BF16 rounding ties.
            hipblasOperation_t ta = HIPBLAS_OP_N, tb = HIPBLAS_OP_N;
            LT_CHECK(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
            LT_CHECK(hipblasLtMatmulDescSetAttribute(plan->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
            LT_CHECK(hipblasLtMatrixLayoutCreate(&plan->a, HIP_R_16BF, n, k, n));
            LT_CHECK(hipblasLtMatrixLayoutCreate(&plan->b, HIP_R_16BF, k, m, k));
            LT_CHECK(hipblasLtMatrixLayoutCreate(&plan->c, HIP_R_16BF, n, m, n));
            hipblasLtMatmulPreference_t pref = nullptr;
            LT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
            auto status = hipblasLtMatmulPreferenceSetAttribute(pref,
                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
            hipblasLtMatmulHeuristicResult_t result{};
            int count = 0;
            if (!status) status = hipblasLtMatmulAlgoGetHeuristic(state.handle, plan->desc,
                plan->a, plan->b, plan->c, plan->c, pref, 1, &result, &count);
            hipblasLtMatmulPreferenceDestroy(pref);
            if (status) return int(status);
            if (count != 1 || result.state != HIPBLAS_STATUS_SUCCESS)
                return int(HIPBLAS_STATUS_NOT_SUPPORTED);
            plan->algo = result.algo;
            plan->workspace_bytes = result.workspaceSize;
            found = state.plans.emplace(key, std::move(plan)).first;
        }
        auto &plan = *found->second;
        if (plan.workspace_bytes > workspace_bytes || (plan.workspace_bytes && !workspace))
            return int(HIPBLAS_STATUS_INVALID_VALUE);
        float alpha = 1.f, beta = 0.f;
        return int(hipblasLtMatmul(state.handle, plan.desc, &alpha, weight, plan.a, input,
            plan.b, &beta, out, plan.c, out, plan.c, &plan.algo, workspace,
            workspace_bytes, reinterpret_cast<hipStream_t>(stream)));
    } catch (...) { return int(HIPBLAS_STATUS_ALLOC_FAILED); }
}
