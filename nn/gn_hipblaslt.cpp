/* SPDX-License-Identifier: MIT
 * Original optional bridge to AMD's public hipBLASLt API. No BLAS code is
 * embedded. The SDK headers/runtime are external, opt-in build dependencies.
 */
#include "gn_hipblaslt.h"
#include <cstdio>
#include <hipblaslt/hipblaslt.h>
#include <new>
#include <vector>

struct LtPlan {
    int m, n, k;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace = 0;
    ~LtPlan() {
        if (a)
            hipblasLtMatrixLayoutDestroy(a);
        if (b)
            hipblasLtMatrixLayoutDestroy(b);
        if (c)
            hipblasLtMatrixLayoutDestroy(c);
        if (desc)
            hipblasLtMatmulDescDestroy(desc);
    }
};
struct LtContext {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    std::vector<LtPlan *> plans;
    ~LtContext() {
        for (auto *p : plans)
            delete p;
        if (pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if (handle)
            hipblasLtDestroy(handle);
    }
};
static bool lt_ok(hipblasStatus_t status, const char *operation) {
    if (status == HIPBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr, "hipBLASLt %s failed: status=%d\n", operation, (int)status);
    return false;
}
extern "C" void *gn_lt_open(void) {
    auto *c = new (std::nothrow) LtContext;
    if (!c)
        return nullptr;
    if (!lt_ok(hipblasLtCreate(&c->handle), "create") ||
        !lt_ok(hipblasLtMatmulPreferenceCreate(&c->pref), "preference")) {
        delete c;
        return nullptr;
    }
    int version = 0;
    hipblasLtGetVersion(c->handle, &version);
    std::fprintf(stderr, "hipBLASLt version=%d: explicit BF16/FP32 matrix backend\n", version);
    return c;
}
extern "C" void gn_lt_close(void *context) { delete static_cast<LtContext *>(context); }
extern "C" int gn_lt_run(void *context, void *y, const void *a, const void *b, int M, int N, int K,
                         float beta, void *workspace, size_t workspace_bytes) {
    auto *c = static_cast<LtContext *>(context);
    if (!c || !y || !a || !b || M < 1 || N < 1 || K < 1)
        return -1;
    LtPlan *plan = nullptr;
    for (auto *p : c->plans)
        if (p->m == M && p->n == N && p->k == K) {
            plan = p;
            break;
        }
    if (!plan) {
        if (c->plans.size() >= 256)
            return -1;
        auto *p = new (std::nothrow) LtPlan;
        if (!p)
            return -1;
        p->m = M;
        p->n = N;
        p->k = K;
        int stride = (K + 31) & ~31;
        hipblasOperation_t trans = HIPBLAS_OP_T;
        uint64_t budget = workspace_bytes;
        /* Compute Y^T = B * A^T in column-major: no output transpose. */
        bool ok =
            lt_ok(hipblasLtMatmulDescCreate(&p->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), "desc") &&
            lt_ok(hipblasLtMatmulDescSetAttribute(p->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans,
                                                  sizeof(trans)),
                  "transpose") &&
            lt_ok(hipblasLtMatrixLayoutCreate(&p->a, HIP_R_16BF, K, N, stride), "layout B") &&
            lt_ok(hipblasLtMatrixLayoutCreate(&p->b, HIP_R_16BF, K, M, stride), "layout A") &&
            lt_ok(hipblasLtMatrixLayoutCreate(&p->c, HIP_R_32F, N, M, N), "layout Y") &&
            lt_ok(hipblasLtMatmulPreferenceSetAttribute(
                      c->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &budget, sizeof(budget)),
                  "workspace");
        hipblasLtMatmulHeuristicResult_t results[8]{};
        int count = 0;
        if (ok)
            ok = lt_ok(hipblasLtMatmulAlgoGetHeuristic(c->handle, p->desc, p->a, p->b, p->c, p->c,
                                                       c->pref, 8, results, &count),
                       "heuristic");
        int selected = -1;
        for (int i = 0; ok && i < count; i++)
            if (results[i].state == HIPBLAS_STATUS_SUCCESS && results[i].workspaceSize <= budget) {
                selected = i;
                break;
            }
        if (selected < 0) {
            std::fprintf(stderr, "hipBLASLt: no supported algorithm M=%d N=%d K=%d\n", M, N, K);
            delete p;
            return -1;
        }
        p->algo = results[selected].algo;
        p->workspace = results[selected].workspaceSize;
        try {
            c->plans.push_back(p);
        } catch (...) {
            delete p;
            return -1;
        }
        plan = p;
    }
    if (plan->workspace > workspace_bytes)
        return -1;
    float alpha = 1;
    return lt_ok(hipblasLtMatmul(c->handle, plan->desc, &alpha, b, plan->a, a, plan->b, &beta, y,
                                 plan->c, y, plan->c, &plan->algo, workspace, plan->workspace,
                                 nullptr),
                 "matmul")
               ? 0
               : -1;
}
