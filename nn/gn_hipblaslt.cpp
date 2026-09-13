/* SPDX-License-Identifier: MIT
 * Original optional bridge to AMD's public hipBLASLt API. No BLAS code is
 * embedded. The SDK headers/runtime are external, opt-in build dependencies.
 */
#include "gn_hipblaslt.h"
#include <cstdio>
#include <dlfcn.h>
#include <hipblaslt/hipblaslt.h>
#include <new>
#include <vector>

struct LtPlan {
    int m, n, k, fp16;
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
    int tune = 0;
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    std::vector<LtPlan *> plans;
    // ROCEW's legacy exported pointer names collide with HIP function names.
    // Resolve the real functions from their library, never call those names
    // directly in this SDK-compiled translation unit.
    void *runtime = nullptr;
    decltype(&hipEventCreate) event_create = nullptr;
    decltype(&hipEventRecord) event_record = nullptr;
    decltype(&hipEventSynchronize) event_sync = nullptr;
    decltype(&hipEventElapsedTime) event_elapsed = nullptr;
    decltype(&hipEventDestroy) event_destroy = nullptr;
    ~LtContext() {
        for (auto *p : plans)
            delete p;
        if (pref)
            hipblasLtMatmulPreferenceDestroy(pref);
        if (handle)
            hipblasLtDestroy(handle);
        if (runtime)
            dlclose(runtime);
    }
};
static bool lt_ok(hipblasStatus_t status, const char *operation) {
    if (status == HIPBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr, "hipBLASLt %s failed: status=%d\n", operation, (int)status);
    return false;
}
extern "C" void *gn_lt_open(int tune) {
    auto *c = new (std::nothrow) LtContext;
    if (!c)
        return nullptr;
    c->tune = tune;
    c->runtime = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
    if (c->runtime) {
        c->event_create =
            reinterpret_cast<decltype(c->event_create)>(dlsym(c->runtime, "hipEventCreate"));
        c->event_record =
            reinterpret_cast<decltype(c->event_record)>(dlsym(c->runtime, "hipEventRecord"));
        c->event_sync =
            reinterpret_cast<decltype(c->event_sync)>(dlsym(c->runtime, "hipEventSynchronize"));
        c->event_elapsed =
            reinterpret_cast<decltype(c->event_elapsed)>(dlsym(c->runtime, "hipEventElapsedTime"));
        c->event_destroy =
            reinterpret_cast<decltype(c->event_destroy)>(dlsym(c->runtime, "hipEventDestroy"));
    }
    if (!c->event_create || !c->event_record || !c->event_sync || !c->event_elapsed ||
        !c->event_destroy) {
        std::fprintf(stderr, "hipBLASLt: cannot resolve runtime event API\n");
        delete c;
        return nullptr;
    }
    if (!lt_ok(hipblasLtCreate(&c->handle), "create") ||
        !lt_ok(hipblasLtMatmulPreferenceCreate(&c->pref), "preference")) {
        delete c;
        return nullptr;
    }
    int version = 0;
    hipblasLtGetVersion(c->handle, &version);
    std::fprintf(stderr, "hipBLASLt version=%d: explicit 16-bit/FP32 matrix backend\n", version);
    return c;
}
extern "C" void gn_lt_close(void *context) { delete static_cast<LtContext *>(context); }
extern "C" int gn_lt_run(void *context, void *y, const void *a, const void *b, int M, int N, int K,
                         float beta, void *workspace, size_t workspace_bytes, int fp16) {
    auto *c = static_cast<LtContext *>(context);
    if (!c || !y || !a || !b || M < 1 || N < 1 || K < 1)
        return -1;
    size_t output_bytes = ((size_t)M * N * 4 + 255) & ~(size_t)255;
    if (!workspace || workspace_bytes < output_bytes)
        return -1;
    size_t library_bytes = workspace_bytes - output_bytes;
    void *trial = static_cast<char *>(workspace) + library_bytes;
    LtPlan *plan = nullptr;
    for (auto *p : c->plans)
        if (p->m == M && p->n == N && p->k == K && p->fp16 == fp16) {
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
        p->fp16 = fp16;
        int stride = (K + 31) & ~31;
        hipblasOperation_t trans = HIPBLAS_OP_T;
        uint64_t budget = library_bytes;
        /* Compute Y^T = B * A^T in column-major: no output transpose. */
        bool ok =
            lt_ok(hipblasLtMatmulDescCreate(&p->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F), "desc") &&
            lt_ok(hipblasLtMatmulDescSetAttribute(p->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &trans,
                                                  sizeof(trans)),
                  "transpose") &&
            lt_ok(hipblasLtMatrixLayoutCreate(&p->a, fp16 ? HIP_R_16F : HIP_R_16BF, K, N, stride),
                  "layout B") &&
            lt_ok(hipblasLtMatrixLayoutCreate(&p->b, fp16 ? HIP_R_16F : HIP_R_16BF, K, M, stride),
                  "layout A") &&
            lt_ok(hipblasLtMatrixLayoutCreate(&p->c, HIP_R_32F, N, M, N), "layout Y") &&
            lt_ok(hipblasLtMatmulPreferenceSetAttribute(
                      c->pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &budget, sizeof(budget)),
                  "workspace");
        hipblasLtMatmulHeuristicResult_t results[32]{};
        int count = 0;
        if (ok)
            ok = lt_ok(hipblasLtMatmulAlgoGetHeuristic(c->handle, p->desc, p->a, p->b, p->c, p->c,
                                                       c->pref, 32, results, &count),
                       "heuristic");
        int selected = -1;
        float best = 1e30f, alpha = 1;
        hipEvent_t start = nullptr, stop = nullptr;
        if (c->event_create(&start) != hipSuccess || c->event_create(&stop) != hipSuccess) {
            if (start)
                (void)c->event_destroy(start);
            delete p;
            return -1;
        }
        for (int i = 0; ok && i < count; i++)
            if (results[i].state == HIPBLAS_STATUS_SUCCESS && results[i].workspaceSize <= budget) {
                if (!c->tune) {
                    selected = i;
                    break;
                }
                auto run = [&] {
                    return hipblasLtMatmul(c->handle, p->desc, &alpha, b, p->a, a, p->b, &beta, y,
                                           p->c, trial, p->c, &results[i].algo, workspace,
                                           results[i].workspaceSize, nullptr);
                };
                // Warm each candidate, then time repeated non-destructive trials.
                if (run() != HIPBLAS_STATUS_SUCCESS ||
                    c->event_record(start, nullptr) != hipSuccess)
                    continue;
                bool valid = true;
                for (int repeat = 0; repeat < 5; repeat++)
                    valid = run() == HIPBLAS_STATUS_SUCCESS && valid;
                float ms = 0;
                valid = c->event_record(stop, nullptr) == hipSuccess && valid;
                valid = c->event_sync(stop) == hipSuccess && valid;
                valid = c->event_elapsed(&ms, start, stop) == hipSuccess && valid;
                if (valid && ms < best) {
                    selected = i;
                    best = ms;
                }
            }
        (void)c->event_destroy(start);
        (void)c->event_destroy(stop);
        if (selected < 0) {
            std::fprintf(stderr, "hipBLASLt: no supported algorithm M=%d N=%d K=%d\n", M, N, K);
            delete p;
            return -1;
        }
        p->algo = results[selected].algo;
        p->workspace = results[selected].workspaceSize;
        if (c->tune)
            std::fprintf(stderr, "hipBLASLt tuned M=%d N=%d K=%d beta=%g candidate=%d/%d ms=%.6g\n",
                         M, N, K, beta, selected, count, best / 5);
        try {
            c->plans.push_back(p);
        } catch (...) {
            delete p;
            return -1;
        }
        plan = p;
    }
    if (plan->workspace > library_bytes)
        return -1;
    float alpha = 1;
    return lt_ok(hipblasLtMatmul(c->handle, plan->desc, &alpha, b, plan->a, a, plan->b, &beta, y,
                                 plan->c, y, plan->c, &plan->algo, workspace, plan->workspace,
                                 nullptr),
                 "matmul")
               ? 0
               : -1;
}
