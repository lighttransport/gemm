/*
 * mm_blaslt_bridge.cpp - Multi-shape hipBLASLt BF16 GEMM cache for the LLM
 * runner. See mm_blaslt_bridge.h for the call convention.
 */

#include "mm_blaslt_bridge.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#define HBLT_RET(expr)                                                         \
  do {                                                                         \
    hipblasStatus_t _st = (expr);                                              \
    if (_st != HIPBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "[mm_blaslt] hipBLASLt error %s:%d status=%d\n",    \
                   __FILE__, __LINE__, static_cast<int>(_st));                 \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define HIP_RET(expr)                                                          \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      std::fprintf(stderr, "[mm_blaslt] HIP error %s:%d: %s\n", __FILE__,      \
                   __LINE__, hipGetErrorString(_err));                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

namespace {

struct Plan {
  hipblasLtMatmulDesc_t matmul = nullptr;
  hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  void *workspace = nullptr;
  size_t workspace_size = 0;
  bool valid = false;
};

struct ShapeKey {
  int M, N, K;
  bool operator==(const ShapeKey &o) const noexcept {
    return M == o.M && N == o.N && K == o.K;
  }
};

struct ShapeKeyHash {
  size_t operator()(const ShapeKey &k) const noexcept {
    /* mix M/N/K — all positive ints in practice */
    size_t h = static_cast<size_t>(k.M) * 0x9E3779B185EBCA87ull;
    h ^= static_cast<size_t>(k.N) * 0xC2B2AE3D27D4EB4Full;
    h ^= static_cast<size_t>(k.K) * 0x165667B19E3779F9ull;
    return h;
  }
};

struct State {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;
  std::unordered_map<ShapeKey, Plan, ShapeKeyHash> plans;
  bool initialized = false;
  int verbose = 0;
};

State g_state;

void destroy_plan(Plan &p) {
  if (p.workspace) {
    (void)hipFree(p.workspace);
    p.workspace = nullptr;
  }
  if (p.d) hipblasLtMatrixLayoutDestroy(p.d);
  if (p.c) hipblasLtMatrixLayoutDestroy(p.c);
  if (p.b) hipblasLtMatrixLayoutDestroy(p.b);
  if (p.a) hipblasLtMatrixLayoutDestroy(p.a);
  if (p.matmul) hipblasLtMatmulDescDestroy(p.matmul);
  p = Plan{};
}

int build_plan(int M, int N, int K, Plan &p) {
  HBLT_RET(hipblasLtMatmulDescCreate(&p.matmul, HIPBLAS_COMPUTE_32F,
                                     HIP_R_32F));
  hipblasOperation_t trans_a = HIPBLAS_OP_T;
  hipblasOperation_t trans_b = HIPBLAS_OP_N;
  HBLT_RET(hipblasLtMatmulDescSetAttribute(
      p.matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  HBLT_RET(hipblasLtMatmulDescSetAttribute(
      p.matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  /* W [N,K] BF16 row-major == [K,N] col-major with op=T */
  HBLT_RET(hipblasLtMatrixLayoutCreate(&p.a, HIP_R_16BF, K, N, K));
  /* X [M,K] BF16 row-major == [K,M] col-major with op=N */
  HBLT_RET(hipblasLtMatrixLayoutCreate(&p.b, HIP_R_16BF, K, M, K));
  /* Y [M,N] F32 row-major == [N,M] col-major */
  HBLT_RET(hipblasLtMatrixLayoutCreate(&p.c, HIP_R_32F, N, M, N));
  HBLT_RET(hipblasLtMatrixLayoutCreate(&p.d, HIP_R_32F, N, M, N));

  std::vector<hipblasLtMatmulHeuristicResult_t> results(64);
  int returned = 0;
  HBLT_RET(hipblasLtMatmulAlgoGetHeuristic(g_state.handle, p.matmul, p.a, p.b,
                                            p.c, p.d, g_state.pref,
                                            static_cast<int>(results.size()),
                                            results.data(), &returned));
  if (returned == 0) {
    std::fprintf(stderr, "[mm_blaslt] no algos for M=%d N=%d K=%d\n", M, N, K);
    return -1;
  }
  /* Pick first valid result. */
  int best = -1;
  for (int i = 0; i < returned; ++i) {
    if (results[i].state == HIPBLAS_STATUS_SUCCESS) {
      best = i;
      break;
    }
  }
  if (best < 0) {
    std::fprintf(stderr, "[mm_blaslt] no successful algo for M=%d N=%d K=%d\n",
                 M, N, K);
    return -1;
  }
  p.algo = results[best].algo;
  p.workspace_size = results[best].workspaceSize;
  if (p.workspace_size > 0) {
    HIP_RET(hipMalloc(&p.workspace, p.workspace_size));
  }
  p.valid = true;
  if (g_state.verbose) {
    std::fprintf(stderr,
                 "[mm_blaslt] cached plan M=%d N=%d K=%d algo_idx=%d ws=%zu\n",
                 M, N, K, hipblaslt_ext::getIndexFromAlgo(p.algo),
                 p.workspace_size);
  }
  return 0;
}

}  // namespace

extern "C" int mm_blaslt_init(void) {
  if (g_state.initialized) return 0;
  HBLT_RET(hipblasLtCreate(&g_state.handle));
  HBLT_RET(hipblasLtMatmulPreferenceCreate(&g_state.pref));
  uint64_t max_ws = 256ull << 20;
  HBLT_RET(hipblasLtMatmulPreferenceSetAttribute(
      g_state.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
      sizeof(max_ws)));
  if (const char *v = std::getenv("MM_BLASLT_VERBOSE")) {
    g_state.verbose = std::atoi(v);
  }
  g_state.initialized = true;
  return 0;
}

extern "C" int mm_blaslt_run_bf16(void *d_y_f32, const void *d_w_bf16,
                                  const void *d_x_bf16, int M, int N, int K,
                                  void *stream) {
  if (!g_state.initialized) {
    if (mm_blaslt_init() != 0) return -1;
  }
  ShapeKey key{M, N, K};
  auto it = g_state.plans.find(key);
  if (it == g_state.plans.end()) {
    Plan p;
    if (build_plan(M, N, K, p) != 0) {
      destroy_plan(p);
      return -1;
    }
    it = g_state.plans.emplace(key, p).first;
  }
  Plan &p = it->second;
  if (!p.valid) return -1;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  HBLT_RET(hipblasLtMatmul(g_state.handle, p.matmul, &alpha, d_w_bf16, p.a,
                           d_x_bf16, p.b, &beta, d_y_f32, p.c, d_y_f32, p.d,
                           &p.algo, p.workspace, p.workspace_size,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

extern "C" void mm_blaslt_destroy(void) {
  for (auto &kv : g_state.plans) {
    destroy_plan(kv.second);
  }
  g_state.plans.clear();
  if (g_state.pref) {
    hipblasLtMatmulPreferenceDestroy(g_state.pref);
    g_state.pref = nullptr;
  }
  if (g_state.handle) {
    hipblasLtDestroy(g_state.handle);
    g_state.handle = nullptr;
  }
  g_state.initialized = false;
}
