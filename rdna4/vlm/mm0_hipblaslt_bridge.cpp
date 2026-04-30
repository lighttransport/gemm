/*
 * mm0_hipblaslt_bridge.cpp
 *
 * Thin C-callable wrapper around hipBLASLt for the Qwen3 vision projector
 * `mm0` BF16 GEMM (M=1024, N=4608, K=4608).
 *
 * Layout matches hip_vision_encoder.c:
 *   X row-major [M,K], W row-major [N,K], Y row-major [M,N] = X * W^T.
 * hipBLASLt sees this as column-major Y^T[N,M] = W[N,K] * X^T[K,M], so
 *   A=W (op=T, [K,N] col-major), B=X (op=N, [K,M] col-major), C/D=Y^T [N,M].
 *
 * Pin algo index via mm0_hipblaslt_set_algo_index() (default 73624 from
 * bench_vlm_hipblaslt's earlier sweep — 78.9% of peak on RX 9070 XT).
 */

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define HBLT_RET(expr)                                                         \
  do {                                                                         \
    hipblasStatus_t _st = (expr);                                              \
    if (_st != HIPBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "[mm0_blaslt] hipBLASLt error %s:%d status=%d\n",  \
                   __FILE__, __LINE__, static_cast<int>(_st));                 \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define HIP_RET(expr)                                                          \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      std::fprintf(stderr, "[mm0_blaslt] HIP error %s:%d: %s\n", __FILE__,    \
                   __LINE__, hipGetErrorString(_err));                         \
      return -1;                                                               \
    }                                                                          \
  } while (0)

namespace {

struct State {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t matmul = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  bool algo_valid = false;
  void* workspace = nullptr;
  size_t workspace_size = 0;
  int M = 0, N = 0, K = 0;
};

State g_state;
int g_target_algo = 73624;

}  // namespace

extern "C" int mm0_hipblaslt_set_algo_index(int idx) {
  g_target_algo = idx;
  return 0;
}

extern "C" int mm0_hipblaslt_get_algo_index(void) {
  if (!g_state.algo_valid) return -1;
  hipblasLtMatmulAlgo_t copy = g_state.algo;
  return hipblaslt_ext::getIndexFromAlgo(copy);
}

extern "C" int mm0_hipblaslt_init(int M, int N, int K, const void* d_bias) {
  if (g_state.handle != nullptr) {
    std::fprintf(stderr, "[mm0_blaslt] already initialized\n");
    return -1;
  }
  g_state.M = M;
  g_state.N = N;
  g_state.K = K;

  HBLT_RET(hipblasLtCreate(&g_state.handle));
  HBLT_RET(hipblasLtMatmulDescCreate(&g_state.matmul, HIPBLAS_COMPUTE_32F,
                                     HIP_R_32F));
  hipblasOperation_t trans_a = HIPBLAS_OP_T;
  hipblasOperation_t trans_b = HIPBLAS_OP_N;
  HBLT_RET(hipblasLtMatmulDescSetAttribute(
      g_state.matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  HBLT_RET(hipblasLtMatmulDescSetAttribute(
      g_state.matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));

  if (d_bias != nullptr) {
    hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_BIAS;
    hipDataType bias_type = HIP_R_32F;
    HBLT_RET(hipblasLtMatmulDescSetAttribute(
        g_state.matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    HBLT_RET(hipblasLtMatmulDescSetAttribute(
        g_state.matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias,
        sizeof(d_bias)));
    HBLT_RET(hipblasLtMatmulDescSetAttribute(
        g_state.matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
        sizeof(bias_type)));
  }

  HBLT_RET(hipblasLtMatrixLayoutCreate(&g_state.a, HIP_R_16BF, K, N, K));
  HBLT_RET(hipblasLtMatrixLayoutCreate(&g_state.b, HIP_R_16BF, K, M, K));
  HBLT_RET(hipblasLtMatrixLayoutCreate(&g_state.c, HIP_R_32F, N, M, N));
  HBLT_RET(hipblasLtMatrixLayoutCreate(&g_state.d, HIP_R_32F, N, M, N));

  HBLT_RET(hipblasLtMatmulPreferenceCreate(&g_state.pref));
  uint64_t max_ws = 256ull << 20;
  HBLT_RET(hipblasLtMatmulPreferenceSetAttribute(
      g_state.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
      sizeof(max_ws)));

  std::vector<hipblasLtMatmulHeuristicResult_t> results(256);
  int returned = 0;
  HBLT_RET(hipblasLtMatmulAlgoGetHeuristic(g_state.handle, g_state.matmul,
                                            g_state.a, g_state.b, g_state.c,
                                            g_state.d, g_state.pref,
                                            static_cast<int>(results.size()),
                                            results.data(), &returned));
  if (returned == 0) {
    std::fprintf(stderr, "[mm0_blaslt] no algos returned\n");
    return -1;
  }

  int best = -1;
  for (int i = 0; i < returned; ++i) {
    if (results[i].state != HIPBLAS_STATUS_SUCCESS) continue;
    hipblasLtMatmulAlgo_t copy = results[i].algo;
    int idx = hipblaslt_ext::getIndexFromAlgo(copy);
    if (idx == g_target_algo) {
      best = i;
      break;
    }
  }
  if (best < 0) {
    std::fprintf(stderr,
                 "[mm0_blaslt] target algo %d not in heuristic, falling back "
                 "to results[0] (idx=%d)\n",
                 g_target_algo,
                 (returned > 0
                      ? hipblaslt_ext::getIndexFromAlgo(results[0].algo)
                      : -1));
    best = 0;
  }

  g_state.algo = results[best].algo;
  g_state.algo_valid = true;
  g_state.workspace_size = results[best].workspaceSize;
  if (g_state.workspace_size > 0) {
    HIP_RET(hipMalloc(&g_state.workspace, g_state.workspace_size));
  }

  std::fprintf(
      stderr,
      "[mm0_blaslt] init M=%d N=%d K=%d algo_idx=%d ws=%zu bias=%s\n",
      M, N, K, hipblaslt_ext::getIndexFromAlgo(g_state.algo),
      g_state.workspace_size, d_bias ? "yes" : "no");
  return 0;
}

extern "C" int mm0_hipblaslt_run(void* d_y, const void* d_w, const void* d_x) {
  if (!g_state.algo_valid) return -1;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  HBLT_RET(hipblasLtMatmul(g_state.handle, g_state.matmul, &alpha, d_w,
                           g_state.a, d_x, g_state.b, &beta, d_y, g_state.c,
                           d_y, g_state.d, &g_state.algo, g_state.workspace,
                           g_state.workspace_size, /*stream=*/0));
  return 0;
}

extern "C" int mm0_hipblaslt_destroy(void) {
  if (g_state.workspace) hipFree(g_state.workspace);
  if (g_state.pref) hipblasLtMatmulPreferenceDestroy(g_state.pref);
  if (g_state.d) hipblasLtMatrixLayoutDestroy(g_state.d);
  if (g_state.c) hipblasLtMatrixLayoutDestroy(g_state.c);
  if (g_state.b) hipblasLtMatrixLayoutDestroy(g_state.b);
  if (g_state.a) hipblasLtMatrixLayoutDestroy(g_state.a);
  if (g_state.matmul) hipblasLtMatmulDescDestroy(g_state.matmul);
  if (g_state.handle) hipblasLtDestroy(g_state.handle);
  g_state = State{};
  return 0;
}
