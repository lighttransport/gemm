/*
 * mm_blaslt_bridge.cpp - Multi-shape hipBLASLt BF16 GEMM cache for the LLM
 * runner. See mm_blaslt_bridge.h for the call convention.
 */

#include "mm_blaslt_bridge.h"

#ifdef MM_BLASLT_DISABLE

/* Build-time fallback for scalar verification on hosts without hipBLASLt dev files. */
extern "C" int mm_blaslt_init(void) { return -1; }

extern "C" int mm_blaslt_run_bf16(void *, const void *, const void *,
                                  int, int, int, void *) { return -1; }
extern "C" int mm_blaslt_run_bf16_strided_batch(
    void *, const void *, const void *, int, int, int, int, void *) { return -1; }
extern "C" int mm_blaslt_run_f32(void *, const void *, const void *,
                                  int, int, int, void *) { return -1; }
extern "C" int mm_blaslt_run_f16(void *, const void *, const void *,
                                  int, int, int, void *) { return -1; }

extern "C" int mm_blaslt_run_bf16_bias(void *, const void *, const void *,
                                       const void *, int, int, int, void *) {
  return -1;
}

extern "C" int mm_blaslt_run_bf16_bias_residual(
    void *, const void *, const void *, const void *, const void *,
    int, int, int, void *) {
  return -1;
}

extern "C" int mm_blaslt_run_bf16_bias_gelu_bf16d(
    void *, const void *, const void *, const void *,
    int, int, int, void *) {
  return -1;
}

extern "C" int mm_blaslt_run_bf16_bias_bf16d(
    void *, const void *, const void *, const void *,
    int, int, int, void *) {
  return -1;
}

extern "C" void mm_blaslt_destroy(void) {}

#else

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include <dlfcn.h>

/* rocew exports HIP entry points as function-pointer variables. A direct HIP
 * call from this C++ translation unit can bind to that variable as executable
 * code. Resolve the runtime functions from their library instead. */
static hipError_t (*bridge_hip_malloc)(void **, size_t) = nullptr;
static hipError_t (*bridge_hip_free)(void *) = nullptr;
static const char *(*bridge_hip_error_string)(hipError_t) = nullptr;

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
                   __LINE__, bridge_hip_error_string(_err));                   \
      return -1;                                                               \
    }                                                                          \
  } while (0)

namespace {

struct Plan {
  hipblasLtMatmulDesc_t matmul = nullptr;
  hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr, d = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  size_t workspace_size = 0;
  /* hipBLASLt workspaces are scratch, not immutable plan state.  Keep one
   * allocation per stream so a target stream and a sidecar stream can issue
   * the same shape concurrently without racing the old shape-global buffer. */
  struct StreamWorkspace {
    void *ptr = nullptr;
  };
  std::unordered_map<uintptr_t, StreamWorkspace> workspaces;
  bool valid = false;
};

struct ShapeKey {
  int M, N, K, batch;
  int flags;  /* bit0: bias, bit1: gelu+bf16-D variant */
  bool operator==(const ShapeKey &o) const noexcept {
    return M == o.M && N == o.N && K == o.K && batch == o.batch && flags == o.flags;
  }
};

struct ShapeKeyHash {
  size_t operator()(const ShapeKey &k) const noexcept {
    size_t h = static_cast<size_t>(k.M) * 0x9E3779B185EBCA87ull;
    h ^= static_cast<size_t>(k.N) * 0xC2B2AE3D27D4EB4Full;
    h ^= static_cast<size_t>(k.K) * 0x165667B19E3779F9ull;
    h ^= static_cast<size_t>(k.batch) * 0x27D4EB2F165667C5ull;
    h ^= static_cast<size_t>(k.flags) * 0x94D049BB133111EBull;
    return h;
  }
};

/* Pinned algo index override per shape. Populated from MM_BLASLT_ALGO_PINS env var. */
struct ShapeKeyInt {
  int M, N, K, flags;
  bool operator==(const ShapeKeyInt &o) const noexcept {
    return M == o.M && N == o.N && K == o.K && flags == o.flags;
  }
};
struct ShapeKeyIntHash {
  size_t operator()(const ShapeKeyInt &k) const noexcept {
    size_t h = static_cast<size_t>(k.M) * 0x9E3779B185EBCA87ull;
    h ^= static_cast<size_t>(k.N) * 0xC2B2AE3D27D4EB4Full;
    h ^= static_cast<size_t>(k.K) * 0x165667B19E3779F9ull;
    h ^= static_cast<size_t>(k.flags) * 0x94D049BB133111EBull;
    return h;
  }
};
static std::unordered_map<ShapeKeyInt, int, ShapeKeyIntHash> g_pinned_algos;

/* Parse MM_BLASLT_ALGO_PINS env var.
 * Format: "MxNxK:algo,MxNxK:algo,..."  e.g. "1024x4096x4096:73624,256x4096x4096:4104" */
static void parse_pinned_algos() {
  const char *pins = std::getenv("MM_BLASLT_ALGO_PINS");
  if (!pins) return;
  const char *p = pins;
  while (*p) {
    int M = 0, N = 0, K = 0, algo = 0;
    int n = 0;
    if (std::sscanf(p, "%dx%dx%d:%d%n", &M, &N, &K, &algo, &n) >= 4 && n > 0) {
      ShapeKeyInt key{M, N, K, 0};
      g_pinned_algos[key] = algo;
      p += n;
      if (*p == ',') ++p;
    } else {
      break;
    }
  }
  /* verbose count printed by mm_blaslt_init after g_state is available */
}

/* Env-var MM_BLASLT_ALGO_INDEX: force all shapes to use this algo index.
 * Set to 0 to disable. Negative or unset = use heuristic. */
static int g_force_algo_index = -1;

/* Env-var MM_BLASLT_LIST_ALGOS: set to 1 to print all returned algos per shape.
 * This does NOT run the algo — just queries the heuristic and prints indices. */
static int g_list_algos = 0;

struct State {
  hipblasLtHandle_t handle = nullptr;
  hipblasHandle_t hipblas = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;
  /* Keep Plan addresses stable after insertion.  Callers release the plan
   * lock before enqueue so a later shape miss must not invalidate an in-flight
   * descriptor by rehashing this cache. */
  std::unordered_map<ShapeKey, std::unique_ptr<Plan>, ShapeKeyHash> plans;
  bool initialized = false;
  int verbose = 0;
};

State g_state;
/* Plan creation touches the shared hipBLASLt handle and unordered-map cache.
 * Hold this lock only for initialization/cache insertion; execution remains
 * unlocked so independent HIP streams can overlap. */
std::mutex g_plan_mutex;
/* Target and sidecar streams may request the same plan concurrently.  Keep
 * only the lazy scratch-map mutation serialized; matmul launches remain on
 * their caller-owned streams and therefore retain overlap. */
std::mutex g_workspace_mutex;
/* hipBLAS (the F16 fallback) stores its stream on the shared handle.  Protect
 * only stream selection plus enqueue; the lock is released before execution
 * completes, so independent device work still overlaps. */
std::mutex g_hipblas_mutex;
/* Bias/epilogue pointers live in the cached matmul descriptor.  Serialize the
 * attribute update with its enqueue so concurrent streams cannot submit with
 * another caller's pointer; the lock is released before device execution. */
std::mutex g_descriptor_mutex;

void destroy_plan(Plan &p) {
  {
    std::lock_guard<std::mutex> lock(g_workspace_mutex);
    for (auto &entry : p.workspaces) {
      if (entry.second.ptr)
        (void)bridge_hip_free(entry.second.ptr);
    }
    p.workspaces.clear();
  }
  if (p.d) hipblasLtMatrixLayoutDestroy(p.d);
  if (p.c) hipblasLtMatrixLayoutDestroy(p.c);
  if (p.b) hipblasLtMatrixLayoutDestroy(p.b);
  if (p.a) hipblasLtMatrixLayoutDestroy(p.a);
  if (p.matmul) hipblasLtMatmulDescDestroy(p.matmul);
  p = Plan{};
}

/* Initialization can fail after one or more library objects have already
 * been created.  Keep rollback in one place so optional bridge setup never
 * leaks a handle and makes a later retry observe stale state. */
void destroy_handles() {
  if (g_state.pref) {
    hipblasLtMatmulPreferenceDestroy(g_state.pref);
    g_state.pref = nullptr;
  }
  if (g_state.handle) {
    hipblasLtDestroy(g_state.handle);
    g_state.handle = nullptr;
  }
  if (g_state.hipblas) {
    hipblasDestroy(g_state.hipblas);
    g_state.hipblas = nullptr;
  }
  g_state.initialized = false;
}

int build_plan(int M, int N, int K, int batch, int flags, Plan &p) {
  if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) {
    std::fprintf(stderr,
                 "[mm_blaslt] rejecting invalid shape M=%d N=%d K=%d batch=%d\n",
                 M, N, K, batch);
    return -1;
  }
  bool with_bias = (flags & 1) != 0;
  bool gelu_bf16d = (flags & 2) != 0;
  bool bias_bf16d = (flags & 4) != 0;
  bool f32_io = (flags & 8) != 0;
  bool f16_io = (flags & 16) != 0;
  HBLT_RET(hipblasLtMatmulDescCreate(&p.matmul, HIPBLAS_COMPUTE_32F,
                                     HIP_R_32F));
  hipblasOperation_t trans_a = HIPBLAS_OP_T;
  hipblasOperation_t trans_b = HIPBLAS_OP_N;
  HBLT_RET(hipblasLtMatmulDescSetAttribute(
      p.matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  HBLT_RET(hipblasLtMatmulDescSetAttribute(
      p.matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
  if (with_bias || gelu_bf16d) {
    hipblasLtEpilogue_t ep = gelu_bf16d ? HIPBLASLT_EPILOGUE_GELU_BIAS
                                         : HIPBLASLT_EPILOGUE_BIAS;
    HBLT_RET(hipblasLtMatmulDescSetAttribute(
        p.matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));
    hipDataType bias_dt = HIP_R_32F;
    HBLT_RET(hipblasLtMatmulDescSetAttribute(
        p.matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dt,
        sizeof(bias_dt)));
  }

  hipDataType d_dt = (gelu_bf16d || bias_bf16d) ? HIP_R_16BF : HIP_R_32F;
  hipDataType io_dt = f32_io ? HIP_R_32F : (f16_io ? HIP_R_16F : HIP_R_16BF);
  /* W [N,K] row-major == [K,N] col-major with op=T */
  HBLT_RET(hipblasLtMatrixLayoutCreate(&p.a, io_dt, K, N, K));
  /* X [M,K] row-major == [K,M] col-major with op=N */
  HBLT_RET(hipblasLtMatrixLayoutCreate(&p.b, io_dt, K, M, K));
  /* Y [M,N] D-type row-major == [N,M] col-major */
  HBLT_RET(hipblasLtMatrixLayoutCreate(&p.c, d_dt, N, M, N));
  HBLT_RET(hipblasLtMatrixLayoutCreate(&p.d, d_dt, N, M, N));
  if (batch > 1) {
    int32_t bc = batch;
    int64_t sa = (int64_t)N * K, sb = (int64_t)M * K, sy = (int64_t)M * N;
    HBLT_RET(hipblasLtMatrixLayoutSetAttribute(p.a, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    HBLT_RET(hipblasLtMatrixLayoutSetAttribute(p.b, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    HBLT_RET(hipblasLtMatrixLayoutSetAttribute(p.c, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    HBLT_RET(hipblasLtMatrixLayoutSetAttribute(p.d, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    HBLT_RET(hipblasLtMatrixLayoutSetAttribute(p.a, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof(sa)));
    HBLT_RET(hipblasLtMatrixLayoutSetAttribute(p.b, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sb, sizeof(sb)));
    HBLT_RET(hipblasLtMatrixLayoutSetAttribute(p.c, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sy, sizeof(sy)));
    HBLT_RET(hipblasLtMatrixLayoutSetAttribute(p.d, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sy, sizeof(sy)));
  }

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

  /* Pick best algo (first valid by default). Supports three override mechanisms:
   * 1. MM_BLASLT_ALGO_INDEX=N  — force N for all shapes (for quick benchmarking).
   * 2. MM_BLASLT_ALGO_PINS="MxNxK:algo,..."  — shape-specific pins.
   * 3. MM_BLASLT_LIST_ALGOS=1  — print all heuristic algos (manual selection aid).
   *
   * NOTE: inline sweep (benchmark all algos → pick fastest) segfaults on gfx1201
   * due to a hipBLASLt driver bug in back-to-back matmul. Pinning via env var
   * avoids this. See also the self-owned WMMA GEMM path which is the long-term
   * replacement for hipBLASLt on gfx1201. */
  int desired_algo_idx = -1;
  if (g_force_algo_index >= 0) {
    desired_algo_idx = g_force_algo_index;
  } else {
    ShapeKeyInt sk{M, N, K, flags};
    auto it = g_pinned_algos.find(sk);
    if (it != g_pinned_algos.end()) {
      desired_algo_idx = it->second;
    }
  }

  int best = -1;
  for (int i = 0; i < returned; ++i) {
    if (results[i].state != HIPBLAS_STATUS_SUCCESS) continue;
    if (g_list_algos) {
      std::fprintf(stderr,
          "[mm_blaslt]  M=%d N=%d K=%d flags=%d  algo[%d/%d] idx=%d ws=%zu\n",
          M, N, K, flags, i, returned,
          hipblaslt_ext::getIndexFromAlgo(results[i].algo),
          results[i].workspaceSize);
    }
    if (desired_algo_idx >= 0) {
      int idx = hipblaslt_ext::getIndexFromAlgo(results[i].algo);
      if (idx == desired_algo_idx) { best = i; break; }
    } else if (best < 0) {
      best = i;  /* first valid, no pin requested */
    }
  }
  if (best < 0) {
    if (g_list_algos) {
      std::fprintf(stderr, "[mm_blaslt]  no match for pinned algo %d; falling back\n",
                   desired_algo_idx);
    }
    /* Fallback: first valid algo */
    for (int i = 0; i < returned; ++i) {
      if (results[i].state == HIPBLAS_STATUS_SUCCESS) { best = i; break; }
    }
  }
  if (best < 0) {
    std::fprintf(stderr, "[mm_blaslt] no successful algo for M=%d N=%d K=%d\n",
                 M, N, K);
    return -1;
  }

  p.algo = results[best].algo;
  p.workspace_size = results[best].workspaceSize;
  p.valid = true;
  if (g_state.verbose) {
    std::fprintf(stderr,
                 "[mm_blaslt] cached plan M=%d N=%d K=%d algo_idx=%d ws=%zu\n",
                 M, N, K, hipblaslt_ext::getIndexFromAlgo(p.algo),
                 p.workspace_size);
  }
  return 0;
}

static Plan *get_cached_plan(int M, int N, int K, int batch, int flags) {
  ShapeKey key{M, N, K, batch, flags};
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  auto it = g_state.plans.find(key);
  if (it != g_state.plans.end()) return it->second.get();
  auto p = std::make_unique<Plan>();
  if (build_plan(M, N, K, batch, flags, *p) != 0) {
    destroy_plan(*p);
    return nullptr;
  }
  auto inserted = g_state.plans.emplace(key, std::move(p));
  return inserted.first->second.get();
}

static int plan_workspace(Plan &p, hipStream_t stream, void **workspace) {
  *workspace = nullptr;
  if (p.workspace_size == 0) return 0;
  const uintptr_t key = reinterpret_cast<uintptr_t>(stream);
  std::lock_guard<std::mutex> lock(g_workspace_mutex);
  auto it = p.workspaces.find(key);
  if (it == p.workspaces.end()) {
    Plan::StreamWorkspace ws;
    hipError_t err = bridge_hip_malloc(&ws.ptr, p.workspace_size);
    if (err != hipSuccess) {
      std::fprintf(stderr, "[mm_blaslt] workspace allocation failed (%zu bytes, stream=%p): %s\n",
                   p.workspace_size, static_cast<void *>(stream),
                   bridge_hip_error_string(err));
      return -1;
    }
    it = p.workspaces.emplace(key, ws).first;
  }
  *workspace = it->second.ptr;
  return 0;
}

}  // namespace

extern "C" int mm_blaslt_init(void) {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  if (g_state.initialized) return 0;
  static void *runtime = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
  if (!runtime) {
    std::fprintf(stderr, "[mm_blaslt] cannot load HIP runtime: %s\n", dlerror());
    return -1;
  }
  bridge_hip_malloc = reinterpret_cast<decltype(bridge_hip_malloc)>(dlsym(runtime, "hipMalloc"));
  bridge_hip_free = reinterpret_cast<decltype(bridge_hip_free)>(dlsym(runtime, "hipFree"));
  bridge_hip_error_string = reinterpret_cast<decltype(bridge_hip_error_string)>(dlsym(runtime, "hipGetErrorString"));
  if (!bridge_hip_malloc || !bridge_hip_free || !bridge_hip_error_string) {
    std::fprintf(stderr, "[mm_blaslt] HIP allocation entry points unavailable\n");
    return -1;
  }
  auto init_failure = [&](const char *what, hipblasStatus_t status) {
    std::fprintf(stderr, "[mm_blaslt] %s failed status=%d\n", what,
                 static_cast<int>(status));
    destroy_handles();
    return -1;
  };
  hipblasStatus_t status = hipblasCreate(&g_state.hipblas);
  if (status != HIPBLAS_STATUS_SUCCESS)
    return init_failure("hipblasCreate", status);
  status = hipblasLtCreate(&g_state.handle);
  if (status != HIPBLAS_STATUS_SUCCESS)
    return init_failure("hipblasLtCreate", status);
  status = hipblasLtMatmulPreferenceCreate(&g_state.pref);
  if (status != HIPBLAS_STATUS_SUCCESS)
    return init_failure("hipblasLtMatmulPreferenceCreate", status);
  uint64_t max_ws = 256ull << 20;
  status = hipblasLtMatmulPreferenceSetAttribute(
      g_state.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
      sizeof(max_ws));
  if (status != HIPBLAS_STATUS_SUCCESS)
    return init_failure("hipblasLtMatmulPreferenceSetAttribute", status);
  if (const char *v = std::getenv("MM_BLASLT_VERBOSE")) {
    g_state.verbose = std::atoi(v);
  }
  if (const char *v = std::getenv("MM_BLASLT_ALGO_INDEX")) {
    g_force_algo_index = std::atoi(v);
    if (g_state.verbose) {
      std::fprintf(stderr, "[mm_blaslt] forcing algo index %d for all shapes\n",
                   g_force_algo_index);
    }
  }
  if (const char *v = std::getenv("MM_BLASLT_LIST_ALGOS")) {
    g_list_algos = std::atoi(v) != 0;
  }
  parse_pinned_algos();
  if (g_state.verbose && !g_pinned_algos.empty()) {
    std::fprintf(stderr, "[mm_blaslt] %zu pinned algos loaded\n",
                 g_pinned_algos.size());
  }
  g_state.initialized = true;
  return 0;
}

extern "C" int mm_blaslt_run_bf16(void *d_y_f32, const void *d_w_bf16,
                                  const void *d_x_bf16, int M, int N, int K,
                                  void *stream) {
  return mm_blaslt_run_bf16_bias(d_y_f32, d_w_bf16, d_x_bf16, nullptr,
                                 M, N, K, stream);
}

extern "C" int mm_blaslt_run_bf16_strided_batch(
    void *d_y_f32, const void *d_w_bf16, const void *d_x_bf16,
    int M, int N, int K, int batch_count, void *stream) {
  if (batch_count <= 0) {
    std::fprintf(stderr, "[mm_blaslt] rejecting invalid batch count %d\n",
                 batch_count);
    return -1;
  }
  if (batch_count == 1)
    return mm_blaslt_run_bf16(d_y_f32, d_w_bf16, d_x_bf16, M, N, K, stream);
  if (!g_state.initialized && mm_blaslt_init() != 0) return -1;
  Plan *cached = get_cached_plan(M, N, K, batch_count, 0);
  if (!cached) return -1;
  Plan &p = *cached;
  if (!p.valid) return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  void *workspace = nullptr;
  if (plan_workspace(p, hip_stream, &workspace) != 0) return -1;
  const float alpha = 1.0f, beta = 0.0f;
  HBLT_RET(hipblasLtMatmul(g_state.handle, p.matmul, &alpha,
                           d_w_bf16, p.a, d_x_bf16, p.b, &beta,
                           d_y_f32, p.c, d_y_f32, p.d, &p.algo,
                           workspace, p.workspace_size, hip_stream));
  return 0;
}

extern "C" int mm_blaslt_run_bf16_bias(void *d_y_f32, const void *d_w_bf16,
                                       const void *d_x_bf16,
                                       const void *d_bias_f32,
                                       int M, int N, int K, void *stream) {
  return mm_blaslt_run_bf16_bias_residual(d_y_f32, nullptr, d_w_bf16, d_x_bf16,
                                          d_bias_f32, M, N, K, stream);
}

extern "C" int mm_blaslt_run_f32(void *d_y_f32, const void *d_w_f32,
                                  const void *d_x_f32, int M, int N, int K,
                                  void *stream) {
  if (!g_state.initialized && mm_blaslt_init() != 0) return -1;
  const int flags = 8;
  Plan *cached = get_cached_plan(M, N, K, 1, flags);
  if (!cached) return -1;
  Plan &p = *cached;
  if (!p.valid) return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  void *workspace = nullptr;
  if (plan_workspace(p, hip_stream, &workspace) != 0) return -1;
  const float alpha = 1.0f, beta = 0.0f;
  HBLT_RET(hipblasLtMatmul(g_state.handle, p.matmul, &alpha,
                           d_w_f32, p.a, d_x_f32, p.b, &beta,
                           d_y_f32, p.c, d_y_f32, p.d, &p.algo,
                           workspace, p.workspace_size, hip_stream));
  return 0;
}

extern "C" int mm_blaslt_run_f16(void *d_y_f32, const void *d_w_f16,
                                  const void *d_x_f16, int M, int N, int K,
                                  void *stream) {
  const float alpha = 1.0f, beta = 0.0f;
  if (!g_state.initialized && mm_blaslt_init() != 0) return -1;
  std::lock_guard<std::mutex> lock(g_hipblas_mutex);
  HBLT_RET(hipblasSetStream(g_state.hipblas, static_cast<hipStream_t>(stream)));
  HBLT_RET(hipblasGemmEx(g_state.hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                         N, M, K, &alpha,
                         d_w_f16, HIP_R_16F, K,
                         d_x_f16, HIP_R_16F, K, &beta,
                         d_y_f32, HIP_R_32F, N,
                         HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  return 0;
}

extern "C" int mm_blaslt_run_bf16_bias_residual(
    void *d_y_f32, const void *d_c_f32, const void *d_w_bf16,
    const void *d_x_bf16, const void *d_bias_f32,
    int M, int N, int K, void *stream) {
  if (!g_state.initialized) {
    if (mm_blaslt_init() != 0) return -1;
  }
  bool with_bias = (d_bias_f32 != nullptr);
  int flags = with_bias ? 1 : 0;
  Plan *cached = get_cached_plan(M, N, K, 1, flags);
  if (!cached) return -1;
  Plan &p = *cached;
  if (!p.valid) return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  void *workspace = nullptr;
  if (plan_workspace(p, hip_stream, &workspace) != 0) return -1;

  const float alpha = 1.0f;
  const float beta = (d_c_f32 != nullptr) ? 1.0f : 0.0f;
  const void *c_ptr = (d_c_f32 != nullptr) ? d_c_f32 : d_y_f32;
  if (with_bias) {
    std::lock_guard<std::mutex> lock(g_descriptor_mutex);
    HBLT_RET(hipblasLtMatmulDescSetAttribute(
        p.matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias_f32,
        sizeof(d_bias_f32)));
    HBLT_RET(hipblasLtMatmul(g_state.handle, p.matmul, &alpha, d_w_bf16, p.a,
                             d_x_bf16, p.b, &beta, c_ptr, p.c, d_y_f32, p.d,
                             &p.algo, workspace, p.workspace_size, hip_stream));
  } else {
    HBLT_RET(hipblasLtMatmul(g_state.handle, p.matmul, &alpha, d_w_bf16, p.a,
                             d_x_bf16, p.b, &beta, c_ptr, p.c, d_y_f32, p.d,
                             &p.algo, workspace, p.workspace_size, hip_stream));
  }
  return 0;
}

extern "C" int mm_blaslt_run_bf16_bias_bf16d(
    void *d_y_bf16, const void *d_w_bf16, const void *d_x_bf16,
    const void *d_bias_f32, int M, int N, int K, void *stream) {
  if (!g_state.initialized) {
    if (mm_blaslt_init() != 0) return -1;
  }
  if (d_bias_f32 == nullptr) {
    std::fprintf(stderr, "[mm_blaslt] bias_bf16d requires bias\n");
    return -1;
  }
  int flags = 1 | 4; /* bias + bf16-D */
  Plan *cached = get_cached_plan(M, N, K, 1, flags);
  if (!cached) return -1;
  Plan &p = *cached;
  if (!p.valid) return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  void *workspace = nullptr;
  if (plan_workspace(p, hip_stream, &workspace) != 0) return -1;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  {
    std::lock_guard<std::mutex> lock(g_descriptor_mutex);
    HBLT_RET(hipblasLtMatmulDescSetAttribute(
        p.matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias_f32,
        sizeof(d_bias_f32)));
    HBLT_RET(hipblasLtMatmul(g_state.handle, p.matmul, &alpha, d_w_bf16, p.a,
                             d_x_bf16, p.b, &beta, d_y_bf16, p.c, d_y_bf16, p.d,
                             &p.algo, workspace, p.workspace_size, hip_stream));
  }
  return 0;
}

extern "C" int mm_blaslt_run_bf16_bias_gelu_bf16d(
    void *d_y_bf16, const void *d_w_bf16, const void *d_x_bf16,
    const void *d_bias_f32, int M, int N, int K, void *stream) {
  if (!g_state.initialized) {
    if (mm_blaslt_init() != 0) return -1;
  }
  if (d_bias_f32 == nullptr) {
    std::fprintf(stderr, "[mm_blaslt] gelu_bf16d requires bias\n");
    return -1;
  }
  int flags = 1 | 2; /* bias + gelu+bf16-D */
  Plan *cached = get_cached_plan(M, N, K, 1, flags);
  if (!cached) return -1;
  Plan &p = *cached;
  if (!p.valid) return -1;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  void *workspace = nullptr;
  if (plan_workspace(p, hip_stream, &workspace) != 0) return -1;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  {
    std::lock_guard<std::mutex> lock(g_descriptor_mutex);
    HBLT_RET(hipblasLtMatmulDescSetAttribute(
        p.matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias_f32,
        sizeof(d_bias_f32)));
    HBLT_RET(hipblasLtMatmul(g_state.handle, p.matmul, &alpha, d_w_bf16, p.a,
                             d_x_bf16, p.b, &beta, d_y_bf16, p.c, d_y_bf16, p.d,
                             &p.algo, workspace, p.workspace_size, hip_stream));
  }
  return 0;
}

extern "C" void mm_blaslt_destroy(void) {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  for (auto &kv : g_state.plans) {
    destroy_plan(*kv.second);
  }
  g_state.plans.clear();
  destroy_handles();
}

#endif /* MM_BLASLT_DISABLE */
