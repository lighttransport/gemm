/*
 * bench_vlm_hipblaslt.cpp - hipBLASLt algorithm tuner for Qwen3.6 vision GEMMs.
 *
 * Layout matches hip_vision_encoder.c:
 *   X row-major [M,K], W row-major [N,K], Y row-major [M,N] = X * W^T.
 * hipBLASLt sees this as column-major Y^T[N,M] = W[N,K] * X^T[K,M].
 */

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,       \
                   hipGetErrorString(_err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define HBLT_CHECK(expr)                                                       \
  do {                                                                         \
    hipblasStatus_t _st = (expr);                                              \
    if (_st != HIPBLAS_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "hipBLASLt error %s:%d: status=%d\n", __FILE__,    \
                   __LINE__, static_cast<int>(_st));                           \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr double kPeakWmma = 195.0;

struct Shape {
  const char* name;
  int m;
  int n;
  int k;
};

const Shape kShapes[] = {
    {"qkv",      4096, 3456, 1152},
    {"attn_out", 4096, 1152, 1152},
    {"ffn_up",   4096, 4304, 1152},
    {"ffn_down", 4096, 1152, 4304},
    {"mm0",      1024, 4608, 4608},
    {"mm2",      1024, 5120, 4608},
};

__global__ void init_f16(uint16_t* p, int64_t n, float base, float scale) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    float v = base + static_cast<float>(i & 15) * scale;
    _Float16 hv = static_cast<_Float16>(v);
    p[i] = *reinterpret_cast<uint16_t*>(&hv);
  }
}

__global__ void init_bf16(uint16_t* p, int64_t n, float base, float scale) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    float v = base + static_cast<float>(i & 15) * scale;
    uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
    uint32_t lsb = (bits >> 16) & 1u;
    p[i] = static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
  }
}

bool matches(const char* want, const char* name) {
  return std::strcmp(want, "all") == 0 || std::strcmp(want, name) == 0;
}

int algo_index(const hipblasLtMatmulAlgo_t& algo) {
  hipblasLtMatmulAlgo_t copy = algo;
  return hipblaslt_ext::getIndexFromAlgo(copy);
}

std::string solution_name(hipblasLtHandle_t handle, const hipblasLtMatmulAlgo_t& algo) {
  hipblasLtMatmulAlgo_t copy = algo;
  try {
    return hipblaslt_ext::getSolutionNameFromAlgo(handle, copy);
  } catch (...) {
    return {};
  }
}

std::string kernel_name(hipblasLtHandle_t handle, const hipblasLtMatmulAlgo_t& algo) {
  hipblasLtMatmulAlgo_t copy = algo;
  try {
    return hipblaslt_ext::getKernelNameFromAlgo(handle, copy);
  } catch (...) {
    return {};
  }
}

struct Problem {
  hipblasLtMatmulDesc_t matmul = nullptr;
  hipblasLtMatrixLayout_t a = nullptr;
  hipblasLtMatrixLayout_t b = nullptr;
  hipblasLtMatrixLayout_t c = nullptr;
  hipblasLtMatrixLayout_t d = nullptr;
  hipblasLtMatmulPreference_t pref = nullptr;

  ~Problem() {
    if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    if (d) hipblasLtMatrixLayoutDestroy(d);
    if (c) hipblasLtMatrixLayoutDestroy(c);
    if (b) hipblasLtMatrixLayoutDestroy(b);
    if (a) hipblasLtMatrixLayoutDestroy(a);
    if (matmul) hipblasLtMatmulDescDestroy(matmul);
  }
};

Problem make_problem(const Shape& s, hipDataType dtype, size_t max_workspace,
                     const char* epilogue_name, const void* bias) {
  Problem p;
  HBLT_CHECK(hipblasLtMatmulDescCreate(&p.matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t trans_a = HIPBLAS_OP_T;
  hipblasOperation_t trans_b = HIPBLAS_OP_N;
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(p.matmul, HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &trans_a, sizeof(trans_a)));
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(p.matmul, HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &trans_b, sizeof(trans_b)));
  if (std::strcmp(epilogue_name, "bias") == 0 || std::strcmp(epilogue_name, "gelu_bias") == 0) {
    hipblasLtEpilogue_t epi = std::strcmp(epilogue_name, "gelu_bias") == 0
                                  ? HIPBLASLT_EPILOGUE_GELU_BIAS
                                  : HIPBLASLT_EPILOGUE_BIAS;
    hipDataType bias_type = HIP_R_32F;
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(p.matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                               &epi, sizeof(epi)));
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(p.matmul, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                               &bias, sizeof(bias)));
    HBLT_CHECK(hipblasLtMatmulDescSetAttribute(p.matmul, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                               &bias_type, sizeof(bias_type)));
  }

  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&p.a, dtype, s.k, s.n, s.k));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&p.b, dtype, s.k, s.m, s.k));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&p.c, HIP_R_32F, s.n, s.m, s.n));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&p.d, HIP_R_32F, s.n, s.m, s.n));

  HBLT_CHECK(hipblasLtMatmulPreferenceCreate(&p.pref));
  uint64_t ws = max_workspace;
  HBLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
      p.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  return p;
}

double time_algo(hipblasLtHandle_t handle,
                 const Problem& p,
                 const hipblasLtMatmulAlgo_t* algo,
                 const void* d_w,
                 const void* d_x,
                 float* d_y,
                 void* workspace,
                 size_t workspace_size,
                 int iters) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  HBLT_CHECK(hipblasLtMatmul(handle, p.matmul,
                             &alpha,
                             d_w, p.a,
                             d_x, p.b,
                             &beta,
                             d_y, p.c,
                             d_y, p.d,
                             algo,
                             workspace, workspace_size,
                             0));
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    HBLT_CHECK(hipblasLtMatmul(handle, p.matmul,
                               &alpha,
                               d_w, p.a,
                               d_x, p.b,
                               &beta,
                               d_y, p.c,
                               d_y, p.d,
                               algo,
                               workspace, workspace_size,
                               0));
  }
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));
  HIP_CHECK(hipGetLastError());
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return static_cast<double>(ms) / iters;
}

}  // namespace

int main(int argc, char** argv) {
  const char* dtype_name = "f16";
  const char* shape_name = "all";
  const char* epilogue_name = "none";
  int iters = 10;
  int requested = 64;
  size_t max_workspace = 256ull << 20;
  int device = 0;
  int target_algo = -1;
  bool print_algos = false;
  bool list_only = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) dtype_name = argv[++i];
    else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc) shape_name = argv[++i];
    else if (std::strcmp(argv[i], "--epilogue") == 0 && i + 1 < argc) epilogue_name = argv[++i];
    else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--algos") == 0 && i + 1 < argc) requested = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--algo-index") == 0 && i + 1 < argc) target_algo = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--workspace-mb") == 0 && i + 1 < argc) max_workspace = static_cast<size_t>(std::atoll(argv[++i])) << 20;
    else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--print-algos") == 0) print_algos = true;
    else if (std::strcmp(argv[i], "--list-only") == 0) {
      print_algos = true;
      list_only = true;
    }
    else {
      std::fprintf(stderr, "Usage: %s [--dtype f16|bf16] [--shape all|qkv|attn_out|ffn_up|ffn_down|mm0|mm2] [--epilogue none|bias|gelu_bias] [--iters N] [--algos N] [--algo-index N] [--workspace-mb N] [--print-algos] [--list-only]\n", argv[0]);
      return 1;
    }
  }
  if (iters < 1) iters = 1;
  if (requested < 1) requested = 1;
  requested = std::min(requested, 256);

  const bool bf16 = std::strcmp(dtype_name, "bf16") == 0;
  if (!bf16 && std::strcmp(dtype_name, "f16") != 0) {
    std::fprintf(stderr, "dtype must be f16 or bf16\n");
    return 1;
  }
  if (std::strcmp(epilogue_name, "none") != 0 &&
      std::strcmp(epilogue_name, "bias") != 0 &&
      std::strcmp(epilogue_name, "gelu_bias") != 0) {
    std::fprintf(stderr, "epilogue must be none, bias, or gelu_bias\n");
    return 1;
  }
  hipDataType dtype = bf16 ? HIP_R_16BF : HIP_R_16F;

  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  hipblasLtHandle_t handle = nullptr;
  HBLT_CHECK(hipblasLtCreate(&handle));

  void* workspace = nullptr;
  if (max_workspace > 0) HIP_CHECK(hipMalloc(&workspace, max_workspace));

  std::printf("# RDNA4 VLM hipBLASLt tuner dtype=%s epilogue=%s iters=%d algos=%d target_algo=%d workspace=%zu MiB device=%s arch=%s peak=195.0 TFLOP/s target80=156.0 TFLOP/s\n",
              dtype_name, epilogue_name, iters, requested, target_algo,
              max_workspace >> 20, prop.name, prop.gcnArchName);

  for (const Shape& s : kShapes) {
    if (!matches(shape_name, s.name)) continue;

    const int64_t x_elems = static_cast<int64_t>(s.m) * s.k;
    const int64_t w_elems = static_cast<int64_t>(s.n) * s.k;
    const int64_t y_elems = static_cast<int64_t>(s.m) * s.n;
    uint16_t* d_x = nullptr;
    uint16_t* d_w = nullptr;
    float* d_y = nullptr;
    float* d_bias = nullptr;
    HIP_CHECK(hipMalloc(&d_x, x_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_w, w_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_y, y_elems * sizeof(float)));
    if (std::strcmp(epilogue_name, "none") != 0) {
      HIP_CHECK(hipMalloc(&d_bias, static_cast<size_t>(s.n) * sizeof(float)));
    }

    if (bf16) {
      init_bf16<<<1024, 256>>>(d_x, x_elems, 0.5f, 0.001f);
      init_bf16<<<1024, 256>>>(d_w, w_elems, 0.25f, 0.002f);
    } else {
      init_f16<<<1024, 256>>>(d_x, x_elems, 0.5f, 0.001f);
      init_f16<<<1024, 256>>>(d_w, w_elems, 0.25f, 0.002f);
    }
    if (d_bias) {
      HIP_CHECK(hipMemset(d_bias, 0, static_cast<size_t>(s.n) * sizeof(float)));
    }
    HIP_CHECK(hipDeviceSynchronize());

    Problem p = make_problem(s, dtype, max_workspace, epilogue_name, d_bias);
    std::vector<hipblasLtMatmulHeuristicResult_t> results(requested);
    int returned = 0;
    hipblasStatus_t hst = hipblasLtMatmulAlgoGetHeuristic(
        handle, p.matmul, p.a, p.b, p.c, p.d, p.pref,
        requested, results.data(), &returned);
    if (hst != HIPBLAS_STATUS_SUCCESS || returned == 0) {
      std::printf("%-8s M=%5d N=%5d K=%5d no_heuristic status=%d returned=%d\n",
                  s.name, s.m, s.n, s.k, static_cast<int>(hst), returned);
      HIP_CHECK(hipFree(d_x));
      HIP_CHECK(hipFree(d_w));
      HIP_CHECK(hipFree(d_y));
      if (d_bias) HIP_CHECK(hipFree(d_bias));
      continue;
    }

    double best_ms = 1.0e30;
    int best_idx = -1;
    int best_return_idx = -1;
    size_t best_ws = 0;
    int tested = 0;
    std::string best_solution;
    std::string best_kernel;
    for (int i = 0; i < returned; ++i) {
      const int idx = algo_index(results[i].algo);
      const bool target_match = target_algo < 0 || idx == target_algo;
      const std::string sol = solution_name(handle, results[i].algo);
      const std::string kern = kernel_name(handle, results[i].algo);
      if (print_algos) {
        std::printf("algo %-8s ret=%3d idx=%6d state=%2d ws=%10zu solution=%s kernel=%s\n",
                    s.name, i, idx, static_cast<int>(results[i].state),
                    results[i].workspaceSize,
                    sol.empty() ? "<unknown>" : sol.c_str(),
                    kern.empty() ? "<unknown>" : kern.c_str());
      }
      if (!target_match) continue;
      if (results[i].state != HIPBLAS_STATUS_SUCCESS) continue;
      if (results[i].workspaceSize > max_workspace) continue;
      if (list_only) continue;
      double ms = 1.0e30;
      bool ok = true;
      try {
        ms = time_algo(handle, p, &results[i].algo, d_w, d_x, d_y,
                       workspace, results[i].workspaceSize, iters);
      } catch (...) {
        ok = false;
      }
      if (!ok) continue;
      tested++;
      if (ms < best_ms) {
        best_ms = ms;
        best_idx = idx;
        best_return_idx = i;
        best_ws = results[i].workspaceSize;
        best_solution = sol;
        best_kernel = kern;
      }
    }

    if (list_only) {
      std::printf("%-8s M=%5d N=%5d K=%5d returned=%d list_only\n",
                  s.name, s.m, s.n, s.k, returned);
    } else if (best_return_idx < 0) {
      std::printf("%-8s M=%5d N=%5d K=%5d returned=%d tested=%d no_supported_algo\n",
                  s.name, s.m, s.n, s.k, returned, tested);
    } else {
      const double flops = 2.0 * static_cast<double>(s.m) * s.n * s.k;
      const double tflops = flops / (best_ms * 1.0e9);
      const double pct = tflops / kPeakWmma * 100.0;
      std::printf("%-8s M=%5d N=%5d K=%5d returned=%3d tested=%3d algo=%5d ret=%3d ws=%8zu ms=%8.4f TFLOP/s=%8.3f peak=%5.1f%% %s\n",
                  s.name, s.m, s.n, s.k, returned, tested, best_idx, best_return_idx,
                  best_ws, best_ms, tflops, pct,
                  tflops >= kPeakWmma * 0.8 ? "PASS80" : "FAIL80");
      std::printf("best %-8s solution=%s kernel=%s\n",
                  s.name,
                  best_solution.empty() ? "<unknown>" : best_solution.c_str(),
                  best_kernel.empty() ? "<unknown>" : best_kernel.c_str());
    }

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_y));
    if (d_bias) HIP_CHECK(hipFree(d_bias));
  }

  if (workspace) HIP_CHECK(hipFree(workspace));
  HBLT_CHECK(hipblasLtDestroy(handle));
  return 0;
}
