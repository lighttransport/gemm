/*
 * bench_vlm_rocblas.cpp - rocBLAS GEMM benchmark for Qwen3.6 vision shapes.
 *
 * Input/output layout matches hip_vision_encoder.c:
 *   X: row-major [M, K]
 *   W: row-major [N, K]
 *   Y: row-major [M, N] = X * W^T
 *
 * rocBLAS is column-major, so the call computes:
 *   Y^T [N, M] = W [N, K] * X^T [K, M]
 * using transposed interpretation of the row-major W buffer.
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

#define ROCBLAS_CHECK(expr)                                                    \
  do {                                                                         \
    rocblas_status _st = (expr);                                               \
    if (_st != rocblas_status_success) {                                       \
      std::fprintf(stderr, "rocBLAS error %s:%d: status=%d\n", __FILE__,      \
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

double time_gemm(rocblas_handle handle,
                 const Shape& s,
                 rocblas_datatype dtype,
                 const void* d_w,
                 const void* d_x,
                 float* d_y,
                 int iters) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // Warm up once outside measured region.
  ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                                rocblas_operation_transpose,
                                rocblas_operation_none,
                                s.n, s.m, s.k,
                                &alpha,
                                d_w, dtype, s.k,
                                d_x, dtype, s.k,
                                &beta,
                                d_y, rocblas_datatype_f32_r, s.n,
                                d_y, rocblas_datatype_f32_r, s.n,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0, 0));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    ROCBLAS_CHECK(rocblas_gemm_ex(handle,
                                  rocblas_operation_transpose,
                                  rocblas_operation_none,
                                  s.n, s.m, s.k,
                                  &alpha,
                                  d_w, dtype, s.k,
                                  d_x, dtype, s.k,
                                  &beta,
                                  d_y, rocblas_datatype_f32_r, s.n,
                                  d_y, rocblas_datatype_f32_r, s.n,
                                  rocblas_datatype_f32_r,
                                  rocblas_gemm_algo_standard,
                                  0, 0));
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
  int iters = 20;
  int device = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) dtype_name = argv[++i];
    else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc) shape_name = argv[++i];
    else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "Usage: %s [--dtype f16|bf16] [--shape all|qkv|attn_out|ffn_up|ffn_down|mm0|mm2] [--iters N] [--device N]\n", argv[0]);
      return 1;
    }
  }
  if (iters < 1) iters = 1;
  const bool bf16 = std::strcmp(dtype_name, "bf16") == 0;
  if (!bf16 && std::strcmp(dtype_name, "f16") != 0) {
    std::fprintf(stderr, "dtype must be f16 or bf16\n");
    return 1;
  }
  const rocblas_datatype dtype = bf16 ? rocblas_datatype_bf16_r : rocblas_datatype_f16_r;

  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  rocblas_handle handle = nullptr;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));
  ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

  std::printf("# RDNA4 VLM rocBLAS GEMM dtype=%s iters=%d device=%s arch=%s peak=195.0 TFLOP/s target80=156.0 TFLOP/s\n",
              dtype_name, iters, prop.name, prop.gcnArchName);
  for (const Shape& s : kShapes) {
    if (!matches(shape_name, s.name)) continue;

    const int64_t x_elems = static_cast<int64_t>(s.m) * s.k;
    const int64_t w_elems = static_cast<int64_t>(s.n) * s.k;
    const int64_t y_elems = static_cast<int64_t>(s.m) * s.n;
    uint16_t* d_x = nullptr;
    uint16_t* d_w = nullptr;
    float* d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, x_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_w, w_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&d_y, y_elems * sizeof(float)));

    if (bf16) {
      init_bf16<<<1024, 256>>>(d_x, x_elems, 0.5f, 0.001f);
      init_bf16<<<1024, 256>>>(d_w, w_elems, 0.25f, 0.002f);
    } else {
      init_f16<<<1024, 256>>>(d_x, x_elems, 0.5f, 0.001f);
      init_f16<<<1024, 256>>>(d_w, w_elems, 0.25f, 0.002f);
    }
    HIP_CHECK(hipDeviceSynchronize());

    const double ms = time_gemm(handle, s, dtype, d_w, d_x, d_y, iters);
    const double flops = 2.0 * static_cast<double>(s.m) * s.n * s.k;
    const double tflops = flops / (ms * 1.0e9);
    const double pct = tflops / kPeakWmma * 100.0;
    std::printf("%-8s M=%5d N=%5d K=%5d ms=%8.4f TFLOP/s=%8.3f peak=%5.1f%% %s\n",
                s.name, s.m, s.n, s.k, ms, tflops, pct,
                tflops >= kPeakWmma * 0.8 ? "PASS80" : "FAIL80");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_y));
  }

  ROCBLAS_CHECK(rocblas_destroy_handle(handle));
  return 0;
}
