#ifndef PIXAL3D_HV15N_GPU_HPP
#define PIXAL3D_HV15N_GPU_HPP
#include "../cublasew.h"
#include "../cuew.h"
#include "host.hpp"
#include <atomic>
#include <chrono>
#include <functional>
namespace hv15n {
struct Gpu;
struct Allocation {
  Gpu *gpu;
  CUdeviceptr pointer = 0;
  size_t bytes;
  Allocation(Gpu *gpu, size_t bytes);
  ~Allocation();
};
struct Tensor {
  std::shared_ptr<Allocation> storage;
  CUdeviceptr pointer = 0;
  std::vector<int> shape;
  size_t count() const { return product(shape); }
  int channels() const { return shape.back(); }
  int rows() const { return int(count() / channels()); }
};
struct Gpu {
  CUcontext context = nullptr;
  CUstream stream = nullptr;
  CUmodule ops = nullptr, mma = nullptr;
  cublasew_context *blas = nullptr;
  size_t allocated = 0, peak = 0, budget;
  uint64_t repo_gemm = 0, blas_gemm = 0, fallback_gemm = 0, attention_calls = 0;
  bool vendor, fallback;
  std::atomic<bool> cancelled{false};
  CUresult deferred_error = CUDA_SUCCESS;
  std::function<bool()> cancel_check;
  std::map<std::string, CUfunction> functions;
  explicit Gpu(int device, int budget_mib, bool vendor, bool fallback);
  ~Gpu();
  void check(CUresult result, const char *operation);
  static std::string compile(const std::string &source);
  static std::string mma_source();
  void poll();
  template <typename... Args>
  void launch(const char *name, int gx, int gy, int gz, int bx, int by,
              int shared, Args... arguments) {
    poll();
    auto it = functions.find(name);
    if (it == functions.end()) {
      CUfunction f;
      check(cuModuleGetFunction(&f, ops, name), name);
      it = functions.emplace(name, f).first;
    }
    void *params[] = {static_cast<void *>(&arguments)...};
    check(cuLaunchKernel(it->second, gx, gy, gz, bx, by, 1, shared, stream,
                         params, nullptr),
          name);
  }
  Tensor empty(const std::vector<int> &shape);
  Tensor upload(const std::vector<float> &data, const std::vector<int> &shape);
  std::vector<float> download(const Tensor &tensor);
  Tensor clone(const Tensor &tensor);
  Tensor rows(const Tensor &tensor, int start, int count);
  Tensor columns(const Tensor &tensor, int start, int count);
  Tensor concat(const Tensor &first, const Tensor &second);
  Tensor weight(Weights &weights, const std::string &name);
  Tensor linear(Weights &weights, const std::string &prefix,
                const Tensor &input, bool precise = false);
  Tensor matmul(const Tensor &input, const Tensor &weight,
                bool precise = false);
  Tensor norm(Weights &weights, const std::string &prefix, const Tensor &input,
              int mode = 0, float eps = 1.e-6f);
  Tensor bare_norm(const Tensor &input, int mode = 0, float eps = 1.e-6f);
  Tensor op(const Tensor &input, int mode, const Tensor *second = nullptr,
            const Tensor *bias = nullptr, float scale = 1.f,
            float offset = 0.f);
  Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, int heads,
                   int kvheads, int mask = 0, int frame_hw = 1,
                   bool precise = false, const Tensor *bias = nullptr,
                   float scale = 0.f);
  void rotary(Tensor &tensor, int heads, int kind, int height = 1,
              int width = 1, float theta = 1000000.f);
  Tensor conv(Weights &weights, const std::string &prefix, const Tensor &input,
              bool causal = true);
  Tensor channel_map(const Tensor &input, int channels, bool mean);
  Tensor mean(const Tensor &input);
  void dump(const Tensor &tensor, const fs::path &directory,
            const std::string &name, bool video = false);
  std::string metrics() const;
};
} // namespace hv15n
#endif
