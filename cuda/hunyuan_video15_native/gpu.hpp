#ifndef PIXAL3D_HV15N_GPU_HPP
#define PIXAL3D_HV15N_GPU_HPP
#include "../cublasew.h"
#include "../cuew.h"
#include "host.hpp"
#include <atomic>
#include <array>
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
  int element_bytes = 4;
  int packed_heads = 0;
  size_t count() const { return product(shape); }
  size_t bytes() const { return count() * element_bytes; }
  int channels() const { return shape.back(); }
  int rows() const { return int(count() / channels()); }
};
struct Gpu {
  CUcontext context = nullptr;
  CUstream stream = nullptr, copy_stream = nullptr;
  struct Staging { void *data; size_t bytes; CUevent ready; };
  std::vector<Staging> staging;
  size_t pinned_bytes = 0;
  std::map<int, CUevent> staged_blocks;
  uint64_t prefetch_bytes = 0;
  CUmodule ops = nullptr, mma = nullptr, flash = nullptr;
  cublasew_context *blas = nullptr;
  size_t allocated = 0, peak = 0, budget;
  uint64_t repo_gemm = 0, blas_gemm = 0, fallback_gemm = 0, attention_calls = 0;
  bool vendor, fallback;
  bool optimized;
  std::multimap<size_t, CUdeviceptr> free_buffers;
  std::map<std::string, Tensor> packed_weights;
  size_t cached_weight_bytes = 0;
  std::string active_weight_file;
  uint64_t allocations = 0, buffer_reuses = 0, upload_bytes = 0, weight_hits = 0;
  uint64_t flash_calls = 0, gemm_v7_calls = 0, conv_chunks = 0;
  uint64_t ieee_tiled_calls = 0;
  std::atomic<bool> cancelled{false};
  CUresult deferred_error = CUDA_SUCCESS;
  std::function<bool()> cancel_check;
  std::map<std::string, CUfunction> functions;
  explicit Gpu(int device, int budget_mib, bool vendor, bool fallback, bool optimized = true);
  ~Gpu();
  void check(CUresult result, const char *operation);
  static std::string compile(const std::string &source);
  static std::string mma_source();
  void trim_pool();
  void clear_weights();
  void upload_staged(const Tensor &destination, const void *source, CUstream target);
  void prefetch_block(Weights &weights, int index);
  void wait_block(int index);
  void release_block(Weights &weights, int index);
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
  Tensor empty_half(const std::vector<int> &shape);
  Tensor upload(const std::vector<float> &data, const std::vector<int> &shape);
  std::vector<float> download(const Tensor &tensor);
  Tensor clone(const Tensor &tensor);
  Tensor rows(const Tensor &tensor, int start, int count);
  Tensor columns(const Tensor &tensor, int start, int count);
  Tensor concat(const Tensor &first, const Tensor &second);
  Tensor weight(Weights &weights, const std::string &name);
  Tensor weight_half(Weights &weights, const std::string &name);
  Tensor half(const Tensor &input);
  Tensor full(const Tensor &input);
  Tensor linear(Weights &weights, const std::string &prefix,
                const Tensor &input, bool precise = false);
  Tensor matmul(const Tensor &input, const Tensor &weight,
                bool precise = false, const Tensor *destination = nullptr);
  Tensor norm(Weights &weights, const std::string &prefix, const Tensor &input,
              int mode = 0, float eps = 1.e-6f);
  Tensor norm_silu(Weights &weights, const std::string &prefix, const Tensor &input);
  Tensor modulate(const Tensor &input, const Tensor &modulation, int start);
  Tensor gated(const Tensor &input, const Tensor &delta, const Tensor &modulation, int start);
  Tensor activate(Tensor input, int mode);
  std::array<Tensor, 3> qkv_heads(Weights &weights, const std::string &prefix,
                                const Tensor &qkv, int height, int width, bool image);
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
