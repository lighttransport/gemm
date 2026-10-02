#include "gpu.hpp"
#include "../gemm/cuda_gemm_ptx_kernels.h"
#include "gemm_native.hpp"
#include "kernels.hpp"
#include <cstring>
#include <sstream>
namespace hv15n {
Allocation::Allocation(Gpu *owner, size_t size) : gpu(owner), bytes(size) {
  require(bytes <= gpu->budget - gpu->allocated,
          "GPU allocation exceeds requested budget");
  auto status = cuMemAlloc(&pointer, bytes);
  if (status != CUDA_SUCCESS) {
    size_t available = 0, total = 0;
    cuMemGetInfo(&available, &total);
    gpu->check(
        status,
        ("allocate " + std::to_string(bytes / (1024 * 1024)) +
         " MiB (managed=" + std::to_string(gpu->allocated / (1024 * 1024)) +
         " MiB, device_free=" + std::to_string(available / (1024 * 1024)) +
         " MiB)")
            .c_str());
  }
  gpu->allocated += bytes;
  gpu->peak = std::max(gpu->peak, gpu->allocated);
}
Allocation::~Allocation() {
  if (pointer) {
    auto sync_status = cuStreamSynchronize(gpu->stream);
    auto free_status = cuMemFree(pointer);
    if (free_status == CUDA_SUCCESS)
      gpu->allocated -= bytes;
    if (gpu->deferred_error == CUDA_SUCCESS)
      gpu->deferred_error =
          sync_status != CUDA_SUCCESS ? sync_status : free_status;
  }
}
void Gpu::check(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    const char *error = nullptr;
    cuGetErrorString(result, &error);
    throw std::runtime_error(std::string(operation) + ": " +
                             (error ? error : "CUDA failure"));
  }
}
std::string Gpu::compile(const std::string &source) {
  require(cuewInit(CUEW_INIT_NVRTC) == CUEW_SUCCESS, "NVRTC unavailable");
  nvrtcProgram program = nullptr;
  require(nvrtcCreateProgram(&program, source.c_str(), "hv15n.cu", 0, nullptr,
                             nullptr) == NVRTC_SUCCESS,
          "NVRTC create failed");
  const char *options[] = {"--gpu-architecture=compute_120", "--std=c++17",
                           "--include-path=/usr/local/cuda/include"};
  auto status = nvrtcCompileProgram(program, 3, options);
  size_t length = 0;
  nvrtcGetProgramLogSize(program, &length);
  std::string log(length, '\0');
  nvrtcGetProgramLog(program, log.data());
  if (status != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("NVRTC compilation: " + log);
  }
  nvrtcGetPTXSize(program, &length);
  std::string ptx(length, '\0');
  nvrtcGetPTX(program, ptx.data());
  nvrtcDestroyProgram(&program);
  return ptx;
}
std::string Gpu::mma_source() {
  std::string source = k_gemm_f16_src;
  auto position = source.find("    extern __shared__");
  require(position != std::string::npos, "repository GEMM signature changed");
  source.insert(position,
                "    Y += (size_t)blockIdx.z*M*N; X += (size_t)blockIdx.z*M*K; "
                "W += (size_t)blockIdx.z*N*K;\n");
  return source + large_gemm_source;
}
Gpu::Gpu(int device, int budget_mib, bool use_vendor, bool allow_fallback)
    : budget(size_t(budget_mib - 3072) * 1024 * 1024), vendor(use_vendor),
      fallback(allow_fallback) {
  require(budget_mib >= 4096 && budget_mib <= 14336,
          "GPU budget must be 4096..14336 MiB");
  require(cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) == CUEW_SUCCESS,
          "CUDA/NVRTC libraries unavailable");
  check(cuInit(0), "initialize CUDA driver");
  CUdevice d;
  check(cuDeviceGet(&d, device), "select device");
  int major = 0;
  check(cuDeviceGetAttribute(&major,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d),
        "compute capability");
  require(major == 12,
          "initial native port requires CUDA compute capability 12.x");
  try {
    check(cuCtxCreate(&context, 0, d), "create context");
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create stream");
    auto ptx = compile(ops_source);
    check(cuModuleLoadData(&ops, ptx.c_str()), "load operators");
    ptx = compile(mma_source());
    check(cuModuleLoadData(&mma, ptx.c_str()), "load repository GEMM");
    if (vendor || fallback)
      require(cublasewCreate(&blas, stream) == 0,
              "requested cuBLAS is unavailable");
    if (blas) {
      cublasew_set_tf32(blas, 0);
      require(cublasew_disallow_reduced_precision_reduction(blas) == 0,
              "cannot enforce FP32 cuBLAS reductions");
    }
  } catch (...) {
    if (blas)
      cublasewDestroy(blas);
    if (mma)
      cuModuleUnload(mma);
    if (ops)
      cuModuleUnload(ops);
    if (stream)
      cuStreamDestroy(stream);
    if (context)
      cuCtxDestroy(context);
    throw;
  }
}
Gpu::~Gpu() {
  if (stream)
    cuStreamSynchronize(stream);
  if (blas)
    cublasewDestroy(blas);
  if (mma)
    cuModuleUnload(mma);
  if (ops)
    cuModuleUnload(ops);
  if (stream)
    cuStreamDestroy(stream);
  if (context)
    cuCtxDestroy(context);
}
void Gpu::poll() {
  check(deferred_error, "deferred CUDA operation");
  if (cancel_check && cancel_check())
    cancelled = true;
  require(!cancelled.load(), "cancelled");
}
Tensor Gpu::empty(const std::vector<int> &shape) {
  size_t count = product(shape);
  require(count <= INT32_MAX, "kernel index range exceeded");
  auto storage = std::make_shared<Allocation>(this, count * sizeof(float));
  return {storage, storage->pointer, shape};
}
Tensor Gpu::upload(const std::vector<float> &data,
                   const std::vector<int> &shape) {
  require(data.size() == product(shape), "upload shape mismatch");
  auto result = empty(shape);
  check(cuMemcpyHtoD(result.pointer, data.data(), data.size() * 4), "upload");
  // Pageable copies may return after host staging. The nonblocking compute
  // stream must not consume the still-pending default-stream DMA.
  check(cuCtxSynchronize(), "finish upload");
  return result;
}
std::vector<float> Gpu::download(const Tensor &x) {
  check(cuStreamSynchronize(stream), "synchronize");
  std::vector<float> result(x.count());
  check(cuMemcpyDtoH(result.data(), x.pointer, result.size() * 4), "download");
  for (float v : result)
    require(std::isfinite(v), "nonfinite inference tensor");
  return result;
}
Tensor Gpu::clone(const Tensor &x) {
  auto result = empty(x.shape);
  check(cuMemcpyDtoDAsync(result.pointer, x.pointer, x.count() * 4, stream),
        "clone");
  return result;
}
Tensor Gpu::rows(const Tensor &x, int start, int count) {
  require(start >= 0 && count > 0 && start + count <= x.rows(),
          "invalid row slice");
  return {x.storage,
          x.pointer + size_t(start) * x.channels() * 4,
          {count, x.channels()}};
}
Tensor Gpu::columns(const Tensor &x, int start, int count) {
  require(start >= 0 && count > 0 && start + count <= x.channels(),
          "invalid column slice");
  auto shape = x.shape;
  shape.back() = count;
  auto y = empty(shape);
  launch("columns", int((y.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer,
         x.pointer, x.rows(), x.channels(), start, count);
  return y;
}
Tensor Gpu::concat(const Tensor &a, const Tensor &b) {
  require(a.channels() == b.channels(), "concat channel mismatch");
  auto y = empty({a.rows() + b.rows(), a.channels()});
  check(cuMemcpyDtoDAsync(y.pointer, a.pointer, a.count() * 4, stream),
        "concat first");
  check(cuMemcpyDtoDAsync(y.pointer + a.count() * 4, b.pointer, b.count() * 4,
                          stream),
        "concat second");
  return y;
}
Tensor Gpu::weight(Weights &w, const std::string &name) {
  return upload(w.floats(name), w.shape(name));
}
Tensor Gpu::matmul(const Tensor &x, const Tensor &w, bool precise) {
  int m = x.rows(), k = x.channels(), n = w.shape[0];
  require(int(w.count() / n) == k, "GEMM shape mismatch");
  auto y = empty({m, n});
  bool use_blas = vendor || (precise && fallback);
  if (precise) {
    if (use_blas) {
      require(cublasew_gemm_f32_pedantic_rowmajor_nt(blas, y.pointer, w.pointer,
                                                     x.pointer, m, n, k) == 0,
              "IEEE cuBLAS GEMM failed");
      blas_gemm++;
      if (!vendor)
        fallback_gemm++;
    } else {
      launch("gemm_ieee", (n + 15) / 16, (m + 15) / 16, 1, 16, 16, 0, y.pointer,
             x.pointer, w.pointer, m, n, k);
      repo_gemm++;
    }
    return y;
  }
  int padded = (k + 15) / 16 * 16;
  auto hx = empty({m, (padded + 1) / 2}), hw = empty({n, (padded + 1) / 2});
  launch("convert_half", (m * padded + 255) / 256, 1, 1, 256, 1, 0, hx.pointer,
         x.pointer, m, k, padded);
  launch("convert_half", (n * padded + 255) / 256, 1, 1, 256, 1, 0, hw.pointer,
         w.pointer, n, k, padded);
  if (use_blas) {
    require(cublasew_gemm_f16_f16_f32_rowmajor_nt(
                blas, y.pointer, hw.pointer, hx.pointer, m, n, padded) == 0,
            "cuBLAS F16 GEMM failed");
    blas_gemm++;
  } else {
    CUfunction fn;
    bool large = m >= 2048;
    check(cuModuleGetFunction(&fn, mma, large ? "gemm_f16_large" : "gemm_f16"),
          "GEMM lookup");
    void *params[] = {&y.pointer, &hx.pointer, &hw.pointer, &m, &n, &padded};
    int ntile = large ? 128 : 256, mtile = large ? 64 : 16;
    check(cuLaunchKernel(fn, (n + ntile - 1) / ntile, (m + mtile - 1) / mtile,
                         1, 128, 1, 1, large ? 2048 : 512, stream, params,
                         nullptr),
          "repository GEMM");
    repo_gemm++;
  }
  return y;
}
Tensor Gpu::linear(Weights &w, const std::string &prefix, const Tensor &x,
                   bool precise) {
  auto weights = weight(w, prefix + ".weight");
  auto y = matmul(x, weights, precise);
  if (w.has(prefix + ".bias")) {
    auto b = weight(w, prefix + ".bias");
    y = op(y, 6, nullptr, &b);
  }
  return y;
}
Tensor Gpu::bare_norm(const Tensor &x, int mode, float eps) {
  auto y = empty(x.shape);
  CUdeviceptr zero = 0;
  launch("hv15n_norm", x.rows(), 1, 1, 256, 1, 0, y.pointer, x.pointer, zero,
         zero, x.rows(), x.channels(), mode, eps);
  return y;
}
Tensor Gpu::norm(Weights &w, const std::string &prefix, const Tensor &x,
                 int mode, float eps) {
  auto name =
      w.has(prefix + ".weight") ? prefix + ".weight" : prefix + ".gamma";
  auto weight_tensor = weight(w, name);
  require(weight_tensor.count() == size_t(x.channels()),
          "normalization weight shape mismatch");
  Tensor bias;
  if (w.has(prefix + ".bias"))
    bias = weight(w, prefix + ".bias");
  auto y = empty(x.shape);
  launch("hv15n_norm", x.rows(), 1, 1, 256, 1, 0, y.pointer, x.pointer,
         weight_tensor.pointer, bias.pointer, x.rows(), x.channels(), mode,
         eps);
  return y;
}
Tensor Gpu::op(const Tensor &x, int mode, const Tensor *second,
               const Tensor *bias, float scale, float offset) {
  auto y = empty(x.shape);
  CUdeviceptr z = second ? second->pointer : 0, b = bias ? bias->pointer : 0;
  if (mode == 1 || mode == 2)
    require(second && second->count() == x.count(),
            "elementwise size mismatch");
  if (mode == 6 || mode == 7 || mode == 8)
    require((mode == 6 ? bias : second) &&
                ((mode == 6 ? bias : second)->count() == size_t(x.channels())),
            "broadcast size mismatch");
  launch("element", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer,
         x.pointer, z, b, int(x.count()), x.channels(), mode, scale, offset);
  return y;
}
void Gpu::rotary(Tensor &x, int heads, int kind, int height, int width,
                 float theta) {
  int dim = x.channels() / heads;
  require(dim % 2 == 0, "odd RoPE dimension");
  launch("rope", int((x.count() / 2 + 255) / 256), 1, 1, 256, 1, 0, x.pointer,
         x.rows(), heads, dim, kind, height, width, theta);
}
Tensor Gpu::attention(const Tensor &q, const Tensor &k, const Tensor &v,
                      int heads, int kvheads, int mask, int frame_hw,
                      bool precise, const Tensor *bias, float scale) {
  int rows = q.rows(), dim = q.channels() / heads;
  require(rows == k.rows() && rows == v.rows() &&
              k.channels() == kvheads * dim && v.channels() == k.channels(),
          "attention shape mismatch");
  require(heads % kvheads == 0 && dim <= 2048, "unsupported attention heads");
  if (scale == 0.f)
    scale = 1.f / std::sqrt(float(dim));
  auto y = empty(q.shape);
  attention_calls++;
  if (precise || heads != kvheads || bias) {
    CUdeviceptr b = bias ? bias->pointer : 0;
    launch("attention_ieee", rows, heads, 1, 256, 1, 0, y.pointer, q.pointer,
           k.pointer, v.pointer, b, rows, heads, kvheads, dim, mask, frame_hw,
           scale);
    return y;
  }
  require(dim % 16 == 0, "tensor-core attention dimension must align to 16");
  constexpr int tile = 128;
  auto packed = empty({heads, rows, dim / 2}),
       keys = empty({heads, tile, dim / 2}),
       values = empty({heads, dim, tile / 2});
  auto scores = empty({heads, rows, tile}),
       prob = empty({heads, rows, tile / 2}), acc = empty({heads, rows, dim}),
       partial = empty({heads, rows, dim});
  auto maxima = upload(std::vector<float>(size_t(heads) * rows, -INFINITY),
                       {heads, rows}),
       sums =
           upload(std::vector<float>(size_t(heads) * rows, 0.f), {heads, rows});
  check(cuMemsetD32Async(acc.pointer, 0, acc.count(), stream),
        "clear attention");
  launch("pack_heads", int((q.count() + 255) / 256), 1, 1, 256, 1, 0,
         packed.pointer, q.pointer, rows, heads, dim);
  CUfunction fn;
  bool large = rows >= 2048;
  check(cuModuleGetFunction(&fn, mma, large ? "gemm_f16_large" : "gemm_f16"),
        "attention GEMM lookup");
  auto batch = [&](Tensor &out, Tensor &x, Tensor &w, int m, int n, int inner) {
    void *params[] = {&out.pointer, &x.pointer, &w.pointer, &m, &n, &inner};
    if (vendor) {
      for (int h = 0; h < heads; h++)
        require(cublasew_gemm_f16_f16_f32_rowmajor_nt(
                    blas, out.pointer + size_t(h) * m * n * 4,
                    w.pointer + size_t(h) * n * inner * 2,
                    x.pointer + size_t(h) * m * inner * 2, m, n, inner) == 0,
                "attention cuBLAS failed");
      blas_gemm += heads;
    } else {
      int ntile = large ? 128 : 256, mtile = large ? 64 : 16;
      check(cuLaunchKernel(fn, (n + ntile - 1) / ntile, (m + mtile - 1) / mtile,
                           heads, 128, 1, 1, large ? 2048 : 512, stream, params,
                           nullptr),
            "attention repository GEMM");
      repo_gemm += heads;
    }
  };
  for (int start = 0; start < rows; start += tile) {
    int count = std::min(tile, rows - start);
    launch("pack_keys", (heads * tile * dim + 255) / 256, 1, 1, 256, 1, 0,
           keys.pointer, k.pointer, rows, heads, dim, start, tile, 0);
    launch("pack_keys", (heads * tile * dim + 255) / 256, 1, 1, 256, 1, 0,
           values.pointer, v.pointer, rows, heads, dim, start, tile, 1);
    batch(scores, packed, keys, rows, tile, dim);
    launch("softmax_tile", (rows * heads + 7) / 8, 1, 1, 256, 1, 0,
           scores.pointer, prob.pointer, acc.pointer, maxima.pointer,
           sums.pointer, rows, heads, dim, count, start, tile, mask, frame_hw,
           scale);
    batch(partial, prob, values, rows, dim, tile);
    // Keep the online accumulator resident across key tiles. Allocating
    // another full query/head output here forces a fence for every tile.
    CUdeviceptr no_bias = 0;
    launch("element", int((acc.count() + 255) / 256), 1, 1, 256, 1, 0,
           acc.pointer, acc.pointer, partial.pointer, no_bias, int(acc.count()),
           acc.channels(), 1, 1.f, 0.f);
  }
  launch("finish_attention", int((q.count() + 255) / 256), 1, 1, 256, 1, 0,
         y.pointer, acc.pointer, sums.pointer, rows, heads, dim);
  return y;
}
Tensor Gpu::conv(Weights &w, const std::string &prefix, const Tensor &x,
                 bool causal) {
  require(x.shape.size() == 4, "convolution requires THWC");
  auto ws = w.shape(prefix + ".weight");
  require(ws.size() == 5 && ws[1] == x.channels(),
          "convolution shape mismatch");
  int kt = ws[2], kh = ws[3], kw = ws[4], t = x.shape[0], h = x.shape[1],
      width = x.shape[2];
  auto weights = weight(w, prefix + ".weight");
  weights.shape = {ws[0], ws[1] * kt * kh * kw};
  auto y = empty({t, h, width, ws[0]});
  Tensor bias;
  if (w.has(prefix + ".bias"))
    bias = weight(w, prefix + ".bias");
  int chunk =
      std::max(1, std::min(1024, int(32 * 1024 * 1024 /
                                     (weights.channels() * sizeof(float)))));
  for (int start = 0; start < x.rows(); start += chunk) {
    int count = std::min(chunk, x.rows() - start);
    auto col = empty({count, weights.channels()});
    launch("im2col", int((col.count() + 255) / 256), 1, 1, 256, 1, 0,
           col.pointer, x.pointer, t, h, width, x.channels(), kt, kh, kw,
           causal ? kt - 1 : 0, kh / 2, kw / 2, causal ? 1 : 0, start, count);
    auto out = matmul(col, weights);
    if (bias.pointer)
      out = op(out, 6, nullptr, &bias);
    check(cuMemcpyDtoDAsync(y.pointer + size_t(start) * ws[0] * 4, out.pointer,
                            out.count() * 4, stream),
          "convolution result");
  }
  return y;
}
Tensor Gpu::channel_map(const Tensor &x, int channels, bool mean) {
  require(mean ? x.channels() % channels == 0 : channels % x.channels() == 0,
          "channel map mismatch");
  auto s = x.shape;
  s.back() = channels;
  auto y = empty(s);
  launch("channel_map", int((y.count() + 255) / 256), 1, 1, 256, 1, 0,
         y.pointer, x.pointer, x.rows(), x.channels(), channels, int(mean));
  return y;
}
Tensor Gpu::mean(const Tensor &x) {
  auto y = empty({1, x.channels()});
  launch("mean_rows", (x.channels() + 255) / 256, 1, 1, 256, 1, 0, y.pointer,
         x.pointer, x.rows(), x.channels());
  return y;
}
void Gpu::dump(const Tensor &x, const fs::path &dir, const std::string &name,
               bool video) {
  if (dir.empty())
    return;
  fs::create_directories(dir);
  auto data = download(x);
  auto shape = x.shape;
  if (video) {
    require(shape.size() == 4, "video dump shape");
    int c = shape.back(), r = x.rows();
    std::vector<float> ncthw(data.size());
    for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++)
        ncthw[size_t(j) * r + i] = data[size_t(i) * c + j];
    data.swap(ncthw);
    shape = {1, c, shape[0], shape[1], shape[2]};
  } else if (shape.size() == 2)
    shape.insert(shape.begin(), 1);
  std::ofstream raw(dir / (name + ".f32"), std::ios::binary);
  raw.write(reinterpret_cast<const char *>(data.data()), data.size() * 4);
  require(bool(raw), "dump write failed");
  std::ofstream meta(dir / (name + ".json"));
  meta << "{\"shape\":[";
  for (size_t i = 0; i < shape.size(); i++)
    meta << (i ? "," : "") << shape[i];
  meta << "],\"dtype\":\"float32\",\"layout\":\"" << (video ? "NCTHW" : "NTC")
       << "\"}\n";
  require(bool(meta), "dump metadata failed");
}
std::string Gpu::metrics() const {
  std::ostringstream s;
  s << "{\"backend\":\"hv15n_cuda\",\"repo_gemm_calls\":" << repo_gemm
    << ",\"cublas_gemm_calls\":" << blas_gemm
    << ",\"fallback_gemm_calls\":" << fallback_gemm
    << ",\"attention_calls\":" << attention_calls
    << ",\"managed_peak_vram_mib\":" << double(peak) / (1024 * 1024)
    << ",\"parity\":\"unverified\"}";
  return s.str();
}
} // namespace hv15n
