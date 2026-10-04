// SPDX-License-Identifier: MIT
// HIP implementation of the existing native Hunyuan GPU interface.
#include "../../cuda/hunyuan_video15_native/gpu.hpp"
#include "../../cuda/hunyuan_video15_native/kernels.hpp"
#include "../video_common/kernels.hpp"
#include <cstring>
#include <sstream>
namespace hv15n {
Allocation::Allocation(Gpu *owner, size_t size) : gpu(owner), bytes(size) {
    if (gpu->optimized) {
        auto it = gpu->free_buffers.lower_bound(bytes);
        if (it != gpu->free_buffers.end() && it->first <= bytes + bytes / 2) {
            bytes = it->first;
            pointer = it->second;
            gpu->free_buffers.erase(it);
            gpu->buffer_reuses++;
            return;
        }
        if (bytes > gpu->budget - gpu->allocated)
            gpu->trim_pool();
    }
    require(bytes <= gpu->budget - gpu->allocated, "GPU allocation exceeds requested budget");
    size_t available = 0, total = 0;
    gpu->check(cuMemGetInfo(&available, &total), "device headroom");
    if (gpu->optimized && (bytes > available || available - bytes < (256ull << 20))) {
        gpu->trim_pool();
        gpu->check(cuMemGetInfo(&available, &total), "device headroom after pool trim");
    }
    require(bytes <= available && available - bytes >= (256ull << 20),
            "GPU allocation would consume device headroom");
    gpu->check(cuMemAlloc(&pointer, bytes), "allocate tensor");
    gpu->allocated += bytes;
    gpu->peak = std::max(gpu->peak, gpu->allocated);
    gpu->allocations++;
}
Allocation::~Allocation() {
    if (!pointer)
        return;
    if (gpu->optimized) {
        // Reuse is safe because all consumers and copies use the same stream.
        gpu->free_buffers.emplace(bytes, pointer);
        return;
    }
    auto sync_status = cuStreamSynchronize(gpu->stream);
    auto free_status = cuMemFree(pointer);
    if (free_status == CUDA_SUCCESS)
        gpu->allocated -= bytes;
    if (gpu->deferred_error == CUDA_SUCCESS)
        gpu->deferred_error = sync_status != CUDA_SUCCESS ? sync_status : free_status;
}
void Gpu::trim_pool() {
    check(cuStreamSynchronize(stream), "drain buffer pool");
    for (const auto &entry : free_buffers) {
        check(cuMemFree(entry.second), "release pooled buffer");
        allocated -= entry.first;
    }
    free_buffers.clear();
}
void Gpu::clear_weights() {
    if (copy_stream) {
        auto status = cuStreamSynchronize(copy_stream);
        if (deferred_error == CUDA_SUCCESS)
            deferred_error = status;
    }
    for (const auto &entry : staged_blocks)
        cuEventDestroy(entry.second);
    staged_blocks.clear();
    packed_weights.clear();
    cached_weight_bytes = 0;
    active_weight_file.clear();
}
void Gpu::upload_staged(const Tensor &destination, const void *source, CUstream target) {
    constexpr size_t chunk_bytes = 64ull << 20;
    for (size_t offset = 0; offset < destination.bytes(); offset += chunk_bytes) {
        const size_t count = std::min(chunk_bytes, destination.bytes() - offset);
        Staging *slot = nullptr;
        for (auto &entry : staging)
            if (entry.bytes >= count && cuEventQuery(entry.ready) == CUDA_SUCCESS) {
                slot = &entry;
                break;
            }
        if (!slot && staging.size() >= 2) {
            slot = &staging.front();
            check(cuEventSynchronize(slot->ready), "recycle pinned transfer");
        }
        if (!slot) {
            Staging entry{};
            entry.bytes = chunk_bytes;
            check(cuMemHostAlloc(&entry.data, entry.bytes, 0), "allocate pinned staging");
            try {
                check(cuEventCreate(&entry.ready, CU_EVENT_DISABLE_TIMING), "staging event");
                staging.push_back(entry);
            } catch (...) {
                if (entry.ready)
                    cuEventDestroy(entry.ready);
                cuMemFreeHost(entry.data);
                throw;
            }
            pinned_bytes += entry.bytes;
            slot = &staging.back();
        }
        std::memcpy(slot->data, static_cast<const unsigned char *>(source) + offset, count);
        check(cuMemcpyHtoDAsync(destination.pointer + offset, slot->data, count, target),
              "staged transfer");
        check(cuEventRecord(slot->ready, target), "staging completion");
        upload_bytes += count;
    }
}
void Gpu::prefetch_block(Weights &w, int index) {
    if (!optimized || vendor || index < 0 || index >= 54)
        return;
    if (active_weight_file != w.identity) {
        clear_weights();
        active_weight_file = w.identity;
    }
    if (staged_blocks.count(index))
        return;
    // A pooled destination may still have consumers in the compute stream.
    CUevent safe = nullptr;
    check(cuEventCreate(&safe, CU_EVENT_DISABLE_TIMING), "transfer safety event");
    check(cuEventRecord(safe, stream), "transfer buffer safety");
    check(cuStreamWaitEvent(copy_stream, safe, 0), "wait for prior buffer consumers");
    cuEventDestroy(safe);
    std::string prefix = "double_blocks." + std::to_string(index) + ".";
    for (int i = 0; i < w.context->n_tensors; i++) {
        std::string name = w.context->tensors[i].name;
        if (name.compare(0, prefix.size(), prefix) != 0 || w.dtype(name) != "F16")
            continue;
        std::string key = w.identity + ":f16:" + name;
        if (packed_weights.count(key))
            continue;
        auto destination = empty_half(w.shape(name));
        upload_staged(destination, w.data(name), copy_stream);
        prefetch_bytes += destination.bytes();
        cached_weight_bytes += destination.bytes();
        packed_weights.emplace(key, destination);
    }
    CUevent done = nullptr;
    check(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING), "block transfer event");
    check(cuEventRecord(done, copy_stream), "block transfer completion");
    staged_blocks.emplace(index, done);
}
void Gpu::wait_block(int index) {
    auto it = staged_blocks.find(index);
    if (it != staged_blocks.end())
        check(cuStreamWaitEvent(stream, it->second, 0), "wait for block weights");
}
void Gpu::release_block(Weights &w, int index) {
    auto event = staged_blocks.find(index);
    if (event != staged_blocks.end()) {
        cuEventDestroy(event->second);
        staged_blocks.erase(event);
    }
    if (index < 6)
        return;
    std::string suffix = "double_blocks." + std::to_string(index) + ".";
    for (auto it = packed_weights.begin(); it != packed_weights.end();) {
        if (it->first.compare(0, w.identity.size(), w.identity) == 0 &&
            it->first.find(suffix, w.identity.size()) != std::string::npos) {
            cached_weight_bytes -= it->second.bytes();
            it = packed_weights.erase(it);
        } else
            ++it;
    }
}
void Gpu::check(CUresult result, const char *operation) {
    if (result != CUDA_SUCCESS) {
        const char *error = nullptr;
        cuGetErrorString(result, &error);
        throw std::runtime_error(std::string(operation) + ": " + (error ? error : "HIP failure"));
    }
}
static thread_local std::string target_arch = "gfx1201";
std::string Gpu::compile(const std::string &source) {
    require(rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) == ROCEW_SUCCESS,
            "HIP/HIPRTC unavailable");
    std::string translated = source;
    auto at = translated.find("#include <cuda_fp16.h>");
    if (at != std::string::npos)
        translated.replace(at, std::strlen("#include <cuda_fp16.h>"), "#include <hip/hip_fp16.h>");
    translated = "#define __shfl_xor_sync(mask,x,d) __shfl_xor(x,d)\n" + translated;
    hiprtcProgram program = nullptr;
    require(hiprtcCreateProgram(&program, translated.c_str(), "video.hip", 0, nullptr, nullptr) ==
                HIPRTC_SUCCESS,
            "HIPRTC create failed");
    std::string arch = "--gpu-architecture=" + target_arch;
    const char *options[] = {arch.c_str(), "--std=c++17", "-O3", "-ffp-contract=off",
                             "-I/opt/rocm/include"};
    auto status = hiprtcCompileProgram(program, 5, options);
    size_t length = 0;
    hiprtcGetProgramLogSize(program, &length);
    std::string log(length, '\0');
    if (length)
        hiprtcGetProgramLog(program, log.data());
    if (status != HIPRTC_SUCCESS) {
        hiprtcDestroyProgram(&program);
        throw std::runtime_error("HIPRTC: " + log);
    }
    hiprtcGetCodeSize(program, &length);
    std::string code(length, '\0');
    hiprtcGetCode(program, code.data());
    hiprtcDestroyProgram(&program);
    return code;
}
std::string Gpu::mma_source() { return video_rocm::gemm_source; }
Gpu::Gpu(int device, int budget_mib, bool use_vendor, bool allow_fallback, bool use_optimized)
    : budget(size_t(budget_mib - 3072) * 1048576), vendor(use_vendor), fallback(allow_fallback),
      optimized(use_optimized) {
    require(budget_mib >= 4096 && budget_mib <= 14336, "GPU budget must be 4096..14336 MiB");
    require(rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) == ROCEW_SUCCESS,
            "HIP/HIPRTC unavailable");
    check(hipInit(0), "initialize HIP");
    int count = 0;
    check(hipGetDeviceCount(&count), "device count");
    require(device >= 0 && device < count, "invalid AMD device");
    const char *arch = rocewGetRDNA4ArchString(device);
    require(arch, "video backend requires gfx1200/gfx1201");
    target_arch = arch;
    try {
        check(hipCtxCreate(&context, 0, device), "HIP context");
        check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "compute stream");
        check(cuStreamCreate(&copy_stream, CU_STREAM_NON_BLOCKING), "transfer stream");
        auto code = compile(std::string(ops_source) + video_rocm::precision_source);
        check(cuModuleLoadData(&ops, code.data()), "load video operators");
        code = compile(mma_source());
        check(cuModuleLoadData(&mma, code.data()), "load WMMA GEMM");
        code = compile(video_rocm::attention_source);
        check(cuModuleLoadData(&flash, code.data()), "load WMMA attention");
        if (vendor || fallback)
            require(cublasewCreate(&blas, stream) == 0, "requested hipBLAS unavailable");
    } catch (...) {
        if (blas)
            cublasewDestroy(blas);
        if (flash)
            cuModuleUnload(flash);
        if (mma)
            cuModuleUnload(mma);
        if (ops)
            cuModuleUnload(ops);
        if (copy_stream)
            cuStreamDestroy(copy_stream);
        if (stream)
            cuStreamDestroy(stream);
        if (context)
            cuCtxDestroy(context);
        throw;
    }
}
Gpu::~Gpu() {
    if (context)
        cuCtxSetCurrent(context);
    if (copy_stream)
        cuStreamSynchronize(copy_stream);
    if (stream)
        cuStreamSynchronize(stream);
    for (const auto &entry : staging) {
        cuEventDestroy(entry.ready);
        cuMemFreeHost(entry.data);
    }
    clear_weights();
    if (stream && !free_buffers.empty()) {
        cuStreamSynchronize(stream);
        for (const auto &entry : free_buffers)
            cuMemFree(entry.second);
        free_buffers.clear();
    }
    if (flash)
        cuModuleUnload(flash);
    if (stream)
        cuStreamSynchronize(stream);
    aot_library.reset();
    if (blas)
        cublasewDestroy(blas);
    if (mma)
        cuModuleUnload(mma);
    if (ops)
        cuModuleUnload(ops);
    if (copy_stream)
        cuStreamDestroy(copy_stream);
    if (stream)
        cuStreamDestroy(stream);
    if (context)
        cuCtxDestroy(context);
}
void Gpu::use_aotriton(const char *path) {
    require(path && *path, "AOTriton bridge path is required");
    check(cuCtxSetCurrent(context), "select context for AOTriton");
    check(cuStreamSynchronize(stream), "drain attention before changing provider");
    std::unique_ptr<void, int (*)(void *)> candidate(dlopen(path, RTLD_NOW | RTLD_LOCAL), dlclose);
    if (!candidate) {
        const char *error = dlerror();
        throw std::runtime_error(std::string("cannot load standalone AOTriton bridge: ") +
                                 (error ? error : "unknown loader error"));
    }
    auto abi = reinterpret_cast<int (*)(void)>(dlsym(candidate.get(), "video_aotriton_bridge_abi"));
    require(abi && abi() == 1, "unsupported AOTriton bridge ABI");
    auto rows = reinterpret_cast<video_aotriton_forward_fn>(
        dlsym(candidate.get(), "video_aotriton_forward"));
    auto heads = reinterpret_cast<video_aotriton_forward_fn>(
        dlsym(candidate.get(), "video_aotriton_forward_heads"));
    require(rows && heads, "bridge needs row and head-packed attention entry points");
    aot_library = std::move(candidate);
    aot_forward = rows;
    aot_heads = heads;
}
void Gpu::poll() {
    check(deferred_error, "deferred HIP operation");
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
Tensor Gpu::empty_half(const std::vector<int> &shape) {
    size_t count = product(shape);
    require(count <= INT32_MAX, "kernel index range exceeded");
    auto storage = std::make_shared<Allocation>(this, count * 2);
    return {storage, storage->pointer, shape, 2, 0, true};
}
Tensor Gpu::round_half(Tensor x) {
    require(x.element_bytes == 4, "F32 storage required for FP16 rounding");
    launch("hv15_round_half", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, x.pointer,
           int(x.count()));
    x.fp16_values = true;
    return x;
}
Tensor Gpu::half(const Tensor &x) {
    if (x.element_bytes == 2)
        return x;
    auto y = empty_half(x.shape);
    launch("convert_half", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, x.pointer,
           x.rows(), x.channels(), x.channels());
    return y;
}
Tensor Gpu::full(const Tensor &x) {
    if (x.element_bytes == 4)
        return x;
    auto y = empty(x.shape);
    if (x.packed_heads)
        launch("unpack_heads_half", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer,
               x.pointer, x.rows(), x.packed_heads, x.channels() / x.packed_heads);
    else
        launch("convert_float", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, x.pointer,
               int(x.count()));
    y.fp16_values = true;
    return y;
}
Tensor Gpu::upload(const std::vector<float> &data, const std::vector<int> &shape) {
    require(data.size() == product(shape), "upload shape mismatch");
    auto result = empty(shape);
    check(cuMemcpyHtoDAsync(result.pointer, data.data(), data.size() * 4, stream), "upload");
    check(cuStreamSynchronize(stream), "finish pageable upload");
    upload_bytes += data.size() * 4;
    return result;
}
std::vector<float> Gpu::download(const Tensor &x) {
    if (x.element_bytes == 2)
        return download(full(x));
    check(cuStreamSynchronize(stream), "synchronize");
    std::vector<float> result(x.count());
    check(cuMemcpyDtoH(result.data(), x.pointer, result.size() * 4), "download");
    for (float v : result)
        require(std::isfinite(v), "nonfinite inference tensor");
    return result;
}
Tensor Gpu::clone(const Tensor &x) {
    auto result = x.element_bytes == 2 ? empty_half(x.shape) : empty(x.shape);
    result.packed_heads = x.packed_heads;
    result.fp16_values = x.fp16_values;
    check(cuMemcpyDtoDAsync(result.pointer, x.pointer, x.bytes(), stream), "clone");
    return result;
}
Tensor Gpu::rows(const Tensor &x, int start, int count) {
    require(start >= 0 && count > 0 && start + count <= x.rows(), "invalid row slice");
    require(!x.packed_heads, "packed heads do not support row views");
    return {x.storage,
            x.pointer + size_t(start) * x.channels() * x.element_bytes,
            {count, x.channels()},
            x.element_bytes,
            0,
            x.fp16_values};
}
Tensor Gpu::columns(const Tensor &x, int start, int count) {
    require(start >= 0 && count > 0 && start + count <= x.channels(), "invalid column slice");
    auto shape = x.shape;
    shape.back() = count;
    auto y = empty(shape);
    launch("columns", int((y.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, x.pointer, x.rows(),
           x.channels(), start, count);
    y.fp16_values = x.fp16_values;
    return y;
}
Tensor Gpu::concat(const Tensor &a, const Tensor &b) {
    require(a.channels() == b.channels(), "concat channel mismatch");
    if (a.packed_heads || b.packed_heads) {
        require(a.packed_heads == b.packed_heads && a.element_bytes == 2 && b.element_bytes == 2,
                "packed concat mismatch");
        auto y = empty_half({a.rows() + b.rows(), a.channels()});
        y.packed_heads = a.packed_heads;
        launch("concat_heads", int((y.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, a.pointer,
               b.pointer, a.rows(), b.rows(), a.packed_heads, a.channels() / a.packed_heads);
        return y;
    }
    auto y = empty({a.rows() + b.rows(), a.channels()});
    check(cuMemcpyDtoDAsync(y.pointer, a.pointer, a.count() * 4, stream), "concat first");
    check(cuMemcpyDtoDAsync(y.pointer + a.count() * 4, b.pointer, b.count() * 4, stream),
          "concat second");
    y.fp16_values = a.fp16_values && b.fp16_values;
    return y;
}
Tensor Gpu::weight(Weights &w, const std::string &name) {
    if (!optimized)
        return upload(w.floats(name), w.shape(name));
    if (active_weight_file != w.identity) {
        clear_weights();
        active_weight_file = w.identity;
    }
    std::string key = w.identity + ":f32:" + name;
    auto it = packed_weights.find(key);
    if (it != packed_weights.end()) {
        weight_hits++;
        return it->second;
    }
    Tensor result;
    auto half_entry = packed_weights.find(w.identity + ":f16:" + name);
    if (half_entry != packed_weights.end())
        result = full(half_entry->second);
    else {
        auto shape = w.shape(name);
        auto type = w.dtype(name);
        require(type == "F32" || type == "F16" || type == "BF16", "unsupported weight dtype");
        if (type == "F32") {
            result = empty(shape);
            upload_staged(result, w.data(name), stream);
        } else {
            auto raw = empty_half(shape);
            upload_staged(raw, w.data(name), stream);
            result = empty(shape);
            launch(type == "F16" ? "convert_float" : "convert_bfloat",
                   int((raw.count() + 255) / 256), 1, 1, 256, 1, 0, result.pointer, raw.pointer,
                   int(raw.count()));
        }
    }
    result.fp16_values = w.dtype(name) == "F16";
    const size_t limit =
        w.identity.find("/vae/") != std::string::npos
            ? budget / 2
            : std::min(
                  size_t(w.identity.find("diffusion_models") != std::string::npos ? 2048 : 4096)
                      << 20,
                  budget / 2);
    if (cached_weight_bytes <= limit && result.bytes() <= limit - cached_weight_bytes) {
        packed_weights.emplace(key, result);
        cached_weight_bytes += result.bytes();
    }
    return result;
}
Tensor Gpu::weight_half(Weights &w, const std::string &name) {
    if (active_weight_file != w.identity) {
        clear_weights();
        active_weight_file = w.identity;
    }
    std::string key = w.identity + ":f16:" + name;
    auto it = packed_weights.find(key);
    if (it != packed_weights.end()) {
        weight_hits++;
        return it->second;
    }
    auto shape = w.shape(name);
    Tensor result;
    if (w.dtype(name) == "F16") {
        result = empty_half(shape);
        upload_staged(result, w.data(name), stream);
    } else {
        result = half(weight(w, name));
    }
    // Keep a fixed prefix resident: a scan of all 54 blocks must not thrash an LRU.
    result.fp16_values = w.dtype(name) == "F16";
    const size_t limit =
        w.identity.find("/vae/") != std::string::npos
            ? budget / 2
            : std::min(
                  size_t(w.identity.find("diffusion_models") != std::string::npos ? 2048 : 4096)
                      << 20,
                  budget / 2);
    if (cached_weight_bytes <= limit && result.bytes() <= limit - cached_weight_bytes) {
        packed_weights.emplace(key, result);
        cached_weight_bytes += result.bytes();
    }
    return result;
}
Tensor Gpu::matmul(const Tensor &x, const Tensor &w, bool precise, const Tensor *destination) {
    int m = x.rows(), k = x.channels(), n = w.shape[0];
    require(int(w.count() / n) == k, "GEMM shape mismatch");
    auto y = destination ? *destination : empty({m, n});
    require(y.element_bytes == 4 && y.rows() == m && y.channels() == n, "GEMM destination shape");
    bool use_blas = vendor || (precise && fallback);
    if (precise) {
        require(x.element_bytes == 4 && w.element_bytes == 4 && !x.packed_heads && !w.packed_heads,
                "IEEE GEMM requires unpacked FP32 tensors");
        if (use_blas) {
            require(cublasew_gemm_f32_pedantic_rowmajor_nt(blas, y.pointer, w.pointer, x.pointer, m,
                                                           n, k) == 0,
                    "IEEE cuBLAS GEMM failed");
            blas_gemm++;
            if (!vendor)
                fallback_gemm++;
        } else {
            bool tiled = optimized && m >= 32 && n >= 64 && k >= 64;
            launch(tiled ? "gemm_ieee_tiled" : "gemm_ieee",
                   (n + (tiled ? 63 : 15)) / (tiled ? 64 : 16),
                   (m + (tiled ? 63 : 15)) / (tiled ? 64 : 16), 1, 16, 16, 0, y.pointer, x.pointer,
                   w.pointer, m, n, k);
            ieee_tiled_calls += tiled;
            repo_gemm++;
        }
        return y;
    }
    int padded = (k + 15) / 16 * 16;
    Tensor hx;
    if (x.element_bytes == 2 && padded == k)
        hx = x;
    else {
        auto xf = full(x);
        hx = empty_half({m, padded});
        launch("convert_half", (m * padded + 255) / 256, 1, 1, 256, 1, 0, hx.pointer, xf.pointer, m,
               k, padded);
    }
    Tensor hw;
    if (w.element_bytes == 2 && padded == k) {
        hw = w;
    } else {
        auto wf = full(w);
        hw = empty_half({n, padded});
        launch("convert_half", (n * padded + 255) / 256, 1, 1, 256, 1, 0, hw.pointer, wf.pointer, n,
               k, padded);
    }
    if (use_blas) {
        require(cublasew_gemm_f16_f16_f32_rowmajor_nt(blas, y.pointer, hw.pointer, hx.pointer, m, n,
                                                      padded) == 0,
                "cuBLAS F16 GEMM failed");
        blas_gemm++;
    } else {
        CUfunction fn;
        const bool tiled = optimized && m >= 128 && n >= 128 && padded >= 64;
        const int tile = tiled ? 128 : 32;
        check(cuModuleGetFunction(&fn, mma, tiled ? "video_gemm_f16_tiled" : "video_gemm_f16"),
              "WMMA lookup");
        void *params[] = {&y.pointer, &hx.pointer, &hw.pointer, &m, &n, &padded};
        poll();
        check(cuLaunchKernel(fn, (n + tile - 1) / tile, (m + tile - 1) / tile, 1, 128, 1, 1, 0,
                             stream, params, nullptr),
              "FP16 WMMA GEMM");
        repo_gemm++;
        f16_tiled_calls += tiled;
    }
    return y;
}
Tensor Gpu::linear(Weights &w, const std::string &prefix, const Tensor &x, bool precise) {
    auto weights =
        optimized && !precise ? weight_half(w, prefix + ".weight") : weight(w, prefix + ".weight");
    auto y = matmul(x, weights, precise);
    if (w.has(prefix + ".bias")) {
        auto b = weight(w, prefix + ".bias");
        y = op(y, 6, nullptr, &b);
    }
    return dit_fp16 && !precise ? round_half(std::move(y)) : y;
}
Tensor Gpu::norm_silu(Weights &w, const std::string &prefix, const Tensor &x) {
    require(x.element_bytes == 4, "fused VAE normalization input");
    auto gamma = weight(w, w.has(prefix + ".weight") ? prefix + ".weight" : prefix + ".gamma");
    Tensor bias;
    if (w.has(prefix + ".bias"))
        bias = weight(w, prefix + ".bias");
    auto y = empty_half(x.shape);
    launch("norm_silu_half", (x.rows() + 7) / 8, 1, 1, 256, 1, 0, y.pointer, x.pointer,
           gamma.pointer, bias.pointer, x.rows(), x.channels());
    return y;
}
Tensor Gpu::modulate(const Tensor &x, const Tensor &mod, int start) {
    require(start >= 0 && size_t(start + 2 * x.channels()) <= mod.count(), "modulation bounds");
    auto y = empty(x.shape);
    launch(dit_fp16 && mod.fp16_values ? "hv15_autocast_modulate" : "modulate_norm", x.rows(), 1, 1,
           256, 1, 0, y.pointer, x.pointer, mod.pointer, x.rows(), x.channels(), start);
    y.fp16_values = false;
    return y;
}
Tensor Gpu::gated(const Tensor &x, const Tensor &delta, const Tensor &mod, int start) {
    require(x.count() == delta.count() && start >= 0 && size_t(start + x.channels()) <= mod.count(),
            "gate bounds");
    auto y = empty(x.shape);
    const bool half_product = dit_fp16 && delta.fp16_values && mod.fp16_values;
    const bool half_output = half_product && x.fp16_values;
    launch(half_product ? (half_output ? "hv15_half_gate" : "hv15_float_gate_half") : "gate_add",
           int((x.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, x.pointer, delta.pointer,
           mod.pointer, int(x.count()), x.channels(), start);
    y.fp16_values = half_output;
    return y;
}
Tensor Gpu::activate(Tensor x, int mode) {
    require(x.storage.use_count() == 1 && (mode == 3 || mode == 4 || mode == 5),
            "exclusive activation buffer required");
    CUdeviceptr zero = 0;
    launch("element", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, x.pointer, x.pointer, zero,
           zero, int(x.count()), x.channels(), mode, 1.f, 0.f);
    return dit_fp16 && x.fp16_values ? round_half(std::move(x)) : x;
}
std::array<Tensor, 3> Gpu::qkv_heads(Weights &w, const std::string &prefix, const Tensor &x,
                                     int height, int width, bool image) {
    require(x.channels() == 6144, "joint QKV shape");
    std::array<Tensor, 3> result{empty_half({x.rows(), 2048}), empty_half({x.rows(), 2048}),
                                 empty_half({x.rows(), 2048})};
    auto qw = weight(w, prefix + "_q_norm.weight"), kw = weight(w, prefix + "_k_norm.weight");
    for (auto &tensor : result)
        tensor.packed_heads = 16;
    launch(dit_fp16 ? "hv15_half_qkv" : "qkv_heads", x.rows(), 16, 1, 128, 1, 0, result[0].pointer,
           result[1].pointer, result[2].pointer, x.pointer, qw.pointer, kw.pointer, x.rows(),
           height, width, int(image));
    return result;
}
Tensor Gpu::bare_norm(const Tensor &x, int mode, float eps) {
    auto y = empty(x.shape);
    CUdeviceptr zero = 0;
    launch(dit_fp16 && mode == 0 && x.channels() % 4 == 0 ? "hv15_autocast_norm"
           : dit_fp16 && x.fp16_values && mode != 0       ? "hv15_half_norm"
                                                          : "hv15n_norm",
           x.rows(), 1, 1, 256, 1, 0, y.pointer, x.pointer, zero, zero, x.rows(), x.channels(),
           mode, eps);
    y.fp16_values = dit_fp16 && x.fp16_values && mode != 0;
    return y;
}
Tensor Gpu::norm(Weights &w, const std::string &prefix, const Tensor &x, int mode, float eps) {
    auto name = w.has(prefix + ".weight") ? prefix + ".weight" : prefix + ".gamma";
    auto weight_tensor = weight(w, name);
    require(weight_tensor.count() == size_t(x.channels()), "normalization weight shape mismatch");
    Tensor bias;
    if (w.has(prefix + ".bias"))
        bias = weight(w, prefix + ".bias");
    auto y = empty(x.shape);
    launch(dit_fp16 && mode == 0 && x.channels() % 4 == 0 ? "hv15_autocast_norm"
           : dit_fp16 && x.fp16_values && mode != 0       ? "hv15_half_norm"
                                                          : "hv15n_norm",
           x.rows(), 1, 1, 256, 1, 0, y.pointer, x.pointer, weight_tensor.pointer, bias.pointer,
           x.rows(), x.channels(), mode, eps);
    y.fp16_values = dit_fp16 && x.fp16_values && mode != 0;
    return y;
}
Tensor Gpu::op(const Tensor &x, int mode, const Tensor *second, const Tensor *bias, float scale,
               float offset) {
    auto y = empty(x.shape);
    CUdeviceptr z = second ? second->pointer : 0, b = bias ? bias->pointer : 0;
    if (mode == 1 || mode == 2)
        require(second && second->count() == x.count(), "elementwise size mismatch");
    if (mode == 6 || mode == 7 || mode == 8)
        require((mode == 6 ? bias : second) &&
                    ((mode == 6 ? bias : second)->count() == size_t(x.channels())),
                "broadcast size mismatch");
    if (dit_fp16 && mode == 7 && second->fp16_values) {
        require(bias && bias->count() == size_t(x.channels()), "modulation shift bounds");
        launch("hv15_modulate_values", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer,
               x.pointer, second->pointer, bias->pointer, int(x.count()), x.channels());
        return y;
    }
    const bool half_output = dit_fp16 && x.fp16_values && (!second || second->fp16_values) &&
                             (!bias || bias->fp16_values);
    launch(half_output ? "hv15_half_element" : "element", int((x.count() + 255) / 256), 1, 1, 256,
           1, 0, y.pointer, x.pointer, z, b, int(x.count()), x.channels(), mode, scale, offset);
    y.fp16_values = half_output;
    return y;
}
void Gpu::rotary(Tensor &x, int heads, int kind, int height, int width, float theta) {
    int dim = x.channels() / heads;
    require(dim % 2 == 0, "odd RoPE dimension");
    launch("rope", int((x.count() / 2 + 255) / 256), 1, 1, 256, 1, 0, x.pointer, x.rows(), heads,
           dim, kind, height, width, theta);
    if (dit_fp16 && x.fp16_values)
        x = round_half(std::move(x));
}
Tensor Gpu::attention(const Tensor &q, const Tensor &k, const Tensor &v, int heads, int kvheads,
                      int mask, int frame_hw, bool precise, const Tensor *bias, float scale) {
    int rows = q.rows(), dim = q.channels() / heads;
    require(rows == k.rows() && rows == v.rows() && k.channels() == kvheads * dim &&
                v.channels() == k.channels(),
            "attention shape mismatch");
    require(heads % kvheads == 0 && dim <= 2048, "unsupported attention heads");
    if (scale == 0.f)
        scale = 1.f / std::sqrt(float(dim));
    auto y = empty(q.shape);
    attention_calls++;
    if (precise || heads != kvheads || bias) {
        CUdeviceptr b = bias ? bias->pointer : 0;
        launch("attention_ieee", rows, heads, 1, 256, 1, 0, y.pointer, q.pointer, k.pointer,
               v.pointer, b, rows, heads, kvheads, dim, mask, frame_hw, scale);
        return dit_fp16 && !precise ? round_half(std::move(y)) : y;
    }
    if (dit_fp16 && aot_forward && !precise && !bias && !mask && dim == 128 && heads == kvheads &&
        rows >= 128) {
        const bool packed =
            q.packed_heads && q.packed_heads == k.packed_heads && q.packed_heads == v.packed_heads;
        auto aq = packed ? q : half(full(q));
        auto ak = packed ? k : half(full(k));
        auto av = packed ? v : half(full(v));
        auto output = empty_half(q.shape), lse = empty({heads, rows});
        auto fn = packed ? aot_heads : aot_forward;
        check(CUresult(fn(aq.pointer, ak.pointer, av.pointer, output.pointer, lse.pointer, rows,
                          heads, kvheads, dim, 2, scale, stream)),
              "standalone FP16 AOTriton attention");
        launch("convert_float", int((y.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer,
               output.pointer, int(y.count()));
        y.fp16_values = true;
        aot_calls++;
        return y;
    }
    if (optimized && (!vendor || dit_fp16) && dim == 128 && mask == 0 && heads == kvheads) {
        auto fq = full(q), fk = full(k), fv = full(v);
        CUfunction fn;
        check(cuModuleGetFunction(&fn, flash, dit_fp16 ? "video_attention_flex_f16"
                                                            : "video_attention_tile_f16"), "WMMA attention lookup");
        void *params[] = {&y.pointer, &fq.pointer, &fk.pointer, &fv.pointer, &rows,
                          &heads,     &dim,        &scale,      &kvheads,    &mask};
        poll();
        check(
            cuLaunchKernel(fn, heads, (rows + 127) / 128, 1, 256, 1, 1, 0, stream, params, nullptr),
            "WMMA attention");
        flash_calls++;
        return dit_fp16 && !precise ? round_half(std::move(y)) : y;
    }
    require(dim % 16 == 0, "tensor-core attention dimension must align to 16");
    constexpr int tile = 128;
    auto packed = empty({heads, rows, dim / 2}), keys = empty({heads, tile, dim / 2}),
         values = empty({heads, dim, tile / 2});
    auto scores = empty({heads, rows, tile}), prob = empty({heads, rows, tile / 2}),
         acc = empty({heads, rows, dim}), partial = empty({heads, rows, dim});
    auto maxima = upload(std::vector<float>(size_t(heads) * rows, -INFINITY), {heads, rows}),
         sums = upload(std::vector<float>(size_t(heads) * rows, 0.f), {heads, rows});
    check(cuMemsetD32Async(acc.pointer, 0, acc.count(), stream), "clear attention");
    launch("pack_heads", int((q.count() + 255) / 256), 1, 1, 256, 1, 0, packed.pointer, q.pointer,
           rows, heads, dim);
    CUfunction fn;

    check(cuModuleGetFunction(&fn, mma, "video_gemm_f16"), "attention GEMM lookup");
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
            int ntile = 32, mtile = 32;
            check(cuLaunchKernel(fn, (n + ntile - 1) / ntile, (m + mtile - 1) / mtile, heads, 128,
                                 1, 1, 0, stream, params, nullptr),
                  "attention repository GEMM");
            repo_gemm += heads;
        }
    };
    for (int start = 0; start < rows; start += tile) {
        int count = std::min(tile, rows - start);
        launch("pack_keys", (heads * tile * dim + 255) / 256, 1, 1, 256, 1, 0, keys.pointer,
               k.pointer, rows, heads, dim, start, tile, 0);
        launch("pack_keys", (heads * tile * dim + 255) / 256, 1, 1, 256, 1, 0, values.pointer,
               v.pointer, rows, heads, dim, start, tile, 1);
        batch(scores, packed, keys, rows, tile, dim);
        launch("softmax_tile", (rows * heads + 7) / 8, 1, 1, 256, 1, 0, scores.pointer,
               prob.pointer, acc.pointer, maxima.pointer, sums.pointer, rows, heads, dim, count,
               start, tile, mask, frame_hw, scale);
        batch(partial, prob, values, rows, dim, tile);
        // Keep the online accumulator resident across key tiles. Allocating
        // another full query/head output here forces a fence for every tile.
        CUdeviceptr no_bias = 0;
        launch("element", int((acc.count() + 255) / 256), 1, 1, 256, 1, 0, acc.pointer, acc.pointer,
               partial.pointer, no_bias, int(acc.count()), acc.channels(), 1, 1.f, 0.f);
    }
    launch("finish_attention", int((q.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer,
           acc.pointer, sums.pointer, rows, heads, dim);
    return dit_fp16 && !precise ? round_half(std::move(y)) : y;
}
Tensor Gpu::conv(Weights &w, const std::string &prefix, const Tensor &x, bool causal) {
    require(x.shape.size() == 4, "convolution requires THWC");
    auto ws = w.shape(prefix + ".weight");
    require(ws.size() == 5 && ws[1] == x.channels(), "convolution shape mismatch");
    int kt = ws[2], kh = ws[3], kw = ws[4], t = x.shape[0], h = x.shape[1], width = x.shape[2];
    auto packed = packed_weights.find(w.identity + ":conv:" + prefix);
    auto weights = optimized && active_weight_file == w.identity && packed != packed_weights.end()
                       ? packed->second
                   : optimized ? weight_half(w, prefix + ".weight")
                               : weight(w, prefix + ".weight");
    weights.shape = {ws[0], ws[1] * kt * kh * kw};
    auto y = empty({t, h, width, ws[0]});
    Tensor bias;
    if (w.has(prefix + ".bias"))
        bias = weight(w, prefix + ".bias");
    if (optimized && kt == 1 && kh == 1 && kw == 1) {
        auto flat = x;
        flat.shape = {x.rows(), x.channels()};
        auto dest = y;
        dest.shape = {y.rows(), y.channels()};
        matmul(flat, weights, false, &dest);
        // Autocast MIOpen materializes the FP16 convolution before bias.
        if (dit_fp16)
            y = round_half(std::move(y));
        if (bias.pointer) {
            CUdeviceptr zero = 0;
            launch("element", int((y.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, y.pointer,
                   zero, bias.pointer, int(y.count()), y.channels(), 6, 1.f, 0.f);
        }
        conv_chunks++;
        return dit_fp16 ? round_half(std::move(y)) : y;
    }
    int chunk =
        optimized ? std::max(1, std::min(32768, int((128ull << 20) / (weights.channels() * 2ull))))
                  : std::max(1, std::min(1024, int((32ull << 20) / (weights.channels() * 4ull))));
    if (optimized && chunk >= 64)
        chunk = chunk / 64 * 64;
    auto im2col_input = full(x);
    for (int start = 0; start < x.rows(); start += chunk) {
        int count = std::min(chunk, x.rows() - start);
        auto col = optimized ? empty_half({count, weights.channels()})
                             : empty({count, weights.channels()});
        launch(optimized ? "im2col_half" : "im2col", int((col.count() + 255) / 256), 1, 1, 256, 1,
               0, col.pointer, im2col_input.pointer, t, h, width, x.channels(), kt, kh, kw,
               causal ? kt - 1 : 0, kh / 2, kw / 2, causal ? 1 : 0, start, count);
        auto dest = rows(y, start, count);
        auto out = matmul(col, weights, false, optimized ? &dest : nullptr);
        if (dit_fp16)
            out = round_half(std::move(out));
        if (bias.pointer) {
            if (optimized) {
                CUdeviceptr zero = 0;
                launch("element", int((out.count() + 255) / 256), 1, 1, 256, 1, 0, out.pointer,
                       out.pointer, zero, bias.pointer, int(out.count()), out.channels(), 6, 1.f,
                       0.f);
            } else
                out = op(out, 6, nullptr, &bias);
        }
        if (!optimized)
            check(cuMemcpyDtoDAsync(y.pointer + size_t(start) * ws[0] * 4, out.pointer,
                                    out.count() * 4, stream),
                  "convolution result");
        conv_chunks++;
    }
    return dit_fp16 ? round_half(std::move(y)) : y;
}
Tensor Gpu::channel_map(const Tensor &x, int channels, bool mean) {
    require(mean ? x.channels() % channels == 0 : channels % x.channels() == 0,
            "channel map mismatch");
    auto s = x.shape;
    s.back() = channels;
    auto y = empty(s);
    launch("channel_map", int((y.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, x.pointer,
           x.rows(), x.channels(), channels, int(mean));
    return y;
}
Tensor Gpu::mean(const Tensor &x) {
    auto y = empty({1, x.channels()});
    launch("mean_rows", (x.channels() + 255) / 256, 1, 1, 256, 1, 0, y.pointer, x.pointer, x.rows(),
           x.channels());
    return y;
}
void Gpu::dump(const Tensor &x, const fs::path &dir, const std::string &name, bool video) {
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
    meta << "],\"dtype\":\"float32\",\"layout\":\""
         << (video               ? "NCTHW"
             : shape.size() == 4 ? "THWC"
                                 : "NTC")
         << "\"}\n";
    require(bool(meta), "dump metadata failed");
}
std::string Gpu::metrics() const {
    std::ostringstream s;
    s << "{\"backend\":\"hv15n_rocm\",\"repo_gemm_calls\":" << repo_gemm
      << ",\"hipblas_gemm_calls\":" << blas_gemm << ",\"fallback_gemm_calls\":" << fallback_gemm
      << ",\"attention_calls\":" << attention_calls
      << ",\"managed_peak_vram_mib\":" << double(peak) / (1024 * 1024)
      << ",\"optimized\":" << (optimized ? "true" : "false")
      << ",\"device_allocations\":" << allocations << ",\"buffer_reuses\":" << buffer_reuses
      << ",\"weight_cache_hits\":" << weight_hits << ",\"upload_bytes\":" << upload_bytes
      << ",\"flash_calls\":" << flash_calls << ",\"gemm_v7_calls\":" << gemm_v7_calls
      << ",\"ieee_tiled_calls\":" << ieee_tiled_calls << ",\"conv_chunks\":" << conv_chunks
      << ",\"aotriton_attention_calls\":" << aot_calls
      << ",\"wmma_f16_tiled_calls\":" << f16_tiled_calls << ",\"prefetch_bytes\":" << prefetch_bytes
      << ",\"pinned_staging_mib\":" << double(pinned_bytes) / 1048576
      << ",\"parity\":\"unverified\"}";
    return s.str();
}
} // namespace hv15n
