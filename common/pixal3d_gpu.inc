/* Shared CUDA/HIP implementation; compiled by each backend's own compiler.
 * Legacy calls exchange host F32 arrays; the resident protocol uses opaque
 * device buffers. SPDX-License-Identifier: MIT */
#include "pixal3d_device.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef PX_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#if __has_include(<hipblas/hipblas.h>)
#include <hipblas/hipblas.h>
#else
#include "pixal3d_hipblas_abi.h"
#endif
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuCopyKind hipMemcpyKind
#define gpuStream hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamDestroy hipStreamDestroy
#define blasSetStream hipblasSetStream
#define gpuH2D hipMemcpyHostToDevice
#define gpuD2H hipMemcpyDeviceToHost
#define gpuSetDevice hipSetDevice
#define gpuMemGetInfo hipMemGetInfo
#define gpuDeviceProp hipDeviceProp_t
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuGetLastError hipGetLastError
#define gpuEvent hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventDestroy hipEventDestroy
#define blasHandle hipblasHandle_t
#define blasCreate hipblasCreate
#define blasDestroy hipblasDestroy
#define blasSgemm hipblasSgemm
#define blasGemmEx hipblasGemmEx
#define OP_T HIPBLAS_OP_T
#define OP_N HIPBLAS_OP_N
#define BF16_TYPE HIP_R_16BF
#define F16_TYPE HIP_R_16F
#define F32_TYPE HIP_R_32F
#define COMPUTE_F32 HIPBLAS_COMPUTE_32F
#define ALGO_DEFAULT HIPBLAS_GEMM_DEFAULT
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuCopyKind cudaMemcpyKind
#define gpuStream cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamDestroy cudaStreamDestroy
#define blasSetStream cublasSetStream
#define gpuH2D cudaMemcpyHostToDevice
#define gpuD2H cudaMemcpyDeviceToHost
#define gpuSetDevice cudaSetDevice
#define gpuMemGetInfo cudaMemGetInfo
#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuGetLastError cudaGetLastError
#define gpuEvent cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventDestroy cudaEventDestroy
#define blasHandle cublasHandle_t
#define blasCreate cublasCreate
#define blasDestroy cublasDestroy
#define blasSgemm cublasSgemm
#define blasGemmEx cublasGemmEx
#define OP_T CUBLAS_OP_T
#define OP_N CUBLAS_OP_N
#define BF16_TYPE CUDA_R_16BF
#define F16_TYPE CUDA_R_16F
#define F32_TYPE CUDA_R_32F
#define COMPUTE_F32 CUBLAS_COMPUTE_32F
#define ALGO_DEFAULT CUBLAS_GEMM_DEFAULT
#endif

static void check(int code, const char *name) {
    if (code)
        throw std::runtime_error(std::string(name) + " failed: " + std::to_string(code));
}
#define GPU(call) check(int(call), #call)
struct Device {
    blasHandle blas{};
    gpuStream stream{};
    int ordinal = 0;
    size_t budget = 0, live = 0, peak = 0;
    std::string error;
    bool pool_enabled = false, profile = false;
    int kernels = 0;
    bool mma_supported = false;
    std::vector<std::pair<void *, size_t>> pool;
    px_device_metrics metrics{};
    ~Device() {
        (void)gpuSetDevice(ordinal);
        if (stream)
            (void)gpuStreamSynchronize(stream);
        for (auto b : pool)
            (void)gpuFree(b.first);
        if (blas)
            (void)blasDestroy(blas);
        if (stream)
            (void)gpuStreamDestroy(stream);
    }
};
static int transfer(Device &d, void *dst, const void *src, size_t bytes, gpuCopyKind kind) {
    GPU(gpuMemcpyAsync(dst, src, bytes, kind, d.stream));
    GPU(gpuStreamSynchronize(d.stream));
    if (kind == gpuH2D)
        d.metrics.uploads += bytes;
    else if (kind == gpuD2H)
        d.metrics.downloads += bytes;
    return 0;
}
struct Buffer {
    Device &d;
    void *p = nullptr;
    size_t bytes;
    Buffer(Device &dev, size_t n) : d(dev), bytes(n) {
        GPU(gpuSetDevice(d.ordinal));
        if (d.pool_enabled) {
            auto best = d.pool.end();
            for (auto it = d.pool.begin(); it != d.pool.end(); ++it)
                if (it->second >= n && it->second <= n + n / 4 &&
                    (best == d.pool.end() || it->second < best->second))
                    best = it;
            if (best != d.pool.end()) {
                p = best->first;
                bytes = best->second;
                d.pool.erase(best);
                return;
            }
            while (n > d.budget - d.live && !d.pool.empty()) {
                auto b = d.pool.back();
                d.pool.pop_back();
                GPU(gpuFree(b.first));
                d.live -= b.second;
            }
        }
        if (n > d.budget - d.live)
            throw std::runtime_error("Pixal3D GPU workspace exceeds memory budget");
        GPU(gpuMalloc(&p, n));
        d.metrics.allocations++;
        d.live += n;
        d.peak = std::max(d.peak, d.live);
    }
    ~Buffer() {
        if (p) {
            (void)gpuSetDevice(d.ordinal);
            if (d.pool_enabled) {
                try {
                    d.pool.emplace_back(p, bytes);
                    return;
                } catch (...) {
                    // Release directly if the host cannot grow the free list.
                }
            }
            (void)gpuFree(p);
            d.live -= bytes;
        }
    }
    float *f() { return static_cast<float *>(p); }
};
__device__ float round_bfloat(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7f800000u) != 0x7f800000u)
        u += 0x7fff + ((u >> 16) & 1);
    return __uint_as_float(u & 0xffff0000u);
}
__device__ float round_precision(float x, int precision) {
    return precision == 1 ? round_bfloat(x) : precision == 2 ? __half2float(__float2half_rn(x)) : x;
}
__global__ void cast_bfloat(unsigned short *y, const float *x, size_t n, int precision) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = precision == 1 ? __float_as_uint(round_bfloat(x[i])) >> 16
                              : __half_as_ushort(__float2half_rn(x[i]));
}
__global__ void finish_gemm(float *y, const float *bias, size_t n, int c, int bf) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = y[i] + (bias ? round_precision(bias[i % c], bf) : 0);
        y[i] = round_precision(v, bf);
    }
}
__global__ void softmax_rows(float *s, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows)
        return;
    __shared__ float tmp[256];
    float m = -INFINITY;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        m = fmaxf(m, s[size_t(row) * cols + j]);
    tmp[threadIdx.x] = m;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (threadIdx.x < n)
            tmp[threadIdx.x] = fmaxf(tmp[threadIdx.x], tmp[threadIdx.x + n]);
        __syncthreads();
    }
    m = tmp[0];
    float sum = 0;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = expf(s[size_t(row) * cols + j] - m);
        s[size_t(row) * cols + j] = v;
        sum += v;
    }
    tmp[threadIdx.x] = sum;
    __syncthreads();
    for (int n = 128; n; n /= 2) {
        if (threadIdx.x < n)
            tmp[threadIdx.x] += tmp[threadIdx.x + n];
        __syncthreads();
    }
    sum = tmp[0];
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        s[size_t(row) * cols + j] /= sum;
}
extern "C" void *px_gpu_create(int device, size_t budget) {
    Device *d = new Device;
    try {
        d->ordinal = device;
        GPU(gpuSetDevice(device));
        gpuDeviceProp properties{};
        GPU(gpuGetDeviceProperties(&properties, device));
#ifdef PX_HIP
        d->mma_supported = !std::strncmp(properties.gcnArchName, "gfx1200", 7) ||
                           !std::strncmp(properties.gcnArchName, "gfx1201", 7);
#else
        d->mma_supported = properties.major >= 8;
#endif
        size_t free, total;
        GPU(gpuMemGetInfo(&free, &total));
        if (free <= 512ull * 1024 * 1024)
            throw std::runtime_error("Insufficient free GPU memory");
        d->budget = std::min<size_t>(budget, free - 512ull * 1024 * 1024);
        GPU(blasCreate(&d->blas));
        GPU(gpuStreamCreate(&d->stream));
        GPU(blasSetStream(d->blas, d->stream));
        return d;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Pixal3D: %s\n", e.what());
        delete d;
        return nullptr;
    }
}
extern "C" void px_gpu_destroy(void *p) {
    auto *d = static_cast<Device *>(p);
    if (d) {
        (void)gpuSetDevice(d->ordinal);
        delete d;
    }
}
extern "C" size_t px_gpu_peak(void *p) {
    return static_cast<Device *>(p)->peak;
}
extern "C" const char *px_gpu_error(void *p) {
    return static_cast<Device *>(p)->error.c_str();
}
extern "C" int px_gpu_gemm(void *p, float *out, const float *x, const float *w, const float *bias, int rows,
                           int co, int ci, int bf) {
    Device &d = *static_cast<Device *>(p);
    try {
        GPU(gpuSetDevice(d.ordinal));
        size_t nw = size_t(co) * ci;
        Buffer wf(d, nw * 4), b(d, size_t(co) * 4);
        GPU(transfer(d, wf.p, w, nw * 4, gpuH2D));
        if (bias)
            GPU(transfer(d, b.p, bias, size_t(co) * 4, gpuH2D));
        Buffer wb(d, bf ? nw * 2 : 2);
        if (bf)
            cast_bfloat<<<(nw + 255) / 256, 256, 0, d.stream>>>((unsigned short *)wb.p, wf.f(), nw, bf);
        int tile = std::min(rows, 1024);
        Buffer xf(d, size_t(tile) * ci * 4), xb(d, bf ? size_t(tile) * ci * 2 : 2),
            y(d, size_t(tile) * co * 4);
        float one = 1, zero = 0;
        for (int start = 0; start < rows; start += tile) {
            int n = std::min(tile, rows - start);
            size_t nx = size_t(n) * ci, ny = size_t(n) * co;
            GPU(transfer(d, xf.p, x + size_t(start) * ci, nx * 4, gpuH2D));
            if (bf) {
                cast_bfloat<<<(nx + 255) / 256, 256, 0, d.stream>>>((unsigned short *)xb.p, xf.f(), nx, bf);
                auto type = bf == 1 ? BF16_TYPE : F16_TYPE;
                GPU(blasGemmEx(d.blas, OP_T, OP_N, co, n, ci, &one, wb.p, type, ci, xb.p, type, ci, &zero,
                               y.p, F32_TYPE, co, COMPUTE_F32, ALGO_DEFAULT));
            } else
                GPU(blasSgemm(d.blas, OP_T, OP_N, co, n, ci, &one, wf.f(), ci, xf.f(), ci, &zero, y.f(), co));
            d.metrics.gemms++;
            finish_gemm<<<(ny + 255) / 256, 256, 0, d.stream>>>(y.f(), bias ? b.f() : nullptr, ny, co, bf);
            GPU(transfer(d, out + size_t(start) * co, y.p, ny * 4, gpuD2H));
        }
        return 0;
    } catch (const std::exception &e) {
        d.error = e.what();
        return -1;
    }
}
extern "C" int px_gpu_attention(void *p, float *out, const float *q, const float *k, const float *v, int n,
                                int m, int heads, int hd) {
    Device &d = *static_cast<Device *>(p);
    try {
        GPU(gpuSetDevice(d.ordinal));
        d.metrics.attentions++;
        int c = heads * hd, tile = std::min(n, 128);
        size_t nk = size_t(m) * c, nq = size_t(n) * c;
        Buffer dq(d, nq * 4), dk(d, nk * 4), dv(d, nk * 4), y(d, nq * 4), s(d, size_t(tile) * m * 4);
        GPU(transfer(d, dq.p, q, nq * 4, gpuH2D));
        GPU(transfer(d, dk.p, k, nk * 4, gpuH2D));
        GPU(transfer(d, dv.p, v, nk * 4, gpuH2D));
        float scale = 1 / std::sqrt(float(hd)), one = 1, zero = 0;
        for (int h = 0; h < heads; ++h)
            for (int start = 0; start < n; start += tile) {
                int rows = std::min(tile, n - start);
                GPU(blasSgemm(d.blas, OP_T, OP_N, m, rows, hd, &scale, dk.f() + h * hd, c,
                              dq.f() + size_t(start) * c + h * hd, c, &zero, s.f(), m));
                softmax_rows<<<rows, 256, 0, d.stream>>>(s.f(), rows, m);
                GPU(blasSgemm(d.blas, OP_N, OP_N, hd, rows, m, &one, dv.f() + h * hd, c, s.f(), m, &zero,
                              y.f() + size_t(start) * c + h * hd, c));
            }
        GPU(transfer(d, out, y.p, nq * 4, gpuD2H));
        return 0;
    } catch (const std::exception &e) {
        d.error = e.what();
        return -1;
    }
}

#include "pixal3d_resident.inc"
