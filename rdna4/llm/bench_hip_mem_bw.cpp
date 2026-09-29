/* Minimal device-to-device bandwidth probe for ROCm clock diagnostics. */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_HIP(call) do { \
    hipError_t error = (call); \
    if (error != hipSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); \
        return 1; \
    } \
} while (0)

__global__ void copy_kernel(float4 *dst, const float4 *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

static int copy_h2d(const char *label, void *dst, const void *src,
                    size_t copy_bytes, int copies) {
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipMemcpyAsync(dst, src, copy_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < copies; ++i)
        CHECK_HIP(hipMemcpyAsync(dst, src, copy_bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    double gib = (double)copy_bytes * copies / (double)(1ull << 30);
    std::printf("%s: %.2f GiB/s (%d x %zu KiB, %.2f ms)\n",
                label, gib / (ms / 1000.0), copies, copy_bytes / 1024, ms);
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    return 0;
}

int main() {
    constexpr size_t bytes = 256ull << 20;
    constexpr size_t n = bytes / sizeof(float4);
    float4 *src = nullptr, *dst = nullptr;
    CHECK_HIP(hipMalloc(&src, bytes));
    CHECK_HIP(hipMalloc(&dst, bytes));
    CHECK_HIP(hipMemset(src, 0, bytes));
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    const dim3 grid((unsigned)((n + 255) / 256));
    copy_kernel<<<grid, 256>>>(dst, src, n);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < 16; ++i) copy_kernel<<<grid, 256>>>(dst, src, n);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    std::printf("device copy: %.2f GiB/s (16 x 256 MiB, %.2f ms)\n",
                8.0 / (ms / 1000.0), ms);
    void *host_pinned = nullptr;
    CHECK_HIP(hipHostMalloc(&host_pinned, bytes));
    std::memset(host_pinned, 0, bytes);
    void *host_pageable = std::malloc(bytes);
    if (!host_pageable) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }
    std::memset(host_pageable, 0, bytes);
    if (copy_h2d("pinned H2D large", dst, host_pinned, bytes, 16) ||
        copy_h2d("pageable H2D large", dst, host_pageable, bytes, 16) ||
        copy_h2d("pinned H2D expert", dst, host_pinned, 512ull << 10, 8192) ||
        copy_h2d("pageable H2D expert", dst, host_pageable, 512ull << 10, 8192))
        return 1;
    std::free(host_pageable);
    CHECK_HIP(hipHostFree(host_pinned));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipFree(dst));
    CHECK_HIP(hipFree(src));
    return 0;
}
