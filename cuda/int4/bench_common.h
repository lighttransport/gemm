#ifndef INT4_BENCH_COMMON_H
#define INT4_BENCH_COMMON_H
#include "int4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static inline void cuda_check(CUresult result, const char *expr, int line) {
    if (result != CUDA_SUCCESS) {
        const char *message = "unknown CUDA error";
        if (cuGetErrorString) cuGetErrorString(result, &message);
        fprintf(stderr, "line %d: %s: %s (%d)\n", line, expr, message, result);
        exit(1);
    }
}
#define CUDA(call) cuda_check((call), #call, __LINE__)
#define REQUIRE(expr) do { if (!(expr)) { fprintf(stderr, "line %d: %s FAILED\n", __LINE__, #expr); exit(1); } } while (0)

static inline void *host_alloc(size_t size) {
    void *p = malloc(size);
    REQUIRE(p != NULL); return p;
}

static inline CUdeviceptr device_alloc(size_t size) {
    CUdeviceptr p; CUDA(cuMemAlloc(&p, size)); return p;
}

static inline int4_matrix matrix_alloc(int rows, int k) {
    int4_matrix m = {0, 0, rows, k, int4_padded_k(k)};
    m.codes = device_alloc(int4_codes_size(rows, k));
    m.scales = device_alloc(int4_scales_size(rows, k));
    return m;
}

static inline void matrix_free(int4_matrix m) { CUDA(cuMemFree(m.codes)); CUDA(cuMemFree(m.scales)); }

static inline uint32_t random_u32(uint32_t *state) {
    uint32_t x = *state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *state = x; return x;
}

static inline void default_cubin_dir(char *dst, size_t size) {
    ssize_t n = readlink("/proc/self/exe", dst, size - 1);
    REQUIRE(n > 0 && (size_t)n < size - 1);
    dst[n] = 0;
    char *slash = strrchr(dst, '/'); REQUIRE(slash != NULL); *slash = 0;
}

static inline CUcontext start_cuda(void) {
    REQUIRE(cuewInit(CUEW_INIT_CUDA) == CUEW_SUCCESS);
    CUDA(cuInit(0));
    CUdevice dev; CUcontext context;
    CUDA(cuDeviceGet(&dev, 0)); CUDA(cuCtxCreate(&context, 0, dev));
    return context;
}

static inline float launch_time(CUfunction fn, int blocks, int threads, void **args, int iters) {
    CUevent begin, end;
    CUDA(cuEventCreate(&begin, 0)); CUDA(cuEventCreate(&end, 0));
    CUDA(cuEventRecord(begin, NULL));
    for (int i = 0; i < iters; i++) CUDA(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, NULL, args, NULL));
    CUDA(cuEventRecord(end, NULL)); CUDA(cuEventSynchronize(end));
    float ms; CUDA(cuEventElapsedTime(&ms, begin, end));
    CUDA(cuEventDestroy(begin)); CUDA(cuEventDestroy(end)); return ms / iters;
}
#endif
