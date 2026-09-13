/* SPDX-License-Identifier: MIT */
#pragma once
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define HIP_OK(expr)                                                                               \
    do {                                                                                           \
        hipError_t e = (expr);                                                                     \
        if (e != hipSuccess) {                                                                     \
            std::fprintf(stderr, "%s: %s\n", #expr, hipGetErrorString(e));                         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)
template <typename F> static double timed(F fn, int iterations) {
    for (int i = 0; i < 10; i++)
        fn();
    HIP_OK(hipDeviceSynchronize());
    hipEvent_t start, end;
    HIP_OK(hipEventCreate(&start));
    HIP_OK(hipEventCreate(&end));
    HIP_OK(hipEventRecord(start));
    for (int i = 0; i < iterations; i++)
        fn();
    HIP_OK(hipEventRecord(end));
    HIP_OK(hipEventSynchronize(end));
    float ms;
    HIP_OK(hipEventElapsedTime(&ms, start, end));
    HIP_OK(hipEventDestroy(start));
    HIP_OK(hipEventDestroy(end));
    return ms / iterations;
}
static float value(unsigned i) {
    unsigned x = i * 747796405U + 2891336453U;
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737U;
    x = (x >> 22) ^ x;
    return (float)(x >> 8) * (2.0f / 16777216) - 1;
}
