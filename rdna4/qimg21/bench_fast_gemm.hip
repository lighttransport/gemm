// SPDX-License-Identifier: MIT
// Parity + throughput test: tiled vs reference W8A8 GEMM from libq21_hip_fast_gemm.so (exact INT32 math ->
// outputs must be bit-identical). usage: bench_fast_gemm <lib.so>
#include <hip/hip_runtime.h>
#pragma clang diagnostic ignored "-Wunused-value"
#pragma clang diagnostic ignored "-Wunused-result"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
typedef int (*gemm_fn)(void*,int,const void*,const float*,const void*,const float*,int,int,int,void*);
int main(int argc, char **argv) {
    void *lib = dlopen(argc > 1 ? argv[1] : "rdna4/qimg21/libq21_hip_fast_gemm.so", RTLD_NOW);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    gemm_fn ref = (gemm_fn)dlsym(lib, "q21f_i8_gemm_reference"), til = (gemm_fn)dlsym(lib, "q21f_i8_gemm_tiled");
    int shapes[][3] = {{8320, 4096, 4096}, {8320, 12288, 4096}, {8320, 24576, 4096}, {8320, 4096, 12288}, {200, 4096, 4096}, {4160, 4096, 4096}};
    std::mt19937 rng(1);
    int fails = 0;
    for (auto &sh : shapes) {
        int m = sh[0], n = sh[1], k = sh[2];
        std::vector<signed char> hx((size_t)m * k), hw((size_t)n * k);
        std::vector<float> hxs(m), hws(n);
        for (auto &v : hx) v = (signed char)(rng() % 255 - 127);
        for (auto &v : hw) v = (signed char)(rng() % 255 - 127);
        for (auto &v : hxs) v = 1e-3f * (1 + rng() % 100);
        for (auto &v : hws) v = 1e-3f * (1 + rng() % 100);
        void *x, *w, *y0, *y1; float *xs, *ws;
        hipMalloc(&x, hx.size()); hipMalloc(&w, hw.size()); hipMalloc(&xs, m * 4); hipMalloc(&ws, n * 4);
        hipMalloc(&y0, (size_t)m * n * 2); hipMalloc(&y1, (size_t)m * n * 2);
        hipMemcpy(x, hx.data(), hx.size(), hipMemcpyHostToDevice); hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice);
        hipMemcpy(xs, hxs.data(), m * 4, hipMemcpyHostToDevice); hipMemcpy(ws, hws.data(), n * 4, hipMemcpyHostToDevice);
        hipMemset(y0, 0, (size_t)m * n * 2); hipMemset(y1, 0xff, (size_t)m * n * 2);
        ref(y0, n, x, xs, w, ws, m, n, k, 0); til(y1, n, x, xs, w, ws, m, n, k, 0); hipDeviceSynchronize();
        std::vector<unsigned short> a((size_t)m * n), b((size_t)m * n);
        hipMemcpy(a.data(), y0, a.size() * 2, hipMemcpyDeviceToHost); hipMemcpy(b.data(), y1, b.size() * 2, hipMemcpyDeviceToHost);
        size_t diff = 0; for (size_t i = 0; i < a.size(); i++) diff += a[i] != b[i];
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        float t[2];
        for (int which = 0; which < 2; which++) {
            gemm_fn f = which ? til : ref; int reps = which ? 10 : 2;
            f(which ? y1 : y0, n, x, xs, w, ws, m, n, k, 0); hipDeviceSynchronize();
            hipEventRecord(e0); for (int r = 0; r < reps; r++) f(which ? y1 : y0, n, x, xs, w, ws, m, n, k, 0);
            hipEventRecord(e1); hipEventSynchronize(e1); hipEventElapsedTime(&t[which], e0, e1); t[which] /= reps;
        }
        double ops = 2.0 * m * n * k;
        printf("m=%5d n=%5d k=%5d  ref %8.2f ms (%6.1f TOPS)  tiled %8.2f ms (%6.1f TOPS)  speedup %5.1fx  mismatches %zu\n",
               m, n, k, t[0], ops / t[0] / 1e9, t[1], ops / t[1] / 1e9, t[0] / t[1], diff);
        fails += diff != 0;
        hipFree(x); hipFree(w); hipFree(xs); hipFree(ws); hipFree(y0); hipFree(y1);
    }
    printf(fails ? "FAIL\n" : "PASS (bit-identical)\n");
    return fails != 0;
}
