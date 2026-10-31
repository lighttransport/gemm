// SPDX-License-Identifier: MIT
// Throughput of the Edit-2511 runner's tiled WMMA GEMMs (qimg_gemm_wmma.hip) on the 1024^2 + one-reference
// image stream (M = 8192 tokens): INT4-g64 (narrow + wide loaders), INT8-g64 and dense BF16 weights, so the
// loop ceiling (dense) and the decode cost (int4/int8) separate. usage: bench_qgemm
#include <hip/hip_runtime.h>
#pragma clang diagnostic ignored "-Wunused-result"
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include "qimg_gemm_wmma.hip"

typedef void (*qk_t)(float *, const unsigned char *, const unsigned short *, const unsigned short *, const float *, int, int, int, int,
                     const unsigned short *, const unsigned char *, int);

int main() {
    const int M = 8192;
    int shapes[][2] = {{3072, 3072}, {12288, 3072}, {3072, 12288}};   /* N, K */
    std::mt19937 rng(1);
    for (auto &sh : shapes) {
        int N = sh[0], K = sh[1];
        std::vector<unsigned short> hx((size_t)M * K), hw((size_t)N * K), hs((size_t)N * K / 64);
        std::vector<unsigned char> h4((size_t)N * K / 2), h8((size_t)N * K);
        for (auto &v : hx) v = 0x3f80 ^ (rng() & 0x80ff); for (auto &v : hw) v = 0x3c00 ^ (rng() & 0x80ff);
        for (auto &v : hs) v = 0x3c00; for (auto &v : h4) v = rng(); for (auto &v : h8) v = rng();
        unsigned short *x, *w, *s; unsigned char *w4, *w8; float *y;
        hipMalloc(&x, hx.size() * 2); hipMalloc(&w, hw.size() * 2); hipMalloc(&s, hs.size() * 2);
        hipMalloc(&w4, h4.size()); hipMalloc(&w8, h8.size()); hipMalloc(&y, (size_t)M * N * 4);
        hipMemcpy(x, hx.data(), hx.size() * 2, hipMemcpyHostToDevice); hipMemcpy(w, hw.data(), hw.size() * 2, hipMemcpyHostToDevice);
        hipMemcpy(s, hs.data(), hs.size() * 2, hipMemcpyHostToDevice); hipMemcpy(w4, h4.data(), h4.size(), hipMemcpyHostToDevice);
        hipMemcpy(w8, h8.data(), h8.size(), hipMemcpyHostToDevice);
        struct { const char *name; qk_t k; const unsigned char *W; const unsigned short *S; int threads; } ks[] = {
            {"i4w 128x128w8", qgemm_i4w_128x128w8, w4, s, 256}, {"i4  128x128w8", qgemm_i4_128x128w8, w4, s, 256},
            {"i8  128x128w8", qgemm_i8_128x128w8, w8, s, 256}, {"bf16 128x128w8", qgemm_bf16_128x128w8, (const unsigned char *)w, nullptr, 256},
            {"i4  128x128w4", qgemm_i4_128x128, w4, s, 128}, {"bf16 128x128w4", qgemm_bf16_128x128, (const unsigned char *)w, nullptr, 128},
            {"i4  128x256w8", qgemm_i4_128x256, w4, s, 256}};
        for (auto &k : ks) {
            int bn = strstr(k.name, "x256") ? 256 : 128; dim3 grid(((M + 127) / 128) * ((N + bn - 1) / bn));
            auto run = [&]() { k.k<<<grid, k.threads>>>(y, k.W, k.S, x, nullptr, N, K, M, 0, nullptr, nullptr, 0); };
            run(); hipDeviceSynchronize();
            hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); hipEventRecord(a);
            for (int r = 0; r < 10; r++) run();
            hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms, a, b); ms /= 10;
            printf("M=%d N=%5d K=%5d %-15s %6.2f ms %6.1f TF\n", M, N, K, k.name, ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12);
        }
        hipFree(x); hipFree(w); hipFree(s); hipFree(w4); hipFree(w8); hipFree(y);
    }
}
