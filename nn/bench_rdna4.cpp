/* SPDX-License-Identifier: MIT
 * Bounded RDNA4 hardware diagnostic. Validates against independent CPU double
 * dots, and separates useful GEMM work, padded WMMA issues, and packing cost.
 */
#define GN_HIP 1
#include "gn_kernels.cu"
#ifdef GN_HIPBLASLT
#include "gn_hipblaslt.h"
#endif
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
template <int MR, int NR, int BK = 32>
__global__ __launch_bounds__(128) void tile_variant(float *y, const unsigned short *a,
                                                    const unsigned short *b, int M, int N, int K,
                                                    int add) {
    gn_rdna4_body<true, MR, NR, BK>(y, a, b, M, N, K, add);
}
/* Register issue diagnostic is deliberately NOT called a GEMM/ML throughput. */
__global__ void issue_ceiling(float *out, int steps) {
    ushort8 a = {0x3b00, 0x3b00, 0x3b00, 0x3b00, 0x3b00, 0x3b00, 0x3b00, 0x3b00};
    float8 acc[8] = {};
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++)
            acc[j][i] = (float)(j * 8 + i) / 128;
    for (int i = 0; i < steps; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, a, acc[j]);
    float sum = 0;
    for (int j = 0; j < 8; j++)
        for (int i = 0; i < 8; i++)
            sum += acc[j][i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
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
int main(int argc, char **argv) {
    if (argc != 1 && argc != 7 && argc != 8 && argc != 9) {
        std::fprintf(stderr, "usage: bench_rdna4 [M N K transpose_A transpose_B iterations "
                             "[dense_peak_tflops [mode]]]\n");
        return 2;
    }
    int M = argc >= 7 ? std::atoi(argv[1]) : 1296, N = argc >= 7 ? std::atoi(argv[2]) : 256;
    int K = argc >= 7 ? std::atoi(argv[3]) : 2304, ta = argc >= 7 ? std::atoi(argv[4]) : 0;
    int tb = argc >= 7 ? std::atoi(argv[5]) : 1, iterations = argc >= 7 ? std::atoi(argv[6]) : 50;
    double peak = 195; /* RX 9070 XT nominal dense 16-bit matrix reference. */
    if (argc >= 8) {
        char *end;
        peak = std::strtod(argv[7], &end);
        if (*end || !std::isfinite(peak) || peak <= 0)
            return 2;
    }
    if (M < 1 || N < 1 || K < 1 || M > 8192 || N > 8192 || K > 8192 || ta < 0 || ta > 1 || tb < 0 ||
        tb > 1 || iterations < 1 || iterations > 1000)
        return 2;
    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    if (std::strncmp(prop.gcnArchName, "gfx1201", 7))
        return 77;
    std::printf("{\"device\":\"%s\",\"arch\":\"%s\",\"hip_multiprocessors_WGPs\":%d,\"M\":%d,\"N\":"
                "%d,\"K\":%d,\"ta\":%d,\"tb\":%d}\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount, M, N, K, ta, tb);
    std::printf("{\"nominal_dense_matrix_tflops\":%.6g,\"target_fraction\":0.75,\"not_end_to_end_"
                "training_gate\":true}\n",
                peak);
    float *a, *b, *y, *high, *low, *sink;
    unsigned short *pa, *pb;
    int stride = (K + 31) & ~31, blocks = prop.multiProcessorCount * 4;
    HIP_OK(hipMalloc(&a, (size_t)M * K * 4));
    HIP_OK(hipMalloc(&b, (size_t)N * K * 4));
    HIP_OK(hipMalloc(&y, (size_t)M * N * 4));
    HIP_OK(hipMalloc(&high, (size_t)M * N * 4));
    HIP_OK(hipMalloc(&low, (size_t)M * N * 4));
    HIP_OK(hipMalloc(&pa, (size_t)M * stride * 6));
    HIP_OK(hipMalloc(&pb, (size_t)N * stride * 6));
    HIP_OK(hipMalloc(&sink, blocks * 256 * 4));
    double issue_ms = timed([&] { issue_ceiling<<<blocks, 256>>>(sink, 1024); }, 50);
    std::printf("{\"register_issue_only_bf16_tflops\":%.6g,\"not_gemm_or_training_peak\":true}\n",
                (double)blocks * 8 * 1024 * 8 * 8192 / (issue_ms * 1e9));
    std::vector<float> ha((size_t)M * K), hb((size_t)N * K), hy((size_t)M * N);
    for (size_t i = 0; i < ha.size(); i++)
        ha[i] = value((unsigned)i);
    for (size_t i = 0; i < hb.size(); i++)
        hb[i] = value((unsigned)i + 170011);
    HIP_OK(hipMemcpy(a, ha.data(), ha.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(b, hb.data(), hb.size() * 4, hipMemcpyHostToDevice));
    const char *names[] = {
        "legacy_bf16x6",          "tiled_64x64_bf16x6",     "tiled_64x64_bf16",
        "tiled_32x64_bf16x6",     "tiled_64x32_bf16x6",     "tiled_32x32_bf16x6",
        "tiled_64x128_bf16x6",    "hipblaslt_bf16x6",       "hipblaslt_bf16",
        "tiled_32x32_k64_bf16x6", "tiled_64x64_k64_bf16x6", "tiled_32x64_k64_bf16x6",
        "tiled_32x32_bf16x3"};
    int modes = 13, failed = 0;
    if (argc == 9) {
        bool found = false;
        for (const char *name : names)
            found |= !std::strcmp(name, argv[8]);
        if (!found)
            return 2;
    }
#ifdef GN_HIPBLASLT
    void *lt = gn_lt_open(1), *workspace;
    if (!lt)
        return 1;
    size_t workspace_bytes = 64u * 1024 * 1024 + (((size_t)M * N * 4 + 255) & ~(size_t)255);
    HIP_OK(hipMalloc(&workspace, workspace_bytes));
#endif
    for (int mode = 0; mode < modes; mode++) {
        if (argc == 9 && std::strcmp(names[mode], argv[8]))
            continue;
#ifndef GN_HIPBLASLT
        if (mode == 7 || mode == 8) {
            if (argc == 9)
                return 77;
            continue;
        }
#endif
        int precise = mode == 12 ? 2 : mode != 2 && mode != 8, add = 0;
        int products = mode == 12 ? 3 : precise ? 6 : 1;
        auto pack = [&] {
            if (!mode)
                return;
            gn_pack_bf16<<<dim3((K + 31) / 32, (M + 31) / 32), 256>>>(pa, a, M, K, ta, precise);
            gn_pack_bf16<<<dim3((K + 31) / 32, (N + 31) / 32), 256>>>(pb, b, N, K, !tb, precise);
        };
        auto kernel = [&] {
            switch (mode) {
            case 12:
                gn_mm_bf16x3<<<dim3((N + 31) / 32, (M + 31) / 32), 128>>>(y, pa, pb, M, N, K, add);
                break;
            case 0:
                gn_mm<<<dim3((N + 15) / 16, (M + 15) / 16), 32>>>(y, a, b, M, N, K, ta, tb, add, 1);
                break;
            case 1:
                tile_variant<2, 2>
                    <<<dim3((N + 63) / 64, (M + 63) / 64), 128>>>(y, pa, pb, M, N, K, add);
                break;
            case 2:
                gn_mm_tiled_fast<<<dim3((N + 63) / 64, (M + 63) / 64), 128>>>(y, pa, pb, M, N, K,
                                                                              add);
                break;
            case 3:
                tile_variant<1, 2>
                    <<<dim3((N + 63) / 64, (M + 31) / 32), 128>>>(y, pa, pb, M, N, K, add);
                break;
            case 4:
                tile_variant<2, 1>
                    <<<dim3((N + 31) / 32, (M + 63) / 64), 128>>>(y, pa, pb, M, N, K, add);
                break;
            case 5:
                gn_mm_tiled<<<dim3((N + 31) / 32, (M + 31) / 32), 128>>>(y, pa, pb, M, N, K, add);
                break;
            case 6:
                tile_variant<2, 4>
                    <<<dim3((N + 127) / 128, (M + 63) / 64), 128>>>(y, pa, pb, M, N, K, add);
                break;
            case 9:
                tile_variant<1, 1, 64>
                    <<<dim3((N + 31) / 32, (M + 31) / 32), 128>>>(y, pa, pb, M, N, K, add);
                break;
            case 10:
                tile_variant<2, 2, 64>
                    <<<dim3((N + 63) / 64, (M + 63) / 64), 128>>>(y, pa, pb, M, N, K, add);
                break;
            case 11:
                tile_variant<1, 2, 64>
                    <<<dim3((N + 63) / 64, (M + 31) / 32), 128>>>(y, pa, pb, M, N, K, add);
                break;
#ifdef GN_HIPBLASLT
            case 7: {
                const int ai[] = {0, 1, 0, 1, 2, 0}, bi[] = {0, 0, 1, 1, 0, 2};
                for (int q = 0; q < 6; q++)
                    if (gn_lt_run(lt, q ? low : high, pa + (size_t)ai[q] * M * stride,
                                  pb + (size_t)bi[q] * N * stride, M, N, K, q > 1 ? 1 : 0,
                                  workspace, workspace_bytes))
                        std::exit(1);
                gn_lt_combine<<<(M * N + 255) / 256, 256>>>(y, high, low, M * N, add);
                break;
            }
            case 8:
                if (gn_lt_run(lt, y, pa, pb, M, N, K, add ? 1 : 0, workspace, workspace_bytes))
                    std::exit(1);
                break;
#endif
            }
        };
        pack();
        double kernel_ms = timed(kernel, iterations);
        double total_ms = timed(
            [&] {
                pack();
                kernel();
            },
            iterations);
        HIP_OK(hipMemcpy(hy.data(), y, hy.size() * 4, hipMemcpyDeviceToHost));
        double delta = 0, base = 0;
        for (float v : hy)
            if (!std::isfinite(v))
                failed = 1;
        for (int i = 0; i < 256; i++) {
            int r = (unsigned)(i * 739 + 13) % M, c = (unsigned)(i * 491 + 5) % N;
            double ref = 0;
            for (int k = 0; k < K; k++)
                ref += (double)ha[ta ? k * M + r : r * K + k] * hb[tb ? c * K + k : k * N + c];
            double d = hy[r * N + c] - ref;
            delta += d * d;
            base += ref * ref;
        }
        double relative = std::sqrt(delta / std::fmax(base, 1e-30));
        if (!std::isfinite(relative) || relative > (precise ? 2e-5 : .02))
            failed = 1;
        int pm = mode == 0                                                         ? 16
                 : mode == 3 || mode == 5 || mode == 9 || mode == 11 || mode == 12 ? 32
                                                                                   : 64;
        int pn = mode == 0                                           ? 16
                 : mode == 4 || mode == 5 || mode == 9 || mode == 12 ? 32
                 : mode == 6                                         ? 128
                                                                     : 64;
        int pk = mode == 0 ? 16 : mode >= 9 && mode <= 11 ? 64 : 32;
        double executed = 2.0 * ((M + pm - 1) / pm * pm) * ((N + pn - 1) / pn * pn) *
                          ((K + pk - 1) / pk * pk) * products;
        std::printf("{\"mode\":\"%s\",\"kernel_ms\":%.6g,\"pack_plus_kernel_ms\":%.6g,\"useful_"
                    "tflops\":%.6g,\"matrix_products_tflops\":%.6g,\"sample_relative_l2\":%.6g",
                    names[mode], kernel_ms, total_ms, 2.0 * M * N * K / (total_ms * 1e9),
                    2.0 * M * N * K * products / (kernel_ms * 1e9), relative);
        /* Vendor instruction padding is not known; never fabricate its count. */
        if (mode < 7 || mode >= 9)
            std::printf(",\"executed_wmma_tflops\":%.6g", executed / (kernel_ms * 1e9));
        double fraction = 2.0 * M * N * K * products / (kernel_ms * 1e9 * peak);
        std::printf(",\"kernel_product_75pct_rate_met\":%s", fraction >= .75 ? "true" : "false");
        std::printf(",\"matrix_product_peak_fraction\":%.6g,\"kernel_product_95pct_target_met\":%s",
                    fraction,
                    fraction >= .95 && relative <= (precise ? 2e-5 : .02) ? "true" : "false");
        std::printf("}\n");
        add = 1;
        kernel();
        kernel();
        std::vector<float> accumulated(hy.size());
        HIP_OK(hipMemcpy(accumulated.data(), y, hy.size() * 4, hipMemcpyDeviceToHost));
        size_t add_failures = 0;
        double max_add_error = 0;
        for (size_t i = 0; i < hy.size(); i++)
            if (!std::isfinite(accumulated[i]) ||
                std::fabs(accumulated[i] - 3 * hy[i]) > 1e-5f + 1e-6f * std::fabs(hy[i])) {
                failed = 1;
                add_failures++;
                max_add_error = std::fmax(max_add_error, std::fabs(accumulated[i] - 3 * hy[i]));
            }
        if (add_failures)
            std::fprintf(stderr, "%s: add mismatch count=%zu max_abs=%g\n", names[mode],
                         add_failures, max_add_error);
    }
#ifdef GN_HIPBLASLT
    gn_lt_close(lt);
    HIP_OK(hipFree(workspace));
#endif
    HIP_OK(hipFree(a));
    HIP_OK(hipFree(b));
    HIP_OK(hipFree(y));
    HIP_OK(hipFree(high));
    HIP_OK(hipFree(low));
    HIP_OK(hipFree(pa));
    HIP_OK(hipFree(pb));
    HIP_OK(hipFree(sink));
    std::printf("{\"matrix_diagnostic_pass\":%s}\n", failed ? "false" : "true");
    return failed;
}
