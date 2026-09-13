/* SPDX-License-Identifier: MIT
 * Standalone hardware diagnostic, not a production CUDA SDK dependency.
 * Counts useful 2MNK separately from six-product executed BF16 operations.
 */
#include "gn_kernels.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CUDA_OK(expr)                                                                              \
    do {                                                                                           \
        cudaError_t e = (expr);                                                                    \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(e));                             \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

/* Register-resident issue ceiling ONLY. This is explicitly not a GEMM. */
template <bool Integer> __global__ void issue_ceiling(float *out, int iterations) {
    unsigned a[4] = {0x3b003b00, 0x3b003b00, 0x3b003b00, 0x3b003b00};
    unsigned b[2] = {0x3b003b00, 0x3b003b00};
    float d[8][4] = {};
    int id[8][4] = {};
    for (int i = 0; i < iterations; i++) {
#pragma unroll
        for (int q = 0; q < 8; q++) {
            if (Integer)
                gn_imma(id[q], a, b);
            else
                mma(d[q][0], d[q][1], d[q][2], d[q][3], a, b);
        }
    }
    float sum = 0;
#pragma unroll
    for (int q = 0; q < 8; q++)
#pragma unroll
        for (int j = 0; j < 4; j++)
            sum += Integer ? (float)id[q][j] : d[q][j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
template <typename F> static double timed(F fn, int n) {
    for (int i = 0; i < 10; i++)
        fn();
    CUDA_OK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start));
    for (int i = 0; i < n; i++)
        fn();
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    return ms / n;
}
static float value(unsigned i) {
    unsigned x = i * 747796405U + 2891336453U;
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737U;
    x = (x >> 22) ^ x;
    return (float)(x >> 8) * (2.0f / 16777216) - 1;
}
static int check_fused_columns() {
    int failed = 0;
    for (int channels : {1, 3, 5, 80, 256}) {
        int side = channels < 8 ? 3 : 9, R = 2 * side * side;
        int kernel = channels == 80 ? 5 : 3, K = channels * kernel * kernel;
        int stride = (K + 31) & ~31;
        float *x, *col;
        unsigned short *fused, *reference;
        CUDA_OK(cudaMalloc(&x, (size_t)R * channels * 4));
        CUDA_OK(cudaMalloc(&col, (size_t)R * K * 4));
        size_t count = std::max((size_t)R * stride, (size_t)K * ((R + 31) & ~31)) * 3;
        CUDA_OK(cudaMalloc(&fused, count * 2));
        CUDA_OK(cudaMalloc(&reference, count * 2));
        std::vector<float> input((size_t)R * channels);
        for (size_t i = 0; i < input.size(); i++)
            input[i] = value((unsigned)i);
        CUDA_OK(cudaMemcpy(x, input.data(), input.size() * 4, cudaMemcpyHostToDevice));
        gn_columns<<<(R * K + 255) / 256, 256>>>(col, x, R, channels, side, kernel);
        for (int trans : {0, 1}) {
            for (int precise : {0, 1, 2}) {
                int planes = precise == 2 ? 2 : precise ? 3 : 1;
                int rows = trans ? K : R, cols = trans ? R : K;
                gn_pack_bf16<<<dim3((cols + 31) / 32, (rows + 31) / 32), 256>>>(
                    reference, col, rows, cols, trans, precise);
                CUDA_OK(cudaMemset(fused, 0xcd, count * 2));
                if (trans)
                    gn_columns_bf16_back<<<dim3((K + 31) / 32, (R + 31) / 32), 256>>>(
                        fused, x, R, channels, side, kernel, precise);
                else
                    gn_columns_bf16<<<(R * stride / 4 + 255) / 256, 256>>>(
                        fused, x, R, channels, side, kernel, precise);
                std::vector<unsigned short> a((size_t)rows * ((cols + 31) & ~31) * planes), b(a.size());
                CUDA_OK(cudaMemcpy(a.data(), fused, a.size() * 2, cudaMemcpyDeviceToHost));
                CUDA_OK(cudaMemcpy(b.data(), reference, b.size() * 2, cudaMemcpyDeviceToHost));
                if (a != b) {
                    fprintf(stderr, "fused columns mismatch C=%d precise=%d trans=%d\n",
                            channels, precise, trans);
                    failed = 1;
                }
            }
        }
        CUDA_OK(cudaFree(x));
        CUDA_OK(cudaFree(col));
        CUDA_OK(cudaFree(fused));
        CUDA_OK(cudaFree(reference));
    }
    printf("{\"fused_columns_bitwise_pass\":%s,\"channel_shapes\":5,\"precision_modes\":3,"
           "\"transpose_modes\":2}\n",
           failed ? "false" : "true");
    return failed;
}
int main(int argc, char **argv) {
    if (argc != 1 && argc != 7) {
        fprintf(stderr, "usage: bench_sm120 [M N K transpose_A transpose_B iterations]\n");
        return 2;
    }
    int M = argc == 7 ? atoi(argv[1]) : 1296, N = argc == 7 ? atoi(argv[2]) : 256;
    int K = argc == 7 ? atoi(argv[3]) : 2304, ta = argc == 7 ? atoi(argv[4]) : 0;
    int tb = argc == 7 ? atoi(argv[5]) : 1, iters = argc == 7 ? atoi(argv[6]) : 20;
    if (M < 1 || N < 1 || K < 1 || M > 8192 || N > 8192 || K > 8192 || ta < 0 || ta > 1 || tb < 0 ||
        tb > 1 || iters < 1 || iters > 1000)
        return 2;
    cudaDeviceProp prop;
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 12 || prop.minor != 0)
        return 77;
    printf("{\"device\":\"%s\",\"sms\":%d,\"M\":%d,\"N\":%d,\"K\":%d,\"ta\":%d,\"tb\":%d}\n",
           prop.name, prop.multiProcessorCount, M, N, K, ta, tb);
    float *a, *b, *y, *scales_a, *scales_b, *sink;
    unsigned short *pa, *pb;
    int stride = (K + 31) & ~31;
    CUDA_OK(cudaMalloc(&a, (size_t)M * K * 4));
    CUDA_OK(cudaMalloc(&b, (size_t)N * K * 4));
    CUDA_OK(cudaMalloc(&y, (size_t)M * N * 4));
    CUDA_OK(cudaMalloc(&pa, (size_t)M * stride * 6));
    CUDA_OK(cudaMalloc(&pb, (size_t)N * stride * 6));
    CUDA_OK(cudaMalloc(&scales_a, M * 4));
    CUDA_OK(cudaMalloc(&scales_b, N * 4));
    int blocks = prop.multiProcessorCount * 4;
    CUDA_OK(cudaMalloc(&sink, blocks * 256 * 4));
    for (int integer = 0; integer < 2; integer++) {
        double samples[3];
        for (double &ms : samples)
            ms = timed(
                [&] {
                    if (integer)
                        issue_ceiling<true><<<blocks, 256>>>(sink, 1024);
                    else
                        issue_ceiling<false><<<blocks, 256>>>(sink, 1024);
                },
                20);
        std::sort(samples, samples + 3);
        double rate = (double)blocks * 8 * 1024 * 8 * (integer ? 8192 : 4096) / (samples[1] * 1e9);
        printf("{\"register_issue_only\":\"%s\",\"tera_ops_s\":%.6g,\"not_gemm_or_training_peak\":"
               "true}\n",
               integer ? "int8" : "bf16", rate);
    }
    std::vector<float> ha((size_t)M * K), hb((size_t)N * K), hy((size_t)M * N);
    for (size_t i = 0; i < ha.size(); i++)
        ha[i] = value((unsigned)i);
    for (size_t i = 0; i < hb.size(); i++)
        hb[i] = value((unsigned)i + 170011);
    CUDA_OK(cudaMemcpy(a, ha.data(), ha.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
    int failed = check_fused_columns();
    const char *names[] = {"legacy_bf16x6", "tiled_bf16x6", "tiled_bf16", "experimental_int8",
                           "experimental_int16", "tiled_bf16x3", "tiled_bf16x3_n128"};
    for (int mode = 0; mode < 7; mode++) {
        int precise = mode >= 5 ? 2 : mode != 2, bits = mode == 3 ? 8 : 16, add = 0;
        auto pack = [&] {
            if (mode == 0)
                return;
            if (mode < 3 || mode >= 5) {
                gn_pack_bf16<<<dim3((K + 31) / 32, (M + 31) / 32), 256>>>(pa, a, M, K, ta, precise);
                gn_pack_bf16<<<dim3((K + 31) / 32, (N + 31) / 32), 256>>>(pb, b, N, K, !tb,
                                                                          precise);
            } else {
                gn_pack_integer<<<M, 256>>>((signed char *)pa, scales_a, a, M, K, ta, bits);
                gn_pack_integer<<<N, 256>>>((signed char *)pb, scales_b, b, N, K, !tb, bits);
            }
        };
        auto kernel = [&] {
            if (mode == 0)
                gn_mm<<<dim3((N + 7) / 8, (M + 15) / 16), 32>>>(y, a, b, M, N, K, ta, tb, add, 1);
            else if (mode == 1)
                gn_mm_tiled<<<dim3((N + 63) / 64, (M + 31) / 32), 128>>>(y, pa, pb, M, N, K, add);
            else if (mode == 2)
                gn_mm_tiled_fast<<<dim3((N + 63) / 64, (M + 63) / 64), 128>>>(y, pa, pb, M, N, K,
                                                                              add);
            else if (mode == 5)
                gn_mm_bf16x3<<<dim3((N + 63) / 64, (M + 31) / 32), 128>>>(y, pa, pb, M, N, K, add);
            else if (mode == 6)
                gn_mm_bf16x3_n128<<<dim3((N + 127) / 128, (M + 31) / 32), 256>>>(
                    y, pa, pb, M, N, K, add);
            else
                gn_mm_integer<<<dim3((N + 7) / 8, (M + 15) / 16), 32>>>(
                    y, (signed char *)pa, (signed char *)pb, scales_a, scales_b, M, N, K, add,
                    bits);
        };
        pack();
        double kernel_ms = timed(kernel, iters);
        double total_ms = timed(
            [&] {
                pack();
                kernel();
            },
            iters);
        CUDA_OK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
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
        double relative = sqrt(delta / fmax(base, 1e-30));
        double tolerance = mode < 2 || mode >= 5 ? 2e-5 : mode == 4 ? 2e-4 : .02;
        if (!std::isfinite(relative) || relative > tolerance)
            failed = 1;
        double useful = 2.0 * M * N * K / (total_ms * 1e9);
        /* Include tail instructions in executed ops, never in useful ops. */
        double pm = mode >= 5 ? 32 : mode == 0 || mode > 2 ? 16 : mode == 1 ? 32 : 64;
        double pn = mode == 6 ? 128 : mode == 5 ? 64 : mode == 0 || mode > 2 ? 8 : 64;
        double pk = mode == 0 ? 16 : 32;
        double products = mode >= 5 ? 3 : mode < 2 ? 6 : mode == 4 ? 4 : 1;
        double executed = 2 * ceil(M / pm) * pm * ceil(N / pn) * pn * ceil(K / pk) * pk * products /
                          (kernel_ms * 1e9);
        printf("{\"mode\":\"%s\",\"kernel_ms\":%.6g,\"pack_plus_kernel_ms\":%.6g,\"useful_tera_ops_"
               "s\":%.6g,"
               "\"executed_tera_ops_s\":%.6g,\"sample_relative_l2\":%.6g}\n",
               names[mode], kernel_ms, total_ms, useful, executed, relative);
        add = 1;
        kernel();
        kernel();
        std::vector<float> accumulated(hy.size());
        CUDA_OK(cudaMemcpy(accumulated.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < hy.size(); i++)
            if (!std::isfinite(accumulated[i]) ||
                fabsf(accumulated[i] - 3 * hy[i]) > 1e-5f + 1e-6f * fabsf(hy[i]))
                failed = 1;
    }
    CUDA_OK(cudaFree(a));
    CUDA_OK(cudaFree(b));
    CUDA_OK(cudaFree(y));
    CUDA_OK(cudaFree(pa));
    CUDA_OK(cudaFree(pb));
    CUDA_OK(cudaFree(scales_a));
    CUDA_OK(cudaFree(scales_b));
    CUDA_OK(cudaFree(sink));
    return failed;
}
