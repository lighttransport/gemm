/* SPDX-License-Identifier: MIT
 * Reduced-precision diagnostics. Timing is not a training qualification.
 * Exact integer tests inspect INT64 output BEFORE conversion/dequantization.
 */
#define GN_HIP 1
#include "bench_hip_common.h"
#include "gn_kernels.cu"
template <int MR, int NR>
__global__ __launch_bounds__(128) void integer_variant(float *y, const signed char *a,
                                                       const signed char *b, const float *as,
                                                       const float *bs, int M, int N, int K,
                                                       int add) {
    gn_rdna4_integer<16, true, MR, NR>(y, a, b, as, bs, M, N, K, add, nullptr);
}

static int bf16_exact() {
    /* Exactly representable +/-1/8 products keep all partial sums inside
     * BF16's exact range. Exercise lane layout, tails, chunk boundaries and
     * residual-add semantics independently of expected roundoff experiments. */
    constexpr int M = 35, N = 67;
    int failed = 0;
    for (int K : {1, 16, 49, 128, 256}) {
        int stride = (K + 31) & ~31;
        std::vector<unsigned short> a(M * stride), b(N * stride);
        for (int r = 0; r < M; r++)
            for (int k = 0; k < K; k++)
                a[r * stride + k] = ((r * 173 + k * 13) % 17 < 8) ? 0x3f00 : 0xbf00;
        for (int c = 0; c < N; c++)
            for (int k = 0; k < K; k++)
                b[c * stride + k] = ((c * 137 + k * 7) % 19 < 9) ? 0x3e80 : 0xbe80;
        unsigned short *da, *db;
        float *y;
        HIP_OK(hipMalloc(&da, a.size() * 2));
        HIP_OK(hipMalloc(&db, b.size() * 2));
        HIP_OK(hipMalloc(&y, M * N * 4));
        HIP_OK(hipMemcpy(da, a.data(), a.size() * 2, hipMemcpyHostToDevice));
        HIP_OK(hipMemcpy(db, b.data(), b.size() * 2, hipMemcpyHostToDevice));
        for (int chunk : {0, 128}) {
            std::vector<float> actual(M * N);
            for (int add : {0, 1}) {
                gn_mm_bf16_acc<<<dim3((N + 63) / 64, (M + 63) / 64), 128>>>(y, da, db, M, N, K, add,
                                                                            chunk);
                HIP_OK(hipMemcpy(actual.data(), y, M * N * 4, hipMemcpyDeviceToHost));
                for (int r = 0; r < M; r++)
                    for (int c = 0; c < N; c++) {
                        float ref = 0;
                        for (int k = 0; k < K; k++)
                            ref +=
                                ((a[r * stride + k] ^ b[c * stride + k]) & 0x8000) ? -.125f : .125f;
                        if (actual[r * N + c] != ref * (1 + add))
                            failed = 1;
                    }
            }
        }
        HIP_OK(hipFree(da));
        HIP_OK(hipFree(db));
        HIP_OK(hipFree(y));
    }
    std::printf("{\"bf16_acc_representable_exact\":%s}\n", failed ? "false" : "true");
    return failed;
}

static int columns_exact() {
    int failed = 0;
    for (int side : {3, 9})
        for (int C : {3, 8, 32, 80, 256})
            for (int kernel : {3, 5})
                for (int precise : {0, 1, 2}) {
                    int R = 3 * side * side, K = C * kernel * kernel, stride = (K + 31) & ~31,
                        back_stride = (R + 31) & ~31,
                        planes = precise == 2 ? 2
                                 : precise    ? 3
                                              : 1;
                    size_t capacity = (size_t)std::max(R * stride, K * back_stride) * planes;
                    float *x, *col;
                    unsigned short *a, *b;
                    std::vector<float> input(R * C);
                    for (size_t i = 0; i < input.size(); i++)
                        input[i] = value((unsigned)i);
                    HIP_OK(hipMalloc(&x, input.size() * 4));
                    HIP_OK(hipMalloc(&col, R * K * 4));
                    HIP_OK(hipMalloc(&a, capacity * 2));
                    HIP_OK(hipMalloc(&b, capacity * 2));
                    HIP_OK(hipMemcpy(x, input.data(), input.size() * 4, hipMemcpyHostToDevice));
                    gn_columns<<<(R * K + 255) / 256, 256>>>(col, x, R, C, side, kernel);
                    gn_pack_bf16<<<dim3((K + 31) / 32, (R + 31) / 32), 256>>>(a, col, R, K, 0,
                                                                              precise);
                    gn_columns_bf16<<<(R * stride + 255) / 256, 256>>>(b, x, R, C, side, kernel,
                                                                       precise);
                    std::vector<unsigned short> ref(R * stride * planes), actual(ref.size());
                    HIP_OK(hipMemcpy(ref.data(), a, ref.size() * 2, hipMemcpyDeviceToHost));
                    HIP_OK(hipMemcpy(actual.data(), b, actual.size() * 2, hipMemcpyDeviceToHost));
                    if (ref != actual)
                        failed = 1;
                    gn_pack_bf16<<<dim3((R + 31) / 32, (K + 31) / 32), 256>>>(a, col, K, R, 1,
                                                                              precise);
                    gn_columns_bf16_back<<<dim3((K + 31) / 32, (R + 31) / 32), 256>>>(
                        b, x, R, C, side, kernel, precise);
                    ref.resize(K * back_stride * planes);
                    actual.resize(ref.size());
                    HIP_OK(hipMemcpy(ref.data(), a, ref.size() * 2, hipMemcpyDeviceToHost));
                    HIP_OK(hipMemcpy(actual.data(), b, actual.size() * 2, hipMemcpyDeviceToHost));
                    if (ref != actual)
                        failed = 1;
                    HIP_OK(hipFree(x));
                    HIP_OK(hipFree(col));
                    HIP_OK(hipFree(a));
                    HIP_OK(hipFree(b));
                }
    std::printf("{\"fused_columns_forward_backward_bit_exact\":%s}\n", failed ? "false" : "true");
    return failed;
}

static int integer_exact(int bits, int wide, int K, int pattern) {
    constexpr int M = 35, N = 67;
    int stride = (K + 31) & ~31, tile = bits == 16 ? 32 : 64, planes = bits / 8;
    std::vector<signed char> a((size_t)M * stride * planes), b((size_t)N * stride * planes);
    std::vector<int> ha((size_t)M * K), hb((size_t)N * K);
    auto input = [&](std::vector<signed char> &out, std::vector<int> &host, int R, int salt) {
        int lower = bits == 16 ? -32768 : -128, upper = bits == 16 ? 32767 : 127;
        for (int r = 0; r < R; r++)
            for (int k = 0; k < K; k++) {
                int q =
                    pattern == 0 ? lower
                    : pattern == 1
                        ? upper
                        : (int)((unsigned)(k * 173 + r * 313 + salt) % (upper - lower + 1)) + lower;
                host[r * K + k] = q;
                out[r * stride + k] = (signed char)(bits == 8 ? q : q >> 8);
                if (bits == 16)
                    out[(R + r) * stride + k] = (signed char)(q & 255);
            }
    };
    input(a, ha, M, 31);
    input(b, hb, N, 109);
    signed char *da, *db;
    float *y, *sa, *sb;
    long long *exact;
    std::vector<float> ones(M + N, 1);
    HIP_OK(hipMalloc(&da, a.size()));
    HIP_OK(hipMalloc(&db, b.size()));
    HIP_OK(hipMalloc(&y, M * N * 4));
    HIP_OK(hipMalloc(&exact, M * N * 8));
    HIP_OK(hipMalloc(&sa, M * 4));
    HIP_OK(hipMalloc(&sb, N * 4));
    HIP_OK(hipMemcpy(da, a.data(), a.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(db, b.data(), b.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(sa, ones.data(), M * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(sb, ones.data(), N * 4, hipMemcpyHostToDevice));
    gn_mm_integer<<<dim3((N + tile - 1) / tile, (M + 63) / 64), 128>>>(y, da, db, sa, sb, M, N, K,
                                                                       0, bits, wide, exact);
    std::vector<long long> actual(M * N);
    HIP_OK(hipMemcpy(actual.data(), exact, M * N * 8, hipMemcpyDeviceToHost));
    int failed = 0;
    /* All outputs for bounded K; samples for overflow-length tests. */
    int checks = K < 20000 ? M * N : 256;
    for (int i = 0; i < checks; i++) {
        int index = K < 20000 ? i : (i * 739 + 13) % (M * N), r = index / N, c = index % N;
        long long ref = 0;
        for (int k = 0; k < K; k++)
            ref += (long long)ha[r * K + k] * hb[c * K + k];
        if (actual[index] != ref) {
            if (!failed)
                std::fprintf(stderr, "integer exact mismatch bits=%d K=%d got=%lld want=%lld\n",
                             bits, K, actual[index], ref);
            failed = 1;
        }
    }
    std::printf(
        "{\"integer_exact\":%s,\"bits\":%d,\"wide\":%d,\"K\":%d,\"pattern\":%d,\"checked\":%d}\n",
        failed ? "false" : "true", bits, wide, K, pattern, checks);
    HIP_OK(hipFree(da));
    HIP_OK(hipFree(db));
    HIP_OK(hipFree(y));
    HIP_OK(hipFree(exact));
    HIP_OK(hipFree(sa));
    HIP_OK(hipFree(sb));
    return failed;
}
int main(int argc, char **argv) {
    if (argc != 1 && argc != 7 && argc != 9 && !(argc == 2 && !std::strcmp(argv[1], "check"))) {
        std::fprintf(stderr, "usage: bench_rdna4_precision [M N K ta tb iterations "
                             "[bf16_peak_tflops int8_peak_tops]] | check\n");
        return 2;
    }
    hipDeviceProp_t prop;
    HIP_OK(hipGetDeviceProperties(&prop, 0));
    if (std::strncmp(prop.gcnArchName, "gfx1201", 7))
        return 77;
    if (argc == 2) {
        int failed = bf16_exact() | columns_exact();
        for (int bits : {8, 16})
            for (int wide : {0, 1})
                for (int pattern = 0; pattern < 3; pattern++) {
                    failed |= integer_exact(bits, wide, 49, pattern);
                    failed |= integer_exact(bits, wide, 16385, pattern);
                    if (bits == 8)
                        failed |= integer_exact(bits, wide, 131073, pattern);
                }
        return failed;
    }
    int M = argc >= 7 ? std::atoi(argv[1]) : 1296, N = argc >= 7 ? std::atoi(argv[2]) : 256;
    int K = argc >= 7 ? std::atoi(argv[3]) : 2304, ta = argc >= 7 ? std::atoi(argv[4]) : 0;
    int tb = argc >= 7 ? std::atoi(argv[5]) : 1, iterations = argc >= 7 ? std::atoi(argv[6]) : 100;
    double bf_peak = 195, int_peak = 389;
    if (argc == 9) {
        char *end;
        bf_peak = std::strtod(argv[7], &end);
        if (*end)
            return 2;
        int_peak = std::strtod(argv[8], &end);
        if (*end)
            return 2;
    }
    if (M < 1 || N < 1 || K < 1 || M > 8192 || N > 8192 || K > 8192 || ta < 0 || ta > 1 || tb < 0 ||
        tb > 1 || iterations < 1 || iterations > 1000 || !std::isfinite(bf_peak) ||
        !std::isfinite(int_peak) || bf_peak <= 0 || int_peak <= 0)
        return 2;
    std::printf(
        "{\"device\":\"%s\",\"arch\":\"%s\",\"M\":%d,\"N\":%d,\"K\":%d,\"ta\":%d,\"tb\":%d,"
        "\"iterations\":%d,\"bf16_dense_peak_tflops\":%.6g,\"int8_dense_peak_tops\":%.6g}\n",
        prop.name, prop.gcnArchName, M, N, K, ta, tb, iterations, bf_peak, int_peak);
    int stride = (K + 31) & ~31, failed = 0;
    float *a, *b, *y, *sa, *sb, *transpose;
    unsigned short *pa, *pb;
    long long *exact;
    HIP_OK(hipMalloc(&a, (size_t)M * K * 4));
    HIP_OK(hipMalloc(&b, (size_t)N * K * 4));
    HIP_OK(hipMalloc(&y, (size_t)M * N * 4));
    HIP_OK(hipMalloc(&exact, (size_t)M * N * 8));
    HIP_OK(hipMalloc(&pa, (size_t)M * stride * 2));
    HIP_OK(hipMalloc(&pb, (size_t)N * stride * 2));
    HIP_OK(hipMalloc(&sa, (size_t)M * 4));
    HIP_OK(hipMalloc(&sb, (size_t)N * 4));
    HIP_OK(hipMalloc(&transpose, (size_t)(M > N ? M : N) * K * 4));
    std::vector<float> ha((size_t)M * K), hb((size_t)N * K), hy((size_t)M * N);
    for (size_t i = 0; i < ha.size(); i++)
        ha[i] = value((unsigned)i);
    for (size_t i = 0; i < hb.size(); i++)
        hb[i] = value((unsigned)i + 170011);
    HIP_OK(hipMemcpy(a, ha.data(), ha.size() * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(b, hb.data(), hb.size() * 4, hipMemcpyHostToDevice));
    const char *names[] = {"bf16_fp32acc",    "bf16_bf16acc",         "bf16_bf16acc128_fp32",
                           "int8_i32acc",     "int8_i32partials_i64", "int16_tile32x32",
                           "int16_tile32x64", "int16_tile64x32",      "int16_tile64x64"};
    for (int mode = 0; mode < 9; mode++) {
        int bits = mode >= 5 ? 16 : 8, tile = mode == 5 ? 32 : 64, add = 0;
        int tm = mode == 6 ? 32 : tile, tn = mode == 7 ? 32 : tile;
        auto pack = [&] {
            if (mode < 3) {
                gn_pack_bf16<<<dim3((K + 31) / 32, (M + 31) / 32), 256>>>(pa, a, M, K, ta, 0);
                gn_pack_bf16<<<dim3((K + 31) / 32, (N + 31) / 32), 256>>>(pb, b, N, K, !tb, 0);
            } else {
                auto rowpack = [&](signed char *out, float *scale, const float *in, int R,
                                   int trans) {
                    if (trans && R >= 32 && K >= 32) {
                        gn_transpose_rows<<<dim3((K + 31) / 32, (R + 31) / 32), 256>>>(transpose,
                                                                                       in, R, K);
                        in = transpose;
                        trans = 0;
                    }
                    gn_pack_integer<<<R, 256>>>(out, scale, in, R, K, trans, bits);
                };
                rowpack((signed char *)pa, sa, a, M, ta);
                rowpack((signed char *)pb, sb, b, N, !tb);
            }
        };
        auto kernel = [&] {
            dim3 grid((N + tn - 1) / tn, (M + tm - 1) / tm);
            if (mode == 0)
                gn_mm_tiled_fast<<<grid, 128>>>(y, pa, pb, M, N, K, add);
            else if (mode < 3)
                gn_mm_bf16_acc<<<grid, 128>>>(y, pa, pb, M, N, K, add, mode == 2 ? 128 : 0);
            else if (mode == 5)
                integer_variant<1, 1>
                    <<<grid, 128>>>(y, (signed char *)pa, (signed char *)pb, sa, sb, M, N, K, add);
            else if (mode == 6)
                integer_variant<1, 2>
                    <<<grid, 128>>>(y, (signed char *)pa, (signed char *)pb, sa, sb, M, N, K, add);
            else if (mode == 7)
                integer_variant<2, 1>
                    <<<grid, 128>>>(y, (signed char *)pa, (signed char *)pb, sa, sb, M, N, K, add);
            else if (mode == 8)
                integer_variant<2, 2>
                    <<<grid, 128>>>(y, (signed char *)pa, (signed char *)pb, sa, sb, M, N, K, add);
            else
                gn_mm_integer<<<grid, 128>>>(y, (signed char *)pa, (signed char *)pb, sa, sb, M, N,
                                             K, add, bits, mode == 4, nullptr);
        };
        pack();
        double ms = timed(kernel, iterations);
        double total = timed(
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
            int r = (i * 739 + 13) % M, c = (i * 491 + 5) % N;
            double ref = 0;
            for (int k = 0; k < K; k++)
                ref += (double)ha[ta ? k * M + r : r * K + k] * hb[tb ? c * K + k : k * N + c];
            double d = hy[r * N + c] - ref;
            delta += d * d;
            base += ref * ref;
        }
        double rel = std::sqrt(delta / std::fmax(base, 1e-30));
        /* Original-FP32 approximation quality is reported, never called an
         * exact-accumulator pass or used to weaken the normal training gate. */
        if (!std::isfinite(rel) || (mode == 0 && rel > .02))
            failed = 1;
        if (mode >= 3) {
            std::vector<signed char> qa((size_t)M * stride * bits / 8),
                qb((size_t)N * stride * bits / 8);
            std::vector<float> as(M), bs(N);
            std::vector<long long> out((size_t)M * N);
            HIP_OK(hipMemcpy(qa.data(), pa, qa.size(), hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(qb.data(), pb, qb.size(), hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(as.data(), sa, M * 4, hipMemcpyDeviceToHost));
            HIP_OK(hipMemcpy(bs.data(), sb, N * 4, hipMemcpyDeviceToHost));
            auto qvalue = [&](const std::vector<signed char> &q, int R, int r, int k) {
                int h = q[r * stride + k];
                return bits == 8 ? h : 256 * h + (unsigned char)q[(R + r) * stride + k];
            };
            /* Check every quantized input, including transpose and padding. */
            for (int operand = 0; operand < 2; operand++) {
                int R = operand ? N : M, trans = operand ? !tb : ta,
                    bound = bits == 8 ? 127 : 32767;
                auto &q = operand ? qb : qa;
                auto &src = operand ? hb : ha;
                auto &scale = operand ? bs : as;
                for (int r = 0; r < R; r++)
                    for (int k = 0; k < stride; k++) {
                        float x = k < K ? src[trans ? k * R + r : r * K + k] : 0;
                        int expected = (int)std::nearbyint(x / scale[r]);
                        expected = expected < -bound ? -bound : expected > bound ? bound : expected;
                        if (qvalue(q, R, r, k) != expected)
                            failed = 1;
                    }
            }
            int exact_tile = bits == 16 ? 32 : 64;
            gn_mm_integer<<<dim3((N + exact_tile - 1) / exact_tile, (M + 63) / 64), 128>>>(
                y, (signed char *)pa, (signed char *)pb, sa, sb, M, N, K, 0, bits, mode == 4,
                exact);
            HIP_OK(hipMemcpy(out.data(), exact, out.size() * 8, hipMemcpyDeviceToHost));
            for (int i = 0; i < 256; i++) {
                int r = (i * 739 + 13) % M, c = (i * 491 + 5) % N;
                long long ref = 0;
                for (int k = 0; k < K; k++)
                    ref += (long long)qvalue(qa, M, r, k) * qvalue(qb, N, c, k);
                if (ref != out[r * N + c])
                    failed = 1;
                float dequant = (float)ref * as[r] * bs[c];
                if (std::fabs(dequant - hy[r * N + c]) > 1e-6f * std::fabs(dequant) + 1e-5f)
                    failed = 1;
            }
        }
        double products = 2.0 * M * N * K * (mode >= 5 ? 4 : 1),
               peak = mode < 3 ? bf_peak : int_peak;
        double issued = 2.0 * ((M + tm - 1) / tm * tm) * ((N + tn - 1) / tn * tn) * stride *
                        (mode >= 5 ? 4 : 1);
        std::printf(
            "{\"mode\":\"%s\",\"unit\":\"%s\",\"kernel_ms\":%.6g,\"pack_plus_kernel_ms\":%.6g,"
            "\"useful_tops_including_pack\":%.6g,\"matrix_product_tops\":%.6g,\"issued_wmma_tops\":"
            "%.6g,"
            "\"matrix_product_peak_pct\":%.6g,\"issued_wmma_peak_pct\":%.6g,\"sample_relative_l2\":"
            "%.6g,"
            "\"training_qualified\":false,\"kernel_product_95pct_reached\":%s}\n",
            names[mode], mode < 3 ? "TFLOP/s" : "TIOP/s (INT8 instruction products)", ms, total,
            2.0 * M * N * K / (total * 1e9), products / (ms * 1e9), issued / (ms * 1e9),
            products / (ms * 1e9 * peak) * 100, issued / (ms * 1e9 * peak) * 100, rel,
            products / (ms * 1e9 * peak) >= .95 ? "true" : "false");
        add = 1;
        kernel();
        kernel();
        std::vector<float> summed(hy.size());
        HIP_OK(hipMemcpy(summed.data(), y, hy.size() * 4, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < hy.size(); i++)
            if (!std::isfinite(summed[i]) ||
                std::fabs(summed[i] - 3 * hy[i]) > 1e-5f + 1e-6f * std::fabs(hy[i]))
                failed = 1;
    }
    HIP_OK(hipFree(a));
    HIP_OK(hipFree(b));
    HIP_OK(hipFree(y));
    HIP_OK(hipFree(pa));
    HIP_OK(hipFree(pb));
    HIP_OK(hipFree(sa));
    HIP_OK(hipFree(sb));
    HIP_OK(hipFree(exact));
    HIP_OK(hipFree(transpose));
    if (failed)
        std::fprintf(stderr, "precision diagnostic correctness failure\n");
    return failed;
}
