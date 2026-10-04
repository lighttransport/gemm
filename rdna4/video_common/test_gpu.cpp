// SPDX-License-Identifier: MIT
#include "../../cuda/hunyuan_video15_native/gpu.hpp"
#include "../minimax_h3/kernels.hpp"
#include "test_norm_boundary.hpp"
#include "timestep_basis.hpp"
#include <iostream>
using namespace hv15n;
static Tensor bytes(Gpu &g, size_t n) {
    Tensor t;
    t.shape = {int(n)};
    t.element_bytes = 1;
    t.storage = std::make_shared<Allocation>(&g, n);
    t.pointer = t.storage->pointer;
    return t;
}
int main() {
    try {
        Gpu g(0, 4096, false, false);
        {
            auto half_value = [](float value) { return float((_Float16)value); };
            g.dit_fp16 = true;
            std::vector<float> residual(64), delta(64), modulation(96);
            for (int i = 0; i < 64; ++i) {
                residual[i] = half_value((i - 27) * .03117f);
                delta[i] = half_value((i - 39) * .04329f);
            }
            for (int i = 0; i < 96; ++i)
                modulation[i] = half_value((i - 41) * .01327f);
            auto x = g.round_half(g.upload(residual, {2, 32}));
            auto z = g.round_half(g.upload(delta, {2, 32}));
            auto mod = g.round_half(g.upload(modulation, {1, 96}));
            auto output = g.gated(x, z, mod, 64);
            auto actual = g.download(output);
            require(output.fp16_values, "FP16 residual dtype propagation");
            int separate_rounding = 0;
            for (int i = 0; i < 64; ++i) {
                float expected =
                    half_value(residual[i] + half_value(delta[i] * modulation[64 + i % 32]));
                require(actual[i] == expected, "FP16 multiply/add boundary");
                separate_rounding +=
                    expected != half_value(residual[i] + delta[i] * modulation[64 + i % 32]);
            }
            require(separate_rounding > 0, "adversarial FP16 gate fixture");
            std::vector<float> signs(64);
            for (int i = 0; i < 64; ++i)
                signs[i] = (i & 1) ? 1.f : -1.f;
            auto normalized = g.round_half(g.upload(signs, {2, 32}));
            output = g.modulate(normalized, mod, 0);
            actual = g.download(output);
            require(!output.fp16_values, "autocast modulation must stay FP32");
            for (int i = 0; i < 64; ++i) {
                float norm = signs[i] / std::sqrt(1.f + 1.e-6f);
                float expected =
                    norm * half_value(1.f + modulation[32 + i % 32]) + modulation[i % 32];
                require(std::abs(actual[i] - expected) < 2e-7,
                        "FP32 LayerNorm/modulation with FP16 scale-plus-one");
            }
            auto fp32_residual = g.upload(residual, {2, 32});
            output = g.gated(fp32_residual, z, mod, 64);
            actual = g.download(output);
            require(!output.fp16_values, "mixed text residual must remain FP32");
            for (int i = 0; i < 64; ++i)
                require(actual[i] == residual[i] + half_value(delta[i] * modulation[64 + i % 32]),
                        "FP32 text residual with FP16 gate product");
            auto ln = g.bare_norm(normalized);
            require(!ln.fp16_values, "autocast LayerNorm output dtype");
            auto shift = g.columns(mod, 0, 32), scale = g.columns(mod, 32, 32);
            output = g.op(ln, 7, &scale, &shift);
            auto separate = g.download(output);
            auto fused = g.download(g.modulate(normalized, mod, 0));
            require(separate == fused, "fused/separate mixed-precision modulation");
            auto view = g.rows(x, 1, 1);
            require(view.fp16_values && g.columns(x, 3, 7).fp16_values && g.clone(x).fp16_values,
                    "FP16 view/copy dtype propagation");
            g.dit_fp16 = false;
            output = g.gated(x, z, mod, 64);
            actual = g.download(output);
            require(!output.fp16_values, "FP32 graph restored after DiT scope");
            for (int i = 0; i < 64; ++i)
                require(actual[i] == residual[i] + delta[i] * modulation[64 + i % 32],
                        "FP32 residual path preserved");
            std::cout << "FP16 DiT rounding boundaries pass\n";
        }
        {
            // A shifted, low-variance activation exposes cancellation in E[x^2]-E[x]^2.
            g.dit_fp16 = true;
            for (int channels : {1152, 2048, 3072}) {
                std::vector<float> input(channels);
                double mean = 0, variance = 0;
                for (int i = 0; i < channels; ++i) {
                    input[i] = 6.f + .2f * std::sin(i * .013f);
                    mean += input[i];
                }
                mean /= channels;
                for (float value : input)
                    variance += (value - mean) * (value - mean);
                variance /= channels;
                auto actual = g.download(g.bare_norm(g.upload(input, {1, channels}), 0, 1.e-6f));
                for (int i = 0; i < channels; ++i) {
                    double expected = (input[i] - mean) / std::sqrt(variance + 1.e-6);
                    require(std::abs(actual[i] - expected) < 3.e-5,
                            "stable autocast LayerNorm for shifted activations");
                }
            }
            g.dit_fp16 = false;
            std::cout << "FP32 autocast Welford LayerNorm pass\n";
        }
        {
            auto basis = video_rocm::hv15_time_basis();
            auto frequency = g.upload(basis, {128});
            auto embedding = g.empty({1, 256});
            for (float time : {0.f, 1.25f, 127.375f, 753.1875f, 1000.f}) {
                g.launch("hv15_time_embedding", 1, 1, 1, 128, 1, 0, embedding.pointer,
                         frequency.pointer, time);
                auto actual = g.download(embedding);
                for (int i = 0; i < 128; ++i) {
                    float phase = time * basis[i];
                    require(std::abs(actual[i] - std::cos(phase)) < 2.e-7f &&
                                std::abs(actual[i + 128] - std::sin(phase)) < 2.e-7f,
                            "GPU timestep trigonometry at arbitrary times");
                }
            }
            std::cout << "GPU timestep basis/trigonometry pass\n";
        }
        {
            constexpr int rows = 4096, channels = 128;
            std::vector<float> input(rows * channels);
            uint32_t state = 0x157321u;
            for (int row = 0; row < rows; ++row)
                for (int c = 0; c < channels; ++c) {
                    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                    float value = (float(int(state & 65535) - 32768) / 8192.f) *
                                  std::exp2(float(row % 9 - 4));
                    input[row * channels + c] = float((_Float16)value);
                }
            g.dit_fp16 = true;
            auto actual = g.download(g.bare_norm(g.round_half(g.upload(input, {rows, channels})),
                                                 1, 1.e-6f));
            for (int row = 0; row < rows; ++row) {
                float sums[32];
                for (int lane = 0; lane < 32; ++lane) {
                    sums[lane] = 0.f;
                    for (int j = 0; j < 4; ++j) {
                        float value = input[row * channels + lane * 4 + j];
                        sums[lane] += value * value;
                    }
                }
                for (int offset = 1; offset < 32; offset <<= 1)
                    for (int lane = 0; lane < 32 - offset; ++lane)
                        sums[lane] += sums[lane + offset];
                float inverse = float(1.0 / std::sqrt(double(sums[0] / 128.f + 1.e-6f)));
                for (int c = 0; c < channels; ++c) {
                    volatile float product = input[row * channels + c] * inverse;
                    float expected = float((_Float16)product);
                    require(actual[row * channels + c] == expected,
                            "RMSNorm reciprocal and FP32-to-Half rounding boundaries");
                }
            }
            require(actual[12 * channels + 38] == .533203125f,
                    "RMSNorm adversarial double-rounding fixture");
            constexpr int packed_rows = 16, heads = 16;
            std::vector<float> packed(packed_rows * 6144), gamma(128);
            for (int d = 0; d < 128; ++d) gamma[d] = 1.f + (d % 11) * .03125f;
            for (int row = 0; row < packed_rows; ++row)
                for (int h = 0; h < heads; ++h)
                    for (int d = 0; d < 128; ++d) {
                        int source = (row * heads + h) * 128 + d;
                        packed[row * 6144 + h * 128 + d] = input[source];
                        packed[row * 6144 + 2048 + h * 128 + d] = input[source + 256 * 128];
                        packed[row * 6144 + 4096 + h * 128 + d] = input[source + 512 * 128];
                    }
            auto x = g.upload(packed, {packed_rows, 6144}), w = g.upload(gamma, {128});
            auto q = g.empty_half({heads, packed_rows, 128});
            auto k = g.empty_half(q.shape), v = g.empty_half(q.shape);
            g.launch("hv15_half_qkv", packed_rows, heads, 1, 128, 1, 0, q.pointer, k.pointer,
                     v.pointer, x.pointer, w.pointer, w.pointer, packed_rows, 1, 1, 0);
            auto qa = g.download(q), ka = g.download(k), va = g.download(v);
            for (int row = 0; row < packed_rows; ++row)
                for (int h = 0; h < heads; ++h)
                    for (int d = 0; d < 128; ++d) {
                        int source = (row * heads + h) * 128 + d;
                        int target = (h * packed_rows + row) * 128 + d;
                        require(qa[target] == float((_Float16)(actual[source] * gamma[d])) &&
                                ka[target] == float((_Float16)(actual[source + 256 * 128] * gamma[d])) &&
                                va[target] == input[source + 512 * 128],
                                "fused Q/K RMSNorm matches standalone with Half affine weights");
                    }
            g.dit_fp16 = false;
            std::cout << "FP16 RMSNorm 524288-value rounding corpus pass\n";
        }
        CUfunction f;
        for (auto name : {"video_gemm_i8", "video_gemm_i8_tiled", "video_gemm_i8_pipeline"}) {
            g.check(cuModuleGetFunction(&f, g.mma, name), name);
            g.functions[name] = f;
        }
        auto code = Gpu::compile(std::string(h3::prelude) + h3::source);
        CUmodule mod;
        g.check(cuModuleLoadData(&mod, code.data()), "h3 module");
        for (auto name : {"h3_rotate", "h3_quant", "h3_norm", "h3_qwen_attention"}) {
            g.check(cuModuleGetFunction(&f, mod, name), name);
            g.functions[name] = f;
        }
        {
            std::vector<float> input(128), gamma(128, 1.f);
            double square = 0;
            for (int i = 0; i < 128; i++) {
                uint32_t bits = uint32_t(norm_boundary_bits[i]) << 16;
                std::memcpy(&input[i], &bits, sizeof(float));
                square += double(input[i]) * input[i];
            }
            gamma[103] = 1.921875f;
            float expected = float(input[103] * gamma[103] / std::sqrt(square / 128 + 1e-6));
            uint32_t bits;
            std::memcpy(&bits, &expected, sizeof(float));
            bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
            std::memcpy(&expected, &bits, sizeof(float));
            require(expected == -.96484375f, "RMSNorm boundary fixture");
            auto x = g.upload(input, {1, 128}), w = g.upload(gamma, {128}), y = g.empty({1, 128});
            g.launch("h3_norm", 1, 1, 1, 256, 1, 0, y.pointer, x.pointer, w.pointer, CUdeviceptr(0),
                     1, 128, 2, 1e-6f, 1);
            require(g.download(y)[103] == expected, "RMSNorm BF16 boundary regression");
            std::cout << "PASS H3 RMSNorm BF16 boundary against FP64 oracle\n";
        }
        {
            // Pinned PyTorch scalar division uses a rounded reciprocal. These
            // maxima distinguish it from correctly rounded direct division.
            std::vector<float> rotated(512, 0.f);
            rotated[0] = 2.234375f;
            rotated[256] = 1.6171875f;
            auto x = g.upload(rotated, {2, 256}), q = bytes(g, 512), scales = g.empty({2});
            g.launch("h3_quant", 2, 1, 1, 256, 1, 0, q.pointer, scales.pointer, x.pointer, 2, 256,
                     1);
            auto actual = g.download(scales);
            require(actual[0] == .01759350299835205f && actual[1] == .012733759358525276f,
                    "INT8 FP32 reciprocal row-scale regression");
            std::cout << "PASS H3 FP32 reciprocal row scales against pinned Torch fixture\n";
        }
        g.check(cuModuleGetFunction(&f, g.mma, "video_gemm_bf16"), "BF16 GEMM lookup");
        g.functions["video_gemm_bf16"] = f;
        for (auto shape : {std::array<int, 3>{1, 1, 1}, {36, 33, 17}, {17, 35, 257}}) {
            int m = shape[0], n = shape[1], k = shape[2];
            std::vector<float> input(m * k), weights(n * k);
            std::vector<uint16_t> packed(weights.size());
            for (size_t i = 0; i < input.size(); i++)
                input[i] = float(int(i * 7 % 31) - 15) / 16;
            for (size_t i = 0; i < weights.size(); i++) {
                float v = i % 7 ? float(int(i * 11 % 23) - 11) / 128
                                : std::ldexp(1.f + float(i % 128) / 128, -22);
                uint32_t bits;
                std::memcpy(&bits, &v, sizeof(bits));
                packed[i] = bits >> 16;
                bits &= 0xffff0000u;
                std::memcpy(&weights[i], &bits, sizeof(bits));
            }
            if (k == 1)
                input[0] = 1.25f;
            auto x = g.upload(input, {m, k}), w = bytes(g, packed.size() * 2), y = g.empty({m, n});
            g.upload_staged(w, packed.data(), g.stream);
            g.launch("video_gemm_bf16", (n + 31) / 32, (m + 31) / 32, 1, 128, 1, 0, y.pointer,
                     x.pointer, w.pointer, m, n, k);
            auto actual = g.download(y);
            for (int row = 0; row < m; row++)
                for (int col = 0; col < n; col++) {
                    double expected = 0;
                    for (int p = 0; p < k; p++)
                        expected += double(input[row * k + p]) * weights[col * k + p];
                    require(std::abs(actual[row * n + col] - expected) <=
                                2e-6 * std::max(1., std::abs(expected)),
                            "BF16 FP64 GEMM oracle");
                    if (k == 1)
                        require(actual[0] == float(expected),
                                "BF16 precision below FP16 normal range");
                }
            std::cout << "PASS BF16 WMMA " << m << "x" << n << "x" << k << " FP64 oracle\n";
        }
        for (auto name : {"video_gemm_f16", "video_gemm_f16_tiled"}) {
            g.check(cuModuleGetFunction(&f, g.mma, name), name);
            g.functions[name] = f;
        }
        for (auto shape : {std::array<int, 3>{1, 1, 16},
                           {129, 131, 67},
                           {129, 131, 96},
                           {131, 129, 512},
                           {129, 131, 5376}}) {
            int m = shape[0], n = shape[1], k = shape[2];
            std::vector<_Float16> input(m * k), weights(n * k);
            uint32_t state = 17;
            auto value = [&]() {
                state = state * 1664525u + 1013904223u;
                return _Float16((float(state >> 8) / 16777216.f - .5f) * 3.f);
            };
            for (auto &v : input)
                v = value();
            for (auto &v : weights)
                v = value();
            auto x = g.empty_half({m, k}), w = g.empty_half({n, k});
            g.upload_staged(x, input.data(), g.stream);
            g.upload_staged(w, weights.data(), g.stream);
            auto selected = g.matmul(x, w);
            auto b = g.download(selected);
            if (k % 16 == 0) {
                auto basic = g.empty({m, n}), tiled = g.empty({m, n});
                g.launch("video_gemm_f16", (n + 31) / 32, (m + 31) / 32, 1, 128, 1, 0,
                         basic.pointer, x.pointer, w.pointer, m, n, k);
                g.launch("video_gemm_f16_tiled", (n + 127) / 128, (m + 127) / 128, 1, 128, 1, 0,
                         tiled.pointer, x.pointer, w.pointer, m, n, k);
                auto a = g.download(basic), direct = g.download(tiled);
                require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0 &&
                            std::memcmp(a.data(), direct.data(), a.size() * sizeof(float)) == 0,
                        "FP16 tiled WMMA must preserve basic-kernel accumulation");
            }
            for (int row = 0; row < m; row++)
                for (int col = 0; col < n; col++) {
                    double expected = 0, sum_abs = 0;
                    for (int p = 0; p < k; p++) {
                        double product = double(input[row * k + p]) * double(weights[col * k + p]);
                        expected += product;
                        sum_abs += std::abs(product);
                    }
                    // Long reductions accumulate FP32 rounding, including in
                    // the basic kernel. Bound error by the sum's condition,
                    // while requiring bit-exact agreement between kernels.
                    require(std::abs(double(b[row * n + col]) - expected) <
                                std::max(5e-4, 1e-6 * sum_abs),
                            "FP16 tiled FP64 GEMM oracle");
                }
            std::cout << "PASS FP16 tiled WMMA " << m << 'x' << n << 'x' << k
                      << (k % 16 ? " padded K and FP64 oracle\n"
                                 : " bit-exact basic kernel and FP64 oracle\n");
        }
        require(g.f16_tiled_calls > 0, "large FP16 matrices must exercise tiled WMMA");
        for (auto kernel : {"video_gemm_i8", "video_gemm_i8_tiled", "video_gemm_i8_pipeline"}) {
            int tile = std::string(kernel) == "video_gemm_i8" ? 32 : 128;
            int threads = std::string(kernel) == "video_gemm_i8_tiled" ? 256 : 128;
            for (auto shape : {std::array<int, 3>{1, 1, 17},
                               {33, 65, 255},
                               {7, 41, 256},
                               {16, 16, 5376},
                               {129, 131, 32},
                               {131, 129, 96}}) {
                int m = shape[0], n = shape[1], k = shape[2];
                if (std::string(kernel) == "video_gemm_i8_pipeline" && k % 32)
                    continue;
                std::vector<signed char> x(m * k), w(n * k);
                for (size_t i = 0; i < x.size(); i++)
                    x[i] = (i * 29 % 256) - 128;
                for (size_t i = 0; i < w.size(); i++)
                    w[i] = (i * 19 % 256) - 128;
                auto a = bytes(g, x.size()), b = bytes(g, w.size());
                g.upload_staged(a, x.data(), g.stream);
                g.upload_staged(b, w.data(), g.stream);
                auto y = g.empty({m, n}), raw = g.empty({m, n}),
                     xs = g.upload(std::vector<float>(m, .003f), {m}),
                     ws = g.upload(std::vector<float>(n, .007f), {n});
                g.launch(kernel, (n + tile - 1) / tile, (m + tile - 1) / tile, 1, threads, 1, 0,
                         y.pointer, a.pointer, b.pointer, xs.pointer, ws.pointer, m, n, k, 0,
                         raw.pointer);
                auto out = g.download(y);
                std::vector<int> integers(m * n);
                g.check(cuMemcpyDtoH(integers.data(), raw.pointer, raw.bytes()), "read raw");
                for (int r = 0; r < m; r++)
                    for (int c = 0; c < n; c++) {
                        int sum = 0;
                        for (int j = 0; j < k; j++)
                            sum += int(x[r * k + j]) * int(w[c * k + j]);
                        require(integers[r * n + c] == sum, "INT8 integer oracle mismatch");
                        require(std::abs(out[r * n + c] - sum * (.003f * .007f)) < .0001f,
                                "INT8 scale mismatch");
                    }
                std::cout << "PASS signed INT8 WMMA " << kernel << " " << m << "x" << n << "x" << k
                          << "\n";
            }
        }
        {
            const int m = 256, n = 21504, k = 5376;
            auto x = bytes(g, size_t(m) * k), w = bytes(g, size_t(n) * k);
            std::vector<signed char> a(x.bytes(), 1), b(w.bytes(), 1);
            g.upload_staged(x, a.data(), g.stream);
            g.upload_staged(w, b.data(), g.stream);
            auto y = g.empty({m, n}), xs = g.upload(std::vector<float>(m, 1), {m}),
                 ws = g.upload(std::vector<float>(n, 1), {n});
            for (auto name : {"video_gemm_i8", "video_gemm_i8_tiled", "video_gemm_i8_pipeline"}) {
                int tile = std::string(name) == "video_gemm_i8" ? 32 : 128;
                g.check(cuStreamSynchronize(g.stream), "benchmark start");
                auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 3; i++)
                    g.launch(name, (n + tile - 1) / tile, (m + tile - 1) / tile, 1,
                             std::string(name) == "video_gemm_i8_tiled" ? 256 : 128, 1, 0,
                             y.pointer, x.pointer, w.pointer, xs.pointer, ws.pointer, m, n, k, 0,
                             CUdeviceptr(0));
                g.check(cuStreamSynchronize(g.stream), "benchmark end");
                double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                        .count() /
                    3;
                auto result = g.download(y);
                for (float v : result)
                    require(v == k, "large INT8 constant oracle");
                std::cout << name << " 256x21504x5376 ms=" << seconds * 1000
                          << " TOPS=" << 2. * m * n * k / seconds / 1e12 << "\n";
            }
        }
        for (int dim : {64, 128})
            for (bool causal : {false, true})
                for (const char *path :
                     {"original", "base2", "tiled_base2", "tiled_bf16", "tiled_f16", "flex_f16", "qwen_math"}) {
                    bool qwen = std::string(path) == "qwen_math";
                    bool flex = std::string(path) == "flex_f16";
                    bool base2 = std::string(path) == "base2";
                    bool tiled_base2 = std::string(path) == "tiled_base2";
                    if ((base2 || tiled_base2 || flex) && dim != 128)
                        continue;
                    if (qwen && (!causal || dim != 128))
                        continue;
                    for (int rows : {1, 17, 67, 257, 512}) {
                        if (!qwen && !base2 && !tiled_base2 && !flex && rows != 67)
                            continue;
                        const int heads = 4, kvheads = 2;
                        std::vector<float> q(rows * heads * dim), k(rows * kvheads * dim),
                            v(k.size());
                        for (size_t i = 0; i < q.size(); i++)
                            q[i] = float(int(i * 7 % 31) - 15) / 16;
                        for (size_t i = 0; i < k.size(); i++) {
                            k[i] = float(int(i * 11 % 29) - 14) / 16;
                            v[i] = float(int(i * 13 % 23) - 11) / 16;
                        }
                        auto tq = g.upload(q, {rows, heads * dim}),
                             tk = g.upload(k, {rows, kvheads * dim}),
                             tv = g.upload(v, {rows, kvheads * dim}), out = g.empty(tq.shape);
                        const char *name =
                            dim == 128 ? "video_attention_bf16" : "video_attention_bf16_64";
                        bool tiled = std::string(path) != "original" && !base2;
                        if (base2)
                            name = "video_attention_bf16_exp2";
                        if (tiled)
                            name = std::string(path) == "tiled_bf16"
                                       ? (dim == 128 ? "video_attention_tile_bf16"
                                                     : "video_attention_tile_bf16_64")
                                       : (dim == 128 ? "video_attention_tile_f16"
                                                     : "video_attention_tile_f16_64");
                        if (tiled_base2)
                            name = "video_attention_tile_bf16_exp2";
                        if (flex) name = "video_attention_flex_f16";
                        if (qwen) {
                            g.launch("h3_qwen_attention", rows, heads, 1, 32, 1, 0, out.pointer,
                                     tq.pointer, tk.pointer, tv.pointer, rows, heads, kvheads, dim,
                                     float(std::sqrt(1.0 / std::sqrt(double(dim)))));
                        } else {
                            g.check(cuModuleGetFunction(&f, g.flash, name), name);
                            g.functions[name] = f;
                            g.launch(name, heads, (rows + (tiled ? 127 : 63)) / (tiled ? 128 : 64),
                                     1, tiled ? 256 : 128, 1, 0, out.pointer, tq.pointer,
                                     tk.pointer, tv.pointer, rows, heads, dim,
                                     1.f / std::sqrt(float(dim)), kvheads, int(causal));
                        }
                        auto actual = g.download(out);
                        double err = 0, refnorm = 0, actualnorm = 0, dot = 0;
                        for (int r = 0; r < rows; r++)
                            for (int h = 0; h < heads; h++) {
                                std::vector<double> prob(rows);
                                double maximum = -1e30, sum = 0;
                                for (int j = 0; j < rows; j++) {
                                    double dot = 0;
                                    for (int d = 0; d < dim; d++)
                                        dot += q[(r * heads + h) * dim + d] *
                                               k[(j * kvheads + h / (heads / kvheads)) * dim + d];
                                    prob[j] =
                                        (causal && j > r) ? -1e30 : dot / std::sqrt(double(dim));
                                    maximum = std::max(maximum, prob[j]);
                                }
                                for (auto &p : prob) {
                                    p = std::exp(p - maximum);
                                    sum += p;
                                }
                                for (int d = 0; d < dim; d++) {
                                    double expected = 0;
                                    for (int j = 0; j < rows; j++)
                                        expected +=
                                            prob[j] / sum *
                                            v[(j * kvheads + h / (heads / kvheads)) * dim + d];
                                    double delta = actual[(r * heads + h) * dim + d] - expected;
                                    err += delta * delta;
                                    refnorm += expected * expected;
                                    actualnorm += double(actual[(r * heads + h) * dim + d]) *
                                                  actual[(r * heads + h) * dim + d];
                                    dot += actual[(r * heads + h) * dim + d] * expected;
                                }
                            }
                        std::cerr << "attention dim=" << dim << " causal=" << causal
                                  << " error=" << std::sqrt(err / refnorm) << " first=" << actual[0]
                                  << "\n";
                        require(std::sqrt(err / refnorm) < (qwen ? .003 : .02) &&
                                    dot / std::sqrt(actualnorm * refnorm) >= .9999,
                                "BF16 attention CPU softmax oracle mismatch");
                        std::cout << "PASS " << path << " GQA attention dim=" << dim
                                  << " rows=" << rows << " causal=" << causal
                                  << " rel_l2=" << std::sqrt(err / refnorm) << "\n";
                    }
                }
        std::vector<float> x(512);
        for (size_t i = 0; i < x.size(); i++)
            x[i] = float(int(i % 29) - 14) / 16;
        auto a = g.upload(x, {2, 256}), b = g.empty({2, 256});
        g.launch("h3_rotate", 2, 1, 1, 256, 1, 0, b.pointer, a.pointer, 2, 0);
        auto out = g.download(b);
        for (int group = 0; group < 2; group++)
            for (int c = 0; c < 256; c++) {
                float sum = 0;
                for (int j = 0; j < 256; j++) {
                    int sign = 1;
                    for (int d = 1; d <= 64; d *= 4)
                        if ((c / d) % 4 + (j / d) % 4 == 3)
                            sign = -sign;
                    sum += x[group * 256 + j] * sign / 16;
                }
                require(std::abs(sum - out[group * 256 + c]) < 1e-5f, "ConvRot oracle mismatch");
            }
        auto q = bytes(g, 512), scale = g.empty({2});
        g.launch("h3_quant", 2, 1, 1, 256, 1, 0, q.pointer, scale.pointer, b.pointer, 2, 256, 0);
        auto scales = g.download(scale);
        std::vector<signed char> quant(512);
        g.check(cuMemcpyDtoH(quant.data(), q.pointer, 512), "quant");
        for (int r = 0; r < 2; r++) {
            float mx = 0;
            for (int c = 0; c < 256; c++)
                mx = std::max(mx, std::abs(out[r * 256 + c]));
            require(std::abs(scales[r] - mx / 127) < 1e-7, "scale oracle mismatch");
            for (int c = 0; c < 256; c++)
                require(quant[r * 256 + c] == int(std::nearbyint(out[r * 256 + c] / scales[r])),
                        "quant oracle mismatch");
        }
        std::cout << "PASS ConvRot-256 and rowwise INT8 quantization\n";
        g.check(cuStreamSynchronize(g.stream), "sync");
        cuModuleUnload(mod);
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
