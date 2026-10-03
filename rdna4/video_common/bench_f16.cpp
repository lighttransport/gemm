// SPDX-License-Identifier: MIT
#include "../../cuda/hunyuan_video15_native/gpu.hpp"
#include <iostream>
using namespace hv15n;
int main() {
    try {
        Gpu gpu(0, 4096, false, false);
        for (auto name : {"video_gemm_f16", "video_gemm_f16_tiled"}) {
            CUfunction function;
            gpu.check(cuModuleGetFunction(&function, gpu.mma, name), name);
            gpu.functions[name] = function;
        }
        for (auto dimensions :
             {std::array<int, 3>{256, 8192, 2048}, std::array<int, 3>{256, 21504, 5376},
              std::array<int, 3>{4096, 2048, 2048}}) {
            const int m = dimensions[0], n = dimensions[1], k = dimensions[2];
            {
                std::vector<float> input(size_t(m) * k), weights(size_t(n) * k);
                for (size_t i = 0; i < input.size(); ++i)
                    input[i] = float(int(i % 29) - 14) / 32.f;
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = float(int(i % 31) - 15) / 32.f;
                auto x = gpu.half(gpu.upload(input, {m, k}));
                auto w = gpu.half(gpu.upload(weights, {n, k}));
                auto basic = gpu.empty({m, n}), tiled = gpu.empty({m, n});
                auto measure = [&](const char *kernel, int tile, Tensor &output) {
                    auto launch = [&] {
                        gpu.launch(kernel, (n + tile - 1) / tile, (m + tile - 1) / tile, 1, 128, 1,
                                   0, output.pointer, x.pointer, w.pointer, m, n, k);
                    };
                    for (int repeat = 0; repeat < 3; ++repeat)
                        launch();
                    gpu.check(cuStreamSynchronize(gpu.stream), "warmup");
                    auto start = std::chrono::steady_clock::now();
                    for (int repeat = 0; repeat < 10; ++repeat)
                        launch();
                    gpu.check(cuStreamSynchronize(gpu.stream), "benchmark");
                    double ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count() /
                                10;
                    std::cout << kernel << " " << m << "x" << n << "x" << k << " ms=" << ms
                              << " TFLOPS=" << 2. * m * n * k / (ms * 1e9) << std::endl;
                };
                measure("video_gemm_f16", 32, basic);
                measure("video_gemm_f16_tiled", 128, tiled);
                auto reference = gpu.download(basic), actual = gpu.download(tiled);
                size_t mismatches = 0;
                double maximum_error = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    mismatches += reference[i] != actual[i];
                    maximum_error =
                        std::max(maximum_error, std::abs(double(actual[i]) - reference[i]));
                }
                std::cout << "mismatches=" << mismatches << " max_abs=" << maximum_error
                          << std::endl;
                require(!mismatches, "tiled FP16 benchmark must match the basic kernel exactly");
                std::cout << "bit_exact=pass" << std::endl;
            }
            gpu.trim_pool();
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
