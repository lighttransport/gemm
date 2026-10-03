// SPDX-License-Identifier: MIT
#include "../../cuda/hunyuan_video15_native/gpu.hpp"
#include <iostream>
using namespace hv15n;
int main(int argc, char **argv) {
    try {
        require(argc == 2, "test_aotriton BRIDGE");
        Gpu gpu(0, 4096, false, false);
        gpu.use_aotriton(argv[1]);
        gpu.dit_fp16 = true;
        const int rows = 137, heads = 2, dim = 128, channels = heads * dim;
        std::array<std::vector<float>, 3> values;
        std::array<Tensor, 3> row, packed;
        for (int operand = 0; operand < 3; ++operand) {
            values[operand].resize(rows * channels);
            std::vector<_Float16> rearranged(rows * channels);
            for (int r = 0; r < rows; ++r)
                for (int h = 0; h < heads; ++h)
                    for (int d = 0; d < dim; ++d) {
                        float value = float(
                            (_Float16)(std::sin(float(r * 17 + d * 3 + h * 11 + operand * 23)) *
                                       .5f));
                        values[operand][r * channels + h * dim + d] = value;
                        rearranged[(h * rows + r) * dim + d] = (_Float16)value;
                    }
            row[operand] = gpu.upload(values[operand], {rows, channels});
            packed[operand] = gpu.empty_half({rows, channels});
            packed[operand].packed_heads = heads;
            gpu.upload_staged(packed[operand], rearranged.data(), gpu.stream);
        }
        auto a = gpu.download(gpu.attention(row[0], row[1], row[2], heads, heads));
        auto b = gpu.download(gpu.attention(packed[0], packed[1], packed[2], heads, heads));
        require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
                "AOTriton row/head input layouts must match exactly");
        double maximum = 0;
        for (int r = 0; r < rows; ++r)
            for (int h = 0; h < heads; ++h) {
                std::vector<double> probabilities(rows);
                double top = -INFINITY, sum = 0;
                for (int k = 0; k < rows; ++k) {
                    double score = 0;
                    for (int d = 0; d < dim; ++d)
                        score += double(values[0][r * channels + h * dim + d]) *
                                 values[1][k * channels + h * dim + d];
                    probabilities[k] = score / std::sqrt(double(dim));
                    top = std::max(top, probabilities[k]);
                }
                for (auto &p : probabilities) {
                    p = std::exp(p - top);
                    sum += p;
                }
                for (int d = 0; d < dim; ++d) {
                    double expected = 0;
                    for (int k = 0; k < rows; ++k)
                        expected += probabilities[k] * values[2][k * channels + h * dim + d];
                    expected /= sum;
                    maximum = std::max(maximum, std::abs(expected - a[r * channels + h * dim + d]));
                }
            }
        require(maximum < 5e-4, "AOTriton FP16 FP64 softmax oracle");
        require(gpu.aot_calls == 2, "provider must handle both input layouts");
        std::cout << "PASS FP16 AOTriton row/head layouts and FP64 oracle max_abs=" << maximum
                  << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
