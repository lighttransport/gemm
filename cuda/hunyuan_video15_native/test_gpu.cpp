#include "gpu.hpp"
#include <array>
#include <iostream>
using namespace hv15n;

static std::vector<float> values(size_t n, float phase = 0.f) {
  std::vector<float> result(n);
  for (size_t i = 0; i < n; ++i)
    result[i] = .2f * std::sin(float(i) * .71f + phase);
  return result;
}
static void compare(const std::vector<float> &actual,
                    const std::vector<float> &expected, double tolerance,
                    const char *name) {
  double error = 0, norm = 0, dot = 0, aa = 0;
  require(actual.size() == expected.size(), "comparison shape");
  for (size_t i = 0; i < actual.size(); ++i) {
    error += std::pow(actual[i] - expected[i], 2);
    norm += double(expected[i]) * expected[i];
    aa += double(actual[i]) * actual[i];
    dot += double(actual[i]) * expected[i];
  }
  double relative = std::sqrt(error / std::max(norm, 1.e-30));
  double cosine = dot / std::sqrt(aa * norm);
  std::cout << name << " cosine=" << cosine << " relative_l2=" << relative
            << '\n';
  require(relative <= tolerance && cosine >= .9999,
          std::string(name) + " mismatch");
}
static void attention_test(Gpu &g, int rows, int heads, int kvheads, int dim,
                           int mask, bool precise, bool biased) {
  auto q = values(size_t(rows) * heads * dim);
  auto k = values(size_t(rows) * kvheads * dim, .9f);
  auto v = values(k.size(), 2.f), bias = values(size_t(32) * heads);
  auto qt = g.upload(q, {rows, heads * dim}),
       kt = g.upload(k, {rows, kvheads * dim});
  auto vt = g.upload(v, {rows, kvheads * dim}),
       bt = g.upload(bias, {32, heads});
  float scale = biased ? 1.f : 1.f / std::sqrt(float(dim));
  auto result = g.download(g.attention(qt, kt, vt, heads, kvheads, mask, 7,
                                       precise, biased ? &bt : nullptr, scale));
  std::vector<float> expected(result.size());
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < heads; ++h) {
      std::vector<double> scores(rows, -INFINITY);
      double maximum = -INFINITY;
      int kh = h / (heads / kvheads);
      for (int j = 0; j < rows; ++j) {
        if ((mask == 1 && j > r) || (mask == 2 && j / 7 > r / 7))
          continue;
        double dot = 0;
        for (int d = 0; d < dim; ++d)
          dot += double(q[(r * heads + h) * dim + d]) *
                 k[(j * kvheads + kh) * dim + d];
        scores[j] = dot * scale +
                    (biased ? bias[relative_bucket(r, j) * heads + h] : 0);
        maximum = std::max(maximum, scores[j]);
      }
      double total = 0;
      for (auto &score : scores) {
        score = std::exp(score - maximum);
        total += score;
      }
      for (int d = 0; d < dim; ++d) {
        double sum = 0;
        for (int j = 0; j < rows; ++j)
          sum += scores[j] * v[(j * kvheads + kh) * dim + d];
        expected[(r * heads + h) * dim + d] = float(sum / total);
      }
    }
  compare(result, expected,
          precise || biased || heads != kvheads ? 2.e-5 : .004,
          precise ? "IEEE attention" : "repository attention");
}
int main(int argc, char **argv) {
  try {
    bool vendor = argc > 1 && std::string(argv[1]) == "cublas";
    bool fallback = !(argc > 1 && std::string(argv[1]) == "repo-only");
    bool memory = argc > 1 && std::string(argv[1]) == "memory";
    Gpu g(0, memory ? 14336 : 4096, vendor, fallback);
    for (const char *name : {"gemm_f16", "gemm_f16_large"}) {
      CUfunction fn;
      int local = 0, regs = 0;
      g.check(cuModuleGetFunction(&fn, g.mma, name), "kernel lookup");
      g.check(
          cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn),
          "kernel local memory");
      g.check(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn),
              "kernel registers");
      std::cout << name << " local_bytes=" << local << " registers=" << regs
                << '\n';
    }
    if (memory) {
      size_t before = 0, after = 0, total = 0;
      g.check(cuMemGetInfo(&before, &total), "memory before");
      auto x =
          g.upload(std::vector<float>(size_t(33390) * 2048), {33390, 2048});
      auto w = g.upload(std::vector<float>(size_t(8192) * 2048), {8192, 2048});
      auto y = g.matmul(x, w);
      g.check(cuStreamSynchronize(g.stream), "memory test synchronize");
      g.check(cuMemGetInfo(&after, &total), "memory after");
      std::cout << "large GEMM managed_mib=" << g.allocated / 1048576.
                << " device_growth_mib=" << double(before - after) / 1048576.
                << '\n';
      require(before >= after && before - after <= g.allocated + (512ull << 20),
              "large GEMM has excessive unmanaged scratch");
      return 0;
    }
    for (auto shape :
         {std::array<int, 3>{19, 384, 1472}, std::array<int, 3>{2083, 273, 37}})
      for (bool precise : {false, true}) {
        int m = shape[0], n = shape[1], k = shape[2];
        auto x = values(m * k), w = values(n * k, .5f);
        std::vector<float> expected(m * n);
        for (int r = 0; r < m; ++r)
          for (int c = 0; c < n; ++c) {
            double sum = 0;
            for (int j = 0; j < k; ++j)
              sum += double(x[r * k + j]) * w[c * k + j];
            expected[r * n + c] = float(sum);
          }
        compare(g.download(g.matmul(g.upload(x, {m, k}), g.upload(w, {n, k}),
                                    precise)),
                expected, precise ? 2.e-5 : .004,
                precise ? "IEEE GEMM" : "repository GEMM");
      }
    for (int mask : {0, 1, 2})
      attention_test(g, 137, 2, 2, 32, mask, false, false);
    attention_test(g, 2049, 2, 2, 32, 1, false, false);
    attention_test(g, 17, 4, 2, 16, 1, true, false);
    attention_test(g, 19, 6, 6, 64, 0, true, true);
    auto x = values(11 * 37);
    auto tensor = g.upload(x, {11, 37});
    for (int mode : {0, 1, 2}) {
      auto expected = x;
      for (int r = 0; r < 11; ++r) {
        double sum = 0, square = 0;
        for (int c = 0; c < 37; ++c) {
          double v = x[r * 37 + c];
          sum += v;
          square += v * v;
        }
        double mean = mode ? 0 : sum / 37;
        double inverse = mode == 2
                             ? std::sqrt(37.) / std::sqrt(square)
                             : 1 / std::sqrt(square / 37 - mean * mean + 1.e-6);
        for (int c = 0; c < 37; ++c)
          expected[r * 37 + c] = float((x[r * 37 + c] - mean) * inverse);
      }
      compare(g.download(g.bare_norm(tensor, mode)), expected, 2.e-5,
              "normalization");
    }
    g.cancelled = true;
    bool cancelled = false;
    try {
      g.op(tensor, 3);
    } catch (const std::exception &) {
      cancelled = true;
    }
    require(cancelled, "GPU cancellation ignored");
    std::cout << "PASS " << g.metrics() << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
