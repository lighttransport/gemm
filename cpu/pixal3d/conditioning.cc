/* Native adaptation of Pixal3D DINOv3 conditioning and NAF.
 * NAF portions adapted from valeoai/NAF under Apache-2.0; see
 * ref/pixal3d/licenses/NAF-Apache-2.0.txt and THIRD_PARTY.md.
 * Modified: portable C++ convolutions and projected-query neighborhood attention. */
#include "engine.hh"
#include <array>
#include <set>

namespace px {
/* DINO/NAF use axial split-half 2D RoPE, unlike the flow's complex 3D RoPE. */
static void rope2d(Vec &x, int grid, int heads, int dim, int prefix, const float *periods = nullptr) {
    int c = heads * dim, half = dim / 2, quarter = dim / 4;
#pragma omp parallel for schedule(static)
    for (int p = 0; p < grid * grid; ++p)
        for (int h = 0; h < heads; ++h)
            for (int j = 0; j < half; ++j) {
                float coord = 2 * ((j < quarter ? p / grid : p % grid) + .5f) / grid - 1;
                float period = periods ? periods[j % quarter] : std::pow(100.f, float(j % quarter) / quarter);
                float angle = 6.283185307179586f * coord / period, co = std::cos(angle), si = std::sin(angle);
                size_t k = size_t(p + prefix) * c + h * dim + j;
                float a = x[k], b = x[k + half];
                x[k] = a * co - b * si;
                x[k + half] = b * co + a * si;
            }
}
Vec dino(Engine &e, Weights &w, const Vec &chw, int size, int blocks) {
    require(size > 0 && size % 16 == 0 && chw.size() == size_t(size) * size * 3, "Invalid DINO image");
    auto shape = w.shape("patch_embed.proj.weight");
    require(shape == std::vector<int>({1024, 3, 16, 16}), "Expected DINOv3 ViT-L/16 weights");
    int grid = size / 16, n = grid * grid + 5, c = 1024;
    Vec patches(size_t(grid) * grid * 768);
#pragma omp parallel for schedule(static)
    for (int p = 0; p < grid * grid; ++p)
        for (int ch = 0; ch < 3; ++ch)
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x)
                    patches[size_t(p) * 768 + ch * 256 + y * 16 + x] =
                        chw[size_t(ch) * size * size + (p / grid * 16 + y) * size + p % grid * 16 + x];
    Vec h(size_t(n) * c);
    e.gemm(h.data() + 5 * c, patches.data(), w.get("patch_embed.proj.weight"), w.get("patch_embed.proj.bias"),
           n - 5, c, 768);
    std::copy_n(w.get("cls_token"), c, h.data());
    std::copy_n(w.get(w.has("reg_token") ? "reg_token" : "storage_tokens"), 4 * c, h.data() + c);
    for (int block = 0; block < blocks; ++block) {
        std::string b = "blocks." + std::to_string(block) + ".";
        Vec x;
        norm(x, h, c, 1e-5f, w.get(b + "norm1.weight"), w.get(b + "norm1.bias"));
        Vec qkv = e.linear(x, w, b + "attn.qkv"), q(h.size()), k(h.size()), v(h.size());
        for (int i = 0; i < n; ++i) {
            std::copy_n(qkv.data() + size_t(i) * 3 * c, c, q.data() + size_t(i) * c);
            std::copy_n(qkv.data() + size_t(i) * 3 * c + c, c, k.data() + size_t(i) * c);
            std::copy_n(qkv.data() + size_t(i) * 3 * c + 2 * c, c, v.data() + size_t(i) * c);
        }
        rope2d(q, grid, 16, 64, 5);
        rope2d(k, grid, 16, 64, 5);
        e.attention(x.data(), q.data(), k.data(), v.data(), n, n, 16, 64);
        x = e.linear(x, w, b + "attn.proj");
        const float *gamma = w.get(w.has(b + "gamma_1") ? b + "gamma_1" : b + "ls1.gamma");
        for (size_t i = 0; i < h.size(); ++i)
            h[i] += x[i] * gamma[i % c];
        norm(x, h, c, 1e-5f, w.get(b + "norm2.weight"), w.get(b + "norm2.bias"));
        x = e.linear(x, w, b + "mlp.fc1");
        gelu(x, false);
        x = e.linear(x, w, b + "mlp.fc2");
        gamma = w.get(w.has(b + "gamma_2") ? b + "gamma_2" : b + "ls2.gamma");
        for (size_t i = 0; i < h.size(); ++i)
            h[i] += x[i] * gamma[i % c];
    }
    Vec out;
    norm(out, h, c, 1e-5f);
    return out;
}

static Vec conv2d(Engine &e, Weights &w, const Vec &x, int size, const std::string &name) {
    auto s = w.shape(name + ".weight");
    require(s.size() == 4 && s[2] == s[3], "Invalid NAF convolution");
    int co = s[0], ci = s[1], kernel = s[2], kk = ci * kernel * kernel, rows = size * size;
    require(x.size() == size_t(rows) * ci, "Bad NAF activation shape");
    Vec out(size_t(rows) * co), gather(size_t(512) * kk);
    for (int start = 0; start < rows; start += 512) {
        int n = std::min(512, rows - start);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < n; ++r)
            for (int ch = 0; ch < ci; ++ch)
                for (int y = 0; y < kernel; ++y)
                    for (int z = 0; z < kernel; ++z) {
                        int py = (start + r) / size + y - kernel / 2,
                            px = (start + r) % size + z - kernel / 2;
                        if (py < 0)
                            py = -py;
                        if (py >= size)
                            py = 2 * size - 2 - py;
                        if (px < 0)
                            px = -px;
                        if (px >= size)
                            px = 2 * size - 2 - px;
                        gather[size_t(r) * kk + (ch * kernel + y) * kernel + z] =
                            x[(size_t(py) * size + px) * ci + ch];
                    }
        e.gemm(out.data() + size_t(start) * co, gather.data(), w.get(name + ".weight"), w.get(name + ".bias"),
               n, co, kk);
    }
    return out;
}
static void group_silu(Vec &x, int channels, Weights &w, const std::string &name) {
    const int groups = 8, cg = channels / groups;
    size_t rows = x.size() / channels, count = rows * cg;
    const float *gamma = w.get(name + ".weight"), *bias = w.get(name + ".bias");
#pragma omp parallel for schedule(static)
    for (int g = 0; g < groups; ++g) {
        double mean = 0, var = 0;
        for (size_t p = 0; p < rows; ++p)
            for (int c = g * cg; c < (g + 1) * cg; ++c)
                mean += x[p * channels + c];
        mean /= count;
        for (size_t p = 0; p < rows; ++p)
            for (int c = g * cg; c < (g + 1) * cg; ++c) {
                double d = x[p * channels + c] - mean;
                var += d * d;
            }
        float inv = 1 / std::sqrt(float(var / count) + 1e-5f);
        for (size_t p = 0; p < rows; ++p)
            for (int c = g * cg; c < (g + 1) * cg; ++c) {
                float v = (x[p * channels + c] - float(mean)) * inv * gamma[c] + bias[c];
                x[p * channels + c] = v / (1 + std::exp(-v));
            }
    }
}
static Vec average(const Vec &x, int source, int target, int channels) {
    if (source == target)
        return x;
    Vec out(size_t(target) * target * channels);
#pragma omp parallel for schedule(static)
    for (int p = 0; p < target * target; ++p) {
        int y = p / target, z = p % target;
        int y0 = y * source / target, y1 = ((y + 1) * source + target - 1) / target;
        int x0 = z * source / target, x1 = ((z + 1) * source + target - 1) / target;
        for (int c = 0; c < channels; ++c) {
            float v = 0;
            for (int i = y0; i < y1; ++i)
                for (int j = x0; j < x1; ++j)
                    v += x[(size_t(i) * source + j) * channels + c];
            out[size_t(p) * channels + c] = v / ((y1 - y0) * (x1 - x0));
        }
    }
    return out;
}
Vec naf_guide(Engine &e, Weights &w, const Vec &image, int size, int target) {
    require(size >= 3 && size <= 4 * target && image.size() == size_t(size) * size * 3, "Invalid NAF image");
    Vec guide(size_t(target) * target * 256);
    for (int branch = 0; branch < 2; ++branch) {
        std::string b = branch ? "image_encoder.sem_encoder." : "image_encoder.encoder.";
        Vec x = conv2d(e, w, image, size, b + "0");
        for (int block = 1; block <= 2; ++block) {
            std::string p = b + std::to_string(block) + ".";
            group_silu(x, 128, w, p + "norm1");
            x = conv2d(e, w, x, size, p + "conv1");
            group_silu(x, 128, w, p + "norm2");
            x = conv2d(e, w, x, size, p + "conv2");
        }
        x = average(x, size, target, 128);
        for (int p = 0; p < target * target; ++p)
            std::copy_n(x.data() + size_t(p) * 128, 128, guide.data() + size_t(p) * 256 + branch * 128);
    }
    rope2d(guide, target, 4, 64, 0, w.get("image_encoder.rope.periods"));
    return guide;
}
Vec naf_sample(const Vec &guide, int target, const Vec &patches, int grid, const Vec &xy) {
    require(grid >= 9 && target % grid == 0 && guide.size() == size_t(target) * target * 256 &&
                patches.size() == size_t(grid) * grid * 1024,
            "Invalid NAF feature geometry");
    Vec keys = average(guide, target, grid, 256);
    std::vector<std::array<int, 4>> ids(xy.size() / 2);
    Vec frac(xy.size());
    std::set<int> pixels;
    for (size_t i = 0; i < ids.size(); ++i) {
        float x = std::clamp((xy[2 * i] + 1) * target * .5f - .5f, 0.f, float(target - 1));
        float y = std::clamp((xy[2 * i + 1] + 1) * target * .5f - .5f, 0.f, float(target - 1));
        int x0 = int(x), y0 = int(y), x1 = std::min(x0 + 1, target - 1), y1 = std::min(y0 + 1, target - 1);
        ids[i] = {y0 * target + x0, y0 * target + x1, y1 * target + x0, y1 * target + x1};
        pixels.insert(ids[i].begin(), ids[i].end());
        frac[2 * i] = x - x0;
        frac[2 * i + 1] = y - y0;
    }
    std::vector<int> unique(pixels.begin(), pixels.end());
    std::vector<int> index(size_t(target) * target, -1);
    for (size_t i = 0; i < unique.size(); ++i)
        index[unique[i]] = int(i);
    Vec sampled(unique.size() * 1024);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < unique.size(); ++i) {
        int p = unique[i], dilation = target / grid;
        int sy = std::clamp(p / target / dilation - 4, 0, grid - 9),
            sx = std::clamp(p % target / dilation - 4, 0, grid - 9);
        for (int head = 0; head < 4; ++head) {
            float scores[81], max = -INFINITY, sum = 0;
            int neighbors[81];
            for (int y = 0; y < 9; ++y)
                for (int x = 0; x < 9; ++x) {
                    int j = y * 9 + x, k = (sy + y) * grid + sx + x;
                    neighbors[j] = k;
                    float dot = 0;
                    for (int c = 0; c < 64; ++c)
                        dot += guide[size_t(p) * 256 + head * 64 + c] * keys[size_t(k) * 256 + head * 64 + c];
                    scores[j] = dot * .125f;
                    max = std::max(max, scores[j]);
                }
            for (float &v : scores) {
                v = std::exp(v - max);
                sum += v;
            }
            float *dst = sampled.data() + i * 1024 + head * 256;
            for (int j = 0; j < 81; ++j) {
                float a = scores[j] / sum;
                const float *v = patches.data() + size_t(neighbors[j]) * 1024 + head * 256;
                for (int c = 0; c < 256; ++c)
                    dst[c] += a * v[c];
            }
        }
    }
    Vec out(ids.size() * 1024);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ids.size(); ++i) {
        float x = frac[2 * i], y = frac[2 * i + 1];
        const float *p0 = sampled.data() + size_t(index[ids[i][0]]) * 1024,
                    *p1 = sampled.data() + size_t(index[ids[i][1]]) * 1024;
        const float *p2 = sampled.data() + size_t(index[ids[i][2]]) * 1024,
                    *p3 = sampled.data() + size_t(index[ids[i][3]]) * 1024;
        for (int c = 0; c < 1024; ++c)
            out[i * 1024 + c] = ((1 - x) * p0[c] + x * p1[c]) * (1 - y) + ((1 - x) * p2[c] + x * p3[c]) * y;
    }
    return out;
}
Vec naf(Engine &e, Weights &w, const Vec &image, int size, const Vec &patches, int grid, int target,
        const Vec &xy) {
    return naf_sample(naf_guide(e, w, image, size, target), target, patches, grid, xy);
}
} // namespace px
