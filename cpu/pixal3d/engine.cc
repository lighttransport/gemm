#include "engine.hh"
#include <cblas.h>
#include <dlfcn.h>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <omp.h>

namespace px {
static float half_to_float(uint16_t h) {
    int sign = h >> 15, exp = (h >> 10) & 31, frac = h & 1023;
    float x = exp == 0 ? std::ldexp(float(frac), -24)
              : exp == 31
                  ? (frac ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                  : std::ldexp(float(1024 + frac), exp - 25);
    return sign ? -x : x;
}
Weights::Weights(const std::string &path) {
    st = safetensors_open(path.c_str());
    require(st, "Cannot open weights: " + path);
}
Weights::~Weights() {
    if (st)
        safetensors_close(st);
}
bool Weights::has(const std::string &name) const {
    return safetensors_find(st, name.c_str()) >= 0;
}
std::vector<int> Weights::shape(const std::string &name) const {
    int i = safetensors_find(st, name.c_str());
    require(i >= 0, "Missing tensor: " + name);
    std::vector<int> s;
    for (int j = 0; j < safetensors_ndims(st, i); ++j) {
        auto d = safetensors_shape(st, i)[j];
        require(d > 0 && d <= INT32_MAX, "Invalid tensor dimension: " + name);
        s.push_back(int(d));
    }
    return s;
}
const float *Weights::get(const std::string &name) {
    int i = safetensors_find(st, name.c_str());
    require(i >= 0, "Missing tensor: " + name);
    const char *dtype = safetensors_dtype(st, i);
    size_t n = 1;
    for (int d : shape(name)) {
        require(n <= SIZE_MAX / size_t(d), "Tensor size overflow: " + name);
        n *= d;
    }
    require(n <= SIZE_MAX / 4, "Tensor size overflow: " + name);
    if (!std::strcmp(dtype, "F32")) {
        require(safetensors_nbytes(st, i) == n * 4, "Invalid F32 tensor: " + name);
        return static_cast<const float *>(safetensors_data(st, i));
    }
    auto it = converted.find(name);
    if (it != converted.end())
        return it->second.data();
    require(!std::strcmp(dtype, "F16") || !std::strcmp(dtype, "BF16"),
            "Unsupported learned tensor dtype: " + name);
    require(safetensors_nbytes(st, i) == n * 2, "Invalid 16-bit tensor: " + name);
    auto &out = converted[name];
    out.resize(n);
    auto *h = static_cast<const uint16_t *>(safetensors_data(st, i));
    for (size_t j = 0; j < n; ++j) {
        if (!std::strcmp(dtype, "F16"))
            out[j] = half_to_float(h[j]);
        else {
            uint32_t u = uint32_t(h[j]) << 16;
            std::memcpy(&out[j], &u, 4);
        }
    }
    return out.data();
}
Engine::Engine(const pixal3d_options &o) : threads(o.threads) {
    require(threads > 0, "threads must be positive");
    openblas_set_num_threads(threads);
    omp_set_num_threads(threads);
    if (o.backend == PIXAL3D_CPU)
        return;
    require(o.backend == PIXAL3D_CUDA || o.backend == PIXAL3D_ROCM, "Invalid backend");
    const char *name = o.backend == PIXAL3D_CUDA ? "libpixal3d_cuda.so" : "libpixal3d_rocm.so";
    Dl_info info{};
    dladdr(reinterpret_cast<void *>(&pixal3d_project), &info);
    std::filesystem::path base =
        info.dli_fname ? std::filesystem::absolute(info.dli_fname).parent_path() : ".";
    std::filesystem::path root = base / "../..";
    std::filesystem::path path = root / (o.backend == PIXAL3D_CUDA ? "cuda/pixal3d" : "rdna4/pixal3d") / name;
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
        const char *err = dlerror();
        throw std::runtime_error("Cannot load " + path.string() + ": " + (err ? err : "unknown"));
    }
#define PX_SYM(field, symbol)                                                                                \
    api_.field = reinterpret_cast<decltype(api_.field)>(dlsym(library_, symbol));                            \
    require(api_.field, "Missing GPU entrypoint " symbol)
    try {
        PX_SYM(create, "px_gpu_create");
        PX_SYM(destroy, "px_gpu_destroy");
        PX_SYM(gemm, "px_gpu_gemm");
        PX_SYM(attention, "px_gpu_attention");
        PX_SYM(peak, "px_gpu_peak");
        PX_SYM(error, "px_gpu_error");
        gpu_ = api_.create(o.device, o.vram_budget_mib * 1024 * 1024);
        require(gpu_, "GPU initialization failed; verify device access and memory budget");
    } catch (...) {
        dlclose(library_);
        library_ = nullptr;
        throw;
    }
#undef PX_SYM
}
Engine::~Engine() {
    if (gpu_)
        api_.destroy(gpu_);
    if (library_)
        dlclose(library_);
}
size_t Engine::peak() const {
    return gpu_ ? api_.peak(gpu_) : 0;
}
void Engine::gemm(float *out, const float *x, const float *w, const float *b, int n, int co, int ci, int bf) {
    require(out && x && w && n > 0 && co > 0 && ci > 0, "Invalid GEMM");
    if (gpu_) {
        int rc = api_.gemm(gpu_, out, x, w, b, n, co, ci, bf);
        require(rc == 0, api_.error(gpu_));
        return;
    }
    Vec bx, bw;
    if (bf) {
        bx.assign(x, x + size_t(n) * ci);
        bw.assign(w, w + size_t(co) * ci);
        round_precision(bx, bf);
        round_precision(bw, bf);
        x = bx.data();
        w = bw.data();
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, co, ci, 1, x, ci, w, ci, 0, out, co);
    for (size_t i = 0; i < size_t(n) * co; ++i) {
        if (b)
            out[i] += rounded(b[i % co], bf);
        if (bf)
            out[i] = rounded(out[i], bf);
    }
}
Vec Engine::linear(const Vec &x, Weights &w, const std::string &name, int bf) {
    auto s = w.shape(name + ".weight");
    require(s.size() == 2 && x.size() % s[1] == 0, "Invalid linear shape: " + name);
    int n = int(x.size() / s[1]);
    Vec out(size_t(n) * s[0]);
    gemm(out.data(), x.data(), w.get(name + ".weight"),
         w.has(name + ".bias") ? w.get(name + ".bias") : nullptr, n, s[0], s[1], bf);
    return out;
}
void Engine::attention(float *out, const float *q, const float *k, const float *v, int n, int m, int heads,
                       int d) {
    if (gpu_) {
        int rc = api_.attention(gpu_, out, q, k, v, n, m, heads, d);
        require(rc == 0, api_.error(gpu_));
        return;
    }
    // Larger host tiles amortize BLAS dispatch without materializing N x N.
    // The GPU plugin retains its independent, smaller device workspace tiles.
    const int tile = 1024, c = heads * d;
    // Query tiling bounds memory while each query still attends to every key.
    for (int h = 0; h < heads; ++h)
        for (int start = 0; start < n; start += tile) {
            int rows = std::min(tile, n - start);
            Vec scores(size_t(rows) * m);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, m, d, 1 / std::sqrt(float(d)),
                        q + size_t(start) * c + h * d, c, k + h * d, c, 0, scores.data(), m);
#pragma omp parallel for schedule(static) if (size_t(rows) * m >= 65536)
            for (int r = 0; r < rows; ++r) {
                float *s = scores.data() + size_t(r) * m;
                float max = *std::max_element(s, s + m);
                double sum = 0;
                for (int j = 0; j < m; ++j) {
                    s[j] = std::exp(s[j] - max);
                    sum += s[j];
                }
                for (int j = 0; j < m; ++j)
                    s[j] /= float(sum);
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, d, m, 1, scores.data(), m, v + h * d,
                        c, 0, out + size_t(start) * c + h * d, c);
        }
}
void norm(Vec &y, const Vec &x, int c, float eps, const float *w, const float *b) {
    y.resize(x.size());
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < x.size() / c; ++row) {
        double mean = 0, var = 0;
        for (int j = 0; j < c; ++j)
            mean += x[row * c + j];
        mean /= c;
        for (int j = 0; j < c; ++j) {
            double delta = x[row * c + j] - mean;
            var += delta * delta;
        }
        float inv = 1 / std::sqrt(float(var / c) + eps);
        for (int j = 0; j < c; ++j)
            y[row * c + j] = (x[row * c + j] - float(mean)) * inv * (w ? w[j] : 1) + (b ? b[j] : 0);
    }
}
void rms(Vec &x, int heads, int d, const float *gamma) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size() / d; ++i) {
        double s = 0;
        for (int j = 0; j < d; ++j)
            s += double(x[i * d + j]) * x[i * d + j];
        float a = std::sqrt(float(d)) / std::max(std::sqrt(float(s)), 1e-12f);
        for (int j = 0; j < d; ++j)
            x[i * d + j] *= a * gamma[(i % heads) * d + j];
    }
}
void rope(Vec &x, const Coords &coords, int heads, int d) {
    int nf = d / 6, c = heads * d;
    Vec cosine(coords.size() / 4 * 3 * nf), sine(cosine.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < coords.size() / 4; ++i)
        for (int a = 0; a < 3; ++a)
            for (int f = 0; f < nf; ++f) {
                float theta = coords[4 * i + a + 1] * std::pow(10000.f, -float(f) / nf);
                size_t p = i * 3 * nf + a * nf + f;
                cosine[p] = std::cos(theta);
                sine[p] = std::sin(theta);
            }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < coords.size() / 4; ++i)
        for (int h = 0; h < heads; ++h)
            for (int a = 0; a < 3; ++a)
                for (int f = 0; f < nf; ++f) {
                    size_t p = i * 3 * nf + a * nf + f;
                    float co = cosine[p], si = sine[p];
                    size_t j = i * c + h * d + 2 * (a * nf + f);
                    float u = x[j], v = x[j + 1];
                    x[j] = u * co - v * si;
                    x[j + 1] = u * si + v * co;
                }
}
} // namespace px
