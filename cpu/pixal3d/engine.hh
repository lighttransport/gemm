#pragma once
#include "../../common/pixal3d.h"
#include "../../common/pixal3d_device.h"
#include "../../common/safetensors.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace px {
using Vec = std::vector<float>;
using Coords = std::vector<int32_t>;
inline void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
inline float bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) != 0x7f800000u)
        u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    std::memcpy(&x, &u, 4);
    return x;
}
void round_bf16(Vec &x);
inline float fp16(float x) { return float(static_cast<_Float16>(x)); }
inline float rounded(float x, int precision) {
    return precision == 1 ? bf16(x) : precision == 2 ? fp16(x) : x;
}
void round_precision(Vec &x, int precision);
void bias_round(float *x, const float *bias, int rows, int columns, int precision,
                bool bias_is_rounded = false);
void add_residual(Vec &x, const Vec &h, const float *gate, int channels, bool bf);
void apply_modulation(Vec &x, const Vec &mod, int offset, int channels, bool bf);
void gelu(Vec &x, bool approximate, int precision = 0);
struct Weights {
    uint64_t identity;
    st_context *st = nullptr;
    std::map<std::string, Vec> converted;
    explicit Weights(const std::string &path);
    ~Weights();
    Weights(const Weights &) = delete;
    const float *get(const std::string &name);
    int storage_precision(const std::string &name) const;
    std::vector<int> shape(const std::string &name) const;
    bool has(const std::string &name) const;
};

/* Legacy plugin calls exchange host F32 arrays. Optional versioned calls own
 * resident buffers and bounded scratch; no Python or PyTorch is involved. */
struct GpuApi {
    int (*version)() = nullptr;
    int (*configure)(void *, int, int) = nullptr;
    void *(*allocate)(void *, size_t) = nullptr;
    void (*release)(void *) = nullptr;
    int (*copy)(void *, void *, void *, size_t, int) = nullptr;
    int (*execute)(void *, const px_device_command *) = nullptr;
    int (*metrics)(void *, px_device_metrics *) = nullptr;
    int (*trim)(void *) = nullptr;
    void *(*create)(int, size_t) = nullptr;
    void (*destroy)(void *) = nullptr;
    int (*gemm)(void *, float *, const float *, const float *, const float *, int, int, int, int) = nullptr;
    int (*attention)(void *, float *, const float *, const float *, const float *, int, int, int,
                     int) = nullptr;
    size_t (*peak)(void *) = nullptr;
    const char *(*error)(void *) = nullptr;
};
struct Tensor {
    std::shared_ptr<void> memory;
    size_t size = 0;
    int precision = 0;
    void *get() const { return memory.get(); }
};
struct DeviceConditioning {
    int precision = -1;
    Vec global_input, projected_input;
    Coords coords;
    Tensor global, projected, positions, rope_phases;
    Tensor keys[30], values[30];
};
class Engine {
    void *library_ = nullptr, *gpu_ = nullptr;
    GpuApi api_;
    bool resident_ = false;
    pixal3d_gpu_kernels kernels_ = PIXAL3D_KERNEL_AUTO;
    pixal3d_flow_precision flow_precision_ = PIXAL3D_FLOW_BF16;
    uint64_t model_identity_ = 0;
    std::map<std::string, Tensor> weights_;
    std::vector<std::shared_ptr<DeviceConditioning>> conditioning_cache_;
    size_t cache_bytes_ = 0, cache_limit_ = 0, resident_budget_ = 0;
    std::string profile_;
    std::map<std::string, double> timings_;

  public:
    int threads;
    explicit Engine(const pixal3d_options &options);
    ~Engine();
    void configure(const pixal3d_gpu_options &options);
    bool resident() const { return resident_; }
    size_t resident_budget() const { return resident_budget_; }
    pixal3d_flow_precision flow_mode(bool requested) const {
        if (!requested || flow_precision_ == PIXAL3D_FLOW_FP32)
            return PIXAL3D_FLOW_FP32;
        return flow_precision_;
    }
    void bind(Weights &w);
    void clear_weights();
    std::shared_ptr<DeviceConditioning> condition(Weights &w, const Vec &global, const Vec &projected,
                                                  const Coords &coords, int precision);
    Tensor tensor(size_t count, int precision = 0);
    Tensor upload(const void *data, size_t count, int precision = 0);
    Tensor upload(const Vec &v) { return upload(v.data(), v.size()); }
    Vec download(const Tensor &t);
    Tensor weight(Weights &w, const std::string &name, int precision = 0);
    void execute(px_device_command command);
    Tensor linear(const Tensor &x, Weights &w, const std::string &name, int precision = 0);
    Tensor operation(int op, const Tensor &x, int c, int precision = 0, const Tensor &w = {},
                     const Tensor &b = {}, int k = 0, int offset = 0, float epsilon = 0, int extra = 0);
    void inplace(int op, Tensor &x, int c, int precision = 0, const Tensor &w = {}, const Tensor &b = {},
                 int k = 0, int offset = 0, float epsilon = 0, int extra = 0);
    Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, int heads, int hd, int precision);
    Tensor convolution(const Tensor &x, Weights &w, const std::string &name, const Tensor &neighbors,
                       int precision, bool dense = false);
    void record(const std::string &name, double seconds);
    void write_profile();
    void begin_profile();
    void gemm(float *out, const float *x, const float *w, const float *bias, int rows, int outputs,
              int inputs, int precision = 0);
    Vec linear(const Vec &x, Weights &w, const std::string &name, int precision = 0);
    void attention(float *out, const float *q, const float *k, const float *v, int rows, int keys, int heads,
                   int dim);
    size_t peak() const;
};
void norm(Vec &y, const Vec &x, int channels, float eps, const float *weight = nullptr,
          const float *bias = nullptr);
void rms(Vec &x, int heads, int dim, const float *gamma);
void rope(Vec &x, const Coords &coords, int heads, int dim);
Vec flow(Engine &e, Weights &w, const Vec &input, const Coords &coords, float t, const Vec &global,
         const Vec &projected, int blocks, bool bfloat);
Vec flow_resident(Engine &e, Weights &w, const Vec &input, const Coords &coords, float t, const Vec &global,
                  const Vec &projected, int blocks, pixal3d_flow_precision precision);
Vec naf(Engine &e, Weights &w, const Vec &image, int image_size, const Vec &patches, int patch_grid,
        int target, const Vec &xy);
Vec naf_guide(Engine &e, Weights &w, const Vec &image, int size, int target);
Vec naf_sample_gpu(Engine &e, const Vec &guide, int target, const Vec &patches, int grid, const Vec &xy);
Vec naf_sample(const Vec &guide, int target, const Vec &patches, int grid, const Vec &xy);
Vec dino(Engine &e, Weights &w, const Vec &chw, int size, int blocks = 24);
struct Sparse {
    Coords coords;
    Vec feats;
    int channels = 0;
    int rows() const { return int(coords.size() / 4); }
};
struct Subdivision {
    std::vector<int> parents, slots;
    Coords coords;
};
Sparse decode_sparse(Engine &e, Weights &w, const Sparse &input, bool upsample_only,
                     std::vector<Subdivision> &subdivisions, bool guided, int precision = 2,
                     std::vector<Sparse> *subdivision_logits = nullptr);
Vec decode_structure(Engine &e, Weights &w, const Vec &latent, int precision = 2);
} // namespace px
