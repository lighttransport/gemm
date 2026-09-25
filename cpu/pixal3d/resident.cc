/* Resident neural operations; no CUDA/HIP headers in the host library. */
#include "engine.hh"
#include <fstream>
#include <iomanip>

namespace px {
void Engine::configure(const pixal3d_gpu_options &o) {
    require(o.struct_size >= sizeof(o) && o.version == 1, "Unsupported GPU options version");
    require(o.execution >= PIXAL3D_GPU_LEGACY && o.execution <= PIXAL3D_GPU_RESIDENT &&
                o.kernels >= PIXAL3D_KERNEL_AUTO && o.kernels <= PIXAL3D_KERNEL_MMA &&
                o.flow_precision >= PIXAL3D_FLOW_BF16 && o.flow_precision <= PIXAL3D_FLOW_MIXED,
            "Invalid GPU execution options");
    require(!o.execution || (gpu_ && api_.configure), "Resident execution requires a compatible GPU plugin");
    clear_weights();
    const char *commands = std::getenv("PIXAL3D_PROFILE_COMMANDS");
    bool detailed_profile = o.profile_json && *o.profile_json && commands && !std::strcmp(commands, "1");
    if (gpu_ && api_.configure)
        require(api_.configure(gpu_, o.kernels, detailed_profile) == 0, api_.error(gpu_));
    resident_ = o.execution == PIXAL3D_GPU_RESIDENT;
    kernels_ = o.kernels;
    flow_precision_ = o.flow_precision;
    profile_ = o.profile_json ? o.profile_json : "";
}
void Engine::plugin_error() const {
    std::string message = api_.error(gpu_);
    // Budget refusals and device OOM from gpuMalloc (CUDA/HIP code 2) let
    // callers retry with a lower-memory path; other failures stay fatal.
    if (message.find("exceeds memory budget") != std::string::npos ||
        (message.find("gpuMalloc") != std::string::npos && message.find("failed: 2") != std::string::npos))
        throw BudgetError(message);
    throw std::runtime_error(message);
}
void Engine::clear_weights() {
    conditioning_cache_.clear();
    weights_.clear();
    cache_bytes_ = 0;
    model_identity_ = 0;
    if (gpu_ && api_.trim)
        require(api_.trim(gpu_) == 0, api_.error(gpu_));
}
void Engine::bind(Weights &w) {
    if (model_identity_ != w.identity) {
        clear_weights();
        model_identity_ = w.identity;
    }
}
std::shared_ptr<DeviceConditioning> Engine::condition(Weights &w, const Vec &global, const Vec &projected,
                                                      const Coords &coords, int precision) {
    bind(w);
    for (auto &cached : conditioning_cache_)
        if (cached->precision == precision && cached->global_input == global &&
            cached->projected_input == projected && cached->coords == coords)
            return cached;
    // At most positive and negative guidance conditions for this model. Exact
    // value comparisons prevent stale hits when a caller reuses host addresses.
    if (conditioning_cache_.size() == 2)
        conditioning_cache_.erase(conditioning_cache_.begin());
    auto cached = std::make_shared<DeviceConditioning>();
    cached->precision = precision;
    cached->global_input = global;
    cached->projected_input = projected;
    cached->coords = coords;
    cached->global = upload(global);
    cached->projected = upload(projected);
    cached->positions = upload(coords.data(), coords.size());
    int heads = w.shape("blocks.0.self_attn.q_rms_norm.gamma")[0];
    int hd = w.shape("input_layer.weight")[0] / heads;
    cached->rope_phases = tensor(coords.size() / 4 * hd);
    execute({PX_ROPE_PHASE, 0, int(coords.size() / 4), hd, 0, 0, 0, 0, 0, cached->rope_phases.get(), nullptr,
             cached->positions.get(), nullptr, nullptr});
    inplace(PX_ROUND, cached->global, 1, precision);
    inplace(PX_ROUND, cached->projected, 1, precision);
    conditioning_cache_.push_back(cached);
    return cached;
}
Tensor Engine::tensor(size_t count, int precision) {
    require(resident_ && count && count <= SIZE_MAX / 4 && precision >= 0 && precision <= 2,
            "Invalid resident tensor allocation");
    size_t bytes = count * (precision ? 2 : 4);
    void *p = api_.allocate(gpu_, bytes);
    if (!p && !weights_.empty()) {
        // Cached tensors currently referenced by a command remain alive.
        weights_.clear();
        cache_bytes_ = 0;
        p = api_.allocate(gpu_, bytes);
    }
    if (!p)
        plugin_error();
    auto release = api_.release;
    return {std::shared_ptr<void>(p, [release](void *q) { release(q); }), count, precision};
}
Tensor Engine::upload(const void *data, size_t count, int precision) {
    auto t = tensor(count, precision);
    require(api_.copy(gpu_, t.get(), const_cast<void *>(data), count * (precision ? 2 : 4), 0) == 0,
            api_.error(gpu_));
    return t;
}
Vec Engine::download(const Tensor &t) {
    require(t.precision == 0, "Host download requires F32 tensor");
    Vec v(t.size);
    require(api_.copy(gpu_, t.get(), v.data(), v.size() * 4, 1) == 0, api_.error(gpu_));
    return v;
}
Tensor Engine::weight(Weights &w, const std::string &name, int precision) {
    if (!w.has(name))
        return {};
    bind(w);
    int storage_precision = precision == 3 ? 1 : precision;
    std::string key = name + ":" + std::to_string(storage_precision);
    auto it = weights_.find(key);
    if (it != weights_.end())
        return it->second;
    size_t n = 1;
    for (int d : w.shape(name)) {
        require(n <= SIZE_MAX / size_t(d), "Weight size overflow");
        n *= d;
    }
    int index = safetensors_find(w.st, name.c_str());
    const char *dtype = safetensors_dtype(w.st, index);
    Tensor t;
    if (storage_precision && !std::strcmp(dtype, storage_precision == 1 ? "BF16" : "F16")) {
        require(safetensors_nbytes(w.st, index) == n * 2, "Invalid packed weight size: " + name);
        t = upload(safetensors_data(w.st, index), n, storage_precision);
    } else if (storage_precision) {
        const float *v = w.get(name);
        std::vector<uint16_t> packed(n);
        for (size_t i = 0; i < n; ++i) {
            if (storage_precision == 1) {
                float f = bf16(v[i]);
                uint32_t u;
                std::memcpy(&u, &f, 4);
                packed[i] = u >> 16;
            } else {
                _Float16 h = static_cast<_Float16>(v[i]);
                std::memcpy(&packed[i], &h, 2);
            }
        }
        t = upload(packed.data(), n, storage_precision);
    } else
        t = upload(w.get(name), n);
    size_t bytes = n * (storage_precision ? 2 : 4);
    if (cache_bytes_ + bytes <= cache_limit_) {
        weights_[key] = t;
        cache_bytes_ += bytes;
    }
    return t;
}
void Engine::execute(px_device_command op) {
    if (api_.execute(gpu_, &op) != 0)
        plugin_error();
}
Tensor Engine::linear(const Tensor &x, Weights &w, const std::string &name, int precision) {
    auto shape = w.shape(name + ".weight");
    require(shape.size() == 2 && x.size % shape[1] == 0 && x.size / shape[1] <= INT32_MAX,
            "Invalid resident linear");
    auto weight_tensor = weight(w, name + ".weight", precision), bias = weight(w, name + ".bias");
    auto y = tensor(x.size / shape[1] * shape[0]);
    px_device_command op{};
    op.op = PX_LINEAR;
    op.precision = precision;
    op.n = x.size / shape[1];
    op.c = shape[0];
    op.k = shape[1];
    op.out = y.get();
    op.x = x.get();
    op.w = weight_tensor.get();
    op.b = bias.get();
    execute(op);
    return y;
}
Tensor Engine::operation(int op, const Tensor &x, int c, int precision, const Tensor &w, const Tensor &b,
                         int k, int offset, float epsilon, int extra) {
    require(c > 0 && x.size % c == 0 && x.size / c <= INT32_MAX, "Invalid resident operation dimensions");
    if (op == PX_PART)
        require(k > 0 && x.size / c % k == 0 && offset >= 0 && offset < k, "Invalid resident slice");
    size_t size = op == PX_PART ? x.size / size_t(k) : x.size;
    auto y = tensor(size);
    px_device_command command{op,    precision, int(size / c), c,       k,       0,       offset,
                              extra, epsilon,   y.get(),       x.get(), w.get(), b.get(), nullptr};
    execute(command);
    return y;
}
void Engine::inplace(int op, Tensor &x, int c, int precision, const Tensor &w, const Tensor &b, int k,
                     int offset, float epsilon, int extra) {
    require(c > 0 && x.size % c == 0 && x.size / c <= INT32_MAX, "Invalid resident operation dimensions");
    execute({op, precision, int(x.size / c), c, k, 0, offset, extra, epsilon, x.get(), x.get(), w.get(),
             b.get(), nullptr});
}
Tensor Engine::attention(const Tensor &q, const Tensor &k, const Tensor &v, int heads, int hd,
                         int precision) {
    auto y = tensor(q.size);
    require(heads > 0 && hd > 0 && int64_t(heads) * hd <= INT32_MAX, "Invalid attention head size");
    int c = heads * hd;
    require(q.size % c == 0 && k.size % c == 0 && k.size == v.size && q.size / c <= INT32_MAX &&
                k.size / c <= INT32_MAX,
            "Invalid attention tensors");
    execute({PX_ATTENTION, precision, int(q.size / c), hd, int(k.size / c), heads, 0, 0, 0, y.get(), q.get(),
             k.get(), nullptr, v.get()});
    return y;
}
void Engine::record(const std::string &name, double seconds) { timings_[name] += seconds; }
Tensor Engine::convolution(const Tensor &x, Weights &w, const std::string &name, const Tensor &neighbors,
                           int precision, bool dense) {
    auto s = w.shape(name + ".weight");
    require(s.size() == 5 && s[dense ? 2 : 1] == 3 && s[dense ? 3 : 2] == 3 && s[dense ? 4 : 3] == 3,
            "Expected 3x3x3 resident convolution");
    int ci = dense ? s[1] : s.back(), co = s[0], n = int(x.size / ci);
    require(x.size % ci == 0 && neighbors.size == size_t(n) * 27, "Invalid resident convolution geometry");
    auto wt = weight(w, name + ".weight", precision), bias = weight(w, name + ".bias");
    auto out = tensor(size_t(n) * co);
    for (int start = 0; start < n; start += 2048) {
        int rows = std::min(2048, n - start);
        auto gather = tensor(size_t(rows) * ci * 27, precision);
        execute({PX_GATHER, precision, rows, ci * 27, 0, 0, start, int(dense), 0, gather.get(), x.get(),
                 neighbors.get(), nullptr, nullptr});
        execute({PX_LINEAR, precision, rows, co, ci * 27, 0, start, int(precision != 0), 0, out.get(), gather.get(), wt.get(),
                 bias.get(), nullptr});
    }
    return out;
}
void Engine::convolution_tiles(const Tensor &x, Weights &w, const std::string &name, const Tensor &neighbors,
                               int precision, const std::function<void(const Tensor &, int, int)> &consume) {
    auto s = w.shape(name + ".weight");
    require(s.size() == 5 && s[1] == 3 && s[2] == 3 && s[3] == 3, "Expected sparse 3x3x3 resident convolution");
    int ci = s.back(), co = s[0], n = int(x.size / ci);
    require(x.size % ci == 0 && neighbors.size == size_t(n) * 27, "Invalid resident convolution geometry");
    auto wt = weight(w, name + ".weight", precision), bias = weight(w, name + ".bias");
    auto out = tensor(size_t(std::min(n, 2048)) * co);
    for (int start = 0; start < n; start += 2048) {
        int rows = std::min(2048, n - start);
        auto gather = tensor(size_t(rows) * ci * 27, precision);
        execute({PX_GATHER, precision, rows, ci * 27, 0, 0, start, 0, 0, gather.get(), x.get(), neighbors.get(),
                 nullptr, nullptr});
        execute({PX_LINEAR, precision, rows, co, ci * 27, 0, 0, int(precision != 0), 0, out.get(), gather.get(),
                 wt.get(), bias.get(), nullptr});
        consume(out, start, rows);
    }
}
void Engine::begin_profile() {
    timings_.clear();
    if (gpu_ && api_.metrics)
        require(api_.metrics(gpu_, nullptr) == 0, api_.error(gpu_));
}
void Engine::write_profile() {
    if (profile_.empty())
        return;
    px_device_metrics m{};
    if (gpu_ && api_.metrics)
        require(api_.metrics(gpu_, &m) == 0, api_.error(gpu_));
    std::ofstream f(profile_);
    require(bool(f), "Cannot write GPU profile: " + profile_);
    f << "{\"execution\":\"" << (resident_ ? "resident" : "legacy") << "\",\"kernels\":\""
      << (kernels_ == PIXAL3D_KERNEL_AUTO   ? "auto"
          : kernels_ == PIXAL3D_KERNEL_BLAS ? "blas"
                                            : "mma")
      << "\",\"timings_seconds\":{";
    bool first = true;
    for (const auto &item : timings_) {
        if (!first)
            f << ',';
        first = false;
        f << std::quoted(item.first) << ':' << item.second;
    }
    f << "},\"h2d_bytes\":" << m.uploads << ",\"d2h_bytes\":" << m.downloads
      << ",\"allocations\":" << m.allocations << ",\"gemms\":" << m.gemms << ",\"mma_gemms\":" << m.mma_gemms
      << ",\"attentions\":" << m.attentions << ",\"mma_attentions\":" << m.mma_attentions
      << ",\"resident_command_ms\":" << m.kernel_ms << ",\"effective_budget_bytes\":"
      << m.effective_budget_bytes << ",\"active_device_bytes\":" << m.active_bytes
      << ",\"pooled_device_bytes\":" << m.pooled_bytes << ",\"peak_active_device_bytes\":"
      << m.peak_active_bytes << ",\"largest_allocation_bytes\":" << m.largest_allocation_bytes
      << ",\"peak_reserved_device_bytes\":" << peak() << "}\n";
    require(bool(f), "Failed writing GPU profile");
}

Vec flow_resident(Engine &e, Weights &w, const Vec &input, const Coords &coords, float t,
                  const Vec &global_input, const Vec &projected_input, int blocks,
                  pixal3d_flow_precision precision) {
    e.bind(w);
    bool bf = precision == PIXAL3D_FLOW_BF16;
    bool mixed = precision == PIXAL3D_FLOW_MIXED;
    int linear_precision = mixed ? 3 : (bf ? 1 : 0);
    int op_precision = bf ? 1 : 0;
    int attention_precision = mixed ? 1 : op_precision;
    int c = w.shape("input_layer.weight")[0], heads = w.shape("blocks.0.self_attn.q_rms_norm.gamma")[0];
    require(heads > 0 && c % heads == 0 && coords.size() % 4 == 0 &&
                input.size() == coords.size() / 4 * size_t(w.shape("input_layer.weight")[1]) &&
                global_input.size() ==
                    size_t(5) * w.shape("blocks.0.cross_attn.cross_attn_block.to_kv.weight")[1] &&
                projected_input.size() ==
                    coords.size() / 4 * size_t(w.shape("blocks.0.cross_attn.proj_linear.weight")[1]) &&
                blocks > 0 && blocks <= 30,
            "Invalid resident flow geometry");
    int hd = c / heads;
    Vec time(256);
    for (int j = 0; j < 128; ++j) {
        float phase = t * 1000.f * std::exp(-std::log(10000.f) * j / 128.f);
        time[j] = std::cos(phase);
        time[128 + j] = std::sin(phase);
    }
    auto tm = e.linear(e.upload(time), w, "t_embedder.mlp.0");
    e.inplace(PX_SILU, tm, c);
    tm = e.linear(tm, w, "t_embedder.mlp.2");
    e.inplace(PX_SILU, tm, c);
    // Boundary layers stay FP32 as upstream and the CPU path do; the BF16 mode
    // then rounds their outputs like upstream's manual_cast.
    auto modulation = e.linear(tm, w, "adaLN_modulation.1");
    auto hidden = e.linear(e.upload(input), w, "input_layer");
    for (Tensor *x : {&modulation, &hidden})
        e.inplace(PX_ROUND, *x, 1, bf);
    auto cached = e.condition(w, global_input, projected_input, coords, bf ? 1 : 0);
    auto global = cached->global, projected = cached->projected, phases = cached->rope_phases;
    for (int i = 0; i < blocks; ++i) {
        std::string b = "blocks." + std::to_string(i) + ".", ca = b + "cross_attn.cross_attn_block.";
        auto mod = e.operation(PX_ADD, modulation, 6 * c, op_precision, e.weight(w, b + "modulation"));
        auto h = e.operation(PX_NORM, hidden, c, 0, {}, {}, 0, 0, 1e-6f);
        e.inplace(PX_MODULATE, h, c, op_precision, mod);
        auto qkv = e.linear(h, w, b + "self_attn.to_qkv", linear_precision);
        auto q = e.operation(PX_PART, qkv, c, 0, {}, {}, 3, 0);
        auto k = e.operation(PX_PART, qkv, c, 0, {}, {}, 3, 1);
        auto v = e.operation(PX_PART, qkv, c, 0, {}, {}, 3, 2);
        qkv = {};
        e.inplace(PX_RMS, q, hd, op_precision, e.weight(w, b + "self_attn.q_rms_norm.gamma"), {}, heads);
        e.inplace(PX_RMS, k, hd, op_precision, e.weight(w, b + "self_attn.k_rms_norm.gamma"), {}, heads);
        e.inplace(PX_ROPE, q, hd, op_precision, phases, {}, heads, 0, 0, 1);
        e.inplace(PX_ROPE, k, hd, op_precision, phases, {}, heads, 0, 0, 1);
        h = e.attention(q, k, v, heads, hd, attention_precision);
        h = e.linear(h, w, b + "self_attn.to_out", linear_precision);
        e.inplace(PX_RESIDUAL, hidden, c, op_precision, h, mod, 0, 2 * c);
        h = e.operation(PX_NORM, hidden, c, bf, e.weight(w, b + "norm2.weight"),
                        e.weight(w, b + "norm2.bias"), 0, 0, 1e-6f);
        q = e.linear(h, w, ca + "to_q", linear_precision);
        if (!cached->keys[i].get()) {
            auto kv = e.linear(global, w, ca + "to_kv", linear_precision);
            cached->keys[i] = e.operation(PX_PART, kv, c, 0, {}, {}, 2, 0);
            cached->values[i] = e.operation(PX_PART, kv, c, 0, {}, {}, 2, 1);
            e.inplace(PX_RMS, cached->keys[i], hd, op_precision, e.weight(w, ca + "k_rms_norm.gamma"), {}, heads);
        }
        k = cached->keys[i];
        v = cached->values[i];
        e.inplace(PX_RMS, q, hd, op_precision, e.weight(w, ca + "q_rms_norm.gamma"), {}, heads);
        h = e.attention(q, k, v, heads, hd, attention_precision);
        h = e.linear(h, w, ca + "to_out", linear_precision);
        auto projection = e.linear(projected, w, b + "cross_attn.proj_linear", linear_precision);
        e.inplace(PX_ADD, h, c, op_precision, projection, {}, 1);
        e.inplace(PX_RESIDUAL, hidden, c, op_precision, h);
        h = e.operation(PX_NORM, hidden, c, 0, {}, {}, 0, 0, 1e-6f);
        e.inplace(PX_MODULATE, h, c, op_precision, mod, {}, 0, 3 * c);
        h = e.linear(h, w, b + "mlp.mlp.0", linear_precision);
        e.inplace(PX_GELU, h, 1, op_precision, {}, {}, 1);
        h = e.linear(h, w, b + "mlp.mlp.2", linear_precision);
        e.inplace(PX_RESIDUAL, hidden, c, op_precision, h, mod, 0, 5 * c);
    }
    auto h = e.operation(PX_NORM, hidden, c, 0, {}, {}, 0, 0, 1e-5f);
    return e.download(e.linear(h, w, "out_layer"));
}
} // namespace px
