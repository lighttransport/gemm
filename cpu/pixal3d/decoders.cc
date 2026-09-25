#include "engine.hh"
#include <cstdio>
#include <unordered_map>

namespace px {
static uint64_t key(int x, int y, int z) {
    return uint64_t(x) | (uint64_t(y) << 21) | (uint64_t(z) << 42);
}
static std::vector<int> neighbors(const Coords &coords) {
    int n = int(coords.size() / 4);
    std::unordered_map<uint64_t, int> map;
    map.reserve(n * 2);
    for (int i = 0; i < n; ++i)
        map.emplace(key(coords[4 * i + 1], coords[4 * i + 2], coords[4 * i + 3]), i);
    std::vector<int> out(size_t(n) * 27, -1);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        for (int dx = 0; dx < 3; ++dx)
            for (int dy = 0; dy < 3; ++dy)
                for (int dz = 0; dz < 3; ++dz) {
                    int x = coords[4 * i + 1] + dx - 1, y = coords[4 * i + 2] + dy - 1,
                        z = coords[4 * i + 3] + dz - 1;
                    if (x < 0 || y < 0 || z < 0)
                        continue;
                    auto it = map.find(key(x, y, z));
                    if (it != map.end())
                        out[size_t(i) * 27 + dx * 9 + dy * 3 + dz] = it->second;
                }
    return out;
}
static Vec convolution(Engine &e, Weights &w, const Vec &x, const std::vector<int> &nbr,
                       const std::string &name, int precision, bool dense = false) {
    auto shape = w.shape(name + ".weight");
    int co = shape[0], ci = dense ? shape[1] : shape.back();
    require(shape.size() == 5 && shape[dense ? 2 : 1] == 3, "Expected 3x3x3 decoder convolution");
    int n = int(nbr.size() / 27);
    require(x.size() == size_t(n) * ci, "Invalid convolution input: " + name);
    const float *weight = w.get(name + ".weight");
    if (dense) {
        auto &converted = w.converted[name + ".ndhwc"];
        if (converted.empty()) {
            converted.resize(size_t(co) * 27 * ci);
            for (int o = 0; o < co; ++o)
                for (int k = 0; k < 27; ++k)
                    for (int c = 0; c < ci; ++c)
                        converted[(size_t(o) * 27 + k) * ci + c] = weight[(size_t(o) * ci + c) * 27 + k];
        }
        weight = converted.data();
    }
    const int tile = 2048;
    Vec gather(size_t(std::min(tile, n)) * 27 * ci), out(size_t(n) * co);
    for (int start = 0; start < n; start += tile) {
        int rows = std::min(tile, n - start);
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; ++r)
            for (int k = 0; k < 27; ++k) {
                int i = nbr[size_t(start + r) * 27 + k];
                float *dst = gather.data() + (size_t(r) * 27 + k) * ci;
                if (i >= 0)
                    std::copy_n(x.data() + size_t(i) * ci, ci, dst);
                else
                    std::fill_n(dst, ci, 0.f);
            }
        e.gemm(out.data() + size_t(start) * co, gather.data(), weight, w.get(name + ".bias"), rows, co,
               27 * ci, precision);
    }
    return out;
}
static void activate(Vec &x, int precision) {
#pragma omp parallel for schedule(static) if (x.size() > 16384)
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = rounded(x[i] / (1 + std::exp(-x[i])), precision);
}
static void layer_norm(Vec &out, const Vec &x, int c, Weights &w, const std::string &name, int precision,
                       float eps = 1e-6f) {
    norm(out, x, c, eps, w.has(name + ".weight") ? w.get(name + ".weight") : nullptr,
         w.has(name + ".bias") ? w.get(name + ".bias") : nullptr);
    round_precision(out, precision);
}
static Sparse decode_sparse_gpu(Engine &e, Weights &w, const Sparse &input, bool upsample_only,
                                std::vector<Subdivision> &subdivisions, bool guided, int precision,
                                std::vector<Sparse> *subdivision_logits) {
    e.bind(w);
    Coords coords = input.coords;
    int channels = w.shape("from_latent.weight")[0];
    auto h = e.linear(e.upload(input.feats), w, "from_latent");
    e.inplace(PX_ROUND, h, 1, precision);
    if (!guided)
        subdivisions.clear();
    // px_norm reads each element and writes it from the same thread after the
    // row reduction, so normalizing in place is exact and saves one activation.
    auto norm_gpu = [&](Tensor &x, int c, const std::string &name) {
        e.inplace(PX_NORM, x, c, precision, e.weight(w, name + ".weight"), e.weight(w, name + ".bias"), 0, 0,
                  1e-6f);
    };
    // The neighbor map of `coords`; the previous stage's conv2 map is reused.
    Tensor nbr;
    auto ensure_neighbors = [&] {
        if (!nbr.get()) {
            auto map = neighbors(coords);
            nbr = e.upload(map.data(), map.size());
        }
    };
    // ConvNeXt MLP over row chunks. Chunks are multiples of the 2048-row GEMM
    // tile, so every GEMM and rounding step matches the unchunked sequence;
    // only the 4x-wide hidden activation shrinks to one chunk.
    auto mlp = [&](const Tensor &y, int c, const std::string &b) {
        int n = int(y.size / c), wide = w.shape(b + "mlp.0.weight")[0];
        int chunk = std::max(2048, int((size_t(128) << 20) / (size_t(wide) * 4) / 2048 * 2048));
        auto w0 = e.weight(w, b + "mlp.0.weight", precision), b0 = e.weight(w, b + "mlp.0.bias");
        auto w2 = e.weight(w, b + "mlp.2.weight", precision), b2 = e.weight(w, b + "mlp.2.bias");
        int most = std::min(n, chunk);
        auto a = e.tensor(size_t(most) * wide), o = e.tensor(size_t(most) * c);
        for (int start = 0; start < n; start += chunk) {
            int rows = std::min(chunk, n - start);
            e.execute({PX_LINEAR, precision, rows, wide, c, start, 0, 0, 0, a.get(), y.get(), w0.get(), b0.get(),
                       nullptr});
            e.execute({PX_SILU, precision, rows * wide, 1, 0, 0, 0, 0, 0, a.get(), a.get(), nullptr, nullptr,
                       nullptr});
            e.execute({PX_LINEAR, precision, rows, c, wide, 0, 0, 0, 0, o.get(), a.get(), w2.get(), b2.get(),
                       nullptr});
            // h + mlp(y); ADD offsets out and w, and IEEE addition commutes.
            e.execute({PX_ADD, precision, rows, c, 1, start, 0, 0, 0, h.get(), o.get(), h.get(), nullptr,
                       nullptr});
        }
    };
    for (int stage = 0; stage < 5; ++stage) {
        if (stage == 4 && upsample_only)
            return {coords, e.download(h), channels};
        std::string prefix = "blocks." + std::to_string(stage) + ".";
        int block = 0;
        while (w.has(prefix + std::to_string(block) + ".conv.weight")) {
            ensure_neighbors();
            std::string b = prefix + std::to_string(block) + ".";
            auto y = e.convolution(h, w, b + "conv", nbr, precision);
            norm_gpu(y, channels, b + "norm");
            mlp(y, channels, b);
            ++block;
        }
        if (stage == 4)
            break;
        std::string b = prefix + std::to_string(block) + ".";
        int ci = channels, co = w.shape(b + "conv2.weight")[0];
        require(ci % 8 == 0 && co % (ci / 8) == 0, "Invalid resident subdivision ratio");
        Subdivision sub;
        if (!guided || subdivision_logits) {
            auto logits = e.download(e.linear(h, w, b + "to_subdiv", precision));
            if (subdivision_logits)
                subdivision_logits->push_back({coords, logits, 8});
            if (!guided) {
                for (size_t p = 0; p < coords.size() / 4; ++p)
                    for (int s = 0; s < 8; ++s)
                        if (logits[p * 8 + s] > 0) {
                            sub.parents.push_back(p);
                            sub.slots.push_back(s);
                            sub.coords.push_back(0);
                            for (int a = 0; a < 3; ++a)
                                sub.coords.push_back(coords[p * 4 + a + 1] * 2 + ((s >> a) & 1));
                        }
                subdivisions.push_back(sub);
            }
        }
        if (guided) {
            require(subdivisions.size() == 4, "Decoder requires four subdivisions");
            sub = subdivisions[stage];
        }
        require(!sub.parents.empty(), "Resident decoder predicted no occupied voxels");
        for (size_t r = 0; r < sub.parents.size(); ++r)
            require(sub.parents[r] >= 0 && size_t(sub.parents[r]) < coords.size() / 4 && sub.slots[r] >= 0 &&
                        sub.slots[r] < 8,
                    "Invalid subdivision guide");
        for (size_t r = 1; r < sub.parents.size(); ++r)
            require(sub.parents[r - 1] <= sub.parents[r], "Subdivision parents must be ascending");
        auto parents = e.upload(sub.parents.data(), sub.parents.size()),
             slots = e.upload(sub.slots.data(), sub.slots.size());
        ensure_neighbors();
        auto x = e.operation(PX_NORM, h, ci, precision, e.weight(w, b + "norm1.weight"),
                             e.weight(w, b + "norm1.bias"), 0, 0, 1e-6f);
        e.inplace(PX_SILU, x, 1, precision);
        // conv1 has 8x the output channels; scatter each 2048-parent tile to
        // its children instead of storing the whole conv1 output. Children of
        // a tile are contiguous because parents are ascending.
        int rows = int(sub.parents.size());
        auto fine = e.tensor(size_t(rows) * co);
        size_t child = 0;
        e.convolution_tiles(x, w, b + "conv1", nbr, precision, [&](const Tensor &tile, int start, int count) {
            size_t end = child;
            while (end < sub.parents.size() && sub.parents[end] < start + count)
                ++end;
            if (end > child)
                e.execute({PX_C2S, 0, int(end - child), co, ci, int(child), start, 0, 0, fine.get(), tile.get(),
                           parents.get(), slots.get(), nullptr});
            child = end;
        });
        require(child == sub.parents.size(), "Subdivision parents exceed decoder rows");
        x = {};
        nbr = {};
        auto skip = e.tensor(size_t(rows) * co);
        e.execute(
            {PX_SKIP, 0, rows, co, ci, 0, 0, 0, 0, skip.get(), h.get(), parents.get(), slots.get(), nullptr});
        h = {};
        parents = {};
        slots = {};
        norm_gpu(fine, co, b + "norm2");
        e.inplace(PX_SILU, fine, 1, precision);
        coords = std::move(sub.coords);
        ensure_neighbors();
        // conv2 + skip, accumulated into skip one tile at a time.
        e.convolution_tiles(fine, w, b + "conv2", nbr, precision, [&](const Tensor &tile, int start, int count) {
            e.execute({PX_ADD, precision, count, co, 1, start, 0, 0, 0, skip.get(), tile.get(), skip.get(),
                       nullptr, nullptr});
        });
        fine = {};
        h = std::move(skip);
        channels = co;
        std::fprintf(stderr, "Pixal3D resident decoder stage %d: %zu voxels\n", stage, coords.size() / 4);
    }
    e.inplace(PX_NORM, h, channels, 0, {}, {}, 0, 0, 1e-5f);
    h = e.linear(h, w, "output_layer");
    return {coords, e.download(h), w.shape("output_layer.weight")[0]};
}
Sparse decode_sparse(Engine &e, Weights &w, const Sparse &input, bool upsample_only,
                     std::vector<Subdivision> &subdivisions, bool guided, int precision,
                     std::vector<Sparse> *subdivision_logits) {
    // The resident decoder streams its widest activations in tiles and fits
    // dense four-view shapes in the 7 GiB budget. If an allocation still
    // exceeds the budget or device memory, release it and rerun on the tiled
    // host-offloaded decoder, which needs far less device memory but is much
    // slower and rounds some steps on the host.
    if (e.resident()) {
        size_t logits = subdivision_logits ? subdivision_logits->size() : 0;
        try {
            return decode_sparse_gpu(e, w, input, upsample_only, subdivisions, guided, precision,
                                     subdivision_logits);
        } catch (const BudgetError &error) {
            std::fprintf(stderr, "Pixal3D decoder: resident path exceeded device memory (%s); "
                                 "using the tiled low-memory path for %d input voxels\n",
                         error.what(), input.rows());
            if (subdivision_logits)
                subdivision_logits->resize(logits);
            e.clear_weights();
        }
    }
    Sparse h{input.coords, e.linear(input.feats, w, "from_latent"), w.shape("from_latent.weight")[0]};
    round_precision(h.feats, precision);
    if (!guided)
        subdivisions.clear();
    for (int stage = 0; stage < 5; ++stage) {
        if (stage == 4 && upsample_only)
            return h;
        auto nbr = neighbors(h.coords);
        int block = 0;
        std::string prefix = "blocks." + std::to_string(stage) + ".";
        while (w.has(prefix + std::to_string(block) + ".conv.weight")) {
            std::string b = prefix + std::to_string(block) + ".";
            Vec x = convolution(e, w, h.feats, nbr, b + "conv", precision), y;
            layer_norm(y, x, h.channels, w, b + "norm", precision);
            y = e.linear(y, w, b + "mlp.0", precision);
            activate(y, precision);
            y = e.linear(y, w, b + "mlp.2", precision);
            for (size_t i = 0; i < y.size(); ++i)
                h.feats[i] = rounded(h.feats[i] + y[i], precision);
            ++block;
        }
        if (stage == 4)
            break;
        std::string b = prefix + std::to_string(block) + ".";
        Subdivision sub;
        int ci = h.channels, co = w.shape(b + "conv2.weight")[0];
        require(ci % 8 == 0 && co % (ci / 8) == 0, "Invalid channel-to-spatial ratio");
        Vec logits;
        if (!guided || subdivision_logits)
            logits = e.linear(h.feats, w, b + "to_subdiv", precision);
        if (subdivision_logits)
            subdivision_logits->push_back({h.coords, logits, 8});
        if (guided) {
            require(subdivisions.size() == 4, "Decoder requires four subdivisions");
            sub = subdivisions[stage];
        } else {
            for (int p = 0; p < h.rows(); ++p)
                for (int s = 0; s < 8; ++s)
                    if (logits[size_t(p) * 8 + s] > 0) {
                        sub.parents.push_back(p);
                        sub.slots.push_back(s);
                        sub.coords.push_back(0);
                        for (int a = 0; a < 3; ++a)
                            sub.coords.push_back(h.coords[4 * p + a + 1] * 2 + ((s >> a) & 1));
                    }
            subdivisions.push_back(sub);
        }
        require(!sub.parents.empty(),
                "Decoder predicted no occupied voxels at stage " + std::to_string(stage));
        Vec x;
        layer_norm(x, h.feats, ci, w, b + "norm1", precision);
        activate(x, precision);
        x = convolution(e, w, x, nbr, b + "conv1", precision);
        Vec fine(sub.parents.size() * co), skip(fine.size());
        for (size_t r = 0; r < sub.parents.size(); ++r) {
            int p = sub.parents[r], s = sub.slots[r];
            require(p >= 0 && p < h.rows() && s >= 0 && s < 8, "Invalid subdivision guide");
            std::copy_n(x.data() + (size_t(p) * 8 + s) * co, co, fine.data() + r * co);
            for (int c = 0; c < co; ++c)
                skip[r * co + c] = h.feats[size_t(p) * ci + s * (ci / 8) + c / (co / (ci / 8))];
        }
        layer_norm(x, fine, co, w, b + "norm2", precision);
        activate(x, precision);
        nbr = neighbors(sub.coords);
        fine = convolution(e, w, x, nbr, b + "conv2", precision);
        for (size_t i = 0; i < fine.size(); ++i)
            fine[i] = rounded(fine[i] + skip[i], precision);
        h = {std::move(sub.coords), std::move(fine), co};
        std::fprintf(stderr, "Pixal3D decoder stage %d: %d voxels\n", stage, h.rows());
    }
    Vec x;
    norm(x, h.feats, h.channels, 1e-5f);
    h.feats = e.linear(x, w, "output_layer");
    h.channels = w.shape("output_layer.weight")[0];
    return h;
}
static std::vector<int> dense_neighbors(int size) {
    std::vector<int> out(size_t(size) * size * size * 27, -1);
#pragma omp parallel for schedule(static)
    for (int p = 0; p < size * size * size; ++p)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int c = 0; c < 3; ++c) {
                    int x = p / (size * size) + a - 1, y = p / size % size + b - 1, z = p % size + c - 1;
                    if (x >= 0 && x < size && y >= 0 && y < size && z >= 0 && z < size)
                        out[size_t(p) * 27 + a * 9 + b * 3 + c] = (x * size + y) * size + z;
                }
    return out;
}
static Vec decode_structure_gpu(Engine &e, Weights &w, const Vec &latent, int precision) {
    e.bind(w);
    int size = 16, c = 512;
    auto map = dense_neighbors(size);
    auto nbr = e.upload(map.data(), map.size());
    auto h = e.convolution(e.upload(latent), w, "input_layer", nbr, 0, true);
    e.inplace(PX_ROUND, h, 1, precision);
    auto norm_gpu = [&](const Tensor &x, const std::string &name, int p) {
        return e.operation(PX_NORM, x, c, p, e.weight(w, name + ".weight"), e.weight(w, name + ".bias"), 0, 0,
                           1e-5f);
    };
    auto resblock = [&](const std::string &b) {
        auto x = norm_gpu(h, b + "norm1", precision);
        e.inplace(PX_SILU, x, 1, precision);
        x = e.convolution(x, w, b + "conv1", nbr, precision, true);
        auto y = norm_gpu(x, b + "norm2", precision);
        e.inplace(PX_SILU, y, 1, precision);
        y = e.convolution(y, w, b + "conv2", nbr, precision, true);
        e.inplace(PX_ADD, h, c, precision, y, {}, 1);
    };
    resblock("middle_block.0.");
    resblock("middle_block.1.");
    for (int block = 0; block < 8; ++block) {
        std::string b = "blocks." + std::to_string(block) + ".";
        if (w.has(b + "conv1.weight")) {
            resblock(b);
            continue;
        }
        auto x = e.convolution(h, w, b + "conv", nbr, precision, true);
        c = w.shape(b + "conv.weight")[0] / 8;
        int next = size * 2, rows = next * next * next;
        std::vector<int> parents(rows), slots(rows);
        for (int p = 0; p < size * size * size; ++p)
            for (int slot = 0; slot < 8; ++slot) {
                int a = p / (size * size) * 2 + (slot >> 2), b = p / size % size * 2 + ((slot >> 1) & 1),
                    z = p % size * 2 + (slot & 1);
                int r = (a * next + b) * next + z;
                parents[r] = p;
                slots[r] = slot;
            }
        auto dp = e.upload(parents.data(), parents.size()), ds = e.upload(slots.data(), slots.size());
        h = e.tensor(size_t(rows) * c);
        e.execute({PX_C2S, 0, rows, c, 0, 0, 0, 1, 0, h.get(), x.get(), dp.get(), ds.get(), nullptr});
        size = next;
        map = dense_neighbors(size);
        nbr = e.upload(map.data(), map.size());
    }
    auto x = norm_gpu(h, "out_layer.0", 0);
    e.inplace(PX_SILU, x, 1, 0);
    return e.download(e.convolution(x, w, "out_layer.2", nbr, 0, true));
}
Vec decode_structure(Engine &e, Weights &w, const Vec &latent, int precision) {
    require(latent.size() == 4096 * 8, "Expected 16-cubed structure latent");
    if (e.resident())
        return decode_structure_gpu(e, w, latent, precision);
    int size = 16, c = 512;
    auto nbr = dense_neighbors(size);
    Vec h = convolution(e, w, latent, nbr, "input_layer", 0, true);
    round_precision(h, precision);
    auto resblock = [&](const std::string &b) {
        Vec x;
        layer_norm(x, h, c, w, b + "norm1", precision, 1e-5f);
        activate(x, precision);
        x = convolution(e, w, x, nbr, b + "conv1", precision, true);
        Vec y;
        layer_norm(y, x, c, w, b + "norm2", precision, 1e-5f);
        activate(y, precision);
        y = convolution(e, w, y, nbr, b + "conv2", precision, true);
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = rounded(h[i] + y[i], precision);
    };
    resblock("middle_block.0.");
    resblock("middle_block.1.");
    for (int block = 0; block < 8; ++block) {
        std::string b = "blocks." + std::to_string(block) + ".";
        if (w.has(b + "conv1.weight"))
            resblock(b);
        else {
            Vec x = convolution(e, w, h, nbr, b + "conv", precision, true);
            c = w.shape(b + "conv.weight")[0] / 8;
            int next = size * 2;
            h.resize(size_t(next) * next * next * c);
#pragma omp parallel for schedule(static)
            for (int p = 0; p < size * size * size; ++p)
                for (int s = 0; s < 8; ++s) {
                    int a = (p / (size * size) * 2 + (s >> 2)), b = (p / size % size * 2 + ((s >> 1) & 1)),
                        z = p % size * 2 + (s & 1);
                    for (int ch = 0; ch < c; ++ch)
                        h[(size_t(a) * next * next + b * next + z) * c + ch] =
                            x[size_t(p) * c * 8 + ch * 8 + s];
                }
            size = next;
            nbr = dense_neighbors(size);
        }
    }
    Vec x;
    layer_norm(x, h, c, w, "out_layer.0", 0, 1e-5f);
    activate(x, 0);
    return convolution(e, w, x, nbr, "out_layer.2", 0, true);
}
} // namespace px
