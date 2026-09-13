#include "engine.hh"
#include <cstdio>

namespace px {
static Vec part(const Vec &x, int n, int parts, int c, int which) {
    Vec y(size_t(n) * c);
    for (int i = 0; i < n; ++i)
        std::copy_n(x.data() + (size_t(i) * parts + which) * c, c, y.data() + size_t(i) * c);
    return y;
}
static void silu(Vec &x) {
#pragma omp parallel for schedule(static) if (x.size() > 16384)
    for (size_t i = 0; i < x.size(); ++i)
        x[i] /= 1.f + std::exp(-x[i]);
}
static void modulate(Vec &y, const Vec &x, int c, const Vec &mod, int offset, bool bf) {
    norm(y, x, c, 1e-6f);
    apply_modulation(y, mod, offset, c, bf);
}
Vec flow(Engine &e, Weights &w, const Vec &input, const Coords &coords, float t, const Vec &global_input,
         const Vec &projected_input, int blocks, bool bf) {
    const int n = int(coords.size() / 4), c = w.shape("input_layer.weight")[0];
    auto gs = w.shape("blocks.0.self_attn.q_rms_norm.gamma");
    require(gs.size() == 2, "Invalid QK normalization weights");
    int heads = gs[0], hd = gs[1];
    require(c == heads * hd && n > 0 && blocks > 0, "Invalid DiT dimensions");
    Vec time(256);
    for (int j = 0; j < 128; ++j) {
        float phase = t * 1000.f * std::exp(-std::log(10000.f) * j / 128.f);
        time[j] = std::cos(phase);
        time[128 + j] = std::sin(phase);
    }
    time = e.linear(time, w, "t_embedder.mlp.0");
    silu(time);
    time = e.linear(time, w, "t_embedder.mlp.2");
    silu(time);
    Vec modulation = e.linear(time, w, "adaLN_modulation.1");
    Vec hidden = e.linear(input, w, "input_layer"), global = global_input, projected = projected_input;
    if (bf) {
        round_bf16(hidden);
        round_bf16(modulation);
        round_bf16(global);
        round_bf16(projected);
    }
    int cond_dim = w.shape("blocks.0.cross_attn.cross_attn_block.to_kv.weight")[1];
    require(global.size() == size_t(5) * cond_dim, "Pixal3D requires five global tokens");
    require(projected.size() == size_t(n) * w.shape("blocks.0.cross_attn.proj_linear.weight")[1],
            "Bad projected conditioning shape");
    for (int block = 0; block < blocks; ++block) {
        std::string b = "blocks." + std::to_string(block) + ".", ca = b + "cross_attn.cross_attn_block.";
        Vec mod = modulation;
        const float *bias = w.get(b + "modulation");
        for (int j = 0; j < 6 * c; ++j) {
            mod[j] += bias[j];
            if (bf)
                mod[j] = bf16(mod[j]);
        }
        Vec h;
        modulate(h, hidden, c, mod, 0, bf);
        Vec qkv = e.linear(h, w, b + "self_attn.to_qkv", bf);
        Vec q = part(qkv, n, 3, c, 0), k = part(qkv, n, 3, c, 1), v = part(qkv, n, 3, c, 2);
        qkv.clear();
        qkv.shrink_to_fit();
        rms(q, heads, hd, w.get(b + "self_attn.q_rms_norm.gamma"));
        rms(k, heads, hd, w.get(b + "self_attn.k_rms_norm.gamma"));
        if (bf) {
            round_bf16(q);
            round_bf16(k);
        }
        rope(q, coords, heads, hd);
        rope(k, coords, heads, hd);
        if (bf) {
            round_bf16(q);
            round_bf16(k);
        }
        h.resize(size_t(n) * c);
        e.attention(h.data(), q.data(), k.data(), v.data(), n, n, heads, hd);
        if (bf)
            round_bf16(h);
        h = e.linear(h, w, b + "self_attn.to_out", bf);
        add_residual(hidden, h, mod.data() + 2 * c, c, bf);
        norm(h, hidden, c, 1e-6f, w.get(b + "norm2.weight"), w.get(b + "norm2.bias"));
        if (bf)
            round_bf16(h);
        q = e.linear(h, w, ca + "to_q", bf);
        Vec kv = e.linear(global, w, ca + "to_kv", bf);
        k = part(kv, 5, 2, c, 0);
        v = part(kv, 5, 2, c, 1);
        rms(q, heads, hd, w.get(ca + "q_rms_norm.gamma"));
        rms(k, heads, hd, w.get(ca + "k_rms_norm.gamma"));
        if (bf) {
            round_bf16(q);
            round_bf16(k);
        }
        e.attention(h.data(), q.data(), k.data(), v.data(), n, 5, heads, hd);
        if (bf)
            round_bf16(h);
        h = e.linear(h, w, ca + "to_out", bf);
        Vec projection = e.linear(projected, w, b + "cross_attn.proj_linear", bf);
#pragma omp parallel for schedule(static) if (h.size() > 16384)
        for (size_t i = 0; i < h.size(); ++i) {
            h[i] += projection[i];
            if (bf)
                h[i] = bf16(h[i]);
        }
        add_residual(hidden, h, nullptr, c, bf);
        modulate(h, hidden, c, mod, 3 * c, bf);
        h = e.linear(h, w, b + "mlp.mlp.0", bf);
        gelu(h, true, bf);
        h = e.linear(h, w, b + "mlp.mlp.2", bf);
        add_residual(hidden, h, mod.data() + 5 * c, c, bf);
    }
    Vec h;
    norm(h, hidden, c, 1e-5f);
    return e.linear(h, w, "out_layer");
}
} // namespace px
