#include "models.hpp"
namespace hv15n {
static Tensor residual(Gpu &g, Weights &w, const std::string &p, const Tensor &x) {
    auto z = g.optimized ? g.norm_silu(w,p+".norm1",x) : g.op(g.norm(w, p + ".norm1", x, 2), 3);
    z = g.conv(w, p + ".conv1.conv", z);
    z = g.optimized ? g.norm_silu(w,p+".norm2",z) : g.op(g.norm(w, p + ".norm2", z, 2), 3);
    z = g.conv(w, p + ".conv2.conv", z);
    auto skip = x;
    if (w.has(p + ".nin_shortcut.weight"))
        skip = g.conv(w, p + ".nin_shortcut", x, false);
    return g.op(skip, 1, &z);
}
static Tensor middle(Gpu &g, Weights &w, const std::string &p, Tensor x) {
    x = residual(g, w, p + ".block_1", x);
    auto z = g.norm(w, p + ".attn_1.norm", x, 2);
    auto q = g.conv(w, p + ".attn_1.q", z, false), k = g.conv(w, p + ".attn_1.k", z, false),
         v = g.conv(w, p + ".attn_1.v", z, false);
    auto attn = g.attention(q, k, v, 1, 1, 2, x.shape[1] * x.shape[2]);
    attn.shape = x.shape;
    auto out = g.conv(w, p + ".attn_1.proj_out", attn, false);
    x = g.op(x, 1, &out);
    return residual(g, w, p + ".block_2", x);
}
Tensor vae(Gpu &g, Weights &w, const Tensor &input, bool encode) {
    auto x = input;
    if (encode) {
        x = g.conv(w, "encoder.conv_in.conv", x);
        for (int level = 0; level < 5; level++) {
            auto p = "encoder.down." + std::to_string(level);
            for (int block = 0; block < 2; block++)
                x = residual(g, w, p + ".block." + std::to_string(block), x);
            if (level < 4) {
                bool temporal = level >= 2;
                require(x.shape[1] % 2 == 0 && x.shape[2] % 2 == 0, "odd VAE downsample shape");
                auto conv = g.conv(w, p + ".downsample.conv.conv", x);
                int c = conv.channels() * (temporal ? 8 : 4);
                auto y = g.empty(
                    {temporal ? (x.shape[0] + 1) / 2 : x.shape[0], x.shape[1] / 2, x.shape[2] / 2, c});
                g.launch("downshuffle", int((y.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer,
                         conv.pointer, x.pointer, x.shape[0], x.shape[1], x.shape[2], x.channels(),
                         conv.channels(), c, int(temporal));
                x = std::move(y);
            }
        }
        x = middle(g, w, "encoder.mid", x);
        auto skip = g.channel_map(x, 64, true);
        x = g.optimized ? g.norm_silu(w,"encoder.norm_out",x) : g.op(g.norm(w, "encoder.norm_out", x, 2), 3);
        x = g.conv(w, "encoder.conv_out.conv", x);
        x = g.op(x, 1, &skip);
        return g.op(g.columns(x, 0, 32), 0, nullptr, nullptr, 1.03682f);
    }
    x = g.op(input, 0, nullptr, nullptr, 1.f / 1.03682f);
    auto skip = g.channel_map(x, 1024, false);
    x = g.conv(w, "decoder.conv_in.conv", x);
    x = g.op(x, 1, &skip);
    x = middle(g, w, "decoder.mid", x);
    for (int level = 0; level < 5; level++) {
        auto p = "decoder.up." + std::to_string(level);
        for (int block = 0; block < 3; block++)
            x = residual(g, w, p + ".block." + std::to_string(block), x);
        if (level < 4) {
            bool temporal = level < 2;
            auto conv = g.conv(w, p + ".upsample.conv.conv", x);
            int c = conv.channels() / (temporal ? 8 : 4);
            auto y = g.empty(
                {temporal ? (x.shape[0] - 1) * 2 + 1 : x.shape[0], x.shape[1] * 2, x.shape[2] * 2, c});
            g.launch("upshuffle", int((y.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, conv.pointer,
                     x.pointer, x.shape[0], x.shape[1], x.shape[2], x.channels(), c, int(temporal));
            x = std::move(y);
        }
    }
    x = g.optimized ? g.norm_silu(w,"decoder.norm_out",x) : g.op(g.norm(w, "decoder.norm_out", x, 2), 3);
    return g.conv(w, "decoder.conv_out.conv", x);
}
struct Tile {
    std::vector<float> data;
    int t, h, w, c;
    size_t at(int frame, int y, int x, int channel) const {
        return ((size_t(frame) * h + y) * w + x) * c + channel;
    }
};
Tensor vae_tiled(Gpu &g, Weights &w, const Tensor &input, bool encode) {
    require(input.shape.size() == 4, "VAE tile input requires THWC");
    int tile = encode ? 128 : 8, stride = encode ? 96 : 6, blend = encode ? 2 : 32, keep = encode ? 6 : 96;
    int it = input.shape[0], ih = input.shape[1], iw = input.shape[2], ic = input.shape[3];
    require(!encode || (ih % 16 == 0 && iw % 16 == 0), "VAE encode dimensions must align to 16");
    if (ih <= tile && iw <= tile)
        return vae(g, w, input, encode);
    auto source = g.download(input);
    int ot = encode ? (it - 1) / 4 + 1 : (it - 1) * 4 + 1, oh = encode ? ih / 16 : ih * 16,
        ow = encode ? iw / 16 : iw * 16, oc = encode ? 32 : 3;
    std::vector<float> output(product({ot, oh, ow, oc}));
#ifndef HV15N_ROCM
    // Pipelined tiles: tile k+1 is uploaded (pinned staging, no stream sync) and enqueued
    // before tile k is downloaded (async, pinned) and blended on the host, so host work
    // overlaps GPU decoding. Blending arithmetic is unchanged.
    std::vector<std::array<int, 4>> order; // y, x, oy, ox
    std::vector<int> row_start;
    for (int y = 0, oy = 0; y < ih; y += stride, oy += keep) {
        row_start.push_back(int(order.size()));
        for (int x = 0, ox = 0; x < iw; x += stride, ox += keep)
            order.push_back({y, x, oy, ox});
    }
    struct Slot {
        float *host = nullptr;
        CUevent ready = nullptr;
        std::vector<int> shape;
    };
    const size_t slot_bytes = size_t(ot) * (encode ? tile / 16 : tile * 16) *
                              (encode ? tile / 16 : tile * 16) * oc * 4;
    std::array<Slot, 2> slots;
    struct SlotGuard {
        std::array<Slot, 2> &s;
        Gpu &g;
        ~SlotGuard() {
            cuStreamSynchronize(g.stream);
            for (auto &x : s) {
                if (x.ready) cuEventDestroy(x.ready);
                if (x.host) cuMemFreeHost(x.host);
            }
        }
    } guard{slots, g};
    for (auto &slot : slots) {
        g.check(cuMemHostAlloc(reinterpret_cast<void **>(&slot.host), slot_bytes, 0), "VAE pinned tile");
        g.check(cuEventCreate(&slot.ready, CU_EVENT_DISABLE_TIMING), "VAE tile event");
    }
    auto start = [&](size_t k) {
        auto [y, x, oy, ox] = order[k];
        (void)oy;
        (void)ox;
        int th = std::min(tile, ih - y), tw = std::min(tile, iw - x);
        std::vector<float> pixels(product({it, th, tw, ic}));
        for (int f = 0; f < it; f++)
            for (int yy = 0; yy < th; yy++)
                std::copy_n(source.data() + ((size_t(f) * ih + y + yy) * iw + x) * ic, tw * ic,
                            pixels.data() + (size_t(f) * th + yy) * tw * ic);
        auto staged = g.empty({it, th, tw, ic});
        g.upload_staged(staged, pixels.data(), g.stream);
        auto decoded = vae(g, w, staged, encode);
        auto &slot = slots[k % 2];
        require(decoded.element_bytes == 4 && decoded.bytes() <= slot_bytes, "VAE tile slot size");
        slot.shape = decoded.shape;
        g.check(cuMemcpyDtoHAsync(slot.host, decoded.pointer, decoded.bytes(), g.stream),
                "VAE tile download");
        g.check(cuEventRecord(slot.ready, g.stream), "VAE tile ready");
    };
    std::vector<Tile> previous, row;
    start(0);
    for (size_t k = 0; k < order.size(); k++) {
        g.poll();
        if (k + 1 < order.size())
            start(k + 1);
        auto [y, x, oy, ox] = order[k];
        (void)y;
        if (x == 0 && k) {
            previous = std::move(row);
            row.clear();
        }
        auto &slot = slots[k % 2];
        g.check(cuEventSynchronize(slot.ready), "VAE tile wait");
        auto &sh = slot.shape;
        Tile current{std::vector<float>(slot.host, slot.host + product(sh)), sh[0], sh[1], sh[2], sh[3]};
        for (float v : current.data)
            require(std::isfinite(v), "nonfinite inference tensor");
        require(current.t == ot && current.c == oc, "VAE tile output shape mismatch");
        size_t col = row.size();
        if (!previous.empty()) {
            const auto &above = previous.at(col);
            int extent = std::min({blend, current.h, above.h});
            require(above.w == current.w, "VAE vertical tile width mismatch");
            for (int f = 0; f < ot; f++)
                for (int yy = 0; yy < extent; yy++)
                    for (int xx = 0; xx < current.w; xx++)
                        for (int c = 0; c < oc; c++) {
                            auto at = current.at(f, yy, xx, c);
                            float mix = float(yy) / extent;
                            current.data[at] =
                                above.data[above.at(f, above.h - extent + yy, xx, c)] * (1.f - mix) +
                                current.data[at] * mix;
                        }
        }
        if (!row.empty()) {
            const auto &left = row.back();
            int extent = std::min({blend, current.w, left.w});
            require(left.h == current.h, "VAE horizontal tile height mismatch");
            for (int f = 0; f < ot; f++)
                for (int yy = 0; yy < current.h; yy++)
                    for (int xx = 0; xx < extent; xx++)
                        for (int c = 0; c < oc; c++) {
                            auto at = current.at(f, yy, xx, c);
                            float mix = float(xx) / extent;
                            current.data[at] =
                                left.data[left.at(f, yy, left.w - extent + xx, c)] * (1.f - mix) +
                                current.data[at] * mix;
                        }
        }
        int ch = std::min(keep, current.h), cw = std::min(keep, current.w);
        require(oy + ch <= oh && ox + cw <= ow, "VAE tile bounds");
        for (int f = 0; f < ot; f++)
            for (int yy = 0; yy < ch; yy++)
                std::copy_n(current.data.data() + current.at(f, yy, 0, 0), cw * oc,
                            output.data() + ((size_t(f) * oh + oy + yy) * ow + ox) * oc);
        row.push_back(std::move(current));
    }
    (void)row_start;
#else
    std::vector<Tile> previous;
    for (int y = 0, oy = 0; y < ih; y += stride, oy += keep) {
        std::vector<Tile> row;
        for (int x = 0, ox = 0; x < iw; x += stride, ox += keep) {
            g.poll();
            int th = std::min(tile, ih - y), tw = std::min(tile, iw - x);
            std::vector<float> pixels(product({it, th, tw, ic}));
            for (int f = 0; f < it; f++)
                for (int yy = 0; yy < th; yy++)
                    std::copy_n(source.data() + ((size_t(f) * ih + y + yy) * iw + x) * ic, tw * ic,
                                pixels.data() + (size_t(f) * th + yy) * tw * ic);
            auto decoded = vae(g, w, g.upload(pixels, {it, th, tw, ic}), encode);
            Tile current{g.download(decoded), decoded.shape[0], decoded.shape[1], decoded.shape[2],
                         decoded.shape[3]};
            require(current.t == ot && current.c == oc, "VAE tile output shape mismatch");
            size_t col = row.size();
            if (!previous.empty()) {
                const auto &above = previous.at(col);
                int extent = std::min({blend, current.h, above.h});
                require(above.w == current.w, "VAE vertical tile width mismatch");
                for (int f = 0; f < ot; f++)
                    for (int yy = 0; yy < extent; yy++)
                        for (int xx = 0; xx < current.w; xx++)
                            for (int c = 0; c < oc; c++) {
                                auto at = current.at(f, yy, xx, c);
                                float mix = float(yy) / extent;
                                current.data[at] =
                                    above.data[above.at(f, above.h - extent + yy, xx, c)] * (1.f - mix) +
                                    current.data[at] * mix;
                            }
            }
            if (!row.empty()) {
                const auto &left = row.back();
                int extent = std::min({blend, current.w, left.w});
                require(left.h == current.h, "VAE horizontal tile height mismatch");
                for (int f = 0; f < ot; f++)
                    for (int yy = 0; yy < current.h; yy++)
                        for (int xx = 0; xx < extent; xx++)
                            for (int c = 0; c < oc; c++) {
                                auto at = current.at(f, yy, xx, c);
                                float mix = float(xx) / extent;
                                current.data[at] =
                                    left.data[left.at(f, yy, left.w - extent + xx, c)] * (1.f - mix) +
                                    current.data[at] * mix;
                            }
            }
            int ch = std::min(keep, current.h), cw = std::min(keep, current.w);
            require(oy + ch <= oh && ox + cw <= ow, "VAE tile bounds");
            for (int f = 0; f < ot; f++)
                for (int yy = 0; yy < ch; yy++)
                    std::copy_n(current.data.data() + current.at(f, yy, 0, 0), cw * oc,
                                output.data() + ((size_t(f) * oh + oy + yy) * ow + ox) * oc);
            row.push_back(std::move(current));
        }
        previous = std::move(row);
    }
#endif
    return g.upload(output, {ot, oh, ow, oc});
}
} // namespace hv15n
