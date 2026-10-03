// SPDX-License-Identifier: MIT
#pragma once
#include "../../cuda/hunyuan_video15_native/gpu.hpp"
#include "../../cuda/hunyuan_video15_native/tokenizer.hpp"
#include "../video_common/aotriton_bridge.h"
#include "h3.h"
#include "kernels.hpp"
#include <iostream>
#include <random>
namespace h3 {
using namespace hv15n;
inline float bf(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7f800000u) != 0x7f800000u)
        u += 0x7fff + ((u >> 16) & 1);
    u &= 0xffff0000u;
    std::memcpy(&x, &u, 4);
    return x;
}
inline float fp16(float x) { return float(_Float16(x)); }
inline float blend_half(float a, float b, int position, int extent) {
    float fraction = fp16(float(position) / extent);
    return fp16(fp16(a * fp16(1.f - fraction)) + fp16(b * fraction));
}
struct Engine {
    Gpu g;
    CUmodule module = nullptr;
    uint64_t int8_calls = 0, bf16_calls = 0, bf16_blas_calls = 0, convrot_blas_calls = 0,
             fp32_blas_calls = 0;
    fs::path root;
    fs::path trace;
    bool convrot_hipblas, bf16_hipblas, vae_hipblas;
    bool fp32_hipblas = true;
    cublasew_context *rotation_blas = nullptr;
    Tensor rotation_weight;
    std::unique_ptr<void, int (*)(void *)> aot_library{nullptr, dlclose};
    video_aotriton_forward_fn aot_attention = nullptr;
    uint64_t aot_calls = 0;
    explicit Engine(const h3_config &c)
        : g(c.device, c.vram_budget_mib, false, false), root(c.model_dir),
          convrot_hipblas(c.convrot_hipblas != 0), bf16_hipblas(c.bf16_hipblas != 0),
          vae_hipblas(c.vae_hipblas != 0) {
        require(c.bf16_hipblas == 0 || c.bf16_hipblas == 1, "BF16 backend must be 0 or 1");
        require(c.convrot_hipblas == 0 || c.convrot_hipblas == 1, "ConvRot backend must be 0 or 1");
        require(c.vae_hipblas == 0 || c.vae_hipblas == 1, "VAE backend must be 0 or 1");
        auto code = Gpu::compile(source);
        g.check(cuModuleLoadData(&module, code.data()), "H3 module");
        try {
            if (c.aotriton_bridge) {
                auto path = fs::absolute(c.aotriton_bridge).string();
                aot_library.reset(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL));
                if (!aot_library) {
                    const char *error = dlerror();
                    throw std::runtime_error(
                        std::string("cannot load standalone AOTriton bridge: ") +
                        (error ? error : "unknown loader error"));
                }
                auto abi = reinterpret_cast<int (*)(void)>(
                    dlsym(aot_library.get(), "video_aotriton_bridge_abi"));
                require(abi && abi() == 1, "unsupported AOTriton bridge ABI");
                aot_attention = reinterpret_cast<video_aotriton_forward_fn>(
                    dlsym(aot_library.get(), "video_aotriton_forward"));
                require(aot_attention != nullptr, "missing AOTriton attention entry point");
            }
            for (const char *name : {"h3_round", "h3_pack_bf16", "h3_angles", "h3_qwen_angles",
                                     "h3_rotate", "h3_quant", "h3_norm", "h3_swiglu", "h3_qkv",
                                     "h3_rope", "h3_mod", "h3_gate", "h3_scale_add", "h3_patch",
                                     "h3_unpatch", "h3_decode_patch", "h3_qwen_attention"}) {
                CUfunction f;
                g.check(cuModuleGetFunction(&f, module, name), name);
                g.functions[name] = f;
            }
            for (auto name : {"video_gemm_i8", "video_gemm_i8_tiled", "video_gemm_i8_pipeline",
                              "video_gemm_bf16"}) {
                CUfunction f;
                g.check(cuModuleGetFunction(&f, g.mma, name), name);
                g.functions[name] = f;
            }
            for (auto name : {"video_attention_bf16", "video_attention_bf16_exp2",
                              "video_attention_tile_bf16_exp2", "video_attention_bf16_64",
                              "video_attention_tile_bf16", "video_attention_tile_bf16_64",
                              "video_attention_tile_f16", "video_attention_tile_f16_64"}) {
                CUfunction f;
                g.check(cuModuleGetFunction(&f, g.flash, name), name);
                g.functions[name] = f;
            }
        } catch (...) {
            cuModuleUnload(module);
            module = nullptr;
            throw;
        }
    }
    ~Engine() {
        cuCtxSetCurrent(g.context);
        cuStreamSynchronize(g.stream);
        if (rotation_blas)
            cublasewDestroy(rotation_blas);
        if (module) {
            cuCtxSetCurrent(g.context);
            cuStreamSynchronize(g.stream);
            cuModuleUnload(module);
        }
    }
    Tensor byte_tensor(std::vector<int> shape) {
        Tensor t;
        t.shape = std::move(shape);
        t.element_bytes = 1;
        t.storage = std::make_shared<Allocation>(&g, t.bytes());
        t.pointer = t.storage->pointer;
        return t;
    }
    Tensor rounded(Tensor x, int kind = 1) {
        g.launch("h3_round", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, x.pointer,
                 int64_t(x.count()), kind);
        return x;
    }
    Tensor rotate(const Tensor &x, int kind) {
        auto out = g.empty(x.shape);
        if (kind != 1 || !convrot_hipblas) {
            g.launch("h3_rotate", int(x.count() / 256), 1, 1, 256, 1, 0, out.pointer, x.pointer,
                     int(x.count() / 256), kind);
            return out;
        }
        if (!rotation_blas) {
            require(cublasewCreate(&rotation_blas, g.stream) == 0,
                    "BF16 ConvRot requires hipBLAS; use --convrot-hipblas 0 for native FHT");
        }
        if (!rotation_weight.pointer) {
            std::vector<uint16_t> matrix(256 * 256);
            for (int row = 0; row < 256; row++)
                for (int col = 0; col < 256; col++) {
                    int sign = 0;
                    for (int digit = 0; digit < 4; digit++)
                        sign ^= (((row >> (2 * digit)) & 3) + ((col >> (2 * digit)) & 3) == 3);
                    matrix[row * 256 + col] = sign ? 0xbd80 : 0x3d80; // BF16 +/-1/16
                }
            rotation_weight = g.empty_half({256, 256});
            g.upload_staged(rotation_weight, matrix.data(), g.stream);
        }
        auto packed = g.empty_half(x.shape);
        g.launch("h3_pack_bf16", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, packed.pointer,
                 x.pointer, int64_t(x.count()));
        // torch.matmul folds the leading group dimensions into one GEMM.
        require(video_hipblas_gemm(rotation_blas, out.pointer, rotation_weight.pointer,
                                   packed.pointer, int(x.count() / 256), 256, 256, 14) == 0,
                "BF16 ConvRot hipBLAS GEMM failed");
        convrot_blas_calls++;
        return rounded(out, 1);
    }
    struct Linear {
        Engine &e;
        Tensor w, scale, bias;
        int kind;
        bool quant, precise;
        Linear(Engine &owner, Weights &weights, const std::string &name, int dtype = 1,
               bool fp32 = false)
            : e(owner), kind(dtype), quant(weights.dtype(name + ".weight") == "I8"), precise(fp32) {
            auto shape = weights.shape(name + ".weight");
            require(shape.size() == 2, "linear matrix rank");
            if (quant) {
                require(shape[1] % 256 == 0 && weights.has(name + ".comfy_quant"),
                        "INT8 requires ConvRot-256 metadata");
                auto md = weights.data(name + ".comfy_quant");
                auto bytes = product(weights.shape(name + ".comfy_quant"));
                std::unique_ptr<json_val, decltype(&json_free)> meta(
                    json_parse(static_cast<const char *>(md), int(bytes)), json_free);
                require(meta && string(meta.get(), "format") == "int8_tensorwise" &&
                            field(meta.get(), "convrot") &&
                            field(meta.get(), "convrot")->type == JSON_TRUE &&
                            field(meta.get(), "convrot_groupsize") &&
                            field(meta.get(), "convrot_groupsize")->num == 256,
                        "unsupported H3 INT8 format");
                require(weights.shape(name + ".weight_scale") == std::vector<int>({shape[0], 1}),
                        "INT8 row scale shape");
                w = e.byte_tensor(shape);
                e.g.upload_staged(w, weights.data(name + ".weight"), e.g.stream);
                scale = e.g.weight(weights, name + ".weight_scale");
            } else if (kind == 1 && !precise) {
                // Preserve BF16 weights; FP16 conversion loses subnormal bits.
                w = e.g.empty_half(shape);
                if (weights.dtype(name + ".weight") == "BF16") {
                    e.g.upload_staged(w, weights.data(name + ".weight"), e.g.stream);
                } else {
                    auto full = e.g.weight(weights, name + ".weight");
                    e.g.launch("h3_pack_bf16", int((w.count() + 255) / 256), 1, 1, 256, 1, 0,
                               w.pointer, full.pointer, int64_t(w.count()));
                }
            } else {
                w = precise ? e.g.weight(weights, name + ".weight")
                            : e.g.weight_half(weights, name + ".weight");
            }
            if (weights.has(name + ".bias"))
                bias = e.g.weight(weights, name + ".bias");
        }
        Tensor operator()(const Tensor &x) {
            require(x.channels() == w.shape[1] && x.element_bytes == 4, "linear input");
            Tensor y;
            if (quant) {
                auto rotated = e.rotate(x, kind), q = e.byte_tensor(x.shape),
                     xs = e.g.empty({x.rows()});
                e.g.launch("h3_quant", x.rows(), 1, 1, 256, 1, 0, q.pointer, xs.pointer,
                           rotated.pointer, x.rows(), x.channels(), kind);
                y = e.g.empty({x.rows(), w.shape[0]});
                const int tile = x.rows() >= 128 ? 128 : 32;
                e.g.launch(tile == 128 ? "video_gemm_i8_pipeline" : "video_gemm_i8",
                           (y.channels() + tile - 1) / tile, (x.rows() + tile - 1) / tile, 1, 128,
                           1, 0, y.pointer, q.pointer, w.pointer, xs.pointer, scale.pointer,
                           x.rows(), y.channels(), x.channels(), int(kind == 1), CUdeviceptr(0));
                e.int8_calls++;
                if (kind == 2)
                    y = e.rounded(y, kind);
            } else if (kind == 1 && !precise) {
                y = e.g.empty({x.rows(), w.shape[0]});
                if (e.bf16_hipblas) {
                    if (!e.rotation_blas)
                        require(cublasewCreate(&e.rotation_blas, e.g.stream) == 0,
                                "BF16 GEMM requires hipBLAS; use --bf16-hipblas 0 for native WMMA");
                    auto packed = e.g.empty_half(x.shape);
                    e.g.launch("h3_pack_bf16", int((x.count() + 255) / 256), 1, 1, 256, 1, 0,
                               packed.pointer, x.pointer, int64_t(x.count()));
                    require(video_hipblas_gemm(e.rotation_blas, y.pointer, w.pointer,
                                               packed.pointer, x.rows(), y.channels(), x.channels(),
                                               14) == 0,
                            "BF16 hipBLAS GEMM failed");
                    e.bf16_blas_calls++;
                } else {
                    e.g.launch("video_gemm_bf16", (y.channels() + 31) / 32, (x.rows() + 31) / 32, 1,
                               128, 1, 0, y.pointer, x.pointer, w.pointer, x.rows(), y.channels(),
                               x.channels());
                    e.bf16_calls++;
                    e.g.repo_gemm++;
                }
            } else if (precise && e.fp32_hipblas) {
                if (!e.rotation_blas)
                    require(cublasewCreate(&e.rotation_blas, e.g.stream) == 0,
                            "FP32 GEMM requires hipBLAS; use --fp32-hipblas 0 for native IEEE");
                y = e.g.empty({x.rows(), w.shape[0]});
                require(video_hipblas_gemm(e.rotation_blas, y.pointer, w.pointer, x.pointer,
                                           x.rows(), y.channels(), x.channels(), 0) == 0,
                        "FP32 hipBLAS GEMM failed");
                e.fp32_blas_calls++;
            } else {
                y = e.g.matmul(x, w, precise);
            }
            if (bias.pointer) {
                y = e.g.op(y, 6, nullptr, &bias);
                if (!precise)
                    y = e.rounded(y, kind);
            }
            if (!quant && !precise)
                y = e.rounded(y, kind);
            return y;
        }
    };
    Tensor linear(Weights &w, const std::string &p, const Tensor &x, int kind = 1,
                  bool precise = false) {
        return Linear(*this, w, p, kind, precise)(x);
    }
    Tensor norm(Weights *w, const std::string &p, const Tensor &x, int kind = 1, int mode = 2,
                float eps = 1e-5) {
        Tensor gamma, bias;
        if (w && w->has(p + ".weight"))
            gamma = g.weight(*w, p + ".weight");
        if (w && w->has(p + ".bias"))
            bias = g.weight(*w, p + ".bias");
        auto y = g.empty(x.shape);
        g.launch("h3_norm", x.rows(), 1, 1, 256, 1, 0, y.pointer, x.pointer, gamma.pointer,
                 bias.pointer, x.rows(), x.channels(), mode, eps, kind);
        return y;
    }
    Tensor head_norm(Weights *w, const std::string &p, Tensor x, int dim, int kind = 1,
                     float eps = 1e-5, int mode = 2) {
        auto shape = x.shape;
        x.shape = {int(x.count() / dim), dim};
        auto y = norm(w, p, x, kind, mode, eps);
        y.shape = shape;
        return y;
    }
    void rope(Tensor &x, const Tensor &angles, int heads, int dim, int pairs, int kind,
              bool eager = true) {
        g.launch("h3_rope", (x.rows() * heads * pairs + 255) / 256, 1, 1, 256, 1, 0, x.pointer,
                 angles.pointer, x.rows(), heads, dim, pairs, kind, int(eager));
    }
    Tensor attention(Tensor q, Tensor k, Tensor v, int heads, int kvheads, int dim, int kind = 1,
                     bool causal = false) {
        require(q.rows() == k.rows() && k.rows() == v.rows() && q.channels() == heads * dim &&
                    k.channels() == kvheads * dim && v.channels() == kvheads * dim &&
                    (dim == 64 || dim == 128) && heads > 0 && kvheads > 0 && heads % kvheads == 0,
                "H3 attention shape");
        // Qwen's short causal sequence keeps FP32 softmax and value accumulation;
        // quantized projections and the resulting hidden states remain BF16.
        if (causal) {
            require(q.rows() <= 512 && kind == 1 && dim == 128, "Qwen attention geometry");
            auto out = g.empty(q.shape);
            g.launch("h3_qwen_attention", q.rows(), heads, 1, 32, 1, 0, out.pointer, q.pointer,
                     k.pointer, v.pointer, q.rows(), heads, kvheads, dim,
                     float(std::sqrt(1.0 / std::sqrt(double(dim)))));
            g.attention_calls++;
            return out;
        }
        if (aot_attention && kind == 1 && dim == 128 && q.rows() >= 128) {
            const auto shape = q.shape;
            const int rows = q.rows();
            auto pq = g.empty_half(q.shape), pk = g.empty_half(k.shape), pv = g.empty_half(v.shape);
            for (auto pair : {std::pair<Tensor *, Tensor *>{&pq, &q}, {&pk, &k}, {&pv, &v}})
                g.launch("h3_pack_bf16", int((pair.second->count() + 255) / 256), 1, 1, 256, 1, 0,
                         pair.first->pointer, pair.second->pointer, int64_t(pair.second->count()));
            // The calling graph transfers ownership so these large FP32 buffers
            // can be reused before allocating the provider output.
            q = {};
            k = {};
            v = {};
            auto packed_out = g.empty_half(shape), lse = g.empty({heads, rows}),
                 out = g.empty(shape);
            g.poll();
            g.check(aot_attention(pq.pointer, pk.pointer, pv.pointer, packed_out.pointer,
                                  lse.pointer, rows, heads, kvheads, dim, kind,
                                  1.f / std::sqrt(float(dim)), g.stream),
                    "standalone AOTriton attention");
            g.launch("convert_bfloat", int((out.count() + 255) / 256), 1, 1, 256, 1, 0, out.pointer,
                     packed_out.pointer, int(out.count()));
            aot_calls++;
            g.attention_calls++;
            g.flash_calls++;
            return out;
        }
        auto out = g.empty(q.shape);
        const char *name = dim == 128 ? "video_attention_bf16_exp2" : "video_attention_bf16_64";
        const bool tiled = q.rows() >= 128 || kind == 2;
        if (tiled)
            name = kind == 2
                       ? (dim == 128 ? "video_attention_tile_f16" : "video_attention_tile_f16_64")
                       : (dim == 128 ? "video_attention_tile_bf16_exp2"
                                     : "video_attention_tile_bf16_64");
        const int tile = tiled ? 128 : 64;
        g.launch(name, heads, (q.rows() + tile - 1) / tile, 1, tiled ? 256 : 128, 1, 0, out.pointer,
                 q.pointer, k.pointer, v.pointer, q.rows(), heads, dim, 1.f / std::sqrt(float(dim)),
                 kvheads, int(causal));
        g.attention_calls++;
        g.flash_calls++;
        return rounded(out, kind);
    }
    Tensor self_attention(Weights &w, const std::string &p, const Tensor &x, int heads, int dim,
                          int kind, const Tensor *angles = nullptr, int pairs = 0,
                          bool interleaved = false) {
        auto packed = linear(w, p + (interleaved ? ".to_qkv" : ".qkv_proj"), x, kind);
        if (!trace.empty())
            g.dump(packed, trace, p + ".qkv");
        auto q = g.empty({x.rows(), heads * dim}), k = g.empty(q.shape), v = g.empty(q.shape);
        g.launch("h3_qkv", int((q.count() + 255) / 256), 1, 1, 256, 1, 0, q.pointer, k.pointer,
                 v.pointer, packed.pointer, x.rows(), heads, dim, int(interleaved));
        packed = {};
        q = head_norm(interleaved ? nullptr : &w, p + ".q_norm", q, dim, kind);
        k = head_norm(interleaved ? nullptr : &w, p + ".k_norm", k, dim, kind);
        if (angles) {
            rope(q, *angles, heads, dim, pairs, kind);
            rope(k, *angles, heads, dim, pairs, kind);
        }
        if (!trace.empty()) {
            g.dump(q, trace, p + ".q");
            g.dump(k, trace, p + ".k");
            g.dump(v, trace, p + ".v");
        }
        auto attended =
            attention(std::move(q), std::move(k), std::move(v), heads, heads, dim, kind);
        if (!trace.empty()) {
            g.dump(attended, trace, p + ".context");
        }
        return linear(w, p + (interleaved ? ".to_out" : ".out_proj"), attended, kind);
    }
    Tensor swiglu(const Tensor &x, int kind) {
        auto out = g.empty({x.rows(), x.channels() / 2});
        g.launch("h3_swiglu", int((out.count() + 255) / 256), 1, 1, 256, 1, 0, out.pointer,
                 x.pointer, x.rows(), out.channels(), kind);
        return out;
    }
    Tensor ffn(Weights &w, const std::string &p, const Tensor &x, int kind = 1, bool vae = false) {
        Linear up(*this, w, p + (vae ? ".w1" : ".fc1"), kind),
            down(*this, w, p + (vae ? ".w2" : ".fc2"), kind);
        auto out = g.empty(x.shape);
        for (int row = 0; row < x.rows(); row += 256) {
            int count = std::min(256, x.rows() - row);
            auto part = down(swiglu(up(g.rows(x, row, count)), kind));
            g.check(cuMemcpyDtoDAsync(out.pointer + size_t(row) * x.channels() * 4, part.pointer,
                                      part.bytes(), g.stream),
                    "FFN chunk");
        }
        return out;
    }
    Tensor residual(Tensor x, const Tensor &delta, int kind) {
        return rounded(g.op(x, 1, &delta), kind);
    }
    Tensor modulate(const Tensor &x, const Tensor &mod, int text, int audio, int chunk,
                    int kind = 1) {
        auto y = g.empty(x.shape);
        g.launch("h3_mod", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, y.pointer, x.pointer,
                 mod.pointer, x.rows(), x.channels(), text, audio, chunk, kind);
        return y;
    }
    void gated(Tensor &x, const Tensor &delta, const Tensor &mod, int text, int audio, int chunk) {
        g.launch("h3_gate", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, x.pointer, delta.pointer,
                 mod.pointer, x.rows(), x.channels(), text, audio, chunk, 1);
    }
    Tensor qwen_angles(int rows) {
        require(rows > 0 && rows <= 512, "Qwen rotary sequence length");
        auto out = g.empty({rows, 64, 2});
        g.launch("h3_qwen_angles", (rows * 64 + 255) / 256, 1, 1, 256, 1, 0, out.pointer, rows);
        return out;
    }
    Tensor encode(const std::string &prompt, const fs::path &dump) {
        Weights w(root / "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors");
        Tokenizer tokenizer(root / "tokenizer/tokenizer.json");
        auto ids = tokenizer.encode(prompt);
        require(!ids.empty() && ids.size() <= 512, "prompt must tokenize to 1..512 tokens");
        require(w.shape("model.embed_tokens.weight") == std::vector<int>({151936, 5120}),
                "Qwen embedding shape");
        std::vector<float> data(ids.size() * 5120);
        auto embed = static_cast<const uint16_t *>(w.data("model.embed_tokens.weight"));
        for (size_t r = 0; r < ids.size(); r++)
            for (int c = 0; c < 5120; c++) {
                uint32_t u = uint32_t(embed[size_t(ids[r]) * 5120 + c]) << 16;
                std::memcpy(&data[r * 5120 + c], &u, 4);
            }
        auto x = g.upload(data, {int(ids.size()), 5120});
        auto rotation = qwen_angles(int(ids.size()));
        if (!trace.empty())
            g.dump(rotation, trace, "qwen_rotation");
        for (int i = 0; i < 50; i++) {
            g.poll();
            std::string p = "model.layers." + std::to_string(i);
            auto capture = [&](const Tensor &value, const char *name) {
                if (!trace.empty() && (i == 0 || std::string(name) == "after_mlp"))
                    g.dump(value, trace, p + "." + name);
            };
            auto z = norm(&w, p + ".input_layernorm", x, 1, 2, 1e-6);
            capture(z, "input_norm");
            auto q = linear(w, p + ".self_attn.q_proj", z),
                 k = linear(w, p + ".self_attn.k_proj", z),
                 v = linear(w, p + ".self_attn.v_proj", z);
            capture(q, "raw_q");
            capture(k, "raw_k");
            capture(v, "v");
            q = head_norm(&w, p + ".self_attn.q_norm", q, 128, 1, 1e-6);
            k = head_norm(&w, p + ".self_attn.k_norm", k, 128, 1, 1e-6);
            capture(q, "norm_q");
            capture(k, "norm_k");
            rope(q, rotation, 64, 128, 64, 1, false);
            rope(k, rotation, 8, 128, 64, 1, false);
            capture(q, "rope_q");
            capture(k, "rope_k");
            auto context = attention(q, k, v, 64, 8, 128, 1, true);
            capture(context, "context");
            x = residual(x, linear(w, p + ".self_attn.o_proj", context), 1);
            capture(x, "after_attn");
            z = norm(&w, p + ".post_attention_layernorm", x, 1, 2, 1e-6);
            capture(z, "ffn_norm");
            auto gate = linear(w, p + ".mlp.gate_proj", z), up = linear(w, p + ".mlp.up_proj", z);
            capture(gate, "gate");
            capture(up, "up");
            gate = g.op(gate, 3);
            gate = rounded(gate);
            auto activated = rounded(g.op(gate, 2, &up));
            capture(activated, "activated");
            x = residual(x, linear(w, p + ".mlp.down_proj", activated), 1);
            capture(x, "after_mlp");
            g.clear_weights();
            if (i % 10 == 9)
                std::cerr << "H3 Qwen layer " << i + 1 << "/50\n";
        }
        g.dump(x, dump, "qwen_hidden");
        g.clear_weights();
        return x; // layer 50 intentionally has no final RMSNorm
    }
    Tensor refine(Weights &w, Tensor x, const fs::path &dump) {
        x = linear(w, "condition_proj", x);
        if (!trace.empty())
            g.dump(x, trace, "condition_proj");
        for (int i = 0; i < 2; i++) {
            auto p = "token_refiner.blocks." + std::to_string(i);
            auto z = norm(&w, p + ".norm1", x);
            if (!trace.empty())
                g.dump(z, trace, p + ".norm1");
            x = residual(x, self_attention(w, p + ".attn", z, 56, 128, 1), 1);
            if (!trace.empty())
                g.dump(x, trace, p + ".after_attn");
            x = residual(x, ffn(w, p + ".mlp", norm(&w, p + ".norm2", x)), 1);
            if (!trace.empty())
                g.dump(x, trace, p + ".after_mlp");
            g.clear_weights();
        }
        x = norm(&w, "token_refiner.final_norm", x);
        g.dump(x, dump, "refined_text");
        g.clear_weights();
        return x;
    }
    Tensor dit_rope(Weights &w, int text, int audio_t, int t, int h, int width) {
        int total = text + 2 * audio_t + t * (h / 2) * (width / 2);
        std::vector<float> phases(size_t(total) * 48);
        auto frequencies = w.floats("rope.inv_freq");
        require(frequencies.size() == 16, "H3 rotary shape");
        double area = std::sqrt(double(h * width));
        auto axis = [&](int i, int dim) {
            double ratio = dim / area;
            return ((1 - ratio) / 2 + i * ratio / (dim / 2)) * 32;
        };
        auto set = [&](int r, double time, double y, double x) {
            double coords[] = {time, y, x};
            for (int a = 0; a < 3; a++)
                for (int p = 0; p < 16; p++) {
                    float angle = float(coords[a]) * frequencies[p];
                    phases[size_t(r) * 48 + a * 16 + p] = angle;
                }
        };
        for (int r = 0; r < text; r++)
            set(r, r, 0, 0);
        for (int ch = 0; ch < 2; ch++)
            for (int r = 0; r < audio_t; r++)
                set(text + ch * audio_t + r, text + r, 0, axis(ch ? width / 2 - 1 : 0, width));
        double time = text;
        int row = text + 2 * audio_t;
        for (int tt = 0; tt < t; tt++) {
            for (int y = 0; y < h / 2; y++)
                for (int x = 0; x < width / 2; x++)
                    set(row++, time, axis(y, h), axis(x, width));
            time += (tt % 5 == 0 ? 1 : 4) * (5. / 3.);
        }
        auto input = g.upload(phases, {total, 48});
        auto out = g.empty({total, 48, 2});
        g.launch("h3_angles", int((phases.size() + 255) / 256), 1, 1, 256, 1, 0, out.pointer,
                 input.pointer, int64_t(phases.size()), 1);
        return out;
    }
    Tensor time_embed(Weights &w, float tv, float ta) {
        auto table = w.floats("adaln_t_table");
        require(table.size() == 1025 * 8, "AdaLN basis shape");
        std::vector<float> values(16);
        float times[] = {tv, ta};
        for (int r = 0; r < 2; r++) {
            float pos = std::clamp(times[r], 0.f, 1.f) * 1024;
            int i = std::min(int(std::floor(pos)), 1023);
            float f = pos - i;
            for (int c = 0; c < 8; c++) {
                const float start = table[i * 8 + c], end = table[(i + 1) * 8 + c];
                // Match the pinned torch.lerp's stable branch and GPU FMA.
                values[r * 8 + c] = f < .5f ? std::fma(f, end - start, start)
                                            : std::fma(-(1.f - f), end - start, end);
            }
        }
        return g.upload(values, {2, 8});
    }
    std::array<Tensor, 2> denoise(Weights &w, const Tensor &text, const Tensor &video,
                                  const Tensor &audio, const Tensor &rotation, int t, int h,
                                  int width, float tv, float ta, int step) {
        auto patches = g.empty({t * (h / 2) * (width / 2), 96});
        g.launch("h3_patch", int((video.count() + 255) / 256), 1, 1, 256, 1, 0, patches.pointer,
                 video.pointer, t, h, width);
        auto vi = rounded(linear(w, "video_patch_proj", patches, 1, true)),
             au = rounded(linear(w, "audio_patch_proj", audio, 1, true));
        auto x = g.concat(g.concat(text, au), vi);
        auto emb = time_embed(w, tv, ta);
        g.clear_weights();
        require(trace.empty() || x.rows() <= 256, "DiT tracing requires at most 256 tokens");
        for (int i = 0; i < 50; i++) {
            auto capture = [&](const char *stage) {
                if (!trace.empty()) {
                    char name[96];
                    std::snprintf(name, sizeof(name), "dit_step_block_%02d_%s", i, stage);
                    g.dump(x, trace, name);
                }
            };
            std::string p = "blocks." + std::to_string(i);
            auto modulation = linear(w, p + ".adaln_proj.linear", emb, 0, true);
            auto z = modulate(norm(&w, p + ".norm1", x), modulation, text.rows(), audio.rows(), 0);
            gated(x, self_attention(w, p + ".attn", z, 56, 128, 1, &rotation, 48), modulation,
                  text.rows(), audio.rows(), 2);
            capture("after_attn");
            z = modulate(norm(&w, p + ".norm2", x), modulation, text.rows(), audio.rows(), 3);
            gated(x, ffn(w, p + ".mlp", z), modulation, text.rows(), audio.rows(), 5);
            capture("after_mlp");
            g.clear_weights();
            if (i % 10 == 9)
                std::cerr << "H3 step " << step << " block " << i + 1 << "/50\n";
        }
        auto finalmod = linear(w, "final_layer.adaln_proj.linear", emb, 0, true);
        auto values = g.download(finalmod);
        auto finish = [&](int start, int rows, int time, const char *name) {
            auto z = norm(&w, "final_layer.norm", g.rows(x, start, rows));
            std::vector<float> m(6 * 5376 * 6, 0);
            for (int c = 0; c < 5376; c++) {
                m[c] = values[time * 10752 + c];
                m[5376 + c] = values[time * 10752 + 5376 + c];
            }
            auto mod = g.upload(m, {6 * 6 * 5376});
            z = modulate(z, mod, 0, 0, 0, 0);
            return linear(w, name, z, 0, true);
        };
        auto av = finish(text.rows(), audio.rows(), 1, "final_layer.audio_out"),
             vv = finish(text.rows() + audio.rows(), vi.rows(), 0, "final_layer.video_out");
        auto velocity = g.empty(video.shape);
        g.launch("h3_unpatch", int((video.count() + 255) / 256), 1, 1, 256, 1, 0, velocity.pointer,
                 vv.pointer, t, h, width);
        g.clear_weights();
        return {velocity, av};
    }
    std::vector<float> decode_tile(Weights &w, const std::vector<float> &z, int t, int h,
                                   int width) {
        auto input = rounded(g.upload(z, {t, h, width, 24}), 2);
        auto postw = g.weight_half(w, "post_quant_conv.weight");
        postw.shape = {24, 24};
        auto post = g.matmul(input, postw);
        auto bias = g.weight(w, "post_quant_conv.bias");
        post = rounded(g.op(post, 6, nullptr, &bias), 2);
        auto x = linear(w, "decoder.x_embedder", post, 2);
        auto registers = g.weight(w, "decoder.register_tokens");
        registers.shape = {4, 2048};
        x = g.concat(g.concat(x, registers), g.upload(std::vector<float>(2048), {1, 2048}));
        std::vector<float> rot(size_t(x.rows()) * 24 * 2, 0);
        for (int r = 0; r < x.rows(); r++) {
            float coord[3] = {0, 0, 0};
            if (r < t * h * width) {
                auto coordinate = [](int index, int size) {
                    return fp16(fp16(fp16((index + .5f) / size) * 2.f) - 1.f);
                };
                coord[0] = coordinate(r / (h * width), t);
                coord[1] = coordinate((r / width) % h, h);
                coord[2] = coordinate(r % width, width);
            }
            for (int a = 0; a < 3; a++)
                for (int p = 0; p < 8; p++) {
                    float angle =
                        float(2 * 3.141592653589793) * coord[a] / std::pow(100.f, p / 8.f);
                    auto at = (size_t(r) * 24 + a * 8 + p) * 2;
                    rot[at] = std::cos(angle);
                    rot[at + 1] = std::sin(angle);
                }
        }
        auto angles = rounded(g.upload(rot, {x.rows(), 24, 2}), 2);
        for (int i = 0; i < 36; i++) {
            auto p = "decoder.transformer_blocks." + std::to_string(i);
            auto d = self_attention(w, p + ".attn", norm(&w, p + ".norm1", x, 2), 32, 64, 2,
                                    &angles, 24, true);
            auto s = g.weight(w, p + ".scale1");
            g.launch("h3_scale_add", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, x.pointer,
                     d.pointer, s.pointer, x.rows(), 2048, 2);
            d = ffn(w, p + ".ff", norm(&w, p + ".norm2", x, 2), 2, true);
            s = g.weight(w, p + ".scale2");
            g.launch("h3_scale_add", int((x.count() + 255) / 256), 1, 1, 256, 1, 0, x.pointer,
                     d.pointer, s.pointer, x.rows(), 2048, 2);
        }
        auto patches = linear(w, "decoder.proj_out", norm(&w, "decoder.norm_out", x, 2, 0), 2);
        auto pixels = g.empty({t * 4, h * 16, width * 16, 3});
        g.launch("h3_decode_patch", int((pixels.count() + 255) / 256), 1, 1, 256, 1, 0,
                 pixels.pointer, patches.pointer, t, h, width);
        auto result = g.download(pixels);
        return result;
    }
    struct Tiles {
        std::vector<int> start, length, overlap;
    };
    static Tiles tiles(int n) {
        Tiles out;
        if (n <= 256) {
            out.start = {0};
            out.length = {n};
            return out;
        }
        int count = (n + 255) / 256;
        while (256 * count - 64 * (count - 1) < n)
            count++;
        out.overlap.assign(count - 1, 64);
        int units = (256 * count - 64 * (count - 1) - n) / 16;
        for (int i = 0; i < units; i++)
            out.overlap[i % (count - 1)] += 16;
        out.start = {0};
        out.length.assign(count, 256);
        for (int i = 0; i < count - 1; i++)
            out.start.push_back(out.start.back() + 256 - out.overlap[i]);
        return out;
    }
    std::vector<float> decode_spatial(Weights &w, const std::vector<float> &latent, int t, int h,
                                      int width) {
        const int height = h * 16, wide = width * 16;
        auto ys = tiles(height), xs = tiles(wide);
        std::vector<float> canvas(size_t(t * 4) * height * wide * 3), strip;
        for (size_t iy = 0; iy < ys.start.size(); iy++) {
            std::vector<float> newstrip, left;
            int overlap_y = iy ? ys.overlap[iy - 1] : 0;
            if (iy + 1 < ys.start.size())
                newstrip.resize(size_t(t * 4) * ys.overlap[iy] * wide * 3);
            for (size_t ix = 0; ix < xs.start.size(); ix++) {
                int th = ys.length[iy] / 16, tw = xs.length[ix] / 16, y0 = ys.start[iy] / 16,
                    x0 = xs.start[ix] / 16;
                std::vector<float> z(size_t(t) * th * tw * 24);
                for (int tt = 0; tt < t; tt++)
                    for (int y = 0; y < th; y++)
                        for (int x = 0; x < tw; x++)
                            for (int c = 0; c < 24; c++)
                                z[(((size_t)tt * th + y) * tw + x) * 24 + c] =
                                    latent[(((size_t)tt * h + y0 + y) * width + x0 + x) * 24 + c];
                auto pixel = decode_tile(w, z, t, th, tw);
                int ph = th * 16, pw = tw * 16, ox = ix ? xs.overlap[ix - 1] : 0;
                for (int tt = 0; tt < t * 4; tt++)
                    for (int y = 0; y < ph; y++)
                        for (int x = 0; x < pw; x++)
                            for (int c = 0; c < 3; c++) {
                                size_t at = (((size_t)tt * ph + y) * pw + x) * 3 + c;
                                float v = pixel[at];
                                if (y < overlap_y) {
                                    v = blend_half(strip[(((size_t)tt * overlap_y + y) * wide +
                                                          xs.start[ix] + x) *
                                                             3 +
                                                         c],
                                                   v, y, overlap_y);
                                }
                                if (x < ox) {
                                    v = blend_half(left[(((size_t)tt * ph + y) * ox + x) * 3 + c],
                                                   v, x, ox);
                                }
                                pixel[at] = v;
                            }
                int keepw = pw - (ix + 1 < xs.start.size() ? xs.overlap[ix] : 0),
                    keeph = ph - (iy + 1 < ys.start.size() ? ys.overlap[iy] : 0);
                if (ix + 1 < xs.start.size()) {
                    int tail = xs.overlap[ix];
                    left.resize(size_t(t * 4) * ph * tail * 3);
                    for (int tt = 0; tt < t * 4; tt++)
                        for (int y = 0; y < ph; y++)
                            std::copy_n(pixel.data() + (((size_t)tt * ph + y) * pw + pw - tail) * 3,
                                        tail * 3, left.data() + ((size_t)tt * ph + y) * tail * 3);
                }
                for (int tt = 0; tt < t * 4; tt++)
                    for (int y = 0; y < keeph; y++)
                        std::copy_n(
                            pixel.data() + ((size_t)tt * ph + y) * pw * 3, keepw * 3,
                            canvas.data() +
                                (((size_t)tt * height + ys.start[iy] + y) * wide + xs.start[ix]) *
                                    3);
                if (iy + 1 < ys.start.size()) {
                    int tail = ys.overlap[iy];
                    for (int tt = 0; tt < t * 4; tt++)
                        for (int y = 0; y < tail; y++)
                            std::copy_n(pixel.data() + ((size_t)tt * ph + ph - tail + y) * pw * 3,
                                        keepw * 3,
                                        newstrip.data() +
                                            (((size_t)tt * tail + y) * wide + xs.start[ix]) * 3);
                }
                std::cerr << "H3 VAE tile " << iy + 1 << "," << ix + 1 << "\n";
            }
            strip = std::move(newstrip);
        }
        return canvas;
    }
    void decode(const Tensor &video, int t, int h, int width, int requested, const h3_callbacks &cb,
                const fs::path &dump) {
        struct VendorGuard {
            Gpu &gpu;
            bool previous;
            ~VendorGuard() {
                gpu.clear_weights();
                gpu.vendor = previous;
            }
        } guard{g, g.vendor};
        // Retain the decoder's FP16 weights across spatial/temporal tiles.
        // Gpu::weight_half caps the checkpoint cache and allocations enforce
        // the same process budget; each model evaluation restores this scope.
        g.clear_weights();
        if (vae_hipblas) {
            if (!g.blas)
                require(cublasewCreate(&g.blas, g.stream) == 0, "VAE GEMM requires hipBLAS");
            g.vendor = true;
        }
        Weights w(root / "vae/minimax_h3_video_vae_fp16.safetensors");
        auto latent = g.download(video), mean = w.floats("latents_mean"),
             stddev = w.floats("latents_std");
        for (size_t i = 0; i < latent.size(); i++)
            latent[i] = latent[i] * stddev[i % 24] + mean[i % 24];
        int padded = (5 - (t + 3) % 5) % 5, chunks = (t + 3 + padded) / 5 - 1;
        if (chunks < 1) {
            padded += 5;
            chunks++;
        }
        std::vector<float> last(latent.end() - size_t(h) * width * 24, latent.end());
        for (int i = 0; i < padded; i++)
            latent.insert(latent.end(), last.begin(), last.end());
        const size_t pixels = size_t(h * 16) * width * 16 * 3;
        std::vector<float> overlap;
        int emitted = 0;
        auto emit = [&](const float *frame) {
            if (emitted >= requested)
                return;
            g.poll();
            std::vector<unsigned char> rgb(pixels);
            const float m[] = {.485f, .456f, .406f}, s[] = {.229f, .224f, .225f};
            std::vector<float> values(pixels);
            for (size_t i = 0; i < pixels; i++) {
                values[i] = std::clamp(frame[i] * s[i % 3] + m[i % 3], 0.f, 1.f);
                rgb[i] = static_cast<unsigned char>(std::nearbyint(values[i] * 255));
            }
            if (!dump.empty()) {
                char name[64];
                std::snprintf(name, sizeof(name), "frame_%03d.f32", emitted);
                std::ofstream file(dump / name, std::ios::binary);
                file.write(reinterpret_cast<char *>(values.data()), values.size() * 4);
                require(bool(file), "frame dump write");
            }
            require(!cb.frame || cb.frame(emitted, width * 16, h * 16, rgb.data(), cb.user) == 0,
                    "frame callback cancelled");
            emitted++;
        };
        for (int i = 0; i < chunks; i++) {
            int count = std::min(7, t + padded - i * 5);
            std::vector<float> clip(latent.begin() + size_t(i * 5) * h * width * 24,
                                    latent.begin() + size_t(i * 5 + count) * h * width * 24);
            auto frames = decode_spatial(w, clip, count, h, width);
            int first = std::min(20, count * 4) - 3;
            if (!overlap.empty())
                for (int f = 0; f < int(overlap.size() / pixels) && f < first; f++)
                    for (size_t j = 0; j < pixels; j++) {
                        auto &v = frames[size_t(f + 3) * pixels + j];
                        v = blend_half(overlap[size_t(f) * pixels + j], v, f, 5);
                    }
            for (int f = 0; f < first; f++)
                emit(frames.data() + size_t(f + 3) * pixels);
            int tail = count * 4 - 23;
            overlap.clear();
            if (tail > 0)
                overlap.assign(frames.begin() + 23 * pixels, frames.end());
            if (i + 1 == chunks)
                for (int f = 0; f < tail; f++)
                    emit(overlap.data() + size_t(f) * pixels);
        }
        require(emitted == requested, "VAE frame count mismatch");
    }
};
} // namespace h3
