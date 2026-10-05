// SPDX-License-Identifier: MIT
#include "runtime.hpp"
int main(int argc, char **argv) {
    try {
        using namespace h3;
        require(
            argc >= 5,
            "component_probe MODEL qwen PROMPT OUT | MODEL linear WEIGHTS PREFIX INPUT ROWS OUT");
        h3_config config;
        h3_config_defaults(&config);
        const char *conditioning = nullptr;
        while (argc >= 3) {
            std::string option = argv[argc - 2];
            if (option == "--aotriton-bridge")
                config.aotriton_bridge = argv[argc - 1];
            else if (option == "--conditioning")
                conditioning = argv[argc - 1];
            else if (option == "--vae-hipblas") {
                std::string value = argv[argc - 1];
                require(value == "0" || value == "1", "VAE backend must be 0 or 1");
                config.vae_hipblas = value == "1";
            } else if (option == "--vram-budget-mib") {
                config.vram_budget_mib = std::stoi(argv[argc - 1]);
            } else
                break;
            argc -= 2;
        }
        config.model_dir = argv[1];
        Engine engine(config);
        if (conditioning) {
            engine.conditioning = conditioning;
            Json receipt(engine.conditioning / "manifest.json");
            engine.variant = string(receipt.value.get(), "variant");
            engine.condition_text_rows = int(field(receipt.value.get(), "text_rows")->num);
            engine.condition_rows = int(field(receipt.value.get(), "condition_rows")->num);
        }
        auto &g = engine.g;
        if (std::string(argv[2]) == "qwen-rope") {
            require(argc == 5, "qwen-rope needs sequence length and output directory");
            g.dump(engine.qwen_angles(std::stoi(argv[3])), argv[4], "qwen_rotation");
        } else if (std::string(argv[2]) == "qwen") {
            require(argc == 5, "qwen probe arguments");
            engine.trace = argv[4];
            engine.encode(argv[3], argv[4]);
        } else if (std::string(argv[2]) == "refine") {
            require(argc == 5, "refine needs a native Qwen capture directory and output directory");
            Json meta(fs::path(argv[3]) / "qwen_hidden.json");
            auto shape = field(meta.value.get(), "shape");
            require(shape && shape->type == JSON_ARRAY && shape->arr.count == 3, "Qwen shape");
            int rows = int(shape->arr.items[1].num);
            auto x = g.upload(read_f32(fs::path(argv[3]) / "qwen_hidden.f32", size_t(rows) * 5120),
                              {rows, 5120});
            Weights weights(engine.checkpoint());
            engine.trace = argv[4];
            engine.refine(weights, x, argv[4]);
        } else if (std::string(argv[2]) == "vae") {
            require(argc == 6, "vae needs a THWC latent capture directory, output and frame count");
            Json meta(fs::path(argv[3]) / "latent_video.json");
            auto dims = field(meta.value.get(), "shape");
            require(dims && dims->type == JSON_ARRAY && dims->arr.count == 4 &&
                        dims->arr.items[3].num == 24,
                    "VAE latent shape");
            int t = int(dims->arr.items[0].num), h = int(dims->arr.items[1].num),
                w = int(dims->arr.items[2].num), frames = std::stoi(argv[5]);
            h3_request request{"probe", nullptr, nullptr, nullptr, w * 16, h * 16, frames, 2, 42};
            char error[1024];
            require(h3_validate(&request, error, sizeof(error)) == 0, error);
            require(t == (frames - 5) / 17 * 5 + 2, "VAE temporal geometry");
            auto video =
                g.upload(read_f32(fs::path(argv[3]) / "latent_video.f32", product({t, h, w, 24})),
                         {t, h, w, 24});
            fs::create_directories(argv[4]);
            engine.decode(video, t, h, w, frames, h3_callbacks{}, argv[4]);
            require(!g.vendor, "decoder GEMM backend state leaked into the next graph");
            require(g.packed_weights.empty() && g.cached_weight_bytes == 0,
                    "decoder checkpoint cache leaked into the next graph");
        } else if (std::string(argv[2]) == "linear" || std::string(argv[2]) == "norm") {
            require(argc == 9, "linear probe arguments");
            Weights weights(relative_file(argv[1], argv[3]));
            std::string prefix = argv[4];
            auto shape = weights.shape(prefix + ".weight");
            int features = shape.back();
            int rows = std::stoi(argv[6]);
            require(rows > 0 && rows <= 1024, "probe row count");
            auto input = g.upload(read_f32(argv[5], size_t(rows) * features), {rows, features});
            int kind = std::stoi(argv[8]);
            require(kind >= 0 && kind <= 2, "precision kind 0=f32,1=bf16,2=fp16");
            input = engine.rounded(input, kind);
            g.dump(input, argv[7], "input");
            auto output = std::string(argv[2]) == "norm"
                              ? engine.norm(&weights, prefix, input, kind)
                              : engine.linear(weights, prefix, input, kind, kind == 0);
            g.dump(output, argv[7], "output");
        } else if (std::string(argv[2]) == "dit-block" || std::string(argv[2]) == "dit-step" || std::string(argv[2]) == "dit-stream") {
            require(argc == 5, "dit-block probe needs capture directory and output directory");
            fs::path in = argv[3], out = argv[4];
            Json meta(in / "noise_video.json");
            auto dims = field(meta.value.get(), "shape");
            require(dims && dims->type == JSON_ARRAY && dims->arr.count == 4,
                    "THWC capture required");
            int t = int(dims->arr.items[0].num), h = int(dims->arr.items[1].num),
                w = int(dims->arr.items[2].num);
            auto capture = [&](const char *name) {
                Json info(in / (std::string(name) + ".json"));
                auto shape = field(info.value.get(), "shape");
                std::vector<int> d;
                for (int i = 0; i < shape->arr.count; i++)
                    d.push_back(int(shape->arr.items[i].num));
                return g.upload(read_f32(in / (std::string(name) + ".f32"), product(d)), d);
            };
            auto video = capture("noise_video"), audio = capture("noise_audio"),
                 text = capture("refined_text");
            audio.shape = {audio.rows(), 32};
            text.shape = {text.rows(), 5376};
            Weights weights(fs::path(argv[1]) /
                            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors");
            auto rotation = engine.dit_rope(weights, text.rows(), audio.rows() / 2, t, h, w);
            if (std::string(argv[2]) == "dit-step") {
                engine.trace = out;
                auto velocity =
                    engine.denoise(weights, text, video, audio, rotation, t, h, w, 0, 0, 1);
                g.dump(velocity[0], out, "video_velocity");
                g.dump(velocity[1], out, "audio_velocity");
                return 0;
            }
            auto emb = engine.time_embed(weights, 0, 0);
            auto previous = std::chrono::steady_clock::now();
            auto stamp = [&](const char *name) {
                g.check(cuStreamSynchronize(g.stream), name);
                auto now = std::chrono::steady_clock::now();
                std::cout << name
                          << " seconds=" << std::chrono::duration<double>(now - previous).count()
                          << std::endl;
                previous = now;
            };
            auto patches = g.empty({t * (h / 2) * (w / 2), 96});
            g.launch("h3_patch", int((video.count() + 255) / 256), 1, 1, 256, 1, 0, patches.pointer,
                     video.pointer, t, h, w);
            auto vi = engine.rounded(engine.linear(weights, "video_patch_proj", patches, 1, true)),
                 au = engine.rounded(engine.linear(weights, "audio_patch_proj", audio, 1, true));
            auto x = engine.pack_rows({&text, &au, &vi});
            g.clear_weights();
            stamp("embedding");
            auto mod = engine.linear(weights, "blocks.0.adaln_proj.linear", emb, 0, true);
            auto z = engine.modulate(engine.norm(&weights, "blocks.0.norm1", x), mod, text.rows(),
                                     audio.rows(), 0);
            stamp("norm_and_modulate");
            if (std::string(argv[2]) == "dit-stream") {
                auto delta = engine.self_attention(weights, "blocks.0.attn", z, 56, 128, 1, &rotation, 48);
                engine.gated(x, delta, mod, text.rows(), audio.rows(), 2);
                delta = {};
                z = engine.modulate(engine.norm(&weights, "blocks.0.norm2", x), mod, text.rows(), audio.rows(), 3);
                delta = engine.ffn(weights, "blocks.0.mlp", z);
                engine.gated(x, delta, mod, text.rows(), audio.rows(), 5);
                stamp("streamed_attention_and_ffn");
                g.dump(x, out, "dit_block_0");
                fs::create_directories(out);
                std::ofstream file(out / "metrics.json");
                file << "{\"scope\":\"one_block_memory_probe\",\"packed_rows\":" << x.rows()
                     << ",\"convrot_hipblaslt_calls\":" << engine.convrot_lt_calls
                     << ",\"gpu\":" << g.metrics() << "}\n";
                return 0;
            }
            auto packed = engine.linear(weights, "blocks.0.attn.qkv_proj", z);
            stamp("qkv");
            auto q = g.empty({x.rows(), 7168}), k = g.empty(q.shape), v = g.empty(q.shape);
            g.launch("h3_qkv", int((q.count() + 255) / 256), 1, 1, 256, 1, 0, q.pointer, k.pointer,
                     v.pointer, packed.pointer, x.rows(), 56, 128, 0);
            packed = {};
            q = engine.head_norm(&weights, "blocks.0.attn.q_norm", q, 128);
            k = engine.head_norm(&weights, "blocks.0.attn.k_norm", k, 128);
            engine.rope(q, rotation, 56, 128, 48, 1);
            engine.rope(k, rotation, 56, 128, 48, 1);
            stamp("qk_norm_rope");
            if (x.rows() <= 256) {
                g.dump(q, out, "dit_q");
                g.dump(k, out, "dit_k");
                g.dump(v, out, "dit_v");
            }
            auto attended = engine.attention(q, k, v, 56, 56, 128);
            if (x.rows() <= 256)
                g.dump(attended, out, "dit_context");
            stamp("attention");
            q = {};
            k = {};
            v = {};
            auto delta = engine.linear(weights, "blocks.0.attn.out_proj", attended);
            attended = {};
            engine.gated(x, delta, mod, text.rows(), audio.rows(), 2);
            delta = {};
            stamp("out_projection");
            z = engine.modulate(engine.norm(&weights, "blocks.0.norm2", x), mod, text.rows(),
                                audio.rows(), 3);
            delta = engine.ffn(weights, "blocks.0.mlp", z);
            engine.gated(x, delta, mod, text.rows(), audio.rows(), 5);
            stamp("ffn");
            g.dump(x, out, "dit_block_0");
            fs::create_directories(out);
            std::ofstream file(out / "metrics.json");
            file << g.metrics() << "\n";
        } else
            throw std::runtime_error("unknown H3 component");
        std::cout << g.metrics() << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
