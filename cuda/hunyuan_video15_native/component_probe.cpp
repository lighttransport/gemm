#include "models.hpp"
#include <iostream>
using namespace hv15n;
int main(int argc, char **argv) {
    try {
        require(argc >= 5, "component_probe MODEL COMPONENT INPUT OUTPUT [T H W]");
        fs::path model = argv[1], out = argv[4];
        Json manifest(model / "model.json");
        auto root = manifest.value.get();
        auto component = [&](const char *key) {
            return relative_file(model, string(field(root, "components"), key));
        };
#ifdef HV15N_ROCM
        Gpu g(0, 14336, false, false);
#else
        Gpu g(0, 14336, false, true);
#endif
        std::string name = argv[2];
        Tensor result;
        bool video = false;
        if (name == "qwen") {
            Tokenizer tokenizer(component("tokenizer"));
            Weights w(component("qwen"));
            result = qwen(g, w, tokenizer, argv[3]);
        } else if (name == "byt5") {
            Weights w(component("byt5"));
            result = byt5(g, w, argv[3]);
            if (!result.pointer)
                result = g.upload(std::vector<float>(1472), {1, 1472});
        } else if (name == "siglip") {
            Weights w(component("vision"));
            result = siglip(g, w, argv[3]);
        } else if (name == "dit") {
            require(argc == 8 || argc == 9, "DiT probe needs THW dimensions and optional profile");
            fs::path input = argv[3];
            int t = std::stoi(argv[5]), h = std::stoi(argv[6]), width = std::stoi(argv[7]);
            auto capture = [&](const char *key) {
                Json meta(input / (std::string(key) + ".json"));
                auto shape = field(meta.value.get(), "shape");
                require(shape && shape->type == JSON_ARRAY && shape->arr.count == 3,
                        "conditioning probe shape");
                int rows = int(shape->arr.items[1].num), channels = int(shape->arr.items[2].num);
                return g.upload(read_f32(input / (std::string(key) + ".f32"), product({rows, channels})),
                                {rows, channels});
            };
            auto latent =
                g.upload(read_f32(input / "latent_thwc.f32", product({t, h, width, 32})), {t, h, width, 32});
            g.dump(latent, out, "noise_input", true);
            auto condition = g.upload(std::vector<float>(product({t, h, width, 33})), {t, h, width, 33});
            auto text = capture("qwen_hidden");
            Tensor glyph, vision;
            if (fs::exists(input / "byt5_hidden.f32"))
                glyph = capture("byt5_hidden");
            if (fs::exists(input / "siglip_hidden.f32"))
                vision = capture("siglip_hidden");
            std::string profile = argc == 9 ? argv[8] : "fast12_i2v";
            Weights w(relative_file(model, string(field(root, "checkpoints"), profile.c_str())));
            result = dit(g, w, latent, condition, text, glyph, vision, 1000.f, schedule(12, 7.f)[1] * 1000.f);
            video = true;
        } else if (name == "vae_encode" || name == "vae_decode") {
            require(argc == 8, "VAE probe needs THW dimensions");
            int t = std::stoi(argv[5]), h = std::stoi(argv[6]), width = std::stoi(argv[7]);
            bool encode = name == "vae_encode";
            int channels = encode ? 3 : 32;
            auto input =
                g.upload(read_f32(argv[3], product({t, h, width, channels})), {t, h, width, channels});
            Weights w(component("vae"));
            result = vae_tiled(g, w, input, encode);
            video = true;
        } else {
            throw std::runtime_error("unknown component");
        }
        std::string capture = name == "dit"          ? "dit_first"
                              : name == "vae_encode" ? "vae_encoded"
                              : name == "vae_decode" ? "vae_decoded"
                                                     : name + "_hidden";
        g.dump(result, out, capture, video);
        std::cout << g.metrics() << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
