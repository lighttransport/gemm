#include "host.hpp"
#include "hv15_native.h"
#include "tokenizer.hpp"
#include <iostream>
using namespace hv15n;
int main(int argc, char **argv) {
    try {
        require(product({21, 53, 30, 32}) == 1068480, "latent dimensions");
        bool threw = false;
        try {
            product({INT32_MAX, INT32_MAX, INT32_MAX});
        } catch (const std::exception &) {
            threw = true;
        }
        require(threw, "size overflow accepted");
        require(half_float(0x3c00) == 1.f && half_float(0xc000) == -2.f && std::isinf(half_float(0x7c00)),
                "half decode");
        std::vector<uint16_t> half_values(65536);
        std::vector<float> decoded(65536);
        for (int i = 0; i < 65536; ++i)
            half_values[i] = uint16_t(i);
        decode_f16(half_values.data(), decoded.data(), decoded.size());
        for (int i = 0; i < 65536; ++i) {
            int exp = (i >> 10) & 31, mantissa = i & 1023;
            float reference = exp == 31 ? (mantissa ? NAN : INFINITY)
                              : exp     ? std::ldexp(1.f + float(mantissa) / 1024.f, exp - 15)
                                        : std::ldexp(float(mantissa), -24);
            if (i & 0x8000)
                reference = -reference;
            require(std::isnan(reference) ? std::isnan(decoded[i]) && std::isnan(half_float(uint16_t(i)))
                                          : decoded[i] == reference && half_float(uint16_t(i)) == reference &&
                                                std::signbit(decoded[i]) == std::signbit(reference),
                    "FP16 exhaustive conversion mismatch");
        }
        auto times = schedule(12, 7.f);
        require(times.front() == 1.f && times.back() == 0.f, "schedule endpoints");
        for (size_t i = 1; i < times.size(); i++)
            require(times[i] < times[i - 1], "schedule monotonicity");
        require(relative_bucket(4, 4) == 0 && relative_bucket(4, 5) == 17 && relative_bucket(4, 3) == 1 &&
                    relative_bucket(0, 10000) == 31,
                "T5 relative buckets");
        auto texts = glyph_texts("A sign \"Hello\" beside “日本語” and \"Hello\".");
        require(texts == std::vector<std::string>({"Hello", "日本語"}), "glyph deduplication");
        require(glyph_texts("Unmatched \" then “Hello”") == std::vector<std::string>({"Hello"}),
                "unmatched quote swallowed a later glyph");
        require(glyph_texts("Unmatched \"\nthen \"Hello\"") == std::vector<std::string>({"Hello"}),
                "multiline quote swallowed a later glyph");
        auto tokens = byt5_tokens("Show \"A\"");
        std::string formatted = "Text \"A\". ";
        require(tokens.size() == formatted.size() + 1 && tokens.back() == 1, "ByT5 EOS");
        for (size_t i = 0; i < formatted.size(); i++)
            require(tokens[i] == int(static_cast<unsigned char>(formatted[i])) + 3, "ByT5 byte encoding");
        require(byt5_tokens("Smile naturally").empty(), "unquoted prompt has glyph tokens");
        require(byt5_tokens("\"" + std::string(400, 'a') + "\"").size() == 256, "ByT5 truncation");
        hv15n_request r;
        hv15n_request_defaults(&r);
        r.task = "t2v";
        r.prompt = "A person smiles";
        char error[512];
        require(hv15n_set_aotriton_bridge(nullptr, "missing", error, sizeof(error)) != 0,
                "null attention-provider context accepted");
        require(hv15n_validate(&r, error, sizeof(error)) == 0, "T2V rejected");
        r.preset = "fast12";
        require(hv15n_validate(&r, error, sizeof(error)) != 0, "fast12 T2V accepted");
        r.preset = "quality";
        r.image = "portrait.png";
        require(hv15n_validate(&r, error, sizeof(error)) != 0, "T2V portrait accepted");
        r.task = "i2v";
        r.vision_pixels = "pixels.f32";
        require(hv15n_validate(&r, error, sizeof(error)) == 0, "I2V rejected");
        r.frames = 121;
        require(hv15n_validate(&r, error, sizeof(error)) != 0, "unsupported duration accepted");
        if (argc > 1) {
            Tokenizer tokenizer(argv[1]);
            auto ids = tokenizer.encode(qwen_prefix());
            require(!ids.empty() && ids[0] == 151644, "Qwen special token");
            for (auto s : {"A person smiles.", "日本語の動画。", "Hello 123\n世界", "don't blink"})
                require(!tokenizer.encode(s).empty(), "Qwen tokenization empty");
        }
        std::cout << "PASS sizes, FP16, schedule, T5 buckets, glyphs, requests and tokenizer\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
