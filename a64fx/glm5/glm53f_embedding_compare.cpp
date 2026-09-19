#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "llama-model-loader.h"
#include "ggml.h"
#include "../../common/glm53f_safetensors.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static float bf16_to_f32(uint16_t x) {
    uint32_t bits = (uint32_t) x << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint64_t fnv1a(const std::vector<float> & v) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (float x : v) {
        const auto * p = reinterpret_cast<const unsigned char *>(&x);
        for (size_t i = 0; i < sizeof(x); ++i) {
            h ^= p[i];
            h *= UINT64_C(1099511628211);
        }
    }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s FIRST_GGUF_SHARD SAFETENSORS_DIR TOKEN_ID [TOKEN_ID ...]\n", argv[0]);
        return 2;
    }
    try {
        std::vector<std::string> splits;
        llama_model_loader loader(nullptr, nullptr, nullptr, argv[1], splits, nullptr,
                                   LLAMA_LOAD_MODE_NONE, true, true, false, nullptr, nullptr);
        const auto * ew = loader.get_weight("token_embd.weight");
        if (!ew) throw std::runtime_error("missing GGUF token_embd.weight");
        const auto * traits = ggml_get_type_traits(ew->tensor->type);
        if (!traits || !traits->to_float) throw std::runtime_error("missing GGUF dequantizer");
        const size_t row_bytes = ggml_row_size(ew->tensor->type, 4096);
        glm53f_st_context * st = glm53f_st_open(argv[2]);
        if (!st) throw std::runtime_error("cannot open safetensors model");
        std::vector<unsigned char> raw(row_bytes);
        std::vector<float> gguf(4096), safe(4096);
        std::vector<uint16_t> bf16(safe.size());
        int failed = 0;
        for (int arg = 3; arg < argc; ++arg) {
            const int token = std::atoi(argv[arg]);
            if (token < 0 || token >= 154880) return 2;
            loader.load_data_range(*ew, (size_t) token * row_bytes, row_bytes, raw.data());
            traits->to_float(raw.data(), gguf.data(), gguf.size());
            const int rc = glm53f_st_read(st, "model.language_model.embed_tokens.weight",
                                          (size_t) token * safe.size() * sizeof(uint16_t),
                                          bf16.data(), bf16.size() * sizeof(uint16_t));
            if (rc) throw std::runtime_error("cannot read safetensors embedding row");
            for (size_t i = 0; i < safe.size(); ++i) safe[i] = bf16_to_f32(bf16[i]);
            double d2 = 0.0, n2 = 0.0;
            float max_abs = 0.0f;
            for (size_t i = 0; i < safe.size(); ++i) {
                const float d = gguf[i] - safe[i];
                d2 += (double) d * d;
                n2 += (double) safe[i] * safe[i];
                max_abs = std::fmax(max_abs, std::fabs(d));
            }
            const double rel = n2 > 0.0 ? std::sqrt(d2 / n2) : d2;
            const bool match = rel <= 1e-6;
            std::printf("GLM53F_EMBED_COMPARE token=%d gguf_fnv=%016llx safetensors_fnv=%016llx rel_l2=%.9g max_abs=%.9g %s\n",
                        token, (unsigned long long) fnv1a(gguf),
                        (unsigned long long) fnv1a(safe), rel, max_abs,
                        match ? "MATCH" : "DIFFERENT");
            failed |= !match;
        }
        glm53f_st_close(st);
        return failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "GLM53F_EMBED_COMPARE FAIL %s\n", e.what());
        return 1;
    }
}
