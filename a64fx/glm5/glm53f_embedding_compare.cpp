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

static bool compare_vectors(const char * tag, int token, const std::vector<float> & gguf,
                            const std::vector<float> & safe) {
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
    std::printf("GLM53F_%s_COMPARE token=%d gguf_fnv=%016llx safetensors_fnv=%016llx rel_l2=%.9g max_abs=%.9g %s\n",
                tag, token, (unsigned long long) fnv1a(gguf),
                (unsigned long long) fnv1a(safe), rel, max_abs,
                match ? "MATCH" : "DIFFERENT");
    return match;
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
        const auto * hw = loader.get_weight("output.weight");
        const auto * htraits = hw ? ggml_get_type_traits(hw->tensor->type) : nullptr;
        if (!hw || !htraits || !htraits->to_float) throw std::runtime_error("missing GGUF output.weight dequantizer");
        const size_t head_row_bytes = ggml_row_size(hw->tensor->type, 4096);
        const auto * nw = loader.get_weight("output_norm.weight");
        const auto * ntraits = nw ? ggml_get_type_traits(nw->tensor->type) : nullptr;
        if (!nw || (nw->tensor->type != GGML_TYPE_F32 && (!ntraits || !ntraits->to_float))) {
            throw std::runtime_error("missing GGUF output_norm.weight dequantizer");
        }
        glm53f_st_context * st = glm53f_st_open(argv[2]);
        if (!st) throw std::runtime_error("cannot open safetensors model");
        std::vector<unsigned char> raw(row_bytes);
        std::vector<unsigned char> head_raw(head_row_bytes);
        std::vector<float> gguf(4096), safe(4096);
        std::vector<float> head_gguf(4096), head_safe(4096);
        std::vector<float> norm_gguf(4096), norm_safe(4096);
        const size_t norm_bytes = ggml_row_size(nw->tensor->type, 4096);
        std::vector<unsigned char> norm_raw(norm_bytes);
        std::vector<uint16_t> bf16(safe.size());
        int failed = 0;
        if (nw->tensor->ne[0] != 4096) throw std::runtime_error("unexpected GGUF output norm width");
        loader.load_data_range(*nw, 0, norm_bytes, norm_raw.data());
        if (nw->tensor->type == GGML_TYPE_F32) std::memcpy(norm_gguf.data(), norm_raw.data(), 4096 * sizeof(float));
        else ntraits->to_float(norm_raw.data(), norm_gguf.data(), norm_gguf.size());
        std::vector<uint16_t> norm_bf16(norm_safe.size());
        if (glm53f_st_read(st, "model.language_model.norm.weight", 0,
                           norm_bf16.data(), norm_bf16.size() * sizeof(uint16_t))) {
            throw std::runtime_error("cannot read safetensors output norm");
        }
        for (size_t i = 0; i < norm_safe.size(); ++i) norm_safe[i] = bf16_to_f32(norm_bf16[i]);
        failed |= !compare_vectors("NORM", -1, norm_gguf, norm_safe);
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
            failed |= !compare_vectors("EMBED", token, gguf, safe);

            loader.load_data_range(*hw, (size_t) token * head_row_bytes, head_row_bytes, head_raw.data());
            htraits->to_float(head_raw.data(), head_gguf.data(), head_gguf.size());
            if (glm53f_st_read(st, "lm_head.weight",
                               (size_t) token * head_safe.size() * sizeof(uint16_t),
                               bf16.data(), bf16.size() * sizeof(uint16_t))) {
                throw std::runtime_error("cannot read safetensors output head row");
            }
            for (size_t i = 0; i < head_safe.size(); ++i) head_safe[i] = bf16_to_f32(bf16[i]);
            failed |= !compare_vectors("HEAD", token, head_gguf, head_safe);
        }
        glm53f_st_close(st);
        return failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "GLM53F_EMBED_COMPARE FAIL %s\n", e.what());
        return 1;
    }
}
