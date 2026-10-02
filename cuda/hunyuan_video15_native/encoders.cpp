#include "models.hpp"
namespace hv15n {
static Tensor embedding(Gpu &g, Weights &w, const std::string &name, const std::vector<int> &tokens) {
    auto shape = w.shape(name);
    require(shape.size() == 2, "embedding shape");
    std::vector<float> values(size_t(tokens.size()) * shape[1]);
    auto type = w.dtype(name);
    require(type == "F16" || type == "F32" || type == "BF16", "embedding dtype");
    for (size_t i = 0; i < tokens.size(); i++) {
        require(tokens[i] >= 0 && tokens[i] < shape[0], "embedding index outside vocabulary");
        for (int c = 0; c < shape[1]; c++) {
            size_t at = size_t(tokens[i]) * shape[1] + c;
            if (type == "F16")
                values[i * shape[1] + c] = half_float(static_cast<const uint16_t *>(w.data(name))[at]);
            else if (type == "F32")
                values[i * shape[1] + c] = static_cast<const float *>(w.data(name))[at];
            else {
                uint32_t bits = uint32_t(static_cast<const uint16_t *>(w.data(name))[at]) << 16;
                std::memcpy(&values[i * shape[1] + c], &bits, sizeof(bits));
            }
        }
    }
    return g.upload(values, {int(tokens.size()), shape[1]});
}
Tensor qwen(Gpu &g, Weights &w, const Tokenizer &tokenizer, const std::string &prompt) {
    auto prefix = qwen_prefix();
    int crop = int(tokenizer.encode(prefix).size());
    auto tokens = tokenizer.encode(prefix + prompt + "<|im_end|>\n<|im_start|>assistant\n");
    if (tokens.size() > size_t(crop + 1000))
        tokens.resize(size_t(crop + 1000));
    auto x = embedding(g, w, "model.embed_tokens.weight", tokens);
    fs::path diagnostic = std::getenv("HV15N_DIAGNOSTIC_DIR") ? std::getenv("HV15N_DIAGNOSTIC_DIR") : "";
    if (!diagnostic.empty())
        g.dump(x, diagnostic, "qwen_embedding");
    for (int i = 0; i < 26; i++) {
        auto p = "model.layers." + std::to_string(i);
        auto z = g.norm(w, p + ".input_layernorm", x, 1);
        auto q = g.linear(w, p + ".self_attn.q_proj", z, true),
             k = g.linear(w, p + ".self_attn.k_proj", z, true),
             v = g.linear(w, p + ".self_attn.v_proj", z, true);
        g.rotary(q, 28, 0);
        g.rotary(k, 4, 0);
        auto attn = g.attention(q, k, v, 28, 4, 1, 1, true);
        if (i == 0 && !diagnostic.empty()) {
            g.dump(z, diagnostic, "qwen_norm");
            g.dump(q, diagnostic, "qwen_q");
            g.dump(k, diagnostic, "qwen_k");
            g.dump(v, diagnostic, "qwen_v");
            g.dump(attn, diagnostic, "qwen_attention");
        }
        auto out = g.linear(w, p + ".self_attn.o_proj", attn, true);
        x = g.op(x, 1, &out);
        if (i == 0 && !diagnostic.empty())
            g.dump(x, diagnostic, "qwen_layer0");
        z = g.norm(w, p + ".post_attention_layernorm", x, 1);
        auto up = g.linear(w, p + ".mlp.up_proj", z, true),
             gate = g.op(g.linear(w, p + ".mlp.gate_proj", z, true), 3);
        auto product = g.op(up, 2, &gate);
        out = g.linear(w, p + ".mlp.down_proj", product, true);
        x = g.op(x, 1, &out);
    }
    return g.clone(g.rows(x, crop, x.rows() - crop));
}
Tensor siglip(Gpu &g, Weights &w, const fs::path &pixels) {
    fs::path diagnostic = std::getenv("HV15N_DIAGNOSTIC_DIR") ? std::getenv("HV15N_DIAGNOSTIC_DIR") : "";
    auto input = g.upload(read_f32(pixels, 3 * 384 * 384), {3, 384, 384});
    auto patches = g.empty({729, 3 * 14 * 14});
    g.launch("vision_patches", int((patches.count() + 255) / 256), 1, 1, 256, 1, 0, patches.pointer,
             input.pointer, 27, 384, 14);
    auto x = g.linear(w, "vision_model.embeddings.patch_embedding", patches, true);
    auto positions = g.weight(w, "vision_model.embeddings.position_embedding.weight");
    x = g.op(x, 1, &positions);
    if (!diagnostic.empty()) {
        g.dump(patches, diagnostic, "siglip_patches");
        g.dump(x, diagnostic, "siglip_embedding");
    }
    for (int i = 0; i < 27; i++) {
        auto p = "vision_model.encoder.layers." + std::to_string(i);
        auto z = g.norm(w, p + ".layer_norm1", x, 0, 1.e-6f);
        auto q = g.linear(w, p + ".self_attn.q_proj", z, true),
             k = g.linear(w, p + ".self_attn.k_proj", z, true),
             v = g.linear(w, p + ".self_attn.v_proj", z, true);
        auto out = g.linear(w, p + ".self_attn.out_proj", g.attention(q, k, v, 16, 16, 0, 1, true), true);
        x = g.op(x, 1, &out);
        if (i == 0 && !diagnostic.empty())
            g.dump(x, diagnostic, "siglip_after_attention");
        z = g.norm(w, p + ".layer_norm2", x, 0, 1.e-6f);
        out = g.linear(w, p + ".mlp.fc2", g.op(g.linear(w, p + ".mlp.fc1", z, true), 4), true);
        x = g.op(x, 1, &out);
        if (i == 0 && !diagnostic.empty())
            g.dump(x, diagnostic, "siglip_layer0");
    }
    return g.norm(w, "vision_model.post_layernorm", x, 0, 1.e-6f);
}
Tensor byt5(Gpu &g, Weights &w, const std::string &prompt) {
    auto tokens = byt5_tokens(prompt);
    if (tokens.empty())
        return {};
    auto x = embedding(g, w, "shared.weight", tokens);
    fs::path diagnostic = std::getenv("HV15N_DIAGNOSTIC_DIR") ? std::getenv("HV15N_DIAGNOSTIC_DIR") : "";
    if (!diagnostic.empty())
        g.dump(x, diagnostic, "byt5_embedding");
    auto bias = g.weight(w, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
    for (int i = 0; i < 12; i++) {
        auto p = "encoder.block." + std::to_string(i);
        auto z = g.norm(w, p + ".layer.0.layer_norm", x, 1);
        auto a = p + ".layer.0.SelfAttention";
        auto q = g.linear(w, a + ".q", z, true), k = g.linear(w, a + ".k", z, true),
             v = g.linear(w, a + ".v", z, true);
        if (i == 0 && !diagnostic.empty()) {
            g.dump(z, diagnostic, "byt5_norm");
            g.dump(q, diagnostic, "byt5_q");
            g.dump(k, diagnostic, "byt5_k");
            g.dump(v, diagnostic, "byt5_v");
        }
        auto out = g.linear(w, a + ".o", g.attention(q, k, v, 6, 6, 0, 1, true, &bias, 1.f), true);
        x = g.op(x, 1, &out);
        if (i == 0 && !diagnostic.empty())
            g.dump(x, diagnostic, "byt5_layer0");
        z = g.norm(w, p + ".layer.1.layer_norm", x, 1);
        a = p + ".layer.1.DenseReluDense";
        auto gate = g.op(g.linear(w, a + ".wi_0", z, true), 4), up = g.linear(w, a + ".wi_1", z, true);
        auto product = g.op(gate, 2, &up);
        out = g.linear(w, a + ".wo", product, true);
        x = g.op(x, 1, &out);
    }
    return g.norm(w, "encoder.final_layer_norm", x, 1);
}
} // namespace hv15n
