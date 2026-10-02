#ifndef PIXAL3D_SIGLIP_HPP
#define PIXAL3D_SIGLIP_HPP
#include "model/te/clip.hpp"
#include "core/hv15_cuda_math.h"
#include <unordered_set>
/* SigLIP SO400M/14: no CLS or pre-LN, biased patch projection, GELU-tanh,
 * 27 layers, epsilon 1e-6, post-LN token features (not pooled features). */
struct HV15SiglipMLP : public CLIPMLP {
    HV15SiglipMLP() : CLIPMLP(1152, 4304) { use_gelu = true; }
};
struct HV15SiglipLayer : public CLIPLayer {
    HV15SiglipLayer() : CLIPLayer(1152, 16, 4304) {
        blocks["layer_norm1"] = std::make_shared<LayerNorm>(1152, 1e-6f);
        blocks["layer_norm2"] = std::make_shared<LayerNorm>(1152, 1e-6f);
        blocks["mlp"] = std::make_shared<HV15SiglipMLP>();
    }
};
struct HV15SiglipEmbeddings : public GGMLBlock {
    void init_params(ggml_context *ctx, const String2TensorStorage& storage = {},
                     const std::string prefix = "") override {
        params["patch_embedding.weight"] = ggml_new_tensor_4d(ctx,
            get_type(prefix + "patch_embedding.weight", storage, GGML_TYPE_F16), 14, 14, 3, 1152);
        params["patch_embedding.bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1152);
        params["position_embedding.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1152, 729);
    }
    ggml_tensor *forward(GGMLRunnerContext *ctx, ggml_tensor *pixels) {
        GGML_ASSERT(pixels->ne[0] == 384 && pixels->ne[1] == 384);
        /* The generic convolution rounds im2col to F16. SigLIP's official
         * FP32 activation profile preserves the normalized pixels here. */
        auto g = ctx->ggml_ctx;
        auto weight = params["patch_embedding.weight"];
        auto patches = ggml_im2col(g, weight, pixels, 14, 14, 0, 0, 1, 1,
                                   true, GGML_TYPE_F32);
        auto x = ggml_mul_mat(g,
            ggml_reshape_2d(g, patches, 14 * 14 * 3, 729 * pixels->ne[3]),
            ggml_cast(g, ggml_reshape_2d(g, weight, 14 * 14 * 3, 1152), GGML_TYPE_F32));
        x = ggml_reshape_4d(g, x, 27, 27, pixels->ne[3], 1152);
        x = ggml_cont(g, ggml_permute(g, x, 0, 1, 3, 2));
        x = ggml_add(g, x, ggml_reshape_4d(g, params["patch_embedding.bias"], 1, 1, 1152, 1));
        x = ggml_reshape_3d(ctx->ggml_ctx, x, 729, 1152, pixels->ne[3]);
        x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));
        return ggml_add(ctx->ggml_ctx, x, params["position_embedding.weight"]);
    }
};
struct HV15SiglipModel : public GGMLBlock {
    HV15SiglipModel() {
        blocks["embeddings"] = std::make_shared<HV15SiglipEmbeddings>();
        for (int i = 0; i < 27; ++i)
            blocks["encoder.layers." + std::to_string(i)] = std::make_shared<HV15SiglipLayer>();
        blocks["post_layernorm"] = std::make_shared<LayerNorm>(1152, 1e-6f);
    }
    static void require_f32_accumulation(ggml_tensor *tensor, std::unordered_set<ggml_tensor *>& visited) {
        if (!tensor || !visited.insert(tensor).second) return;
        if (tensor->op == GGML_OP_MUL_MAT) ggml_prec_set_acc(tensor, GGML_PREC_F32);
        for (auto source : tensor->src) require_f32_accumulation(source, visited);
    }
    ggml_tensor *forward(GGMLRunnerContext *ctx, ggml_tensor *pixels) {
        auto x = std::dynamic_pointer_cast<HV15SiglipEmbeddings>(blocks["embeddings"])->forward(ctx, pixels);
        /* Head dimension 72: bounded dense vision attention; video uses flash attention. */
        ctx->flash_attn_enabled = false;
        for (int i = 0; i < 27; ++i) {
            x = std::dynamic_pointer_cast<HV15SiglipLayer>(blocks["encoder.layers." + std::to_string(i)])->forward(ctx, x);
            sd::ggml_graph_cut::mark_graph_cut(x, "hv15.siglip.layers." + std::to_string(i), "x");
        }
        x = std::dynamic_pointer_cast<LayerNorm>(blocks["post_layernorm"])->forward(ctx, x);
        std::unordered_set<ggml_tensor *> visited;
        require_f32_accumulation(x, visited);
        return x;
    }
};
#endif
