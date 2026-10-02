#ifndef HV15_VISION_PROJECTION_HPP
#define HV15_VISION_PROJECTION_HPP
#include "model/common/block.hpp"

/* Hunyuan 1.5 uses nn.GELU() (erf), unlike the tanh GELU in WAN::MLPProj. */
struct HV15VisionProjection : public UnaryBlock {
    HV15VisionProjection(int64_t input_dim, int64_t output_dim) {
        blocks["proj.0"] = std::make_shared<LayerNorm>(input_dim, 1e-5f);
        blocks["proj.1"] = std::make_shared<Linear>(input_dim, input_dim);
        blocks["proj.3"] = std::make_shared<Linear>(input_dim, output_dim);
        blocks["proj.4"] = std::make_shared<LayerNorm>(output_dim, 1e-5f);
    }
    ggml_tensor *forward(GGMLRunnerContext *ctx, ggml_tensor *x) override {
        x = std::dynamic_pointer_cast<LayerNorm>(blocks["proj.0"])->forward(ctx, x);
        x = std::dynamic_pointer_cast<Linear>(blocks["proj.1"])->forward(ctx, x);
        x = ggml_gelu_erf_inplace(ctx->ggml_ctx, x);
        x = std::dynamic_pointer_cast<Linear>(blocks["proj.3"])->forward(ctx, x);
        return std::dynamic_pointer_cast<LayerNorm>(blocks["proj.4"])->forward(ctx, x);
    }
};
#endif
