/* C++ implementation of the clip vision bridge using llama.cpp
 * This file is compiled as a shared library and loaded with RTLD_DEEPBIND
 * to avoid symbol conflicts with the main executable's gguf functions. */
#include "llamacpp_vision_bridge.h"
#include <clip-impl.h>
#include <dlfcn.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>

struct llamacpp_vision_ctx {
    struct clip_ctx *clip;
    int n_mmproj_embd;
};

llamacpp_vision_ctx *llamacpp_vision_init(const char *mmproj_path, int use_gpu) {
    struct clip_context_params params;
    std::memset(&params, 0, sizeof(params));
    params.use_gpu = !!use_gpu;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    params.warmup = true;

    auto result = clip_init(mmproj_path, params);
    if (!result.ctx_v) {
        fprintf(stderr, "llamacpp_vision: clip_init failed for %s\n", mmproj_path);
        return NULL;
    }

    llamacpp_vision_ctx *ctx = (llamacpp_vision_ctx *)calloc(1, sizeof(*ctx));
    ctx->clip = result.ctx_v;
    ctx->n_mmproj_embd = clip_n_mmproj_embd(ctx->clip);
    fprintf(stderr, "llamacpp_vision: loaded %s, proj_dim=%d\n", mmproj_path, ctx->n_mmproj_embd);
    return ctx;
}

int llamacpp_vision_n_mmproj_embd(llamacpp_vision_ctx *ctx) {
    return ctx ? ctx->n_mmproj_embd : 0;
}

int llamacpp_vision_n_output_tokens(llamacpp_vision_ctx *ctx, int img_w, int img_h) {
    if (!ctx) return 0;
    size_t nbytes = clip_embd_nbytes_by_img(ctx->clip, img_w, img_h);
    return (int)(nbytes / (ctx->n_mmproj_embd * sizeof(float)));
}

int llamacpp_vision_encode(llamacpp_vision_ctx *ctx, const float *rgb_norm,
                           int img_w, int img_h, float *out_embd) {
    if (!ctx || !ctx->clip) return -1;
    bool ok = clip_encode_float_image(ctx->clip, 1, const_cast<float*>(rgb_norm),
                                       img_h, img_w, out_embd);
    return ok ? 0 : -1;
}

void llamacpp_vision_free(llamacpp_vision_ctx *ctx) {
    if (!ctx) return;
    if (ctx->clip) clip_free(ctx->clip);
    free(ctx);
}
