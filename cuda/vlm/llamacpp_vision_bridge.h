/* C-compatible bridge to llama.cpp's clip vision encoder */
#ifndef LLAMACPP_VISION_BRIDGE_H
#define LLAMACPP_VISION_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct llamacpp_vision_ctx llamacpp_vision_ctx;

/* Load mmproj. Returns NULL on failure. */
llamacpp_vision_ctx *llamacpp_vision_init(const char *mmproj_path, int use_gpu);

/* Get output dimensions */
int llamacpp_vision_n_mmproj_embd(llamacpp_vision_ctx *ctx);
int llamacpp_vision_n_output_tokens(llamacpp_vision_ctx *ctx, int img_w, int img_h);

/* Encode an image.
 * rgb_norm: float RGB pixels, normalized (HWC, interleaved, values in [-1,1] roughly).
 * img_w, img_h: image dimensions.
 * out_embd: output buffer (caller allocates, size = n_output_tokens * n_mmproj_embd * sizeof(float)).
 * Returns 0 on success, -1 on failure. */
int llamacpp_vision_encode(llamacpp_vision_ctx *ctx, const float *rgb_norm,
                           int img_w, int img_h, float *out_embd);

/* Free context */
void llamacpp_vision_free(llamacpp_vision_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* LLAMACPP_VISION_BRIDGE_H */
