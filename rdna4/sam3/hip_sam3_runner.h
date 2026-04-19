/* HIP/RDNA4 SAM 3 runner — public interface (Phase 1 scaffolding).
 *
 * Current coverage: preprocess (ImageNet) + patch_embed (k=14 s=14) +
 * pos_embed tile + pre-block LayerNorm. Subsequent ViT blocks, FPN,
 * CLIP text, DETR encoder/decoder, mask decoder and post-process are
 * staged in follow-up passes against ref/sam3 dumps.
 */
#ifndef HIP_SAM3_RUNNER_H_
#define HIP_SAM3_RUNNER_H_
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hip_sam3_ctx hip_sam3_ctx;

typedef struct {
    const char *ckpt_path;   /* sam3.model.safetensors */
    int image_size;          /* fixed 1008 */
    int device_ordinal;
    int verbose;
} hip_sam3_config;

hip_sam3_ctx *hip_sam3_create(const hip_sam3_config *cfg);
void          hip_sam3_destroy(hip_sam3_ctx *ctx);

/* Normalize + patch_embed + pos_embed + pre-block LN. Token embeddings
 * remain resident on the GPU; use hip_sam3_get_vit_embed to read back. */
int hip_sam3_set_image(hip_sam3_ctx *ctx, const uint8_t *rgb, int h, int w);

/* Bit-close alternative: pre-normalized pixel values (3, S, S) CHW F32. */
int hip_sam3_set_pixel_values(hip_sam3_ctx *ctx, const float *pixel_values_chw);

/* Copy token embeddings (post patch+pos+LN, 5184×1024 F32) into out_host. */
int hip_sam3_get_vit_embed(const hip_sam3_ctx *ctx, float *out_host,
                            int *out_n_tok, int *out_dim);

#ifdef __cplusplus
}
#endif
#endif
