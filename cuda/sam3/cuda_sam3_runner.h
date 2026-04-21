/* HIP/RDNA4 SAM 3 runner — public interface (Phase 1 scaffolding).
 *
 * Current coverage: preprocess (ImageNet) + patch_embed (k=14 s=14) +
 * pos_embed tile + pre-block LayerNorm. Subsequent ViT blocks, FPN,
 * CLIP text, DETR encoder/decoder, mask decoder and post-process are
 * staged in follow-up passes against ref/sam3 dumps.
 */
#ifndef CUDA_SAM3_RUNNER_H_
#define CUDA_SAM3_RUNNER_H_
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cuda_sam3_ctx cuda_sam3_ctx;

typedef struct {
    const char *ckpt_path;   /* sam3.model.safetensors */
    int image_size;          /* fixed 1008 */
    int device_ordinal;
    int verbose;
} cuda_sam3_config;

cuda_sam3_ctx *cuda_sam3_create(const cuda_sam3_config *cfg);
void          cuda_sam3_destroy(cuda_sam3_ctx *ctx);

/* Normalize + patch_embed + pos_embed + pre-block LN. Token embeddings
 * remain resident on the GPU; use cuda_sam3_get_vit_embed to read back. */
int cuda_sam3_set_image(cuda_sam3_ctx *ctx, const uint8_t *rgb, int h, int w);

/* Bit-close alternative: pre-normalized pixel values (3, S, S) CHW F32. */
int cuda_sam3_set_pixel_values(cuda_sam3_ctx *ctx, const float *pixel_values_chw);

/* Copy token embeddings (pre-LN patch+pos F32 5184×1024) into out_host.
 * After cuda_sam3_run_vit(bi), this returns the stream after block bi.  */
int cuda_sam3_get_vit_embed(const cuda_sam3_ctx *ctx, float *out_host,
                            int *out_n_tok, int *out_dim);

/* Run ViT blocks up through `stop_at_block` (inclusive, 0..31). On the
 * first call the pre-block LayerNorm is applied to the residual stream.
 * Subsequent calls resume from the last completed block. */
int cuda_sam3_run_vit(cuda_sam3_ctx *ctx, int stop_at_block);

/* Run the 4-level FPN neck on the current ViT output. Requires ViT to
 * have been run through block 31. Produces 4 feature maps at:
 *   level 0: (256, 288, 288)
 *   level 1: (256, 144, 144)
 *   level 2: (256,  72,  72)
 *   level 3: (256,  36,  36) */
int cuda_sam3_run_fpn(cuda_sam3_ctx *ctx);

/* Copy FPN feature map at `level` to host as (C, H, W) F32. */
int cuda_sam3_get_fpn(const cuda_sam3_ctx *ctx, int level, float *out_host,
                      int *out_c, int *out_h, int *out_w);

/* CLIP text encoder (24 layers, 1024 dim, 16h, causal, erf-GELU).
 * Input ids array length must be 32 (SAM 3 fixed context length). Pass
 * attention mask (1=valid, 0=pad) or NULL to mark all valid. */
int cuda_sam3_set_input_ids(cuda_sam3_ctx *ctx, const int32_t *ids,
                            const int32_t *attn_mask);
int cuda_sam3_run_text(cuda_sam3_ctx *ctx);

/* Copy text output (32, 1024) F32 into out_host (post final LN). */
int cuda_sam3_get_text_output(const cuda_sam3_ctx *ctx, float *out_host,
                               int *out_len, int *out_dim);

/* DETR encoder (6 layers). Requires ViT, FPN, text to have run.
 * Uses FPN level 2 (256, 72, 72) as vision input. */
int cuda_sam3_run_detr_enc(cuda_sam3_ctx *ctx);
int cuda_sam3_get_detr_enc(const cuda_sam3_ctx *ctx, float *out_host,
                            int *out_n, int *out_dim);

/* DETR decoder (6 layers, 200 queries + 1 presence). Requires detr_enc. */
int cuda_sam3_run_detr_dec(cuda_sam3_ctx *ctx);
/* Copy final pred_boxes (200, 4) xyxy normalized, 0..1. */
int cuda_sam3_get_detr_dec_boxes(const cuda_sam3_ctx *ctx, float *out);
/* Copy per-layer presence logits (6,) clamped. */
int cuda_sam3_get_detr_dec_presence(const cuda_sam3_ctx *ctx, float *out);
/* Copy last-layer output_layer_norm(query_hidden) (200, 256). */
int cuda_sam3_get_detr_dec_hidden(const cuda_sam3_ctx *ctx, float *out);

/* Dot-product scoring: per-layer query↔text logits (6, 200), clamp ±12. */
int cuda_sam3_run_dot_score(cuda_sam3_ctx *ctx);
int cuda_sam3_get_dot_scores(const cuda_sam3_ctx *ctx, float *out);

/* Mask decoder: prompt_cross_attn → pixel_decoder → instance/semantic →
 * mask_embedder → einsum. pred_masks (200, 288, 288) F32. */
int cuda_sam3_run_mask_dec(cuda_sam3_ctx *ctx);
int cuda_sam3_get_pred_masks(const cuda_sam3_ctx *ctx, float *out,
                              int *out_q, int *out_h, int *out_w);
int cuda_sam3_get_semantic_seg(const cuda_sam3_ctx *ctx, float *out,
                                int *out_h, int *out_w);

/* Post-process: score = sigmoid(logit_last) * sigmoid(presence_last);
 * keep where score > score_threshold; sigmoid(pred_masks[q]) bilinear
 * resize (align_corners=False) to (target_h, target_w); threshold →
 * uint8. Requires run_dot_score + run_mask_dec. */
int cuda_sam3_run_postprocess(cuda_sam3_ctx *ctx, int target_h, int target_w,
                              float score_threshold, float mask_threshold);
const float *cuda_sam3_get_final_scores(const cuda_sam3_ctx *ctx, int *out_n);
const float *cuda_sam3_get_final_boxes(const cuda_sam3_ctx *ctx, int *out_n);
const uint8_t *cuda_sam3_get_final_masks(const cuda_sam3_ctx *ctx,
                                         int *out_n, int *out_h, int *out_w);

#ifdef __cplusplus
}
#endif
#endif
