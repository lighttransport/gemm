/* SAM 3 CPU runner — public interface (incremental implementation).
 *
 * SAM 3 = concept-level promptable segmentation: image + text phrase
 * → variable-count instance masks. Graph (from HF transformers.Sam3Model):
 *
 *   pre-process    resize (bilinear) to 1008x1008, (px/127.5 - 1) normalize.
 *   patch_embed    Conv2d(3, 1024, k=14, s=14, bias=False). Out (1,5184,1024).
 *   pos_embed      Learned (1,576,1024) for 24x24 grid; TILED 3x3 to 72x72.
 *   pre_norm       LayerNorm(1024) applied once before block 0.
 *   ViT x32        2D-axial-RoPE MHA, pairwise rotation, window=24 per block
 *                  except layers [7,15,23,31] (global, 72x72 scale=1/3).
 *   FPN neck       4-scale pyramid (stub).
 *   CLIP text x24  causal MHA (stub).
 *   DETR enc/dec   6+6 layers, 200 queries (stub).
 *   mask head      einsum + pixel decoder (stub).
 *
 * Implemented now: preprocess + patch_embed + pos_embed + pre_norm + ViT (32 blocks).
 * Downstream stages stubbed with sam3_predict_text returning -1.
 */
#ifndef SAM3_RUNNER_H_
#define SAM3_RUNNER_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sam3_ctx sam3_ctx;

typedef struct {
    const char *ckpt_path;
    int image_size;        /* fixed at 1008 by arch */
    int num_threads;       /* 0 = auto (sysconf) */
} sam3_config;

sam3_ctx *sam3_create(const sam3_config *cfg);
void      sam3_destroy(sam3_ctx *ctx);

/* Preprocess + patch_embed + pos_embed. Stores token embeddings internally.
 * Returns 0 on success, -1 on error. */
int sam3_set_image(sam3_ctx *ctx, const uint8_t *rgb, int h, int w);

/* Bit-exact alternative: caller supplies already-normalized pixel values,
 * shape (3, image_size, image_size) CHW fp32 in the same space as HF's
 * Sam3ImageProcessor output (mean=0.5 std=0.5, rescale 1/255). Skips the
 * stb resize path which differs slightly from PIL BILINEAR. */
int sam3_set_pixel_values(sam3_ctx *ctx, const float *pixel_values_chw);

/* Accessor: token embeddings after patch_embed + pos_embed. Layout
 * (n_tok, embed_dim) row-major. n_tok=72*72=5184, dim=1024. Pointer valid
 * until the next sam3_set_image / destroy. */
const float *sam3_get_vit_embed(const sam3_ctx *ctx,
                                int *out_n_tok, int *out_dim);

/* Run ViT forward through block `stop_at_block` inclusive (0..31). Operates
 * on the tokens produced by the most recent sam3_set_image/set_pixel_values.
 * Applies the pre-block LayerNorm once (on the first call after set_image).
 * Returns 0 on success. Output stored internally; use sam3_get_vit_output. */
int sam3_run_vit(sam3_ctx *ctx, int stop_at_block);

/* Accessor: output of the most recent sam3_run_vit call. Shape
 * (n_tok, embed_dim) row-major = (72*72, 1024). */
const float *sam3_get_vit_output(const sam3_ctx *ctx,
                                  int *out_n_tok, int *out_dim);

/* Run the FPN neck on the most recent ViT output. Produces 4 feature maps
 * at scales [4.0, 2.0, 1.0, 0.5] → spatial sizes 288, 144, 72, 36 with
 * 256 channels. Run sam3_run_vit(31) first. */
int sam3_run_fpn(sam3_ctx *ctx);

/* Accessor: FPN level `level` (0..3) output, layout (C=256, H, W)
 * row-major. Sets *out_c=256, *out_h, *out_w. */
const float *sam3_get_fpn(const sam3_ctx *ctx, int level,
                          int *out_c, int *out_h, int *out_w);

/* Supply pre-tokenized CLIP input_ids (ctx_len = 32). BPE tokenizer is
 * not yet implemented; use ref/sam3/gen_image_ref.py to dump
 * input_input_ids.npy for now. Also takes attention_mask (1s for real
 * tokens including BOS/EOS, 0s for PAD). */
int sam3_set_input_ids(sam3_ctx *ctx, const int32_t *input_ids,
                       const int32_t *attention_mask);

/* Run the CLIP text encoder (24 layers, causal). Output shape (32, 1024),
 * same as HF `text_encoder.last_hidden_state`. */
int sam3_run_text(sam3_ctx *ctx);

const float *sam3_get_text_output(const sam3_ctx *ctx,
                                   int *out_len, int *out_dim);

/* Run the DETR encoder (6 layers). Fuses FPN level 2 (72x72,256) vision
 * features with projected text features (32,256) through self-attn +
 * text-cross-attn + MLP stacks. Requires sam3_run_fpn and sam3_run_text
 * to have been called. Output shape (5184, 256). */
int sam3_run_detr_enc(sam3_ctx *ctx);

const float *sam3_get_detr_enc(const sam3_ctx *ctx,
                                int *out_n, int *out_dim);

/* Run the DETR decoder (6 layers, 200 queries + presence token). Produces
 * refined boxes (xyxy, sigmoid-space), per-layer presence logits, and the
 * final-layer post-LN query hidden states. Requires sam3_run_detr_enc. */
int sam3_run_detr_dec(sam3_ctx *ctx);

/* Accessor: final pred_boxes (200, 4) xyxy sigmoid-space. */
const float *sam3_get_detr_dec_boxes(const sam3_ctx *ctx);

/* Accessor: per-layer presence logits (6,). */
const float *sam3_get_detr_dec_presence(const sam3_ctx *ctx);

/* Accessor: output_layer_norm(query_hidden_states) of the last layer,
 * shape (200, 256). Valid until the next sam3_run_detr_dec / destroy. */
const float *sam3_get_detr_dec_hidden(const sam3_ctx *ctx);

/* Run dot-product scoring. Produces per-layer pred_logits for the 200
 * queries (6, 200) using the stored decoder intermediate hidden states
 * and pooled (mask-aware) text features. Requires sam3_run_detr_dec. */
int sam3_run_dot_score(sam3_ctx *ctx);

/* Accessor: all 6 layers of scoring logits, shape (6, 200). */
const float *sam3_get_dot_scores(const sam3_ctx *ctx);

/* Run the mask decoder: prompt cross-attn on vision tokens + FPN pixel
 * decoder (72→144→288) + instance/semantic projection + einsum with
 * mask embedder. Produces pred_masks (200, 288, 288) and semantic_seg
 * (288, 288). Requires sam3_run_detr_dec and sam3_run_fpn. */
int sam3_run_mask_dec(sam3_ctx *ctx);

/* Accessor: pred_masks, layout (num_queries=200, H, W). */
const float *sam3_get_pred_masks(const sam3_ctx *ctx,
                                  int *out_q, int *out_h, int *out_w);

/* Accessor: semantic segmentation logits, layout (H, W). */
const float *sam3_get_semantic_seg(const sam3_ctx *ctx,
                                    int *out_h, int *out_w);

/* Post-process: combine last-layer pred_logits + presence into scores,
 * threshold-filter, scale boxes to (target_h, target_w), sigmoid + bilinear
 * upsample pred_masks and binarize. Requires sam3_run_dot_score and
 * sam3_run_mask_dec. Score and mask thresholds match HF defaults (0.3/0.5). */
int sam3_run_postprocess(sam3_ctx *ctx, int target_h, int target_w,
                         float score_threshold, float mask_threshold);

const float   *sam3_get_final_scores(const sam3_ctx *ctx, int *out_n);
const float   *sam3_get_final_boxes (const sam3_ctx *ctx, int *out_n);
const uint8_t *sam3_get_final_masks (const sam3_ctx *ctx,
                                     int *out_n, int *out_h, int *out_w);

/* Placeholder for text-prompted predict (not yet implemented). */
int sam3_predict_text(sam3_ctx *ctx, const char *phrase,
                      int max_masks,
                      float *out_masks, float *out_scores);

#ifdef __cplusplus
}
#endif

#endif /* SAM3_RUNNER_H_ */
