/* sam3d_body_runner.c — CPU runner for SAM 3D Body.
 *
 * Supports the DINOv3-H+ and ViT-H backbones, raw-image preprocessing,
 * decoder/MHR execution, and ref-dump override paths used by the stage
 * verifiers.
 */

#include "sam3d_body_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#define DINOV3_IMPLEMENTATION
#include "dinov3.h"

#define SAM3D_BODY_VIT_IMPLEMENTATION
#include "sam3d_body_vit.h"

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#define SAM3D_BODY_DECODER_FULL_IMPLEMENTATION
#define SAM3D_BODY_MHR_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "sam3d_body_mhr.h"

#define SAM3D_BODY_E_OK               (0)
#define SAM3D_BODY_E_INVAL            (-1)
#define SAM3D_BODY_E_NOT_IMPLEMENTED  (-2)
#define SAM3D_BODY_E_LOAD_FAILED      (-3)

#define SAM3D_BODY_ENCODER_SIZE 512
#define SAM3D_BODY_VERTEX_COUNT 18439
#define SAM3D_BODY_KP_COUNT     70

typedef struct {
    float *data;
    int    n, c;
} f32_2d;

struct sam3d_body_ctx {
    sam3d_body_config cfg;

    /* Models (lazy-loaded). */
    dinov3_model              *encoder_model;       /* DINOv3-H+ variant */
    sam3d_body_vit_model      *encoder_model_vith;  /* vit_hmr_512_384 variant */
    sam3d_body_decoder_model  *decoder_model;
    sam3d_body_mhr_assets     *mhr_assets;

    /* Backbone geometry cached at run_encoder so self_drive_decoder_inputs
     * doesn't have to inspect the variant-specific model struct. */
    int enc_grid_h, enc_grid_w;   /* patch grid (32×32 dinov3, 32×24 vith) */
    int enc_n_prefix;             /* 1+n_storage dinov3, 0 vith */
    int enc_image_h, enc_image_w; /* model input dims (512×512 vs 512×384) */

    /* Inputs. */
    uint8_t *image_rgb;  int img_w, img_h;
    float    bbox[4];    int has_bbox;
    uint8_t *mask;       int mask_w, mask_h;
    float   *kp_xy;      int n_kp;
    float    focal_hint;

    /* Self-driven preprocess outputs (set by run_encoder when dec_in_set==0). */
    int       self_pp_set;
    float     self_warp[6];
    float     self_center[2];
    float     self_scale[2];
    float     self_cam_int[9];

    /* Pre-computed decoder inputs (debug-override path; Step 8c-i). */
    int       dec_in_set;
    int       dec_H, dec_W;
    float    *dec_image_emb_chw;     /* (kv_dim, H, W) */
    float    *dec_image_pe_chw;      /* (kv_dim, H, W) */
    float    *dec_init_x;            /* (145, 1024)    */
    float    *dec_init_xpe;          /* (145, 1024)    */
    sam3d_body_camera_batch dec_cam_batch;

    /* Stage outputs (all NULL until the corresponding stage lands). */
    f32_2d   encoder_tokens;   /* [n_tokens, dim] */
    float   *mhr_params;       int mhr_params_n;
    float    cam_t[3];
    float    focal_px;
    float   *vertices;         int n_vertices;   /* V×3 */
    int32_t *faces;            int n_faces;      /* F×3 */
    float   *keypoints_3d;     int n_kp_3d;      /* K×3 */
    float   *keypoints_2d;     int n_kp_2d;      /* K×2 */
};

static void f32_2d_free(f32_2d *b) {
    free(b->data); b->data = NULL; b->n = b->c = 0;
}

static double sam3d_body_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static int sam3d_body_timing_enabled(const sam3d_body_ctx *ctx)
{
    const char *env = getenv("SAM3D_BODY_TIMING");
    return (ctx && ctx->cfg.verbose) || (env && env[0] && strcmp(env, "0") != 0);
}

sam3d_body_ctx *sam3d_body_create(const sam3d_body_config *cfg)
{
    if (!cfg || !cfg->safetensors_dir) return NULL;
    sam3d_body_ctx *c = (sam3d_body_ctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cfg = *cfg;
    if (c->cfg.n_threads <= 0) {
        const char *env = getenv("OMP_NUM_THREADS");
        c->cfg.n_threads = env ? atoi(env) : 1;
        if (c->cfg.n_threads < 1) c->cfg.n_threads = 1;
    }
    return c;
}

void sam3d_body_destroy(sam3d_body_ctx *ctx)
{
    if (!ctx) return;
    if (ctx->encoder_model) dinov3_free(ctx->encoder_model);
    if (ctx->encoder_model_vith) sam3d_body_vit_free(ctx->encoder_model_vith);
    if (ctx->decoder_model) sam3d_body_decoder_free(ctx->decoder_model);
    if (ctx->mhr_assets)    sam3d_body_mhr_free(ctx->mhr_assets);
    free(ctx->image_rgb);
    free(ctx->mask);
    free(ctx->kp_xy);
    free(ctx->dec_image_emb_chw);
    free(ctx->dec_image_pe_chw);
    free(ctx->dec_init_x);
    free(ctx->dec_init_xpe);
    f32_2d_free(&ctx->encoder_tokens);
    free(ctx->mhr_params);
    free(ctx->vertices);
    free(ctx->faces);
    free(ctx->keypoints_3d);
    free(ctx->keypoints_2d);
    free(ctx);
}

/* Load decoder + MHR-head safetensors; lazy. The decoder + mhr_head weights
 * differ per-variant (verified 2026-04-26: same shapes but values diverge by
 * up to 0.87 max-abs across ckpts), so prefer the variant-tagged slices when
 * present and fall back to the legacy unprefixed names. */
static int file_exists(const char *p) {
    FILE *f = fopen(p, "rb"); if (!f) return 0; fclose(f); return 1;
}
static int ensure_decoder_loaded(sam3d_body_ctx *ctx)
{
    if (ctx->decoder_model) return SAM3D_BODY_E_OK;
    const char *tag = (ctx->cfg.backbone == SAM3D_BODY_BACKBONE_VITH)
                          ? "vith" : "dinov3";
    char path_dec[1024], path_mhr[1024];
    snprintf(path_dec, sizeof(path_dec),
             "%s/sam3d_body_%s_decoder.safetensors",
             ctx->cfg.safetensors_dir, tag);
    snprintf(path_mhr, sizeof(path_mhr),
             "%s/sam3d_body_%s_mhr_head.safetensors",
             ctx->cfg.safetensors_dir, tag);
    if (!file_exists(path_dec)) {
        snprintf(path_dec, sizeof(path_dec),
                 "%s/sam3d_body_decoder.safetensors", ctx->cfg.safetensors_dir);
    }
    if (!file_exists(path_mhr)) {
        snprintf(path_mhr, sizeof(path_mhr),
                 "%s/sam3d_body_mhr_head.safetensors", ctx->cfg.safetensors_dir);
    }
    if (ctx->cfg.verbose) {
        fprintf(stderr, "[sam3d_body] decoder: %s\n[sam3d_body] mhr_head: %s\n",
                path_dec, path_mhr);
    }
    ctx->decoder_model = sam3d_body_decoder_load(path_dec, path_mhr);
    return ctx->decoder_model ? SAM3D_BODY_E_OK : SAM3D_BODY_E_LOAD_FAILED;
}

/* Load MHR skinning assets; lazy. */
static int ensure_mhr_loaded(sam3d_body_ctx *ctx)
{
    if (ctx->mhr_assets) return SAM3D_BODY_E_OK;
    if (!ctx->cfg.mhr_assets_dir) {
        fprintf(stderr, "[sam3d_body] mhr_assets_dir not configured\n");
        return SAM3D_BODY_E_INVAL;
    }
    char path_sft[1024], path_json[1024];
    snprintf(path_sft,  sizeof(path_sft),
             "%s/sam3d_body_mhr_jit.safetensors", ctx->cfg.mhr_assets_dir);
    snprintf(path_json, sizeof(path_json),
             "%s/sam3d_body_mhr_jit.json",        ctx->cfg.mhr_assets_dir);
    ctx->mhr_assets = sam3d_body_mhr_load(path_sft, path_json);
    return ctx->mhr_assets ? SAM3D_BODY_E_OK : SAM3D_BODY_E_LOAD_FAILED;
}

/* ---------------- inputs ---------------- */

int sam3d_body_set_image(sam3d_body_ctx *ctx, const uint8_t *rgb,
                         int width, int height, const float bbox[4])
{
    if (!ctx || !rgb || width <= 0 || height <= 0) return SAM3D_BODY_E_INVAL;
    size_t bytes = (size_t)width * height * 3;
    free(ctx->image_rgb);
    ctx->image_rgb = (uint8_t *)malloc(bytes);
    if (!ctx->image_rgb) return SAM3D_BODY_E_INVAL;
    memcpy(ctx->image_rgb, rgb, bytes);
    ctx->img_w = width;
    ctx->img_h = height;
    ctx->has_bbox = 0;
    if (bbox) {
        memcpy(ctx->bbox, bbox, sizeof(ctx->bbox));
        ctx->has_bbox = 1;
    }
    return SAM3D_BODY_E_OK;
}

int sam3d_body_set_keypoints_2d(sam3d_body_ctx *ctx,
                                const float *kp_xy, int k)
{
    if (!ctx || !kp_xy || k <= 0) return SAM3D_BODY_E_INVAL;
    free(ctx->kp_xy);
    ctx->kp_xy = (float *)malloc((size_t)k * 2 * sizeof(float));
    if (!ctx->kp_xy) return SAM3D_BODY_E_INVAL;
    memcpy(ctx->kp_xy, kp_xy, (size_t)k * 2 * sizeof(float));
    ctx->n_kp = k;
    return SAM3D_BODY_E_OK;
}

int sam3d_body_set_mask(sam3d_body_ctx *ctx, const uint8_t *mask,
                        int width, int height)
{
    if (!ctx || !mask || width <= 0 || height <= 0) return SAM3D_BODY_E_INVAL;
    size_t bytes = (size_t)width * height;
    free(ctx->mask);
    ctx->mask = (uint8_t *)malloc(bytes);
    if (!ctx->mask) return SAM3D_BODY_E_INVAL;
    memcpy(ctx->mask, mask, bytes);
    ctx->mask_w = width;
    ctx->mask_h = height;
    return SAM3D_BODY_E_OK;
}

int sam3d_body_set_focal(sam3d_body_ctx *ctx, float focal_px)
{
    if (!ctx) return SAM3D_BODY_E_INVAL;
    ctx->focal_hint = focal_px;
    return SAM3D_BODY_E_OK;
}

/* ---------------- pipeline stages ---------------- */

int sam3d_body_run_encoder(sam3d_body_ctx *ctx)
{
    if (!ctx) return SAM3D_BODY_E_INVAL;
    /* When decoder inputs are pre-populated (ref-dump mode), the encoder
     * is already baked into dec_image_emb_chw — skip. */
    if (ctx->dec_in_set) return SAM3D_BODY_E_OK;
    if (!ctx->image_rgb) return SAM3D_BODY_E_INVAL;

    const int is_vith = (ctx->cfg.backbone == SAM3D_BODY_BACKBONE_VITH);

    /* Both variants run TopdownAffine to a 512×512 working canvas; vith
     * then crops 64 columns on each side to (512, 384). */
    const int S = 512;

    /* TopdownAffine: GetBBoxCenterScale + 2× fix_aspect_ratio + get_warp_matrix.
     * If no bbox supplied, use the full image as the bbox. */
    float bbox_xyxy[4];
    if (ctx->has_bbox) {
        memcpy(bbox_xyxy, ctx->bbox, sizeof(bbox_xyxy));
    } else {
        bbox_xyxy[0] = 0; bbox_xyxy[1] = 0;
        bbox_xyxy[2] = (float)ctx->img_w; bbox_xyxy[3] = (float)ctx->img_h;
    }
    sam3d_body_compute_bbox_affine(bbox_xyxy, /*padding=*/1.25f,
                                   /*aspect_ratio_pre=*/0.75f, S, S,
                                   ctx->self_center, ctx->self_scale,
                                   ctx->self_warp);

    /* Default cam_int (or focal hint override on the diagonal). */
    sam3d_body_default_cam_int(ctx->img_w, ctx->img_h, ctx->self_cam_int);
    if (ctx->focal_hint > 0) {
        ctx->self_cam_int[0] = ctx->focal_hint;
        ctx->self_cam_int[4] = ctx->focal_hint;
    }
    ctx->self_pp_set = 1;

    /* cv2.warpAffine + ImageNet norm + HWC→CHW into the (3, 512, 512) canvas. */
    float *chw = (float *)malloc((size_t)3 * S * S * sizeof(float));
    if (!chw) return SAM3D_BODY_E_INVAL;
    int rc_pp = sam3d_body_preprocess_image(ctx->image_rgb, ctx->img_w, ctx->img_h,
                                            ctx->self_warp, S, S, chw);
    if (rc_pp != 0) { free(chw); return SAM3D_BODY_E_INVAL; }

    if (!is_vith) {
        if (!ctx->encoder_model) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/sam3d_body_dinov3.safetensors",
                     ctx->cfg.safetensors_dir);
            if (ctx->cfg.verbose)
                fprintf(stderr, "[sam3d_body] loading encoder: %s\n", path);
            ctx->encoder_model = dinov3_load_safetensors(path);
            if (!ctx->encoder_model) {
                fprintf(stderr, "[sam3d_body] failed to load %s\n", path);
                free(chw); return SAM3D_BODY_E_LOAD_FAILED;
            }
            /* sam-3d-body consumes the learned final LayerNorm, not the
             * unparameterized one used by TRELLIS.2. */
            ctx->encoder_model->use_learned_final_norm = 1;
        }
        dinov3_result r = dinov3_encode_from_normalized(ctx->encoder_model, chw,
                                                        S, S, ctx->cfg.n_threads);
        free(chw);
        if (!r.features) return SAM3D_BODY_E_LOAD_FAILED;

        f32_2d_free(&ctx->encoder_tokens);
        ctx->encoder_tokens.data = r.features;  /* take ownership */
        ctx->encoder_tokens.n = r.n_tokens;
        ctx->encoder_tokens.c = r.dim;

        ctx->enc_grid_h    = ctx->encoder_model->grid_h;
        ctx->enc_grid_w    = ctx->encoder_model->grid_w;
        ctx->enc_n_prefix  = 1 + ctx->encoder_model->n_storage;
        ctx->enc_image_h   = S;
        ctx->enc_image_w   = S;
        return SAM3D_BODY_E_OK;
    }

    /* ---- ViT-H (vit_hmr_512_384) path ---- */
    if (!ctx->encoder_model_vith) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/sam3d_body_vith.safetensors",
                 ctx->cfg.safetensors_dir);
        if (ctx->cfg.verbose)
            fprintf(stderr, "[sam3d_body] loading vith encoder: %s\n", path);
        ctx->encoder_model_vith = sam3d_body_vit_load_safetensors(path);
        if (!ctx->encoder_model_vith) {
            fprintf(stderr, "[sam3d_body] failed to load %s\n", path);
            free(chw); return SAM3D_BODY_E_LOAD_FAILED;
        }
    }

    /* W-axis crop [:, :, :, 64:-64] → (3, 512, 384). chw is (3, S, S) row-major. */
    const int W_crop = S - 128;  /* 384 */
    float *chw_cropped = (float *)malloc((size_t)3 * S * W_crop * sizeof(float));
    if (!chw_cropped) { free(chw); return SAM3D_BODY_E_INVAL; }
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < S; y++) {
            const float *src = chw + ((size_t)c * S + y) * S + 64;
            float       *dst = chw_cropped + ((size_t)c * S + y) * W_crop;
            memcpy(dst, src, (size_t)W_crop * sizeof(float));
        }
    }
    free(chw);

    sam3d_body_vit_result r = sam3d_body_vit_encode_from_normalized(
            ctx->encoder_model_vith, chw_cropped, W_crop, S, ctx->cfg.n_threads);
    free(chw_cropped);
    if (!r.tokens) return SAM3D_BODY_E_LOAD_FAILED;

    f32_2d_free(&ctx->encoder_tokens);
    ctx->encoder_tokens.data = r.tokens;  /* take ownership */
    ctx->encoder_tokens.n    = r.n_patches;
    ctx->encoder_tokens.c    = r.dim;

    ctx->enc_grid_h   = r.grid_h;
    ctx->enc_grid_w   = r.grid_w;
    ctx->enc_n_prefix = 0;          /* vit_hmr has no CLS / register tokens */
    ctx->enc_image_h  = S;
    ctx->enc_image_w  = W_crop;
    return SAM3D_BODY_E_OK;
}

/* Build the (kv_dim, H, W) post-ray_cond image_emb + (kv_dim, H, W) dense PE +
 * (145, 1024) build_tokens outputs + sam3d_body_camera_batch from the encoder
 * tokens and self-driven preprocess outputs. Result goes directly into the
 * ctx->dec_* fields and ctx->dec_cam_batch, then dec_in_set is set so the
 * downstream forward_full path runs unchanged. */
static int self_drive_decoder_inputs(sam3d_body_ctx *ctx)
{
    if (!ctx->encoder_tokens.data || !ctx->self_pp_set ||
        ctx->enc_grid_h <= 0 || ctx->enc_grid_w <= 0)
        return SAM3D_BODY_E_INVAL;

    int rc = ensure_decoder_loaded(ctx);
    if (rc != SAM3D_BODY_E_OK) return rc;

    const sam3d_body_decoder_model *dm = ctx->decoder_model;
    const int Dc = dm->kv_dim;        /* 1280 */
    const int D  = dm->dim;           /* 1024 */
    const int N_q = 145;

    /* Encoder grid (32×32 for DINOv3-H+, 32×24 for vit_hmr_512_384). */
    const int gh = ctx->enc_grid_h;
    const int gw = ctx->enc_grid_w;
    const int n_prefix = ctx->enc_n_prefix;
    const int n_patches = gh * gw;
    if (ctx->encoder_tokens.n != n_prefix + n_patches ||
        ctx->encoder_tokens.c != Dc) {
        fprintf(stderr, "[sam3d_body] encoder shape mismatch: tokens=(%d,%d), "
                        "expected (%d,%d)\n",
                ctx->encoder_tokens.n, ctx->encoder_tokens.c,
                n_prefix + n_patches, Dc);
        return SAM3D_BODY_E_INVAL;
    }

    /* image_emb_pre: drop CLS+register tokens, permute (HW, Dc) → (Dc, H, W). */
    float *img_emb_pre = (float *)malloc((size_t)Dc * gh * gw * sizeof(float));
    if (!img_emb_pre) return SAM3D_BODY_E_INVAL;
    const float *patch_tokens = ctx->encoder_tokens.data + (size_t)n_prefix * Dc;
    for (int n = 0; n < n_patches; n++) {
        for (int c = 0; c < Dc; c++) {
            img_emb_pre[(size_t)c * n_patches + n] =
                patch_tokens[(size_t)n * Dc + c];
        }
    }

    /* ray_cond_xyz at the encoder grid resolution.
     * For vith (W cropped from 512 to 384), shift the affine X-translation
     * by -64 so j=0 in the cropped grid maps back to source pixels via the
     * same x_full[j] = j/a00 - t02_shifted/a00 relation that upstream uses
     * after slicing [:, :, :, 64:-64]. Equivalent to evaluating the original
     * affine at x' = x + 64. */
    const int img_h_for_rays = ctx->enc_image_h;
    const int img_w_for_rays = ctx->enc_image_w;
    float warp_for_rays[6];
    memcpy(warp_for_rays, ctx->self_warp, sizeof(warp_for_rays));
    if (ctx->cfg.backbone == SAM3D_BODY_BACKBONE_VITH) {
        warp_for_rays[2] = ctx->self_warp[2] - 64.0f * ctx->self_warp[0];
    }
    float *rays_xyz = (float *)malloc((size_t)gh * gw * 3 * sizeof(float));
    if (!rays_xyz) { free(img_emb_pre); return SAM3D_BODY_E_INVAL; }
    rc = sam3d_body_compute_ray_cond_xyz(ctx->self_cam_int, warp_for_rays,
                                         img_h_for_rays, img_w_for_rays,
                                         gh, gw, rays_xyz);
    if (rc != 0) { free(img_emb_pre); free(rays_xyz); return rc; }

    /* image_emb POST ray_cond. */
    free(ctx->dec_image_emb_chw);
    ctx->dec_image_emb_chw = (float *)malloc((size_t)Dc * gh * gw * sizeof(float));
    if (!ctx->dec_image_emb_chw) {
        free(img_emb_pre); free(rays_xyz); return SAM3D_BODY_E_INVAL;
    }
    rc = sam3d_body_ray_cond_emb_forward(dm, img_emb_pre, rays_xyz, gh, gw,
                                         ctx->cfg.n_threads,
                                         ctx->dec_image_emb_chw);
    free(img_emb_pre);
    free(rays_xyz);
    if (rc != 0) return rc;

    /* dense_pe (image_pe). */
    free(ctx->dec_image_pe_chw);
    ctx->dec_image_pe_chw = (float *)malloc((size_t)Dc * gh * gw * sizeof(float));
    if (!ctx->dec_image_pe_chw) return SAM3D_BODY_E_INVAL;
    rc = sam3d_body_get_dense_pe(dm, gh, gw, ctx->cfg.n_threads,
                                 ctx->dec_image_pe_chw);
    if (rc != 0) return rc;

    /* condition_info (CLIFF). */
    float ori_img_size[2] = { (float)ctx->img_w, (float)ctx->img_h };
    float img_size[2]     = { (float)img_w_for_rays, (float)img_h_for_rays };
    float bbox_scale1[1]  = { ctx->self_scale[0] };  /* (1,) — only sw is used */
    float condition_info[3];
    sam3d_body_compute_condition_info(ctx->self_center, bbox_scale1,
                                      ori_img_size, ctx->self_cam_int,
                                      /*use_intrin_center=*/0,
                                      condition_info);

    /* init_input (525) = [condition_info(3), init_pose(519), init_camera(3)]
     * prev_input (522) = [init_pose(519), init_camera(3)]                  */
    const float *ip = (const float *)dm->init_pose.data;     /* (519,) */
    const float *ic = (const float *)dm->init_camera.data;   /* (3,)   */
    float init_input[525];
    float prev_input[522];
    memcpy(init_input,         condition_info, 3 * sizeof(float));
    memcpy(init_input + 3,     ip,           519 * sizeof(float));
    memcpy(init_input + 3+519, ic,             3 * sizeof(float));
    memcpy(prev_input,         ip,           519 * sizeof(float));
    memcpy(prev_input + 519,   ic,             3 * sizeof(float));

    /* prompt_in (1280,) — single dummy invalid keypoint. */
    float prompt_in[1280];
    rc = sam3d_body_invalid_prompt_token(dm, prompt_in);
    if (rc != 0) return rc;

    /* Build (145, 1024) tokens + posembs. */
    free(ctx->dec_init_x);
    free(ctx->dec_init_xpe);
    ctx->dec_init_x   = (float *)calloc((size_t)N_q * D, sizeof(float));
    ctx->dec_init_xpe = (float *)calloc((size_t)N_q * D, sizeof(float));
    if (!ctx->dec_init_x || !ctx->dec_init_xpe) return SAM3D_BODY_E_INVAL;
    rc = sam3d_body_build_tokens(dm, init_input, prev_input, prompt_in,
                                 ctx->cfg.n_threads,
                                 ctx->dec_init_x, ctx->dec_init_xpe);
    if (rc != 0) return rc;

    ctx->dec_H = gh; ctx->dec_W = gw;

    /* camera_batch. */
    memcpy(ctx->dec_cam_batch.cam_int,      ctx->self_cam_int, 9 * sizeof(float));
    memcpy(ctx->dec_cam_batch.bbox_center,  ctx->self_center,  2 * sizeof(float));
    ctx->dec_cam_batch.bbox_scale = ctx->self_scale[0];
    memcpy(ctx->dec_cam_batch.ori_img_size, ori_img_size, 2 * sizeof(float));
    memcpy(ctx->dec_cam_batch.img_size,     img_size,     2 * sizeof(float));
    /* For vith, the decoder sees the (384, 512) cropped canvas, so the
     * affine must include the same -64 X-shift used for ray_cond_xyz. */
    memcpy(ctx->dec_cam_batch.affine_trans, warp_for_rays, 6 * sizeof(float));
    ctx->dec_cam_batch.use_intrin_center    = 0;
    ctx->dec_cam_batch.default_scale_factor = 1.0f;

    ctx->dec_in_set = 1;
    return SAM3D_BODY_E_OK;
}

int sam3d_body_run_decoder(sam3d_body_ctx *ctx)
{
    if (!ctx) return SAM3D_BODY_E_INVAL;

    /* Self-driven path (Step 8c-ii): if no debug-override decoder inputs are
     * pre-populated, derive them from encoder_tokens + the bbox-affine /
     * cam_int captured by run_encoder. */
    if (!ctx->dec_in_set) {
        int rc = self_drive_decoder_inputs(ctx);
        if (rc != SAM3D_BODY_E_OK) return rc;
    }

    int rc = ensure_decoder_loaded(ctx);
    if (rc != SAM3D_BODY_E_OK) return rc;
    rc = ensure_mhr_loaded(ctx);
    if (rc != SAM3D_BODY_E_OK) return rc;

    const int V = SAM3D_BODY_VERTEX_COUNT;
    const int K = SAM3D_BODY_KP_COUNT;

    /* Allocate output buffers (vertices + keypoints). */
    free(ctx->vertices);
    ctx->vertices = (float *)malloc((size_t)V * 3 * sizeof(float));
    if (!ctx->vertices) return SAM3D_BODY_E_INVAL;

    sam3d_body_decoder_full_result r;
    memset(&r, 0, sizeof(r));
    r.pred_vertices = ctx->vertices;

    rc = sam3d_body_decoder_forward_full(
            ctx->decoder_model,
            (struct sam3d_body_mhr_assets_t *)ctx->mhr_assets,
            &ctx->dec_cam_batch,
            ctx->dec_image_emb_chw, ctx->dec_image_pe_chw,
            ctx->dec_H, ctx->dec_W,
            ctx->dec_init_x, ctx->dec_init_xpe,
            ctx->cfg.n_threads, &r);
    if (rc != 0) {
        fprintf(stderr, "[sam3d_body] decoder_forward_full rc=%d\n", rc);
        return rc;
    }
    ctx->n_vertices = V;

    /* mhr_params + cam_t. */
    free(ctx->mhr_params);
    ctx->mhr_params = (float *)malloc(519 * sizeof(float));
    if (!ctx->mhr_params) return SAM3D_BODY_E_INVAL;
    memcpy(ctx->mhr_params, r.mhr_params, 519 * sizeof(float));
    ctx->mhr_params_n = 519;
    memcpy(ctx->cam_t, r.pred_cam_t_world, sizeof(ctx->cam_t));
    /* Focal — diagonal of cam_int (model already estimates s∈cam_t[0] as
     * scale factor; focal lives in cam_int). */
    ctx->focal_px = ctx->dec_cam_batch.cam_int[0];

    /* Populate keypoints from the result. */
    free(ctx->keypoints_3d);
    ctx->keypoints_3d = (float *)malloc((size_t)K * 3 * sizeof(float));
    if (!ctx->keypoints_3d) return SAM3D_BODY_E_INVAL;
    memcpy(ctx->keypoints_3d, r.pred_keypoints_3d, (size_t)K * 3 * sizeof(float));
    ctx->n_kp_3d = K;

    free(ctx->keypoints_2d);
    ctx->keypoints_2d = (float *)malloc((size_t)K * 2 * sizeof(float));
    if (!ctx->keypoints_2d) return SAM3D_BODY_E_INVAL;
    memcpy(ctx->keypoints_2d, r.pred_keypoints_2d, (size_t)K * 2 * sizeof(float));
    ctx->n_kp_2d = K;

    /* Populate faces from the MHR head asset (stored as int64 (F,3)). */
    if (ctx->decoder_model->faces.data && !ctx->faces) {
        int F = (int)ctx->decoder_model->faces.dims[0];
        ctx->faces = (int32_t *)malloc((size_t)F * 3 * sizeof(int32_t));
        if (!ctx->faces) return SAM3D_BODY_E_INVAL;
        const int64_t *src = (const int64_t *)ctx->decoder_model->faces.data;
        for (int i = 0; i < F * 3; i++)
            ctx->faces[i] = (int32_t)src[i];
        ctx->n_faces = F;
    }
    return SAM3D_BODY_E_OK;
}

/* MHR stage is folded into run_decoder (decoder_forward_full runs MHR
 * inline); keep run_mhr as a no-op shim for API symmetry. */
int sam3d_body_run_mhr(sam3d_body_ctx *ctx)
{
    if (!ctx) return SAM3D_BODY_E_INVAL;
    return ctx->vertices ? SAM3D_BODY_E_OK : SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int sam3d_body_run_all(sam3d_body_ctx *ctx)
{
    const int timing = sam3d_body_timing_enabled(ctx);
    double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0;
    if (timing) t0 = sam3d_body_time_ms();
    int rc = sam3d_body_run_encoder(ctx);
    if (rc != SAM3D_BODY_E_OK) return rc;
    if (timing) t1 = sam3d_body_time_ms();
    rc = sam3d_body_run_decoder(ctx);
    if (rc != SAM3D_BODY_E_OK) return rc;
    if (timing) t2 = sam3d_body_time_ms();
    rc = sam3d_body_run_mhr(ctx);
    if (rc != SAM3D_BODY_E_OK) return rc;
    if (timing) {
        t3 = sam3d_body_time_ms();
        fprintf(stderr,
                "[sam3d_body][timing] run_all %.3f ms encoder %.3f decoder %.3f mhr %.3f\n",
                t3 - t0, t1 - t0, t2 - t1, t3 - t2);
    }
    return rc;
}

/* ---------------- readbacks ---------------- */

int sam3d_body_get_encoder_tokens(sam3d_body_ctx *ctx, float *out,
                                  int *out_n_tokens, int *out_dim)
{
    if (!ctx || !out_n_tokens || !out_dim) return SAM3D_BODY_E_INVAL;
    *out_n_tokens = ctx->encoder_tokens.n;
    *out_dim      = ctx->encoder_tokens.c;
    if (out && ctx->encoder_tokens.data)
        memcpy(out, ctx->encoder_tokens.data,
               (size_t)ctx->encoder_tokens.n *
               (size_t)ctx->encoder_tokens.c * sizeof(float));
    return ctx->encoder_tokens.data ? SAM3D_BODY_E_OK
                                    : SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int sam3d_body_get_mhr_params(sam3d_body_ctx *ctx, float *out, int *out_n)
{
    if (!ctx || !out_n) return SAM3D_BODY_E_INVAL;
    *out_n = ctx->mhr_params_n;
    if (out && ctx->mhr_params)
        memcpy(out, ctx->mhr_params,
               (size_t)ctx->mhr_params_n * sizeof(float));
    return ctx->mhr_params ? SAM3D_BODY_E_OK : SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int sam3d_body_get_cam(sam3d_body_ctx *ctx, float *out_cam_t_xyz,
                       float *out_focal_px)
{
    if (!ctx) return SAM3D_BODY_E_INVAL;
    if (out_cam_t_xyz) memcpy(out_cam_t_xyz, ctx->cam_t, sizeof(ctx->cam_t));
    if (out_focal_px)  *out_focal_px = ctx->focal_px;
    return ctx->focal_px > 0 ? SAM3D_BODY_E_OK : SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int sam3d_body_get_vertices(sam3d_body_ctx *ctx, float *out, int *out_v)
{
    if (!ctx || !out_v) return SAM3D_BODY_E_INVAL;
    *out_v = ctx->n_vertices;
    if (out && ctx->vertices)
        memcpy(out, ctx->vertices,
               (size_t)ctx->n_vertices * 3 * sizeof(float));
    return ctx->vertices ? SAM3D_BODY_E_OK : SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int sam3d_body_get_faces(sam3d_body_ctx *ctx, int32_t *out, int *out_f)
{
    if (!ctx || !out_f) return SAM3D_BODY_E_INVAL;
    *out_f = ctx->n_faces;
    if (out && ctx->faces)
        memcpy(out, ctx->faces,
               (size_t)ctx->n_faces * 3 * sizeof(int32_t));
    return ctx->faces ? SAM3D_BODY_E_OK : SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int sam3d_body_get_keypoints_3d(sam3d_body_ctx *ctx, float *out, int *out_k)
{
    if (!ctx || !out_k) return SAM3D_BODY_E_INVAL;
    *out_k = ctx->n_kp_3d;
    if (out && ctx->keypoints_3d)
        memcpy(out, ctx->keypoints_3d,
               (size_t)ctx->n_kp_3d * 3 * sizeof(float));
    return ctx->keypoints_3d ? SAM3D_BODY_E_OK : SAM3D_BODY_E_NOT_IMPLEMENTED;
}

int sam3d_body_get_keypoints_2d(sam3d_body_ctx *ctx, float *out, int *out_k)
{
    if (!ctx || !out_k) return SAM3D_BODY_E_INVAL;
    *out_k = ctx->n_kp_2d;
    if (out && ctx->keypoints_2d)
        memcpy(out, ctx->keypoints_2d,
               (size_t)ctx->n_kp_2d * 2 * sizeof(float));
    return ctx->keypoints_2d ? SAM3D_BODY_E_OK : SAM3D_BODY_E_NOT_IMPLEMENTED;
}

/* ---------------- debug overrides ---------------- */

int sam3d_body_debug_override_encoder(sam3d_body_ctx *ctx,
                                      const float *tokens, int n, int dim)
{
    if (!ctx || !tokens || n <= 0 || dim <= 0) return SAM3D_BODY_E_INVAL;
    f32_2d_free(&ctx->encoder_tokens);
    ctx->encoder_tokens.data = (float *)malloc((size_t)n * dim * sizeof(float));
    if (!ctx->encoder_tokens.data) return SAM3D_BODY_E_INVAL;
    memcpy(ctx->encoder_tokens.data, tokens,
           (size_t)n * dim * sizeof(float));
    ctx->encoder_tokens.n = n;
    ctx->encoder_tokens.c = dim;
    return SAM3D_BODY_E_OK;
}

int sam3d_body_debug_override_mhr_params(sam3d_body_ctx *ctx,
                                         const float *params, int n)
{
    if (!ctx || !params || n <= 0) return SAM3D_BODY_E_INVAL;
    free(ctx->mhr_params);
    ctx->mhr_params = (float *)malloc((size_t)n * sizeof(float));
    if (!ctx->mhr_params) return SAM3D_BODY_E_INVAL;
    memcpy(ctx->mhr_params, params, (size_t)n * sizeof(float));
    ctx->mhr_params_n = n;
    return SAM3D_BODY_E_OK;
}

int sam3d_body_debug_override_decoder_inputs(
    sam3d_body_ctx *ctx,
    const float *image_emb_chw,
    const float *image_pe_chw,
    int H, int W,
    const float *init_x,
    const float *init_xpe,
    const float *cam_int,
    const float *bbox_center,
    const float *bbox_scale,
    const float *ori_img_size,
    const float *img_size,
    const float *affine_trans,
    int use_intrin_center,
    float default_scale_factor)
{
    if (!ctx || !image_emb_chw || !image_pe_chw || !init_x || !init_xpe ||
        !cam_int || !bbox_center || !bbox_scale || !ori_img_size ||
        !img_size || !affine_trans || H <= 0 || W <= 0)
        return SAM3D_BODY_E_INVAL;

    int rc = ensure_decoder_loaded(ctx);
    if (rc != SAM3D_BODY_E_OK) return rc;

    const int Dc = ctx->decoder_model->kv_dim;
    const int Nq = 145;
    const int D  = ctx->decoder_model->dim;

    free(ctx->dec_image_emb_chw);
    free(ctx->dec_image_pe_chw);
    free(ctx->dec_init_x);
    free(ctx->dec_init_xpe);

    size_t img_bytes = (size_t)Dc * H * W * sizeof(float);
    size_t tok_bytes = (size_t)Nq * D  * sizeof(float);
    ctx->dec_image_emb_chw = (float *)malloc(img_bytes);
    ctx->dec_image_pe_chw  = (float *)malloc(img_bytes);
    ctx->dec_init_x        = (float *)malloc(tok_bytes);
    ctx->dec_init_xpe      = (float *)malloc(tok_bytes);
    if (!ctx->dec_image_emb_chw || !ctx->dec_image_pe_chw ||
        !ctx->dec_init_x || !ctx->dec_init_xpe) {
        free(ctx->dec_image_emb_chw); ctx->dec_image_emb_chw = NULL;
        free(ctx->dec_image_pe_chw);  ctx->dec_image_pe_chw  = NULL;
        free(ctx->dec_init_x);        ctx->dec_init_x        = NULL;
        free(ctx->dec_init_xpe);      ctx->dec_init_xpe      = NULL;
        return SAM3D_BODY_E_INVAL;
    }
    memcpy(ctx->dec_image_emb_chw, image_emb_chw, img_bytes);
    memcpy(ctx->dec_image_pe_chw,  image_pe_chw,  img_bytes);
    memcpy(ctx->dec_init_x,        init_x,        tok_bytes);
    memcpy(ctx->dec_init_xpe,      init_xpe,      tok_bytes);
    ctx->dec_H = H; ctx->dec_W = W;

    memcpy(ctx->dec_cam_batch.cam_int,      cam_int,      9 * sizeof(float));
    memcpy(ctx->dec_cam_batch.bbox_center,  bbox_center,  2 * sizeof(float));
    ctx->dec_cam_batch.bbox_scale = bbox_scale[0];
    memcpy(ctx->dec_cam_batch.ori_img_size, ori_img_size, 2 * sizeof(float));
    memcpy(ctx->dec_cam_batch.img_size,     img_size,     2 * sizeof(float));
    memcpy(ctx->dec_cam_batch.affine_trans, affine_trans, 6 * sizeof(float));
    ctx->dec_cam_batch.use_intrin_center    = use_intrin_center;
    ctx->dec_cam_batch.default_scale_factor = default_scale_factor;
    ctx->dec_in_set = 1;
    return SAM3D_BODY_E_OK;
}
