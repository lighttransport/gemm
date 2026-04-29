/* sam3d_runner.c — v1 CPU runner for SAM-3D-Objects. */

#include "sam3d_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

#define DINOV2_IMPLEMENTATION
#include "dinov2.h"

#define SAM3D_COND_FUSER_IMPLEMENTATION
#include "sam3d_cond_fuser.h"

#define SAM3D_SS_FLOW_DIT_IMPLEMENTATION
#include "sam3d_ss_flow_dit.h"

#define SPARSE3D_IMPLEMENTATION
#include "sparse3d.h"

#define SAM3D_GS_DECODER_IMPLEMENTATION
#include "sam3d_gs_decoder.h"

#define SAM3D_SLAT_DIT_IMPLEMENTATION
#include "sam3d_slat_dit.h"

#define T2_SS_DEC_IMPLEMENTATION
#include "trellis2_ss_decoder.h"

#include "sam3d_shortcut_solver.h"

#define SAM3D_E_NOT_IMPLEMENTED (-2)

typedef struct {
    float *data;
    int    n, c;
} f32_2d;

typedef struct {
    float *data;
    int    dims[4];
    int    ndim;
} f32_nd;

typedef struct {
    float   *feats;
    int32_t *coords;  /* [n, 4] */
    int      n, c;
} slat_buf;

struct sam3d_ctx {
    sam3d_config cfg;

    /* Resolved safetensors directory (cfg.safetensors_dir or inferred
     * from pipeline_yaml's parent dir). */
    char sft_dir[1024];

    /* Lazily-loaded submodules. */
    dinov2_model            *dinov2;
    sam3d_ppe_model         *ppe;
    sam3d_fuser_model       *fuser;
    sam3d_ss_flow_dit_model *ss_dit;
    t2_ss_dec               *ss_dec;
    sam3d_slat_dit_model    *slat_dit;
    sam3d_gs_decoder_model  *gs_decoder;

    /* Inputs. */
    uint8_t *rgba;      int   rgba_w, rgba_h;
    uint8_t *mask;      int   mask_w, mask_h;
    float   *pointmap;  int   pmap_w, pmap_h;

    /* Stage outputs. */
    f32_2d dinov2_tokens;   /* [2 * (1+n_patches), dim] — image + mask branch,
                             * register tokens dropped (Dino.forward parity) */
    f32_2d cond_tokens;     /* fuser output */
    f32_nd ss_latent;       /* [8, 16, 16, 16] */
    f32_nd occupancy;       /* [64, 64, 64] */
    slat_buf slat;
    f32_2d gaussians;       /* [n, 17] */
};

static void f32_2d_free(f32_2d *b) { free(b->data); b->data = NULL; b->n = b->c = 0; }
static void f32_nd_free(f32_nd *b) { free(b->data); b->data = NULL; b->ndim = 0; }
static void slat_free(slat_buf *b) {
    free(b->feats); free(b->coords);
    b->feats = NULL; b->coords = NULL; b->n = b->c = 0;
}

sam3d_ctx *sam3d_create(const sam3d_config *cfg)
{
    if (!cfg || !cfg->pipeline_yaml) return NULL;
    sam3d_ctx *c = (sam3d_ctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cfg = *cfg;
    /* Defaults from the shipped checkpoints' yaml: SS DiT is a
     * shortcut ODE (2 steps); SLAT DiT is standard flow-matching
     * (12 steps); SS cfg_strength on shape modality is 2.0
     * (SLAT cfg_strength is 0). */
    if (c->cfg.ss_steps   <= 0) c->cfg.ss_steps   = 2;
    if (c->cfg.slat_steps <= 0) c->cfg.slat_steps = 12;
    if (c->cfg.cfg_scale  <= 0) c->cfg.cfg_scale  = 2.0f;
    if (c->cfg.n_threads  <= 0) {
        const char *env = getenv("OMP_NUM_THREADS");
        c->cfg.n_threads = env ? atoi(env) : 1;
        if (c->cfg.n_threads < 1) c->cfg.n_threads = 1;
    }

    /* Resolve per-module safetensors directory. */
    if (cfg->safetensors_dir && cfg->safetensors_dir[0]) {
        snprintf(c->sft_dir, sizeof(c->sft_dir), "%s", cfg->safetensors_dir);
    } else {
        /* Fall back to siblingof(pipeline_yaml's dir)/safetensors/. */
        const char *slash = strrchr(cfg->pipeline_yaml, '/');
        if (slash) {
            size_t len = (size_t)(slash - cfg->pipeline_yaml);
            if (len >= sizeof(c->sft_dir)) len = sizeof(c->sft_dir) - 32;
            memcpy(c->sft_dir, cfg->pipeline_yaml, len);
            c->sft_dir[len] = '\0';
            /* sibling: strip trailing "/checkpoints" if present. */
            char *tail = strrchr(c->sft_dir, '/');
            if (tail && strcmp(tail, "/checkpoints") == 0) *tail = '\0';
            size_t cur = strlen(c->sft_dir);
            snprintf(c->sft_dir + cur, sizeof(c->sft_dir) - cur, "/safetensors");
        } else {
            snprintf(c->sft_dir, sizeof(c->sft_dir), "./safetensors");
        }
    }

    if (c->cfg.verbose) {
        fprintf(stderr, "[sam3d] create: pipeline_yaml=%s seed=%llu "
                        "ss_steps=%d slat_steps=%d cfg=%.1f threads=%d\n",
                cfg->pipeline_yaml, (unsigned long long)cfg->seed,
                c->cfg.ss_steps, c->cfg.slat_steps, c->cfg.cfg_scale,
                c->cfg.n_threads);
        fprintf(stderr, "[sam3d] safetensors_dir=%s\n", c->sft_dir);
    }
    return c;
}

const char *sam3d_safetensors_dir(const sam3d_ctx *ctx)
{
    return ctx ? ctx->sft_dir : NULL;
}

void sam3d_destroy(sam3d_ctx *ctx)
{
    if (!ctx) return;
    if (ctx->dinov2) dinov2_free(ctx->dinov2);
    if (ctx->ppe)    sam3d_ppe_free(ctx->ppe);
    if (ctx->fuser)  sam3d_fuser_free(ctx->fuser);
    if (ctx->ss_dit) sam3d_ss_flow_dit_free(ctx->ss_dit);
    if (ctx->ss_dec) t2_ss_dec_free(ctx->ss_dec);
    if (ctx->slat_dit) sam3d_slat_dit_free(ctx->slat_dit);
    if (ctx->gs_decoder) sam3d_gs_decoder_free(ctx->gs_decoder);
    free(ctx->rgba); free(ctx->mask); free(ctx->pointmap);
    f32_2d_free(&ctx->dinov2_tokens);
    f32_2d_free(&ctx->cond_tokens);
    f32_nd_free(&ctx->ss_latent);
    f32_nd_free(&ctx->occupancy);
    slat_free(&ctx->slat);
    f32_2d_free(&ctx->gaussians);
    free(ctx);
}

static int dup_buf_u8(uint8_t **dst, int *dw, int *dh,
                      const uint8_t *src, int w, int h, int ch)
{
    free(*dst);
    size_t n = (size_t)w * h * ch;
    *dst = (uint8_t *)malloc(n);
    if (!*dst) return -1;
    memcpy(*dst, src, n);
    *dw = w; *dh = h;
    return 0;
}

int sam3d_set_image_rgba(sam3d_ctx *ctx, const uint8_t *rgba, int w, int h)
{
    if (!ctx || !rgba || w <= 0 || h <= 0) return -1;
    return dup_buf_u8(&ctx->rgba, &ctx->rgba_w, &ctx->rgba_h, rgba, w, h, 4);
}

int sam3d_set_mask(sam3d_ctx *ctx, const uint8_t *mask, int w, int h)
{
    if (!ctx || !mask || w <= 0 || h <= 0) return -1;
    return dup_buf_u8(&ctx->mask, &ctx->mask_w, &ctx->mask_h, mask, w, h, 1);
}

int sam3d_set_pointmap(sam3d_ctx *ctx, const float *pmap, int w, int h)
{
    if (!ctx || !pmap || w <= 0 || h <= 0) return -1;
    free(ctx->pointmap);
    size_t n = (size_t)w * h * 3;
    ctx->pointmap = (float *)malloc(n * sizeof(float));
    if (!ctx->pointmap) return -1;
    memcpy(ctx->pointmap, pmap, n * sizeof(float));
    ctx->pmap_w = w; ctx->pmap_h = h;
    return 0;
}

/* ---- Stage entrypoints ---- */

static int f32_2d_alloc(f32_2d *b, int n, int c) {
    free(b->data);
    b->data = (float *)malloc((size_t)n * c * sizeof(float));
    if (!b->data) return -1;
    b->n = n; b->c = c;
    return 0;
}

/* Preprocess a 1-channel mask → normalized fp32 CHW at [3 × oh × ow]
 * as a binary-mask-replicated-to-3-channels and ImageNet-normalized,
 * matching the pytorch `rgb_image_mask = (alpha > 0).float()` path
 * followed by DINOv2's 1→3 channel repeat and (x - mean) / std.
 * pad_to_square_centered + nearest-neighbor Resize to (oh, ow).
 * Caller allocates out_chw as 3*oh*ow floats. */
static void sam3d_prep_mask_branch_f32(float *out_chw,
                                       const uint8_t *mask, int mw, int mh,
                                       int ow, int oh,
                                       const float mean[3],
                                       const float std[3])
{
    int s = (mw > mh) ? mw : mh;
    int ox = (s - mw) / 2;
    int oy = (s - mh) / 2;
    float scale_y = (float)s / (float)oh;
    float scale_x = (float)s / (float)ow;
    /* Pytorch F.interpolate mode="nearest" (legacy/backward-compat):
     * src_idx = floor(dst_idx * src / dst). NOT the centered variant.
     * my/mx clamp to [0, s-1] is unneeded: y<oh and scale_y=s/oh give
     * my<s; non-negativity is guaranteed by y>=0. */
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < oh; y++) {
        int my_src = (int)floorf((float)y * scale_y) - oy;
        for (int x = 0; x < ow; x++) {
            int mx_src = (int)floorf((float)x * scale_x) - ox;
            float v = 0.0f;
            if (my_src >= 0 && my_src < mh && mx_src >= 0 && mx_src < mw)
                v = (mask[my_src * mw + mx_src] > 0) ? 1.0f : 0.0f;
            for (int c = 0; c < 3; c++) {
                out_chw[c * oh * ow + y * ow + x] = (v - mean[c]) / std[c];
            }
        }
    }
}

/* Preprocess RGBA → normalized fp32 CHW at [3 × oh × ow]. Matches the
 * pytorch ss_preprocessor "full" branch: pad_to_square_centered →
 * bilinear Resize(ow) → div 255 → (x - mean) / std. align_corners=False
 * matches F.interpolate. Caller allocates out_chw as 3*oh*ow floats. */
static void sam3d_prep_rgb_branch_f32(float *out_chw,
                                      const uint8_t *rgba, int iw, int ih,
                                      int ow, int oh,
                                      const float mean[3], const float std[3])
{
    int s = (iw > ih) ? iw : ih;
    int ox = (s - iw) / 2;
    int oy = (s - ih) / 2;
    float scale_y = (float)s / (float)oh;
    float scale_x = (float)s / (float)ow;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < oh; y++) {
        float fy = ((float)y + 0.5f) * scale_y - 0.5f;
        int y0 = (int)floorf(fy);
        int y1 = y0 + 1;
        float dy = fy - (float)y0;
        int y0s = y0 - oy, y1s = y1 - oy;

        for (int x = 0; x < ow; x++) {
            float fx = ((float)x + 0.5f) * scale_x - 0.5f;
            int x0 = (int)floorf(fx);
            int x1 = x0 + 1;
            float dx = fx - (float)x0;
            int x0s = x0 - ox, x1s = x1 - ox;

            for (int c = 0; c < 3; c++) {
                float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
                if (y0s >= 0 && y0s < ih) {
                    if (x0s >= 0 && x0s < iw)
                        a00 = (float)rgba[(y0s * iw + x0s) * 4 + c];
                    if (x1s >= 0 && x1s < iw)
                        a01 = (float)rgba[(y0s * iw + x1s) * 4 + c];
                }
                if (y1s >= 0 && y1s < ih) {
                    if (x0s >= 0 && x0s < iw)
                        a10 = (float)rgba[(y1s * iw + x0s) * 4 + c];
                    if (x1s >= 0 && x1s < iw)
                        a11 = (float)rgba[(y1s * iw + x1s) * 4 + c];
                }
                float v = a00 * (1 - dy) * (1 - dx)
                        + a01 * (1 - dy) * dx
                        + a10 * dy       * (1 - dx)
                        + a11 * dy       * dx;
                v /= 255.0f;
                out_chw[c * oh * ow + y * ow + x] = (v - mean[c]) / std[c];
            }
        }
    }
}

int sam3d_run_dinov2(sam3d_ctx *ctx)
{
    if (!ctx || !ctx->rgba) return -1;

    /* Lazy-load. */
    if (!ctx->dinov2) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_dinov2.safetensors", ctx->sft_dir);
        ctx->dinov2 = dinov2_load_safetensors(path);
        if (!ctx->dinov2) {
            fprintf(stderr, "[sam3d] dinov2: failed to load %s\n", path);
            return -3;
        }
    }

    int img_sz = ctx->dinov2->image_size;   /* 518 */
    int n_reg  = ctx->dinov2->n_register;   /* 4 */
    int dim    = ctx->dinov2->dim;          /* 1024 */
    int nthr   = ctx->cfg.n_threads;

    /* Image branch: pad_to_square_centered → bilinear resize → ImageNet
     * normalization, all in fp32 to avoid uint8 quantization drift
     * amplified through 24 transformer blocks. */
    float *chw_img = (float *)malloc((size_t)3 * img_sz * img_sz * sizeof(float));
    if (!chw_img) return -5;
    sam3d_prep_rgb_branch_f32(chw_img, ctx->rgba, ctx->rgba_w, ctx->rgba_h,
                              img_sz, img_sz,
                              ctx->dinov2->image_mean, ctx->dinov2->image_std);
    dinov2_result r_img = dinov2_encode_f32(ctx->dinov2, chw_img, img_sz, img_sz, nthr);
    free(chw_img);
    if (!r_img.features) return -4;
    dinov2_result_drop_registers(&r_img, n_reg);

    /* Mask branch: binary mask broadcast to 3 channels, then same
     * pad_to_square + resize + ImageNet-norm path. pytorch's DINOv2
     * wrapper does the 1→3 broadcast internally; we do it here so the
     * downstream encoder path is identical for both branches. */
    dinov2_result r_msk = {0};
    if (ctx->mask) {
        float *chw_msk = (float *)malloc((size_t)3 * img_sz * img_sz * sizeof(float));
        if (!chw_msk) { dinov2_result_free(&r_img); return -5; }
        sam3d_prep_mask_branch_f32(chw_msk, ctx->mask, ctx->mask_w, ctx->mask_h,
                                   img_sz, img_sz,
                                   ctx->dinov2->image_mean, ctx->dinov2->image_std);
        r_msk = dinov2_encode_f32(ctx->dinov2, chw_msk, img_sz, img_sz, nthr);
        free(chw_msk);
        if (!r_msk.features) { dinov2_result_free(&r_img); return -4; }
        dinov2_result_drop_registers(&r_msk, n_reg);
    }

    /* Concatenate image + mask tokens along the token axis. */
    int n_tok  = r_img.n_tokens;            /* 1 + n_patches after drop */
    int n_branches = r_msk.features ? 2 : 1;
    int n_total = n_branches * n_tok;
    if (f32_2d_alloc(&ctx->dinov2_tokens, n_total, dim) != 0) {
        dinov2_result_free(&r_img); dinov2_result_free(&r_msk);
        return -5;
    }
    memcpy(ctx->dinov2_tokens.data, r_img.features,
           (size_t)n_tok * dim * sizeof(float));
    if (r_msk.features) {
        memcpy(ctx->dinov2_tokens.data + (size_t)n_tok * dim, r_msk.features,
               (size_t)n_tok * dim * sizeof(float));
    }
    dinov2_result_free(&r_img);
    dinov2_result_free(&r_msk);
    return 0;
}
int sam3d_run_cond_fuser(sam3d_ctx *ctx)
{
    if (!ctx) return -1;
    if (!ctx->dinov2_tokens.data) {
        fprintf(stderr, "[sam3d] cond_fuser: dinov2_tokens not set — run "
                        "sam3d_run_dinov2() (or override) first\n");
        return -1;
    }
    int nthr = ctx->cfg.n_threads;

    /* Lazy-load PPE + fuser. */
    if (!ctx->ppe) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_point_patch_embed.safetensors",
                 ctx->sft_dir);
        ctx->ppe = sam3d_ppe_load_safetensors(path);
        if (!ctx->ppe) return -3;
    }
    if (!ctx->fuser) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_cond_fuser.safetensors",
                 ctx->sft_dir);
        ctx->fuser = sam3d_fuser_load_safetensors(path);
        if (!ctx->fuser) return -3;
    }

    int dim_in  = ctx->dinov2_tokens.c;                 /* 1024 */
    int D_out   = ctx->fuser->embed_dim_out;            /* 1024 */
    int dino_n  = ctx->dinov2_tokens.n;                 /* 2*1370 */
    int branch  = dino_n / 2;                           /* 1370 */

    /* v1 only runs the "full" variants: 1× dino image + 1× dino mask +
     * 1× pointmap. Cropped variants (+ another pass of each modality)
     * are deferred until MoGe / crop preprocessing lands. */
    const int pos = SAM3D_FUSER_POS_FULL;

    float *dino_img = sam3d_fuser_project(ctx->fuser, SAM3D_FUSER_MOD_DINO_IMG,
                                          ctx->dinov2_tokens.data,
                                          branch, nthr);
    if (!dino_img) return -5;
    sam3d_fuser_add_pos(ctx->fuser, pos, dino_img, branch);

    float *dino_msk = sam3d_fuser_project(ctx->fuser, SAM3D_FUSER_MOD_DINO_MSK,
                                          ctx->dinov2_tokens.data
                                              + (size_t)branch * dim_in,
                                          branch, nthr);
    if (!dino_msk) { free(dino_img); return -5; }
    sam3d_fuser_add_pos(ctx->fuser, pos, dino_msk, branch);

    /* Point branch — if no pointmap is available, cond tokens will
     * carry only the two dino branches. */
    float *point = NULL;
    int point_n = 0;
    if (ctx->pointmap) {
        float *ppe_tokens = sam3d_ppe_encode(ctx->ppe, ctx->pointmap,
                                             ctx->pmap_h, ctx->pmap_w,
                                             NULL, nthr);
        if (!ppe_tokens) { free(dino_img); free(dino_msk); return -5; }
        int n_ppe = ctx->ppe->num_patches * ctx->ppe->num_patches;
        point = sam3d_fuser_project(ctx->fuser, SAM3D_FUSER_MOD_POINT,
                                    ppe_tokens, n_ppe, nthr);
        free(ppe_tokens);
        if (!point) { free(dino_img); free(dino_msk); return -5; }
        sam3d_fuser_add_pos(ctx->fuser, pos, point, n_ppe);
        point_n = n_ppe;
    }

    /* Concatenate along the token axis. */
    int n_total = branch * 2 + point_n;
    if (f32_2d_alloc(&ctx->cond_tokens, n_total, D_out) != 0) {
        free(dino_img); free(dino_msk); free(point);
        return -5;
    }
    float *dst = ctx->cond_tokens.data;
    memcpy(dst, dino_img, (size_t)branch * D_out * sizeof(float));
    memcpy(dst + (size_t)branch * D_out, dino_msk,
           (size_t)branch * D_out * sizeof(float));
    if (point) {
        memcpy(dst + (size_t)2 * branch * D_out, point,
               (size_t)point_n * D_out * sizeof(float));
    }
    free(dino_img); free(dino_msk); free(point);
    return 0;
}
/* xorshift64* PRNG + Box-Muller → standard normal. Deterministic per seed. */
static inline uint64_t sam3d_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static inline float sam3d_rng_u01(uint64_t *state) {
    uint64_t r = sam3d_rng_next(state) >> 11;
    return (float)((double)r * (1.0 / 9007199254740992.0));
}
static void sam3d_fill_randn(float *buf, int n, uint64_t *state) {
    for (int i = 0; i < n; i += 2) {
        float u1 = sam3d_rng_u01(state); if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = sam3d_rng_u01(state);
        float r = sqrtf(-2.0f * logf(u1));
        float a = 6.2831853f * u2;
        buf[i] = r * cosf(a);
        if (i + 1 < n) buf[i + 1] = r * sinf(a);
    }
}

static const int sam3d_ss_lat_elts[5] = { 4096 * 8, 1 * 6, 1 * 3, 1 * 3, 1 * 1 };

int sam3d_run_ss_dit(sam3d_ctx *ctx)
{
    if (!ctx) return -1;
    if (!ctx->cond_tokens.data) {
        fprintf(stderr, "[sam3d] ss_dit: cond tokens not set — run "
                        "sam3d_run_cond_fuser() (or override) first\n");
        return -1;
    }
    if (!ctx->ss_dit) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_ss_dit.safetensors", ctx->sft_dir);
        ctx->ss_dit = sam3d_ss_flow_dit_load_safetensors(path);
        if (!ctx->ss_dit) {
            fprintf(stderr, "[sam3d] ss_dit: failed to load %s\n", path);
            return -3;
        }
    }
    sam3d_ss_flow_dit_model *m = ctx->ss_dit;
    int nthr = ctx->cfg.n_threads;
    int steps = ctx->cfg.ss_steps;

    /* Allocate one buffer per modality. */
    float *lat[SAM3D_SS_DIT_N_LATENTS] = {0};
    float *vel[SAM3D_SS_DIT_N_LATENTS] = {0};
    float *vel_u[SAM3D_SS_DIT_N_LATENTS] = {0};
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        lat[i] = (float *)malloc((size_t)sam3d_ss_lat_elts[i] * sizeof(float));
        vel[i] = (float *)malloc((size_t)sam3d_ss_lat_elts[i] * sizeof(float));
        vel_u[i] = (float *)malloc((size_t)sam3d_ss_lat_elts[i] * sizeof(float));
        if (!lat[i] || !vel[i] || !vel_u[i]) goto oom;
    }

    /* Seeded Gaussian init. Upstream ShortCut.generate_iter samples from
     * randn at t=1 and denoises toward t=0. */
    uint64_t rng = ctx->cfg.seed ? ctx->cfg.seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++)
        sam3d_fill_randn(lat[i], sam3d_ss_lat_elts[i], &rng);

    /* Upstream pointmap inference runs the shortcut generator with
     * no_shortcut=True and ss_rescale_t=3 unless overridden by the pipeline.
     * That means pure flow matching (d=0), not a shortcut jump. */
    float *times = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    sam3d_shortcut_make_times(times, steps, 3.0f, /*reversed=*/0);
    float d = sam3d_shortcut_d(steps, /*no_shortcut=*/1);

    /* time_scale=1000 from ss_generator.yaml. Upstream `_generate_dynamics`
     * multiplies t,d by time_scale before calling the model; the Euler step
     * stays in caller-visible [0,1] space. The model's TimestepEmbedder was
     * trained on t in [0, 1000] — passing raw t in [0,1] gives a near-zero
     * sinusoidal embedding and yields all-negative occupancy. */
    const float TIME_SCALE = m->time_scale;

    float *zero_cond = NULL;
    if (ctx->cfg.cfg_scale > 0.0f) {
        zero_cond = (float *)calloc((size_t)ctx->cond_tokens.n * ctx->cond_tokens.c,
                                    sizeof(float));
        if (!zero_cond) goto oom;
    }

    int cfg_steps = 0;
    for (int s = 0; s < steps; s++) {
        float t = times[s];
        float dt = times[s + 1] - times[s];
        float ts = t * TIME_SCALE;

        if (sam3d_ss_flow_dit_forward(m,
                                      (const float *const *)lat, vel,
                                      ctx->cond_tokens.data, ctx->cond_tokens.n,
                                      ts, d * TIME_SCALE, nthr) != 0) {
            fprintf(stderr, "[sam3d] ss_dit: forward failed at step %d\n", s);
            free(times); free(zero_cond);
            for (int k = 0; k < SAM3D_SS_DIT_N_LATENTS; k++) {
                free(lat[k]); free(vel[k]); free(vel_u[k]);
            }
            return -4;
        }
        if (zero_cond && sam3d_shortcut_cfg_active(ts, 0.0f, 500.0f)) {
            if (sam3d_ss_flow_dit_forward(m,
                                          (const float *const *)lat, vel_u,
                                          zero_cond, ctx->cond_tokens.n,
                                          ts, d * TIME_SCALE, nthr) != 0) {
                fprintf(stderr, "[sam3d] ss_dit: uncond forward failed at step %d\n", s);
                free(times); free(zero_cond);
                for (int k = 0; k < SAM3D_SS_DIT_N_LATENTS; k++) {
                    free(lat[k]); free(vel[k]); free(vel_u[k]);
                }
                return -4;
            }
            for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
                sam3d_shortcut_cfg_combine(vel[i], vel[i], vel_u[i],
                                           ctx->cfg.cfg_scale,
                                           sam3d_ss_lat_elts[i]);
            }
            cfg_steps++;
        }
        for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++)
            sam3d_shortcut_euler_step(lat[i], vel[i], dt, sam3d_ss_lat_elts[i]);
    }
    free(times); free(zero_cond);
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) { free(vel[i]); free(vel_u[i]); }

    /* Persist shape latent as NCDHW [8,16,16,16] in ctx->ss_latent
     * (ready for the SS-VAE decoder). Source layout from DiT is [N=4096,C=8]. */
    ctx->ss_latent.ndim = 4;
    ctx->ss_latent.dims[0] = 8; ctx->ss_latent.dims[1] = 16;
    ctx->ss_latent.dims[2] = 16; ctx->ss_latent.dims[3] = 16;
    free(ctx->ss_latent.data);
    ctx->ss_latent.data = (float *)malloc((size_t)8 * 4096 * sizeof(float));
    if (!ctx->ss_latent.data) goto oom;
    for (int n = 0; n < 4096; n++)
        for (int c = 0; c < 8; c++)
            ctx->ss_latent.data[c * 4096 + n] = lat[SAM3D_SS_LAT_SHAPE][n * 8 + c];

    if (ctx->cfg.verbose) {
        float mn = ctx->ss_latent.data[0], mx = ctx->ss_latent.data[0];
        double sum = 0.0;
        for (int i = 0; i < 8 * 4096; i++) {
            float v = ctx->ss_latent.data[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        fprintf(stderr, "[sam3d] ss_dit: latent min=%.6g max=%.6g mean=%.6g d=%.3g rescale_t=3 reversed=0 cfg_steps=%d\n",
                mn, mx, sum / (8.0 * 4096.0), d, cfg_steps);
    }

    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) free(lat[i]);
    return 0;

oom:
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        free(lat[i]); free(vel[i]); free(vel_u[i]);
    }
    return -5;
}

int sam3d_run_ss_decode(sam3d_ctx *ctx)
{
    if (!ctx) return -1;
    if (!ctx->ss_latent.data) {
        fprintf(stderr, "[sam3d] ss_decode: ss_latent not set — run "
                        "sam3d_run_ss_dit() (or override) first\n");
        return -1;
    }
    if (ctx->ss_latent.ndim != 4 ||
        ctx->ss_latent.dims[0] != 8 ||
        ctx->ss_latent.dims[1] != 16 ||
        ctx->ss_latent.dims[2] != 16 ||
        ctx->ss_latent.dims[3] != 16) {
        fprintf(stderr, "[sam3d] ss_decode: expected [8,16,16,16], got "
                        "ndim=%d [%d,%d,%d,%d]\n",
                ctx->ss_latent.ndim,
                ctx->ss_latent.dims[0], ctx->ss_latent.dims[1],
                ctx->ss_latent.dims[2], ctx->ss_latent.dims[3]);
        return -2;
    }
    if (!ctx->ss_dec) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_ss_decoder.safetensors", ctx->sft_dir);
        ctx->ss_dec = t2_ss_dec_load(path);
        if (!ctx->ss_dec) {
            fprintf(stderr, "[sam3d] ss_decode: failed to load %s\n", path);
            return -3;
        }
    }
    float *logits = t2_ss_dec_forward(ctx->ss_dec, ctx->ss_latent.data,
                                       ctx->cfg.n_threads);
    if (!logits) return -4;

    f32_nd_free(&ctx->occupancy);
    ctx->occupancy.ndim = 3;
    ctx->occupancy.dims[0] = 64;
    ctx->occupancy.dims[1] = 64;
    ctx->occupancy.dims[2] = 64;
    ctx->occupancy.data = logits;   /* take ownership */
    if (ctx->cfg.verbose) {
        float mn = logits[0], mx = logits[0];
        double sum = 0.0;
        int pos = 0;
        for (int i = 0; i < 64 * 64 * 64; i++) {
            float v = logits[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
            pos += (v > 0.0f);
        }
        fprintf(stderr, "[sam3d] ss_decode: occupancy min=%.6g max=%.6g mean=%.6g pos=%d\n",
                mn, mx, sum / (64.0 * 64.0 * 64.0), pos);
    }
    return 0;
}

int sam3d_run_slat_dit(sam3d_ctx *ctx)
{
    if (!ctx) return -1;
    if (!ctx->occupancy.data) {
        fprintf(stderr, "[sam3d] slat_dit: occupancy not set — run "
                        "sam3d_run_ss_decode() (or override) first\n");
        return -1;
    }
    if (!ctx->cond_tokens.data) {
        fprintf(stderr, "[sam3d] slat_dit: cond tokens not set\n");
        return -1;
    }
    if (!ctx->slat_dit) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_slat_dit.safetensors", ctx->sft_dir);
        ctx->slat_dit = sam3d_slat_dit_load_safetensors(path);
        if (!ctx->slat_dit) {
            fprintf(stderr, "[sam3d] slat_dit: failed to load %s\n", path);
            return -3;
        }
    }
    sam3d_slat_dit_model *m = ctx->slat_dit;
    int nthr = ctx->cfg.n_threads;

    /* Voxel prune: argwhere(occ > 0). Grid is 64³ NCDHW [1,D,H,W] but we
     * stored it as 3D (D, H, W). Keep coords in (b,z,y,x) = (0,d,h,w). */
    int D = ctx->occupancy.dims[0];
    int H = ctx->occupancy.dims[1];
    int W = ctx->occupancy.dims[2];
    int cap = 0;
    for (int i = 0, n = D * H * W; i < n; i++)
        if (ctx->occupancy.data[i] > 0.0f) cap++;
    if (cap == 0) {
        fprintf(stderr, "[sam3d] slat_dit: occupancy is fully negative — "
                        "no voxels to decode\n");
        return -2;
    }

    int32_t *coords = (int32_t *)malloc((size_t)cap * 4 * sizeof(int32_t));
    float   *feats  = (float *)calloc((size_t)cap * m->in_channels, sizeof(float));
    if (!coords || !feats) { free(coords); free(feats); return -5; }

    int k = 0;
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (ctx->occupancy.data[(z * H + y) * W + x] > 0.0f) {
                    coords[k * 4 + 0] = 0;
                    coords[k * 4 + 1] = z;
                    coords[k * 4 + 2] = y;
                    coords[k * 4 + 3] = x;
                    k++;
                }

    /* Seeded Gaussian init for the SLAT latent tokens (x_T ~ N(0, I)). */
    uint64_t rng = (ctx->cfg.seed ^ 0x5851F42D4C957F2DULL);
    sam3d_fill_randn(feats, cap * m->in_channels, &rng);

    sp3d_tensor *x = sp3d_create(coords, feats, cap, m->in_channels, 1);
    if (!x) { free(coords); free(feats); return -5; }

    /* Flow-matching loop (strength=0 per yaml — no CFG on SLAT).
     * time_scale=1000 from slat_generator.yaml: upstream `_generate_dynamics`
     * multiplies t by time_scale before calling the model. */
    const float TIME_SCALE = 1000.0f;
    int steps = ctx->cfg.slat_steps;
    for (int s = 0; s < steps; s++) {
        float t = 1.0f - (float)s / (float)steps;
        if (sam3d_slat_dit_forward(m, &x, t * TIME_SCALE,
                                    ctx->cond_tokens.data, ctx->cond_tokens.n,
                                    nthr) != 0) {
            fprintf(stderr, "[sam3d] slat_dit: forward failed at step %d "
                            "(NOTE: forward is reserved for task #44 — "
                            "currently a stub)\n", s);
            sp3d_free(x); free(coords); free(feats);
            return -4;
        }
    }

    slat_free(&ctx->slat);
    ctx->slat.n = cap;
    ctx->slat.c = m->out_channels;
    ctx->slat.coords = (int32_t *)malloc((size_t)cap * 4 * sizeof(int32_t));
    ctx->slat.feats  = (float *)malloc((size_t)cap * m->out_channels * sizeof(float));
    if (!ctx->slat.coords || !ctx->slat.feats) {
        slat_free(&ctx->slat); sp3d_free(x); free(coords); free(feats);
        return -5;
    }
    memcpy(ctx->slat.coords, coords, (size_t)cap * 4 * sizeof(int32_t));
    memcpy(ctx->slat.feats,  x->feats,
           (size_t)cap * m->out_channels * sizeof(float));

    /* SLAT un-normalization: pytorch inference_pipeline does
     *   slat = slat * slat_std + slat_mean
     * between the flow DiT output and the GS decoder input. The stats are
     * 8-dim (one per feature channel) and live in pipeline.yaml; we hardcode
     * them here — they are fixed for the shipped sam-3d-objects checkpoint. */
    static const float SLAT_MEAN[8] = {
         0.12211431f,  0.37204156f, -1.26521907f, -2.05276058f,
        -3.10432536f, -0.11294304f, -0.85146744f,  0.45506954f,
    };
    static const float SLAT_STD[8] = {
         2.37326008f,  2.13174402f,  2.2413953f,   2.30589401f,
         2.1191894f,   1.8969511f,   2.41684989f,  2.08374642f,
    };
    if (m->out_channels == 8) {
        int C = m->out_channels;
        for (int i = 0; i < cap; i++) {
            float *r = ctx->slat.feats + (size_t)i * C;
            for (int c = 0; c < C; c++) r[c] = r[c] * SLAT_STD[c] + SLAT_MEAN[c];
        }
    }

    sp3d_free(x); free(coords); free(feats);
    return 0;
}

/* log(softplus(x)) computed stably across the full x range.
 *   x →  +∞ : softplus(x) ≈ x              →  log ≈ log(x)
 *   x →  0  : softplus(x) = log1p(e^x)     →  log of that
 *   x → -∞ : softplus(x) ≈ e^x             →  log ≈ x
 * `logf(1 + expf(x))` underflows when expf(x) is below fp32
 * ULP-at-1 (~1e-7); `log1pf` stays accurate for small arguments. */
static float sam3d_log_softplus(float x) {
    if (x >  20.0f) return logf(x);
    if (x < -15.0f) return x;
    return logf(log1pf(expf(x)));
}

int sam3d_run_slat_gs_decode(sam3d_ctx *ctx)
{
    if (!ctx || !ctx->slat.feats || !ctx->slat.coords) {
        fprintf(stderr, "[sam3d] slat_gs_decode: SLAT tokens not set — run "
                        "sam3d_run_slat_dit() (or override) first\n");
        return -1;
    }

    if (!ctx->gs_decoder) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_slat_gs_decoder.safetensors",
                 ctx->sft_dir);
        ctx->gs_decoder = sam3d_gs_decoder_load_safetensors(path);
        if (!ctx->gs_decoder) {
            fprintf(stderr, "[sam3d] slat_gs_decode: failed to load %s\n", path);
            return -3;
        }
    }
    sam3d_gs_decoder_model *m = ctx->gs_decoder;
    int N = ctx->slat.n;
    int G = m->num_gaussians;
    int total = N * G;
    int nthr = ctx->cfg.n_threads;

    sp3d_tensor *x = sp3d_create(ctx->slat.coords, ctx->slat.feats,
                                  N, ctx->slat.c, 1);
    if (!x) return -5;

    float *out_feats = NULL;
    if (sam3d_gs_decoder_transformer(m, x, &out_feats, nthr) != 0) {
        sp3d_free(x);
        return -4;
    }

    float *xyz = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *dc  = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *scl = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *rot = (float *)malloc((size_t)total * 4 * sizeof(float));
    float *op  = (float *)malloc((size_t)total     * sizeof(float));
    if (!xyz || !dc || !scl || !rot || !op) {
        free(xyz); free(dc); free(scl); free(rot); free(op);
        free(out_feats); sp3d_free(x);
        return -5;
    }
    sam3d_gs_decoder_to_representation(m, ctx->slat.coords, out_feats, N,
                                        xyz, dc, scl, rot, op);
    free(out_feats);
    sp3d_free(x);

    if (f32_2d_alloc(&ctx->gaussians, total, SAM3D_GS_STRIDE) != 0) {
        free(xyz); free(dc); free(scl); free(rot); free(op);
        return -5;
    }

    /* Convert raw representation to INRIA-PLY storage convention:
     *   opacity → raw + opacity_bias                 (pre-sigmoid logit)
     *   scale   → log(softplus(raw + inv_softplus(scaling_bias)))
     * Rotation/xyz/f_dc pass through unchanged. */
    const float inv_sp_sb = logf(expf(m->scaling_bias) - 1.0f);
    for (int i = 0; i < total; i++) {
        float *row = ctx->gaussians.data + (size_t)i * SAM3D_GS_STRIDE;
        row[0] = xyz[i * 3 + 0];
        row[1] = xyz[i * 3 + 1];
        row[2] = xyz[i * 3 + 2];
        row[3] = row[4] = row[5] = 0.0f;         /* normals */
        row[6] = dc[i * 3 + 0];
        row[7] = dc[i * 3 + 1];
        row[8] = dc[i * 3 + 2];
        row[9] = op[i] + m->opacity_bias;
        for (int a = 0; a < 3; a++) {
            row[10 + a] = sam3d_log_softplus(scl[i * 3 + a] + inv_sp_sb);
        }
        row[13] = rot[i * 4 + 0];
        row[14] = rot[i * 4 + 1];
        row[15] = rot[i * 4 + 2];
        row[16] = rot[i * 4 + 3];
    }

    free(xyz); free(dc); free(scl); free(rot); free(op);
    return 0;
}

/* ---- Accessors ---- */

static int copy_2d(const f32_2d *src, float *out, int *on, int *oc) {
    if (!src->data) return -1;
    if (on) *on = src->n;
    if (oc) *oc = src->c;
    if (out) memcpy(out, src->data, (size_t)src->n * src->c * sizeof(float));
    return 0;
}
static int copy_nd(const f32_nd *src, float *out, int *odims) {
    if (!src->data) return -1;
    size_t total = 1;
    for (int i = 0; i < src->ndim; i++) {
        if (odims) odims[i] = src->dims[i];
        total *= (size_t)src->dims[i];
    }
    if (out) memcpy(out, src->data, total * sizeof(float));
    return 0;
}

int sam3d_get_dinov2_tokens(sam3d_ctx *ctx, float *o, int *n, int *c)
    { return copy_2d(&ctx->dinov2_tokens, o, n, c); }
int sam3d_get_cond_tokens(sam3d_ctx *ctx, float *o, int *n, int *c)
    { return copy_2d(&ctx->cond_tokens, o, n, c); }
int sam3d_get_ss_latent(sam3d_ctx *ctx, float *o, int *dims)
    { return copy_nd(&ctx->ss_latent, o, dims); }
int sam3d_get_occupancy(sam3d_ctx *ctx, float *o, int *dims)
    { return copy_nd(&ctx->occupancy, o, dims); }

int sam3d_get_slat_tokens(sam3d_ctx *ctx, float *of, int32_t *oc,
                          int *on, int *oc_dim)
{
    if (!ctx->slat.feats) return -1;
    if (on) *on = ctx->slat.n;
    if (oc_dim) *oc_dim = ctx->slat.c;
    if (of) memcpy(of, ctx->slat.feats,
                   (size_t)ctx->slat.n * ctx->slat.c * sizeof(float));
    if (oc) memcpy(oc, ctx->slat.coords,
                   (size_t)ctx->slat.n * 4 * sizeof(int32_t));
    return 0;
}

int sam3d_get_gaussians(sam3d_ctx *ctx, float *o, int *n)
{
    if (!ctx->gaussians.data) return -1;
    if (n) *n = ctx->gaussians.n;
    if (o) memcpy(o, ctx->gaussians.data,
                  (size_t)ctx->gaussians.n * ctx->gaussians.c * sizeof(float));
    return 0;
}

/* ---- Debug overrides ---- */

static int set_2d(f32_2d *dst, const float *src, int n, int c) {
    free(dst->data);
    dst->data = (float *)malloc((size_t)n * c * sizeof(float));
    if (!dst->data) return -1;
    memcpy(dst->data, src, (size_t)n * c * sizeof(float));
    dst->n = n; dst->c = c;
    return 0;
}
static int set_nd(f32_nd *dst, const float *src, const int *dims, int ndim) {
    free(dst->data);
    size_t total = 1;
    for (int i = 0; i < ndim; i++) { dst->dims[i] = dims[i]; total *= (size_t)dims[i]; }
    dst->ndim = ndim;
    dst->data = (float *)malloc(total * sizeof(float));
    if (!dst->data) return -1;
    memcpy(dst->data, src, total * sizeof(float));
    return 0;
}

int sam3d_debug_override_dinov2(sam3d_ctx *ctx, const float *t, int n, int c)
    { return set_2d(&ctx->dinov2_tokens, t, n, c); }
int sam3d_debug_override_cond(sam3d_ctx *ctx, const float *t, int n, int c)
    { return set_2d(&ctx->cond_tokens, t, n, c); }
int sam3d_debug_override_ss_latent(sam3d_ctx *ctx, const float *l, const int *dims)
    { return set_nd(&ctx->ss_latent, l, dims, 4); }
int sam3d_debug_override_occupancy(sam3d_ctx *ctx, const float *o, const int *dims)
    { return set_nd(&ctx->occupancy, o, dims, 3); }
int sam3d_debug_override_slat(sam3d_ctx *ctx, const float *f,
                              const int32_t *co, int n, int c)
{
    slat_free(&ctx->slat);
    ctx->slat.feats = (float *)malloc((size_t)n * c * sizeof(float));
    ctx->slat.coords = (int32_t *)malloc((size_t)n * 4 * sizeof(int32_t));
    if (!ctx->slat.feats || !ctx->slat.coords) return -1;
    memcpy(ctx->slat.feats, f, (size_t)n * c * sizeof(float));
    memcpy(ctx->slat.coords, co, (size_t)n * 4 * sizeof(int32_t));
    ctx->slat.n = n; ctx->slat.c = c;
    return 0;
}
