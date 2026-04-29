/* sam3d_cpu.c — CPU helpers used by the CUDA SAM 3D Objects runner.
 *
 * Phase 1a: provides a CPU-fallback DINOv2 forward pass so the CUDA
 * runner can validate downstream plumbing (D2H readback, debug
 * overrides, verify_dinov2 numerics) before per-stage NVRTC kernels
 * land. Subsequent phases replace the calls inside this TU one stage
 * at a time; the wrapper API surface stays stable so verify_*.c can
 * keep diffing against the same /tmp/sam3d_ref/ dumps.
 *
 * SAFETENSORS_IMPLEMENTATION is provided exclusively by
 * cuda_sam3d_runner.c. We pre-include the safetensors header here
 * (declarations only) so its include guard fires before dinov2.h's
 * SAFETENSORS_IMPLEMENTATION-prefixed include runs, matching the
 * sam3d_body_cpu.c pattern.
 */

#include "sam3d_cpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../common/safetensors.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define DINOV2_IMPLEMENTATION
#include "../../common/dinov2.h"

#define SAM3D_COND_FUSER_IMPLEMENTATION
#include "../../common/sam3d_cond_fuser.h"

#define SAM3D_SS_FLOW_DIT_IMPLEMENTATION
#include "../../common/sam3d_ss_flow_dit.h"

#include "../../common/sam3d_shortcut_solver.h"

#define T2_SS_DEC_IMPLEMENTATION
#include "../../common/trellis2_ss_decoder.h"

#define SPARSE3D_IMPLEMENTATION
#include "../../common/sparse3d.h"

#define SAM3D_SLAT_DIT_IMPLEMENTATION
#include "../../common/sam3d_slat_dit.h"

#define SAM3D_GS_DECODER_IMPLEMENTATION
#include "../../common/sam3d_gs_decoder.h"

struct sam3d_cpu_dinov2 {
    dinov2_model *m;
};

sam3d_cpu_dinov2 *sam3d_cpu_dinov2_load(const char *path)
{
    if (!path) return NULL;
    dinov2_model *m = dinov2_load_safetensors(path);
    if (!m) return NULL;
    sam3d_cpu_dinov2 *w = (sam3d_cpu_dinov2 *)calloc(1, sizeof(*w));
    if (!w) { dinov2_free(m); return NULL; }
    w->m = m;
    return w;
}

void sam3d_cpu_dinov2_free(sam3d_cpu_dinov2 *w)
{
    if (!w) return;
    if (w->m) dinov2_free(w->m);
    free(w);
}

int sam3d_cpu_dinov2_image_size(const sam3d_cpu_dinov2 *w)
{
    return w && w->m ? w->m->image_size : 0;
}

int sam3d_cpu_dinov2_dim(const sam3d_cpu_dinov2 *w)
{
    return w && w->m ? w->m->dim : 0;
}

int sam3d_cpu_dinov2_n_register(const sam3d_cpu_dinov2 *w)
{
    return w && w->m ? w->m->n_register : 0;
}

/* Same fp32 RGB bilinear-resize + ImageNet-normalize used by
 * cpu/sam3d/sam3d_runner.c. Kept private here so the CPU-fallback
 * matches the CPU runner numerics byte-for-byte. */
static void sp_prep_rgb_f32(float *out_chw,
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

static void sp_prep_mask_f32(float *out_chw,
                             const uint8_t *mask, int mw, int mh,
                             int ow, int oh,
                             const float mean[3], const float std[3])
{
    int s = (mw > mh) ? mw : mh;
    int ox = (s - mw) / 2;
    int oy = (s - mh) / 2;
    float scale_y = (float)s / (float)oh;
    float scale_x = (float)s / (float)ow;
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
            for (int c = 0; c < 3; c++)
                out_chw[c * oh * ow + y * ow + x] = (v - mean[c]) / std[c];
        }
    }
}

float *sam3d_cpu_dinov2_preprocess_rgb(sam3d_cpu_dinov2 *w,
                                       const uint8_t *rgba, int iw, int ih)
{
    if (!w || !w->m || !rgba) return NULL;
    int S = w->m->image_size;
    float *chw = (float *)malloc((size_t)3 * S * S * sizeof(float));
    if (!chw) return NULL;
    sp_prep_rgb_f32(chw, rgba, iw, ih, S, S, w->m->image_mean, w->m->image_std);
    return chw;
}

float *sam3d_cpu_dinov2_preprocess_mask(sam3d_cpu_dinov2 *w,
                                        const uint8_t *mask, int mw, int mh)
{
    if (!w || !w->m || !mask) return NULL;
    int S = w->m->image_size;
    float *chw = (float *)malloc((size_t)3 * S * S * sizeof(float));
    if (!chw) return NULL;
    sp_prep_mask_f32(chw, mask, mw, mh, S, S, w->m->image_mean, w->m->image_std);
    return chw;
}

float *sam3d_cpu_dinov2_encode_rgba(sam3d_cpu_dinov2 *w,
                                    const uint8_t *rgba, int iw, int ih,
                                    const uint8_t *mask, int mw, int mh,
                                    int n_threads,
                                    int *n_tokens_out, int *dim_out)
{
    if (!w || !w->m || !rgba) return NULL;
    dinov2_model *m = w->m;
    int img_sz = m->image_size;
    int n_reg  = m->n_register;
    int dim    = m->dim;

    float *chw_img = (float *)malloc((size_t)3 * img_sz * img_sz * sizeof(float));
    if (!chw_img) return NULL;
    sp_prep_rgb_f32(chw_img, rgba, iw, ih, img_sz, img_sz,
                    m->image_mean, m->image_std);
    dinov2_result r_img = dinov2_encode_f32(m, chw_img, img_sz, img_sz, n_threads);
    free(chw_img);
    if (!r_img.features) return NULL;
    dinov2_result_drop_registers(&r_img, n_reg);

    dinov2_result r_msk = {0};
    if (mask) {
        float *chw_msk = (float *)malloc((size_t)3 * img_sz * img_sz * sizeof(float));
        if (!chw_msk) { dinov2_result_free(&r_img); return NULL; }
        sp_prep_mask_f32(chw_msk, mask, mw, mh, img_sz, img_sz,
                         m->image_mean, m->image_std);
        r_msk = dinov2_encode_f32(m, chw_msk, img_sz, img_sz, n_threads);
        free(chw_msk);
        if (!r_msk.features) { dinov2_result_free(&r_img); return NULL; }
        dinov2_result_drop_registers(&r_msk, n_reg);
    }

    int n_tok = r_img.n_tokens;
    int n_branches = r_msk.features ? 2 : 1;
    int n_total = n_branches * n_tok;
    float *out = (float *)malloc((size_t)n_total * dim * sizeof(float));
    if (!out) { dinov2_result_free(&r_img); dinov2_result_free(&r_msk); return NULL; }
    memcpy(out, r_img.features, (size_t)n_tok * dim * sizeof(float));
    if (r_msk.features) {
        memcpy(out + (size_t)n_tok * dim, r_msk.features,
               (size_t)n_tok * dim * sizeof(float));
    }
    dinov2_result_free(&r_img);
    dinov2_result_free(&r_msk);
    if (n_tokens_out) *n_tokens_out = n_total;
    if (dim_out)      *dim_out      = dim;
    return out;
}

/* ===== CondEmbedderFuser ===== */

struct sam3d_cpu_fuser {
    sam3d_ppe_model   *ppe;
    sam3d_fuser_model *fuser;
};

sam3d_cpu_fuser *sam3d_cpu_fuser_load(const char *safetensors_dir)
{
    if (!safetensors_dir) return NULL;
    char path[1200];
    sam3d_cpu_fuser *w = (sam3d_cpu_fuser *)calloc(1, sizeof(*w));
    if (!w) return NULL;

    snprintf(path, sizeof(path), "%s/sam3d_point_patch_embed.safetensors",
             safetensors_dir);
    w->ppe = sam3d_ppe_load_safetensors(path);
    if (!w->ppe) { sam3d_cpu_fuser_free(w); return NULL; }

    snprintf(path, sizeof(path), "%s/sam3d_cond_fuser.safetensors",
             safetensors_dir);
    w->fuser = sam3d_fuser_load_safetensors(path);
    if (!w->fuser) { sam3d_cpu_fuser_free(w); return NULL; }
    return w;
}

void sam3d_cpu_fuser_free(sam3d_cpu_fuser *w)
{
    if (!w) return;
    if (w->ppe)   sam3d_ppe_free(w->ppe);
    if (w->fuser) sam3d_fuser_free(w->fuser);
    free(w);
}

int sam3d_cpu_fuser_dim_out(const sam3d_cpu_fuser *w)
{
    return (w && w->fuser) ? w->fuser->embed_dim_out : 0;
}

/* Shared core: takes already-computed PPE tokens (or NULL) and runs
 * the per-modality projections + pos embeds + concat. */
static float *cs3d_fuser_assemble(sam3d_cpu_fuser *w,
                                  const float *dino_tokens,
                                  int dino_n, int dino_dim,
                                  const float *ppe_tokens,
                                  int n_ppe, int ppe_dim,
                                  int n_threads,
                                  int *n_tokens_out, int *dim_out)
{
    sam3d_fuser_model *fuser = w->fuser;
    int D_out  = fuser->embed_dim_out;
    int branch = dino_n / 2;
    if (branch <= 0 || branch * 2 != dino_n) return NULL;
    if (ppe_tokens && ppe_dim != w->ppe->embed_dim) return NULL;
    const int pos = SAM3D_FUSER_POS_FULL;

    float *dino_img = sam3d_fuser_project(fuser, SAM3D_FUSER_MOD_DINO_IMG,
                                          dino_tokens, branch, n_threads);
    if (!dino_img) return NULL;
    sam3d_fuser_add_pos(fuser, pos, dino_img, branch);

    float *dino_msk = sam3d_fuser_project(fuser, SAM3D_FUSER_MOD_DINO_MSK,
                                          dino_tokens + (size_t)branch * dino_dim,
                                          branch, n_threads);
    if (!dino_msk) { free(dino_img); return NULL; }
    sam3d_fuser_add_pos(fuser, pos, dino_msk, branch);

    float *point = NULL;
    int point_n = 0;
    if (ppe_tokens && n_ppe > 0) {
        point = sam3d_fuser_project(fuser, SAM3D_FUSER_MOD_POINT,
                                    ppe_tokens, n_ppe, n_threads);
        if (!point) { free(dino_img); free(dino_msk); return NULL; }
        sam3d_fuser_add_pos(fuser, pos, point, n_ppe);
        point_n = n_ppe;
    }

    int n_total = branch * 2 + point_n;
    float *out = (float *)malloc((size_t)n_total * D_out * sizeof(float));
    if (!out) { free(dino_img); free(dino_msk); free(point); return NULL; }
    memcpy(out, dino_img, (size_t)branch * D_out * sizeof(float));
    memcpy(out + (size_t)branch * D_out, dino_msk,
           (size_t)branch * D_out * sizeof(float));
    if (point) {
        memcpy(out + (size_t)2 * branch * D_out, point,
               (size_t)point_n * D_out * sizeof(float));
    }
    free(dino_img); free(dino_msk); free(point);

    if (n_tokens_out) *n_tokens_out = n_total;
    if (dim_out)      *dim_out      = D_out;
    return out;
}

float *sam3d_cpu_fuser_run(sam3d_cpu_fuser *w,
                           const float *dino_tokens,
                           int dino_n, int dino_dim,
                           const float *pointmap_xyz, int ph, int pw,
                           int n_threads,
                           int *n_tokens_out, int *dim_out)
{
    if (!w || !w->fuser || !w->ppe || !dino_tokens || dino_n <= 0 || dino_dim <= 0)
        return NULL;
    if (n_threads < 1) n_threads = 1;

    float *ppe_tokens = NULL;
    int n_ppe = 0, ppe_dim = 0;
    if (pointmap_xyz && ph > 0 && pw > 0) {
        ppe_tokens = sam3d_ppe_encode(w->ppe, pointmap_xyz, ph, pw,
                                      NULL, n_threads);
        if (!ppe_tokens) return NULL;
        n_ppe   = w->ppe->num_patches * w->ppe->num_patches;
        ppe_dim = w->ppe->embed_dim;
    }
    float *out = cs3d_fuser_assemble(w, dino_tokens, dino_n, dino_dim,
                                     ppe_tokens, n_ppe, ppe_dim,
                                     n_threads, n_tokens_out, dim_out);
    free(ppe_tokens);
    return out;
}

float *sam3d_cpu_fuser_run_with_ppe_tokens(sam3d_cpu_fuser *w,
                           const float *dino_tokens,
                           int dino_n, int dino_dim,
                           const float *ppe_tokens,
                           int n_ppe, int ppe_dim,
                           int n_threads,
                           int *n_tokens_out, int *dim_out)
{
    if (!w || !w->fuser || !w->ppe || !dino_tokens || dino_n <= 0 || dino_dim <= 0)
        return NULL;
    if (n_threads < 1) n_threads = 1;
    return cs3d_fuser_assemble(w, dino_tokens, dino_n, dino_dim,
                               ppe_tokens, n_ppe, ppe_dim,
                               n_threads, n_tokens_out, dim_out);
}

int sam3d_cpu_fuser_ppe_num_patches(const sam3d_cpu_fuser *w) {
    return (w && w->ppe) ? w->ppe->num_patches : 0;
}
int sam3d_cpu_fuser_ppe_input_size(const sam3d_cpu_fuser *w) {
    return (w && w->ppe) ? w->ppe->input_size : 0;
}
int sam3d_cpu_fuser_ppe_embed_dim(const sam3d_cpu_fuser *w) {
    return (w && w->ppe) ? w->ppe->embed_dim : 0;
}
struct sam3d_ppe_model *sam3d_cpu_fuser_ppe_model(sam3d_cpu_fuser *w) {
    return w ? w->ppe : NULL;
}
struct sam3d_fuser_model *sam3d_cpu_fuser_fuser_model(sam3d_cpu_fuser *w) {
    return w ? w->fuser : NULL;
}

/* ===== SS Flow DiT ===== */

struct sam3d_cpu_ss_dit {
    sam3d_ss_flow_dit_model *m;
};

/* Five modality element counts per latent_mapping (yaml).
 * SHAPE [4096*8], 6DROT [1*6], TRANSLATION [1*3], SCALE [1*3], TRANSLATION_SCALE [1*1]. */
static const int sam3d_cpu_ss_dit_lat_elts_table[SAM3D_SS_DIT_N_LATENTS] = {
    4096 * 8, 1 * 6, 1 * 3, 1 * 3, 1 * 1
};

sam3d_cpu_ss_dit *sam3d_cpu_ss_dit_load(const char *safetensors_dir)
{
    if (!safetensors_dir) return NULL;
    char path[1200];
    snprintf(path, sizeof(path), "%s/sam3d_ss_dit.safetensors", safetensors_dir);
    sam3d_ss_flow_dit_model *m = sam3d_ss_flow_dit_load_safetensors(path);
    if (!m) return NULL;
    sam3d_cpu_ss_dit *w = (sam3d_cpu_ss_dit *)calloc(1, sizeof(*w));
    if (!w) { sam3d_ss_flow_dit_free(m); return NULL; }
    w->m = m;
    return w;
}

void sam3d_cpu_ss_dit_free(sam3d_cpu_ss_dit *w)
{
    if (!w) return;
    if (w->m) sam3d_ss_flow_dit_free(w->m);
    free(w);
}

int sam3d_cpu_ss_dit_n_blocks      (const sam3d_cpu_ss_dit *w) { return (w && w->m) ? w->m->n_blocks      : 0; }
int sam3d_cpu_ss_dit_dim           (const sam3d_cpu_ss_dit *w) { return (w && w->m) ? w->m->dim           : 0; }
int sam3d_cpu_ss_dit_cond_channels (const sam3d_cpu_ss_dit *w) { return (w && w->m) ? w->m->cond_channels : 0; }
int sam3d_cpu_ss_dit_is_shortcut   (const sam3d_cpu_ss_dit *w) { return (w && w->m) ? w->m->is_shortcut   : 0; }
int sam3d_cpu_ss_dit_n_latents     (void)                      { return SAM3D_SS_DIT_N_LATENTS; }
int sam3d_cpu_ss_dit_lat_elts(int i)
{
    if (i < 0 || i >= SAM3D_SS_DIT_N_LATENTS) return 0;
    return sam3d_cpu_ss_dit_lat_elts_table[i];
}

struct sam3d_ss_flow_dit_model *sam3d_cpu_ss_dit_model(sam3d_cpu_ss_dit *w)
{
    return w ? w->m : NULL;
}

int sam3d_cpu_ss_dit_forward(sam3d_cpu_ss_dit *w,
                             const float *const *latents_in,
                             float *const *latents_out,
                             const float *cond, int n_cond,
                             float t, float d, int n_threads)
{
    if (!w || !w->m || !latents_in || !latents_out || !cond) return -1;
    if (n_threads < 1) n_threads = 1;
    return sam3d_ss_flow_dit_forward(w->m, latents_in, latents_out,
                                     cond, n_cond, t, d, n_threads);
}

/* xorshift64* + Box-Muller; identical seed schedule to cpu/sam3d/sam3d_runner.c
 * so byte-for-byte numerics match the CPU runner. */
static inline uint64_t cs3d_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static inline float cs3d_rng_u01(uint64_t *state) {
    uint64_t r = cs3d_rng_next(state) >> 11;
    return (float)((double)r * (1.0 / 9007199254740992.0));
}
static void cs3d_fill_randn(float *buf, int n, uint64_t *state) {
    for (int i = 0; i < n; i += 2) {
        float u1 = cs3d_rng_u01(state); if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = cs3d_rng_u01(state);
        float r = sqrtf(-2.0f * logf(u1));
        float a = 6.2831853f * u2;
        buf[i] = r * cosf(a);
        if (i + 1 < n) buf[i + 1] = r * sinf(a);
    }
}

int sam3d_cpu_ss_dit_run_ode(sam3d_cpu_ss_dit *w,
                             const float *cond, int n_cond,
                             int steps, uint64_t seed, float cfg_scale,
                             int n_threads,
                             float *ss_latent_ncdhw)
{
    if (!w || !w->m || !cond || !ss_latent_ncdhw || steps <= 0) return -1;
    if (n_threads < 1) n_threads = 1;

    float *lat[SAM3D_SS_DIT_N_LATENTS] = {0};
    float *vel[SAM3D_SS_DIT_N_LATENTS] = {0};
    float *vel_u[SAM3D_SS_DIT_N_LATENTS] = {0};
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        int n = sam3d_cpu_ss_dit_lat_elts_table[i];
        lat[i] = (float *)malloc((size_t)n * sizeof(float));
        vel[i] = (float *)malloc((size_t)n * sizeof(float));
        vel_u[i] = (float *)malloc((size_t)n * sizeof(float));
        if (!lat[i] || !vel[i] || !vel_u[i]) goto oom;
    }

    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++)
        cs3d_fill_randn(lat[i], sam3d_cpu_ss_dit_lat_elts_table[i], &rng);

    float *times = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    if (!times) goto oom;
    sam3d_shortcut_make_times(times, steps, 3.0f, /*reversed=*/0);
    float d = sam3d_shortcut_d(steps, /*no_shortcut=*/1);
    /* Pre-scale t,d by time_scale=1000 to match upstream `_generate_dynamics`
     * (model trained on t in [0, 1000]; raw t in [0,1] gives garbage). */
    const float TIME_SCALE = w->m->time_scale;

    float *zero_cond = NULL;
    if (cfg_scale > 0.0f) {
        zero_cond = (float *)calloc((size_t)n_cond * w->m->cond_channels,
                                    sizeof(float));
        if (!zero_cond) { free(times); goto oom; }
    }

    for (int s = 0; s < steps; s++) {
        float t  = times[s];
        float dt = times[s + 1] - times[s];
        float ts = t * TIME_SCALE;
        if (sam3d_ss_flow_dit_forward(w->m,
                                      (const float *const *)lat, vel,
                                      cond, n_cond,
                                      ts, d * TIME_SCALE,
                                      n_threads) != 0) {
            free(times); free(zero_cond);
            goto oom;
        }
        if (zero_cond && sam3d_shortcut_cfg_active(ts, 0.0f, 500.0f)) {
            if (sam3d_ss_flow_dit_forward(w->m,
                                          (const float *const *)lat, vel_u,
                                          zero_cond, n_cond,
                                          ts, d * TIME_SCALE,
                                          n_threads) != 0) {
                free(times); free(zero_cond);
                goto oom;
            }
            for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
                sam3d_shortcut_cfg_combine(vel[i], vel[i], vel_u[i],
                                           cfg_scale,
                                           sam3d_cpu_ss_dit_lat_elts_table[i]);
            }
        }
        for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++)
            sam3d_shortcut_euler_step(lat[i], vel[i], dt,
                                      sam3d_cpu_ss_dit_lat_elts_table[i]);
    }
    free(times); free(zero_cond);

    /* SHAPE → NCDHW [8, 16, 16, 16]. Source layout from DiT is [N=4096, C=8]. */
    for (int n = 0; n < 4096; n++)
        for (int c = 0; c < 8; c++)
            ss_latent_ncdhw[c * 4096 + n] = lat[SAM3D_SS_LAT_SHAPE][n * 8 + c];

    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        free(lat[i]); free(vel[i]); free(vel_u[i]);
    }
    return 0;

oom:
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        free(lat[i]); free(vel[i]); free(vel_u[i]);
    }
    return -2;
}

/* ===== SS-VAE 3D-conv decoder ===== */

struct sam3d_cpu_ss_dec {
    t2_ss_dec *m;
};

sam3d_cpu_ss_dec *sam3d_cpu_ss_dec_load(const char *safetensors_dir)
{
    if (!safetensors_dir) return NULL;
    char path[1200];
    snprintf(path, sizeof(path), "%s/sam3d_ss_decoder.safetensors", safetensors_dir);
    t2_ss_dec *m = t2_ss_dec_load(path);
    if (!m) return NULL;
    sam3d_cpu_ss_dec *w = (sam3d_cpu_ss_dec *)calloc(1, sizeof(*w));
    if (!w) { t2_ss_dec_free(m); return NULL; }
    w->m = m;
    return w;
}

void sam3d_cpu_ss_dec_free(sam3d_cpu_ss_dec *w)
{
    if (!w) return;
    if (w->m) t2_ss_dec_free(w->m);
    free(w);
}

void *sam3d_cpu_ss_dec_model(sam3d_cpu_ss_dec *w)
{
    return w ? w->m : NULL;
}

float *sam3d_cpu_ss_dec_forward(sam3d_cpu_ss_dec *w,
                                const float *latent_ncdhw,
                                int n_threads)
{
    if (!w || !w->m || !latent_ncdhw) return NULL;
    if (n_threads < 1) n_threads = 1;
    return t2_ss_dec_forward(w->m, latent_ncdhw, n_threads);
}

/* ===== SLAT Flow DiT ===== */

struct sam3d_cpu_slat_dit {
    sam3d_slat_dit_model *m;
};

sam3d_cpu_slat_dit *sam3d_cpu_slat_dit_load(const char *safetensors_dir)
{
    if (!safetensors_dir) return NULL;
    char path[1200];
    snprintf(path, sizeof(path), "%s/sam3d_slat_dit.safetensors", safetensors_dir);
    sam3d_slat_dit_model *m = sam3d_slat_dit_load_safetensors(path);
    if (!m) return NULL;
    sam3d_cpu_slat_dit *w = (sam3d_cpu_slat_dit *)calloc(1, sizeof(*w));
    if (!w) { sam3d_slat_dit_free(m); return NULL; }
    w->m = m;
    return w;
}

void sam3d_cpu_slat_dit_free(sam3d_cpu_slat_dit *w)
{
    if (!w) return;
    if (w->m) sam3d_slat_dit_free(w->m);
    free(w);
}

int sam3d_cpu_slat_dit_in_channels  (const sam3d_cpu_slat_dit *w) { return (w && w->m) ? w->m->in_channels   : 0; }
int sam3d_cpu_slat_dit_out_channels (const sam3d_cpu_slat_dit *w) { return (w && w->m) ? w->m->out_channels  : 0; }
int sam3d_cpu_slat_dit_cond_channels(const sam3d_cpu_slat_dit *w) { return (w && w->m) ? w->m->cond_channels : 0; }
void *sam3d_cpu_slat_dit_model(sam3d_cpu_slat_dit *w) { return w ? w->m : NULL; }

void sam3d_cpu_slat_dit_set_transformer_hook(sam3d_cpu_slat_transformer_hook_fn fn,
                                             void *user)
{
    sam3d_slat_dit_set_transformer_hook((sam3d_slat_dit_transformer_hook_fn)fn, user);
}

void sam3d_cpu_slat_dit_set_ape_transformer_hook(sam3d_cpu_slat_ape_transformer_hook_fn fn,
                                                 void *user)
{
    sam3d_slat_dit_set_ape_transformer_hook((sam3d_slat_dit_ape_transformer_hook_fn)fn, user);
}

void sam3d_cpu_slat_dit_set_input_layer_hook(sam3d_cpu_slat_input_layer_hook_fn fn,
                                             void *user)
{
    sam3d_slat_dit_set_input_layer_hook((sam3d_slat_dit_input_layer_hook_fn)fn, user);
}

void sam3d_cpu_slat_dit_set_io_block_hook(sam3d_cpu_slat_io_block_hook_fn fn,
                                          void *user)
{
    sam3d_slat_dit_set_io_block_hook((sam3d_slat_dit_io_block_hook_fn)fn, user);
}

void sam3d_cpu_slat_dit_set_final_layer_hook(sam3d_cpu_slat_final_layer_hook_fn fn,
                                             void *user)
{
    sam3d_slat_dit_set_final_layer_hook((sam3d_slat_dit_final_layer_hook_fn)fn, user);
}

float *sam3d_cpu_slat_dit_forward(sam3d_cpu_slat_dit *w,
                                  const int32_t *coords,
                                  const float *feats, int N,
                                  float t,
                                  const float *cond, int n_cond,
                                  int n_threads)
{
    if (!w || !w->m || !coords || !feats || N <= 0 || !cond) return NULL;
    if (n_threads < 1) n_threads = 1;
    sp3d_tensor *x = sp3d_create(coords, feats, N, w->m->in_channels, 1);
    if (!x) return NULL;
    if (sam3d_slat_dit_forward(w->m, &x, t, cond, n_cond, n_threads) != 0) {
        sp3d_free(x); return NULL;
    }
    int oc = w->m->out_channels;
    size_t bytes = (size_t)x->N * oc * sizeof(float);
    float *out = (float *)malloc(bytes);
    if (!out) { sp3d_free(x); return NULL; }
    memcpy(out, x->feats, bytes);
    sp3d_free(x);
    return out;
}

/* SLAT un-normalization. Stats are 8-dim and fixed for facebook/sam-3d-objects. */
static const float CS3D_SLAT_MEAN[8] = {
     0.12211431f,  0.37204156f, -1.26521907f, -2.05276058f,
    -3.10432536f, -0.11294304f, -0.85146744f,  0.45506954f,
};
static const float CS3D_SLAT_STD[8] = {
     2.37326008f,  2.13174402f,  2.2413953f,   2.30589401f,
     2.1191894f,   1.8969511f,   2.41684989f,  2.08374642f,
};

int sam3d_cpu_slat_dit_run_ode(sam3d_cpu_slat_dit *w,
                               const float *occupancy,
                               int D, int H, int W,
                               const float *cond, int n_cond,
                               int steps, uint64_t seed, int n_threads,
                               int32_t **out_coords, float **out_feats,
                               int *out_n)
{
    if (!w || !w->m || !occupancy || !cond || !out_coords || !out_feats ||
        !out_n || D <= 0 || H <= 0 || W <= 0 || steps <= 0) return -1;

    int cap = 0;
    for (int i = 0, n = D * H * W; i < n; i++)
        if (occupancy[i] > 0.0f) cap++;
    if (cap == 0) return -2;

    int32_t *coords = (int32_t *)malloc((size_t)cap * 4 * sizeof(int32_t));
    if (!coords) return -3;

    int k = 0;
    for (int z = 0; z < D; z++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (occupancy[(z * H + y) * W + x] > 0.0f) {
                    coords[k * 4 + 0] = 0;
                    coords[k * 4 + 1] = z;
                    coords[k * 4 + 2] = y;
                    coords[k * 4 + 3] = x;
                    k++;
                }

    int rc = sam3d_cpu_slat_dit_run_ode_from_coords(w, coords, cap,
                                                    cond, n_cond,
                                                    steps, seed, n_threads,
                                                    out_coords, out_feats, out_n);
    free(coords);
    return rc;
}

int sam3d_cpu_slat_dit_run_ode_from_coords(sam3d_cpu_slat_dit *w,
                                           const int32_t *coords_in, int cap,
                                           const float *cond, int n_cond,
                                           int steps, uint64_t seed, int n_threads,
                                           int32_t **out_coords, float **out_feats,
                                           int *out_n)
{
    if (!w || !w->m || !coords_in || !cond || !out_coords || !out_feats ||
        !out_n || cap <= 0 || steps <= 0) return -1;
    if (n_threads < 1) n_threads = 1;
    sam3d_slat_dit_model *m = w->m;

    int32_t *coords = (int32_t *)malloc((size_t)cap * 4 * sizeof(int32_t));
    float   *feats  = (float *)calloc((size_t)cap * m->in_channels, sizeof(float));
    if (!coords || !feats) { free(coords); free(feats); return -3; }
    memcpy(coords, coords_in, (size_t)cap * 4 * sizeof(int32_t));

    uint64_t rng = seed ^ 0x5851F42D4C957F2DULL;
    cs3d_fill_randn(feats, cap * m->in_channels, &rng);

    sp3d_tensor *x = sp3d_create(coords, feats, cap, m->in_channels, 1);
    if (!x) { free(coords); free(feats); return -3; }

    /* time_scale=1000 from slat_generator.yaml — same contract as ss_dit. */
    const float TIME_SCALE = 1000.0f;
    for (int s = 0; s < steps; s++) {
        float t = 1.0f - (float)s / (float)steps;
        if (sam3d_slat_dit_forward(m, &x, t * TIME_SCALE,
                                   cond, n_cond, n_threads) != 0) {
            sp3d_free(x); free(coords); free(feats); return -4;
        }
    }

    int oc = m->out_channels;
    int32_t *cout = (int32_t *)malloc((size_t)cap * 4 * sizeof(int32_t));
    float   *fout = (float *)malloc((size_t)cap * oc * sizeof(float));
    if (!cout || !fout) {
        free(cout); free(fout);
        sp3d_free(x); free(coords); free(feats); return -3;
    }
    memcpy(cout, coords, (size_t)cap * 4 * sizeof(int32_t));
    memcpy(fout, x->feats, (size_t)cap * oc * sizeof(float));
    if (oc == 8) {
        for (int i = 0; i < cap; i++) {
            float *r = fout + (size_t)i * oc;
            for (int c = 0; c < oc; c++) r[c] = r[c] * CS3D_SLAT_STD[c] + CS3D_SLAT_MEAN[c];
        }
    }

    sp3d_free(x); free(coords); free(feats);
    *out_coords = cout;
    *out_feats  = fout;
    *out_n      = cap;
    return 0;
}

/* ===== SLAT GS decoder ===== */

struct sam3d_cpu_gs_decoder {
    sam3d_gs_decoder_model *m;
};

sam3d_cpu_gs_decoder *sam3d_cpu_gs_decoder_load(const char *safetensors_dir)
{
    if (!safetensors_dir) return NULL;
    char path[1200];
    snprintf(path, sizeof(path), "%s/sam3d_slat_gs_decoder.safetensors",
             safetensors_dir);
    sam3d_gs_decoder_model *m = sam3d_gs_decoder_load_safetensors(path);
    if (!m) return NULL;
    sam3d_cpu_gs_decoder *w = (sam3d_cpu_gs_decoder *)calloc(1, sizeof(*w));
    if (!w) { sam3d_gs_decoder_free(m); return NULL; }
    w->m = m;
    return w;
}

void sam3d_cpu_gs_decoder_free(sam3d_cpu_gs_decoder *w)
{
    if (!w) return;
    if (w->m) sam3d_gs_decoder_free(w->m);
    free(w);
}

int sam3d_cpu_gs_decoder_in_channels  (const sam3d_cpu_gs_decoder *w) { return (w && w->m) ? w->m->in_channels   : 0; }
int sam3d_cpu_gs_decoder_out_channels (const sam3d_cpu_gs_decoder *w) { return (w && w->m) ? w->m->out_channels  : 0; }
int sam3d_cpu_gs_decoder_num_gaussians(const sam3d_cpu_gs_decoder *w) { return (w && w->m) ? w->m->num_gaussians : 0; }
void *sam3d_cpu_gs_decoder_model(sam3d_cpu_gs_decoder *w) { return w ? w->m : NULL; }

void sam3d_cpu_gs_decoder_set_input_ape_hook(sam3d_cpu_gs_input_ape_hook_fn fn,
                                             void *user)
{
    sam3d_gs_decoder_set_input_ape_hook((sam3d_gs_input_ape_hook_fn)fn, user);
}

void sam3d_cpu_gs_decoder_set_final_layer_hook(sam3d_cpu_gs_final_layer_hook_fn fn,
                                               void *user)
{
    sam3d_gs_decoder_set_final_layer_hook((sam3d_gs_final_layer_hook_fn)fn, user);
}

void sam3d_cpu_gs_decoder_set_window_attn_hook(sam3d_cpu_gs_window_attn_hook_fn fn,
                                               void *user)
{
    sam3d_gs_decoder_set_window_attn_hook((sam3d_gs_window_attn_hook_fn)fn, user);
}

void sam3d_cpu_gs_decoder_set_attn_block_hook(sam3d_cpu_gs_attn_block_hook_fn fn,
                                              void *user)
{
    sam3d_gs_decoder_set_attn_block_hook((sam3d_gs_attn_block_hook_fn)fn, user);
}

void sam3d_cpu_gs_decoder_set_mlp_hook(sam3d_cpu_gs_mlp_hook_fn fn,
                                       void *user)
{
    sam3d_gs_decoder_set_mlp_hook((sam3d_gs_mlp_hook_fn)fn, user);
}

void sam3d_cpu_gs_decoder_set_block_hook(sam3d_cpu_gs_block_hook_fn fn,
                                         void *user)
{
    sam3d_gs_decoder_set_block_hook((sam3d_gs_block_hook_fn)fn, user);
}

void sam3d_cpu_gs_decoder_set_stack_hook(sam3d_cpu_gs_stack_hook_fn fn,
                                         void *user)
{
    sam3d_gs_decoder_set_stack_hook((sam3d_gs_stack_hook_fn)fn, user);
}

void sam3d_cpu_gs_decoder_set_transformer_hook(sam3d_cpu_gs_transformer_hook_fn fn,
                                               void *user)
{
    sam3d_gs_decoder_set_transformer_hook((sam3d_gs_transformer_hook_fn)fn, user);
}

float *sam3d_cpu_gs_decoder_transformer(sam3d_cpu_gs_decoder *w,
                                        const int32_t *coords,
                                        const float *feats, int N,
                                        int n_threads)
{
    if (!w || !w->m || !coords || !feats || N <= 0) return NULL;
    if (n_threads < 1) n_threads = 1;
    sp3d_tensor *x = sp3d_create(coords, feats, N, w->m->in_channels, 1);
    if (!x) return NULL;
    float *out = NULL;
    if (sam3d_gs_decoder_transformer(w->m, x, &out, n_threads) != 0) {
        sp3d_free(x); free(out); return NULL;
    }
    sp3d_free(x);
    return out;
}

int sam3d_cpu_gs_decoder_to_representation(sam3d_cpu_gs_decoder *w,
                                           const int32_t *coords,
                                           const float *feats_out, int N,
                                           float *xyz_out, float *dc_out,
                                           float *scaling_out, float *rotation_out,
                                           float *opacity_out)
{
    if (!w || !w->m || !coords || !feats_out || N <= 0) return -1;
    return sam3d_gs_decoder_to_representation(w->m, coords, feats_out, N,
                                              xyz_out, dc_out, scaling_out,
                                              rotation_out, opacity_out);
}

/* log(softplus(x)) — stable across full x range. Mirrors
 * cpu/sam3d/sam3d_runner.c so PLY-layout numerics are byte-identical. */
static float cs3d_log_softplus(float x) {
    if (x >  20.0f) return logf(x);
    if (x < -15.0f) return x;
    return logf(log1pf(expf(x)));
}

int sam3d_cpu_gs_decoder_pack_ply(const sam3d_cpu_gs_decoder *w,
                                  const float *xyz, const float *dc,
                                  const float *scl, const float *rot,
                                  const float *op,
                                  int total, int stride, float *out_ply)
{
    if (!w || !w->m || !xyz || !dc || !scl || !rot || !op || !out_ply ||
        total <= 0 || stride < 17) return -1;
    const float inv_sp_sb = logf(expf(w->m->scaling_bias) - 1.0f);
#if defined(_OPENMP)
    #pragma omp parallel for
#endif
    for (int i = 0; i < total; i++) {
        float *r = out_ply + (size_t)i * stride;
        r[0] = xyz[i * 3 + 0];
        r[1] = xyz[i * 3 + 1];
        r[2] = xyz[i * 3 + 2];
        r[3] = r[4] = r[5] = 0.0f;        /* normals */
        r[6] = dc[i * 3 + 0];
        r[7] = dc[i * 3 + 1];
        r[8] = dc[i * 3 + 2];
        r[9] = op[i] + w->m->opacity_bias;
        for (int a = 0; a < 3; a++)
            r[10 + a] = cs3d_log_softplus(scl[i * 3 + a] + inv_sp_sb);
        r[13] = rot[i * 4 + 0];
        r[14] = rot[i * 4 + 1];
        r[15] = rot[i * 4 + 2];
        r[16] = rot[i * 4 + 3];
    }
    return 0;
}
