/* CUDA runner for FAIR's SAM 3D Objects — Phase 0 scaffold.
 *
 * What's wired right now:
 *   - cuew runtime + NVRTC compile bring-up (sentinel kernel).
 *   - Per-module safetensors directory resolution (explicit override or
 *     pipeline_yaml sibling fallback).
 *   - Public API stubs: every cuda_sam3d_run_* and cuda_sam3d_get_* is
 *     callable; each returns CUDA_SAM3D_E_NOT_IMPLEMENTED until its
 *     stage lands.
 *   - Debug overrides hold pytorch ref tensors on host; later stages
 *     consume them once the corresponding kernels go live.
 *
 * Subsequent phases bring in real kernels in the order:
 *   1. DINOv2-L/14+reg encoder      (mirror cuda/sam3d_body's DINOv3 path)
 *   2. CondEmbedderFuser            (Llama SwiGLU + concat)
 *   3. SS Flow DiT (shortcut ODE)   (mirror cuda/trellis2's DiT)
 *   4. SS-VAE 3D-conv decoder
 *   5. SLAT Flow DiT (shift-window)
 *   6. SLAT GS decoder + PLY emit
 *
 * Reference numerics live in /tmp/sam3d_ref/ (from
 * ref/sam3d/gen_image_ref.py); each stage gets a verify_*.c that diffs
 * against the same dumps the CPU runner uses.
 */

#include "cuda_sam3d_runner.h"
#include "../cuew.h"
#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"
#include "../cuda_hip_compat.h"
#include "cuda_sam3d_kernels.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define CUDA_SAM3D_DINOV2_GPU_IMPLEMENTATION
#include "cuda_sam3d_dinov2_gpu.h"
#include "cuda_sam3d_dinov2_forward.h"
#define CUDA_SAM3D_PPE_GPU_IMPLEMENTATION
#include "cuda_sam3d_ppe_gpu.h"
#include "cuda_sam3d_ppe_forward.h"
#define CUDA_SAM3D_FUSER_GPU_IMPLEMENTATION
#include "cuda_sam3d_fuser_gpu.h"
#include "cuda_sam3d_fuser_forward.h"
#define CUDA_SAM3D_SSDIT_GPU_IMPLEMENTATION
#include "cuda_sam3d_ssdit_gpu.h"
#undef  CUDA_SAM3D_SSDIT_GPU_IMPLEMENTATION
#define CUDA_SAM3D_SSDIT_FORWARD_IMPLEMENTATION
#include "cuda_sam3d_ssdit_forward.h"
#undef  CUDA_SAM3D_SSDIT_FORWARD_IMPLEMENTATION
#define CUDA_SAM3D_SSDIT_OUTER_IMPLEMENTATION
#include "cuda_sam3d_ssdit_outer.h"
#undef  CUDA_SAM3D_SSDIT_OUTER_IMPLEMENTATION
#define CUDA_SAM3D_SS_DECODER_GPU_IMPLEMENTATION
#include "cuda_sam3d_ss_decoder_gpu.h"
#undef  CUDA_SAM3D_SS_DECODER_GPU_IMPLEMENTATION
#define CUDA_SAM3D_SS_DECODER_FORWARD_IMPLEMENTATION
#include "cuda_sam3d_ss_decoder_forward.h"
#undef  CUDA_SAM3D_SS_DECODER_FORWARD_IMPLEMENTATION
#define CUDA_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION
#include "cuda_sam3d_slat_dit_gpu.h"
#undef  CUDA_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION
#define CUDA_SAM3D_SLAT_DIT_FORWARD_IMPLEMENTATION
#include "cuda_sam3d_slat_dit_forward.h"
#undef  CUDA_SAM3D_SLAT_DIT_FORWARD_IMPLEMENTATION
#include "../../common/sam3d_shortcut_solver.h"
#include "sam3d_cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ===== context state ===== */

typedef struct {
    int      n;
    int      c;
    float   *data;       /* host, malloc'd */
} cs3d_host_2d;

typedef struct {
    int      ndim;
    int      dims[4];
    float   *data;       /* host, malloc'd */
} cs3d_host_nd;

typedef struct {
    int      n, c;
    float   *feats;      /* host, malloc'd */
    int32_t *coords;     /* host, malloc'd */
} cs3d_host_slat;

struct cuda_sam3d_ctx {
    cuda_sam3d_config cfg;
    char    *safetensors_dir_resolved;

    /* cuew + NVRTC. */
    int          device_id;
    int          sm;
    hipModule_t  mod;
    hipFunction_t fn_sentinel;
    int          compiled;

    /* Inputs. */
    uint8_t *img_rgba;   int img_w, img_h;
    uint8_t *mask_u8;    int msk_w, msk_h;
    float   *pmap_f32;   int pmap_w, pmap_h;

    /* Debug-override host buffers (kept until consumed). */
    cs3d_host_2d   ovr_dinov2;
    cs3d_host_2d   ovr_cond;
    cs3d_host_nd   ovr_ss_latent;
    cs3d_host_nd   ovr_occupancy;
    cs3d_host_slat ovr_slat;

    /* Stage outputs (Phase 1a uses CPU fallback; host mirror kept so
     * verify_*.c can read back without a D2H round-trip until real
     * kernels land). */
    sam3d_cpu_dinov2 *cpu_dinov2;
    cs3d_host_2d      dinov2_tokens;
    CUdeviceptr       d_dinov2_tokens;   /* device mirror, [n_tok × dim] f32 */

    /* Phase 1b GPU DINOv2 forward — lazy-init on first run_dinov2 call. */
    cs3d_dinov2_gpu      gpu_dinov2;
    int                  gpu_dinov2_loaded;
    cs3d_dinov2_fns      gpu_dinov2_fns;
    cs3d_dinov2_block_ws gpu_dinov2_ws;
    int                  gpu_dinov2_ws_alloced;

    sam3d_cpu_fuser  *cpu_fuser;
    cs3d_host_2d      cond_tokens;
    CUdeviceptr       d_cond_tokens;     /* device mirror, [n_tok × D_out] f32 */

    /* Phase 2b.8d GPU PPE — lazy-init on first run_cond_fuser call. */
    cs3d_ppe_gpu      gpu_ppe;
    int               gpu_ppe_loaded;
    cs3d_ppe_fns      gpu_ppe_fns;
    cs3d_ppe_ws       gpu_ppe_ws;
    int               gpu_ppe_ws_alloced;

    /* Phase 2b.8e GPU fuser projection. */
    cs3d_fuser_gpu    gpu_fuser;
    int               gpu_fuser_loaded;
    cs3d_fuser_fns    gpu_fuser_fns;
    cs3d_fuser_ws     gpu_fuser_ws;
    int               gpu_fuser_ws_alloced;

    sam3d_cpu_ss_dit *cpu_ss_dit;
    cs3d_host_nd      ss_latent;         /* [8,16,16,16] f32 NCDHW */
    CUdeviceptr       d_ss_latent;       /* device mirror */

    /* Phase 2c.13 GPU SS Flow DiT — lazy-init on first ss_dit call. */
    cs3d_ssdit_gpu        gpu_ssdit;
    int                   gpu_ssdit_loaded;
    cs3d_ssdit_outer_fns  gpu_ssdit_fns;
    cs3d_ssdit_block_ws   gpu_ssdit_block_ws;
    cs3d_ssdit_outer_ws   gpu_ssdit_outer_ws;
    int                   gpu_ssdit_ws_alloced;
    int                   gpu_ssdit_ws_n_c;

    sam3d_cpu_ss_dec *cpu_ss_dec;
    cs3d_host_nd      occupancy;         /* [64,64,64] f32 logits */
    CUdeviceptr       d_occupancy;       /* device mirror */

    /* Phase 4b GPU SS-VAE decoder. */
    cs3d_ssdec_gpu    gpu_ssdec;
    int               gpu_ssdec_loaded;
    cs3d_ssdec_fns    gpu_ssdec_fns;
    cs3d_ssdec_ws     gpu_ssdec_ws;
    int               gpu_ssdec_ws_alloced;

    sam3d_cpu_slat_dit *cpu_slat_dit;
    cs3d_host_slat      slat_tokens;     /* (coords[N,4], feats[N,out_ch]) f32 */
    CUdeviceptr         d_slat_feats;    /* device mirror */
    CUdeviceptr         d_slat_coords;   /* device mirror */
    cs3d_slatdit_gpu       gpu_slatdit;
    int                    gpu_slatdit_loaded;
    cs3d_slatdit_fns       gpu_slatdit_fns;
    cs3d_slatdit_block_ws  gpu_slatdit_ws;
    int                    gpu_slatdit_ws_alloced;
    int                    gpu_slatdit_ws_n;
    int                    gpu_slatdit_ws_nc;
    CUdeviceptr            d_slat_hook_x;
    CUdeviceptr            d_slat_hook_t_emb;
    size_t                 slat_hook_x_bytes;
    size_t                 slat_hook_t_emb_bytes;

    sam3d_cpu_gs_decoder *cpu_gs;
    cs3d_host_2d          gaussians;      /* [N*G, 17] PLY-layout f32 */
    CUdeviceptr           d_gaussians;    /* device mirror */
};

/* ===== tiny helpers ===== */

static char *cs3d_strdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *r = (char *)malloc(n);
    if (r) memcpy(r, s, n);
    return r;
}

/* If cfg->safetensors_dir is non-NULL, return a strdup. Otherwise derive
 * from pipeline_yaml's parent directory by appending /../safetensors. */
static char *cs3d_resolve_safetensors_dir(const cuda_sam3d_config *cfg) {
    if (cfg->safetensors_dir && cfg->safetensors_dir[0])
        return cs3d_strdup(cfg->safetensors_dir);
    if (!cfg->pipeline_yaml || !cfg->pipeline_yaml[0]) return NULL;
    /* dirname(pipeline_yaml) + "/../safetensors" */
    const char *p = cfg->pipeline_yaml;
    const char *slash = strrchr(p, '/');
    size_t dir_len = slash ? (size_t)(slash - p) : 0;
    const char *suffix = "/../safetensors";
    size_t suf_len = strlen(suffix);
    char *r = (char *)malloc(dir_len + suf_len + 1);
    if (!r) return NULL;
    if (dir_len) memcpy(r, p, dir_len);
    memcpy(r + dir_len, suffix, suf_len + 1);
    return r;
}

static void cs3d_free_2d(cs3d_host_2d *t)   { free(t->data); t->data = NULL; t->n = t->c = 0; }
static void cs3d_free_nd(cs3d_host_nd *t)   { free(t->data); t->data = NULL; t->ndim = 0; }
static void cs3d_free_slat(cs3d_host_slat *t) {
    free(t->feats); free(t->coords);
    t->feats = NULL; t->coords = NULL; t->n = t->c = 0;
}

/* ===== public API ===== */

cuda_sam3d_ctx *cuda_sam3d_create(const cuda_sam3d_config *cfg)
{
    if (!cfg) return NULL;

    cuda_sam3d_ctx *c = (cuda_sam3d_ctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cfg = *cfg;
    c->device_id = cfg->device_ordinal;
    if (c->cfg.ss_steps   <= 0) c->cfg.ss_steps   = 2;
    if (c->cfg.slat_steps <= 0) c->cfg.slat_steps = 12;
    if (c->cfg.cfg_scale  <= 0) c->cfg.cfg_scale  = 2.0f;

    c->safetensors_dir_resolved = cs3d_resolve_safetensors_dir(cfg);
    if (!c->safetensors_dir_resolved) {
        fprintf(stderr, "cuda_sam3d: need safetensors_dir or pipeline_yaml\n");
        free(c); return NULL;
    }

    if (cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC) != CUEW_SUCCESS) {
        fprintf(stderr, "cuda_sam3d: cuew init failed (no CUDA driver/runtime?)\n");
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    if (cuInit(0) != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_sam3d: cuInit failed\n");
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    CUdevice dev;
    if (cuDeviceGet(&dev, c->device_id) != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_sam3d: cuDeviceGet(%d) failed\n", c->device_id);
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    CUcontext cu_ctx;
    if (cuCtxCreate(&cu_ctx, 0, dev) != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_sam3d: cuCtxCreate failed\n");
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }

    c->sm = cu_compile_kernels(&c->mod, dev, cuda_sam3d_kernel_src,
                               "sam3d_kernels.cu", c->cfg.verbose,
                               "cuda_sam3d");
    if (c->sm < 0) {
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    if (cuModuleGetFunction(&c->fn_sentinel, c->mod, "sam3d_sentinel") != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_sam3d: missing sam3d_sentinel\n");
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    c->compiled = 1;

    if (c->cfg.verbose >= 1) {
        fprintf(stderr, "cuda_sam3d: created (sm_%d, safetensors_dir=%s)\n",
                c->sm, c->safetensors_dir_resolved);
    }
    return c;
}

void cuda_sam3d_destroy(cuda_sam3d_ctx *ctx)
{
    if (!ctx) return;
    free(ctx->img_rgba); free(ctx->mask_u8); free(ctx->pmap_f32);
    cs3d_free_2d(&ctx->ovr_dinov2);
    cs3d_free_2d(&ctx->ovr_cond);
    cs3d_free_nd(&ctx->ovr_ss_latent);
    cs3d_free_nd(&ctx->ovr_occupancy);
    cs3d_free_slat(&ctx->ovr_slat);
    cs3d_free_2d(&ctx->dinov2_tokens);
    if (ctx->d_dinov2_tokens) cuMemFree(ctx->d_dinov2_tokens);
    if (ctx->cpu_dinov2) sam3d_cpu_dinov2_free(ctx->cpu_dinov2);
    if (ctx->gpu_dinov2_ws_alloced) cs3d_dinov2_block_ws_free(&ctx->gpu_dinov2_ws);
    if (ctx->gpu_dinov2_loaded) cs3d_dinov2_gpu_free(&ctx->gpu_dinov2);
    cs3d_free_2d(&ctx->cond_tokens);
    if (ctx->d_cond_tokens) cuMemFree(ctx->d_cond_tokens);
    if (ctx->gpu_ppe_ws_alloced)   cs3d_ppe_ws_free(&ctx->gpu_ppe_ws);
    if (ctx->gpu_ppe_loaded)       cs3d_ppe_gpu_free(&ctx->gpu_ppe);
    if (ctx->gpu_fuser_ws_alloced) cs3d_fuser_ws_free(&ctx->gpu_fuser_ws);
    if (ctx->gpu_fuser_loaded)     cs3d_fuser_gpu_free(&ctx->gpu_fuser);
    if (ctx->cpu_fuser) sam3d_cpu_fuser_free(ctx->cpu_fuser);
    cs3d_free_nd(&ctx->ss_latent);
    if (ctx->d_ss_latent) cuMemFree(ctx->d_ss_latent);
    if (ctx->gpu_ssdit_ws_alloced) {
        cs3d_ssdit_outer_ws_free(&ctx->gpu_ssdit_outer_ws);
        cs3d_ssdit_block_ws_free(&ctx->gpu_ssdit_block_ws);
    }
    if (ctx->gpu_ssdit_loaded) cs3d_ssdit_gpu_free(&ctx->gpu_ssdit);
    if (ctx->cpu_ss_dit) sam3d_cpu_ss_dit_free(ctx->cpu_ss_dit);
    cs3d_free_nd(&ctx->occupancy);
    if (ctx->d_occupancy) cuMemFree(ctx->d_occupancy);
    if (ctx->gpu_ssdec_ws_alloced) cs3d_ssdec_ws_free(&ctx->gpu_ssdec_ws);
    if (ctx->gpu_ssdec_loaded)     cs3d_ssdec_gpu_free(&ctx->gpu_ssdec);
    if (ctx->cpu_ss_dec) sam3d_cpu_ss_dec_free(ctx->cpu_ss_dec);
    cs3d_free_slat(&ctx->slat_tokens);
    if (ctx->d_slat_feats)  cuMemFree(ctx->d_slat_feats);
    if (ctx->d_slat_coords) cuMemFree(ctx->d_slat_coords);
    if (ctx->d_slat_hook_x)     cuMemFree(ctx->d_slat_hook_x);
    if (ctx->d_slat_hook_t_emb) cuMemFree(ctx->d_slat_hook_t_emb);
    if (ctx->gpu_slatdit_ws_alloced) cs3d_slatdit_block_ws_free(&ctx->gpu_slatdit_ws);
    if (ctx->gpu_slatdit_loaded)     cs3d_slatdit_gpu_free(&ctx->gpu_slatdit);
    if (ctx->cpu_slat_dit) sam3d_cpu_slat_dit_free(ctx->cpu_slat_dit);
    cs3d_free_2d(&ctx->gaussians);
    if (ctx->d_gaussians) cuMemFree(ctx->d_gaussians);
    if (ctx->cpu_gs) sam3d_cpu_gs_decoder_free(ctx->cpu_gs);
    if (ctx->compiled) cuModuleUnload(ctx->mod);
    free(ctx->safetensors_dir_resolved);
    free(ctx);
}

/* ===== inputs ===== */

int cuda_sam3d_set_image_rgba(cuda_sam3d_ctx *ctx, const uint8_t *rgba,
                              int width, int height) {
    if (!ctx || !rgba || width <= 0 || height <= 0) return CUDA_SAM3D_E_INVAL;
    free(ctx->img_rgba);
    size_t n = (size_t)width * height * 4;
    ctx->img_rgba = (uint8_t *)malloc(n);
    if (!ctx->img_rgba) return CUDA_SAM3D_E_LOAD;
    memcpy(ctx->img_rgba, rgba, n);
    ctx->img_w = width; ctx->img_h = height;
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_set_mask(cuda_sam3d_ctx *ctx, const uint8_t *mask,
                        int width, int height) {
    if (!ctx || !mask || width <= 0 || height <= 0) return CUDA_SAM3D_E_INVAL;
    free(ctx->mask_u8);
    size_t n = (size_t)width * height;
    ctx->mask_u8 = (uint8_t *)malloc(n);
    if (!ctx->mask_u8) return CUDA_SAM3D_E_LOAD;
    memcpy(ctx->mask_u8, mask, n);
    ctx->msk_w = width; ctx->msk_h = height;
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_set_pointmap(cuda_sam3d_ctx *ctx, const float *pmap,
                            int width, int height) {
    if (!ctx || !pmap || width <= 0 || height <= 0) return CUDA_SAM3D_E_INVAL;
    free(ctx->pmap_f32);
    size_t n = (size_t)width * height * 3;
    ctx->pmap_f32 = (float *)malloc(n * sizeof(float));
    if (!ctx->pmap_f32) return CUDA_SAM3D_E_LOAD;
    memcpy(ctx->pmap_f32, pmap, n * sizeof(float));
    ctx->pmap_w = width; ctx->pmap_h = height;
    return CUDA_SAM3D_E_OK;
}

/* ===== stage stubs ===== */

/* Adopt a freshly-computed [n × c] f32 token block as the dinov2
 * stage output. Frees any prior host/device buffers. Takes ownership
 * of `feats` (must be malloc'd). */
static int cs3d_adopt_dinov2_tokens(cuda_sam3d_ctx *ctx, float *feats,
                                    int n, int c)
{
    cs3d_free_2d(&ctx->dinov2_tokens);
    if (ctx->d_dinov2_tokens) {
        cuMemFree(ctx->d_dinov2_tokens);
        ctx->d_dinov2_tokens = 0;
    }
    if (!feats || n <= 0 || c <= 0) return CUDA_SAM3D_E_INVAL;

    size_t bytes = (size_t)n * c * sizeof(float);
    ctx->d_dinov2_tokens = cu_upload_raw(feats, bytes);
    if (!ctx->d_dinov2_tokens) { free(feats); return CUDA_SAM3D_E_LOAD; }
    ctx->dinov2_tokens.data = feats;
    ctx->dinov2_tokens.n    = n;
    ctx->dinov2_tokens.c    = c;
    return CUDA_SAM3D_E_OK;
}

/* Phase 1b.8: run one branch (image or mask) through the GPU forward.
 * chw_norm is [3, S, S] f32, normalized. Returns malloc'd [n_tok × dim]
 * with the n_register tokens removed (matching CPU numerics). */
static float *cs3d_gpu_dinov2_encode_branch(cuda_sam3d_ctx *ctx,
                                            const float *chw_norm)
{
    cs3d_dinov2_gpu *g = &ctx->gpu_dinov2;
    size_t n_img = (size_t)3 * g->image_size * g->image_size;
    size_t n_out = (size_t)g->n_tokens * g->dim;

    CUdeviceptr d_img = cu_upload_raw(chw_norm, n_img * sizeof(float));
    if (!d_img) return NULL;
    CUdeviceptr d_out = 0;
    if (cuMemAlloc(&d_out, n_out * sizeof(float)) != CUDA_SUCCESS) {
        cuMemFree(d_img); return NULL;
    }
    int rc = cs3d_dinov2_gpu_forward(&ctx->gpu_dinov2_fns, &ctx->gpu_dinov2_ws,
                                     g, d_img, d_out);
    cuMemFree(d_img);
    if (rc < 0) { cuMemFree(d_out); return NULL; }

    float *host = (float *)malloc(n_out * sizeof(float));
    if (!host) { cuMemFree(d_out); return NULL; }
    if (cuMemcpyDtoH(host, d_out, n_out * sizeof(float)) != CUDA_SUCCESS) {
        free(host); cuMemFree(d_out); return NULL;
    }
    cuMemFree(d_out);

    /* Drop register tokens in-place: keep [CLS] then patch tokens, skip
     * the n_register tokens between them. */
    int dim       = g->dim;
    int n_reg     = g->n_register;
    int keep_pat  = g->n_tokens - 1 - n_reg;
    if (n_reg > 0 && keep_pat > 0) {
        memmove(host + (size_t)dim,
                host + (size_t)(1 + n_reg) * dim,
                (size_t)keep_pat * dim * sizeof(float));
    }
    return host;
}

int cuda_sam3d_run_dinov2(cuda_sam3d_ctx *ctx) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    if (!ctx->img_rgba) return CUDA_SAM3D_E_NO_INPUT;

    /* CPU model kept for preprocessing (image_mean/std + bilinear-resize
     * + ImageNet-normalize); the encode itself runs on the GPU. */
    if (!ctx->cpu_dinov2) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_dinov2.safetensors",
                 ctx->safetensors_dir_resolved);
        ctx->cpu_dinov2 = sam3d_cpu_dinov2_load(path);
        if (!ctx->cpu_dinov2) {
            fprintf(stderr, "cuda_sam3d: dinov2 load failed (%s)\n", path);
            return CUDA_SAM3D_E_LOAD;
        }
    }
    if (!ctx->gpu_dinov2_loaded) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_dinov2.safetensors",
                 ctx->safetensors_dir_resolved);
        if (cs3d_dinov2_gpu_load(&ctx->gpu_dinov2, path, ctx->cfg.verbose) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu dinov2 weight upload failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        ctx->gpu_dinov2_loaded = 1;
        if (cs3d_dinov2_fns_lookup(&ctx->gpu_dinov2_fns, ctx->mod) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu dinov2 fns lookup failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        if (cs3d_dinov2_block_ws_alloc(&ctx->gpu_dinov2_ws,
                                       ctx->gpu_dinov2.n_tokens,
                                       ctx->gpu_dinov2.dim,
                                       ctx->gpu_dinov2.ffn_hidden) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu dinov2 ws alloc failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        ctx->gpu_dinov2_ws_alloced = 1;
    }

    int dim   = ctx->gpu_dinov2.dim;
    int n_reg = ctx->gpu_dinov2.n_register;
    int n_per = ctx->gpu_dinov2.n_tokens - n_reg;  /* CLS + patches */

    float *chw_img = sam3d_cpu_dinov2_preprocess_rgb(
        ctx->cpu_dinov2, ctx->img_rgba, ctx->img_w, ctx->img_h);
    if (!chw_img) return CUDA_SAM3D_E_LOAD;
    float *feat_img = cs3d_gpu_dinov2_encode_branch(ctx, chw_img);
    free(chw_img);
    if (!feat_img) {
        fprintf(stderr, "cuda_sam3d: gpu dinov2 image branch failed\n");
        return CUDA_SAM3D_E_LOAD;
    }

    float *feat_msk = NULL;
    if (ctx->mask_u8) {
        float *chw_msk = sam3d_cpu_dinov2_preprocess_mask(
            ctx->cpu_dinov2, ctx->mask_u8, ctx->msk_w, ctx->msk_h);
        if (!chw_msk) { free(feat_img); return CUDA_SAM3D_E_LOAD; }
        feat_msk = cs3d_gpu_dinov2_encode_branch(ctx, chw_msk);
        free(chw_msk);
        if (!feat_msk) {
            fprintf(stderr, "cuda_sam3d: gpu dinov2 mask branch failed\n");
            free(feat_img);
            return CUDA_SAM3D_E_LOAD;
        }
    }

    int n_branches = feat_msk ? 2 : 1;
    int n_total    = n_branches * n_per;
    float *feats   = (float *)malloc((size_t)n_total * dim * sizeof(float));
    if (!feats) { free(feat_img); free(feat_msk); return CUDA_SAM3D_E_LOAD; }
    memcpy(feats, feat_img, (size_t)n_per * dim * sizeof(float));
    if (feat_msk) {
        memcpy(feats + (size_t)n_per * dim, feat_msk,
               (size_t)n_per * dim * sizeof(float));
    }
    free(feat_img); free(feat_msk);
    return cs3d_adopt_dinov2_tokens(ctx, feats, n_total, dim);
}
/* Active dinov2 source: prefer override (verify_*.c injects ref data
 * via cuda_sam3d_debug_override_dinov2) so a single verify can isolate
 * cond_fuser drift from upstream dinov2 drift. */
static const cs3d_host_2d *cs3d_active_dinov2(const cuda_sam3d_ctx *ctx)
{
    return ctx->ovr_dinov2.data ? &ctx->ovr_dinov2 : &ctx->dinov2_tokens;
}

static int cs3d_adopt_cond_tokens(cuda_sam3d_ctx *ctx, float *feats,
                                  int n, int c)
{
    cs3d_free_2d(&ctx->cond_tokens);
    if (ctx->d_cond_tokens) {
        cuMemFree(ctx->d_cond_tokens);
        ctx->d_cond_tokens = 0;
    }
    if (!feats || n <= 0 || c <= 0) return CUDA_SAM3D_E_INVAL;
    size_t bytes = (size_t)n * c * sizeof(float);
    ctx->d_cond_tokens = cu_upload_raw(feats, bytes);
    if (!ctx->d_cond_tokens) { free(feats); return CUDA_SAM3D_E_LOAD; }
    ctx->cond_tokens.data = feats;
    ctx->cond_tokens.n    = n;
    ctx->cond_tokens.c    = c;
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_run_cond_fuser(cuda_sam3d_ctx *ctx) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_2d *dino = cs3d_active_dinov2(ctx);
    if (!dino->data) {
        fprintf(stderr, "cuda_sam3d: cond_fuser needs dinov2 tokens — "
                        "run dinov2 (or override) first\n");
        return CUDA_SAM3D_E_NO_INPUT;
    }

    if (!ctx->cpu_fuser) {
        ctx->cpu_fuser = sam3d_cpu_fuser_load(ctx->safetensors_dir_resolved);
        if (!ctx->cpu_fuser) {
            fprintf(stderr, "cuda_sam3d: fuser/PPE load failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
    }

#if defined(_OPENMP)
    int nthr = omp_get_max_threads();
    if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif

    sam3d_ppe_model   *ppe_m = sam3d_cpu_fuser_ppe_model  (ctx->cpu_fuser);
    sam3d_fuser_model *fus_m = sam3d_cpu_fuser_fuser_model(ctx->cpu_fuser);
    int branch  = dino->n / 2;
    int Di_dino = dino->c;
    int have_ppe = (ctx->pmap_f32 && ppe_m && ctx->pmap_h > 0 && ctx->pmap_w > 0
                    && fus_m && fus_m->n_modalities >= 3);
    int S      = have_ppe ? ppe_m->input_size : 0;
    int Np     = have_ppe ? ppe_m->num_patches : 0;
    int Nwin   = have_ppe ? Np * Np : 0;
    int D_ppe  = have_ppe ? ppe_m->embed_dim : 0;
    int n_total = branch * 2 + Nwin;
    int Do = sam3d_cpu_fuser_dim_out(ctx->cpu_fuser);

    if (branch <= 0 || branch * 2 != dino->n || Do <= 0) {
        fprintf(stderr, "cuda_sam3d: cond_fuser bad shape (branch=%d dim=%d Do=%d)\n",
                branch, Di_dino, Do);
        return CUDA_SAM3D_E_INVAL;
    }

    /* Lazy-load PPE GPU. */
    if (have_ppe && !ctx->gpu_ppe_loaded) {
        if (cs3d_ppe_gpu_load(&ctx->gpu_ppe, ppe_m, ctx->cfg.verbose) < 0)
            return CUDA_SAM3D_E_LOAD;
        ctx->gpu_ppe_loaded = 1;
        if (cs3d_ppe_fns_lookup(&ctx->gpu_ppe_fns, ctx->mod) < 0)
            return CUDA_SAM3D_E_LOAD;
        if (cs3d_ppe_ws_alloc(&ctx->gpu_ppe_ws,
                              ctx->gpu_ppe.Np, ctx->gpu_ppe.P,
                              ctx->gpu_ppe.D,  ctx->gpu_ppe.ffn) < 0)
            return CUDA_SAM3D_E_LOAD;
        ctx->gpu_ppe_ws_alloced = 1;
    }

    /* Lazy-load fuser GPU + workspace. */
    if (!ctx->gpu_fuser_loaded) {
        if (cs3d_fuser_gpu_load(&ctx->gpu_fuser, fus_m, ctx->cfg.verbose) < 0)
            return CUDA_SAM3D_E_LOAD;
        ctx->gpu_fuser_loaded = 1;
        if (cs3d_fuser_fns_lookup(&ctx->gpu_fuser_fns, ctx->mod) < 0)
            return CUDA_SAM3D_E_LOAD;
        int N_max = branch > Nwin ? branch : Nwin;
        int Di_max = 0, Dh_max = 0;
        for (int i = 0; i < ctx->gpu_fuser.n_modalities; i++) {
            if (ctx->gpu_fuser.projs[i].Di > Di_max) Di_max = ctx->gpu_fuser.projs[i].Di;
            if (ctx->gpu_fuser.projs[i].Dh > Dh_max) Dh_max = ctx->gpu_fuser.projs[i].Dh;
        }
        if (cs3d_fuser_ws_alloc(&ctx->gpu_fuser_ws, N_max, Di_max, Dh_max) < 0)
            return CUDA_SAM3D_E_LOAD;
        ctx->gpu_fuser_ws_alloced = 1;
    }

    /* Run PPE on-device into d_ppe[Nwin, D_ppe]. */
    CUdeviceptr d_ppe = 0;
    if (have_ppe) {
        float *pmap_S = (float *)malloc((size_t)S * S * 3 * sizeof(float));
        if (!pmap_S) return CUDA_SAM3D_E_LOAD;
        int IH = ctx->pmap_h, IW = ctx->pmap_w;
        for (int oy = 0; oy < S; oy++) {
            int iy = (int)((float)oy * IH / S); if (iy >= IH) iy = IH - 1;
            for (int ox = 0; ox < S; ox++) {
                int ix = (int)((float)ox * IW / S); if (ix >= IW) ix = IW - 1;
                const float *src = ctx->pmap_f32 + ((size_t)iy * IW + ix) * 3;
                float *dst = pmap_S + ((size_t)oy * S + ox) * 3;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
            }
        }
        CUdeviceptr d_pmap = cu_upload_raw(pmap_S, (size_t)S * S * 3 * sizeof(float));
        free(pmap_S);
        if (!d_pmap ||
            cuMemAlloc(&d_ppe, (size_t)Nwin * D_ppe * sizeof(float)) != CUDA_SUCCESS) {
            if (d_pmap) cuMemFree(d_pmap);
            return CUDA_SAM3D_E_LOAD;
        }
        if (cs3d_ppe_forward(&ctx->gpu_ppe_fns, &ctx->gpu_ppe_ws,
                             &ctx->gpu_ppe.w, d_pmap, d_ppe,
                             ctx->gpu_ppe.n_heads, ctx->gpu_ppe.ln_eps) < 0) {
            cuMemFree(d_pmap); cuMemFree(d_ppe);
            return CUDA_SAM3D_E_LOAD;
        }
        cuMemFree(d_pmap);
    }

    /* Upload dino tokens, allocate output cond buffer. */
    CUdeviceptr d_dino = cu_upload_raw(dino->data,
                                       (size_t)dino->n * Di_dino * sizeof(float));
    CUdeviceptr d_cond = 0;
    if (!d_dino ||
        cuMemAlloc(&d_cond, (size_t)n_total * Do * sizeof(float)) != CUDA_SUCCESS) {
        if (d_dino) cuMemFree(d_dino);
        if (d_ppe)  cuMemFree(d_ppe);
        return CUDA_SAM3D_E_LOAD;
    }

    /* Project each modality, writing into the right slice of d_cond.
     * idx_emb add is folded into the final w2 gemm via gemm_f32_bias. */
    CUdeviceptr pos_row = ctx->gpu_fuser.idx_emb_rows[SAM3D_FUSER_POS_FULL];
    {
        const cs3d_fuser_proj_w *p0 = &ctx->gpu_fuser.projs[SAM3D_FUSER_MOD_DINO_IMG];
        CUdeviceptr in_img = d_dino;
        CUdeviceptr out_img = d_cond + 0;
        if (cs3d_fuser_project_forward(&ctx->gpu_fuser_fns, &ctx->gpu_fuser_ws,
                                       p0, in_img, out_img, branch, pos_row, 1e-5f) < 0)
            goto fuser_fail;
    }
    {
        const cs3d_fuser_proj_w *p1 = &ctx->gpu_fuser.projs[SAM3D_FUSER_MOD_DINO_MSK];
        CUdeviceptr in_msk = d_dino + (size_t)branch * Di_dino * sizeof(float);
        CUdeviceptr out_msk = d_cond + (size_t)branch * Do * sizeof(float);
        if (cs3d_fuser_project_forward(&ctx->gpu_fuser_fns, &ctx->gpu_fuser_ws,
                                       p1, in_msk, out_msk, branch, pos_row, 1e-5f) < 0)
            goto fuser_fail;
    }
    if (have_ppe) {
        const cs3d_fuser_proj_w *p2 = &ctx->gpu_fuser.projs[SAM3D_FUSER_MOD_POINT];
        CUdeviceptr out_pt = d_cond + (size_t)2 * branch * Do * sizeof(float);
        if (cs3d_fuser_project_forward(&ctx->gpu_fuser_fns, &ctx->gpu_fuser_ws,
                                       p2, d_ppe, out_pt, Nwin, pos_row, 1e-5f) < 0)
            goto fuser_fail;
    }
    cuCtxSynchronize();

    /* D2H mirror so verify_*.c and downstream stages can read host. */
    float *feats = (float *)malloc((size_t)n_total * Do * sizeof(float));
    if (!feats) goto fuser_fail;
    cuMemcpyDtoH(feats, d_cond, (size_t)n_total * Do * sizeof(float));
    if (d_dino) cuMemFree(d_dino);
    if (d_ppe)  cuMemFree(d_ppe);
    cuMemFree(d_cond);

    (void)nthr;
    return cs3d_adopt_cond_tokens(ctx, feats, n_total, Do);

fuser_fail:
    if (d_dino) cuMemFree(d_dino);
    if (d_ppe)  cuMemFree(d_ppe);
    if (d_cond) cuMemFree(d_cond);
    fprintf(stderr, "cuda_sam3d: GPU fuser projection failed\n");
    return CUDA_SAM3D_E_LOAD;
}
/* Active cond source: prefer override (verify_*.c injects ref data). */
static const cs3d_host_2d *cs3d_active_cond(const cuda_sam3d_ctx *ctx)
{
    return ctx->ovr_cond.data ? &ctx->ovr_cond : &ctx->cond_tokens;
}

static int cs3d_ensure_ss_dit(cuda_sam3d_ctx *ctx) {
    if (ctx->cpu_ss_dit) return CUDA_SAM3D_E_OK;
    ctx->cpu_ss_dit = sam3d_cpu_ss_dit_load(ctx->safetensors_dir_resolved);
    if (!ctx->cpu_ss_dit) {
        fprintf(stderr, "cuda_sam3d: ss_dit load failed (dir=%s)\n",
                ctx->safetensors_dir_resolved);
        return CUDA_SAM3D_E_LOAD;
    }
    return CUDA_SAM3D_E_OK;
}

static int cs3d_adopt_ss_latent(cuda_sam3d_ctx *ctx, float *data,
                                const int dims[4])
{
    cs3d_free_nd(&ctx->ss_latent);
    if (ctx->d_ss_latent) { cuMemFree(ctx->d_ss_latent); ctx->d_ss_latent = 0; }
    if (!data) return CUDA_SAM3D_E_INVAL;
    size_t numel = 1;
    for (int i = 0; i < 4; i++) {
        if (dims[i] <= 0) { free(data); return CUDA_SAM3D_E_INVAL; }
        ctx->ss_latent.dims[i] = dims[i];
        numel *= (size_t)dims[i];
    }
    ctx->ss_latent.ndim = 4;
    ctx->ss_latent.data = data;
    ctx->d_ss_latent = cu_upload_raw(data, numel * sizeof(float));
    if (!ctx->d_ss_latent) {
        cs3d_free_nd(&ctx->ss_latent);
        return CUDA_SAM3D_E_LOAD;
    }
    return CUDA_SAM3D_E_OK;
}

static int cs3d_ensure_ssdit_gpu(cuda_sam3d_ctx *ctx, int n_cond_min);

/* Match sam3d_cpu.c's xorshift64* + Box-Muller schedule byte-for-byte so
 * a fixed seed yields identical noise across CPU and GPU paths. */
static inline uint64_t cs3d_ode_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static inline float cs3d_ode_rng_u01(uint64_t *state) {
    uint64_t r = cs3d_ode_rng_next(state) >> 11;
    return (float)((double)r * (1.0 / 9007199254740992.0));
}
static void cs3d_ode_fill_randn(float *buf, int n, uint64_t *state) {
    for (int i = 0; i < n; i += 2) {
        float u1 = cs3d_ode_rng_u01(state); if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = cs3d_ode_rng_u01(state);
        float r = sqrtf(-2.0f * logf(u1));
        float a = 6.2831853f * u2;
        buf[i] = r * cosf(a);
        if (i + 1 < n) buf[i + 1] = r * sinf(a);
    }
}

int cuda_sam3d_run_ss_dit(cuda_sam3d_ctx *ctx) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_2d *cond = cs3d_active_cond(ctx);
    if (!cond->data) {
        fprintf(stderr, "cuda_sam3d: ss_dit needs cond tokens — "
                        "run cond_fuser (or override) first\n");
        return CUDA_SAM3D_E_NO_INPUT;
    }
    int rc = cs3d_ensure_ssdit_gpu(ctx, cond->n);
    if (rc != CUDA_SAM3D_E_OK) return rc;

    const sam3d_ss_flow_dit_model *m = sam3d_cpu_ss_dit_model(ctx->cpu_ss_dit);
    if (!m) return CUDA_SAM3D_E_LOAD;

    int steps = ctx->cfg.ss_steps > 0 ? ctx->cfg.ss_steps : 2;

    float *lat[SAM3D_SS_DIT_N_LATENTS] = {0};
    float *vel[SAM3D_SS_DIT_N_LATENTS] = {0};
    float *vel_u[SAM3D_SS_DIT_N_LATENTS] = {0};
    int    nlat[SAM3D_SS_DIT_N_LATENTS] = {0};
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        nlat[i] = sam3d_cpu_ss_dit_lat_elts(i);
        lat[i] = (float *)malloc((size_t)nlat[i] * sizeof(float));
        vel[i] = (float *)malloc((size_t)nlat[i] * sizeof(float));
        vel_u[i] = (float *)malloc((size_t)nlat[i] * sizeof(float));
        if (!lat[i] || !vel[i] || !vel_u[i]) goto oom;
    }

    uint64_t rng = ctx->cfg.seed ? (uint64_t)ctx->cfg.seed
                                 : 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++)
        cs3d_ode_fill_randn(lat[i], nlat[i], &rng);

    float *times = (float *)malloc((size_t)(steps + 1) * sizeof(float));
    if (!times) goto oom;
    /* Match upstream pointmap inference: no_shortcut=True, ss_rescale_t=3,
     * reversed_timestamp=False. The model still has a d-embedder, but
     * production inference passes d=0. */
    sam3d_shortcut_make_times(times, steps, 3.0f, /*reversed=*/0);
    float d = sam3d_shortcut_d(steps, /*no_shortcut=*/1);
    const float TIME_SCALE = m->time_scale;

    /* One-shot cond HtoD upload — every step then sees it via the
     * outer workspace (NULL host pointer skips the per-step copy). */
    if (cs3d_ssdit_outer_upload_cond(&ctx->gpu_ssdit_outer_ws,
                                     cond->data, cond->n, m->dim) < 0) {
        free(times);
        fprintf(stderr, "cuda_sam3d: ss_dit cond upload failed\n");
        goto oom;
    }

    float *zero_cond = NULL;
    if (ctx->cfg.cfg_scale > 0.0f) {
        zero_cond = (float *)calloc((size_t)cond->n * cond->c, sizeof(float));
        if (!zero_cond) { free(times); goto oom; }
    }
    int cfg_steps = 0;

    for (int s = 0; s < steps; s++) {
        float t  = times[s];
        float dt = times[s + 1] - times[s];
        float ts = t * TIME_SCALE;
        if (cs3d_ssdit_outer_forward(&ctx->gpu_ssdit, &ctx->gpu_ssdit_fns,
                                     &ctx->gpu_ssdit_block_ws,
                                     &ctx->gpu_ssdit_outer_ws,
                                     (const float *const *)lat, vel,
                                     /*cond=*/NULL, cond->n,
                                     ts, d * TIME_SCALE) < 0) {
            free(times); free(zero_cond);
            fprintf(stderr, "cuda_sam3d: ss_dit GPU forward failed (step %d)\n", s);
            goto oom;
        }
        if (zero_cond && sam3d_shortcut_cfg_active(ts, 0.0f, 500.0f)) {
            if (cs3d_ssdit_outer_upload_cond(&ctx->gpu_ssdit_outer_ws,
                                             zero_cond, cond->n, m->dim) < 0 ||
                cs3d_ssdit_outer_forward(&ctx->gpu_ssdit, &ctx->gpu_ssdit_fns,
                                         &ctx->gpu_ssdit_block_ws,
                                         &ctx->gpu_ssdit_outer_ws,
                                         (const float *const *)lat, vel_u,
                                         /*cond=*/NULL, cond->n,
                                         ts, d * TIME_SCALE) < 0) {
                free(times); free(zero_cond);
                fprintf(stderr, "cuda_sam3d: ss_dit GPU uncond forward failed (step %d)\n", s);
                goto oom;
            }
            for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
                sam3d_shortcut_cfg_combine(vel[i], vel[i], vel_u[i],
                                           ctx->cfg.cfg_scale, nlat[i]);
            }
            cfg_steps++;
            if (s + 1 < steps &&
                cs3d_ssdit_outer_upload_cond(&ctx->gpu_ssdit_outer_ws,
                                             cond->data, cond->n, m->dim) < 0) {
                free(times); free(zero_cond);
                fprintf(stderr, "cuda_sam3d: ss_dit cond restore failed\n");
                goto oom;
            }
        }
        for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++)
            sam3d_shortcut_euler_step(lat[i], vel[i], dt, nlat[i]);
    }
    free(times); free(zero_cond);

    /* SHAPE → NCDHW [8, 16, 16, 16] from DiT layout [N=4096, C=8]. */
    const int dims[4] = {8, 16, 16, 16};
    size_t numel = 8 * 16 * 16 * 16;
    float *latent = (float *)malloc(numel * sizeof(float));
    if (!latent) goto oom;
    for (int n = 0; n < 4096; n++)
        for (int c = 0; c < 8; c++)
            latent[c * 4096 + n] = lat[SAM3D_SS_LAT_SHAPE][n * 8 + c];

    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        free(lat[i]); free(vel[i]); free(vel_u[i]);
    }
    if (ctx->cfg.verbose >= 1) {
        float mn = latent[0], mx = latent[0];
        double sum = 0.0;
        for (int i = 0; i < 8 * 4096; i++) {
            float v = latent[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        fprintf(stderr,
                "cuda_sam3d: ss_dit latent min=%.6g max=%.6g mean=%.6g d=%.3g rescale_t=3 reversed=0 cfg_steps=%d\n",
                mn, mx, sum / (8.0 * 4096.0), d, cfg_steps);
    }
    return cs3d_adopt_ss_latent(ctx, latent, dims);

oom:
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        free(lat[i]); free(vel[i]); free(vel_u[i]);
    }
    return CUDA_SAM3D_E_LOAD;
}

/* Lazy-init GPU SS Flow DiT (weights upload + fns lookup + workspaces).
 * `n_cond_min` is the largest cond-token count we must support; the outer
 * workspace is grown on-demand if a later call exceeds the cap. */
static int cs3d_ensure_ssdit_gpu(cuda_sam3d_ctx *ctx, int n_cond_min)
{
    int rc = cs3d_ensure_ss_dit(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;
    sam3d_ss_flow_dit_model *m = sam3d_cpu_ss_dit_model(ctx->cpu_ss_dit);
    if (!m) return CUDA_SAM3D_E_LOAD;

    if (!ctx->gpu_ssdit_loaded) {
        if (cs3d_ssdit_gpu_load(&ctx->gpu_ssdit, m, ctx->cfg.verbose) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu ss_dit weight upload failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        ctx->gpu_ssdit_loaded = 1;
        if (cs3d_ssdit_outer_fns_lookup(&ctx->gpu_ssdit_fns, ctx->mod) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu ss_dit fns lookup failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
    }

    if (ctx->gpu_ssdit_ws_alloced && n_cond_min > ctx->gpu_ssdit_ws_n_c) {
        cs3d_ssdit_outer_ws_free(&ctx->gpu_ssdit_outer_ws);
        cs3d_ssdit_block_ws_free(&ctx->gpu_ssdit_block_ws);
        ctx->gpu_ssdit_ws_alloced = 0;
    }
    if (!ctx->gpu_ssdit_ws_alloced) {
        const cs3d_ssdit_gpu *g = &ctx->gpu_ssdit;
        int N_s = g->latent[SAM3D_SS_LAT_SHAPE].token_len;
        int N_p = 0;
        for (int i = SAM3D_SS_LAT_6DROT; i <= SAM3D_SS_LAT_TRANSLATION_SCALE; i++)
            N_p += g->latent[i].token_len;
        if (cs3d_ssdit_block_ws_alloc(&ctx->gpu_ssdit_block_ws,
                                      N_s, N_p, n_cond_min,
                                      g->dim, g->mlp_hidden) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu ss_dit block_ws alloc failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        if (cs3d_ssdit_outer_ws_alloc(&ctx->gpu_ssdit_outer_ws, g, n_cond_min) < 0) {
            cs3d_ssdit_block_ws_free(&ctx->gpu_ssdit_block_ws);
            fprintf(stderr, "cuda_sam3d: gpu ss_dit outer_ws alloc failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        ctx->gpu_ssdit_ws_alloced = 1;
        ctx->gpu_ssdit_ws_n_c = n_cond_min;
    }
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_debug_ss_dit_forward(cuda_sam3d_ctx *ctx,
                                    const float *const *latents_in,
                                    float *const *latents_out,
                                    const float *cond, int n_cond,
                                    float t, float d)
{
    if (!ctx || !latents_in || !latents_out || !cond || n_cond <= 0)
        return CUDA_SAM3D_E_INVAL;
    int rc = cs3d_ensure_ssdit_gpu(ctx, n_cond);
    if (rc != CUDA_SAM3D_E_OK) return rc;
    return cs3d_ssdit_outer_forward(&ctx->gpu_ssdit, &ctx->gpu_ssdit_fns,
                                    &ctx->gpu_ssdit_block_ws,
                                    &ctx->gpu_ssdit_outer_ws,
                                    latents_in, latents_out,
                                    cond, n_cond, t, d) == 0
           ? CUDA_SAM3D_E_OK : CUDA_SAM3D_E_LOAD;
}

int cuda_sam3d_ss_dit_info(cuda_sam3d_ctx *ctx,
                           int *out_n_blocks, int *out_dim,
                           int *out_cond_channels, int *out_is_shortcut)
{
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    int rc = cs3d_ensure_ss_dit(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;
    if (out_n_blocks)      *out_n_blocks      = sam3d_cpu_ss_dit_n_blocks     (ctx->cpu_ss_dit);
    if (out_dim)           *out_dim           = sam3d_cpu_ss_dit_dim          (ctx->cpu_ss_dit);
    if (out_cond_channels) *out_cond_channels = sam3d_cpu_ss_dit_cond_channels(ctx->cpu_ss_dit);
    if (out_is_shortcut)   *out_is_shortcut   = sam3d_cpu_ss_dit_is_shortcut  (ctx->cpu_ss_dit);
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_ss_dit_n_latents(void) { return sam3d_cpu_ss_dit_n_latents(); }
int cuda_sam3d_ss_dit_lat_elts(int i) { return sam3d_cpu_ss_dit_lat_elts(i); }

/* Active ss_latent source: prefer override. Returns NULL if neither is set. */
static const cs3d_host_nd *cs3d_active_ss_latent(const cuda_sam3d_ctx *ctx)
{
    return ctx->ovr_ss_latent.data ? &ctx->ovr_ss_latent : &ctx->ss_latent;
}

static int cs3d_adopt_occupancy(cuda_sam3d_ctx *ctx, float *data,
                                const int dims[3])
{
    cs3d_free_nd(&ctx->occupancy);
    if (ctx->d_occupancy) { cuMemFree(ctx->d_occupancy); ctx->d_occupancy = 0; }
    if (!data) return CUDA_SAM3D_E_INVAL;
    size_t numel = 1;
    for (int i = 0; i < 3; i++) {
        if (dims[i] <= 0) { free(data); return CUDA_SAM3D_E_INVAL; }
        ctx->occupancy.dims[i] = dims[i];
        numel *= (size_t)dims[i];
    }
    ctx->occupancy.ndim = 3;
    ctx->occupancy.data = data;
    ctx->d_occupancy = cu_upload_raw(data, numel * sizeof(float));
    if (!ctx->d_occupancy) {
        cs3d_free_nd(&ctx->occupancy);
        return CUDA_SAM3D_E_LOAD;
    }
    return CUDA_SAM3D_E_OK;
}

static int cs3d_ensure_ssdec_gpu(cuda_sam3d_ctx *ctx)
{
    if (!ctx->cpu_ss_dec) {
        ctx->cpu_ss_dec = sam3d_cpu_ss_dec_load(ctx->safetensors_dir_resolved);
        if (!ctx->cpu_ss_dec) {
            fprintf(stderr, "cuda_sam3d: ss_decoder load failed (dir=%s)\n",
                    ctx->safetensors_dir_resolved);
            return CUDA_SAM3D_E_LOAD;
        }
    }
    if (!ctx->gpu_ssdec_loaded) {
        const t2_ss_dec *m = (const t2_ss_dec *)sam3d_cpu_ss_dec_model(ctx->cpu_ss_dec);
        if (!m || cs3d_ssdec_gpu_load(&ctx->gpu_ssdec, m, ctx->cfg.verbose) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu ss_decoder weight upload failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        ctx->gpu_ssdec_loaded = 1;
        if (cs3d_ssdec_fns_init(&ctx->gpu_ssdec_fns, ctx->mod) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu ss_decoder fns lookup failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
    }
    if (!ctx->gpu_ssdec_ws_alloced) {
        if (cs3d_ssdec_ws_alloc(&ctx->gpu_ssdec_ws) < 0) {
            fprintf(stderr, "cuda_sam3d: gpu ss_decoder workspace alloc failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        ctx->gpu_ssdec_ws_alloced = 1;
    }
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_run_ss_decode(cuda_sam3d_ctx *ctx) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_nd *lat = cs3d_active_ss_latent(ctx);
    if (!lat->data) {
        fprintf(stderr, "cuda_sam3d: ss_decode needs ss_latent — "
                        "run ss_dit (or override) first\n");
        return CUDA_SAM3D_E_NO_INPUT;
    }
    if (lat->ndim != 4 || lat->dims[0] != 8 || lat->dims[1] != 16 ||
        lat->dims[2] != 16 || lat->dims[3] != 16) {
        fprintf(stderr, "cuda_sam3d: ss_decode expects ss_latent [8,16,16,16], "
                        "got ndim=%d [%d,%d,%d,%d]\n",
                lat->ndim, lat->dims[0], lat->dims[1], lat->dims[2], lat->dims[3]);
        return CUDA_SAM3D_E_INVAL;
    }
    int rc = cs3d_ensure_ssdec_gpu(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;

    CUdeviceptr d_lat = 0, d_out = 0;
    size_t lat_bytes = (size_t)8 * 16 * 16 * 16 * sizeof(float);
    d_lat = cu_upload_raw(lat->data, lat_bytes);
    if (!d_lat) return CUDA_SAM3D_E_LOAD;
    if (cuMemAlloc(&d_out, (size_t)64 * 64 * 64 * sizeof(float)) != CUDA_SUCCESS) {
        cuMemFree(d_lat);
        return CUDA_SAM3D_E_LOAD;
    }
    if (cs3d_ssdec_forward(&ctx->gpu_ssdec, &ctx->gpu_ssdec_fns,
                           &ctx->gpu_ssdec_ws, d_lat, d_out,
                           ctx->cfg.verbose) < 0) {
        fprintf(stderr, "cuda_sam3d: gpu ss_decoder forward failed\n");
        cuMemFree(d_lat); cuMemFree(d_out);
        return CUDA_SAM3D_E_LOAD;
    }
    cuMemFree(d_lat);

    float *logits = (float *)malloc((size_t)64 * 64 * 64 * sizeof(float));
    if (!logits) { cuMemFree(d_out); return CUDA_SAM3D_E_LOAD; }
    if (cuMemcpyDtoH(logits, d_out, (size_t)64 * 64 * 64 * sizeof(float)) != CUDA_SUCCESS) {
        free(logits); cuMemFree(d_out); return CUDA_SAM3D_E_LOAD;
    }
    cuMemFree(d_out);
    if (ctx->cfg.verbose >= 1) {
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
        fprintf(stderr,
                "cuda_sam3d: ss_decode occupancy min=%.6g max=%.6g mean=%.6g pos=%d\n",
                mn, mx, sum / (64.0 * 64.0 * 64.0), pos);
    }
    const int dims[3] = {64, 64, 64};
    return cs3d_adopt_occupancy(ctx, logits, dims);
}
/* Active occupancy source: prefer override. */
static const cs3d_host_nd *cs3d_active_occupancy(const cuda_sam3d_ctx *ctx)
{
    return ctx->ovr_occupancy.data ? &ctx->ovr_occupancy : &ctx->occupancy;
}

static int cs3d_ensure_slat_dit(cuda_sam3d_ctx *ctx) {
    if (ctx->cpu_slat_dit) return CUDA_SAM3D_E_OK;
    ctx->cpu_slat_dit = sam3d_cpu_slat_dit_load(ctx->safetensors_dir_resolved);
    if (!ctx->cpu_slat_dit) {
        fprintf(stderr, "cuda_sam3d: slat_dit load failed (dir=%s)\n",
                ctx->safetensors_dir_resolved);
        return CUDA_SAM3D_E_LOAD;
    }
    return CUDA_SAM3D_E_OK;
}

static int cs3d_ensure_slatdit_gpu(cuda_sam3d_ctx *ctx, int n_max, int n_cond)
{
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;
    sam3d_slat_dit_model *m = sam3d_cpu_slat_dit_model(ctx->cpu_slat_dit);
    if (!m) return CUDA_SAM3D_E_LOAD;

    if (!ctx->gpu_slatdit_loaded) {
        if (cs3d_slatdit_gpu_load_transformer(&ctx->gpu_slatdit, m, ctx->cfg.verbose) != 0) {
            fprintf(stderr, "cuda_sam3d: gpu slat_dit transformer weight upload failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        ctx->gpu_slatdit_loaded = 1;
        if (cs3d_slatdit_fns_lookup(&ctx->gpu_slatdit_fns, ctx->mod) != 0) {
            fprintf(stderr, "cuda_sam3d: gpu slat_dit fns lookup failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
    }

    if (ctx->gpu_slatdit_ws_alloced &&
        (n_max > ctx->gpu_slatdit_ws_n || n_cond > ctx->gpu_slatdit_ws_nc)) {
        cs3d_slatdit_block_ws_free(&ctx->gpu_slatdit_ws);
        ctx->gpu_slatdit_ws_alloced = 0;
    }
    if (!ctx->gpu_slatdit_ws_alloced) {
        if (cs3d_slatdit_block_ws_alloc(&ctx->gpu_slatdit_ws, n_max, n_cond,
                                        ctx->gpu_slatdit.dim,
                                        ctx->gpu_slatdit.mlp_hidden) != 0) {
            fprintf(stderr, "cuda_sam3d: gpu slat_dit transformer ws alloc failed\n");
            return CUDA_SAM3D_E_LOAD;
        }
        ctx->gpu_slatdit_ws_alloced = 1;
        ctx->gpu_slatdit_ws_n = n_max;
        ctx->gpu_slatdit_ws_nc = n_cond;
    }
    return CUDA_SAM3D_E_OK;
}

static int cs3d_ensure_devbuf(CUdeviceptr *ptr, size_t *cap, size_t need)
{
    if (!ptr || !cap || need == 0) return -1;
    if (*ptr && *cap >= need) return 0;
    if (*ptr) {
        cuMemFree(*ptr);
        *ptr = 0;
        *cap = 0;
    }
    if (cuMemAlloc(ptr, need) != CUDA_SUCCESS) return -1;
    *cap = need;
    return 0;
}

static int cs3d_slat_transformer_gpu_hook(void *user, float *feats, int N,
                                          const int32_t *coords,
                                          const float *t_emb,
                                          const float *cond, int n_cond,
                                          int dim, int n_blocks)
{
    cuda_sam3d_ctx *ctx = (cuda_sam3d_ctx *)user;
    (void)coords;
    if (!ctx || !feats || !t_emb || !cond || N <= 0 || n_cond <= 0)
        return -1;
    if (cs3d_ensure_slatdit_gpu(ctx, N, n_cond) != CUDA_SAM3D_E_OK)
        return -1;
    if (dim != ctx->gpu_slatdit.dim || n_blocks != ctx->gpu_slatdit.n_blocks)
        return -1;

    size_t x_bytes = (size_t)N * dim * sizeof(float);
    size_t c_bytes = (size_t)n_cond * dim * sizeof(float);
    if (cs3d_ensure_devbuf(&ctx->d_slat_hook_x, &ctx->slat_hook_x_bytes, x_bytes) != 0 ||
        cs3d_ensure_devbuf(&ctx->d_slat_hook_t_emb, &ctx->slat_hook_t_emb_bytes,
                           (size_t)dim * sizeof(float)) != 0)
        return -1;
    if (cuMemcpyHtoD(ctx->d_slat_hook_x, feats, x_bytes) != CUDA_SUCCESS ||
        cuMemcpyHtoD(ctx->d_slat_hook_t_emb, t_emb, (size_t)dim * sizeof(float)) != CUDA_SUCCESS)
        return -1;

    int own_cond = 1;
    CUdeviceptr d_cond = 0;
    if (cond == ctx->cond_tokens.data && ctx->d_cond_tokens &&
        ctx->cond_tokens.n == n_cond && ctx->cond_tokens.c == dim) {
        d_cond = ctx->d_cond_tokens;
        own_cond = 0;
    } else {
        d_cond = cu_upload_raw(cond, c_bytes);
    }
    if (!d_cond) {
        if (d_cond && own_cond) cuMemFree(d_cond);
        return -1;
    }

    int ok = (cs3d_slatdit_stack_forward(&ctx->gpu_slatdit_fns,
                                         &ctx->gpu_slatdit_ws,
                                         &ctx->gpu_slatdit, 0, n_blocks,
                                         ctx->d_slat_hook_t_emb,
                                         ctx->d_slat_hook_x, N, d_cond,
                                         n_cond) == 0);
    if (ok && cuMemcpyDtoH(feats, ctx->d_slat_hook_x, x_bytes) != CUDA_SUCCESS)
        ok = 0;
    if (own_cond) cuMemFree(d_cond);
    return ok ? 0 : -1;
}

static int cs3d_adopt_slat(cuda_sam3d_ctx *ctx, int32_t *coords, float *feats,
                           int n, int c)
{
    cs3d_free_slat(&ctx->slat_tokens);
    if (ctx->d_slat_feats)  { cuMemFree(ctx->d_slat_feats);  ctx->d_slat_feats  = 0; }
    if (ctx->d_slat_coords) { cuMemFree(ctx->d_slat_coords); ctx->d_slat_coords = 0; }
    if (!coords || !feats || n <= 0 || c <= 0) {
        free(coords); free(feats);
        return CUDA_SAM3D_E_INVAL;
    }
    size_t f_bytes = (size_t)n * c * sizeof(float);
    size_t k_bytes = (size_t)n * 4 * sizeof(int32_t);
    ctx->d_slat_feats  = cu_upload_raw(feats,  f_bytes);
    ctx->d_slat_coords = cu_upload_raw(coords, k_bytes);
    if (!ctx->d_slat_feats || !ctx->d_slat_coords) {
        if (ctx->d_slat_feats)  { cuMemFree(ctx->d_slat_feats);  ctx->d_slat_feats  = 0; }
        if (ctx->d_slat_coords) { cuMemFree(ctx->d_slat_coords); ctx->d_slat_coords = 0; }
        free(coords); free(feats);
        return CUDA_SAM3D_E_LOAD;
    }
    ctx->slat_tokens.feats  = feats;
    ctx->slat_tokens.coords = coords;
    ctx->slat_tokens.n      = n;
    ctx->slat_tokens.c      = c;
    return CUDA_SAM3D_E_OK;
}

static int cs3d_slat_argwhere_gpu(cuda_sam3d_ctx *ctx, const cs3d_host_nd *occ,
                                  int32_t **out_coords, int *out_n)
{
    if (!ctx || !occ || !occ->data || !out_coords || !out_n ||
        occ->ndim != 3 || occ->dims[0] <= 0 || occ->dims[1] <= 0 ||
        occ->dims[2] <= 0) return CUDA_SAM3D_E_INVAL;

    hipFunction_t fn = NULL;
    if (cuModuleGetFunction(&fn, ctx->mod,
                            "slat_occ_argwhere_serial_f32") != CUDA_SUCCESS) {
        fprintf(stderr, "cuda_sam3d: missing slat_occ_argwhere_serial_f32\n");
        return CUDA_SAM3D_E_LOAD;
    }

    int D = occ->dims[0], H = occ->dims[1], W = occ->dims[2];
    int occ_n = D * H * W;
    CUdeviceptr d_occ = 0, d_count = 0, d_coords = 0;
    int own_occ = 0;
    if (occ == &ctx->occupancy && ctx->d_occupancy) {
        d_occ = ctx->d_occupancy;
    } else {
        d_occ = cu_upload_raw(occ->data, (size_t)occ_n * sizeof(float));
        own_occ = 1;
        if (!d_occ) return CUDA_SAM3D_E_LOAD;
    }
    if (cuMemAlloc(&d_count, sizeof(int)) != CUDA_SUCCESS ||
        cuMemAlloc(&d_coords, (size_t)occ_n * 4 * sizeof(int32_t)) != CUDA_SUCCESS) {
        if (d_count) cuMemFree(d_count);
        if (d_coords) cuMemFree(d_coords);
        if (own_occ) cuMemFree(d_occ);
        return CUDA_SAM3D_E_LOAD;
    }

    void *args[] = { &d_occ, &D, &H, &W, &d_count, &d_coords };
    CUresult lrc = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    if (lrc == CUDA_SUCCESS) lrc = cuCtxSynchronize();
    int count = 0;
    if (lrc == CUDA_SUCCESS)
        lrc = cuMemcpyDtoH(&count, d_count, sizeof(int));
    if (lrc != CUDA_SUCCESS || count < 0 || count > occ_n) {
        cuMemFree(d_count);
        cuMemFree(d_coords);
        if (own_occ) cuMemFree(d_occ);
        return CUDA_SAM3D_E_LOAD;
    }
    if (count == 0) {
        cuMemFree(d_count);
        cuMemFree(d_coords);
        if (own_occ) cuMemFree(d_occ);
        *out_coords = NULL;
        *out_n = 0;
        return CUDA_SAM3D_E_OK;
    }

    int32_t *coords = (int32_t *)malloc((size_t)count * 4 * sizeof(int32_t));
    if (!coords) {
        cuMemFree(d_count);
        cuMemFree(d_coords);
        if (own_occ) cuMemFree(d_occ);
        return CUDA_SAM3D_E_LOAD;
    }
    lrc = cuMemcpyDtoH(coords, d_coords, (size_t)count * 4 * sizeof(int32_t));
    cuMemFree(d_count);
    cuMemFree(d_coords);
    if (own_occ) cuMemFree(d_occ);
    if (lrc != CUDA_SUCCESS) {
        free(coords);
        return CUDA_SAM3D_E_LOAD;
    }
    *out_coords = coords;
    *out_n = count;
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_run_slat_dit(cuda_sam3d_ctx *ctx) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_2d *cond = cs3d_active_cond(ctx);
    if (!cond->data) {
        fprintf(stderr, "cuda_sam3d: slat_dit needs cond tokens — "
                        "run cond_fuser (or override) first\n");
        return CUDA_SAM3D_E_NO_INPUT;
    }
    const cs3d_host_nd *occ = cs3d_active_occupancy(ctx);
    if (!occ->data) {
        fprintf(stderr, "cuda_sam3d: slat_dit needs occupancy — "
                        "run ss_decode (or override) first\n");
        return CUDA_SAM3D_E_NO_INPUT;
    }
    if (occ->ndim != 3 || occ->dims[0] != 64 || occ->dims[1] != 64 ||
        occ->dims[2] != 64) {
        fprintf(stderr, "cuda_sam3d: slat_dit expects occupancy [64,64,64], "
                        "got ndim=%d [%d,%d,%d]\n",
                occ->ndim, occ->dims[0], occ->dims[1], occ->dims[2]);
        return CUDA_SAM3D_E_INVAL;
    }
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;

#if defined(_OPENMP)
    int nthr = omp_get_max_threads(); if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif
    int steps = ctx->cfg.slat_steps > 0 ? ctx->cfg.slat_steps : 12;

    int32_t *active_coords = NULL;
    int active_voxels = 0;
    rc = cs3d_slat_argwhere_gpu(ctx, occ, &active_coords, &active_voxels);
    if (rc != CUDA_SAM3D_E_OK) return rc;
    if (ctx->cfg.verbose >= 1) {
        fprintf(stderr, "cuda_sam3d: slat_dit active voxels=%d (gpu argwhere occ>0), steps=%d\n",
                active_voxels, steps);
    }
    if (active_voxels <= 0) {
        fprintf(stderr, "cuda_sam3d: slat_dit occupancy is fully negative\n");
        free(active_coords);
        return CUDA_SAM3D_E_LOAD;
    }

    int32_t *out_coords = NULL;
    float   *out_feats  = NULL;
    int      out_n      = 0;
    sam3d_cpu_slat_dit_set_transformer_hook(cs3d_slat_transformer_gpu_hook, ctx);
    int slat_rc = sam3d_cpu_slat_dit_run_ode_from_coords(ctx->cpu_slat_dit,
                                                         active_coords, active_voxels,
                                                         cond->data, cond->n,
                                                         steps, ctx->cfg.seed, nthr,
                                                         &out_coords, &out_feats, &out_n);
    sam3d_cpu_slat_dit_set_transformer_hook(NULL, NULL);
    free(active_coords);
    if (slat_rc != 0) {
        free(out_coords); free(out_feats);
        fprintf(stderr, "cuda_sam3d: slat_dit ODE failed rc=%d active_voxels=%d\n",
                slat_rc, active_voxels);
        return CUDA_SAM3D_E_LOAD;
    }
    int out_c = sam3d_cpu_slat_dit_out_channels(ctx->cpu_slat_dit);
    return cs3d_adopt_slat(ctx, out_coords, out_feats, out_n, out_c);
}

int cuda_sam3d_debug_slat_dit_forward(cuda_sam3d_ctx *ctx,
                                      const int32_t *coords,
                                      const float *feats, int N,
                                      float t,
                                      const float *cond, int n_cond,
                                      float *out_feats)
{
    if (!ctx || !coords || !feats || N <= 0 || !cond || n_cond <= 0 || !out_feats)
        return CUDA_SAM3D_E_INVAL;
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;
#if defined(_OPENMP)
    int nthr = omp_get_max_threads(); if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif
    sam3d_cpu_slat_dit_set_transformer_hook(cs3d_slat_transformer_gpu_hook, ctx);
    float *out = sam3d_cpu_slat_dit_forward(ctx->cpu_slat_dit,
                                            coords, feats, N, t,
                                            cond, n_cond, nthr);
    sam3d_cpu_slat_dit_set_transformer_hook(NULL, NULL);
    if (!out) return CUDA_SAM3D_E_LOAD;
    int out_c = sam3d_cpu_slat_dit_out_channels(ctx->cpu_slat_dit);
    memcpy(out_feats, out, (size_t)N * out_c * sizeof(float));
    free(out);
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_slat_dit_info(cuda_sam3d_ctx *ctx,
                             int *out_in_channels, int *out_out_channels,
                             int *out_cond_channels)
{
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;
    if (out_in_channels)   *out_in_channels   = sam3d_cpu_slat_dit_in_channels  (ctx->cpu_slat_dit);
    if (out_out_channels)  *out_out_channels  = sam3d_cpu_slat_dit_out_channels (ctx->cpu_slat_dit);
    if (out_cond_channels) *out_cond_channels = sam3d_cpu_slat_dit_cond_channels(ctx->cpu_slat_dit);
    return CUDA_SAM3D_E_OK;
}

/* Active SLAT source: prefer override (verify_slat_gs injects ref data). */
static const cs3d_host_slat *cs3d_active_slat(const cuda_sam3d_ctx *ctx)
{
    return ctx->ovr_slat.feats ? &ctx->ovr_slat : &ctx->slat_tokens;
}

static int cs3d_ensure_gs(cuda_sam3d_ctx *ctx) {
    if (ctx->cpu_gs) return CUDA_SAM3D_E_OK;
    ctx->cpu_gs = sam3d_cpu_gs_decoder_load(ctx->safetensors_dir_resolved);
    if (!ctx->cpu_gs) {
        fprintf(stderr, "cuda_sam3d: gs_decoder load failed (dir=%s)\n",
                ctx->safetensors_dir_resolved);
        return CUDA_SAM3D_E_LOAD;
    }
    return CUDA_SAM3D_E_OK;
}

static int cs3d_adopt_gaussians(cuda_sam3d_ctx *ctx, float *data, int n_total)
{
    cs3d_free_2d(&ctx->gaussians);
    if (ctx->d_gaussians) { cuMemFree(ctx->d_gaussians); ctx->d_gaussians = 0; }
    if (!data || n_total <= 0) { free(data); return CUDA_SAM3D_E_INVAL; }
    size_t bytes = (size_t)n_total * CUDA_SAM3D_GS_STRIDE * sizeof(float);
    ctx->d_gaussians = cu_upload_raw(data, bytes);
    if (!ctx->d_gaussians) { free(data); return CUDA_SAM3D_E_LOAD; }
    ctx->gaussians.data = data;
    ctx->gaussians.n    = n_total;
    ctx->gaussians.c    = CUDA_SAM3D_GS_STRIDE;
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_debug_slat_gs_transformer(cuda_sam3d_ctx *ctx,
                                         const int32_t *coords,
                                         const float *feats, int N,
                                         float **out_feats, int *out_c)
{
    if (!ctx || !coords || !feats || N <= 0 || !out_feats) return CUDA_SAM3D_E_INVAL;
    int rc = cs3d_ensure_gs(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;
#if defined(_OPENMP)
    int nthr = omp_get_max_threads(); if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif
    float *out = sam3d_cpu_gs_decoder_transformer(ctx->cpu_gs, coords, feats,
                                                  N, nthr);
    if (!out) return CUDA_SAM3D_E_LOAD;
    *out_feats = out;
    if (out_c) *out_c = sam3d_cpu_gs_decoder_out_channels(ctx->cpu_gs);
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_debug_slat_gs_to_representation(cuda_sam3d_ctx *ctx,
                                               const int32_t *coords,
                                               const float *feats_out, int N,
                                               float *xyz_out, float *dc_out,
                                               float *scaling_out, float *rotation_out,
                                               float *opacity_out)
{
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    int rc = cs3d_ensure_gs(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;
    return sam3d_cpu_gs_decoder_to_representation(ctx->cpu_gs, coords, feats_out, N,
                                                  xyz_out, dc_out,
                                                  scaling_out, rotation_out,
                                                  opacity_out) == 0
           ? CUDA_SAM3D_E_OK : CUDA_SAM3D_E_LOAD;
}

int cuda_sam3d_slat_gs_info(cuda_sam3d_ctx *ctx,
                            int *out_in_channels, int *out_out_channels,
                            int *out_num_gaussians)
{
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    int rc = cs3d_ensure_gs(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;
    if (out_in_channels)   *out_in_channels   = sam3d_cpu_gs_decoder_in_channels  (ctx->cpu_gs);
    if (out_out_channels)  *out_out_channels  = sam3d_cpu_gs_decoder_out_channels (ctx->cpu_gs);
    if (out_num_gaussians) *out_num_gaussians = sam3d_cpu_gs_decoder_num_gaussians(ctx->cpu_gs);
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_run_slat_gs_decode(cuda_sam3d_ctx *ctx) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_slat *slat = cs3d_active_slat(ctx);
    if (!slat->feats || !slat->coords) {
        fprintf(stderr, "cuda_sam3d: slat_gs_decode needs SLAT tokens — "
                        "run slat_dit (or override) first\n");
        return CUDA_SAM3D_E_NO_INPUT;
    }
    int rc = cs3d_ensure_gs(ctx);
    if (rc != CUDA_SAM3D_E_OK) return rc;

#if defined(_OPENMP)
    int nthr = omp_get_max_threads(); if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif
    int N = slat->n;
    int G = sam3d_cpu_gs_decoder_num_gaussians(ctx->cpu_gs);
    int total = N * G;

    float *out_feats = sam3d_cpu_gs_decoder_transformer(ctx->cpu_gs,
                                                        slat->coords,
                                                        slat->feats, N, nthr);
    if (!out_feats) return CUDA_SAM3D_E_LOAD;

    float *xyz = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *dc  = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *scl = (float *)malloc((size_t)total * 3 * sizeof(float));
    float *rot = (float *)malloc((size_t)total * 4 * sizeof(float));
    float *op  = (float *)malloc((size_t)total     * sizeof(float));
    float *ply = (float *)malloc((size_t)total * CUDA_SAM3D_GS_STRIDE * sizeof(float));
    if (!xyz || !dc || !scl || !rot || !op || !ply) {
        free(xyz); free(dc); free(scl); free(rot); free(op); free(ply);
        free(out_feats);
        return CUDA_SAM3D_E_LOAD;
    }
    sam3d_cpu_gs_decoder_to_representation(ctx->cpu_gs, slat->coords,
                                           out_feats, N,
                                           xyz, dc, scl, rot, op);
    free(out_feats);
    sam3d_cpu_gs_decoder_pack_ply(ctx->cpu_gs, xyz, dc, scl, rot, op,
                                  total, CUDA_SAM3D_GS_STRIDE, ply);
    free(xyz); free(dc); free(scl); free(rot); free(op);
    return cs3d_adopt_gaussians(ctx, ply, total);
}

/* ===== read-back stubs ===== */

int cuda_sam3d_get_dinov2_tokens(cuda_sam3d_ctx *ctx, float *out,
                                 int *out_n, int *out_c) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_2d *src = cs3d_active_dinov2(ctx);
    if (!src->data) return CUDA_SAM3D_E_NO_INPUT;
    if (out_n) *out_n = src->n;
    if (out_c) *out_c = src->c;
    if (out) memcpy(out, src->data, (size_t)src->n * src->c * sizeof(float));
    return CUDA_SAM3D_E_OK;
}
int cuda_sam3d_get_cond_tokens(cuda_sam3d_ctx *ctx, float *out,
                               int *out_n, int *out_c) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_2d *src = cs3d_active_cond(ctx);
    if (!src->data) return CUDA_SAM3D_E_NO_INPUT;
    if (out_n) *out_n = src->n;
    if (out_c) *out_c = src->c;
    if (out) memcpy(out, src->data, (size_t)src->n * src->c * sizeof(float));
    return CUDA_SAM3D_E_OK;
}
int cuda_sam3d_get_ss_latent(cuda_sam3d_ctx *ctx, float *out, int *out_dims) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_nd *src = cs3d_active_ss_latent(ctx);
    if (!src->data) return CUDA_SAM3D_E_NO_INPUT;
    size_t numel = 1;
    for (int i = 0; i < src->ndim; i++) numel *= (size_t)src->dims[i];
    if (out_dims) for (int i = 0; i < src->ndim; i++) out_dims[i] = src->dims[i];
    if (out) memcpy(out, src->data, numel * sizeof(float));
    return CUDA_SAM3D_E_OK;
}
int cuda_sam3d_get_occupancy(cuda_sam3d_ctx *ctx, float *out, int *out_dims) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_nd *src = cs3d_active_occupancy(ctx);
    if (!src->data) return CUDA_SAM3D_E_NO_INPUT;
    size_t numel = 1;
    for (int i = 0; i < src->ndim; i++) numel *= (size_t)src->dims[i];
    if (out_dims) for (int i = 0; i < src->ndim; i++) out_dims[i] = src->dims[i];
    if (out) memcpy(out, src->data, numel * sizeof(float));
    return CUDA_SAM3D_E_OK;
}
int cuda_sam3d_get_slat_tokens(cuda_sam3d_ctx *ctx, float *out_feats,
                               int32_t *out_coords, int *out_n, int *out_c) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    const cs3d_host_slat *src = ctx->ovr_slat.feats
                              ? &ctx->ovr_slat
                              : &ctx->slat_tokens;
    if (!src->feats) return CUDA_SAM3D_E_NO_INPUT;
    if (out_n) *out_n = src->n;
    if (out_c) *out_c = src->c;
    if (out_feats)
        memcpy(out_feats, src->feats, (size_t)src->n * src->c * sizeof(float));
    if (out_coords)
        memcpy(out_coords, src->coords, (size_t)src->n * 4 * sizeof(int32_t));
    return CUDA_SAM3D_E_OK;
}
int cuda_sam3d_get_gaussians(cuda_sam3d_ctx *ctx, float *out, int *out_n) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    if (!ctx->gaussians.data) return CUDA_SAM3D_E_NO_INPUT;
    if (out_n) *out_n = ctx->gaussians.n;
    if (out)
        memcpy(out, ctx->gaussians.data,
               (size_t)ctx->gaussians.n * CUDA_SAM3D_GS_STRIDE * sizeof(float));
    return CUDA_SAM3D_E_OK;
}

/* ===== debug overrides — host-side store, consumed when the
 * downstream stage's kernel is wired up. ===== */

static int cs3d_set_2d(cs3d_host_2d *dst, const float *src, int n, int c) {
    cs3d_free_2d(dst);
    if (!src || n <= 0 || c <= 0) return CUDA_SAM3D_E_INVAL;
    size_t bytes = (size_t)n * c * sizeof(float);
    dst->data = (float *)malloc(bytes);
    if (!dst->data) return CUDA_SAM3D_E_LOAD;
    memcpy(dst->data, src, bytes);
    dst->n = n; dst->c = c;
    return CUDA_SAM3D_E_OK;
}

static int cs3d_set_nd(cs3d_host_nd *dst, const float *src,
                        const int *dims, int ndim) {
    cs3d_free_nd(dst);
    if (!src || !dims || ndim <= 0 || ndim > 4) return CUDA_SAM3D_E_INVAL;
    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) return CUDA_SAM3D_E_INVAL;
        numel *= (size_t)dims[i];
        dst->dims[i] = dims[i];
    }
    dst->ndim = ndim;
    dst->data = (float *)malloc(numel * sizeof(float));
    if (!dst->data) return CUDA_SAM3D_E_LOAD;
    memcpy(dst->data, src, numel * sizeof(float));
    return CUDA_SAM3D_E_OK;
}

int cuda_sam3d_debug_override_dinov2(cuda_sam3d_ctx *ctx, const float *tokens,
                                     int n, int c) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    return cs3d_set_2d(&ctx->ovr_dinov2, tokens, n, c);
}
int cuda_sam3d_debug_override_cond(cuda_sam3d_ctx *ctx, const float *tokens,
                                   int n, int c) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    return cs3d_set_2d(&ctx->ovr_cond, tokens, n, c);
}
int cuda_sam3d_debug_override_ss_latent(cuda_sam3d_ctx *ctx, const float *latent,
                                        const int *dims) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    return cs3d_set_nd(&ctx->ovr_ss_latent, latent, dims, 4);
}
int cuda_sam3d_debug_override_occupancy(cuda_sam3d_ctx *ctx, const float *occ,
                                        const int *dims) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    return cs3d_set_nd(&ctx->ovr_occupancy, occ, dims, 3);
}
int cuda_sam3d_debug_override_slat(cuda_sam3d_ctx *ctx, const float *feats,
                                   const int32_t *coords, int n, int c) {
    if (!ctx) return CUDA_SAM3D_E_INVAL;
    cs3d_free_slat(&ctx->ovr_slat);
    if (!feats || !coords || n <= 0 || c <= 0) return CUDA_SAM3D_E_INVAL;
    size_t f_bytes = (size_t)n * c * sizeof(float);
    size_t k_bytes = (size_t)n * 4 * sizeof(int32_t);
    ctx->ovr_slat.feats  = (float *)malloc(f_bytes);
    ctx->ovr_slat.coords = (int32_t *)malloc(k_bytes);
    if (!ctx->ovr_slat.feats || !ctx->ovr_slat.coords) {
        cs3d_free_slat(&ctx->ovr_slat);
        return CUDA_SAM3D_E_LOAD;
    }
    memcpy(ctx->ovr_slat.feats, feats, f_bytes);
    memcpy(ctx->ovr_slat.coords, coords, k_bytes);
    ctx->ovr_slat.n = n; ctx->ovr_slat.c = c;
    return CUDA_SAM3D_E_OK;
}
