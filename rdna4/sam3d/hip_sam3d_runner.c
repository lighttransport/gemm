/* HIP/ROCm runner for FAIR's SAM 3D Objects — Phase 0 scaffold.
 *
 * What's wired right now:
 *   - rocew runtime + HIPRTC compile bring-up (sentinel kernel).
 *   - Per-module safetensors directory resolution (explicit override or
 *     pipeline_yaml sibling fallback).
 *   - Public API stubs: every hip_sam3d_run_* and hip_sam3d_get_* is
 *     callable; each returns HIP_SAM3D_E_NOT_IMPLEMENTED until its
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

#include "hip_sam3d_runner.h"
#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include "hip_sam3d_kernels.h"

#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#define HIP_SAM3D_DINOV2_GPU_IMPLEMENTATION
#include "hip_sam3d_dinov2_gpu.h"
#include "hip_sam3d_dinov2_forward.h"
#define HIP_SAM3D_PPE_GPU_IMPLEMENTATION
#include "hip_sam3d_ppe_gpu.h"
#include "hip_sam3d_ppe_forward.h"
#define HIP_SAM3D_FUSER_GPU_IMPLEMENTATION
#include "hip_sam3d_fuser_gpu.h"
#include "hip_sam3d_fuser_forward.h"
#define HIP_SAM3D_SSDIT_GPU_IMPLEMENTATION
#include "hip_sam3d_ssdit_gpu.h"
#undef  HIP_SAM3D_SSDIT_GPU_IMPLEMENTATION
#define HIP_SAM3D_SSDIT_FORWARD_IMPLEMENTATION
#include "hip_sam3d_ssdit_forward.h"
#undef  HIP_SAM3D_SSDIT_FORWARD_IMPLEMENTATION
#define HIP_SAM3D_SSDIT_OUTER_IMPLEMENTATION
#include "hip_sam3d_ssdit_outer.h"
#undef  HIP_SAM3D_SSDIT_OUTER_IMPLEMENTATION
#define HIP_SAM3D_SS_DECODER_GPU_IMPLEMENTATION
#include "hip_sam3d_ss_decoder_gpu.h"
#undef  HIP_SAM3D_SS_DECODER_GPU_IMPLEMENTATION
#define HIP_SAM3D_SS_DECODER_FORWARD_IMPLEMENTATION
#include "hip_sam3d_ss_decoder_forward.h"
#undef  HIP_SAM3D_SS_DECODER_FORWARD_IMPLEMENTATION
#define HIP_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION
#include "hip_sam3d_slat_dit_gpu.h"
#undef  HIP_SAM3D_SLAT_DIT_GPU_IMPLEMENTATION
#define HIP_SAM3D_SLAT_DIT_FORWARD_IMPLEMENTATION
#include "hip_sam3d_slat_dit_forward.h"
#undef  HIP_SAM3D_SLAT_DIT_FORWARD_IMPLEMENTATION
#include "../../common/sam3d_shortcut_solver.h"
#include "../../common/sam3d_gs_decoder.h"
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

typedef struct {
    hipDeviceptr_t norm1_w, norm1_b;
    hipDeviceptr_t conv1_w, conv1_b;
    hipDeviceptr_t conv2_w, conv2_b;
    hipDeviceptr_t emb_w, emb_b;
    hipDeviceptr_t skip_w, skip_b;
    int C_in, C_out;
    int updown, has_skip;
} cs3d_slat_io_block_gpu;

typedef struct {
    cs3d_slat_io_block_gpu input[2];
    cs3d_slat_io_block_gpu output[2];
    hipDeviceptr_t input_w, input_b;
    int input_in, input_out;
    hipDeviceptr_t out_w, out_b;
    int out_in, out_out;
    size_t total_bytes;
    int loaded;
} cs3d_slat_io_gpu;

typedef struct {
    hipFunction_t gemm, down, up, idx, ln, silu, conv, modln, resadd;
    int loaded;
} cs3d_slat_io_fns;

typedef struct {
    hipDeviceptr_t d_in_coords, d_in_feats;
    hipDeviceptr_t d_coords, d_base;
    hipDeviceptr_t d_counts, d_outN;
    hipDeviceptr_t d_index;
    hipDeviceptr_t d_t, d_t_silu, d_emb;
    hipDeviceptr_t d_skip, d_h1, d_h2, d_h3, d_h4;
    size_t cap_in_coords, cap_in_feats;
    size_t cap_coords, cap_base;
    size_t cap_counts, cap_outN;
    size_t cap_index;
    size_t cap_t, cap_t_silu, cap_emb;
    size_t cap_skip, cap_h1, cap_h2, cap_h3, cap_h4;
} cs3d_slat_io_ws;

typedef struct {
    hipDeviceptr_t qkv_w, qkv_b;
    hipDeviceptr_t out_w, out_b;
    hipDeviceptr_t fc1_w, fc1_b;
    hipDeviceptr_t fc2_w, fc2_b;
} cs3d_gs_mlp_block_gpu;

typedef struct {
    hipDeviceptr_t input_w, input_b;
    hipDeviceptr_t out_w, out_b;
    cs3d_gs_mlp_block_gpu *mlp_blocks;
    int input_in, input_out;
    int out_in, out_out;
    int n_mlp_blocks, mlp_dim, mlp_hidden;
    size_t total_bytes;
    int loaded;
} cs3d_gs_head_gpu;

typedef struct { int64_t key; int idx; } cs3d_gs_win_kv;
typedef struct { int start; int len; } cs3d_gs_win_run;

typedef struct {
    hipDeviceptr_t d_coords, d_in, d_h, d_ln, d_out;
    hipDeviceptr_t d_qkv, d_fwd, d_attn, d_q, d_k, d_v, d_win, d_mlp;
    size_t cap_coords, cap_in, cap_h, cap_ln, cap_out;
    size_t cap_qkv, cap_fwd, cap_attn, cap_q, cap_k, cap_v, cap_win, cap_mlp;
} cs3d_gs_head_ws;

struct hip_sam3d_ctx {
    hip_sam3d_config cfg;
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
    hipDeviceptr_t       d_dinov2_tokens;   /* device mirror, [n_tok × dim] f32 */

    /* Phase 1b GPU DINOv2 forward — lazy-init on first run_dinov2 call. */
    cs3d_dinov2_gpu      gpu_dinov2;
    int                  gpu_dinov2_loaded;
    cs3d_dinov2_fns      gpu_dinov2_fns;
    cs3d_dinov2_block_ws gpu_dinov2_ws;
    int                  gpu_dinov2_ws_alloced;

    sam3d_cpu_fuser  *cpu_fuser;
    cs3d_host_2d      cond_tokens;
    hipDeviceptr_t       d_cond_tokens;     /* device mirror, [n_tok × D_out] f32 */

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
    hipDeviceptr_t       d_ss_latent;       /* device mirror */

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
    hipDeviceptr_t       d_occupancy;       /* device mirror */

    /* Phase 4b GPU SS-VAE decoder. */
    cs3d_ssdec_gpu    gpu_ssdec;
    int               gpu_ssdec_loaded;
    cs3d_ssdec_fns    gpu_ssdec_fns;
    cs3d_ssdec_ws     gpu_ssdec_ws;
    int               gpu_ssdec_ws_alloced;

    sam3d_cpu_slat_dit *cpu_slat_dit;
    cs3d_host_slat      slat_tokens;     /* (coords[N,4], feats[N,out_ch]) f32 */
    hipDeviceptr_t         d_slat_feats;    /* device mirror */
    hipDeviceptr_t         d_slat_coords;   /* device mirror */
    cs3d_slatdit_gpu       gpu_slatdit;
    int                    gpu_slatdit_loaded;
    cs3d_slatdit_fns       gpu_slatdit_fns;
    cs3d_slatdit_block_ws  gpu_slatdit_ws;
    int                    gpu_slatdit_ws_alloced;
    int                    gpu_slatdit_ws_n;
    int                    gpu_slatdit_ws_nc;
    hipDeviceptr_t            d_slat_hook_x;
    hipDeviceptr_t            d_slat_hook_coords;
    hipDeviceptr_t            d_slat_hook_t_emb;
    size_t                 slat_hook_x_bytes;
    size_t                 slat_hook_coords_bytes;
    size_t                 slat_hook_t_emb_bytes;
    int32_t               *slat_hook_coords_cache;
    size_t                 slat_hook_coords_cache_bytes;
    cs3d_slat_io_gpu       gpu_slat_io;
    cs3d_slat_io_fns       gpu_slat_io_fns;
    cs3d_slat_io_ws        gpu_slat_io_ws;

    sam3d_cpu_gs_decoder *cpu_gs;
    cs3d_host_2d          gaussians;      /* [N*G, 17] PLY-layout f32 */
    hipDeviceptr_t           d_gaussians;    /* device mirror */
    cs3d_gs_head_gpu      gpu_gs_head;
    cs3d_gs_head_ws       gpu_gs_head_ws;
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
static char *cs3d_resolve_safetensors_dir(const hip_sam3d_config *cfg) {
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

static void cs3d_slat_io_block_gpu_free(cs3d_slat_io_block_gpu *b)
{
    if (!b) return;
    if (b->norm1_w) hipFree(b->norm1_w);
    if (b->norm1_b) hipFree(b->norm1_b);
    if (b->conv1_w) hipFree(b->conv1_w);
    if (b->conv1_b) hipFree(b->conv1_b);
    if (b->conv2_w) hipFree(b->conv2_w);
    if (b->conv2_b) hipFree(b->conv2_b);
    if (b->emb_w)   hipFree(b->emb_w);
    if (b->emb_b)   hipFree(b->emb_b);
    if (b->skip_w)  hipFree(b->skip_w);
    if (b->skip_b)  hipFree(b->skip_b);
    memset(b, 0, sizeof(*b));
}

static void cs3d_slat_io_gpu_free(cs3d_slat_io_gpu *g)
{
    if (!g) return;
    for (int i = 0; i < 2; i++) {
        cs3d_slat_io_block_gpu_free(&g->input[i]);
        cs3d_slat_io_block_gpu_free(&g->output[i]);
    }
    if (g->input_w) hipFree(g->input_w);
    if (g->input_b) hipFree(g->input_b);
    if (g->out_w) hipFree(g->out_w);
    if (g->out_b) hipFree(g->out_b);
    memset(g, 0, sizeof(*g));
}

static void cs3d_slat_io_ws_free(cs3d_slat_io_ws *w)
{
    if (!w) return;
    if (w->d_in_coords) hipFree(w->d_in_coords);
    if (w->d_in_feats)  hipFree(w->d_in_feats);
    if (w->d_coords)    hipFree(w->d_coords);
    if (w->d_base)      hipFree(w->d_base);
    if (w->d_counts)    hipFree(w->d_counts);
    if (w->d_outN)      hipFree(w->d_outN);
    if (w->d_index)     hipFree(w->d_index);
    if (w->d_t)         hipFree(w->d_t);
    if (w->d_t_silu)    hipFree(w->d_t_silu);
    if (w->d_emb)       hipFree(w->d_emb);
    if (w->d_skip)      hipFree(w->d_skip);
    if (w->d_h1)        hipFree(w->d_h1);
    if (w->d_h2)        hipFree(w->d_h2);
    if (w->d_h3)        hipFree(w->d_h3);
    if (w->d_h4)        hipFree(w->d_h4);
    memset(w, 0, sizeof(*w));
}

static void cs3d_gs_head_gpu_free(cs3d_gs_head_gpu *g)
{
    if (!g) return;
    if (g->input_w) hipFree(g->input_w);
    if (g->input_b) hipFree(g->input_b);
    if (g->out_w) hipFree(g->out_w);
    if (g->out_b) hipFree(g->out_b);
    if (g->mlp_blocks) {
        for (int i = 0; i < g->n_mlp_blocks; i++) {
            if (g->mlp_blocks[i].qkv_w) hipFree(g->mlp_blocks[i].qkv_w);
            if (g->mlp_blocks[i].qkv_b) hipFree(g->mlp_blocks[i].qkv_b);
            if (g->mlp_blocks[i].out_w) hipFree(g->mlp_blocks[i].out_w);
            if (g->mlp_blocks[i].out_b) hipFree(g->mlp_blocks[i].out_b);
            if (g->mlp_blocks[i].fc1_w) hipFree(g->mlp_blocks[i].fc1_w);
            if (g->mlp_blocks[i].fc1_b) hipFree(g->mlp_blocks[i].fc1_b);
            if (g->mlp_blocks[i].fc2_w) hipFree(g->mlp_blocks[i].fc2_w);
            if (g->mlp_blocks[i].fc2_b) hipFree(g->mlp_blocks[i].fc2_b);
        }
        free(g->mlp_blocks);
    }
    memset(g, 0, sizeof(*g));
}

static void cs3d_gs_head_ws_free(cs3d_gs_head_ws *w)
{
    if (!w) return;
    if (w->d_coords) hipFree(w->d_coords);
    if (w->d_in)     hipFree(w->d_in);
    if (w->d_h)      hipFree(w->d_h);
    if (w->d_ln)     hipFree(w->d_ln);
    if (w->d_out)    hipFree(w->d_out);
    if (w->d_qkv)    hipFree(w->d_qkv);
    if (w->d_fwd)    hipFree(w->d_fwd);
    if (w->d_attn)   hipFree(w->d_attn);
    if (w->d_q)      hipFree(w->d_q);
    if (w->d_k)      hipFree(w->d_k);
    if (w->d_v)      hipFree(w->d_v);
    if (w->d_win)    hipFree(w->d_win);
    if (w->d_mlp)    hipFree(w->d_mlp);
    memset(w, 0, sizeof(*w));
}

/* ===== public API ===== */

hip_sam3d_ctx *hip_sam3d_create(const hip_sam3d_config *cfg)
{
    if (!cfg) return NULL;

    hip_sam3d_ctx *c = (hip_sam3d_ctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cfg = *cfg;
    c->device_id = cfg->device_ordinal;
    if (c->cfg.ss_steps   <= 0) c->cfg.ss_steps   = 2;
    if (c->cfg.slat_steps <= 0) c->cfg.slat_steps = 12;
    if (c->cfg.cfg_scale  <= 0) c->cfg.cfg_scale  = 2.0f;

    c->safetensors_dir_resolved = cs3d_resolve_safetensors_dir(cfg);
    if (!c->safetensors_dir_resolved) {
        fprintf(stderr, "hip_sam3d: need safetensors_dir or pipeline_yaml\n");
        free(c); return NULL;
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_sam3d: rocew init failed (no HIP/ROCm runtime?)\n");
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    if (hipInit(0) != hipSuccess) {
        fprintf(stderr, "hip_sam3d: hipInit failed\n");
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    hipDevice_t dev;
    if ((dev = (c->device_id), hipSetDevice(c->device_id)) != hipSuccess) {
        fprintf(stderr, "hip_sam3d: hipSetDevice(%d) failed\n", c->device_id);
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    hipCtx_t cu_ctx;
    if (hipCtxCreate(&cu_ctx, 0, dev) != hipSuccess) {
        fprintf(stderr, "hip_sam3d: hipCtxCreate failed\n");
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }

    c->sm = hip_compile_kernels(&c->mod, c->device_id, hip_sam3d_kernel_src,
                               "sam3d_kernels.cu", c->cfg.verbose,
                               "hip_sam3d");
    if (c->sm < 0) {
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    if (hipModuleGetFunction(&c->fn_sentinel, c->mod, "sam3d_sentinel") != hipSuccess) {
        fprintf(stderr, "hip_sam3d: missing sam3d_sentinel\n");
        free(c->safetensors_dir_resolved); free(c);
        return NULL;
    }
    c->compiled = 1;

    if (c->cfg.verbose >= 1) {
        fprintf(stderr, "hip_sam3d: created (sm_%d, safetensors_dir=%s)\n",
                c->sm, c->safetensors_dir_resolved);
    }
    return c;
}

void hip_sam3d_destroy(hip_sam3d_ctx *ctx)
{
    if (!ctx) return;
    free(ctx->img_rgba); free(ctx->mask_u8); free(ctx->pmap_f32);
    cs3d_free_2d(&ctx->ovr_dinov2);
    cs3d_free_2d(&ctx->ovr_cond);
    cs3d_free_nd(&ctx->ovr_ss_latent);
    cs3d_free_nd(&ctx->ovr_occupancy);
    cs3d_free_slat(&ctx->ovr_slat);
    cs3d_free_2d(&ctx->dinov2_tokens);
    if (ctx->d_dinov2_tokens) hipFree(ctx->d_dinov2_tokens);
    if (ctx->cpu_dinov2) sam3d_cpu_dinov2_free(ctx->cpu_dinov2);
    if (ctx->gpu_dinov2_ws_alloced) cs3d_dinov2_block_ws_free(&ctx->gpu_dinov2_ws);
    if (ctx->gpu_dinov2_loaded) cs3d_dinov2_gpu_free(&ctx->gpu_dinov2);
    cs3d_free_2d(&ctx->cond_tokens);
    if (ctx->d_cond_tokens) hipFree(ctx->d_cond_tokens);
    if (ctx->gpu_ppe_ws_alloced)   cs3d_ppe_ws_free(&ctx->gpu_ppe_ws);
    if (ctx->gpu_ppe_loaded)       cs3d_ppe_gpu_free(&ctx->gpu_ppe);
    if (ctx->gpu_fuser_ws_alloced) cs3d_fuser_ws_free(&ctx->gpu_fuser_ws);
    if (ctx->gpu_fuser_loaded)     cs3d_fuser_gpu_free(&ctx->gpu_fuser);
    if (ctx->cpu_fuser) sam3d_cpu_fuser_free(ctx->cpu_fuser);
    cs3d_free_nd(&ctx->ss_latent);
    if (ctx->d_ss_latent) hipFree(ctx->d_ss_latent);
    if (ctx->gpu_ssdit_ws_alloced) {
        cs3d_ssdit_outer_ws_free(&ctx->gpu_ssdit_outer_ws);
        cs3d_ssdit_block_ws_free(&ctx->gpu_ssdit_block_ws);
    }
    if (ctx->gpu_ssdit_loaded) cs3d_ssdit_gpu_free(&ctx->gpu_ssdit);
    if (ctx->cpu_ss_dit) sam3d_cpu_ss_dit_free(ctx->cpu_ss_dit);
    cs3d_free_nd(&ctx->occupancy);
    if (ctx->d_occupancy) hipFree(ctx->d_occupancy);
    if (ctx->gpu_ssdec_ws_alloced) cs3d_ssdec_ws_free(&ctx->gpu_ssdec_ws);
    if (ctx->gpu_ssdec_loaded)     cs3d_ssdec_gpu_free(&ctx->gpu_ssdec);
    if (ctx->cpu_ss_dec) sam3d_cpu_ss_dec_free(ctx->cpu_ss_dec);
    cs3d_free_slat(&ctx->slat_tokens);
    if (ctx->d_slat_feats)  hipFree(ctx->d_slat_feats);
    if (ctx->d_slat_coords) hipFree(ctx->d_slat_coords);
    if (ctx->d_slat_hook_x)     hipFree(ctx->d_slat_hook_x);
    if (ctx->d_slat_hook_coords) hipFree(ctx->d_slat_hook_coords);
    if (ctx->d_slat_hook_t_emb) hipFree(ctx->d_slat_hook_t_emb);
    free(ctx->slat_hook_coords_cache);
    if (ctx->gpu_slatdit_ws_alloced) cs3d_slatdit_block_ws_free(&ctx->gpu_slatdit_ws);
    if (ctx->gpu_slatdit_loaded)     cs3d_slatdit_gpu_free(&ctx->gpu_slatdit);
    if (ctx->gpu_slat_io.loaded)     cs3d_slat_io_gpu_free(&ctx->gpu_slat_io);
    cs3d_slat_io_ws_free(&ctx->gpu_slat_io_ws);
    if (ctx->cpu_slat_dit) sam3d_cpu_slat_dit_free(ctx->cpu_slat_dit);
    cs3d_free_2d(&ctx->gaussians);
    if (ctx->d_gaussians) hipFree(ctx->d_gaussians);
    if (ctx->gpu_gs_head.loaded) cs3d_gs_head_gpu_free(&ctx->gpu_gs_head);
    cs3d_gs_head_ws_free(&ctx->gpu_gs_head_ws);
    if (ctx->cpu_gs) sam3d_cpu_gs_decoder_free(ctx->cpu_gs);
    if (ctx->compiled) hipModuleUnload(ctx->mod);
    free(ctx->safetensors_dir_resolved);
    free(ctx);
}

/* ===== inputs ===== */

int hip_sam3d_set_image_rgba(hip_sam3d_ctx *ctx, const uint8_t *rgba,
                              int width, int height) {
    if (!ctx || !rgba || width <= 0 || height <= 0) return HIP_SAM3D_E_INVAL;
    free(ctx->img_rgba);
    size_t n = (size_t)width * height * 4;
    ctx->img_rgba = (uint8_t *)malloc(n);
    if (!ctx->img_rgba) return HIP_SAM3D_E_LOAD;
    memcpy(ctx->img_rgba, rgba, n);
    ctx->img_w = width; ctx->img_h = height;
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_set_mask(hip_sam3d_ctx *ctx, const uint8_t *mask,
                        int width, int height) {
    if (!ctx || !mask || width <= 0 || height <= 0) return HIP_SAM3D_E_INVAL;
    free(ctx->mask_u8);
    size_t n = (size_t)width * height;
    ctx->mask_u8 = (uint8_t *)malloc(n);
    if (!ctx->mask_u8) return HIP_SAM3D_E_LOAD;
    memcpy(ctx->mask_u8, mask, n);
    ctx->msk_w = width; ctx->msk_h = height;
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_set_pointmap(hip_sam3d_ctx *ctx, const float *pmap,
                            int width, int height) {
    if (!ctx || !pmap || width <= 0 || height <= 0) return HIP_SAM3D_E_INVAL;
    free(ctx->pmap_f32);
    size_t n = (size_t)width * height * 3;
    ctx->pmap_f32 = (float *)malloc(n * sizeof(float));
    if (!ctx->pmap_f32) return HIP_SAM3D_E_LOAD;
    memcpy(ctx->pmap_f32, pmap, n * sizeof(float));
    ctx->pmap_w = width; ctx->pmap_h = height;
    return HIP_SAM3D_E_OK;
}

/* ===== stage stubs ===== */

/* Adopt a freshly-computed [n × c] f32 token block as the dinov2
 * stage output. Frees any prior host/device buffers. Takes ownership
 * of `feats` (must be malloc'd). */
static int cs3d_adopt_dinov2_tokens(hip_sam3d_ctx *ctx, float *feats,
                                    int n, int c)
{
    cs3d_free_2d(&ctx->dinov2_tokens);
    if (ctx->d_dinov2_tokens) {
        hipFree(ctx->d_dinov2_tokens);
        ctx->d_dinov2_tokens = 0;
    }
    if (!feats || n <= 0 || c <= 0) return HIP_SAM3D_E_INVAL;

    size_t bytes = (size_t)n * c * sizeof(float);
    ctx->d_dinov2_tokens = hip_upload_raw(feats, bytes);
    if (!ctx->d_dinov2_tokens) { free(feats); return HIP_SAM3D_E_LOAD; }
    ctx->dinov2_tokens.data = feats;
    ctx->dinov2_tokens.n    = n;
    ctx->dinov2_tokens.c    = c;
    return HIP_SAM3D_E_OK;
}

/* Phase 1b.8: run one branch (image or mask) through the GPU forward.
 * chw_norm is [3, S, S] f32, normalized. Returns malloc'd [n_tok × dim]
 * with the n_register tokens removed (matching CPU numerics). */
static float *cs3d_gpu_dinov2_encode_branch(hip_sam3d_ctx *ctx,
                                            const float *chw_norm)
{
    cs3d_dinov2_gpu *g = &ctx->gpu_dinov2;
    size_t n_img = (size_t)3 * g->image_size * g->image_size;
    size_t n_out = (size_t)g->n_tokens * g->dim;

    hipDeviceptr_t d_img = hip_upload_raw(chw_norm, n_img * sizeof(float));
    if (!d_img) return NULL;
    hipDeviceptr_t d_out = 0;
    if (hipMalloc(&d_out, n_out * sizeof(float)) != hipSuccess) {
        hipFree(d_img); return NULL;
    }
    int rc = cs3d_dinov2_gpu_forward(&ctx->gpu_dinov2_fns, &ctx->gpu_dinov2_ws,
                                     g, d_img, d_out);
    hipFree(d_img);
    if (rc < 0) { hipFree(d_out); return NULL; }

    float *host = (float *)malloc(n_out * sizeof(float));
    if (!host) { hipFree(d_out); return NULL; }
    if (hipMemcpyDtoH(host, d_out, n_out * sizeof(float)) != hipSuccess) {
        free(host); hipFree(d_out); return NULL;
    }
    hipFree(d_out);

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

int hip_sam3d_run_dinov2(hip_sam3d_ctx *ctx) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    if (!ctx->img_rgba) return HIP_SAM3D_E_NO_INPUT;

    /* CPU model kept for preprocessing (image_mean/std + bilinear-resize
     * + ImageNet-normalize); the encode itself runs on the GPU. */
    if (!ctx->cpu_dinov2) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_dinov2.safetensors",
                 ctx->safetensors_dir_resolved);
        ctx->cpu_dinov2 = sam3d_cpu_dinov2_load(path);
        if (!ctx->cpu_dinov2) {
            fprintf(stderr, "hip_sam3d: dinov2 load failed (%s)\n", path);
            return HIP_SAM3D_E_LOAD;
        }
    }
    if (!ctx->gpu_dinov2_loaded) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/sam3d_dinov2.safetensors",
                 ctx->safetensors_dir_resolved);
        if (cs3d_dinov2_gpu_load(&ctx->gpu_dinov2, path, ctx->cfg.verbose) < 0) {
            fprintf(stderr, "hip_sam3d: gpu dinov2 weight upload failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        ctx->gpu_dinov2_loaded = 1;
        if (cs3d_dinov2_fns_lookup(&ctx->gpu_dinov2_fns, ctx->mod) < 0) {
            fprintf(stderr, "hip_sam3d: gpu dinov2 fns lookup failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        if (cs3d_dinov2_block_ws_alloc(&ctx->gpu_dinov2_ws,
                                       ctx->gpu_dinov2.n_tokens,
                                       ctx->gpu_dinov2.dim,
                                       ctx->gpu_dinov2.ffn_hidden) < 0) {
            fprintf(stderr, "hip_sam3d: gpu dinov2 ws alloc failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        ctx->gpu_dinov2_ws_alloced = 1;
    }

    int dim   = ctx->gpu_dinov2.dim;
    int n_reg = ctx->gpu_dinov2.n_register;
    int n_per = ctx->gpu_dinov2.n_tokens - n_reg;  /* CLS + patches */

    float *chw_img = sam3d_cpu_dinov2_preprocess_rgb(
        ctx->cpu_dinov2, ctx->img_rgba, ctx->img_w, ctx->img_h);
    if (!chw_img) return HIP_SAM3D_E_LOAD;
    float *feat_img = cs3d_gpu_dinov2_encode_branch(ctx, chw_img);
    free(chw_img);
    if (!feat_img) {
        fprintf(stderr, "hip_sam3d: gpu dinov2 image branch failed\n");
        return HIP_SAM3D_E_LOAD;
    }

    float *feat_msk = NULL;
    if (ctx->mask_u8) {
        float *chw_msk = sam3d_cpu_dinov2_preprocess_mask(
            ctx->cpu_dinov2, ctx->mask_u8, ctx->msk_w, ctx->msk_h);
        if (!chw_msk) { free(feat_img); return HIP_SAM3D_E_LOAD; }
        feat_msk = cs3d_gpu_dinov2_encode_branch(ctx, chw_msk);
        free(chw_msk);
        if (!feat_msk) {
            fprintf(stderr, "hip_sam3d: gpu dinov2 mask branch failed\n");
            free(feat_img);
            return HIP_SAM3D_E_LOAD;
        }
    }

    int n_branches = feat_msk ? 2 : 1;
    int n_total    = n_branches * n_per;
    float *feats   = (float *)malloc((size_t)n_total * dim * sizeof(float));
    if (!feats) { free(feat_img); free(feat_msk); return HIP_SAM3D_E_LOAD; }
    memcpy(feats, feat_img, (size_t)n_per * dim * sizeof(float));
    if (feat_msk) {
        memcpy(feats + (size_t)n_per * dim, feat_msk,
               (size_t)n_per * dim * sizeof(float));
    }
    free(feat_img); free(feat_msk);
    return cs3d_adopt_dinov2_tokens(ctx, feats, n_total, dim);
}
/* Active dinov2 source: prefer override (verify_*.c injects ref data
 * via hip_sam3d_debug_override_dinov2) so a single verify can isolate
 * cond_fuser drift from upstream dinov2 drift. */
static const cs3d_host_2d *cs3d_active_dinov2(const hip_sam3d_ctx *ctx)
{
    return ctx->ovr_dinov2.data ? &ctx->ovr_dinov2 : &ctx->dinov2_tokens;
}

static int cs3d_adopt_cond_tokens(hip_sam3d_ctx *ctx, float *feats,
                                  int n, int c)
{
    cs3d_free_2d(&ctx->cond_tokens);
    if (ctx->d_cond_tokens) {
        hipFree(ctx->d_cond_tokens);
        ctx->d_cond_tokens = 0;
    }
    if (!feats || n <= 0 || c <= 0) return HIP_SAM3D_E_INVAL;
    size_t bytes = (size_t)n * c * sizeof(float);
    ctx->d_cond_tokens = hip_upload_raw(feats, bytes);
    if (!ctx->d_cond_tokens) { free(feats); return HIP_SAM3D_E_LOAD; }
    ctx->cond_tokens.data = feats;
    ctx->cond_tokens.n    = n;
    ctx->cond_tokens.c    = c;
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_run_cond_fuser(hip_sam3d_ctx *ctx) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_2d *dino = cs3d_active_dinov2(ctx);
    if (!dino->data) {
        fprintf(stderr, "hip_sam3d: cond_fuser needs dinov2 tokens — "
                        "run dinov2 (or override) first\n");
        return HIP_SAM3D_E_NO_INPUT;
    }

    if (!ctx->cpu_fuser) {
        ctx->cpu_fuser = sam3d_cpu_fuser_load(ctx->safetensors_dir_resolved);
        if (!ctx->cpu_fuser) {
            fprintf(stderr, "hip_sam3d: fuser/PPE load failed\n");
            return HIP_SAM3D_E_LOAD;
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
        fprintf(stderr, "hip_sam3d: cond_fuser bad shape (branch=%d dim=%d Do=%d)\n",
                branch, Di_dino, Do);
        return HIP_SAM3D_E_INVAL;
    }

    /* Lazy-load PPE GPU. */
    if (have_ppe && !ctx->gpu_ppe_loaded) {
        if (cs3d_ppe_gpu_load(&ctx->gpu_ppe, ppe_m, ctx->cfg.verbose) < 0)
            return HIP_SAM3D_E_LOAD;
        ctx->gpu_ppe_loaded = 1;
        if (cs3d_ppe_fns_lookup(&ctx->gpu_ppe_fns, ctx->mod) < 0)
            return HIP_SAM3D_E_LOAD;
        if (cs3d_ppe_ws_alloc(&ctx->gpu_ppe_ws,
                              ctx->gpu_ppe.Np, ctx->gpu_ppe.P,
                              ctx->gpu_ppe.D,  ctx->gpu_ppe.ffn) < 0)
            return HIP_SAM3D_E_LOAD;
        ctx->gpu_ppe_ws_alloced = 1;
    }

    /* Lazy-load fuser GPU + workspace. */
    if (!ctx->gpu_fuser_loaded) {
        if (cs3d_fuser_gpu_load(&ctx->gpu_fuser, fus_m, ctx->cfg.verbose) < 0)
            return HIP_SAM3D_E_LOAD;
        ctx->gpu_fuser_loaded = 1;
        if (cs3d_fuser_fns_lookup(&ctx->gpu_fuser_fns, ctx->mod) < 0)
            return HIP_SAM3D_E_LOAD;
        int N_max = branch > Nwin ? branch : Nwin;
        int Di_max = 0, Dh_max = 0;
        for (int i = 0; i < ctx->gpu_fuser.n_modalities; i++) {
            if (ctx->gpu_fuser.projs[i].Di > Di_max) Di_max = ctx->gpu_fuser.projs[i].Di;
            if (ctx->gpu_fuser.projs[i].Dh > Dh_max) Dh_max = ctx->gpu_fuser.projs[i].Dh;
        }
        if (cs3d_fuser_ws_alloc(&ctx->gpu_fuser_ws, N_max, Di_max, Dh_max) < 0)
            return HIP_SAM3D_E_LOAD;
        ctx->gpu_fuser_ws_alloced = 1;
    }

    /* Run PPE on-device into d_ppe[Nwin, D_ppe]. */
    hipDeviceptr_t d_ppe = 0;
    if (have_ppe) {
        float *pmap_S = (float *)malloc((size_t)S * S * 3 * sizeof(float));
        if (!pmap_S) return HIP_SAM3D_E_LOAD;
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
        hipDeviceptr_t d_pmap = hip_upload_raw(pmap_S, (size_t)S * S * 3 * sizeof(float));
        free(pmap_S);
        if (!d_pmap ||
            hipMalloc(&d_ppe, (size_t)Nwin * D_ppe * sizeof(float)) != hipSuccess) {
            if (d_pmap) hipFree(d_pmap);
            return HIP_SAM3D_E_LOAD;
        }
        if (cs3d_ppe_forward(&ctx->gpu_ppe_fns, &ctx->gpu_ppe_ws,
                             &ctx->gpu_ppe.w, d_pmap, d_ppe,
                             ctx->gpu_ppe.n_heads, ctx->gpu_ppe.ln_eps) < 0) {
            hipFree(d_pmap); hipFree(d_ppe);
            return HIP_SAM3D_E_LOAD;
        }
        hipFree(d_pmap);
    }

    /* Upload dino tokens, allocate output cond buffer. */
    hipDeviceptr_t d_dino = hip_upload_raw(dino->data,
                                       (size_t)dino->n * Di_dino * sizeof(float));
    hipDeviceptr_t d_cond = 0;
    if (!d_dino ||
        hipMalloc(&d_cond, (size_t)n_total * Do * sizeof(float)) != hipSuccess) {
        if (d_dino) hipFree(d_dino);
        if (d_ppe)  hipFree(d_ppe);
        return HIP_SAM3D_E_LOAD;
    }

    /* Project each modality, writing into the right slice of d_cond.
     * idx_emb add is folded into the final w2 gemm via gemm_f32_bias. */
    hipDeviceptr_t pos_row = ctx->gpu_fuser.idx_emb_rows[SAM3D_FUSER_POS_FULL];
    {
        const cs3d_fuser_proj_w *p0 = &ctx->gpu_fuser.projs[SAM3D_FUSER_MOD_DINO_IMG];
        hipDeviceptr_t in_img = d_dino;
        hipDeviceptr_t out_img = d_cond + 0;
        if (cs3d_fuser_project_forward(&ctx->gpu_fuser_fns, &ctx->gpu_fuser_ws,
                                       p0, in_img, out_img, branch, pos_row, 1e-5f) < 0)
            goto fuser_fail;
    }
    {
        const cs3d_fuser_proj_w *p1 = &ctx->gpu_fuser.projs[SAM3D_FUSER_MOD_DINO_MSK];
        hipDeviceptr_t in_msk = d_dino + (size_t)branch * Di_dino * sizeof(float);
        hipDeviceptr_t out_msk = d_cond + (size_t)branch * Do * sizeof(float);
        if (cs3d_fuser_project_forward(&ctx->gpu_fuser_fns, &ctx->gpu_fuser_ws,
                                       p1, in_msk, out_msk, branch, pos_row, 1e-5f) < 0)
            goto fuser_fail;
    }
    if (have_ppe) {
        const cs3d_fuser_proj_w *p2 = &ctx->gpu_fuser.projs[SAM3D_FUSER_MOD_POINT];
        hipDeviceptr_t out_pt = d_cond + (size_t)2 * branch * Do * sizeof(float);
        if (cs3d_fuser_project_forward(&ctx->gpu_fuser_fns, &ctx->gpu_fuser_ws,
                                       p2, d_ppe, out_pt, Nwin, pos_row, 1e-5f) < 0)
            goto fuser_fail;
    }
    hipDeviceSynchronize();

    /* D2H mirror so verify_*.c and downstream stages can read host. */
    float *feats = (float *)malloc((size_t)n_total * Do * sizeof(float));
    if (!feats) goto fuser_fail;
    hipMemcpyDtoH(feats, d_cond, (size_t)n_total * Do * sizeof(float));
    if (d_dino) hipFree(d_dino);
    if (d_ppe)  hipFree(d_ppe);
    hipFree(d_cond);

    (void)nthr;
    return cs3d_adopt_cond_tokens(ctx, feats, n_total, Do);

fuser_fail:
    if (d_dino) hipFree(d_dino);
    if (d_ppe)  hipFree(d_ppe);
    if (d_cond) hipFree(d_cond);
    fprintf(stderr, "hip_sam3d: GPU fuser projection failed\n");
    return HIP_SAM3D_E_LOAD;
}
/* Active cond source: prefer override (verify_*.c injects ref data). */
static const cs3d_host_2d *cs3d_active_cond(const hip_sam3d_ctx *ctx)
{
    return ctx->ovr_cond.data ? &ctx->ovr_cond : &ctx->cond_tokens;
}

static int cs3d_ensure_ss_dit(hip_sam3d_ctx *ctx) {
    if (ctx->cpu_ss_dit) return HIP_SAM3D_E_OK;
    ctx->cpu_ss_dit = sam3d_cpu_ss_dit_load(ctx->safetensors_dir_resolved);
    if (!ctx->cpu_ss_dit) {
        fprintf(stderr, "hip_sam3d: ss_dit load failed (dir=%s)\n",
                ctx->safetensors_dir_resolved);
        return HIP_SAM3D_E_LOAD;
    }
    return HIP_SAM3D_E_OK;
}

static int cs3d_adopt_ss_latent(hip_sam3d_ctx *ctx, float *data,
                                const int dims[4])
{
    cs3d_free_nd(&ctx->ss_latent);
    if (ctx->d_ss_latent) { hipFree(ctx->d_ss_latent); ctx->d_ss_latent = 0; }
    if (!data) return HIP_SAM3D_E_INVAL;
    size_t numel = 1;
    for (int i = 0; i < 4; i++) {
        if (dims[i] <= 0) { free(data); return HIP_SAM3D_E_INVAL; }
        ctx->ss_latent.dims[i] = dims[i];
        numel *= (size_t)dims[i];
    }
    ctx->ss_latent.ndim = 4;
    ctx->ss_latent.data = data;
    ctx->d_ss_latent = hip_upload_raw(data, numel * sizeof(float));
    if (!ctx->d_ss_latent) {
        cs3d_free_nd(&ctx->ss_latent);
        return HIP_SAM3D_E_LOAD;
    }
    return HIP_SAM3D_E_OK;
}

static int cs3d_ensure_ssdit_gpu(hip_sam3d_ctx *ctx, int n_cond_min);

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

int hip_sam3d_run_ss_dit(hip_sam3d_ctx *ctx) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_2d *cond = cs3d_active_cond(ctx);
    if (!cond->data) {
        fprintf(stderr, "hip_sam3d: ss_dit needs cond tokens — "
                        "run cond_fuser (or override) first\n");
        return HIP_SAM3D_E_NO_INPUT;
    }
    int rc = cs3d_ensure_ssdit_gpu(ctx, cond->n);
    if (rc != HIP_SAM3D_E_OK) return rc;

    const sam3d_ss_flow_dit_model *m = sam3d_cpu_ss_dit_model(ctx->cpu_ss_dit);
    if (!m) return HIP_SAM3D_E_LOAD;

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
        fprintf(stderr, "hip_sam3d: ss_dit cond upload failed\n");
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
            fprintf(stderr, "hip_sam3d: ss_dit GPU forward failed (step %d)\n", s);
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
                fprintf(stderr, "hip_sam3d: ss_dit GPU uncond forward failed (step %d)\n", s);
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
                fprintf(stderr, "hip_sam3d: ss_dit cond restore failed\n");
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
                "hip_sam3d: ss_dit latent min=%.6g max=%.6g mean=%.6g d=%.3g rescale_t=3 reversed=0 cfg_steps=%d\n",
                mn, mx, sum / (8.0 * 4096.0), d, cfg_steps);
    }
    return cs3d_adopt_ss_latent(ctx, latent, dims);

oom:
    for (int i = 0; i < SAM3D_SS_DIT_N_LATENTS; i++) {
        free(lat[i]); free(vel[i]); free(vel_u[i]);
    }
    return HIP_SAM3D_E_LOAD;
}

/* Lazy-init GPU SS Flow DiT (weights upload + fns lookup + workspaces).
 * `n_cond_min` is the largest cond-token count we must support; the outer
 * workspace is grown on-demand if a later call exceeds the cap. */
static int cs3d_ensure_ssdit_gpu(hip_sam3d_ctx *ctx, int n_cond_min)
{
    int rc = cs3d_ensure_ss_dit(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    sam3d_ss_flow_dit_model *m = sam3d_cpu_ss_dit_model(ctx->cpu_ss_dit);
    if (!m) return HIP_SAM3D_E_LOAD;

    if (!ctx->gpu_ssdit_loaded) {
        if (cs3d_ssdit_gpu_load(&ctx->gpu_ssdit, m, ctx->cfg.verbose) < 0) {
            fprintf(stderr, "hip_sam3d: gpu ss_dit weight upload failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        ctx->gpu_ssdit_loaded = 1;
        if (cs3d_ssdit_outer_fns_lookup(&ctx->gpu_ssdit_fns, ctx->mod) < 0) {
            fprintf(stderr, "hip_sam3d: gpu ss_dit fns lookup failed\n");
            return HIP_SAM3D_E_LOAD;
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
            fprintf(stderr, "hip_sam3d: gpu ss_dit block_ws alloc failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        if (cs3d_ssdit_outer_ws_alloc(&ctx->gpu_ssdit_outer_ws, g, n_cond_min) < 0) {
            cs3d_ssdit_block_ws_free(&ctx->gpu_ssdit_block_ws);
            fprintf(stderr, "hip_sam3d: gpu ss_dit outer_ws alloc failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        ctx->gpu_ssdit_ws_alloced = 1;
        ctx->gpu_ssdit_ws_n_c = n_cond_min;
    }
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_debug_ss_dit_forward(hip_sam3d_ctx *ctx,
                                    const float *const *latents_in,
                                    float *const *latents_out,
                                    const float *cond, int n_cond,
                                    float t, float d)
{
    if (!ctx || !latents_in || !latents_out || !cond || n_cond <= 0)
        return HIP_SAM3D_E_INVAL;
    int rc = cs3d_ensure_ssdit_gpu(ctx, n_cond);
    if (rc != HIP_SAM3D_E_OK) return rc;
    return cs3d_ssdit_outer_forward(&ctx->gpu_ssdit, &ctx->gpu_ssdit_fns,
                                    &ctx->gpu_ssdit_block_ws,
                                    &ctx->gpu_ssdit_outer_ws,
                                    latents_in, latents_out,
                                    cond, n_cond, t, d) == 0
           ? HIP_SAM3D_E_OK : HIP_SAM3D_E_LOAD;
}

int hip_sam3d_ss_dit_info(hip_sam3d_ctx *ctx,
                           int *out_n_blocks, int *out_dim,
                           int *out_cond_channels, int *out_is_shortcut)
{
    if (!ctx) return HIP_SAM3D_E_INVAL;
    int rc = cs3d_ensure_ss_dit(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    if (out_n_blocks)      *out_n_blocks      = sam3d_cpu_ss_dit_n_blocks     (ctx->cpu_ss_dit);
    if (out_dim)           *out_dim           = sam3d_cpu_ss_dit_dim          (ctx->cpu_ss_dit);
    if (out_cond_channels) *out_cond_channels = sam3d_cpu_ss_dit_cond_channels(ctx->cpu_ss_dit);
    if (out_is_shortcut)   *out_is_shortcut   = sam3d_cpu_ss_dit_is_shortcut  (ctx->cpu_ss_dit);
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_ss_dit_n_latents(void) { return sam3d_cpu_ss_dit_n_latents(); }
int hip_sam3d_ss_dit_lat_elts(int i) { return sam3d_cpu_ss_dit_lat_elts(i); }

/* Active ss_latent source: prefer override. Returns NULL if neither is set. */
static const cs3d_host_nd *cs3d_active_ss_latent(const hip_sam3d_ctx *ctx)
{
    return ctx->ovr_ss_latent.data ? &ctx->ovr_ss_latent : &ctx->ss_latent;
}

static int cs3d_adopt_occupancy(hip_sam3d_ctx *ctx, float *data,
                                const int dims[3])
{
    cs3d_free_nd(&ctx->occupancy);
    if (ctx->d_occupancy) { hipFree(ctx->d_occupancy); ctx->d_occupancy = 0; }
    if (!data) return HIP_SAM3D_E_INVAL;
    size_t numel = 1;
    for (int i = 0; i < 3; i++) {
        if (dims[i] <= 0) { free(data); return HIP_SAM3D_E_INVAL; }
        ctx->occupancy.dims[i] = dims[i];
        numel *= (size_t)dims[i];
    }
    ctx->occupancy.ndim = 3;
    ctx->occupancy.data = data;
    ctx->d_occupancy = hip_upload_raw(data, numel * sizeof(float));
    if (!ctx->d_occupancy) {
        cs3d_free_nd(&ctx->occupancy);
        return HIP_SAM3D_E_LOAD;
    }
    return HIP_SAM3D_E_OK;
}

static int cs3d_ensure_ssdec_gpu(hip_sam3d_ctx *ctx)
{
    if (!ctx->cpu_ss_dec) {
        ctx->cpu_ss_dec = sam3d_cpu_ss_dec_load(ctx->safetensors_dir_resolved);
        if (!ctx->cpu_ss_dec) {
            fprintf(stderr, "hip_sam3d: ss_decoder load failed (dir=%s)\n",
                    ctx->safetensors_dir_resolved);
            return HIP_SAM3D_E_LOAD;
        }
    }
    if (!ctx->gpu_ssdec_loaded) {
        const t2_ss_dec *m = (const t2_ss_dec *)sam3d_cpu_ss_dec_model(ctx->cpu_ss_dec);
        if (!m || cs3d_ssdec_gpu_load(&ctx->gpu_ssdec, m, ctx->cfg.verbose) < 0) {
            fprintf(stderr, "hip_sam3d: gpu ss_decoder weight upload failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        ctx->gpu_ssdec_loaded = 1;
        if (cs3d_ssdec_fns_init(&ctx->gpu_ssdec_fns, ctx->mod) < 0) {
            fprintf(stderr, "hip_sam3d: gpu ss_decoder fns lookup failed\n");
            return HIP_SAM3D_E_LOAD;
        }
    }
    if (!ctx->gpu_ssdec_ws_alloced) {
        if (cs3d_ssdec_ws_alloc(&ctx->gpu_ssdec_ws) < 0) {
            fprintf(stderr, "hip_sam3d: gpu ss_decoder workspace alloc failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        ctx->gpu_ssdec_ws_alloced = 1;
    }
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_run_ss_decode(hip_sam3d_ctx *ctx) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_nd *lat = cs3d_active_ss_latent(ctx);
    if (!lat->data) {
        fprintf(stderr, "hip_sam3d: ss_decode needs ss_latent — "
                        "run ss_dit (or override) first\n");
        return HIP_SAM3D_E_NO_INPUT;
    }
    if (lat->ndim != 4 || lat->dims[0] != 8 || lat->dims[1] != 16 ||
        lat->dims[2] != 16 || lat->dims[3] != 16) {
        fprintf(stderr, "hip_sam3d: ss_decode expects ss_latent [8,16,16,16], "
                        "got ndim=%d [%d,%d,%d,%d]\n",
                lat->ndim, lat->dims[0], lat->dims[1], lat->dims[2], lat->dims[3]);
        return HIP_SAM3D_E_INVAL;
    }
    int rc = cs3d_ensure_ssdec_gpu(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;

    hipDeviceptr_t d_lat = 0, d_out = 0;
    size_t lat_bytes = (size_t)8 * 16 * 16 * 16 * sizeof(float);
    d_lat = hip_upload_raw(lat->data, lat_bytes);
    if (!d_lat) return HIP_SAM3D_E_LOAD;
    if (hipMalloc(&d_out, (size_t)64 * 64 * 64 * sizeof(float)) != hipSuccess) {
        hipFree(d_lat);
        return HIP_SAM3D_E_LOAD;
    }
    if (cs3d_ssdec_forward(&ctx->gpu_ssdec, &ctx->gpu_ssdec_fns,
                           &ctx->gpu_ssdec_ws, d_lat, d_out,
                           ctx->cfg.verbose) < 0) {
        fprintf(stderr, "hip_sam3d: gpu ss_decoder forward failed\n");
        hipFree(d_lat); hipFree(d_out);
        return HIP_SAM3D_E_LOAD;
    }
    hipFree(d_lat);

    float *logits = (float *)malloc((size_t)64 * 64 * 64 * sizeof(float));
    if (!logits) { hipFree(d_out); return HIP_SAM3D_E_LOAD; }
    if (hipMemcpyDtoH(logits, d_out, (size_t)64 * 64 * 64 * sizeof(float)) != hipSuccess) {
        free(logits); hipFree(d_out); return HIP_SAM3D_E_LOAD;
    }
    hipFree(d_out);
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
                "hip_sam3d: ss_decode occupancy min=%.6g max=%.6g mean=%.6g pos=%d\n",
                mn, mx, sum / (64.0 * 64.0 * 64.0), pos);
    }
    const int dims[3] = {64, 64, 64};
    return cs3d_adopt_occupancy(ctx, logits, dims);
}
/* Active occupancy source: prefer override. */
static const cs3d_host_nd *cs3d_active_occupancy(const hip_sam3d_ctx *ctx)
{
    return ctx->ovr_occupancy.data ? &ctx->ovr_occupancy : &ctx->occupancy;
}

static int cs3d_ensure_slat_dit(hip_sam3d_ctx *ctx) {
    if (ctx->cpu_slat_dit) return HIP_SAM3D_E_OK;
    ctx->cpu_slat_dit = sam3d_cpu_slat_dit_load(ctx->safetensors_dir_resolved);
    if (!ctx->cpu_slat_dit) {
        fprintf(stderr, "hip_sam3d: slat_dit load failed (dir=%s)\n",
                ctx->safetensors_dir_resolved);
        return HIP_SAM3D_E_LOAD;
    }
    return HIP_SAM3D_E_OK;
}

static int cs3d_ensure_slatdit_gpu(hip_sam3d_ctx *ctx, int n_max, int n_cond)
{
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    sam3d_slat_dit_model *m = sam3d_cpu_slat_dit_model(ctx->cpu_slat_dit);
    if (!m) return HIP_SAM3D_E_LOAD;

    if (!ctx->gpu_slatdit_loaded) {
        if (cs3d_slatdit_gpu_load_transformer(&ctx->gpu_slatdit, m, ctx->cfg.verbose) != 0) {
            fprintf(stderr, "hip_sam3d: gpu slat_dit transformer weight upload failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        ctx->gpu_slatdit_loaded = 1;
        if (cs3d_slatdit_fns_lookup(&ctx->gpu_slatdit_fns, ctx->mod) != 0) {
            fprintf(stderr, "hip_sam3d: gpu slat_dit fns lookup failed\n");
            return HIP_SAM3D_E_LOAD;
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
            fprintf(stderr, "hip_sam3d: gpu slat_dit transformer ws alloc failed\n");
            return HIP_SAM3D_E_LOAD;
        }
        ctx->gpu_slatdit_ws_alloced = 1;
        ctx->gpu_slatdit_ws_n = n_max;
        ctx->gpu_slatdit_ws_nc = n_cond;
    }
    return HIP_SAM3D_E_OK;
}

static int cs3d_ensure_devbuf(hipDeviceptr_t *ptr, size_t *cap, size_t need)
{
    if (!ptr || !cap || need == 0) return -1;
    if (*ptr && *cap >= need) return 0;
    if (*ptr) {
        hipFree(*ptr);
        *ptr = 0;
        *cap = 0;
    }
    if (hipMalloc(ptr, need) != hipSuccess) return -1;
    *cap = need;
    return 0;
}

static int cs3d_slat_io_upload_qtensor(const qtensor *t, const char *name,
                                       hipDeviceptr_t *out, size_t *total_bytes)
{
    if (!t || !t->data || t->n_rows <= 0 || t->n_cols <= 0 || !out)
        return -1;
    int n = t->n_rows * t->n_cols;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (!buf) return -1;
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        int bs = 0, ts = 0;
        switch (t->type) {
            case GGML_TYPE_Q8_0: bs = 32;  ts = 34;  break;
            case GGML_TYPE_Q4_K: bs = 256; ts = 144; break;
            case GGML_TYPE_Q6_K: bs = 256; ts = 210; break;
            default:
                free(buf);
                fprintf(stderr, "hip_sam3d: unsupported SLAT IO tensor %s\n", name);
                return -1;
        }
        size_t row_bytes = (size_t)((t->n_cols + bs - 1) / bs) * ts;
        for (int r = 0; r < t->n_rows; r++) {
            const void *row = (const uint8_t *)t->data + (size_t)r * row_bytes;
            dequant_row(t->type, row, buf + (size_t)r * t->n_cols, t->n_cols);
        }
    }
    size_t bytes = (size_t)n * sizeof(float);
    *out = hip_upload_raw(buf, bytes);
    free(buf);
    if (!*out) {
        fprintf(stderr, "hip_sam3d: SLAT IO upload failed for %s\n", name);
        return -1;
    }
    if (total_bytes) *total_bytes += bytes;
    return 0;
}

static int cs3d_slat_io_load_block(cs3d_slat_io_block_gpu *dst,
                                   const sam3d_slat_io_block *src,
                                   const char *name,
                                   size_t *total_bytes)
{
    if (!dst || !src) return -1;
    memset(dst, 0, sizeof(*dst));
    dst->C_in = src->C_in;
    dst->C_out = src->C_out;
    dst->updown = src->updown;
    dst->has_skip = src->has_skip;
#define UP_IO_(field) do { \
    char tname[128]; \
    snprintf(tname, sizeof(tname), "%s.%s", name, #field); \
    if (cs3d_slat_io_upload_qtensor(&src->field, tname, &dst->field, total_bytes) != 0) goto fail; \
} while (0)
    UP_IO_(norm1_w); UP_IO_(norm1_b);
    UP_IO_(conv1_w); UP_IO_(conv1_b);
    UP_IO_(conv2_w); UP_IO_(conv2_b);
    UP_IO_(emb_w);   UP_IO_(emb_b);
    if (src->has_skip) {
        UP_IO_(skip_w); UP_IO_(skip_b);
    }
#undef UP_IO_
    return 0;
fail:
    cs3d_slat_io_block_gpu_free(dst);
    return -1;
}

static int cs3d_slat_io_fns_lookup(cs3d_slat_io_fns *f, hipModule_t mod)
{
    if (!f) return -1;
    if (f->loaded) return 0;
    if (hipModuleGetFunction(&f->gemm, mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&f->down, mod, "slat_downsample2_mean_include_self_serial_f32") != hipSuccess ||
        hipModuleGetFunction(&f->up, mod, "slat_upsample2_nearest_f32") != hipSuccess ||
        hipModuleGetFunction(&f->idx, mod, "slat_build_coord_index64_i32") != hipSuccess ||
        hipModuleGetFunction(&f->ln, mod, "layernorm_token_f32") != hipSuccess ||
        hipModuleGetFunction(&f->silu, mod, "silu_inplace_f32") != hipSuccess ||
        hipModuleGetFunction(&f->conv, mod, "slat_submconv3x3_f32") != hipSuccess ||
        hipModuleGetFunction(&f->modln, mod, "modulated_ln_f32") != hipSuccess ||
        hipModuleGetFunction(&f->resadd, mod, "residual_add_f32") != hipSuccess)
        return -1;
    f->loaded = 1;
    return 0;
}

static int cs3d_ensure_slat_io_gpu(hip_sam3d_ctx *ctx)
{
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    sam3d_slat_dit_model *m = sam3d_cpu_slat_dit_model(ctx->cpu_slat_dit);
    if (!m || m->n_io_res_blocks != 2) return HIP_SAM3D_E_LOAD;
    if (cs3d_slat_io_fns_lookup(&ctx->gpu_slat_io_fns, ctx->mod) != 0) {
        fprintf(stderr, "hip_sam3d: gpu slat IO fns lookup failed\n");
        return HIP_SAM3D_E_LOAD;
    }
    if (ctx->gpu_slat_io.loaded) return HIP_SAM3D_E_OK;

    size_t total = 0;
    if (cs3d_slat_io_load_block(&ctx->gpu_slat_io.input[0], &m->input_blocks[0], "input_blocks.0", &total) != 0 ||
        cs3d_slat_io_load_block(&ctx->gpu_slat_io.input[1], &m->input_blocks[1], "input_blocks.1", &total) != 0 ||
        cs3d_slat_io_load_block(&ctx->gpu_slat_io.output[0], &m->out_blocks[0], "out_blocks.0", &total) != 0 ||
        cs3d_slat_io_load_block(&ctx->gpu_slat_io.output[1], &m->out_blocks[1], "out_blocks.1", &total) != 0 ||
        cs3d_slat_io_upload_qtensor(&m->input_w, "input_layer.weight", &ctx->gpu_slat_io.input_w, &total) != 0 ||
        cs3d_slat_io_upload_qtensor(&m->input_b, "input_layer.bias", &ctx->gpu_slat_io.input_b, &total) != 0 ||
        cs3d_slat_io_upload_qtensor(&m->out_w, "out_layer.weight", &ctx->gpu_slat_io.out_w, &total) != 0 ||
        cs3d_slat_io_upload_qtensor(&m->out_b, "out_layer.bias", &ctx->gpu_slat_io.out_b, &total) != 0) {
        cs3d_slat_io_gpu_free(&ctx->gpu_slat_io);
        fprintf(stderr, "hip_sam3d: gpu slat IO weight upload failed\n");
        return HIP_SAM3D_E_LOAD;
    }
    ctx->gpu_slat_io.input_in = m->input_w.n_cols;
    ctx->gpu_slat_io.input_out = m->input_w.n_rows;
    ctx->gpu_slat_io.out_in = m->out_w.n_cols;
    ctx->gpu_slat_io.out_out = m->out_w.n_rows;
    ctx->gpu_slat_io.total_bytes = total;
    ctx->gpu_slat_io.loaded = 1;
    if (ctx->cfg.verbose >= 1)
        fprintf(stderr, "hip_sam3d: SLAT IO GPU weights %.2f MiB\n",
                (double)total / (1024.0 * 1024.0));
    return HIP_SAM3D_E_OK;
}

static int cs3d_slat_io_block_gpu_hook(void *user, int is_output,
                                       int block_idx, const void *bk_void,
                                       void *xp_void, const float *t_emb,
                                       const int32_t *up_target_coords,
                                       int up_target_N, int dim, float ln_eps)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    sp3d_tensor **xp = (sp3d_tensor **)xp_void;
    const sam3d_slat_io_block *bk = (const sam3d_slat_io_block *)bk_void;
    if (!ctx || !xp || !*xp || !bk || !t_emb || block_idx < 0 || block_idx >= 2)
        return -1;
    sp3d_tensor *x = *xp;
    if (x->N <= 0 || x->C != bk->C_in) return -1;
    if (cs3d_ensure_slat_io_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    cs3d_slat_io_block_gpu *gb = is_output ? &ctx->gpu_slat_io.output[block_idx]
                                           : &ctx->gpu_slat_io.input[block_idx];
    cs3d_slat_io_fns *fn = &ctx->gpu_slat_io_fns;
    int C_in = gb->C_in, C_out = gb->C_out;
    if (C_in != bk->C_in || C_out != bk->C_out) return -1;
    if (gb->updown == SAM3D_SLAT_UPDOWN_UP && (!up_target_coords || up_target_N <= 0))
        return -1;

    int inN = x->N;
    int outN = (gb->updown == SAM3D_SLAT_UPDOWN_UP) ? up_target_N : inN;
    cs3d_slat_io_ws *ws = &ctx->gpu_slat_io_ws;
    hipDeviceptr_t d_coords = 0, d_base = 0;
    int32_t *host_coords = NULL;
    float *host_feats = NULL;
    sp3d_tensor *new_t = NULL;
    int rc = -1;

    size_t in_coord_bytes = (size_t)inN * 4 * sizeof(int32_t);
    size_t in_feat_bytes = (size_t)inN * C_in * sizeof(float);
    size_t t_bytes = (size_t)dim * sizeof(float);
    if (cs3d_ensure_devbuf(&ws->d_in_coords, &ws->cap_in_coords, in_coord_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_in_feats,  &ws->cap_in_feats,  in_feat_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_t,         &ws->cap_t,         t_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_in_coords, x->coords, in_coord_bytes) != hipSuccess ||
        hipMemcpyHtoD(ws->d_in_feats, x->feats, in_feat_bytes) != hipSuccess ||
        hipMemcpyHtoD(ws->d_t, t_emb, t_bytes) != hipSuccess)
        goto done;

    if (gb->updown == SAM3D_SLAT_UPDOWN_DOWN) {
        if (cs3d_ensure_devbuf(&ws->d_coords, &ws->cap_coords, in_coord_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_base,   &ws->cap_base,   in_feat_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_counts, &ws->cap_counts, (size_t)inN * sizeof(int)) != 0 ||
            cs3d_ensure_devbuf(&ws->d_outN,   &ws->cap_outN,   sizeof(int)) != 0)
            goto done;
        d_coords = ws->d_coords;
        d_base = ws->d_base;
        void *a_down[] = { &ws->d_in_coords, &ws->d_in_feats, &inN, &C_in,
                           &d_coords, &d_base, &ws->d_counts, &ws->d_outN };
        if (hipModuleLaunchKernel(fn->down, 1, 1, 1, 1, 1, 1, 0, 0, a_down, NULL) != hipSuccess)
            goto done;
        if (hipMemcpyDtoH(&outN, ws->d_outN, sizeof(int)) != hipSuccess ||
            outN <= 0 || outN > inN)
            goto done;
    } else if (gb->updown == SAM3D_SLAT_UPDOWN_UP) {
        size_t out_coord_bytes = (size_t)up_target_N * 4 * sizeof(int32_t);
        size_t base_bytes = (size_t)up_target_N * C_in * sizeof(float);
        if (cs3d_ensure_devbuf(&ws->d_coords, &ws->cap_coords, out_coord_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_base,   &ws->cap_base,   base_bytes) != 0)
            goto done;
        if (hipMemcpyHtoD(ws->d_coords, up_target_coords, out_coord_bytes) != hipSuccess)
            goto done;
        d_coords = ws->d_coords;
        d_base = ws->d_base;
        void *a_up[] = { &ws->d_in_coords, &ws->d_in_feats, &inN, &C_in,
                         &d_coords, &up_target_N, &d_base };
        if (hipModuleLaunchKernel(fn->up, (up_target_N + 127) / 128, 1, 1, 128, 1, 1, 0, 0, a_up, NULL) != hipSuccess)
            goto done;
    } else {
        d_coords = ws->d_in_coords;
        d_base = ws->d_in_feats;
    }

    size_t index_bytes = (size_t)64 * 64 * 64 * sizeof(int);
    size_t emb_bytes = (size_t)2 * C_out * sizeof(float);
    size_t skip_bytes = (size_t)outN * C_out * sizeof(float);
    size_t h1_bytes = (size_t)outN * C_in * sizeof(float);
    if (cs3d_ensure_devbuf(&ws->d_index,  &ws->cap_index,  index_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_t_silu, &ws->cap_t_silu, t_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_emb,    &ws->cap_emb,    emb_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_skip,   &ws->cap_skip,   skip_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h1,     &ws->cap_h1,     h1_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h2,     &ws->cap_h2,     skip_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h3,     &ws->cap_h3,     skip_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h4,     &ws->cap_h4,     skip_bytes) != 0)
        goto done;

    if (hipMemsetD8(ws->d_index, 0xff, index_bytes) != hipSuccess ||
        hipMemcpyDtoD(ws->d_t_silu, ws->d_t, t_bytes) != hipSuccess)
        goto done;
    void *a_idx[] = { &d_coords, &outN, &ws->d_index };
    if (hipModuleLaunchKernel(fn->idx, (outN + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_idx, NULL) != hipSuccess)
        goto done;
    void *a_s0[] = { &ws->d_t_silu, &dim };
    if (hipModuleLaunchKernel(fn->silu, (dim + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_s0, NULL) != hipSuccess)
        goto done;
    int twoCout = 2 * C_out, one = 1;
    { unsigned gx = 1, gy = (twoCout + 15) / 16;
      void *a[] = { &ws->d_emb, &ws->d_t_silu, &gb->emb_w, &gb->emb_b, &one, &dim, &twoCout };
      if (hipModuleLaunchKernel(fn->gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess) goto done; }
    if (gb->has_skip) {
        unsigned gx = (outN + 15) / 16, gy = (C_out + 15) / 16;
        void *a[] = { &ws->d_skip, &d_base, &gb->skip_w, &gb->skip_b, &outN, &C_in, &C_out };
        if (hipModuleLaunchKernel(fn->gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess) goto done;
    } else {
        if (C_in != C_out ||
            hipMemcpyDtoD(ws->d_skip, d_base, (size_t)outN * C_out * sizeof(float)) != hipSuccess)
            goto done;
    }

    int affine = 1;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    int n_elem_in = outN * C_in;
    int n_elem_out = outN * C_out;
    int total_conv = outN * C_out;
    hipDeviceptr_t d_scale = ws->d_emb;
    hipDeviceptr_t d_shift = ws->d_emb + (size_t)C_out * sizeof(float);
    void *a_ln[] = { &ws->d_h1, &d_base, &gb->norm1_w, &gb->norm1_b,
                     &outN, &C_in, &ln_eps, &affine };
    if (hipModuleLaunchKernel(fn->ln, outN, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_ln, NULL) != hipSuccess)
        goto done;
    void *a_s1[] = { &ws->d_h1, &n_elem_in };
    if (hipModuleLaunchKernel(fn->silu, (n_elem_in + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_s1, NULL) != hipSuccess)
        goto done;
    void *a_c1[] = { &d_coords, &ws->d_h1, &ws->d_index, &gb->conv1_w, &gb->conv1_b,
                     &outN, &C_in, &C_out, &ws->d_h2 };
    if (hipModuleLaunchKernel(fn->conv, (total_conv + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_c1, NULL) != hipSuccess)
        goto done;
    void *a_ml[] = { &ws->d_h3, &ws->d_h2, &d_shift, &d_scale, &outN, &C_out, &ln_eps };
    if (hipModuleLaunchKernel(fn->modln, outN, 1, 1, threads, 1, 1, (unsigned)ln_smem, 0, a_ml, NULL) != hipSuccess)
        goto done;
    void *a_s2[] = { &ws->d_h3, &n_elem_out };
    if (hipModuleLaunchKernel(fn->silu, (n_elem_out + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_s2, NULL) != hipSuccess)
        goto done;
    void *a_c2[] = { &d_coords, &ws->d_h3, &ws->d_index, &gb->conv2_w, &gb->conv2_b,
                     &outN, &C_out, &C_out, &ws->d_h4 };
    if (hipModuleLaunchKernel(fn->conv, (total_conv + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_c2, NULL) != hipSuccess)
        goto done;
    void *a_ra[] = { &ws->d_h4, &ws->d_skip, &n_elem_out };
    if (hipModuleLaunchKernel(fn->resadd, (n_elem_out + 255) / 256, 1, 1, 256, 1, 1, 0, 0, a_ra, NULL) != hipSuccess)
        goto done;

    host_coords = (int32_t *)malloc((size_t)outN * 4 * sizeof(int32_t));
    host_feats = (float *)malloc((size_t)outN * C_out * sizeof(float));
    if (!host_coords || !host_feats) goto done;
    if (hipMemcpyDtoH(host_coords, d_coords, (size_t)outN * 4 * sizeof(int32_t)) != hipSuccess ||
        hipMemcpyDtoH(host_feats, ws->d_h4, (size_t)outN * C_out * sizeof(float)) != hipSuccess)
        goto done;
    new_t = sp3d_create(host_coords, host_feats, outN, C_out, x->batch_size);
    if (!new_t) goto done;
    sp3d_free(x);
    *xp = new_t;
    new_t = NULL;
    rc = 0;

done:
    if (new_t) sp3d_free(new_t);
    free(host_coords);
    free(host_feats);
    return rc;
}

static int cs3d_slat_input_layer_gpu_hook(void *user, void *xp_void,
                                          const void *input_w_void,
                                          const void *input_b_void,
                                          int out_channels)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    sp3d_tensor **xp = (sp3d_tensor **)xp_void;
    (void)input_w_void;
    (void)input_b_void;
    if (!ctx || !xp || !*xp || out_channels <= 0) return -1;
    sp3d_tensor *x = *xp;
    if (x->N <= 0 || x->C <= 0) return -1;
    if (cs3d_ensure_slat_io_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    if (x->C != ctx->gpu_slat_io.input_in ||
        out_channels != ctx->gpu_slat_io.input_out)
        return -1;

    cs3d_slat_io_ws *ws = &ctx->gpu_slat_io_ws;
    cs3d_slat_io_fns *fn = &ctx->gpu_slat_io_fns;
    int N = x->N;
    int C = x->C;
    int outC = out_channels;
    size_t in_bytes = (size_t)N * C * sizeof(float);
    size_t out_bytes = (size_t)N * outC * sizeof(float);
    float *host_feats = NULL;
    sp3d_tensor *new_t = NULL;
    int rc = -1;

    if (cs3d_ensure_devbuf(&ws->d_in_feats, &ws->cap_in_feats, in_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h1,       &ws->cap_h1,       out_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_in_feats, x->feats, in_bytes) != hipSuccess)
        goto done;
    { unsigned gx = (N + 15) / 16, gy = (outC + 15) / 16;
      void *a[] = { &ws->d_h1, &ws->d_in_feats, &ctx->gpu_slat_io.input_w,
                    &ctx->gpu_slat_io.input_b, &N, &C, &outC };
      if (hipModuleLaunchKernel(fn->gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }

    host_feats = (float *)malloc(out_bytes);
    if (!host_feats) goto done;
    if (hipMemcpyDtoH(host_feats, ws->d_h1, out_bytes) != hipSuccess)
        goto done;
    new_t = sp3d_create(x->coords, host_feats, N, outC, x->batch_size);
    if (!new_t) goto done;
    sp3d_free(x);
    *xp = new_t;
    new_t = NULL;
    rc = 0;

done:
    if (new_t) sp3d_free(new_t);
    free(host_feats);
    return rc;
}

static int cs3d_slat_final_layer_gpu_hook(void *user, void *xp_void,
                                          const void *out_w_void,
                                          const void *out_b_void,
                                          int out_channels, float eps)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    sp3d_tensor **xp = (sp3d_tensor **)xp_void;
    (void)out_w_void;
    (void)out_b_void;
    if (!ctx || !xp || !*xp || out_channels <= 0) return -1;
    sp3d_tensor *x = *xp;
    if (x->N <= 0 || x->C <= 0) return -1;
    if (cs3d_ensure_slat_io_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    if (x->C != ctx->gpu_slat_io.out_in ||
        out_channels != ctx->gpu_slat_io.out_out)
        return -1;

    cs3d_slat_io_ws *ws = &ctx->gpu_slat_io_ws;
    cs3d_slat_io_fns *fn = &ctx->gpu_slat_io_fns;
    int N = x->N;
    int C = x->C;
    int outC = out_channels;
    size_t in_bytes = (size_t)N * C * sizeof(float);
    size_t out_bytes = (size_t)N * outC * sizeof(float);
    float *host_feats = NULL;
    sp3d_tensor *new_t = NULL;
    int rc = -1;

    if (cs3d_ensure_devbuf(&ws->d_in_feats, &ws->cap_in_feats, in_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h1,       &ws->cap_h1,       in_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h2,       &ws->cap_h2,       out_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_in_feats, x->feats, in_bytes) != hipSuccess)
        goto done;

    int affine = 0;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    void *a_ln[] = { &ws->d_h1, &ws->d_in_feats,
                     &ctx->gpu_slat_io.out_w, &ctx->gpu_slat_io.out_b,
                     &N, &C, &eps, &affine };
    if (hipModuleLaunchKernel(fn->ln, N, 1, 1, threads, 1, 1,
                       (unsigned)ln_smem, 0, a_ln, NULL) != hipSuccess)
        goto done;

    { unsigned gx = (N + 15) / 16, gy = (outC + 15) / 16;
      void *a[] = { &ws->d_h2, &ws->d_h1, &ctx->gpu_slat_io.out_w,
                    &ctx->gpu_slat_io.out_b, &N, &C, &outC };
      if (hipModuleLaunchKernel(fn->gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }

    host_feats = (float *)malloc(out_bytes);
    if (!host_feats) goto done;
    if (hipMemcpyDtoH(host_feats, ws->d_h2, out_bytes) != hipSuccess)
        goto done;
    new_t = sp3d_create(x->coords, host_feats, N, outC, x->batch_size);
    if (!new_t) goto done;
    sp3d_free(x);
    *xp = new_t;
    new_t = NULL;
    rc = 0;

done:
    if (new_t) sp3d_free(new_t);
    free(host_feats);
    return rc;
}

static int cs3d_slat_transformer_gpu_hook_impl(void *user, float *feats, int N,
                                               const int32_t *coords,
                                               const float *t_emb,
                                               const float *cond, int n_cond,
                                               int dim, int n_blocks,
                                               int apply_ape)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    if (!ctx || !feats || !t_emb || !cond || N <= 0 || n_cond <= 0 ||
        (apply_ape && !coords))
        return -1;
    if (cs3d_ensure_slatdit_gpu(ctx, N, n_cond) != HIP_SAM3D_E_OK)
        return -1;
    if (dim != ctx->gpu_slatdit.dim || n_blocks != ctx->gpu_slatdit.n_blocks)
        return -1;

    size_t x_bytes = (size_t)N * dim * sizeof(float);
    size_t coord_bytes = (size_t)N * 4 * sizeof(int32_t);
    size_t c_bytes = (size_t)n_cond * dim * sizeof(float);
    if (cs3d_ensure_devbuf(&ctx->d_slat_hook_x, &ctx->slat_hook_x_bytes, x_bytes) != 0 ||
        cs3d_ensure_devbuf(&ctx->d_slat_hook_t_emb, &ctx->slat_hook_t_emb_bytes,
                           (size_t)dim * sizeof(float)) != 0)
        return -1;
    if (hipMemcpyHtoD(ctx->d_slat_hook_x, feats, x_bytes) != hipSuccess ||
        hipMemcpyHtoD(ctx->d_slat_hook_t_emb, t_emb, (size_t)dim * sizeof(float)) != hipSuccess)
        return -1;
    if (apply_ape) {
        hipFunction_t ape_fn = NULL;
        int freq_dim = dim / 3 / 2;
        int filled = freq_dim * 2 * 3;
        long long total = (long long)N * filled;
        if (filled <= 0 ||
            cs3d_ensure_devbuf(&ctx->d_slat_hook_coords, &ctx->slat_hook_coords_bytes,
                               coord_bytes) != 0 ||
            hipModuleGetFunction(&ape_fn, ctx->mod, "slat_ape_add_f32") != hipSuccess)
            return -1;
        int coords_cached = 0;
        if (ctx->slat_hook_coords_cache &&
            ctx->slat_hook_coords_cache_bytes == coord_bytes &&
            memcmp(ctx->slat_hook_coords_cache, coords, coord_bytes) == 0) {
            coords_cached = 1;
        }
        if (!coords_cached) {
            int32_t *cache = (int32_t *)realloc(ctx->slat_hook_coords_cache,
                                                coord_bytes);
            if (!cache) return -1;
            memcpy(cache, coords, coord_bytes);
            ctx->slat_hook_coords_cache = cache;
            ctx->slat_hook_coords_cache_bytes = coord_bytes;
            if (hipMemcpyHtoD(ctx->d_slat_hook_coords, coords, coord_bytes) != hipSuccess)
                return -1;
        }
        void *a_ape[] = { &ctx->d_slat_hook_x, &ctx->d_slat_hook_coords, &N, &dim };
        if (hipModuleLaunchKernel(ape_fn, (unsigned)((total + 255) / 256), 1, 1,
                           256, 1, 1, 0, 0, a_ape, NULL) != hipSuccess)
            return -1;
    }

    int own_cond = 1;
    hipDeviceptr_t d_cond = 0;
    if (cond == ctx->cond_tokens.data && ctx->d_cond_tokens &&
        ctx->cond_tokens.n == n_cond && ctx->cond_tokens.c == dim) {
        d_cond = ctx->d_cond_tokens;
        own_cond = 0;
    } else {
        d_cond = hip_upload_raw(cond, c_bytes);
    }
    if (!d_cond) {
        if (d_cond && own_cond) hipFree(d_cond);
        return -1;
    }

    int ok = (cs3d_slatdit_stack_forward(&ctx->gpu_slatdit_fns,
                                         &ctx->gpu_slatdit_ws,
                                         &ctx->gpu_slatdit, 0, n_blocks,
                                         ctx->d_slat_hook_t_emb,
                                         ctx->d_slat_hook_x, N, d_cond,
                                         n_cond) == 0);
    if (ok && hipMemcpyDtoH(feats, ctx->d_slat_hook_x, x_bytes) != hipSuccess)
        ok = 0;
    if (own_cond) hipFree(d_cond);
    return ok ? 0 : -1;
}

static int cs3d_slat_ape_transformer_gpu_hook(void *user, float *feats, int N,
                                              const int32_t *coords,
                                              const float *t_emb,
                                              const float *cond, int n_cond,
                                              int dim, int n_blocks)
{
    return cs3d_slat_transformer_gpu_hook_impl(user, feats, N, coords, t_emb,
                                               cond, n_cond, dim, n_blocks, 1);
}

static int cs3d_slat_transformer_gpu_hook(void *user, float *feats, int N,
                                          const int32_t *coords,
                                          const float *t_emb,
                                          const float *cond, int n_cond,
                                          int dim, int n_blocks)
{
    return cs3d_slat_transformer_gpu_hook_impl(user, feats, N, coords, t_emb,
                                               cond, n_cond, dim, n_blocks, 0);
}

static int cs3d_adopt_slat(hip_sam3d_ctx *ctx, int32_t *coords, float *feats,
                           int n, int c)
{
    cs3d_free_slat(&ctx->slat_tokens);
    if (ctx->d_slat_feats)  { hipFree(ctx->d_slat_feats);  ctx->d_slat_feats  = 0; }
    if (ctx->d_slat_coords) { hipFree(ctx->d_slat_coords); ctx->d_slat_coords = 0; }
    if (!coords || !feats || n <= 0 || c <= 0) {
        free(coords); free(feats);
        return HIP_SAM3D_E_INVAL;
    }
    size_t f_bytes = (size_t)n * c * sizeof(float);
    size_t k_bytes = (size_t)n * 4 * sizeof(int32_t);
    ctx->d_slat_feats  = hip_upload_raw(feats,  f_bytes);
    ctx->d_slat_coords = hip_upload_raw(coords, k_bytes);
    if (!ctx->d_slat_feats || !ctx->d_slat_coords) {
        if (ctx->d_slat_feats)  { hipFree(ctx->d_slat_feats);  ctx->d_slat_feats  = 0; }
        if (ctx->d_slat_coords) { hipFree(ctx->d_slat_coords); ctx->d_slat_coords = 0; }
        free(coords); free(feats);
        return HIP_SAM3D_E_LOAD;
    }
    ctx->slat_tokens.feats  = feats;
    ctx->slat_tokens.coords = coords;
    ctx->slat_tokens.n      = n;
    ctx->slat_tokens.c      = c;
    return HIP_SAM3D_E_OK;
}

static int cs3d_slat_argwhere_gpu(hip_sam3d_ctx *ctx, const cs3d_host_nd *occ,
                                  int32_t **out_coords, int *out_n)
{
    if (!ctx || !occ || !occ->data || !out_coords || !out_n ||
        occ->ndim != 3 || occ->dims[0] <= 0 || occ->dims[1] <= 0 ||
        occ->dims[2] <= 0) return HIP_SAM3D_E_INVAL;

    hipFunction_t fn = NULL;
    if (hipModuleGetFunction(&fn, ctx->mod,
                            "slat_occ_argwhere_serial_f32") != hipSuccess) {
        fprintf(stderr, "hip_sam3d: missing slat_occ_argwhere_serial_f32\n");
        return HIP_SAM3D_E_LOAD;
    }

    int D = occ->dims[0], H = occ->dims[1], W = occ->dims[2];
    int occ_n = D * H * W;
    hipDeviceptr_t d_occ = 0, d_count = 0, d_coords = 0;
    int own_occ = 0;
    if (occ == &ctx->occupancy && ctx->d_occupancy) {
        d_occ = ctx->d_occupancy;
    } else {
        d_occ = hip_upload_raw(occ->data, (size_t)occ_n * sizeof(float));
        own_occ = 1;
        if (!d_occ) return HIP_SAM3D_E_LOAD;
    }
    if (hipMalloc(&d_count, sizeof(int)) != hipSuccess ||
        hipMalloc(&d_coords, (size_t)occ_n * 4 * sizeof(int32_t)) != hipSuccess) {
        if (d_count) hipFree(d_count);
        if (d_coords) hipFree(d_coords);
        if (own_occ) hipFree(d_occ);
        return HIP_SAM3D_E_LOAD;
    }

    void *args[] = { &d_occ, &D, &H, &W, &d_count, &d_coords };
    hipError_t lrc = hipModuleLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    if (lrc == hipSuccess) lrc = hipDeviceSynchronize();
    int count = 0;
    if (lrc == hipSuccess)
        lrc = hipMemcpyDtoH(&count, d_count, sizeof(int));
    if (lrc != hipSuccess || count < 0 || count > occ_n) {
        hipFree(d_count);
        hipFree(d_coords);
        if (own_occ) hipFree(d_occ);
        return HIP_SAM3D_E_LOAD;
    }
    if (count == 0) {
        hipFree(d_count);
        hipFree(d_coords);
        if (own_occ) hipFree(d_occ);
        *out_coords = NULL;
        *out_n = 0;
        return HIP_SAM3D_E_OK;
    }

    int32_t *coords = (int32_t *)malloc((size_t)count * 4 * sizeof(int32_t));
    if (!coords) {
        hipFree(d_count);
        hipFree(d_coords);
        if (own_occ) hipFree(d_occ);
        return HIP_SAM3D_E_LOAD;
    }
    lrc = hipMemcpyDtoH(coords, d_coords, (size_t)count * 4 * sizeof(int32_t));
    hipFree(d_count);
    hipFree(d_coords);
    if (own_occ) hipFree(d_occ);
    if (lrc != hipSuccess) {
        free(coords);
        return HIP_SAM3D_E_LOAD;
    }
    *out_coords = coords;
    *out_n = count;
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_run_slat_dit(hip_sam3d_ctx *ctx) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_2d *cond = cs3d_active_cond(ctx);
    if (!cond->data) {
        fprintf(stderr, "hip_sam3d: slat_dit needs cond tokens — "
                        "run cond_fuser (or override) first\n");
        return HIP_SAM3D_E_NO_INPUT;
    }
    const cs3d_host_nd *occ = cs3d_active_occupancy(ctx);
    if (!occ->data) {
        fprintf(stderr, "hip_sam3d: slat_dit needs occupancy — "
                        "run ss_decode (or override) first\n");
        return HIP_SAM3D_E_NO_INPUT;
    }
    if (occ->ndim != 3 || occ->dims[0] != 64 || occ->dims[1] != 64 ||
        occ->dims[2] != 64) {
        fprintf(stderr, "hip_sam3d: slat_dit expects occupancy [64,64,64], "
                        "got ndim=%d [%d,%d,%d]\n",
                occ->ndim, occ->dims[0], occ->dims[1], occ->dims[2]);
        return HIP_SAM3D_E_INVAL;
    }
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;

#if defined(_OPENMP)
    int nthr = omp_get_max_threads(); if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif
    int steps = ctx->cfg.slat_steps > 0 ? ctx->cfg.slat_steps : 12;

    int32_t *active_coords = NULL;
    int active_voxels = 0;
    rc = cs3d_slat_argwhere_gpu(ctx, occ, &active_coords, &active_voxels);
    if (rc != HIP_SAM3D_E_OK) return rc;
    if (ctx->cfg.verbose >= 1) {
        fprintf(stderr, "hip_sam3d: slat_dit active voxels=%d (gpu argwhere occ>0), steps=%d\n",
                active_voxels, steps);
    }
    if (active_voxels <= 0) {
        fprintf(stderr, "hip_sam3d: slat_dit occupancy is fully negative\n");
        free(active_coords);
        return HIP_SAM3D_E_LOAD;
    }

    int32_t *out_coords = NULL;
    float   *out_feats  = NULL;
    int      out_n      = 0;
    sam3d_cpu_slat_dit_set_input_layer_hook(cs3d_slat_input_layer_gpu_hook, ctx);
    sam3d_cpu_slat_dit_set_io_block_hook(cs3d_slat_io_block_gpu_hook, ctx);
    sam3d_cpu_slat_dit_set_final_layer_hook(cs3d_slat_final_layer_gpu_hook, ctx);
    sam3d_cpu_slat_dit_set_ape_transformer_hook(cs3d_slat_ape_transformer_gpu_hook, ctx);
    sam3d_cpu_slat_dit_set_transformer_hook(cs3d_slat_transformer_gpu_hook, ctx);
    int slat_rc = sam3d_cpu_slat_dit_run_ode_from_coords(ctx->cpu_slat_dit,
                                                         active_coords, active_voxels,
                                                         cond->data, cond->n,
                                                         steps, ctx->cfg.seed, nthr,
                                                         &out_coords, &out_feats, &out_n);
    sam3d_cpu_slat_dit_set_transformer_hook(NULL, NULL);
    sam3d_cpu_slat_dit_set_ape_transformer_hook(NULL, NULL);
    sam3d_cpu_slat_dit_set_final_layer_hook(NULL, NULL);
    sam3d_cpu_slat_dit_set_io_block_hook(NULL, NULL);
    sam3d_cpu_slat_dit_set_input_layer_hook(NULL, NULL);
    free(active_coords);
    if (slat_rc != 0) {
        free(out_coords); free(out_feats);
        fprintf(stderr, "hip_sam3d: slat_dit ODE failed rc=%d active_voxels=%d\n",
                slat_rc, active_voxels);
        return HIP_SAM3D_E_LOAD;
    }
    int out_c = sam3d_cpu_slat_dit_out_channels(ctx->cpu_slat_dit);
    return cs3d_adopt_slat(ctx, out_coords, out_feats, out_n, out_c);
}

int hip_sam3d_debug_slat_dit_forward(hip_sam3d_ctx *ctx,
                                      const int32_t *coords,
                                      const float *feats, int N,
                                      float t,
                                      const float *cond, int n_cond,
                                      float *out_feats)
{
    if (!ctx || !coords || !feats || N <= 0 || !cond || n_cond <= 0 || !out_feats)
        return HIP_SAM3D_E_INVAL;
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
#if defined(_OPENMP)
    int nthr = omp_get_max_threads(); if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif
    sam3d_cpu_slat_dit_set_input_layer_hook(cs3d_slat_input_layer_gpu_hook, ctx);
    sam3d_cpu_slat_dit_set_io_block_hook(cs3d_slat_io_block_gpu_hook, ctx);
    sam3d_cpu_slat_dit_set_final_layer_hook(cs3d_slat_final_layer_gpu_hook, ctx);
    sam3d_cpu_slat_dit_set_ape_transformer_hook(cs3d_slat_ape_transformer_gpu_hook, ctx);
    sam3d_cpu_slat_dit_set_transformer_hook(cs3d_slat_transformer_gpu_hook, ctx);
    float *out = sam3d_cpu_slat_dit_forward(ctx->cpu_slat_dit,
                                            coords, feats, N, t,
                                            cond, n_cond, nthr);
    sam3d_cpu_slat_dit_set_transformer_hook(NULL, NULL);
    sam3d_cpu_slat_dit_set_ape_transformer_hook(NULL, NULL);
    sam3d_cpu_slat_dit_set_final_layer_hook(NULL, NULL);
    sam3d_cpu_slat_dit_set_io_block_hook(NULL, NULL);
    sam3d_cpu_slat_dit_set_input_layer_hook(NULL, NULL);
    if (!out) return HIP_SAM3D_E_LOAD;
    int out_c = sam3d_cpu_slat_dit_out_channels(ctx->cpu_slat_dit);
    memcpy(out_feats, out, (size_t)N * out_c * sizeof(float));
    free(out);
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_slat_dit_info(hip_sam3d_ctx *ctx,
                             int *out_in_channels, int *out_out_channels,
                             int *out_cond_channels)
{
    if (!ctx) return HIP_SAM3D_E_INVAL;
    int rc = cs3d_ensure_slat_dit(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    if (out_in_channels)   *out_in_channels   = sam3d_cpu_slat_dit_in_channels  (ctx->cpu_slat_dit);
    if (out_out_channels)  *out_out_channels  = sam3d_cpu_slat_dit_out_channels (ctx->cpu_slat_dit);
    if (out_cond_channels) *out_cond_channels = sam3d_cpu_slat_dit_cond_channels(ctx->cpu_slat_dit);
    return HIP_SAM3D_E_OK;
}

/* Active SLAT source: prefer override (verify_slat_gs injects ref data). */
static const cs3d_host_slat *cs3d_active_slat(const hip_sam3d_ctx *ctx)
{
    return ctx->ovr_slat.feats ? &ctx->ovr_slat : &ctx->slat_tokens;
}

static int cs3d_ensure_gs(hip_sam3d_ctx *ctx) {
    if (ctx->cpu_gs) return HIP_SAM3D_E_OK;
    ctx->cpu_gs = sam3d_cpu_gs_decoder_load(ctx->safetensors_dir_resolved);
    if (!ctx->cpu_gs) {
        fprintf(stderr, "hip_sam3d: gs_decoder load failed (dir=%s)\n",
                ctx->safetensors_dir_resolved);
        return HIP_SAM3D_E_LOAD;
    }
    return HIP_SAM3D_E_OK;
}

static int cs3d_ensure_gs_head_gpu(hip_sam3d_ctx *ctx)
{
    int rc = cs3d_ensure_gs(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    if (ctx->gpu_gs_head.loaded) return HIP_SAM3D_E_OK;
    sam3d_gs_decoder_model *m =
        (sam3d_gs_decoder_model *)sam3d_cpu_gs_decoder_model(ctx->cpu_gs);
    if (!m) return HIP_SAM3D_E_LOAD;
    size_t total = 0;
    if (cs3d_slat_io_upload_qtensor(&m->input_w, "gs.input_layer.weight",
                                    &ctx->gpu_gs_head.input_w, &total) != 0 ||
        cs3d_slat_io_upload_qtensor(&m->input_b, "gs.input_layer.bias",
                                    &ctx->gpu_gs_head.input_b, &total) != 0 ||
        cs3d_slat_io_upload_qtensor(&m->out_w, "gs.out_layer.weight",
                                    &ctx->gpu_gs_head.out_w, &total) != 0 ||
        cs3d_slat_io_upload_qtensor(&m->out_b, "gs.out_layer.bias",
                                    &ctx->gpu_gs_head.out_b, &total) != 0) {
        goto fail;
    }
    ctx->gpu_gs_head.mlp_blocks = (cs3d_gs_mlp_block_gpu *)calloc(
        (size_t)m->n_blocks, sizeof(ctx->gpu_gs_head.mlp_blocks[0]));
    if (!ctx->gpu_gs_head.mlp_blocks) goto fail;
    ctx->gpu_gs_head.n_mlp_blocks = m->n_blocks;
    ctx->gpu_gs_head.mlp_dim = m->dim;
    ctx->gpu_gs_head.mlp_hidden = (int)(m->dim * m->mlp_ratio + 0.5f);
    for (int b = 0; b < m->n_blocks; b++) {
        char name[128];
        const sam3d_gs_block *blk = &m->blocks[b];
        cs3d_gs_mlp_block_gpu *gb = &ctx->gpu_gs_head.mlp_blocks[b];
        snprintf(name, sizeof(name), "gs.blocks.%d.attn_qkv.weight", b);
        if (cs3d_slat_io_upload_qtensor(&blk->attn_qkv_w, name, &gb->qkv_w, &total) != 0)
            goto fail;
        snprintf(name, sizeof(name), "gs.blocks.%d.attn_qkv.bias", b);
        if (cs3d_slat_io_upload_qtensor(&blk->attn_qkv_b, name, &gb->qkv_b, &total) != 0)
            goto fail;
        snprintf(name, sizeof(name), "gs.blocks.%d.attn_out.weight", b);
        if (cs3d_slat_io_upload_qtensor(&blk->attn_out_w, name, &gb->out_w, &total) != 0)
            goto fail;
        snprintf(name, sizeof(name), "gs.blocks.%d.attn_out.bias", b);
        if (cs3d_slat_io_upload_qtensor(&blk->attn_out_b, name, &gb->out_b, &total) != 0)
            goto fail;
        snprintf(name, sizeof(name), "gs.blocks.%d.mlp_fc1.weight", b);
        if (cs3d_slat_io_upload_qtensor(&blk->mlp_fc1_w, name, &gb->fc1_w, &total) != 0)
            goto fail;
        snprintf(name, sizeof(name), "gs.blocks.%d.mlp_fc1.bias", b);
        if (cs3d_slat_io_upload_qtensor(&blk->mlp_fc1_b, name, &gb->fc1_b, &total) != 0)
            goto fail;
        snprintf(name, sizeof(name), "gs.blocks.%d.mlp_fc2.weight", b);
        if (cs3d_slat_io_upload_qtensor(&blk->mlp_fc2_w, name, &gb->fc2_w, &total) != 0)
            goto fail;
        snprintf(name, sizeof(name), "gs.blocks.%d.mlp_fc2.bias", b);
        if (cs3d_slat_io_upload_qtensor(&blk->mlp_fc2_b, name, &gb->fc2_b, &total) != 0)
            goto fail;
    }
    ctx->gpu_gs_head.input_in = m->input_w.n_cols;
    ctx->gpu_gs_head.input_out = m->input_w.n_rows;
    ctx->gpu_gs_head.out_in = m->out_w.n_cols;
    ctx->gpu_gs_head.out_out = m->out_w.n_rows;
    ctx->gpu_gs_head.total_bytes = total;
    ctx->gpu_gs_head.loaded = 1;
    return HIP_SAM3D_E_OK;

fail:
    cs3d_gs_head_gpu_free(&ctx->gpu_gs_head);
    return HIP_SAM3D_E_LOAD;
}

static int cs3d_adopt_gaussians(hip_sam3d_ctx *ctx, float *data, int n_total)
{
    cs3d_free_2d(&ctx->gaussians);
    if (ctx->d_gaussians) { hipFree(ctx->d_gaussians); ctx->d_gaussians = 0; }
    if (!data || n_total <= 0) { free(data); return HIP_SAM3D_E_INVAL; }
    size_t bytes = (size_t)n_total * HIP_SAM3D_GS_STRIDE * sizeof(float);
    ctx->d_gaussians = hip_upload_raw(data, bytes);
    if (!ctx->d_gaussians) { free(data); return HIP_SAM3D_E_LOAD; }
    ctx->gaussians.data = data;
    ctx->gaussians.n    = n_total;
    ctx->gaussians.c    = HIP_SAM3D_GS_STRIDE;
    return HIP_SAM3D_E_OK;
}

static int cs3d_gs_input_ape_gpu_hook(void *user,
                                      const int32_t *coords,
                                      const float *feats,
                                      int N, int in_channels,
                                      const void *input_w_void,
                                      const void *input_b_void,
                                      int dim,
                                      float **out_h)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    (void)input_w_void;
    (void)input_b_void;
    if (!ctx || !coords || !feats || !out_h || N <= 0 || in_channels <= 0 || dim <= 0)
        return -1;
    if (cs3d_ensure_gs_head_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    if (in_channels != ctx->gpu_gs_head.input_in ||
        dim != ctx->gpu_gs_head.input_out)
        return -1;
    hipFunction_t gemm = NULL, ape = NULL;
    if (hipModuleGetFunction(&gemm, ctx->mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&ape, ctx->mod, "slat_ape_add_f32") != hipSuccess)
        return -1;

    cs3d_gs_head_ws *ws = &ctx->gpu_gs_head_ws;
    size_t coord_bytes = (size_t)N * 4 * sizeof(int32_t);
    size_t in_bytes = (size_t)N * in_channels * sizeof(float);
    size_t h_bytes = (size_t)N * dim * sizeof(float);
    float *host_h = NULL;
    int rc = -1;
    if (cs3d_ensure_devbuf(&ws->d_coords, &ws->cap_coords, coord_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_in,     &ws->cap_in,     in_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h,      &ws->cap_h,      h_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_coords, coords, coord_bytes) != hipSuccess ||
        hipMemcpyHtoD(ws->d_in, feats, in_bytes) != hipSuccess)
        goto done;
    { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
      void *a[] = { &ws->d_h, &ws->d_in, &ctx->gpu_gs_head.input_w,
                    &ctx->gpu_gs_head.input_b, &N, &in_channels, &dim };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    int freq_dim = dim / 3 / 2;
    int filled = freq_dim * 2 * 3;
    long long total = (long long)N * filled;
    void *a_ape[] = { &ws->d_h, &ws->d_coords, &N, &dim };
    if (hipModuleLaunchKernel(ape, (unsigned)((total + 255) / 256), 1, 1,
                       256, 1, 1, 0, 0, a_ape, NULL) != hipSuccess)
        goto done;
    host_h = (float *)malloc(h_bytes);
    if (!host_h) goto done;
    if (hipMemcpyDtoH(host_h, ws->d_h, h_bytes) != hipSuccess)
        goto done;
    *out_h = host_h;
    host_h = NULL;
    rc = 0;

done:
    free(host_h);
    return rc;
}

static int cs3d_gs_final_layer_gpu_hook(void *user,
                                        const float *h,
                                        int N, int dim,
                                        const void *out_w_void,
                                        const void *out_b_void,
                                        int out_channels,
                                        float eps,
                                        float **out_feats)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    (void)out_w_void;
    (void)out_b_void;
    if (!ctx || !h || !out_feats || N <= 0 || dim <= 0 || out_channels <= 0)
        return -1;
    if (cs3d_ensure_gs_head_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    if (dim != ctx->gpu_gs_head.out_in ||
        out_channels != ctx->gpu_gs_head.out_out)
        return -1;
    hipFunction_t gemm = NULL, ln = NULL;
    if (hipModuleGetFunction(&gemm, ctx->mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&ln, ctx->mod, "layernorm_token_f32") != hipSuccess)
        return -1;

    cs3d_gs_head_ws *ws = &ctx->gpu_gs_head_ws;
    size_t h_bytes = (size_t)N * dim * sizeof(float);
    size_t out_bytes = (size_t)N * out_channels * sizeof(float);
    float *host_out = NULL;
    int rc = -1;
    if (cs3d_ensure_devbuf(&ws->d_h,   &ws->cap_h,   h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_ln,  &ws->cap_ln,  h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_out, &ws->cap_out, out_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_h, h, h_bytes) != hipSuccess)
        goto done;
    int affine = 0;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    void *a_ln[] = { &ws->d_ln, &ws->d_h, &ctx->gpu_gs_head.out_w,
                     &ctx->gpu_gs_head.out_b, &N, &dim, &eps, &affine };
    if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                       (unsigned)ln_smem, 0, a_ln, NULL) != hipSuccess)
        goto done;
    { unsigned gx = (N + 15) / 16, gy = (out_channels + 15) / 16;
      void *a[] = { &ws->d_out, &ws->d_ln, &ctx->gpu_gs_head.out_w,
                    &ctx->gpu_gs_head.out_b, &N, &dim, &out_channels };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    host_out = (float *)malloc(out_bytes);
    if (!host_out) goto done;
    if (hipMemcpyDtoH(host_out, ws->d_out, out_bytes) != hipSuccess)
        goto done;
    *out_feats = host_out;
    host_out = NULL;
    rc = 0;

done:
    free(host_out);
    return rc;
}

static int cs3d_gs_mlp_gpu_hook(void *user,
                                float *h,
                                int N, int dim,
                                const void *blk_void,
                                int hidden,
                                float eps)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    const sam3d_gs_block *blk = (const sam3d_gs_block *)blk_void;
    if (!ctx || !h || !blk || N <= 0 || dim <= 0 || hidden <= 0)
        return -1;
    if (cs3d_ensure_gs_head_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    sam3d_gs_decoder_model *m =
        (sam3d_gs_decoder_model *)sam3d_cpu_gs_decoder_model(ctx->cpu_gs);
    if (!m || dim != ctx->gpu_gs_head.mlp_dim ||
        hidden != ctx->gpu_gs_head.mlp_hidden)
        return -1;
    int bi = (int)(blk - m->blocks);
    if (bi < 0 || bi >= ctx->gpu_gs_head.n_mlp_blocks)
        return -1;
    cs3d_gs_mlp_block_gpu *gb = &ctx->gpu_gs_head.mlp_blocks[bi];
    hipFunction_t gemm = NULL, ln = NULL, gelu = NULL, resadd = NULL;
    if (hipModuleGetFunction(&gemm, ctx->mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&ln, ctx->mod, "layernorm_token_f32") != hipSuccess ||
        hipModuleGetFunction(&gelu, ctx->mod, "gelu_tanh_inplace_f32") != hipSuccess ||
        hipModuleGetFunction(&resadd, ctx->mod, "residual_add_f32") != hipSuccess)
        return -1;

    cs3d_gs_head_ws *ws = &ctx->gpu_gs_head_ws;
    size_t h_bytes = (size_t)N * dim * sizeof(float);
    size_t mlp_bytes = (size_t)N * hidden * sizeof(float);
    int n_h = N * dim;
    int n_mlp = N * hidden;
    int rc = -1;
    if (cs3d_ensure_devbuf(&ws->d_h,   &ws->cap_h,   h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_ln,  &ws->cap_ln,  h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_out, &ws->cap_out, h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_mlp, &ws->cap_mlp, mlp_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_h, h, h_bytes) != hipSuccess)
        goto done;

    int affine = 0;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    void *a_ln[] = { &ws->d_ln, &ws->d_h, &gb->fc2_w, &gb->fc2_b,
                     &N, &dim, &eps, &affine };
    if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                       (unsigned)ln_smem, 0, a_ln, NULL) != hipSuccess)
        goto done;
    { unsigned gx = (N + 15) / 16, gy = (hidden + 15) / 16;
      void *a[] = { &ws->d_mlp, &ws->d_ln, &gb->fc1_w, &gb->fc1_b,
                    &N, &dim, &hidden };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    { void *a[] = { &ws->d_mlp, &n_mlp };
      if (hipModuleLaunchKernel(gelu, (n_mlp + 255) / 256, 1, 1,
                         256, 1, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
      void *a[] = { &ws->d_out, &ws->d_mlp, &gb->fc2_w, &gb->fc2_b,
                    &N, &hidden, &dim };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    { void *a[] = { &ws->d_h, &ws->d_out, &n_h };
      if (hipModuleLaunchKernel(resadd, (n_h + 255) / 256, 1, 1,
                         256, 1, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    if (hipMemcpyDtoH(h, ws->d_h, h_bytes) != hipSuccess)
        goto done;
    rc = 0;

done:
    return rc;
}

static int cs3d_gs_win_kv_cmp(const void *a, const void *b)
{
    int64_t ka = ((const cs3d_gs_win_kv *)a)->key;
    int64_t kb = ((const cs3d_gs_win_kv *)b)->key;
    return (ka < kb) ? -1 : (ka > kb);
}

static int cs3d_gs_build_partition(const sp3d_tensor *x, int window_size,
                                   const int shift[3], int **fwd_out,
                                   cs3d_gs_win_run **runs_out,
                                   int *n_runs_out, int *max_len_out)
{
    if (!x || x->N <= 0 || !fwd_out || !runs_out || !n_runs_out || !max_len_out)
        return -1;
    int N = x->N;
    int32_t mx[3] = {0, 0, 0};
    for (int i = 0; i < N; i++) {
        for (int a = 0; a < 3; a++) {
            int32_t v = x->coords[i * 4 + 1 + a] + shift[a];
            if (v > mx[a]) mx[a] = v;
        }
    }
    int nw[3];
    for (int a = 0; a < 3; a++) nw[a] = (int)(mx[a] / window_size) + 1;
    int64_t off[3];
    off[2] = 1;
    off[1] = (int64_t)nw[2];
    off[0] = (int64_t)nw[1] * nw[2];
    int64_t per_batch = (int64_t)nw[0] * nw[1] * nw[2];

    cs3d_gs_win_kv *kv = (cs3d_gs_win_kv *)malloc((size_t)N * sizeof(*kv));
    int *fwd = (int *)malloc((size_t)N * sizeof(int));
    cs3d_gs_win_run *runs = (cs3d_gs_win_run *)malloc((size_t)N * sizeof(*runs));
    if (!kv || !fwd || !runs) {
        free(kv); free(fwd); free(runs);
        return -1;
    }
    for (int i = 0; i < N; i++) {
        int32_t b = x->coords[i * 4];
        int64_t key = (int64_t)b * per_batch;
        for (int a = 0; a < 3; a++) {
            int32_t v = (x->coords[i * 4 + 1 + a] + shift[a]) / window_size;
            key += (int64_t)v * off[a];
        }
        kv[i].key = key;
        kv[i].idx = i;
    }
    qsort(kv, (size_t)N, sizeof(*kv), cs3d_gs_win_kv_cmp);
    for (int i = 0; i < N; i++) fwd[i] = kv[i].idx;
    int nr = 0, max_len = 0;
    for (int i = 0; i < N;) {
        int j = i + 1;
        while (j < N && kv[j].key == kv[i].key) j++;
        runs[nr].start = i;
        runs[nr].len = j - i;
        if (runs[nr].len > max_len) max_len = runs[nr].len;
        nr++;
        i = j;
    }
    free(kv);
    *fwd_out = fwd;
    *runs_out = runs;
    *n_runs_out = nr;
    *max_len_out = max_len;
    return 0;
}

static int cs3d_gs_window_attn_gpu_hook(void *user,
                                        float *out,
                                        const float *qkv,
                                        const void *x_void,
                                        int window_size,
                                        const int shift[3],
                                        int n_heads,
                                        int head_dim)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    const sp3d_tensor *x = (const sp3d_tensor *)x_void;
    if (!ctx || !out || !qkv || !x || x->N <= 0 || !x->coords ||
        window_size <= 0 || !shift || n_heads <= 0 || head_dim <= 0)
        return -1;
    hipFunction_t gather = NULL, scatter = NULL, sdpa = NULL;
    if (hipModuleGetFunction(&gather, ctx->mod, "sam3d_gs_gather_qkv_window_f32") != hipSuccess ||
        hipModuleGetFunction(&scatter, ctx->mod, "sam3d_gs_scatter_window_f32") != hipSuccess ||
        hipModuleGetFunction(&sdpa, ctx->mod, "sdpa_f32") != hipSuccess)
        return -1;

    int *fwd = NULL;
    cs3d_gs_win_run *runs = NULL;
    int n_runs = 0, max_len = 0;
    if (cs3d_gs_build_partition(x, window_size, shift, &fwd, &runs,
                                &n_runs, &max_len) != 0)
        return -1;

    cs3d_gs_head_ws *ws = &ctx->gpu_gs_head_ws;
    int N = x->N;
    int dim = n_heads * head_dim;
    size_t qkv_bytes = (size_t)N * 3 * dim * sizeof(float);
    size_t out_bytes = (size_t)N * dim * sizeof(float);
    size_t fwd_bytes = (size_t)N * sizeof(int);
    size_t win_bytes = (size_t)max_len * dim * sizeof(float);
    float scale = 1.0f / sqrtf((float)head_dim);
    int rc = -1;

    if (cs3d_ensure_devbuf(&ws->d_qkv,  &ws->cap_qkv,  qkv_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_fwd,  &ws->cap_fwd,  fwd_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_attn, &ws->cap_attn, out_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_q,    &ws->cap_q,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_k,    &ws->cap_k,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_v,    &ws->cap_v,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_win,  &ws->cap_win,  win_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_qkv, qkv, qkv_bytes) != hipSuccess ||
        hipMemcpyHtoD(ws->d_fwd, fwd, fwd_bytes) != hipSuccess)
        goto done;

    for (int r = 0; r < n_runs; r++) {
        int start = runs[r].start;
        int len = runs[r].len;
        int elems = len * dim;
        void *a_g[] = { &ws->d_q, &ws->d_k, &ws->d_v, &ws->d_qkv,
                        &ws->d_fwd, &start, &len, &dim };
        if (hipModuleLaunchKernel(gather, (elems + 255) / 256, 1, 1,
                           256, 1, 1, 0, 0, a_g, NULL) != hipSuccess)
            goto done;
        size_t smem = (size_t)(256 + len) * sizeof(float);
        void *a_s[] = { &ws->d_win, &ws->d_q, &ws->d_k, &ws->d_v,
                        &len, &len, &n_heads, &head_dim, &scale };
        if (hipModuleLaunchKernel(sdpa, len, n_heads, 1,
                           256, 1, 1, (unsigned)smem, 0, a_s, NULL) != hipSuccess)
            goto done;
        void *a_sc[] = { &ws->d_attn, &ws->d_win, &ws->d_fwd,
                         &start, &len, &dim };
        if (hipModuleLaunchKernel(scatter, (elems + 255) / 256, 1, 1,
                           256, 1, 1, 0, 0, a_sc, NULL) != hipSuccess)
            goto done;
    }
    if (hipMemcpyDtoH(out, ws->d_attn, out_bytes) != hipSuccess)
        goto done;
    rc = 0;

done:
    free(fwd);
    free(runs);
    return rc;
}

static int cs3d_gs_attn_block_gpu_hook(void *user,
                                       float *h,
                                       const void *x_void,
                                       int N, int dim,
                                       const void *blk_void,
                                       int window_size,
                                       const int shift[3],
                                       int n_heads,
                                       int head_dim,
                                       float eps)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    const sp3d_tensor *x = (const sp3d_tensor *)x_void;
    const sam3d_gs_block *blk = (const sam3d_gs_block *)blk_void;
    if (!ctx || !h || !x || !x->coords || !blk || N <= 0 || x->N != N ||
        dim <= 0 || window_size <= 0 || !shift ||
        n_heads <= 0 || head_dim <= 0 || dim != n_heads * head_dim)
        return -1;
    if (cs3d_ensure_gs_head_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    sam3d_gs_decoder_model *m =
        (sam3d_gs_decoder_model *)sam3d_cpu_gs_decoder_model(ctx->cpu_gs);
    if (!m || dim != ctx->gpu_gs_head.mlp_dim)
        return -1;
    int bi = (int)(blk - m->blocks);
    if (bi < 0 || bi >= ctx->gpu_gs_head.n_mlp_blocks)
        return -1;
    cs3d_gs_mlp_block_gpu *gb = &ctx->gpu_gs_head.mlp_blocks[bi];

    hipFunction_t gemm = NULL, ln = NULL, gather = NULL, scatter = NULL;
    hipFunction_t sdpa = NULL, resadd = NULL;
    if (hipModuleGetFunction(&gemm, ctx->mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&ln, ctx->mod, "layernorm_token_f32") != hipSuccess ||
        hipModuleGetFunction(&gather, ctx->mod, "sam3d_gs_gather_qkv_window_f32") != hipSuccess ||
        hipModuleGetFunction(&scatter, ctx->mod, "sam3d_gs_scatter_window_f32") != hipSuccess ||
        hipModuleGetFunction(&sdpa, ctx->mod, "sdpa_f32") != hipSuccess ||
        hipModuleGetFunction(&resadd, ctx->mod, "residual_add_f32") != hipSuccess)
        return -1;

    int *fwd = NULL;
    cs3d_gs_win_run *runs = NULL;
    int n_runs = 0, max_len = 0;
    if (cs3d_gs_build_partition(x, window_size, shift, &fwd, &runs,
                                &n_runs, &max_len) != 0)
        return -1;

    cs3d_gs_head_ws *ws = &ctx->gpu_gs_head_ws;
    size_t h_bytes = (size_t)N * dim * sizeof(float);
    size_t qkv_bytes = (size_t)N * 3 * dim * sizeof(float);
    size_t fwd_bytes = (size_t)N * sizeof(int);
    size_t win_bytes = (size_t)max_len * dim * sizeof(float);
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_h = N * dim;
    int rc = -1;

    if (cs3d_ensure_devbuf(&ws->d_h,    &ws->cap_h,    h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_ln,   &ws->cap_ln,   h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_qkv,  &ws->cap_qkv,  qkv_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_fwd,  &ws->cap_fwd,  fwd_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_attn, &ws->cap_attn, h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_out,  &ws->cap_out,  h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_q,    &ws->cap_q,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_k,    &ws->cap_k,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_v,    &ws->cap_v,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_win,  &ws->cap_win,  win_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_h, h, h_bytes) != hipSuccess ||
        hipMemcpyHtoD(ws->d_fwd, fwd, fwd_bytes) != hipSuccess)
        goto done;

    int affine = 0;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    void *a_ln[] = { &ws->d_ln, &ws->d_h, &gb->out_w, &gb->out_b,
                     &N, &dim, &eps, &affine };
    if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                       (unsigned)ln_smem, 0, a_ln, NULL) != hipSuccess)
        goto done;
    { int qkv_dim = 3 * dim;
      unsigned gx = (N + 15) / 16, gy = (qkv_dim + 15) / 16;
      void *a[] = { &ws->d_qkv, &ws->d_ln, &gb->qkv_w, &gb->qkv_b,
                    &N, &dim, &qkv_dim };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }

    for (int r = 0; r < n_runs; r++) {
        int start = runs[r].start;
        int len = runs[r].len;
        int elems = len * dim;
        void *a_g[] = { &ws->d_q, &ws->d_k, &ws->d_v, &ws->d_qkv,
                        &ws->d_fwd, &start, &len, &dim };
        if (hipModuleLaunchKernel(gather, (elems + 255) / 256, 1, 1,
                           256, 1, 1, 0, 0, a_g, NULL) != hipSuccess)
            goto done;
        size_t smem = (size_t)(256 + len) * sizeof(float);
        void *a_s[] = { &ws->d_win, &ws->d_q, &ws->d_k, &ws->d_v,
                        &len, &len, &n_heads, &head_dim, &scale };
        if (hipModuleLaunchKernel(sdpa, len, n_heads, 1,
                           256, 1, 1, (unsigned)smem, 0, a_s, NULL) != hipSuccess)
            goto done;
        void *a_sc[] = { &ws->d_attn, &ws->d_win, &ws->d_fwd,
                         &start, &len, &dim };
        if (hipModuleLaunchKernel(scatter, (elems + 255) / 256, 1, 1,
                           256, 1, 1, 0, 0, a_sc, NULL) != hipSuccess)
            goto done;
    }
    { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
      void *a[] = { &ws->d_out, &ws->d_attn, &gb->out_w, &gb->out_b,
                    &N, &dim, &dim };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    { void *a[] = { &ws->d_h, &ws->d_out, &n_h };
      if (hipModuleLaunchKernel(resadd, (n_h + 255) / 256, 1, 1,
                         256, 1, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    if (hipMemcpyDtoH(h, ws->d_h, h_bytes) != hipSuccess)
        goto done;
    rc = 0;

done:
    free(fwd);
    free(runs);
    return rc;
}

static int cs3d_gs_block_gpu_hook(void *user,
                                  float *h,
                                  const void *x_void,
                                  int N, int dim,
                                  const void *blk_void,
                                  int window_size,
                                  const int shift[3],
                                  int n_heads,
                                  int head_dim,
                                  int hidden,
                                  float eps)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    const sp3d_tensor *x = (const sp3d_tensor *)x_void;
    const sam3d_gs_block *blk = (const sam3d_gs_block *)blk_void;
    if (!ctx || !h || !x || !x->coords || !blk || N <= 0 || x->N != N ||
        dim <= 0 || hidden <= 0 || window_size <= 0 || !shift ||
        n_heads <= 0 || head_dim <= 0 || dim != n_heads * head_dim)
        return -1;
    if (cs3d_ensure_gs_head_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    sam3d_gs_decoder_model *m =
        (sam3d_gs_decoder_model *)sam3d_cpu_gs_decoder_model(ctx->cpu_gs);
    if (!m || dim != ctx->gpu_gs_head.mlp_dim ||
        hidden != ctx->gpu_gs_head.mlp_hidden)
        return -1;
    int bi = (int)(blk - m->blocks);
    if (bi < 0 || bi >= ctx->gpu_gs_head.n_mlp_blocks)
        return -1;
    cs3d_gs_mlp_block_gpu *gb = &ctx->gpu_gs_head.mlp_blocks[bi];

    hipFunction_t gemm = NULL, ln = NULL, gather = NULL, scatter = NULL;
    hipFunction_t sdpa = NULL, resadd = NULL, gelu = NULL;
    if (hipModuleGetFunction(&gemm, ctx->mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&ln, ctx->mod, "layernorm_token_f32") != hipSuccess ||
        hipModuleGetFunction(&gather, ctx->mod, "sam3d_gs_gather_qkv_window_f32") != hipSuccess ||
        hipModuleGetFunction(&scatter, ctx->mod, "sam3d_gs_scatter_window_f32") != hipSuccess ||
        hipModuleGetFunction(&sdpa, ctx->mod, "sdpa_f32") != hipSuccess ||
        hipModuleGetFunction(&resadd, ctx->mod, "residual_add_f32") != hipSuccess ||
        hipModuleGetFunction(&gelu, ctx->mod, "gelu_tanh_inplace_f32") != hipSuccess)
        return -1;

    int *fwd = NULL;
    cs3d_gs_win_run *runs = NULL;
    int n_runs = 0, max_len = 0;
    if (cs3d_gs_build_partition(x, window_size, shift, &fwd, &runs,
                                &n_runs, &max_len) != 0)
        return -1;

    cs3d_gs_head_ws *ws = &ctx->gpu_gs_head_ws;
    size_t h_bytes = (size_t)N * dim * sizeof(float);
    size_t qkv_bytes = (size_t)N * 3 * dim * sizeof(float);
    size_t mlp_bytes = (size_t)N * hidden * sizeof(float);
    size_t fwd_bytes = (size_t)N * sizeof(int);
    size_t win_bytes = (size_t)max_len * dim * sizeof(float);
    float scale = 1.0f / sqrtf((float)head_dim);
    int n_h = N * dim;
    int n_mlp = N * hidden;
    int rc = -1;

    if (cs3d_ensure_devbuf(&ws->d_h,    &ws->cap_h,    h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_ln,   &ws->cap_ln,   h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_qkv,  &ws->cap_qkv,  qkv_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_fwd,  &ws->cap_fwd,  fwd_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_attn, &ws->cap_attn, h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_out,  &ws->cap_out,  h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_q,    &ws->cap_q,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_k,    &ws->cap_k,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_v,    &ws->cap_v,    win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_win,  &ws->cap_win,  win_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_mlp,  &ws->cap_mlp,  mlp_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_h, h, h_bytes) != hipSuccess ||
        hipMemcpyHtoD(ws->d_fwd, fwd, fwd_bytes) != hipSuccess)
        goto done;

    int affine = 0;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    void *a_ln1[] = { &ws->d_ln, &ws->d_h, &gb->out_w, &gb->out_b,
                      &N, &dim, &eps, &affine };
    if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                       (unsigned)ln_smem, 0, a_ln1, NULL) != hipSuccess)
        goto done;
    { int qkv_dim = 3 * dim;
      unsigned gx = (N + 15) / 16, gy = (qkv_dim + 15) / 16;
      void *a[] = { &ws->d_qkv, &ws->d_ln, &gb->qkv_w, &gb->qkv_b,
                    &N, &dim, &qkv_dim };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }

    for (int r = 0; r < n_runs; r++) {
        int start = runs[r].start;
        int len = runs[r].len;
        int elems = len * dim;
        void *a_g[] = { &ws->d_q, &ws->d_k, &ws->d_v, &ws->d_qkv,
                        &ws->d_fwd, &start, &len, &dim };
        if (hipModuleLaunchKernel(gather, (elems + 255) / 256, 1, 1,
                           256, 1, 1, 0, 0, a_g, NULL) != hipSuccess)
            goto done;
        size_t smem = (size_t)(256 + len) * sizeof(float);
        void *a_s[] = { &ws->d_win, &ws->d_q, &ws->d_k, &ws->d_v,
                        &len, &len, &n_heads, &head_dim, &scale };
        if (hipModuleLaunchKernel(sdpa, len, n_heads, 1,
                           256, 1, 1, (unsigned)smem, 0, a_s, NULL) != hipSuccess)
            goto done;
        void *a_sc[] = { &ws->d_attn, &ws->d_win, &ws->d_fwd,
                         &start, &len, &dim };
        if (hipModuleLaunchKernel(scatter, (elems + 255) / 256, 1, 1,
                           256, 1, 1, 0, 0, a_sc, NULL) != hipSuccess)
            goto done;
    }
    { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
      void *a[] = { &ws->d_out, &ws->d_attn, &gb->out_w, &gb->out_b,
                    &N, &dim, &dim };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    { void *a[] = { &ws->d_h, &ws->d_out, &n_h };
      if (hipModuleLaunchKernel(resadd, (n_h + 255) / 256, 1, 1,
                         256, 1, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }

    void *a_ln2[] = { &ws->d_ln, &ws->d_h, &gb->fc2_w, &gb->fc2_b,
                      &N, &dim, &eps, &affine };
    if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                       (unsigned)ln_smem, 0, a_ln2, NULL) != hipSuccess)
        goto done;
    { unsigned gx = (N + 15) / 16, gy = (hidden + 15) / 16;
      void *a[] = { &ws->d_mlp, &ws->d_ln, &gb->fc1_w, &gb->fc1_b,
                    &N, &dim, &hidden };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    { void *a[] = { &ws->d_mlp, &n_mlp };
      if (hipModuleLaunchKernel(gelu, (n_mlp + 255) / 256, 1, 1,
                         256, 1, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
      void *a[] = { &ws->d_out, &ws->d_mlp, &gb->fc2_w, &gb->fc2_b,
                    &N, &hidden, &dim };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    { void *a[] = { &ws->d_h, &ws->d_out, &n_h };
      if (hipModuleLaunchKernel(resadd, (n_h + 255) / 256, 1, 1,
                         256, 1, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }

    if (hipMemcpyDtoH(h, ws->d_h, h_bytes) != hipSuccess)
        goto done;
    rc = 0;

done:
    free(fwd);
    free(runs);
    return rc;
}

static int cs3d_gs_stack_gpu_hook(void *user,
                                  float *h,
                                  const void *x_void,
                                  int N, int dim,
                                  const void *blocks_void,
                                  int n_blocks,
                                  int window_size,
                                  int n_heads,
                                  int head_dim,
                                  int hidden,
                                  float eps)
{
    hip_sam3d_ctx *ctx = (hip_sam3d_ctx *)user;
    const sp3d_tensor *x = (const sp3d_tensor *)x_void;
    const sam3d_gs_block *blocks = (const sam3d_gs_block *)blocks_void;
    if (!ctx || !h || !x || !x->coords || !blocks || N <= 0 || x->N != N ||
        dim <= 0 || hidden <= 0 || n_blocks <= 0 || window_size <= 0 ||
        n_heads <= 0 || head_dim <= 0 || dim != n_heads * head_dim)
        return -1;
    if (cs3d_ensure_gs_head_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    sam3d_gs_decoder_model *m =
        (sam3d_gs_decoder_model *)sam3d_cpu_gs_decoder_model(ctx->cpu_gs);
    if (!m || blocks != m->blocks || n_blocks != m->n_blocks ||
        dim != ctx->gpu_gs_head.mlp_dim ||
        hidden != ctx->gpu_gs_head.mlp_hidden)
        return -1;

    hipFunction_t gemm = NULL, ln = NULL, gather = NULL, scatter = NULL;
    hipFunction_t sdpa = NULL, resadd = NULL, gelu = NULL;
    if (hipModuleGetFunction(&gemm, ctx->mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&ln, ctx->mod, "layernorm_token_f32") != hipSuccess ||
        hipModuleGetFunction(&gather, ctx->mod, "sam3d_gs_gather_qkv_window_f32") != hipSuccess ||
        hipModuleGetFunction(&scatter, ctx->mod, "sam3d_gs_scatter_window_f32") != hipSuccess ||
        hipModuleGetFunction(&sdpa, ctx->mod, "sdpa_f32") != hipSuccess ||
        hipModuleGetFunction(&resadd, ctx->mod, "residual_add_f32") != hipSuccess ||
        hipModuleGetFunction(&gelu, ctx->mod, "gelu_tanh_inplace_f32") != hipSuccess)
        return -1;

    cs3d_gs_head_ws *ws = &ctx->gpu_gs_head_ws;
    size_t h_bytes = (size_t)N * dim * sizeof(float);
    size_t qkv_bytes = (size_t)N * 3 * dim * sizeof(float);
    size_t mlp_bytes = (size_t)N * hidden * sizeof(float);
    size_t fwd_bytes = (size_t)N * sizeof(int);
    int n_h = N * dim;
    int n_mlp = N * hidden;
    int affine = 0;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    int rc = -1;

    if (cs3d_ensure_devbuf(&ws->d_h,    &ws->cap_h,    h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_ln,   &ws->cap_ln,   h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_qkv,  &ws->cap_qkv,  qkv_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_fwd,  &ws->cap_fwd,  fwd_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_attn, &ws->cap_attn, h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_out,  &ws->cap_out,  h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_mlp,  &ws->cap_mlp,  mlp_bytes) != 0)
        goto done;
    if (hipMemcpyHtoD(ws->d_h, h, h_bytes) != hipSuccess)
        goto done;

    for (int bi = 0; bi < n_blocks; bi++) {
        cs3d_gs_mlp_block_gpu *gb = &ctx->gpu_gs_head.mlp_blocks[bi];
        int shift_v = (bi & 1) ? (window_size / 2) : 0;
        int shift[3] = {shift_v, shift_v, shift_v};
        int *fwd = NULL;
        cs3d_gs_win_run *runs = NULL;
        int n_runs = 0, max_len = 0;
        if (cs3d_gs_build_partition(x, window_size, shift, &fwd, &runs,
                                    &n_runs, &max_len) != 0)
            goto done;

        size_t win_bytes = (size_t)max_len * dim * sizeof(float);
        float scale = 1.0f / sqrtf((float)head_dim);
        if (cs3d_ensure_devbuf(&ws->d_q,   &ws->cap_q,   win_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_k,   &ws->cap_k,   win_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_v,   &ws->cap_v,   win_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_win, &ws->cap_win, win_bytes) != 0 ||
            hipMemcpyHtoD(ws->d_fwd, fwd, fwd_bytes) != hipSuccess) {
            free(fwd);
            free(runs);
            goto done;
        }

        void *a_ln1[] = { &ws->d_ln, &ws->d_h, &gb->out_w, &gb->out_b,
                          &N, &dim, &eps, &affine };
        if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                           (unsigned)ln_smem, 0, a_ln1, NULL) != hipSuccess) {
            free(fwd); free(runs); goto done;
        }
        { int qkv_dim = 3 * dim;
          unsigned gx = (N + 15) / 16, gy = (qkv_dim + 15) / 16;
          void *a[] = { &ws->d_qkv, &ws->d_ln, &gb->qkv_w, &gb->qkv_b,
                        &N, &dim, &qkv_dim };
          if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess) {
              free(fwd); free(runs); goto done;
          } }

        for (int r = 0; r < n_runs; r++) {
            int start = runs[r].start;
            int len = runs[r].len;
            int elems = len * dim;
            void *a_g[] = { &ws->d_q, &ws->d_k, &ws->d_v, &ws->d_qkv,
                            &ws->d_fwd, &start, &len, &dim };
            if (hipModuleLaunchKernel(gather, (elems + 255) / 256, 1, 1,
                               256, 1, 1, 0, 0, a_g, NULL) != hipSuccess) {
                free(fwd); free(runs); goto done;
            }
            size_t smem = (size_t)(256 + len) * sizeof(float);
            void *a_s[] = { &ws->d_win, &ws->d_q, &ws->d_k, &ws->d_v,
                            &len, &len, &n_heads, &head_dim, &scale };
            if (hipModuleLaunchKernel(sdpa, len, n_heads, 1,
                               256, 1, 1, (unsigned)smem, 0, a_s, NULL) != hipSuccess) {
                free(fwd); free(runs); goto done;
            }
            void *a_sc[] = { &ws->d_attn, &ws->d_win, &ws->d_fwd,
                             &start, &len, &dim };
            if (hipModuleLaunchKernel(scatter, (elems + 255) / 256, 1, 1,
                               256, 1, 1, 0, 0, a_sc, NULL) != hipSuccess) {
                free(fwd); free(runs); goto done;
            }
        }
        free(fwd);
        free(runs);

        { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
          void *a[] = { &ws->d_out, &ws->d_attn, &gb->out_w, &gb->out_b,
                        &N, &dim, &dim };
          if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
        { void *a[] = { &ws->d_h, &ws->d_out, &n_h };
          if (hipModuleLaunchKernel(resadd, (n_h + 255) / 256, 1, 1,
                             256, 1, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }

        void *a_ln2[] = { &ws->d_ln, &ws->d_h, &gb->fc2_w, &gb->fc2_b,
                          &N, &dim, &eps, &affine };
        if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                           (unsigned)ln_smem, 0, a_ln2, NULL) != hipSuccess)
            goto done;
        { unsigned gx = (N + 15) / 16, gy = (hidden + 15) / 16;
          void *a[] = { &ws->d_mlp, &ws->d_ln, &gb->fc1_w, &gb->fc1_b,
                        &N, &dim, &hidden };
          if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
        { void *a[] = { &ws->d_mlp, &n_mlp };
          if (hipModuleLaunchKernel(gelu, (n_mlp + 255) / 256, 1, 1,
                             256, 1, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
        { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
          void *a[] = { &ws->d_out, &ws->d_mlp, &gb->fc2_w, &gb->fc2_b,
                        &N, &hidden, &dim };
          if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
        { void *a[] = { &ws->d_h, &ws->d_out, &n_h };
          if (hipModuleLaunchKernel(resadd, (n_h + 255) / 256, 1, 1,
                             256, 1, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
    }

    if (hipMemcpyDtoH(h, ws->d_h, h_bytes) != hipSuccess)
        goto done;
    rc = 0;

done:
    return rc;
}

static int cs3d_gs_transformer_gpu_forward(hip_sam3d_ctx *ctx,
                                           const sp3d_tensor *x,
                                           const sam3d_gs_decoder_model *m,
                                           hipDeviceptr_t src_d_coords,
                                           hipDeviceptr_t src_d_feats,
                                           float **out_feats)
{
    if (!ctx || !x || !x->coords || !x->feats || !m || x->N <= 0 || x->C <= 0)
        return -1;
    if (cs3d_ensure_gs_head_gpu(ctx) != HIP_SAM3D_E_OK) return -1;
    sam3d_gs_decoder_model *ctx_m =
        (sam3d_gs_decoder_model *)sam3d_cpu_gs_decoder_model(ctx->cpu_gs);
    if (m != ctx_m) return -1;

    int N = x->N;
    int in_channels = x->C;
    int dim = m->dim;
    int hidden = ctx->gpu_gs_head.mlp_hidden;
    int n_heads = m->n_heads;
    int head_dim = m->head_dim;
    int window_size = m->window_size;
    int out_channels = m->out_channels;
    if (in_channels != ctx->gpu_gs_head.input_in ||
        dim != ctx->gpu_gs_head.input_out ||
        dim != ctx->gpu_gs_head.mlp_dim ||
        dim != n_heads * head_dim ||
        out_channels != ctx->gpu_gs_head.out_out ||
        hidden <= 0)
        return -1;

    hipFunction_t gemm = NULL, ape = NULL, ln = NULL, gather = NULL;
    hipFunction_t scatter = NULL, sdpa = NULL, resadd = NULL, gelu = NULL;
    if (hipModuleGetFunction(&gemm, ctx->mod, "gemm_f32_bias") != hipSuccess ||
        hipModuleGetFunction(&ape, ctx->mod, "slat_ape_add_f32") != hipSuccess ||
        hipModuleGetFunction(&ln, ctx->mod, "layernorm_token_f32") != hipSuccess ||
        hipModuleGetFunction(&gather, ctx->mod, "sam3d_gs_gather_qkv_window_f32") != hipSuccess ||
        hipModuleGetFunction(&scatter, ctx->mod, "sam3d_gs_scatter_window_f32") != hipSuccess ||
        hipModuleGetFunction(&sdpa, ctx->mod, "sdpa_f32") != hipSuccess ||
        hipModuleGetFunction(&resadd, ctx->mod, "residual_add_f32") != hipSuccess ||
        hipModuleGetFunction(&gelu, ctx->mod, "gelu_tanh_inplace_f32") != hipSuccess)
        return -1;

    cs3d_gs_head_ws *ws = &ctx->gpu_gs_head_ws;
    size_t coord_bytes = (size_t)N * 4 * sizeof(int32_t);
    size_t in_bytes = (size_t)N * in_channels * sizeof(float);
    size_t h_bytes = (size_t)N * dim * sizeof(float);
    size_t qkv_bytes = (size_t)N * 3 * dim * sizeof(float);
    size_t mlp_bytes = (size_t)N * hidden * sizeof(float);
    size_t fwd_bytes = (size_t)N * sizeof(int);
    size_t out_bytes = (size_t)N * out_channels * sizeof(float);
    int n_h = N * dim;
    int n_mlp = N * hidden;
    int affine = 0;
    unsigned threads = 256;
    size_t ln_smem = 2 * threads * sizeof(float);
    float eps_block = 1e-6f;
    float eps_final = 1e-5f;
    float *host_out = NULL;
    int rc = -1;

    if (cs3d_ensure_devbuf(&ws->d_coords, &ws->cap_coords, coord_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_in,     &ws->cap_in,     in_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_h,      &ws->cap_h,      h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_ln,     &ws->cap_ln,     h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_qkv,    &ws->cap_qkv,    qkv_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_fwd,    &ws->cap_fwd,    fwd_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_attn,   &ws->cap_attn,   h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_out,    &ws->cap_out,    h_bytes) != 0 ||
        cs3d_ensure_devbuf(&ws->d_mlp,    &ws->cap_mlp,    mlp_bytes) != 0)
        goto done;
    if (src_d_coords) {
        if (hipMemcpyDtoD(ws->d_coords, src_d_coords, coord_bytes) != hipSuccess)
            goto done;
    } else if (hipMemcpyHtoD(ws->d_coords, x->coords, coord_bytes) != hipSuccess) {
        goto done;
    }
    if (src_d_feats) {
        if (hipMemcpyDtoD(ws->d_in, src_d_feats, in_bytes) != hipSuccess)
            goto done;
    } else if (hipMemcpyHtoD(ws->d_in, x->feats, in_bytes) != hipSuccess) {
        goto done;
    }

    { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
      void *a[] = { &ws->d_h, &ws->d_in, &ctx->gpu_gs_head.input_w,
                    &ctx->gpu_gs_head.input_b, &N, &in_channels, &dim };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }
    {
      int freq_dim = dim / 3 / 2;
      int filled = freq_dim * 2 * 3;
      long long total = (long long)N * filled;
      void *a_ape[] = { &ws->d_h, &ws->d_coords, &N, &dim };
      if (filled <= 0 ||
          hipModuleLaunchKernel(ape, (unsigned)((total + 255) / 256), 1, 1,
                         256, 1, 1, 0, 0, a_ape, NULL) != hipSuccess)
          goto done;
    }

    for (int bi = 0; bi < m->n_blocks; bi++) {
        cs3d_gs_mlp_block_gpu *gb = &ctx->gpu_gs_head.mlp_blocks[bi];
        int shift_v = (bi & 1) ? (window_size / 2) : 0;
        int shift[3] = {shift_v, shift_v, shift_v};
        int *fwd = NULL;
        cs3d_gs_win_run *runs = NULL;
        int n_runs = 0, max_len = 0;
        if (cs3d_gs_build_partition(x, window_size, shift, &fwd, &runs,
                                    &n_runs, &max_len) != 0)
            goto done;

        size_t win_bytes = (size_t)max_len * dim * sizeof(float);
        float scale = 1.0f / sqrtf((float)head_dim);
        if (cs3d_ensure_devbuf(&ws->d_q,   &ws->cap_q,   win_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_k,   &ws->cap_k,   win_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_v,   &ws->cap_v,   win_bytes) != 0 ||
            cs3d_ensure_devbuf(&ws->d_win, &ws->cap_win, win_bytes) != 0 ||
            hipMemcpyHtoD(ws->d_fwd, fwd, fwd_bytes) != hipSuccess) {
            free(fwd); free(runs); goto done;
        }

        void *a_ln1[] = { &ws->d_ln, &ws->d_h, &gb->out_w, &gb->out_b,
                          &N, &dim, &eps_block, &affine };
        if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                           (unsigned)ln_smem, 0, a_ln1, NULL) != hipSuccess) {
            free(fwd); free(runs); goto done;
        }
        { int qkv_dim = 3 * dim;
          unsigned gx = (N + 15) / 16, gy = (qkv_dim + 15) / 16;
          void *a[] = { &ws->d_qkv, &ws->d_ln, &gb->qkv_w, &gb->qkv_b,
                        &N, &dim, &qkv_dim };
          if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess) {
              free(fwd); free(runs); goto done;
          } }

        for (int r = 0; r < n_runs; r++) {
            int start = runs[r].start;
            int len = runs[r].len;
            int elems = len * dim;
            void *a_g[] = { &ws->d_q, &ws->d_k, &ws->d_v, &ws->d_qkv,
                            &ws->d_fwd, &start, &len, &dim };
            if (hipModuleLaunchKernel(gather, (elems + 255) / 256, 1, 1,
                               256, 1, 1, 0, 0, a_g, NULL) != hipSuccess) {
                free(fwd); free(runs); goto done;
            }
            size_t smem = (size_t)(256 + len) * sizeof(float);
            void *a_s[] = { &ws->d_win, &ws->d_q, &ws->d_k, &ws->d_v,
                            &len, &len, &n_heads, &head_dim, &scale };
            if (hipModuleLaunchKernel(sdpa, len, n_heads, 1,
                               256, 1, 1, (unsigned)smem, 0, a_s, NULL) != hipSuccess) {
                free(fwd); free(runs); goto done;
            }
            void *a_sc[] = { &ws->d_attn, &ws->d_win, &ws->d_fwd,
                             &start, &len, &dim };
            if (hipModuleLaunchKernel(scatter, (elems + 255) / 256, 1, 1,
                               256, 1, 1, 0, 0, a_sc, NULL) != hipSuccess) {
                free(fwd); free(runs); goto done;
            }
        }
        free(fwd);
        free(runs);

        { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
          void *a[] = { &ws->d_out, &ws->d_attn, &gb->out_w, &gb->out_b,
                        &N, &dim, &dim };
          if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
        { void *a[] = { &ws->d_h, &ws->d_out, &n_h };
          if (hipModuleLaunchKernel(resadd, (n_h + 255) / 256, 1, 1,
                             256, 1, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }

        void *a_ln2[] = { &ws->d_ln, &ws->d_h, &gb->fc2_w, &gb->fc2_b,
                          &N, &dim, &eps_block, &affine };
        if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                           (unsigned)ln_smem, 0, a_ln2, NULL) != hipSuccess)
            goto done;
        { unsigned gx = (N + 15) / 16, gy = (hidden + 15) / 16;
          void *a[] = { &ws->d_mlp, &ws->d_ln, &gb->fc1_w, &gb->fc1_b,
                        &N, &dim, &hidden };
          if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
        { void *a[] = { &ws->d_mlp, &n_mlp };
          if (hipModuleLaunchKernel(gelu, (n_mlp + 255) / 256, 1, 1,
                             256, 1, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
        { unsigned gx = (N + 15) / 16, gy = (dim + 15) / 16;
          void *a[] = { &ws->d_out, &ws->d_mlp, &gb->fc2_w, &gb->fc2_b,
                        &N, &hidden, &dim };
          if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
        { void *a[] = { &ws->d_h, &ws->d_out, &n_h };
          if (hipModuleLaunchKernel(resadd, (n_h + 255) / 256, 1, 1,
                             256, 1, 1, 0, 0, a, NULL) != hipSuccess)
              goto done; }
    }

    void *a_lnf[] = { &ws->d_ln, &ws->d_h, &ctx->gpu_gs_head.out_w,
                      &ctx->gpu_gs_head.out_b, &N, &dim, &eps_final, &affine };
    if (hipModuleLaunchKernel(ln, N, 1, 1, threads, 1, 1,
                       (unsigned)ln_smem, 0, a_lnf, NULL) != hipSuccess)
        goto done;
    { unsigned gx = (N + 15) / 16, gy = (out_channels + 15) / 16;
      void *a[] = { &ws->d_out, &ws->d_ln, &ctx->gpu_gs_head.out_w,
                    &ctx->gpu_gs_head.out_b, &N, &dim, &out_channels };
      if (hipModuleLaunchKernel(gemm, gx, gy, 1, 16, 16, 1, 0, 0, a, NULL) != hipSuccess)
          goto done; }

    if (out_feats) {
        host_out = (float *)malloc(out_bytes);
        if (!host_out) goto done;
        if (hipMemcpyDtoH(host_out, ws->d_out, out_bytes) != hipSuccess)
            goto done;
        *out_feats = host_out;
        host_out = NULL;
    }
    rc = 0;

done:
    free(host_out);
    return rc;
}

static int cs3d_gs_transformer_gpu_hook(void *user,
                                        const void *x_void,
                                        const void *m_void,
                                        float **out_feats)
{
    return cs3d_gs_transformer_gpu_forward((hip_sam3d_ctx *)user,
                                           (const sp3d_tensor *)x_void,
                                           (const sam3d_gs_decoder_model *)m_void,
                                           0, 0,
                                           out_feats);
}

int hip_sam3d_debug_slat_gs_transformer(hip_sam3d_ctx *ctx,
                                         const int32_t *coords,
                                         const float *feats, int N,
                                         float **out_feats, int *out_c)
{
    if (!ctx || !coords || !feats || N <= 0 || !out_feats) return HIP_SAM3D_E_INVAL;
    int rc = cs3d_ensure_gs(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
#if defined(_OPENMP)
    int nthr = omp_get_max_threads(); if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif
    sam3d_cpu_gs_decoder_set_input_ape_hook(cs3d_gs_input_ape_gpu_hook, ctx);
    sam3d_cpu_gs_decoder_set_final_layer_hook(cs3d_gs_final_layer_gpu_hook, ctx);
    sam3d_cpu_gs_decoder_set_window_attn_hook(cs3d_gs_window_attn_gpu_hook, ctx);
    sam3d_cpu_gs_decoder_set_attn_block_hook(cs3d_gs_attn_block_gpu_hook, ctx);
    sam3d_cpu_gs_decoder_set_mlp_hook(cs3d_gs_mlp_gpu_hook, ctx);
    sam3d_cpu_gs_decoder_set_block_hook(cs3d_gs_block_gpu_hook, ctx);
    sam3d_cpu_gs_decoder_set_stack_hook(cs3d_gs_stack_gpu_hook, ctx);
    sam3d_cpu_gs_decoder_set_transformer_hook(cs3d_gs_transformer_gpu_hook, ctx);
    float *out = sam3d_cpu_gs_decoder_transformer(ctx->cpu_gs, coords, feats,
                                                  N, nthr);
    sam3d_cpu_gs_decoder_set_transformer_hook(NULL, NULL);
    sam3d_cpu_gs_decoder_set_stack_hook(NULL, NULL);
    sam3d_cpu_gs_decoder_set_block_hook(NULL, NULL);
    sam3d_cpu_gs_decoder_set_mlp_hook(NULL, NULL);
    sam3d_cpu_gs_decoder_set_attn_block_hook(NULL, NULL);
    sam3d_cpu_gs_decoder_set_window_attn_hook(NULL, NULL);
    sam3d_cpu_gs_decoder_set_final_layer_hook(NULL, NULL);
    sam3d_cpu_gs_decoder_set_input_ape_hook(NULL, NULL);
    if (!out) return HIP_SAM3D_E_LOAD;
    *out_feats = out;
    if (out_c) *out_c = sam3d_cpu_gs_decoder_out_channels(ctx->cpu_gs);
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_debug_slat_gs_to_representation(hip_sam3d_ctx *ctx,
                                               const int32_t *coords,
                                               const float *feats_out, int N,
                                               float *xyz_out, float *dc_out,
                                               float *scaling_out, float *rotation_out,
                                               float *opacity_out)
{
    if (!ctx) return HIP_SAM3D_E_INVAL;
    int rc = cs3d_ensure_gs(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    return sam3d_cpu_gs_decoder_to_representation(ctx->cpu_gs, coords, feats_out, N,
                                                  xyz_out, dc_out,
                                                  scaling_out, rotation_out,
                                                  opacity_out) == 0
           ? HIP_SAM3D_E_OK : HIP_SAM3D_E_LOAD;
}

int hip_sam3d_slat_gs_info(hip_sam3d_ctx *ctx,
                            int *out_in_channels, int *out_out_channels,
                            int *out_num_gaussians)
{
    if (!ctx) return HIP_SAM3D_E_INVAL;
    int rc = cs3d_ensure_gs(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    if (out_in_channels)   *out_in_channels   = sam3d_cpu_gs_decoder_in_channels  (ctx->cpu_gs);
    if (out_out_channels)  *out_out_channels  = sam3d_cpu_gs_decoder_out_channels (ctx->cpu_gs);
    if (out_num_gaussians) *out_num_gaussians = sam3d_cpu_gs_decoder_num_gaussians(ctx->cpu_gs);
    return HIP_SAM3D_E_OK;
}

static int cs3d_gs_pack_ply_gpu_device(hip_sam3d_ctx *ctx,
                                       const sam3d_gs_decoder_model *m,
                                       hipDeviceptr_t d_coords,
                                       hipDeviceptr_t d_feats,
                                       int N,
                                       float **out_ply,
                                       int *out_total)
{
    if (!ctx || !m || !d_coords || !d_feats || N <= 0 || !out_ply || !out_total)
        return HIP_SAM3D_E_INVAL;
    hipFunction_t fn = NULL;
    if (hipModuleGetFunction(&fn, ctx->mod, "sam3d_gs_pack_ply_f32") != hipSuccess) {
        fprintf(stderr, "hip_sam3d: missing sam3d_gs_pack_ply_f32\n");
        return HIP_SAM3D_E_LOAD;
    }

    int C = m->out_channels;
    int G = m->num_gaussians;
    int stride = HIP_SAM3D_GS_STRIDE;
    int total = N * G;
    size_t ply_bytes = (size_t)total * stride * sizeof(float);
    hipDeviceptr_t d_ply = 0, d_perturb = 0;
    float *ply = NULL;
    int rc = HIP_SAM3D_E_LOAD;

    if (hipMalloc(&d_ply, ply_bytes) != hipSuccess)
        goto done;

    int has_perturb = (m->perturb_offset && m->offset_perturbation.data) ? 1 : 0;
    if (has_perturb &&
        cs3d_slat_io_upload_qtensor(&m->offset_perturbation,
                                    "gs.offset_perturbation",
                                    &d_perturb, NULL) != 0)
        goto done;

    int resolution = m->resolution;
    float voxel_size = m->voxel_size;
    float scaling_bias = m->scaling_bias;
    float opacity_bias = m->opacity_bias;
    int r_xyz0 = m->r_xyz[0];
    int r_dc0 = m->r_features_dc[0];
    int r_scl0 = m->r_scaling[0];
    int r_rot0 = m->r_rotation[0];
    int r_op0 = m->r_opacity[0];
    float lr_xyz = m->lr_xyz;
    float lr_features_dc = m->lr_features_dc;
    float lr_scaling = m->lr_scaling;
    float lr_rotation = m->lr_rotation;
    float lr_opacity = m->lr_opacity;
    void *args[] = { &d_ply, &d_coords, &d_feats, &d_perturb, &has_perturb,
                     &N, &C, &G, &stride, &resolution,
                     &voxel_size, &scaling_bias, &opacity_bias,
                     &r_xyz0, &r_dc0, &r_scl0, &r_rot0, &r_op0,
                     &lr_xyz, &lr_features_dc, &lr_scaling,
                     &lr_rotation, &lr_opacity };
    if (hipModuleLaunchKernel(fn, (total + 255) / 256, 1, 1,
                       256, 1, 1, 0, 0, args, NULL) != hipSuccess)
        goto done;
    ply = (float *)malloc(ply_bytes);
    if (!ply) goto done;
    if (hipMemcpyDtoH(ply, d_ply, ply_bytes) != hipSuccess)
        goto done;
    *out_ply = ply;
    *out_total = total;
    ply = NULL;
    rc = HIP_SAM3D_E_OK;

done:
    free(ply);
    if (d_ply) hipFree(d_ply);
    if (d_perturb) hipFree(d_perturb);
    return rc;
}

static int cs3d_gs_pack_ply_gpu(hip_sam3d_ctx *ctx,
                                const sam3d_gs_decoder_model *m,
                                const int32_t *coords,
                                const float *feats_out,
                                int N,
                                float **out_ply,
                                int *out_total)
{
    if (!ctx || !m || !coords || !feats_out || N <= 0 || !out_ply || !out_total)
        return HIP_SAM3D_E_INVAL;

    int C = m->out_channels;
    size_t coord_bytes = (size_t)N * 4 * sizeof(int32_t);
    size_t feat_bytes = (size_t)N * C * sizeof(float);
    hipDeviceptr_t d_coords = hip_upload_raw(coords, coord_bytes);
    hipDeviceptr_t d_feats = hip_upload_raw(feats_out, feat_bytes);
    if (!d_coords || !d_feats) {
        if (d_coords) hipFree(d_coords);
        if (d_feats) hipFree(d_feats);
        return HIP_SAM3D_E_LOAD;
    }

    int rc = cs3d_gs_pack_ply_gpu_device(ctx, m, d_coords, d_feats, N,
                                         out_ply, out_total);
    hipFree(d_coords);
    hipFree(d_feats);
    return rc;
}

int hip_sam3d_debug_slat_gs_pack_ply(hip_sam3d_ctx *ctx,
                                      const int32_t *coords,
                                      const float *feats_out, int N,
                                      float **out_ply, int *out_total)
{
    if (!ctx || !coords || !feats_out || N <= 0 || !out_ply || !out_total)
        return HIP_SAM3D_E_INVAL;
    int rc = cs3d_ensure_gs(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;
    sam3d_gs_decoder_model *gm =
        (sam3d_gs_decoder_model *)sam3d_cpu_gs_decoder_model(ctx->cpu_gs);
    return cs3d_gs_pack_ply_gpu(ctx, gm, coords, feats_out, N,
                                out_ply, out_total);
}

int hip_sam3d_run_slat_gs_decode(hip_sam3d_ctx *ctx) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_slat *slat = cs3d_active_slat(ctx);
    if (!slat->feats || !slat->coords) {
        fprintf(stderr, "hip_sam3d: slat_gs_decode needs SLAT tokens — "
                        "run slat_dit (or override) first\n");
        return HIP_SAM3D_E_NO_INPUT;
    }
    int rc = cs3d_ensure_gs(ctx);
    if (rc != HIP_SAM3D_E_OK) return rc;

    int N = slat->n;
    int G = sam3d_cpu_gs_decoder_num_gaussians(ctx->cpu_gs);
    int total = N * G;

    sam3d_gs_decoder_model *gm =
        (sam3d_gs_decoder_model *)sam3d_cpu_gs_decoder_model(ctx->cpu_gs);
    sp3d_tensor x = {0};
    x.coords = (int32_t *)slat->coords;
    x.feats = (float *)slat->feats;
    x.N = N;
    x.C = slat->c;
    x.batch_size = 1;

    hipDeviceptr_t src_d_coords = 0;
    hipDeviceptr_t src_d_feats = 0;
    if (slat == &ctx->slat_tokens &&
        ctx->d_slat_coords && ctx->d_slat_feats &&
        ctx->slat_tokens.n == N && ctx->slat_tokens.c == slat->c) {
        src_d_coords = ctx->d_slat_coords;
        src_d_feats = ctx->d_slat_feats;
    }
    if (cs3d_gs_transformer_gpu_forward(ctx, &x, gm,
                                        src_d_coords, src_d_feats,
                                        NULL) != 0)
        return HIP_SAM3D_E_LOAD;

    float *ply = NULL;
    int packed_total = 0;
    rc = cs3d_gs_pack_ply_gpu_device(ctx, gm,
                                     ctx->gpu_gs_head_ws.d_coords,
                                     ctx->gpu_gs_head_ws.d_out,
                                     N, &ply, &packed_total);
    if (rc != HIP_SAM3D_E_OK) {
        free(ply);
        return rc;
    }
    if (packed_total != total) {
        free(ply);
        return HIP_SAM3D_E_LOAD;
    }
    return cs3d_adopt_gaussians(ctx, ply, packed_total);
}

/* ===== read-back stubs ===== */

int hip_sam3d_get_dinov2_tokens(hip_sam3d_ctx *ctx, float *out,
                                 int *out_n, int *out_c) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_2d *src = cs3d_active_dinov2(ctx);
    if (!src->data) return HIP_SAM3D_E_NO_INPUT;
    if (out_n) *out_n = src->n;
    if (out_c) *out_c = src->c;
    if (out) memcpy(out, src->data, (size_t)src->n * src->c * sizeof(float));
    return HIP_SAM3D_E_OK;
}
int hip_sam3d_get_cond_tokens(hip_sam3d_ctx *ctx, float *out,
                               int *out_n, int *out_c) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_2d *src = cs3d_active_cond(ctx);
    if (!src->data) return HIP_SAM3D_E_NO_INPUT;
    if (out_n) *out_n = src->n;
    if (out_c) *out_c = src->c;
    if (out) memcpy(out, src->data, (size_t)src->n * src->c * sizeof(float));
    return HIP_SAM3D_E_OK;
}
int hip_sam3d_get_ss_latent(hip_sam3d_ctx *ctx, float *out, int *out_dims) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_nd *src = cs3d_active_ss_latent(ctx);
    if (!src->data) return HIP_SAM3D_E_NO_INPUT;
    size_t numel = 1;
    for (int i = 0; i < src->ndim; i++) numel *= (size_t)src->dims[i];
    if (out_dims) for (int i = 0; i < src->ndim; i++) out_dims[i] = src->dims[i];
    if (out) memcpy(out, src->data, numel * sizeof(float));
    return HIP_SAM3D_E_OK;
}
int hip_sam3d_get_occupancy(hip_sam3d_ctx *ctx, float *out, int *out_dims) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_nd *src = cs3d_active_occupancy(ctx);
    if (!src->data) return HIP_SAM3D_E_NO_INPUT;
    size_t numel = 1;
    for (int i = 0; i < src->ndim; i++) numel *= (size_t)src->dims[i];
    if (out_dims) for (int i = 0; i < src->ndim; i++) out_dims[i] = src->dims[i];
    if (out) memcpy(out, src->data, numel * sizeof(float));
    return HIP_SAM3D_E_OK;
}
int hip_sam3d_get_slat_tokens(hip_sam3d_ctx *ctx, float *out_feats,
                               int32_t *out_coords, int *out_n, int *out_c) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    const cs3d_host_slat *src = ctx->ovr_slat.feats
                              ? &ctx->ovr_slat
                              : &ctx->slat_tokens;
    if (!src->feats) return HIP_SAM3D_E_NO_INPUT;
    if (out_n) *out_n = src->n;
    if (out_c) *out_c = src->c;
    if (out_feats)
        memcpy(out_feats, src->feats, (size_t)src->n * src->c * sizeof(float));
    if (out_coords)
        memcpy(out_coords, src->coords, (size_t)src->n * 4 * sizeof(int32_t));
    return HIP_SAM3D_E_OK;
}
int hip_sam3d_get_gaussians(hip_sam3d_ctx *ctx, float *out, int *out_n) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    if (!ctx->gaussians.data) return HIP_SAM3D_E_NO_INPUT;
    if (out_n) *out_n = ctx->gaussians.n;
    if (out)
        memcpy(out, ctx->gaussians.data,
               (size_t)ctx->gaussians.n * HIP_SAM3D_GS_STRIDE * sizeof(float));
    return HIP_SAM3D_E_OK;
}

/* ===== debug overrides — host-side store, consumed when the
 * downstream stage's kernel is wired up. ===== */

static int cs3d_set_2d(cs3d_host_2d *dst, const float *src, int n, int c) {
    cs3d_free_2d(dst);
    if (!src || n <= 0 || c <= 0) return HIP_SAM3D_E_INVAL;
    size_t bytes = (size_t)n * c * sizeof(float);
    dst->data = (float *)malloc(bytes);
    if (!dst->data) return HIP_SAM3D_E_LOAD;
    memcpy(dst->data, src, bytes);
    dst->n = n; dst->c = c;
    return HIP_SAM3D_E_OK;
}

static int cs3d_set_nd(cs3d_host_nd *dst, const float *src,
                        const int *dims, int ndim) {
    cs3d_free_nd(dst);
    if (!src || !dims || ndim <= 0 || ndim > 4) return HIP_SAM3D_E_INVAL;
    size_t numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) return HIP_SAM3D_E_INVAL;
        numel *= (size_t)dims[i];
        dst->dims[i] = dims[i];
    }
    dst->ndim = ndim;
    dst->data = (float *)malloc(numel * sizeof(float));
    if (!dst->data) return HIP_SAM3D_E_LOAD;
    memcpy(dst->data, src, numel * sizeof(float));
    return HIP_SAM3D_E_OK;
}

int hip_sam3d_debug_override_dinov2(hip_sam3d_ctx *ctx, const float *tokens,
                                     int n, int c) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    return cs3d_set_2d(&ctx->ovr_dinov2, tokens, n, c);
}
int hip_sam3d_debug_override_cond(hip_sam3d_ctx *ctx, const float *tokens,
                                   int n, int c) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    return cs3d_set_2d(&ctx->ovr_cond, tokens, n, c);
}
int hip_sam3d_debug_override_ss_latent(hip_sam3d_ctx *ctx, const float *latent,
                                        const int *dims) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    return cs3d_set_nd(&ctx->ovr_ss_latent, latent, dims, 4);
}
int hip_sam3d_debug_override_occupancy(hip_sam3d_ctx *ctx, const float *occ,
                                        const int *dims) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    return cs3d_set_nd(&ctx->ovr_occupancy, occ, dims, 3);
}
int hip_sam3d_debug_override_slat(hip_sam3d_ctx *ctx, const float *feats,
                                   const int32_t *coords, int n, int c) {
    if (!ctx) return HIP_SAM3D_E_INVAL;
    cs3d_free_slat(&ctx->ovr_slat);
    if (!feats || !coords || n <= 0 || c <= 0) return HIP_SAM3D_E_INVAL;
    size_t f_bytes = (size_t)n * c * sizeof(float);
    size_t k_bytes = (size_t)n * 4 * sizeof(int32_t);
    ctx->ovr_slat.feats  = (float *)malloc(f_bytes);
    ctx->ovr_slat.coords = (int32_t *)malloc(k_bytes);
    if (!ctx->ovr_slat.feats || !ctx->ovr_slat.coords) {
        cs3d_free_slat(&ctx->ovr_slat);
        return HIP_SAM3D_E_LOAD;
    }
    memcpy(ctx->ovr_slat.feats, feats, f_bytes);
    memcpy(ctx->ovr_slat.coords, coords, k_bytes);
    ctx->ovr_slat.n = n; ctx->ovr_slat.c = c;
    return HIP_SAM3D_E_OK;
}
