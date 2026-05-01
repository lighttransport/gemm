/* DINOv2-L/14+reg device-side weight buffers — Phase 1b.0 scaffold.
 *
 * Single-header. Define HIP_SAM3D_DINOV2_GPU_IMPLEMENTATION in
 * exactly one translation unit before including. Owns one hipDeviceptr_t
 * per F32 weight tensor; the upcoming Phase 1b.1–1b.7 NVRTC kernels
 * read directly from these buffers.
 *
 * sam3d_dinov2.safetensors is F32 throughout (verified at port time),
 * so each load is a direct safetensors mmap → host_ptr → hipMemcpyHtoD.
 * No dequant pass.
 */

#ifndef HIP_SAM3D_DINOV2_GPU_H_
#define HIP_SAM3D_DINOV2_GPU_H_

#include <stddef.h>
#include "../rocew.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    hipDeviceptr_t norm1_w,  norm1_b;        /* [dim]                   */
    hipDeviceptr_t qkv_w,    qkv_b;          /* [3*dim, dim], [3*dim]   */
    hipDeviceptr_t proj_w,   proj_b;         /* [dim, dim],   [dim]     */
    hipDeviceptr_t ls1;                      /* [dim]                   */
    hipDeviceptr_t norm2_w,  norm2_b;        /* [dim]                   */
    hipDeviceptr_t fc1_w,    fc1_b;          /* [ffn, dim],   [ffn]     */
    hipDeviceptr_t fc2_w,    fc2_b;          /* [dim, ffn],   [dim]     */
    hipDeviceptr_t ls2;                      /* [dim]                   */
} cs3d_dinov2_gpu_block;

typedef struct {
    /* Geometry, derived from tensor shapes at load time. */
    int dim;
    int n_heads, head_dim;
    int n_blocks;
    int ffn_hidden;
    int patch_size;
    int image_size;
    int grid_h, grid_w, n_patches;
    int n_register;
    int orig_grid;                        /* sqrt(pos_embed - 1)     */
    int n_tokens;                         /* 1 + n_register + n_pat. */
    float ln_eps;                         /* 1e-6 (DINOv2 default)   */

    /* Top-level weights. */
    hipDeviceptr_t cls_token;                /* [dim]                   */
    hipDeviceptr_t register_tokens;          /* [n_register, dim] | 0   */
    hipDeviceptr_t pos_embed;                /* [1+orig_grid^2, dim]    */
    hipDeviceptr_t patch_w;                  /* [dim, 3, ps, ps]        */
    hipDeviceptr_t patch_b;                  /* [dim]                   */
    hipDeviceptr_t norm_w, norm_b;           /* final LN, [dim]         */

    cs3d_dinov2_gpu_block *blocks;        /* [n_blocks]              */

    size_t total_bytes;                   /* sum of all device allocs */
    int    loaded;
} cs3d_dinov2_gpu;

/* Load all DINOv2-L weights from a safetensors path into freshly-
 * allocated device buffers. Returns 0 on success, <0 on failure (any
 * partially-allocated buffers are released). The caller-supplied `g`
 * is zeroed first. `verbose` ≥ 1 prints geometry + total bytes. */
int  cs3d_dinov2_gpu_load(cs3d_dinov2_gpu *g, const char *st_path, int verbose);

/* Release every hipDeviceptr_t held by `g` and zero its fields. Safe to
 * call on an unloaded (zeroed) struct. */
void cs3d_dinov2_gpu_free(cs3d_dinov2_gpu *g);

#ifdef __cplusplus
}
#endif

#endif /* HIP_SAM3D_DINOV2_GPU_H_ */

/* ============================ implementation ============================ */
#ifdef HIP_SAM3D_DINOV2_GPU_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* SAFETENSORS_IMPLEMENTATION must be defined exactly once across the
 * program. We assume the including TU has already done so (or will
 * include this header alongside something that does). */
#include "../../common/safetensors.h"
#include "../hip_runner_common.h"

/* Locate `name`, sanity-check F32 dtype, upload to device. Returns 0
 * on missing-but-optional (caller decides; pass require=0). */
static int cs3d_dv2g_upload_named(st_context *st, const char *name,
                                  size_t expect_n, int require,
                                  hipDeviceptr_t *out_d, size_t *out_bytes)
{
    int i = safetensors_find(st, name);
    if (i < 0) {
        if (require) {
            fprintf(stderr, "cs3d_dinov2_gpu: missing required tensor %s\n", name);
            return -1;
        }
        *out_d = 0;
        return 0;
    }
    if (strcmp(safetensors_dtype(st, i), "F32") != 0) {
        fprintf(stderr, "cs3d_dinov2_gpu: %s not F32 (%s)\n",
                name, safetensors_dtype(st, i));
        return -1;
    }
    size_t nb = safetensors_nbytes(st, i);
    if (expect_n && nb / 4 != expect_n) {
        fprintf(stderr, "cs3d_dinov2_gpu: %s size mismatch: got %zu f32, expected %zu\n",
                name, nb / 4, expect_n);
        return -1;
    }
    hipDeviceptr_t d = hip_upload_raw(safetensors_data(st, i), nb);
    if (!d) {
        fprintf(stderr, "cs3d_dinov2_gpu: hipMalloc failed for %s (%zu bytes)\n",
                name, nb);
        return -1;
    }
    *out_d = d;
    *out_bytes += nb;
    return 0;
}

/* Convenience wrapper for required tensors. */
static int cs3d_dv2g_up_req(st_context *st, const char *name,
                            size_t expect_n,
                            hipDeviceptr_t *out_d, size_t *out_bytes)
{
    return cs3d_dv2g_upload_named(st, name, expect_n, 1, out_d, out_bytes);
}

int cs3d_dinov2_gpu_load(cs3d_dinov2_gpu *g, const char *st_path, int verbose)
{
    if (!g || !st_path) return -1;
    memset(g, 0, sizeof(*g));

    st_context *st = safetensors_open(st_path);
    if (!st) {
        fprintf(stderr, "cs3d_dinov2_gpu: safetensors_open(%s) failed\n", st_path);
        return -1;
    }

    /* ---- Geometry detection (mirrors common/dinov2.h). ---- */
    int dim = 1024, head_dim = 64, patch_size = 14, image_size = 518;
    int n_blocks = 0, ffn_hidden = 4096;
    int n_register = 0, orig_grid = 37;

    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const uint64_t *sh = safetensors_shape(st, i);
        int nd = safetensors_ndims(st, i);

        if (!strcmp(nm, "patch_embed.proj.weight") && nd == 4) {
            dim = (int)sh[0];
            patch_size = (int)sh[2];
        } else if (!strcmp(nm, "register_tokens")) {
            n_register = (nd == 3) ? (int)sh[1] : (int)sh[0];
        } else if (!strcmp(nm, "pos_embed")) {
            int n_minus_cls = (nd == 3) ? (int)sh[1] - 1 : (int)sh[0] - 1;
            int gv = (int)(sqrtf((float)n_minus_cls) + 0.5f);
            if (gv * gv == n_minus_cls) orig_grid = gv;
        } else if (!strcmp(nm, "blocks.0.mlp.fc1.weight")) {
            ffn_hidden = (int)sh[0];
        }
        const char *bp = strstr(nm, "blocks.");
        if (bp) {
            bp += 7;
            int blk = 0;
            while (*bp >= '0' && *bp <= '9') { blk = blk * 10 + (*bp - '0'); bp++; }
            if (blk + 1 > n_blocks) n_blocks = blk + 1;
        }
    }
    int n_heads = dim / head_dim;
    int grid = image_size / patch_size;
    int n_patches = grid * grid;
    int n_tokens  = 1 + n_register + n_patches;

    g->dim = dim;
    g->n_heads = n_heads;
    g->head_dim = head_dim;
    g->n_blocks = n_blocks;
    g->ffn_hidden = ffn_hidden;
    g->patch_size = patch_size;
    g->image_size = image_size;
    g->grid_h = grid;
    g->grid_w = grid;
    g->n_patches = n_patches;
    g->n_register = n_register;
    g->orig_grid = orig_grid;
    g->n_tokens = n_tokens;
    g->ln_eps = 1e-6f;

    if (verbose >= 1) {
        fprintf(stderr,
            "cs3d_dinov2_gpu: dim=%d heads=%d head_dim=%d blocks=%d ffn=%d "
            "patch=%d image=%d grid=%dx%d orig=%d n_reg=%d n_tokens=%d\n",
            dim, n_heads, head_dim, n_blocks, ffn_hidden,
            patch_size, image_size, grid, grid, orig_grid, n_register, n_tokens);
    }

    g->blocks = (cs3d_dinov2_gpu_block *)calloc((size_t)n_blocks,
                                                sizeof(cs3d_dinov2_gpu_block));
    if (!g->blocks) {
        safetensors_close(st);
        return -1;
    }

    size_t bytes = 0;
    int rc = 0;

    /* ---- Top-level. ---- */
    rc |= cs3d_dv2g_up_req(st, "cls_token",
                           (size_t)dim, &g->cls_token, &bytes);
    rc |= cs3d_dv2g_upload_named(st, "register_tokens",
                                 n_register ? (size_t)n_register * dim : 0,
                                 0 /* optional */,
                                 &g->register_tokens, &bytes);
    rc |= cs3d_dv2g_up_req(st, "pos_embed",
                           (size_t)(1 + orig_grid * orig_grid) * dim,
                           &g->pos_embed, &bytes);
    rc |= cs3d_dv2g_up_req(st, "patch_embed.proj.weight",
                           (size_t)dim * 3 * patch_size * patch_size,
                           &g->patch_w, &bytes);
    rc |= cs3d_dv2g_up_req(st, "patch_embed.proj.bias",
                           (size_t)dim, &g->patch_b, &bytes);
    rc |= cs3d_dv2g_up_req(st, "norm.weight",
                           (size_t)dim, &g->norm_w, &bytes);
    rc |= cs3d_dv2g_up_req(st, "norm.bias",
                           (size_t)dim, &g->norm_b, &bytes);

    /* ---- Per-block. ---- */
    for (int L = 0; L < n_blocks && rc == 0; L++) {
        cs3d_dinov2_gpu_block *b = &g->blocks[L];
        char nm[160];
        #define K(suf) (snprintf(nm, sizeof(nm), "blocks.%d." suf, L), nm)
        rc |= cs3d_dv2g_up_req(st, K("norm1.weight"),     (size_t)dim,            &b->norm1_w, &bytes);
        rc |= cs3d_dv2g_up_req(st, K("norm1.bias"),       (size_t)dim,            &b->norm1_b, &bytes);
        rc |= cs3d_dv2g_up_req(st, K("attn.qkv.weight"),  (size_t)3*dim*dim,      &b->qkv_w,   &bytes);
        rc |= cs3d_dv2g_up_req(st, K("attn.qkv.bias"),    (size_t)3*dim,          &b->qkv_b,   &bytes);
        rc |= cs3d_dv2g_up_req(st, K("attn.proj.weight"), (size_t)dim*dim,        &b->proj_w,  &bytes);
        rc |= cs3d_dv2g_up_req(st, K("attn.proj.bias"),   (size_t)dim,            &b->proj_b,  &bytes);
        rc |= cs3d_dv2g_up_req(st, K("ls1.gamma"),        (size_t)dim,            &b->ls1,     &bytes);
        rc |= cs3d_dv2g_up_req(st, K("norm2.weight"),     (size_t)dim,            &b->norm2_w, &bytes);
        rc |= cs3d_dv2g_up_req(st, K("norm2.bias"),       (size_t)dim,            &b->norm2_b, &bytes);
        rc |= cs3d_dv2g_up_req(st, K("mlp.fc1.weight"),   (size_t)ffn_hidden*dim, &b->fc1_w,   &bytes);
        rc |= cs3d_dv2g_up_req(st, K("mlp.fc1.bias"),     (size_t)ffn_hidden,     &b->fc1_b,   &bytes);
        rc |= cs3d_dv2g_up_req(st, K("mlp.fc2.weight"),   (size_t)dim*ffn_hidden, &b->fc2_w,   &bytes);
        rc |= cs3d_dv2g_up_req(st, K("mlp.fc2.bias"),     (size_t)dim,            &b->fc2_b,   &bytes);
        rc |= cs3d_dv2g_up_req(st, K("ls2.gamma"),        (size_t)dim,            &b->ls2,     &bytes);
        #undef K
    }

    safetensors_close(st);

    if (rc != 0) {
        cs3d_dinov2_gpu_free(g);
        return -1;
    }

    g->total_bytes = bytes;
    g->loaded = 1;

    if (verbose >= 1) {
        fprintf(stderr, "cs3d_dinov2_gpu: uploaded %.1f MiB across %d blocks\n",
                (double)bytes / (1024.0 * 1024.0), n_blocks);
    }
    return 0;
}

void cs3d_dinov2_gpu_free(cs3d_dinov2_gpu *g)
{
    if (!g) return;
    #define FREE_D(p) do { if (p) { hipFree(p); (p) = 0; } } while (0)
    FREE_D(g->cls_token);
    FREE_D(g->register_tokens);
    FREE_D(g->pos_embed);
    FREE_D(g->patch_w);
    FREE_D(g->patch_b);
    FREE_D(g->norm_w);
    FREE_D(g->norm_b);
    if (g->blocks) {
        for (int L = 0; L < g->n_blocks; L++) {
            cs3d_dinov2_gpu_block *b = &g->blocks[L];
            FREE_D(b->norm1_w); FREE_D(b->norm1_b);
            FREE_D(b->qkv_w);   FREE_D(b->qkv_b);
            FREE_D(b->proj_w);  FREE_D(b->proj_b);
            FREE_D(b->ls1);
            FREE_D(b->norm2_w); FREE_D(b->norm2_b);
            FREE_D(b->fc1_w);   FREE_D(b->fc1_b);
            FREE_D(b->fc2_w);   FREE_D(b->fc2_b);
            FREE_D(b->ls2);
        }
        free(g->blocks);
        g->blocks = NULL;
    }
    #undef FREE_D
    memset(g, 0, sizeof(*g));
}

#endif /* HIP_SAM3D_DINOV2_GPU_IMPLEMENTATION */
