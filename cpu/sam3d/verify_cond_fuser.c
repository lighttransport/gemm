/*
 * verify_cond_fuser — diff PointPatchEmbed + EmbedderFuser output
 * against /tmp/sam3d_ref/*.npy. In v1 we run only the "full" pass
 * (no cropped variants), so the reference file we compare against
 * is cond_tokens_full.npy (or ppe_tokens.npy for the point branch
 * only). If neither is present we do a smoke run.
 *
 * Usage:
 *   verify_cond_fuser --ckpt <pipeline.yaml> --refdir /tmp/sam3d_ref
 *                     [--safetensors-dir <dir>] [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR:
 *   pointmap.npy             (H, W, 3)             f32  (optional)
 *   dinov2_tokens.npy        (2, 1+Np, D) or (1, …) f32
 *   cond_tokens_full.npy     (N_cond, D_out)       f32  (optional target)
 *   ppe_tokens.npy           (Nwin, D_ppe)         f32  (optional; compares
 *                                                        only the point branch)
 */

#include "sam3d_runner.h"
#include "verify_npy.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *ckpt = NULL, *refdir = NULL, *sft_dir = NULL;
    int verbose = 0, n_threads = 1;
    float threshold = 5e-3f;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt    = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc)       n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v"))                     verbose = 1;
        else if (!strcmp(argv[i], "--use-ref-inputs"))       { /* implied */ }
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!ckpt || !refdir) {
        fprintf(stderr, "Usage: %s --ckpt <pipeline.yaml> --refdir <dir> "
                        "[--safetensors-dir <dir>] [-t N] [-v]\n", argv[0]);
        return 2;
    }

    char path[1024];
    int nd = 0, dims[8] = {0}, is_f32 = 0;

    /* --- Inject dinov2 tokens from ref (required). --- */
    snprintf(path, sizeof(path), "%s/dinov2_tokens.npy", refdir);
    float *dino = (float *)verify_read_npy(path, &nd, dims, &is_f32);
    if (!dino || !is_f32) {
        fprintf(stderr, "missing/bad %s (required)\n", path);
        free(dino); return 3;
    }
    int dino_branches = 1, dino_ntok = 0, dino_dim = 0;
    if (nd == 3) {
        dino_branches = dims[0]; dino_ntok = dims[1]; dino_dim = dims[2];
    } else if (nd == 2) {
        dino_ntok = dims[0]; dino_dim = dims[1];
    } else {
        fprintf(stderr, "bad dinov2_tokens rank=%d\n", nd);
        free(dino); return 3;
    }
    fprintf(stderr, "[verify_cond_fuser] dino: branches=%d n_tok=%d dim=%d\n",
            dino_branches, dino_ntok, dino_dim);

    /* --- Pointmap (optional — enables point branch). --- */
    snprintf(path, sizeof(path), "%s/pointmap.npy", refdir);
    float *pmap = (float *)verify_read_npy(path, &nd, dims, &is_f32);
    int ph = 0, pw = 0;
    if (pmap && is_f32 && nd == 3 && dims[2] == 3) {
        ph = dims[0]; pw = dims[1];
        fprintf(stderr, "[verify_cond_fuser] pointmap: %dx%d\n", ph, pw);
    } else if (pmap) {
        fprintf(stderr, "[verify_cond_fuser] ignoring pointmap (bad shape)\n");
        free(pmap); pmap = NULL;
    } else {
        fprintf(stderr, "[verify_cond_fuser] no pointmap.npy — skipping point branch\n");
    }

    /* --- Optional targets. --- */
    snprintf(path, sizeof(path), "%s/cond_tokens_full.npy", refdir);
    int tdims[8] = {0}, t_nd = 0, t_f32 = 0;
    float *cond_ref = (float *)verify_read_npy(path, &t_nd, tdims, &t_f32);
    if (cond_ref && !t_f32) { free(cond_ref); cond_ref = NULL; }
    int cond_n = 0, cond_c = 0;
    if (cond_ref && t_nd == 2) { cond_n = tdims[0]; cond_c = tdims[1]; }

    /* --- Build runner. --- */
    sam3d_config cfg = {
        .pipeline_yaml = ckpt,
        .safetensors_dir = sft_dir,
        .seed = 42,
        .n_threads = n_threads,
        .verbose = verbose,
    };
    sam3d_ctx *ctx = sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_create failed\n"); free(dino); free(pmap); free(cond_ref); return 5; }

    /* Inject dino tokens. If ref has 2 branches (image + mask), pass all;
     * otherwise treat as a single branch. Shape passed to the override
     * is (n_total_tokens, dim). */
    int total_ntok = dino_branches * dino_ntok;
    if (sam3d_debug_override_dinov2(ctx, dino, total_ntok, dino_dim) != 0) {
        fprintf(stderr, "debug_override_dinov2 failed\n");
        sam3d_destroy(ctx); free(dino); free(pmap); free(cond_ref);
        return 5;
    }
    free(dino);

    if (pmap) {
        if (sam3d_set_pointmap(ctx, pmap, pw, ph) != 0) {
            fprintf(stderr, "set_pointmap failed\n");
            sam3d_destroy(ctx); free(pmap); free(cond_ref);
            return 5;
        }
        free(pmap);
    }

    /* --- Run. --- */
    int rc = sam3d_run_cond_fuser(ctx);
    if (rc != 0) {
        fprintf(stderr, "sam3d_run_cond_fuser rc=%d\n", rc);
        sam3d_destroy(ctx); free(cond_ref); return 6;
    }

    int on = 0, oc = 0;
    sam3d_get_cond_tokens(ctx, NULL, &on, &oc);
    float *ours = (float *)malloc((size_t)on * oc * sizeof(float));
    sam3d_get_cond_tokens(ctx, ours, &on, &oc);
    fprintf(stderr, "[verify_cond_fuser] ours: n_tok=%d dim=%d\n", on, oc);

    int rc_out = 0;
    if (cond_ref) {
        if (oc != cond_c) {
            fprintf(stderr, "dim mismatch: ours=%d ref=%d\n", oc, cond_c);
            rc_out = 7;
        } else {
            int n = (on < cond_n ? on : cond_n) * oc;
            double mean_abs = 0.0;
            float mx = verify_max_abs(ours, cond_ref, n, &mean_abs);
            fprintf(stderr, "[verify_cond_fuser] n=%d dim=%d  "
                            "max_abs=%.6e mean_abs=%.6e (threshold %.1e)\n",
                    n / oc, oc, mx, mean_abs, threshold);
            rc_out = (mx < threshold) ? 0 : 1;
        }
    } else {
        fprintf(stderr, "[verify_cond_fuser] no cond_tokens_full.npy — smoke only\n");
    }

    free(ours); free(cond_ref);
    sam3d_destroy(ctx);
    return rc_out;
}
