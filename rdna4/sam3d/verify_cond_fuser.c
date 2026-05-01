/*
 * verify_cond_fuser (CUDA) — diff PointPatchEmbed + EmbedderFuser
 * output against /tmp/sam3d_ref/cond_tokens_full.npy. dinov2 tokens
 * are injected via hip_sam3d_debug_override_dinov2 so this stage's
 * drift is isolated from upstream encoder drift.
 *
 * Usage:
 *   verify_cond_fuser --safetensors-dir DIR --refdir /tmp/sam3d_ref
 *                     [--threshold F] [-v]
 */

#include "hip_sam3d_runner.h"
#include "../../common/npy_io.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *yaml = NULL, *refdir = NULL;
    int verbose = 0;
    float threshold = 5e-3f;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir   = argv[++i];
        else if (!strcmp(a, "--pipeline-yaml")   && i+1 < argc) yaml      = argv[++i];
        else if (!strcmp(a, "--refdir")          && i+1 < argc) refdir    = argv[++i];
        else if (!strcmp(a, "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-v"))                              verbose   = 1;
        else if (!strcmp(a, "--use-ref-inputs")) { /* implied */ }
        else { fprintf(stderr, "unknown arg: %s\n", a); return 2; }
    }
    if ((!sft_dir && !yaml) || !refdir) {
        fprintf(stderr,
                "Usage: %s (--safetensors-dir DIR | --pipeline-yaml YAML) "
                "--refdir DIR [--threshold F] [-v]\n", argv[0]);
        return 2;
    }

    char path[1024];
    int nd = 0, dims[8] = {0}, is_f32 = 0;

    /* dinov2 tokens (required). */
    snprintf(path, sizeof(path), "%s/dinov2_tokens.npy", refdir);
    float *dino = (float *)npy_load(path, &nd, dims, &is_f32);
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
    fprintf(stderr, "[verify_cond_fuser.cuda] dino: branches=%d n_tok=%d dim=%d\n",
            dino_branches, dino_ntok, dino_dim);

    /* pointmap (optional — enables point branch). */
    snprintf(path, sizeof(path), "%s/pointmap.npy", refdir);
    float *pmap = (float *)npy_load(path, &nd, dims, &is_f32);
    int ph = 0, pw = 0;
    if (pmap && is_f32 && nd == 3 && dims[2] == 3) {
        ph = dims[0]; pw = dims[1];
        fprintf(stderr, "[verify_cond_fuser.cuda] pointmap: %dx%d\n", ph, pw);
    } else if (pmap) {
        fprintf(stderr, "[verify_cond_fuser.cuda] ignoring pointmap (bad shape)\n");
        free(pmap); pmap = NULL;
    } else {
        fprintf(stderr, "[verify_cond_fuser.cuda] no pointmap.npy — skipping point branch\n");
    }

    /* Optional target. */
    snprintf(path, sizeof(path), "%s/cond_tokens_full.npy", refdir);
    int t_nd = 0, t_dims[8] = {0}, t_f32 = 0;
    float *cond_ref = (float *)npy_load(path, &t_nd, t_dims, &t_f32);
    if (cond_ref && !t_f32) { free(cond_ref); cond_ref = NULL; }
    int cond_n = 0, cond_c = 0;
    if (cond_ref && t_nd == 2) { cond_n = t_dims[0]; cond_c = t_dims[1]; }

    hip_sam3d_config cfg = {0};
    cfg.safetensors_dir = sft_dir;
    cfg.pipeline_yaml   = yaml;
    cfg.verbose         = verbose;
    cfg.precision       = "fp16";
    cfg.seed            = 42;
    hip_sam3d_ctx *ctx = hip_sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "hip_sam3d_create failed\n");
        free(dino); free(pmap); free(cond_ref); return 5; }

    int total_ntok = dino_branches * dino_ntok;
    if (hip_sam3d_debug_override_dinov2(ctx, dino, total_ntok, dino_dim) != 0) {
        fprintf(stderr, "debug_override_dinov2 failed\n");
        hip_sam3d_destroy(ctx); free(dino); free(pmap); free(cond_ref);
        return 5;
    }
    free(dino);

    if (pmap) {
        if (hip_sam3d_set_pointmap(ctx, pmap, pw, ph) != 0) {
            fprintf(stderr, "set_pointmap failed\n");
            hip_sam3d_destroy(ctx); free(pmap); free(cond_ref);
            return 5;
        }
        free(pmap);
    }

    int rc = hip_sam3d_run_cond_fuser(ctx);
    if (rc != 0) {
        fprintf(stderr, "hip_sam3d_run_cond_fuser rc=%d\n", rc);
        hip_sam3d_destroy(ctx); free(cond_ref); return 6;
    }

    int on = 0, oc = 0;
    hip_sam3d_get_cond_tokens(ctx, NULL, &on, &oc);
    float *ours = (float *)malloc((size_t)on * oc * sizeof(float));
    hip_sam3d_get_cond_tokens(ctx, ours, &on, &oc);
    fprintf(stderr, "[verify_cond_fuser.cuda] ours: n_tok=%d dim=%d\n", on, oc);

    int rc_out = 0;
    if (cond_ref) {
        if (oc != cond_c) {
            fprintf(stderr, "dim mismatch: ours=%d ref=%d\n", oc, cond_c);
            rc_out = 7;
        } else {
            int n = (on < cond_n ? on : cond_n) * oc;
            double mean_abs = 0.0;
            float mx = npy_max_abs_f32(ours, cond_ref, n, &mean_abs);
            fprintf(stderr, "[verify_cond_fuser.cuda] n=%d dim=%d  "
                            "max_abs=%.6e mean_abs=%.6e (threshold %.1e)\n",
                    n / oc, oc, mx, mean_abs, threshold);
            rc_out = (mx < threshold) ? 0 : 1;
        }
    } else {
        fprintf(stderr, "[verify_cond_fuser.cuda] no cond_tokens_full.npy — smoke only\n");
    }

    free(ours); free(cond_ref);
    hip_sam3d_destroy(ctx);
    return rc_out;
}
