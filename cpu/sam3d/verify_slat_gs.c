/*
 * verify_slat_gs — stage-3 SLAT → 3D-Gaussian decoder verify.
 *
 * Step 8a (default): loader smoke-test — opens sam3d_slat_gs_decoder.safetensors
 *                    and confirms architecture / shape summary.
 * Step 8b (--refdir): additionally runs the transformer forward on the
 *                     input sparse tensor from $REFDIR/slat_gs_in_{coords,feats}.npy
 *                     and diffs against:
 *                       slat_gs_out_feats.npy     [N, 448]
 *                       slat_gs_rep_xyz.npy       [N*32, 3]
 *                       slat_gs_rep_dc.npy        [N*32, 1, 3]
 *                       slat_gs_rep_scaling.npy   [N*32, 3]
 *                       slat_gs_rep_rotation.npy  [N*32, 4]
 *                       slat_gs_rep_opacity.npy   [N*32, 1]
 *
 * Usage:
 *   verify_slat_gs --ckpt <pipeline.yaml> [--safetensors-dir <dir>]
 *                  [--refdir <dir>] [-v]
 */
#include "safetensors.h"         /* declarations only; impl in sam3d_runner.c */
#include "sam3d_gs_decoder.h"    /* impl linked in from sam3d_runner.c       */

#include "sam3d_runner.h"
#include "verify_npy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int diff_and_report(const char *label, const float *a, const float *b,
                           int n, float threshold) {
    double mean = 0.0;
    float max_abs = verify_max_abs(a, b, n, &mean);
    int ok = (max_abs <= threshold);
    fprintf(stderr, "[verify_slat_gs] %-30s max_abs=%.4g mean_abs=%.4g n=%d %s\n",
            label, max_abs, mean, n, ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char **argv)
{
    const char *ckpt = NULL, *refdir = NULL, *sft_dir = NULL;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt    = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "-v"))                     verbose = 1;
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!ckpt) {
        fprintf(stderr, "Usage: %s --ckpt <pipeline.yaml> "
                        "[--safetensors-dir <dir>] [--refdir <dir>] [-v]\n", argv[0]);
        return 2;
    }

    sam3d_config cfg = {
        .pipeline_yaml = ckpt,
        .safetensors_dir = sft_dir,
        .seed = 42,
        .verbose = verbose,
    };
    sam3d_ctx *ctx = sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_create failed\n"); return 5; }

    char path[1280];
    snprintf(path, sizeof(path), "%s/sam3d_slat_gs_decoder.safetensors",
             sam3d_safetensors_dir(ctx));
    fprintf(stderr, "[verify_slat_gs] loading %s\n", path);

    sam3d_gs_decoder_model *m = sam3d_gs_decoder_load_safetensors(path);
    if (!m) {
        fprintf(stderr, "[verify_slat_gs] loader failed\n");
        sam3d_destroy(ctx);
        return 3;
    }
    fprintf(stderr,
            "[verify_slat_gs] loaded OK: dim=%d heads=%d head_dim=%d "
            "blocks=%d in=%d out=%d num_gaussians=%d window=%d\n",
            m->dim, m->n_heads, m->head_dim, m->n_blocks,
            m->in_channels, m->out_channels,
            m->num_gaussians, m->window_size);

    /* Sanity: block 0 attention qkv + top-level out shape. */
    if (m->n_blocks > 0) {
        fprintf(stderr,
                "[verify_slat_gs] block[0].attn_qkv_w shape=(%llu,%llu) dtype=%u\n",
                (unsigned long long)m->blocks[0].attn_qkv_w.dims[0],
                (unsigned long long)m->blocks[0].attn_qkv_w.dims[1],
                m->blocks[0].attn_qkv_w.type);
    }
    fprintf(stderr,
            "[verify_slat_gs] out_layer.weight shape=(%d,%d)\n",
            m->out_w.n_rows, m->out_w.n_cols);
    fprintf(stderr,
            "[verify_slat_gs] layout xyz=[%d,%d) dc=[%d,%d) scl=[%d,%d) "
            "rot=[%d,%d) op=[%d,%d)\n",
            m->r_xyz[0], m->r_xyz[1],
            m->r_features_dc[0], m->r_features_dc[1],
            m->r_scaling[0], m->r_scaling[1],
            m->r_rotation[0], m->r_rotation[1],
            m->r_opacity[0], m->r_opacity[1]);

    int rc = 0;

    if (refdir) {
        char pbuf[1536];
        int nd, dims[8], is_f32;

        /* Load input coords + feats. */
        snprintf(pbuf, sizeof(pbuf), "%s/slat_gs_in_coords.npy", refdir);
        int32_t *coords = (int32_t *)verify_read_npy(pbuf, &nd, dims, &is_f32);
        if (!coords) { fprintf(stderr, "cannot read %s\n", pbuf); rc = 6; goto done; }
        if (nd != 2 || dims[1] != 4) {
            fprintf(stderr, "unexpected coords shape\n"); rc = 6; goto done;
        }
        int N = dims[0];

        snprintf(pbuf, sizeof(pbuf), "%s/slat_gs_in_feats.npy", refdir);
        float *feats = (float *)verify_read_npy(pbuf, &nd, dims, &is_f32);
        if (!feats || !is_f32 || nd != 2 || dims[0] != N || dims[1] != m->in_channels) {
            fprintf(stderr, "unexpected feats shape\n"); rc = 6; goto done;
        }

        sp3d_tensor *x = sp3d_create(coords, feats, N, m->in_channels, 1);
        if (!x) { fprintf(stderr, "sp3d_create failed\n"); rc = 7; goto done; }

        /* Run transformer. */
        float *out_feats = NULL;
        if (sam3d_gs_decoder_transformer(m, x, &out_feats, 4) != 0) {
            fprintf(stderr, "transformer forward failed\n"); rc = 8;
            sp3d_free(x); goto done;
        }

        /* Diff 1: transformer output feats. */
        snprintf(pbuf, sizeof(pbuf), "%s/slat_gs_out_feats.npy", refdir);
        float *ref_out = (float *)verify_read_npy(pbuf, &nd, dims, &is_f32);
        if (ref_out && is_f32 && nd == 2 && dims[0] == N && dims[1] == m->out_channels) {
            if (!diff_and_report("transformer out_feats", ref_out, out_feats,
                                 N * m->out_channels, 5e-3f)) rc = 9;
        } else {
            fprintf(stderr, "cannot read %s\n", pbuf);
        }
        free(ref_out);

        /* to_representation. */
        int G = m->num_gaussians;
        float *xyz_c  = (float *)malloc((size_t)N * G * 3 * sizeof(float));
        float *dc_c   = (float *)malloc((size_t)N * G * 3 * sizeof(float));
        float *scl_c  = (float *)malloc((size_t)N * G * 3 * sizeof(float));
        float *rot_c  = (float *)malloc((size_t)N * G * 4 * sizeof(float));
        float *op_c   = (float *)malloc((size_t)N * G * sizeof(float));
        sam3d_gs_decoder_to_representation(m, coords, out_feats, N,
                                            xyz_c, dc_c, scl_c, rot_c, op_c);

        struct { const char *name; const char *label; float *c_buf; int elts; } diffs[] = {
            {"slat_gs_rep_xyz.npy",      "rep_xyz",      xyz_c, N * G * 3},
            {"slat_gs_rep_dc.npy",       "rep_dc",       dc_c,  N * G * 3},
            {"slat_gs_rep_scaling.npy",  "rep_scaling",  scl_c, N * G * 3},
            {"slat_gs_rep_rotation.npy", "rep_rotation", rot_c, N * G * 4},
            {"slat_gs_rep_opacity.npy",  "rep_opacity",  op_c,  N * G},
        };
        for (size_t i = 0; i < sizeof(diffs)/sizeof(diffs[0]); i++) {
            snprintf(pbuf, sizeof(pbuf), "%s/%s", refdir, diffs[i].name);
            float *ref_buf = (float *)verify_read_npy(pbuf, &nd, dims, &is_f32);
            if (!ref_buf || !is_f32) {
                fprintf(stderr, "cannot read %s\n", pbuf);
                continue;
            }
            if (!diff_and_report(diffs[i].label, ref_buf, diffs[i].c_buf,
                                 diffs[i].elts, 5e-3f)) rc = 9;
            free(ref_buf);
        }

        free(xyz_c); free(dc_c); free(scl_c); free(rot_c); free(op_c);
        free(out_feats);
        sp3d_free(x);
        free(coords); free(feats);
    }

done:
    sam3d_gs_decoder_free(m);
    sam3d_destroy(ctx);
    if (rc == 0)
        fprintf(stderr, "[verify_slat_gs] OK (step 8b numerics verify)\n");
    else
        fprintf(stderr, "[verify_slat_gs] FAIL rc=%d\n", rc);
    return rc;
}
