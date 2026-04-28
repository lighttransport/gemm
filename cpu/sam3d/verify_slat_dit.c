/*
 * verify_slat_dit — stage-2 sparse-latent flow DiT verify.
 *
 * Step 7a: loader smoke-test (always runs).
 * Step 7b: with --refdir, also runs one transformer forward against
 *          $REFDIR/slat_dit_{in_coords,in_feats,t,cond,out_feats}.npy.
 *
 * Usage:
 *   verify_slat_dit --ckpt <pipeline.yaml> [--safetensors-dir <dir>]
 *                   [--refdir <dir>] [-t N] [--threshold F] [-v]
 */
#include "safetensors.h"         /* declarations only; impl in sam3d_runner.c */
#include "sam3d_slat_dit.h"      /* impl linked in from sam3d_runner.c */
#include "sparse3d.h"

#include "sam3d_runner.h"
#include "verify_npy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *ckpt = NULL, *refdir = NULL, *sft_dir = NULL;
    int verbose = 0, n_threads = 1;
    float threshold = 5e-2f;  /* bf16 floor */

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt    = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc)       n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v"))                     verbose = 1;
        else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!ckpt) {
        fprintf(stderr, "Usage: %s --ckpt <pipeline.yaml> "
                        "[--safetensors-dir <dir>] [--refdir <dir>] "
                        "[-t N] [--threshold F] [-v]\n", argv[0]);
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
    snprintf(path, sizeof(path), "%s/sam3d_slat_dit.safetensors",
             sam3d_safetensors_dir(ctx));
    fprintf(stderr, "[verify_slat_dit] loading %s\n", path);

    sam3d_slat_dit_model *m = sam3d_slat_dit_load_safetensors(path);
    if (!m) {
        fprintf(stderr, "[verify_slat_dit] loader failed\n");
        sam3d_destroy(ctx);
        return 3;
    }
    fprintf(stderr,
            "[verify_slat_dit] loaded OK: dim=%d heads=%d head_dim=%d "
            "blocks=%d io=%d cond=%d in=%d out=%d\n",
            m->dim, m->n_heads, m->head_dim, m->n_blocks,
            m->n_io_res_blocks, m->cond_channels,
            m->in_channels, m->out_channels);

    /* Sanity: block 0 self-attn qkv shape. */
    if (m->n_blocks > 0) {
        fprintf(stderr,
                "[verify_slat_dit] block[0].sa_qkv_w shape=(%llu,%llu) dtype=%u\n",
                (unsigned long long)m->blocks[0].sa_qkv_w.dims[0],
                (unsigned long long)m->blocks[0].sa_qkv_w.dims[1],
                m->blocks[0].sa_qkv_w.type);
    }

    int rc = 0;
    if (refdir) {
        char p[1536];
        int nd, dims[8], is_f32;

        snprintf(p, sizeof(p), "%s/slat_dit_in_coords.npy", refdir);
        int32_t *coords = (int32_t *)verify_read_npy(p, &nd, dims, &is_f32);
        if (!coords || nd != 2 || dims[1] != 4) {
            fprintf(stderr, "[verify_slat_dit] bad %s\n", p); rc = 4; goto done;
        }
        int N = dims[0];

        snprintf(p, sizeof(p), "%s/slat_dit_in_feats.npy", refdir);
        float *feats = (float *)verify_read_npy(p, &nd, dims, &is_f32);
        if (!feats || !is_f32 || nd != 2 || dims[0] != N || dims[1] != m->in_channels) {
            fprintf(stderr, "[verify_slat_dit] bad %s\n", p); rc = 4; goto done;
        }

        snprintf(p, sizeof(p), "%s/slat_dit_cond.npy", refdir);
        float *cond = (float *)verify_read_npy(p, &nd, dims, &is_f32);
        if (!cond || !is_f32 || nd != 3 || dims[2] != m->cond_channels) {
            fprintf(stderr, "[verify_slat_dit] bad %s\n", p); rc = 4; goto done;
        }
        int n_cond = dims[1];

        snprintf(p, sizeof(p), "%s/slat_dit_t.npy", refdir);
        float *tnp = (float *)verify_read_npy(p, &nd, dims, &is_f32);
        if (!tnp || !is_f32) {
            fprintf(stderr, "[verify_slat_dit] bad %s\n", p); rc = 4; goto done;
        }
        float t_val = tnp[0];
        free(tnp);

        snprintf(p, sizeof(p), "%s/slat_dit_out_feats.npy", refdir);
        float *ref_out = (float *)verify_read_npy(p, &nd, dims, &is_f32);
        if (!ref_out || !is_f32 || nd != 2 || dims[0] != N || dims[1] != m->out_channels) {
            fprintf(stderr, "[verify_slat_dit] bad %s\n", p); rc = 4; goto done;
        }

        sp3d_tensor *x = sp3d_create(coords, feats, N, m->in_channels, 1);
        free(coords); free(feats);
        if (!x) { fprintf(stderr, "sp3d_create failed\n"); rc = 5; goto done; }

        fprintf(stderr,
                "[verify_slat_dit] running forward N=%d n_cond=%d t=%.4f nthr=%d\n",
                N, n_cond, t_val, n_threads);
        if (sam3d_slat_dit_forward(m, &x, t_val, cond, n_cond, n_threads) != 0) {
            fprintf(stderr, "[verify_slat_dit] forward failed\n");
            sp3d_free(x); free(cond); free(ref_out); rc = 7; goto done;
        }

        double mean = 0.0;
        float mx = verify_max_abs(x->feats, ref_out, N * m->out_channels, &mean);
        int ok = (mx <= threshold);
        fprintf(stderr,
                "[verify_slat_dit] out_feats max_abs=%.4g mean_abs=%.4g n=%d  %s\n",
                (double)mx, mean, N * m->out_channels, ok ? "OK" : "FAIL");

        sp3d_free(x); free(cond); free(ref_out);
        if (!ok) rc = 9;
    }

done:
    sam3d_slat_dit_free(m);
    sam3d_destroy(ctx);
    if (rc == 0)
        fprintf(stderr, "[verify_slat_dit] %s\n",
                refdir ? "OK (step 7b numerics verify)"
                       : "OK (step 7a loader smoke-test)");
    else
        fprintf(stderr, "[verify_slat_dit] FAIL rc=%d\n", rc);
    return rc;
}
