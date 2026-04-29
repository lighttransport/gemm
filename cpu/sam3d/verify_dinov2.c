/*
 * verify_dinov2 — diff our DINOv2-L/14+reg encoder against the
 * pytorch reference dump.
 *
 * Usage:
 *   verify_dinov2 --ckpt <pipeline.yaml> --refdir /tmp/sam3d_ref [-v]
 *                 [--safetensors-dir <dir>] [-t N]
 *
 * Expects in $REFDIR:
 *   image_processed.npy      (518, 518, 3)  uint8 or f32 — preprocessed RGB
 *   mask_processed.npy       (518, 518)     uint8 or f32 — binary mask
 *   dinov2_tokens.npy        (2, 1374, 1024) f32 — [image|mask] branches
 *                                                  (or (1374,1024) single-branch)
 *
 * Exit 0 if max_abs < threshold (default 5e-2 — covers long-tailed fp32
 * drift through 24 transformer blocks; mean_abs should stay ~1e-4).
 */

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

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
    float threshold = 5e-2f;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--ckpt")   && i+1 < argc) ckpt    = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc)       n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v"))                     verbose = 1;
        else if (!strcmp(argv[i], "--use-ref-inputs")) {
            /* stage 0 — no upstream to substitute. */
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (!ckpt || !refdir) {
        fprintf(stderr,
                "Usage: %s --ckpt <pipeline.yaml> --refdir <dir> "
                "[--safetensors-dir <dir>] [--threshold F] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }

    /* Load reference inputs & target. Prefer preprocessed dumps; fall
     * back to the raw PIL-dumped names gen_image_ref.py writes when
     * --skip-run was used. */
    char path[1024];
    int nd = 0, dims[8] = {0}, is_f32 = 0;

    snprintf(path, sizeof(path), "%s/image_processed.npy", refdir);
    void *img_raw = verify_read_npy(path, &nd, dims, &is_f32);
    if (!img_raw) {
        snprintf(path, sizeof(path), "%s/input_image.npy", refdir);
        img_raw = verify_read_npy(path, &nd, dims, &is_f32);
    }
    if (!img_raw || nd != 3 || dims[2] != 3) {
        fprintf(stderr, "bad/missing %s (want (H,W,3))\n", path);
        free(img_raw); return 3;
    }
    int ih = dims[0], iw = dims[1];
    uint8_t *img = (uint8_t *)malloc((size_t)ih * iw * 3);
    if (is_f32) {
        float *f = (float *)img_raw;
        for (int i = 0; i < ih * iw * 3; i++) {
            float v = f[i];
            if (v <= 1.0f) v *= 255.0f;
            if (v < 0) v = 0; if (v > 255) v = 255;
            img[i] = (uint8_t)(v + 0.5f);
        }
    } else {
        memcpy(img, img_raw, (size_t)ih * iw * 3);
    }
    free(img_raw);

    snprintf(path, sizeof(path), "%s/mask_processed.npy", refdir);
    void *msk_raw = verify_read_npy(path, &nd, dims, &is_f32);
    if (!msk_raw) {
        snprintf(path, sizeof(path), "%s/input_mask.npy", refdir);
        msk_raw = verify_read_npy(path, &nd, dims, &is_f32);
    }
    uint8_t *msk = NULL; int mh = 0, mw = 0;
    if (msk_raw && nd >= 2) {
        mh = dims[0]; mw = dims[1];
        msk = (uint8_t *)malloc((size_t)mh * mw);
        if (is_f32) {
            float *f = (float *)msk_raw;
            for (int i = 0; i < mh * mw; i++) {
                float v = f[i];
                if (v <= 1.0f) v *= 255.0f;
                if (v < 0) v = 0; if (v > 255) v = 255;
                msk[i] = (uint8_t)(v + 0.5f);
            }
        } else {
            memcpy(msk, msk_raw, (size_t)mh * mw);
        }
    }
    free(msk_raw);

    snprintf(path, sizeof(path), "%s/dinov2_tokens.npy", refdir);
    int ref_dims[8] = {0}, ref_nd = 0, ref_f32 = 0;
    float *ref = (float *)verify_read_npy(path, &ref_nd, ref_dims, &ref_f32);
    if (ref && !ref_f32) {
        fprintf(stderr, "%s is not f32; ignoring\n", path);
        free(ref); ref = NULL;
    }
    int ref_branches = 1, ref_ntok = 0, ref_dim = 0;
    if (ref) {
        if (ref_nd == 3) {
            ref_branches = ref_dims[0]; ref_ntok = ref_dims[1]; ref_dim = ref_dims[2];
        } else if (ref_nd == 2) {
            ref_ntok = ref_dims[0]; ref_dim = ref_dims[1];
        } else {
            fprintf(stderr, "bad ref rank=%d\n", ref_nd);
            free(img); free(msk); free(ref); return 3;
        }
        fprintf(stderr, "[verify_dinov2] ref: branches=%d n_tok=%d dim=%d\n",
                ref_branches, ref_ntok, ref_dim);
    } else {
        fprintf(stderr, "[verify_dinov2] no dinov2_tokens.npy — smoke run only\n");
    }

    /* Build runner. */
    sam3d_config cfg = {
        .pipeline_yaml = ckpt,
        .safetensors_dir = sft_dir,
        .seed = 42,
        .n_threads = n_threads,
        .verbose = verbose,
    };
    sam3d_ctx *ctx = sam3d_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_create failed\n"); return 5; }

    /* Feed preprocessed image as RGBA (alpha=255) — the dinov2 stage
     * will re-normalize. */
    uint8_t *rgba = (uint8_t *)malloc((size_t)ih * iw * 4);
    for (int i = 0; i < ih * iw; i++) {
        rgba[i*4+0] = img[i*3+0];
        rgba[i*4+1] = img[i*3+1];
        rgba[i*4+2] = img[i*3+2];
        rgba[i*4+3] = 255;
    }
    if (sam3d_set_image_rgba(ctx, rgba, iw, ih) != 0 ||
        (msk && sam3d_set_mask(ctx, msk, mw, mh) != 0)) {
        fprintf(stderr, "set inputs failed\n");
        free(rgba); free(img); free(msk); free(ref);
        sam3d_destroy(ctx); return 5;
    }
    free(rgba); free(img); free(msk);

    /* Run. */
    int rc = sam3d_run_dinov2(ctx);
    if (rc != 0) {
        fprintf(stderr, "sam3d_run_dinov2 rc=%d\n", rc);
        free(ref); sam3d_destroy(ctx); return 6;
    }

    /* Read back and diff. */
    int out_n = 0, out_c = 0;
    sam3d_get_dinov2_tokens(ctx, NULL, &out_n, &out_c);
    float *ours = (float *)malloc((size_t)out_n * out_c * sizeof(float));
    sam3d_get_dinov2_tokens(ctx, ours, &out_n, &out_c);
    fprintf(stderr, "[verify_dinov2] ours: n_tok=%d dim=%d\n", out_n, out_c);

    int rc_out = 0;
    if (ref) {
        if (out_c != ref_dim) {
            fprintf(stderr, "dim mismatch: ours=%d ref=%d\n", out_c, ref_dim);
            rc_out = 7;
        } else {
            int n_branches_cmp = (ref_branches < (out_n / ref_ntok))
                                 ? ref_branches : (out_n / ref_ntok);
            if (n_branches_cmp < 1) n_branches_cmp = 1;
            int n = n_branches_cmp * ref_ntok * ref_dim;
            double mean_abs = 0.0;
            float mx = verify_max_abs(ours, ref, n, &mean_abs);
            /* Two-gate tolerance: per-element `threshold` bounds outliers
             * (fp32 drift across 24 blocks is long-tailed), and a tighter
             * mean gate catches systematic regressions a single outlier
             * could otherwise hide. */
            const float mean_gate = threshold * 1e-2f;
            fprintf(stderr, "[verify_dinov2] branches_cmp=%d n_tok=%d dim=%d  "
                            "max_abs=%.6e mean_abs=%.6e (threshold max=%.1e mean=%.1e)\n",
                    n_branches_cmp, ref_ntok, ref_dim, mx, mean_abs,
                    threshold, mean_gate);
            rc_out = (mx < threshold && mean_abs < mean_gate) ? 0 : 1;
        }
    }

    free(ours); free(ref);
    sam3d_destroy(ctx);
    return rc_out;
}
