/*
 * verify_end_to_end — Step 8d: drive the public runner via
 * sam3d_body_debug_override_decoder_inputs and diff the resulting
 * vertices + keypoints against the python reference dump.
 *
 * Usage:
 *   verify_end_to_end --safetensors-dir <dir> --mhr-assets <dir> \
 *                     --refdir /tmp/sam3d_body_ref [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR (produced by ref/sam3d-body/gen_image_ref.py):
 *   image_embeddings_after_ray.npy
 *   decoder_layer0_in__{x,x_pe,context_pe}.npy
 *   decoder_batch__{cam_int,bbox_center,bbox_scale,ori_img_size,
 *                   img_size,affine_trans}.npy
 *   out_vertices.npy        (V, 3) — final vertices in camera frame
 *   out_keypoints_3d.npy    (K, 3) — final 3D keypoints
 *   out_keypoints_2d.npy    (K, 2) — final 2D keypoints (image space)
 */

#include "sam3d_body_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "npy_io.h"

static float *load_or_die(const char *refdir, const char *name)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    int nd, dims[8], is_f32;
    float *d = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!d || !is_f32) {
        fprintf(stderr, "[verify_end_to_end] missing/non-f32 %s\n", path);
        free(d);
        return NULL;
    }
    return d;
}

static int diff_pair(const char *label, const float *a, const float *b,
                     size_t n, float thresh)
{
    double sum = 0.0; float mx = 0.0f; size_t mxi = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; mxi = i; }
        sum += d;
    }
    double mean = sum / (double)n;
    int fail = (mx >= thresh);
    fprintf(stderr, "[verify_end_to_end] %-32s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *mhr_dir = NULL, *refdir = NULL;
    float threshold = 5e-3f;
    int n_threads = 1, verbose = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets")      && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !mhr_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir <dir> --mhr-assets <dir> "
                "--refdir <dir> [--threshold F] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }

    sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .mhr_assets_dir  = mhr_dir,
        .backbone        = SAM3D_BODY_BACKBONE_DINOV3,
        .seed            = 42,
        .n_threads       = n_threads,
        .verbose         = verbose,
    };
    sam3d_body_ctx *ctx = sam3d_body_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_body_create failed\n"); return 5; }

    /* Load decoder inputs (matches verify_decoder_full.c). */
    const int H = 32, W = 32, Dc = 1280;
    float *img_emb     = load_or_die(refdir, "image_embeddings_after_ray");
    float *ctx_pe_tok  = load_or_die(refdir, "decoder_layer0_in__context_pe");
    float *init_x      = load_or_die(refdir, "decoder_layer0_in__x");
    float *init_xpe    = load_or_die(refdir, "decoder_layer0_in__x_pe");
    float *cam_int     = load_or_die(refdir, "decoder_batch__cam_int");
    float *bbox_center = load_or_die(refdir, "decoder_batch__bbox_center");
    float *bbox_scale  = load_or_die(refdir, "decoder_batch__bbox_scale");
    float *ori_img_size = load_or_die(refdir, "decoder_batch__ori_img_size");
    float *img_size    = load_or_die(refdir, "decoder_batch__img_size");
    float *affine_trans = load_or_die(refdir, "decoder_batch__affine_trans");
    if (!img_emb || !ctx_pe_tok || !init_x || !init_xpe || !cam_int ||
        !bbox_center || !bbox_scale || !ori_img_size || !img_size ||
        !affine_trans) {
        free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
        free(cam_int); free(bbox_center); free(bbox_scale);
        free(ori_img_size); free(img_size); free(affine_trans);
        sam3d_body_destroy(ctx); return 6;
    }

    /* context_pe is dumped as token (1, 1024, 1280); permute to CHW. */
    float *image_pe_chw = (float *)malloc((size_t)Dc * H * W * sizeof(float));
    for (int n = 0; n < H * W; n++)
        for (int c = 0; c < Dc; c++)
            image_pe_chw[(size_t)c * H * W + n] = ctx_pe_tok[(size_t)n * Dc + c];
    free(ctx_pe_tok);

    int rc = sam3d_body_debug_override_decoder_inputs(
            ctx, img_emb, image_pe_chw, H, W, init_x, init_xpe,
            cam_int, bbox_center, bbox_scale, ori_img_size, img_size,
            affine_trans, /*use_intrin_center=*/0,
            /*default_scale_factor=*/1.0f);
    free(img_emb); free(init_x); free(init_xpe); free(image_pe_chw);
    free(cam_int); free(bbox_center); free(bbox_scale);
    free(ori_img_size); free(img_size); free(affine_trans);
    if (rc != 0) {
        fprintf(stderr, "[verify_end_to_end] override rc=%d\n", rc);
        sam3d_body_destroy(ctx); return 7;
    }

    rc = sam3d_body_run_all(ctx);
    if (rc != 0) {
        fprintf(stderr, "[verify_end_to_end] run_all rc=%d\n", rc);
        sam3d_body_destroy(ctx); return 8;
    }

    int rc_total = 0;

    /* --- vertices --- */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/out_vertices.npy", refdir);
        int nd = 0, dims[8] = {0}, is_f32 = 0;
        float *ref_v = (float *)npy_load(path, &nd, dims, &is_f32);
        int v = 0;
        sam3d_body_get_vertices(ctx, NULL, &v);
        if (!ref_v || nd < 2 || dims[nd-1] != 3) {
            fprintf(stderr, "[verify_end_to_end] missing/bad %s — skipping vertices diff\n", path);
            free(ref_v);
        } else {
            int v_ref = (nd == 2) ? dims[0] : dims[nd-2]; /* (V,3) or (1,V,3) */
            float *ours = (float *)malloc((size_t)v * 3 * sizeof(float));
            sam3d_body_get_vertices(ctx, ours, &v);
            if (v != v_ref) {
                fprintf(stderr, "[verify_end_to_end] V mismatch ours=%d ref=%d\n",
                        v, v_ref);
                rc_total |= 1;
            } else {
                rc_total |= diff_pair("vertices (V,3)", ours, ref_v,
                                      (size_t)v * 3, threshold);
            }
            free(ours); free(ref_v);
        }
    }

    /* --- keypoints 3d --- */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/out_keypoints_3d.npy", refdir);
        int nd = 0, dims[8] = {0}, is_f32 = 0;
        float *ref_k = (float *)npy_load(path, &nd, dims, &is_f32);
        int k = 0;
        sam3d_body_get_keypoints_3d(ctx, NULL, &k);
        if (!ref_k || nd < 2 || dims[nd-1] != 3) {
            fprintf(stderr, "[verify_end_to_end] missing/bad %s — skipping kp3d diff\n", path);
            free(ref_k);
        } else {
            int k_ref = (nd == 2) ? dims[0] : dims[nd-2];
            float *ours = (float *)malloc((size_t)k * 3 * sizeof(float));
            sam3d_body_get_keypoints_3d(ctx, ours, &k);
            if (k != k_ref) {
                fprintf(stderr, "[verify_end_to_end] K3D mismatch ours=%d ref=%d\n",
                        k, k_ref);
                rc_total |= 1;
            } else {
                rc_total |= diff_pair("keypoints_3d (K,3)", ours, ref_k,
                                      (size_t)k * 3, threshold);
            }
            free(ours); free(ref_k);
        }
    }

    /* --- keypoints 2d --- */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/out_keypoints_2d.npy", refdir);
        int nd = 0, dims[8] = {0}, is_f32 = 0;
        float *ref_k = (float *)npy_load(path, &nd, dims, &is_f32);
        int k = 0;
        sam3d_body_get_keypoints_2d(ctx, NULL, &k);
        if (!ref_k || nd < 2 || dims[nd-1] != 2) {
            fprintf(stderr, "[verify_end_to_end] missing/bad %s — skipping kp2d diff\n", path);
            free(ref_k);
        } else {
            int k_ref = (nd == 2) ? dims[0] : dims[nd-2];
            float *ours = (float *)malloc((size_t)k * 2 * sizeof(float));
            sam3d_body_get_keypoints_2d(ctx, ours, &k);
            if (k != k_ref) {
                fprintf(stderr, "[verify_end_to_end] K2D mismatch ours=%d ref=%d\n",
                        k, k_ref);
                rc_total |= 1;
            } else {
                /* 2D keypoints are pixels; pixel-level threshold is looser. */
                float t2d = (threshold < 1e-2f) ? 1e-2f : threshold;
                rc_total |= diff_pair("keypoints_2d (K,2) px", ours, ref_k,
                                      (size_t)k * 2, t2d);
            }
            free(ours); free(ref_k);
        }
    }

    sam3d_body_destroy(ctx);
    return rc_total;
}
