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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static float *load_or_die_dims(const char *refdir, const char *name,
                               int *out_nd, int out_dims[8])
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    int is_f32;
    float *d = (float *)npy_load(path, out_nd, out_dims, &is_f32);
    if (!d || !is_f32) {
        fprintf(stderr, "[verify_end_to_end] missing/non-f32 %s\n", path);
        free(d);
        return NULL;
    }
    return d;
}

static float *load_or_die(const char *refdir, const char *name)
{
    int nd = 0, dims[8] = {0};
    return load_or_die_dims(refdir, name, &nd, dims);
}

static float *load_ref_f32_prefer(const char *refdir, const char *primary,
                                  const char *fallback, int *out_nd,
                                  int out_dims[8], char *out_name,
                                  size_t out_name_size)
{
    char path[1024];
    int is_f32 = 0;

    snprintf(path, sizeof(path), "%s/%s.npy", refdir, primary);
    float *d = (float *)npy_load(path, out_nd, out_dims, &is_f32);
    if (d && is_f32) {
        snprintf(out_name, out_name_size, "%s", primary);
        return d;
    }
    free(d);

    snprintf(path, sizeof(path), "%s/%s.npy", refdir, fallback);
    d = (float *)npy_load(path, out_nd, out_dims, &is_f32);
    if (d && is_f32) {
        snprintf(out_name, out_name_size, "%s", fallback);
        return d;
    }
    free(d);

    fprintf(stderr, "[verify_end_to_end] missing/non-f32 %s/%s.npy "
                    "(fallback %s.npy)\n",
            refdir, primary, fallback);
    out_name[0] = '\0';
    return NULL;
}

static int diff_pair(const char *label, const float *a, const float *b,
                     size_t n, float thresh)
{
    if (n == 0) {
        fprintf(stderr, "[verify_end_to_end] %-32s empty diff input FAIL\n",
                label);
        return 1;
    }
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
    const char *image_path = NULL;
    float threshold = 5e-3f;
    float threshold_2d = -1.0f;
    float bbox[4] = {0};
    int has_bbox = 0;
    int n_threads = 1, verbose = 0;
    sam3d_body_backbone_t backbone = SAM3D_BODY_BACKBONE_DINOV3;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets")      && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--image")           && i+1 < argc) image_path = argv[++i];
        else if (!strcmp(argv[i], "--bbox") && i+4 < argc) {
            bbox[0] = strtof(argv[++i], NULL);
            bbox[1] = strtof(argv[++i], NULL);
            bbox[2] = strtof(argv[++i], NULL);
            bbox[3] = strtof(argv[++i], NULL);
            has_bbox = 1;
        }
        else if (!strcmp(argv[i], "--backbone") && i+1 < argc) {
            const char *v = argv[++i];
            if      (!strcmp(v, "dinov3")) backbone = SAM3D_BODY_BACKBONE_DINOV3;
            else if (!strcmp(v, "vith"))   backbone = SAM3D_BODY_BACKBONE_VITH;
            else { fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n", v); return 2; }
        }
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--threshold-2d")    && i+1 < argc) threshold_2d = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !mhr_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir <dir> --mhr-assets <dir> "
                "--refdir <dir> [--image IMG --bbox x0 y0 x1 y1] "
                "[--backbone dinov3|vith] [--threshold F] "
                "[--threshold-2d PX] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }
    if (image_path && !has_bbox) {
        fprintf(stderr, "[verify_end_to_end] --image mode requires fixed --bbox x0 y0 x1 y1\n");
        return 2;
    }

    sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .mhr_assets_dir  = mhr_dir,
        .backbone        = backbone,
        .seed            = 42,
        .n_threads       = n_threads,
        .verbose         = verbose,
    };
    sam3d_body_ctx *ctx = sam3d_body_create(&cfg);
    if (!ctx) { fprintf(stderr, "sam3d_body_create failed\n"); return 5; }

    int rc = 0;
    if (image_path) {
        int iw = 0, ih = 0, ich = 0;
        uint8_t *pixels = stbi_load(image_path, &iw, &ih, &ich, 3);
        if (!pixels) {
            fprintf(stderr, "[verify_end_to_end] cannot decode %s\n", image_path);
            sam3d_body_destroy(ctx); return 6;
        }
        fprintf(stderr, "[verify_end_to_end] self-driven image=%s size=%dx%d "
                        "bbox=[%.3f %.3f %.3f %.3f] backbone=%s\n",
                image_path, iw, ih, bbox[0], bbox[1], bbox[2], bbox[3],
                backbone == SAM3D_BODY_BACKBONE_VITH ? "vith" : "dinov3");
        rc = sam3d_body_set_image(ctx, pixels, iw, ih, bbox);
        stbi_image_free(pixels);
        if (rc != 0) {
            fprintf(stderr, "[verify_end_to_end] set_image rc=%d\n", rc);
            sam3d_body_destroy(ctx); return 7;
        }
    } else {
        /* Load decoder inputs (matches verify_decoder_full.c). */
        int img_nd = 0, img_dims[8] = {0};
        int pe_nd = 0, pe_dims[8] = {0};
        int x_nd = 0, x_dims[8] = {0};
        int xpe_nd = 0, xpe_dims[8] = {0};
        float *img_emb = load_or_die_dims(refdir, "image_embeddings_after_ray",
                                          &img_nd, img_dims);
        float *ctx_pe_tok = load_or_die_dims(refdir, "decoder_layer0_in__context_pe",
                                             &pe_nd, pe_dims);
        float *init_x = load_or_die_dims(refdir, "decoder_layer0_in__x",
                                         &x_nd, x_dims);
        float *init_xpe = load_or_die_dims(refdir, "decoder_layer0_in__x_pe",
                                           &xpe_nd, xpe_dims);
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
        if (img_nd != 4 || pe_nd != 3 || x_nd != 3 || xpe_nd != 3 ||
            x_dims[1] != 145 || xpe_dims[1] != 145 ||
            x_dims[2] != xpe_dims[2] ||
            img_dims[1] != pe_dims[2] ||
            pe_dims[1] != img_dims[2] * img_dims[3]) {
            fprintf(stderr, "[verify_end_to_end] bad override ref shapes: "
                    "image_embeddings_after_ray rank=%d [%d,%d,%d,%d], "
                    "context_pe rank=%d [%d,%d,%d], x rank=%d [%d,%d,%d], "
                    "x_pe rank=%d [%d,%d,%d]\n",
                    img_nd, img_dims[0], img_dims[1], img_dims[2], img_dims[3],
                    pe_nd, pe_dims[0], pe_dims[1], pe_dims[2],
                    x_nd, x_dims[0], x_dims[1], x_dims[2],
                    xpe_nd, xpe_dims[0], xpe_dims[1], xpe_dims[2]);
            free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
            free(cam_int); free(bbox_center); free(bbox_scale);
            free(ori_img_size); free(img_size); free(affine_trans);
            sam3d_body_destroy(ctx); return 6;
        }
        const int Dc = img_dims[1];
        const int H = img_dims[2];
        const int W = img_dims[3];

        /* context_pe is dumped as token (1, 1024, 1280); permute to CHW. */
        float *image_pe_chw = (float *)malloc((size_t)Dc * H * W * sizeof(float));
        for (int n = 0; n < H * W; n++)
            for (int c = 0; c < Dc; c++)
                image_pe_chw[(size_t)c * H * W + n] = ctx_pe_tok[(size_t)n * Dc + c];
        free(ctx_pe_tok);

        rc = sam3d_body_debug_override_decoder_inputs(
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
    }

    rc = sam3d_body_run_all(ctx);
    if (rc != 0) {
        fprintf(stderr, "[verify_end_to_end] run_all rc=%d\n", rc);
        sam3d_body_destroy(ctx); return 8;
    }

    int rc_total = 0;

    /* --- vertices --- */
    {
        int nd = 0, dims[8] = {0};
        char ref_name[64];
        float *ref_v = load_ref_f32_prefer(refdir, "body_out_vertices",
                                           "out_vertices", &nd, dims,
                                           ref_name, sizeof(ref_name));
        int v = 0;
        sam3d_body_get_vertices(ctx, NULL, &v);
        if (!ref_v || nd < 2 || dims[nd-1] != 3) {
            fprintf(stderr, "[verify_end_to_end] missing/bad vertices ref\n");
            free(ref_v);
            rc_total |= 1;
        } else {
            int v_ref = (nd == 2) ? dims[0] : dims[nd-2]; /* (V,3) or (1,V,3) */
            float *ours = (float *)malloc((size_t)v * 3 * sizeof(float));
            sam3d_body_get_vertices(ctx, ours, &v);
            if (v != v_ref) {
                fprintf(stderr, "[verify_end_to_end] V mismatch ours=%d ref=%d\n",
                        v, v_ref);
                rc_total |= 1;
            } else {
                char label[96];
                snprintf(label, sizeof(label), "vertices vs %s", ref_name);
                rc_total |= diff_pair(label, ours, ref_v,
                                      (size_t)v * 3, threshold);
            }
            free(ours); free(ref_v);
        }
    }

    /* --- keypoints 3d --- */
    {
        int nd = 0, dims[8] = {0};
        char ref_name[64];
        float *ref_k = load_ref_f32_prefer(refdir, "body_out_keypoints_3d",
                                           "out_keypoints_3d", &nd, dims,
                                           ref_name, sizeof(ref_name));
        int k = 0;
        sam3d_body_get_keypoints_3d(ctx, NULL, &k);
        if (!ref_k || nd < 2 || dims[nd-1] != 3) {
            fprintf(stderr, "[verify_end_to_end] missing/bad kp3d ref\n");
            free(ref_k);
            rc_total |= 1;
        } else {
            int k_ref = (nd == 2) ? dims[0] : dims[nd-2];
            float *ours = (float *)malloc((size_t)k * 3 * sizeof(float));
            sam3d_body_get_keypoints_3d(ctx, ours, &k);
            if (k != k_ref) {
                fprintf(stderr, "[verify_end_to_end] K3D mismatch ours=%d ref=%d\n",
                        k, k_ref);
                rc_total |= 1;
            } else {
                char label[96];
                snprintf(label, sizeof(label), "keypoints_3d vs %s", ref_name);
                rc_total |= diff_pair(label, ours, ref_k,
                                      (size_t)k * 3, threshold);
            }
            free(ours); free(ref_k);
        }
    }

    /* --- keypoints 2d --- */
    {
        int nd = 0, dims[8] = {0};
        char ref_name[64];
        float *ref_k = load_ref_f32_prefer(refdir, "body_out_keypoints_2d",
                                           "out_keypoints_2d", &nd, dims,
                                           ref_name, sizeof(ref_name));
        int k = 0;
        sam3d_body_get_keypoints_2d(ctx, NULL, &k);
        if (!ref_k || nd < 2 || dims[nd-1] != 2) {
            fprintf(stderr, "[verify_end_to_end] missing/bad kp2d ref\n");
            free(ref_k);
            rc_total |= 1;
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
                float t2d = (threshold_2d >= 0.0f)
                    ? threshold_2d
                    : ((threshold < 1e-2f) ? 1e-2f : threshold);
                char label[96];
                snprintf(label, sizeof(label), "keypoints_2d px vs %s", ref_name);
                rc_total |= diff_pair(label, ours, ref_k,
                                      (size_t)k * 2, t2d);
            }
            free(ours); free(ref_k);
        }
    }

    sam3d_body_destroy(ctx);
    return rc_total;
}
