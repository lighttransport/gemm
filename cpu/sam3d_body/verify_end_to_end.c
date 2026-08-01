/*
 * verify_end_to_end — Step 8d: drive the public runner via
 * sam3d_body_debug_override_decoder_inputs and diff the resulting
 * vertices + keypoints against the python reference dump.
 *
 * Usage:
 *   verify_end_to_end --safetensors-dir <dir> --mhr-assets <dir> \
 *                     --refdir /tmp/sam3d_body_ref [--threshold F] \
 *                     [--threshold-2d PX] [-t N] [-v]
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

static void infer_dinov3_input_shape(const char *refdir, int *out_h, int *out_w)
{
    if (!refdir || !out_h || !out_w || (*out_h > 0 && *out_w > 0)) return;

    char path[1024];
    int nd = 0, dims[8] = {0}, is_f32 = 0;
    snprintf(path, sizeof(path), "%s/dinov3_input.npy", refdir);
    void *d = npy_load(path, &nd, dims, &is_f32);
    free(d);
    if (nd == 4 && dims[0] == 1 && dims[1] == 3 &&
        dims[2] > 0 && dims[3] > 0) {
        if (*out_h <= 0) *out_h = dims[2];
        if (*out_w <= 0) *out_w = dims[3];
    }
}

static int ref_backbone_is_float32(const char *refdir)
{
    if (!refdir) return 0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/backbone_dtype.txt", refdir);
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char buf[128];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return strstr(buf, "float32") != NULL;
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

static void report_pair(const char *label, const float *a, const float *b,
                        size_t n)
{
    if (n == 0) {
        fprintf(stderr, "[verify_end_to_end] %-32s empty diagnostic input\n",
                label);
        return;
    }
    double sum = 0.0; float mx = 0.0f; size_t mxi = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; mxi = i; }
        sum += d;
    }
    fprintf(stderr, "[verify_end_to_end] diag %-27s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e\n",
            label, mx, mxi, sum / (double)n);
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *mhr_dir = NULL, *refdir = NULL;
    const char *image_path = NULL;
    float threshold = -1.0f;
    float threshold_2d = -1.0f;
    float bbox[4] = {0};
    int has_bbox = 0;
    int image_height = 0;
    int image_width = 0;
    int n_threads = 1, verbose = 0;
    int diagnose_self_inputs = 0;
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
        else if (!strcmp(argv[i], "--image-height") && i+1 < argc) image_height = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--image-width")  && i+1 < argc) image_width  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--threshold-2d")    && i+1 < argc) threshold_2d = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--diagnose-self-inputs")) diagnose_self_inputs = 1;
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !mhr_dir || !refdir) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir <dir> --mhr-assets <dir> "
                "--refdir <dir> [--image IMG --bbox x0 y0 x1 y1] "
                "[--backbone dinov3|vith] [--image-height H] [--image-width W] "
                "[--threshold F] "
                "[--threshold-2d PX] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }
    if (image_path && !has_bbox) {
        fprintf(stderr, "[verify_end_to_end] --image mode requires fixed --bbox x0 y0 x1 y1\n");
        return 2;
    }
    if (threshold < 0.0f) {
        int f32_ref = ref_backbone_is_float32(refdir);
        if (image_path && backbone == SAM3D_BODY_BACKBONE_DINOV3)
            threshold = 2e-2f;
        else if (image_path && backbone == SAM3D_BODY_BACKBONE_VITH)
            threshold = f32_ref ? 2e-2f : 8e-2f;
        else
            threshold = 5e-3f;
    }
    if (threshold_2d < 0.0f) {
        int f32_ref = ref_backbone_is_float32(refdir);
        if (image_path && backbone == SAM3D_BODY_BACKBONE_DINOV3)
            threshold_2d = 0.5f;
        else if (image_path && backbone == SAM3D_BODY_BACKBONE_VITH)
            threshold_2d = f32_ref ? 0.5f : 30.0f;
        else
            threshold_2d = (threshold < 1e-2f) ? 1e-2f : threshold;
    }

    if (image_path && backbone == SAM3D_BODY_BACKBONE_DINOV3)
        infer_dinov3_input_shape(refdir, &image_height, &image_width);

    sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .mhr_assets_dir  = mhr_dir,
        .backbone        = backbone,
        .image_height    = image_height,
        .image_width     = image_width,
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
        if (backbone == SAM3D_BODY_BACKBONE_DINOV3)
            fprintf(stderr, "[verify_end_to_end] encoder input=%dx%d\n",
                    cfg.image_height, cfg.image_width);
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

    if (diagnose_self_inputs) {
        const float *dec_img = NULL, *dec_pe = NULL, *dec_x = NULL, *dec_xpe = NULL;
        const float *cam_int = NULL, *bbox_center = NULL, *ori_img_size = NULL;
        const float *img_size = NULL, *affine_trans = NULL;
        float bbox_scale = 0.0f, default_scale_factor = 0.0f;
        int H = 0, W = 0, Dc = 0, Nq = 0, D = 0, use_intrin_center = 0;
        rc = sam3d_body_debug_get_decoder_inputs(
                ctx, &dec_img, &dec_pe, &dec_x, &dec_xpe, &H, &W, &Dc,
                &Nq, &D, &cam_int, &bbox_center, &bbox_scale,
                &ori_img_size, &img_size, &affine_trans,
                &use_intrin_center, &default_scale_factor);
        if (rc != 0) {
            fprintf(stderr, "[verify_end_to_end] diagnose decoder inputs rc=%d\n", rc);
        } else {
            int nd = 0, dims[8] = {0};
            float *ref_img = load_or_die_dims(refdir, "image_embeddings_after_ray",
                                              &nd, dims);
            if (ref_img && nd == 4 && dims[1] == Dc && dims[2] == H && dims[3] == W)
                report_pair("self image_embeddings", dec_img, ref_img,
                            (size_t)Dc * H * W);
            free(ref_img);

            nd = 0; memset(dims, 0, sizeof(dims));
            float *ref_pe_tok = load_or_die_dims(refdir, "decoder_layer0_in__context_pe",
                                                 &nd, dims);
            if (ref_pe_tok && nd == 3 && dims[1] == H * W && dims[2] == Dc) {
                float *ref_pe_chw = (float *)malloc((size_t)Dc * H * W * sizeof(float));
                if (ref_pe_chw) {
                    for (int n = 0; n < H * W; n++)
                        for (int c = 0; c < Dc; c++)
                            ref_pe_chw[(size_t)c * H * W + n] =
                                ref_pe_tok[(size_t)n * Dc + c];
                    report_pair("self context_pe", dec_pe, ref_pe_chw,
                                (size_t)Dc * H * W);
                    free(ref_pe_chw);
                }
            }
            free(ref_pe_tok);

            nd = 0; memset(dims, 0, sizeof(dims));
            float *ref_x = load_or_die_dims(refdir, "decoder_layer0_in__x",
                                            &nd, dims);
            if (ref_x && nd == 3 && dims[1] == Nq && dims[2] == D)
                report_pair("self init_x", dec_x, ref_x, (size_t)Nq * D);
            free(ref_x);

            nd = 0; memset(dims, 0, sizeof(dims));
            float *ref_xpe = load_or_die_dims(refdir, "decoder_layer0_in__x_pe",
                                              &nd, dims);
            if (ref_xpe && nd == 3 && dims[1] == Nq && dims[2] == D)
                report_pair("self init_x_pe", dec_xpe, ref_xpe, (size_t)Nq * D);
            free(ref_xpe);

            float *ref_cam_int = load_or_die(refdir, "decoder_batch__cam_int");
            float *ref_bbox_center = load_or_die(refdir, "decoder_batch__bbox_center");
            float *ref_bbox_scale = load_or_die(refdir, "decoder_batch__bbox_scale");
            float *ref_ori_img_size = load_or_die(refdir, "decoder_batch__ori_img_size");
            float *ref_img_size = load_or_die(refdir, "decoder_batch__img_size");
            float *ref_affine_trans = load_or_die(refdir, "decoder_batch__affine_trans");
            if (ref_cam_int)       report_pair("self cam_int", cam_int, ref_cam_int, 9);
            if (ref_bbox_center)   report_pair("self bbox_center", bbox_center, ref_bbox_center, 2);
            if (ref_bbox_scale)    report_pair("self bbox_scale", &bbox_scale, ref_bbox_scale, 1);
            if (ref_ori_img_size)  report_pair("self ori_img_size", ori_img_size, ref_ori_img_size, 2);
            if (ref_img_size)      report_pair("self img_size", img_size, ref_img_size, 2);
            if (ref_affine_trans)  report_pair("self affine_trans", affine_trans, ref_affine_trans, 6);
            fprintf(stderr, "[verify_end_to_end] diag self flags use_intrin_center=%d "
                            "default_scale_factor=%.6g\n",
                    use_intrin_center, default_scale_factor);
            free(ref_cam_int); free(ref_bbox_center); free(ref_bbox_scale);
            free(ref_ori_img_size); free(ref_img_size); free(ref_affine_trans);

            int enc_n = 0, enc_dim = 0;
            sam3d_body_get_encoder_tokens(ctx, NULL, &enc_n, &enc_dim);
            if (enc_n > 0 && enc_dim == Dc) {
                float *enc = (float *)malloc((size_t)enc_n * enc_dim * sizeof(float));
                if (enc) {
                    sam3d_body_get_encoder_tokens(ctx, enc, &enc_n, &enc_dim);
                    const char *tok_name =
                        (backbone == SAM3D_BODY_BACKBONE_VITH)
                            ? "vith_tokens" : "dinov3_tokens";
                    nd = 0; memset(dims, 0, sizeof(dims));
                    float *ref_tok = load_or_die_dims(refdir, tok_name,
                                                      &nd, dims);
                    size_t ref_tok_n = 0;
                    if (ref_tok && nd == 3 && dims[0] == 1 &&
                        dims[1] == enc_n && dims[2] == enc_dim)
                        ref_tok_n = (size_t)enc_n * enc_dim;
                    else if (ref_tok && nd == 2 &&
                             dims[0] == enc_n && dims[1] == enc_dim)
                        ref_tok_n = (size_t)enc_n * enc_dim;
                    if (ref_tok_n) {
                        report_pair(tok_name, enc, ref_tok, ref_tok_n);
                    } else if (ref_tok && nd == 4 && dims[1] == Dc &&
                               dims[2] == H && dims[3] == W &&
                               enc_n >= H * W) {
                        int n_prefix = enc_n - H * W;
                        float *patch_chw = (float *)malloc((size_t)Dc * H * W * sizeof(float));
                        if (patch_chw) {
                            const float *patch = enc + (size_t)n_prefix * Dc;
                            for (int n = 0; n < H * W; n++)
                                for (int c = 0; c < Dc; c++)
                                    patch_chw[(size_t)c * H * W + n] =
                                        patch[(size_t)n * Dc + c];
                            report_pair(tok_name, patch_chw, ref_tok,
                                        (size_t)Dc * H * W);
                            free(patch_chw);
                        }
                    }
                    free(ref_tok);
                    free(enc);
                }
            }
        }
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
                char label[96];
                snprintf(label, sizeof(label), "keypoints_2d px vs %s", ref_name);
                rc_total |= diff_pair(label, ours, ref_k,
                                      (size_t)k * 2, threshold_2d);
            }
            free(ours); free(ref_k);
        }
    }

    sam3d_body_destroy(ctx);
    return rc_total;
}
