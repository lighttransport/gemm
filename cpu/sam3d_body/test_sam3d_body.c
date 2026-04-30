/*
 * test_sam3d_body — CLI for the SAM 3D Body CPU runner (DINOv3 v1).
 *
 * Two modes are supported in v1:
 *
 *   1. Ref-driven (Step 8c-i):
 *      test_sam3d_body --safetensors-dir <dir> --mhr-assets <dir>
 *                      --refdir /tmp/sam3d_body_ref [-o out.obj]
 *      Feeds pre-computed decoder inputs (image_embeddings_after_ray,
 *      decoder_layer0_in__{x,x_pe,context_pe}, decoder_batch__*) from
 *      the python reference dump into the runner via
 *      sam3d_body_debug_override_decoder_inputs, then exercises the
 *      production sam3d_body_decoder_forward_full path and emits OBJ.
 *
 *   2. Self-driven:
 *      test_sam3d_body --safetensors-dir <dir> --mhr-assets <dir>
 *                      --image <path> [--bbox x0 y0 x1 y1] [--focal F]
 *                      [-o out.obj]
 *      Runs encoder + ray_cond + prompt_encoder + TopdownAffine from raw
 *      RGB end-to-end. If --bbox is omitted the full image is used.
 */

#include "sam3d_body_runner.h"

#define OBJ_WRITER_IMPLEMENTATION
#include "obj_writer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINY_GLTF_IMPLEMENTATION
#include "../../common/tiny_gltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#define RT_DETR_IMPLEMENTATION
#include "../../common/rt_detr.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "npy_io.h"

static double cli_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static int cli_timing_enabled(int verbose)
{
    const char *env = getenv("SAM3D_BODY_TIMING");
    return verbose || (env && env[0] && strcmp(env, "0") != 0);
}

static int has_suffix(const char *s, const char *suffix)
{
    if (!s || !suffix) return 0;
    size_t ns = strlen(s), nx = strlen(suffix);
    return ns >= nx && strcmp(s + ns - nx, suffix) == 0;
}

static int write_body_mesh(const char *path,
                           const float *verts, int nv,
                           const int32_t *faces, int nf)
{
    if (has_suffix(path, ".glb")) {
        int *tri = (int *)malloc((size_t)nf * 3 * sizeof(int));
        if (!tri) return 1;
        for (int i = 0; i < nf * 3; i++) tri[i] = (int)faces[i];
        int rc = tinygltf_write_glb_mesh(path, verts, nv, tri, nf);
        free(tri);
        return rc == 0 ? 0 : 1;
    }
    return obj_write(path, verts, nv, faces, nf);
}

static float *load_ref_f32(const char *refdir, const char *name)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    int nd, dims[8], is_f32;
    float *d = (float *)npy_load(path, &nd, dims, &is_f32);
    if (!d || !is_f32) {
        fprintf(stderr, "[test_sam3d_body] missing/non-f32 %s\n", path);
        free(d);
        return NULL;
    }
    return d;
}

static int run_refdir(const char *sft_dir, const char *mhr_dir,
                      const char *refdir, const char *out_path,
                      sam3d_body_backbone_t backbone,
                      int n_threads, int verbose)
{
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

    /* Decoder spatial dims for DINOv3-H+ are 32x32. */
    const int H = 32, W = 32, Dc = 1280;

    float *img_emb     = load_ref_f32(refdir, "image_embeddings_after_ray");
    float *ctx_pe_tok  = load_ref_f32(refdir, "decoder_layer0_in__context_pe");
    float *init_x      = load_ref_f32(refdir, "decoder_layer0_in__x");
    float *init_xpe    = load_ref_f32(refdir, "decoder_layer0_in__x_pe");
    if (!img_emb || !ctx_pe_tok || !init_x || !init_xpe) {
        free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
        sam3d_body_destroy(ctx); return 6;
    }

    /* context_pe is dumped in token form (1, 1024, 1280) — permute to CHW. */
    float *image_pe_chw = (float *)malloc((size_t)Dc * H * W * sizeof(float));
    for (int n = 0; n < H * W; n++)
        for (int c = 0; c < Dc; c++)
            image_pe_chw[(size_t)c * H * W + n] = ctx_pe_tok[(size_t)n * Dc + c];
    free(ctx_pe_tok);

    float *cam_int       = load_ref_f32(refdir, "decoder_batch__cam_int");
    float *bbox_center   = load_ref_f32(refdir, "decoder_batch__bbox_center");
    float *bbox_scale    = load_ref_f32(refdir, "decoder_batch__bbox_scale");
    float *ori_img_size  = load_ref_f32(refdir, "decoder_batch__ori_img_size");
    float *img_size      = load_ref_f32(refdir, "decoder_batch__img_size");
    float *affine_trans  = load_ref_f32(refdir, "decoder_batch__affine_trans");
    if (!cam_int || !bbox_center || !bbox_scale || !ori_img_size ||
        !img_size || !affine_trans) {
        free(img_emb); free(init_x); free(init_xpe); free(image_pe_chw);
        free(cam_int); free(bbox_center); free(bbox_scale);
        free(ori_img_size); free(img_size); free(affine_trans);
        sam3d_body_destroy(ctx); return 7;
    }

    int rc = sam3d_body_debug_override_decoder_inputs(
            ctx, img_emb, image_pe_chw, H, W, init_x, init_xpe,
            cam_int, bbox_center, bbox_scale, ori_img_size, img_size,
            affine_trans, /*use_intrin_center=*/0,
            /*default_scale_factor=*/1.0f);
    free(img_emb); free(init_x); free(init_xpe); free(image_pe_chw);
    free(cam_int); free(bbox_center); free(bbox_scale);
    free(ori_img_size); free(img_size); free(affine_trans);
    if (rc != 0) {
        fprintf(stderr, "[test_sam3d_body] debug_override rc=%d\n", rc);
        sam3d_body_destroy(ctx); return 8;
    }

    rc = sam3d_body_run_all(ctx);
    if (rc != 0) {
        fprintf(stderr, "[test_sam3d_body] run_all rc=%d\n", rc);
        sam3d_body_destroy(ctx); return 9;
    }

    int nv = 0, nf = 0;
    sam3d_body_get_vertices(ctx, NULL, &nv);
    sam3d_body_get_faces(ctx, NULL, &nf);
    if (nv <= 0 || nf <= 0) {
        fprintf(stderr, "[test_sam3d_body] empty output: V=%d F=%d\n", nv, nf);
        sam3d_body_destroy(ctx); return 10;
    }

    float   *verts = (float *)malloc((size_t)nv * 3 * sizeof(float));
    int32_t *faces = (int32_t *)malloc((size_t)nf * 3 * sizeof(int32_t));
    sam3d_body_get_vertices(ctx, verts, &nv);
    sam3d_body_get_faces(ctx, faces, &nf);

    rc = write_body_mesh(out_path, verts, nv, faces, nf);
    if (rc == 0)
        fprintf(stderr, "[test_sam3d_body] wrote %s (V=%d F=%d)\n",
                out_path, nv, nf);
    else
        fprintf(stderr, "[test_sam3d_body] mesh write %s failed\n", out_path);

    free(verts); free(faces);
    sam3d_body_destroy(ctx);
    return rc;
}

static int run_image(const char *sft_dir, const char *mhr_dir,
                     const char *image_path, float *bbox, int has_bbox,
                     int auto_bbox, const char *rt_detr_model,
                     float auto_score_thresh,
                     float focal_hint, uint64_t seed, const char *out_path,
                     sam3d_body_backbone_t backbone,
                     int n_threads, int verbose)
{
    const int timing = cli_timing_enabled(verbose);
    const double t_total0 = cli_time_ms();
    double t_decode_ms = 0.0, t_rt_load_ms = 0.0, t_rt_detect_ms = 0.0;
    double t_create_ms = 0.0, t_set_image_ms = 0.0, t_run_all_ms = 0.0;
    double t_write_obj_ms = 0.0, t_write_json_ms = 0.0;
    double t0 = cli_time_ms();
    int iw = 0, ih = 0, ichan = 0;
    uint8_t *pixels = stbi_load(image_path, &iw, &ih, &ichan, 3);
    if (!pixels) { fprintf(stderr, "cannot decode %s\n", image_path); return 3; }
    t_decode_ms = cli_time_ms() - t0;

    /* If --auto-bbox is set and the user did not supply --bbox, run
     * RT-DETR-S on the image to detect the largest person. */
    if (auto_bbox && !has_bbox) {
        t0 = cli_time_ms();
        rt_detr_t *det = rt_detr_load(rt_detr_model);
        t_rt_load_ms = cli_time_ms() - t0;
        if (!det) {
            fprintf(stderr, "[test_sam3d_body] failed to load detector at %s\n",
                    rt_detr_model);
            stbi_image_free(pixels); return 4;
        }
        rt_detr_box_t box;
        t0 = cli_time_ms();
        int rc = rt_detr_detect_largest_person(det, pixels, iw, ih,
                                               auto_score_thresh, &box);
        t_rt_detect_ms = cli_time_ms() - t0;
        rt_detr_free(det);
        if (rc != 0) {
            fprintf(stderr, "[test_sam3d_body] no person detected (thresh=%.2f)\n",
                    auto_score_thresh);
            stbi_image_free(pixels); return 4;
        }
        bbox[0] = box.x0; bbox[1] = box.y0;
        bbox[2] = box.x1; bbox[3] = box.y1;
        has_bbox = 1;
        fprintf(stderr,
            "[test_sam3d_body] auto-bbox: score=%.3f bbox=(%.1f,%.1f,%.1f,%.1f)\n",
            box.score, box.x0, box.y0, box.x1, box.y1);
    }

    sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .mhr_assets_dir  = mhr_dir,
        .backbone        = backbone,
        .seed            = seed,
        .n_threads       = n_threads,
        .verbose         = verbose,
    };
    t0 = cli_time_ms();
    sam3d_body_ctx *ctx = sam3d_body_create(&cfg);
    t_create_ms = cli_time_ms() - t0;
    if (!ctx) {
        fprintf(stderr, "sam3d_body_create failed\n");
        stbi_image_free(pixels); return 5;
    }
    t0 = cli_time_ms();
    sam3d_body_set_image(ctx, pixels, iw, ih, has_bbox ? bbox : NULL);
    if (focal_hint > 0) sam3d_body_set_focal(ctx, focal_hint);
    t_set_image_ms = cli_time_ms() - t0;

    t0 = cli_time_ms();
    int rc = sam3d_body_run_all(ctx);
    t_run_all_ms = cli_time_ms() - t0;
    if (rc != 0) {
        fprintf(stderr, "[test_sam3d_body] run_all rc=%d\n", rc);
        sam3d_body_destroy(ctx); stbi_image_free(pixels); return rc;
    }

    int nv = 0, nf = 0;
    sam3d_body_get_vertices(ctx, NULL, &nv);
    sam3d_body_get_faces(ctx, NULL, &nf);
    if (nv > 0 && nf > 0) {
        float   *verts = (float *)malloc((size_t)nv * 3 * sizeof(float));
        int32_t *faces = (int32_t *)malloc((size_t)nf * 3 * sizeof(int32_t));
        sam3d_body_get_vertices(ctx, verts, &nv);
        sam3d_body_get_faces(ctx, faces, &nf);
        t0 = cli_time_ms();
        rc = write_body_mesh(out_path, verts, nv, faces, nf);
        t_write_obj_ms = cli_time_ms() - t0;
        free(verts); free(faces);
        if (rc == 0)
            fprintf(stderr, "[test_sam3d_body] wrote %s (V=%d F=%d)\n",
                    out_path, nv, nf);
        else
            fprintf(stderr, "[test_sam3d_body] mesh write %s failed\n", out_path);
    }

    /* Sidecar JSON: MHR rig params, camera, 3D + 2D keypoints. */
    {
        char json_path[1024];
        snprintf(json_path, sizeof(json_path), "%s.json", out_path);
        t0 = cli_time_ms();
        FILE *jf = fopen(json_path, "w");
        if (jf) {
            int np = 0; sam3d_body_get_mhr_params(ctx, NULL, &np);
            int nk3 = 0, nk2 = 0;
            sam3d_body_get_keypoints_3d(ctx, NULL, &nk3);
            sam3d_body_get_keypoints_2d(ctx, NULL, &nk2);
            float *mp = np  ? (float *)malloc((size_t)np  * sizeof(float)) : NULL;
            float *k3 = nk3 ? (float *)malloc((size_t)nk3 * 3 * sizeof(float)) : NULL;
            float *k2 = nk2 ? (float *)malloc((size_t)nk2 * 2 * sizeof(float)) : NULL;
            float cam_t[3] = {0}, focal_px = 0;
            if (mp) sam3d_body_get_mhr_params(ctx, mp, &np);
            if (k3) sam3d_body_get_keypoints_3d(ctx, k3, &nk3);
            if (k2) sam3d_body_get_keypoints_2d(ctx, k2, &nk2);
            sam3d_body_get_cam(ctx, cam_t, &focal_px);

            fprintf(jf, "{\n");
            if (has_bbox)
                fprintf(jf, "  \"bbox\": [%.3f, %.3f, %.3f, %.3f],\n",
                        bbox[0], bbox[1], bbox[2], bbox[3]);
            fprintf(jf, "  \"image\": {\"width\": %d, \"height\": %d},\n", iw, ih);
            fprintf(jf, "  \"focal_px\": %.6f,\n", focal_px);
            fprintf(jf, "  \"cam_t\": [%.6f, %.6f, %.6f],\n",
                    cam_t[0], cam_t[1], cam_t[2]);
            fprintf(jf, "  \"mhr_params\": [");
            for (int i = 0; i < np; i++)
                fprintf(jf, "%s%.6g", i ? "," : "", mp[i]);
            fprintf(jf, "],\n");
            fprintf(jf, "  \"keypoints_3d\": [");
            for (int i = 0; i < nk3; i++)
                fprintf(jf, "%s[%.6f,%.6f,%.6f]", i ? "," : "",
                        k3[i*3+0], k3[i*3+1], k3[i*3+2]);
            fprintf(jf, "],\n");
            fprintf(jf, "  \"keypoints_2d\": [");
            for (int i = 0; i < nk2; i++)
                fprintf(jf, "%s[%.3f,%.3f]", i ? "," : "",
                        k2[i*2+0], k2[i*2+1]);
            fprintf(jf, "]\n}\n");
            fclose(jf);
            free(mp); free(k3); free(k2);
            fprintf(stderr, "[test_sam3d_body] wrote %s "
                            "(mhr_params=%d kp3d=%d kp2d=%d)\n",
                    json_path, np, nk3, nk2);
        }
        t_write_json_ms = cli_time_ms() - t0;
    }

    if (timing) {
        const double total_ms = cli_time_ms() - t_total0;
        fprintf(stderr,
                "[test_sam3d_body][timing] total %.3f ms decode %.3f "
                "rt_load %.3f rt_detect %.3f create %.3f set_image %.3f "
                "run_all %.3f write_obj %.3f write_json %.3f\n",
                total_ms, t_decode_ms, t_rt_load_ms, t_rt_detect_ms,
                t_create_ms, t_set_image_ms, t_run_all_ms,
                t_write_obj_ms, t_write_json_ms);
    }

    sam3d_body_destroy(ctx);
    stbi_image_free(pixels);
    return rc;
}

int main(int argc, char **argv)
{
    const char *sft_dir     = NULL;
    const char *mhr_assets  = NULL;
    const char *image_path  = NULL;
    const char *refdir      = NULL;
    const char *out_path    = "body.obj";
    const char *rt_detr_model =
        "/mnt/disk01/models/rt_detr_s/model.safetensors";
    float       bbox[4]     = {0};
    int         has_bbox    = 0;
    int         auto_bbox   = 0;
    float       auto_thresh = 0.5f;
    float       focal_hint  = 0;
    uint64_t    seed        = 42;
    int         n_threads   = 0;
    int         verbose     = 0;
    sam3d_body_backbone_t backbone = SAM3D_BODY_BACKBONE_DINOV3;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir    = argv[++i];
        else if (!strcmp(a, "--mhr-assets") && i+1 < argc)      mhr_assets = argv[++i];
        else if (!strcmp(a, "--image") && i+1 < argc)           image_path = argv[++i];
        else if (!strcmp(a, "--refdir") && i+1 < argc)          refdir     = argv[++i];
        else if (!strcmp(a, "--bbox") && i+4 < argc) {
            bbox[0] = strtof(argv[++i], NULL);
            bbox[1] = strtof(argv[++i], NULL);
            bbox[2] = strtof(argv[++i], NULL);
            bbox[3] = strtof(argv[++i], NULL);
            has_bbox = 1;
        }
        else if (!strcmp(a, "--auto-bbox"))                     auto_bbox  = 1;
        else if (!strcmp(a, "--backbone") && i+1 < argc) {
            const char *v = argv[++i];
            if      (!strcmp(v, "dinov3")) backbone = SAM3D_BODY_BACKBONE_DINOV3;
            else if (!strcmp(v, "vith"))   backbone = SAM3D_BODY_BACKBONE_VITH;
            else { fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n", v); return 2; }
        }
        else if (!strcmp(a, "--rt-detr-model") && i+1 < argc)   rt_detr_model = argv[++i];
        else if (!strcmp(a, "--auto-thresh") && i+1 < argc)     auto_thresh = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--focal") && i+1 < argc) focal_hint = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--seed")  && i+1 < argc) seed       = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "-o") && i+1 < argc)      out_path   = argv[++i];
        else if (!strcmp(a, "-t") && i+1 < argc)      n_threads  = atoi(argv[++i]);
        else if (!strcmp(a, "-v"))                    verbose    = 1;
        else if (a[0] != '-') {
            /* Backwards-compat: positional <safetensors-dir> <image>. */
            if      (positional == 0) sft_dir    = a;
            else if (positional == 1) image_path = a;
            positional++;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            return 2;
        }
    }
    if (!sft_dir || (!image_path && !refdir)) {
        fprintf(stderr,
            "Usage:\n"
            "  %s --safetensors-dir DIR --mhr-assets DIR --refdir REFDIR "
            "[-o body.obj] [-t N] [-v]\n"
            "  %s --safetensors-dir DIR --mhr-assets DIR --image IMG.jpg "
            "[--bbox x0 y0 x1 y1 | --auto-bbox [--rt-detr-model PATH] "
            "[--auto-thresh F]] [--focal F] [-o body.obj] [-t N] [-v]\n"
            "  %s SFT_DIR IMG.jpg ...   (legacy positional)\n",
            argv[0], argv[0], argv[0]);
        return 2;
    }
    if (refdir)
        return run_refdir(sft_dir, mhr_assets, refdir, out_path,
                          backbone, n_threads, verbose);
    return run_image(sft_dir, mhr_assets, image_path,
                     bbox, has_bbox, auto_bbox, rt_detr_model, auto_thresh,
                     focal_hint, seed, out_path,
                     backbone, n_threads, verbose);
}
