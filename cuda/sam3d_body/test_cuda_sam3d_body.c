/*
 * test_cuda_sam3d_body — CLI for the CUDA SAM 3D Body runner.
 *
 * Usage:
 *   test_cuda_sam3d_body --safetensors-dir <DIR> --image <image.jpg>
 *       --mhr-assets <DIR>
 *       [--bbox x0 y0 x1 y1] [--focal F]
 *       [-o body.obj] [--precision bf16|fp16] [--device N] [-v]
 *
 *   test_cuda_sam3d_body <safetensors-dir> <image.jpg> ...
 */

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#define OBJ_WRITER_IMPLEMENTATION
#include "../../common/obj_writer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINY_GLTF_IMPLEMENTATION
#include "../../common/tiny_gltf.h"

/* SAFETENSORS_IMPLEMENTATION is supplied by cuda_sam3d_body_runner.c. */
#include "../../common/safetensors.h"

#define RT_DETR_IMPLEMENTATION
#include "../../common/rt_detr.h"

#include "cuda_sam3d_body_runner.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double cli_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s --safetensors-dir DIR --image IMG.jpg --mhr-assets DIR "
            "[--bbox x0 y0 x1 y1 | --auto-bbox [--rt-detr-model PATH] [--auto-thresh F]] "
            "[--focal F] [-o body.obj] "
            "[--backbone dinov3|vith] "
            "[--precision bf16|fp16] [--device N] [-v]\n"
            "  %s SFT_DIR IMG.jpg ...   (legacy positional)\n",
            prog, prog);
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

static void write_sidecar_json(cuda_sam3d_body_ctx *ctx,
                               const char *out_path,
                               const float bbox[4], int has_bbox,
                               int iw, int ih)
{
    char json_path[1024];
    snprintf(json_path, sizeof(json_path), "%s.json", out_path);
    FILE *jf = fopen(json_path, "w");
    if (!jf) return;

    int np = 0; cuda_sam3d_body_get_mhr_params(ctx, NULL, &np);
    int nk3 = 0, nk2 = 0;
    cuda_sam3d_body_get_keypoints_3d(ctx, NULL, &nk3);
    cuda_sam3d_body_get_keypoints_2d(ctx, NULL, &nk2);
    float *mp = np  ? (float *)malloc((size_t)np  * sizeof(float)) : NULL;
    float *k3 = nk3 ? (float *)malloc((size_t)nk3 * 3 * sizeof(float)) : NULL;
    float *k2 = nk2 ? (float *)malloc((size_t)nk2 * 2 * sizeof(float)) : NULL;
    float cam_t[3] = {0}, focal_px = 0;
    if (mp) cuda_sam3d_body_get_mhr_params(ctx, mp, &np);
    if (k3) cuda_sam3d_body_get_keypoints_3d(ctx, k3, &nk3);
    if (k2) cuda_sam3d_body_get_keypoints_2d(ctx, k2, &nk2);
    cuda_sam3d_body_get_cam(ctx, cam_t, &focal_px);

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
    fprintf(stderr, "[test_cuda_sam3d_body] wrote %s "
                    "(mhr_params=%d kp3d=%d kp2d=%d)\n",
            json_path, np, nk3, nk2);
}

int main(int argc, char **argv)
{
    const double t_main0 = cli_time_ms();
    double t_decode_image_ms = 0.0, t_auto_bbox_ms = 0.0;
    double t_rt_detr_load_ms = 0.0, t_rt_detr_detect_ms = 0.0;
    double t_create_ms = 0.0, t_set_image_ms = 0.0;
    double t_run_all_ms = 0.0, t_write_obj_ms = 0.0;
    const char *sft_dir    = NULL;
    const char *image_path = NULL;
    const char *mhr_assets = NULL;
    const char *out_path   = "body.obj";
    const char *precision  = "bf16";
    float bbox[4] = {0}; int has_bbox = 0;
    float focal_hint = 0;
    int device = 0, verbose = 0;
    int auto_bbox = 0;
    const char *rt_detr_model = "/mnt/disk01/models/rt_detr_s/model.safetensors";
    float auto_thresh = 0.5f;
    cuda_sam3d_body_backbone_t backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--help") || !strcmp(a, "-h")) {
            print_usage(argv[0]);
            return 0;
        }
        else if (!strcmp(a, "--safetensors-dir") && i+1 < argc) sft_dir    = argv[++i];
        else if (!strcmp(a, "--image")           && i+1 < argc) image_path = argv[++i];
        else if (!strcmp(a, "--bbox") && i+4 < argc) {
            bbox[0] = strtof(argv[++i], NULL); bbox[1] = strtof(argv[++i], NULL);
            bbox[2] = strtof(argv[++i], NULL); bbox[3] = strtof(argv[++i], NULL);
            has_bbox = 1;
        }
        else if (!strcmp(a, "--mhr-assets") && i+1 < argc) mhr_assets = argv[++i];
        else if (!strcmp(a, "--precision")  && i+1 < argc) precision  = argv[++i];
        else if (!strcmp(a, "--device")     && i+1 < argc) device     = atoi(argv[++i]);
        else if (!strcmp(a, "--focal")      && i+1 < argc) focal_hint = strtof(argv[++i], NULL);
        else if (!strcmp(a, "-o")           && i+1 < argc) out_path   = argv[++i];
        else if (!strcmp(a, "-v"))                         verbose    = 1;
        else if (!strcmp(a, "--auto-bbox"))                auto_bbox  = 1;
        else if (!strcmp(a, "--rt-detr-model") && i+1 < argc) rt_detr_model = argv[++i];
        else if (!strcmp(a, "--auto-thresh")   && i+1 < argc) auto_thresh   = strtof(argv[++i], NULL);
        else if (!strcmp(a, "--backbone") && i+1 < argc) {
            const char *v = argv[++i];
            if      (!strcmp(v, "dinov3")) backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;
            else if (!strcmp(v, "vith"))   backbone = CUDA_SAM3D_BODY_BACKBONE_VITH;
            else { fprintf(stderr, "unknown backbone: %s (expected dinov3|vith)\n", v); return 2; }
        }
        else if (a[0] != '-') {
            if      (positional == 0) sft_dir    = a;
            else if (positional == 1) image_path = a;
            positional++;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            return 2;
        }
    }
    if (!sft_dir || !image_path || !mhr_assets) {
        print_usage(argv[0]);
        return 2;
    }

    double t0 = cli_time_ms();
    int iw = 0, ih = 0, ichan = 0;
    uint8_t *pixels = stbi_load(image_path, &iw, &ih, &ichan, 3);
    if (!pixels) { fprintf(stderr, "cannot decode %s\n", image_path); return 3; }
    t_decode_image_ms = cli_time_ms() - t0;

    if (auto_bbox && !has_bbox) {
        const double t_auto0 = cli_time_ms();
        t0 = cli_time_ms();
        rt_detr_t *det = rt_detr_load(rt_detr_model);
        t_rt_detr_load_ms = cli_time_ms() - t0;
        if (!det) {
            fprintf(stderr, "[test_cuda_sam3d_body] rt_detr_load failed: %s\n",
                    rt_detr_model);
            stbi_image_free(pixels);
            return 4;
        }
        rt_detr_box_t box;
        t0 = cli_time_ms();
        int rc = rt_detr_detect_largest_person(det, pixels, iw, ih,
                                               auto_thresh, &box);
        t_rt_detr_detect_ms = cli_time_ms() - t0;
        rt_detr_free(det);
        if (rc != 0) {
            fprintf(stderr, "[test_cuda_sam3d_body] no person detected "
                    "(thresh=%.2f)\n", auto_thresh);
            stbi_image_free(pixels);
            return 4;
        }
        bbox[0] = box.x0; bbox[1] = box.y0;
        bbox[2] = box.x1; bbox[3] = box.y1;
        has_bbox = 1;
        if (verbose)
            fprintf(stderr, "[test_cuda_sam3d_body] auto-bbox score=%.4f "
                    "x0=%.1f y0=%.1f x1=%.1f y1=%.1f\n",
                    box.score, box.x0, box.y0, box.x1, box.y1);
        t_auto_bbox_ms = cli_time_ms() - t_auto0;
    }

    cuda_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .mhr_assets_dir  = mhr_assets,
        .image_size      = 512,
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = precision,
        .backbone        = backbone,
    };
    t0 = cli_time_ms();
    cuda_sam3d_body_ctx *ctx = cuda_sam3d_body_create(&cfg);
    if (!ctx) { stbi_image_free(pixels); fprintf(stderr, "create failed\n"); return 5; }
    t_create_ms = cli_time_ms() - t0;

    t0 = cli_time_ms();
    cuda_sam3d_body_set_image(ctx, pixels, iw, ih, has_bbox ? bbox : NULL);
    if (focal_hint > 0) cuda_sam3d_body_set_focal(ctx, focal_hint);
    t_set_image_ms = cli_time_ms() - t0;

    t0 = cli_time_ms();
    int rc = cuda_sam3d_body_run_all(ctx);
    t_run_all_ms = cli_time_ms() - t0;
    if (rc != 0) {
        fprintf(stderr, "[test_cuda_sam3d_body] run_all rc=%d\n", rc);
        cuda_sam3d_body_destroy(ctx);
        stbi_image_free(pixels);
        return rc;
    }

    int nv = 0, nf = 0;
    cuda_sam3d_body_get_vertices(ctx, NULL, &nv);
    cuda_sam3d_body_get_faces(ctx, NULL, &nf);
    if (nv > 0 && nf > 0) {
        t0 = cli_time_ms();
        float   *verts = (float *)malloc((size_t)nv * 3 * sizeof(float));
        int32_t *faces = (int32_t *)malloc((size_t)nf * 3 * sizeof(int32_t));
        cuda_sam3d_body_get_vertices(ctx, verts, &nv);
        cuda_sam3d_body_get_faces(ctx, faces, &nf);
        rc = write_body_mesh(out_path, verts, nv, faces, nf);
        free(verts); free(faces);
        if (rc == 0)
            fprintf(stderr, "[test_cuda_sam3d_body] wrote %s (V=%d F=%d)\n",
                    out_path, nv, nf);
        else
            fprintf(stderr, "[test_cuda_sam3d_body] mesh write %s failed\n",
                    out_path);
        t_write_obj_ms = cli_time_ms() - t0;
    } else {
        fprintf(stderr, "[test_cuda_sam3d_body] empty output: V=%d F=%d\n", nv, nf);
        rc = 1;
    }

    if (rc == 0)
        write_sidecar_json(ctx, out_path, bbox, has_bbox, iw, ih);

    if (verbose) {
        fprintf(stderr,
                "[test_cuda_sam3d_body][timing] total %.3f ms "
                "(image_decode %.3f, auto_bbox %.3f, create/load %.3f, "
                "set_image %.3f, run_all %.3f, write_obj %.3f)\n",
                cli_time_ms() - t_main0, t_decode_image_ms, t_auto_bbox_ms,
                t_create_ms, t_set_image_ms, t_run_all_ms, t_write_obj_ms);
        if (auto_bbox) {
            fprintf(stderr,
                    "[test_cuda_sam3d_body][timing] auto_bbox split: "
                    "rt_detr_load %.3f ms, rt_detr_detect %.3f ms\n",
                    t_rt_detr_load_ms, t_rt_detr_detect_ms);
        }
    }

    cuda_sam3d_body_destroy(ctx);
    stbi_image_free(pixels);
    return rc;
}
