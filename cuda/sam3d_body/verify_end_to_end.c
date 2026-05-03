/*
 * verify_end_to_end (CUDA) — self-driven raw-image pipeline diff.
 *
 * This verifies the production path:
 *   image + fixed bbox -> CUDA encoder -> CUDA/CPU decoder+MHR loop
 *                       -> vertices + 3D/2D keypoints
 *
 * Reference dumps should be generated with the same fixed bbox, e.g.:
 *   ref/sam3d-body/.venv/bin/python ref/sam3d-body/gen_image_ref.py \
 *       --image IMG --bbox x0 y0 x1 y1 --local-ckpt-dir CKPT \
 *       --outdir /tmp/sam3d_body_ref_fixed
 */

#include "cuda_sam3d_body_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"
#include "../../common/npy_io.h"

static int diff_pair(const char *label, const float *a, const float *b,
                     size_t n, float thresh)
{
    if (n == 0) {
        fprintf(stderr, "[cuda verify_end_to_end] %-32s empty diff input FAIL\n",
                label);
        return 1;
    }
    double sum = 0.0;
    float mx = 0.0f;
    size_t mxi = 0;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) { mx = d; mxi = i; }
        sum += d;
    }
    double mean = sum / (double)n;
    int fail = (mx >= thresh);
    fprintf(stderr, "[cuda verify_end_to_end] %-32s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

static float *load_ref_f32_prefer(const char *refdir, const char *primary,
                                  const char *fallback, int *out_nd,
                                  int out_dims[8], char *out_name,
                                  size_t out_name_size)
{
    int is_f32 = 0;
    char path[1024];
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

    fprintf(stderr, "[cuda verify_end_to_end] missing/non-f32 %s/%s.npy "
                    "(fallback %s.npy)\n",
            refdir, primary, fallback);
    out_name[0] = '\0';
    return NULL;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL;
    const char *mhr_dir = NULL;
    const char *refdir = NULL;
    const char *image_path = NULL;
    const char *precision = "bf16";
    float threshold = 5e-3f;
    float threshold_2d = -1.0f;
    float bbox[4] = {0};
    int has_bbox = 0;
    int device = 0;
    int verbose = 0;
    cuda_sam3d_body_backbone_t backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets")      && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir = argv[++i];
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
            if      (!strcmp(v, "dinov3")) backbone = CUDA_SAM3D_BODY_BACKBONE_DINOV3;
            else if (!strcmp(v, "vith"))   backbone = CUDA_SAM3D_BODY_BACKBONE_VITH;
            else { fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n", v); return 2; }
        }
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--threshold-2d") && i+1 < argc) threshold_2d = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc) precision = argv[++i];
        else if (!strcmp(argv[i], "--device")    && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    if (!sft_dir || !mhr_dir || !refdir || !image_path || !has_bbox) {
        fprintf(stderr,
                "Usage: %s --safetensors-dir DIR --mhr-assets DIR "
                "--refdir DIR --image IMG --bbox x0 y0 x1 y1 "
                "[--backbone dinov3|vith] [--threshold F] "
                "[--threshold-2d PX] [--device N] [--precision bf16|fp16] [-v]\n",
                argv[0]);
        return 2;
    }

    int iw = 0, ih = 0, ich = 0;
    uint8_t *pixels = stbi_load(image_path, &iw, &ih, &ich, 3);
    if (!pixels) {
        fprintf(stderr, "[cuda verify_end_to_end] cannot decode %s\n", image_path);
        return 3;
    }

    fprintf(stderr, "[cuda verify_end_to_end] image=%s size=%dx%d "
                    "bbox=[%.3f %.3f %.3f %.3f] backbone=%s\n",
            image_path, iw, ih, bbox[0], bbox[1], bbox[2], bbox[3],
            backbone == CUDA_SAM3D_BODY_BACKBONE_VITH ? "vith" : "dinov3");

    cuda_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .mhr_assets_dir  = mhr_dir,
        .image_size      = 512,
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = precision,
        .backbone        = backbone,
    };
    cuda_sam3d_body_ctx *ctx = cuda_sam3d_body_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "[cuda verify_end_to_end] create failed\n");
        stbi_image_free(pixels);
        return 4;
    }

    int rc = cuda_sam3d_body_set_image(ctx, pixels, iw, ih, bbox);
    stbi_image_free(pixels);
    if (rc != 0) {
        fprintf(stderr, "[cuda verify_end_to_end] set_image rc=%d\n", rc);
        cuda_sam3d_body_destroy(ctx);
        return 5;
    }
    rc = cuda_sam3d_body_run_all(ctx);
    if (rc != 0) {
        fprintf(stderr, "[cuda verify_end_to_end] run_all rc=%d\n", rc);
        cuda_sam3d_body_destroy(ctx);
        return 6;
    }

    int rc_total = 0;
    int nd = 0, dims[8] = {0};

    char ref_name[64];
    float *ref_v = load_ref_f32_prefer(refdir, "body_out_vertices",
                                       "out_vertices", &nd, dims,
                                       ref_name, sizeof(ref_name));
    int nv = 0;
    cuda_sam3d_body_get_vertices(ctx, NULL, &nv);
    if (ref_v && nd >= 2 && dims[nd-1] == 3) {
        int ref_nv = (nd == 2) ? dims[0] : dims[nd-2];
        float *ours = (float *)malloc((size_t)nv * 3 * sizeof(float));
        cuda_sam3d_body_get_vertices(ctx, ours, &nv);
        if (nv != ref_nv) {
            fprintf(stderr, "[cuda verify_end_to_end] V mismatch ours=%d ref=%d\n",
                    nv, ref_nv);
            rc_total |= 1;
        } else {
            char label[96];
            snprintf(label, sizeof(label), "vertices vs %s", ref_name);
            rc_total |= diff_pair(label, ours, ref_v,
                                  (size_t)nv * 3, threshold);
        }
        free(ours);
    } else {
        rc_total |= 1;
    }
    free(ref_v);

    float *ref_k3 = load_ref_f32_prefer(refdir, "body_out_keypoints_3d",
                                        "out_keypoints_3d", &nd, dims,
                                        ref_name, sizeof(ref_name));
    int nk3 = 0;
    cuda_sam3d_body_get_keypoints_3d(ctx, NULL, &nk3);
    if (ref_k3 && nd >= 2 && dims[nd-1] == 3) {
        int ref_nk = (nd == 2) ? dims[0] : dims[nd-2];
        float *ours = (float *)malloc((size_t)nk3 * 3 * sizeof(float));
        cuda_sam3d_body_get_keypoints_3d(ctx, ours, &nk3);
        if (nk3 != ref_nk) {
            fprintf(stderr, "[cuda verify_end_to_end] K3D mismatch ours=%d ref=%d\n",
                    nk3, ref_nk);
            rc_total |= 1;
        } else {
            char label[96];
            snprintf(label, sizeof(label), "keypoints_3d vs %s", ref_name);
            rc_total |= diff_pair(label, ours, ref_k3,
                                  (size_t)nk3 * 3, threshold);
        }
        free(ours);
    } else {
        rc_total |= 1;
    }
    free(ref_k3);

    float *ref_k2 = load_ref_f32_prefer(refdir, "body_out_keypoints_2d",
                                        "out_keypoints_2d", &nd, dims,
                                        ref_name, sizeof(ref_name));
    int nk2 = 0;
    cuda_sam3d_body_get_keypoints_2d(ctx, NULL, &nk2);
    if (ref_k2 && nd >= 2 && dims[nd-1] == 2) {
        int ref_nk = (nd == 2) ? dims[0] : dims[nd-2];
        float *ours = (float *)malloc((size_t)nk2 * 2 * sizeof(float));
        cuda_sam3d_body_get_keypoints_2d(ctx, ours, &nk2);
        if (nk2 != ref_nk) {
            fprintf(stderr, "[cuda verify_end_to_end] K2D mismatch ours=%d ref=%d\n",
                    nk2, ref_nk);
            rc_total |= 1;
        } else {
            float t2d = (threshold_2d >= 0.0f)
                ? threshold_2d
                : ((threshold < 1e-2f) ? 1e-2f : threshold);
            char label[96];
            snprintf(label, sizeof(label), "keypoints_2d px vs %s", ref_name);
            rc_total |= diff_pair(label, ours, ref_k2,
                                  (size_t)nk2 * 2, t2d);
        }
        free(ours);
    } else {
        rc_total |= 1;
    }
    free(ref_k2);

    cuda_sam3d_body_destroy(ctx);
    return rc_total;
}
