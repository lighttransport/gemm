/*
 * verify_ray_cond — diff sam3d_body ray_cond_emb sub-module output
 * (Fourier positional encoding + 1×1 conv + LN2d) against the pytorch
 * reference dump from ref/sam3d-body/gen_image_ref.py.
 *
 * Bypasses the antialiased 512→32 ray downsample by feeding the
 * cached downsampled+z-appended rays directly (ray_cond_ds_xyz.npy).
 * This isolates the Fourier+conv+LN2d pipeline from torch-specific
 * resampling, which can be revisited once the rest of the decoder
 * is green.
 *
 * Usage:
 *   verify_ray_cond --safetensors-dir <dir> --refdir <dir> \
 *                   [--threshold F] [-t N] [-v]
 *
 * Expects in $REFDIR:
 *   ray_cond__image_embeddings_pre_ray.npy  (1,1280,H,W) f32
 *   ray_cond_ds_xyz.npy                     (1,H,W,3)    f32
 *   image_embeddings_after_ray.npy          (1,1280,H,W) f32
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "npy_io.h"

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void resolve_variant_path(const char *dir, const char *bucket,
                                 const char *tag, char *out, size_t out_sz)
{
    snprintf(out, out_sz, "%s/sam3d_body_%s_%s.safetensors",
             dir, tag, bucket);
    if (file_exists(out)) return;
    snprintf(out, out_sz, "%s/sam3d_body_%s.safetensors", dir, bucket);
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    const char *backbone = "dinov3";
    /* 1×1 conv over 1379 channels + LN2d accumulates ~1e-3 max_abs in
     * f32 vs the fp32 reference. mean_abs is ~1.5e-4 (~0.05% of typical
     * feature magnitude). Gates set at the observed f32 floor. */
    float threshold = 2e-3f;
    int n_threads = 1, verbose = 0;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--backbone") && i+1 < argc) {
            backbone = argv[++i];
            if (strcmp(backbone, "dinov3") && strcmp(backbone, "vith")) {
                fprintf(stderr, "unknown --backbone %s (use dinov3|vith)\n",
                        backbone);
                return 2;
            }
        }
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F] [--backbone dinov3|vith] [-t N] [-v]\n",
                argv[0]);
        return 2;
    }
    (void)verbose;

    char path[1024];
    int nd = 0, dims[8] = {0};

    snprintf(path, sizeof(path), "%s/ray_cond__image_embeddings_pre_ray.npy", refdir);
    float *img_emb = (float *)npy_load(path, &nd, dims, NULL);
    if (!img_emb || nd != 4 || dims[0] != 1 || dims[1] != 1280 ||
        dims[2] <= 0 || dims[3] <= 0) {
        fprintf(stderr, "[verify_ray_cond] missing/invalid %s\n", path);
        free(img_emb); return 3;
    }
    int C = dims[1], H = dims[2], W = dims[3];

    /* ray_cond_ds_xyz: (1, 32, 32, 3) — HWC layout */
    snprintf(path, sizeof(path), "%s/ray_cond_ds_xyz.npy", refdir);
    float *rays = (float *)npy_load(path, &nd, dims, NULL);
    if (!rays || nd != 4 || dims[0] != 1 || dims[1] != H ||
        dims[2] != W || dims[3] != 3) {
        fprintf(stderr, "[verify_ray_cond] missing/invalid %s\n", path);
        free(img_emb); free(rays); return 3;
    }

    /* Reference output: (1, 1280, 32, 32) */
    snprintf(path, sizeof(path), "%s/image_embeddings_after_ray.npy", refdir);
    float *ref = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref || nd != 4 || dims[1] != C || dims[2] != H || dims[3] != W) {
        fprintf(stderr, "[verify_ray_cond] missing/invalid %s\n", path);
        free(img_emb); free(rays); free(ref); return 3;
    }

    char mhr_path[1024];
    resolve_variant_path(sft_dir, "decoder", backbone, path, sizeof(path));
    resolve_variant_path(sft_dir, "mhr_head", backbone,
                         mhr_path, sizeof(mhr_path));
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(path, mhr_path);
    if (!m) { free(img_emb); free(rays); free(ref); return 5; }

    /* Run forward. Output buffer: (C, H*W). */
    float *out = (float *)malloc((size_t)C * (size_t)H * (size_t)W * sizeof(float));
    if (!out) { sam3d_body_decoder_free(m); free(img_emb); free(rays); free(ref); return 6; }
    int rc = sam3d_body_ray_cond_emb_forward(m, img_emb, rays, H, W, n_threads, out);
    if (rc != SAM3D_BODY_DECODER_E_OK) {
        fprintf(stderr, "[verify_ray_cond] ray_cond_emb_forward rc=%d\n", rc);
        free(out); sam3d_body_decoder_free(m);
        free(img_emb); free(rays); free(ref); return 7;
    }

    /* Diff: ours[c, h, w] vs ref[0, c, h, w]. Both already (C, H*W) = (C, H, W). */
    double sum = 0.0; float mx = 0.0f;
    int mx_c = 0, mx_h = 0, mx_w = 0;
    size_t n_el = (size_t)C * H * W;
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                float ov = out[(size_t)c * H * W + h * W + w];
                float rv = ref[((0 * C + c) * H + h) * W + w];
                float d = fabsf(ov - rv);
                if (d > mx) { mx = d; mx_c = c; mx_h = h; mx_w = w; }
                sum += d;
            }
    double mean_abs = sum / (double)n_el;
    float mean_gate = threshold * 0.15f;
    fprintf(stderr, "[verify_ray_cond] C=%d H=%d W=%d  "
                    "max_abs=%.6e (c=%d h=%d w=%d)  mean_abs=%.6e  "
                    "(max_gate=%.1e mean_gate=%.1e)\n",
            C, H, W, mx, mx_c, mx_h, mx_w, mean_abs, threshold, mean_gate);
    int rc_out = (mx < threshold && mean_abs < mean_gate) ? 0 : 1;

    free(out);
    sam3d_body_decoder_free(m);
    free(img_emb); free(rays); free(ref);
    return rc_out;
}
