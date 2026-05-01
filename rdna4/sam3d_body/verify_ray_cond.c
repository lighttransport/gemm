/*
 * verify_ray_cond (CUDA) — diff CUDA ray_cond_emb output against
 * /tmp/sam3d_body_ref/image_embeddings_after_ray.npy.
 *
 * Inputs (shared with the CPU port):
 *   <refdir>/ray_cond__image_embeddings_pre_ray.npy  (1,1280,32,32) f32
 *   <refdir>/ray_cond_ds_xyz.npy                     (1,32,32,3)    f32
 *   <refdir>/image_embeddings_after_ray.npy          (1,1280,32,32) f32
 */

#include "hip_sam3d_body_runner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../common/npy_io.h"

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    /* CPU port hits ~1e-3 max_abs vs PyTorch on this stage; CUDA reproduces
     * CPU bit-tightly. Gates set at the CPU floor. */
    float threshold = 2e-3f;
    float mean_threshold = 3e-4f;
    int device = 0, verbose = 0;
    const char *precision = "bf16";
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--mean-threshold") && i+1 < argc) mean_threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--device") && i+1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--precision") && i+1 < argc) precision = argv[++i];
        else if (!strcmp(argv[i], "-v")) verbose = 1;
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir DIR --refdir DIR "
                        "[--threshold F] [--mean-threshold F] "
                        "[--device N] [--precision bf16|fp16] [-v]\n",
                argv[0]);
        return 2;
    }

    char path[1024];
    int nd = 0, dims[8] = {0};

    /* (1, 1280, 32, 32) f32 image_emb_pre_ray */
    snprintf(path, sizeof(path), "%s/ray_cond__image_embeddings_pre_ray.npy", refdir);
    float *img_emb = (float *)npy_load(path, &nd, dims, NULL);
    if (!img_emb || nd != 4 || dims[0] != 1 || dims[1] != 1280 ||
        dims[2] != 32 || dims[3] != 32) {
        fprintf(stderr, "[cuda verify_ray_cond] missing/invalid %s\n", path);
        free(img_emb); return 3;
    }
    int C = dims[1], H = dims[2], W = dims[3];

    /* (1, 32, 32, 3) f32 rays */
    snprintf(path, sizeof(path), "%s/ray_cond_ds_xyz.npy", refdir);
    float *rays = (float *)npy_load(path, &nd, dims, NULL);
    if (!rays || nd != 4 || dims[0] != 1 || dims[1] != H ||
        dims[2] != W || dims[3] != 3) {
        fprintf(stderr, "[cuda verify_ray_cond] missing/invalid %s\n", path);
        free(img_emb); free(rays); return 3;
    }

    /* (1, 1280, 32, 32) reference output */
    snprintf(path, sizeof(path), "%s/image_embeddings_after_ray.npy", refdir);
    float *ref = (float *)npy_load(path, &nd, dims, NULL);
    if (!ref || nd != 4 || dims[1] != C || dims[2] != H || dims[3] != W) {
        fprintf(stderr, "[cuda verify_ray_cond] missing/invalid %s\n", path);
        free(img_emb); free(rays); free(ref); return 3;
    }
    fprintf(stderr, "[cuda verify_ray_cond] ref: img=(1,%d,%d,%d) "
                    "rays=(1,%d,%d,3) out=(1,%d,%d,%d)\n",
            C, H, W, H, W, C, H, W);

    hip_sam3d_body_config cfg = {
        .safetensors_dir = sft_dir,
        .image_size      = 512,
        .device_ordinal  = device,
        .verbose         = verbose,
        .precision       = precision,
    };
    hip_sam3d_body_ctx *ctx = hip_sam3d_body_create(&cfg);
    if (!ctx) {
        fprintf(stderr, "create failed\n");
        free(img_emb); free(rays); free(ref); return 5;
    }

    float *out = (float *)malloc((size_t)C * H * W * sizeof(float));
    if (!out) {
        hip_sam3d_body_destroy(ctx);
        free(img_emb); free(rays); free(ref); return 6;
    }
    int rc = hip_sam3d_body_debug_run_ray_cond(ctx, img_emb, rays, H, W, out);
    if (rc != 0) {
        fprintf(stderr, "debug_run_ray_cond rc=%d\n", rc);
        free(out); hip_sam3d_body_destroy(ctx);
        free(img_emb); free(rays); free(ref); return 7;
    }

    /* Diff: out[c, h, w] vs ref[0, c, h, w]. Both (C, H, W). */
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
    fprintf(stderr, "[cuda verify_ray_cond] C=%d H=%d W=%d  "
                    "max_abs=%.6e (c=%d h=%d w=%d)  mean_abs=%.6e  "
                    "(max_gate=%.1e mean_gate=%.1e)\n",
            C, H, W, mx, mx_c, mx_h, mx_w, mean_abs, threshold, mean_threshold);
    int rc_out = (mx < threshold && mean_abs < mean_threshold) ? 0 : 1;

    free(out);
    hip_sam3d_body_destroy(ctx);
    free(img_emb); free(rays); free(ref);
    return rc_out;
}
