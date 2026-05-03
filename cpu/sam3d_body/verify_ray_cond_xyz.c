/*
 * verify_ray_cond_xyz — port-level sanity for sam3d_body_compute_ray_cond_xyz
 * (meta_arch.sam3d_body.get_ray_condition + CameraEncoder antialias
 * downsample + z-append).
 *
 * Diffs the (1, 32, 32, 3) output against ray_cond_ds_xyz.npy.
 *
 * Usage:
 *   verify_ray_cond_xyz --refdir /tmp/sam3d_body_ref [--threshold F]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "npy_io.h"

static float *load_or_die(const char *refdir, const char *name)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    int nd, dims[8];
    float *d = (float *)npy_load(path, &nd, dims, NULL);
    if (!d) {
        fprintf(stderr, "[verify_ray_cond_xyz] missing %s\n", path);
        return NULL;
    }
    return d;
}

static float *load_or_die_dims(const char *refdir, const char *name,
                               int *out_nd, int out_dims[8])
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    float *d = (float *)npy_load(path, out_nd, out_dims, NULL);
    if (!d) {
        fprintf(stderr, "[verify_ray_cond_xyz] missing %s\n", path);
        return NULL;
    }
    return d;
}

int main(int argc, char **argv)
{
    const char *refdir = NULL;
    float threshold = 1e-5f;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!refdir) {
        fprintf(stderr, "Usage: %s --refdir <dir> [--threshold F]\n", argv[0]);
        return 2;
    }

    float *cam_int      = load_or_die(refdir, "decoder_batch__cam_int");
    float *affine_trans = load_or_die(refdir, "decoder_batch__affine_trans");
    float *img_size     = load_or_die(refdir, "decoder_batch__img_size");
    int ref_nd = 0, ref_dims[8] = {0};
    float *ref_ds       = load_or_die_dims(refdir, "ray_cond_ds_xyz",
                                           &ref_nd, ref_dims);
    if (!cam_int || !affine_trans || !img_size || !ref_ds) {
        free(cam_int); free(affine_trans); free(img_size); free(ref_ds);
        return 3;
    }
    if (ref_nd != 4 || ref_dims[0] != 1 || ref_dims[1] <= 0 ||
        ref_dims[2] <= 0 || ref_dims[3] != 3) {
        fprintf(stderr, "[verify_ray_cond_xyz] invalid ray_cond_ds_xyz shape\n");
        free(cam_int); free(affine_trans); free(img_size); free(ref_ds);
        return 3;
    }

    /* img_size is (B, 1, 2) = [H_in, W_in]. */
    const int H_in = (int)img_size[0];
    const int W_in = (int)img_size[1];
    const int H_out = ref_dims[1], W_out = ref_dims[2];
    fprintf(stderr, "[verify_ray_cond_xyz] H_in=%d W_in=%d → H_out=%d W_out=%d\n",
            H_in, W_in, H_out, W_out);

    float *ours = (float *)malloc((size_t)H_out * W_out * 3 * sizeof(float));
    int rc = sam3d_body_compute_ray_cond_xyz(cam_int, affine_trans,
                                              H_in, W_in, H_out, W_out, ours);
    if (rc != SAM3D_BODY_DECODER_E_OK) {
        fprintf(stderr, "[verify_ray_cond_xyz] compute rc=%d\n", rc);
        free(cam_int); free(affine_trans); free(img_size); free(ref_ds); free(ours);
        return 4;
    }

    double sum = 0.0; float mx = 0.0f; size_t mxi = 0;
    const size_t total = (size_t)H_out * W_out * 3;
    for (size_t i = 0; i < total; i++) {
        float d = fabsf(ours[i] - ref_ds[i]);
        if (d > mx) { mx = d; mxi = i; }
        sum += d;
    }
    double mean = sum / (double)total;
    int fail = (mx >= threshold);
    fprintf(stderr, "[verify_ray_cond_xyz] %-32s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            "ray_cond_xyz", mx, mxi, mean, threshold,
            fail ? "FAIL" : "OK");

    free(cam_int); free(affine_trans); free(img_size); free(ref_ds); free(ours);
    return fail ? 1 : 0;
}
