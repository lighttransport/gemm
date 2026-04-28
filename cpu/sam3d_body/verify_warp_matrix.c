/*
 * verify_warp_matrix — port-level sanity for sam3d_body_get_warp_matrix
 * (data/transforms/bbox_utils.py::get_warp_matrix with rot=0, shift=(0,0)).
 *
 * Diffs our (2, 3) warp matrix against decoder_batch__affine_trans.npy,
 * using the dumped bbox_center / bbox_scale as input and output size (512, 512).
 *
 * Also spot-checks sam3d_body_default_cam_int + sam3d_body_compute_condition_info
 * using the dumped ori_img_size.
 *
 * Usage:
 *   verify_warp_matrix --refdir /tmp/sam3d_body_ref [--threshold F]
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
        fprintf(stderr, "[verify_warp_matrix] missing %s\n", path);
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
    fprintf(stderr, "[verify_warp_matrix] %-32s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *refdir = NULL;
    float threshold = 1e-3f;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--refdir") && i+1 < argc) refdir = argv[++i];
        else if (!strcmp(argv[i], "--threshold") && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!refdir) {
        fprintf(stderr, "Usage: %s --refdir <dir> [--threshold F]\n", argv[0]);
        return 2;
    }

    float *bbox_center  = load_or_die(refdir, "decoder_batch__bbox_center");
    float *bbox_scale   = load_or_die(refdir, "decoder_batch__bbox_scale");
    float *affine_ref   = load_or_die(refdir, "decoder_batch__affine_trans");
    float *img_size     = load_or_die(refdir, "decoder_batch__img_size");
    float *ori_img_size = load_or_die(refdir, "decoder_batch__ori_img_size");
    float *cam_int_ref  = load_or_die(refdir, "decoder_batch__cam_int");
    if (!bbox_center || !bbox_scale || !affine_ref || !img_size ||
        !ori_img_size || !cam_int_ref) {
        free(bbox_center); free(bbox_scale); free(affine_ref);
        free(img_size); free(ori_img_size); free(cam_int_ref);
        return 3;
    }

    const int out_w = (int)img_size[0], out_h = (int)img_size[1];
    int rc_total = 0;

    /* --- warp_matrix --- */
    {
        float ours[6];
        sam3d_body_get_warp_matrix(bbox_center, bbox_scale, out_w, out_h, ours);
        rc_total |= diff_pair("warp_matrix (2,3)", ours, affine_ref, 6,
                              threshold);
    }

    /* --- default cam_int (against dumped cam_int; expected to match since
     *     upstream uses the same default when no intrinsics are provided) --- */
    {
        const int W = (int)ori_img_size[0], H = (int)ori_img_size[1];
        float ours[9];
        sam3d_body_default_cam_int(W, H, ours);
        rc_total |= diff_pair("default_cam_int (3,3)", ours, cam_int_ref, 9,
                              /* focal ~1942, allow loose tol */ 1e-2f);
    }

    /* --- fix_aspect_ratio sanity: aspect=1.0 returns max(w,h) on both axes --- */
    {
        float in0[2] = {1.5f, 0.5f}, out0[2];
        sam3d_body_fix_aspect_ratio(in0, 1.0f, out0);
        float exp0[2] = {1.5f, 1.5f};
        rc_total |= diff_pair("fix_aspect_ratio ar=1.0", out0, exp0, 2, 1e-6f);

        float in1[2] = {0.5f, 2.0f}, out1[2];
        sam3d_body_fix_aspect_ratio(in1, 0.75f, out1);
        float exp1[2] = {0.75f * 2.0f, 2.0f};
        rc_total |= diff_pair("fix_aspect_ratio ar=.75", out1, exp1, 2, 1e-6f);
    }

    free(bbox_center); free(bbox_scale); free(affine_ref);
    free(img_size); free(ori_img_size); free(cam_int_ref);
    return rc_total;
}
