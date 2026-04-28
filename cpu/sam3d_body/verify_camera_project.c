/*
 * verify_camera_project — port-level sanity for sam3d_body_camera_project
 * (PerspectiveHead.perspective_projection + BaseModel._full_to_crop).
 *
 * For each layer i in 0..5, feed:
 *   pred_keypoints_3d (from MHRHead, post-flip pre-cam_t)
 *   pred_cam          (head_camera + init_camera)
 *   decoder_batch__*  (cam_int, bbox_center, bbox_scale, ori_img_size,
 *                      img_size, affine_trans)
 * and diff against:
 *   pred_keypoints_2d, pred_keypoints_2d_cropped,
 *   pred_keypoints_2d_depth, pred_cam_t.
 *
 * Usage:
 *   verify_camera_project --refdir /tmp/sam3d_body_ref [--threshold F]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* We only need the camera_project helper from the decoder header. The
 * full implementation block also pulls in safetensors + the rest, but
 * loads no model — we never call sam3d_body_decoder_load. */
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
        fprintf(stderr, "[verify_camera_project] missing %s\n", path);
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
    int fail = (mx >= thresh) || (mean >= thresh * 0.15f);
    fprintf(stderr, "[verify_camera_project] %-44s max_abs=%.4e (i=%zu) "
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

    /* Load batch params (shapes (1, 1, ...) flatten into the typed buffer). */
    float *cam_int       = load_or_die(refdir, "decoder_batch__cam_int");
    float *bbox_center   = load_or_die(refdir, "decoder_batch__bbox_center");
    float *bbox_scale    = load_or_die(refdir, "decoder_batch__bbox_scale");
    float *ori_img_size  = load_or_die(refdir, "decoder_batch__ori_img_size");
    float *img_size      = load_or_die(refdir, "decoder_batch__img_size");
    float *affine_trans  = load_or_die(refdir, "decoder_batch__affine_trans");
    if (!cam_int || !bbox_center || !bbox_scale || !ori_img_size ||
        !img_size || !affine_trans) {
        free(cam_int); free(bbox_center); free(bbox_scale);
        free(ori_img_size); free(img_size); free(affine_trans);
        return 3;
    }

    sam3d_body_camera_batch B;
    memcpy(B.cam_int,      cam_int,      9 * sizeof(float));
    memcpy(B.bbox_center,  bbox_center,  2 * sizeof(float));
    B.bbox_scale = bbox_scale[0];
    memcpy(B.ori_img_size, ori_img_size, 2 * sizeof(float));
    memcpy(B.img_size,     img_size,     2 * sizeof(float));
    memcpy(B.affine_trans, affine_trans, 6 * sizeof(float));
    B.use_intrin_center    = 0;
    B.default_scale_factor = 1.0f;

    free(cam_int); free(bbox_center); free(bbox_scale);
    free(ori_img_size); free(img_size); free(affine_trans);

    const int K = 70;
    float kp2d[K * 2], kp2d_crop[K * 2], kp2d_depth[K], cam_t[3];
    int rc_total = 0;
    for (int li = 0; li < 6; li++) {
        char nm[128];

        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_3d", li);
        float *kp3d = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_cam", li);
        float *pcam = load_or_die(refdir, nm);

        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d", li);
        float *ref_kp2d = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_cropped", li);
        float *ref_kp2d_crop = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_depth", li);
        float *ref_kp2d_depth = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_cam_t", li);
        float *ref_cam_t = load_or_die(refdir, nm);

        if (!kp3d || !pcam || !ref_kp2d || !ref_kp2d_crop ||
            !ref_kp2d_depth || !ref_cam_t) {
            free(kp3d); free(pcam); free(ref_kp2d); free(ref_kp2d_crop);
            free(ref_kp2d_depth); free(ref_cam_t);
            rc_total = 4; break;
        }

        int rc = sam3d_body_camera_project(kp3d, pcam, &B, K,
                                           kp2d, kp2d_crop,
                                           kp2d_depth, cam_t);
        if (rc) {
            fprintf(stderr, "[verify_camera_project] layer %d rc=%d\n", li, rc);
            free(kp3d); free(pcam); free(ref_kp2d); free(ref_kp2d_crop);
            free(ref_kp2d_depth); free(ref_cam_t);
            rc_total = 5; break;
        }

        char lbl[64];
        snprintf(lbl, sizeof(lbl), "layer%d cam_t (3,)",         li);
        rc_total |= diff_pair(lbl, cam_t,      ref_cam_t,      3,         threshold);
        snprintf(lbl, sizeof(lbl), "layer%d kp2d_depth (70,)",   li);
        rc_total |= diff_pair(lbl, kp2d_depth, ref_kp2d_depth, K,         threshold);
        snprintf(lbl, sizeof(lbl), "layer%d kp2d (70,2)",        li);
        rc_total |= diff_pair(lbl, kp2d,       ref_kp2d,       K * 2,     threshold);
        snprintf(lbl, sizeof(lbl), "layer%d kp2d_cropped (70,2)", li);
        rc_total |= diff_pair(lbl, kp2d_crop,  ref_kp2d_crop,  K * 2,     threshold);

        free(kp3d); free(pcam); free(ref_kp2d); free(ref_kp2d_crop);
        free(ref_kp2d_depth); free(ref_cam_t);
    }

    return rc_total;
}
