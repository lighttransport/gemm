/*
 * verify_decoder_forward — exercises the public sam3d_body_decoder_forward
 * wrapper standalone (the convenience entry point that bundles
 * ray_cond_emb + get_dense_pe + invalid_prompt_token + build_tokens +
 * forward_full).
 *
 * verify_decoder_full pre-computes the decoder inputs and only diffs
 * forward_full. This test instead drives the wrapper from the same starting
 * point a caller would: pre-ray patch tokens + cam intrinsics + bbox params,
 * letting the wrapper build everything internally. Same diff targets as
 * verify_decoder_full (head_pose_proj_raw / head_camera_proj_raw) so
 * pass/fail is comparable.
 *
 * Usage:
 *   verify_decoder_forward --safetensors-dir <dir> --mhr-assets <dir> \
 *                          --refdir <dir> [--threshold F] [-t N]
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAM3D_BODY_DECODER_IMPLEMENTATION
#define SAM3D_BODY_DECODER_FULL_IMPLEMENTATION
#define SAM3D_BODY_MHR_IMPLEMENTATION
#include "sam3d_body_decoder.h"
#include "sam3d_body_mhr.h"
#include "npy_io.h"

static float *load_or_die(const char *refdir, const char *name)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.npy", refdir, name);
    int nd, dims[8];
    float *d = (float *)npy_load(path, &nd, dims, NULL);
    if (!d) {
        fprintf(stderr, "[verify_decoder_forward] missing %s\n", path);
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
    fprintf(stderr, "[verify_decoder_forward] %-40s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *mhr_dir = NULL, *refdir = NULL;
    float threshold = 5e-3f;
    int n_threads = 1;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets")      && i+1 < argc) mhr_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !mhr_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --mhr-assets <dir> "
                "--refdir <dir> [--threshold F] [-t N]\n", argv[0]);
        return 2;
    }

    int rc_main = 1;
    sam3d_body_decoder_model *m = NULL;
    sam3d_body_mhr_assets *mhr = NULL;
    float *pre_ray_chw = NULL, *image_tokens_patch = NULL, *rays_hwc = NULL;
    float *cam_int = NULL, *bbox_center = NULL, *bbox_scale = NULL;
    float *ori_img_size = NULL, *img_size = NULL, *affine_trans = NULL;
    float *ref_pose = NULL, *ref_cam = NULL, *ref_kp3d = NULL;
    float *ref_kp2d_crop = NULL, *ref_kp2d_dep = NULL, *ref_cam_t = NULL;
    sam3d_body_decoder_full_result r;
    memset(&r, 0, sizeof(r));

    char p[1024], p2[1024];
    snprintf(p,  sizeof(p),  "%s/sam3d_body_decoder.safetensors", sft_dir);
    snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_head.safetensors", sft_dir);
    m = sam3d_body_decoder_load(p, p2);
    if (!m) goto done;

    snprintf(p,  sizeof(p),  "%s/sam3d_body_mhr_jit.safetensors", mhr_dir);
    snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_jit.json", mhr_dir);
    mhr = sam3d_body_mhr_load(p, p2);
    if (!mhr) goto done;

    const int H = 32, W = 32;
    const int Dc = m->kv_dim;       /* 1280 */
    const int K = m->n_keypoints;   /* 70   */
    const int V = 18439;

    /* Pre-ray patch tokens — load CHW dump, permute to (H*W, Dc). */
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/ray_cond__image_embeddings_pre_ray.npy", refdir);
        int nd, dims[8];
        pre_ray_chw = (float *)npy_load(path, &nd, dims, NULL);
        if (!pre_ray_chw) {
            fprintf(stderr, "[verify_decoder_forward] missing %s\n", path);
            goto done;
        }
    }
    image_tokens_patch = (float *)malloc((size_t)H * W * Dc * sizeof(float));
    if (!image_tokens_patch) goto done;
    for (int n = 0; n < H * W; n++)
        for (int c = 0; c < Dc; c++)
            image_tokens_patch[(size_t)n * Dc + c] =
                pre_ray_chw[(size_t)c * H * W + n];

    cam_int       = load_or_die(refdir, "decoder_batch__cam_int");
    bbox_center   = load_or_die(refdir, "decoder_batch__bbox_center");
    bbox_scale    = load_or_die(refdir, "decoder_batch__bbox_scale");
    ori_img_size  = load_or_die(refdir, "decoder_batch__ori_img_size");
    img_size      = load_or_die(refdir, "decoder_batch__img_size");
    affine_trans  = load_or_die(refdir, "decoder_batch__affine_trans");
    if (!cam_int || !bbox_center || !bbox_scale || !ori_img_size ||
        !img_size || !affine_trans) goto done;

    sam3d_body_camera_batch B;
    memcpy(B.cam_int,      cam_int,      9 * sizeof(float));
    memcpy(B.bbox_center,  bbox_center,  2 * sizeof(float));
    B.bbox_scale = bbox_scale[0];
    memcpy(B.ori_img_size, ori_img_size, 2 * sizeof(float));
    memcpy(B.img_size,     img_size,     2 * sizeof(float));
    memcpy(B.affine_trans, affine_trans, 6 * sizeof(float));
    B.use_intrin_center    = 0;
    B.default_scale_factor = 1.0f;

    /* img_size is (W, H); compute_ray_cond_xyz takes (img_h, img_w). */
    rays_hwc = (float *)malloc((size_t)H * W * 3 * sizeof(float));
    if (!rays_hwc) goto done;
    int rc = sam3d_body_compute_ray_cond_xyz(cam_int, affine_trans,
                                             (int)img_size[1],
                                             (int)img_size[0],
                                             H, W, rays_hwc);
    if (rc != 0) {
        fprintf(stderr, "[verify_decoder_forward] compute_ray_cond_xyz rc=%d\n", rc);
        goto done;
    }

    float bbox_scale1[1] = { bbox_scale[0] };
    float condition_info[3];
    sam3d_body_compute_condition_info(bbox_center, bbox_scale1,
                                      ori_img_size, cam_int,
                                      /*use_intrin_center=*/0,
                                      condition_info);

    r.pred_vertices = (float *)malloc((size_t)V * 3 * sizeof(float));
    if (!r.pred_vertices) goto done;

    rc = sam3d_body_decoder_forward(
            m, (struct sam3d_body_mhr_assets_t *)mhr, &B,
            image_tokens_patch, H, W,
            rays_hwc, condition_info,
            n_threads, &r);
    if (rc != 0) {
        fprintf(stderr, "[verify_decoder_forward] forward rc=%d\n", rc);
        goto done;
    }

    ref_pose      = load_or_die(refdir, "head_pose_proj_raw");
    ref_cam       = load_or_die(refdir, "head_camera_proj_raw");
    ref_kp3d      = load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_3d");
    ref_kp2d_crop = load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_2d_cropped");
    ref_kp2d_dep  = load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_2d_depth");
    ref_cam_t     = load_or_die(refdir, "decoder_pose_layer5__pred_cam_t");
    if (!ref_pose || !ref_cam || !ref_kp3d || !ref_kp2d_crop ||
        !ref_kp2d_dep || !ref_cam_t) goto done;

    {
        const float *ip = (const float *)m->init_pose.data;
        const float *ic = (const float *)m->init_camera.data;
        float pose_raw[519], cam_raw[3];
        for (int i = 0; i < 519; i++) pose_raw[i] = r.mhr_params[i] - ip[i];
        for (int i = 0; i < 3;   i++) cam_raw[i]  = r.cam_t[i]      - ic[i];

        int rc_total = 0;
        rc_total |= diff_pair("head_pose_proj_raw (519,)", pose_raw, ref_pose, 519, threshold);
        rc_total |= diff_pair("head_camera_proj_raw (3,)", cam_raw,  ref_cam,  3,   threshold);
        rc_total |= diff_pair("final pred_keypoints_3d (70,3)",
                              r.pred_keypoints_3d, ref_kp3d, K * 3, threshold);
        rc_total |= diff_pair("final pred_keypoints_2d_cropped (70,2)",
                              r.pred_keypoints_2d_cropped, ref_kp2d_crop, K * 2, threshold);
        rc_total |= diff_pair("final pred_keypoints_2d_depth (70,)",
                              r.pred_keypoints_2d_depth, ref_kp2d_dep, K, threshold);
        rc_total |= diff_pair("final pred_cam_t (3,)",
                              r.pred_cam_t_world, ref_cam_t, 3, threshold);
        rc_main = rc_total;
    }

done:
    free(pre_ray_chw); free(image_tokens_patch); free(rays_hwc);
    free(cam_int); free(bbox_center); free(bbox_scale);
    free(ori_img_size); free(img_size); free(affine_trans);
    free(ref_pose); free(ref_cam); free(ref_kp3d);
    free(ref_kp2d_crop); free(ref_kp2d_dep); free(ref_cam_t);
    free(r.pred_vertices);
    if (mhr) sam3d_body_mhr_free(mhr);
    if (m)   sam3d_body_decoder_free(m);
    return rc_main;
}
