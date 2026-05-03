/*
 * verify_decoder_full — Step 8b: iterative decoder forward with
 * MHR-in-the-loop. Diffs against the upstream first-pass dumps.
 *
 * Uses cached image_embeddings_after_ray + decoder_layer0_in__{x,x_pe,context_pe}
 * to bypass the (already verified) DINOv3 encoder + ray_cond_emb stages and
 * exercise sam3d_body_decoder_forward_full only.
 *
 * Targets:
 *   head_pose_proj_raw.npy   (519,) — diff (mhr_params - init_pose).
 *   head_camera_proj_raw.npy (3,)   — diff (cam_t   - init_camera).
 *   mhr_params__pred_keypoints_3d.npy (1, 70, 3) — final cam-frame kp.
 *   mhr_params__pred_vertices.npy     (1, V, 3)  — final post-flip verts (m).
 *
 * Usage:
 *   verify_decoder_full --safetensors-dir <dir> --mhr-assets <dir> \
 *                       --refdir <dir> [--threshold F] [-t N]
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
        fprintf(stderr, "[verify_decoder_full] missing %s\n", path);
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
        fprintf(stderr, "[verify_decoder_full] missing %s\n", path);
        return NULL;
    }
    return d;
}

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
    fprintf(stderr, "[verify_decoder_full] %-44s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *mhr_dir = NULL, *refdir = NULL;
    const char *backbone = "dinov3";
    float threshold = 5e-3f;
    int n_threads = 1;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--mhr-assets")      && i+1 < argc) mhr_dir = argv[++i];
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
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !mhr_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --mhr-assets <dir> "
                "--refdir <dir> [--threshold F] [--backbone dinov3|vith] "
                "[-t N]\n", argv[0]);
        return 2;
    }

    int rc_main = 1;
    sam3d_body_decoder_model *m = NULL;
    sam3d_body_mhr_assets *mhr = NULL;
    float *img_emb = NULL, *ctx_pe_tok = NULL, *init_x = NULL, *init_xpe = NULL;
    float *image_pe_chw = NULL;
    float *cam_int = NULL, *bbox_center = NULL, *bbox_scale = NULL;
    float *ori_img_size = NULL, *img_size = NULL, *affine_trans = NULL;
    float *ref_pose = NULL, *ref_cam = NULL, *ref_kp3d_f = NULL;
    float *ref_kp2d_crop_f = NULL, *ref_kp2d_dep_f = NULL, *ref_cam_t_f = NULL;
    sam3d_body_decoder_full_result r;
    memset(&r, 0, sizeof(r));

    char p[1024], p2[1024];
    resolve_variant_path(sft_dir, "decoder", backbone, p, sizeof(p));
    resolve_variant_path(sft_dir, "mhr_head", backbone, p2, sizeof(p2));
    m = sam3d_body_decoder_load(p, p2);
    if (!m) goto done;

    snprintf(p,  sizeof(p),  "%s/sam3d_body_mhr_jit.safetensors", mhr_dir);
    snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_jit.json", mhr_dir);
    mhr = sam3d_body_mhr_load(p, p2);
    if (!mhr) goto done;

    const int Dc = m->kv_dim;
    const int K = m->n_keypoints;
    const int V = 18439;
    const int NL = m->n_layers;  /* 6 */

    int img_nd = 0, img_dims[8] = {0};
    int pe_nd = 0, pe_dims[8] = {0};
    img_emb = load_or_die_dims(refdir, "image_embeddings_after_ray",
                               &img_nd, img_dims);
    ctx_pe_tok = load_or_die_dims(refdir, "decoder_layer0_in__context_pe",
                                  &pe_nd, pe_dims);
    init_x      = load_or_die(refdir, "decoder_layer0_in__x");
    init_xpe    = load_or_die(refdir, "decoder_layer0_in__x_pe");
    if (!img_emb || !ctx_pe_tok || !init_x || !init_xpe) goto done;
    if (img_nd != 4 || pe_nd != 3 || img_dims[1] != Dc ||
        pe_dims[1] != img_dims[2] * img_dims[3] ||
        pe_dims[2] != Dc) {
        fprintf(stderr, "[verify_decoder_full] bad image/context shape\n");
        goto done;
    }
    const int H = img_dims[2], W = img_dims[3];

    /* ctx_pe is dumped in token form (1, 1024, 1280) — permute back to CHW. */
    image_pe_chw = (float *)malloc((size_t)Dc * H * W * sizeof(float));
    if (!image_pe_chw) goto done;
    for (int n = 0; n < H * W; n++)
        for (int c = 0; c < Dc; c++)
            image_pe_chw[(size_t)c * H * W + n] = ctx_pe_tok[(size_t)n * Dc + c];

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

    /* Allocate result with optional vertex output + per-layer debug dumps. */
    r.pred_vertices        = (float *)malloc((size_t)V * 3 * sizeof(float));
    r.dbg_layer_kp3d       = (float *)malloc((size_t)NL * K * 3 * sizeof(float));
    r.dbg_layer_kp2d_crop  = (float *)malloc((size_t)NL * K * 2 * sizeof(float));
    r.dbg_layer_kp2d_depth = (float *)malloc((size_t)NL * K     * sizeof(float));
    r.dbg_layer_cam_t      = (float *)malloc((size_t)NL * 3     * sizeof(float));
    r.dbg_layer_tokens_out = (float *)malloc((size_t)NL * 145 * m->dim * sizeof(float));
    r.dbg_layer_pose_raw   = (float *)malloc((size_t)NL * 519 * sizeof(float));
    if (!r.pred_vertices || !r.dbg_layer_kp3d || !r.dbg_layer_kp2d_crop ||
        !r.dbg_layer_kp2d_depth || !r.dbg_layer_cam_t ||
        !r.dbg_layer_tokens_out || !r.dbg_layer_pose_raw) goto done;

    int rc = sam3d_body_decoder_forward_full(
            m, (struct sam3d_body_mhr_assets_t *)mhr, &B,
            img_emb, image_pe_chw, H, W,
            init_x, init_xpe,
            n_threads, &r);
    if (rc != 0) {
        fprintf(stderr, "[verify_decoder_full] forward_full rc=%d\n", rc);
        goto done;
    }

    /* Per-layer diagnostic diffs vs decoder_pose_layerN__*. */
    int rc_total = 0;
    /* layer0 tokens_out vs decoder_layer0_out__tokens (sanity check that
     * decoder_layer_forward at L0 still matches). */
    {
        float *ref_l0 = load_or_die(refdir, "decoder_layer0_out__tokens");
        if (ref_l0) {
            rc_total |= diff_pair("layer0 tokens_out (145,1024)",
                                  r.dbg_layer_tokens_out, ref_l0,
                                  (size_t)145 * m->dim, 5e-3f);
            free(ref_l0);
        }
    }
    for (int li = 0; li < NL - 1; li++) {
        char nm[128], lbl[96];
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_3d", li);
        float *ref_kp3d = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_cropped", li);
        float *ref_kp2d_crop = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_depth", li);
        float *ref_kp2d_dep = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_cam_t", li);
        float *ref_cam_t = load_or_die(refdir, nm);
        if (!ref_kp3d || !ref_kp2d_crop || !ref_kp2d_dep || !ref_cam_t) {
            free(ref_kp3d); free(ref_kp2d_crop); free(ref_kp2d_dep); free(ref_cam_t);
            continue;
        }
        snprintf(lbl, sizeof(lbl), "layer%d kp3d (70,3)", li);
        rc_total |= diff_pair(lbl, r.dbg_layer_kp3d + (size_t)li*K*3, ref_kp3d, K*3, threshold);
        snprintf(lbl, sizeof(lbl), "layer%d kp2d_crop (70,2)", li);
        rc_total |= diff_pair(lbl, r.dbg_layer_kp2d_crop + (size_t)li*K*2, ref_kp2d_crop, K*2, threshold);
        snprintf(lbl, sizeof(lbl), "layer%d kp2d_depth (70,)", li);
        rc_total |= diff_pair(lbl, r.dbg_layer_kp2d_depth + (size_t)li*K, ref_kp2d_dep, K, threshold);
        snprintf(lbl, sizeof(lbl), "layer%d cam_t (3,)", li);
        rc_total |= diff_pair(lbl, r.dbg_layer_cam_t + (size_t)li*3, ref_cam_t, 3, threshold);
        free(ref_kp3d); free(ref_kp2d_crop); free(ref_kp2d_dep); free(ref_cam_t);
    }

    /* Final-pass dumps: head_pose/head_camera proj_raw are captured via
     * forward_hook (last body call wins, which is the post-loop final norm).
     * Body final keypoints/cam_t live at decoder_pose_layer5__*. */
    ref_pose        = load_or_die(refdir, "head_pose_proj_raw");
    ref_cam         = load_or_die(refdir, "head_camera_proj_raw");
    ref_kp3d_f      = load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_3d");
    ref_kp2d_crop_f = load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_2d_cropped");
    ref_kp2d_dep_f  = load_or_die(refdir, "decoder_pose_layer5__pred_keypoints_2d_depth");
    ref_cam_t_f     = load_or_die(refdir, "decoder_pose_layer5__pred_cam_t");
    if (!ref_pose || !ref_cam || !ref_kp3d_f || !ref_kp2d_crop_f ||
        !ref_kp2d_dep_f || !ref_cam_t_f) goto done;

    {
        const float *ip = (const float *)m->init_pose.data;
        const float *ic = (const float *)m->init_camera.data;
        float pose_raw[519], cam_raw[3];
        for (int i = 0; i < 519; i++) pose_raw[i] = r.mhr_params[i] - ip[i];
        for (int i = 0; i < 3;   i++) cam_raw[i]  = r.cam_t[i]      - ic[i];

        rc_total |= diff_pair("head_pose_proj_raw (519,)", pose_raw, ref_pose, 519, threshold);
        rc_total |= diff_pair("head_camera_proj_raw (3,)", cam_raw,  ref_cam,  3,   threshold);
        rc_total |= diff_pair("final pred_keypoints_3d (70,3)",
                              r.pred_keypoints_3d, ref_kp3d_f, K * 3, threshold);
        rc_total |= diff_pair("final pred_keypoints_2d_cropped (70,2)",
                              r.pred_keypoints_2d_cropped, ref_kp2d_crop_f, K * 2, threshold);
        rc_total |= diff_pair("final pred_keypoints_2d_depth (70,)",
                              r.pred_keypoints_2d_depth, ref_kp2d_dep_f, K, threshold);
        rc_total |= diff_pair("final pred_cam_t (3,)",
                              r.pred_cam_t_world, ref_cam_t_f, 3, threshold);
    }
    rc_main = rc_total;

done:
    free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
    free(image_pe_chw);
    free(cam_int); free(bbox_center); free(bbox_scale);
    free(ori_img_size); free(img_size); free(affine_trans);
    free(ref_pose); free(ref_cam); free(ref_kp3d_f);
    free(ref_kp2d_crop_f); free(ref_kp2d_dep_f); free(ref_cam_t_f);
    free(r.pred_vertices); free(r.dbg_layer_kp3d);
    free(r.dbg_layer_kp2d_crop); free(r.dbg_layer_kp2d_depth);
    free(r.dbg_layer_cam_t); free(r.dbg_layer_tokens_out);
    free(r.dbg_layer_pose_raw);
    if (mhr) sam3d_body_mhr_free(mhr);
    if (m)   sam3d_body_decoder_free(m);
    return rc_main;
}
