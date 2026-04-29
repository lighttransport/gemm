/*
 * verify_decoder_e2e — full chained decoder forward (preset path) +
 * head_pose / head_camera regression. Diffs against the upstream dumps:
 *
 *   head_pose_proj_input.npy   (1024,)  — input to head_pose.proj
 *                                          (= norm_final(tokens)[0,0])
 *   head_pose_proj_raw.npy     (519,)   — head_pose.proj output
 *                                          (BEFORE adding init_pose)
 *   head_camera_proj_raw.npy   (3,)     — head_camera.proj output
 *                                          (BEFORE adding init_camera)
 *
 * Inputs (loaded from REFDIR):
 *   image_embeddings_after_ray.npy   (1, 1280, 32, 32)
 *   decoder_layer0_in__context_pe.npy (1, 1024, 1280) → permuted back to CHW
 *   decoder_layer0_in__x.npy         (1, 145, 1024)
 *   decoder_layer0_in__x_pe.npy      (1, 145, 1024)
 *   decoder_pose_layer{0..4}__pred_keypoints_2d_cropped.npy (1, 70, 2)
 *   decoder_pose_layer{0..4}__pred_keypoints_2d_depth.npy   (1, 70)
 *   decoder_pose_layer{0..4}__pred_keypoints_3d.npy         (1, 70, 3)
 *
 * Usage:
 *   verify_decoder_e2e --safetensors-dir <dir> --refdir <dir>
 *                      [--threshold F] [-t N]
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
        fprintf(stderr, "[verify_decoder_e2e] missing %s\n", path);
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
    fprintf(stderr, "[verify_decoder_e2e] %-32s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    float threshold = 5e-3f;
    int n_threads = 1;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F] [-t N]\n", argv[0]);
        return 2;
    }

    char p[1024];
    snprintf(p, sizeof(p), "%s/sam3d_body_decoder.safetensors", sft_dir);
    char p2[1024];
    snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_head.safetensors", sft_dir);
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(p, p2);
    if (!m) return 3;

    const int H = 32, W = 32;
    const int N_Q = 145, D = m->dim, K = m->n_keypoints;
    const int Dc = m->kv_dim;

    /* Load image embeddings + image PE. The PE dump is in token form
     * (N_C, Dc) via decoder_layer0_in__context_pe — permute back to CHW. */
    float *img_emb = load_or_die(refdir, "image_embeddings_after_ray");
    float *ctx_pe_tok = load_or_die(refdir, "decoder_layer0_in__context_pe");
    float *init_x  = load_or_die(refdir, "decoder_layer0_in__x");
    float *init_xpe = load_or_die(refdir, "decoder_layer0_in__x_pe");
    if (!img_emb || !ctx_pe_tok || !init_x || !init_xpe) {
        free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
        sam3d_body_decoder_free(m);
        return 4;
    }
    /* ctx_pe_tok shape (1, 1024, 1280); permute to (1280, 32, 32). */
    float *image_pe_chw = (float *)malloc((size_t)Dc * H * W * sizeof(float));
    if (!image_pe_chw) {
        free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
        sam3d_body_decoder_free(m);
        return 5;
    }
    for (int n = 0; n < H * W; n++)
        for (int c = 0; c < Dc; c++)
            image_pe_chw[(size_t)c * H * W + n] = ctx_pe_tok[(size_t)n * Dc + c];

    /* Per-layer kp inputs (5 intermediate layers). */
    float *kp2d_pl = (float *)malloc((size_t)5 * K * 2 * sizeof(float));
    float *dep_pl  = (float *)malloc((size_t)5 * K * sizeof(float));
    float *kp3d_pl = (float *)malloc((size_t)5 * K * 3 * sizeof(float));
    if (!kp2d_pl || !dep_pl || !kp3d_pl) {
        free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
        free(image_pe_chw); free(kp2d_pl); free(dep_pl); free(kp3d_pl);
        sam3d_body_decoder_free(m);
        return 6;
    }
    int rc = 0;
    for (int li = 0; li < 5; li++) {
        char nm[128];
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_cropped", li);
        float *a = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_depth", li);
        float *b = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_3d", li);
        float *c = load_or_die(refdir, nm);
        if (!a || !b || !c) { rc = 7; free(a); free(b); free(c); break; }
        memcpy(kp2d_pl + (size_t)li * K * 2, a, (size_t)K * 2 * sizeof(float));
        memcpy(dep_pl  + (size_t)li * K,     b, (size_t)K     * sizeof(float));
        memcpy(kp3d_pl + (size_t)li * K * 3, c, (size_t)K * 3 * sizeof(float));
        free(a); free(b); free(c);
    }
    if (rc) {
        free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
        free(image_pe_chw); free(kp2d_pl); free(dep_pl); free(kp3d_pl);
        sam3d_body_decoder_free(m);
        return rc;
    }

    /* Run end-to-end preset forward. */
    sam3d_body_decoder_result r;
    rc = sam3d_body_decoder_forward_preset(
        m, img_emb, image_pe_chw, H, W,
        init_x, init_xpe,
        kp2d_pl, dep_pl, kp3d_pl,
        n_threads, &r);
    free(img_emb); free(ctx_pe_tok); free(init_x); free(init_xpe);
    free(image_pe_chw); free(kp2d_pl); free(dep_pl); free(kp3d_pl);
    if (rc != SAM3D_BODY_DECODER_E_OK) {
        fprintf(stderr, "[verify_decoder_e2e] forward_preset rc=%d\n", rc);
        sam3d_body_decoder_free(m);
        return 8;
    }

    /* mhr_params (519,) = head_pose.proj(pose_token) + init_pose.
     * head_pose_proj_raw is the post-add value too (per ref dumper) — they
     * should match directly. cam_t same story for head_camera. */
    float *ref_pose = load_or_die(refdir, "head_pose_proj_raw");
    float *ref_cam  = load_or_die(refdir, "head_camera_proj_raw");
    if (!ref_pose || !ref_cam) {
        free(ref_pose); free(ref_cam);
        sam3d_body_decoder_free(m);
        return 9;
    }

    /* head_pose_proj_raw / head_camera_proj_raw are captured via forward_hook
     * on the inner `.proj` FFN — that is the PRE add-init_pose value (MHRHead
     * does `pred = self.proj(x); pred = pred + init_estimate` in its parent
     * forward). r.mhr_params already has init_pose added, so subtract it here
     * to recover the raw projection output and compare directly. */
    const float *ip = (const float *)m->init_pose.data;
    const float *ic = (const float *)m->init_camera.data;
    float pose_raw[519], cam_raw[3];
    for (int i = 0; i < 519; i++) pose_raw[i] = r.mhr_params[i] - ip[i];
    for (int i = 0; i < 3;   i++) cam_raw[i]  = r.cam_t[i]      - ic[i];
    int rc_total = 0;
    rc_total |= diff_pair("head_pose_proj_raw (519,)", pose_raw, ref_pose, 519, threshold);
    rc_total |= diff_pair("head_camera_proj_raw (3,)", cam_raw,  ref_cam,  3,   threshold);

    free(ref_pose); free(ref_cam);
    sam3d_body_decoder_free(m);
    return rc_total;
}
