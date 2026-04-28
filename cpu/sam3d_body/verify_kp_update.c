/*
 * verify_kp_update — diff sam3d_body_kp_token_update against the per-layer
 * pre/post-kp_update token + augment dumps for layers 0..n_layers-2.
 *
 * Inputs per layer i:
 *   decoder_kp_layer{i}__tokens_pre_kp.npy        (1, 145, 1024)
 *   decoder_kp_layer{i}__augment_pre_kp.npy       (1, 145, 1024)
 *   decoder_pose_layer{i}__pred_keypoints_2d_cropped.npy   (1, 70, 2)
 *   decoder_pose_layer{i}__pred_keypoints_2d_depth.npy     (1, 70)
 *   decoder_pose_layer{i}__pred_keypoints_3d.npy           (1, 70, 3)
 *   image_embeddings_after_ray.npy                          (1, 1280, 32, 32)
 *
 * Refs per layer i:
 *   decoder_kp_layer{i}__tokens_post_kp.npy
 *   decoder_kp_layer{i}__augment_post_kp.npy
 *
 * Usage:
 *   verify_kp_update --safetensors-dir <dir> --refdir <dir>
 *                    [--threshold F] [--layer N] [-t N]
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
        fprintf(stderr, "[verify_kp_update] missing %s\n", path);
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
    fprintf(stderr, "[verify_kp_update] %-40s max_abs=%.4e (i=%zu) "
                    "mean_abs=%.4e (max=%.1e) %s\n",
            label, mx, mxi, mean, thresh, fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

int main(int argc, char **argv)
{
    const char *sft_dir = NULL, *refdir = NULL;
    float threshold = 5e-4f;
    int one_layer = -1, n_threads = 1;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--safetensors-dir") && i+1 < argc) sft_dir = argv[++i];
        else if (!strcmp(argv[i], "--refdir")          && i+1 < argc) refdir  = argv[++i];
        else if (!strcmp(argv[i], "--threshold")       && i+1 < argc) threshold = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "--layer")           && i+1 < argc) one_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i+1 < argc) n_threads = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }
    if (!sft_dir || !refdir) {
        fprintf(stderr, "Usage: %s --safetensors-dir <dir> --refdir <dir> "
                "[--threshold F] [--layer N] [-t N]\n", argv[0]);
        return 2;
    }

    char p[1024];
    snprintf(p, sizeof(p), "%s/sam3d_body_decoder.safetensors", sft_dir);
    char p2[1024];
    snprintf(p2, sizeof(p2), "%s/sam3d_body_mhr_head.safetensors", sft_dir);
    sam3d_body_decoder_model *m = sam3d_body_decoder_load(p, p2);
    if (!m) { fprintf(stderr, "load failed\n"); return 3; }

    /* Image embeddings (post ray_cond) — same input across layers. */
    float *img_emb = load_or_die(refdir, "image_embeddings_after_ray");
    if (!img_emb) { sam3d_body_decoder_free(m); return 4; }
    /* Layout (1, 1280, 32, 32) -- channel-first matches kp_update API. */

    const int N_Q = 145, D = m->dim, K = m->n_keypoints;
    const int H = 32, W = 32;
    int rc_total = 0;
    /* Last layer (n_layers-1) is short-circuited per upstream guard. */
    int lo = (one_layer >= 0) ? one_layer : 0;
    int hi = (one_layer >= 0) ? one_layer + 1 : (m->n_layers - 1);

    float *tokens   = (float *)malloc((size_t)N_Q * D * sizeof(float));
    float *augment  = (float *)malloc((size_t)N_Q * D * sizeof(float));
    if (!tokens || !augment) {
        free(tokens); free(augment); free(img_emb);
        sam3d_body_decoder_free(m);
        return 5;
    }

    for (int li = lo; li < hi; li++) {
        char nm[128];

        snprintf(nm, sizeof(nm), "decoder_kp_layer%d__tokens_pre_kp", li);
        float *tok_in = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_kp_layer%d__augment_pre_kp", li);
        float *aug_in = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_kp_layer%d__tokens_post_kp", li);
        float *tok_ref = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_kp_layer%d__augment_post_kp", li);
        float *aug_ref = load_or_die(refdir, nm);

        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_cropped", li);
        float *kp2d = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_2d_depth", li);
        float *dep  = load_or_die(refdir, nm);
        snprintf(nm, sizeof(nm), "decoder_pose_layer%d__pred_keypoints_3d", li);
        float *kp3d = load_or_die(refdir, nm);

        if (!tok_in || !aug_in || !tok_ref || !aug_ref || !kp2d || !dep || !kp3d) {
            fprintf(stderr, "[verify_kp_update] layer %d: missing input(s)\n", li);
            free(tok_in); free(aug_in); free(tok_ref); free(aug_ref);
            free(kp2d); free(dep); free(kp3d);
            rc_total = 6; break;
        }

        memcpy(tokens,  tok_in, (size_t)N_Q * D * sizeof(float));
        memcpy(augment, aug_in, (size_t)N_Q * D * sizeof(float));

        int rc = sam3d_body_kp_token_update(
            m, li, img_emb, H, W, kp2d, dep, kp3d,
            N_Q, n_threads, tokens, augment);
        if (rc != SAM3D_BODY_DECODER_E_OK) {
            fprintf(stderr, "[verify_kp_update] layer %d: rc=%d\n", li, rc);
            free(tok_in); free(aug_in); free(tok_ref); free(aug_ref);
            free(kp2d); free(dep); free(kp3d);
            rc_total = 7; break;
        }

        char lbl1[64], lbl2[64];
        snprintf(lbl1, sizeof(lbl1), "layer%d tokens_post_kp",  li);
        snprintf(lbl2, sizeof(lbl2), "layer%d augment_post_kp", li);
        rc_total |= diff_pair(lbl1, tokens,  tok_ref, (size_t)N_Q * D, threshold);
        rc_total |= diff_pair(lbl2, augment, aug_ref, (size_t)N_Q * D, threshold);
        (void)K;

        free(tok_in); free(aug_in); free(tok_ref); free(aug_ref);
        free(kp2d); free(dep); free(kp3d);
    }

    free(tokens); free(augment); free(img_emb);
    sam3d_body_decoder_free(m);
    return rc_total;
}
